# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from anonymizer.config.anonymizer_config import TextRecord
from anonymizer.config.online import OnlineDetectConfig
from anonymizer.engine.detection.postprocess import EntitySpan, resolve_overlaps
from anonymizer.interface.errors import (
    InvalidInputError,
    OnlineDetectionError,
    OnlineDetectionResponseError,
    OnlineDetectionTimeoutError,
)

_CHUNK_LENGTH = 384
_CHUNK_OVERLAP = 128


@dataclass(frozen=True)
class OnlineEntitySpan:
    """One entity anchored to an exact slice of a submitted text record.

    ``start_position`` is inclusive and ``end_position`` is exclusive. Raw
    entity text is intentionally omitted; callers can derive it from the
    submitted source text when needed.

    Attributes:
        label: Normalized entity label returned by the detector.
        start_position: Inclusive character offset into the source text.
        end_position: Exclusive character offset into the source text.
        score: Detector confidence score from 0.0 to 1.0.
    """

    label: str
    start_position: int
    end_position: int
    score: float


@dataclass(frozen=True)
class OnlineDetectionResult:
    """Detected entity spans correlated with one caller-defined record ID.

    Attributes:
        record_id: ID from the submitted ``TextRecord``.
        spans: Non-overlapping spans ordered by source position.
    """

    record_id: str
    spans: tuple[OnlineEntitySpan, ...]


class OnlineDetector:
    """Reusable asynchronous client for request-time GLiNER detection.

    Callers should normally use this class as an asynchronous context manager.
    A caller-supplied HTTP client remains owned by the caller and is not closed
    with the detector. Submitted source text is sent to the configured endpoint.

    Args:
        endpoint: Base URL of a compatible GLiNER service, normally ending in
            ``/v1``.
        model: Model identifier sent in each chat-completion request.
        api_key: Optional bearer token for the detector service.
        timeout_seconds: Per-request HTTP timeout.
        max_concurrency: Maximum concurrent requests across calls made through
            this detector instance.
        client: Optional reusable HTTP client owned by the caller.

    Raises:
        ValueError: If connection settings are invalid.
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 20.0,
        max_concurrency: int = 8,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._endpoint = _validate_endpoint(endpoint)
        self._model = _validate_model(model)
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a finite number greater than zero")
        if isinstance(max_concurrency, bool) or not isinstance(max_concurrency, int) or max_concurrency <= 0:
            raise ValueError("max_concurrency must be a positive integer")
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._timeout_seconds = float(timeout_seconds)
        self._max_concurrency = max_concurrency
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient()
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._closed = False

    async def __aenter__(self) -> OnlineDetector:
        return self

    async def __aexit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        await self.close()

    async def detect(
        self,
        records: Sequence[TextRecord],
        *,
        config: OnlineDetectConfig | None = None,
    ) -> tuple[OnlineDetectionResult, ...]:
        """Detect entities while preserving input IDs and order.

        Args:
            records: Ordered in-memory text records with unique caller IDs.
            config: Optional detection policy. Defaults to Anonymizer's standard
                label set and GLiNER threshold.

        Returns:
            Results in the same order as ``records``.

        Raises:
            InvalidInputError: If record IDs are not unique.
            OnlineDetectionTimeoutError: If any detector request times out.
            OnlineDetectionResponseError: If the detector response cannot be
                mapped safely to source spans.
            OnlineDetectionError: If the session is closed or another detector
                request fails. Any unfinished sibling requests are cancelled.
        """

        self._ensure_open()
        submitted_records = tuple(records)
        record_ids = [record.id for record in submitted_records]
        if len(record_ids) != len(set(record_ids)):
            raise InvalidInputError("Online detection record IDs must be unique")

        detect_config = config or OnlineDetectConfig()
        if not submitted_records:
            return ()

        work = iter(enumerate(submitted_records))
        results: dict[int, OnlineDetectionResult] = {}
        tasks = [
            asyncio.create_task(self._detect_records(work, results, detect_config))
            for _ in range(min(len(submitted_records), self._max_concurrency))
        ]
        try:
            await asyncio.gather(*tasks)
        except (Exception, asyncio.CancelledError):
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return tuple(results[index] for index in range(len(submitted_records)))

    async def close(self) -> None:
        """Close the internally owned HTTP client, if any.

        This method is idempotent and never closes a caller-supplied client.
        """

        if self._closed:
            return
        if self._owns_client:
            await self._client.aclose()
        self._closed = True

    async def _detect_records(
        self,
        work: Iterator[tuple[int, TextRecord]],
        results: dict[int, OnlineDetectionResult],
        config: OnlineDetectConfig,
    ) -> None:
        for index, record in work:
            results[index] = await self._detect_record(record, config)

    async def _detect_record(self, record: TextRecord, config: OnlineDetectConfig) -> OnlineDetectionResult:
        if not record.text:
            return OnlineDetectionResult(record_id=record.id, spans=())
        async with self._semaphore:
            response = await self._request(
                "POST",
                "chat/completions",
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": record.text}],
                    "labels": list(config.effective_entity_labels),
                    "threshold": config.gliner_threshold,
                    "chunk_length": _CHUNK_LENGTH,
                    "overlap": _CHUNK_OVERLAP,
                    "flat_ner": False,
                },
            )
        return OnlineDetectionResult(
            record_id=record.id,
            spans=_parse_entities(response, record.text, allowed_labels=config.effective_entity_labels),
        )

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        self._ensure_open()
        failure: OnlineDetectionError | None = None
        try:
            response = await self._client.request(
                method,
                f"{self._endpoint}/{path}",
                headers=self._headers,
                timeout=self._timeout_seconds,
                **kwargs,
            )
            response.raise_for_status()
        except httpx.TimeoutException:
            failure = OnlineDetectionTimeoutError("Online detector request timed out")
        except httpx.HTTPStatusError as error:
            failure = OnlineDetectionError(f"Online detector returned HTTP {error.response.status_code}")
        except (httpx.HTTPError, httpx.InvalidURL):
            failure = OnlineDetectionError("Online detector request failed")
        except RuntimeError:
            if not self._client.is_closed:
                raise
            failure = OnlineDetectionError("Online detector client is unavailable")
        if failure is not None:
            raise failure
        return response

    def _ensure_open(self) -> None:
        if self._closed or self._client.is_closed:
            raise OnlineDetectionError("Online detector is closed")


def _validate_endpoint(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("endpoint must be an absolute HTTP(S) URL")
    endpoint = value.strip().rstrip("/")
    if any(character.isspace() for character in endpoint):
        raise ValueError("endpoint must be an absolute HTTP(S) URL")
    try:
        parsed = httpx.URL(endpoint)
    except httpx.InvalidURL:
        raise ValueError("endpoint must be an absolute HTTP(S) URL") from None
    if parsed.scheme not in {"http", "https"} or not parsed.host:
        raise ValueError("endpoint must be an absolute HTTP(S) URL")
    if parsed.userinfo:
        raise ValueError("endpoint must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("endpoint must not contain a query string or fragment")
    return endpoint


def _validate_model(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("model must not be empty")
    model = value.strip()
    if not model:
        raise ValueError("model must not be empty")
    return model


def _response_json(response: httpx.Response) -> Any:
    try:
        payload = response.json()
    except (ValueError, UnicodeDecodeError):
        pass
    else:
        return payload
    raise OnlineDetectionResponseError("Online detector returned invalid JSON")


def _parse_entities(
    response: httpx.Response,
    source: str,
    *,
    allowed_labels: Sequence[str],
) -> tuple[OnlineEntitySpan, ...]:
    payload = _response_json(response)
    if not isinstance(payload, dict):
        raise OnlineDetectionResponseError("Online detector returned an invalid completion")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise OnlineDetectionResponseError("Online detector returned an invalid completion")
    choice = choices[0]
    if choice.get("finish_reason") != "stop":
        raise OnlineDetectionResponseError("Online detector returned an incomplete completion")
    message = choice.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise OnlineDetectionResponseError("Online detector returned an invalid completion")
    try:
        detector_payload = json.loads(message["content"])
    except (TypeError, json.JSONDecodeError):
        detector_payload = None
    if not isinstance(detector_payload, dict):
        raise OnlineDetectionResponseError("Online detector returned an invalid completion")
    raw_entities = detector_payload.get("entities")
    if not isinstance(raw_entities, list):
        raise OnlineDetectionResponseError("Online detector returned an invalid entity collection")

    allowed = set(allowed_labels)
    parsed = [
        _parse_entity(raw_entity, source, index, allowed_labels=allowed)
        for index, raw_entity in enumerate(raw_entities)
    ]
    resolved = resolve_overlaps(parsed, prefer_highest_score=True)
    return tuple(
        OnlineEntitySpan(
            label=entity.label,
            start_position=entity.start_position,
            end_position=entity.end_position,
            score=entity.score,
        )
        for entity in resolved
    )


def _parse_entity(raw_entity: Any, source: str, index: int, *, allowed_labels: set[str]) -> EntitySpan:
    if not isinstance(raw_entity, dict):
        raise OnlineDetectionResponseError("Online detector returned a non-object entity")
    label = raw_entity.get("label")
    start = raw_entity.get("start")
    end = raw_entity.get("end")
    score = raw_entity.get("score")
    if not isinstance(label, str) or not label.strip():
        raise OnlineDetectionResponseError("Online detector returned an entity without a label")
    if isinstance(start, bool) or not isinstance(start, int) or isinstance(end, bool) or not isinstance(end, int):
        raise OnlineDetectionResponseError("Online detector returned non-integer entity offsets")
    if start < 0 or end <= start or end > len(source):
        raise OnlineDetectionResponseError("Online detector returned out-of-bounds entity offsets")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise OnlineDetectionResponseError("Online detector returned an invalid entity score")
    try:
        normalized_score = float(score)
    except (OverflowError, ValueError):
        normalized_score = None
    if normalized_score is None or not math.isfinite(normalized_score):
        raise OnlineDetectionResponseError("Online detector returned an invalid entity score")
    if not 0.0 <= normalized_score <= 1.0:
        raise OnlineDetectionResponseError("Online detector returned an entity score outside 0..1")
    normalized_label = label.strip().casefold()
    if normalized_label not in allowed_labels:
        raise OnlineDetectionResponseError("Online detector returned an entity outside the requested label set")
    return EntitySpan(
        entity_id=f"{normalized_label}_{start}_{end}_{index}",
        value=source[start:end],
        label=normalized_label,
        start_position=start,
        end_position=end,
        score=normalized_score,
        source="detector",
    )
