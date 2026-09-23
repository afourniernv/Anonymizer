# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from pydantic import ValidationError

from anonymizer import (
    DEFAULT_ENTITY_LABELS,
    OnlineDetectConfig,
    OnlineDetectionError,
    OnlineDetectionResponseError,
    OnlineDetectionResult,
    OnlineDetectionTimeoutError,
    OnlineDetector,
    TextRecord,
)
from anonymizer.interface.errors import InvalidInputError


def _completion(entities: object, *, finish_reason: str = "stop") -> dict[str, object]:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"entities": entities}),
                },
                "finish_reason": finish_reason,
            }
        ]
    }


def _entity(*, label: str = "first_name", start: object = 0, end: object = 5, score: object = 0.9) -> dict[str, object]:
    return {"text": "normalized-echo", "label": label, "start": start, "end": end, "score": score}


def test_online_detect_config_normalizes_labels_and_applies_exclusions() -> None:
    config = OnlineDetectConfig(
        entity_labels=[" Email ", "first_name", "email"],
        excluded_entity_labels=["EMAIL"],
        gliner_threshold=0.4,
    )

    assert config.entity_labels == ["email", "first_name"]
    assert config.excluded_entity_labels == ["email"]
    assert config.effective_entity_labels == ("first_name",)
    assert config.gliner_threshold == 0.4


def test_online_detect_config_uses_defaults_and_rejects_an_empty_policy() -> None:
    assert OnlineDetectConfig().effective_entity_labels == tuple(DEFAULT_ENTITY_LABELS)

    with pytest.raises(ValidationError, match="leaves no labels"):
        OnlineDetectConfig(excluded_entity_labels=list(DEFAULT_ENTITY_LABELS))


def test_detect_preserves_order_and_sends_the_expected_request() -> None:
    requests: list[dict[str, object]] = []
    authorizations: list[str | None] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        authorizations.append(request.headers.get("authorization"))
        text = body["messages"][0]["content"]
        if text.startswith("Alice"):
            await asyncio.sleep(0.01)
            entities = [_entity()]
        else:
            entities = [_entity(label="last_name", start=0, end=3, score=0.8)]
        return httpx.Response(200, json=_completion(entities))

    async def run() -> tuple[OnlineDetectionResult, ...]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            detector = OnlineDetector(
                "https://detector.example/v1/",
                "privacy-model",
                api_key="test-token",
                client=client,
            )
            results = await detector.detect(
                [TextRecord(id="a", text="Alice joined"), TextRecord(id="b", text="Bob joined")],
                config=OnlineDetectConfig(entity_labels=["first_name", "last_name"], gliner_threshold=0.4),
            )
            await detector.close()
            assert not client.is_closed
            return results

    results = asyncio.run(run())

    assert [result.record_id for result in results] == ["a", "b"]
    assert results[0].spans[0].start_position == 0
    assert results[0].spans[0].end_position == 5
    assert results[1].spans[0].label == "last_name"
    assert {request["model"] for request in requests} == {"privacy-model"}
    assert {request["threshold"] for request in requests} == {0.4}
    assert all(request["labels"] == ["first_name", "last_name"] for request in requests)
    assert all(request["chunk_length"] == 384 for request in requests)
    assert all(request["overlap"] == 128 for request in requests)
    assert all(request["flat_ner"] is False for request in requests)
    assert authorizations == ["Bearer test-token", "Bearer test-token"]


def test_detect_skips_empty_text_and_rejects_duplicate_ids() -> None:
    calls = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=_completion([]))

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            detector = OnlineDetector("https://detector.example/v1", "privacy-model", client=client)
            result = await detector.detect([TextRecord(id="empty", text="")])
            assert result[0].spans == ()
            with pytest.raises(InvalidInputError, match="IDs must be unique"):
                await detector.detect([TextRecord(id="same", text="a"), TextRecord(id="same", text="b")])

    asyncio.run(run())
    assert calls == 0


@pytest.mark.parametrize(
    "response_factory",
    [
        lambda: httpx.Response(200, json={"choices": []}),
        lambda: httpx.Response(200, json=_completion([_entity(end=100)])),
        lambda: httpx.Response(200, json=_completion([_entity(start="0")])),
        lambda: httpx.Response(200, json=_completion([_entity(score=float("nan"))])),
        lambda: httpx.Response(200, json=_completion([_entity(score=10**400)])),
        lambda: httpx.Response(200, json=_completion([{"start": 0, "end": 5, "score": 0.9}])),
        lambda: httpx.Response(200, json=_completion([_entity(label="not_requested")])),
        lambda: httpx.Response(200, json=_completion([], finish_reason="content_filter")),
        lambda: httpx.Response(200, content=b"not-json"),
        lambda: httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "not-json"}, "finish_reason": "stop"}]},
        ),
    ],
    ids=[
        "missing-choice",
        "out-of-bounds-offset",
        "non-integer-offset",
        "non-finite-score",
        "overflowing-score",
        "missing-label",
        "out-of-policy-label",
        "incomplete-finish-reason",
        "invalid-outer-json",
        "invalid-content-json",
    ],
)
def test_detect_rejects_malformed_or_unsafe_responses(
    response_factory: Callable[[], httpx.Response],
) -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return response_factory()

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            detector = OnlineDetector("https://detector.example/v1", "privacy-model", client=client)
            with pytest.raises(OnlineDetectionResponseError):
                await detector.detect([TextRecord(id="record", text="Alice")])

    asyncio.run(run())


def test_detect_maps_timeout_http_and_transport_failures_without_response_text() -> None:
    canary = "marisol.vega@example.com"

    async def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout(canary, request=request)

    async def http_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text=canary)

    async def transport_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(canary, request=request)

    async def invoke(
        handler: Callable[[httpx.Request], Coroutine[Any, Any, httpx.Response]],
    ) -> OnlineDetectionError:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            detector = OnlineDetector("https://detector.example/v1", "privacy-model", client=client)
            try:
                await detector.detect([TextRecord(id="record", text=canary)])
            except OnlineDetectionError as error:
                return error
        raise AssertionError("detection unexpectedly succeeded")

    async def run() -> None:
        with pytest.raises(OnlineDetectionTimeoutError) as timeout_error:
            async with httpx.AsyncClient(transport=httpx.MockTransport(timeout_handler)) as client:
                detector = OnlineDetector("https://detector.example/v1", "privacy-model", client=client)
                await detector.detect([TextRecord(id="record", text=canary)])
        assert canary not in str(timeout_error.value)
        errors = [timeout_error.value, await invoke(http_handler), await invoke(transport_handler)]
        for error in errors:
            assert canary not in str(error)
            assert error.__cause__ is None
            assert error.__context__ is None

    asyncio.run(run())


def test_invalid_completion_error_does_not_retain_response_content() -> None:
    canary = "marisol.vega@example.com"

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=canary)

    async def run() -> OnlineDetectionResponseError:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            detector = OnlineDetector("https://detector.example/v1", "privacy-model", client=client)
            try:
                await detector.detect([TextRecord(id="record", text=canary)])
            except OnlineDetectionResponseError as error:
                return error
        raise AssertionError("detection unexpectedly succeeded")

    error = asyncio.run(run())
    assert canary not in str(error)
    assert error.__cause__ is None
    assert error.__context__ is None


def test_detect_uses_source_offsets_and_resolves_overlaps() -> None:
    source = "👋 Alice Smith"
    entities = [
        _entity(label="first_name", start=2, end=7, score=0.99),
        _entity(label="full_name", start=2, end=13, score=0.8),
    ]

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion(entities))

    async def run() -> tuple[OnlineDetectionResult, ...]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            detector = OnlineDetector("https://detector.example/v1", "privacy-model", client=client)
            return await detector.detect(
                [TextRecord(id="record", text=source)],
                config=OnlineDetectConfig(entity_labels=["first_name", "full_name"]),
            )

    result = asyncio.run(run())[0]
    assert result.spans[0].label == "full_name"
    assert (result.spans[0].start_position, result.spans[0].end_position) == (2, 13)
    assert source[result.spans[0].start_position : result.spans[0].end_position] == "Alice Smith"


def test_detect_propagates_external_cancellation_to_requests() -> None:
    request_started = asyncio.Event()
    request_cancelled = asyncio.Event()

    async def handler(_request: httpx.Request) -> httpx.Response:
        request_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            request_cancelled.set()
            raise
        raise AssertionError("unreachable")

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            detector = OnlineDetector("https://detector.example/v1", "privacy-model", client=client)
            detection = asyncio.create_task(detector.detect([TextRecord(id="record", text="Alice")]))
            await request_started.wait()
            detection.cancel()
            with pytest.raises(asyncio.CancelledError):
                await detection
            assert request_cancelled.is_set()

    asyncio.run(run())


def test_constructor_rejects_invalid_connection_and_limit_values() -> None:
    for endpoint in ["relative/v1", "https://example.com:bad/v1", "https://exa mple.com/v1"]:
        with pytest.raises(ValueError, match="absolute HTTP"):
            OnlineDetector(endpoint, "privacy-model")
    with pytest.raises(ValueError, match="credentials"):
        OnlineDetector("https://user:secret@example.com/v1", "privacy-model")
    with pytest.raises(ValueError, match="model"):
        OnlineDetector("https://detector.example/v1", " ")
    for timeout in [0.0, -1.0, float("nan"), float("inf"), True]:
        with pytest.raises(ValueError, match="timeout_seconds"):
            OnlineDetector("https://detector.example/v1", "privacy-model", timeout_seconds=timeout)
    for concurrency in [0, -1, True, 1.5]:
        with pytest.raises(ValueError, match="max_concurrency"):
            OnlineDetector(
                "https://detector.example/v1",
                "privacy-model",
                max_concurrency=cast(Any, concurrency),
            )


def test_detect_cancels_sibling_requests_after_one_fails() -> None:
    slow_started = asyncio.Event()
    slow_cancelled = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        text = body["messages"][0]["content"]
        if text == "fail":
            await slow_started.wait()
            return httpx.Response(500, text="failure")
        slow_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            slow_cancelled.set()
            raise
        raise AssertionError("unreachable")

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            detector = OnlineDetector(
                "https://detector.example/v1",
                "privacy-model",
                max_concurrency=2,
                client=client,
            )
            with pytest.raises(OnlineDetectionError, match="HTTP 500"):
                await detector.detect([TextRecord(id="failure", text="fail"), TextRecord(id="slow", text="slow")])
            assert slow_cancelled.is_set()

    asyncio.run(run())


def test_detect_shares_the_concurrency_limit_across_calls() -> None:
    active = 0
    peak = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return httpx.Response(200, json=_completion([]))

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            detector = OnlineDetector(
                "https://detector.example/v1",
                "privacy-model",
                max_concurrency=2,
                client=client,
            )
            await asyncio.gather(
                detector.detect([TextRecord(id=f"left-{index}", text="left") for index in range(3)]),
                detector.detect([TextRecord(id=f"right-{index}", text="right") for index in range(3)]),
            )

    asyncio.run(run())
    assert peak == 2


def test_close_is_idempotent_and_does_not_close_an_injected_client() -> None:
    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200))) as client:
            detector = OnlineDetector("https://detector.example/v1", "privacy-model", client=client)
            await detector.close()
            await detector.close()
            assert not client.is_closed
            with pytest.raises(OnlineDetectionError, match="closed"):
                await detector.detect([])

    asyncio.run(run())


def test_online_public_import_does_not_load_data_designer() -> None:
    source_root = Path(__file__).parents[2] / "src"
    code = """
import sys
from anonymizer import OnlineDetectConfig, OnlineDetector
assert OnlineDetectConfig is not None
assert OnlineDetector is not None
assert not any(name == 'data_designer' or name.startswith('data_designer.') for name in sys.modules)
"""

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source_root)
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        cwd=source_root.parent,
        env=environment,
    )
