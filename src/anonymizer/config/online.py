# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

from anonymizer.engine.constants import DEFAULT_ENTITY_LABELS


class OnlineDetectConfig(BaseModel):
    """Entity policy for one online GLiNER detection call.

    Attributes:
        entity_labels: Labels sent to GLiNER. ``None`` uses Anonymizer's
            built-in default label set.
        excluded_entity_labels: Labels removed from the effective label set.
        gliner_threshold: Minimum GLiNER confidence score from 0.0 to 1.0.
    """

    entity_labels: list[str] | None = Field(
        default=None,
        description="Labels to detect. None uses the built-in default label set.",
    )
    excluded_entity_labels: list[str] | None = Field(
        default=None,
        description="Labels removed from the effective detector allowlist.",
    )
    gliner_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="GLiNER detection confidence threshold.",
    )

    @field_validator("entity_labels", "excluded_entity_labels")
    @classmethod
    def normalize_entity_labels(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        labels = sorted({label.strip().casefold() for label in value if label.strip()})
        if not labels:
            raise ValueError("entity label lists must not be empty")
        return labels

    @model_validator(mode="after")
    def validate_effective_entity_labels(self) -> OnlineDetectConfig:
        if not self.effective_entity_labels:
            raise ValueError("excluded_entity_labels leaves no labels to detect")
        return self

    @property
    def effective_entity_labels(self) -> tuple[str, ...]:
        """Return the normalized detector allowlist after exclusions."""

        included = self.entity_labels if self.entity_labels is not None else DEFAULT_ENTITY_LABELS
        excluded = set(self.excluded_entity_labels or ())
        return tuple(label for label in included if label not in excluded)
