# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from importlib.metadata import version
from typing import TYPE_CHECKING

__version__ = version("nemo-anonymizer")

from anonymizer.config.anonymizer_config import (
    AnonymizerConfig,
    AnonymizerInput,
    Detect,
    EvaluateConfig,
    Rewrite,
    RiskTolerance,
    TextRecord,
    TextRecordsInput,
)
from anonymizer.config.replace_strategies import Annotate, Hash, Redact, Substitute
from anonymizer.config.rewrite import PrivacyGoal
from anonymizer.engine.constants import DEFAULT_ENTITY_LABELS as _DEFAULT_ENTITY_LABELS
from anonymizer.interface.errors import (
    AnonymizerError,
    AnonymizerIOError,
    InvalidConfigError,
    InvalidInputError,
    OnlineDetectionError,
    OnlineDetectionResponseError,
    OnlineDetectionTimeoutError,
)
from anonymizer.logging import LoggingConfig, configure_logging

if TYPE_CHECKING:
    from data_designer.config.models import ModelProvider as ModelProvider
    from data_designer.config.run_config import RunConfig as RunConfig

    from anonymizer.config.online import OnlineDetectConfig as OnlineDetectConfig
    from anonymizer.interface.anonymizer import Anonymizer as Anonymizer
    from anonymizer.interface.online import (
        OnlineDetectionResult as OnlineDetectionResult,
    )
    from anonymizer.interface.online import (
        OnlineDetector as OnlineDetector,
    )
    from anonymizer.interface.online import (
        OnlineEntitySpan as OnlineEntitySpan,
    )

# Export as an immutable public constant so callers can inspect defaults
# without mutating the internal source-of-truth list.
DEFAULT_ENTITY_LABELS: tuple[str, ...] = tuple(_DEFAULT_ENTITY_LABELS)


def __getattr__(name: str) -> object:
    if name == "Anonymizer":
        from anonymizer.interface.anonymizer import Anonymizer

        return Anonymizer
    if name in {"ModelProvider", "RunConfig"}:
        from data_designer.config.models import ModelProvider
        from data_designer.config.run_config import RunConfig

        return {"ModelProvider": ModelProvider, "RunConfig": RunConfig}[name]
    if name == "OnlineDetectConfig":
        from anonymizer.config.online import OnlineDetectConfig

        return OnlineDetectConfig
    if name in {"OnlineDetectionResult", "OnlineDetector", "OnlineEntitySpan"}:
        from anonymizer.interface.online import OnlineDetectionResult, OnlineDetector, OnlineEntitySpan

        return {
            "OnlineDetectionResult": OnlineDetectionResult,
            "OnlineDetector": OnlineDetector,
            "OnlineEntitySpan": OnlineEntitySpan,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Anonymizer",
    "AnonymizerConfig",
    "AnonymizerError",
    "AnonymizerInput",
    "AnonymizerIOError",
    "Annotate",
    "DEFAULT_ENTITY_LABELS",
    "Detect",
    "EvaluateConfig",
    "Hash",
    "InvalidConfigError",
    "InvalidInputError",
    "LoggingConfig",
    "ModelProvider",
    "OnlineDetectConfig",
    "OnlineDetectionError",
    "OnlineDetectionResult",
    "OnlineDetectionResponseError",
    "OnlineDetectionTimeoutError",
    "OnlineDetector",
    "OnlineEntitySpan",
    "PrivacyGoal",
    "Redact",
    "Rewrite",
    "RiskTolerance",
    "RunConfig",
    "Substitute",
    "TextRecord",
    "TextRecordsInput",
    "configure_logging",
]
