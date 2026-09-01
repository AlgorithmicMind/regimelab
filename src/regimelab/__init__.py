"""regimelab: institutional market-regime detection and systemic-risk telemetry.

The public surface is intentionally small. Everything reachable from here is
considered stable API; submodules are free to reorganise.
"""

from __future__ import annotations

from regimelab.models.types import (
    MarketRegime,
    RegimeProbabilities,
    TelemetryPayload,
)
from regimelab.settings import Settings

__version__ = "0.1.0"

__all__ = [
    "MarketRegime",
    "RegimeProbabilities",
    "Settings",
    "TelemetryPayload",
    "__version__",
]
