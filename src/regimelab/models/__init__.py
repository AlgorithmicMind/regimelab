"""Latent-state regime classifiers and their shared schemas."""

from __future__ import annotations

from regimelab.models.base import (
    CausalStandardiser,
    FittedRegime,
    InsufficientDataError,
    RegimeModel,
    StateEstimate,
)
from regimelab.models.filters import FilterResult, forward_filter, gaussian_log_density
from regimelab.models.gmm import GaussianMixtureRegimeModel
from regimelab.models.hamilton import MarkovSwitchingRegimeModel
from regimelab.models.hmm import HiddenMarkovRegimeModel
from regimelab.models.labeling import (
    LabelAssignment,
    OccupancyError,
    RankingMethod,
    StateStatistics,
    assign_labels,
    relabel_probabilities,
    summarise_states,
)
from regimelab.models.registry import CLASSIFIERS, build_classifier
from regimelab.models.types import (
    REGIME_LADDER,
    AllocationWeights,
    FeatureSnapshot,
    MarketRegime,
    ModelMetadata,
    RegimeProbabilities,
    TelemetryPayload,
    regime_ladder,
)

__all__ = [
    "CLASSIFIERS",
    "REGIME_LADDER",
    "AllocationWeights",
    "CausalStandardiser",
    "FeatureSnapshot",
    "FilterResult",
    "FittedRegime",
    "GaussianMixtureRegimeModel",
    "HiddenMarkovRegimeModel",
    "InsufficientDataError",
    "LabelAssignment",
    "MarketRegime",
    "MarkovSwitchingRegimeModel",
    "ModelMetadata",
    "OccupancyError",
    "RankingMethod",
    "RegimeModel",
    "RegimeProbabilities",
    "StateEstimate",
    "StateStatistics",
    "TelemetryPayload",
    "assign_labels",
    "build_classifier",
    "forward_filter",
    "gaussian_log_density",
    "regime_ladder",
    "relabel_probabilities",
    "summarise_states",
]
