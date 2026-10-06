"""The signal universe is decoupled from the investable universe.

``risk_assets`` drives only the backtest and allocation; ``correlation_assets``
drives the cross-asset correlation features. All tests run on the synthetic
provider -- offline, and per-symbol deterministic, so fetching different symbol
sets never changes a symbol's series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from regimelab.data.alignment import align_panel
from regimelab.data.fetcher import PanelFetcher
from regimelab.data.providers import resolve_provider
from regimelab.features import build_feature_matrix
from regimelab.models.types import MarketRegime
from regimelab.settings import BacktestSettings, DataSettings, Settings


def _features_for(settings: Settings) -> pd.DataFrame:
    """Fetch, align and featurise exactly as the pipeline does."""
    provider = resolve_provider(settings)
    fetcher = PanelFetcher(provider, settings.artifacts_dir / "cache", ttl_hours=0)
    raw = fetcher.fetch(
        settings.data.all_symbols,
        settings.data.start,
        settings.data.end,
    )
    aligned = align_panel(raw, anchor=settings.data.benchmark)
    return build_feature_matrix(aligned, settings)


def _strip_symbol(
    weights: dict[MarketRegime, dict[str, float]],
    symbol: str,
) -> dict[MarketRegime, dict[str, float]]:
    """Drop ``symbol`` from every regime's target weights."""
    return {
        regime: {s: w for s, w in sleeve.items() if s != symbol}
        for regime, sleeve in weights.items()
    }


def test_investable_universe_does_not_change_features(test_settings: Settings) -> None:
    """Different ``risk_assets`` produce a bit-identical feature matrix."""
    # The coherence validator requires every regime_weights symbol to be in
    # risk_assets, so an investable universe without TLT needs target weights
    # without TLT too.
    narrowed = test_settings.model_copy(
        update={
            "data": test_settings.data.model_copy(update={"risk_assets": ("SPY", "GLD")}),
            "backtest": test_settings.backtest.model_copy(
                update={
                    "regime_weights": _strip_symbol(test_settings.backtest.regime_weights, "TLT")
                }
            ),
        }
    )

    features_default = _features_for(test_settings)
    features_narrowed = _features_for(narrowed)

    pd.testing.assert_frame_equal(features_default, features_narrowed)


def test_correlation_universe_changes_mean_correlation(
    test_settings: Settings,
) -> None:
    """``correlation_assets`` is what the correlation feature actually reads."""
    subset = test_settings.model_copy(
        update={
            "data": test_settings.data.model_copy(update={"correlation_assets": ("SPY", "GLD")})
        }
    )

    baseline = _features_for(test_settings)["mean_correlation"]
    narrowed = _features_for(subset)["mean_correlation"]

    both = baseline.notna() & narrowed.notna()
    assert both.any()
    assert not np.allclose(baseline[both], narrowed[both])


def test_correlation_assets_need_two_symbols() -> None:
    """A single-asset correlation universe is a configuration error."""
    with pytest.raises(ValidationError, match="correlation_assets"):
        DataSettings(correlation_assets=("SPY",))


def test_correlation_assets_join_all_symbols() -> None:
    """Signal-universe symbols are fetched even when not investable."""
    settings = Settings(
        data=DataSettings(risk_assets=("SPY", "GLD")),
        # regime_weights must remain a subset of risk_assets (coherence check).
        backtest=BacktestSettings(
            regime_weights=_strip_symbol(BacktestSettings().regime_weights, "TLT")
        ),
    )

    # TLT is not investable here, but it is in correlation_assets, so the
    # engine still fetches it for the signal.
    assert "TLT" not in settings.data.risk_assets
    assert "TLT" in settings.data.all_symbols
