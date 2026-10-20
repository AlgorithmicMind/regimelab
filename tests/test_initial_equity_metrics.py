"""Initial-capital peaks are opt-in; legacy metrics remain unchanged."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regimelab.backtest.metrics import compute_metrics, drawdown_series


def test_initial_peak_is_explicit() -> None:
    equity = pd.Series([0.9, 0.8, 0.85])
    legacy = drawdown_series(equity)
    observed = drawdown_series(equity, initial=1.0)
    np.testing.assert_allclose(legacy, [0.0, 0.8 / 0.9 - 1, 0.85 / 0.9 - 1])
    np.testing.assert_allclose(observed, [-0.1, -0.2, -0.15])


def test_metrics_include_initial_loss_only_when_requested() -> None:
    returns = pd.Series([-0.1, 0.0], index=pd.date_range("2024-01-02", periods=2))
    legacy = compute_metrics(returns)
    observed = compute_metrics(returns, initial_equity=1.0)
    assert legacy.max_drawdown == 0.0
    assert observed.max_drawdown == pytest.approx(-0.1)
    assert observed.calmar == pytest.approx(observed.cagr / 0.1)
    assert observed.total_return == legacy.total_return
    assert observed.sharpe == legacy.sharpe


@pytest.mark.parametrize("initial", [0.0, -1.0, np.nan, np.inf])
def test_invalid_initial_peak_is_rejected(initial: float) -> None:
    with pytest.raises(ValueError, match="initial equity"):
        drawdown_series(pd.Series([1.0]), initial=initial)
    with pytest.raises(ValueError, match="initial equity"):
        compute_metrics(pd.Series(dtype="float64"), initial_equity=initial)
