"""The single pipeline data path (0.3.0).

``_fetch_and_align`` delegates to
:func:`~regimelab.data.fetcher.load_aligned_panel`, so the panel a pipeline
run sees is byte-identical to the shared loader's, cached under
``data.cache_dir`` within ``data.cache_ttl_hours``.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pytest

from regimelab.data import fetcher
from regimelab.data.fetcher import load_aligned_panel
from regimelab.data.providers import resolve_provider
from regimelab.pipeline import PipelineResult
from regimelab.settings import Settings


def test_pipeline_panel_equals_load_aligned_panel(
    test_settings: Settings,
    pipeline_result: PipelineResult,
) -> None:
    """``run_pipeline`` returns exactly the panel the shared loader produces."""
    expected = load_aligned_panel(test_settings)

    # A cache round-trip can drop the DatetimeIndex freq metadata on either
    # side; the aligned values are what matter.
    pd.testing.assert_frame_equal(pipeline_result.panel.close(), expected.close(), check_freq=False)
    pd.testing.assert_frame_equal(
        pipeline_result.panel.staleness, expected.staleness, check_freq=False
    )
    assert pipeline_result.panel.dropped_symbols == expected.dropped_symbols


class _CountingProvider:
    """Protocol-conforming wrapper that counts provider ``fetch`` calls."""

    def __init__(self, inner: object) -> None:
        self._inner = inner
        self.calls = 0

    @property
    def name(self) -> str:
        return self._inner.name  # type: ignore[attr-defined]

    @property
    def is_point_in_time(self) -> bool:
        return self._inner.is_point_in_time  # type: ignore[attr-defined]

    def fetch(
        self,
        symbols: Sequence[str],
        start: dt.date,
        end: dt.date | None = None,
    ) -> pd.DataFrame:
        self.calls += 1
        return self._inner.fetch(symbols, start, end)  # type: ignore[attr-defined]


def test_force_refresh_bypasses_the_panel_cache(
    test_settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``force_refresh=True`` refetches even with a fresh cache entry."""
    settings = test_settings.model_copy(
        update={
            "data": test_settings.data.model_copy(
                update={"cache_dir": tmp_path / "cache"},
            ),
        },
    )
    counting = _CountingProvider(resolve_provider(settings))
    monkeypatch.setattr(fetcher, "resolve_provider", lambda _settings: counting)

    first = load_aligned_panel(settings)
    # Within ``cache_ttl_hours`` the second call is served from the cache.
    cached = load_aligned_panel(settings)
    refreshed = load_aligned_panel(settings, force_refresh=True)

    assert counting.calls == 2
    # The cache round-trip drops the DatetimeIndex freq metadata; values are
    # what matter.
    pd.testing.assert_frame_equal(first.close(), cached.close(), check_freq=False)
    pd.testing.assert_frame_equal(first.close(), refreshed.close(), check_freq=False)
