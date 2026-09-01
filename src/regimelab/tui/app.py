"""Textual dashboard for the latest telemetry payload.

The dashboard is read-only: it renders the current pipeline result and does not
perform any computation. That keeps the TUI simple and prevents it from
introducing look-ahead by re-running the model interactively.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from regimelab.pipeline import latest_telemetry

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from regimelab.models.types import TelemetryPayload
    from regimelab.pipeline import PipelineResult


class RegimeLabApp:
    """Simple Textual dashboard for a pipeline result.

    The constructor is parameter-compatible with a real
    :class:`textual.app.App`, but the class is only usable when ``textual`` is
    installed.
    """

    def __init__(self, result: PipelineResult) -> None:
        self._result = result

    def run(self) -> None:
        """Launch the dashboard."""

        from textual.app import App  # noqa: PLC0415
        from textual.containers import Container, Horizontal, Vertical  # noqa: PLC0415
        from textual.widgets import DataTable, Footer, Header, Label, ProgressBar  # noqa: PLC0415

        payload = self._result_to_payload()
        backtest = self._result.backtest

        class _RegimeDashboard(App[None]):
            """Internal app class."""

            CSS = """
            Screen { align: center middle; }
            #main { padding: 1 2; width: 100%; height: 100%; }
            #header { height: auto; }
            #metrics { width: 30%; height: 100%; border: solid $primary; }
            #payload { width: 70%; height: 100%; border: solid $primary; }
            .big { text-style: bold; }
            .muted { color: $text-muted; }
            """

            def compose(self) -> ComposeResult:
                yield Header(show_clock=True)
                with Container(id="main"), Horizontal():
                    with Vertical(id="metrics"):
                        yield Label("regimelab", classes="big")
                        yield Label(f"as-of {payload.asof}", classes="muted")
                        yield Label("")
                        yield Label("Regime", classes="big")
                        yield Label(payload.regime.value)
                        yield Label("")
                        if backtest is not None:
                            yield Label("Strategy CAGR", classes="big")
                            yield Label(f"{backtest.metrics.cagr:.2%}")
                            yield Label("Benchmark CAGR", classes="big")
                            yield Label(f"{backtest.benchmark_metrics.cagr:.2%}")
                            yield Label("Max drawdown", classes="big")
                            yield Label(f"{backtest.metrics.max_drawdown:.2%}")
                        else:
                            yield Label("Backtest unavailable")

                    with Vertical(id="payload"):
                        yield Label("Feature snapshot", classes="big")
                        table: Any = DataTable(zebra_stripes=True)
                        table.add_columns("Feature", "Value")
                        for name in payload.features.model_fields:
                            value = getattr(payload.features, name)
                            rendered = "—" if value is None else f"{value:.4f}"
                            table.add_row(name, rendered)
                        yield table

                        yield Label("")
                        yield Label("Regime probabilities", classes="big")
                        for regime, prob in payload.probabilities.probabilities.items():
                            bar = ProgressBar(total=100, show_percentage=False)
                            bar.update(progress=int(prob * 100))
                            yield Label(f"{regime.value}  {prob:.1%}")
                            yield bar

                        if payload.target_weights is not None:
                            yield Label("")
                            yield Label("Target weights", classes="big")
                            weights = payload.target_weights.normalise()
                            for symbol, weight in weights.items():
                                yield Label(f"{symbol}: {weight:.2%}")
                yield Footer()

        return _RegimeDashboard().run()

    def _result_to_payload(self) -> TelemetryPayload:
        """Build the latest telemetry payload from the result."""
        return latest_telemetry(self._result)
