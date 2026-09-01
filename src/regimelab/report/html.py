"""Render a self-contained HTML report from a :class:`~regimelab.pipeline.PipelineResult`.

The report is intentionally conservative: the narrative is driven by the numbers
produced by the pipeline, not by decorative prose. Every number in the report
can be traced back to a pandas frame or Pydantic model in the code.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go

from regimelab.pipeline import latest_telemetry

if TYPE_CHECKING:
    from regimelab.pipeline import PipelineResult

__all__ = ["HTMLReport", "write_html_report"]


# Inline CSS so the report is a single file with no external assets.
_CSS = """
:root {
    --bg: #0f172a;
    --panel: #1e293b;
    --text: #e2e8f0;
    --muted: #94a3b8;
    --accent: #38bdf8;
    --positive: #4ade80;
    --negative: #f87171;
    --border: #334155;
}

body {
    margin: 0;
    padding: 0;
    background-color: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    line-height: 1.5;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
}

h1, h2, h3 {
    margin-top: 1.5rem;
    margin-bottom: 0.75rem;
    color: var(--text);
}

h1 { font-size: 2rem; border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; }
h2 { font-size: 1.5rem; color: var(--accent); }

.provenance {
    color: var(--muted);
    font-size: 0.9rem;
    margin-bottom: 2rem;
}

.grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}

.card {
    background-color: var(--panel);
    border: 1px solid var(--border);
    border-radius: 0.5rem;
    padding: 1rem;
}

.card .label {
    color: var(--muted);
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.card .value {
    font-size: 1.75rem;
    font-weight: 600;
    margin-top: 0.25rem;
}

.positive { color: var(--positive); }
.negative { color: var(--negative); }

table {
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0;
    background-color: var(--panel);
    border: 1px solid var(--border);
    border-radius: 0.5rem;
    overflow: hidden;
}

th, td {
    padding: 0.75rem 1rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
}

th {
    background-color: #334155;
    color: var(--accent);
    font-weight: 600;
}

tr:last-child td { border-bottom: none; }

.section { margin-top: 2.5rem; }

.plot {
    margin: 1.5rem 0;
    border: 1px solid var(--border);
    border-radius: 0.5rem;
    overflow: hidden;
}
"""


def _pct(value: float | None) -> str:
    """Format a decimal as a percentage with two decimals."""
    if value is None or not pd.notna(value):
        return "—"
    return f"{value * 100:,.2f}%"


def _num(value: float | None) -> str:
    """Format a decimal as a number with four significant figures."""
    if value is None or not pd.notna(value):
        return "—"
    return f"{value:,.4f}"


class HTMLReport:
    """Build a self-contained HTML report from a pipeline result."""

    def __init__(self, result: PipelineResult) -> None:
        self._result = result

    @property
    def result(self) -> PipelineResult:
        return self._result

    def _summary_cards(self) -> str:
        """Top-line performance cards."""
        if self._result.backtest is None:
            return "<p>No backtest result available.</p>"

        metrics = self._result.backtest.metrics
        bench = self._result.backtest.benchmark_metrics
        excess = self._result.backtest.excess_return

        def card(label: str, value: str, *, positive: bool | None = None) -> str:
            cls = ""
            if positive is True:
                cls = " positive"
            elif positive is False:
                cls = " negative"
            return (
                f'<div class="card">'
                f'<div class="label">{label}</div>'
                f'<div class="value{cls}">{value}</div>'
                f"</div>"
            )

        return (
            '<div class="grid">'
            + card("Strategy CAGR", _pct(metrics.cagr), positive=metrics.cagr > bench.cagr)
            + card("Benchmark CAGR", _pct(bench.cagr))
            + card("Excess return", _pct(excess), positive=excess > 0)
            + card("Max drawdown", _pct(metrics.max_drawdown), positive=False)
            + card("Sharpe", _num(metrics.sharpe), positive=metrics.sharpe > 1.0)
            + card("Annual turnover", _num(metrics.annual_turnover))
            + "</div>"
        )

    def _regime_table(self) -> str:
        """Regime-conditional metrics table."""
        if self._result.backtest is None:
            return ""

        df = self._result.backtest.regime_metrics.reset_index(drop=True)
        if df.empty:
            return ""

        rows = []
        for _, row in df.iterrows():
            rows.append(
                f"<tr>"
                f"<td>{row['regime']}</td>"
                f"<td>{int(row['sessions'])}</td>"
                f"<td>{_pct(row['occupancy'])}</td>"
                f"<td>{_pct(row['ann_return'])}</td>"
                f"<td>{_num(row['ann_volatility'])}</td>"
                f"<td>{_num(row['sharpe'])}</td>"
                f"<td>{_pct(row['hit_rate'])}</td>"
                f"<td>{_pct(row['worst_drawdown'])}</td>"
                f"</tr>"
            )

        return (
            "<table>\n<thead>\n<tr>"
            "<th>Regime</th><th>Sessions</th><th>Occupancy</th><th>Ann. return</th>"
            "<th>Ann. volatility</th><th>Sharpe</th><th>Hit rate</th><th>Worst DD</th>"
            "</tr>\n</thead>\n<tbody>\n" + "\n".join(rows) + "\n</tbody>\n</table>"
        )

    def _feature_table(self) -> str:
        """Latest feature snapshot table."""
        payload = latest_telemetry(self._result)
        rows = []
        for name in payload.features.model_fields:
            value = getattr(payload.features, name)
            formatted = _num(value) if isinstance(value, (int, float)) else str(value)
            rows.append(f"<tr><td>{name}</td><td>{formatted}</td></tr>")

        return (
            "<table>\n<thead>\n<tr><th>Feature</th><th>Value</th></tr>\n</thead>\n"
            f"<tbody>\n{''.join(rows)}\n</tbody>\n</table>"
        )

    def _plotly_figures(self) -> list[tuple[str, str]]:
        """Build Plotly figures and return them as (title, HTML div) pairs."""
        figures: list[tuple[str, str]] = []

        backtest = self._result.backtest
        if backtest is not None:
            equity = backtest.equity
            bench = backtest.benchmark_equity
            combined = pd.DataFrame({"Strategy": equity, "Benchmark": bench})
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=combined.index, y=combined["Strategy"], name="Strategy"))
            fig.add_trace(go.Scatter(x=combined.index, y=combined["Benchmark"], name="Benchmark"))
            fig.update_layout(
                title="Cumulative equity curves",
                xaxis_title="Date",
                yaxis_title="Growth of $1",
                template="plotly_dark",
                paper_bgcolor="#0f172a",
                plot_bgcolor="#1e293b",
                font={"color": "#e2e8f0"},
            )
            figures.append(("Equity", fig.to_html(full_html=False, include_plotlyjs="cdn")))

            drawdown = backtest.drawdown
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown * 100,
                    fill="tozeroy",
                    name="Drawdown",
                    line={"color": "#f87171"},
                )
            )
            fig.update_layout(
                title="Strategy drawdown",
                xaxis_title="Date",
                yaxis_title="Drawdown %",
                template="plotly_dark",
                paper_bgcolor="#0f172a",
                plot_bgcolor="#1e293b",
                font={"color": "#e2e8f0"},
            )
            figures.append(("Drawdown", fig.to_html(full_html=False, include_plotlyjs="cdn")))

        # Regime path: stacked area of the posterior probabilities.
        probs = self._result.walkforward.probabilities
        fig = go.Figure()
        for regime in probs.columns:
            fig.add_trace(
                go.Scatter(
                    x=probs.index,
                    y=probs[regime] * 100,
                    name=regime,
                    mode="lines",
                    stackgroup="one",
                    line={"width": 0.5},
                )
            )
        fig.update_layout(
            title="Regime probabilities",
            xaxis_title="Date",
            yaxis_title="Probability %",
            template="plotly_dark",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#1e293b",
            font={"color": "#e2e8f0"},
        )
        figures.append(("Regimes", fig.to_html(full_html=False, include_plotlyjs="cdn")))

        return figures

    def _plots(self) -> str:
        """Render all Plotly figures into the report body."""
        try:
            figures = self._plotly_figures()
        except ModuleNotFoundError as exc:
            return f'<p class="negative">{exc}</p>'

        parts = []
        for title, div in figures:
            section = f'<div class="section"><h3>{title}</h3>'
            plot = f'<div class="plot">{div}</div></div>'
            parts.append(section + plot)
        return "\n".join(parts)

    def render(self) -> str:
        """Return the full HTML document as a string."""
        manifest = self._result.manifest
        asof = self._result.walkforward.labels.index[-1].date()

        css_b64 = base64.b64encode(_CSS.encode()).decode()
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>regimelab report — {asof}</title>
    <link rel="stylesheet" href="data:text/css;base64,{css_b64}">
</head>
<body>
<div class="container">
    <h1>regimelab</h1>
    <p class="provenance">{manifest.provenance_line()} | as-of {asof}</p>

    <div class="section">
        <h2>Performance summary</h2>
        {self._summary_cards()}
    </div>

    <div class="section">
        <h2>Regime-conditional metrics</h2>
        {self._regime_table()}
    </div>

    <div class="section">
        <h2>Latest feature snapshot</h2>
        {self._feature_table()}
    </div>

    {self._plots()}
</div>
</body>
</html>"""


def write_html_report(result: PipelineResult, path: Path | str) -> Path:
    """Render and write a report to disk.

    Parameters
    ----------
    result
        Output of :func:`~regimelab.pipeline.run_pipeline`.
    path
        Destination file path. Parent directories are created on demand.

    Returns
    -------
    pathlib.Path
        The path the report was written to.
    """
    report = HTMLReport(result)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.render(), encoding="utf-8")
    return output
