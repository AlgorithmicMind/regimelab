"""Typer command-line interface.

The CLI is the only user-facing entry point; the library imports nothing from it,
so importing :mod:`regimelab` in a notebook never pulls in ``typer``.
"""

from __future__ import annotations

import datetime as dt
import tomllib
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from regimelab.models.types import TelemetryPayload
from regimelab.pipeline import run_pipeline, run_single_asof
from regimelab.report.html import write_html_report
from regimelab.settings import Settings
from regimelab.tui import RegimeLabApp

app = typer.Typer(
    name="regimelab",
    no_args_is_help=True,
    rich_markup_mode="rich",
    help="Point-in-time market-regime detection and backtesting engine.",
)


def _load_settings(config: Path | None = None) -> Settings:
    """Load settings, optionally from a specific TOML file.

    Pydantic-settings reads ``.env`` and ``regimelab.toml`` by default. A custom
    ``--config`` path is parsed and passed as init overrides, so it has higher
    precedence than the default file. If the file does not exist, defaults are
    used and the user is warned.
    """
    overrides: dict[str, Any] = {}
    if config is not None:
        if not config.is_file():
            typer.secho(
                f"config file {config} not found, using defaults",
                fg=typer.colors.YELLOW,
                err=True,
            )
        else:
            with config.open("rb") as fh:
                overrides = tomllib.load(fh)
    try:
        return Settings(**overrides)
    except Exception as exc:
        raise typer.BadParameter(f"could not load settings: {exc}") from exc


@app.command()
def run(
    report: Annotated[
        Path | None,
        typer.Option(
            "--report",
            "-r",
            help="Write an HTML report to this path.",
        ),
    ] = None,
    tui: Annotated[
        bool,
        typer.Option(
            "--tui",
            "-t",
            help="Launch the Textual dashboard after the run.",
        ),
    ] = False,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Path to a TOML configuration file.",
        ),
    ] = None,
) -> None:
    """Run the full walk-forward pipeline and backtest."""
    settings = _load_settings(config)
    console = Console()
    total_phases = 4 if report is not None else 3

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        overall = progress.add_task("Initializing pipeline...", total=total_phases)
        fold_task = progress.add_task("", total=None, visible=False)

        def _status(message: str) -> None:
            progress.update(overall, description=message, advance=1)

        def _progress_callback(current_fold: int, total_folds: int, test_date: str) -> None:
            if current_fold == 1:
                progress.update(
                    fold_task,
                    visible=True,
                    total=total_folds,
                    completed=0,
                )
            progress.update(
                fold_task,
                completed=current_fold,
                description=(
                    f"Running walk-forward regime estimation "
                    f"(fold {current_fold} of {total_folds}) — {test_date}"
                ),
            )

        result = run_pipeline(
            settings,
            command="run",
            progress_callback=_progress_callback,
            status_callback=_status,
        )

        if report is not None:
            _status(f"Writing interactive HTML report to {report}...")
            path = write_html_report(result, report)
            progress.update(overall, completed=total_phases)

        if tui:
            try:
                RegimeLabApp(result).run()
            except ModuleNotFoundError as exc:
                raise typer.BadParameter(str(exc)) from exc

    if result.backtest is not None:
        summary_lines = [
            f"Strategy CAGR: {result.backtest.metrics.cagr:.2%}",
            f"Benchmark CAGR: {result.backtest.benchmark_metrics.cagr:.2%}",
            f"Sharpe Ratio: {result.backtest.metrics.sharpe:.2f}",
        ]
    else:
        summary_lines = ["Backtest result not available."]

    if report is not None:
        summary_lines.append(f"Report written: {path}")

    console.print(
        Panel.fit(
            "\n".join(summary_lines),
            title="RegimeLab Results",
            border_style="blue",
        ),
    )


@app.command()
def asof(
    date: Annotated[
        str,
        typer.Argument(
            help="Historical as-of date in ISO format (YYYY-MM-DD).",
        ),
    ],
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Path to a TOML configuration file.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Write the payload JSON to this path.",
        ),
    ] = None,
) -> None:
    """Produce a telemetry payload for a specific historical as-of date."""
    try:
        asof_date = dt.date.fromisoformat(date)
    except ValueError as exc:
        raise typer.BadParameter("date must be in YYYY-MM-DD format") from exc
    settings = _load_settings(config)
    payload = run_single_asof(settings, asof_date)
    _emit_payload(payload, output)


@app.command()
def report(
    output: Annotated[
        Path,
        typer.Argument(
            help="Destination HTML report path.",
        ),
    ],
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Path to a TOML configuration file.",
        ),
    ] = None,
) -> None:
    """Run the pipeline and write an HTML report."""
    settings = _load_settings(config)
    result = run_pipeline(settings, command="report")
    path = write_html_report(result, output)
    typer.echo(f"report written: {path}")


@app.command()
def tui(
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Path to a TOML configuration file.",
        ),
    ] = None,
) -> None:
    """Run the pipeline and launch the Textual dashboard."""
    settings = _load_settings(config)
    result = run_pipeline(settings, command="tui")
    try:
        RegimeLabApp(result).run()
    except ModuleNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _emit_payload(payload: TelemetryPayload, output: Path | None) -> None:
    """Print a payload or write it to disk."""
    json = payload.model_dump_json(indent=2)
    if output is None:
        typer.echo(json)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json, encoding="utf-8")
        typer.echo(f"payload written: {output}")


def main() -> None:
    """Entry point for ``python -m regimelab`` and the console script."""
    app()
