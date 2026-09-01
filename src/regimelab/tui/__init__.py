"""Textual TUI dashboard.

The TUI is optional: it is only required if the ``tui`` extra is installed
(``uv pip install "regimelab[tui]"`` or ``pip install textual``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["RegimeLabApp"]

if TYPE_CHECKING:
    from regimelab.pipeline import PipelineResult
    from regimelab.tui.app import RegimeLabApp as _RegimeLabApp


def _check_textual() -> None:
    try:
        import textual  # noqa: F401, PLC0415
    except ModuleNotFoundError as exc:
        msg = (
            "textual is required for the TUI; install it with "
            "`uv pip install textual` or `pip install textual`"
        )
        raise ModuleNotFoundError(msg) from exc


class _LazyApp:
    """Proxy that imports the real Textual app only when requested."""

    def __call__(self, result: PipelineResult) -> _RegimeLabApp:
        _check_textual()
        from regimelab.tui.app import RegimeLabApp as _App  # noqa: PLC0415

        return _App(result)


RegimeLabApp = _LazyApp()
