"""Automated HTML reporting.

Reports are self-contained single HTML files with embedded interactive Plotly
charts. They do not depend on any backend server or internet connection at view
time; all data is inlined.
"""

from __future__ import annotations

from regimelab.report.html import HTMLReport, write_html_report

__all__ = ["HTMLReport", "write_html_report"]
