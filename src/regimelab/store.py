"""Artefact persistence and run provenance.

"The model is deterministic" is an unverifiable claim unless each run records
what it ran on. Every invocation writes a :class:`RunManifest` capturing the git
commit, the working-tree cleanliness, the resolved settings, the RNG seed, the
pinned versions of the numerical stack and a content digest of the input panel.

Two runs whose manifests agree must produce identical outputs. If they do not,
the bug is in the engine; if the manifests differ, the inputs differ. That
distinction is what makes a regression debuggable months later.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from regimelab.settings import Settings

__all__ = [
    "GitState",
    "RunManifest",
    "RunStore",
    "digest_frame",
]

# Packages whose version materially changes numerical output.
_TRACKED_PACKAGES: Final[tuple[str, ...]] = (
    "regimelab",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "hmmlearn",
    "statsmodels",
    "pyarrow",
)

_GIT_TIMEOUT_SECONDS: Final[float] = 5.0


def _run_git(*args: str) -> str | None:
    """Run a git command, returning ``None`` if git or the repo is unavailable."""
    try:
        completed = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
            cwd=Path(__file__).resolve().parent,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


class GitState(BaseModel):
    """Source-control state at run time."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    commit: str | None = None
    branch: str | None = None
    dirty: bool | None = None
    """``True`` if the working tree had uncommitted changes. A dirty run is not
    reproducible from the commit alone, and the reports say so explicitly."""

    @classmethod
    def capture(cls) -> GitState:
        """Read the current git state, degrading gracefully outside a repo."""
        commit = _run_git("rev-parse", "HEAD")
        branch = _run_git("rev-parse", "--abbrev-ref", "HEAD")
        status = _run_git("status", "--porcelain")
        return cls(
            commit=commit,
            branch=branch,
            dirty=None if status is None else bool(status),
        )


def _package_versions() -> dict[str, str]:
    """Installed versions of the packages that affect numerical results."""
    versions: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def digest_frame(frame: pd.DataFrame | pd.Series[Any]) -> str:
    """Return a stable SHA-256 digest of a frame's contents.

    Hashes the values, the index and the column labels. Deliberately *not* a
    hash of a serialised Parquet file: Parquet output is not byte-stable across
    library versions or compression settings, so it would report spurious
    changes.

    Parameters
    ----------
    frame
        Data to digest. Column order is included in the digest because feature
        ordering is part of the model contract.

    Returns
    -------
    str
        Hex-encoded SHA-256 digest.
    """
    hasher = hashlib.sha256()

    if isinstance(frame, pd.Series):
        labels: list[str] = [str(frame.name)]
    else:
        labels = [str(column) for column in frame.columns]
    hasher.update("\x1f".join(labels).encode())
    hasher.update(str(frame.shape).encode())

    row_hashes = pd.util.hash_pandas_object(frame, index=True)
    hasher.update(row_hashes.to_numpy(dtype="uint64").tobytes())
    return hasher.hexdigest()


class RunManifest(BaseModel):
    """Immutable provenance record for a single engine invocation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: dt.datetime = Field(
        default_factory=lambda: dt.datetime.now(dt.UTC),
    )
    command: str
    """CLI command that produced the run, e.g. ``"scan"``."""

    asof: dt.date | None = None
    """Decision date, when the run targets a single as-of date."""

    git: GitState = Field(default_factory=GitState.capture)
    python_version: str = Field(default_factory=lambda: sys.version.split()[0])
    platform: str = Field(default_factory=platform.platform)
    packages: dict[str, str] = Field(default_factory=_package_versions)

    seed: int
    settings: dict[str, Any]
    """Fully resolved settings, so the run can be replayed without the original
    environment or TOML file."""

    input_digest: str | None = None
    """Digest of the aligned input panel, from :func:`digest_frame`."""

    @classmethod
    def capture(
        cls,
        *,
        command: str,
        settings: Settings,
        asof: dt.date | None = None,
        input_digest: str | None = None,
    ) -> RunManifest:
        """Build a manifest from the live process state.

        Parameters
        ----------
        command
            Name of the CLI command being executed.
        settings
            Resolved configuration; serialised into the manifest.
        asof
            Decision date, for single-date runs.
        input_digest
            Digest of the input panel, if already computed.
        """
        return cls(
            command=command,
            asof=asof,
            seed=settings.model.seed,
            settings=settings.model_dump(mode="json"),
            input_digest=input_digest,
        )

    @property
    def is_reproducible(self) -> bool:
        """Whether this run can be reconstructed from source control alone."""
        return self.git.commit is not None and self.git.dirty is False

    def provenance_line(self) -> str:
        """One-line human-readable provenance summary for reports and the TUI."""
        commit = self.git.commit[:8] if self.git.commit else "unknown"
        suffix = "" if self.git.dirty is False else " (dirty tree)"
        return f"run {self.run_id} @ {commit}{suffix} seed={self.seed}"


class RunStore:
    """Filesystem layout for a single run's artefacts.

    Artefacts live under ``<artifacts_dir>/<YYYYMMDD-HHMMSS>-<run_id>/`` so runs
    never overwrite each other and sort chronologically.
    """

    def __init__(self, root: Path, manifest: RunManifest) -> None:
        stamp = manifest.created_at.strftime("%Y%m%d-%H%M%S")
        self._dir = Path(root) / f"{stamp}-{manifest.run_id}"
        self._manifest = manifest

    @property
    def directory(self) -> Path:
        """Run directory. Created on first write, not on construction."""
        return self._dir

    @property
    def manifest(self) -> RunManifest:
        """The manifest this store was opened with."""
        return self._manifest

    def _ensure_dir(self) -> Path:
        self._dir.mkdir(parents=True, exist_ok=True)
        return self._dir

    def write_manifest(self) -> Path:
        """Persist the run manifest as JSON."""
        path = self._ensure_dir() / "manifest.json"
        payload = self._manifest.model_dump(mode="json")
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def write_frame(self, frame: pd.DataFrame, name: str) -> Path:
        """Persist a dataframe as Parquet.

        Parquet rather than CSV so dtypes, the DatetimeIndex and NaN-versus-empty
        survive the round trip; a CSV round trip silently changes numerical
        results.
        """
        path = self._ensure_dir() / f"{name}.parquet"
        frame.to_parquet(path, index=True, compression="zstd")
        return path

    def write_json(self, payload: BaseModel, name: str) -> Path:
        """Persist a Pydantic model as indented JSON."""
        path = self._ensure_dir() / f"{name}.json"
        path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
        return path

    def write_text(self, content: str, name: str) -> Path:
        """Persist arbitrary text, e.g. a rendered HTML report."""
        path = self._ensure_dir() / name
        path.write_text(content, encoding="utf-8")
        return path
