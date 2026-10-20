"""Inspect built artifacts without importing the checkout or running a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import tomllib
import zipfile
from email.parser import Parser
from pathlib import Path

from packaging.metadata import Metadata

_MODULES = (
    "__init__.py",
    "__main__.py",
    "cli.py",
    "py.typed",
    "settings.py",
    "pipeline.py",
    "models/hmm.py",
    "models/types.py",
    "backtest/__init__.py",
    "backtest/engine.py",
    "backtest/metrics.py",
    "backtest/portfolio.py",
    "backtest/_portfolio_inputs.py",
    "backtest/_portfolio_ledger.py",
)
_SOURCE_FILES = (
    "pyproject.toml",
    "uv.lock",
    "README.md",
    "CHANGELOG.md",
    "LICENSE",
    "docs/architecture.md",
    "docs/next_open_execution.md",
    "tests/data/backtest_golden.parquet",
    "tests/release_smoke.py",
    "tests/release_artifacts.py",
    "tests/test_legacy_price_validation.py",
    "tests/test_telemetry_quality.py",
    "tests/test_next_open_ledger.py",
    "tests/test_next_open_inputs.py",
    "tests/_next_open_fixtures.py",
)


def main() -> None:
    """Check exact source bytes, metadata and absence of private/build debris."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--sdist", type=Path, required=True)
    arguments = parser.parse_args()
    source = arguments.source.resolve()
    version = "0.4.0"
    project = tomllib.loads((source / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["version"] == version
    with zipfile.ZipFile(arguments.wheel) as wheel:
        names = wheel.namelist()
        for module in _MODULES:
            assert (
                wheel.read(f"regimelab/{module}")
                == (source / "src/regimelab" / module).read_bytes()
            )
        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        assert len(metadata_names) == 1
        metadata_bytes = wheel.read(metadata_names[0])
        Metadata.from_email(metadata_bytes, validate=True)
        metadata = Parser().parsestr(metadata_bytes.decode("utf-8"))
        assert metadata["Name"] == "regimelab"
        assert metadata["Version"] == version
        assert metadata["Requires-Python"] == ">=3.12"
        entrypoints = [name for name in names if name.endswith(".dist-info/entry_points.txt")]
        assert len(entrypoints) == 1
        assert "regimelab = regimelab.cli:app" in wheel.read(entrypoints[0]).decode("utf-8")
        for name in names:
            assert not name.startswith(("tests/", "artifacts/", "workspaces/", ".venv/"))
            assert ".env" not in Path(name).parts

    prefix = f"regimelab-{version}/"
    with tarfile.open(arguments.sdist, "r:gz") as sdist:
        files = {member.name: member for member in sdist.getmembers() if member.isfile()}
        package_info = sdist.extractfile(files[f"{prefix}PKG-INFO"])
        assert package_info is not None
        source_metadata = Metadata.from_email(package_info.read(), validate=True)
        assert str(source_metadata.version) == version
        for relative in (*_SOURCE_FILES, *(f"src/regimelab/{m}" for m in _MODULES)):
            member = sdist.extractfile(files[f"{prefix}{relative}"])
            assert member is not None
            assert member.read() == (source / relative).read_bytes(), relative
        for name in files:
            relative = name.removeprefix(prefix)
            assert not relative.startswith(("artifacts/", "workspaces/", ".venv/", ".git/"))
            assert ".env" not in Path(relative).parts

    print(
        json.dumps(
            {
                "version": version,
                "wheel": str(arguments.wheel),
                "wheel_sha256": hashlib.sha256(arguments.wheel.read_bytes()).hexdigest(),
                "sdist": str(arguments.sdist),
                "sdist_sha256": hashlib.sha256(arguments.sdist.read_bytes()).hexdigest(),
                "required_sources_match": True,
                "golden_and_documentation_in_sdist": True,
                "typed_public_api_in_wheel": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
