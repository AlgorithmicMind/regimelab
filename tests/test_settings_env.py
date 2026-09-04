"""Regression tests for pydantic-settings environment handling."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from regimelab.settings import Settings


def test_settings_ignores_unrelated_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown env vars without the REGIMELAB_ prefix must not break Settings()."""
    monkeypatch.setenv("FOREIGN_API_KEY", "xyz")
    monkeypatch.setenv("GOOGLE_API_KEY", "super-secret")

    # Should not raise despite unrelated variables in os.environ.
    settings = Settings()
    assert settings is not None


def test_settings_still_maps_prefixed_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """Recognised REGIMELAB_ variables must keep overriding defaults."""
    monkeypatch.setenv("REGIMELAB_MODEL__N_STATES", "3")

    settings = Settings()
    assert settings.model.n_states == 3


def test_settings_rejects_invalid_prefixed_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validation still applies to known fields."""
    monkeypatch.setenv("REGIMELAB_MODEL__N_STATES", "99")

    with pytest.raises(ValidationError):
        Settings()
