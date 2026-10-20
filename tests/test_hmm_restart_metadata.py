"""The chosen HMM restart signs its own convergence metadata."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, sentinel

import numpy as np
import pytest

from regimelab.models.hmm import HiddenMarkovRegimeModel
from regimelab.settings import ModelSettings


@pytest.mark.parametrize(
    ("convergence", "expected"),
    [((True, False), False), ((False, True), True), ((True, True), True), ((False, False), False)],
)
def test_convergence_belongs_to_selected_restart(
    monkeypatch: pytest.MonkeyPatch,
    convergence: tuple[bool, bool],
    expected: bool,
) -> None:
    first = SimpleNamespace(
        fit=Mock(),
        score=Mock(return_value=-100.0),
        monitor_=SimpleNamespace(converged=convergence[0]),
    )
    second = SimpleNamespace(
        fit=Mock(),
        score=Mock(return_value=-50.0),
        monitor_=SimpleNamespace(converged=convergence[1]),
    )
    factory = Mock(side_effect=[first, second])
    monkeypatch.setattr("hmmlearn.hmm.GaussianHMM", factory)
    classifier = HiddenMarkovRegimeModel(ModelSettings(n_states=3, n_init=2))
    package = Mock(return_value=sentinel.estimate)
    monkeypatch.setattr(classifier, "_estimate_from_model", package)
    matrix = np.zeros((10, 5), dtype="float64")
    result = classifier._estimate(matrix, seed=1)
    assert result is sentinel.estimate
    assert factory.call_count == 2
    package.assert_called_once()
    assert package.call_args.args[0] is second
    assert package.call_args.args[1] is matrix
    assert package.call_args.args[2] is expected
