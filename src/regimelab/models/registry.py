"""Classifier registry.

Keeps the name-to-class mapping in one place so the CLI, the pipeline and the
walk-forward loop cannot drift apart on which classifiers exist.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from regimelab.models.base import RegimeModel
from regimelab.models.gmm import GaussianMixtureRegimeModel
from regimelab.models.hamilton import MarkovSwitchingRegimeModel
from regimelab.models.hmm import HiddenMarkovRegimeModel

if TYPE_CHECKING:
    from regimelab.settings import ModelSettings

__all__ = ["CLASSIFIERS", "build_classifier"]

CLASSIFIERS: dict[str, type[RegimeModel]] = {
    HiddenMarkovRegimeModel.name: HiddenMarkovRegimeModel,
    GaussianMixtureRegimeModel.name: GaussianMixtureRegimeModel,
    MarkovSwitchingRegimeModel.name: MarkovSwitchingRegimeModel,
}
"""Available classifiers, keyed by the name used in configuration."""


def build_classifier(settings: ModelSettings, name: str | None = None) -> RegimeModel:
    """Instantiate a classifier by name.

    Parameters
    ----------
    settings
        Model configuration section.
    name
        Classifier name. Defaults to ``settings.classifier``.

    Returns
    -------
    RegimeModel
        Ready-to-fit classifier.

    Raises
    ------
    ValueError
        If ``name`` is not registered.
    """
    key = name or settings.classifier
    try:
        factory = CLASSIFIERS[key]
    except KeyError:
        available = ", ".join(sorted(CLASSIFIERS))
        msg = f"unknown classifier {key!r}; available: {available}"
        raise ValueError(msg) from None

    return factory(settings)
