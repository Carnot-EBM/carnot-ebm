"""PlattScaledJEPA: temperature-scaled wrapper around an EORM-style energy scorer.

**Researcher summary:**
    Platt scaling is a simple post-hoc calibration technique that divides a
    model's raw logit (or energy score) by a learned scalar temperature T.
    Exp 646 measured T via the Expected Calibration Error (ECE) objective on a
    held-out set, finding T that minimises the gap between predicted confidence
    and empirical accuracy.

    This module wraps any ``EORMModel``-compatible energy scorer and applies
    that temperature division so that the pipeline's Tier 2 confidence scores
    are properly calibrated (ECE < 0.10).

**Detailed explanation for engineers:**
    Why does calibration matter for a cascade verifier?

    The ThreeTierPipeline's Tier 2 makes a binary "clear / don't clear" decision
    based on whether the energy score is below a threshold.  If the raw EORM
    scores are poorly calibrated — e.g., the model is overconfident, assigning
    very low energies even to incorrect responses — then the false-negative rate
    rises: wrong answers get cleared too often.

    Platt scaling fixes this: dividing by temperature T > 1 "widens" the
    distribution, reducing overconfidence.  Dividing by T < 1 narrows it,
    reducing underconfidence.  The net effect is that the energy values more
    faithfully reflect the model's actual accuracy, letting the threshold work
    as intended.

    **Interface contract:**
    ``PlattScaledJEPA.energy(cot_input)`` has the EXACT same signature as
    ``EORMModel.energy(cot_input)``, so it is a drop-in replacement for Tier 2
    in ``ThreeTierPipeline`` with no other code changes.

    **Temperature source:**
    The temperature is loaded from the Exp 646 JSON artifact
    (``platt_temperature`` field).  At runtime it can be overridden with the
    environment variable ``JEPA_TIER2_PLATT_TEMPERATURE`` for debugging.

Spec: REQ-VERIFY-151, SCENARIO-VERIFY-204, SCENARIO-VERIFY-205
"""

from __future__ import annotations

import os

from carnot.models.eorm import CoTEnergyInput, EORMModel


class PlattScaledJEPA:
    """EORM energy model with Platt-scaling temperature calibration applied.

    Wraps an ``EORMModel`` and divides every raw energy score by a scalar
    temperature before returning it to the caller.  This is the Tier 2 model
    deployed by Exp 657 (JEPA v14 cascade deployment).

    Parameters
    ----------
    eorm_model : EORMModel
        The underlying EORM model whose ``energy()`` method will be called.
    temperature : float
        Platt scaling temperature learned in Exp 646.  Must be > 0.
        Loaded from ``results/experiment_646_jepa_platt.json`` by the caller.
        Can be overridden at runtime via the ``JEPA_TIER2_PLATT_TEMPERATURE``
        environment variable (useful for ablation studies without re-running
        the full experiment).

    Spec: REQ-VERIFY-151
    """

    def __init__(self, eorm_model: EORMModel, temperature: float) -> None:
        if temperature <= 0:
            raise ValueError(
                f"Platt temperature must be > 0, got {temperature}. "
                "A non-positive temperature would flip or collapse the energy scale."
            )
        # Allow env-var override so operators can tune calibration in prod without
        # re-deploying code (follows the pattern of JEPA_TIER2_PLATT_TEMPERATURE).
        env_temp = os.environ.get("JEPA_TIER2_PLATT_TEMPERATURE")
        if env_temp is not None:
            try:
                temperature = float(env_temp)
                if temperature <= 0:
                    raise ValueError(f"JEPA_TIER2_PLATT_TEMPERATURE must be > 0, got {env_temp}")
            except ValueError as exc:
                raise ValueError(
                    f"JEPA_TIER2_PLATT_TEMPERATURE env var is not a valid float: {env_temp!r}"
                ) from exc

        self._eorm_model = eorm_model
        self.temperature = temperature

    def energy(self, cot_input: CoTEnergyInput) -> float:
        """Return the temperature-scaled energy for a (question, response) pair.

        Calls the underlying EORM model's ``energy()`` and divides by the
        Platt temperature.  Lower score still means "more likely correct";
        temperature scaling only changes the magnitude, not the ordering.

        Parameters
        ----------
        cot_input : CoTEnergyInput
            The (question_text, response_text) pair to score.

        Returns
        -------
        float
            Raw EORM energy divided by ``self.temperature``.

        Spec: REQ-VERIFY-151-1
        """
        raw_energy = float(self._eorm_model.energy(cot_input))
        return raw_energy / self.temperature
