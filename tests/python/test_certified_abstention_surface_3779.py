"""Tests for the Exp 3779 certified abstention product surface.

Spec: REQ-SPOE-3779, SCENARIO-SPOE-3779.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.pipeline import certified_abstention_surface as abstention
from carnot.pipeline import second_pair_detector as spd


ROOT = Path(__file__).resolve().parents[2]
EXP3771_PATH = ROOT / "results/experiment_3771_certified_abstention_operating_point.json"


def _domain_examples(domain: str = "math", *, n: int = 80) -> list[spd.LabeledDetectorExample]:
    examples: list[spd.LabeledDetectorExample] = []
    for idx in range(n):
        label = 1 if idx < n // 2 else 0
        ensemble = 0.95 - 0.004 * idx if label else 0.05 + 0.001 * (idx - n // 2)
        confidence_error = 0.82 - 0.003 * idx if label else 0.18 + 0.001 * (idx - n // 2)
        examples.append(
            spd.LabeledDetectorExample(
                domain=domain,
                label=label,
                ensemble_energy=ensemble,
                confidence_error=confidence_error,
                example_id=f"{domain}-3779-{idx}",
            )
        )
    return examples


def test_req_spoe_3779_loads_certified_threshold_from_exp3771_artifact() -> None:
    """REQ-SPOE-3779: the default threshold is loaded from Exp 3771."""

    config = abstention.load_certified_abstention_config(EXP3771_PATH)
    artifact = json.loads(EXP3771_PATH.read_text(encoding="utf-8"))

    assert config.threshold == pytest.approx(artifact["selected_threshold"])
    assert config.coverage == pytest.approx(artifact["coverage_at_operating_point"])
    assert config.certified_risk_bound == pytest.approx(artifact["certified_risk_bound"])
    assert config.delta == pytest.approx(0.05)
    assert config.n_calibration == artifact["n_calibration"]
    assert config.threshold_source == str(EXP3771_PATH)


def test_scenario_spoe_3779_default_off_preserves_score_candidates_shape() -> None:
    """SCENARIO-SPOE-3779: default-off does not add abstention verdict fields."""

    response = spd.score_candidates(
        [
            spd.CandidateScoreInput(
                candidate_id="default-off",
                domain="math",
                text="We compute 8 + 5 = 13.",
                confidence_error=0.9,
                ensemble_energy=0.95,
            )
        ],
        examples=_domain_examples(),
    )

    row = response["scores"][0]
    assert row["abstained"] is False
    assert "abstention_mode_enabled" not in row
    assert "abstention_verdict" not in row
    assert row["calibrated_error_score"] is not None
    assert row["operating_point"]["fpr_budget"] == 0.10


def test_scenario_spoe_3779_above_threshold_confident_below_threshold_abstains() -> None:
    """SCENARIO-SPOE-3779: enabled mode returns confident vs review verdicts."""

    config = abstention.load_certified_abstention_config(EXP3771_PATH)
    response = spd.score_candidates(
        [
            spd.CandidateScoreInput(
                candidate_id="confident-error",
                domain="math",
                text="We compute 8 + 5 = 14.",
                confidence_error=1.0,
                ensemble_energy=1.0,
            ),
            spd.CandidateScoreInput(
                candidate_id="uncertain-midpoint",
                domain="math",
                text="We compute 8 + 5 = 13.",
                confidence_error=0.5,
                ensemble_energy=0.5,
            ),
        ],
        examples=_domain_examples(),
        abstention_mode=True,
    )

    rows = {row["candidate_id"]: row for row in response["scores"]}
    confident = rows["confident-error"]
    uncertain = rows["uncertain-midpoint"]

    assert response["abstention_mode"]["enabled"] is True
    assert response["abstention_mode"]["certified_threshold"] == pytest.approx(config.threshold)
    assert confident["abstention_mode_enabled"] is True
    assert confident["abstention_score"] >= config.threshold
    assert confident["abstention_verdict"] == abstention.CONFIDENT_ERROR_VERDICT
    assert confident["route_to_review"] is False
    assert confident["abstained"] is False

    assert uncertain["abstention_mode_enabled"] is True
    assert uncertain["abstention_score"] < config.threshold
    assert uncertain["abstention_verdict"] == abstention.ABSTAIN_VERDICT
    assert uncertain["route_to_review"] is True
    assert uncertain["abstained"] is True
    assert uncertain["certified_abstention"]["coverage"] == pytest.approx(config.coverage)
    assert uncertain["certified_abstention"]["certified_risk_bound"] == pytest.approx(
        config.certified_risk_bound
    )
    assert uncertain["certified_abstention"]["delta"] == pytest.approx(config.delta)
    assert uncertain["certified_abstention"]["n_calibration"] == config.n_calibration
    assert uncertain["certified_abstention"]["threshold_source"] == str(EXP3771_PATH)


def test_req_spoe_3779_operator_threshold_override_is_explicit() -> None:
    """REQ-SPOE-3779: operators can tune the threshold without editing code."""

    response = spd.score_candidates(
        [
            spd.CandidateScoreInput(
                candidate_id="operator-override",
                domain="math",
                text="We compute 8 + 5 = 13.",
                confidence_error=0.5,
                ensemble_energy=0.5,
            )
        ],
        examples=_domain_examples(),
        abstention_mode=True,
        abstention_threshold=0.5,
    )

    row = response["scores"][0]
    assert response["abstention_mode"]["operator_threshold_override"] is True
    assert row["abstention_threshold"] == pytest.approx(0.5)
    assert row["abstention_verdict"] != abstention.ABSTAIN_VERDICT
