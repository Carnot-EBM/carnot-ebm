"""Regression tests for REQ-VERIFY-7158's exact-source substrate."""

from __future__ import annotations

import json
from pathlib import Path

import scripts.adversarial_verify as av


def test_req_verify_7158_exact_source_fixture_has_no_model_duration_floor() -> None:
    """REQ-VERIFY-7158 classifies deterministic construction as tiny CPU work."""

    artifact = {
        "inference_substrate": "exact_source_fixture_construction",
        "inference_substrate_class": "cpu_exact_solver_or_simulator",
        "execution_venue": "host",
        "duration_s": 0.1,
        "verifier_is_oracle": False,
        "verdict_class": "positive",
        "honest_verdict": "complete_positive_counterfactual_fixture_ready_no_verifier_value_claim",
        "random_seed": 7_158_202_609_09,
        "reproducibility_checksum": "sha256:fixture",
        "rows": [{"fixture_id": "eef-example", "condition": "supported"}],
    }
    floor = av.duration_floor_for_artifact(artifact)
    assert floor == {
        "substrate": "exact_source_fixture_construction",
        "min_duration_s": av.DETERMINISTIC_VERIFIER_MIN_DURATION_S,
        "reason": "deterministic_verifier",
    }


def test_scenario_verify_7158_adversarial_gate_accepts_fixture_substrate(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-7158-ARTIFACT rejects no legitimate short CPU fixture."""

    artifact = {
        "inference_substrate": "exact_source_fixture_construction",
        "inference_substrate_class": "cpu_exact_solver_or_simulator",
        "execution_venue": "host",
        "duration_s": 0.1,
        "verifier_is_oracle": False,
        "verdict_class": "positive",
        "honest_verdict": "complete_positive_counterfactual_fixture_ready_no_verifier_value_claim",
        "random_seed": 7_158_202_609_09,
        "reproducibility_checksum": "sha256:fixture",
        "rows": [{"fixture_id": "eef-example", "condition": "supported"}],
    }
    path = tmp_path / "experiment_7158_fixture.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    report = av.verify_artifact(path)
    kinds = {flag["kind"] for flag in report["flags"]}
    assert "SUBSTRATE_HAS_NO_DURATION_FLOOR" not in kinds
    assert "DURATION_TOO_SHORT" not in kinds
