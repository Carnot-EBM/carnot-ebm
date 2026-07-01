"""Tests for Exp 5117 TACO harm-gated CSP scale diagnostic.

Spec refs: REQ-VERIFY-5117, SCENARIO-VERIFY-5117.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot import experiment_5117_taco_harm_gated_scale_v469 as exp
from scripts import experiment_5117_taco_harm_gated_scale_v469 as script_mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
ARTIFACT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def test_req_verify_5117_spec_declares_harm_gate_contract() -> None:
    """REQ-VERIFY-5117: OpenSpec anchors the harm-gated scale diagnostic."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5117") :]

    for marker in (
        "REQ-VERIFY-5117",
        "SCENARIO-VERIFY-5117",
        "harm-gated adaptive policy",
        exp.RESULT_RELATIVE_PATH,
        exp.EXPERIMENT_ID,
        exp.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_req_verify_5117_scale_suite_varies_density_arity_and_frustration() -> None:
    """REQ-VERIFY-5117: deterministic suite spans varied CSP structure."""

    first = exp.build_scaled_csp_suite()
    second = exp.build_scaled_csp_suite()

    assert first == second
    assert len(first) >= 18
    assert {instance.split for instance in first} == {"train", "dev", "heldout"}
    assert len({instance.n_colors for instance in first}) >= 2
    assert {instance.expected_colorable for instance in first} == {False, True}
    assert len({instance.density_bucket for instance in first}) >= 3
    assert {arity for instance in first for arity in instance.constraint_arities} >= {2, 3, 4}
    assert len({instance.frustration for instance in first}) >= 3


def test_scenario_verify_5117_guard_falls_back_on_known_harmful_even_wheel() -> None:
    """SCENARIO-VERIFY-5117: predicted harm falls back to the exact baseline order."""

    family = {instance.instance_id: instance for instance in exp.build_scaled_csp_suite()}
    solver = exp.ExactCspSolver()
    instance = family["dev_wheel6_even_sat"]
    result = exp.evaluate_instance(instance, solver)

    assert result["unguarded_harmful"] is True
    assert result["guarded_harmful"] is False
    assert result["gate_decision"]["use_adaptive"] is False
    assert result["guarded"]["variable_order"] == list(exp.baseline_order(instance))
    assert result["guarded"]["effort"]["total_effort_score"] == result["baseline"]["effort"]["total_effort_score"]
    assert result["baseline"]["colorable"] is result["guarded"]["colorable"]


def test_req_verify_5117_no_wrong_label_regressions_across_all_policies() -> None:
    """REQ-VERIFY-5117: baseline, unguarded, and guarded labels all match exact authority."""

    solver = exp.ExactCspSolver()

    for instance in exp.build_scaled_csp_suite():
        row = exp.evaluate_instance(instance, solver)
        exact_label = row["exact_label"]["colorable"]
        assert row["wrong_label"] is False
        assert exact_label is instance.expected_colorable
        for arm_name in ("baseline", "unguarded", "guarded"):
            arm = row[arm_name]
            assert arm["colorable"] is exact_label
            if exact_label:
                assert arm["solution_verified"] is True
            else:
                assert arm["assignment"] is None


def test_req_verify_5117_artifact_gate_metrics_and_exp5103_reproduction(tmp_path: Path) -> None:
    """REQ-VERIFY-5117: artifact reports required fields and conservative harm reduction."""

    artifact = exp.write_artifact(
        root=tmp_path,
        run_date="20260701",
        duration_s=1.0,
        tests_run=["tests/python/test_experiment_5117_taco_harm_gated_scale_v469.py"],
    )
    payload = json.loads((tmp_path / exp.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert payload == artifact
    exp.validate_artifact(artifact)
    assert set(exp.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["experiment_id"] == exp.EXPERIMENT_ID
    assert artifact["milestone"] == exp.MILESTONE
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["exp5103_reproduction"]["matches_artifact"] is True
    assert artifact["wrong_label_count"] == 0
    assert artifact["taco_harm_gate_ready"] is True
    assert artifact["guarded_effort"]["total_effort_score"] < artifact["baseline_effort"]["total_effort_score"]
    assert artifact["harmful_instance_count_guarded"] < artifact["harmful_instance_count_unguarded"]
    assert artifact["flagged_adversarial"] is False
    assert artifact["tests_run"] == ["tests/python/test_experiment_5117_taco_harm_gated_scale_v469.py"]


def test_scenario_verify_5117_script_entrypoint_writes_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5117: CLI wrapper writes the terminal JSON artifact."""

    path = script_mod.main(
        root=tmp_path,
        date="20260701",
        duration_s=1.0,
        tests_run=["tests/python/test_experiment_5117_taco_harm_gated_scale_v469.py"],
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path == tmp_path / exp.RESULT_RELATIVE_PATH
    exp.validate_artifact(payload)
    assert payload["taco_harm_gate_ready"] is True


def test_deliverable_file_validates_for_scenario_verify_5117() -> None:
    """SCENARIO-VERIFY-5117: checked-in deliverable satisfies the terminal schema."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    exp.validate_artifact(artifact)
    assert artifact["experiment_id"] == exp.EXPERIMENT_ID
    assert artifact["wrong_label_count"] == 0
