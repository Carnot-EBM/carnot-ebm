"""Tests for Exp 5130 TACO sampler held-out CSP scaling.

Spec refs: REQ-SAMPLE-5130, SCENARIO-SAMPLE-5130.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5130_taco_sampler_heldout_scale_v470 as exp
from scripts import experiment_5130_taco_sampler_heldout_scale_v470 as script_mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/samplers/spec.md"
ARTIFACT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def test_req_sample_5130_spec_declares_heldout_contract() -> None:
    """REQ-SAMPLE-5130: OpenSpec declares the held-out CSP trace contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-SAMPLE-5130") :]

    for marker in (
        "REQ-SAMPLE-5130",
        "SCENARIO-SAMPLE-5130",
        "adaptive_2dpt_ready=true",
        exp.RESULT_RELATIVE_PATH,
        exp.EXPERIMENT_ID,
        exp.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_req_sample_5130_heldout_suite_is_disjoint_and_hashed() -> None:
    """REQ-SAMPLE-5130: held-out instances are deterministic and content-addressed."""

    first = exp.build_heldout_csp_suite()
    second = exp.build_heldout_csp_suite()
    hashes = exp.heldout_instance_hashes(first)
    tuning_hashes = exp.tuning_instance_hashes()

    assert first == second
    assert len(first) >= 8
    assert {case.instance.split for case in first} == {"heldout"}
    assert len({case.family for case in first}) >= 4
    assert {case.instance.expected_colorable for case in first} == {False, True}
    assert len({item["sha256"] for item in hashes}) == len(first)
    assert all(len(item["sha256"]) == 64 for item in hashes)
    assert {item["sha256"] for item in hashes}.isdisjoint(tuning_hashes)


def test_req_sample_5130_exact_authority_preserves_all_policy_labels() -> None:
    """REQ-SAMPLE-5130: no heuristic label is counted without exact agreement."""

    solver = exp.ExactCspSolver()
    sampler_features = exp.load_exp5129_sampler_features(REPO)

    for case in exp.build_heldout_csp_suite():
        row = exp.evaluate_heldout_case(case, solver=solver, sampler_features=sampler_features)
        exact_label = row["exact_label"]["colorable"]

        assert row["exact_enumerator"]["agrees_with_solver"] is True
        assert row["wrong_label"] is False
        assert row["heuristic_only_answer_counted"] is False
        for arm_name in ("baseline", "unguarded", "guarded", "sampler_feature"):
            arm = row[arm_name]
            assert arm["colorable"] is exact_label
            assert arm["exact_authority_agrees"] is True
            if exact_label:
                assert arm["solution_verified"] is True
            else:
                assert arm["assignment"] is None


def test_req_sample_5130_artifact_schema_ready_gate_and_metrics(tmp_path: Path) -> None:
    """REQ-SAMPLE-5130: artifact emits required fields and FR-11 readiness."""

    artifact = exp.write_artifact(
        root=tmp_path,
        run_date="20260701",
        duration_s=1.0,
        tests_run=["tests/python/test_experiment_5130_taco_sampler_heldout_scale_v470.py"],
    )
    payload = json.loads((tmp_path / exp.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert payload == artifact
    exp.validate_artifact(artifact)
    assert set(exp.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["experiment_id"] == exp.EXPERIMENT_ID
    assert artifact["milestone"] == exp.MILESTONE
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["exp5117_baseline_loaded"] is True
    assert artifact["exp5129_sampler_features_loaded"] is True
    assert artifact["wrong_label_count"] == 0
    assert artifact["timeout_rate"] == 0.0
    assert artifact["heldout_csp_trace_suite_ready"] is True
    assert artifact["flagged_adversarial"] is False
    assert artifact["conductor_modified"] is False
    assert artifact["harmful_instance_count_guarded"] <= artifact["harmful_instance_count_unguarded"]
    assert len(artifact["per_family_results"]) >= 4


def test_req_sample_5130_hard_blocks_when_exp5129_gate_is_closed(tmp_path: Path) -> None:
    """REQ-SAMPLE-5130: closed Exp 5129 readiness gate writes a blocked artifact."""

    gate_path = tmp_path / exp.EXP5129_RELATIVE_PATH
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(
        json.dumps({"experiment_id": "exp5129-hubo-adaptive-2dpt-v470", "adaptive_2dpt_ready": False}),
        encoding="utf-8",
    )

    artifact = exp.build_artifact(
        root=tmp_path,
        run_date="20260701",
        duration_s=1.0,
        tests_run=["blocked-gate-test"],
    )

    exp.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["failure_type"] == "gate_blocked"
    assert artifact["exp5129_sampler_features_loaded"] is False
    assert artifact["heldout_csp_trace_suite_ready"] is False
    assert artifact["instance_count"] == 0


def test_req_sample_5130_validation_rejects_missing_required_field(tmp_path: Path) -> None:
    """REQ-SAMPLE-5130: terminal validation rejects malformed payloads."""

    artifact = exp.write_artifact(
        root=tmp_path,
        run_date="20260701",
        duration_s=1.0,
        tests_run=["tests/python/test_experiment_5130_taco_sampler_heldout_scale_v470.py"],
    )
    malformed = dict(artifact)
    malformed.pop("heldout_instance_hashes")

    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact(malformed)


def test_scenario_sample_5130_script_entrypoint_writes_artifact(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-5130: CLI wrapper writes the terminal JSON artifact."""

    path = script_mod.main(
        root=tmp_path,
        date="20260701",
        duration_s=1.0,
        tests_run=["tests/python/test_experiment_5130_taco_sampler_heldout_scale_v470.py"],
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path == tmp_path / exp.RESULT_RELATIVE_PATH
    exp.validate_artifact(payload)
    assert payload["heldout_csp_trace_suite_ready"] is True
    assert payload["conductor_modified"] is False


def test_deliverable_file_validates_for_scenario_sample_5130() -> None:
    """SCENARIO-SAMPLE-5130: checked-in deliverable satisfies the terminal schema."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    exp.validate_artifact(artifact)
    assert artifact["experiment_id"] == exp.EXPERIMENT_ID
    assert artifact["heldout_csp_trace_suite_ready"] is True
