"""Tests for Exp 5142 TACO/CSP harm root-cause scaling.

Spec refs: REQ-SAMPLE-5142, SCENARIO-SAMPLE-5142.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5142_taco_harm_rootcause_scale_v471 as exp
from scripts import experiment_5142_taco_harm_rootcause_scale_v471 as script_mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/samplers/spec.md"
ARTIFACT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def test_req_sample_5142_spec_declares_scaled_harm_gate_contract() -> None:
    """REQ-SAMPLE-5142: OpenSpec declares the V471 trace-suite contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("### REQ-SAMPLE-5142")
    section = spec[start:]

    assert "SCENARIO-SAMPLE-5142" in section
    assert "at least 80" in section
    assert "near-tie score" in section
    assert "sampler entropy" in section
    assert exp.RESULT_RELATIVE_PATH in section
    assert exp.EXPERIMENT_ID in section
    assert exp.INFERENCE_SUBSTRATE in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_req_sample_5142_suite_scales_across_families_and_bands() -> None:
    """REQ-SAMPLE-5142: the suite has >=80 exact-checkable held-out cases."""

    suite = exp.build_scaled_csp_suite()
    hashes = exp.heldout_instance_hashes(suite)
    families = {case.family for case in suite}
    density_bands = {case.instance.density_bucket for case in suite}
    frustration_bands = {case.instance.frustration for case in suite}

    assert len(suite) >= 80
    assert all(case.instance.split == "heldout" for case in suite)
    assert len(families) >= 6
    assert {"low", "medium", "high"}.issubset(density_bands)
    assert {"low", "medium", "high"}.issubset(frustration_bands)
    assert {case.instance.expected_colorable for case in suite} == {False, True}
    assert len({item["sha256"] for item in hashes}) == len(suite)
    assert all(len(item["sha256"]) == 64 for item in hashes)


def test_req_sample_5142_reproduces_exp5130_effort_measurements() -> None:
    """REQ-SAMPLE-5142: Exp 5130 baseline, guarded, and sampler metrics reproduce."""

    reproduction = exp.load_and_reproduce_exp5130(REPO)

    assert reproduction["loaded"] is True
    assert reproduction["reproduction_matches"] is True
    assert reproduction["source_path"] == exp.EXP5130_RELATIVE_PATH
    assert reproduction["measured"]["baseline_effort"]["total_effort_score"] == reproduction["artifact"]["baseline_effort"]["total_effort_score"]
    assert reproduction["measured"]["guarded_effort"]["total_effort_score"] == reproduction["artifact"]["guarded_effort"]["total_effort_score"]
    assert reproduction["measured"]["sampler_feature_effort"]["total_effort_score"] == reproduction["artifact"]["sampler_feature_effort"]["total_effort_score"]
    assert reproduction["measured"]["wrong_label_count"] == 0


def test_req_sample_5142_evaluation_clusters_harm_and_repairs_gate() -> None:
    """REQ-SAMPLE-5142: harmful regimes are clustered and sampler guidance abstains."""

    evaluation = exp.evaluate_scaled_suite(root=REPO)
    ablations = evaluation["ablation_results"]
    repaired_gate = evaluation["repaired_harm_gate"]

    assert evaluation["instance_count"] >= 80
    assert evaluation["wrong_label_count"] == 0
    assert evaluation["label_disagreements"] == []
    assert evaluation["original_harmful_instance_count_guarded"] > 0
    assert evaluation["harmful_instance_count_guarded"] < evaluation["original_harmful_instance_count_guarded"]
    assert evaluation["average_effort_reduction_ratio_guarded"] > 0.0
    assert repaired_gate["sampler_feature_policy"] == "abstain_on_identified_harm_regimes"
    assert repaired_gate["rejected_sampler_feature_count"] > 0
    assert repaired_gate["accepted_sampler_feature_count"] == 0
    assert "sampler_feature" not in repaired_gate["selected_arm_counts"]
    assert all(item["feature_summary"]["instance_count"] > 0 for item in evaluation["harmful_instance_root_causes"])
    assert all("constraint_density" in item["feature_summary"] for item in evaluation["harmful_instance_root_causes"])
    assert all("sampler_entropy" in item["feature_summary"] for item in evaluation["harmful_instance_root_causes"])
    assert ablations["repaired_guard"]["harmful_instance_count"] == evaluation["harmful_instance_count_guarded"]
    assert ablations["sampler_feature_raw"]["harmful_instance_count"] >= ablations["repaired_guard"]["harmful_instance_count"]


def test_req_sample_5142_artifact_schema_and_ready_gate(tmp_path: Path) -> None:
    """REQ-SAMPLE-5142: artifact emits required fields and readiness gate."""

    artifact = exp.write_artifact(
        root=tmp_path,
        run_date="20260702",
        duration_s=1.0,
        tests_run=["tests/python/test_experiment_5142_taco_harm_rootcause_scale_v471.py"],
    )
    payload = json.loads((tmp_path / exp.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert payload == artifact
    exp.validate_artifact(artifact)
    assert set(exp.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["experiment_id"] == exp.EXPERIMENT_ID
    assert artifact["milestone"] == exp.MILESTONE
    assert artifact["honest_verdict"] == exp.READY_VERDICT
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] == pytest.approx(1.0)
    assert artifact["exp5130_baseline_loaded"] is True
    assert artifact["instance_count"] >= 80
    assert len(artifact["task_families"]) >= 6
    assert artifact["wrong_label_count"] == 0
    assert artifact["harmful_instance_count_guarded"] < artifact["original_harmful_instance_count_guarded"]
    assert artifact["trace_suite_v2_ready"] is True
    assert artifact["conductor_modified"] is False


def test_req_sample_5142_validation_rejects_unsafe_or_malformed_payload(tmp_path: Path) -> None:
    """REQ-SAMPLE-5142: validation rejects malformed and unsafe terminal payloads."""

    artifact = exp.write_artifact(root=tmp_path, run_date="20260702", duration_s=1.0, tests_run=["focused"])

    missing = dict(artifact)
    missing.pop("repaired_harm_gate")
    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact(missing)

    unsafe = dict(artifact)
    unsafe["conductor_modified"] = True
    with pytest.raises(ValueError, match="conductor_modified"):
        exp.validate_artifact(unsafe)

    wrong = dict(artifact)
    wrong["wrong_label_count"] = 1
    with pytest.raises(ValueError, match="wrong_label_count"):
        exp.validate_artifact(wrong)


def test_scenario_sample_5142_script_entrypoint_writes_artifact(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-5142: CLI wrapper writes the terminal JSON artifact."""

    path = script_mod.main(
        root=tmp_path,
        date="20260702",
        duration_s=1.0,
        tests_run=["tests/python/test_experiment_5142_taco_harm_rootcause_scale_v471.py"],
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path == tmp_path / exp.RESULT_RELATIVE_PATH
    exp.validate_artifact(payload)
    assert payload["trace_suite_v2_ready"] is True
    assert payload["conductor_modified"] is False


def test_deliverable_file_validates_for_scenario_sample_5142() -> None:
    """SCENARIO-SAMPLE-5142: checked-in deliverable satisfies the terminal schema."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    exp.validate_artifact(artifact)
    assert artifact["experiment_id"] == exp.EXPERIMENT_ID
    assert artifact["trace_suite_v2_ready"] is True
