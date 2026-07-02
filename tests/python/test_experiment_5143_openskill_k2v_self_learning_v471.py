"""Tests for Exp 5143 OpenSkill/K2V no-weight self-learning.

Spec refs: REQ-LEARN-5143,
SCENARIO-LEARN-5143-PROMOTE-ANCHORS,
SCENARIO-LEARN-5143-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5143_openskill_k2v_self_learning_v471 as exp
from scripts import experiment_5143_openskill_k2v_self_learning_v471 as script_mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"
ARTIFACT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def test_req_learn_5143_spec_declares_anchor_learning_contract() -> None:
    """REQ-LEARN-5143: OpenSpec anchors exact verifier-anchor learning."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5143") :]

    for marker in (
        "REQ-LEARN-5143",
        "SCENARIO-LEARN-5143-PROMOTE-ANCHORS",
        "SCENARIO-LEARN-5143-BLOCKED-PRECONDITION",
        exp.EXPERIMENT_ID,
        exp.RESULT_RELATIVE_PATH,
        exp.INFERENCE_SUBSTRATE,
        "V470 case-policy baseline",
        "random anchor selection",
        "exact-constraint-only guard",
    ):
        assert marker in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_req_learn_5143_model_specs_are_mandated_local_ggufs_only() -> None:
    """REQ-LEARN-5143: MODEL_SPECS contain only mandated local GGUF IDs."""

    assert [item["hf_id"] for item in exp.MODEL_SPECS] == list(exp.MANDATED_GGUF_IDS)
    assert all(item["hf_id"].startswith("unsloth/") for item in exp.MODEL_SPECS)
    assert all(item["hf_id"].endswith("-GGUF") for item in exp.MODEL_SPECS)
    assert {item["usage"] for item in exp.MODEL_SPECS} == {
        "proposal_only_exact_validator_authority"
    }


def test_scenario_learn_5143_blocks_when_trace_suite_v2_not_ready(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5143-BLOCKED-PRECONDITION: closed Exp 5142 gate blocks."""

    _write_json(
        tmp_path / exp.EXP5142_RELATIVE_PATH,
        {
            "experiment_id": "exp5142-taco-harm-rootcause-scale-v471",
            "trace_suite_v2_ready": False,
            "honest_verdict": "complete_trace_suite_v2_not_ready",
        },
    )

    artifact = exp.build_artifact(
        root=tmp_path,
        run_date="20260702",
        duration_s=1.0,
        tests_run=["blocked-precondition-test"],
    )

    exp.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("blocked_exp5142_")
    assert artifact["source_trace_artifacts"][0]["trace_suite_v2_ready"] is False
    assert artifact["verification_anchor_manifest"]["blocked"] is True
    assert artifact["virtual_task_manifest"]["blocked"] is True
    assert artifact["promotion_safe"] is False
    assert artifact["rollback_receipt"]["rollback_applied"] is True
    assert artifact["no_weight_update"] is True
    assert artifact["wrong_label_count"] == 0


def test_scenario_learn_5143_anchor_and_virtual_manifests_are_exact_checked() -> None:
    """SCENARIO-LEARN-5143-PROMOTE-ANCHORS: anchors and virtual tasks are exact-checked."""

    source = exp.load_source_trace_artifact(REPO)
    split = exp.build_trace_split(source)
    anchor_manifest = exp.build_verification_anchor_manifest(split)
    virtual_manifest = exp.build_virtual_task_manifest(split)

    id_sets = {
        name: {row["instance_id"] for row in split[name]["rows"]}
        for name in ("anchor_source", "virtual_practice", "heldout", "nonforgetting")
    }
    for left, right in (
        ("anchor_source", "virtual_practice"),
        ("anchor_source", "heldout"),
        ("anchor_source", "nonforgetting"),
        ("virtual_practice", "heldout"),
        ("virtual_practice", "nonforgetting"),
        ("heldout", "nonforgetting"),
    ):
        assert id_sets[left].isdisjoint(id_sets[right])

    assert anchor_manifest["blocked"] is False
    assert anchor_manifest["anchor_count"] > 0
    assert anchor_manifest["uses_exact_labels"] is True
    assert all(anchor["exact_label_counts"] for anchor in anchor_manifest["anchors"])
    assert all(anchor["selected_policy_hint"] in exp.CANDIDATE_ARMS for anchor in anchor_manifest["anchors"])
    assert {anchor["proposal_source_model"] for anchor in anchor_manifest["anchors"]}.issubset(
        set(exp.MANDATED_GGUF_IDS)
    )

    assert virtual_manifest["blocked"] is False
    assert virtual_manifest["task_count"] == len(split["virtual_practice"]["rows"])
    assert virtual_manifest["hidden_label_read_for_generation"] is False
    assert all(task["exact_validator_receipt"]["validator_accepts_task"] is True for task in virtual_manifest["tasks"])
    assert all(task["source_hidden_label_read"] is False for task in virtual_manifest["tasks"])


def test_scenario_learn_5143_evaluation_promotes_only_under_strict_gates() -> None:
    """SCENARIO-LEARN-5143-PROMOTE-ANCHORS: held-out and nonforgetting gates pass."""

    source = exp.load_source_trace_artifact(REPO)
    split = exp.build_trace_split(source)
    anchors = exp.build_verification_anchor_manifest(split)
    evaluation = exp.evaluate_anchor_policy(split, anchors)

    assert set(evaluation["arm_comparison"]) == set(exp.EVALUATION_ARMS)
    assert evaluation["heldout_delta"] > 0.0
    assert evaluation["nonforgetting_delta"] >= 0.0
    assert evaluation["harmful_regime_delta"] >= 0.0
    assert evaluation["wrong_label_count"] == 0
    assert evaluation["promotion_safe"] is True
    assert evaluation["arm_comparison"]["learned_verifier_anchor_policy"]["total_effort"] < evaluation[
        "arm_comparison"
    ]["exact_constraint_only_guard"]["total_effort"]
    assert evaluation["arm_comparison"]["v470_case_policy_baseline"]["source_promotion_safe"] is False


def test_scenario_learn_5143_writes_success_artifact(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5143-PROMOTE-ANCHORS: artifact includes required schema fields."""

    artifact = exp.write_artifact(
        root=tmp_path,
        run_date="20260702",
        duration_s=1.0,
        tests_run=["tests/python/test_experiment_5143_openskill_k2v_self_learning_v471.py"],
    )
    payload = json.loads((tmp_path / exp.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert payload == artifact
    exp.validate_artifact(artifact)
    assert set(exp.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["experiment_id"] == exp.EXPERIMENT_ID
    assert artifact["milestone"] == exp.MILESTONE
    assert artifact["honest_verdict"].startswith("success_")
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["heldout_delta"] > 0.0
    assert artifact["nonforgetting_delta"] >= 0.0
    assert artifact["harmful_regime_delta"] >= 0.0
    assert artifact["wrong_label_count"] == 0
    assert artifact["promotion_safe"] is True
    assert artifact["rollback_receipt"]["rollback_available"] is True
    assert artifact["rollback_receipt"]["disable_learned_anchors"] == "set active_anchor_manifest_id to null"
    assert artifact["no_weight_update"] is True
    assert artifact["conductor_modified"] is False
    assert set(artifact["field_principles"]) >= set(exp.REQUIRED_ARTIFACT_FIELDS)


def test_req_learn_5143_validation_rejects_malformed_or_unsafe_payload(tmp_path: Path) -> None:
    """REQ-LEARN-5143: terminal validation rejects malformed promotion artifacts."""

    artifact = exp.write_artifact(root=tmp_path, run_date="20260702", duration_s=1.0, tests_run=["focused"])

    missing = dict(artifact)
    missing.pop("rollback_receipt")
    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact(missing)

    unsafe = dict(artifact)
    unsafe["promotion_safe"] = True
    unsafe["heldout_delta"] = 0.0
    with pytest.raises(ValueError, match="heldout_delta"):
        exp.validate_artifact(unsafe)

    corrupt = dict(artifact)
    corrupt["wrong_label_count"] = 1
    with pytest.raises(ValueError, match="wrong_label_count"):
        exp.validate_artifact(corrupt)


def test_scenario_learn_5143_script_entrypoint_writes_artifact(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5143-PROMOTE-ANCHORS: CLI wrapper writes JSON."""

    path = script_mod.main(
        root=tmp_path,
        date="20260702",
        duration_s=1.0,
        tests_run=["tests/python/test_experiment_5143_openskill_k2v_self_learning_v471.py"],
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path == tmp_path / exp.RESULT_RELATIVE_PATH
    exp.validate_artifact(payload)
    assert payload["promotion_safe"] is True
    assert payload["no_weight_update"] is True


def test_deliverable_file_validates_for_scenario_learn_5143() -> None:
    """SCENARIO-LEARN-5143-PROMOTE-ANCHORS: checked-in artifact validates."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    exp.validate_artifact(artifact)
    assert artifact["experiment_id"] == exp.EXPERIMENT_ID
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["promotion_safe"] is True
