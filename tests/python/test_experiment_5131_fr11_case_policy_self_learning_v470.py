"""Tests for Exp 5131 FR-11 no-weight case-policy self-learning.

Spec refs: REQ-LEARN-5131,
SCENARIO-LEARN-5131-CASE-POLICY-NO-PROMOTE,
SCENARIO-LEARN-5131-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5131_fr11_case_policy_self_learning_v470 as exp
from scripts import experiment_5131_fr11_case_policy_self_learning_v470 as script_mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"
ARTIFACT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def test_req_learn_5131_spec_declares_exact_trace_case_policy_contract() -> None:
    """REQ-LEARN-5131: OpenSpec anchors no-weight exact-trace case policy."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5131") :]
    module_text = (REPO / exp.MODULE_RELATIVE_PATH).read_text(encoding="utf-8")

    for marker in (
        "REQ-LEARN-5131",
        "SCENARIO-LEARN-5131-CASE-POLICY-NO-PROMOTE",
        "SCENARIO-LEARN-5131-BLOCKED-PRECONDITION",
        exp.EXPERIMENT_ID,
        exp.RESULT_RELATIVE_PATH,
        exp.INFERENCE_SUBSTRATE,
        "no-learning, naive retrieval, case-policy, and case-policy-with-harm-gate",
    ):
        assert marker in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    assert "model weights" in module_text


def test_scenario_learn_5131_blocks_when_exp5130_not_ready(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5131-BLOCKED-PRECONDITION: closed source gate hard-blocks."""

    _write_json(
        tmp_path / exp.SOURCE_TRACE_RELATIVE_PATH,
        {
            "experiment_id": "exp5130-taco-sampler-heldout-scale-v470",
            "heldout_csp_trace_suite_ready": False,
            "honest_verdict": "complete_heldout_csp_trace_suite_not_ready",
        },
    )

    artifact = exp.build_artifact(
        root=tmp_path,
        run_date="20260701",
        duration_s=1.0,
        tests_run=["blocked-precondition-test"],
    )

    exp.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("blocked_exp5130_")
    assert artifact["source_trace_artifacts"][0]["ready"] is False
    assert artifact["trace_split_manifest"]["blocked"] is True
    assert artifact["promotion_attempted"] is False
    assert artifact["promotion_safe"] is False
    assert artifact["rollback_receipt"]["rollback_applied"] is True
    assert artifact["no_weight_update"] is True
    assert artifact["exact_solver_correctness_preserved"] is False


def test_scenario_learn_5131_split_policy_and_arm_evaluation_are_disjoint() -> None:
    """SCENARIO-LEARN-5131-CASE-POLICY-NO-PROMOTE: policy uses disjoint exact traces."""

    source = exp.load_source_trace_artifact(REPO)
    split = exp.build_trace_split(source)
    policy = exp.fit_case_policy(split)
    evaluation = exp.evaluate_policy_arms(split, policy)

    id_sets = {
        name: {row["instance_id"] for row in split[name]["rows"]}
        for name in ("learning", "validation", "heldout")
    }
    assert id_sets["learning"].isdisjoint(id_sets["validation"])
    assert id_sets["learning"].isdisjoint(id_sets["heldout"])
    assert id_sets["validation"].isdisjoint(id_sets["heldout"])
    assert split["manifest"]["strategy"] == "deterministic_instance_family_partition"
    assert len(split["manifest"]["split_hashes"]["heldout"]) > 20

    assert policy["policy_type"] == "nonparametric_contextual_case_policy"
    assert policy["no_weight_update"] is True
    assert policy["case_count"] == len(split["learning"]["rows"])
    assert policy["policy_hints"]
    assert all(hint["guarded"] is True for hint in policy["policy_hints"])
    assert all(hint["ttl_remaining"] > 0 for hint in policy["policy_hints"])
    assert any(hint["advantage_estimate"] != 0.0 for hint in policy["policy_hints"])

    assert set(evaluation["arms"]) == {
        "no_learning",
        "naive_retrieval",
        "case_policy",
        "case_policy_with_harm_gate",
    }
    assert evaluation["exact_solver_correctness_preserved"] is True
    assert evaluation["harmful_promotion_count"] >= 0
    assert evaluation["regret_telemetry"]["heldout_coverage"] >= 0.0


def test_scenario_learn_5131_writes_complete_no_promote_artifact(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5131-CASE-POLICY-NO-PROMOTE: gates write rollback artifact."""

    artifact = exp.write_artifact(
        root=tmp_path,
        run_date="20260701",
        duration_s=1.0,
        tests_run=["tests/python/test_experiment_5131_fr11_case_policy_self_learning_v470.py"],
    )
    payload = json.loads((tmp_path / exp.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert payload == artifact
    exp.validate_artifact(artifact)
    assert set(exp.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["experiment_id"] == exp.EXPERIMENT_ID
    assert artifact["milestone"] == exp.MILESTONE
    assert artifact["honest_verdict"].startswith("complete_fr11_case_policy_no_promote_")
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["no_weight_update"] is True
    assert artifact["promotion_attempted"] is True
    assert artifact["promotion_safe"] is False
    assert artifact["rollback_receipt"]["promoted_metadata_ids"] == []
    assert artifact["rollback_receipt"]["active_policy_after_rollback"] == "no_learning"
    assert artifact["exact_solver_correctness_preserved"] is True
    assert artifact["flagged_adversarial"] is False
    assert artifact["conductor_modified"] is False
    assert set(artifact["field_principles"]) >= set(exp.REQUIRED_ARTIFACT_FIELDS)


def test_req_learn_5131_validation_rejects_malformed_artifact(tmp_path: Path) -> None:
    """REQ-LEARN-5131: terminal schema validation rejects missing required fields."""

    artifact = exp.write_artifact(
        root=tmp_path,
        run_date="20260701",
        duration_s=1.0,
        tests_run=["tests/python/test_experiment_5131_fr11_case_policy_self_learning_v470.py"],
    )
    malformed = dict(artifact)
    malformed.pop("rollback_receipt")

    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact(malformed)


def test_scenario_learn_5131_script_entrypoint_writes_artifact(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5131-CASE-POLICY-NO-PROMOTE: CLI wrapper writes JSON."""

    path = script_mod.main(
        root=tmp_path,
        date="20260701",
        duration_s=1.0,
        tests_run=["tests/python/test_experiment_5131_fr11_case_policy_self_learning_v470.py"],
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path == tmp_path / exp.RESULT_RELATIVE_PATH
    exp.validate_artifact(payload)
    assert payload["experiment_id"] == exp.EXPERIMENT_ID
    assert payload["no_weight_update"] is True


def test_deliverable_file_validates_for_scenario_learn_5131() -> None:
    """SCENARIO-LEARN-5131-CASE-POLICY-NO-PROMOTE: checked-in artifact validates."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    exp.validate_artifact(artifact)
    assert artifact["experiment_id"] == exp.EXPERIMENT_ID
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["no_weight_update"] is True
