"""Tests for Exp6162 prospective admission replication.

Spec refs: REQ-VERIFY-6162, REQ-VERIFY-6162-1, REQ-VERIFY-6162-2,
REQ-VERIFY-6162-3, REQ-VERIFY-6162-4, REQ-VERIFY-6162-5,
REQ-VERIFY-6162-6, REQ-VERIFY-6162-7, REQ-VERIFY-6162-8,
REQ-VERIFY-6162-9, REQ-VERIFY-6162-10, REQ-VERIFY-6162-11,
SCENARIO-VERIFY-6162-ONE-SHOT-MANIFEST,
SCENARIO-VERIFY-6162-PER-MODEL-GATES,
SCENARIO-VERIFY-6162-ATTACKS-RETIREMENT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

import scripts.adversarial_verify as adversarial_verify
from carnot import experiment_6162_prospective_admission_replication as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/verifiable-reasoning/spec.md"


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _run_artifact(tmp_path: Path, *, write: bool = False) -> dict[str, Any]:
    return mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        test_exit_codes=_passing_exit_codes(),
        duration_s=1.75,
        write=write,
    )


def _write_artifact(tmp_path: Path, artifact: dict[str, Any]) -> Path:
    path = tmp_path / mod.RESULT_RELATIVE_PATH.name
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return path


def test_req_6162_spec_declares_prospective_replication_fields() -> None:
    """REQ-VERIFY-6162: spec names the prospective held replication contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-VERIFY-6162") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-6162-1",
        "REQ-VERIFY-6162-2",
        "REQ-VERIFY-6162-3",
        "REQ-VERIFY-6162-4",
        "REQ-VERIFY-6162-5",
        "REQ-VERIFY-6162-6",
        "REQ-VERIFY-6162-7",
        "REQ-VERIFY-6162-8",
        "REQ-VERIFY-6162-9",
        "REQ-VERIFY-6162-10",
        "REQ-VERIFY-6162-11",
        "SCENARIO-VERIFY-6162-ONE-SHOT-MANIFEST",
        "SCENARIO-VERIFY-6162-PER-MODEL-GATES",
        "SCENARIO-VERIFY-6162-ATTACKS-RETIREMENT",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_6162_one_shot_guard_materializes_synthetic_outcomes_once() -> None:
    """SCENARIO-VERIFY-6162-ONE-SHOT-MANIFEST: held labels open exactly once."""

    rows_by_model = {
        "model-a": [
            {
                "event_id": "e1",
                "partition": "future_known",
                "current_outcome": "accepted",
                "unsafe_label": 1,
            },
            {
                "event_id": "e2",
                "partition": "shifted_family_held",
                "current_outcome": "rejected",
                "unsafe_label": 0,
            },
            {
                "event_id": "e3",
                "partition": "calibration",
                "current_outcome": "accepted",
                "unsafe_label": 1,
            },
            {
                "event_id": "e4",
                "partition": "ignored_partition",
                "current_outcome": "accepted",
                "unsafe_label": 1,
            },
            {
                "event_id": "e5",
                "partition": "future_known",
                "current_outcome": "accepted",
                "unsafe_label": 1,
            },
        ]
    }
    outcomes = {
        "e1": {"post_outcome": {"unsafe_label": 0}},
        "e2": {"post_outcome": {"unsafe_label": 1}},
        "e3": {"post_outcome": {"unsafe_label": 0}},
    }
    expected = {"future_known": ["e1", "e5"], "shifted_family_held": ["e2"]}
    guard = mod.HeldOutcomeAccessGuard(prior_receipt_seen=False)

    held, receipt = guard.unseal(
        rows_by_model,
        outcomes,
        expected_event_ids_by_partition=expected,
    )

    assert guard.access_count == 1
    assert receipt["held_access_count_before"] == 0
    assert receipt["held_access_count_after"] == 1
    assert receipt["future_known_label_read_count"] == 2
    assert receipt["shifted_family_held_label_read_count"] == 1
    assert receipt["calibration_label_read_count"] == 0
    assert receipt["held_label_read_count"] == 3
    assert receipt["model_row_label_mismatch_count"] == 2
    assert receipt["missing_outcome_event_ids"] == ["e5"]
    assert {row["partition"] for row in held["model-a"]} == set(mod.HELD_PARTITIONS)
    assert [row["unsafe_label"] for row in held["model-a"]] == [0, 1, 1]

    with pytest.raises(mod.HeldAccessError, match="exactly one"):
        guard.unseal(rows_by_model, outcomes, expected_event_ids_by_partition=expected)
    with pytest.raises(mod.HeldAccessError, match="prior held-access"):
        mod.HeldOutcomeAccessGuard(prior_receipt_seen=True).unseal(
            rows_by_model,
            outcomes,
            expected_event_ids_by_partition=expected,
        )


def test_req_6162_real_artifact_is_complete_positive_and_conjunctive(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-6162-3/4/5/6/7/8/10/11: both models pass every held gate."""

    artifact = _run_artifact(tmp_path, write=True)

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete_positive"
    assert artifact["honest_verdict"].startswith("complete_positive:")
    assert artifact["prospective_admission_replication_ready_score"] == 1.0
    assert artifact["retirement_triggered"] is False
    assert artifact["retirement_reason"] == "not_triggered_positive_replication"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert mod.validate_artifact(artifact) is True
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text()) == artifact

    access = artifact["first_and_only_held_access_receipt"]
    assert access["held_access_count_before"] == 0
    assert access["held_access_count_after"] == 1
    assert access["held_label_read_count"] == 288
    assert access["calibration_label_read_count"] == 0

    refits = artifact["selector_and_threshold_refit_counts"]
    assert all(value == 0 for value in refits["counts"].values())
    assert refits["all_zero"] is True

    conservation = artifact["row_conservation"]
    assert conservation["all_models_conserved"] is True
    for model_id in mod.MANDATED_MODEL_IDS:
        assert set(conservation["by_model"][model_id]) == set(mod.HELD_PARTITIONS)
        assert conservation["by_model"][model_id]["future_known"]["row_count"] == 64
        assert conservation["by_model"][model_id]["shifted_family_held"]["row_count"] == 80

    intervals = artifact[
        "per_model_future_known_and_shifted_decision_utility_intervals"
    ]["by_model"]
    metrics = artifact["brier_ece_and_descriptive_auroc_auprc_metrics"]["by_model"]
    gates = artifact["per_model_and_conjunctive_gate_matrix"]
    assert gates["conjunctive_pass"] is True
    for model_id in mod.MANDATED_MODEL_IDS:
        assert gates["by_model"][model_id]["model_pass"] is True
        for partition in mod.HELD_PARTITIONS:
            block = intervals[model_id][partition]
            assert block["decision_calibrated_minus_global"]["ci95"][0] > 0.0
            assert block["decision_calibrated_minus_exp6147_fixed"]["ci95"][0] > 0.0
            by_policy = metrics[model_id][partition]["policies"]
            selected = by_policy["decision_calibrated_task_energy"]
            assert selected["brier"] < by_policy["global_energy"]["brier"]
            assert selected["brier"] < by_policy["exp6147_fixed_task_aware"]["brier"]
            assert selected["auroc"] == 1.0
            assert selected["auprc"] == 1.0
            assert selected["action_counts"]["false_unsafe_acceptance"] == 0

    noninf = artifact["unsafe_admission_and_known_family_noninferiority_gates"]
    assert noninf["all_gates_pass"] is True
    assert all(
        gate["passed"]
        for model_gates in noninf["by_model"].values()
        for partition_gates in model_gates.values()
        for gate in partition_gates.values()
    )

    attacks = artifact["shortcut_poison_duplicate_boundary_and_order_attacks"]
    assert attacks["all_required_attacks_present"] is True
    assert attacks["any_attack_wins"] is False
    for name in mod.REQUIRED_ATTACKS:
        assert name in attacks["required_attacks"]


def test_req_6162_manifest_prior_access_validation_and_retirement(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-6162-1/2/9/10/11: blockers, validation, and retirement fail closed."""

    artifact = _run_artifact(tmp_path)

    prior_path = tmp_path / "prior.json"
    prior_path.write_text(
        json.dumps({"first_and_only_held_access_receipt": {"held_access_count_after": 1}}),
        encoding="utf-8",
    )
    prior = mod.run(
        result_path=prior_path,
        test_exit_codes=_passing_exit_codes(),
        duration_s=0.25,
    )
    assert prior["status"] == "blocked"
    assert prior["first_and_only_held_access_receipt"]["held_access_count_after"] == 0
    assert "prior_held_access_receipt" in prior["honest_verdict"]
    assert mod.validate_artifact(prior) is True

    exp6161 = mod.load_json(mod.REPO_ROOT / mod.EXP6161_RESULT_RELATIVE_PATH)
    exp6161["policy_manifest_path_hash_and_contents"]["contents_hash"] = mod.sha256_text(
        "mismatch"
    )
    mismatch = mod.run(
        result_path=tmp_path / "mismatch.json",
        exp6161_artifact=exp6161,
        test_exit_codes=_passing_exit_codes(),
        duration_s=0.25,
    )
    assert mismatch["status"] == "blocked"
    assert mismatch["first_and_only_held_access_receipt"]["held_access_count_after"] == 0
    assert "policy_manifest_mismatch" in mismatch["honest_verdict"]
    assert mod.validate_artifact(mismatch) is True

    bad_access = deepcopy(artifact)
    bad_access["first_and_only_held_access_receipt"]["held_access_count_after"] = 2
    bad_access["reproducibility_checksum"] = mod.reproducibility_checksum(bad_access)
    with pytest.raises(ValueError, match="first_and_only_held_access_receipt"):
        mod.validate_artifact(bad_access)

    bad_refit = deepcopy(artifact)
    bad_refit["selector_and_threshold_refit_counts"]["counts"]["threshold_refit_count"] = 1
    bad_refit["selector_and_threshold_refit_counts"]["all_zero"] = False
    bad_refit["prospective_admission_replication_ready_score"] = mod.ready_score(bad_refit)
    bad_refit["retirement_triggered"] = mod.retirement_triggered(bad_refit)
    bad_refit["retirement_reason"] = mod.retirement_reason(bad_refit)
    bad_refit["status"] = mod.status(bad_refit)
    bad_refit["honest_verdict"] = mod.honest_verdict(bad_refit)
    bad_refit["reproducibility_checksum"] = mod.reproducibility_checksum(bad_refit)
    assert bad_refit["prospective_admission_replication_ready_score"] == 0.0
    with pytest.raises(ValueError, match="selector_and_threshold_refit_counts"):
        mod.validate_artifact(bad_refit)

    retired = deepcopy(artifact)
    retired["per_model_and_conjunctive_gate_matrix"]["by_model"][
        mod.MANDATED_MODEL_IDS[0]
    ]["model_pass"] = False
    retired["per_model_and_conjunctive_gate_matrix"]["by_model"][
        mod.MANDATED_MODEL_IDS[0]
    ]["future_known"]["decision_utility_above_global"] = False
    retired["per_model_and_conjunctive_gate_matrix"]["conjunctive_pass"] = False
    retired["prospective_admission_replication_ready_score"] = mod.ready_score(retired)
    retired["retirement_triggered"] = mod.retirement_triggered(retired)
    retired["retirement_reason"] = mod.retirement_reason(retired)
    retired["status"] = mod.status(retired)
    retired["honest_verdict"] = mod.honest_verdict(retired)
    retired["reproducibility_checksum"] = mod.reproducibility_checksum(retired)
    assert retired["status"] == "retired"
    assert retired["retirement_triggered"] is True
    assert "repeated_decision_grade_null" in retired["retirement_reason"]
    assert mod.validate_artifact(retired) is True

    single_null = deepcopy(artifact)
    single_null["prior_failure_receipt"]["prior_decision_grade_null"] = False
    single_null["per_model_and_conjunctive_gate_matrix"]["by_model"][
        mod.MANDATED_MODEL_IDS[0]
    ]["model_pass"] = False
    single_null["per_model_and_conjunctive_gate_matrix"]["conjunctive_pass"] = False
    single_null["prospective_admission_replication_ready_score"] = mod.ready_score(
        single_null
    )
    single_null["retirement_triggered"] = mod.retirement_triggered(single_null)
    single_null["retirement_reason"] = mod.retirement_reason(single_null)
    single_null["status"] = mod.status(single_null)
    single_null["honest_verdict"] = mod.honest_verdict(single_null)
    single_null["reproducibility_checksum"] = mod.reproducibility_checksum(single_null)
    assert single_null["status"] == "complete_null"
    assert single_null["retirement_reason"] == "not_triggered_single_null_without_matching_prior"
    assert single_null["honest_verdict"].startswith("complete_null:")
    assert mod.validate_artifact(single_null) is True


def test_req_6162_schema_adversarial_and_helper_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-6162-2/8/9/10: schema and helper edges stay deterministic."""

    artifact = _run_artifact(tmp_path, write=True)
    report = adversarial_verify.verify_artifact(_write_artifact(tmp_path, artifact))
    kinds = {flag["kind"] for flag in report["flags"]}

    assert adversarial_verify._classify_inference_substrate(artifact)["kind"] == "no_llm"
    assert "DURATION_TOO_SHORT" not in kinds
    assert "METHODOLOGY_MISSING" not in kinds

    assert mod.load_json(tmp_path / "missing.json") == {}
    assert mod.load_jsonl(tmp_path / "missing.jsonl") == []
    assert mod._quantile([], 0.5) == 0.0
    assert mod._fixed_threshold_for_scores([]) == 0.0
    assert mod._partition_entries([], None, "future_known") == []
    assert mod._chronological_drift([], {}, {}) == {"window_count": 0, "windows": []}

    policy_configs = mod._policy_configs(
        mod.load_json(mod.REPO_ROOT / mod.EXP6147_RESULT_RELATIVE_PATH),
        mod.load_json(mod.REPO_ROOT / mod.EXP6161_RESULT_RELATIVE_PATH),
    )
    assert set(policy_configs) == set(mod.POLICY_NAMES)
    empty_metrics = mod._policy_metric([], "global_energy", policy_configs, {})
    assert empty_metrics["row_count"] == 0
    assert empty_metrics["utility_per_row"] == 0.0

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = mod.sha256_text("wrong")
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)

    bad_provenance_type = deepcopy(artifact)
    bad_provenance_type["field_provenance"] = []
    bad_provenance_type["reproducibility_checksum"] = mod.reproducibility_checksum(
        bad_provenance_type
    )
    with pytest.raises(ValueError, match="field_provenance"):
        mod.validate_artifact(bad_provenance_type)

    bad_provenance = deepcopy(artifact)
    bad_provenance["field_provenance"]["status"]["principle"] = "wrong"
    bad_provenance["reproducibility_checksum"] = mod.reproducibility_checksum(
        bad_provenance
    )
    with pytest.raises(ValueError, match="field_provenance:status"):
        mod.validate_artifact(bad_provenance)

    bad_ready = deepcopy(artifact)
    bad_ready["prospective_admission_replication_ready_score"] = 0.0
    bad_ready["reproducibility_checksum"] = mod.reproducibility_checksum(bad_ready)
    with pytest.raises(ValueError, match="prospective_admission_replication_ready_score"):
        mod.validate_artifact(bad_ready)

    bad_retirement_flag = deepcopy(artifact)
    bad_retirement_flag["retirement_triggered"] = True
    bad_retirement_flag["reproducibility_checksum"] = mod.reproducibility_checksum(
        bad_retirement_flag
    )
    with pytest.raises(ValueError, match="retirement_triggered"):
        mod.validate_artifact(bad_retirement_flag)

    bad_retirement_reason = deepcopy(artifact)
    bad_retirement_reason["retirement_reason"] = "wrong"
    bad_retirement_reason["reproducibility_checksum"] = mod.reproducibility_checksum(
        bad_retirement_reason
    )
    with pytest.raises(ValueError, match="retirement_reason"):
        mod.validate_artifact(bad_retirement_reason)

    bad_status = deepcopy(artifact)
    bad_status["status"] = "complete_null"
    bad_status["reproducibility_checksum"] = mod.reproducibility_checksum(bad_status)
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(bad_status)

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"] = "complete_positive: wrong"
    bad_verdict["reproducibility_checksum"] = mod.reproducibility_checksum(bad_verdict)
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad_verdict)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "cached_authentic_sota_rows_cpu_analysis"
    bad_substrate["prospective_admission_replication_ready_score"] = mod.ready_score(
        bad_substrate
    )
    bad_substrate["retirement_triggered"] = mod.retirement_triggered(bad_substrate)
    bad_substrate["retirement_reason"] = mod.retirement_reason(bad_substrate)
    bad_substrate["status"] = mod.status(bad_substrate)
    bad_substrate["honest_verdict"] = mod.honest_verdict(bad_substrate)
    bad_substrate["reproducibility_checksum"] = mod.reproducibility_checksum(
        bad_substrate
    )
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate)

    bad_verifier = deepcopy(artifact)
    bad_verifier["verifier_is_oracle"] = True
    bad_verifier["prospective_admission_replication_ready_score"] = mod.ready_score(
        bad_verifier
    )
    bad_verifier["retirement_triggered"] = mod.retirement_triggered(bad_verifier)
    bad_verifier["retirement_reason"] = mod.retirement_reason(bad_verifier)
    bad_verifier["status"] = mod.status(bad_verifier)
    bad_verifier["honest_verdict"] = mod.honest_verdict(bad_verifier)
    bad_verifier["reproducibility_checksum"] = mod.reproducibility_checksum(
        bad_verifier
    )
    with pytest.raises(ValueError, match="verifier_is_oracle"):
        mod.validate_artifact(bad_verifier)

    no_access = deepcopy(artifact)
    no_access["first_and_only_held_access_receipt"]["held_access_count_after"] = 0
    assert "first_and_only_held_access_receipt" in mod._blocked_reasons(no_access)
    no_access["prospective_admission_replication_ready_score"] = mod.ready_score(no_access)
    no_access["retirement_triggered"] = mod.retirement_triggered(no_access)
    no_access["retirement_reason"] = mod.retirement_reason(no_access)
    no_access["status"] = mod.status(no_access)
    no_access["honest_verdict"] = mod.honest_verdict(no_access)
    no_access["reproducibility_checksum"] = mod.reproducibility_checksum(no_access)
    with pytest.raises(ValueError, match="first_and_only_held_access_receipt"):
        mod.validate_artifact(no_access)

    reason_probe = deepcopy(artifact)
    reason_probe["first_and_only_held_access_receipt"]["held_access_count_after"] = 2
    reason_probe["selector_and_threshold_refit_counts"]["all_zero"] = False
    reason_probe["protected_files_unchanged"]["unchanged"] = False
    reason_probe["test_exit_codes"][mod.DEFAULT_TEST_COMMANDS[0]] = 1
    reason_probe["inference_substrate"] = "wrong"
    reason_probe["verifier_is_oracle"] = True
    reasons = set(mod._blocked_reasons(reason_probe))
    assert "first_and_only_held_access_receipt" in reasons
    assert "selector_and_threshold_refit_counts" in reasons
    assert "protected_files_changed" in reasons
    assert "test_command_failed" in reasons
    assert "inference_substrate" in reasons
    assert "verifier_is_oracle" in reasons

    missing = dict(artifact)
    missing.pop("status")
    with pytest.raises(ValueError, match="missing required"):
        mod.validate_artifact(missing)

    overridden = mod.run(
        result_path=tmp_path / "overridden.json",
        exp6147_artifact={},
        exp6148_artifact={},
        exp6159_artifact={},
        exp6160_artifact={},
        exp6161_artifact={},
        test_exit_codes=_passing_exit_codes(),
        duration_s=0.2,
    )
    assert overridden["status"] == "blocked"
    assert "exp6159_ready" in overridden["honest_verdict"]
    assert mod.validate_artifact(overridden) is True
