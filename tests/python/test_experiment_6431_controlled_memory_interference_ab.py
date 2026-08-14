"""Tests for Exp6431 controlled memory-interference A/B.

Spec refs: REQ-LEARN-6431, SCENARIO-LEARN-6431-GATES,
SCENARIO-LEARN-6431-FREEZE, SCENARIO-LEARN-6431-PATHS,
SCENARIO-LEARN-6431-METRICS, SCENARIO-LEARN-6431-ATTACKS,
SCENARIO-LEARN-6431-READY.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6431_controlled_memory_interference_ab as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _passing_tests() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _artifact(tmp_path: Path) -> mod.JsonDict:
    return mod.build_artifact(
        root=REPO,
        run_date=mod.RUN_DATE,
        duration_s=0.25,
        tests_run=_passing_tests(),
        output_path=tmp_path / "experiment_6431.json",
    )


def test_req_learn_6431_spec_declares_fields_principles_and_scenarios() -> None:
    """REQ-LEARN-6431: OpenSpec owns the Exp6431 contract."""

    spec = SPEC.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-LEARN-6431") : spec.index("REQ-LEARN-6409")]
    normalized = " ".join(section.split())

    for marker in (
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        "SCENARIO-LEARN-6431-GATES",
        "SCENARIO-LEARN-6431-FREEZE",
        "SCENARIO-LEARN-6431-PATHS",
        "SCENARIO-LEARN-6431-METRICS",
        "SCENARIO-LEARN-6431-ATTACKS",
        "SCENARIO-LEARN-6431-READY",
        "memory_interference_safety_ready_score",
        "target exposure",
        "downstream-use failure",
    ):
        assert marker in section

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section

    for key, principle in mod.FIELD_PRINCIPLES.items():
        assert " ".join(principle.split()) in normalized, key


def test_scenario_learn_6431_gates_hashes_resources_and_seals(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6431-GATES: sealed inputs gate the run."""

    artifact = _artifact(tmp_path)
    gates = artifact["exp6430_gate_receipts"]
    hashes = artifact["upstream_row_manifest_policy_checker_and_head_hashes"]
    preconditions = artifact["preconditions_checked"]

    assert gates["all_gates_passed"] is True
    assert gates["exp6430"]["status"] == "complete_ready"
    assert gates["exp6430"]["ready_score"] == 1.0
    assert gates["exp6420"]["status"] == "complete_null"
    assert gates["exp6420"]["v552_defects_visible"] is True

    assert hashes["exp6430_artifact"]["present"] is True
    assert hashes["manifest_sidecar"]["present"] is True
    assert hashes["per_unit_row_hash"] == artifact["per_unit_rows"]["upstream_exp6430_row_hash"]
    assert hashes["authority_schema"]["valid"] is True
    assert hashes["memory_policy"]["capacity_frozen"] is True
    assert hashes["exact_checkers"]["exact_support_checker"] is True
    assert hashes["protected_future_seal"]["untouched_before_evaluation"] is True

    assert preconditions["all_preconditions_passed"] is True
    assert preconditions["spec_contains_req"] is True
    assert preconditions["cpu_ram_disk_checked"] is True
    assert preconditions["ram_total_bytes"] > 0
    assert artifact["blocked_reason"] == ""


def test_scenario_learn_6431_freezes_matrix_and_matched_arms(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6431-FREEZE: matrix and arms freeze first."""

    artifact = _artifact(tmp_path)
    matrix = artifact["preregistered_interference_matrix"]
    contract = artifact["preregistered_capacity_matched_arm_contract"]
    per_unit = artifact["per_unit_rows"]

    assert matrix["frozen_before_downstream_outcomes"] is True
    assert matrix["relationship_classes"] == list(mod.RELATIONSHIP_CLASSES)
    assert matrix["relationship_count"] == len(mod.RELATIONSHIP_CLASSES)
    assert matrix["post_outcome_relation_label_count"] == 0
    assert matrix["future_outcomes_used_for_labeling_count"] == 0
    assert all(row["label_frozen_before_outcome"] for row in matrix["rows"])

    assert contract["capacities"] == list(mod.CAPACITIES)
    assert contract["arms"] == list(mod.ARMS)
    assert contract["capacity_matched"] is True
    assert contract["matched_event_order_evidence_query_work_and_initial_head"] is True
    assert len({row["initial_head_hash"] for row in contract["by_capacity_arm"].values()}) == 1
    assert contract["held_outcomes_visible_before_contract"] is False

    assert per_unit["written_before_aggregates"] is True
    assert per_unit["row_count"] == mod.FUTURE_EVENT_COUNT * len(mod.CAPACITIES) * len(mod.ARMS)
    assert {row["arm"] for row in per_unit["rows"]} == set(mod.ARMS)
    assert {row["capacity"] for row in per_unit["rows"]} == set(mod.CAPACITIES)


def test_scenario_learn_6431_lifecycle_paths_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6431-PATHS: lifecycle controls use real memory paths."""

    artifact = _artifact(tmp_path)
    rows = artifact["per_unit_rows"]["rows"]
    authority_rows = [row for row in rows if row["arm"] == mod.AUTHORITY_AWARE_ARM]
    baseline_rows = [row for row in rows if row["arm"] == mod.BASELINE_ARM]

    assert all(row["transactional_memory_module"] == mod.TRANSACTIONAL_MEMORY_MODULE for row in rows)
    assert all(row["write_path"]["exact_support_checked"] for row in rows)
    assert all(row["retrieval_path"]["real_retrieve_called"] for row in rows)
    assert all(row["exact_retention_check_passed"] for row in authority_rows)

    invalid = {
        "contradiction",
        "source_authority_conflict",
        "temporal_invalidity",
        "poisoned_evidence",
    }
    invalid_authority = [row for row in authority_rows if row["relationship_class"] in invalid]
    assert invalid_authority
    assert all(row["write_path"]["commit_committed"] is False for row in invalid_authority)
    assert all(row["accepted_invalid_memory"] is False for row in authority_rows)

    supersession_rows = [
        row for row in authority_rows if row["relationship_class"] == "supersession"
    ]
    assert supersession_rows
    assert any(row["write_path"]["supersession_receipt"]["revoked_old"] for row in supersession_rows)
    assert any(row["write_path"]["commit_committed"] for row in supersession_rows)
    assert artifact["valid_higher_authority_update_count"] > 0

    expired_rows = [row for row in authority_rows if row["relationship_class"] == "temporal_invalidity"]
    poison_rows = [row for row in authority_rows if row["relationship_class"] == "poisoned_evidence"]
    assert all(row["write_path"]["expiry_receipt"]["expired"] for row in expired_rows)
    assert all("poison" in row["write_path"]["rejection_reasons"] for row in poison_rows)
    assert artifact["authority_spoof_accept_count"] == 0
    assert artifact["expired_or_superseded_accept_count"] == 0
    assert artifact["poisoned_evidence_accept_count"] == 0

    assert any(row["target_exposed"] is False for row in baseline_rows)
    assert any(row["downstream_used"] is False and row["target_exposed"] for row in baseline_rows)
    assert all(row["rollback_path"]["rollback_restored"] for row in authority_rows)


def test_scenario_learn_6431_metrics_recompute_by_relationship_and_family(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6431-METRICS: exposure and use are separate."""

    artifact = _artifact(tmp_path)
    results = artifact[
        "per_relationship_capacity_model_and_family_exposure_retrieval_use_coverage_precision_plasticity_stability_contamination_rollback_yield_latency_and_work_results"
    ]
    recomputed = artifact["aggregate_recomputation_receipts"]
    deltas = artifact["reported_vs_recomputed_deltas"]

    assert results["cell_axes"] == [
        "relationship_class",
        "capacity",
        "arm",
        "model_family",
        "factor_family",
    ]
    assert results["empty_or_underpowered_cells_pooled"] is False
    assert results["cell_count"] == len(results["cells"])
    assert results["underpowered_cell_count"] > 0
    assert results["by_arm"][mod.BASELINE_ARM]["exposure_failure_count"] > 0
    assert results["by_arm"][mod.BASELINE_ARM]["downstream_use_failure_count"] > 0
    assert results["by_arm"][mod.AUTHORITY_AWARE_ARM]["accepted_invalid_memory_count"] == 0
    assert results["by_arm"][mod.AUTHORITY_AWARE_ARM]["contamination_after_rollback"] == 0

    assert artifact["exposure_failure_count"] == results["by_arm"][mod.BASELINE_ARM]["exposure_failure_count"]
    assert artifact["downstream_use_failure_count"] == results["by_arm"][mod.BASELINE_ARM]["downstream_use_failure_count"]
    assert artifact["protected_stability_delta"]["value"] == 0.0
    assert artifact["contamination_after_rollback"] == 0
    assert recomputed["all_recomputed_from_per_unit_rows"] is True
    assert deltas["all_zero"] is True
    assert artifact["memory_interference_safety_ready_score"] == 1.0


def test_scenario_learn_6431_attacks_ready_score_and_schema_mutations_fail(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6431-ATTACKS: critical invalid memory fails closed."""

    artifact = _artifact(tmp_path)
    attacks = artifact["attack_matrix"]

    assert {row["attack_id"] for row in attacks["rows"]} == set(mod.ATTACK_IDS)
    assert attacks["all_critical_attacks_fail_closed"] is True
    assert attacks["authority_aware_invalid_accept_count"] == 0
    assert attacks["post_outcome_relation_label_count"] == 0
    assert artifact["memory_interference_safety_ready_score"] == 1.0
    assert artifact["harm_underpowered_missing_and_flagged_cells"]["weak_cells_visible"] is True

    oracle = artifact["verifier_is_oracle"]
    assert oracle["value"] is True
    assert set(oracle["true_for"]) == {
        "exact_support_checker",
        "authority_checker",
        "expiry_checker",
        "supersession_checker",
        "release_checker",
        "retention_checker",
    }
    assert oracle["false_for"]["retrieval_score"] is False
    assert oracle["false_for"]["memory_score"] is False
    assert artifact["honest_verdict"].startswith(("complete:", "success:", "passed:", "shipped:"))
    assert mod.validate_artifact(artifact) is True

    mutations = [
        ("required_fields", lambda data: data.pop("field_principles")),
        ("required_fields", lambda data: data.__setitem__("extra", True)),
        ("preregistered_interference_matrix", lambda data: data["preregistered_interference_matrix"].__setitem__("post_outcome_relation_label_count", 1)),
        ("preregistered_capacity_matched_arm_contract", lambda data: data["preregistered_capacity_matched_arm_contract"].__setitem__("capacity_matched", False)),
        ("per_unit_rows", lambda data: data["per_unit_rows"].__setitem__("written_before_aggregates", False)),
        ("authority_spoof_accept_count", lambda data: data.__setitem__("authority_spoof_accept_count", 1)),
        ("expired_or_superseded_accept_count", lambda data: data.__setitem__("expired_or_superseded_accept_count", 1)),
        ("poisoned_evidence_accept_count", lambda data: data.__setitem__("poisoned_evidence_accept_count", 1)),
        ("valid_higher_authority_update_count", lambda data: data.__setitem__("valid_higher_authority_update_count", 0)),
        ("protected_stability_delta", lambda data: data.__setitem__("protected_stability_delta", -0.1)),
        ("contamination_after_rollback", lambda data: data.__setitem__("contamination_after_rollback", 1)),
        ("reported_vs_recomputed_deltas", lambda data: data["reported_vs_recomputed_deltas"].__setitem__("all_zero", False)),
        ("attack_matrix", lambda data: data["attack_matrix"]["rows"][0].__setitem__("fail_closed", False)),
        ("verifier_is_oracle", lambda data: data["verifier_is_oracle"]["false_for"].__setitem__("memory_score", True)),
        ("memory_interference_safety_ready_score", lambda data: data.__setitem__("memory_interference_safety_ready_score", 0.0)),
        ("honest_verdict", lambda data: data.__setitem__("honest_verdict", "done")),
        ("reproducibility_checksum", lambda data: data.__setitem__("reproducibility_checksum", "sha256:bad")),
    ]

    for expected, mutate in mutations:
        bad = deepcopy(artifact)
        mutate(bad)
        if expected != "reproducibility_checksum":
            bad["reproducibility_checksum"] = mod.payload_checksum(bad)
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad)


def test_req_learn_6431_helper_failures_and_stable_write(tmp_path: Path) -> None:
    """REQ-LEARN-6431: helper failures are explicit and writes are stable."""

    non_object = tmp_path / "list.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="json_object"):
        mod.read_json(non_object)

    outside = mod.path_receipt(Path("/tmp/nonexistent-exp6431-file"), relative_to=REPO)
    assert outside["present"] is False

    context = mod.load_context(REPO)
    bad_context = deepcopy(context)
    bad_context["exp6430"]["prospective_write_once_csl_ready_score"] = 0.0
    bad_context["exp6430"]["status"] = "complete_null"
    bad_context["exp6430"]["reported_vs_recomputed_deltas"]["all_zero"] = False
    gates = mod.exp6430_gate_receipts(REPO, bad_context)
    assert gates["all_gates_passed"] is False
    assert {
        "exp6430_not_ready",
        "exp6430_ready_score_not_one",
        "exp6430_aggregates_do_not_recompute",
    }.issubset(set(gates["blocked_reasons"]))

    blocked_artifact = mod.build_artifact(
        root=REPO,
        run_date="19000101",
        duration_s=0.25,
        tests_run=_passing_tests(),
        output_path=tmp_path / "blocked.json",
    )
    assert blocked_artifact["status"] == "blocked_precondition"
    assert blocked_artifact["blocked_reason"]
    assert blocked_artifact["honest_verdict"].startswith("blocked:")

    timed_artifact = mod.build_artifact(
        root=REPO,
        run_date=mod.RUN_DATE,
        duration_s=None,
        tests_run=_passing_tests(),
        output_path=tmp_path / "timed.json",
    )
    assert timed_artifact["duration_s"] > 0.0001

    null_artifact = _artifact(tmp_path / "null")
    null_artifact["tests_run"]["all_passed"] = False
    assert mod.status(null_artifact) == "complete_null"
    assert mod.honest_verdict(null_artifact).startswith("complete_null:")

    output = tmp_path / "experiment_6431.json"
    written = mod.write_artifact(
        output_path=output,
        root=REPO,
        run_date=mod.RUN_DATE,
        duration_s=0.25,
        tests_run=_passing_tests(),
    )
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded == written
    assert loaded["reproducibility_checksum"] == mod.payload_checksum(loaded)
