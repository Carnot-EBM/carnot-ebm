"""Tests for Exp6432 held-shift process-restart CSL replication.

Spec refs: REQ-LEARN-6432, SCENARIO-LEARN-6432-GATES,
SCENARIO-LEARN-6432-PREREGISTRATION, SCENARIO-LEARN-6432-RESTARTS,
SCENARIO-LEARN-6432-ROWS, SCENARIO-LEARN-6432-ATTACKS,
SCENARIO-LEARN-6432-READY.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from carnot import experiment_6432_held_shift_process_restart_csl_replication as mod


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
        data_dir=tmp_path / "data",
        output_path=tmp_path / "experiment_6432.json",
    )


def test_req_learn_6432_spec_declares_fields_principles_and_scenarios() -> None:
    """REQ-LEARN-6432: OpenSpec owns the Exp6432 contract."""

    spec = SPEC.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-LEARN-6432") : spec.index("REQ-LEARN-6409")]
    normalized = " ".join(section.split())

    for marker in (
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        "SCENARIO-LEARN-6432-GATES",
        "SCENARIO-LEARN-6432-PREREGISTRATION",
        "SCENARIO-LEARN-6432-RESTARTS",
        "SCENARIO-LEARN-6432-ROWS",
        "SCENARIO-LEARN-6432-ATTACKS",
        "SCENARIO-LEARN-6432-READY",
        "held_shift_restart_csl_ready_score",
        "current_adversarial_flag_count",
    ):
        assert marker in section

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section

    for key, principle in mod.FIELD_PRINCIPLES.items():
        assert " ".join(principle.split()) in normalized, key


def test_scenario_learn_6432_gates_models_tokenizers_and_absence(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6432-GATES: clean gates and path absence gate the run."""

    artifact = _artifact(tmp_path)
    gates = artifact["exp6430_and_exp6431_gate_receipts"]
    specs = artifact["MODEL_SPECS"]
    helper = artifact["cached_sota_pair_receipts"]
    hashes = artifact["model_file_and_embedded_tokenizer_hashes"]
    absence = artifact["held_manifest_and_raw_output_path_absence_receipts"]
    policy = artifact["frozen_memory_policy_capacity_checker_model_prompt_and_head_hashes"]
    preconditions = artifact["preconditions_checked"]

    assert gates["all_gates_passed"] is True
    assert gates["exp6430"]["ready_score"] == 1.0
    assert gates["exp6431"]["ready_score"] == 1.0
    assert gates["exp6420"]["v552_defects_visible"] is True
    assert [row["hf_id"] for row in specs] == list(mod.MANDATED_MODEL_IDS)
    assert artifact["models_used"] == list(mod.MANDATED_MODEL_IDS)
    assert helper["helper"] == "cached_sota_pair"
    assert helper["all_mandated_models_returned"] is True
    assert set(helper["returned_hf_ids"]) == set(mod.MANDATED_MODEL_IDS)
    assert len(helper["calls"]) >= 2

    assert artifact["autotokenizer_usage_count"] == 0
    assert len(hashes) == 3
    assert all(row["tokenizer_loadable"] for row in hashes)
    assert all(row["tokenizer_method"] == mod.TOKENIZER_METHOD for row in hashes)
    assert all(row["bytes_in_use_event_count"] > 0 for row in hashes)

    assert absence["new_stream_paths_absent_before_generation"] is True
    assert absence["held_manifest_absent_before_run"] is True
    assert absence["artifact_absent_before_run"] is True
    assert absence["raw_output_dir_absent_before_run"] is True
    assert absence["expected_raw_output_paths_absent_before_run"] is True

    assert policy["selected_capacity"] == mod.SELECTED_CAPACITY
    assert policy["policy_frozen_before_held_outcomes"] is True
    assert policy["hidden_retuning_count"] == 0
    assert policy["exact_checkers"]["exact_feedback_checker"] is True
    assert policy["persisted_head_hash"] == mod.selected_exp6430_head_hash(REPO)
    assert preconditions["all_preconditions_passed"] is True
    assert preconditions["spec_contains_req"] is True
    assert artifact["blocked_reason"] == ""


def test_scenario_learn_6432_preregistration_and_fresh_raw_outputs(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6432-PREREGISTRATION: held plan freezes first."""

    artifact = _artifact(tmp_path)
    manifest = artifact[
        "held_manifest_path_hash_counts_balance_shift_restart_expiry_supersession_and_partition_seals"
    ]
    receipts = artifact["task_scoped_process_gpu_runner_and_raw_output_receipts"]
    freeze = artifact["per_event_unique_raw_output_and_pre_outcome_freeze_records"]

    assert manifest["event_count"] == mod.HELD_EVENT_COUNT
    assert manifest["session_count"] == mod.HELD_SESSION_COUNT
    assert manifest["held_factor_family_shift"]["frozen_before_held_outcomes"] is True
    assert manifest["model_balance"]["balanced"] is True
    assert manifest["process_restart_boundary_count"] == mod.HELD_SESSION_COUNT
    assert manifest["expiry_boundary_count"] >= 2
    assert manifest["supersession_boundary_count"] >= 2
    assert manifest["partition_seals"]["held_future"]["untouched_before_evaluation"] is True
    assert manifest["chronological_order_preserved"] is True
    assert manifest["development_pooling_count"] == 0

    assert receipts["generated_with_task_scoped_helper"] is True
    assert receipts["event_receipt_count"] == mod.HELD_EVENT_COUNT
    assert receipts["fresh_raw_output_count"] == mod.HELD_EVENT_COUNT
    assert receipts["raw_output_reuse_count"] == 0
    assert receipts["upstream_raw_hash_overlap_count"] == 0
    assert receipts["gpu_runner_receipts"]["runner_selected"] is True
    assert receipts["model_bytes_in_use_event_count"] == mod.HELD_EVENT_COUNT

    assert freeze["event_count"] == mod.HELD_EVENT_COUNT
    assert freeze["unique_event_id_count"] == mod.HELD_EVENT_COUNT
    assert freeze["unique_raw_output_hash_count"] == mod.HELD_EVENT_COUNT
    assert freeze["proposal_rows_frozen_before_outcome_count"] == mod.HELD_EVENT_COUNT
    assert freeze["future_outcomes_visible_before_proposal_freeze_count"] == 0
    assert artifact["raw_output_reuse_count"] == 0


def test_scenario_learn_6432_restart_recovery_uses_persisted_head(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6432-RESTARTS: new processes recover the head from disk."""

    artifact = _artifact(tmp_path)
    restarts = artifact["process_restart_and_persisted_head_recovery_receipts"]

    assert restarts["selected_capacity"] == mod.SELECTED_CAPACITY
    assert restarts["expected_persisted_head_hash"] == mod.selected_exp6430_head_hash(REPO)
    assert restarts["session_restart_count"] == mod.HELD_SESSION_COUNT
    assert restarts["unique_child_pid_count"] == mod.HELD_SESSION_COUNT
    assert restarts["all_recovered_heads_match"] is True
    assert restarts["restart_recovery_rate"] == 1.0
    assert restarts["no_in_memory_state_survived_except_hashed_schema"] is True
    assert all(row["child_pid"] != restarts["parent_pid"] for row in restarts["rows"])
    assert all(row["recovered_from_disk"] for row in restarts["rows"])


def test_scenario_learn_6432_rows_recompute_and_ready_score(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6432-ROWS: matched rows precede held aggregates."""

    artifact = _artifact(tmp_path)
    per_unit = artifact["per_unit_rows"]
    results = artifact[
        "per_arm_model_family_session_coverage_precision_selection_future_yield_transfer_retention_forgetting_negative_transfer_contamination_restart_latency_and_gpu_cost_results"
    ]
    uncertainty = artifact["effective_sample_sizes_and_uncertainty"]
    recomputed = artifact["aggregate_recomputation_receipts"]
    deltas = artifact["reported_vs_recomputed_deltas"]

    assert per_unit["written_before_aggregates"] is True
    assert per_unit["row_count"] == mod.HELD_EVENT_COUNT * len(mod.ARMS)
    assert {row["arm"] for row in per_unit["rows"]} == set(mod.ARMS)
    assert all(row["recorded_before_aggregate"] for row in per_unit["rows"])
    assert all(row["matched_work_units"] == mod.MATCHED_WORK_UNITS for row in per_unit["rows"])

    frozen = results["by_arm"][mod.FROZEN_ARM]
    selected = results["by_arm"][mod.SELECTED_ARM]
    assert frozen["future_exact_yield"] == 0.0
    assert selected["future_exact_yield"] > frozen["future_exact_yield"]
    assert selected["retention"] >= frozen["retention"]
    assert selected["negative_transfer"] == 0.0
    assert selected["contamination"] == 0.0
    assert selected["restart_recovery"] == 1.0
    assert results["empty_or_underpowered_cells_pooled"] is False
    assert results["cell_axes"] == ["arm", "model_family", "session_id"]

    assert artifact["held_future_exact_yield_delta"] > 0.0
    assert artifact["protected_retention_delta"] >= 0.0
    assert artifact["negative_transfer_delta"] <= mod.NEGATIVE_TRANSFER_BOUND
    assert artifact["contamination_propagation_rate"] == 0.0
    assert uncertainty["minimum_effective_sample_size"] >= mod.HELD_EVENT_COUNT
    assert all(len(row["future_exact_yield_ci95"]) == 2 for row in uncertainty["rows"])
    assert all(math.isfinite(float(row["future_exact_yield"])) for row in results["cells"])
    assert recomputed["all_recomputed_from_per_unit_rows"] is True
    assert deltas["all_zero"] is True
    assert artifact["held_shift_restart_csl_ready_score"] == 1.0


def test_scenario_learn_6432_attacks_oracle_and_schema_mutations_fail(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6432-ATTACKS: critical held attacks fail closed."""

    artifact = _artifact(tmp_path)
    attacks = artifact["attack_matrix"]

    assert {row["attack_id"] for row in attacks["rows"]} == set(mod.ATTACK_IDS)
    assert attacks["all_critical_attacks_fail_closed"] is True
    assert attacks["committed_attack_count"] == 0
    assert attacks["promoted_attack_count"] == 0
    assert artifact["cache_resurrection_count"] == 0
    assert artifact["hidden_retuning_count"] == 0
    assert artifact["protected_leakage_count"] == 0
    assert artifact["current_adversarial_flag_count"] == 0
    assert artifact["harm_underpowered_missing_and_flagged_cells"]["weak_cells_visible"] is True

    oracle = artifact["verifier_is_oracle"]
    assert oracle["value"] is True
    assert set(oracle["true_for"]) == {
        "exact_feedback_checker",
        "persistence_integrity_checker",
        "release_checker",
        "protected_retention_checker",
    }
    assert oracle["false_for"]["model_output"] is False
    assert oracle["false_for"]["memory"] is False
    assert artifact["honest_verdict"].startswith(("complete:", "success:", "passed:", "shipped:"))
    assert mod.validate_artifact(artifact) is True

    mutations = [
        ("required_fields", lambda data: data.pop("field_principles")),
        ("required_fields", lambda data: data.__setitem__("extra", True)),
        ("MODEL_SPECS", lambda data: data["MODEL_SPECS"].pop()),
        ("autotokenizer_usage_count", lambda data: data.__setitem__("autotokenizer_usage_count", 1)),
        ("held_manifest_and_raw_output_path_absence_receipts", lambda data: data["held_manifest_and_raw_output_path_absence_receipts"].__setitem__("new_stream_paths_absent_before_generation", False)),
        ("held_manifest", lambda data: data["held_manifest_path_hash_counts_balance_shift_restart_expiry_supersession_and_partition_seals"].__setitem__("development_pooling_count", 1)),
        ("frozen_memory_policy", lambda data: data["frozen_memory_policy_capacity_checker_model_prompt_and_head_hashes"].__setitem__("hidden_retuning_count", 1)),
        ("task_scoped_process_gpu_runner_and_raw_output_receipts", lambda data: data["task_scoped_process_gpu_runner_and_raw_output_receipts"].__setitem__("raw_output_reuse_count", 1)),
        ("per_unit_rows", lambda data: data["per_unit_rows"].__setitem__("written_before_aggregates", False)),
        ("per_event_unique_raw_output", lambda data: data["per_event_unique_raw_output_and_pre_outcome_freeze_records"].__setitem__("future_outcomes_visible_before_proposal_freeze_count", 1)),
        ("process_restart", lambda data: data["process_restart_and_persisted_head_recovery_receipts"].__setitem__("all_recovered_heads_match", False)),
        ("held_future_exact_yield_delta", lambda data: data.__setitem__("held_future_exact_yield_delta", 0.0)),
        ("protected_retention_delta", lambda data: data.__setitem__("protected_retention_delta", -0.1)),
        ("negative_transfer_delta", lambda data: data.__setitem__("negative_transfer_delta", mod.NEGATIVE_TRANSFER_BOUND + 0.1)),
        ("contamination_propagation_rate", lambda data: data.__setitem__("contamination_propagation_rate", 0.1)),
        ("reported_vs_recomputed_deltas", lambda data: data["reported_vs_recomputed_deltas"].__setitem__("all_zero", False)),
        ("raw_output_reuse_count", lambda data: data.__setitem__("raw_output_reuse_count", 1)),
        ("cache_resurrection_count", lambda data: data.__setitem__("cache_resurrection_count", 1)),
        ("hidden_retuning_count", lambda data: data.__setitem__("hidden_retuning_count", 1)),
        ("protected_leakage_count", lambda data: data.__setitem__("protected_leakage_count", 1)),
        ("attack_matrix", lambda data: data["attack_matrix"]["rows"][0].__setitem__("fail_closed", False)),
        ("current_adversarial_flag_count", lambda data: data.__setitem__("current_adversarial_flag_count", 1)),
        ("verifier_is_oracle", lambda data: data["verifier_is_oracle"]["false_for"].__setitem__("memory", True)),
        ("held_shift_restart_csl_ready_score", lambda data: data.__setitem__("held_shift_restart_csl_ready_score", 0.0)),
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


def test_req_learn_6432_helper_failures_and_stable_write(tmp_path: Path) -> None:
    """REQ-LEARN-6432: helper failures are explicit and writes are stable."""

    non_object = tmp_path / "list.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="json_object"):
        mod.read_json(non_object)

    outside = mod.path_receipt(Path("/tmp/nonexistent-exp6432-file"), relative_to=REPO)
    assert outside["present"] is False

    context = mod.load_context(REPO)
    bad_context = deepcopy(context)
    bad_context["exp6431"]["memory_interference_safety_ready_score"] = 0.0
    bad_context["exp6431"]["status"] = "complete_null"
    gates = mod.exp6430_and_exp6431_gate_receipts(REPO, bad_context)
    assert gates["all_gates_passed"] is False
    assert {"exp6431_not_ready", "exp6431_ready_score_not_one"} <= set(gates["blocked_reasons"])

    output = tmp_path / "written.json"
    data_dir = tmp_path / "write-data"
    artifact = mod.write_artifact(
        root=REPO,
        output_path=output,
        data_dir=data_dir,
        run_date=mod.RUN_DATE,
        duration_s=0.25,
        tests_run=_passing_tests(),
    )
    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert mod.validate_artifact(artifact) is True

    blocked = deepcopy(artifact)
    blocked["preconditions_checked"]["all_preconditions_passed"] = False
    assert mod.status(blocked) == "blocked_precondition"
    assert mod.honest_verdict(blocked).startswith("blocked:")

    nullish = deepcopy(artifact)
    nullish["held_future_exact_yield_delta"] = 0.0
    assert mod.ready_score(nullish) == 0.0
    assert mod.status(nullish) == "complete_null"
    assert mod.honest_verdict(nullish).startswith("complete_null:")

    bad_preconditions = mod.preconditions_checked(
        root=REPO,
        run_date="20260815",
        gates={"all_gates_passed": False},
        helper={"all_mandated_models_returned": False},
        model_hashes=[{"tokenizer_loadable": False, "autotokenizer_used": True}],
        absence={"new_stream_paths_absent_before_generation": False},
        manifest={"event_count": 1, "development_pooling_count": 1},
        policy={"policy_frozen_before_held_outcomes": False, "hidden_retuning_count": 1},
        task_receipts={
            "raw_output_reuse_count": 1,
            "upstream_raw_hash_overlap_count": 1,
            "gpu_runner_receipts": {"runner_selected": False},
        },
        restarts={"all_recovered_heads_match": False},
        protected_before={"ops/status.md": None},
        source_before={"module": None},
    )
    assert {
        "wrong_planning_date",
        "upstream_gates_failed",
        "cached_sota_pair_missing_model",
        "embedded_tokenizer_not_loadable",
        "autotokenizer_used",
        "held_paths_present_before_generation",
        "held_event_count_mismatch",
        "development_pooling_used",
        "policy_not_frozen",
        "hidden_retuning_present",
        "raw_output_reuse",
        "raw_output_not_fresh",
        "runner_not_selected",
        "persisted_head_recovery_failed",
        "protected_hash_missing",
        "source_hash_missing",
    } <= set(bad_preconditions["blocked_reasons"])
