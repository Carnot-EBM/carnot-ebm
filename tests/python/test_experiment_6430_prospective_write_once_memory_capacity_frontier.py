"""Tests for Exp6430 prospective write-once memory capacity frontier.

Spec refs: REQ-LEARN-6430, SCENARIO-LEARN-6430-GATES,
SCENARIO-LEARN-6430-STREAM, SCENARIO-LEARN-6430-CAPACITY,
SCENARIO-LEARN-6430-FRONTIER, SCENARIO-LEARN-6430-ATTACKS,
SCENARIO-LEARN-6430-READY.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from carnot import experiment_6430_prospective_write_once_memory_capacity_frontier as mod


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
        output_path=tmp_path / "experiment_6430.json",
    )


def test_req_learn_6430_spec_declares_fields_principles_and_scenarios() -> None:
    """REQ-LEARN-6430: OpenSpec owns the Exp6430 contract."""

    spec = SPEC.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-LEARN-6430") : spec.index("REQ-LEARN-6409")]
    normalized = " ".join(section.split())

    for marker in (
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        "SCENARIO-LEARN-6430-GATES",
        "SCENARIO-LEARN-6430-STREAM",
        "SCENARIO-LEARN-6430-CAPACITY",
        "SCENARIO-LEARN-6430-FRONTIER",
        "SCENARIO-LEARN-6430-ATTACKS",
        "SCENARIO-LEARN-6430-READY",
        "prospective_write_once_csl_ready_score",
        "current_adversarial_flag_count",
    ):
        assert marker in section

    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section or field.startswith(
            ("gate:", "capacity:", "write:", "frontier:")
        )
        assert " ".join(principle.split()) in normalized


def test_scenario_learn_6430_gates_models_tokenizers_and_path_absence(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6430-GATES: receipts and path absence gate the run."""

    artifact = _artifact(tmp_path)
    gates = artifact["exp6428_gate_receipts"]
    specs = artifact["MODEL_SPECS"]
    helper = artifact["cached_sota_pair_receipts"]
    hashes = artifact["model_file_and_embedded_tokenizer_hashes"]
    absence = artifact["manifest_absence_before_run_receipt"]
    preconditions = artifact["preconditions_checked"]

    assert gates["all_gates_passed"] is True
    assert gates["exp6428"]["ready_score"] == 1.0
    assert gates["exp6426"]["ready_score"] == 1.0
    assert gates["exp6420"]["ready_score"] == 0.0
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
    assert all(row["autotokenizer_used"] is False for row in specs)

    assert absence["manifest_absent_before_run"] is True
    assert absence["artifact_absent_before_run"] is True
    assert absence["new_stream_paths_absent_before_generation"] is True
    assert preconditions["all_preconditions_passed"] is True
    assert preconditions["spec_contains_req"] is True
    assert artifact["blocked_reason"] == ""


def test_scenario_learn_6430_stream_freezes_unique_raw_outputs(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6430-STREAM: events and raw outputs are fresh."""

    artifact = _artifact(tmp_path)
    manifest = artifact[
        "chronological_manifest_path_hash_event_session_drift_restart_expiry_supersession_counts_and_partition_seals"
    ]
    receipts = artifact["task_scoped_process_gpu_runner_and_raw_output_receipts"]
    freeze = artifact["per_event_unique_raw_output_and_pre_outcome_freeze_records"]

    assert manifest["event_count"] >= 120
    assert manifest["session_count"] == 5
    assert manifest["drift_regime_count"] == 3
    assert manifest["model_family_count"] == 3
    assert manifest["process_restart_boundary_count"] >= 5
    assert manifest["expiry_boundary_count"] >= 3
    assert manifest["supersession_boundary_count"] >= 3
    assert manifest["partition_seals"]["future"]["untouched_before_evaluation"] is True
    assert manifest["chronological_order_preserved"] is True

    assert receipts["generated_with_task_scoped_helper"] is True
    assert receipts["event_receipt_count"] == manifest["event_count"]
    assert receipts["fresh_raw_output_count"] == manifest["event_count"]
    assert receipts["raw_output_reuse_count"] == 0
    assert receipts["gpu_runner_receipts"]["runner_selected"] is True

    assert freeze["event_count"] == manifest["event_count"]
    assert freeze["unique_event_id_count"] == manifest["event_count"]
    assert freeze["unique_raw_output_hash_count"] == manifest["event_count"]
    assert freeze["proposal_rows_frozen_before_outcome_count"] == manifest["event_count"]
    assert freeze["future_outcomes_visible_before_proposal_freeze_count"] == 0
    assert artifact["raw_output_reuse_count"] == 0


def test_scenario_learn_6430_capacity_transitions_follow_exact_feedback(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6430-CAPACITY: exact feedback controls writes."""

    artifact = _artifact(tmp_path)
    contract = artifact["preregistered_capacity_and_arm_contract"]
    feedback = artifact["exact_feedback_receipts"]
    history = artifact["memory_schema_head_and_transition_history"]
    counts = artifact["commit_reject_quarantine_defer_evict_expire_and_supersede_counts"]

    assert contract["capacities"] == list(mod.CAPACITIES)
    assert contract["capacities_frozen_before_outcomes"] is True
    assert contract["arms"] == [mod.arm_for_capacity(capacity) for capacity in mod.CAPACITIES]
    assert len({row["initial_head_hash"] for row in contract["by_capacity"].values()}) == 1
    assert contract["matched_event_order_model_calls_prompts_tokens_checker_calls_consumer_work"] is True

    assert feedback["feedback_count"] == mod.EVENT_COUNT * len(mod.CAPACITIES)
    assert feedback["exact_feedback_before_write_count"] == feedback["feedback_count"]
    assert feedback["release_check_failures"] == 0
    assert feedback["protected_retention_failures"] == 0
    assert feedback["verifier_is_oracle_for_exact_checks"] is True

    assert history["schema_version"] == mod.MEMORY_SCHEMA_VERSION
    assert history["head_transition_count"] == len(history["transitions"])
    assert history["all_transitions_after_exact_feedback"] is True
    assert history["all_active_counts_within_capacity"] is True
    assert history["by_capacity"]["0"]["final_active_count"] == 0
    assert history["by_capacity"]["32"]["final_active_count"] <= 32

    for disposition in mod.DISPOSITIONS:
        assert disposition in counts["total"]
    assert counts["by_capacity"]["0"]["Commit"] == 0
    assert counts["by_capacity"]["4"]["Evict"] > 0
    assert counts["by_capacity"]["8"]["Expire"] > 0
    assert counts["by_capacity"]["16"]["Supersede"] > 0


def test_scenario_learn_6430_frontier_recomputes_and_selects_without_retuning(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6430-FRONTIER: per-unit rows precede aggregates."""

    artifact = _artifact(tmp_path)
    per_unit = artifact["per_unit_rows"]
    results = artifact[
        "per_capacity_coverage_precision_selection_future_yield_transfer_retention_forgetting_contamination_growth_eviction_restart_and_cost_results"
    ]
    frontier = artifact["capacity_utility_frontier"]
    uncertainty = artifact["effective_sample_sizes_and_uncertainty"]
    selection = artifact["best_capacity_selected_without_held_tuning"]

    assert per_unit["written_before_aggregates"] is True
    assert per_unit["row_count"] == mod.FUTURE_EVENT_COUNT * len(mod.CAPACITIES)
    assert len(per_unit["rows"]) == per_unit["row_count"]
    assert {row["capacity"] for row in per_unit["rows"]} == set(mod.CAPACITIES)

    assert results["by_capacity"]["0"]["future_exact_yield"] == 0.0
    assert results["by_capacity"]["16"]["future_exact_yield"] > results["by_capacity"]["0"]["future_exact_yield"]
    assert results["by_capacity"]["16"]["write_precision"] >= results["by_capacity"]["0"]["write_precision"]
    assert results["by_capacity"]["16"]["retention"] >= results["by_capacity"]["0"]["retention"]
    assert results["by_capacity"]["16"]["contamination"] == 0.0
    assert all(math.isfinite(float(row["future_exact_yield"])) for row in frontier["rows"])
    assert frontier["counts"]["capacity_count"] == len(mod.CAPACITIES)
    assert frontier["best_nonzero_capacity"] == selection["selected_capacity"]
    assert frontier["capacity_selected_after_held_outcomes"] is False

    assert uncertainty["minimum_effective_sample_size"] >= mod.FUTURE_EVENT_COUNT
    assert all(len(row["future_exact_yield_ci95"]) == 2 for row in uncertainty["rows"])
    assert selection["selection_rule_frozen_before_held_outcomes"] is True
    assert selection["selected_capacity"] in mod.CAPACITIES
    assert artifact["aggregate_recomputation_receipts"]["all_recomputed_from_per_unit_rows"] is True
    assert artifact["reported_vs_recomputed_deltas"]["all_zero"] is True
    assert artifact["prospective_write_once_csl_ready_score"] == 1.0


def test_scenario_learn_6430_attacks_ready_score_and_schema_mutations_fail(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6430-ATTACKS: critical attacks fail closed."""

    artifact = _artifact(tmp_path)
    attacks = artifact["attack_matrix"]

    assert {row["attack_id"] for row in attacks["rows"]} == set(mod.ATTACK_IDS)
    assert attacks["all_critical_attacks_fail_closed"] is True
    assert attacks["committed_attack_count"] == 0
    assert artifact["cache_resurrection_count"] == 0
    assert artifact["same_step_write_count"] == 0
    assert artifact["contamination_propagation_rate"] == 0.0
    assert artifact["exact_veto_override_count"] == 0
    assert artifact["protected_leakage_count"] == 0
    assert artifact["current_adversarial_flag_count"] == 0

    oracle = artifact["verifier_is_oracle"]
    assert oracle["value"] is True
    assert set(oracle["true_for"]) == {
        "exact_feedback_checker",
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
        ("raw_output_reuse_count", lambda data: data.__setitem__("raw_output_reuse_count", 1)),
        ("cache_resurrection_count", lambda data: data.__setitem__("cache_resurrection_count", 1)),
        ("same_step_write_count", lambda data: data.__setitem__("same_step_write_count", 1)),
        ("contamination_propagation_rate", lambda data: data.__setitem__("contamination_propagation_rate", 0.1)),
        ("exact_veto_override_count", lambda data: data.__setitem__("exact_veto_override_count", 1)),
        ("protected_leakage_count", lambda data: data.__setitem__("protected_leakage_count", 1)),
        ("current_adversarial_flag_count", lambda data: data.__setitem__("current_adversarial_flag_count", 1)),
        ("per_unit_rows", lambda data: data["per_unit_rows"].__setitem__("written_before_aggregates", False)),
        (
            "reported_vs_recomputed_deltas",
            lambda data: data["reported_vs_recomputed_deltas"].__setitem__("all_zero", False),
        ),
        (
            "attack_matrix",
            lambda data: data["attack_matrix"]["rows"][0].__setitem__("fail_closed", False),
        ),
        (
            "best_capacity_selected_without_held_tuning",
            lambda data: data["best_capacity_selected_without_held_tuning"].__setitem__(
                "selection_rule_frozen_before_held_outcomes",
                False,
            ),
        ),
        (
            "verifier_is_oracle",
            lambda data: data["verifier_is_oracle"]["false_for"].__setitem__("memory", True),
        ),
        (
            "prospective_write_once_csl_ready_score",
            lambda data: data.__setitem__("prospective_write_once_csl_ready_score", 0.0),
        ),
        ("honest_verdict", lambda data: data.__setitem__("honest_verdict", "done")),
        (
            "reproducibility_checksum",
            lambda data: data.__setitem__("reproducibility_checksum", "sha256:bad"),
        ),
    ]

    for expected, mutate in mutations:
        bad = deepcopy(artifact)
        mutate(bad)
        if expected != "reproducibility_checksum":
            bad["reproducibility_checksum"] = mod.payload_checksum(bad)
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad)


def test_req_learn_6430_helper_failures_and_stable_write(tmp_path: Path) -> None:
    """REQ-LEARN-6430: helper failures are explicit and writes are stable."""

    non_object = tmp_path / "list.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="json_object"):
        mod.read_json(non_object)

    context = mod.load_context(REPO)
    outside = mod.path_receipt(Path("/tmp/nonexistent-exp6430-file"), relative_to=REPO)
    assert outside["present"] is False

    bad_context = deepcopy(context)
    bad_context["exp6428"]["clean_write_time_admission_ready_score"] = 0.0
    bad_context["exp6428"]["status"] = "complete_null"
    bad_context["exp6428"]["current_adversarial_flag_count"] = 1
    bad_context["exp6428"]["reported_vs_recomputed_deltas"]["all_zero"] = False
    gates = mod.exp6428_gate_receipts(REPO, bad_context)
    assert gates["all_gates_passed"] is False
    assert {
        "exp6428_not_ready",
        "exp6428_ready_score_not_one",
        "exp6428_adversarial_flags_present",
        "exp6428_aggregates_do_not_recompute",
    }.issubset(set(gates["blocked_reasons"]))

    worse_context = deepcopy(context)
    worse_context["exp6426"]["runtime_receipt_contract_ready_score"] = 0.0
    worse_context["exp6420"]["harm_underpowered_missing_and_flagged_cells"][
        "open_critical_attack_ids"
    ] = []
    worse_gates = mod.exp6428_gate_receipts(REPO, worse_context)
    assert {
        "exp6426_receipt_gate_failed",
        "exp6420_v552_defects_not_visible",
    }.issubset(set(worse_gates["blocked_reasons"]))

    preconditions = mod.preconditions_checked(
        root=REPO,
        run_date="19000101",
        gates={"all_gates_passed": False},
        helper={"all_mandated_models_returned": False},
        model_hashes=[
            {"tokenizer_loadable": False, "autotokenizer_used": True},
        ],
        task_receipts={
            "generated_with_task_scoped_helper": False,
            "gpu_runner_receipts": {"runner_selected": False},
        },
        absence={"new_stream_paths_absent_before_generation": False},
        manifest={
            "event_count": 0,
            "partition_seals": {"future": {"untouched_before_evaluation": False}},
        },
        contract={"capacities_frozen_before_outcomes": False},
        protected_before={"missing": None},
        source_before={"missing": None},
    )
    assert {
        "wrong_planning_date",
        "upstream_gates_failed",
        "cached_sota_pair_missing_model",
        "embedded_tokenizer_not_loadable",
        "autotokenizer_used",
        "task_scoped_helper_missing",
        "runner_not_selected",
        "manifest_or_artifact_present_before_run",
        "event_count_too_small",
        "future_partition_touched",
        "capacities_not_frozen",
        "protected_hash_missing",
        "source_hash_missing",
    }.issubset(set(preconditions["blocked_reasons"]))

    blocked_artifact = mod.build_artifact(
        root=REPO,
        run_date="19000101",
        duration_s=0.25,
        tests_run=_passing_tests(),
        data_dir=tmp_path / "blocked_data",
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
        data_dir=tmp_path / "timed_data",
        output_path=tmp_path / "timed.json",
    )
    assert timed_artifact["duration_s"] > 0.0001

    null_artifact = _artifact(tmp_path / "null")
    null_artifact["tests_run"]["all_passed"] = False
    assert mod.status(null_artifact) == "complete_null"
    assert mod.honest_verdict(null_artifact).startswith("complete_null:")
    assert mod._ci95(0, 0) == [0.0, 0.0]

    bad_artifact = _artifact(tmp_path / "bad")
    bad_artifact["MODEL_SPECS"][0]["hf_id"] = "wrong"
    bad_artifact["reproducibility_checksum"] = mod.payload_checksum(bad_artifact)
    with pytest.raises(ValueError, match="MODEL_SPECS"):
        mod.validate_artifact(bad_artifact)

    output = tmp_path / "experiment_6430.json"
    written = mod.write_artifact(
        output_path=output,
        root=REPO,
        run_date=mod.RUN_DATE,
        duration_s=0.25,
        tests_run=_passing_tests(),
        data_dir=tmp_path / "write_data",
    )
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded == written
    assert loaded["reproducibility_checksum"] == mod.payload_checksum(loaded)
