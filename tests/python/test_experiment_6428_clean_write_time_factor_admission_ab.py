"""Tests for Exp6428 clean write-time factor admission A/B.

Spec refs: REQ-LEARN-6428, SCENARIO-LEARN-6428-GATES,
SCENARIO-LEARN-6428-MATCHED-ARMS, SCENARIO-LEARN-6428-ADMISSION,
SCENARIO-LEARN-6428-FUTURE, SCENARIO-LEARN-6428-ATTACKS,
SCENARIO-LEARN-6428-READY.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from carnot import experiment_6428_clean_write_time_factor_admission_ab as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _passing_tests() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _artifact() -> mod.JsonDict:
    return mod.build_artifact(
        root=REPO,
        run_date=mod.RUN_DATE,
        duration_s=0.25,
        tests_run=_passing_tests(),
    )


def test_req_learn_6428_spec_declares_fields_principles_and_scenarios() -> None:
    """REQ-LEARN-6428: OpenSpec owns the Exp6428 contract."""

    spec = SPEC.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-LEARN-6428") : spec.index("REQ-LEARN-6418")]
    normalized = " ".join(section.split())

    for marker in (
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        "SCENARIO-LEARN-6428-GATES",
        "SCENARIO-LEARN-6428-MATCHED-ARMS",
        "SCENARIO-LEARN-6428-ADMISSION",
        "SCENARIO-LEARN-6428-FUTURE",
        "SCENARIO-LEARN-6428-ATTACKS",
        "SCENARIO-LEARN-6428-READY",
        "clean_write_time_admission_ready_score",
        "current_adversarial_flag_count",
    ):
        assert marker in section

    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section or field.startswith(("gate:", "arm:"))
        assert " ".join(principle.split()) in normalized


def test_scenario_learn_6428_gates_bind_clean_exp6427_corpus() -> None:
    """SCENARIO-LEARN-6428-GATES: clean Exp6427 evidence gates the run."""

    artifact = _artifact()
    gates = artifact["exp6427_gate_receipts"]
    raw = artifact["upstream_model_process_raw_output_and_row_hashes"]
    corpus = artifact["corpus_event_order_partition_checker_license_and_head_hashes"]

    assert gates["gate_passed"] is True
    assert gates["fresh_row_recomputable_factor_corpus_ready_score"] == 1.0
    assert gates["current_adversarial_flag_count"] == 0
    assert gates["upstream_exp6417_flagged_duration"] is True
    assert raw["row_count"] == 144
    assert raw["all_row_hashes_match"] is True
    assert raw["all_raw_hashes_match"] is True
    assert raw["new_model_generation_count"] == 0

    assert corpus["event_order"]["row_count"] == 144
    assert corpus["partitions"]["acquisition"]["row_count"] == 48
    assert corpus["partitions"]["protected_retention"]["source_partition"] == "calibration"
    assert corpus["partitions"]["future"]["used_for_proposals"] is False
    assert corpus["license"]["licensed_row_count"] == 64
    assert corpus["initial_factor_head"]["head_hash"] == artifact["matched_work_receipts"]["initial_head_hash"]
    assert artifact["preconditions_checked"]["all_preconditions_passed"] is True
    assert artifact["blocked_reason"] == ""


def test_scenario_learn_6428_matched_arms_and_exact_admission_dispositions() -> None:
    """SCENARIO-LEARN-6428-ADMISSION: exact support owns commits."""

    artifact = _artifact()
    contract = artifact["preregistered_frozen_write_everything_and_exact_admission_arm_contract"]
    work = artifact["matched_work_receipts"]
    records = artifact["atomic_disposition_records"]
    bindings = artifact[
        "per_proposal_source_model_license_checker_predecessor_expiry_and_supersession_bindings"
    ]

    assert set(contract["arms"]) == set(mod.ARMS)
    assert contract["future_partition_opened_after_dispositions"] is True
    assert len({row["event_order_sha256"] for row in contract["arms"].values()}) == 1
    assert len({row["checker_call_count"] for row in work["by_arm"].values()}) == 1
    assert work["proposal_count_per_arm"] == 96
    assert work["consumer_budget_per_arm"] == 96

    exact_rows = [
        row for row in records["rows"] if row["arm"] == mod.EXACT_ADMISSION_ARM
    ]
    assert records["counts_by_arm"][mod.FROZEN_ARM]["Commit"] == 0
    assert records["counts_by_arm"][mod.EXACT_ADMISSION_ARM]["Commit"] == 11
    assert records["counts_by_arm"][mod.WRITE_EVERYTHING_ARM]["Commit"] > 11
    assert all(row["atomic_recorded"] for row in exact_rows)
    assert all(
        row["joint_exact"] is True and row["license_valid"] is True
        for row in exact_rows
        if row["disposition"] == "Commit"
    )
    assert all(
        row["disposition"] != "Commit"
        for row in exact_rows
        if row["license_valid"] is not True or row["joint_exact"] is not True
    )
    assert records["fail_closed_class_counts"]["unlicensed"] == 53
    assert records["fail_closed_class_counts"]["missing_exact"] > 0

    assert bindings["proposal_count"] == 288
    assert bindings["all_predecessor_heads_bound"] is True
    assert bindings["future_label_visible_before_disposition_count"] == 0
    assert bindings["rows"][0]["refinement_hashes"]["row_hash"].startswith("sha256:")


def test_scenario_learn_6428_future_rows_recompute_ready_metrics_and_oracle_boundary() -> None:
    """SCENARIO-LEARN-6428-READY: clean future gain must not add harm."""

    artifact = _artifact()
    future = artifact["untouched_future_evaluation_receipts"]
    aggregates = artifact["aggregate_recomputation_receipts"]["by_arm"]

    assert future["open_count"] == 1
    assert future["future_row_count"] == 48
    assert future["per_arm_future_row_count"] == 144
    assert future["future_outcomes_visible_before_disposition_count"] == 0
    assert future["per_unit_rows_written_before_aggregates"] is True

    assert artifact["per_unit_rows"]["row_count"] == 144
    assert artifact["per_unit_rows"]["rows"][0]["arm"] in mod.ARMS
    assert artifact["reported_vs_recomputed_deltas"]["all_zero"] is True
    assert artifact["delta_future_exact_yield"] > 0.0
    assert artifact["delta_contamination_propagation_rate"] == 0.0
    assert artifact["protected_retention_delta"] >= 0.0
    assert artifact["false_reject_delta"] < 0.0
    assert all(math.isfinite(float(artifact[field])) for field in mod.BARE_FINITE_FIELDS)

    assert aggregates[mod.EXACT_ADMISSION_ARM]["future_exact_yield"] > aggregates[mod.FROZEN_ARM]["future_exact_yield"]
    assert aggregates[mod.EXACT_ADMISSION_ARM]["contamination_propagation_rate"] == aggregates[mod.FROZEN_ARM]["contamination_propagation_rate"]
    assert aggregates[mod.EXACT_ADMISSION_ARM]["contamination_propagation_rate"] < aggregates[mod.WRITE_EVERYTHING_ARM]["contamination_propagation_rate"]
    assert artifact["factor_growth_by_arm"][mod.EXACT_ADMISSION_ARM] == 11
    assert artifact["exact_work_by_arm"][mod.EXACT_ADMISSION_ARM] == 96
    assert artifact["current_adversarial_flag_count"] == 0
    assert artifact["clean_write_time_admission_ready_score"] == 1.0
    assert artifact["public_factor_claim_eligibility"]["eligible"] is True

    oracle = artifact["verifier_is_oracle"]
    assert oracle["value"] is True
    assert set(oracle["true_for"]) == {"exact_event_checker", "protected_retention_checker"}
    assert oracle["false_for"]["admission"] is False
    assert oracle["false_for"]["memory"] is False


def test_scenario_learn_6428_attacks_and_schema_mutations_fail_closed() -> None:
    """SCENARIO-LEARN-6428-ATTACKS: unsafe authority cannot pass validation."""

    artifact = _artifact()
    attacks = artifact["attack_matrix"]

    assert {row["attack_id"] for row in attacks["rows"]} == set(mod.ATTACK_IDS)
    assert attacks["all_fail_closed"] is True
    assert attacks["committed_attack_count"] == 0
    assert mod.validate_artifact(artifact) is True

    mutations = [
        ("required_fields", lambda data: data.pop("field_principles")),
        ("required_fields", lambda data: data.__setitem__("extra", True)),
        ("field_principles", lambda data: data["field_principles"].pop("status")),
        ("field_provenance", lambda data: data["field_provenance"].pop("status")),
        ("delta_future_exact_yield", lambda data: data.__setitem__("delta_future_exact_yield", "bad")),
        ("protected_retention_delta", lambda data: data.__setitem__("protected_retention_delta", -0.1)),
        ("exact_veto_override_count", lambda data: data.__setitem__("exact_veto_override_count", 1)),
        ("protected_leakage_count", lambda data: data.__setitem__("protected_leakage_count", 1)),
        ("runtime_field_synthesis_count", lambda data: data.__setitem__("runtime_field_synthesis_count", 1)),
        ("current_adversarial_flag_count", lambda data: data.__setitem__("current_adversarial_flag_count", 1)),
        (
            "reported_vs_recomputed_deltas",
            lambda data: data["reported_vs_recomputed_deltas"].__setitem__("all_zero", False),
        ),
        (
            "attack_matrix",
            lambda data: data["attack_matrix"]["rows"][0].__setitem__("fail_closed", False),
        ),
        (
            "verifier_is_oracle",
            lambda data: data["verifier_is_oracle"]["false_for"].__setitem__("memory", True),
        ),
        (
            "readiness",
            lambda data: data.__setitem__("clean_write_time_admission_ready_score", 0.0),
        ),
        ("status", lambda data: data.__setitem__("status", "bad")),
        ("honest_verdict", lambda data: data.__setitem__("honest_verdict", "done")),
        (
            "reproducibility_checksum",
            lambda data: data.__setitem__("reproducibility_checksum", "sha256:bad"),
        ),
        ("field_principles", lambda data: data["field_principles"].pop("gate:exp6427")),
    ]

    for expected, mutate in mutations:
        bad = deepcopy(artifact)
        mutate(bad)
        if expected != "reproducibility_checksum":
            bad["reproducibility_checksum"] = mod.payload_checksum(bad)
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad)


def test_req_learn_6428_helper_failures_and_stable_write(tmp_path: Path) -> None:
    """REQ-LEARN-6428: helper failures are explicit and writes are stable."""

    non_object = tmp_path / "list.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="json_object"):
        mod.read_json(non_object)

    context = mod.load_context(REPO)
    outside = mod.path_receipt(Path("/tmp/nonexistent-exp6428-file"), relative_to=REPO)
    assert outside["present"] is False

    bad_context = deepcopy(context)
    bad_context["exp6427"]["current_adversarial_flag_count"] = 1
    bad_context["exp6427"]["status"] = "blocked"
    bad_context["exp6427"]["fresh_row_recomputable_factor_corpus_ready_score"] = 0.0
    bad_context["exp6427"]["protected_leakage_count"] = 1
    bad_context["exp6427"]["reported_vs_recomputed_deltas"]["all_zero"] = False
    bad_context["exp6427"]["attack_matrix"]["all_fail_closed"] = False
    gates = mod.exp6427_gate_receipts(REPO, bad_context)
    assert gates["gate_passed"] is False
    assert {
        "exp6427_not_complete",
        "exp6427_ready_score_not_one",
        "exp6427_adversarial_flags_present",
        "exp6427_protected_leakage",
        "exp6427_aggregates_do_not_recompute",
        "exp6427_attack_matrix_open",
    }.issubset(set(gates["blocked_reasons"]))

    bad_raw_context = deepcopy(context)
    bad_raw_context["rows_by_id"][bad_raw_context["ordered_row_ids"][0]]["raw_output_sha256"] = "sha256:bad"
    raw = mod.upstream_model_process_raw_output_and_row_hashes(REPO, bad_raw_context)
    assert raw["all_raw_hashes_match"] is False

    duplicate_context = deepcopy(context)
    duplicate_row_id = next(
        row_id
        for row_id in duplicate_context["ordered_row_ids"]
        if duplicate_context["rows_by_id"][row_id]["partition"] in mod.PROPOSAL_PARTITIONS
        and duplicate_context["rows_by_id"][row_id]["source_license"]["licensed"] is True
    )
    duplicate_context["rows_by_id"][duplicate_row_id]["duplicate"] = True
    corpus = mod.corpus_event_order_partition_checker_license_and_head_hashes(REPO, duplicate_context)
    duplicate_bindings = mod.proposal_bindings(duplicate_context, corpus)
    assert any(row["supersession_state"] == "duplicate" for row in duplicate_bindings["rows"])

    assert mod._exact_fail_reason({"raw_hash_matches": False}) == "source_replacement"
    assert mod._exact_fail_reason({"raw_hash_matches": True, "license_valid": False}) == "unlicensed"
    assert mod._exact_fail_reason({"raw_hash_matches": True, "license_valid": True, "malformed": True}) == "malformed"
    assert mod._exact_fail_reason({"raw_hash_matches": True, "license_valid": True, "duplicate": True}) == "duplicate"
    assert mod._exact_fail_reason({"raw_hash_matches": True, "license_valid": True, "predecessor_fresh": False}) == "stale_head"
    assert mod._exact_fail_reason({"raw_hash_matches": True, "license_valid": True, "predecessor_fresh": True, "evaluable": True, "joint_exact": True}) == "not_joint_exact"
    assert mod._disposition_for_binding({"arm": mod.EXACT_ADMISSION_ARM, "malformed": True})[
        "disposition"
    ] == "Quarantine"

    preconditions = mod.preconditions_checked(
        REPO,
        "19000101",
        {"gate_passed": False},
        {"all_row_hashes_match": False, "all_raw_hashes_match": False},
        {
            "event_order": {"order_is_strict": False},
            "partitions": {mod.FUTURE_PARTITION: {"used_for_proposals": True}},
            "checker": {"all_oracle_scoped": False},
            "license": {"license_matrix_ready": False},
            "initial_factor_head": {},
        },
        {"missing": None},
    )
    assert {
        "wrong_planning_date",
        "exp6427_gate_failed",
        "row_hash_mismatch",
        "raw_hash_mismatch",
        "event_order_not_strict",
        "future_partition_used_for_proposals",
        "checker_scope_failed",
        "license_gate_failed",
        "initial_head_missing",
        "protected_hash_missing",
    }.issubset(set(preconditions["blocked_reasons"]))

    output = tmp_path / "experiment_6428.json"
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
