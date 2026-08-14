"""Tests for Exp6417 authentic write-time factor admission A/B.

Spec refs: REQ-LEARN-6417, SCENARIO-LEARN-6417-GATES,
SCENARIO-LEARN-6417-MATCHED-ARMS, SCENARIO-LEARN-6417-ADMISSION,
SCENARIO-LEARN-6417-FUTURE, SCENARIO-LEARN-6417-ATTACKS,
SCENARIO-LEARN-6417-READY.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from carnot import experiment_6417_authentic_write_time_factor_admission_ab as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _passing_tests() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _artifact() -> mod.JsonDict:
    return mod.build_artifact(
        root=REPO,
        run_date=mod.RUN_DATE,
        duration_s=0.0,
        tests_run=_passing_tests(),
    )


def test_req_learn_6417_spec_declares_fields_principles_and_scenarios() -> None:
    """REQ-LEARN-6417: OpenSpec owns the Exp6417 contract."""

    spec = SPEC.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-LEARN-6417") : spec.index("REQ-LEARN-6409")]
    normalized = " ".join(section.split())

    for marker in (
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        "SCENARIO-LEARN-6417-GATES",
        "SCENARIO-LEARN-6417-MATCHED-ARMS",
        "SCENARIO-LEARN-6417-ADMISSION",
        "SCENARIO-LEARN-6417-FUTURE",
        "SCENARIO-LEARN-6417-ATTACKS",
        "SCENARIO-LEARN-6417-READY",
        "authentic_write_time_admission_ready_score",
        "protected_retention_delta",
    ):
        assert marker in section

    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section or field.startswith("gate:")
        assert " ".join(principle.split()) in normalized


def test_scenario_learn_6417_gates_quarantine_old_claim_and_bind_upstream() -> None:
    """SCENARIO-LEARN-6417-GATES: old powered evidence stays quarantined."""

    artifact = _artifact()
    gates = artifact["exp6412_exp6414_and_exp6416_gate_receipts"]
    raw = artifact["upstream_process_receipt_and_raw_output_hashes"]
    corpus = artifact["corpus_event_order_partition_checker_license_and_head_hashes"]

    assert gates["all_gates_passed"] is True
    assert gates["exp6412"]["old_exp6408_powered_claim_quarantined"] is True
    assert gates["exp6412"]["public_factor_claim_eligible"] is False
    assert gates["exp6414"]["gate_passed"] is True
    assert gates["exp6416"]["gate_passed"] is True
    assert gates["exp6408_quarantine"]["public_claim_reused"] is False

    assert raw["raw_output_row_count"] == 72
    assert raw["all_raw_hashes_match"] is True
    assert raw["accepted_process_receipt_count"] == 72
    assert raw["new_model_generation_count"] == 0
    assert corpus["event_order"]["row_count"] == 72
    assert corpus["partitions"]["future"]["row_count"] == 24
    assert corpus["license"]["license_inheritance_count"] == 0
    assert corpus["initial_factor_head"]["head_hash"] == artifact["matched_work_receipts"]["initial_head_hash"]
    assert artifact["preconditions_checked"]["all_preconditions_passed"] is True


def test_scenario_learn_6417_matched_arms_and_exact_admission_dispositions() -> None:
    """SCENARIO-LEARN-6417-MATCHED-ARMS: exact support controls commits."""

    artifact = _artifact()
    contract = artifact["preregistered_frozen_write_everything_and_exact_admission_arm_contract"]
    work = artifact["matched_work_receipts"]
    records = artifact["atomic_disposition_records"]
    bindings = artifact[
        "per_proposal_raw_source_model_license_checker_predecessor_refinement_expiry_and_supersession_bindings"
    ]

    assert set(contract["arms"]) == set(mod.ARMS)
    assert contract["future_partition_opened_after_dispositions"] is True
    assert len({row["event_order_sha256"] for row in contract["arms"].values()}) == 1
    assert len({row["checker_call_count"] for row in work["by_arm"].values()}) == 1
    assert work["proposal_count_per_arm"] == 48
    assert work["consumer_budget_per_arm"] == 48

    exact_rows = [
        row for row in records["rows"] if row["arm"] == mod.EXACT_ADMISSION_ARM
    ]
    assert records["counts_by_arm"][mod.EXACT_ADMISSION_ARM]["Commit"] == 4
    assert records["counts_by_arm"][mod.WRITE_EVERYTHING_ARM]["Commit"] > 4
    assert records["counts_by_arm"][mod.FROZEN_ARM]["Commit"] == 0
    assert all(row["atomic_recorded"] for row in exact_rows)
    assert all(
        row["exact_label_class"] == "clean" and row["license_valid"] is True
        for row in exact_rows
        if row["disposition"] == "Commit"
    )
    assert all(
        row["disposition"] != "Commit"
        for row in exact_rows
        if row["exact_label_class"] in {"contradicted", "implicit", "stale"}
        or row["license_valid"] is not True
    )
    assert records["fail_closed_class_counts"]["implicit"] > 0
    assert records["fail_closed_class_counts"]["contradicted"] > 0
    assert records["fail_closed_class_counts"]["unlicensed"] > 0

    assert bindings["proposal_count"] == 144
    assert bindings["all_source_spans_valid"] is True
    assert bindings["all_predecessor_heads_bound"] is True
    assert bindings["future_label_visible_before_disposition_count"] == 0
    assert bindings["rows"][0]["refinement_receipt_sha256"].startswith("sha256:")


def test_scenario_learn_6417_future_ready_metrics_and_oracle_boundary() -> None:
    """SCENARIO-LEARN-6417-READY: future gain must not add harm."""

    artifact = _artifact()
    results = artifact[
        "per_arm_cell_exact_yield_contamination_false_accept_false_reject_retention_abstention_growth_escalation_and_work_results"
    ]
    arms = results["by_arm"]
    future = artifact["untouched_future_evaluation_receipts"]

    assert future["open_count"] == 1
    assert future["future_outcomes_visible_before_disposition_count"] == 0
    assert future["future_row_count"] == 24
    assert future["evaluated_once_after_head_freeze"] is True

    assert arms[mod.EXACT_ADMISSION_ARM]["future_exact_yield"] > arms[mod.FROZEN_ARM]["future_exact_yield"]
    assert arms[mod.EXACT_ADMISSION_ARM]["contamination_propagation_rate"] == arms[mod.FROZEN_ARM]["contamination_propagation_rate"]
    assert arms[mod.EXACT_ADMISSION_ARM]["contamination_propagation_rate"] < arms[mod.WRITE_EVERYTHING_ARM]["contamination_propagation_rate"]
    assert artifact["delta_future_exact_yield"] > 0.0
    assert artifact["delta_contamination_propagation_rate"] == 0.0
    assert artifact["protected_retention_delta"] >= 0.0
    assert all(math.isfinite(float(artifact[field])) for field in mod.BARE_FINITE_FIELDS)

    assert artifact["authentic_write_time_admission_ready_score"] == 1.0
    assert artifact["public_factor_claim_eligibility"]["eligible"] is True
    assert artifact["silent_fallback_count"] == 0
    assert artifact["exact_veto_override_count"] == 0
    assert artifact["protected_leakage_count"] == 0
    assert artifact["runtime_field_synthesis_count"] == 0

    oracle = artifact["verifier_is_oracle"]
    assert oracle["value"] is True
    assert "exact_event_checker" in oracle["true_for"]
    assert "retention_checker" in oracle["true_for"]
    for forbidden in ("upstream_model_output", "admission", "memory", "diagnostics"):
        assert oracle["false_for"][forbidden] is False


def test_scenario_learn_6417_attacks_and_schema_mutations_fail_closed() -> None:
    """SCENARIO-LEARN-6417-ATTACKS: unsafe authority cannot pass validation."""

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
        ("silent_fallback_count", lambda data: data.__setitem__("silent_fallback_count", 1)),
        ("exact_veto_override_count", lambda data: data.__setitem__("exact_veto_override_count", 1)),
        ("protected_leakage_count", lambda data: data.__setitem__("protected_leakage_count", 1)),
        ("runtime_field_synthesis_count", lambda data: data.__setitem__("runtime_field_synthesis_count", 1)),
        (
            "attack_matrix",
            lambda data: data["attack_matrix"].__setitem__("all_fail_closed", False),
        ),
        (
            "attack_matrix",
            lambda data: data["attack_matrix"]["rows"][0].__setitem__("fail_closed", False),
        ),
        (
            "verifier_is_oracle",
            lambda data: data["verifier_is_oracle"]["false_for"].__setitem__("admission", True),
        ),
        (
            "readiness",
            lambda data: data.__setitem__("authentic_write_time_admission_ready_score", 0.0),
        ),
        (
            "public_factor_claim_eligibility",
            lambda data: data["public_factor_claim_eligibility"].__setitem__("eligible", False),
        ),
        ("status", lambda data: data.__setitem__("status", "bad")),
        ("honest_verdict", lambda data: data.__setitem__("honest_verdict", "done")),
        (
            "reproducibility_checksum",
            lambda data: data.__setitem__("reproducibility_checksum", "sha256:bad"),
        ),
        (
            "field_principles",
            lambda data: data["field_principles"].pop("gate:exp6412"),
        ),
    ]

    for expected, mutate in mutations:
        bad = deepcopy(artifact)
        mutate(bad)
        if expected != "reproducibility_checksum":
            bad["reproducibility_checksum"] = mod.payload_checksum(bad)
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad)


def test_req_learn_6417_helper_failures_and_stable_write(tmp_path: Path) -> None:
    """REQ-LEARN-6417: helper failures are explicit and writes are stable."""

    non_object = tmp_path / "list.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="json_object"):
        mod.read_json(non_object)

    context = mod.load_context(REPO)
    first_event = context["events_by_id"][context["ordered_row_ids"][0]]
    assert mod.span_valid(str(first_event["source_text"]), {"start": 99, "end": 1}) is False
    outside = mod.path_receipt(Path("/tmp/nonexistent-exp6417-file"), relative_to=REPO)
    assert outside["present"] is False

    bad_context = deepcopy(context)
    bad_context["exp6412"]["public_factor_claim_eligibility"]["eligible"] = True
    gates = mod.gate_receipts(REPO, bad_context)
    assert gates["all_gates_passed"] is False
    assert "exp6412_public_claim_not_quarantined" in gates["blocked_reasons"]

    many_bad = deepcopy(context)
    many_bad["exp6412"]["v551_claim_boundary_ready_score"] = 0.0
    many_bad["exp6412"]["powered_gguf_claim_eligibility"]["eligible"] = True
    many_bad["exp6412"]["audited_source_artifact_sidecar_and_log_hashes"]["artifacts"][
        mod.EXP6408_RELATIVE_PATH.as_posix()
    ]["sha256"] = "sha256:wrong"
    many_bad["exp6414"]["status"] = "blocked"
    many_bad["exp6416"]["selective_refinement_safe_score"] = 0.0
    many_bad_gates = mod.gate_receipts(REPO, many_bad)
    assert {
        "exp6412_boundary_not_ready",
        "exp6412_powered_claim_not_quarantined",
        "exp6414_gate_failed",
        "exp6416_gate_failed",
        "exp6408_audit_hash_mismatch",
    }.issubset(set(many_bad_gates["blocked_reasons"]))

    bad_raw_context = deepcopy(context)
    bad_raw_context["raw_rows_by_id"][bad_raw_context["ordered_row_ids"][0]]["raw_output"][
        "sha256"
    ] = "sha256:bad"
    raw = mod.upstream_process_receipt_and_raw_output_hashes(REPO, bad_raw_context)
    assert raw["all_raw_hashes_match"] is False

    blockers = mod.preconditions_checked(
        REPO,
        "20260813",
        {"all_gates_passed": False},
        {"all_raw_hashes_match": False},
        {
            "event_order": {"order_is_strict": False},
            "partitions": {"future": {"used_for_proposals": True}},
            "checker": {"all_hashes_present": False},
            "license": {"license_matrix_ready": False},
            "initial_factor_head": {"head_hash": None},
        },
        {"missing": None},
    )["blocked_reasons"]
    assert set(blockers) == {
        "wrong_planning_date",
        "upstream_gate_failed",
        "raw_hash_mismatch",
        "event_order_not_strict",
        "future_partition_used_for_proposals",
        "checker_hash_missing",
        "license_gate_failed",
        "initial_head_missing",
        "protected_hash_missing",
    }

    base_binding = {
        "arm": mod.EXACT_ADMISSION_ARM,
        "proposal_id": "p",
        "row_id": "r",
        "partition": "acquisition",
        "chronological_index": 0,
        "exact_label_class": "clean",
        "license_valid": True,
        "exact_support": False,
        "exact_evaluable": True,
        "predecessor_fresh": True,
        "source_spans_valid": True,
        "raw_hash_matches": True,
        "refinement_receipt": {"safe_score": 1.0},
    }
    raw_bad = {**base_binding, "raw_hash_matches": False}
    source_bad = {**base_binding, "source_spans_valid": False}
    missing_exact = {**base_binding, "exact_evaluable": False}
    unknown_label = {**base_binding, "exact_label_class": "surprise"}
    malformed = {**base_binding, "exact_label_class": "malformed"}
    assert mod._disposition_for_binding(raw_bad)["reason"] == "raw_hash_mismatch"
    assert mod._disposition_for_binding(source_bad)["reason"] == "source_span_mismatch"
    assert mod._disposition_for_binding(missing_exact)["reason"] == "missing_exact"
    assert mod._disposition_for_binding(unknown_label)["reason"] == "not_clean_exact_support"
    assert mod._disposition_for_binding(malformed)["disposition"] == "Quarantine"
    assert mod.honest_verdict({"status": "complete_null"}).startswith("complete_null:")

    output = tmp_path / mod.RESULT_RELATIVE_PATH.name
    artifact = mod.write_artifact(
        output_path=output,
        root=REPO,
        run_date=mod.RUN_DATE,
        duration_s=0.0,
        tests_run=_passing_tests(),
    )
    loaded = json.loads(output.read_text(encoding="utf-8"))

    assert loaded == artifact
    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "complete_ready"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
