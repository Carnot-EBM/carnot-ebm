"""Tests for Exp6429 constraint-saturation verification-cost A/B replay.

Spec refs: REQ-CONSTRAINT-VERIFY-6429,
SCENARIO-CONSTRAINT-VERIFY-6429-BUDGETS,
SCENARIO-CONSTRAINT-VERIFY-6429-MATCHED-ARMS,
SCENARIO-CONSTRAINT-VERIFY-6429-ROWS-AND-ATTACKS.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6429_constraint_saturation_verification_cost_ab as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / mod.SPEC_RELATIVE_PATH


def _artifact() -> dict[str, object]:
    return mod.build_artifact(
        root=REPO,
        run_date="20260814",
        duration_s=0.0,
        tests_run={command: 0 for command in mod.DEFAULT_TEST_COMMANDS},
    )


def test_req_constraint_verify_6429_spec_declares_fields_and_principles() -> None:
    """REQ-CONSTRAINT-VERIFY-6429: OpenSpec owns the Exp6429 contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-CONSTRAINT-VERIFY-6429") :]
    normalized = " ".join(section.split())

    for marker in (
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        "SCENARIO-CONSTRAINT-VERIFY-6429-BUDGETS",
        "SCENARIO-CONSTRAINT-VERIFY-6429-MATCHED-ARMS",
        "SCENARIO-CONSTRAINT-VERIFY-6429-ROWS-AND-ATTACKS",
        "verification-cost error",
        "exact_abstention",
        "certified_ccg_reducible",
    ):
        assert marker in section

    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section or field.startswith(
            ("gate:", "arm:", "budget:", "cost_error", "readiness:")
        )
        assert " ".join(principle.split()) in normalized


def test_scenario_constraint_verify_6429_budgets_freeze_before_cost_errors() -> None:
    """SCENARIO-CONSTRAINT-VERIFY-6429-BUDGETS: budgets are outcome-blind."""

    artifact = _artifact()
    contract = artifact["preregistered_arm_and_budget_contract"]

    assert contract["registered_before_exact_outcomes"] is True
    assert contract["budget_choice_uses_outcomes"] is False
    assert set(contract["arms"]) == set(mod.ARM_NAMES)
    assert contract["arms"]["never_refine"]["checker_call_budget"] == 0
    assert contract["arms"]["always_refine"]["checker_call_budget"] == 144
    assert contract["arms"]["selective_refine"]["checker_call_budget"] == 64
    assert contract["arms"]["selective_refine"]["checker_call_budget"] < contract["arms"][
        "always_refine"
    ]["checker_call_budget"]
    assert contract["selective_allowed_triggers"] == list(mod.TRIGGER_CLASSES)
    assert "confidence" in contract["forbidden_acceptance_authorities"]
    assert artifact["confidence_authority_count"] == 0
    assert artifact["verification_cost_error_definition"]["incorrect_row_definition"] == (
        "evaluable row with joint_exact false"
    )


def test_scenario_constraint_verify_6429_matched_arms_and_costs() -> None:
    """SCENARIO-CONSTRAINT-VERIFY-6429-MATCHED-ARMS: selective matches always."""

    artifact = _artifact()
    arms = artifact[
        "per_arm_constraint_count_interaction_model_budget_correctness_abstention_checker_time_and_cost_error_results"
    ]["arms"]

    never = arms["never_refine"]
    always = arms["always_refine"]
    selective = arms["selective_refine"]

    assert never["row_count"] == always["row_count"] == selective["row_count"] == 144
    assert never["false_accepts"] == 49
    assert never["verification_cost_errors"] == 49
    assert always["false_accepts"] == selective["false_accepts"] == 0
    assert always["verification_cost_errors"] == selective["verification_cost_errors"] == 0
    assert always["checker_calls"] == 144
    assert selective["checker_calls"] == 64
    assert never["abstentions"] == always["abstentions"] == selective["abstentions"] == 80
    assert artifact["selective_vs_always_accuracy_delta"] == 0.0
    assert artifact["selective_vs_always_median_and_tail_cost_deltas"]["median_elapsed_time_s"] < 0.0
    assert artifact["selective_vs_always_median_and_tail_cost_deltas"]["p95_elapsed_time_s"] < 0.0
    assert artifact["false_accept_and_false_reject_deltas"]["selective_minus_always"] == {
        "false_accepts": 0,
        "false_rejects": 0,
    }
    assert artifact["verification_cost_study_ready_score"] == 1.0
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_constraint_verify_6429_rows_recompute_decay_and_uncertainty() -> None:
    """REQ-CONSTRAINT-VERIFY-6429: row-derived metrics are recomputable."""

    artifact = _artifact()
    rows = artifact["per_unit_rows"]["rows"]
    recomputed = mod.recompute_from_per_unit_rows(rows)

    assert artifact["per_unit_rows"]["row_count"] == 144
    assert artifact["per_unit_rows"]["arm_row_count"] == 432
    assert artifact["per_constraint_success"] == recomputed["per_constraint_success"]
    assert artifact["joint_success"] == recomputed["joint_success"]
    assert artifact["joint_success_decay_by_constraint_count"] == recomputed[
        "joint_success_decay_by_constraint_count"
    ]
    assert artifact["interaction_penalty"] == recomputed["interaction_penalty"]
    assert artifact["verification_cost_error_rate_by_budget"] == recomputed[
        "verification_cost_error_rate_by_budget"
    ]
    assert artifact["reported_vs_recomputed_deltas"]["all_zero"] is True

    decay = artifact["joint_success_decay_by_constraint_count"]["rows"]
    assert decay[0]["constraint_count"] == 1
    assert decay[-1]["constraint_count"] == 8
    assert decay[0]["joint_success_rate"] >= decay[-1]["joint_success_rate"]

    uncertainty = artifact["effective_sample_sizes_and_uncertainty"]["rows"]
    assert {row["interaction_class"] for row in uncertainty} == {"independent", "interacting"}
    assert all("joint_success_wilson95" in row for row in uncertainty)
    assert artifact["harm_underpowered_missing_and_flagged_cells"]["underpowered_count"] > 0


def test_scenario_constraint_verify_6429_attacks_and_mutations_fail_closed() -> None:
    """SCENARIO-CONSTRAINT-VERIFY-6429-ROWS-AND-ATTACKS: unsafe paths close."""

    artifact = _artifact()
    attacks = artifact["attack_matrix"]

    assert {row["attack_id"] for row in attacks["rows"]} == set(mod.ATTACK_IDS)
    assert attacks["all_fail_closed"] is True
    assert attacks["false_accept_count"] == 0
    assert mod.validate_artifact(artifact) is True

    mutations = [
        ("required_fields", lambda data: data.pop("per_unit_rows")),
        ("required_fields", lambda data: data.__setitem__("extra", True)),
        ("field_principles", lambda data: data["field_principles"].pop("status")),
        ("field_provenance", lambda data: data["field_provenance"].pop("status")),
        ("confidence_authority_count", lambda data: data.__setitem__("confidence_authority_count", 1)),
        (
            "budget_contract",
            lambda data: data["preregistered_arm_and_budget_contract"].__setitem__(
                "budget_choice_uses_outcomes", True
            ),
        ),
        (
            "budget_contract",
            lambda data: data["preregistered_arm_and_budget_contract"]["selective_allowed_triggers"].append(
                "confidence"
            ),
        ),
        (
            "attack_matrix",
            lambda data: data["attack_matrix"].__setitem__("all_fail_closed", False),
        ),
        (
            "attack_matrix",
            lambda data: data["attack_matrix"]["rows"][0].__setitem__("fail_closed", False),
        ),
        (
            "reported_vs_recomputed_deltas",
            lambda data: data["reported_vs_recomputed_deltas"].__setitem__("all_zero", False),
        ),
        (
            "false_accept_delta",
            lambda data: data["false_accept_and_false_reject_deltas"][
                "selective_minus_always"
            ].__setitem__("false_accepts", 1),
        ),
        (
            "verifier_is_oracle",
            lambda data: data["verifier_is_oracle"].__setitem__("confidence_is_oracle", True),
        ),
        ("ready_score", lambda data: data.__setitem__("verification_cost_study_ready_score", 0.0)),
        ("status", lambda data: data.__setitem__("status", "bad")),
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


def test_req_constraint_verify_6429_preconditions_and_write_paths(tmp_path: Path) -> None:
    """REQ-CONSTRAINT-VERIFY-6429: blockers, read errors, and writes are explicit."""

    artifact = _artifact()
    output = tmp_path / mod.RESULT_RELATIVE_PATH.name
    written = mod.write_artifact(
        output_path=output,
        root=REPO,
        run_date="20260814",
        duration_s=0.0,
        tests_run={command: 0 for command in mod.DEFAULT_TEST_COMMANDS},
    )
    assert json.loads(output.read_text(encoding="utf-8")) == written
    assert written["reproducibility_checksum"] == mod.payload_checksum(written)

    non_object = tmp_path / "list.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="json_object"):
        mod.read_json(non_object)
    assert mod.sha256_file(tmp_path / "missing.bin") is None

    context = mod._load_context(REPO)
    bad_context = deepcopy(context)
    bad_context["exp6416"] = {}
    gates = mod._validate_upstream_gates(REPO, bad_context)
    assert gates["exp6416_reference"]["artifact_valid"] is False
    assert gates["both_gates_passed"] is False

    first = deepcopy(context["rows"][0])
    first["source_identity"] = {}
    assert "missing_provenance" in mod._row_triggers(first, True)
    disagree = deepcopy(next(row for row in context["rows"] if row["evaluable"] is True))
    disagree["joint_exact"] = not disagree["joint_exact"]
    assert "checker_disagreement" in mod._row_triggers(disagree, True)

    blockers = mod.preconditions_checked(
        root=REPO,
        run_date="20260813",
        gates={"both_gates_passed": False},
        hashes={
            "row_hash_matches": False,
            "raw_output_hashes_match": False,
            "constraint_strata_balanced": False,
            "checker_versions_present": False,
            "ccg_certificates_all_passed": False,
            "future_partition_used_for_routing": True,
            "monotonic_timing_ok": False,
        },
        protected_before={"missing": None},
        host_receipt={"cpu_count": 0, "ram_total_bytes": 0, "disk_free_bytes": 0},
    )
    assert set(blockers["blocked_reasons"]) == {
        "wrong_planning_date",
        "upstream_gate_failed",
        "row_hash_mismatch",
        "raw_output_hash_mismatch",
        "constraint_strata_unbalanced",
        "checker_version_missing",
        "ccg_certificate_failure",
        "future_partition_used_for_routing",
        "monotonic_timing_failed",
        "host_resource_receipt_incomplete",
        "protected_hash_missing",
    }

    blocked_status = deepcopy(artifact)
    blocked_status["blocked_reason"] = "fixture"
    assert mod.status(blocked_status) == "blocked_precondition"
    blocked_status["status"] = "blocked_precondition"
    assert mod.honest_verdict(blocked_status).startswith("complete_blocked:")
    null_status = deepcopy(artifact)
    null_status["verification_cost_study_ready_score"] = 0.0
    assert mod.status(null_status) == "complete_null"
    null_status["status"] = "complete_null"
    assert mod.honest_verdict(null_status).startswith("complete_null:")

    bad = deepcopy(artifact)
    bad["per_unit_rows"]["rows"] = bad["per_unit_rows"]["rows"][:-1]
    bad["per_unit_rows"]["row_count"] = 143
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    with pytest.raises(ValueError, match="per_unit_rows"):
        mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["per_unit_rows"]["arm_row_count"] = 431
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    with pytest.raises(ValueError, match="per_unit_rows"):
        mod.validate_artifact(bad)
