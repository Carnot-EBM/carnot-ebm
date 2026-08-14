"""Tests for Exp6416 selective exact refinement A/B replay.

Spec refs: REQ-CONSTRAINT-VERIFY-6416,
SCENARIO-CONSTRAINT-VERIFY-6416-TRIGGERS,
SCENARIO-CONSTRAINT-VERIFY-6416-MATCHED-ARMS,
SCENARIO-CONSTRAINT-VERIFY-6416-ATTACKS.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6416_selective_exact_refinement_ab as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/constraint-verification/spec.md"


def _artifact() -> dict[str, object]:
    return mod.build_artifact(
        root=REPO,
        run_date="20260814",
        duration_s=0.0,
        tests_run={command: 0 for command in mod.DEFAULT_TEST_COMMANDS},
    )


def test_req_constraint_verify_6416_spec_declares_fields_and_principles() -> None:
    """REQ-CONSTRAINT-VERIFY-6416: OpenSpec owns the Exp6416 contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-CONSTRAINT-VERIFY-6416") :]
    normalized = " ".join(section.split())

    for marker in (
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        "SCENARIO-CONSTRAINT-VERIFY-6416-TRIGGERS",
        "SCENARIO-CONSTRAINT-VERIFY-6416-MATCHED-ARMS",
        "SCENARIO-CONSTRAINT-VERIFY-6416-ATTACKS",
        "exact_abstention",
        "missing_provenance",
        "checker_disagreement",
        "certified_ccg_reducible",
    ):
        assert marker in section

    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section or field.startswith(("gate:", "arm:"))
        assert " ".join(principle.split()) in normalized


def test_scenario_constraint_verify_6416_triggers_are_exact_and_confidence_free() -> None:
    """SCENARIO-CONSTRAINT-VERIFY-6416-TRIGGERS: confidence has no authority."""

    artifact = _artifact()
    contract = artifact["preregistered_trigger_contract"]
    arm_contract = artifact["preregistered_never_always_and_selective_arm_contract"]

    assert contract["registered_before_acceptance_outcomes"] is True
    assert set(contract["allowed_trigger_classes"]) == set(mod.TRIGGER_CLASSES)
    assert "confidence" in contract["forbidden_acceptance_authorities"]
    assert "selection_score" in contract["fields_excluded_from_routing_authority"]
    assert "exact_label_class" in contract["fields_excluded_from_routing_authority"]
    assert contract["trigger_class_counts"]["exact_abstention"] == 48
    assert contract["trigger_class_counts"]["certified_ccg_reducible"] == 30
    assert contract["trigger_class_counts"]["missing_provenance"] == 0
    assert contract["trigger_class_counts"]["checker_disagreement"] == 0
    assert artifact["confidence_authority_count"] == 0

    assert arm_contract["arms"]["never_refine"]["refinement_budget_rows"] == 0
    assert arm_contract["arms"]["always_refine"]["refinement_budget_rows"] == 72
    assert arm_contract["arms"]["selective_refine"]["refinement_budget_rows"] == 48
    assert arm_contract["arms"]["selective_refine"]["refinement_budget_rows"] < arm_contract["arms"]["always_refine"]["refinement_budget_rows"]


def test_scenario_constraint_verify_6416_matched_arms_improve_yield_without_harm() -> None:
    """SCENARIO-CONSTRAINT-VERIFY-6416-MATCHED-ARMS: selective matches always."""

    artifact = _artifact()
    arms = artifact[
        "per_arm_exact_yield_false_accept_false_reject_abstention_checker_kernel_escalation_latency_and_cost_results"
    ]["arms"]

    never = arms["never_refine"]
    always = arms["always_refine"]
    selective = arms["selective_refine"]

    assert never["row_count"] == always["row_count"] == selective["row_count"] == 72
    assert never["accepted_exact_count"] == 12
    assert always["accepted_exact_count"] == selective["accepted_exact_count"] == 36
    assert never["unresolved_abstentions"] == 48
    assert always["unresolved_abstentions"] == selective["unresolved_abstentions"] == 0
    assert never["false_accepts"] == always["false_accepts"] == selective["false_accepts"] == 0
    assert artifact["delta_exact_yield_over_never_refine"] > 0.0
    assert artifact["selective_vs_always_exact_accuracy_delta"] == 0.0
    assert artifact["selective_vs_always_work_delta"] < 0.0
    assert artifact["selective_refinement_safe_score"] == 1.0
    assert artifact["protected_leakage_count"] == 0


def test_req_constraint_verify_6416_disaggregates_model_and_trigger_results() -> None:
    """REQ-CONSTRAINT-VERIFY-6416: results stay disaggregated."""

    artifact = _artifact()
    disagg = artifact["per_model_family_and_trigger_class_results"]

    assert set(disagg["by_model_family"]) == {"gemma_dense", "gemma_moe", "qwen_moe"}
    assert set(disagg["by_trigger_class"]) == set(mod.TRIGGER_CLASSES)
    assert disagg["by_trigger_class"]["exact_abstention"]["row_count"] == 48
    assert disagg["by_trigger_class"]["certified_ccg_reducible"]["row_count"] == 30
    assert disagg["by_trigger_class"]["missing_provenance"]["row_count"] == 0
    assert disagg["by_trigger_class"]["checker_disagreement"]["row_count"] == 0
    assert all(row["false_accepts"] == 0 for row in disagg["by_trigger_class"].values())

    principles = artifact["field_principles"]
    for key in (
        "gate:exp6414",
        "gate:exp6415",
        "arm:never_refine",
        "arm:always_refine",
        "arm:selective_refine",
        "delta_exact_yield_over_never_refine",
        "selective_refinement_safe_score",
    ):
        assert key in principles


def test_scenario_constraint_verify_6416_attacks_and_mutations_fail_closed() -> None:
    """SCENARIO-CONSTRAINT-VERIFY-6416-ATTACKS: unsafe authority is rejected."""

    artifact = _artifact()
    attacks = artifact["attack_matrix"]

    assert {row["attack_id"] for row in attacks["rows"]} == set(mod.ATTACK_IDS)
    assert attacks["all_fail_closed"] is True
    assert attacks["false_accept_count"] == 0
    assert mod.validate_artifact(artifact) is True

    mutations = [
        ("required_fields", lambda data: data.pop("field_principles")),
        ("required_fields", lambda data: data.__setitem__("extra", True)),
        ("field_principles", lambda data: data["field_principles"].pop("status")),
        ("field_provenance", lambda data: data["field_provenance"].pop("status")),
        ("delta_exact_yield_over_never_refine", lambda data: data.__setitem__("delta_exact_yield_over_never_refine", "bad")),
        ("confidence_authority_count", lambda data: data.__setitem__("confidence_authority_count", 1)),
        ("protected_leakage_count", lambda data: data.__setitem__("protected_leakage_count", 1)),
        (
            "attack_matrix",
            lambda data: data["attack_matrix"].__setitem__("all_fail_closed", False),
        ),
        (
            "attack_matrix",
            lambda data: data["attack_matrix"]["rows"][0].__setitem__("fail_closed", False),
        ),
        (
            "trigger_contract",
            lambda data: data["preregistered_trigger_contract"]["allowed_trigger_classes"].append("confidence"),
        ),
        (
            "trigger_contract",
            lambda data: data["preregistered_trigger_contract"]["forbidden_acceptance_authorities"].remove("confidence"),
        ),
        (
            "verifier_is_oracle",
            lambda data: data["verifier_is_oracle"].__setitem__("confidence_is_oracle", True),
        ),
        (
            "safe_score",
            lambda data: data.__setitem__("selective_refinement_safe_score", 0.0),
        ),
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
        if expected not in {"reproducibility_checksum"}:
            bad["reproducibility_checksum"] = mod.payload_checksum(bad)
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad)


def test_req_constraint_verify_6416_helper_failure_paths_are_explicit(tmp_path: Path) -> None:
    """REQ-CONSTRAINT-VERIFY-6416: helper and precondition failures are explicit."""

    non_object = tmp_path / "list.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="json_object"):
        mod.read_json(non_object)

    context = mod._load_context(REPO)
    bad_context = deepcopy(context)
    bad_context["exp6415"] = {**bad_context["exp6415"], "ccg_kernelization_exact_ready_score": 0.0}
    gates = mod._validate_upstream_artifacts(bad_context)
    assert gates["exp6415"]["gate_passed"] is False

    assert mod._span_valid("abc", {"start": 9, "end": 10, "text_sha256": "sha256:none"}) is False

    tampered_context = deepcopy(context)
    first_row = tampered_context["exact_rows"][0]
    first_row["source_spans"]["obligation"]["start"] = -1
    records = mod._row_records(REPO, tampered_context)
    assert "missing_provenance" in records[0]["trigger_classes"]

    disagreement_context = deepcopy(context)
    exact_row = next(
        row
        for row in disagreement_context["exact_rows"]
        if row["exact_checker_outcome"]["exact_evaluable"] is True
    )
    exact_row["exact_checker_outcome"]["exact_outcome_label"] = "wrong"
    records = mod._row_records(REPO, disagreement_context)
    assert any("checker_disagreement" in row["trigger_classes"] for row in records)

    blockers = mod._preconditions(
        REPO,
        "20260813",
        {"both_gates_passed": False, "exp6414": {}, "exp6415": {}},
        {
            "raw_sidecar_hashes_match": False,
            "ccg_certificates_all_passed": False,
            "checker_versions": [],
            "future_partition": {"used_for_routing": True},
        },
        {"missing": None},
    )["blocked_reasons"]
    assert set(blockers) == {
        "wrong_planning_date",
        "upstream_gate_failed",
        "raw_sidecar_hash_mismatch",
        "ccg_certificate_failure",
        "future_partition_used_for_routing",
        "protected_hash_missing",
    }


def test_req_constraint_verify_6416_write_artifact_is_stable(tmp_path: Path) -> None:
    """REQ-CONSTRAINT-VERIFY-6416: writing preserves schema and checksum."""

    output = tmp_path / mod.RESULT_RELATIVE_PATH.name
    artifact = mod.write_artifact(
        output_path=output,
        root=REPO,
        run_date="20260814",
        duration_s=0.0,
        tests_run={command: 0 for command in mod.DEFAULT_TEST_COMMANDS},
    )
    loaded = json.loads(output.read_text(encoding="utf-8"))

    assert loaded == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) == set(artifact)
    assert artifact["status"] == "complete_safe"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"]["value"] is True
    assert artifact["verifier_is_oracle"]["routing_is_oracle"] is False
    assert artifact["verifier_is_oracle"]["confidence_is_oracle"] is False
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.validate_artifact(artifact) is True
