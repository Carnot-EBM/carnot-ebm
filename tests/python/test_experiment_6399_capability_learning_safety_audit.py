"""Tests for Exp6399 V550 capability-learning safety audit.

Spec refs: REQ-LEARN-6399, SCENARIO-LEARN-6399-REGISTRATION,
SCENARIO-LEARN-6399-CLASS-PRESERVATION,
SCENARIO-LEARN-6399-LICENSE-BOUNDARY,
SCENARIO-LEARN-6399-TRANSACTION-BOUNDARY, SCENARIO-LEARN-6399-READY.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6399_capability_learning_safety_audit as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _artifact(tmp_path: Path, *, write: bool = True) -> dict[str, Any]:
    return mod.run(
        date="20260813",
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        duration_s=1.0,
        test_exit_codes=_passing_exit_codes(),
        write=write,
    )


def _refresh(artifact: dict[str, Any]) -> dict[str, Any]:
    mod.refresh_terminal_fields(artifact)
    return artifact


def test_req_learn_6399_spec_declares_required_contract() -> None:
    """REQ-LEARN-6399: OpenSpec owns the audit contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-LEARN-6399") : text.index("REQ-LEARN-6383")]
    for token in (
        "SCENARIO-LEARN-6399-REGISTRATION",
        "SCENARIO-LEARN-6399-CLASS-PRESERVATION",
        "SCENARIO-LEARN-6399-LICENSE-BOUNDARY",
        "SCENARIO-LEARN-6399-TRANSACTION-BOUNDARY",
        "SCENARIO-LEARN-6399-READY",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        "verifier_is_oracle` SHALL be\nbare `false`",
    ):
        assert token in section
    normalized = " ".join(section.split())
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_learn_6399_registration_freezes_scope_before_conclusions(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6399-REGISTRATION: scope hashes precede conclusions."""

    artifact = _artifact(tmp_path)
    registration = artifact["audit_registration_path_hash_and_expected_scope"]
    preconditions = artifact["preconditions_checked"]
    matrix = artifact["present_absent_blocked_skipped_null_flagged_and_retired_artifact_matrix"]

    assert registration["registration_written_before_conclusion_reads"] is True
    assert Path(registration["path"]).is_file()
    assert registration["sha256"].startswith("sha256:")
    assert registration["expected_scope"]["task_ids"] == list(mod.EXPECTED_TASK_IDS)
    assert registration["expected_scope"]["model_ids"] == list(mod.MANDATED_MODEL_IDS)
    assert registration["expected_scope"]["llm_call_budget"] == 0
    assert registration["expected_scope"]["upstream_rerun_budget"] == 0
    assert registration["expected_scope"]["exact_checker_versions"]
    assert preconditions["registration_written_before_conclusion_reads"] is True
    assert preconditions["artifact_classes_frozen_before_conclusion_reads"] is True
    assert preconditions["all_preconditions_checked"] is True
    assert matrix["classification_before_conclusion_reads"] is True
    assert matrix["class_counts"]["positive"] >= 5
    assert matrix["missing_or_blocked_relabelled_clean_count"] == 0


def test_scenario_learn_6399_class_preservation_for_absent_and_blocked_inputs(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6399-CLASS-PRESERVATION: bad inputs stay bad."""

    blocked_path = tmp_path / "blocked_6398.json"
    blocked_path.write_text(
        json.dumps(
            {
                "status": "blocked_precondition",
                "honest_verdict": "blocked: fixture",
                "default_off_transactional_consumer_ready_score": 1.0,
            }
        ),
        encoding="utf-8",
    )
    artifact = mod.run(
        date="20260813",
        result_path=tmp_path / "artifact.json",
        duration_s=1.0,
        test_exit_codes=_passing_exit_codes(),
        upstream_path_overrides={
            "exp6394": tmp_path / "missing_6394.json",
            "exp6398": blocked_path,
        },
        write=False,
    )
    matrix = artifact["present_absent_blocked_skipped_null_flagged_and_retired_artifact_matrix"]
    gates = artifact["recomputed_readiness_scores_and_gates"]

    assert matrix["by_artifact"]["exp6394"]["evidence_class"] == "absent"
    assert matrix["by_artifact"]["exp6398"]["evidence_class"] == "blocked"
    assert matrix["class_counts"]["absent"] == 1
    assert matrix["class_counts"]["blocked"] == 1
    assert matrix["missing_or_blocked_relabelled_clean_count"] == 0
    assert gates["scores"]["exp6398_default_off_transactional_consumer_ready_score"] == 0.0
    assert gates["claim_gates"]["all_required_artifacts_clean"] is False
    assert artifact["public_factor_claim_eligibility"] is False
    assert artifact["status"] == "complete_null"


def test_scenario_learn_6399_license_boundary_stays_narrow(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6399-LICENSE-BOUNDARY: partial cells cannot pool."""

    artifact = _artifact(tmp_path)
    attacks = artifact[
        "family_model_harness_schema_license_fallback_abstention_and_pooling_attack_results"
    ]
    gates = artifact["recomputed_readiness_scores_and_gates"]
    findings = artifact["critical_major_and_minor_findings"]

    assert attacks["all_fail_closed"] is True
    assert attacks["narrow_license_cell_count"] == 4
    assert attacks["expected_model_family_cell_count"] == 9
    assert attacks["unlicensed_or_rejected_cell_count"] == 5
    assert attacks["fallback_approval_count"] == 0
    assert attacks["inherited_license_count"] == 0
    assert attacks["abstentions_pooled_as_success"] is False
    assert gates["claim_gates"]["narrow_license_blocks_public_general_claim"] is True
    assert gates["claim_gates"]["no_partial_cell_pooling"] is True
    assert artifact["utility_promotion_count"] == 0
    assert artifact["public_factor_claim_eligibility"] is False
    assert "narrow_license_scope" in findings["major"]


def test_scenario_learn_6399_transaction_and_consumer_attacks_fail_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6399-TRANSACTION-BOUNDARY: heads do not advance."""

    artifact = _artifact(tmp_path)
    transactions = artifact[
        "predecessor_effect_evidence_optional_stopping_atomicity_concurrency_restart_and_renewal_attack_results"
    ]
    consumer = artifact[
        "exact_checker_rollback_revocation_consumer_write_and_enablement_attack_results"
    ]

    assert set(transactions["attacks"]) == set(mod.TRANSACTION_ATTACKS)
    assert transactions["all_fail_closed"] is True
    assert transactions["failed_transaction_head_change_count"] == 0
    assert transactions["unauthorized_license_renewal_count"] == 0
    assert transactions["interrupted_atomic_write_survivor_count"] == 0
    assert transactions["all_commit_rows_predecessor_bound"] is True
    assert transactions["all_atomic_writes_recorded"] is True
    assert consumer["all_fail_closed"] is True
    assert consumer["rollback_underreach_count"] == 0
    assert consumer["revoked_descendant_survival_count"] == 0
    assert consumer["consumer_factor_write_count"] == 0
    assert consumer["production_enable_count"] == 0
    assert consumer["verifier_is_oracle_for_audit"] is False


def test_scenario_learn_6399_model_policy_and_substrate_checks(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-6399: model policy checks stay explicit."""

    artifact = _artifact(tmp_path)
    checks = artifact["model_policy_and_inference_substrate_checks"]

    assert checks["MODEL_SPECS_match_mandated_ids"] is True
    assert checks["cached_sota_receipts_present"] is True
    assert checks["embedded_tokenizer_use_only"] is True
    assert checks["autotokenizer_usage_count"] == 0
    assert checks["no_legacy_headline_result"] is True
    assert checks["accurate_inference_substrate"] is True
    assert checks["task_linked_gpu_evidence_where_applicable"] is True
    assert checks["inference_substrate_by_artifact"]["exp6399"] == mod.INFERENCE_SUBSTRATE
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False


def test_scenario_learn_6399_ready_gate_fails_closed_on_bad_fields(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6399-READY: malformed claim fields fail closed."""

    artifact = _artifact(tmp_path)

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_provenance"])
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert artifact["tests_run"]["all_passed"] is True
    assert mod.validate_artifact(artifact) is None

    negative_cases = {
        "nested_score": lambda row: row["recomputed_readiness_scores_and_gates"]["scores"].update(
            {"exp6395_held_factor_transport_license_ready_score": {"value": 1.0}}
        ),
        "nan_score": lambda row: row["recomputed_readiness_scores_and_gates"]["scores"].update(
            {"exp6396_capability_qualified_frontier_ready_score": math.nan}
        ),
        "bool_score": lambda row: row["recomputed_readiness_scores_and_gates"]["scores"].update(
            {"exp6397_transactional_continuous_self_learning_ready_score": True}
        ),
        "consumer_write": lambda row: row[
            "exact_checker_rollback_revocation_consumer_write_and_enablement_attack_results"
        ].update({"consumer_factor_write_count": 1}),
        "failed_test": lambda row: row["tests_run"]["exit_codes"].update(
            {mod.DEFAULT_TEST_COMMANDS[0]: 1}
        ),
        "verifier_oracle": lambda row: row.update({"verifier_is_oracle": True}),
    }
    for mutate in negative_cases.values():
        candidate = deepcopy(artifact)
        mutate(candidate)
        _refresh(candidate)
        assert candidate["public_factor_claim_eligibility"] is False
        assert candidate["status"] == "complete_null"

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)


def test_req_learn_6399_helpers_and_cli_paths(tmp_path: Path) -> None:
    """REQ-LEARN-6399: helper paths and CLI remain deterministic."""

    assert mod.sha256_text("x").startswith("sha256:")
    assert mod.sha256_file(tmp_path / "missing.json") is None
    assert mod.as_mapping([]) == {}
    assert mod.as_sequence("not-a-list") == ()
    assert mod.bare_finite_number(1.0) == 1.0
    assert mod.bare_finite_number(True) == 0.0
    assert mod.bare_finite_number({"value": 1.0}) == 0.0
    assert mod.bare_finite_number(float("inf")) == 0.0
    assert mod.evidence_class("missing") == "absent"
    assert mod.evidence_class("surprise") == "malformed"
    assert mod.unlicensed_records(
        {"exp6395": {}, "exp6398": {"unlicensed_cell_abstention_records": [{"cell_id": "x"}]}}
    ) == [{"cell_id": "x"}]
    assert mod.honest_verdict({"public_factor_claim_eligibility": True}).startswith(
        "complete_positive:"
    )
    with pytest.raises(ValueError, match="forced"):
        mod.require(False, "forced")

    malformed = tmp_path / "bad.json"
    malformed.write_text("{bad", encoding="utf-8")
    assert mod.read_json_object(malformed) is None
    assert mod.read_json_object(tmp_path / "missing.json") is None
    assert mod.path_receipt(tmp_path / "missing.json")["present"] is False
    written = tmp_path / "written.json"
    mod.write_json(written, {"ok": True})
    assert mod.read_json_object(written) == {"ok": True}
    assert mod.relative_or_absolute(REPO / "AGENTS.md") == "AGENTS.md"

    output = tmp_path / mod.RESULT_RELATIVE_PATH.name
    assert mod.main(["--date", "20260813", "--output", str(output), "--validate"]) == 0
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert artifact["duration_receipt_source"]["duration_source"] == "time.perf_counter"
