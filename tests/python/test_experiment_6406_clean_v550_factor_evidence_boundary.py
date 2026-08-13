"""Tests for Exp6406 clean V550 factor evidence boundary.

Spec refs: REQ-LEARN-6406, SCENARIO-LEARN-6406-REGISTRATION,
SCENARIO-LEARN-6406-INCLUSION, SCENARIO-LEARN-6406-RECOMPUTE,
SCENARIO-LEARN-6406-ATTACKS, SCENARIO-LEARN-6406-LEDGER,
SCENARIO-LEARN-6406-READY.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6406_clean_v550_factor_evidence_boundary as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _artifact(tmp_path: Path, *, write: bool = True) -> dict[str, object]:
    return mod.run(
        date="20260813",
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        duration_s=1.0,
        test_exit_codes=_passing_exit_codes(),
        write=write,
    )


def _refresh(report: dict[str, object]) -> dict[str, object]:
    mod.refresh_terminal_fields(report)
    return report


def test_req_learn_6406_spec_declares_required_contract() -> None:
    """REQ-LEARN-6406: OpenSpec owns the clean-boundary contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-LEARN-6406") : text.index("REQ-LEARN-6383")]

    for token in (
        "SCENARIO-LEARN-6406-REGISTRATION",
        "SCENARIO-LEARN-6406-INCLUSION",
        "SCENARIO-LEARN-6406-RECOMPUTE",
        "SCENARIO-LEARN-6406-ATTACKS",
        "SCENARIO-LEARN-6406-LEDGER",
        "SCENARIO-LEARN-6406-READY",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        "`clean_factor_evidence_boundary_ready_score=1.0`",
        "`public_factor_claim_eligibility=false`",
    ):
        assert token in section

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_learn_6406_registration_freezes_scope_before_conclusions(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6406-REGISTRATION: registration precedes conclusions."""

    artifact = _artifact(tmp_path)
    registration = artifact["audit_registration_path_hash_and_expected_scope"]
    matrix = artifact["v550_artifact_hash_verdict_conductor_duration_and_flag_matrix"]
    preconditions = artifact["preconditions_checked"]

    assert registration["registration_written_before_conclusion_reads"] is True
    assert registration["expected_scope"]["task_ids"] == list(mod.EXPECTED_TASK_IDS)
    assert registration["expected_scope"]["model_ids"] == list(mod.MANDATED_MODEL_IDS)
    assert registration["expected_scope"]["license_record_count"] == 4
    assert registration["expected_scope"]["llm_call_budget"] == 0
    assert registration["expected_scope"]["upstream_rerun_budget"] == 0
    assert registration["expected_scope"]["source_files"]
    assert Path(registration["path"]).is_file()
    assert registration["sha256"].startswith("sha256:")
    assert preconditions["registration_written_before_conclusion_reads"] is True
    assert preconditions["all_preconditions_checked"] is True
    assert matrix["classes_frozen_before_conclusion_reads"] is True
    assert matrix["rows"]["exp6394"]["artifact_sha256"].startswith("sha256:")
    assert matrix["rows"]["exp6394"]["duration_s"] > 0
    assert matrix["rows"]["exp6394"]["conductor_outcome"] == "OK"
    assert matrix["rows"]["exp6385"]["adversarial_flag"] is True
    assert matrix["rows"]["exp6385"]["terminal_class"] == "flagged"
    assert matrix["rows"]["exp6399"]["terminal_class"] == "null"
    assert matrix["rows"]["exp6403"]["terminal_class"] == "complete"


def test_scenario_learn_6406_inclusion_and_preservation_receipts(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6406-INCLUSION: only clean V550 factor rows enter."""

    artifact = _artifact(tmp_path)
    included = artifact["included_clean_artifact_records"]
    excluded = artifact[
        "excluded_nonclean_blocked_null_absent_unlicensed_rejected_and_flagged_records"
    ]
    excluded_ids = {row["id"] for row in excluded}

    assert [row["task_id"] for row in included] == [
        "exp6394",
        "exp6395",
        "exp6396",
        "exp6397",
        "exp6398",
    ]
    assert all(row["clean"] is True for row in included)
    assert all(row["hash_complete"] is True for row in included)
    assert all(row["source_bound"] is True for row in included)
    assert all(row["inside_declared_scope"] is True for row in included)

    assert {"exp6385", "exp6399", "exp6403"} <= excluded_ids
    assert "cell:unsloth--qwen3.6-35b-a3b-gguf::threshold_guard" in excluded_ids
    assert "cell:unsloth--gemma-4-31b-it-gguf::conservation_guard" in excluded_ids
    assert "cell:unsloth--gemma-4-26b-a4b-it-gguf::threshold_guard" in excluded_ids
    assert artifact["exp6385_preservation_receipt"]["preserved_as"] == "flagged_nonclean"
    assert artifact["exp6385_preservation_receipt"]["included_in_clean_boundary"] is False
    assert artifact["exp6385_preservation_receipt"]["rerun_or_repaired"] is False
    assert artifact["exp6399_preservation_receipt"]["preserved_as"] == "null_public_audit"
    assert artifact["exp6399_preservation_receipt"]["public_factor_claim_eligibility"] is False
    assert artifact["upstream_artifacts_modified"] is False


def test_scenario_learn_6406_recomputes_narrow_internal_claims(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6406-RECOMPUTE: public and universal gates stay false."""

    artifact = _artifact(tmp_path)
    states = artifact[
        "recomputed_narrow_harness_license_frontier_learning_consumer_and_safety_states"
    ]

    assert states["harness_state"]["ready"] is True
    assert states["license_state"]["ready"] is True
    assert states["license_state"]["licensed_cell_count"] == 4
    assert states["license_state"]["unlicensed_or_rejected_cell_count"] == 5
    assert states["frontier_state"]["ready"] is True
    assert states["frontier_state"]["delta_verified_future_exact_yield"] > 0
    assert states["transactional_learning_state"]["ready"] is True
    assert states["transactional_learning_state"]["commit_count"] == 2
    assert states["consumer_state"]["ready"] is True
    assert states["consumer_state"]["production_enable_count"] == 0
    assert states["safety_state"]["exp6399_public_audit_null_preserved"] is True
    assert states["safety_state"]["exp6385_flagged_preserved"] is True
    assert artifact["universal_support_claimed"] is False
    assert artifact["public_factor_claim_eligibility"] is False
    assert artifact["allowed_internal_claims"]
    assert any("narrow" in claim for claim in artifact["allowed_internal_claims"])
    assert any("public" in claim for claim in artifact["forbidden_claims"])
    assert any("universal" in claim for claim in artifact["forbidden_claims"])


def test_scenario_learn_6406_attacks_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6406-ATTACKS: attacks do not widen the boundary."""

    artifact = _artifact(tmp_path)
    attacks = artifact[
        "substitution_laundering_date_model_family_license_sidecar_conductor_and_flag_attack_matrix"
    ]

    assert set(attacks["attacks"]) == set(mod.ATTACKS)
    assert attacks["all_fail_closed"] is True
    assert attacks["included_row_added_by_attack_count"] == 0
    assert attacks["excluded_row_suppressed_by_attack_count"] == 0
    assert attacks["public_claim_enabled_by_attack_count"] == 0
    assert attacks["results"]["flagged_input_omission"]["decision"] == "reject"
    assert attacks["results"]["missing_sidecar"]["decision"] == "reject"
    assert artifact["clean_factor_evidence_boundary_ready_score"] == 1.0


def test_scenario_learn_6406_claim_ledger_binds_boundary(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6406-LEDGER: ledger records the exact boundary hash."""

    artifact = _artifact(tmp_path)
    ledger = artifact["claim_ledger_path_hash_and_rows"]
    boundary_hash = artifact["preconditions_checked"]["evidence_boundary_hash"]

    assert Path(ledger["path"]).is_file()
    assert ledger["sha256"].startswith("sha256:")
    assert ledger["row_count"] == len(ledger["rows"])
    assert ledger["rows"][0]["evidence_boundary_hash"] == boundary_hash
    assert ledger["rows"][0]["included_artifact_hashes"]
    assert ledger["rows"][0]["excluded_artifact_hashes"]
    assert ledger["rows"][0]["allowed_internal_claims"] == artifact["allowed_internal_claims"]
    assert ledger["rows"][0]["forbidden_claims"] == artifact["forbidden_claims"]
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)


def test_scenario_learn_6406_ready_gate_fails_closed(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6406-READY: any widened claim zeroes readiness."""

    artifact = _artifact(tmp_path)

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) == set(artifact)
    assert artifact["verifier_is_oracle"] is False
    assert artifact["tests_run"]["all_passed"] is True
    assert mod.validate_report(artifact) == []
    for key in (
        "clean_factor_evidence_boundary_ready_score",
        "universal_support_claimed",
        "public_factor_claim_eligibility",
        "include.exp6394",
        "exclude.exp6385",
        "exclude.exp6399",
    ):
        assert key in artifact["field_principles"]

    cases = {
        "nonclean_included": lambda row: row["included_clean_artifact_records"][0].update(
            {"clean": False}
        ),
        "public_claim": lambda row: row.update({"public_factor_claim_eligibility": True}),
        "universal_claim": lambda row: row.update({"universal_support_claimed": True}),
        "attack_success": lambda row: row[
            "substitution_laundering_date_model_family_license_sidecar_conductor_and_flag_attack_matrix"
        ].update({"included_row_added_by_attack_count": 1}),
        "failed_test": lambda row: row["tests_run"]["exit_codes"].update(
            {mod.DEFAULT_TEST_COMMANDS[0]: 1}
        ),
        "oracle": lambda row: row.update({"verifier_is_oracle": True}),
    }
    for mutate in cases.values():
        candidate = deepcopy(artifact)
        mutate(candidate)
        _refresh(candidate)
        assert candidate["clean_factor_evidence_boundary_ready_score"] == 0.0
        assert candidate["status"] == "complete_null"

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum mismatch" in mod.validate_report(bad_checksum)


def test_req_learn_6406_helpers_and_cli_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-6406: helpers and CLI paths stay deterministic."""

    assert mod.sha256_text("x").startswith("sha256:")
    assert mod.sha256_file(tmp_path / "missing.json") is None
    assert mod.as_mapping([]) == {}
    assert mod.as_sequence("x") == ()
    assert mod.bare_finite_number(1.0) == 1.0
    assert mod.bare_finite_number(True) == 0.0
    assert mod.bare_finite_number({"value": 1.0}) == 0.0
    assert mod.bare_finite_number(float("inf")) == 0.0
    assert mod.terminal_class_for_missing_or_bad("missing") == "absent"
    assert mod._source_bound_for_task("unknown", {}) is False
    assert mod._inside_declared_scope("unknown", {}) is False

    malformed = tmp_path / "bad.json"
    malformed.write_text("{bad", encoding="utf-8")
    assert mod.read_json_object(malformed) is None
    assert mod.read_json_object(tmp_path / "missing.json") is None
    assert mod.path_receipt(tmp_path / "missing.json")["present"] is False
    assert mod.relative_or_absolute(REPO / "AGENTS.md") == "AGENTS.md"

    def fail_replace(_src: object, _dst: object) -> None:
        raise RuntimeError("replace failed")

    monkeypatch.setattr(mod.os, "replace", fail_replace)
    with pytest.raises(RuntimeError, match="replace failed"):
        mod.atomic_write_local_text(tmp_path / "cleanup.txt", "x")
    assert not list(tmp_path.glob(".cleanup.txt.*.tmp"))
    monkeypatch.undo()

    missing_sidecar_rows = mod.excluded_records(
        {},
        {"rows": {"exp6385": {}, "exp6399": {}, "exp6403": {}}},
        {"exp6394": [{"present": False, "sha256": None, "path": "missing"}]},
    )
    assert any(row["record_type"] == "missing_sidecar" for row in missing_sidecar_rows)

    output = tmp_path / mod.RESULT_RELATIVE_PATH.name
    assert mod.main(["--date", "20260813", "--output", str(output), "--validate"]) == 0
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert artifact["honest_verdict"].startswith("complete:")

    for mutate, message in (
        (lambda row: row.update({"verifier_is_oracle": True}), "verifier_is_oracle must be false"),
        (
            lambda row: row.update({"inference_substrate": "live_llm"}),
            "inference_substrate mismatch",
        ),
        (
            lambda row: row.update({"public_factor_claim_eligibility": True}),
            "public_factor_claim_eligibility must be false",
        ),
        (
            lambda row: row.update({"universal_support_claimed": True}),
            "universal_support_claimed must be false",
        ),
        (
            lambda row: row.update({"upstream_artifacts_modified": True}),
            "upstream_artifacts_modified must be false",
        ),
        (lambda row: row.update({"honest_verdict": "ok"}), "honest_verdict lacks accepted prefix"),
        (
            lambda row: row["field_principles"].pop("universal_support_claimed"),
            "missing field_principles entry: universal_support_claimed",
        ),
    ):
        candidate = deepcopy(artifact)
        mutate(candidate)
        candidate["reproducibility_checksum"] = mod.payload_checksum(candidate)
        assert message in mod.validate_report(candidate)

    candidate = deepcopy(artifact)
    del candidate["status"]
    candidate["reproducibility_checksum"] = mod.payload_checksum(candidate)
    assert "missing required field: status" in mod.validate_report(candidate)

    candidate = deepcopy(artifact)
    candidate["extra"] = True
    candidate["reproducibility_checksum"] = mod.payload_checksum(candidate)
    assert "extra top-level field: extra" in mod.validate_report(candidate)

    monkeypatch.setattr(mod, "validate_report", lambda _report: ["bad"])
    with pytest.raises(ValueError, match="bad"):
        mod.run(
            date="20260813",
            result_path=tmp_path / "invalid.json",
            test_exit_codes=_passing_exit_codes(),
            write=False,
            validate=True,
        )
