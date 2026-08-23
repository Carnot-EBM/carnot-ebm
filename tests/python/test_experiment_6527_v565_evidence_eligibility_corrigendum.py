"""Tests for Exp6527 V565 evidence eligibility corrigendum.

Spec refs: REQ-CAPSTONE-6527, SCENARIO-CAPSTONE-6527-ACTIVATION,
SCENARIO-CAPSTONE-6527-IMMUTABLE-ROWS,
SCENARIO-CAPSTONE-6527-LIVE-RECHECK,
SCENARIO-CAPSTONE-6527-RETIRED-DEPENDENCIES,
SCENARIO-CAPSTONE-6527-TERMINAL.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6527_v565_evidence_eligibility_corrigendum as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH
TESTS_RUN = [{"command": command, "exit_code": 0} for command in mod.DEFAULT_TEST_COMMANDS]


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """REQ-CAPSTONE-6527: build a temp artifact without touching history."""

    root = tmp_path_factory.mktemp("exp6527")
    return mod.build_artifact(
        repo_root=REPO,
        result_path=root / mod.RESULT_RELATIVE_PATH.name,
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
    )


def test_req_capstone_6527_spec_declares_contract() -> None:
    """REQ-CAPSTONE-6527: OpenSpec owns the Exp6527 contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-CAPSTONE-6527") : text.index("SCENARIO-CAPSTONE-4618")]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-CAPSTONE-6527-ACTIVATION",
        "SCENARIO-CAPSTONE-6527-IMMUTABLE-ROWS",
        "SCENARIO-CAPSTONE-6527-LIVE-RECHECK",
        "SCENARIO-CAPSTONE-6527-RETIRED-DEPENDENCIES",
        "SCENARIO-CAPSTONE-6527-TERMINAL",
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "`verifier_is_oracle=true`",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_capstone_6527_activation_schema_and_checksum(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-CAPSTONE-6527-ACTIVATION/TERMINAL: root schema is stable."""

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "complete_v565_evidence_root_eligible"
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["verdict_class"] is None
    assert artifact["v565_evidence_root_ready_score"] == 1.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert set(artifact["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)
    assert mod.validate_artifact(artifact) == []

    receipt = artifact["activation_manifest_receipt"]
    assert receipt["active_milestone"] == "2026.08.565"
    assert receipt["first_task_id"] == "exp6527-v565-evidence-eligibility-corrigendum"
    assert receipt["deliverable"] == mod.RESULT_RELATIVE_PATH.as_posix()
    assert receipt["roadmap_receipt"]["sha256"].startswith("sha256:")
    assert receipt["v565_roadmap_document_receipt"]["sha256"].startswith("sha256:")
    assert receipt["conductor_plan_row"]["status"] == "OK"
    assert receipt["conductor_activation_row"]["status"] == "OK"

    preconditions = artifact["preconditions_checked"]
    assert preconditions["active_milestone"] == "2026.08.565"
    assert preconditions["planned_milestone"] == "2026.08.565"
    assert preconditions["monotonic_clock_support"]["monotonic"]["monotonic"] is True
    assert preconditions["resources"]["cpu_count"] >= 1
    assert "python" in preconditions["tool_versions"]
    assert preconditions["git_status_initial"] == preconditions["git_status_final"]


def test_scenario_capstone_6527_immutable_row_recomputation(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-CAPSTONE-6527-IMMUTABLE-ROWS: adopted claims replay from rows."""

    task_rows = {row["task_id"]: row for row in artifact["v564_task_rows"]}
    assert set(task_rows) == set(mod.ADOPTED_TASKS)
    assert all(row["exists"] is True for row in task_rows.values())
    assert task_rows["6520"]["flagged_adversarial"] is True
    assert task_rows["6520"]["required_field"] == "safety_net_router_ready_score"
    assert task_rows["6526"]["verdict_class"] == "partial"

    recomputed = artifact["row_recomputation"]
    assert recomputed["6518"]["row_count"] == 126
    assert recomputed["6518"]["exact_answer_equality_passed"] is True
    assert recomputed["6518"]["best_arm_held_charged_benefit_units"] == 667
    assert recomputed["6519"]["source_aggregate_fields_used"] is False
    assert recomputed["6519"]["row_count"] == 136
    assert recomputed["6520"]["route_row_count"] == 144
    assert recomputed["6520"]["candidate_preservation_passed"] is True
    assert recomputed["6520"]["exact_answer_equality_passed"] is True
    assert recomputed["6520"]["best_learned_held_charged_benefit_units"] == 695
    assert recomputed["6520"]["held_benefit_beyond_best_structural_units"] == 28
    assert recomputed["6521"]["unsafe_admission_count"] == 0
    assert recomputed["6521"]["unsafe_use_count"] == 0
    assert recomputed["6522"]["row_count"] == 531
    assert recomputed["6522"]["charged_held_future_benefit_positive"] is True
    assert recomputed["6523"]["row_count"] == 280
    assert recomputed["6523"]["source_row_replay_matches"] is True

    attacks = {row["attack_id"]: row for row in recomputed["attack_rows"]}
    assert set(attacks) == set(mod.ATTACK_IDS)
    assert all(row["fail_closed"] is True for row in attacks.values())
    assert attacks["status_only_success"]["observed_value"] == "rows_recomputed"
    assert attacks["hidden_historical_file_edits"]["observed_value"] == "unchanged"


def test_scenario_capstone_6527_live_recheck_and_corrected_claims(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-CAPSTONE-6527-LIVE-RECHECK: Exp6520 disagreement is explicit."""

    historical = artifact["exp6520_historical_flag_receipt"]
    assert historical["historical_flagged_adversarial"] is True
    assert historical["historical_duration_s"] == 0.080421
    assert historical["historical_fields_rewritten"] is False
    assert {row["kind"] for row in historical["historical_corrigendum_pending"]} == {
        "DURATION_TOO_SHORT",
        "METHODOLOGY_MISSING",
    }

    live = artifact["live_adversarial_recheck_receipt"]
    assert live["artifact_path"] == mod.EXP6520_RELATIVE_PATH.as_posix()
    assert live["current_recheck_clean"] is True
    assert live["adversarial_verify"]["exit_code"] == 0
    assert live["row_consistency_lint"]["exit_code"] == 0
    assert live["adversarial_verify"]["stdout_sha256"].startswith("sha256:")

    duration = artifact["monotonic_duration_receipt"]
    assert duration["duration_floor_s"] == mod.EXP6520_VALIDATION_DURATION_FLOOR_S
    assert duration["credible_duration"] is True
    assert duration["validation_receipt"]["command"] == mod.EXP6520_VALIDATE_COMMAND
    assert duration["validation_receipt"]["exit_code"] == 1
    assert duration["historical_validation_disagreement_expected"] is True
    assert "required field set mismatch" in duration["historical_validation_errors"]

    claims = {row["claim_id"]: row for row in artifact["corrected_claim_eligibility_rows"]}
    assert claims["structural_headroom"]["corrected_eligibility"] == "eligible"
    assert claims["learned_router"]["corrected_eligibility"] == "corrected_eligible"
    assert claims["learned_router"]["historical_flag_preserved"] is True
    assert claims["learned_router"]["current_code_and_rows_clear_defect"] is True
    assert claims["historical_exp6520_artifact_self_validation"]["corrected_eligibility"] == (
        "ineligible_historical_file_preserved"
    )
    assert claims["arc_generalization"]["corrected_eligibility"] == "blocked_not_adopted"
    assert claims["hardware_continuity"]["corrected_eligibility"] == "blocked_not_adopted"


def test_scenario_capstone_6527_retired_dependency_and_aggregate_rows(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-CAPSTONE-6527-RETIRED-DEPENDENCIES: gates avoid retired IDs."""

    retired_rows = artifact["retired_dependency_attack_rows"]
    assert retired_rows
    assert all(row["retired_dependency_violation_count"] == 0 for row in retired_rows)
    assert all(row["direct_historical_read_is_hash_only"] is True for row in retired_rows)

    aggregate = artifact["aggregate_row_recomputation"]
    assert aggregate["v565_evidence_root_ready_score_from_rows"] == 1.0
    assert aggregate["verdict_class_from_rows"] is None
    assert aggregate["adopted_claim_count"] == 5
    assert aggregate["corrected_eligible_claim_count"] == 1
    assert aggregate["blocked_not_adopted_claim_count"] == 2
    assert aggregate["historical_self_validation_preserved_count"] == 1
    assert aggregate["per_unit_row_count"] == len(artifact["per_unit_rows"])
    assert artifact["gate_check_summary"]["all_root_checks_passed"] is True
    assert artifact["protected_files_unchanged"]["all_protected_files_unchanged"] is True

    row_types = {row["row_type"] for row in artifact["per_unit_rows"]}
    assert row_types >= {
        "v564_task",
        "corrected_claim",
        "retired_dependency_attack",
        "attack",
        "command_receipt",
        "protected_file",
    }


def test_scenario_capstone_6527_helper_defensive_paths(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-6527-LIVE-RECHECK: command receipts carry digests."""

    assert mod.sha256_file(tmp_path / "missing") == "missing"
    assert mod._package_version("definitely-not-a-real-package-name") == "not_installed"
    assert mod._parse_conductor_row("| too | short |") == {}
    assert mod._best_arm({}) == (None, 0)

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="expected JSON object"):
        mod.read_json_object(bad_json)

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("- item\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected YAML object"):
        mod.read_yaml_object(bad_yaml)

    fake_root = tmp_path / "fake-root"
    fake_root.mkdir()
    (fake_root / mod.ROADMAP_RELATIVE_PATH).write_text(
        "tasks:\n"
        "  - bad-list-entry\n"
        "  - id: retired-edge\n"
        "    requires:\n"
        "      - exp6507-exact-branch-counterfactual-dataset\n",
        encoding="utf-8",
    )
    retired_rows = mod.build_retired_dependency_attack_rows(fake_root)
    assert retired_rows[0]["retired_dependency_violation_count"] == 1

    receipt = mod.run_command_receipt(
        ".venv/bin/python -c \"print('receipt-ok')\"",
        cwd=REPO,
        duration_floor_s=0.0,
        code_hash="sha256:test",
    )
    assert receipt["exit_code"] == 0
    assert receipt["stdout_text"] == "receipt-ok\n"
    assert receipt["stderr_text"] == ""
    assert receipt["stdout_sha256"].startswith("sha256:")
    assert receipt["duration_s"] >= 0.0
    assert receipt["duration_floor_met"] is True


def test_scenario_capstone_6527_validation_rejects_bad_artifacts(
    artifact: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-CAPSTONE-6527-TERMINAL: malformed roots fail validation."""

    missing = deepcopy(artifact)
    missing.pop("activation_manifest_receipt")
    assert "missing required field: activation_manifest_receipt" in mod.validate_artifact(missing)

    bad_checksum = deepcopy(artifact)
    bad_checksum["v565_evidence_root_ready_score"] = 0.0
    assert "reproducibility_checksum mismatch" in mod.validate_artifact(bad_checksum)

    extra = deepcopy(artifact)
    extra["extra"] = True
    extra["reproducibility_checksum"] = mod.reproducibility_checksum(extra)
    assert "unexpected fields: extra" in mod.validate_artifact(extra)

    bad_principles = deepcopy(artifact)
    bad_principles["field_principles"] = {}
    bad_principles["reproducibility_checksum"] = mod.reproducibility_checksum(bad_principles)
    assert "field_principles mismatch" in mod.validate_artifact(bad_principles)

    bad_provenance = deepcopy(artifact)
    bad_provenance["field_provenance"] = {}
    bad_provenance["reproducibility_checksum"] = mod.reproducibility_checksum(bad_provenance)
    assert "field_provenance must cover required fields" in mod.validate_artifact(
        bad_provenance
    )

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    bad_substrate["reproducibility_checksum"] = mod.reproducibility_checksum(bad_substrate)
    assert "inference_substrate mismatch" in mod.validate_artifact(bad_substrate)

    bad_verifier = deepcopy(artifact)
    bad_verifier["verifier_is_oracle"] = False
    bad_verifier["reproducibility_checksum"] = mod.reproducibility_checksum(bad_verifier)
    assert "verifier_is_oracle must be true for receipt-only governance checks" in (
        mod.validate_artifact(bad_verifier)
    )

    bad_score = deepcopy(artifact)
    bad_score["v565_evidence_root_ready_score"] = 1.0
    bad_score["live_adversarial_recheck_receipt"]["current_recheck_clean"] = False
    bad_score["reproducibility_checksum"] = mod.reproducibility_checksum(bad_score)
    assert "ready score requires clean live recheck" in mod.validate_artifact(bad_score)

    bad_duration = deepcopy(artifact)
    bad_duration["monotonic_duration_receipt"]["credible_duration"] = False
    bad_duration["reproducibility_checksum"] = mod.reproducibility_checksum(bad_duration)
    assert "ready score requires credible duration receipt" in mod.validate_artifact(
        bad_duration
    )

    bad_retired = deepcopy(artifact)
    bad_retired["aggregate_row_recomputation"]["retired_dependency_violation_count"] = 1
    bad_retired["reproducibility_checksum"] = mod.reproducibility_checksum(bad_retired)
    assert "ready score requires zero retired dependency violations" in mod.validate_artifact(
        bad_retired
    )

    bad_protected = deepcopy(artifact)
    bad_protected["protected_files_unchanged"]["all_protected_files_unchanged"] = False
    bad_protected["reproducibility_checksum"] = mod.reproducibility_checksum(bad_protected)
    assert "ready score requires protected files unchanged" in mod.validate_artifact(
        bad_protected
    )

    bad_gate = deepcopy(artifact)
    bad_gate["gate_check_summary"]["all_root_checks_passed"] = False
    bad_gate["reproducibility_checksum"] = mod.reproducibility_checksum(bad_gate)
    assert "ready score requires passing gate_check_summary" in mod.validate_artifact(bad_gate)

    bad_honest = deepcopy(artifact)
    bad_honest["honest_verdict"] = "done"
    bad_honest["reproducibility_checksum"] = mod.reproducibility_checksum(bad_honest)
    assert "honest_verdict lacks terminal prefix" in mod.validate_artifact(bad_honest)

    bad_score_type = deepcopy(artifact)
    bad_score_type["v565_evidence_root_ready_score"] = 0.5
    bad_score_type["reproducibility_checksum"] = mod.reproducibility_checksum(bad_score_type)
    assert "v565_evidence_root_ready_score must be 0.0 or 1.0" in mod.validate_artifact(
        bad_score_type
    )

    bad_oracle = deepcopy(artifact)
    bad_oracle["verdict_class"] = "positive"
    bad_oracle["reproducibility_checksum"] = mod.reproducibility_checksum(bad_oracle)
    assert "verdict_class must not declare a positive scientific class" in mod.validate_artifact(
        bad_oracle
    )

    blocked = deepcopy(artifact)
    blocked["verdict_class"] = "blocked"
    blocked["v565_evidence_root_ready_score"] = 0.0
    blocked["gate_check_summary"]["failed_checks"] = []
    blocked["reproducibility_checksum"] = mod.reproducibility_checksum(blocked)
    assert "blocked verdict must populate failed gate_check_summary" in mod.validate_artifact(
        blocked
    )

    monkeypatch.setattr(mod, "validate_artifact", lambda _: ["forced"])
    command_receipts = {
        "adversarial_verify": artifact["live_adversarial_recheck_receipt"]["adversarial_verify"],
        "row_consistency_lint": artifact["live_adversarial_recheck_receipt"][
            "row_consistency_lint"
        ],
        "exp6520_validation": artifact["monotonic_duration_receipt"]["validation_receipt"],
    }
    with pytest.raises(ValueError, match="forced"):
        mod.build_artifact(
            repo_root=REPO,
            result_path=tmp_path / "forced.json",
            write=False,
            duration_s=1.0,
            tests_run=TESTS_RUN,
            command_receipts=command_receipts,
        )

    monkeypatch.undo()
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert mod.main(["--validate", "--result-path", str(path)]) == 0

    monkeypatch.setattr(mod, "build_artifact", lambda **_: {"ok": True})
    assert mod.run(date="20260823", result_path=path) == {"ok": True}
    assert mod.main(["--date", "20260823", "--result-path", str(path)]) == 0

    monkeypatch.setattr(mod, "validate_artifact", lambda _: ["bad"])
    with pytest.raises(ValueError, match="bad"):
        mod.main(["--validate", "--result-path", str(path)])
