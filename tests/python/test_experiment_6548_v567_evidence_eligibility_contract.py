"""Tests for Exp6548 V567 evidence eligibility contract.

Spec refs: REQ-REPORT-6548, SCENARIO-REPORT-6548-ADDITIVE,
SCENARIO-REPORT-6548-IMPORT, SCENARIO-REPORT-6548-GATES,
SCENARIO-REPORT-6548-MODEL-HARDWARE, SCENARIO-REPORT-6548-SCHEMA.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6548_v567_evidence_eligibility_contract as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH
TESTS_RUN = [{"command": "focused-exp6548", "exit_code": 0}]


def _fake_check_results(*, exp6541_clean: bool = False) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for artifact in mod.V566_ARTIFACTS:
        flags: list[dict[str, Any]] = []
        exit_code = 0
        if artifact.exp_id == "exp6541" and not exp6541_clean:
            flags = [
                {
                    "kind": "DURATION_TOO_SHORT",
                    "severity": "critical",
                    "detail": "duration_s=52.54 below live model floor and model receipt incomplete",
                }
            ]
            exit_code = 1
        results[artifact.exp_id] = {
            "adversarial": {
                "command": f"adversarial {artifact.exp_id}",
                "exit_code": exit_code,
                "flag_count": len(flags),
                "max_severity": 2 if flags else -1,
                "flags": flags,
            },
            "row_consistency": {
                "command": f"row-lint {artifact.exp_id}",
                "exit_code": 0,
                "status": "ok",
                "findings": [],
            },
        }
    return results


@pytest.fixture(scope="module")
def artifact() -> dict[str, Any]:
    """REQ-REPORT-6548: build from checked-in V566 artifacts with fake live checks."""

    return mod.build_artifact(
        repo_root=REPO,
        result_path=Path("/tmp/experiment_6548_test_result.json"),
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        check_results=_fake_check_results(),
        run_date="20260823",
    )


def test_req_report_6548_spec_declares_required_contract() -> None:
    """REQ-REPORT-6548: OpenSpec owns the Exp6548 evidence contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-REPORT-6548") :]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-REPORT-6548-ADDITIVE",
        "SCENARIO-REPORT-6548-IMPORT",
        "SCENARIO-REPORT-6548-GATES",
        "SCENARIO-REPORT-6548-MODEL-HARDWARE",
        "SCENARIO-REPORT-6548-SCHEMA",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_report_6548_additive_exp6541_quarantine(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-REPORT-6548-ADDITIVE: Exp6541 stays visible but not imported."""

    assert mod.validate_artifact(artifact) == []
    rows = {row["exp_id"]: row for row in artifact["v566_artifact_eligibility_rows"]}
    exp6541 = rows["exp6541"]

    assert len(rows) == 7
    assert exp6541["expected_path"] == mod.V566_ARTIFACTS[0].relative_path.as_posix()
    assert exp6541["exists"] is True
    assert exp6541["sha256"].startswith("sha256:")
    assert exp6541["readiness_field"] == "v566_direct_source_ready_score"
    assert exp6541["readiness_score"] == 1.0
    assert exp6541["live_critical_flag_count"] == 1
    assert exp6541["model_receipt_gap_open"] is True
    assert exp6541["eligible_for_clean_import"] is False
    assert exp6541["disposition"] == "quarantined_not_imported"
    assert artifact["exp6541_disposition"]["disposition"] == "quarantined_not_imported"
    assert "DURATION_TOO_SHORT" in artifact["exp6541_disposition"]["unresolved_reasons"]


def test_scenario_report_6548_import_ledger_content_addresses_clean_v566(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-REPORT-6548-IMPORT: Exp6542-Exp6547 support the external root."""

    ledger = artifact["clean_v566_import_ledger"]
    imported = {row["exp_id"]: row for row in ledger["imported_rows"]}

    assert ledger["imported_exp_ids"] == [
        "exp6542",
        "exp6543",
        "exp6544",
        "exp6545",
        "exp6546",
        "exp6547",
    ]
    assert "exp6541" not in imported
    assert artifact["v566_external_transfer_eligible_score"] == 1.0
    for exp_id, row in imported.items():
        assert row["sha256"].startswith("sha256:")
        assert row["eligible_for_clean_import"] is True
        assert row["row_consistency_status"] == "ok"
        assert row["live_critical_flag_count"] == 0
        assert row["readiness_score"] == 1.0, exp_id


def test_scenario_report_6548_gates_exact_field_contract(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-REPORT-6548-GATES: structured gates resolve in-roadmap fields."""

    rows = artifact["v567_gate_contract_rows"]
    assert rows
    assert all(row["upstream_in_v567"] for row in rows)
    assert all(row["artifact_field_declared_by_upstream"] for row in rows)
    assert all(row["retired_upstream"] is False for row in rows)

    key = {(row["task_id"], row["upstream"], row["artifact_field"]) for row in rows}
    assert (
        "exp6549-production-safety-net-adapter",
        "exp6548-v567-evidence-eligibility-contract",
        "v567_evidence_contract_ready_score",
    ) in key
    assert (
        "exp6553-prospective-sota-continuous-self-learning",
        "exp6548-v567-evidence-eligibility-contract",
        "v566_external_transfer_eligible_score",
    ) in key
    assert artifact["gate_check_summary"]["task_field_gate_contract_closed"] is True


def test_scenario_report_6548_model_hardware_and_architecture_scope(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-REPORT-6548-MODEL-HARDWARE: no legacy or retired scope reopens."""

    contract = artifact["v567_model_and_hardware_contract"]
    architecture = artifact["architecture_freshness_receipt"]

    assert contract["mandated_model_ids"] == list(mod.MANDATED_MODEL_IDS)
    assert contract["headline_model_tasks"] == [
        "exp6553-prospective-sota-continuous-self-learning",
        "exp6556-sota-constraint-saturation-intervention-ab",
    ]
    assert contract["legacy_models_headline_excluded"] is True
    assert contract["gguf_loader_rule"] == "cached_sota_pair(gpu_indices=(0, 1)) plus llama.cpp"
    assert contract["arc_no_solve_rule"]["task_id"] == "exp6558-arc-live-redirect-ledger"
    assert contract["arc_no_solve_rule"]["no_game_or_level_solve_claim"] is True
    assert contract["gatemate_receipt_rule"]["task_id"] == "exp6559-gatemate-changed-state-continuity"
    assert contract["gatemate_receipt_rule"]["requires_receipt_newer_than_exp6525"] is True
    assert contract["retired_scope_isolation"]["schema_supported_constraintir_reopened"] is False
    assert architecture["last_reconciled"] == "2026-07-03"
    assert architecture["age_days_at_planning"] == 51
    assert architecture["checked_against_current_code"] is True


def test_scenario_report_6548_schema_write_and_validation_edges(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6548-SCHEMA: artifact output is checksummed and validated."""

    result_path = tmp_path / "exp6548.json"
    written = mod.build_artifact(
        repo_root=REPO,
        result_path=result_path,
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        check_results=_fake_check_results(),
        run_date="20260823",
    )
    loaded = json.loads(result_path.read_text(encoding="utf-8"))

    assert loaded["reproducibility_checksum"] == written["reproducibility_checksum"]
    assert written["reproducibility_checksum"] == mod.reproducibility_checksum(written)
    assert written["status"] == "complete_v567_evidence_eligibility_contract_ready"
    assert written["verdict_class"] is None
    assert written["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert written["verifier_is_oracle"] is True
    assert written["v567_evidence_contract_ready_score"] == 1.0
    assert set(written["field_provenance"]) >= set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert set(written["field_principles"]) >= set(mod.REQUIRED_ARTIFACT_FIELDS)

    validations = [
        ("delete", "status", "missing required fields"),
        ("set", ("honest_verdict", "ready"), "honest_verdict lacks terminal prefix"),
        ("set", ("inference_substrate", "live_llm_inference"), "inference_substrate mismatch"),
        ("set", ("verifier_is_oracle", False), "verifier_is_oracle must be true"),
        ("set", ("v567_evidence_contract_ready_score", 1.0), "Exp6541 must remain quarantined"),
        ("set", ("v566_external_transfer_eligible_score", 1.0), "external transfer score must derive"),
        ("set", ("protected_files_unchanged", {"all_unchanged": False}), "protected files changed"),
        ("set", ("reproducibility_checksum", "sha256:bad"), "reproducibility_checksum mismatch"),
    ]
    for mode, spec, expected in validations:
        bad = deepcopy(written)
        if mode == "delete":
            del bad[spec]
        else:
            key, value = spec
            if expected == "Exp6541 must remain quarantined":
                bad["v566_artifact_eligibility_rows"][0]["eligible_for_clean_import"] = True
            elif expected == "external transfer score must derive":
                bad["v566_artifact_eligibility_rows"][1]["eligible_for_clean_import"] = False
            else:
                bad[key] = value
        if expected != "reproducibility_checksum mismatch":
            bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
        assert any(expected in error for error in mod.validate_artifact(bad))

    alias = deepcopy(written)
    alias["clean_v566_import_ledger"]["imported_rows"][0]["sha256"] = "sha256:alias"
    alias["reproducibility_checksum"] = mod.reproducibility_checksum(alias)
    assert any("clean import ledger hash alias" in error for error in mod.validate_artifact(alias))

    missing_field = deepcopy(written)
    missing_field["v567_gate_contract_rows"][0]["artifact_field_declared_by_upstream"] = False
    missing_field["reproducibility_checksum"] = mod.reproducibility_checksum(missing_field)
    assert any("gate contract has undeclared field" in error for error in mod.validate_artifact(missing_field))

    bad_class = deepcopy(written)
    bad_class["verdict_class"] = "positive"
    bad_class["reproducibility_checksum"] = mod.reproducibility_checksum(bad_class)
    assert "verdict_class is outside closed class" in mod.validate_artifact(bad_class)

    bad_provenance = deepcopy(written)
    bad_provenance["field_provenance"] = {}
    bad_provenance["reproducibility_checksum"] = mod.reproducibility_checksum(bad_provenance)
    assert "field_provenance must cover required fields" in mod.validate_artifact(bad_provenance)

    bad_principles = deepcopy(written)
    bad_principles["field_principles"] = {}
    bad_principles["reproducibility_checksum"] = mod.reproducibility_checksum(bad_principles)
    assert "field_principles must cover required fields" in mod.validate_artifact(bad_principles)

    bad_gate = deepcopy(written)
    bad_gate["v567_gate_contract_rows"] = [
        "not-a-row",
        {
            "artifact_field_declared_by_upstream": True,
            "upstream_in_v567": False,
            "retired_upstream": True,
        },
    ]
    bad_gate["reproducibility_checksum"] = mod.reproducibility_checksum(bad_gate)
    gate_errors = mod.validate_artifact(bad_gate)
    assert "gate contract row must be a mapping" in gate_errors
    assert "gate contract has out-of-roadmap upstream" in gate_errors
    assert "gate contract has retired upstream" in gate_errors

    ready_failed = deepcopy(written)
    ready_failed["gate_check_summary"]["failed_checks"] = ["should_not_open"]
    ready_failed["reproducibility_checksum"] = mod.reproducibility_checksum(ready_failed)
    assert "ready score cannot be open with failed checks" in mod.validate_artifact(ready_failed)

    exp6541_imported = deepcopy(written)
    exp6541_imported["v566_artifact_eligibility_rows"][0]["disposition"] = "clean_imported"
    exp6541_imported["reproducibility_checksum"] = mod.reproducibility_checksum(exp6541_imported)
    assert any("Exp6541 must remain quarantined" in error for error in mod.validate_artifact(exp6541_imported))


def test_scenario_report_6548_missing_clean_import_yields_partial(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6548-IMPORT: incomplete clean evidence cannot open V567."""

    payloads = mod.load_v566_payloads(REPO)
    payloads["exp6544"] = deepcopy(payloads["exp6544"])
    del payloads["exp6544"]["external_structural_headroom_ready_score"]

    partial = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "partial.json",
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        check_results=_fake_check_results(),
        input_payloads=payloads,
        run_date="20260823",
    )

    assert partial["status"] == "partial_v567_evidence_eligibility_contract"
    assert partial["verdict_class"] == "partial"
    assert partial["v567_evidence_contract_ready_score"] == 0.0
    assert partial["v566_external_transfer_eligible_score"] == 0.0
    assert "exp6544_ready_field_present" in partial["gate_check_summary"]["failed_checks"]
    assert mod.validate_artifact(partial) == []


def test_scenario_report_6548_clean_exp6541_can_be_disposed_but_not_required(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6548-ADDITIVE: Exp6541 never gates external transfer."""

    payloads = mod.load_v566_payloads(REPO)
    payloads["exp6541"] = deepcopy(payloads["exp6541"])
    payloads["exp6541"]["model_cache_resolution_rows"] = [
        {"model_path_exists": True, "sha256": "sha256:" + "a" * 64}
    ]
    clean = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "clean-exp6541.json",
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        check_results=_fake_check_results(exp6541_clean=True),
        input_payloads=payloads,
        run_date="20260823",
    )

    assert clean["exp6541_disposition"]["disposition"] == "eligible_but_not_import_required"
    assert clean["v566_external_transfer_eligible_score"] == 1.0
    assert "exp6541" not in clean["clean_v566_import_ledger"]["imported_exp_ids"]
    assert mod.validate_artifact(clean) == []


def test_scenario_report_6548_helper_edges_are_deterministic() -> None:
    """SCENARIO-REPORT-6548-SCHEMA: helper edge cases fail closed."""

    assert mod.sha256_bytes(b"x").startswith("sha256:")
    assert mod.sha256_file(Path("/tmp/definitely-missing-exp6548-file")) == "missing"
    assert mod.default_v566_paths(REPO)["exp6541"].name.startswith("experiment_6541")
    assert mod._command_text(["a b", "c"]) == "'a b' c"
    assert mod._python_executable(REPO).endswith("python")
    assert mod._stamped_flags({"corrigendum_pending": [{"kind": "X"}, "skip"]}) == [
        {"kind": "X"}
    ]
    assert mod._exp6541_model_receipt_gap({}) is True

    row = mod._eligibility_row(
        mod.V566_ARTIFACTS[1],
        repo_root=REPO,
        payload={},
        check_result={
            "adversarial": {
                "command": "adv",
                "exit_code": 1,
                "flags": [{"kind": "OTHER", "severity": "critical", "detail": "x"}],
            },
            "row_consistency": {
                "command": "row",
                "exit_code": 1,
                "status": "findings",
                "findings": ["hard"],
            },
        },
    )
    assert row["disposition"] == "not_imported_failed_checks"
    assert row["unresolved_reasons"] == [
        "live_critical_flags",
        "row_consistency_blocking",
        "exp6542_ready_field_present",
    ]

    retired = mod._retired_experiment_ids(
        {
            "retired": [None, {"id": "retired-a", "experiment_ids": ["retired-b"]}],
            "retired_experiments": "not-a-list",
        }
    )
    assert retired == {"retired-a", "retired-b"}

    gate_rows = mod._gate_contract_rows(
        {"tasks": ["skip", {"id": "task", "gated_on": ["skip-gate"]}]},
        {"upstream": {"field"}},
        {"upstream"},
    )
    assert gate_rows == []

    summary = mod._gate_check_summary(
        rows=[
            {
                "exp_id": "exp6542",
                "readiness_field_present": False,
                "eligible_for_clean_import": False,
                "expected_path": "missing.json",
            }
        ],
        ledger={"all_expected_clean_imported": False, "imported_exp_ids": []},
        gate_rows=[
            {
                "task_id": "task",
                "upstream": "retired-upstream",
                "artifact_field": "bad_field",
                "upstream_in_v567": False,
                "artifact_field_declared_by_upstream": False,
                "retired_upstream": True,
            }
        ],
        architecture={"checked_against_current_code": False},
        model_contract={
            "all_headline_model_tasks_present": False,
            "headline_model_tasks": [],
            "retired_scope_isolation": {"schema_supported_constraintir_reopened": True},
        },
        protected={"all_unchanged": False, "changed_paths": ["research-roadmap.yaml"]},
    )
    assert summary["all_gates_passed"] is False
    assert {
        "exp6542_ready_field_present",
        "exp6542_clean_import_eligible",
        "gate_upstream_in_v567",
        "gate_artifact_field_declared",
        "gate_retired_upstream",
        "clean_v566_import_ledger_complete",
        "architecture_checked_against_current_code",
        "headline_model_tasks_present",
        "schema_supported_constraintir_not_reopened",
        "protected_files_unchanged",
    } <= set(summary["failed_checks"])

    assert mod._status_and_verdict(False, 1.0, ["gate"]) == (
        "partial_v567_evidence_eligibility_contract",
        "partial_v567_evidence_eligibility_contract: usable V566 subset exists but one or more V567 evidence, field, gate, model, hardware, or scope checks failed",
        "partial",
    )
    assert mod._status_and_verdict(False, 0.0, []) == (
        "blocked_v567_evidence_eligibility_contract",
        "blocked_v567_evidence_eligibility_contract: no clean V566 input set was available",
        "blocked",
    )
