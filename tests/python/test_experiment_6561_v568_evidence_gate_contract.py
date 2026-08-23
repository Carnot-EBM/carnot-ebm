"""Tests for Exp6561 V568 evidence and gate contract.

Spec refs: REQ-REPORT-6561, SCENARIO-REPORT-6561-IMPORT,
SCENARIO-REPORT-6561-GATES, SCENARIO-REPORT-6561-PRIOR-FAILURE,
SCENARIO-REPORT-6561-MODEL-HARDWARE, SCENARIO-REPORT-6561-SCHEMA.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6561_v568_evidence_gate_contract as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH
TESTS_RUN = [{"command": "focused-exp6561", "exit_code": 0}]


def _fake_check_results() -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for artifact in mod.V567_ARTIFACTS:
        results[artifact.exp_id] = {
            "adversarial": {
                "command": f"adversarial {artifact.exp_id}",
                "exit_code": 0,
                "flag_count": 0,
                "max_severity": -1,
                "flags": [],
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
    """REQ-REPORT-6561: build from checked-in V567 artifacts with fake live checks."""

    return mod.build_artifact(
        repo_root=REPO,
        result_path=Path("/tmp/experiment_6561_test_result.json"),
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        check_results=_fake_check_results(),
        run_date="20260823",
    )


def test_req_report_6561_spec_declares_required_contract() -> None:
    """REQ-REPORT-6561: OpenSpec owns the V568 evidence contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-REPORT-6561") :]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-REPORT-6561-IMPORT",
        "SCENARIO-REPORT-6561-GATES",
        "SCENARIO-REPORT-6561-PRIOR-FAILURE",
        "SCENARIO-REPORT-6561-MODEL-HARDWARE",
        "SCENARIO-REPORT-6561-SCHEMA",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_report_6561_import_rows_classify_v567_boundary(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-REPORT-6561-IMPORT: V567 rows are content addressed and classified."""

    assert mod.validate_artifact(artifact) == []
    rows = {row["exp_id"]: row for row in artifact["v567_artifact_eligibility_rows"]}

    assert list(rows) == [artifact.exp_id for artifact in mod.V567_ARTIFACTS]
    assert rows["exp6548"]["eligible_for_v568_contract"] is True
    assert rows["exp6549"]["eligible_for_production_canary"] is True
    assert rows["exp6550"]["eligible_for_production_canary"] is True
    assert rows["exp6551"]["eligible_for_production_canary"] is True
    assert rows["exp6553"]["disposition"] == "blocked_infrastructure_evidence"
    assert rows["exp6553"]["scientific_audit_classification"] == "not_null_science"
    assert rows["exp6554"]["disposition"] == "blocked_infrastructure_evidence"
    assert rows["exp6557"]["disposition"] == "conductor_pre_gate_block"
    assert rows["exp6557"]["scientific_audit_classification"] == "not_scientific_audit"
    assert rows["exp6559"]["disposition"] == "zero_command_hardware_block_preserved"
    assert rows["exp6559"]["zero_command_hardware_receipt_preserved"] is True

    for row in rows.values():
        assert row["expected_path"].startswith("results/experiment_")
        assert row["sha256"].startswith("sha256:")
        assert row["row_consistency_status"] == "ok"
        assert row["live_verifier_exit_code"] == 0


def test_scenario_report_6561_gate_rows_use_exact_active_roadmap_fields(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-REPORT-6561-GATES: gates resolve to active task fields."""

    rows = artifact["v568_gate_contract_rows"]
    key = {(row["task_id"], row["upstream"], row["artifact_field"]) for row in rows}

    assert all(row["upstream_in_active_roadmap"] for row in rows)
    assert all(row["artifact_field_declared_by_upstream"] for row in rows)
    assert all(row["retired_upstream"] is False for row in rows)
    assert (
        "exp6563-production-safety-net-workload-canary",
        "exp6561-v568-evidence-gate-contract",
        "production_v567_evidence_ready_score",
    ) in key
    assert (
        "exp6564-rust-pyo3-safety-net-nfr01",
        "exp6563-production-safety-net-workload-canary",
        "production_workload_canary_ready_score",
    ) in key
    assert artifact["gate_check_summary"]["task_field_gate_contract_closed"] is True


def test_scenario_report_6561_prior_failure_rows_are_complete(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-REPORT-6561-PRIOR-FAILURE: rerun scope has all four fields."""

    rows = artifact["prior_failure_contract_rows"]
    by_pair = {(row["task_id"], row["experiment_id"]): row for row in rows}

    assert len(rows) == 4
    assert all(row["complete_prior_failure_contract"] for row in rows)
    assert all(row["retired_dependency_chain"] is False for row in rows)
    assert (
        by_pair[
            (
                "exp6562-constraint-saturation-independent-audit-v2",
                "exp6557-constraint-saturation-independent-audit",
            )
        ]["changed_method"]
        is True
    )
    assert (
        by_pair[
            (
                "exp6563-production-safety-net-workload-canary",
                "exp6551-production-safety-net-independent-audit",
            )
        ]["retire_if_same_verdict"]
        is True
    )


def test_scenario_report_6561_model_hardware_contract_freezes_boundaries(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-REPORT-6561-MODEL-HARDWARE: no legacy model or hardware shortcut opens."""

    model = artifact["model_and_sequential_runtime_contract"]
    hardware = artifact["hardware_claim_boundary"]

    assert model["mandated_model_ids"] == list(mod.MANDATED_MODEL_IDS)
    assert model["headline_llm_tasks"] == ["exp6566", "exp6568"]
    assert model["sequential_load_rule"]["capacity_prediction_authority"] is False
    assert model["sequential_load_rule"]["actual_load_required"] is True
    assert model["gguf_loader_rule"]["auto_tokenizer_from_gguf_repo_allowed"] is False
    assert model["legacy_model_policy"]["legacy_smoke_models_can_support_headline"] is False
    assert model["planned_v568_experiment_ids"] == list(range(6561, 6574))
    assert model["proposal_inventory_source_sha256"].startswith("sha256:")

    assert hardware["exp6559_zero_command_receipt_preserved"] is True
    assert hardware["exp6561_hardware_command_count"] == 0
    assert hardware["gatemate"]["later_command_requires_new_operator_receipt"] is True
    assert hardware["tsu"]["authenticated_api_available"] is False
    assert hardware["arc_no_solve_rule"]["no_game_or_level_solve_claim"] is True


def test_scenario_report_6561_schema_write_and_validation_edges(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6561-SCHEMA: output is atomic, checksummed, and recomputed."""

    result_path = tmp_path / "exp6561.json"
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
    assert written["status"] == "complete_v568_evidence_gate_contract_ready"
    assert written["verdict_class"] is None
    assert written["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert written["verifier_is_oracle"] is True
    assert written["v568_evidence_contract_ready_score"] == 1.0
    assert written["production_v567_evidence_ready_score"] == 1.0
    assert set(written["field_provenance"]) >= set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert set(written["field_principles"]) >= set(mod.REQUIRED_ARTIFACT_FIELDS)

    validations = [
        ("delete", "status", "missing required fields"),
        ("set", ("honest_verdict", "ready"), "honest_verdict lacks terminal prefix"),
        ("set", ("verdict_class", "positive"), "verdict_class is outside closed class"),
        ("set", ("inference_substrate", "live_llm_inference"), "inference_substrate mismatch"),
        ("set", ("verifier_is_oracle", False), "verifier_is_oracle must be true"),
        ("set", ("protected_files_unchanged", {"all_unchanged": False}), "protected files changed"),
        ("set", ("reproducibility_checksum", "sha256:bad"), "reproducibility_checksum mismatch"),
    ]
    for mode, spec, expected in validations:
        bad = deepcopy(written)
        if mode == "delete":
            del bad[spec]
        else:
            key, value = spec
            bad[key] = value
        if expected != "reproducibility_checksum mismatch":
            bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
        assert any(expected in error for error in mod.validate_artifact(bad))

    alias = deepcopy(written)
    alias["v567_artifact_eligibility_rows"][1]["sha256"] = "sha256:alias"
    alias["reproducibility_checksum"] = mod.reproducibility_checksum(alias)
    assert any("production canary hash alias" in error for error in mod.validate_artifact(alias))

    missing_gate_field = deepcopy(written)
    missing_gate_field["v568_gate_contract_rows"][0]["artifact_field_declared_by_upstream"] = False
    missing_gate_field["reproducibility_checksum"] = mod.reproducibility_checksum(
        missing_gate_field
    )
    assert any(
        "gate contract has undeclared field" in error
        for error in mod.validate_artifact(missing_gate_field)
    )

    missing_prior = deepcopy(written)
    del missing_prior["prior_failure_contract_rows"][0]["addressed_by"]
    missing_prior["reproducibility_checksum"] = mod.reproducibility_checksum(missing_prior)
    assert any(
        "prior failure row missing required fields" in error
        for error in mod.validate_artifact(missing_prior)
    )

    bad_model = deepcopy(written)
    bad_model["model_and_sequential_runtime_contract"]["mandated_model_ids"] = ["legacy"]
    bad_model["reproducibility_checksum"] = mod.reproducibility_checksum(bad_model)
    assert "mandated GGUF model identities changed" in mod.validate_artifact(bad_model)

    bad_provenance = deepcopy(written)
    bad_provenance["field_provenance"] = {}
    bad_provenance["reproducibility_checksum"] = mod.reproducibility_checksum(bad_provenance)
    assert "field_provenance must cover required fields" in mod.validate_artifact(bad_provenance)

    bad_principles = deepcopy(written)
    bad_principles["field_principles"] = {}
    bad_principles["reproducibility_checksum"] = mod.reproducibility_checksum(bad_principles)
    assert "field_principles must cover required fields" in mod.validate_artifact(bad_principles)

    bad_prod = deepcopy(written)
    bad_prod["v567_artifact_eligibility_rows"][1]["eligible_for_production_canary"] = False
    bad_prod["reproducibility_checksum"] = mod.reproducibility_checksum(bad_prod)
    assert any(
        "production evidence score must derive" in error
        for error in mod.validate_artifact(bad_prod)
    )

    bad_gate_rows = deepcopy(written)
    bad_gate_rows["v568_gate_contract_rows"] = [
        "not-a-row",
        {
            "artifact_field_declared_by_upstream": True,
            "upstream_in_active_roadmap": False,
            "retired_upstream": True,
        },
    ]
    bad_gate_rows["reproducibility_checksum"] = mod.reproducibility_checksum(bad_gate_rows)
    gate_errors = mod.validate_artifact(bad_gate_rows)
    assert "gate contract row must be a mapping" in gate_errors
    assert "gate contract has out-of-roadmap upstream" in gate_errors
    assert "gate contract has retired upstream" in gate_errors

    bad_prior_rows = deepcopy(written)
    bad_prior_rows["prior_failure_contract_rows"] = [
        "not-a-row",
        {
            "experiment_id": "retired",
            "verdict": "x",
            "addressed_by": "x",
            "retire_if_same_verdict": True,
            "complete_prior_failure_contract": True,
            "retired_dependency_chain": True,
        },
    ]
    bad_prior_rows["reproducibility_checksum"] = mod.reproducibility_checksum(bad_prior_rows)
    prior_errors = mod.validate_artifact(bad_prior_rows)
    assert "prior failure row must be a mapping" in prior_errors
    assert "prior failure row uses retired dependency chain" in prior_errors

    bad_loader = deepcopy(written)
    bad_loader["model_and_sequential_runtime_contract"]["gguf_loader_rule"][
        "auto_tokenizer_from_gguf_repo_allowed"
    ] = True
    bad_loader["reproducibility_checksum"] = mod.reproducibility_checksum(bad_loader)
    assert "legacy GGUF tokenizer substitution opened" in mod.validate_artifact(bad_loader)

    bad_sequential = deepcopy(written)
    bad_sequential["model_and_sequential_runtime_contract"]["sequential_load_rule"][
        "actual_load_required"
    ] = False
    bad_sequential["reproducibility_checksum"] = mod.reproducibility_checksum(bad_sequential)
    assert "sequential actual-load rule changed" in mod.validate_artifact(bad_sequential)

    bad_hardware = deepcopy(written)
    bad_hardware["hardware_claim_boundary"]["exp6561_hardware_command_count"] = 1
    bad_hardware["reproducibility_checksum"] = mod.reproducibility_checksum(bad_hardware)
    assert "hardware command boundary violated" in mod.validate_artifact(bad_hardware)

    bad_zero = deepcopy(written)
    bad_zero["hardware_claim_boundary"]["exp6559_zero_command_receipt_preserved"] = False
    bad_zero["reproducibility_checksum"] = mod.reproducibility_checksum(bad_zero)
    assert "Exp6559 zero-command receipt not preserved" in mod.validate_artifact(bad_zero)

    ready_failed = deepcopy(written)
    ready_failed["gate_check_summary"]["failed_checks"] = ["should_not_open"]
    ready_failed["reproducibility_checksum"] = mod.reproducibility_checksum(ready_failed)
    assert "ready score cannot be open with failed checks" in mod.validate_artifact(ready_failed)

    bad_aggregate = deepcopy(written)
    bad_aggregate["aggregate_row_recomputation"]["v568_evidence_contract_ready_from_rows"] = False
    bad_aggregate["reproducibility_checksum"] = mod.reproducibility_checksum(bad_aggregate)
    assert "ready score must derive from aggregate recomputation" in mod.validate_artifact(
        bad_aggregate
    )


def test_scenario_report_6561_missing_production_input_blocks(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6561-IMPORT: missing input closes blocked, not null."""

    paths = mod.default_v567_paths(REPO)
    paths["exp6550"] = tmp_path / "missing-exp6550.json"
    blocked = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "blocked.json",
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        check_results=_fake_check_results(),
        artifact_paths=paths,
        run_date="20260823",
    )

    assert blocked["status"] == "blocked_v568_evidence_gate_contract_missing_inputs"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["production_v567_evidence_ready_score"] == 0.0
    assert "exp6550_input_exists" in blocked["gate_check_summary"]["failed_checks"]
    assert mod.validate_artifact(blocked) == []


def test_scenario_report_6561_helper_edges_are_deterministic() -> None:
    """SCENARIO-REPORT-6561-SCHEMA: helper edge cases fail closed."""

    assert mod.sha256_bytes(b"x").startswith("sha256:")
    assert mod.sha256_file(Path("/tmp/definitely-missing-exp6561-file")) == "missing"
    assert mod.default_v567_paths(REPO)["exp6548"].name.startswith("experiment_6548")
    assert mod._command_text(["a b", "c"]) == "'a b' c"
    assert mod._python_executable(REPO).endswith("python")
    assert mod._stamped_flags({"corrigendum_pending": [{"kind": "X"}, "skip"]}) == [{"kind": "X"}]
    assert mod._coerce_closed_verdict_class(None) is None
    assert mod._coerce_closed_verdict_class("null") == "null"
    assert mod._coerce_closed_verdict_class("weird") == "disqualified"

    fields = mod._parse_required_fields(
        "REQUIRED ARTIFACT FIELDS:\n  status:\n    principle: x\nRun command:"
    )
    assert fields == {"status"}

    failed_row = mod._eligibility_row(
        mod.V567_ARTIFACTS[1],
        path=REPO / mod.V567_ARTIFACTS[1].relative_path,
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
    assert failed_row["disposition"] == "not_imported_failed_checks"
    assert failed_row["unresolved_reasons"] == [
        "exp6549_readiness_field_present",
        "live_critical_flags",
        "row_consistency_blocking",
    ]

    retired = mod._retired_experiment_ids(
        {
            "retired": [None, {"id": "retired-a", "experiment_ids": ["retired-b"]}],
            "retired_experiments": "not-a-list",
        }
    )
    assert retired == {"retired-a", "retired-b"}

    assert mod._requires_retired_ids(
        {
            "tasks": [
                "skip",
                {"id": "task-a", "requires": ["retired-a"]},
                {"id": "task-b", "gated_on": [{"upstream": "retired-b"}]},
            ]
        },
        {"retired-a", "retired-b"},
    ) == {"retired-a", "retired-b"}

    gate_rows = mod._gate_contract_rows(
        {"tasks": ["skip", {"id": "task", "gated_on": ["skip-gate"]}]},
        {"upstream": {"field"}},
        {"upstream"},
    )
    assert gate_rows == []

    prior_rows = mod._prior_failure_contract_rows(
        {
            "tasks": [
                "skip",
                {"id": "bad", "prior_failures": ["not-a-row"]},
                {"id": "task", "prior_failures": [{"experiment_id": "old"}]},
            ]
        },
        {"old"},
        {"old"},
    )
    assert prior_rows[0]["complete_prior_failure_contract"] is False
    assert prior_rows[1]["complete_prior_failure_contract"] is False
    assert prior_rows[1]["retired_dependency_chain"] is True

    summary = mod._gate_check_summary(
        rows=[
            {
                "exp_id": "exp6549",
                "exists": False,
                "expected_path": "missing.json",
                "eligible_for_production_canary": False,
                "eligible_for_v568_contract": False,
            }
        ],
        production={"production_v567_evidence_ready": False, "eligible_exp_ids": []},
        gate_rows=[
            {
                "task_id": "task",
                "upstream": "retired-upstream",
                "artifact_field": "bad_field",
                "upstream_in_active_roadmap": False,
                "artifact_field_declared_by_upstream": False,
                "retired_upstream": True,
            }
        ],
        prior_rows=[{"complete_prior_failure_contract": False, "retired_dependency_chain": True}],
        model_contract={"all_model_contract_checks_passed": False},
        hardware_contract={"all_hardware_boundary_checks_passed": False},
        protected={"all_unchanged": False, "changed_paths": ["research-roadmap.yaml"]},
    )
    assert {
        "exp6549_input_exists",
        "exp6549_contract_eligible",
        "production_v567_evidence_ready",
        "gate_upstream_in_active_roadmap",
        "gate_artifact_field_declared",
        "gate_retired_upstream",
        "prior_failure_contract_complete",
        "prior_failure_retired_dependency_chain_absent",
        "model_contract_closed",
        "hardware_boundary_closed",
        "protected_files_unchanged",
    } <= set(summary["failed_checks"])

    assert mod._status_and_verdict(True, False, 1.0, []) == (
        "complete_v568_evidence_gate_contract_ready",
        "complete_v568_evidence_gate_contract_ready: V567 artifacts are content-addressed; production Exp6549-Exp6551 are eligible; V568 gate, prior-failure, model, hardware, and protected-file contracts close",
        None,
    )
    assert mod._status_and_verdict(False, True, 1.0, ["missing"]) == (
        "blocked_v568_evidence_gate_contract_missing_inputs",
        "blocked_v568_evidence_gate_contract_missing_inputs: required V567 input artifact is missing; failed checks are recorded",
        "blocked",
    )
    assert mod._status_and_verdict(False, False, 1.0, ["gate"]) == (
        "partial_v568_evidence_gate_contract",
        "partial_v568_evidence_gate_contract: usable V567 evidence exists but one or more gate, prior-failure, model, hardware, or protected-file checks failed",
        "partial",
    )
    assert mod._status_and_verdict(False, False, 0.0, []) == (
        "blocked_v568_evidence_gate_contract",
        "blocked_v568_evidence_gate_contract: no usable V567 input set was available",
        "blocked",
    )
