"""Tests for Exp6349 V546 adversarial capstone.

Spec refs: REQ-INFRA-6349, SCENARIO-INFRA-6349-1,
SCENARIO-INFRA-6349-2, SCENARIO-INFRA-6349-3,
SCENARIO-INFRA-6349-4, SCENARIO-INFRA-6349-5.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_6349_v546_adversarial_capstone as mod
from carnot.experiment_artifacts import ARTIFACT_ROOT_ENV


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/research-harnesses/spec.md"


def _report() -> dict[str, object]:
    return mod.build_report(
        REPO,
        date="20260812",
        command_receipts=[{"command": "focused", "exit_code": 0}],
        before_hashes=mod.protected_hashes(REPO),
        started_at=0.0,
    )


def test_req_infra_6349_spec_declares_required_contract() -> None:
    """REQ-INFRA-6349: OpenSpec records the capstone contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6349") :]

    for marker in (
        "SCENARIO-INFRA-6349-1",
        "SCENARIO-INFRA-6349-2",
        "SCENARIO-INFRA-6349-3",
        "SCENARIO-INFRA-6349-4",
        "SCENARIO-INFRA-6349-5",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenarios_infra_6349_terminal_matrix_and_gates() -> None:
    """SCENARIO-INFRA-6349-1 and 2: exact terminal rows and gates decide skips."""

    report = _report()

    assert mod.validate_report(report) == []
    assert len(report["declared_task_ids_and_deliverables"]) == 13
    assert report["dependency_recomputation"]["ok"] is True

    matrix = report["artifact_existence_hash_schema_status_and_honest_verdict_by_task"]
    assert matrix["exp6337-v546-bounded-terminal-handoff"]["terminal_class"] == "flagged"
    assert matrix["exp6340-parser-jit-semantic-diversity-canary"]["terminal_class"] == "null"
    assert matrix["exp6341-prospective-prefix-utility-ab"]["terminal_class"] == "skipped"
    assert matrix["exp6341-prospective-prefix-utility-ab"]["status_raw"] == "blocked"
    assert matrix["exp6349-v546-adversarial-capstone"]["self_referential_hash_excluded"] is True

    gate = report["structured_gate_recomputation"]
    exp6341_gate = next(
        row for row in gate["gates"] if row["task_id"] == "exp6341-prospective-prefix-utility-ab"
    )
    assert exp6341_gate["passed"] is False
    assert exp6341_gate["actual"] == 0.0
    assert exp6341_gate["skip_effect"] == "structured_skip_preserved"
    assert gate["failed_gate_count"] == 1

    skipped = report["skipped_task_handling"]
    assert skipped["structured_skipped_task_ids"] == ["exp6341-prospective-prefix-utility-ab"]
    assert skipped["rows"]["exp6341-prospective-prefix-utility-ab"]["no_agent_execution"] is True
    assert skipped["rows"]["exp6341-prospective-prefix-utility-ab"]["hidden_utility_claim"] is False


def test_scenarios_infra_6349_model_oracle_mutation_and_gap_audits() -> None:
    """SCENARIO-INFRA-6349-3 and 4: model and gap audits keep boundaries narrow."""

    report = _report()

    assert report["model_policy_and_MODEL_SPECS_audit"]["ok"] is True
    assert report["llama_cpp_embedded_tokenizer_audit"]["ok"] is True
    assert report["gpu_offload_and_memory_release_audit"]["ok"] is True
    assert report["source_model_weight_mutation_audit"]["total_mutation_count"] == 0
    assert report["generated_label_and_hidden_state_audit"]["generated_label_count"] == 0
    assert report["generated_label_and_hidden_state_audit"]["hidden_state_access_count"] == 0
    assert (
        report["exact_oracle_and_learned_claim_boundary_audit"]["capstone_verifier_is_oracle"]
        is False
    )
    assert (
        "exp6339-incremental-prefix-enforcement-substrate"
        in report["exact_oracle_and_learned_claim_boundary_audit"]["upstream_oracle_tasks"]
    )

    assert report["prefix_generation_determination"]["closure"] == "not_closed"
    assert report["certified_continuous_learning_determination"]["closure"] == "closed"
    assert report["eprocess_and_factor_lifecycle_determination"]["rollback_identity_ok"] is True
    assert report["safety_audit_determination"]["utility_promotion_count"] == 0
    assert report["arc_action_influence_determination"]["closure"] == "closed_no_solve"
    assert report["solve_provenance_audit"]["solve_claim_count"] == 0
    assert report["arc_registry_immutability_audit"]["registry_update_count"] == 0
    assert report["hardware_nonuse_and_inference_substrate_audit"]["v546_hardware_claim_count"] == 0
    assert report["verification_cost_accounting_audit"]["missing_cost_task_ids"] == []

    gaps = report["three_gap_closure_matrix"]
    assert gaps["gap_1_prefix_generation"]["state"] == "skipped_after_null_canary"
    assert gaps["gap_2_certified_self_learning"]["state"] == "closed"
    assert gaps["gap_3_arc_action_influence"]["state"] == "closed_no_solve"


def test_scenario_infra_6349_fields_write_and_failure_aware_validation(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6349-5: fields are annotated, checksummed, and failure aware."""

    report = _report()

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) == set(report["field_principles"])
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) == set(report["field_provenance"])
    assert report["reproducibility_checksum"] == mod.payload_checksum(report)
    assert report["llm_call_count"] == 0
    assert report["verifier_is_oracle"] is False

    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    path = mod.write_report(report, REPO, env={ARTIFACT_ROOT_ENV: str(artifact_root)})
    assert path == artifact_root / mod.RESULT_RELATIVE_PATH.name
    assert json.loads(path.read_text(encoding="utf-8")) == report

    failed = mod.build_report(
        REPO,
        date="20260812",
        command_receipts=[{"command": "bad", "exit_code": 2}],
        before_hashes=mod.protected_hashes(REPO),
        started_at=0.0,
    )
    assert failed["status"] == "blocked_validation_command_failed"
    assert failed["honest_verdict"].startswith("blocked:")
    assert mod.validate_report(failed) == []

    bad = copy.deepcopy(report)
    bad["llm_call_count"] = 1
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "llm_call_count must be bare 0" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["test_exit_codes"] = {"bad": 1}
    bad["status"] = "complete_mixed_terminal_record"
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert any("nonzero command exit" in error for error in mod.validate_report(bad))


def test_req_infra_6349_helper_edges(tmp_path: Path) -> None:
    """REQ-INFRA-6349: helper edges fail closed without fabricating evidence."""

    roadmap = tmp_path / mod.ROADMAP_RELATIVE_PATH
    roadmap.write_text("tasks: not-a-list\n", encoding="utf-8")
    assert mod._roadmap_tasks(tmp_path) == []
    assert mod._module_name_for_task({"id": "exp1-fallback", "deliverable": "not-json"}) == (
        "exp1_fallback"
    )
    prompt = "REQUIRED ARTIFACT FIELDS: status, honest_verdict, duration_s.\n\nNEXT"
    assert mod.required_artifact_fields_from_prompt(prompt) == [
        "duration_s",
        "honest_verdict",
        "status",
    ]
    assert mod.required_artifact_fields_from_prompt("no block") == []
    assert mod.read_json_mapping(tmp_path / "missing.json")[1]["error"] == "missing"

    malformed = tmp_path / "bad.json"
    malformed.write_text("{bad", encoding="utf-8")
    assert str(mod.read_json_mapping(malformed)[1]["error"]).startswith("json_error:")

    scalar = tmp_path / "scalar.json"
    scalar.write_text("[1]", encoding="utf-8")
    assert mod.read_json_mapping(scalar)[1]["error"] == "json_not_mapping"

    assert mod._bare_value({"value": 3, "principle": "wrapped"}) == 3
    assert mod._bare_value({"value": 3}) == {"value": 3}
    assert mod._numeric_count(True) == 1
    assert mod._numeric_count(2.0) == 2
    assert mod._schema_status({}, self_task=False) == "missing"
    assert mod._compare_gate(1, "!=", 2) is True
    assert mod._compare_gate(1, ">", 0) is False
    assert mod._command_exit_codes([{"command": "ok", "exit_code": "0"}]) == {"ok": 0}
    assert mod._status_from_exit_codes({"bad": 1}) == "blocked_validation_command_failed"

    assert (
        mod.dependency_recomputation([{"id": "downstream", "requires": ["missing"]}], {})["ok"]
        is False
    )
    gates = mod.structured_gate_recomputation(
        [
            {
                "id": "downstream",
                "gated_on": [
                    "bad",
                    {"upstream": "up", "artifact_field": "x", "op": "==", "value": 1},
                ],
            }
        ],
        {"up": {"x": 0}},
        {"downstream": {"terminal_class": "complete"}},
    )
    assert gates["gates"][0]["skip_effect"] == "gate_failed_without_structured_skip"
    assert (
        mod.prior_failure_and_retirement_audit(
            [
                {
                    "id": "x",
                    "prior_failures": ["bad", {"verdict": "", "retire_if_same_verdict": True}],
                }
            ],
            {"x": {"honest_verdict": "complete: different"}},
        )["prior_failure_count"]
        == 1
    )
    assert mod.prompt_contract_audit([{"id": "x", "prompt": ""}])["ok"] is False

    assert (
        mod.model_policy_and_MODEL_SPECS_audit(
            {"exp6340-parser-jit-semantic-diversity-canary": {"MODEL_SPECS": []}},
            {"exp6340-parser-jit-semantic-diversity-canary": {"terminal_class": "complete"}},
        )["ok"]
        is False
    )
    assert (
        mod.llama_cpp_embedded_tokenizer_audit(
            {
                "exp6340-parser-jit-semantic-diversity-canary": {
                    "llama_cpp_embedded_tokenizer_receipts": []
                }
            },
            {"exp6340-parser-jit-semantic-diversity-canary": {"terminal_class": "complete"}},
        )["ok"]
        is False
    )
    assert (
        mod.gpu_offload_and_memory_release_audit(
            {
                "exp6340-parser-jit-semantic-diversity-canary": {
                    "cuda_gpu_offload_and_memory_release_receipts_by_model": []
                }
            },
            {"exp6340-parser-jit-semantic-diversity-canary": {"terminal_class": "complete"}},
        )["ok"]
        is False
    )
    assert mod.source_model_weight_mutation_audit({"x": "not-map"})["total_mutation_count"] == 0
    assert (
        mod.hardware_nonuse_and_inference_substrate_audit(
            {"x": "not-map", "y": {"hardware_claim_count": 1}}
        )["v546_hardware_claim_count"]
        == 1
    )
    assert mod.verification_cost_accounting_audit({})["missing_cost_task_ids"]
    protected_change = mod.protected_files_changed_with_reasons(
        REPO, {mod.ROADMAP_RELATIVE_PATH.as_posix(): "sha256:different"}
    )
    assert protected_change["changed_count"] >= 1


def test_req_infra_6349_validation_and_cli_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-INFRA-6349: validation, external receipts, run, and CLI edges are covered."""

    report = _report()
    bad = copy.deepcopy(report)
    del bad["status"]
    bad["field_principles"] = "bad"
    bad["field_provenance"] = "bad"
    bad["honest_verdict"] = "not-terminal"
    bad["verifier_is_oracle"] = True
    bad["reproducibility_checksum"] = ""
    errors = mod.validate_report(bad)
    assert "missing required field: status" in errors
    assert "field_principles is not a mapping" in errors
    assert "field_provenance is not a mapping" in errors
    assert "honest_verdict lacks terminal prefix" in errors
    assert "verifier_is_oracle must be false" in errors
    assert "reproducibility_checksum missing" in errors

    bad = copy.deepcopy(report)
    bad["field_principles"] = {}
    bad["field_provenance"] = {}
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    errors = mod.validate_report(bad)
    assert any(error.startswith("missing field_principles entry") for error in errors)
    assert any(error.startswith("missing field_provenance entry") for error in errors)

    bad = copy.deepcopy(report)
    bad["reproducibility_checksum"] = "sha256:wrong"
    assert "reproducibility_checksum mismatch" in mod.validate_report(bad)

    with pytest.raises(ValueError, match="invalid Exp6349 report"):
        mod.write_report(bad, REPO)

    receipt_path = tmp_path / "receipts.json"
    monkeypatch.setattr(mod, "EXTERNAL_TEST_RECEIPT_PATH", receipt_path)
    assert mod.read_external_test_receipts() == [{"command": mod.RUN_COMMAND, "exit_code": 0}]
    receipt_path.write_text(json.dumps({"cmd": 7}), encoding="utf-8")
    assert mod.read_external_test_receipts() == [{"command": "cmd", "exit_code": 7}]
    receipt_path.write_text(
        json.dumps([{"command": "list-cmd", "exit_code": 3}, {}]), encoding="utf-8"
    )
    assert mod.read_external_test_receipts() == [{"command": "list-cmd", "exit_code": 3}]

    artifact_root = tmp_path / "run-artifacts"
    artifact_root.mkdir()
    monkeypatch.setenv(ARTIFACT_ROOT_ENV, str(artifact_root))
    run_report = mod.run(
        date="20260812",
        root=REPO,
        write=True,
        command_receipts=[{"command": "ok", "exit_code": 0}],
    )
    assert run_report["status"] == "complete_mixed_terminal_record"
    assert (artifact_root / mod.RESULT_RELATIVE_PATH.name).exists()

    monkeypatch.setattr(
        mod,
        "run",
        lambda date: {"status": f"fake-{date}"},
    )
    assert mod.main(["--date", "20990101"]) == 0
    assert "fake-20990101" in capsys.readouterr().out
