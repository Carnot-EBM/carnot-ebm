"""Tests for Exp6436 V554 terminal handoff.

Spec refs: REQ-INFRA-6436, SCENARIO-INFRA-6436-1,
SCENARIO-INFRA-6436-2, SCENARIO-INFRA-6436-3,
SCENARIO-INFRA-6436-4, SCENARIO-INFRA-6436-5,
SCENARIO-INFRA-6436-6.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_6436_v554_terminal_handoff_and_queue_preflight as mod
from carnot.experiment_artifacts import ARTIFACT_ROOT_ENV


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH
_REPORT_CACHE: dict[str, object] | None = None


def _report() -> dict[str, object]:
    global _REPORT_CACHE
    if _REPORT_CACHE is None:
        _REPORT_CACHE = mod.build_report(
            REPO,
            date="20260815",
            command_receipts=[{"command": "focused", "exit_code": 0}],
            before_hashes=mod.protected_hashes(REPO),
            duration_s=1.0,
        )
    return copy.deepcopy(_REPORT_CACHE)


def test_req_infra_6436_spec_declares_required_contract() -> None:
    """REQ-INFRA-6436: OpenSpec owns the V554 handoff contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6436") : text.index("REQ-INFRA-6404")]

    for marker in (
        "SCENARIO-INFRA-6436-1",
        "SCENARIO-INFRA-6436-2",
        "SCENARIO-INFRA-6436-3",
        "SCENARIO-INFRA-6436-4",
        "SCENARIO-INFRA-6436-5",
        "SCENARIO-INFRA-6436-6",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_infra_6436_v553_terminal_rows_preserve_facts() -> None:
    """SCENARIO-INFRA-6436-1: V553 terminal facts remain separate."""

    report = _report()

    assert mod.validate_report(report) == []
    assert report["v553_artifact_count"] == len(mod.EXPECTED_V553_TASK_IDS)
    assert [row["task_id"] for row in report["v553_terminal_rows"]] == list(
        mod.EXPECTED_V553_TASK_IDS
    )

    rows = {row["task_id"]: row for row in report["v553_terminal_rows"]}
    exp6434 = rows["exp6434-arc-state-key-reachability-ab"]
    assert exp6434["byte_count"] == 0
    assert exp6434["artifact_status"] == "zero_byte"
    assert exp6434["claim_eligibility"]["eligible"] is False
    assert "missing_scientific_evidence_zero_byte_artifact" in exp6434["claim_eligibility"][
        "blockers"
    ]

    assert {row["task_id"] for row in report["v553_flagged_artifacts"]} == {
        "exp6429-constraint-saturation-verification-cost-ab",
        "exp6432-held-shift-process-restart-csl-replication",
    }
    assert {row["task_id"] for row in report["v553_underpowered_artifacts"]} >= {
        "exp6429-constraint-saturation-verification-cost-ab",
        "exp6431-controlled-memory-interference-ab",
        "exp6432-held-shift-process-restart-csl-replication",
        "exp6433-csl-row-recomputation-safety-audit",
    }

    determinations = report["v553_terminal_claim_determinations"]
    assert determinations["public_factor"]["eligible"] is True
    assert determinations["verification_cost"]["eligible"] is False
    assert determinations["prospective_csl"]["eligible"] is False
    assert determinations["internal_arc_reachability"]["eligible"] is False
    assert determinations["public_arc"]["eligible"] is False
    assert determinations["hardware"]["eligible"] is False


def test_scenario_infra_6436_active_v554_queue_fails_closed_at_nine_tasks() -> None:
    """SCENARIO-INFRA-6436-2: active queue identity mismatch blocks readiness."""

    report = _report()

    assert report["status"] == "complete_blocked_v554_queue_incomplete"
    assert report["task_count"] == 9
    assert report["v554_queue_ready_score"] == 0.0
    assert report["blocked_reason"] == "task_count: expected 12 observed 9"
    assert report["gate_check_summary"]["failed_check"] == "task_count"
    assert report["gate_check_summary"]["expected_condition"] == "task_count == 12"
    assert report["gate_check_summary"]["observed_value"] == 9
    assert report["gate_check_summary"]["evidence_path"] == "research-roadmap.yaml"

    identity = report["unique_id_and_deliverable_check"]
    assert identity["ok"] is False
    assert identity["task_count"] == 9
    assert identity["expected_task_count"] == 12
    assert identity["missing_expected_task_ids"] == [
        "exp6445-arc-state-key-reachability-sharded-ab",
        "exp6446-joint-pathway-dependence-audit",
        "exp6447-v554-adversarial-capstone",
    ]
    assert identity["unique_task_ids"] is True
    assert identity["unique_deliverables"] is True

    assert report["milestone_consistency_check"]["ok"] is True
    assert report["schema_validation"]["ok"] is True
    assert report["prior_failure_validation"]["ok"] is False
    assert report["exclusion_manifest_validation"]["ok"] is True


def test_scenario_infra_6436_gates_model_policy_and_prompt_contracts() -> None:
    """SCENARIO-INFRA-6436-3 and 4: gates and prompt contracts are explicit."""

    report = _report()

    assert report["structured_gate_validation"]["ok"] is True
    assert report["structured_gate_validation"]["gate_count"] == 8
    assert report["gate_producer_contract_rows"]
    assert all(row["producer_declares_artifact_field"] for row in report["gate_producer_contract_rows"])
    assert {
        (row["consumer_task_id"], row["upstream"], row["artifact_field"])
        for row in report["gate_producer_contract_rows"]
    } >= {
        (
            "exp6437-generation-to-verdict-receipt-replay-contract",
            "exp6436-v554-terminal-handoff-and-queue-preflight",
            "v554_queue_ready_score",
        ),
        (
            "exp6443-fresh-held-restart-csl-replication",
            "exp6442-skill-misevolution-quarantine-rollback-ab",
            "misevolution_safety_ready_score",
        ),
    }

    policy = report["model_policy_validation"]
    assert policy["ok"] is True
    assert policy["llm_task_ids"] == [
        "exp6439-factor-clause-influence-ab",
        "exp6440-held-factor-revocation-binding-shift-ab",
        "exp6441-prospective-query-conditioned-factor-reuse",
        "exp6442-skill-misevolution-quarantine-rollback-ab",
        "exp6443-fresh-held-restart-csl-replication",
    ]
    assert policy["failures"] == []

    per_unit = report["per_unit_row_contract_validation"]
    assert per_unit["ok"] is True
    assert "exp6438-powered-verification-cost-repair-ab" in per_unit["comparative_task_ids"]
    assert per_unit["failures"] == []

    prompt = report["prompt_terminal_line_validation"]
    assert prompt["ok"] is True
    assert prompt["failures"] == []


def test_scenario_infra_6436_prior_failure_contract_names_missing_rerun() -> None:
    """SCENARIO-INFRA-6436-5: all rerun-scope tasks need four prior fields."""

    report = _report()
    validation = report["prior_failure_validation"]

    assert validation["ok"] is False
    assert validation["required_rerun_task_ids"] == list(mod.REQUIRED_PRIOR_FAILURE_TASK_IDS)
    assert validation["missing_required_rerun_task_ids"] == [
        "exp6445-arc-state-key-reachability-sharded-ab"
    ]
    present = {
        row["task_id"]: row
        for row in validation["required_prior_failure_rows"]
        if row["present_in_roadmap"]
    }
    for task_id in (
        "exp6438-powered-verification-cost-repair-ab",
        "exp6440-held-factor-revocation-binding-shift-ab",
        "exp6441-prospective-query-conditioned-factor-reuse",
        "exp6443-fresh-held-restart-csl-replication",
        "exp6444-csl-lifecycle-recomputation-audit",
    ):
        assert present[task_id]["complete"] is True
        assert present[task_id]["retire_if_same_verdict"] is True


def test_scenario_infra_6436_schema_write_and_validation_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-INFRA-6436-6: artifact schema is stable and atomic."""

    report = _report()

    assert report["protected_files_unchanged"]["ok"] is True
    assert report["verifier_is_oracle"] is False
    assert report["random_seed"] is None
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(report["field_principles"])
    assert "acceptance_gate:v554_queue_ready_score" in report["field_principles"]
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) == set(report["field_provenance"])
    assert set(report["field_provenance"].values()) <= {
        "measured",
        "derived",
        "constant",
        "upstream",
    }
    assert report["reproducibility_checksum"] == mod.payload_checksum(report)

    validations = [
        ("delete", "status", "missing required field: status"),
        ("set", ("verifier_is_oracle", True), "verifier_is_oracle must be false"),
        ("set", ("random_seed", 6436), "random_seed must be null"),
        ("set", ("v554_queue_ready_score", 1.0), "ready score cannot pass with failed checks"),
        ("set", ("gate_check_summary", []), "gate_check_summary must be a mapping"),
        (
            "set",
            ("v553_terminal_claim_determinations.public_factor.eligible", False),
            "public factor eligibility must remain true",
        ),
        (
            "set",
            ("v553_terminal_claim_determinations.verification_cost.eligible", True),
            "verification cost must remain blocked",
        ),
        (
            "set",
            ("unique_id_and_deliverable_check.ok", True),
            "queue identity check must fail while task_count is 9",
        ),
        (
            "set",
            ("prior_failure_validation.missing_required_rerun_task_ids", []),
            "Exp6445 missing prior-failure task must be visible",
        ),
        ("set", ("protected_files_unchanged.ok", False), "protected files changed"),
        ("set", ("honest_verdict", "ok"), "honest_verdict lacks terminal prefix"),
        ("set", ("reproducibility_checksum", "sha256:bad"), "reproducibility_checksum mismatch"),
    ]
    for mode, spec, expected in validations:
        bad = copy.deepcopy(report)
        if mode == "delete":
            del bad[spec]
        else:
            dotted, value = spec
            target = bad
            parts = dotted.split(".")
            for part in parts[:-1]:
                target = target[part]
            target[parts[-1]] = value
        if expected != "reproducibility_checksum mismatch":
            bad["reproducibility_checksum"] = mod.payload_checksum(bad)
        assert any(expected in error for error in mod.validate_report(bad))

    bad = copy.deepcopy(report)
    del bad["field_principles"]["status"]
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "missing field_principles entry: status" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["field_provenance"] = {}
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "field_provenance must cover exactly required fields" in mod.validate_report(bad)

    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    path = mod.write_report(report, REPO, env={ARTIFACT_ROOT_ENV: str(artifact_root)})
    assert path == artifact_root / mod.RESULT_RELATIVE_PATH.name
    assert json.loads(path.read_text(encoding="utf-8")) == report

    monkeypatch.setattr(
        mod,
        "run",
        lambda *, date, root=REPO, write=True, command_receipts=None: {
            "status": f"complete-{date}",
            "honest_verdict": "complete: patched",
        },
    )
    assert mod.main(["--date", "20260815"]) == 0
    assert mod.RESULT_RELATIVE_PATH.name in capsys.readouterr().out


def test_req_infra_6436_helper_edges_and_dirty_queue_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFRA-6436: malformed queue inputs fail closed without fabrication."""

    assert mod.path_receipt(tmp_path / "missing.json")["present"] is False
    assert mod.read_json_mapping(tmp_path / "missing.json")[1]["error"] == "missing"
    assert mod._artifact_status({}, {"error": "missing", "size_bytes": None}) == "missing"
    assert mod._artifact_status({}, {"error": "json_error", "size_bytes": 4}) == "malformed"
    assert mod._artifact_status({}, {"error": "json_error", "size_bytes": 0}) == "zero_byte"
    assert mod._has_underpowered_cells({"harm_underpowered_missing_and_flagged_cells": []}) is False
    assert mod._has_underpowered_cells(
        {"harm_underpowered_missing_and_flagged_cells": {"underpowered_cell_count": 1}}
    ) is True
    assert mod._tasks({"tasks": "bad"}) == []
    gate_rows, gate_failures, gate_expressions = mod._gate_rows(
        [{"id": "exp1-test", "gated_on": ["bad"]}], {}, {}
    )
    assert gate_rows == []
    assert gate_expressions == []
    assert gate_failures == [
        {"task_id": "exp1-test", "reason": "gate_not_mapping", "gate": "bad"}
    ]

    monkeypatch.setattr(
        mod,
        "_prior_failure_linter",
        lambda root: {"schema_errors": [], "prior_failure_violations": []},
    )
    monkeypatch.setattr(
        mod,
        "_gate_audit",
        lambda root: {"roadmap_gate_audit_passed": True, "n_prior_failures_missing": 0},
    )
    prior_validation = mod._prior_validation(
        {"exp6438-powered-verification-cost-repair-ab": {"prior_failures": []}},
        REPO,
    )
    assert prior_validation["required_prior_failure_rows"][0] == {
        "task_id": "exp6438-powered-verification-cost-repair-ab",
        "present_in_roadmap": True,
        "complete": False,
    }

    data = copy.deepcopy(mod.read_yaml_mapping(REPO / mod.ACTIVE_ROADMAP_RELATIVE_PATH))
    checks = mod.validate_v554_queue_data(data, REPO, "20260815", retired_exp_ids=set())
    assert checks["task_count"] == 9
    assert checks["unique_id_and_deliverable_check"]["ok"] is False

    dirty = copy.deepcopy(data)
    tasks = dirty["tasks"]
    tasks[1]["id"] = tasks[0]["id"]
    tasks[2]["deliverable"] = "not-results.txt"
    tasks[3]["gated_on"] = [
        {"upstream": "exp9999-missing", "artifact_field": "missing_field", "op": "??", "value": 1.0}
    ]
    tasks[4]["milestone"] = "2026.08.553"
    tasks[5]["prior_failures"] = [{"experiment_id": "", "verdict": "", "addressed_by": ""}]
    tasks[6]["prompt"] = (
        "CONTEXT\nTASK\nCONCRETE STEPS\nmust compare arms blocked_verdict\n"
        "Run command: x\nDo NOT push."
    )
    tasks[7]["prompt"] = (
        "CONTEXT\nEXISTING CODE TO READ FIRST\nTASK\nCONCRETE STEPS\n"
        "MODEL_SPECS Bad/Unexpected-GGUF cached_sota_pair() embedded tokenizer "
        "AutoTokenizer.from_pretrained raw output\nRun command: x\n"
        "Do NOT push. Do NOT modify scripts/research_conductor.py."
    )
    tasks[8]["requires"] = [tasks[8]["id"], "exp6436-v554-terminal-handoff-and-queue-preflight"]
    dirty_checks = mod.validate_v554_queue_data(dirty, REPO, "20260815", retired_exp_ids={6436})
    assert dirty_checks["schema_validation"]["ok"] is False
    assert dirty_checks["unique_id_and_deliverable_check"]["ok"] is False
    assert dirty_checks["milestone_consistency_check"]["ok"] is False
    assert dirty_checks["structured_gate_validation"]["ok"] is False
    assert dirty_checks["prior_failure_validation"]["ok"] is False
    assert dirty_checks["model_policy_validation"]["ok"] is False
    assert dirty_checks["prompt_terminal_line_validation"]["ok"] is False

    assert mod._first_failed_check({"task_count": 1})["failed_check"] == "task_count"
    assert (
        mod._first_failed_check({"task_count": 12, "schema_validation": {"ok": False}})[
            "failed_check"
        ]
        == "schema_validation"
    )
    assert mod._first_failed_check({"task_count": 12})["failed_check"] == "unknown"
    assert mod._test_rows(None)[0]["source"] == "declared"

    receipt = tmp_path / "receipts.json"
    assert mod.read_external_test_receipts(receipt) == []
    receipt.write_text("{", encoding="utf-8")
    assert mod.read_external_test_receipts(receipt) == []
    receipt.write_text("{}", encoding="utf-8")
    assert mod.read_external_test_receipts(receipt) == []
    receipt.write_text('[{"command": "ok", "exit_code": 0}, "skip"]', encoding="utf-8")
    assert mod.read_external_test_receipts(receipt) == [{"command": "ok", "exit_code": 0}]

    writes: list[dict[str, object]] = []
    real_validate_report = mod.validate_report

    def fake_build_report(
        root: Path,
        *,
        date: str,
        command_receipts: list[dict[str, object]],
        before_hashes: dict[str, str | None],
        duration_s: float,
    ) -> dict[str, object]:
        return {
            "date": date,
            "command_receipts": command_receipts,
            "before_hashes": before_hashes,
            "duration_s": duration_s,
            "reproducibility_checksum": "sha256:fake",
        }

    monkeypatch.setattr(mod, "protected_hashes", lambda root: {"x": "sha256:x"})
    monkeypatch.setattr(mod, "read_external_test_receipts", lambda: [{"command": "external"}])
    monkeypatch.setattr(mod, "build_report", fake_build_report)
    monkeypatch.setattr(mod, "validate_report", lambda report: [])
    monkeypatch.setattr(mod, "write_report", lambda report, root: writes.append(report))

    report = mod.run(date="20260815", root=REPO, write=True)
    assert report["command_receipts"] == [{"command": "external"}]
    assert writes == [report]

    monkeypatch.setattr(mod, "validate_report", lambda report: ["bad"])
    with pytest.raises(ValueError, match="bad"):
        mod.run(date="20260815", root=REPO, write=False, command_receipts=[{"command": "c"}])
    monkeypatch.setattr(mod, "validate_report", real_validate_report)

    base = _report()
    for dotted, expected in (
        ("v553_terminal_claim_determinations", "v553_terminal_claim_determinations must be a mapping"),
        ("field_principles", "field_principles must be a mapping"),
        ("field_provenance", "field_provenance must be a mapping"),
    ):
        bad = copy.deepcopy(base)
        bad[dotted] = []
        bad["reproducibility_checksum"] = mod.payload_checksum(bad)
        assert expected in mod.validate_report(bad)

    bad = copy.deepcopy(base)
    bad["field_provenance"] = dict.fromkeys(mod.REQUIRED_ARTIFACT_FIELDS, "bad_kind")
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "field_provenance has invalid classification" in mod.validate_report(bad)

    bad = copy.deepcopy(base)
    bad["blocked_reason"] = None
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "blocked report must name blocked_reason" in mod.validate_report(bad)
