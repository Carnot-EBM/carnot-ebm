"""Tests for Exp6448 V555 queue integrity.

Spec refs: REQ-REPORT-6448, SCENARIO-REPORT-6448-V554-FREEZE,
SCENARIO-REPORT-6448-V555-QUEUE,
SCENARIO-REPORT-6448-GATES-PRIORS-MODELS,
SCENARIO-REPORT-6448-SCHEMA.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_6448_v555_terminal_handoff_and_queue_integrity as mod
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


def test_req_report_6448_spec_declares_required_contract() -> None:
    """REQ-REPORT-6448: OpenSpec owns the V555 queue audit contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-REPORT-6448") :]

    for marker in (
        "SCENARIO-REPORT-6448-V554-FREEZE",
        "SCENARIO-REPORT-6448-V555-QUEUE",
        "SCENARIO-REPORT-6448-GATES-PRIORS-MODELS",
        "SCENARIO-REPORT-6448-SCHEMA",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_report_6448_v554_terminal_rows_preserve_mixed_evidence() -> None:
    """SCENARIO-REPORT-6448-V554-FREEZE: V554 evidence stays as found."""

    report = _report()

    assert mod.validate_report(report) == []
    assert report["v554_activated_task_count"] == 9
    assert [row["task_id"] for row in report["v554_terminal_rows"]] == list(
        mod.EXPECTED_V554_ACTIVATED_TASK_IDS
    )

    rows = {row["task_id"]: row for row in report["v554_terminal_rows"]}
    assert rows["exp6436-v554-terminal-handoff-and-queue-preflight"]["readiness_fields"][
        "v554_queue_ready_score"
    ] == 0.0
    assert rows["exp6444-csl-lifecycle-recomputation-audit"]["readiness_fields"][
        "csl_audit_ready_score"
    ] == 0.0

    missing = {
        row["task_id"]
        for row in report["v554_missing_zero_byte_or_blocked_artifacts"]
        if row["artifact_state"] == "missing"
    }
    assert missing == {
        "exp6438-powered-verification-cost-repair-ab",
        "exp6439-factor-clause-influence-ab",
        "exp6441-prospective-query-conditioned-factor-reuse",
        "exp6443-fresh-held-restart-csl-replication",
    }
    blocked = {
        row["task_id"]
        for row in report["v554_missing_zero_byte_or_blocked_artifacts"]
        if row["artifact_state"] == "blocked"
    }
    assert blocked >= {
        "exp6437-generation-to-verdict-receipt-replay-contract",
        "exp6440-held-factor-revocation-binding-shift-ab",
        "exp6442-skill-misevolution-quarantine-rollback-ab",
        "exp6444-csl-lifecycle-recomputation-audit",
    }

    determinations = report["v554_terminal_claim_determinations"]
    assert determinations["v554_queue_integrity"]["eligible"] is False
    assert determinations["prospective_csl"]["eligible"] is False
    assert determinations["public_arc"]["eligible"] is False
    assert determinations["hardware"]["eligible"] is False


def test_scenario_report_6448_v555_queue_identity_passes_but_prior_validator_blocks() -> None:
    """SCENARIO-REPORT-6448-V555-QUEUE: exact queue is still fail-closed."""

    report = _report()

    assert report["task_count"] == 12
    assert report["task_ids_in_order"] == list(mod.EXPECTED_V555_TASK_IDS)
    assert report["unique_id_and_deliverable_check"]["ok"] is True
    assert report["milestone_consistency_check"]["ok"] is True
    assert report["schema_validation"]["ok"] is True
    assert report["exclusion_manifest_validation"]["ok"] is True
    assert report["prior_failure_validation"]["ok"] is False
    assert report["structured_gate_validation"]["ok"] is True
    assert report["v555_queue_integrity_score"] == 0.0
    assert report["status"] == "complete_blocked_v555_queue_integrity_failed"
    assert report["gate_check_summary"]["failed_check"] == "prior_failure_validation"
    assert [
        failure["failed_check"] for failure in report["gate_check_summary"]["failed_checks"]
    ] == ["prior_failure_validation"]
    assert "prior_failure_validation" in report["blocked_reason"]
    assert "exp6455-prospective-verifier-bounded-factor-weight-csl" in json.dumps(
        report["prior_failure_validation"], sort_keys=True
    )


def test_scenario_report_6448_gates_models_per_unit_and_prompts() -> None:
    """SCENARIO-REPORT-6448-GATES-PRIORS-MODELS: roadmap contracts are explicit."""

    report = _report()

    assert report["structured_gate_validation"]["gate_count"] == 5
    assert report["structured_gate_validation"]["gate_failures"] == []
    assert all(row["producer_declares_artifact_field"] for row in report["gate_producer_contract_rows"])
    assert {
        (row["consumer_task_id"], row["upstream"], row["artifact_field"])
        for row in report["gate_producer_contract_rows"]
    } >= {
        (
            "exp6451-typed-fact-grounding-fixed-policy-logic-ab",
            "exp6450-sota-fixed-policy-candidate-corpus",
            "sota_corpus_ready_score",
        ),
        (
            "exp6456-corrupt-feedback-held-restart-csl-replication",
            "exp6455-prospective-verifier-bounded-factor-weight-csl",
            "verifier_bounded_csl_ready_score",
        ),
    }

    assert report["model_policy_validation"]["ok"] is True
    assert report["model_policy_validation"]["llm_task_ids"] == list(mod.LLM_TASK_IDS)
    assert report["per_unit_row_contract_validation"]["ok"] is True
    assert report["prompt_terminal_line_validation"]["ok"] is True

    override_rows = {
        row["task_id"]: row for row in report["prior_failure_validation"]["operator_override_rows"]
    }
    assert override_rows["exp6448-v555-terminal-handoff-and-queue-integrity"][
        "cites_standing_transition_directive"
    ] is True


def test_scenario_report_6448_schema_write_and_validation_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-REPORT-6448-SCHEMA: artifact schema is stable."""

    report = _report()

    assert report["protected_files_unchanged"]["ok"] is True
    assert report["verifier_is_oracle"] is False
    assert report["random_seed"] is None
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(report["field_principles"])
    assert "acceptance_gate:v555_queue_integrity_score" in report["field_principles"]
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
        ("set", ("random_seed", 6448), "random_seed must be null"),
        ("set", ("v555_queue_integrity_score", 1.0), "integrity score cannot pass"),
        ("set", ("gate_check_summary", []), "gate_check_summary must be a mapping"),
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

    for field_name, expected in (
        ("field_principles", "field_principles must be a mapping"),
        ("field_provenance", "field_provenance must be a mapping"),
        ("v554_terminal_claim_determinations", "v554_terminal_claim_determinations must be a mapping"),
    ):
        bad = copy.deepcopy(report)
        bad[field_name] = []
        bad["reproducibility_checksum"] = mod.payload_checksum(bad)
        assert expected in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["field_provenance"] = {}
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "field_provenance must cover exactly required fields" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["field_provenance"] = dict.fromkeys(mod.REQUIRED_ARTIFACT_FIELDS, "bad_kind")
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "field_provenance has invalid classification" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["field_principles"] = dict(bad["field_principles"])
    del bad["field_principles"]["status"]
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "missing field_principles entry: status" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["v554_terminal_claim_determinations"]["prospective_csl"]["eligible"] = True
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "prospective_csl must remain blocked" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["blocked_reason"] = None
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "blocked report must name blocked_reason" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["gate_check_summary"] = dict(bad["gate_check_summary"])
    bad["gate_check_summary"]["failed_check"] = None
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "blocked report must name failed_check" in mod.validate_report(bad)

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


def test_req_report_6448_helper_edges_and_dirty_queue_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6448: malformed inputs fail closed without fabrication."""

    assert mod.path_receipt(tmp_path / "missing.json")["present"] is False
    assert mod._artifact_status({}, {"error": "missing", "size_bytes": None}) == "missing"
    assert mod._artifact_status({}, {"error": "json_error", "size_bytes": 4}) == "malformed"
    assert mod._artifact_status({}, {"error": "json_error", "size_bytes": 0}) == "zero_byte"
    assert mod._artifact_state({"status": "complete"}, {"error": None, "size_bytes": 2}) == "complete"
    assert mod._artifact_state({"status": "odd"}, {"error": None, "size_bytes": 2}) == "odd"
    assert mod._tasks({"tasks": "bad"}) == []
    assert mod._is_blocked({}, {"error": "missing"}) is False
    assert mod._is_blocked({"status": "blocked"}, {"error": None}) is True
    assert mod._claim_eligibility_for_task(
        "exp6436-v554-terminal-handoff-and-queue-preflight",
        "complete",
        {"v554_queue_ready_score": 1.0},
    ) == {"eligible": True, "blockers": [], "scope": "queue integrity"}
    assert mod._claim_eligibility_for_task("exp1-other", "complete", {}) == {
        "eligible": False,
        "blockers": ["no_claim_promoted_by_exp6448"],
        "scope": "exp1-other",
    }
    assert mod._v554_claim_determinations(REPO, [])["v554_queue_integrity"]["eligible"] is False
    with monkeypatch.context() as scoped:
        scoped.setattr(mod, "_artifact_meta", lambda root, rel: ({}, {}))
        assert mod._v554_claim_determinations(REPO, [])["prospective_csl"]["blockers"] == [
            "v554_csl_evidence_missing_or_blocked"
        ]
    assert {
        failure["reason"]
        for failure in mod._model_policy_failures(
            "exp-test",
            "Bad/Unexpected-GGUF AutoTokenizer.from_pretrained",
        )
    } >= {
        "missing_model_specs",
        "missing_cache_resolver",
        "missing_embedded_tokenizer_rule",
        "forbidden_autotokenizer_from_pretrained",
    }
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
        lambda root: {
            "roadmap_gate_audit_passed": True,
            "n_prior_failures_missing": 0,
            "failure_details": [],
        },
    )
    prior_validation = mod._prior_validation(
        {"exp6449-generation-to-verdict-path-receipt-contract": {"prior_failures": []}},
        REPO,
    )
    assert prior_validation["required_prior_failure_rows"][0] == {
        "task_id": "exp6449-generation-to-verdict-path-receipt-contract",
        "present_in_roadmap": True,
        "complete": False,
    }
    invalid_prior_validation = mod._prior_validation(
        {
            "exp6449-generation-to-verdict-path-receipt-contract": {
                "prior_failures": [{"experiment_id": "", "verdict": "", "addressed_by": ""}]
            },
            mod.EXPERIMENT_ID: {
                "operator_override": "2026-05-29 operator directive (standing): test"
            },
        },
        REPO,
    )
    assert invalid_prior_validation["required_prior_failure_rows"][0]["complete"] is False
    assert invalid_prior_validation["failures"][0]["reason"] == "missing_experiment_id"

    data = copy.deepcopy(mod.read_yaml_mapping(REPO / mod.ACTIVE_ROADMAP_RELATIVE_PATH))
    checks = mod.validate_v555_queue_data(data, REPO, "20260815", retired_exp_ids=set())
    assert checks["task_count"] == 12
    assert checks["unique_id_and_deliverable_check"]["ok"] is True

    dirty = copy.deepcopy(data)
    tasks = dirty["tasks"]
    tasks[1]["id"] = tasks[0]["id"]
    tasks[2]["deliverable"] = "not-results.txt"
    tasks[3]["gated_on"] = [
        {"upstream": "exp9999-missing", "artifact_field": "missing_field", "op": "??", "value": 1.0}
    ]
    tasks[4]["milestone"] = "2026.08.554"
    tasks[5]["prompt"] = "CONTEXT\nTASK\nCONCRETE STEPS\nRun command: x\nDo NOT push."
    tasks[6]["prompt"] = (
        "CONTEXT\nEXISTING CODE TO READ FIRST\nTASK\nCONCRETE STEPS\n"
        "MODEL_SPECS Bad/Unexpected-GGUF cached_sota_pair() embedded tokenizer "
        "AutoTokenizer.from_pretrained raw output\nRun command: x\n"
        "Do NOT push. Do NOT modify scripts/research_conductor.py."
    )
    tasks[7]["requires"] = [tasks[7]["id"], "exp6448-v555-terminal-handoff-and-queue-integrity"]
    dirty_checks = mod.validate_v555_queue_data(dirty, REPO, "20260815", retired_exp_ids={6448})
    assert dirty_checks["schema_validation"]["ok"] is False
    assert dirty_checks["unique_id_and_deliverable_check"]["ok"] is False
    assert dirty_checks["milestone_consistency_check"]["ok"] is False
    assert dirty_checks["structured_gate_validation"]["ok"] is False
    assert dirty_checks["model_policy_validation"]["ok"] is False
    assert dirty_checks["prompt_terminal_line_validation"]["ok"] is False

    retired_gate = copy.deepcopy(data)
    retired_gate["tasks"][1]["gated_on"] = [
        {
            "upstream": "exp6448-v555-terminal-handoff-and-queue-integrity",
            "artifact_field": "v555_queue_integrity_score",
            "op": "==",
            "value": 1.0,
        }
    ]
    retired_checks = mod.validate_v555_queue_data(
        retired_gate, REPO, "20260815", retired_exp_ids={6448}
    )
    assert retired_checks["structured_gate_validation"]["retired_references"] == [
        {
            "task_id": "exp6449-generation-to-verdict-path-receipt-contract",
            "gate_upstream": "exp6448-v555-terminal-handoff-and-queue-integrity",
        }
    ]

    bad_per_unit = copy.deepcopy(data)
    bad_per_unit["tasks"][4]["per_unit_rows"] = False
    bad_per_unit["tasks"][4]["prompt"] = (
        "CONTEXT\nEXISTING CODE TO READ FIRST\nTASK\nCONCRETE STEPS\n"
        "must compare arms and write blocked_ verdict\nRun command: x\n"
        "Do NOT push. Do NOT modify scripts/research_conductor.py."
    )
    per_unit_checks = mod.validate_v555_queue_data(
        bad_per_unit, REPO, "20260815", retired_exp_ids=set()
    )
    assert {
        failure["reason"]
        for failure in per_unit_checks["per_unit_row_contract_validation"]["failures"]
    } >= {
        "per_unit_rows_not_true",
        "missing_per_unit_rows_required_field",
        "missing_row_emission_rule",
        "missing_gate_check_summary",
    }

    assert mod._first_failed_check({"task_count": 1})["failed_check"] == "task_count"
    assert (
        mod._first_failed_check({"task_count": 12, "schema_validation": {"ok": False}})[
            "failed_check"
        ]
        == "schema_validation"
    )
    failed_tests = mod._first_failed_check(
        {"task_count": 12, "tests_run": [{"command": "full", "exit_code": 130}]}
    )
    assert failed_tests["failed_check"] == "tests_run"
    assert failed_tests["failed_checks"][0]["observed_value"] == [
        {"command": "full", "exit_code": 130}
    ]
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
    monkeypatch.setattr(mod, "protected_hashes", lambda root: {"x": "sha256:x"})
    monkeypatch.setattr(mod, "read_external_test_receipts", lambda: [{"command": "external"}])
    monkeypatch.setattr(
        mod,
        "build_report",
        lambda root, *, date, command_receipts, before_hashes, duration_s: {
            "date": date,
            "command_receipts": command_receipts,
            "before_hashes": before_hashes,
            "duration_s": duration_s,
            "reproducibility_checksum": "sha256:fake",
        },
    )
    monkeypatch.setattr(mod, "validate_report", lambda report: [])
    monkeypatch.setattr(mod, "write_report", lambda report, root: writes.append(report))

    report = mod.run(date="20260815", root=REPO, write=True)
    assert report["command_receipts"] == [{"command": "external"}]
    assert writes == [report]

    monkeypatch.setattr(mod, "validate_report", lambda report: ["bad"])
    with pytest.raises(ValueError, match="bad"):
        mod.run(date="20260815", root=REPO, write=False, command_receipts=[{"command": "c"}])
