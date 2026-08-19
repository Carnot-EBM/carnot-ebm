"""Tests for Exp6460 V556 queue integrity.

Spec refs: REQ-REPORT-6460, SCENARIO-REPORT-6460-V555-FREEZE,
SCENARIO-REPORT-6460-V556-QUEUE,
SCENARIO-REPORT-6460-GATES-PRIORS-MODELS,
SCENARIO-REPORT-6460-SCHEMA.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_6460_v556_terminal_handoff_and_queue_integrity as mod
from carnot.experiment_artifacts import ARTIFACT_ROOT_ENV


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH
_REPORT_CACHE: dict[str, object] | None = None


def _report() -> dict[str, object]:
    global _REPORT_CACHE
    if _REPORT_CACHE is None:
        _REPORT_CACHE = mod.build_report(
            REPO,
            date="20260819",
            command_receipts=[{"command": "focused", "exit_code": 0}],
            before_hashes=mod.protected_hashes(REPO),
            duration_s=1.0,
        )
    return copy.deepcopy(_REPORT_CACHE)


def test_req_report_6460_spec_declares_required_contract() -> None:
    """REQ-REPORT-6460: OpenSpec owns the V556 queue audit contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-REPORT-6460") :]

    assert mod.RUN_DATE == "20260819"
    assert "--date 20260819" in mod.RUN_COMMAND
    for marker in (
        "SCENARIO-REPORT-6460-V555-FREEZE",
        "SCENARIO-REPORT-6460-V556-QUEUE",
        "SCENARIO-REPORT-6460-GATES-PRIORS-MODELS",
        "SCENARIO-REPORT-6460-SCHEMA",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_report_6460_v555_terminal_rows_preserve_mixed_evidence() -> None:
    """SCENARIO-REPORT-6460-V555-FREEZE: V555 evidence stays as found."""

    report = _report()

    assert mod.validate_report(report) == []
    assert [row["task_id"] for row in report["v555_terminal_rows"]] == list(
        mod.EXPECTED_V555_TASK_IDS
    )
    rows = {row["task_id"]: row for row in report["v555_terminal_rows"]}
    assert (
        rows["exp6450-sota-fixed-policy-candidate-corpus"]["readiness_fields"][
            "sota_corpus_ready_score"
        ]
        == 0.0
    )
    assert (
        rows["exp6457-independent-verifier-bounded-csl-audit"]["readiness_fields"][
            "csl_audit_ready_score"
        ]
        == 0.0
    )
    assert (
        rows["exp6458-arc-representation-objective-generalization-ab"]["readiness_fields"][
            "arc_objective_generalization_ready_score"
        ]
        == 0.0
    )

    missing = {
        row["task_id"] for row in report["v555_terminal_rows"] if row["artifact_state"] == "missing"
    }
    assert missing == {
        "exp6452-representation-objective-causal-ab",
        "exp6454-held-exact-constraint-energy-selection-ab",
    }
    blocked = {
        row["task_id"] for row in report["v555_terminal_rows"] if row["artifact_state"] == "blocked"
    }
    assert blocked >= {
        "exp6451-typed-fact-grounding-fixed-policy-logic-ab",
        "exp6453-held-verifier-budget-allocation-ab",
        "exp6459-v555-adversarial-capstone",
    }
    assert (
        rows["exp6449-generation-to-verdict-path-receipt-contract"]["final_claim_eligibility"][
            "eligible"
        ]
        is True
    )
    assert (
        rows["exp6455-prospective-verifier-bounded-factor-weight-csl"]["final_claim_eligibility"][
            "eligible"
        ]
        is False
    )


def test_scenario_report_6460_v556_queue_identity_and_validators_pass() -> None:
    """SCENARIO-REPORT-6460-V556-QUEUE: exact V556 queue validates."""

    report = _report()

    assert report["active_roadmap_hash"]["milestone"] == mod.MILESTONE_V556
    assert report["staged_roadmap_hash"]["present"] is False
    assert report["task_count"] == 13
    assert report["task_ids_in_order"] == list(mod.EXPECTED_V556_TASK_IDS)
    assert report["unique_id_and_deliverable_check"]["ok"] is True
    assert report["milestone_consistency_check"]["ok"] is True
    assert report["schema_validation"]["ok"] is True
    assert report["prior_failure_validation"]["ok"] is True
    assert report["exclusion_manifest_validation"]["ok"] is True
    assert report["structured_gate_validation"]["ok"] is True
    assert report["v556_queue_integrity_score"] == 1.0
    assert report["status"] == "complete_v556_queue_integrity_passed"
    assert report["blocked_reason"] is None
    assert report["gate_check_summary"]["failed_check"] is None


def test_scenario_report_6460_gates_models_per_unit_and_prompts() -> None:
    """SCENARIO-REPORT-6460-GATES-PRIORS-MODELS: roadmap contracts are explicit."""

    report = _report()

    assert report["structured_gate_validation"]["gate_count"] == 6
    assert report["structured_gate_validation"]["gate_failures"] == []
    assert all(
        row["producer_declares_artifact_field"] for row in report["gate_producer_contract_rows"]
    )
    assert {
        (row["consumer_task_id"], row["upstream"], row["artifact_field"])
        for row in report["gate_producer_contract_rows"]
    } >= {
        (
            "exp6463-sota-fixed-policy-candidate-corpus-v2",
            "exp6462-sota-raw-persistence-uniqueness-canary",
            "raw_persistence_canary_ready_score",
        ),
        (
            "exp6469-unique-event-csl-corruption-restart",
            "exp6468-unique-event-verifier-bounded-csl",
            "unique_event_csl_ready_score",
        ),
    }
    assert report["model_policy_validation"]["ok"] is True
    assert report["model_policy_validation"]["llm_task_ids"] == list(mod.LIVE_LLM_TASK_IDS)
    assert report["per_unit_rows"]["ok"] is True
    assert report["per_unit_rows"]["task_ids_missing_flag"] == []
    assert report["prompt_terminal_line_validation"]["ok"] is True


def test_scenario_report_6460_schema_write_and_validation_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-REPORT-6460-SCHEMA: artifact schema is stable."""

    report = _report()

    assert report["protected_files_unchanged"]["ok"] is True
    assert report["verifier_is_oracle"] is False
    assert report["random_seed"] is None
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(report["field_principles"])
    assert "acceptance_gate:v556_queue_integrity_score" in report["field_principles"]
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
        ("set", ("random_seed", 6460), "random_seed must be null"),
        ("set", ("v556_queue_integrity_score", 1.0), "integrity score cannot pass"),
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
            if expected == "integrity score cannot pass":
                bad["schema_validation"] = {"ok": False}
            bad["reproducibility_checksum"] = mod.payload_checksum(bad)
        assert any(expected in error for error in mod.validate_report(bad))

    for field_name, expected in (
        ("field_principles", "field_principles must be a mapping"),
        ("field_provenance", "field_provenance must be a mapping"),
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
    bad["v556_queue_integrity_score"] = 0.0
    bad["blocked_reason"] = None
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "blocked report must name blocked_reason" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["status"] = "complete_blocked_v556_queue_integrity_failed"
    bad["v556_queue_integrity_score"] = 0.0
    bad["blocked_reason"] = "x"
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
    assert mod.main(["--date", "20260819"]) == 0
    assert mod.RESULT_RELATIVE_PATH.name in capsys.readouterr().out


def test_req_report_6460_helper_edges_and_dirty_queue_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6460: malformed inputs fail closed without fabrication."""

    assert mod.path_receipt(tmp_path / "missing.json")["present"] is False
    assert mod._artifact_status({}, {"error": "missing", "size_bytes": None}) == "missing"
    assert mod._artifact_status({}, {"error": "json_error", "size_bytes": 4}) == "malformed"
    assert mod._artifact_status({}, {"error": "json_error", "size_bytes": 0}) == "zero_byte"
    assert (
        mod._artifact_state({"status": "complete"}, {"error": None, "size_bytes": 2}) == "complete"
    )
    assert mod._artifact_state({"status": "odd"}, {"error": None, "size_bytes": 2}) == "odd"
    assert mod._tasks({"tasks": "bad"}) == []
    assert mod._is_blocked({}, {"error": "missing"}) is False
    assert mod._is_blocked({"status": "blocked"}, {"error": None}) is True
    assert mod._claim_eligibility_for_task(
        "exp6449-generation-to-verdict-path-receipt-contract",
        "complete",
        {"path_receipt_ready_score": 1.0},
    ) == {"eligible": True, "blockers": [], "scope": "path receipt"}
    assert mod._claim_eligibility_for_task("exp1-other", "complete", {}) == {
        "eligible": False,
        "blockers": ["no_claim_promoted_by_exp6460"],
        "scope": "exp1-other",
    }
    assert {
        failure["reason"]
        for failure in mod._model_policy_failures(
            "exp-test",
            "Bad/Unexpected-GGUF AutoTokenizer.from_pretrained",
        )
    } >= {
        "missing_model_specs",
        "missing_cache_resolution",
        "missing_embedded_tokenizer_rule",
        "forbidden_autotokenizer_from_pretrained",
    }
    gate_rows, gate_failures, gate_expressions = mod._gate_rows(
        [{"id": "exp1-test", "gated_on": ["bad"]}], {}, {}
    )
    assert gate_rows == []
    assert gate_expressions == []
    assert gate_failures == [{"task_id": "exp1-test", "reason": "gate_not_mapping", "gate": "bad"}]

    active_data = copy.deepcopy(mod.read_yaml_mapping(REPO / mod.ACTIVE_ROADMAP_RELATIVE_PATH))
    checks = mod.validate_v556_queue_data(
        active_data,
        REPO,
        "20260819",
        roadmap_path=mod.ACTIVE_ROADMAP_RELATIVE_PATH,
        retired_exp_ids=set(),
    )
    assert checks["task_count"] == 13
    assert checks["unique_id_and_deliverable_check"]["ok"] is True

    dirty = copy.deepcopy(active_data)
    tasks = dirty["tasks"]
    tasks[1]["id"] = tasks[0]["id"]
    tasks[2]["deliverable"] = "not-results.txt"
    tasks[3]["gated_on"] = [
        {"upstream": "exp9999-missing", "artifact_field": "missing_field", "op": "??", "value": 1.0}
    ]
    tasks[4]["milestone"] = "2026.08.555"
    tasks[5]["prompt"] = "CONTEXT\nTASK\nCONCRETE STEPS\nRun command: x\nDo NOT push."
    tasks[6]["prompt"] = (
        "CONTEXT\nEXISTING CODE TO READ FIRST\nTASK\nCONCRETE STEPS\n"
        "MODEL_SPECS Bad/Unexpected-GGUF embedded tokenizer "
        "AutoTokenizer.from_pretrained compare arms blocked_ verdict\nRun command: x\n"
        "Do NOT push. Do NOT modify scripts/research_conductor.py."
    )
    tasks[7]["requires"] = [tasks[7]["id"], "exp6448-v555-terminal-handoff-and-queue-integrity"]
    dirty_checks = mod.validate_v556_queue_data(
        dirty,
        REPO,
        "20260819",
        roadmap_path=mod.ACTIVE_ROADMAP_RELATIVE_PATH,
        retired_exp_ids={6448},
    )
    assert dirty_checks["schema_validation"]["ok"] is False
    assert dirty_checks["unique_id_and_deliverable_check"]["ok"] is False
    assert dirty_checks["milestone_consistency_check"]["ok"] is False
    assert dirty_checks["structured_gate_validation"]["ok"] is False
    assert dirty_checks["model_policy_validation"]["ok"] is False
    assert dirty_checks["prompt_terminal_line_validation"]["ok"] is False

    bad_per_unit = copy.deepcopy(active_data)
    bad_per_unit["tasks"][4]["per_unit_rows"] = False
    bad_per_unit["tasks"][4]["prompt"] = (
        "CONTEXT\nEXISTING CODE TO READ FIRST\nTASK\nCONCRETE STEPS\n"
        "must compare arms and write blocked_ verdict\nRun command: x\n"
        "Do NOT push. Do NOT modify scripts/research_conductor.py."
    )
    per_unit_checks = mod.validate_v556_queue_data(
        bad_per_unit,
        REPO,
        "20260819",
        roadmap_path=mod.ACTIVE_ROADMAP_RELATIVE_PATH,
        retired_exp_ids=set(),
    )
    assert {failure["reason"] for failure in per_unit_checks["per_unit_rows"]["failures"]} >= {
        "per_unit_rows_not_true",
        "missing_per_unit_rows_required_field",
        "missing_row_emission_rule",
        "missing_gate_check_summary",
    }

    assert mod._first_failed_check({"task_count": 1})["failed_check"] == "task_count"
    assert (
        mod._first_failed_check({"task_count": 13, "schema_validation": {"ok": False}})[
            "failed_check"
        ]
        == "schema_validation"
    )
    failed_tests = mod._first_failed_check(
        {"task_count": 13, "tests_run": [{"command": "full", "exit_code": 130}]}
    )
    assert failed_tests["failed_check"] == "tests_run"
    assert failed_tests["failed_checks"][0]["observed_value"] == [
        {"command": "full", "exit_code": 130}
    ]
    assert mod._first_failed_check({"task_count": 13})["failed_check"] == "unknown"
    assert mod._test_rows(None)[0]["source"] == "declared"

    receipt = tmp_path / "receipts.json"
    assert mod.read_external_test_receipts(receipt) == []
    receipt.write_text("{", encoding="utf-8")
    assert mod.read_external_test_receipts(receipt) == []
    receipt.write_text("{}", encoding="utf-8")
    assert mod.read_external_test_receipts(receipt) == []
    receipt.write_text('[{"command": "ok", "exit_code": 0}, "skip"]', encoding="utf-8")
    assert mod.read_external_test_receipts(receipt) == [{"command": "ok", "exit_code": 0}]

    active = tmp_path / "research-roadmap.yaml"
    staged = tmp_path / "research-roadmap-next.yaml"
    active.write_text("milestone: '2026.08.555'\ntasks: []\n", encoding="utf-8")
    staged.write_text("milestone: '2026.08.556'\ntasks: []\n", encoding="utf-8")
    selected_data, selected_receipt = mod.select_v556_roadmap(tmp_path)
    assert selected_data["milestone"] == mod.MILESTONE_V556
    assert selected_receipt["selected_path"] == mod.STAGED_ROADMAP_RELATIVE_PATH.as_posix()

    staged.write_text("milestone: '2026.08.557'\ntasks: []\n", encoding="utf-8")
    selected_data, selected_receipt = mod.select_v556_roadmap(tmp_path)
    assert selected_data["milestone"] == "2026.08.557"
    assert selected_receipt["selection_note"] == "V556 roadmap milestone was not found"

    assert {
        failure["reason"]
        for failure in mod._model_policy_failures(
            "exp-test",
            ("MODEL_SPECS unsloth/Qwen3.6-35B-A3B-GGUF cached paths embedded tokenizer Qwen3.5"),
        )
    } >= {"headline_legacy_model"}

    monkeypatch.setattr(
        mod,
        "_prior_failure_linter",
        lambda root, roadmap_path: {"schema_errors": [], "prior_failure_violations": []},
    )
    monkeypatch.setattr(
        mod,
        "_gate_audit",
        lambda root, roadmap_path: {
            "roadmap_gate_audit_passed": True,
            "n_prior_failures_missing": 0,
            "failure_details": [],
        },
    )
    prior_validation = mod._prior_validation(
        {
            "exp-no-prior": {},
            "exp-empty-prior": {"prior_failures": []},
            "exp-bad-prior": {
                "prior_failures": [{"experiment_id": "", "verdict": "", "addressed_by": ""}]
            },
        },
        REPO,
        mod.ACTIVE_ROADMAP_RELATIVE_PATH,
    )
    assert {row["task_id"] for row in prior_validation["prior_failure_rows"]} == {
        "exp-empty-prior",
        "exp-bad-prior",
    }
    assert {failure["reason"] for failure in prior_validation["failures"]} >= {
        "missing_or_empty_prior_failures",
        "missing_experiment_id",
    }

    retired_gate = copy.deepcopy(active_data)
    retired_gate["tasks"][1]["gated_on"] = [
        {
            "upstream": "exp6448-v555-terminal-handoff-and-queue-integrity",
            "artifact_field": "v555_queue_integrity_score",
            "op": "==",
            "value": 1.0,
        }
    ]
    retired_gate_checks = mod.validate_v556_queue_data(
        retired_gate,
        REPO,
        "20260819",
        roadmap_path=mod.ACTIVE_ROADMAP_RELATIVE_PATH,
        retired_exp_ids={6448},
    )
    assert retired_gate_checks["structured_gate_validation"]["retired_references"] == [
        {
            "task_id": "exp6461-v556-primary-source-freshness-receipt",
            "gate_upstream": "exp6448-v555-terminal-handoff-and-queue-integrity",
        }
    ]

    late_gate = copy.deepcopy(active_data)
    late_gate["tasks"][1]["gated_on"] = [
        {
            "upstream": late_gate["tasks"][2]["id"],
            "artifact_field": "status",
            "op": "exists",
            "value": True,
        }
    ]
    late_gate_checks = mod.validate_v556_queue_data(
        late_gate,
        REPO,
        "20260819",
        roadmap_path=mod.ACTIVE_ROADMAP_RELATIVE_PATH,
        retired_exp_ids=set(),
    )
    assert late_gate_checks["structured_gate_validation"]["gate_order_failures"] == [
        {
            "task_id": "exp6461-v556-primary-source-freshness-receipt",
            "gate_upstream": "exp6462-sota-raw-persistence-uniqueness-canary",
        }
    ]

    fake_checks = {
        "task_count": 13,
        "task_ids_in_order": list(mod.EXPECTED_V556_TASK_IDS),
        "unique_id_and_deliverable_check": {"ok": True},
        "milestone_consistency_check": {"ok": True},
        "schema_validation": {"ok": False},
        "prior_failure_validation": {"ok": True},
        "exclusion_manifest_validation": {"ok": True},
        "structured_gate_validation": {"ok": True},
        "gate_producer_contract_rows": [],
        "model_policy_validation": {"ok": True},
        "per_unit_rows": {"ok": True},
        "prompt_terminal_line_validation": {"ok": True},
    }
    monkeypatch.setattr(
        mod,
        "select_v556_roadmap",
        lambda root: (
            {"milestone": mod.MILESTONE_V556},
            {
                "selected_path": mod.ACTIVE_ROADMAP_RELATIVE_PATH.as_posix(),
                "selection_note": "fixture",
                "active_roadmap": {"present": True},
                "staged_roadmap": {"present": False},
            },
        ),
    )
    monkeypatch.setattr(mod, "_v555_terminal_rows", lambda root: [])
    monkeypatch.setattr(mod, "validate_v556_queue_data", lambda *args, **kwargs: fake_checks)
    monkeypatch.setattr(mod, "protected_hashes", lambda root: {"x": "sha256:x"})
    blocked_report = mod.build_report(
        tmp_path,
        date="20260819",
        command_receipts=[{"command": "focused", "exit_code": 0}],
        before_hashes={"x": "sha256:x"},
        duration_s=0.1,
    )
    assert blocked_report["status"] == "complete_blocked_v556_queue_integrity_failed"
    assert blocked_report["gate_check_summary"]["failed_check"] == "schema_validation"

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

    report = mod.run(date="20260819", root=REPO, write=True)
    assert report["command_receipts"] == [{"command": "external"}]
    assert writes == [report]

    monkeypatch.setattr(mod, "validate_report", lambda report: ["bad"])
    with pytest.raises(ValueError, match="bad"):
        mod.run(date="20260819", root=REPO, write=False, command_receipts=[{"command": "c"}])
