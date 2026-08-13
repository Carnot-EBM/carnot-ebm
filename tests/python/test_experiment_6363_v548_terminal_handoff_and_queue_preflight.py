"""Tests for Exp6363 V548 terminal handoff.

Spec refs: REQ-INFRA-6363, SCENARIO-INFRA-6363-1,
SCENARIO-INFRA-6363-2, SCENARIO-INFRA-6363-3,
SCENARIO-INFRA-6363-4, SCENARIO-INFRA-6363-5.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_6363_v548_terminal_handoff_and_queue_preflight as mod
from carnot.experiment_artifacts import ARTIFACT_ROOT_ENV


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _report() -> dict[str, object]:
    return mod.build_report(
        REPO,
        date="20260813",
        command_receipts=[{"command": "focused", "exit_code": 0}],
        before_hashes=mod.protected_hashes(REPO),
        duration_s=1.0,
    )


def test_req_infra_6363_spec_declares_fields_and_scenarios() -> None:
    """REQ-INFRA-6363: OpenSpec records the V548 handoff contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6363") : text.index("REQ-INFRA-6351")]

    for marker in (
        "SCENARIO-INFRA-6363-1",
        "SCENARIO-INFRA-6363-2",
        "SCENARIO-INFRA-6363-3",
        "SCENARIO-INFRA-6363-4",
        "SCENARIO-INFRA-6363-5",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_infra_6363_v547_states_and_proposal_only_ids() -> None:
    """SCENARIO-INFRA-6363-1: V547 states stay separate and exact."""

    report = _report()

    assert mod.validate_report(report) == []
    assert report["v547_active_task_ids"] == list(mod.EXPECTED_V547_TASK_IDS)
    assert report["proposal_only_v547_ids_not_executed"]["task_ids"] == list(
        mod.PROPOSAL_ONLY_V547_IDS
    )
    assert report["proposal_only_v547_ids_not_executed"]["executed_count"] == 0

    rows = report["v547_terminal_artifacts_by_task"]
    for task_id, row in rows.items():
        if task_id == "exp6354-prospective-live-certified-factor-learning":
            assert row["summary_receipt"]["invoked_before_field_import"] is False
        else:
            assert row["summary_receipt"]["invoked_before_field_import"] is True

    states = report["v547_flagged_blocked_missing_and_null_states"]
    assert list(states["flagged"]) == ["exp6350-v547-bounded-terminal-handoff"]
    assert rows["exp6350-v547-bounded-terminal-handoff"]["terminal_class"] == "flagged"
    assert states["blocked"] == [
        "exp6353-live-counterexample-factor-proposal-ab",
        "exp6355-default-off-certified-factor-consumer-ab",
    ]
    assert states["missing"] == ["exp6354-prospective-live-certified-factor-learning"]
    assert states["retired_upstream"] == ["exp6354-prospective-live-certified-factor-learning"]
    assert states["null"] == [
        "exp6351-v547-post-marker-source-scope-freeze",
        "exp6352-live-factor-proposal-authenticity-preflight",
        "exp6356-live-certified-learning-safety-audit",
    ]

    outcomes = report["v547_conductor_outcomes_and_attempt_counts"]
    assert outcomes["exp6353-live-counterexample-factor-proposal-ab"]["GATE_BLOCK"] == 3
    assert outcomes["exp6354-prospective-live-certified-factor-learning"]["GATE_BLOCK"] == 3
    assert (
        outcomes["exp6354-prospective-live-certified-factor-learning"]["preemptive_skip_count"] == 3
    )
    assert outcomes["exp6355-default-off-certified-factor-consumer-ab"]["GATE_BLOCK"] == 3
    assert outcomes["exp6356-live-certified-learning-safety-audit"]["OK"] == 1


def test_scenario_infra_6363_exp6352_failure_and_drift_receipts() -> None:
    """SCENARIO-INFRA-6363-2: Exp6352 drift is recorded without a diagnosis."""

    report = _report()
    failure = report["exp6352_generation_failure_receipt"]
    drift = report["exp6352_source_artifact_drift_receipt"]

    assert failure["all_generation_children_returned_code_1"] is True
    assert failure["total_raw_byte_count"] == 0
    assert failure["total_prompt_tokens"] == 0
    assert failure["total_completion_tokens"] == 0
    assert failure["models_used_empty"] is True
    assert failure["live_autoregressive_generation_invoked"] is False
    assert failure["stderr_preserved_in_artifact"] is False
    assert failure["root_cause_inferred"] is False

    assert drift["source_sampling_n_ctx"] == 2048
    assert drift["artifact_process_n_ctx_values"] == [512]
    assert drift["n_ctx_mismatch"] is True
    assert drift["top_level_random_seed_present"] is False
    assert drift["prose_vs_boolean_generation_contradiction"] is True
    assert drift["root_cause_inferred"] is False


def test_scenario_infra_6363_v548_queue_checks_fail_closed() -> None:
    """SCENARIO-INFRA-6363-3 and 4: current V548 queue drift is explicit."""

    report = _report()

    hashes = report["v548_milestone_doc_and_queue_hashes"]
    assert hashes["requested_next_roadmap"]["present"] is False
    assert hashes["audited_queue"]["path"] == "research-roadmap.yaml"

    assert report["v548_task_ids"] == list(mod.ACTIVE_V548_TASK_IDS)
    id_check = report["v548_id_collision_check"]
    assert id_check["ok"] is False
    assert id_check["task_count"] == 4
    assert id_check["expected_task_count"] == 14
    assert id_check["missing_expected_task_ids"] == list(mod.MISSING_ACTIVE_V548_TASK_IDS)

    assert report["v548_deliverable_checks"]["unique_deliverables"] is True
    assert report["v548_dependency_and_structured_gate_checks"]["gate_count"] == 1
    assert report["v548_dependency_and_structured_gate_checks"]["ok"] is True
    assert report["v548_gate_field_cross_reference_checks"]["ok"] is True
    assert report["v548_prior_failure_checks"]["ok"] is True
    assert report["v548_agent_model_and_llm_policy_checks"]["ok"] is True

    prompt_checks = report["prompt_contract_checks"]
    assert prompt_checks["ok"] is False
    assert prompt_checks["checked_task_count"] == 4
    assert {row["reason"] for row in prompt_checks["failures"]} >= {
        "missing_project_root_literal",
        "missing_date_literal",
    }


def test_scenario_infra_6363_report_schema_write_and_validation_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-INFRA-6363-5: output is annotated, checksummed, and atomic."""

    report = _report()

    assert report["status"] == "blocked_v548_queue_incomplete"
    assert report["honest_verdict"].startswith("blocked_v548_queue_incomplete:")
    assert report["active_roadmap_modified"] is False
    assert report["conductor_modified"] is False
    assert report["verifier_is_oracle"] is False
    assert report["random_seed"] is None
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(report["field_principles"])
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) == set(report["field_provenance"])
    for expression in report["v548_dependency_and_structured_gate_checks"][
        "structured_gate_expressions"
    ]:
        assert expression in report["field_principles"]
    assert report["reproducibility_checksum"] == mod.payload_checksum(report)

    bad = copy.deepcopy(report)
    del bad["status"]
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "missing required field: status" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["verifier_is_oracle"] = True
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "verifier_is_oracle must be false" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["v547_flagged_blocked_missing_and_null_states"]["flagged"] = []
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "Exp6350 flagged state must be preserved" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["field_principles"] = {}
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert any("missing field_principles entry" in err for err in mod.validate_report(bad))

    bad = copy.deepcopy(report)
    bad["field_principles"] = []
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "field_principles must be a mapping" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["field_provenance"] = []
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "field_provenance must be a mapping" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["field_provenance"] = {}
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "field_provenance must cover exactly required fields" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["random_seed"] = 6363
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "random_seed must be null" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["honest_verdict"] = "ok"
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "honest_verdict lacks terminal prefix" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum mismatch" in mod.validate_report(bad)

    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    path = mod.write_report(report, REPO, env={ARTIFACT_ROOT_ENV: str(artifact_root)})
    assert path == artifact_root / mod.RESULT_RELATIVE_PATH.name
    assert json.loads(path.read_text(encoding="utf-8")) == report

    monkeypatch.setattr(
        mod,
        "run",
        lambda *, date, root=REPO, write=True, command_receipts=None: {"status": f"blocked-{date}"},
    )
    assert mod.main(["--date", "20260813"]) == 0
    assert mod.RESULT_RELATIVE_PATH.name in capsys.readouterr().out


def test_req_infra_6363_helper_edges_and_dirty_queue_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFRA-6363: helper edges fail closed without fabricating fields."""

    missing_payload, missing_meta = mod.read_json_mapping(tmp_path / "missing.json")
    assert missing_payload == {}
    assert missing_meta["error"] == "missing"

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert mod.read_json_mapping(bad_json)[1]["error"].startswith("json_error:")

    not_mapping = tmp_path / "array.json"
    not_mapping.write_text("[]", encoding="utf-8")
    assert mod.read_json_mapping(not_mapping)[1]["error"] == "json_not_mapping"

    assert mod._conductor_rows(tmp_path, "exp6350-v547-bounded-terminal-handoff") == []
    log = tmp_path / mod.CONDUCTOR_LOG_RELATIVE_PATH
    log.parent.mkdir(parents=True)
    log.write_text("Bounded V546 terminal evidence handoff into V547\n", encoding="utf-8")
    assert mod._conductor_rows(tmp_path, "exp6350-v547-bounded-terminal-handoff") == []

    active = tmp_path / mod.ACTIVE_ROADMAP_RELATIVE_PATH
    next_path = tmp_path / mod.ROADMAP_NEXT_RELATIVE_PATH
    active.write_text(
        "milestone: '2026.08.000'\nmilestone_title: Old\nmilestone_doc: doc.md\ntasks: []\n",
        encoding="utf-8",
    )
    next_path.write_text(
        "milestone: '2026.08.548'\nmilestone_title: Next\nmilestone_doc: doc.md\ntasks: []\n",
        encoding="utf-8",
    )
    _, identity = mod.load_v548_queue(tmp_path)
    assert identity["requested_next_roadmap"]["present"] is True
    assert identity["audited_queue"]["path"] == mod.ROADMAP_NEXT_RELATIVE_PATH.as_posix()
    assert mod._tasks({"tasks": "not-list"}) == []

    dirty = {
        "milestone": mod.MILESTONE_V548,
        "milestone_title": "Dirty",
        "milestone_doc": "doc.md",
        "tasks": [
            {
                "id": "exp6363-v548-terminal-handoff-and-queue-preflight",
                "milestone": mod.MILESTONE_V548,
                "deliverable": "not-results.txt",
                "title": "Dirty one",
                "prompt": "REQUIRED ARTIFACT FIELDS:\n- status\nRun command: broken",
                "requires": [
                    "exp6363-v548-terminal-handoff-and-queue-preflight",
                    "exp9999-missing",
                ],
                "gated_on": [
                    "not-a-gate",
                    {
                        "upstream": "exp6364-v548-post-marker-source-scope-freeze",
                        "artifact_field": "not_declared",
                        "op": "==",
                        "value": 1.0,
                    },
                    {
                        "upstream": "exp6364-v548-post-marker-source-scope-freeze",
                        "artifact_field": "status",
                        "op": "contains",
                        "value": "ready",
                    },
                ],
                "prior_failures": [],
                "agent_type": "codex",
                "model": "opus",
                "requires_gpu": True,
            },
            {
                "id": "exp6364-v548-post-marker-source-scope-freeze",
                "milestone": mod.MILESTONE_V548,
                "deliverable": "results/dirty.json",
                "title": "Dirty two",
                "prompt": (
                    "REQUIRED ARTIFACT FIELDS:\n- status\n"
                    "Bad/Unexpected-GGUF\nRun command: broken\n"
                    "Do NOT push. Do NOT modify scripts/research_conductor.py."
                ),
                "prior_failures": [
                    {
                        "experiment_id": "",
                        "verdict": "",
                        "addressed_by": "",
                        "retire_if_same_verdict": False,
                    }
                ],
                "agent_type": "gemini",
                "model": "gemini-3.1-pro-preview",
            },
            {
                "id": "exp6365-gguf-child-failure-forensics-runtime-contract",
                "milestone": mod.MILESTONE_V548,
                "deliverable": "results/dirty-3.json",
                "title": "Dirty three",
                "prompt": "REQUIRED ARTIFACT FIELDS:\n- status\nRun command: broken",
                "prior_failures": [
                    {
                        "experiment_id": "exp6352-live-factor-proposal-authenticity-preflight",
                        "verdict": "blocked",
                        "addressed_by": "changed",
                        "retire_if_same_verdict": True,
                    }
                ],
                "agent_type": "claude",
                "model": "gpt-5.5",
            },
        ],
    }
    checks = mod.validate_v548_queue_data(dirty, REPO, "20260813")
    assert checks["schema_validation"]["ok"] is False
    assert checks["v548_deliverable_checks"]["ok"] is False
    assert checks["v548_dependency_and_structured_gate_checks"]["ok"] is False
    assert checks["v548_gate_field_cross_reference_checks"]["ok"] is False
    assert checks["v548_prior_failure_checks"]["ok"] is False
    assert checks["v548_agent_model_and_llm_policy_checks"]["ok"] is False
    assert checks["prompt_contract_checks"]["ok"] is False
    assert mod._test_rows(None)[0]["source"] == "declared"


def test_req_infra_6363_receipt_reader_and_run_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFRA-6363: external receipts and run branches are explicit."""

    receipt = tmp_path / "receipts.json"
    assert mod.read_external_test_receipts(receipt) == []
    receipt.write_text("{", encoding="utf-8")
    assert mod.read_external_test_receipts(receipt) == []
    receipt.write_text("{}", encoding="utf-8")
    assert mod.read_external_test_receipts(receipt) == []
    receipt.write_text('[{"command": "ok", "exit_code": 0}, "skip"]', encoding="utf-8")
    assert mod.read_external_test_receipts(receipt) == [{"command": "ok", "exit_code": 0}]

    writes: list[dict[str, object]] = []

    def fake_build_report(
        root: Path,
        *,
        date: str,
        command_receipts: list[dict[str, object]],
        before_hashes: dict[str, str],
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

    report = mod.run(date="20260813", root=REPO, write=True)
    assert report["command_receipts"] == [{"command": "external"}]
    assert writes == [report]

    monkeypatch.setattr(mod, "validate_report", lambda report: ["bad"])
    with pytest.raises(ValueError, match="bad"):
        mod.run(date="20260813", root=REPO, write=False, command_receipts=[{"command": "c"}])
