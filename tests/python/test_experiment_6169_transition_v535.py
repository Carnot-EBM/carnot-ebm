"""Tests for the Exp6169 V535 transition receipt.

Spec refs: REQ-REPORT-6169,
SCENARIO-REPORT-6169-EXACT-TERMINAL,
SCENARIO-REPORT-6169-APPEND-ONCE,
SCENARIO-REPORT-6169-ROADMAP-VALIDATION,
SCENARIO-REPORT-6169-PARTIAL-ACTIVATION-BLOCKS,
SCENARIO-REPORT-6169-SCHEMA.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import yaml

from carnot import experiment_6169_transition_v535 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"


def _write_text(root: Path, rel_path: Path | str, text: str = "fixture\n") -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(root: Path, rel_path: Path | str, payload: Any) -> None:
    _write_text(root, rel_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _source_capstone_payload() -> JsonDict:
    matrix: JsonDict = {}
    classes = {
        "exp6156-transition-v534": ("complete", "complete", True, "OK"),
        "exp6157-repo-wide-artifact-isolation-closure": ("missing", "missing", False, "FAIL"),
        "exp6158-v534-source-delta-ingestion": ("null", "null", True, "OK"),
        "exp6159-decision-calibrated-stream": ("positive", "positive", True, "OK"),
        "exp6160-sota-decision-calibration-corpus": ("positive", "positive", True, "OK"),
        "exp6161-decision-calibrated-energy-policy": ("flagged", "positive", True, "FLAGGED"),
        "exp6162-prospective-admission-replication": ("flagged", "positive", True, "FLAGGED"),
        "exp6163-certified-strategy-store-scaleup": ("skipped", "skipped", False, "GATE_BLOCK"),
        "exp6164-continuous-strategy-learning-ab": ("internal_blocked", "blocked", True, "FAIL"),
        "exp6165-strategy-memory-shadow-adapter": ("skipped", "skipped", False, "GATE_BLOCK"),
        "exp6166-mode-jumping-factor-thermalization": ("blocked", "blocked", True, "FAIL"),
        "exp6167-arc-task-aware-multiseed-replication": ("positive", "positive", True, "OK"),
    }
    for task_id, title, deliverable in mod.SOURCE_TASKS_WITHOUT_CAPSTONE:
        terminal, underlying, present, conductor_status = classes[task_id]
        matrix[task_id] = {
            "task_id": task_id,
            "title": title,
            "declared_deliverable": deliverable.as_posix(),
            "milestone": mod.SOURCE_MILESTONE,
            "present": present,
            "terminal_class": terminal,
            "underlying_terminal_class": underlying,
            "terminal_evidence_source": (
                "exact_declared_artifact" if present else "conductor_structured_gate_receipt"
            ),
            "sha256": "sha256:" + task_id[-64:].rjust(64, "0")[:64] if present else None,
            "conductor_receipt": {
                "present": True,
                "status": conductor_status,
                "timestamp": "2026-08-06 20:08 UTC",
                "line": f"| fixture | {title[:20]} | {conductor_status} | fixture |",
            },
        }
    return {
        "experiment_id": "exp6168-v534-capstone-reconciliation",
        "milestone": mod.SOURCE_MILESTONE,
        "status": "complete_with_blocks_missing_skips_and_quarantine",
        "honest_verdict": (
            "complete: .534 reconciled with missing isolation artifact, conductor skips, "
            "mandatory CSL internal block, flagged decision artifacts, software-only "
            "stochastic block, and ARC no-solve positive preserved"
        ),
        "activated_task_and_declared_deliverable_matrix": matrix,
        "present_missing_skipped_internal_blocked_null_retired_flagged_and_positive_counts": {
            "missing": 1,
            "skipped": 2,
            "internal_blocked": 1,
            "null": 1,
            "retired": 0,
            "flagged": 2,
            "blocked": 1,
            "complete": 1,
            "positive": 3,
            "present": 9,
        },
        "research_complete_append_count": 0,
        "duplicate_history_amplification_count": 0,
    }


def _completion_payload(include_534_blocks: int = 1) -> JsonDict:
    canonical = {
        "id": mod.SOURCE_MILESTONE,
        "title": mod.SOURCE_MILESTONE_TITLE,
        "doc": mod.ROADMAP_DOC_RELATIVE_PATH.as_posix(),
        "completed": "2026-08-06",
        "finding": "See conductor log for per-experiment results.",
        "tasks": [
            {
                "id": task_id,
                "title": title,
                "deliverable": deliverable.as_posix(),
                "result": "OK (conductor)",
            }
            for task_id, title, deliverable in mod.SOURCE_TASKS
        ],
    }
    return {"milestones": [deepcopy(canonical) for _ in range(include_534_blocks)]}


def _v535_task(
    task_id: str,
    title: str,
    deliverable: Path,
    *,
    track: str,
    prompt: str | None = None,
    gated_on: list[JsonDict] | None = None,
    prior_failures: list[JsonDict] | None = None,
) -> JsonDict:
    task: JsonDict = {
        "id": task_id,
        "milestone": mod.TARGET_MILESTONE,
        "title": title,
        "priority": "high",
        "track": track,
        "deliverable": deliverable.as_posix(),
        "model": "gpt-5.5" if "transition" in task_id else "sonnet",
        "max_turns": 30,
        "estimated_wall_time_min": 120,
        "requires_gpu": False,
        "prompt": prompt
        or (
            "CONTEXT:\n"
            "Use local-GGUF models when this task invokes LLM work.\n"
            "REQUIRED ARTIFACT FIELDS: inference_substrate principle: fixture.\n"
            "Run command, 'Do NOT push. Do NOT modify scripts/research_conductor.py.'\n"
        ),
    }
    if gated_on is not None:
        task["gated_on"] = gated_on
    if prior_failures is not None:
        task["prior_failures"] = prior_failures
    return task


def _v535_payload(tasks: list[JsonDict]) -> JsonDict:
    return {
        "milestone": mod.TARGET_MILESTONE,
        "milestone_title": mod.TARGET_MILESTONE_TITLE,
        "milestone_doc": mod.ROADMAP_DOC_RELATIVE_PATH.as_posix(),
        "tasks": tasks,
    }


def _full_v535_tasks() -> list[JsonDict]:
    tasks = [
        _v535_task(
            task_id,
            title,
            deliverable,
            track=mod.EXPECTED_TRACKS[task_id],
            prior_failures=[
                {
                    "experiment_id": "exp0000-fixture",
                    "verdict": "complete_null: fixture",
                    "addressed_by": "fixture addresses the previous outcome",
                    "retire_if_same_verdict": True,
                }
            ]
            if task_id in {"exp6169-v535-transition", "exp6171-v535-source-delta-ingestion"}
            else None,
            gated_on=[
                {
                    "upstream": "exp6173-cctu-item-bank-preregistration",
                    "artifact_field": "cctu_item_bank_ready_score",
                    "op": "==",
                    "value": 1,
                }
            ]
            if task_id == "exp6174-cctu-authentic-k8-pool"
            else None,
        )
        for task_id, title, deliverable in mod.EXPECTED_NEXT_TASKS
    ]
    return tasks


def _make_root(root: Path, *, full_active: bool, include_next: bool, include_534_blocks: int) -> None:
    capstone = _source_capstone_payload()
    _write_json(root, mod.SOURCE_CAPSTONE_RELATIVE_PATH, capstone)
    _write_json(
        root,
        mod.SOURCE_TRANSITION_RELATIVE_PATH,
        {
            "status": "complete_with_terminal_receipts",
            "honest_verdict": "complete: .533 archived; .534 activation mode=already_active",
        },
    )
    for task_id, _title, deliverable in mod.SOURCE_TASKS_WITHOUT_CAPSTONE:
        row = capstone["activated_task_and_declared_deliverable_matrix"][task_id]
        if row["present"]:
            payload = {
                "status": row["terminal_class"],
                "honest_verdict": row["terminal_class"] + ": fixture",
            }
            if row["terminal_class"] == "flagged":
                payload["flagged_adversarial"] = True
                payload["corrigendum_pending"] = [{"kind": "DURATION_TOO_SHORT"}]
            _write_json(root, deliverable, payload)
    _write_text(
        root,
        mod.RESEARCH_COMPLETE_RELATIVE_PATH,
        yaml.safe_dump(_completion_payload(include_534_blocks=include_534_blocks), sort_keys=False),
    )
    full_tasks = _full_v535_tasks()
    active_tasks = full_tasks if full_active else full_tasks[:7]
    _write_text(root, mod.ROADMAP_RELATIVE_PATH, yaml.safe_dump(_v535_payload(active_tasks)))
    if include_next:
        _write_text(root, mod.ROADMAP_NEXT_RELATIVE_PATH, yaml.safe_dump(_v535_payload(full_tasks)))
    _write_text(
        root,
        mod.ROADMAP_DOC_RELATIVE_PATH,
        "# Research Roadmap vNEXT - Milestone 2026.08.535\n"
        "**Experiment range:** Exp6169-Exp6182 (14 tasks, four phases)\n"
        "Exp6182 capstone\n",
    )
    _write_text(
        root,
        mod.CONDUCTOR_LOG_RELATIVE_PATH,
        "| 2026-08-06 21:35 UTC | Plan milestone 2026.08.535 | OK | 14 tasks proposed |\n"
        "| 2026-08-06 21:38 UTC | Milestone 2026.08.535 activated | OK | 14 tasks queued |\n"
        "| 2026-08-06 21:12 UTC | Branch-independent .534 capstone | OK | fixture |\n",
    )
    for rel_path in [
        mod.AGENTS_RELATIVE_PATH,
        mod.CODEX_RELATIVE_PATH,
        mod.CLAUDE_RELATIVE_PATH,
        mod.RESEARCH_PROGRAM_RELATIVE_PATH,
        mod.EXCLUSION_MANIFEST_RELATIVE_PATH,
        mod.STATUS_RELATIVE_PATH,
        mod.CHANGELOG_RELATIVE_PATH,
        mod.E2E_PLAN_RELATIVE_PATH,
        mod.CONDUCTOR_RELATIVE_PATH,
        mod.DETERMINATION_LINT_RELATIVE_PATH,
        mod.SPEC_RELATIVE_PATH,
    ]:
        _write_text(root, rel_path, f"{rel_path.as_posix()} fixture\nREQ-REPORT-6169\n")


def test_req_report_6169_spec_declares_transition_contract() -> None:
    """REQ-REPORT-6169: OpenSpec names the V535 transition contract."""
    text = SPEC_PATH.read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-6169") :]
    section = section[: section.index("## Implementation Status (REQ-REPORT-6169)")]
    for required in [
        "Exp6169 through Exp6182",
        "append the canonical `.534` block",
        "research-roadmap-next.yaml",
        "scripts/research_conductor.py",
        "deterministic_repository_transition",
        "SCENARIO-REPORT-6169-EXACT-TERMINAL",
        "SCENARIO-REPORT-6169-APPEND-ONCE",
        "SCENARIO-REPORT-6169-ROADMAP-VALIDATION",
        "SCENARIO-REPORT-6169-PARTIAL-ACTIVATION-BLOCKS",
        "SCENARIO-REPORT-6169-SCHEMA",
    ]:
        assert required in section


def test_scenario_report_6169_full_staged_activation_appends_once(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6169-APPEND-ONCE: staged activation and history are idempotent."""
    _make_root(tmp_path, full_active=False, include_next=True, include_534_blocks=0)
    staged_hash = mod.path_sha256(tmp_path / mod.ROADMAP_NEXT_RELATIVE_PATH)

    report = mod.build_report(
        tmp_path,
        apply_mutations=True,
        test_exit_codes={"focused": 0},
        duration_s=0.125,
    )

    assert report["status"] == "complete_with_v535_activation"
    assert report["research_complete_append_count"] == 1
    assert report["research_complete_count_before_after"]["after_count"] == 1
    assert report["staged_and_activated_roadmap_hashes"]["staged_before_sha256"] == staged_hash
    assert report["staged_and_activated_roadmap_hashes"]["active_after_sha256"] == staged_hash
    assert mod.path_sha256(tmp_path / mod.ROADMAP_RELATIVE_PATH) == staged_hash
    assert report["activation_mode"] == "staged_atomic_copy"


def test_scenario_report_6169_partial_active_roadmap_blocks_without_mutation(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6169-PARTIAL-ACTIVATION-BLOCKS: seven-task active roadmaps fail closed."""
    _make_root(tmp_path, full_active=False, include_next=False, include_534_blocks=1)
    active_before = mod.path_sha256(tmp_path / mod.ROADMAP_RELATIVE_PATH)
    history_before = mod.path_sha256(tmp_path / mod.RESEARCH_COMPLETE_RELATIVE_PATH)

    report = mod.build_report(
        tmp_path,
        apply_mutations=True,
        test_exit_codes={"focused": 0},
        duration_s=0.25,
    )

    assert report["status"] == "blocked_partial_v535_activation"
    assert report["activated_task_count"] == 7
    assert report["research_complete_append_count"] == 0
    assert report["research_complete_count_before_after"]["after_count"] == 1
    assert report["activation_mode"] == "already_active_partial_mismatch"
    assert report["rollback_receipt"]["performed"] is False
    assert report["rollback_receipt"]["reason"] == "no_mutation_before_block"
    assert mod.path_sha256(tmp_path / mod.ROADMAP_RELATIVE_PATH) == active_before
    assert mod.path_sha256(tmp_path / mod.RESEARCH_COMPLETE_RELATIVE_PATH) == history_before
    assert report["protected_files_unchanged"]["unchanged"] is True
    assert "exp6176-hidden-state-surface-qualification" in report[
        "optional_field_and_prior_failure_validation"
    ]["missing_expected_task_ids"]


def test_scenario_report_6169_validation_catches_collisions_and_gate_errors() -> None:
    """SCENARIO-REPORT-6169-ROADMAP-VALIDATION: collisions and invalid gates are explicit."""
    tasks = _full_v535_tasks()
    tasks[1]["deliverable"] = tasks[0]["deliverable"]
    tasks[3]["gated_on"] = [
        {
            "upstream": "exp6173-cctu-item-bank-preregistration",
            "artifact_field": "cctu_item_bank_ready_score",
            "op": "approximately",
            "value": 1,
        }
    ]
    validation = mod.validate_v535_roadmap(_v535_payload(tasks))

    assert validation["task_count"] == 14
    assert validation["task_id_unique"] is True
    assert validation["deliverable_unique"] is False
    assert validation["duplicate_deliverables"] == [tasks[0]["deliverable"]]
    assert validation["gate_reference_validation"]["supported_operators"] is False
    assert validation["ready"] is False


def test_scenario_report_6169_terminal_matrix_and_required_schema(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6169-SCHEMA: terminal classes, principles, and checksum are stable."""
    _make_root(tmp_path, full_active=True, include_next=False, include_534_blocks=1)

    report = mod.build_report(
        tmp_path,
        apply_mutations=False,
        test_exit_codes={"focused": 0},
        duration_s=0.5,
    )

    assert report["status"] == "complete_with_v535_activation"
    assert report["source_capstone_hash_and_honest_verdict"]["present"] is True
    assert report["source_exact_terminal_classification"]["exp6157-repo-wide-artifact-isolation-closure"][
        "terminal_class"
    ] == "missing"
    assert report["source_exact_terminal_classification"]["exp6164-continuous-strategy-learning-ab"][
        "terminal_class"
    ] == "internal_blocked"
    assert report["source_exact_terminal_classification"]["exp6168-v534-capstone-reconciliation"][
        "terminal_class"
    ] == "complete"
    assert report["quarantine_and_determination_before_after_matrix"]["byte_preserved"] is True
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in report
        assert field in report["field_provenance"]
        assert report["field_provenance"][field]["principle"]
    assert report["inference_substrate"] == "deterministic_repository_transition"
    assert report["honest_verdict"].startswith("complete:")
    assert report["reproducibility_checksum"] == mod.payload_checksum(report)


def test_req_report_6169_defensive_validation_branches(tmp_path: Path) -> None:
    """REQ-REPORT-6169: malformed local receipts fail closed with explicit reasons."""
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    payload, meta = mod._read_json(bad_json)
    assert payload == {}
    assert meta["error"].startswith("json_error:")
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    assert mod._read_json(list_json)[1]["error"] == "json_not_mapping"

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("x: [", encoding="utf-8")
    assert mod._read_yaml(bad_yaml)[1]["error"].startswith("yaml_error:")
    _write_text(tmp_path, mod.RESEARCH_COMPLETE_RELATIVE_PATH, "milestones: nope\n")
    assert mod._history_blocks(tmp_path)[0] == []
    assert mod._task_signature({"tasks": "not-list"}) == ()

    malformed_tasks = [
        {"id": "exp6169-v535-transition", "prior_failures": None, "gated_on": None},
        {"id": "exp6170-v535-task-artifact-isolation-canary", "prior_failures": "bad"},
        {"id": "exp6171-v535-source-delta-ingestion", "prior_failures": [{}]},
        {"id": "exp6172-current-rule-quarantine-determination", "gated_on": "bad"},
        {"id": "exp6173-cctu-item-bank-preregistration", "gated_on": ["bad"]},
        {
            "id": "exp6174-cctu-authentic-k8-pool",
            "gated_on": [{"upstream": "missing", "artifact_field": "x", "op": "==", "value": 1}],
        },
    ]
    prior = mod._valid_prior_failures(malformed_tasks)
    gates = mod._valid_gates(malformed_tasks)
    assert prior["valid"] is False
    assert len(prior["invalid_rows"]) == 2
    assert gates["valid"] is False
    assert {row["reason"] for row in gates["invalid_gates"]} == {
        "gated_on_not_list",
        "gate_not_mapping",
        "unknown_upstream",
    }

    no_gguf = _full_v535_tasks()
    for task in no_gguf:
        if task["id"] == "exp6177-clue-latent-selector-freeze":
            task["prompt"] = "missing model declaration" + mod.PROMPT_ENDING
    model_receipt = mod._model_spec_receipt(no_gguf)
    assert model_receipt["valid"] is False
    assert model_receipt["missing_local_gguf_declarations"] == [
        "exp6177-clue-latent-selector-freeze"
    ]

    assert mod._capstone_conductor_receipt(tmp_path) == {"present": False}
    empty_validation = mod.validate_v535_roadmap({})
    activation = mod._activate_if_ready(
        tmp_path,
        {"present": False},
        empty_validation,
        empty_validation,
        apply_mutations=False,
    )
    assert activation["mode"] == "blocked_missing_or_mismatched_roadmap"


def test_req_report_6169_writer_receipts_and_payload_validator(tmp_path: Path) -> None:
    """REQ-REPORT-6169: writer and payload validation cover checksum and schema errors."""
    _make_root(tmp_path, full_active=True, include_next=False, include_534_blocks=1)
    receipts = tmp_path / "receipts.json"
    receipts.write_text(json.dumps({"focused": "0"}), encoding="utf-8")
    assert mod._load_test_receipts(None) == {}
    assert mod._load_test_receipts(receipts) == {"focused": 0}

    report = mod.write_report(tmp_path, test_exit_codes={"focused": 0})
    written = tmp_path / mod.RESULT_RELATIVE_PATH
    assert written.exists()
    assert json.loads(written.read_text(encoding="utf-8"))["experiment_id"] == mod.EXPERIMENT_ID
    assert mod._validate_required_payload(report) == []

    broken = dict(report)
    broken.pop("status")
    broken["field_provenance"] = []
    broken["inference_substrate"] = "wrong"
    broken["honest_verdict"] = "maybe"
    broken["reproducibility_checksum"] = "sha256:bad"
    errors = mod._validate_required_payload(broken)
    assert "status" in errors
    assert "field_provenance_not_mapping" in errors
    assert "inference_substrate" in errors
    assert "honest_verdict_prefix" in errors
    assert "reproducibility_checksum" in errors

    missing_principle = dict(report)
    missing_principle["field_provenance"] = dict(report["field_provenance"])
    missing_principle["field_provenance"]["status"] = {}
    missing_principle["reproducibility_checksum"] = mod.payload_checksum(missing_principle)
    assert "missing_principle:status" in mod._validate_required_payload(missing_principle)


def test_req_report_6169_default_commands_cover_required_check_categories() -> None:
    """REQ-REPORT-6169: command receipts name every required transition check class."""
    commands = "\n".join(mod.DEFAULT_TEST_COMMANDS)
    for required in [
        "test_experiment_6169_transition_v535.py",
        "coverage report",
        "validate_prior_failures.py research-roadmap.yaml",
        "audit_roadmap_gates.py research-roadmap.yaml",
        "check_exclusion_manifest.py",
        "transition_integrity.py absent",
        "research-complete .534 duplicate-history OK",
        "determination_preservation_lint.py HEAD",
        "scripts/research_conductor.py protected",
        "E2E plan inspected",
        "root_clutter_sweep.py",
        "pytest tests/python -q",
    ]:
        assert required in commands
