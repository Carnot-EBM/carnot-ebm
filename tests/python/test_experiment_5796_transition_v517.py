"""Tests for the Exp5796 V517 transition receipt.

Spec refs: REQ-REPORT-5796, SCENARIO-REPORT-5796,
SCENARIO-REPORT-5796-COLLISION-BLOCK,
SCENARIO-REPORT-5796-IDENTITY-BLOCK,
SCENARIO-REPORT-5796-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_5796_transition_v517 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"


def _write_json(root: Path, rel_path: Path | str, payload: Any) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(root: Path, rel_path: Path | str, text: str = "context\n") -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _artifact_payload(task_id: str) -> JsonDict:
    payloads: dict[str, JsonDict] = {
        "exp5782-transition-v516": {
            "status": "complete",
            "honest_verdict": "complete: transition ready",
            "next_range_collision_count": 0,
        },
        "exp5783-v516-source-delta-ingestion": {
            "status": "complete",
            "honest_verdict": "complete: no accepted post-V516 source deltas",
            "accepted_finding_count": 0,
        },
        "exp5784-evidence-index-terminal-qualification": {
            "status": "complete",
            "honest_verdict": "complete: exact index qualified",
            "evidence_index_ready_score": 1.0,
        },
        "exp5785-hardness-surface-prospective-fixture": {
            "status": "complete",
            "honest_verdict": "complete: sealed_hardness_surface_exact_fixture_ready",
            "fixture_ready_score": 1.0,
        },
        "exp5786-sota-hardness-controlled-constraint-stream": {
            "status": "complete",
            "honest_verdict": (
                "complete: sota_constraint_response_stream_collected_not_ready:"
                "parser_failure_threshold"
            ),
            "stream_ready_score": 0.0,
            "qwen_truncation_count": 360,
        },
        "exp5787-validation-gated-constraint-skill-ab": {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
        },
        "exp5789-constraint-skill-shadow-adapter": {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
        },
        "exp5790-arc-world-model-admission-contract": {
            "status": "complete",
            "honest_verdict": "complete: immutable_world_model_admission_contract_ready_no_solve_credit",
            "solve_claimed": False,
        },
        "exp5791-arc-sota-independent-hypothesis-panel": {
            "status": "blocked",
            "honest_verdict": "blocked: headline_gpu_offload_receipts_present",
            "panel_ready_score": 0.0,
        },
        "exp5793-arc-live-world-model-ab": {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
        },
        "exp5794-hardware-terminal-action-receipt": {
            "status": "complete_cached_hardware_reconciliation_no_board_commands",
            "honest_verdict": "complete: cached hardware reconciliation no_speedup_claim",
        },
        "exp5795-v516-capstone-reconciliation": {
            "status": "complete",
            "honest_verdict": "complete: v516 reconciled",
            "retired_task_ids": ["exp5773-prior-retired", "exp5709-prior-retired"],
        },
    }
    return payloads[task_id]


def _research_complete_payload(*, duplicate_516_blocks: int = 2) -> JsonDict:
    v515_block = {
        "id": "2026.07.515",
        "title": "Prior duplicate history preserved",
        "tasks": [
            {
                "id": "exp5769-transition-v515",
                "deliverable": "results/experiment_5769_transition_v515.json",
                "result": "OK (conductor)",
            }
        ],
    }
    v516_block = {
        "id": mod.MILESTONE_FROM,
        "title": "Terminal V516",
        "doc": "openspec/change-proposals/research-roadmap-vNEXT.md",
        "completed": "2026-07-22",
        "finding": "See conductor log for per-experiment results.",
        "tasks": [
            {
                "id": task_id,
                "title": f"title for {task_id}",
                "deliverable": rel_path.as_posix(),
                "result": "OK (conductor)",
            }
            for task_id, rel_path in mod.TASK_ARTIFACT_PATHS.items()
        ],
    }
    return {"milestones": [v515_block, *[v516_block for _ in range(duplicate_516_blocks)]]}


def _active_roadmap_payload() -> JsonDict:
    return {
        "milestone": mod.MILESTONE_TO,
        "tasks": [
            {
                "id": task_id,
                "milestone": mod.MILESTONE_TO,
                "deliverable": mod.ACTIVE_TASK_ARTIFACT_PATHS[task_id].as_posix(),
            }
            for task_id in mod.ACTIVE_TASK_IDS
        ],
    }


def _vnext_doc() -> str:
    lines = ["# Research Roadmap vNEXT", "", "**Milestone:** 2026.07.517", ""]
    lines.extend(
        f"`{mod.NEXT_TASK_ARTIFACT_PATHS[task_id].as_posix()}`"
        for task_id in mod.NEXT_TASK_IDS
    )
    return "\n".join(lines) + "\n"


def _conductor_log() -> str:
    return "\n".join(
        [
            "| 2026-07-22 07:35 UTC | Transition terminal .515 evidence and allocate col | OK | 89 passed |",
            "| 2026-07-22 08:33 UTC | Time-windowed literature freshness receipt | OK | 88 passed |",
            "| 2026-07-22 09:30 UTC | Qualify the existing exact-deliverable index with | OK | 93 passed |",
            "| 2026-07-22 10:15 UTC | Gated on Exp5784 readiness: build a sealed hardnes | OK | 87 passed |",
            "| 2026-07-22 11:06 UTC | Gated on Exp5785 fixture readiness: run the three- | OK | 85 passed |",
            "| 2026-07-22 11:08 UTC | Gated on Exp5786 clean drift headroom: run continu | GATE_BLOCK | 2 of 5 gate(s) failed |",
            "| 2026-07-22 11:10 UTC | Gated on Exp5786 clean drift headroom: run continu | GATE_BLOCK | 2 of 5 gate(s) failed |",
            "| 2026-07-22 11:12 UTC | Gated on Exp5786 clean drift headroom: run continu | GATE_BLOCK | 2 of 5 gate(s) failed |",
            "| 2026-07-22 11:14 UTC | Causal future-family holdout of versioned rule sta | GATE_BLOCK | upstream retired |",
            "| 2026-07-22 11:14 UTC | Gated on Exp5788 transfer: wire a disabled typed-c | GATE_BLOCK | upstream artifact not found |",
            "| 2026-07-22 11:16 UTC | Causal future-family holdout of versioned rule sta | GATE_BLOCK | upstream retired |",
            "| 2026-07-22 11:16 UTC | Gated on Exp5788 transfer: wire a disabled typed-c | GATE_BLOCK | upstream artifact not found |",
            "| 2026-07-22 11:18 UTC | Causal future-family holdout of versioned rule sta | GATE_BLOCK | upstream retired |",
            "| 2026-07-22 11:18 UTC | Gated on Exp5788 transfer: wire a disabled typed-c | GATE_BLOCK | upstream artifact not found |",
            "| 2026-07-22 11:36 UTC | Pivotal-dynamics accreditation contract for immuta | OK | 87 passed |",
            "| 2026-07-22 12:22 UTC | Gated on Exp5790 admission readiness: run a matche | FAIL | artifact_not_updated_past_bootstrap |",
            "| 2026-07-22 12:48 UTC | Gated on Exp5790 admission readiness: run a matche | FAIL | artifact_not_updated_past_bootstrap |",
            "| 2026-07-22 13:19 UTC | Gated on Exp5790 admission readiness: run a matche | FAIL | artifact_not_updated_past_bootstrap |",
            "| 2026-07-22 13:21 UTC | Frozen calibration chooser over immutable simulato | GATE_BLOCK | upstream retired |",
            "| 2026-07-22 13:21 UTC | Gated on Exp5792 selector benefit: measure selecte | GATE_BLOCK | upstream artifact not found |",
            "| 2026-07-22 13:23 UTC | Frozen calibration chooser over immutable simulato | GATE_BLOCK | upstream retired |",
            "| 2026-07-22 13:23 UTC | Gated on Exp5792 selector benefit: measure selecte | GATE_BLOCK | upstream artifact not found |",
            "| 2026-07-22 13:25 UTC | Frozen calibration chooser over immutable simulato | GATE_BLOCK | upstream retired |",
            "| 2026-07-22 13:25 UTC | Gated on Exp5792 selector benefit: measure selecte | GATE_BLOCK | upstream artifact not found |",
            "| 2026-07-22 13:51 UTC | Board-state hash ledger and operator handoff packe | OK | 91 passed |",
            "| 2026-07-22 14:35 UTC | Reconcile .516 evidence, phase telemetry, specs, o | OK | 86 passed |",
        ]
    )


def _make_root(root: Path, *, duplicate_516_blocks: int = 2) -> None:
    for task_id, rel_path in mod.TASK_ARTIFACT_PATHS.items():
        if task_id in mod.MISSING_TASK_IDS:
            continue
        _write_json(root, rel_path, _artifact_payload(task_id))

    _write_text(root, mod.ROADMAP_RELATIVE_PATH, yaml.safe_dump(_active_roadmap_payload()))
    _write_text(root, mod.VNEXT_RELATIVE_PATH, _vnext_doc())
    _write_text(
        root,
        mod.RESEARCH_COMPLETE_RELATIVE_PATH,
        yaml.safe_dump(_research_complete_payload(duplicate_516_blocks=duplicate_516_blocks)),
    )
    _write_text(root, mod.CONDUCTOR_LOG_RELATIVE_PATH, _conductor_log())
    _write_text(root, mod.EXCLUSION_MANIFEST_RELATIVE_PATH, "retired: []\n")
    _write_text(root, mod.CONDUCTOR_RELATIVE_PATH, "# conductor fixture\n")


def test_spec_contains_req_report_5796_contract() -> None:
    """REQ-REPORT-5796: OpenSpec names V516 identity and V517 collision gate."""

    text = SPEC_PATH.read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-5796") :]

    assert str(mod.RESULT_RELATIVE_PATH) in section
    assert "(milestone, task_id, declared_deliverable)" in section
    assert "SCENARIO-REPORT-5796-COLLISION-BLOCK" in section
    assert "complete-negative" in section
    assert "Exp5796-Exp5808" in section


def test_scenario_report_5796_archives_terminal_v516_by_exact_identity(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5796: declared paths and disjoint classes are canonical."""

    _make_root(tmp_path)
    report = mod.build_report(
        tmp_path,
        tests_run=[{"command": "unit", "exit_code": 0}],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert report["status"] == "complete"
    assert report["honest_verdict"].startswith("complete:")
    assert report["milestone_from"] == "2026.07.516"
    assert report["milestone_to"] == "2026.07.517"
    assert report["archived_task_ids"] == list(mod.EXPECTED_TASK_IDS)
    assert len(report["declared_deliverable_matrix"]) == 14
    assert report["positive_result_task_ids"] == list(mod.POSITIVE_RESULT_TASK_IDS)
    assert report["scientific_null_task_ids"] == ["exp5783-v516-source-delta-ingestion"]
    assert report["negative_result_task_ids"] == [
        "exp5786-sota-hardness-controlled-constraint-stream"
    ]
    assert report["blocked_precondition_task_ids"] == []
    assert report["blocked_gate_task_ids"] == [
        "exp5787-validation-gated-constraint-skill-ab",
        "exp5789-constraint-skill-shadow-adapter",
        "exp5793-arc-live-world-model-ab",
    ]
    assert report["failed_delivery_task_ids"] == [
        "exp5791-arc-sota-independent-hypothesis-panel"
    ]
    assert report["missing_task_ids"] == [
        "exp5788-constraint-skill-transfer-audit",
        "exp5792-arc-calibration-only-selector",
    ]
    assert report["no_solve_task_ids"] == [
        "exp5790-arc-world-model-admission-contract"
    ]
    outcome_sets = [
        set(report[field])
        for field in [
            "positive_result_task_ids",
            "scientific_null_task_ids",
            "negative_result_task_ids",
            "blocked_precondition_task_ids",
            "blocked_gate_task_ids",
            "failed_delivery_task_ids",
            "missing_task_ids",
        ]
    ]
    assert sum(len(items) for items in outcome_sets) == len(set().union(*outcome_sets))
    assert set(report["no_solve_task_ids"]).issubset(set(report["positive_result_task_ids"]))
    assert report["research_complete_append_count"] == 0
    assert report["duplicate_history_diagnostics"]["milestone_from_block_count"] == 2
    assert report["preconditions_checked"]["roadmaps"]["next"]["present"] is False
    assert report["preconditions_checked"]["resource_receipts"]["disk_free_bytes"] > 0
    assert report["canonical_artifact_hashes"]["exp5788-constraint-skill-transfer-audit"][
        "status"
    ] == "missing"
    exp5791 = report["conductor_outcomes"]["exp5791-arc-sota-independent-hypothesis-panel"]
    assert exp5791["delivery_failure_count"] == 3
    assert exp5791["artifact_status"] == "blocked-precondition"
    assert report["next_task_range"] == "exp5796-exp5808"
    assert report["next_range_collision_count"] == 0
    assert isinstance(report["next_range_collision_count"], int)
    assert report["research_roadmap_unchanged"] is True
    assert report["conductor_unchanged"] is True


def test_scenario_report_5796_alias_groups_never_replace_declared_paths(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5796: same-number files are diagnostics only."""

    _make_root(tmp_path)
    _write_json(
        tmp_path,
        "results/experiment_5788_proxy_alias.json",
        {"status": "complete", "honest_verdict": "complete: proxy alias"},
    )
    _write_json(
        tmp_path,
        "results/experiment_5791_older_attempt_alias.json",
        {"status": "complete", "honest_verdict": "complete: old alias"},
    )

    report = mod.build_report(
        tmp_path,
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    exp5788 = report["same_number_alias_groups"]["5788"]
    assert exp5788["canonical"]["path"] == (
        "results/experiment_5788_constraint_skill_transfer_audit.json"
    )
    assert exp5788["canonical"]["present"] is False
    assert [row["path"] for row in exp5788["aliases"]] == [
        "results/experiment_5788_proxy_alias.json"
    ]
    exp5791 = report["same_number_alias_groups"]["5791"]
    assert exp5791["canonical"]["path"] == (
        "results/experiment_5791_arc_sota_independent_hypothesis_panel.json"
    )
    assert [row["path"] for row in exp5791["aliases"]] == [
        "results/experiment_5791_older_attempt_alias.json"
    ]


def test_scenario_report_5796_collision_blocks_allocation(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5796-COLLISION-BLOCK: occupied V517 ids fail closed."""

    _make_root(tmp_path)
    _write_json(tmp_path, "results/experiment_5800_existing_collision.json", {"status": "old"})
    payload = _research_complete_payload()
    payload["milestones"].append(
        {
            "id": "2026.07.400",
            "title": "stale next-range reference",
            "finding": "mentions exp5804-arc-bootstrap-safe-sota-panel",
            "tasks": [],
        }
    )
    _write_text(tmp_path, mod.RESEARCH_COMPLETE_RELATIVE_PATH, yaml.safe_dump(payload))

    report = mod.build_report(
        tmp_path,
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert report["status"] == "blocked"
    assert report["honest_verdict"].startswith("blocked:")
    assert report["next_range_collision_count"] == 2
    assert [row["path"] for row in report["collision_scan"]["preexisting_collisions"]] == [
        "research-complete.yaml",
        "results/experiment_5800_existing_collision.json",
    ]


def test_scenario_report_5796_identity_failures_block(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5796-IDENTITY-BLOCK: ambiguous declared paths fail closed."""

    _make_root(tmp_path)
    (tmp_path / mod.TASK_ARTIFACT_PATHS["exp5784-evidence-index-terminal-qualification"]).unlink()
    missing_report = mod.build_report(
        tmp_path,
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    assert missing_report["status"] == "blocked"
    assert (
        "missing_or_malformed_declared_deliverables=['exp5784-evidence-index-terminal-qualification']"
        in missing_report["preconditions_checked"]["failed_preconditions"]
    )

    mismatch_root = tmp_path / "mismatch"
    _make_root(mismatch_root)
    payload = _research_complete_payload()
    payload["milestones"][-1]["tasks"][2]["deliverable"] = "results/wrong.json"
    _write_text(mismatch_root, mod.RESEARCH_COMPLETE_RELATIVE_PATH, yaml.safe_dump(payload))
    mismatch_report = mod.build_report(
        mismatch_root,
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    assert any(
        item.startswith("declared_deliverable_mismatch=")
        for item in mismatch_report["preconditions_checked"]["failed_preconditions"]
    )

    duplicate_root = tmp_path / "duplicate"
    _make_root(duplicate_root)
    duplicate_payload = _research_complete_payload()
    duplicate_task = dict(duplicate_payload["milestones"][-1]["tasks"][0])
    duplicate_task["deliverable"] = "results/conflicting.json"
    duplicate_payload["milestones"][-1]["tasks"].append(duplicate_task)
    _write_text(
        duplicate_root,
        mod.RESEARCH_COMPLETE_RELATIVE_PATH,
        yaml.safe_dump(duplicate_payload),
    )
    duplicate_report = mod.build_report(
        duplicate_root,
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    assert "duplicate_task_id_conflicts=" in ";".join(
        duplicate_report["preconditions_checked"]["failed_preconditions"]
    )

    ambiguous_root = tmp_path / "ambiguous"
    _make_root(ambiguous_root)
    ambiguous_payload = _research_complete_payload()
    altered = json.loads(json.dumps(ambiguous_payload["milestones"][-1]))
    altered["tasks"][0]["deliverable"] = "results/other.json"
    ambiguous_payload["milestones"].append(altered)
    _write_text(
        ambiguous_root,
        mod.RESEARCH_COMPLETE_RELATIVE_PATH,
        yaml.safe_dump(ambiguous_payload),
    )
    ambiguous = mod.build_report(
        ambiguous_root,
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    assert (
        "ambiguous_research_complete_declared_task_blocks"
        in ambiguous["preconditions_checked"]["failed_preconditions"]
    )


def test_scenario_report_5796_emit_report_and_field_principles(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5796-FIELD-PRINCIPLES: emitted artifact is stable."""

    _make_root(tmp_path)
    history_before = (tmp_path / mod.RESEARCH_COMPLETE_RELATIVE_PATH).read_bytes()
    output = tmp_path / mod.RESULT_RELATIVE_PATH
    report = mod.emit_report(
        tmp_path,
        output_path=output,
        tests_run=[{"command": "unit", "exit_code": 0}],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert (tmp_path / mod.RESEARCH_COMPLETE_RELATIVE_PATH).read_bytes() == history_before
    assert written["reproducibility_checksum"] == report["reproducibility_checksum"]
    assert mod.payload_checksum(written) == written["reproducibility_checksum"]
    assert set(report).issubset(report["field_principles"])
    assert all(report["field_principles"][field] for field in report)
    assert mod._load_tests_run(None)[0]["status"] == "not_run"

    original = mod.FIELD_PRINCIPLES.pop("status")
    try:
        with pytest.raises(KeyError, match="missing field principles"):
            mod.build_report(
                tmp_path,
                modification_overrides={
                    mod.ROADMAP_RELATIVE_PATH: False,
                    mod.CONDUCTOR_RELATIVE_PATH: False,
                },
            )
    finally:
        mod.FIELD_PRINCIPLES["status"] = original


def test_scenario_report_5796_defensive_precondition_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-REPORT-5796-IDENTITY-BLOCK: defensive checks are explicit."""

    assert mod._task_signature({"tasks": "not-list"}) == ()
    assert mod._payload_status({}, {"exists": True, "loadable": False}, "task") == "malformed"
    assert (
        mod._payload_status({"status": "blocked"}, {"exists": True, "loadable": True}, "task")
        == "blocked"
    )
    assert mod._payload_status({}, {"exists": True, "loadable": True}, "task") == "unknown"
    assert mod._parse_conductor_log(tmp_path / "missing-log-file") == []
    assert mod._collision_scan(tmp_path / "empty-scan")["preexisting_collision_count"] == 0
    assert mod._outcome_class("exp9999-unclassified") == "unclassified"

    malformed_yaml = tmp_path / "bad.yaml"
    malformed_yaml.write_text("not: [closed\n", encoding="utf-8")
    _payload, malformed_meta = mod._read_yaml_with_meta(malformed_yaml)
    assert malformed_meta["parsed"] is False
    assert malformed_meta["error"]
    list_yaml = tmp_path / "list.yaml"
    list_yaml.write_text("- item\n", encoding="utf-8")
    _payload, list_meta = mod._read_yaml_with_meta(list_yaml)
    assert list_meta["parsed"] is False
    assert list_meta["error"] == "expected mapping, got list"

    no_history_root = tmp_path / "no-history"
    _make_root(no_history_root)
    _write_text(no_history_root, mod.RESEARCH_COMPLETE_RELATIVE_PATH, "milestones: nope\n")
    no_history = mod.build_report(
        no_history_root,
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    assert no_history["research_complete_append_count"] == 1

    missing_capstone_root = tmp_path / "missing-capstone"
    _make_root(missing_capstone_root)
    (missing_capstone_root / mod.TASK_ARTIFACT_PATHS["exp5795-v516-capstone-reconciliation"]).unlink()
    assert mod._retired_task_ids(missing_capstone_root) == []

    short_log_root = tmp_path / "short-log"
    short_log_root.mkdir()
    _write_text(short_log_root, mod.CONDUCTOR_LOG_RELATIVE_PATH, "| too | short |\n")
    assert mod._parse_conductor_log(short_log_root) == []

    wrong_roadmap_root = tmp_path / "wrong-roadmap"
    _make_root(wrong_roadmap_root)
    _write_text(
        wrong_roadmap_root,
        mod.ROADMAP_RELATIVE_PATH,
        yaml.safe_dump({"milestone": "2026.07.516", "tasks": []}),
    )
    wrong_roadmap = mod.build_report(
        wrong_roadmap_root,
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: True,
            mod.CONDUCTOR_RELATIVE_PATH: True,
        },
    )
    failures = wrong_roadmap["preconditions_checked"]["failed_preconditions"]
    assert "active_roadmap_milestone='2026.07.516'" in failures
    assert "research_roadmap_modified" in failures
    assert "research_conductor_modified" in failures

    bad_active_root = tmp_path / "bad-active"
    _make_root(bad_active_root)
    _write_text(bad_active_root, mod.ROADMAP_RELATIVE_PATH, "bad: [yaml\n")
    bad_active = mod.build_report(
        bad_active_root,
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    assert (
        "active_roadmap_unparseable"
        in bad_active["preconditions_checked"]["failed_preconditions"]
    )

    bad_task_root = tmp_path / "bad-task"
    _make_root(bad_task_root)
    _write_text(
        bad_task_root,
        mod.ROADMAP_RELATIVE_PATH,
        yaml.safe_dump(
            {
                "milestone": mod.MILESTONE_TO,
                "tasks": [{"id": "exp9999-not-allocated", "deliverable": "results/x.json"}],
            }
        ),
    )
    bad_task = mod.build_report(
        bad_task_root,
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    assert any(
        item.startswith("active_roadmap_task_ids=")
        for item in bad_task["preconditions_checked"]["failed_preconditions"]
    )

    bad_next_root = tmp_path / "bad-next"
    _make_root(bad_next_root)
    _write_text(bad_next_root, mod.ROADMAP_NEXT_RELATIVE_PATH, "bad: [yaml\n")
    bad_next = mod.build_report(
        bad_next_root,
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    assert "next_roadmap_unparseable" in bad_next["preconditions_checked"]["failed_preconditions"]

    missing_log_root = tmp_path / "missing-log"
    _make_root(missing_log_root)
    _write_text(missing_log_root, mod.CONDUCTOR_LOG_RELATIVE_PATH, "no rows\n")
    missing_log = mod.build_report(
        missing_log_root,
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    assert any(
        item.startswith("missing_conductor_outcomes=")
        for item in missing_log["preconditions_checked"]["failed_preconditions"]
    )

    monkeypatch.setattr(mod, "EXPECTED_TASK_IDS", ("not-an-exp-task",))
    assert mod._same_number_alias_groups(tmp_path, {}) == {}
