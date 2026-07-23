"""Tests for the Exp5823 V519 transition receipt.

Spec refs: REQ-REPORT-5823, SCENARIO-REPORT-5823,
SCENARIO-REPORT-5823-RETIREMENT,
SCENARIO-REPORT-5823-COLLISION-BLOCK,
SCENARIO-REPORT-5823-FIELD-PROVENANCE.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_5823_transition_v519 as mod


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
        "exp5809-transition-v518": {
            "status": "complete",
            "honest_verdict": "complete: transition ready",
            "next_task_range": "exp5809-exp5822",
            "next_range_collision_count": 0,
        },
        "exp5810-v518-source-delta-ingestion": {
            "status": "complete",
            "honest_verdict": "complete: no accepted post-V518 source deltas",
            "accepted_finding_count": 0,
        },
        "exp5811-exp5799-event-provenance-audit": {
            "status": "complete",
            "honest_verdict": "complete: evidence audit ready",
            "canary_evidence_ready_score": 1.0,
        },
        "exp5812-split-budget-channel-contract": {
            "status": "complete",
            "honest_verdict": "complete: split budget contract ready",
            "split_budget_contract_ready_score": 1.0,
        },
        "exp5813-split-budget-sota-canary": {
            "status": "complete",
            "honest_verdict": "complete: answer_channel_not_ready_split_budget_sota_canary",
            "answer_channel_ready_score": 0.0,
            "qualified_real_sota_model_count": 0,
            "qualified_sota_family_count": 0,
            "exact_label_count": 2,
            "total_row_count": 144,
            "parser_failure_count": 138,
            "truncation_count": 134,
        },
        "exp5814-channel-qualified-constraint-stream": {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gate_check_summary": {"failed_gate_count": 2},
        },
    }
    return payloads[task_id]


def _research_complete_payload(*, duplicate_518_blocks: int = 1) -> JsonDict:
    v518_block = {
        "id": mod.MILESTONE_FROM,
        "title": "Terminal V518",
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
    return {"milestones": [v518_block for _ in range(duplicate_518_blocks)]}


def _active_roadmap_payload() -> JsonDict:
    return {
        "milestone": mod.MILESTONE_TO,
        "tasks": [
            {
                "id": task_id,
                "milestone": mod.MILESTONE_TO,
                "deliverable": mod.NEXT_TASK_ARTIFACT_PATHS[task_id].as_posix(),
            }
            for task_id in mod.ACTIVE_TASK_IDS
        ],
    }


def _vnext_doc() -> str:
    lines = [
        "# Research Roadmap vNEXT",
        "",
        "**Milestone:** 2026.07.519",
        "**Task range:** Exp5823-Exp5836",
        "",
    ]
    lines.extend(
        f"`{mod.NEXT_TASK_ARTIFACT_PATHS[task_id].as_posix()}`"
        for task_id in mod.NEXT_TASK_IDS
    )
    lines.extend(f"proposal-only `{task_id}`" for task_id in mod.RESERVED_UNACTIVATED_TASK_IDS)
    return "\n".join(lines) + "\n"


def _conductor_log() -> str:
    return "\n".join(
        [
            "| 2026-07-22 19:38 UTC | Transition four activated .517 tasks, quarantine f | OK | 118 passed |",
            "| 2026-07-22 20:14 UTC | Time-windowed post-V518 literature and implementat | OK | 88 passed |",
            "| 2026-07-22 20:35 UTC | Companion audit for Exp5799 event definitions, row | OK | 97 passed |",
            "| 2026-07-22 21:11 UTC | Gated on Exp5811 clean evidence: implement split-b | OK | 87 passed |",
            "| 2026-07-22 21:35 UTC | Gated on Exp5812 contract: changed-mechanism canar | FAIL | timeout |",
            "| 2026-07-22 21:37 UTC | Gated on Exp5812 contract: changed-mechanism canar | SKIP | pre-tests failing |",
            "| 2026-07-22 22:10 UTC | Gated on Exp5812 contract: changed-mechanism canar | OK | 81 passed |",
            "| 2026-07-22 22:12 UTC | Gated on Exp5813 all-family qualification: prospec | GATE_BLOCK | answer_channel_ready_score=0.0 |",
            "| 2026-07-22 22:14 UTC | Gated on Exp5813 all-family qualification: prospec | GATE_BLOCK | answer_channel_ready_score=0.0 |",
            "| 2026-07-22 22:16 UTC | Gated on Exp5813 all-family qualification: prospec | GATE_BLOCK | answer_channel_ready_score=0.0 |",
            "| 2026-07-22 22:18 UTC | Gated on Exp5814 clean stream: future-validated ty | GATE_BLOCK | Pre-emptive skip: upstream retired |",
        ]
    )


def _manifest_payload(include_retirement: bool = True) -> JsonDict:
    payload: JsonDict = {"retired": [], "retired_experiments": []}
    if include_retirement:
        payload["retired_experiments"].append(mod.EXPECTED_RETIREMENT_ENTRY)
    return payload


def _make_root(root: Path, *, include_retirement: bool = True, duplicate_518_blocks: int = 2) -> None:
    for task_id, rel_path in mod.TASK_ARTIFACT_PATHS.items():
        if task_id == "exp5815-future-validated-constraint-skill-ab":
            continue
        _write_json(root, rel_path, _artifact_payload(task_id))
    _write_text(root, mod.EXP5813_ROW_RELATIVE_PATH, '{"row_sequence_index":0}\n')
    _write_text(root, mod.ROADMAP_RELATIVE_PATH, yaml.safe_dump(_active_roadmap_payload()))
    _write_text(
        root,
        mod.RESEARCH_COMPLETE_RELATIVE_PATH,
        yaml.safe_dump(_research_complete_payload(duplicate_518_blocks=duplicate_518_blocks)),
    )
    _write_text(root, mod.VNEXT_RELATIVE_PATH, _vnext_doc())
    _write_text(root, mod.CONDUCTOR_LOG_RELATIVE_PATH, _conductor_log())
    _write_text(root, mod.EXCLUSION_MANIFEST_RELATIVE_PATH, yaml.safe_dump(_manifest_payload(include_retirement)))
    _write_text(root, mod.CONDUCTOR_RELATIVE_PATH, "# conductor fixture\n")


def _clean_build(root: Path) -> JsonDict:
    return mod.build_report(
        root,
        tests_run=[{"command": "unit", "exit_code": 0}],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
        duration_s=1.25,
    )


def test_spec_contains_req_report_5823_contract() -> None:
    """REQ-REPORT-5823: OpenSpec names exact identity, retirement, and range gates."""

    text = SPEC_PATH.read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-5823") :]

    assert str(mod.RESULT_RELATIVE_PATH) in section
    assert "(milestone, task_id, declared_deliverable)" in section
    assert "SCENARIO-REPORT-5823-RETIREMENT" in section
    assert "Exp5816 through Exp5822" in section
    assert "Exp5823 through Exp5836" in section
    for field in mod.REQUIRED_PRINCIPLE_FIELDS:
        assert f"`{field}`" in section


def test_scenario_report_5823_archives_terminal_v518_by_exact_identity(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5823: declared V518 paths and disjoint classes are canonical."""

    _make_root(tmp_path)
    report = _clean_build(tmp_path)

    assert report["status"] == "complete"
    assert report["honest_verdict"].startswith("complete:")
    assert report["milestone_transition"] == {
        "source_milestone": "2026.07.518",
        "destination_milestone": "2026.07.519",
        "task_identity_tuple": ["milestone", "task_id", "declared_deliverable"],
    }
    assert len(report["declared_deliverable_matrix"]) == 7
    assert report["outcome_classification"]["clean_positive_task_ids"] == [
        "exp5809-transition-v518",
        "exp5811-exp5799-event-provenance-audit",
        "exp5812-split-budget-channel-contract",
    ]
    assert report["outcome_classification"]["clean_null_task_ids"] == [
        "exp5810-v518-source-delta-ingestion"
    ]
    assert report["outcome_classification"]["clean_negative_task_ids"] == [
        "exp5813-split-budget-sota-canary"
    ]
    assert report["outcome_classification"]["gate_blocked_task_ids"] == [
        "exp5814-channel-qualified-constraint-stream"
    ]
    assert report["outcome_classification"]["preemptively_skipped_task_ids"] == [
        "exp5815-future-validated-constraint-skill-ab"
    ]
    assert report["outcome_classification"]["flagged_task_ids"] == []
    assert report["outcome_classification"]["missing_task_ids"] == []
    assert report["outcome_classification"]["proposal_only_task_ids"] == list(
        mod.RESERVED_UNACTIVATED_TASK_IDS
    )
    assert report["outcome_classification"]["clean_success_task_ids"] == [
        "exp5809-transition-v518",
        "exp5811-exp5799-event-provenance-audit",
        "exp5812-split-budget-channel-contract",
    ]

    exp5813 = next(
        row
        for row in report["declared_deliverable_matrix"]
        if row["task_id"] == "exp5813-split-budget-sota-canary"
    )
    assert exp5813["outcome_class"] == "clean-negative"
    assert exp5813["answer_channel_ready_score"] == 0.0
    assert exp5813["exact_label_count"] == 2
    assert exp5813["parser_failure_count"] == 138
    assert exp5813["truncation_count"] == 134
    assert exp5813["generated_answer_success"] is False

    exp5815 = report["declared_deliverable_matrix"][-1]
    assert exp5815["task_id"] == "exp5815-future-validated-constraint-skill-ab"
    assert exp5815["canonical_artifact_present"] is False
    assert exp5815["conductor_outcome"] == "GATE_BLOCK"
    assert exp5815["outcome_class"] == "preemptively-skipped"

    assert report["reserved_unactivated_task_ids"] == list(mod.RESERVED_UNACTIVATED_TASK_IDS)
    assert report["research_complete_append_count"] == 0
    assert report["duplicate_history_diagnostics"]["duplicate_history_block_count"] == 1
    assert report["preconditions_checked"]["roadmaps"]["next"]["present"] is False
    assert report["preconditions_checked"]["input_hashes"]["row_files"][
        mod.EXP5813_ROW_RELATIVE_PATH.as_posix()
    ]["present"] is True
    assert report["preconditions_checked"]["resource_receipts"]["disk_free_bytes"] > 0
    assert report["next_task_range"] == "exp5823-exp5836"
    assert report["next_range_collision_count"] == 0
    assert report["docs_reconciled"]["operator_owned_docs_deferred"] is True
    assert report["research_roadmap_unchanged"] is True
    assert report["conductor_unchanged"] is True


def test_scenario_report_5823_retirement_manifest_is_required(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5823-RETIREMENT: generated-answer retry scope is retired."""

    _make_root(tmp_path)
    report = _clean_build(tmp_path)
    retirement = report["answer_transport_retirement"]

    assert retirement["manifest_entry_present"] is True
    assert retirement["scope_key"] == mod.RETIREMENT_SCOPE_KEY
    assert retirement["retire_if_same_verdict"] is True
    assert "finite-id generated-answer retry" in retirement["blocked_same_mechanism_retries"]
    assert "split-budget generated-answer retry" in retirement["blocked_same_mechanism_retries"]
    assert "grammar generated-answer retry" in retirement["blocked_same_mechanism_retries"]
    assert "stop/parser generated-answer retry" in retirement["blocked_same_mechanism_retries"]
    assert "sota paired embeddings" in retirement["preserved_open_surfaces"]
    assert retirement["terminal_evidence"]["answer_channel_ready_score"] == 0.0
    assert retirement["terminal_evidence"]["qualified_sota_family_count"] == 0
    assert retirement["terminal_evidence"]["exact_label_count"] == 2
    assert retirement["terminal_evidence"]["parser_failure_count"] == 138
    assert retirement["terminal_evidence"]["truncation_count"] == 134

    missing_root = tmp_path / "missing-retirement"
    _make_root(missing_root, include_retirement=False)
    missing = _clean_build(missing_root)
    assert missing["status"] == "blocked"
    assert "answer_transport_retirement_missing" in missing["preconditions_checked"][
        "failed_preconditions"
    ]


def test_scenario_report_5823_collision_blocks_allocation(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5823-COLLISION-BLOCK: occupied V519 ids fail closed."""

    _make_root(tmp_path)
    _write_json(tmp_path, "results/experiment_5836_existing_collision.json", {"status": "old"})
    payload = _research_complete_payload()
    payload["milestones"].append(
        {
            "id": "2026.07.400",
            "title": "stale next-range reference",
            "finding": "mentions exp5834-bounded-adaptive-memory-microkernel",
            "tasks": [],
        }
    )
    _write_text(tmp_path, mod.RESEARCH_COMPLETE_RELATIVE_PATH, yaml.safe_dump(payload))

    report = _clean_build(tmp_path)

    assert report["status"] == "blocked"
    assert report["honest_verdict"].startswith("blocked:")
    assert report["next_range_collision_count"] == 2
    assert [row["path"] for row in report["collision_scan"]["preexisting_collisions"]] == [
        "research-complete.yaml",
        "results/experiment_5836_existing_collision.json",
    ]


def test_scenario_report_5823_emit_report_field_provenance_and_checksum(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5823-FIELD-PROVENANCE: emitted artifact is stable."""

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
        duration_s=1.25,
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert (tmp_path / mod.RESEARCH_COMPLETE_RELATIVE_PATH).read_bytes() == history_before
    assert written == report
    assert mod.payload_checksum(written) == written["reproducibility_checksum"]
    assert set(mod.REQUIRED_PRINCIPLE_FIELDS).issubset(report["field_principles"])
    assert set(mod.REQUIRED_PRINCIPLE_FIELDS).issubset(report["field_provenance"])
    assert all(report["field_principles"][field] for field in mod.REQUIRED_PRINCIPLE_FIELDS)
    assert all(
        report["field_provenance"][field]["sources"]
        for field in mod.REQUIRED_PRINCIPLE_FIELDS
    )
    assert report["duration_s"] == 1.25
    assert report["inference_substrate"] == "aggregation_from_upstream_artifacts"

    original = mod.FIELD_PRINCIPLES.pop("status")
    try:
        with pytest.raises(KeyError, match="missing field principles"):
            mod.build_report(
                tmp_path,
                modification_overrides={
                    mod.ROADMAP_RELATIVE_PATH: False,
                    mod.CONDUCTOR_RELATIVE_PATH: False,
                },
                duration_s=1.25,
            )
    finally:
        mod.FIELD_PRINCIPLES["status"] = original


def test_scenario_report_5823_defensive_precondition_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-5823: malformed inputs and identity ambiguity fail closed."""

    assert mod._task_signature({"tasks": "not-list"}) == ()
    assert mod._parse_conductor_log(tmp_path / "missing-log-file") == []
    assert mod._artifact_terminal_status({}, {"exists": False, "loadable": False}) == "missing"
    assert mod._artifact_terminal_status({}, {"exists": True, "loadable": False}) == "malformed"
    assert (
        mod._artifact_terminal_status(
            {"honest_verdict": "blocked: no input"},
            {"exists": True, "loadable": True},
        )
        == "blocked"
    )
    assert mod._artifact_terminal_status({}, {"exists": True, "loadable": True}) == "unknown"
    assert mod._task_number("not-an-exp") is None
    fallback_evidence = mod._exp5813_terminal_evidence(
        {
            "answer_channel_ready_score": 0.0,
            "qualified_real_sota_model_count": 0,
            "independent_failure_metrics": {
                "row_count": 144,
                "exact_label_coverage": 2 / 144,
                "parser_failure_count": 138,
                "truncation_count": 134,
            },
        }
    )
    assert fallback_evidence["exact_label_count"] == 2
    assert fallback_evidence["qualified_sota_family_count"] == 0

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
    no_history = _clean_build(no_history_root)
    assert no_history["research_complete_append_count"] == 1

    missing_artifact_root = tmp_path / "missing-artifact"
    _make_root(missing_artifact_root)
    (
        missing_artifact_root
        / mod.TASK_ARTIFACT_PATHS["exp5812-split-budget-channel-contract"]
    ).unlink()
    missing_artifact = _clean_build(missing_artifact_root)
    assert missing_artifact["status"] == "blocked"
    assert (
        "missing_or_malformed_declared_deliverables=['exp5812-split-budget-channel-contract']"
        in missing_artifact["preconditions_checked"]["failed_preconditions"]
    )

    bad_active_root = tmp_path / "bad-active"
    _make_root(bad_active_root)
    _write_text(bad_active_root, mod.ROADMAP_RELATIVE_PATH, "bad: [yaml\n")
    bad_active = _clean_build(bad_active_root)
    assert "active_roadmap_unparseable" in bad_active["preconditions_checked"][
        "failed_preconditions"
    ]

    bad_next_root = tmp_path / "bad-next"
    _make_root(bad_next_root)
    _write_text(bad_next_root, mod.ROADMAP_NEXT_RELATIVE_PATH, "bad: [yaml\n")
    bad_next = _clean_build(bad_next_root)
    assert "next_roadmap_unparseable" in bad_next["preconditions_checked"][
        "failed_preconditions"
    ]

    missing_log_root = tmp_path / "missing-log"
    _make_root(missing_log_root)
    _write_text(missing_log_root, mod.CONDUCTOR_LOG_RELATIVE_PATH, "no rows\n")
    missing_log = _clean_build(missing_log_root)
    assert any(
        item.startswith("missing_conductor_outcomes=")
        for item in missing_log["preconditions_checked"]["failed_preconditions"]
    )

    short_log_root = tmp_path / "short-log"
    short_log_root.mkdir()
    _write_text(short_log_root, mod.CONDUCTOR_LOG_RELATIVE_PATH, "| too | short |\n")
    assert mod._parse_conductor_log(short_log_root) == []

    alias_root = tmp_path / "alias"
    _make_root(alias_root)
    _write_json(
        alias_root,
        "results/experiment_5813_same_number_alias.json",
        {"status": "complete"},
    )
    alias_report = _clean_build(alias_root)
    assert alias_report["same_number_alias_groups"]["5813"]["aliases"][0]["path"] == (
        "results/experiment_5813_same_number_alias.json"
    )

    row_missing_root = tmp_path / "row-missing"
    _make_root(row_missing_root)
    (row_missing_root / mod.EXP5813_ROW_RELATIVE_PATH).unlink()
    row_missing = _clean_build(row_missing_root)
    assert "exp5813_row_file_missing" in row_missing["preconditions_checked"][
        "failed_preconditions"
    ]

    bad_metrics_root = tmp_path / "bad-metrics"
    _make_root(bad_metrics_root)
    bad_metrics = _artifact_payload("exp5813-split-budget-sota-canary")
    bad_metrics["exact_label_count"] = 1
    _write_json(
        bad_metrics_root,
        mod.TASK_ARTIFACT_PATHS["exp5813-split-budget-sota-canary"],
        bad_metrics,
    )
    bad_metrics_report = _clean_build(bad_metrics_root)
    assert "exp5813_clean_negative_metrics_not_confirmed" in bad_metrics_report[
        "preconditions_checked"
    ]["failed_preconditions"]

    missing_gate_root = tmp_path / "missing-gate"
    _make_root(missing_gate_root)
    (missing_gate_root / mod.TASK_ARTIFACT_PATHS["exp5814-channel-qualified-constraint-stream"]).unlink()
    missing_gate = _clean_build(missing_gate_root)
    assert "exp5814_gate_block_not_confirmed" in missing_gate["preconditions_checked"][
        "failed_preconditions"
    ]

    ambiguous_root = tmp_path / "ambiguous-history"
    _make_root(ambiguous_root)
    ambiguous_payload = _research_complete_payload()
    altered = json.loads(json.dumps(ambiguous_payload["milestones"][0]))
    altered["tasks"][0]["deliverable"] = "results/other.json"
    ambiguous_payload["milestones"].append(altered)
    _write_text(ambiguous_root, mod.RESEARCH_COMPLETE_RELATIVE_PATH, yaml.safe_dump(ambiguous_payload))
    ambiguous = _clean_build(ambiguous_root)
    assert "ambiguous_research_complete_declared_task_blocks" in ambiguous[
        "preconditions_checked"
    ]["failed_preconditions"]

    duplicate_task_root = tmp_path / "duplicate-task"
    _make_root(duplicate_task_root)
    duplicate_payload = _research_complete_payload()
    duplicate_task = dict(duplicate_payload["milestones"][0]["tasks"][0])
    duplicate_task["deliverable"] = "results/conflicting.json"
    duplicate_payload["milestones"][0]["tasks"].append(duplicate_task)
    _write_text(
        duplicate_task_root,
        mod.RESEARCH_COMPLETE_RELATIVE_PATH,
        yaml.safe_dump(duplicate_payload),
    )
    duplicate_task_report = _clean_build(duplicate_task_root)
    assert any(
        item.startswith("duplicate_task_id_conflicts=")
        for item in duplicate_task_report["preconditions_checked"]["failed_preconditions"]
    )
    assert any(
        item.startswith("declared_task_ids_mismatch=")
        for item in duplicate_task_report["preconditions_checked"]["failed_preconditions"]
    )

    transition_collision_root = tmp_path / "transition-collision"
    _make_root(transition_collision_root)
    _write_text(transition_collision_root, "results/experiment_5830_directory", "")
    (transition_collision_root / "results/experiment_5830_directory").unlink()
    (transition_collision_root / "results/experiment_5830_directory").mkdir()
    _write_text(
        transition_collision_root,
        "results/experiment_5831_transition_bad.json",
        "{",
    )
    _write_json(
        transition_collision_root,
        "results/experiment_5700_transition_alloc.json",
        {"next_task_range": "exp5823-exp5836"},
    )
    transition_collision = _clean_build(transition_collision_root)
    assert {
        "path": "results/experiment_5700_transition_alloc.json",
        "kind": "prior_transition_allocation",
    } in transition_collision["collision_scan"]["preexisting_collisions"]

    flagged = mod._classify_outcomes(
        {"exp5813-split-budget-sota-canary": {"status": "complete", "artifact_flagged_adversarial": True}},
        {"exp5813-split-budget-sota-canary": {"outcome": "OK", "detail": ""}},
    )
    assert flagged["flagged_task_ids"] == ["exp5813-split-budget-sota-canary"]
    blocked = mod._classify_outcomes(
        {"exp5814-channel-qualified-constraint-stream": {"status": "blocked"}},
        {"exp5814-channel-qualified-constraint-stream": {"outcome": "OK", "detail": ""}},
    )
    assert blocked["gate_blocked_task_ids"] == ["exp5814-channel-qualified-constraint-stream"]

    original_expected = mod.EXPECTED_TASK_IDS
    monkeypatch.setattr(mod, "EXPECTED_TASK_IDS", ("exp9999-other",))
    try:
        assert mod._classify_outcomes(
            {"exp9999-other": {"status": "complete"}},
            {"exp9999-other": {"outcome": "OK", "detail": ""}},
        )["missing_task_ids"] == ["exp9999-other"]
    finally:
        monkeypatch.setattr(mod, "EXPECTED_TASK_IDS", original_expected)

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
    bad_task = _clean_build(bad_task_root)
    assert any(
        item.startswith("active_roadmap_task_ids=")
        for item in bad_task["preconditions_checked"]["failed_preconditions"]
    )

    bad_retirement_payload_root = tmp_path / "bad-retirement-payload"
    _make_root(bad_retirement_payload_root)
    payload = _manifest_payload(include_retirement=True)
    payload["retired_experiments"][0]["retire_if_same_verdict"] = False
    _write_text(
        bad_retirement_payload_root,
        mod.EXCLUSION_MANIFEST_RELATIVE_PATH,
        yaml.safe_dump(payload),
    )
    bad_retirement = _clean_build(bad_retirement_payload_root)
    assert "answer_transport_retirement_missing" in bad_retirement["preconditions_checked"][
        "failed_preconditions"
    ]

    modified_root = tmp_path / "modified"
    _make_root(modified_root)
    modified = mod.build_report(
        modified_root,
        tests_run=[{"command": "unit", "exit_code": 1}],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: True,
            mod.CONDUCTOR_RELATIVE_PATH: True,
        },
    )
    failed = modified["preconditions_checked"]["failed_preconditions"]
    assert "research_roadmap_modified" in failed
    assert "research_conductor_modified" in failed
    assert any(item.startswith("test_failures=") for item in failed)
    assert modified["duration_s"] > 0
    assert mod._tests_failed([{"command": "global baseline", "exit_code": 1, "blocking": False}]) == []

    monkeypatch.setattr(mod, "EXPECTED_TASK_IDS", ("not-an-exp-task",))
    assert mod._same_number_alias_groups(tmp_path, {}) == {}
