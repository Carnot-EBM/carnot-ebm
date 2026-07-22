"""Tests for the Exp5782 V516 transition receipt.

Spec refs: REQ-REPORT-5782, SCENARIO-REPORT-5782,
SCENARIO-REPORT-5782-COLLISION-BLOCK,
SCENARIO-REPORT-5782-IDENTITY-BLOCK,
SCENARIO-REPORT-5782-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_5782_transition_v516 as mod


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
    verdicts = {
        "exp5769-transition-v515": (
            "complete: archived terminal .514 evidence by exact declared deliverables "
            "into .515; same-number aliases disclosed; next_range_collision_count=0; "
            "research_complete_append_count=0"
        ),
        "exp5770-v515-source-delta-ingestion": (
            "complete: accepted one post-marker ARC play-adequacy control; closed scopes preserved"
        ),
        "exp5771-evidence-index-collision-preflight": (
            "blocked: evidence index preflight failed closed: tests_not_recorded_passing"
        ),
        "exp5773-prospective-constraint-acquisition-ab": "blocked_gate_check_failed",
        "exp5775-constraint-sidecar-shadow-integration": "blocked_gate_check_failed",
    }
    if task_id in {
        "exp5773-prospective-constraint-acquisition-ab",
        "exp5775-constraint-sidecar-shadow-integration",
    }:
        return {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "blocked_at_layer": "conductor_pre_gate",
            "honest_verdict": verdicts[task_id],
        }
    return {
        "experiment_id": task_id,
        "status": "blocked"
        if task_id == "exp5771-evidence-index-collision-preflight"
        else "complete",
        "honest_verdict": verdicts[task_id],
    }


def _research_complete_payload(*, duplicate_514_blocks: int = 2) -> JsonDict:
    v514_block = {
        "id": "2026.07.514",
        "title": "Prior duplicate history preserved",
        "tasks": [
            {
                "id": "exp5755-transition-v514",
                "deliverable": "results/experiment_5755_transition_v514.json",
                "result": "OK (conductor)",
            }
        ],
    }
    v515_block = {
        "id": mod.MILESTONE_FROM,
        "title": "Prospective Constraint Drift, Accredited World Models, and Verified Online Adaptation",
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
    return {"milestones": [v514_block for _ in range(duplicate_514_blocks)] + [v515_block]}


def _roadmap_payload() -> JsonDict:
    return {
        "milestone": mod.MILESTONE_TO,
        "tasks": [
            {
                "id": task_id,
                "milestone": mod.MILESTONE_TO,
                "deliverable": mod.NEXT_TASK_ARTIFACT_PATHS[task_id].as_posix(),
            }
            for task_id in mod.NEXT_TASK_IDS
        ],
    }


def _conductor_log() -> str:
    return "\n".join(
        [
            "| 2026-07-22 00:07 UTC | Milestone 2026.07.515 activated | OK | 7 tasks queued |",
            "| 2026-07-22 00:55 UTC | Archive terminal .514 evidence with collision-disc | OK | 89 passed, 1 warning in 13.21s |",
            "| 2026-07-22 01:33 UTC | Ingest post-V515 source deltas with bounded biblio | OK | 87 passed, 2 warnings in 12.99s |",
            "| 2026-07-22 02:14 UTC | Build an exact-deliverable evidence index and fail | FAIL | artifact_not_updated_past_bootstrap (deliverable=results/experiment_5771_evidenc |",
            "| 2026-07-22 02:32 UTC | Build an exact-deliverable evidence index and fail | FAIL | artifact_not_updated_past_bootstrap (deliverable=results/experiment_5771_evidenc |",
            "| 2026-07-22 03:01 UTC | Build an exact-deliverable evidence index and fail | FAIL | artifact_not_updated_past_bootstrap (deliverable=results/experiment_5771_evidenc |",
            "| 2026-07-22 03:03 UTC | Gated on Exp5771 evidence readiness: build a prosp | GATE_BLOCK | Pre-emptive skip: upstream retired (exp5771-evidence-index-collision-preflight,  |",
            "| 2026-07-22 03:03 UTC | Gated on Exp5772 clean drift headroom: run prospec | GATE_BLOCK | 5 of 5 gate(s) failed; first failure: exp5772-sota-constraint-drift-stream.strea |",
            "| 2026-07-22 03:05 UTC | Gated on Exp5771 evidence readiness: build a prosp | GATE_BLOCK | Pre-emptive skip: upstream retired (exp5771-evidence-index-collision-preflight,  |",
            "| 2026-07-22 03:05 UTC | Gated on Exp5772 clean drift headroom: run prospec | GATE_BLOCK | 5 of 5 gate(s) failed; first failure: exp5772-sota-constraint-drift-stream.strea |",
            "| 2026-07-22 03:07 UTC | Gated on Exp5771 evidence readiness: build a prosp | GATE_BLOCK | Pre-emptive skip: upstream retired (exp5771-evidence-index-collision-preflight,  |",
            "| 2026-07-22 03:07 UTC | Gated on Exp5772 clean drift headroom: run prospec | GATE_BLOCK | 5 of 5 gate(s) failed; first failure: exp5772-sota-constraint-drift-stream.strea |",
            "| 2026-07-22 03:09 UTC | Gated on Exp5773 credited learning: test leave-one | GATE_BLOCK | Pre-emptive skip: upstream retired (exp5773-prospective-constraint-acquisition-a |",
            "| 2026-07-22 03:09 UTC | Gated on Exp5774 cross-family transfer: wire a dis | GATE_BLOCK | 5 of 5 gate(s) failed; first failure: exp5774-constraint-transfer-forgetting-aud |",
            "| 2026-07-22 03:11 UTC | Gated on Exp5773 credited learning: test leave-one | GATE_BLOCK | Pre-emptive skip: upstream retired (exp5773-prospective-constraint-acquisition-a |",
            "| 2026-07-22 03:11 UTC | Gated on Exp5774 cross-family transfer: wire a dis | GATE_BLOCK | 5 of 5 gate(s) failed; first failure: exp5774-constraint-transfer-forgetting-aud |",
            "| 2026-07-22 03:13 UTC | Gated on Exp5773 credited learning: test leave-one | GATE_BLOCK | Pre-emptive skip: upstream retired (exp5773-prospective-constraint-acquisition-a |",
            "| 2026-07-22 03:13 UTC | Gated on Exp5774 cross-family transfer: wire a dis | GATE_BLOCK | 5 of 5 gate(s) failed; first failure: exp5774-constraint-transfer-forgetting-aud |",
        ]
    )


def _make_root(root: Path, *, duplicate_514_blocks: int = 2) -> None:
    for task_id, rel_path in mod.TASK_ARTIFACT_PATHS.items():
        if task_id in mod.MISSING_GATE_ARTIFACT_TASK_IDS:
            continue
        _write_json(root, rel_path, _artifact_payload(task_id))

    _write_text(root, mod.ROADMAP_RELATIVE_PATH, yaml.safe_dump(_roadmap_payload()))
    _write_text(
        root,
        mod.VNEXT_RELATIVE_PATH,
        "# Research Roadmap vNEXT\n\n"
        "**Milestone:** 2026.07.516\n\n"
        "**Task range:** Exp5782-Exp5795\n\n"
        "`results/experiment_5782_transition_v516.json`\n"
        "`results/experiment_5795_v516_capstone_reconciliation.json`\n",
    )
    _write_text(
        root,
        mod.RESEARCH_COMPLETE_RELATIVE_PATH,
        yaml.safe_dump(_research_complete_payload(duplicate_514_blocks=duplicate_514_blocks)),
    )
    _write_text(root, mod.CONDUCTOR_LOG_RELATIVE_PATH, _conductor_log())
    _write_text(root, mod.EXCLUSION_MANIFEST_RELATIVE_PATH, "exclusions: []\n")
    _write_text(root, mod.CONDUCTOR_RELATIVE_PATH, "# conductor fixture\n")


def test_spec_contains_req_report_5782_contract() -> None:
    """REQ-REPORT-5782: OpenSpec names exact V515 identity and V516 collision gate."""

    text = SPEC_PATH.read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-5782") :]

    assert str(mod.RESULT_RELATIVE_PATH) in section
    assert "(milestone, task_id, declared_deliverable)" in section
    assert "SCENARIO-REPORT-5782-COLLISION-BLOCK" in section
    assert "artifact_not_updated_past_bootstrap" in section


def test_scenario_report_5782_archives_terminal_v515_by_exact_identity(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5782: declared paths, not numeric prefixes, are canonical."""

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
    assert report["milestone_from"] == "2026.07.515"
    assert report["milestone_to"] == "2026.07.516"
    assert report["artifact_selection_policy"] == "exact_declared_deliverable"
    assert report["canonical_identity_contract"]["identity_tuple"] == [
        "milestone",
        "task_id",
        "declared_deliverable",
    ]
    assert report["archived_task_ids"] == list(mod.EXPECTED_TASK_IDS)
    assert len(report["declared_deliverable_matrix"]) == 7
    assert report["research_complete_append_count"] == 0
    assert report["docs_reconciled"]["mode"] == (
        "already_archived_preserving_duplicate_history_no_rewrite"
    )
    assert report["preconditions_checked"]["roadmaps"]["active"]["parsed"] is True
    assert report["preconditions_checked"]["roadmaps"]["next"]["present"] is False
    assert report["preconditions_checked"]["resource_receipts"]["disk_free_bytes"] > 0
    assert report["duplicate_history_diagnostics"]["milestone_from_block_count"] == 1
    assert report["duplicate_history_diagnostics"]["duplicate_milestone_blocks"] == [
        {
            "block_count": 2,
            "milestone": "2026.07.514",
            "mutation": "preserved_read_only",
            "unique_block_signature_count": 1,
        }
    ]
    missing_gate = report["canonical_artifact_hashes"]["exp5772-sota-constraint-drift-stream"]
    assert missing_gate["path"] == "results/experiment_5772_sota_constraint_drift_stream.json"
    assert missing_gate["present"] is False
    assert missing_gate["status"] == "missing-gate-block"
    assert missing_gate["non_artifact_outcome_authorized"] is True
    assert report["next_task_range"] == "exp5782-exp5795"
    assert report["next_range_collision_count"] == 0
    assert isinstance(report["next_range_collision_count"], int)
    assert report["research_roadmap_unchanged"] is True
    assert report["conductor_unchanged"] is True


def test_scenario_report_5782_preserves_operational_blocks_and_gate_skips(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5782: gate blocks are not scientific nulls or positives."""

    _make_root(tmp_path)
    report = mod.build_report(
        tmp_path,
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    assert report["completed_operational_task_ids"] == [
        "exp5769-transition-v515",
        "exp5770-v515-source-delta-ingestion",
    ]
    assert report["delivery_failed_task_ids"] == ["exp5771-evidence-index-collision-preflight"]
    assert report["blocked_task_ids"] == [
        "exp5771-evidence-index-collision-preflight",
        "exp5772-sota-constraint-drift-stream",
        "exp5773-prospective-constraint-acquisition-ab",
        "exp5774-constraint-transfer-forgetting-audit",
        "exp5775-constraint-sidecar-shadow-integration",
    ]
    assert report["gate_skipped_task_ids"] == [
        "exp5772-sota-constraint-drift-stream",
        "exp5773-prospective-constraint-acquisition-ab",
        "exp5774-constraint-transfer-forgetting-audit",
        "exp5775-constraint-sidecar-shadow-integration",
    ]
    assert report["scientific_null_task_ids"] == []
    assert report["positive_result_task_ids"] == []
    assert set(report["gate_skipped_task_ids"]).isdisjoint(report["scientific_null_task_ids"])
    exp5771 = report["conductor_outcomes"]["exp5771-evidence-index-collision-preflight"]
    assert exp5771["delivery_failure_count"] == 3
    assert exp5771["delivery_failure_reason"] == "artifact_not_updated_past_bootstrap"
    assert exp5771["terminal_artifact_honest_verdict"] == (
        "blocked: evidence index preflight failed closed: tests_not_recorded_passing"
    )
    assert (
        report["conductor_outcomes"]["exp5773-prospective-constraint-acquisition-ab"][
            "artifact_status"
        ]
        == "blocked-gate"
    )
    assert (
        report["conductor_outcomes"]["exp5774-constraint-transfer-forgetting-audit"][
            "artifact_status"
        ]
        == "missing-gate-block"
    )


def test_scenario_report_5782_alias_groups_never_replace_declared_paths(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5782: same-number files stay aliases, even when canonical is missing."""

    _make_root(tmp_path)
    _write_json(
        tmp_path,
        "results/experiment_5771_older_attempt_alias.json",
        {"status": "complete", "honest_verdict": "complete: old alias"},
    )
    _write_json(
        tmp_path,
        "results/experiment_5772_proxy_alias.json",
        {"status": "complete", "honest_verdict": "complete: proxy alias"},
    )

    report = mod.build_report(
        tmp_path,
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    exp5771 = report["same_number_alias_groups"]["5771"]
    assert exp5771["canonical"]["path"] == (
        "results/experiment_5771_evidence_index_collision_preflight.json"
    )
    assert [row["path"] for row in exp5771["aliases"]] == [
        "results/experiment_5771_older_attempt_alias.json"
    ]
    exp5772 = report["same_number_alias_groups"]["5772"]
    assert exp5772["canonical"]["path"] == (
        "results/experiment_5772_sota_constraint_drift_stream.json"
    )
    assert exp5772["canonical"]["present"] is False
    assert [row["path"] for row in exp5772["aliases"]] == [
        "results/experiment_5772_proxy_alias.json"
    ]
    assert (
        report["canonical_artifact_hashes"]["exp5772-sota-constraint-drift-stream"]["sha256"]
        is None
    )


def test_scenario_report_5782_collision_blocks_allocation(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5782-COLLISION-BLOCK: occupied V516 ids fail closed."""

    _make_root(tmp_path)
    _write_json(tmp_path, "results/experiment_5786_existing_collision.json", {"status": "old"})
    payload = _research_complete_payload()
    payload["milestones"].append(
        {
            "id": "2026.07.400",
            "title": "stale next-range reference",
            "finding": "mentions exp5791-arc-sota-independent-hypothesis-panel",
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
        "results/experiment_5786_existing_collision.json",
    ]


def test_scenario_report_5782_identity_failures_block(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5782-IDENTITY-BLOCK: ambiguous declared paths fail closed."""

    _make_root(tmp_path)
    (tmp_path / mod.TASK_ARTIFACT_PATHS["exp5770-v515-source-delta-ingestion"]).unlink()
    missing_report = mod.build_report(
        tmp_path,
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    assert missing_report["status"] == "blocked"
    assert (
        "missing_or_malformed_declared_deliverables=['exp5770-v515-source-delta-ingestion']"
        in missing_report["preconditions_checked"]["failed_preconditions"]
    )

    mismatch_root = tmp_path / "mismatch"
    _make_root(mismatch_root)
    payload = _research_complete_payload()
    payload["milestones"][-1]["tasks"][1]["deliverable"] = "results/wrong.json"
    _write_text(mismatch_root, mod.RESEARCH_COMPLETE_RELATIVE_PATH, yaml.safe_dump(payload))
    mismatch_report = mod.build_report(
        mismatch_root,
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    assert mismatch_report["status"] == "blocked"
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

    empty_root = tmp_path / "empty"
    _make_root(empty_root)
    _write_text(empty_root, mod.RESEARCH_COMPLETE_RELATIVE_PATH, yaml.safe_dump({"milestones": []}))
    empty_report = mod.build_report(
        empty_root,
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    assert (
        "research_complete_515_block_count=0"
        in empty_report["preconditions_checked"]["failed_preconditions"]
    )


def test_scenario_report_5782_emit_report_and_field_principles(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5782-FIELD-PRINCIPLES: emitted artifact is stable."""

    _make_root(tmp_path)
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

    assert written["reproducibility_checksum"] == report["reproducibility_checksum"]
    assert mod.payload_checksum(written) == written["reproducibility_checksum"]
    assert set(report).issubset(report["field_principles"])
    assert all(report["field_principles"][field] for field in report)
    assert mod._load_tests_run(None)[0]["status"] == "not_run"
    _payload, missing_meta = mod._read_yaml_with_meta(tmp_path / "missing.yaml")
    assert missing_meta["present"] is False
    assert missing_meta["parsed"] is False

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


def test_scenario_report_5782_defensive_precondition_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-REPORT-5782-IDENTITY-BLOCK: defensive checks are explicit."""

    assert mod._task_signature({"tasks": "not-list"}) == ()
    assert mod._payload_status({}, {"exists": True, "loadable": False}) == "malformed"
    assert mod._payload_status({}, {"exists": True, "loadable": True}) == "unknown"

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

    no_history_root = tmp_path / "no-history-list"
    no_history_root.mkdir()
    _write_text(no_history_root, mod.RESEARCH_COMPLETE_RELATIVE_PATH, "milestones: nope\n")
    assert mod._research_complete_blocks(no_history_root) == []
    assert mod._parse_conductor_log(tmp_path / "missing-log-file") == []
    short_log_root = tmp_path / "short-log"
    _write_text(short_log_root, mod.CONDUCTOR_LOG_RELATIVE_PATH, "| too | short |\n")
    assert mod._parse_conductor_log(short_log_root) == []
    assert mod._collision_scan(tmp_path / "empty-scan")["preexisting_collision_count"] == 0
    monkeypatch.setattr(mod, "EXPECTED_TASK_IDS", ("not-an-exp-task",))
    assert mod._same_number_alias_groups(tmp_path, {}) == {}
    monkeypatch.setattr(
        mod,
        "EXPECTED_TASK_IDS",
        tuple(mod.TASK_ARTIFACT_PATHS),
    )

    wrong_roadmap_root = tmp_path / "wrong-roadmap"
    _make_root(wrong_roadmap_root)
    _write_text(
        wrong_roadmap_root,
        mod.ROADMAP_RELATIVE_PATH,
        yaml.safe_dump({"milestone": "2026.07.515", "tasks": []}),
    )
    wrong_roadmap = mod.build_report(
        wrong_roadmap_root,
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: True,
            mod.CONDUCTOR_RELATIVE_PATH: True,
        },
    )
    failures = wrong_roadmap["preconditions_checked"]["failed_preconditions"]
    assert "active_roadmap_milestone='2026.07.515'" in failures
    assert any(item.startswith("active_roadmap_task_ids=") for item in failures)
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
        "active_roadmap_unparseable" in bad_active["preconditions_checked"]["failed_preconditions"]
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

    ambiguous_root = tmp_path / "ambiguous"
    _make_root(ambiguous_root)
    ambiguous_payload = _research_complete_payload()
    altered = json.loads(json.dumps(ambiguous_payload["milestones"][-1]))
    altered["tasks"][0]["deliverable"] = "results/other.json"
    ambiguous_payload["milestones"].append(altered)
    _write_text(
        ambiguous_root, mod.RESEARCH_COMPLETE_RELATIVE_PATH, yaml.safe_dump(ambiguous_payload)
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

    verdict_root = tmp_path / "bad-verdict"
    _make_root(verdict_root)
    bad_payload = _artifact_payload("exp5771-evidence-index-collision-preflight")
    bad_payload["honest_verdict"] = "blocked: different"
    _write_json(
        verdict_root,
        mod.TASK_ARTIFACT_PATHS["exp5771-evidence-index-collision-preflight"],
        bad_payload,
    )
    bad_verdict = mod.build_report(
        verdict_root,
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    assert (
        "exp5771_terminal_artifact_verdict_mismatch"
        in bad_verdict["preconditions_checked"]["failed_preconditions"]
    )
