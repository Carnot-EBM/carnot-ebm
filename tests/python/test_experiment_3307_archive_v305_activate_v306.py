"""Tests for Exp 3307 archive .305 and activate .306 handoff.

Spec refs: REQ-REPORT-3307, SCENARIO-REPORT-3307.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot.reporting import archive_v305_activate_v306_3307 as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/research-reporting/spec.md"


def _write_json(root: Path, rel_path: Path, payload: Mapping[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(root: Path, rel_path: Path, text: str) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quality_flag(kind: str, severity: str = "critical") -> dict[str, str]:
    return {
        "kind": kind,
        "severity": severity,
        "detail": f"{kind} detail",
    }


def _capstone_v305() -> dict[str, Any]:
    return {
        "artifact": "experiment_3306_capstone_v305",
        "experiment_id": "exp3306",
        "task_id": "exp3306-capstone-v305",
        "milestone": "2026.05.305",
        "capstone_v305_ready": True,
        "paper_ready": False,
        "publication_blocker_count": 8,
        "blocker_delta_from_v304": -2,
        "garak_gate_passed": True,
        "garak_attack_success_rate": 0.0,
        "repair_headline_claim_allowed": False,
        "fr11_memory_replay_safe": True,
        "kan_headline_retired": True,
        "next_top_gap": "clear_garak_dataflip_and_quality_flags",
        "gate_status_details": {
            "garak_gate": {
                "source_experiment_id": "exp3300",
                "garak_gate_passed": True,
                "dataflip_gate_passed": False,
                "blocker_reasons": ["dataflip_gate_failed", "dataflip_gate_passed=false"],
                "quality_flags": [_quality_flag("TAUTOLOGY"), _quality_flag("DURATION_TOO_SHORT")],
            },
            "repair_headline": {
                "source_experiment_id": "exp3303",
                "repair_headline_claim_allowed": False,
                "source_headline_claim_allowed": False,
                "blocker_reasons": [
                    "headline_claim_allowed_after_audit=false",
                    "source_headline_claim_allowed=false",
                    "source_provenance_clean=false",
                    "substrate_consistency_passed=false",
                ],
                "quality_flags": [_quality_flag("DURATION_TOO_SHORT")],
            },
            "fr11_replay": {
                "source_experiment_id": "exp3304",
                "fr11_replay_safe": True,
                "controller_memory_only": True,
                "foundation_weight_updates_performed": False,
                "blocker_reasons": [],
                "quality_flags": [],
            },
        },
        "honest_verdict": (
            "complete: capstone_v305_ready=true; paper_ready=false; "
            "publication_blocker_count=8; garak_gate_passed=true; "
            "garak_attack_success_rate=0.0; repair_headline_claim_allowed=false; "
            "fr11_memory_replay_safe=true; next_top_gap=clear_garak_dataflip_and_quality_flags"
        ),
    }


def _row(
    experiment_id: str,
    *,
    summary: Mapping[str, Any] | None = None,
    quality_flags: list[Mapping[str, str]] | None = None,
    blocker_reasons: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "summary": dict(summary or {}),
        "quality_flags": [dict(flag) for flag in (quality_flags or [])],
        "blocker_reasons": list(blocker_reasons or []),
        "present": True,
        "ready": True,
    }


def _matrix_v37() -> dict[str, Any]:
    return {
        "artifact": "experiment_3305_evidence_matrix_v37",
        "experiment_id": "exp3305",
        "task_id": "exp3305-evidence-matrix-v37",
        "matrix_v37_ready": True,
        "paper_ready": False,
        "paper_blocker_count": 8,
        "garak_gate_passed": True,
        "dataflip_gate_passed": False,
        "repair_headline_claim_allowed": False,
        "fr11_replay_safe": True,
        "top_gap": "clear_garak_dataflip_and_quality_flags",
        "gate_summary": {
            "garak_gate": {
                "source_experiment_id": "exp3300",
                "garak_gate_passed": True,
                "dataflip_gate_passed": False,
                "blocker_reasons": ["dataflip_gate_failed", "dataflip_gate_passed=false"],
                "quality_flags": [_quality_flag("TAUTOLOGY"), _quality_flag("DURATION_TOO_SHORT")],
            },
            "repair_headline": {
                "source_experiment_id": "exp3303",
                "repair_headline_claim_allowed": False,
                "blocker_reasons": [
                    "headline_claim_allowed_after_audit=false",
                    "source_provenance_clean=false",
                    "substrate_consistency_passed=false",
                ],
                "quality_flags": [_quality_flag("DURATION_TOO_SHORT")],
            },
            "fr11_replay": {
                "source_experiment_id": "exp3304",
                "fr11_replay_safe": True,
                "controller_memory_only": True,
                "foundation_weight_updates_performed": False,
            },
        },
        "rows": [
            _row(
                "exp3300",
                summary={
                    "garak_gate_passed": True,
                    "dataflip_gate_passed": False,
                    "attack_success_rate": 0.0,
                },
                quality_flags=[_quality_flag("TAUTOLOGY"), _quality_flag("DURATION_TOO_SHORT")],
                blocker_reasons=["dataflip_gate_failed"],
            ),
            _row(
                "exp3303",
                summary={
                    "headline_claim_allowed_after_audit": False,
                    "source_provenance_clean": False,
                    "substrate_consistency_passed": False,
                },
                quality_flags=[_quality_flag("DURATION_TOO_SHORT")],
                blocker_reasons=[
                    "headline_claim_allowed_after_audit=false",
                    "source_provenance_clean=false",
                    "substrate_consistency_passed=false",
                ],
            ),
            _row(
                "exp3304",
                summary={
                    "fr11_redteam_repair_memory_replay_ready": True,
                    "controller_memory_only": True,
                    "foundation_weight_updates_performed": False,
                    "retention_score": 0.982143,
                    "adaptation_score": 1.0,
                },
            ),
        ],
        "honest_verdict": "complete: matrix_v37_ready=true; paper_ready=false; paper_blockers=8",
    }


def _research_complete_yaml() -> str:
    lines = [
        "milestones:",
        "- id: 2026.05.305",
        "  title: Garak Red-Team Gate Pass + Headline-Eligible Repair Evidence",
        "  doc: openspec/change-proposals/research-roadmap-vNEXT.md",
        "  completed: '2026-05-29'",
        "  finding: See conductor log for per-experiment results.",
        "  tasks:",
    ]
    for task in mod.PRIOR_TASKS:
        lines.extend(
            [
                f"  - id: {task['id']}",
                f"    title: {task['title']}",
                f"    deliverable: {task['deliverable']}",
                "    result: OK (conductor)",
            ]
        )
    return "\n".join(lines) + "\n"


def _roadmap_yaml(milestone: str = "2026.05.306") -> str:
    return (
        f'milestone: "{milestone}"\n'
        'milestone_title: "DataFlip + Quality-Flag Cleanup For Publication-Ready Evidence"\n'
        'milestone_doc: "openspec/change-proposals/research-roadmap-vNEXT.md"\n'
        "tasks:\n"
        '  - id: "exp3307-archive-v305-activate-v306"\n'
        f'    milestone: "{milestone}"\n'
        '    deliverable: "results/experiment_3307_archive_v305_activate_v306.json"\n'
    )


def _conductor_log() -> str:
    return (
        "\n".join(
            [
                "| 2026-05-28 22:02 UTC | Close .304 ledger and open .305 Garak gate queue | OK | 81 passed in 3.96s |",
                "| 2026-05-28 22:15 UTC | Garak failure-mode autopsy v1 | OK | 81 passed in 3.58s |",
                "| 2026-05-28 22:28 UTC | Evidence substrate corrigendum and KAN no-retry ledger v1 | OK | 81 passed in 3.12s |",
                "| 2026-05-28 22:40 UTC | Prefix-closed Garak rogue-string guard pilot v1 | OK | 81 passed in 3.52s |",
                "| 2026-05-28 23:11 UTC | Red-team energy telemetry and routing policy v1 | OK | 81 passed in 3.22s |",
                "| 2026-05-28 23:42 UTC | Garak defense ablation v1 | OK | 81 passed in 3.43s |",
                "| 2026-05-29 00:03 UTC | Full Garak/DataFlip gate rerun v3 | OK | 101 passed in 4.03s |",
                "| 2026-05-29 00:16 UTC | Exact repair panel manifest v11 | OK | 81 passed in 3.67s |",
                "| 2026-05-29 00:36 UTC | Headline SOTA repair panel v11 | OK | 81 passed in 3.12s |",
                "| 2026-05-29 00:49 UTC | Repair headline evidence audit v1 | OK | 81 passed in 3.35s |",
                "| 2026-05-29 01:02 UTC | FR-11 red-team and repair memory replay v2 | OK | 81 passed in 3.31s |",
                "| 2026-05-29 01:17 UTC | Evidence matrix v37 | OK | 81 passed in 3.19s |",
                "| 2026-05-29 01:30 UTC | Capstone v305 | OK | 81 passed in 3.38s |",
                "| 2026-05-29 02:08 UTC | Milestone 2026.05.306 activated | OK | 14 tasks queued |",
            ]
        )
        + "\n"
    )


def _write_sources(root: Path, *, archived: bool = True) -> None:
    _write_json(root, mod.CAPSTONE_V305_REL_PATH, _capstone_v305())
    _write_json(root, mod.MATRIX_V37_REL_PATH, _matrix_v37())
    _write_text(
        root,
        mod.RESEARCH_COMPLETE_REL_PATH,
        _research_complete_yaml() if archived else "milestones: []\n",
    )
    _write_text(root, mod.ACTIVE_ROADMAP_REL_PATH, _roadmap_yaml())
    _write_text(root, mod.CONDUCTOR_LOG_REL_PATH, _conductor_log())
    _write_text(root, mod.CONDUCTOR_REL_PATH, "# protected conductor\n")


def test_req_report_3307_spec_anchor_declares_handoff_schema() -> None:
    """REQ-REPORT-3307: OpenSpec declares the .305/.306 handoff first."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-3307" in spec
    assert "SCENARIO-REPORT-3307" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_report_3307_existing_archive_opens_v306_without_mutation(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3307: existing .305 archive opens quality-cleanup .306 queue."""

    _write_sources(tmp_path)
    before_complete = (tmp_path / mod.RESEARCH_COMPLETE_REL_PATH).read_text(encoding="utf-8")
    before_roadmap = (tmp_path / mod.ACTIVE_ROADMAP_REL_PATH).read_text(encoding="utf-8")
    before_conductor = (tmp_path / mod.CONDUCTOR_REL_PATH).read_text(encoding="utf-8")

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=6.25)
    second = mod.build_artifact(tmp_path, started_s=10.0, now_s=11.0)
    output = mod.write_artifact(tmp_path, started_s=4.0, now_s=5.0)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert mod.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3307"
    assert artifact["task_id"] == mod.TASK_ID
    assert artifact["run_date"] == "20260529"
    assert artifact["source_milestone"] == "2026.05.305"
    assert artifact["target_milestone"] == "2026.05.306"
    assert artifact["archive_v305_activate_v306_ready"] is True
    assert artifact["v305_closed_v306_opened"] is True
    assert artifact["paper_ready"] is False
    assert artifact["publication_blocker_count"] == 8
    assert artifact["inherited_top_gap"] == "clear_garak_dataflip_and_quality_flags"
    assert artifact["garak_gate_passed"] is True
    assert artifact["garak_attack_success_rate"] == pytest.approx(0.0)
    assert artifact["dataflip_gate_passed"] is False
    assert artifact["repair_headline_claim_allowed"] is False
    assert artifact["fr11_memory_replay_safe"] is True
    assert artifact["protected_files_unchanged"] is True
    assert artifact["protected_file_checksums"]["research-roadmap.yaml"]["unchanged"] is True
    assert (
        artifact["protected_file_checksums"]["scripts/research_conductor.py"]["unchanged"] is True
    )
    assert artifact["research_complete_update"] == {
        "path": "research-complete.yaml",
        "appended": False,
        "already_present": True,
    }
    assert artifact["research_complete_source_summary"]["task_count"] == len(mod.PRIOR_TASKS)
    assert artifact["v306_queue"]["selected_queue_milestone"] == "2026.05.306"
    assert artifact["v306_queue"]["queue_first_task"] == mod.TASK_ID
    assert artifact["v306_activation_observed"] is True
    assert len(artifact["conductor_log_terminal_rows"]) == len(mod.PRIOR_TASKS)
    assert artifact["conductor_log_terminal_status_counts"] == {"OK": len(mod.PRIOR_TASKS)}
    assert artifact["terminal_v305_blockers"]["dataflip_failure"]["dataflip_gate_passed"] is False
    assert {flag["kind"] for flag in artifact["terminal_v305_blockers"]["quality_flags"]} == {
        "DURATION_TOO_SHORT",
        "TAUTOLOGY",
    }
    assert (
        "source_provenance_clean=false"
        in artifact["terminal_v305_blockers"]["repair_headline_provenance_failure"][
            "blocker_reasons"
        ]
    )
    assert artifact["v306_start_conditions"]["fr11_controller_memory_safety"] is True
    assert "DataFlip failure" in artifact["v306_activation_reason"]
    assert "quality flags" in artifact["v306_activation_reason"]
    assert "repair headline provenance failure" in artifact["v306_activation_reason"]
    assert "FR-11 controller-memory safety" in artifact["v306_activation_reason"]
    assert artifact["duration_s"] == pytest.approx(5.25)
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["reproducibility_checksum"] == second["reproducibility_checksum"]
    assert artifact["source_checksums"][mod.CAPSTONE_V305_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.CAPSTONE_V305_REL_PATH
    )
    assert saved["duration_s"] == pytest.approx(1.0)
    assert saved["honest_verdict"].startswith("complete:")
    assert "v305_closed_v306_opened=true" in saved["honest_verdict"]
    assert (tmp_path / mod.RESEARCH_COMPLETE_REL_PATH).read_text(
        encoding="utf-8"
    ) == before_complete
    assert (tmp_path / mod.ACTIVE_ROADMAP_REL_PATH).read_text(encoding="utf-8") == before_roadmap
    assert (tmp_path / mod.CONDUCTOR_REL_PATH).read_text(encoding="utf-8") == before_conductor
    mod.validate_artifact(artifact)


def test_req_report_3307_appends_missing_archive_once(tmp_path: Path) -> None:
    """REQ-REPORT-3307: missing .305 archive is materialized exactly once."""

    _write_sources(tmp_path, archived=False)

    output = mod.write_artifact(tmp_path, started_s=2.0, now_s=4.0)
    saved = json.loads(output.read_text(encoding="utf-8"))
    archive_text = (tmp_path / mod.RESEARCH_COMPLETE_REL_PATH).read_text(encoding="utf-8")
    ensure_result = mod.ensure_research_complete_entry(tmp_path)

    assert saved["v305_closed_v306_opened"] is True
    assert saved["research_complete_update"]["appended"] is True
    assert saved["research_complete_update"]["already_present"] is False
    assert archive_text.count("- id: 2026.05.305") == 1
    assert ensure_result == {
        "path": "research-complete.yaml",
        "appended": False,
        "already_present": True,
    }


def test_req_report_3307_fail_closed_helpers_and_validation(tmp_path: Path) -> None:
    """REQ-REPORT-3307: malformed inputs stay explicit and non-fabricated."""

    _write_sources(tmp_path)
    _write_json(
        tmp_path,
        mod.CAPSTONE_V305_REL_PATH,
        _capstone_v305()
        | {
            "capstone_v305_ready": False,
            "paper_ready": True,
            "publication_blocker_count": 0,
            "next_top_gap": "wrong",
            "garak_gate_passed": False,
            "dataflip_gate_passed": True,
            "repair_headline_claim_allowed": True,
            "fr11_memory_replay_safe": False,
            "gate_status_details": {},
        },
    )
    _write_json(
        tmp_path,
        mod.MATRIX_V37_REL_PATH,
        _matrix_v37()
        | {
            "matrix_v37_ready": False,
            "paper_blocker_count": 0,
            "top_gap": "wrong",
            "dataflip_gate_passed": True,
            "repair_headline_claim_allowed": True,
            "fr11_replay_safe": False,
            "rows": [],
            "gate_summary": {},
        },
    )
    _write_text(tmp_path, mod.ACTIVE_ROADMAP_REL_PATH, _roadmap_yaml("2026.05.305"))
    _write_text(tmp_path, mod.RESEARCH_COMPLETE_REL_PATH, "[")
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("tasks: [", encoding="utf-8")

    baseline = mod.protected_file_checksums(tmp_path)
    _write_text(tmp_path, mod.CONDUCTOR_REL_PATH, "# modified protected conductor\n")
    artifact = mod.build_artifact(
        tmp_path,
        protected_hash_baseline=baseline,
        started_s=8.0,
        now_s=3.0,
    )

    assert artifact["v305_closed_v306_opened"] is False
    assert artifact["archive_v305_activate_v306_ready"] is False
    assert artifact["duration_s"] == 0.0
    assert artifact["paper_ready"] is True
    assert artifact["publication_blocker_count"] == 0
    assert artifact["inherited_top_gap"] == "wrong"
    assert artifact["garak_gate_passed"] is False
    assert artifact["dataflip_gate_passed"] is True
    assert artifact["repair_headline_claim_allowed"] is True
    assert artifact["fr11_memory_replay_safe"] is False
    assert artifact["protected_files_unchanged"] is False
    assert "capstone_v305 authority is not ready" in artifact["blocked_reasons"]
    assert "matrix_v37 authority is not ready" in artifact["blocked_reasons"]
    assert "publication blocker count is not 8" in artifact["blocked_reasons"]
    assert (
        "inherited top gap is not clear_garak_dataflip_and_quality_flags"
        in artifact["blocked_reasons"]
    )
    assert "selected queue milestone is not 2026.05.306" in artifact["blocked_reasons"]
    assert "protected files changed during handoff" in artifact["blocked_reasons"]
    assert "DataFlip gate must remain failed at .306 activation" in artifact["blocked_reasons"]
    assert artifact["honest_verdict"].startswith("complete:")

    assert mod.read_json_object(bad_json) == {}
    assert mod.read_json_object(list_json) == {}
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.read_yaml_document(bad_yaml) == {}
    assert mod.read_yaml_document(tmp_path / "missing.yaml") == {}
    assert mod.sha256_file(tmp_path / "missing.json") is None
    assert mod._milestone_entries("bad") == []
    assert mod._milestone_entries([{"id": "x"}, "bad"]) == [{"id": "x"}]
    assert mod._as_mapping([]) == {}
    assert mod._as_list({}) == []
    assert mod._int_value(True) == 0
    assert mod._int_value(7) == 7
    assert mod._float_value(True) == 0.0
    assert mod._float_value(1) == 1.0
    assert mod._file_contains(tmp_path / "missing.log", "needle") is False
    assert mod._terminal_prefix_ok("success: done") is True
    assert mod._terminal_prefix_ok("blocked") is False
    assert mod._parse_conductor_line("not a conductor row") == {}
    assert all(
        row["status"] == "missing"
        for row in mod._conductor_log_terminal_rows(tmp_path / "missing-log-root")
    )
    assert mod._quality_flags({"quality_flags": [_quality_flag("TAUTOLOGY")]}) == [
        _quality_flag("TAUTOLOGY")
    ]
    assert mod._quality_flags({"quality_flags": "bad"}) == []
    assert mod._dedupe_quality_flags([_quality_flag("TAUTOLOGY"), _quality_flag("TAUTOLOGY")]) == [
        _quality_flag("TAUTOLOGY")
    ]

    empty_archive = tmp_path / "empty" / mod.RESEARCH_COMPLETE_REL_PATH
    mod._append_research_complete_entry(empty_archive)
    assert empty_archive.read_text(encoding="utf-8").startswith("milestones:\n- id: 2026.05.305")
    no_newline_archive = tmp_path / "no-newline" / mod.RESEARCH_COMPLETE_REL_PATH
    no_newline_archive.parent.mkdir(parents=True)
    no_newline_archive.write_text("milestones:\n- id: 2026.05.304\n  tasks: []", encoding="utf-8")
    mod._append_research_complete_entry(no_newline_archive)
    assert no_newline_archive.read_text(encoding="utf-8").count("- id: 2026.05.305") == 1

    summary_root = tmp_path / "summary"
    _write_text(
        summary_root,
        mod.RESEARCH_COMPLETE_REL_PATH,
        "milestones:\n- id: 2026.05.304\n  tasks: []\n- id: 2026.05.305\n  tasks: []\n",
    )
    assert mod._research_complete_task_summary(summary_root)["task_count"] == 0

    good_artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.0)
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({})
    with pytest.raises(ValueError, match="experiment_id"):
        mod.validate_artifact(good_artifact | {"experiment_id": "bad"})
    with pytest.raises(ValueError, match="task_id"):
        mod.validate_artifact(good_artifact | {"task_id": "bad"})
    with pytest.raises(ValueError, match="source_milestone"):
        mod.validate_artifact(good_artifact | {"source_milestone": "bad"})
    with pytest.raises(ValueError, match="target_milestone"):
        mod.validate_artifact(good_artifact | {"target_milestone": "bad"})
    with pytest.raises(ValueError, match="random_seed"):
        mod.validate_artifact(good_artifact | {"random_seed": 0})
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(good_artifact | {"inference_substrate": "live"})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(good_artifact | {"honest_verdict": "blocked"})
    with pytest.raises(ValueError, match="publication_blocker_count"):
        mod.validate_artifact(good_artifact | {"publication_blocker_count": -1})
    with pytest.raises(ValueError, match="no_push"):
        mod.validate_artifact(good_artifact | {"no_push": False})

    assert mod._gate_detail(
        {},
        {"gate_summary": {"garak_gate": {"source_experiment_id": "exp3300"}}},
        "garak_gate",
        "exp3300",
    ) == {"source_experiment_id": "exp3300"}
    assert mod._garak_attack_success_rate(
        {},
        {"rows": [{"experiment_id": "exp3300", "summary": {"attack_success_rate": 0.25}}]},
    ) == pytest.approx(0.25)
    assert (
        mod._dataflip_gate_passed(
            {},
            {"gate_summary": {"garak_gate": {"dataflip_gate_passed": False}}},
        )
        is False
    )
    assert (
        mod._dataflip_gate_passed(
            {},
            {"rows": [{"experiment_id": "exp3300", "summary": {"dataflip_gate_passed": True}}]},
        )
        is True
    )
    assert mod._repair_headline_claim_allowed({}, {"repair_headline_claim_allowed": True}) is True
    assert (
        mod._repair_headline_claim_allowed(
            {},
            {"gate_summary": {"repair_headline": {"repair_headline_claim_allowed": True}}},
        )
        is True
    )
    assert mod._fr11_memory_replay_safe({}, {"fr11_replay_safe": True}) is True
    assert (
        mod._fr11_memory_replay_safe(
            {},
            {"gate_summary": {"fr11_replay": {"fr11_replay_safe": True}}},
        )
        is True
    )
