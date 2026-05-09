"""Tests for the Exp 1614 `.123` archive and `.124` state artifact.

Spec: REQ-REPORT-067, SCENARIO-REPORT-067.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from carnot.reporting.milestone_124_archive import (
    EXPECTED_123_TASK_IDS,
    REQUIRED_ARTIFACT_FIELDS,
    _load_yaml,
    _protected_files_clean,
    _read_text,
    _relative_path,
    build_artifact,
    run,
    write_in_progress_artifact,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _research_complete_payload() -> dict[str, object]:
    titles = {
        "exp1601-archive-122": "Exp 1601: Archive .122 and initialize .123",
        "exp1602-rkan-prototype": "Exp 1602: Exact-Rational KAN (RKAN) prototype",
        "exp1603-ebcn-scorer": "Exp 1603: Energy-Based Constraint Network scorer",
        "exp1604-sparse-kan-clustering": "Exp 1604: Sparse KAN clustering",
        "exp1605-latent-gradient-repair": "Exp 1605: Latent gradient editing using EBCN",
        "exp1606-dccd-multihop": "Exp 1606: DCCD repair on multi-hop reasoning tasks",
        "exp1607-dsl-humaneval": "Exp 1607: Extract DSL constraints from HumanEval",
        "exp1608-fr11-cerce-scale": "Exp 1608: FR-11 CerCE scale",
        "exp1609-context-sensitive-induction": "Exp 1609: Context-sensitive induction",
        "exp1610-z3-formal-bridge": "Exp 1610: Connect RKAN to Z3",
        "exp1611-ebcn-sota-validation": "Exp 1611: EBCN validation on mandated SOTA",
        "exp1612-hardware-rkan-accounting": "Exp 1612: Hardware accounting for RKAN",
        "exp1613-retro-123": "Exp 1613: Milestone .123 Retro",
    }
    deliverables = {
        "exp1601-archive-122": "results/experiment_1601_archive.json",
        "exp1602-rkan-prototype": "results/experiment_1602_rkan.json",
        "exp1603-ebcn-scorer": "results/experiment_1603_ebcn.json",
        "exp1604-sparse-kan-clustering": "results/experiment_1604_sparse_kan.json",
        "exp1605-latent-gradient-repair": "results/experiment_1605_latent_repair.json",
        "exp1606-dccd-multihop": "results/experiment_1606_dccd_multihop.json",
        "exp1607-dsl-humaneval": "results/experiment_1607_dsl_humaneval.json",
        "exp1608-fr11-cerce-scale": "results/experiment_1608_fr11_cerce.json",
        "exp1609-context-sensitive-induction": "results/experiment_1609_context_induction.json",
        "exp1610-z3-formal-bridge": "results/experiment_1610_z3_bridge.json",
        "exp1611-ebcn-sota-validation": "results/experiment_1611_ebcn_sota.json",
        "exp1612-hardware-rkan-accounting": "results/experiment_1612_rkan_accounting.json",
        "exp1613-retro-123": "results/experiment_1613_retro.json",
    }
    tasks = [
        {
            "id": task_id,
            "title": titles[task_id],
            "deliverable": deliverables[task_id],
            "result": "OK (conductor)",
        }
        for task_id in EXPECTED_123_TASK_IDS
    ]
    return {
        "milestones": [
            {
                "id": "2026.05.123",
                "title": "Phase-2 Formal KANs, EBCN Latent Reasoning, and CerCE Scale",
                "doc": "openspec/change-proposals/research-roadmap-vNEXT.md",
                "completed": "2026-05-09",
                "finding": "See conductor log for per-experiment results.",
                "tasks": tasks,
            }
        ]
    }


def _active_124_roadmap() -> dict[str, object]:
    return {
        "milestone": "2026.05.124",
        "milestone_title": "Phase-2 Latent Space Navigation, KANELE RTL Synthesis, and Energy-Guided Decoding",
        "tasks": [
            {"id": "exp1614-archive-123", "deliverable": "results/experiment_1614_archive.json"},
            {"id": "exp1615-ets-decoding"},
            {"id": "exp1616-nabla-reasoner"},
            {"id": "exp1618-pwa-kan-abstraction"},
            {"id": "exp1621-kanele-lut-mapping"},
        ],
    }


def _roadmap_doc() -> str:
    return """
Milestone .123 completed Exact-Rational KAN, RKAN to Z3, Energy-Based Constraint
Network EBCN scoring, Sparse KAN clustering, DCCD, DSL extraction, FR-11 CerCE,
and hardware RKAN accounting.
Milestone .124 now covers Energy-Guided Test-Time Scaling, Nabla-Reasoner
continuous latent navigation, Piecewise Affine KAN abstraction, MILP
verification, KANELE LUT mapping, Verilog lint, Adaptive Energy Landscape
Reconfiguration, Task Allocation Router, and KV260 simulator-only scope.
"""


def test_scenario_report_067_archives_123_and_activates_124() -> None:
    """SCENARIO-REPORT-067: .123 archive and .124 state fields are complete."""

    deliverable_exists = {
        task["deliverable"]: True
        for task in _research_complete_payload()["milestones"][0]["tasks"]  # type: ignore[index]
    }
    deliverable_exists["results/experiment_1612_rkan_accounting.json"] = False
    artifact = build_artifact(
        research_complete=_research_complete_payload(),
        active_roadmap=_active_124_roadmap(),
        roadmap_doc_text=_roadmap_doc(),
        changelog_text="## 2026-05-09 (Milestone 2026.05.123 Operational Retrospective)",
        deliverable_exists=deliverable_exists,
        protected_files_unchanged=True,
    )

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["predecessor_archived"] is True
    assert artifact["predecessor_task_count"] == 13
    assert artifact["predecessor_task_ids"] == EXPECTED_123_TASK_IDS
    assert artifact["predecessor_tasks_terminal"] is True
    assert artifact["active_roadmap_milestone"] == "2026.05.124"
    assert artifact["active_roadmap_task_count"] == 5
    assert artifact["first_active_task_id"] == "exp1614-archive-123"
    assert artifact["status_moved_to_changelog"] is True
    assert artifact["setup_124_state"] is True
    assert artifact["latent_navigation_track_ready"] is True
    assert artifact["formal_milp_kan_track_ready"] is True
    assert artifact["kanele_rtl_track_ready"] is True
    assert artifact["self_learning_consolidation_track_ready"] is True
    assert artifact["missing_task_deliverables"] == [
        {
            "task_id": "exp1612-hardware-rkan-accounting",
            "deliverable": "results/experiment_1612_rkan_accounting.json",
        }
    ]
    assert artifact["research_roadmap_yaml_modified"] is False
    assert artifact["scripts_research_conductor_modified"] is False
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_report_067_blocks_missing_archive_state() -> None:
    """REQ-REPORT-067: missing archive, changelog, or roadmap state blocks completion."""

    artifact = build_artifact(
        research_complete={"milestones": []},
        active_roadmap={"milestone": "2026.05.123", "tasks": []},
        roadmap_doc_text="",
        changelog_text="",
        deliverable_exists={},
        protected_files_unchanged=False,
    )

    assert artifact["status"] == "blocked"
    assert artifact["predecessor_archived"] is False
    assert artifact["predecessor_tasks_terminal"] is False
    assert artifact["status_moved_to_changelog"] is False
    assert artifact["setup_124_state"] is False
    assert "research-complete.yaml has no 2026.05.123 archive" in artifact["blocked_reasons"]
    assert "active roadmap is not 2026.05.124" in artifact["blocked_reasons"]
    assert "ops/changelog.md lacks the 2026.05.123 status entry" in artifact["blocked_reasons"]
    assert "protected files changed" in artifact["blocked_reasons"]
    assert artifact["honest_verdict"].startswith("blocked:")


def test_req_report_067_blocks_partial_archive_and_incomplete_124_tracks() -> None:
    """REQ-REPORT-067: malformed archives and incomplete .124 setup stay blocked."""

    archive = _research_complete_payload()
    tasks = archive["milestones"][0]["tasks"]  # type: ignore[index]
    tasks.pop()  # type: ignore[union-attr]
    tasks[0]["result"] = "running"  # type: ignore[index]
    artifact = build_artifact(
        research_complete=archive,
        active_roadmap={
            "milestone": "2026.05.124",
            "tasks": [{"id": "exp1615-ets-decoding"}],
        },
        roadmap_doc_text="Energy-Guided Test-Time Scaling only",
        changelog_text="Milestone 2026.05.123",
        deliverable_exists={},
        protected_files_unchanged=True,
    )

    assert artifact["status"] == "blocked"
    assert (
        "2026.05.123 archive task ids do not match exp1601-exp1613"
        in artifact["blocked_reasons"]
    )
    assert "not every 2026.05.123 archive task has a terminal result" in artifact["blocked_reasons"]
    assert "active roadmap does not start with exp1614-archive-123" in artifact["blocked_reasons"]
    assert (
        "2026.05.124 roadmap state lacks one or more expected tracks" in artifact["blocked_reasons"]
    )


def test_req_report_067_run_writes_bootstrap_and_terminal_json(tmp_path: Path) -> None:
    """REQ-REPORT-067: run writes in-progress first, then terminal archive JSON."""

    out_path = tmp_path / "results" / "experiment_1614_archive.json"
    bootstrap = write_in_progress_artifact(out_path)
    assert bootstrap["status"] == "in_progress"
    assert json.loads(out_path.read_text(encoding="utf-8"))["status"] == "in_progress"

    _write_json(tmp_path / "research-complete.yaml", _research_complete_payload())
    _write_json(tmp_path / "research-roadmap.yaml", _active_124_roadmap())
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops" / "changelog.md").write_text(
        "Milestone 2026.05.123 operational retrospective complete.", encoding="utf-8"
    )
    (tmp_path / "openspec" / "change-proposals").mkdir(parents=True)
    (tmp_path / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md").write_text(
        _roadmap_doc(), encoding="utf-8"
    )
    for task in _research_complete_payload()["milestones"][0]["tasks"]:  # type: ignore[index]
        deliverable = task["deliverable"]  # type: ignore[index]
        if deliverable != "results/experiment_1612_rkan_accounting.json":
            _write_json(tmp_path / deliverable, {"status": "complete"})

    artifact = run(root=tmp_path, out_path=out_path, protected_files_unchanged=True)
    written = json.loads(out_path.read_text(encoding="utf-8"))

    assert artifact == written
    assert written["status"] == "complete"
    assert written["source_inputs_read"]["research-complete.yaml"]["exists"] is True
    assert written["source_inputs_read"]["research-roadmap.yaml"]["exists"] is True
    assert written["source_inputs_read"]["ops/changelog.md"]["exists"] is True
    assert (
        written["source_inputs_read"]["openspec/change-proposals/research-roadmap-vNEXT.md"][
            "exists"
        ]
        is True
    )
    assert written["missing_task_deliverables"][0]["task_id"] == "exp1612-hardware-rkan-accounting"


def test_req_report_067_helpers_are_deterministic(tmp_path: Path, monkeypatch) -> None:
    """REQ-REPORT-067: helper functions keep missing inputs and git checks explicit."""

    assert _load_yaml(tmp_path / "missing.yaml") == {}
    assert _read_text(tmp_path / "missing.md") == ""
    assert _relative_path(tmp_path / "results" / "artifact.json") == "results/artifact.json"
    assert _relative_path(tmp_path / "research-roadmap.yaml") == "research-roadmap.yaml"

    monkeypatch.setattr(
        "carnot.reporting.milestone_124_archive.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    assert _protected_files_clean(tmp_path) is True
    monkeypatch.setattr(
        "carnot.reporting.milestone_124_archive.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1),
    )
    assert _protected_files_clean(tmp_path) is False
