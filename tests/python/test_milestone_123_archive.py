"""Tests for the Exp 1601 `.122` archive and `.123` state artifact.

Spec: REQ-REPORT-066, SCENARIO-REPORT-066.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from carnot.reporting.milestone_123_archive import (
    EXPECTED_122_TASK_IDS,
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
        "exp1588-nsvif-dsl": "Exp 1588: Build bounded instruction-to-constraint DSL",
        "exp1589-dsl-sota-validation": "Exp 1589: Evaluate zero-false-accept on mandated SOTA GGUF",
        "exp1590-csr-mask": "Exp 1590: STATIC-style CSR-mask prototype for DSL",
        "exp1591-dccd-adapter": "Exp 1591: Upgrade DCCD smoke to reusable structured verdict adapter",
        "exp1592-dccd-repair-sota": "Exp 1592: Run DCCD repair on FoVer cases using SOTA",
        "exp1593-cdg-repair": "Exp 1593: Compare CDG-guided repair against flat validator ordering",
        "exp1594-cerce-ledger": "Exp 1594: Add CerCE-style certificate ledger around FR-11",
        "exp1595-cerce-bounds": "Exp 1595: Pre/post constraint violation bounds check",
        "exp1596-fr11-v16": "Exp 1596: Run FR-11 v16 skill-promotion with CerCE gate",
        "exp1597-inertial-ising": "Exp 1597: CPU-only inertial-update Ising ablation",
        "exp1598-z1-drift": "Exp 1598: Convert Z1 drift simulation into SamplerBackend test",
        "exp1599-kanele-audit": "Exp 1599: KANELE LUT-based evaluation audit for QuantKAN",
        "exp1600-ot-rewrite": "Exp 1600: Paper v6 OT terminology rewrite",
    }
    deliverables = {
        "exp1588-nsvif-dsl": "results/experiment_1588_nsvif_dsl.json",
        "exp1589-dsl-sota-validation": "results/experiment_1589_dsl_sota_validation.json",
        "exp1590-csr-mask": "results/experiment_1590_csr_mask.json",
        "exp1591-dccd-adapter": "results/experiment_1591_dccd_adapter.json",
        "exp1592-dccd-repair-sota": "results/experiment_1592_dccd_repair_sota.json",
        "exp1593-cdg-repair": "results/experiment_1593_cdg_repair.json",
        "exp1594-cerce-ledger": "results/experiment_1594_cerce_ledger.json",
        "exp1595-cerce-bounds": "results/experiment_1595_cerce_bounds.json",
        "exp1596-fr11-v16": "results/experiment_1596_fr11_v16.json",
        "exp1597-inertial-ising": "results/experiment_1597_inertial_ising.json",
        "exp1598-z1-drift": "results/experiment_1598_z1_drift.json",
        "exp1599-kanele-audit": "results/experiment_1599_kanele_audit.json",
        "exp1600-ot-rewrite": "results/experiment_1600_ot_rewrite.json",
    }
    tasks = [
        {
            "id": task_id,
            "title": titles[task_id],
            "deliverable": deliverables[task_id],
            "result": "OK (conductor)",
        }
        for task_id in EXPECTED_122_TASK_IDS
    ]
    return {
        "milestones": [
            {
                "id": "2026.05.122",
                "title": "Phase-1 Ship Readiness + Constraint Extraction + CerCE FR-11",
                "doc": "openspec/change-proposals/research-roadmap-vNEXT.md",
                "completed": "2026-05-09",
                "finding": "See conductor log for per-experiment results.",
                "tasks": tasks,
            }
        ]
    }


def _active_123_roadmap() -> dict[str, object]:
    return {
        "milestone": "2026.05.123",
        "milestone_title": "Phase-2 Formal KANs, EBCN Latent Reasoning, and CerCE Scale",
        "tasks": [
            {"id": "exp1601-archive-122", "deliverable": "results/experiment_1601_archive.json"},
            {"id": "exp1602-rkan-prototype"},
            {"id": "exp1603-ebcn-scorer"},
        ],
    }


def _roadmap_doc() -> str:
    return """
Milestone .122 proved bounded instruction-to-constraint DSL, DCCD adapter reuse,
STATIC-style CSR masks, and CerCE-style certificate ledger behavior.
Milestone .123 now covers Exact-Rational KAN, RKAN to Z3, Energy-Based Constraint
Network EBCN scoring, latent gradient editing, Sparse KAN clustering, DCCD,
DSL extraction, hardware RKAN accounting, and FR-11 CerCE scale.
"""


def test_scenario_report_066_archives_122_and_activates_123() -> None:
    """SCENARIO-REPORT-066: .122 archive and .123 state fields are complete."""

    deliverable_exists = {
        f"results/experiment_1588_nsvif_dsl.json": True,
        f"results/experiment_1589_dsl_sota_validation.json": True,
        f"results/experiment_1590_csr_mask.json": True,
        f"results/experiment_1591_dccd_adapter.json": True,
        f"results/experiment_1592_dccd_repair_sota.json": True,
        f"results/experiment_1593_cdg_repair.json": True,
        f"results/experiment_1594_cerce_ledger.json": True,
        f"results/experiment_1595_cerce_bounds.json": True,
        f"results/experiment_1596_fr11_v16.json": True,
        f"results/experiment_1597_inertial_ising.json": True,
        f"results/experiment_1598_z1_drift.json": True,
        f"results/experiment_1599_kanele_audit.json": True,
        f"results/experiment_1600_ot_rewrite.json": False,
    }
    artifact = build_artifact(
        research_complete=_research_complete_payload(),
        active_roadmap=_active_123_roadmap(),
        roadmap_doc_text=_roadmap_doc(),
        changelog_text="## 2026-05-09 (Milestone 2026.05.122 Operational Retrospective)",
        deliverable_exists=deliverable_exists,
        protected_files_unchanged=True,
    )

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["predecessor_archived"] is True
    assert artifact["predecessor_task_count"] == 13
    assert artifact["predecessor_task_ids"] == EXPECTED_122_TASK_IDS
    assert artifact["predecessor_tasks_terminal"] is True
    assert artifact["active_roadmap_milestone"] == "2026.05.123"
    assert artifact["active_roadmap_task_count"] == 3
    assert artifact["first_active_task_id"] == "exp1601-archive-122"
    assert artifact["status_moved_to_changelog"] is True
    assert artifact["setup_123_state"] is True
    assert artifact["formal_kan_track_ready"] is True
    assert artifact["ebcn_latent_track_ready"] is True
    assert artifact["cerce_scale_track_ready"] is True
    assert artifact["dccd_dsl_scale_track_ready"] is True
    assert artifact["hardware_accounting_track_ready"] is True
    assert artifact["missing_task_deliverables"] == [
        {
            "task_id": "exp1600-ot-rewrite",
            "deliverable": "results/experiment_1600_ot_rewrite.json",
        }
    ]
    assert artifact["research_roadmap_yaml_modified"] is False
    assert artifact["scripts_research_conductor_modified"] is False
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_report_066_blocks_missing_archive_state() -> None:
    """REQ-REPORT-066: missing archive, changelog, or roadmap state blocks completion."""

    artifact = build_artifact(
        research_complete={"milestones": []},
        active_roadmap={"milestone": "2026.05.122", "tasks": []},
        roadmap_doc_text="",
        changelog_text="",
        deliverable_exists={},
        protected_files_unchanged=False,
    )

    assert artifact["status"] == "blocked"
    assert artifact["predecessor_archived"] is False
    assert artifact["predecessor_tasks_terminal"] is False
    assert artifact["status_moved_to_changelog"] is False
    assert artifact["setup_123_state"] is False
    assert "research-complete.yaml has no 2026.05.122 archive" in artifact["blocked_reasons"]
    assert "active roadmap is not 2026.05.123" in artifact["blocked_reasons"]
    assert "ops/changelog.md lacks the 2026.05.122 status entry" in artifact["blocked_reasons"]
    assert "protected files changed" in artifact["blocked_reasons"]
    assert artifact["honest_verdict"].startswith("blocked:")


def test_req_report_066_blocks_partial_archive_and_incomplete_123_tracks() -> None:
    """REQ-REPORT-066: malformed archives and incomplete .123 setup stay blocked."""

    archive = _research_complete_payload()
    tasks = archive["milestones"][0]["tasks"]  # type: ignore[index]
    tasks.pop()  # type: ignore[union-attr]
    tasks[0]["result"] = "running"  # type: ignore[index]
    artifact = build_artifact(
        research_complete=archive,
        active_roadmap={
            "milestone": "2026.05.123",
            "tasks": [{"id": "exp1602-rkan-prototype"}],
        },
        roadmap_doc_text="Exact-Rational KAN only",
        changelog_text="Milestone 2026.05.122",
        deliverable_exists={},
        protected_files_unchanged=True,
    )

    assert artifact["status"] == "blocked"
    assert (
        "2026.05.122 archive task ids do not match exp1588-exp1600" in artifact["blocked_reasons"]
    )
    assert "not every 2026.05.122 archive task has a terminal result" in artifact["blocked_reasons"]
    assert "active roadmap does not start with exp1601-archive-122" in artifact["blocked_reasons"]
    assert (
        "2026.05.123 roadmap state lacks one or more expected tracks" in artifact["blocked_reasons"]
    )


def test_req_report_066_run_writes_bootstrap_and_terminal_json(tmp_path: Path) -> None:
    """REQ-REPORT-066: run writes in-progress first, then terminal archive JSON."""

    out_path = tmp_path / "results" / "experiment_1601_archive.json"
    bootstrap = write_in_progress_artifact(out_path)
    assert bootstrap["status"] == "in_progress"
    assert json.loads(out_path.read_text(encoding="utf-8"))["status"] == "in_progress"

    _write_json(tmp_path / "research-complete.yaml", _research_complete_payload())
    _write_json(tmp_path / "research-roadmap.yaml", _active_123_roadmap())
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops" / "changelog.md").write_text(
        "Milestone 2026.05.122 operational retrospective complete.", encoding="utf-8"
    )
    (tmp_path / "openspec" / "change-proposals").mkdir(parents=True)
    (tmp_path / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md").write_text(
        _roadmap_doc(), encoding="utf-8"
    )
    for task in _research_complete_payload()["milestones"][0]["tasks"]:  # type: ignore[index]
        deliverable = task["deliverable"]  # type: ignore[index]
        if deliverable != "results/experiment_1600_ot_rewrite.json":
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
    assert written["missing_task_deliverables"][0]["task_id"] == "exp1600-ot-rewrite"


def test_req_report_066_helpers_are_deterministic(tmp_path: Path, monkeypatch) -> None:
    """REQ-REPORT-066: helper functions keep missing inputs and git checks explicit."""

    assert _load_yaml(tmp_path / "missing.yaml") == {}
    assert _read_text(tmp_path / "missing.md") == ""
    assert _relative_path(tmp_path / "results" / "artifact.json") == "results/artifact.json"
    assert _relative_path(tmp_path / "research-roadmap.yaml") == "research-roadmap.yaml"

    monkeypatch.setattr(
        "carnot.reporting.milestone_123_archive.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    assert _protected_files_clean(tmp_path) is True
    monkeypatch.setattr(
        "carnot.reporting.milestone_123_archive.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1),
    )
    assert _protected_files_clean(tmp_path) is False
