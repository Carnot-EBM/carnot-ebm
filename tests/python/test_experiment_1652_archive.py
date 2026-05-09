"""Tests for the Exp 1652 `.126` archive and `.127` initialization artifact.

Spec: REQ-REPORT-068, SCENARIO-REPORT-068.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts import experiment_1652_archive as exp


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _research_complete_payload() -> dict[str, object]:
    titles = {
        "exp1640-nsvif-dsl-parser": "Exp 1640: NSVIF instruction-to-constraint DSL implementation",
        "exp1641-nsvif-live-sota": "Exp 1641: Live SOTA validation of NSVIF zero-false-accepts",
        "exp1642-llguidance-adapter": "Exp 1642: llguidance adapter",
        "exp1643-static-csr-mask": "Exp 1643: STATIC-style CSR-mask prototype",
        "exp1644-cerce-ledger": "Exp 1644: CerCE-style certificate ledger",
        "exp1645-fr11-cerce-learning": "Exp 1645: FR-11 CerCE continuous self-learning loop",
        "exp1646-ebcn-prototype": "Exp 1646: Energy-Based Constraint Networks prototype",
        "exp1647-rkan-lean4-export": "Exp 1647: Exact-Rational KANs export for Lean 4",
        "exp1648-sparse-kan-clustering": "Exp 1648: Sparse KANs with spectral constraints",
        "exp1649-vivado-potts-synthesis": "Exp 1649: KV260 Vivado bitfile synthesis for q=3 Potts machine",
        "exp1650-kv260-potts-bringup": "Exp 1650: KV260 board bring-up for Potts hardware",
        "exp1651-milestone-retro": "Exp 1651: Milestone 126 Retrospective",
    }
    deliverables = {
        "exp1640-nsvif-dsl-parser": "results/experiment_1640_nsvif_dsl.json",
        "exp1641-nsvif-live-sota": "results/experiment_1641_nsvif_sota.json",
        "exp1642-llguidance-adapter": "results/experiment_1642_llguidance.json",
        "exp1643-static-csr-mask": "results/experiment_1643_static_csr.json",
        "exp1644-cerce-ledger": "results/experiment_1644_cerce_ledger.json",
        "exp1645-fr11-cerce-learning": "results/experiment_1645_fr11_cerce.json",
        "exp1646-ebcn-prototype": "results/experiment_1646_ebcn.json",
        "exp1647-rkan-lean4-export": "results/experiment_1647_rkan.json",
        "exp1648-sparse-kan-clustering": "results/experiment_1648_sparse_kan.json",
        "exp1649-vivado-potts-synthesis": "results/experiment_1649_vivado_synthesis.json",
        "exp1650-kv260-potts-bringup": "results/experiment_1650_kv260_bringup.json",
        "exp1651-milestone-retro": "results/experiment_1651_retro.json",
    }
    return {
        "milestones": [
            {
                "id": "2026.05.126",
                "title": "Phase-4 Structured Verdict Scaling, CerCE Continual Learning, and Formal KAN Verification",
                "doc": "openspec/change-proposals/research-roadmap-vNEXT.md",
                "completed": "2026-05-09",
                "finding": "See conductor log for per-experiment results.",
                "tasks": [
                    {
                        "id": task_id,
                        "title": titles[task_id],
                        "deliverable": deliverables[task_id],
                        "result": "OK (conductor)",
                    }
                    for task_id in exp.EXPECTED_126_TASK_IDS
                ],
            }
        ]
    }


def _active_127_roadmap() -> dict[str, object]:
    return {
        "milestone": "2026.05.127",
        "milestone_title": "Phase-5 EBRM Trace Scoring, SMGI Continuous Learning, and Energy-Guided Decoding",
        "tasks": [
            {"id": "exp1652-archive-126", "deliverable": "results/experiment_1652_archive.json"},
            {"id": "exp1653-nsvif-sota-integration"},
            {"id": "exp1654-energy-guided-decoding"},
            {"id": "exp1656-ebrm-trace-scorer"},
            {"id": "exp1659-smgi-certified-updates"},
        ],
    }


def _roadmap_doc() -> str:
    return """
Milestone 2026.05.126 successfully shipped NSVIF DSL, STATIC CSR Mask,
FR-11 CerCE Ledger, and KV260 Potts Synthesis.
Milestone 2026.05.127 now covers EBRM Trace Scoring, SMGI certified updates,
Energy-Guided Decoding, NSVIF SOTA integration, STATIC CSR masks, KV260 Potts
hardware offload, LTLZinc, and Pi-net.
"""


def _artifact_payloads() -> dict[str, dict[str, object]]:
    return {
        "results/experiment_1640_nsvif_dsl.json": {
            "status": "complete",
            "parser_success": True,
            "false_accept_rate": 0.0,
            "honest_verdict": "complete: NSVIF instruction DSL parsed into Carnot constraints",
        },
        "results/experiment_1641_nsvif_sota.json": {
            "status": "complete",
            "false_accepts": 0,
            "validation_rate": 1.0,
            "honest_verdict": "complete: NSVIF SOTA zero false accepts",
        },
        "results/experiment_1644_cerce_ledger.json": {
            "status": "complete",
            "cerce_ledger_ready": True,
            "ledger_implemented": True,
            "honest_verdict": "complete: cerce_ledger_added",
        },
        "results/experiment_1649_vivado_synthesis.json": {
            "vivado_available": False,
            "synthesis_success": False,
            "honest_verdict": "vivado_not_installed",
        },
        "results/experiment_1651_retro.json": {
            "status": "complete",
            "milestone": "2026.05.126",
            "criteria_met": 8,
            "criteria_total": 11,
            "honest_verdict": "milestone_126_retrospective_filed_8_of_11_complete",
        },
    }


def test_scenario_report_068_archives_126_and_initializes_127() -> None:
    """SCENARIO-REPORT-068: .126 archive and .127 initialization fields are complete."""

    deliverable_exists = {
        task["deliverable"]: True
        for task in _research_complete_payload()["milestones"][0]["tasks"]  # type: ignore[index]
    }
    deliverable_exists["results/experiment_1645_fr11_cerce.json"] = False
    deliverable_exists["results/experiment_1650_kv260_bringup.json"] = False

    artifact = exp.build_artifact(
        research_complete=_research_complete_payload(),
        active_roadmap=_active_127_roadmap(),
        roadmap_doc_text=_roadmap_doc(),
        changelog_text="Milestone 2026.05.126 operational retrospective complete.",
        artifact_payloads=_artifact_payloads(),
        deliverable_exists=deliverable_exists,
        protected_files_unchanged=True,
    )

    assert exp.REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["predecessor_archived"] is True
    assert artifact["predecessor_task_count"] == 12
    assert artifact["predecessor_task_ids"] == exp.EXPECTED_126_TASK_IDS
    assert artifact["predecessor_tasks_terminal"] is True
    assert artifact["active_roadmap_milestone"] == "2026.05.127"
    assert artifact["first_active_task_id"] == "exp1652-archive-126"
    assert artifact["status_moved_to_changelog"] is True
    assert artifact["setup_127_state"] is True
    assert artifact["nsvif_dsl_landed"] is True
    assert artifact["nsvif_zero_false_accepts"] is True
    assert artifact["cerce_ledger_landed"] is True
    assert artifact["kv260_potts_synthesis_landed"] is True
    assert artifact["kv260_potts_vivado_success"] is False
    assert artifact["hardware_execution_claimed"] is False
    assert artifact["missing_task_deliverables"] == [
        {
            "task_id": "exp1645-fr11-cerce-learning",
            "deliverable": "results/experiment_1645_fr11_cerce.json",
        },
        {
            "task_id": "exp1650-kv260-potts-bringup",
            "deliverable": "results/experiment_1650_kv260_bringup.json",
        },
    ]
    assert artifact["research_roadmap_yaml_modified"] is False
    assert artifact["scripts_research_conductor_modified"] is False
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_report_068_blocks_missing_archive_state() -> None:
    """REQ-REPORT-068: missing archive, changelog, or roadmap state blocks completion."""

    artifact = exp.build_artifact(
        research_complete={"milestones": []},
        active_roadmap={"milestone": "2026.05.126", "tasks": []},
        roadmap_doc_text="",
        changelog_text="",
        artifact_payloads={},
        deliverable_exists={},
        protected_files_unchanged=False,
    )

    assert artifact["status"] == "blocked"
    assert artifact["predecessor_archived"] is False
    assert artifact["predecessor_tasks_terminal"] is False
    assert artifact["setup_127_state"] is False
    assert artifact["nsvif_dsl_landed"] is False
    assert artifact["kv260_potts_synthesis_landed"] is False
    assert artifact["cerce_ledger_landed"] is False
    assert "research-complete.yaml has no 2026.05.126 archive" in artifact["blocked_reasons"]
    assert "active roadmap is not 2026.05.127" in artifact["blocked_reasons"]
    assert "ops/changelog.md lacks the 2026.05.126 status entry" in artifact["blocked_reasons"]
    assert "protected files changed" in artifact["blocked_reasons"]


def test_req_report_068_blocks_partial_archive_and_incomplete_tracks() -> None:
    """REQ-REPORT-068: malformed archives and incomplete .127 setup stay blocked."""

    archive = _research_complete_payload()
    tasks = archive["milestones"][0]["tasks"]  # type: ignore[index]
    tasks.pop()  # type: ignore[union-attr]
    tasks[0]["result"] = "running"  # type: ignore[index]

    artifact = exp.build_artifact(
        research_complete=archive,
        active_roadmap={
            "milestone": "2026.05.127",
            "tasks": [{"id": "exp1653-nsvif-sota-integration"}],
        },
        roadmap_doc_text="NSVIF DSL only",
        changelog_text="Milestone 2026.05.126",
        artifact_payloads={},
        deliverable_exists={},
        protected_files_unchanged=True,
    )

    assert artifact["status"] == "blocked"
    assert (
        "2026.05.126 archive task ids do not match exp1640-exp1651"
        in artifact["blocked_reasons"]
    )
    assert "not every 2026.05.126 archive task has a terminal result" in artifact["blocked_reasons"]
    assert "active roadmap does not start with exp1652-archive-126" in artifact["blocked_reasons"]
    assert "2026.05.127 roadmap state lacks one or more expected tracks" in artifact["blocked_reasons"]
    assert "NSVIF DSL landing evidence incomplete" in artifact["blocked_reasons"]


def test_req_report_068_run_writes_bootstrap_and_terminal_json(tmp_path: Path) -> None:
    """REQ-REPORT-068: run writes in-progress first, then terminal archive JSON."""

    output_path = tmp_path / "results" / "experiment_1652_archive.json"
    bootstrap = exp.write_in_progress_artifact(output_path)
    assert bootstrap["status"] == "in_progress"
    assert json.loads(output_path.read_text(encoding="utf-8"))["status"] == "in_progress"

    _write_json(tmp_path / "research-complete.yaml", _research_complete_payload())
    _write_json(tmp_path / "research-roadmap.yaml", _active_127_roadmap())
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops" / "changelog.md").write_text(
        "Milestone 2026.05.126 operational retrospective complete.", encoding="utf-8"
    )
    (tmp_path / "openspec" / "change-proposals").mkdir(parents=True)
    (tmp_path / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md").write_text(
        _roadmap_doc(), encoding="utf-8"
    )
    for deliverable, payload in _artifact_payloads().items():
        _write_json(tmp_path / deliverable, payload)

    artifact = exp.run(root=tmp_path, output_path=output_path, protected_files_unchanged=True)
    written = json.loads(output_path.read_text(encoding="utf-8"))

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
    assert written["missing_task_deliverable_count"] == 7


def test_req_report_068_helpers_are_deterministic(tmp_path: Path, monkeypatch) -> None:
    """REQ-REPORT-068: helper functions keep missing inputs and git checks explicit."""

    assert exp._load_yaml(tmp_path / "missing.yaml") == {}
    assert exp._read_json(tmp_path / "missing.json") == {}
    assert exp._read_text(tmp_path / "missing.md") == ""
    assert exp._relative_path(tmp_path / "results" / "artifact.json") == "results/artifact.json"
    assert exp._relative_path(tmp_path / "research-roadmap.yaml") == "research-roadmap.yaml"

    monkeypatch.setattr(
        "scripts.experiment_1652_archive.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    assert exp._protected_files_clean(tmp_path) is True
