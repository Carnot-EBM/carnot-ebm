"""Tests for the Exp 2008 `.156` archive and `.157` activation artifact.

Spec: REQ-REPORT-2008, SCENARIO-REPORT-2008.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from carnot.reporting import milestone_156_archive_157_activation as activation
from carnot.reporting.milestone_156_archive_157_activation import (
    REQUIRED_ARTIFACT_FIELDS,
    SOURCE_FILES,
    _load_sources,
    _protected_files_clean,
    _read_json,
    _read_text,
    _relative_path,
    build_artifact,
    run,
    write_in_progress_artifact,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _source_payloads() -> dict[str, dict[str, object]]:
    payloads: dict[str, dict[str, object]] = {
        "exp1996": {
            "status": "complete",
            "success": True,
            "honest_verdict": "complete: nsvif_z3_zero_false_positives",
        },
        "exp1997": {"status": "blocked", "honest_verdict": "blocked_gate_check_failed"},
        "exp1998": {
            "status": "success",
            "success": True,
            "inference_mode": "live_gpu",
            "honest_verdict": "complete: live baselines established",
        },
        "exp1999": {
            "status": "complete",
            "honest_verdict": "ising_guided_fuzzing_implemented",
        },
        "exp2000": {
            "status": "complete",
            "honest_verdict": "implementation_complete_and_verified",
        },
        "exp2001": {"status": "blocked", "honest_verdict": "blocked_gate_check_failed"},
        "exp2002": {"status": "blocked", "honest_verdict": "blocked_gate_check_failed"},
        "exp2003": {"status": "blocked", "honest_verdict": "blocked_gate_check_failed"},
        "exp2004": {
            "status": "complete",
            "evaluation_results": {"mocked": True},
            "honest_verdict": "complete: mocked ebt reasoning trace evaluation",
        },
        "exp2005": {
            "status": "complete",
            "honest_verdict": "complete: adaptive KAEM spline topology updated",
        },
        "exp2006": {
            "status": "success",
            "artifacts_exist": True,
            "valid_schema_confirmed": True,
            "sota_models_utilized": True,
            "honest_verdict": "Audit complete: all .156 artifacts exist",
        },
        "exp2007": {
            "status": "complete",
            "milestone": "2026.05.156",
            "retro_complete": True,
            "completed_task_count": 7,
            "blocked_task_count": 4,
            "failed_task_count": 0,
            "criteria_met": 11,
            "criteria_total": 11,
            "blocked_experiments": [1997, 2001, 2002, 2003],
            "recommendations": [
                "Re-propose COLD Decoding with prior_failures naming exp1969 and exp533.",
                "Re-propose Tier 2 memory with exp1484, exp788, and exp926.",
                "Run EBT Transformer Reasoning Evaluation with real GGUF model inference.",
            ],
            "gate_contract_gap_note": "Verdict terminal-prefix discipline violated.",
            "honest_verdict": "complete: milestone_156_retro_filed",
        },
    }
    return payloads


def _research_complete_text() -> str:
    lines = [
        "- id: 2026.05.156",
        "  title: Robust Constraint Extraction, Advanced Neural Solvers, and Constraint Memory",
        "  tasks:",
    ]
    for exp_id, filename in SOURCE_FILES.items():
        lines.append(f"  - id: {exp_id}")
        lines.append(f"    deliverable: results/{filename}")
    return "\n".join(lines)


def _roadmap_text() -> str:
    return """
milestone: "2026.05.157"
milestone_title: "Formal Verification, EBM-CoT Latent Refinement, and Tier 3 Predictive Learning"
tasks:
  - id: exp2008-archive-156-activate-157
    deliverable: "results/experiment_2008_archive_156_activate_157.json"
  - id: exp2013-tier3-predictive-verification-fr11
    deliverable: "results/experiment_2013_tier3_predictive_verification_fr11.json"
"""


def _roadmap_doc_text() -> str:
    return """
# Research Roadmap: Milestone 2026.05.157
Phase 0: Archive `.156` artifacts and initialize `.157`.
Phase 1: Implement EBM-CoT latent thought calibration.
Phase 2: Implement PWA abstractions and MILP verification for KANs.
Phase 3: Tier 3 predictor and preemptive guided decoding.
"""


def _conductor_log_text() -> str:
    return """
| 2026-05-13 04:02 UTC | Exp 2007: Milestone .156 Retrospective | OK | 81 passed |
| 2026-05-13 04:14 UTC | Plan milestone 2026.05.157 | OK | 10 tasks proposed |
| 2026-05-13 04:16 UTC | Milestone 2026.05.157 activated | OK | 10 tasks queued |
"""


def test_scenario_report_2008_builds_complete_archive_activation_artifact() -> None:
    """SCENARIO-REPORT-2008: .156 archived state activates the .157 environment."""

    artifact = build_artifact(
        sources=_source_payloads(),
        missing_source_paths=[],
        research_complete_text=_research_complete_text(),
        roadmap_text=_roadmap_text(),
        roadmap_doc_text=_roadmap_doc_text(),
        conductor_log_text=_conductor_log_text(),
        protected_files_unchanged=True,
        tests_run=["targeted coverage pending"],
    )

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["success"] is True
    assert artifact["previous_milestone_artifacts_archived"] is True
    assert artifact["archive_move_required"] is False
    assert len(artifact["archive_artifacts"]) == len(SOURCE_FILES)
    assert artifact["missing_artifacts"] == []
    assert artifact["milestone_environment_ready"] is True
    assert artifact["roadmap_157_active"] is True
    assert artifact["conductor_activation_logged"] is True
    assert artifact["protected_files_unchanged"] is True
    assert artifact["predecessor_summary"]["criteria_met"] == 11
    assert artifact["predecessor_summary"]["blocked_task_count"] == 4
    assert artifact["handoff_requirements"]["terminal_prefix_required"] is True
    assert artifact["handoff_requirements"]["real_gguf_ebt_eval_required"] is True
    assert artifact["tests_run"] == ["targeted coverage pending"]
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_report_2008_blocks_when_archive_or_environment_is_incomplete() -> None:
    """REQ-REPORT-2008: missing archive evidence prevents terminal activation success."""

    sources = _source_payloads()
    sources.pop("exp2007")
    artifact = build_artifact(
        sources=sources,
        missing_source_paths=["results/experiment_2007_milestone_156_retro.json"],
        research_complete_text="- id: 2026.05.155",
        roadmap_text='milestone: "2026.05.156"',
        roadmap_doc_text="no active target",
        conductor_log_text="",
        protected_files_unchanged=False,
        tests_run=[],
    )

    assert artifact["status"] == "blocked"
    assert artifact["success"] is False
    assert artifact["previous_milestone_artifacts_archived"] is False
    assert artifact["archive_move_required"] is True
    assert artifact["milestone_environment_ready"] is False
    assert artifact["roadmap_157_active"] is False
    assert artifact["conductor_activation_logged"] is False
    assert artifact["protected_files_unchanged"] is False
    assert "missing predecessor artifacts" in artifact["blocked_reasons"]
    assert "research-complete.yaml does not archive .156" in artifact["blocked_reasons"]
    assert "protected files changed" in artifact["blocked_reasons"]
    assert artifact["honest_verdict"].startswith("blocked:")


def test_req_report_2008_run_writes_in_progress_then_terminal_json(tmp_path: Path) -> None:
    """REQ-REPORT-2008: run writes bootstrap and terminal JSON artifacts."""

    out_path = tmp_path / "results" / "experiment_2008_archive_156_activate_157.json"
    bootstrap = write_in_progress_artifact(out_path)
    assert bootstrap["status"] == "in_progress"
    assert json.loads(out_path.read_text(encoding="utf-8"))["status"] == "in_progress"

    for exp_id, filename in SOURCE_FILES.items():
        _write_json(tmp_path / "results" / filename, _source_payloads()[exp_id])
    (tmp_path / "research-complete.yaml").write_text(_research_complete_text(), encoding="utf-8")
    (tmp_path / "research-roadmap.yaml").write_text(_roadmap_text(), encoding="utf-8")
    (tmp_path / "openspec" / "change-proposals").mkdir(parents=True)
    (tmp_path / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md").write_text(
        _roadmap_doc_text(), encoding="utf-8"
    )
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops" / "conductor-log.md").write_text(_conductor_log_text(), encoding="utf-8")

    artifact = run(
        root=tmp_path,
        out_path=out_path,
        protected_files_unchanged=True,
        tests_run=["pytest pending"],
    )
    written = json.loads(out_path.read_text(encoding="utf-8"))

    assert artifact == written
    assert written["status"] == "complete"
    assert written["success"] is True
    assert written["tests_run"] == ["pytest pending"]
    assert written["source_inputs_read"]["research-complete.yaml"]["exists"] is True


def test_req_report_2008_helpers_preserve_missing_and_malformed_inputs(
    tmp_path: Path, monkeypatch
) -> None:
    """REQ-REPORT-2008: helper functions keep source state explicit."""

    assert _read_json(tmp_path / "missing.json") is None
    assert _read_text(tmp_path / "missing.md") == ""
    assert _relative_path(tmp_path / "results" / "artifact.json") == "results/artifact.json"
    assert _relative_path(tmp_path / "other.json") == "other.json"
    malformed = tmp_path / "bad.json"
    malformed.write_text("{", encoding="utf-8")
    assert _read_json(malformed) is None

    _write_json(tmp_path / "results" / SOURCE_FILES["exp1996"], _source_payloads()["exp1996"])
    loaded, missing = _load_sources(tmp_path / "results")
    assert loaded["exp1996"]["success"] is True
    assert f"results/{SOURCE_FILES['exp1997']}" in missing

    assert activation._status({"status": "Success"}) == "success"
    assert activation._status({}) == ""
    assert activation._retro_complete(_source_payloads()["exp2007"]) is True
    assert activation._retro_complete({"status": "complete", "milestone": "2026.05.155"}) is False

    monkeypatch.setattr(
        "carnot.reporting.milestone_156_archive_157_activation.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    assert _protected_files_clean(tmp_path) is True
