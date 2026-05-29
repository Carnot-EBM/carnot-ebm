"""Tests for Exp 3337 archive .308 and activate .309 handoff.

Spec refs: REQ-REPORT-3337, SCENARIO-REPORT-3337.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pytest
import yaml

from carnot.reporting import archive_v308_activate_v309_3337 as mod


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


def _research_complete_yaml(*, archived: bool = True) -> str:
    if not archived:
        return "milestones: []\n"
    lines = [
        "milestones:",
        "- id: 2026.05.308",
        "  title: Phase-3 Path Recovery, Verifier Grounding, and FR-11 Nonforgetting",
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


def _roadmap_yaml(milestone: str = "2026.05.309", doc: str = mod.VNEXT_DOC) -> str:
    return (
        f'milestone: "{milestone}"\n'
        'milestone_title: "Runtime-Proven Energy Descent"\n'
        f'milestone_doc: "{doc}"\n'
        "tasks:\n"
        '  - id: "exp3337-archive-v308-activate-v309"\n'
        f'    milestone: "{milestone}"\n'
        '    deliverable: "results/experiment_3337_archive_v308_activate_v309.json"\n'
        '  - id: "exp3338-sota-gguf-tokenizer-runtime-receipt-v1"\n'
        f'    milestone: "{milestone}"\n'
    )


def _conductor_log() -> str:
    statuses = {
        "exp3325-archive-v307-activate-v308": "FAIL",
        "exp3326-phase3-path-preflight-manifest-v1": "FAIL",
        "exp3327-energy-descent-substrate-bootstrap-v1": "FAIL",
        "exp3328-energy-descent-vs-ar-sota-panel-v2": "GATE_BLOCK",
        "exp3330-verifier-diversity-remediation-plan-v1": "FAIL",
        "exp3335-reproducer-pack-and-evidence-matrix-v39": "FAIL",
        "exp3336-capstone-v308": "FAIL",
    }
    details = {
        "FAIL": "Codex CLI error: invalid request or missing usable artifact",
        "GATE_BLOCK": "Pre-emptive skip: upstream retired",
        "OK": "81 passed in 3.33s",
    }
    lines = ["| 2026-05-29 08:47 UTC | Milestone 2026.05.308 activated | OK | 12 tasks queued |"]
    for index, task in enumerate(mod.PRIOR_TASKS, start=49):
        status = statuses.get(str(task["id"]), "OK")
        lines.append(
            f"| 2026-05-29 09:{index % 60:02d} UTC | {task['log_title']} | "
            f"{status} | {details[status]} |"
        )
    lines.append("| 2026-05-29 11:29 UTC | Milestone 2026.05.309 activated | OK | 13 tasks queued |")
    return "\n".join(lines) + "\n"


def _write_sources(root: Path, *, archived: bool = True) -> None:
    _write_text(root, mod.RESEARCH_COMPLETE_REL_PATH, _research_complete_yaml(archived=archived))
    _write_text(root, mod.ACTIVE_ROADMAP_REL_PATH, _roadmap_yaml())
    _write_text(
        root,
        mod.VNEXT_PROPOSAL_REL_PATH,
        "# Research Roadmap vNEXT: Milestone 2026.05.309\n\n"
        "The first gap is SOTA GGUF tokenizer/runtime recovery.\n",
    )
    _write_text(root, mod.CONDUCTOR_LOG_REL_PATH, _conductor_log())
    _write_json(
        root,
        mod.OPERATIONAL_RETRO_REL_PATH,
        {
            "milestone": "2026.05.308",
            "total_wall_time_minutes": 0,
            "experiments_completed": 0,
            "compute_bound_experiments_count": 0,
            "slowest_experiments": [],
            "summary": "No completed timing rows.",
        },
    )
    _write_json(
        root,
        Path("results/experiment_3327_energy_descent_substrate_bootstrap_v1.json"),
        {
            "experiment": 3327,
            "status": "blocked",
            "honest_verdict": "blocked_gpu_setup",
            "blocked_reasons": ["gpu_setup_failed: tokenizer requires sentencepiece or tiktoken"],
            "duration_s": 4.035,
            "inference_substrate": "live_llm_inference",
            "random_seed": 3327,
            "reproducibility_checksum": "3327",
        },
    )
    _write_json(
        root,
        Path("results/experiment_3329_verifier_ensemble_diversity_audit_v2.json"),
        {
            "experiment": 3329,
            "status": "success",
            "honest_verdict": "usable for Phase-3 authority",
            "verifier_diversity_audit_v2_ready": True,
            "n_cases": 1000,
            "effective_k": 4.66196577560932,
            "lambda_min_sigma": 0.0179188149916219,
            "collapsed_pairs": [{"pair": ["exact", "symbolic"], "agreement": 0.964}],
            "duration_s": 0.0006728172302246094,
            "random_seed": 3329,
            "reproducibility_checksum": "3329",
        },
    )
    _write_json(
        root,
        Path("results/experiment_3331_ebt_sidecar_adapter_smoke_v2.json"),
        {
            "honest_verdict": "sidecar_ready",
            "adapter_ready": True,
            "claim_boundary": "sidecar_diagnostic_only",
            "n_cases": 2,
            "duration_s": 0.0004146099090576172,
            "random_seed": 3331,
            "reproducibility_checksum": "3331",
        },
    )
    _write_json(
        root,
        Path("results/experiment_3332_interwhen_monitor_pilot_v1.json"),
        {
            "honest_verdict": "monitor_pilot_provides_useful_trajectory_signal",
            "monitor_pilot_ready": True,
            "n_cases": 2,
            "duration_s": 0.00015163421630859375,
            "random_seed": 3332,
            "reproducibility_checksum": "3332",
        },
    )
    _write_json(
        root,
        Path("results/experiment_3333_energy_guided_ttscaling_sota_ablation_v1.json"),
        {
            "experiment": 3333,
            "status": "success",
            "honest_verdict": "ttscaling_evaluated",
            "ttscaling_ablation_ready": True,
            "flagged_adversarial": True,
            "corrigendum_pending": [
                {"kind": "DURATION_TOO_SHORT", "severity": "critical", "detail": "duration_s=43.882"}
            ],
            "n_cases": 2,
            "duration_s": 43.882,
            "random_seed": 3333,
            "reproducibility_checksum": "3333",
        },
    )
    _write_json(
        root,
        Path("results/experiment_3334_fr11_online_verifier_memory_nonforgetting_v4.json"),
        {
            "honest_verdict": "complete: online verifier memory nonforgetting confirmed",
            "fr11_nonforgetting_ready": True,
            "new_task_delta": 0.05,
            "old_task_delta": -0.02,
            "rollback_count": 2,
            "duration_s": 0.0003063678741455078,
            "random_seed": 3334,
            "reproducibility_checksum": "3334",
        },
    )


def _ids(rows: list[Mapping[str, Any]]) -> list[str]:
    return [str(row["task_id"]) for row in rows]


def test_req_report_3337_spec_anchor_declares_archive_schema() -> None:
    """REQ-REPORT-3337: OpenSpec declares the .308/.309 archive first."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-3337" in spec
    assert "SCENARIO-REPORT-3337" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_report_3337_classifies_v308_and_writes_activation(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3337: .308 evidence is archived without inflating claims."""

    _write_sources(tmp_path)
    before_complete = (tmp_path / mod.RESEARCH_COMPLETE_REL_PATH).read_text(encoding="utf-8")
    before_roadmap = (tmp_path / mod.ACTIVE_ROADMAP_REL_PATH).read_text(encoding="utf-8")
    before_log = (tmp_path / mod.CONDUCTOR_LOG_REL_PATH).read_text(encoding="utf-8")

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=3.5)
    second = mod.build_artifact(tmp_path, started_s=10.0, now_s=11.0)
    output = mod.write_artifact(tmp_path, started_s=4.0, now_s=5.25)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert mod.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3337"
    assert artifact["archived_milestone"] == "2026.05.308"
    assert artifact["activated_milestone"] == "2026.05.309"
    assert artifact["archive_v308_activate_v309_ready"] is True
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert _ids(artifact["completed_artifacts"]) == [
        "exp3329-verifier-ensemble-diversity-audit-v2",
        "exp3331-ebt-sidecar-adapter-smoke-v2",
        "exp3332-interwhen-monitor-pilot-v1",
        "exp3333-energy-guided-ttscaling-sota-ablation-v1",
        "exp3334-fr11-online-verifier-memory-nonforgetting-v4",
    ]
    assert _ids(artifact["blocked_artifacts"]) == ["exp3327-energy-descent-substrate-bootstrap-v1"]
    assert _ids(artifact["gate_blocked_artifacts"]) == [
        "exp3328-energy-descent-vs-ar-sota-panel-v2"
    ]
    assert _ids(artifact["missing_artifacts"]) == [
        "exp3325-archive-v307-activate-v308",
        "exp3326-phase3-path-preflight-manifest-v1",
        "exp3330-verifier-diversity-remediation-plan-v1",
        "exp3335-reproducer-pack-and-evidence-matrix-v39",
        "exp3336-capstone-v308",
    ]
    assert _ids(artifact["duration_flagged_artifacts"]) == [
        "exp3333-energy-guided-ttscaling-sota-ablation-v1"
    ]
    assert "DURATION_TOO_SHORT" in artifact["duration_flagged_artifacts"][0]["duration_flags"][0]
    assert artifact["roadmap_validation"]["active"]["milestone"] == "2026.05.309"
    assert artifact["roadmap_validation"]["active"]["points_to_vnext"] is True
    assert artifact["roadmap_validation"]["staged"]["exists"] is False
    assert artifact["roadmap_validation"]["activated_milestone_confirmed"] is True
    assert artifact["research_complete_update"] == {
        "path": "research-complete.yaml",
        "appended": False,
        "already_present": True,
    }
    assert artifact["research_complete_source_summary"]["task_count"] == len(mod.PRIOR_TASKS)
    assert artifact["conductor_log_terminal_status_counts"] == {
        "FAIL": 6,
        "GATE_BLOCK": 1,
        "OK": 5,
    }
    assert artifact["source_checksums"][
        "results/experiment_3327_energy_descent_substrate_bootstrap_v1.json"
    ] == _sha256(tmp_path / "results/experiment_3327_energy_descent_substrate_bootstrap_v1.json")
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["reproducibility_checksum"] == second["reproducibility_checksum"]
    assert saved["duration_s"] == pytest.approx(1.25)
    assert saved["honest_verdict"].startswith("complete:")
    assert "blocked=1" in saved["honest_verdict"]
    assert "missing=5" in saved["honest_verdict"]
    assert (tmp_path / mod.RESEARCH_COMPLETE_REL_PATH).read_text(encoding="utf-8") == before_complete
    assert (tmp_path / mod.ACTIVE_ROADMAP_REL_PATH).read_text(encoding="utf-8") == before_roadmap
    assert (tmp_path / mod.CONDUCTOR_LOG_REL_PATH).read_text(encoding="utf-8") == before_log
    assert yaml.safe_load((tmp_path / mod.ACTIVE_ROADMAP_REL_PATH).read_text(encoding="utf-8"))
    mod.validate_artifact(artifact)


def test_req_report_3337_appends_missing_archive_once(tmp_path: Path) -> None:
    """REQ-REPORT-3337: missing .308 archive is materialized exactly once."""

    _write_sources(tmp_path, archived=False)

    output = mod.write_artifact(tmp_path, started_s=2.0, now_s=3.0)
    saved = json.loads(output.read_text(encoding="utf-8"))
    archive_text = (tmp_path / mod.RESEARCH_COMPLETE_REL_PATH).read_text(encoding="utf-8")
    ensure_result = mod.ensure_research_complete_entry(tmp_path)

    assert saved["archive_v308_activate_v309_ready"] is True
    assert saved["research_complete_update"]["appended"] is True
    assert saved["research_complete_update"]["already_present"] is False
    assert archive_text.count("- id: 2026.05.308") == 1
    assert ensure_result == {
        "path": "research-complete.yaml",
        "appended": False,
        "already_present": True,
    }


def test_req_report_3337_fail_closed_helpers_and_validation(tmp_path: Path) -> None:
    """REQ-REPORT-3337: malformed inputs stay explicit and non-fabricated."""

    _write_sources(tmp_path, archived=False)
    _write_text(tmp_path, mod.ACTIVE_ROADMAP_REL_PATH, "tasks: [")
    _write_text(tmp_path, mod.RESEARCH_COMPLETE_REL_PATH, "[")
    _write_text(tmp_path, mod.CONDUCTOR_LOG_REL_PATH, "| malformed |\n")
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("tasks: [", encoding="utf-8")

    artifact = mod.build_artifact(tmp_path, started_s=8.0, now_s=3.0)

    assert artifact["archive_v308_activate_v309_ready"] is False
    assert artifact["duration_s"] == 0.0
    assert artifact["roadmap_validation"]["active"]["yaml_parse_ok"] is False
    assert "active roadmap milestone is not 2026.05.309" in artifact["blocked_reasons"]
    assert "active roadmap does not point to openspec/change-proposals/research-roadmap-vNEXT.md" in (
        artifact["blocked_reasons"]
    )
    assert "research-complete.yaml does not contain the .308 task summary" in (
        artifact["blocked_reasons"]
    )
    assert artifact["honest_verdict"].startswith("complete:")

    assert mod.read_json_object(bad_json) == {}
    assert mod.read_json_object(list_json) == {}
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.read_yaml_document(bad_yaml) == {}
    assert mod.read_yaml_document(tmp_path / "missing.yaml") == {}
    assert mod.sha256_file(tmp_path / "missing.json") is None
    assert mod._as_mapping([]) == {}
    assert mod._as_list({}) == []
    assert mod._task_ids({"tasks": ["bad", {"id": "x"}]}) == ["x"]
    assert mod._milestone_entries("bad") == []
    assert mod._milestone_entries([{"id": "x"}, "bad"]) == [{"id": "x"}]
    assert mod._terminal_prefix_ok("shipped: done") is True
    assert mod._terminal_prefix_ok("blocked") is False
    assert mod._parse_conductor_line("not a conductor row") == {}
    assert mod._duration_flag_reasons({"corrigendum_pending": "bad"}) == []
    assert mod._duration_flag_reasons({"corrigendum_pending": ["bad"], "flagged_adversarial": True}) == [
        "flagged_adversarial=true"
    ]
    assert mod._file_contains(tmp_path / "missing.log", "needle") is False
    assert all(
        row["status"] == "missing"
        for row in mod._conductor_log_terminal_rows(tmp_path / "missing-log-root")
    )

    empty_archive = tmp_path / "empty" / mod.RESEARCH_COMPLETE_REL_PATH
    mod._append_research_complete_entry(empty_archive)
    assert empty_archive.read_text(encoding="utf-8").startswith("milestones:\n- id: 2026.05.308")
    list_archive = tmp_path / "list" / mod.RESEARCH_COMPLETE_REL_PATH
    list_archive.parent.mkdir(parents=True)
    list_archive.write_text("milestones: []\n", encoding="utf-8")
    mod._append_research_complete_entry(list_archive)
    assert list_archive.read_text(encoding="utf-8").count("- id: 2026.05.308") == 1
    no_newline_archive = tmp_path / "no-newline" / mod.RESEARCH_COMPLETE_REL_PATH
    no_newline_archive.parent.mkdir(parents=True)
    no_newline_archive.write_text("milestones:\n- id: 2026.05.307\n  tasks: []", encoding="utf-8")
    mod._append_research_complete_entry(no_newline_archive)
    assert no_newline_archive.read_text(encoding="utf-8").count("- id: 2026.05.308") == 1
    summary_root = tmp_path / "summary"
    _write_text(
        summary_root,
        mod.RESEARCH_COMPLETE_REL_PATH,
        "milestones:\n- id: 2026.05.307\n  tasks: []\n- id: 2026.05.308\n  tasks: []\n",
    )
    assert mod._research_complete_task_summary(summary_root)["task_count"] == 0

    good_artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.0)
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({})
    with pytest.raises(ValueError, match="experiment_id"):
        mod.validate_artifact(good_artifact | {"experiment_id": "bad"})
    with pytest.raises(ValueError, match="archived_milestone"):
        mod.validate_artifact(good_artifact | {"archived_milestone": "bad"})
    with pytest.raises(ValueError, match="activated_milestone"):
        mod.validate_artifact(good_artifact | {"activated_milestone": "bad"})
    with pytest.raises(ValueError, match="random_seed"):
        mod.validate_artifact(good_artifact | {"random_seed": 0})
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(good_artifact | {"inference_substrate": "live"})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(good_artifact | {"honest_verdict": "blocked"})
    with pytest.raises(ValueError, match="no_push"):
        mod.validate_artifact(good_artifact | {"no_push": False})
    with pytest.raises(ValueError, match="files_updated"):
        mod.validate_artifact(good_artifact | {"files_updated": []})
