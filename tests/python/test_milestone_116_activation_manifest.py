"""Tests for the Exp 1506 `.116` activation manifest.

Spec: REQ-REPORT-054, SCENARIO-REPORT-054.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting.milestone_116_activation_manifest import (
    ALLOWED_116_TRACKS,
    GATED_116_TRACKS,
    MANDATED_SOTA_MODELS,
    REQUIRED_ARTIFACT_FIELDS,
    RETIRED_HEADLINE_SIGNALS,
    _read_json,
    _read_text,
    _line_decision_ready,
    _prior_thrml_parity_ready,
    _relative_path,
    _research_complete_has_115_entry,
    build_artifact,
    run,
    write_in_progress_artifact,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _retro_payload() -> dict[str, object]:
    return {
        "status": "complete",
        "milestone": "2026.04.115",
        "criteria_met": 12,
        "criteria_total": 12,
        "honest_verdict": "complete: milestone_115_12_of_12_criteria_met_claim_boundaries_preserved",
        "line_decisions": {
            "trigger_certificate_export": {
                "decision": "graduated",
                "evidence": "parse_rate=0.3, validation_rate=0.1, false_accept_rate=0.0",
            },
            "constrainprompt_validator_compiler": {
                "decision": "graduated",
                "evidence": "compile_rate=0.933333, false_accept_rate=0.0",
            },
            "interwhen_hover_monitoring": {
                "decision": "graduated",
                "evidence": "monitor_ready=True, safe_prefix_ready=True",
            },
            "fr11_trace2skill": {
                "decision": "graduated",
                "evidence": "skills_evaluated=24, promoted=12, unreachable=0",
            },
        },
        "protected_files": {
            "research-roadmap.yaml": "unchanged",
            "scripts/research_conductor.py": "unchanged",
        },
    }


def _exp1502_payload() -> dict[str, object]:
    return {
        "status": "complete",
        "kan_hardware_accounting_ready": True,
        "accounting_only_no_synthesis_claim": True,
        "hardware_claim_allowed": False,
        "blockers": [
            "no_vivado_synthesis_or_board_measurement_for_exp1502",
            "quantkan_and_kaem_proxy_shapes_must_be_normalized_before_any_future_synthesis",
        ],
        "honest_verdict": "complete: kan hardware accounting ready; no synthesis or hardware claim",
    }


def _exp1504_payload() -> dict[str, object]:
    return {
        "status": "complete",
        "parity_experiment_ran": True,
        "parity_pass_count": 2,
        "parity_fail_count": 0,
        "simulator_only": True,
        "hardware_claim_allowed": False,
        "metadata": {"tsu_hardware_execution": False},
        "honest_verdict": "complete_thrml_carnot_simulator_parity_passed_no_hardware_claim",
    }


def _conductor_log_text() -> str:
    return "\n".join(
        f"| 2026-05-07 21:00 UTC | exp{exp_id} milestone task | OK | 81 passed |"
        for exp_id in range(1492, 1506)
    )


def _research_complete_with_115() -> str:
    return """
- id: 2026.04.115
  title: Executable Constraint Monitors + FR-11 Self-Learning Hygiene + Hardware Gates
  completed: '2026-05-07'
"""


def _ops_status_text() -> str:
    return """
Milestone 2026.04.116 PLANNED after .115 completion.
Continuous self-learning requirement is satisfied by exp1512-fr11-verifier-feedback-policy-cache-v11.
Mandated local SOTA GGUF MODEL_SPECS: unsloth/Qwen3.6-35B-A3B-GGUF,
unsloth/gemma-4-31B-it-GGUF, and unsloth/gemma-4-26B-A4B-it-GGUF.
Structured gates include exp1508, exp1509, exp1513, exp1514, exp1515,
exp1516, and exp1517.
"""


def _ops_changelog_text() -> str:
    return """
Semantic Energy/logit telemetry and V_1 pairwise remain retired as headline
signals. Generated Python verifier code is not trusted unless compiled through
the safe DSL. THRML has no TSU hardware claim, KV260 has no board claim,
KAN has no synthesis claim, and legacy small models are smoke tests only.
"""


def _roadmap_text() -> str:
    return """
allowed tracks: safe-DSL verifier induction, trigger+grammar certificate decoding,
executable monitor runtime, plan-graph structural contracts, product-line solver oracle,
FR-11 verifier-feedback replay, trace2skill portable pack, THRML SamplerBackend
conformance, KAN shape normalization, and KV260 source-level RTL properties.
Structured gates: exp1515 on exp1506.prior_thrml_parity_ready == true; exp1516
on exp1506.prior_kan_shape_blocker_recorded == true; exp1517 on
exp1506.prior_kv260_source_track_active == true.
"""


def _roadmap_doc_text() -> str:
    return """
Semantic Energy/logit telemetry and V_1 pairwise self-verification remain retired
as headline signals. Decoded-quality claims from injected-failure localization
remain blocked. LLM-generated verifier code is not trusted directly; generated
verifiers must pass the safe DSL. THRML work remains simulator/software
conformance only. KAN/KAEM work remains accounting and shape-normalization only
with no synthesis or board claim. KV260 work remains source-level RTL/property
testing only with no board execution or bitstream claim. Legacy small models are
not headline evidence.
"""


def _hardware_text() -> str:
    return """
KV260/FPGA Discrete SB RTL lint and simulation is active source-level work.
No KV260 board, bitfile, or latency claim until Vivado synthesis, bitfile flashing,
and board commands are captured.
"""


def test_scenario_report_054_activates_116_and_archives_prior_evidence() -> None:
    """SCENARIO-REPORT-054: .116 activation archives .115 readiness gates."""

    artifact, manifest = build_artifact(
        retro=_retro_payload(),
        exp1502=_exp1502_payload(),
        exp1504=_exp1504_payload(),
        conductor_log_text=_conductor_log_text(),
        research_complete_text=_research_complete_with_115(),
        ops_status_text=_ops_status_text(),
        ops_changelog_text=_ops_changelog_text(),
        roadmap_text=_roadmap_text(),
        roadmap_doc_text=_roadmap_doc_text(),
        hardware_wishlist_text=_hardware_text(),
        architecture_text=_hardware_text(),
        manifest_path="ops/milestone_116_activation_manifest.md",
        protected_file_diffs=[],
    )

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["milestone"] == "2026.04.116"
    assert artifact["predecessor_milestone"] == "2026.04.115"
    assert artifact["predecessor_criteria_met"] == 12
    assert artifact["predecessor_criteria_total"] == 12
    assert artifact["activation_manifest_complete"] is True
    assert artifact["prior_trigger_certificates_ready"] is True
    assert artifact["prior_validator_compiler_ready"] is True
    assert artifact["prior_monitor_replay_ready"] is True
    assert artifact["prior_fr11_daily_eval_ready"] is True
    assert artifact["prior_thrml_parity_ready"] is True
    assert artifact["prior_kan_shape_blocker_recorded"] is True
    assert artifact["prior_kv260_source_track_active"] is True
    assert artifact["continuous_self_learning_required"] is True
    assert artifact["research_complete_has_115_entry"] is True
    assert artifact["mandated_sota_models"] == MANDATED_SOTA_MODELS
    assert artifact["retired_headline_signals"] == RETIRED_HEADLINE_SIGNALS
    assert [track["track"] for track in artifact["allowed_116_tracks"]] == [
        track["track"] for track in ALLOWED_116_TRACKS
    ]
    assert [track["track"] for track in artifact["gated_116_tracks"]] == [
        track["track"] for track in GATED_116_TRACKS
    ]
    assert artifact["conductor_log_exp1492_to_exp1505"]["missing_experiments"] == []
    assert artifact["protected_files_unchanged"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert "safe-DSL verifier induction" in manifest
    assert "THRML SamplerBackend conformance" in manifest
    assert "arbitrary generated-Python verifier trust" in manifest
    assert "KV260 board claims" in manifest
    assert "Prior Readiness" in manifest


def test_req_report_054_blocks_missing_archive_and_prior_readiness() -> None:
    """REQ-REPORT-054: incomplete predecessor evidence blocks completion."""

    artifact, manifest = build_artifact(
        retro={
            "status": "complete",
            "milestone": "2026.04.115",
            "criteria_met": 11,
            "criteria_total": 12,
        },
        exp1502={"status": "complete", "blockers": []},
        exp1504={"status": "complete", "simulator_only": False, "hardware_claim_allowed": True},
        conductor_log_text="exp1492 OK\n",
        research_complete_text="- id: 2026.04.114\n",
        ops_status_text="",
        ops_changelog_text="",
        roadmap_text="",
        roadmap_doc_text="",
        hardware_wishlist_text="",
        architecture_text="",
        manifest_path="ops/milestone_116_activation_manifest.md",
        protected_file_diffs=["research-roadmap.yaml"],
    )

    assert artifact["status"] == "blocked"
    assert artifact["activation_manifest_complete"] is False
    assert artifact["research_complete_has_115_entry"] is False
    assert artifact["prior_thrml_parity_ready"] is False
    assert artifact["prior_kan_shape_blocker_recorded"] is False
    assert artifact["prior_kv260_source_track_active"] is False
    assert "predecessor retro criteria not complete" in artifact["blocked_reasons"]
    assert "research-complete.yaml lacks 2026.04.115 archive row" in artifact["blocked_reasons"]
    assert "protected files changed" in artifact["blocked_reasons"]
    assert artifact["honest_verdict"].startswith("passed:")
    assert "Manifest blocked" in manifest


def test_req_report_054_run_writes_bootstrap_manifest_and_terminal_json(tmp_path: Path) -> None:
    """REQ-REPORT-054: run writes bootstrap, markdown, and terminal artifact."""

    out_path = tmp_path / "results" / "experiment_1506_115_completion_archive_116_activation.json"
    manifest_path = tmp_path / "ops" / "milestone_116_activation_manifest.md"

    bootstrap = write_in_progress_artifact(out_path)
    assert bootstrap["status"] == "in_progress"
    assert json.loads(out_path.read_text(encoding="utf-8"))["status"] == "in_progress"

    _write_json(tmp_path / "results" / "experiment_1505_milestone_115_retro.json", _retro_payload())
    _write_json(
        tmp_path / "results" / "experiment_1502_kan_hardware_accounting_quantkan_kaem.json",
        _exp1502_payload(),
    )
    _write_json(
        tmp_path / "results" / "experiment_1504_thrml_carnot_simulator_parity_v3.json",
        _exp1504_payload(),
    )
    (tmp_path / "ops").mkdir(exist_ok=True)
    (tmp_path / "ops" / "conductor-log.md").write_text(_conductor_log_text(), encoding="utf-8")
    (tmp_path / "ops" / "status.md").write_text(_ops_status_text(), encoding="utf-8")
    (tmp_path / "ops" / "changelog.md").write_text(_ops_changelog_text(), encoding="utf-8")
    (tmp_path / "research-roadmap.yaml").write_text(_roadmap_text(), encoding="utf-8")
    (tmp_path / "research-complete.yaml").write_text(
        _research_complete_with_115(),
        encoding="utf-8",
    )
    (tmp_path / "research-hardware-wishlist.md").write_text(_hardware_text(), encoding="utf-8")
    (tmp_path / "_bmad").mkdir(exist_ok=True)
    (tmp_path / "_bmad" / "architecture.md").write_text(_hardware_text(), encoding="utf-8")
    (tmp_path / "openspec" / "change-proposals").mkdir(parents=True, exist_ok=True)
    (tmp_path / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md").write_text(
        _roadmap_doc_text(),
        encoding="utf-8",
    )

    artifact = run(root=tmp_path, out_path=out_path, manifest_path=manifest_path)
    written = json.loads(out_path.read_text(encoding="utf-8"))
    manifest = manifest_path.read_text(encoding="utf-8")

    assert artifact == written
    assert written["status"] == "complete"
    assert written["manifest_path"] == "ops/milestone_116_activation_manifest.md"
    assert (
        written["source_inputs_read"]["results/experiment_1505_milestone_115_retro.json"]["exists"]
        is True
    )
    assert "Allowed .116 Tracks" in manifest
    assert "Gated .116 Tracks" in manifest


def test_req_report_054_defensive_helpers_and_missing_inputs(tmp_path: Path) -> None:
    """REQ-REPORT-054: helpers and missing inputs remain explicit."""

    assert _read_json(tmp_path / "missing.json") is None
    assert _read_text(tmp_path / "missing.md") == ""
    assert _relative_path(tmp_path / "ops" / "manifest.md") == "ops/manifest.md"
    assert _relative_path(tmp_path / "results" / "artifact.json") == "results/artifact.json"
    assert _relative_path(tmp_path / "loose.txt") == "loose.txt"
    assert _research_complete_has_115_entry("- id: 2026.04.115\n") is True
    assert _research_complete_has_115_entry('id: "2026.04.115"\n') is True
    assert _research_complete_has_115_entry("id: '2026.04.115'\n") is True
    assert _research_complete_has_115_entry("- id: 2026.04.114\n") is False
    assert _line_decision_ready({"line_decisions": []}, "x", "graduated") is False
    assert _line_decision_ready({"line_decisions": {"x": []}}, "x", "graduated") is False
    assert _prior_thrml_parity_ready({"metadata": [], "honest_verdict": ""}) is False

    artifact, manifest = build_artifact(
        retro={},
        exp1502={},
        exp1504={},
        conductor_log_text="",
        research_complete_text="",
        ops_status_text="",
        ops_changelog_text="",
        roadmap_text="",
        roadmap_doc_text="",
        hardware_wishlist_text="",
        architecture_text="",
        manifest_path="ops/milestone_116_activation_manifest.md",
        protected_file_diffs=[],
    )

    assert artifact["status"] == "blocked"
    assert artifact["conductor_log_exp1492_to_exp1505"]["ok_count"] == 0
    assert "prior trigger certificates not ready" in artifact["blocked_reasons"]
    assert "Manifest blocked" in manifest
