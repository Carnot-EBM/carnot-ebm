"""Tests for the Exp 1479 `.114` activation manifest.

Spec: REQ-REPORT-050, SCENARIO-REPORT-050.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting.milestone_114_activation_manifest import (
    ALLOWED_114_TRACKS,
    FORBIDDEN_REOPEN_TRACKS,
    REQUIRED_ARTIFACT_FIELDS,
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


def _retro_payload() -> dict[str, object]:
    return {
        "status": "complete",
        "milestone": "2026.04.113",
        "criteria_met": 12,
        "criteria_total": 12,
        "honest_verdict": (
            "milestone_113_12_of_12_criteria_met_success_threshold_met_"
            "halt_spilled_retired_telemetry_headline_blocked"
        ),
        "carry_forward_tracks": [
            {
                "track": "live_sota_telemetry",
                "status": "raw_telemetry_ready",
                "topk_logprobs_available": True,
                "rule": "Preserve raw top-k telemetry, but do not make a headline signal claim.",
            },
            {
                "track": "self_learning",
                "status": "preserved",
                "self_learning_delta_overall": 12,
                "soundness_mistakes": 0,
                "completeness_mistakes": 140,
                "rule": (
                    "Carry forward the narrow verified-memory-growth claim with zero "
                    "soundness mistakes and completeness caveat."
                ),
            },
            {
                "track": "telemetry_headline_claim",
                "status": "blocked",
                "telemetry_validity_verdict": (
                    "invalid_for_headline_claim_superficial_or_mechanical_gate"
                ),
                "rule": "Do not claim telemetry validity as a headline.",
            },
            {
                "track": "hardware_simulation",
                "status": "preserved_with_environmental_blocker",
                "rtl_regression_complete": True,
                "thrml_available": False,
                "hardware_claim_allowed": False,
                "rule": (
                    "KV260 remains source-level RTL; THRML/NPIM remains simulator-only "
                    "with no hardware claim."
                ),
            },
        ],
        "preserved_lineages": [
            {
                "lineage": "FR-11 v8 verified-memory-growth pivot",
                "rule": "Carry forward only the narrow zero-soundness-mistake memory-growth claim.",
            },
            {
                "lineage": "KV260 source-level RTL regression",
                "rule": "Carry forward source-level lint/simulation only; no board claim.",
            },
            {
                "lineage": "THRML/NPIM simulator-only parity probe",
                "rule": "Carry forward simulator-only tracking; no TSU hardware claim.",
            },
        ],
        "retired_lineages": [
            {
                "lineage": "HALT/Spilled Energy telemetry diagnostic",
                "reason": "Non-headline telemetry signal was flat or confounded.",
            }
        ],
        "research_roadmap_yaml_modified": False,
        "scripts_research_conductor_modified": False,
    }


def _research_complete_with_113() -> str:
    return """
- id: 2026.04.112
  title: Prior
- id: 2026.04.113
  title: Live SOTA Verifier Telemetry + BEAVER Bounds + Self-Learning Pivot
  completed: '2026-05-07'
"""


def _conductor_log_text() -> str:
    rows = []
    titles = {
        1467: ".112 Completion Archive + .113 Activation Manifest",
        1468: "Live SOTA GGUF Logprob Telemetry Preflight",
        1469: "HALT + Spilled Energy Diagnostic Micro-Benchmark",
        1470: "BEAVER-Lite Deterministic Bound Smoke",
        1471: "FR-11 v8 Verified-Memory-Growth Pivot",
        1472: "Online Verifier Asymmetric Mistake-Budget Audit",
        1473: "Live Telemetry Adversarial Validity Audit",
        1474: "T-SKM Linear Constraint Projection Smoke",
        1475: "STATIC CSR Certificate Automaton Smoke",
        1476: "KV260 Discrete SB RTL Regression Pack",
        1477: "THRML + NPIM Simulator Parity Micro-Probe",
        1478: "Milestone .113 Retrospective",
    }
    for exp_id, title in titles.items():
        rows.append(
            f"| 2026-05-07 11:00 UTC | {title} exp{exp_id} | OK | focused tests passed |"
        )
    return "\n".join(rows)


def _ops_status_text() -> str:
    return """
Milestone 2026.04.114 PLANNED after .113 completion.
live telemetry exists but headline claims remain blocked.
FR-11 moves from verified memory growth to query-time utility.
Hardware evidence remains bounded to dual RTX 3090 runtime, KV260 RTL source/sim, and THRML simulator preflight.
"""


def _ops_changelog_text() -> str:
    return """
Created research-roadmap-next.yaml with .114 tasks: balanced live SOTA telemetry v2,
BEAVER-lite live prefix calibration, HalluGuard-style risk-bound fit,
FR-11 query-time memory policy, CCTU executable constraint micro-benchmark,
V_1 pairwise verification, THRML installability import preflight,
THRML/Carnot simulator parity, Kona/EBT partial-trace localization.
Did NOT modify research-roadmap.yaml or scripts/research_conductor.py.
"""


def test_scenario_report_050_activates_114_and_preserves_guardrails() -> None:
    """SCENARIO-REPORT-050: .114 activation preserves .113 guardrails."""

    artifact, manifest = build_artifact(
        retro=_retro_payload(),
        conductor_log_text=_conductor_log_text(),
        research_complete_text=_research_complete_with_113(),
        ops_status_text=_ops_status_text(),
        ops_changelog_text=_ops_changelog_text(),
        manifest_path="ops/milestone_114_activation_manifest.md",
    )

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["milestone"] == "2026.04.114"
    assert artifact["predecessor_milestone"] == "2026.04.113"
    assert artifact["predecessor_criteria_met"] == 12
    assert artifact["predecessor_criteria_total"] == 12
    assert artifact["activation_manifest_complete"] is True
    assert artifact["telemetry_headline_block_preserved"] is True
    assert artifact["self_learning_followup_allowed"] is True
    assert artifact["research_complete_has_113_entry"] is True
    assert [track["track"] for track in artifact["allowed_114_tracks"]] == [
        track["track"] for track in ALLOWED_114_TRACKS
    ]
    assert [track["track"] for track in artifact["forbidden_reopen_tracks"]] == [
        track["track"] for track in FORBIDDEN_REOPEN_TRACKS
    ]
    assert artifact["hardware_claim_boundaries"]["kv260"]["allowed_evidence"] == [
        "rtl_source",
        "rtl_simulation",
    ]
    assert artifact["hardware_claim_boundaries"]["thrml_tsu"]["hardware_claim_allowed"] is False
    assert artifact["conductor_log_exp1467_to_exp1478"]["missing_experiments"] == []
    assert artifact["no_change_confirmations"] == {
        "research-roadmap.yaml": "unchanged_by_exp1479_activation_workflow",
        "scripts/research_conductor.py": "unchanged_by_exp1479_activation_workflow",
    }
    assert "Adversarial Balanced Telemetry" in manifest
    assert "Telemetry Headline Claims" in manifest
    assert "dual RTX 3090 runtime" in manifest
    assert "No-Change Confirmation" in manifest


def test_req_report_050_records_research_complete_archive_gap() -> None:
    """REQ-REPORT-050: absent .113 archive row is reported explicitly."""

    artifact, manifest = build_artifact(
        retro=_retro_payload(),
        conductor_log_text=_conductor_log_text(),
        research_complete_text="- id: 2026.04.112\n",
        ops_status_text=_ops_status_text(),
        ops_changelog_text=_ops_changelog_text(),
        manifest_path="ops/milestone_114_activation_manifest.md",
    )

    assert artifact["status"] == "complete"
    assert artifact["research_complete_has_113_entry"] is False
    assert artifact["research_complete_archive_update_needed"] is True
    assert artifact["archive_gap"] == {
        "missing_milestone": "2026.04.113",
        "recommended_action": (
            "append .113 archive row to research-complete.yaml without modifying research-roadmap.yaml"
        ),
    }
    assert "Archive gap: `research-complete.yaml` lacks `2026.04.113`." in manifest


def test_req_report_050_blocks_when_predecessor_guardrails_are_missing() -> None:
    """REQ-REPORT-050: missing predecessor guardrails block activation completion."""

    retro = _retro_payload()
    retro["criteria_met"] = 11
    retro["carry_forward_tracks"] = []
    artifact, manifest = build_artifact(
        retro=retro,
        conductor_log_text="exp1467 OK\n",
        research_complete_text=_research_complete_with_113(),
        ops_status_text="",
        ops_changelog_text="",
        manifest_path="ops/milestone_114_activation_manifest.md",
    )

    assert artifact["status"] == "blocked"
    assert artifact["activation_manifest_complete"] is False
    assert artifact["telemetry_headline_block_preserved"] is False
    assert artifact["self_learning_followup_allowed"] is False
    assert "predecessor retro criteria not complete" in artifact["blocked_reasons"]
    assert "telemetry headline block not preserved" in artifact["blocked_reasons"]
    assert "Manifest blocked" in manifest


def test_req_report_050_run_writes_bootstrap_manifest_and_terminal_json(tmp_path: Path) -> None:
    """REQ-REPORT-050: run writes bootstrap, markdown, and terminal artifact."""

    out_path = tmp_path / "results" / "experiment_1479_113_completion_archive_114_activation.json"
    manifest_path = tmp_path / "ops" / "milestone_114_activation_manifest.md"

    bootstrap = write_in_progress_artifact(out_path)
    assert bootstrap["status"] == "in_progress"
    assert json.loads(out_path.read_text(encoding="utf-8"))["status"] == "in_progress"

    _write_json(tmp_path / "results" / "experiment_1478_milestone_113_retro.json", _retro_payload())
    (tmp_path / "ops").mkdir(exist_ok=True)
    (tmp_path / "ops" / "conductor-log.md").write_text(_conductor_log_text(), encoding="utf-8")
    (tmp_path / "ops" / "status.md").write_text(_ops_status_text(), encoding="utf-8")
    (tmp_path / "ops" / "changelog.md").write_text(_ops_changelog_text(), encoding="utf-8")
    (tmp_path / "research-complete.yaml").write_text(
        _research_complete_with_113(),
        encoding="utf-8",
    )

    artifact = run(root=tmp_path, out_path=out_path, manifest_path=manifest_path)
    written = json.loads(out_path.read_text(encoding="utf-8"))
    manifest = manifest_path.read_text(encoding="utf-8")

    assert artifact == written
    assert written["status"] == "complete"
    assert written["manifest_path"] == "ops/milestone_114_activation_manifest.md"
    assert written["source_inputs_read"]["ops/status.md"]["exists"] is True
    assert "Allowed .114 Tracks" in manifest
    assert "Forbidden Reopen Tracks" in manifest


def test_req_report_050_defensive_helpers_and_incomplete_evidence(tmp_path: Path) -> None:
    """REQ-REPORT-050: helpers and incomplete evidence stay explicit."""

    assert _read_json(tmp_path / "missing.json") is None
    assert _read_text(tmp_path / "missing.md") == ""
    assert _relative_path(tmp_path / "ops" / "manifest.md") == "ops/manifest.md"
    assert _relative_path(tmp_path / "results" / "artifact.json") == "results/artifact.json"
    assert _relative_path(tmp_path / "loose.txt") == "loose.txt"

    artifact, manifest = build_artifact(
        retro={"milestone": "2026.04.113", "criteria_met": 12, "criteria_total": 12},
        conductor_log_text="",
        research_complete_text="",
        ops_status_text="",
        ops_changelog_text="",
        manifest_path="ops/milestone_114_activation_manifest.md",
    )

    assert artifact["status"] == "blocked"
    assert artifact["conductor_log_exp1467_to_exp1478"]["ok_count"] == 0
    assert "telemetry headline block not preserved" in artifact["blocked_reasons"]
    assert "self-learning follow-up guardrail missing" in artifact["blocked_reasons"]
    assert "Manifest blocked" in manifest
