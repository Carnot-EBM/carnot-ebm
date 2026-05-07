"""Tests for the Exp 1492 `.115` activation manifest.

Spec: REQ-REPORT-052, SCENARIO-REPORT-052.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting.milestone_115_activation_manifest import (
    ALLOWED_115_TRACKS,
    GATED_115_TRACKS,
    GUARDRAIL_BLOCKS,
    MANDATED_SOTA_MODELS,
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
        "milestone": "2026.04.114",
        "criteria_met": 12,
        "criteria_total": 13,
        "success_threshold_met": True,
        "blocked_task_ids": ["exp1489"],
        "honest_structured_gate_skip_count": 1,
        "honest_structured_gate_skips": [
            {
                "task_id": "exp1489",
                "criterion": "thrml_parity",
                "reason": "Structured gate skip: THRML import readiness was false.",
            }
        ],
        "honest_verdict": (
            "complete: milestone_114_12_of_13_criteria_met_success_threshold_met_"
            "1_honest_gate_skips_ops_reconciliation_delegated"
        ),
        "carry_forward_recommendations": [
            {
                "track": "beaver_lite_bounds",
                "source_experiment": "exp1482",
                "evidence": "sound_bound_live_exp1480_plus_exp1468_calibrated",
                "next_focus": "Expand only with zero bound violations.",
            },
            {
                "track": "fr11_query_time_memory",
                "source_experiment": "exp1484/exp1485",
                "evidence": {
                    "policy": (
                        "query_time_memory_policy_improves_bounded_replay_without_"
                        "false_accepts"
                    ),
                    "completeness": "completeness_reduction_candidate_allowed_zero_soundness",
                },
                "next_focus": "Promote cautiously under the zero-soundness-mistake gate.",
            },
            {
                "track": "cctu_executable_constraints",
                "source_experiment": "exp1486/exp1487",
                "evidence": "complete: executable CCTU micro-benchmark ready",
                "next_focus": "Use deterministic validators as the baseline.",
            },
            {
                "track": "partial_trace_localization",
                "source_experiment": "exp1490",
                "evidence": (
                    "bounded_injected_failure_localization_beats_random_no_"
                    "decoded_quality_claim"
                ),
                "next_focus": "Carry forward injected-failure localization only.",
            },
        ],
        "retired_lineages": [
            {
                "lineage": "semantic_energy_headline_telemetry",
                "decision": "retired",
                "source_experiment": "exp1481",
            },
            {
                "lineage": "v1_pairwise_self_verification_promotion_path",
                "decision": "do_not_promote",
                "source_experiment": "exp1487",
            },
            {
                "lineage": "thrml_carnot_simulator_parity_until_import_ready",
                "decision": "gate_blocked",
                "source_experiment": "exp1488/exp1489",
            },
            {
                "lineage": "prior_scope_reduction_blocks",
                "decision": "preserved",
                "source_experiment": "exp1479",
            },
        ],
        "protected_file_checks": {
            "research-roadmap.yaml": "unchanged",
            "scripts/research_conductor.py": "unchanged",
        },
        "research_roadmap_yaml_modified": False,
        "scripts_research_conductor_modified": False,
    }


def _research_complete_with_114() -> str:
    return """
- id: 2026.04.113
  title: Prior
- id: 2026.04.114
  title: Adversarial Telemetry Bounds + Query-Time Self-Learning + Executable Verification
  completed: '2026-05-07'
"""


def _conductor_log_text() -> str:
    rows = []
    for exp_id in range(1479, 1492):
        rows.append(
            f"| 2026-05-07 16:00 UTC | Milestone task exp{exp_id} | OK | focused tests passed |"
        )
    rows.append(
        "| 2026-05-07 16:17 UTC | exp1489 THRML parity | GATE_BLOCK | import readiness false |"
    )
    return "\n".join(rows)


def _ops_status_text() -> str:
    return """
Milestone 2026.04.115 PLANNED after .114 completion.
Continuous self-learning requirement is satisfied by exp1497.
Mandated local SOTA GGUF MODEL_SPECS: unsloth/Qwen3.6-35B-A3B-GGUF,
unsloth/gemma-4-31B-it-GGUF, and unsloth/gemma-4-26B-A4B-it-GGUF.
Structured gates include exp1495, exp1496, exp1498, exp1500, and exp1504.
Did NOT modify research-roadmap.yaml or scripts/research_conductor.py.
"""


def _ops_changelog_text() -> str:
    return """
Planned milestone 2026.04.115 with trigger-token certificate export,
ConstrainPrompt validator compiler audit, interwhen monitor prototype,
HoVer safe-prefix continuation, FR-11 trace2skill daily eval, artifact
reachability, verifier orthogonality, graph-energy adapter, KAN hardware
accounting, THRML import readiness repair/gate, and gated THRML parity.
Semantic Energy/logit telemetry and V_1 pairwise are retired as headline
signals. Legacy small models are smoke tests only.
"""


def test_scenario_report_052_activates_115_and_preserves_guardrails() -> None:
    """SCENARIO-REPORT-052: .115 activation preserves .114 guardrails."""

    artifact, manifest = build_artifact(
        retro=_retro_payload(),
        conductor_log_text=_conductor_log_text(),
        research_complete_text=_research_complete_with_114(),
        ops_status_text=_ops_status_text(),
        ops_changelog_text=_ops_changelog_text(),
        manifest_path="ops/milestone_115_activation_manifest.md",
    )

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["milestone"] == "2026.04.115"
    assert artifact["predecessor_milestone"] == "2026.04.114"
    assert artifact["predecessor_criteria_met"] == 12
    assert artifact["predecessor_criteria_total"] == 13
    assert artifact["activation_manifest_complete"] is True
    assert artifact["continuous_self_learning_required"] is True
    assert artifact["research_complete_has_114_entry"] is True
    assert artifact["mandated_sota_models"] == MANDATED_SOTA_MODELS
    assert [track["track"] for track in artifact["allowed_115_tracks"]] == [
        track["track"] for track in ALLOWED_115_TRACKS
    ]
    assert [track["track"] for track in artifact["gated_115_tracks"]] == [
        track["track"] for track in GATED_115_TRACKS
    ]
    assert artifact["retired_headline_signals"] == [
        "Semantic Energy/logit telemetry headline claims",
        "V_1 pairwise self-verification headline claims",
    ]
    assert artifact["guardrail_blocks_preserved"] is True
    assert artifact["conductor_log_exp1479_to_exp1491"]["missing_experiments"] == []
    assert artifact["no_change_confirmations"] == {
        "research-roadmap.yaml": "unchanged_by_exp1492_activation_workflow",
        "scripts/research_conductor.py": "unchanged_by_exp1492_activation_workflow",
    }
    assert artifact["honest_verdict"].startswith("complete:")
    assert "Trigger-Token Certificate Export" in manifest
    assert "Semantic Energy/logit telemetry headline claims" in manifest
    assert "legacy small-model headline results" in manifest
    assert "No-Change Confirmation" in manifest


def test_req_report_052_records_research_complete_archive_gap() -> None:
    """REQ-REPORT-052: absent .114 archive row is reported explicitly."""

    artifact, manifest = build_artifact(
        retro=_retro_payload(),
        conductor_log_text=_conductor_log_text(),
        research_complete_text="- id: 2026.04.113\n",
        ops_status_text=_ops_status_text(),
        ops_changelog_text=_ops_changelog_text(),
        manifest_path="ops/milestone_115_activation_manifest.md",
    )

    assert artifact["status"] == "complete"
    assert artifact["research_complete_has_114_entry"] is False
    assert artifact["research_complete_archive_update_needed"] is True
    assert artifact["archive_gap"] == {
        "missing_milestone": "2026.04.114",
        "recommended_action": (
            "append .114 archive row to research-complete.yaml without modifying "
            "research-roadmap.yaml"
        ),
    }
    assert "Archive gap: `research-complete.yaml` lacks `2026.04.114`." in manifest


def test_req_report_052_blocks_when_predecessor_guardrails_are_missing() -> None:
    """REQ-REPORT-052: missing predecessor guardrails block activation completion."""

    retro = _retro_payload()
    retro["criteria_met"] = 11
    retro["retired_lineages"] = []
    artifact, manifest = build_artifact(
        retro=retro,
        conductor_log_text="exp1479 OK\n",
        research_complete_text=_research_complete_with_114(),
        ops_status_text="",
        ops_changelog_text="",
        manifest_path="ops/milestone_115_activation_manifest.md",
    )

    assert artifact["status"] == "blocked"
    assert artifact["activation_manifest_complete"] is False
    assert artifact["guardrail_blocks_preserved"] is False
    assert "predecessor retro criteria not complete" in artifact["blocked_reasons"]
    assert "guardrail blocks not preserved" in artifact["blocked_reasons"]
    assert "Manifest blocked" in manifest


def test_req_report_052_run_writes_bootstrap_manifest_and_terminal_json(tmp_path: Path) -> None:
    """REQ-REPORT-052: run writes bootstrap, markdown, and terminal artifact."""

    out_path = (
        tmp_path / "results" / "experiment_1492_114_completion_archive_115_activation.json"
    )
    manifest_path = tmp_path / "ops" / "milestone_115_activation_manifest.md"

    bootstrap = write_in_progress_artifact(out_path)
    assert bootstrap["status"] == "in_progress"
    assert json.loads(out_path.read_text(encoding="utf-8"))["status"] == "in_progress"

    _write_json(tmp_path / "results" / "experiment_1491_milestone_114_retro.json", _retro_payload())
    (tmp_path / "ops").mkdir(exist_ok=True)
    (tmp_path / "ops" / "conductor-log.md").write_text(_conductor_log_text(), encoding="utf-8")
    (tmp_path / "ops" / "status.md").write_text(_ops_status_text(), encoding="utf-8")
    (tmp_path / "ops" / "changelog.md").write_text(_ops_changelog_text(), encoding="utf-8")
    (tmp_path / "research-complete.yaml").write_text(
        _research_complete_with_114(),
        encoding="utf-8",
    )

    artifact = run(root=tmp_path, out_path=out_path, manifest_path=manifest_path)
    written = json.loads(out_path.read_text(encoding="utf-8"))
    manifest = manifest_path.read_text(encoding="utf-8")

    assert artifact == written
    assert written["status"] == "complete"
    assert written["manifest_path"] == "ops/milestone_115_activation_manifest.md"
    assert written["source_inputs_read"]["ops/status.md"]["exists"] is True
    assert "Allowed .115 Tracks" in manifest
    assert "Gated .115 Tracks" in manifest


def test_req_report_052_defensive_helpers_and_incomplete_evidence(tmp_path: Path) -> None:
    """REQ-REPORT-052: helpers and incomplete evidence stay explicit."""

    assert _read_json(tmp_path / "missing.json") is None
    assert _read_text(tmp_path / "missing.md") == ""
    assert _relative_path(tmp_path / "ops" / "manifest.md") == "ops/manifest.md"
    assert _relative_path(tmp_path / "results" / "artifact.json") == "results/artifact.json"
    assert _relative_path(tmp_path / "loose.txt") == "loose.txt"
    assert "THRML parity before import readiness" in GUARDRAIL_BLOCKS

    artifact, manifest = build_artifact(
        retro={"milestone": "2026.04.114", "criteria_met": 12, "criteria_total": 13},
        conductor_log_text="",
        research_complete_text="",
        ops_status_text="",
        ops_changelog_text="",
        manifest_path="ops/milestone_115_activation_manifest.md",
    )

    assert artifact["status"] == "blocked"
    assert artifact["conductor_log_exp1479_to_exp1491"]["ok_count"] == 0
    assert "structured THRML gate skip not recorded" in artifact["blocked_reasons"]
    assert "guardrail blocks not preserved" in artifact["blocked_reasons"]
    assert "Manifest blocked" in manifest
