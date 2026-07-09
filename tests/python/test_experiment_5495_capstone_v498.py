"""Tests for the Exp5495 V498 capstone synthesis.

Spec refs: REQ-REPORT-5495, SCENARIO-REPORT-5495,
SCENARIO-REPORT-5495-GATE-SKIPS.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_5495_capstone_v498 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"


def _write_json(root: Path, rel_path: str, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_context(root: Path) -> None:
    for rel_path in mod.SOURCE_CONTEXT_PATHS:
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        text = "milestone: 2026.07.498\n" if rel_path.suffix in {".yaml", ".yml"} else ""
        path.write_text(text or f"context for {rel_path.as_posix()}\n", encoding="utf-8")
    (root / "research-roadmap-next.yaml").unlink()
    (root / "scripts").mkdir(exist_ok=True)
    (root / "scripts/research_conductor.py").write_text("# conductor fixture\n", encoding="utf-8")
    (root / "ops/conductor-log.md").write_text(
        "\n".join(
            [
                "Exp 5483 source delta | SKIP | Pre-tests failing",
                "Exp 5484 CSL corrigendum | SKIP | Pre-tests failing",
                "Exp 5485 Preference-MaxSAT | SKIP | Pre-tests failing",
                "Exp 5486 SOTA concept | GATE_BLOCK | upstream retired exp5485",
                "Exp 5487 helper contracts | SKIP | Pre-tests failing",
                "Exp 5488 CSL replay | GATE_BLOCK | upstream retired exp5484",
                "Exp 5489 SOTA CSL | GATE_BLOCK | upstream retired exp5484",
                "Exp 5490 fixed-point | GATE_BLOCK | exp5488 artifact not found",
                "Exp 5494 ARC live | FLAGGED | DURATION_TOO_SHORT",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _populate_artifacts(root: Path) -> None:
    _write_json(
        root,
        "results/experiment_5482_transition_v498.json",
        {
            "status": "complete",
            "honest_verdict": "complete: transition; Exp5474 TAUTOLOGY flag recorded",
            "exp5474_tautology_flag_recorded": True,
            "guided_decoding_quarantine_status": "quarantined",
            "blocked_lanes": [{"lane": "guided_decoding"}],
            "hardware_speedup_claim": False,
        },
    )
    _write_json(
        root,
        "results/experiment_5490_csl_kan_fixed_point_update_ledger_v498.json",
        {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gates_evaluated": [
                {"upstream": "exp5488-csl-latent-exploration-replay-v498", "passed": False}
            ],
        },
    )
    _write_json(
        root,
        "results/experiment_5491_active_constraint_subproblem_descriptor_v498.json",
        {
            "status": "complete",
            "honest_verdict": "complete: descriptors ready; no hardware speedup claim",
            "subproblem_descriptor_ready": True,
            "descriptor_count": 8,
            "exact_fallback_completeness": 1.0,
            "unsafe_false_accept_count": 0,
            "hardware_speedup_claim": False,
        },
    )
    _write_json(
        root,
        "results/experiment_5492_hardware_receipts_v498.json",
        {
            "honest_verdict": "complete: receipt-only; hardware_speedup_claim=false",
            "hardware_receipts_ready": True,
            "hardware_speedup_claim": False,
            "reachable_boards": ["polarfire"],
            "blocked_boards": {
                "kv260": {"blocked_reason": "blocked_kv260_ssh_identity"},
                "gatemate": {"blocked_reason": "blocked_gatemate_jtag_identity"},
            },
            "result_hash_match_rate": 1.0,
            "authenticated_board_identity_count": 1,
        },
    )
    _write_json(
        root,
        "results/experiment_5493_arc_trajectory_target_precheck_v498.json",
        {
            "status": "complete",
            "honest_verdict": "complete: dc22 L3 trajectory precheck ready",
            "arc_trajectory_precheck_ready": True,
            "selected_game": "dc22",
            "selected_target_level": 3,
            "prior_levels_reproduced": 2,
        },
    )
    _write_json(
        root,
        "results/experiment_5494_arc_live_trajectory_levelup_v498.json",
        {
            "status": "honest_null",
            "honest_verdict": "honest_null: dc22 L3 bounded_budget_no_target_level_reproduction",
            "selected_game": "dc22",
            "target_level": 3,
            "prior_levels_reproduced": 2,
            "post_levels_reproduced": 2,
            "new_level_banked": False,
            "registry_updated": False,
            "reproduced_levels": 0,
            "offline_reproduced": False,
            "trajectory_hypothesis_count": 2,
            "live_attempt_count": 47,
            "failure_mode": "bounded_budget_no_target_level_reproduction",
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
        },
    )


def test_req_report_5495_spec_declares_required_fields() -> None:
    """REQ-REPORT-5495: OpenSpec anchors the capstone artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-5495") :]

    assert "SCENARIO-REPORT-5495" in section
    assert "SCENARIO-REPORT-5495-GATE-SKIPS" in section
    assert str(mod.RESULT_RELATIVE_PATH) in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_report_5495_synthesizes_truth_from_actual_artifacts(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5495: present and missing artifacts drive the truth table."""

    _write_context(tmp_path)
    _populate_artifacts(tmp_path)

    report = mod.build_report(tmp_path, tests_run=["unit 5495"])

    assert report["milestone"] == "2026.07.498"
    assert report["artifacts_read"] == [
        "results/experiment_5482_transition_v498.json",
        "results/experiment_5490_csl_kan_fixed_point_update_ledger_v498.json",
        "results/experiment_5491_active_constraint_subproblem_descriptor_v498.json",
        "results/experiment_5492_hardware_receipts_v498.json",
        "results/experiment_5493_arc_trajectory_target_precheck_v498.json",
        "results/experiment_5494_arc_live_trajectory_levelup_v498.json",
    ]
    assert "results/experiment_5484_csl_tautology_corrigendum_v498.json" in report[
        "artifacts_missing"
    ]
    assert report["lane_truth_table"]["guided_decoding"]["classification"] == "blocked"
    assert report["lane_truth_table"]["csl_corrigendum"]["classification"] == "skipped_by_gate"
    assert report["lane_truth_table"]["preference_maxsat_verification"][
        "classification"
    ] == "skipped_by_gate"
    assert report["lane_truth_table"]["concept_sota_telemetry"][
        "classification"
    ] == "skipped_by_gate"
    assert report["lane_truth_table"]["helper_contracts"]["classification"] == "skipped_by_gate"
    assert report["lane_truth_table"]["csl_independent_metrics"][
        "classification"
    ] == "skipped_by_gate"
    assert report["lane_truth_table"]["fixed_point_kan_ledger"][
        "classification"
    ] == "skipped_by_gate"
    assert report["lane_truth_table"]["active_constraints"]["classification"] == "bounded"
    assert report["lane_truth_table"]["hardware"]["classification"] == "bounded"
    assert report["lane_truth_table"]["arc"]["classification"] == "honest_null"
    assert report["lane_truth_table"]["synthesis"]["classification"] == "bounded"

    assert report["headline_ready_lanes"] == []
    assert {row["lane"] for row in report["bounded_lanes"]} == {
        "transition_source_delta",
        "active_constraints",
        "hardware",
        "synthesis",
    }
    assert {row["lane"] for row in report["blocked_lanes"]} == {"guided_decoding"}
    assert {row["lane"] for row in report["honest_null_lanes"]} == {"arc"}
    assert {row["lane"] for row in report["skipped_by_gate_lanes"]} == {
        "csl_corrigendum",
        "preference_maxsat_verification",
        "concept_sota_telemetry",
        "helper_contracts",
        "csl_independent_metrics",
        "fixed_point_kan_ledger",
    }

    assert report["exp5474_tautology_resolved"] is False
    assert report["guided_decoding_quarantine_status"] == "quarantined"
    assert report["csl_status"].startswith("blocked:")
    assert report["arc_registry_delta"] == 0
    assert report["hardware_speedup_claim"] is False
    assert report["prd_gap_table"]["FR-11 continuous self-learning"]["status"].startswith(
        "blocked"
    )
    assert report["prd_gap_table"]["FR-12 verifiable reasoning"]["status"] == "bounded"
    assert report["prd_gap_table"]["hardware acceleration"]["status"] == "bounded_receipts_only"
    assert report["failure_taxonomy"]["arc_no_bank"]["failure_mode"] == (
        "bounded_budget_no_target_level_reproduction"
    )
    assert len(report["next_recommendations"]) == 3
    assert report["roadmap_yaml_unchanged"] is True
    assert report["conductor_unchanged"] is True
    assert report["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert report["honest_verdict"].startswith("complete:")


def test_scenario_report_5495_write_report_persists_required_json(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5495-GATE-SKIPS: written artifact remains auditable."""

    _write_context(tmp_path)
    _populate_artifacts(tmp_path)

    payload = mod.write_report(tmp_path, tests_run=["unit 5495"])
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert written == payload
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(written)
    assert written["reproducibility_checksum"] == mod.payload_checksum(written)
    assert written["source_context_missing"] == ["research-roadmap-next.yaml"]
