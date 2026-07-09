"""Tests for the Exp5496 .499 transition receipt.

Spec refs: REQ-REPORT-5496, SCENARIO-REPORT-5496,
SCENARIO-REPORT-5496-BLOCKED-INPUT.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from carnot import experiment_5496_transition_v499 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_context(root: Path) -> None:
    for rel_path in mod.SOURCE_CONTEXT_PATHS:
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if rel_path == mod.ROADMAP_RELATIVE_PATH:
            path.write_text(
                yaml.safe_dump(
                    {
                        "milestone": mod.MILESTONE,
                        "milestone_doc": mod.VNEXT_RELATIVE_PATH.as_posix(),
                        "tasks": [
                            {"id": task_id, "milestone": mod.MILESTONE}
                            for task_id in mod.EXPECTED_TASK_IDS
                        ],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
        elif rel_path == mod.VNEXT_RELATIVE_PATH:
            path.write_text(
                "\n".join(
                    [
                        "# Research Roadmap vNEXT - Milestone 2026.07.499",
                        "",
                        "**Previous milestone:** 2026.07.498",
                        "**Task range:** Exp 5496-5509",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text(f"context for {rel_path.as_posix()}\n", encoding="utf-8")
    (root / mod.ROADMAP_NEXT_RELATIVE_PATH).unlink()
    (root / mod.CONDUCTOR_LOG_RELATIVE_PATH).write_text(
        "\n".join(
            [
                "| 2026-07-09 13:24 UTC | Execution-time 2025-2026 source delta for .498 | SKIP | Pre-tests failing, self-heal failed: 1 failed, 86 passed |",
                "| 2026-07-09 13:36 UTC | CSL tautology corrigendum and metric-independence | SKIP | Pre-tests failing, self-heal failed: 1 failed, 86 passed |",
                "| 2026-07-09 13:43 UTC | Preference-MaxSAT typed claim-state fixture | SKIP | Pre-tests failing, self-heal failed: 1 failed, 86 passed |",
                "| 2026-07-09 13:45 UTC | Gated local SOTA concept evidence telemetry panel | GATE_BLOCK | Pre-emptive skip: upstream retired (exp5485-preference-maxsat-claim-fixture-v498 |",
                "| 2026-07-09 13:50 UTC | Natural-language helper-contract repair | SKIP | Pre-tests failing, self-heal failed: 1 failed, 86 passed |",
                "| 2026-07-09 13:56 UTC | CSL KAN fixed-point update ledger | GATE_BLOCK | 1 of 1 gate(s) failed; first failure: exp5488-csl-latent-exploration-replay-v498 |",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _row(
    lane: str,
    classification: str,
    source_artifacts: list[str],
    evidence: dict[str, Any],
    claim_boundary: str = "fixture boundary",
) -> dict[str, Any]:
    return {
        "lane": lane,
        "classification": classification,
        "source_artifacts": source_artifacts,
        "evidence": evidence,
        "claim_boundary": claim_boundary,
    }


def _capstone_payload() -> dict[str, Any]:
    blocked_boards = {
        "kv260": {
            "board_identity": "kv260",
            "blocked_reason": "blocked_kv260_ssh_identity",
            "reachable": False,
        },
        "gatemate": {
            "board_identity": "gatemate",
            "blocked_reason": "blocked_gatemate_jtag_identity",
            "reachable": False,
            "diagnostic_only": True,
        },
    }
    return {
        "milestone": mod.PREVIOUS_MILESTONE,
        "status": "complete",
        "honest_verdict": "complete: .498 capstone recorded missing/skipped lanes",
        "artifacts_read": [
            "results/experiment_5482_transition_v498.json",
            "results/experiment_5490_csl_kan_fixed_point_update_ledger_v498.json",
            "results/experiment_5491_active_constraint_subproblem_descriptor_v498.json",
            "results/experiment_5492_hardware_receipts_v498.json",
            "results/experiment_5493_arc_trajectory_target_precheck_v498.json",
            "results/experiment_5494_arc_live_trajectory_levelup_v498.json",
        ],
        "artifacts_missing": [
            "results/experiment_5483_source_delta_v498.json",
            "results/experiment_5484_csl_tautology_corrigendum_v498.json",
            "results/experiment_5485_preference_maxsat_claim_fixture_v498.json",
            "results/experiment_5486_sota_concept_evidence_panel_v498.json",
            "results/experiment_5487_helper_contract_nl_spec_repair_v498.json",
            "results/experiment_5488_csl_latent_exploration_replay_v498.json",
            "results/experiment_5489_sota_csl_independent_metrics_v498.json",
        ],
        "lane_truth_table": {
            "transition_source_delta": _row(
                "transition_source_delta",
                "bounded",
                [
                    "results/experiment_5482_transition_v498.json",
                    "results/experiment_5483_source_delta_v498.json",
                ],
                {"transition_complete": True, "source_delta_missing": True},
            ),
            "active_constraints": _row(
                "active_constraints",
                "bounded",
                ["results/experiment_5491_active_constraint_subproblem_descriptor_v498.json"],
                {
                    "subproblem_descriptor_ready": True,
                    "descriptor_count": 8,
                    "exact_fallback_completeness": 1.0,
                    "unsafe_false_accept_count": 0,
                    "hardware_speedup_claim": False,
                },
            ),
            "hardware": _row(
                "hardware",
                "bounded",
                ["results/experiment_5492_hardware_receipts_v498.json"],
                {
                    "hardware_receipts_ready": True,
                    "reachable_boards": ["polarfire"],
                    "blocked_boards": blocked_boards,
                    "result_hash_match_rate": 1.0,
                    "hardware_speedup_claim": False,
                },
            ),
            "arc": _row(
                "arc",
                "honest_null",
                [
                    "results/experiment_5493_arc_trajectory_target_precheck_v498.json",
                    "results/experiment_5494_arc_live_trajectory_levelup_v498.json",
                ],
                {
                    "precheck_ready": True,
                    "selected_game": "dc22",
                    "target_level": 3,
                    "arc_registry_delta": 0,
                    "new_level_banked": False,
                    "flagged_adversarial": True,
                },
            ),
            "synthesis": _row(
                "synthesis",
                "bounded",
                [mod.PRIOR_CAPSTONE_RELATIVE_PATH.as_posix()],
                {"inference_substrate": mod.INFERENCE_SUBSTRATE, "upstream_missing_count": 7},
            ),
            "guided_decoding": _row(
                "guided_decoding",
                "blocked",
                ["results/experiment_5482_transition_v498.json"],
                {"quarantine_status": "quarantined"},
            ),
            "fixed_point_kan_ledger": _row(
                "fixed_point_kan_ledger",
                "skipped_by_gate",
                ["results/experiment_5490_csl_kan_fixed_point_update_ledger_v498.json"],
                {"blocked_at_layer": "conductor_pre_gate"},
            ),
        },
        "exp5474_tautology_resolved": False,
        "csl_status": "blocked: Exp5474 tautology unresolved",
        "arc_registry_delta": 0,
        "hardware_speedup_claim": False,
        "roadmap_yaml_unchanged": True,
        "conductor_unchanged": True,
        "inference_substrate": mod.INFERENCE_SUBSTRATE,
    }


def _write_artifacts(root: Path) -> None:
    blocked_boards = _capstone_payload()["lane_truth_table"]["hardware"]["evidence"][
        "blocked_boards"
    ]
    _write_json(root, mod.PRIOR_CAPSTONE_RELATIVE_PATH, _capstone_payload())
    _write_json(
        root,
        mod.HARDWARE_RELATIVE_PATH,
        {
            "milestone": mod.PREVIOUS_MILESTONE,
            "honest_verdict": "complete: PolarFire hash receipts matched; no speedup claim",
            "hardware_receipts_ready": True,
            "reachable_boards": ["polarfire"],
            "blocked_boards": blocked_boards,
            "result_hash_match_rate": 1.0,
            "authenticated_board_identity_count": 1,
            "hardware_speedup_claim": False,
            "board_receipts": [
                {
                    "board_identity": "polarfire",
                    "aggregate_output_hash": "f59d47034d3de26b",
                    "repeat_count": 3,
                    "matched_repeat_count": 3,
                    "invalid_repeat_count": 0,
                }
            ],
        },
    )
    _write_json(
        root,
        mod.ARC_LIVE_RELATIVE_PATH,
        {
            "milestone": mod.PREVIOUS_MILESTONE,
            "status": "honest_null",
            "honest_verdict": "honest_null: dc22 L3 bounded_budget_no_target_level_reproduction",
            "selected_game": "dc22",
            "target_level": 3,
            "prior_levels_reproduced": 2,
            "post_levels_reproduced": 2,
            "new_level_banked": False,
            "registry_updated": False,
            "offline_reproduced": False,
            "failure_mode": "bounded_budget_no_target_level_reproduction",
            "flagged_adversarial": True,
            "corrigendum_pending": [
                {"kind": "DURATION_TOO_SHORT", "severity": "critical"},
                {"kind": "METHODOLOGY_MISSING", "severity": "warn"},
            ],
        },
    )


def test_req_report_5496_spec_declares_required_fields() -> None:
    """REQ-REPORT-5496: OpenSpec anchors the transition artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-5496") :]

    assert "SCENARIO-REPORT-5496" in section
    assert "SCENARIO-REPORT-5496-BLOCKED-INPUT" in section
    assert str(mod.RESULT_RELATIVE_PATH) in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_report_5496_summarizes_v498_facts(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5496: observed .498 artifacts drive the lane summary."""

    _write_context(tmp_path)
    _write_artifacts(tmp_path)

    report = mod.build_report(tmp_path, tests_run=["unit 5496"])

    assert report["milestone"] == "2026.07.499"
    assert report["previous_milestone"] == "2026.07.498"
    assert report["prior_capstone_path"] == "results/experiment_5495_capstone_v498.json"
    assert report["previous_task_range"] == "exp5482-exp5495"
    assert report["next_task_range"] == "exp5496-exp5509"
    assert {row["lane"] for row in report["clean_lanes"]} == {
        "transition",
        "active_constraint_subproblem_descriptors",
        "hardware_polarfire_hash_receipts",
        "arc_target_precheck",
        "capstone_synthesis",
    }
    assert {row["lane"] for row in report["missing_or_skipped_lanes"]} == {
        "source_delta",
        "csl_tautology_corrigendum",
        "preference_maxsat_fixture",
        "concept_telemetry",
        "helper_contract_repair",
        "csl_independent_metrics",
        "downstream_gate_blocked_csl_hardware_mapping",
    }
    assert {row["lane"] for row in report["blocked_lanes"]} == {
        "guided_decoding_quarantine",
        "kv260_ssh_identity",
        "gatemate_jtag_identity",
    }
    assert {row["lane"] for row in report["honest_null_lanes"]} == {
        "arc_dc22_l3_no_bank",
        "arc_registry_delta_zero",
        "hardware_speedup_claim_false",
    }
    assert {row["lane"] for row in report["flagged_lanes"]} == {
        "exp5474_tautology_unresolved",
        "exp5494_arc_methodology_flag",
    }
    assert report["exp5474_tautology_still_blocks_csl_headlines"] is True
    assert report["roadmap_yaml_unchanged"] is True
    assert report["conductor_unchanged"] is True
    assert report["source_context_missing"] == ["research-roadmap-next.yaml"]
    assert report["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert report["status"] == "complete"
    assert report["honest_verdict"].startswith("complete:")


def test_scenario_report_5496_dirty_protected_file_blocks_but_preserves_evidence(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5496-BLOCKED-INPUT: protected-file dirt fails closed."""

    _write_context(tmp_path)
    _write_artifacts(tmp_path)

    report = mod.build_report(
        tmp_path,
        tests_run=["unit 5496"],
        modification_overrides={mod.ROADMAP_RELATIVE_PATH: True},
    )

    assert report["status"] == "blocked"
    assert report["honest_verdict"].startswith("blocked:")
    assert report["roadmap_yaml_unchanged"] is False
    assert report["conductor_unchanged"] is True
    assert "research-roadmap.yaml_modified" in report["failed_preconditions"]
    assert {row["lane"] for row in report["clean_lanes"]} >= {
        "hardware_polarfire_hash_receipts",
        "capstone_synthesis",
    }


def test_scenario_report_5496_defensive_artifact_fallbacks_preserve_schema() -> None:
    """SCENARIO-REPORT-5496-BLOCKED-INPUT: partial inputs keep stable rows."""

    assert mod._truth_row({}, "missing") == {}
    assert mod._sources({}, ["fallback.json"]) == ["fallback.json"]
    assert mod._board_receipt_summary({}) == []

    blocked = mod.derive_blocked_lanes(_capstone_payload(), {})

    assert {row["lane"] for row in blocked} == {
        "guided_decoding_quarantine",
        "kv260_ssh_identity",
        "gatemate_jtag_identity",
    }
    assert blocked[1]["evidence"]["blocked_reason"] == "blocked_kv260_ssh_identity"


def test_scenario_report_5496_write_report_persists_required_json(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5496: written artifact contains the required fields."""

    _write_context(tmp_path)
    _write_artifacts(tmp_path)

    payload = mod.write_report(tmp_path, tests_run=["unit 5496"])
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert written == payload
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(written)
    assert written["reproducibility_checksum"] == mod.payload_checksum(written)
