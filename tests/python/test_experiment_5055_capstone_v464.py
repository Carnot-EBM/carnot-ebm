"""Tests for Exp 5055 .464 capstone aggregation.

Spec refs: REQ-CAPSTONE-5055, SCENARIO-CAPSTONE-5055,
SCENARIO-CAPSTONE-5055-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5055_capstone_v464 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/capstone/spec.md"


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _gate(state: str = "execution_incomplete") -> JsonDict:
    return {
        "experiment": "experiment_5050_moat_gate_resolution_v464",
        "experiment_id": 5050,
        "honest_verdict": "complete_moat_execution_incomplete_v464_blocked_or_missing_phase_d",
        "moat_state": state,
        "best_arm": "D1",
        "best_arm_delta": 0.08,
        "best_arm_ci": [0.0, 0.165],
        "second_corpus_confirmed": False,
        "cascade_efficiency_win": False,
        "bounded_retirement_ok": state == "retired_bounded",
        "execution_incomplete_reasons": [
            "D1 blocked: blocked_sota_candidate_refresh_unavailable",
            "D2 flagged: complete_process_reward_no_win_musr_minus_0p030",
            "D6 blocked: blocked_gate_check_failed",
            "D6 missing: /repo/results/experiment_5048_cross_model_cascade_repair.json",
        ],
        "blocked_upstream_artifacts": [
            {
                "arm": "powered_lora_ebm_eorm",
                "arm_id": "D1",
                "honest_verdict": "blocked_sota_candidate_refresh_unavailable",
                "path": "/repo/results/experiment_5045_powered_lora_ebm_eorm_musr.json",
            },
            {
                "arm": "cross_model_cascade",
                "arm_id": "D6",
                "honest_verdict": "blocked_gate_check_failed",
                "path": "/repo/results/experiment_5048_d6.json",
                "status": "blocked",
            },
        ],
        "flagged_upstream_artifacts": [
            {
                "arm": "vpr_process_reward",
                "arm_id": "D2",
                "honest_verdict": "complete_process_reward_no_win_musr_minus_0p030",
            },
            {
                "arm": "second_corpus_confirmation",
                "arm_id": "D4",
                "honest_verdict": "success_second_corpus_confirms_musr_margin_constraintbench_exact_v1_plus_0p370",
            },
        ],
        "missing_upstream_artifacts": [
            {
                "arm": "cross_model_cascade",
                "arm_id": "D6",
                "path": "/repo/results/experiment_5048_cross_model_cascade_repair.json",
            }
        ],
        "per_arm_table": [
            {
                "arm": "powered_lora_ebm_eorm",
                "arm_id": "D1",
                "delta_vs_tuned_sc": 0.08,
                "paired_ci95": [0.0, 0.165],
                "execution_status": "blocked",
                "proper_musr_win": False,
                "clean_no_win": False,
                "verifier_is_oracle": False,
                "headroom_present": True,
                "honest_verdict": "blocked_sota_candidate_refresh_unavailable",
            },
            {
                "arm": "kan_purm_energy_calibration",
                "arm_id": "D3",
                "delta_vs_tuned_sc": 0.02,
                "paired_ci95": [-0.105, -0.015],
                "execution_status": "clean",
                "proper_musr_win": False,
                "clean_no_win": True,
                "verifier_is_oracle": False,
                "headroom_present": True,
            },
        ],
        "cascade_artifact": {
            "arm": "cross_model_cascade",
            "arm_id": "D6",
            "execution_status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "efficiency_win": False,
            "judge_call_fraction": None,
            "paired_ci95": None,
        },
        "second_corpus_artifact": {
            "arm": "second_corpus_confirmation",
            "arm_id": "D4",
            "best_arm": "D1",
            "execution_status": "flagged",
            "honest_verdict": "success_second_corpus_confirms_musr_margin_constraintbench_exact_v1_plus_0p370",
            "second_corpus_confirmed": True,
            "delta_vs_tuned_sc_second": 0.37,
            "paired_ci95_second": [0.28, 0.47],
        },
    }


def _fr11(heldout_delta: float = -0.05) -> JsonDict:
    return {
        "experiment": "experiment_5051_verifier_trace_self_learning",
        "experiment_id": 5051,
        "honest_verdict": "complete_verifier_trace_self_learning_replay_memory_minus_0p050",
        "self_learning_loop_executed": True,
        "near_miss_count": 312,
        "verified_trace_count": 225,
        "pre_update_accuracy": 0.7,
        "post_update_accuracy": round(0.7 + heldout_delta, 6),
        "heldout_delta": heldout_delta,
        "contamination_guard_passed": True,
        "delta_vs_genuine_tuned_sc": 0.125,
        "checkpoint_or_memory_path": "results/replay_memory/experiment_5051_verifier_trace_self_learning_memory.json",
        "fr11_evidence": {
            "prd_ref": "FR-11",
            "objective_evaluator": "held-out verifier accuracy delta",
            "guardrails": ["contamination guard"],
        },
    }


def _hardware() -> JsonDict:
    return {
        "experiment": 5052,
        "experiment_id": 5052,
        "honest_verdict": "success_kv260_pbit_timing_ratio_packet_built",
        "kv260_ssh_reachable": True,
        "overlay_loaded": True,
        "loaded_overlay": "carnot_ising_v2_n64",
        "timing_ratio_packet_built": True,
        "cpu_reference_ok": True,
        "kv260_result_ok": True,
        "local_claim_scope": "local_ssh_attached_kv260_python_parity_workload_only_no_general_fpga_speedup_claim",
        "timing_ratio_packet": {
            "parity_match": True,
            "cpu_to_kv260_board_workload_ratio": 0.031477449031,
        },
    }


def _sota() -> JsonDict:
    return {
        "experiment": "experiment_5053_sota_ingestion_v465",
        "experiment_id": 5053,
        "honest_verdict": "success_sota_ingestion_v465_actionable_references_added",
        "research_references_updated": True,
        "n_sources_checked": 8,
        "next_milestone_candidates": [
            {
                "candidate": "Tool-first verifier gate before judge fallback",
                "candidate_flag": "flagged_for_v465 (.465): tool_first_small_verifier_gate",
                "source_ids": ["2504.04718", "2604.01993"],
            }
        ],
    }


def _arc() -> JsonDict:
    return {
        "experiment": "experiment_5054_arc_live_path_self_discovery",
        "experiment_id": 5054,
        "honest_verdict": "complete_tu93_no_new_level_residual_duplicate_depth",
        "target_game": "tu93",
        "target_level": 6,
        "prior_reproduced_level": 5,
        "new_levels_banked": 0,
        "offline_reproduced": False,
        "duplicate_solve_avoided": True,
        "registry_precheck_passed": True,
        "reproducible_total_levels_before": 69,
        "reproducible_total_levels_after": 69,
        "solve_claim": {
            "claimed": False,
            "provenance": "live_agent_self_discovery",
            "reproduction_gate": {"reproduced": False, "reached_level": 0},
        },
    }


def _write_all(
    root: Path,
    *,
    gate: JsonDict | None = None,
    fr11: JsonDict | None = None,
    omit: set[str] | None = None,
) -> None:
    payloads = {
        "moat_gate": gate if gate is not None else _gate(),
        "fr11_self_learning": fr11 if fr11 is not None else _fr11(),
        "hardware": _hardware(),
        "sota": _sota(),
        "arc": _arc(),
    }
    omitted = omit or set()
    for source, payload in payloads.items():
        if source not in omitted:
            _write_json(root / mod.UPSTREAMS[source].relative_path, payload)


def test_req_capstone_5055_spec_declares_v464_capstone_contract() -> None:
    """REQ-CAPSTONE-5055: OpenSpec anchors the .464 capstone artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-CAPSTONE-5055",
        "SCENARIO-CAPSTONE-5055",
        "SCENARIO-CAPSTONE-5055-FIELD-PRINCIPLES",
        "experiment_5055_capstone_v464.py",
        "results/experiment_5055_capstone_v464.json",
        "fr11_self_learning_result",
        "next_milestone_pointer",
        "execution-incomplete",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_5055_aggregates_current_execution_incomplete_state(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-5055: blocked and no-bank evidence is preserved."""

    _write_all(tmp_path)

    artifact = mod.run(root=tmp_path, artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH)

    assert artifact["honest_verdict"] == (
        "complete_capstone_v464_execution_incomplete_fr11_no_credible_positive_evidence"
    )
    assert artifact["capstone_ready"] is True
    assert artifact["milestone"] == "2026.06.464"
    assert artifact["moat_state"] == "execution_incomplete"
    assert artifact["best_arm_and_delta"] == {
        "arm_id": "D1",
        "delta": 0.08,
        "ci95": [0.0, 0.165],
        "evidence_status": "blocked",
        "headline_countable": False,
    }
    assert artifact["best_verifier_evidence"]["honest_verdict"] == (
        "blocked_sota_candidate_refresh_unavailable"
    )
    assert "D6 missing" in "\n".join(artifact["moat_resolution"]["execution_incomplete_reasons"])
    assert artifact["second_corpus_state"]["state"] == "flagged_not_counted"
    assert artifact["cascade_state"]["state"] == "blocked"
    assert artifact["fr11_self_learning_result"]["credible_evidence"] is False
    assert artifact["fr11_self_learning_result"]["state"] == "guarded_negative"
    assert artifact["fr11_self_learning_result"]["heldout_delta"] == -0.05
    assert artifact["hardware_result"]["timing_ratio_packet_built"] is True
    assert artifact["hardware_result"]["claim_scope"].endswith("no_general_fpga_speedup_claim")
    assert artifact["arc_result"]["state"] == "no_bank"
    assert artifact["arc_result"]["new_levels_banked"] == 0
    assert artifact["arc_result"]["reproducible_total_levels_after"] == 69
    assert artifact["next_milestone_pointer"]["selected"]["experiment_class"] == (
        "phase_d_execution_repair_and_confirmation"
    )
    assert set(artifact["next_milestone_pointer"]["by_moat_state"]) == {
        "moat_realized",
        "musr_scoped_positive",
        "retired_bounded",
        "execution_incomplete",
    }
    assert artifact["docs_updated"] == {
        "openspec_capstone_spec": True,
        "ops_status": False,
        "ops_changelog": False,
        "_bmad_traceability": False,
        "reason": "operator stop rule delegates ops/status/changelog/traceability reconciliation",
    }
    assert {row["source"] for row in artifact["cited_upstream_artifacts"]} == set(mod.UPSTREAMS)
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert mod.artifact_schema_errors(artifact) == []


@pytest.mark.parametrize(
    ("state", "label", "experiment_class"),
    [
        ("moat_realized", "realized_moat", "scale_realized_verifier"),
        ("musr_scoped_positive", "musr_scoped_positive", "second_corpus_confirmation"),
        ("retired_bounded", "bounded_retirement", "new_verifier_direction_from_sota"),
        (
            "execution_incomplete",
            "execution_incomplete",
            "phase_d_execution_repair_and_confirmation",
        ),
    ],
)
def test_scenario_capstone_5055_routes_every_blunt_moat_state(
    tmp_path: Path,
    state: str,
    label: str,
    experiment_class: str,
) -> None:
    """SCENARIO-CAPSTONE-5055: .465 pointers cover all allowed capstone states."""

    _write_all(tmp_path, gate=_gate(state), fr11=_fr11(heldout_delta=0.04))

    artifact = mod.run(root=tmp_path, artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH)

    assert artifact["honest_verdict"] == f"complete_capstone_v464_{label}_fr11_credible_positive"
    assert artifact["next_milestone_pointer"]["selected_state"] == state
    assert artifact["next_milestone_pointer"]["selected"]["experiment_class"] == experiment_class
    assert artifact["fr11_self_learning_result"]["credible_evidence"] is True
    assert artifact["fr11_self_learning_result"]["state"] == "credible_positive"
    assert mod.artifact_schema_errors(artifact) == []


def test_req_capstone_5055_missing_inputs_and_validation_edges(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-CAPSTONE-5055: missing capstone inputs are explicit, not fabricated."""

    _write_all(tmp_path, omit={"arc"})
    loaded = mod.load_upstream_artifacts(tmp_path)
    artifact = mod.build_artifact(loaded, duration_s=0.0)

    assert artifact["capstone_ready"] is False
    assert artifact["missing_capstone_inputs"] == [
        {
            "source": "arc",
            "experiment_id": 5054,
            "path": "results/experiment_5054_arc_live_path_self_discovery.json",
        }
    ]
    assert artifact["arc_result"]["state"] == "missing"
    assert artifact["duration_s"] == 0.0001
    assert "honest_verdict" in mod.artifact_schema_errors({})

    invalid = dict(artifact)
    invalid["honest_verdict"] = "bad"
    invalid["moat_state"] = "bad"
    invalid["capstone_ready"] = "yes"
    invalid["fr11_self_learning_result"] = []
    invalid["docs_updated"] = {"ops_status": True}
    errors = mod.artifact_schema_errors(invalid)
    for field in (
        "honest_verdict",
        "moat_state",
        "capstone_ready",
        "fr11_self_learning_result",
        "docs_updated",
    ):
        assert field in errors

    _write_all(tmp_path)
    exit_code = mod.main(root=tmp_path, artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH)
    captured = capsys.readouterr()
    printed = json.loads(captured.out)
    assert exit_code == 0
    assert printed["experiment"] == mod.EXPERIMENT
    assert printed["reproducibility_checksum"] == mod.payload_checksum(printed)
