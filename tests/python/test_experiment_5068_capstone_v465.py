"""Tests for Exp 5068 .465 capstone aggregation.

Spec refs: REQ-CAPSTONE-5068, SCENARIO-CAPSTONE-5068,
SCENARIO-CAPSTONE-5068-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5068_capstone_v465 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _moat(state: str = "execution_incomplete") -> JsonDict:
    return {
        "experiment": "experiment_5063_moat_gate_resolution_v465",
        "experiment_id": 5063,
        "honest_verdict": "complete_moat_execution_incomplete_v465_blocked_flagged_or_unclean",
        "moat_state": state,
        "best_arm": "D6" if state == "execution_incomplete" else "D1",
        "best_arm_delta": 0.08,
        "best_arm_ci": [0.0, 0.165],
        "second_corpus_confirmed": state == "moat_realized",
        "second_corpus_audit_clean": state == "moat_realized",
        "cascade_efficiency_win": state in {"moat_realized", "execution_incomplete"},
        "guided_decoding_frontier_state": "guided_gain_observed_plus_0p111",
        "bounded_retirement_ok": state == "retired_bounded",
        "execution_incomplete_reasons": [
            "D1 flagged: complete_d1_sota_refresh_audit_no_proper_win_plus_0p080",
            "D4 audit not clean: second_corpus_audit_clean=false",
            "D6 efficiency observed but accuracy CI does not exclude zero",
        ]
        if state == "execution_incomplete"
        else [],
        "per_arm_table": [
            {
                "artifact_id": "D1",
                "arm": "d1_sota_refresh_audit",
                "status": "flagged" if state == "execution_incomplete" else "clean",
                "delta": 0.08,
                "ci95": [0.0, 0.165],
                "honest_verdict": "complete_d1_sota_refresh_audit_no_proper_win_plus_0p080",
                "verifier_is_oracle": False,
            },
            {
                "artifact_id": "D6",
                "arm": "d6_tool_first_cascade",
                "status": "clean",
                "delta": 0.08,
                "ci95": [0.0, 0.165],
                "honest_verdict": "success_tool_first_cascade_parity_at_0pct_judge_calls",
                "efficiency_win": True,
                "verifier_is_oracle": False,
            },
        ],
    }


def _fr11(*, promoted: bool = False) -> JsonDict:
    delta = 0.04 if promoted else -0.05
    return {
        "experiment": "experiment_5064_audited_skillgraph_self_learning",
        "experiment_id": 5064,
        "honest_verdict": "success_promoted_skill_graph_plus_0p040"
        if promoted
        else "complete_guarded_no_promote_minus_0p050",
        "continuous_self_learning_task": True,
        "self_learning_loop_executed": True,
        "near_miss_count": 262,
        "candidate_skill_count": 2,
        "verified_skill_count": 2,
        "promoted": promoted,
        "promoted_skill_ids": ["skill_5064_positive_utility_promotion_gate"] if promoted else [],
        "promotion_decision": {
            "promoted": promoted,
            "no_promote_reason": "" if promoted else "heldout_delta_nonpositive;nonforgetting_regressed",
        },
        "no_promote_reason": "" if promoted else "heldout_delta_nonpositive;nonforgetting_regressed",
        "pre_update_accuracy": 0.7,
        "post_update_accuracy": 0.74 if promoted else 0.65,
        "heldout_delta": delta,
        "nonforgetting_delta": 0.0 if promoted else -0.142857,
        "contamination_guard_passed": True,
        "external_verifier_audit_receipts": [
            {"verifier": "deterministic_external_skill_audit_v1", "passed": True}
        ],
        "skill_graph_path": "results/replay_memory/experiment_5064_skill_graph.json",
        "skill_graph_sha256": "sha256:" + "1" * 64,
        "legacy_models_smoke_only": True,
    }


def _hardware() -> JsonDict:
    return {
        "experiment": 5065,
        "honest_verdict": "success_kv260_testbench_timing_packet_built",
        "kv260_ssh_reachable": True,
        "overlay_loaded": True,
        "loaded_overlay": "carnot_ising_v2_n64",
        "timing_ratio_packet_built": True,
        "cpu_reference_ok": True,
        "kv260_result_ok": True,
        "local_claim_scope": (
            "local_ssh_attached_kv260_python_testbench_on_confirmed_carnot_overlay_only_"
            "no_general_fpga_speedup_claim_no_gpu_benchmark_claim_no_external_2026_paper_claim"
        ),
        "board_transcript_path": "results/experiment_5065.transcript.jsonl",
        "transcript_sha256": "0400a1f87e3feb949c08d3d65c6d58d4f5e2018df40cdb385c5c4dfbb43f2956",
        "structured_testbench_evidence": {"status": "packet_built"},
        "timing_ratio_packet": {
            "parity_match": True,
            "cpu_to_kv260_board_workload_ratio": 0.03302962545,
        },
        "optional_board_prechecks": {
            "gatemate": {"status": "not_run_scope_guard"},
            "polarfire": {"status": "not_run_scope_guard"},
        },
    }


def _sota() -> JsonDict:
    return {
        "experiment": "experiment_5066_sota_ingestion_v466",
        "experiment_id": 5066,
        "honest_verdict": "success_sota_ingestion_v466_actionable_references_added",
        "research_references_updated": True,
        "n_sources_checked": 9,
        "selected_sources": [{"source_id": "2606.00001", "hook": "fresh verifier route"}],
        "duplicate_filter": {"deduplicated_against_exp5053": True},
        "next_milestone_candidates": [
            {"candidate": "Fresh verifier route", "source_ids": ["2606.00001"]}
        ],
        "preconditions_checked": {"research_references_present": True},
    }


def _arc() -> JsonDict:
    return {
        "experiment": "experiment_5067_arc_live_path_self_discovery",
        "experiment_id": 5067,
        "honest_verdict": "complete_re86_no_new_level_residual_duplicate_depth",
        "target_game": "re86",
        "target_level": 3,
        "prior_reproduced_level": 2,
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
        "solve_provenance": "live_agent_self_discovery",
        "provenance_evidence": {
            "runtime_self_discovery": True,
            "offline_source_reading_used": False,
            "offline_ground_truth_bfs_used": False,
            "hand_built_adapter_used": False,
            "llm_reasoning_invoked": False,
        },
        "live_agent_attempts": [{"target_game": "re86", "max_level_reached": 0}],
    }


def _prior_capstone() -> JsonDict:
    return {
        "experiment": "experiment_5055_capstone_v464",
        "experiment_id": 5055,
        "honest_verdict": "complete_capstone_v464_execution_incomplete_fr11_no_credible_positive_evidence",
        "capstone_ready": True,
        "moat_state": "execution_incomplete",
    }


def _write_all(
    root: Path,
    *,
    moat_state: str = "execution_incomplete",
    promoted: bool = False,
    include_sota: bool = False,
) -> None:
    payloads = {
        "moat_gate": _moat(moat_state),
        "fr11_self_learning": _fr11(promoted=promoted),
        "hardware": _hardware(),
        "arc": _arc(),
        "prior_capstone": _prior_capstone(),
    }
    if include_sota:
        payloads["sota"] = _sota()
    for source, payload in payloads.items():
        _write_json(root / mod.UPSTREAMS[source].relative_path, payload)


def test_req_capstone_5068_spec_declares_v465_capstone_contract() -> None:
    """REQ-CAPSTONE-5068: OpenSpec anchors the .465 capstone artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-CAPSTONE-5068",
        "SCENARIO-CAPSTONE-5068",
        "SCENARIO-CAPSTONE-5068-FIELD-PRINCIPLES",
        "experiment_5068_capstone_v465.py",
        "results/experiment_5068_capstone_v465.json",
        "fr11_self_learning_result",
        "docs_update_required",
        "missing Exp5066",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_5068_preserves_current_incomplete_and_missing_sota(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-5068: missing SOTA is explicit and not fabricated."""

    _write_all(tmp_path, include_sota=False)

    artifact = mod.run(root=tmp_path, artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH)

    assert artifact["honest_verdict"] == (
        "complete_capstone_v465_execution_incomplete_fr11_no_credible_positive_evidence_missing_sota"
    )
    assert artifact["capstone_ready"] is False
    assert artifact["moat_state"] == "execution_incomplete"
    assert artifact["best_verifier_evidence"]["best_arm"] == "D6"
    assert artifact["best_verifier_evidence"]["headline_countable"] is False
    assert artifact["best_verifier_evidence"]["source_row"]["artifact_id"] == "D6"
    assert artifact["fr11_self_learning_result"]["credible_positive_evidence"] is False
    assert artifact["fr11_self_learning_result"]["state"] == "no_credible_positive_evidence"
    assert artifact["fr11_self_learning_result"]["no_promote_reason"] == (
        "heldout_delta_nonpositive;nonforgetting_regressed"
    )
    assert artifact["hardware_result"]["state"] == "packet_built"
    assert artifact["hardware_result"]["no_general_speedup_claim"] is True
    assert artifact["sota_result"]["state"] == "missing"
    assert artifact["sota_result"]["claim_boundary"] == "missing_artifact_no_sota_ingestion_claim"
    assert artifact["arc_result"]["state"] == "no_bank"
    assert artifact["arc_result"]["new_levels_banked"] == 0
    assert artifact["next_milestone_pointer"]["selected"]["experiment_class"] == (
        "execution_repair_before_claim_or_retirement"
    )
    assert artifact["next_milestone_pointer"]["sota_ingestion_missing"] is True
    assert artifact["missing_upstream_artifacts"] == [
        {
            "source": "sota",
            "experiment_id": 5066,
            "path": "results/experiment_5066_sota_ingestion_v466.json",
        }
    ]
    assert {row["source"] for row in artifact["cited_upstream_artifacts"]} == set(
        mod.UPSTREAMS
    )
    sota_citation = [
        row for row in artifact["cited_upstream_artifacts"] if row["source"] == "sota"
    ][0]
    assert sota_citation["status"] == "missing"
    assert sota_citation["sha256"] is None
    assert artifact["docs_update_required"]["deferred_by_stop_rule"] is True
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert mod.artifact_schema_errors(artifact) == []


@pytest.mark.parametrize(
    ("state", "label", "experiment_class"),
    [
        ("moat_realized", "realized_moat", "scale_realized_verifier"),
        ("musr_scoped_positive", "scoped_positive", "confirm_scoped_musr_positive"),
        (
            "second_corpus_scoped_positive",
            "scoped_positive",
            "repair_musr_or_cascade_for_second_corpus_positive",
        ),
        ("retired_bounded", "bounded_retirement", "pivot_to_new_verifier_direction"),
        (
            "execution_incomplete",
            "execution_incomplete",
            "execution_repair_before_claim_or_retirement",
        ),
    ],
)
def test_scenario_capstone_5068_routes_every_roadmap_moat_state(
    tmp_path: Path,
    state: str,
    label: str,
    experiment_class: str,
) -> None:
    """SCENARIO-CAPSTONE-5068: the .466 pointer follows the roadmap state table."""

    _write_all(tmp_path, moat_state=state, promoted=True, include_sota=True)

    artifact = mod.run(root=tmp_path, artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH)

    assert artifact["honest_verdict"] == f"complete_capstone_v465_{label}_fr11_credible_positive"
    assert artifact["capstone_ready"] is True
    assert artifact["next_milestone_pointer"]["selected_state"] == state
    assert artifact["next_milestone_pointer"]["selected"]["experiment_class"] == experiment_class
    assert artifact["next_milestone_pointer"]["candidate_classes"] == ["Fresh verifier route"]
    assert artifact["fr11_self_learning_result"]["credible_positive_evidence"] is True
    assert artifact["sota_result"]["state"] == "references_updated"
    assert artifact["missing_upstream_artifacts"] == []
    assert mod.artifact_schema_errors(artifact) == []


def test_req_capstone_5068_malformed_inputs_and_validation_edges(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-CAPSTONE-5068: malformed inputs and schema failures are explicit."""

    _write_all(tmp_path, include_sota=True)
    malformed = tmp_path / mod.UPSTREAMS["moat_gate"].relative_path
    malformed.write_text("{not-json", encoding="utf-8")

    loaded = mod.load_upstream_artifacts(tmp_path)
    artifact = mod.build_artifact(loaded, duration_s=0.0)

    assert artifact["capstone_ready"] is False
    assert artifact["moat_state"] == "execution_incomplete"
    assert artifact["malformed_upstream_artifacts"][0]["source"] == "moat_gate"
    assert artifact["duration_s"] == 0.0001
    non_object_path = tmp_path / "list.json"
    non_object_path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="top-level JSON value"):
        mod.read_json_object(non_object_path)
    assert mod._best_source_row({"best_arm": "D1", "per_arm_table": ["bad", {"artifact_id": "D1"}]}) == {
        "artifact_id": "D1"
    }
    assert mod._fr11_result({})["claim_boundary"] == "missing_artifact_no_fr11_claim"
    assert mod._hardware_result({})["claim_boundary"] == "missing_artifact_no_hardware_claim"
    assert mod._arc_result({})["claim_boundary"] == "missing_artifact_no_arc_claim"
    assert mod._number(True) is None
    assert mod._number("bad") is None
    assert mod._list("bad") == []
    assert mod._mapping([]) == {}
    assert "honest_verdict" in mod.artifact_schema_errors({})

    invalid = dict(artifact)
    invalid["honest_verdict"] = "bad"
    invalid["capstone_ready"] = "yes"
    invalid["moat_state"] = "bad"
    invalid["fr11_self_learning_result"] = []
    invalid["cited_upstream_artifacts"] = "bad"
    invalid["docs_update_required"] = []
    errors = mod.artifact_schema_errors(invalid)
    for field in (
        "honest_verdict",
        "capstone_ready",
        "moat_state",
        "fr11_self_learning_result",
        "cited_upstream_artifacts",
        "docs_update_required",
    ):
        assert field in errors

    _write_all(tmp_path, include_sota=True)
    exit_code = mod.main(root=tmp_path, artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH)
    captured = capsys.readouterr()
    printed = json.loads(captured.out)
    assert exit_code == 0
    assert printed["experiment"] == mod.EXPERIMENT
    assert printed["reproducibility_checksum"] == mod.payload_checksum(printed)
