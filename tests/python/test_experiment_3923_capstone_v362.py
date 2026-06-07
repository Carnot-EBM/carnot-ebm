"""Tests for Exp 3923 capstone v362 verifier scorecard.

Spec refs: REQ-CAPSTONE-3923, SCENARIO-CAPSTONE-3923.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import capstone_v362_3923 as exp3923


SPEC_PATH = Path("openspec/capabilities/capstone/spec.md")


def _write_json(root: Path, rel_path: str, payload: dict[str, object]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _publication_gate(*, paper_ready: bool = True) -> dict[str, object]:
    return {
        "paper_ready": paper_ready,
        "gates": {
            "G1": {"pass": True},
            "G2": {"pass": paper_ready},
            "G3": {"pass": True},
            "G4": {"pass": True},
        },
        "unmet_gates": [] if paper_ready else ["G2"],
    }


def _summary_statuses(*, live_critical: set[int] | None = None) -> dict[int, dict[str, object]]:
    critical = live_critical or set()
    return {
        experiment_id: {"returncode": 2 if experiment_id in critical else 0}
        for experiment_id in exp3923.UPSTREAM_IDS
    }


def _seed_v362_fixture(root: Path) -> None:
    _write_json(
        root,
        "results/experiment_3915_robust_gguf_inference_harness.json",
        {
            "honest_verdict": "complete: gguf_inference_harness_READY_modelgemma_live_path_unblocked",
            "flagged_adversarial": False,
            "unit_test_passed": True,
            "smoke_tokens": 1,
            "reproducibility_checksum": "a" * 64,
        },
    )
    _write_json(
        root,
        "results/experiment_3916_moat_scissor_accuracy.json",
        {
            "honest_verdict": "complete: moat_scissor_MOAT_SURVIVES_residcatch_strong0.9143",
            "flagged_adversarial": False,
            "reproducibility_checksum": "b" * 64,
        },
    )
    _write_json(
        root,
        "results/experiment_3917_efficiency_head_to_head.json",
        {
            "honest_verdict": "complete: efficiency_CHEAPER_11512.51x_but_NOT_PARITY_energy0.8100",
            "flagged_adversarial": False,
            "accuracy_parity": False,
            "cost_ratio_walltime": 11512.51,
            "energy_auroc": 0.8100,
            "llm_judge_auroc": 0.4423,
            "reproducibility_checksum": "c" * 64,
        },
    )
    _write_json(
        root,
        "results/experiment_3918_cascade_router_prototype.json",
        {
            "honest_verdict": "complete: cascade_router_WINS_gap-0.3896_11512.51x_cheaper",
            "flagged_adversarial": False,
            "cascade_cost_ratio": 11512.51,
            "reproducibility_checksum": "d" * 64,
        },
    )
    _write_json(
        root,
        "results/experiment_3919_arc_agi3_harness_scaffold.json",
        {
            "honest_verdict": "complete: arc_agi3_scaffold_READY_pruned8_synthetic_only",
            "flagged_adversarial": False,
            "unit_test_passed": True,
            "reproducibility_checksum": "e" * 64,
        },
    )
    _write_json(
        root,
        "results/experiment_3920_facts_graph_grounding_last_retry.json",
        {
            "honest_verdict": "blocked_llama_cpp_inference_failed",
            "flagged_adversarial": True,
            "reproducibility_checksum": "f" * 64,
        },
    )
    _write_json(
        root,
        "results/experiment_3921_fr11_v25_independence_reweighting.json",
        {
            "honest_verdict": "complete: fr11_v25_INVARIANT_HELD_auroc0.9078",
            "flagged_adversarial": False,
            "frozen_headline_unchanged": True,
            "frozen_headline_ensemble_auroc": 0.9131,
            "learned_ensemble_auroc_in_frozen_ci": True,
            "memory_ablation_contribution_min_met": True,
            "reproducibility_checksum": "1" * 64,
        },
    )
    _write_json(
        root,
        "results/experiment_3922_hardware_continuity_consolidated.json",
        {
            "honest_verdict": "success: hardware_continuity_gatemateblocked_no_fabric_claim",
            "flagged_adversarial": True,
            "fabric_acceleration_claimed": False,
            "reproducibility_checksum": "2" * 64,
        },
    )


def test_req_capstone_3923_spec_declares_v362_contract() -> None:
    """REQ-CAPSTONE-3923: OpenSpec anchors the v362 scorecard behavior."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-3923" in spec
    assert "SCENARIO-CAPSTONE-3923" in spec
    assert "verifier_earns_its_place" in spec
    assert "frozen 0.9131 headline unchanged" in spec


def test_req_capstone_3923_derivation_helpers_are_conditioned() -> None:
    """REQ-CAPSTONE-3923: helper verdicts use only landed clean inputs."""

    assert exp3923.numeric(True) is None
    assert exp3923.numeric("1.0") is None
    assert exp3923.derive_moat_verdict({"honest_verdict": "complete: MOAT_SURVIVES"}) == "MOAT_SURVIVES"
    assert exp3923.derive_moat_verdict({"honest_verdict": "complete: SUBSUMED"}) == "SUBSUMED"
    assert exp3923.derive_moat_verdict(None) == "INCONCLUSIVE"
    assert exp3923.derive_efficiency_cost_ratio(None) == 0.0
    assert exp3923.derive_efficiency_cost_ratio({"efficiency_cost_ratio": 3.0}) == 3.0
    assert exp3923.derive_efficiency_cost_ratio({"efficiency_cost_ratio": "bad"}) == 0.0
    assert exp3923.derive_efficiency_verdict(None) == "NOT_CHEAPER"
    assert exp3923.derive_efficiency_verdict({"accuracy_parity": True, "cost_ratio_walltime": 2.0}) == (
        "PARITY_AND_CHEAPER"
    )
    assert exp3923.derive_efficiency_verdict({"accuracy_parity": False, "cost_ratio_walltime": 2.0}) == (
        "CHEAPER_NOT_PARITY"
    )
    assert exp3923.derive_efficiency_verdict({"accuracy_parity": True, "cost_ratio_walltime": 0.9}) == "NOT_CHEAPER"
    assert exp3923.derive_efficiency_verdict(
        {"honest_verdict": "complete: efficiency_PARITY_AND_CHEAPER", "cost_ratio_walltime": 2.0}
    ) == "PARITY_AND_CHEAPER"
    assert exp3923.derive_verifier_earns_place("PARITY_AND_CHEAPER", 2.0) is True
    assert exp3923.derive_verifier_earns_place("CHEAPER_NOT_PARITY", 11512.51) is False
    assert exp3923.derive_cascade_verdict({"honest_verdict": "complete: cascade_router_WINS"}) == "WINS"
    assert exp3923.derive_cascade_verdict({"honest_verdict": "complete: cascade_router_close"}) == "MARGINAL"
    assert exp3923.derive_gguf_unblocked({"honest_verdict": "complete: READY_live_path_unblocked"}) is True
    assert exp3923.derive_arc_scaffold_ready({"honest_verdict": "complete: arc_agi3_scaffold_READY"}) is True
    assert exp3923.derive_facts_outcome(None, exp3920_was_flagged=True) == "EXCLUDED_EXP3920_FLAGGED"
    assert exp3923.derive_facts_outcome({"honest_verdict": "complete: graph_grounding_READY"}, exp3920_was_flagged=False) == "READY"
    assert exp3923.derive_fr11_v25_invariant({"honest_verdict": "complete: INVARIANT_HELD"}) == "INVARIANT_HELD"
    assert exp3923.derive_fr11_v25_invariant({"honest_verdict": "complete: partial"}) == "INCONCLUSIVE"
    assert exp3923.derive_hardware_outcome(None, exp3922_was_flagged=True) == "EXCLUDED_EXP3922_FLAGGED"
    assert exp3923.derive_hardware_outcome({"honest_verdict": "blocked_board"}, exp3922_was_flagged=False) == "BLOCKED"
    assert exp3923.derive_hardware_outcome(
        {"honest_verdict": "success: no_fabric_claim"},
        exp3922_was_flagged=False,
    ) == "TERMINAL_OR_CONTINUITY_NO_FABRIC_CLAIM"
    assert exp3923.derive_hardware_outcome(
        {"honest_verdict": "success: partial"},
        exp3922_was_flagged=False,
    ) == "PARTIAL_NO_FABRIC_CLAIM"
    assert exp3923.frozen_headline_unchanged({1: {"frozen_headline_ensemble_auroc": 0.902}}) is False
    assert exp3923.frozen_headline_unchanged({1: {"frozen_fover_auroc_unchanged": 0.902}}) is False


def test_scenario_capstone_3923_writes_conditioned_scorecard(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-3923: clean landed verdicts produce the v362 scorecard."""

    _seed_v362_fixture(tmp_path)
    artifact = exp3923.build_artifact(
        tmp_path,
        publication_gate_data=_publication_gate(),
        summary_statuses=_summary_statuses(live_critical={3920, 3922}),
        started_s=1.0,
        now_s=1.00005,
    )

    exp3923.validate_artifact(artifact)

    assert artifact["honest_verdict"] == (
        "complete: capstone_v362_moatMOAT_SURVIVES_efficiencyCHEAPER_NOT_PARITY_"
        "earnsfalse_paper_ready_true_frozen_unchanged"
    )
    assert artifact["moat_verdict"] == "MOAT_SURVIVES"
    assert artifact["efficiency_verdict"] == "CHEAPER_NOT_PARITY"
    assert artifact["efficiency_cost_ratio"] == pytest.approx(11512.51)
    assert artifact["cascade_verdict"] == "WINS"
    assert artifact["verifier_earns_its_place"] is False
    assert artifact["gguf_inference_unblocked"] is True
    assert artifact["arc_scaffold_ready"] is True
    assert artifact["facts_outcome"] == "EXCLUDED_EXP3920_FLAGGED"
    assert artifact["fr11_v25_invariant"] == "INVARIANT_HELD"
    assert artifact["hardware_outcome"] == "EXCLUDED_EXP3922_FLAGGED"
    assert artifact["both_energy_theses_bounded"] is True
    assert artifact["paper_ready"] is True
    assert artifact["unmet_gates"] == []
    assert artifact["frozen_headline_unchanged"] is True
    assert "ARC-AGI-3 agentic-proof venue" in artifact["operator_next_step_recommendation"]
    assert "loop recommends, the operator decides" in artifact["operator_next_step_recommendation"]
    assert {item["experiment_id"] for item in artifact["flagged_artifacts_excluded"]} == {3920, 3922}
    assert artifact["preconditions_checked"]["capstone_complete"] is True
    assert artifact["preconditions_checked"]["all_landed_nonflagged_verdicts_aggregated"] is True
    assert "GGUF" not in artifact["inference_substrate"]
    assert "CUDA" not in artifact["inference_substrate"]
    assert artifact["duration_s"] == pytest.approx(0.0001)
    assert exp3923.is_sha256(artifact["reproducibility_checksum"])

    for field in exp3923.STRING_VERDICT_FIELDS:
        assert isinstance(artifact[field], str)
        assert not isinstance(artifact[field], dict)
    for field in exp3923.BOOL_VERDICT_FIELDS:
        assert isinstance(artifact[field], bool)
        assert not isinstance(artifact[field], dict)

    output = exp3923.write_artifact(
        tmp_path,
        output_path="results/out.json",
        publication_gate_data=_publication_gate(),
        summary_statuses=_summary_statuses(live_critical={3920, 3922}),
        started_s=2.0,
        now_s=2.5,
    )
    saved = json.loads(output.read_text(encoding="utf-8"))
    exp3923.validate_artifact(saved)
    assert saved["honest_verdict"] == artifact["honest_verdict"]


def test_req_capstone_3923_blocks_when_publication_gate_or_headline_regresses(tmp_path: Path) -> None:
    """REQ-CAPSTONE-3923: CAPSTONE_COMPLETE requires paper-ready and frozen guards."""

    _seed_v362_fixture(tmp_path)
    gate_regressed = exp3923.build_artifact(
        tmp_path,
        publication_gate_data=_publication_gate(paper_ready=False),
        summary_statuses=_summary_statuses(live_critical={3920, 3922}),
        started_s=3.0,
        now_s=3.1,
    )

    exp3923.validate_artifact(gate_regressed)
    assert gate_regressed["honest_verdict"] == (
        "blocked_publication_gate: capstone_v362_moatMOAT_SURVIVES_efficiencyCHEAPER_NOT_PARITY_"
        "earnsfalse_paper_ready_false_frozen_unchanged"
    )
    assert gate_regressed["unmet_gates"] == ["G2"]

    fr11 = json.loads((tmp_path / "results/experiment_3921_fr11_v25_independence_reweighting.json").read_text())
    fr11["frozen_headline_unchanged"] = False
    fr11["frozen_headline_ensemble_auroc"] = 0.902
    _write_json(tmp_path, "results/experiment_3921_fr11_v25_independence_reweighting.json", fr11)
    frozen_regressed = exp3923.build_artifact(
        tmp_path,
        publication_gate_data=_publication_gate(),
        summary_statuses=_summary_statuses(live_critical={3920, 3922}),
        started_s=4.0,
        now_s=4.1,
    )

    exp3923.validate_artifact(frozen_regressed)
    assert frozen_regressed["honest_verdict"] == (
        "blocked_frozen_headline: capstone_v362_moatMOAT_SURVIVES_efficiencyCHEAPER_NOT_PARITY_"
        "earnsfalse_paper_ready_true_frozen_changed"
    )
    assert frozen_regressed["frozen_headline_unchanged"] is False


def test_req_capstone_3923_missing_upstreams_remain_honest(tmp_path: Path) -> None:
    """REQ-CAPSTONE-3923: partial landing records missing states without a false headline."""

    _write_json(
        tmp_path,
        "results/experiment_3917_efficiency_head_to_head.json",
        {
            "honest_verdict": "complete: efficiency_PARITY_AND_CHEAPER_4.00x",
            "flagged_adversarial": False,
            "accuracy_parity": True,
            "cost_ratio_walltime": 4.0,
            "reproducibility_checksum": "c" * 64,
        },
    )

    artifact = exp3923.build_artifact(
        tmp_path,
        publication_gate_data=_publication_gate(),
        summary_statuses=_summary_statuses(),
        started_s=5.0,
        now_s=5.1,
    )

    exp3923.validate_artifact(artifact)

    assert artifact["moat_verdict"] == "INCONCLUSIVE"
    assert artifact["efficiency_verdict"] == "PARITY_AND_CHEAPER"
    assert artifact["verifier_earns_its_place"] is True
    assert artifact["gguf_inference_unblocked"] is False
    assert artifact["arc_scaffold_ready"] is False
    assert artifact["preconditions_checked"]["upstream_artifacts"][3916]["exists"] is False
    assert artifact["preconditions_checked"]["capstone_complete"] is True
