"""Tests for Exp 3933 capstone v363 hardened verifier scorecard.

Spec refs: REQ-CAPSTONE-3933, SCENARIO-CAPSTONE-3933.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import capstone_v363_3933 as exp3933


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
        for experiment_id in exp3933.UPSTREAM_IDS
    }


def _seed_v363_fixture(root: Path) -> None:
    _write_json(
        root,
        "results/experiment_3924_archive_v362_activate_v363_retire_facts.json",
        {
            "honest_verdict": "complete: archived_v362_v363_active_facts_retired",
            "flagged_adversarial": False,
            "reproducibility_checksum": "1" * 64,
        },
    )
    _write_json(
        root,
        "results/experiment_3925_competent_judge_build.json",
        {
            "honest_verdict": "complete: competent_judge_positive_control_passed",
            "flagged_adversarial": False,
            "judge_positive_control_passed": True,
            "reproducibility_checksum": "2" * 64,
        },
    )
    _write_json(
        root,
        "results/experiment_3926_valid_efficiency_head_to_head.json",
        {
            "honest_verdict": "complete: efficiency_VALID_EARNS_PLACE_12.50x_cheaper_vs_competent_judge",
            "flagged_adversarial": False,
            "judge_positive_control_passed": True,
            "accuracy_parity": True,
            "pareto_dominates": False,
            "cost_ratio_walltime": 12.5,
            "reproducibility_checksum": "3" * 64,
        },
    )
    _write_json(
        root,
        "results/experiment_3927_non_degenerate_cascade_router.json",
        {
            "honest_verdict": "complete: cascade_router_WINS_escfrac0.2500_gap0.0100_6.00x_cheaper_at_matched_accuracy_non_degenerate",
            "flagged_adversarial": False,
            "escalation_fraction": 0.25,
            "cascade_degenerate": False,
            "cascade_cost_ratio": 6.0,
            "frozen_fover_auroc_unchanged": 0.9131,
            "reproducibility_checksum": "4" * 64,
        },
    )
    _write_json(
        root,
        "results/experiment_3928_moat_scissor_replication.json",
        {
            "honest_verdict": "complete: moat_scissor_REPLICATES_on_processbench_slice",
            "flagged_adversarial": False,
            "moat_replicates": True,
            "frozen_fover_auroc_unchanged": 0.9131,
            "reproducibility_checksum": "5" * 64,
        },
    )
    _write_json(
        root,
        "results/experiment_3929_arc_agi3_action_efficiency.json",
        {
            "honest_verdict": "complete: arc_agi3_verifier_router_HELPS_ratio1.959",
            "flagged_adversarial": False,
            "action_efficiency_ratio": 1.9591836734693875,
            "reproducibility_checksum": "6" * 64,
        },
    )
    _write_json(
        root,
        "results/experiment_3930_fr11_v26_cascade_band_online_learning.json",
        {
            "honest_verdict": "complete: fr11_v26_INVARIANT_HELD_auroc0.908",
            "flagged_adversarial": False,
            "frozen_headline_unchanged": True,
            "frozen_headline_ensemble_auroc": 0.9131,
            "learned_ensemble_auroc_in_frozen_ci": True,
            "memory_ablation_contribution_min_met": True,
            "reproducibility_checksum": "7" * 64,
        },
    )
    _write_json(
        root,
        "results/experiment_3931_hardware_continuity_clean_rerun.json",
        {
            "honest_verdict": "success: hardware_continuity_clean_gatemateblocked_pfterminal_hash_verified_kvnonterminal_no_fabric_claim",
            "flagged_adversarial": False,
            "fabric_acceleration_claimed": False,
            "reproducibility_checksum": "8" * 64,
        },
    )
    _write_json(
        root,
        "results/experiment_3932_literature_synthesis_agentic_verification.json",
        {
            "honest_verdict": "complete: literature_synthesis_positioned_0_new_refs_public_docs_untouched",
            "flagged_adversarial": False,
            "reproducibility_checksum": "9" * 64,
        },
    )


def test_req_capstone_3933_spec_declares_v363_contract() -> None:
    """REQ-CAPSTONE-3933: OpenSpec anchors the v363 scorecard behavior."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-3933" in spec
    assert "SCENARIO-CAPSTONE-3933" in spec
    assert "hardened verifier scorecard" in spec
    assert "frozen 0.9131 headline unchanged" in spec


def test_req_capstone_3933_derivation_helpers_are_conditioned() -> None:
    """REQ-CAPSTONE-3933: helper verdicts only use landed clean inputs."""

    assert exp3933.numeric(True) is None
    assert exp3933.numeric("12.5") is None
    assert exp3933.derive_judge_was_competent({"judge_positive_control_passed": True}) is True
    assert exp3933.derive_judge_was_competent({"honest_verdict": "complete: positive_control_passed"}) is True
    assert exp3933.derive_judge_was_competent(None) is False
    assert exp3933.derive_efficiency_cost_ratio(None) == 0.0
    assert exp3933.derive_efficiency_cost_ratio({"cost_ratio_walltime": 12.5}) == 12.5
    assert exp3933.derive_efficiency_cost_ratio({"efficiency_cost_ratio": 7.5}) == 7.5
    assert exp3933.derive_efficiency_cost_ratio({"efficiency_cost_ratio": "bad"}) == 0.0
    assert exp3933.derive_efficiency_verdict(None) == "INCONCLUSIVE"
    assert exp3933.derive_efficiency_verdict({"judge_positive_control_passed": False, "cost_ratio_walltime": 99.0}) == "INCONCLUSIVE"
    assert exp3933.derive_efficiency_verdict(
        {"judge_positive_control_passed": True, "accuracy_parity": True, "cost_ratio_walltime": 12.5}
    ) == "VALID_EARNS_PLACE"
    assert exp3933.derive_efficiency_verdict(
        {"judge_positive_control_passed": True, "pareto_dominates": True, "cost_ratio_walltime": 12.5}
    ) == "VALID_EARNS_PLACE"
    assert exp3933.derive_efficiency_verdict(
        {"judge_positive_control_passed": True, "accuracy_parity": False, "cost_ratio_walltime": 12.5}
    ) == "CHEAPER_BUT_LESS_ACCURATE"
    assert exp3933.derive_efficiency_verdict(
        {"judge_positive_control_passed": True, "accuracy_parity": True, "cost_ratio_walltime": 9.9}
    ) == "INCONCLUSIVE"
    assert exp3933.derive_efficiency_verdict(
        {
            "honest_verdict": "complete: efficiency_VALID_EARNS_PLACE",
            "judge_positive_control_passed": True,
            "cost_ratio_walltime": 12.5,
        }
    ) == "VALID_EARNS_PLACE"
    assert exp3933.derive_verifier_earns_place("VALID_EARNS_PLACE", 12.5, judge_was_competent=True) is True
    assert exp3933.derive_verifier_earns_place("VALID_EARNS_PLACE", 12.5, judge_was_competent=False) is False
    assert exp3933.derive_moat_replicated({"moat_replicates": True}) is True
    assert exp3933.derive_moat_replicated({"honest_verdict": "complete: MOAT_REPLICATES"}) is True
    assert exp3933.derive_cascade_verdict({"honest_verdict": "complete: cascade_router_WINS", "escalation_fraction": 0.2}) == "WINS_NON_DEGENERATE"
    assert exp3933.derive_cascade_verdict({"honest_verdict": "complete: cascade_router_MARGINAL", "escalation_fraction": 0.0}) == "MARGINAL_DEGENERATE"
    assert exp3933.derive_arc_agentic_advantage({"action_efficiency_ratio": 1.959}) == pytest.approx(1.959)
    assert exp3933.derive_fr11_v26_invariant(None) == "INCONCLUSIVE"
    assert exp3933.derive_fr11_v26_invariant({"honest_verdict": "complete: fr11_v26_INVARIANT_HELD"}) == "INVARIANT_HELD"
    assert exp3933.derive_fr11_v26_invariant({"honest_verdict": "complete: fr11_v26_INVARIANT_BROKEN"}) == "INVARIANT_BROKEN"
    assert exp3933.derive_fr11_v26_invariant({"honest_verdict": "complete: fr11_v26_partial"}) == "INCONCLUSIVE"
    assert exp3933.derive_hardware_outcome({"honest_verdict": "success: no_fabric_claim"}) == "TERMINAL_OR_CONTINUITY_NO_FABRIC_CLAIM"
    assert exp3933.derive_hardware_outcome({"honest_verdict": "blocked_board"}) == "BLOCKED"
    assert exp3933.derive_hardware_outcome({"honest_verdict": "success: partial"}) == "PARTIAL_NO_FABRIC_CLAIM"
    assert exp3933.derive_facts_route_retired({"honest_verdict": "complete: facts_retired"}) is True
    assert exp3933.derive_facts_route_retired(None) is False
    assert exp3933.frozen_headline_unchanged({1: {"frozen_headline_unchanged": False}}) is False
    assert exp3933.frozen_headline_unchanged({1: {"frozen_headline_ensemble_auroc": 0.902}}) is False
    assert exp3933.frozen_headline_unchanged({1: {"frozen_fover_auroc_unchanged": 0.902}}) is False


def test_scenario_capstone_3933_writes_hardened_scorecard(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-3933: clean landed verdicts produce the hardened scorecard."""

    _seed_v363_fixture(tmp_path)
    artifact = exp3933.build_artifact(
        tmp_path,
        publication_gate_data=_publication_gate(),
        summary_statuses=_summary_statuses(),
        started_s=1.0,
        now_s=1.00005,
    )

    exp3933.validate_artifact(artifact)

    assert artifact["honest_verdict"] == (
        "complete: capstone_v363_efficiencyVALID_EARNS_PLACE_"
        "moat_replicatedtrue_earnstrue_paper_ready_true_frozen_unchanged"
    )
    assert artifact["efficiency_verdict"] == "VALID_EARNS_PLACE"
    assert artifact["efficiency_cost_ratio"] == pytest.approx(12.5)
    assert artifact["judge_was_competent"] is True
    assert artifact["moat_replicated"] is True
    assert artifact["cascade_verdict"] == "WINS_NON_DEGENERATE"
    assert artifact["verifier_earns_its_place"] is True
    assert artifact["arc_agentic_advantage"] == pytest.approx(1.9591836734693875)
    assert artifact["fr11_v26_invariant"] == "INVARIANT_HELD"
    assert artifact["hardware_outcome"] == "TERMINAL_OR_CONTINUITY_NO_FABRIC_CLAIM"
    assert artifact["facts_route_retired"] is True
    assert artifact["both_energy_theses_bounded"] is True
    assert artifact["paper_ready"] is True
    assert artifact["unmet_gates"] == []
    assert artifact["frozen_headline_unchanged"] is True
    assert "scale the ARC-AGI-3 agentic-proof venue" in artifact["operator_next_step_recommendation"]
    assert "loop recommends, the operator decides" in artifact["operator_next_step_recommendation"]
    assert artifact["flagged_artifacts_excluded"] == []
    assert artifact["preconditions_checked"]["capstone_complete"] is True
    assert artifact["preconditions_checked"]["all_landed_nonflagged_verdicts_aggregated"] is True
    assert "GGUF" not in artifact["inference_substrate"]
    assert "CUDA" not in artifact["inference_substrate"]
    assert artifact["duration_s"] == pytest.approx(0.0001)
    assert exp3933.is_sha256(artifact["reproducibility_checksum"])

    for field in exp3933.STRING_VERDICT_FIELDS:
        assert isinstance(artifact[field], str)
        assert not isinstance(artifact[field], dict)
    for field in exp3933.BOOL_VERDICT_FIELDS:
        assert isinstance(artifact[field], bool)
        assert not isinstance(artifact[field], dict)

    output = exp3933.write_artifact(
        tmp_path,
        output_path="results/out.json",
        publication_gate_data=_publication_gate(),
        summary_statuses=_summary_statuses(),
        started_s=2.0,
        now_s=2.5,
    )
    saved = json.loads(output.read_text(encoding="utf-8"))
    exp3933.validate_artifact(saved)
    assert saved["honest_verdict"] == artifact["honest_verdict"]


def test_req_capstone_3933_excludes_flagged_efficiency_and_moat_inputs(tmp_path: Path) -> None:
    """REQ-CAPSTONE-3933: flagged upstreams cannot supply scorecard numbers."""

    _seed_v363_fixture(tmp_path)
    flagged_efficiency = {
        "honest_verdict": "complete: efficiency_VALID_EARNS_PLACE_99.0x",
        "flagged_adversarial": True,
        "judge_positive_control_passed": True,
        "accuracy_parity": True,
        "cost_ratio_walltime": 99.0,
        "reproducibility_checksum": "a" * 64,
    }
    flagged_moat = {
        "honest_verdict": "complete: moat_scissor_REPLICATES_on_processbench_slice",
        "flagged_adversarial": True,
        "moat_replicates": True,
        "reproducibility_checksum": "b" * 64,
    }
    _write_json(root=tmp_path, rel_path="results/experiment_3926_valid_efficiency_head_to_head.json", payload=flagged_efficiency)
    _write_json(root=tmp_path, rel_path="results/experiment_3928_moat_scissor_replication.json", payload=flagged_moat)

    artifact = exp3933.build_artifact(
        tmp_path,
        publication_gate_data=_publication_gate(),
        summary_statuses=_summary_statuses(live_critical={3926, 3928}),
        started_s=3.0,
        now_s=3.2,
    )

    exp3933.validate_artifact(artifact)

    assert artifact["efficiency_verdict"] == "INCONCLUSIVE"
    assert artifact["efficiency_cost_ratio"] == 0.0
    assert artifact["judge_was_competent"] is True
    assert artifact["moat_replicated"] is False
    assert artifact["verifier_earns_its_place"] is False
    assert artifact["honest_verdict"] == (
        "complete: capstone_v363_efficiencyINCONCLUSIVE_"
        "moat_replicatedfalse_earnsfalse_paper_ready_true_frozen_unchanged"
    )
    assert {item["experiment_id"] for item in artifact["flagged_artifacts_excluded"]} == {3926, 3928}
    assert artifact["preconditions_checked"]["live_critical_artifacts_observed"] == [3926, 3928]
    assert artifact["preconditions_checked"]["upstream_artifacts"][3926]["included"] is False
    assert artifact["preconditions_checked"]["upstream_artifacts"][3928]["included"] is False


def test_req_capstone_3933_missing_upstreams_and_gate_blocks_are_honest(tmp_path: Path) -> None:
    """REQ-CAPSTONE-3933: partial landing and gate regressions do not create a headline."""

    _write_json(
        tmp_path,
        "results/experiment_3924_archive_v362_activate_v363_retire_facts.json",
        {"honest_verdict": "complete: facts_retired", "flagged_adversarial": False},
    )
    _write_json(
        tmp_path,
        "results/experiment_3930_fr11_v26_cascade_band_online_learning.json",
        {
            "honest_verdict": "complete: fr11_v26_INVARIANT_HELD",
            "flagged_adversarial": False,
            "frozen_headline_unchanged": True,
            "frozen_headline_ensemble_auroc": 0.9131,
        },
    )

    partial = exp3933.build_artifact(
        tmp_path,
        publication_gate_data=_publication_gate(),
        summary_statuses=_summary_statuses(),
        started_s=4.0,
        now_s=4.1,
    )
    exp3933.validate_artifact(partial)
    assert partial["honest_verdict"] == (
        "complete: capstone_v363_efficiencyINCONCLUSIVE_"
        "moat_replicatedfalse_earnsfalse_paper_ready_true_frozen_unchanged"
    )
    assert partial["preconditions_checked"]["upstream_artifacts"][3925]["exists"] is False

    gate_regressed = exp3933.build_artifact(
        tmp_path,
        publication_gate_data=_publication_gate(paper_ready=False),
        summary_statuses=_summary_statuses(),
        started_s=5.0,
        now_s=5.1,
    )
    exp3933.validate_artifact(gate_regressed)
    assert gate_regressed["honest_verdict"].startswith("blocked_publication_gate:")
    assert gate_regressed["unmet_gates"] == ["G2"]

    fr11 = json.loads((tmp_path / "results/experiment_3930_fr11_v26_cascade_band_online_learning.json").read_text())
    fr11["frozen_headline_ensemble_auroc"] = 0.902
    _write_json(tmp_path, "results/experiment_3930_fr11_v26_cascade_band_online_learning.json", fr11)
    frozen_regressed = exp3933.build_artifact(
        tmp_path,
        publication_gate_data=_publication_gate(),
        summary_statuses=_summary_statuses(),
        started_s=6.0,
        now_s=6.1,
    )
    exp3933.validate_artifact(frozen_regressed)
    assert frozen_regressed["honest_verdict"].startswith("blocked_frozen_headline:")
    assert frozen_regressed["frozen_headline_unchanged"] is False
