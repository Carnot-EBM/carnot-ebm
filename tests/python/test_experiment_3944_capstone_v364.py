"""Tests for Exp 3944 capstone v364 hardened verifier scorecard.

Spec refs: REQ-CAPSTONE-3944, SCENARIO-CAPSTONE-3944.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import capstone_v364_3944 as exp3944


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


def _summary_statuses() -> dict[int, dict[str, object]]:
    return {experiment_id: {"returncode": 0} for experiment_id in exp3944.UPSTREAM_IDS}


def _seed_v364_fixture(root: Path) -> None:
    _write_json(
        root,
        "results/experiment_3935_competent_judge_build.json",
        {
            "honest_verdict": "complete: competent_judge_READY_positive_control_passed",
            "flagged_adversarial": False,
            "judge_positive_control_passed": True,
            "reproducibility_checksum": "1" * 64,
        },
    )
    _write_json(
        root,
        "results/experiment_3936_valid_efficiency_head_to_head.json",
        {
            "honest_verdict": "complete: valid_efficiency_PARITY_PARETO_energy_12.4x_cheaper",
            "flagged_adversarial": False,
            "parity_or_pareto_landed": True,
            "energy_cheaper_than_competent_judge_x": 12.4,
            "accuracy_parity": True,
            "reproducibility_checksum": "2" * 64,
        },
    )
    _write_json(
        root,
        "results/experiment_3937_non_degenerate_cascade_router.json",
        {
            "honest_verdict": "complete: cascade_router_WINS_non_degenerate",
            "flagged_adversarial": False,
            "non_degenerate_cascade": True,
            "escalation_fraction": 0.21,
            "reproducibility_checksum": "3" * 64,
        },
    )
    _write_json(
        root,
        "results/experiment_3938_moat_replication.json",
        {
            "honest_verdict": "complete: independent_corpus_moat_replicated",
            "flagged_adversarial": False,
            "moat_replicates": True,
            "independent_corpus_moat": True,
            "reproducibility_checksum": "4" * 64,
        },
    )
    _write_json(
        root,
        "results/experiment_3939_arc_agi3_step2.json",
        {
            "honest_verdict": "complete: arc_agi3_step2_verifier_vs_learned_value_ratio2.18",
            "flagged_adversarial": False,
            "action_efficiency_ratio": 2.18,
            "reproducibility_checksum": "5" * 64,
        },
    )
    _write_json(
        root,
        "results/experiment_3940_fr11_v27.json",
        {
            "honest_verdict": "complete: fr11_v27_INVARIANT_HELD",
            "flagged_adversarial": False,
            "frozen_headline_unchanged": True,
            "frozen_headline_ensemble_auroc": 0.9131,
            "reproducibility_checksum": "6" * 64,
        },
    )
    _write_json(
        root,
        "results/experiment_3941_hardware_continuity.json",
        {
            "honest_verdict": "success: hardware_continuity_gatemate_blocked_polarfire_terminal_kv260_nonterminal_no_fabric_claim",
            "flagged_adversarial": False,
            "fabric_acceleration_claimed": False,
            "reproducibility_checksum": "7" * 64,
        },
    )
    _write_json(
        root,
        "results/experiment_3942_cross_domain_map.json",
        {
            "honest_verdict": "complete: cross_domain_map_energy_moat_holds_process_collapses_factual",
            "flagged_adversarial": False,
            "cross_domain_boundary": "HOLDS_PROCESS_COLLAPSES_FACTUAL",
            "reproducibility_checksum": "8" * 64,
        },
    )
    _write_json(
        root,
        "results/experiment_3943_literature_synthesis.json",
        {
            "honest_verdict": "complete: literature_synthesis_positioned_0_new_refs_public_docs_untouched",
            "flagged_adversarial": False,
            "reproducibility_checksum": "9" * 64,
        },
    )


def test_req_capstone_3944_spec_declares_v364_contract() -> None:
    """REQ-CAPSTONE-3944: OpenSpec anchors the v364 scorecard behavior."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-3944" in spec
    assert "SCENARIO-CAPSTONE-3944" in spec
    assert "hardened verifier scorecard" in spec
    assert "frozen 0.9131 headline unchanged" in spec


def test_req_capstone_3944_derivation_helpers_are_conditioned() -> None:
    """REQ-CAPSTONE-3944: helper verdicts only use clean landed inputs."""

    assert exp3944.numeric(True) is None
    assert exp3944.numeric("12.5") is None
    assert exp3944.is_terminal_landed({"honest_verdict": "complete: ok"}) is True
    assert exp3944.is_terminal_landed({"honest_verdict": "success: ok"}) is True
    assert exp3944.is_terminal_landed({"honest_verdict": "failure: measured_negative"}) is True
    assert exp3944.is_terminal_landed({"honest_verdict": "blocked_gpu"}) is False
    assert exp3944.is_terminal_landed(None) is False
    assert exp3944.derive_judge_was_competent({"judge_positive_control_passed": True}) is True
    assert exp3944.derive_judge_was_competent({"honest_verdict": "complete: positive_control_passed"}) is True
    assert exp3944.derive_judge_was_competent(
        {"honest_verdict": "complete: competent_judge_READY_fixture_auroc1.0000_valid_comparator_landed"}
    ) is True
    assert exp3944.derive_judge_was_competent(None) is False
    assert exp3944.derive_efficiency_cost_ratio(None) == 0.0
    assert exp3944.derive_efficiency_cost_ratio({"energy_cheaper_than_competent_judge_x": 12.4}) == 12.4
    assert exp3944.derive_efficiency_cost_ratio({"efficiency_cost_ratio": 7.5}) == 7.5
    assert exp3944.derive_efficiency_cost_ratio({"cost_ratio_walltime": "bad"}) == 0.0
    assert exp3944.derive_efficiency_verdict(None, judge_was_competent=True) == "INCONCLUSIVE"
    assert exp3944.derive_efficiency_verdict(
        {"parity_or_pareto_landed": True, "energy_cheaper_than_competent_judge_x": 12.4},
        judge_was_competent=False,
    ) == "INCONCLUSIVE"
    assert exp3944.derive_efficiency_verdict(
        {"parity_or_pareto_landed": True, "energy_cheaper_than_competent_judge_x": 12.4},
        judge_was_competent=True,
    ) == "VALID_EARNS_PLACE"
    assert exp3944.derive_efficiency_verdict(
        {"pareto_dominates": True, "energy_cheaper_than_competent_judge_x": 12.4},
        judge_was_competent=True,
    ) == "VALID_EARNS_PLACE"
    assert exp3944.derive_efficiency_verdict(
        {"accuracy_parity": False, "energy_cheaper_than_competent_judge_x": 12.4},
        judge_was_competent=True,
    ) == "CHEAPER_BUT_LESS_ACCURATE"
    assert exp3944.derive_efficiency_verdict(
        {"accuracy_parity": True, "energy_cheaper_than_competent_judge_x": 1.2},
        judge_was_competent=True,
    ) == "INCONCLUSIVE"
    assert exp3944.derive_verifier_earns_place("VALID_EARNS_PLACE", 12.4, judge_was_competent=True) is True
    assert exp3944.derive_verifier_earns_place("VALID_EARNS_PLACE", 12.4, judge_was_competent=False) is False
    assert exp3944.derive_moat_replicated({"moat_replicates": True}) is True
    assert exp3944.derive_moat_replicated({"honest_verdict": "complete: MOAT_REPLICATED"}) is True
    assert exp3944.derive_cascade_verdict({"honest_verdict": "complete: cascade_WINS", "escalation_fraction": 0.2}) == "WINS_ESCALATION_GT_0"
    assert exp3944.derive_cascade_verdict({"honest_verdict": "complete: cascade_WINS", "non_degenerate_cascade": True}) == "WINS_ESCALATION_GT_0"
    assert exp3944.derive_cascade_verdict({"honest_verdict": "complete: cascade_MARGINAL", "escalation_fraction": 0.0}) == "MARGINAL_DEGENERATE"
    assert exp3944.derive_arc_agentic_advantage_vs_learned_value({"action_efficiency_ratio": 2.18}) == pytest.approx(2.18)
    assert exp3944.derive_arc_agentic_advantage_vs_learned_value({"verifier_vs_learned_value_action_efficiency": 1.5}) == pytest.approx(1.5)
    assert exp3944.derive_arc_agentic_advantage_vs_learned_value({"action_efficiency_ratio": "bad"}) == 0.0
    assert exp3944.derive_fr11_v27_invariant(None) == "INCONCLUSIVE"
    assert exp3944.derive_fr11_v27_invariant({"honest_verdict": "complete: fr11_v27_INVARIANT_HELD"}) == "INVARIANT_HELD"
    assert exp3944.derive_fr11_v27_invariant({"honest_verdict": "complete: fr11_v27_INVARIANT_BROKEN"}) == "INVARIANT_BROKEN"
    assert exp3944.derive_fr11_v27_invariant({"honest_verdict": "complete: fr11_v27_partial"}) == "INCONCLUSIVE"
    assert exp3944.derive_cross_domain_boundary({"cross_domain_boundary": "HOLDS_PROCESS"}) == "HOLDS_PROCESS"
    assert exp3944.derive_cross_domain_boundary({"honest_verdict": "complete: energy_moat_holds_code_collapses_facts"}) == "HOLDS_CODE_COLLAPSES_FACTS"
    assert exp3944.derive_cross_domain_boundary({"honest_verdict": "complete: cross_domain_partial"}) == "INCONCLUSIVE"
    assert exp3944.derive_hardware_outcome({"honest_verdict": "success: no_fabric_claim"}) == "TERMINAL_OR_CONTINUITY_NO_FABRIC_CLAIM"
    assert exp3944.derive_hardware_outcome({"honest_verdict": "blocked_board"}) == "BLOCKED"
    assert exp3944.derive_hardware_outcome({"honest_verdict": "success: partial"}) == "PARTIAL_NO_FABRIC_CLAIM"
    assert exp3944.frozen_headline_unchanged({1: {"frozen_headline_unchanged": False}}) is False
    assert exp3944.frozen_headline_unchanged({1: {"frozen_headline_ensemble_auroc": 0.902}}) is False
    assert exp3944.frozen_headline_unchanged({1: {"frozen_fover_auroc_unchanged": 0.9131}}) is True
    assert exp3944.frozen_headline_unchanged({1: {"frozen_fover_auroc_unchanged": 0.902}}) is False


def test_scenario_capstone_3944_writes_scorecard_that_earns_place(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-3944: clean landed verdicts produce the scorecard."""

    _seed_v364_fixture(tmp_path)
    artifact = exp3944.build_artifact(
        tmp_path,
        publication_gate_data=_publication_gate(),
        summary_statuses=_summary_statuses(),
        started_s=1.0,
        now_s=1.00005,
    )

    exp3944.validate_artifact(artifact)

    assert artifact["honest_verdict"] == (
        "complete: capstone_v364_efficiencyVALID_EARNS_PLACE_"
        "moat_replicatedtrue_earnstrue_paper_ready_true_frozen_unchanged"
    )
    assert artifact["judge_was_competent"] is True
    assert artifact["efficiency_verdict"] == "VALID_EARNS_PLACE"
    assert artifact["efficiency_cost_ratio"] == pytest.approx(12.4)
    assert artifact["moat_replicated"] is True
    assert artifact["cascade_verdict"] == "WINS_ESCALATION_GT_0"
    assert artifact["verifier_earns_its_place"] is True
    assert artifact["arc_agentic_advantage_vs_learned_value"] == pytest.approx(2.18)
    assert artifact["fr11_v27_invariant"] == "INVARIANT_HELD"
    assert artifact["cross_domain_boundary"] == "HOLDS_PROCESS_COLLAPSES_FACTUAL"
    assert artifact["hardware_outcome"] == "TERMINAL_OR_CONTINUITY_NO_FABRIC_CLAIM"
    assert artifact["both_energy_theses_bounded"] is True
    assert artifact["paper_ready"] is True
    assert artifact["unmet_gates"] == []
    assert artifact["frozen_headline_unchanged"] is True
    assert "verifier earns its place" in artifact["operator_next_step_recommendation"]
    assert "scale the ARC-AGI-3 agentic-proof venue" in artifact["operator_next_step_recommendation"]
    assert "loop recommends, the operator decides" in artifact["operator_next_step_recommendation"]
    assert artifact["flagged_artifacts_excluded"] == []
    assert artifact["preconditions_checked"]["capstone_complete"] is True
    assert artifact["preconditions_checked"]["all_landed_nonflagged_verdicts_aggregated"] is True
    assert "GGUF" not in artifact["inference_substrate"]
    assert "CUDA" not in artifact["inference_substrate"]
    assert artifact["duration_s"] == pytest.approx(0.0001)
    assert exp3944.is_sha256(artifact["reproducibility_checksum"])

    for field in exp3944.STRING_VERDICT_FIELDS:
        assert isinstance(artifact[field], str)
        assert not isinstance(artifact[field], dict)
    for field in exp3944.BOOL_VERDICT_FIELDS:
        assert isinstance(artifact[field], bool)
        assert not isinstance(artifact[field], dict)

    output = exp3944.write_artifact(
        tmp_path,
        output_path="results/out.json",
        publication_gate_data=_publication_gate(),
        summary_statuses=_summary_statuses(),
        started_s=2.0,
        now_s=2.5,
    )
    saved = json.loads(output.read_text(encoding="utf-8"))
    exp3944.validate_artifact(saved)
    assert saved["honest_verdict"] == artifact["honest_verdict"]


def test_req_capstone_3944_excludes_flagged_inputs_before_metrics(tmp_path: Path) -> None:
    """REQ-CAPSTONE-3944: flagged upstreams cannot supply scorecard numbers."""

    _seed_v364_fixture(tmp_path)
    _write_json(
        tmp_path,
        "results/experiment_3936_valid_efficiency_head_to_head.json",
        {
            "honest_verdict": "complete: valid_efficiency_PARITY_PARETO_energy_99.0x_cheaper",
            "flagged_adversarial": True,
            "parity_or_pareto_landed": True,
            "energy_cheaper_than_competent_judge_x": 99.0,
            "reproducibility_checksum": "a" * 64,
        },
    )
    _write_json(
        tmp_path,
        "results/experiment_3938_moat_replication.json",
        {
            "honest_verdict": "complete: independent_corpus_moat_replicated",
            "flagged_adversarial": True,
            "moat_replicates": True,
            "reproducibility_checksum": "b" * 64,
        },
    )

    artifact = exp3944.build_artifact(
        tmp_path,
        publication_gate_data=_publication_gate(),
        summary_statuses=_summary_statuses(),
        started_s=3.0,
        now_s=3.2,
    )

    exp3944.validate_artifact(artifact)
    assert artifact["efficiency_verdict"] == "INCONCLUSIVE"
    assert artifact["efficiency_cost_ratio"] == 0.0
    assert artifact["judge_was_competent"] is True
    assert artifact["moat_replicated"] is False
    assert artifact["verifier_earns_its_place"] is False
    assert artifact["honest_verdict"] == (
        "complete: capstone_v364_efficiencyINCONCLUSIVE_"
        "moat_replicatedfalse_earnsfalse_paper_ready_true_frozen_unchanged"
    )
    assert {item["experiment_id"] for item in artifact["flagged_artifacts_excluded"]} == {3936, 3938}
    assert artifact["preconditions_checked"]["upstream_artifacts"][3936]["included"] is False
    assert artifact["preconditions_checked"]["upstream_artifacts"][3938]["included"] is False


def test_req_capstone_3944_missing_upstreams_and_gate_blocks_are_honest(tmp_path: Path) -> None:
    """REQ-CAPSTONE-3944: partial landing and gate regressions do not create a headline."""

    _write_json(
        tmp_path,
        "results/experiment_3935_competent_judge_build.json",
        {
            "honest_verdict": "complete: competent_judge_READY_positive_control_passed",
            "flagged_adversarial": False,
            "judge_positive_control_passed": True,
        },
    )
    _write_json(
        tmp_path,
        "results/experiment_3943_literature_synthesis.json",
        {
            "honest_verdict": "complete: literature_synthesis_positioned_0_new_refs_public_docs_untouched",
            "flagged_adversarial": False,
        },
    )

    partial = exp3944.build_artifact(
        tmp_path,
        publication_gate_data=_publication_gate(),
        summary_statuses=_summary_statuses(),
        started_s=4.0,
        now_s=4.1,
    )
    exp3944.validate_artifact(partial)
    assert partial["honest_verdict"] == (
        "complete: capstone_v364_efficiencyINCONCLUSIVE_"
        "moat_replicatedfalse_earnsfalse_paper_ready_true_frozen_unchanged"
    )
    assert partial["judge_was_competent"] is True
    assert partial["preconditions_checked"]["upstream_artifacts"][3936]["exists"] is False
    assert partial["preconditions_checked"]["upstream_artifacts"][3936]["honest_verdict"] == "missing"
    assert partial["preconditions_checked"]["capstone_complete"] is True

    _write_json(
        tmp_path,
        "results/experiment_3940_fr11_v27.json",
        {
            "honest_verdict": "blocked_resource: fr11_v27_not_landed",
            "flagged_adversarial": False,
            "frozen_headline_ensemble_auroc": 0.9131,
        },
    )
    blocked = exp3944.build_artifact(
        tmp_path,
        publication_gate_data=_publication_gate(),
        summary_statuses=_summary_statuses(),
        started_s=4.0,
        now_s=4.1,
    )
    exp3944.validate_artifact(blocked)
    assert blocked["preconditions_checked"]["upstream_artifacts"][3940]["exists"] is True
    assert blocked["preconditions_checked"]["upstream_artifacts"][3940]["landed"] is False
    assert blocked["fr11_v27_invariant"] == "INCONCLUSIVE"

    gate_regressed = exp3944.build_artifact(
        tmp_path,
        publication_gate_data=_publication_gate(paper_ready=False),
        summary_statuses=_summary_statuses(),
        started_s=5.0,
        now_s=5.1,
    )
    exp3944.validate_artifact(gate_regressed)
    assert gate_regressed["honest_verdict"].startswith("blocked_publication_gate:")
    assert gate_regressed["unmet_gates"] == ["G2"]

    _write_json(
        tmp_path,
        "results/experiment_3940_fr11_v27.json",
        {
            "honest_verdict": "complete: fr11_v27_INVARIANT_HELD",
            "flagged_adversarial": False,
            "frozen_headline_ensemble_auroc": 0.902,
        },
    )
    frozen_regressed = exp3944.build_artifact(
        tmp_path,
        publication_gate_data=_publication_gate(),
        summary_statuses=_summary_statuses(),
        started_s=6.0,
        now_s=6.1,
    )
    exp3944.validate_artifact(frozen_regressed)
    assert frozen_regressed["honest_verdict"].startswith("blocked_frozen_headline:")
    assert frozen_regressed["frozen_headline_unchanged"] is False
