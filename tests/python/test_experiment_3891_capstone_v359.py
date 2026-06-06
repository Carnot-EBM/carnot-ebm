"""Tests for Exp 3891 capstone v359 forward-bet aggregation.

Spec refs: REQ-CAPSTONE-3891, SCENARIO-CAPSTONE-3891.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import capstone_v359_3891 as exp3891


SPEC_PATH = Path("openspec/capabilities/capstone/spec.md")


def _write_json(root: Path, rel_path: str, payload: dict[str, object]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _publication_gate() -> dict[str, object]:
    return {
        "paper_ready": True,
        "gates": {
            "G1": {"pass": True},
            "G2": {"pass": True},
            "G3": {"pass": True},
            "G4": {"pass": True},
        },
        "unmet_gates": [],
    }


def _summary_statuses() -> dict[int, dict[str, object]]:
    return {experiment_id: {"returncode": 0} for experiment_id in exp3891.UPSTREAM_IDS}


def _seed_v359_fixture(root: Path) -> None:
    _write_json(
        root,
        "results/experiment_3882_thesis_a_partb_killgate.json",
        {
            "honest_verdict": (
                "complete: thesis_a_partb_FUNDAMENTAL_beam0.000_"
                "argmin0.000_both_fail_vs_ar0.940_landscape_misshaped"
            ),
            "flagged_adversarial": False,
            "duration_s": 3673.32,
            "reproducibility_checksum": "a" * 64,
        },
    )
    _write_json(
        root,
        "results/experiment_3883_ebt_system2_kcurve.json",
        {
            "honest_verdict": "complete: ebt_system2_BOUNDED_PLATEAU_no_usable_descent_signal_at_scale",
            "flagged_adversarial": False,
            "best_k_accuracy": 0.0,
            "reproducibility_checksum": "b" * 64,
        },
    )
    _write_json(
        root,
        "results/experiment_3884_in_distribution_error_rich_corpus.json",
        {
            "honest_verdict": "complete: in_distribution_corpus_READY_nerr150_auroc0.9667_moat_scissor_can_run",
            "flagged_adversarial": False,
            "corpus_ready": True,
            "n_incorrect_steps": 150,
            "reproducibility_checksum": "c" * 64,
        },
    )
    _write_json(
        root,
        "results/experiment_3885_moat_scissor_in_distribution.json",
        {
            "honest_verdict": "complete: moat_scissor_indist_INCONCLUSIVE_reasoner_self_verify_auroc",
            "flagged_adversarial": True,
            "error_overlap_jaccard": 0.0,
            "reproducibility_checksum": "d" * 64,
        },
    )
    _write_json(
        root,
        "results/experiment_3886_graph_grounding_fact_verifier_defabricated.json",
        {
            "honest_verdict": "blocked_graph_verifier_not_invoked",
            "flagged_adversarial": True,
            "facts_catch_delta": 0.0,
            "reproducibility_checksum": "e" * 64,
        },
    )
    _write_json(
        root,
        "results/experiment_3887_facts_complementarity.json",
        {
            "honest_verdict": "blocked_upstream_scores_missing",
            "flagged_adversarial": False,
            "reproducibility_checksum": "f" * 64,
        },
    )
    _write_json(
        root,
        "results/experiment_3888_fr11_v24_independence_reweighting.json",
        {
            "honest_verdict": "complete: fr11_v24_INVARIANT_HELD_auroc0.9075_memcontrib0.0185_state_persisted",
            "flagged_adversarial": False,
            "learned_ensemble_auroc_in_frozen_ci": True,
            "memory_ablation_contribution_min_met": True,
            "frozen_headline_unchanged": True,
            "frozen_headline_ensemble_auroc": 0.9131,
            "reproducibility_checksum": "1" * 64,
        },
    )
    _write_json(
        root,
        "results/experiment_3889_gatemate_continuity_corrigendum.json",
        {
            "honest_verdict": "blocked_gatemate_board_unreachable",
            "flagged_adversarial": False,
            "reproducibility_checksum": "2" * 64,
        },
    )
    _write_json(
        root,
        "results/experiment_3890_polarfire_kv260_continuity.json",
        {
            "honest_verdict": (
                "success: polarfire_kv260_continuity_pfterminal_hash_verified_"
                "soft_cpu_ssh_dispatch_kvnonterminal_carnot_ising_inactive_"
                "uio_present_no_fabric_claim"
            ),
            "flagged_adversarial": False,
            "no_fpga_fabric_claim": True,
            "reproducibility_checksum": "3" * 64,
        },
    )


def test_req_capstone_3891_spec_declares_v359_contract() -> None:
    """REQ-CAPSTONE-3891: OpenSpec anchors the v359 capstone behavior."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-3891" in spec
    assert "SCENARIO-CAPSTONE-3891" in spec
    assert "skip every `flagged_adversarial:true` artifact" in spec
    assert "frozen 0.9131 headline unchanged" in spec


def test_req_capstone_3891_derivation_helpers_are_conditioned() -> None:
    """REQ-CAPSTONE-3891: helper verdicts are conditioned on landed clean inputs."""

    assert exp3891.numeric(True) is None
    assert exp3891.numeric(None) is None
    assert exp3891.numeric("not-numeric") is None
    assert exp3891.derive_ebt_adjudication({"honest_verdict": "complete: ARTIFACT"}) == "ARTIFACT"
    assert exp3891.derive_ebt_adjudication({"honest_verdict": "complete: FUNDAMENTAL"}) == "FUNDAMENTAL"
    assert exp3891.derive_ebt_adjudication(None) == "INCONCLUSIVE"
    assert exp3891.derive_system2_outcome({"honest_verdict": "complete: SUPPORTED"}) == "SUPPORTED"
    assert exp3891.derive_system2_outcome({"honest_verdict": "complete: BOUNDED_PLATEAU"}) == "BOUNDED"
    assert exp3891.derive_system2_outcome(None) == "BOUNDED"
    assert exp3891.derive_moat_verdict(None, scissor_was_flagged=True) == "INCONCLUSIVE"
    assert exp3891.derive_moat_verdict({"honest_verdict": "complete: MOAT_SURVIVES"}, scissor_was_flagged=False) == "MOAT_SURVIVES"
    assert exp3891.derive_moat_verdict({"honest_verdict": "complete: SUBSUMED"}, scissor_was_flagged=False) == "SUBSUMED"
    assert exp3891.derive_moat_verdict({"honest_verdict": "complete: weak"}, scissor_was_flagged=False) == "INCONCLUSIVE"
    assert exp3891.derive_facts_outcome({}, exp3886_was_flagged=True) == "EXCLUDED_EXP3886_FLAGGED"
    assert exp3891.derive_facts_outcome({3887: {"honest_verdict": "blocked_upstream_scores_missing"}}, exp3886_was_flagged=False) == "INCONCLUSIVE"
    assert exp3891.derive_facts_outcome({3887: {"honest_verdict": "complete: complementary_graph_grounding"}}, exp3886_was_flagged=False) == "COMPLEMENTARY"
    assert exp3891.derive_facts_outcome({3886: {"facts_catch_delta": 0.12}}, exp3886_was_flagged=False) == "REPRODUCED_NO_COMPLEMENTARITY_AUDIT"
    assert exp3891.derive_facts_outcome({}, exp3886_was_flagged=False) == "INCONCLUSIVE"
    assert exp3891.derive_fr11_v24_invariant({"honest_verdict": "complete: fr11_v24_INVARIANT_HELD"}) == "INVARIANT_HELD"
    assert exp3891.derive_fr11_v24_invariant({"honest_verdict": "complete: no invariant"}) == "INCONCLUSIVE"
    assert exp3891.derive_fr11_v24_invariant(None) == "INCONCLUSIVE"
    assert exp3891.derive_hardware_outcome({}) == "GATEMATE_MISSING_POLARFIRE_KV260_MISSING"
    assert exp3891.derive_hardware_outcome(
        {
            3889: {"gatemate_bitstream_flashed": True},
            3890: {"honest_verdict": "blocked_polarfire_ssh_unreachable"},
        }
    ) == "GATEMATE_DEFLAGGED_POLARFIRE_KV260_BLOCKED"
    assert exp3891.derive_hardware_outcome(
        {
            3889: {"honest_verdict": "complete: caveated"},
            3890: {"honest_verdict": "complete: partial"},
        }
    ) == "GATEMATE_PARTIAL_POLARFIRE_KV260_PARTIAL_NO_FABRIC_CLAIM"
    assert exp3891.frozen_headline_unchanged(None) is False
    assert "next candidate thesis" in exp3891.operator_next_thesis_recommendation("ARTIFACT", "MOAT_SURVIVES")
    assert "operator seed" in exp3891.operator_next_thesis_recommendation("INCONCLUSIVE", "INCONCLUSIVE")


def test_scenario_capstone_3891_writes_required_conditioned_artifact(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-3891: the artifact excludes flagged upstreams and preserves G1-G4."""

    _seed_v359_fixture(tmp_path)
    artifact = exp3891.build_artifact(
        tmp_path,
        publication_gate_data=_publication_gate(),
        summary_statuses=_summary_statuses(),
        started_s=1.0,
        now_s=1.00005,
    )

    exp3891.validate_artifact(artifact)
    cited_ids = {item["experiment_id"] for item in artifact["cited_upstream_artifacts"]}
    excluded_ids = {item["experiment_id"] for item in artifact["flagged_artifacts_excluded"]}

    assert artifact["honest_verdict"] == (
        "complete: capstone_v359_ebtFUNDAMENTAL_moatINCONCLUSIVE_"
        "factsEXCLUDED_EXP3886_FLAGGED_paper_ready_true_frozen_unchanged"
    )
    assert artifact["ebt_adjudication"] == "FUNDAMENTAL"
    assert artifact["ebt_system2_outcome"] == "BOUNDED"
    assert artifact["moat_verdict"] == "INCONCLUSIVE"
    assert artifact["facts_outcome"] == "EXCLUDED_EXP3886_FLAGGED"
    assert artifact["fr11_v24_invariant"] == "INVARIANT_HELD"
    assert artifact["hardware_outcome"] == "GATEMATE_BLOCKED_POLARFIRE_KV260_CONTINUITY_NO_FABRIC_CLAIM"
    assert artifact["paper_ready"] is True
    assert artifact["unmet_gates"] == []
    assert artifact["frozen_headline_unchanged"] is True
    assert "FUNDAMENTAL" in artifact["operator_next_thesis_recommendation"]
    assert "do not manufacture a new headline" in artifact["operator_next_thesis_recommendation"]
    assert excluded_ids == {3885, 3886}
    assert {3885, 3886}.isdisjoint(cited_ids)
    assert cited_ids == {3882, 3883, 3884, 3887, 3888, 3889, 3890}
    assert artifact["preconditions_checked"]["upstream_artifacts"][3885]["included"] is False
    assert artifact["preconditions_checked"]["upstream_artifacts"][3886]["included"] is False
    assert "GGUF" not in artifact["inference_substrate"]
    assert "CUDA" not in artifact["inference_substrate"]
    assert artifact["duration_s"] == pytest.approx(0.0001)
    assert exp3891.is_sha256(artifact["reproducibility_checksum"])

    for field in (
        "ebt_adjudication",
        "ebt_system2_outcome",
        "moat_verdict",
        "facts_outcome",
        "fr11_v24_invariant",
        "hardware_outcome",
        "operator_next_thesis_recommendation",
        "inference_substrate",
    ):
        assert isinstance(artifact[field], str)
        assert not isinstance(artifact[field], dict)

    output = exp3891.write_artifact(
        tmp_path,
        output_path="results/out.json",
        publication_gate_data=_publication_gate(),
        summary_statuses=_summary_statuses(),
        started_s=2.0,
        now_s=2.5,
    )
    saved = json.loads(output.read_text(encoding="utf-8"))
    exp3891.validate_artifact(saved)
    assert saved["honest_verdict"] == artifact["honest_verdict"]


def test_req_capstone_3891_blocks_when_publication_gate_or_headline_regresses(tmp_path: Path) -> None:
    """REQ-CAPSTONE-3891: CAPSTONE_COMPLETE requires paper-ready and frozen headline guards."""

    _seed_v359_fixture(tmp_path)
    regressed_gate = _publication_gate()
    regressed_gate["paper_ready"] = False
    regressed_gate["unmet_gates"] = ["G2"]

    artifact = exp3891.build_artifact(
        tmp_path,
        publication_gate_data=regressed_gate,
        summary_statuses=_summary_statuses(),
        started_s=5.0,
        now_s=5.2,
    )

    exp3891.validate_artifact(artifact)
    assert artifact["honest_verdict"] == (
        "blocked_publication_gate: capstone_v359_ebtFUNDAMENTAL_moatINCONCLUSIVE_"
        "factsEXCLUDED_EXP3886_FLAGGED_paper_ready_false_frozen_unchanged"
    )
    assert artifact["paper_ready"] is False
    assert artifact["unmet_gates"] == ["G2"]

    _write_json(
        tmp_path,
        "results/experiment_3888_fr11_v24_independence_reweighting.json",
        {
            "honest_verdict": "complete: fr11_v24_INVARIANT_HELD_auroc0.9075_memcontrib0.0185_state_persisted",
            "flagged_adversarial": False,
            "frozen_headline_unchanged": False,
            "frozen_headline_ensemble_auroc": 0.902,
            "reproducibility_checksum": "1" * 64,
        },
    )
    frozen_regressed = exp3891.build_artifact(
        tmp_path,
        publication_gate_data=_publication_gate(),
        summary_statuses=_summary_statuses(),
        started_s=6.0,
        now_s=6.2,
    )

    exp3891.validate_artifact(frozen_regressed)
    assert frozen_regressed["honest_verdict"] == (
        "blocked_frozen_headline: capstone_v359_ebtFUNDAMENTAL_moatINCONCLUSIVE_"
        "factsEXCLUDED_EXP3886_FLAGGED_paper_ready_true_frozen_changed"
    )
    assert frozen_regressed["frozen_headline_unchanged"] is False
