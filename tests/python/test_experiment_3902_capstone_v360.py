"""Tests for Exp 3902 capstone v360 verdict aggregation.

Spec refs: REQ-CAPSTONE-3902, SCENARIO-CAPSTONE-3902.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import capstone_v360_3902 as exp3902


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
        for experiment_id in exp3902.UPSTREAM_IDS
    }


def _seed_v360_fixture(root: Path) -> None:
    _write_json(
        root,
        "results/experiment_3893_ebt_fundamental_replication.json",
        {
            "honest_verdict": (
                "complete: ebt_fundamental_REPLICATED_beam0.000_argmin0.000_"
                "vs_ar0.910_nseeds2_energy_as_generator_banked_negative"
            ),
            "flagged_adversarial": False,
            "replication_outcome": "REPLICATED",
            "reproducibility_checksum": "a" * 64,
        },
    )
    _write_json(
        root,
        "results/experiment_3894_reasoner_self_verify_harness.json",
        {
            "honest_verdict": "complete: reasoner_self_verify_harness_READY_fixture_auroc0.9167",
            "flagged_adversarial": False,
            "fixture_auroc": 0.9167,
            "reproducibility_checksum": "b" * 64,
        },
    )
    _write_json(
        root,
        "results/experiment_3895_moat_scissor_tested_harness.json",
        {
            "honest_verdict": "complete: moat_scissor_MOAT_SURVIVES_reasoner_self_verify_not_subsumed",
            "flagged_adversarial": False,
            "reasoner_self_verify_auroc": 0.54,
            "carnot_ensemble_auroc": 0.96,
            "reproducibility_checksum": "c" * 64,
        },
    )
    _write_json(
        root,
        "results/experiment_3896_graph_grounding_verifier_harness.json",
        {
            "honest_verdict": "complete: graph_grounding_verifier_READY_fixture_model_invokedtrue",
            "flagged_adversarial": False,
            "fixture_auroc": 0.87,
            "reproducibility_checksum": "d" * 64,
        },
    )
    _write_json(
        root,
        "results/experiment_3897_graph_grounding_facts_run.json",
        {
            "honest_verdict": "complete: graph_grounding_facts_REPRODUCED_signal_catches_fact_errors",
            "flagged_adversarial": False,
            "graph_grounding_signal_reproduced": True,
            "reproducibility_checksum": "e" * 64,
        },
    )
    _write_json(
        root,
        "results/experiment_3898_facts_complementarity.json",
        {
            "honest_verdict": "complete: facts_complementarity_COMPLEMENTARY_independent_signal",
            "flagged_adversarial": False,
            "facts_complementary": True,
            "reproducibility_checksum": "f" * 64,
        },
    )
    _write_json(
        root,
        "results/experiment_3899_fr11_v25.json",
        {
            "honest_verdict": "complete: fr11_v25_INVARIANT_HELD_state_persisted",
            "flagged_adversarial": False,
            "frozen_headline_unchanged": True,
            "frozen_headline_ensemble_auroc": 0.9131,
            "reproducibility_checksum": "1" * 64,
        },
    )
    _write_json(
        root,
        "results/experiment_3900_gatemate_terminal_confirmation.json",
        {
            "honest_verdict": "blocked_gatemate_board_unreachable",
            "flagged_adversarial": False,
            "terminal_state_reached": False,
            "reproducibility_checksum": "2" * 64,
        },
    )
    _write_json(
        root,
        "results/experiment_3901_polarfire_kv260_continuity.json",
        {
            "honest_verdict": (
                "success: polarfire_kv260_continuity_pfterminal_hash_verified_"
                "soft_cpu_ssh_dispatch_kvnonterminal_carnot_ising_inactive_"
                "uio_present_no_fabric_claim"
            ),
            "flagged_adversarial": False,
            "fabric_acceleration_claimed": False,
            "no_fpga_fabric_claim": True,
            "reproducibility_checksum": "3" * 64,
        },
    )


def test_req_capstone_3902_spec_declares_v360_contract() -> None:
    """REQ-CAPSTONE-3902: OpenSpec anchors the v360 capstone behavior."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-3902" in spec
    assert "SCENARIO-CAPSTONE-3902" in spec
    assert "not stamped `flagged_adversarial:true`" in spec
    assert "frozen 0.9131 headline unchanged" in spec


def test_req_capstone_3902_derivation_helpers_are_conditioned() -> None:
    """REQ-CAPSTONE-3902: helper verdicts only use landed clean inputs."""

    assert exp3902.numeric(True) is None
    assert exp3902.numeric("0.2") is None
    assert exp3902.derive_ebt_replication_outcome(None) == "INCONCLUSIVE"
    assert exp3902.derive_ebt_replication_outcome({"replication_outcome": "REPLICATED"}) == "REPLICATED"
    assert exp3902.derive_ebt_replication_outcome({"honest_verdict": "complete: REPLICATED"}) == "REPLICATED"
    assert exp3902.derive_ebt_replication_outcome({"honest_verdict": "complete: REFUTED"}) == "REFUTED"
    assert exp3902.derive_ebt_replication_outcome({"honest_verdict": "complete: weak"}) == "INCONCLUSIVE"
    assert exp3902.derive_moat_verdict({"honest_verdict": "complete: MOAT_SURVIVES"}) == "MOAT_SURVIVES"
    assert exp3902.derive_moat_verdict({"honest_verdict": "complete: SUBSUMED"}) == "SUBSUMED"
    assert exp3902.derive_moat_verdict({"honest_verdict": "complete: weak"}) == "INCONCLUSIVE"
    assert exp3902.derive_facts_outcome({}, exp3897_was_flagged=True) == "EXCLUDED_EXP3897_FLAGGED"
    assert exp3902.derive_facts_outcome({3897: {"honest_verdict": "complete: reproduced"}}, exp3897_was_flagged=False) == "REPRODUCED"
    assert exp3902.derive_facts_outcome(
        {
            3897: {"graph_grounding_signal_reproduced": True},
            3898: {"honest_verdict": "complete: complementary"},
        },
        exp3897_was_flagged=False,
    ) == "REPRODUCED_COMPLEMENTARY"
    assert exp3902.derive_facts_outcome(
        {3898: {"facts_complementary": True}},
        exp3897_was_flagged=False,
    ) == "COMPLEMENTARY_WITHOUT_REPRODUCTION_ARTIFACT"
    assert exp3902.derive_fr11_v25_invariant({"honest_verdict": "complete: fr11_v25_INVARIANT_HELD"}) == "INVARIANT_HELD"
    assert exp3902.derive_fr11_v25_invariant({"honest_verdict": "complete: weak"}) == "INCONCLUSIVE"
    assert exp3902.derive_hardware_outcome({}) == "GATEMATE_MISSING_POLARFIRE_KV260_MISSING"
    assert exp3902.derive_hardware_outcome(
        {
            3900: {"terminal_state_reached": True},
            3901: {"honest_verdict": "blocked_polarfire_ssh_unreachable"},
        }
    ) == "GATEMATE_TERMINAL_CONFIRMED_POLARFIRE_KV260_BLOCKED"
    assert exp3902.derive_hardware_outcome(
        {
            3900: {"honest_verdict": "complete: partial"},
            3901: {"honest_verdict": "complete: partial"},
        }
    ) == "GATEMATE_PARTIAL_POLARFIRE_KV260_PARTIAL_NO_FABRIC_CLAIM"
    assert exp3902.frozen_headline_unchanged({3899: {"frozen_headline_unchanged": False}}) is False
    assert exp3902.frozen_headline_unchanged({3899: {"frozen_headline_ensemble_auroc": 0.902}}) is False
    assert exp3902.frozen_headline_unchanged({}) is True
    assert exp3902.both_energy_theses_bounded("REPLICATED") is True
    assert exp3902.both_energy_theses_bounded("INCONCLUSIVE") is False


def test_scenario_capstone_3902_writes_forced_conditioned_artifact(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-3902: clean landed verdicts produce the v360 capstone."""

    _seed_v360_fixture(tmp_path)
    artifact = exp3902.build_artifact(
        tmp_path,
        publication_gate_data=_publication_gate(),
        summary_statuses=_summary_statuses(),
        started_s=1.0,
        now_s=1.00005,
    )

    exp3902.validate_artifact(artifact)

    assert artifact["honest_verdict"] == (
        "complete: capstone_v360_ebtREPLICATED_moatMOAT_SURVIVES_"
        "factsREPRODUCED_COMPLEMENTARY_paper_ready_true_frozen_unchanged"
    )
    assert artifact["ebt_replication_outcome"] == "REPLICATED"
    assert artifact["moat_verdict"] == "MOAT_SURVIVES"
    assert artifact["facts_outcome"] == "REPRODUCED_COMPLEMENTARY"
    assert artifact["fr11_v25_invariant"] == "INVARIANT_HELD"
    assert artifact["hardware_outcome"] == "GATEMATE_BLOCKED_POLARFIRE_KV260_CONTINUITY_NO_FABRIC_CLAIM"
    assert artifact["both_energy_theses_bounded"] is True
    assert artifact["paper_ready"] is True
    assert artifact["unmet_gates"] == []
    assert artifact["frozen_headline_unchanged"] is True
    assert "verifier as a durable, broad external second-opinion layer" in artifact["operator_next_thesis_recommendation"]
    assert "energy as generator/selector" in artifact["operator_next_thesis_recommendation"]
    assert artifact["flagged_artifacts_excluded"] == []
    assert artifact["preconditions_checked"]["capstone_complete"] is True
    assert artifact["preconditions_checked"]["all_landed_nonflagged_verdicts_aggregated"] is True
    assert "GGUF" not in artifact["inference_substrate"]
    assert "CUDA" not in artifact["inference_substrate"]
    assert artifact["duration_s"] == pytest.approx(0.0001)
    assert exp3902.is_sha256(artifact["reproducibility_checksum"])

    for field in exp3902.STRING_VERDICT_FIELDS:
        assert isinstance(artifact[field], str)
        assert not isinstance(artifact[field], dict)
    assert isinstance(artifact["both_energy_theses_bounded"], bool)

    output = exp3902.write_artifact(
        tmp_path,
        output_path="results/out.json",
        publication_gate_data=_publication_gate(),
        summary_statuses=_summary_statuses(),
        started_s=2.0,
        now_s=2.5,
    )
    saved = json.loads(output.read_text(encoding="utf-8"))
    exp3902.validate_artifact(saved)
    assert saved["honest_verdict"] == artifact["honest_verdict"]


def test_req_capstone_3902_excludes_flagged_and_live_critical_inputs(tmp_path: Path) -> None:
    """REQ-CAPSTONE-3902: stamped flags and summarizer criticals are not aggregated."""

    _seed_v360_fixture(tmp_path)
    flagged = json.loads((tmp_path / "results/experiment_3897_graph_grounding_facts_run.json").read_text())
    flagged["flagged_adversarial"] = True
    _write_json(tmp_path, "results/experiment_3897_graph_grounding_facts_run.json", flagged)

    artifact = exp3902.build_artifact(
        tmp_path,
        publication_gate_data=_publication_gate(),
        summary_statuses=_summary_statuses(live_critical={3896}),
        started_s=3.0,
        now_s=3.2,
    )

    exp3902.validate_artifact(artifact)

    cited_ids = {item["experiment_id"] for item in artifact["cited_upstream_artifacts"]}
    excluded_ids = {item["experiment_id"] for item in artifact["flagged_artifacts_excluded"]}
    live_critical_ids = set(artifact["preconditions_checked"]["live_critical_artifacts_observed"])

    assert artifact["facts_outcome"] == "EXCLUDED_EXP3897_FLAGGED"
    assert excluded_ids == {3897}
    assert 3897 not in cited_ids
    assert 3896 not in cited_ids
    assert live_critical_ids == {3896}
    assert artifact["preconditions_checked"]["upstream_artifacts"][3896]["included"] is False


def test_req_capstone_3902_missing_upstreams_remain_inconclusive(tmp_path: Path) -> None:
    """REQ-CAPSTONE-3902: partial landing records honest missing states."""

    _write_json(
        tmp_path,
        "results/experiment_3895_moat_scissor_tested_harness.json",
        {
            "honest_verdict": "complete: moat_scissor_INCONCLUSIVE_reasoner_self_verify_auroc",
            "flagged_adversarial": False,
            "reproducibility_checksum": "c" * 64,
        },
    )

    artifact = exp3902.build_artifact(
        tmp_path,
        publication_gate_data=_publication_gate(),
        summary_statuses=_summary_statuses(),
        started_s=4.0,
        now_s=4.1,
    )

    exp3902.validate_artifact(artifact)

    assert artifact["honest_verdict"] == (
        "complete: capstone_v360_ebtINCONCLUSIVE_moatINCONCLUSIVE_"
        "factsINCONCLUSIVE_paper_ready_true_frozen_unchanged"
    )
    assert artifact["ebt_replication_outcome"] == "INCONCLUSIVE"
    assert artifact["facts_outcome"] == "INCONCLUSIVE"
    assert artifact["fr11_v25_invariant"] == "INCONCLUSIVE"
    assert artifact["both_energy_theses_bounded"] is False
    assert artifact["preconditions_checked"]["upstream_artifacts"][3893]["exists"] is False
    assert artifact["preconditions_checked"]["capstone_complete"] is True


def test_req_capstone_3902_blocks_when_publication_gate_or_headline_regresses(tmp_path: Path) -> None:
    """REQ-CAPSTONE-3902: CAPSTONE_COMPLETE requires paper-ready and frozen guards."""

    _seed_v360_fixture(tmp_path)
    gate_regressed = exp3902.build_artifact(
        tmp_path,
        publication_gate_data=_publication_gate(paper_ready=False),
        summary_statuses=_summary_statuses(),
        started_s=5.0,
        now_s=5.2,
    )

    exp3902.validate_artifact(gate_regressed)
    assert gate_regressed["honest_verdict"] == (
        "blocked_publication_gate: capstone_v360_ebtREPLICATED_moatMOAT_SURVIVES_"
        "factsREPRODUCED_COMPLEMENTARY_paper_ready_false_frozen_unchanged"
    )
    assert gate_regressed["unmet_gates"] == ["G2"]

    fr11 = json.loads((tmp_path / "results/experiment_3899_fr11_v25.json").read_text())
    fr11["frozen_headline_unchanged"] = False
    fr11["frozen_headline_ensemble_auroc"] = 0.902
    _write_json(tmp_path, "results/experiment_3899_fr11_v25.json", fr11)
    frozen_regressed = exp3902.build_artifact(
        tmp_path,
        publication_gate_data=_publication_gate(),
        summary_statuses=_summary_statuses(),
        started_s=6.0,
        now_s=6.2,
    )

    exp3902.validate_artifact(frozen_regressed)
    assert frozen_regressed["honest_verdict"] == (
        "blocked_frozen_headline: capstone_v360_ebtREPLICATED_moatMOAT_SURVIVES_"
        "factsREPRODUCED_COMPLEMENTARY_paper_ready_true_frozen_changed"
    )
    assert frozen_regressed["frozen_headline_unchanged"] is False
