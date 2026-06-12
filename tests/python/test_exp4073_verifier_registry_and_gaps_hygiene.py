"""Tests for Exp 4073 verifier registry and gaps hygiene.

Spec refs: REQ-VERIFY-4073, SCENARIO-VERIFY-4073.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot.reporting import verifier_registry_and_gaps_hygiene_4073 as exp4073


REPO_ROOT = Path(__file__).parents[2]


def _minimal_registry() -> dict[str, Any]:
    return {
        "verifiers": [
            {
                "verifier_id": "gap4_program_induction_stack",
                "domain": "arc_agi2_grid",
                "version": 1,
                "kind": "process_verifier",
                "code_path": "python/carnot/agentic/gap4_program_induction_stack.py",
                "eval": {
                    "metric": "pass_at_1",
                    "arc2_gold": 19,
                    "arc2_n": 31,
                },
                "status": "candidate",
            }
        ]
    }


def _write_minimal_repo(tmp_path: Path) -> None:
    (tmp_path / "ops").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "ops" / "verifier_registry.yaml").write_text(
        yaml.safe_dump(_minimal_registry(), sort_keys=False),
        encoding="utf-8",
    )
    (tmp_path / "ops" / "verifier_gaps.md").write_text(
        "# Verifier Gaps\n\n"
        "### GAP-CODE-EXEC-DEMOFIT: code hidden-semantic execution discriminator\n"
        "- status: open\n"
        "- evidence: prior evidence\n"
        "- failure mode: visible tests miss hidden semantic failures.\n"
        "- missing discriminator: code_demo_fit_visible_tests_do_not_discriminate_hidden_semantics.\n"
        "- candidate design: stronger hidden-property tests.\n"
        "- priority: high\n\n"
        "### GAP-DECENTRALIZATION-MOE-BASE-4048: MoE-base local support gaps\n"
        "- status: open\n"
        "- evidence: prior evidence\n"
        "- failure mode: local generator support is incomplete.\n"
        "- missing discriminator: recognize rules not surfaced by local candidates.\n"
        "- candidate design: stronger base or verifier-guided generation.\n"
        "- priority: high\n",
        encoding="utf-8",
    )
    for name in (
        "arc3_gap4_rule_exec_verifier.json",
        "arc3_gap4_arc2_rule_exec_verifier.json",
        "arc3_gap4_arc2_chain_ensemble.json",
        "experiment_4068_offarc_transfer_power_sync.json",
        "experiment_4069_decentralization_moe_sync.json",
        "experiment_4046_closed_loop_replan_over_vc33_wm.json",
    ):
        shutil.copy2(REPO_ROOT / "results" / name, tmp_path / "results" / name)


def _g1_corpus_artifact(
    *,
    corpus: str = "evalplus",
    accumulated_n_tasks: int = 160,
    oracle_headroom_present: bool = True,
    demofit_ci_excludes_zero: bool = False,
    best_arm: str = "armC_symbolic",
    best_arm_ci_excludes_zero: bool = False,
    verdict: str = "complete: fixture",
) -> dict[str, Any]:
    return {
        "experiment": "experiment_4068_offarc_transfer_power_sync",
        "honest_verdict": verdict,
        "accumulated_n_tasks": accumulated_n_tasks,
        "powered_task_floor": 160,
        "corpus": corpus,
        "evaluation_corpus": "EvalPlus" if corpus == "evalplus" else "LiveCodeBench v6",
        "corpus_routed_reason": "fixture route",
        "oracle_headroom_present": oracle_headroom_present,
        "oracle_passrate": 0.75 if oracle_headroom_present else 0.9625,
        "demofit_ci_excludes_zero": demofit_ci_excludes_zero,
        "demofit_delta_pp": 8.0 if demofit_ci_excludes_zero else 0.0,
        "demofit_bootstrap_ci95": [1.0, 14.0] if demofit_ci_excludes_zero else [0.0, 0.0],
        "best_arm": best_arm,
        "best_arm_ci_excludes_zero": best_arm_ci_excludes_zero,
        "best_arm_delta_pp": 7.5 if best_arm_ci_excludes_zero else 0.0,
        "best_arm_ci95": [0.5, 15.0] if best_arm_ci_excludes_zero else [0.0, 0.0],
        "armA_vote_passrate": 0.5,
        "armB_demofit_passrate": 0.5,
        "armApp_aces_passrate": 0.5,
        "armC_symbolic_passrate": 0.5,
        "armApp_ci_excludes_zero": False,
        "armC_ci_excludes_zero": best_arm_ci_excludes_zero,
        "missing_verifier_gaps": [],
        "reproducibility_checksum": "fixture",
        "mechanism": "single_synchronous_resume_accumulate_no_background",
    }


def _g3_sync_artifact(
    *,
    coverage: float = 0.376,
    diagnosis: str = "latent",
    n_tasks: int = 30,
    complete: bool = True,
) -> dict[str, Any]:
    return {
        "experiment": "experiment_4069_decentralization_moe_sync",
        "honest_verdict": f"complete: decentralization_moe_cov_{coverage}_{diagnosis}",
        "accumulated_n_tasks": n_tasks,
        "target_n_tasks": n_tasks,
        "new_tasks_processed": n_tasks,
        "moe_base_demo_perfect_coverage": coverage,
        "n_demo_perfect_tasks": int(round(coverage * n_tasks)),
        "coverage_delta_vs_12b": 0.1169 if diagnosis == "latent" else -0.0248,
        "bootstrap_ci95": [0.01, 0.20] if diagnosis == "latent" else [-0.05, 0.05],
        "oracle_coverage": 0.6129,
        "local_support_diagnosis": diagnosis,
        "oracle_positive_control_saturated": False,
        "missing_verifier_gaps": ["17cae0c1"],
        "reproducibility_checksum": "fixture",
        "mechanism": "single_synchronous_resume_accumulate_no_background",
        "summarize_artifact": complete,
    }


def test_req_4073_spec_declared() -> None:
    # REQ-VERIFY-4073: OpenSpec declares the .376 hygiene runner and required fields.
    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4073",
        "SCENARIO-VERIFY-4073",
        "exp4073_verifier_registry_and_gaps_hygiene.py",
        "experiment_4068_offarc_transfer_power_sync.json",
        "experiment_4069_decentralization_moe_sync.json",
        "offline_reeval_bitexact",
        "g1_off_arc_outcome_recorded",
        "g3_decentralization_outcome_recorded",
        "g2_vc33_ceiling_logged",
        "aggregation_from_upstream_artifacts",
    ):
        assert marker in spec


def test_replay_gap4_headlines_from_cached_artifacts_bitexact() -> None:
    # SCENARIO-VERIFY-4073: cached ARC artifacts reproduce ARC-1 and ARC-2 headline numbers.
    replay = exp4073.replay_gap4_headlines(REPO_ROOT)
    assert replay["offline_reeval_bitexact"] is True
    assert replay["arc1_rule_exec"] == {
        "n": 31,
        "vote_pass2": pytest.approx(0.4516),
        "gated_pass2": pytest.approx(0.5806),
        "headroom_recovered": 4,
        "vote_wins_lost": 0,
    }
    assert replay["arc2_rule_exec"] == {
        "n": 31,
        "vote_pass2": pytest.approx(0.0645),
        "gated_pass2": pytest.approx(0.0645),
        "headroom_recovered": 0,
        "vote_wins_lost": 0,
    }
    assert replay["arc2_registered_chain"] == {
        "gold": 19,
        "n": 31,
        "pass_at_1": pytest.approx(0.6129),
    }


def test_classifies_actual_376_g1_g3_and_g2_outcomes() -> None:
    # REQ-VERIFY-4073: actual .376 state records G1 accumulating, G3 absent, and G2 ceiling.
    g1 = exp4073.classify_g1_corpus_routed_outcome(REPO_ROOT)
    g3 = exp4073.classify_g3_decentralization_outcome(REPO_ROOT)
    g2 = exp4073.classify_g2_vc33_ceiling(REPO_ROOT)

    assert g1["g1_off_arc_outcome_recorded"] == "g1_evalplus_accumulating"
    assert g1["accumulated_n_tasks"] == 160
    assert g1["demofit_bootstrap_ci95"] == [0.0, 3.125]
    assert g1["corpus"] == "evalplus"
    assert g3["g3_decentralization_outcome_recorded"] == (
        "g3_decentralization_absent_coverage_0.2333"
    )
    assert g3["accumulated_coverage"] == pytest.approx(0.2333)
    assert g3["accumulated_n_tasks"] == 30
    assert g2["g2_vc33_ceiling_logged"] is True
    assert g2["per_step_wm_real_divergence_rate"] == pytest.approx(0.207031)


def test_g1_corpus_classifier_distinguishes_outcomes(tmp_path: Path) -> None:
    # REQ-VERIFY-4073: G1 routing separates accumulating, demo-fit, stronger, open, and pending.
    assert exp4073.classify_g1_corpus_routed_outcome(tmp_path)[
        "g1_off_arc_outcome_recorded"
    ] == "g1_corpus_routed_pending"

    results = tmp_path / "results"
    results.mkdir()
    path = results / "experiment_4068_offarc_transfer_power_sync.json"

    path.write_text(json.dumps(_g1_corpus_artifact(accumulated_n_tasks=12)), encoding="utf-8")
    assert exp4073.classify_g1_corpus_routed_outcome(tmp_path)[
        "g1_off_arc_outcome_recorded"
    ] == "g1_evalplus_accumulating"

    path.write_text(
        json.dumps(_g1_corpus_artifact(oracle_headroom_present=False)),
        encoding="utf-8",
    )
    assert exp4073.classify_g1_corpus_routed_outcome(tmp_path)[
        "g1_off_arc_outcome_recorded"
    ] == "g1_evalplus_accumulating"

    path.write_text(
        json.dumps(_g1_corpus_artifact(demofit_ci_excludes_zero=True)),
        encoding="utf-8",
    )
    assert exp4073.classify_g1_corpus_routed_outcome(tmp_path)[
        "g1_off_arc_outcome_recorded"
    ] == "g1_evalplus_demofit_ci_excludes_zero"

    path.write_text(
        json.dumps(_g1_corpus_artifact(best_arm_ci_excludes_zero=True)),
        encoding="utf-8",
    )
    assert exp4073.classify_g1_corpus_routed_outcome(tmp_path)[
        "g1_off_arc_outcome_recorded"
    ] == "g1_evalplus_stronger_armC_symbolic_ci_excludes_zero"

    path.write_text(
        json.dumps(_g1_corpus_artifact(corpus="livecodebench_v6")),
        encoding="utf-8",
    )
    assert exp4073.classify_g1_corpus_routed_outcome(tmp_path)[
        "g1_off_arc_outcome_recorded"
    ] == "g1_livecodebench_v6_all_arms_touch_zero_gap_open"


def test_g3_classifier_distinguishes_latent_absent_accumulating_and_pending(tmp_path: Path) -> None:
    # REQ-VERIFY-4073: G3 routing records latent/absent/accumulating/pending honestly.
    assert exp4073.classify_g3_decentralization_outcome(tmp_path)[
        "g3_decentralization_outcome_recorded"
    ] == "g3_decentralization_moe_sync_4069_pending"

    results = tmp_path / "results"
    results.mkdir()
    path = results / "experiment_4069_decentralization_moe_sync.json"

    path.write_text(json.dumps(_g3_sync_artifact(diagnosis="latent")), encoding="utf-8")
    assert exp4073.classify_g3_decentralization_outcome(tmp_path)[
        "g3_decentralization_outcome_recorded"
    ] == "g3_decentralization_latent_coverage_0.376"

    path.write_text(json.dumps(_g3_sync_artifact(diagnosis="absent")), encoding="utf-8")
    assert exp4073.classify_g3_decentralization_outcome(tmp_path)[
        "g3_decentralization_outcome_recorded"
    ] == "g3_decentralization_absent_coverage_0.376"

    path.write_text(
        json.dumps(_g3_sync_artifact(diagnosis="uninformative", n_tasks=12, complete=False)),
        encoding="utf-8",
    )
    assert exp4073.classify_g3_decentralization_outcome(tmp_path)[
        "g3_decentralization_outcome_recorded"
    ] == "g3_decentralization_accumulating_coverage_0.376"


def test_ensure_ledgers_record_actual_accumulating_absent_and_ceiling() -> None:
    # SCENARIO-VERIFY-4073: ledgers record actual G1/G3 states and the G2 ceiling.
    registry = _minimal_registry()
    gaps_text = "# Verifier Gaps\n"
    replay = {
        "offline_reeval_bitexact": True,
        "arc1_rule_exec": {"vote_pass2": 0.4516, "gated_pass2": 0.5806},
        "arc2_rule_exec": {"vote_pass2": 0.0645, "gated_pass2": 0.0645},
    }
    g1 = {
        "g1_off_arc_outcome_recorded": "g1_evalplus_accumulating",
        "status": "accumulating",
        "artifact_path": exp4073.G1_CORPUS_ROUTED_PATH,
        "accumulated_n_tasks": 160,
        "powered_task_floor": 160,
        "corpus": "evalplus",
        "evaluation_corpus": "EvalPlus",
        "corpus_routed_reason": "fixture",
        "oracle_headroom_present": False,
        "oracle_passrate": 0.9625,
        "demofit_bootstrap_ci95": [0.0, 3.125],
        "best_arm": "armC_symbolic",
        "best_arm_ci95": [0.0, 3.125],
    }
    g3 = {
        "g3_decentralization_outcome_recorded": "g3_decentralization_absent_coverage_0.2333",
        "status": "absent",
        "artifact_path": exp4073.G3_DECENTRALIZATION_PATH,
        "accumulated_coverage": 0.2333,
        "accumulated_n_tasks": 30,
        "n_demo_perfect_tasks": 7,
        "local_support_diagnosis": "absent",
        "bootstrap_ci95": [-0.1581, 0.1419],
        "missing_verifier_gaps": ["17cae0c1"],
    }
    g2 = {
        "g2_vc33_ceiling_logged": True,
        "g2_vc33_ceiling_outcome_recorded": "g2_vc33_sim2real_ceiling_logged",
        "status": "gap_logged",
        "artifact_path": exp4073.G2_CLOSED_LOOP_PATH,
        "per_step_wm_real_divergence_rate": 0.207031,
        "divergence_gate_fired_count": 1,
    }

    new_registry, new_gaps, summary = exp4073.ensure_ledgers_record_outcomes(
        registry,
        gaps_text,
        replay,
        g1,
        g3,
        g2,
    )

    assert summary == {"registry_updated": True, "gaps_updated": True}
    gap4_eval = new_registry["verifiers"][0]["eval"]
    assert gap4_eval["eval_exp_4073"] == exp4073.EXP4073_ARTIFACT_PATH
    assert "GAP-CODE-EXEC-DEMOFIT" in new_gaps
    assert "g1_evalplus_accumulating" in new_gaps
    assert "GAP-DECENTRALIZATION-MOE-BASE-4048" in new_gaps
    assert "g3_decentralization_absent_coverage_0.2333" in new_gaps
    assert "GAP-ARC3-VC33-SIM2REAL-CEILING" in new_gaps
    assert "0.207031" in new_gaps


def test_ensure_ledgers_can_register_g1_transfer_and_stronger_discriminator() -> None:
    # REQ-VERIFY-4073: positive G1 outcomes become registry entries instead of gap text only.
    replay = {
        "offline_reeval_bitexact": True,
        "arc1_rule_exec": {"vote_pass2": 0.4516, "gated_pass2": 0.5806},
        "arc2_rule_exec": {"vote_pass2": 0.0645, "gated_pass2": 0.0645},
    }
    g3 = exp4073.classify_g3_decentralization_outcome(REPO_ROOT)
    g2 = exp4073.classify_g2_vc33_ceiling(REPO_ROOT)

    transfer_registry, _, transfer_summary = exp4073.ensure_ledgers_record_outcomes(
        _minimal_registry(),
        "# Verifier Gaps\n",
        replay,
        exp4073.classify_g1_corpus_routed_outcome_fixture(
            _g1_corpus_artifact(demofit_ci_excludes_zero=True)
        ),
        g3,
        g2,
    )
    assert transfer_summary["registry_updated"] is True
    assert exp4073.G1_DEMOFIT_VERIFIER_ID in {
        entry["verifier_id"] for entry in transfer_registry["verifiers"]
    }

    stronger_registry, _, stronger_summary = exp4073.ensure_ledgers_record_outcomes(
        _minimal_registry(),
        "# Verifier Gaps\n",
        replay,
        exp4073.classify_g1_corpus_routed_outcome_fixture(
            _g1_corpus_artifact(best_arm_ci_excludes_zero=True)
        ),
        g3,
        g2,
    )
    assert stronger_summary["registry_updated"] is True
    assert "gap4_code_armC_symbolic_transfer_4068" in {
        entry["verifier_id"] for entry in stronger_registry["verifiers"]
    }


def test_helper_edges_stay_deterministic() -> None:
    # REQ-VERIFY-4073: malformed ledgers and corpus labels still produce bounded records.
    replay = {
        "offline_reeval_bitexact": True,
        "arc1_rule_exec": {"vote_pass2": 0.4516, "gated_pass2": 0.5806},
        "arc2_rule_exec": {"vote_pass2": 0.0645, "gated_pass2": 0.0645},
    }
    g1 = exp4073.classify_g1_corpus_routed_outcome_fixture(
        _g1_corpus_artifact(corpus="LiveCodeBench  v6")
    )
    g3 = exp4073.classify_g3_decentralization_outcome(REPO_ROOT)
    g2 = exp4073.classify_g2_vc33_ceiling(REPO_ROOT)

    registry, _, summary = exp4073.ensure_ledgers_record_outcomes(
        {"verifiers": []},
        "# Verifier Gaps\n",
        replay,
        g1,
        g3,
        g2,
    )

    assert registry["verifiers"][0]["verifier_id"] == exp4073.GAP4_VERIFIER_ID
    assert g1["g1_off_arc_outcome_recorded"] == "g1_livecodebench_v6_all_arms_touch_zero_gap_open"
    assert summary["registry_updated"] is True
    assert exp4073._registry_contains_outcomes({}, g1) is False
    assert exp4073._registry_contains_outcomes(
        {"verifiers": [{"verifier_id": exp4073.GAP4_VERIFIER_ID, "eval": {}}]},
        g1,
    ) is False


def test_run_hygiene_writes_terminal_artifact_and_ledgers(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4073: run writes the required artifact and updates registry/gaps.
    _write_minimal_repo(tmp_path)
    artifact = exp4073.run_hygiene(tmp_path)
    exp4073.validate_artifact(artifact)

    out_path = tmp_path / exp4073.EXP4073_ARTIFACT_PATH
    assert out_path.exists()
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["honest_verdict"] == (
        "complete: gap4_reeval_bitexact_g1_g1_evalplus_accumulating_"
        "g3_g3_decentralization_absent_coverage_0.2333_g2_ceiling_recorded"
    )
    assert written["offline_reeval_bitexact"] is True
    assert written["g1_off_arc_outcome_recorded"] == "g1_evalplus_accumulating"
    assert written["g3_decentralization_outcome_recorded"] == (
        "g3_decentralization_absent_coverage_0.2333"
    )
    assert written["g2_vc33_ceiling_logged"] is True
    assert written["registry_updated"] is True
    assert written["gaps_updated"] is True

    registry = yaml.safe_load((tmp_path / "ops" / "verifier_registry.yaml").read_text())
    assert registry["verifiers"][0]["eval"]["eval_exp_4073"] == exp4073.EXP4073_ARTIFACT_PATH
    gaps = (tmp_path / "ops" / "verifier_gaps.md").read_text(encoding="utf-8")
    assert "GAP-CODE-EXEC-DEMOFIT" in gaps
    assert "GAP-DECENTRALIZATION-MOE-BASE-4048" in gaps
    assert "GAP-ARC3-VC33-SIM2REAL-CEILING" in gaps
