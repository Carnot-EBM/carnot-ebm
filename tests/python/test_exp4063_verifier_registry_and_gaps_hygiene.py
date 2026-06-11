"""Tests for Exp 4063 verifier registry and gaps hygiene.

Spec refs: REQ-VERIFY-4063, SCENARIO-VERIFY-4063.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot.reporting import verifier_registry_and_gaps_hygiene_4063 as exp4063


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
        "experiment_4057_offarc_power_evalplus.json",
        "experiment_4046_closed_loop_replan_over_vc33_wm.json",
    ):
        shutil.copy2(REPO_ROOT / "results" / name, tmp_path / "results" / name)


def _g1_evalplus_artifact(
    *,
    accumulated_n_tasks: int = 160,
    oracle_headroom_present: bool = True,
    demofit_ci_excludes_zero: bool = False,
    best_arm: str = "armC_symbolic",
    best_arm_ci_excludes_zero: bool = False,
) -> dict[str, Any]:
    return {
        "experiment": "experiment_4057_offarc_power_evalplus",
        "honest_verdict": "complete: fixture",
        "accumulated_n_tasks": accumulated_n_tasks,
        "powered_task_floor": 160,
        "raw_artifact_present": True,
        "oracle_headroom_present": oracle_headroom_present,
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
        "oracle_passrate": 0.75 if oracle_headroom_present else 1.0,
        "missing_verifier_gaps": [],
        "reproducibility_checksum": "fixture",
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
    }


def _g3_artifact(
    *,
    coverage: float = 0.375,
    diagnosis: str = "latent",
    n_tasks: int = 32,
    raw_complete: bool = True,
) -> dict[str, Any]:
    return {
        "experiment": "experiment_4059_decentralization_moe_resume",
        "honest_verdict": f"complete: decentralization_moe_resume_cov_{coverage}_{diagnosis}",
        "moe_base_demo_perfect_coverage": coverage,
        "coverage_delta_vs_12b": 0.1169,
        "bootstrap_ci95": [0.01, 0.20] if diagnosis == "latent" else [-0.05, 0.05],
        "n_tasks_scored": n_tasks,
        "ACCUMULATED-N": n_tasks,
        "oracle_coverage": 0.6129,
        "local_support_diagnosis": diagnosis,
        "raw_complete": raw_complete,
        "missing_verifier_gaps": ["17cae0c1"],
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
    }


def test_req_4063_spec_declared() -> None:
    # REQ-VERIFY-4063: OpenSpec declares the .375 hygiene runner and required fields.
    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4063",
        "SCENARIO-VERIFY-4063",
        "exp4063_verifier_registry_and_gaps_hygiene.py",
        "offline_reeval_bitexact",
        "g1_off_arc_outcome_recorded",
        "g3_decentralization_outcome_recorded",
        "g2_vc33_ceiling_logged",
        "aggregation_from_upstream_artifacts",
    ):
        assert marker in spec


def test_replay_gap4_headlines_from_cached_artifacts_bitexact() -> None:
    # SCENARIO-VERIFY-4063: cached ARC artifacts reproduce ARC-1 and ARC-2 headline numbers.
    replay = exp4063.replay_gap4_headlines(REPO_ROOT)
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


def test_classifies_actual_375_g1_g3_and_g2_outcomes() -> None:
    # REQ-VERIFY-4063: actual .375 state records G1 accumulating, G3 pending, and G2 ceiling.
    g1 = exp4063.classify_g1_evalplus_outcome(REPO_ROOT)
    g3 = exp4063.classify_g3_decentralization_outcome(REPO_ROOT)
    g2 = exp4063.classify_g2_vc33_ceiling(REPO_ROOT)

    assert g1["g1_off_arc_outcome_recorded"] == "g1_evalplus_accumulating"
    assert g1["accumulated_n_tasks"] == 0
    assert g3["g3_decentralization_outcome_recorded"] == (
        "g3_decentralization_moe_base_4048_pending"
    )
    assert g2["g2_vc33_ceiling_logged"] is True
    assert g2["per_step_wm_real_divergence_rate"] == pytest.approx(0.207031)


def test_g1_evalplus_classifier_distinguishes_outcomes(tmp_path: Path) -> None:
    # REQ-VERIFY-4063: G1 routing separates accumulating, demo-fit, stronger, open, and pending.
    assert exp4063.classify_g1_evalplus_outcome(tmp_path)["g1_off_arc_outcome_recorded"] == (
        "g1_evalplus_pending"
    )

    results = tmp_path / "results"
    results.mkdir()
    path = results / "experiment_4057_offarc_power_evalplus.json"

    path.write_text(json.dumps(_g1_evalplus_artifact(accumulated_n_tasks=12)), encoding="utf-8")
    assert exp4063.classify_g1_evalplus_outcome(tmp_path)["g1_off_arc_outcome_recorded"] == (
        "g1_evalplus_accumulating"
    )

    path.write_text(
        json.dumps(_g1_evalplus_artifact(demofit_ci_excludes_zero=True)),
        encoding="utf-8",
    )
    assert exp4063.classify_g1_evalplus_outcome(tmp_path)["g1_off_arc_outcome_recorded"] == (
        "g1_evalplus_demofit_ci_excludes_zero"
    )

    path.write_text(
        json.dumps(_g1_evalplus_artifact(best_arm_ci_excludes_zero=True)),
        encoding="utf-8",
    )
    assert exp4063.classify_g1_evalplus_outcome(tmp_path)["g1_off_arc_outcome_recorded"] == (
        "g1_evalplus_stronger_armC_symbolic_ci_excludes_zero"
    )

    path.write_text(json.dumps(_g1_evalplus_artifact()), encoding="utf-8")
    assert exp4063.classify_g1_evalplus_outcome(tmp_path)["g1_off_arc_outcome_recorded"] == (
        "g1_evalplus_all_arms_touch_zero_gap_open"
    )


def test_g3_classifier_distinguishes_latent_absent_accumulating_and_pending(tmp_path: Path) -> None:
    # REQ-VERIFY-4063: G3 routing records latent/absent/accumulating/pending honestly.
    assert exp4063.classify_g3_decentralization_outcome(tmp_path)[
        "g3_decentralization_outcome_recorded"
    ] == "g3_decentralization_moe_base_4048_pending"

    results = tmp_path / "results"
    results.mkdir()
    path = results / "experiment_4059_decentralization_moe_resume.json"

    path.write_text(json.dumps(_g3_artifact(diagnosis="latent")), encoding="utf-8")
    assert exp4063.classify_g3_decentralization_outcome(tmp_path)[
        "g3_decentralization_outcome_recorded"
    ] == "g3_decentralization_latent_coverage_0.375"

    path.write_text(json.dumps(_g3_artifact(diagnosis="absent")), encoding="utf-8")
    assert exp4063.classify_g3_decentralization_outcome(tmp_path)[
        "g3_decentralization_outcome_recorded"
    ] == "g3_decentralization_absent_coverage_0.375"

    path.write_text(
        json.dumps(_g3_artifact(diagnosis="uninformative", n_tasks=12, raw_complete=False)),
        encoding="utf-8",
    )
    assert exp4063.classify_g3_decentralization_outcome(tmp_path)[
        "g3_decentralization_outcome_recorded"
    ] == "g3_decentralization_accumulating_coverage_0.375"


def test_ensure_ledgers_record_actual_pending_and_ceiling() -> None:
    # SCENARIO-VERIFY-4063: ledgers record actual G1/G3 pending states and the G2 ceiling.
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
        "artifact_path": exp4063.G1_EVALPLUS_PATH,
        "accumulated_n_tasks": 0,
        "powered_task_floor": 160,
        "demofit_bootstrap_ci95": [0.0, 0.0],
        "best_arm": "armC_symbolic",
        "best_arm_ci95": [0.0, 0.0],
        "oracle_headroom_present": False,
    }
    g3 = {
        "g3_decentralization_outcome_recorded": "g3_decentralization_moe_base_4048_pending",
        "status": "pending",
        "artifact_path": exp4063.G3_DECENTRALIZATION_PATH,
        "reason": "missing_exp4059_artifact",
    }
    g2 = {
        "g2_vc33_ceiling_logged": True,
        "g2_vc33_ceiling_outcome_recorded": "g2_vc33_sim2real_ceiling_logged",
        "status": "gap_logged",
        "artifact_path": exp4063.G2_CLOSED_LOOP_PATH,
        "per_step_wm_real_divergence_rate": 0.207031,
        "divergence_gate_fired_count": 1,
    }

    new_registry, new_gaps, summary = exp4063.ensure_ledgers_record_outcomes(
        registry,
        gaps_text,
        replay,
        g1,
        g3,
        g2,
    )

    assert summary == {"registry_updated": True, "gaps_updated": True}
    gap4_eval = new_registry["verifiers"][0]["eval"]
    assert gap4_eval["eval_exp_4063"] == exp4063.EXP4063_ARTIFACT_PATH
    assert "GAP-CODE-EXEC-DEMOFIT" in new_gaps
    assert "g1_evalplus_accumulating" in new_gaps
    assert "GAP-DECENTRALIZATION-MOE-BASE-4048" in new_gaps
    assert "g3_decentralization_moe_base_4048_pending" in new_gaps
    assert "GAP-ARC3-VC33-SIM2REAL-CEILING" in new_gaps
    assert "0.207031" in new_gaps


def test_run_hygiene_writes_terminal_artifact_and_ledgers(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4063: run writes the required artifact and updates registry/gaps.
    _write_minimal_repo(tmp_path)
    artifact = exp4063.run_hygiene(tmp_path)
    exp4063.validate_artifact(artifact)

    out_path = tmp_path / exp4063.EXP4063_ARTIFACT_PATH
    assert out_path.exists()
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["honest_verdict"] == (
        "complete: gap4_reeval_bitexact_g1_g1_evalplus_accumulating_"
        "g3_g3_decentralization_moe_base_4048_pending_g2_ceiling_recorded"
    )
    assert written["offline_reeval_bitexact"] is True
    assert written["g2_vc33_ceiling_logged"] is True
    assert written["registry_updated"] is True
    assert written["gaps_updated"] is True

    registry = yaml.safe_load((tmp_path / "ops" / "verifier_registry.yaml").read_text())
    assert registry["verifiers"][0]["eval"]["eval_exp_4063"] == exp4063.EXP4063_ARTIFACT_PATH
    gaps = (tmp_path / "ops" / "verifier_gaps.md").read_text(encoding="utf-8")
    assert "GAP-CODE-EXEC-DEMOFIT" in gaps
    assert "GAP-DECENTRALIZATION-MOE-BASE-4048" in gaps
    assert "GAP-ARC3-VC33-SIM2REAL-CEILING" in gaps
