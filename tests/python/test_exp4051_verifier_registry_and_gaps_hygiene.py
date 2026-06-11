"""Tests for Exp 4051 verifier registry and gaps hygiene.

Spec refs: REQ-VERIFY-4051, SCENARIO-VERIFY-4051.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot.reporting import verifier_registry_and_gaps_hygiene_4051 as exp4051


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
        "- failure mode: candidates can pass visible demo tests while failing hidden semantic tests.\n"
        "- missing discriminator: code_demo_fit_visible_tests_do_not_discriminate_hidden_semantics.\n"
        "- candidate design: stronger hidden-property tests.\n"
        "- priority: high\n",
        encoding="utf-8",
    )
    for name in (
        "arc3_gap4_rule_exec_verifier.json",
        "arc3_gap4_arc2_rule_exec_verifier.json",
        "arc3_gap4_arc2_chain_ensemble.json",
        "experiment_4045_offarc_transfer_power.json",
        "experiment_4046_closed_loop_replan_over_vc33_wm.json",
    ):
        shutil.copy2(REPO_ROOT / "results" / name, tmp_path / "results" / name)


def _g1_artifact(
    *,
    n_tasks: int = 160,
    demofit_ci_excludes_zero: bool = False,
    demofit_ci: list[float] | None = None,
    demofit_delta: float = 0.0,
    best_arm: str = "armC_symbolic",
    best_arm_ci_excludes_zero: bool = False,
    best_arm_ci: list[float] | None = None,
    best_arm_delta: float = 0.0,
    oracle_headroom: bool = True,
) -> dict[str, Any]:
    return {
        "experiment": "experiment_4045_offarc_transfer_power",
        "honest_verdict": "complete: fixture",
        "n_tasks": n_tasks,
        "powered_task_floor": 160,
        "raw_artifact_present": True,
        "demofit_ci_excludes_zero": demofit_ci_excludes_zero,
        "demofit_delta_pp": demofit_delta,
        "demofit_bootstrap_ci95": demofit_ci or [0.0, 0.0],
        "best_arm": best_arm,
        "best_arm_ci_excludes_zero": best_arm_ci_excludes_zero,
        "best_arm_delta_pp": best_arm_delta,
        "best_arm_ci95": best_arm_ci or [0.0, 0.0],
        "armA_vote_passrate": 0.5,
        "armB_demofit_passrate": 0.5,
        "armApp_aces_passrate": 0.5,
        "armC_symbolic_passrate": 0.5,
        "oracle_passrate": 0.75 if oracle_headroom else 0.5,
        "oracle_headroom": oracle_headroom,
        "reproducibility_checksum": "fixture",
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
    }


def _g2_artifact(
    *,
    broke_wall: bool,
    divergence_rate: float,
    gate_count: int,
) -> dict[str, Any]:
    verdict = (
        "complete: closed_loop_solved_vc33_L1_real_env_confirmed"
        if broke_wall
        else f"complete: closed_loop_no_solve_vc33_wm_sim2real_ceiling_divergence_{divergence_rate:.3f}"
    )
    return {
        "experiment": "experiment_4046_closed_loop_replan_over_vc33_wm",
        "honest_verdict": verdict,
        "closed_loop_broke_wall": broke_wall,
        "new_levels_solved_this_task": 1 if broke_wall else 0,
        "per_step_wm_real_divergence_rate": divergence_rate,
        "divergence_gate_fired_count": gate_count,
        "real_env_confirmed": broke_wall,
        "degenerate_plan_refused": False,
        "goal_predicate_heldout_precision": 1.0,
        "inference_substrate": ("offline_arc_agi3_closed_loop_replanning_with_real_env_grounding"),
    }


def test_req_4051_spec_declared() -> None:
    # REQ-VERIFY-4051: OpenSpec declares the hygiene runner and required fields.
    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4051",
        "SCENARIO-VERIFY-4051",
        "exp4051_verifier_registry_and_gaps_hygiene.py",
        "offline_reeval_bitexact",
        "g1_off_arc_outcome_recorded",
        "g2_closed_loop_outcome_recorded",
        "aggregation_from_upstream_artifacts",
    ):
        assert marker in spec


def test_replay_gap4_headlines_from_cached_artifacts_bitexact() -> None:
    # SCENARIO-VERIFY-4051: cached ARC artifacts reproduce ARC-1 and ARC-2 headline numbers.
    replay = exp4051.replay_gap4_headlines(REPO_ROOT)
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


def test_classifies_actual_374_g1_and_g2_outcomes() -> None:
    # REQ-VERIFY-4051: incomplete G1 records pending, and no-solve G2 logs sim2real ceiling.
    g1 = exp4051.classify_g1_off_arc_outcome(REPO_ROOT)
    g2 = exp4051.classify_g2_closed_loop_outcome(REPO_ROOT)
    assert g1["g1_off_arc_outcome_recorded"] == "g1_off_arc_power_pending"
    assert g1["n_tasks"] == 22
    assert g2["g2_closed_loop_outcome_recorded"] == "g2_sim2real_ceiling_gap_logged"
    assert g2["per_step_wm_real_divergence_rate"] == pytest.approx(0.207031)


def test_g1_classifier_distinguishes_transfer_stronger_and_open_bound(tmp_path: Path) -> None:
    # REQ-VERIFY-4051: G1 routing separates demo-fit transfer, stronger-arm closure, and open gap.
    assert exp4051.classify_g1_off_arc_outcome(tmp_path)["g1_off_arc_outcome_recorded"] == (
        "g1_off_arc_power_pending"
    )

    results = tmp_path / "results"
    results.mkdir()
    g1_path = results / "experiment_4045_offarc_transfer_power.json"

    g1_path.write_text(
        json.dumps(
            _g1_artifact(
                demofit_ci_excludes_zero=True,
                demofit_delta=9.5,
                demofit_ci=[1.0, 18.0],
            )
        ),
        encoding="utf-8",
    )
    assert exp4051.classify_g1_off_arc_outcome(tmp_path)["g1_off_arc_outcome_recorded"] == (
        "g1_demofit_ci_excludes_zero"
    )

    g1_path.write_text(
        json.dumps(
            _g1_artifact(
                best_arm="armC_symbolic",
                best_arm_ci_excludes_zero=True,
                best_arm_delta=7.5,
                best_arm_ci=[0.5, 15.0],
            )
        ),
        encoding="utf-8",
    )
    assert exp4051.classify_g1_off_arc_outcome(tmp_path)["g1_off_arc_outcome_recorded"] == (
        "g1_stronger_armC_symbolic_ci_excludes_zero"
    )

    g1_path.write_text(json.dumps(_g1_artifact()), encoding="utf-8")
    open_gap = exp4051.classify_g1_off_arc_outcome(tmp_path)
    assert open_gap["g1_off_arc_outcome_recorded"] == "g1_all_arms_touch_zero_gap_open"
    assert open_gap["demofit_bootstrap_ci95"] == [0.0, 0.0]


def test_g2_classifier_distinguishes_capability_gap_and_pending(tmp_path: Path) -> None:
    # REQ-VERIFY-4051: G2 routing separates solved capability, sim2real gap, and pending.
    assert exp4051.classify_g2_closed_loop_outcome(tmp_path)["g2_closed_loop_outcome_recorded"] == (
        "g2_closed_loop_pending"
    )

    results = tmp_path / "results"
    results.mkdir()
    g2_path = results / "experiment_4046_closed_loop_replan_over_vc33_wm.json"
    g2_path.write_text(
        json.dumps(_g2_artifact(broke_wall=False, divergence_rate=0.25, gate_count=2)),
        encoding="utf-8",
    )
    assert exp4051.classify_g2_closed_loop_outcome(tmp_path)["g2_closed_loop_outcome_recorded"] == (
        "g2_sim2real_ceiling_gap_logged"
    )

    g2_path.write_text(
        json.dumps(_g2_artifact(broke_wall=True, divergence_rate=0.0, gate_count=0)),
        encoding="utf-8",
    )
    assert exp4051.classify_g2_closed_loop_outcome(tmp_path)["g2_closed_loop_outcome_recorded"] == (
        "g2_closed_loop_capability_registered"
    )


def test_ensure_ledgers_record_pending_g1_and_g2_sim2real_gap() -> None:
    # SCENARIO-VERIFY-4051: ledger text records pending G1 and the G2 sim2real-ceiling gap.
    registry = _minimal_registry()
    gaps_text = "# Verifier Gaps\n"
    replay = {
        "offline_reeval_bitexact": True,
        "arc1_rule_exec": {"vote_pass2": 0.4516, "gated_pass2": 0.5806},
        "arc2_rule_exec": {"vote_pass2": 0.0645, "gated_pass2": 0.0645},
    }
    g1 = {
        "g1_off_arc_outcome_recorded": "g1_off_arc_power_pending",
        "status": "pending",
        "artifact_path": exp4051.G1_OFF_ARC_PATH,
        "n_tasks": 22,
        "powered_task_floor": 160,
        "demofit_bootstrap_ci95": [0.0, 0.0],
        "best_arm": "armC_symbolic",
        "best_arm_ci95": [0.0, 0.0],
        "reason": "partial_or_incomplete_exp4045",
    }
    g2 = {
        "g2_closed_loop_outcome_recorded": "g2_sim2real_ceiling_gap_logged",
        "status": "gap_logged",
        "artifact_path": exp4051.G2_CLOSED_LOOP_PATH,
        "per_step_wm_real_divergence_rate": 0.207031,
        "divergence_gate_fired_count": 1,
        "closed_loop_broke_wall": False,
    }

    new_registry, new_gaps, summary = exp4051.ensure_ledgers_record_outcomes(
        registry,
        gaps_text,
        replay,
        g1,
        g2,
    )

    assert summary == {"registry_updated": True, "gaps_updated": True}
    gap4_eval = new_registry["verifiers"][0]["eval"]
    assert gap4_eval["eval_exp_4051"] == exp4051.EXP4051_ARTIFACT_PATH
    assert "GAP-CODE-EXEC-DEMOFIT" in new_gaps
    assert "g1_off_arc_power_pending" in new_gaps
    assert "GAP-ARC3-VC33-SIM2REAL-CEILING" in new_gaps
    assert "0.207031" in new_gaps


def test_ensure_ledgers_adds_g1_and_g2_registry_entries() -> None:
    # REQ-VERIFY-4051: successful G1/G2 outcomes are represented as registry capabilities.
    registry = _minimal_registry()
    replay = {"offline_reeval_bitexact": True}
    g1 = {
        "g1_off_arc_outcome_recorded": "g1_demofit_ci_excludes_zero",
        "status": "transfer",
        "artifact_path": exp4051.G1_OFF_ARC_PATH,
        "n_tasks": 160,
        "demofit_delta_pp": 8.0,
        "demofit_bootstrap_ci95": [1.0, 14.0],
        "best_arm": "armB_demofit",
    }
    g2 = {
        "g2_closed_loop_outcome_recorded": "g2_closed_loop_capability_registered",
        "status": "capability_registered",
        "artifact_path": exp4051.G2_CLOSED_LOOP_PATH,
        "per_step_wm_real_divergence_rate": 0.0,
        "divergence_gate_fired_count": 0,
        "closed_loop_broke_wall": True,
        "goal_predicate_heldout_precision": 1.0,
    }

    new_registry, new_gaps, summary = exp4051.ensure_ledgers_record_outcomes(
        registry,
        "# Verifier Gaps\n",
        replay,
        g1,
        g2,
    )
    ids = {entry["verifier_id"] for entry in new_registry["verifiers"]}
    assert exp4051.G1_DEMOFIT_VERIFIER_ID in ids
    assert exp4051.G2_CLOSED_LOOP_VERIFIER_ID in ids
    assert summary == {"registry_updated": True, "gaps_updated": False}
    assert "GAP-ARC3-VC33-SIM2REAL-CEILING" not in new_gaps


def test_ensure_ledgers_adds_stronger_g1_registry_entry() -> None:
    # REQ-VERIFY-4051: stronger-arm G1 closure is a code-domain registry entry.
    registry = _minimal_registry()
    g1 = {
        "g1_off_arc_outcome_recorded": "g1_stronger_armC_symbolic_ci_excludes_zero",
        "status": "stronger_discriminator_registered",
        "artifact_path": exp4051.G1_OFF_ARC_PATH,
        "n_tasks": 160,
        "best_arm": "armC_symbolic",
        "best_arm_delta_pp": 7.5,
        "best_arm_ci95": [0.5, 15.0],
    }
    g2 = {
        "g2_closed_loop_outcome_recorded": "g2_closed_loop_capability_registered",
        "status": "capability_registered",
        "artifact_path": exp4051.G2_CLOSED_LOOP_PATH,
        "per_step_wm_real_divergence_rate": 0.0,
        "closed_loop_broke_wall": True,
    }
    new_registry, _, summary = exp4051.ensure_ledgers_record_outcomes(
        registry,
        "# Verifier Gaps\n",
        {"offline_reeval_bitexact": True},
        g1,
        g2,
    )
    ids = {entry["verifier_id"] for entry in new_registry["verifiers"]}
    assert "gap4_code_armC_symbolic_transfer_4045" in ids
    assert summary == {"registry_updated": True, "gaps_updated": False}


def test_private_helpers_cover_idempotent_and_malformed_paths(tmp_path: Path) -> None:
    # REQ-VERIFY-4051: helper paths keep repeated runs stable and malformed YAML bounded.
    invalid = tmp_path / "registry.yaml"
    invalid.write_text("- not-a-map\n", encoding="utf-8")
    assert exp4051._load_registry(invalid) == {"verifiers": []}

    entry = {"verifier_id": "v", "field": 1}
    registry = {"verifiers": [dict(entry)]}
    assert exp4051._upsert_verifier(registry, dict(entry)) is False
    assert exp4051._upsert_verifier(registry, {"verifier_id": "v", "field": 2}) is True
    assert registry["verifiers"][0]["field"] == 2

    original = exp4051._replace_marked_block("", "marker", "first")
    replaced = exp4051._replace_marked_block(original, "marker", "second")
    assert "first" not in replaced
    assert "second" in replaced

    assert exp4051._registry_contains_outcomes({}, {}, {}) is False
    incomplete = _minimal_registry()
    assert exp4051._registry_contains_outcomes(incomplete, {}, {}) is False


def test_run_hygiene_writes_terminal_artifact_and_ledgers(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4051: run writes the required artifact and updates registry/gaps.
    _write_minimal_repo(tmp_path)
    artifact = exp4051.run_hygiene(tmp_path)
    exp4051.validate_artifact(artifact)

    out_path = tmp_path / exp4051.EXP4051_ARTIFACT_PATH
    assert out_path.exists()
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["honest_verdict"] == (
        "complete: gap4_reeval_bitexact_g1_g1_off_arc_power_pending_"
        "g2_g2_sim2real_ceiling_gap_logged_recorded"
    )
    assert written["offline_reeval_bitexact"] is True
    assert written["registry_updated"] is True
    assert written["gaps_updated"] is True

    registry = yaml.safe_load((tmp_path / "ops" / "verifier_registry.yaml").read_text())
    assert registry["verifiers"][0]["eval"]["eval_exp_4051"] == exp4051.EXP4051_ARTIFACT_PATH
    gaps = (tmp_path / "ops" / "verifier_gaps.md").read_text(encoding="utf-8")
    assert "GAP-ARC3-VC33-SIM2REAL-CEILING" in gaps
