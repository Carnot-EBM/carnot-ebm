"""Tests for Exp 4103 verifier registry/gaps hygiene.

Spec refs: REQ-VERIFY-4103, SCENARIO-VERIFY-4103.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot.reporting import verifier_registry_gaps_hygiene_4103 as exp4103


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
                "eval": {"metric": "pass_at_1"},
                "status": "candidate",
                "training_time_roles": [
                    {
                        "role_id": "verifier_as_reward_rft_4079",
                        "experiment": "results/experiment_4079_verifier_reward_rft_eval_collect.json",
                        "role": "training_time_reward_signal",
                        "status": "blocked",
                    }
                ],
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
    (tmp_path / "ops" / "verifier_gaps.md").write_text("# Verifier Gaps\n", encoding="utf-8")
    for name in (
        "arc3_gap3_stage2_eval_pool.json.gz",
        "arc3_gap4_induced_programs.json",
        "experiment_4099_trm_pool_verifier_discrimination_probe.json",
        "experiment_4100_trm_verifier_rft_conditional.json",
    ):
        shutil.copy2(REPO_ROOT / "results" / name, tmp_path / "results" / name)


def test_req_4103_spec_declared() -> None:
    # REQ-VERIFY-4103: OpenSpec declares the runner and required artifact fields.
    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4103",
        "SCENARIO-VERIFY-4103",
        "exp4103_verifier_registry_gaps_hygiene.py",
        "experiment_4103_verifier_registry_gaps_hygiene.json",
        "GAP-TRM-GRID-DISCRIMINATION",
        "regression_guard_passed",
        "gaps_updated",
        "captured_pp=0.0",
        "verifier_ensemble_against_cached_candidates",
    ):
        assert marker in spec


def test_replay_gap4_arc1_regression_guard_is_bitexact() -> None:
    # SCENARIO-VERIFY-4103: cached candidates reproduce vote 0.4516 -> gated 0.5806.
    replay = exp4103.replay_gap4_arc1(REPO_ROOT)
    assert replay["regression_guard_passed"] is True
    assert replay["arc1_rule_exec"] == {
        "n": 31,
        "vote_pass2": pytest.approx(0.4516),
        "gated_pass2": pytest.approx(0.5806),
        "headroom_recovered": 4,
        "vote_wins_lost": 0,
    }
    assert replay["no_codex_calls"] is True
    assert replay["no_gguf_inference"] is True


def test_classifies_exp4099_actual_open_gap_and_exp4100_smoke() -> None:
    # REQ-VERIFY-4103: present-but-unselectable Exp 4099 logs an open missing-verifier gap.
    gap = exp4103.classify_exp4099_gap(REPO_ROOT)
    assert gap["gap_id"] == exp4103.TRM_GRID_GAP_ID
    assert gap["status"] == "open_captured_pp_0.0000"
    assert gap["verifier_beats_trm_vote"] is False
    assert gap["captured_pp"] == pytest.approx(0.0)
    assert gap["best_reranker"] == "K_OF_N_AGREEMENT"

    outcome = exp4103.classify_exp4100_outcome(REPO_ROOT)
    assert outcome["branch_taken"] == "smoke"
    assert outcome["rft_vs_ablation_delta"]["status"] == "not_run_no_verifier_signal"
    assert outcome["bottleneck"] == "verifier_discrimination_on_trm_grids"
    assert outcome["trm_native_trainer_checkpoint_ok"] is True


def test_ensure_ledgers_record_open_gap_and_training_role() -> None:
    # SCENARIO-VERIFY-4103: registry and gaps carry replay, open gap, and smoke role.
    replay = exp4103.replay_gap4_arc1(REPO_ROOT)
    gap = exp4103.classify_exp4099_gap(REPO_ROOT)
    outcome = exp4103.classify_exp4100_outcome(REPO_ROOT)

    registry, gaps, summary = exp4103.ensure_ledgers_record_outcomes(
        _minimal_registry(),
        "# Verifier Gaps\n",
        replay,
        gap,
        outcome,
    )

    assert summary == {
        "registry_updated": True,
        "gaps_updated": [exp4103.TRM_GRID_GAP_ID, exp4103.TRM_RFT_GAP_ID],
        "trm_grid_gap_recorded": True,
        "exp4100_outcome_recorded": True,
    }
    gap4 = registry["verifiers"][0]
    assert gap4["eval"]["eval_exp_4103"] == exp4103.EXP4103_ARTIFACT_PATH
    assert gap4["eval"]["exp4103_regression_guard_passed"] is True
    role_ids = [role["role_id"] for role in gap4["training_time_roles"]]
    assert "verifier_as_reward_rft_4079" in role_ids
    assert exp4103.TRM_TRAINING_ROLE_ID in role_ids
    role = next(role for role in gap4["training_time_roles"] if role["role_id"] == exp4103.TRM_TRAINING_ROLE_ID)
    assert role["status"] == "smoke"
    assert role["bottleneck"] == "verifier_discrimination_on_trm_grids"
    assert role["rft_vs_ablation_delta"]["delta"] == pytest.approx(0.0)
    assert exp4103.TRM_GRID_GAP_ID in gaps
    assert "captured_pp=0.0" in gaps
    assert "status: open_captured_pp_0.0000" in gaps
    assert exp4103.TRM_RFT_GAP_ID in gaps


def test_ensure_ledgers_creates_missing_registry_entry() -> None:
    # REQ-VERIFY-4103: malformed local registries are repaired before reconciliation.
    replay = exp4103.replay_gap4_arc1(REPO_ROOT)
    gap = exp4103.classify_exp4099_gap(REPO_ROOT)
    outcome = exp4103.classify_exp4100_outcome(REPO_ROOT)

    registry, _gaps, summary = exp4103.ensure_ledgers_record_outcomes(
        {},
        "# Verifier Gaps\n",
        replay,
        gap,
        outcome,
    )

    assert summary["registry_updated"] is True
    assert registry["verifiers"][0]["verifier_id"] == exp4103.GAP4_VERIFIER_ID
    assert exp4103._registry_contains_outcomes({}) is False
    exp4103._ensure_trm_training_role({}, gap, outcome)


def test_classifiers_handle_filled_gap_and_rft_branch(tmp_path: Path) -> None:
    # REQ-VERIFY-4103: filled gaps and RFT deltas are represented when upstream says so.
    (tmp_path / "results").mkdir()
    exp4099 = {
        "honest_verdict": "success: fixture",
        "verifier_beats_trm_vote": True,
        "best_reranker": "DEMO_FIT",
        "captured_pp_directional": 0.125,
        "per_reranker": {"DEMO_FIT": {"captured_pp": 0.125, "captured_pp_ci95": [0.03, 0.21]}},
        "trm_vote_pass2": 0.45,
        "reproducibility_checksum": "gap-fixture",
    }
    (tmp_path / "results" / "experiment_4099_trm_pool_verifier_discrimination_probe.json").write_text(
        json.dumps(exp4099), encoding="utf-8"
    )
    exp4100 = {
        "honest_verdict": "success: fixture",
        "branch_taken": "rft",
        "trm_native_trainer_checkpoint_ok": True,
        "rft_vs_ablation_delta": {"metric": "heldout_pass@2", "delta": 0.14, "ci95": [0.04, 0.23]},
        "verifier_gap": {"bottleneck": "verifier_discrimination_on_trm_grids"},
        "reproducibility_checksum": "rft-fixture",
    }
    (tmp_path / "results" / "experiment_4100_trm_verifier_rft_conditional.json").write_text(
        json.dumps(exp4100), encoding="utf-8"
    )

    gap = exp4103.classify_exp4099_gap(tmp_path)
    outcome = exp4103.classify_exp4100_outcome(tmp_path)

    assert gap["status"] == "filled_by_DEMO_FIT_captured_pp_0.1250"
    assert gap["captured_pp"] == pytest.approx(0.125)
    assert gap["captured_pp_ci95"] == [0.03, 0.21]
    assert outcome["branch_taken"] == "rft"
    assert outcome["status"] == "rft_complete"
    assert outcome["rft_vs_ablation_delta"]["delta"] == pytest.approx(0.14)


def test_build_artifact_validates_required_fields() -> None:
    # REQ-VERIFY-4103: terminal artifact exposes required schema fields and principles.
    replay = exp4103.replay_gap4_arc1(REPO_ROOT)
    gap = exp4103.classify_exp4099_gap(REPO_ROOT)
    outcome = exp4103.classify_exp4100_outcome(REPO_ROOT)
    artifact = exp4103.build_artifact(
        offline_replay=replay,
        exp4099_gap=gap,
        exp4100_outcome=outcome,
        registry_updated=True,
        gaps_updated=[exp4103.TRM_GRID_GAP_ID, exp4103.TRM_RFT_GAP_ID],
        duration_s=0.012,
    )

    exp4103.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["regression_guard_passed"] is True
    assert artifact["gaps_updated"] == [exp4103.TRM_GRID_GAP_ID, exp4103.TRM_RFT_GAP_ID]
    assert artifact["field_principles"]["honest_verdict"].startswith("Terminal-prefixed")
    assert artifact["field_principles"]["regression_guard_passed"].startswith("Bare bool")
    assert artifact["field_principles"]["gaps_updated"].startswith("Lists")


def test_run_hygiene_writes_terminal_artifact_and_ledgers(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4103: run writes the deliverable JSON and stable ledger entries.
    _write_minimal_repo(tmp_path)

    artifact = exp4103.run_hygiene(tmp_path)
    exp4103.validate_artifact(artifact)

    out_path = tmp_path / exp4103.EXP4103_ARTIFACT_PATH
    assert out_path.exists()
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["regression_guard_passed"] is True
    assert written["exp4099_gap"]["captured_pp"] == pytest.approx(0.0)
    assert written["exp4100_outcome"]["branch_taken"] == "smoke"

    registry = yaml.safe_load((tmp_path / "ops" / "verifier_registry.yaml").read_text())
    gap4 = registry["verifiers"][0]
    assert gap4["eval"]["eval_exp_4103"] == exp4103.EXP4103_ARTIFACT_PATH
    assert any(role["role_id"] == exp4103.TRM_TRAINING_ROLE_ID for role in gap4["training_time_roles"])

    gaps = (tmp_path / "ops" / "verifier_gaps.md").read_text(encoding="utf-8")
    assert "GAP-TRM-GRID-DISCRIMINATION" in gaps
    assert "captured_pp=0.0" in gaps
    assert "Exp 4100 TRM verifier-RFT outcome" in gaps
