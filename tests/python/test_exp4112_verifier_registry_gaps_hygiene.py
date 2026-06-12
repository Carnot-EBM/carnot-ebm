"""Tests for Exp 4112 verifier registry/gaps hygiene.

Spec refs: REQ-VERIFY-4112, SCENARIO-VERIFY-4112.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot.reporting import verifier_registry_gaps_hygiene_4112 as exp4112


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
                        "role_id": "trm_grid_discriminator_training_time_4100",
                        "experiment": "results/experiment_4100_trm_verifier_rft_conditional.json",
                        "role": "candidate_trm_training_time_reward_signal",
                        "status": "smoke",
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
        "experiment_4109_carnot_verifier_graft_sudoku.json",
    ):
        shutil.copy2(REPO_ROOT / "results" / name, tmp_path / "results" / name)


def test_req_4112_spec_declared() -> None:
    # REQ-VERIFY-4112: OpenSpec declares the runner and required artifact fields.
    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4112",
        "SCENARIO-VERIFY-4112",
        "exp4112_verifier_registry_gaps_hygiene.py",
        "experiment_4112_verifier_registry_gaps_hygiene.json",
        "GAP-TRM-GRID-DISCRIMINATION",
        "GAP-SUDOKU-EXECUTABLE-VERIFIER-4109",
        "captured_pp=-0.2258",
        "verifier_value_added=false",
        "regression_guard_passed",
        "gaps_updated",
    ):
        assert marker in spec
    assert exp4112.FIELD_PRINCIPLES["honest_verdict"] in spec
    assert exp4112.FIELD_PRINCIPLES["regression_guard_passed"] in spec
    assert exp4112.FIELD_PRINCIPLES["gaps_updated"] in spec


def test_scenario_4112_replay_gap4_arc1_regression_guard_is_bitexact() -> None:
    # SCENARIO-VERIFY-4112: cached candidates reproduce vote 0.4516 -> gated 0.5806.
    replay = exp4112.replay_gap4_arc1(REPO_ROOT)
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


def test_req_4112_classifies_379_anti_discrimination_and_380_sudoku_null() -> None:
    # REQ-VERIFY-4112: present-but-unselectable .379 and .380 outcomes stay honest gaps.
    trm = exp4112.classify_trm_grid_discrimination(REPO_ROOT)
    assert trm["gap_id"] == exp4112.TRM_GRID_GAP_ID
    assert trm["status"] == "open_anti_discrimination_captured_pp_-0.2258"
    assert trm["best_reranker"] == "K_OF_N_AGREEMENT"
    assert trm["best_captured_pp"] == pytest.approx(0.0)
    assert trm["anti_discrimination_captured_pp"] == pytest.approx(-0.2258)
    assert trm["anti_discrimination_captured_pp_rounded"] == pytest.approx(-0.23)
    assert "DEMO_FIT" in trm["anti_discriminating_rerankers"]
    assert trm["verifier_beats_trm_vote"] is False

    sudoku = exp4112.classify_sudoku_verifier(REPO_ROOT)
    assert sudoku["gap_id"] == exp4112.SUDOKU_GAP_ID
    assert sudoku["status"] == "open_honest_null_no_value_added"
    assert sudoku["verifier_value_added"] is False
    assert sudoku["native_training_launched"] is False
    assert sudoku["rft_vs_ablation_delta"]["delta"] == pytest.approx(0.0)
    assert sudoku["rft_vs_ablation_delta"]["status"] == "honest_null_ci95_includes_zero"
    assert sudoku["corpus_summary"]["n_matched"] == 15


def test_scenario_4112_ensure_ledgers_record_gaps_and_training_role() -> None:
    # SCENARIO-VERIFY-4112: registry and gaps carry replay, .379 gap, and .380 role.
    replay = exp4112.replay_gap4_arc1(REPO_ROOT)
    trm = exp4112.classify_trm_grid_discrimination(REPO_ROOT)
    sudoku = exp4112.classify_sudoku_verifier(REPO_ROOT)

    registry, gaps, summary = exp4112.ensure_ledgers_record_outcomes(
        _minimal_registry(),
        "# Verifier Gaps\n",
        replay,
        trm,
        sudoku,
    )

    assert summary == {
        "registry_updated": True,
        "gaps_updated": [exp4112.TRM_GRID_GAP_ID, exp4112.SUDOKU_GAP_ID],
        "trm_grid_gap_recorded": True,
        "sudoku_gap_recorded": True,
    }
    gap4 = registry["verifiers"][0]
    assert gap4["eval"]["eval_exp_4112"] == exp4112.EXP4112_ARTIFACT_PATH
    assert gap4["eval"]["exp4112_regression_guard_passed"] is True
    role = next(
        role
        for role in gap4["training_time_roles"]
        if role["role_id"] == exp4112.SUDOKU_TRAINING_ROLE_ID
    )
    assert role["status"] == "honest_null_no_value_added"
    assert role["verifier_value_added"] is False
    assert role["native_training_launched"] is False
    assert role["rft_vs_ablation_delta"]["delta"] == pytest.approx(0.0)
    assert exp4112._registry_contains_outcomes(registry) is True
    assert exp4112._registry_contains_outcomes({}) is False
    repaired_registry, _repaired_gaps, repaired_summary = exp4112.ensure_ledgers_record_outcomes(
        {},
        "# Verifier Gaps\n",
        replay,
        trm,
        sudoku,
    )
    assert repaired_summary["registry_updated"] is True
    assert repaired_registry["verifiers"][0]["verifier_id"] == exp4112.GAP4_VERIFIER_ID
    assert exp4112.TRM_GRID_GAP_ID in gaps
    assert "captured_pp=-0.2258" in gaps
    assert "captured_pp_rounded=-0.23" in gaps
    assert exp4112.SUDOKU_GAP_ID in gaps
    assert "verifier_value_added=false" in gaps


def test_req_4112_build_artifact_validates_required_fields() -> None:
    # REQ-VERIFY-4112: terminal artifact exposes required schema fields and principles.
    replay = exp4112.replay_gap4_arc1(REPO_ROOT)
    trm = exp4112.classify_trm_grid_discrimination(REPO_ROOT)
    sudoku = exp4112.classify_sudoku_verifier(REPO_ROOT)
    artifact = exp4112.build_artifact(
        offline_replay=replay,
        trm_grid_discrimination=trm,
        sudoku_verifier=sudoku,
        registry_updated=True,
        gaps_updated=[exp4112.TRM_GRID_GAP_ID, exp4112.SUDOKU_GAP_ID],
        duration_s=0.012,
    )

    exp4112.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["regression_guard_passed"] is True
    assert artifact["gaps_updated"] == [exp4112.TRM_GRID_GAP_ID, exp4112.SUDOKU_GAP_ID]
    assert artifact["field_principles"] == exp4112.FIELD_PRINCIPLES
    assert artifact["trm_grid_discrimination"]["anti_discrimination_captured_pp"] == pytest.approx(
        -0.2258
    )
    assert artifact["sudoku_verifier"]["verifier_value_added"] is False

    for field in exp4112.REQUIRED_ARTIFACT_FIELDS:
        malformed = dict(artifact)
        malformed.pop(field)
        with pytest.raises(ValueError, match=field):
            exp4112.validate_artifact(malformed)
    with pytest.raises(ValueError, match="terminal prefix"):
        exp4112.validate_artifact({**artifact, "honest_verdict": "not terminal"})
    with pytest.raises(ValueError, match="bare bool"):
        exp4112.validate_artifact({**artifact, "regression_guard_passed": "yes"})
    with pytest.raises(ValueError, match="bare bool"):
        exp4112.validate_artifact({**artifact, "registry_updated": "yes"})
    with pytest.raises(ValueError, match="list"):
        exp4112.validate_artifact({**artifact, "gaps_updated": "GAP"})
    with pytest.raises(ValueError, match="inference_substrate"):
        exp4112.validate_artifact({**artifact, "inference_substrate": "live"})


def test_scenario_4112_run_hygiene_writes_terminal_artifact_and_ledgers(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4112: run writes the deliverable JSON and stable ledger entries.
    _write_minimal_repo(tmp_path)
    artifact = exp4112.run_hygiene(tmp_path)
    exp4112.validate_artifact(artifact)

    out_path = tmp_path / exp4112.EXP4112_ARTIFACT_PATH
    assert out_path.exists()
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["regression_guard_passed"] is True
    assert written["trm_grid_discrimination"]["anti_discrimination_captured_pp"] == pytest.approx(
        -0.2258
    )
    assert written["sudoku_verifier"]["verifier_value_added"] is False

    registry = yaml.safe_load((tmp_path / "ops" / "verifier_registry.yaml").read_text())
    role_ids = [role["role_id"] for role in registry["verifiers"][0]["training_time_roles"]]
    assert exp4112.SUDOKU_TRAINING_ROLE_ID in role_ids
    gaps = (tmp_path / "ops" / "verifier_gaps.md").read_text(encoding="utf-8")
    assert "Exp 4112 .379 TRM-grid anti-discrimination update" in gaps
    assert "Exp 4112 .380 Sudoku executable-verifier update" in gaps
