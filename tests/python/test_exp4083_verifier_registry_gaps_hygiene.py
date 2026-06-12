"""Tests for Exp 4083 verifier registry/gaps hygiene.

Spec refs: REQ-VERIFY-4083, SCENARIO-VERIFY-4083.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot.reporting import verifier_registry_gaps_hygiene_4083 as exp4083


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
        "arc3_gap4_rule_exec_verifier.json",
        "experiment_4079_verifier_reward_rft_eval_collect.json",
    ):
        shutil.copy2(REPO_ROOT / "results" / name, tmp_path / "results" / name)


def _blocked_pivot() -> dict[str, Any]:
    return {
        "experiment": 4079,
        "status": "blocked",
        "honest_verdict": "blocked_gate_check_failed",
        "gate_check_summary": "1 of 1 gate(s) failed; first failure: fixture",
        "blocked_at_layer": "conductor_pre_gate",
    }


def test_req_4083_spec_declared() -> None:
    # REQ-VERIFY-4083: OpenSpec declares the 4083 runner and required fields.
    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4083",
        "SCENARIO-VERIFY-4083",
        "exp4083_verifier_registry_gaps_hygiene.py",
        "experiment_4083_verifier_registry_gaps_hygiene.json",
        "gap4_arc1_reproduced",
        "pivot_outcome_recorded",
        "safety_gate_regression_passed",
        "verifier_ensemble_against_cached_candidates",
    ):
        assert marker in spec


def test_replay_gap4_arc1_from_cached_artifact_bitexact() -> None:
    # SCENARIO-VERIFY-4083: cached ARC-1 artifact reproduces the shipped safety-gate numbers.
    replay = exp4083.replay_gap4_arc1(REPO_ROOT)
    assert replay["gap4_arc1_reproduced"] is True
    assert replay["safety_gate_regression_passed"] is True
    assert replay["arc1_rule_exec"] == {
        "n": 31,
        "vote_pass2": pytest.approx(0.4516),
        "gated_pass2": pytest.approx(0.5806),
        "headroom_recovered": 4,
        "vote_wins_lost": 0,
    }


def test_replay_gap4_arc1_detects_drift() -> None:
    # REQ-VERIFY-4083: replay failure preserves observed and expected values.
    replay = exp4083.replay_gap4_arc1_fixture(
        {
            "n_tasks": 31,
            "rankers": {
                "TRM_VOTE": {"pass@2": 0.4516},
                "GAP4_GATED": {"pass@2": 0.5484},
            },
            "gates": {"headroom_recovered": 3, "vote_wins_lost": 1},
        }
    )
    assert replay["gap4_arc1_reproduced"] is False
    assert replay["safety_gate_regression_passed"] is False
    assert replay["arc1_rule_exec"]["gated_pass2"] == pytest.approx(0.5484)
    assert replay["expected"]["arc1_rule_exec"]["gated_pass2"] == pytest.approx(0.5806)


def test_classifies_actual_4079_blocked_pivot() -> None:
    # REQ-VERIFY-4083: Exp 4079 blocked gate is recorded as blocked, not as a win.
    outcome = exp4083.classify_pivot_outcome(REPO_ROOT)
    assert outcome["pivot_outcome_recorded"] == "verifier_as_reward_rft_blocked"
    assert outcome["status"] == "blocked"
    assert outcome["training_time_role"] is True
    assert outcome["honest_verdict"] == "blocked_gate_check_failed"
    assert outcome["blocked_at_layer"] == "conductor_pre_gate"


def test_pivot_classifier_handles_complete_and_missing(tmp_path: Path) -> None:
    # REQ-VERIFY-4083: pivot classification is bounded for complete and absent artifacts.
    assert exp4083.classify_pivot_outcome(tmp_path)["pivot_outcome_recorded"] == (
        "verifier_as_reward_rft_pending"
    )

    results = tmp_path / "results"
    results.mkdir()
    path = results / "experiment_4079_verifier_reward_rft_eval_collect.json"
    path.write_text(
        json.dumps(
            {
                "experiment": 4079,
                "status": "complete",
                "honest_verdict": "complete: verifier_label_signal_absent",
                "arm_a_vs_b_delta": 0.0,
            }
        ),
        encoding="utf-8",
    )
    outcome = exp4083.classify_pivot_outcome(tmp_path)
    assert outcome["pivot_outcome_recorded"] == "verifier_as_reward_rft_complete"
    assert outcome["status"] == "complete"

    path.write_text(
        json.dumps({"experiment": 4079, "status": "running", "honest_verdict": "running"}),
        encoding="utf-8",
    )
    outcome = exp4083.classify_pivot_outcome(tmp_path)
    assert outcome["pivot_outcome_recorded"] == "verifier_as_reward_rft_accumulating"
    assert outcome["status"] == "running"


def test_ensure_ledgers_record_pivot_and_registry_training_role() -> None:
    # SCENARIO-VERIFY-4083: ledgers record replay and blocked training-time pivot.
    replay = {
        "gap4_arc1_reproduced": True,
        "safety_gate_regression_passed": True,
        "arc1_rule_exec": {
            "n": 31,
            "vote_pass2": 0.4516,
            "gated_pass2": 0.5806,
            "headroom_recovered": 4,
            "vote_wins_lost": 0,
        },
    }
    pivot = exp4083.classify_pivot_outcome_fixture(_blocked_pivot())

    registry, gaps, summary = exp4083.ensure_ledgers_record_outcomes(
        _minimal_registry(),
        "# Verifier Gaps\n",
        replay,
        pivot,
    )

    assert summary == {
        "registry_updated": True,
        "gaps_updated": True,
        "pivot_outcome_recorded": True,
    }
    gap4 = registry["verifiers"][0]
    assert gap4["eval"]["eval_exp_4083"] == exp4083.EXP4083_ARTIFACT_PATH
    assert gap4["eval"]["exp4083_arc1_safety_gate_regression_passed"] is True
    assert gap4["training_time_roles"][0]["status"] == "blocked"
    assert "GAP-TRAINING-VERIFIER-AS-REWARD-RFT-4079" in gaps
    assert "verifier_as_reward_rft_blocked" in gaps
    assert "blocked_gate_check_failed" in gaps


def test_build_artifact_validates_required_fields() -> None:
    # REQ-VERIFY-4083: terminal artifact exposes required bare bools and substrate.
    replay = exp4083.replay_gap4_arc1(REPO_ROOT)
    pivot = exp4083.classify_pivot_outcome_fixture(_blocked_pivot())
    artifact = exp4083.build_artifact(
        offline_replay=replay,
        pivot_outcome=pivot,
        registry_updated=True,
        gaps_updated=True,
        pivot_outcome_recorded=True,
        duration_s=0.0123,
    )

    exp4083.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["gap4_arc1_reproduced"] is True
    assert artifact["pivot_outcome_recorded"] is True
    assert artifact["inference_substrate"] == exp4083.INFERENCE_SUBSTRATE


def test_run_hygiene_writes_terminal_artifact_and_ledgers(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4083: run writes the deliverable JSON and stable ledger entries.
    _write_minimal_repo(tmp_path)
    artifact = exp4083.run_hygiene(tmp_path)
    exp4083.validate_artifact(artifact)

    out_path = tmp_path / exp4083.EXP4083_ARTIFACT_PATH
    assert out_path.exists()
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["gap4_arc1_reproduced"] is True
    assert written["safety_gate_regression_passed"] is True
    assert written["pivot_outcome_recorded"] is True
    assert written["inference_substrate"] == exp4083.INFERENCE_SUBSTRATE

    registry = yaml.safe_load((tmp_path / "ops" / "verifier_registry.yaml").read_text())
    gap4 = registry["verifiers"][0]
    assert gap4["eval"]["eval_exp_4083"] == exp4083.EXP4083_ARTIFACT_PATH
    assert gap4["training_time_roles"][0]["outcome"] == "verifier_as_reward_rft_blocked"
    gaps = (tmp_path / "ops" / "verifier_gaps.md").read_text(encoding="utf-8")
    assert "GAP-TRAINING-VERIFIER-AS-REWARD-RFT-4079" in gaps
    assert "training-time verifier role" in gaps
