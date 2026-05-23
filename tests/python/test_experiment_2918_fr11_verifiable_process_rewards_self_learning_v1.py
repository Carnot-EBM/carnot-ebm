"""Tests for Exp 2918 FR-11 verifiable process-reward replay.

Spec: REQ-LEARN-2918,
      SCENARIO-LEARN-2918,
      SCENARIO-LEARN-2918-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import fr11_verifiable_process_rewards_self_learning_v1 as exp


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_ready_2911(root: Path) -> None:
    _write_json(
        root / "results" / exp.EXP2911_FILENAME,
        {
            "artifact": "experiment_2911_code_hallucination_taxonomy_verifier_v1",
            "code_hallucination_verifier_ready": True,
            "honest_verdict": "complete: deterministic taxonomy ready",
            "per_candidate_labels": [
                {
                    "candidate_index": 0,
                    "labels": [],
                    "passed": True,
                    "pass_status": "passed",
                    "row_status": "candidate_passed",
                    "stable_id": "task-pass",
                    "syntax_success": True,
                    "task_key": "MBPP:task-pass",
                },
                {
                    "candidate_index": 1,
                    "labels": ["undefined_name", "invented_import"],
                    "passed": False,
                    "pass_status": "failed",
                    "row_status": "candidate_failed",
                    "stable_id": "task-static",
                    "syntax_success": True,
                    "task_key": "MBPP:task-static",
                },
                {
                    "candidate_index": 2,
                    "labels": ["syntax_error"],
                    "passed": False,
                    "pass_status": "failed",
                    "row_status": "candidate_syntax_failed",
                    "stable_id": "task-syntax",
                    "syntax_success": False,
                    "task_key": "MBPP:task-syntax",
                },
                {
                    "candidate_index": 3,
                    "labels": ["true_test_failure"],
                    "passed": False,
                    "pass_status": "failed",
                    "row_status": "candidate_failed",
                    "stable_id": "task-test",
                    "syntax_success": True,
                    "task_key": "MBPP:task-test",
                },
            ],
        },
    )


def _write_ready_2912(root: Path) -> None:
    _write_json(
        root / "results" / exp.EXP2912_FILENAME,
        {
            "artifact": "experiment_2912_kv260_same_basis_cpu_gibbs_baseline_v1",
            "honest_verdict": "complete: same_basis_cpu_gibbs_baseline_ready_no_speedup_claim",
            "same_basis_cpu_baseline_ready": True,
            "matched_sparse_topology": True,
            "matched_coupling_tensor": True,
            "matched_field_tensor": True,
            "cpu_per_seed_results": [
                {
                    "cpu_latency_us_median": 100.0,
                    "final_energy": -20.0,
                    "sample_count": 100,
                    "seed": 42,
                },
                {
                    "cpu_latency_us_median": 200.0,
                    "final_energy": -10.0,
                    "sample_count": 100,
                    "seed": 137,
                },
                {
                    "cpu_latency_us_median": 120.0,
                    "final_energy": -18.0,
                    "sample_count": 1000,
                    "seed": 42,
                },
            ],
        },
    )


def _write_prior_fr11(root: Path) -> None:
    _write_json(
        root / "results" / exp.EXP2887_FILENAME,
        {
            "artifact": "experiment_2887_fr11_fast_slow_memory_corrigendum_v2",
            "best_policy": "fast_slow_memory",
            "fr11_scaleup_clean": True,
            "policy_metrics": {
                "fast_slow_memory": {
                    "applied_event_ids": ["prior-a", "prior-b", "prior-c"],
                    "contradiction_rate": 0.0,
                    "energy_delta_mean": 0.2,
                    "forgetting_regression_count": 0,
                }
            },
        },
    )
    _write_json(
        root / "results" / exp.EXP2906_FILENAME,
        {
            "artifact": "experiment_2906_fr11_hardware_accelerated_replay_pilot_v1",
            "dispatch_path_validated": True,
            "honest_verdict": "complete: kv260_replay_dispatch_path_validated_pilot_only",
        },
    )


def test_scenario_learn_2918_blocked_prerequisites_write_required_schema(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-2918-BLOCKED: missing or unready prerequisites fail closed."""

    missing = exp.run_experiment(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            results_dir=tmp_path / "results",
            started_at=1.0,
            clock=lambda: 2.5,
        )
    )

    assert set(exp.REQUIRED_ARTIFACT_FIELDS).issubset(missing)
    assert missing["honest_verdict"] == "blocked_missing_exp2911_artifact"
    assert missing["online_self_learning_ready"] is False
    assert missing["online_update_performed"] is False
    assert missing["replay_scheduler_updated"] is False
    assert missing["duration_s"] == pytest.approx(1.5)
    assert (tmp_path / "results" / exp.OUTPUT_FILENAME).is_file()

    _write_json(
        tmp_path / "results" / exp.EXP2911_FILENAME,
        {"code_hallucination_verifier_ready": False},
    )
    _write_ready_2912(tmp_path)
    unready = exp.run_experiment(
        exp.ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        write=False,
    )

    assert unready["honest_verdict"] == "blocked_exp2911_not_ready"
    assert "exp2911_not_ready" in unready["failed_preconditions"]


def test_scenario_learn_2918_verified_replay_updates_scheduler(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-2918: verifier process rewards update replay priorities."""

    _write_ready_2911(tmp_path)
    _write_ready_2912(tmp_path)
    _write_prior_fr11(tmp_path)

    artifact = exp.run_experiment(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            results_dir=tmp_path / "results",
            started_at=10.0,
            clock=lambda: 14.0,
        )
    )

    assert set(exp.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["online_self_learning_ready"] is True
    assert artifact["fr11_requirement_targeted"] == "FR-11"
    assert artifact["online_update_performed"] is True
    assert artifact["replay_scheduler_updated"] is True
    assert artifact["hardware_replay_used"] is True
    assert artifact["inference_substrate"] == "deterministic_verifier_plus_replay"
    assert artifact["run_date"] == "20260523"
    assert artifact["duration_s"] == pytest.approx(4.0)

    assert artifact["replay_corpus_summary"]["code_rows"] == 4
    assert artifact["replay_corpus_summary"]["hardware_rows"] == 3
    assert artifact["replay_corpus_summary"]["prior_fr11_rows"] == 3
    assert artifact["replay_corpus_summary"]["held_out_prior_rows"] == 1
    assert artifact["process_reward_definition"]["code"]["drives"] == "syntax_static_runtime"
    assert artifact["process_reward_definition"]["hardware"]["drives"] == "basis_energy_latency"

    assert artifact["delta_overall"] > 0.0
    assert artifact["delta_energy_proxy"] > 0.0
    assert artifact["contradiction_rate_after"] <= artifact["contradiction_rate_before"]
    assert artifact["forgetting_rate"] == pytest.approx(0.0)
    assert artifact["pdi_proxy"] > 0.0
    assert artifact["model_weights_mutated"] is False
    assert artifact["scheduler_update_scope"] == "replay_priority_table_only_no_model_weights"
    assert artifact["priority_summary"]["mean_after"] != artifact["priority_summary"]["mean_before"]

    saved = json.loads((tmp_path / "results" / exp.OUTPUT_FILENAME).read_text(encoding="utf-8"))
    assert saved == artifact


def test_req_learn_2918_reward_helpers_are_deterministic() -> None:
    """REQ-LEARN-2918-2/3: process rewards are deterministic verifier functions."""

    passed = exp.code_process_reward(
        {"passed": True, "syntax_success": True, "labels": [], "row_status": "candidate_passed"}
    )
    static_fail = exp.code_process_reward(
        {
            "passed": False,
            "syntax_success": True,
            "labels": ["undefined_name", "invented_import"],
            "row_status": "candidate_failed",
        }
    )
    syntax_fail = exp.code_process_reward(
        {
            "passed": False,
            "syntax_success": False,
            "labels": ["syntax_error"],
            "row_status": "candidate_syntax_failed",
        }
    )
    extracted_fail = exp.code_process_reward(
        {"passed": False, "syntax_success": True, "labels": [], "row_status": "candidate_failed"}
    )

    assert passed.weight == pytest.approx(0.95)
    assert static_fail.weight == pytest.approx(0.25)
    assert syntax_fail.weight < static_fail.weight
    assert extracted_fail.weight < passed.weight
    assert exp.code_process_reward(
        {"passed": True, "syntax_success": True, "labels": [], "row_status": "candidate_passed"}
    ) == passed

    fast_low_energy = exp.hardware_process_reward(
        {"final_energy": -20.0, "cpu_latency_us_median": 100.0},
        min_energy=-20.0,
        max_energy=-10.0,
        min_latency=100.0,
        max_latency=200.0,
        basis_matched=True,
    )
    slow_high_energy = exp.hardware_process_reward(
        {"final_energy": -10.0, "cpu_latency_us_median": 200.0},
        min_energy=-20.0,
        max_energy=-10.0,
        min_latency=100.0,
        max_latency=200.0,
        basis_matched=True,
    )
    unmatched = exp.hardware_process_reward(
        {"final_energy": -20.0, "cpu_latency_us_median": 100.0},
        min_energy=-20.0,
        max_energy=-10.0,
        min_latency=100.0,
        max_latency=200.0,
        basis_matched=False,
    )

    assert fast_low_energy.weight == pytest.approx(0.95)
    assert slow_high_energy.weight == pytest.approx(0.45)
    assert unmatched.weight == pytest.approx(0.45)

