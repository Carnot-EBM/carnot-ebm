"""Tests for Exp 1566 candidate warm-start vs cold-start benchmark.

Spec refs: REQ-SAMPLE-060, SCENARIO-SAMPLE-088.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carnot.sampling import init_policy_benchmark as exp1566


def test_spec_mentions_exp1566_contract() -> None:
    """REQ-SAMPLE-060, SCENARIO-SAMPLE-088: Exp 1566 is spec-anchored."""

    spec = (
        exp1566.PROJECT_ROOT / "openspec/capabilities/training-inference/spec.md"
    ).read_text(encoding="utf-8")

    assert "REQ-SAMPLE-060" in spec
    assert "SCENARIO-SAMPLE-088" in spec
    assert "experiment_1566_candidate_warm_start_vs_cold_start_benchmark.json" in spec
    assert "candidate-warm-start" in spec


def test_req_sample_060_corpus_and_initialization_policies_are_distinct() -> None:
    """REQ-SAMPLE-060: corpus has N>=200 rows and three explicit init policies."""

    config = exp1566.BenchmarkConfig(n_cases=200, seed=1400)
    cases = exp1566.generate_verification_corpus(config)
    rng = np.random.default_rng(config.seed)

    labels = {case.case_kind for case in cases}
    candidate_init = exp1566.initial_state_matrix(cases, "candidate_warm_start", rng)
    cold_init = exp1566.initial_state_matrix(cases, "cold_start", rng)
    cached_init = exp1566.initial_state_matrix(cases, "cached_state_warm_start", rng)

    assert len(cases) == 200
    assert {"correct", "incorrect", "edge"} <= labels
    assert sum(case.oracle_verdict for case in cases) > sum(not case.oracle_verdict for case in cases)
    assert candidate_init.shape == cold_init.shape == cached_init.shape == (200, config.block_size)
    np.testing.assert_array_equal(candidate_init, np.asarray([case.candidate_bits for case in cases]))
    assert not np.array_equal(cold_init, candidate_init)
    assert not np.array_equal(cached_init, candidate_init)
    assert all(set(case.candidate).issubset({"0", "1"}) for case in cases)

    with pytest.raises(ValueError, match="unknown init policy"):
        exp1566.initial_state_matrix(cases, "reuse_global_cache", rng)


def test_scenario_sample_088_benchmark_meets_dt_mcmc_stateless_gates() -> None:
    """SCENARIO-SAMPLE-088: benchmark validates candidate warm-start and rejects cache reuse."""

    artifact = exp1566.run_benchmark(exp1566.BenchmarkConfig(n_cases=200, seed=1400))

    by_policy = artifact["measurements_by_policy"]
    warm = by_policy["candidate_warm_start"]
    cold = by_policy["cold_start"]
    cached = by_policy["cached_state_warm_start"]

    assert artifact["status"] == "complete"
    assert artifact["candidate_warm_start_validated"] is True
    assert artifact["cold_start_accuracy_drop_percent_at_k100"] >= 50.0
    assert artifact["cached_state_worse_than_cold_start"] is True
    assert artifact["recommended_deployment_policy"] == "candidate_warm_start"
    assert artifact["honest_verdict"].startswith("complete:")
    assert warm["100"]["accuracy"] >= 0.99 * warm["1000"]["accuracy"]
    assert warm["100"]["accuracy"] > cold["100"]["accuracy"]
    assert cached["100"]["accuracy"] < cold["100"]["accuracy"]
    assert set(warm) == {"10", "50", "100", "500", "1000"}
    assert all(row["p95_latency_ms_10ms_granularity"] % 10 == 0 for row in warm.values())


def test_req_sample_060_run_experiment_writes_complete_artifact(tmp_path: Path) -> None:
    """REQ-SAMPLE-060: runner writes the terminal JSON schema."""

    output_path = tmp_path / "experiment_1566.json"

    artifact = exp1566.run_experiment(
        output_path=output_path,
        config=exp1566.BenchmarkConfig(n_cases=200, seed=1400),
    )

    assert exp1566.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["metadata"]["spec_refs"] == ["REQ-SAMPLE-060", "SCENARIO-SAMPLE-088"]
    assert artifact["metadata"]["corpus_size"] >= 200
    assert artifact["acceptance_gates_passed"] is True
    assert artifact["measurements_by_policy"]["cold_start"]["100"]["accuracy"] < artifact[
        "measurements_by_policy"
    ]["cold_start"]["1000"]["accuracy"]
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact


def test_req_sample_060_validate_artifact_rejects_bad_terminal_values() -> None:
    """REQ-SAMPLE-060: artifacts require complete terminal semantics."""

    valid = exp1566.run_benchmark(exp1566.BenchmarkConfig(n_cases=200, seed=1400))

    assert exp1566.validate_artifact(valid) is None

    missing = dict(valid)
    missing.pop("recommended_deployment_policy")
    with pytest.raises(ValueError, match="missing required fields"):
        exp1566.validate_artifact(missing)

    bad_status = dict(valid, status="blocked")
    with pytest.raises(ValueError, match="status must be complete"):
        exp1566.validate_artifact(bad_status)

    bad_verdict = dict(valid, honest_verdict="candidate warm-start validated")
    with pytest.raises(ValueError, match="honest_verdict"):
        exp1566.validate_artifact(bad_verdict)

    bad_warm = dict(valid, candidate_warm_start_validated=False)
    with pytest.raises(ValueError, match="candidate_warm_start_validated"):
        exp1566.validate_artifact(bad_warm)

    bad_drop = dict(valid, cold_start_accuracy_drop_percent_at_k100=49.9)
    with pytest.raises(ValueError, match="cold_start_accuracy_drop_percent_at_k100"):
        exp1566.validate_artifact(bad_drop)

    bad_cache = dict(valid, cached_state_worse_than_cold_start=False)
    with pytest.raises(ValueError, match="cached_state_worse_than_cold_start"):
        exp1566.validate_artifact(bad_cache)
