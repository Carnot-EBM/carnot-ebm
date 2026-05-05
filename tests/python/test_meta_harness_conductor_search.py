from pathlib import Path

from scripts.meta_harness_conductor_search import (
    CandidatePolicy,
    candidate_has_hardcoded_leakage,
    default_eval_cases,
    default_candidate_policies,
    evaluate_policy,
    run_search,
)


def test_default_eval_suite_has_required_cases():
    """Spec: REQ-HARNESS-010, SCENARIO-HARNESS-005."""

    cases = default_eval_cases()

    assert len(cases) >= 8
    assert {case.case_id for case in cases} >= {
        "bootstrap_only_artifact",
        "gated_downstream_task",
        "local_verifier_mismatch",
        "malformed_json_artifact",
        "blocked_missing_tool",
    }
    assert sum(1 for case in cases if case.held_out) >= 2


def test_default_candidate_set_has_baseline_and_five_candidates():
    """Spec: REQ-HARNESS-010, SCENARIO-HARNESS-006."""

    policies = default_candidate_policies()

    assert len(policies) >= 5
    assert policies[0].candidate_id == "candidate_000"
    assert any("acceptance_alignment" in policy.capabilities for policy in policies)


def test_evaluate_policy_rewards_full_acceptance_alignment():
    """Spec: REQ-HARNESS-010, SCENARIO-HARNESS-006."""

    cases = default_eval_cases()
    baseline, *_, full_policy = default_candidate_policies()

    baseline_score = evaluate_policy(baseline, cases)
    full_score = evaluate_policy(full_policy, cases)

    assert full_score["score"] > baseline_score["score"]
    assert full_score["eval_cases_passed"] == len(cases)
    assert full_score["false_complete_count"] == 0


def test_hardcoded_leakage_audit_flags_experiment_ids():
    """Spec: REQ-HARNESS-012, SCENARIO-HARNESS-007."""

    leaking = CandidatePolicy(
        candidate_id="candidate_999",
        title="Leaking policy",
        capabilities=("structured_gate",),
        policy_text="Special-case exp1281 and call it complete.",
        recommended_changes=("Do not use this.",),
        average_runtime_s=0.1,
    )

    assert candidate_has_hardcoded_leakage(leaking)
    assert not any(
        candidate_has_hardcoded_leakage(policy) for policy in default_candidate_policies()
    )


def test_run_search_writes_trace_store_and_terminal_artifact(tmp_path: Path):
    """Spec: REQ-HARNESS-010, REQ-HARNESS-011, SCENARIO-HARNESS-005."""

    trace_store = tmp_path / "meta_harness_runs"
    result_path = tmp_path / "result.json"

    result = run_search(trace_store, result_path)

    assert result_path.exists()
    assert result["candidate_harnesses_evaluated"] >= 5
    assert result["eval_cases_defined"] >= 8
    assert result["best_score"] > result["baseline_score"]
    assert result["pareto_frontier_written"] is True
    assert result["trace_store_written"] is True
    assert result["hardcoded_leakage_audit_passed"] is True
    assert result["honest_verdict"] == "meta_harness_conductor_search_complete"

    candidate_dirs = sorted(trace_store.glob("candidate_*"))
    assert len(candidate_dirs) >= 5
    for candidate_dir in candidate_dirs:
        assert (candidate_dir / "policy.md").exists()
        assert (candidate_dir / "score.json").exists()
        assert (candidate_dir / "traces/verifier_outputs.jsonl").exists()
        assert (candidate_dir / "results/final_artifact.json").exists()


def test_run_search_records_pareto_frontier(tmp_path: Path):
    """Spec: REQ-HARNESS-011, SCENARIO-HARNESS-006."""

    trace_store = tmp_path / "meta_harness_runs"
    result = run_search(trace_store, tmp_path / "result.json")

    assert (trace_store / "frontier.json").exists()
    assert result["pareto_frontier"]
    assert result["pareto_frontier"][0]["candidate_id"] == result["best_candidate_id"]
