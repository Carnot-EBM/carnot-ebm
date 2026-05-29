import os
import json
import tempfile
from carnot.verify.verifier_diversity_remediation import VerifierDiversityRemediationPlan, save_plan_artifact

def test_remediation_plan_validation():
    # Spec: REQ-VERIFY-3341
    plan = VerifierDiversityRemediationPlan(
        source_audit="experiment_3329",
        lambda_min_sigma_before=0.0179,
        effective_k_before=4.66,
        collapsed_pairs=["exact_vs_symbolic"],
        proposed_axis="trajectory_consistency",
        retired_scopes_avoided=[
            "diversity-maximizing greedy selection",
            "greedy verifier selection"
        ],
        acceptance_criteria="exp3342 lambda_min_sigma > 0.05",
        downstream_tasks=["exp3342", "exp3343"]
    )
    assert plan.validate() is True

def test_remediation_plan_validation_fails_if_retired_missing():
    # Spec: REQ-VERIFY-3341
    plan = VerifierDiversityRemediationPlan(
        source_audit="experiment_3329",
        lambda_min_sigma_before=0.0179,
        effective_k_before=4.66,
        collapsed_pairs=["exact_vs_symbolic"],
        proposed_axis="trajectory_consistency",
        retired_scopes_avoided=["some other thing"],
        acceptance_criteria="exp3342 lambda_min_sigma > 0.05",
        downstream_tasks=["exp3342", "exp3343"]
    )
    assert plan.validate() is False

def test_remediation_plan_validation_fails_if_proposed_is_retired():
    # Spec: REQ-VERIFY-3341
    plan = VerifierDiversityRemediationPlan(
        source_audit="experiment_3329",
        lambda_min_sigma_before=0.0179,
        effective_k_before=4.66,
        collapsed_pairs=["exact_vs_symbolic"],
        proposed_axis="diversity-maximizing greedy selection",
        retired_scopes_avoided=[
            "diversity-maximizing greedy selection",
            "greedy verifier selection"
        ],
        acceptance_criteria="exp3342 lambda_min_sigma > 0.05",
        downstream_tasks=["exp3342", "exp3343"]
    )
    assert plan.validate() is False

def test_save_plan_artifact():
    plan = VerifierDiversityRemediationPlan(
        source_audit="experiment_3329",
        lambda_min_sigma_before=0.0179,
        effective_k_before=4.66,
        collapsed_pairs=["exact_vs_symbolic"],
        proposed_axis="trajectory_consistency",
        retired_scopes_avoided=[
            "diversity-maximizing greedy selection",
            "greedy verifier selection"
        ],
        acceptance_criteria="exp3342 lambda_min_sigma > 0.05",
        downstream_tasks=["exp3342", "exp3343"]
    )
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "test.json")
        save_plan_artifact(
            path=p,
            plan=plan,
            honest_verdict="complete: test",
            inference_substrate="cpu",
            random_seed=42,
            reproducibility_checksum="abc",
            duration_s=1.0,
            files_updated=[]
        )
        with open(p) as f:
            data = json.load(f)
        assert data["honest_verdict"] == "complete: test"
        assert data["proposed_axis"] == "trajectory_consistency"

def test_remediation_plan_validation_fails_if_greedy_verifier_missing():
    # Spec: REQ-VERIFY-3341
    plan = VerifierDiversityRemediationPlan(
        source_audit="experiment_3329",
        lambda_min_sigma_before=0.0179,
        effective_k_before=4.66,
        collapsed_pairs=["exact_vs_symbolic"],
        proposed_axis="trajectory_consistency",
        retired_scopes_avoided=[
            "diversity-maximizing greedy selection",
        ],
        acceptance_criteria="exp3342 lambda_min_sigma > 0.05",
        downstream_tasks=["exp3342", "exp3343"]
    )
    assert plan.validate() is False
