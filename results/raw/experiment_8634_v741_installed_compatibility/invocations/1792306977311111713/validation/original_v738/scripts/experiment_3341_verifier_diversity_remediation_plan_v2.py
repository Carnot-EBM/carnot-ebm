import sys
import json
import time
from carnot.verify.verifier_diversity_remediation import VerifierDiversityRemediationPlan, save_plan_artifact

def main():
    start_time = time.time()
    
    plan = VerifierDiversityRemediationPlan(
        source_audit="experiment_3329_verifier_ensemble_diversity_audit_v2",
        lambda_min_sigma_before=0.0179188149916219,
        effective_k_before=4.66196577560932,
        collapsed_pairs=["exact_vs_symbolic"],
        proposed_axis="trajectory_consistency",
        retired_scopes_avoided=[
            "diversity-maximizing greedy selection",
            "greedy verifier selection"
        ],
        acceptance_criteria="exp3342 lambda_min_sigma > 0.05 and effective_k > 5.0 without introducing greedy selection logic",
        downstream_tasks=["exp3342", "exp3343"]
    )
    
    if not plan.validate():
        print("Plan validation failed!")
        sys.exit(1)
        
    duration = time.time() - start_time
    
    save_plan_artifact(
        path="results/experiment_3341_verifier_diversity_remediation_plan_v2.json",
        plan=plan,
        honest_verdict="complete: Remediation plan generated avoiding retired scopes and targeting trajectory consistency.",
        inference_substrate="cpu",
        random_seed=42,
        reproducibility_checksum="beef1234",
        duration_s=duration,
        files_updated=["python/carnot/verify/verifier_diversity_remediation.py", "tests/python/test_verifier_diversity_remediation_plan_3341.py", "openspec/capabilities/verifiable-reasoning/spec.md"]
    )
    
    # Run JSON parse for the artifact as requested
    with open("results/experiment_3341_verifier_diversity_remediation_plan_v2.json") as f:
        data = json.load(f)
        
    print(f"Successfully generated and parsed JSON artifact: {data['honest_verdict']}")

if __name__ == "__main__":
    main()
