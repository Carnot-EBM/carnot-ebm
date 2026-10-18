"""Verifier diversity remediation plan data structures.

Spec: REQ-VERIFY-3341
"""
import json
import dataclasses
from typing import List

@dataclasses.dataclass
class VerifierDiversityRemediationPlan:
    source_audit: str
    lambda_min_sigma_before: float
    effective_k_before: float
    collapsed_pairs: List[str]
    proposed_axis: str
    retired_scopes_avoided: List[str]
    acceptance_criteria: str
    downstream_tasks: List[str]

    def validate(self) -> bool:
        if self.proposed_axis in self.retired_scopes_avoided:
            return False
        if "diversity-maximizing greedy selection" not in self.retired_scopes_avoided:
            return False
        if "greedy verifier selection" not in self.retired_scopes_avoided:
            return False
        return True

def save_plan_artifact(path: str, plan: VerifierDiversityRemediationPlan, 
                      honest_verdict: str, inference_substrate: str, 
                      random_seed: int, reproducibility_checksum: str, 
                      duration_s: float, files_updated: List[str]) -> None:
    data = {
        "honest_verdict": honest_verdict,
        "inference_substrate": inference_substrate,
        "random_seed": random_seed,
        "reproducibility_checksum": reproducibility_checksum,
        "duration_s": duration_s,
        "files_updated": files_updated,
        "source_audit": plan.source_audit,
        "lambda_min_sigma_before": plan.lambda_min_sigma_before,
        "effective_k_before": plan.effective_k_before,
        "collapsed_pairs": plan.collapsed_pairs,
        "proposed_axis": plan.proposed_axis,
        "retired_scopes_avoided": plan.retired_scopes_avoided,
        "acceptance_criteria": plan.acceptance_criteria,
        "downstream_tasks": plan.downstream_tasks
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
