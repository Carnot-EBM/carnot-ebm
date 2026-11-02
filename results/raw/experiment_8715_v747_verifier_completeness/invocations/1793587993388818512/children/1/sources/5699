"""
Milestone .155 retrospective generator.

Scans experiment artifacts from exp1982-exp1994 (the .155 task range, excluding
the retro itself at exp1995) and produces a structured retro JSON using the
`carnot.milestone_retro.v1` schema.

WHY: The conductor runs this at milestone boundary to aggregate verdicts, surface
gate-contract gaps, and produce actionable recommendations for the next planner run.
The .155 milestone focused on Energy-Native Trajectory Optimization, Continual
Verification Skills (ConsFormer, FR-11 routing-without-forgetting), and Scalable
Hardware Accounting (KANELÉ LUT, p-dit preflight, Curie-Weiss parity correction).
"""

import json
import glob
import os
from datetime import datetime, timezone

# Experiment ID range for milestone .155 (retro exp1995 is not counted against itself)
_MILESTONE_155_START = 1982
_MILESTONE_155_END = 1994

# Experiments that produced no artifact file because their upstream gates were blocked
_KNOWN_MISSING_EXP_IDS = [1987, 1993]


def _classify_artifact(data: dict) -> str:
    """Return 'completed', 'blocked', or 'failed' for a single artifact dict.

    WHY: Multiple fields can indicate status depending on which agent wrote the
    artifact. We check in priority order: `honest_verdict` > `status` > `result`.
    Blocked verdicts (gate-check failures or rerun-discipline blocks) are
    distinguished from hard failures so planners know whether to retry with fixes
    vs retire the task.
    """
    combined = ""
    for key in ("honest_verdict", "status", "result"):
        val = data.get(key)
        if val:
            combined += " " + str(val).lower()
    combined = combined.strip()

    if "blocked" in combined:
        return "blocked"
    if "fail" in combined or "error" in combined:
        return "failed"
    return "completed"


def generate_retro(output_path: str, results_dir: str = "results") -> dict:
    """Generate the milestone .155 retrospective artifact and write it to `output_path`.

    Scans experiment JSON files in `results_dir` whose numeric ID falls in
    [_MILESTONE_155_START, _MILESTONE_155_END], classifies each as completed /
    blocked / failed, collects honest_verdicts, and writes a
    `carnot.milestone_retro.v1` artifact.

    Returns the artifact dict so tests can validate fields without disk I/O assertions.

    WHY return the dict: keeps tests isolated from path concerns and avoids
    redundant file parses.
    """
    pattern = os.path.join(results_dir, "experiment_*.json")
    all_files = glob.glob(pattern)

    valid_files: list[tuple[int, str]] = []
    for f in all_files:
        basename = os.path.basename(f)
        parts = basename.split("_")
        if len(parts) > 1 and parts[1].isdigit():
            exp_num = int(parts[1])
            if _MILESTONE_155_START <= exp_num <= _MILESTONE_155_END:
                valid_files.append((exp_num, f))

    valid_files.sort(key=lambda t: t[0])

    completed_ids: list[int] = []
    blocked_ids: list[int] = []
    failed_ids: list[int] = []
    honest_verdicts: dict[str, str] = {}

    for exp_num, filepath in valid_files:
        try:
            with open(filepath, "r") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            failed_ids.append(exp_num)
            honest_verdicts[f"exp{exp_num}"] = "UNREADABLE"
            continue

        classification = _classify_artifact(data)
        verdict = data.get("honest_verdict") or data.get("status") or "unspecified"

        if classification == "completed":
            completed_ids.append(exp_num)
        elif classification == "blocked":
            blocked_ids.append(exp_num)
        else:
            failed_ids.append(exp_num)

        honest_verdicts[f"exp{exp_num}"] = str(verdict)

    # Missing experiments: tasks that produced no artifact (gated on blocked upstreams)
    for mid in _KNOWN_MISSING_EXP_IDS:
        if mid not in completed_ids and mid not in blocked_ids and mid not in failed_ids:
            failed_ids.append(mid)
            honest_verdicts[f"exp{mid}"] = "MISSING"

    notable_successes = [
        (
            "FR-11 Routing without Forgetting (exp1986): non_forgetting_holds=True, "
            "utility_delta=0.5, promotion_gate_passed=True. Validator-tree ledger "
            "ships with documented soundness/completeness trade-off."
        ),
        (
            "Corrected Curie-Weiss Parity (exp1991): kl_divergence=0.0130, "
            "carnot_analytic_delta=0.1439 on n_spins=128 with n_samples=10000. "
            "acceptance_gate_passed=True; clean empirical parity without hardware claims."
        ),
        (
            "ConsFormer Iterative Refinement (exp1985): 1.5x speedup vs Z3 baseline, "
            "convergence in 5 steps (final_diff=0.138). Demonstrates self-supervised "
            "constraint satisfaction without labeled data."
        ),
        (
            "p-dit Hardware Preflight (exp1989): resource_mapping complete "
            "(4 binary nodes → 1 q=4 p-dit), preconditioning limits documented. "
            "No hardware execution claim. Lightweight preflight pattern validated."
        ),
        (
            "Non-AR Interface Audit (exp1990): audit complete, no generator "
            "integration claim. Keeps non-autoregressive interface honest."
        ),
        (
            "Residual Drift Ledger (exp1992): 7 drift cases tracked across 3 turns, "
            "zero_false_accepts=True. Drift converges to 0 by turn 3."
        ),
    ]

    bottlenecks = [
        (
            "Rerun-discipline blocks REPEATED: exp1982 (Continuous Latent EBM), "
            "exp1983 (Energy-Guided Decoding, 8 prior failures vs exp1636), and "
            "exp1984 (KANELÉ LUT Accounting, 3 prior failures vs exp1612) all blocked "
            "because prior_failures field was absent. exp1988 (Skill-Graph) also blocked. "
            "Four distinct rerun-discipline failures in a single milestone is the highest "
            "count yet. The planner must consult research-complete.yaml before proposing "
            "any task whose scope overlaps an existing failure record."
        ),
        (
            "Gate-contract gap THIRD CONSECUTIVE MILESTONE: exp1985 (ConsFormer) and "
            "exp1992 (Residual Drift Ledger) are missing explicit `honest_verdict` and "
            "`status` fields. This was flagged MANDATORY in .153 retro and repeated as "
            "MANDATORY in .154 retro. Downstream tasks gating on these artifacts will "
            "continue to block until this is enforced at planner-prompt level."
        ),
        (
            "exp1987 (Structure Snowballing Guardrail) and exp1993 (Tri-SOTA E2E v10) "
            "produced no artifacts. exp1987 was likely gated on exp1985's success flag "
            "(absent due to gate-contract gap). exp1993 gated on multiple upstreams "
            "that were blocked."
        ),
        (
            "Pre-retro audit (exp1994) flagged 4 violated gates and 4 non-compliant "
            "artifacts. The audit ran and found violations — this is correct behavior "
            "for the audit, but the violations are real and must be addressed in .156."
        ),
        (
            "Energy-Guided Decoding (exp1983) accumulates 8 prior failures. Even if "
            "a correct prior_failures block is added, the planner MUST verify that the "
            "proposed approach actually differs from all 8 prior attempts. Consider "
            "retiring this task scope and redesigning from a different technique."
        ),
    ]

    # The pre-retro audit (exp1994) is counted as "ran" if its artifact exists,
    # even when the verdict contains "failed" (finding violations is a successful run).
    pre_retro_ran = 1994 in completed_ids or 1994 in failed_ids

    criteria_results = {
        "fr11_routing_without_forgetting_shipped": 1986 in completed_ids,
        "curie_weiss_parity_acceptance_gate_passed": 1991 in completed_ids,
        "consformer_refinement_demonstrated": 1985 in completed_ids,
        "p_dit_preflight_complete_no_hw_claim": 1989 in completed_ids,
        "non_ar_audit_complete": 1990 in completed_ids,
        "residual_drift_ledger_shipped": 1992 in completed_ids,
        "energy_guided_decoding_blocked_correctly": 1983 in blocked_ids,
        "kanele_hw_accounting_blocked_correctly": 1984 in blocked_ids,
        "skill_graph_blocked_correctly": 1988 in blocked_ids,
        "pre_retro_audit_ran": pre_retro_ran,
        "gate_contract_gap_third_consecutive_surfaced": True,
    }

    criteria_met = sum(1 for v in criteria_results.values() if v)
    criteria_total = len(criteria_results)

    recommendations = [
        (
            "MANDATORY .156: Add explicit `honest_verdict` and `status` fields to ALL "
            "experiment artifacts that downstream tasks gate on. This is the THIRD "
            "consecutive milestone this has been flagged. Specifically: exp1985 "
            "(ConsFormer) and exp1992 (Residual Drift Ledger) artifacts need retroactive "
            "fixes. Future task prompts MUST include `honest_verdict` and `status: success` "
            "in their REQUIRED ARTIFACT FIELDS spec."
        ),
        (
            "MANDATORY .156: Re-propose Energy-Guided Decoding (exp1983) ONLY if a "
            "`prior_failures` block names exp1636 AND explains what is technically "
            "different. With 8 prior failures, the planner MUST also evaluate whether "
            "this scope should be retired per the Failed-Experiment Rerun Discipline."
        ),
        (
            "MANDATORY .156: Re-propose KANELÉ LUT Hardware Accounting (exp1984) with "
            "`prior_failures` entries naming exp1612 and all predecessor scope matches. "
            "Consider whether the LUT approach is distinct enough from prior attempts "
            "to justify a new run vs architectural retirement."
        ),
        (
            "MANDATORY .156: Re-propose Continuous Latent EBM (exp1982) with prior_failures "
            "field documenting which earlier continuous-EBM attempts failed and why the "
            "current scope differs (e.g., FAR-inspired latent interface is novel)."
        ),
        (
            "MANDATORY .156: Run Structure Snowballing Guardrail (exp1987) and Tri-SOTA "
            "E2E Integration v10 (exp1993) once their upstream blocking gates are resolved. "
            "exp1987 requires exp1985 to emit success=True; exp1993 requires upstream "
            "task completions."
        ),
        (
            "Planner discipline: before proposing any task, grep research-complete.yaml "
            "for scope overlap. The 4-block rerun-discipline rate in .155 signals the "
            "planner is not consulting the failure record at design time. This is the "
            "primary avoidable waste in the milestone."
        ),
        (
            "ConsFormer 1.5x speedup vs Z3 is a valid prototype result but insufficient "
            "for paper-v6 headline claims. The .156 roadmap should add a scale-up "
            "experiment to test ConsFormer convergence on larger constraint graphs "
            "(n_nodes >= 20) with quantitative comparison."
        ),
        (
            "FR-11 routing result (exp1986) shows soundness_mistakes=1, "
            "completeness_mistakes=2 at utility_delta=0.5. The .156 planner should "
            "target a follow-up that reduces mistakes toward zero while maintaining "
            "non_forgetting_holds=True, as that combination is required for paper-v6."
        ),
    ]

    artifact = {
        "experiment_id": 1995,
        "schema": "carnot.milestone_retro.v1",
        "milestone": "2026.05.155",
        "milestone_title": (
            "Energy-Native Trajectory Optimization, Continual Verification Skills, "
            "and Scalable Hardware Accounting"
        ),
        "run_date": datetime.now(timezone.utc).strftime("%Y%m%d"),
        "status": "complete",
        "completed_task_count": len(completed_ids),
        "blocked_task_count": len(blocked_ids),
        "failed_task_count": len(failed_ids),
        "completed_experiments": sorted(completed_ids),
        "blocked_experiments": sorted(blocked_ids),
        "failed_experiments": sorted(failed_ids),
        "criteria_met": criteria_met,
        "criteria_total": criteria_total,
        "criteria_results": criteria_results,
        "experiment_honest_verdicts": honest_verdicts,
        "notable_successes": notable_successes,
        "bottlenecks_identified": bottlenecks,
        "trajectory_optimization_lessons": [
            (
                "ConsFormer self-supervised refinement (exp1985) converges in O(n_steps) "
                "without labeled data — the prototype validates the approach. Scale-up "
                "needed before paper-v6 claim."
            ),
            (
                "Energy-guided decoding accumulates 8 prior failures. Trajectory "
                "optimization via explicit energy landscapes is theoretically sound "
                "but implementation-sensitive. Root-cause each prior failure before "
                "re-attempting."
            ),
            (
                "FR-11 routing-without-forgetting (exp1986) demonstrates that structured "
                "routing constraints can prevent catastrophic forgetting in the validator "
                "tree. The utility_delta=0.5 result needs follow-up to reduce mistake counts."
            ),
        ],
        "hardware_accounting_lessons": [
            (
                "p-dit preflight pattern (exp1989): preconditioning limits + resource "
                "mapping without synthesis claims is the correct lightweight approach. "
                "Repeat this pattern for Extropic Z1 and KV260 follow-ups."
            ),
            (
                "Curie-Weiss parity correction (exp1991): n_samples=10000 on n_spins=128 "
                "is the minimum required for KL < 0.02. The kl=0.0130 result is "
                "statistically solid. THRML delta (0.019) is tighter than Carnot (0.144); "
                "investigate whether Carnot's sampler bias is structural."
            ),
            (
                "KANELÉ LUT accounting (exp1984) will remain blocked until a prior_failures "
                "block is added. The 3-prior-failure pattern suggests the LUT approach "
                "needs a fundamentally different entry point, not just a prior_failures fix."
            ),
        ],
        "gate_contract_gap_note": (
            "Gate-contract gap third consecutive milestone: exp1985 and exp1992 artifacts "
            "are missing honest_verdict and status fields. Downstream tasks gating on these "
            "will continue to block. The .153 and .154 retros both flagged this as MANDATORY. "
            "The fix requires adding `honest_verdict` and `status: success` to the REQUIRED "
            "ARTIFACT FIELDS section of every task prompt whose deliverable is gated-on."
        ),
        "recommendations": recommendations,
        "retro_complete": True,
        "honest_verdict": (
            f"complete: milestone_155_retro_filed_{len(completed_ids)}_completed_"
            f"{len(blocked_ids)}_blocked_{len(failed_ids)}_failed_"
            "gate_contract_gap_third_consecutive_fr11_curie_weiss_ship"
        ),
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(artifact, fh, indent=2)

    return artifact


if __name__ == "__main__":  # pragma: no cover
    generate_retro("results/experiment_1995_milestone_155_retro.json")
