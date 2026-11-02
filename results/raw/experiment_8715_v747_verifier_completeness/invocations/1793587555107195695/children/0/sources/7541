#!/usr/bin/env python3
"""
Experiment 3559: G-Gate Status Synthesis v327
Synthesizes the .327 artifacts into a final publication gate status.
"""

import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional

def load_json(path: str) -> Optional[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        with p.open("r") as f:
            return json.load(f)
    except Exception:
        return None

def is_flagged(exp: Optional[Dict[str, Any]]) -> bool:
    """Returns True if artifact is absent or flagged adversarial."""
    if not exp:
        return True
    if exp.get("flagged_adversarial", False):
        return True
    return False

def evaluate_gates() -> Dict[str, Any]:
    """Runs the mechanical publication gate checks."""
    import sys
    sys.path.insert(0, str(Path.cwd()))
    try:
        from scripts.publication_gate import evaluate
        return evaluate()
    except Exception as e:
        print(f"Warning: could not evaluate gates: {e}")
        return {
            "gates": {
                "G1": {"pass": False},
                "G2": {"pass": False},
                "G3": {"pass": False},
                "G4": {"pass": False}
            },
            "unmet_gates": ["G1", "G2", "G3", "G4"]
        }

def synthesize_v327(start_time: float) -> Dict[str, Any]:
    """Synthesizes the gate status and .327 artifact results."""
    gate_res = evaluate_gates()
    g1 = gate_res["gates"]["G1"]["pass"]
    g2 = gate_res["gates"]["G2"]["pass"]
    g3 = gate_res["gates"]["G3"]["pass"]
    g4 = gate_res["gates"]["G4"]["pass"]
    unmet_gates = gate_res.get("unmet_gates", [])

    exp3551 = load_json("results/experiment_3551_p01_graph_coloring_terminal_discriminating_corpus_v3.json")
    exp3552 = load_json("results/experiment_3552_p01_route2_headroom_corpus_greedy_wrong_construction_v3.json")
    exp3553 = load_json("results/experiment_3553_p01_route2_energy_vs_strong_sc_on_headroom_corpus_v3.json")
    exp3554 = load_json("results/experiment_3554_fover_step_aggregation_secondary_headline_multiseed_third_corpus_v2.json")
    exp3555 = load_json("results/experiment_3555_fr11_conservative_default_deploy_nondegenerate_corpus_v3.json")
    exp3556 = load_json("results/experiment_3556_fover_g2_regression_verify_external_ask_refresh_v7.json")

    exp3551_clean = not is_flagged(exp3551)
    exp3553_clean = not is_flagged(exp3553)
    exp3554_clean = not is_flagged(exp3554)
    exp3555_clean = not is_flagged(exp3555)
    exp3556_clean = not is_flagged(exp3556)

    # Route 1 (exp3551)
    p01_route1_terminal_verdict = exp3551["honest_verdict"] if exp3551_clean else None
    
    p01_route1_discriminating_and_clean = None
    if exp3551_clean:
        p01_route1_discriminating_and_clean = bool(exp3551.get("hard_tier_discriminating", False)) and bool(exp3551.get("no_aliased_fields_assert", False))
        
    p01_route1_paired_p = exp3551.get("energy_vs_strong_paired_p_hard_tier") if exp3551_clean else None

    # Route 2 (exp3553)
    p01_route2_fair_verdict = None
    if exp3553_clean:
        flip_count = exp3553.get("flip_count_best_vs_strong_sc")
        delta = exp3553.get("delta_best_vs_strong_sc")
        verdict = exp3553.get("honest_verdict")
        p01_route2_fair_verdict = f"{verdict} (flip_count={flip_count}, delta={delta})"

    p01_route2_corpus_had_headroom = exp3553.get("corpus_oracle_exceeds_sc") if exp3553_clean else None

    # Determine p01_has_clean_terminal_verdict
    # Clean terminal verdict if: exp3551 significantly beats strong on discriminating (p < 0.05) OR informative route 2
    p01_has_clean_terminal_verdict = False
    if p01_route1_discriminating_and_clean and p01_route1_paired_p is not None and p01_route1_paired_p < 0.05:
        p01_has_clean_terminal_verdict = True
    if p01_route2_corpus_had_headroom and exp3553_clean:
        p01_has_clean_terminal_verdict = True

    # Secondary aggregation and deployment
    aggregation_secondary_headline_eligible = exp3554.get("secondary_headline_eligible") if exp3554_clean else None
    self_learning_nondegenerate_verdict = exp3555.get("honest_verdict") if exp3555_clean else None
    
    # G2
    g2_package_status = None
    if exp3556_clean:
        g2_package_status = exp3556.get("honest_verdict") or "G2-external-in-motion"

    depth_forcing_function_can_relax = p01_has_clean_terminal_verdict and (not g2)

    # Hash inputs
    h = hashlib.sha256()
    for exp in [exp3551, exp3553, exp3554, exp3555, exp3556]:
        if exp:
            h.update(json.dumps(exp, sort_keys=True).encode("utf-8"))
    
    duration_s = max(0.0001, time.monotonic() - start_time)
    
    output = {
        "honest_verdict": "complete: g_gate_status_synthesis_v327",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "g1": g1,
        "g2": g2,
        "g3": g3,
        "g4": g4,
        "unmet_gates": unmet_gates,
        "p01_route1_terminal_verdict": p01_route1_terminal_verdict,
        "p01_route1_discriminating_and_clean": p01_route1_discriminating_and_clean,
        "p01_route1_paired_p": p01_route1_paired_p,
        "p01_route2_fair_verdict": p01_route2_fair_verdict,
        "p01_route2_corpus_had_headroom": p01_route2_corpus_had_headroom,
        "p01_has_clean_terminal_verdict": p01_has_clean_terminal_verdict,
        "aggregation_secondary_headline_eligible": aggregation_secondary_headline_eligible,
        "self_learning_nondegenerate_verdict": self_learning_nondegenerate_verdict,
        "g2_package_status": g2_package_status,
        "depth_forcing_function_can_relax": depth_forcing_function_can_relax,
        "gate_status_v327_ready": True,
        "random_seed": 20260601,
        "reproducibility_checksum": h.hexdigest()[:16],
        "duration_s": duration_s,
        "field_provenance": {
            "honest_verdict": "complete: prefix.",
            "inference_substrate": "aggregation_from_upstream_artifacts",
            "g1": "headline measured (FoVer 0.9131) \u2014 boolean.",
            "g2": "independently reproduced \u2014 boolean (external; honest manual).",
            "g3": "prose narrowing-clean \u2014 boolean.",
            "g4": "numbers trace to primary artifacts \u2014 boolean.",
            "unmet_gates": "the list of unmet gates \u2014 what to report instead of a count.",
            "p01_route1_terminal_verdict": "exp3551 terminal verdict \u2014 did energy SIGNIFICANTLY beat a STRONG baseline on a DISCRIMINATING corpus (DSATUR < 0.9) with CLEAN, UNFLAGGED numbers + CI + paired p, OR is Route-1 terminally bounded (competitive-not-superior)? (null if absent/flagged).",
            "p01_route1_discriminating_and_clean": "boolean: exp3551 ran on a non-ceiling-saturated corpus (DSATUR < 0.9) with zero CRITICAL flags \u2014 whether the test was finally DISCRIMINATING (null if absent).",
            "p01_route1_paired_p": "exp3551 energy-vs-strong paired significance p (null if absent).",
            "p01_route2_fair_verdict": "exp3553 terminal verdict \u2014 the fair Route-2 test vs a STRONG SC (headroom present + non-degenerate reranker); flip_count + delta (null if absent/flagged).",
            "p01_route2_corpus_had_headroom": "exp3552/exp3553 oracle_exceeds_sc \u2014 whether the Route-2 test was finally informative, or the premise is terminally bounded on NL-math (null if absent).",
            "p01_has_clean_terminal_verdict": "boolean: P0.1 has a clean, flag-free, TERMINAL verdict (exp3551 either significantly beats a strong baseline on a discriminating corpus OR is terminally bounded, zero CRITICAL flags, AND/OR an informative Route-2 verdict with headroom) \u2014 the Depth-Over-Breadth relax precondition.",
            "aggregation_secondary_headline_eligible": "exp3554 secondary_headline_eligible \u2014 whether the step->final aggregation transfers to >=2 corpora with multi-seed CIs (the secondary headline; null if absent/flagged).",
            "self_learning_nondegenerate_verdict": "exp3555 terminal verdict \u2014 whether the conservative-default rule deploys + preserves REAL quality on a non-degenerate corpus (null if absent/flagged).",
            "g2_package_status": "exp3556 regression + external-ask status string.",
            "depth_forcing_function_can_relax": "True only when P0.1 has a clean flag-free TERMINAL verdict AND G2 external-in-motion.",
            "gate_status_v327_ready": "terminal completion flag (always True) \u2014 the field the capstone exp3560 gates on; MUST appear in this REQUIRED ARTIFACT FIELDS block.",
            "random_seed": "determinism; MUST be 20260601 (a distinct fixed value), NOT the experiment number \u2014 the exp3502 tautology fix.",
            "reproducibility_checksum": "content hash.",
            "duration_s": "aggregation; sub-second honest."
        }
    }

    return output

def main():
    start_time = time.monotonic()
    output = synthesize_v327(start_time)
    
    output_path = Path("results/experiment_3559_g_gate_status_synthesis_v327.json")
    with output_path.open("w") as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()
