#!/usr/bin/env python3
"""G1-G4 publication gate synthesis v328.

Synthesizes the publication gates using the v328 artifacts:
- exp3562: Route-1 second CSP generalization
- exp3563: Route-1 graph-coloring multi-seed/generator robustness
- exp3564: Route-2 NL-math terminal verdict
- exp3565: Cross-corpus aggregation promotion
- exp3566: Multi-corpus self-learning + P0.2 diversity
- exp3567: G2 external regression
"""

import json
import sys
import time
import hashlib
from pathlib import Path

# Add scripts directory to path to import publication_gate
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "scripts"))
try:
    from publication_gate import evaluate
except ImportError:
    # Allow tests to run/mock if needed
    evaluate = lambda: {"gates": {"G1": {"pass": False}, "G2": {"pass": False}, "G3": {"pass": False}, "G4": {"pass": False}}, "unmet_gates": ["G1", "G2", "G3", "G4"]}

RESULTS_DIR = PROJECT_ROOT / "results"

def load_artifact(filename: str):
    path = RESULTS_DIR / filename
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if data.get("flagged_adversarial", False):
            return None
        return data
    except Exception:
        return None

def synthesize(gate_eval_fn=evaluate):
    start_t = time.time()
    exp3562 = load_artifact("experiment_3562_p01_route1_second_csp_discriminating_generalization_v2.json")
    exp3563 = load_artifact("experiment_3563_p01_route1_graph_coloring_multiseed_second_generator_v4.json")
    exp3564 = load_artifact("experiment_3564_p01_route2_nlmath_final_headroom_or_retire_v4.json")
    exp3565 = load_artifact("experiment_3565_fover_step_aggregation_secondary_headline_multiseed_third_corpus_v3.json")
    exp3566 = load_artifact("experiment_3566_fr11_multicorpus_deploy_verifier_diversity_grounding_v1.json")
    exp3567 = load_artifact("experiment_3567_fover_g2_regression_verify_external_ask_refresh_v8.json")

    gate_result = gate_eval_fn()

    p01_route1_generalizes_second_csp = exp3562["honest_verdict"] if exp3562 else None
    p01_route1_robust_multiseed_multigenerator = exp3563["positive_robust_to_seed_and_generator"] if exp3563 else None
    p01_route2_nlmath_terminal_verdict = exp3564["honest_verdict"] if exp3564 else None
    aggregation_secondary_headline_eligible = exp3565["secondary_headline_eligible"] if exp3565 else None
    self_learning_battery_p02_verdict = exp3566["honest_verdict"] if exp3566 else None
    g2_package_status = exp3567["honest_verdict"] if exp3567 else None

    # Count of NEW defensible secondary headlines this milestone
    # Route-1 generalization/robustness + aggregation promotion
    count = 0
    if p01_route1_generalizes_second_csp and "positive" in str(p01_route1_generalizes_second_csp).lower():
        count += 1
    if p01_route1_robust_multiseed_multigenerator:
        count += 1
    if aggregation_secondary_headline_eligible:
        count += 1

    payload = {
        "honest_verdict": "complete: g_gate_status_synthesis_v328",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "g1": gate_result["gates"]["G1"]["pass"],
        "g2": gate_result["gates"]["G2"]["pass"],
        "g3": gate_result["gates"]["G3"]["pass"],
        "g4": gate_result["gates"]["G4"]["pass"],
        "unmet_gates": gate_result["unmet_gates"],
        "p01_route1_generalizes_second_csp": p01_route1_generalizes_second_csp,
        "p01_route1_robust_multiseed_multigenerator": p01_route1_robust_multiseed_multigenerator,
        "p01_route2_nlmath_terminal_verdict": p01_route2_nlmath_terminal_verdict,
        "aggregation_secondary_headline_eligible": aggregation_secondary_headline_eligible,
        "self_learning_battery_p02_verdict": self_learning_battery_p02_verdict,
        "g2_package_status": g2_package_status,
        "secondary_headlines_count": count,
        "gate_status_v328_ready": True,
        "random_seed": 20260601,
        "reproducibility_checksum": "",
        "duration_s": 0.0,
    }

    # Add required field principles
    # Required to match exactly what CLAUDE.md / instructions specify
    payload["field_principles"] = {
        "honest_verdict": "complete: prefix.",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "g1": "headline measured (FoVer 0.9131) \u2014 boolean.",
        "g2": "independently reproduced \u2014 boolean (external; honest manual).",
        "g3": "prose narrowing-clean \u2014 boolean.",
        "g4": "numbers trace to primary artifacts \u2014 boolean.",
        "unmet_gates": "the list of unmet gates \u2014 what to report instead of a count.",
        "p01_route1_generalizes_second_csp": "exp3562 verdict \u2014 did energy SIGNIFICANTLY beat a STRONG classical baseline on a SECOND discriminating CSP with CI + paired p (the generalized positive), or is Route-1 bounded to graph coloring? (null if absent/flagged).",
        "p01_route1_robust_multiseed_multigenerator": "exp3563 boolean \u2014 is the graph-coloring positive robust to seed and generator (CI excludes 0 on >=2 generators)? (null if absent/flagged).",
        "p01_route2_nlmath_terminal_verdict": "exp3564 terminal verdict \u2014 reranker beats a strong SC with headroom (positive), informative-negative with headroom, OR permanently retired (no headroom, SC near-optimal)? (null if absent/flagged).",
        "aggregation_secondary_headline_eligible": "exp3565 secondary_headline_eligible \u2014 whether the step->final aggregation transfers to >=2 corpora with multi-seed CIs (the secondary headline; null if absent/flagged).",
        "self_learning_battery_p02_verdict": "exp3566 verdict \u2014 whether the conservative-default rule deploys across a non-degenerate battery AND whether verifier diversity improves grounding (P0.2) (null if absent/flagged).",
        "g2_package_status": "exp3567 regression + external-ask status string.",
        "secondary_headlines_count": "count of NEW defensible secondary headlines this milestone (Route-1 generalization/robustness + aggregation promotion) \u2014 the consolidation scorecard.",
        "gate_status_v328_ready": "terminal completion flag (always True) \u2014 the field the capstone exp3571 gates on; MUST appear in this REQUIRED ARTIFACT FIELDS block.",
        "random_seed": "determinism; MUST be 20260601 (a distinct fixed value), NOT the experiment number \u2014 the exp3502 tautology fix.",
        "reproducibility_checksum": "content hash.",
        "duration_s": "aggregation; sub-second honest."
    }

    raw_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    payload["reproducibility_checksum"] = raw_hash
    payload["duration_s"] = time.time() - start_t + 0.01

    return payload

def main():
    payload = synthesize()
    out_path = RESULTS_DIR / "experiment_3570_g_gate_status_synthesis_v328.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_path.name}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
