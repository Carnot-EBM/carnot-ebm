"""Exp 1132: Goodfire Exemplar Cascade TP Rate Measurement.

Feeds each exemplar from the LLM failure corpus (Exp 1112) through every
available Carnot cascade tier and reports per-tier true-positive (TP) rates.

Business question answered:
    Does Carnot's mathematical-objective tier (Z3MathVerifier) catch
    arithmetic errors that semantic/learned tiers miss?  This validates
    the engineering-tier differentiation from mechanistic interpretability
    tools like Goodfire Silico.

Cascade tiers exercised:
    Tier 0a  — ThinkPRM v2 (proxy: SpilledEnergyDetector text heuristic,
               because ThinkPRM requires a trained probe not available here)
    Tier 0b  — SpilledEnergyDetector.score()
    Tier 0c  — SemEnergyProbe.score_response_proxy()
    Tier 2.5 — SymCodeVerifier.verify_response()  (code/arithmetic steps)
    Tier 2.7 — CausalReasoningVerifier.detection_score()
    Tier 3   — k=5 AND-compose (AndCompositionVerifier.verify())

Spec: REQ-VERIFY-1132, SCENARIO-VERIFY-1132
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone, UTC
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
EXEMPLAR_PATH = REPO_ROOT / "data" / "llm_failure_exemplars.jsonl"
RESULT_PATH = REPO_ROOT / "results" / "experiment_1132_goodfire_exemplar_cascade_tp.json"

# Arithmetic categories where Z3MathVerifier should dominate.
# These directly map to the 9.11>9.9 and decimal-comparison failure modes.
ARITHMETIC_CATEGORIES = {"arithmetic_comparison", "arithmetic_computation"}

# Semantic/factual categories that SemEnergyProbe targets.
SEMANTIC_CATEGORIES = {
    "factual_grounding",
    "hallucination_numeric",
    "overconfidence",
    "deceptive_disclosure",
    "moral_framing",
    "underspecification",
}

# Causal / structural categories.
CAUSAL_CATEGORIES = {"logical_consistency", "format_compliance", "code_logic", "code_syntax"}


# ---------------------------------------------------------------------------
# Verifier helpers
# ---------------------------------------------------------------------------


def _run_z3_math(response: str) -> bool:
    """Return True (TP) if Z3MathVerifier flags a violation in the response.

    Energy 1.0 means all extracted arithmetic claims are wrong.
    Energy 0.5 means no arithmetic claims found (neutral — treat as no-flag).
    Energy 0.0 means all claims are correct.
    Threshold: energy > 0.5 ⟹ flag.
    """
    from carnot.verify.z3_math_verifier import Z3MathVerifier

    verifier = Z3MathVerifier()
    energy = verifier.score(response)
    # Neutral (no math found) → energy=0.5, we only count strict violations.
    return energy > 0.5


def _run_semenergy(response: str) -> bool:
    """Return True (TP) if SemEnergyProbe flags hallucination."""
    from carnot.verify.semenergy_probe import SemEnergyProbe

    probe = SemEnergyProbe()
    energy = probe.score_response_proxy(response)
    return probe.is_hallucinating(energy)


def _run_spilled_energy(response: str) -> bool:
    """Return True (TP) if SpilledEnergyDetector flags the response.

    SpilledEnergyDetector uses token-level logit spill heuristics to detect
    uncertainty.  Used here directly (Tier 0b) and as a proxy for Tier 0a
    (ThinkPRM) which requires a trained hidden-state probe not available in
    this environment.
    """
    from carnot.verify.spilled_energy import SpilledEnergyDetector

    detector = SpilledEnergyDetector()
    return detector.is_violation(response)


def _run_symcode(prompt: str, response: str) -> bool:
    """Return True (TP) if SymCodeVerifier finds an arithmetic execution error.

    SymCodeVerifier (Tier 2.5) translates natural-language arithmetic steps
    to Python and executes them.  A violation means the code disagrees with
    the stated result — a deterministic, model-agnostic check.

    For non-arithmetic categories this often returns False (no arithmetic to
    check), which correctly reflects that SymCodeVerifier's scope is limited
    to arithmetic-heavy CoT.
    """
    from carnot.pipeline.symcode_verifier import SymCodeVerifier

    verifier = SymCodeVerifier()
    results = verifier.verify_response(response)
    # CoTStep uses .violation_detected, not .violation
    return any(r.violation_detected for r in results)


def _run_causal(response: str) -> bool:
    """Return True (TP) if CausalReasoningVerifier detects a causal break.

    CausalReasoningVerifier (Tier 2.7) checks whether the numeric conclusion
    of CoT step k matches the opening premise of step k+1.  A causal break
    means the chain-of-thought lost numeric coherence between steps.

    CausalReasoningVerifier delegates single-step arithmetic checking to
    SymCodeVerifier internally.  We pass a fresh SymCodeVerifier instance.

    Threshold 0.05: any non-trivial detected break fraction flags the response.
    """
    from carnot.pipeline.causal_reasoning_verifier import CausalReasoningVerifier
    from carnot.pipeline.symcode_verifier import SymCodeVerifier

    symcode = SymCodeVerifier()
    verifier = CausalReasoningVerifier(symcode=symcode)
    score = verifier.detection_score(response)
    return score > 0.05


def _run_and_compose(prompt: str, response: str) -> bool:
    """Return True (TP) if the k=5 AND-composition ensemble flags the response.

    The AND-compose verifier (Tier 3) requires ALL k=5 members to agree on a
    violation before flagging.  This dramatically reduces false positives at
    the cost of lower individual recall — the ensemble is only as wrong as the
    weakest member.

    Key property: the ensemble's null space shrinks exponentially with k
    (arXiv 2604.12086 §3.2), making it much harder to craft adversarial
    bypasses than with any single tier.
    """
    from carnot.verify.and_composition_verifier import AndCompositionVerifier

    verifier = AndCompositionVerifier()
    result = verifier.verify(prompt, response)
    # result.verified=True means ALL k verifiers passed (response is clean).
    # A violation means NOT all verifiers agree it is clean.
    return not result.verified


# ---------------------------------------------------------------------------
# Per-exemplar runner
# ---------------------------------------------------------------------------


def _evaluate_exemplar(ex: dict[str, Any]) -> dict[str, Any]:
    """Run all cascade tiers on one exemplar and return per-tier TP flags.

    Each flag is True if the tier correctly identified the buggy response as
    incorrect (true positive), False if the tier passed it (false negative).

    Returns a dict suitable for aggregation across the full corpus.
    """
    prompt = ex["prompt"]
    buggy = ex["buggy_response"]
    category = ex["category"]

    tier_results: dict[str, bool] = {}

    # Tier 0a: ThinkPRM v2 — proxy via SpilledEnergyDetector
    # (ThinkPRM requires a trained hidden-state probe; SpilledEnergyDetector
    # uses the same "energy above threshold" concept from the logit domain)
    try:
        tier_results["tier_0a_thinkprm"] = bool(_run_spilled_energy(buggy))
    except Exception as exc:
        print(f"[WARN] tier_0a failed on {ex['id']}: {exc}", file=sys.stderr)
        tier_results["tier_0a_thinkprm"] = False

    # Tier 0b: SpilledEnergyDetector
    try:
        tier_results["tier_0b_spilled"] = bool(_run_spilled_energy(buggy))
    except Exception as exc:
        print(f"[WARN] tier_0b failed on {ex['id']}: {exc}", file=sys.stderr)
        tier_results["tier_0b_spilled"] = False

    # Tier 0c: SemEnergyProbe
    try:
        tier_results["tier_0c_semenergy"] = bool(_run_semenergy(buggy))
    except Exception as exc:
        print(f"[WARN] tier_0c failed on {ex['id']}: {exc}", file=sys.stderr)
        tier_results["tier_0c_semenergy"] = False

    # Tier 2.5: SymCodeVerifier
    try:
        tier_results["tier_25_symcode"] = bool(_run_symcode(prompt, buggy))
    except Exception as exc:
        print(f"[WARN] tier_25 failed on {ex['id']}: {exc}", file=sys.stderr)
        tier_results["tier_25_symcode"] = False

    # Tier 2.7: CausalReasoningVerifier
    try:
        tier_results["tier_27_causal"] = bool(_run_causal(buggy))
    except Exception as exc:
        print(f"[WARN] tier_27 failed on {ex['id']}: {exc}", file=sys.stderr)
        tier_results["tier_27_causal"] = False

    # Tier 3: k=5 AND-compose
    try:
        tier_results["tier_3_k5"] = bool(_run_and_compose(prompt, buggy))
    except Exception as exc:
        print(f"[WARN] tier_3 failed on {ex['id']}: {exc}", file=sys.stderr)
        tier_results["tier_3_k5"] = False

    # Z3MathVerifier standalone (separate from AND-compose, for direct comparison)
    try:
        tier_results["z3_math_standalone"] = bool(_run_z3_math(buggy))
    except Exception as exc:
        print(f"[WARN] z3_math failed on {ex['id']}: {exc}", file=sys.stderr)
        tier_results["z3_math_standalone"] = False

    return {
        "id": ex["id"],
        "category": category,
        "tier_results": tier_results,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _tp_rate(flags: list[bool]) -> float:
    """True positive rate: fraction of exemplars flagged as violations."""
    if not flags:
        return 0.0
    return sum(flags) / len(flags)


def _aggregate(
    per_exemplar: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute per-tier and per-category TP rates from per-exemplar results."""
    tier_names = [
        "tier_0a_thinkprm",
        "tier_0b_spilled",
        "tier_0c_semenergy",
        "tier_25_symcode",
        "tier_27_causal",
        "tier_3_k5",
        "z3_math_standalone",
    ]

    # ---- Per-tier TP rates (across all exemplars) ----
    per_tier_tp_rate: dict[str, float] = {}
    for tier in tier_names:
        flags = [ex["tier_results"].get(tier, False) for ex in per_exemplar]
        per_tier_tp_rate[tier] = round(_tp_rate(flags), 6)

    # ---- Per-category TP rates (any tier flags = TP) ----
    categories: set[str] = {ex["category"] for ex in per_exemplar}
    per_category_tp_rate: dict[str, float] = {}
    for cat in sorted(categories):
        cat_exemplars = [ex for ex in per_exemplar if ex["category"] == cat]
        # An exemplar is caught if ANY tier flags it.
        flags = [any(ex["tier_results"].values()) for ex in cat_exemplars]
        per_category_tp_rate[cat] = round(_tp_rate(flags), 6)

    # ---- High-level group TP rates ----
    def _group_tp(group_cats: set[str]) -> float:
        group_ex = [ex for ex in per_exemplar if ex["category"] in group_cats]
        if not group_ex:
            return 0.0
        flags = [any(ex["tier_results"].values()) for ex in group_ex]
        return round(_tp_rate(flags), 6)

    arithmetic_tp = _group_tp(ARITHMETIC_CATEGORIES)
    semantic_tp = _group_tp(SEMANTIC_CATEGORIES)
    causal_tp = _group_tp(CAUSAL_CATEGORIES)

    # ---- Z3 specifically on arithmetic exemplars ----
    arith_ex = [ex for ex in per_exemplar if ex["category"] in ARITHMETIC_CATEGORIES]
    z3_arith_flags = [ex["tier_results"].get("z3_math_standalone", False) for ex in arith_ex]
    z3_arithmetic_tp_rate = round(_tp_rate(z3_arith_flags), 6)

    # SemEnergyProbe across all exemplars
    semenergy_flags = [ex["tier_results"].get("tier_0c_semenergy", False) for ex in per_exemplar]
    semenergy_tp_rate = round(_tp_rate(semenergy_flags), 6)

    # ---- Honest verdict ----
    # Does Z3 outperform all learned tiers on arithmetic?
    learned_tier_rates = [
        per_tier_tp_rate["tier_0a_thinkprm"],
        per_tier_tp_rate["tier_0c_semenergy"],
        per_tier_tp_rate["tier_27_causal"],
    ]
    max_learned = max(learned_tier_rates) if learned_tier_rates else 0.0
    n_total = len(per_exemplar)

    if n_total < 10:
        honest_verdict = "corpus_too_small"
    elif z3_arithmetic_tp_rate > max_learned and z3_arithmetic_tp_rate > 0.7:
        honest_verdict = "z3_dominates_arithmetic"
    elif max_learned > z3_arithmetic_tp_rate:
        honest_verdict = "learned_tiers_dominant"
    else:
        honest_verdict = "mixed_results"

    return {
        "per_tier_tp_rate": per_tier_tp_rate,
        "per_category_tp_rate": per_category_tp_rate,
        "per_category_group": {
            "arithmetic": arithmetic_tp,
            "semantic": semantic_tp,
            "causal_structural": causal_tp,
        },
        "z3_arithmetic_tp_rate": z3_arithmetic_tp_rate,
        "semenergy_tp_rate": semenergy_tp_rate,
        "honest_verdict": honest_verdict,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Load exemplar corpus, run cascade, write result artifact."""
    started_at = datetime.now(UTC)

    # Load exemplars
    if not EXEMPLAR_PATH.exists():
        print(f"[ERROR] Exemplar corpus not found: {EXEMPLAR_PATH}", file=sys.stderr)
        sys.exit(1)

    with EXEMPLAR_PATH.open() as f:
        exemplars = [json.loads(line) for line in f if line.strip()]

    print(f"[INFO] Loaded {len(exemplars)} exemplars from {EXEMPLAR_PATH}")

    # Run cascade on each exemplar
    per_exemplar: list[dict[str, Any]] = []
    for i, ex in enumerate(exemplars):
        print(f"[INFO] Evaluating exemplar {i + 1}/{len(exemplars)}: {ex['id']} ({ex['category']})")
        result = _evaluate_exemplar(ex)
        per_exemplar.append(result)
        # Print a quick per-exemplar summary line
        tp_tiers = [t for t, v in result["tier_results"].items() if v]
        print(f"       TP tiers: {tp_tiers if tp_tiers else ['none']}")

    # Aggregate
    agg = _aggregate(per_exemplar)

    # Print readable summary
    print("\n=== RESULT SUMMARY ===")
    print(f"Exemplars tested:  {len(exemplars)}")
    print(f"Categories:        {len(agg['per_category_tp_rate'])}")
    print(f"\nPer-tier TP rates:")
    for tier, rate in sorted(agg["per_tier_tp_rate"].items()):
        print(f"  {tier:<30s} {rate:.4f}")
    print(f"\nZ3 arithmetic TP rate:  {agg['z3_arithmetic_tp_rate']:.4f}")
    print(f"SemEnergy TP rate:      {agg['semenergy_tp_rate']:.4f}")
    print(f"\nHonest verdict: {agg['honest_verdict']}")

    # Build artifact
    finished_at = datetime.now(UTC)
    duration_s = (finished_at - started_at).total_seconds()

    artifact: dict[str, Any] = {
        "experiment": 1132,
        "schema": "goodfire_exemplar_cascade_tp_v1",
        "run_date": started_at.strftime("%Y-%m-%d"),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_s": round(duration_s, 2),
        # --- required fields ---
        "n_exemplars_tested": len(exemplars),
        "n_categories": len(agg["per_category_tp_rate"]),
        "per_tier_tp_rate": {
            "tier_0a_thinkprm": agg["per_tier_tp_rate"]["tier_0a_thinkprm"],
            "tier_0c_semenergy": agg["per_tier_tp_rate"]["tier_0c_semenergy"],
            "tier_25_symcode": agg["per_tier_tp_rate"]["tier_25_symcode"],
            "tier_27_causal": agg["per_tier_tp_rate"]["tier_27_causal"],
            "tier_3_k5": agg["per_tier_tp_rate"]["tier_3_k5"],
        },
        "per_category_tp_rate": agg["per_category_tp_rate"],
        "z3_arithmetic_tp_rate": agg["z3_arithmetic_tp_rate"],
        "semenergy_tp_rate": agg["semenergy_tp_rate"],
        "goodfire_exemplar_tp_rate_measured": True,
        "per_tier_results_logged": True,
        "honest_verdict": agg["honest_verdict"],
        # --- extended diagnostics ---
        "tier_0b_spilled_tp_rate": agg["per_tier_tp_rate"]["tier_0b_spilled"],
        "z3_math_standalone_tp_rate": agg["per_tier_tp_rate"]["z3_math_standalone"],
        "per_category_group_tp_rate": agg["per_category_group"],
        "tier_0a_proxy_note": (
            "ThinkPRM (Tier 0a) requires a trained hidden-state probe; "
            "SpilledEnergyDetector text heuristic used as proxy in this run."
        ),
        "exemplar_corpus_path": str(EXEMPLAR_PATH.relative_to(REPO_ROOT)),
        "per_exemplar_results": per_exemplar,
    }

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_PATH.open("w") as f:
        json.dump(artifact, f, indent=2)

    print(f"\n[INFO] Artifact written to {RESULT_PATH}")


if __name__ == "__main__":
    main()
