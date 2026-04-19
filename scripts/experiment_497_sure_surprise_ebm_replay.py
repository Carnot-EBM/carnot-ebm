#!/usr/bin/env python3
"""Exp 497: SuRe Surprise EBM Replay — compare SuRe vs uniform replay for PPSConstraintLearner.

Compares partition_isolation_score achieved by SuRePriorityReplay (arXiv 2511.22367)
vs uniform random replay on interleaved real CoT steps from fover_labeled_steps_live.json.

Spec: REQ-SELFLEARN-021, REQ-SELFLEARN-022, FR-11 Tier 2
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

# Repo root on sys.path for scripts/ and carnot.pipeline imports
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "python"))

from carnot.pipeline.env_autofix import apply_env_autofix
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from carnot.pipeline.deliverable_guard import DeliverableGuard
from carnot.pipeline.pps_constraint_learner import (
    ConstraintDomain,
    PPSConstraintLearner,
    PartitionIsolationScore,
)
from carnot.pipeline.ppsebm_real_validator import InterleavedViolationSequence
from carnot.pipeline.lsebm_replayer import LSEBMConstraintReplayer
from carnot.pipeline.sure_priority_replay import SuRePriorityReplay, SuReReplayResult
from scripts.experiment_template import ExperimentTemplate


# ---------------------------------------------------------------------------
# Domain assignment for fover_labeled_steps_live.json
# ---------------------------------------------------------------------------

# The live steps are arithmetic CoT steps — all steps from fover_labeled_steps_live.json
# are labeled with 'label' (correct/incorrect) and contain arithmetic reasoning.
# We assign domain based on content heuristics: presence of code keywords → 'code',
# presence of logical connectives → 'logical', else 'arithmetic'.

def assign_domain(step: dict) -> str:
    """Heuristically assign a constraint domain to a CoT step.

    WHY heuristics (not a trained classifier): this is a CPU-only experiment.
    We need domain labels to construct InterleavedViolationSequence.  The fover steps
    are real CoT data without explicit domain annotations.  Keyword heuristics give
    domain diversity without requiring inference, matching the pattern from Exp 485.

    Returns:
        'code', 'logical', or 'arithmetic'
    """
    text = step.get("step_text", "").lower()
    code_keywords = ["def ", "class ", "import ", "print(", "function", "algorithm",
                     "variable", "loop", "array", "list", "dict", "string", "integer"]
    logical_keywords = ["if and only if", "therefore", "implies", "contrapositive",
                        "contradiction", "premise", "conclusion", "infer", "hence",
                        "thus we can conclude", "logical", "proposition"]

    if any(kw in text for kw in code_keywords):
        return "code"
    if any(kw in text for kw in logical_keywords):
        return "logical"
    return "arithmetic"


def make_violation_string(step: dict) -> str:
    """Convert a labeled step to a violation type string for PPSConstraintLearner.

    WHY a violation type string (not the raw step text): PPSConstraintLearner.fit_domain()
    accepts a list of violation type strings.  These strings are used to compute a
    vocabulary-based gradient.  We encode the violation as label + question_id to give
    each violation a unique but reproducible type string.

    Returns:
        String like 'incorrect_156' or 'correct_159'
    """
    label = step.get("label", "unknown")
    qid = step.get("question_id", "0")
    return f"{label}_{qid}"


# ---------------------------------------------------------------------------
# Run PPSConstraintLearner with a given replay strategy
# ---------------------------------------------------------------------------

def run_with_replay_strategy(
    steps: list[dict],
    replay_strategy: str | SuRePriorityReplay,
    rng_seed: int = 42,
) -> float:
    """Run PPSConstraintLearner on interleaved steps; return partition_isolation_score.

    Args:
        steps: List of step dicts with 'domain' key added.
        replay_strategy: 'uniform' for random replay, or a SuRePriorityReplay instance.
        rng_seed: RNG seed for uniform replay reproducibility.

    Returns:
        partition_isolation_score (cosine distance between domain gradient directions).
    """
    domains = [ConstraintDomain.ARITHMETIC, ConstraintDomain.CODE, ConstraintDomain.LOGICAL]
    base_replayer = LSEBMConstraintReplayer(n_replay=5, ebm_n_iter=10)
    learner = PPSConstraintLearner(domains=domains, replayer=base_replayer)

    # Group steps by domain for fit_domain calls.
    domain_violations: dict[str, list[str]] = {
        "arithmetic": [],
        "code": [],
        "logical": [],
    }
    for step in steps:
        d = step["domain"]
        v_str = make_violation_string(step)
        domain_violations[d].append(v_str)

    # Fit each domain with its violations.
    learner.fit_domain(ConstraintDomain.ARITHMETIC, domain_violations["arithmetic"])
    learner.fit_domain(ConstraintDomain.CODE, domain_violations["code"])
    learner.fit_domain(ConstraintDomain.LOGICAL, domain_violations["logical"])

    # Compute isolation score on the partitions after training.
    iso = PartitionIsolationScore(learner.partitions)
    base_score = iso.score()

    # Replay phase: select violations for replay and run an additional fit_domain pass.
    all_violations = []
    for d_str, viols in domain_violations.items():
        for v in viols:
            all_violations.append((d_str, v))

    n_replay = max(1, int(len(all_violations) * 0.3))

    if replay_strategy == "uniform":
        rng = random.Random(rng_seed)
        replay_items = rng.sample(all_violations, min(n_replay, len(all_violations)))
    else:
        # SuRePriorityReplay: add all violations with simulated energy, then get top-n.
        sure: SuRePriorityReplay = replay_strategy
        for i, (d_str, v_str) in enumerate(all_violations):
            # Simulate EBM energy as a hash-based pseudo-random value for reproducibility.
            # WHY: CPU-only experiment — we do not have a live Ising EBM model.
            # We simulate energy as a deterministic function of the violation string,
            # which gives each violation a stable "EBM energy" for the surprise computation.
            energy = (hash(v_str) % 1000) / 1000.0  # deterministic in [0, 1)
            sure.add(violation={"v_str": v_str, "domain": d_str}, domain=d_str, energy=energy)

        replay_dicts = sure.get_replay_batch(n=n_replay)
        replay_items = [(r["domain"], r["v_str"]) for r in replay_dicts]

    # Group replay items by domain and run a second fit_domain pass.
    replay_by_domain: dict[str, list[str]] = {"arithmetic": [], "code": [], "logical": []}
    for d_str, v_str in replay_items:
        replay_by_domain[d_str].append(v_str)

    domain_map = {
        "arithmetic": ConstraintDomain.ARITHMETIC,
        "code": ConstraintDomain.CODE,
        "logical": ConstraintDomain.LOGICAL,
    }
    for d_str, viols in replay_by_domain.items():
        if viols:
            learner.fit_domain(domain_map[d_str], viols)

    iso_after = PartitionIsolationScore(learner.partitions)
    return iso_after.score()


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main() -> None:
    apply_env_autofix()

    deliverable = "results/experiment_497_sure_surprise_ebm_replay.json"
    guard = DeliverableGuard(str(_REPO_ROOT / deliverable))

    tmpl = ExperimentTemplate(
        497,
        "SuRe Surprise EBM Replay",
        deliverable,
        requires_gpu=False,
    )
    tmpl.setup()

    with ExperimentTimeoutWatchdog(497, timeout_minutes=40):
        # Load real CoT steps
        steps_path = _REPO_ROOT / "results" / "fover_labeled_steps_live.json"
        with open(steps_path) as f:
            raw_steps = json.load(f)

        # Assign domain labels
        for step in raw_steps:
            step["domain"] = assign_domain(step)

        # Construct interleaved sequence to verify data structure
        seq = InterleavedViolationSequence(raw_steps)
        n_steps = len(seq.steps)
        interleaving_rate = seq.interleaving_rate

        # Run uniform replay baseline
        isolation_score_uniform = run_with_replay_strategy(
            list(raw_steps),
            replay_strategy="uniform",
            rng_seed=42,
        )

        # Run SuRe priority replay
        sure_replay = SuRePriorityReplay(
            replay_buffer_size=200,
            top_k_fraction=0.3,
            surprise_threshold=0.0,  # include all violations in ranking
        )
        isolation_score_sure = run_with_replay_strategy(
            list(raw_steps),
            replay_strategy=sure_replay,
            rng_seed=42,
        )

        n_replay_items = max(1, int(n_steps * 0.3))
        result = SuReReplayResult(
            n_violations_processed=n_steps,
            n_replay_items=n_replay_items,
            isolation_score_uniform=isolation_score_uniform,
            isolation_score_sure=isolation_score_sure,
        )

        fr11_tier2_status = "improved" if result.sure_better else "no_improvement"
        honest_verdict = (
            "sure_improves_isolation" if result.sure_better else "no_improvement"
        )

        artifact = tmpl.build_result(
            {
                "schema": "carnot.sure_replay.v1",
                "n_violations_processed": result.n_violations_processed,
                "n_replay_items": result.n_replay_items,
                "isolation_score_uniform": round(isolation_score_uniform, 6),
                "isolation_score_sure": round(isolation_score_sure, 6),
                "isolation_improvement": result.isolation_improvement,
                "sure_better": result.sure_better,
                "interleaving_rate": round(interleaving_rate, 4),
                "fr11_tier2_status": fr11_tier2_status,
                "honest_verdict": honest_verdict,
            },
            status="success",
        )

        out_path = _REPO_ROOT / deliverable
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(artifact, f, indent=2)

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
