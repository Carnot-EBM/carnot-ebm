"""GAP-4 Program-Induction Stack — reusable tiered selector for ARC program-induction harnesses.

WHAT THIS IMPLEMENTS (from the synthesis harness_recommendation in
results/arc3_gap4_chain_arms_adversarial_verify.json):

  T1 — graded pool-snap τ≤0.005:
        If the pool contains a candidate with demo_fit >= 1 - tau (essentially perfect),
        select the top-scoring candidate by graded score (no generator needed).
        Measured zero pass@2 losses on both ARC-1 and ARC-2 venues.

  T2 — promote-first-FRESH-demo-perfect chain raw output:
        Among the saved programs for this task, pick the first FRESH arm (source is NOT
        'probe_chain') that is demo_perfect.  This is the workhorse tier.
        Measured ARC-2 pass@1 19/31=0.6129 (vs banked vote 1/31); ARC-1 sanity 28/31.

  T3 — TRM vote fallback:
        When neither T1 nor T2 applies, return None (caller falls back to vote or pool).

AGREEMENT as a CONFIDENCE LABEL ONLY — fresh arms only:
  If >=2 FRESH demo-perfect programs produce the same output, tag the selection with
  agreement=True.  Agreement is NEVER used as a promotion gate (agreement-first measured
  net -1 pass@1 in the ARC-2 chain-arms round); it is a confidence label for downstream
  routing (e.g. to skip the slower T3 vote).  Stale/probe arms MUST NOT participate in
  the quorum (the 9aaea919 lesson: including probe arms inflated "agreement" from 0.31
  gold to 0.73 gold, poisoning the gate).

CAVEATS (must not be pruned — see registry entry):
  - Rerank numbers ride on an oracle-degenerate 4/31 pool for ARC-2.
  - Agreement precision n≤10 and not significant vs 0.52 or 0.73.
  - Chain coverage measured only on the 12 probe-chain-feasible tasks.
  - ARC-1 numbers carry a probable-pretraining-contamination caveat.
  - Closed-weight gpt-5.5 inducer at this tier; the local-GGUF arm is exp4002.
  - Task-level unanimity-with-abstention (post-hoc 5/5 gold) stays OUT of the harness
    until its pre-registered fresh confirmation lands.
"""

from __future__ import annotations

from typing import Any


def _is_fresh_arm(arm: dict) -> bool:
    """Return True for arms that are NOT probe_chain (i.e. fresh/independent arms)."""
    return arm.get("source", "") != "probe_chain"


def snap_select(pool_entries: list[dict], tau: float = 0.005) -> dict | None:
    """T1: graded pool-snap selector.

    Scans the pool for a candidate with demo_fit >= 1 - tau (essentially perfect on demos).
    Returns the highest-graded candidate meeting the threshold, or None.

    WHY τ=0.005: validated on ARC-1 with zero pass@2 losses — this threshold is tight enough
    to avoid selecting near-misses while forgiving floating-point rounding in demo_fit scores.
    Rerank-safe: the snap operates on demo_fit alone (no gold, no oracle).

    Args:
        pool_entries: list of pool-entry dicts with at least a 'demo_fit' float field.
        tau: snap threshold; entries with demo_fit >= 1 - tau are snap-eligible.

    Returns:
        The best (highest demo_fit) snap-eligible entry, or None if none exist.
    """
    threshold = 1.0 - tau
    eligible = [e for e in pool_entries if e.get("demo_fit", 0.0) >= threshold]
    if not eligible:
        return None
    return max(eligible, key=lambda e: e.get("demo_fit", 0.0))


def promote_first_fresh_demo_perfect(arms: list[dict]) -> dict | None:
    """T2: promote-first-FRESH-demo-perfect selector (the workhorse).

    Among the saved programs for a task, returns the first FRESH arm (source != 'probe_chain')
    that is demo_perfect.  If no fresh demo-perfect arm exists, returns None.

    WHY fresh-only: the probe arm is a chained, potentially stale program with gold rate 0.31 vs
    fresh rate 0.73 (the 9aaea919 lesson).  Including it in T2 promotes weaker programs.

    Args:
        arms: list of arm dicts with 'source', 'demo_perfect', and 'code' fields.

    Returns:
        The first fresh demo-perfect arm dict, or None.
    """
    for arm in arms:
        if _is_fresh_arm(arm) and arm.get("demo_perfect", False):
            return arm
    return None


def agreement_label(arms: list[dict], pred_hash_fn: Any = None) -> bool:
    """Compute the agreement CONFIDENCE LABEL (fresh arms only, never a gate).

    Returns True if >=2 FRESH demo-perfect arms produce hash-identical outputs.

    CRITICAL: this function returns a label, not a gate.  The caller must NOT use
    agreement=True to bypass T2/T3 selection or promote a different candidate; it is
    informational only.  See harness_recommendation: 'agreement is a CONFIDENCE LABEL ONLY
    (fresh-fresh precision 0.857 n=7 / pooled 0.80 n=10, ns vs base rates) — NEVER as a
    promotion gate.'

    Args:
        arms: list of arm dicts.
        pred_hash_fn: optional callable(pred_grid) -> hashable.  If None, skips hash check.

    Returns:
        True if >=2 fresh demo-perfect arms agree (same pred_hash), False otherwise.
    """
    if pred_hash_fn is None:
        return False
    fresh_perfect = [a for a in arms if _is_fresh_arm(a) and a.get("demo_perfect", False)]
    if len(fresh_perfect) < 2:
        return False
    hashes = []
    for arm in fresh_perfect:
        pred = arm.get("pred_grid") or arm.get("preds")
        if pred is None:
            continue
        try:
            hashes.append(pred_hash_fn(pred))
        except Exception:  # noqa: BLE001
            continue
    if len(hashes) < 2:
        return False
    for h in set(hashes):
        if hashes.count(h) >= 2:
            return True
    return False


def tiered_select(
    task_id: str,
    arms: list[dict],
    pool_entries: list[dict] | None = None,
    tau: float = 0.005,
    pred_hash_fn: Any = None,
) -> dict:
    """Apply the three-tier GAP-4 selection policy to one task.

    Returns a selection-result dict with:
      tier: int (1, 2, or 3=fallback)
      selected_arm: the arm dict chosen (T1: pool entry, T2: arm dict), or None (T3)
      agreement: bool confidence label (fresh-only, never a gate)
      task_id: str

    WHY three tiers: T1 handles tasks where the pool already contains a near-perfect
    candidate (cheap, fast); T2 handles tasks where fresh induction produced a demo-perfect
    program (the main coverage path); T3 signals that the caller should fall back to vote
    or another mechanism.  The tiered design is necessary because ARC tasks vary widely in
    how much demo information constrains the rule.
    """
    # T1: graded pool snap
    if pool_entries:
        snap = snap_select(pool_entries, tau=tau)
        if snap is not None:
            return {
                "task_id": task_id,
                "tier": 1,
                "selected_arm": snap,
                "agreement": False,
            }

    # T2: promote-first-fresh-demo-perfect
    winner = promote_first_fresh_demo_perfect(arms)
    if winner is not None:
        conf = agreement_label(arms, pred_hash_fn=pred_hash_fn)
        return {
            "task_id": task_id,
            "tier": 2,
            "selected_arm": winner,
            "agreement": conf,
        }

    # T3: no selection — caller falls back to vote
    return {
        "task_id": task_id,
        "tier": 3,
        "selected_arm": None,
        "agreement": False,
    }


# ---------------------------------------------------------------------------
# Offline-replay helpers (bit-exact re-derivation from saved artifacts)
# ---------------------------------------------------------------------------

def replay_arc2_pass_at_1_from_saved(
    chain_ensemble_artifact: dict,
    pool_size: int = 31,
) -> tuple[int, int]:
    """Re-derive the ARC-2 19/31 number from the saved chain_ensemble artifact.

    The stored artifact records per_arm_gold_given_perfect.fresh.gold = 19 (number of
    fresh demo-perfect arms whose executed output matched gold) and n = 26 (total fresh
    demo-perfect arms).  The pass@1 figure 19/31 = gold / pool_size is an arm-level count
    of gold-correct fresh programs relative to the total pool size.

    This is a READ-ONLY re-derivation — zero new model calls.

    Args:
        chain_ensemble_artifact: loaded dict from arc3_gap4_arc2_chain_ensemble.json.
        pool_size: ARC-2 pool entry count (31).

    Returns:
        (gold_count, pool_size) where gold_count/pool_size == 19/31.
    """
    fresh = chain_ensemble_artifact["per_arm_gold_given_perfect"]["fresh"]
    return int(fresh["gold"]), pool_size


def replay_arc1_demo_perfect_coverage_from_saved(
    induced_programs_artifact: dict,
    pool_size: int = 31,
) -> tuple[int, int]:
    """Re-derive the ARC-1 28/31 number from the saved induced_programs artifact.

    Counts unique tasks that have at least one demo_perfect program.  This is the
    'promote-first-FRESH-demo-perfect' coverage metric for ARC-1 (28 of 31 pool tasks
    have a demo-perfect inducer program), used as an ARC-1 sanity check in the harness
    recommendation.

    Args:
        induced_programs_artifact: loaded dict from arc3_gap4_induced_programs.json.
        pool_size: ARC-1 pool entry count (31).

    Returns:
        (covered_task_count, pool_size) where covered_task_count/pool_size == 28/31.
    """
    programs = induced_programs_artifact.get("programs", [])
    tasks_with_dp = {p["task"] for p in programs if p.get("demo_perfect", False)}
    return len(tasks_with_dp), pool_size
