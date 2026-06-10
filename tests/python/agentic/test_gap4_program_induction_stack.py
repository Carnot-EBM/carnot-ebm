"""Unit tests for the GAP-4 program-induction stack tiered selector.

These tests verify:
 - The tiered policy logic (T1/T2/T3) using in-memory fixtures.
 - The offline replay helpers reproduce 19/31 (ARC-2) and 28/31 (ARC-1)
   bit-exact from the saved artifacts on disk — without any new model calls.

REQ reference: GAP-4 harness_recommendation (results/arc3_gap4_chain_arms_adversarial_verify.json)
"""

import json
from pathlib import Path

import pytest

from carnot.agentic.gap4_program_induction_stack import (
    agreement_label,
    promote_first_fresh_demo_perfect,
    replay_arc1_demo_perfect_coverage_from_saved,
    replay_arc2_pass_at_1_from_saved,
    snap_select,
    tiered_select,
)


# ---------------------------------------------------------------------------
# Fixtures: minimal synthetic arm / pool data
# ---------------------------------------------------------------------------

PROBE_ARM = {"source": "probe_chain", "demo_perfect": True, "code": "def transform(g): return g"}
FRESH_ARM_PERFECT = {"source": "fresh_chain1", "demo_perfect": True, "code": "def transform(g): return g"}
FRESH_ARM_IMPERFECT = {"source": "fresh_chain2", "demo_perfect": False, "code": None}
FRESH_ARM_PERFECT_2 = {"source": "fresh_chain3", "demo_perfect": True, "code": "def transform(g): return g"}


# ---------------------------------------------------------------------------
# snap_select (T1)
# ---------------------------------------------------------------------------

def test_snap_select_returns_none_on_empty():
    assert snap_select([]) is None


def test_snap_select_returns_none_no_eligible():
    entries = [{"demo_fit": 0.90}, {"demo_fit": 0.85}]
    assert snap_select(entries, tau=0.005) is None


def test_snap_select_returns_best_eligible():
    entries = [{"demo_fit": 0.999, "id": "a"}, {"demo_fit": 1.0, "id": "b"}]
    result = snap_select(entries, tau=0.005)
    assert result is not None
    assert result["id"] == "b"


def test_snap_select_threshold_at_tau():
    # demo_fit == 1 - tau is exactly eligible
    entries = [{"demo_fit": 0.995, "id": "ok"}]
    assert snap_select(entries, tau=0.005) is not None

    entries_below = [{"demo_fit": 0.994, "id": "nope"}]
    assert snap_select(entries_below, tau=0.005) is None


# ---------------------------------------------------------------------------
# promote_first_fresh_demo_perfect (T2)
# ---------------------------------------------------------------------------

def test_promote_first_returns_none_on_empty():
    assert promote_first_fresh_demo_perfect([]) is None


def test_promote_first_skips_probe_arm():
    # Only probe arm — should return None
    assert promote_first_fresh_demo_perfect([PROBE_ARM]) is None


def test_promote_first_skips_imperfect_fresh():
    assert promote_first_fresh_demo_perfect([FRESH_ARM_IMPERFECT]) is None


def test_promote_first_returns_first_fresh_perfect():
    arms = [PROBE_ARM, FRESH_ARM_PERFECT, FRESH_ARM_PERFECT_2]
    result = promote_first_fresh_demo_perfect(arms)
    assert result is FRESH_ARM_PERFECT


def test_promote_first_probe_and_imperfect_then_perfect():
    arms = [PROBE_ARM, FRESH_ARM_IMPERFECT, FRESH_ARM_PERFECT]
    result = promote_first_fresh_demo_perfect(arms)
    assert result is FRESH_ARM_PERFECT


# ---------------------------------------------------------------------------
# agreement_label
# ---------------------------------------------------------------------------

def test_agreement_label_false_with_no_hash_fn():
    arms = [FRESH_ARM_PERFECT, FRESH_ARM_PERFECT_2]
    assert agreement_label(arms) is False


def test_agreement_label_false_single_fresh_perfect():
    called = []

    def hfn(g):
        called.append(1)
        return "hash_x"

    arm1 = {"source": "fresh_chain1", "demo_perfect": True, "pred_grid": [[1]]}
    # Only one fresh arm — no quorum possible
    assert agreement_label([arm1], pred_hash_fn=hfn) is False


def test_agreement_label_true_when_two_agree():
    def hfn(g):
        return "same_hash"  # all arms agree

    arm1 = {"source": "fresh_chain1", "demo_perfect": True, "pred_grid": [[1]]}
    arm2 = {"source": "fresh_chain2", "demo_perfect": True, "pred_grid": [[1]]}
    assert agreement_label([arm1, arm2], pred_hash_fn=hfn) is True


def test_agreement_label_false_when_disagree():
    counter = {"n": 0}

    def hfn(g):
        counter["n"] += 1
        return f"hash_{counter['n']}"  # each call returns a different hash

    arm1 = {"source": "fresh_chain1", "demo_perfect": True, "pred_grid": [[1]]}
    arm2 = {"source": "fresh_chain2", "demo_perfect": True, "pred_grid": [[2]]}
    assert agreement_label([arm1, arm2], pred_hash_fn=hfn) is False


def test_agreement_label_ignores_probe_arms():
    # Probe arm should not count toward the quorum even if demo_perfect
    def hfn(g):
        return "same"

    probe = {"source": "probe_chain", "demo_perfect": True, "pred_grid": [[1]]}
    fresh = {"source": "fresh_chain1", "demo_perfect": True, "pred_grid": [[1]]}
    # Only 1 fresh arm — quorum not met
    assert agreement_label([probe, fresh], pred_hash_fn=hfn) is False


# ---------------------------------------------------------------------------
# tiered_select (integration)
# ---------------------------------------------------------------------------

def test_tiered_select_t1_snap():
    pool = [{"demo_fit": 1.0, "id": "winner"}]
    result = tiered_select("task_x", arms=[], pool_entries=pool)
    assert result["tier"] == 1
    assert result["selected_arm"]["id"] == "winner"
    assert result["agreement"] is False


def test_tiered_select_t2_fresh_perfect():
    # No snap-eligible pool, but has fresh perfect arm
    result = tiered_select(
        "task_y",
        arms=[PROBE_ARM, FRESH_ARM_PERFECT],
        pool_entries=[{"demo_fit": 0.5}],
    )
    assert result["tier"] == 2
    assert result["selected_arm"] is FRESH_ARM_PERFECT


def test_tiered_select_t3_fallback():
    # No snap, no fresh perfect
    result = tiered_select(
        "task_z",
        arms=[PROBE_ARM, FRESH_ARM_IMPERFECT],
        pool_entries=[{"demo_fit": 0.5}],
    )
    assert result["tier"] == 3
    assert result["selected_arm"] is None


def test_tiered_select_t1_takes_priority_over_t2():
    # Pool has snap-eligible candidate AND there's a fresh-perfect arm.
    # T1 should win because it's tried first.
    pool = [{"demo_fit": 1.0, "id": "snap_winner"}]
    result = tiered_select(
        "task_both",
        arms=[FRESH_ARM_PERFECT],
        pool_entries=pool,
    )
    assert result["tier"] == 1


# ---------------------------------------------------------------------------
# Offline replay — bit-exact 19/31 and 28/31 from saved artifacts
# ---------------------------------------------------------------------------

CARNOT_ROOT = Path(__file__).parents[3]
ARC2_CHAIN_ENSEMBLE_PATH = CARNOT_ROOT / "results" / "arc3_gap4_arc2_chain_ensemble.json"
ARC1_INDUCED_PROGRAMS_PATH = CARNOT_ROOT / "results" / "arc3_gap4_induced_programs.json"
ARC2_POOL_SIZE = 31
ARC1_POOL_SIZE = 31


@pytest.mark.skipif(
    not ARC2_CHAIN_ENSEMBLE_PATH.exists(),
    reason="arc3_gap4_arc2_chain_ensemble.json not present (offline replay requires saved artifacts)",
)
def test_offline_replay_arc2_19of31():
    """SCENARIO: offline replay reproduces ARC-2 fresh-arm gold count 19/31 bit-exact.

    Loads the saved chain_ensemble artifact and verifies per_arm_gold_given_perfect.fresh.gold == 19.
    No new model calls; pure read from disk.  A mismatch would invalidate the deployment number.
    """
    artifact = json.loads(ARC2_CHAIN_ENSEMBLE_PATH.read_text())
    gold_count, pool_size = replay_arc2_pass_at_1_from_saved(artifact, pool_size=ARC2_POOL_SIZE)
    assert gold_count == 19, f"Expected ARC-2 fresh-arm gold=19, got {gold_count}"
    assert pool_size == ARC2_POOL_SIZE


@pytest.mark.skipif(
    not ARC1_INDUCED_PROGRAMS_PATH.exists(),
    reason="arc3_gap4_induced_programs.json not present",
)
def test_offline_replay_arc1_28of31():
    """SCENARIO: offline replay reproduces ARC-1 demo-perfect coverage 28/31 bit-exact.

    Counts unique tasks with demo_perfect programs from the saved induced_programs artifact.
    No new model calls.  This is the 'ARC-1 sanity' number (0.9032) in the harness recommendation.
    """
    artifact = json.loads(ARC1_INDUCED_PROGRAMS_PATH.read_text())
    covered, pool_size = replay_arc1_demo_perfect_coverage_from_saved(
        artifact, pool_size=ARC1_POOL_SIZE
    )
    assert covered == 28, f"Expected ARC-1 demo-perfect coverage=28 tasks, got {covered}"
    assert pool_size == ARC1_POOL_SIZE
