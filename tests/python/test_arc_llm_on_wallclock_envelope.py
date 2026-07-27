"""Tests for the MAX_ACTIONS wall-clock envelope analyser's load-bearing statistics.

WHY THESE SPECIFIC TESTS
========================
The analyser's conclusions rest on two small functions, and this project's own record says that is
exactly where measurement bugs live: a p-value reported without its reachable floor made a 2-cell
support look like evidence of absence, and a one-sided test made a REVERSAL read as no effect. So the
tests below are written against those incidents rather than against a happy path:

  - the p-FLOOR must be reported and must be correct, because "p=0.25, no effect" is a different
    statement from "p=0.25, and 0.25 is the smallest number this design could ever produce";
  - the test must be TWO-tailed, so a decrease is detected with the same sensitivity as an increase;
  - the bootstrap CI must widen as n shrinks and must degrade gracefully at n=0/n=1 rather than
    raising, because the envelope is built per budget and a crashed cell can empty a bucket.

Spec: REQ-ARC-WMTE-5984 (the action-budget wall-clock envelope is measured LLM-ON, reported per
candidate cap, and separated from both score value and generator reliability) --
SCENARIO-ARC-WMTE-5984-1 (concurrent per-slot context overflow breaks induction) and
SCENARIO-ARC-WMTE-5984-2 (a wrong mechanism is retracted in place, not deleted). These tests exercise
the statistics behind that requirement's clauses 2 and 9: the same-config-replicate comparison in
`adjacent_step_tests` and the power/uncertainty reporting in
`budget_curve.*.wall_s_per_game_mean_ci`.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _load():
    """Import the analyser by path.

    Loaded this way rather than via a package import because `scripts/` is not an installed package;
    a `sys.path` hack in the test would be equivalent but less explicit about what is being loaded.
    """
    spec = importlib.util.spec_from_file_location(
        "wallclock_envelope", REPO / "scripts" / "analyze_arc_llm_on_wallclock_envelope.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


M = _load()


# --------------------------------------------------------------------------------------------
# two_tailed_sign_test
# --------------------------------------------------------------------------------------------


def test_empty_support_is_marked_uninterpretable_not_silently_null():
    """No data must be distinguishable from a computed null.

    A caller that got `p_two_sided: None` without `interpretable: False` could not tell "no games
    moved" from "the test was never run" -- the uninstrumented-arm-reads-as-a-clean-result failure.
    """
    r = M.two_tailed_sign_test(0, 0)
    assert r["interpretable"] is False
    assert r["p_two_sided"] is None
    assert r["min_reachable_p"] is None


def test_p_floor_at_tiny_supports_matches_the_incident_numbers():
    """The documented floors: 1 cell -> 1.0, 2 -> 0.5, 5 -> 0.0625, 6 -> 0.03125.

    These are the exact values that make the difference between an underpowered design and a usable
    one. 5 games CANNOT clear 0.05 on a two-tailed sign test; 6 can. This session added a sixth game
    for precisely that reason, so the arithmetic behind that decision is pinned here.
    """
    assert M.two_tailed_sign_test(1, 0)["min_reachable_p"] == 1.0
    assert M.two_tailed_sign_test(2, 0)["min_reachable_p"] == 0.5
    assert M.two_tailed_sign_test(5, 0)["min_reachable_p"] == 0.0625
    assert M.two_tailed_sign_test(6, 0)["min_reachable_p"] == 0.03125
    assert M.two_tailed_sign_test(5, 0)["can_ever_reach_0_05"] is False
    assert M.two_tailed_sign_test(6, 0)["can_ever_reach_0_05"] is True


def test_unanimous_support_attains_its_own_floor():
    """When every pair agrees, p must EQUAL the floor -- not merely approach it.

    If a unanimous result came back above the floor the floor would be unreachable by construction,
    which would make `can_ever_reach_0_05` a lie.
    """
    for n in (2, 5, 6, 8):
        r = M.two_tailed_sign_test(n, 0)
        assert r["p_two_sided"] == r["min_reachable_p"], (n, r)


def test_is_two_tailed_a_reversal_is_as_detectable_as_an_increase():
    """The failure this guards: a one-sided test reports a REVERSAL as no effect.

    Mirrored inputs must give identical p and opposite directions.
    """
    up = M.two_tailed_sign_test(6, 0)
    down = M.two_tailed_sign_test(0, 6)
    assert up["p_two_sided"] == down["p_two_sided"]
    assert up["direction"] == "increase"
    assert down["direction"] == "decrease"


def test_a_tie_is_reported_as_a_tie_and_never_significant():
    r = M.two_tailed_sign_test(3, 3)
    assert r["direction"] == "tie"
    assert r["p_two_sided"] == 1.0


def test_p_is_a_probability_and_monotone_in_disagreement():
    """p must stay in [0,1] and must RISE as the split becomes more even.

    Monotonicity is the property that makes p usable as evidence at all; an implementation that
    double-counted the central term would violate it while still returning plausible-looking numbers.
    """
    ps = [M.two_tailed_sign_test(8 - k, k)["p_two_sided"] for k in range(0, 5)]
    assert all(0.0 <= p <= 1.0 for p in ps)
    assert ps == sorted(ps), ps


# --------------------------------------------------------------------------------------------
# boot_ci
# --------------------------------------------------------------------------------------------


def test_boot_ci_on_empty_bucket_returns_nulls_instead_of_raising():
    """A crashed cell can empty a budget bucket. The analyser must survive that.

    Raising here would abort the whole envelope because ONE cell's generator died -- turning a
    dropped row into a lost report.
    """
    r = M.boot_ci([])
    assert r["n"] == 0
    assert r["mean"] is None and r["lo"] is None and r["hi"] is None


def test_boot_ci_single_value_is_a_degenerate_point_interval():
    """With one game there is no sampling variation to find, and the CI must SAY so by collapsing.

    A spuriously wide interval here would invent uncertainty; a spuriously narrow one on real data
    would hide it. Collapsing is the honest answer at n=1 -- and it is why `scope_and_power` has to
    be read beside the interval.
    """
    r = M.boot_ci([147.02])
    assert r["mean"] == r["lo"] == r["hi"] == 147.02
    assert r["n"] == 1


def test_boot_ci_brackets_the_mean_and_is_reproducible():
    vals = [80.46, 119.03, 123.83, 228.43, 147.02]
    r1 = M.boot_ci(vals)
    r2 = M.boot_ci(vals)
    assert r1 == r2, "same seed must give the same interval or the artifact is not reproducible"
    assert r1["lo"] <= r1["mean"] <= r1["hi"]
    # Tolerance is 0.005, not 1e-6: `boot_ci` rounds every reported figure to 2 decimal places
    # (139.754 -> 139.75). Written as half a cent-of-a-second rather than "close enough" so that a
    # future change to the rounding precision fails this test loudly instead of drifting past it.
    assert abs(r1["mean"] - sum(vals) / len(vals)) <= 0.005


def test_boot_ci_widens_as_the_sample_shrinks():
    """Fewer games must mean a wider interval. This is the property that stops a 2-game measurement
    from being quoted with the confidence of a 20-game one."""
    wide = M.boot_ci([50.0, 400.0])
    narrow = M.boot_ci([50.0, 400.0] * 10)
    assert (wide["hi"] - wide["lo"]) > (narrow["hi"] - narrow["lo"])


def test_boot_ci_uses_the_mean_not_the_median_on_a_skewed_sample():
    """The envelope multiplies a per-game cost by ~110 games, so the MEAN is the right functional.

    A median would understate a heavy right tail -- and the LLM-on corpus is visibly heavy-tailed
    (45s..606s), so choosing the median would systematically overstate the affordable budget.
    """
    skewed = [10.0, 10.0, 10.0, 10.0, 1000.0]
    r = M.boot_ci(skewed)
    assert r["mean"] == 208.0  # the mean; the median would be 10.0
