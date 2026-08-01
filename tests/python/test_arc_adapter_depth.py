"""Guard the measured ARC adapter-depth baseline.

Spec coverage: REQ-ARC-ADAPTER-DEPTH-001

Origin: 2026-07-31. `ops/arc_solve_registry.yaml` describes each game's solver in PROSE --
lf52's reads "L1-L2: GameAdapter _lf52 ... L3: <a different mechanism>". Nothing checked that
prose against the code, so an adapter could silently regress while the registry kept
asserting the old capability. `scripts/arc_adapter_depth_probe.py` replaces the prose claim
with a MEASURED number; this file guards that record.

WHY THIS TEST DOES NOT RE-SOLVE ANYTHING. A single `solve_adaptered` call was measured at
over 10 minutes -- it is a verifier-routed best-first search. A 24-game sweep is a
milestone-close job, not a pytest. So the expensive measurement lives in the script and this
test guards its output: every registered adapter is covered, and no recorded depth is ever
silently lowered.

That split is the point. A test that tried to do the solving would be either too slow to run
or quietly reduced to a subset -- and a coverage check that silently shrinks is the failure
mode this whole line of work keeps turning up.
"""

import json

import pytest

from carnot.paths import repo_root

BASELINE_PATH = repo_root() / "ops" / "arc_adapter_depth_baseline.json"


def _baseline() -> dict:
    if not BASELINE_PATH.exists():
        pytest.skip(
            "no adapter-depth baseline yet; generate with "
            "`python scripts/arc_adapter_depth_probe.py`"
        )
    return json.loads(BASELINE_PATH.read_text())


def _registered_adapters() -> set[str]:
    from carnot.agentic import arc_game_adapters as adapters

    return set(adapters._BUILDERS)


class TestBaselineCoversEveryAdapter:
    """REQ-ARC-ADAPTER-DEPTH-001: a new adapter cannot dodge the record by not being in it."""

    def test_every_registered_adapter_is_in_the_baseline(self) -> None:
        """The check that stops coverage silently shrinking.

        Adding an adapter without re-probing would leave it unmeasured, and an unmeasured
        adapter is exactly a prose claim again -- which is what this replaced.
        """
        recorded = set(_baseline()["games"])
        missing = _registered_adapters() - recorded
        assert not missing, (
            f"adapters with no measured depth: {sorted(missing)}. Re-run "
            f"`python scripts/arc_adapter_depth_probe.py --games {','.join(sorted(missing))}` "
            "and commit the updated baseline."
        )

    def test_baseline_has_no_stale_games(self) -> None:
        """A removed adapter must not linger as a depth claim nothing can satisfy."""
        recorded = set(_baseline()["games"])
        stale = recorded - _registered_adapters()
        assert not stale, f"baseline records games with no adapter: {sorted(stale)}"


class TestBaselineIsHonest:
    """The record must not overstate what was actually measured."""

    def test_declares_its_own_bounds(self) -> None:
        """measured_depth is a LOWER BOUND and the artifact has to say so.

        Without the caveat a reader treats 0 as "this adapter is broken" when it may mean
        "the probe timed out at the cap" -- the difference between a bug and an unproven
        claim, which is the distinction this whole exercise is about.
        """
        b = _baseline()
        assert "max_level_probed" in b and "timeout_s_per_attempt" in b
        assert "lower bound" in b.get("caveats", "").lower()

    def test_every_depth_is_supported_by_a_recorded_attempt(self) -> None:
        """A depth of N must have an attempt at N that actually reached N.

        Guards against a hand-edited baseline: the number and its evidence travel together
        or the claim is prose again.
        """
        for game, rec in _baseline()["games"].items():
            depth = rec["measured_depth"]
            if depth == 0:
                continue
            hit = [
                a
                for a in rec["attempts"]
                if a.get("level") == depth
                and a.get("status") == "ok"
                and a.get("reached", 0) >= depth
            ]
            assert hit, f"{game}: claims depth {depth} with no successful attempt at that level"


class TestNoSilentRegression:
    """The reason the file is committed rather than regenerated: it is a ratchet."""

    def test_depths_are_non_negative_ints(self) -> None:
        for game, rec in _baseline()["games"].items():
            d = rec["measured_depth"]
            assert isinstance(d, int) and d >= 0, f"{game}: bad measured_depth {d!r}"

    def test_regression_is_visible_in_the_diff(self) -> None:
        """Lowering a depth must be a deliberate, reviewable edit.

        This test cannot detect a regression on its own -- re-solving is what detects it, and
        that is the probe's job. What it CAN do is ensure the baseline is committed and
        structured so that a drop shows up as a git diff a reviewer sees, rather than being
        recomputed silently on every run. Stated explicitly so nobody mistakes this file for
        a live regression detector.
        """
        b = _baseline()
        assert b.get("schema") == "carnot.arc_adapter_depth_baseline.v1"
        assert b["games"], "empty baseline would make every other assertion vacuous"
