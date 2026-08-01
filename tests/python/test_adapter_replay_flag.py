"""`GameAdapter.replay` — does an adapter replay a stored plan, or search?

Spec coverage: REQ-ARC-WMTE-5717

Origin: 2026-07-31, from an operator asking three times what an L1-only GameAdapter is for.
Measured: 18 of 25 REPLAY. Their `action_labels` return the next label of a banked plan (or,
for s5i5, one fixed click), so `solve_adaptered`'s verifier-routed search, hazard pruning,
learned-verifier warm start and state dedup never engage -- the search is a straight line
through a solution that was already known.

The flag exists because the distinction was invisible without reading each adapter's source,
and because it makes downstream numbers honest: a replay adapter "reaching L1" in
ops/arc_adapter_depth_baseline.json measures the STORED PLAN, not adapter capability.

WHY THE CLASSIFICATION IS PINNED RATHER THAN DERIVED. Branching factor cannot separate the
classes: tn36 also returns a single label at L0, but COMPUTES it from live state, so it
survives a layout change where a stored plan would not. There is no cheap runtime predicate
for "derived from state vs read from a list", so the flag is declared and this file pins it.
`test_replay_adapters_run_out_at_l0` supplies the one empirical check that IS available.
"""

import pytest

from carnot.paths import repo_root

pytestmark = pytest.mark.skipif(
    not (repo_root() / "environment_files").exists(),
    reason="offline ARC environment_files not present",
)

# Measured 2026-07-31 by exhausting each adapter's L0 plan and reading what remains.
EXPECTED_REPLAY = {
    "ar25",
    "bp35",
    "cn04",
    "ft09",
    "g50t",
    "ka59",
    "lf52",
    "ls20",
    "r11l",
    "re86",
    "s5i5",
    "sb26",
    "sc25",
    "sk48",
    "sp80",
    "su15",
    "vc33",
    "wa30",
}
EXPECTED_SEARCH = {"cd82", "dc22", "lp85", "m0r0", "tn36", "tr87", "tu93"}

# s5i5 is a replay adapter that does NOT match the others' shape: it returns one fixed
# ACTION6 click at (48,21) regardless of path, rather than indexing a plan. Recorded as an
# exception instead of being forced into the pattern, so the empirical check below stays
# truthful about what it does and does not cover.
RUNS_OUT_EXEMPT = {"s5i5"}


@pytest.fixture(scope="module")
def adapters():
    from carnot.agentic import arc_game_adapters

    return arc_game_adapters


class TestClassificationIsPinned:
    """Silent drift in either direction is the thing being prevented."""

    def test_replay_set_matches(self, adapters) -> None:
        actual = {g for g in adapters._BUILDERS if adapters.get_adapter(g).replay}
        assert actual == EXPECTED_REPLAY, (
            f"replay set changed: +{sorted(actual - EXPECTED_REPLAY)} "
            f"-{sorted(EXPECTED_REPLAY - actual)}. If an adapter was genuinely upgraded from "
            "replay to search, update this test deliberately and re-run the depth baseline -- "
            "its numbers mean something different for the two classes."
        )

    def test_search_set_matches(self, adapters) -> None:
        actual = {g for g in adapters._BUILDERS if not adapters.get_adapter(g).replay}
        assert actual == EXPECTED_SEARCH

    def test_every_adapter_is_classified(self, adapters) -> None:
        """A new adapter must land in one bucket or the other, not silently default."""
        assert set(adapters._BUILDERS) == EXPECTED_REPLAY | EXPECTED_SEARCH


class TestReplayClaimsAreTraceable:
    """A replay adapter must name where its plan came from."""

    def test_replay_adapters_declare_a_source(self, adapters) -> None:
        missing = [
            g
            for g in sorted(EXPECTED_REPLAY)
            if not (adapters.get_adapter(g).replay_source or "").strip()
        ]
        assert not missing, (
            f"replay adapters with no replay_source: {missing}. The source is what makes the "
            "claim traceable to an artifact or constant instead of asserted."
        )

    def test_search_adapters_do_not_claim_a_source(self, adapters) -> None:
        """A search adapter naming a plan source would mean the flag is wrong."""
        spurious = [g for g in sorted(EXPECTED_SEARCH) if adapters.get_adapter(g).replay_source]
        assert not spurious, f"search adapters declaring a replay_source: {spurious}"


class TestEmpiricalCheck:
    """The one property that IS mechanically checkable."""

    def test_replay_adapters_run_out_at_l0(self, adapters) -> None:
        """Exhaust the plan and a replay adapter has nothing left to offer.

        This is the measurement the flag came from: with a path longer than any banked plan,
        17 of 18 return ZERO labels at level 0. That is the finding -- removing the plan does
        not reveal a search space underneath, because the plan IS the L0 behaviour. It is why
        "upgrade the replay adapters to genuine ones" is per-game reverse engineering rather
        than un-bypassing something already written.

        s5i5 is exempt and says so above.
        """
        from carnot.agentic import arc_solver_kit as kit

        arc = kit.offline_arcade()
        offenders = []
        for game in sorted(EXPECTED_REPLAY - RUNS_OUT_EXEMPT):
            env = arc.make(game, scorecard_id=arc.open_scorecard())
            frame = env.reset()
            labels = adapters.get_adapter(game).action_labels(env, frame, tuple(range(500)))
            if labels:
                offenders.append((game, len(labels)))
        assert not offenders, (
            f"replay adapters that still offer actions once their plan is exhausted: "
            f"{offenders}. Either they gained a real L0 action space (upgrade -- reclassify "
            "them) or the plan-exhaustion assumption no longer holds."
        )
