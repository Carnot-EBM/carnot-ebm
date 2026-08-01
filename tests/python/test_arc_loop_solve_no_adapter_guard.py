"""The unregistered-game guard in `arc_loop_solve._live_verifier_for_adapter`.

Spec coverage: REQ-ARC-ADAPTER-DEPTH-001

Origin: 2026-07-31. `adapters.get_adapter()` returns None for a game that was never
registered, and this function walked straight into `adapter.hand_verifier` on that None.
The resulting `AttributeError: 'NoneType' object has no attribute 'hand_verifier'` reads
like a broken solve rather than a missing registration -- so sc25, tn36 and wa30 looked
UNSOLVABLE offline when they were merely UNREGISTERED, and sat outside every A/B corpus
built through `build_progress_window` until someone read the traceback carefully. All three
were registered on 2026-07-31; the guard remains for the next unregistered game.

The distinction the guard buys is not cosmetic. "This game has no adapter" is actionable
and cheap to fix; "AttributeError deep in a verifier loader" sends you looking for a bug in
the solver. The first framing would have surfaced sc25/tn36 immediately.

SCOPE, stated so nobody over-reads this: offline dev path only. The scored agent
(`arc_competition_agent.E3AgentPolicy`) has ZERO adapter references -- confirmed by
import-closure analysis over its 43 carnot modules -- because a hidden game can never have
an adapter. Nothing here can affect a submission.
"""

import sys

import pytest

from carnot.paths import repo_root

sys.path.insert(0, str(repo_root() / "scripts"))


@pytest.fixture(scope="module")
def loop():
    return pytest.importorskip("arc_loop_solve")


@pytest.fixture(scope="module")
def adapters():
    from carnot.agentic import arc_game_adapters

    return arc_game_adapters


class TestUnregisteredGameIsNamed:
    """REQ: an unregistered game must say so, not raise AttributeError."""

    def test_none_adapter_raises_no_adapter_for_game(self, loop) -> None:
        with pytest.raises(loop.NoAdapterForGame):
            loop._live_verifier_for_adapter("zz99", None)

    def test_it_is_not_an_attributeerror(self, loop) -> None:
        """The regression itself. AttributeError is what sent the last reader hunting."""
        with pytest.raises(loop.NoAdapterForGame) as exc:
            loop._live_verifier_for_adapter("zz99", None)
        assert not isinstance(exc.value, AttributeError)

    def test_message_names_the_game_and_the_fix(self, loop) -> None:
        """A guard that fires without saying what to do just relocates the confusion."""
        with pytest.raises(loop.NoAdapterForGame) as exc:
            loop._live_verifier_for_adapter("zz99", None)
        msg = str(exc.value)
        assert "zz99" in msg
        assert "_BUILDERS" in msg, "message should name where to register an adapter"

    def test_message_lists_what_is_registered(self, loop, adapters) -> None:
        """Listing the registered set turns 'is this a typo?' into a one-glance answer."""
        with pytest.raises(loop.NoAdapterForGame) as exc:
            loop._live_verifier_for_adapter("zz99", None)
        msg = str(exc.value)
        for game in list(adapters.adaptered_games())[:3]:
            assert game in msg

    def test_distinct_type_so_a_sweep_can_skip(self, loop) -> None:
        """A caller sweeping every game must be able to tell 'not registered' from a real
        solve failure and skip, rather than aborting the sweep."""
        assert issubclass(loop.NoAdapterForGame, LookupError)
        assert loop.NoAdapterForGame is not LookupError


class TestRegisteredGamesUnaffected:
    """The guard must not change behaviour for the 25 games that DO have adapters."""

    def test_registered_game_still_resolves_a_verifier(self, loop, adapters) -> None:
        ad = adapters.get_adapter("tr87")
        assert ad is not None, "tr87 should be registered; fixture assumption broken"
        verifier, source = loop._live_verifier_for_adapter("tr87", ad)
        assert verifier is not None
        assert isinstance(source, str) and source

    def test_all_25_public_games_are_registered(self, adapters) -> None:
        """Pins full coverage so LOSING an adapter is a deliberate, visible edit.

        History, kept because the test changed meaning and the reason matters. It was
        originally `test_wa30_is_the_only_unregistered_public_game`, asserting 24/25 -- wa30
        was excluded because its registry row had no runnable reproduction path
        (`reproduce: None`, `action_model: None`). That turned out to be a budget artifact,
        not a lost solve: the banked 670-action route gate-verified to L9 in 0.8s offline
        once run outside an interactive step budget
        (results/outer_loop_arc_wa30_reproduction_gate_20260731.json). wa30 got its adapter
        the same day, and this assertion inverted from "one is missing" to "none are".

        The failure it caught is the point: adding `_wa30` broke this test, which is exactly
        what a pinning test is for -- it forced this edit to be conscious rather than
        letting coverage drift silently in either direction.
        """
        public_25 = {
            "ar25",
            "bp35",
            "cd82",
            "cn04",
            "dc22",
            "ft09",
            "g50t",
            "ka59",
            "lf52",
            "lp85",
            "ls20",
            "m0r0",
            "r11l",
            "re86",
            "s5i5",
            "sb26",
            "sc25",
            "sk48",
            "sp80",
            "su15",
            "tn36",
            "tr87",
            "tu93",
            "vc33",
            "wa30",
        }
        unregistered = public_25 - set(adapters.adaptered_games())
        assert unregistered == set(), (
            f"public games with no GameAdapter: {sorted(unregistered)}. All 25 were registered "
            "as of 2026-07-31; losing one silently removes a game from every offline induction "
            "corpus. If this is intentional, update this test deliberately."
        )
