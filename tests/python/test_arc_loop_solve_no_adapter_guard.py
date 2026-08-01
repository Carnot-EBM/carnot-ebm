"""The unregistered-game guard in `arc_loop_solve._live_verifier_for_adapter`.

Spec coverage: REQ-ARC-ADAPTER-DEPTH-001

Origin: 2026-07-31. `adapters.get_adapter()` returns None for a game that was never
registered, and this function walked straight into `adapter.hand_verifier` on that None.
The resulting `AttributeError: 'NoneType' object has no attribute 'hand_verifier'` reads
like a broken solve rather than a missing registration -- so sc25, tn36 and wa30 looked
UNSOLVABLE offline when they were merely UNREGISTERED, and sat outside every A/B corpus
built through `build_progress_window` until someone read the traceback carefully.

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
    """The guard must not change behaviour for the 24 games that DO have adapters."""

    def test_registered_game_still_resolves_a_verifier(self, loop, adapters) -> None:
        ad = adapters.get_adapter("tr87")
        assert ad is not None, "tr87 should be registered; fixture assumption broken"
        verifier, source = loop._live_verifier_for_adapter("tr87", ad)
        assert verifier is not None
        assert isinstance(source, str) and source

    def test_wa30_is_the_only_unregistered_public_game(self, adapters) -> None:
        """Pins the current state so re-registering wa30 -- or losing an adapter -- is a
        deliberate, visible edit rather than a silent drift.

        wa30 is unregistered because its registry row has no runnable reproduction path at
        all (`reproduce: None`, `action_model: None`), filed in ops/known-issues.md
        2026-07-31. Every other public game's RE was already captured and its adapter
        either existed or was transcribable from the registry.
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
        assert unregistered == {"wa30"}, (
            f"expected only wa30 unregistered, got {sorted(unregistered)}. If an adapter was "
            "added or removed, update this test deliberately."
        )
