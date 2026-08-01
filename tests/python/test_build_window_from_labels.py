"""`build_window_from_labels` — induction windows without a GameAdapter.

Spec coverage: REQ-ARC-WMTE-5717

Origin: 2026-07-31, from an operator question asked three times: what is the purpose of a
GameAdapter that only solves one level?

Measured answer: for 18 of the 25 public games, almost none. Those adapters' `action_labels`
return a SINGLE forced label at each step, replaying a plan that was already banked. The
verifier-routed search, hazard pruning, learned-verifier warm start and state dedup that
`solve_adaptered` provides are all unexercised when the action set has one element -- the
search is a straight line through a known solution. They were not solving L1; they were
satisfying a function signature, because `build_window` obtained its labels via
`solve_adaptered`, which requires an adapter.

`arc_solver_kit.reproduce()` already showed the coupling was unnecessary: it takes
`(labels, apply)` and was used the same day to gate-verify wa30's entire 670-action L9 route
with no adapter at all. `build_window_from_labels` is that contract for window building.

The load-bearing test here is `test_label_path_matches_adapter_path`: the refactor is only
safe if the two routes produce byte-identical transitions, so a corpus built either way is
the same corpus.
"""

import json

import pytest

from carnot.paths import repo_root

pytestmark = pytest.mark.skipif(
    not (repo_root() / "environment_files").exists(),
    reason="offline ARC environment_files not present",
)

# MODULE-LEVEL, deliberately, following tests/python/test_arc_live_generator_pin.py.
# `build_window("wa30")` runs a real `solve_adaptered` and loads the offline arcade, costing
# ~520 MB. This repo's pytest memory watchdog attributes any growth between a test's setup and
# teardown to that test, so computing it inside a test or fixture fails with a spurious
# "Memory leak: +519MB". Doing it at COLLECTION time puts the cost outside the watchdog's
# measurement window -- the same fix that file documents for its own 590 MB import.
_EQUIVALENCE_PAIR = None
if (repo_root() / "environment_files").exists():  # pragma: no cover - offline SDK boundary
    try:
        import json as _json
        import sys as _sys

        _sys.path.insert(0, str(repo_root() / "scripts"))
        from carnot.agentic.arc_game_adapters import _default_json_apply, _json_action_label
        from carnot.experiment_5717_playbook_exemplars_stall_induction_ab import (
            build_window,
            build_window_from_labels,
        )

        _probe = repo_root() / "results" / "outer_loop_fable5_wa30_probe_l9.json"
        _seq = _json.loads(_probe.read_text())["action_sequence"][:33]
        _labels = [_json_action_label(int(a["action"])) for a in _seq]
        _EQUIVALENCE_PAIR = (
            build_window_from_labels("wa30", _labels, _default_json_apply),
            build_window("wa30"),
        )
    except Exception:  # pragma: no cover - a collection-time failure must not hide the suite
        _EQUIVALENCE_PAIR = None


@pytest.fixture(scope="module")
def mod():
    return pytest.importorskip("carnot.experiment_5717_playbook_exemplars_stall_induction_ab")


@pytest.fixture(scope="module")
def helpers():
    from carnot.agentic.arc_game_adapters import _default_json_apply, _json_action_label

    return _default_json_apply, _json_action_label


@pytest.fixture(scope="module")
def wa30_l1_labels(helpers):
    """wa30's 33-action L1 prefix, from the route gate-verified to L9 on 2026-07-31."""
    _, label = helpers
    probe = repo_root() / "results" / "outer_loop_fable5_wa30_probe_l9.json"
    seq = json.loads(probe.read_text())["action_sequence"][:33]
    return [label(int(a["action"])) for a in seq]


class TestLabelPathNeedsNoAdapter:
    """REQ-ARC-WMTE-5717: a banked route can build a window on its own."""

    def test_builds_a_window_from_labels_alone(self, mod, helpers, wa30_l1_labels) -> None:
        apply, _ = helpers
        out = mod.build_window_from_labels("wa30", wa30_l1_labels, apply)
        assert out is not None, "banked L1 route should yield a window"
        window, full, cell = out
        assert len(window) > 0 and len(full) == len(wa30_l1_labels)
        assert cell >= 1

    def test_window_contains_the_levelup(self, mod, helpers, wa30_l1_labels) -> None:
        """The window's whole purpose is to straddle the L0->L1 boundary."""
        apply, _ = helpers
        window, _, _ = mod.build_window_from_labels("wa30", wa30_l1_labels, apply)
        levelups = [t for t in window if t.level_after > t.level_before]
        assert len(levelups) >= 1, "window must contain a real level-up transition"

    def test_empty_labels_returns_none(self, mod, helpers) -> None:
        apply, _ = helpers
        assert mod.build_window_from_labels("wa30", [], apply) is None


class TestEquivalence:
    """The refactor is only safe if both routes yield the same corpus."""

    def test_label_path_matches_adapter_path(self) -> None:
        """THE load-bearing test.

        `build_window` (adapter -> solve_adaptered -> labels) and
        `build_window_from_labels` (labels given directly) must produce identical
        transitions. If they diverge, a corpus built one way is not the corpus built the
        other way, and every A/B measured through either becomes incomparable.

        Both windows are built at module scope; see the note there on the memory watchdog.
        """
        if _EQUIVALENCE_PAIR is None:
            pytest.skip("windows could not be built at collection time")
        via_labels, via_adapter = _EQUIVALENCE_PAIR
        assert via_labels is not None and via_adapter is not None

        def sig(triple):
            window, full, cell = triple
            return (
                [(t.action, t.data, t.level_before, t.level_after) for t in window],
                len(full),
                cell,
            )

        assert sig(via_labels) == sig(via_adapter)


class TestUnparseableLabelFailsLoud:
    """A wrong action would not crash -- it would teach false dynamics. Worse than a crash."""

    @staticmethod
    def _lenient_apply(env, _label, _frame=None):
        """Steps the env for ANY label, JSON or not.

        `_default_json_apply` cannot be used here: it does `json.loads(label)`, so a
        payload-carrying label like "C:1" dies inside apply() before reaching the
        action-parsing code under test. In the real ka59 flow the adapter's own apply
        handles that label; this stand-in isolates the parse step.
        """
        from arcengine import GameAction

        from carnot.agentic.arc_agi3_live_adapter import _game_action

        return env.step(_game_action(GameAction, 1), data=None)

    def test_non_integer_label_without_hook_raises(self, mod) -> None:
        with pytest.raises(ValueError, match="cannot parse action label"):
            mod.build_window_from_labels("wa30", ["C:1"], self._lenient_apply)

    def test_error_names_the_remedy(self, mod) -> None:
        """The message must point at label_to_action_data, or the next reader repeats the
        ka59 investigation from scratch."""
        with pytest.raises(ValueError) as exc:
            mod.build_window_from_labels("wa30", ["C:1"], self._lenient_apply)
        assert "label_to_action_data" in str(exc.value)

    def test_hook_is_used_when_supplied(self, mod) -> None:
        """With a hook, a payload-carrying label resolves instead of raising."""
        seen = []

        def hook(_env, label):
            seen.append(label)
            return 6, {"x": 1, "y": 2}

        # A single click label will not reach a level-up, so the call returns None -- but it
        # must get far enough to CONSULT the hook rather than raising on the parse.
        mod.build_window_from_labels(
            "wa30", ["C:1"], self._lenient_apply, label_to_action_data=hook
        )
        assert seen == ["C:1"]
