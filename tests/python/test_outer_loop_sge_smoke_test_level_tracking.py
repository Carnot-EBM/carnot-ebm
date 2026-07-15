"""Regression test for REQ-ARC-FCP-5699-5: honest level tracking in the SGE smoke-test
harness (scripts/outer_loop_sge_smoke_test.py).

Origin: direct inspection of every artifact this harness had ever produced (7 real-GPU
runs across g50t/sk48/cd82) found `level_before`/`level_after` = 0 on every logged action
in every run -- the real environment level never left 0. The reported `max_level_reached`
(2 for g50t, 1 for sk48/cd82) was an unenforced-floor artifact: `run_game()` initialized
`max_level = prior_levels` (an INFORMATIONAL label from ops/arc_solve_registry.yaml, never
applied as an env seed -- this harness only ever does a bare `env.reset()`) and folded real
observations into that SAME variable, so `max(max_level, after_level)` silently reported
the assumed starting point forever since `prior_levels` was always >= the real level ever
observed.

This test mocks out the ARC-specific machinery (arcade/env/policy) entirely so it can
verify the counting logic in isolation, without a real GPU or the offline arcade.
`outer_loop_sge_smoke_test.py` still imports `LocalGGUFProposer`/`E3AgentPolicy` at
module scope, which pulls in torch -- ~500MB+ RSS on first import in a worker process,
tripping the project's per-test RSS watchdog (`pyproject.toml`, `memory_watchdog_skip`
is the designed exemption for known high-RSS tests; see `test_arc_live_ttt_gated.py` for
the precedent).

SCENARIO-ARC-FCP-5699-5-LEVEL-TRACKING-NEVER-BLENDS-WITH-UNVERIFIED-PRIOR
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.memory_watchdog_skip

REPO = Path(__file__).resolve().parents[2]


def _load_smoke_test_module():
    spec = importlib.util.spec_from_file_location(
        "outer_loop_sge_smoke_test", str(REPO / "scripts" / "outer_loop_sge_smoke_test.py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class _FakeFrame:
    def __init__(self, level: int) -> None:
        self.levels_completed = level  # read by arc_agi3_live_adapter._levels_completed


class _ScriptedFakeEnv:
    """A fake ARC env whose level trajectory is scripted, independent of prior_levels."""

    def __init__(self, level_sequence: list[int]) -> None:
        # level_sequence[0] is the level right after reset(); each subsequent entry is
        # the level after one step().
        self._sequence = list(level_sequence)
        self._index = 0

    def reset(self):
        self._index = 0
        return _FakeFrame(self._sequence[self._index])

    def step(self, action, data=None):
        self._index += 1
        if self._index >= len(self._sequence):
            return None
        return _FakeFrame(self._sequence[self._index])


class _ScriptedFakePolicy:
    """Drives a fixed number of real actions (action id 1, no data) then stops."""

    def __init__(self, num_actions: int) -> None:
        self._remaining = num_actions
        self._did_reset = False

    def is_done(self, frames, latest):
        return self._remaining <= 0 and self._did_reset

    def next_move(self, frames, latest):
        if not self._did_reset:
            self._did_reset = True
            return "RESET", None
        if self._remaining <= 0:
            return None, None
        self._remaining -= 1
        return 1, None


def _run_with_fakes(
    mod, *, level_sequence: list[int], num_actions: int, prior_levels: int, target_level: int
):
    fake_env = _ScriptedFakeEnv(level_sequence)
    fake_arcade = MagicMock()
    fake_arcade.make.return_value = fake_env
    fake_arcade.open_scorecard.return_value = "sc-1"
    mod.kit.offline_arcade = MagicMock(return_value=fake_arcade)  # type: ignore[attr-defined]

    fake_policy = _ScriptedFakePolicy(num_actions)
    mod.E3AgentPolicy = MagicMock(return_value=fake_policy)  # type: ignore[attr-defined]

    fake_gguf = MagicMock()
    fake_gguf.repo_substr = "fake-model"

    # router.last_diagnostics is read every loop iteration; a bare SGECandidateRouter
    # backed by a no-op completer never raises and its diagnostics default is harmless
    # for this test (only level-tracking is under test here).
    class _NoOpCompleter:
        def complete_text(self, *args, **kwargs):
            return False, "no completer configured in this test"

    real_router_cls = mod.SGECandidateRouter
    real_proposer_cls = mod.LLMStrategyProposer

    def _make_router(**kwargs):
        kwargs["proposer"] = real_proposer_cls(completer=_NoOpCompleter())
        return real_router_cls(**kwargs)

    mod.SGECandidateRouter = _make_router  # type: ignore[assignment]
    try:
        return mod.run_game("fake_game", prior_levels, target_level, budget=50, gguf=fake_gguf)
    finally:
        mod.SGECandidateRouter = real_router_cls


def test_real_level_never_advances_reports_leveled_up_false_regardless_of_prior_levels_label():
    """The core regression: a run whose real env level never advances past its initial
    value must report leveled_up=False, even when prior_levels claims a much deeper
    starting point than the real trajectory ever shows -- the exact 2026-07-15 bug."""
    mod = _load_smoke_test_module()
    result = _run_with_fakes(
        mod,
        level_sequence=[0] * 10,  # level never changes across resets/steps
        num_actions=5,
        prior_levels=2,  # an unverified, much-higher informational label
        target_level=3,
    )
    assert result["real_initial_level"] == 0
    assert result["real_max_level_observed"] == 0
    assert result["leveled_up"] is False
    # the informational labels are preserved (for provenance) but must NOT have been
    # blended into the honest fields above
    assert result["prior_levels_reproduced"] == 2
    assert result["max_level_reached"] == 0  # NOT 2 (the pre-fix bug's value)


def test_real_level_advancing_reports_leveled_up_true():
    """A genuine level advance (0 -> 1 after some steps) is correctly detected and
    reported, independent of what prior_levels claims."""
    mod = _load_smoke_test_module()
    result = _run_with_fakes(
        mod,
        level_sequence=[0, 0, 0, 1, 1, 1],  # advances to level 1 partway through
        num_actions=5,
        prior_levels=1,
        target_level=2,
    )
    assert result["real_initial_level"] == 0
    assert result["real_max_level_observed"] == 1
    assert result["leveled_up"] is True


def test_methodology_note_present_and_labels_prior_levels_as_informational():
    mod = _load_smoke_test_module()
    result = _run_with_fakes(
        mod,
        level_sequence=[0] * 10,
        num_actions=3,
        prior_levels=1,
        target_level=2,
    )
    assert "INFORMATIONAL" in result["methodology_note"]
    assert "does NOT seed" in result["methodology_note"]


def test_req_arc_fcp_5699_5_spec_declares_honest_level_tracking() -> None:
    spec_path = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"
    spec = spec_path.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-FCP-5699-5") : spec.index("### REQ-ARC-WMTE-5596")]

    for marker in (
        "REQ-ARC-FCP-5699-5",
        "SCENARIO-ARC-FCP-5699-5-LEVEL-TRACKING-NEVER-BLENDS-WITH-UNVERIFIED-PRIOR",
        "real_initial_level",
        "real_max_level_observed",
        "methodology_note",
    ):
        assert marker in section
