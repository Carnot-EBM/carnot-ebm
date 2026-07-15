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
    mod,
    *,
    level_sequence: list[int],
    num_actions: int,
    prior_levels: int,
    target_level: int,
    router_mode: str = "sge",
    induction_enabled: bool = False,
):
    fake_env = _ScriptedFakeEnv(level_sequence)
    fake_arcade = MagicMock()
    fake_arcade.make.return_value = fake_env
    fake_arcade.open_scorecard.return_value = "sc-1"
    mod.kit.offline_arcade = MagicMock(return_value=fake_arcade)  # type: ignore[attr-defined]

    fake_policy = _ScriptedFakePolicy(num_actions)
    e3_policy_mock = MagicMock(return_value=fake_policy)
    mod.E3AgentPolicy = e3_policy_mock  # type: ignore[attr-defined]

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
        result = mod.run_game(
            "fake_game",
            prior_levels,
            target_level,
            budget=50,
            gguf=fake_gguf,
            router_mode=router_mode,
            induction_enabled=induction_enabled,
        )
        result["_e3_policy_call_kwargs"] = dict(e3_policy_mock.call_args.kwargs)
        return result
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


def test_baseline_router_mode_uses_deterministic_router_and_still_tracks_honestly():
    """REQ-ARC-FCP-5699-6: router_mode="baseline" swaps in the REAL (not mocked)
    BoundedStrategyCandidateRouter -- no LLM call at all -- and honest level tracking
    behaves identically regardless of which router drove the actions."""
    mod = _load_smoke_test_module()
    result = _run_with_fakes(
        mod,
        level_sequence=[0, 0, 0, 1, 1, 1],
        num_actions=5,
        prior_levels=1,
        target_level=2,
        router_mode="baseline",
    )
    assert result["router_mode"] == "baseline"
    assert result["real_initial_level"] == 0
    assert result["real_max_level_observed"] == 1
    assert result["leveled_up"] is True
    assert result["llm_strategy_proposer_used_any_step"] is False
    assert (
        result["inference_substrate"] == "offline_arcade_live_agent_runtime_self_discovery_no_llm"
    )


def test_induction_disabled_by_default_passes_noop_stub():
    """The pre-existing default: E3AgentPolicy gets the _NoOpInductionProposer stub, never
    a real GPU-backed proposer, unless induction_enabled=True is explicitly passed."""
    mod = _load_smoke_test_module()
    result = _run_with_fakes(
        mod,
        level_sequence=[0] * 10,
        num_actions=3,
        prior_levels=1,
        target_level=2,
    )
    assert result["induction_enabled"] is False
    passed_proposer = result["_e3_policy_call_kwargs"]["proposer"]
    assert isinstance(passed_proposer, mod._NoOpInductionProposer)
    assert result["induction_attempts"] == []
    assert result["induction_attempts_not_skipped"] == 0


def test_induction_enabled_passes_real_local_gguf_proposer_on_a_dedicated_port():
    """REQ-ARC-FCP-5699-8: induction_enabled=True constructs a REAL LocalGGUFProposer (not
    mocked -- construction is lazy/cheap, no GPU/network call happens until .induce() is
    invoked, which this fake policy never calls) using the frozen live-submission
    generator's defaults, on port 8930 -- distinct from the SGE completer's own 8929, so
    the two different models (Qwen3.5-9B-MTP for induction, gemma-4-12B-it for SGE
    strategy proposals) never fight over the same llama-server."""
    mod = _load_smoke_test_module()
    result = _run_with_fakes(
        mod,
        level_sequence=[0] * 10,
        num_actions=3,
        prior_levels=1,
        target_level=2,
        induction_enabled=True,
    )
    assert result["induction_enabled"] is True
    passed_proposer = result["_e3_policy_call_kwargs"]["proposer"]
    assert isinstance(passed_proposer, mod.LocalGGUFProposer)
    assert passed_proposer.repo_substr == "Qwen3.5-9B-MTP"
    assert passed_proposer.port == 8930
    assert passed_proposer.mtp is True
    # a fake policy never calls .induce(), so induction_attempts stays empty here -- this
    # test verifies WHAT was configured, not that induction ran (that needs a real GPU run)
    assert result["induction_attempts"] == []


def _canned_run_game_result(game: str, *_args, **_kwargs) -> dict:
    return {
        "game": game,
        "router_mode": "sge",
        "induction_enabled": False,
        "induction_attempts": [],
        "induction_attempts_not_skipped": 0,
        "prior_levels_reproduced": 1,
        "target_level": 2,
        "real_initial_level": 0,
        "real_max_level_observed": 0,
        "leveled_up": False,
        "attempts": 0,
        "duration_s": 0.0,
        "llm_strategy_proposer_used_any_step": False,
        "reflection_nudge_fired_any_step": False,
    }


def test_req_arc_fcp_5699_9_single_game_subset_run_does_not_clobber_full_suite_summary(
    monkeypatch, tmp_path
):
    """REQ-ARC-FCP-5699-9: the exact bug found while running sp80 alone -- a subset run
    (explicit game args) must write its summary to a suffixed path, never overwriting
    the shared outer_loop_sge_smoke_test_suite.json a full default run produces."""
    mod = _load_smoke_test_module()
    monkeypatch.setattr(mod, "REPO", tmp_path)
    (tmp_path / "results").mkdir()
    # simulate a pre-existing full-suite summary (as the real committed one is)
    full_suite_path = tmp_path / "results" / "outer_loop_sge_smoke_test_suite.json"
    full_suite_path.write_text('{"games": ["g50t", "sk48", "cd82"]}')

    monkeypatch.setattr(mod, "run_game", _canned_run_game_result)
    monkeypatch.setattr(mod.sys, "argv", ["outer_loop_sge_smoke_test.py", "sp80"])

    rc = mod.main()
    assert rc == 0
    # the pre-existing full-suite summary must be untouched
    assert full_suite_path.read_text() == '{"games": ["g50t", "sk48", "cd82"]}'
    # the sp80 subset run gets its own, non-colliding summary path
    subset_summary_path = tmp_path / "results" / "outer_loop_sge_smoke_test_suite_sp80.json"
    assert subset_summary_path.exists()
    per_game_path = tmp_path / "results" / "outer_loop_sge_smoke_test_sp80.json"
    assert per_game_path.exists()


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


def test_req_arc_fcp_5699_6_spec_declares_baseline_control() -> None:
    spec_path = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"
    spec = spec_path.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-FCP-5699-6") : spec.index("### REQ-ARC-WMTE-5596")]

    for marker in (
        "REQ-ARC-FCP-5699-6",
        "SCENARIO-ARC-FCP-5699-6-CONTROL-ISOLATES-THE-ROUTER-UNDER-TEST",
        "BoundedStrategyCandidateRouter",
        "router_mode",
        "--baseline",
    ):
        assert marker in section


def test_req_arc_fcp_5699_7_spec_declares_budget_flag() -> None:
    spec_path = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"
    spec = spec_path.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-FCP-5699-7") : spec.index("### REQ-ARC-WMTE-5596")]

    for marker in (
        "REQ-ARC-FCP-5699-7",
        "SCENARIO-ARC-FCP-5699-7-BUDGET-INCREASE-ALONE-DOES-NOT-CHANGE-A-STRUCTURAL-NULL",
        "--budget",
        "budget=250",
        "real_max_level_observed=0",
    ):
        assert marker in section


def test_budget_override_flows_through_to_run_game(monkeypatch):
    """--budget N is a CLI-level parse concern in main(); this verifies the underlying
    run_game() budget parameter (already covered by real usage above) is what actually
    bounds the loop, by checking a small budget caps attempts at that count."""
    mod = _load_smoke_test_module()
    fake_env = _ScriptedFakeEnv([0] * 20)
    fake_arcade = MagicMock()
    fake_arcade.make.return_value = fake_env
    fake_arcade.open_scorecard.return_value = "sc-1"
    mod.kit.offline_arcade = MagicMock(return_value=fake_arcade)  # type: ignore[attr-defined]
    mod.E3AgentPolicy = MagicMock(return_value=_ScriptedFakePolicy(50))  # type: ignore[attr-defined]
    fake_gguf = MagicMock()
    fake_gguf.repo_substr = "fake-model"
    result = mod.run_game("fake_game", 1, 2, budget=3, gguf=fake_gguf, router_mode="baseline")
    assert result["attempts"] <= 3


def test_req_arc_fcp_5699_8_spec_declares_induction_trust_gate() -> None:
    spec_path = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"
    spec = spec_path.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-FCP-5699-8") : spec.index("### REQ-ARC-WMTE-5596")]

    for marker in (
        "REQ-ARC-FCP-5699-8",
        "SCENARIO-ARC-FCP-5699-8-INDUCTION-RE-ENABLED-STILL-GATED-BY-TRUST-CHECK",
        "hidden_state_trust_below_threshold",
        "HIDDEN_STATE_GAME_IDS",
        "--induction",
    ):
        assert marker in section


def test_req_arc_fcp_5699_9_spec_declares_non_hidden_state_finding() -> None:
    spec_path = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"
    spec = spec_path.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-FCP-5699-9") : spec.index("### REQ-ARC-WMTE-5596")]

    for marker in (
        "REQ-ARC-FCP-5699-9",
        "SCENARIO-ARC-FCP-5699-9-DOCUMENTED-LEVER-DOES-NOT-RESCUE-A-GENUINELY-ZERO-SCORING-CANDIDATE",
        "world_model_accuracy_below_threshold",
        "CARNOT_ARC_TRUST_METRIC",
        "verify_cell_recall",
    ):
        assert marker in section
