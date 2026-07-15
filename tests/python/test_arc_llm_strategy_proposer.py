"""Tests for the LLM Strategy-Guided Exploration (SGE) candidate router.

Outer-loop investigation, 2026-07-10: verifies the mechanism arXiv:2603.02045 describes
(mixed-temperature parallel strategy sampling + reflection) with a fake, deterministic
`TextCompleter` -- no GPU/model required for this suite. A separate manual smoke script
exercises the real `LocalGGUFProposer` path against the offline ARC arcade.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"

from carnot.agentic.arc_llm_strategy_proposer import (
    AntiStagnationDiversityController,
    StrategyCollapseThresholds,
    LLMStrategyProposer,
    SGECandidateRouter,
    _candidate_action,
    _candidate_signature,
    _fallback_score,
    _outcome_is_null,
    _strategy_distance,
    parse_propose_reply,
    parse_reflect_reply,
)


def _candidate(action: int, x: int, y: int, **scores: float) -> dict:
    row = {"action": action, "data": {"x": x, "y": y}}
    row.update(scores)
    return row


class FakeCompleter:
    """Deterministic stand-in for LocalGGUFProposer.complete_text.

    `script` is a list of (ok, text) tuples consumed in call order; when exhausted the
    last entry repeats. Records every call for assertion.
    """

    def __init__(self, script: list[tuple[bool, str]]) -> None:
        self.script = script
        self.calls: list[dict] = []

    def complete_text(self, prompt, *, max_tokens=None, temperature=0.1, stop=None):
        index = min(len(self.calls), len(self.script) - 1)
        self.calls.append(
            {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature, "stop": stop}
        )
        return self.script[index]


# ---------------------------------------------------------------------------
# parse_propose_reply / parse_reflect_reply
# ---------------------------------------------------------------------------


def test_parse_propose_reply_well_formed():
    parsed = parse_propose_reply("STRATEGY: probe the top-right corner\nCHOICE: 2\n")
    assert parsed == {
        "parse_ok": True,
        "strategy_text": "probe the top-right corner",
        "chosen_index": 2,
        "raw": "STRATEGY: probe the top-right corner\nCHOICE: 2\n",
    }


def test_parse_propose_reply_missing_choice():
    parsed = parse_propose_reply("STRATEGY: probe the corner\n")
    assert parsed["parse_ok"] is False
    assert parsed["chosen_index"] is None
    assert parsed["strategy_text"] == "probe the corner"


def test_parse_propose_reply_missing_strategy():
    parsed = parse_propose_reply("CHOICE: 3\n")
    assert parsed["parse_ok"] is False
    assert parsed["chosen_index"] is None


def test_parse_propose_reply_non_integer_choice():
    parsed = parse_propose_reply("STRATEGY: x\nCHOICE: not-a-number\n")
    assert parsed["parse_ok"] is False


def test_parse_propose_reply_empty_text():
    parsed = parse_propose_reply("")
    assert parsed == {"parse_ok": False, "strategy_text": "", "chosen_index": None, "raw": ""}


def test_parse_reflect_reply_well_formed():
    assert (
        parse_reflect_reply("REVISED_STRATEGY: try the left panel next\n")
        == "try the left panel next"
    )


def test_parse_reflect_reply_malformed():
    assert parse_reflect_reply("no marker here") == ""


# ---------------------------------------------------------------------------
# LLMStrategyProposer
# ---------------------------------------------------------------------------


def test_propose_one_parses_completer_output():
    completer = FakeCompleter([(True, "STRATEGY: try salient blob\nCHOICE: 1\n")])
    proposer = LLMStrategyProposer(completer=completer)
    result = proposer.propose_one(
        "Game: g1", ["[0] action=6 x=1,y=1", "[1] action=6 x=2,y=2"], temperature=0.5
    )
    assert result["parse_ok"] is True
    assert result["chosen_index"] == 1
    assert result["strategy_text"] == "try salient blob"
    assert result["completer_ok"] is True
    assert result["temperature"] == 0.5
    assert completer.calls[0]["temperature"] == 0.5


def test_propose_one_handles_completer_failure():
    completer = FakeCompleter([(False, "GPU llama-server failed")])
    proposer = LLMStrategyProposer(completer=completer)
    result = proposer.propose_one("Game: g1", ["[0] action=6 x=1,y=1"], temperature=0.3)
    assert result["completer_ok"] is False
    assert result["parse_ok"] is False
    assert result["chosen_index"] is None


def test_propose_many_samples_each_temperature():
    completer = FakeCompleter(
        [
            (True, "STRATEGY: a\nCHOICE: 0\n"),
            (True, "STRATEGY: b\nCHOICE: 1\n"),
            (True, "STRATEGY: c\nCHOICE: 0\n"),
        ]
    )
    proposer = LLMStrategyProposer(completer=completer)
    results = proposer.propose_many(
        "Game: g1", ["[0] action=6 x=1,y=1", "[1] action=6 x=2,y=2"], temperatures=(0.3, 0.6, 0.9)
    )
    assert [r["temperature"] for r in results] == [0.3, 0.6, 0.9]
    assert [r["chosen_index"] for r in results] == [0, 1, 0]
    assert len(completer.calls) == 3


def test_reflect_with_empty_history_short_circuits():
    completer = FakeCompleter([(True, "REVISED_STRATEGY: unused\n")])
    proposer = LLMStrategyProposer(completer=completer)
    result = proposer.reflect("Game: g1", [])
    assert result == {"parse_ok": False, "revised_strategy": "", "raw": ""}
    assert completer.calls == []  # never invoked the completer for empty history


def test_reflect_parses_revised_strategy():
    completer = FakeCompleter([(True, "REVISED_STRATEGY: focus on the left panel\n")])
    proposer = LLMStrategyProposer(completer=completer)
    history = [{"strategy_text": "probe corners", "outcome": "no_change"}]
    result = proposer.reflect("Game: g1", history)
    assert result["parse_ok"] is True
    assert result["revised_strategy"] == "focus on the left panel"
    assert "probe corners" in completer.calls[0]["prompt"]
    assert "no_change" in completer.calls[0]["prompt"]


def test_reflect_handles_completer_failure():
    completer = FakeCompleter([(False, "GPU down")])
    proposer = LLMStrategyProposer(completer=completer)
    result = proposer.reflect("Game: g1", [{"strategy_text": "x", "outcome": "y"}])
    assert result == {
        "parse_ok": False,
        "revised_strategy": "",
        "raw": "GPU down",
        "completer_ok": False,
        "nudge_fired": False,
        "consecutive_null_outcomes": 0,
    }


# ---------------------------------------------------------------------------
# reflect() anti-stagnation prompt nudge (REQ-ARC-FCP-5699-3)
# ---------------------------------------------------------------------------


def test_reflect_no_nudge_on_healthy_history():
    """A single non-null outcome and no taboo strategies -> plain prompt, no nudge."""
    completer = FakeCompleter([(True, "REVISED_STRATEGY: keep probing\n")])
    proposer = LLMStrategyProposer(completer=completer)
    history = [{"strategy_text": "probe corners", "outcome": "level_up"}]
    result = proposer.reflect("Game: g1", history)
    assert result["nudge_fired"] is False
    assert result["consecutive_null_outcomes"] == 0
    assert "ANTI-STAGNATION" not in completer.calls[0]["prompt"]


def test_reflect_nudge_fires_on_consecutive_null_outcomes():
    """_REFLECT_NUDGE_NULL_STREAK consecutive null outcomes -> nudge spliced into the
    prompt, even with no explicit taboo_strategies passed in."""
    completer = FakeCompleter([(True, "REVISED_STRATEGY: try a different action type\n")])
    proposer = LLMStrategyProposer(completer=completer)
    history = [
        {"strategy_text": "wait and observe", "outcome": "no_change"},
        {"strategy_text": "wait and observe again", "outcome": "no_change"},
    ]
    result = proposer.reflect("Game: g1", history)
    assert result["nudge_fired"] is True
    assert result["consecutive_null_outcomes"] == 2
    prompt = completer.calls[0]["prompt"]
    assert "ANTI-STAGNATION WARNING" in prompt
    assert "the last 2 attempt(s)" in prompt
    assert "must NOT be another minor variation" in prompt


def test_reflect_uses_wider_max_tokens_when_nudge_fires():
    """The nudge lengthens the prompt; the completion call gets more output-token room
    specifically on that path, not on every reflect() call (found empirically necessary
    2026-07-15: a real-GPU nudge-fired reflect call failed to parse at the default
    budget)."""
    completer = FakeCompleter([(True, "REVISED_STRATEGY: try a different action type\n")])
    proposer = LLMStrategyProposer(completer=completer, max_tokens=96, reflect_nudge_max_tokens=160)
    history = [
        {"strategy_text": "wait and observe", "outcome": "no_change"},
        {"strategy_text": "wait and observe again", "outcome": "no_change"},
    ]
    proposer.reflect("Game: g1", history)
    assert completer.calls[0]["max_tokens"] == 160


def test_reflect_uses_default_max_tokens_without_nudge():
    completer = FakeCompleter([(True, "REVISED_STRATEGY: keep going\n")])
    proposer = LLMStrategyProposer(completer=completer, max_tokens=96, reflect_nudge_max_tokens=160)
    history = [{"strategy_text": "probe corners", "outcome": "level_up"}]
    proposer.reflect("Game: g1", history)
    assert completer.calls[0]["max_tokens"] == 96


def test_reflect_nudge_names_taboo_strategies_when_given():
    """Explicit taboo_strategies (from AntiStagnationDiversityController.taboo_set) are
    named verbatim in the nudge, so the model sees exactly what NOT to repeat."""
    completer = FakeCompleter([(True, "REVISED_STRATEGY: click the unexplored panel\n")])
    proposer = LLMStrategyProposer(completer=completer)
    history = [{"strategy_text": "wait and observe", "outcome": "unchanged"}]
    result = proposer.reflect(
        "Game: g1", history, taboo_strategies=["wait and observe", "check the corner"]
    )
    assert result["nudge_fired"] is True
    prompt = completer.calls[0]["prompt"]
    assert "wait and observe; check the corner" in prompt


def test_reflect_nudge_does_not_fire_below_streak_threshold():
    """A single null outcome (below _REFLECT_NUDGE_NULL_STREAK=2) and no taboo strategies
    -> no nudge yet; the softer signal must accumulate before the prompt-level warning
    kicks in, matching the harder deterministic gate's own graduated-signal design."""
    completer = FakeCompleter([(True, "REVISED_STRATEGY: try again\n")])
    proposer = LLMStrategyProposer(completer=completer)
    history = [{"strategy_text": "probe corners", "outcome": "no_change"}]
    result = proposer.reflect("Game: g1", history)
    assert result["nudge_fired"] is False
    assert result["consecutive_null_outcomes"] == 1
    assert "ANTI-STAGNATION" not in completer.calls[0]["prompt"]


# ---------------------------------------------------------------------------
# SGECandidateRouter
# ---------------------------------------------------------------------------


def test_rank_empty_candidates_returns_empty_and_honest_diagnostics():
    router = SGECandidateRouter(
        proposer=LLMStrategyProposer(completer=FakeCompleter([])), game_id="g1"
    )
    result = router.rank(frame=None, candidates=[])
    assert result == []
    assert router.last_diagnostics["llm_strategy_proposer_used"] is False
    assert router.last_diagnostics["reason"] == "no_candidates"


def test_rank_uses_llm_votes_to_promote_a_candidate():
    # All three temperature samples vote for candidate index 1 -> it must rank first
    # even though it has no deterministic score fields at all.
    completer = FakeCompleter(
        [
            (True, "STRATEGY: chase the moving sprite\nCHOICE: 1\n"),
            (True, "STRATEGY: chase the moving sprite\nCHOICE: 1\n"),
            (True, "STRATEGY: chase the moving sprite\nCHOICE: 1\n"),
        ]
    )
    router = SGECandidateRouter(
        proposer=LLMStrategyProposer(completer=completer), game_id="g1", k=3
    )
    candidates = [
        _candidate(6, 0, 0, salience_score=9.0),  # high deterministic score, zero votes
        _candidate(6, 5, 5),  # zero deterministic score, all three votes
    ]
    ranked = router.rank(frame=None, candidates=candidates)
    assert ranked[0] is candidates[1]
    assert router.last_diagnostics["llm_strategy_proposer_used"] is True
    assert router.last_diagnostics["votes_by_index"] == {"1": 3}
    assert router.last_diagnostics["parse_failure_count"] == 0


def test_rank_falls_back_to_deterministic_scores_when_completer_unavailable():
    completer = FakeCompleter([(False, "GPU llama-server failed")] * 3)
    router = SGECandidateRouter(
        proposer=LLMStrategyProposer(completer=completer), game_id="g1", k=3
    )
    candidates = [
        _candidate(6, 0, 0, salience_score=1.0),
        _candidate(6, 5, 5, salience_score=9.0),
    ]
    ranked = router.rank(frame=None, candidates=candidates)
    assert ranked[0] is candidates[1]  # higher salience_score wins the fallback order
    assert router.last_diagnostics["llm_strategy_proposer_used"] is False
    assert router.last_diagnostics["completer_failure_count"] == 3


def test_rank_falls_back_when_all_samples_fail_to_parse():
    completer = FakeCompleter([(True, "not the expected format")] * 3)
    router = SGECandidateRouter(
        proposer=LLMStrategyProposer(completer=completer), game_id="g1", k=3
    )
    candidates = [_candidate(6, 0, 0, salience_score=1.0), _candidate(6, 5, 5, salience_score=9.0)]
    ranked = router.rank(frame=None, candidates=candidates)
    assert ranked[0] is candidates[1]
    assert router.last_diagnostics["llm_strategy_proposer_used"] is False
    assert router.last_diagnostics["parse_failure_count"] == 3


def test_rank_suppresses_repeated_coordinates_across_calls():
    completer = FakeCompleter([(True, "STRATEGY: a\nCHOICE: 0\n")] * 30)
    router = SGECandidateRouter(
        proposer=LLMStrategyProposer(completer=completer), game_id="g1", k=1, temperatures=(0.5,)
    )
    candidates = [_candidate(6, 1, 1), _candidate(6, 2, 2)]
    first = router.rank(frame=None, candidates=candidates)
    assert first[0] is candidates[0]
    # candidates[0]'s coordinate (1,1) was selected -> a fresh call offering the SAME
    # coordinate again as the top candidate must suppress it and fall through.
    second = router.rank(frame=None, candidates=[_candidate(6, 1, 1), _candidate(6, 3, 3)])
    assert (1, 1) not in [tuple(c["data"].values()) for c in second[:1]]


def test_rank_degrades_gracefully_when_every_candidate_is_a_repeat():
    completer = FakeCompleter([(True, "STRATEGY: a\nCHOICE: 0\n")] * 10)
    router = SGECandidateRouter(
        proposer=LLMStrategyProposer(completer=completer), game_id="g1", k=1, temperatures=(0.5,)
    )
    router.rank(frame=None, candidates=[_candidate(6, 1, 1)])
    # every candidate offered next call repeats the only coordinate seen so far
    second = router.rank(frame=None, candidates=[_candidate(6, 1, 1)])
    assert len(second) == 1  # falls back to the unsuppressed order rather than returning nothing


def test_rank_ignores_out_of_range_vote_indices():
    completer = FakeCompleter([(True, "STRATEGY: a\nCHOICE: 99\n")])
    router = SGECandidateRouter(
        proposer=LLMStrategyProposer(completer=completer), game_id="g1", k=1, temperatures=(0.5,)
    )
    candidates = [_candidate(6, 0, 0, salience_score=5.0), _candidate(6, 1, 1, salience_score=1.0)]
    ranked = router.rank(frame=None, candidates=candidates)
    # the out-of-range vote is discarded -> no votes recorded -> falls back to
    # deterministic scoring, so the higher-salience candidate should lead.
    assert router.last_diagnostics["votes_by_index"] == {}
    assert ranked[0] is candidates[0]


def test_rank_triggers_reflection_on_schedule():
    # 4 propose calls per rank() (k=1 propose + no reflect until step 2), reflect_every=2
    completer = FakeCompleter(
        [
            (True, "STRATEGY: a\nCHOICE: 0\n"),  # step 1 propose
            (True, "STRATEGY: b\nCHOICE: 0\n"),  # step 2 propose
            (True, "REVISED_STRATEGY: focus elsewhere\n"),  # step 2 reflect
        ]
    )
    router = SGECandidateRouter(
        proposer=LLMStrategyProposer(completer=completer),
        game_id="g1",
        k=1,
        temperatures=(0.5,),
        reflect_every=2,
    )
    candidates = [_candidate(6, 0, 0), _candidate(6, 1, 1)]
    router.rank(frame=None, candidates=candidates)
    assert router.last_diagnostics["reflected_this_call"] is False
    router.rank(frame=None, candidates=[_candidate(6, 2, 2), _candidate(6, 3, 3)])
    assert router.last_diagnostics["reflected_this_call"] is True
    assert router._reflection_note == "focus elsewhere"
    assert "focus elsewhere" in router._context()


def test_rank_reflection_nudge_fires_after_null_outcome():
    """SGECandidateRouter.rank() feeds AntiStagnationDiversityController.taboo_set() (recent
    null-outcome strategies) into the reflect() call, so the LLM's reflection prompt names
    exactly what NOT to repeat -- not just a generic 'what would you try differently'."""
    completer = FakeCompleter(
        [
            (True, "STRATEGY: wait quietly\nCHOICE: 0\n"),  # step 1 propose
            (True, "STRATEGY: wait quietly again\nCHOICE: 0\n"),  # step 2 propose
            (True, "REVISED_STRATEGY: try clicking instead\n"),  # step 2 reflect
        ]
    )
    router = SGECandidateRouter(
        proposer=LLMStrategyProposer(completer=completer),
        game_id="g1",
        k=1,
        temperatures=(0.5,),
        reflect_every=2,
    )
    candidates = [_candidate(6, 0, 0), _candidate(6, 1, 1)]
    router.rank(frame=None, candidates=candidates)
    router.record_outcome("no_change")  # step 1's chosen strategy led nowhere
    router.rank(frame=None, candidates=[_candidate(6, 2, 2), _candidate(6, 3, 3)])
    assert router.last_diagnostics["reflected_this_call"] is True
    assert router.last_diagnostics["reflection_nudge_fired"] is True
    reflect_prompt = completer.calls[-1]["prompt"]
    assert "ANTI-STAGNATION WARNING" in reflect_prompt
    assert "wait quietly" in reflect_prompt


def test_rank_reflects_early_when_soft_signal_precedes_schedule():
    """REQ-ARC-FCP-5699-4: a null outcome triggers reflect() on the VERY NEXT call even
    when reflect_every is far from its own boundary (here reflect_every=6, but the
    stagnation signal appears at step 2) -- catching stagnation before the harder
    collapse gate (checked every call too) can pre-empt the whole propose/reflect path
    for the rest of the game, the exact 2026-07-15 real-GPU failure mode."""
    completer = FakeCompleter(
        [
            (True, "STRATEGY: wait quietly\nCHOICE: 0\n"),  # step 1 propose
            (True, "STRATEGY: wait quietly again\nCHOICE: 0\n"),  # step 2 propose
            (True, "REVISED_STRATEGY: try clicking instead\n"),  # step 2 EARLY reflect
        ]
    )
    router = SGECandidateRouter(
        proposer=LLMStrategyProposer(completer=completer),
        game_id="g1",
        k=1,
        temperatures=(0.5,),
        reflect_every=6,  # would not schedule until step 6 on its own
    )
    router.rank(frame=None, candidates=[_candidate(6, 0, 0), _candidate(6, 1, 1)])
    assert router.last_diagnostics["reflected_this_call"] is False
    router.record_outcome("no_change")  # step 1's chosen strategy led nowhere
    router.rank(frame=None, candidates=[_candidate(6, 2, 2), _candidate(6, 3, 3)])
    assert router.last_diagnostics["reflected_this_call"] is True
    assert router.last_diagnostics["reflection_trigger"] == "early_stagnation_signal"
    assert "ANTI-STAGNATION WARNING" in completer.calls[-1]["prompt"]


def test_rank_scheduled_reflection_trigger_label_without_stagnation():
    """A healthy run (no null outcomes) still reflects on the periodic schedule, labeled
    'scheduled' -- the early-trigger path does not change the pre-existing cadence when
    nothing is stagnating."""
    completer = FakeCompleter(
        [
            (True, "STRATEGY: a\nCHOICE: 0\n"),
            (True, "STRATEGY: b\nCHOICE: 0\n"),
            (True, "REVISED_STRATEGY: focus elsewhere\n"),
        ]
    )
    router = SGECandidateRouter(
        proposer=LLMStrategyProposer(completer=completer),
        game_id="g1",
        k=1,
        temperatures=(0.5,),
        reflect_every=2,
    )
    router.rank(frame=None, candidates=[_candidate(6, 0, 0), _candidate(6, 1, 1)])
    router.record_outcome("level_advanced")  # NOT a null outcome
    router.rank(frame=None, candidates=[_candidate(6, 2, 2), _candidate(6, 3, 3)])
    assert router.last_diagnostics["reflected_this_call"] is True
    assert router.last_diagnostics["reflection_trigger"] == "scheduled"
    assert "ANTI-STAGNATION" not in completer.calls[-1]["prompt"]


def test_rank_no_reflection_when_neither_scheduled_nor_stagnating():
    """A run with no null outcomes at all, and reflect_every far off, does not trigger
    reflect() at all -- the early-trigger path is not a blanket 'reflect every call'
    change; it only fires on a genuine stagnation signal (a null outcome populates the
    taboo set immediately, so this test must avoid null outcomes entirely, not just
    stay below the streak threshold)."""
    completer = FakeCompleter([(True, "STRATEGY: a\nCHOICE: 0\n")] * 3)
    router = SGECandidateRouter(
        proposer=LLMStrategyProposer(completer=completer),
        game_id="g1",
        k=1,
        temperatures=(0.5,),
        reflect_every=6,
    )
    router.rank(frame=None, candidates=[_candidate(6, 0, 0), _candidate(6, 1, 1)])
    router.record_outcome("level_advanced")  # NOT a null outcome
    router.rank(frame=None, candidates=[_candidate(6, 2, 2), _candidate(6, 3, 3)])
    assert router.last_diagnostics["reflected_this_call"] is False
    assert router.last_diagnostics["reflection_trigger"] is None


def test_rank_no_reflection_nudge_without_anti_stagnation_controller():
    """A router configured with anti_stagnation_controller=None still reflects on schedule
    (backward-compatible), just never nudges -- matching reflect()'s own default of an
    empty taboo_strategies tuple when no controller-derived signal is available."""
    completer = FakeCompleter(
        [
            (True, "STRATEGY: wait quietly\nCHOICE: 0\n"),
            (True, "STRATEGY: wait quietly again\nCHOICE: 0\n"),
            (True, "REVISED_STRATEGY: keep going\n"),
        ]
    )
    router = SGECandidateRouter(
        proposer=LLMStrategyProposer(completer=completer),
        game_id="g1",
        k=1,
        temperatures=(0.5,),
        reflect_every=2,
        anti_stagnation_controller=None,
    )
    router.rank(frame=None, candidates=[_candidate(6, 0, 0), _candidate(6, 1, 1)])
    router.record_outcome("no_change")
    router.rank(frame=None, candidates=[_candidate(6, 2, 2), _candidate(6, 3, 3)])
    assert router.last_diagnostics["reflected_this_call"] is True
    assert router.last_diagnostics["reflection_nudge_fired"] is False
    assert "ANTI-STAGNATION" not in completer.calls[-1]["prompt"]


def test_record_outcome_updates_last_history_row():
    completer = FakeCompleter([(True, "STRATEGY: a\nCHOICE: 0\n")])
    router = SGECandidateRouter(
        proposer=LLMStrategyProposer(completer=completer), game_id="g1", k=1, temperatures=(0.5,)
    )
    router.rank(frame=None, candidates=[_candidate(6, 0, 0)])
    router.record_outcome("level_advanced")
    assert router.history[-1]["outcome"] == "level_advanced"


def test_record_outcome_no_op_with_empty_history():
    completer = FakeCompleter([])
    router = SGECandidateRouter(proposer=LLMStrategyProposer(completer=completer), game_id="g1")
    router.record_outcome("anything")  # must not raise
    assert router.history == []


def test_portfolio_descriptors_shape():
    router = SGECandidateRouter(
        proposer=LLMStrategyProposer(completer=FakeCompleter([])), game_id="g1", k=4
    )
    descriptors = router.portfolio_descriptors()
    assert len(descriptors) == 1
    assert descriptors[0]["k"] == 4
    assert descriptors[0]["live_path_hook"] == "candidate_router.rank"


def test_context_without_reflection_note():
    router = SGECandidateRouter(
        proposer=LLMStrategyProposer(completer=FakeCompleter([])), game_id="g42"
    )
    assert router._context() == "Game: g42"


# ---------------------------------------------------------------------------
# REQ-ARC-FCP-5575 / SCENARIO-ARC-FCP-5575 anti-stagnation controller
# ---------------------------------------------------------------------------


def _seed_collapsed_wait_history(router: SGECandidateRouter, n: int = 4) -> None:
    for step in range(1, n + 1):
        router.history.append(
            {
                "step": step,
                "strategy_text": "Observe the initial state and wait for automatic changes.",
                "chosen_signature": "A1#0",
                "outcome": "no_change",
            }
        )


def test_anti_stagnation_activation_forces_diverse_portfolio_without_llm_call():
    # REQ-ARC-FCP-5575 / SCENARIO-ARC-FCP-5575: collapsed SGE history must trigger
    # a deterministic live-path portfolio before spending another LLM completion.
    completer = FakeCompleter([(True, "STRATEGY: should not be called\nCHOICE: 0\n")])
    router = SGECandidateRouter(
        proposer=LLMStrategyProposer(completer=completer),
        game_id="g1",
        k=3,
        temperatures=(0.3, 0.6, 0.9),
    )
    _seed_collapsed_wait_history(router)
    candidates = [
        _candidate(1, 0, 0),
        _candidate(6, 5, 5, salience_score=3.0),
        _candidate(2, 0, 0),
        _candidate(6, 9, 2, verifier_score=4.0),
        _candidate(5, 0, 0, reset_score=5.0),
    ]

    ranked = router.rank(frame=None, candidates=candidates)

    anti = router.last_diagnostics["anti_stagnation"]
    assert anti["collapse_detected"] is True
    assert completer.calls == []
    assert {row["name"] for row in anti["forced_portfolio_selected"]} >= {
        "observation",
        "active_coordinate_probe",
        "action_type_probe",
        "mechanic_falsification",
        "recovery_reset",
    }
    assert ranked


def test_anti_stagnation_reports_diversity_increase_after_forced_portfolio():
    # REQ-ARC-FCP-5575: the precheck needs before/after diversity metrics proving
    # the forced portfolio increases strategy/action variety over the collapsed trace.
    router = SGECandidateRouter(
        proposer=LLMStrategyProposer(completer=FakeCompleter([(False, "unused")])),
        game_id="g1",
        k=1,
        temperatures=(0.5,),
    )
    _seed_collapsed_wait_history(router)
    candidates = [
        _candidate(1, 0, 0),
        _candidate(2, 0, 0),
        _candidate(3, 0, 0),
        _candidate(5, 0, 0, reset_score=2.0),
        _candidate(6, 4, 4, effect_score=3.0),
        _candidate(6, 8, 9, verifier_score=4.0),
    ]

    router.rank(frame=None, candidates=candidates)
    metrics = router.last_diagnostics["anti_stagnation"]["diversity_metrics_before_after"]

    assert metrics["before"]["unique_normalized_strategy_count"] == 1
    assert metrics["before"]["max_normalized_strategy_repeat"] >= 4
    assert metrics["after"]["forced_portfolio_category_count"] >= 5
    assert (
        metrics["after"]["selected_unique_signature_count"]
        > metrics["before"]["unique_action_signature_count"]
    )


def test_anti_stagnation_stable_fallback_when_portfolio_cannot_fill():
    # REQ-ARC-FCP-5575: collapse handling must remain stable on tiny candidate
    # pools and malformed/failed proposer output.
    completer = FakeCompleter([(False, "GPU unavailable")])
    router = SGECandidateRouter(
        proposer=LLMStrategyProposer(completer=completer),
        game_id="g1",
        k=1,
        temperatures=(0.5,),
    )
    _seed_collapsed_wait_history(router)
    candidates = [_candidate(6, 1, 1, salience_score=1.0)]

    ranked = router.rank(frame=None, candidates=candidates)

    assert ranked == candidates
    assert router.last_diagnostics["anti_stagnation"]["collapse_detected"] is True
    assert router.last_diagnostics["anti_stagnation"]["stable_fallback_used"] is True
    assert completer.calls == []


def test_anti_stagnation_prompts_do_not_leak_win_or_level_checks():
    # REQ-ARC-FCP-5575: normal SGE prompts and reflection context must not expose
    # oracle/win/level/source/scorecard signals to ranking.
    completer = FakeCompleter(
        [
            (True, "STRATEGY: probe a visible object\nCHOICE: 0\n"),
            (True, "REVISED_STRATEGY: try a different visible object\n"),
        ]
    )
    router = SGECandidateRouter(
        proposer=LLMStrategyProposer(completer=completer),
        game_id="g1",
        k=1,
        temperatures=(0.5,),
        reflect_every=1,
    )

    router.rank(frame=None, candidates=[_candidate(6, 1, 1)])
    prompts = "\n".join(call["prompt"].lower() for call in completer.calls)

    for forbidden in ("win", "level", "oracle", "scorecard", "source"):
        assert forbidden not in prompts
    assert router.last_diagnostics["win_check_used_for_ranking"] is False


def test_anti_stagnation_controller_is_reachable_from_e3_import_path():
    # REQ-ARC-FCP-5575 / SCENARIO-ARC-FCP-5575: this must be the router object
    # consumed by E3AgentPolicy, not a standalone experiment-only path.
    router = SGECandidateRouter(
        proposer=LLMStrategyProposer(completer=FakeCompleter([(False, "unused")])),
        game_id="g1",
        anti_stagnation_controller=AntiStagnationDiversityController(),
    )
    repo = Path(__file__).resolve().parents[2]
    competition_agent = (repo / "python/carnot/agentic/arc_competition_agent.py").read_text()
    graph_explore = (repo / "python/carnot/agentic/arc_graph_explore.py").read_text()

    assert "candidate_router: Any = _DEFAULT_CANDIDATE_ROUTER" in competition_agent
    assert "candidate_router=candidate_router" in competition_agent
    assert "candidate_router.rank(frame, out, previous_frame=previous_frame)" in graph_explore
    assert isinstance(router.anti_stagnation_controller, AntiStagnationDiversityController)


def test_anti_stagnation_helper_edges_for_fixed_collapse_definition():
    # REQ-ARC-FCP-5575: edge cases in the fixed collapse signals must be explicit
    # rather than depending on incidental parser behavior.
    assert _strategy_distance("", "") == 0.0
    assert _strategy_distance("observe", "") == 1.0
    assert _outcome_is_null({"level_before": 1, "level_after": 1}) is True
    assert _outcome_is_null({"level_before": "bad", "level_after": 1, "changed": False}) is True
    assert _outcome_is_null({"effect": "none"}) is True

    class WeirdCandidate:
        action = "not-an-int"

    assert _candidate_action(WeirdCandidate()) == 0
    assert _candidate_signature({"action": 6, "data": {"x": "bad", "y": 1}}, 3) == "A6#3"
    assert _candidate_signature({"action": 2, "data": {}}, 4) == "A2#4"
    assert (
        _fallback_score(
            {"salience_score": "not-a-float", "score": 2.0}, ("salience_score", "score")
        )
        == 2.0
    )


def test_anti_stagnation_forced_portfolio_avoids_recent_failed_signature():
    # REQ-ARC-FCP-5575: outcome-conditioned failed action signatures should not
    # be re-selected by the active coordinate probe when an alternative exists.
    router = SGECandidateRouter(
        proposer=LLMStrategyProposer(completer=FakeCompleter([(False, "unused")])),
        game_id="g1",
        k=1,
        temperatures=(0.5,),
    )
    _seed_collapsed_wait_history(router)
    router.history.append(
        {
            "step": 99,
            "strategy_text": "Observe the initial state and wait for automatic changes.",
            "chosen_signature": "A6@5,5",
            "outcome": "no_change",
        }
    )
    candidates = [
        _candidate(1, 0, 0),
        _candidate(6, 5, 5, salience_score=9.0),
        _candidate(6, 7, 7, salience_score=1.0),
        _candidate(2, 0, 0),
        _candidate(5, 0, 0),
    ]

    router.rank(frame=None, candidates=candidates)
    selected = router.last_diagnostics["anti_stagnation"]["forced_portfolio_selected"]
    active = [row for row in selected if row["name"] == "active_coordinate_probe"]
    assert active == [{"name": "active_coordinate_probe", "signature": "A6@7,7"}]


def test_anti_stagnation_forced_portfolio_rotates_across_repeated_calls():
    # REQ-ARC-FCP-5699-2 / SCENARIO-ARC-FCP-5699-2-FORCED-PORTFOLIO-ROTATES: a frozen
    # candidate pool (unchanging across calls, matching a stalled game state) must not
    # collapse the "observation" category onto the exact same signature every call --
    # exp5699's real live run measured the SAME 2-signature pair selected on 44
    # consecutive forced-portfolio calls once collapse fired, a partial-escape bug.
    controller = AntiStagnationDiversityController(
        thresholds=StrategyCollapseThresholds(window_size=8)
    )
    # A realistically-sized pool: `SGECandidateRouter.max_candidates` defaults to 8, and
    # `rank_forced_portfolio`'s own `fallback_fill` step tops up every call to that many
    # selections -- a pool sized close to max_candidates would exhaust globally within a
    # round or two purely from fallback_fill sweeping the remainder (not a bug; a separate
    # test below covers that exhaustion-then-fallback path explicitly). 20 candidates vs.
    # max_candidates=8 leaves real headroom to observe rotation across multiple rounds.
    candidates = [_candidate(1, i, i, score=float(20 - i)) for i in range(20)]
    history: list[dict] = []
    observation_signatures: list[str] = []

    for step in range(1, 4):
        forced = controller.rank_forced_portfolio(
            candidates,
            history=history,
            fallback_score_fields=("score",),
            max_candidates=8,
            seen_coordinates=set(),
        )
        selected = forced["forced_portfolio_selected"]
        observation = next(row for row in selected if row["name"] == "observation")
        observation_signatures.append(observation["signature"])
        history.append(
            {
                "step": step,
                "strategy_text": "anti_stagnation_forced:"
                + ",".join(row["name"] for row in selected),
                "chosen_signature": selected[0]["signature"],
                "forced_signatures": [row["signature"] for row in selected],
                "outcome": "pending",
            }
        )

    # A genuinely diverse pool must rotate: no two consecutive calls select the same
    # "observation" candidate, unlike the pre-fix behavior (the exact same signature
    # every single call, confirmed live on exp5699's real 44-step run).
    assert observation_signatures[0] != observation_signatures[1]
    assert observation_signatures[1] != observation_signatures[2]


def test_anti_stagnation_forced_portfolio_rotation_falls_back_when_pool_exhausted():
    # REQ-ARC-FCP-5699-2: once every candidate in a category's pool has been recently
    # forced, rotation must fall back to re-selecting rather than leaving the category
    # unfilled -- the pre-existing category-fill guarantee is not weakened by rotation.
    controller = AntiStagnationDiversityController(
        thresholds=StrategyCollapseThresholds(window_size=8)
    )
    candidates = [_candidate(1, 0, 0, score=3.0)]  # single-candidate pool, no alternative
    history = [
        {
            "step": 1,
            "strategy_text": "anti_stagnation_forced:observation",
            "chosen_signature": _candidate_signature(candidates[0], 0),
            "forced_signatures": [_candidate_signature(candidates[0], 0)],
            "outcome": "pending",
        }
    ]

    forced = controller.rank_forced_portfolio(
        candidates,
        history=history,
        fallback_score_fields=("score",),
        max_candidates=5,
        seen_coordinates=set(),
    )

    selected = forced["forced_portfolio_selected"]
    observation = next(row for row in selected if row["name"] == "observation")
    assert observation["signature"] == _candidate_signature(candidates[0], 0)
    assert "observation" in forced["rotation_exhausted_categories"]


def test_recently_forced_signatures_reads_plural_field():
    controller = AntiStagnationDiversityController()
    history = [
        {
            "strategy_text": "anti_stagnation_forced:observation,recovery_reset",
            "chosen_signature": "A1@0,0",
            "forced_signatures": ["A1@0,0", "A5@3,3"],
            "outcome": "pending",
        },
        {
            "strategy_text": "Observe and wait.",
            "chosen_signature": "A1@0,0",
            "outcome": "no_change",
        },
    ]
    assert controller._recently_forced_signatures(history) == {"A1@0,0", "A5@3,3"}


def test_recently_forced_signatures_falls_back_to_singular_field_legacy_row():
    controller = AntiStagnationDiversityController()
    # A forced-portfolio row recorded before `forced_signatures` existed (or by any other
    # caller that only sets the singular field) should still contribute its one known
    # signature rather than being silently ignored.
    history = [
        {
            "strategy_text": "anti_stagnation_forced:observation",
            "chosen_signature": "A2@1,1",
            "outcome": "pending",
        }
    ]
    assert controller._recently_forced_signatures(history) == {"A2@1,1"}


def test_anti_stagnation_forced_portfolio_fallback_fills_to_budget():
    # REQ-ARC-FCP-5575: if the five-category portfolio cannot be filled, the
    # deterministic fallback still returns a bounded stable ranking.
    controller = AntiStagnationDiversityController(
        thresholds=StrategyCollapseThresholds(window_size=4)
    )
    history = [
        {"strategy_text": "wait", "chosen_signature": "A1#0", "outcome": "no_change"}
        for _ in range(4)
    ]
    candidates = [
        {"action": 5, "data": {"x": 1, "y": 1}, "reset_score": 1.0},
        {"action": 5, "data": {"x": 2, "y": 2}, "reset_score": 0.5},
    ]

    forced = controller.rank_forced_portfolio(
        candidates,
        history=history,
        fallback_score_fields=("reset_score", "score"),
        max_candidates=2,
        seen_coordinates=set(),
    )

    assert len(forced["ranked"]) == 2
    assert forced["stable_fallback_used"] is True
    assert any(row["name"] == "fallback_fill" for row in forced["forced_portfolio_selected"])


def test_anti_stagnation_tabooed_proposals_do_not_vote():
    # REQ-ARC-FCP-5575: recently failed normalized strategies are tabooed before
    # vote aggregation, while the fallback order remains deterministic.
    completer = FakeCompleter([(True, "STRATEGY: repeat failed tactic\nCHOICE: 1\n")])
    router = SGECandidateRouter(
        proposer=LLMStrategyProposer(completer=completer),
        game_id="g1",
        k=1,
        temperatures=(0.5,),
    )
    router.history.append(
        {
            "step": 1,
            "strategy_text": "repeat failed tactic",
            "chosen_signature": "A1#0",
            "outcome": "no_change",
        }
    )
    candidates = [_candidate(1, 0, 0, salience_score=2.0), _candidate(2, 0, 0, salience_score=1.0)]

    ranked = router.rank(frame=None, candidates=candidates)

    assert ranked[0] is candidates[0]
    assert router.last_diagnostics["votes_by_index"] == {}
    assert router.last_diagnostics["anti_stagnation"]["tabooed_proposal_count"] == 1


def test_rank_respects_max_candidates_after_normal_vote():
    # REQ-ARC-FCP-5575 regression guard: bounded normal ranking still stops at
    # max_candidates after anti-stagnation instrumentation is added.
    completer = FakeCompleter([(True, "STRATEGY: probe visible object\nCHOICE: 0\n")])
    router = SGECandidateRouter(
        proposer=LLMStrategyProposer(completer=completer),
        game_id="g1",
        k=1,
        temperatures=(0.5,),
        max_candidates=1,
    )
    candidates = [_candidate(6, 1, 1), _candidate(6, 2, 2)]

    ranked = router.rank(frame=None, candidates=candidates)

    assert ranked == [candidates[0]]


def test_req_arc_fcp_5699_2_spec_declares_rotation_fix() -> None:
    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-FCP-5699-2") :]

    for marker in (
        "REQ-ARC-FCP-5699-2",
        "SCENARIO-ARC-FCP-5699-2-FORCED-PORTFOLIO-ROTATES",
        "_recently_forced_signatures",
        "rotation_exhausted_categories",
    ):
        assert marker in section


def test_req_arc_fcp_5699_3_spec_declares_reflect_prompt_nudge() -> None:
    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-FCP-5699-3") : spec.index("### REQ-ARC-FCP-5699-4")]

    for marker in (
        "REQ-ARC-FCP-5699-3",
        "SCENARIO-ARC-FCP-5699-3-REFLECT-PROMPT-NAMES-THE-STAGNATION",
        "_REFLECT_NUDGE_NULL_STREAK",
        "taboo_strategies",
        "nudge_fired",
    ):
        assert marker in section


def test_req_arc_fcp_5699_4_spec_declares_early_trigger() -> None:
    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-FCP-5699-4") : spec.index("### REQ-ARC-WMTE-5596")]

    for marker in (
        "REQ-ARC-FCP-5699-4",
        "SCENARIO-ARC-FCP-5699-4-EARLY-TRIGGER-RACES-HARD-COLLAPSE",
        "reflection_trigger",
        "early_stagnation_signal",
        "nudge_would_fire_early",
    ):
        assert marker in section
