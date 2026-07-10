"""Tests for the LLM Strategy-Guided Exploration (SGE) candidate router.

Outer-loop investigation, 2026-07-10: verifies the mechanism arXiv:2603.02045 describes
(mixed-temperature parallel strategy sampling + reflection) with a fake, deterministic
`TextCompleter` -- no GPU/model required for this suite. A separate manual smoke script
exercises the real `LocalGGUFProposer` path against the offline ARC arcade.
"""

from __future__ import annotations

from carnot.agentic.arc_llm_strategy_proposer import (
    LLMStrategyProposer,
    SGECandidateRouter,
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
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature, "stop": stop})
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
    assert parse_reflect_reply("REVISED_STRATEGY: try the left panel next\n") == "try the left panel next"


def test_parse_reflect_reply_malformed():
    assert parse_reflect_reply("no marker here") == ""


# ---------------------------------------------------------------------------
# LLMStrategyProposer
# ---------------------------------------------------------------------------


def test_propose_one_parses_completer_output():
    completer = FakeCompleter([(True, "STRATEGY: try salient blob\nCHOICE: 1\n")])
    proposer = LLMStrategyProposer(completer=completer)
    result = proposer.propose_one("Game: g1", ["[0] action=6 x=1,y=1", "[1] action=6 x=2,y=2"], temperature=0.5)
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
    assert result == {"parse_ok": False, "revised_strategy": "", "raw": "GPU down", "completer_ok": False}


# ---------------------------------------------------------------------------
# SGECandidateRouter
# ---------------------------------------------------------------------------


def test_rank_empty_candidates_returns_empty_and_honest_diagnostics():
    router = SGECandidateRouter(proposer=LLMStrategyProposer(completer=FakeCompleter([])), game_id="g1")
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
    router = SGECandidateRouter(proposer=LLMStrategyProposer(completer=completer), game_id="g1", k=3)
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
    router = SGECandidateRouter(proposer=LLMStrategyProposer(completer=completer), game_id="g1", k=3)
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
    router = SGECandidateRouter(proposer=LLMStrategyProposer(completer=completer), game_id="g1", k=3)
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
    router = SGECandidateRouter(proposer=LLMStrategyProposer(completer=completer), game_id="g1", k=1, temperatures=(0.5,))
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


def test_record_outcome_updates_last_history_row():
    completer = FakeCompleter([(True, "STRATEGY: a\nCHOICE: 0\n")])
    router = SGECandidateRouter(proposer=LLMStrategyProposer(completer=completer), game_id="g1", k=1, temperatures=(0.5,))
    router.rank(frame=None, candidates=[_candidate(6, 0, 0)])
    router.record_outcome("level_advanced")
    assert router.history[-1]["outcome"] == "level_advanced"


def test_record_outcome_no_op_with_empty_history():
    completer = FakeCompleter([])
    router = SGECandidateRouter(proposer=LLMStrategyProposer(completer=completer), game_id="g1")
    router.record_outcome("anything")  # must not raise
    assert router.history == []


def test_portfolio_descriptors_shape():
    router = SGECandidateRouter(proposer=LLMStrategyProposer(completer=FakeCompleter([])), game_id="g1", k=4)
    descriptors = router.portfolio_descriptors()
    assert len(descriptors) == 1
    assert descriptors[0]["k"] == 4
    assert descriptors[0]["live_path_hook"] == "candidate_router.rank"


def test_context_without_reflection_note():
    router = SGECandidateRouter(proposer=LLMStrategyProposer(completer=FakeCompleter([])), game_id="g42")
    assert router._context() == "Game: g42"
