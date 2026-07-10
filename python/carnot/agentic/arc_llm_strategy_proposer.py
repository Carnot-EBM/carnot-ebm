"""LLM Strategy-Guided Exploration (SGE) for the ARC-AGI-3 live candidate-router hook.

Implements the mechanism described in arXiv:2603.02045 ("Strategy-Guided Exploration"):
an LLM states a concise natural-language exploration STRATEGY, sampled at several
temperatures in parallel so multiple qualitatively different strategies compete for the
action budget, refined by an outcome-grounded reflection step that revises strategy from
what actually happened. This is genuinely distinct from every other candidate-generation
mechanism already tried on this project's ARC-AGI-3 wall (see the 2026-07-06
`ops/known-issues.md` ARC entry, task 6) -- it is not a tool-use loop, not a subgoal
value-head, not a novelty bonus, and not the deterministic-score-field "strategy
portfolio" in `arc_bounded_strategy_router.py` (which never invokes an LLM at all; that
router's own artifacts honestly declare `llm_strategy_proposer_used=false`).

Why this exists (root-cause context, 2026-07-10 outer-loop investigation): the conductor's
first "strategy-routed" attempt (`experiment_5534_arc_strategy_routed_levelup.py`) reused
`BoundedStrategyCandidateRouter`'s four hand-coded deterministic scoring templates under a
"strategy" label without ever loading a model. This module is the real thing.

Adapted for Carnot's `candidate_router.rank(frame, candidates, previous_frame=...)` hook:
rather than SGE's own free-form action-generation loop, each sampled strategy VOTES for
one candidate from the perception layer's already-generated list (a natural-language-
grounded choice, not a hand-coded score field). Votes across the K parallel samples are
combined into an `llm_strategy_score` the router ranks by. This targets the exact failure
mode the project's own TRM-generator v4 pilot diagnosed: a single deterministic guess is
wrong when the true "right answer" is a whole basin of plausible choices depending on
hidden intent; K diverse LLM-grounded strategies sampling that basin, rather than one
fixed heuristic, is the fix being tested here on the GENERATION side specifically.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


class TextCompleter(Protocol):
    """Structural type for anything providing LocalGGUFProposer's `complete_text` shape."""

    def complete_text(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float = 0.1,
        stop: list | None = None,
    ) -> tuple[bool, str]: ...


_STRATEGY_RE = re.compile(r"STRATEGY:\s*(.+?)(?:\n|$)", re.IGNORECASE)
_CHOICE_RE = re.compile(r"CHOICE:\s*(-?\d+)", re.IGNORECASE)

_PROPOSE_INSTRUCTIONS = (
    "You are exploring an unfamiliar interactive game one action at a time. You do not "
    "know the rules yet; you must infer them from what happens when you act. Below is the "
    "recent context and a numbered list of candidate next actions.\n\n"
    "State ONE short exploration strategy (a single sentence: what you are trying to learn "
    "or achieve right now and why), then pick the candidate number that best fits that "
    "strategy. Reply in EXACTLY this format, nothing else:\n"
    "STRATEGY: <one short sentence>\n"
    "CHOICE: <candidate number>\n\n"
)

_REFLECT_INSTRUCTIONS = (
    "You are exploring an unfamiliar interactive game. Below are your recent exploration "
    "strategies and what happened after each. State ONE short sentence on what to try "
    "differently next, grounded in what actually happened (not a generic restatement). "
    "Reply in EXACTLY this format, nothing else:\n"
    "REVISED_STRATEGY: <one short sentence>\n\n"
)


def _format_candidate_lines(candidates: Sequence[Any]) -> list[str]:
    lines: list[str] = []
    for index, candidate in enumerate(candidates):
        data = candidate.get("data") if isinstance(candidate, Mapping) else getattr(candidate, "data", None)
        data = data if isinstance(data, Mapping) else {}
        action = candidate.get("action") if isinstance(candidate, Mapping) else getattr(candidate, "action", None)
        coord = f"x={data['x']},y={data['y']}" if "x" in data and "y" in data else "no-coordinate"
        hints = []
        for hint_field in ("salience_score", "effect_score", "verifier_score", "score"):
            value = candidate.get(hint_field) if isinstance(candidate, Mapping) else getattr(candidate, hint_field, None)
            if value is not None:
                hints.append(f"{hint_field}={float(value):.2f}")
        hint_text = f" ({', '.join(hints)})" if hints else ""
        lines.append(f"[{index}] action={action} {coord}{hint_text}")
    return lines


def parse_propose_reply(text: str) -> dict[str, Any]:
    """Parse a `STRATEGY: ...\\nCHOICE: <int>` reply. Never fabricates a choice on
    malformed output -- returns `parse_ok=False` and `chosen_index=None` instead, per
    this project's no-silent-degradation discipline (callers must fall back honestly,
    not guess)."""

    strategy_match = _STRATEGY_RE.search(text)
    choice_match = _CHOICE_RE.search(text)
    strategy_text = strategy_match.group(1).strip() if strategy_match else ""
    if choice_match is None or not strategy_text:
        return {"parse_ok": False, "strategy_text": strategy_text, "chosen_index": None, "raw": text}
    try:
        chosen_index = int(choice_match.group(1))
    except ValueError:
        return {"parse_ok": False, "strategy_text": strategy_text, "chosen_index": None, "raw": text}
    return {"parse_ok": True, "strategy_text": strategy_text, "chosen_index": chosen_index, "raw": text}


def parse_reflect_reply(text: str) -> str:
    match = re.search(r"REVISED_STRATEGY:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    return match.group(1).strip() if match else ""


@dataclass
class LLMStrategyProposer:
    """Wraps a `TextCompleter` (normally `LocalGGUFProposer`) with the SGE propose/reflect
    prompt contracts. Kept separate from the router so it is independently unit-testable
    with a fake completer, and independently reusable outside the candidate_router hook."""

    completer: TextCompleter
    max_tokens: int = 96

    def propose_one(self, context: str, candidate_lines: Sequence[str], *, temperature: float) -> dict[str, Any]:
        prompt = _PROPOSE_INSTRUCTIONS + context + "\n\nCandidates:\n" + "\n".join(candidate_lines) + "\n\n"
        ok, text = self.completer.complete_text(
            prompt, max_tokens=self.max_tokens, temperature=temperature, stop=["\n\n"]
        )
        if not ok:
            return {
                "parse_ok": False,
                "strategy_text": "",
                "chosen_index": None,
                "raw": text,
                "temperature": temperature,
                "completer_ok": False,
            }
        parsed = parse_propose_reply(text)
        parsed["temperature"] = temperature
        parsed["completer_ok"] = True
        return parsed

    def propose_many(
        self, context: str, candidate_lines: Sequence[str], *, temperatures: Sequence[float]
    ) -> list[dict[str, Any]]:
        return [self.propose_one(context, candidate_lines, temperature=t) for t in temperatures]

    def reflect(self, context: str, history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not history:
            return {"parse_ok": False, "revised_strategy": "", "raw": ""}
        history_lines = [
            f"- strategy: \"{row.get('strategy_text', '')}\" -> outcome: {row.get('outcome', 'unknown')}"
            for row in history
        ]
        prompt = _REFLECT_INSTRUCTIONS + context + "\n\nRecent attempts:\n" + "\n".join(history_lines) + "\n\n"
        ok, text = self.completer.complete_text(prompt, max_tokens=self.max_tokens, temperature=0.2, stop=["\n\n"])
        if not ok:
            return {"parse_ok": False, "revised_strategy": "", "raw": text, "completer_ok": False}
        revised = parse_reflect_reply(text)
        return {"parse_ok": bool(revised), "revised_strategy": revised, "raw": text, "completer_ok": True}


def _candidate_coordinate(candidate: Any) -> tuple[int, int] | None:
    data = candidate.get("data") if isinstance(candidate, Mapping) else getattr(candidate, "data", None)
    data = data if isinstance(data, Mapping) else {}
    if "x" in data and "y" in data:
        try:
            return int(data["x"]), int(data["y"])
        except (TypeError, ValueError):
            return None
    return None


def _candidate_signature(candidate: Any, index: int) -> str:
    coord = _candidate_coordinate(candidate)
    action = candidate.get("action") if isinstance(candidate, Mapping) else getattr(candidate, "action", None)
    if coord is not None:
        return f"A{action}@{coord[0]},{coord[1]}"
    return f"A{action}#{index}"


def _fallback_score(candidate: Any, fields: Sequence[str]) -> float:
    best = 0.0
    for field_name in fields:
        value = candidate.get(field_name) if isinstance(candidate, Mapping) else getattr(candidate, field_name, None)
        try:
            best = max(best, float(value or 0.0))
        except (TypeError, ValueError):
            continue
    return best


@dataclass
class SGECandidateRouter:
    """Live-compatible candidate router implementing PTRM/SGE-style stochastic
    multi-strategy exploration. Matches `candidate_router.rank(frame, candidates,
    previous_frame=...)` exactly, so it is a drop-in replacement for
    `BoundedStrategyCandidateRouter` in `E3AgentPolicy(candidate_router=...)`.

    Each `rank()` call samples K natural-language strategies (one LLM completion per
    temperature) that each vote for one candidate; votes become `llm_strategy_score`.
    Every `reflect_every` steps, a reflection call revises the strategy context from
    recent (strategy, outcome) history. Falls back honestly to `fallback_score_fields`
    (the existing deterministic scores) when the LLM proposer is unavailable or every
    sample fails to parse -- never fabricates a choice.
    """

    proposer: LLMStrategyProposer
    game_id: str = "unknown_game"
    k: int = 3
    temperatures: tuple[float, ...] = (0.3, 0.6, 0.9)
    max_candidates: int = 8
    reflect_every: int = 6
    fallback_score_fields: tuple[str, ...] = (
        "salience_score",
        "effect_score",
        "verifier_score",
        "reset_score",
        "score",
    )
    suppress_repeated_coordinates: bool = True

    history: list[dict[str, Any]] = field(default_factory=list, init=False)
    last_diagnostics: dict[str, Any] = field(default_factory=dict, init=False)
    _reflection_note: str = field(default="", init=False)
    _step: int = field(default=0, init=False)
    _seen_coordinates: set[tuple[int, int]] = field(default_factory=set, init=False)

    def _context(self) -> str:
        note = f"Current guidance from reflection: {self._reflection_note}\n" if self._reflection_note else ""
        return f"Game: {self.game_id}\n{note}".strip()

    def rank(self, frame: Any, candidates: Sequence[Any], *, previous_frame: Any | None = None) -> list[Any]:
        del frame, previous_frame
        self._step += 1
        ordered = list(candidates)
        if not ordered:
            self.last_diagnostics = {
                "llm_strategy_proposer_used": False,
                "reason": "no_candidates",
                "strategy_texts": [],
                "votes_by_index": {},
                "reflection_note": self._reflection_note,
            }
            return []

        temperatures = tuple(self.temperatures[: self.k]) or (0.5,)
        candidate_lines = _format_candidate_lines(ordered)
        proposals = self.proposer.propose_many(self._context(), candidate_lines, temperatures=temperatures)

        votes: dict[int, int] = {}
        strategy_texts: list[str] = []
        any_completer_ok = False
        for proposal in proposals:
            if proposal.get("completer_ok"):
                any_completer_ok = True
            if not proposal.get("parse_ok"):
                continue
            index = proposal["chosen_index"]
            if isinstance(index, int) and 0 <= index < len(ordered):
                votes[index] = votes.get(index, 0) + 1
                strategy_texts.append(proposal["strategy_text"])

        llm_used = any_completer_ok and bool(votes)

        def sort_key(pair: tuple[int, Any]) -> tuple[float, float, str]:
            index, candidate = pair
            return (
                -float(votes.get(index, 0)),
                -_fallback_score(candidate, self.fallback_score_fields),
                _candidate_signature(candidate, index),
            )

        ranked_pairs = sorted(enumerate(ordered), key=sort_key)

        selected: list[Any] = []
        seen_this_call: set[tuple[int, int]] = set()
        suppressed = 0
        for index, candidate in ranked_pairs:
            coord = _candidate_coordinate(candidate)
            if self.suppress_repeated_coordinates and coord is not None:
                if coord in self._seen_coordinates or coord in seen_this_call:
                    suppressed += 1
                    continue
            selected.append(candidate)
            if coord is not None:
                seen_this_call.add(coord)
            if len(selected) >= self.max_candidates:
                break
        if not selected:
            # every candidate was a repeat coordinate -- degrade to the unsuppressed
            # order rather than returning nothing, matching BoundedStrategyCandidateRouter.
            selected = [candidate for _, candidate in ranked_pairs[: self.max_candidates]]
        self._seen_coordinates |= seen_this_call

        top_index, _ = ranked_pairs[0]
        chosen_strategy = next((p["strategy_text"] for p in proposals if p.get("chosen_index") == top_index and p.get("parse_ok")), "")
        self.history.append(
            {
                "step": self._step,
                "strategy_text": chosen_strategy,
                "chosen_signature": _candidate_signature(ordered[top_index], top_index),
                "votes": dict(votes),
                "outcome": "pending",
            }
        )

        reflected = False
        if llm_used and self.reflect_every > 0 and self._step % self.reflect_every == 0:
            reflection = self.proposer.reflect(self._context(), self.history[-self.reflect_every :])
            if reflection.get("parse_ok"):
                self._reflection_note = reflection["revised_strategy"]
            reflected = True

        self.last_diagnostics = {
            "llm_strategy_proposer_used": bool(llm_used),
            "strategy_texts": strategy_texts,
            "votes_by_index": {str(k): v for k, v in votes.items()},
            "parse_failure_count": sum(1 for p in proposals if not p.get("parse_ok")),
            "completer_failure_count": sum(1 for p in proposals if not p.get("completer_ok")),
            "reflection_note": self._reflection_note,
            "reflected_this_call": reflected,
            "suppressed_coordinate_count": suppressed,
            "step": self._step,
        }
        return selected

    def record_outcome(self, outcome: str) -> None:
        """Caller-driven: attach the real outcome of the last-ranked step to history,
        so the next reflection call is grounded in what actually happened rather than
        the placeholder "pending". Optional -- reflection still runs without it, just
        with less-informative context."""

        if self.history:
            self.history[-1]["outcome"] = outcome

    def portfolio_descriptors(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "llm_strategy_guided_exploration",
                "mechanism": "arXiv:2603.02045 SGE, mixed-temperature parallel strategy sampling + reflection",
                "k": self.k,
                "temperatures": list(self.temperatures),
                "live_path_hook": "candidate_router.rank",
            }
        ]
