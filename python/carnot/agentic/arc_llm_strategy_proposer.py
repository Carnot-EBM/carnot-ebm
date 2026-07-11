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
from collections import Counter
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
_CHOICE_RE = re.compile(r"CHOICE:\s*([^\n]+)", re.IGNORECASE)

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


@dataclass(frozen=True)
class StrategyCollapseThresholds:
    """Fixed anti-stagnation thresholds for REQ-ARC-FCP-5575.

    These values are intentionally small and explicit because they gate live LLM
    spend: once recent history already proves repeated passive behavior, the
    router should switch to a deterministic diversity portfolio before asking the
    model for another version of the same strategy.
    """

    window_size: int = 8
    repeated_normalized_strategy_count: int = 3
    max_mean_pairwise_strategy_distance: float = 0.35
    repeated_action_proposal_count: int = 4
    consecutive_null_outcomes: int = 4
    min_triggered_signals: int = 3

    def as_dict(self) -> dict[str, Any]:
        return {
            "window_size": self.window_size,
            "repeated_normalized_strategy_count": self.repeated_normalized_strategy_count,
            "max_mean_pairwise_strategy_distance": self.max_mean_pairwise_strategy_distance,
            "repeated_action_proposal_count": self.repeated_action_proposal_count,
            "consecutive_null_outcomes": self.consecutive_null_outcomes,
            "min_triggered_signals": self.min_triggered_signals,
        }


FORCED_ANTI_STAGNATION_PORTFOLIO: tuple[dict[str, str], ...] = (
    {
        "name": "observation",
        "hypothesis": "fresh visible-state observation",
        "principle": "reserve one bounded slot for a non-click observation or low-commitment action.",
    },
    {
        "name": "active_coordinate_probe",
        "hypothesis": "fresh coordinate interaction",
        "principle": "force at least one coordinate action away from recently repeated/passive proposals.",
    },
    {
        "name": "action_type_probe",
        "hypothesis": "different action family",
        "principle": "force an action id that recent failed proposals did not exercise.",
    },
    {
        "name": "mechanic_falsification",
        "hypothesis": "refute the current passive mechanic guess",
        "principle": "try the strongest non-taboo candidate that contradicts repeated waiting.",
    },
    {
        "name": "recovery_reset",
        "hypothesis": "recover from a stale local state",
        "principle": "reserve a bounded reset/recovery-style candidate when the frontier has stalled.",
    },
)

_NULL_OUTCOME_TOKENS = (
    "no_change",
    "no visible change",
    "null",
    "null_outcome",
    "no_effect",
    "same_state",
    "same level",
    "level_unchanged",
    "no_level_change",
    "stalled",
    "unchanged",
)


def _normalize_strategy_text(text: Any) -> str:
    words = re.findall(r"[a-z0-9]+", str(text or "").lower())
    return " ".join(words)


def _strategy_distance(left: str, right: str) -> float:
    left_tokens = set(_normalize_strategy_text(left).split())
    right_tokens = set(_normalize_strategy_text(right).split())
    if not left_tokens and not right_tokens:
        return 0.0
    if not left_tokens or not right_tokens:
        return 1.0
    return 1.0 - (len(left_tokens & right_tokens) / len(left_tokens | right_tokens))


def _outcome_is_null(outcome: Any) -> bool:
    if isinstance(outcome, Mapping):
        before = outcome.get("level_before")
        after = outcome.get("level_after")
        try:
            if before is not None and after is not None and int(after) <= int(before):
                return True
        except (TypeError, ValueError):
            pass
        if outcome.get("changed") is False or outcome.get("effect") == "none":
            return True
    text = str(outcome or "").strip().lower()
    return bool(text and any(token in text for token in _NULL_OUTCOME_TOKENS))


def _consecutive_null_outcomes(history: Sequence[Mapping[str, Any]]) -> int:
    count = 0
    for row in reversed(history):
        if _outcome_is_null(row.get("outcome")):
            count += 1
            continue
        break
    return count


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


def _candidate_action(candidate: Any) -> int:
    if isinstance(candidate, Mapping):
        value = candidate.get("action", candidate.get("action_id", 0))
    else:
        value = getattr(candidate, "action", getattr(candidate, "action_id", 0))
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _candidate_signature(candidate: Any, index: int) -> str:
    coord = _candidate_coordinate(candidate)
    action = _candidate_action(candidate)
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


@dataclass(frozen=True)
class AntiStagnationDiversityController:
    """Detect SGE strategy collapse and force a bounded diverse portfolio.

    The controller is deterministic and uses only router-local history. It never
    reads the environment's source, scorecard, level counter, or hidden success
    flag; outcomes are reduced to coarse visible-effect labels before they reach
    the strategy logic.
    """

    thresholds: StrategyCollapseThresholds = field(default_factory=StrategyCollapseThresholds)
    forced_portfolio: tuple[Mapping[str, str], ...] = FORCED_ANTI_STAGNATION_PORTFOLIO

    def collapse_definition(self) -> dict[str, Any]:
        return {
            "signals": [
                "repeated_normalized_strategy_text",
                "low_pairwise_strategy_distance",
                "repeated_action_proposals",
                "consecutive_null_outcomes",
            ],
            "thresholds": self.thresholds.as_dict(),
        }

    def diversity_metrics(self, history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        rows = list(history)[-self.thresholds.window_size :]
        strategy_texts = [str(row.get("strategy_text", "")) for row in rows if row.get("strategy_text")]
        normalized = [_normalize_strategy_text(text) for text in strategy_texts if _normalize_strategy_text(text)]
        strategy_counts = Counter(normalized)
        distances = [
            _strategy_distance(left, right)
            for i, left in enumerate(strategy_texts)
            for right in strategy_texts[i + 1 :]
        ]
        action_signatures = [str(row.get("chosen_signature", "")) for row in rows if row.get("chosen_signature")]
        action_counts = Counter(action_signatures)
        return {
            "history_window_size": len(rows),
            "strategy_text_count": len(strategy_texts),
            "unique_normalized_strategy_count": len(strategy_counts),
            "max_normalized_strategy_repeat": max(strategy_counts.values(), default=0),
            "mean_pairwise_strategy_distance": (sum(distances) / len(distances) if distances else 1.0),
            "unique_action_signature_count": len(action_counts),
            "max_action_signature_repeat": max(action_counts.values(), default=0),
            "consecutive_null_outcomes": _consecutive_null_outcomes(rows),
        }

    def assess(self, history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        metrics = self.diversity_metrics(history)
        signals = {
            "repeated_normalized_strategy_text": (
                metrics["max_normalized_strategy_repeat"]
                >= self.thresholds.repeated_normalized_strategy_count
            ),
            "low_pairwise_strategy_distance": (
                metrics["strategy_text_count"] >= 3
                and metrics["mean_pairwise_strategy_distance"]
                <= self.thresholds.max_mean_pairwise_strategy_distance
            ),
            "repeated_action_proposals": (
                metrics["max_action_signature_repeat"]
                >= self.thresholds.repeated_action_proposal_count
            ),
            "consecutive_null_outcomes": (
                metrics["consecutive_null_outcomes"] >= self.thresholds.consecutive_null_outcomes
            ),
        }
        triggered = [name for name, active in signals.items() if active]
        return {
            "collapse_detected": len(triggered) >= self.thresholds.min_triggered_signals,
            "signals": signals,
            "triggered_signals": triggered,
            "metrics": metrics,
            "collapse_definition": self.collapse_definition(),
        }

    def taboo_set(self, history: Sequence[Mapping[str, Any]]) -> list[str]:
        recent = list(history)[-self.thresholds.window_size :]
        taboo = {
            _normalize_strategy_text(row.get("strategy_text", ""))
            for row in recent
            if _outcome_is_null(row.get("outcome")) and row.get("strategy_text")
        }
        return sorted(text for text in taboo if text)

    def _recent_failed_signatures(self, history: Sequence[Mapping[str, Any]]) -> set[str]:
        return {
            str(row.get("chosen_signature", ""))
            for row in list(history)[-self.thresholds.window_size :]
            if _outcome_is_null(row.get("outcome")) and row.get("chosen_signature")
        }

    def rank_forced_portfolio(
        self,
        candidates: Sequence[Any],
        *,
        history: Sequence[Mapping[str, Any]],
        fallback_score_fields: Sequence[str],
        max_candidates: int,
        seen_coordinates: set[tuple[int, int]],
    ) -> dict[str, Any]:
        ordered = list(candidates)
        selected: list[Any] = []
        selected_ids: set[int] = set()
        selected_rows: list[dict[str, Any]] = []
        failed_signatures = self._recent_failed_signatures(history)
        recent_action_counts = Counter()
        for signature in failed_signatures:
            match = re.match(r"A(-?\d+)", signature)
            if match:
                recent_action_counts[int(match.group(1))] += 1

        def add(candidate: Any, category: str, *, allow_failed_signature: bool = False) -> bool:
            if id(candidate) in selected_ids or len(selected) >= max_candidates:
                return False
            signature = _candidate_signature(candidate, ordered.index(candidate))
            if not allow_failed_signature and signature in failed_signatures:
                return False
            selected.append(candidate)
            selected_ids.add(id(candidate))
            selected_rows.append({"name": category, "signature": signature})
            return True

        def ranked_pool(pool: Sequence[Any], *, prefer_low_recent_action: bool = False) -> list[Any]:
            def sort_key(candidate: Any) -> tuple[float, float, str]:
                recent = float(recent_action_counts.get(_candidate_action(candidate), 0))
                return (
                    recent if prefer_low_recent_action else 0.0,
                    -_fallback_score(candidate, fallback_score_fields),
                    _candidate_signature(candidate, ordered.index(candidate)),
                )

            return sorted(pool, key=sort_key)

        def add_from_pool(
            pool: Sequence[Any],
            category: str,
            *,
            allow_failed_signature: bool = False,
            prefer_low_recent_action: bool = False,
        ) -> bool:
            for candidate in ranked_pool(pool, prefer_low_recent_action=prefer_low_recent_action):
                if add(candidate, category, allow_failed_signature=allow_failed_signature):
                    return True
            return False

        add_from_pool(
            [candidate for candidate in ordered if _candidate_action(candidate) not in {0, 5, 6}],
            "observation",
            allow_failed_signature=True,
        )

        add_from_pool(
            [
                candidate
                for candidate in ordered
                if _candidate_action(candidate) == 6
                and (_candidate_coordinate(candidate) not in seen_coordinates)
            ],
            "active_coordinate_probe",
        )

        add_from_pool(
            [
                candidate
                for candidate in ordered
                if recent_action_counts.get(_candidate_action(candidate), 0) == 0
                and _candidate_action(candidate) not in {0, 5}
            ],
            "action_type_probe",
            allow_failed_signature=True,
            prefer_low_recent_action=True,
        )

        add_from_pool(
            [
                candidate
                for candidate in ordered
                if id(candidate) not in selected_ids and _candidate_action(candidate) not in {0, 5}
            ],
            "mechanic_falsification",
        )

        add_from_pool(
            [
                candidate
                for candidate in ordered
                if _candidate_action(candidate) in {0, 5}
                or _fallback_score(candidate, ("reset_score",)) > 0.0
            ],
            "recovery_reset",
            allow_failed_signature=True,
        )

        fallback_used = len({row["name"] for row in selected_rows}) < len(self.forced_portfolio)
        if len(selected) < min(max_candidates, len(ordered)):
            fallback_order = sorted(
                ordered,
                key=lambda candidate: (
                    -_fallback_score(candidate, fallback_score_fields),
                    _candidate_signature(candidate, ordered.index(candidate)),
                ),
            )
            for candidate in fallback_order:
                if add(candidate, "fallback_fill", allow_failed_signature=True):
                    fallback_used = True
                if len(selected) >= max_candidates:
                    break

        return {
            "ranked": selected[:max_candidates],
            "forced_portfolio_selected": selected_rows,
            "stable_fallback_used": bool(fallback_used),
            "diversity_after": {
                "forced_portfolio_category_count": len(
                    {row["name"] for row in selected_rows if row["name"] != "fallback_fill"}
                ),
                "selected_unique_signature_count": len({row["signature"] for row in selected_rows}),
                "selected_count": len(selected),
            },
            "taboo_set": self.taboo_set(history),
            "taboo_policy": (
                "normalize recently failed strategy text from null outcomes; ignore matching "
                "LLM proposals and force deterministic portfolio categories on collapse"
            ),
        }


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
    anti_stagnation_controller: AntiStagnationDiversityController | None = field(
        default_factory=AntiStagnationDiversityController
    )

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
            anti = (
                self.anti_stagnation_controller.assess(self.history)
                if self.anti_stagnation_controller is not None
                else {"collapse_detected": False}
            )
            self.last_diagnostics = {
                "llm_strategy_proposer_used": False,
                "reason": "no_candidates",
                "strategy_texts": [],
                "votes_by_index": {},
                "reflection_note": self._reflection_note,
                "anti_stagnation": anti,
                "win_check_used_for_ranking": False,
            }
            return []

        anti_before = (
            self.anti_stagnation_controller.assess(self.history)
            if self.anti_stagnation_controller is not None
            else {"collapse_detected": False, "metrics": {}}
        )
        if self.anti_stagnation_controller is not None and anti_before.get("collapse_detected"):
            forced = self.anti_stagnation_controller.rank_forced_portfolio(
                ordered,
                history=self.history,
                fallback_score_fields=self.fallback_score_fields,
                max_candidates=self.max_candidates,
                seen_coordinates=self._seen_coordinates,
            )
            selected = list(forced["ranked"])
            seen_this_call = {
                coord
                for candidate in selected
                for coord in [_candidate_coordinate(candidate)]
                if coord is not None
            }
            self._seen_coordinates |= seen_this_call
            first = selected[0] if selected else ordered[0]
            first_index = ordered.index(first)
            forced_names = [row["name"] for row in forced["forced_portfolio_selected"]]
            self.history.append(
                {
                    "step": self._step,
                    "strategy_text": "anti_stagnation_forced:" + ",".join(forced_names),
                    "chosen_signature": _candidate_signature(first, first_index),
                    "votes": {},
                    "outcome": "pending",
                }
            )
            anti_diagnostics = dict(anti_before)
            anti_diagnostics.update(
                {
                    "forced_portfolio": [dict(row) for row in self.anti_stagnation_controller.forced_portfolio],
                    "forced_portfolio_selected": forced["forced_portfolio_selected"],
                    "taboo_set": forced["taboo_set"],
                    "taboo_policy": forced["taboo_policy"],
                    "stable_fallback_used": forced["stable_fallback_used"],
                    "diversity_metrics_before_after": {
                        "before": anti_before.get("metrics", {}),
                        "after": forced["diversity_after"],
                    },
                }
            )
            self.last_diagnostics = {
                "llm_strategy_proposer_used": False,
                "strategy_texts": [],
                "votes_by_index": {},
                "parse_failure_count": 0,
                "completer_failure_count": 0,
                "reflection_note": self._reflection_note,
                "reflected_this_call": False,
                "suppressed_coordinate_count": 0,
                "step": self._step,
                "anti_stagnation": anti_diagnostics,
                "win_check_used_for_ranking": False,
            }
            return selected

        temperatures = tuple(self.temperatures[: self.k]) or (0.5,)
        candidate_lines = _format_candidate_lines(ordered)
        proposals = self.proposer.propose_many(self._context(), candidate_lines, temperatures=temperatures)
        taboo_set = (
            set(self.anti_stagnation_controller.taboo_set(self.history))
            if self.anti_stagnation_controller is not None
            else set()
        )

        votes: dict[int, int] = {}
        strategy_texts: list[str] = []
        any_completer_ok = False
        tabooed_proposals = 0
        for proposal in proposals:
            if proposal.get("completer_ok"):
                any_completer_ok = True
            if not proposal.get("parse_ok"):
                continue
            normalized_strategy = _normalize_strategy_text(proposal.get("strategy_text", ""))
            if normalized_strategy and normalized_strategy in taboo_set:
                tabooed_proposals += 1
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
            "anti_stagnation": {
                **anti_before,
                "taboo_set": sorted(taboo_set),
                "tabooed_proposal_count": tabooed_proposals,
                "taboo_policy": (
                    "normalize recently failed strategy text from null outcomes; ignore matching "
                    "LLM proposals before vote aggregation"
                ),
            },
            "win_check_used_for_ranking": False,
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
                "anti_stagnation_controller": self.anti_stagnation_controller is not None,
            }
        ]
