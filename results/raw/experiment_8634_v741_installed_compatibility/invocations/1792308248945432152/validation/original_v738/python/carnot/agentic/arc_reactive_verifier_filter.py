"""Verifier-filtered reactive loop for ARC-AGI-3 (2026-07-22, operator-directed architectural pivot).

**Researcher summary:**
    The standing E3 architecture, on a stall, induces an explicit symbolic Python world model
    (`engine()`/`is_level_complete()`) from observed transitions, then plans inside it. Every recent
    measurement (exp5727's full-registry sweep, exp5753's generic-primitive A/B, this session's own
    held-out sc25 probe) shows this near-universally fails on the frozen 9B generator:
    `heldout_accuracy ≈ 0.0`, zero new levels found. A dedicated CEGIS-refinement follow-up
    (`GAP-ARC-INDUCTION-REFINEMENT-NULL`) found that giving the model MORE COMPUTE to repair its own
    wrong guess does not help either -- corroborating arXiv:2606.31511 ("Falsification, Not Exposure"):
    frozen small models cannot self-correct from feedback; only comparison against real, external ground
    truth reliably filters bad outputs.

    This module is the resulting architectural pivot: instead of inducing a general symbolic model up
    front, propose ONE next action at a time (a proposal task, not a synthesis task -- much easier for a
    9B model) and let a VERIFIER filter the proposal against REAL, directly-observed evidence before it
    is ever committed to the environment. Two real, already-existing, oracle-distinct verifier signals
    are reused (not reinvented): (1) an exact-match reject against the agent's own observed dead-end/
    no-op history (zero-cost, 100% grounded -- no model involved), and (2) the already-trained, already
    live-validated `FrameChangeScorer` CNN (`arc_frame_change_predictor.py`, oracle-distinct: it predicts
    real pixel-level change, never reads the env's own win flag). This is the Duck-Harness-style reactive
    shape (react turn-by-turn, no upfront symbolic model) WITHOUT deleting the verifier -- the design
    doc's own warning (`docs/research-notes/arc-induction-quality-improvement-design-2026-07-20.md` §2:
    "Duck abandons verification; that is precisely what Carnot's founding thesis rejects").

**Detailed explanation for engineers:**
    `arc_llm_guided_solve.py` already tried the LLM-proposes half of this shape, but its execution loop
    is UNVERIFIED (`for mv in proposed: step(mv)` -- every proposal is blindly executed). This module
    reuses that file's prompt/parse helpers directly (no duplication) and adds the missing filter step.

    **A real bug found while building this, not assumed:** `arc_llm_guided_solve.py` calls
    `proposer.generate(prompt, required=(), validate=None, tries=1)` for its free-text `ACTIONS_JSON:`
    proposal task. `LocalGGUFProposer.generate()` is hardcoded for CODE generation --
    `arc_executable_world_model.py:1741,1755` unconditionally run `_extract_python(text)` then
    `ast.parse(code)` regardless of the `validate` argument (`validate` only adds an EXTRA check on top,
    it does not bypass extraction/parsing). A free-text JSON-line response is not valid Python, so this
    call structurally fails every time (confirmed directly: `local model code unusable ... syntax error`
    on every call in a live smoke test). This means `arc_llm_guided_solve.py`'s proposal loop has likely
    never actually worked -- no result artifact for it exists anywhere in this project's history. This
    module therefore does NOT reuse `proposer.generate()`; `_raw_llm_completion()` below is a minimal,
    correct free-text completion call (same `/completion` endpoint, same server-lifecycle reuse via
    `proposer._ensure_server()`/`proposer._url()`, but no code-extraction/ast-parse gate).

Spec: openspec/capabilities/arc-world-model-trust-energy/spec.md REQ-ARC-WMTE-5827
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np


@dataclass
class ReactiveFilterResult:
    game: str
    start_level: int
    reached_level: int
    levels_gained: int
    solved: bool
    total_actions: int
    llm_calls: int
    dead_end_rejections: int
    frame_change_rejections: int
    fallback_explore_steps: int
    first_levelup_actions: Optional[int]
    wall_s: float
    error: Optional[str]


class _ProposedAction:
    """Lightweight ArcAction-like wrapper so FrameChangeScorer.candidate_score() (which reads
    .action_id/.data) can score an LLM-proposed candidate the same way it scores a
    rich_action_candidates() candidate -- no separate scoring code path to maintain."""

    __slots__ = ("action_id", "data")

    def __init__(self, action_id: int, data: Optional[dict]) -> None:
        self.action_id = int(action_id)
        self.data = data


def _dead_end_key(state_key: str, action_id: int, data: Optional[dict]) -> tuple:
    x = data.get("x") if data else None
    y = data.get("y") if data else None
    return (state_key, int(action_id), x, y)


def _filter_candidates(
    proposed: list[dict],
    *,
    state_key: str,
    dead_end_keys: set[tuple],
    frame: Any,
    frame_change_scorer: Any,
) -> tuple[Optional[_ProposedAction], int, int]:
    """Reject known dead-ends, then rank survivors by the oracle-distinct FrameChangeScorer.

    Returns (chosen_or_None, dead_end_rejections, scored_count). A candidate rejected by the
    dead-end filter never reaches the model-based scorer -- the deterministic, zero-cost check
    always runs first, matching CLAUDE.md's "never trust a model check where a grounded one
    already answers the question" spirit.
    """
    survivors: list[_ProposedAction] = []
    dead_end_rejections = 0
    for item in proposed:
        aid = item.get("action")
        data = item.get("data")
        if aid is None:
            continue
        key = _dead_end_key(state_key, aid, data)
        if key in dead_end_keys:
            dead_end_rejections += 1
            continue
        survivors.append(_ProposedAction(aid, data))
    if not survivors:
        return None, dead_end_rejections, 0
    if frame_change_scorer is None:
        return survivors[0], dead_end_rejections, 0
    best = None
    best_score = float("-inf")
    for cand in survivors:
        try:
            score = float(frame_change_scorer.candidate_score(frame, cand))
        except Exception:
            score = 0.0
        if score > best_score:
            best_score = score
            best = cand
    return best, dead_end_rejections, len(survivors)


_JSON_ONLY_DIRECTIVE = (
    "/no_think\n"
    "CRITICAL OUTPUT RULES -- obey EXACTLY:\n"
    "1. Do NOT analyze the grid. Do NOT reason step-by-step. Do NOT explain your choice.\n"
    "2. Respond with ONLY the JSON array body that continues the line below -- nothing else.\n\n"
)
_JSON_PRIMER = "ACTIONS_JSON: ["


def _raw_llm_completion(proposer: Any, prompt: str, *, max_tokens: int) -> tuple[bool, str]:
    """A minimal, CORRECT free-text completion call for a proposal task (not code synthesis).

    Two real, non-mocked failure modes were found and fixed while building this (not assumed):
    (1) the raw `/completion` endpoint with a plain prompt does NOT reliably suppress this GGUF's
    reasoning for a non-chat-templated prompt -- the model emitted a correctly-formatted
    `ACTIONS_JSON:` line, then derailed mid-JSON into a `<think>` block, producing unparseable
    truncated output on every call. (2) Switching to the `/v1/chat/completions` endpoint
    (`_chat_complete_request`) did not fix it either -- the model spent the entire token budget
    reasoning and never reached a final answer (`reasoning_content` non-empty, `content` empty) --
    the SAME failure mode REQ-ARC-WMTE-5714 already found for the induce task ("reasoned ~7000+
    tokens ... never emitted the functions"). The fix that DOES work, reused from the project's own
    proven `_L2_CODEONLY_DIRECTIVE` pattern (`arc_executable_world_model.py:1110`): PRIME the
    completion by opening the answer for the model in the prompt itself (here, `ACTIONS_JSON: [`)
    with a stop sequence at the array's close, so the completion continues an already-open
    structure instead of choosing to reason first. Verified directly: this reliably produces
    parseable JSON where both prior approaches did not.
    """
    if not proposer._ensure_server():
        return False, "gpu llama-server failed to start"
    import json as _json
    import urllib.request

    primed_prompt = _JSON_ONLY_DIRECTIVE + prompt + "\n" + _JSON_PRIMER
    try:
        req = urllib.request.Request(
            proposer._url() + "/completion",
            data=_json.dumps(
                {
                    "prompt": primed_prompt,
                    "n_predict": int(max_tokens),
                    "temperature": 0.2,
                    "cache_prompt": True,
                    "stop": ["]"],
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=proposer.timeout) as r:
            response = _json.load(r)
        body = str(response.get("content") or "")
        return True, _JSON_PRIMER + body + "]"
    except Exception as exc:  # noqa: BLE001 - a completion failure is a datum, not a crash
        return False, f"completion request failed: {exc!r}"[:200]


def run_reactive_verifier_filter_progress(
    game: str,
    *,
    proposer: Any,
    seed: int = 0,
    budget: int = 400,
    propose_n: int = 5,
    max_llm_calls: int = 60,
    warmup_explore: int = 24,
    frame_change_scorer: Any = "load_submitted",
) -> ReactiveFilterResult:
    """Drive the verifier-filtered reactive loop on `game` for a bounded run, mirroring
    `arc_actions_to_progress.run_bounded_progress`'s call/return shape so the two mechanisms are
    directly A/B-comparable on the same games/budgets. Zero quota (offline arcade only).

    `frame_change_scorer="load_submitted"` (default) loads the SAME already-trained, already
    live-validated CNN the scored submission kernel uses (`_load_submitted_frame_change_scorer`);
    pass an explicit instance or None to override (e.g. for a no-scorer ablation control).
    """
    import random
    import time

    t0 = time.monotonic()
    rng = random.Random(seed)

    from arcengine import GameAction
    from carnot.agentic.arc_agi3_live_adapter import _game_action, _levels_completed
    from carnot.agentic.arc_agi3_world_model import frame_hash, grid_of
    from carnot.agentic.arc_executable_world_model import detect_cell, to_logical
    from carnot.agentic.arc_frame_change_predictor import frame_state_key
    from carnot.agentic.arc_graph_explore import _warm, rich_action_candidates
    from carnot.agentic.arc_llm_guided_solve import _delta_desc, _parse_actions, _prompt
    from carnot.agentic import arc_solver_kit as kit

    if frame_change_scorer == "load_submitted":
        from carnot.agentic.arc_competition_agent import _load_submitted_frame_change_scorer

        frame_change_scorer = _load_submitted_frame_change_scorer()

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    # warmup=True: some games (sc25) consume the first post-reset action as a no-op --
    # per ops/arc_solve_registry.yaml general_gotchas:first_step_after_reset_consumed.
    f = _warm(env, True)

    def ok(fr) -> bool:
        try:
            return np.asarray(grid_of(fr)).ndim == 2
        except Exception:
            return False

    if not ok(f):
        return ReactiveFilterResult(
            game=game,
            start_level=0,
            reached_level=0,
            levels_gained=0,
            solved=False,
            total_actions=0,
            llm_calls=0,
            dead_end_rejections=0,
            frame_change_rejections=0,
            fallback_explore_steps=0,
            first_levelup_actions=None,
            wall_s=round(time.monotonic() - t0, 3),
            error="degenerate_start_frame",
        )

    cell = detect_cell(grid_of(f))
    start_level = _levels_completed(f)
    best_level = start_level
    first_levelup: Optional[int] = None
    actions = 0
    llm_calls = 0
    dead_end_rejections_total = 0
    fallback_explore_steps = 0
    dead_end_keys: set[tuple] = set()
    recent: list[str] = []
    gp = to_logical(grid_of(f), cell)
    error: Optional[str] = None

    def alabel(aid, data):
        return f"{aid} (x={data['x']},y={data['y']})" if data and "x" in data else f"{aid}"

    def step(aid, data) -> bool:
        nonlocal f, actions, best_level, first_levelup, gp
        prev = gp
        prev_key = frame_state_key(f)
        prev_raw_frame = f
        nf = env.step(_game_action(GameAction, int(aid)), data=data)
        actions += 1
        if nf is None or not ok(nf):
            f = env.reset()
            gp = to_logical(grid_of(f), cell) if ok(f) else prev
            recent.append(f"{alabel(aid, data)} -> reset")
            return False
        f = nf
        # Keep the GroundTruthValidatedFrameChangeScorer's validation gate satisfied -- it
        # returns a flat 0.0 (ranking degenerates to first-survivor) until it has observed at
        # least one real transition, mirroring the live path's own fcs.observe_transition call
        # (arc_competition_agent.py:1739).
        if frame_change_scorer is not None and hasattr(frame_change_scorer, "observe_transition"):
            try:
                frame_change_scorer.observe_transition(prev_raw_frame, int(aid), data, nf)
            except Exception:
                pass
        cur = to_logical(grid_of(nf), cell)
        delta = _delta_desc(prev, cur)
        recent.append(f"{alabel(aid, data)} -> {delta}")
        if delta == "no change":
            dead_end_keys.add(_dead_end_key(prev_key, aid, data))
        gp = cur
        lvl = _levels_completed(nf)
        if lvl > best_level:
            best_level = lvl
            if first_levelup is None:
                first_levelup = actions
            return True
        return False

    frame_change_rejections = 0
    try:
        # Warmup: gather action->effect examples so the model has real grounding, matching
        # arc_llm_guided_solve.py's own warmup precedent (not a novel choice).
        while actions < warmup_explore and actions < budget:
            cands = rich_action_candidates(f) if ok(f) else []
            if not cands:
                f = _warm(env, False)
                continue
            c = cands[rng.randrange(min(len(cands), 8))]
            step(int(c.action_id), c.data)

        while actions < budget and best_level == start_level and llm_calls < max_llm_calls:
            avail = list(getattr(f, "available_actions", []) or range(1, 7))
            ok_code, text = _raw_llm_completion(
                proposer, _prompt(gp, recent, avail, propose_n), max_tokens=512
            )
            llm_calls += 1
            proposed = _parse_actions(text) if ok_code else []
            state_key = frame_state_key(f)
            chosen, dead_end_rejections, scored_count = _filter_candidates(
                proposed,
                state_key=state_key,
                dead_end_keys=dead_end_keys,
                frame=f,
                frame_change_scorer=frame_change_scorer,
            )
            dead_end_rejections_total += dead_end_rejections
            if scored_count > 1:
                # every survivor but the chosen one was implicitly deprioritized by the scorer
                frame_change_rejections += scored_count - 1
            if chosen is None:
                # every proposal was a known dead-end (or the LLM gave nothing usable) -- fall
                # back to one structured-exploration step rather than stalling on a bad proposer.
                fallback_explore_steps += 1
                cands = rich_action_candidates(f) if ok(f) else []
                if cands and actions < budget:
                    c = cands[rng.randrange(min(len(cands), 8))]
                    step(int(c.action_id), c.data)
                continue
            if step(chosen.action_id, chosen.data):
                break
    except Exception as exc:  # a policy crash on a game is itself a datum, not a harness bug
        error = f"{type(exc).__name__}: {exc}"[:300]

    return ReactiveFilterResult(
        game=game,
        start_level=start_level,
        reached_level=best_level,
        levels_gained=max(0, best_level - start_level),
        solved=best_level > start_level,
        total_actions=actions,
        llm_calls=llm_calls,
        dead_end_rejections=dead_end_rejections_total,
        frame_change_rejections=frame_change_rejections,
        fallback_explore_steps=fallback_explore_steps,
        first_levelup_actions=first_levelup,
        wall_s=round(time.monotonic() - t0, 3),
        error=error,
    )
