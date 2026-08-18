"""Tool-calling loop for ARC world-model induction (REQ-ARC-WMTE-6460).

WHAT THIS REPLACES. Single-shot induce() sends the whole transition corpus once and
the model mentally simulates its rule hypotheses against it inside a 38k-94k-token
think stream. This loop gives the model four in-process tools (arc_induction_tools):
it proposes an engine, `run_engine_on_transitions` RUNS it and returns a bounded
mismatch report, and the model iterates on measurements instead of imagination.
Prefill (report tokens) is ~15x cheaper than decode (think tokens), so every mental
simulation replaced by a tool round is a net win.

DEFAULT OFF. `CARNOT_ARC_INDUCE_TOOL_LOOP` unset or != "1" means this module is never
imported by the live path and induction is byte-identical to the shipped single-shot.
The loop earns any default flip through measurement, not assertion.

TERMINATION, all layered: hard turn cap (12, the in-repo refinement precedent), a
per-turn thinking budget (llama.cpp `thinking_budget_tokens` -- without it each turn
opens a fresh think block and 10 turns return to single-shot parity), the induce
timeout as a wall-clock deadline, best-so-far monotone accept (the fewest-mismatch
candidate wins at any cap), and early stop after 2 consecutive non-improving
candidates. If no parseable engine ever lands, the caller falls back to the shipped
single-shot induce -- failure is never worse than today.

TRANSPORT NOTE. llama-server b9606 grammar-enforces tool-call JSON lazily (armed
outside the think block), so thinking and tool calls coexist. gemma-4 has a native
chat handler; Qwen3.8 goes through the generic PEG autoparser, whose reliability was
UNVERIFIED at design time -- this loop therefore counts the tool-call parse rate per
run (`last_tool_loop_stats`), because a high parse-failure rate kills the approach
and must be visible, not inferred.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Any, Optional

import numpy as np

from carnot.agentic.arc_induction_tools import (
    TOOL_SCHEMAS,
    InductionToolSession,
    dispatch_tool,
)

DEFAULT_TURN_CAP = 12
DEFAULT_THINK_BUDGET = 3072
DEFAULT_EARLY_STOP_AFTER = 2


def tool_loop_enabled() -> bool:
    """Master switch. Unset -> the shipped single-shot path, byte-identical."""
    return os.environ.get("CARNOT_ARC_INDUCE_TOOL_LOOP") == "1"


def tool_loop_repair_enabled() -> bool:
    """REPAIR mode (REQ-ARC-WMTE-6470): single-shot stays primary; the recall-gated
    resample (REQ-ARC-WMTE-6410) routes its re-draw through this loop, seeded with the
    failed engine. Deliberately a DIFFERENT value of the same env var, so the two modes
    are mutually exclusive by construction and `induce()`'s `== "1"` hook stays dead."""
    return os.environ.get("CARNOT_ARC_INDUCE_TOOL_LOOP") == "repair"


def _turn_cap() -> int:
    try:
        return max(1, int(os.environ.get("CARNOT_ARC_INDUCE_TOOL_TURNS", DEFAULT_TURN_CAP)))
    except ValueError:
        return DEFAULT_TURN_CAP


def _think_budget() -> int:
    """Per-TURN think budget. 0 disables the request field (unbounded thinking)."""
    try:
        return int(os.environ.get("CARNOT_ARC_INDUCE_TOOL_THINK_BUDGET", DEFAULT_THINK_BUDGET))
    except ValueError:
        return DEFAULT_THINK_BUDGET


def _early_stop_after() -> int:
    try:
        return max(
            1,
            int(os.environ.get("CARNOT_ARC_INDUCE_TOOL_EARLY_STOP", DEFAULT_EARLY_STOP_AFTER)),
        )
    except ValueError:
        return DEFAULT_EARLY_STOP_AFTER


def _stall_turn_cap() -> int:
    """Consecutive turns WITHOUT a new engine submission before the loop aborts.
    0 (the default) disables the cap. Measured motivation: every 12-turn failure in
    the 13-cell A/B submitted only 1-2 candidates and spent the other turns
    inspecting -- the early-stop counter never sees those turns, so they burn
    10+ minutes per cell that a stall cap ends at the fallback instead."""
    try:
        return max(0, int(os.environ.get("CARNOT_ARC_INDUCE_TOOL_STALL_TURNS", 0)))
    except ValueError:
        return 0


_TOOL_INSTRUCTIONS = """
You have TOOLS. Use them instead of simulating grids in your head:

  * diff_grids(t) -- every cell transition t changed, with the action taken.
  * query_region(t, r0, r1, c0, c1, which) -- plain integer cells, no run-length decoding.
  * run_engine_on_transitions(code) -- RUNS your candidate engine on the observed
    transitions and returns concrete mismatches plus a held-out score. This is the only
    reliable test of a rule hypothesis. Do NOT hand-trace transforms; submit the
    candidate and read the report.
  * run_goal_on_states(code) -- runs a candidate is_level_complete on observed grids.

Do not hardcode observed coordinates: the report flags that, and held-out transitions
will fail. Prefer simple general rules.

You have a LIMITED number of turns. Only engines submitted through
run_engine_on_transitions count for anything; inspection alone scores zero. Submit
your FIRST candidate engine within your first 3 tool calls, even if it is rough --
the mismatch report is how you learn the rules, and you refine from it. When the
report shows 0 mismatches (or you cannot improve further), reply with ONLY one final
```python code block containing BOTH engine(grid, action, data) and
is_level_complete(grid).
""".strip()

# The measured failure this nudge answers (probe 1, tu93/Qwen3.8, 2026-08-17): 12
# turns, 44 tool calls, ALL of them diff_grids/query_region -- the model perceived
# forever and never submitted an engine, so the loop burned 33.9k decode tokens for
# zero candidates. After this turn count with no engine submission, the loop injects
# a user demand for one.
DEFAULT_FORCE_ENGINE_TURN = 3

_FORCE_ENGINE_NUDGE = (
    "STOP inspecting. You have spent your inspection budget. Your NEXT tool call MUST "
    "be run_engine_on_transitions with your best current candidate engine, however "
    "rough. The mismatch report will tell you what to fix."
)


def _force_engine_turn() -> int:
    try:
        return max(
            1,
            int(
                os.environ.get(
                    "CARNOT_ARC_INDUCE_TOOL_FORCE_ENGINE_TURN", DEFAULT_FORCE_ENGINE_TURN
                )
            ),
        )
    except ValueError:
        return DEFAULT_FORCE_ENGINE_TURN


def _sampling_overrides(payload: dict[str, Any]) -> None:
    """Honor the same opt-in env sampling knobs generate() reads. Absent -> no-op."""
    t = os.environ.get("CARNOT_ARC_INDUCE_TEMPERATURE")
    if t:
        try:
            payload["temperature"] = float(t)
        except ValueError:
            pass
    for env, key, cast in (
        ("CARNOT_ARC_INDUCE_TOP_P", "top_p", float),
        ("CARNOT_ARC_INDUCE_TOP_K", "top_k", int),
    ):
        v = os.environ.get(env)
        if v:
            try:
                payload[key] = cast(v)
            except ValueError:
                pass


def _post_chat(
    proposer: Any,
    messages: list[dict[str, Any]],
    *,
    turn: int,
    timeout_s: float,
) -> dict[str, Any]:
    """One /v1/chat/completions request with the tool schemas attached.

    Returns the raw response dict. Raises on transport failure -- the loop converts
    that into a clean fallback, same contract as _chat_complete_request."""
    payload: dict[str, Any] = {
        "messages": messages,
        "tools": TOOL_SCHEMAS,
        "tool_choice": "auto",
        "max_tokens": int(proposer.max_tokens),
        "temperature": 0.2,
        "cache_prompt": True,
    }
    _sampling_overrides(payload)
    # Per-turn seed via the proposer's own ladder helper, so a seeded run is
    # deterministic per turn without collapsing all turns onto one draw.
    seed = proposer.sampling_seed(turn)
    if seed is not None:
        payload["seed"] = seed
    tb = _think_budget()
    if tb > 0:
        payload["thinking_budget_tokens"] = tb
    req = urllib.request.Request(
        proposer._url() + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.load(r)


def _completion_tokens(raw: dict[str, Any]) -> int:
    """Decode-token count for one response; same two sources _chat_complete_request reads."""
    timings = raw.get("timings") if isinstance(raw.get("timings"), dict) else {}
    n = timings.get("predicted_n")
    if isinstance(n, int):
        return n
    usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
    ct = usage.get("completion_tokens")
    return ct if isinstance(ct, int) else 0


def _record_turn(proposer: Any, raw: dict[str, Any], msg: dict[str, Any]) -> None:
    """Feed the shared completion diagnostics, so existing instrumentation
    (stop-type logs, raw-length monkeypatches in the measurement harnesses) sees
    every loop turn exactly as it sees single-shot responses."""
    content = str(msg.get("content") or "")
    reasoning = str(msg.get("reasoning_content") or "")
    full = f"<think>\n{reasoning}\n</think>\n{content}" if reasoning else content
    choice = (raw.get("choices") or [{}])[0]
    normalized: dict[str, Any] = {
        "content": full,
        "stop_type": "limit" if choice.get("finish_reason") == "length" else "eos",
        "truncated": bool(raw.get("truncated")),
        "timings": {"predicted_n": _completion_tokens(raw)},
    }
    try:
        proposer._record_completion_diagnostics(normalized)
    except Exception:
        pass


def _extract_final_code(content: str) -> str:
    """The final-answer code block, if the message carries one ('' otherwise)."""
    if "```python" in content:
        body = content.split("```python", 1)[1]
        return body.split("```", 1)[0].strip()
    if "```" in content:
        body = content.split("```", 1)[1]
        body = body.split("```", 1)[0].strip()
        if "def engine" in body:
            return body
    return ""


def _looks_like_unparsed_tool_call(content: str) -> bool:
    """Qwn-through-generic-PEG failure shape: tool-call text left in `content` because
    the server's parser did not lift it into `tool_calls`. Counted, never executed."""
    head = content[:2000]
    return "<tool_call>" in content or ('"name":' in head and '"arguments"' in head)


def induce_with_tool_loop(
    proposer: Any,
    game: str,
    trans: list[Any],
    cell: int,
    *,
    previous_level_complete_grid: Optional[np.ndarray] = None,
    win_transition: Optional[Any] = None,
    seed_engine_code: Optional[str] = None,
    hud_mask: Any = None,
) -> tuple[bool, str]:
    """Run tool-assisted induction. Returns (True, note) after writing world_model.py,
    or (False, reason) -- in which case the caller runs the shipped single-shot path.

    Stats land on `proposer.last_tool_loop_stats` whatever the outcome, so a probe or
    an artifact can always report decode tokens, parse rate, and termination cause."""
    from carnot.agentic.arc_executable_world_model import (
        _induce_transitions_k,
        induce_prompt,
    )

    t0 = time.time()
    deadline = t0 + float(proposer.timeout)
    # Cold start: the single-shot path launches the server inside generate(); this loop
    # posts directly, so it must ensure the server itself or a cold start would always
    # fall back to single-shot without the loop ever running.
    try:
        if not proposer._ensure_server():
            return False, "tool loop: generator server unavailable"
    except Exception as exc:  # noqa: BLE001
        return False, f"tool loop: server ensure raised {type(exc).__name__}: {exc}"
    # `hud_mask` threads through so a CEGIS caller (REQ-ARC-WMTE-6480) grades candidates on
    # the same HUD-collapsed comparison its own gate uses. None (every prior caller) is
    # byte-identical to before.
    session = InductionToolSession(list(trans), cell=int(cell), hud_mask=hud_mask)
    # The prompt shows the VISIBLE rows only. Rendering the full window would hand the
    # model the held-out tail's deltas in-prompt and quietly defeat the aggregate-only
    # held-out score -- the exact leak the CEGIS acceptance split (REQ-ARC-WMTE-6090)
    # was built to close.
    base = induce_prompt(
        game,
        list(session.visible),
        int(cell),
        previous_level_complete_grid=previous_level_complete_grid,
        win_transition=win_transition,
        k=_induce_transitions_k(),
        include_playbook_exemplars=getattr(proposer, "include_playbook_exemplars", False),
    )
    # REPAIR SEEDING (REQ-ARC-WMTE-6470). Score the failed engine as candidate zero and
    # show the model the code WITH its measured mismatch report. Two properties follow:
    # the monotone accept can never return something with more visible mismatches than
    # the seed (the floor), and the model starts from a concrete measured diagnosis
    # instead of a blank page.
    seed_note = ""
    seed_report: Optional[dict[str, Any]] = None
    if seed_engine_code:
        seed_report = session.run_engine_on_transitions(seed_engine_code)
        if seed_report.get("ok"):
            brief = {
                k: seed_report.get(k)
                for k in (
                    "n_transitions_tested",
                    "n_correct",
                    "accuracy",
                    "cell_recall",
                    "mismatches",
                    "held_out",
                    "memorization_scan",
                )
            }
            seed_note = (
                "\n\nREPAIR MODE. A previous attempt produced the engine below. Its "
                "measured report on the observed transitions:\n"
                + json.dumps(brief, default=str)
                + "\n\nPrevious engine:\n```python\n"
                + seed_engine_code.strip()
                + "\n```\nFix THIS engine using the mismatch report. Do not start from "
                "scratch unless the report shows the approach itself is wrong."
            )
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": base + "\n\n" + _TOOL_INSTRUCTIONS + seed_note}
    ]
    stats: dict[str, Any] = {
        "seeded": bool(seed_engine_code),
        "seed_scoreable": bool(seed_report.get("ok")) if seed_report is not None else None,
        "seed_visible_mismatches": (
            session.candidates[-1].visible_mismatches if session.candidates else None
        ),
        "turns": 0,
        "decode_tokens_total": 0,
        "decode_tokens_per_turn": [],
        "tool_calls_total": 0,
        "tool_calls_by_name": {},
        "tool_calls_per_turn": [],
        "tool_call_parse_failures": 0,
        "unparsed_tool_call_text_turns": 0,
        "candidates_scored": 0,
        "force_engine_nudges": 0,
        "terminated_by": "",
        "final_answer_seen": False,
        "wall_s": 0.0,
    }
    proposer.last_tool_loop_stats = stats

    def _finish(reason: str) -> tuple[bool, str]:
        stats["terminated_by"] = reason
        stats["turns_completed"] = stats["turns"]
        stats["candidates_scored"] = len(session.candidates)
        # Per-candidate trajectories, in submission order: is the loop CONVERGING
        # (mismatches falling) or thrashing? This is the number that distinguishes
        # "needs a tighter early-exit" from "cannot converge on this cell".
        stats["mismatch_trajectory"] = [c.visible_mismatches for c in session.candidates]
        stats["holdout_trajectory"] = [c.holdout_accuracy for c in session.candidates]
        stats["wall_s"] = round(time.time() - t0, 2)
        best = session.best_candidate()
        if best is None:
            return False, f"tool loop: no scoreable engine ({reason})"
        stats["best_visible_mismatches"] = best.visible_mismatches
        stats["best_holdout_accuracy"] = best.holdout_accuracy
        stats["best_is_memorizing"] = best.is_memorizing
        code = best.code
        if "def is_level_complete" not in code:
            goal_code = next((c.code for c in session.candidates if c.has_goal), None)
            goal_def = _goal_only(goal_code) if goal_code is not None else None
            if goal_def is not None:
                # Append ONLY the extracted is_level_complete def. Appending the donor
                # verbatim would let its engine bind last and shadow the best engine --
                # the split-induce shadowing bug, re-created at accept time.
                code = code.rstrip() + "\n\n" + goal_def
            else:
                ok_g, goal = proposer.generate(
                    proposer._goal_only_prompt(game, previous_level_complete_grid, trans),
                    ("is_level_complete",),
                    tries=1,
                    codeonly_eligible=True,
                    engine_transitions=trans,
                )
                if not ok_g:
                    return False, f"tool loop: engine ok but no goal predicate ({reason})"
                code = proposer._combine_world_model(code, goal)
        if "import numpy" not in code:
            code = "import numpy as np\n\n" + code
        ok, note = proposer._write_world_model(
            game,
            code,
            note=(
                f"tool loop: {reason}, {stats['turns']} turns, "
                f"{best.visible_mismatches} visible mismatches"
            ),
        )
        return ok, note

    best_mismatches: Optional[int] = None
    if session.candidates:
        # The seed is the floor: a tool-round candidate only counts as improvement if it
        # beats the seed's visible-mismatch count.
        best_mismatches = session.candidates[-1].visible_mismatches
        if best_mismatches == 0:
            # The caller fired repair on a catastrophic-recall draw, yet the seed fits the
            # visible split exactly -- nothing for a mismatch-driven loop to iterate on.
            return _finish("seed_zero_mismatches")
    non_improving = 0
    cap = _turn_cap()
    early_after = _early_stop_after()
    stall_cap = _stall_turn_cap()
    stats["stall_cap"] = stall_cap
    # Turns since the model last SUBMITTED an engine (any submission resets it, even a
    # non-improving one -- non-improving submissions are the early-stop counter's job).
    turns_since_submission = 0
    # THE MODEL's submissions, distinct from the candidate ledger. In repair mode the
    # SEED occupies candidate zero, so a `session.candidates`-based nudge test can never
    # fire and the probe-1 defect (inspect forever, submit nothing) comes back through
    # the seeded door -- measured: a seeded cell went 12 turns with zero submissions.
    model_submitted = False
    for turn in range(cap):
        remaining = deadline - time.time()
        if remaining <= 0:
            return _finish("deadline")
        try:
            raw = _post_chat(
                proposer,
                messages,
                turn=turn,
                timeout_s=min(float(proposer.timeout), max(1.0, remaining)),
            )
        except Exception as exc:  # noqa: BLE001 - transport failure -> clean fallback
            stats["transport_error"] = f"{type(exc).__name__}: {exc}"[:300]
            return _finish("transport_error")
        stats["turns"] += 1
        choice = (raw.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        _record_turn(proposer, raw, msg)
        n_tok = _completion_tokens(raw)
        stats["decode_tokens_total"] += n_tok
        stats["decode_tokens_per_turn"].append(n_tok)
        content = str(msg.get("content") or "")
        tool_calls = msg.get("tool_calls") or []

        if tool_calls:
            # Feed the assistant turn back WITHOUT reasoning_content: reasoning is not
            # part of the conversation contract, and re-prefilling it would spend the
            # tokens the per-turn think budget exists to save.
            # `content or ""`: some chat templates reject a null content field on an
            # assistant tool-call turn; an empty string renders safely everywhere.
            messages.append(
                {"role": "assistant", "content": content or "", "tool_calls": tool_calls}
            )
            improved_this_turn = False
            turn_names: list[str] = []
            for tc in tool_calls:
                fn = tc.get("function") or {}
                name = str(fn.get("name") or "")
                args = fn.get("arguments") or "{}"
                result = dispatch_tool(session, name, args)
                stats["tool_calls_total"] += 1
                turn_names.append(name)
                stats["tool_calls_by_name"][name] = stats["tool_calls_by_name"].get(name, 0) + 1
                err = str(result.get("error") or "")
                if "unparseable JSON arguments" in err or "unknown tool" in err:
                    stats["tool_call_parse_failures"] += 1
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(tc.get("id") or f"call_{turn}"),
                        "content": json.dumps(result),
                    }
                )
                if name == "run_engine_on_transitions" and result.get("ok"):
                    m = session.candidates[-1].visible_mismatches
                    if best_mismatches is None or m < best_mismatches:
                        best_mismatches = m
                        improved_this_turn = True
            stats["tool_calls_per_turn"].append(turn_names)
            if (
                session.candidates
                and not improved_this_turn
                and "run_engine_on_transitions" in turn_names
            ):
                non_improving += 1
            elif improved_this_turn:
                non_improving = 0
            if best_mismatches == 0:
                return _finish("zero_mismatches")
            if non_improving >= early_after:
                return _finish("early_stop_non_improving")
            # Any submission counts as engagement, even a non-improving or failed one:
            # the stall cap targets inspection-only turns, not slow convergence.
            if "run_engine_on_transitions" in turn_names:
                turns_since_submission = 0
                model_submitted = True
            else:
                turns_since_submission += 1
            if stall_cap > 0 and turns_since_submission >= stall_cap:
                return _finish("stall_turns")
            # THE PROBE-1 FAILURE MODE, ANSWERED HERE. The model can perceive forever
            # (44 diff/query calls, zero engines, measured) because every inspection
            # turn looks locally productive. After the inspection budget, demand a
            # candidate: only engines submitted to the verifier count for anything.
            # `model_submitted`, NOT `session.candidates`: the repair seed pre-fills the
            # ledger, and a ledger-based test let seeded runs inspect forever unnudged.
            if not model_submitted and stats["turns"] >= _force_engine_turn():
                messages.append({"role": "user", "content": _FORCE_ENGINE_NUDGE})
                stats["force_engine_nudges"] += 1
            continue

        final_code = _extract_final_code(content)
        if final_code and "def engine" in final_code:
            stats["final_answer_seen"] = True
            # Score the final answer so the monotone accept can compare it fairly
            # against every tool-round candidate. The report is discarded; the
            # candidate ledger entry is what matters.
            session.run_engine_on_transitions(final_code)
            return _finish("final_answer")
        if _looks_like_unparsed_tool_call(content):
            stats["unparsed_tool_call_text_turns"] += 1
        stats["tool_calls_per_turn"].append([])  # keep the per-turn record aligned
        # A turn with no tool call and no final answer is a stall turn too.
        turns_since_submission += 1
        if stall_cap > 0 and turns_since_submission >= stall_cap:
            return _finish("stall_turns")
        # No tool call, no final code: one nudge, bounded by the turn cap.
        messages.append({"role": "assistant", "content": content})
        messages.append(
            {
                "role": "user",
                "content": (
                    "Call a tool, or reply with ONLY the final ```python block "
                    "containing engine and is_level_complete."
                ),
            }
        )
    return _finish("turn_cap")


def _goal_only(donor_code: str) -> Optional[str]:
    """Extract just the is_level_complete definition from donor code.

    None when the definition cannot be cleanly extracted -- the caller then asks the
    focused goal prompt instead. Never returns the donor verbatim: a donor carrying
    its own engine would bind last and shadow the accepted engine."""
    import ast

    try:
        tree = ast.parse(donor_code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "is_level_complete"
        ):
            seg = ast.get_source_segment(donor_code, node)
            if seg:
                return seg
    return None
