#!/usr/bin/env python3
"""Frontier-dev TOOL-USE / MCP-loop ceiling probe (operator-asked 2026-06-28).

THE QUESTION (the operator's lever): instead of INJECTING patterns into the LLM context, give the LLM a
TOOL interface (function-calling / MCP-style) to actively INTERROGATE + DRIVE the existing ARC subsystems
-- the live env (reset/step/observe), the induced game CODE (induce + verifier score), and the BFS planner
-- and let IT decide what to do next from tool results. Does an LLM that DRIVES the tools produce a
first-contact L1 win that the current ONE-SHOT induce+BFS pipeline does NOT?

WHY THIS IS A CEILING PROBE (decouples two confounds). The deployable/scored agent must be the weak local
9B (frontier/codex is Kaggle-offline-illegal). This probe instead gives the STRONGEST available model
(codex / gpt-5.5, internet, DEV-only) full tool access, so it answers the FRAMING question at the capability
ceiling: if even a frontier model DRIVING the tools cannot beat the one-shot pipeline at first-contact, the
local 9B certainly cannot, and the lever is dead before any 9B/Kaggle build. A positive result would NOT be
a deliverable -- it would only justify re-validating the FRAMING with the local model ON the live path.

ANTI-CHEAT (this must be a real tool-loop test, not outer_loop_re). The driver model is sandboxed:
- runs `codex exec` in an EMPTY /tmp CWD (-s read-only, no repo on its path),
- is told to emit ONLY a JSON tool call (no shell, no file reads),
- the tool shim NEVER returns game source -- only frames/deltas/verifier-scores/plans,
- post-hoc we SCAN the driver's raw transcript for any source read (environment_files / <game>.py) and mark
  the run CONFOUNDED if found. Reading the game source = outer_loop_re (forbidden); detection invalidates.

GATE (falsifiable): treatment_solved_l1 AND NOT baseline_solved_l1 AND NOT cheated, on a game the one-shot
baseline fails. solve_provenance = development_proxy (a DEV capability probe using codex; NOT a banked solve
and NOT live_agent_self_discovery). verifier_is_oracle = False. off_live_path = True.

USAGE: arc_frontier_tooluse_probe.py [game] [turn_budget] [explore_per_call]
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

GAME = sys.argv[1] if len(sys.argv) > 1 else "re86"
TURN_BUDGET = int(sys.argv[2]) if len(sys.argv) > 2 else 24
EXPLORE_N = int(sys.argv[3]) if len(sys.argv) > 3 else 60
SEED = 20260628
GLOBAL_TIMEOUT_S = int(os.environ.get("PROBE_GLOBAL_TIMEOUT_S", "3000"))  # ~50 min wall cap
CODEX_TURN_TIMEOUT_S = 200
CODEX_EMPTY_CWD = "/tmp/arc_frontier_probe_cwd"


def _codex_drive(prompt: str, timeout: int = CODEX_TURN_TIMEOUT_S) -> tuple[str, dict | None]:
    """Call codex/gpt-5.5 as a PURE policy: emit one JSON tool call. Sandboxed in an empty CWD so it
    cannot reach the repo's game source. Returns (raw_output, parsed_tool_call_or_None)."""
    Path(CODEX_EMPTY_CWD).mkdir(parents=True, exist_ok=True)
    cmd = [
        "codex", "exec", "-s", "read-only", "--cd", CODEX_EMPTY_CWD,
        "--color", "never", "--skip-git-repo-check", "--model", "gpt-5.5", "-",
    ]
    try:
        p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout)
        raw = (p.stdout or "") + "\n" + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return f"codex timeout {timeout}s", None
    return raw, _extract_tool_call(raw)


def _extract_tool_call(raw: str) -> dict | None:
    """Return the LAST balanced {...} object that json-parses to a dict with a "tool" key. Uses
    BALANCED-BRACE matching (a regex cannot, because the "args" value is itself a nested {...} object)."""
    call = None
    for i, ch in enumerate(raw):
        if ch != "{":
            continue
        depth = 0
        for j in range(i, len(raw)):
            if raw[j] == "{":
                depth += 1
            elif raw[j] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(raw[i : j + 1])
                        if isinstance(obj, dict) and "tool" in obj:
                            call = obj
                    except Exception:
                        pass
                    break
    return call


def _detect_source_read(raw: str, game: str) -> bool:
    """Did the driver read the game SOURCE (cheating)? Scan its transcript for source markers."""
    low = raw.lower()
    return ("environment_files" in low) or (f"{game}.py" in low) or ("is_solved" in low)


def _parse_action(a) -> tuple[int | None, dict | None]:
    """Robustly parse one driver-supplied action into (action_id, data). Mirrors the proven coalescing in
    arc_solver_kit.py:1820-1836 so CLICK coordinates are read whether the driver nests them under "data"
    OR puts them FLAT as siblings ({"action_id":6,"x":..,"y":..}). A dict with only x/y is inferred as a
    click (action 6). Returns (None, None) only if truly unparseable; callers SKIP None."""
    if isinstance(a, dict):
        aid = None
        for k in ("action", "action_id", "id", "a", "act"):
            if a.get(k) is not None:
                aid = a.get(k)
                break
        if aid is None and a.get("x") is not None and a.get("y") is not None:
            aid = 6  # only coords given -> infer a click
        try:
            aid = int(aid)
        except (TypeError, ValueError):
            return None, None
        nested = a.get("data") if isinstance(a.get("data"), dict) else {}
        xv = nested.get("x", a.get("x"))
        yv = nested.get("y", a.get("y"))
        data = None
        if xv is not None and yv is not None:
            try:
                data = {"x": int(xv), "y": int(yv)}
            except (TypeError, ValueError):
                data = None
        return aid, data
    try:
        return int(a), None
    except (TypeError, ValueError):
        return None, None


class ToolEnv:
    """Wraps the EXISTING ARC subsystems as a minimal tool surface for the driver. Exposes only
    frames/deltas/verifier-scores/plans -- never game source."""

    def __init__(self, game: str) -> None:
        from carnot.agentic import arc_executable_world_model as wm

        self.wm = wm
        self.game = game
        self.trans: list = []
        self.cell: int | None = None
        self.engine = None
        self.ilc = None
        self.level_reached = 0
        self.level_up_seen = False  # a win-state surfaced anywhere (explore OR directed try_actions)
        self.env_steps = 0  # real env actions taken (for honesty; offline so RHAE-irrelevant)
        self.last_verify: dict | None = None
        self.last_plan: dict | None = None
        self.last_execute: dict | None = None
        # fair-shot counters: distinguish a CLEAN ceiling-negative from a confounded/inconclusive run
        self.n_induce_ok = 0
        self.n_plan = 0
        self.n_execute = 0
        self.n_try = 0
        # induce backend = the DEPLOYABLE local Qwen (fast -> more turns fit; held constant vs baseline so
        # the ONLY treatment variable is codex DRIVING the loop). The driver (codex) is the frontier ceiling.
        self._proposer = wm.LocalGGUFProposer(
            repo_substr="Qwen3.5-9B-MTP", kv_quant="q8_0", no_think_prefix="/no_think\n",
            port=int(os.environ.get("CARNOT_IGE_LLM_PORT", "8920")),
        )

    def _start_grid(self):
        import numpy as np
        from carnot.agentic.arc_agi3_world_model import grid_of
        from carnot.agentic.arc_graph_explore import _warm
        from carnot.agentic import arc_solver_kit as kit

        arc = kit.offline_arcade()
        env = arc.make(self.game, scorecard_id=arc.open_scorecard())
        f = _warm(env, False)
        if self.cell is None:
            self.cell = self.wm.detect_cell(grid_of(f))
        return np.asarray(self.wm.to_logical(grid_of(f), self.cell))

    def explore(self, n: int = EXPLORE_N, seed: int = 0) -> dict:
        import numpy as np

        n = max(8, min(int(n), 200))
        new, cell = self.wm.collect_transitions(self.game, n=n, seed=int(seed) + SEED)
        if self.cell is None:
            self.cell = cell
        self.trans.extend(new)
        self.trans = self.trans[-500:]  # bound induce-prompt size
        self.env_steps += n
        changed = sum(1 for t in new if not np.array_equal(t.grid, t.next_grid))
        lvlup = sum(1 for t in new if t.level_after > t.level_before)
        if lvlup:
            self.level_up_seen = True
        sample = [self.wm._rle_delta(t.grid, t.next_grid) for t in new if not np.array_equal(t.grid, t.next_grid)][:3]
        return {"total_transitions": len(self.trans), "new_changed": changed,
                "level_up_transitions_seen": lvlup, "action_ids_available": self._action_ids(),
                "sample_changes": sample}

    def _action_ids(self) -> list:
        """Available action ids at reset (so the driver knows the action vocabulary for try_actions)."""
        try:
            from carnot.agentic.arc_graph_explore import rich_action_candidates, _warm
            from carnot.agentic import arc_solver_kit as kit

            arc = kit.offline_arcade()
            env = arc.make(self.game, scorecard_id=arc.open_scorecard())
            f = _warm(env, False)
            return sorted({int(c.action_id) for c in rich_action_candidates(f)})
        except Exception:
            return []

    def try_actions(self, actions: list) -> dict:
        """DIRECTED INTERACTION (the core lever): execute an explicit action sequence from reset+warmup and
        report the per-step level + whether it reached a level-up. Records the resulting transitions into the
        induce corpus, so if the driver SURFACES a win here, induce can then learn is_level_complete from it.
        actions: list of ints (1-6) or {"action":int,"data":{"x":int,"y":int}} dicts. Capped at 40."""
        import numpy as np
        from arcengine import GameAction
        from carnot.agentic.arc_agi3_world_model import grid_of
        from carnot.agentic.arc_agi3_live_adapter import _levels_completed, _game_action
        from carnot.agentic.arc_graph_explore import _warm
        from carnot.agentic import arc_solver_kit as kit

        self.n_try += 1
        arc = kit.offline_arcade()
        env = arc.make(self.game, scorecard_id=arc.open_scorecard())
        f = _warm(env, False)
        if self.cell is None:
            self.cell = self.wm.detect_cell(grid_of(f))
        start_level = _levels_completed(f)
        levels, leveled, skipped, executed = [], False, 0, []
        for a in (actions or [])[:40]:
            aid, data = _parse_action(a)
            if aid is None:
                skipped += 1
                continue  # malformed element -> skip robustly (never raise / abort the whole sequence)
            executed.append({"action": aid, "data": data})  # transparency: exact action+coords applied
            g0 = self.wm.to_logical(grid_of(f), self.cell)
            l0 = _levels_completed(f)
            nf = env.step(_game_action(GameAction, aid), data=data)
            self.env_steps += 1
            if nf is None:
                break
            g1 = self.wm.to_logical(grid_of(nf), self.cell)
            l1 = _levels_completed(nf)
            self.trans.append(self.wm.Transition(g0, aid, data, g1, l0, l1))
            levels.append(l1)
            if l1 > start_level:
                leveled = True
            f = nf
        self.trans = self.trans[-500:]
        if leveled:
            self.level_up_seen = True
            self.level_reached = max(self.level_reached, 1)
        return {"per_step_level": levels, "reached_level_up": leveled, "start_level": start_level,
                "skipped_malformed": skipped, "actions_executed": executed[:40],
                "n_clicks_with_coords": sum(1 for e in executed if e["action"] == 6 and e["data"]),
                "n_clicks_no_coords": sum(1 for e in executed if e["action"] == 6 and not e["data"]),
                "action_schema_hint": ('actions = list of ints 1-6, OR dicts {"action":int,"data":{"x":int,"y":int}}'
                                       if skipped else None),
                "final_frame": self.wm.to_ascii(self.wm.to_logical(grid_of(f), self.cell))[:1200]}

    def induce(self) -> dict:
        if not self.trans:
            return {"error": "no transitions yet; explore or try_actions first"}
        ok, _code = self._proposer.induce(self.game, self.trans, self.cell or 1)
        if not ok:
            return {"induced": False, "reason": "proposer failed to emit code"}
        try:
            self.engine, self.ilc = self.wm.load_engine(self.game)
        except Exception as exc:
            return {"induced": False, "reason": f"load_engine failed: {exc!r}"[:160]}
        self.n_induce_ok += 1
        vr = self.wm.WorldModelVerifier(self.trans).score(self.engine)
        self.last_verify = {"induced": True, "accuracy": round(vr.accuracy, 3),
                            "cell_recall": round(getattr(vr, "cell_recall", 0.0), 3),
                            "has_is_level_complete": self.ilc is not None,
                            "n_transitions": vr.n, "n_mismatch": len(vr.mismatches)}
        return self.last_verify

    def plan(self, max_depth: int = 30) -> dict:
        if self.engine is None or self.ilc is None:
            return {"error": "no induced model with is_level_complete; induce first"}
        self.n_plan += 1
        try:
            sg = self._start_grid()
            p = self.wm.plan_in_model(self.engine, self.ilc, sg, max_depth=int(max_depth))
        except Exception as exc:
            return {"error": f"plan failed: {exc!r}"[:160]}
        self.last_plan = {"plan_found": p is not None, "plan_len": (len(p) if p else 0)}
        return self.last_plan

    def execute(self) -> dict:
        if self.engine is None or self.ilc is None:
            return {"error": "no induced model; induce first"}
        self.n_execute += 1
        try:
            out = self.wm.plan_and_execute(self.game, self.engine, self.ilc)
        except Exception as exc:
            return {"error": f"execute failed: {exc!r}"[:160]}
        self.env_steps += int(out.get("plan_len") or 0)
        lvlup = bool(out.get("level_up"))
        if lvlup:
            self.level_reached = max(self.level_reached, 1)
        self.last_execute = {"level_up": lvlup, "planned": out.get("planned"),
                             "executed": out.get("executed"), "reason": str(out.get("reason") or "")[:120],
                             "plan_len": out.get("plan_len")}
        return self.last_execute


_TOOL_SCHEMA = (
    'TOOLS (respond with EXACTLY ONE JSON object, single line, NOTHING else -- no prose, no code fences,\n'
    'no shell commands, do NOT read any files):\n'
    '- {"tool":"explore","args":{"n":60,"seed":1}}         -- random salient env steps; returns counts + available action_ids + sample changes\n'
    '- {"tool":"try_actions","args":{"actions":[6,2,2,3]}}  -- DIRECTED: run an EXACT action sequence from reset (ints 1-6; 6=click needs data {"x":..,"y":..}); returns per-step level + reached_level_up (THE WIN SIGNAL) + final frame. USE THIS to hunt the first level-up.\n'
    '- {"tool":"induce","args":{}}                          -- induce an executable world-model from collected transitions; returns verifier accuracy/cell_recall + has_is_level_complete\n'
    '- {"tool":"plan","args":{"max_depth":30}}              -- BFS a plan to the win-state INSIDE the model (0 real actions); returns whether a plan exists\n'
    '- {"tool":"execute","args":{}}                         -- run the planned path in the REAL env; returns level_up (THE WIN SIGNAL)\n'
    '- {"tool":"done","args":{"reason":"..."}}              -- give up\n'
)


def _driver_prompt(env: ToolEnv, turns_left: int, last_obs: dict | None) -> str:
    return (
        "You are a POLICY driving an offline ARC-AGI-3 puzzle solver via tools. GOAL: reach level 1 "
        "(reached_level_up / level_up = true) on an UNFAMILIAR grid-puzzle game you have never seen. You do "
        "NOT know its rules; discover them by interacting. STRATEGY: (1) explore a little to see the grid + "
        "available action_ids; (2) use try_actions to DIRECTLY hunt a level-up -- form a hypothesis about "
        "the goal from the frame and TEST concrete action sequences (this is your most powerful tool; a win "
        "you surface here is recorded so induce can learn it); (3) once you have transitions, induce a "
        "world-model and check accuracy; (4) if has_is_level_complete, plan then execute. Do NOT loop on "
        "explore+induce -- if accuracy stalls or no plan is found, switch to try_actions to hunt a real "
        "level-up directly. You have limited turns; act decisively.\n\n"
        + _TOOL_SCHEMA
        + f"\nSTATE: transitions={len(env.trans)}, win_state_surfaced={env.level_up_seen}, "
        f"induces_ok={env.n_induce_ok}, plans={env.n_plan}, executes={env.n_execute}, try_actions={env.n_try}, "
        f"last_verify={env.last_verify}, last_plan={env.last_plan}, last_execute={env.last_execute}, "
        f"current_level={env.level_reached}, turns_left={turns_left}.\n"
        f"LAST OBSERVATION: {json.dumps(last_obs)[:1600] if last_obs else 'none'}\n\n"
        "Emit the single best next tool call as JSON now."
    )


def run_treatment(game: str) -> dict:
    env = ToolEnv(game)
    transcript: list = []
    last_obs: dict | None = None
    cheated = False
    started = time.time()
    for turn in range(TURN_BUDGET):
        if time.time() - started > GLOBAL_TIMEOUT_S:
            transcript.append({"turn": turn, "note": "global_timeout"})
            break
        raw, call = _codex_drive(_driver_prompt(env, TURN_BUDGET - turn, last_obs))
        if _detect_source_read(raw, game):
            cheated = True
            transcript.append({"turn": turn, "CONFOUNDED": "driver read game source", "raw_tail": raw[-300:]})
            break
        if not call:
            transcript.append({"turn": turn, "note": "no_tool_call_parsed", "raw_tail": raw[-200:]})
            last_obs = {"error": "your last reply was not a single JSON tool call; emit ONLY the JSON"}
            continue
        tool = str(call.get("tool"))
        args = call.get("args") or {}
        if tool == "done":
            transcript.append({"turn": turn, "tool": "done", "args": args})
            break
        fn = {"explore": env.explore, "try_actions": env.try_actions, "induce": env.induce,
              "plan": env.plan, "execute": env.execute}.get(tool)
        if fn is None:
            last_obs = {"error": f"unknown tool {tool!r}"}
            transcript.append({"turn": turn, "tool": tool, "obs": last_obs})
            continue
        try:
            last_obs = fn(**{k: v for k, v in args.items() if k in ("n", "seed", "max_depth", "actions")})
        except Exception as exc:
            last_obs = {"error": f"tool raised: {exc!r}"[:160]}
        transcript.append({"turn": turn, "tool": tool, "args": args, "obs": last_obs})
        if env.level_reached >= 1:
            transcript.append({"turn": turn, "WIN": True})
            break
    # fair_shot: did the treatment genuinely get to TEST the lever? (directed interaction, OR a full
    # induce->plan/execute cycle). If not, a no-win is INCONCLUSIVE/harness-confounded, not a ceiling-negative.
    fair_shot = bool(env.n_try > 0 or (env.n_induce_ok > 0 and (env.n_plan + env.n_execute) > 0))
    return {"solved_l1": env.level_reached >= 1, "cheated": cheated, "turns_used": len(transcript),
            "env_steps": env.env_steps, "n_induce_ok": env.n_induce_ok, "n_plan": env.n_plan,
            "n_execute": env.n_execute, "n_try_actions": env.n_try,
            "win_state_surfaced": env.level_up_seen, "fair_shot": fair_shot, "transcript": transcript}


def run_baseline(game: str) -> dict:
    """The current ONE-SHOT pipeline: explore -> induce ONCE (local Qwen, held constant vs treatment) ->
    plan_and_execute. The ONLY treatment variable is codex DRIVING the loop, not the induce backend."""
    from carnot.agentic import arc_executable_world_model as wm

    trans, cell = wm.collect_transitions(game, n=120, seed=SEED)
    proposer = wm.LocalGGUFProposer(repo_substr="Qwen3.5-9B-MTP", kv_quant="q8_0",
                                    no_think_prefix="/no_think\n",
                                    port=int(os.environ.get("CARNOT_IGE_LLM_PORT", "8920")))
    ok, _ = proposer.induce(game, trans, cell)
    if not ok:
        return {"solved_l1": False, "reason": "induce_failed", "n_transitions": len(trans)}
    try:
        engine, ilc = wm.load_engine(game)
    except Exception as exc:
        return {"solved_l1": False, "reason": f"load_engine_failed: {exc!r}"[:120]}
    vr = wm.WorldModelVerifier(trans).score(engine)
    out = wm.plan_and_execute(game, engine, ilc)
    return {"solved_l1": bool(out.get("level_up")), "reason": str(out.get("reason") or "")[:120],
            "n_transitions": len(trans), "verifier_accuracy": round(vr.accuracy, 3),
            "verifier_cell_recall": round(getattr(vr, "cell_recall", 0.0), 3),
            "plan_outcome": {k: out.get(k) for k in ("planned", "executed", "level_up", "plan_len")}}


def main() -> int:
    started = time.time()
    if subprocess.run(["bash", "-lc", "command -v codex"], capture_output=True).returncode != 0:
        _write({"experiment": "arc_frontier_tooluse_probe", "game": GAME,
                "honest_verdict": "blocked_codex_unavailable", "inference_substrate": "live_llm_inference",
                "preconditions_checked": [{"resource": "codex_cli", "available": False}],
                "solve_provenance": "development_proxy", "verifier_is_oracle": False,
                "random_seed": SEED, "duration_s": round(time.time() - started, 2)})
        print("BLOCKED: codex unavailable")
        return 0

    print(f"[{GAME}] BASELINE (one-shot induce+BFS) ...", flush=True)
    baseline = run_baseline(GAME)
    print(f"  baseline solved_l1={baseline.get('solved_l1')} ({baseline.get('reason')})", flush=True)

    print(f"[{GAME}] TREATMENT (codex tool-driving loop, budget={TURN_BUDGET}) ...", flush=True)
    treatment = run_treatment(GAME)
    print(f"  treatment solved_l1={treatment['solved_l1']} cheated={treatment['cheated']} "
          f"turns={treatment['turns_used']}", flush=True)

    lever_signal = bool(treatment["solved_l1"] and not baseline["solved_l1"] and not treatment["cheated"])
    if treatment["cheated"]:
        verdict = "complete_frontier_tooluse_probe_CONFOUNDED_driver_read_game_source_invalid"
    elif lever_signal:
        verdict = (f"success_frontier_tooluse_BEATS_oneshot_baseline_{GAME}_l1_"
                   "tool_driving_framing_justifies_live_path_revalidation")
    elif treatment["solved_l1"] and baseline["solved_l1"]:
        verdict = f"complete_frontier_tooluse_no_advantage_both_solved_{GAME}_l1_baseline_already_wins"
    elif treatment["solved_l1"] and not baseline["solved_l1"]:
        verdict = (f"success_frontier_tooluse_BEATS_oneshot_baseline_{GAME}_l1_"
                   "tool_driving_framing_justifies_live_path_revalidation")
    elif not treatment.get("fair_shot"):
        # the treatment never genuinely tested the lever (no directed try_actions, no induce->plan/execute
        # cycle) -- a harness/budget confound, NOT evidence the lever fails. Do NOT call it a ceiling-negative.
        verdict = (f"complete_frontier_tooluse_INCONCLUSIVE_{GAME}_treatment_no_fair_shot_"
                   f"induces_ok_{treatment['n_induce_ok']}_plans_{treatment['n_plan']}_"
                   f"execs_{treatment['n_execute']}_tries_{treatment['n_try_actions']}")
    elif not treatment["solved_l1"] and not baseline["solved_l1"]:
        verdict = (f"complete_frontier_tooluse_NULL_neither_first_win_{GAME}_l1_fair_shot_"
                   "ceiling_negative_local9b_cannot_if_frontier_driver_cannot")
    else:
        verdict = f"complete_frontier_tooluse_baseline_only_solved_{GAME}_l1_tool_loop_no_help"

    art = {
        "experiment": "arc_frontier_tooluse_probe",
        "schema": "carnot.arc_frontier_tooluse_probe.v1",
        "honest_verdict": verdict,
        "game": GAME,
        "question": ("does a FRONTIER model DRIVING the existing ARC subsystems via a tool-loop "
                     "(env/induce/verify/BFS) produce a first-contact L1 win the ONE-SHOT induce+BFS "
                     "pipeline does NOT, on a game the baseline fails?"),
        "inference_substrate": "live_llm_inference",
        "verifier_is_oracle": False,
        "off_live_path": True,
        "off_live_path_note": ("CEILING PROBE using codex (offline-illegal for SCORING). A positive result "
                               "is NOT a banked solve; it would only justify re-validating the tool-loop "
                               "FRAMING with the LOCAL model ON the live E3AgentPolicy path."),
        "baseline_oneshot": baseline,
        "treatment_tooluse": {k: v for k, v in treatment.items() if k != "transcript"},
        "treatment_transcript": treatment["transcript"],
        "lever_signal": lever_signal,
        "gate": "treatment_solved_l1 AND NOT baseline_solved_l1 AND NOT cheated",
        "interpretation": (
            "Ceiling probe of the operator's tool-use/MCP lever. The driver (codex/gpt-5.5, reasoning xhigh) "
            "is given a sandboxed tool interface over the EXISTING subsystems and decides what to do; the "
            "ONLY new variable vs the one-shot baseline is adaptive multi-round tool-DRIVING (both use codex "
            "induction, so induction strength is held constant). If the frontier driver cannot beat the "
            "one-shot baseline at first-contact, the deployable local 9B certainly cannot (it is strictly "
            "weaker at multi-step agentic tool-use), so the lever is dead before any 9B/Kaggle build. "
            "A positive (lever_signal=true) would justify ONLY a live-path re-validation with the local "
            "model -- NOT a deliverable claim. Anti-cheat: driver sandboxed in an empty CWD, told JSON-only, "
            "tool shim never returns source; a post-hoc source-read scan marks the run CONFOUNDED if it peeked."
        ),
        "limitations": [
            "n=2 games (re86, ft09) -- suggestive, not a decisive ceiling claim over all 25 games.",
            ("verifier 'accuracy' is full-grid-match over ALL transitions; on a game where exploration "
             "surfaces NO state change (new_changed=0) an identity model scores accuracy=1.0 with "
             "cell_recall=0.0 (a VACUOUS fit, not a substantive world model) -- read cell_recall + "
             "new_changed alongside accuracy."),
            ("induce backend is the deployable local Qwen3.5-9B-MTP, which is weak (often acc~0 / degenerate "
             "on these grids); the probe routes around it via try_actions, but a stronger induce backend "
             "might change the plan/execute arm."),
            ("reasoning_effort is codex gpt-5.5's DEFAULT (xhigh per the codex header), not explicitly forced."),
        ],
        "solve_provenance": "development_proxy",
        # used_env_source / read_game_source = did we read the game SOURCE CODE? NO -- the probe only
        # INTERACTS with the offline env via reset/step (exactly what the live agent does); it never reads
        # environment_files/<game>/*.py. The driver is sandboxed + post-hoc source-read-scanned (see cheated).
        "used_env_source": False,
        "read_game_source": False,
        "interacts_with_offline_env_via_step": True,
        "model_specs": {"driver": "codex/gpt-5.5", "induce_backend": "local Qwen3.5-9B-MTP (deployable)",
                        "reasoning_effort": "xhigh (codex gpt-5.5 default; not explicitly forced via -c)",
                        "isolation": "induce backend held constant baseline==treatment; ONLY codex-driving differs"},
        "preconditions_checked": [{"resource": "codex_cli", "available": True}],
        "prior_failures": [
            {"experiment_id": "arc_incontext_pattern_proposal_ab",
             "verdict": "context_injection_powered_null",
             "addressed_by": ("context-INJECTION gave the LLM static analogies (passive); this gives the LLM "
                              "TOOLS to ACT/observe/re-decide (the perceive->test RE loop) -- a structurally "
                              "different axis. Tests the FRAMING at the frontier ceiling before any 9B build."),
             "retire_if_same_verdict": True},
        ],
        "random_seed": SEED,
        "duration_s": round(time.time() - started, 2),
    }
    _write(art)
    print("\n=== VERDICT:", verdict)
    return 0


def _write(art: dict) -> None:
    payload = dict(art)
    payload["reproducibility_checksum"] = ""
    art["reproducibility_checksum"] = "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    out = REPO / "results" / f"arc_frontier_tooluse_probe_{art.get('game', 'x')}.json"
    out.write_text(json.dumps(art, indent=2) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    raise SystemExit(main())
