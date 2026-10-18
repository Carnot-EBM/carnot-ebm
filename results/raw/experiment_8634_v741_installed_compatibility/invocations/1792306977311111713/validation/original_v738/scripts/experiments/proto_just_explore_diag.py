"""proto_just_explore_diag.py — Diagnostic harness: just-explore HeuristicAgent vs our offline arcade.

Runs just-explore's HeuristicAgent UNMODIFIED (solving logic untouched) against the same
color-permuted variant-1 offline arcade used in exp4605, at two budgets:
  - budget 200  (same as our agent's exp4605 budget) -> isolates MECHANISM
  - budget 2000 (raised budget)                      -> isolates BUDGET

Our agent baseline: first_win_rate = 0.04 (only lp85 solves at budget 200, exp4605).

Artifact: results/proto_just_explore_diag.json
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
import traceback
import types
from pathlib import Path
from typing import Any

import numpy as np

# ─── Path constants ───────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
JE_ROOT = Path("/home/ianblenke/arc-sota-refs/arc-agi-3-just-explore")
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ─── 1. Load just-explore modules WITHOUT their broken __init__.py ─────────────
#     Their agents/__init__.py imports langgraph (not installed). Load each file
#     directly via importlib so we skip it.
def _load_je_modules() -> dict[str, Any]:
    """Load just-explore structs, tracing, recorder, agent, graph_explorer, heuristic_agent."""
    # Add JE root to sys.path for graph_explorer.py (top-level module)
    if str(JE_ROOT) not in sys.path:
        sys.path.insert(0, str(JE_ROOT))

    # Stub the agents package so sub-imports like 'from agents.structs import ...' work
    agents_pkg = types.ModuleType("agents")
    sys.modules["agents"] = agents_pkg

    modules: dict[str, Any] = {}
    for mod_name, rel_path in [
        ("agents.structs", "agents/structs.py"),
        ("agents.tracing", "agents/tracing.py"),
        ("agents.recorder", "agents/recorder.py"),
        ("agents.agent", "agents/agent.py"),
    ]:
        spec = importlib.util.spec_from_file_location(mod_name, JE_ROOT / rel_path)
        m = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        sys.modules[mod_name] = m
        spec.loader.exec_module(m)  # type: ignore[union-attr]
        setattr(agents_pkg, mod_name.split(".")[-1], m)
        modules[mod_name] = m

    # graph_explorer is a top-level module inside JE_ROOT
    spec_ge = importlib.util.spec_from_file_location(
        "graph_explorer", JE_ROOT / "graph_explorer.py"
    )
    ge_m = importlib.util.module_from_spec(spec_ge)  # type: ignore[arg-type]
    sys.modules["graph_explorer"] = ge_m
    spec_ge.loader.exec_module(ge_m)  # type: ignore[union-attr]
    modules["graph_explorer"] = ge_m

    spec_ha = importlib.util.spec_from_file_location(
        "agents.heuristic_agent", JE_ROOT / "agents/heuristic_agent.py"
    )
    ha_m = importlib.util.module_from_spec(spec_ha)  # type: ignore[arg-type]
    sys.modules["agents.heuristic_agent"] = ha_m
    spec_ha.loader.exec_module(ha_m)  # type: ignore[union-attr]
    modules["agents.heuristic_agent"] = ha_m

    return modules


JE_MODS = _load_je_modules()
JEFrameData = JE_MODS["agents.structs"].FrameData
JEGameAction = JE_MODS["agents.structs"].GameAction
JEGameState = JE_MODS["agents.structs"].GameState
HeuristicAgent = JE_MODS["agents.heuristic_agent"].HeuristicAgent

# ─── 2. Load Carnot agentic stack ─────────────────────────────────────────────
from carnot.agentic import arc_solver_kit as kit  # noqa: E402
from carnot.agentic.arc_agi3_live_adapter import (  # noqa: E402
    _available_action_ids,
    _levels_completed,
)
from carnot.agentic.arc_agi3_world_model import grid_of  # noqa: E402
from carnot.agentic.arc_variant_generator import VariantEnv  # noqa: E402
from arcengine import GameAction as OurGameAction  # noqa: E402

# ─── 3. Public game list ───────────────────────────────────────────────────────
def _public_games() -> list[str]:
    env_dir = REPO_ROOT / "environment_files"
    if not env_dir.is_dir():
        return []
    return sorted(p.name for p in env_dir.iterdir() if p.is_dir())


# ─── 4. Frame conversion: our FrameDataRaw -> JE FrameData ────────────────────
def _our_raw_to_je_fd(
    raw: Any,
    game_id: str,
    start_level: int,
) -> JEFrameData:
    """Convert a Carnot FrameDataRaw to a just-explore FrameData.

    The JE agent expects:
      - frame: list[list[list[int]]] — layers x H x W
      - state: GameState enum (NOT_FINISHED / WIN / GAME_OVER / NOT_PLAYED)
      - score: int (0..254); we map levels_completed here
      - available_actions: list[int]

    We use one layer (the 64x64 grid from our env) wrapped in a list.
    Level detection: WIN when levels_completed > start_level (i.e. first level-up).
    """
    grid = grid_of(raw)  # (64, 64) int16
    # JE expects layers x H x W — we have exactly 1 layer
    frame_3d = [grid.tolist()]

    raw_state = str(getattr(raw, "state", "") or "").upper()
    lc = _levels_completed(raw)

    if lc > start_level:
        je_state = JEGameState.WIN
    elif "GAME_OVER" in raw_state or "LOSE" in raw_state:
        je_state = JEGameState.GAME_OVER
    elif "NOT_PLAYED" in raw_state and lc == 0:
        je_state = JEGameState.NOT_PLAYED
    else:
        je_state = JEGameState.NOT_FINISHED

    avail = _available_action_ids(raw)  # list[int], excludes 0 (RESET)

    return JEFrameData(
        game_id=game_id,
        frame=frame_3d,
        state=je_state,
        score=lc,
        available_actions=avail,
    )


# ─── 5. Action conversion: JE GameAction -> our OurGameAction + data ──────────
def _je_action_to_ours(je_action: JEGameAction) -> tuple[str, OurGameAction, dict | None]:
    """Convert a JE GameAction to our (label_str, OurGameAction, data_or_None).

    Returns:
      label_str: human-readable JSON for replay logging
      our_ga: the OurGameAction enum member to pass to env.step
      data: dict with x/y for ACTION6, else None
    """
    aid = je_action.value  # int 0..6

    if aid == 0:
        return "RESET", OurGameAction.RESET, None

    # ACTION6 is complex (click with x, y)
    if aid == 6:
        ad = je_action.action_data  # ComplexAction
        x, y = int(ad.x), int(ad.y)
        return json.dumps({"action": 6, "x": x, "y": y}), OurGameAction.ACTION6, {"x": x, "y": y}

    # ACTION1..5 are simple (no coordinates)
    our_ga = getattr(OurGameAction, f"ACTION{aid}")
    return json.dumps({"action": aid}), our_ga, None


# ─── 6. Run one game at one budget ────────────────────────────────────────────
def run_one_game(
    game_id: str,
    budget: int,
    arc: Any,
) -> dict:
    """Run HeuristicAgent on one game (variant 1) for up to `budget` actions.

    Returns a result dict with keys:
      game, budget, reached_level, solved, actions_used,
      actions_to_first_levelup, adapter_failed, adapter_error
    """
    result: dict = {
        "game": game_id,
        "budget": budget,
        "reached_level": 0,
        "solved": False,
        "actions_used": 0,
        "actions_to_first_levelup": None,
        "adapter_failed": False,
        "adapter_error": None,
        "action_labels": [],  # for potential reproduction gate
    }

    try:
        # Build the env (variant 1 = color-permuted)
        sc = arc.open_scorecard()
        base_env = arc.make(game_id, scorecard_id=sc)
        env = VariantEnv(base_env, game_id, 1)
        raw = env.reset()
        start_level = _levels_completed(raw)

        # Instantiate agent (no network, no recording)
        agent = HeuristicAgent(
            card_id="diag_card",
            game_id=game_id,
            agent_name="just_explore_diag",
            ROOT_URL="http://localhost:0",  # never used — we drive the loop
            record=False,
        )
        agent.MAX_ACTIONS = budget
        # Suppress the 0.31s per-action API rate-limit sleep.
        # This is adapter configuration (the rate-limit exists to avoid hammering the live API),
        # not solving logic. The solving logic (choose_action/is_done) is untouched.
        agent.minimal_step_time = 0.0

        # Agent expects an initial NOT_PLAYED frame to kick off with RESET
        je_fd_init = JEFrameData(
            game_id=game_id,
            frame=[],
            state=JEGameState.NOT_PLAYED,
            score=0,
            available_actions=[],
        )
        je_frames: list[JEFrameData] = [je_fd_init]

        max_lc = start_level
        prev_score = 0  # mirrors their main() 'score' variable for level_up detection

        for step in range(budget):
            latest_je = je_frames[-1]

            # Check done
            if agent.is_done(je_frames, latest_je):
                break

            # Choose action (UNMODIFIED just-explore logic)
            try:
                je_action = agent.choose_action(je_frames, latest_je)
            except Exception as exc:
                # Mirrors their main() error handling: set failed/level_up and retry last action
                agent.failed = True
                agent.level_up = True
                je_action = agent.last_action_object

            # Convert to our action
            label, our_ga, data = _je_action_to_ours(je_action)
            result["action_labels"].append(label)

            # Execute in our env
            if our_ga == OurGameAction.RESET:
                raw = env.reset()
                start_level_new = _levels_completed(raw)
                # After reset, update start_level only if we haven't scored yet
                if max_lc == 0:
                    start_level = start_level_new
            else:
                raw = env.step(our_ga, data=data)

            lc = _levels_completed(raw)
            new_score = lc  # their score == our levels_completed

            # Level-up detection: mirrors their main() loop
            if new_score > prev_score:
                agent.level_up = True
                agent.status_bar_mask = None
            elif agent.status_bar_mask is not None:
                agent.level_up = False
            prev_score = new_score

            # Build JE frame from our raw
            je_fd = _our_raw_to_je_fd(raw, game_id, start_level)
            # Append to agent's internal frames (so the graph explorer can track nodes)
            je_frames.append(je_fd)
            agent.frames.append(je_fd)
            agent.action_counter = step + 1
            if je_fd.guid:
                agent.guid = je_fd.guid

            max_lc = max(max_lc, lc)
            if lc > start_level and result["actions_to_first_levelup"] is None:
                result["actions_to_first_levelup"] = step + 1

            if je_fd.state == JEGameState.WIN:
                # agent.is_done will catch this next iteration
                break

        result["actions_used"] = agent.action_counter
        result["reached_level"] = max(0, max_lc - start_level)
        result["solved"] = result["reached_level"] >= 1

    except Exception as exc:
        result["adapter_failed"] = True
        result["adapter_error"] = traceback.format_exc()

    return result


# ─── 7. Main diagnostic run ───────────────────────────────────────────────────
def main() -> None:
    t0 = time.time()
    random_seed = 4730

    games = _public_games()
    print(f"Public games ({len(games)}): {games}")
    budgets = [200, 2000]

    arc = kit.offline_arcade()

    # Smoke test: lp85 at budget 200 first
    print("\n=== SMOKE TEST: lp85 @ budget 200 ===")
    smoke = run_one_game("lp85", 200, arc)
    print(
        f"  reached_level={smoke['reached_level']} solved={smoke['solved']} "
        f"actions_used={smoke['actions_used']} "
        f"adapter_failed={smoke['adapter_failed']}"
    )
    if smoke["adapter_failed"]:
        print("  SMOKE FAILED:\n", smoke["adapter_error"])
        print("  Aborting.")
        sys.exit(1)
    print("  Smoke passed.")

    # Full run
    all_results: list[dict] = []
    per_budget: dict[int, dict] = {}

    for budget in budgets:
        print(f"\n=== Budget {budget} ===")
        budget_results: list[dict] = []
        solved_games: list[str] = []
        failed_games: list[str] = []
        actions_to_first: list[int] = []

        for game in games:
            print(f"  {game} @ {budget}...", end=" ", flush=True)
            r = run_one_game(game, budget, arc)
            all_results.append(r)
            budget_results.append(r)

            if r["adapter_failed"]:
                failed_games.append(game)
                print(f"ADAPTER_FAILED")
            elif r["solved"]:
                solved_games.append(game)
                atfl = r["actions_to_first_levelup"]
                actions_to_first.append(atfl if atfl is not None else budget)
                print(f"SOLVED (level={r['reached_level']}, actions={r['actions_used']}, first_levelup={atfl})")
            else:
                print(f"no_solve (level={r['reached_level']}, actions={r['actions_used']})")

        games_ran_cleanly = len(games) - len(failed_games)
        first_win_rate = len(solved_games) / len(games) if games else 0.0
        median_atfl = (
            float(np.median(actions_to_first)) if actions_to_first else None
        )

        per_budget[budget] = {
            "budget": budget,
            "games_ran_cleanly": games_ran_cleanly,
            "games_adapter_failed": failed_games,
            "solved_games": solved_games,
            "first_win_rate": round(first_win_rate, 4),
            "n_solved": len(solved_games),
            "n_total": len(games),
            "median_actions_to_first_levelup": median_atfl,
        }
        print(
            f"\n  Budget {budget} summary: "
            f"first_win_rate={first_win_rate:.4f} ({len(solved_games)}/{len(games)}) "
            f"clean={games_ran_cleanly} failed={len(failed_games)}"
        )
        print(f"  Solved: {solved_games}")
        if failed_games:
            print(f"  Failed adapter: {failed_games}")

    duration_s = round(time.time() - t0, 2)

    our_baseline = 0.04
    je_b200 = per_budget[200]["first_win_rate"]
    je_b2000 = per_budget[2000]["first_win_rate"]
    mechanism_delta = round(je_b200 - our_baseline, 4)
    budget_delta = round(je_b2000 - je_b200, 4)

    # Build artifact payload
    payload = {
        "our_baseline_first_win_rate": our_baseline,
        "our_baseline_note": "exp4605: only lp85 solves at budget 200, variant 1 color-permuted",
        "just_explore_first_win_rate_b200": je_b200,
        "just_explore_first_win_rate_b2000": je_b2000,
        "mechanism_delta": mechanism_delta,
        "budget_delta": budget_delta,
        "per_budget": per_budget,
        "all_game_results": all_results,
        "n_games_total": len(games),
        "games_tested": games,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "verifier_is_oracle": False,
        "solve_provenance": "development_proxy",
        "random_seed": random_seed,
        "duration_s": duration_s,
        "minimal_step_time_override": 0.0,
        "adapter_notes": (
            "minimal_step_time set to 0.0 (API rate-limit suppressed for offline env; "
            "solving logic choose_action/is_done unmodified). "
            "JE HeuristicAgent.MAX_ACTIONS set per budget. "
            "level_up detection mirrors JE main() loop (new_score > prev_score). "
            "These are PUBLIC games — just-explore may be tuned for them (caveat)."
        ),
        "fairness_caveat": (
            "just-explore was trained/tuned on the 25 public ARC-AGI-3 games. "
            "Our agent's 0.04 was measured on variant-1 (color-permuted) copies of the same games. "
            "This comparison is mechanically fair (same env, same variant) but just-explore "
            "may have implicit public-game priors baked into its segmentation/exploration logic. "
            "The hidden Kaggle games are the only unbiased comparison; this is a development diagnostic."
        ),
    }

    # Reproducibility checksum (SHA-256 of payload excluding the checksum itself)
    payload_for_hash = {k: v for k, v in payload.items() if k != "reproducibility_checksum"}
    payload_json_bytes = json.dumps(payload_for_hash, sort_keys=True, default=str).encode()
    chksum = hashlib.sha256(payload_json_bytes).hexdigest()
    payload["reproducibility_checksum"] = chksum

    # honest_verdict
    if mechanism_delta > 0:
        verdict = (
            f"complete: just-explore mechanism wins (b200 {je_b200:.2f} vs ours {our_baseline:.2f}, "
            f"delta={mechanism_delta:+.2f}); budget bonus delta={budget_delta:+.2f}"
        )
    elif mechanism_delta == 0:
        verdict = (
            f"complete: mechanism parity (both b200={je_b200:.2f}={our_baseline:.2f}); "
            f"budget bonus delta={budget_delta:+.2f}"
        )
    else:
        verdict = (
            f"complete: just-explore b200={je_b200:.2f} < our baseline {our_baseline:.2f} "
            f"(mechanism_delta={mechanism_delta:+.2f}); budget bonus delta={budget_delta:+.2f}"
        )
    payload["honest_verdict"] = verdict

    out_path = RESULTS_DIR / "proto_just_explore_diag.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n=== RESULTS ===")
    print(f"  Our baseline (exp4605):         first_win_rate = {our_baseline:.4f}")
    print(f"  just-explore @ budget 200:      first_win_rate = {je_b200:.4f}")
    print(f"  just-explore @ budget 2000:     first_win_rate = {je_b2000:.4f}")
    print(f"  mechanism_delta (b200 - ours):  {mechanism_delta:+.4f}")
    print(f"  budget_delta (b2000 - b200):    {budget_delta:+.4f}")
    print(f"  Duration: {duration_s}s")
    print(f"  Artifact: {out_path}")
    print(f"  Verdict: {verdict}")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    main()
