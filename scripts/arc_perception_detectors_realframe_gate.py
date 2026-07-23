#!/usr/bin/env python3
"""Real-frame integration gate for the entity+HUD perception detectors (REQ-ARC-WMTE-5833).

The unit tests prove the detector LOGIC on synthetic frames. This gate proves they FIRE ON REAL GAME
FRAMES: it steps each public game through the OFFLINE arcade (no LLM, no network, no quota) with a fixed
scripted exploration, gathers the agent's OWN (before, action, after) transitions, runs the detectors, and
reports what they recovered vs the source-diagnosed ground truth (REQ-ARC-WMTE-5831). Public-game frames are
stepped for offline dev validation only (authorized); the detectors themselves use ONLY observed transitions
(no source), so they are exactly what the live hidden-game agent would run.

Expected ground truth per game (from the 5831 diagnosis, for judging only -- NOT given to the detectors):
  bp35: fill HUD on the last row (move-budget bar).            lf52: fill HUD on row 0 (move counter).
  sc25: HUD on cols ~62/63 (mana meter). player moves 1-4.     tu93: deplete HUD on last row. player color ~9.
  ls20: player (color ~9) moves 1-4.                           cn04/r11l/ft09: pieces (harder; contrast).

inference_substrate: offline_arcade_live_agent_runtime_self_discovery_no_llm (pure env-stepping + detectors,
no GGUF load, no CUDA).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

GAMES = ["bp35", "lf52", "sc25", "tu93", "ls20", "cn04", "r11l", "ft09"]

# Source-diagnosed ground truth (REQ-ARC-WMTE-5831), for offline judging only.
GROUND_TRUTH = {
    "bp35": {"hud": "fill on last row (move-budget/lose bar)", "player": "yes (moves 3/4 under gravity)"},
    "lf52": {"hud": "fill on row 0 (move counter)", "player": "no discrete mover (block-merge)"},
    "sc25": {"hud": "cols ~62/63 (mana/move meter)", "player": "yes (avatar moves 1-4)"},
    "tu93": {"hud": "deplete on last row (step counter)", "player": "yes (token color ~9 moves 1-4)"},
    "ls20": {"hud": "(none prominent)", "player": "yes (avatar color ~9 moves 1-4)"},
    "cn04": {"hud": "(none prominent)", "player": "pieces move/rotate"},
    "r11l": {"hud": "(none prominent)", "player": "pieces (two-click move)"},
    "ft09": {"hud": "(none prominent)", "player": "no mover (click-only CSP)"},
}


def _click_cells(h: int, w: int) -> list[tuple[int, int]]:
    """A spread of logical (col, row) click targets to exercise any counter + unstick the agent."""
    return [
        (1, 1), (w // 3, h // 3), (w // 2, h // 2), (2 * w // 3, 2 * h // 3),
        (w - 2, h - 2), (0, h - 1), (w - 1, 0), (5, 5), (w // 4, 3 * h // 4), (3 * w // 4, h // 4),
    ]


def _gather(game: str):
    from arcengine import GameAction

    from carnot.agentic.arc_agi3_live_adapter import (
        _available_action_ids,
        _game_action,
        _game_over,
    )
    from carnot.agentic.arc_agi3_world_model import grid_of
    from carnot.agentic.arc_entity_hud_perception import Transition
    from carnot.agentic.arc_executable_world_model import detect_cell, to_logical
    from carnot.agentic.arc_solver_kit import frame_level, offline_arcade

    arc = offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    frame = env.reset()
    raw = grid_of(frame)
    cell = detect_cell(raw)
    logical = to_logical(raw, cell)
    h, w = logical.shape
    clicks = _click_cells(h, w)
    # ADAPTIVE plan: a repeating action pattern, but at each step we take the next PATTERN action that is
    # actually AVAILABLE this frame (per-game action sets differ -- a fixed plan under-explored r11l/ft09 to
    # 8 steps because most planned actions were unavailable). Guarantees directional coverage + periodic
    # clicks within whatever vocabulary the game exposes.
    pattern = [1, 2, 3, 4, 1, 2, 3, 4, 6, 6]
    max_steps = 60
    transitions: list[Transition] = []
    click_i = 0
    pat_i = 0
    steps = 0
    while steps < max_steps and not _game_over(frame):
        avail = set(_available_action_ids(frame) or [1, 2, 3, 4, 6])
        aid = None
        for _ in range(len(pattern)):
            cand = pattern[pat_i % len(pattern)]
            pat_i += 1
            if cand in avail:
                aid = cand
                break
        if aid is None:
            aid = next(iter(avail), None)
        if aid is None:
            break
        cx = cy = None
        data = None
        if aid == 6:
            cx, cy = clicks[click_i % len(clicks)]
            click_i += 1
            data = {"x": min(63, max(0, cx * cell + cell // 2)), "y": min(63, max(0, cy * cell + cell // 2))}
        before = to_logical(grid_of(frame), cell)
        frame = env.step(_game_action(GameAction, aid), data=data)
        after = to_logical(grid_of(frame), cell)
        transitions.append(Transition(before=before, action=aid, after=after, x=cx, y=cy))
        steps += 1
    return transitions, (h, w), int(frame_level(frame))


def main() -> int:
    from carnot.agentic.arc_entity_hud_perception import (
        detect_hud_registers,
        detect_mover,
        perceive_entities,
    )

    t0 = time.time()
    per_game = []
    for game in GAMES:
        try:
            trans, shape, lvl = _gather(game)
        except Exception as exc:  # noqa: BLE001 -- a per-game failure is a datum, not a crash
            per_game.append({"game": game, "error": repr(exc)[:200]})
            print(f"[{game}] ERROR: {exc!r}")
            continue
        hud = detect_hud_registers(trans)
        mover = detect_mover(trans)
        percept = perceive_entities(trans[-1].after, trans) if trans else None
        rec = {
            "game": game,
            "n_transitions": len(trans),
            "logical_shape": list(shape),
            "hud_bands": [
                {"axis": b.axis, "index": b.index, "direction": b.direction,
                 "changed_fraction": b.changed_fraction, "monotone_ratio": b.monotone_ratio}
                for b in hud
            ],
            "mover": None if mover is None else {"color": mover.color, "alignment": mover.alignment, "evidence": mover.evidence},
            "ground_truth": GROUND_TRUTH.get(game, {}),
            "perception_text": percept.text if percept else "",
        }
        per_game.append(rec)
        hud_s = ", ".join(f"{b.axis}{b.index}/{b.direction}" for b in hud) or "none"
        mv_s = f"color {mover.color} (align {mover.alignment}, ev {mover.evidence})" if mover else "none"
        print(f"[{game}] {len(trans)} trans | HUD: {hud_s} | mover: {mv_s}")
        print(f"        truth: HUD={GROUND_TRUTH[game]['hud']}; player={GROUND_TRUTH[game]['player']}")

    art = {
        "experiment": "outer_loop_arc_perception_detectors_realframe_gate",
        "experiment_id": "REQ-ARC-WMTE-5833",
        "run_date": "2026-07-23",
        "schema": "carnot.arc_perception_detectors_realframe_gate.v1",
        "title": "Do the entity+HUD perception detectors recover the known HUD band + player from REAL game frames (offline arcade, no LLM)?",
        "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        "verifier_is_oracle": False,
        "solve_provenance": "development_proxy",
        "random_seed": 5833,
        "reproducibility_checksum": "",
        "methodology_note": "Offline arcade stepping with a FIXED scripted exploration (deterministic, no LLM). Detectors use ONLY the agent's own observed transitions; source ground truth is used for offline judging only, never fed to the detectors. Public-game frames stepped for offline dev validation (authorized); never in the hidden submission.",
        "per_game": per_game,
        "duration_s": round(time.time() - t0, 2),
    }
    import hashlib

    art["reproducibility_checksum"] = "sha256:" + hashlib.sha256(json.dumps(art, sort_keys=True).encode()).hexdigest()
    out = ROOT / "results" / "outer_loop_arc_perception_detectors_realframe_gate_20260723.json"
    out.write_text(json.dumps(art, indent=2))
    print("wrote", out, f"({art['duration_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
