"""LIVE ARC-AGI-3 multi-game run: replay our 11 OFFLINE-reproduced game solutions
(13 levels) against the LIVE scored env under ONE scorecard. Mode-1 pure action
replay (env-match confirmed for r11l; this validates the other 10 live).

TWO-PHASE per the operator's choice (2026-06-17 "validate all 11 live, then submit"):
  DEFAULT (no --submit): VALIDATE — open a scorecard, play all 11 live, read the level
    each reaches, and DO NOT close it (an unclosed scorecard is NOT submitted; no
    leaderboard record). Reports which games env-match live.
  --submit: SUBMIT — same play-through, then CLOSE the scorecard (records the score on
    the leaderboard). Operator-gated, irreversible. External Publication is operator-only.

Reuses the metaharness's banked-trajectory loader so the live replay uses EXACTLY the
offline-reproduced action sequences (no re-solving, no verifier needed at submit-time).
Usage:
  .venv/bin/python scripts/arc3_live_submit.py            # VALIDATE live (no submission)
  .venv/bin/python scripts/arc3_live_submit.py --submit   # SUBMIT (close scorecard) — operator-gated
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from arcengine import GameAction

# the 11 reproduced games + their claimed (offline-reproduced) level. sc25 excluded
# (reproduces to L0 offline — would only waste live calls).
CLAIMED = {"r11l": 1, "lp85": 3, "ls20": 1, "wa30": 1, "cd82": 1, "sp80": 1,
           "su15": 1, "tu93": 1, "cn04": 1, "m0r0": 1, "sk48": 1}


def _load_metaharness():
    spec = importlib.util.spec_from_file_location(
        "mh", str(REPO / "scripts" / "arc3_replay_scorecard_metaharness.py"))
    mh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mh)  # type: ignore
    return mh


def _arc_api_key() -> str:
    out = subprocess.run(["sops", "-d", str(REPO / "secrets" / "arc_api.enc.yaml")],
                         capture_output=True, text=True, check=True).stdout
    import yaml
    return str(yaml.safe_load(out)["ARC_API_KEY"])


def online_arcade():
    from arc_agi import Arcade
    from arc_agi.base import OperationMode
    return Arcade(arc_api_key=_arc_api_key(), operation_mode=OperationMode.ONLINE,
                  environments_dir="", recordings_dir=str(REPO / "recordings" / "arc_live_submit"))


def resolve_game_id(arcade, short: str) -> str:
    for info in arcade.get_environments():
        gid = getattr(info, "game_id", getattr(info, "id", "")) or ""
        if str(gid).split("-", 1)[0] == short:
            return str(gid)
    return short


def replay_live(arcade, short: str, scorecard_id: str, actions: list[dict], mh) -> int:
    gid = resolve_game_id(arcade, short)
    env = arcade.make(gid, scorecard_id=scorecard_id)
    frame = env.reset()
    from carnot.agentic.arc_agi3_live_adapter import _levels_completed
    for a in actions:
        aid, data = mh.normalize(a)
        if aid is None:
            continue
        ge = getattr(GameAction, f"ACTION{aid}")
        frame = env.step(ge, data=data, reasoning={"policy": "offline_reproduced_replay"})
        if frame is None:
            break
    return _levels_completed(frame) if frame is not None else -1


def main(argv) -> int:
    submit = "--submit" in argv
    mh = _load_metaharness()
    mode = "SUBMIT (will CLOSE scorecard — leaderboard record)" if submit else "VALIDATE (no close, no submission)"
    print(f"== LIVE ARC-AGI-3 multi-game replay — mode: {mode} ==", flush=True)
    print(f"  games: {len(CLAIMED)}  claimed levels: {sum(CLAIMED.values())}", flush=True)

    arcade = online_arcade()
    scorecard_id = arcade.open_scorecard()
    print(f"  opened LIVE scorecard: {scorecard_id}", flush=True)

    rows, total, matched = [], 0, 0
    for short, claimed in CLAIMED.items():
        src = mh.RESOLVED_ARTIFACTS.get(short, mh.GAME_ARTIFACTS.get(short))
        actions = mh.load_actions(src) if src else []
        if not actions:
            rows.append({"game": short, "claimed": claimed, "live_level": None, "error": "no banked actions"})
            print(f"    {short:5} claimed L{claimed} -> NO banked actions (skip)", flush=True)
            continue
        t0 = time.time()
        try:
            lvl = replay_live(arcade, short, scorecard_id, actions, mh)
        except Exception as e:
            rows.append({"game": short, "claimed": claimed, "live_level": None, "error": repr(e)[:140]})
            print(f"    {short:5} claimed L{claimed} -> ERROR {repr(e)[:70]}", flush=True)
            continue
        ok = lvl >= claimed
        matched += int(ok)
        total += max(0, lvl)
        rows.append({"game": short, "claimed": claimed, "live_level": lvl, "env_match": ok})
        print(f"    {short:5} claimed L{claimed} -> LIVE L{lvl}  {'MATCH' if ok else 'MISMATCH'}  [{time.time()-t0:.0f}s]", flush=True)

    print(f"\n  LIVE TOTAL: {total} levels; {matched}/{len(CLAIMED)} games env-matched", flush=True)

    submitted = False
    if submit:
        card = arcade.close_scorecard(scorecard_id)
        submitted = card is not None
        print(f"  SUBMITTED: scorecard CLOSED -> leaderboard record ({type(card).__name__ if card else 'None'})", flush=True)
    else:
        print("  NOT submitted (scorecard left open; no leaderboard record). Re-run with --submit to record.", flush=True)

    out = REPO / "results" / "arc3_live_submit.json"
    out.write_text(json.dumps({
        "experiment": "arc3_live_submit", "mode": "submit" if submit else "validate",
        "scorecard_id": scorecard_id, "live_total_levels": total,
        "games_env_matched": matched, "games": len(CLAIMED),
        "claimed_total_levels": sum(CLAIMED.values()), "per_game": rows,
        "leaderboard_submitted": submitted, "run_date": "2026-06-17",
        "inference_substrate": "live_arc_agi3_env_multi_game_replay",
    }, indent=2))
    print(f"  wrote {out.relative_to(REPO)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
