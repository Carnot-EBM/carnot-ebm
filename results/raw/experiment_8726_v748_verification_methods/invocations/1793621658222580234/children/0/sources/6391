"""Live-vs-offline env-match probe — confirms our OFFLINE-derived solutions replay
on the LIVE env (so a Mode-1 replay submission is valid, no verifier needed at
submit-time). Replays r11l's offline-reproducible 4-action solution and reports
the level reached.

DEFAULT IS OFFLINE (zero quota, no network) — a dry-run that proves the replay
logic + that r11l reaches L1 on our local env. The ONLINE path is the single,
operator-greenlit live call: it PLAYS r11l live with the same actions and reads
the level. It opens a scorecard to play but does NOT close/submit it — this is an
env-match check, NOT a leaderboard submission (External Publication is operator-
only). One game, ~5 live calls (reset + 4 actions); minimal quota.

Usage:
  .venv/bin/python scripts/arc3_live_env_match_probe.py            # OFFLINE dry-run (safe)
  .venv/bin/python scripts/arc3_live_env_match_probe.py --online   # SINGLE live call (operator-greenlit only)
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from arcengine import GameAction
from carnot.agentic.arc_solver_kit import frame_level

GAME = "r11l"
SOLUTION = [  # r11l L1, offline-reproducible (results/experiment_4296 solve_trace)
    {"x": 7, "y": 36}, {"x": 34, "y": 20}, {"x": 27, "y": 59}, {"x": 42, "y": 20},
]
CLAIMED_LEVEL = 1


def _arc_api_key() -> str:
    out = subprocess.run(["sops", "-d", str(REPO / "secrets" / "arc_api.enc.yaml")],
                         capture_output=True, text=True, check=True).stdout
    import yaml
    return str(yaml.safe_load(out)["ARC_API_KEY"])


def offline_arcade():
    from arc_agi import Arcade
    from arc_agi.base import OperationMode
    return Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE,
                  environments_dir=str(REPO / "environment_files"))


def online_arcade():
    from arc_agi import Arcade
    from arc_agi.base import OperationMode
    return Arcade(arc_api_key=_arc_api_key(), operation_mode=OperationMode.ONLINE,
                  environments_dir="", recordings_dir=str(REPO / "recordings" / "arc_live_probe"))


def resolve_game_id(arcade) -> str:
    for info in arcade.get_environments():
        gid = getattr(info, "game_id", getattr(info, "id", "")) or ""
        if str(gid).split("-", 1)[0] == GAME:
            return str(gid)
    return GAME


def play(arcade) -> int:
    gid = resolve_game_id(arcade)
    env = arcade.make(gid, scorecard_id=arcade.open_scorecard())  # open to PLAY; NOT closed/submitted
    f = env.reset()
    for a in SOLUTION:
        f = env.step(GameAction.ACTION6, data={"x": a["x"], "y": a["y"]})
    return frame_level(f)


def main(argv) -> int:
    online = "--online" in argv
    mode = "ONLINE (single live call)" if online else "OFFLINE dry-run (zero quota)"
    print(f"== ARC live-vs-offline env-match probe: {GAME} L{CLAIMED_LEVEL} replay, mode={mode} ==")
    if not online:
        lvl = play(offline_arcade())
        print(f"  OFFLINE replay reached L{lvl} (expect {CLAIMED_LEVEL})")
        print(f"  dry-run {'PASS' if lvl >= CLAIMED_LEVEL else 'FAIL'} — replay logic validated, zero quota.")
        print("  To confirm offline==live, re-run with --online (operator-greenlit; one live game, no submission).")
        return 0 if lvl >= CLAIMED_LEVEL else 1
    print("  LIVE: opening online arcade (registered ARC_API_KEY), playing r11l, reading level. NOT submitting.")
    lvl = play(online_arcade())
    match = lvl >= CLAIMED_LEVEL
    print(f"  ONLINE replay reached L{lvl} (expect {CLAIMED_LEVEL})")
    print(f"  ENV-MATCH: {'CONFIRMED' if match else 'MISMATCH'} — offline solution {'replays' if match else 'does NOT replay'} live.")
    print(f"  => Mode-1 replay submissions are {'VALID (no verifier needed at submit-time)' if match else 'INVALID; must solve live (verifier needed)'}.")
    return 0 if match else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
