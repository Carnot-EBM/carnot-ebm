#!/usr/bin/env python3
"""ARC live-agent self-solve adversarial audit (Layer 2 -- milestone-close hostile review).

WHY (2026-06-22 operator directive, 2nd recurrence): "Update the adversarial agent to prevent this from
happening again ... It needs to be aggressively caught and stopped. We want to help the live agent find
ways of solving hidden games on its own ... based on its own attempts and RE of the game."

The ARC-AGI-3 deliverable is a LIVE agent that DISCOVERS solves to HIDDEN games on its own -- from its OWN
attempts + runtime RE -- NOT a human/outer-loop reverse-engineering the game (reading its source, running
an exhaustive offline ground-truth BFS, hand-building a per-game model/adapter), and NOT a parallel solver
the live agent cannot reach. Twice an outer-loop session has built an off-path solver and "solved" a game
the live agent already solved.

This is Layer 2 of the defense (the aggressive ones run earlier):
  * Layer 1a (commit-time HARD STOP): scripts/arc_orphan_solver_lint.py -- a solver-like module the live
    agent can't reach blocks the commit.
  * Layer 1b (per-artifact mechanical): adversarial_verify.check_arc_outer_loop_solve -- flags solve
    artifacts that are outer-loop / off-path / duplicate / undeclared-provenance.
  * Layer 2 (this, milestone-close): a MECHANICAL pre-pass (always runs) + an optional hostile LLM review
    that catches the SUBTLE cases the mechanical layers miss (an artifact that looks self-discovered but
    leans on hidden outer-loop RE; a "new" solve that is really a duplicate; framing drift).
  * Layer 3: CLAUDE.md "ARC Live-Path Reachability Discipline" -- the design-time contract.

The audit NEVER edits anything -- it writes ops/arc_self_solve_audit_report.md for the operator. Non-fatal.

Usage:
  python scripts/arc_self_solve_audit.py [--model claude|gemini] [--since-days 7]
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "ops" / "arc_self_solve_audit_report.md"
LIVE_ENTRYPOINTS = ("scripts/arc_loop_solve.py", "python/carnot/agentic/arc_competition_agent.py")

OUTER_LOOP_INPUT_FLAGS = (
    "used_env_source",
    "read_game_source",
    "offline_ground_truth_bfs",
    "exhaustive_bfs_calibration",
    "hand_calibrated_per_game",
)

HOSTILE_PROMPT = """You are a HOSTILE reviewer guarding ONE principle for the Carnot ARC-AGI-3 project:

  The deliverable is a LIVE agent that DISCOVERS solves to HIDDEN games ON ITS OWN -- from its OWN
  attempts + runtime reverse-engineering. It is NOT a human / outer-loop reverse-engineering a game
  (reading the game's source code, running an exhaustive offline ground-truth BFS, hand-building a
  per-game model or adapter) and NOT a parallel solver the live agent cannot reach.

Below is a MECHANICAL pre-pass (reachability of solver modules + recent ARC solve artifacts and their
declared provenance) plus the two live entrypoints. For EACH recent ARC solve artifact, decide:

  SELF_DISCOVERY_ADVANCE -- the live agent advanced via its own attempts/runtime RE (good)
  OUTER_LOOP_RE          -- a human/outer-loop reverse-engineered it (read source / offline BFS /
                            hand-built per-game) -- the anti-pattern, even if dressed as a result
  OFF_PATH               -- solved via a mechanism not reachable from the live entrypoints
  DUPLICATE              -- re-solves a level the registry already records (no new live capability)
  UNCLEAR                -- cannot tell from the evidence

Be aggressive: if an artifact claims a solve but does NOT clearly show the LIVE agent did it from its own
attempts, say so. Output: a TL;DR verdict line, then per-artifact {verdict, evidence, recommended action},
then a "Pattern watch" note on any drift toward outer-loop solving. Keep it tight."""


def call_claude(prompt: str, body: str, model: str = "claude-opus-4-8") -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            ["claude", "--model", model, "--effort", "max", "--print", f"{prompt}\n\n---\n{body}"],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
            cwd=PROJECT_ROOT,
        )
        return (proc.returncode == 0, proc.stdout if proc.returncode == 0 else proc.stderr[:300])
    except Exception as exc:
        return False, str(exc)


def call_gemini(prompt: str, body: str, model: str = "gemini-3.1-pro-preview") -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            ["gemini", "--model", model, "--yolo", "-p", f"{prompt}\n\n---\n{body}"],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
            cwd=PROJECT_ROOT,
        )
        return (proc.returncode == 0, proc.stdout if proc.returncode == 0 else proc.stderr[:300])
    except Exception as exc:
        return False, str(exc)


def call_codex(prompt: str, body: str, model: str = "gpt-5.5") -> tuple[bool, str]:
    """Codex (gpt-5.5) hostile reviewer — quota-conserve path (mirrors the conductor's codex exec
    pattern; prompt on stdin via `-`). Added 2026-06-30 for the Claude-quota-conserve window."""
    try:
        proc = subprocess.run(
            ["codex", "exec", "--dangerously-bypass-approvals-and-sandbox", "--color", "never",
             "--model", model, "--cd", str(PROJECT_ROOT), "--ephemeral", "-"],
            input=f"{prompt}\n\n---\n{body}",
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
            cwd=PROJECT_ROOT,
        )
        return (proc.returncode == 0, proc.stdout if proc.returncode == 0 else proc.stderr[:300])
    except Exception as exc:
        return False, str(exc)


def _reachability() -> str:
    """Run the orphan-solver lint and capture its verdict (the live-path reachability pre-pass)."""
    try:
        p = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "arc_orphan_solver_lint.py")],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            cwd=PROJECT_ROOT,
        )
        return f"(exit {p.returncode})\n{p.stdout.strip()}"
    except Exception as exc:
        return f"(reachability lint failed: {exc})"


def _recent_solve_artifacts(since_days: int) -> list[dict]:
    cutoff = time.time() - since_days * 86400
    out = []
    for path in glob.glob(str(PROJECT_ROOT / "results" / "**" / "*.json"), recursive=True):
        try:
            if Path(path).stat().st_mtime < cutoff:
                continue
            d = json.load(open(path))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        if d.get("offline_reproduced") is not True or not isinstance(d.get("game"), str):
            continue
        lvl = next(
            (
                d[k]
                for k in ("reproduced_levels", "reached_level", "levels_completed")
                if isinstance(d.get(k), (int, float)) and not isinstance(d.get(k), bool)
            ),
            None,
        )
        if lvl is None or lvl < 1:
            continue
        out.append(
            {
                "artifact": str(Path(path).relative_to(PROJECT_ROOT)),
                "game": d.get("game"),
                "level": lvl,
                "solve_provenance": d.get("solve_provenance", "<<UNDECLARED>>"),
                "honest_verdict": str(d.get("honest_verdict", ""))[:160],
                "outer_loop_inputs_declared": [
                    k for k in OUTER_LOOP_INPUT_FLAGS if d.get(k) is True
                ],
                "mode": d.get("mode"),
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=("claude", "gemini", "codex", "none"), default="claude")
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--since-days", type=int, default=7)
    args = ap.parse_args()

    reach = _reachability()
    arts = _recent_solve_artifacts(args.since_days)

    # MECHANICAL findings (always available, no LLM needed) -------------------------------------------
    mech = []
    for a in arts:
        prov = a["solve_provenance"]
        if prov == "<<UNDECLARED>>":
            mech.append(
                f"- {a['artifact']} ({a['game']} L{a['level']}): UNDECLARED solve_provenance -> declare it"
            )
        elif prov == "outer_loop_re":
            mech.append(
                f"- {a['artifact']} ({a['game']} L{a['level']}): solve_provenance=outer_loop_re -> NOT a live-agent solve"
            )
        if a["outer_loop_inputs_declared"]:
            mech.append(
                f"- {a['artifact']}: declares outer-loop-only inputs {a['outer_loop_inputs_declared']}"
            )

    body_lines = [
        "LIVE ENTRYPOINTS (a solver must be reachable from one of these):",
        *[f"  - {e}" for e in LIVE_ENTRYPOINTS],
        "",
        "REACHABILITY PRE-PASS (scripts/arc_orphan_solver_lint.py):",
        reach,
        "",
        f"RECENT ARC SOLVE ARTIFACTS (last {args.since_days}d): {len(arts)}",
        json.dumps(arts, indent=2),
    ]
    body = "\n".join(body_lines)

    out = [
        "# ARC live-agent self-solve audit",
        "",
        "Generated by `scripts/arc_self_solve_audit.py` (Layer 2; advisory -- never edits anything).",
        "Principle: the live agent must self-discover hidden-game solves from its OWN attempts + runtime RE.",
        "",
        "## Mechanical pre-pass",
        "",
        "### Live-path reachability",
        "```",
        reach,
        "```",
        "",
        "### Recent solve artifacts -- mechanical findings",
        *(mech or ["- (none flagged mechanically)"]),
        "",
    ]

    if args.model != "none":
        caller = {"claude": call_claude, "gemini": call_gemini, "codex": call_codex}[args.model]
        kwargs = {"model": args.model_name} if args.model_name else {}
        ok, resp = caller(HOSTILE_PROMPT, body, **kwargs)
        out += [
            "## Hostile LLM review",
            "",
            resp if ok else f"(LLM review unavailable: {resp})",
            "",
        ]
    else:
        out += ["## Hostile LLM review", "", "(skipped: --model none)", ""]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(out))
    print(
        f"ARC self-solve audit -> {REPORT_PATH.relative_to(PROJECT_ROOT)} "
        f"({len(arts)} recent solve artifacts, {len(mech)} mechanical findings)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
