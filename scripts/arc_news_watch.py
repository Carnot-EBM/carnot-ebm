#!/usr/bin/env python3
"""Daily ARC-AGI-3 competition news watch.

Spec refs: n/a (operational tooling, not a research experiment).

WHY THIS EXISTS: the project has a standing floor of ARC-AGI-3 work through
the November 2026 Kaggle submission deadline (see CLAUDE.md "ARC-AGI-3
November-Submission Standing Floor"). The operator asked (2026-07-11) to be
kept apprised of competition announcements -- rule changes, new games, and
milestone results -- without having to remember to ask every session. This
script is the durable, systemd-timer-driven mechanism: it runs once daily
independent of any interactive Claude Code session, uses `codex exec`
(verified to have real web-search tool access, not just training-data
recall -- see the 2026-07-11 verification note in
docs/research-notes/arc-agi3-news-watch.md) to check the two authoritative
sources (the ARC Prize blog and the Kaggle competition page), and appends a
dated entry to the durable log only when it finds something the previous
run had not already recorded. A "checked, nothing new" run still logs its
timestamp (so a human/agent reader can tell the watch is alive) but keeps
the entry to one line instead of repeating the full state every day.

This script does NOT interpret findings or take action -- it is read-only
research-gathering, matching "Operator-Only External Publication" and the
general principle that competition-facing decisions are the operator's,
not an autonomous script's.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = REPO_ROOT / "ops" / ".arc_news_watch_state.json"
LOG_PATH = REPO_ROOT / "docs" / "research-notes" / "arc-agi3-news-watch.md"
CODEX_TIMEOUT_S = 300
CODEX_MODEL = "gpt-5.6-sol"

KNOWN_BASELINE = """\
Known baseline as of 2026-07-11 (do not re-report these as new):
- ARC-AGI-3 launched 2026-03-25 (fireside chat, Chollet + Altman, YC HQ).
- 135 total environments: 25 Public Demo, 55 Semi-Private, 55 Fully Private.
- ARC Prize 2026 total purse $850K: Grand Prize $700K (100% score), $75K top-score
  pool, $75K milestone-prize pool.
- Milestone #1 deadline 2026-06-30, results published ~2026-07-07
  (https://arcprize.org/blog/arc-prize-2026-milestone-1): $37.5K awarded.
  1st Tufa Labs "The Duck" (agent-writes-code, Qwen 3.6 27B FP8 local,
  live REPL, multimodal perception). 2nd "Reki" (vision-LLM policy,
  Gemma-4-31B local, reflection memory, dead-signature action-skip). 3rd
  "forge" (Md Boktiar Mahbub Murad; vision-LLM policy, Gemma-4-31B,
  candidate generator + scoring arbiter).
- Milestone #2 deadline 2026-09-30 ($37.5K, same 1st/2nd/3rd split) -- no
  rule-change details published yet as of 2026-07-11.
- Final submission deadline 2026-11-02; results announced 2026-12-04.
- Rules confirmed: open-source required for prize eligibility; no internet
  access during evaluation; hardware/compute limits provided at launch.
"""

PROMPT_TEMPLATE = """\
Use web search to check for ARC-AGI-3 / ARC Prize 2026 competition news.
Check these sources: https://arcprize.org/blog and
https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3 (and its
discussion tab if reachable).

{baseline}

{prior_findings_section}

Report ONLY genuinely new information not already covered by the baseline
or prior findings above: rule changes, new or modified games, milestone
results, leaderboard shakeups, submission-format changes, or other official
announcements. Cite the source URL for anything you report.

If there is nothing new beyond what is already known, reply with EXACTLY
the single line: NO_NEW_NEWS
Do not pad the response, do not repeat the baseline back, do not speculate.
"""


def _load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"last_findings": "", "last_checked": None, "history_count": 0}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2) + "\n")


def _run_codex_check(prior_findings: str) -> tuple[str, int]:
    prior_section = (
        f"Findings already reported in the previous check:\n{prior_findings}"
        if prior_findings
        else "This is the first check -- no prior findings recorded yet."
    )
    prompt = PROMPT_TEMPLATE.format(baseline=KNOWN_BASELINE, prior_findings_section=prior_section)
    try:
        proc = subprocess.run(
            [
                "codex",
                "exec",
                "-m",
                CODEX_MODEL,
                "--dangerously-bypass-approvals-and-sandbox",
                "--color",
                "never",
                "--cd",
                str(REPO_ROOT),
                prompt,
            ],
            capture_output=True,
            text=True,
            timeout=CODEX_TIMEOUT_S,
            cwd=REPO_ROOT,
        )
    except subprocess.TimeoutExpired:
        return "CHECK_TIMED_OUT", 1
    except FileNotFoundError:
        return "CODEX_CLI_NOT_FOUND", 1
    if proc.returncode != 0:
        return f"CODEX_EXIT_{proc.returncode}: {proc.stderr[-500:]}", proc.returncode
    return proc.stdout.strip(), 0


def _append_log_entry(*, findings: str, is_new: bool, error: str | None) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LOG_PATH.exists():
        LOG_PATH.write_text(
            "# ARC-AGI-3 Competition News Watch\n\n"
            "Daily automated check (systemd timer `arc-news-watch.timer`, "
            "see `scripts/arc_news_watch.py`) for ARC Prize / ARC-AGI-3 "
            "competition announcements, ahead of the November 2026 Kaggle "
            "submission deadline. Entries below are appended, never "
            "rewritten, per the project's never-prune documentation "
            "discipline.\n\n"
        )
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    with LOG_PATH.open("a") as f:
        if error:
            f.write(f"## {timestamp} -- check failed\n\n{error}\n\n")
        elif is_new:
            f.write(f"## {timestamp} -- NEW\n\n{findings}\n\n")
        else:
            f.write(f"## {timestamp} -- checked, nothing new\n\n")


def main() -> int:
    state = _load_state()
    findings, exit_code = _run_codex_check(state.get("last_findings", ""))

    if exit_code != 0:
        _append_log_entry(findings="", is_new=False, error=findings)
        return exit_code

    is_new = findings != "NO_NEW_NEWS" and bool(findings.strip())
    _append_log_entry(findings=findings, is_new=is_new, error=None)

    if is_new:
        state["last_findings"] = findings
    state["last_checked"] = datetime.now(UTC).isoformat()
    state["history_count"] = int(state.get("history_count", 0)) + 1
    _save_state(state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
