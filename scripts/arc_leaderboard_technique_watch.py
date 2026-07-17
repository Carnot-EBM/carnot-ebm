#!/usr/bin/env python3
"""Daily ARC-AGI-3 leaderboard technique watch.

Spec refs: n/a (operational tooling, not a research experiment).

WHY THIS EXISTS: operator directive (2026-07-15) to track what techniques/directions the top
ARC-AGI-3 leaderboard contenders are using, on an ongoing daily basis, so Carnot can learn from
or adopt genuinely useful ideas -- the way the 2026-06-20 one-off competitive-intel dive
(docs/research-notes/arc-leaderboard-competitive-intel-2026-06-20.md) found the CNN frame-change/
clickability predictor idea, which was subsequently integrated
(SUBMITTED_AGENT_CONFIG["frame_change_predictor_enabled"] = True). This script is the durable,
systemd-timer-driven mechanism that makes that a standing practice instead of a one-off dive: it
runs once daily independent of any interactive session, uses `codex exec` (same verified
web-search-capable pattern as scripts/arc_news_watch.py) to check the current leaderboard and the
top contenders' published code/writeups (this is a CODE COMPETITION -- milestone-eligible
entrants must open-source, per docs/research-notes/arc-agi3-news-watch.md's baseline: "Rules
confirmed: open-source required for prize eligibility"), and appends a dated entry to a durable
log only when it finds something genuinely new.

Sibling to scripts/arc_news_watch.py (which tracks competition RULES/ANNOUNCEMENTS/rank
movements) -- this script is scoped narrower and deeper: not "who is leading with what score" but
"HOW are they doing it, and is there anything here we should learn from." Two focused daily
checks, not one overloaded prompt.

This script does NOT interpret findings into an implementation decision or touch any live-path
code -- it is read-only research-gathering, matching "Operator-Only External Publication" and
CLAUDE.md's "ARC Live-Path Reachability Discipline" (source-reading is fine for OTHER teams'
already-published, already-public code; it is never applied to Carnot's own hidden-game
submission). Any "should we adopt X" decision stays with the operator or a future session that
reads this log, exactly like the 2026-06-20 dive did.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = REPO_ROOT / "ops" / ".arc_leaderboard_technique_watch_state.json"
LOG_PATH = REPO_ROOT / "docs" / "research-notes" / "arc-agi3-leaderboard-technique-watch.md"
# Investigating 5-10 contenders' individual code/writeups is meaningfully more work than the
# sibling arc_news_watch.py's 2-page announcement check (which uses 300s and works fine) -- a
# first real verification run (2026-07-15) timed out at 420s. Raised to 900s (15min); the systemd
# timer runs this unattended once daily, so a longer wall-clock budget costs nothing but the
# investigation's own runtime.
CODEX_TIMEOUT_S = 900
CODEX_MODEL = "gpt-5.6-sol"

KNOWN_BASELINE = """\
Known baseline as of 2026-07-15 (do not re-report these as new; prior one-off dive on 2026-06-20,
see docs/research-notes/arc-leaderboard-competitive-intel-2026-06-20.md for full detail):

- 2026-06-20 leaderboard: Tufa Labs led at 1.21 ("StochasticGoose" -- CNN frame-change/
  clickability predictor trained via RL). Field then: Blind Squirrel (state-graph + ResNet18
  value model), a "Persistent Memory BFS" public notebook (0.46, DQN + PrioritizedExperienceReplay
  + cross-game PersistentAEM action-effect memory + CBAM-attention-CNN value net + IDA*/BFS), a
  Hybrid BFS+CNN entry (hidden-field state-hash probing + level-to-level transfer). Carnot was at
  0.08 (bare-BFS, first submission).
- The winning paradigm as of 6/20 was a LEARNED action-effect model + RL/search + persistent
  memory -- NOT LLM world-model induction. Action efficiency (score = min(human/agent,1)^2) and
  cross-game learning were the two levers that separated leaders from the pack; generalization
  (transfer to unseen games) was the universally unsolved hard part even for the leader.
- ALREADY ADOPTED from that dive: a CNN frame-change/clickability predictor is now wired into
  Carnot's live path (SUBMITTED_AGENT_CONFIG["frame_change_predictor_enabled"] = True). Hidden-
  field state-hash probing was NOT yet adopted as of this baseline.
- ARC Prize Milestone #1 (deadline 2026-06-30, results published ~2026-07-07,
  https://arcprize.org/blog/arc-prize-2026-milestone-1): 1st Tufa Labs "The Duck" (agent-writes-
  code, Qwen 3.6 27B FP8 LOCAL, live REPL, multimodal perception). 2nd "Reki" (vision-LLM policy,
  Gemma-4-31B LOCAL, reflection memory, dead-signature action-skip). 3rd "forge" (Md Boktiar
  Mahbub Murad; vision-LLM policy, Gemma-4-31B, candidate generator + scoring arbiter). Note the
  #1 milestone winner ("The Duck") uses a DIFFERENT paradigm than the 6/20 public-leaderboard
  leader (StochasticGoose) -- milestone judging and the live public leaderboard are not the same
  ranking, and both are worth tracking.
- Milestone #2 deadline is 2026-09-30 (results likely published shortly after) -- no writeup yet
  as of this baseline.
- Public leaderboard as of 2026-07-14 (from the sibling arc-news-watch.md log): YUTO KOJIMA led at
  1.86, ahead of Tecnod8.AI (1.61); a three-way tie (Mathurin Ache, anngle, NoOneAhead) at 1.56 for
  third. These names/scores have not yet been technique-mined -- their code/writeups (if
  published) are a priority for the first run of this watch.
- Known caveat (from an earlier, now possibly-stale investigation): some leaderboard leaders have
  historically won via public-game source-reading exploits rather than genuine hidden-game
  generalization, since the public leaderboard's games overlap with games whose source is
  inspectable. ALWAYS check whether a given contender's approach reads game source / hardcodes
  per-game logic (an exploit, not a transferable technique) versus general-purpose methods that
  would plausibly generalize to a truly hidden game -- flag which kind each finding is.
"""

PROMPT_TEMPLATE = """\
Use web search to investigate the CURRENT top ARC-AGI-3 / ARC Prize 2026 Kaggle leaderboard
contenders and what techniques they are using.

Steps:
1. Check the current leaderboard: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard
2. For the top 5-10 teams, search for their published code, Kaggle notebooks, discussion posts,
   or writeups (this is a CODE competition -- open-source is required for prize eligibility, so
   milestone-eligible entrants should have public code somewhere: Kaggle kernels, GitHub, the
   competition discussion tab, or an ARC Prize blog writeup).
3. Also check https://arcprize.org/blog for any new milestone results or technique writeups.

{baseline}

{prior_findings_section}

Report ONLY genuinely new information not already covered by the baseline or prior findings
above:
- New leaderboard movement (only if a NEW team enters the top 5, or an existing top team's
  technique changes -- do not just re-report the same rank/score churn the sibling
  arc-news-watch.md log already tracks).
- Any NEWLY discovered or NEWLY published technique detail for a top contender: model/architecture
  choices, search strategy, learning approach (RL/supervised/etc.), perception method,
  cross-game/cross-level memory mechanisms, action-efficiency tricks, or anything else concrete
  and specific enough to potentially learn from.
- For each technique reported, explicitly flag: (a) does it read/exploit game source or hardcode
  per-game logic (an exploit, not a generalizable technique), or (b) is it a general-purpose
  method that would plausibly transfer to a genuinely unseen hidden game?
- If you find something that seems concretely adoptable and DIFFERENT from what Carnot already
  does (a verifier-routed search over a locally-hosted open-weight LLM generator + executable
  world-model induction, energy/verifier-based candidate ranking, no RL training), flag it
  explicitly as "POSSIBLE CARNOT LEVER" with a one-line reason why.

Cite the source URL for anything you report.

If there is nothing new beyond what is already known, reply with EXACTLY the single line:
NO_NEW_NEWS
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
            "# ARC-AGI-3 Leaderboard Technique Watch\n\n"
            "Daily automated check (systemd timer `arc-leaderboard-technique-watch.timer`, "
            "see `scripts/arc_leaderboard_technique_watch.py`) for what techniques/directions "
            "the top ARC-AGI-3 leaderboard contenders are taking, so Carnot can learn from or "
            "adopt genuinely useful ideas -- see the 2026-06-20 one-off dive "
            "(`docs/research-notes/arc-leaderboard-competitive-intel-2026-06-20.md`) that this "
            "watch makes a standing daily practice instead of an occasional manual check. "
            "Sibling to `docs/research-notes/arc-agi3-news-watch.md` (rules/announcements/rank "
            "movements) -- this log is scoped to HOW top contenders are achieving their scores, "
            "not just who is leading with what number. Entries below are appended, never "
            "rewritten, per the project's never-prune documentation discipline. This log is "
            "read-only research-gathering; it does not itself decide to adopt anything -- that "
            "stays an operator/session decision.\n\n"
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
