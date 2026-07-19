# ARC-AGI-3 Leaderboard Technique Watch

Daily automated check (systemd timer `arc-leaderboard-technique-watch.timer`, see `scripts/arc_leaderboard_technique_watch.py`) for what techniques/directions the top ARC-AGI-3 leaderboard contenders are taking, so Carnot can learn from or adopt genuinely useful ideas -- see the 2026-06-20 one-off dive (`docs/research-notes/arc-leaderboard-competitive-intel-2026-06-20.md`) that this watch makes a standing daily practice instead of an occasional manual check. Sibling to `docs/research-notes/arc-agi3-news-watch.md` (rules/announcements/rank movements) -- this log is scoped to HOW top contenders are achieving their scores, not just who is leading with what number. Entries below are appended, never rewritten, per the project's never-prune documentation discipline. This log is read-only research-gathering; it does not itself decide to adopt anything -- that stays an operator/session decision.

## 2026-07-15 07:50 UTC -- check failed

CHECK_TIMED_OUT

## 2026-07-15 08:08 UTC -- NEW

- **Dinesh Kumar Thiyagarajan — current #7 at 1.50:** A team member’s public [`arc-solutions` dataset](https://www.kaggle.com/datasets/dineshkumar0705/arc-solutions) contains exact scripts keyed by game ID and level: 1,844 actions covering 97 levels across 14 public games. These include explicit `ACTION6` click coordinates and fixed action sequences. The current leaderboard submission is private, so its use there is unconfirmed. ([Leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard))

  **Classification: (a) EXPLOIT/HARDCODE.** This is per-game/per-level replay logic, not a general-purpose method for unseen hidden games. No evidence establishes that it reads hidden game source.

## 2026-07-15 13:45 UTC -- check failed

CHECK_TIMED_OUT

## 2026-07-16 13:41 UTC -- checked, nothing new

## 2026-07-17 13:37 UTC -- NEW

- **暗黑AGI — current #9:** Its July 16 public notebook run matches its leaderboard submission timestamp within two seconds. The customization hook is empty: it directly runs Tufa Labs’ published Duck/TAAF bundle with Qwen3.6-27B-FP8. Newly exposed implementation detail: the Python REPL receives 4-connected object segmentation with translation-invariant hashes, containment and adjacency—not the raw numeric grid—and explicitly rejects changing border strips as likely HUD/timer state. [Notebook](https://www.kaggle.com/code/boristown/taaf-duck-harness-kaggle-share-reresubmission), [source bundle](https://www.kaggle.com/datasets/jeroencottaar/taaf-kaggle-source-share), [leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard)

  **Classification: (b) GENERAL-PURPOSE.** The published path operates from observations through the competition gateway; no game-source reading or per-game hardcoded actions are visible.

## 2026-07-19 13:44 UTC -- checked, nothing new

