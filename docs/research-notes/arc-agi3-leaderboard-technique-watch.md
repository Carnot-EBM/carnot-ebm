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

## 2026-07-20 13:43 UTC -- checked, nothing new

## 2026-07-21 13:35 UTC -- NEW

- **DhanaLakshmiMalla** newly entered the top five at **#3, 1.60**; **ippeiogawa** newly entered at **#4, 1.58**. [Current Kaggle leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard)

- Neither entrant currently exposes submission-linked code or a technique writeup. DhanaLakshmiMalla has no public notebooks; ippeiogawa’s public notebooks are unrelated to ARC-AGI-3. Therefore, no evidence-based **(a) exploit** or **(b) general-purpose** classification—and no Carnot lever—can yet be assigned. [DhanaLakshmiMalla notebooks](https://www.kaggle.com/dhanalakshmimalla/code), [ippeiogawa notebooks](https://www.kaggle.com/ippeiogawa/code)

## 2026-07-22 13:40 UTC -- NEW

- **ippeiogawa — newly discovered ARC-3 model artifact:** A July 17 Kaggle dataset is explicitly labeled an “arc3 teacher-probe artifact” containing `Qwen3.5-122B-A10B-NVFP4`: a multimodal MoE with 122B total/10B active parameters, NVFP4 quantization, and vLLM deployment support. No public harness links it to the leaderboard submission, so it establishes an ARC-3 experimentation direction—not the score’s mechanism. [Kaggle artifact](https://www.kaggle.com/datasets/ippeiogawa/qwen35-122b-a10b-nvfp4), [model architecture](https://huggingface.co/nvidia/Qwen3.5-122B-A10B-NVFP4)

  **Classification: (b) GENERAL-PURPOSE artifact.** The published checkpoint contains no game-source reading or per-game hardcoding and is plausibly applicable to unseen games; the surrounding teacher-probe procedure remains undisclosed.

## 2026-07-23 13:40 UTC -- NEW

- **Dinesh Kumar Thiyagarajan — newly located public-game exploit artifacts.** His Kaggle dataset contains exact action arrays keyed to 14 named public games—97 levels and 1,844 actions. A separate notebook dynamically imports each game’s `.py`, searches an unscored local instance through direct `perform_action` calls, then replays the solution. No evidence links these artifacts to his current leaderboard submission, so this is not score attribution. [Leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard), [hardcoded solutions](https://www.kaggle.com/datasets/dineshkumar0705/arc-solutions), [source-reading solver](https://www.kaggle.com/datasets/dineshkumar0705/brain-v2)

  **Classification: (a) EXPLOIT/HARDCODED.** It depends on known game IDs, stored trajectories, or inspectable game source and would not transfer to a genuinely hidden game.

## 2026-07-25 13:37 UTC -- NEW

- **Dinesh Kumar Thiyagarajan (#10) — newly discovered `brain-v1` artifact.** Its `BrainV2` package implements object-centric parsing, hash-state frontier BFS, an explicit EXPLORE→MODEL→PLAN→RECOVER controller, competing mechanic hypotheses, an LLM-written executable world model backtested against observed transitions, cross-game mechanic memory, and failure reflection. **Classification: (b) GENERAL-PURPOSE components.** However, its prompt also maps known public game prefixes to mechanic classes and accepts pre-solved/analytical prefix lists. **Classification: (a) EXPLOIT/HARDCODED contamination.** No game-source reading appears in this artifact, and nothing links it to the current submission; bundled “Omega Prime” additions are explicitly disabled/unscored. [Dataset](https://www.kaggle.com/datasets/dineshkumar0705/brain-v1), [leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard)

## 2026-07-26 13:33 UTC -- NEW

- **New top-five entrant:** **Yuchen20** (`yuchen2066`) is now **#5 at 1.58**, tied with ippeiogawa. No submission-linked public code or ARC-AGI-3 writeup was found, so no evidence-based exploit/general-purpose classification or Carnot lever can be assigned. [Leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard), [public notebooks](https://www.kaggle.com/yuchen2066/code)

## 2026-07-27 13:35 UTC -- checked, nothing new

## 2026-07-28 13:34 UTC -- checked, nothing new

## 2026-07-29 13:38 UTC -- checked, nothing new

## 2026-07-30 13:38 UTC -- checked, nothing new

## 2026-07-31 13:40 UTC -- NEW

- **New top-five entrant:** **Andy liu** (`codinggodandyliu`, `ichuqinggaove`) is **#2 at 1.69**. No submission-linked public code or writeup was found; **exploit/general-purpose classification: unknown**. [Leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard), [member notebooks](https://www.kaggle.com/codinggodandyliu/code)

- **New top-five entrant:** **GeniusYY** (`alexyy`) is **#3 at 1.64**. No submission-linked public code or writeup was found; **exploit/general-purpose classification: unknown**. [Leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard), [public notebooks](https://www.kaggle.com/alexyy/code)

- **New top-five entrant:** **DhanaLakshmiMalla** (`dhanalakshmimalla`) is **#5 at 1.60**. No submission-linked public code or writeup was found; **exploit/general-purpose classification: unknown**. [Leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard), [public notebooks](https://www.kaggle.com/dhanalakshmimalla/code)

## 2026-08-01 13:41 UTC -- NEW

- **New top-five entrant:** **FOYSAL** (`foysalemonshanto`) is **#5 at 1.61**. The current August 1 submission is not publicly linked to code, so its **exploit/general-purpose classification is unknown**. [Leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard), [public notebooks](https://www.kaggle.com/foysalemonshanto/code)

- **Newly discovered artifact, not current-score attribution:** FOYSAL’s older public notebook—an approximately **0.41-scoring light fork** of Ashvin Singh’s agent—imports each game’s `.py` class, probes mutable scalar fields for hidden state, searches unscored game copies with BFS, then replays the solution. **Classification: (a) EXPLOIT/SOURCE-READING.** Its novel general-purpose component first replays the preceding level’s solution, then matches objects by color/area, estimates their mean centroid displacement, and translates all `ACTION6` coordinates accordingly. **Classification: (b) GENERAL-PURPOSE technique**, although this implementation verifies it through the source-reading path. [FOYSAL notebook](https://www.kaggle.com/code/foysalemonshanto/ash-s-arc-agi-3-agent-d83ab9), [upstream notebook](https://www.kaggle.com/code/ashvinsingh/ash-s-arc-agi-3-agent)

  **POSSIBLE CARNOT LEVER:** verifier-gated object-relative trajectory transfer—re-anchor a previously successful level trace to matched objects before invoking expensive LLM world-model induction.

## 2026-08-02 13:37 UTC -- checked, nothing new

## 2026-08-03 13:41 UTC -- checked, nothing new

## 2026-08-04 13:41 UTC -- NEW

- **New top-five entrant:** **Andy liu** (`codinggodandyliu`, with `ichuqinggaove`) is **#2 at 1.69**. No ARC-AGI-3 code or writeup is publicly attached. **Exploit/general-purpose classification: unknown.** [Leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard), [Andy’s notebooks](https://www.kaggle.com/codinggodandyliu/code), [teammate’s notebooks](https://www.kaggle.com/ichuqinggaove/code)

- **New top-five entrant:** **GeniusYY** (`alexyy`) is **#3 at 1.64**. Their public notebooks are unrelated to ARC-AGI-3, and no solution writeup is attached. **Exploit/general-purpose classification: unknown.** [Leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard), [public notebooks](https://www.kaggle.com/alexyy/code)

## 2026-08-05 13:34 UTC -- NEW

- **New top-five entrant:** **Lord Han Solo** (`lordhansolo`) is **#3 at 1.65**. No public ARC-AGI-3 notebook, dataset, code repository, or technique writeup was found. **Exploit/general-purpose classification: unknown.** [Leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard), [public notebooks](https://www.kaggle.com/lordhansolo/code)

## 2026-08-06 13:35 UTC -- checked, nothing new

