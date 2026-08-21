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

## 2026-08-07 13:38 UTC -- NEW

- **New top-five entrant:** **Andy liu** (`codinggodandyliu`, `ichuqinggaove`) is **#2 at 1.69**. No public ARC-AGI-3 notebook, dataset, repository, discussion, or technique writeup was found. **Exploit/general-purpose classification: unknown.** [Leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard), [member code](https://www.kaggle.com/codinggodandyliu/code), [member code](https://www.kaggle.com/ichuqinggaove/code)

- **New top-five entrant:** **GeniusYY** (`alexyy`) is **#4 at 1.64**. Its public notebooks are unrelated to ARC-AGI-3; no ARC agent code, dataset, repository, discussion, or technique writeup was found. **Exploit/general-purpose classification: unknown.** [Leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard), [public notebooks](https://www.kaggle.com/alexyy/code)

## 2026-08-08 13:38 UTC -- checked, nothing new

## 2026-08-09 13:36 UTC -- checked, nothing new

## 2026-08-11 13:37 UTC -- NEW

- **New top-five entrant:** **Lord Han Solo** (`lordhansolo`) is **#3 at 1.65**. No public ARC-AGI-3 notebook, dataset, repository, discussion, or technique writeup was found. **Exploit/general-purpose classification: unknown.** [Leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard), [member code](https://www.kaggle.com/lordhansolo/code)

- **New technique disclosure:** Current top-10 team **Helmut AGI** published its **TAAF Anim Agent**. It adds:

  - An exact `(level, board-state hash, action)` no-op cache that blocks already-proven ineffective repetitions before they consume an environment action.
  - Full intermediate-animation perception: every action gets compact metadata—frame count, distinct-frame count, whether the settled board returned to its original state, transient-pixel count, and bounding box.
  - An on-demand `animation()` tool returning deduplicated, token-bounded frame-diff timelines or cropped transient frames, plus a hint when the agent remains stuck amid informative animations.
  - The reported public-game A/B was statistically null: **+1.4% mean score, p=0.92**; context cost rose **17% per action**, and only **2 of 96** informative animations were inspected. This makes selective routing crucial.

  **Classification: (b) general-purpose.** The published code consumes only frames returned by the API; no game-source reading or per-game policy hardcoding was found. [Writeup](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/734369), [notebook](https://www.kaggle.com/code/jakobbrggen/taaf-anim-arc-agi-3-solver), [source bundle](https://www.kaggle.com/datasets/jakobbrggen/taaf-kaggle-source-anim-20260807-anim)

  **POSSIBLE CARNOT LEVER:** Promote Carnot’s existing multi-layer-frame primitive into automatic live transition evidence—cheap transient-animation metadata by default, compressed timelines only when verifier uncertainty or settle-back/no-op ambiguity warrants them.

## 2026-08-12 13:45 UTC -- check failed

CHECK_TIMED_OUT

## 2026-08-13 13:35 UTC -- NEW

- **New top-five entrant:** **cstl** (`gatamaz`, `tehnar`) is **#1 at 2.70**. No public ARC-AGI-3 notebook, dataset, repository, discussion, or writeup was found. **Exploit/general-purpose classification: unknown.** [Leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard), [gatamaz code](https://www.kaggle.com/gatamaz/code), [tehnar code](https://www.kaggle.com/tehnar/code)

- **New top-five entrant:** **BambooCopter Analytics** (`cky666`, `kilin173`, `oliverwp`, `paradox23`, `ronglinlu`) is **#5 at 1.64**. No submission-linked ARC-AGI-3 code or technique disclosure was found. **Exploit/general-purpose classification: unknown.** [Leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard), [member code](https://www.kaggle.com/cky666/code), [member code](https://www.kaggle.com/kilin173/code), [member code](https://www.kaggle.com/oliverwp/code), [member code](https://www.kaggle.com/paradox23/code), [member code](https://www.kaggle.com/ronglinlu/code)

## 2026-08-15 13:33 UTC -- NEW

- **New top-five entrant:** **Daniel Franzen** (`dfranzen`) is **#2 at 2.58**. No submission-linked ARC-AGI-3 code or writeup was found. **Exploit/general-purpose classification: unknown.** [Leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard), [public notebooks](https://www.kaggle.com/dfranzen/code)

- **New top-five entrant:** **Nikita Sorokin** (`nikitasorokin`) is **#3 at 2.10**. No public notebooks, datasets, repositories, discussions, or technique disclosures were found for this submission. **Classification: unknown.** [Leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard), [public notebooks](https://www.kaggle.com/nikitasorokin/code)

- **New top-five entrant:** **Yusaku Muroya** (`ymuroya47`) is **#4 at 1.98**. His public notebooks are unrelated to ARC-AGI-3; no submission-linked method was found. **Classification: unknown.** [Leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard), [public notebooks](https://www.kaggle.com/ymuroya47/code)

- **New top-five entrant:** **AbeLincoln1865** (`abelincoln1865`) is **#5 at 1.90**. No public ARC-AGI-3 code or technique writeup was found. **Classification: unknown.** [Leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard), [public notebooks](https://www.kaggle.com/abelincoln1865/code)

## 2026-08-16 13:31 UTC -- NEW

- **New top-five entrant:** **Fufront-RyanX-AGI-Team** (`ryanxatlasai`) is **#3 at 2.25**. No submission-linked public code or technique disclosure was found. **Exploit/general-purpose classification: unknown.** [Leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard), [public notebooks](https://www.kaggle.com/ryanxatlasai/code)

- **New top-five entrant:** **egangu** is **#5 at 2.01**. No submission-linked public code, notebook, dataset, discussion, or writeup was found. **Exploit/general-purpose classification: unknown.** [Leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard), [public notebooks](https://www.kaggle.com/egangu/code)

## 2026-08-17 13:39 UTC -- NEW

- **New top-five entrant:** **rellik13** is **#4 at 2.53**. No submission-linked public code or technique disclosure was found. **Classification: unknown.** [Leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard)

- **Fufront-RyanX-AGI-Team — newly discovered artifact:** Its public dataset contains **7,525 pre-recorded actions across 25 named public games** plus a `TraceReplayAgent` that loads the matching `{game_id}.json` and executes the fixed sequence. No evidence connects this artifact to its current hidden-game submission. **Classification: (a) EXPLOIT/HARDCODED; not transferable to unseen games.** [Dataset](https://www.kaggle.com/datasets/ryanxatlasai/arc-agi3-traces-v3)

- **Kevin E R MILLE (#6) — newly discovered public “Sophia” notebook:** A CPU-only, model-free online explorer restricts itself to declared legal actions, suppresses `ACTION7` when alternatives exist, retires action arms after eight frame-level no-ops, and chooses the least-tried action for each exact frame hash weighted by its observed effect rate with 10% random exploration. Clicks target small changed regions or one occupied cell per non-background 4×4 block, retaining 25% uniform-random coverage and preferentially revisiting responsive targets. No source reading or per-game policy hardcoding appears; linkage to the current submission is unconfirmed. **Classification: (b) GENERAL-PURPOSE.** [Notebook](https://www.kaggle.com/code/kevinermille/arc3-sophia-no-action7-v1)

## 2026-08-18 13:40 UTC -- NEW

- **New top-five entrant:** **Tanaka Ai24** is now **#5 at 2.35**. Their older public Duck derivatives add:

  - A compact action/outcome ledger—step, level, action, frame change, level change—exposed to the Python tool for detecting repeated no-ops. Its reported 25-game validation averaged 0.83, and a later kernel removed it.
  - A batch guard that stops before repeating a direction after the preceding identical move produced no visible change, plus a 60→90-second analyzer-yield allowance.

  These kernels predate the current submission, so score linkage is unconfirmed. **Classification: (b) GENERAL-PURPOSE; no source reading or per-game hardcoding.** [Leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard), [safety kernel](https://www.kaggle.com/code/tanakaai24/arc3-qwen3-6-duck-lb117-safety-v1), [trajectory kernel](https://www.kaggle.com/code/tanakaai24/arc3-qwen3-6-duck-compact-trajectory-v1)

- **Samrish B (#8) — newly discovered TAAF graft stack:** Their public notebook enables a report-only efficiency layer that detects exact net-zero round trips, ≥8-action frame stagnation, and repeated exact-state revisits, then injects quadratic action-cost pressure into the LLM prompt. It also truncates homogeneous action batches after two consecutive zero-reward no-ops with unchanged frames and legal actions. **Classification: (b) GENERAL-PURPOSE; runtime observations only.** Current-submission linkage is unconfirmed. [Notebook](https://www.kaggle.com/code/samrishb/just-resubmission-rn-working-on-experiments), [source bundle](https://www.kaggle.com/datasets/thtennant/taaf-kaggle-source-share-fork)

  **POSSIBLE CARNOT LEVER:** Add session-level cycle/stagnation pressure to generator context; Carnot’s existing filter handles individual known-dead actions but not long net-zero round trips.

  The same bundle contains—but the visible notebook does not enable—an exact-fingerprint cross-clone trace bank: one scout publishes pruned completed-level trajectories, sibling clones replay them, and replay aborts on first divergence. **Classification: (a) NO source/hardcoding; (b) CONDITIONAL—general over runtime-detected repeats, but useless on a singleton unseen game.**

  **POSSIBLE CARNOT LEVER:** A divergence-gated, run-local exact-clone plan bank could reuse solved prefixes across anonymized duplicate evaluation runs.

## 2026-08-19 13:33 UTC -- NEW

- **New top-five entrant:** **Jonathan Wang2022** is now **#3 at 2.59**. No public ARC-AGI-3 notebook, dataset, discussion, repository, or writeup was found. **Classification: unknown.** [Leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard), [public notebooks](https://www.kaggle.com/jonathanwang2022/code)

- **New top-five entrant:** **Junhua Yang** is now **#5 at 2.57**. No submission-linked ARC-AGI-3 code or technique disclosure was found. His documented ARC-AGI-1/2 DSL work does not establish the current ARC-AGI-3 method. **Classification: unknown.** [Leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard), [public notebooks](https://www.kaggle.com/junhuay/code)

## 2026-08-20 13:37 UTC -- NEW

- **The AGI Boys — newly discovered public agent variant:** Team member Roman Tamrazov’s notebook modifies the known Reki-style policy with transition-contingent replanning: queued actions are discarded after no-op, repeated-state, or level-up transitions; failed click coordinates are cached per state and repaired using deterministic ±1/2/3/5-pixel jitter; fallbacks prefer recent changed pixels and least-used non-stalling actions; timeouts trigger two cheap fallback turns. It replaces Reki’s broader structural dead-signature suppression with exact state/coordinate evidence. The notebook predates the current submission, so score attribution is unconfirmed. **Classification: (b) GENERAL-PURPOSE**—the scored path uses observations and legal-action metadata, with no game-source reading or per-game hardcoding found. [Notebook](https://www.kaggle.com/code/romantamrazov/arc-real-agi-solution), [leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard)

  **POSSIBLE CARNOT LEVER:** Add verifier-gated neighborhood jitter after an exact click failure; it can rescue coarse near-miss coordinates without RL or unsafe structural-class pruning.

## 2026-08-21 13:38 UTC -- NEW

- **New top-five entrant:** **Akhil Tolani** (`akhiltolani`) is now **#4 at 2.73**. No public ARC-AGI-3 notebook, dataset, repository, discussion, or technique writeup was found. **Classification: unknown**; no Carnot lever can be assigned. [Leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard), [Kaggle profile](https://www.kaggle.com/akhiltolani/code)

