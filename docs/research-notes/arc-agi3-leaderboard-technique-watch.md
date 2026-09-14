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

## 2026-08-22 13:36 UTC -- NEW

- **New top-five entrant:** **AbeLincoln1865** is now **#5 at 2.72**. No public ARC-AGI-3 notebook, repository, discussion, or technique writeup was found. **Exploit vs. transferable classification: unknown**; no Carnot lever can be assigned. [Leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard), [Kaggle profile](https://www.kaggle.com/abelincoln1865/code)

## 2026-08-23 13:37 UTC -- NEW

- **Tufa Labs — new model direction:** Dries Smit published a **Qwen3.8-27B-FP8 multimodal checkpoint** on August 18, replacing Qwen3.6 as the team’s latest visible ARC-oriented model artifact. It supports image input, adjustable reasoning effort, persistent multi-turn thinking, and block-FP8 quantization. The artifact predates Tufa’s latest submission, but no public notebook proves it powered that score. **Classification: (b) GENERAL-PURPOSE; no game-source access or per-game hardcoding.** [Leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard), [Tufa Kaggle artifact](https://www.kaggle.com/datasets/driessmit1/qwen3-8-27b-fp8-hf-017b9c7a), [model card](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)

- **Akhil Tolani — new Duck/DFlash2 inference stack:** Tolani published the same Qwen3.8-27B-FP8 model family plus an ARC-specific **vLLM 0.27.1/CUDA 12.9 wheelhouse whose README explicitly adds DFlash2 draft-model support to the Duck harness**. DFlash2 drafts an entire token block in parallel, uses dynamic depthwise convolution to propagate within-block context, retains top-K candidates per position, and selects a coherent path for target-model verification. Published H200 tests report **2.67–3.43× single-request throughput** over autoregressive Qwen3.8, though this is not an ARC/RTX-6000 measurement. These artifacts predate Tolani’s current submission, but submission linkage remains unproven. **Classification: (b) GENERAL-PURPOSE; inference optimization only, with no source-reading or per-game logic.** [Wheelhouse](https://www.kaggle.com/datasets/saltb0x/arc3-vllm-wheelhouse-v0271-cu129), [Qwen artifact](https://www.kaggle.com/datasets/saltb0x/qwen3-8-27b-fp8), [DFlash2 implementation](https://github.com/vllm-project/vllm/pull/52816), [DFlash2 model/results](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)

  **POSSIBLE CARNOT LEVER:** Benchmark lossless DFlash2 speculative decoding on the RTX Pro 6000; higher local-generator throughput could buy more verifier-routed candidates and replanning turns within the nine-hour limit without adding RL.

## 2026-08-24 13:36 UTC -- checked, nothing new

## 2026-08-25 13:40 UTC -- NEW

- **Tong Hui Kang — new top-five entrant:** Now **#4 at 3.39**. His public ARC-AGI-3 repository exposes a supervised imitation policy using 384 binary features per cell: current/intermediate/level frames, six action-linked history frames, previous-level endpoints, and positional planes. Legal-action, click-color, and color-conditioned coordinate heads are trained jointly. Public ablations favor a four-block, 128-channel ConvNeXt, although the production architecture and linkage to today’s submission remain unproven. **Classification: (b) GENERAL-PURPOSE; the published code contains no game-source access or per-game branches.** [Leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard), [repository](https://github.com/tonghuikang/arc3), [feature/loss code](https://github.com/tonghuikang/arc3/blob/master/autoresearch/model.py), [ablation results](https://github.com/tonghuikang/arc3/blob/master/autoresearch/metrics_export/index.json)

  **POSSIBLE CARNOT LEVER:** Train a lightweight action-and-click imitation prior from recorded trajectories and route its top candidates into verifier search; this extends the existing frame-change predictor from clickability to complete action proposals without requiring RL.

- **Tufa Labs — newly exposed Duck implementation details:** The released Milestone-1 harness makes an object graph—not the raw grid—its primary programmatic representation: four-connected components carry translation-invariant shape/color hashes, boundaries, containment relationships, and edge adjacency. The agent maintains explicit world/goal/action models and cross-level notes, then writes BFS/DFS/beam/shortest-path searches in its sandbox and batches reliable action sequences with terminal-state guards. This is Milestone-1 code, not evidence that Tufa’s current submission is unchanged. **Classification: (b) GENERAL-PURPOSE; the runtime path exposes no game source or game-ID-specific logic.** [Duck harness](https://github.com/Tufalabs/duck-harness), [segmentation implementation](https://github.com/Tufalabs/duck-harness/blob/main/ARC3-Inference/inference/utils/segmentation.py), [runtime/search prompt](https://github.com/Tufalabs/duck-harness/blob/main/ARC3-Inference/inference/agent/prompts.py), [structured memory implementation](https://github.com/Tufalabs/duck-harness/blob/main/ARC3-Inference/inference/agent/tool_agent.py)

## 2026-08-26 13:33 UTC -- NEW

- **New top-five entrant:** **Tony G** (`junvalue`) is now **#5 at 3.17**. No submission-linked public code, notebook, dataset, discussion, repository, or writeup was found. **Exploit/general-purpose classification: unknown**; no Carnot lever can be assigned. [Leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard), [Kaggle code profile](https://www.kaggle.com/junvalue/code)

## 2026-08-27 13:39 UTC -- NEW

- **New top-five entrant:** **rfbr** (`romainfabre`) is now **#5 at 3.37**, displacing Tony G to #6. No attributable public code, notebook, discussion, repository, or technique writeup was found. **Exploit/general-purpose classification: unknown**; no **POSSIBLE CARNOT LEVER** can be assigned. [Leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard), [Kaggle code profile](https://www.kaggle.com/romainfabre/code)

## 2026-08-28 13:44 UTC -- check failed

CHECK_TIMED_OUT

## 2026-08-29 13:39 UTC -- checked, nothing new

## 2026-08-30 13:37 UTC -- checked, nothing new

## 2026-08-31 13:43 UTC -- check failed

CHECK_TIMED_OUT

## 2026-09-01 13:39 UTC -- NEW

- **New top-five entrant:** **Son Pham & Mark Barney** are now **#5 at 4.42**. [Kaggle leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard)

- **New public artifact; current-score linkage is not proven:** Son Pham published a complete Duck-derived agent bundle using **Qwen3.8-Flash-Next-NVFP4**—a ~180B multimodal MoE with 512 experts/layer and top-10 routing. Its single-GPU configuration uses PLE CPU offload, TP=1, a 32K runtime context, 22 admitted sequences, and 28 concurrent game clients. [Agent/model bundle](https://www.kaggle.com/datasets/sonphamorg/arc3-flashnext-serving-part-a-v1), [model card](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4/blob/main/README.md)

- **New cross-game mechanism:** A non-blocking 29th LLM thread continuously reads the 28 gameplay agents’ current world models. Once at least three games provide new evidence, it synthesizes an atomic replacement ledger of at most six falsifiable cross-game priors. The current ledger is injected into each action prompt but stripped before that prompt enters persistent history, preventing stale duplication and context growth. **Classification: (b) GENERAL-PURPOSE.** The scored branch consumes gateway observations/actions; no game-source imports or per-game policy branches were found. Public `environment_files` are used only during non-submission offline validation. [Source bundle](https://www.kaggle.com/datasets/sonphamorg/arc3-flashnext-serving-part-a-v1)

  **POSSIBLE CARNOT LEVER:** Add an asynchronous curator that distills accepted per-game world models into a small, verifier-labeled prior ledger for later games—cross-game transfer without RL or blocking live search.

- **New animation-evidence routing:** After a five-action warm-up, the agent detects unusually long animation tails using a rolling median/MAD threshold conditioned first on level/action family and then game-wide history. It tracks 8-connected changed regions across consecutive frames, discards one-transition HUD noise without assuming a border location, interrupts action batches only for novel tails, and supplies token-bounded storyboards or agent-selected crops from up to 12 retained native frames. **Classification: (b) GENERAL-PURPOSE; no source-reading or game-specific hardcoding.** [Experiment manifest and implementation bundle](https://www.kaggle.com/datasets/sonphamorg/arc3-flashnext-serving-part-a-v1)

  **POSSIBLE CARNOT LEVER:** Trigger re-induction only on statistically novel, persistent animation tails, adding temporal evidence while keeping frame/token costs bounded.

## 2026-09-02 13:42 UTC -- NEW

- **Newly discovered artifact for current #6 Matija Ludvig (4.17); submission linkage is unproven:** His EVE ARC paper describes a non-LLM neuro-symbolic pipeline combining frame-hash BFS/DFS, reusable mechanic-class solvers, and program transfer. Its “~41%” result is 78/182 levels on the 25 public games—not hidden-leaderboard RHAE; the transfer-only public RHAE was 1.78%. The implementation is available only upon request, preventing source-access auditing. [Leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard), [paper](https://zenodo.org/records/19458465)

- **New concrete routing mechanism:** EVE trains a 305,995-parameter `1024→256→128→64→32` task encoder from scratch on 100 solved internal tasks to predict broad strategy classes. It retrieves the nearest solved task by embedding cosine similarity and transfers its executable program. A coarse seven-class representation transferred better than a 36-class version (155 versus 120 ARC-AGI-1 tasks), suggesting that broad functional similarity beats fine mechanic classification. **Classification: (b) GENERAL-PURPOSE AS DESCRIBED**—training is non-ARC, retrieval does not use domain labels, and ARC-AGI-3 search is observation/frame-based; however, unavailable source code prevents independent confirmation. [EVE ARC paper](https://zenodo.org/records/19458465)

  **POSSIBLE CARNOT LEVER:** Add a lightweight coarse-strategy embedding router that retrieves reusable executable programs before LLM induction, cheaply expanding the verifier’s candidate pool without RL.

## 2026-09-03 13:35 UTC -- NEW

- **mostik.ai — newly disclosed latent model cooperation.** WIRED reports that Mostik used its latent-handoff approach for its ARC-AGI-3 system, although the exact ARC models, harness, and code remain undisclosed. A frozen sender passes hidden states through a separately trained bridge directly into a frozen receiver; no textual message is generated between them. Their published non-ARC demonstration uses GLM-5.2 753B for prefill/reasoning and Qwen-3.5 4B for decoding; this pairing is illustrative, not confirmed as the ARC configuration. **Classification: (b) GENERAL-PURPOSE AS DISCLOSED.** No game-source or per-game mechanism is disclosed, but unavailable ARC code prevents an exploit audit. [Mostik technical disclosure](https://mostik.ai/read-more), [ARC linkage reported by WIRED](https://www.wired.com/story/russian-startup-mostik-ai-models-communication/)

  **POSSIBLE CARNOT LEVER:** Investigate a learned hidden-state advisor→generator bridge that preserves Carnot’s verifier routing while transferring richer reasoning than a text-only handoff.

- **MIKE DAN7 — newly discovered ARC-specific Duck variant; current-submission linkage unproven.** The public notebook runs Qwen3.8-27B-FP8 over the known TAAF animation-aware Duck but changes visual perception from the upstream 4× render to an 8× current-grid render: **256×256/64 visual tokens → 512×512/256 visual tokens**. It uses a 65,536-token serving context, 32,768-token agent context, and 28 concurrent game clients. **Classification: (b) GENERAL-PURPOSE.** Its scored branch uses competition-gateway observations; the visible public-game IDs belong only to offline evaluation, with no source-reading or per-game action logic. [Public Kaggle notebook and logs](https://www.kaggle.com/code/mikedan7/arc-agi-3-qwen3-8-27b-fp8-submit/log)

  **POSSIBLE CARNOT LEVER:** A/B an uncertainty-gated 512px frame view, buying finer spatial evidence only when the verifier detects perception ambiguity rather than paying the fourfold visual-token cost every turn.

## 2026-09-04 13:37 UTC -- NEW

- **Third Intelligence — newly disclosed world-model training direction; current-submission linkage unproven.** The current #2 team says it is training world models on `arc-agi-3-wm-traces`: 14.6M state/action/next-state transitions spanning human play, prior-game traces, random/scripted rollouts, mined successful trajectories, novelty/frontier-generated trajectories, and static ARC transformations. Architecture and planning code remain undisclosed. **Classification: (b) GENERAL-PURPOSE AS DISCLOSED.** No game-source reading or per-game hardcoding is shown, although unavailable submission code prevents a full exploit audit. [Leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard), [Third Intelligence disclosure](https://huggingface.co/datasets/fredericowieser/arc-agi-3-wm-traces/discussions/2), [dataset details](https://huggingface.co/datasets/fredericowieser/arc-agi-3-wm-traces)

  **POSSIBLE CARNOT LEVER:** Pretrain an action-conditioned transition proposer on the mixed trace corpus, then admit imagined transitions or plans only when Carnot’s executable verifier confirms them.

- **ARC Prize — newly published Astra harness analysis, separate from Kaggle.** The Provider Adapter maintains a continuous native conversation, preserves opaque reasoning state, and compacts long histories; the Standard harness instead relies on model-written visible notes. Across 167 game/reasoning pairs solved by both, Provider Adapter runs used 49% fewer tokens and were 3.66× faster. Astra also created compact symbolic state/plan notation and runtime-generated parsers, world models, search algorithms, and planners. **Classification: (b) GENERAL-PURPOSE.** The tools were synthesized during play; no source reading or prewritten per-game logic was reported. [ARC Prize analysis](https://arcprize.org/blog/astra), [open-source harness](https://github.com/arcprize/arc-agi-3-benchmarking)

  **POSSIBLE CARNOT LEVER:** Benchmark a persistent generator conversation with verifier-triggered compaction, preserving reasoning state while reducing repeated reconstruction and token cost.

## 2026-09-05 13:34 UTC -- NEW

- **Leaderboard change:** Daniel Franzen has re-entered the top five at **#2 with 6.66**, behind mostik.ai (7.51). No submission-linked ARC-AGI-3 code or writeup is public; his visible notebooks cover ARC 2024/2025 only. **Classification: unknown**—no exploit audit or transferable technique attribution is currently possible. [Leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard), [public notebooks](https://www.kaggle.com/dfranzen/code)

## 2026-09-06 13:38 UTC -- NEW

- **jinbo wang1 — newly discovered public implementation; current-score linkage unproven.** The current #4 team’s latest documented submission candidate is Tufa’s Duck harness with **Qwen3.8-27B-FP8, multimodal input, and 28 concurrent games**. A September 1 team playbook separately instructs members to reproduce Son Pham’s public **Qwen3.8-Flash-Next-NVFP4** bundle and then test MTP speculative decoding. The September 3 notebook that produced 5.49 is not public, so neither configuration can be attributed to that score. **Classification: (b) GENERAL-PURPOSE candidates**—their competition paths use gateway observations without game-source access or per-game branches. [Leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard), [Qwen3.8/Duck submission script](https://github.com/jinbowang1/arc-prize-2026/blob/f00713494053bd3da841829bf60219552c081c1f/kaggle_agent/duck/submit_lb9.sh), [Flash-Next reproduction playbook](https://github.com/jinbowang1/arc3-teammate-playbook/blob/e9f448013b9c9c15ba12f07cc67d5b120187a94a/README.md)

- **New general-purpose REPL mechanisms from the same team:** automatic keyboard/object probes; counter/HUD-cell suppression; color-count and object-movement ledgers; exact pixel-replay validation of model-written transition functions over every observed transition; repeated-code and three-idle-round breakers; six-exchange context truncation; and cross-game persistence of compact notes plus model-written Python analysis tools. **Classification: (b) GENERAL-PURPOSE;** the published Kaggle architecture explicitly notes that hidden games are accessed only through the scored gateway. [REPL implementation](https://github.com/jinbowang1/arc-prize-2026/blob/f00713494053bd3da841829bf60219552c081c1f/kaggle_agent/repl_agent.py), [Kaggle-path documentation](https://github.com/jinbowang1/arc-prize-2026/blob/f00713494053bd3da841829bf60219552c081c1f/kaggle_agent/README.md)

  **POSSIBLE CARNOT LEVER:** Persist verifier-tested, game-agnostic analysis functions—not merely textual hypotheses—across games, giving later inductions an executable tool library without RL training.

- **Exploit audit:** The repository’s impressive public-game perfect-score solvers are separate and non-transferable: rules were manually induced, `sc25` explicitly used white-box source reading, searches used free cloned simulators, and exact per-game solution traces are stored. **Classification: (a) SOURCE-READING/HARDCODED; do not attribute these results to the hidden leaderboard score.** [Repository methodology and limitations](https://github.com/jinbowang1/arc-prize-2026#readme)

## 2026-09-07 13:45 UTC -- check failed

CHECK_TIMED_OUT

## 2026-09-08 13:43 UTC -- check failed

CHECK_TIMED_OUT

## 2026-09-09 13:37 UTC -- NEW

- **New top-five entrant — NVARC3:** #5 at **6.17** as of September 9, 13:31 UTC. Its exact submission remains private. [Current Kaggle leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard)

- **Newly attributable public technique lineage; current-score linkage unproven:** NVARC3 member Jean‑François Puget co-authored NVIDIA’s **AVO** ARC-AGI-3 system. It deliberately avoids an explicit programmatic world model: the agent directly consumes exact 64×64 text grids, maintains persistent evidence, and is redirected by a separate supervisor when its trajectory stagnates. No images, game rules, goals, or action descriptions are supplied. **Classification: (b) GENERAL-PURPOSE AS DESCRIBED;** no source-reading or per-game branches are disclosed, although the ARC implementation is not publicly auditable. Its reported result used Claude Opus 5 on the public games, so that exact configuration cannot explain the offline Kaggle score. [NVIDIA AVO writeup](https://developer.nvidia.com/blog/nvidia-avo-reaches-100-on-arc-agi-3-demonstrating-a-frontier-level-general-purpose-architecture-for-long-horizon-autonomous-agents/), [CPMP/J.-F. Puget identity](https://www.kaggle.com/cpmpml)

- Two other NVARC3 members, Gal Kaplun and Elad Sarafian, co-authored the public **NOOA ARC agent**: a CodeAct loop that writes syntax-checked NumPy entity/transition helpers, reloads them as executable world-model modules, and persists discoveries through associative memory or Markdown. Its anti-exploit boundary anonymizes game IDs, separates agent and harness processes, restricts generated file access, and redacts identity-bearing paths. **Classification: (b) GENERAL-PURPOSE.** The published configuration uses hosted GPT-5.5, so current Kaggle-score linkage is again unproven. [ARC community entry](https://arcprize.org/leaderboard/community), [NOOA ARC implementation](https://github.com/NVIDIA-NeMo/labs-OO-Agents/tree/main/examples/arc_agi_3)

  **POSSIBLE CARNOT LEVER:** Add an exact-grid direct-interaction fallback that proposes verifier-routed actions without waiting for executable world-model admission, targeting games where induction itself is the bottleneck.

## 2026-09-10 13:42 UTC -- NEW

- **Newly discovered Kyutai prototype; current-submission linkage unproven:** Current #6 team member Hippolyte Pilchen published an online, per-game **expert-iteration** agent. Round 1 uses hash-pruned BFS/best-first search plus a Go-Explore-style graph explorer; discovered trajectories fine-tune an action-conditioned latent dynamics/reward model. Later rounds use predicted reward to prioritize search, then exploit through hierarchical MPC: MPPI selects latent macro-action subgoals and low-level random shooting selects primitive actions. The public defaults run two search→fine-tune rounds with warm-started weights. [Current leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard), [public Kaggle notebook](https://www.kaggle.com/code/hippolytepilchen/fork-of-notebooka824efee1d)

  **Classification: (b) GENERAL-PURPOSE AS PUBLISHED.** The vendored implementation operates on observed frames, legal actions, rewards, and state hashes; it contains no game-source access or per-game solution branches. Its external pretrained PLDM checkpoint is not publicly auditable, and the July notebook is not proven to be Kyutai’s current scoring submission.

  **POSSIBLE CARNOT LEVER:** Add lightweight within-game supervised adaptation of a latent transition/reward model so early verifier-confirmed probes become a learned heuristic for later search and MPC.

## 2026-09-11 13:44 UTC -- NEW

- **Son Pham & Mark Barney — newly discovered public development workspace; exact current-submission linkage unproven.** The current #10 team’s best documented single-agent configuration uses Qwen3.8-27B-FP8 with a host-enforced checkpoint after eight environment actions: it returns the settled frame and unexecuted suffix so the model must re-ground before continuing. Their sampling sweep favored `temperature=1.0, top_p=0.95, top_k=20`; the equivalent NVFP4 model generated substantially more actions but scored 19.3% lower than FP8 in two public-set replicas, so it was not promoted. [Leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard), [checkpoint manifest](https://github.com/sonpham-org/arc-3/blob/c40ae7137309ae5b15981d907c37a7e4ddadedb8/harnesses/taaf-kaggle203-nocap-control/MANIFEST.md), [implementation patch](https://github.com/sonpham-org/arc-3/blob/c40ae7137309ae5b15981d907c37a7e4ddadedb8/harnesses/taaf-plain-checkpoint8/patch/taaf-plain-checkpoint8.patch), [run evidence](https://github.com/sonpham-org/arc-3/blob/c40ae7137309ae5b15981d907c37a7e4ddadedb8/scripts/export_runs_index.py)

  **Classification: (b) GENERAL-PURPOSE.** The submission path obtains games through the competition gateway; the checkpoint and sampling policy contain no game-specific branches or source access.

- **New action-efficiency ablation from the same team:** Their “no-impact” guard learns deterministic HUD/timer regions online using per-row/column change frequencies (`window=20`, `threshold=0.9`, `warmup=8`), optionally unioned with an LLM-written HUD transition function that is continually execution-verified and discarded after repeated prediction failures. An exploratory action whose masked result is unchanged immediately cancels the remaining batch. In a two-pass public-game ablation excluding `ft09`, it improved mean score from 1.046 to 1.624 and completed 21 versus 15 levels. Conversely, exposing an explicit state graph to the model regressed roughly 32% because graph reasoning raised generated tokens per action from 366 to 644; they disabled it. [Implementation and ablation](https://github.com/sonpham-org/arc-3/blob/c40ae7137309ae5b15981d907c37a7e4ddadedb8/harnesses/ffa7g/MANIFEST.md)

  **Classification: (b) GENERAL-PURPOSE.** It uses only observed frame/action transitions. The repository’s public game sources are used for offline evaluation, not by this mechanism.

  **POSSIBLE CARNOT LEVER:** Use Carnot’s existing safe HUD mask to cancel queued exploratory actions on a masked self-loop and force immediate replanning, converting detection into direct action savings.

- **Newly published training direction:** The team now maintains a 927-game practice catalog and code that reconstructs multimodal tool-use trajectories, keeps only verifier-confirmed solved levels, discards subsequent failing/flailing turns, and emits level-granular SFT records with optional recolor/noise/merge/occlusion augmentation. Their documentation explicitly says the synthetic-game→Kaggle causal loop has not yet been validated, so this is not established as contributing to the current score. [Catalog](https://github.com/sonpham-org/arc-3/blob/c40ae7137309ae5b15981d907c37a7e4ddadedb8/README.md), [pipeline description](https://github.com/sonpham-org/arc-3/blob/c40ae7137309ae5b15981d907c37a7e4ddadedb8/docs/how-this-feeds-kaggle.md), [SFT extractor](https://github.com/sonpham-org/arc-3/blob/c40ae7137309ae5b15981d907c37a7e4ddadedb8/ARC3-Inference/distill/extract_sft.py)

  **Classification: (b) GENERAL-PURPOSE PIPELINE.** It trains from agent-generated trajectories on authored practice environments, not hidden-game source.

  **POSSIBLE CARNOT LEVER:** Fine-tune Carnot’s local generator exclusively on verifier-confirmed solved prefixes from mechanics-diverse synthetic games, avoiding imitation of failed search tails.

## 2026-09-12 13:37 UTC -- NEW

- **NVARC3 — newly attributable NVIDIA DreamTeam lineage; exact current-submission linkage is unproven.** Team members Elad Sarafian and Gal Kaplun coauthored an open-source six-agent system built around three executable artifacts: observation/rendering, transition/history simulation, and reusable policies. Before each live action it commits a prediction; afterward it computes field-level errors, replays recent transitions, and routes counterexamples to the artifact’s responsible agent. The leader gates imagined rollouts on retrodiction reliability, falling back to information-seeking probes when the world model is immature. Confident action sequences can execute with lightweight per-step observer/simulator checks and early interruption. The paper reports 38.4% average public-demo RHAE and 31% fewer environment actions than its matched baseline. However, the published configuration uses hosted GPT-5.5 and Claude Opus 4.8 models, so it cannot establish what powers NVARC3’s offline Kaggle submission. [Current leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard), [repository](https://github.com/NVIDIA/dream-team), [paper](https://arxiv.org/abs/2605.09650), [published configuration](https://github.com/NVIDIA/dream-team/blob/main/runs/arc_agi_3/configs/offline_focused.yaml)

  **Classification: (b) GENERAL-PURPOSE.** The published competition path operates on API observations; no public-game source reading or per-game hardcoding was found.

  **POSSIBLE CARNOT LEVER:** Gate search on recent field-level retrodiction accuracy—when an executable model is insufficiently validated, issue an information-gathering probe and route its precise errors only to the responsible observation, transition, or policy component.

## 2026-09-13 13:36 UTC -- NEW

- **New top-five entrant — Ebi (`ebinan92`):** now **#2 at 8.68**, behind Tufa Labs. The scoring notebook remains private; no ARC-specific code, notebook, discussion, repository, or writeup is publicly attributable. **Classification: unknown**—source-reading/hardcoding versus general-purpose cannot yet be determined. [Current leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard), [public Kaggle notebooks](https://www.kaggle.com/ebinan92/code), [GitHub profile](https://github.com/ebinan92)

## 2026-09-14 13:34 UTC -- checked, nothing new

