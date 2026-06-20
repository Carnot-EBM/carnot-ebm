# ARC-AGI-3 leaderboard competitive intel — 2026-06-20

Quick dive (operator-requested) into who's at the top and WHY. Sources: the live Kaggle leaderboard +
the top PUBLIC notebooks (code competition → their actual code is readable) + the ARC Prize 30-day
preview report. NOT the /deep-research fan-out (it rate-limited to zero output earlier this session).

## The leaderboard (2026-06-20)

| Rank | Team | Score | Note |
|---|---|---|---|
| 1 | **Tufa Labs** | **1.21** | "first meaningful jump" past the template ceiling (0.68→1.17→1.21) |
| 2 | Tong Hui Kang | 0.70 | |
| 3 | Redfield Rentals | 0.68 | |
| 4–8 | Barada Sahu / Kevin Miller / SVG / Poetker / face-of-agi | 0.63–0.66 | a dense cluster |
| — | **Carnot (us)** | **0.08** | first submission, bare-BFS path |

So the field: one breakaway leader (~1.2), a wall at ~0.63–0.70, and us well below. Even the leader is
~1.2% — ARC-AGI-3 is brutally unsolved (the 30-day report: all preview winners scored **< 13%**; GPT-5
/ Claude / Gemini all **< 1%**).

## What the top agents actually do (from their code)

The winning paradigm is **a LEARNED action-effect model + RL/search + persistent memory** — NOT LLM
world-model induction.

1. **StochasticGoose (Tufa Labs, the leader).** A **CNN that predicts which actions cause a FRAME
   CHANGE**, trained as RL. The entire edge is exploration efficiency: it learns "what is clickable /
   what does anything" early instead of wasting actions. The 30-day report: it first wasted "nearly 350
   moves clicking with no result," then exploited the learned clickability map.

2. **Blind Squirrel (2nd in preview, 6.71%).** Explore-and-learn: **builds a state graph from frames,
   prunes actions that create loops or don't change state, trains a ResNet18 VALUE model.**

3. **[0.46] Persistent Memory BFS (top public notebook).** A full DQN stack: **Prioritized Experience
   Replay** (SumTree), **expert injection** (imitation demos seeded at 5× priority), **PersistentAEM** =
   an action-effect memory `(frame-diff, action, reward)` **persisted ACROSS games**, a **CBAM
   attention-CNN** value net, **IDA\*** + BFS with a color-count heuristic ("fewer colors = closer"), and
   a careful **per-game TimeBudget** (12h-aware). Score 0.46 — ~6× ours.

4. **Hybrid BFS+CNN.** Offline BFS over the REAL game class (`find_game_source_and_class`) with
   **hidden-field probing baked into the state hash**, **BFS→CNN handoff** when BFS stalls (<20 states in
   10s), and **level-to-level transfer** (reuse the previous level's solution).

## What separates the top from baseline (the 30-day report + the code)

- **Action efficiency** (the scoring lever — `min(human/agent,1)²`). Random brute force "may eventually
  complete a level but requires far more actions." The winners turn environment feedback into a strategy
  FAST.
- **Cross-game LEARNING / persistent memory.** Action-effect knowledge carried between games.
- **The hard unsolved part = GENERALIZATION.** All winners < 13%; "agents that learned one game's
  patterns failed on others — limited transfer." (This is exactly our GAP-ARCH-FEATURES finding: our
  verifier doesn't transfer cross-game, LOO-AUROC 0.503.)

## How this maps onto Carnot (the gaps it exposes)

| Top-agent technique | Carnot today | Gap |
|---|---|---|
| **CNN frame-change / clickability predictor** (StochasticGoose, the #1 idea) | blind centroid-clicking + BFS; NO learned action-effect model | **biggest missing piece** — directly attacks action efficiency, the scoring lever |
| **CNN/ResNet/attention VALUE model** over the raw grid | LINEAR/logistic head over 41 hand-features (weak; LOO=chance) | our `.414` A2 (richer features) is the right direction but stops short of a learned CNN |
| **Persistent action-effect memory across games** (PersistentAEM) | re-derive per game | no cross-game action memory |
| **Prioritized experience replay + expert injection** | none (we don't RL-train an agent) | a different paradigm we haven't tried |
| **Per-game TimeBudget within 12h** | `explore_budget=80` + MAX_ACTIONS | no global wall-clock budgeting across ~110 games |
| Offline BFS over the real game + **hidden-field probing in the state hash** | offline BFS yes; hidden-field probing NO | = our GAP-ARCH-GRID-ONLY-STATE (ka59/ar25 L2 stall) — they SOLVED it via state-hash probing |
| Level-to-level transfer | `recommend_approach` (cross-GAME) | they do intra-game level transfer; we route cross-game |

## The strongest method to steal (flag for the roadmap)

**A CNN frame-change / clickability predictor** is the highest-leverage import: it is the leader's core
idea, it directly targets action efficiency (the score multiplier we're leaving on the table with
RESET-replay + blind clicking), and we have nothing like it. Pair it with **hidden-field probing in the
state hash** (their fix for exactly our ka59/ar25 deepening-tail stall). Both are CNN-over-grid / search
techniques, not LLM — cheap to run offline, and they attack the two things the 30-day report says matter:
efficiency + generalization.

**Caveat / honest read:** these are a DIFFERENT paradigm (learned RL action-model + replay) from Carnot's
verifier-energy thesis. The leader is at 1.2% — nobody has cracked this, so there's no proven recipe to
copy wholesale, only techniques. The frame-change predictor + state-hash hidden-field probing are the two
that (a) are proven to help on the public board and (b) slot into our existing offline-search +
value-head architecture without abandoning the verifier thesis.

**→ Roadmap flag (.415/.416):** add a CNN frame-change/clickability predictor to the live explorer
(action-efficiency lever) + hidden-field probing in the state hash (the deepening-tail fix). These
complement the queued `.414` A1/A2 (integration + richer features) and the `.415` trust-energy task.

Sources: Kaggle `arc-prize-2026-arc-agi-3` leaderboard + public kernels
(`nihilisticneuralnet/0-46-...persistent-memory-bfs`, `imaadmahmood/stochasticgoose-cnn-frame-change-agent`,
`vyankteshdwivedi/arc-agi-3-hybrid-solver-bfs-cnn-heuristics`); arcprize.org/blog/arc-agi-3-preview-30-day-learnings;
digg.com/ai/k8o9t7me (Tufa 0.68→1.17 jump).
