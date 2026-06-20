# CNN frame-change / clickability predictor — action-efficiency task spec (.415/.416)

**Origin:** 2026-06-20 leaderboard competitive intel
(`arc-leaderboard-competitive-intel-2026-06-20.md`). The leader (Tufa Labs, 1.21) and 2nd place (Blind
Squirrel, ResNet18) both win on a **learned action-effect model**: a CNN that predicts which actions
(and click locations) cause a FRAME CHANGE, so the agent stops wasting actions on no-ops. The 30-day
report: StochasticGoose wasted "nearly 350 moves clicking with no result" before learning what was
clickable — then exploited it. Action efficiency is the scoring lever (`min(human/agent,1)²`), and our
0.08 agent is on the wrong side of it (blind centroid-clicking + RESET-replay nav).

## The gap (our explorer is effect-blind)

`arc_graph_explore.py:44 rich_action_candidates(...)` enumerates object-centroid clicks + keyboard
actions and the explorer tries them with **no prioritization by predicted effect** — every candidate is
equally likely to be a no-op. On the OOD-dominant config/click games this burns the action budget (and
the squared efficiency term collapses even on solved levels). We have a value head (steps-to-go) but NO
action-effect / clickability model.

## The task

**Title:** CNN frame-change / clickability predictor for action-efficient exploration.

**Deliverable:** a small CNN `predict(frame) -> (click_heatmap[H×W], directional_change[5])` giving the
probability that each click cell / each ACTION1-5 changes the frame; wired into the explorer to RANK
`rich_action_candidates` by predicted change (try high-change actions first), measurably reducing
actions-to-first-level-up on held-out games.

**Training data (free, self-supervised, offline, zero LLM/quota):** every `(frame, action, next_frame)`
the explorer already generates during a sweep IS a labeled example — label = did the frame change
(binary) + change magnitude. Pool across ALL games (the StochasticGoose/PersistentAEM "persist
action-effect across games" idea). Build a corpus from the existing offline arcade + banked
trajectories; no new env access needed.

**Architecture:** start small — a few conv layers over the 64×64×(color-onehot) grid → a click heatmap
head + a 5-dim directional head (the leader uses a plain CNN; the [0.46] public agent adds CBAM
attention — add attention only if the plain CNN underperforms). torch 2.11 is in the venv AND on Kaggle;
inference is a sub-ms CPU forward pass, live-legal (frame-only, no `env._game`).

**Falsifiable acceptance gate (efficiency, oracle-distinct from execution):**
- On a HELD-OUT set of games, the predictor-ranked explorer reaches the first level-up in STRICTLY FEWER
  actions than the blind-BFS baseline (report median actions-to-first-levelup before/after + the implied
  `min(human/agent,1)²` efficiency delta), frame-only on the offline arcade.
- **FALSE_NEGATIVE_RISK guard:** a POSITIVE CONTROL game where clickability is plainly learnable (most
  clicks are no-ops, a few aren't) to prove the harness detects a real win; if ranking does NOT reduce
  actions, report the honest null.
- It must NOT reduce SOLVE-rate (efficiency gain can't come from giving up on hard games).

**inference_substrate:** training = `verifier_ensemble_against_cached_candidates` (offline, on cached
transitions); the predictor is a CPU CNN at eval. **Precondition:** the import smoke
(`.venv/bin/python -c "from carnot.agentic import arc_solver_kit as k; k.offline_arcade()"`) + a torch
import check — NOT a pytest target.

## Secondary (same theme, cheaper): hidden-field probing in the state hash

The Hybrid-BFS+CNN public agent bakes **hidden-field probing into its state hash**
(`_probe_hidden_fields`) — its solved version of OUR exact ka59/ar25 L2 deepening stall
(GAP-ARCH-GRID-ONLY-STATE: grid-only state can't represent a step counter / undo stack). Extend the
explorer's node-identity key (currently grid + `hud_mask`, `arc_competition_agent.py:319-321`) to probe
and include induced hidden scalars, so deduplication distinguishes states that differ only in hidden
fields. Smaller, surgical, and directly unblocks the deepening tail.

## ENERGY-AUGMENTATION (operator directive 2026-06-20 — the differentiator, not a pure copy)

The pure StochasticGoose CNN predicts WHICH actions change the frame (exploration). Carnot's version adds
an energy layer that scores whether the change is PROGRESS: rank the explorer frontier by
`P(frame_change) · (−ΔE)`, where ΔE is the change in an objective energy (goal-consistency / constraint
satisfaction) between frame and predicted-next-frame. Their predictor finds the dynamics; our energy says
which dynamics are good — exploration × exploitation in one ranking. This is the graft that targets the
field's generalization wall (see `arc-energy-augmented-strategy.md`): the frame-change CNN is game-specific,
but an energy over STRUCTURAL features is the candidate that transfers. NOTE: the energy term is only
additive value once the features are structural (GAP-ARCH-FEATURES / `.414` A2); ship the predictor first
(it stands alone on efficiency), then add the energy progress-score once A2 lands.

## Honest read (paradigm + sequencing)

- This is a LEARNED action-model — a DIFFERENT paradigm from Carnot's verifier-energy thesis. It
  COMPLEMENTS, not replaces: it makes the explorer (tier-1) action-efficient; the verifier/energy still
  routes + grounds. Nobody has cracked ARC-AGI-3 (leader 1.2%), so this is a proven-useful TECHNIQUE, not
  a recipe to copy wholesale.
- **Highest-leverage of the queued ARC work for the SCORE** (action efficiency is the metric, and we
  have zero of it). Sequence: AFTER `.414` A1 (integration — wires the stronger stack so the predictor
  has a good explorer to rank within), ALONGSIDE/BEFORE the `.415` trust-energy task (trust-energy is
  deeper-moat but smaller immediate score-delta; the frame-change predictor moves 0.08 more directly).

**Cross-refs:** `docs/research-notes/arc-leaderboard-competitive-intel-2026-06-20.md`;
`arc_graph_explore.py:44` (`rich_action_candidates`, the ranking slot); `arc_competition_agent.py:319-321`
(state-hash, the hidden-field slot); `ops/verifier_gaps.md` GAP-ARCH-FRAME-CHANGE-PREDICTOR +
GAP-ARCH-GRID-ONLY-STATE; the public notebooks `imaadmahmood/stochasticgoose-cnn-frame-change-agent`,
`nihilisticneuralnet/0-46-...persistent-memory-bfs`.
