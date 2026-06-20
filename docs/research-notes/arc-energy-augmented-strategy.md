# Energy-augmented ARC strategy — the Carnot differentiator (2026-06-20 operator directive)

**Operator 2026-06-20:** "we definitely want to lean into the energy model possibilities that augment the
approaches that others are taking to improve the outcomes."

This note is the strategic through-line: NOT "copy the leaderboard's CNN/RL techniques," but **graft
Carnot's objective-energy thesis onto them where it attacks the field's actual bottleneck**. It supersedes
the framing of the queued ARC specs (frame-change predictor, trust-energy) — they are now read as the
energy-AUGMENTED hybrid, not pure copies.

## The wall everyone is hitting (the opening)

The ARC-AGI-3 30-day report: every preview winner scored **< 13%**; the named hard problem is
**GENERALIZATION** — "agents that learned one game's patterns failed on others." Reading their code, the
top agents (Tufa StochasticGoose CNN, Blind Squirrel ResNet18 value model, the [0.46] DQN+PER) are all
**learned-from-success regressors / classifiers**: they fit a CNN to "what changed" or "steps-to-go" on
the games they've seen. That is exactly why they memorize and don't transfer.

**Carnot hits the SAME wall today:** our linear value head + the discriminative head are also
learned-from-success — leave-one-game-out AUROC **0.503 == chance** (`results/arc_discriminative_verifier.json`).
So we are NOT behind because we lack their CNN; we are behind on the same axis they are stuck on.

## The differentiator: OBJECTIVE energy is game-agnostic where learned value is not

A learned value `v(state) ≈ steps-to-go` is fit to a game's reward signal → game-specific. An
**objective energy** scores a state/transition by structure that holds REGARDLESS of game —
constraint-satisfaction, consistency, symmetry, conservation. The bet (the project's whole thesis): an
energy grounded in objective structure can transfer across novel games where a learned value cannot,
because the structure is not game-specific. **This is the one move available to us that the pure-RL/CNN
teams have not made** — and it targets the exact problem (generalization) that is capping the whole field
at <13%.

**Honest caveat (load-bearing):** objective energy generalizes ONLY if it is computed over game-agnostic
STRUCTURE, not frame-marginals. Our current energy/verifier features ARE frame-marginals (color counts,
occupancy) — which is *why* our LOO is chance. So the energy-augmentation thesis is GATED on
GAP-ARCH-FEATURES (richer relational / Δframe / invariant features, the `.414` A2 work). Energy over the
right features is the differentiator; energy over the current features is no better than their CNN. We
must prove the transfer, not assume it.

## The three energy-augmentation grafts (each augments a winning technique)

1. **Frame-change predictor → energy-scored PROGRESS, not just change.** Their CNN predicts WHICH actions
   change the frame (exploration efficiency). Augment it: an energy over `(frame, action, predicted-next)`
   that scores whether the change is PROGRESS toward a low-energy (goal-consistent) state, not merely
   non-zero. Their predictor finds the dynamics; our energy says which dynamics are good. Combine in the
   explorer frontier: rank by `P(change) · (−ΔE)`. (Augments StochasticGoose; spec:
   `arc-frame-change-predictor-spec.md`.)

2. **Learned value model → energy verifier trained CONTRASTIVELY on objective violations.** Their ResNet18
   value (and our linear head) regress to success-distance. Augment it: train the energy with NCE /
   contrastive on (constraint-satisfying, constraint-VIOLATING) pairs over game-agnostic structure — the
   off-path-negatives + discriminative head are step 1 of this; the generalization comes from the
   FEATURES being structural. The energy landscape, not a point regressor, is what search descends.
   (Augments Blind Squirrel's value model + our discriminative verifier; gated on GAP-ARCH-FEATURES.)

3. **World-model induction → energy TRUST gate over hidden-state games.** Their agents trust the first
   model that clears a threshold. Augment it (already live + specced): the consistency energy ranks
   candidate induced models by held-out generalization, specifically where execution is no oracle (the
   hidden-state games). This is the oracle-DISTINCT slot — the moat. (Spec:
   `arc-world-model-trust-energy-spec.md` / GAP-ARCH-WORLD-MODEL-TRUST-ENERGY.)

## The sequenced program (this is the ARC research spine now)

| Step | Task | Role | Status |
|---|---|---|---|
| 0 | `.414` A1 integration | wire the stronger explorer (the substrate the energy rides on) | running |
| 1 | `.414` A2 features v3 | **the gate** — structural features so energy can transfer (LOO > 0.6) | running |
| 2 | frame-change predictor + **energy progress score** | action efficiency + energy-guided exploit | queued .415/.416 |
| 3 | world-model **trust energy** (hidden-state) | the oracle-distinct moat | queued .415 |
| 4 | contrastive energy value over structural features | the generalizing verifier | follows step 1 |

**The thesis in one line:** steal their learned DYNAMICS (frame-change predictor) for exploration
efficiency; add Carnot's OBJECTIVE ENERGY over STRUCTURAL features for progress-scoring + cross-game
generalization — the axis the whole field is stuck on. Proven only when LOO-AUROC clears chance (step 1);
until then it is the hypothesis we are testing, stated honestly.

**Cross-refs:** `arc-leaderboard-competitive-intel-2026-06-20.md` (what they do),
`arc-frame-change-predictor-spec.md`, `arc-world-model-trust-energy-spec.md`,
`results/arc_discriminative_verifier.json` (the LOO=chance evidence), CLAUDE.md "Circularity /
Oracle-Distinctness Discipline" (the oracle-distinct moat definition), `ops/verifier_gaps.md`
GAP-ARCH-FEATURES (the gate).
