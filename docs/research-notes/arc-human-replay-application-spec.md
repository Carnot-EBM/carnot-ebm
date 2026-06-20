# Applying the ARC human replays to improve LIVE solves — grounded spec (2026-06-20)

Operator: how can we apply the 342 human step-by-step replays to improve our live solves?

## The data (VERIFIED, not assumed)

- **Official source:** ARC Prize Public Demo dataset (blog: arcprize.org/blog/arc-agi-3-human-dataset →
  dub.link/vfwCqvb). Use the OFFICIAL source for the submission pipeline + verify its license is
  CC0/MIT-0-compatible before bundling any replay-derived weights.
- **Usable mirror (inspected):** Kaggle `jihangli1121/arc-agi-3-replays-v1` (CC BY 4.0 — attribution; do
  NOT bundle its `best.pth` into an MIT-0 submission). Contents:
  - `environment_files/<game>/replays/<game>-<uuid>.json` — full step-by-step human replay per game
    (frames + actions; 2–98 MB each → per-step 64×64 frame data).
  - **`action_effect_dict.npz` — a PRECOMPUTED supervised corpus** (the key asset). Verified schema:
    `feature_keys` (14672×256 float32 state features), `action_ids` (1–7), `xs`/`ys` (click coords),
    `frame_deltas` (int, #cells changed, 0–4096), `level_progresses` (0/1, did the action level-up),
    `game_ids`/`game_id_idx`. **14,672 examples across all 25 games** (132–1556 per game).
  - `best.pth` (191 MB trained model — reference baseline only; do not bundle).
- **Key statistic:** humans caused a frame change on **14,243 / 14,672 actions (97%)** — i.e. humans
  almost never waste an action. THAT is the action-efficiency the score rewards
  (`(human_actions/ai_actions)²`). 132 actions caused a level-up (the sparse win signal).

## The honest framing (load-bearing)

These replays are for the **25 PUBLIC games**, which we already solve via mode-1 banked replay. They
improve LIVE (hidden-eval) solves ONLY through models that **GENERALIZE** to unseen games — NOT by
memorizing the public games. So every application below is "train a generalizing model on human data,"
and its payoff is measured by the variant-transfer rate (currently 7/25) and ultimately the leaderboard,
not by public-game solve count.

## Applications, prioritized

### 1. Frame-change / clickability predictor — HIGHEST LEVERAGE, data is READY (GAP-ARCH-FRAME-CHANGE-PREDICTOR)
The corpus IS the training set: `(state, action, x, y) → frame_delta`. Train a model predicting
P(frame changes | state, action, click-cell) and rank the explorer's candidate actions by it — the
StochasticGoose edge (the leaderboard leader's core idea), directly attacking action efficiency. The
human data de-risks the queued predictor task: real labels exist, and a precomputed dict bootstraps it.
**Recompute the 256-dim features with OUR frame featurizer** from the raw replay frames so the predictor
is compatible with the live agent's frame-only access (don't depend on the mirror's opaque featurization).

### 2. Behavior-cloning action PRIOR — CHEAPEST IMMEDIATE WIN
`state → action_id` (+ click heatmap from xs/ys) = a human prior over the explorer's candidate
enumeration: try the actions/locations humans use first. Even with NO trained net, the marginal
distribution is a free prior (ACTION6 clicks dominant @4359, directional 1–4 ~2000 each, ACTION7
~never @12) — order the frontier by it instead of raster/centroid order.

### 3. Value/energy head bootstrap — feeds the energy-augmented strategy
Use `level_progress` + position-in-trajectory as steps-to-level-up labels to bootstrap the cross-game
value head / the contrastive energy head from HUMAN (efficient) trajectories — 14,672 states vs our ~480
banked. Feeds the energy-augmented strategy (contrastive energy over human-vs-corrupted states). Same
gate as everything: generalizes only if the features are STRUCTURAL (GAP-ARCH-FEATURES) — human data adds
volume + quality but does not by itself fix the frame-marginal transfer ceiling.

### 4. Action-efficiency target / RHAE calibration
Human action COUNTS per level = the score denominator. The replays give the human action economy per
public game → an offline target to measure "how far from human-efficient is our agent" (public-game-only,
but the closest RHAE calibration we have without the hidden games).

## Caveats (do not skip)

1. **Public-games-only → value is via generalization, not memorization.** Track the variant-transfer rate.
2. **Recompute features OUR way** from raw frames (don't inherit the mirror's 256-dim featurization).
3. **Licensing:** official ARC source for the pipeline; verify CC0/MIT-0 before bundling weights. The
   jihangli mirror is CC BY 4.0 — fine for understanding/format, attribution required, do not bundle its
   best.pth.
4. **Frame-only at eval:** the live agent sees only rendered frames (no `env._game`); train the predictor
   on frame-derived features so it transfers to the live path.

Cross-refs: `arc-frame-change-predictor-spec.md` (the predictor — data now confirmed available),
`arc-energy-augmented-strategy.md` (value/energy bootstrap), `arc-human-baseline-and-replay-signal.md`
(why this is the underused signal), `arc-leaderboard-competitive-intel-2026-06-20.md` (expert-injection).
