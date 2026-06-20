# ARC-AGI-3 human baseline structure + the underused human-replay signal (2026-06-20)

Operator question 2026-06-20: do humans see the hidden games machines can't train on?

## Answer (from arcprize.org + docs.arcprize.org/methodology)

- **Published human data = the 25 PUBLIC games only**: 342 step-by-step human replays, 145 solves,
  458 participants. The open human dataset is the same 25 games machines can train on.
- **Scoring normalizes AI per-level against a human baseline**: `level_score =
  (human_baseline_actions / ai_actions)^2`, capped at 1.15x human; RHAE = median-human action
  efficiency per level per game.
- **Humans MUST have played the hidden eval games** (high-confidence structural inference, not
  directly stated to avoid leaking eval detail): the score formula on a hidden game requires a
  human_baseline_actions value for that game's levels, which can only exist if humans played it.
  So ARC's calibration panel plays the held-out games — but **first-contact** ("every participant
  only saw environments once, single attempt"), NOT as training.
- **Both humans and machines face the hidden games first-contact.** Neither trains on them. The
  human 100% (4/5 on the reference panel) is itself a first-contact GENERALIZATION result. So the
  human-vs-AI gap (100% vs ~1% field-wide) is genuine novel-game-generalization capability, not an
  unfair data advantage.

## The actionable edge: mine the 342 human replays (underused)

ARC open-sourced 342 human step-by-step replays for the 25 public games. This is IMITATION /
behavior-cloning data we are not using — exactly the "expert injection" the [0.46] leaderboard DQN
seeds into its prioritized replay. Concrete uses:
- Bootstrap the CNN frame-change / clickability predictor (GAP-ARCH-FRAME-CHANGE-PREDICTOR) with
  human action-effect demonstrations instead of only self-generated transitions.
- Seed the value/energy heads with human progress trajectories (human "steps-to-go" is a strong,
  efficient label — humans are at the action-efficiency ceiling the score rewards).
- Use human action counts as the efficiency target the agent should approach (the score cap is
  1.15x human, so matching human action-economy is the ceiling).

## Strategic reframe

1. The wall is GENERALIZATION — same conclusion as the 30-day report + GAP-ARCH-FEATURES (LOO=chance).
   The energy-over-structural-features strategy aims at exactly this.
2. Offline dev is structurally blind to the eval (we only have 25 games + replays); manufactured
   held-out VARIANTS are the only proxy for the unseen games — which is why the variant work matters.
3. The human-replay corpus is the one piece of "human sees what machines can train on" that IS public
   and that we should exploit.

Cross-refs: `arc-leaderboard-competitive-intel-2026-06-20.md` (expert-injection technique),
`arc-frame-change-predictor-spec.md` (the predictor to bootstrap), `arc-energy-augmented-strategy.md`
(the generalization differentiator); sources: arcprize.org/blog/arc-agi-3-human-dataset,
docs.arcprize.org/methodology, arcprize.org/blog/arc-agi-3-preview-30-day-learnings.
