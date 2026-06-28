# StochasticGoose (ARC-AGI-3 preview winner) code — borrowability analysis (2026-06-28)

**Origin:** operator 2026-06-28 (option 2 of the fresh-SOTA-options scan): "study the winner's code and
report what's borrowable." Source: `github.com/DriesSmit/ARC3-solution` (Tufa Labs / Dries Smit), the
ARC-AGI-3 Agent Preview competition winner (12.58% preview → 0.25% full unseen, per arXiv:2603.24621).
Cloned to /tmp (NOT into the repo tree — embedded-gitlink discipline) and read in full (~882 LOC).

## Verdict: nothing borrowable that Carnot lacks — we independently built the same approach AND tested it to the same conclusion.

The winner's agent ("StochasticGoose") is a frame-change-prediction CNN that biases exploration toward
actions predicted to change the frame. Component-by-component, Carnot already has every piece:

| StochasticGoose component (custom_agents/action.py + README) | Carnot equivalent (already built + tested) |
|---|---|
| `ActionModel`: 16-ch one-hot 64×64 → 4-layer shared conv backbone (32→64→128→256) → action head (ACTION1-5) + conv coordinate head (64×64 click logits, 2D inductive bias); predicts P(frame changes) | `arc_frame_change_predictor.py:63` `SmallFrameChangeCNN` — "CNN with click heatmap and directional heads" (REQ-ARC-FCP-4490; exp4490/4547) |
| Supervised (state,action)→frame_changed, BCE + light entropy reg; online | `arc_online_action_effect_scorer.py` online action-effect CNN (exp4710 / exp4726) |
| Experience buffer `deque(maxlen=200000)` + hash-dedup on (frame,action) | `arc_online_action_effect_scorer.py` buffer + Carnot's state-hash dedup throughout the solver |
| Dynamic reset on new level (clear buffer, reset model) | `arc_online_action_effect_scorer.py:331` `reset(level=...)` — restores the prior snapshot, clears Adam state, `_buffer.clear()` on level-up |
| Hierarchical sampling (action type, then conv coordinate); sigmoid probs | `SmallFrameChangeCNN` click-heatmap + directional heads; `arc_live_ttt.py:138` conv per-cell heads (2D bias) |
| Exploration biased toward predicted-change actions | exp4629 action-effect CNN + PersistentAEM live ranker (BANKED, efficiency 2.0→1.0) |

**The decisive convergence.** Carnot's instances of this exact family reached the winner's exact outcome:
- exp4629 (action-effect CNN + PersistentAEM): `validated_positive` — action-EFFICIENCY up (2.0→1.0
  actions-to-first-levelup), **but no first-win/generation lift**.
- exp4710 / exp4726 (online action-learning CNN with level-up reset): **honest_null** — online learning
  gives **no first-win lift** (the signal is too sparse; `winning prefix still not proposed`).

This mirrors StochasticGoose's published trajectory: 12.58% on the preview (efficiency where games are
partly crackable) → **0.25% on the full unseen set, indistinguishable from frontier LLMs** (arXiv:2603.24621).
Two independent teams built the same frame-change-CNN-biased-exploration agent and hit the same wall.

## Borrowable micro-deltas (all SECONDARY — none move the deliverable)
- The winner's exact CNN depth (4-layer 32→64→128→256) and the pure-conv coordinate head are marginally
  different from `SmallFrameChangeCNN`'s specifics. *Potential* tiny action-efficiency bump on
  already-partly-solved games — but Carnot's click targeting is already near-optimal (the click-heatmap
  finding: 99.1% of clicks land ≤2px of an object centroid → `guidance_not_coverage`), so the EV is ~0.
- No reset/buffer/dedup idea is new (Carnot has all three).
- **There is NO generation/first-win lever in the winner's code** — because the winner doesn't have one
  either (hence its collapse).

## What this confirms (the real value of the exercise)
1. **Carnot is at architectural parity with the frontier preview-winner.** Our standing 0.08 is not a gap
   to a known better method — it is the same family, and the winner's full-set score (0.25%) collapsed
   to ~our ballpark / frontier-LLM baseline. We are not behind a borrowable trick.
2. **Convergent independent confirmation of the information-theoretic generation wall.** Two teams,
   different code, same architecture, same outcome (efficiency yes, generalizable first-win no). This
   strengthens the `WALL_IS_HIDDEN_STATE` closure with external, independent evidence.
3. **Option 2 closes with no code to borrow.** The deliverable stands: the 0.08 submission (competitive)
   + the FoVer paper. The energy verifier's value remains in verification-bottlenecked domains.

## Cross-references
- `docs/research-notes/arc-generation-sota-fresh-options-2026-06-28.md` (the options scan; arXiv:2603.24621).
- `docs/research-notes/arc-generation-program-2026-06-28.md` (the collapsed generation program).
- Carnot equivalents: `arc_frame_change_predictor.py` (exp4490/4547), `arc_online_action_effect_scorer.py`
  (exp4710/4726), exp4629 (action-effect CNN banked efficiency-positive), `arc_live_ttt.py`.
- Winner: `github.com/DriesSmit/ARC3-solution`; report arXiv:2603.24621.
