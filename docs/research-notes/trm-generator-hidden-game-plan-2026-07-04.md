# From public-game TRM training to a hidden-game-capable live generator: a staged plan (2026-07-04)

**Provenance:** operator question, granting the premise that TRM training on the 25 public games
(via the newly-found human replay corpus, once the staging bug is fixed --
`docs/research-notes/human-replay-corpus-staging-bug-and-opportunity-2026-07-04.md`) succeeds: *"what
is the plan for a TRM based generator that may solve the hidden game levels in the live agent?"* This
note answers that specifically -- it is not a re-litigation of whether the training itself will work
(that is what the leave-one-game-out pilot in
`docs/research-notes/trm-arc-action-sequence-generator-2026-07-04.md` tests).

## The framing this plan has to respect

A TRM trained on the 25 public games' mechanics does not directly solve a hidden game, because the
hidden game's specific rules are, by definition, something the model never saw. This is not new to
this note -- it is this project's own settled framing
(`feedback_arc_value_is_process_not_weights.md`, memory): weights trained on public games do not
transfer to a hidden game; a reusable *process* does. Every stage below is built around what
generalizes (an architecture, a recipe, an adaptation mechanism), never around the idea that the
Stage 1 checkpoint itself is the hidden-game solver.

## Stage 1 -- public-game training produces a validated recipe + general prior, not a solver

The output that matters from training on all 25 games' human-winning trajectories (once
`data/arc_public_demo_human_replay_corpus/` is re-staged to preserve win signal) is not "TRM that
solves ar25/bp35/etc." It is:

- A working architecture: recursion depth, a **dynamic** (not fixed) recursion schedule per
  arXiv:2604.07822's strongest-extrapolation finding, and an overthinking-calibration curve (accuracy
  vs. iteration count, since the same paper shows this can peak and then degrade).
- A validated frame/action encoding scheme: how an ARC-AGI-3 frame (grid) and an
  `{"action": int, "data": {"x": int, "y": int}}`-shaped step get turned into TRM's fixed-tensor
  input/output format -- the genuine architectural adaptation flagged as open in the prior note.
- A general-prior checkpoint capturing whatever *is* shared across ARC-AGI-3 games specifically
  (common visual conventions -- background colors, sprite-like objects; common action semantics --
  which of the ~6 actions tend to matter and when; general spatial/procedural patterns), not any one
  game's win condition.

**How Stage 1's success is measured, honestly:** the leave-one-game-out pilot from the prior note.
If TRM only memorizes training games and shows no held-out generalization signal, Stage 1 has not
actually produced a usable prior, and Stages 2-4 below are not worth building on top of it -- this is
the gate, not a formality.

## Stage 2 -- wire TRM in as a generator feeding the EXISTING verifier gate, never a replacement for it

Per "ARC Live-Path Reachability Discipline" (CLAUDE.md), this must slot into `E3AgentPolicy`'s
existing cascade (`python/carnot/agentic/arc_competition_agent.py`), not sit as a parallel standalone
script the live agent cannot reach. Concretely:

- TRM proposes candidate action sequences, warm-started from the Stage 1 checkpoint, given the
  agent's current frame/state.
- The existing `WorldModelVerifier` scores and gates TRM's proposals exactly as it does for any other
  candidate source (StepwiseExplorer, the online-induced world model) -- no special-cased bypass.
- TRM can only ever *add* a candidate to the pool the verifier already selects from. It never removes
  or disables the existing exploration/search fallback.

This is deliberately low-risk by construction: the integration point is additive, and the existing
verifier-routing structure is unchanged. It is also the direct instantiation of this project's
generator/verifier split -- TRM is a generator, never treated as an oracle.

## Stage 3 -- online adaptation during play is the part that actually lets TRM help on an unseen game

This is the load-bearing stage. Without it, Stage 1's checkpoint is just a fixed prior with no
game-specific knowledge of whatever hidden game is in front of the agent. As the live agent explores
the hidden game and accumulates its own transitions (including failed attempts and `GAME_OVER`
events, exactly the kind of session structure confirmed present in the human replay corpus), it
should periodically **fine-tune the TRM checkpoint on just that game's accumulated data so far** --
full fine-tune, not LoRA, per this project's own TRM test-time-adaptation finding
(`reference_trm_tta_mcgovern.md`, memory: full-FT beats LoRA for OOD adaptation). This mirrors the
exact pattern the scored agent already uses for its world-model component (`arc_live_ttt` / online
world-model induction gated by `WorldModelVerifier`) -- so it is architecturally consistent with how
the live agent already handles novelty, rather than introducing a new, unprecedented mechanism. TRM's
recursive refiner becomes another component that gets induced/adapted online, alongside the existing
world model.

**Honest open question for this stage:** how much within-game data is enough to make a full-FT
adaptation step worthwhile, and how often to trigger it (every N actions? every level-up? every
`GAME_OVER`?) without spending so much wall-clock on fine-tuning that it eats into the action budget
RHAE penalizes. This needs its own small, falsifiable measurement before being wired in for real --
not assumed to just work at whatever cadence seems reasonable.

## Stage 4 -- mandatory guardrail: TRM must never be allowed to regress the baseline

If TRM's candidates consistently fail the verifier gate, or don't demonstrably beat the existing
exploration mechanism on a held-out measure, the agent falls back to what already works, unchanged.
This is the same `solve_rate_dropped` guardrail pattern already specified in `exp4490`'s own capability
spec (`openspec/capabilities/arc-human-replay-frame-change/spec.md`) for the frame-change predictor --
efficiency or capability gains from a new component must never come at the cost of solve rate. Applied
here: TRM integration is reversible and additive at every stage; nothing about wiring it in should be
able to make the live agent worse than it already is.

## What this note is explicitly NOT proposing

- Not proposing to skip Stage 1's own gate (the leave-one-game-out pilot). Granting the premise "TRM
  training on the 25 public games succeeds" does not mean skipping the honest check for what "succeeds"
  actually has to mean (held-out generalization, not memorization).
- Not proposing Stage 3's online-adaptation cadence be decided by intuition -- it needs its own
  measurement, flagged above, before being wired into the live path for real.
- Not proposing TRM replace or bypass the existing verifier-gated candidate architecture at any stage.
  Every stage keeps the existing `WorldModelVerifier` as the arbiter.
- Not touching the live/scored submission stack directly from this note. Any pilot built toward this
  plan is `solve_provenance: development_proxy` (offline, public-games-only) until each stage is
  independently validated, per "ARC Solve Reproducibility + Solver-Reuse Discipline" and "ARC Live-Path
  Reachability Discipline."

## Cross-references

- `docs/research-notes/arc1-arc2-capability-transfer-to-arc3-2026-07-04.md` -- the broader ARC-1/2
  capability-transfer question, the ARC-1-to-ARC-2 degradation calibration point, and the parallel
  unbuilt methodology bridge (Family-B induction in `exp4544`)
- `docs/research-notes/trm-arc-action-sequence-generator-2026-07-04.md` -- the original TRM proposal
  and the leave-one-game-out pilot this plan's Stage 1 depends on
- `docs/research-notes/human-replay-corpus-staging-bug-and-opportunity-2026-07-04.md` -- the training
  data source Stage 1 needs re-staged before it can run
- `feedback_arc_value_is_process_not_weights.md` (memory) -- the framing every stage above respects
- `reference_trm_tta_mcgovern.md` (memory) -- the full-FT-over-LoRA finding underlying Stage 3
- `python/carnot/agentic/arc_competition_agent.py` (`E3AgentPolicy`) -- the live cascade Stage 2 wires
  into; the existing `arc_live_ttt` / online world-model induction pattern Stage 3 mirrors
- `openspec/capabilities/arc-human-replay-frame-change/spec.md` -- the `solve_rate_dropped` guardrail
  pattern Stage 4 reuses
- CLAUDE.md "ARC Live-Path Reachability Discipline" -- the two live entrypoints and the
  registry-precheck-before-building-a-parallel-solver rule this plan is designed to satisfy at every
  stage
- CLAUDE.md "ARC-AGI-3 IS a Live Hidden-Game Discovery Agent" -- the foundational framing (the
  deliverable is runtime discovery, not trained weights) this entire plan is structured around
