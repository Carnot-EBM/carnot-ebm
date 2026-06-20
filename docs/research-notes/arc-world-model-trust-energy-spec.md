# ARC world-model trust energy — the oracle-distinct EBM slot (.415 task spec)

**Origin:** 2026-06-20 operator question "can the Carnot EBM help our ARC-AGI-3 effort?" + the
codebase audit that answered it. The audit found the project has TWO energies, with opposite verdicts
for ARC:

- The **FoVer / text energy ensemble** (the 0.9131 headline: Curry-Howard, logical-consistency,
  HalluGuard) scores reasoning-TEXT steps. Its only ARC contact is a synthetic 3×3-grid demo harness
  (`arc_agi3_harness.py`) that is CIRCULAR (the encoder writes the "proof" text from oracle-known
  progress, then the verifier detects the planted contradiction — the flagged exp3929 tautology).
  Grafting it onto real ARC is **low-leverage / a distraction**.
- The **world-model consistency energy** (`energy = 1 − dynamics_accuracy`) is grid-grounded, real
  (a wrong prediction is fully possible — no oracle), and **already wired into the submitted agent**:
  `arc_competition_agent.py:698` computes `dsl_model.consistency_energy(...)`, and `:779-780` gates
  planning on `WorldModelVerifier(self.transitions).score(engine).accuracy < 0.5` ("too weak to trust").

This spec is about strengthening the SECOND one — the only ARC slot where energy is both on the
critical path AND oracle-distinct.

## The gap (why the current gate is weak)

The live trust gate is a **binary threshold**: `accuracy < 0.5 → don't plan`. Two problems:

1. **It's a hard cutoff, not a ranking.** When the E3 proposer emits several candidate world-models,
   the agent takes the FIRST that clears 0.5 — not the one that GENERALIZES best on held-out
   transitions. A 0.55-accuracy model that overfits the observed prefix is trusted over a 0.52 model
   that generalizes.
2. **It conflates Markov and hidden-state games.** For a **Markov** game (visible grid + action ⇒ next
   grid) you can just RUN the model and compare — execution is a cheap oracle, energy is redundant, and
   "verifier IS oracle" is CIRCULAR (label `execution_grounded`, `verifier_is_oracle: true`). For the
   ~11 **hidden-state** games (visible grid does NOT determine the next state — ka59 step counter, ar25
   undo stack) there is NO cheap oracle for "is this induced model right / which candidate generalizes."
   That is exactly the `verifier_is_oracle: false`, gate-eligible frontier (CLAUDE.md Circularity
   Discipline) — the project's actual moat claim.

Measured precedent (`results/arc3_m2_world_model.json`): the consistency energy already SEPARATES
hidden-state from Markov games (mean energy 0.88 vs 0.75 in the memorization condition,
`separation_accuracy_memorization: 0.8`). Modest but real, oracle-distinct signal — the raw material
for a learned ranking energy.

## The .415 task

**Title:** Learned world-model TRUST ENERGY for hidden-state ARC games — replace the binary
`accuracy < 0.5` gate with an oracle-distinct ranking energy.

**Deliverable:** `results/experiment_<id>_world_model_trust_energy.json` + a module that, given a game's
recorded transitions and SEVERAL candidate induced engines, computes a learned/calibrated energy
`E(transitions, engine) → trust` that RANKS the candidates by held-out generalization (not the observed
prefix), and selects the best — specifically discriminating on the hidden-state games.

**Method (offline, zero live quota):**
1. For each hidden-state game (the ~11 with `verifier_is_oracle: false` candidates), collect ≥3
   candidate induced engines (from cached E3 inductions + perturbations) and split their transitions
   into observed-prefix vs held-out-suffix.
2. Compute the consistency energy on BOTH splits per candidate. The discriminating feature is the
   prefix-vs-held-out energy GAP (overfit candidates have low prefix energy, high held-out energy).
3. Learn/calibrate a ranking (logistic or isotonic over the energy features) whose label is "this
   candidate has the lowest held-out misprediction" — a held-out generalization target, not the prefix.
4. Wire it behind a flag in `arc_competition_agent.py` to REPLACE the hard 0.5 cutoff for hidden-state
   games (keep the cheap execution check for Markov games — don't pay energy where an oracle exists).

**Falsifiable acceptance gate (oracle-distinct):**
- `verifier_is_oracle: false` (REQUIRED — this is the whole point; a circular win does NOT count).
- The learned trust energy picks the best-held-out-generalizing candidate at a rate STRICTLY ABOVE the
  "first-clears-0.5" baseline, on a held-out set of hidden-state games, n ≥ the hidden-state game count.
- **FALSE_NEGATIVE_RISK guard:** include a POSITIVE CONTROL (a Markov game where execution adjudicates)
  to prove the harness can detect a real win; if the energy does NOT separate on hidden-state games,
  report the honest null — "world-model trust energy does not beat the binary gate" is a valid finding,
  not a failure to hide.

**inference_substrate:** `verifier_ensemble_against_cached_candidates` (scores cached candidate engines
offline; no live LLM, no GPU). **Precondition:** the import smoke
(`.venv/bin/python -c "from carnot.agentic import arc_solver_kit as k; k.offline_arcade()"`), NOT a
pytest target (the .414 A4/A5 block was a non-existent pytest file).

**Why this is the right EBM move (not a distraction):**
- On the critical path (the live agent's plan-vs-keep-exploring decision, `arc_competition_agent.py:780`).
- Oracle-distinct (hidden-state games, where the project's discipline says the moat lives).
- Incremental on shipping code (strengthen the existing gate), not a new subsystem.
- Dovetails with the queued `.414` A4 (goal-predicate + HUD-register state) — A4 is hard precisely
  because hidden state is where the deciding variable lives; a trust energy over that hidden state is
  the verifier-side complement.

**Sequencing:** AFTER the `.414` score-drivers (integration A1 + features-v3 A2 land first — they move
the 0.08 directly). This is a `.415+` task: real moat work, lower immediate score-delta than the
integration, but it is the one place the energy/EBM thesis is genuinely load-bearing for ARC.

**Cross-refs:** `ops/verifier_gaps.md` GAP-ARCH-WORLD-MODEL-TRUST-ENERGY; `arc_world_model_dsl.py:305`
(`consistency_energy`); `arc_executable_world_model.py:146` (`WorldModelVerifier`);
`arc_competition_agent.py:698,779-780` (the live gate); `results/arc3_m2_world_model.json` (the
separation measurement); CLAUDE.md "Circularity / Oracle-Distinctness Discipline".
