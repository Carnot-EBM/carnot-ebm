# Does ARC-AGI-1/2 capability apply to ARC-AGI-3? (2026-07-04)

**Provenance:** operator question: *"do we know if ARC-AGI-1 and ARC-AGI-2 capability has any
potential application to ARC-AGI-3 games?"* Investigated directly against existing project artifacts
and memory rather than reasoned from first principles -- the honest answer is nuanced: no direct
empirical test exists, but there is a real methodological bridge already identified and never
finished building, plus a real pre-trained asset worth being aware of.

## Headline: no direct test has been run

Searched specifically for an experiment that took ARC-1/2-capable models or methods and measured
them against ARC-3 directly. There isn't one. This is a genuine gap in this project's own record,
not a measured negative -- nobody has tried it and reported a result either way.

## The one adjacent data point we do have, and it should temper expectations

The project's own GAP-3/GAP-4 program measured transfer *between* ARC-1 and ARC-2 themselves -- the
two most similar variants of each other, both static single-shot grid-transformation puzzles -- and
found real, substantial degradation. An LLM rule-induction-and-verify pipeline (induce a Python
transform function from demo pairs, verify it by execution) scored 0.93 induction rate / 0.90
precision on ARC-1, dropping to 0.57 / 0.47 on ARC-2 (`ops/verifier_gaps.md`, "ARC-2 TRANSFER PROBE
(2026-06-10, 5/5 adversarially CONFIRMED)"). If capability degrades that much moving between two
variants of the *same task format*, that is a real reason for caution before assuming ARC-1/2 skill
would transfer cleanly to ARC-3 -- a much bigger domain jump: static grid transformation versus live,
multi-step, interactive gameplay with a hidden win condition that must itself be discovered.

## The bridge that does exist: shared methodology, not shared weights or scores

This project's own SOTA-ingestion work (`docs/research-notes/arc-llm-inducer-sota-420.md`,
2026-06-21) already identified the real connection. The winning ARC-AGI-3 architecture -- Family-B
"Executable World Models" (arXiv:2605.05138, "Executable World Models for ARC-AGI-3 in the Era of
Coding Agents") -- uses the *exact same paradigm* as the ARC-1/2 rule-induction work this project
already built (`ops/verifier_gaps.md` GAP-3/GAP-4; cf. "ALGO: Synthesizing Algorithmic Programs with
LLM-Generated Oracle Verifiers", arXiv:2305.14591, and "Procedural Refinement by LLM-driven
Algorithmic Debugging for ARC-AGI-2", arXiv:2603.20334): an LLM induces executable code from a
handful of examples, and that code is verified by actually running it against held-out data, refined
via counterexample-guided debugging (arXiv:2606.11521, "Counterexample Guided Learning in the Large
using Reasoning Agents"). For ARC-1/2 the induced code is a static input-to-output transform
function; for ARC-3 it is a dynamics/transition model plus a goal-detection predicate. Same method,
different target -- the same "process transfers, weights don't" shape found for the TRM-as-generator
thread (`docs/research-notes/trm-generator-hidden-game-plan-2026-07-04.md`), just at the level of how
you build a verifier-checked inducer rather than what a recursive-refiner learned.

**The catch: this was mapped, never built.** The 2026-06-21 note explicitly flagged combining Family-B
induction with counterexample-guided refinement inside `exp4544`'s existing GOAL+DYNAMICS proposer as
a `.421` candidate. Per the levers-tried tracking
(`docs/research-notes/arc-agi3-levers-tried-x-verdict-2026-06-25.md`), it is still marked
`unbuilt_mapped_only` -- nobody has actually wired it in.

## Note: pre-trained TRM models already exist for ARC-1/ARC-2 -- this is a real, available asset

Separate from the methodology bridge above, and relevant to anyone weighing whether an ARC-1/2-capable
system needs to be built or trained from scratch: the official `arcprize/trm_arc_prize_verification`
checkpoint (loaded via `scripts/experiments/trm_arc_eval_harness.py`) is a genuinely capable,
already-trained TRM for ARC-1. Reproduced and verified directly in this project on 2026-06-09
(`results/trm_verifier_rerank_opportunity.json`, commit `5f56ccc37`): pass@2 ~0.52, pass@1000 ~0.62 on
a 29-task ARC-1 subset, matching the checkpoint's known published performance range. This is not
something that needs to be trained -- it already exists, is already cached/loadable in this
environment, and is already confirmed working. (A separate pair of artifacts,
`results/trm_arc_baseline_arc_v1.json`/`arc_v2.json`, show near-zero pass@K for the same checkpoint
name -- confirmed on 2026-07-04 to be a methodology artifact of an unbounded/incomplete-voting eval,
not a weak checkpoint; see the correction note in
`docs/research-notes/trm-arc-action-sequence-generator-2026-07-04.md`.)

**Why this matters for the "does ARC-1/2 capability apply to ARC-3" question specifically:** it means
any future investigation of ARC-1/2-capability-transfer does not need to start with "can we get a
model that's good at ARC-1/2 at all" -- that part is already solved and sitting in this repo. The
open question is purely about the transfer step: does this checkpoint's recursive-refinement
*mechanism* (not its ARC-1-specific trained weights) generalize to the ARC-3 action-sequence domain,
which is exactly what the TRM-as-generator note's leave-one-game-out pilot is designed to test. This
existing checkpoint is also a candidate starting point for that pilot's architecture/hyperparameter
choices (recursion depth, layer count) even though its trained weights themselves would not carry
over to ARC-3.

## What this note is NOT proposing

- Not proposing the ARC-1/2-to-ARC-3 methodology bridge (Family-B induction in `exp4544`) and the
  TRM-as-generator plan are the same thing. They are two independent, parallel bridges from
  ARC-1/2-adjacent work to ARC-3 -- one about a shared code-induction-and-verify methodology, the
  other about a shared recursive-refinement architecture. Either, both, or neither could pan out.
- Not claiming the ARC-1→ARC-2 degradation number (0.93→0.57) directly predicts an ARC-1/2→ARC-3
  transfer rate. It is the closest measured analog available, offered as a calibration point, not a
  forecast.
- Not proposing to build the `exp4544` Family-B integration immediately. This note documents what is
  known and what remains unbuilt; it does not commit to executing it without further scoping.

## Cross-references

- `ops/verifier_gaps.md` "ARC-2 TRANSFER PROBE (2026-06-10)" -- the ARC-1-to-ARC-2 degradation
  measurement
- `docs/research-notes/arc-llm-inducer-sota-420.md` -- the Family-B / counterexample-guided-refinement
  mapping for `exp4544`, `.421` candidate, still `unbuilt_mapped_only`
- `docs/research-notes/arc-agi3-levers-tried-x-verdict-2026-06-25.md` -- the levers-tried table
  confirming the unbuilt status
- `results/trm_verifier_rerank_opportunity.json` (commit `5f56ccc37`) -- the verified, genuinely
  capable pre-trained ARC-1 TRM checkpoint (`arcprize/trm_arc_prize_verification`)
- `docs/research-notes/trm-arc-action-sequence-generator-2026-07-04.md` -- the correction of the
  near-zero `trm_arc_baseline_arc_v1`/`arc_v2` numbers as a methodology artifact, not a weak
  checkpoint
- `docs/research-notes/trm-generator-hidden-game-plan-2026-07-04.md` -- the parallel
  architecture-transfer bridge (recursive refinement mechanism, not trained weights)
- `reference_arc_agi3_sota_and_plan.md` (memory) -- prior citation of Family-B as the SOTA ARC-3
  architecture
- CLAUDE.md "ARC-AGI-3 IS a Live Hidden-Game Discovery Agent" -- the framing both bridges in this note
  respect: transferable process/architecture, not transferable trained weights or scores
