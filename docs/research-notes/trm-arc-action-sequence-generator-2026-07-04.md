# TRM as the ARC-AGI-3 action-sequence generator (2026-07-04)

**Provenance:** outer-loop discussion connecting arXiv:2604.07822 ("Loop, Think, & Generalize:
Implicit Reasoning in Recurrent-Depth Transformers") to `ops/verifier_gaps.md`'s existing, open,
never-attempted `GAP-ARC-TRM-TRAINED-ON-ARC` entry, verified against actual repo state (not memory
alone) before writing this up. Read the primary paper and the cited artifacts before implementation.

## This is not a new idea — it's an existing, unaddressed gap, now with two supporting arguments

`GAP-ARC-TRM-TRAINED-ON-ARC` already exists in `ops/verifier_gaps.md`, status `open`: *"the running
TRM (nano-trm) is sudoku-trained; no TRM is trained on ARC solve traces... a TRM trained on the
accumulated captured ARC trajectories that proposes/refines candidate action sequences."* This note
adds two things that weren't in the original gap entry: a confirmed cross-domain precedent, and a
mechanistic reason from fresh literature for why it should work on ARC specifically.

## The precedent (verified against the actual artifact, not recalled from memory)

`results/experiment_sudoku_energy_vs_ar_v1.json` — the recursive-refinement mechanism used there IS
literally the TRM recipe (confirmed from the artifact's own `refiner_finding` field: `z ← block(x,y,z);
y ← head(z)`, 8 recursion cycles, 2 layers, trained end-to-end with deep supervision on the
generation task). Results: recursive refinement solved 18.2% of held-out Sudoku puzzles vs
autoregressive decoding's ~0-0.2%, while a *near-perfect energy scorer* (NCE loss 0.0001, gold-vs-
corrupt gap of 17) still produced 0% via argmin/descent decoding — worse than random. **This is the
exact failure shape ARC-AGI-3's own program has diagnosed**: GAP-4891 found a correctly-separating
goal-detection energy that doesn't help search reach the goal; the enumeration wall is a generation-
paradigm problem, not a scoring problem. TRM already broke through the identical shape of wall once,
on a different combinatorial domain.

## What's already been tried on ARC with TRM, and why it doesn't count against this

**Correction (2026-07-04, after the operator asked "did we already try that before?"):** an earlier
draft of this note cited `results/trm_arc_baseline_arc_v1.json`/`arc_v2.json` (pass@1000 = 0.005 /
0.025) as evidence of a weak "sudoku-trained checkpoint tested on ARC" attempt. Verified against the
actual artifacts and git history and that attribution was wrong. `arc_v1`/`arc_v2` are just named
checkpoint slots in `scripts/experiments/trm_arc_eval_harness.py`'s own CLI, referring to the
**official `arcprize/trm_arc_prize_verification` checkpoint** (genuinely ARC-trained, not Sudoku).
The 0.005/0.025 numbers are most likely a methodology artifact: the same checkpoint, reproduced
properly on 2026-06-09 (`results/trm_verifier_rerank_opportunity.json`, commit `5f56ccc37`, bounded
to complete each task's full ~1000-augmentation vote) scored pass@2 ~0.52, pass@1000 ~0.62 on a
29-task subset — genuinely strong, matching the checkpoint's known published range. The commit
message for that reproduction explicitly flags that an *unbounded* eval risks incomplete per-task
voting, which plausibly explains the degenerate near-zero numbers in the other two artifacts on the
same checkpoint name. **Corrected conclusion: static ARC-1 grid-transform solving with TRM has been
tried and genuinely works well (~45-62% pass@K on the properly-evaluated official checkpoint) — a
stronger existing precedent than originally stated here, not a weaker one.** This does not change the
note's actual claim, which was never about static grid-transform solving: no TRM has been trained on
or evaluated against ARC-AGI-3's *live, interactive action-sequence* format, in any of these
artifacts, at any performance level. That distinction is what `GAP-ARC-TRM-TRAINED-ON-ARC` and this
note are about, and it holds unchanged after this correction.

Separately, `exp4100`'s "TRM ARC engine" (`docs/research-notes/arc-agi3-levers-tried-x-verdict-
2026-06-25.md`) was retired — but that was a verifier-RFT'd TRM used as a *dynamics/world-model
engine* (predicting state transitions), a different mechanism from directly refining candidate action
sequences. Any task drawn from this note should cite both of these explicitly as scope-adjacent-but-
distinct if the exclusion-manifest lint flags a match, per CLAUDE.md's Failed-Experiment Rerun
Discipline.

## What the new paper (arXiv:2604.07822) adds beyond the Sudoku precedent

Not just an analogy — a mechanistic argument. The paper shows recurrent-depth transformers achieve
**systematic generalization** (composing facts never jointly used in training) that standard
transformers structurally cannot, via a three-stage grokking process, and **depth extrapolation**
(reasoning deeper than training depth) by scaling inference-time recursion. ARC-AGI-3's own challenge
is compositional: a hidden game's mechanics must be composed, freshly discovered, into a winning
action sequence never seen in that exact combination before. That's the same *shape* of problem the
paper studies (though a different domain — symbolic fact-chains vs. spatial/procedural game
mechanics; the transfer is a real, untested hypothesis, not a proven one). Two concrete, actionable
design rules the paper adds if this gets built: train with **dynamic** recursion depth, not fixed
(strongest extrapolation in their results); and explicitly instrument for **overthinking** — accuracy
peaks at some recursion depth and can *degrade* past it, so more iterations is not automatically
better.

## The real engineering gap to bridge

`python/carnot/agentic/arc_solver_kit.py` represents a plan as a list of step dicts: `{"action":
<int>, "data": {"x": int, "y": int}}` — a tractable, structured representation to adapt TRM's
grid-refinement recipe to (encode as a padded tensor of action-ids + coordinates, the same way TRM
handles a padded Sudoku grid), but a genuine adaptation, not a drop-in — TRM currently refines
fixed-size grids, not variable-length action sequences.

**Update 2026-07-04 (operator asked whether human-generated solutions were already available):**
this question is resolved far more favorably than the paragraph below originally assumed. See
`docs/research-notes/human-replay-corpus-staging-bug-and-opportunity-2026-07-04.md`: a real, licensed
human replay corpus already exists (`data/arc_public_demo_human_replay_corpus/`) with **144 complete
winning human trajectories across all 25 public games** (roughly 90,000+ actions), each with validated
per-step level-completion signal. The corpus's *staged* form currently has a bug that drops this
signal entirely (verified: `level_progress=0.0` in all 14,797 staged rows, including full winning
sessions) — the raw source is intact, but needs re-staging before it's usable. This is now the more
promising data source for a training pilot than the 69-level Carnot-agent corpus discussed below,
which remains true but is superseded in priority by the human corpus once re-staged.

**Training data is the real open question, and this note does not assume an answer either way.** The
existing candidate design in `ops/verifier_gaps.md` already specifies the outline (prepare a
trajectory dataset from captured solves, full-FT not LoRA per the TTA-TRM finding, train on a 3090,
wire as a rollout engine) — but commits to that full build before knowing whether 24 games' worth of
captured trajectories is enough signal at all. `reproducible_total_levels=69` is a level count, not a
training-example count — each level's trajectory contributes many (state, action) pairs, so the
effective training set could be meaningfully larger than "69 examples" suggests, but this note does
not assume that either; it should be the first thing measured. (Superseded in priority by the human
corpus above — measure that first.)

## Proposed first pilot — small, falsifiable, before the full build

**Leave-one-game-out overfit-generalization test**, entirely offline, entirely on public games:

1. Count the actual number of (state, action) supervision pairs available across the captured,
   reproduced trajectories (not level count) — the first concrete number this whole gap has been
   missing.
2. Train a nano-TRM-scale model (matching the ~1.8M-param Sudoku recipe) on all-but-one game's
   captured trajectories, refining a whole candidate action sequence per the gap's own framing
   ("proposes/refines candidate action sequences"), not single-action prediction.
3. Test on the held-out game, offline, against the existing captured/reproduced trajectory for that
   game (not a live hidden game) — does the TRM's refined output do any better than a trivial
   baseline (random action sequence, or the current search baseline's own performance on that same
   game)?

**Falsifiable gate:** if held-out performance is no better than the trivial baseline, that's an honest
signal that the current trajectory corpus doesn't carry enough signal yet — the right next move is
bootstrapping more training data (self-play, augmentation) before retrying TRM-as-generator, not
abandoning the idea outright. If it clears the gate, that's the justification for the heavier
full-FT-on-a-3090 build the existing gap entry already scoped.

## Discipline compliance for any task drawn from this note

- `solve_provenance: development_proxy` — this is dev/registry-proxy work, not a live hidden-game
  solve, per "ARC Live-Path Reachability Discipline."
- If a pilot graduates past this stage, it must be wired into the live agent path (a `GameAdapter` /
  fed into `E3AgentPolicy`), never a standalone offline script the live agent can't call.
- `verifier_is_oracle: false` if this is ever framed as a moat/verifier-value claim — it isn't; this is
  a generator-mechanism proposal, not a verifier claim, per the Circularity/Oracle-Distinctness
  Discipline's own scope (that discipline governs verifier claims specifically).
- Per "ARC-AGI-3 Incremental-Progress Scoping," any resulting task targets a scoped test, never "solve
  everything."

## Cross-references

- `docs/research-notes/trm-leave-one-game-out-pilot-results-2026-07-05.md` — the falsifiable first
  pilot proposed here, run: mixed/inconclusive result, a redesign is needed before a firmer verdict
- `docs/research-notes/arc1-arc2-capability-transfer-to-arc3-2026-07-04.md` — the broader question of
  whether ARC-1/2 capability (not just TRM specifically) applies to ARC-3; also notes the pre-trained
  ARC-1 TRM checkpoint's availability
- `docs/research-notes/trm-generator-hidden-game-plan-2026-07-04.md` — the staged plan for turning a
  successfully-trained public-game TRM into something that can help on hidden games (weights don't
  transfer, but a recipe + online adaptation might)
- `docs/research-notes/human-replay-corpus-staging-bug-and-opportunity-2026-07-04.md` — the much
  larger, better-validated training-data source (144 winning human trajectories, all 25 public games)
  found in response to the operator's direct question, and the staging bug currently blocking it
- `ops/verifier_gaps.md` `GAP-ARC-TRM-TRAINED-ON-ARC` — the existing open gap this note adds evidence
  to (update this entry with a pointer to this note rather than tracking the idea in two places)
- `results/experiment_sudoku_energy_vs_ar_v1.json` — the cross-domain precedent
- `results/trm_verifier_rerank_opportunity.json` (commit `5f56ccc37`) — the properly-bounded
  reproduction of the official checkpoint's real static-ARC-1 performance (pass@2 ~0.52, pass@1000
  ~0.62 on a 29-task subset); `results/trm_arc_baseline_arc_v1.json` / `arc_v2.json` — the same
  checkpoint evaluated a different (likely unbounded/incomplete-voting) way, giving degenerate
  near-zero numbers -- a methodology artifact, not a weak-checkpoint finding (correction 2026-07-04)
- `docs/research-notes/arc-agi3-levers-tried-x-verdict-2026-06-25.md` — exp4100's retired TRM-as-
  dynamics-engine thread (scope-distinct from this proposal)
- `results/trm_runs/DO_NOT_RELAUNCH` — the retirement sentinel; scoped narrowly to the Sudoku-Extreme
  verifier-graft auto-relaunch mechanism specifically, does not block this proposal
- `docs/research-notes/arc-multilevel-deepening-literature-2026-07-03.md` — the sibling literature
  note from the previous day; GAP-4891's enumeration-wall diagnosis is the shared motivating problem
- `reference_trm_tta_mcgovern.md` (memory) — the full-FT > LoRA finding already informing the existing
  gap entry's candidate design
