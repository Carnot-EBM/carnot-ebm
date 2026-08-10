# Pinductor-style REx refinement for ARC world-model induction — plan (2026-08-09)

**Status:** PLAN + PREPARATION. Operator directive 2026-08-09: "Let's plan out and prepare
Pinductor to be run." That directive is the explicit operator decision the standing hold in
`ops/known-issues.md` required ("Do NOT propose a follow-up on the refinement /
single-shot-GGUF-induction axis ... it is banked, not queued" — banked for "whoever later has a
healthy host and an operator decision to spend a slot on it"). The hold is now lifted for this
one prepared experiment. It is not lifted in general.

**Reads as required input:** CLAUDE.md "Failed-Experiment Rerun Discipline", "ARC Live-Path
Reachability Discipline", "Adversarial Artifact Verification", `feedback_audit_untrusted_code`
(memory), `docs/research-notes/arc-lever-triangulation-2026-07-23.md`.

## 1. What Pinductor is, in our terms

Paper: "Learning POMDP World Models from Observations with Language-Model Priors"
(arXiv:2605.13740). Code: `github.com/atomresearch/pinductor` (MIT), cloned read-only to
`~/arc-sota-refs/pinductor` for reference. **Inspiration tier only** per
`feedback_audit_untrusted_code`: we reimplement the mechanisms; no code is copied and none of
theirs is executed.

Pinductor induces an executable world model by having an LLM propose candidate programs, then
running a refinement loop ("REx", their Algorithm 1) with four mechanisms:

| Pinductor mechanism | What it does | Our current equivalent |
|---|---|---|
| M candidates + UCB1 parent selection over a refinement TREE | Each round, pick WHICH prior candidate to refine next by UCB1 (quality + exploration bonus); keeps a population, not one lineage | NONE — `execute_bounded_llm_reinduction` refines only the latest candidate, linearly |
| Continuous kernel pseudo-likelihood scoring (their Eq. 7/8) | Rank candidates on a SOFT score so refinement has a gradient below the pass/fail bar | EXISTS — `VerifyResult.change_fidelity` is exactly this (symmetric union fidelity, continuous). But our refinement loop does not USE it to steer; the production gate is binary |
| QBC committee vote entropy (their Eq. 9) | Where candidate models DISAGREE is where evidence is most informative; surface those transitions to the LLM | NONE — our refactor prompt shows the FIRST 5 mismatches in corpus order |
| Near-best softmax final selection (their Eq. 11/12) | Avoid greedily overfitting a noisy score | Not adopted (see §3, deviation D2) |

Their particle filter / belief machinery exists because MiniGrid is partially observed. ARC-AGI-3
frames are fully observed, so that entire half of the paper is dropped. What remains is a
search-structure claim: **population + UCB1 tree + disagreement-selected counterexamples beats a
single greedy refinement lineage at the same LLM-call budget.** That claim is what we test.

## 2. Prior failures (Failed-Experiment Rerun Discipline block)

This axis has three prior results. Each is named with its root cause and what is different now.

1. **exp5766 — CEGIS refinement null.** Root cause: INSTRUMENT DEFECT, not a refuted idea. The
   refactor prompt never contained the engine source (0 of 454 substantive lines reached the
   prompt on 13 of 13 games), so "refactor while keeping what works" was unachievable by
   construction. The fix (`CARNOT_ARC_REFACTOR_SHOW_ENGINE`) is banked, tested, default-OFF, and
   has never yet been part of a completed measurement.
   *Different now:* this experiment runs with the fixed instrument ON, in both arms.
2. **exp6091 — engine-visible rerun.** Root cause: INFRASTRUCTURE (the llama-server reaper killed
   every server; 0 valid cells). The line was closed on the independent measured negative: 0 of
   296 clean engine-units reach held-out `change_accuracy >= 0.5`.
   *Different now:* (a) that negative is about SINGLE-SHOT induction quality against a BINARY
   exact-match bar; this experiment measures whether structured refinement CLIMBS the continuous
   `change_fidelity` score, which is a different question the 0/296 result does not answer;
   (b) the reaper risk is mitigated by per-cell checkpointing, per-cell retry, and the option to
   run under conductor isolation (two correlational instances this session show clean runs with
   the conductor stopped — see `ops/known-issues.md` 2026-08-09 entries).
3. **exp5764 / exp5722 — 31B single-shot held-out induction 0.378; 0 live levels.** Root cause:
   single-shot induction quality is genuinely low.
   *Different now:* nothing about single-shot changes — this experiment is precisely about
   whether SEARCH STRUCTURE on top of the same generator moves the same held-out metric. If it
   does not, that is a real negative about the refinement-structure hypothesis, not a rerun of
   the single-shot measurement.

**Retirement condition (pre-registered):** if the REx arm fails its gate (§4), the
Pinductor-style refinement variant joins the retired refinement axis. No further refinement
variant is proposed without a NEW mechanism class and a fresh operator decision
(`retire_if_same_verdict: true` semantics).

## 3. Design

**One code path, two configurations** (budget parity by construction), in a new module
`python/carnot/agentic/arc_rex_refinement.py` (REQ-ARC-WMTE-6248):

- **Arm `linear` (baseline):** 1 induce + (B-1) refinements, each refining the LATEST candidate,
  mismatches shown in natural corpus order. This is the current production shape
  (`execute_bounded_llm_reinduction`'s structure) expressed in the same harness.
- **Arm `rex` (treatment):** 1 induce + (B-1) refinements where each round (a) picks the parent
  candidate by UCB1 over ALL candidates so far (quality = VALID-slice `change_fidelity`,
  exploration bonus with the `(visits+1)` denominator — adopting the reference implementation's
  own fix note, since a fresh-node `+inf` shortcut degenerates the search), (b) reorders the
  parent's mismatch list by QBC committee vote entropy (Shannon entropy over the distinct
  next-grid predictions of ALL current candidates on that transition) so the 5 mismatches the
  prompt can carry are the 5 most-disagreed-on, and (c) writes the parent's source into the
  isolated engine store before calling `refactor`, so the engine-visible prompt shows the actual
  parent being refined.
- **Final selection, both arms:** argmax VALID `change_fidelity` (ties: earliest candidate).

**Data split per game:** collect N=60 transitions (`e3.collect_transitions`, no LLM) →
TRAIN 40 (available to the induce/refactor prompts) / VALID 10 (selection scoring + QBC; never
shown to the LLM) / HELD 10 (final A/B metric only; never used during the run). The three-way
split exists so selection cannot overfit the reported metric.

**Scoring:** `WorldModelVerifier(..., hud_mask=<game's own mask>, hud_mask_enabled=True)`, the
same forced-HUD-mask convention as REQ-ARC-WMTE-6233/6246.

**Roster (6 games):** ft09, tr87, cn04, ar25 (from exp6246's roster, minus m0r0 which timed out
there) + ka59, re86 (the Mode-A hard games; collection verified working in exp6244/6247).

**Budget:** B=4 LLM calls per arm per game (1 induce + 3 refinements) = 48 calls total.
Estimated wall: roughly 3-7 hours at observed 31B decode rates; per-cell checkpointing makes
interruption cheap.

**Generator:** the frozen live generator (gemma-4-31B-it-qat, GPU 1, q8_0 KV, port 8940,
`no_think_prefix=""`, timeout 1500 s) — identical configuration to exp6246/6247 so results are
comparable.

**Isolation:** `CARNOT_ARC_E3_DIR` MUST point at a private scratch directory (the script refuses
to start otherwise — the exp6246 guard, reinforced by exp6247's own engine-store-clobber
incident). `CARNOT_ARC_REFACTOR_SHOW_ENGINE=1` forced in both arms.

**Declared deviations from the paper (so nobody later mistakes this for a faithful replication):**
- D1: no particle filter / belief machinery (ARC is fully observed).
- D2: final selection is deterministic argmax, not near-best softmax — an n=6 paired A/B cannot
  afford selection stochasticity; noted as untested.
- D3: their per-round candidate count M collapses to 1 proposal per round (the tree provides the
  population across rounds); a wider M multiplies LLM cost 5x and is a follow-up knob, not part
  of this gate.
- D4: quality signal is our `change_fidelity`, not their distance-kernel likelihood — same role
  (continuous, soft), different formula, already validated on this corpus.

## 4. Pre-registered gate

Primary: **REx final beats linear final on HELD `change_fidelity` in >= 4 of 6 games AND the
pooled mean paired delta is > 0.** Fewer than 4, or a non-positive pooled delta, is a negative:
the refinement-structure hypothesis is retired per §2.

Secondary (reported, not gated): does ANY candidate in either arm reach HELD
`change_fidelity >= 0.5` (the live trust-gate threshold)? This is the number that would matter
for the scored path; the primary gate can pass without it (structure helps but not enough), and
that distinction is reported honestly.

A gate PASS does not wire anything into the live path by itself. It authorizes the follow-up
decision (wire REx into `execute_bounded_llm_reinduction` behind a default-OFF flag, then a
live-path A/B per the Live-Path Reachability Discipline). A pass on n=6 games is
promising-but-preliminary, same standard as every other lever this month.

## 5. Deliverables prepared in this session

1. `python/carnot/agentic/arc_rex_refinement.py` — the REx loop (UCB1, QBC entropy, candidate
   tree, one code path for both arms). Pure logic separated from LLM calls for testability.
2. `tests/python/test_arc_rex_refinement.py` — CPU-only unit tests (UCB1 math, entropy edges,
   QBC reordering, budget accounting, arm behavioral difference, final selection) with a fake
   proposer; no GPU, no network.
3. `scripts/experiments/experiment_6248_pinductor_rex_ab.py` — the A/B driver (checkpointed,
   isolated store, liveness-witnessed), ready to launch.
4. REQ-ARC-WMTE-6248 in `openspec/capabilities/arc-world-model-trust-energy/spec.md`.
5. This note.

**Launch command (staged, not yet run):**

```bash
mkdir -p /home/ianblenke/github.com/ianblenke/carnot/results/arc_e3_exp6248_scratch
cd /home/ianblenke/github.com/ianblenke/carnot
setsid nohup timeout 28800 env \
  CARNOT_ARC_E3_DIR=$PWD/results/arc_e3_exp6248_scratch \
  CARNOT_ARC_GENERATOR_CUDA_GPU=1 \
  CARNOT_ARC_REFACTOR_SHOW_ENGINE=1 \
  .venv/bin/python scripts/experiments/experiment_6248_pinductor_rex_ab.py \
  > /tmp/exp6248_run.log 2>&1 < /dev/null &
```

Recommended launch condition: while heavy conductor GPU work is idle, or with the conductor
deliberately stopped (operator call), given the two correlational reaper instances recorded
2026-08-09. The script checkpoints per cell either way.

## 6. Cross-references

- arXiv:2605.13740 (Pinductor) · `~/arc-sota-refs/pinductor` (reference clone, MIT, inspiration
  tier)
- `ops/known-issues.md`: the standing refinement-axis hold (now lifted for this one experiment by
  the 2026-08-09 operator directive), exp6091/exp5766 history, the 2026-08-09 reaper entries
- `docs/research-notes/arc-lever-triangulation-2026-07-23.md` — the binding-constraint diagnosis
  this experiment attacks (induction-grade sequence routing)
- `research-studying.md` 2026-08-09 SOTA scan — the ingestion that surfaced Pinductor
- REQ-ARC-WMTE-6246 (`experiment_6246_induce_prompt_enrichment_heldout_ab.py`) — the harness
  pattern this reuses (collect/split/induce/score, isolated store, forced HUD mask)
- arXiv:2606.11521 — the counterexample-guided-refinement idea already partially adopted
  (2026-07-06 fix); Pinductor's QBC selection is the principled upgrade to "which
  counterexamples"
