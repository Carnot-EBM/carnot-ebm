# Research Roadmap — Milestone 2026.06.342

**Status:** Pre-staged by outer-loop Claude (Opus 4.8), 2026-06-02.
**Predecessor:** 2026.06.341 (Phase-3 Thesis-A EBT bring-up — the FIRST human-seeded paradigm
alternative since energy-SELECTION was bounded; operator seeded energy-as-GENERATOR, EBT,
arXiv:2507.02092 / github.com/alexiglad/EBT).
**Milestone doc for:** `research-roadmap-next.yaml` (`milestone: 2026.06.342`)

---

## 1. What the previous milestone (.341) ACTUALLY proved (read the artifacts, not the verdict)

`.341` was the cheap half of the Thesis-A kill-gate: *can a tiny EBT even train stably on the 3090
rig?* The recorded milestone verdict was **kill-gate part-(a) FAIL — "energy-as-generator bounded
at small scale, STOP."** **That verdict is wrong.** Read via `scripts/summarize_artifact.py` +
direct artifact inspection:

| Step | Artifact | What it ACTUALLY shows |
|---|---|---|
| EBT vendored + audited | exp3725 | `importable=true`, Apache-2.0, commit pinned `19420cbe…`, energy path audited (energy descent via Langevin/MCMC, contrastive + CE training objective), CPU smoke energy `0.554` finite. **Clean PASS.** |
| Tiny-EBT single-step smoke | exp3726 | **Clean PASS.** 38M-param EBT, fits one 3090 at **1283 MB** peak VRAM, GSM8K n=2048, loss finite AND **monotonically decreasing** across 10 steps (−0.077 → −37.74), 64 train steps wired correctly. This is the POSITIVE CONTROL and it passed. |
| Matched-compute eval harness | exp3727 | Built + unit-tested (FLOP accounting + budget-matcher). The instrument for the .342 decisive comparison is ready. |
| Bounded checkpointed training | exp3728 | **`blocked_ebt`, 0 steps, 65.5 s.** `preconditions_checked={ebt_vendored:false, smoke_passed:false}` — **both demonstrably false**: exp3725/3726 prove the module IS importable and the smoke DID pass. The task's precondition logic (`import carnot.phase3.ebt_upstream` + `os.path.exists("results/experiment_3726_*.json")`, a RELATIVE path) evaluated False because the run executed in a bad cwd / sys.path / venv. **It never trained a single step.** |
| Kill-gate verdict | exp3729 | Read exp3728's `blocked_ebt`/steps=0 and concluded "energy-as-generator is bounded at small scale." `green_light_342=false`. |

**Diagnosis: the .341 kill-gate "FAIL" is an INFRASTRUCTURE FALSE-NEGATIVE, not a scientific
finding.** It is exactly the failure mode CLAUDE.md's **FALSE_NEGATIVE_RISK** discipline names: a
null/negative claim ("the route is bounded") asserted **without a positive control passing the
bounded run**. The positive control (exp3726 single-step smoke) PASSED — the EBT trains, fits the
GPU, and the loss decreases. exp3728 bailed before training on a cwd/import-path bug; exp3729
mislabeled that infra failure as a mechanism result. Per the **Reading-Results Discipline**, a
verdict synthesized over a `blocked_*` upstream with `steps=0` cannot bound a mechanism.

**Strategic position:** Thesis A is NOT dead. The kill-gate part-(a) question — *does a tiny EBT
train to stable convergence in a bounded budget?* — is **still open**, because no genuine bounded
training run has happened yet. `.342` must (1) correct the record honestly, (2) FIX the harness bug
and run the genuine bounded training, (3) render the REAL part-(a) verdict, and if stable,
(4) run the matched-COMPUTE comparison — kill-gate part (b), the actual thesis test.

This is NOT breadth churn (north-star §1): it is the direct, gated continuation of the
operator-seeded Thesis-A bring-up, recovering from a same-day infrastructure false-negative.

---

## 2. The three biggest gaps (PRD vision vs current state)

1. **The Phase-3 generator thesis has no genuine empirical signal yet.** The PRD endgame (Phase 3)
   is an open-source EBM/EBT foundation model with non-autoregressive reasoning. Thesis A is the
   first concrete test of that mechanism. `.341` produced a positive single-step control and then a
   spurious negative; `.342` must produce the **genuine** bounded-training stability signal and, if
   stable, the matched-COMPUTE accuracy comparison. Until a real bounded run exists, the route is
   neither alive nor dead — it is **untested**, and an untested route blocks the Phase-3 decision.

2. **The record carries an unsupported negative claim.** `.341`'s capstone + kill-gate state
   "energy-as-generator bounded at small scale." Left uncorrected, a future planner reads that as
   a settled negative (like P0.1) and never revisits the route — enclosing a venture bet on a
   harness bug. `.342` issues a CLEAN corrigendum (exp1850 pattern): preserve exp3729's numbers,
   annotate the false-negative root cause, and re-open part-(a) as untested.

3. **Continuous self-learning has not yet touched the training loop itself.** FR-11 self-learning
   (research-program.md Tiers 1–4) has been exercised on verifier-template consolidation, never on
   the EBT bring-up. The most likely failure mode of EBT training is divergence; which stabilizer
   recipe prevents it is exactly the kind of online, counter-update (Tier-1) knowledge the
   self-learning system should accumulate. `.342` wires a Tier-1 online tracker of per-stabilizer
   divergence-prevention efficacy across training chunks — a self-learning experiment that is also
   directly load-bearing for the kill-gate.

---

## 3. Architecture (Thesis-A bring-up, recovered)

```
                       ┌──────────────────────────────────────────────┐
   .341 (DONE, but    │  exp3725 vendor+audit  ✔   exp3726 smoke  ✔    │
   verdict false-neg) │  exp3727 matched-compute harness (tested) ✔   │
                       │  exp3728 BLOCKED (cwd/import bug) ✘ steps=0   │
                       │  exp3729 verdict = FALSE-NEGATIVE  ✘          │
                       └───────────────────────┬──────────────────────┘
                                                │
  .342 PHASE 0  archive/activate (exp3732) + CLEAN corrigendum of exp3729 (exp3733; record honest)
                                                │
  .342 PHASE 1  ┌─────────────────────────────────────────────────────────────────┐
  (the genuine  │ exp3734  FIX exp3728 (robust importlib check + abs paths, run    │
   part-a)      │          from {project_root}); ship driver w/ EBT stability      │
                │          recipe (random-alpha + replay buffer + Langevin noise +  │
                │          grad-clip + KL-term CD); FIRST bounded train chunk        │
                │          EBT + matched-AR  ──checkpoint──▶                         │
                │ exp3735  RESUME chunk 2 (gated: chunk-1 trained >0 steps)          │
                └───────────────────────────┬─────────────────────────────────────┘
                                             │
  .342 PHASE 2  exp3736  REAL kill-gate part-(a) verdict over the GENUINE run
                          (supersedes false-negative exp3729) → green_light_342 (bare bool)
                                             │  (gate: stable)
                ┌────────────────────────────┴────────────────────────────┐
                │ exp3737 EBT energy-descent GENERATION smoke on held-out    │
                │ exp3738 matched-COMPUTE comparison: EBT energy-descent vs   │
                │         AR best-of-N at EQUAL FLOPs (exp3727 harness)       │
                │ exp3739 kill-gate part-(b) verdict: EBT beat AR @ equal     │
                │         compute? does gap narrow w/ 2x train? (honest)      │
                └────────────────────────────────────────────────────────────┘
                                             │
  .342 PHASE 3  exp3740 FR-11 v15 Tier-1 online stabilizer-efficacy tracker (self-learning)
                exp3741 KV260 opportunistic continuity (terminal confirm)
                exp3742 capstone .342 (honest: record corrected; real part-a; part-b verdict)

  INVARIANTS (every task): paper_ready stays TRUE (G1-G4 closed 2026-05-31); frozen FoVer 0.9131
  stays frozen; P0.1 / energy-SELECTION stays settled-bounded (this tests GENERATION, a different
  mechanism); never edit ops/north-star.md; never trigger CI; never push.
```

## 4. Phase descriptions

- **Phase 0 — Transition + record correction.** Archive `.341` honestly (the kill-gate was an
  infra false-negative; part-(a) re-opened as untested). Issue a CLEAN corrigendum of exp3729
  (preserve numbers, annotate root cause). Cheap, codex, aggregation-only.
- **Phase 1 — The genuine kill-gate part (a).** Fix exp3728's cwd/import precondition bug (robust
  `importlib.util.find_spec` check, absolute artifact paths, run from `{project_root}`), ship the
  training driver with the EBT-paper stability recipe (random-alpha + replay buffer + Langevin
  noise + gradient clamp + the KL-term CD fix from arXiv:2012.01316), and run two bounded
  checkpointed chunks of the tiny EBT + a matched tiny AR baseline on the SAME corpus/budget.
  Heartbeat every K steps; ≤1500 s/run then checkpoint+exit; resume. **requires_claude + opus** —
  divergence diagnosis is open-ended judgment under ambiguity (operator directive 2026-06-02).
- **Phase 2 — Real verdict + the actual thesis test.** Render the genuine part-(a) verdict over the
  real run (supersedes exp3729). If stable, smoke EBT generation, then run the matched-COMPUTE
  comparison (EBT energy-descent vs AR best-of-N at EQUAL inference FLOPs via the exp3727 harness),
  and render the honest part-(b) verdict. The matched-COMPUTE discipline is load-bearing: a
  matched-PARAMS "win" is just extra inference FLOPs (the exact P0.1 trap).
- **Phase 3 — Self-learning + hardware + capstone.** FR-11 v15 Tier-1 online stabilizer-efficacy
  tracker (CPU counter updates; self-learning mandate). KV260 opportunistic terminal-confirm.
  Capstone: record the corrected history, the real part-(a) outcome, and the part-(b) verdict —
  paper_ready stays TRUE, frozen 0.9131 unchanged.

## 5. Dependency graph (bare-value gates only; no None-read cascades)

```
exp3732 (archive/activate) ─▶ exp3733 (corrigendum exp3729)
exp3734 (fix+driver+chunk1, gpu/opus)
   └─ cumulative_steps_trained>0 (BARE int) ─▶ exp3735 (resume chunk2, gpu/opus)
exp3734,exp3735 ─▶ exp3736 (REAL part-a verdict) ─▶ green_light_342 (BARE bool)
   └─ green_light_342==true ─▶ exp3737 (generation smoke, gpu)
        └─ ebt_can_generate==true (BARE bool) ─▶ exp3738 (matched-compute, gpu)
              └─▶ exp3739 (part-b verdict)
exp3740 (FR-11 v15 self-learning) · exp3741 (KV260) · exp3742 (capstone) — ungated, graceful
disk-presence fallback (read upstream if present, else record honest "not-run")
```

`gated_on` is used ONLY on the expensive GPU tasks (exp3735/3737/3738) to genuinely skip GPU+Sonnet
time when the prerequisite is unmet, and ONLY against **bare-value** fields (per
`feedback_gated_fields_must_be_bare`: a principle-annotated `{value,principle}` dict breaks the
comparator). Verdict/aggregation/self-learning/hardware/capstone tasks carry NO `gated_on` and read
upstream artifacts with a graceful disk-presence fallback (the .340 proven-safe pattern).

## 6. Hardware requirements

- **2x RTX 3090 (CUDA):** Phase-1 training (exp3734/3735) + Phase-2 generation/eval (exp3737/3738).
  The tiny 38M EBT fits one 3090 at ~1.3 GB (exp3726); the matched AR baseline is the same band.
- **KV260 (SSH only):** opportunistic terminal-confirm (exp3741). SSH-reachability precondition
  only — NEVER a host `/dev/mmcblk*` check (KV260 SSH-Not-SD-Card discipline).
- gemini still crashes real GPU workloads (exp3703) → codex is the non-claude default; the heavy
  open-ended training-debug tasks are requires_claude+opus.

## 7. Routing & discipline summary

- **requires_claude + opus:** exp3734 (fix+driver+chunk1), exp3735 (resume chunk2) — EBT training is
  finicky; divergence diagnosis + multi-file driver work meets the requires_claude bar (operator
  directive 2026-06-02).
- **codex + requires_codex + gpu:** exp3737 (generation smoke), exp3738 (matched-compute) — running
  the trained models through the already-tested exp3727 harness is mechanical with a deterministic
  FLOP-accounting criterion; gemini crashes GPU so codex is the cheap-default.
- **codex + requires_codex (no gpu):** all archive/corrigendum/verdict/self-learning/hardware/
  capstone tasks (aggregation or CPU).
- **PRECONDITIONS blocks** on every GPU task (CUDA + robust EBT-importable + corpus). The exp3728
  fix is the headline lesson: robust `importlib.util.find_spec` check + absolute paths + run from
  `{project_root}`; a missing resource → `blocked_<resource>`, never a fabricated pass.
- **Failed-Experiment Rerun Discipline:** exp3734/3735/3736 scope-match exp3728/exp3729 → each
  carries a `prior_failures:` block (root cause = infra false-negative, what's different = fixed
  harness + real training, `retire_if_same_verdict: true`).
- **operator_override:** routine archive/capstone/KV260/FR-11-lineage tasks carry the standing
  2026-05-29 override string for false-positive scope-matches.
- **inference-substrate hygiene:** aggregation tasks set `aggregation_from_upstream_artifacts` and
  carry NO GGUF/CUDA marker; GPU training/eval sets `live_llm_inference` with full methodology.
- **Anti-poison-test clause** on tasks that ship code+tests (exp3734, exp3740): the test must assert
  against the script's real behavior, never poison the conductor pre-test gate.

## 8. Honest framing (what a green-light does and does not mean)

A part-(a) PASS means "the tiny EBT trains stably enough to run the matched-compute comparison" —
NOT "energy-as-generator works." A part-(b) result is the actual thesis signal, and an honest
NEGATIVE there (EBT does not beat matched-AR at equal compute, gap does not narrow with 2x train) is
as valuable as a positive — it bounds the route cheaply, exactly as the kill-gate was designed to.
This remains a venture bet with a knife to its own throat; banking the verifier product is
unaffected (paper_ready stays TRUE; frozen 0.9131 stays frozen).
