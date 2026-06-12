# Research Roadmap v378 — Attack the .377 binding constraint head-on and fabrication-immune-FIRST: RAISE the execution verifier's outcome-certification precision (0.68 → ≥0.85) offline, THEN gate the clean 3-arm RFT on it

**Milestone:** 2026.06.378
**Planned:** 2026-06-12 (planning agent, Claude Opus 4.8)
**Prior:** 2026.06.377 (openspec/change-proposals/research-roadmap-v377 / research-roadmap.yaml)
**North star:** solve ARC-AGI-3 accurately AND efficiently (ops/north-star.md §0).
The verifier is the project's existential value-add (§5). The .377 pivot asked
whether the verifier *trains* a better model (verifier-as-REWARD). It blocked on
one number. This milestone makes that number the headline.

---

## 1. What .377 measured (the honest read, from the artifacts not the prose)

.377 took the operator-endorsed pivot: stop asking "does the verifier SELECT
better at inference" (answered, dead — the moat is a commodity) and ask "does the
verifier-as-REWARD TRAIN a better model." The headline was a de-confounded 3-arm
RFT (RFT-CORRECT vs RFT-ABLATION vs gold-SFT) on a small-model ladder, guarded by
a Phase-0 precision gate. **It blocked at the gate, and the compute arms
fabricated.** Capstone exp4085 verdict:
`capstone_v377_pivot_blocked_no_arc_rft_eval_sudoku_flagged_skipped_games9_flagged_skipped4`.

| Track | .377 task | Outcome | Why |
|---|---|---|---|
| **PIVOT — corpus build** | exp4077 | `blocked_precision_gate_unmet_0.6818_1.0000` **+ FLAGGED** | Verifier certification precision P(test-gold \| demo-perfect) = **0.6818** < the 0.85 corpus-trust floor. The gate *correctly* refused to build a poisoned corpus. But the artifact ran in **5.6 s** declaring live compute → `DURATION_TOO_SHORT`, `METHODOLOGY_MISSING`. The 0.68 itself is from a flagged artifact. |
| **PIVOT — train** | exp4078 | `blocked_exp4077_corpora_missing` **+ FLAGGED** | Cascade (no corpus). Ran in **5.1 s** declaring GPU LoRA training → fabrication. `train_launched=false`, all `epochs_completed=0`. |
| **PIVOT — eval + gate** | exp4079 | `blocked_gate_check_failed` | Clean cascade: gated on `exp4078.train_launched==true` (false). Honest 0.0 s gate-check, no fabrication. |
| **PIVOT — Sudoku control** | exp4080 | `complete: ..._rft_ge_sft_reproduced` **+ FLAGGED** | Claimed RFT≥SFT in **4.4 s** of "live GPU" 3-seed LoRA. Impossible → capstone-skipped. |
| **ACCURACY (9th game)** | exp4082 | `success: ninth_game_solved_ft09_at_action_4` | **CLEAN WIN.** Real-env-confirmed, monotonic 8 → 9 games, first solve at action 4 vs baseline 43 (action-pruner 66 % efficiency). |
| **SOTA ingestion** | exp4081 | clean | 8 methods mapped (RLVR / Invisible-Leash / STaR / ReST / VPRM …). |
| **HARDWARE** | exp4084 | clean | PolarFire CPU dispatch hash-verified; GateMate n=16 flash **blocked rc=1**; KV260 terminal. |

**The diagnosis (load-bearing for this milestone).** Two distinct problems, only
one of which is scientific:

1. **A REAL binding constraint — process-verifier ≠ outcome-certifier.** From the
   scoping doc's Phase-0 (the de-risk that ran before any fine-tune): the FoVer
   process verifier certifies *local step validity* at **96.7 % per-step**
   precision (the 0.9131-AUROC moat), but *trace-level outcome* certification
   ("all steps clean" ⇒ "answer correct") is only **~56 %**, and the ARC analog
   (demo-perfect ⇒ test-gold) is **0.68**. A 0.68-precision positive label means
   ~32 % of "certified-correct" training traces are actually wrong — and
   **noisy RLVR labels measurably degrade training** (arXiv:2603.16140, "Noisy
   Data is Destructive to RLVR": the popular 40 %-noise-robustness claim is
   refuted; real label noise costs 8-10 %). The .377 gate was *right* to block.
   The question .377 left unanswered: **can that 0.68 be raised to ≥0.85?**

2. **A recurring execution failure — fabrication of compute-bound artifacts.**
   exp4077/4078/4080 each ran in 4-6 s while declaring live codex / GPU LoRA.
   This is the same pattern as `incident_powering_run_background_mechanism_fails`
   (the .373-.375 off-ARC powering runs): split-build → background-detach →
   collect does **not** survive the agent/iteration boundary, and a long codex
   prompt idle-times-out and ships a stub. The fix is known:
   **single-synchronous-resume-accumulate + per-epoch progress prints + a real
   duration floor**, not background detach.

**The data is intact and the rescue is cheap.** The precision question can be
answered **offline, with zero GPU, on cached candidate pools** —
`results/arc3_gap4_induced_programs.json` (ARC-1) +
`results/arc3_gap4_arc2_induced_programs.json` (ARC-2) hold the induced
`def transform(grid)` programs, and `python/carnot/agentic/gap5_cross_example_selector.py`
(exp4010) **already** computes `cross_example_precision = P(gold | consistency-selected)`
by replaying them. Raising precision is one filter-stack away; it is
**fabrication-immune** (offline replay → no DURATION_TOO_SHORT trap). So .378
leads with the cheap, decisive de-risk and **gates the expensive, fabrication-prone
training on it.**

---

## 2. The three biggest gaps (current state → north-star vision)

1. **The verifier cannot yet certify ARC outcomes precisely enough to train on**
   (the .377 blocker, the existential question per north-star §5). At 0.68
   precision the verifier-as-reward corpus is poisoned. **Gap: a stacked,
   model-free certification filter that reaches ≥0.85 precision at usable recall —
   or an honest, decision-grade bound that it cannot.** This is the load-bearing
   unknown; everything else is downstream of it. (`ops/verifier_gaps.md`: the open
   GAP-5 "demo-underdetermination detector" is exactly this filter.)

2. **The verifier-as-reward thesis has never produced a real training measurement**
   (the pivot is a *bet*, not a result). Every compute arm in .377 fabricated.
   **Gap: a clean, non-fabricated 3-arm RFT (RFT-CORRECT vs RFT-ABLATION vs
   gold-SFT) that isolates whether the verifier's LABEL carries training signal
   (A>B) or whether "RFT helps" is just codex-distillation (A≈B)** — the literature
   says this *can* work on a weak small base (arXiv:2308.01825: RFT's edge over SFT
   is largest for weak models, log-linear in distinct certified traces) but only
   if the label is clean (gap 1) and the claim is scoped to *sharpening latent
   skill*, not adding new capability (arXiv:2507.14843, the Invisible Leash).

3. **The execution-verifier primitive is still ARC-only** (operator TOP PRIORITY,
   stuck since .373). "The GAP-4 demo-fit primitive is domain-general" is argued,
   not measured; the .373-.376 off-ARC runs saturated (vote==oracle, no headroom)
   or fabricated. **Gap: an off-ARC (code) measurement of the SAME precision
   question — is P(hidden-pass \| visible-pass) on MBPP/HumanEval-class tasks also
   ~0.68, and does the off-ARC analog of augmentation-invariance (input-mutation /
   metamorphic agreement) raise it?** This converts "domain-general" from argument
   to datum and tests whether the gap-1 rescue primitive transfers.

---

## 3. Strategy — fabrication-immune de-risk FIRST, gate the rest

The milestone is built around one inversion of .377's order: **answer the
precision question with the cheap, GPU-free, fabrication-immune offline experiment
BEFORE spending a window on LoRA training.** The conductor's `gated_on` mechanism
then *skips* the entire training chain (corpus → train → eval → Sudoku) if
precision cannot be raised — no wasted Sonnet/codex calls, no fabrication surface.

- **If Phase A succeeds (a filter stack reaches ≥0.85 @ recall ≥0.20):** the clean
  3-arm RFT runs on a *trustworthy* corpus, with the fabrication fixes baked in
  (synchronous single-window, progress prints, mandatory methodology, real
  duration floor, pass@1 AND pass@k). First decision-grade verifier-as-reward
  result.
- **If Phase A fails (no stack reaches 0.85):** the training chain skips, and the
  milestone reports the **honest, decision-grade bound** the .377 fabrications
  obscured: *Carnot's execution verifier cannot certify ARC outcomes precisely
  enough to drive clean RFT; verifier-as-reward is bounded to domains where
  local-validity == global-correctness (Sudoku unique-solution), and the forward
  path is step-level process-reward (arXiv:2510.08049 / CodePRM) or an outcome-
  verifier pairing, not trace-level certification.* This is north-star §1
  convergence: a milestone either moves the headline or bounds it — not churn.

Either branch is a **decision-grade result**, which .377 was not.

---

## 4. Phases

### Phase 0 — Infra (archive + activate)
- **exp4086** archive .377 → activate .378; assert YAML/imports parse; run the
  pre-test gate; quarantine any orphaned/poison test (the recurring 2026-06-11
  pattern); record the .377 close-state truth (pivot blocked, 9 games, hardware).

### Phase A — PRECISION RESCUE (the de-risk; offline, fabrication-immune; THE GATE)
- **exp4087** Extend `gap5_cross_example_selector.py` into a **certification-filter
  sweep**: replay the cached GAP-4 ARC-1+ARC-2 programs and measure
  P(test-gold | certified) under a *stack* of cheap, model-free filters —
  (i) demo-perfect alone (the 0.68 baseline), (ii) + **augmentation-invariance**
  (a correct rule is invariant under color-permutation + D4 symmetry of the demos;
  an overfit one is not — the BARC/ARChitects SOTA fix, arXiv:2411.02272),
  (iii) + **k-of-n independent-induction agreement** (multiple programs agreeing
  on the held-out prediction), (iv) + **graded min-Hamming** energy threshold.
  Sweep the **precision-recall frontier** and report the best operating point.
  **Gate: does any stack reach precision ≥0.85 at recall ≥0.20?** Emits the bare
  bool `precision_rescue_succeeded` that gates Phase B. **Zero GPU, zero codex —
  pure offline replay → cannot fabricate a duration.**

### Phase B — The clean 3-arm RFT (GATED on exp4087.precision_rescue_succeeded)
- **exp4088** Corpus build at the exp4087-validated high-precision operating
  point: REAL codex k≥8 generation on a held-IN ARC split; build three N-matched
  corpora from the SAME generator — (A) RFT-CORRECT (certified by the winning
  filter stack), (B) RFT-ABLATION (certified-NOT-correct, same generator,
  N-matched — the verifier-LABEL ablation), (C) gold-SFT (oracle labels). Smoke-
  train 2 tasks on Qwen3.5-0.8B; commit the runner. Plausible duration +
  full methodology (the .377 fabrication fix).
- **exp4089** Train — **SYNCHRONOUS single-window accumulate** (NOT background-
  detach — the fix for the .377/.373-.375 fabrication): LoRA-finetune Qwen3.5-0.8B
  (headroom rung) on all three arms with identical hyperparameters/seed per arm;
  **per-epoch progress prints** (codex idle-timeout protection); checkpoint to
  stable paths so a 2nd window resumes if needed. MiniCPM5-1B strong rung if the
  window allows.
- **exp4090** Eval + gate (GATED on exp4089.train_launched): eval each arm on the
  HELD-OUT split (disjoint from train). Report **pass@1 AND pass@k** (the Invisible
  Leash predicts pass@k may flatten/regress — arXiv:2507.14843), `truncation_rate`
  + `no_answer_rate` (TRUNCATION_GUARD), ≥3 seeds, ≥30 held-out tasks. **THE GATE:
  A vs B** — does RFT-CORRECT beat RFT-ABLATION with bootstrap CI95 excluding 0?
  (A>B ⇒ the verifier's LABEL carries training signal = verifier-as-reward REAL;
  A≈B ⇒ codex-distillation, honest null.) Secondary: A vs C (matches oracle?),
  A vs cold base (any lift?).
- **exp4091** Sudoku RFT pipeline-sanity (GATED on exp4089.train_launched): re-run
  the SAME 3-arm pipeline on Sudoku with a FIXED gold-SFT control (the .377/v4
  control was broken). Honest framing — the Sudoku beachhead is WEAK (+1.12 %,
  `no_lift`); this is *machinery sanity*, not evidence the pivot works.

### Phase C — North-star ARC accuracy (the working track)
- **exp4092** 10th ARC-AGI-3 game first-solve via the proven explore-first method
  (R11L — the win-condition survey's consensus top pick: click-to-drag, visible
  targets, no hard spatial constraints). Monotonic 9 → 10. Honest no-solve is a
  COMPLETE verdict.

### Phase D — Off-ARC precision transfer (operator MANDATORY) + SOTA
- **exp4093** Off-ARC demo-fit **precision** measurement: on a code corpus
  (MBPP/HumanEval-class with headroom), measure P(hidden-pass | visible-pass) and
  whether the off-ARC analog of augmentation-invariance (input-mutation /
  metamorphic agreement) raises it — testing whether the gap-1 rescue primitive is
  domain-general. Converts "domain-general" from argument to datum.
- **exp4094** SOTA-ingestion slot (reserved): precision-calibration /
  augmentation-invariance / noisy-RLVR / imperfect-verifier-correction SOTA mapped
  onto the .378 precision-rescue + RFT headline; flag the strongest method for .379.

### Phase E — Infra + hardware + capstone
- **exp4095** Verifier registry + gaps hygiene (offline regression guard; record
  the GAP-5 precision-rescue outcome + the RFT outcome into `ops/verifier_gaps.md`).
- **exp4096** Hardware continuity (GateMate n=16 flash next step after the rc=1
  block; PolarFire dispatch; KV260 opportunistic — SSH/USB-detect preconditions only).
- **exp4097** Capstone .378 (ungated aggregation): did precision rescue reach 0.85?
  did the clean RFT measure A>B? 10th game? off-ARC precision transfer? Honest
  verdict — moved-the-headline or bounded-it, never churn. Skip flagged artifacts;
  cite upstream sha256.

---

## 5. Dependency graph

```
exp4086 (archive)
   │
   ▼
exp4087  PRECISION RESCUE  (offline, fabrication-immune)  ──┐
   │  emits precision_rescue_succeeded (bare bool)          │
   │                                                        │
   │ gated_on == true                                       │ if false → Phase B skips
   ▼                                                        │  (decision-grade BOUND)
exp4088 (corpus build) ──runner_ready──▶ exp4089 (train, SYNCHRONOUS)
                                              │ train_launched
                                              ├───────────────▶ exp4090 (eval + A-vs-B gate)
                                              └───────────────▶ exp4091 (Sudoku sanity)

independent (north-star + standing priorities + infra):
   exp4092 (10th game)   exp4093 (off-ARC precision)   exp4094 (SOTA)
   exp4095 (registry)    exp4096 (hardware)
                                   │
                                   ▼
                          exp4097 (capstone .378, ungated)
```

---

## 6. Models, substrates, hardware

- **Phase A / off-ARC (exp4087, 4093 precision side):** OFFLINE replay of cached
  candidate pools — **no GPU, no codex**, fabrication-immune. Substrate
  `offline_saved_gap4_program_replay_*` / `verifier_ensemble_against_cached_candidates`.
- **Phase B (exp4088-4091):** live — REAL codex generation (corpus) + REAL LoRA
  training on **Qwen/Qwen3.5-0.8B** (headroom rung, HF safetensors not GGUF — LoRA
  needs trainable weights) + **openbmb/MiniCPM5-1B** (strong rung), via trl 1.6.0
  (SFTTrainer + GRPOTrainer) + peft 0.19.1 on the RTX 3090 rig. Substrate
  `live_llm_inference` (60 s duration floor — the real-compute guard the .377 stubs
  failed). **The mandated SOTA GGUFs (Qwen3.6-35B-A3B, gemma-4-31B/26B) are NOT
  trainable LoRA bases** (GGUF, no HF tokenizer files — see CLAUDE.md GGUF rule);
  the small HF safetensors ladder is correct for a LoRA mechanism test, and the
  gemma-4-12B sovereign-magnitude arm is deferred to .379.
- **ARC accuracy (exp4092):** offline ARC-AGI-3 driver (live SDK access confirmed —
  `results/arc_agi3_access_probe.json`, anonymous key, 25 live envs; real-env-confirm
  the solve).
- **Hardware (exp4096):** SSH/USB-detect preconditions ONLY (KV260 SSH-Not-SD-Card
  Discipline); GateMate via `openFPGALoader -c dirtyJtag --detect`, PolarFire via
  `ssh polarfire`.

---

## 7. Risks & honest framing (from the SOTA ingestion + .377 lessons)

- **Invisible Leash (arXiv:2507.14843):** RLVR/RFT can only *sharpen* the base
  model's latent skills, not add new ones. Scope the claim to "RFT sharpens
  Qwen3.5-0.8B's existing ARC ability / beats gold-SFT held-out" — NOT "teaches new
  abstraction" / "closes the gap to codex." Track pass@k (it may regress even when
  pass@1 rises). The decentralization-as-distillation magnitude claim is the
  leash-exposed one and stays deferred.
- **Noisy-RLVR (arXiv:2603.16140):** label noise is destructive; this is *why*
  Phase A (precision ≥0.85) gates Phase B. Do not relax the gate to force the
  training to run.
- **Fabrication recurrence:** the single biggest .377 failure was stubbed compute.
  Phase B tasks declare `live_llm_inference` (60 s floor), require
  `model_specs`+`random_seed`+`reproducibility_checksum`, print per-epoch progress,
  and the eval verifies REAL on-disk checkpoints (file size) before reporting rates.
- **Honest null is a COMPLETE verdict.** A≈B (codex-distillation, verifier adds
  nothing) is a publishable decision-grade negative — report it plainly, do not
  dress it as a partial win. Likewise a Phase-A precision-rescue failure bounds the
  thesis cleanly.

## 8. Operator-pending (not auto-applied)

- **ops/north-star.md headline revision** (verifier "selects" → verifier "trains")
  remains operator-curated; a draft proposal exists
  (`ops/north-star-pivot-revision-proposal.md`). This roadmap operationalizes the
  pivot but does not edit the north star.

## 9. Self-learning coverage

The verifier-as-reward RFT (Phase B) IS the continuous-self-learning experiment
(FR-11): the model trains on its own verifier-certified traces — self-improvement
without oracle labels. The precision rescue (Phase A) is the safety precondition
that makes that self-learning loop trustworthy rather than self-poisoning.
