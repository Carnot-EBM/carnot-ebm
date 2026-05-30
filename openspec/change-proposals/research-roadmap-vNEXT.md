# Research Roadmap — Milestone 2026.05.319

**Depth-Over-Breadth V: P0.1 Got Real Numbers — Now Make the Energy *Informative* (Trained Reranker) and Take G2 to an External Run**

**Planner:** Claude Opus 4.8 (2026-05-30), under the MANDATORY Depth-Over-Breadth
Forcing Function (CLAUDE.md, 2026-05-30) and `ops/north-star.md`.

---

## 1. What the previous milestone (.318) proved

`.318` was the milestone where **P0.1 finally produced real, non-degenerate numbers**.
The decoupling worked exactly as designed:

| Exp | Outcome | Status |
|---|---|---|
| exp3448 (generation builder) | n=47/120 cached, resumable, per-token logprobs, SC non-degenerate (warm-up SC ≥ greedy, > 0.30). 1041s, exited clean on budget. | ✅ defeated the 1201s idle-timeout that killed exp3437 |
| exp3449 (P0.1 v4 cached scoring) | SC = energy-weighted vote = energy×SC hybrid = **0.87234** exactly; energy-argmin = greedy-AR = **0.78723**. delta 0.0, McNemar p=1.0. | ⚠️ **FLAGGED (tautology)** — but the cause is mechanistic, not a bug |
| exp3450 (energy-correctness calibration) | `energy_as_correctness_auroc = 0.516` (≈ chance), Spearman −0.03. | ✅ clean — **explains the tautology** |
| exp3451 (G2 CI + Docker clean-room) | CI workflow authored + Docker clean-room reproduced AUROC 0.9131. `g2_status = ci_and_docker_ready_external_run_pending`. | ✅ clean — G2 one step from closeable |
| exp3452 (FR-11 grounding collapse) | ARM A collapsed (entropy 0.028, mode-mass→1); ARM B (entropy β=0.5) held (entropy 4.98). | ⚠️ **FLAGGED (tautology)**: `pass_rate ≡ true_accuracy` — finding real, metrics need separating |
| exp3456 (gate synthesis) | G1 ✅ G2 ❌ G3 ✅ G4 ✅; `unmet_gates=['G2']`; `depth_forcing_function_can_relax=False` | ✅ clean |
| Hardware | KV260 blocked (SSH unreachable), GateMate blocked (toolchain missing), PolarFire reachable | honest blocks |

### The single most important takeaway

P0.1's honest answer is now **legible and convergent across three measurements**: at
matched compute the **untrained** energy (IsingVerifier arithmetic-violation +
EbmCotCalibrator adjacent-contradiction heuristics, weight 1.0, T=1.0) **does not beat
self-consistency, because the energy does not track answer correctness (AUROC 0.516 ≈
chance).** An uninformative energy → near-uniform softmax weights → energy-weighted vote
*degenerates to* majority vote (= SC), and energy-argmin *degenerates to* the greedy pick.
That degeneracy is precisely the bit-identical tautology exp3449 was flagged for. The
.317 Kona result (`energy_is_global_heuristic_hybrid_solves_pure_descent_does_not`)
converges with the same shape.

**This is not yet the final P0.1 verdict — it is the verdict *for an untrained energy*.**
The literature (arXiv:2505.14999 EORM; arXiv:2603.25450; arXiv:2506.09338) says the fix
is a *trained* outcome-label energy reranker. `.319`'s decisive depth test is whether a
**trained** energy crosses AUROC 0.55 and beats SC. If it still loses, the Phase-3
"energy-as-ground-truth-for-selection" premise is honestly REFUTED on this substrate; if
it wins, it is the **first real Phase-3 justification** in the project's history.

---

## 2. The three biggest gaps (current state → PRD vision)

1. **P0.1 is *almost* answered but the answer is "untrained energy is uninformative."**
   The PRD's foundation-model endgame rests on "the energy function is ground truth."
   `.318` showed the *current untrained* energy is NOT ground truth for GSM8K final-answer
   selection (AUROC 0.516). The gap: we have never tested a *trained* energy reranker
   (EORM-style) on this corpus. Closing it either validates or honestly refutes the
   selection premise. **This is the milestone's center of gravity.**

2. **G2 (independent reproduction) is the SOLE unmet publication gate, and it is now
   operator/CI-gated.** exp3451 built the CI workflow + a Docker clean-room that
   reproduces 0.9131. The only missing step is an *actual non-operator run*. Autonomous
   work cannot push or trigger external CI, but it CAN (a) validate the workflow runs
   green in a simulated/containerized runner so the operator's eventual trigger is
   low-risk, and (b) assemble the external-reproducer handoff package.

3. **Two .318 findings are correct-but-FLAGGED and cannot be cited until de-flagged.**
   The FR-11 grounding-collapse result (entropy-reg prevents self-distillation collapse —
   the mandatory continuous-self-learning finding) and the P0.1 scoring numbers both
   carry `flagged_adversarial=true`. Per the 2026-05-30 fabrication gate, flagged
   artifacts are quarantined from all forward claims. The gap: produce CLEAN re-runs
   (distinct metrics, held-out eval) so these results can enter the record.

---

## 3. Milestone architecture

```
PHASE A — OPS transition
  exp3458  archive .318, write retro, activate .319

PHASE B — DEPTH BLOCK (the majority; zero breadth churn)
  exp3459  P0.1 corpus EXTEND 47→120  ──────────────┐  (resume exp3448 builder)
                                                     │  live 26B GGUF, resumable
  exp3460  P0.1 v5: TRAINED energy reranker (EORM)   │  cached scoring, held-out
           + FoVer-verifier energy  vs  SC  ◄────────┘  THE decisive crux
              │                                         tautology-clean by design
              ├─ explained by ─► exp3461  calibration v2: does the TRAINED /
              │                            FoVer energy track correctness?
              │                            (vs untrained 0.516 baseline)
  exp3462  FR-11 grounding-collapse CLEAN re-run (de-flag; mandatory self-learning)
  exp3463  G2 external-reproducer readiness: CI dry-run + Docker re-confirm + handoff
  exp3464  Kona depth: does a TRAINED energy lift the global-opt HYBRID solve-rate?
           (extends .317 exp3440 "hybrid solves")

PHASE C — HARDWARE (light + opportunistic; north-star §3)
  exp3465  KV260 terminal latency transcript v5 (SSH-gated; drive-to-terminal)
  exp3466  GateMate opportunistic detect + toolchain continuity v3
  exp3467  PolarFire opportunistic reachability v5

PHASE D — OPS synthesis + capstone
  exp3468  G1–G4 gate-status synthesis v319  (gated on exp3460 clean verdict)
  exp3469  capstone v319                       (gated on exp3468)
```

### Dependency graph (data, not conductor order)

- `exp3459 → data/p01_gsm8k_generations.jsonl` (extended corpus) → consumed by
  `exp3460`, `exp3461`, `exp3464`.
- `exp3460` (trained-energy crux) → its `honest_verdict` gates `exp3468` (synthesis).
- `exp3461` is the *mechanistic explainer* for `exp3460` (read together).
- `exp3468 → gate_status_v319_ready` gates `exp3469` (capstone).
- Hardware tasks are independent (no downstream gating).

The gate chain deliberately depends on the **cheap, reliable cached-scoring** task
(exp3460), never on the live generation task — the lesson of the .317 cascade where
gating on a heavy retired P0.1 blocked the whole capstone.

---

## 4. Why each depth task answers a *new* question (no vN+1 churn)

Per the Depth-Over-Breadth Forcing Function, every task must advance the headline,
close a G-gate, or test a load-bearing-unproven link — never re-measure an
already-measured artifact:

| Task | New question (never answered) |
|---|---|
| exp3459 | Completes an *unfinished* corpus (47→120) — not a re-measurement; the builder is resumable by design. |
| exp3460 | Does a **trained** energy reranker beat SC? (.318 only tested an *untrained* energy.) THE crux. |
| exp3461 | Does the **trained / FoVer** energy track correctness, vs the untrained 0.516 floor? |
| exp3462 | De-flagged confirmation that at-risk grounding causes self-distillation collapse + entropy-reg cures it (mandatory self-learning; .318 version was flagged). |
| exp3463 | Does the G2 CI workflow actually run green in an isolated runner — i.e. is G2 *closeable* by a non-operator trigger? |
| exp3464 | Does a **trained** energy lift the Kona global-opt HYBRID solve-rate? (.317 only had an untrained heuristic.) |

---

## 5. Hardware requirements

- **exp3459, exp3460, exp3461, exp3464**: GPU (RTX 3090) for the cached corpus build
  (live 26B GGUF generation in exp3459) and for energy scoring. exp3460/3461/3464 are
  cached scoring + small-reranker training (CPU-feasible, GPU optional).
- **exp3462, exp3463, exp3468, exp3469, exp3458**: CPU only.
- **exp3465**: KV260 over SSH (`ssh kria`) — SSH-reachability precondition ONLY
  (never host `/dev/mmcblk*`, per KV260 SSH-Not-SD-Card Discipline).
- **exp3466**: GateMate via `openFPGALoader -c dirtyJtag --detect` + himbaechel toolchain.
- **exp3467**: PolarFire via `ssh polarfire`.

## 6. Models

All LLM work uses the mandated SOTA GGUF via the `.gguf` path (embedded tokenizer; NEVER
`AutoTokenizer` on a `-GGUF` repo id, per the 2026-05-29 GGUF tokenizer rule):
`unsloth/gemma-4-26B-A4B-it-GGUF` (fastest SOTA MoE, used for the resumable corpus build).
Cached scoring + reranker training run against the cached corpus (no live model → cannot
time out).

## 7. Continuous self-learning coverage

`exp3462` is the mandatory continuous-self-learning experiment (PRD FR-11): a clean,
de-flagged measurement of whether the FR-11 self-improvement loop mode-collapses onto the
at-risk verifier null space (exp3439: λ_min≈0, eff-k 3.54) and whether entropy
regularization is the antidote — directly actionable for any future Phase-5 deployment.

## 8. Risk notes

- **Small-n training risk (exp3460):** n=120 problems × 6 samples ≈ 720 labeled
  candidates is thin for training a reranker. Mitigation: cross-validation / held-out
  split with the split documented; report the train/test boundary explicitly so the win
  (if any) is not leakage. If n stays < 80, report a *preliminary* trained-energy verdict
  and resume the corpus next milestone.
- **Tautology-clean by construction:** exp3460 and exp3462 must NOT emit two
  conceptually-distinct metrics that are forced equal. Where a metric legitimately equals
  another (e.g. trained-energy-vote degenerating to SC under a still-uninformative energy),
  record it once with a `methodology_note`, not as two bit-identical fields.
- **Depth-Over-Breadth does NOT relax this milestone** unless exp3460 lands a CLEAN P0.1
  verdict AND G2 has a concrete in-flight external reproducer. The capstone reports the
  relax decision.
