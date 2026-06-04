# Research Roadmap — Milestone 2026.06.351

**Status:** PROPOSED (planner draft, 2026-06-04, Claude Opus 4.8)
**Theme:** Recursive-Refiner Phase-3 thesis — UNBLOCK + cheap kill-gates + the
distillation-oracle path; verifier-moat deepening (contamination-free formal
core + error-independence); hardware continuity; lean close-out.
**Prior milestone:** 2026.06.350 (operator-activated; recursive-refiner Task 0
+ FoVer formal-vs-learned ablation).
**Milestone doc this file:** `openspec/change-proposals/research-roadmap-vNEXT.md`
**Provenance:** `docs/research-notes/phase3-recursive-refiner-thesis.md`
(operator-seeded 2026-06-04), `[[reference_tiny_recursive_reasoning_models]]`,
`[[reference_deep_think_post_bounded_2026_06]]`.

---

## 1. What the previous milestone (.350) proved

`.350 was a single-focus operator-activated milestone with two tasks:

- **exp3819 — Latent-Symbol Bridge (recursive-refiner Task 0):**
  `blocked_trm_checkpoint_not_available`. The probe never ran — the ONLY
  blocker was a missing pretrained TRM checkpoint (and no bounded tiny-train
  was attempted within budget). The scientific question (can an external
  discrete verifier read a recursive refiner's intermediate latents in-loop?)
  remains OPEN, not answered.
- **exp3820 — FoVer formal-vs-learned ablation:**
  `complete: INCONCLUSIVE_ablation_harness_unfaithful_full0.8929_expected_0.9131`.
  The ablation harness reproduced **0.8929** for the full ensemble, outside the
  ±0.01 sanity band around the frozen **0.9131** headline → the deltas are not
  yet trustworthy. BUT the run surfaced a **striking, provisional** signal worth
  confirming: `formal_only_auroc = 0.9612` (AST + SemanticConsistency + Z3,
  **no trained weights**) > `full = 0.8929` > `learned_only = 0.8898`. If real,
  the **contamination-free formal core retains/exceeds the entire moat** — the
  strongest possible answer to the Deep-Think Q4 OOD-contamination critique.
  Cannot be cited until the harness reproduces 0.9131.

**Converged invariants carried forward (unchanged):** `paper_ready = TRUE`
(G1–G4 pass; G2 closed via the CI reproducer); FoVer **0.9131** frozen;
energy-as-selector (P0.1) bounded; energy-as-generator (EBT / Thesis-A) bounded
at small scale. The loop does NOT self-seed a new paradigm (DT-P3 Verification
Trap) — the recursive-refiner thesis is an **operator seed**, which is why it is
the live thread this milestone advances.

## 2. The three biggest gaps (current state vs PRD vision)

1. **The one live operator-seeded Phase-3 bet is stalled on a $0 blocker.**
   The recursive-refiner thesis (TRM-style refiner + Carnot verifier as the
   distillation oracle for a continuous Q-head) is the project's strongest path
   to the foundation-model mission given both energy routes are bounded. Its
   first kill-gate (Task 0) blocked purely on checkpoint availability. The
   `.351 literature sweep RESOLVED this: `huggingface.co/arcprize/trm_arc_prize_verification`
   (real ARC TRM checkpoint) + `github.com/olivkoch/nano-trm` (clean Sudoku
   tiny-train). **Gap: run the Deep-Think-revised kill-gate sequence now that it
   is unblocked.**

2. **The moat's robustness is asserted, not measured.** DT-post-bounded P2 found
   the verifier moat is **error-INDEPENDENCE, not AUROC**, and is fragile to
   subsumption by a strong reasoner. exp3820 hints the formal core is the robust
   part — but the harness was unfaithful, and no residual-FPR / error-overlap
   test against a strong SOTA reasoner has ever run. **Gap: make the formal-core
   finding trustworthy AND measure whether the verifier catches errors a strong
   reasoner misses (the scissor-plot / error-independence test).**

3. **No methods spine for the distillation-oracle path, and continuous
   self-learning has plateaued.** FR-11 reached v21 with diminishing returns
   (Tier-3 predictor AUROC 0.9715, fast-path skip ~0.56). The PRD's continuous
   self-learning goal (Tiers 1–4) now points squarely at **distilling the
   verifier into a continuous Q-head** (VerifierQ / RL^V machinery) — the same
   artifact the recursive-refiner thesis needs. **Gap: stand up the offline
   Q-head distillation pipeline as the next self-learning frontier.**

## 3. Architecture of the milestone

```
 PHASE A — recursive-refiner CHEAP KILL-GATES (decide whether the thesis proceeds)
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ exp3821  Latent-Symbol Bridge Task 0 RE-RUN  (UNBLOCKED via arcprize/      │
 │          nano-trm) — can an external discrete verifier read intermediate   │
 │          latents in-loop? Expected FALSIFIED → confirms distillation pivot │
 │ exp3822  P1 falsification — vanilla TRM vs matched-COMPUTE AR on a 1D       │
 │          variable-length headroom task: does the paradigm escape grids?    │
 │ exp3823  P2 falsification — energy-fit to TRM's Δh (curl-free test):        │
 │          confirm TRM ≠ EBT (asymmetric vector field, not energy descent)   │
 └──────────────────────────────────────────────────────────────────────────┘
                                   │ (gate: headroom must exist; P1 must pass)
 PHASE B — the DISTILLATION-ORACLE path (the actual integration)
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ exp3824  Headroom-gate corpus + positive control (MANDATORY before any     │
 │          generator training): AR greedy ≈20% AND AR+SC@32 < 50%; ABORT if  │
 │          AR+SC@32 > 75% (the P0.1/Thesis-A ceiling-pollution trap)         │
 │ exp3825  Distillation-oracle prototype — Carnot verifier scores final TRM  │
 │          trajectories → train a continuous Q-head (VerifierQ/RL^V offline   │
 │          Q-learning). [CONTINUOUS SELF-LEARNING — Tier 3/4]                 │
 │          gated_on exp3824 headroom_confirmed == true                       │
 └──────────────────────────────────────────────────────────────────────────┘
 PHASE C — VERIFIER-MOAT deepening (banked product; DT-Q4 + DT-P2)
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ exp3826  FoVer ablation FAITHFUL re-run — reproduce 0.9131 ±0.01, then     │
 │          trust formal_only (the 0.9612 contamination-free-core finding)    │
 │ exp3827  Verifier error-INDEPENDENCE scissor test — does the ensemble      │
 │          catch errors a strong SOTA reasoner (Qwen3.6-35B) misses?         │
 │ exp3828  Verifier-ensemble vs adaptive prompt-injection corpus            │
 │          (MANDATORY-NEXT priority; AND-composition vs single-KAN 0.475)     │
 └──────────────────────────────────────────────────────────────────────────┘
 PHASE D — HARDWARE continuity + INFRA + close-out
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ exp3829  GateMate n=16 Ising bitstream flash + timing smoke v3 (owed)      │
 │ exp3830  KV260 + PolarFire opportunistic continuity audit (SSH precond.)   │
 │ exp3831  Capstone v350 + external-research refresh (file .351 references)   │
 │ exp3832  Archive .350 / activate .351 manifest                            │
 └──────────────────────────────────────────────────────────────────────────┘
```

## 4. Phase descriptions

### Phase A — Recursive-refiner cheap kill-gates
The Deep-Think-revised plan (thesis doc §"Revised plan") says: run Task 0 FIRST
(it can kill the naive in-loop integration for $0), then the P1/P2 cheap
falsifications. All three are now runnable on the dev rig with the unblocked
checkpoint. **None requires a SOTA GGUF** — the experimental object is the tiny
TRM substrate itself, not a general LLM.
- **exp3821** force-decodes intermediate latents at every refinement step and
  queries a programmatic grid verifier; measures intermediate-state parseable
  rate, Spearman(verifier-signal, step), and decode+verify latency overhead.
  FALSIFIED (expected) confirms Carnot's role is the **offline distillation
  oracle**, not the runtime Q-head. VIABLE (surprising) reopens in-loop.
- **exp3822** tests DT-P1: does TRM beat matched-COMPUTE AR on a strictly-1D
  variable-length task with AR headroom (algorithmic sequence correction /
  multi-step parity)? If TRM fails there → the paradigm is grid-bound and cannot
  reach the foundation-model mission. Honest-negative is informative.
- **exp3823** tests DT-P2: fit a scalar energy E to a trained TRM's latent
  updates Δh under the constraint Δh ≈ −∇E(h). The curl-free constraint should
  make this FAIL — proving TRM expresses an asymmetric vector field EBT cannot,
  so the bounded-EBT result does NOT cover TRM. Pure numerical → codex.

### Phase B — The distillation-oracle path
- **exp3824** is the non-negotiable headroom gate (tighter per DT): curate a
  grid corpus where AR greedy ≈ 20% AND AR+Self-Consistency@32 plateaus FIRMLY
  < 50%, and **ABORT if AR+SC@32 > 75%** (the ceiling-pollution trap that
  produced two prior expensive false negatives). Explicitly NOT a P0.1 rerun:
  combinatorial grid with confirmed oracle>AR headroom, not NL-math.
- **exp3825** stands up the offline Q-head distillation pipeline: Carnot's
  verifier scores unrolled FINAL TRM trajectories → BCE/MSE (or VerifierQ-style
  IQL/conservative-Q) trains a continuous Q-head that natively reads continuous
  latents. This is the **continuous self-learning** experiment for this milestone
  (Tier 3 predictive → Tier 4 adaptive). Scaffolding-heavy + well-documented
  recipe → codex. Gated on exp3824 confirming headroom (if the corpus is
  ceiling-polluted, training is meaningless — skip).

### Phase C — Verifier-moat deepening
- **exp3826** repairs the exp3820 harness so the full ensemble reproduces 0.9131
  ±0.01 (positive control), making the `formal_only = 0.9612` decomposition
  citable. If real, the contamination-free formal core is the moat's robust floor
  and the DT-Q4 OOD-contamination critique is answered.
- **exp3827** implements the DT-P2 scissor test: take a strong SOTA reasoner
  (`unsloth/Qwen3.6-35B-A3B-GGUF`) as the "subsumer", and measure the
  **residual false-negative / error-overlap**: of the step-errors the strong
  reasoner's own self-verification misses, what fraction does Carnot's ensemble
  catch? Moat survives iff error-INDEPENDENCE is high (not just AUROC parity).
- **exp3828** runs the pending MANDATORY-NEXT priority: does AND-composition beat
  the single-KAN 0.475 baseline on an adaptive prompt-injection perturbation
  corpus? Verifier-scoring against cached/generated adversarial candidates.

### Phase D — Hardware continuity + infra + close-out
- **exp3829** discharges the Hardware-Task Continuity Discipline debt for the
  GateMate A1-EVB-2M (not yet terminal): yosys `synth_gatemate` →
  `nextpnr-himbaechel --device CCGM1A1` → `gmpack` for an n=16 Ising tile,
  flash via `openFPGALoader -c dirtyJtag -b olimex_gatemateevb`, record
  sample-level timing. PRECONDITIONS gate on the himbaechel toolchain + board
  IDCODE; `blocked_*` rather than fabricate if the board is detached.
- **exp3830** opportunistic SSH-precondition audit of KV260 (terminal) +
  PolarFire (not yet terminal — needs a hash-verified dispatch run).
- **exp3831 / exp3832** are the standard aggregation close-out (capstone +
  research refresh; archive/activate). `inference_substrate:
  aggregation_from_upstream_artifacts`.

## 5. Dependency graph

```
exp3821 ─┐ (checkpoint reused)
         ├─► exp3823 (needs a trained TRM's Δh)
exp3824 ─┴─► exp3825 (gated_on headroom_confirmed == true)
exp3822  (independent — 1D task, own AR baseline)
exp3826 ─► (informs the moat story; independent of A/B)
exp3827  (independent; strong-reasoner SOTA GGUF)
exp3828  (independent; cached/adversarial corpus)
exp3829, exp3830 (independent hardware)
exp3831, exp3832 (close-out; read all upstream artifacts)
```

## 6. Hardware requirements

- **exp3821/3822/3823/3825:** tiny TRM (~7–50M params). CPU-runnable for the
  arcprize-checkpoint inference probe; small-Sudoku tiny-train is GPU-opportunistic
  (≤20 min on one RTX 3090). GPU experiments MUST invoke `.venv/bin/python`, never
  bare `python` (the EBT venv/CUDA infra trap — `[[incident_ebt_training_venv_python_cuda]]`).
- **exp3827:** one RTX 3090 for the `Qwen3.6-35B-A3B-GGUF` strong-reasoner pass
  (llama.cpp CUDA path; load via the `.gguf` path, NOT `AutoTokenizer` on the
  GGUF repo id — the GGUF tokenizer rule).
- **exp3829:** GateMate A1-EVB-2M (USB DirtyJTAG `1209:c0ca`) + oss-cad-suite
  himbaechel flow. **exp3830:** `ssh kria` + `ssh polarfire`.
- **Loop-safety:** keep Phase A GPU-light to avoid kill_zombies contention;
  gemini is NOT used for any model/GPU task (it crashes real GPU workloads,
  `[[incident_333_gemini_quota_crash_wipeout]]`, exp3703) — claude/opus for
  judgment+GPU, codex/gpt-5.5 for formulaic numerical/scaffolding.

## 7. SOTA-model compliance

Per CLAUDE.md, the only task needing a general reasoning LLM is **exp3827**,
which uses `unsloth/Qwen3.6-35B-A3B-GGUF` (flagship MoE) as the strong-reasoner
subsumer. The recursive-refiner tasks study the tiny TRM substrate (the object
under test) and the verifier-scoring tasks read cached corpora — neither is a
"needs a general LLM" case, so no SOTA GGUF is mandated for them.

## 8. Kill-gates / what would retire a thread

- **exp3821 VIABLE** (surprising) reopens the in-loop Q-head; FALSIFIED (expected)
  commits the milestone to the distillation-oracle path. Either way the thesis
  proceeds via distillation — exp3821 only decides WHERE the verifier sits.
- **exp3822 BOUNDED** (TRM ≤ matched-compute AR on the 1D task) → the paradigm
  is grid-bound; record and de-prioritize the foundation-model ambition (the
  thesis narrows to "Carnot owns the oracle for the *grid-class* small-reasoners").
- **exp3824 ABORT** (AR+SC@32 > 75%) → corpus is ceiling-polluted; fix the corpus
  before any generator training (do NOT proceed to exp3825).
- **exp3826 INCONCLUSIVE again** → the ablation harness is structurally unable to
  reproduce 0.9131; escalate to operator (do not cite formal_only).

## 9. Decentralization implications (CLAUDE.md rules 1–7)

All tasks run on local open substrates (tiny TRM, local GGUF via llama.cpp,
on-prem FPGA boards). The distillation Q-head (exp3825) is a small, local,
open-weight artifact — a sovereignty-positive component (rule 1/5). No
closed-weight dependency is introduced. exp3831 files mirrorable references.

## 10. Parallel staged thesis (NOT activated this milestone)

EDLM (energy as residual-corrector over discrete diffusion) remains an
operator-seeded staged route (exp3793 preflight GO, exp3815 seed staged). It is
NOT allocated a task here — the operator activated the recursive-refiner thread
for `.350, and this milestone continues it. The exp3825 distillation pipeline is
substrate-agnostic and would serve an EDLM Q-head too if the operator later
activates that route.
