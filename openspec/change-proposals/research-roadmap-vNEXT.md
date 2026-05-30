# Research Roadmap — Milestone 2026.05.318

**Depth-Over-Breadth IV: Decouple the P0.1 Crux So It Can Finally Answer + Take G2 to an External Reproducer**

**Planner:** Claude Opus 4.8 (2026-05-30), per the Depth-Over-Breadth Forcing
Function (planner switched off gemini precisely so this prose discipline is
followed reliably).

**Status of the governing constraint:** Depth-Over-Breadth does **NOT** relax.
P0.1 (the energy-vs-autoregressive premise that the entire Phase-3 / Kona
foundation-model endgame rests on) is still unanswered after three failed
attempts. G2 (the sole unmet publication gate) is internally clean-room
reproducible but has no non-operator run yet. Until P0.1 has a *clean* verdict
**and** G2 has a concrete in-flight external reproducer, the milestone is
reserved for depth.

---

## 1. What the previous milestone (.317) proved

`.317` (Depth-Over-Breadth III) routed every depth task to `claude` (the
gemini-cli has been systemically down since .315) and landed three of four depth
verdicts cleanly. The crux still did not answer:

| Task | Outcome | Reading |
|---|---|---|
| **P0.1 v3** (exp3437, energy-vote vs SC, real harness) | **TIMED OUT 3× → RETIRED** | `Wall-clock+idle timeout after 1201s (1201s silence)`. The live 35B generation over 200×k samples ran silently past the ~20-min idle ceiling, produced no artifact. **P0.1 still unanswered.** |
| **G2** (exp3438, clean-room root-cause+fix) | **FIXED** | Root cause = undeclared `scikit-learn` dep. A fresh worktree+venv now reproduces AUROC **0.9131** + the FR-11 contribution in their CIs. `cleanroom_reproducible_internal_external_run_pending`. |
| **P0.2** (exp3439, λ_min diversity audit) | honest negative | `null_space_collapse_confirmed`; λ_min(Σ)=−0.0, eff-k=3.54. α_t grounding **at risk**. |
| **Kona** (exp3440, solve-rate gate) | honest negative | Encoding valid (E==0 for a solved board), pure energy descent `solve_rate=0.0`; only the energy+constraint-propagation **hybrid** solves. `energy_is_global_heuristic`. |
| **Injection** (exp3441) | landed | k=15 ensemble AUROC **0.831** vs single-KAN 0.475; `beats_sidecar_but_below_replacement_grade`. |
| KV260 (exp3442) | blocked | `blocked_kv260_ssh_unreachable` (operator action). |
| Gate-synth (exp3445) + Capstone (exp3446) | **GATE_BLOCKed** | Both gated on the retired P0.1 (exp3437) → cascade skip. The .317 capstone never landed. |

**The single most important lesson is architectural, not scientific.** P0.1 has
now failed three *distinct* ways — flagged (exp3312), degenerate 0.0-vs-0.0 tie
harness (exp3426), and idle-timeout (exp3437). The first two were measurement
bugs; the third is a hard infrastructure limit: **one in-session live-inference
job of 200×k 35B generations cannot complete inside the agent's ~20-minute
wall-clock+idle budget.** Every prior P0.1 attempt coupled generation and scoring
in a single task, so a slow/silent generation step killed the whole experiment.

---

## 2. The three biggest gaps (current state vs PRD vision)

1. **P0.1 is unanswerable in its current single-task shape.** The PRD's
   foundation-model endgame (REQ-KONA-001/002: non-autoregressive reasoning on
   continuous latents) has *never* been tested on a real task vs a real baseline.
   The blocker is now purely structural — the experiment times out before it
   measures anything.
2. **G2 has no non-operator reproducer.** It is internally clean-room
   reproducible (exp3438) but the gate explicitly requires "≥1 reproducer who is
   not the operator." The Phase-1 ship gate counts "a CI run" as that reproducer
   — but no CI reproduction workflow exists yet.
3. **The α_t grounding that the self-learning thesis rests on is measured
   at-risk but its consequence is untested.** exp3439 showed λ_min(Σ)≈0; whether
   that actually causes self-distillation collapse in an FR-11 self-improvement
   loop has never been run (the continuous-self-learning depth question).

---

## 3. The decisive structural change: split P0.1 into generation + scoring

```
                       .318 P0.1 PIPELINE (decoupled)

  exp3448  P0.1 GENERATION CORPUS BUILDER            exp3449  P0.1 v4 SCORING
  ┌──────────────────────────────────────┐          ┌────────────────────────────┐
  │ live gemma-4-26B-A4B-it-GGUF (fastest │          │ reads data/p01_*.jsonl      │
  │ SOTA MoE, ~4B active)                 │  cached  │ (no live model → cannot     │
  │ • k samples/problem on GSM8K          │  corpus  │  idle-timeout)              │
  │ • per-token logprobs (self-certainty) │ ───────► │ 6 conditions @ matched      │
  │ • CHECKPOINT per problem (resumable)  │ .jsonl   │  compute:                   │
  │ • PRINT progress per problem          │          │  greedy AR · SC-majority ·  │
  │   (defeats the 1201s idle-timeout)    │          │  self-certainty BoN ·       │
  │ • exit clean on partial: progress     │          │  energy-argmin ·            │
  │   not failure                         │          │  energy-weighted vote ·     │
  └──────────────────────────────────────┘          │  energy×SC HYBRID           │
                                                     │ + NON-DEGENERATE-SC gate    │
                                                     │ + McNemar / paired bootstrap│
                                                     └────────────────────────────┘
```

Why this finally works:

- **The generation builder cannot idle-timeout** because it prints a progress
  line *per problem* (the subprocess is never silent for 20 minutes) and appends
  each completed problem to the JSONL immediately, so a re-invocation resumes.
- **The scoring task cannot time out** because it invokes no live model — it is
  `verifier_ensemble_against_cached_candidates`, the same substrate that makes
  the FoVer headline (exp2837) and G2 clean-room (exp3438) robust (seconds, not
  hours).
- **A partial corpus is progress, not failure.** The builder exits 0 with
  `complete: generation_corpus_at_n=NN`. The scoring task answers P0.1 on
  whatever n≥30 exists (preliminary at n=30–80, headline-eligible at n≥80) and
  the corpus extends over future milestones.
- **The capstone chain no longer cascade-blocks.** Gate-synth (exp3456) gates on
  the *cheap, reliable scoring* task (exp3449), not the heavy generation task —
  and exp3449 emits a `complete:` verdict under every outcome (including a clean
  blocked-on-small-corpus verdict), so it can never retire-and-cascade the way
  exp3437 did.

The scoring task incorporates the .317 + new-references findings: the honest bar
is **energy (or the energy×SC hybrid) ≥ plain majority-vote self-consistency**
(arXiv:2410.12608, 2510.14913), not merely "energy beats greedy AR"; a hybrid
condition is added because both the Kona result (exp3440) and arXiv:2510.14913
show verifier+sampling hybrids beat either alone.

---

## 4. Architecture (unchanged; the relevant slice)

```
  GSM8K problem
       │
       ▼
  ┌─────────────────────────┐     k samples (temp~0.8)      ┌───────────────────────┐
  │ SOTA GGUF (gemma-4-26B-  │ ───────────────────────────► │ per-sample answer      │
  │ A4B, llama.cpp embedded  │     + per-token logprobs      │ extraction + logprob   │
  │ tokenizer)               │                               │ confidence             │
  └─────────────────────────┘                               └──────────┬────────────┘
                                                                        │
          ┌─────────────────────────────────────────────────┬─────────┴───────────┐
          ▼                         ▼                         ▼                     ▼
   majority vote (SC)      self-certainty BoN        Boltzmann-GPT / verifier   energy-weighted
                                                     energy per candidate        vote = softmax(-E/T)
                                                     (carnot.phase3)             + energy×SC hybrid
```

The verifier/Boltzmann energy substrate (`python/carnot/phase3/boltzmann_gpt.py`,
`continuous_ebm.py`) scores each cached candidate; no new architecture is built.

---

## 5. Phases and tasks (11 tasks; exp3447–exp3457)

**Phase A — OPS transition (1 task)**
- `exp3447` Archive .316 **and** .317 honestly (the .317 capstone never landed),
  activate .318.

**Phase B — Depth block (5 tasks; the majority of substantive slots; all on claude)**
- `exp3448` **P0.1 generation corpus builder** — resumable, progress-printing,
  live gemma-4-26B-A4B-it-GGUF. The timeout fix.
- `exp3449` **P0.1 v4 scoring** — energy-weighted vote vs SC vs self-certainty
  BoN vs energy×SC hybrid, on the cached corpus, non-degenerate-SC gate, paired
  significance. **THE crux answer.**
- `exp3450` **Energy-correctness calibration audit** — does the energy signal
  rank-correlate with answer correctness on the same corpus? Explains *why* P0.1
  comes out the way it does (an energy that doesn't track correctness cannot beat
  SC).
- `exp3451` **G2 external-reproducer: CI workflow + Docker clean-room** — author
  a GitHub Actions workflow + run a Docker-isolated clean-room reproduction so a
  non-operator runner can close G2.
- `exp3452` **FR-11 grounding-collapse stress test** — the mandatory
  continuous-self-learning task: does the .317 at-risk grounding (λ_min≈0)
  actually cause self-distillation collapse in an FR-11 loop?

**Phase C — Hardware (3 tasks; light + opportunistic, north-star §3)**
- `exp3453` KV260 terminal latency transcript (SSH precondition; drive to
  terminal then freeze).
- `exp3454` GateMate opportunistic detect + continuity (light, no flash mandate;
  avoids the .317 TAUTOLOGY flag).
- `exp3455` PolarFire opportunistic reachability + continuity.

**Phase D — OPS synthesis + capstone (2 tasks)**
- `exp3456` G1–G4 gate-status synthesis v318 (gated on the cheap P0.1 *scoring*
  task, not generation — breaks the .317 cascade).
- `exp3457` Capstone v318 (gated on exp3456).

### Dependency graph

```
exp3447 (archive/activate)
   │
   ├── exp3448 (P0.1 generation) ──writes──► data/p01_gsm8k_generations.jsonl
   │                                              │
   │        exp3449 (P0.1 scoring) ◄──reads───────┤  (reads corpus directly;
   │             │                                │   NOT a hard gated_on, so a
   │             │              exp3450 (energy   │   partial corpus still scores)
   │             │               calibration) ◄───┘
   │             │
   ├── exp3451 (G2 CI/Docker)        [independent]
   ├── exp3452 (FR-11 collapse)      [independent]
   ├── exp3453/3454/3455 (hardware)  [independent]
   │
   └── exp3456 (gate-synth) ──gated_on exp3449.honest_verdict contains 'complete'──►
              │
              └── exp3457 (capstone) ──gated_on exp3456.gate_status_v318_ready==true
```

---

## 6. Hardware requirements

- **exp3448** (P0.1 generation): 1× RTX 3090 (CUDA), gemma-4-26B-A4B-it-GGUF via
  llama.cpp. The only live-inference task.
- **exp3449/3450/3451/3452**: CPU only (cached-candidate scoring + isolated
  install). No GPU.
- **exp3453**: KV260 over SSH (`ssh kria`). **SSH-reachability is the ONLY
  precondition** — never a host SD-card-slot check (KV260 SSH-Not-SD-Card
  Discipline).
- **exp3454/3455**: GateMate USB-JTAG detect / PolarFire SSH — light, no flash.

---

## 7. Discipline compliance

- **Depth-Over-Breadth:** every depth task answers a never-answered question
  (P0.1 energy-vs-SC, energy-correctness calibration, FR-11 collapse) or advances
  the sole publication gate (G2). **No vN+1 re-measurement of an already-measured
  artifact.** P0.2/Kona/injection landed clean verdicts in .317 and are NOT
  re-run; their findings feed the capstone.
- **Gemini-Default carve-out:** all experiment tasks route to `claude`
  (`requires_claude: true`) because the gemini-cli is demonstrably down (every
  gemini task crashed 3× in .315 *and* .316) and the depth tasks need multi-file
  judgment. Heavy/long tasks pre-route to `model: opus`.
- **Pre-Launch Preconditions:** every compute-bound / hardware / SSH / GGUF task
  carries a step-0 PRECONDITIONS block with explicit `blocked_*` fallbacks.
- **GGUF tokenizer rule:** exp3448 loads via the `.gguf` path + `llama_cpp`
  (vocab_only preflight), never `AutoTokenizer` on a `-GGUF` repo id.
- **Adversarial-verify / fabrication gate:** exp3448 emits a real ≥60 s duration,
  seed, checksum, model_specs; cached-scoring tasks declare the correct
  `inference_substrate`; the capstone skips any `flagged_adversarial` artifact.
  exp3454 explicitly avoids emitting two bit-identical metrics (the .317 GateMate
  TAUTOLOGY flag).
- **Prior-failure / exclusion-manifest:** every scope-matched task carries a
  `prior_failures:` block (all four sub-fields) and/or an `operator_override:`.
- **Public-doc discipline:** no task edits `docs/index.html`, `README.md`,
  `ops/north-star.md`, or other operator-curated docs.
