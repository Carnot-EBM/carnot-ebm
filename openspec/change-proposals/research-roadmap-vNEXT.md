# Research Roadmap — Milestone 2026.06.343

**Phase-3 Thesis A — attempt 3: pin the venv/CUDA, then run the GENUINE
energy-as-generator kill-gate to completion.**

Outer-loop / operator pre-staged roadmap, drafted 2026-06-02 (Claude Opus 4.8).

---

## TL;DR

The operator-seeded Phase-3 Thesis A (energy-as-GENERATOR / EBT, arXiv:2507.02092)
has now been blocked **three times by infrastructure, never by the mechanism**:

| Milestone | What blocked the kill-gate | Verdict mislabel |
|---|---|---|
| `.341` | exp3728 cwd/import-path bug → 0 steps | exp3729 called it "bounded at small scale" (false-negative) |
| `.342` | exp3734 ran 2 steps with **`cuda:false`** (on CPU, 100 MB VRAM) yet claimed "stable_so_far"; exp3735 hit **`blocked_cuda`** | exp3736 correctly said "untested — training did not complete" |

`.342` correctly *recovered* the `.341` false-negative (exp3733 corrigendum) and
correctly *refused to overclaim* (exp3736 = untested, exp3739 = part-b not-run).
But it **never actually trained the EBT on the GPU**. The mechanism is still
untested.

**Root cause (verified live 2026-06-02):** the experiment Run commands use bare
`python scripts/...`. On this box, `/usr/bin/python3` has **no torch at all**, and
the only interpreter with torch (2.11.0+cu128, both RTX 3090s visible,
`cuda.is_available()==True`) is **`.venv/bin/python`**. When the conductor's agent
ran exp3734 it reached a torch whose `cuda.is_available()` returned `False`, then
**silently dropped to CPU** and reported a 2-step "stable_so_far" signal — a soft
false-positive — instead of blocking.

**`.343` is the direct, gated continuation that fixes this for good:**

1. **Pin `.venv/bin/python`** in every GPU task's Run command AND precondition check.
2. **HARD-block on `cuda:false`** — emit `blocked_cuda` and STOP. **Never** train on
   CPU and report a stability signal (the exp3734 anti-pattern). Capture
   `sys.executable`, `CUDA_VISIBLE_DEVICES`, and `nvidia-smi` on block so the cause
   is auditable.
3. Run the **GENUINE** bounded checkpointed training of the tiny EBT + matched tiny
   AR baseline (the EBT-paper stability recipe), render the **real part-(a) verdict**
   superseding exp3736, and — if stable — run the **matched-COMPUTE** comparison
   (kill-gate part b).

**INVARIANTS (do not regress):** `paper_ready` stays TRUE (G1–G4 closed 2026-05-31);
frozen FoVer headline 0.9131 stays frozen; P0.1 / energy-**SELECTION** stays
settled-bounded — this milestone tests **GENERATION**, a different mechanism. The
banked verifier product is unaffected; this is the venture bet with a knife to its
own throat.

---

## What `.342` proved (read honestly)

- The record-honesty machinery works: the `.341` infra false-negative was cleanly
  corrected (exp3733, exp1850 pattern), and the `.342` verdicts refused to overclaim
  an un-run test (exp3736 "untested", exp3739 "not-run"). **This discipline is the
  reason we can keep betting cheaply.**
- The matched-compute harness (exp3727) is built + unit-tested and unused — it waits
  on a trained EBT.
- The single-step smoke (exp3726, `.341`) already showed a 38M EBT fits one 3090 at
  1283 MB with finite, decreasing loss. The model is not the problem.
- **The only thing standing between us and a real Thesis-A signal is reliable CUDA
  in the experiment subprocess.** That is a one-line-class fix (`.venv/bin/python`)
  plus a discipline fix (block, never CPU-drop).

## The 3 biggest gaps vs the PRD vision

1. **No empirical signal on energy-as-generator.** Phase 3's endgame is a
   hardware-acceleratable EBM/EBT foundation model. The operator seeded the most
   direct test of that thesis and it remains untested at bounded scale purely for
   infra reasons. `.343` closes this gap or bounds the route with a real divergence.
2. **The experiment harness silently degrades GPU→CPU.** A training task that finds
   `cuda:false` and trains 2 CPU steps while reporting "stable" is a fabrication-class
   failure mode the adversarial-verify floor did not catch (2 steps is too short to
   trip duration heuristics, and `inference_substrate` was self-declared
   `live_llm_inference`). `.343` ships the discipline fix: GPU tasks block on
   `cuda:false`, never CPU-drop.
3. **Continuous self-learning has nothing yet to learn from on this substrate.**
   FR-11 v15 (`.342`) initialised a stabilizer-efficacy tracker over 3 *aborted*
   chunks. Once the EBT actually trains, the tracker gets real divergence/stability
   data to upweight an effective recipe (Tier-1 CPU counter-updates) — the
   self-learning experiment mandated by research-program.md.

---

## Architecture of the milestone

```
 exp3743  archive .342 / activate .343            (ops, codex)
    |
 exp3744  corrigendum: .342 part-(a) was AGAIN     (phase3, codex, aggregation)
    |     infra-blocked (cuda:false bare-python;
    |     exp3734 over-claimed "stable" on 2 CPU
    |     steps); part-(a) STILL untested
    |
 exp3745  THE FIX: .venv/bin/python + HARD cuda     (phase3, claude/opus, GPU)
    |     block (never CPU-drop) + real bounded
    |     training chunk 1 of tiny EBT + matched AR
    |          | cumulative_steps_trained (bare int)
    v          v
 exp3746  resume bounded training chunk 2     [gated: exp3745 steps > 0]
    |          | (phase3, claude/opus, GPU)
    v
 exp3747  REAL part-(a) verdict (supersedes exp3736) (phase3, codex, aggregation)
    |          | green_light_343 (bare bool)
    v          v
 exp3748  EBT energy-descent generation smoke  [gated: exp3747 green_light_343==true]
    |          | ebt_can_generate (bare bool)   (phase3, codex, GPU)
    v          v
 exp3749  THE THESIS TEST: matched-COMPUTE      [gated: exp3748 ebt_can_generate==true]
    |          | EBT energy-descent vs AR best-of-M at EQUAL FLOPs, n>=100
    |          | accuracy_delta (bare float)     (phase3, codex, GPU)
    v          v
 exp3750  part-(b) verdict over exp3749               (phase3, codex, aggregation)
              ebt_beats_ar_at_matched_compute (bare bool)

 exp3751  FR-11 self-learning v16 — Tier-1 stabilizer tracker (self-learning, codex)
          resumes exp3740 (v15) state with REAL training diagnostics
 exp3752  KV260 opportunistic terminal-state audit          (hardware, codex)
 exp3753  capstone .343 — state the honest Thesis-A outcome (ops, codex)
```

### The genuine two-part kill-gate (unchanged in substance from `.342`, now actually runnable)

- **Part (a) — stability.** The tiny EBT trains to STABLE convergence within the
  bounded 3090 budget. PASS requires ALL of: `cumulative_steps_trained > 0` on a REAL
  GPU run (not the CPU-drop); no NaN/inf/divergence; bounded gradient norms; a
  non-degenerate loss/energy trajectory (NOT a monotonic runaway to −∞ — EBT energy
  is unbounded, so a collapsing energy can masquerade as convergence).
- **Part (b) — the thesis.** At EQUAL total inference FLOPs, EBT energy-descent
  generation beats the matched AR best-of-M baseline on held-out GSM8K reasoning
  accuracy (n≥100), AND/OR the gap narrows with 2× training.
- **Matched-COMPUTE, never matched-params (the P0.1 lesson, load-bearing):** energy
  descent runs N forward passes per prediction, so a matched-params "win" is just
  extra inference FLOPs. Give AR an equal-FLOP best-of-M budget via the exp3727
  harness. NFE is the cleaner inner-loop unit (arXiv:2511.05562); FLOPs as cross-check
  (arXiv:2408.03314). A win counts ONLY at equal total inference compute.

**An honest NEGATIVE at either part is a real finding** that bounds the route cheaply
— as valuable as a positive. The kill-gate makes this a *bounded* bet, not a
P0.1-style grind.

---

## Routing rationale (current backend reality)

- **gemini still CRASHES real GPU workloads** (exp3703/exp3714 history) → the
  cheap-default for REAL GPU/CPU mechanical work is **codex** (`requires_codex`),
  not gemini.
- The two open-ended training-debug tasks (**exp3745** fix+diagnose+train,
  **exp3746** resume+stabilize) are **claude + opus** (`requires_claude`): diagnosing
  why CUDA was unavailable in-subprocess, the venv-pinning + hard-block fix, and
  EBT divergence stabilization are multi-file choreography + open-ended judgment
  under ambiguity (standing operator directive 2026-06-02). The most likely real
  failure mode lives here.
- The GPU EVAL tasks (**exp3748** generation smoke, **exp3749** matched-compute) are
  **codex + requires_codex + gpu**: running the trained models through the
  already-built + unit-tested exp3727 harness is mechanical with a deterministic
  FLOP-accounting criterion.
- Everything else (archive, corrigendum, verdicts, FR-11, KV260, capstone) is
  **codex + requires_codex** (aggregation / CPU).

## Gating discipline

- `gated_on` is used ONLY on the expensive GPU tasks (exp3746/3748/3749) against
  **BARE-value** upstream fields (per `feedback_gated_fields_must_be_bare` — a
  `{value, principle}` dict breaks the gate). Every gated upstream field
  (`cumulative_steps_trained`, `green_light_343`, `ebt_can_generate`) is emitted as a
  bare scalar.
- Verdict / aggregation / self-learning / hardware / capstone tasks carry NO
  `gated_on` and read upstream with a graceful disk-presence fallback (the `.340`
  proven-safe pattern: read if present, else record an honest "not-run", never crash
  on a None read).
- `prior_failures` blocks (all four sub-fields) on every task whose scope matches a
  prior failed/blocked attempt (exp3728/3734/3735 for training; exp3729/3736 for the
  part-a verdict; exp3729 for part-b). `operator_override` on the routine
  archive/capstone/KV260/FR-11-lineage tasks.

## Hardware requirements

- **2× RTX 3090** (CUDA, the training + eval substrate). Verified idle and CUDA-live
  via `.venv/bin/python` on 2026-06-02. The tiny EBT fits one 3090 at ~1.3 GB.
- **KV260** opportunistic terminal-state confirm (SSH-only, per KV260
  SSH-Not-SD-Card discipline). Terminal since `.340`.

## New references folded in (2026-06-02)

arXiv:2408.03314 (FLOP-matched test-time compute, the methodology template),
arXiv:2511.05562 (IterRef — NFE fair-compute accounting), arXiv:2504.01005
(when-to-solve-vs-verify, compute-matched), arXiv:2505.14999 (EORM — small energy
reranker, the baseline to beat), arXiv:2603.12248 (energy-based fine-tuning),
arXiv:2307.01668 (diffusion contrastive divergence — fallback negative-sampler),
arXiv:2510.08554 (DCoLT diffusion GSM8K — direction not magnitude). See
research-references.md `.343 additions`.

## Success criteria for `.343`

1. The EBT genuinely trains on the GPU (`cumulative_steps_trained > 0` with
   `cuda:true`), and a REAL part-(a) verdict (exp3747) supersedes the
   "untested" exp3736 — **stable→green-light, or a genuine divergence→bounded.**
   Either is a real result; another infra-block is the only outcome that counts as a
   failure of this milestone.
2. If part-(a) green-lights: a matched-COMPUTE part-(b) verdict (exp3750) with the
   honest EBT-vs-AR delta at equal FLOPs, n≥100.
3. `paper_ready` stays TRUE; frozen 0.9131 unchanged; P0.1 stays settled-bounded.
