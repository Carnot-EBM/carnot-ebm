# Research Roadmap — Milestone 2026.06.364

**Planned:** 2026-06-08 (outer-loop planning agent, Claude Opus 4.8)
**Milestone doc for:** `research-roadmap-next.yaml` (`milestone: 2026.06.364`)
**Prior milestone:** 2026.06.363

---

## 0. One-line thesis

**.363 set up the load-bearing "does the verifier earn its place?" proof but
failed to LAND it. .364 lands it.** The competent GenRM/ThinkPRM judge, the
valid efficiency head-to-head, the non-degenerate cascade, and the
independent-corpus moat replication were all BLOCKED in .363 by two concrete,
now-understood failures. .364 executes and validates the work that is already
drafted on disk, fixes the two blockers, and finally answers the operator's
question against a *competent* comparator.

This is the highest-leverage milestone available: per `ops/north-star.md` §5,
the verifier is Carnot's entire surviving value-add, and whether it "earns its
place" is currently `INCONCLUSIVE` (capstone .363: `efficiency=INCONCLUSIVE,
moat_replicated=false, earns=false`). Every other research direction is
downstream of this answer.

---

## 1. What the previous milestone (.363) proved — and where it stalled

### Landed (genuine)
- **exp3929 — first ARC-AGI-3 agentic run:** verifier-as-router HELPS,
  action-efficiency ratio **1.959** (CI95 [1.742, 2.194]) on a synthetic
  ARC-AGI-3-style env; real-benchmark reachability preflight = true. The first
  datum for the agentic-proof venue (north-star §5 Phase-4).
- **exp3930 — FR-11 v26 self-learning:** invariant HELD across v25→v26
  (AUROC 0.908, memory-ablation contribution +0.0185, frozen 0.9131
  unchanged). The self-learning mandate continues to hold.
- **exp3924 — facts route retired:** the graph-grounding fact-verifier scope
  (exp3920) blocked/fabricated a 4th consecutive time and was retired to
  honest future-work in `ops/exclusion_manifest.yaml`. The verifier stays
  **math/step-error domain-bound** — an honest scoping, not a regression.

### Stalled (the .364 work-list)
The four load-bearing science tasks all BLOCKED:

| Exp (.363) | Verdict | Root cause (verified on disk) |
|---|---|---|
| exp3925 competent judge build | **no artifact landed** | Module (`competent_llm_judge.py`), test, and runner script were all WRITTEN (Jun 7) but the experiment artifact never landed and never reached the conductor log — the classic **max-turns-exhaustion / bootstrap-and-bail** failure: one task tried to diagnose + build + unit-test + run a live 35B judge in a single budget. |
| exp3926 valid efficiency | `blocked_upstream_competent_judge_not_ready` | Disk-read of the missing exp3925 artifact failed → correctly self-blocked (the .363 disk-fallback discipline working as designed: a missing upstream costs ONE task, not the chain). |
| exp3927 non-degenerate cascade | `blocked_upstream_valid_efficiency_missing` | Cascaded from exp3926. |
| exp3928 moat replication (independent corpus) | `blocked_all_gguf_inference_failed` | `AttributeError("'ExperimentConfig' object has no attribute 'max_tokens_weak'")` raised inside `load_robust_generator` (`moat_scissor_accuracy_3916.py`). The field has since been added to `moat_scissor_replication_3928.py:ExperimentConfig` (lines 161-162) but the experiment never re-ran to confirm. |

**Crucially, GGUF inference itself is sound.** exp3915 (robust harness) is
READY, and exp3916/exp3917 ran live 400-490s GPU inference in .362. The
blockers are an unexecuted task and a one-line config mismatch — NOT an infra
wash. This is why .364 is an execute/validate/fix milestone, not a rebuild.

---

## 2. The three biggest gaps between current state and the PRD vision

1. **The verifier's value is UNPROVEN against a competent competitor.**
   north-star §5 makes this existential: with the generator now commodity
   (open LLM + TRM refiner), the energy verifier IS Carnot. The accuracy moat
   landed in-distribution (exp3916 MOAT_SURVIVES) but (a) was never replicated
   on an independent corpus, and (b) the efficiency comparison used a
   below-chance strawman judge. **Gap: a credible, competent-judge efficiency
   result + an independent-corpus accuracy result.** This milestone closes it.

2. **No deployable "matched accuracy at Nx cheaper" artifact.** The Meta-EBM
   Cascade Router is the deployable form of the efficiency win, but the .362/.363
   cascade was degenerate (escalation_fraction = 0.0 — it never escalated).
   **Gap: a non-degenerate cascade with a measured cost ratio.**

3. **The verifier's domain boundary is asserted, not mapped.** north-star §5
   says math is strong, facts earned-negative, code weak — but only math/facts
   were measured. Per the now-RETIRED Depth-Over-Breadth forcing function
   (its retirement condition is met: P0.1 answered, G2 closed, paper_ready=true),
   breadth toward NEW directions is explicitly invited. **Gap: a disciplined
   cross-domain discriminating-value map** (where does the moat hold?). One
   such probe is included, honestly framed so a code-domain negative is an
   informative boundary, not a failure.

---

## 3. Architecture / dependency graph

```
PHASE 0 — transition + green-gate
  exp3934  archive .363 -> activate .364; record the .363 unblock-state;
           green-gate (yaml parses, core pretest green, .363 modules import)
     |
     v
PHASE 1 — LAND THE OFFLINE VERIFIER PROOF (the .363 science that blocked)
  exp3935  RUN + VALIDATE the competent GenRM/ThinkPRM judge   [GPU, live]
           (module/test/runner already drafted in .363; EXECUTE them;
            fixture AUROC > 0.65 positive control)
     | (disk-read; missing upstream costs ONE task, no hard gate)
     +--> exp3936  VALID efficiency head-to-head: energy vs competent judge  [GPU, live]
     |         | (reuse exp3936 per-item judge scores — no re-inference)
     |         +--> exp3937  NON-DEGENERATE cascade router (escalation > 0)
     |
  exp3938  MOAT replication on an INDEPENDENT corpus (ProcessBench /        [GPU, live]
           fover_test_v4) — FIX the max_tokens_weak AttributeError, then run
     |
     v
PHASE 2 — AGENTIC PROOF VENUE (sequenced second, per north-star §5)
  exp3939  ARC-AGI-3 agentic step 2: richer env + energy-router vs a
           learned-value baseline + real-benchmark access preflight   [GPU-free]
     |
     v
PHASE 3 — SELF-LEARNING MANDATE + HARDWARE + BREADTH
  exp3940  FR-11 v27 self-learning: online-learn the exp3937 cascade band;  [CPU]
           confirm +0.0185 / frozen 0.9131 invariant
  exp3941  hardware continuity clean: GateMate terminal re-confirm +        [FPGA/SSH]
           PolarFire/KV260 opportunistic (distinct timers, no fabric claim)
  exp3942  cross-domain verifier discriminating-value MAP (NEW direction):  [GPU, live]
           where does the energy moat hold vs the competent judge?
     |
     v
PHASE 4 — synthesis + capstone
  exp3943  literature synthesis / study update (no new inference)
  exp3944  Capstone .364 — verifier scorecard, efficiency axis FINALLY
           landed against a competent judge; flip verifier_earns_its_place
           honestly; paper_ready stays TRUE, frozen 0.9131 unchanged
```

**Critical path:** exp3934 → exp3935 → exp3936 → exp3937 → exp3944.
**Independent:** exp3938 (own harness), exp3939, exp3940, exp3941, exp3942.

---

## 4. Why this is NOT churn (north-star §1 self-check)

Every .364 task either advances the headline/win-condition or closes a G-gate
or lands a load-bearing-unproven link:

- exp3935-3938 are NOT `vN+1` re-measurements — they answer questions .363
  blocked on (a competent-judge efficiency number; an independent-corpus moat
  number; a non-degenerate cascade). Each is a *blocked* task being unblocked,
  not a re-run of an answered question.
- exp3939 advances the agentic-proof venue with a NEW baseline (learned-value
  ablation) the synthetic exp3929 lacked.
- exp3940 is the standing self-learning MANDATE (research-program.md), now
  tied to the new cascade band.
- exp3942 is a NEW direction (cross-domain map) explicitly invited by
  north-star §5 now that Depth-Over-Breadth has retired.

Re-measurement-for-its-own-sake (`vN+1 because vN exists`) is excluded.

---

## 5. Hardware requirements

- **2× RTX 3090 (CUDA):** exp3935, 3936, 3938, 3942 (live GGUF judge +
  generator inference). Use the robust exp3915 harness (`gguf_inference.py`),
  never `llama_cpp.Llama` directly; `.gguf` path + embedded tokenizer (never
  `AutoTokenizer` on a GGUF repo id).
- **CPU:** exp3937 (cascade reuses cached judge scores), exp3940 (FR-11
  counter updates), exp3934/3943/3944 (aggregation).
- **FPGA / SSH:** exp3941 — GateMate (`openFPGALoader -c dirtyJtag --detect`),
  PolarFire (`ssh polarfire`), KV260 (`ssh kria` + `xmutil`; NEVER host
  `/dev/mmcblk*`). `hardware_smoke` substrate; opportunistic, do not block the
  milestone on board reachability.

---

## 6. SOTA models (per CLAUDE.md)

Live-model tasks prefer (via the robust harness fallback order):
`unsloth/Qwen3.6-35B-A3B-GGUF` → `unsloth/gemma-4-31B-it-GGUF` →
`unsloth/gemma-4-26B-A4B-it-GGUF`. The .362 26B-0-shot judge underperformed
(AUROC 0.4423); exp3935 prefers a stronger model at reduced `n_gpu_layers`.
GGUF tokenizer rule honored (`.gguf` path, llama.cpp `vocab_only` preflight).

---

## 7. Invariants (carried forward)

- `paper_ready` stays **TRUE** (G1-G4 met); this milestone adds credibility
  lenses, not a new headline.
- FoVer **0.9131** frozen; never moved; never aggregate `flagged_adversarial`.
- Energy-as-generator is **closed-negative** — NO generator experiments.
- Verifier is **math/step-error domain-bound** (facts retired in .363); the
  cross-domain probe (exp3942) MAPS the boundary, it does not re-open facts.
- No external publication (operator-only).
- **Routing:** all tasks `codex` + `requires_codex` + `gpt-5.5` (anti-wipeout;
  gemini crashes GPU workloads and 429-wiped .333/.355; standing operator
  gemini↔codex flip authority 2026-06-05). Live-model Run commands use
  `{project_root}/.venv/bin/python`.
- **Hardening carried from .356-.363:** no unit-test wall-clock floor on
  fixtures; robust live-model path via the exp3915 harness; positive control
  on every comparison; NO hard `gated_on` on the critical path (disk-read
  fallback — a missing upstream costs ONE task); BARE field emission.
