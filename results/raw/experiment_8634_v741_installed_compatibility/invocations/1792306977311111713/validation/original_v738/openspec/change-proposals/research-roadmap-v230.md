# Research Roadmap — Milestone 2026.05.230

**Prepared:** 2026-05-18
**Milestone:** 2026.05.230
**Experiment IDs:** exp2350 – exp2363
**Previous milestone:** 2026.05.229 (3/14 criteria met; pre-test cascade failed 10th time but all 3
targeted tests confirmed passing on direct inspection; Semantic Energy Tier 0g prototype AUROC=1.0)

---

## What Milestone .229 Proved

Milestone .229 produced a 14-task roadmap (exp2336–exp2349) and completed 3 tasks:

- **exp2336 (archive):** Ran with `archive_ready: false`. Precondition found "2026.05.229" in
  research-roadmap.yaml instead of expected "2026.05.228". Root cause: conductor activates the
  new milestone by swapping research-roadmap.yaml BEFORE the archive task runs.
  Fix for .230: check for "2026.05.229" (the now-current milestone) not "2026.05.228".
- **exp2337 (pre-test fix v10, requires_claude:true):** Failed for the 10th consecutive time.
  Three 1201-second timeouts produced no deliverable. Root cause: agents run the FULL test
  suite (with coverage, 21k+ tests) which takes many minutes, triggering the 1201s timeout
  before writing any artifact. The 3 targeted fixes (potts artifact + 2 xdist markers) take
  < 1 minute to apply, but agents timed out before completing.
  **CRITICAL FINDING POST-.229:** Direct inspection confirms all 3 targeted tests NOW PASS.
  The `results/experiment_1692_potts_export.json` artifact exists with `"status": "success"`.
  Tests B and C pass under xdist. The pre-test cascade is RESOLVED at the test level.
- **exp2338 (Semantic Energy Tier 0g, UNGATED):** SUCCESS. AUROC=1.0 on 100-example synthetic
  corpus. 3 unit tests passed. Module landed at `python/carnot/verify/semantic_energy.py`.
  The ungated structural innovation prevented a 9th consecutive empty-experiment milestone.
- **exp2339–exp2347 (Phase 2–3, gated on pretest_fixed):** All GATE_BLOCK.
- **exp2348 (capstone):** GATE_BLOCK (upstream gates never set).
- **exp2349 (retro):** Ran; confirmed 2 research tasks + retro = 3/14.

**STRUCTURAL FINDING:** The 10 consecutive pre-test cascade failures were NOT because the
tests are broken — they are because agents timed out running the full suite instead of making
the 3 targeted fixes and then running only those 3 tests. The tests pass. The cascade was
a PROCESS failure, not a code failure.

---

## Three Biggest Gaps vs PRD Vision

### Gap 1: NSVIF neuro-symbolic extraction never ran (PRD Priority #1 since 2026-04-11)

`research-program.md` lists "Rebuild constraint extraction for real models" as the HIGHEST
PRIORITY. NSVIF (arXiv:2601.17789) — Z3-based neuro-symbolic replacement for the regex
ArithmeticExtractor — has been proposed for 5 consecutive milestones (.225–.229) but has never
executed. Its gated_on predecessor (pretest_fixed) has now been confirmed passing on direct
test.

**Fix for .230:** exp2352 (NSVIF v5) is UNGATED — no gated_on field. It includes its own
precondition check (z3 imports). This is the same pattern that made exp2338 succeed in .229.

### Gap 2: All gated experiments still blocked despite pre-test cascade being resolved

10 research experiments (FST live gen, FR-11, KAN-CL, VERGE, Eidoku, Projected-Langevin,
KV260, ML-Ising init) have been blocked for 5–10 consecutive milestones by a cascade from
pretest_fixed. All are now structurally unblocked.

**Fix for .230:** ALL research experiments (exp2351–exp2361) are UNGATED. Each includes its
own precondition check (module import, z3 availability, GGUF cache). No experiment gates on
exp2350's archive_ready or any cascade proxy.

### Gap 3: Semantic Energy (exp2338) validated on synthetic only — needs real LLM logits

exp2338 achieved AUROC=1.0 on synthetic logit arrays (Normal(0,0.5) vs Normal(0,2.0)).
This validates the formula but not its discriminative power on real LLM logit distributions.
Paper-v6 citation requires validation on actual model outputs.

**Fix for .230:** exp2351 (Semantic Energy Real LLM) validates exp2338's prototype using
actual penultimate layer logits from a cached SOTA GGUF model (requires own GGUF precondition
check, no dependency on pretest_fixed).

---

## New Research from Post-.229 Arxiv Sweep

Four new papers added to research-references.md:

1. **EBM-CoT** (arXiv:2511.07124): EBM calibration for implicit CoT via Langevin dynamics.
   Directly implements Carnot's Phase-4 free-energy verifier hypothesis. Candidate for exp2358.
2. **DiffuTruth** (arXiv:2602.11364): Hallucination detection via diffusion model likelihoods
   (energy of reconstructed vs. original text). Strong peer for paper-v6. Candidate Tier 0h.
3. **Self-Adaptive Ising** (arXiv:2501.04971): Lagrange relaxation to reshape Ising energy
   landscape for constrained optimization. Advances the hardware constraint-satisfaction track.
   Candidate for exp2359.
4. **VeriCoT** (arXiv:2511.04662): Neuro-symbolic CoT validation via logical consistency checks.
   46% improvement in CoT verification. Candidate k=17 verifier class alongside Semantic Energy.

---

## Architecture Snapshot (2026-05-18)

```
                    [ LLM Inference ]
                    (SOTA GGUF: Qwen3.6-35B, Gemma4-31B, Gemma4-26B)
                           |
                    [ ODAR Router ]         <- arXiv:2602.23681
                    (fast / deliberative)
                    /              \
          [ Fast Path ]      [ Deliberative Path ]
          SpilledEnergy       FST context prep
          SemanticEntropy     CASAL constraint enforcement
          SemEnergy(0g)  <-- exp2338 DONE (AUROC=1.0 synthetic)
                              + Real LLM validation <- exp2351 .230
                              + Freq-Aware Attn(0f)
                           |
                    [ Verifier Cascade ]
                    Tier 0g: Semantic Energy (exp2338 DONE)
                    Tier 1:  Ising (Gibbs, KAN) [WORKING]
                    Tier 2:  NSVIF + Z3        <- exp2352 .230 (UNGATED)
                    Tier 2.5: VERGE repair     <- exp2353 .230 (UNGATED)
                    Tier 2.8: Eidoku CSP       <- exp2354 .230 (UNGATED)
                    Tier 3:  Projected-Langevin <- exp2355 .230 (UNGATED)
                           |
                    [ KAN-CL continual learning ]
                    n=256 per-knot importance   <- exp2356 .230 (UNGATED)
                           |
                    [ FST Self-Learning (FR-11) ]
                    Fast-slow weights           <- exp2357 .230 (UNGATED, CSL)
                           |
                    [ Hardware (KV260 RTL) ]    <- exp2360 .230 (UNGATED)
                           |
                    [ Output / Repair ]
```

---

## Phase Structure

### Phase 0: Transition (1 task)
Archive .229 milestone to research-complete.yaml.
Key fix: precondition checks for "2026.05.229" (not ".228") since conductor activates
new milestones by swapping research-roadmap.yaml before any tasks run.

### Phase 1: Semantic Energy Real LLM + New Research (2 tasks)
- exp2351: Semantic Energy real LLM validation — validates exp2338 prototype on actual logits
- exp2358: EBM-CoT integration prototype — implements arXiv:2511.07124 Langevin-guided CoT
  calibration alongside Carnot's Phase-4 active-inference track

Both UNGATED. exp2351 requires GGUF cache (own precondition). exp2358 uses synthetic only.

### Phase 2: Constraint Extraction (2 tasks, PRD Priority #1)
- exp2352: NSVIF Z3 extractor v5 (arXiv:2601.17789) — first actual execution after 5 blocked
- exp2353: VERGE SMT repair v5 (arXiv:2601.20055) — first actual execution after 5 blocked
Both UNGATED. Own precondition: z3 imports.

### Phase 3: Verifier Extensions (3 tasks)
- exp2354: Eidoku CSP v6 (arXiv:2512.20664) — smooth-falsehood detection via CSP
- exp2355: Projected-Langevin v6 (arXiv:2605.05387) — alternative to CASAL
- exp2359: Self-Adaptive Ising (arXiv:2501.04971) — Lagrangian constraint shaping
All UNGATED.

### Phase 4: Continual Learning (2 tasks, includes FR-11 mandatory)
- exp2356: KAN-CL n=256 v9 (arXiv:2605.12306) — per-knot importance, 5th blocked attempt
- exp2357: FR-11 FST multi-domain v7 — mandatory continuous self-learning experiment

Both UNGATED. Own preconditions on substrate_kan and fast_slow imports.

### Phase 5: Hardware (1 task)
- exp2360: KV260 RTL lint v9 — Verilator + Icarus on parallel Ising RTL
UNGATED. Own precondition: verilator + iverilog available.

### Phase 6: Live Generation (2 tasks)
- exp2361: FST live gen v10 — first actual run after 10 blocked attempts; GGUF precondition
- exp2362: Capstone .230 (opus, gated on exp2361.fst_live_validated + exp2356.kancl_n256_validated)

exp2361 UNGATED. exp2362 gated on research results only (NOT on pretest_fixed or archive_ready).

### Phase 7: Retro (1 task)
- exp2363: Retro v230 — UNGATED, always runs last

---

## Dependency Graph

```
exp2350 (archive)   ── ungated ──
                                 ↓
exp2351 (sem-energy-real) ── ungated, own GGUF precondition ──→ result
exp2352 (nsvif-z3)        ── ungated, own z3 precondition  ──→ result
exp2353 (verge-smt)       ── ungated, own z3 precondition  ──→ result
exp2354 (eidoku-csp)      ── ungated                       ──→ result
exp2355 (proj-langevin)   ── ungated                       ──→ result
exp2356 (kancl-n256)      ── ungated, own kan precondition ──→ kancl_n256_validated
exp2357 (fr11-fst)        ── ungated, own fst precondition ──→ fr11_multidomain_passed
exp2358 (ebm-cot)         ── ungated                       ──→ result
exp2359 (self-adaptive-ising) ── ungated                   ──→ result
exp2360 (kv260-rtl)       ── ungated, verilator precondition ──→ result
exp2361 (fst-live-gen)    ── ungated, GGUF precondition   ──→ fst_live_validated
                                                                ↓              ↓
exp2362 (capstone)        ─── gated on fst_live_validated + kancl_n256_validated ───→
                                                                               ↓
exp2363 (retro)           ── ungated, always last ──────────────────────────────→
```

All 12 research experiments (exp2351–exp2361) are UNGATED. Even if exp2350 fails again,
all research experiments proceed.

---

## FR-11 Continuous Self-Learning (Mandatory)

exp2357 (FR-11 FST multi-domain v7) satisfies the mandatory CSL requirement:
- `continuous_self_learning_task: true` in the artifact contract
- Gate: `cross_domain_retention_rate >= 0.75`
- Scope: 3 sequential domains (arithmetic, code constraints, logic propositions)
- Validates FST fast-weight updates generalize across domains

---

## Hardware Requirements

- **Experiments needing GGUF:** exp2351, exp2361, exp2362
  - Precondition: `ls ~/.cache/huggingface/hub/ | grep -i "gemma-4-26B\|Qwen3.6-35B\|gemma-4-31B"`
  - These experiments are ungated but self-block if the model is not cached
- **Experiments needing Z3:** exp2352, exp2353
  - Precondition: `python -c "import z3"` — auto-install if missing
- **Experiments needing Verilator + Icarus:** exp2360
  - Precondition: `command -v verilator && command -v iverilog`
- **CPU-only experiments:** exp2353–exp2360 (no GPU required)
- **GPU for capstone:** exp2362 (via GGUF inference, precondition checks first)

---

## Decentralization Check (CLAUDE.md Rules 1–7)

Rule 1 (local-first): All GGUF experiments use cached local models (no closed API calls). ✓
Rule 2 (closed models optional): All experiments have local fallbacks or synthetic paths. ✓
Rule 3 (mirroring): No new weights generated this milestone; distribution not applicable. ✓
Rule 4 (multiple surfaces): Verifier cascade exposes CLI + MCP + API. ✓
Rule 5 (hardware portability): KV260 RTL (exp2360) advances FPGA portability track. ✓
Rule 6 (data minimization): No closed-weight calls in this milestone. ✓
Rule 7 (no vendor core deps): All new modules in python/carnot/verify/ use open libraries. ✓

---

## Exclusion Manifest Cross-Check

Checked `ops/exclusion_manifest.yaml` before planning. None of the proposed tasks match
retired experiment scopes:
- GRPO/VPRM lineage: NOT proposed (retired_milestone: 2026.04.112) ✓
- WOPR puzzle cartridges: NOT proposed (retired_milestone: 2026.04.112) ✓
- HardNet++/DSP: NOT proposed (retired_milestone: 2026.04.112) ✓
- THRML scaling sweep: NOT proposed (retired_milestone: 2026.05.120) ✓
- SpecAnn: NOT proposed (retired_milestone: 2026.05.120) ✓
- exp2091 (gemini CLI): NOT proposed ✓
- iCE40 PIMI: NOT proposed ✓

NSVIF/VERGE/Eidoku/Projected-Langevin/KAN-CL/FST-live-gen all have prior_failures blocks
documenting 5–10 consecutive `blocked_gate_check_failed` verdicts (not real execution failures).
These are not scope-failed experiments — they are structurally blocked experiments that have
never actually run. The .230 structural change (UNGATED) removes the structural block.

---

## Failed-Experiment Rerun Compliance

All repeated experiments document prior failures in the YAML `prior_failures:` field:

| Task | Prior Failures | Root Cause Named | What Changed |
|------|---------------|-----------------|--------------|
| exp2352 (NSVIF v5) | exp2342 (blocked_gate_check_failed) | Gate on pretest_fixed | Ungated in .230 |
| exp2353 (VERGE v5) | exp2343 (blocked_gate_check_failed) | Gate on pretest_fixed | Ungated in .230 |
| exp2354 (Eidoku v6) | exp2344 (blocked_gate_check_failed) | Gate on pretest_fixed | Ungated in .230 |
| exp2355 (Proj-Langevin v6) | exp2345 (blocked_gate_check_failed) | Gate on pretest_fixed | Ungated in .230 |
| exp2356 (KAN-CL v9) | exp2341 (blocked_gate_check_failed) | Gate on pretest_fixed | Ungated in .230 |
| exp2357 (FR-11 v7) | exp2340 (blocked_gate_check_failed) | Gate on pretest_fixed | Ungated in .230 |
| exp2360 (KV260 v9) | exp2346 (blocked_gate_check_failed) | Gate on pretest_fixed | Ungated in .230 |
| exp2361 (FST live v10) | exp2339 (blocked_gate_check_failed x10) | Gate on pretest_fixed | Ungated in .230 |

All `retire_if_same_verdict: false` because the verdict was `blocked_gate_check_failed`,
not an execution failure. The experiments never ran. Removing the gate is the structural fix.
