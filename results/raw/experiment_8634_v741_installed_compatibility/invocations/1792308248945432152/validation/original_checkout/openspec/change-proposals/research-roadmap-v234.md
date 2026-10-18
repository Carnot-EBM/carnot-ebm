# Research Roadmap v234: AUROC Ceiling Assault + Phase 1 Ship Gate

**Milestone:** 2026.05.234  
**Title:** AUROC Ceiling Assault + Phase 1 Ship Gate: Full-Ensemble v3, Hierarchical LogCons, KV260 Yosys v3, FR-11 v3  
**Planned:** 2026-05-18  
**Previous milestone:** 2026.05.233 (Codex Recovery Sprint)

---

## What Milestone .233 Proved

Milestone .233 was the Codex Recovery Sprint after .232's catastrophic Codex CLI failure (11/14 tasks FAIL with identical truncated-message error traced to a transient OpenAI backend window on 2026-05-18).

**Successes in .233:**
- Codex CLI infrastructure confirmed healthy by exp2393 (root cause: transient OpenAI backend, not a structural conductor bug)
- FregeLogic Z3+Neural hybrid achieved AUROC=0.8831 — **beat HalluScan 0.88 baseline** (first time Carnot exceeds this anchor)
- HALT Tier 0j achieved AUROC=0.8539 (significant lift over SemanticEnergy 0.685 baseline)
- HIVE 3-verifier ensemble achieved AUROC=0.8539 (3 verifiers fused; not all 4 Tier 0 verifiers included)
- FST live inference validated via PATH A GGUF (first actual live GGUF inference in the project)
- Freq-Aware Attn AUROC=0.7045 (modest improvement over baseline)
- Typed CoT AUROC=0.508 (below baseline — Curry-Howard type-checking does not discriminate on this corpus)

**Missing from .233 (no artifact produced — session ended before these tasks ran):**
- FR-11 NSVIF Online (exp2400) — mandatory continuous self-learning task
- KV260 Yosys synthesis (exp2401) — next hardware milestone
- Kinetic Langevin vs CASAL (exp2402) — sampler comparison
- Phase 1 Ship Gate audit (exp2403) — Phase 1 completion gate
- .233 Retro (exp2405)

---

## The 3 Biggest Gaps (vs PRD Vision)

### Gap 1: AUROC ceiling at 0.8831, HIVE peer at 0.9236

FregeLogic beat HalluScan (0.88) but the HIVE peer system (arXiv:2604.26139) achieves 0.9236.
The gap is 0.04. Key interventions:
- Full 4-verifier HIVE ensemble (0f freq-aware + 0g semantic energy + 0h LaaB + 0j HALT): .233's ensemble only had 3 verifiers
- Hierarchical Alignment LogCons (arXiv:2604.09075): extends FregeLogic's Z3 tiebreaker to instruction hierarchies; 91% adherence vs 67% baseline in peer experiments
- HALT-RAG calibrated NLI ensemble (arXiv:2509.07475): adds precision-calibration via selective abstention

### Gap 2: Phase 1 ship gate not met

Per CLAUDE.md Phase Vision, Phase 1 ships when:
1. PyPI package `carnot-ebm` published
2. HuggingFace mirror at huggingface.co/Carnot-EBM
3. MCP server documentation
4. CLI documentation
5. At least one external reproducer

None of these have been confirmed checked in .231-.233. The Phase 1 ship gate audit (exp2403 equivalent) never produced an artifact.

### Gap 3: FR-11 continuous self-learning mandatory but never completed

The FR-11 functional requirement (autonomous self-learning loop) must be satisfied in every milestone. The NSVIF online learning approach was designed across 3 milestones but never actually executed (always blocked by infrastructure issues or session timeouts). .234 removes all blockers: Codex CLI healthy, pre-test cascade resolved, no gating dependencies.

---

## Architecture Snapshot (as of .233 close)

```
Tier 0 Verifiers (training-free, cached telemetry):
  0b  Boolean energy (Ising)         [legacy]
  0f  Freq-Aware Attn                AUROC=0.704  [new .233]
  0g  Semantic Energy (Boltzmann)    AUROC=0.685  [exp2351]
  0h  LaaB logical consistency       AUROC=0.7xx  [exp2368]
  0i  SpilledEnergy k=18             AUROC=0.7xx  [exp2369]
  0j  HALT latent probe              AUROC=0.854  [exp2394]
  0k  FregeLogic Z3+Neural           AUROC=0.883  [exp2395] ← NEW BEST
  [ensemble] HIVE (3 verifiers)      AUROC=0.854  [exp2398]
  Peer ceiling: HIVE (arXiv:2604.26139) AUROC=0.9236

Tier 1 Verifiers (symbolic / formal):
  NSVIF Z3 extractor (exp2352, exp2366) — active
  Eidoku CSP (exp2354) — active
  FregeLogic hybrid (exp2395) — active
  VERGE SMT repair (exp2353) — active

Tier 2 MCMC Samplers:
  CASAL — baseline
  Projected-Langevin — KL delta +0.333 [exp2355]
  Kinetic Langevin — NOT YET BENCHMARKED [carry-forward]
  Dikin-Langevin — NOT YET IMPLEMENTED [new .234]
  DE-PSGLD — NOT YET IMPLEMENTED [new .234]

Continual Learning:
  KAN-CL n=256 [exp2356] — active
  FR-11 NSVIF Online — NOT COMPLETED [carry-forward mandatory]

Hardware:
  KV260 FPGA: RTL lint-clean [exp2372] → Yosys synthesis PENDING
  ROCm gfx1150: available; dual RTX 3090: available

Phase 1 ship gate: NOT YET CHECKED
```

---

## Phase Descriptions

### Phase 0 — Archive + Retro (2 tasks)
Archive .233 to research-complete.yaml, write the missing .233 retro, activate .234.

### Phase 1 — AUROC Ceiling Assault (3 tasks)
Push AUROC beyond HIVE peer 0.9236 with three complementary approaches:
- Full 4-verifier HIVE ensemble (all Tier 0 verifiers properly fused, learned soft-vote weights)
- Hierarchical Alignment LogCons (arXiv:2604.09075) — Z3 partial-order instruction checker, 91% adherence baseline
- HALT-RAG NLI calibrated ensemble (arXiv:2509.07475) — abstention-based precision improvement

### Phase 2 — FR-11 + FST Advancement (2 tasks)
- FR-11 NSVIF Online v3 (mandatory continuous self-learning task)
- FST Constrained MCMC Generation (arXiv:2506.05754) — apply unified MCMC to FST pipeline now that PATH A works

### Phase 3 — Hardware + Samplers (4 tasks)
- KV260 Yosys v3 (carry-forward; lint-clean RTL from exp2372)
- Kinetic Langevin v3 (carry-forward; BAOAB splitting)
- Dikin-Langevin sampler (arXiv:2510.04582) — NEW, polyhedral constraints with formal convergence
- DE-PSGLD sampler (arXiv:2605.00723) — NEW, decentralized proximal SGLD

### Phase 4 — Phase 1 Ship Gate (1 task)
Formal audit of all 5 Phase 1 completion criteria (PyPI, HF mirror, CLI docs, MCP docs, external reproducer).

### Phase 5 — Synthesis (2 tasks)
Paper-v6 capstone + milestone retro.

---

## Dependency Graph

```
exp2406 (archive)
  ↓ (after archive)
exp2407 (retro .233) — ungated, always runs
exp2408 (HIVE 4-verifier v3) — ungated, reads .233 Tier 0 verifier artifacts
exp2409 (Hierarchical LogCons) — ungated, new scope
exp2410 (HALT-RAG NLI) — ungated, new scope
exp2411 (FR-11 NSVIF v3) — ungated, mandatory
exp2412 (FST Constrained MCMC) — ungated, requires FST PATH A from exp2399
exp2413 (KV260 Yosys v3) — ungated, requires lint-clean RTL from exp2372
exp2414 (Kinetic Langevin v3) — ungated, requires scipy
exp2415 (Dikin-Langevin) — ungated, new scope
exp2416 (DE-PSGLD) — ungated, new scope
exp2417 (Phase 1 ship gate v3) — ungated, checks live URLs
exp2418 (capstone) — gated: exp2408.ensemble_auroc_improved==true OR exp2409.logcons_auroc>=0.85
exp2419 (retro .234) — ungated, always runs
```

---

## Hardware Requirements

| Task | Hardware | Notes |
|------|----------|-------|
| exp2408-2410 (AUROC) | CPU | sklearn, scipy; no GPU needed |
| exp2411 (FR-11) | CPU | z3-solver only |
| exp2412 (FST MCMC) | CPU + optional GGUF | PATH A uses llama_cpp |
| exp2413 (KV260 Yosys) | CPU | Yosys binary required |
| exp2414-2416 (samplers) | CPU | numpy, scipy |
| exp2417 (ship gate) | Network | curl to PyPI, HF APIs |
| exp2418 (capstone) | CPU | artifact synthesis only |

---

## Decentralization Check (Rules 1–7, CLAUDE.md)

1. Local-first open models: all verifiers tested on cached GGUF-generated telemetry ✓
2. Closed-weight integration optional: no task requires closed-weight models ✓
3. Distribution mirroring: exp2417 audits HF mirror ✓
4. Multiple integration surfaces: FR-11, FST, MCP all in scope ✓
5. Hardware portability: KV260 Yosys advances FPGA track ✓
6. Data minimization: no closed-weight calls ✓
7. No vendor abstractions in core: all new verifiers live in carnot.verify.* ✓

---

## Exclusion Manifest Cross-Check

Checked against `ops/exclusion_manifest.yaml` — no proposed task matches any retired experiment scope:
- GRPO/VPRM lineage — not proposed ✓
- WOPR puzzle cartridges — not proposed ✓
- HardNet++/DSP repair stack — not proposed ✓
- THRML scaling sweep — not proposed ✓
- SpecAnn Phase 3 sampler — not proposed ✓
- exp2091 (gemini CLI bail-out) — not proposed ✓
- iCE40 PIMI — not proposed ✓
- HalluSAEGeometricProbe — not proposed ✓
- Discriminative JEPA — not proposed ✓

All 14 tasks are either genuinely new scope or documented carry-forwards with prior_failures blocks.

---

## Failed-Experiment Rerun Compliance

| Task | Prior Failures | Root Cause Named | What's Different | Retire Gate |
|------|---------------|-----------------|-----------------|-------------|
| exp2408 HIVE v3 | exp2398 (3-verifier only), exp2380 (Codex CLI .232) | .233 HIVE had 3 verifiers, missing freq-aware attn | All 4 Tier 0 verifiers available now (exp2394/2395/2397 artifacts present) | retire if still 3-verifier-only |
| exp2411 FR-11 v3 | exp2400 (no artifact — session end), exp2383 (Codex CLI .232) | Session ended before task ran; infra failure | Codex CLI confirmed healthy (exp2393); no downstream gates blocking | retire if still no artifact |
| exp2413 KV260 v3 | exp2401 (no artifact — session end), exp2384 (Codex CLI .232) | Session ended before task ran | Codex CLI healthy; all prior .233 tasks complete before session end | retire if still no artifact |
| exp2414 Kinetic v3 | exp2402 (no artifact — session end), exp2385 (Codex CLI .232) | Session ended before task ran | Same — session timeout issue resolved | retire if still no artifact |
| exp2417 ship gate v3 | exp2403 (no artifact — session end), exp2388 (Codex CLI .232) | Session ended before task ran | Same — not a scope failure | retire if still no artifact |
