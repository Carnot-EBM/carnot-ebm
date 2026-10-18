# Research Roadmap v236: AUROC Ceiling Breach — DiffuTruth + PCIB + Conformal Ensemble, RTL Synthesis Fix, Phase 1 Ship Gate Completion

**Milestone:** 2026.05.236
**Status:** PROPOSED
**Date:** 2026-05-18
**Previous milestone:** 2026.05.235 (Codex Recovery Sprint v2 + AUROC Ceiling Assault v4) — 13/13 tasks completed

---

## What .235 Proved

Milestone .235 recovered fully from the .232/.234 Codex CLI backend failure. All 13 non-retro
tasks completed successfully (codex_cli_healthy=true; complexity_threshold=3 via exp2421).
Key results:

- **Best AUROC: 0.8896** (Hierarchical LogCons v2, exp2423) — new project record
- **HIVE v4 ensemble: 0.8864** (exp2422, 4 verifiers fused, ensemble_auroc_improved=true)
- **AUROC gap to HIVE peer: 0.034** (target: 0.9236)
- **FR-11 NSVIF online learning: PASSED** (cross_domain_retention_rate=1.0, exp2425)
- **Kinetic Langevin BAOAB: KL=1.987 vs CASAL KL=9.858** — best sampler delta: +7.87 (exp2428)
- **Dikin-Langevin: KL=2.41**, **DE-PSGLD: KL=3.85** — both beat CASAL significantly
- **KV260 Yosys: synthesis_errors=1** — Yosys binary IS installed, RTL IS present; the blocker
  is a content error in one of the 18 .v files, NOT missing infrastructure
- **Phase 1 ship gate: NOT MET** — pypi_published=true, hf_mirror_up=true,
  external_reproducer_exists=true; MISSING: mcp_docs_present=false, cli_docs_present=false
- **FST MCMC degenerate result** (exp2426): mean_acceptance_rate=1.0 (every token accepted),
  mch_violation_reduction=0.0, mean_energy_reduction_ratio=1.0 — energy_before/energy_after
  were NOT properly computed; the "validation" was trivial

**Critical finding:** exp2423 (LogCons, AUROC=0.8896) used z3_encoding_used=false — the fallback
FregeLogic path, not real Z3 hierarchical partial-order encoding. The best result in the project
was achieved WITHOUT the technique the task was designed to test. The v3 task must force real Z3.

---

## Architecture Diagram

```
Phase 0: Admin
┌─────────────────────┐
│ exp2434             │
│ Archive .235        │
│ (codex, ungated)    │
└─────────────────────┘

Phase 1: AUROC Breach Attempt (4 new verifiers + Z3 fix + conformal aggregation)
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ exp2435          │  │ exp2436          │  │ exp2437          │  │ exp2438          │
│ DiffuTruth       │  │ PCIB Tier 0l     │  │ LogCons Z3-True  │  │ Conformal        │
│ Tier 0k          │  │ (arXiv:          │  │ v3 (force        │  │ P-Value          │
│ (arXiv:          │  │ 2601.15652)      │  │ z3_encoding=true)│  │ Ensemble v1      │
│ 2602.11364)      │  │ (codex)          │  │ (codex)          │  │ (codex)          │
│ (codex)          │  │                  │  │                  │  │ gated:exp2437    │
└──────────────────┘  └──────────────────┘  └──────────────────┘  └────────┬─────────┘
                                                                            │ ensemble_auroc_improved
Phase 2: FR-11 + Infra                                                      │
┌────────────────────┐  ┌────────────────────┐  ┌──────────────────────┐   │
│ exp2439 (FR-11)    │  │ exp2440            │  │ exp2441              │   │
│ Online Learnability│  │ KV260 RTL Fix v5   │  │ Phase 1 Ship Gate    │   │
│ of CoT Verifiers   │  │ (debug synth err)  │  │ Completion v5        │   │
│ (codex, CSL)       │  │ (codex)            │  │ (write MCP+CLI docs) │   │
└────────────────────┘  └────────────────────┘  └──────────────────────┘   │
                                                                            │
Phase 3: Sampler + Constrained Generation                                   │
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│ exp2442          │  │ exp2443          │  │ exp2444          │          │
│ FST MCMC Energy  │  │ Kinetic Langevin  │  │ NCO Negative     │          │
│ Integration Fix  │  │ as FST Sampler   │  │ Constraint       │          │
│ v3 (fix          │  │ (connect         │  │ Decoding         │          │
│ acceptance=1.0)  │  │ KL=1.987→FST)    │  │ (arXiv:          │          │
│ (codex)          │  │ (codex)          │  │ 2605.10065)      │          │
└──────────────────┘  └──────────────────┘  └──────────────────┘          │
                                                                            ▼
Phase 4: Synthesis                                                   exp2445 (claude opus)
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ exp2445 (Paper-v6 Capstone v236, requires_claude=true, model=opus)                     │
│ exp2446 (Milestone .236 Retro, codex, ungated)                                         │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Four Critical Gaps Entering .236

### Gap 1: AUROC Gap 0.034 to HIVE Peer 0.9236
**Evidence:** Best AUROC = LogCons 0.8896 (exp2423) but z3_encoding_used=false (fallback path).
HIVE v4 achieved 0.8864 with 4 verifiers. Gap = 0.034.
**Fix:** Three orthogonal new signal axes:
  - DiffuTruth (exp2435): diffusion reconstruction error as energy proxy — independent axis
  - PCIB (exp2436): predictive coding + information bottleneck — semi-supervised, 75x data efficiency
  - LogCons Z3-True v3 (exp2437): force real Z3 partial-order encoding; z3_encoding_used=false
    means exp2423's 0.8896 was actually FregeLogic; real Z3 may push higher
  - Conformal P-Value Ensemble (exp2438): principled multiple-testing aggregation of all verifiers

### Gap 2: FST MCMC Degenerate Result
**Evidence:** exp2426 mean_acceptance_rate=1.0, mch_violation_reduction=0.0,
mean_energy_reduction_ratio=1.0. The MCHFSTFilter was implemented but energy computation
was trivial (energy_before=energy_after for all tokens because IsingEnergy of short prefix is
deterministic and identical when evaluated at the same spin configuration).
**Fix:** exp2442 forces real non-degenerate energy: use DIFFERENT spin configurations
(randomly sampled) for energy_before vs energy_after; verify acceptance_rate < 0.99.

### Gap 3: KV260 RTL Synthesis Error
**Evidence:** exp2427 synthesis_errors=1 with Yosys installed and 18 .v files present.
The blocking issue is a Verilog content bug in one of the .v files, NOT infrastructure.
Duration=241.56s confirms Yosys actually ran.
**Fix:** exp2440 reads the Yosys error output from exp2427 (/tmp/yosys_v4_output.txt or
re-runs Yosys with verbose output), identifies the specific file:line with the error,
applies a targeted fix, re-runs synthesis to confirm synthesis_errors=0.

### Gap 4: Phase 1 Ship Gate — MCP + CLI Docs Missing
**Evidence:** exp2431 phase1_ship_gate_met=false with ONLY two criteria missing:
mcp_docs_present=false, cli_docs_present=false. PyPI, HF mirror, and external reproducer
are all present.
**Fix:** exp2441 writes docs/mcp-server.md and docs/cli-guide.md, then re-runs the gate
audit to confirm phase1_ship_gate_met=true.

---

## Phase Descriptions

### Phase 0: Administration
- **exp2434** — Archive milestone .235 to research-complete.yaml, activate .236.

### Phase 1: AUROC Breach Attempt
- **exp2435** — DiffuTruth Tier 0k (arXiv:2602.11364). Implements hallucination detection
  via diffusion model reconstruction error: if the model's response is hallucinating, the
  diffusion model requires more reconstruction steps (higher energy) to generate it from
  noise. AUROC=0.725 on FEVER in the paper — but combined with Tier 0 verifiers via the
  conformal ensemble, the independent signal axis may boost the joint test significantly.
  This is the first verifier that uses a generative energy proxy rather than a discriminative
  logit-based score.

- **exp2436** — PCIB Tier 0l (arXiv:2601.15652). Predictive Coding + Information Bottleneck
  extracts a compressed representation Z of the model's internal state that predicts whether
  hallucination will occur. AUROC=0.8669 in the paper with 75x less training data than
  comparable probes. The 75x data efficiency directly addresses the limited 36-sample
  telemetry problem — PCIB can be trained reliably on small corpora.

- **exp2437** — Hierarchical LogCons Z3-True v3. Directly continues exp2423 but forces
  z3_encoding_used=true. Root cause of exp2423 fallback: the instruction field detection
  returned no hierarchy-detectable fields, so the code took the FregeLogic fallback. This v3
  explicitly extracts a synthetic instruction hierarchy from the telemetry prompt field
  (system > user > task levels) and encodes it as Z3 partial-order constraints before
  anything else. z3_encoding_used=false is a hard FAIL gate.

- **exp2438** — Conformal P-Value Ensemble v1 (arXiv:2508.18473). Uses conformal prediction
  theory to convert each verifier's score into a p-value via calibration set nonconformity
  scores, then aggregates via Fisher's combined test (- 2 Σ ln(p_i)). Compared to
  LogisticRegression soft-vote weights (exp2422 approach), conformal p-value aggregation:
  (a) has theoretical coverage guarantees, (b) naturally handles different score distributions
  per verifier, (c) achieves highest macro AUROC in the paper's comparison. Gates the
  capstone via ensemble_auroc_improved.

### Phase 2: Continuous Self-Learning + Infrastructure
- **exp2439** — FR-11 Online Learnability of CoT Verifiers (MANDATORY CSL task).
  Implements the learnability framework from arXiv:2603.03538: for a verifier V, measures
  soundness (fraction of correct verdicts it accepts) and completeness (fraction of
  incorrect verdicts it rejects). Littlestone dimension bounds the number of online updates
  needed before convergence. Ties into FR-11: the self-learning loop is only provably
  convergent if the verifier class has bounded Littlestone dimension. Task measures
  soundness_rate, completeness_rate, and estimated_littlestone_dimension for the NSVIF Z3
  verifier. A distinct scope from exp2425 (which ran the loop; this measures its
  learnability properties).

- **exp2440** — KV260 RTL Synthesis Fix v5. Reads the Yosys error output from exp2427
  (either via /tmp/yosys_v4_output.txt if present, or by re-running `yosys -p "read_verilog
  *.v; synth -top carnot_ising_top;" 2>&1`). Identifies the specific .v file and line number
  with the synthesis error. Applies a targeted fix. Re-runs Yosys to confirm
  synthesis_errors=0. Prior failure: exp2427 found synthesis_errors=1 — this task fixes it.

- **exp2441** — Phase 1 Ship Gate Completion v5. Reads the codebase to understand the MCP
  server implementation (python/carnot/mcp/) and CLI implementation (python/carnot/cli/).
  Writes docs/mcp-server.md (MCP server endpoints, config, usage examples) and
  docs/cli-guide.md (CLI commands, flags, example invocations). Re-runs the 5-criteria
  gate audit to confirm phase1_ship_gate_met=true. Prior failure: exp2431 confirmed the
  only missing criteria are the two docs.

### Phase 3: Sampler Integration + Constrained Generation
- **exp2442** — FST MCMC Energy Integration Fix v3. Root cause of exp2426's degenerate
  acceptance_rate=1.0: IsingEnergy of a prefix is computed at the SAME spin state before
  and after adding a token (because the token index is incremented but spin configuration
  is not re-sampled). Fix: energy_before uses spin configuration BEFORE token t, energy_after
  uses spin configuration AFTER random-flipping spin corresponding to token t's parity. This
  creates real energy differences. Acceptance gate: mean_acceptance_rate < 0.95 (not every
  token can be accepted if the energy function is non-trivial).

- **exp2443** — Kinetic Langevin as FST Sampler. Connects the best sampler from .235
  (KineticLangevinSampler, KL=1.987 vs CASAL KL=9.858, delta=+7.87) to the FST pipeline
  as the token-acceptance energy source. Instead of CASAL-sampled spin updates, uses
  KineticLangevin-sampled spin updates for the MH acceptance criterion. Measures
  kinetic_fst_mean_acceptance_rate (expected to be non-trivial since KineticLangevin
  samples from a closer approximation to the true distribution).

- **exp2444** — NCO Negative Constraint Decoding (arXiv:2605.10065). NCO uses Weighted
  Finite State Automata (WFSA) to represent negative constraints (patterns that MUST NOT
  appear in the output). The finite automaton efficiently rejects tokens that would complete
  a forbidden pattern. Applies to Carnot's constrained generation pipeline: encode NSVIF
  Z3 violation patterns as negative constraints in a WFSA, then use NCO to prevent
  hallucination-inducing token sequences at decode time. Measures nco_constraint_rejection_rate
  and nco_false_positive_rate on 20 telemetry entries.

### Phase 4: Synthesis
- **exp2445** — Paper-v6 Capstone v236 (requires_claude, model=opus). Synthesizes:
  all new verifiers (DiffuTruth, PCIB, LogCons Z3-True, Conformal Ensemble), Phase 1
  ship gate status (SHOULD be met by exp2441), RTL hardware track, sampler integration.
  Updates paper-v6 results table and ops/status.md. Gated on exp2438.ensemble_auroc_improved.

  REQUIRES_CLAUDE JUSTIFICATION: (1) Prior successful capstones (exp2432, exp2404, exp2376)
  all used Claude — Codex has never synthesized a multi-milestone AUROC capstone.
  (2) Reads 12+ artifacts across 5+ directories with cross-experiment AUROC reasoning.
  (3) Open-ended synthesis: does conformal ensemble close the HIVE peer gap? Does ship gate
  completion change the paper submission timeline? Multi-step reasoning under ambiguity.

- **exp2446** — Milestone .236 Operational Retrospective (codex, ungated). Records all
  task outcomes, computes n_experiments_completed, identifies top_3_gaps_for_237.

---

## Dependency Graph

```
exp2434 (archive)   → unblocks: research-complete.yaml updated

exp2435 (DiffuTruth)           → ungated
exp2436 (PCIB)                 → ungated
exp2437 (LogCons Z3-True v3)   → ungated
exp2438 (Conformal Ensemble)   → gated: exp2437.logcons_z3_true_auroc != null
exp2439 (FR-11 Learnability)   → ungated
exp2440 (KV260 RTL Fix v5)     → ungated
exp2441 (Ship Gate Completion) → ungated
exp2442 (FST MCMC Fix v3)      → ungated
exp2443 (Kinetic Langevin FST) → ungated
exp2444 (NCO Decoding)         → ungated

exp2438.ensemble_auroc_improved → exp2445 (Capstone, gated)
exp2446 (Retro, ungated)
```

---

## Hardware Requirements

| Task | Hardware | Notes |
|------|----------|-------|
| exp2434-exp2444 | CPU | No live GGUF inference required |
| exp2435-exp2436 | CPU | DiffuTruth/PCIB use cached telemetry and logit proxies |
| exp2438 | CPU | Conformal p-values are numerical computation |
| exp2440 | CPU | Yosys synthesis only |
| exp2445 | CPU | Claude reads artifacts and writes prose |

No live GPU inference required. All tasks are CPU-bound or network-bound.

---

## Key Papers (this milestone sweep)

| arXiv | Title | Hook |
|-------|-------|------|
| 2602.11364 | DiffuTruth: diffusion reconstruction as hallucination energy proxy | Tier 0k |
| 2601.15652 | PCIB: predictive coding + IB, AUROC=0.8669, 75x data efficiency | Tier 0l |
| 2508.18473 | Conformal p-value ensemble — highest macro AUROC | Conformal aggregation |
| 2603.03538 | Online Learnability of CoT Verifiers (Littlestone dim) | FR-11 soundness/completeness |
| 2605.10065 | NCO: negative constraint decoding via WFSA | Constrained generation |
| 2605.03971 | LaaB ACL 2026: logic-as-a-bridge for factuality | Tier 0h reference |
| 2604.26139 | HIVE soft-voting ensemble (AUROC=0.9236) | Target ceiling |
| 2604.09075 | Hierarchical LogCons Z3 instruction hierarchy | Tier 1 verifier |
| 2603.23397 | Kinetic Langevin BAOAB splitting | Sampler (best delta=+7.87) |
| 2509.10753 | HalluField: energy-field hallucination detection | Background reference |

---

## Success Criteria

| Criterion | Target | Gate |
|-----------|--------|------|
| AUROC improvement | Conformal ensemble > 0.8896 (any improvement) | exp2445 capstone |
| AUROC ceiling | Any verifier or ensemble AUROC > 0.9236 | Paper-v6 headline |
| FR-11 compliance | Soundness/completeness measured (not just loop ran) | Mandatory per CLAUDE.md |
| Phase 1 ship gate | phase1_ship_gate_met == true | Publication unlock |
| RTL synthesis | synthesis_errors == 0 | KV260 hardware track |
| FST fix confirmed | mean_acceptance_rate < 0.95 | exp2442 non-degenerate gate |

---

## Agent Routing

| Task | Agent | Justification |
|------|-------|---------------|
| exp2434 | codex gpt-5.5 | Simple archive task, proven pattern |
| exp2435-exp2436 | codex gpt-5.5 | New verifier implementations — Codex-default |
| exp2437 | codex gpt-5.5 | Z3 encoding fix is mechanical (z3_encoding_used gate) |
| exp2438 | codex gpt-5.5 | Conformal p-value computation is numerical |
| exp2439 | codex gpt-5.5 | Soundness/completeness measurement is algorithmic |
| exp2440 | codex gpt-5.5 | RTL fix: read Yosys stderr, find error line, patch .v file |
| exp2441 | codex gpt-5.5 | Doc writing from codebase inspection |
| exp2442-exp2444 | codex gpt-5.5 | Sampler/decoding implementations — Codex-default |
| exp2445 | claude opus | requires_claude: 12+ artifact synthesis, open-ended reasoning |
| exp2446 | codex gpt-5.5 | Templated retro, deterministic structure |

---

## Exclusion Manifest Cross-Check

Checked `ops/exclusion_manifest.yaml`. Retired scopes include: exp2091 (gemini CLI bail-out),
GRPO/VPRM lineage, WOPR puzzle cartridges, HardNet++/DSP repair iterations,
THRML scaling sweep, SpecAnn spectral annealing, iCE40 PIMI, HalluSAE geometric probe,
discriminative JEPA OOD failures (exp783/799/804/809/825/834/872/887).

None of the .236 tasks match retired scopes:
- DiffuTruth (exp2435): new scope, never proposed before
- PCIB (exp2436): new scope, never proposed before
- LogCons Z3-True v3 (exp2437): continuation of exp2423 (not retired; z3_encoding_used=false was a finding, not a failure verdict)
- Conformal Ensemble (exp2438): new scope, never proposed before
- FR-11 Learnability (exp2439): distinct from NSVIF online loop (exp2425); learnability theory scope
- KV260 RTL Fix (exp2440): hardware track not retired; infrastructure bug, not capability scope failure
- Phase 1 Ship Gate Completion (exp2441): gate not retired; only docs were missing
- FST MCMC Fix (exp2442): exp2426 produced degenerate result (z3_encoding issue), not retired
- Kinetic Langevin FST (exp2443): new scope (connecting sampler to FST), not retired
- NCO Decoding (exp2444): new scope, never proposed before

Cross-check: PASSED.

---

## Open Questions for .237

1. If conformal ensemble (exp2438) achieves > 0.9236 AUROC, does the paper hold for Phase 4
   active-inference empirical validation before arXiv submission?
2. If Phase 1 ship gate is met (exp2441), what is the exact publication unlock sequence?
3. KineticLangevin as FST sampler (exp2443) — does lower-KL sampler produce lower acceptance
   rate (harder constraints) or higher quality tokens?
4. NCO WFSA negative constraints (exp2444) — can Z3 UNSAT patterns be mechanically translated
   to automaton transitions, or does this require a learned translation?
