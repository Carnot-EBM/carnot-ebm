# Research Roadmap vNEXT: Milestone 2026.04.106

Planned: 2026-05-05
Status: Draft for conductor execution
Predecessor: 2026.04.105 Terminal Certificate Evidence + Skill-Localized Semantic Repair + Verifier-Selected Self-Learning
Roadmap YAML: `research-roadmap-next.yaml`

## What Milestone .105 Proved

| Track | Evidence | Finding |
|---|---|---|
| .104 handoff | `exp1351` | Carry-forward audit complete; `exp1340` correctly classified as missing. |
| Certificate preflight | `exp1352` | `max_token_budget_sufficient=true`, `dynamic_dispatch_preserved=true`, `sota_run_allowed=true` — preflight passed. |
| SOTA certificate run | `exp1353` | Ran; produced 4 cases; `certificate_parse_rate=0.0`, `trigger_token_hit_rate=0.0`. Dominant blocker: `missing_structural_tag`. Thinking-mode tokens (`<think>...</think>`) consumed the generation budget before the structural branch-selector tag was emitted. Terminal negative evidence, not an artifact-missing failure. |
| LogicSkills skill split | `exp1354` | Ran on exp1353 cases; `skill_split_claim_allowed` tied to 4 parse-zero cases. Minimal evidence. |
| Semantic validator | `exp1355` | Gate-blocked: `certificate_parse_rate=0.0 < 0.75`. |
| MCS repair / scheduler | `exp1356`, `exp1357` | Missing or blocked; no semantic repair evidence. |
| Continuous self-learning | `exp1358` | Positive replay-only: `self_learning_delta_overall=1.596429`, `dvi_ready=true`, `fresh_verified_sample_count=0`, `nonforgetting_certificate_rate=1.0`, `memory_regression_count=0`. Mandatory requirement satisfied; not headline. |
| DVI / GRPO | `exp1359`, `exp1360` | Missing or blocked behind semantic/DVI gates. |
| Hardware mapping | `exp1361` | CPU-only p-dit certificate-state mapping: `state_expansion_ratio=4.0`, `energy_equivalence_error=0.0`. No hardware claim. |
| Publication boundary | `exp1362` | Hold active; no external-dependency claim. |
| Retro | `exp1363` | 9 of 12 criteria met; named carry-forward tasks. |

**Key root cause diagnosis.** The `.105` retro verdict explicitly names the blocker for `.106`:
> "Force tag-first emission or retire the trigger-before-constrain branch; acceptance must require `parse_rate >= 0.75` plus truthfulness and UNKNOWN preservation."

The structural tag that selects the grammar branch (SAT / UNSAT / UNKNOWN / repair) is never emitted because Qwen3 and Gemma4-it models generate `<think>...</think>` thinking tokens as the first output, consuming the token budget before the constrained certificate section is reached. The trigger-before-constrain approach assumes the model will emit a trigger token early; thinking-mode models do not.

## Research Signals Added Before Planning

The post-.105 sweep added the following 2025–2026 sources to
`research-references.md` before this roadmap was designed:

- `arXiv:2512.20664`, Eidoku: CSP neuro-symbolic verification gate that
  checks structural connectivity, feature-space consistency, and logical
  entailment independently of any certificate grammar. Directly maps to
  Carnot's Ising (structural), KAN (geometric), and Z3/NSVIF (symbolic)
  tiers.
- `arXiv:2602.11364`, DiffuTruth: non-equilibrium hallucination detection
  via diffusion model reconstruction energy. Unsupervised SOTA on FEVER
  (AUROC 0.725). Complementary non-equilibrium energy signal for Carnot's
  Ising/KAN pipeline.
- `arXiv:2502.09061`, CRANE: alternating unconstrained reasoning and grammar-
  constrained generation. Reports 30%+ gains on symbolic tasks. Combined with
  prefix injection of the structural tag, CRANE's alternating design prevents
  thinking-mode pre-emption of the constrained region.
- `arXiv:2505.15960`, FOVER: 80K formally verified PRM training data with
  Z3/Isabelle step labels. Alignment target for Carnot's LogicSkills
  certificate skill split.
- `arXiv:2602.06737`, Optimal KAN Verification: PWA abstraction for KAN
  spline units + MILP encoding for property verification. Formal correctness
  bounds for Carnot's GS-KAN energy tier.
- `arXiv:2604.17109`, Fully Parallel Ising with Inertia: synchronous p-bit
  updates with inertia term on FPGA. Reduced sweeps-to-convergence for dense
  graphs. Directly relevant to Carnot's `parallel_ising.py` and KV260 v4 RTL.
- `arXiv:2511.00746`, Ising-NN Correspondence: systematic mapping from
  trained feed-forward neural networks to Ising hardware. Theoretic foundation
  for mapping Carnot's KAN/Boltzmann tiers to FPGA without separate Vivado
  bitfile synthesis.

## Three Biggest Gaps

1. **Structural tag never emitted by thinking models.** `exp1353` confirmed
   that the trigger-before-constrain approach fails for Qwen3/Gemma4-it
   because thinking tokens appear before any user-requested structural tag.
   The fix is **prefix injection**: supply the structural tag as the first
   token(s) of the assistant turn before generation begins. Llama.cpp's
   `--grammar` flag combined with a forced assistant prefix achieves this.
   If prefix injection also fails, the trigger-before-constrain branch must
   be retired (per the `.105` retro's `retire_if_same_verdict: true` entry)
   and Eidoku CSP becomes the primary verification path.

2. **No headline self-learning evidence yet.** `exp1358` is mandatory and
   positive but replay-only. FR-11 requires fresh verifier-selected samples.
   This is blocked by the certificate gate: without parsed certificates, the
   semantic validator cannot produce verified cases for memory promotion.
   Closing this gap requires either (a) tag-first prefix injection succeeding
   and the semantic chain opening, or (b) Eidoku CSP feasibility scores
   substituting as the verifier signal for memory promotion.

3. **Publication hold active; paper integrity not closed.** The arXiv
   submission is blocked by the claim boundary audit and open paper-integrity
   issues. `.106` must close or explicitly defer each pending publication
   blocker based on honest local evidence, not analogy.

## Architecture Target

```text
Phase 0: .105 handoff closure
  exp1364 .105 artifact integrity audit
      |
      v
Phase 1: Tag-first prefix injection + independent CSP verification
  exp1365 Eidoku CSP neuro-symbolic probe (CPU, unconditional)
      |
  exp1366 Certificate v8 — tag-first prefix injection CRANE pattern (GPU)
      |    gated: exp1364.terminal_certificate_required == true
      |
  exp1367 DiffuTruth energy-of-falsehood probe (CPU, unconditional)
      |
Phase 2: Semantic repair chain (all gated on exp1366.certificate_parse_rate >= 0.75)
  exp1368 FOVER-aligned LogicSkills skill audit
      |
  exp1369 Semantic validator v2 — NSVIF + partial SMT
      |
  exp1370 VERGE MCS repair localization v2
      |
  exp1371 Margin-aware Cactus/BEAVER scheduler v3
      |
Phase 3: Hardware and formal verification (unconditional)
  exp1372 Optimal KAN PWA formal verification (CPU)
  exp1373 Fully parallel Ising with inertia (CPU)
      |
Phase 4: Self-learning + publication + retro
  exp1374 FR-11 continuous self-learning v3 (unconditional, fallback to replay)
  exp1375 Publication hold + claim boundary v15
  exp1376 Milestone 2026.04.106 retrospective
```

## Dependency Graph

```
exp1364 ──→ exp1366 (gate: terminal_certificate_required)
exp1366 ──→ exp1368, exp1369 (gate: certificate_parse_rate >= 0.75)
exp1369 ──→ exp1370 (gate: validator_execution_pass_rate >= 0.5)
exp1370 ──→ exp1371 (gate: repair_hint_precision >= 0.5)
exp1369 ──→ exp1374 (gate: semantic_validator_claim_allowed)
exp1365 [unconditional — parallel CSP path]
exp1367 [unconditional — non-equilibrium complement]
exp1372 [unconditional — formal KAN verification]
exp1373 [unconditional — parallel Ising inertia]
exp1374 [unconditional with fallback to replay if gate misses]
exp1375 [unconditional — reads all available .106 evidence]
exp1376 [unconditional — milestone closeout]
```

## Hardware Requirements

| Experiment | GPU | CPU only |
|---|---|---|
| exp1364, 1365, 1367, 1372, 1373, 1374, 1375, 1376 | No | Yes |
| exp1366 | Yes (RTX 3090 x2 preferred) | Fallback smoke test |
| exp1368–1371 | No | Yes |

## Phase Descriptions

### Phase 0 — Handoff Closure (unconditional)

`exp1364` reads all `.105` artifacts and classifies:
- `terminal_certificate_required`: true (thinking-mode blocker is terminal negative evidence, not missing artifact)
- `thinking_mode_blocker_confirmed`: true (structural tag consumed by `<think>` tokens)
- `prior_certificate_parse_rate`: 0.0 (from exp1353)
- `semantic_work_completed`: false (exp1355–1357 blocked)
- `self_learning_state`: replay-only, positive (`self_learning_delta_overall=1.596429`)

### Phase 1 — Tag-First + Parallel CSP (partially unconditional)

**exp1365 — Eidoku CSP probe (unconditional, CPU-only)**

Implements Eidoku's three CSP proxy costs on existing FoVer corpus cases
and any available SOTA GGUF free-text outputs:
- Structural cost: graph connectivity of reasoning steps
- Geometric cost: feature-space consistency via embedding similarity
- Symbolic cost: Z3 entailment check for extractable claims

Eidoku is the fallback verification path: if `exp1366` fails (tag-first
also produces parse_rate=0.0), Eidoku CSP scores serve as the verifier
signal for self-learning memory promotion and publication evidence.

**exp1366 — Certificate v8 Tag-First Prefix Injection (GPU)**

Gated on `exp1364.terminal_certificate_required == true`.
Root cause from `.105`: thinking-mode tokens precede structural tag.
Fix: prefix-inject the structural tag into the assistant turn using
llama.cpp's partial assistant response or `--grammar` with a forced
prefix. The grammar constraint is active from generation token 0.
Combined with CRANE's alternating pattern: unconstrained reasoning phase
(capped token budget) followed by forced-prefix constrained certificate.

This experiment must either produce `certificate_parse_rate >= 0.75` or
write a `terminal_blocker` artifact with `retire_if_same_verdict: true`
evidence. A missing or bootstrap-only artifact is not acceptable.

**exp1367 — DiffuTruth energy-of-falsehood probe (unconditional, CPU-only)**

Implements the Generative Stress Test from `arXiv:2602.11364` using local
text embedding perturbation as a diffusion proxy (full discrete diffusion
is not required for a feasibility probe). Measures whether reconstruction
energy correlates with Ising energy on FoVer corpus cases.

### Phase 2 — Semantic Repair Chain (gated on parse_rate >= 0.75)

**exp1368 — FOVER-aligned LogicSkills skill audit (gated on exp1366)**

Builds on `.105` `exp1354` (LogicSkills skill split). Aligns Carnot's
certificate skill failures with FOVER-80K's Z3/Isabelle error taxonomy.
Measures whether Carnot's symbolization/countermodel/validity gaps match
the FOVER error distribution; this determines whether FOVER training data
could reduce the dominant skill gap.

**exp1369 — Semantic validator v2 (gated on exp1366)**

Extended `.105` `exp1355`. Adds NSVIF-style Z3 constraint models
(`arXiv:2601.17789`) for structured text claims alongside partial-SMT
validation. Tracks `unknown_preservation_rate`, `z3_constraint_pass_rate`,
and `semantic_validator_claim_allowed`.

**exp1370 — VERGE MCS repair v2 (gated on exp1369)**

Extended `.105` `exp1356`. More cases, tighter precision measurement.
Locates Minimal Correction Subsets for semantic validator failures.

**exp1371 — Margin-aware Cactus/BEAVER scheduler v3 (gated on exp1370)**

Extended `.105` `exp1357`. Margin-aware verifier routing with MCS repair
hint integration. Only claims verifier-call reduction if false acceptance
is zero.

### Phase 3 — Hardware and Formal Verification (unconditional)

**exp1372 — Optimal KAN PWA formal verification (CPU-only)**

Implements PWA abstraction for Carnot's GS-KAN splines per
`arXiv:2602.06737`. Encodes a simple energy-bound property as a MILP.
No hardware claim; this is a formal software correctness proof for the
KAN energy tier.

**exp1373 — Fully parallel Ising with inertia (CPU-only)**

Adds inertia dynamics to `parallel_ising.py` per `arXiv:2604.17109`.
CPU simulation benchmark against existing checkerboard results on FoVer
constraint problems. Records FPGA mapping estimate for KV260 v4 RTL.

### Phase 4 — Self-Learning, Publication, Retro (unconditional)

**exp1374 — FR-11 continuous self-learning v3 (unconditional)**

Always runs. Primary path: if `exp1369` semantic validator passed, use
verifier-selected cases for memory promotion and measure
`fresh_verified_sample_count`. Fallback path: if semantic gate is closed,
use Eidoku CSP feasibility scores from `exp1365` as the verifier signal
for promotion, with `headline_result_allowed=false` unless CSP scores
are independently validated. Reports `self_learning_delta_overall`,
`nonforgetting_certificate_rate`, `headline_result_allowed`.

**exp1375 — Publication hold + claim boundary v15 (unconditional)**

Reads all `.106` evidence. Updates claim boundary based on tag-first
certificate result, Eidoku CSP viability, formal KAN properties, and
self-learning state. Publication hold stays active until certificate
evidence clears the semantic repair chain.

**exp1376 — Milestone 2026.04.106 retrospective (unconditional)**

Evaluates all 12 success criteria. Produces `.107` carry-forward plan.

## Success Criteria (12 total)

| # | Criterion | Source |
|---|---|---|
| 1 | `audit_105_complete` — thinking-mode blocker confirmed, carry-forward classified | exp1364 |
| 2 | `eidoku_csp_probe_complete` — CSP feasibility measured on FoVer corpus | exp1365 |
| 3 | `cert_v8_tag_first_ran` — tag-first attempt produced terminal evidence (positive or retire) | exp1366 |
| 4 | `diffu_truth_probe_complete` — reconstruction energy measured | exp1367 |
| 5 | `fover_skill_audit_ran` — FOVER alignment measured (may be gate-blocked) | exp1368 |
| 6 | `semantic_validator_v2_ran` — validator ran (may be gate-blocked) | exp1369 |
| 7 | `mcs_repair_v2_ran` — MCS repair ran (may be gate-blocked) | exp1370 |
| 8 | `kan_pwa_formal_verified` — KAN formal property verified or bounded | exp1372 |
| 9 | `parallel_ising_inertia_measured` — convergence speedup measured | exp1373 |
| 10 | `self_learning_v3_complete` — mandatory self-learning with non-forgetting | exp1374 |
| 11 | `publication_boundary_refreshed` — hold state documented with .106 evidence | exp1375 |
| 12 | `retro_complete` | exp1376 |

## Decentralization Implications

All experiments use local SOTA GGUFs (Qwen3.6-35B-A3B-GGUF, Gemma4-31B-it-GGUF,
Gemma4-26B-A4B-it-GGUF) for any LLM-bearing work. Eidoku CSP, DiffuTruth, KAN
formal verification, and Ising inertia are CPU-only — no closed-weight dependency.
Tag-first prefix injection uses llama.cpp grammar constraints, which are
fully open-source and reproducible. No vendor-specific API is required for any
experiment in this milestone.
