# Research Milestone 2026.04.84
## FR-11 SOTA Validation + Position Paper arXiv + FPGA Scale + RLVR-SSD Integration

**Status:** PROPOSED  
**Planned experiments:** exp1077–exp1089 (13 experiments)  
**Estimated wall time:** ~510 min  
**Primary constraint:** 2026-05-15 arXiv submission target for position paper

---

## What Milestone .83 Proved

Milestone .83 achieved 14/15 success criteria — the strongest milestone performance to date.
Key results that unlock .84's research agenda:

| Result | Value | .84 Implication |
|--------|-------|-----------------|
| KV260 FPGA smoke test | 24.83μs latency, 70 unique values | Scale benchmark possible: 64→1024 spins |
| SOS-KAN v3 AUROC | 0.9545 (target 0.72) | Production-quality energy verifier available |
| Triple integration E2E | All 4 tiers active, 50/50 questions | Cascade ready for real LLM validation |
| FR-11 alpha_t | 0.78 (but with Qwen3.5-0.8B) | MUST re-run with SOTA flagship model |
| Position paper draft | 6267 words, 8 sections | ArXiv prep needed; 5 figures missing |
| WOPR gallery | Sudoku live + GTW + Lights Out | N-Queens cartridge is natural next step |
| DualGPU | 2x RTX 3090 CUDA 12 live | Large GGUF models can run on real hardware |

One NOT_MET criterion:
- **Codex routing retired** — user directive 2026-04-30: stop proposing `agent_type: codex`.
  No .84 tasks will use codex backend.

---

## The 3 Biggest Gaps Between Current State and PRD Vision

### Gap 1: FR-11 headline result uses CPU smoke-test model (CRITICAL)

FR-11 alpha_t=0.78 was measured with `Qwen/Qwen3.5-0.8B` — explicitly the CPU smoke-test tier
model that CLAUDE.md prohibits as a headline result. The alpha_t cannot be cited in the position
paper's §3 until re-run with `unsloth/Qwen3.6-35B-A3B-GGUF` (the mandatory SOTA MoE model).

This blocks the position paper's empirical section. **Must run first in .84.**

### Gap 2: Position paper not arXiv-ready (STRATEGIC BLOCKER)

Draft v1 (6267 words, 8 sections) exists but lacks:
- Architecture diagram (cascade tier diagram)
- Energy landscape plot (SOS-KAN v3 on FoVer corpus)
- Hardware latency comparison (KV260 24.83μs vs CPU 290ms)
- Complete reference list (missing 4+ citations)
- arXiv metadata (categories, keywords, author affiliations)

Target date 2026-05-15 gives ~15 days. **Gemini agent for long-context technical review.**

### Gap 3: No live benchmark showing positive result with SOTA IT model

The triple integration cascade is validated on FoVer-labeled synthetic data (exp1073).
Research Program priority #2 ("Establish REAL baselines with instruction-tuned models")
remains unmet since milestone .33 (honest negatives only). The DualGPU is now live.
The cascade is validated. The SOTA models are cached. **No more excuses.**

---

## Milestone Architecture Diagram

```
Phase 0 — MANDATORY FR-11 SOTA (unconditional)
  exp1077 ─── FR-11 alpha_t SOTA v4 (Qwen3.6-35B-A3B-GGUF, live GPU)
               └── confirms alpha_t with flagship model; unlocks paper §3

Phase 1 — Position Paper (parallel with Phase 0)
  exp1078 ─── Position paper v2 arXiv prep (gemini, long-context)
               └── figures + technical review + arXiv metadata

Phase 2 — Real LLM Benchmark (gated on exp1077 GPU confirmed)
  exp1079 ─── Live SOTA IT benchmark v2 (100 GSM8K + 50 HumanEval)
               │    VeriCoT extraction on Qwen3.6-35B + Gemma-4-31B
               ├─── exp1083: RLVR+SSD integration v1 (gated on exp1079)
               └─── exp1085: Cascade validation on SOTA outputs (gated on exp1079)

Phase 3 — FPGA Expansion (standalone, KV260 smoke confirmed)
  exp1081 ─── FPGA scale benchmark 64→1024 spins
  exp1082 ─── Potts machine q=3 Verilog + simulation

Phase 4 — New Research (arxiv-inspired, standalone)
  exp1080 ─── SemEnergy probe v1 (arXiv 2508.14496 logit-space energy)
  exp1084 ─── Step-level PRM data generation (arXiv 2604.17957 MCTS pattern)

Phase 5 — WOPR Gallery (standalone)
  exp1086 ─── WOPR N-Queens cartridge (Ising ground-state)
  exp1087 ─── Gemini worktree conductor Tier B (MANDATORY — 3 milestones overdue)
  exp1088 ─── HF Spaces gallery update (gated on exp1086)

Phase 6 — Retro
  exp1089 ─── Milestone 2026.04.84 retrospective
```

---

## Dependency Graph

```
exp1077 (GPU live) ──► exp1079 (benchmark) ──► exp1083 (RLVR+SSD)
                                            └──► exp1085 (cascade on SOTA outputs)

exp1086 (cartridge) ──► exp1088 (HF deploy)

All others: standalone (no gate dependency on .84 experiments)
```

---

## Phase Descriptions

### Phase 0 — FR-11 SOTA (MANDATORY FIRST, Opus)

**exp1077:** The alpha_t=0.78 measurement from .83 is real research but cannot be the headline
result because it used `Qwen3.5-0.8B`. This experiment re-runs with `unsloth/Qwen3.6-35B-A3B-GGUF`
(mandatory SOTA MoE per CLAUDE.md). Expected: higher alpha_t (35B model has more nuanced outputs
that the 5-tier AND-composed verifier can distinguish from temperature-only filtering). Writes
100+ FR-11 training examples to `data/fr11_zenil_distill_v2.jsonl`. Budget: 60 min (Opus, GPU).

### Phase 1 — Position Paper Submission Prep (Gemini)

**exp1078:** Draft v1 is a solid skeleton (6267 words). The gemini agent (1M context window)
reads the ENTIRE research notes chain, all 5 Deep Think round documents, the experimental
results, and the arxiv scan, then produces:
(a) 5 matplotlib figures embedded as ASCII art + data + captions
(b) Technical review: verify theorem statements match actual implementation
(c) Reference completeness: ensure all cited papers have correct arXiv IDs
(d) arXiv metadata: primary category (cs.LG), secondary (cs.AI, quant-ph), keywords, abstract

Output: `docs/position-paper-draft-v2.md` + `docs/figures/` directory with Python figure scripts.

### Phase 2 — Real LLM Benchmark (Gated, Sonnet)

**exp1079:** First ever full live benchmark with SOTA instruction-tuned models (not base models,
not 0.8B). Uses `Qwen3.6-35B-A3B-GGUF` + `Gemma-4-31B-it-GGUF`. Uses `VeriCoTStepValidator`
(Exp 453, TP=8/20) not regex `ArithmeticExtractor` (TP=0). Runs 100 GSM8K + 50 HumanEval.
Reports first honest result (positive or negative) with a SOTA IT model and real extraction.

**exp1083:** Combines Carnot energy-based filter with self-distillation fine-tuning (arXiv
2604.03128 + 2604.01193 recipes). Three-way comparison: RLVR-only (Carnot filter selects)
vs. SSD-only (self-distillation on own outputs) vs. RLVR+SSD (filter then distill).
Trains on FR-11 training data from exp1077 + exp1079. Budget: 60 min.

**exp1085:** Takes the 150 real LLM outputs from exp1079 and runs them through the triple
integration cascade (already validated in exp1073). Measures: which tier catches which errors,
false positive rate by tier, whether cascade order is optimal. First real validation of cascade
on SOTA IT model outputs.

### Phase 3 — FPGA Expansion (Opus)

**exp1081:** KV260 smoke test passed at 24.83μs for 64 spins. This experiment scales: test at
128, 256, 512, 1024 spins by reconfiguring the bitstream's N_SPINS parameter (or uploading
different bitstreams for different N). Compare against `python/carnot/samplers/parallel_ising.py`
on the same sizes. Find the crossover point where FPGA beats CPU. Expected: crossover at ~256
spins (CPU is fast for tiny problems, FPGA shines at scale).

**exp1082:** The binary Ising (q=2) can encode correct/violated. The Potts machine (q=3) encodes
correct/partial/violated — enabling partial-credit constraint scoring. This experiment implements
`hardware/kv260/potts_sampler_v1.v` (q=3 states, n=64 spins, same AXI-Lite interface as Ising
v1). Since Vivado may not be available on this machine, includes a full Python simulation using
the synchronous checkerboard approach (validated in Exp 624). Deliverable: RTL + Python sim.

### Phase 4 — New Research from arxiv (Sonnet)

**exp1080:** Semantic Energy probe (arXiv 2508.14496) — implements Tier 0c logit-space energy
detector. Instead of token-level activations or output probabilities, reads pre-softmax logits
from the penultimate transformer layer and applies a Boltzmann-inspired energy calculation.
Trains on FoVer corpus. Target: AUROC > 0.80, faster inference than full Ising tier (goal: <1ms).

**exp1084:** Step-level PRM data generation (arXiv 2604.17957 MCTS pattern). Uses the triple
integration cascade (already working) as the step scorer: generate partial reasoning trajectories,
score each prefix with the cascade, assign step-level labels. Target: 5000+ step-level labeled
examples for ThinkPRM retraining. If successful, retrains ThinkPRM and measures AUROC improvement.

### Phase 5 — WOPR Gallery + Gemini Conductor (Mandatory)

**exp1086:** N-Queens cartridge. Harder than Lights Out (8-Queens is a classical constraint
satisfaction benchmark). Ising formulation: E = Σ (row_violations + col_violations + diag_violations).
Uses existing `parallel_ising.py`. Shows energy descent as queens are placed. Tests: final_energy=0
(valid placement), convergence within 10000 iterations.

**exp1087:** Gemini worktree conductor Tier B. This is MANDATORY per CLAUDE.md's Overdue-Priority
Forcing Function — "parallel-multi-agent-conductor" has been pending 3 consecutive milestones
(.81, .82, .83). The codex Tier A is paused (user directive), but gemini Tier B is independent.
Implements: `~/.config/systemd/user/carnot-conductor-gemini.service`, per-worktree state file
`ops/conductor-state_gemini.json`, merge-back protocol for gemini-produced artifacts. Budget: 35 min.

**exp1088:** HF Spaces gallery update — deploy N-Queens cartridge (and any others from .84) to
`https://huggingface.co/spaces/Carnot-EBM/wopr-games`. Gated on exp1086 (cartridge must be
written and tested before deploy).

### Phase 6 — Retrospective

**exp1089:** Standard milestone retro — evaluate all 13 success criteria, write process notes,
update ops/changelog.md and ops/conductor-log.md. max_turns: 20.

---

## Success Criteria (13)

| # | Criterion | Experiment | What Counts |
|---|-----------|------------|-------------|
| 1 | fr11_alpha_t_sota_confirmed | exp1077 | alpha_t measured with Qwen3.6-35B-A3B-GGUF, inference_mode=live_gpu |
| 2 | position_paper_arxiv_ready | exp1078 | figures generated, reference list complete, arXiv metadata present |
| 3 | live_benchmark_honest_result | exp1079 | 100 GSM8K + 50 HumanEval with SOTA IT model, honest_verdict reported |
| 4 | semenergy_probe_auroc_above_07 | exp1080 | AUROC > 0.70 on FoVer corpus with logit-space energy |
| 5 | fpga_speedup_vs_cpu | exp1081 | KV260 faster than CPU at 256+ spins (crossover measured) |
| 6 | potts_simulation_validated | exp1082 | q=3 Potts sampler simulates without errors, RTL written |
| 7 | rlvr_ssd_honest_result | exp1083 | Three-way comparison complete, honest_verdict reported |
| 8 | prm_data_generated | exp1084 | >= 2000 step-level labeled examples generated |
| 9 | cascade_validated_sota_outputs | exp1085 | Cascade runs on real SOTA model outputs, tier breakdown reported |
| 10 | nqueens_cartridge_shipped | exp1086 | final_energy=0.0 in tests, WOPRGame interface implemented |
| 11 | gemini_worktree_implemented | exp1087 | systemctl service stub written, routing validated |
| 12 | gallery_updated_hf_spaces | exp1088 | N-Queens live on HF Spaces (200 OK) |
| 13 | retro_complete | exp1089 | this artifact |

---

## Hardware Requirements

| Experiment | Hardware | Reason |
|------------|----------|--------|
| exp1077 | 2x RTX 3090 (CUDA 12) | Qwen3.6-35B GGUF inference requires ~20GB VRAM |
| exp1079 | 2x RTX 3090 (CUDA 12) | SOTA IT model inference (35B + 31B) |
| exp1081 | KV260 FPGA (192.168.51.98) | Scale benchmark requires board SSH access |
| exp1082 | None (Python sim + Verilog) | Vivado not available; simulation suffices for RTL verification |
| exp1083 | 2x RTX 3090 (CUDA 12) | SSD fine-tuning requires GPU |
| Others | CPU only | — |

---

## Decentralization Implications

All experiments respect CLAUDE.md decentralization rules:
1. Local-first: All LLM inference uses local GGUF models (Qwen3.6-35B, Gemma-4-31B)
2. Closed frontier models optional: Position paper uses gemini for synthesis only (not for core results)
3. Distribution: New model artifacts published to HF Spaces (already established pipeline)
4. Multiple surfaces: Python API, MCP server, and CLI paths all active
5. Hardware portability: KV260 + Python CPU paths both validated per experiments 1081-1082
6. Data minimization: No external API calls with research data
7. No vendor-specific core dependencies: All cascade tiers use open abstractions

---

## Failed-Experiment Rerun Discipline

Experiments carrying prior_failures blocks in .84:

| Experiment | Prior failure | Root cause addressed |
|------------|--------------|---------------------|
| exp1079 | exp1079 is a respawn of the benchmark pattern from Exps 427/439 | Both prior failures used wrong extraction (regex) and wrong models (base/0.8B). v2 uses VeriCoT + SOTA IT models — genuinely different scope. |
| exp1083 | Energy-Selection SSD (Exp 1031, partial) | Exp 1031 measured energy-selection alone without RLVR combination. This experiment adds RLVR signal — different architecture, not a rerun. |

No experiment with `retire_if_same_verdict: true` is proposed without a verified change.

---

## CLAUDE.md Compliance Checklist

- [x] At least one self-learning experiment: exp1077 (FR-11), exp1083 (RLVR+SSD), exp1084 (PRM data)
- [x] SOTA models in all GPU experiments: Qwen3.6-35B-A3B-GGUF + Gemma-4-31B-it-GGUF
- [x] No codex tasks (user directive)
- [x] No emoji in public documentation
- [x] Mandatory overdue priority addressed: gemini worktree conductor (exp1087, 3 milestones pending)
- [x] Hardware wishlist checked: KV260 scale benchmark (exp1081), Potts machine (exp1082)
- [x] research-references.md updated with new papers before experiments designed
- [x] 10-14 experiments: 13 total (1077-1089)
