# Carnot Roadmap

Forward-looking research and product roadmap. For the chronological development record (per-milestone narrative going back to milestone `.39`), see `docs/research-log.md`.

## Current Milestone

**`2026.06.376` — Verifier-as-Reward pivot (in flight); FoVer headline paper publication-ready**

The project's north star is now **solving ARC-AGI-3, accurately and efficiently** (operator directive 2026-06-08); everything below is in service of it. Two current facts:

- **The FoVer verification headline is publication-ready.** The production verifier ensemble reaches **AUROC 0.9131** on the FoVer step-error corpus under a 5-seed dual-condition protocol (this repins the earlier 0.9857 figure downward after a 2026-05 adversarial audit). All four publication gates (G1–G4) now pass and the headline was **independently re-computed from a clean continuous-integration checkout in a non-operator environment** (2026-05-31) — `paper_ready: True`, pending the operator's submission decision.
- **The 2026-06-11 strategic pivot: verifier-as-reward.** The evidence across recent milestones established that the verifier-as-inference-*selector* role is a commodity (it does not beat self-consistency voting on real candidate pools), so the program pivoted to the verifier as a *training-time* reward signal — using the un-hallucinating execution verifier to certify training traces and fine-tune a generator. This is an **open bet under test** (milestone `.377`), not a proven result.

Earlier (2026-05) multi-corpus generalization work expanded evaluation beyond FoVer to verify the headline is not single-corpus overfit:

| ID | Corpus | Conditions | What it measures |
|---|---|---|---|
| `exp2802` | **MBPP** (500 problems) | architecture-only + production | Code-generation correctness verification on a new domain. Code execution gives clean ground-truth. |
| `exp2803` | **TruthfulQA** (817 questions) | architecture-only + production | Factual truthfulness verification. Different shape entirely from FoVer step pairs; will surface FoVer-shape overfit if present. BLEURT-based labels (no closed-weight judge per the decentralization rule). |
| `exp2804` | **HumanEval full** (164 problems) | architecture-only + production | Code-generation, full benchmark not the 50-problem subset. Extends the defensible code-repair results (Ising-guided fuzzing +18pp, CRANE +15pp) to ensemble AUROC. |
| `exp2807a` | **FoVer** (1000-subset) | architecture-only + production | Explicit memory-leakage control. Quantifies how much the production AUROC depends on accumulated NEXUS rules / constraint templates vs the verifier architecture itself. |
| `exp2805` | Cross-corpus matrix | derived | Classifies each verifier in the ensemble: `ARCHITECTURE_TRANSFER` (works without memory across all corpora), `MEMORY_AUGMENTED` (production high but architecture-only low), `CORPUS_SPECIFIC` (one-corpus-only — e.g. Tier 0s on arithmetic), `LOW_SIGNAL` (retirement candidates). |
| `exp2807` | Paper-v6 §5 integration | derived | Four-column dual-condition results table: Architecture-only AUROC / Production AUROC / Learning Δ / Peer baseline. Includes a §5.1 self-learning contribution disclosure. |
| `exp2808` | Capstone | claude (synthesis) | Confirms or refutes the FoVer-shape-overfit thesis; recommends whether to re-pin the headline; lists priorities for `.266+. |

After `.265 lands, the operator decides on the deferred corpora (HaluEval, FEVER) and any remediation work prompted by the dual-condition deltas.

## Recent milestones (last five)

| Milestone | Theme | Outcome |
|---|---|---|
| `2026.05.264` | Verifier energy debug v6 + Delta H2 regression fix + multi-track research | Closed; capstone exp2800 synthesized |
| `2026.05.263` | Differentiable conformal calibration + weak-strong policy fix + conformal selective acting | Closed; exp2757/2758/2759 all `complete:` |
| `2026.05.262` | Phase 4 FEP Adversarial Recheck (LOO-CV + held-out corpus); Tier 0z verifier | Phase 4 TAUTOLOGY flag resolved; Tier 0z landed but underperformed (auroc=0.5065) — flagged for fix-or-retire in `.264 |
| `2026.05.261` | Verifier Live-GPU v4; Ensemble v12 integration; ORCA-NEXUS Tier 3+ | Closed; capstone exp2763 (10/12 acceptance criteria met) |
| `2026.05.260` | Activate `.260; arXiv Package v3; Paper v6 Theory v4 | Closed; arXiv package refreshed for operator review |

For the full chronological record, see `docs/research-log.md`.

## Breakthrough Results

Results are labeled with provenance: **LIVE** (real model inference on GPU), **DERIVED** (post-hoc analysis of prior live artifacts), or **SIMULATED** (synthetic benchmark or canned CI cases). The 2026-04-16 provenance audit re-labeled several previously-unlabeled results.

| Result | Value | Source | Provenance | Significance |
|---|---|---|---|---|
| HumanEval code verification | +3.0pp [+0.6, +6.1] CI | Exp 226 | LIVE | Statistically significant on 164 official problems (gemma-4-E4B-it, 1574s runtime) |
| Ising-guided fuzzing (HumanEval 50) | 0.66 → 0.84 pass rate (+18pp) | Exp 1999 | LIVE | Execution-grounded repair |
| CRANE constrained decoding (HumanEval 50) | 0.70 → 0.85 pass rate (+15pp) | Exp 2090 | LIVE | vs a rigid grammar |
| PBT bug detection rate | 99.3% (144/145) | Exp 220 | LIVE | Property-based testing catches nearly all wrong code (Qwen+Gemma, 816s) |
| Typed IR constraints | +4.9pp (Gemma4) | Exp 221 | LIVE | Prompt-side constraint extraction works (81 cases, 459s) |
| EstimationVerifier on SVAMP | 0.90 AUC (vs 0.125 FoVer baseline) | Exp 908 | LIVE | Math reasoning verification |
| CCTU constrained tool-use | 4% → 12% completion | Exp 219 | LIVE | Tool-use micro-benchmark |
| Prompt-injection classifier | 0.91 AUROC (publication gate) | Exp 724 | LIVE | First passing of the 0.90 publication gate for the safety KAN line |
| VeriCoT GSM8K extraction TP rate | 0.5 → 1.0 | Exp 1101 | LIVE | Math-extraction fix for SOTA models |
| HalluGuard v3 cascade routing | 0.0pp accuracy delta with 4.4% cost savings | Cascade-routing analysis | DERIVED | Production deployment data |
| Two-GPU parallel retrain | 2.0× speedup, identical losses | Exp 746 | LIVE | Training-infra win |
| KV260 FPGA Ising sampler | Live on silicon | Exp 1041, 2026-04-22 | LIVE (hardware) | First AXI-Lite read and write returned from real KV260 silicon |
| Verifier ensemble AUROC (FoVer) | 0.9131 (5-seed dual-condition production; 0.8947 architecture-only; delta +0.0185) | Exp 2837 | LIVE | Repinned downward from the v2 0.9857 (Exp 2546) after the 2026-05-23 Deep Think round. Dual-condition protocol per `docs/blog/why-two-aurocs.html`. The HIVE peer comparator (+0.0061 lead at the corrected number) is no longer load-bearing; was +0.0621 at the v2 headline. |
| Adversarial PRM-BiasBench attacks | k=5 ensemble catches 60/60 | Exp 1133 + 1278 | LIVE | Defensible adversarial-audit result |

The earlier headline AUROC=0.9857 has been retracted to 0.9131 after the 2026-05-23 Deep Think round's dual-condition breakdown (Exp 2837): production AUROC 0.9131 (5-seed), architecture-only AUROC 0.8947, delta +0.0185. The repin is the defensible figure; the methodology that produced it (dual-condition protocol on architecture-only vs production state) is itself a paper contribution documented at `docs/blog/why-two-aurocs.html` and `docs/blog/two-retractions-and-a-rescue.html`. The multi-corpus dual-condition matrix (currently at v14, 29 clean rows) gives reviewers actionable evidence the single-corpus single-number alternative could not.

## Product Roadmap

| Tier | Products | Status |
|---|---|---|
| A: Ship Now | LLM output verification, code quality scorer, candidate ranker | Built |
| B: Build Next | Safety classifier (from gpt-oss-safeguard), compliance checker, multi-agent arbiter | Planned |
| C: Needs Hardware | Factual grounding gate, anomaly detector, prompt quality scorer | Phase 2 |
| D: Foundation Model | Data quality filter, synthetic data validator, test oracle | Phase 3 |

## Hardware Acceleration

| Hardware | Type | Status |
|---|---|---|
| RTX 3090 × 2 | CUDA GPU | Working — primary training/inference platform |
| Strix Point gfx1150 + XDNA2 NPU | ROCm + integrated NPU | ROCm 7.2.3 verified; sovereignty anchor |
| KV260 FPGA | Digital Ising (32 spins on silicon, 4K target) | Functional 2026-04-22; AXI r/w verified |
| GateMate A1-EVB-2M | Open-toolchain FPGA (n=16 Ising tile) | Terminal: bitstream flashed + on-board sampler at 951 Hz |
| PolarFire SoC Discovery Kit | RISC-V + FPGA | Terminal: end-to-end Carnot dispatch verified |
| D-Wave Advantage | Quantum annealing | Sampler built (Exp 320) |
| Extropic Z1 | Thermodynamic sampling | Early-access; monitor |
| Vulkan compute | Universal GPU | Planned for Phase 2 |
| Intel Loihi 2 | Neuromorphic | Need INRC access |
| NTT CIM | Coherent optical (100K+ spins) | Monitor |

## Phase 3: EBM/EBT Foundation Model

The long-term vision: an open-source foundation model based on hardware-acceleratable Energy-Based Models, with functional parity to Logical Intelligence's Kona.

- Continuous energy landscapes (bridge from discrete Ising/Z3)
- Non-autoregressive reasoning (generate via energy minimization)
- Language-free verification (learn constraint structure directly)
- Open-source (Apache 2.0) and hardware-portable (Vulkan / FPGA / D-Wave / TSU)

## Maintenance

This document is operator-curated per CLAUDE.md "Public Documentation Discipline". The autonomous loop does NOT auto-append to this file — per-milestone chronological narrative goes to `docs/research-log.md` instead. When the operator wants to update the Current Milestone section, the Recent Milestones table, the Breakthrough Results table, the Product Roadmap, the Hardware Acceleration table, or the Phase 3 section, they edit the file directly.
| 2026.05.241 | complete: best_241_auroc=0.975, phase4_validated_any=False, arxiv_ready=False | 8 experiments | 1 missing, 1 blocked; Phase 4 unvalidated; Operator hold persists |
| 2026.05.266 | Pre-test cascade structural deadlock — zero research artifacts produced | no experiments in timing scope | Deadlock confirmed: conductor cannot self-heal a gate it is behind; outer-loop intervention required for .267 |
