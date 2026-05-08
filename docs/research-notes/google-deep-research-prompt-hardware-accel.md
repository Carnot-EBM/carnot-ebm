# Google Deep Research prompt — Carnot hardware acceleration survey

Drafted 2026-05-08. Paste the section below into Google Deep Research.

---

## Background

Carnot is an open-source Energy-Based Model (EBM) framework for verifying
and repairing Large Language Model outputs. It runs an ensemble of k=6
to k=15 verifiers (Z3 SMT, AST structural, semantic consistency, ThinkPRM,
SOSKAN-Energy, SemEnergy probes), AND-composes their accept regions,
and uses block-Gibbs sampling on a tiny-Ising substrate as the inference
path. The substrate scales from n=4 to n=128 production spins, with
n=256+ planned for Phase 3 deployment.

The project explicitly targets **sovereignty / decentralization**: any
hardware path must allow Carnot to run without closed-source vendor
lock-in. Hardware portability is treated as a political requirement,
not just engineering.

**Apache-2.0 licensed**. Repository: `github.com/Carnot-EBM/carnot-ebm`.
Vendored sampler: Extropic THRML 0.1.3 (Apache-2.0, CPU/GPU JAX-native).

## Current hardware portfolio

| Tier | Hardware | Role |
|---|---|---|
| Primary training | 2× NVIDIA RTX 3090 (CUDA) | Discrete dual-GPU; 48 GB VRAM |
| Secondary | AMD Strix Point gfx1150 APU (ROCm) | Integrated GPU; 67 GB unified memory |
| Sovereignty anchor | Consumer NPUs (Intel AI Boost, AMD Ryzen AI/XDNA, Apple Neural Engine, Qualcomm Hexagon) | Verifier inference on hardware users already own |
| Web | WebGPU gateway | Browser-side energy evaluation distributed via WebSocket |
| POC tier | Xilinx KV260 FPGA | Quadratic-Ising acceptance circuit demo (not load-bearing) |
| Research-class | Extropic Z1 (when shipped) | Future thermodynamic ASIC; awaiting public specs + access |

## What we want to accelerate

1. **Block-Gibbs sampling**: K=100 sweeps per inference request, n=128
   production scale, candidate-warm-start initialization. The sampler is
   the load-bearing inference primitive.
2. **AND-composed verifier evaluation**: k=6 → k=15 verifiers; each
   verifier evaluated per accepted sample; Soft-Gibbs Residual rejection
   `µ_res^β(y) ∝ µ(y)·exp(−β V(y))` where `V(y) = Σ 1{y ∉ Ŝ_i}`.
3. **Phase 5 in-situ training**: adaptive-K Persistent Contrastive
   Divergence with simulated-annealing temperature cycling (or parallel
   tempering) on non-convex Ising landscapes. K must scale dynamically;
   K=1 PCD diverges per a recent theoretical audit.
4. **Energy function evaluation**: pairwise Ising `E(y) = −Σ J_ij y_i y_j
   − Σ h_i y_i` plus k-AND verifier indicator sums. Both linear-in-y
   for unary scores and pairwise φ.

## Constraints (must respect ALL)

- **Sovereignty (rule 1)**: every capability must work end-to-end with
  fully open-source hardware paths. Closed-stack vendor hardware can
  be a *bonus* tier but never the *only* tier.
- **Distribution mirroring (rule 3)**: any vendor SDK we depend on
  must be mirrored to project-controlled repos; closed-source SDKs
  with no mirror path are disqualified.
- **No vendor abstractions in core (rule 7)**: hardware-specific code
  lives in clearly-named submodules; core depends on a `SamplerBackend`
  abstract protocol.
- **Cost ceiling**: target hardware access for ≤$10,000 per
  deployment (small lab budget); cloud rentals up to $500/month
  acceptable for evaluation; production deployment must remain
  reachable for a single individual or institution under
  sanctions/compute-export-restrictions.

## What I want you to investigate

### Section A: Currently-available hardware (2025-2026)

For each candidate platform below, find:
- Status (shipping / pre-order / vapor)
- Approximate cost-per-unit at small-buyer quantities
- Sample-throughput claim (samples/sec for a 128-spin Ising or
  equivalent Boltzmann sampling task)
- License of the toolchain (open-source / closed / royalty-bearing)
- PyTorch / JAX / direct-Python compatibility
- Any independent third-party benchmarks (not just vendor PR)
- Sovereignty profile (closed-stack vendor / open-FPGA / open-ASIC)

**Platforms to evaluate:**

1. **Thermodynamic computing**:
   - Extropic Z1 (status, timeline, public specs, access program)
   - Normal Computing (any product? roadmap?)
   - Other thermodynamic-ASIC startups (research scan)

2. **Analog Ising machines / Coherent Ising machines**:
   - NTT coherent Ising machine (commercial access? remote benchmarks?)
   - D-Wave (current generation, accessibility, performance vs
     classical Gibbs at n=128)
   - Memcomputing Inc. (still active? products?)
   - Spatial Photonic Ising Machines (papers cite Pierangeli et al.,
     Honjo et al.; commercial follow-ups?)
   - Magnetic-oscillator Ising machines (Si et al. Nature Comm 2024
     follow-up)

3. **Photonic compute**:
   - Lightmatter (Envise / Passage status, MAC throughput, software
     stack openness)
   - Lightelligence
   - Q.ANT
   - Salience Labs
   - Specifically: which photonic platforms can run Boltzmann
     sampling vs only matrix-multiply?

4. **Open-FPGA paths**:
   - Xilinx KV260 alternatives with similar Ising-circuit fit
   - Lattice ECP5 (open SymbiFlow toolchain)
   - Lattice Nexus
   - GOWIN GW5A
   - Microchip PolarFire SoC
   - Specifically: which boards have cheap open-toolchain bitstream
     generation and decent BRAM/DSP for k=15 verifier evaluation?

5. **Custom ML ASICs (open-software-stack ones)**:
   - Tenstorrent Wormhole / Blackhole (cards, software openness,
     PyTorch support, sample-per-watt vs RTX 3090)
   - Cerebras CS-3 (cloud access, openness)
   - Groq LPU (inference focus; useful for verifier evaluation?)
   - SambaNova
   - Specifically: which platforms have permissive licensing for
     research workloads?

6. **NPUs (consumer edge)**:
   - Intel AI Boost (Lunar Lake / Arrow Lake): peak TOPS, tooling
     status (OpenVINO maturity for EBM-class workloads?)
   - AMD Ryzen AI / XDNA: Ryzen AI 1.x SDK status, ONNX support
   - Apple Neural Engine: CoreML for non-Apple-published models?
   - Qualcomm Hexagon (Snapdragon X): QNN status, Linux access path
   - NVIDIA Jetson Orin / Thor: not strictly an "NPU" but
     consumer-edge GPU class

7. **Quantum sampling** (long-shot but worth checking):
   - Any 2026-available quantum annealer or gate-model device that
     can sample Boltzmann distributions at n=128 with reasonable
     fidelity? Not just "we ran a 5-qubit demo" — production-class
     access?

### Section B: Coming hardware (12-18 month horizon)

What's on roadmaps but not yet shipping? Vendor announcements + credible
analyst reports. Specifically interested in:
- 2027 thermodynamic computing roadmap (Extropic Z2? others?)
- Open-source ASIC tape-outs (anything from Efabless / OpenROAD-class)
- 1-bit / TBN (Ternary Bit Networks) hardware — does it exist commercially?
- Magnetic-tunnel-junction (MTJ) p-bit hardware — Sayed Kara at
  Purdue and others; commercial path?

### Section C: Software stacks worth knowing about

For each platform identified, what's the *open-source* layer between
PyTorch/JAX and the hardware? Specifically:
- Does it support arbitrary Ising graph topology (signed couplings)?
- Does it support block-Gibbs OR forces single-site MH?
- Can it expose sample distributions (not just argmin solutions)?
- Bitstream / firmware reproducibility — can a third-party rebuild
  from sources?

### Section D: Strategic recommendations

After surveying, give me:
1. **The top 3 platforms by sovereignty profile** that are practically
   reachable in 2026 (≤$10k, open toolchain).
2. **The top 3 platforms by raw sample-throughput on n=128 Ising**,
   regardless of openness, for benchmarking ground truth.
3. **The top 3 NPU paths** that are *already* in consumer hardware
   users own today, for the verifier-edge-inference case.
4. **One contrarian pick**: the platform you think the field is
   sleeping on for Boltzmann-sampling-class workloads. Justify.
5. **A red-team take**: what's the weakest claim or biggest hidden
   cost in the current Carnot hardware portfolio (RTX 3090 + Strix
   Point + KV260 + future Z1)? What would a hardware-savvy reviewer
   immediately flag?

## Output format

Long-form research report. Cite sources (vendor pages, papers,
independent benchmarks, GitHub repos). Flag uncertainty when vendor
claims and independent measurements disagree. Note when something
you'd expect to find isn't searchable (signal of vapor or NDA-only
access).

Length: as long as needed for thorough coverage. Don't pad.

Date your sources — claims from 2024 may be stale; prioritize 2025-2026
information.
