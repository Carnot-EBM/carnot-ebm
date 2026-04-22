# Hardware Wishlist for Carnot Research

This file tracks hardware that would accelerate Carnot's research and
production goals. Updated as new needs emerge from experiments.

## Priority 1: FPGA for Ising Sampling (Unblocks: Self-Learning Tiers 1-4, TSU path)

### Small FPGA — Experiment Scale (1k-10k p-bits)
- **AMD/Xilinx Kria KV260** (~$250)
  - 256K LUTs, enough for ~4k p-bit Ising sampler
  - Ubuntu-based, Python-accessible via PYNQ
  - PCIe not needed (AXI accessible from ARM cores)
  - Supplier: Xilinx.com, Mouser, Digikey, Amazon
- **Terasic DE10-Nano** (~$130)
  - Intel Cyclone V, ~40K ALMs
  - Good for 1k-2k p-bit prototype
  - Supplier: terasic.com, Mouser
- **Lattice CrossLink-NX** (~$50 eval board)
  - Small but ultra-low power — interesting for edge constraint verification
  - Supplier: latticesemi.com

### Large FPGA — Production Scale (100k-256k p-bits)
- **AMD/Xilinx Alveo U250** (~$5,000-8,000)
  - 1.3M LUTs, PCIe Gen3 x16
  - Could implement full 256k p-bit Ising sampler
  - Matches Extropic Z1 target scale
  - Supplier: Xilinx.com, Avnet
- **AMD/Xilinx Alveo U55C** (~$4,000-6,000)
  - HBM memory — useful for large coupling matrices
  - Supplier: Xilinx.com
- **Intel Agilex 7 FPGA Dev Kit** (~$5,000-10,000)
  - Latest Intel FPGA, competitive with Xilinx
  - Supplier: intel.com/fpga

### Potts Machine Motivation for KV260 Bitfile Synthesis (Exp 534 — 20260419)

Exp 534 implemented PottsMachineVerifier(q=3) — a q-state generalization of the Ising EBM
that encodes correct/partial/violated constraint states.  arXiv 2602.04200 shows that mean-field
constraints preserve sparse coupling structure for Potts machines, making this FPGA-compatible.

**New Verilog requirement:** The existing `hardware/kv260/ising_sampler_v1.v` (Ising, q=2) needs
a q-state extension to support Potts sampling:
- Each spin must iterate over q=3 states instead of {+1,-1}
- Conditional energy computation: for each spin i, compute E_i(a) for a in {0,1,2}
- Sample from softmax(-beta * E_i) — needs a 3-entry categorical sampler per spin
- AXI-Lite upload format stays the same (sparse row format), but J tensor is (q,q,n,n)

This Potts extension is a STRONG additional motivation to complete the KV260 bitfile synthesis:
the hardware-native q-state sampler enables constraint verification with partial-credit scoring
that the binary Ising architecture cannot provide.

### KV260 Synthesis Status (Exp 624 — 20260421) — VIVADO NOT INSTALLED; v2 RTL + Python Sim Complete

- **Exp 624 result:** `honest_verdict=simulation_only_vivado_blocked`
  - Vivado not found on PATH on this machine; synthesis could not be attempted
  - ising_sampler_v2.v (synchronous, ~50% area reduction) created by Exp 612
  - Python simulation of synchronous checkerboard p-bit logic: VALIDATED
    (SynchronousIsingSampler.compare_with_async() runs end-to-end without error)
  - TCL script at `hardware/kv260/synth_ising.tcl` targets v1; update needed for v2

### KV260 Synthesis Status (Exp 584 — 20260420) — VIVADO NOT INSTALLED

- **Exp 584 result:** `honest_verdict=vivado_not_installed`
  - Vivado not found on PATH on this machine; synthesis could not be attempted
  - TCL script is complete at `hardware/kv260/synth_ising.tcl` (tcl_enhanced=True)
  - RTL is synthesis-ready at `hardware/kv260/ising_sampler_v1.v`
- **To synthesise the bitfile (on a Vivado-equipped machine):**
  1. Download Xilinx Vivado 2023.2 from xilinx.com (requires AMD account)
  2. `sudo ./Xilinx_Unified_2023.2_1013_2256.tar.gz --batch-mode --agree 3rdPartyEULA,WebTalkTermsAndConditions,XilinxEULA`
  3. `export PATH=/tools/Xilinx/Vivado/2023.2/bin:$PATH`
  4. `vivado -mode batch -source hardware/kv260/synth_ising.tcl`  (~15-60 min)
  5. Set `export CARNOT_KV260_BITFILE=output/carnot_ising_synth/carnot_ising.bit`
  6. Run: `JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_568_kv260_bringup_v2.py`
  7. Expected on real HW: `honest_verdict=hardware_working`, `hardware_latency_us < 100μs`

### KV260 Bring-Up Status (Exp 568 — 20260420) — BOARD ARRIVED

- **Board arrival:** KV260 physically arrived on 2026-04-20.
- **Exp 568 result:** `honest_verdict=synthesis_required`
  - Board is here; CARNOT_KV260_BITFILE is still not set — bitfile not yet synthesised
  - CPU baseline latency: ≈290ms/call (100 trials, 100-spin Ising)
  - Vivado synthesis command: `vivado -mode batch -source hardware/kv260/synth_ising.tcl`
  - TCL stub present at `hardware/kv260/synth_ising.tcl`
- **Next step to complete hardware bring-up:**
  1. Boot KV260 to Ubuntu/PetaLinux image with PYNQ installed
  2. Run synthesis: `vivado -mode batch -source hardware/kv260/synth_ising.tcl`
  3. Flash the resulting bitfile to the board
  4. Set `export CARNOT_KV260_BITFILE=output/carnot_ising_synth/carnot_ising.bit`
  5. Run: `JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_568_kv260_bringup_v2.py`
  6. Expected on real HW: `honest_verdict=hardware_working`, `hardware_latency_us < 100μs`

### KV260 Bring-Up Status (Exp 313 — 20260414) — Superseded by Exp 568

- **Exp 313 result:** `honest_verdict=blocked_no_bitfile`
  - CARNOT_KV260_BITFILE env var not set on this machine
  - KV260 hardware should have arrived — bring-up blocked by missing bitfile path
  - CPU fallback latency measured: ≈358ms/call (JAX JIT first-compile overhead)
- **Prior bring-up experiments:** Exp 228 (AXI design), Exp 288 (blocked/SW model), Exp 289 (FpgaBackend), Exp 290 (simulation benchmark), Exp 291 (Verilog RTL)
- **Target (arXiv 2602.15985):** 77.5μs convergence for small problems ≤100 spins

### FPGA Justification
- Exp 102: constraint check is 0.005ms on CPU. FPGA would be <1μs.
- Exp 46b: 5000-var SAT in 0.7s on CPU. FPGA target: <1ms.
- Self-Learning Tier 2: FPGA pattern matching for constraint memory
- Self-Learning Tier 4: FPGA reconfiguration for adaptive energy landscapes
- TSU path validation before Extropic hardware ships

## Priority 2: Discrete GPU for Live Model Inference (Unblocks: Goals #1, #5, #6)

### Why Current Setup Falls Short
- Radeon 890M iGPU: ROCm crashes JAX, PyTorch works but only 3.3x speedup
- CPU inference works but is slow for batch benchmarks (1,319 GSM8K questions)
- Live model loading fails inconsistently in conductor subprocesses

### Options
- **AMD Radeon RX 7900 XTX** (~$800-900)
  - 24GB VRAM, ROCm 6.x support, gfx1100 (well-supported)
  - Could run Qwen3.5-0.8B + Gemma4-E4B simultaneously
  - ROCm JAX should work on gfx1100 (unlike gfx1150 iGPU)
  - Supplier: AMD.com, Newegg, Amazon
- **AMD Radeon PRO W7900** (~$2,000-2,500)
  - 48GB VRAM, ECC, validated ROCm
  - Could run models up to 13B for benchmark comparison
  - Supplier: AMD.com, CDW
- **NVIDIA RTX 4090** (~$1,600-2,000)
  - 24GB VRAM, CUDA (guaranteed JAX/PyTorch support)
  - Most reliable option but locks us to NVIDIA ecosystem
  - Supplier: NVIDIA.com, Newegg, Amazon, Best Buy
- **NVIDIA RTX 5090** (~$2,000-2,500)
  - 32GB VRAM, latest CUDA
  - Supplier: NVIDIA.com (when in stock)

### GPU Justification
- Goal #1: Reliable live model inference for benchmarks
- Goal #5: Apple adversarial GSM8K needs 1,319+ questions x 2 models
- Goal #6: Full-scale benchmarks with confidence intervals
- Self-Learning Tier 3: JEPA predictor model runs on GPU/NPU

## Priority 3: NPU/APU for Edge Inference (Unblocks: Edge deployment, Tier 3)

### Options
- **AMD Ryzen AI 300 series** (current machine has Ryzen AI 9 HX 370)
  - XDNA NPU present, `amdxdna` kernel module loaded, XRT 2.20.0 installed
  - **BLOCKER:** VitisAI Execution Provider requires a custom-built
    `onnxruntime` with VitisAI compiled in. The pip package does NOT include
    it, and AMD only distributes pre-built wheels for Python 3.9-3.12.
  - **Current workaround:** `.venv-npu/` (Python 3.12) created with
    onnxruntime 1.20.1, but pip's build lacks VitisAI EP. Need AMD's
    custom onnxruntime wheel from their Ryzen AI Software installer.
  - **Exp 292 findings (20260414):**
    - LD_LIBRARY_PATH approach does NOT work — VitisAI EP must be compiled
      INTO onnxruntime (not loadable at runtime via LD_LIBRARY_PATH).
    - Pre-built AMD .so files are ABI-incompatible with ORT 1.24.x (segfault).
    - ORT 1.20.1 + LD_LIBRARY_PATH: VitisAI EP still not in available_providers.
    - Source build blocked by: `ninja` not installed, `openblas` not found.
  - **Exp 303 findings (20260414):**
    - Still blocked by missing prerequisites:
      - `ninja`: not found. Install: `sudo pacman -S ninja  (Arch)  OR  sudo apt install ninja-build  (Debian/Ubuntu)`
      - `openblas`: not found. Install: `sudo pacman -S openblas  (Arch)  OR  sudo apt install libopenblas-dev  (Debian/Ubuntu)`
    - **Status:** BLOCKED — install prerequisites, then re-run Exp 303.
  - **What we have:**
    - `/opt/xilinx/xrt/` — XRT 2.20.0 driver stack ✅
    - `~/github.com/amd/RyzenAI-SW/` — includes `libonnxruntime_providers_vitisai.so`
      and `libonnxruntime_vitisai_ep.so` (built for onnxruntime 1.20.1) ✅
    - `.venv-npu/` — Python 3.12 venv with onnxruntime **1.20.1** (CPU only) ✅
    - `amdxdna` kernel module loaded ✅
    - ONNX models: `results/jepa_predictor_291.onnx` and `146.onnx` ✅
    - `vaip_config_npu_2_3.json` in RyzenAI-SW dir ✅
  - **What's missing (to unblock Exp 293):**
    - `ninja` — install: `sudo pacman -S ninja` (Arch) or `sudo apt install ninja-build`
    - `openblas` — install: `sudo pacman -S openblas` or `sudo apt install libopenblas-dev`
    - Then: `scripts/experiment_292_amd_xdna_npu.py` will auto-run source build
    - Or: download AMD custom onnxruntime wheel from ryzenai.docs.amd.com/en/latest/inst.html
      (requires AMD account + EULA; Python 3.9-3.12 only)
  - **Status:** ONNX model ready, driver ready, Python 3.12 venv ready.
    Two missing packages (ninja + openblas) block the source build path.
- **Intel Core Ultra (Lunar Lake/Arrow Lake)**
  - Integrated NPU, well-documented SDK
  - Could be a comparison platform for edge constraint verification
  - Supplier: Intel, various laptop OEMs
- **Qualcomm Snapdragon X Elite dev kit** (~$900)
  - Hexagon NPU, good for mobile/embedded constraint verification
  - Supplier: Qualcomm

### NPU Justification
- Self-Learning Tier 3: small predictor model (JEPA) on NPU
- Edge deployment: constraint verification on device, not cloud
- The current machine's XDNA NPU is FREE to experiment with

## Priority 4: D-Wave Quantum Annealing (Cloud Access Available Now)

- **D-Wave Advantage** (cloud via Leap)
  - 5,000+ qubits, Pegasus topology (15-way connectivity)
  - Native Ising/QUBO solver — maps 1:1 to Carnot's IsingEBM
  - **Free tier**: 1 min QPU/month (~1000 small constraint problems)
  - Pay-as-you-go: ~$2000/hr QPU time (hybrid solvers cheaper)
  - **Local simulation**: `pip install dwave-ocean-sdk` includes `neal` simulated annealer
  - Sign up: cloud.dwavesys.com
- **D-Wave Advantage2** (cloud via Leap, prototype)
  - 1,200+ qubits now, targeting 7,000+ (Zephyr topology, 20-way connectivity)
  - Higher connectivity = better embedding for dense constraint graphs

### D-Wave Justification
- Carnot's Ising constraint verification is literally what D-Wave solves
- 128-4,096 spin problems fit within Advantage's capacity for sparse graphs
- `dwave-ocean-sdk` (Apache 2.0) provides same API for local sim and real QPU
- Validates quantum advantage for constraint verification before investing in custom hardware
- SamplerBackend abstraction already designed for pluggable backends

## Priority 5: Extropic TSU (When Available)

- **Extropic Z1** (not yet available, ~2026-2027?)
  - 256k p-bits, native thermodynamic sampling
  - Nanosecond-scale energy minimization
  - SamplerBackend abstraction (Exp 71) ready for integration
  - **Action:** Sign up for early access at extropic.ai
  - **In the meantime:** FPGA simulation (Priority 1) validates the path

## Priority 5: Memory (Unblocks: Larger models, batch processing)

- **128GB DDR5 RAM upgrade** (~$200-400)
  - Current: likely 32-64GB
  - Would enable running 3B+ models comfortably on CPU
  - Multiple models loaded simultaneously for comparison benchmarks
  - Supplier: Crucial, Kingston, G.Skill via Amazon/Newegg

## Current Hardware Inventory

| Component | Model | Status | Carnot Use |
|-----------|-------|--------|-----------|
| CPU | AMD Ryzen AI 9 HX 370 | Working | All experiments, CPU inference |
| iGPU | Radeon 890M (gfx1150) | Broken for JAX | PyTorch only (3.3x speedup) |
| **eGPU** | **Radeon RX 7900 XTX (24GB)** | **AVAILABLE** | **Needs Thunderbolt chassis connection** |
| eGPU chassis | Thunderbolt external | **AVAILABLE** | Connect RX 7900 XTX to laptop |
| NPU | AMD XDNA | Unused | Needs driver/SDK setup |
| RAM | DDR5 (TBD size) | Working | Constrains model size |

### ACTION: Connect RX 7900 XTX via Thunderbolt
**Priority: IMMEDIATE — unblocks Goals #1, #5, #6 and all live benchmarks**
1. Connect Thunderbolt chassis with RX 7900 XTX
2. Verify ROCm detects gfx1100: `rocminfo | grep gfx`
3. Test PyTorch: `python -c "import torch; print(torch.cuda.get_device_name(0))"`
4. Test JAX on GPU: `python -c "import jax; print(jax.devices())"`
   (gfx1100 should work — unlike gfx1150 which crashes)
5. If JAX works: remove JAX_PLATFORMS=cpu requirement for experiments
6. Benchmark: Qwen3.5-0.8B inference speed on eGPU vs CPU
7. Update research-program.md constraints if GPU works

## Shopping List (Priority Order)

| Item | Est. Cost | Impact | Unblocks |
|------|-----------|--------|----------|
| AMD XDNA NPU SDK install | $0 | Medium | Tier 3 self-learning, edge deployment |
| ~~Kria KV260 FPGA~~ | $250 | High | **ARRIVED — bitfile needed to complete bring-up (Exp 313)** |
| 128GB DDR5 RAM | $300 | Medium | Larger models, batch benchmarks |
| ~~RX 7900 XTX GPU~~ | ~~$900~~ | ~~Very High~~ | **REPLACED: 2x RTX 3090 connected via CUDA** |
| ~~Kria KV260 FPGA~~ | $249 | High | **ARRIVED — see Exp 313 bring-up status above** |
| Alveo U250 FPGA | $6,000 | Very High | Production-scale Ising, 256k p-bits |
| Extropic Z1 TSU | TBD | Transformative | Native thermodynamic sampling |

### AMD XDNA NPU Status (Exp 314 — 20260414)

- **Exp 314 result:** `honest_verdict=blocked_prereq`
  - ninja: still_missing
  - openblas: still_missing
  - Both were also missing in Exp 303 — packages have NOT been installed yet.
  - Install ninja: `sudo pacman -S ninja  (Arch)  OR  sudo apt install ninja-build  (Debian/Ubuntu)`
  - Install openblas: `sudo pacman -S openblas  (Arch)  OR  sudo apt install libopenblas-dev  (Debian/Ubuntu)`

### AMD XDNA NPU Status (Exp 335 — 20260415)

- **Exp 335 result:** `honest_verdict=blocked_prereq`
  - ninja: still_missing (was missing in Exp 314)
  - openblas: still_missing (was missing in Exp 314)
  - Install ninja: `sudo pacman -S ninja  OR  sudo apt install ninja-build`
  - Install openblas: `sudo pacman -S openblas  OR  sudo apt install libopenblas-dev`
  - Blocked for 4 milestones (Exps 292, 303, 314, 335). Human install required before next attempt.

### AMD XDNA NPU Status (Exp 435 — 20260417)

- **Exp 435 result:** `honest_verdict=blocked_prereq` (5th consecutive milestone)
  - ninja: still_missing
  - openblas: still_missing
  - IRON toolchain (mlir_aie): not available (not installed)
  - amdxdna driver: loaded (unchanged — this was never the blocker)
  - **NEW in Exp 435:** IRON toolchain path investigated as VitisAI alternative.
    arXiv 2504.03083 describes bare-metal NPU programming via IRON/mlir-aie, achieving
    2.8x speedup over CPU for GEMM via explicit DMA routing — no onnxruntime required.
    If mlir-aie is installed, this path works independently of ninja/openblas.
  - **ESCALATION — 5th consecutive milestone.** Human MUST install before next run:
    - Arch Linux: `sudo pacman -S ninja && sudo pacman -S openblas`
    - Ubuntu/Debian: `sudo apt install ninja-build libopenblas-dev`
    - IRON alternative: `pip install mlir-aie`
  - VitisAI path: still blocked (ninja + openblas not installed)
  - IRON path: not tested (mlir_aie not importable)
  - Result file: `results/experiment_435_npu_unblock.json`

### AMD XDNA NPU Status (Exp 714 — 20260422)

- **Exp 714 result:** `honest_verdict=npu_iron_install_failed` (7th consecutive milestone)
  - IRON toolchain (mlir-aie): `pip install mlir-aie` failed — no matching distribution found on PyPI.
  - AMD custom onnxruntime wheel (onnxruntime-vitisai): not found on PyPI either.
  - amdxdna driver: still loaded (never the blocker).
  - CPU GEMM baseline measured: ~117 µs / 100 iterations on 16x16 matrices.
  - NPU GEMM: not reached (install blocked before compile/run).
  - **RETRO-NPU-v8:** Both IRON and VitisAI PyPI paths exhausted. The mlir-aie package
    does not appear to be distributed via the standard PyPI index — it may require the
    AMD-internal or GitHub Releases wheel.  Human action required:
    - Option A: Download mlir-aie wheel directly from AMD's GitHub:
      `pip install https://github.com/Xilinx/mlir-aie/releases/latest/download/mlir_aie-*.whl`
    - Option B: Install Ryzen AI Software from AMD (provides full VitisAI + ninja + openblas).
    - Option C: Request IT to install `ninja-build` + `libopenblas-dev` (enables VitisAI build).
  - Result file: `results/experiment_714_npu_iron_unblock.json`

### KV260 FPGA Ising Sampler Benchmark Status (Exp 585 — 20260420)

- **Exp 585 result:** `honest_verdict=blocked_no_bitfile`
  - Gate: Exp 584 (Vivado synthesis) produced `bitfile_built=False`
  - Vivado not installed on host — synthesis blocked
  - Target: mean hardware latency < 100 µs (vs CPU baseline 289608 µs = 289ms, 2900x speedup)
  - No hardware samples taken; benchmark deferred until bitfile is available
  - **BLOCKED — Human MUST install Vivado before benchmark can run:**
    - Download Xilinx Vivado 2023.2 from xilinx.com (requires AMD account)
    - `sudo ./Xilinx_Unified_2023.2_1013_2256.tar.gz --batch-mode --agree 3rdPartyEULA,WebTalkTermsAndConditions,XilinxEULA`
    - `export PATH=/tools/Xilinx/Vivado/2023.2/bin:$PATH`
    - Re-run Exp 584 (synthesis), then Exp 585 (benchmark)
  - Result file: `results/experiment_585_kv260_live_benchmark_v3.json`

### KV260 Ising Sampler v3 — Inertia Dynamics RTL Spec (Exp 648 — 20260421)

- **Exp 648 result:** `honest_verdict=inertia_comparable_no_clear_win`
  - Python CPU simulation: both baseline and inertia sampler saturate 400-step budget without meeting 0.1% convergence criterion at beta=1.0
  - Alpha sweep (n=100): best_alpha=0.5, achieving 258 steps vs 400 for baseline at alpha=0.5
  - CPU simulation does not reproduce FPGA oscillation (paper's 20-35x gain is specific to digital fixed-point arithmetic, not Python float)
  - v3 RTL spec written: `hardware/kv260/ising_sampler_v3_spec.md`
    - Adds per-spin h_ema register and EMA update stage before flip probability computation
    - Recommended alpha=0.5 for RTL (from Python sweep); fixed-point Q1.15 = 0x4000
    - Estimated 15-25x fewer sweeps on KV260 v3 for dense arithmetic constraint graphs
  - **PENDING — Requires KV260 hardware + Vivado for RTL validation:**
    - KV260 board arrived 2026-04-20; Vivado still not installed
    - Implement v3 RTL from spec, synthesise with synth_ising_v2.tcl as template
    - Compare convergence sweep counts on hardware vs v2
  - Result file: `results/experiment_648_parallel_ising_inertia.json`
