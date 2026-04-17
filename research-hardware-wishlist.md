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

### KV260 Bring-Up Status (Exp 313 — 20260414)

- **Exp 313 result:** `honest_verdict=blocked_no_bitfile`
  - CARNOT_KV260_BITFILE env var not set on this machine
  - KV260 hardware should have arrived — bring-up blocked by missing bitfile path
  - CPU fallback latency measured: ≈358ms/call (JAX JIT first-compile overhead)
- **To resume bring-up on the KV260:**
  1. Boot KV260 to Ubuntu/PetaLinux image with PYNQ installed
  2. Build or flash the Carnot Ising bitfile (see `hardware/kv260/ising_sampler_v1.v`)
  3. Set `export CARNOT_KV260_BITFILE=/path/to/carnot_ising.bit`
  4. Run: `JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_313_kv260_bringup.py`
  5. Expected on real HW: `honest_verdict=hardware_working`, `mean_latency_us ≤ 100μs`
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
