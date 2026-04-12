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
  - **What we have:**
    - `/opt/xilinx/xrt/` — XRT 2.20.0 driver stack ✅
    - `~/github.com/amd/RyzenAI-SW/` — includes `libonnxruntime_providers_vitisai.so`
      and `libonnxruntime_vitisai_ep.so` (built for onnxruntime 1.20.1) ✅
    - `.venv-npu/` — Python 3.12 venv with onnxruntime 1.20.1 (CPU only) ✅
    - `amdxdna` kernel module loaded ✅
    - ONNX model exported by Exp 146 (`results/jepa_predictor_146.onnx`) ✅
  - **What's missing:**
    - AMD's custom `onnxruntime` Python wheel with VitisAI EP compiled in
    - Download from: ryzenai.docs.amd.com/en/latest/inst.html (requires
      AMD account + EULA agreement)
    - Or build onnxruntime 1.20.1 from source with `-Donnxruntime_USE_VITISAI=ON`
  - **Status:** ONNX model ready, driver ready, Python 3.12 venv ready.
    Just need the VitisAI-enabled onnxruntime wheel to unlock NPU inference.
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

## Priority 4: Extropic TSU (When Available)

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
| Kria KV260 FPGA | $250 | High | TSU path, FPGA sampling prototype |
| 128GB DDR5 RAM | $300 | Medium | Larger models, batch benchmarks |
| ~~RX 7900 XTX GPU~~ | ~~$900~~ | ~~Very High~~ | **REPLACED: 2x RTX 3090 connected via CUDA** |
| ~~Kria KV260 FPGA~~ | $249 | High | **ORDERED — arriving in 4 business days** |
| Alveo U250 FPGA | $6,000 | Very High | Production-scale Ising, 256k p-bits |
| Extropic Z1 TSU | TBD | Transformative | Native thermodynamic sampling |
