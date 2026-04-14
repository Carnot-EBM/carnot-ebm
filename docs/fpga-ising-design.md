# FPGA Ising Sampler Design

## Goal

Target a Kria KV260-class FPGA backend for dynamic Ising sampling with:

- up to **4096 spins**
- runtime coupling uploads rather than fixed-at-synthesis weights
- a PYNQ-accessible AXI-Lite control plane
- a software model that exercises the same upload, trigger, and readback path

The hardware design is intentionally **sparse**. A dense 4096×4096 coupling
matrix would require over 16 million weights and is not a realistic fit for
the KV260 control/data memory budget.

## Array Architecture

The proposed p-bit array is:

- **32 tiles × 128 spins/tile = 4096 spins**
- **Q8.8** fixed-point biases and couplings
- **max_degree = 32** directed neighbors per spin
- **even/odd update phases** to stay compatible with Carnot's existing
  checkerboard-style parallel Ising approximation

Each tile owns:

- a 128-bit state RAM
- 128 bias words
- a sparse row-pointer slice for its spins
- edge records for local and inter-tile neighbors
- a tile-local accumulator / probability / RNG update pipeline

### Why 128-spin tiles

- 128 spins is large enough to amortize control overhead
- 32 tiles maps cleanly onto a 4096-spin target
- tile-local state is tiny, so routing pressure is dominated by sparse edge
  fetch and accumulation rather than state storage

## Sparse Memory Budget

At the current software-model contract:

- bias RAM: `4096 * 16 bits = 8 KiB`
- row pointers: `4097 * 32 bits ≈ 16 KiB`
- edge store: `4096 * 32 * 32 bits = 512 KiB` worst case at `max_degree=32`
- packed sample readback: `4096 / 32 = 128` words per sample

That edge-store budget is tight but plausible for a KV260 when split across
BRAM and URAM. The design therefore treats `max_degree=32` as a hard control
plane contract for the first overlay.

## Update Pipeline

One full sweep is split into two global phases:

1. Update all even-indexed spins
2. Update all odd-indexed spins

For each active spin:

1. Read current spin state and bias
2. Walk the sparse edge list for that row
3. Accumulate `bias + Σ(weight * neighbor_state)`
4. Convert the field to a flip probability via a LUT
5. Compare against an LFSR-based random number
6. Write the new bit back into the state RAM

This is the hardware analogue of Carnot's current parallel/checkerboard CPU
sampler. The first FPGA target is therefore a control-plane-compatible
accelerator, not a mathematically different sampler.

## Verilog Module Plan

Recommended RTL partition:

- `carnot_ising_top`
  AXI-Lite slave, global scheduler, status/result mux
- `control_regs`
  Control/status/config registers
- `bias_bram`
  Bias window storage
- `row_ptr_bram`
  CSR row pointers
- `edge_bram`
  Packed edge records
- `state_ram`
  Current spin bits
- `pbit_tile`
  Tile-local update datapath
- `field_accumulator`
  Sparse neighbor accumulation
- `probability_lut`
  Fixed-point sigmoid/tanh approximation
- `rng_lfsr`
  Per-lane pseudo-random source
- `sample_packer`
  Packs 32 spins per readback word

## AXI-Lite Register Map

Control registers:

| Address | Name | Description |
|---------|------|-------------|
| `0x0000` | `CONTROL` | Bit 0 = `START`, bit 1 = `RESET`, bit 2 = `CLEAR_RESULTS` |
| `0x0004` | `STATUS` | Bit 0 = `READY`, bit 1 = `BUSY`, bit 2 = `DONE`, bit 3 = `ERROR` |
| `0x0008` | `SPIN_COUNT` | Active spin count (`<= 4096`) |
| `0x000C` | `SAMPLE_COUNT` | Number of samples to emit |
| `0x0010` | `WARMUP_STEPS` | Anneal/warmup sweeps before collection |
| `0x0014` | `STEPS_PER_SAMPLE` | Decorrelation sweeps between emitted samples |
| `0x0018` | `BETA_INIT` | Initial inverse temperature, Q8.8 |
| `0x001C` | `BETA_FINAL` | Final inverse temperature, Q8.8 |
| `0x0020` | `RUN_FLAGS` | Bit 0 = `RUN_MINIMIZE` |

Memory windows:

| Address Range | Name | Format |
|---------------|------|--------|
| `0x1000 + 4*i` | Bias window | one Q8.8 signed word per spin |
| `0x2000 + 4*i` | Row-pointer window | CSR row pointers, `n_spins + 1` entries |
| `0x4000 + 4*i` | Edge window | packed edge records |
| `0x8010 + 4*i` | Sample window | 32 spins packed per 32-bit word |

Packed edge format:

- bits `[11:0]`: neighbor index
- bits `[15:12]`: reserved
- bits `[31:16]`: coupling weight in signed Q8.8

## Python Control Plane

`python/carnot/samplers/fpga_ising.py` implements the control plane in three
modes:

- `mode="hardware"`
  Bind to `pynq.Overlay(...).carnot_ising_0.mmio` when available
- `mode="software"`
  Use `SoftwareFPGAOverlay`, which executes the same MMIO contract in-process
- `mode="auto"`
  Try hardware first, otherwise fall back to CPU

`FPGAIsingSampler` implements the existing `SamplerBackend` protocol and uses
the same sparse upload path for both hardware and software transports.

## Benchmark Note

The current benchmark is intentionally a **software-model benchmark**, not a
hardware-performance claim.

Measured on the local CPU host for a sparse **128-spin** problem with
`n_samples=16`, `n_steps=100`, `beta=6.0`:

- `fpga_sim`: **0.824549 s**
- `cpu`: **0.288092 s**

Interpretation:

- the software model is slower because it still computes on the CPU and adds
  MMIO-style packing/unpacking overhead
- these numbers validate the control path and artifact contract
- they are **not** a proxy for synthesized FPGA throughput

The real success criterion for the first overlay is not beating the CPU in the
Python software model. It is preserving the host/overlay contract so the same
backend can switch from `fpga_sim` to real PYNQ MMIO once the bitstream exists.

## Bring-Up History

### Exp 242 (2026-04-13) — Blocked: no bitfile configured

Result: `blocked` — `CARNOT_KV260_BITFILE` was not set.  No overlay load
was attempted.  See `results/experiment_242_results.json`.

### Exp 288 (2026-04-14) — Bring-up attempt with 60 s hard timeout

Script: `scripts/experiment_288_kv260_bringup.py`

The script:
1. Checks `CARNOT_KV260_BITFILE` first; emits blocked immediately if unset.
2. Attempts `pynq.Overlay(bitfile)` within a 60-second wall-clock timeout.
3. Exercises the AXI-Lite register map: writes CONTROL, reads STATUS.
4. Uploads the sparse ring coupling matrix, triggers sampling, reads back
   packed spin words.
5. Converts boolean spins to ±1 and records `spin_state_valid`.
6. Labels the path as `hardware`, `software_model`, or `blocked`.

Result: `blocked` — `CARNOT_KV260_BITFILE` was not set on the development
host (KV260 not yet networked to the build machine).
See `results/experiment_288_results.json`.

## Next Hardware Step

1. Connect the KV260 to the build network and set `CARNOT_KV260_BITFILE` to
   the synthesized bitstream path.
2. Synthesize `carnot_ising_top` with the register map above and export it as
   `carnot_ising_0` in the PYNQ overlay.
3. Re-run `scripts/experiment_288_kv260_bringup.py` — it will record
   `overlay_load_ms`, `register_roundtrip_us`, and `spin_state_valid` on first
   successful hardware path.
4. Replace software-model timing in this doc with on-board throughput numbers.
