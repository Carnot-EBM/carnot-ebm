# KV260 Ising Sampler — Hardware RTL (Exp 291)

128-spin Ising p-bit sampler for the AMD Kria KV260 FPGA.
Synthesizable Verilog RTL with AXI-Lite host interface.

## Files

| File | Description |
|------|-------------|
| `ising_sampler_v1.v` | Top-level module + LFSR submodule |

## Port List

| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `S_AXI_ACLK` | in | 1 | AXI-Lite clock |
| `S_AXI_ARESETN` | in | 1 | AXI-Lite reset (active-low) |
| `S_AXI_AWADDR` | in | 16 | Write address |
| `S_AXI_AWPROT` | in | 3 | Write protection (ignored) |
| `S_AXI_AWVALID` | in | 1 | Write address valid |
| `S_AXI_AWREADY` | out | 1 | Write address ready |
| `S_AXI_WDATA` | in | 32 | Write data |
| `S_AXI_WSTRB` | in | 4 | Write byte strobe |
| `S_AXI_WVALID` | in | 1 | Write data valid |
| `S_AXI_WREADY` | out | 1 | Write data ready |
| `S_AXI_BRESP` | out | 2 | Write response (00=OKAY) |
| `S_AXI_BVALID` | out | 1 | Write response valid |
| `S_AXI_BREADY` | in | 1 | Write response ready |
| `S_AXI_ARADDR` | in | 16 | Read address |
| `S_AXI_ARPROT` | in | 3 | Read protection (ignored) |
| `S_AXI_ARVALID` | in | 1 | Read address valid |
| `S_AXI_ARREADY` | out | 1 | Read address ready |
| `S_AXI_RDATA` | out | 32 | Read data |
| `S_AXI_RRESP` | out | 2 | Read response (00=OKAY) |
| `S_AXI_RVALID` | out | 1 | Read data valid |
| `S_AXI_RREADY` | in | 1 | Read data ready |

## AXI-Lite Register Map

All addresses are byte addresses; all registers are 32-bit words.

### Control Registers

| Address | Name | Access | Description |
|---------|------|--------|-------------|
| `0x0000` | `CONTROL` | R/W | Bit 0 = `START` (pulse high to begin sampling); Bit 1 = `RESET` (return to IDLE) |
| `0x0004` | `STATUS` | R | Bit 0 = `READY`; Bit 1 = `BUSY`; Bit 2 = `DONE` |
| `0x0008` | `SPIN_COUNT` | R/W | Active spin count (1–128) |
| `0x001C` | `BETA_FINAL` | R/W | Final inverse temperature in Q8.8 format (e.g., 5.0 → 1280 = `0x0500`) |

### Memory Windows

| Address Range | Name | Format | Description |
|---------------|------|--------|-------------|
| `0x1000 + 4*i` | `bias_ram[i]` | Q8.8 signed 16-bit | Bias h_i for spin i (i = 0..SPIN_COUNT−1) |
| `0x2000 + 4*(i*32+k)` | `adj_ram[i][k]` | Signed 16-bit | Neighbour index for spin i, slot k; `0xFFFF` (−1) = empty slot |
| `0x4000 + 4*(i*32+k)` | `coupl_ram[i][k]` | Q8.8 signed 16-bit | Coupling weight J_{i, adj[i][k]} |
| `0x8010 + 4*w` | `spin_out[w]` | Packed bits | 32 spins per word; bit b of word w = spin (32w+b), where +1→1 and −1→0 |

### Q8.8 Fixed-Point Encoding

Q8.8 means 8 integer bits + 8 fractional bits in a signed 16-bit integer.

- To encode a float `f`: `q88 = int(round(f * 256))`, clipped to `[-32768, 32767]`
- To decode `q88` back to float: `f = q88 / 256.0`
- Range: `[-128.0, +127.996]`, resolution `1/256 ≈ 0.0039`

**Example:** β = 5.0 → `int(5.0 * 256) = 1280 = 0x0500`

## Annealing Schedule

The β schedule follows arXiv 2604.04606 with Mpemba initialization (arXiv 2603.24183):

1. **Mpemba hot-start** (first 10% of `N_STEPS`): β = 0 (infinite temperature).
   All spins update with 50% probability, erasing initialization bias.
2. **Log-linear ramp** (remaining 90% of `N_STEPS`):
   β(t) = β_min × (β_max / β_min)^(t / T)

   In v1 RTL this is approximated as a linear ramp for simplicity; a future v2
   will use a log-linear ROM for exact geometric interpolation.

Parameters:
- `N_STEPS = 1000` (synthesized constant, configurable as Verilog parameter)
- `N_HOT = N_STEPS / 10 = 100`
- `BETA_MIN = 0.1` (hardcoded Q8.8: 26)
- `BETA_FINAL` = from register `0x001C`

## Gibbs Update Pipeline

For each spin i in checkerboard order (even spins first, then odd spins):

1. Load `h_eff = bias_ram[i]`
2. Walk `adj_ram[i][0..31]` (stop at first `−1` sentinel):
   `h_eff += coupl_ram[i][k] * (spin[adj[i][k]] == 1 ? +1 : -1)`
3. Compute sigmoid LUT index from `2 * β * h_eff`
4. Threshold LFSR against `sigmoid_lut[index]`
5. Write new spin bit to `state_ram`

All arithmetic is Q8.8 with saturation. The LFSR advances every clock.

## LFSR Specification

16-bit Fibonacci LFSR, polynomial x^16 + x^14 + x^13 + x^11 + 1.

- Reset seed: `0xACE1`
- Period: 2^16 − 1 = 65,535 (maximal length)
- Taps (0-indexed): bits 15, 13, 12, 10

## Synthesis Steps (Vivado 2023.x)

```tcl
# 1. Create project targeting KV260 (xck26-sfvc784-2LV-c)
create_project ising_sampler_v1 ./vivado_proj -part xck26-sfvc784-2LV-c

# 2. Add sources
add_files hardware/kv260/ising_sampler_v1.v

# 3. Set top module
set_property top ising_sampler_128 [current_fileset]

# 4. Run synthesis
launch_runs synth_1 -jobs 4
wait_on_run synth_1

# 5. Run implementation
launch_runs impl_1 -jobs 4
wait_on_run impl_1

# 6. Generate bitstream
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1

# 7. Export bitfile for PYNQ
# Bitfile will be at: vivado_proj/ising_sampler_v1.runs/impl_1/ising_sampler_128.bit
# Copy to KV260 and set: export CARNOT_KV260_BITFILE=/path/to/ising_sampler_128.bit
```

### Expected Resource Usage (128 spins, max_degree=32)

| Resource | Estimated | Description |
|----------|-----------|-------------|
| LUT | ~2000 | Control logic + sigmoid LUT |
| FF | ~500 | FSM + accumulator registers |
| BRAM18 | 9 | bias(1) + adj(4) + coupl(4) |
| DSP | 2 | Q8.8 multiplier |

## Host-Side Usage (PYNQ)

```python
import os
from pynq import Overlay

# Load bitfile (produced by Vivado synthesis)
overlay = Overlay(os.environ["CARNOT_KV260_BITFILE"])
mmio = overlay.ising_sampler_128_0.mmio  # AXI-Lite IP name in block design

# Write bias (example: all zeros, 128 spins)
for i in range(128):
    mmio.write(0x1000 + 4*i, 0)

# Write adjacency and couplings (ring topology)
for i in range(128):
    mmio.write(0x2000 + 4*(i*32 + 0), (i+1) % 128)   # right neighbour
    mmio.write(0x4000 + 4*(i*32 + 0), 0x0100)          # J=1.0 in Q8.8
    mmio.write(0x2000 + 4*(i*32 + 1), (i-1) % 128)   # left neighbour
    mmio.write(0x4000 + 4*(i*32 + 1), 0x0100)
    mmio.write(0x2000 + 4*(i*32 + 2), 0xFFFF)          # sentinel (-1)

# Set beta_final = 5.0 (Q8.8 = 1280 = 0x500)
mmio.write(0x001C, 0x0500)

# Start sampling
mmio.write(0x0000, 0x1)  # CONTROL.START

# Poll STATUS until DONE (bit 2)
import time
while not (mmio.read(0x0004) & 0x4):
    time.sleep(0.001)

# Read packed spin output (4 words × 32 spins = 128 spins)
spins_packed = [mmio.read(0x8010 + 4*w) for w in range(4)]
# Unpack: bit b of word w = spin (32*w + b); +1 if bit=1, -1 if bit=0
spins = []
for w, word in enumerate(spins_packed):
    for b in range(32):
        spins.append(1 if (word >> b) & 1 else -1)
```

## Python Behavioral Simulation

`scripts/simulate_ising_sampler.py` implements identical logic in pure
numpy/Python for test validation without physical FPGA:

```python
from scripts.simulate_ising_sampler import IsingSimulator, _build_ring_problem

sim = IsingSimulator(n_spins=128, max_degree=32, n_steps=1000,
                     beta_min=0.1, beta_max=5.0)
j, h = _build_ring_problem(128)
sim.load_problem(j, h)
spins = sim.run()
print("Energy:", sim.compute_energy())
```

Tests for the RTL logic are in `tests/python/test_ising_sampler_rtl.py`.
Run with: `JAX_PLATFORMS=cpu .venv/bin/pytest tests/python/test_ising_sampler_rtl.py -v`

## Design History

| Experiment | Date | Description |
|-----------|------|-------------|
| Exp 228 | 2026-04-13 | 4096-spin sparse Ising sampler architecture design |
| Exp 288 | 2026-04-14 | KV260 bring-up attempt (blocked: no bitfile) |
| Exp 289 | 2026-04-14 | FpgaBackend Python SamplerBackend + Q8.8 AXI map |
| Exp 291 | 2026-04-14 | This file — 128-spin Verilog RTL prototype |

## Known Limitations (v1)

1. **Linear β approximation**: The ramp phase uses linear increments rather
   than the exact log-linear (geometric) schedule from arXiv 2604.04606.
   Impact: slightly sub-optimal convergence; fixable in v2 with a ROM-based
   schedule or CORDIC log computation.

2. **Sequential pipeline**: One spin is updated per clock group (edge-walk
   loop). A parallel implementation (all even spins simultaneously) would
   require `N_SPINS/2` accumulator pipelines and would be ~64× faster.

3. **N_STEPS is a synthesis-time constant**: Runtime step count requires
   adding a `WARMUP_STEPS` register to the AXI map (trivial extension).

4. **No sample batching**: The v1 produces one sample per run. Batching
   requires adding a sample counter and a multi-word result FIFO.
