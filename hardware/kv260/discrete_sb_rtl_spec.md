# Discrete SB KV260 RTL Specification

Spec refs: REQ-ISING-023, SCENARIO-ISING-033.

Run date: 20260506
Status: specification and synthesis plan only.

## Source Evidence

Exp 1399 measured a CPU-only Discrete Simulated Bifurcation convergence speedup
of 1.38462x against the Gibbs baseline.  Its KV260 evidence was arithmetic
only: the dense int8 coupling matrix and one update unit fit estimated BRAM/LUT
budgets, while Vivado synthesis, bitfile generation, and KV260 board execution
were not performed.

## Datapath

The RTL target is a single-update-unit dSB datapath for N=256.  The
unit preserves the Exp 1399 update rule:

```text
x_i(t+1) = sign(x_i(t) + eta * sum_j J_ij * x_j(t) - pressure(t))
```

The proposed pipeline is:

1. Snapshot the current spin/position vector into `spin_cur`.
2. For each row `i`, stream `J[i, 0..N-1]` from dense BRAM and accumulate
   `field_i = sum_j J_ij * spin_cur[j]` in a signed fixed-point accumulator.
3. Compute `candidate_i = spin_cur[i] + eta * field_i - pressure`.
4. Write `spin_next[i] = +1` when `candidate_i >= 0`, otherwise `-1`.
5. Commit `spin_next` to `spin_cur` only after all N rows have been evaluated.

The row-serial schedule keeps the LUT estimate close to Exp 1399's one-update
unit estimate.  Parallel row units are a later optimization and must be
synthesized separately because they multiply the accumulator and memory-port
pressure.

## Memory Layout

The dense coupling matrix is row-major signed int8:

```text
J_MATRIX[row][col] at byte offset row * 256 + col
```

Resource arithmetic:

- 256 x 256 x 8 bits = 65536 bytes = 64.0 KB.
- KV260 budget basis: 144 BRAM_36 blocks = 648 KB.
- Estimated update logic: 2000 LUTs.
- KV260 LUT comparison budget: 117000 LUTs.
- Budget fit from arithmetic estimate: True.

Additional small memories are expected for `spin_cur`, `spin_next`, optional
host-provided initial spins, best-state readback, and scalar control registers.
Those buffers are tiny compared with the 64 KB dense J matrix, but final BRAM
count must come from Vivado.

## Random/Noise Source Assumptions

Exp 1399 used small Gaussian CPU noise only to choose the initial signs.  The
RTL plan does not add stochastic noise during the dSB update.  Hardware should
support two initialization modes:

- Host-loaded packed initial spins for deterministic replay.
- A seedable 32-bit LFSR/xorshift initializer that fills `spin_cur` before
  `START`, with the seed exposed through the host register map.

Any future thermal or dither noise source must be benchmarked as a new
algorithmic variant, not silently mixed into this Exp 1399-equivalent RTL.

## Update Schedule

`pressure(t)` is a fixed-point linear ramp from 0 to 1 across `MAX_STEPS`, using
the same schedule family as Exp 1399.  The RTL stores `PRESSURE_START`,
`PRESSURE_DELTA`, and `MAX_STEPS`; each completed sweep increments the pressure
accumulator.  A sweep means all N row updates have written `spin_next` and the
snapshot commit has occurred.

The default synthesis configuration should use:

- `N_VARIABLES = 256`
- `COUPLING_BITS = 8`
- `ETA_Q1_15` matching Exp 1399's eta = 0.09
- `MAX_STEPS = 128`
- one row accumulator / update unit

## Host Interface

Use AXI-Lite control plus a memory-mapped BRAM upload window with 32-bit byte
addresses.  The dense matrix alone occupies 64 KB, so the address map must not
reuse the older 16-bit-only v1 prototype assumption.

| Address | Name | Access | Description |
| --- | --- | --- | --- |
| `0x00000` | `CONTROL` | R/W | bit0 START, bit1 RESET, bit2 LOAD_PRNG_INIT |
| `0x00004` | `STATUS` | R | bit0 READY, bit1 BUSY, bit2 DONE, bit3 ERROR |
| `0x00008` | `N_VARIABLES` | R | default 256 |
| `0x0000C` | `MAX_STEPS` | R/W | default 128 |
| `0x00010` | `ETA_Q1_15` | R/W | fixed-point eta |
| `0x00014` | `PRESSURE_START_Q1_15` | R/W | default 0 |
| `0x00018` | `PRESSURE_DELTA_Q1_15` | R/W | ramp increment |
| `0x0001C` | `RNG_SEED` | R/W | seed for optional initializer |
| `0x01000..0x10FFF` | `J_MATRIX` | W | 65536 byte row-major int8 matrix |
| `0x12000..0x1201F` | `SPIN_INIT` | W | packed initial spins, 1 bit per spin |
| `0x12100..0x1211F` | `SPIN_OUT` | R | packed final spins, 1 bit per spin |
| `0x12200` | `BEST_ENERGY` | R | optional signed fixed-point best energy |

## Synthesis Plan

The intended non-board synthesis command for a Vivado-equipped host is:

```bash
vivado -mode batch -source hardware/kv260/synth_discrete_sb.tcl
```

The planned TCL should target `xck26-sfvc784-2LV-c`, add the future
`hardware/kv260/discrete_sb_256.v` source, set the top module, run synthesis,
write utilization/timing reports, and stop before any board programming step.
Board validation remains a separate action after a bitfile exists and
`CARNOT_KV260_BITFILE` is configured.

## Claim Boundary

No hardware execution was performed for Exp 1422.  No Vivado synthesis report,
bitfile, timing closure, or KV260 board readback exists for this dSB RTL plan.
The only allowed claim is that the RTL specification is complete and the
resource estimate fits the KV260 arithmetic budget inherited from Exp 1399.

Hardware execution and hardware correctness claims are not allowed until a later
run produces synthesis reports and/or KV260 board validation artifacts.
