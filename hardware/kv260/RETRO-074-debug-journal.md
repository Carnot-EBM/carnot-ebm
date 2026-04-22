# RETRO-074: KV260 Ising Sampler On-Hardware Hang — Debug Journal

**Status as of 2026-04-22:** Still open after four on-hardware attempts.
Five hypotheses eliminated with trace evidence; two remain; RTL fix verified
correct in simulation but hardware still wedged on first AXI access.

This document consolidates the scattered `results/kv260_*.json` artifacts
into a single readable record so future sessions can pick up the thread
without re-reading nine separate files.

## TL;DR

- **What works:** toolchain (Vivado 2025.2.1), block-design wrapper,
  app packaging (bit.bin + dtbo + shell.json), `xmutil loadapp`, /dev/uio
  node creation, clock control via devmem.
- **What doesn't:** first AXI register access from the PS wedges the
  whole PS AXI interconnect, requiring a power-cycle.  Reproduced four
  times this session with increasingly aggressive fixes.
- **Current top suspect:** cold-start power-rail droop from 128 LFSRs
  unconditionally advancing on every clock cycle (2048 flops switching
  continuously).  Fix in flight: gate LFSR advances on `reg_control[0]`
  so they idle until the CPU writes the "start sampler" bit.

## Timeline

| Event | Commit | Result JSON |
|---|---|---|
| First build (N=128) — too large for XCK26 | f9bedd8b | `kv260_bd_build.json` |
| N=64, 40 MHz — first timing-clean bitstream | f9bedd8b | `kv260_bd_build_N64.json` |
| Exp 661 first deploy — **PS hang #1** | 733642a2 | `experiment_661_kv260_n64_benchmark.json` |
| App packaging (.bit.bin + dtbo + shell.json) | cee1367d | — |
| Exp 661 redeploy — **PS hang #2** | — | — |
| Vivado xsim bare-DUT — RTL AXI passes | 21f733d7 | `kv260_axi_simulation_v2.json` |
| 50 MHz rebuild — marginal timing | aac5ef79 | `kv260_bd_build_N64.json` (updated) |
| PL clock probe — fclk ignores overlay, 60 MHz actual | 3fbb794a | `kv260_pl_clock_probe.json` |
| Force 30 MHz via devmem — **PS hang #3** | 77e3d5de | `kv260_axi_30mhz_forced.json` |
| DCP netlist inspection — wiring clean | e11a73a0 | `kv260_dcp_netlist_inspection.json` |
| SmartConnect co-sim — HANG REPRODUCED | fa0209d2 | `kv260_smartconnect_cosim.json` |
| RTL AXI-Lite fix (aw_done/w_done) | (conductor auto-commit) | — |
| Split-channel TB verifies fix | (same) | — |
| 30 MHz + fixed RTL — **PS hang #4** | d184074a | `kv260_fourth_hang.json` |
| LFSR clock-gate RTL — rebuild in flight | (current) | pending |

## Hypotheses — What the investigation ruled in and out

### Eliminated with trace evidence

1. **RTL AXI-Lite protocol deadlock (split AW/W channels).**  The prior
   RTL required AWVALID+WVALID to coincide before either AWREADY/WREADY
   asserted.  AXI-Lite §A3.3.1 explicitly permits independent AW/W.
   *Bare-DUT xsim passed* because the BFM held both VALIDs continuously
   (accidentally avoiding the bug).  *SmartConnect co-sim reproduced the
   hang deterministically* because SmartConnect drives AW/W with
   independent pipelines.  **Fixed** with `aw_done`/`w_done` completion
   flags for independent per-channel ACK.  Split-channel testbench
   verifies the fix.

2. **pl_clk0 not running (CLKACT=0).**  CRL_APB.PL0_REF_CTRL register
   probe shows CLKACT=1 in every state tested.

3. **Clock-frequency mismatch causing setup timing violations.**  Kria's
   fclk framework ignores overlay `assigned-clock-rates` hints —
   register reads 0x01011900 (DIVISOR0=25, 60 MHz) regardless of whether
   we request 40 or 50 MHz.  Forced DIVISOR0=50 via devmem for exact
   30 MHz actual with the 30-MHz-synth bitstream (+9.4 ns slack).
   Hardware still hung.  Timing is not the root cause.

4. **proc_sys_reset wiring bug — ARESETN not reaching sampler.**  DCP
   netlist inspection: `S_AXI_ARESETN` trace has 75 segments from
   `proc_sys_reset_0/U0/peripheral_aresetn[0]` through the SmartConnect
   down to the sampler's reset pin.  Chain is complete and routed.

5. **AXI ACLK not reaching the sampler.**  DCP inspection: `S_AXI_ACLK`
   trace has 1169 segments through the clock tree to every clocked
   primitive in the sampler (including all 1024+64 DSP48E2 CLK pins).

6. **Address decoding — PS M_AXI_HPM0_FPD routing to wrong address.**
   `report_bd_addr_segs` in the BD shows PS Data master → sampler reg0
   at offset 0xA0000000, range 128 KB.  Matches the address our test
   harness uses.

### Surviving hypotheses

7. **Cold-start power-rail droop from unconditional LFSR advances
   (TOP SUSPECT, fix in flight).**  Pre-fix RTL unconditionally advanced
   128 LFSRs × 16 flops = 2048 flops every clock cycle, even in
   FSM_IDLE.  Switching current from these is our largest suspect for
   VCCINT droop at startup, which could corrupt PS AXI handshake timing
   or reset synchronizers.  Fix: gate LFSR advance on `reg_control[0]`
   so they idle until CPU writes the start bit.  Bitstream rebuild
   currently running (bg `bglfptzu6`).

8. **Clock-switch glitch from devmem DIVISOR write mid-run.**  Possible
   explanation for post-devmem hangs specifically (hangs #3 and #4).
   Does NOT explain hangs #1 and #2 which happened before any devmem
   write.  Lower-ranked but not eliminated.

### Moot or subsumed

9. **SmartConnect protocol checker panic.**  Replaced by a concrete
   understanding: SmartConnect is doing the right thing; our RTL was
   non-AXI-Lite-compliant.  Fixed in hypothesis #1.

10. **Dynamic power integrity from 1088 DSPs switching on first cycle.**
    Re-examined during hypothesis #7 work — DSPs aren't switching at
    cold-start because their inputs are static (bias/coupling RAM
    defaults don't change).  LFSRs ARE switching every cycle though.
    Subsumed by #7.

## Key infrastructure decisions

- **N_SPINS=64, MAX_DEGREE=16** — the original N=128 design overflowed
  XCK26 LUT budget by 2.5×.  Parametric override in
  `hardware/kv260/build_bd.tcl` via `set_property CONFIG.N_SPINS`.
- **30 MHz target** — 1500 MHz IOPLL / 50 / 1 = 30.00 MHz exact.  One
  of the divisor values fclk will actually land on.  Previous attempts
  at 40 MHz → 60 MHz actual, 50 MHz → 48.485 MHz actual.
- **Kria app bundle format** — `/lib/firmware/xilinx/<app>/<app>.bit.bin
  + <app>.dtbo + shell.json`.  `.bit.bin` is bootgen-wrapped, not the
  raw Vivado `.bit`.  `dfx-mgr-client -load <raw.bit>` fails with
  `Load Error: -1`; `xmutil loadapp <app>` is the canonical loader.

## Tooling built along the way

- `hardware/kv260/build_bd.tcl` — BD + synth + impl pipeline with
  env-driven N_SPINS/MAX_DEGREE/FREQMHZ overrides.
- `hardware/kv260/app/package_app.sh` — bootgen + dtc packaging.
- `hardware/kv260/ising_sampler_v2_axi_tb.v` — bare-DUT AXI-Lite TB.
- `hardware/kv260/ising_sampler_v2_axi_tb_split.v` — split-channel TB
  that specifically exercises the pattern SmartConnect generates.
- `hardware/kv260/smartconnect_sampler_tb.sv` — full AXI-full master
  BFM + SmartConnect gate-level sim + sampler co-simulation.
- `hardware/kv260/inspect_axi_nets.tcl` — DCP pin inventory + constant-
  driver check.
- `hardware/kv260/inspect_axi_nets_v2.tcl` — hierarchical net tracing
  via `-top_net_of_hierarchical_group` and `-segments`.
- `hardware/kv260/inspect_bd_address.tcl` — BD address-map report.
- `scripts/kv260_probe_pl_clk.sh` — read-only probe of pl_clk0 state
  via `/dev/mem` mmap of CRL_APB.PL0_REF_CTRL.  Optional `--load-app`
  flag loads the app before probing.
- `scripts/experiment_661_kv260_n64_benchmark.py` — end-to-end
  deployment experiment with honest-verdict enum and bounded SIGALRM
  timeouts.

## Current state of the bitstream

`hardware/kv260/ising_sampler_v2.v` contains:
- Fixed AXI-Lite state machine (aw_done/w_done independent ACK, commit
  `fa0209d2` context and later conductor auto-commit).
- **Pending in current build:** LFSR advances gated on `reg_control[0]`
  (this session's edit, building in bg `bglfptzu6`).

`hardware/kv260/build_bd.tcl` contains:
- N_SPINS=64, MAX_DEGREE=16, FREQMHZ=30 (line 161), RETRO-073 comment
  block documenting the clock-drop rationale.

## Operator action log

The operator has performed **four hard power-cycles** on the KV260 this
session, each after a wedged PS hang.  Each power-cycle also caused
filesystem corruption — `/lib/firmware/xilinx/carnot_ising_v2_n64/*`
files came back with correct metadata but zeroed content at least
twice.  Redeploying a fresh bundle from the host before each attempt.

## What should happen next

1. **LFSR-gated bitstream completes** (bg `bglfptzu6`, in flight).
2. **Verify timing** — should still be comfortable ~9 ns WNS at 30 MHz
   because the LFSR gate is a single LUT level added only to the mux
   select for the LFSR register input, not in the critical path.
3. **Repackage + deploy** once the operator authorises a fifth attempt.
4. **Do NOT force clock via devmem** this time — try letting fclk set
   it to whatever it chooses.  The 30 MHz target is generous enough
   that even if fclk snaps to 60 MHz actual, the bitstream has margin
   (rebuilt with tight timing) — UNLESS the prior crashes were all
   clock-glitch-induced, in which case NOT doing devmem is the fix.
5. **If that hangs:** open a JTAG ILA probe on the PL AXI signals to
   capture the exact moment of failure.  Requires a USB cable to the
   KV260's JTAG port.  Delivers ground-truth AXI trace; expensive but
   definitive.

## Lessons for similar hardware work

- **Bare-DUT simulation is insufficient** for AXI-Lite slaves talking
  to real interconnects.  The master BFM must reproduce the actual
  interconnect's pipelining behavior, or the RTL's subtle non-
  compliance goes undetected.  SmartConnect-in-the-loop was the
  deciding evidence.
- **Five-hypothesis elimination with trace evidence beats one-
  hypothesis debugging.**  Each ruled-out hypothesis narrows the search
  space.  Writing an artifact for each attempt, even failed ones,
  makes the progression visible.
- **Power integrity matters at startup.**  Idle-but-clocked flops draw
  non-trivial switching current.  Any design with thousands of them
  should gate their clocks or enables on a "system active" signal.
- **fclk framework does not honor overlay clock-rate hints on Kria.**
  Test it — don't assume.  Our bitstream has to close timing at
  whatever fclk decides.
