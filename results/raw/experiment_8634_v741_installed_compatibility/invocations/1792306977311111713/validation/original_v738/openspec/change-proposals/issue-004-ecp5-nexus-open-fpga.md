# Ising sampler bitstream variants for Lattice ECP5 / Nexus (fully-open toolchain)

**Status:** Draft change proposal.
**Origin:** [GitHub issue #4](https://github.com/Carnot-EBM/carnot-ebm/issues/4) (2026-04-24).
**Target milestone:** 2026.04.63 — pairs well with milestone .62's Exp 816
  (KV260 open-source synthesis v2) as its non-Xilinx sibling.
**Priority:** Medium. Widens reproducibility and contributor base; not on the
  critical path.
**Depends on:** OSS-CAD-Suite installed 2026-04-24 at `/opt/oss-cad-suite/`
  (yosys 0.64+149, nextpnr-ecp5, nextpnr-nexus, all ice* tools).
**Related:** `openspec/change-proposals/research-roadmap-v62.md` Exp 816
  (KV260 via OSS-CAD-Suite) — same toolchain, different target silicon.

## Summary

Milestone .62 Exp 816 plans the KV260 path via the OSS-CAD-Suite yosys flow.
KV260 is Zynq UltraScale+ (ZU5EV) — ~$300 per board, and UltraScale+ is not
covered by `nextpnr`/`prjxray` end-to-end yet, so Exp 816 still ends up
with partial open-flow (synthesis open, PnR proprietary via Vivado).

Two device families are **fully** covered by the open-source toolchain we
already installed:

| Target | Toolchain | Boards | Cost |
|---|---|---|---|
| **Lattice ECP5** | `yosys` + `nextpnr-ecp5` + `prjtrellis` + `ecppack` | ULX3S, OrangeCrab, ColorLight i5/i9, ECPIX-5 | $30-$160 |
| **Lattice Nexus (LFD2NX)** | `yosys` + `nextpnr-nexus` + `prjoxide` | Radiant OEM, CertusPro-NX | $60-$200 |

Porting the Ising sampler RTL to one or both targets produces reproducible
bitstreams entirely from source with nothing vendor-proprietary, lowers the
hardware barrier ~10×, and broadens the contributor base to the hobbyist
FPGA community, courses, and independent researchers.

See issue #4 for the full comparison.

## Proposed experiments

### Exp A — ECP5 port of Ising sampler v2 (ULX3S target)

**Deliverable:** `hardware/ecp5/ising_sampler_v2_ecp5.v` (thin wrapper over
existing `hardware/kv260/ising_sampler_v2.v`) + `hardware/ecp5/ulx3s.lpf`
constraints + `results/experiment_<N>_ecp5_ising_synthesis.json`.

**What it does:**

1. Reuse the portable RTL (AXI-Lite slave, Ising update, LFSR RNGs,
   sigmoid LUT) unchanged.
2. Swap memory primitives: Xilinx `BRAM36E2` → Lattice `EBR_DP16K`.
   (yosys handles most of this automatically via the `-family ecp5`
   target; explicit inference hints in RTL where it doesn't.)
3. Replace AXI-Lite with a simpler USB-UART register bridge (the board
   has USB-JTAG for `openFPGALoader`; a small TX/RX shim gives us a
   host-readable register window).
4. Target `ECP5-LFE5U-85F`. Run `yosys` → `nextpnr-ecp5` → `ecppack`.

**Acceptance gates:**

1. Synthesis produces a bitstream with `LUT4 usage < 70%`, `EBR usage <
   70%` — must leave headroom for follow-on work.
2. Place-and-route meets 50 MHz target (generous; ECP5 realistic ceiling
   is 150 MHz but first port aims for margin).
3. `openFPGALoader` flashes the bitstream onto a ULX3S board (if operator
   has one) or verifies the bitstream structure via `ecppack --debug`
   (if not). Honest-verdict enum: `ecp5_synth_clean_hw_pending`,
   `ecp5_synth_clean_hw_verified`, `ecp5_synth_failed`,
   `ecp5_pnr_timing_fail`.

### Exp B — Nexus port of Ising sampler v2

Mirror of Exp A targeting CertusPro-NX / LFD2NX-40. Same RTL, different
place-and-route backend (`nextpnr-nexus`) and memory primitive
(`OXIDE_LARGE_DP16K`).

**Acceptance gates:** same shape, scaled for smaller fabric (`LUT usage <
80%` acceptable because the device is smaller).

### Exp C — Host-side Python driver for the USB-UART bridge

Implements the same `SamplerBackend` protocol as `FpgaBackend` (which talks
to KV260 via `/dev/uio4`) — so swapping backends is an env-var change, not
a code change. This is REQ-KONA-006's hardware portability being exercised
on a new target.

### Exp D — CI workflow

Run yosys + nextpnr + ecppack on GitHub Actions with the OSS-CAD-Suite
tarball pre-cached. Publishes the ECP5 bitstream as a CI artefact on every
main-branch commit that touches `hardware/ecp5/**`. Fits a typical free-tier
runner (< 2 GB memory, < 15 min).

## Risks

- **RTL portability assumed, not proven.** Exp 758 showed yosys synthesizes
  the KV260 RTL cleanly (2821 LUTs / 2237 DFFs) — but that was for a
  hypothetical target. ECP5 specifics (EBR, MULT18X18) may need explicit
  inference hints. Budget one day for portability patches.
- **Smaller fabric.** At N=32 / MAX_DEGREE=8 (current KV260 config) the
  ECP5-85F has plenty of room. Scaling toward N=4K on ECP5 is probably not
  feasible without multi-FPGA work. Acknowledge ECP5 as the
  iteration-velocity target, KV260 as the scale-up target.
- **Board availability.** Neither ULX3S nor CertusPro-NX is on-site. Exp A/B
  can complete synthesis-and-bitstream without hardware; actual
  hardware-verified honest-verdict requires a board.
