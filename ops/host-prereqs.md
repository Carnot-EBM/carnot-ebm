# Host Prerequisites Registry

**Purpose:** Machine-readable record of host-level packages required by Carnot
experiment classes.  Each row documents how to check for a package, how to install
it on the two primary supported platforms, and which experiment classes need it.

**Human action note:** These are host-level prerequisites that must be installed
BEFORE the relevant experiment class will succeed.  When an experiment is blocked
by a missing dependency, check this file first — the missing package and install
command are likely already documented here.

Root cause: RETRO-006 (2026.04.24 retrospective) — AMD XDNA NPU experiments
(Exps 292, 303, 314, 335) each independently discovered `ninja` and `openblas`
were missing, wasting ~4 experiment slots.  This registry short-circuits retries.

---

## Prerequisite Table

| Package | Check Command | Install (Arch) | Install (Debian) | Required For |
|---------|--------------|----------------|-----------------|--------------|
| ninja | ninja --version | pacman -S ninja | apt install ninja-build | npu, fpga |
| openblas | pkg-config --libs openblas | pacman -S openblas | apt install libopenblas-dev | npu |
| CARNOT_FORCE_LIVE | env:CARNOT_FORCE_LIVE | export CARNOT_FORCE_LIVE=1 | export CARNOT_FORCE_LIVE=1 | live_gpu, all |
| nvidia-smi | nvidia-smi --version | pacman -S nvidia-utils | apt install nvidia-utils-XXX | live_gpu |
| yosys | yosys --version | pacman -S yosys | apt install yosys | fpga |
| nextpnr-xilinx | nextpnr-xilinx --version | yay -S nextpnr-xilinx | apt install nextpnr-xilinx | fpga |

---

## Column Definitions

- **Package**: Name of the required system package or environment variable.
- **Check Command**: Shell command that exits 0 when the package is present.
  Use `env:VAR_NAME` for environment variable checks.
- **Install (Arch)**: How to install on Arch Linux (pacman, yay, or manual).
- **Install (Debian)**: How to install on Debian/Ubuntu (apt).
- **Required For**: Comma-separated experiment class tags.
  Use `all` to mark a universal requirement.

## Known Blocks

| Experiment | Blocked By | Status |
|-----------|-----------|--------|
| Exp 292 | ninja, openblas | Discovered 2026-04-14; fixed in Exp 303 |
| Exp 303 | ninja, openblas | Fixed 2026-04-14 |
| Exp 314 | ninja, openblas | Fixed 2026-04-14 |
| Exp 335 | ninja, openblas | Fixed 2026-04-15 |
