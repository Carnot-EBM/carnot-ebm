# PolarFire SoC Adaptive K-PCD Prototype Preflight

## Overview
This document outlines the preflight plan for prototyping an adaptive K-PCD algorithm on a PolarFire SoC (RISC-V cores + FPGA fabric).

## Board Acquisition Path
Currently, no PolarFire SoC board is locally available. A development board such as the Microchip PolarFire SoC Icicle Kit or a similar board needs to be acquired to execute hardware tests.

## Open-Toolchain Feasibility
An open-source toolchain path using Yosys exists via OSS-CAD-Suite (yosys is available locally at `/opt/oss-cad-suite/bin/yosys`). However, full synthesis, place-and-route, and bitstream generation for PolarFire typically require vendor tools or specialized plugins.

## Libero Fallback Caveat
Microchip's Libero SoC toolchain is the standard vendor fallback for full bitstream compilation. It is currently not installed on this machine. Installation of privileged software/vendor tools is required if open-toolchain efforts fall short of supporting the specific PolarFire device primitives.

## Simulator-First Plan
Given the lack of physical hardware and Libero tools, the immediate plan is simulator-first:
1. Re-use existing Carnot Verilog components (24 files found, e.g. `ising_sampler_v*`).
2. Compose the n=128 block-Gibbs adaptive K-PCD logic.
3. Validate RTL via Python-driven testbenches and open-source simulators (e.g., Icarus Verilog or Verilator).
4. Run open-source linting and synthesis checks using Yosys to ensure the design is sound and synthesizable before moving to physical hardware.

## Acceptance Gates
1. **RTL Simulation**: Passes all adaptive K-PCD behavior tests in software simulation.
2. **Toolchain readiness**: Either open-toolchain supports bitstream generation, or Libero SoC is installed and functional.
3. **Hardware execution**: PolarFire SoC board is connected, bitstream is programmed, and RISC-V host can successfully dispatch and retrieve K-PCD tasks from the FPGA fabric.
