# PolarFire SoC Adaptive K-PCD Prototype Preflight

## Overview
This document outlines the preflight plan for prototyping the adaptive K-PCD sampler on a PolarFire SoC. The PolarFire SoC is a strong contrarian candidate due to its colocated RISC-V cores and FPGA fabric, enabling tight integration between the sampler control loop and hardware acceleration.

## Board Acquisition Path
No PolarFire SoC board is currently available in the lab. A board such as the PolarFire SoC Icicle Kit or similar development board must be acquired to perform on-hardware testing. Until a board is acquired, all execution must be simulator-based.

## Toolchain Feasibility

### Open-Toolchain (Yosys)
Yosys is available on the system (`/opt/oss-cad-suite/bin/yosys`). While Yosys has some support for FPGAs, full bitstream generation for PolarFire often requires vendor tools. Yosys can be used for synthesis and formal verification of the adaptive K-PCD RTL modules.

### Libero Fallback Caveat
The official Microchip Libero SoC toolchain is **not** currently installed on the system. If the open-source toolchain (Yosys/NextPNR) is insufficient for generating a working bitstream for the PolarFire SoC fabric, Libero will be strictly required as a fallback. Obtaining and licensing Libero requires vendor registration and privileged software installation, which is a current toolchain blocker for full hardware implementation.

## Simulator-First Plan
Given the lack of physical hardware and the proprietary vendor toolchain blocker, the immediate plan is strictly **simulator-first**:
1. Port existing reusable RTL components (e.g., `ising_sampler_v3`) to a generic n=128 block-Gibbs format.
2. Develop Verilator testbenches to simulate the colocated RISC-V (using a soft-core or transaction-level model) interacting with the K-PCD sampler fabric.
3. Validate the adaptive K-PCD convergence properties purely in simulation using Yosys for formal synthesis.

## Acceptance Gates
To progress beyond the preflight phase, the following gates must be met:
1. **Simulation Gate**: Verilator simulation of n=128 block-Gibbs achieves functional parity with the software model.
2. **Toolchain Gate**: A clear path to bitstream generation is established (either via open-source tools or a managed Libero installation).
3. **Hardware Gate**: A PolarFire SoC development board is acquired and accessible in the test environment.
