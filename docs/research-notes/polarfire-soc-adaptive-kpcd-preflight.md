# PolarFire SoC Adaptive K-PCD Prototype Preflight

## 1. Board Acquisition Path
Currently, no PolarFire SoC physical board is available in the lab. The recommended acquisition path is to procure the PolarFire SoC Icicle Kit or a similar evaluation board from authorized Microchip distributors.

## 2. Open-Toolchain Feasibility
While `yosys` is present on the system (via oss-cad-suite), fully open-source toolchain support for the Microchip PolarFire fabric is immature or non-existent. A purely open-source bitstream generation flow is currently not feasible for this architecture.

## 3. Libero Fallback Caveat
Because an open-source flow is unavailable, compiling for the physical PolarFire FPGA requires Microchip's proprietary Libero SoC Design Suite. A local check confirms Libero is not currently installed (`which libero` returned no result). Installing Libero requires managing proprietary licenses and large software installations, which presents a significant environmental blocker.

## 4. Simulator-First Plan
Given the lack of a physical board and the missing Libero toolchain, the prototype development must proceed with a **Simulator-First Plan**.
- We will reuse existing RTL components (e.g., `ising_sampler_v3.v`, `ising_energy_n8_comb.v`).
- Development will focus on building testbenches and verifying the n=128 block-Gibbs logic using open-source simulators (e.g., Verilator or Icarus Verilog) which are available and compatible.

## 5. Acceptance Gates
To progress beyond simulation to hardware, the following gates must be cleared:
- **Gate 1:** Successful RTL simulation of the adaptive K-PCD sampler and integration tests.
- **Gate 2:** Procurement and connection of a supported PolarFire SoC hardware board.
- **Gate 3:** Installation and licensing of the Libero SoC Design Suite (or maturation of a FOSS alternative).
- **Gate 4:** Successful bitstream generation and "Hello World" deployment to the target board.
