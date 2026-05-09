# Hardware Portfolio Rescope (May 2026, Milestone 121)

Following the hardware evaluation report, the Carnot hardware portfolio has been updated:

## AMD Strix Point Unified Memory
- **Rescope:** Strix Point unified memory is unsuitable as the production K=100 sampler path.
- **New Role:** Retained solely as a dev/verifier-edge target.
- **Rationale:** The architecture does not support the required throughput for the primary production sampler, but remains useful for edge verification and development testing.

## Kria KV260 FPGA
- **Retirement:** KV260 board execution and hardware-performance claims that require Vivado or unreachable board access are formally retired.
- **Preservation:** KV260 remains valuable for source-level RTL learning. Local source-level RTL lint and simulation work is preserved as educational and supporting research only.
- **Rationale:** Vivado installation and board bring-up have been consistently blocked, making sovereignty production sampler claims unachievable on this hardware without external proprietary dependencies.
