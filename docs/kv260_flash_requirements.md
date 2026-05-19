# KV260 Flash Requirements and Alternative Paths

This document outlines the requirements and options for flashing a bitstream to the KV260 K26 SOM.

## Bitstream Location
The generated bitstream from experiment 2477 is located at:
`output/exp2477_kv260_bitstream/carnot_kv260.bit`
(or `results/kv260_ising_top.bit` if copied).

## Programmer Options
The Vivado Hardware Manager requires a compatible Xilinx JTAG programmer to flash the board natively:
1. **Digilent JTAG HS2 / HS3:** (~$50-$60) - A widely supported, reliable 3.3V/1.8V compatible cable that works natively with Vivado and openocd.
2. **Xilinx Platform Cable USB II:** (~$250) - The official Xilinx programmer, supported out-of-the-box by Vivado.

## Alternative: DirtyJTAG Compatibility
**Is DirtyJTAG (1209:c0ca) compatible with KV260?** `False`

*Evidence / Rationale:* 
The KV260 K26 SOM JTAG header (J4) exposes 1.8V levels for the Zynq UltraScale+ PS/PL JTAG chain. DirtyJTAG firmware (often running on a Raspberry Pi Pico or STM32 Blue Pill) typically drives JTAG signals at 3.3V without built-in logic-level translation. Connecting a 3.3V DirtyJTAG directly to the 1.8V KV260 JTAG header risks over-volting and damaging the ZynqMP device. Furthermore, Vivado `hw_server` does not natively support DirtyJTAG; an intermediate XVC (Xilinx Virtual Cable) bridge would be required. Thus, DirtyJTAG is not a feasible drop-in replacement.

## Alternative: OpenOCD Flash Feasibility
**Is OpenOCD flash feasible without Vivado?** `False`

*Evidence / Rationale:*
OpenOCD supports the Zynq UltraScale+ TAP topology (via `xilinx_zynqmp.cfg`), allowing access to the PS (ARM A53/R5 cores) and the PL TAP. However, unlike simpler FPGAs (e.g., Kintex/Virtex/Zynq-7000 where `pld load` works natively), directly programming the PL of a ZynqMP over JTAG typically requires initialization of the PS (via PMU/FSBL) to power on the PL domain, or specific intricate TAP sequences that are not handled by a simple `pld load` command in openocd's default configs. OpenFPGALoader also lacks a dedicated `kv260` board profile. Consequently, flashing the bitstream purely via OpenOCD is not a low-friction alternative.

## Vivado Hardware Manager (Tcl Path)
When a supported programmer (e.g., Digilent JTAG HS2) is connected, Vivado can flash the bitstream using the following exact Tcl commands:

```tcl
open_hw_manager
connect_hw_server
open_hw_target
set_property PROGRAM.FILE {carnot_kv260.bit} [get_hw_devices xck26*]
program_hw_devices
```
