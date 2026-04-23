# synth_ising_hls.tcl — Vitis HLS synthesis script for Ising Sampler v4
#
# WHY THIS FILE EXISTS:
#   Vivado is not installed locally, so full Verilog-based synthesis is blocked.
#   Vitis HLS (distributed separately in AMD Vitis 2024.2) can synthesise HLS C++
#   directly to RTL and export as an IP catalog that Vivado can then implement.
#   This TCL script automates the Vitis HLS flow for ising_sampler_hls.cpp.
#
# USAGE (on a machine with Vitis HLS installed):
#   vitis_hls -f hardware/kv260/synth_ising_hls.tcl
#
# OUTPUTS:
#   output/ising_hls_v4/solution1/syn/verilog/ — synthesised RTL
#   output/ising_hls_v4/solution1/impl/ip/      — IP catalog export for Vivado
#
# TARGET PART: xck26-sfvc784-2LV-c (Xilinx KV260 SOM)
# Spec: REQ-HW-010

# Create a new HLS project targeting the KV260 part
create_project ising_hls_v4 ./output/ising_hls_v4 -part xck26-sfvc784-2LV-c

# Add the HLS C++ source file
add_files hardware/kv260/ising_sampler_hls.cpp

# Set the top-level function to synthesise.
# update_spin_kernel is the entry point — it drives the AXI interfaces.
set_top update_spin_kernel

# Open a solution (a named configuration + target device snapshot)
open_solution "solution1" -flow_target vivado

# Re-specify the target part (required inside solution context)
set_part xck26-sfvc784-2LV-c

# Clock constraint: 10 ns period = 100 MHz.
# KV260 PL fabric comfortably runs at 100-200 MHz; 100 MHz is conservative
# and gives HLS maximum scheduling freedom for the inner loop pipeline.
create_clock -period 10 -name default

# Run C synthesis (generates RTL, timing/area estimates)
csynth_design

# Export as IP catalog so Vivado can drag it into a block design
# This produces a .zip that Vivado imports as an AXI4 IP
export_design -format ip_catalog
