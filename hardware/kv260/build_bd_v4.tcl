# build_bd_v4.tcl — Vivado Block Design for ising_sampler_v4 on KV260
#
# **What this script does:**
#   Builds a bitstream-ready Vivado project wrapping ising_sampler_v4 (the
#   Sparse E-MVL sampler from Exp 958) for the KV260/K26 SOM.
#
# **Why v4 differs from build_bd.tcl (v2):**
#   ising_sampler_v4 has NO AXI-Lite ports — only clk, rst, s_out[127:0], valid.
#   The BD therefore does NOT include an axi_smartconnect slave port connected
#   to the sampler.  Instead we use a Xilinx axi_gpio IP as the AXI slave so the
#   PS can read spin state via memory-mapped I/O (0xA0000000 → s_out[31:0]).
#
# **Block Design topology:**
#
#     ┌──────────────────────────────────┐
#     │  zynq_ultra_ps_e_0 (PS)          │
#     │  FCLK_CLK0 @ 60 MHz ─────────────┼──┐ clk to sampler + gpio + interconnect
#     │  pl_resetn0 ─────────────────────┼──┤ reset chain
#     │  M_AXI_HPM0_FPD ─────────────────┼──┤ AXI master → SmartConnect → axi_gpio
#     └──────────────────────────────────┘  │
#                                           │
#     proc_sys_reset_0 ←─────────────── pl_resetn0
#       peripheral_reset[0] ──────────► ising_sampler_v4.rst  (active-high)
#       interconnect_aresetn ──────────► axi_smartconnect
#       peripheral_aresetn  ──────────► axi_gpio s_axi_aresetn
#
#     axi_smartconnect_0: S00_AXI ← PS HPM0; M00_AXI → axi_gpio
#     axi_gpio_0: gpio_io_i[31:0] ← ising_sampler_v4.s_out[31:0]
#     ising_sampler_v4: free-running (clk from FCLK, rst from proc_sys_reset)
#
# **Why axi_gpio for readback:**
#   v4 RTL has no AXI slave — it runs autonomously and outputs spin state on
#   s_out[127:0].  To let the ARM PS read hardware spin state and measure
#   convergence timing, we feed s_out[31:0] into an axi_gpio GPIO_IO input
#   bank.  The PS reads 0xA0000000 (DATA register) to get the current spin
#   state for 32 spins.
#
# **Usage:**
#     source /tools/Xilinx/2025.2.1/Vivado/settings64.sh
#     cd /home/ianblenke/github.com/ianblenke/carnot
#     vivado -mode batch -source hardware/kv260/build_bd_v4.tcl \
#            -log output/carnot_ising_v4_bd/vivado.log \
#            -journal output/carnot_ising_v4_bd/vivado.jou
#
# **Deliverable:**
#     output/carnot_ising_v4_bd/carnot_ising_v4.bit
#
# **Target part:** xck26-sfvc784-2LV-c  (KV260 K26 SOM)
# **Expected build time:** 25-60 min (the sparse v4 design is much smaller than v2)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

set part          "xck26-sfvc784-2LV-c"
set project_name  "carnot_ising_v4"
set bd_name       "carnot_ising_v4_bd"
set top_rtl_module "ising_sampler_v4"
set rtl_files     [list "hardware/kv260/ising_sampler_v4.v"]
set project_dir   "output/carnot_ising_v4_bd/project"
set bitstream_dst "output/carnot_ising_v4_bd/carnot_ising_v4.bit"

file mkdir [file dirname $project_dir]
file mkdir [file dirname $bitstream_dst]

# ---------------------------------------------------------------------------
# Project + RTL sources
# ---------------------------------------------------------------------------

puts "=== [1/7] Create Vivado project (part=$part) ==="
create_project -force $project_name $project_dir -part $part

# K26 board preset (copies 187 PSU__* properties for correct PS-PL wiring).
# Without this, Vivado uses generic ZUS+ defaults that hang AXI on real K26
# silicon (documented in build_bd.tcl RETRO-074 notes).
set_param board.repoPaths "/tools/Xilinx/2025.2.1/data/xhub/boards/XilinxBoardStore/boards/Xilinx"
set_property board_part xilinx.com:kv260_som:part0:1.4 [current_project]

puts "=== [2/7] Add RTL sources ==="
add_files -norecurse $rtl_files
update_compile_order -fileset sources_1

# ---------------------------------------------------------------------------
# Block Design
# ---------------------------------------------------------------------------

puts "=== [3/7] Create Block Design: $bd_name ==="
create_bd_design $bd_name
current_bd_design $bd_name

# --- Zynq UltraScale+ PS (K26 SOM) ---
create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e zynq_ultra_ps_e_0

# Apply K26 board preset so PS-PL interface widths match the actual silicon.
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
    -config {apply_board_preset "1"} \
    [get_bd_cells zynq_ultra_ps_e_0]

# Enable M_AXI_HPM0_FPD (needed to talk to axi_gpio at 0xA0000000).
# 60 MHz: same as build_bd.tcl RETRO-074 choice (exact integer divisor of 1500 MHz IOPLL).
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0        {1} \
    CONFIG.PSU__USE__M_AXI_GP1        {0} \
    CONFIG.PSU__USE__M_AXI_GP2        {0} \
    CONFIG.PSU__USE__S_AXI_GP0        {0} \
    CONFIG.PSU__USE__S_AXI_GP1        {0} \
    CONFIG.PSU__USE__S_AXI_GP2        {0} \
    CONFIG.PSU__FPGA_PL0_ENABLE       {1} \
    CONFIG.PSU__FPGA_PL1_ENABLE       {0} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {60} \
] [get_bd_cells zynq_ultra_ps_e_0]

# --- AXI SmartConnect: PS HPM0 → axi_gpio ---
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect axi_smartconnect_0
set_property -dict [list \
    CONFIG.NUM_SI {1} \
    CONFIG.NUM_MI {1} \
] [get_bd_cells axi_smartconnect_0]

# --- Processor System Reset ---
# Produces active-high peripheral_reset[0] for v4's rst port and
# active-low interconnect_aresetn / peripheral_aresetn for AXI IPs.
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset proc_sys_reset_0

# --- axi_gpio: single 32-bit input bank fed by s_out[31:0] ---
# The PS reads DATA register at 0xA0000000 to sample current spin state.
# WHY gpio width=32: PS AXI data bus is 32-bit; s_out is 128 bits but
# reading 4 AXI words gives full state.  For convergence timing, 32 bits
# is sufficient — a ferromagnetic ring converges all 128 spins together.
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 axi_gpio_0
set_property -dict [list \
    CONFIG.C_GPIO_WIDTH  {32} \
    CONFIG.C_ALL_OUTPUTS {0} \
    CONFIG.C_ALL_INPUTS  {1} \
    CONFIG.C_IS_DUAL     {0} \
] [get_bd_cells axi_gpio_0]

# --- ising_sampler_v4 (module reference, runs free-running) ---
# v4 parameters: N=128, K=16, FIELD_WIDTH=24, J_WIDTH=16 (all defaults).
# WHY free-running: v4 computes E-MVL autonomously on every clock edge.
# The PS does not control the sampler; it only reads spin state via axi_gpio.
create_bd_cell -type module -reference $top_rtl_module ising_sampler_v4_0

# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

puts "=== [4/7] Wire BD (clock + reset + AXI + s_out) ==="

# PS HPM0 → SmartConnect → axi_gpio
connect_bd_intf_net \
    [get_bd_intf_pins zynq_ultra_ps_e_0/M_AXI_HPM0_FPD] \
    [get_bd_intf_pins axi_smartconnect_0/S00_AXI]
connect_bd_intf_net \
    [get_bd_intf_pins axi_smartconnect_0/M00_AXI] \
    [get_bd_intf_pins axi_gpio_0/S_AXI]

# FCLK_CLK0 (60 MHz) → all clocks
connect_bd_net \
    [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] \
    [get_bd_pins proc_sys_reset_0/slowest_sync_clk] \
    [get_bd_pins axi_smartconnect_0/aclk] \
    [get_bd_pins zynq_ultra_ps_e_0/maxihpm0_fpd_aclk] \
    [get_bd_pins axi_gpio_0/s_axi_aclk] \
    [get_bd_pins ising_sampler_v4_0/clk]

# PL reset (active-low from PS) → proc_sys_reset ext_reset_in
connect_bd_net \
    [get_bd_pins zynq_ultra_ps_e_0/pl_resetn0] \
    [get_bd_pins proc_sys_reset_0/ext_reset_in]

# proc_sys_reset → interconnect (AXI SmartConnect)
connect_bd_net \
    [get_bd_pins proc_sys_reset_0/interconnect_aresetn] \
    [get_bd_pins axi_smartconnect_0/aresetn]

# proc_sys_reset → axi_gpio (active-low)
connect_bd_net \
    [get_bd_pins proc_sys_reset_0/peripheral_aresetn] \
    [get_bd_pins axi_gpio_0/s_axi_aresetn]

# proc_sys_reset peripheral_reset[0] (active-HIGH) → ising_sampler_v4 rst
# WHY peripheral_reset not peripheral_aresetn: v4 uses active-high rst.
# peripheral_reset[0] = ~peripheral_aresetn after synchronization.
connect_bd_net \
    [get_bd_pins proc_sys_reset_0/peripheral_reset] \
    [get_bd_pins ising_sampler_v4_0/rst]

# s_out[31:0] → axi_gpio GPIO_IO inputs
# This allows the PS ARM to read current spin values at 0xA0000000.
connect_bd_net \
    [get_bd_pins ising_sampler_v4_0/s_out] \
    [get_bd_pins axi_gpio_0/gpio_io_i]

# ---------------------------------------------------------------------------
# Address assignment + validation
# ---------------------------------------------------------------------------

puts "=== [5/7] Assign addresses + validate ==="
assign_bd_address
validate_bd_design
save_bd_design

# ---------------------------------------------------------------------------
# HDL wrapper + synthesis + implementation + bitstream
# ---------------------------------------------------------------------------

puts "=== [6/7] Generate HDL wrapper + synth + impl ==="

make_wrapper -files [get_files ${bd_name}.bd] -top
add_files -norecurse "${project_dir}/${project_name}.srcs/sources_1/bd/${bd_name}/hdl/${bd_name}_wrapper.v"
set_property top ${bd_name}_wrapper [current_fileset]
update_compile_order -fileset sources_1

launch_runs synth_1 -jobs 8
wait_on_run synth_1

# Check synthesis succeeded before launching impl
set synth_status [get_property STATUS [get_runs synth_1]]
set synth_progress [get_property PROGRESS [get_runs synth_1]]
puts "=== synth_1 status=$synth_status progress=$synth_progress ==="
if {$synth_progress ne "100%"} {
    error "synth_1 did not complete: status=$synth_status"
}

launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1

set impl_status [get_property STATUS [get_runs impl_1]]
set impl_progress [get_property PROGRESS [get_runs impl_1]]
puts "=== impl_1 status=$impl_status progress=$impl_progress ==="
if {$impl_progress ne "100%"} {
    error "impl_1 did not complete: status=$impl_status"
}

# ---------------------------------------------------------------------------
# Copy bitstream to output
# ---------------------------------------------------------------------------

puts "=== [7/7] Verify + copy bitstream ==="

set bit_src "${project_dir}/${project_name}.runs/impl_1/${bd_name}_wrapper.bit"
if {![file exists $bit_src]} {
    error "Bitstream not found at: $bit_src"
}
file copy -force $bit_src $bitstream_dst

# Log LUT utilization to stdout for parsing by the Python caller.
open_run impl_1
set lut_used [get_property USED [get_cells -hierarchical -filter {PRIMITIVE_TYPE =~ LUT.*}]]
puts "=== CARNOT_LUT_COUNT: [llength $lut_used] ==="

puts ""
puts "=== BUILD COMPLETE ==="
puts "Bitstream: $bitstream_dst"
puts "Board programming:"
puts "  scp $bitstream_dst kria@BOARD_IP:/home/kria/carnot_ising_v4.bit"
puts "  ssh kria 'sudo dfx-mgr-client -load /home/kria/carnot_ising_v4.bit'"
