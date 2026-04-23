# build_bd.tcl — Vivado Block Design wrapper for the Carnot Ising sampler on KV260
#
# **What this script does:**
#   Creates a complete, bitstream-loadable Vivado project that wraps the
#   ising_sampler_128_sync RTL (from hardware/kv260/ising_sampler_v2.v) as a
#   PL-side AXI-Lite slave, connected to the Zynq UltraScale+ PS (on the K26
#   SOM) via a SmartConnect.  All S_AXI_* ports are routed INTERNALLY to the
#   PS-PL interconnect — none map to board pins.  This resolves the 3 DRC
#   errors (KLOC-1 / NSTD-1 / UCIO-1) that blocked write_bitstream in the
#   standalone synth_ising.tcl flow (commit e594bd17 / results/kv260_first_synthesis.json).
#
# **Why a BD wrapper is required:**
#   ising_sampler_128_sync exposes 114 S_AXI_* top-level ports that must plug
#   into the Zynq PS via the PL-PS AXI interconnect, NOT as physical KV260
#   pins.  Standalone synthesis auto-places those ports as board I/O and
#   Kria DRC rules correctly block that (two ports landed on K26 VRP
#   calibration pins G4/W9, which are grounded 240-Ω reference resistors).
#   A Block Design wraps the sampler in a context where AXI gets routed
#   through PS internals, not pins.
#
# **Block Design topology:**
#
#     ┌──────────────────────────┐
#     │  zynq_ultra_ps_e_0 (PS)  │
#     │  ─ FCLK_CLK0 @ 50 MHz    │────┐  (RETRO-074: 50 MHz lands cleanly on
#     │  ─ pl_resetn0            │    │
#     │  ─ M_AXI_HPM0_FPD ──────────► │ axi_smartconnect_0 ─── S_AXI on
#     └──────────────────────────┘    │                        ising_sampler_128_sync
#                  ▲                  │
#                  └─── proc_sys_reset_0 (synchronises reset to PL clock)
#
# **Usage:**
#     source /tools/Xilinx/2025.2.1/Vivado/settings64.sh   # if not already
#     cd /home/ianblenke/github.com/ianblenke/carnot
#     vivado -mode batch -source hardware/kv260/build_bd.tcl \
#            -log output/carnot_ising_bd/vivado.log \
#            -journal output/carnot_ising_bd/vivado.jou
#
# **Deliverable:**
#     output/carnot_ising_bd/project/carnot_ising.runs/impl_1/carnot_ising_bd_wrapper.bit
#
# **Target part:**  xck26-sfvc784-2LV-c  (KV260 production K26 SOM)
# **Expected build time:**  25-60 min (synth + place + route + bitstream;
#   the 128-spin sampler is compact but the PS wrapper adds ~30 min of P&R).
#
# **Post-build deployment to KV260:**
#     scp output/carnot_ising_bd/project/carnot_ising.runs/impl_1/carnot_ising_bd_wrapper.bit \
#         kria:/opt/carnot/bitstreams/carnot_ising_v2.bit
#     ssh kria 'sudo dfx-mgr-client -load /opt/carnot/bitstreams/carnot_ising_v2.bit'
#     ssh kria 'xrt-smi examine'   # should now list the accelerator
#
# Spec: closes RETRO-070 (bitstream blocked on I/O wrapping).  Intended
#       companion to Exp 584 / 599 / 636 hardware-track follow-ups.

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

set part          "xck26-sfvc784-2LV-c"
set project_name  "carnot_ising"
set bd_name       "carnot_ising_bd"
set top_rtl_module "ising_sampler_128_sync"
# CARNOT_RTL_FILES can override the default source list for isolation builds
# (e.g. the RETRO-074 minimal AXI responder).  Colon-separated paths.
if {[info exists env(CARNOT_RTL_FILES)]} {
    set rtl_files [split $env(CARNOT_RTL_FILES) ":"]
    puts "=== RTL source override: $rtl_files ==="
} else {
    set rtl_files [list \
        "hardware/kv260/ising_sampler_v2.v" \
    ]
}
# Output goes to an UNCOMMITTED dir — bitstream is ~5 MB, intermediate files
# are ~1 GB.  .gitignore filters output/** so none of this ends up in git.
set project_dir   "output/carnot_ising_bd/project"
set bitstream_dst "output/carnot_ising_bd/carnot_ising_bd_wrapper.bit"

# ---------------------------------------------------------------------------
# RTL parameter overrides  (RETRO-072 sizing fix)
# ---------------------------------------------------------------------------
# The RTL module is parameterised at N_SPINS=128, MAX_DEGREE=32 by default.
# Place_design on XCK26 fails with DRC UTLZ-1 at those defaults: LUT6 reports
# 290k used vs 117k available (2.48x over), LUT-as-logic 3.67x over, CARRY8
# 1.29x over.  Resource usage scales roughly as N_SPINS * MAX_DEGREE; to fit
# the KV260 silicon we override the parameters at BD-instantiation time
# rather than editing the RTL.
#
# Env vars (read if set, otherwise defaults below):
#   CARNOT_N_SPINS     — target spin count (default 64 — fits with headroom)
#   CARNOT_MAX_DEGREE  — max neighbours per spin (default 16)
#
# Sizing sweep expectation (rough, pre-opt):
#   N=128, MAX_DEGREE=32: 290k LUT6  — OVERFLOWS xck26
#   N= 64, MAX_DEGREE=32: 145k LUT6  — still 1.24x over
#   N= 64, MAX_DEGREE=16:  73k LUT6  — fits with headroom
#   N= 32, MAX_DEGREE=32:  73k LUT6  — fits (fewer spins, same fan-in)
#   N= 32, MAX_DEGREE=16:  36k LUT6  — comfortable; leaves room for future
#                                      JEPA-adjacent PL extensions
#
# Empirical numbers: first post-opt place_design on (64,16) will be logged to
# results/kv260_bd_build_N64.json so the expected-vs-actual table stays honest.
set n_spins    [expr {[info exists env(CARNOT_N_SPINS)]    ? $env(CARNOT_N_SPINS)    : 32}]
set max_degree [expr {[info exists env(CARNOT_MAX_DEGREE)] ? $env(CARNOT_MAX_DEGREE) : 8}]
puts "=== RTL parameters: N_SPINS=$n_spins, MAX_DEGREE=$max_degree ==="

file mkdir [file dirname $project_dir]
file mkdir [file dirname $bitstream_dst]

# ---------------------------------------------------------------------------
# Project + source
# ---------------------------------------------------------------------------

puts "=== [1/7] Create Vivado project (part=$part) ==="
create_project -force $project_name $project_dir -part $part

# RETRO-074 hang #9 root cause: without the K26 board files, Vivado generates
# a Zynq UltraScale+ PS wrapper that assumes the generic PSS_REF_CLK=50 MHz
# default.  The K26 SOM silicon provides PSS_REF_CLK=33.333 MHz.  The PS's
# own PLL configuration is set at boot by PMUFW (independent of our
# bitstream) so the clock frequencies themselves turn out OK — but the
# PS→PL interface wrapper (M_AXI_HPM0_FPD data/ID widths, HSCLK/PSCLK
# handoffs, and 180+ other subtle properties) is generated assuming the
# wrong silicon family config, and on real K26 silicon the first AXI
# transaction wedges the interconnect.  The K26 preset in
# xhub/XilinxBoardStore/boards/Xilinx/k26c encodes the 187 PSU__* properties
# that describe the actual K26 silicon; apply_board_preset imports them all
# in one call.
set_param board.repoPaths "/tools/Xilinx/2025.2.1/data/xhub/boards/XilinxBoardStore/boards/Xilinx"
set_property board_part xilinx.com:kv260_som:part0:1.4 [current_project]

puts "=== [2/7] Add RTL sources ==="
# Skip RTL sourcing when CARNOT_USE_AXI_GPIO is set — the BD will use
# Xilinx's axi_gpio IP directly, no custom RTL required.
if {![info exists env(CARNOT_USE_AXI_GPIO)] || $env(CARNOT_USE_AXI_GPIO) != "1"} {
    add_files -norecurse $rtl_files
    # Don't set the RTL module as top — the BD wrapper will become top.
    update_compile_order -fileset sources_1
} else {
    puts "=== axi_gpio isolation build: skipping RTL sources ==="
}

# ---------------------------------------------------------------------------
# Block Design
# ---------------------------------------------------------------------------

puts "=== [3/7] Create Block Design: $bd_name ==="
create_bd_design $bd_name
current_bd_design $bd_name

# --- Zynq UltraScale+ PS (K26 SOM) ---
# K26 board files aren't installed, so we use the default preset + enable
# only the PL-facing signals we need.  DDR/Ethernet/MIO are left at defaults
# — they won't be correctly mapped for the K26's physical pins, but we
# aren't using them (our accelerator is PL-only, communicates via AXI from
# the PS user-space runtime dfx-mgrd).
create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e zynq_ultra_ps_e_0

# Apply the K26 board preset — this imports all 187 PSU__* properties that
# describe the K26 silicon (PSS_REF_CLK=33.333 MHz, correct IOPLL/RPLL
# configuration, MIO mapping, DDR timing, PS-PL interface widths, etc).
# Without this, Vivado uses generic Zynq UltraScale+ defaults which produce
# a bitstream whose PS-PL interconnect does not function on K26 silicon
# (RETRO-074 hangs #1-#9).
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
    -config {apply_board_preset "1"} \
    [get_bd_cells zynq_ultra_ps_e_0]

# Then apply our own overrides on top:
#   PSU__USE__M_AXI_GP0   ← M_AXI_HPM0_FPD (low-power domain, 32-bit AXI slave)
#   PSU__FPGA_PL0_ENABLE  ← PL clock 0 (FCLK_CLK0) @ 50 MHz (RETRO-074:
#                            Kria's fclk framework rounded our prior 40 MHz
#                            request to DIVISOR0=25 giving 60 MHz — a clock
#                            mismatch the 40 MHz-synth bitstream could not
#                            survive (setup violated by ~5.5 ns, metastable
#                            AXI flops, PS hang observed twice on-hardware
#                            per Exp 661).  50 MHz = 1500 MHz IOPLL / 30 / 1
#                            is an exact integer divisor the fclk framework
#                            lands on cleanly, and the design has ~+3.174 ns
#                            slack at 40 MHz so 50 MHz (20 ns period) should
#                            close with ~1 ns margin.  Earlier RETRO-073:
#                            dropped from the 100 MHz default because the N=64
#                            build closed WNS=-11.916 ns at 100 MHz — worst
#                            critical path is ~22 ns. 40 MHz (25 ns period)
#                            gives ~3 ns positive slack with zero RTL/tool
#                            risk. Throughput hit of 2.5x is acceptable for
#                            pipeline validation; recover later via pipeline-
#                            register insertion + Performance_Retiming impl
#                            strategy (RETRO-073 options D + B).
# Disable the other HPM/ACP ports to keep the design minimal.
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0         {1} \
    CONFIG.PSU__USE__M_AXI_GP1         {0} \
    CONFIG.PSU__USE__M_AXI_GP2         {0} \
    CONFIG.PSU__USE__S_AXI_GP0         {0} \
    CONFIG.PSU__USE__S_AXI_GP1         {0} \
    CONFIG.PSU__USE__S_AXI_GP2         {0} \
    CONFIG.PSU__FPGA_PL0_ENABLE        {1} \
    CONFIG.PSU__FPGA_PL1_ENABLE        {0} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {60} \
] [get_bd_cells zynq_ultra_ps_e_0]

# --- AXI SmartConnect ---
# One master (from PS), one slave (to our ising_sampler).  SmartConnect
# handles clock-domain-crossing automatically, so if we ever add a higher-
# frequency clock to the sampler we don't need to insert an axi_clock_converter.
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect axi_smartconnect_0
set_property -dict [list \
    CONFIG.NUM_SI {1} \
    CONFIG.NUM_MI {1} \
] [get_bd_cells axi_smartconnect_0]

# --- Processor System Reset ---
# Synchronises the PS-generated pl_resetn0 to FCLK_CLK0 domain and produces
# an active-low reset for the SmartConnect + ising_sampler.
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset proc_sys_reset_0

# --- Ising sampler (user RTL, added as a module reference) ---
# RETRO-074 axi_gpio isolation: when CARNOT_USE_AXI_GPIO=1 is set, replace
# our custom RTL slave with a Xilinx axi_gpio IP block.  axi_gpio is a
# battle-tested Xilinx-provided AXI-Lite slave; if it responds on hardware
# at 0xA0000000 through our SmartConnect path, then every infrastructure
# piece (PS, SmartConnect, resets, clocks, address decoding) is sound and
# our custom module instantiation pattern has a bug.  If axi_gpio ALSO
# hangs, the issue is above our RTL entirely — a BD/PS config problem the
# board preset didn't address.
set use_axi_gpio [expr {[info exists env(CARNOT_USE_AXI_GPIO)] \
                        && $env(CARNOT_USE_AXI_GPIO) == "1"}]

if {$use_axi_gpio} {
    puts "=== Using Xilinx axi_gpio IP as slave (RETRO-074 isolation) ==="
    create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 ising_sampler_0
    set_property -dict [list \
        CONFIG.C_GPIO_WIDTH       {32} \
        CONFIG.C_ALL_OUTPUTS      {0} \
        CONFIG.C_ALL_INPUTS       {1} \
        CONFIG.C_IS_DUAL          {0} \
    ] [get_bd_cells ising_sampler_0]
    # Tie GPIO inputs to a fixed pattern so we can see whether the read
    # path works by comparing observed RDATA to the constant.
    create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 gpio_const_0
    set_property -dict [list \
        CONFIG.CONST_WIDTH {32} \
        CONFIG.CONST_VAL   {2779096666} \
    ] [get_bd_cells gpio_const_0]
    # 2779096666 = 0xA5A55A5A — xlconstant CONST_VAL takes decimal only
    connect_bd_net \
        [get_bd_pins gpio_const_0/dout] \
        [get_bd_pins ising_sampler_0/gpio_io_i]
} else {
    create_bd_cell -type module -reference $top_rtl_module ising_sampler_0
    # RETRO-072: override module parameters at BD instantiation time
    set_property -dict [list \
        CONFIG.N_SPINS    $n_spins \
        CONFIG.MAX_DEGREE $max_degree \
    ] [get_bd_cells ising_sampler_0]
}

# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

puts "=== [4/7] Wire BD (AXI + clocks + resets) ==="

# PS M_AXI_HPM0_FPD (master) → SmartConnect S00_AXI (slave)
connect_bd_intf_net \
    [get_bd_intf_pins zynq_ultra_ps_e_0/M_AXI_HPM0_FPD] \
    [get_bd_intf_pins axi_smartconnect_0/S00_AXI]

# SmartConnect M00_AXI → ising_sampler S_AXI
connect_bd_intf_net \
    [get_bd_intf_pins axi_smartconnect_0/M00_AXI] \
    [get_bd_intf_pins ising_sampler_0/S_AXI]

# PL clock 0 (FCLK_CLK0) drives all AXI clocks + reset generator.
# axi_gpio uses lowercase s_axi_aclk; our custom RTL uses uppercase S_AXI_ACLK.
set sampler_aclk    [expr {$use_axi_gpio ? "ising_sampler_0/s_axi_aclk"    : "ising_sampler_0/S_AXI_ACLK"}]
set sampler_aresetn [expr {$use_axi_gpio ? "ising_sampler_0/s_axi_aresetn" : "ising_sampler_0/S_AXI_ARESETN"}]

connect_bd_net \
    [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] \
    [get_bd_pins zynq_ultra_ps_e_0/maxihpm0_fpd_aclk] \
    [get_bd_pins axi_smartconnect_0/aclk] \
    [get_bd_pins proc_sys_reset_0/slowest_sync_clk] \
    [get_bd_pins $sampler_aclk]

# RETRO-074 hang #10 candidate: associate maxihpm0_fpd_aclk with
# proc_sys_reset_0/interconnect_aresetn explicitly.  Without this, Vivado's
# IP_Flow 19-5611 infers the aclk has no associated reset and creates a
# "default connection" (tie-off) for M_AXI_HPM0_FPD's reset — which may
# hold the PS's M_AXI master interface in reset or leave its CDC logic
# uninitialised.  This warning has been present in every build since hang
# #1 and was never acted on.
set_property CONFIG.ASSOCIATED_RESET {proc_sys_reset_0/interconnect_aresetn} \
    [get_bd_pins zynq_ultra_ps_e_0/maxihpm0_fpd_aclk]

# PS reset (active-low, from PL0) → proc_sys_reset ext_reset_in
connect_bd_net \
    [get_bd_pins zynq_ultra_ps_e_0/pl_resetn0] \
    [get_bd_pins proc_sys_reset_0/ext_reset_in]

# Synchronised resets from proc_sys_reset:
#   interconnect_aresetn → SmartConnect (Xilinx PG164 best practice: the
#     interconnect needs the EARLIER reset output so its pipelines/FIFOs
#     have settled before any peripheral tries to use it).
#   peripheral_aresetn   → ising_sampler AXI-Lite slave
# RETRO-074 hang #8 root cause: both pins were previously tied to
# peripheral_aresetn.  SmartConnect came out of reset simultaneously with
# the sampler slave — its internal state wasn't fully initialised when the
# first AXI read arrived, and the transaction silently wedged at the
# interconnect level (no ARREADY ever propagated to the slave, no RVALID
# ever propagated back to the master, CPU blocked on the load instruction,
# RCU stalled on CPU3 after 60s).  Vivado flagged this as a Synth 8-7071
# warning from hang #1 onward; it was dismissed as harmless because the
# reset tree still existed — but "connected" and "correct" aren't the same
# thing for this IP.
connect_bd_net \
    [get_bd_pins proc_sys_reset_0/interconnect_aresetn] \
    [get_bd_pins axi_smartconnect_0/aresetn]

connect_bd_net \
    [get_bd_pins proc_sys_reset_0/peripheral_aresetn] \
    [get_bd_pins $sampler_aresetn]

# ---------------------------------------------------------------------------
# Address assignment + validation
# ---------------------------------------------------------------------------

puts "=== [5/7] Assign addresses + validate ==="

# Auto-assign address for the S_AXI slave.  For LPD 32-bit AXI, the legal
# range starts at 0xA000_0000.  Vivado picks 0xA0000000 by default with a
# 64 KB window — plenty for our ~128 spin registers + status.
assign_bd_address

# Validate the BD before generating HDL — fail loud if anything is unwired.
validate_bd_design

# Save the BD so Vivado can regenerate from it later (useful for debugging).
save_bd_design

# ---------------------------------------------------------------------------
# HDL wrapper + synthesis + implementation + bitstream
# ---------------------------------------------------------------------------

puts "=== [6/7] Generate HDL wrapper + synth + impl ==="

# make_wrapper + add_files registers a top-level Verilog wrapper around the BD.
make_wrapper -files [get_files ${bd_name}.bd] -top
add_files -norecurse "${project_dir}/${project_name}.srcs/sources_1/bd/${bd_name}/hdl/${bd_name}_wrapper.v"
set_property top ${bd_name}_wrapper [current_fileset]
update_compile_order -fileset sources_1

# Launch synthesis + implementation runs.  wait_on_run blocks until complete.
# Runtime: ~20-40 min on a 24-core host.
launch_runs synth_1 -jobs 8
wait_on_run synth_1
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1

# ---------------------------------------------------------------------------
# Verify + copy bitstream
# ---------------------------------------------------------------------------

puts "=== [7/7] Verify + copy bitstream ==="

set bit_src "${project_dir}/${project_name}.runs/impl_1/${bd_name}_wrapper.bit"
if {![file exists $bit_src]} {
    error "Bitstream not produced at expected path: $bit_src — check impl_1 logs"
}
file copy -force $bit_src $bitstream_dst

puts ""
puts "=== BUILD COMPLETE ==="
puts "Bitstream: $bitstream_dst"
puts "Next: scp $bitstream_dst kria:/opt/carnot/bitstreams/carnot_ising_v2.bit"
puts "      ssh kria 'sudo dfx-mgr-client -load /opt/carnot/bitstreams/carnot_ising_v2.bit'"
puts "      ssh kria 'xrt-smi examine'"
