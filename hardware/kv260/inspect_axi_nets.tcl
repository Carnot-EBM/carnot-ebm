# inspect_axi_nets.tcl — Post-route DCP inspection for RETRO-074 debug.
#
# Walks the S_AXI_ACLK and S_AXI_ARESETN signals from the ising_sampler's
# pins back to their drivers, checking for:
#   1. The pin exists and is an input.
#   2. The net connected to the pin is driven by the expected source
#      (pl_clk0 for ACLK, proc_sys_reset/peripheral_aresetn for ARESETN).
#   3. The net is fully routed and has no tie-low / tie-high anomalies.
#
# Run from this directory:
#   source /tools/Xilinx/2025.2.1/Vivado/settings64.sh
#   vivado -mode batch -source inspect_axi_nets.tcl
#
# Output lands in axi_net_inspection.log in this directory and also on
# stdout.

set dcp "../../output/carnot_ising_bd/project/carnot_ising.runs/impl_1/carnot_ising_bd_wrapper_routed.dcp"

puts "=== Opening $dcp ==="
open_checkpoint $dcp

# -----------------------------------------------------------------------------
# Find the sampler cell.  The BD wrapper instantiates it as ising_sampler_0
# inside the BD hierarchy.
# -----------------------------------------------------------------------------
set sampler_cells [get_cells -hierarchical -filter {NAME =~ "*ising_sampler_0*"}]
puts ""
puts "=== candidate sampler cells ==="
foreach c $sampler_cells { puts "  $c" }

# The actual 128_sync module instance.
set sampler [lindex [get_cells -hierarchical -filter {REF_NAME == "ising_sampler_128_sync"}] 0]
puts ""
puts "=== DUT cell ==="
puts "  $sampler"

# -----------------------------------------------------------------------------
# S_AXI_ACLK pin
# -----------------------------------------------------------------------------
puts ""
puts "=== S_AXI_ACLK pin + net ==="
set aclk_pin [get_pins -of_objects $sampler -filter {REF_PIN_NAME == "S_AXI_ACLK"}]
puts "  pin: $aclk_pin"

set aclk_net [get_nets -of_objects $aclk_pin]
puts "  net: $aclk_net"

set aclk_drivers [get_pins -of_objects $aclk_net -filter {DIRECTION == "OUT"}]
puts "  driven by:"
foreach d $aclk_drivers { puts "    $d ([get_property REF_NAME [get_cells -of_objects $d]])" }

set aclk_rs [get_property ROUTE_STATUS $aclk_net]
puts "  ROUTE_STATUS: $aclk_rs"

# -----------------------------------------------------------------------------
# S_AXI_ARESETN pin
# -----------------------------------------------------------------------------
puts ""
puts "=== S_AXI_ARESETN pin + net ==="
set arst_pin [get_pins -of_objects $sampler -filter {REF_PIN_NAME == "S_AXI_ARESETN"}]
puts "  pin: $arst_pin"

set arst_net [get_nets -of_objects $arst_pin]
puts "  net: $arst_net"

set arst_drivers [get_pins -of_objects $arst_net -filter {DIRECTION == "OUT"}]
puts "  driven by:"
foreach d $arst_drivers { puts "    $d ([get_property REF_NAME [get_cells -of_objects $d]])" }

set arst_rs [get_property ROUTE_STATUS $arst_net]
puts "  ROUTE_STATUS: $arst_rs"

# -----------------------------------------------------------------------------
# Source clock: walk ACLK back to pl_clk0
# -----------------------------------------------------------------------------
puts ""
puts "=== clock network summary ==="
report_clocks

# -----------------------------------------------------------------------------
# Reset source: confirm proc_sys_reset's peripheral_aresetn is the driver
# -----------------------------------------------------------------------------
puts ""
puts "=== proc_sys_reset outputs ==="
set psr_cells [get_cells -hierarchical -filter {REF_NAME =~ "proc_sys_reset*"}]
foreach c $psr_cells {
    puts "  cell: $c"
    set out_pins [get_pins -of_objects $c -filter {DIRECTION == "OUT"}]
    foreach p $out_pins {
        set n [get_nets -of_objects $p]
        set rs [get_property ROUTE_STATUS $n]
        puts "    $p  -> net $n (route_status=$rs)"
    }
}

# -----------------------------------------------------------------------------
# Any nets with ROUTE_STATUS indicating trouble
# -----------------------------------------------------------------------------
puts ""
puts "=== nets with non-ROUTED status (first 20) ==="
set troubled [get_nets -filter {ROUTE_STATUS != "ROUTED" && ROUTE_STATUS != "INTRASITE"}]
set i 0
foreach n $troubled {
    if {$i >= 20} break
    puts "  [get_property ROUTE_STATUS $n]  $n"
    incr i
}
puts "  total troubled: [llength $troubled]"

puts ""
puts "=== inspection complete ==="
close_design
