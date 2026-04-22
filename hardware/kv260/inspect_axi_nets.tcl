# inspect_axi_nets.tcl — Post-route DCP inspection for RETRO-074 debug.
# Walks S_AXI_ACLK and S_AXI_ARESETN back to their drivers.

set dcp "../../output/carnot_ising_bd/project/carnot_ising.runs/impl_1/carnot_ising_bd_wrapper_routed.dcp"

puts "=== Opening $dcp ==="
open_checkpoint $dcp

# The routed netlist is typically flattened.  Instead of finding a cell by
# REF_NAME, find the sampler's top-level AXI port pins by PORT name.  The
# BD wrapper exposes them via the sampler instance's ports in the hier.
# Try several NAME-match patterns in order, stop at the first non-empty hit.

puts ""
puts "=== searching for sampler instance ==="
set patterns [list \
    {*/ising_sampler_0*inst} \
    {*/ising_sampler_0} \
    {*ising_sampler_0/inst} \
]
set sampler ""
foreach pat $patterns {
    set hits [get_cells -hierarchical -filter "NAME =~ $pat"]
    set n [llength $hits]
    puts "  pattern $pat -> $n hits"
    if {$n >= 1 && $sampler eq ""} {
        set sampler [lindex $hits 0]
    }
}

if {$sampler eq ""} {
    # Fall back: any cell that has an S_AXI_ACLK pin.
    puts ""
    puts "=== fallback: search by S_AXI_ACLK pin ==="
    set pins [get_pins -hierarchical -filter {REF_PIN_NAME == "S_AXI_ACLK"}]
    puts "  [llength $pins] pins named S_AXI_ACLK found in hierarchy:"
    foreach p $pins { puts "    $p" }
    if {[llength $pins] > 0} {
        set sampler [get_cells -of_objects [lindex $pins 0]]
    }
}

if {$sampler eq ""} {
    puts "FATAL: no candidate sampler cell found"
    close_design
    exit 1
}
puts ""
puts "  sampler cell: $sampler"
puts "  REF_NAME:     [get_property REF_NAME [get_cells $sampler]]"

# Pin list inventory (just count + names, no deep traversal)
puts ""
puts "=== S_AXI_* pins on this cell ==="
foreach p [get_pins -of_objects $sampler -filter {REF_PIN_NAME =~ "S_AXI_*"}] {
    set rp [get_property REF_PIN_NAME $p]
    set dir [get_property DIRECTION $p]
    puts "  $rp  ($dir)"
}

# ACLK specifically
puts ""
puts "=== S_AXI_ACLK net ==="
set aclk_pin [lindex [get_pins -of_objects $sampler -filter {REF_PIN_NAME == "S_AXI_ACLK"}] 0]
puts "  pin: $aclk_pin"
if {$aclk_pin ne ""} {
    set aclk_net [get_nets -of_objects $aclk_pin]
    puts "  net: $aclk_net"
    puts "  ROUTE_STATUS: [get_property ROUTE_STATUS $aclk_net]"
    set aclk_drivers [get_pins -of_objects $aclk_net -filter {DIRECTION == "OUT"}]
    puts "  driven by [llength $aclk_drivers] pin(s):"
    foreach d $aclk_drivers {
        set dc [get_cells -of_objects $d]
        set rn [get_property REF_NAME $dc]
        puts "    $d (REF_NAME=$rn)"
    }
}

# ARESETN
puts ""
puts "=== S_AXI_ARESETN net ==="
set arst_pin [lindex [get_pins -of_objects $sampler -filter {REF_PIN_NAME == "S_AXI_ARESETN"}] 0]
puts "  pin: $arst_pin"
if {$arst_pin ne ""} {
    set arst_net [get_nets -of_objects $arst_pin]
    puts "  net: $arst_net"
    puts "  ROUTE_STATUS: [get_property ROUTE_STATUS $arst_net]"
    set arst_drivers [get_pins -of_objects $arst_net -filter {DIRECTION == "OUT"}]
    puts "  driven by [llength $arst_drivers] pin(s):"
    foreach d $arst_drivers {
        set dc [get_cells -of_objects $d]
        set rn [get_property REF_NAME $dc]
        puts "    $d (REF_NAME=$rn)"
    }
}

# Top-level clocks
puts ""
puts "=== top-level clocks ==="
foreach c [get_clocks] {
    set name [get_property NAME $c]
    set period [get_property PERIOD $c]
    puts "  $name  period=$period ns  (~[expr 1000.0/$period] MHz)"
}

# Constant-driver check on critical pins (catches tied-off ARESETN etc.)
puts ""
puts "=== constant-driver check on sampler AXI pins ==="
foreach rp [list S_AXI_ACLK S_AXI_ARESETN S_AXI_AWVALID S_AXI_WVALID S_AXI_ARVALID S_AXI_BREADY S_AXI_RREADY S_AXI_AWREADY S_AXI_WREADY S_AXI_BVALID S_AXI_ARREADY S_AXI_RVALID] {
    set p [lindex [get_pins -of_objects $sampler -filter "REF_PIN_NAME == $rp"] 0]
    if {$p eq ""} { continue }
    set n [get_nets -of_objects $p]
    set is_const 0
    set drv_cells ""
    foreach d [get_pins -of_objects $n -filter {DIRECTION == "OUT"}] {
        set dc [get_cells -of_objects $d]
        set rn [get_property REF_NAME $dc]
        append drv_cells "$rn "
        if {$rn eq "GND" || $rn eq "VCC"} { set is_const 1 }
    }
    set tag ""
    if {$is_const} { set tag "  !! CONSTANT DRIVER" }
    puts "  $rp: drv=[string trim $drv_cells]$tag"
}

puts ""
puts "=== inspection complete ==="
close_design
