# Trace ACLK/ARESETN up through the hierarchy to find the real driver.

set dcp "../../output/carnot_ising_bd/project/carnot_ising.runs/impl_1/carnot_ising_bd_wrapper_routed.dcp"
open_checkpoint $dcp

# get_nets with -segments returns the FULL aliased net path across
# hierarchical boundaries.  -top_net_of_hierarchical_group climbs up.
foreach sig {S_AXI_ACLK S_AXI_ARESETN S_AXI_AWVALID} {
    puts ""
    puts "=== tracing $sig ==="
    set pin "carnot_ising_bd_i/ising_sampler_0/inst/$sig"
    set direct_net [get_nets -of_objects $pin]
    puts "  direct net:      $direct_net"

    # Get ALL segments of the net (climbs hierarchy).
    set segments [get_nets -segments $direct_net]
    puts "  total segments:  [llength $segments]"
    foreach s $segments { puts "    seg: $s" }

    # Get the top-level net (the authoritative one with the driver).
    set top_net [get_nets -top_net_of_hierarchical_group $direct_net]
    puts "  top-level net:   $top_net"
    if {$top_net ne ""} {
        set drvs [get_pins -of_objects $top_net -filter {DIRECTION == "OUT"}]
        puts "  top-net drivers ([llength $drvs]):"
        foreach d $drvs {
            set dc [get_cells -of_objects $d]
            set rn [get_property REF_NAME $dc]
            puts "    $d  (REF_NAME=$rn)"
        }
        puts "  top-net ROUTE_STATUS: [get_property ROUTE_STATUS $top_net]"
    }
}

close_design
