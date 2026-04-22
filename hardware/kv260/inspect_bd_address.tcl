# inspect_bd_address.tcl — verify the PS AXI address map.

set proj "../../output/carnot_ising_bd/project/carnot_ising.xpr"

puts "=== Opening project $proj ==="
open_project $proj

set bd_files [get_files -quiet -filter {FILE_TYPE == "Block Designs"}]
puts ""
puts "=== BD files ==="
foreach b $bd_files { puts "  $b" }

if {[llength $bd_files] == 0} {
    puts "ERROR: no BD files found"
    close_project
    exit 1
}

open_bd_design [lindex $bd_files 0]

puts ""
puts "=== BD address segments detail ==="
foreach s [get_bd_addr_segs] {
    set range   [get_property range $s]
    set offset  [get_property offset $s]
    set access  [get_property access_type $s]
    puts "  offset=$offset  range=$range  access=$access"
    puts "    seg=$s"
}

puts ""
puts "=== addr space of PS master ==="
foreach aspace [get_bd_addr_spaces] {
    puts "  $aspace  range=[get_property range $aspace]"
}

close_bd_design [lindex $bd_files 0]
close_project
