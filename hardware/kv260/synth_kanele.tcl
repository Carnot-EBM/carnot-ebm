# synth_kanele.tcl — Vivado batch-mode synthesis script for kanele_top.v
#
# Target: xck26-sfvc784-2LV-c (KV260 production SOM)

set part "xck26-sfvc784-2LV-c"
set top_module "kanele_top"
set rtl_files [list \
    "hardware/kv260/kanele_lut.v" \
    "hardware/kv260/kanele_top.v" \
]
set output_dir "output/kanele_synth"

# Create output directory
file mkdir $output_dir

# Create in-memory project
create_project -in_memory -part $part

# Add RTL sources
add_files $rtl_files
set_property top $top_module [current_fileset]

# Synthesis step
synth_design -top $top_module -part $part -flatten_hierarchy rebuilt

# Write post-synthesis checkpoint
write_checkpoint -force ${output_dir}/post_synth.dcp

# Implementation (place and route)
opt_design
place_design
route_design

# Write bitfile
write_bitstream -force ${output_dir}/kanele_top.bit

# Report resource utilization
report_utilization -file ${output_dir}/utilization.rpt
report_timing_summary -file ${output_dir}/timing.rpt

puts "Synthesis complete: ${output_dir}/kanele_top.bit"
