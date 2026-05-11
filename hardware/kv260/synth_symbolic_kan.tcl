# synth_symbolic_kan.tcl
# TCL script for out-of-context synthesis of Symbolic-KAN module

create_project -in_memory -part xck26-sfvc784-2LV-c
read_verilog hardware/kv260/symbolic_kan.v
synth_design -top symbolic_kan -mode out_of_context
report_utilization -file results/symbolic_kan_utilization.txt