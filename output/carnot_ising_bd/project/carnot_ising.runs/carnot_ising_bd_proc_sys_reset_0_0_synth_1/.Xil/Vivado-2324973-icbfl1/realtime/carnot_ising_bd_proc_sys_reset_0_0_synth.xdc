set_property SRC_FILE_INFO {cfile:/home/ianblenke/github.com/ianblenke/carnot/output/carnot_ising_bd/project/carnot_ising.gen/sources_1/bd/carnot_ising_bd/ip/carnot_ising_bd_proc_sys_reset_0_0/carnot_ising_bd_proc_sys_reset_0_0_ooc.xdc rfile:../../../../../carnot_ising.gen/sources_1/bd/carnot_ising_bd/ip/carnot_ising_bd_proc_sys_reset_0_0/carnot_ising_bd_proc_sys_reset_0_0_ooc.xdc id:1 order:EARLY scoped_inst:U0} [current_design]
set_property SRC_FILE_INFO {cfile:/home/ianblenke/github.com/ianblenke/carnot/output/carnot_ising_bd/project/carnot_ising.runs/carnot_ising_bd_proc_sys_reset_0_0_synth_1/dont_touch.xdc rfile:../../../dont_touch.xdc id:2} [current_design]
set_property SRC_FILE_INFO {cfile:/tools/Xilinx/2025.2.1/data/ip/xpm/xpm_cdc/tcl/xpm_cdc_single.tcl rfile:../../../../../../../../../../../../../tools/Xilinx/2025.2.1/data/ip/xpm/xpm_cdc/tcl/xpm_cdc_single.tcl id:3 order:LATE scoped_inst:U0/EXT_LPF/ACTIVE_LOW_AUX.ACT_LO_AUX unmanaged:yes} [current_design]
set_property SRC_FILE_INFO {cfile:/tools/Xilinx/2025.2.1/data/ip/xpm/xpm_cdc/tcl/xpm_cdc_single.tcl rfile:../../../../../../../../../../../../../tools/Xilinx/2025.2.1/data/ip/xpm/xpm_cdc/tcl/xpm_cdc_single.tcl id:4 order:LATE scoped_inst:U0/EXT_LPF/ACTIVE_LOW_EXT.ACT_LO_EXT unmanaged:yes} [current_design]
set_property src_info {type:SCOPED_XDC file:1 line:55 export:INPUT save:INPUT read:FILTER_OUT_OF_CONTEXT} [current_design]
create_clock -period 17.000 -name slowest_sync_clk [get_ports slowest_sync_clk]
set_property src_info {type:XDC file:2 line:9 export:INPUT save:INPUT read:READ} [current_design]
set_property KEEP_HIERARCHY SOFT [get_cells U0]
current_instance U0/EXT_LPF/ACTIVE_LOW_AUX.ACT_LO_AUX
set_property src_info {type:SCOPED_XDC file:3 line:5 export:INPUT save:NONE read:READ} [current_design]
set_false_path -to [get_cells {syncstages_ff_reg[0]}]
current_instance
current_instance U0/EXT_LPF/ACTIVE_LOW_EXT.ACT_LO_EXT
set_property src_info {type:SCOPED_XDC file:4 line:5 export:INPUT save:NONE read:READ} [current_design]
set_false_path -to [get_cells {syncstages_ff_reg[0]}]
