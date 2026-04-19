# KV260 FPGA Synthesis Guide: 128-Spin Sparsified Ising Sampler

This document contains step-by-step Vivado commands to synthesize
`ising_sampler_128_sparse.v` and flash the resulting bitfile to the KV260.
These commands are for human execution — automated synthesis was blocked in Exp 471
because Vivado is not installed on the research server.

## Prerequisites

- Vivado 2023.x installed (free Webpack edition covers KV260)
- KV260 connected via USB-JTAG or Ethernet (PYNQ)
- Linux host with `~/.Xilinx/Vivado` licence for the xck26 device

## Step 1: Generate the exp_lut .mif file

The Metropolis acceptance LUT must be pre-computed before synthesis so Vivado
can initialise the BRAM.  Run this on the host before opening Vivado:

```bash
cd /path/to/carnot
python3 - <<'EOF'
import math
beta = 2.0
vals = [round(2**32 * math.exp(-2 * beta * k / 256)) for k in range(256)]
clamped = [min(v, 2**32 - 1) for v in vals]
with open("hardware/kv260/ising_exp_lut.mif", "w") as f:
    for v in clamped:
        f.write(f"{v:08X}\n")
print("Wrote hardware/kv260/ising_exp_lut.mif")
EOF
```

## Step 2: Open Vivado and create a project

```tcl
# Launch Vivado in batch mode (no GUI needed):
vivado -mode batch -source hardware/kv260/synth_ising_sparse.tcl
```

Or interactively in the Vivado Tcl console:

```tcl
# Create project targeting KV260 (xck26-sfvc784-2LV-c)
create_project ising_sparse ./vivado_ising_sparse -part xck26sfvc784-2LV-c

# Add source files
add_files hardware/kv260/ising_sampler_128_sparse.v
add_files hardware/kv260/ising_exp_lut.mif

# Set top module
set_property top ising_sampler_128_sparse [current_fileset]
```

## Step 3: Add constraints (XDC)

Create `hardware/kv260/ising_sparse.xdc` with clock constraint:

```tcl
# 100 MHz clock on KV260 PL fabric (adjust pin if using custom carrier)
create_clock -period 10.000 -name sys_clk [get_ports S_AXI_ACLK]
```

Then add to Vivado:

```tcl
add_files -fileset constrs_1 hardware/kv260/ising_sparse.xdc
```

## Step 4: Run synthesis

```tcl
# Synthesize (this takes 5-15 minutes on the KV260 target)
synth_design -top ising_sampler_128_sparse -part xck26sfvc784-2LV-c

# Check utilisation — target: <50% LUTs, <50% BRAM for comfortable timing closure
report_utilization

# Check timing
report_timing_summary
```

Expected resource usage at 90% sparsity:
- LUTs: ~4,000-8,000 (2-3% of KV260's 256K)
- BRAM18: 3-5 slices (for coupl_ram at 128x128x10b = 163,840 bits)
- FF: ~500

If timing fails, add a pipeline register in the local-field accumulation loop
or reduce N_SPINS in the parameter.

## Step 5: Implement (place and route)

```tcl
opt_design
place_design
route_design
report_timing_summary -warn_on_violation
```

## Step 6: Generate bitfile

```tcl
write_bitstream -force output/ising_sampler_128_sparse.bit
```

The bitfile lands at `output/ising_sampler_128_sparse.bit`.

## Step 7: Flash to KV260 via PYNQ

Copy the bitfile to the KV260 and load via PYNQ:

```bash
scp output/ising_sampler_128_sparse.bit ubuntu@kv260.local:/home/ubuntu/
```

On the KV260:

```python
from pynq import Overlay
ol = Overlay("/home/ubuntu/ising_sampler_128_sparse.bit")
mmio = ol.ip_dict["ising_sampler_0"]["driver"]

# Verify the device is ready
status = mmio.read(0x0004)
print(f"STATUS = 0x{status:08X}")  # should show READY (0x1)
```

## Step 8: Verify with Carnot FpgaBackend

Set the env var and run the experiment:

```bash
export CARNOT_KV260_BITFILE=/home/ubuntu/ising_sampler_128_sparse.bit
JAX_PLATFORMS=cpu python3 scripts/experiment_471_kv260_fpga_v2.py
```

If `honest_verdict = "fpga_executing"` appears in the result JSON, the hardware
path is confirmed live.

## Troubleshooting

- **Timing failure at 100 MHz**: Add `-directive PerformanceExplore` to `place_design`.
- **BRAM inference failure**: Ensure `coupl_ram` is declared as `reg [31:0]` array;
  Vivado infers BRAM automatically for arrays > 1024 words.
- **PYNQ import error**: Install with `pip install pynq` on PetaLinux; standard
  Ubuntu may need the Xilinx PYNQ wheel from the KV260 starter kit.
- **LFSR stuck at 0**: Verify the reset seed (`32'hACE1_2345`) — an all-zero LFSR
  is a fixed point and will never generate random numbers.
