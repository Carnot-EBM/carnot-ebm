# KV260 Discrete SB RTL Regression Manifest

Spec traces: REQ-ISING-027, SCENARIO-ISING-037

Run date: 20260507
Result artifact: `results/experiment_1476_kv260_discrete_sb_rtl_regression_pack.json`

## Source Files

- RTL files: `hardware/kv260/discrete_sb_256.v`
- Testbench files: `hardware/kv260/discrete_sb_256_tb.v`

## Tool Availability

- Verilator available: `true`
- Icarus Verilog available: `true`
- Yosys available: `true`

## Commands

1. Verilator lint
   - Command: `verilator --lint-only --timing -Wall -Wno-DECLFILENAME -Wno-BLKSEQ -Wno-WIDTH hardware/kv260/discrete_sb_256.v hardware/kv260/discrete_sb_256_tb.v`
   - Expected output: return code 0 and no lint errors.
   - Observed pass: `true`
2. Icarus simulation
   - Command: `iverilog -g2012 -o /tmp/discrete_sb_256_tb_1476.vvp hardware/kv260/discrete_sb_256.v hardware/kv260/discrete_sb_256_tb.v && vvp /tmp/discrete_sb_256_tb_1476.vvp`
   - Expected output: return code 0 and `SIMULATION RESULT: PASS`.
   - Observed pass: `true`
3. Yosys availability probe
   - Command: `yosys -q -p help read_verilog`
   - Expected output: return code 0 when Yosys is installed; unavailable tools are recorded.
   - Observed available: `true`
   - Observed probe pass: `true`

## Claim Boundary

Board execution performed: `false`
Bitfile produced: `false`
Latency claimed: `false`

This manifest is source-level RTL regression evidence only. It does not claim
Vivado synthesis, a KV260 bitfile, KV260 board execution, or measured latency.

Honest verdict: `rtl_regression_manifest_complete_source_level_only_no_board_bitfile_or_latency_claim`
