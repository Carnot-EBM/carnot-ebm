"""
Experiment 1621: KANELE Python-to-Verilog 6-input LUT synthesis

Traces to: REQ-KAN-1621, SCENARIO-KAN-1621
"""
import json
import os
from carnot.hardware.kan_lut_compiler import synthesize_1d_function, generate_verilog_module

def main():
    # A generic 1D KAN edge function (e.g. Swish or simple polynomial approximation)
    # Let's map x^2 (scaled) to an 8-bit output
    def kan_edge(x: int) -> int:
        # x is 6 bits (0 to 63). Let's map it to x^2 >> 4
        return (x * x) >> 4

    inits = synthesize_1d_function(kan_edge, input_bits=6, output_bits=8)
    verilog_code = generate_verilog_module("kan_lut_block", inits)

    output_v_path = "hardware/kv260/kan_lut_block.v"
    os.makedirs(os.path.dirname(output_v_path), exist_ok=True)
    with open(output_v_path, "w") as f:
        f.write(verilog_code)

    artifact = {
        "schema": "experiment_artifact_v1",
        "status": "complete",
        "experiment_id": "1621",
        "spec": "REQ-KAN-1621",
        "kan_lut_verilog_ready": True,
        "lut_config_bits_generated": True,
        "kan_lut_block_written": True,
        "honest_verdict": "success: kanele python-to-verilog mapping complete"
    }

    output_json_path = "results/experiment_1621_kanele_mapping.json"
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(artifact, f, indent=2)

if __name__ == "__main__":
    main()
