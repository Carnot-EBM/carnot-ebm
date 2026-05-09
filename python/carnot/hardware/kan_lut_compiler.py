"""
KAN LUT Compiler

Translates generic 1D functions into 6-input LUT configuration bits.
"""
from typing import Callable, List

def generate_lut6_init(func: Callable[[int], int]) -> str:
    """
    Generate a 64-bit INIT string (as a 16-character hex string) for a 6-input LUT
    that implements the 1-bit boolean function `func`.
    `func` takes an integer from 0 to 63 and returns 0 or 1.
    """
    init_val = 0
    for i in range(64):
        bit = func(i) & 1
        init_val |= (bit << i)
    return f"{init_val:016X}"

def synthesize_1d_function(func: Callable[[int], int], input_bits: int = 6, output_bits: int = 8) -> List[str]:
    """
    Synthesize an M-bit output 1D function of an N-bit input into M LUT initialization strings.
    `func` takes an integer and returns an integer.
    Requires input_bits <= 6 to fit in a single layer of LUT6.
    Returns a list of M INIT strings (hex), from LSB (index 0) to MSB.
    """
    if input_bits > 6:
        raise ValueError("Only input_bits <= 6 supported for direct LUT6 synthesis without routing.")
    
    inits = []
    for out_bit in range(output_bits):
        # Create a boolean function for this specific output bit
        def bit_func(x: int, b=out_bit) -> int:
            return (func(x) >> b) & 1
        inits.append(generate_lut6_init(bit_func))
    return inits

def generate_verilog_module(module_name: str, inits: List[str]) -> str:
    """
    Generate a Verilog module that instantiates LUT6 primitives with the given INIT strings.
    """
    output_bits = len(inits)
    lines = [
        f"module {module_name} (",
        "    input wire [5:0] x,",
        f"    output wire [{output_bits-1}:0] y",
        ");",
        ""
    ]
    
    for i, init in enumerate(inits):
        lines.append(f"    LUT6 #(")
        lines.append(f"        .INIT(64'h{init})")
        lines.append(f"    ) lut_{i} (")
        lines.append(f"        .O(y[{i}]),")
        lines.append(f"        .I0(x[0]),")
        lines.append(f"        .I1(x[1]),")
        lines.append(f"        .I2(x[2]),")
        lines.append(f"        .I3(x[3]),")
        lines.append(f"        .I4(x[4]),")
        lines.append(f"        .I5(x[5])")
        lines.append(f"    );")
        lines.append("")
    
    lines.append("endmodule")
    return "\n".join(lines)
