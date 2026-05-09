"""
Tests for KAN LUT Compiler

Traces to: REQ-KAN-1621, SCENARIO-KAN-1621
"""
import pytest
from carnot.hardware.kan_lut_compiler import generate_lut6_init, synthesize_1d_function, generate_verilog_module

def test_generate_lut6_init():
    # Identity function (just returns the LSB)
    def identity(x: int) -> int:
        return x & 1
    # Bits will be 0, 1, 0, 1... so INIT is AAAAAAAA...
    init_hex = generate_lut6_init(identity)
    assert init_hex == "AAAAAAAAAAAAAAAA"

def test_synthesize_1d_function():
    # Double the input: y = 2*x (shift left by 1)
    def double_x(x: int) -> int:
        return (x * 2) & 0xFF
    
    inits = synthesize_1d_function(double_x, input_bits=6, output_bits=8)
    assert len(inits) == 8
    # y[0] is always 0
    assert inits[0] == "0000000000000000"
    # y[1] is x[0], so it should be AAAAAAAAAAAAAAAA
    assert inits[1] == "AAAAAAAAAAAAAAAA"

def test_synthesize_1d_function_too_large():
    def double_x(x: int) -> int:
        return (x * 2) & 0xFF
    with pytest.raises(ValueError):
        synthesize_1d_function(double_x, input_bits=7, output_bits=8)

def test_generate_verilog_module():
    inits = ["0000000000000000", "AAAAAAAAAAAAAAAA"]
    verilog = generate_verilog_module("test_lut", inits)
    assert "module test_lut" in verilog
    assert "LUT6 #(" in verilog
    assert "64'hAAAAAAAAAAAAAAAA" in verilog
    assert "y[1]" in verilog
