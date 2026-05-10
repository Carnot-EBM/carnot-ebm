"""
Tests for KANELÉ LUT Mapper.

References: REQ-KAN-1729, SCENARIO-KAN-1729
"""
import os
import sys
import tempfile

# Add hardware/kv260 to path to import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../hardware/kv260')))
import kanele_lut_mapper

def test_generate_lut_verilog():
    """Test generating verilog from weights. REQ-KAN-1729."""
    weights = [1] * 64
    verilog = kanele_lut_mapper.generate_lut_verilog(weights, "test_mod")
    assert "module test_mod" in verilog
    assert "LUT6" in verilog
    assert "64'h" in verilog
    assert "endmodule" in verilog

def test_map_cikan_to_fpga():
    """Test writing verilog to file. REQ-KAN-1729."""
    weights = [0, 1] * 32
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "out.v")
        kanele_lut_mapper.map_cikan_to_fpga(weights, out_path)
        assert os.path.exists(out_path)
        with open(out_path, "r") as f:
            content = f.read()
        assert "module kanele_lut" in content
