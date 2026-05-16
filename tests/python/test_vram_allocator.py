"""
Tests for multi-model VRAM allocator.
"""

from carnot.hardware.vram_allocator import mock_vram_allocation, calculate_theoretical_max_sequence_length

def test_mock_vram_allocation():
    # Covers REQ-INFER-SOTA-2009
    alloc = mock_vram_allocation(192.0)
    assert alloc["unsloth/Qwen3.6-35B-A3B-GGUF"] == 15.0
    assert alloc["unsloth/gemma-4-31B-it-GGUF"] == 14.0
    assert alloc["total_vram_gb"] == 192.0

def test_calculate_max_sequence_length():
    # Covers SCENARIO-INFER-SOTA-2009-001
    seq_len = calculate_theoretical_max_sequence_length(192.0, 15.0, 14.0, 0.5)
    assert seq_len == int(((192.0 - 29.0) / 0.5) * 1000)
    
def test_calculate_max_sequence_length_zero():
    # Covers boundary
    seq_len = calculate_theoretical_max_sequence_length(20.0, 15.0, 14.0, 0.5)
    assert seq_len == 0
