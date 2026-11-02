"""
Multi-model VRAM allocator test for ROCm.
"""

def mock_vram_allocation(total_vram_gb: float) -> dict:
    """Mock VRAM allocation for SOTA models if hardware absent."""
    return {
        "unsloth/Qwen3.6-35B-A3B-GGUF": 15.0,
        "unsloth/gemma-4-31B-it-GGUF": 14.0,
        "total_vram_gb": total_vram_gb,
        "kv_cache_gb_per_1k_tokens": 0.5
    }

def calculate_theoretical_max_sequence_length(total_vram_gb: float, model1_weights_gb: float, model2_weights_gb: float, kv_cache_gb_per_1k_tokens: float) -> int:
    """Determine theoretical max sequence length."""
    available_for_kv = total_vram_gb - (model1_weights_gb + model2_weights_gb)
    if available_for_kv <= 0:
        return 0
    return int((available_for_kv / kv_cache_gb_per_1k_tokens) * 1000)
