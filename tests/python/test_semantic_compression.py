"""Tests for Semantic Compression.

Spec: REQ-SCP-001, REQ-SCP-002, SCENARIO-SCP-001
"""
import pytest
from carnot.semantic_compression import SemanticCompressor

def test_semantic_compressor():
    compressor = SemanticCompressor("unsloth/Qwen3.6-35B-A3B-GGUF")
    constraints = ["Constraint 1", "Constraint 2"]
    
    compressed = compressor.compress(constraints)
    assert len(compressed["embeddings"]) == 2
    assert len(compressed["paraphrased"]) == 2
    
    metrics = compressor.evaluate_retrieval(constraints, compressed)
    assert "accuracy" in metrics
    assert "reconstruction_loss" in metrics
