import pytest
from carnot.pipeline.verify_repair import VerifyRepairPipeline

def test_otv_probe_fallback():
    # Test fallback behavior when probe is missing or fails
    pipeline = VerifyRepairPipeline(
        model=None, domains=[], max_repairs=1, extractor=None,
        semantic_grounding_verifier=None, semantic_verifier_v2=None, timeout_seconds=10
    )
    # The default return is 0.5 (or whatever prediction it makes if trained)
    # Just ensure it doesn't crash
    prob = pipeline.otv_probe("prompt", "response")
    assert isinstance(prob, float)

def test_route_fast_path_otv():
    pipeline = VerifyRepairPipeline(
        model=None, domains=[], max_repairs=1, extractor=None,
        semantic_grounding_verifier=None, semantic_verifier_v2=None, timeout_seconds=10
    )
    # Mock otv_probe to return high confidence
    pipeline.otv_probe = lambda p, r: 0.9
    res = pipeline.route("prompt", "response", otv_threshold=0.8, odar_threshold=0.3)
    assert res == 'fast_path_otv'

def test_route_fast_path_odar():
    pipeline = VerifyRepairPipeline(
        model=None, domains=[], max_repairs=1, extractor=None,
        semantic_grounding_verifier=None, semantic_verifier_v2=None, timeout_seconds=10
    )
    # Mock otv_probe to return low confidence
    pipeline.otv_probe = lambda p, r: 0.2
    # Ensure ODAR triggers by using a very short prompt (complexity ~ 0.0) -> F < 0.3
    res = pipeline.route("short prompt", "response", otv_threshold=0.8, odar_threshold=0.3)
    assert res == 'fast_path_odar'

def test_route_deliberative():
    pipeline = VerifyRepairPipeline(
        model=None, domains=[], max_repairs=1, extractor=None,
        semantic_grounding_verifier=None, semantic_verifier_v2=None, timeout_seconds=10
    )
    # Mock otv_probe to return low confidence
    pipeline.otv_probe = lambda p, r: 0.2
    # Ensure ODAR does not trigger by using a long prompt
    long_prompt = " ".join(["word"] * 100) # complexity 1.0 -> F = 0.5 >= 0.3
    res = pipeline.route(long_prompt, "response", otv_threshold=0.8, odar_threshold=0.3)
    assert res == 'deliberative_path'
