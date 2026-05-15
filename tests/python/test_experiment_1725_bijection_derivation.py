import os
import pytest

def test_bijection_derivation_exists_and_contains_corollary():
    """
    Test that the theoretical derivation for the LM-EBM bijection was written
    and explicitly addresses the Phase 4 alpha_t invariance corollary.
    """
    derivation_path = "docs/research-notes/lm_ebm_bijection_derivation.md"
    assert os.path.exists(derivation_path), f"File {derivation_path} does not exist"
    
    with open(derivation_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    assert "arXiv:2512.15605v3" in content, "Missing reference to arXiv:2512.15605v3"
    assert "k=16" in content, "Missing reference to the k=16 verifier ensemble"
    assert "corollary" in content.lower(), "Missing discussion of the corollary"
    assert "alpha_t" in content, "Missing reference to the alpha_t invariance"
