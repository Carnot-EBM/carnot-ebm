import pytest
from unittest.mock import MagicMock
from carnot.pipeline.verify_repair import VerifyRepairPipeline

def test_crane_balance_ratio_property():
    pipeline = VerifyRepairPipeline(balance_ratio=0.7)
    assert pipeline.balance_ratio == 0.7
    assert pipeline.get_balance_ratio() == 0.7

def test_crane_baseline_skip():
    # If we set balance_ratio to 0.0, we should ALWAYS get the baseline score (CRANE_FREE)
    pipeline = VerifyRepairPipeline(balance_ratio=0.0)
    pipeline.extract_typed_reasoning = MagicMock(return_value=None)
    pipeline.verify_semantic_grounding = MagicMock(return_value=None)
    pipeline.verify_semantic_verifier_v2 = MagicMock(return_value=None)
    
    result = pipeline.verify("question", "response")
    assert result.verified is True
    assert result.skipped is True
    assert result.mode == "CRANE_FREE"
    assert result.energy == 0.0

def test_crane_full_eval():
    # If balance_ratio is 1.0, we should NOT get CRANE_FREE mode
    pipeline = VerifyRepairPipeline(balance_ratio=1.0)
    pipeline.extract_typed_reasoning = MagicMock(return_value=None)
    pipeline.verify_semantic_grounding = MagicMock(return_value=None)
    pipeline.verify_semantic_verifier_v2 = MagicMock(return_value=None)
    
    # Mocking extractor so it returns an empty list, which leads to FULL eval.
    pipeline._extractor = MagicMock()
    pipeline._extractor.extract.return_value = []
    
    result = pipeline.verify("question", "response")
    assert result.mode != "CRANE_FREE"
