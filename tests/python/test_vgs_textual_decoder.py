import torch
import pytest
from carnot.inference.vgs_textual_decoder import VGSTextualConstraintLogitsProcessor

def test_vgs_textual_constraint_processor():
    """
    Test that the VGS Textual Constraint Logits Processor applies a penalty
    to the autoregressive probabilities correctly.
    
    Tests: REQ-INFER-3412, SCENARIO-INFER-3412-001
    """
    constraints = ["Must contain the word 'hello'"]
    processor = VGSTextualConstraintLogitsProcessor(constraints, penalty_weight=2.0)
    
    input_ids = torch.tensor([[1, 2, 3]])
    scores = torch.tensor([[0.5, 0.2, 0.1, 0.8]])
    
    # Process
    new_scores = processor(input_ids, scores)
    
    assert new_scores.shape == scores.shape
    # Check that scores were penalized
    assert torch.all(new_scores <= scores)
    # Check that at least one score was changed (penalized)
    assert not torch.allclose(new_scores, scores)
