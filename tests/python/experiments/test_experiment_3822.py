import os
import json
import torch
import pytest

# Ensure our scripts module can be imported
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from scripts.experiments.experiment_3822_trm_escapes_grids_p1 import (
    ARModel, 
    TRMModel, 
    ParityDataset, 
    collate_fn
)

def test_ar_model_forward():
    """
    REQ-KONA-040, SCENARIO-KONA-040:
    Verifies AR baseline substrate forwards.
    """
    model = ARModel(vocab_size=3, d_model=16, num_layers=1)
    x = torch.randint(0, 2, (4, 10))
    logits = model(x)
    assert logits.shape == (4, 10, 3)

def test_trm_model_forward():
    """
    REQ-KONA-040, SCENARIO-KONA-040:
    Verifies TRM refiner substrate forwards.
    """
    model = TRMModel(vocab_size=3, d_model=16, nhead=2, dim_feedforward=32, iters=2)
    x = torch.randint(0, 2, (4, 10))
    logits = model(x)
    assert logits.shape == (4, 10, 3)

def test_parity_dataset_logic():
    """
    REQ-KONA-040: Verify the 1D task logic.
    """
    ds = ParityDataset(num_samples=10, min_len=5, max_len=10)
    assert len(ds) == 10
    
    # Check correctness of parity label
    x, y = ds[0]
    expected_y = []
    c = 0
    for bit in x.tolist():
        c ^= bit
        expected_y.append(c)
        
    assert y.tolist() == expected_y

def test_collate_fn_pads_correctly():
    ds = ParityDataset(num_samples=10, min_len=5, max_len=10)
    batch = [ds[0], ds[1]]
    x_padded, y_padded, lengths = collate_fn(batch)
    
    assert x_padded.shape == y_padded.shape
    assert x_padded.shape[0] == 2
    assert x_padded.shape[1] == max(lengths)
    
    for i, l in enumerate(lengths):
        assert torch.all(x_padded[i, l:] == 2)
        assert torch.all(y_padded[i, l:] == 2)
