import pytest
import torch
from carnot.verify.nla_probe import MinimalSAE, NLAClassProbe

def test_minimal_sae():
    sae = MinimalSAE(d_model=16, expansion_factor=2)
    x = torch.randn(5, 16)
    decoded, encoded = sae(x)
    assert decoded.shape == (5, 16)
    assert encoded.shape == (5, 32)
    
    mse = sae.reconstruction_error(x)
    assert mse.shape == (5,)

def test_nla_class_probe():
    probe = NLAClassProbe(d_model=16, expansion_factor=2)
    optimizer = torch.optim.Adam(probe.sae.parameters(), lr=1e-3)
    
    x = torch.randn(5, 16)
    loss = probe.train_step(x, optimizer)
    assert isinstance(loss, float)
    
    score = probe.score("prompt", "candidate", x)
    assert 0.0 <= score <= 1.0
