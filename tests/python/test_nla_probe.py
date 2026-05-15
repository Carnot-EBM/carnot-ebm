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


# --- REQ-NLA-COLLISION: feature_description_collision_rate ---


def test_collision_rate_return_structure():
    """REQ-NLA-COLLISION: result dict has all required keys with correct types."""
    probe = NLAClassProbe(d_model=16, expansion_factor=2)
    x = torch.randn(4, 16)
    result = probe.feature_description_collision_rate(x)
    assert "collision_rate" in result
    assert "n_features" in result
    assert "n_collision_pairs" in result
    assert "n_total_pairs" in result
    assert "cosine_threshold" in result
    assert isinstance(result["collision_rate"], float)
    assert 0.0 <= result["collision_rate"] <= 1.0


def test_collision_rate_pair_counts():
    """REQ-NLA-COLLISION: pair counts are consistent with n_features choose 2."""
    d_sae = 8  # expansion_factor=2 → d_sae=16, but use small model
    probe = NLAClassProbe(d_model=8, expansion_factor=2)  # d_sae=16
    x = torch.randn(2, 8)
    result = probe.feature_description_collision_rate(x)
    n_feat = result["n_features"]
    expected_pairs = n_feat * (n_feat - 1) // 2
    assert result["n_total_pairs"] == expected_pairs
    assert result["n_collision_pairs"] <= result["n_total_pairs"]


def test_collision_rate_identical_features_gives_high_rate():
    """REQ-NLA-COLLISION: SAE with identical decoder cols → collision_rate=1.0."""
    probe = NLAClassProbe(d_model=8, expansion_factor=2)
    # Manually set all decoder columns to the same direction
    with torch.no_grad():
        v = torch.randn(8, 1)
        v = v / v.norm()
        probe.sae.decoder.weight.data = v.expand(8, 16).clone()
    x = torch.randn(2, 8)
    result = probe.feature_description_collision_rate(x, cosine_threshold=0.99)
    assert result["collision_rate"] == pytest.approx(1.0, abs=1e-6)


def test_collision_rate_orthogonal_features_gives_zero_rate():
    """REQ-NLA-COLLISION: orthogonal features → near-zero collision rate."""
    d_model = 8
    expansion_factor = 1  # d_sae = d_model = 8 → can be fully orthogonal
    probe = NLAClassProbe(d_model=d_model, expansion_factor=expansion_factor)
    # Set decoder to identity-like orthogonal matrix
    with torch.no_grad():
        probe.sae.decoder.weight.data = torch.eye(d_model)
    x = torch.randn(2, d_model)
    result = probe.feature_description_collision_rate(x, cosine_threshold=0.95)
    # Identity columns are perfectly orthogonal → |cos| = 0 for all off-diagonal pairs
    assert result["collision_rate"] == pytest.approx(0.0, abs=1e-6)


def test_collision_rate_custom_threshold():
    """REQ-NLA-COLLISION: lowering threshold catches more collisions."""
    probe = NLAClassProbe(d_model=16, expansion_factor=2)
    x = torch.randn(4, 16)
    high_thresh = probe.feature_description_collision_rate(x, cosine_threshold=0.99)
    low_thresh = probe.feature_description_collision_rate(x, cosine_threshold=0.5)
    # More pairs should collide at a lower threshold
    assert low_thresh["n_collision_pairs"] >= high_thresh["n_collision_pairs"]
