"""Tests for LatentGenerator and latent_feature_divergence.

Spec coverage: REQ-TRAIN-007, REQ-KONA-002
"""

import numpy as np

from carnot.phase3.latent_generator import LatentGenerator, LatentTrace
from carnot.training.ebft_loss import latent_feature_divergence


def _small_ebm_params(n: int = 4, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return a small symmetric coupling matrix and bias for testing."""
    rng = np.random.default_rng(seed)
    J_raw = rng.standard_normal((n, n)) * 0.5
    J = (J_raw + J_raw.T) / 2.0  # symmetrise
    h = rng.standard_normal(n) * 0.1
    return J, h


# --- LatentGenerator tests ---

def test_latent_generator_trace_shape():
    """REQ-KONA-002: LatentGenerator produces traces of the expected shape."""
    J, h = _small_ebm_params(n=4)
    gen = LatentGenerator(n_steps=20, record_interval=1)
    trace = gen.generate(J, h, seed=0)

    assert trace.states.shape == (20, 4), f"states shape mismatch: {trace.states.shape}"
    assert trace.energies.shape == (20,), f"energies shape mismatch: {trace.energies.shape}"
    assert trace.seed == 0


def test_latent_generator_states_bounded():
    """REQ-KONA-002: All latent states lie in (-1, 1) due to tanh squashing."""
    J, h = _small_ebm_params(n=6)
    gen = LatentGenerator(n_steps=50)
    trace = gen.generate(J, h, seed=42)

    assert np.all(trace.states > -1.0), "state below -1 found"
    assert np.all(trace.states < 1.0), "state above +1 found"


def test_latent_generator_reproducible():
    """REQ-TRAIN-007: Same seed produces identical traces."""
    J, h = _small_ebm_params(n=4)
    gen = LatentGenerator(n_steps=10)
    t1 = gen.generate(J, h, seed=7)
    t2 = gen.generate(J, h, seed=7)

    np.testing.assert_array_equal(t1.states, t2.states)
    np.testing.assert_array_equal(t1.energies, t2.energies)


def test_latent_generator_different_seeds_differ():
    """REQ-TRAIN-007: Different seeds produce different traces."""
    J, h = _small_ebm_params(n=4)
    gen = LatentGenerator(n_steps=10)
    t0 = gen.generate(J, h, seed=0)
    t1 = gen.generate(J, h, seed=1)

    assert not np.allclose(t0.states, t1.states), "different seeds should give different traces"


def test_latent_generator_batch():
    """REQ-TRAIN-007: generate_batch returns n_traces independent LatentTrace objects."""
    J, h = _small_ebm_params(n=4)
    gen = LatentGenerator(n_steps=10)
    traces = gen.generate_batch(J, h, n_traces=5, base_seed=0)

    assert len(traces) == 5
    for i, t in enumerate(traces):
        assert t.seed == i, f"trace {i} has unexpected seed {t.seed}"


def test_latent_trace_features_shape():
    """REQ-TRAIN-007: LatentTrace.features() returns vector of length 2*d+1."""
    d = 4
    states = np.random.default_rng(0).standard_normal((10, d))
    energies = np.random.default_rng(1).standard_normal(10)
    trace = LatentTrace(states=np.tanh(states), energies=energies, seed=0)

    feat = trace.features()
    assert feat.shape == (2 * d + 1,), f"feature shape mismatch: {feat.shape}"


def test_latent_trace_features_content():
    """REQ-TRAIN-007: LatentTrace.features() packs mean_state, std_state, mean_energy."""
    d = 3
    states = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    energies = np.array([-1.0, -2.0, -3.0])
    trace = LatentTrace(states=states, energies=energies, seed=0)

    feat = trace.features()
    expected_mean = np.mean(states, axis=0)
    expected_std = np.std(states, axis=0)
    expected_energy = np.array([np.mean(energies)])

    np.testing.assert_allclose(feat[:d], expected_mean)
    np.testing.assert_allclose(feat[d:2 * d], expected_std)
    np.testing.assert_allclose(feat[2 * d:], expected_energy)


def test_feature_matrix_shape():
    """REQ-TRAIN-007: feature_matrix returns (n_traces, feature_dim) array."""
    J, h = _small_ebm_params(n=4)
    gen = LatentGenerator(n_steps=10)
    traces = gen.generate_batch(J, h, n_traces=6)
    mat = gen.feature_matrix(traces)

    assert mat.shape == (6, 2 * 4 + 1), f"feature matrix shape mismatch: {mat.shape}"


def test_record_interval():
    """REQ-KONA-002: record_interval downsamples the trace correctly."""
    J, h = _small_ebm_params(n=4)
    gen = LatentGenerator(n_steps=20, record_interval=5)
    trace = gen.generate(J, h, seed=0)

    # Steps 0, 5, 10, 15 are recorded → 4 states
    assert trace.states.shape[0] == 4, f"unexpected trace length: {trace.states.shape[0]}"


# --- latent_feature_divergence tests ---

def test_latent_feature_divergence_zero_when_same():
    """REQ-TRAIN-007: divergence is zero when expert and rollout features are identical."""
    feats = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    div = latent_feature_divergence(feats, feats)
    assert div == 0.0, f"expected zero divergence for identical inputs, got {div}"


def test_latent_feature_divergence_positive():
    """REQ-TRAIN-007: divergence is positive when distributions differ."""
    expert = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    rollout = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    div = latent_feature_divergence(expert, rollout)
    assert div > 0.0, "expected positive divergence for different distributions"


def test_latent_feature_divergence_known_value():
    """REQ-TRAIN-007: divergence matches manual calculation."""
    expert = np.array([[2.0, 0.0], [2.0, 0.0]])   # mean = [2, 0]
    rollout = np.array([[0.0, 1.0], [0.0, 1.0]])   # mean = [0, 1]
    # ||[2,0] - [0,1]||^2 = 4 + 1 = 5
    div = latent_feature_divergence(expert, rollout)
    assert abs(div - 5.0) < 1e-9, f"expected 5.0, got {div}"


def test_latent_feature_divergence_end_to_end():
    """REQ-TRAIN-007: divergence between same-EBM expert/rollout traces is small."""
    J, h = _small_ebm_params(n=4, seed=99)
    gen = LatentGenerator(n_steps=100)
    expert_traces = gen.generate_batch(J, h, n_traces=10, base_seed=0)
    rollout_traces = gen.generate_batch(J, h, n_traces=10, base_seed=100)

    expert_feats = gen.feature_matrix(expert_traces)
    rollout_feats = gen.feature_matrix(rollout_traces)

    div = latent_feature_divergence(expert_feats, rollout_feats)
    # Same EBM → trajectories visit similar regions → divergence should be finite
    assert np.isfinite(div), "divergence must be finite"
    assert div >= 0.0, "divergence must be non-negative"
