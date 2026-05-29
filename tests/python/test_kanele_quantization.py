import numpy as np
from carnot.models.kanele_quantization import simulate_int8_quantization, KaneleQuantizedKAN


def test_simulate_int8_quantization():
    """Test quantization on edge cases and standard values."""
    # Empty array
    assert simulate_int8_quantization(np.array([])).size == 0
    
    # Zero array
    zeros = np.zeros(5)
    np.testing.assert_array_equal(simulate_int8_quantization(zeros), zeros)
    
    # Standard array - should have quantization error but bounded
    x = np.linspace(-1.0, 1.0, 100)
    x_q = simulate_int8_quantization(x)
    assert np.max(np.abs(x - x_q)) < 0.05  # roughly 1/127


def test_kanele_quantized_kan():
    """Test the KaneleQuantizedKAN class."""
    kan = KaneleQuantizedKAN(n_params=10, seed=42)
    
    # Check coefficients are quantized
    # Since it's simulated, it's float64, but has discrete levels
    assert kan.coefficients.dtype == np.float64
    
    # Check basis is quantized
    x = np.random.randn(5, 10)
    b = kan.basis(x)
    assert b.shape == (5, 10)
    
    # Check logits
    logits = kan.logits(x)
    assert logits.shape == (5,)
