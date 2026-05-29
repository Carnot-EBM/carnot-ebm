import numpy as np
from carnot.models.kan import KAN


def simulate_int8_quantization(x: np.ndarray) -> np.ndarray:
    """Simulate 8-bit quantization by scaling, rounding to INT8, and dequantizing.

    Args:
        x: The array to quantize.

    Returns:
        The quantized-then-dequantized array matching the original shape.
    """
    if x.size == 0:
        return x
    
    max_val = np.abs(x).max()
    if max_val == 0:
        return x
        
    scale = 127.0 / max_val
    x_quant = np.clip(np.round(x * scale), -127, 127)
    return x_quant / scale


class KaneleQuantizedKAN(KAN):
    """KAN model with 8-bit Quantization Aware Training (QAT) simulation.
    
    This implements KANELE (arXiv:2512.12850) mapping of KAN splines to FPGA LUTs
    via 8-bit quantization of coefficients and activations.
    """
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Quantize the coefficients
        self.coefficients = simulate_int8_quantization(self.coefficients)
        
    def basis(self, x: np.ndarray) -> np.ndarray:
        """Return the clipped piecewise-linear spline basis, quantized to 8 bits."""
        # Get standard basis
        b = super().basis(x)
        # Quantize activations
        return simulate_int8_quantization(b)
