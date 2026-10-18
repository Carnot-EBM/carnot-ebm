"""KANelE Look-Up Table (LUT) conversion utilities."""

from typing import Callable, List, Tuple

def convert_kan_to_lut(
    edge_fn: Callable[[float], float],
    domain: Tuple[float, float] = (-1.0, 1.0),
    num_points: int = 256
) -> List[float]:
    """
    Evaluate a KAN edge function over a specified domain to create a LUT.

    Args:
        edge_fn: A callable representing the 1D KAN edge function.
        domain: A tuple of (min_val, max_val) defining the domain.
        num_points: The number of points to sample in the LUT.

    Returns:
        A list of evaluated float values.
    """
    if num_points < 2:
        raise ValueError("num_points must be at least 2.")
    
    start, end = domain
    step = (end - start) / (num_points - 1)
    
    lut = []
    for i in range(num_points):
        x = start + i * step
        lut.append(edge_fn(x))
        
    return lut
