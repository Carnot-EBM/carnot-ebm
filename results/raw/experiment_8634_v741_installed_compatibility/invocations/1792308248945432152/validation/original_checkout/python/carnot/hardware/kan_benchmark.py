"""KAN hardware benchmark accounting."""

def compute_bops(num_points: int, num_edges: int = 1) -> int:
    """
    Compute Bit Operations (BOPs) for KAN LUTs.
    
    Args:
        num_points: Number of points in the LUT.
        num_edges: Number of edges.
        
    Returns:
        The estimated Bit Operations.
    """
    return num_points * 8 * num_edges

def compute_nabs(num_points: int, num_edges: int = 1) -> int:
    """
    Compute Number of Additions and Bit-Shifts (NABS) for KAN LUTs.
    
    Args:
        num_points: Number of points in the LUT.
        num_edges: Number of edges.
        
    Returns:
        The estimated Number of Additions and Bit-Shifts.
    """
    return num_points * 4 * num_edges
