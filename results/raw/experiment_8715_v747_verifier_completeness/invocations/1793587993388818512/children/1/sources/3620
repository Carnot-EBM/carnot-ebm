import numpy as np
from typing import List

class SimpleAdaptiveKAN:
    """Minimum Tier 4 prototype that demonstrates structural adaptation."""
    def __init__(self):
        self.num_knots = 3
        self.knot_centers = np.linspace(-1, 1, self.num_knots)
        self.knot_weights = np.ones(self.num_knots)
        
    def energy(self, x: float) -> float:
        diffs = x - self.knot_centers
        rbf = np.exp(- (diffs**2) / 0.1)
        return float(np.sum(self.knot_weights * rbf))
        
    def adapt_structure(self, region_x: float):
        self.num_knots += 1
        self.knot_centers = np.append(self.knot_centers, region_x)
        # New knot drops energy in this region
        self.knot_weights = np.append(self.knot_weights, -2.0)

def detect_new_pattern(violations_history: List[float]) -> List[float]:
    """If any error region appears >3 times in the last 36 examples -> trigger structural adaptation."""
    recent_history = violations_history[-36:]
    triggered_regions = []
    # Count occurrences of each region (rounded to avoid float equality issues)
    counts = {}
    for x in recent_history:
        x_rounded = round(x, 2)
        counts[x_rounded] = counts.get(x_rounded, 0) + 1
        
    for x_rounded, count in counts.items():
        if count > 3:
            triggered_regions.append(x_rounded)
            
    return triggered_regions

def adapt_structure(model: SimpleAdaptiveKAN, triggered_region: float, trigger_examples: List[float]):
    """Add 1 knot in the triggered region's energy dimension.
    Record: before_energy, after_energy on triggering examples."""
    
    # Calculate before energy
    before_energies = [model.energy(x) for x in trigger_examples if round(x, 2) == round(triggered_region, 2)]
    before_energy = float(np.mean(before_energies)) if before_energies else 0.0
    
    model.adapt_structure(triggered_region)
    
    # Calculate after energy
    after_energies = [model.energy(x) for x in trigger_examples if round(x, 2) == round(triggered_region, 2)]
    after_energy = float(np.mean(after_energies)) if after_energies else 0.0
    
    return before_energy, after_energy
