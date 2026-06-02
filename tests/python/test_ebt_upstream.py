"""
Tests for EBT Minimal Vendoring.
References: REQ-EBT-3725, SCENARIO-EBT-3725
"""

import math
import carnot.phase3.ebt_upstream as ebt_upstream

def test_ebt_vendored_importable_and_finite():
    """
    REQ-EBT-3725: EBT Minimal Vendoring
    SCENARIO-EBT-3725: EBT Upstream Smoke Test
    """
    energy = ebt_upstream.smoke_test_cpu()
    assert isinstance(energy, float), "Energy should be a float"
    assert not math.isnan(energy), "Energy should be finite"
    assert not math.isinf(energy), "Energy should be finite"
