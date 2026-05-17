import pytest
import numpy as np
from carnot.inference.ttc_controller import TTCController

def test_ttc_controller_baseline_budget():
    """
    Test SCENARIO-PREM-003, REQ-PREM-004, REQ-PREM-005
    """
    controller = TTCController(base_budget=10, max_budget=100, scaling_factor=2.0)
    # Low variance
    energy_history = [1.0, 1.05, 0.95, 1.0]
    budget = controller.get_budget(energy_history)
    # Variance is very small (around 0.00125), budget should be close to base_budget
    assert budget == 10

def test_ttc_controller_expanded_budget():
    """
    Test SCENARIO-PREM-004, REQ-PREM-004, REQ-PREM-005
    """
    controller = TTCController(base_budget=10, max_budget=100, scaling_factor=10.0)
    # High variance
    energy_history = [1.0, 5.0, -2.0, 10.0]
    budget = controller.get_budget(energy_history)
    # Variance is large (approx 19.5), budget should be expanded
    assert budget > 10
    assert budget <= 100

def test_ttc_controller_short_history():
    """
    Test REQ-PREM-005 with history < 2
    """
    controller = TTCController()
    assert controller.get_budget([1.0]) == 10
    assert controller.get_budget([]) == 10

def test_ttc_controller_max_budget():
    """
    Test REQ-PREM-005 capping at max_budget
    """
    controller = TTCController(base_budget=10, max_budget=50, scaling_factor=1000.0)
    energy_history = [1.0, 100.0]
    budget = controller.get_budget(energy_history)
    assert budget == 50
