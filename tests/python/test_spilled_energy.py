import math
from carnot.metrics.spilled_energy import compute_spilled_energy, compute_marginalized_energy

def test_compute_spilled_energy():
    logprobs = [math.log(0.5), math.log(0.25)]
    # spilled for 0.5 is 1 - 0.5 = 0.5
    # spilled for 0.25 is 1 - 0.25 = 0.75
    # mean is 0.625
    assert math.isclose(compute_spilled_energy(logprobs), 0.625)
    assert compute_spilled_energy([]) == 0.0

def test_compute_marginalized_energy():
    logprobs = [-1.0, -2.0]
    # mean is -1.5, negated is 1.5
    assert math.isclose(compute_marginalized_energy(logprobs), 1.5)
    assert compute_marginalized_energy([]) == 0.0
