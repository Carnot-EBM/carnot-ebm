import pytest
from unittest.mock import MagicMock
from carnot.pipeline.verify_repair import VerifyRepairPipeline, VerificationResult
from dataclasses import dataclass

@dataclass
class MockConstraintResult:
    constraint_type: str
    description: str

class MockPipeline(VerifyRepairPipeline):
    def __init__(self, routing_mode, energies):
        super().__init__(model=None, routing_mode=routing_mode, max_repairs=3)
        self.energies = energies
        self.iteration = 0
        self._has_model_mock = True

    @property
    def has_model(self):
        return self._has_model_mock

    def _generate(self, prompt, max_new_tokens=None):
        return "mock repaired response"

    def verify(self, question, response, domain=None, **kwargs):
        if self.iteration < len(self.energies):
            energy = self.energies[self.iteration]
        else:
            energy = self.energies[-1]
        self.iteration += 1
        return VerificationResult(
            verified=(energy <= 0),
            violations=[MockConstraintResult("mock", "mock")] if energy > 0 else [],
            energy=energy,
            constraints=[],
            certificate={}
        )

def test_odar_routing_stops_early():
    # Energies: starts at 5.0, goes down to 4.5, then gets stuck at 4.6
    # Argmax should do 3 repairs. ODAR should break early because energy goes up.
    energies = [5.0, 4.5, 4.6, 4.7]
    
    pipe_argmax = MockPipeline("argmax", energies)
    res_argmax = pipe_argmax.verify_and_repair("Q", "initial")
    assert res_argmax.iterations == 3
    
    pipe_odar = MockPipeline("odar", energies)
    res_odar = pipe_odar.verify_and_repair("Q", "initial")
    # ODAR should break at iteration 1 or 2 since energy got worse
    assert res_odar.iterations < 3
    
def test_odar_routing_continues_if_improving():
    # Energies: steady improvement
    energies = [5.0, 3.0, 1.0, 0.1]
    
    pipe_argmax = MockPipeline("argmax", energies)
    res_argmax = pipe_argmax.verify_and_repair("Q", "initial")
    assert res_argmax.iterations == 3
    
    pipe_odar = MockPipeline("odar", energies)
    res_odar = pipe_odar.verify_and_repair("Q", "initial")
    # Should continue to max_repairs because VFE is good
    assert res_odar.iterations == 3

def test_odar_routing_stops_if_verified():
    # Energies: drops to 0.0 (verified)
    energies = [5.0, 0.0]
    
    pipe_argmax = MockPipeline("argmax", energies)
    res_argmax = pipe_argmax.verify_and_repair("Q", "initial")
    assert res_argmax.iterations == 1
    
    pipe_odar = MockPipeline("odar", energies)
    res_odar = pipe_odar.verify_and_repair("Q", "initial")
    assert res_odar.iterations == 1
