import pytest
from carnot.inference.samplers import GCoTBranchingSampler

def dummy_energy(content: str) -> float:
    # A simple mock energy function: longer is slightly higher energy,
    # but containing "error" gives a huge penalty.
    energy = len(content) * 0.1
    if "error" in content:
        energy += 10.0
    return energy

def test_gcot_branching_initialization():
    sampler = GCoTBranchingSampler(energy_fn=dummy_energy, energy_threshold=5.0)
    sampler.initialize("start")
    assert len(sampler.branches) == 1
    assert sampler.branches[0].content == "start"
    assert sampler.branches[0].energy == 0.5

def test_gcot_branching_step_and_cull():
    sampler = GCoTBranchingSampler(energy_fn=dummy_energy, energy_threshold=5.0, max_branches=2)
    sampler.initialize("start")
    # "error" will be culled
    sampler.step(["good", "error", "also good"])
    
    assert len(sampler.branches) == 2
    # Should contain "start good" and "start also good"
    contents = [b.content for b in sampler.branches]
    assert "start good" in contents
    assert "start also good" in contents
    assert not any("error" in c for c in contents)

def test_gcot_branching_backtrack():
    sampler = GCoTBranchingSampler(energy_fn=dummy_energy, energy_threshold=5.0, max_branches=2)
    sampler.initialize("start")
    sampler.step(["good"])
    assert len(sampler.branches) == 1
    assert sampler.step_counter == 1
    
    # Now all extensions cause error, should backtrack
    sampler.step(["error 1", "error 2"])
    
    # Should have backtracked to step 0
    assert len(sampler.branches) == 1
    assert sampler.step_counter == 0
    assert sampler.branches[0].content == "start"

def test_gcot_branching_backtrack_empty_history():
    sampler = GCoTBranchingSampler(energy_fn=dummy_energy, energy_threshold=5.0, max_branches=2)
    sampler.initialize("start error")
    sampler.backtrack()
    assert len(sampler.branches) == 0
