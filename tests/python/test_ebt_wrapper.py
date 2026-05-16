import pytest
from carnot.models.boltzmann.ebt_wrapper import EBTWrapper, MODEL_SPECS

def test_ebt_wrapper_initialization():
    wrapper = EBTWrapper("unsloth/gemma-4-31B-it-GGUF")
    assert wrapper.model_id == "unsloth/gemma-4-31B-it-GGUF"
    assert "unsloth/gemma-4-31B-it-GGUF" in MODEL_SPECS

def test_score_trace():
    wrapper = EBTWrapper("unsloth/gemma-4-31B-it-GGUF")
    trace = ["Step 1 is logical", "therefore Step 2"]
    # "therefore" gives -2.0, so energy is max(0.0, -2.0) = 0.0
    energy = wrapper.score_trace(trace)
    assert energy == 0.0
    
    trace_bad = ["Step 1", "This is a contradiction"]
    # "contradiction" gives +10.0, energy = 10.0
    energy_bad = wrapper.score_trace(trace_bad)
    assert energy_bad == 10.0

def test_energy_guided_decoding():
    wrapper = EBTWrapper("unsloth/gemma-4-31B-it-GGUF")
    initial_trace = ["Start step"]
    candidates = ["therefore it works", "This is a contradiction", "neutral step"]
    
    # "therefore it works" -> -2.0 -> max(0, -2) = 0.0
    # "This is a contradiction" -> 10.0
    # "neutral step" -> 0.0
    best, min_energy = wrapper.energy_guided_decoding(initial_trace, candidates)
    
    assert best == "therefore it works"
    assert min_energy == 0.0
    
def test_energy_guided_decoding_empty_candidates():
    wrapper = EBTWrapper("unsloth/gemma-4-31B-it-GGUF")
    best, min_energy = wrapper.energy_guided_decoding(["Start step"], [])
    assert best == ""
    assert min_energy == float("inf")
