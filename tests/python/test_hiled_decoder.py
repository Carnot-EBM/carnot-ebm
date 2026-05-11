"""Tests for the HILED Hardware Integration Prototype for Energy Scoring and Decoding.

Spec: REQ-HW-054, SCENARIO-HW-054
"""

import pytest
import asyncio
from carnot.inference.hiled_decoder import HiledDecoder

def test_hiled_decoder_async_polling():
    """Test that HILED decoder can asynchronously poll the FPGA simulator.
    
    Spec: REQ-HW-054, SCENARIO-HW-054
    """
    decoder = HiledDecoder(simulator_axi_endpoint="mock_axi", max_steps=5)
    
    # Run the minimization asynchronously
    energy = asyncio.run(decoder.minimize_energy_async(initial_state=[1, -1, 1, -1]))
    
    # Verify the energy goes down and polling steps were called
    assert energy < 0
    assert decoder.steps_polled > 0
    assert decoder.steps_polled <= 5

def test_hiled_decoder_sync_wrapper():
    """Test the synchronous wrapper for the decoder."""
    decoder = HiledDecoder(simulator_axi_endpoint="mock_axi", max_steps=5)
    energy = decoder.minimize_energy(initial_state=[1, -1, 1, -1])
    assert energy < 0
    assert decoder.steps_polled > 0

def test_hiled_decoder_software_fallback():
    """Test the software fallback for the decoder.
    
    Spec: REQ-HW-055, SCENARIO-HW-055
    """
    decoder = HiledDecoder(simulator_axi_endpoint="mock_axi", max_steps=5)
    energy = decoder.minimize_energy_software(initial_state=[1, -1, 1, -1])
    assert energy < 0
    assert decoder.steps_polled > 0
