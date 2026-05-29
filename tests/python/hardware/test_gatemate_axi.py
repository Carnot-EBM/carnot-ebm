import pytest
from carnot.hardware.gatemate_driver import GateMateAXIDriver

def test_gatemate_axi_driver_read_write():
    """
    Test the mock AXI driver for GateMate n=16 RTL.
    References: REQ-HW-105, SCENARIO-HW-105
    """
    driver = GateMateAXIDriver()
    
    # Initial state should be zeros
    assert driver.read_register(0x00) == 0
    assert driver.read_register(0x04) == 0
    
    # Write to 'h' register at offset 0x00
    driver.write_register(0x00, 0x1234)
    assert driver.read_register(0x00) == 0x1234
    
    # Simulate a cycle: spins = spins ^ h
    # Initially spins is 0, so new spins = 0 ^ 0x1234 = 0x1234
    driver.tick()
    assert driver.read_register(0x04) == 0x1234
    
    # Simulate another cycle: spins = 0x1234 ^ 0x1234 = 0
    driver.tick()
    assert driver.read_register(0x04) == 0
