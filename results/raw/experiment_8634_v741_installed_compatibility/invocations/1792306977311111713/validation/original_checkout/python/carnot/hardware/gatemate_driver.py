"""
Mock driver for the GateMate n=16 AXI-Lite interface.

This driver mocks the memory-mapped interface added in REQ-HW-105.
"""

class GateMateAXIDriver:
    def __init__(self):
        """Initialize the mock AXI driver."""
        self.registers = {
            0x00: 0,  # h register
            0x04: 0   # spins register
        }

    def write_register(self, offset: int, value: int) -> None:
        """
        Write a value to a memory-mapped register.
        """
        if offset in self.registers:
            self.registers[offset] = value & 0xFFFFFFFF
        else:
            raise ValueError(f"Invalid register offset: {hex(offset)}")

    def read_register(self, offset: int) -> int:
        """
        Read a value from a memory-mapped register.
        """
        if offset in self.registers:
            return self.registers[offset]
        else:
            raise ValueError(f"Invalid register offset: {hex(offset)}")

    def tick(self) -> None:
        """
        Simulate one clock cycle of the core logic:
        spins = spins ^ h
        """
        h = self.registers[0x00]
        spins = self.registers[0x04]
        # In the RTL, delta is spins ^ h. Then spins is assigned to delta.
        # Since it's a 16-bit register, mask with 0xFFFF.
        self.registers[0x04] = (spins ^ h) & 0xFFFF
