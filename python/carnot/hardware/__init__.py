"""carnot.hardware — hardware acceleration backends (NPU, FPGA, GPU).

Exports the AMD XDNA NPU runner (IRON toolchain) for pip-only NPU inference,
and the KV260 FPGA Ising sampler backend (Exp 471).

Spec: REQ-HARDWARE-010, REQ-HARDWARE-011, REQ-HARDWARE-012,
      REQ-HARDWARE-013, REQ-HARDWARE-014, REQ-HARDWARE-015
"""

from carnot.hardware.fpga_backend import FpgaBackend, SparsifiedIsingConfig
from carnot.hardware.iron_runner import IRONRunner, NPUEnvironment

__all__ = ["FpgaBackend", "IRONRunner", "NPUEnvironment", "SparsifiedIsingConfig"]
