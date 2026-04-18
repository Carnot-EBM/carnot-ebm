"""carnot.hardware — hardware acceleration backends (NPU, FPGA, GPU).

Exports the AMD XDNA NPU runner (IRON toolchain) for pip-only NPU inference.

Spec: REQ-HARDWARE-010, REQ-HARDWARE-011, REQ-HARDWARE-012
"""

from carnot.hardware.iron_runner import IRONRunner, NPUEnvironment

__all__ = ["IRONRunner", "NPUEnvironment"]
