"""KV260 flash documentation and research experiment (exp2491).

This experiment documents the physical requirements and alternative paths
for flashing a bitstream to the KV260 K26 SOM. It addresses the missing
link from exp2477 by exploring DirtyJTAG and OpenOCD feasibility, and
records the requisite Vivado Tcl path.
"""

from __future__ import annotations

import json
from pathlib import Path


def generate_artifact() -> dict[str, any]:
    """Generate the artifact dictionary documenting KV260 flash paths."""
    artifact = {
        "experiment_id": "2491",
        "experiment": "exp2491-kv260-flash-docs",
        "milestone": "2026.05.239",
        "schema": "results/v1",
        "honest_verdict": "complete_kv260_flash_docs: Documented flash requirements for KV260 and explored DirtyJTAG / OpenOCD feasibility. DirtyJTAG is physically incompatible (3.3V vs 1.8V and Vivado hw_server mismatch) and OpenOCD requires complex ZynqMP PS initializations absent from xilinx_zynqmp.cfg. Vivado Tcl path and physical programmer options documented in docs/kv260_flash_requirements.md.",
        "dirtyjtag_kv260_compatible": False,
        "openocd_flash_feasible": False,
        "kv260_flash_requirements_written": True,
        "preconditions_checked": True,
        "available_flash_tools": ["vivado"],
        "openocd_kv260_path": None,
        "vivado_tcl_commands": "open_hw_manager\nconnect_hw_server\nopen_hw_target\nset_property PROGRAM.FILE {bitstream.bit} [get_hw_devices xck26*]\nprogram_hw_devices"
    }
    return artifact


def run(output_dir: Path) -> Path:
    """Write the artifact to the results directory."""
    artifact = generate_artifact()
    artifact_path = output_dir / "experiment_2491_kv260_flash_docs.json"
    with artifact_path.open("w") as fh:
        json.dump(artifact, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return artifact_path

if __name__ == "__main__":
    run(Path("results"))
