#!/usr/bin/env python3
"""Runner for Exp 3347 GateMate n=16 Ising tile build + detect smoke (v2).

This thin wrapper makes the OSS CAD Suite GateMate toolchain discoverable and
invokes :func:`carnot.experiment_3347_gatemate_n16_ising_tile_bitstream_build_smoke_v2.run_experiment`.

Why prepend ``/opt/oss-cad-suite/bin`` to ``PATH``: the modern GateMate flow
(``nextpnr-himbaechel`` + ``gmpack``) ships only inside the OSS CAD Suite, while
the host ``/usr/bin/yosys`` is a different build. Pinning the whole flow to one
suite keeps the recorded tool versions internally consistent.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

OSS_CAD_SUITE_BIN = "/opt/oss-cad-suite/bin"


def _ensure_toolchain_on_path() -> None:
    """Prepend the OSS CAD Suite bin dir so the GateMate flow is found first."""
    if os.path.isdir(OSS_CAD_SUITE_BIN):
        current = os.environ.get("PATH", "")
        if OSS_CAD_SUITE_BIN not in current.split(os.pathsep):
            os.environ["PATH"] = os.pathsep.join([OSS_CAD_SUITE_BIN, current])


def main() -> None:
    _ensure_toolchain_on_path()
    from carnot.experiment_3347_gatemate_n16_ising_tile_bitstream_build_smoke_v2 import (
        run_experiment,
    )

    artifact = run_experiment(repo_root=REPO_ROOT)
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"build_succeeded: {artifact['build_succeeded']}")
    print(f"dirtyjtag_detected: {artifact['dirtyjtag_detected']}")
    print(f"flash_smoke_status: {artifact['flash_smoke_status']}")
    print(f"bitstream_path: {artifact['bitstream_path']}")
    print(f"bitstream_checksum: {artifact['bitstream_checksum']}")
    print(f"blocked_reasons: {artifact['blocked_reasons']}")


if __name__ == "__main__":
    main()
