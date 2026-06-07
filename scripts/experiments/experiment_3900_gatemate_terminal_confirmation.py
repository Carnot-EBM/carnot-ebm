#!/usr/bin/env python3
"""Runner for Exp 3900 GateMate terminal confirmation.

Spec refs: REQ-HW-3900, SCENARIO-HW-3900.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))


def main() -> None:
    from carnot.experiment_3900_gatemate_terminal_confirmation import (
        ARTIFACT_FILENAME,
        resolve_toolchain_path,
        run_experiment,
    )

    resolve_toolchain_path()
    artifact = run_experiment(repo_root=REPO_ROOT)
    print(f"artifact: results/{ARTIFACT_FILENAME}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"gatemate_bitstream_flashed: {artifact['gatemate_bitstream_flashed']}")
    print(f"readback_supported: {artifact['readback_supported']}")
    print(f"readback_verified: {artifact['readback_verified']}")
    print(f"terminal_state_reached: {artifact['terminal_state_reached']}")
    print(f"duration_s: {artifact['duration_s']}")
    print(f"run_duration_s: {artifact['run_duration_s']}")


if __name__ == "__main__":  # pragma: no cover
    main()
