#!/usr/bin/env python3
"""Runner for Exp 3866 GateMate Ising tile flash v2.

Spec refs: REQ-HW-109, SCENARIO-HW-109.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))


def main() -> None:
    from carnot.experiment_3866_gatemate_ising_tile_flash_v2 import (
        ARTIFACT_FILENAME,
        resolve_toolchain_path,
        run_experiment,
    )

    resolve_toolchain_path()
    artifact = run_experiment(repo_root=REPO_ROOT)
    print(f"artifact: results/{ARTIFACT_FILENAME}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"synth_pnr_pack_succeeded: {artifact['synth_pnr_pack_succeeded']}")
    print(f"gatemate_bitstream_flashed: {artifact['gatemate_bitstream_flashed']}")
    print(f"fmax_mhz: {artifact['fmax_mhz']}")
    print(f"sample_timing_us: {artifact['sample_timing_us']}")


if __name__ == "__main__":  # pragma: no cover
    main()
