#!/usr/bin/env python3
"""Experiment 525 — ExpandedGPUReaper dry-run audit.

**What this experiment validates (RETRO-033, seven consecutive missed milestones):**
    GPUVRAMGateV2 (Exp 487) and JITVRAMCheck (Exp 513) both run before every GPU
    experiment but miss stale pytest orphan processes because their name-based
    whitelists cannot distinguish 'python3 -u -c ...' children of a dead pytest run
    from legitimate conductor subagent children.

    ExpandedGPUReaper (REQ-INFRA-067/068/069) fixes this by using process-subtree
    membership instead of name matching.  This script runs the reaper in dry_run=True
    mode to audit what WOULD be killed without actually sending SIGKILL — safe for CI
    and for human review before enabling live reaping.

    Deliverable schema: carnot.expanded_gpu_reaper.v1
    Expected honest_verdict: 'reap_dry_run_complete' (GPU host) or
                             'no_nvidia_smi_no_reap' (CI / no CUDA driver)

Spec: REQ-INFRA-067, REQ-INFRA-069,
      SCENARIO-INFRA-076, SCENARIO-INFRA-078
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# apply_env_autofix() MUST be called before any other import that touches GPU state.
# This injects CARNOT_FORCE_LIVE=1 when GPU hardware is detected but the env var is
# absent — a recurring cause of blocked experiments (RETRO-022).
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
sys.path.insert(0, str(Path(__file__).parent))

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

apply_env_autofix()

from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.expanded_gpu_reaper import (  # noqa: E402
    ExpandedGPUReaper,
    ExpandedGPUReaperConfig,
)
from experiment_template import ExperimentTemplate  # noqa: E402

DELIVERABLE = "results/experiment_525_expanded_gpu_reaper.json"
EXP_ID = 525


def main() -> None:
    with ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=15):
        tmpl = ExperimentTemplate(
            EXP_ID,
            "Expanded GPU Reaper",
            DELIVERABLE,
            requires_gpu=False,  # dry_run audit; no actual GPU ops needed
        )
        tmpl.setup()

        # Run the reaper in dry_run mode — compute candidates without killing anything.
        # dry_run=True is required for this experiment: a wrong kill in a live system
        # during the audit phase would be catastrophic and irreversible.
        cfg = ExpandedGPUReaperConfig(
            min_vram_mb=1024,
            min_age_s=1800,
            dry_run=True,
        )
        reaper = ExpandedGPUReaper(cfg)
        result = reaper.reap()

        # Candidates are the skipped entries with reason='dry_run_candidate' —
        # these are the processes that WOULD be killed in a live run.
        candidates = [
            entry for entry in result.skipped if entry.get("reason") == "dry_run_candidate"
        ]

        artifact = tmpl.build_result(
            {
                "schema": "carnot.expanded_gpu_reaper.v1",
                "honest_verdict": result.honest_verdict,
                "candidates": candidates,
                "min_vram_mb": cfg.min_vram_mb,
                "min_age_s": cfg.min_age_s,
                "all_skipped": result.skipped,
                "retro_033_reaper_deployed": True,
            },
            status="success",
        )

        output_path = Path(DELIVERABLE)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(artifact, indent=2) + "\n")

        # assert_deliverable_written() must be the final line — it raises
        # FileNotFoundError if the deliverable was not written, making the
        # failure observable to the conductor rather than silently succeeding.
        tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
