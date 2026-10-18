#!/usr/bin/env python3
"""Driver for Phase 2a of the ARC live-agent improvement plan: re-run exp6199's gemma
think-vs-no-think induction A/B on an EXPANDED roster (16 games instead of the original 12),
now that Phase 0b's two confounds are fixed (both remaining reasoning-suppressing fences gated,
repeat_penalty forwarded on the chat route) AND a third confound found while smoke-testing this
driver is fixed (see experiment_6199_gemma_think_mode_ab.py's _configure_arm docstring: the
no_think arm used to fall through to the now-flipped-on ARC_LIVE_GENERATOR_THINK_SCORED_DEFAULT
instead of forcing itself off).

Deliberately reuses experiment_6199_gemma_think_mode_ab.build_artifact() unmodified rather than
forking it -- that module already carries the checkpoint/resume mechanism (this exact class of
long-running GPU job has been killed mid-flight three times before by session/background-task
cleanup, per its own module docstring) and the treatment-fire preconditions. This driver only
supplies a wider `roster` tuple and writes to a SEPARATE output path so it never overwrites the
original 12-game exp6199 artifact (Finding 3 of the improvement plan already cites that artifact
by its exact numbers; this is a new, wider measurement, not a silent rewrite of the old one).

CARNOT_ARC_INDUCE_N_CTX must be set to 32768 (not the module's own ~106496 default) by the
caller's environment -- that default is sized for the SCORED swarm's 4-concurrent-games worst
case, but this script issues induce requests strictly SEQUENTIALLY (one game/arm at a time), so
it never needs more than a single slot's worth of context. Verified directly: at the default
n_ctx the CUDA guard refuses the single RTX 3090 outright (needs 26624 MiB, card has 24576 MiB
total); at 32768 it fits with room to spare.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.experiment_6199_gemma_think_mode_ab import (  # noqa: E402
    DEFAULT_ROSTER,
    build_artifact,
)

OUT_PATH = REPO_ROOT / "results/experiment_6221_gemma_think_mode_ab_expanded_roster.json"

# Four additional games beyond exp6199's original 12, chosen from the un-tested remainder of the
# 25-game public roster (r11l, sc25, s5i5, lp85 -- all previously exercised without reported
# window-build hangs elsewhere this session, unlike e.g. tr87's documented 8-minute stalls in a
# DIFFERENT harness's window builder). sc25 has no usable level-up window under THIS script's
# window-builder (confirmed by smoke test: "no_levelup_window": true) and will record as such,
# not as an error.
EXPANDED_ROSTER = tuple(DEFAULT_ROSTER) + ("r11l", "sc25", "s5i5", "lp85")


def main() -> int:
    if os.environ.get("CARNOT_ARC_INDUCE_N_CTX") != "32768":
        print(
            "REFUSING: CARNOT_ARC_INDUCE_N_CTX must be set to 32768 in this process's "
            "environment (see module docstring) -- it is currently "
            f"{os.environ.get('CARNOT_ARC_INDUCE_N_CTX')!r}.",
            file=sys.stderr,
        )
        return 2
    print(f"roster ({len(EXPANDED_ROSTER)} games): {EXPANDED_ROSTER}", flush=True)
    artifact = build_artifact(roster=EXPANDED_ROSTER)
    OUT_PATH.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")
    print(f"wrote {OUT_PATH} -- honest_verdict={artifact['honest_verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
