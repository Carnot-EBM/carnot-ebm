#!/usr/bin/env python3
"""Exp 3431: KV260 terminal on-board latency transcript (v2 re-attempt).

Spec: REQ-HW-060 (SCENARIO-HW-060), REQ-HW-061 (SCENARIO-HW-061).

WHY THIS EXPERIMENT EXISTS (in plain terms)
-------------------------------------------
The KV260 FPGA board is Carnot's "sovereignty story" (north-star section 3):
we must show, on real hardware, that the Ising energy function can be evaluated
on a dedicated edge accelerator. To call that story *finished* we need exactly
ONE honest, non-fabricated on-board latency transcript. Recording it graduates
the board to its terminal state, after which the per-milestone hardware mandate
lifts for the KV260.

The previous attempt (exp3420, milestone .315) was honestly blocked:
``blocked_kv260_ssh_unreachable`` because ``ssh kria`` could not resolve the
board's hostname (``kv260.local`` was not on the network at run time). This v2
re-attempt re-runs the identical hardware path. If the board is reachable now,
it records the terminal transcript; if it is still unreachable, it emits the
same honest blocked verdict. We NEVER invent a transcript -- a fabricated
latency number would poison the headline sovereignty claim, which is far worse
than an honest "board was offline".

HOW THIS STAYS DRY AND AUDITED
------------------------------
All of the board interaction logic (SSH preconditions, overlay loading, UIO
register round-trip harness, latency-stat reduction, artifact construction and
schema validation) is already implemented and unit-tested in the v1 module
``experiment_3420_kv260_terminal_latency_transcript_v1``. Re-typing ~900 lines
of audited hardware code for v2 would be a fabrication risk in itself (a subtle
copy edit could change the measurement). Instead v2 *imports* the v1 module and
reuses it verbatim, contributing only:

  * v2-specific output paths (so the v2 run does not overwrite the v1 artifact),
    and
  * a small provenance relabel so the emitted artifact correctly reports
    ``experiment_id=3431`` / the v2 experiment name / today's run date.

HOW THIS STAYS HONEST ABOUT THE PRECONDITION
--------------------------------------------
The ONLY reachability precondition (inherited from v1) is that the board answers
over SSH. We deliberately do NOT check the host machine's SD-card slot -- that
is a retired, wrong-mechanism check (see CLAUDE.md "KV260 SSH-Not-SD-Card
Discipline"). The board is addressed as ``ssh kria`` and the overlay is the
XDC-constrained ``carnot_ising_v2_n64`` / ``carnot_ising_v4`` bitstream.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Load the audited v1 module by file path. We reuse its hardware logic verbatim
# rather than re-implementing it (DRY + fabrication-risk avoidance).
# ---------------------------------------------------------------------------
_V1_SCRIPT_PATH = (
    REPO_ROOT
    / "scripts"
    / "experiment_3420_kv260_terminal_latency_transcript_v1.py"
)


def _load_v1_module() -> Any:
    """Import the v1 implementation module from its file path."""
    spec = importlib.util.spec_from_file_location(
        "experiment_3420_kv260_terminal_latency_transcript_v1", _V1_SCRIPT_PATH
    )
    if spec is None or spec.loader is None:  # pragma: no cover - import guard
        raise RuntimeError(f"cannot load v1 module from {_V1_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec: Python 3.14's @dataclass machinery looks the module
    # up in sys.modules via cls.__module__ while processing the class body.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


v1 = _load_v1_module()

EXPERIMENT_ID = 3431
EXPERIMENT_NAME = "exp3431-kv260-terminal-latency-transcript-v2"
RUN_DATE = "20260530"
RESULT_PATH = (
    REPO_ROOT
    / "results"
    / "experiment_3431_kv260_terminal_latency_transcript_v2.json"
)
TRANSCRIPT_PATH = (
    REPO_ROOT
    / "results"
    / "experiment_3431_kv260_terminal_transcript_v2.log"
)


def relabel_for_v2(artifact: dict[str, Any]) -> dict[str, Any]:
    """Rewrite v1 provenance labels so the artifact reports the v2 identity.

    The v1 ``run_experiment`` stamps ``experiment_id``/``experiment``/``run_date``
    with the v1 (3420) values because those are module constants. For an honest
    v2 provenance record we must report that THIS artifact was produced by the
    v2 re-attempt. We touch ONLY the three identity labels and the recorded
    transcript path; every measured field (latencies, hashes, preconditions,
    verdict) is left exactly as v1 produced it. Mutates and returns the dict.
    """
    artifact["experiment_id"] = EXPERIMENT_ID
    artifact["experiment"] = EXPERIMENT_NAME
    artifact["run_date"] = RUN_DATE
    # Point the recorded transcript path at the v2 log we actually wrote.
    try:
        artifact["board_transcript_path"] = str(
            TRANSCRIPT_PATH.relative_to(REPO_ROOT)
        )
    except ValueError:  # pragma: no cover - transcript always under repo root
        artifact["board_transcript_path"] = str(TRANSCRIPT_PATH)
    return artifact


def run_experiment(
    executor: Any | None = None,
    *,
    result_path: Path = RESULT_PATH,
    transcript_path: Path = TRANSCRIPT_PATH,
) -> dict[str, Any]:
    """Run the v1 hardware flow with v2 paths, relabel provenance, persist.

    ``executor`` is forwarded to the v1 driver: ``None`` triggers the real SSH
    path; tests inject a fake executor to exercise both branches offline.
    """
    artifact = v1.run_experiment(
        executor,
        result_path=result_path,
        transcript_path=transcript_path,
    )
    relabel_for_v2(artifact)
    # The v1 driver already validated the artifact before returning it; our
    # relabel only changes identity strings, so it remains schema-valid. Persist
    # the relabeled version over the v1-written file.
    v1.validate_artifact(artifact)
    v1._write_json(result_path, artifact)
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-result-path", action="store_true")
    args = parser.parse_args(argv)
    artifact = run_experiment()
    if args.print_result_path:
        print(RESULT_PATH)
    else:
        print(
            json.dumps(
                {
                    "honest_verdict": artifact["honest_verdict"],
                    "kv260_terminal_state_reached": artifact[
                        "kv260_terminal_state_reached"
                    ],
                    "result": str(RESULT_PATH),
                }
            )
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
