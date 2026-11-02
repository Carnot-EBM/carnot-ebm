"""Run the existing qualification path with current identity (REQ-REPORT-7916-V687)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from carnot.verify import training_qualification_7916 as current  # noqa: E402

MODEL_SPECS: list[dict[str, str]] = []


def main(argv: list[str] | None = None) -> int:
    """Use explicit private fixture paths while the default route qualifies current work."""
    print("[exp7916] phase=start_flushed completed=0", flush=True)
    q = current.runtime()
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, choices=["20260930"])
    parser.add_argument(
        "--upstream",
        type=Path,
        default=q.ROOT / "results/experiment_7892_v685_source_boundary.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=q.ROOT / "results/experiment_7916_v687_training_qualification.json",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=q.ROOT / "results/raw/experiment_7916_v687_training_qualification",
    )
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--fit-deadline-s", type=float, default=90)
    parser.add_argument("--terminal-recheck", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.cold_replay:
            q.replay(args.cold_replay)
            return 0
        if not args.fixture:
            return int(q.qualify(args.upstream, args.output, args.raw_root, args.date))
        result = q.fit_score(
            args.upstream,
            args.output,
            args.raw_root,
            7916,
            args.date,
            args.resume,
            args.fit_deadline_s,
        )
    except q.custody.InputBlocked as exc:
        result = q.base(7916, args.date, [])
        result.update(
            {
                "honest_verdict": "complete_blocked_source_evidence",
                "verdict_class": "blocked",
                "gate_check_summary": exc.operands,
            }
        )
        result["acceptance_gate_results"].update({"validity": False, "readiness": 0})
    except (ValueError, TimeoutError, KeyError) as exc:
        if args.cold_replay:
            q.progress("replay_mismatch")
            return 2
        result = q.base(7916, args.date, [])
        q.disqualify(
            result,
            [
                {
                    "upstream_id": "owned_fitting",
                    "artifact_path": str(args.upstream),
                    "artifact_sha256": None,
                    "field": "fitting_runtime",
                    "op": "completes",
                    "expected": True,
                    "observed": str(exc),
                }
            ],
        )
    return 0 if q.publish(result, args.output, args.raw_root, args.terminal_recheck) else 2


if __name__ == "__main__":
    raise SystemExit(main())
