"""REQ-VERIFY-8596: bound fresh execution and reuse exact-byte publication consumers."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import signal
import sys

from carnot.reporting.v709_execution import progress


def main(argv: list[str] | None = None) -> int:
    """Runtime dates describe this invocation; private roots expose missing-input controls."""
    args = sys.argv[1:] if argv is None else argv
    if "--worker" not in args:
        progress("8596_start_no_model_load_MODEL_SPECS=[]_current_llm_calls=0")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--config-hash", default="")
    parser.add_argument("--arm", choices=["python", "native"], default="python")
    parser.add_argument("--timeout", type=float, default=15)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--consume", type=Path)
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--input-root", type=Path)
    parsed = parser.parse_args(args)
    try:
        if len(parsed.date) != 8:
            return 2
        datetime.strptime(parsed.date, "%Y%m%d")
    except ValueError:
        return 2
    if parsed.worker:
        from carnot.reporting import resident_complete_worker_8584 as worker

        return int(worker.serve(parsed.worker, parsed.config_hash, parsed.arm))
    from carnot.reporting import resident_complete_cost_8596 as e
    from carnot.reporting import resident_complete_runner_8584 as runner

    if (parsed.input_root and not parsed.private_output) or (
        parsed.private_output and parsed.private_output.resolve().is_relative_to(e.ROOT)
    ):
        return 2
    with e.bindings(parsed.date, parsed.input_root):
        if parsed.cold_replay:
            return int(not e.base.replay(parsed.cold_replay))
        if parsed.consume:
            print(json.dumps(e.base.consume(parsed.consume)), flush=True)
            return 0
        timer = signal.setitimer(signal.ITIMER_REAL, 4800)
        try:
            return int(
                runner.run(
                    parsed.private_output or e.ROOT / "results" / (e.NAME + ".json"),
                    private=parsed.private_output is not None,
                )
            )
        finally:
            signal.setitimer(signal.ITIMER_REAL, *timer)
