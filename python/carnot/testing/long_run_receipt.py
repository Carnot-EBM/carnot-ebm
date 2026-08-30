"""Leave a receipt when a long-running job dies, naming the signal that killed it.

WHY THIS EXISTS (2026-08-30). A 7-hour GPU measurement died at 38 of 39 cells and NOTHING in
the repository could say what killed it. The evidence available afterwards was:

  * the llama-server log stopping mid-decode at 23.37 tok/s, no shutdown line;
  * the runner log stopping with no traceback;
  * no OOM trace, and no actor row from `run_stop_authority.py`, which logs every action it
    takes -- so it was not that reaper.

That is enough to say the process was killed from outside and not enough to say by whom. This
repository already names that state as its own incident class: an unexplained dead process,
first recorded 2026-08-09 as "exit-143, sender never identified". It has now happened twice, and
both times the investigation ended in a plausible story rather than an answer.

A plausible story is the thing to avoid here. There are three reapers in this project
(`experiment_template.kill_gpu_zombies`, `gpu_monitor.detect_zombies`,
`run_stop_authority.py`), each with its own exemption list, and picking the one whose comments
best match the symptom is how a wrong cause gets recorded as fact.

So this does not try to identify the killer. It makes the NEXT death self-describing: the
signal number, the sender's pid where the kernel provides it, and how far the job had got. One
line of evidence at the moment of death is worth more than any amount of reconstruction after.

Deliberately NOT a fix for whatever did the killing. Naming the sender is the precondition for
fixing it, and this project has twice tried to skip that step.
"""

from __future__ import annotations

import json
import os
import signal
import time
from collections.abc import Callable
from pathlib import Path
from types import FrameType

#: Signals worth a receipt. SIGKILL cannot be caught -- that is the point of it -- so a job
#: that vanishes with NO receipt at all is itself evidence: either SIGKILL, or the OOM killer,
#: or a power/kernel event. The absence is informative only because the presence is reliable.
_CAUGHT = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT)


def install(receipt_path: Path, *, progress: Callable[[], object] | None = None) -> None:
    """Write `receipt_path` if this process is signalled, then re-raise the default action.

    `progress` is called at death to record how far the work had got -- cells banked, rows
    written -- so a receipt says "killed at 38/39" rather than only "killed".

    Re-raising rather than exiting is deliberate: a handler that swallows SIGTERM turns a
    supervisor's polite stop into a hang, and this must never make a job harder to stop than it
    was before.
    """

    started = time.time()

    def _handler(signum: int, _frame: FrameType | None) -> None:
        record: dict[str, object] = {
            "killed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "signal": signum,
            "signal_name": signal.Signals(signum).name,
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "elapsed_s": round(time.time() - started, 1),
        }
        if progress is not None:
            try:
                record["progress"] = progress()
            except Exception as exc:  # noqa: BLE001
                # A failing progress callback must never cost us the receipt itself.
                record["progress_error"] = f"{type(exc).__name__}: {exc}"
        try:
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            receipt_path.write_text(json.dumps(record, indent=2) + "\n")
        except OSError as exc:  # noqa: BLE001
            # Nothing useful left to do; dying silently is what we are fixing, so say it
            # on stderr at least.
            print(f"long-run-receipt: could not write {receipt_path}: {exc}", flush=True)
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for sig in _CAUGHT:
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            # Not the main thread, or the platform refuses this signal. Skip it rather than
            # refusing to install the others.
            continue
