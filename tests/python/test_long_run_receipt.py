"""REQ-INFRA-6830: a signalled long-running job leaves a receipt naming its killer.

A 7-hour GPU measurement died at 38/39 cells on 2026-08-29 and nothing could say what killed
it: the server log stopped mid-decode at 23.37 tok/s, the runner log stopped with no traceback,
there was no OOM trace, and `run_stop_authority.py` -- which logs every action it takes -- had
written no actor row. Enough to know it was killed from outside; not enough to know by whom.
The same state was first recorded 2026-08-09 as "exit-143, sender never identified".

These tests run the receipt in a REAL child process and signal it, because the thing under test
is behaviour at process death and an in-process unit test cannot exercise that.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

CHILD = """
import sys, time
sys.path.insert(0, {repo!r} + "/python")
from pathlib import Path
from carnot.testing.long_run_receipt import install
install(Path({receipt!r}), progress=lambda: {{"cells": 38, "of": 39}})
print("ready", flush=True)
time.sleep(60)
"""


def _spawn(tmp_path: Path, receipt: Path) -> subprocess.Popen:
    src = CHILD.format(repo=str(REPO), receipt=str(receipt))
    proc = subprocess.Popen(
        [sys.executable, "-c", src], stdout=subprocess.PIPE, text=True, cwd=str(tmp_path)
    )
    assert proc.stdout is not None
    assert proc.stdout.readline().strip() == "ready"
    return proc


def _wait_for(path: Path, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists():
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                pass  # caught mid-write
        time.sleep(0.05)
    raise AssertionError(f"no receipt at {path} within {timeout}s")


def test_sigterm_leaves_a_receipt_naming_the_signal(tmp_path: Path) -> None:
    """The exact case that went unexplained twice."""
    receipt = tmp_path / "death.json"
    proc = _spawn(tmp_path, receipt)
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=10)
    record = _wait_for(receipt)
    assert record["signal"] == int(signal.SIGTERM)
    assert record["signal_name"] == "SIGTERM"
    assert record["pid"] == proc.pid


def test_the_receipt_records_how_far_the_work_had_got(tmp_path: Path) -> None:
    """ "Killed at 38 of 39" is a different finding from "killed"."""
    receipt = tmp_path / "death.json"
    proc = _spawn(tmp_path, receipt)
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=10)
    assert _wait_for(receipt)["progress"] == {"cells": 38, "of": 39}


def test_the_process_still_dies_on_the_signal(tmp_path: Path) -> None:
    """A handler that swallows SIGTERM turns a polite stop into a hang.

    The job must remain exactly as stoppable as it was before the receipt existed.
    """
    receipt = tmp_path / "death.json"
    proc = _spawn(tmp_path, receipt)
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=10) != 0
    assert -proc.returncode == int(signal.SIGTERM)


def test_sigint_and_sighup_are_also_covered(tmp_path: Path) -> None:
    for sig in (signal.SIGINT, signal.SIGHUP):
        receipt = tmp_path / f"death_{sig}.json"
        proc = _spawn(tmp_path, receipt)
        proc.send_signal(sig)
        proc.wait(timeout=10)
        assert _wait_for(receipt)["signal"] == int(sig)


def test_a_failing_progress_callback_does_not_cost_the_receipt(tmp_path: Path) -> None:
    """The receipt is the point; progress is a bonus. Losing both would be the bug."""
    receipt = tmp_path / "death.json"
    src = CHILD.format(repo=str(REPO), receipt=str(receipt)).replace(
        'progress=lambda: {"cells": 38, "of": 39}',
        'progress=lambda: (_ for _ in ()).throw(RuntimeError("boom"))',
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", src], stdout=subprocess.PIPE, text=True, cwd=str(tmp_path)
    )
    assert proc.stdout is not None
    assert proc.stdout.readline().strip() == "ready"
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=10)
    record = _wait_for(receipt)
    assert record["signal_name"] == "SIGTERM"
    assert "RuntimeError" in str(record.get("progress_error"))


def test_sigkill_leaves_no_receipt_and_that_absence_is_the_signal(tmp_path: Path) -> None:
    """SIGKILL cannot be caught -- by design.

    Pinned so nobody later "fixes" the gap: a job that vanishes with NO receipt is evidence of
    SIGKILL, the OOM killer, or a kernel event, and that is only informative because a catchable
    signal reliably DOES leave one.
    """
    receipt = tmp_path / "death.json"
    proc = _spawn(tmp_path, receipt)
    os.kill(proc.pid, signal.SIGKILL)
    proc.wait(timeout=10)
    time.sleep(0.3)
    assert not receipt.exists()
