"""REQ-VERIFY-8716: keep optional legacy imports in a reaped process.

This helper uses only the standard library so installed callers do not need the
research checkout or accidentally import the dependencies they are checking.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any


def legacy_contract(test: Path) -> dict[str, Any]:
    """Execute every original assertion, then reap the child before returning its RSS."""
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    started = time.monotonic_ns()
    code = (
        "import os,json,resource,runpy,sys;"
        f"runpy.run_path({str(test)!r})['_legacy_contract_assertions']();"
        "print(json.dumps(dict(child_pid=os.getpid(),"
        "child_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"
        "child_import_roots=sorted({n.split('.')[0] for n in sys.modules}),"
        "assertions_passed=True)))"
    )
    print("[exp8716] before_legacy_export_child", flush=True)
    with TemporaryDirectory(prefix="legacy8716-", dir="/var/tmp") as directory:
        result = subprocess.run(
            [sys.executable, "-u", "-c", code],
            cwd=directory,
            env=dict(os.environ, PYTHONUNBUFFERED="1"),
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
    row: dict[str, Any] = json.loads(result.stdout.splitlines()[-1])
    row.update(
        parent_pid=os.getpid(),
        actual_exit=result.returncode,
        reaped=not Path(f"/proc/{row['child_pid']}").exists(),
        parent_peak_before_kib=before,
        parent_peak_after_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        started_monotonic_ns=started,
        ended_monotonic_ns=time.monotonic_ns(),
        test_reference=dict(
            path=str(test.absolute()),
            sha256="sha256:" + hashlib.sha256(test.read_bytes()).hexdigest(),
        ),
    )
    row["passed"] = bool(
        row["assertions_passed"]
        and row["reaped"]
        and row["actual_exit"] == 0
        and row["parent_peak_after_kib"] - before <= 500 * 1024
    )
    destination = os.environ.get("CARNOT_8716_LEGACY_RECEIPTS")
    if destination:
        path = Path(destination) / f"{os.getpid()}-{started}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(row, sort_keys=True) + "\n")
    print("[exp8716] after_legacy_export_child completed=1 pending=0", flush=True)
    return row
