"""REQ-VERIFY-8661: parent-owned pipe children retain complete lifecycle evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any

from carnot.inference import resident_protocol as wire
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.installed_cost_8661 import reference
from carnot.reporting.v709_execution import progress

Json = dict[str, Any]


def session(
    command: list[str],
    config: Path,
    slot: Json,
    requests: list[Json],
    cwd: Path,
    raw: Path,
    *,
    deadline_ns: int = 2**63,
    heartbeat_s: float = 25,
) -> Json:
    """Each fresh child performs every request before a clean bounded shutdown.

    Transport endpoints surround child numerical spans on the same host clock.
    A failure retains all observed frames and kills only this process group.
    """
    if time.monotonic_ns() + 40_000_000_000 >= deadline_ns:
        raise TimeoutError("measurement_deadline_before_launch")
    raw.mkdir(parents=True, exist_ok=True)
    digest = reference(config)["sha256"]
    arm = slot["arm"].split("_")[-1]
    argv = [*command, str(config), digest, arm]
    env = dict(
        os.environ,
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        LLVM_PROFILE_FILE=str(raw / "native-%p.profraw"),
    )
    env.pop("PYTHONPATH", None)
    observations: list[Json] = []
    progress("8661_before_benchmark_" + slot["unit_id"])
    start = time.monotonic_ns()
    load = os.getloadavg()
    with (raw / "stderr").open("wb") as log:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            cwd=cwd,
            env=env,
            start_new_session=True,
        )
        assert proc.stdin is not None and proc.stdout is not None
        inp, out = proc.stdin.fileno(), proc.stdout.fileno()
        try:
            ready = json.loads(wire.read_frame(out, 30) or b"null")
            ready_ns = time.monotonic_ns()
            if not isinstance(ready, dict) or ready.get("status") != "ready":
                raise ValueError("worker_ready")
            progress("8661_after_worker_ready_" + slot["unit_id"], 0, 64 * slot["length"])
            last_heartbeat_ns = ready_ns
            for panel in range(slot["length"]):
                for index, row in enumerate(requests):
                    if time.monotonic_ns() + 35_000_000_000 >= deadline_ns:
                        raise TimeoutError("measurement_deadline_reserved_shutdown")
                    frame = wire.envelope(f"{panel}-{index}", digest, row)
                    sent = time.monotonic_ns()
                    wire.write_frame(inp, frame)
                    response = json.loads(wire.read_frame(out, 30) or b"null")
                    end = time.monotonic_ns()
                    observations.append(
                        dict(request=frame, response=response, start_ns=sent, end_ns=end)
                    )
                    if end - last_heartbeat_ns >= heartbeat_s * 1e9:
                        progress(
                            "8661_session_heartbeat",
                            len(observations),
                            64 * slot["length"] - len(observations),
                        )
                        last_heartbeat_ns = end
                    if response.get("status") != "ok":
                        raise ValueError("worker_response")
                progress("8661_panels_" + slot["unit_id"], panel + 1, slot["length"] - panel - 1)
            wire.write_frame(inp, dict(wire.envelope("shutdown", digest, {}), operation="shutdown"))
            shutdown = json.loads(wire.read_frame(out, 10) or b"null")
            proc.stdin.close()
            proc.wait(timeout=10)
            end = time.monotonic_ns()
            if proc.returncode != 0 or shutdown.get("status") != "shutdown":
                raise ValueError("unclean_shutdown")
        except (OSError, ValueError, KeyError, TypeError, TimeoutError, subprocess.TimeoutExpired):
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)
            atomic_json(
                raw / "failed_child.json",
                dict(
                    command=argv,
                    exit_code=proc.returncode,
                    observations=observations,
                    start_ns=start,
                    end_ns=time.monotonic_ns(),
                ),
            )
            raise
        finally:
            proc.stdin.close()
            proc.stdout.close()
    progress("8661_after_benchmark_" + slot["unit_id"], len(observations), 0)
    return dict(
        slot=slot,
        command=argv,
        cwd=str(cwd),
        pythonpath_unset=True,
        pid=proc.pid,
        start_ns=start,
        end_ns=end,
        ready_ns=ready_ns,
        ready=ready,
        shutdown=shutdown,
        exit_code=proc.returncode,
        observations=observations,
        config_reference=reference(config),
        stderr_reference=reference(raw / "stderr"),
        host_load_before=load,
        host_load_after=os.getloadavg(),
    )
