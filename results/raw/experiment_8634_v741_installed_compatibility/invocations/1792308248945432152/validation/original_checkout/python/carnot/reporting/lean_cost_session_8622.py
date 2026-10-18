"""REQ-VERIFY-8622: charge every interval from launch to clean process exit."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from carnot.reporting import lean_cost_8622 as e
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.resident_worker_8572 import envelope, read_frame, write_frame
from carnot.reporting.v709_execution import progress

Json = dict[str, Any]


def session(bundle: Json, slot: Json, raw: Path, *, deadline_ns: int = 2**63) -> Json:
    """The parent owns one bounded child and preserves raw frame endpoints for replay."""
    if time.monotonic_ns() + 40_000_000_000 >= deadline_ns:
        raise TimeoutError("measurement_deadline_before_launch")
    raw.mkdir(parents=True, exist_ok=True)
    config = raw / "config.json"
    atomic_json(config, dict(head=bundle["head"], extension=bundle["extension"]))
    digest = e.reference(config)["sha256"]
    argv = [sys.executable, "-u", str(e.ROOT / e.WORKER), str(config), digest, slot["arm"]]
    observations: list[Json] = []
    requests: list[Json] = []
    operations: list[Json] = []
    load = os.getloadavg()
    progress("8622_before_child_" + slot["arm"])
    with (raw / "stderr").open("wb") as log:
        start = time.monotonic_ns()
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            start_new_session=True,
            env=dict(
                os.environ,
                OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                NUMEXPR_NUM_THREADS="1",
                LLVM_PROFILE_FILE=str(raw / "native-%p-%m.profraw"),
            ),
        )
        assert proc.stdin is not None and proc.stdout is not None
        inp, out = proc.stdin.fileno(), proc.stdout.fileno()
        try:
            ready = json.loads(read_frame(out, 30) or b"null")
            ready_ns = time.monotonic_ns()
            expected = (
                {k: bundle["extension"][k] for k in ["path", "sha256"]}
                if slot["arm"].endswith("native")
                else None
            )
            if (
                ready["status"] != "ready"
                or ready["config_hash"] != digest
                or ready["config_loads"] != 1
                or ready["loaded_extension"] != expected
            ):
                raise ValueError("worker_ready_or_extension")
            operations.extend(
                [
                    dict(operation="startup", start_ns=start, end_ns=ready["imports_start_ns"]),
                    dict(
                        operation="imports",
                        start_ns=ready["imports_start_ns"],
                        end_ns=ready["imports_end_ns"],
                    ),
                    dict(operation="startup", start_ns=ready["imports_end_ns"], end_ns=ready_ns),
                ]
            )
            for panel in range(slot["length"]):
                for index in range(0, 64, slot["batch"]):
                    if time.monotonic_ns() + 35_000_000_000 >= deadline_ns:
                        raise TimeoutError("measurement_deadline_reserved_shutdown")
                    rows = bundle["requests"][index : index + slot["batch"]]
                    value = envelope(f"{panel}-{index}", digest, dict(rows=rows))
                    sent = time.monotonic_ns()
                    write_frame(inp, value)
                    response = json.loads(read_frame(out, 30) or b"null")
                    end = time.monotonic_ns()
                    if response["status"] != "ok" or response["request_id"] != value["request_id"]:
                        raise ValueError("worker_response")
                    spans = response["spans"]
                    operations.extend(
                        [
                            dict(
                                operation="parent_residual",
                                start_ns=operations[-1]["end_ns"],
                                end_ns=sent,
                            ),
                            dict(
                                operation="frame_transport",
                                start_ns=sent,
                                end_ns=spans[0]["start_ns"],
                            ),
                            *spans,
                            dict(
                                operation="frame_transport",
                                start_ns=spans[-1]["end_ns"],
                                end_ns=end,
                            ),
                        ]
                    )
                    observations.append(
                        dict(request=value, response=response, start_ns=sent, end_ns=end)
                    )
                    for operand, answer in zip(rows, response["responses"], strict=True):
                        requests.append(
                            dict(
                                unit_id=operand["unit_id"],
                                panel=panel,
                                request_index=index,
                                response=json.loads(answer["response"]),
                                latency_s=(end - (sent if observations[:-1] else start)) / 1e9,
                            )
                        )
                progress("8622_session_panels", panel + 1, slot["length"] - panel - 1)
            write_frame(inp, dict(envelope("stop", digest, {}), operation="shutdown"))
            shutdown = json.loads(read_frame(out, 10) or b"null")
            proc.stdin.close()
            proc.wait(timeout=10)
            end = time.monotonic_ns()
            if proc.returncode != 0 or shutdown["status"] != "shutdown":
                raise ValueError("unclean_shutdown")
            operations.append(
                dict(operation="shutdown", start_ns=operations[-1]["end_ns"], end_ns=end)
            )
        except (OSError, ValueError, KeyError, TypeError, TimeoutError, subprocess.TimeoutExpired):
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=10)
            atomic_json(
                raw / "failed_child.json",
                dict(
                    command=argv,
                    exit_code=proc.returncode,
                    observations=observations,
                    status="failed",
                ),
            )
            raise
        finally:
            proc.stdin.close()
            proc.stdout.close()
    progress("8622_after_child_" + slot["arm"], len(requests), 0)
    return dict(
        slot=slot,
        start_ns=start,
        end_ns=end,
        duration_s=(end - start) / 1e9,
        startup_s=(ready_ns - start) / 1e9,
        exit_code=proc.returncode,
        ready=ready,
        ready_ns=ready_ns,
        shutdown=shutdown,
        peak_rss_kib=shutdown["peak_rss_kib"],
        growth_rss_kib=shutdown["peak_rss_kib"] - ready["ready_rss_kib"],
        operations=operations,
        requests=requests,
        observations=observations,
        command=argv,
        config_reference=e.reference(config),
        stderr_reference=e.reference(raw / "stderr"),
        host_load_before=load,
        host_load_after=os.getloadavg(),
    )
