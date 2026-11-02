"""REQ-VERIFY-8634: exercise real installed and private pipe rejection families."""

from __future__ import annotations

import json
import os
from pathlib import Path
import struct
import subprocess
import time
from typing import Any

from carnot.inference import resident_protocol as wire
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.v709_execution import progress

Json = dict[str, Any]
KINDS = [
    "expired",
    "duplicate",
    "unknown",
    "unknown_id",
    "malformed",
    "nonfinite",
    "empty",
    "long_text",
    "stale",
    "partial",
    "timeout",
    "idle",
    "abnormal",
    "backpressure",
    "oversized",
    "fragmented",
]


def checks(command: list[str], config: Json, raw: Path, cwd: Path, arm: str) -> list[Json]:
    """Run original controls with deadlines and retain every actual response and exit."""
    raw.mkdir(parents=True, exist_ok=True)
    rows = []
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    for index, kind in enumerate(KINDS):
        progress("8634_before_" + kind, index, len(KINDS) - index)
        path, log = raw / (kind + ".json"), raw / (kind + ".stderr")
        atomic_json(path, config)
        digest = sha256_file(path)
        argv = [
            *command,
            str(path),
            digest,
            arm,
            "--idle-timeout",
            "0.2" if kind in {"timeout", "idle"} else "15",
        ]
        start = time.monotonic_ns()
        observations: list[Json] = []
        with (
            log.open("wb") as stderr,
            subprocess.Popen(
                argv, cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr
            ) as process,
        ):
            assert process.stdin is not None and process.stdout is not None
            source, sink = process.stdout.fileno(), process.stdin.fileno()
            ready = json.loads(wire.read_frame(source, 20) or b"null")
            assert ready["status"] == "ready"
            value = wire.envelope("first", digest, config["rows"][0])
            wire.write_frame(sink, value)
            first = json.loads(wire.read_frame(source, 20) or b"null")
            assert first["status"] == "ok"
            value = wire.envelope("bad", digest, config["rows"][0])
            terminal = kind in {"stale", "partial", "timeout", "idle", "abnormal", "oversized"}
            forced = False
            try:
                if kind == "abnormal":
                    process.terminate()
                elif kind == "idle":
                    pass
                elif kind == "timeout":
                    os.write(sink, b"\x00")
                elif kind == "oversized":
                    os.write(sink, struct.pack("!I", wire.MAX_FRAME + 1))
                elif kind == "partial":
                    os.write(sink, b"\x00\x00\x00\x20{")
                    process.stdin.close()
                elif kind == "stale":
                    atomic_json(path, dict(head={}))
                    wire.write_frame(sink, value)
                elif kind == "backpressure":
                    import signal

                    os.kill(process.pid, signal.SIGSTOP)
                    os.set_blocking(sink, False)
                    queued = 0
                    try:
                        while True:
                            payload = json.dumps(
                                wire.envelope(f"queued-{queued}", digest, config["rows"][0])
                            ).encode()
                            frame = struct.pack("!I", len(payload)) + payload
                            assert len(frame) <= 4096
                            os.write(sink, frame)
                            queued += 1
                    except BlockingIOError:
                        pass
                    try:
                        wire.write_frame(sink, value, 0.01)
                    except TimeoutError as error:
                        observations.append(dict(error=str(error), passed=True))
                    finally:
                        os.kill(process.pid, signal.SIGCONT)
                    assert observations
                    for j in range(queued):
                        response = json.loads(wire.read_frame(source, 20) or b"null")
                        assert (
                            response["status"] == "ok" and response["request_id"] == f"queued-{j}"
                        )
                        observations.append(response)
                else:
                    if kind == "expired":
                        value["deadline_ns"] = 0
                    elif kind == "duplicate":
                        value["request_id"] = "first"
                    elif kind == "unknown":
                        value["operation"] = "unknown"
                    elif kind == "unknown_id":
                        value["request_id"] = None
                    elif kind == "nonfinite":
                        value["row"] = dict(config["rows"][0], q=float("nan"))
                    elif kind == "empty":
                        value["row"] = {}
                    elif kind == "long_text":
                        value["row"] = dict(
                            config["rows"][0],
                            public=dict(
                                family_id="over-budget",
                                source_bytes=(b"There are 12 apples. " * 600).hex(),
                                answer_bytes=b"There are 12 apples.".hex(),
                            ),
                        )
                    payload = b"{" if kind == "malformed" else json.dumps(value).encode()
                    frame = struct.pack("!I", len(payload)) + payload
                    if kind == "fragmented":
                        for byte in frame:
                            os.write(sink, bytes([byte]))
                    else:
                        os.write(sink, frame)
                    response = json.loads(wire.read_frame(source, 20) or b"null")
                    assert response["status"] == ("ok" if kind == "fragmented" else "error")
                    assert kind == "fragmented" or "response" not in response
                    observations.append(response)
                if not terminal:
                    wire.write_frame(sink, wire.envelope("recovery", digest, config["rows"][1]))
                    recovery = json.loads(wire.read_frame(source, 20) or b"null")
                    assert recovery["request_id"] == "recovery" and recovery["status"] == "ok"
                    observations.append(recovery)
                    wire.write_frame(
                        sink, dict(wire.envelope("stop", digest, {}), operation="shutdown")
                    )
                    assert (
                        json.loads(wire.read_frame(source, 20) or b"null")["status"] == "shutdown"
                    )
                process.wait(timeout=20)
                assert wire.read_frame(source, 1) is None
                assert process.returncode != 0 if terminal else process.returncode == 0
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=10)
                    forced = True
            assert not forced
        rows.append(
            dict(
                unit_id=arm + ":" + kind,
                kind=kind,
                arm=arm,
                passed=True,
                command=argv,
                cwd=str(cwd),
                pythonpath_unset=True,
                observations=observations,
                ready=ready,
                exit_code=process.returncode,
                intentional_death=kind == "abnormal",
                child_reaped=True,
                forced_cleanup=forced,
                duration_s=(time.monotonic_ns() - start) / 1e9,
                log=dict(path=str(log), sha256=sha256_file(log)),
            )
        )
        progress("8634_after_" + kind, index + 1, len(KINDS) - index - 1)
    return rows
