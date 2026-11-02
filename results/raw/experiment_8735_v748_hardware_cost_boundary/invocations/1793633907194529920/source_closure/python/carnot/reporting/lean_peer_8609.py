"""REQ-VERIFY-8609: supervise original workers without changing their inference code."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import time
from typing import Any

from carnot.reporting import lean_resident_8608 as prototype
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.resident_measure_8572 import partition as partition
from carnot.reporting.v709_execution import progress

Json = dict[str, Any]
prototype.energy_module()
import resident_protocol as wire


class Session:
    """Keep real child responses and account for launch through process exit."""

    def __init__(self, config: Json, raw: Path, label: str, timeout: float = 15):
        raw.mkdir(parents=True, exist_ok=True)
        self.raw, self.label = raw, label
        self.path = raw / "configuration.json"
        atomic_json(self.path, config)
        self.digest = prototype.reference(self.path)["sha256"]
        self.graph = raw / "modules.json"
        entry = (
            "from resident_protocol import serve"
            if label.startswith("lean")
            else f"sys.path[:0] = {[str(prototype.ROOT / 'python'), str(prototype.ROOT)]!r}\n"
            "from carnot.reporting.resident_worker_8572 import serve"
        )
        code = (
            "import sys,json,hashlib\nfrom pathlib import Path\n"
            + f"sys.path.insert(0,{str(prototype.ROOT / 'python/carnot/inference')!r})\n"
            + entry
            + "\ntry:\n"
            + f"    result=serve(Path({str(self.path)!r}),{self.digest!r},{label.split('_')[-1]!r},{timeout!r})\n"
            + "finally:\n    rows=[]\n    hashes={}\n"
            + "    for name,module in list(sys.modules.items()):\n"
            + "        path=Path(getattr(module,'__file__','') or '')\n"
            + "        if path.is_file():\n"
            + "            path=path.absolute()\n"
            + "            if str(path) not in hashes:\n"
            + "                hashes[str(path)]='sha256:'+hashlib.sha256(path.read_bytes()).hexdigest()\n"
            + "            rows.append(dict(name=name,path=str(path),sha256=hashes[str(path)]))\n"
            + f"    Path({str(self.graph)!r}).write_text(json.dumps(rows,sort_keys=True))\n"
            + "raise SystemExit(result)\n"
        )
        self.wrapper = raw / "private_child.py"
        self.wrapper.write_text(code)
        self.argv = [sys.executable, "-u", str(self.wrapper)]
        self.log = (raw / "stderr").open("wb")
        self.start = time.monotonic_ns()
        progress("8609_before_child_" + label)
        self.process = subprocess.Popen(
            self.argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            env=dict(os.environ, LLVM_PROFILE_FILE=str(raw / "native-%p-%m.profraw")),
            start_new_session=True,
        )
        assert self.process.stdin is not None and self.process.stdout is not None
        self.stdin, self.stdout = self.process.stdin, self.process.stdout
        self.input, self.output = self.stdin.fileno(), self.stdout.fileno()
        self.ready: Json = {}
        self.ready_ns = time.monotonic_ns()
        self.observations: list[Json] = []
        try:
            self.ready = self.receive()
            if not isinstance(self.ready, dict) or self.ready.get("status") != "ready":
                raise ValueError("worker_ready")
        except (ValueError, TypeError, TimeoutError):
            self.finish(False)
            raise
        self.ready_ns = time.monotonic_ns()

    def receive(self) -> Json:
        """A finite wait prevents a stalled worker from hiding in parent reads."""
        return json.loads(wire.read_frame(self.output, 30) or b"null")  # type: ignore[no-any-return]

    def send(self, value: Json) -> Json:
        """Retain both transfer endpoints so complete operation spans can be checked."""
        began = time.monotonic_ns()
        wire.write_frame(self.input, value)
        response = self.receive()
        self.observations.append(
            dict(request=value, response=response, start_ns=began, end_ns=time.monotonic_ns())
        )
        return response

    def request(self, row: Json, identifier: str) -> Json:
        """Every request gets a fresh envelope and performs real numerical work."""
        return self.send(wire.envelope(identifier, self.digest, row))

    def finish(self, shutdown: bool = True) -> Json:
        """Reap only this owned process; forced cleanup always fails normal qualification."""
        response = None
        if shutdown:
            wire.write_frame(
                self.input, dict(wire.envelope("stop", self.digest, {}), operation="shutdown")
            )
            response = self.receive()
        self.stdin.close()
        forced = False
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)
            forced = True
        self.stdout.close()
        self.log.close()
        result = dict(
            arm=self.label,
            command=self.argv,
            pid=self.process.pid,
            start_ns=self.start,
            ready_ns=self.ready_ns,
            end_ns=time.monotonic_ns(),
            exit_code=self.process.returncode,
            forced_cleanup=forced,
            child_reaped=self.process.poll() is not None,
            ready=self.ready,
            shutdown=response,
            config_reference=prototype.reference(self.path),
            wrapper_reference=prototype.reference(self.wrapper),
            stderr_reference=prototype.reference(self.raw / "stderr"),
            observations=self.observations,
            modules_reference=prototype.reference(self.graph) if self.graph.exists() else None,
        )
        progress("8609_after_child_" + self.label, len(self.observations), 0)
        return result


def lifecycle(config: Json, raw: Path, label: str, kind: str) -> Json:
    """Use pipe and process controls to test failures without replacing inference."""
    peer = Session(config, raw, label, timeout=0.2 if kind in {"idle", "timeout"} else 15)
    row = config["rows"][0]
    response: Json | None = None
    recovery: Json | None = None
    passed = True
    if kind in {"expired", "duplicate", "malformed", "nonfinite", "empty", "long_text"}:
        first = peer.request(row, "first")
        value = wire.envelope("bad", peer.digest, row)
        if kind == "expired":
            value["deadline_ns"] = 0
        elif kind == "duplicate":
            value["request_id"] = "first"
        elif kind == "empty":
            value["row"] = dict(rows=[])
        elif kind == "long_text":
            value["row"] = dict(
                row,
                public=dict(
                    family_id="over-budget",
                    source_bytes=(b"There are 12 apples. " * 600).hex(),
                    answer_bytes=b"There are 12 apples.".hex(),
                ),
            )
        if kind == "malformed":
            os.write(peer.input, b"\x00\x00\x00\x01{")
        elif kind == "nonfinite":
            value["row"] = dict(row, q=float("nan"))
            payload = json.dumps(value).encode()
            os.write(peer.input, struct.pack("!I", len(payload)) + payload)
        else:
            wire.write_frame(peer.input, value)
        response = peer.receive()
        recovery = peer.request(config["rows"][1], "recovery")
        passed = (
            first["status"] == "ok"
            and response["status"] == "error"
            and "response" not in response
            and recovery["status"] == "ok"
            and recovery["request_id"] == "recovery"
            and recovery["response"]["unit_id"] == config["rows"][1]["unit_id"]
        )
        receipt = peer.finish()
        passed = passed and receipt["exit_code"] == 0
    elif kind == "backpressure":
        os.kill(peer.process.pid, signal.SIGSTOP)
        os.set_blocking(peer.input, False)
        sent = 0
        try:
            while True:
                payload = json.dumps(wire.envelope(f"queue-{sent}", peer.digest, row)).encode()
                frame = struct.pack("!I", len(payload)) + payload
                assert len(frame) <= 4096
                os.write(peer.input, frame)
                sent += 1
        except BlockingIOError:
            pass
        try:
            wire.write_frame(peer.input, wire.envelope("not-sent", peer.digest, row), timeout=0.02)
            passed = False
        except TimeoutError:
            response = dict(status="write_deadline", accepted_requests=sent)
        os.kill(peer.process.pid, signal.SIGCONT)
        drained = [peer.receive() for _ in range(sent)]
        recovery = peer.request(config["rows"][1], "recovery")
        passed = passed and [r["request_id"] for r in drained] == [
            f"queue-{i}" for i in range(sent)
        ]
        passed = passed and all(r["status"] == "ok" for r in drained) and recovery["status"] == "ok"
        receipt = peer.finish()
        response = dict(response or {}, drained_responses=drained)
    else:
        if kind == "partial":
            os.write(peer.input, b"\x00\x00\x00\x20{")
            peer.stdin.close()
        elif kind == "timeout":
            os.write(peer.input, b"\x00")
        elif kind == "abnormal":
            os.kill(peer.process.pid, signal.SIGTERM)
        elif kind == "stale":
            atomic_json(peer.path, dict(head={}))
            wire.write_frame(peer.input, wire.envelope("changed", peer.digest, row))
        peer.process.wait(timeout=30)
        tail = wire.read_frame(peer.output, 1)
        receipt = peer.finish(False)
        passed = tail is None and receipt["exit_code"] != 0
    passed = passed and receipt["child_reaped"] and not receipt["forced_cleanup"]
    return dict(
        unit_id=label + ":" + kind,
        arm=label,
        kind=kind,
        passed=passed,
        response=response,
        recovery=recovery,
        lifecycle=receipt,
        missing_reason=None if passed else "isolation_or_cleanup_failure",
    )


def execute(config: Json, raw: Path, label: str, *, deadline_ns: int = 2**63) -> Json:
    """Complete text and numerical responses retain their real operation intervals."""
    peer = Session(config, raw, label)
    try:
        for i, row in enumerate(config["rows"]):
            if time.monotonic_ns() + 35_000_000_000 >= deadline_ns:
                raise TimeoutError("measurement_deadline_reserved_shutdown")
            peer.request(row, f"request-{i}")
            if (i + 1) % 64 == 0:
                progress("8609_parity_" + label, i + 1, len(config["rows"]) - i - 1)
    except (OSError, ValueError, KeyError, TypeError, TimeoutError):
        atomic_json(raw / "failed_child.json", peer.finish(False))
        raise
    receipt = peer.finish()
    receipt["partition"] = partition(receipt, "resident")
    return receipt
