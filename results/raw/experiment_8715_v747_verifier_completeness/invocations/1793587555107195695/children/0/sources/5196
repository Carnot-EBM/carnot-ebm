"""REQ-REPORT-8572: retain real framed children, responses and disjoint clocks."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import time
from typing import Any

from carnot.reporting import resident_worker_8572 as w
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.v709_execution import child, progress as progress

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
CLI = str(ROOT / "scripts/experiments/experiment_8572_v736_resident_inference_qualification.py")


class Session:
    """A private peer owns only its child, pipes and bounded lifecycle receipt."""

    def __init__(self, config: Json, raw: Path, arm: str, timeout: float = 15):
        raw.mkdir(parents=True, exist_ok=True)
        self.path, self.raw, self.arm = raw / "config.json", raw, arm
        atomic_json(self.path, config)
        self.digest = sha256_file(self.path)
        self.log = (raw / "stderr").open("wb")
        self.argv = [
            sys.executable,
            "-u",
            CLI,
            "--worker",
            str(self.path),
            "--config-hash",
            self.digest,
            "--arm",
            arm,
            "--timeout",
            str(timeout),
        ]
        self.start = time.monotonic_ns()
        progress("8572_before_session_" + arm)
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
        self.ready = json.loads(w.read_frame(self.output, 30) or b"null")
        self.ready_ns = time.monotonic_ns()
        if (
            not isinstance(self.ready, dict)
            or self.ready.get("status") != "ready"
            or self.ready.get("config_hash") != self.digest
        ):
            self.process.kill()
            self.process.wait(timeout=10)
            self.stdin.close()
            self.stdout.close()
            self.log.close()
            raise ValueError("session_ready")
        self.observations: list[Json] = []

    def request(self, row: Json, identifier: str) -> Json:
        """Record parent transfer endpoints around the actual child response."""
        request = w.envelope(identifier, self.digest, row)
        start = time.monotonic_ns()
        w.write_frame(self.input, request)
        response: Json = json.loads(w.read_frame(self.output, 30) or b"null")
        end = time.monotonic_ns()
        record = dict(request=request, response=response, start_ns=start, end_ns=end)
        self.observations.append(record)
        return response

    def close(self, *, shutdown: bool = True) -> Json:
        """Normal peers acknowledge shutdown; abnormal peers retain their genuine exit."""
        response = None
        if shutdown:
            request = dict(w.envelope("shutdown", self.digest, {}), operation="shutdown")
            w.write_frame(self.input, request)
            response = json.loads(w.read_frame(self.output, 10) or b"null")
        self.stdin.close()
        self.process.wait(timeout=10)
        end = time.monotonic_ns()
        self.stdout.close()
        self.log.close()
        progress("8572_after_session_" + self.arm, len(self.observations), 0)
        return dict(
            argv=self.argv,
            pid=self.process.pid,
            exit_code=self.process.returncode,
            start_ns=self.start,
            ready_ns=self.ready_ns,
            end_ns=end,
            duration_s=(end - self.start) / 1e9,
            ready=self.ready,
            shutdown=response,
            stderr_path=str(self.raw / "stderr"),
            stderr_sha256=sha256_file(self.raw / "stderr"),
            config_path=str(self.path),
            config_hash=self.digest,
            observations=self.observations,
        )


def roster(bundle: Json) -> list[Json]:
    """Predetermined groups cover order, duplicate text, restart and configuration changes."""
    groups = []
    for arm in ["python", "native"]:
        for kind in ["panel", "resident"]:
            for name in ["answers", "controls"]:
                groups.append(
                    dict(
                        name=f"{arm}-{kind}-{name}",
                        arm=arm,
                        kind=kind,
                        rows=bundle[name],
                        head=bundle["head"],
                        category=name,
                    )
                )
        duplicate = [
            dict(bundle["answers"][i], unit_id=f"duplicate-{j}") for j, i in enumerate([1, 0, 0])
        ]
        for name in ["reordered_duplicate", "restart", "changed_configuration"]:
            head = deepcopy(bundle["head"])
            if name == "changed_configuration":
                head["temperature"] *= 2
            groups.append(
                dict(
                    name=f"{arm}-{name}",
                    arm=arm,
                    kind="resident",
                    rows=duplicate,
                    head=head,
                    category="isolation",
                )
            )
        zero = dict(bundle["head"], parameters=[0.0] * 109, decay_scale=1.0, temperature=1.0)
        import numpy as np

        thresholds = [
            dict(bundle["answers"][0], q=float(q), unit_id=f"threshold-{i}-{j}")
            for i, p in enumerate([0.2, 0.8])
            for j, q in enumerate([np.nextafter(p, 0), p, np.nextafter(p, 1)])
        ]
        groups.append(
            dict(
                name=f"{arm}-threshold",
                arm=arm,
                kind="resident",
                rows=thresholds,
                head=zero,
                category="isolation",
            )
        )
    return groups


def partition(receipt: Json, kind: str) -> Json:
    """Attribute disjoint measured intervals; retain every remaining nanosecond explicitly."""
    start, end = receipt["start_ns"], receipt["end_ns"]
    pieces = [
        dict(operation="startup_import_configuration", start_ns=start, end_ns=receipt["ready_ns"])
    ]
    for row in receipt["observations"]:
        spans = row["response"]["spans"]
        if kind == "resident":
            pieces.append(
                dict(
                    operation="request_transfer_parse",
                    start_ns=row["start_ns"],
                    end_ns=spans[0]["start_ns"],
                )
            )
        pieces.extend(spans)
        if kind == "resident":
            pieces.append(
                dict(
                    operation="serialization_response_transfer",
                    start_ns=spans[-1]["end_ns"],
                    end_ns=row["end_ns"],
                )
            )
    last = pieces[-1]["end_ns"]
    pieces.append(dict(operation="shutdown", start_ns=last, end_ns=end))
    pieces.sort(key=lambda p: p["start_ns"])
    accounted = sum(p["end_ns"] - p["start_ns"] for p in pieces)
    passed = all(start <= p["start_ns"] <= p["end_ns"] <= end for p in pieces)
    passed = passed and all(a["end_ns"] <= b["start_ns"] for a, b in zip(pieces, pieces[1:]))
    return dict(
        passed=passed,
        parent_launch_through_exit_ns=end - start,
        intervals=pieces,
        attributed_ns=accounted,
        unattributed_overhead_ns=end - start - accounted,
        timing_benefit_inferred=False,
    )


def failure_controls(bundle: Json, raw: Path) -> list[Json]:
    """Real abnormal exits are intentional failure fixtures, never successful answers."""
    result = []
    kinds = [
        "malformed",
        "oversize",
        "eof_mid_frame",
        "crash",
        "deadline",
        "full_queue",
        "duplicate_id",
        "shutdown",
        "configuration_change",
        "idle_deadline",
    ]
    for kind in kinds:
        session = Session(
            dict(head=bundle["head"], extension=bundle["extension"]),
            raw / kind,
            "native",
            timeout=0.2 if kind == "idle_deadline" else 15,
        )
        response: Json = dict(status="censored", reason=kind)
        recoverable = kind in {"malformed", "deadline", "duplicate_id"}
        if kind == "malformed":
            os.write(session.input, b"\x00\x00\x00\x01{")
        elif kind == "oversize":
            os.write(session.input, struct.pack("!I", 65537))
        elif kind == "eof_mid_frame":
            os.write(session.input, b"\x00\x00\x00\x20{")
            session.stdin.close()
        elif kind == "crash":
            os.kill(session.process.pid, signal.SIGTERM)
        elif kind == "deadline":
            request = dict(
                w.envelope("expired", session.digest, bundle["answers"][0]), deadline_ns=0
            )
            w.write_frame(session.input, request)
        elif kind == "full_queue":
            os.kill(session.process.pid, signal.SIGSTOP)
            os.set_blocking(session.input, False)
            try:
                while True:
                    os.write(session.input, b"x" * 4096)
            except BlockingIOError:
                pass
            try:
                w.write_frame(session.input, dict(request_id="queue"), timeout=0.02)
            except TimeoutError:
                response["write_deadline_observed"] = True
            finally:
                os.kill(session.process.pid, signal.SIGCONT)
                os.kill(session.process.pid, signal.SIGTERM)
        elif kind == "duplicate_id":
            session.request(bundle["answers"][0], "duplicate")
            w.write_frame(
                session.input, w.envelope("duplicate", session.digest, bundle["answers"][1])
            )
        elif kind == "configuration_change":
            atomic_json(session.path, dict(head={}))
            w.write_frame(
                session.input, w.envelope("changed", session.digest, bundle["answers"][0])
            )
        if recoverable:
            response = json.loads(w.read_frame(session.output, 10) or b"null")
            recovery = session.request(bundle["answers"][1], "recovery")
            passed = response.get("status") == "error" and "response" not in response
            passed = (
                passed and recovery.get("status") == "ok" and recovery["request_id"] == "recovery"
            )
            expected, _ = w.evaluate(bundle["head"], bundle["answers"][1], None)
            actual = recovery.get("response", {})
            passed = passed and actual.get("unit_id") == expected["unit_id"]
            passed = passed and actual.get("decision") == expected["decision"]
            passed = passed and abs(
                actual.get("probability", -1) - expected["probability"]
            ) <= 1e-12 * (1 + abs(expected["probability"]))
        else:
            passed = True
        if kind == "shutdown" or recoverable:
            receipt = session.close()
            passed = (
                passed and receipt["exit_code"] == 0 and receipt["shutdown"]["status"] == "shutdown"
            )
            if kind == "shutdown":
                response = receipt["shutdown"]
        else:
            session.process.wait(timeout=10)
            tail = w.read_frame(session.output, 1)
            receipt = session.close(shutdown=False)
            passed = passed and tail is None and receipt["exit_code"] != 0
            if kind == "full_queue":
                passed = passed and response.get("write_deadline_observed") is True
        result.append(
            dict(
                unit_id="failure-" + kind,
                kind=kind,
                passed=passed,
                response=response,
                lifecycle=receipt,
                missing_reason=None if passed else "failure_control_failed",
            )
        )
    return result


def execute(bundle: Json, raw: Path) -> Json:
    """Measure all four paths on fixed operands without comparing their speed."""
    sessions = []
    for index, group in enumerate(roster(bundle)):
        progress("8572_group_" + group["name"], index, 16 - index)
        directory = raw / group["name"]
        config = dict(head=group["head"], extension=bundle["extension"], rows=group["rows"])
        if group["kind"] == "resident":
            session = Session(config, directory, group["arm"])
            try:
                for i, row in enumerate(group["rows"]):
                    session.request(row, f"request-{i}")
                    if (i + 1) % 100 == 0:
                        progress("8572_requests", i + 1, len(group["rows"]) - i - 1)
                receipt = session.close()
            finally:
                if session.process.poll() is None:
                    session.process.kill()
                    session.process.wait(timeout=10)
        else:
            directory.mkdir(parents=True, exist_ok=True)
            path, output = directory / "config.json", directory / "panel.json"
            atomic_json(path, config)
            child_receipt = child(
                group["name"],
                [
                    sys.executable,
                    "-u",
                    CLI,
                    "--panel",
                    str(path),
                    "--config-hash",
                    sha256_file(path),
                    "--arm",
                    group["arm"],
                    "--worker-output",
                    str(output),
                ],
                directory,
                deadline=180,
            )
            if not child_receipt["passed"]:
                raise ValueError("panel_child_failed")
            data = json.loads(output.read_bytes())
            receipt = dict(
                child_receipt,
                start_ns=child_receipt["started_monotonic_ns"],
                end_ns=child_receipt["ended_monotonic_ns"],
                ready_ns=data["ready_ns"],
                ready=dict(loaded_extension=data["loaded_extension"], config_loads=1),
                observations=[
                    dict(response=dict(status="ok", response=r["response"], spans=r["spans"]))
                    for r in data["rows"]
                ],
                config_path=str(path),
                config_hash=sha256_file(path),
            )
        receipt.update(
            name=group["name"], category=group["category"], kind=group["kind"], arm=group["arm"]
        )
        sessions.append(receipt)
    from carnot.reporting.resident_qualification_8572 import summarize

    primitive = dict(
        sessions=sessions, failure_control_rows=failure_controls(bundle, raw / "failures")
    )
    primitive.update(summarize(primitive, bundle))
    atomic_json(raw / "primitive.json", primitive)
    return primitive
