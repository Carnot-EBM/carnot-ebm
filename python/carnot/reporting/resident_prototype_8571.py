"""REQ-REPORT-8571: eight private requests exercise real bounded resident children."""

from __future__ import annotations

from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from carnot.reporting import resident_inference_8571 as e
from carnot.reporting import resident_worker_8571 as w
from carnot.reporting.current_work_receipt import atomic_json
from carnot.verify import nine_input_8533 as numeric
from carnot.verify.native_nine_input_8533 import load

Json = dict[str, Any]


def execute(bundle: Json, raw: Path) -> Json:
    """Four sessions show restart independence; these clocks make no speed comparison."""
    raw.mkdir(parents=True, exist_ok=True)
    config_path = raw / "immutable_config.json"
    atomic_json(config_path, dict(head=bundle["head"], extension=bundle["extension"]))
    digest = e.reference(config_path)["sha256"]
    rows: list[Json] = []
    lifecycle: list[Json] = []
    for session, arm in enumerate(["python", "native", "python", "native"]):
        print(f"[exp8571] phase=before_session session={session} arm={arm}", flush=True)
        start = time.monotonic_ns()
        log = raw / f"session-{session}.stderr"
        argv = [
            sys.executable,
            "-u",
            str(e.ROOT / e.CLI),
            "--worker",
            str(config_path),
            "--config-hash",
            digest,
            "--arm",
            arm,
        ]
        env = dict(os.environ, LLVM_PROFILE_FILE=str(raw / "native-%p-%m.profraw"))
        with log.open("wb") as err:
            process = subprocess.Popen(
                argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, env=env
            )
            assert process.stdin is not None and process.stdout is not None
            try:
                ready = json.loads(w.read_frame(process.stdout.fileno(), 30) or b"null")
                if (
                    not ready
                    or ready.get("status") != "ready"
                    or ready.get("config_hash") != digest
                ):
                    raise ValueError("session_ready")
                count = 3 if session < 2 else 1
                for slot in range(count):
                    identifier = f"session{session}-request{slot}"
                    row = bundle["requests"][0 if slot == 0 else 1]
                    request = w.envelope(identifier, digest, row)
                    kind = "valid"
                    if session == 0 and slot == 1:
                        os.write(process.stdin.fileno(), b"\x00\x00\x00\x01{")
                        kind = "malformed_json"
                    else:
                        if session == 1 and slot == 1:
                            request["config_hash"] = "changed_model_hash"
                            kind = "changed_config_hash"
                        w.write_frame(process.stdin.fileno(), request)
                    response = json.loads(w.read_frame(process.stdout.fileno(), 30) or b"null")
                    if kind == "valid":
                        reference = json.loads(numeric.request(bundle["head"], row))
                        actual = json.loads(response.get("response", "{}"))
                        passed = (
                            response.get("request_id") == identifier
                            and response.get("status") == "ok"
                            and actual.get("decision") == reference["decision"]
                            and abs(actual.get("probability", -1) - reference["probability"])
                            <= 1e-12
                        )
                    else:
                        passed = response.get("status") == "error" and "response" not in response
                    rows.append(
                        dict(
                            unit_id=identifier,
                            session=session,
                            arm=arm,
                            kind=kind,
                            request=request,
                            response=response,
                            passed=passed,
                            missing_reason=None,
                        )
                    )
                    print(
                        f"[exp8571] phase=private_request completed={len(rows)} pending={8 - len(rows)}",
                        flush=True,
                    )
                shutdown = w.envelope("shutdown", digest, {})
                shutdown["operation"] = "shutdown"
                w.write_frame(process.stdin.fileno(), shutdown)
                closed = json.loads(w.read_frame(process.stdout.fileno(), 5) or b"null")
                process.stdin.close()
                eof = w.read_frame(process.stdout.fileno(), 5) is None
                exit_code = process.wait(timeout=10)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=10)
                if not process.stdin.closed:
                    process.stdin.close()
                process.stdout.close()
            lifecycle.append(
                dict(
                    session=session,
                    arm=arm,
                    pid=process.pid,
                    argv=argv,
                    started_ns=start,
                    ended_ns=time.monotonic_ns(),
                    exit_code=exit_code,
                    ready=ready,
                    config_loads=ready["config_loads"],
                    clean_shutdown=closed["status"] == "shutdown" and eof,
                    stderr_reference=e.reference(log),
                    restart_of=session - 2 if session >= 2 else None,
                )
            )
        print(
            f"[exp8571] phase=after_session completed={session + 1} pending={3 - session}",
            flush=True,
        )
    result = dict(
        immutable_config_reference=e.reference(config_path),
        prototype_request_rows=rows,
        lifecycle_receipts=lifecycle,
    )
    atomic_json(raw / "prototype.json", result)
    return result


def replay(primitive: Json, bundle: Json) -> bool:
    """Bind original inputs and reconstruct each result so rehashed lies still fail."""
    try:
        config = e.read(primitive["immutable_config_reference"])
        if config != dict(head=bundle["head"], extension=bundle["extension"]):
            return False
        rows = primitive["prototype_request_rows"]
        sessions = primitive["lifecycle_receipts"]
        if len(rows) != 8 or len(sessions) != 4 or len({s["pid"] for s in sessions}) != 4:
            return False
        with redirect_stdout(sys.stderr):
            native = load(Path(bundle["extension"]["path"]))
        for index, row in enumerate(rows):
            session = [0, 0, 0, 1, 1, 1, 2, 3][index]
            slot = [0, 1, 2, 0, 1, 2, 0, 0][index]
            request, response = row["request"], row["response"]
            expected_row = bundle["requests"][0 if slot == 0 else 1]
            expected_id = f"session{session}-request{slot}"
            if (
                row["unit_id"] != expected_id
                or row["session"] != session
                or request["row"] != expected_row
            ):
                return False
            expected_kind = (
                "malformed_json" if index == 1 else "changed_config_hash" if index == 4 else "valid"
            )
            if (
                row["kind"] != expected_kind
                or row["arm"] != ["python", "native"][session % 2]
                or row["passed"] is not True
            ):
                return False
            if expected_kind == "valid":
                expected = numeric.request(
                    bundle["head"], expected_row, native if session % 2 else None
                )
                if response != dict(
                    request_id=expected_id,
                    status="ok",
                    config_hash=primitive["immutable_config_reference"]["sha256"],
                    response=expected,
                ):
                    return False
            elif response.get("status") != "error" or "response" in response:
                return False
        for index, receipt in enumerate(sessions):
            if (
                receipt["session"] != index
                or receipt["exit_code"] != 0
                or receipt["config_loads"] != 1
                or not receipt["clean_shutdown"]
            ):
                return False
            ref = receipt["stderr_reference"]
            if e.reference(Path(ref["path"])) != ref:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError, ImportError):
        return False
