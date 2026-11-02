"""REQ-REPORT-8717: actual installed workers process identical complete text panels."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import sysconfig
import time
from typing import Any

import numpy as np

from carnot.inference import resident_protocol as wire
from carnot.reporting import installed_batch_cost_8717 as e
from carnot.reporting.v709_execution import child, progress

Json = dict[str, Any]


def prepare(raw: Path, scratch: Path, inputs: Json) -> tuple[list[str], Path, list[Json]]:
    """Install the qualified wheel offline; only the private extension pathname changes."""
    frozen = inputs["workload"]
    environment = scratch / "environment"
    receipts = [
        child(
            "venv",
            [sys.executable, "-m", "venv", "--system-site-packages", str(environment)],
            raw,
            deadline=90,
        )
    ]
    python = str(environment / "bin/python")
    (site,) = environment.glob("lib/python*/site-packages")
    (site / "local-dependencies.pth").write_text(sysconfig.get_path("purelib") + "\n")
    receipts.append(
        child(
            "install_exact_wheel",
            [
                python,
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                "--ignore-installed",
                frozen["wheel"]["path"],
            ],
            raw,
            deadline=90,
        )
    )
    if not all(r["passed"] for r in receipts):
        raise ValueError("installation_failed")
    extension = site / "carnot_native" / Path(frozen["extension"]["path"]).name
    extension.parent.mkdir()
    shutil.copyfile(frozen["extension"]["path"], extension)
    if e.reference(extension)["sha256"] != frozen["extension"]["sha256"]:
        raise ValueError("installed_extension_hash")
    config = raw / "head.json"
    e.atomic_json(config, dict(frozen["configuration_template"], extension=e.reference(extension)))
    return [python, "-u", "-m", "carnot.inference.resident_protocol"], config, receipts


def session(
    command: list[str],
    config: Path,
    slot: Json,
    requests: list[Json],
    scratch: Path,
    raw: Path,
    deadline_ns: int,
) -> Json:
    """Keep raw frames and parent clocks so setup and failed exits cannot disappear."""
    if time.monotonic_ns() + 40_000_000_000 >= deadline_ns:
        raise TimeoutError("measurement_admission_deadline")
    raw.mkdir(parents=True)
    digest = e.reference(config)["sha256"]
    argv = command + [str(config), digest, slot["arm"]]
    env = dict(os.environ, PYTHONUNBUFFERED="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    progress("8717_before_benchmark_" + slot["unit_id"])
    start = time.monotonic_ns()
    observations: list[Json] = []
    with (raw / "stderr").open("wb") as err:
        proc = subprocess.Popen(
            argv,
            cwd=scratch,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=err,
            start_new_session=True,
        )
        assert proc.stdin is not None and proc.stdout is not None
        inp, out = proc.stdin.fileno(), proc.stdout.fileno()
        try:
            ready = json.loads(wire.read_frame(out, 30) or b"null")
            ready_ns = time.monotonic_ns()
            progress("8717_after_load_" + slot["unit_id"])
            for panel in range(slot["panels"]):
                for index in range(0, len(requests), slot["batch"]):
                    if time.monotonic_ns() + 35_000_000_000 >= deadline_ns:
                        raise TimeoutError("measurement_stop_deadline")
                    begin = time.monotonic_ns()
                    frame = dict(
                        request_id=f"{panel}-{index}",
                        config_hash=digest,
                        deadline_ns=begin + 15_000_000_000,
                        operation="batch",
                        rows=requests[index : index + slot["batch"]],
                    )
                    packed = time.monotonic_ns()
                    wire.write_frame(inp, frame)
                    payload = wire.read_frame(out, 30)
                    received = time.monotonic_ns()
                    response = json.loads(payload or b"null")
                    end = time.monotonic_ns()
                    observations.append(
                        dict(
                            request=frame,
                            response=response,
                            start_ns=begin,
                            packed_ns=packed,
                            received_ns=received,
                            end_ns=end,
                        )
                    )
                    if response.get("status") != "ok":
                        raise ValueError("worker_response")
                progress("8717_panels_" + slot["unit_id"], panel + 1, slot["panels"] - panel - 1)
            wire.write_frame(inp, dict(wire.envelope("shutdown", digest, {}), operation="shutdown"))
            shutdown = json.loads(wire.read_frame(out, 10) or b"null")
            proc.stdin.close()
            proc.wait(timeout=10)
            end = time.monotonic_ns()
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            TimeoutError,
            subprocess.TimeoutExpired,
        ) as error:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)
            e.atomic_json(
                raw / "failed_child.json",
                dict(
                    argv=argv,
                    actual_exit=proc.returncode,
                    error=str(error),
                    start_ns=start,
                    end_ns=time.monotonic_ns(),
                    observations=observations,
                ),
            )
            raise
        finally:
            proc.stdin.close()
            proc.stdout.close()
    progress("8717_after_benchmark_" + slot["unit_id"], 1024, 0)
    value = dict(
        slot=slot,
        argv=argv,
        cwd=str(scratch),
        pid=proc.pid,
        pythonpath_unset=True,
        ready=ready,
        shutdown=shutdown,
        actual_exit=proc.returncode,
        start_ns=start,
        ready_ns=ready_ns,
        end_ns=end,
        observations=observations,
        config_reference=e.reference(config),
        stderr_reference=e.reference(raw / "stderr"),
    )
    e.atomic_json(raw / "session.json", value)
    return value


def reduce(primitive: Json, inputs: Json) -> Json:
    """Rebuild every output check and disjoint cost from retained parent/worker clocks."""
    slot = primitive["slot"]
    config = e.load(primitive["config_reference"])
    ready = primitive["ready"]
    extension = config["extension"] if slot["arm"] == "native" else None
    if (
        config["head"] != inputs["head"]
        or config["max_batch_size"] != 16
        or config["item_budget"] != 1024
        or primitive["actual_exit"] != 0
        or primitive["shutdown"]["status"] != "shutdown"
        or ready
        != dict(
            status="ready",
            pid=primitive["pid"],
            config_hash=primitive["config_reference"]["sha256"],
            loaded_extension=extension,
            config_loads=1,
        )
        or config["extension"]["sha256"] != inputs["workload"]["extension"]["sha256"]
    ):
        raise ValueError("configuration_or_worker_identity")
    requests = inputs["requests"]
    baseline, _ = wire.numeric.batch(inputs["head"], requests, None)
    expected = {r["unit_id"]: json.loads(r["response"]) for r in baseline}
    features = {r["unit_id"]: wire.numeric.extract(r["public"])["feature_hash"] for r in requests}
    rows, parity, costs = [], [], {}
    observations = primitive["observations"]
    if len(observations) != slot["panels"] * 64 // slot["batch"]:
        raise ValueError("request_count")
    previous = primitive["ready_ns"]
    for index, obs in enumerate(observations):
        frame, response = obs["request"], obs["response"]
        panel, offset = divmod(index, 64 // slot["batch"])
        operands = requests[offset * slot["batch"] : (offset + 1) * slot["batch"]]
        digest = primitive["config_reference"]["sha256"]
        if (
            frame["rows"] != operands
            or frame["operation"] != "batch"
            or frame["request_id"] != f"{panel}-{offset * slot['batch']}"
            or response["status"] != "ok"
            or response["request_id"] != frame["request_id"]
            or frame["config_hash"] != digest
            or response["config_hash"] != digest
            or len(response["responses"]) != slot["batch"]
            or len(response["feature_hashes"]) != slot["batch"]
        ):
            raise ValueError("frame_operands")
        if (
            not previous
            <= obs["start_ns"]
            <= obs["packed_ns"]
            <= obs["received_ns"]
            <= obs["end_ns"]
        ):
            raise ValueError("parent_clocks")
        cursor, total = obs["packed_ns"], 0
        for span in response["spans"]:
            if not cursor <= span["start_ns"] <= span["end_ns"] <= obs["received_ns"]:
                raise ValueError("worker_clocks")
            duration = span["end_ns"] - span["start_ns"]
            costs[span["operation"]] = costs.get(span["operation"], 0) + duration
            total += duration
            cursor = span["end_ns"]
        for label, duration in [
            ("envelope_packing", obs["packed_ns"] - obs["start_ns"]),
            ("transport_validation_digest_wire", obs["received_ns"] - obs["packed_ns"] - total),
            ("response_parsing", obs["end_ns"] - obs["received_ns"]),
            ("between_requests_parent", obs["start_ns"] - previous),
        ]:
            costs[label] = costs.get(label, 0) + duration
        for operand, actual, feature in zip(
            operands, response["responses"], response["feature_hashes"], strict=True
        ):
            target = expected[operand["unit_id"]]
            error = abs(actual["probability"] - target["probability"])
            passed = (
                actual["unit_id"] == operand["unit_id"]
                and type(actual["decision"]) is type(target["decision"])
                and actual["decision"] == target["decision"]
                and type(actual["boundary_fallback"]) is bool
                and bool(np.isfinite(actual["probability"]))
                and error <= 1e-12 * (1 + abs(target["probability"]))
                and feature == features[operand["unit_id"]]
            )
            parity.append(
                dict(
                    process_id=slot["unit_id"],
                    panel=panel,
                    unit_id=operand["unit_id"],
                    decision=actual["decision"],
                    expected_decision=target["decision"],
                    probability_error=error,
                    feature_hash=feature,
                    passed=passed,
                )
            )
        rows.append(
            dict(
                process_id=slot["unit_id"],
                request_id=frame["request_id"],
                items=slot["batch"],
                start_ns=obs["start_ns"],
                end_ns=obs["end_ns"],
                latency_ms=(obs["end_ns"] - obs["start_ns"]) / 1e6,
            )
        )
        previous = obs["end_ns"]
    costs["startup_import_configuration"] = primitive["ready_ns"] - primitive["start_ns"]
    costs["shutdown"] = primitive["end_ns"] - previous
    duration = primitive["end_ns"] - primitive["start_ns"]
    if min(costs.values()) < 0 or sum(costs.values()) != duration:
        raise ValueError("complete_cost_clocks")
    process = dict(
        slot,
        status="completed",
        passed=all(p["passed"] for p in parity),
        actual_exit=primitive["actual_exit"],
        duration_s=duration / 1e9,
        throughput_items_s=len(parity) / (duration / 1e9),
        steady_p95_ms=float(np.quantile([r["latency_ms"] for r in rows], 0.95)),
        steady_item_completion_ms=[r["latency_ms"] for r in rows for _ in range(slot["batch"])],
        startup_s=costs["startup_import_configuration"] / 1e9,
        start_ns=primitive["start_ns"],
        end_ns=primitive["end_ns"],
    )
    return dict(
        process_row=process,
        request_rows=rows,
        parity_rows=parity,
        complete_cost_rows=[
            dict(process_id=slot["unit_id"], operation=k, duration_s=v / 1e9)
            for k, v in costs.items()
        ],
    )


def measure(raw: Path, scratch: Path, inputs: Json, deadline_ns: int) -> Json:
    """Retain all intended cells when admission fails or any real process fails."""
    result: Json = dict(
        process_rows=e.roster(e.method()),
        request_rows=[],
        parity_rows=[],
        complete_cost_rows=[],
        references=[],
        receipts=[],
        owned_error=None,
    )
    if not inputs["passed"]:
        return result
    raw.mkdir(parents=True)
    command, config, receipts = prepare(raw, scratch, inputs)
    result["receipts"] = receipts
    result["references"] = [e.reference(config)]
    for index, slot in enumerate(result["process_rows"][:]):
        try:
            primitive = session(
                command,
                config,
                slot,
                inputs["requests"],
                scratch,
                raw / slot["unit_id"],
                deadline_ns,
            )
            reduced = reduce(primitive, inputs)
            result["process_rows"][index] = reduced["process_row"]
            for key in ["request_rows", "parity_rows", "complete_cost_rows"]:
                result[key] += reduced[key]
            result["references"] += [
                e.reference(raw / slot["unit_id"] / "session.json"),
                primitive["stderr_reference"],
            ]
            if not reduced["process_row"]["passed"]:
                raise ValueError("per_item_parity")
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            TimeoutError,
            subprocess.TimeoutExpired,
        ) as error:
            result["process_rows"][index] = dict(
                result["process_rows"][index], status="failed", passed=False, error=str(error)
            )
            failure = raw / slot["unit_id"] / "failed_child.json"
            if failure.is_file():
                result["references"].append(e.reference(failure))
            result["owned_error"] = str(error)
            break
        progress("8717_cells", index + 1, 59 - index)
    return result
