"""REQ-REPORT-8733: pair the original installed worker with opt-in local timing."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import shutil
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import installed_batch_cost_measure_8717 as original

Json = dict[str, Any]
DEADLINES: dict[tuple[str, int, str], int] = {}


def prepare(raw: Path, scratch: Path, inputs: Json) -> tuple[list[str], Path, list[Json]]:
    """Install the exact qualified wheel; both modes import the same diagnostic file."""
    from carnot.reporting import installed_cost_attribution_8733 as a

    command, config, receipts = original.prepare(raw, scratch, inputs)
    (site,) = (Path(command[0]).parents[1] / "lib").glob("python*/site-packages")
    source = a.e.ROOT / "python/carnot/inference/installed_profile_8733.py"
    target = site / "carnot/inference/installed_profile_8733.py"
    shutil.copyfile(source, target)
    assert a.e.reference(source)["sha256"] == a.e.reference(target)["sha256"]
    return command[:-1] + ["carnot.inference.installed_profile_8733"], config, receipts


def session(
    command: list[str],
    config: Path,
    slot: Json,
    requests: list[Json],
    scratch: Path,
    raw: Path,
    deadline_ns: int,
) -> Json:
    """A pair shares real future deadlines so the full transmitted bytes match exactly."""
    from carnot.reporting import installed_cost_attribution_8733 as a

    key = (str(config), slot["repeat"], slot["arm"])
    deadline = DEADLINES.setdefault(key, time.monotonic_ns() + 25_000_000_000)
    writer = original.wire.write_frame

    def fixed(fd: int, value: Json, *args: Any, **kw: Any) -> None:
        if value.get("operation") == "batch":
            value["deadline_ns"] = deadline
        writer(fd, value, *args, **kw)

    profile_path = raw / "profile.json"
    argv = command + ["--profile-output", str(profile_path)]
    if slot["instrumented"]:
        argv += ["--instrument"]
    with patch.object(original.wire, "write_frame", fixed):
        value: Json = original.session(argv, config, slot, requests, scratch, raw, deadline_ns)
    value["profile_reference"] = a.e.reference(profile_path)
    value["profile"] = a.e.load(value["profile_reference"])
    a.e.atomic_json(raw / "session.json", value)
    return value


def reduce(primitive: Json, inputs: Json) -> Json:
    """Compare scientific outputs and sum only disjoint spans inside their own parent."""
    from carnot.reporting import installed_cost_attribution_8733 as a

    slot, profile = primitive["slot"], primitive["profile"]
    config = a.e.load(primitive["config_reference"])
    if (
        primitive["actual_exit"] != 0
        or profile["actual_exit"] != 0
        or profile["instrumented"] != slot["instrumented"]
        or profile["forbidden_imports"]
        or config["head"] != inputs["head"]
        or config["max_batch_size"] != 16
        or config["item_budget"] != 1024
        or primitive["shutdown"]["status"] != "shutdown"
        or config["extension"]["sha256"] != inputs["workload"]["extension"]["sha256"]
        or primitive["ready"]["loaded_extension"]
        != (config["extension"] if slot["arm"] == "native" else None)
    ):
        raise ValueError("configuration_or_worker_identity")
    if a.e.load(primitive["profile_reference"]) != profile:
        raise ValueError("profile_bytes")
    costs: dict[str, float] = defaultdict(float)
    cursor = profile["start_ns"]
    for span in profile["spans"]:
        if not cursor <= span["start_ns"] <= span["end_ns"] <= profile["end_ns"]:
            raise ValueError("local_span_clocks")
        costs[span["operation"]] += (span["end_ns"] - span["start_ns"]) / 1e9
        cursor = span["end_ns"]
    child_duration = (profile["end_ns"] - profile["start_ns"]) / 1e9
    if child_duration < 0 or sum(costs.values()) > child_duration:
        raise ValueError("local_span_sum")
    observations = primitive["observations"]
    if len(observations) != slot["panels"] * 4:
        raise ValueError("request_count")
    baseline, _ = original.wire.numeric.batch(inputs["head"], inputs["requests"], None)
    expected = {r["unit_id"]: json.loads(r["response"]) for r in baseline}
    features = {
        r["unit_id"]: original.wire.numeric.extract(r["public"])["feature_hash"]
        for r in inputs["requests"]
    }
    request_rows, parity, frames, outputs = [], [], [], []
    previous = primitive["ready_ns"]
    for index, obs in enumerate(observations):
        frame, response = obs["request"], obs["response"]
        panel, offset = divmod(index, 4)
        operands = inputs["requests"][offset * 16 : (offset + 1) * 16]
        if (
            frame["rows"] != operands
            or frame["request_id"] != f"{panel}-{offset * 16}"
            or frame["config_hash"] != primitive["config_reference"]["sha256"]
            or response["status"] != "ok"
            or response["request_id"] != frame["request_id"]
            or response["config_hash"] != frame["config_hash"]
            or len(response["responses"]) != 16
            or response["feature_hashes"] != [features[r["unit_id"]] for r in operands]
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
        for actual in response["responses"]:
            target = expected[actual["unit_id"]]
            if (
                actual["decision"] != target["decision"]
                or type(actual["boundary_fallback"]) is not bool
                or not abs(actual["probability"] - target["probability"]) <= 1e-12
            ):
                raise ValueError("scientific_parity")
            parity.append(dict(process_id=slot["unit_id"], unit_id=actual["unit_id"], passed=True))
        frames.append(frame)
        outputs.append(
            dict(responses=response["responses"], feature_hashes=response["feature_hashes"])
        )
        request_rows.append(
            dict(process_id=slot["unit_id"], duration_s=(obs["end_ns"] - obs["start_ns"]) / 1e9)
        )
        previous = obs["end_ns"]
    duration = (primitive["end_ns"] - primitive["start_ns"]) / 1e9
    startup = (primitive["ready_ns"] - primitive["start_ns"]) / 1e9
    shutdown = (primitive["end_ns"] - previous) / 1e9
    residual = duration - startup - shutdown - sum(r["duration_s"] for r in request_rows)
    if min(startup, shutdown, residual) < 0:
        raise ValueError("parent_residual")
    process = dict(
        slot,
        status="completed",
        passed=True,
        actual_exit=0,
        duration_s=duration,
        request_digest=a.e.canonical_hash(frames),
        output_digest=a.e.canonical_hash(outputs),
        components=dict(costs),
        child_duration_s=child_duration,
        child_unexplained_s=child_duration - sum(costs.values()),
        startup_s=startup,
        shutdown_s=shutdown,
        end_to_end_residual_s=residual,
        parent_components=dict(
            process_startup=startup,
            process_shutdown=shutdown,
            request_roundtrips=sum(r["duration_s"] for r in request_rows),
            parent_unexplained=residual,
        ),
    )
    return dict(
        process_row=process,
        request_rows=request_rows,
        parity_rows=parity,
        complete_cost_rows=[
            dict(
                process_id=slot["unit_id"],
                operation=k,
                duration_s=v,
                parent="child_serve",
                parent_duration_s=child_duration,
            )
            for k, v in costs.items()
        ]
        + [
            dict(
                process_id=slot["unit_id"],
                operation=k,
                duration_s=v,
                parent="parent_process",
                parent_duration_s=duration,
            )
            for k, v in process["parent_components"].items()
        ],
    )
