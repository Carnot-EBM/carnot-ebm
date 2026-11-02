"""REQ-REPORT-8494: authenticate durable append before comparing complete clocks.

Narrow adapters retain the historical reporting gates and independent oracle.
The historical snapshot/journal measurement remains a separate comparator.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import append_store_qualification_8493 as upstream
from carnot.reporting import journal_service_cost_8466 as legacy
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import read_bound_sidecar, reader_receipt
from carnot.reporting.v709_execution import child
from carnot.verify import append_service_cost_8494 as k
from carnot.verify.append_store_8493 import Store

Json = dict[str, Any]
ROOT = upstream.ROOT
NAME, TASK = "experiment_8494_v730_append_service_cost", "exp8494-append-service-cost"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_append_service_cost_8494.py"
OWNED = [
    "python/carnot/verify/append_service_cost_8494.py",
    "python/carnot/reporting/append_service_cost_8494.py",
    "python/carnot/reporting/append_service_cost_runner_8494.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
UPSTREAM = "results/experiment_8493_v730_append_store_qualification.json"
UPSTREAM_PIN = "sha256:55ce337aacfdc707ea068bd1c4a2b38ccf220d16e9154c8354370680b0081e52"
REQUIRED = upstream.REQUIRED
reference, read, checksum, progress = (
    upstream.reference,
    upstream.read,
    upstream.checksum,
    k.progress,
)
ORIGINAL_ADMISSION = upstream.admission


def store_gate(
    check: str, path: Path, field: str, expected: Any, observed: Any
) -> upstream.GateFailure:
    """Identify the append producer explicitly so missing input never names the protocol instead."""
    error = upstream.GateFailure(check, path, field, expected, observed)
    error.row.update(upstream="exp8493-append-store-qualification", expected_sha256=UPSTREAM_PIN)
    return error


def admission(root: Path, raw: Path) -> Json:
    """Require both sealed producer identities, live consumers and original cold replay."""
    with patch.object(upstream, "TASK", TASK), patch.object(upstream, "NAME", NAME):
        inputs = ORIGINAL_ADMISSION(root, raw)
    path = root / UPSTREAM
    try:
        value = read(dict(path=str(path), sha256=UPSTREAM_PIN))
    except (OSError, ValueError) as error:
        raise store_gate(
            "qualified_append_store_input",
            path,
            "sha256",
            UPSTREAM_PIN,
            upstream.upstream.sha256_file(path) if path.exists() else None,
        ) from error
    fields = ("append_store_ready_score", "required_checks_passed", "flagged_adversarial")
    expected = dict(zip(fields, (1, True, False), strict=True))
    observed = {f: value.get(f) for f in fields}
    if observed != expected:
        raise store_gate("qualified_append_store", path, " AND ".join(fields), expected, observed)
    terminal = read(reference(Path(value["terminal_validation_sidecar_path"])))
    side = read_bound_sidecar(path, Path(terminal["publication"]["sidecar_path"]))
    if (
        not side["report"]["passed"]
        or not reader_receipt(
            "exp8493-append-store-qualification", path.parent, field="append_store_ready_score"
        )["passed"]
    ):
        raise store_gate("qualified_store_consumer", path, "bound_consumer_passed", True, False)
    for ref in value["code_config_hashes"] + value["source_artifact_hashes"]:
        read_bytes = Path(ref["path"])
        if upstream.upstream.sha256_file(read_bytes) != ref["sha256"]:
            raise store_gate(
                "qualified_store_bytes",
                read_bytes,
                "sha256",
                ref["sha256"],
                upstream.upstream.sha256_file(read_bytes),
            )
    receipt = child(
        "qualified_store_original_replay",
        [sys.executable, "-u", str(ROOT / upstream.CLI), "--cold-replay", str(path)],
        raw / "store_upstream_logs",
        deadline=240,
        scope="upstream",
    )
    if not receipt["passed"]:
        raise store_gate("qualified_store_replay", path, "original_replay_passed", True, False)
    inputs["refs"] += [
        reference(path),
        reference(Path(terminal["publication"]["sidecar_path"])),
        *value["code_config_hashes"],
    ]
    inputs["store_receipt"] = receipt
    inputs["receipt"] = dict(inputs.get("receipt", {}), store_qualification_receipt=receipt)
    return dict(inputs)


def worker(frozen_path: Path, directory: Path, arm: str) -> int:
    """Save genuine primitive clocks and store bytes after complete work and restart."""
    frozen = json.loads(frozen_path.read_bytes())
    row = k.run_repeat(frozen, directory, arm)
    row["frozen_reference"] = reference(frozen_path)
    row["store_references"] = [
        reference(p) for p in sorted((directory / "store").iterdir()) if p.is_file()
    ]
    atomic_json(directory / "primitive.json", row)
    return 0


def measure(root: Path, raw: Path) -> Json:
    """Reuse preflight checks, then retain all24 arms even after failure or deadline."""
    with (
        patch.multiple(upstream, TASK=TASK, NAME=NAME, OWNED=OWNED, TEST=TEST, SLOTS=[]),
        patch.object(upstream, "admission", admission),
    ):
        work = upstream.measure(root, raw)
    work.update(run_date="20261013", warmups=[], pairs=[], cells=[])
    work["preconditions_checked"]["measurement_cap_s"] = 1800
    work["preconditions_checked"]["measurement_filesystem"] = upstream.upstream.filesystem(raw)
    start = time.monotonic_ns()
    for cell_index, cell in enumerate(k.CELLS):
        for repeat in range(6):
            for arm in ("snapshot", "append") if repeat % 2 == 0 else ("append", "snapshot"):
                unit = f"cell{cell_index}-repeat{repeat}-{arm}"
                row: Json = dict(
                    unit_id=unit,
                    family="append8494_engineering_fixture",
                    cell=cell,
                    repeat=repeat,
                    warmup=repeat == 0,
                    arm=arm,
                    status="censored",
                    absolute_metric=None,
                    missing_reason="required_operands_missing",
                )
                if not work["gate_check_summary"] and not work["owned_failures"].get(
                    "preconditions"
                ):
                    remaining = 1800 - (time.monotonic_ns() - start) / 1e9
                    if remaining <= 0:
                        row["missing_reason"] = "measurement_deadline"
                    else:
                        progress(
                            "benchmark_before_" + unit, len(work["rows"]), 24 - len(work["rows"])
                        )
                        try:
                            directory = raw / unit
                            frozen = work["tapes"][cell_index]
                            frozen_path = directory / "frozen.json"
                            atomic_json(frozen_path, frozen)
                            receipt = child(
                                unit,
                                [
                                    sys.executable,
                                    "-u",
                                    str(ROOT / CLI),
                                    "--worker",
                                    str(frozen_path),
                                    "--store",
                                    str(directory),
                                    "--arm",
                                    arm,
                                ],
                                directory / "logs",
                                deadline=min(120, remaining),
                            )
                            if not receipt["passed"]:
                                raise ValueError("measurement_child_failed")
                            ref = reference(directory / "primitive.json")
                            metrics = k.reduce(frozen, read(ref))
                            row.update(
                                status="completed",
                                absolute_metric=metrics["wall_s"],
                                missing_reason=None,
                                metrics=metrics,
                                primitive_reference=ref,
                                receipt=receipt,
                            )
                            work["primitives"].append(ref)
                        except (OSError, ValueError, KeyError, TypeError) as error:
                            work["owned_failures"][unit] = str(error)
                            row.update(status="failed", missing_reason=str(error))
                        progress(
                            "benchmark_after_" + unit, len(work["rows"]) + 1, 23 - len(work["rows"])
                        )
                work["rows"].append(row)
    work["ended_monotonic_ns"] = time.monotonic_ns()
    work["phase_spans"].append(
        dict(phase="complete_service_pairs", start_ns=start, end_ns=work["ended_monotonic_ns"])
    )
    atomic_json(raw / "measurement.json", work)
    return dict(work)


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Reuse qualified complete-service gates; only primary blocks determine benefit."""
    normalized = deepcopy(work)
    for row in normalized["rows"]:
        if row["arm"] == "append":
            row["arm"] = "journal"
    with patch.multiple(
        legacy, TASK=TASK, NAME=NAME, OWNED=OWNED, TEST=TEST, REQUIRED=REQUIRED, k=k
    ):
        value = legacy.build(normalized, receipts, output)
    value.update(
        experiment_id=8494,
        milestone="2026.10.730",
        honest_verdict="complete_" + value["verdict_class"] + "_append_service_cost",
        rows=work["rows"],
        append_speed_benefit_score=value.pop("journal_speed_benefit_score"),
        append_store_ready_score=value.pop("journal_ready_score"),
        primary_speedup_ci95=value["cell_results"][0]["speedup_ci95"],
        secondary_speedup_ci95=value["cell_results"][1]["speedup_ci95"],
        primary_p95_ratio=value["cell_results"][0]["p95_ratio"],
        preserved_v728=work.get("preserved_v728", {}),
        warmup_rows=[r for r in work["rows"] if r["warmup"]],
        bytes_scope="actual Linux process writes minus progress bytes; no timing overhead subtraction",
        memory_scope="fresh arm peak RSS; peak disk sampled at every actual sync including temporary checkpoints",
        p95_scope="append/snapshot ratio of mean repeat p95; transaction latency applies to every batch member",
        service_boundary="unchanged cached numerical oracle; cold reconstruction, startup, features, predictions, delayed feedback, durable response and final drain",
        namespace_scope="successful file create/create-open, directory creation, replace and unlink calls observed identically in both arms",
        historical_validation_receipts=work.get("receipts", []),
        durability_checks_passed=not work["gate_check_summary"] and not work["owned_failures"],
    )
    del value["p95_ratio"]
    for pair in value["paired_repeat_rows"]:
        pair["append"] = pair.pop("journal")
        pair["append_wall_s"] = pair.pop("journal_wall_s")
    value["field_principles"] = {
        key: "Bind complete matched durable work to sealed operands; only primary repeat blocks support conditional engineering benefit."
        for key in value
    }
    value["field_principles"]["append_store_ready_score"] = (
        "Internal compatibility scalar for the shipped publication workflow; equals service_cost_ready_score."
    )
    value["reproducibility_checksum"] = checksum(value)
    return dict(value)


def replay(path: Path) -> bool:
    """Independent sealed tapes and retained store bytes reject repaired primitive summaries."""
    try:
        value = json.loads(path.read_bytes())
        if value["experiment_id"] != 8494 or value["reproducibility_checksum"] != checksum(value):
            return False
        work = read(value["work_reference"])
        if len(work["rows"]) != 24:
            return False
        if (
            work["authority"].get("activated")
            and upstream.upstream.methods.tasks_digest(work["authority"]["tasks"])
            != upstream.upstream.methods.TASKS_PIN
        ):
            return False
        for ref in [
            *value["source_artifact_hashes"],
            *value["code_config_hashes"],
            *value["raw_shard_hashes"],
        ]:
            if upstream.upstream.sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        with TemporaryDirectory(dir="/var/tmp", prefix="exp8494-replay-") as folder:
            if work.get("tapes"):
                inputs = admission(Path(work["root"]), Path(folder))
                if k.base.encode(inputs["tapes"]) != k.base.encode(work["tapes"]):
                    return False
            roster = [
                (cell, repeat, arm)
                for cell in k.CELLS
                for repeat in range(6)
                for arm in (("snapshot", "append") if repeat % 2 == 0 else ("append", "snapshot"))
            ]
            for index, (row, slot) in enumerate(zip(work["rows"], roster, strict=True)):
                progress("cold_primitive", index, 24 - index)
                if (row["cell"], row["repeat"], row["arm"]) != slot:
                    return False
                if row["status"] != "completed":
                    continue
                primitive = read(row["primitive_reference"])
                frozen = read(primitive["frozen_reference"])
                sealed = work["tapes"][k.CELLS.index(row["cell"])]
                if k.base.encode(frozen) != k.base.encode(sealed) or k.base.encode(
                    row["metrics"]
                ) != k.base.encode(k.reduce(frozen, primitive)):
                    return False
                for ref in primitive["store_references"]:
                    if upstream.upstream.sha256_file(Path(ref["path"])) != ref["sha256"]:
                        return False
                target = Path(folder) / str(index)
                shutil.copytree(Path(primitive["directory"]) / "store", target)
                store = (k.base.Snapshot if row["arm"] == "snapshot" else Store)(
                    target, frozen
                )
                if k.base.encode(store.read()) != k.base.encode(primitive["final"]):
                    return False
        return bool(
            k.base.encode(value)
            == k.base.encode(build(work, value["validation_receipts"], Path(work["raw"]), path))
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False
