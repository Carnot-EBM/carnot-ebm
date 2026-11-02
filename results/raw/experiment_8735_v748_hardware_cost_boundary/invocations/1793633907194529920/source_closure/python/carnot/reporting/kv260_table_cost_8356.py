"""REQ-REPORT-8356: table costs need their own authenticated configurations."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from carnot.reporting import spline_table_fidelity_8352 as producer
from carnot.reporting import kv260_local_cost_boundary_8315 as base
from carnot.reporting.request_trace_inventory_8200 import operand
from carnot.reporting.primary_publication import validate_primary, read_bound_sidecar

Json = dict[str, Any]
n = producer.n


def authenticate(path: Path, raw: Path, work: Json) -> Json | None:
    """The table producer stores shard hashes as a map; authenticate its actual schema."""
    work["checks"].append(operand("exists", path, True, path.is_file()))
    if not path.is_file():
        return None
    try:
        value: Json = json.loads(base.pin(path, raw, work["refs"]).read_bytes())
        validate_primary(value, path)
        terminal_path = Path(value["terminal_validation_sidecar_path"])
        terminal = json.loads(base.pin(terminal_path, raw, work["refs"]).read_bytes())
        publication = terminal["publication"]
        report = read_bound_sidecar(path, Path(publication["sidecar_path"]))
        base.pin(Path(publication["sidecar_path"]), raw, work["refs"])
        if (
            publication["primary_sha256"] != base.sha256_file(path)
            or publication["primary_path"] != str(path.absolute())
            or report["primary_path"] != str(path.absolute())
            or report["report"]["passed"] is not True
            or value["required_checks_passed"] is not True
            or value["flagged_adversarial"] is not False
            or value["table_fidelity_ready_score"] != 1
        ):
            raise ValueError("table_terminal_readiness")
        return value
    except (OSError, ValueError, KeyError, TypeError) as error:
        work["checks"].append(operand("table_terminal_authentication", path, "valid", str(error)))
        return None


def transaction(head: Json, size: int, storage: str, interpolation: str, repetition: int) -> Json:
    """Clock lookup, coefficient writes and refresh separately after table preparation."""
    values, _ = n.table(head, size, storage)
    x, _, _ = n.panel(head)
    proposed = n.update(head, n.events()[0], 0)
    start = time.monotonic_ns()
    before = time.monotonic_ns()
    probabilities = n.lookup(head, x, values, storage, interpolation)
    evaluation = time.monotonic_ns() - before
    before = time.monotonic_ns()
    updated = deepcopy(head)
    updated["coefficients"][2:] = proposed["coefficients"][2:]
    writes = time.monotonic_ns() - before
    before = time.monotonic_ns()
    refreshed, entries = n.refresh(head, updated, values, storage)
    refresh = time.monotonic_ns() - before
    end = time.monotonic_ns()
    return dict(
        head=deepcopy(head),
        grid_points=size,
        storage=storage,
        interpolation=interpolation,
        repetition=repetition,
        branch="table",
        arm=f"{size}-{storage}-{interpolation}",
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
        wall_ns=end - start,
        operation_ns=dict(
            table_evaluation=evaluation, coefficient_writes=writes, table_refresh=refresh
        ),
        probabilities=probabilities.tolist(),
        coefficients=updated["coefficients"],
        affected_entries=entries,
        table_hex=refreshed.tobytes().hex(),
        vector_count=len(x),
        refresh_events=1,
    )


def verify(row: Json) -> None:
    """Rebuild the table independently of scoped refresh and preserve approximation error."""
    head, size, storage = row["head"], row["grid_points"], row["storage"]
    if (
        (size, storage, row["interpolation"]) not in n.CONFIGS
        or type(row["repetition"]) is not int
        or row["repetition"] not in range(-1, 5)
        or row["branch"] != "table"
        or row["arm"] != f"{size}-{storage}-{row['interpolation']}"
    ):
        raise ValueError("table_configuration")
    x, _, _ = n.panel(head)
    values, _ = n.table(head, size, storage)
    expected = n.lookup(head, x, values, storage, row["interpolation"]).tolist()
    updated = n.update(head, n.events()[0], 0)
    _, entries = n.refresh(head, updated, values, storage)
    if (
        row["probabilities"] != expected
        or row["coefficients"] != updated["coefficients"]
        or row["table_hex"] != n.table(updated, size, storage)[0].tobytes().hex()
        or row["affected_entries"] != entries
        or row["vector_count"] != len(x)
        or row["refresh_events"] != 1
    ):
        raise ValueError("table_semantics")
    spans = row["operation_ns"]
    if (
        set(spans) != {"table_evaluation", "coefficient_writes", "table_refresh"}
        or any(type(v) is not int or v <= 0 for v in spans.values())
        or row["wall_ns"] != row["ended_monotonic_ns"] - row["started_monotonic_ns"]
        or sum(spans.values()) > row["wall_ns"]
    ):
        raise ValueError("table_clock")


def complete(rows: list[Json]) -> bool:
    """Require five distinct measured repeats for every exact source configuration."""
    return len(rows) == 5 * len(n.CONFIGS) and all(
        {
            row["repetition"]
            for row in rows
            if (row["grid_points"], row["storage"], row["interpolation"]) == config
        }
        == set(range(5))
        for config in n.CONFIGS
    )


def optional(root: Path, raw: Path, work: Json) -> None:
    """An absent table primary blocks only table costs; benchmark errors are owned."""
    from carnot.reporting.kv260_workload_cost_8356 import progress

    path = root / "results" / (producer.NAME + ".json")
    progress("before_authenticate_tables")
    work["table_rows"] = []
    work["table_owned_failure"] = False
    value = authenticate(path, raw, work)
    measuring = False
    if value is not None:
        try:
            if value["task_id"] != producer.TASK or value["milestone"] != "2026.10.720":
                raise ValueError("table_identity")
            if not producer.replay(path):
                raise ValueError("table_independent_replay")
            source = json.loads(
                base.pin(base.checked(value["work_reference"]), raw, work["refs"]).read_bytes()
            )
            for ref in (
                source["inputs"]["refs"]
                + source["code_refs"]
                + source["primitive_refs"]
                + [
                    r[key]
                    for r in source["configurations"]
                    for key in ["table_reference", "probabilities_reference", "timings_reference"]
                ]
            ):
                base.pin(base.checked(ref), raw, work["refs"])
            work["table_source_reference"] = base.reference(path)
            head = source["inputs"]["head"]
            if [
                (r["grid_points"], r["storage"], r["interpolation"])
                for r in source["configurations"]
            ] != n.CONFIGS:
                raise ValueError("table_exact_configurations")
            measuring = True
            for index, config in enumerate(n.CONFIGS):
                for repetition in range(-1, 5):
                    progress(
                        "before_benchmark_table",
                        index * 6 + repetition + 1,
                        72 - index * 6 - repetition - 1,
                    )
                    row = transaction(head, config[0], config[1], config[2], repetition)
                    verify(row)
                    if repetition >= 0:
                        work["table_rows"].append(row)
                    progress(
                        "after_benchmark_table",
                        index * 6 + repetition + 2,
                        71 - index * 6 - repetition - 1,
                    )
        except (OSError, ValueError, KeyError, TypeError) as error:
            work["owned_failure"] |= measuring
            work["table_owned_failure"] = measuring
            work["table_rows"] = []
            work["checks"].append(
                operand("table_authentication_or_benchmark", path, "valid", str(error))
            )
    progress("after_authenticate_tables", len(work["table_rows"]), 60 - len(work["table_rows"]))
