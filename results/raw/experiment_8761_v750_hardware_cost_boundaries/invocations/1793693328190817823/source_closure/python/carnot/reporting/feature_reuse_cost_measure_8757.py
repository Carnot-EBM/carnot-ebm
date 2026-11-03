"""REQ-REPORT-8757: reuse the existing installed worker and complete cost reducer."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import subprocess
import time
from typing import Any

from carnot.reporting import feature_reuse_cost_8757 as e
from carnot.reporting import installed_batch_cost_measure_8717 as original
from carnot.reporting.v709_execution import progress

Json = dict[str, Any]


def observe(
    command: list[str],
    config: Path,
    slot: Json,
    inputs: Json,
    scratch: Path,
    raw: Path,
    deadline_ns: int,
) -> Json:
    """The original protocol retains validation and scoring; diagnostic wrappers stay off."""
    primitive = original.session(
        command, config, slot, inputs["requests"], scratch, raw, deadline_ns
    )
    stored = e.load(primitive["config_reference"])
    if stored["reuse_request_features"] is not (slot["mode"] == "reuse"):
        raise ValueError("optimization_state")
    reduced: Json = original.reduce(primitive, inputs)
    if not reduced["process_row"]["passed"]:
        raise ValueError("numerical_parity")
    outputs = [
        dict(responses=o["response"]["responses"], feature_hashes=o["response"]["feature_hashes"])
        for o in primitive["observations"]
    ]
    reduced["process_row"].update(
        scientific_request_sha256=e.canonical_hash(inputs["requests"]),
        output_sha256=e.canonical_hash(outputs),
        session_reference=e.reference(raw / "session.json"),
    )
    return reduced


def measure(raw: Path, scratch: Path, inputs: Json, deadline_ns: int) -> Json:
    """A single bounded four-arm probe admits all40 cells or leaves all40 censored."""
    result: Json = dict(
        process_rows=e.roster(),
        request_rows=[],
        parity_rows=[],
        complete_cost_rows=[],
        references=[],
        receipts=[],
        owned_error=None,
        gate_check_summary=[],
        probe_rows=[],
    )
    if not inputs["passed"]:
        return result
    raw.mkdir(parents=True)
    command, baseline, receipts = original.prepare(raw, scratch, inputs)
    result["receipts"] = receipts
    configs = dict(baseline=baseline, reuse=raw / "reuse.json")
    e.atomic_json(
        configs["reuse"], dict(e.load(e.reference(baseline)), reuse_request_features=True)
    )
    result["references"] += [e.reference(p) for p in configs.values()]
    # The installed extension survives private scratch cleanup for fresh replay.
    baseline_config = e.load(e.reference(baseline))
    retained = raw / "installed-extension.so"
    retained.write_bytes(Path(baseline_config["extension"]["path"]).read_bytes())
    result["references"].append(e.reference(retained))
    for slot in result["process_rows"][:4]:
        observed = observe(
            command,
            configs[slot["mode"]],
            slot,
            inputs,
            scratch,
            raw / "probe" / slot["unit_id"],
            deadline_ns,
        )
        row = observed["process_row"]
        result["probe_rows"].append(row)
        result["references"].append(row["session_reference"])
    projected = 40 * max(r["duration_s"] for r in result["probe_rows"]) * 1.25
    remaining = min(2700.0, (deadline_ns - time.monotonic_ns()) / 1e9)
    result["admission"] = dict(
        passed=projected <= remaining,
        projected_with_reserve_s=projected,
        budget_s=2700,
        remaining_budget_s=remaining,
        reserve_fraction=0.25,
        probe_process_count=4,
        primary_probe_cells=0,
    )
    if not result["admission"]["passed"]:
        result["gate_check_summary"].append(
            dict(
                upstream="fixed_workload_cost_admission",
                path=str(raw),
                sha256=e.canonical_hash(result["probe_rows"]),
                check="cost_projection",
                field="projected_with_reserve_s",
                operator="<=",
                expected=remaining,
                observed=projected,
                observed_state="over_budget",
            )
        )
        return result
    for index, slot in enumerate(deepcopy(result["process_rows"])):
        try:
            observed = observe(
                command,
                configs[slot["mode"]],
                slot,
                inputs,
                scratch,
                raw / slot["unit_id"],
                deadline_ns,
            )
            row = observed["process_row"]
            for previous in result["process_rows"][:index]:
                if (
                    previous["arm"] == row["arm"]
                    and previous["output_sha256"] != row["output_sha256"]
                ):
                    raise ValueError("same_provider_output_bytes")
            result["process_rows"][index] = row
            for key in ["request_rows", "parity_rows", "complete_cost_rows"]:
                result[key] += observed[key]
            result["references"] += [
                row["session_reference"],
                e.reference(raw / slot["unit_id"] / "stderr"),
            ]
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            TimeoutError,
            subprocess.TimeoutExpired,
        ) as error:
            result["process_rows"][index] = dict(slot, status="failed", error=str(error))
            result["owned_error"] = str(error)
            failure = raw / slot["unit_id"] / "failed_child.json"
            if failure.is_file():
                result["references"].append(e.reference(failure))
            break
        progress("8757_process_cells", index + 1, 39 - index)
    return result
