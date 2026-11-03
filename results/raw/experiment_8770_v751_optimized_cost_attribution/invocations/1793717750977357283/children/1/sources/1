"""REQ-REPORT-8770: twelve unchanged optimized installed process observations."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
from typing import Any

import numpy as np

from carnot.reporting import optimized_cost_attribution_8770 as e
from carnot.reporting import installed_cost_attribution_measure_8733 as diagnostic
from carnot.reporting import installed_batch_cost_measure_8717 as installed
from carnot.reporting.v709_execution import progress

Json = dict[str, Any]


def reduce(primitive: Json, inputs: Json) -> Json:
    """Clip child leaves to the steady parent interval before adding lifecycle costs."""
    if (
        primitive["slot"]["mode"] != "reuse"
        or e.load(primitive["config_reference"])["reuse_request_features"] is not True
    ):
        raise ValueError("optimization_state")
    reduced = diagnostic.reduce(primitive, inputs)
    original_row = installed.reduce(primitive, inputs)["process_row"]
    if not original_row["passed"]:
        raise ValueError("original_numerical_parity")
    row = reduced["process_row"]
    steady_start, steady_end = primitive["ready_ns"], primitive["observations"][-1]["end_ns"]
    costs: dict[str, float] = defaultdict(float)
    for span in primitive["profile"]["spans"]:
        length = max(0, min(span["end_ns"], steady_end) - max(span["start_ns"], steady_start))
        costs[span["operation"]] += length / 1e9
    costs["process_startup"], costs["process_shutdown"] = row["startup_s"], row["shutdown_s"]
    costs["unattributed_residual"] = row["duration_s"] - sum(costs.values())
    if costs["unattributed_residual"] < -1e-12:
        raise ValueError("parent_span_overlap")
    row.update(
        loaded_extension_reference=primitive["ready"]["loaded_extension"],
        complete_components=dict(costs),
        parent_steady_request_p95_ms=original_row["steady_p95_ms"],
        child_steady_scoring_p95_ms=float(
            np.quantile(
                [
                    sum(s["end_ns"] - s["start_ns"] for s in o["response"]["spans"]) / 1e6
                    for o in primitive["observations"]
                ],
                0.95,
            )
        ),
        child_steady_scoring_boundary="numeric.batch only; request-local feature preparation is attributed separately",
        steady_item_completion_ms=original_row["steady_item_completion_ms"],
        scientific_request_sha256=e.canonical_hash(inputs["requests"]),
        output_sha256=e.canonical_hash(
            [
                dict(
                    responses=o["response"]["responses"],
                    feature_hashes=o["response"]["feature_hashes"],
                )
                for o in primitive["observations"]
            ]
        ),
        session_reference=e.reference(Path(primitive["session_reference"]["path"])),
    )
    return reduced


def measure(raw: Path, scratch: Path, inputs: Json, deadline_ns: int) -> Json:
    """Install once; both timing conditions load the exact same diagnostic sources."""
    result: Json = dict(
        process_rows=e.roster(),
        request_rows=[],
        parity_rows=[],
        complete_cost_rows=[],
        references=[],
        receipts=[],
        owned_error=None,
        gate_check_summary=[],
    )
    if not inputs["passed"]:
        return result
    raw.mkdir(parents=True)
    command, config, receipts = installed.prepare(raw, scratch, inputs)
    result["receipts"] = receipts
    (site,) = (Path(command[0]).parents[1] / "lib").glob("python*/site-packages")
    for name in ["installed_profile_8733.py", "optimized_profile_8770.py"]:
        source = e.ROOT / "python/carnot/inference" / name
        target = site / "carnot/inference" / name
        shutil.copyfile(source, target)
        if e.reference(source)["sha256"] != e.reference(target)["sha256"]:
            raise ValueError("diagnostic_install_hash")
    command[-1] = "carnot.inference.optimized_profile_8770"
    retained = raw / "installed-extension.so"
    retained.write_bytes(Path(e.load(e.reference(config))["extension"]["path"]).read_bytes())
    result["references"] += [e.reference(config), e.reference(retained)]
    for index, slot in enumerate(result["process_rows"][:]):
        cell = raw / slot["unit_id"]
        try:
            profile = cell / "profile.json"
            primitive = diagnostic.session(
                command, config, slot, inputs["requests"], scratch, cell, deadline_ns
            )
            primitive.update(
                profile_reference=e.reference(profile),
                profile=e.load(e.reference(profile)),
                session_reference=e.reference(cell / "session.json"),
            )
            # The session binds its own pathname, not its own changing content hash.
            primitive["session_reference"] = dict(path=str(cell / "session.json"))
            e.atomic_json(cell / "session.json", primitive)
            reduced = reduce(primitive, inputs)
            reduced["process_row"]["session_reference"] = e.reference(cell / "session.json")
            result["process_rows"][index] = reduced["process_row"]
            for key in ["request_rows", "parity_rows", "complete_cost_rows"]:
                result[key] += reduced[key]
            result["references"] += [
                e.reference(cell / "session.json"),
                e.reference(profile),
                primitive["stderr_reference"],
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
            break
        progress("8770_process_cells", index + 1, 11 - index)
    return result


original = SimpleNamespace(reduce=reduce)
