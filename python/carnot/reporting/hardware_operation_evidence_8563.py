"""REQ-REPORT-8563: producer seals authorize scope; summaries cannot authorize hardware."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from carnot.reporting import complete_native_cost_8559 as native
from carnot.reporting import hardware_operation_evidence_8550 as previous
from carnot.reporting.primary_publication import read_bound_sidecar

Json = dict[str, Any]
failure, OPERATIONS, BOARDS, ARITHMETIC = (
    previous.failure,
    previous.OPERATIONS,
    previous.BOARDS,
    previous.ARITHMETIC,
)
HISTORY = "results/experiment_8550_v734_hardware_operation_boundary.json"
HISTORY_PIN = "sha256:45edf9a07d66b9e17fe5893b1c3c638096799aa6d9ab91344b176eef2c59c67e"
COST = "results/experiment_8559_v735_complete_native_cost.json"
COST_PIN = "sha256:d08c9868dc613dd1b8f7ee7e53a2e6e5497d16a2526dcf7e85e13e35eff7862c"
VENDOR, LITERATURE = previous.VENDOR, previous.LITERATURE
MAPPING = dict(
    public_text_features="text_features",
    scaling="scaling",
    offset="raw_logit_offset",
    binding_residual="cubic_residual",
    normalization_and_python_fallback="normalization",
    typed_decision="threshold_fallback",
    serialization="response_serialization",
    startup="startup",
    unattributed_orchestration="unattributed_orchestration",
)


def history(root: Path) -> Json:
    """Keep each earlier seal and original date before admitting the V734 primary."""
    result = dict(previous.history(root))
    path = root / HISTORY
    try:
        if previous.old.reference(path)["sha256"] != HISTORY_PIN:
            raise ValueError("original_v734_board_pin")
        value = json.loads(path.read_bytes())
        if value["run_date"] != "20261015" or value["required_checks_passed"] is not True:
            raise ValueError("original_v734_date_or_checks")
        terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
        ref = Path(terminal["publication"]["sidecar_path"])
        if read_bound_sidecar(path, ref)["report"]["passed"] is not True:
            raise ValueError("original_v734_terminal")
        result["refs"] += [previous.old.reference(path), previous.old.reference(ref)]
        result["readers"].append(dict(path=str(path), sha256=HISTORY_PIN, run_date="20261015"))
    except (OSError, ValueError, KeyError, TypeError):
        result["qualified"] = False
        result["checks"].append(
            failure(
                path,
                "original_v734_board_seal",
                HISTORY_PIN,
                previous.old.reference(path)["sha256"] if path.is_file() else None,
                "exp8550-hardware-operation-boundary",
            )
        )
    return result


def reconcile(cells: list[Json]) -> list[Json]:
    """Retain every arm/batch/repeat and reject overlap or incomplete host accounting."""
    expected = {(a, b, r) for a in ["python", "native"] for b in [1, 8] for r in range(10)}
    if len(cells) != 40 or {(c["arm"], c["batch"], c["repeat"]) for c in cells} != expected:
        raise ValueError("reconciled_complete_costs")
    for cell in cells:
        seconds = cell["seconds"]
        valid = set(seconds) == set(MAPPING) and all(
            math.isfinite(s) and s >= 0 for s in seconds.values()
        )
        valid = valid and cell["overlap_s"] == 0 and cell["wall_s"] > 0
        valid = valid and math.isclose(
            sum(seconds.values()), cell["wall_s"], rel_tol=0, abs_tol=1e-12
        )
        if not valid:
            raise ValueError("reconciled_complete_costs")
    return cells


def costs(root: Path) -> Json:
    """The unchanged native-cost consumer cold-reduces original operands before any ceiling."""
    path = root / COST
    result: Json = dict(
        path=str(path),
        qualified=False,
        checks=[],
        refs=[],
        spans=[],
        cells=[],
        conditional_ceiling=None,
    )
    expected: Any
    observed: Any
    field, expected, observed = "complete_native_cost_ready_score", 1, None
    try:
        value = json.loads(path.read_bytes())
        result["refs"].append(previous.old.reference(path))
        for field, expected in dict(
            complete_native_cost_ready_score=1,
            required_checks_passed=True,
            flagged_adversarial=False,
            experiment_id=8559,
            task_id="exp8559-complete-native-cost",
            run_date="20261016",
        ).items():
            observed = value.get(field)
            if type(observed) is not type(expected) or observed != expected:
                raise ValueError(field)
        field, expected, observed = (
            "original_complete_cost_sha256",
            COST_PIN,
            previous.old.reference(path)["sha256"],
        )
        if observed != expected:
            raise ValueError(field)
        field, expected, observed = "bound_terminal_and_original_consumer", True, False
        terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
        side = Path(terminal["publication"]["sidecar_path"])
        if read_bound_sidecar(path, side)["report"]["passed"] is not True:
            raise ValueError(field)
        consumed = native.consume(path)
        field = "reconciled_complete_costs"
        cells = reconcile(consumed["operation_shares"])
        cursor, spans = 0, []
        for source, operation in MAPPING.items():
            ns = round(
                sum(c["seconds"][source] for c in cells if c["arm"] == "python" and c["batch"] == 1)
                * 1e9
            )
            spans.append(dict(operation=operation, start_ns=cursor, end_ns=cursor + ns))
            cursor += ns
        result.update(
            qualified=True,
            cells=cells,
            spans=spans,
            conditional_ceiling=dict(complete_clock_ns=cursor),
            scope=consumed["complete_boundary"],
            excluded_operations=consumed["excluded_operations"],
            aggregation="Python batch1 ten repeats; rounded aggregate nanoseconds; original seconds retained",
            original_cost_consumer_passed=True,
        )
        result["refs"] += [
            previous.old.reference(side),
            value["work_reference"],
            *value["raw_shard_hashes"],
        ]
    except (OSError, ValueError, KeyError, TypeError):
        result["checks"].append(
            failure(path, field, expected, observed, "exp8559-complete-native-cost")
        )
    return result


def ceilings(cost: Json) -> list[Json]:
    """Ideal removal of arithmetic still charges startup, features and all fixed host work."""
    rows = []
    for arm, batch in [(a, b) for a in ["python", "native"] for b in [1, 8]]:
        for subset in [ARITHMETIC, ["cubic_residual"]]:
            cells = [c for c in cost["cells"] if c["arm"] == arm and c["batch"] == batch]
            total = sum(c["wall_s"] for c in cells) if cost["qualified"] else None
            eligible = (
                sum(c["seconds"][s] for c in cells for s, op in MAPPING.items() if op in subset)
                if cost["qualified"]
                else None
            )
            fixed = total - eligible if total is not None else None
            gain = total / fixed if fixed else None
            rows.append(
                dict(
                    arm=arm,
                    batch=batch,
                    repeats=10,
                    independent_count=0,
                    candidate_operations=subset,
                    complete_clock_s=total,
                    offloaded_clock_s=eligible,
                    fixed_host_clock_s=fixed,
                    zero_cost_ceiling=gain,
                    measured_acceleration=None,
                    nfr01_boundary=("insufficient_for_NFR01" if gain < 10 else "conditional_only")
                    if gain is not None
                    else "unknown_missing_qualified_cost",
                    unknown_costs=dict(transfer=None, precision=None, update=None, sampling=None),
                    vendor_exclusions=VENDOR,
                    conditional=True,
                    missing_reason=None if cost["qualified"] else "Exp8559_not_qualified",
                )
            )
    return rows
