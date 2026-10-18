"""REQ-REPORT-8530: compare serialized state rather than treating an update as one change."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from carnot.reporting import brier_online_learning_8517 as historical
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.verify import objective_stream_8517 as stream

Json = dict[str, Any]
HISTORICAL_PIN = "sha256:cf39595a1d7ee1b5dff2212a0b51014eb46b079499cbe5c194d43d9de6def1fd"
read, reference = historical.read, historical.reference


def coordinate_row(
    before: Json, after: Json, writes: list[int], *, event: str, namespace: str
) -> Json:
    """Count assignments separately from state changes, including shared-scale effects.

    Repeated assignment indices count repeatedly as physical writes. A serialized
    coefficient counts once per event when its stored number changes. Multiplying
    by decay_scale gives the effective coefficient used by the predictor.
    """
    left, right = np.asarray(before["parameters"]), np.asarray(after["parameters"])
    if (
        left.shape != right.shape
        or left.ndim != 1
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
        or any(type(i) is not int or not 0 <= i < len(left) for i in writes)
        or not np.isfinite([before["decay_scale"], after["decay_scale"]]).all()
    ):
        raise ValueError("coordinate_contract")
    changed = np.flatnonzero(left != right).tolist()
    effective = np.flatnonzero(
        left * before["decay_scale"] != right * after["decay_scale"]
    ).tolist()
    return dict(
        event_id=event,
        before=deepcopy(before),
        after=deepcopy(after),
        assignment_indices=writes,
        applied_update_events=1,
        physical_coordinate_writes=len(writes),
        changed_coordinate_count=len(changed),
        changed_coordinates=[f"{namespace}:{i}" for i in changed],
        decay_scale_changes=int(before["decay_scale"] != after["decay_scale"]),
        effective_coefficient_changes=len(effective),
        namespace=namespace,
    )


def aggregate(rows: list[Json]) -> Json:
    """Distinct coordinates use the seed/arm namespace so different heads are not merged."""
    keys = [
        "applied_update_events",
        "physical_coordinate_writes",
        "changed_coordinate_count",
        "decay_scale_changes",
        "effective_coefficient_changes",
    ]
    return dict(
        {k: sum(r[k] for r in rows) for k in keys},
        unique_changed_coordinates=len({i for r in rows for i in r["changed_coordinates"]}),
    )


def historical_accounting() -> Json:
    """Replay original primitive updates without training a new scientific stream.

    The original steps seal before/after head hashes and final block states. The
    cost file independently records assignments from committed update calls.
    Neither the old parameter counter nor the reported summary determines changes.
    """
    path = historical.ROOT / "results" / (historical.NAME + ".json")
    value = read(dict(path=str(path), sha256=HISTORICAL_PIN))
    work = read(value["primitive_reference"])
    rows, checks, refs = [], [], [reference(path), value["primitive_reference"]]
    for cell in work["cells"]:
        cfg, costs = read(cell["configuration"]), read(cell["costs"])
        refs.extend([cell["configuration"], cell["costs"], cell["final"], *cell["steps"]])
        event_rows = []
        for step_ref in cell["steps"]:
            step = read(step_ref)
            head = deepcopy(step["sealed_state"]["head"])
            pending = step["sealed_state"]["pending"]
            for record, trace in zip(pending, step["updates"], strict=True):
                before = deepcopy(head)
                if cfg["frozen"]:
                    checks.append(before == step["after_state"]["head"])
                    continue
                kernel = stream._LOG_UPDATE if cell["arm"] == "log_loss" else stream.adapter.update
                head, info = kernel(
                    before, np.asarray(record["x"], dtype=float), trace["target"], 0.01
                )
                ids = stream.shipped.sparse.sparse_gradient(
                    before, np.asarray(record["x"]), trace["target"]
                )[0]
                row = coordinate_row(
                    before,
                    head,
                    [int(i) for i in ids],
                    event=trace["feedback_id"],
                    namespace=cell["identity"],
                )
                event_rows.append(row)
                checks.append(canonical_hash(before) == trace["before_head_hash"])
                checks.append(canonical_hash(head) == trace["after_head_hash"])
                checks.append(
                    (
                        np.asarray(head["parameters"]) * head["decay_scale"]
                        - np.asarray(before["parameters"]) * before["decay_scale"]
                    ).tolist()
                    == trace["coefficient_delta"]
                )
            checks.append(head == step["after_state"]["head"])
        checks.append(len(event_rows) == len(costs["updates"]))
        checks.extend(
            r["physical_coordinate_writes"] == c["coefficient_touches"]
            and r["decay_scale_changes"] == c["global_decay_writes"]
            for r, c in zip(event_rows, costs["updates"], strict=True)
        )
        rows.extend(event_rows)
        historical.progress(
            "accounted_original_cell",
            work["cells"].index(cell) + 1,
            20 - work["cells"].index(cell) - 1,
        )
    totals = aggregate(rows)
    return dict(
        totals,
        rows=rows,
        source_refs=refs,
        reconciliation_passed=bool(checks) and all(checks),
        metric_reconciliation_rows=[
            dict(
                metric="coefficient_change_count",
                old_value=value["coefficient_change_count"],
                corrected_value=totals["changed_coordinate_count"],
                old_semantics="applied update events incorrectly named coefficient changes",
                corrected_semantics="sum of changed serialized coordinates per committed event",
            ),
            dict(
                metric="label_updates",
                old_value=value["label_updates"],
                corrected_value=totals["applied_update_events"],
                corrected_semantics="committed numerical update events; not coordinate changes",
            ),
        ],
    )
