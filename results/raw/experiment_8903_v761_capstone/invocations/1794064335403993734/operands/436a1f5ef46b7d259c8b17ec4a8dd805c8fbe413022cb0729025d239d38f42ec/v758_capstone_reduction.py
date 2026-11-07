"""REQ-REPORT-8865: rebuild independent endpoints without adding reused populations."""

import json
from pathlib import Path
from typing import Any

Json = dict[str, Any]


def load(ref: Json) -> Json:
    """Read an already authenticated operand instead of a reported headline."""
    return dict(json.loads(Path(ref["path"]).read_bytes()))


def reduce(value: Json) -> Json:
    """Keep scientific endpoints, paid draws and CPU request clocks in separate units."""
    n, rows = value["experiment_id"], value.get("rows", [])
    estimands: Json = {}
    streams, worlds, draws = set(), set(), set()
    if n in {8856, 8860}:
        streams = {r["cluster_id"] for r in rows}
    if n in {8857, 8859}:
        trace = load(value["draw_manifest"])
        for rep in trace["replicates"]:
            for split in ("calibration", "evaluation"):
                for draw in rep[split]:
                    draws.add("8857/" + draw["draw_id"])
                    worlds.add("8857/population/" + str(draw["index"]))
    if n == 8859:
        from carnot.reporting.split_certificate_evaluation_8859 import summarize

        ref = next(r for r in value["raw_shard_hashes"] if Path(r["path"]).name == "census.json")
        raw_rows = load(ref)["rows"]
        if raw_rows != rows:
            raise ValueError("certificate_rows_drift")
        estimands = summarize(raw_rows)
    if n == 8860:
        from carnot.reporting.linear_benefit_audit_8860 import reduce as audit

        rebuilt = audit(load(value["raw_shard_hashes"][0]))
        if rebuilt["rows"] != rows:
            raise ValueError("learning_rows_drift")
        estimands = {k: v for k, v in rebuilt.items() if k != "rows"}
    if n == 8861:
        from carnot.reporting.installed_cost_8861 import reduce as cost

        work = load(value["work_reference"])
        if work["rows"] != rows:
            raise ValueError("request_rows_drift")
        estimands = cost(work["rows"], work["clocks"], work["installation_s"])
    if n == 8862:
        from carnot.reporting.linear_durable_cost_8862 import reduce_costs

        work = load(value["primitive_reference"])
        if work["window_rows"] != rows:
            raise ValueError("durable_rows_drift")
        estimands = reduce_costs(work["window_rows"], work["preparation_cost_rows"])
    for key, observed in estimands.items():
        if key in value and value[key] != observed:
            raise ValueError("registered_estimand_drift:" + key)
    return dict(
        experiment_id=n,
        constructed_measurement_count=len(rows) if n in {8856, 8859} else 0,
        independent_constructed_unit_count=len(streams),
        constructed_unit_ids=sorted(streams),
        stream_ids=sorted(streams),
        world_ids=sorted(worlds),
        draw_ids=sorted(draws),
        natural_generalization_count=0,
        cpu_request_count=len(rows) if n == 8861 else 0,
        cpu_cost_window_count=len(rows) if n == 8862 else 0,
        exact_oracle_verdict="circular_positive" if value.get("verifier_is_oracle") else None,
        registered_estimands=estimands,
    )
