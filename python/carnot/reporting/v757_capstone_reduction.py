"""REQ-REPORT-8852: reduce admitted rows without multiplying reused populations."""

import json
from pathlib import Path
from typing import Any

Json = dict[str, Any]


def reduce(value: Json) -> Json:
    """Recompute producer endpoints; raw operations and oracle fixtures remain distinct."""
    n = value["experiment_id"]
    rows = [r for r in value.get("rows", []) if r["status"] in {"completed", "complete"}]
    estimands: Json = {}
    streams, worlds, draws = set(), set(), set()
    support = 0
    if n in {8843, 8846}:
        streams = {r["cluster_id"] for r in rows}
    if n in {8842, 8844}:
        population = (
            value
            if n == 8842
            else json.loads(Path(value["source_artifact_hashes"][0]["path"]).read_bytes())
        )
        draw = json.loads(Path(population["draw_manifest"]["path"]).read_bytes())
        support = draw["population_size"]
        for rep in draw["replicates"]:
            for split in ("calibration", "evaluation"):
                for i, index in enumerate(rep[split]["indices"]):
                    draws.add(f"8842/{rep['replicate']}/{split}/{i}")
                    worlds.add(f"8842/population/{index}")
    if n == 8844:
        from carnot.reporting.selective_availability_8844 import summarize

        estimands = summarize(rows)
    if n == 8846:
        from carnot.reporting.future_benefit_audit_8846 import reduce as future_reduce

        operand = json.loads(Path(value["raw_shard_hashes"][0]["path"]).read_bytes())
        rebuilt = future_reduce(operand)
        if rebuilt["rows"] != rows:
            raise ValueError("future_rows_drift")
        estimands = {
            k: rebuilt[k]
            for k in [
                "next_head_recommendation",
                "paired_interval_rows",
                "retention_interval_rows",
                "causal_benefit_score",
                "addition_benefit_score",
                "retention_pass_score",
                "categorical_noninferiority_score",
            ]
        }
    if n == 8847:
        from carnot.reporting.indexed_cost_8847 import reduce as indexed_reduce

        estimands = indexed_reduce([r for r in value["rows"] if r["split"] == "evaluation"])
        estimands.pop("acceptance_gates")
        estimands["stress_results"] = {
            str(size): indexed_reduce([r for r in value["rows"] if r["size"] == size])
            for size in (16, 32, 64)
        }
    if n == 8848:
        from carnot.reporting.memory_size_cost_8848 import reduce_costs

        estimands = reduce_costs(value["rows"], value["preparation_cost_rows"])
    if n == 8851:
        estimands = {
            k: value[k]
            for k in [
                "board_operation_rows",
                "unsupported_operations",
                "physical_evidence_state",
                "unknown_costs",
                "acquisition_priority",
            ]
        }
    for key, result in estimands.items():
        if key in value and value[key] != result:
            raise ValueError("registered_estimand_drift:" + key)
    measured = n in {8843, 8844}
    return dict(
        experiment_id=n,
        constructed_measurement_count=len(rows) if measured else 0,
        independent_constructed_unit_count=len(streams),
        constructed_unit_ids=sorted(streams),
        stream_ids=sorted(streams),
        world_ids=sorted(worlds),
        draw_ids=sorted(draws),
        population_support=support,
        natural_generalization_count=0,
        cpu_cost_window_count=len(rows) if n == 8848 else 0,
        cpu_request_count=len(value.get("rows", [])) if n == 8847 else 0,
        exact_oracle_verdict="circular_positive" if value.get("verifier_is_oracle") else None,
        registered_estimands=estimands,
    )
