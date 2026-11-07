"""REQ-REPORT-8890: recompute scoped endpoints without pooling different units."""

import json
from pathlib import Path
from typing import Any

from carnot.reporting.v758_capstone_reduction import load

Json = dict[str, Any]


def reduce(value: Json) -> Json:
    """Use retained operands and fixed estimators so producer headlines can be challenged."""
    n, rows = value["experiment_id"], value.get("rows", [])
    estimands: Json = {}
    streams, worlds, replicates = set(), set(), set()
    if n in {8882, 8883}:
        work = load(value["work_reference"])
        if work["rows"] != rows:
            raise ValueError("raw_rows_drift")
        worlds = {r["cluster_id"] for r in rows}
    if n == 8882:
        from carnot.reporting.request_cost_attribution_8882 import reduce as cost

        estimands = {
            k: v
            for k, v in cost(rows).items()
            if k not in {"component_rows", "uninstrumented_rows"}
        }
    if n == 8883:
        from carnot.reporting.evidence_loss_boundary_8883 import summarize

        for row in rows:
            action = row["action"]
            expected = {"support": "accept", "refute": "reject", "insufficient": "escalate"}[
                row["visible_view_truth"]
            ]
            wrong = int(action == "accept" and row["full_source_truth"] != "support")
            flags = dict(
                wrong_source_acceptance=wrong,
                metric=wrong,
                view_semantic_error=int(action != "escalate" and action != expected),
                decisive_coverage=int(action != "escalate"),
                abstention=int(action == "escalate"),
            )
            if any(row[k] != v for k, v in flags.items()):
                raise ValueError("boundary_flags_drift")
        estimands = summarize(rows)
    if n in {8881, 8884, 8886}:
        streams = {r["cluster_id"] for r in rows}
    if n == 8881:
        from carnot.reporting.topology_memory_methods_8881 import summarize

        retained = [load(ref) for ref in value["release_manifest"]["stream_refs"]]
        if [r for s in retained for r in s["rows"]] != rows:
            raise ValueError("stream_rows_drift")
        estimands = dict(summary=summarize(rows))
    if n == 8884:
        from carnot.reporting.topology_causal_streams_8884 import reduce_streams

        raw = Path(value["raw_directory"])
        retained = [
            json.loads(p.read_bytes()) for p in sorted(raw.glob("stream_v760-stream-*.json"))
        ]
        rebuilt = reduce_streams(retained)
        if sorted(rebuilt["rows"], key=lambda r: (r["unit_id"], r["arm"])) != sorted(
            rows, key=lambda r: (r["unit_id"], r["arm"])
        ):
            raise ValueError("stream_rows_drift")
        estimands = {
            k: v
            for k, v in rebuilt.items()
            if k
            in {"descriptive_effects", "maximum_pending", "label_call_counts", "transaction_costs"}
        }
    if n == 8886:
        from carnot.reporting.label_benefit_audit_8886 import reduce as audit
        from carnot.reporting.label_benefit_audit_8886 import SOURCE

        source = json.loads((Path(value["raw_directory"]) / (SOURCE + ".json")).read_bytes())
        rebuilt = audit(source)
        if rebuilt["rows"] != rows:
            raise ValueError("audit_rows_drift")
        estimands = {k: v for k, v in rebuilt.items() if k != "rows"}
    if n in {8880, 8885}:
        replicates = {f"8880/rep/{r['replicate']}" for r in rows}
    if n == 8880:
        ref = next(
            r for r in value["raw_shard_hashes"] if Path(r["path"]).name == "certificates.json"
        )
        if load(ref)["rows"] != rows:
            raise ValueError("certificate_rows_drift")
        population = load(value["population_manifest"])
        worlds = {r["cluster_id"] for r in population["inputs"] if r["split"] == "population"}
    if n == 8885:
        from carnot.reporting.sequential_label_cost_8885 import summarize

        raw_result = load(value["measurement_manifest"])
        if raw_result["rows"] != rows:
            raise ValueError("certificate_rows_drift")
        for row in rows:
            if (
                row["total_labels"] != row["selection_labels"] + row["candidate_draws"]
                or row["metric"] != row["total_labels"]
            ):
                raise ValueError("label_unit_rows_drift")
        for cost_row in raw_result["standalone_costs"]:
            for protocol in ("sequential", "fixed"):
                observed_cost = sum(
                    r["total_labels"]
                    for r in rows
                    if (r["replicate"], r["arm"], r["seed"], r["protocol"])
                    == (cost_row["replicate"], cost_row["arm"], cost_row["seed"], protocol)
                )
                if observed_cost != cost_row[protocol]:
                    raise ValueError("label_cost_rows_drift")
        for union in raw_result["shared_union_costs"]:
            local = [
                r
                for r in rows
                if r["replicate"] == union["replicate"] and r["protocol"] == "sequential"
            ]
            terminal = {
                g: max(r["candidate_draws"] for r in local if r["group"] == g)
                for g in {r["group"] for r in local}
            }
            if (
                terminal != union["sequential_terminal_draws"]
                or 960 + sum(terminal.values()) != union["sequential_total_labels"]
            ):
                raise ValueError("label_union_rows_drift")
        for row in rows:
            accepted, errors = row["population_accepted"], row["population_errors"]
            risk = errors / accepted if accepted else None
            if row["population_risk"] != risk or row["served_risk"] != (
                risk if row["certificate_issued"] else None
            ):
                raise ValueError("served_risk_rows_drift")
        estimands = summarize(raw_result)
    if n == 8888:
        from carnot.reporting.matched_adaptation_cost_8888 import reduce_costs

        work = load(value["primitive_reference"])
        if work["measurement"]["rows"] != rows:
            raise ValueError("cost_rows_drift")
        estimands = dict(per_stream_results=reduce_costs(rows, work["setup_ns"]))
    for key, observed in estimands.items():
        if key in value and value[key] != observed:
            raise ValueError("registered_estimand_drift:" + key)
    return dict(
        experiment_id=n,
        verdict_class=value.get("verdict_class", "blocked"),
        verifier_is_oracle=value.get("verifier_is_oracle", False),
        constructed_measurement_count=len(rows) if n in {8880, 8881, 8883, 8884, 8885} else 0,
        independent_constructed_unit_count=len(streams),
        constructed_unit_ids=sorted(streams),
        stream_ids=sorted(streams),
        world_ids=sorted(worlds),
        calibration_replicate_ids=sorted(replicates),
        natural_generalization_count=0,
        cpu_request_count=len(rows) if n == 8882 else 0,
        cpu_cost_window_count=len(rows) if n == 8888 else 0,
        registered_estimands=estimands,
        scheduled_row_status_counts={
            s: sum(r["status"] == s for r in rows) for s in {r["status"] for r in rows}
        },
        complete_cost_accounting={
            k: work[k] for k in ("installation_s", "disposal_s", "clocks") if k in work
        }
        if n == 8882
        else {},
    )
