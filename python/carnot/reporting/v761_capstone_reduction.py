"""REQ-REPORT-8903: original operands constrain each branch-local scientific claim."""

from collections import Counter
import json
from pathlib import Path
from typing import Any

from carnot.reporting.v758_capstone_reduction import load

Json = dict[str, Any]


def reduce(value: Json) -> Json:
    """Rebuild the registered estimands without pooling unrelated statistical units."""
    n, rows = value["experiment_id"], value.get("rows", [])
    estimands: Json = {}
    streams, worlds, replicates, topologies = set(), set(), set(), set()
    retained = rows
    if n in {8894, 8897, 8899}:
        work = load(value["work_reference"])
        retained = work["rows"]
        worlds = {r["cluster_id"] for r in retained}
    if n == 8892:
        pilots = [load(r) for r in value["pilots"]]
        retained = [r for p in pilots for r in p["rows"]]
    if n == 8895:
        from carnot.reporting.signed_causal_streams_8895 import reduce as causal

        retained = causal([load(r) for r in value["stream_refs"]])["rows"]
        retained = sorted(retained, key=lambda r: (r["unit_id"], r["arm"]))
        rows = sorted(rows, key=lambda r: (r["unit_id"], r["arm"]))
    if n == 8898:
        from carnot.reporting.signed_benefit_audit_8898 import summarize

        source = load(dict(path=str(Path(value["raw_directory"]) / "prospective.json")))
        retained = source["rows"]
        estimands = summarize(retained)
    if n in {8892, 8895, 8898}:
        streams = {r["cluster_id"] for r in retained}
        topologies = {str(r["topology_family"]) for r in retained if "topology_family" in r}
    if n in {8893, 8896}:
        from carnot.reporting.shared_terminal_methods_8893 import ledgers
        from carnot.reporting.sequential_label_cost_8885 import summarize

        raw = load(value["measurement_manifest"])
        retained = raw["rows"]
        standalone, shared = ledgers(retained)
        if standalone != raw["standalone_costs"] or shared != raw["shared_union_costs"]:
            raise ValueError("label_ledger_drift")
        keyed = {
            (r["replicate"], r["arm"], r["seed"], r["group"], r["protocol"]): r for r in retained
        }
        for row in retained:
            accepted, errors = row["population_accepted"], row["population_errors"]
            risk = errors / accepted if accepted else None
            if row["total_labels"] != row["selection_labels"] + row["candidate_draws"] or row[
                "served_risk"
            ] != (risk if row["certificate_issued"] else None):
                raise ValueError("paid_labels_or_risk_drift")
            if row["protocol"] == "sequential":
                fixed = keyed[row["replicate"], row["arm"], row["seed"], row["group"], "fixed"]
                if (
                    row["scheduled_looks"][-1] != fixed["scheduled_looks"][-1]
                    or row["final_prefix_hash"] != fixed["final_prefix_hash"]
                    or fixed["served_accepted"] > row["served_accepted"]
                ):
                    raise ValueError("final_prefix_inclusion_drift")
        estimands = summarize(raw)
        safe = all(r["served_risk"] is None or r["served_risk"] <= 0.05 for r in retained)
        intervals = {r["contrast"]: r for r in estimands["paired_intervals"]}
        estimands.update(
            standalone_label_benefit_score=int(
                safe and intervals["learned_energy/label_savings"]["lower_99"] > 0
            ),
            shared_union_benefit_score=int(
                safe and intervals["shared_union/label_savings"]["lower_99"] > 0
            ),
        )
        estimands["label_savings_score"] = estimands["standalone_label_benefit_score"]
        replicates = {f"8893/rep/{r['replicate']}" for r in retained}
    if n == 8894:
        from carnot.reporting.buffered_clock_methods_8894 import reduce as clocks

        estimands = clocks(retained)
    if n == 8897:
        from carnot.reporting.source_bound_decisions_8897 import evaluate

        observations = [
            {
                k: r[k]
                for k in (
                    "record_decision",
                    "source_coverage_status",
                    "source_semantics_status",
                    "ledger_errors",
                    "budget_exhausted",
                )
            }
            for r in retained
        ]
        rebuilt = evaluate(work["worlds"], observations)
        if rebuilt["rows"] != retained or not all(rebuilt["controls"].values()):
            raise ValueError("source_record_boundary_drift")
        estimands = {
            k: dict.fromkeys(labels, 0) | dict(Counter(r[k] for r in retained))
            for k, labels in [
                ("record_decision", ["support", "refute", "insufficient"]),
                ("source_coverage_status", ["complete", "incomplete", "invalid", "unknown"]),
            ]
        }
        estimands["source_semantics_status"] = "unknown"
    if n == 8899:
        from carnot.reporting.buffered_cost_attribution_8899 import reduce as costs

        estimands = costs(retained)
    if n == 8901:
        from carnot.reporting.signed_adaptation_cost_8901 import reduce_costs

        work = load(value["primitive_reference"])
        retained = work["measurement"]["rows"]
        streams = {r["cluster_id"] for r in retained}
        estimands = dict(per_stream_results=reduce_costs(retained, work["setup_ns"]))
    if retained != rows:
        raise ValueError("retained_rows_drift")
    for key, observed in estimands.items():
        if key in value and value[key] != observed:
            raise ValueError("registered_estimand_drift:" + key)
    return dict(
        experiment_id=n,
        verdict_class=value.get("verdict_class", "blocked"),
        verifier_is_oracle=value.get("verifier_is_oracle", False),
        constructed_measurement_count=0,
        independent_constructed_unit_count=len(streams),
        constructed_unit_ids=sorted(streams),
        stream_ids=sorted(streams),
        topology_ids=sorted(topologies),
        world_ids=sorted(worlds),
        calibration_replicate_ids=sorted(replicates),
        natural_generalization_count=0,
        cpu_request_count=len(rows) if n in {8894, 8899} else 0,
        cpu_cost_window_count=len(rows) if n == 8901 else 0,
        registered_estimands=estimands,
        scheduled_row_status_counts=dict(Counter(r["status"] for r in rows)),
    )
