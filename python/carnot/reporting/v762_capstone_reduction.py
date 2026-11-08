"""REQ-REPORT-8916: rebuild scientific conjunctions from authenticated original operands."""

import json
from pathlib import Path
from typing import Any

from carnot.reporting.v746_retained_evidence import load
from carnot.reporting.v758_capstone_reduction import reduce as empty_reduction

Json = dict[str, Any]


def reduce(value: Json) -> Json:
    """Preserve branch-local statistical units and reject changed rows or registered claims."""
    n, rows = value["experiment_id"], value.get("rows", [])
    print(f"[exp8916] phase=reduction_{n}_before completed=0 pending={len(rows)}", flush=True)
    estimands: Json = {}
    retained = rows
    streams, replicates = set(), set()
    if n == 8905:
        from carnot.reporting.guarded_update_methods_8905 import feasibility

        ref = next(
            r for r in value["raw_shard_hashes"] if Path(r["path"]).name == "measurement.json"
        )
        measurement = load(ref)
        retained = measurement["rows"]
        estimands = feasibility(retained, measurement["promotion_rows"])
    if n == 8906:
        from carnot.reporting.finite_budget_methods_8906 import qualify

        proof = json.loads(json.dumps(qualify()))
        retained = proof["rows"]
        estimands = {k: v for k, v in proof.items() if k in value and k != "rows"}
    if n == 8907:
        work = load(value["work_reference"])
        retained = work["rows"]
        estimands = {k: v for k, v in work["result"].items() if k in value and k != "rows"}
    if n == 8908:
        from carnot.reporting.natural_criterion_runner_8908 import inspect

        primitive = load(value["primitive_reference"])
        rebuilt = inspect(Path(primitive["manifest"]["path"]))
        rebuilt.pop("private_pairs")
        retained = rebuilt["rows"]
        estimands = {k: v for k, v in rebuilt.items() if k in value and k != "rows"}
    if n == 8909:
        from carnot.reporting.guarded_benefit_audit_8911 import reconstruct

        retained = [
            r
            for ref, plan in zip(
                value["stream_refs"], value["protocol"]["frozen_evaluation_manifest"], strict=True
            )
            for r in reconstruct(load(ref), plan)
        ]
    if n == 8910:
        from carnot.reporting import finite_budget_evaluation_8910 as certificates

        raw = Path(value["measurement_manifest"]["path"]).parent
        operands = json.loads((raw / "operands.json").read_bytes())
        cohort, scores = load(operands["population"]), load(operands["scores"])["rows"]
        trace, truth = (
            load(operands["draws"]),
            json.loads((raw / "census_labels.json").read_bytes()),
        )
        selection = json.loads((raw / "selection.json").read_bytes())
        certified = certificates.certify(
            cohort,
            scores,
            trace,
            selection,
            lambda arm, rep, group, stage, requested: {
                u["unit_id"]: truth[u["unit_id"]] for u in requested
            },
        )
        measured = certificates.evaluate(cohort, scores, truth, trace, certified)
        retained = measured["rows"]
        estimands = {k: v for k, v in measured.items() if k in value and k != "rows"}
        replicates = {f"8906/rep/{r['replicate']}" for r in retained}
    if n == 8911:
        from carnot.reporting.guarded_benefit_audit_8911 import reduce as audit

        source = next(
            r
            for r in value["source_artifact_hashes"]
            if Path(r["path"]).name == "experiment_8909_v762_guarded_causal_streams.json"
        )
        measured = audit(load(source))
        retained = measured["rows"]
        estimands = {k: v for k, v in measured.items() if k in value and k != "rows"}
    if n == 8912:
        from carnot.reporting.complete_request_cost_8912 import reduce as costs, clock_rows

        work = load(value["work_reference"])
        retained = work["rows"]
        estimands = costs(retained)
        estimands["raw_clock_rows"] = clock_rows(work)
    if n == 8915:
        from carnot.reporting.hardware_operation_budget_8915 import (
            PRODUCERS,
            ROOT,
            reduce as hardware,
        )

        evidence = {
            branch: json.loads((ROOT / producer[0]).read_bytes())
            for branch, producer in PRODUCERS.items()
        }
        measured = hardware(evidence)
        retained = measured["rows"] + [
            dict(
                unit_id=f"gate/{i}",
                cluster_id=r["upstream"],
                arm="prerequisite",
                seed=7628915,
                denominator=1,
                metric=None,
                status="blocked",
                missing_reason=r["missing_reason"],
                source_gate=r,
            )
            for i, r in enumerate(value["gate_check_summary"])
        ]
        estimands = {k: v for k, v in measured.items() if k in value and k != "rows"}
    if n in {8905, 8909, 8911}:
        streams = {r["cluster_id"] for r in retained}
    if n == 8909:
        retained = sorted(retained, key=lambda r: (r["unit_id"], r["arm"]))
        rows = sorted(rows, key=lambda r: (r["unit_id"], r["arm"]))
    if retained != rows:
        raise ValueError("retained_rows_drift")
    for key, observed in estimands.items():
        if key in value and value[key] != observed:
            raise ValueError("registered_estimand_drift:" + key)
    result = empty_reduction(dict(experiment_id=n, rows=[]))
    result.update(
        verdict_class=value.get("verdict_class", "blocked"),
        verifier_is_oracle=value.get("verifier_is_oracle", False),
        registered_estimands={
            k: v
            for k, v in estimands.items()
            if k.endswith("_score")
            or k
            in {
                "development_rows",
                "proof_receipt_hash",
                "proof_obligations",
                "family_alpha_ledger",
                "paired_intervals",
                "multiplicity_ledger",
                "acceptance_gate_rows",
                "promotion_summary",
                "retention_summary",
                "calibration_summary",
                "continuation_decision",
                "disagreement_rows",
                "standalone_costs",
                "shared_union_costs",
                "worst_served_group_risk",
                "decision_equality_violations",
                "pathwise_inclusion_violations",
                "theoretical_family_error",
                "observer_overhead_intervals",
                "finite_nonnegative_closure",
                "physical_acceleration_demonstrated",
                "unknown_fraction_rows",
                "measured_fraction_rows",
                "amdahl_rows",
                "nfr_01",
            }
        },
        stream_ids=sorted(streams),
        calibration_replicate_ids=sorted(replicates),
        corpus_case_count=len({r["unit_id"] for r in retained if r["status"] == "completed"})
        if n == 8908
        else 0,
        cpu_request_count=len(retained) if n == 8912 else 0,
        constructed_measurement_count=len(retained) if n in {8905, 8909, 8910} else 0,
    )
    print(f"[exp8916] phase=reduction_{n}_after completed={len(retained)} pending=0", flush=True)
    return result
