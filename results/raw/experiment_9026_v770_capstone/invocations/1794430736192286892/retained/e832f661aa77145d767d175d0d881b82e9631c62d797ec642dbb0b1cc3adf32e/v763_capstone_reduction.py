"""REQ-REPORT-8929: preserve each producer's assertions and statistical unit.

These are reductions of retained measurements, never new experiments or model
calls. Reusing the original reducers keeps their causal and durability checks.
"""

import json
from math import ceil
from pathlib import Path
from typing import Any

from carnot.reporting import bfs_error_pilot_8918 as pilot
from carnot.reporting import durable_cost_qualification_8926 as durable
from carnot.reporting import hardware_operation_budget_8928 as hardware
from carnot.reporting import inline_clock_methods_8921 as clock
from carnot.reporting import interpolated_benefit_audit_8925 as learning
from carnot.reporting import interpolated_update_methods_8919 as methods
from carnot.reporting import label_demand_methods_8920 as labels
from carnot.reporting import qwen_natural_canary_8924 as natural
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.v746_retained_evidence import load

Json = dict[str, Any]


def require(condition: bool, reason: str) -> None:
    """Reject changed arithmetic even when an attacker recomputes the outer hash."""
    if not condition:
        raise ValueError(reason)


def collection_plan(result: Json) -> Json:
    """The point estimate sets a proposal; Wilson uncertainty is not a guarantee."""
    passed = result["completed_count"] == 50 and result["n_errors"] >= 10
    rate, interval = result.get("error_rate"), result.get("error_interval")
    return dict(
        pilot_gate_passed=passed,
        recommended_questions=ceil(200 / rate * 1.30) if passed else None,
        uncertainty_question_range=[ceil(260 / interval["high"]), ceil(260 / interval["low"])]
        if passed
        else None,
        target_errors=200,
        margin=0.30,
        execute_now=False,
        uncertainty="95% Wilson rate interval; collection counts are planning estimates, not a predictive guarantee",
        split_unit="graph_seed",
        split_requirement="Freeze fresh graph seeds and disjoint train/calibration/test assignments before detector access; pilot graphs remain excluded",
        next_falsifiable_step="Propose a separately authorized fresh collection and a k sensitivity control"
        if passed
        else "Change future k or graph size, then run a separately authorized pilot",
        detector_training_permitted=False,
        labels="Final answers only; no step localization or natural-language transfer",
        arc_status="suspended",
    )


def compare(value: Json, result: Json, omitted: tuple[str, ...] = ()) -> None:
    """Original producer fields remain assertions, including nulls and failed controls."""
    for key, observed in result.items():
        if key in value and key not in omitted:
            require(value[key] == observed, "producer_reduction_drift:" + key)


def operand(value: Json, name: str) -> Json:
    """Read the original raw identity instead of a newer similarly named measurement."""
    return dict(load(next(r for r in value["raw_shard_hashes"] if Path(r["path"]).name == name)))


def reduce(evidence: dict[int, Json], raw: Path) -> Json:
    """Rebuild branch conjunctions without adding their independent denominators."""
    result: Json = {}
    if 8918 in evidence:
        value = evidence[8918]
        runtime = operand(value, "runtime.json")
        graphs = pilot.graph_manifest()
        require(value["graph_manifest"] == graphs, "graph_truth_drift")
        measured = pilot.reduce_answers(graphs, runtime["responses"])
        compare(value, measured)
        owned, live = pilot.qualification(
            runtime,
            value["validation_receipts"],
            value["owned_statement_coverage"],
            value["fixture_only"],
        )
        expected = pilot.gates(measured, live=live, owned=owned)
        require(
            expected[:3]
            == (
                value["pilot_ready_score"],
                value["pilot_error_gate_score"],
                value["qwen_runtime_ready_score"],
            ),
            "pilot_gate_drift",
        )
        require(value["model_invocation_counts"] == runtime["counts"], "pilot_runtime_binding")
        result["pilot"] = dict(
            measured,
            collection_plan=collection_plan(measured),
            pilot_ready_score=expected[0],
            pilot_error_gate_score=expected[1],
            qwen_runtime_ready_score=expected[2],
        )
    if 8919 in evidence:
        value = evidence[8919]
        measured = methods.feasibility(value["rows"], value["promotion_rows"])
        compare(value, measured)
        result["development"] = measured
    if 8920 in evidence:
        value = evidence[8920]
        measured = labels.qualify()
        compare(value, measured)
        costs = []
        for name, arms in measured["retained_control_operands"].items():
            demand, eager = arms["demand"], arms["eager"]
            costs.append(
                dict(
                    control=name,
                    demand_labels=demand["paid_labels"],
                    eager_labels=eager["paid_labels"],
                    saved_labels=eager["paid_labels"] - demand["paid_labels"],
                    terminal_equality=demand["policies"] == eager["policies"],
                )
            )
        result["labels"] = dict(
            control_cost_rows=costs,
            decision_equality_violations=measured["decision_equality_violations"],
            pathwise_inclusion_violations=measured["pathwise_inclusion_violations"],
            qualified=value["required_checks_passed"] is True
            and value["label_demand_ready_score"] == 1,
            scientific_evaluation="blocked: Exp8923 scheduler skip; no 40-draw savings result",
            independent_certificate_draws_executed=0,
            planned_draws=40,
            proof_hash=canonical_hash(measured),
            cheap_policy_evaluations=measured["cheap_policy_evaluations"],
            original_owned_failures=[r for r in value["validation_receipts"] if not r["passed"]],
        )
    if 8921 in evidence:
        value = evidence[8921]
        work = load(value["work_reference"])
        require(all(clock.authenticate_row(r) for r in work["rows"]), "observer_raw_clock_drift")
        measured = clock.reduce(work["rows"])
        compare(value, measured)
        result["observer"] = measured
    if 8922 in evidence and 8925 in evidence:
        measured = learning.reduce(evidence[8922])
        compare(evidence[8925], measured)
        result["learning"] = {
            k: v for k, v in measured.items() if k not in ("rows", "promotion_rows")
        }
        result["learning"]["rows"] = measured["paired_stream_rows"]
        result["learning"]["row_scope"] = (
            "Whole-stream paired estimates; original per-query rows and all failed promotions remain in the pinned producer and stream operands"
        )
    if 8924 in evidence:
        value = evidence[8924]
        measurement = operand(value, "measurement.json")
        runtime = measurement["runtime"]
        measured = natural.reduce(measurement["cases"], runtime["outputs"])
        compare(value, measured)
        require(value["model_invocation_counts"] == runtime["counts"], "natural_runtime_binding")
        result["natural"] = measured
    if 8926 in evidence:
        value = evidence[8926]
        measured = durable.reduce_retained(
            raw / "durable_reduction", value["preconditions_checked"]
        )
        compare(value, measured, ("process_receipt_references", "adaptation_cost_ready_score"))
        require(len(measured["rows"]) == 48, "durable_census")
        result["durable"] = {
            k: v
            for k, v in measured.items()
            if k
            not in ("durability_rows", "promotion_cost_rows", "feedback_cost_rows", "restart_rows")
        }
        result["durable"]["retained_detail_scope"] = (
            "Original durability, promotion, feedback and restart rows remain in the pinned producer; all were recomputed with original assertions"
        )
    if 8928 in evidence:
        source = {
            key: evidence[n]
            for key, n in [("causal", 8922), ("durable", 8926)]
            if n in evidence and evidence[n]["required_checks_passed"]
        }
        measured = hardware.reduce(source)
        compare(evidence[8928], measured)
        result["hardware"] = measured
    return result
