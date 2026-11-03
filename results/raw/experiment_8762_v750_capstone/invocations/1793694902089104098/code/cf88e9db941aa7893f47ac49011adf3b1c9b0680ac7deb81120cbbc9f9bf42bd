"""REQ-REPORT-8722: reconstruct scientific conclusions from admitted original rows."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from carnot.reporting import v747_capstone_inputs as e
from carnot.reporting.installed_batch_cost_8717 import aggregate
from carnot.reporting.relational_capacity_8712 import upstream as relations
from carnot.reporting.verifier_completeness_8715 import oracle

Json = dict[str, Any]


def same(actual: Any, expected: Any, label: str) -> None:
    """A sealed summary must still agree with its original numerical operands."""
    if actual != expected:
        raise ValueError("scientific_reconstruction:" + label)


def reduce(work: Json) -> Json:
    """Use original thresholds and denominators; no execution or benefit is inferred from skips."""
    values = {
        str(r["experiment_id"]): e.load(r["primary_reference"])
        for r in work["slots"]
        if r["producer_qualified"]
    }
    result: Json = dict(
        learning=dict(
            measured=False,
            future_brier=None,
            joint_retention=None,
            reason="Exp8710/8711/8713/8714 producers did not execute",
        ),
        qwen=dict(
            measured=False,
            language_model_calls=0,
            bounded_call_budget=36,
            reason="No changed authenticated CUDA mechanism; Exp8718 skipped",
        ),
    )
    if "8712" in values:
        value = values["8712"]
        data = e.load(value["primitive_reference"])["measurement"]
        collisions = sum(
            r["left_values"] == r["right_values"] and {r["left_truth"], r["right_truth"]} == {0, 1}
            for r in data["collision_witnesses"]
        )
        symbolic = sum(
            r["decision"] == relations.evaluate(r["records"], r["query"])
            for r in data["symbolic_rows"]
        )
        same(symbolic, 1440, "symbolic_original_truth")
        same(bool(collisions), value["representation_sufficiency_refuted"], "lexical_capacity")
        result["relational"] = dict(
            collision_count=collisions,
            worlds=120,
            views=len(data["rows"]),
            symbolic_correct=symbolic,
            missing_probability_count=value["missing_probability_count"],
            verdict_class="circular_positive",
            independent_generalization=False,
        )
    if "8715" in values:
        value = values["8715"]
        probes = e.load(value["primitive_reference"])["measurement"]["probe_rows"]
        groups: dict[tuple[str, str], list[Json]] = defaultdict(list)
        for row in probes:
            same(oracle(row["facts"], row["query"]), row["expected"], "independent_probe_label")
            if row["partition"] == "sealed_evaluation" and row["variant"] != "correct":
                groups[row["world_id"], row["variant"]].append(row)
        baseline = sum(
            any(r["actual"] != r["expected"] for r in rows if r["suite"] == "baseline")
            for rows in groups.values()
        )
        extended = sum(any(r["actual"] != r["expected"] for r in rows) for rows in groups.values())
        same(baseline, value["baseline_detected_defect_count"], "baseline_detections")
        same(extended, value["extended_detected_defect_count"], "extended_detections")
        result["verifier"] = dict(
            sealed_faulty_units=len(groups),
            baseline_detected=baseline,
            extended_detected=extended,
            unresolved=len(groups) - extended,
            baseline_probes=sum(r["suite"] == "baseline" for rows in groups.values() for r in rows),
            extended_probes=sum(map(len, groups.values())),
            production_completeness=False,
            verdict_class="circular_positive",
        )
    if "8717" in values:
        value = values["8717"]
        rows = e.load(value["primitive_reference"])["measurement"]["process_rows"]
        batch = aggregate(rows)
        for key, actual in batch.items():
            if key != "complete":
                same(actual, value[key], "batch_" + key)
        result["batch"] = dict(
            batch,
            process_count=len(rows),
            batching_threshold="paired lower95 > 1.05 and native B16 p95 <= 50ms",
            same_batch_threshold="paired lower95 >= 10",
            nfr01_score=batch["same_batch_10x_score"],
        )
    if "8719" in values:
        value = values["8719"]
        primitive = e.load(value["primitive_reference"])
        rows = []
        for summary in primitive["summaries"]:
            ref = summary["rows_reference"]
            e.verify(ref)
            import json
            from pathlib import Path

            rows.extend(json.loads(line) for line in Path(ref["path"]).read_text().splitlines())
        same(
            len(rows),
            len(value["native_parity_rows"]) + len(value["boundary_rows"]),
            "native_inventory",
        )
        mismatches = sum(
            not (
                r["integer_state_identical"]
                and r["counters_identical"]
                and r["historical_identical"]
            )
            for r in rows
        )
        same(mismatches, 0, "compiled_integer_parity")
        result["compiled"] = dict(
            update_rows=len(rows),
            jobs=len(primitive["summaries"]),
            integer_state_mismatches=mismatches,
            restart_count=len(value["restart_rows"]),
            restart_parity=all(
                r["state_acknowledgement_and_bytes_identical"]
                and r["real_exit"] == 73
                and r["resume_exit"] == 0
                for r in value["restart_rows"]
            ),
            physical_fpga_execution=False,
            comparative_speed_measured=False,
            verdict_class="circular_positive",
        )
    if "8720" in values:
        value = values["8720"]
        primitive = e.load(value["primitive_reference"])
        same(primitive["new_event_rows"], value["new_event_rows"], "arc_new_identities")
        result["arc"] = dict(
            new_event_count=len(primitive["new_event_rows"]),
            frontier_unchanged=primitive["frontier_before"] == primitive["frontier_after"],
            closed_panel_replayed=False,
            policy_change_authorized=False,
            decision="close_empty_delta_valid_null",
        )
    if "8721" in values:
        value = values["8721"]
        result["hardware"] = {
            key: value[key]
            for key in [
                "board_status_rows",
                "operation_support_rows",
                "complete_cost_rows",
                "conditional_acceleration_bounds",
                "unknown_costs",
                "measurement_to_change_conclusion",
            ]
        }
    return result
