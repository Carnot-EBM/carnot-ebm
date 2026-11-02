"""REQ-REPORT-8736: reduce per-unit evidence without treating absent science as zero."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting import equal_budget_verifier_8729 as verifier
from carnot.reporting import installed_cost_attribution_8733 as attribution
from carnot.reporting import v746_capstone_inputs as io
from carnot.reporting.hardware_cost_rows_8735 import cost_rows
from carnot.reporting.v747_capstone_reduce import same

Json = dict[str, Any]
HISTORY = dict(
    path=str(io.ROOT / "results/experiment_8722_v747_capstone.json"),
    sha256="sha256:5e1e8710b3747577f78704ad5fb30f80b6c86d4baeed1603615a5c7c02d22a9c",
)


def reduce(work: Json) -> Json:
    """Only qualified producers contribute; original worlds and processes keep their denominators."""
    values = {
        r["experiment_id"]: io.load(r["primary_reference"])
        for r in work["slots"]
        if r["producer_qualified"]
    }
    historical = io.load(HISTORY)["scientific_reconstruction"]
    result: Json = dict(
        learning=dict(
            measured=False,
            future_brier=None,
            joint_retention=None,
            reason="Exp8724 owned deadline smoke failed; Exp8725/8727/8728 did not execute",
        ),
        qwen=dict(
            measured=False,
            support_scores=None,
            missing_probabilities=True,
            language_model_calls=0,
            reason="Exp8730 skipped; authenticated CUDA repair absent",
        ),
        batch=historical["batch"],
        historical_v747=historical,
        resident_cost=dict(measured=False, speed_ratio=None),
        installed=dict(measured=False, ready=False),
    )
    if 8729 in values:
        value = values[8729]
        primitive = io.load(value["primitive_reference"])
        summary = verifier.summarize(primitive["frozen"], primitive["measurement"]["rows"])
        for key in ["arm_rows", "paired_world_rows", "paired_intervals", "probe_rows"]:
            same(summary[key], value[key], "equal_budget_" + key)
        result["verifier"] = dict(
            arm_rows=summary["arm_rows"],
            paired_intervals=summary["paired_intervals"],
            gain=summary["matched_call_gain"],
            verdict_class="circular_positive",
            independent_generalization=False,
            equal_wall_time_advantage=False,
            scope="120 exposed constructed worlds;30 evaluation clusters;5 probes per arm/variant",
        )
    if 8731 in values:
        value = values[8731]
        primitive = io.load(value["primitive_reference"])
        rows: list[Json] = []
        for summary in primitive["summaries"]:
            ref = summary["rows_reference"]
            io.verify(ref)
            rows.extend(json.loads(line) for line in Path(ref["path"]).read_text().splitlines())
        mismatches = sum(
            not (
                r["integer_state_identical"]
                and r["counters_identical"]
                and r["historical_identical"]
                and r["passed"]
            )
            for r in rows
        )
        same(
            len(rows),
            len(value["native_parity_rows"]) + len(value["boundary_rows"]),
            "resident_inventory",
        )
        same(mismatches, 0, "resident_exact_parity")
        same(primitive["restart_rows"], value["restart_rows"], "resident_restart_rows")
        result["compiled"] = dict(
            update_rows=len(rows),
            integer_state_mismatches=mismatches,
            restart_count=len(primitive["restart_rows"]),
            restart_parity=all(
                r["passed"]
                and r["real_exit"] == 73
                and r["resume_exit"] == 0
                and r["state_acknowledgement_and_bytes_identical"]
                for r in primitive["restart_rows"]
            ),
            physical_fpga_execution=False,
            comparative_speed_measured=False,
            verdict_class="circular_positive",
        )
    if 8732 in values:
        value = values[8732]
        primitive = io.load(value["primitive_reference"])
        probe = io.load(io.reference(Path(primitive["raw"]) / "durability_probe_rows.json"))
        latencies = [r["latency_ns"] for r in probe["rows"]]
        same(latencies, primitive["durability_probe"]["latencies_ns"], "durability_probe")
        projection = min(latencies) * 122880 / 1e9
        same(
            projection,
            primitive["durability_probe"]["projected_minimum_measurement_s"],
            "durability_projection",
        )
        same(primitive["process_rows"], value["process_rows"], "resident_processes")
        result["resident_cost"] = dict(
            measured=False,
            speed_ratio=None,
            probe_rows=probe["rows"],
            projected_minimum_measurement_s=projection,
            intended_process_count=len(primitive["roster"]),
            completed_process_count=len(primitive["process_rows"]),
            durability_boundary=primitive["durability_boundary"],
            reason="Complete durable roster exceeds3600 seconds; probe is not throughput",
        )
    if 8733 in values:
        value = values[8733]
        primitive = io.load(value["primitive_reference"])
        summary = attribution.aggregate(primitive["measurement"]["process_rows"])
        same(summary["overhead"], value["instrumentation_overhead_interval"], "installed_overhead")
        same(summary["fractions"], value["component_fraction_intervals"], "installed_fractions")
        result["installed"] = dict(
            summary,
            measured=True,
            measured_improvement=False,
            process_count=len(primitive["measurement"]["process_rows"]),
        )
    if 8734 in values:
        value = values[8734]
        primitive = io.load(value["primitive_reference"])
        same(primitive["new_event_rows"], value["new_event_rows"], "arc_delta")
        result["arc"] = dict(
            new_event_count=len(primitive["new_event_rows"]),
            frontier_unchanged=primitive["frontier_before"] == primitive["frontier_after"],
            policy_change_authorized=False,
            closed_panel_replayed=False,
        )
    if 8735 in values:
        value = values[8735]
        evidence = io.load(value["primitive_reference"])["evidence"]
        summary = cost_rows(evidence["branches"], [])
        for key in ["complete_cost_rows", "missing_cost_rows", "conditional_acceleration_bounds"]:
            same(summary[key], value[key], "hardware_" + key)
        result["hardware"] = dict(summary, physical_execution=False)
    return result
