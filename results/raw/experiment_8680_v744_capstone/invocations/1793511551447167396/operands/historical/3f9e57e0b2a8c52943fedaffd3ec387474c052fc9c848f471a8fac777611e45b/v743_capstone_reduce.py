"""REQ-REPORT-8666: cold arithmetic does not create a new scientific cohort."""

from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Any

from carnot.reporting import v743_capstone_inputs as e
from carnot.reporting import complete_source_8659 as source
from carnot.reporting import installed_cost_8661 as installed
from carnot.reporting import arc_reset_transfer_8664 as transfer
from carnot.reporting.hardware_operation_evidence_8665 import installed_cells
from carnot.reporting.current_validator_8580 import xml_nodes

Json = dict[str, Any]


def original_union(value: Json, refs: dict[str, Json]) -> Json:
    """Read actual JUnit nodes instead of crediting a producer's aggregate."""
    union = dict(value["original_node_union"])
    junit = union.get("junit_references", [])
    if not junit:
        junit = [
            e.reference(Path(arg.split("=", 1)[1]))
            for receipt in value["validation_receipts"]
            if receipt["name"].startswith("historical-")
            for arg in receipt.get("argv", [])
            if arg.startswith("--junitxml=")
        ]
    observed = []
    for ref in junit:
        e.verify(ref)
        refs[ref["path"]] = ref
        observed.extend(xml_nodes(Path(ref["path"])))
    expected = union.get("original_node_ids", union.get("original_nodes", []))
    union["independently_verified"] = (
        bool(expected)
        and len(observed) == len(set(observed))
        and sorted(observed) == sorted(expected)
    )
    union["independent_junit_node_ids"] = observed
    if not union["independently_verified"]:
        raise ValueError("original_junit_union")
    return union


def reduce(values: dict[str, Json], protocol: Json, raw: Path) -> tuple[Json, Json, list[Json]]:
    """Rebuild quantities from original predictions, frames and timestamps without training."""
    refs: dict[str, Json] = {}

    def load(ref: Json) -> Any:
        """Each consumed primitive enters the independently checked source closure."""
        e.verify(ref)
        refs[ref["path"]] = ref
        return json.loads(Path(ref["path"]).read_bytes())

    prior = load(protocol["inherited_v742_protocol_reference"])
    learning = load(prior["original_producers"]["8582"])
    predictions = load(learning["prediction_rows_reference"])["rows"]
    retention = load(learning["retention_rows_reference"])
    losses = [(r["probability"] - r["evaluator_target"]) ** 2 for r in predictions]
    if losses != [r["brier"] for r in predictions]:
        raise ValueError("historical_brier_reduction")
    maximum = max((r["probability"] - r["target"]) ** 2 - r["baseline_brier"] for r in retention)
    feedback = load(learning["released_feedback_reference"])
    feedback_losses = []
    for row in feedback:
        primitive = row["primitive"]
        update = load(primitive)["updates"][int(primitive["json_pointer"].rsplit("/", 1)[1])]
        p, y = update["probability_before_update"], update["target"]
        loss = -y * math.log(p) - (1 - y) * math.log1p(-p)
        if not math.isclose(loss, row["log_loss"], abs_tol=1e-15):
            raise ValueError("historical_feedback_reduction")
        feedback_losses.append(loss)
    partition = load(protocol["cohort_partition_reference"])
    core = load(values["8654"]["primitive_reference"])
    original = partition["original_node_ids"]
    passed = [r["node_id"] for r in core["outcomes"] if r["outcome"] == "passed"]
    missing = sorted(set(original) - set(passed))
    learning_audit = dict(
        original_node_ids=original,
        cohort_a_passed_node_ids=passed,
        cohort_b_passed_node_ids=[],
        missing_node_ids=missing,
        missing_count=len(missing),
        original_count=len(original),
        executed_count=len(passed),
        qualified=False,
        source_reuse_refusals=values["8654"]["source_reuse_refusals"],
        original_statement_union=values["8654"]["original_statement_union"],
        original_runner_node_ids=partition["original_runner_node_ids"],
        missing_reason="cohort_B_pre_gated_and_cohort_A_source_identity_changed",
    )
    e.progress("historical_learning_reduced", len(predictions), 0)
    natural = values["8659"]
    work = load(natural["primitive_reference"])
    data = source.inputs(e.ROOT)
    responses = []
    for index, record in enumerate(work["response_records"]):
        result: Json = dict(original=None, streaming=None, invariance=[], costs=[])
        for item in record["references"]:
            observed = load(item["reference"])
            if item["repeat"] < 3:
                name = f"r{index}-repeat{item['repeat']}-{item['arm']}-chunk32"
                receipt = next(r for r in work["measurement_receipts"] if r["name"] == name)
                result["costs"].append(
                    dict(
                        observed["cost"],
                        repeat=item["repeat"],
                        order_in_pair=int(
                            item["arm"]
                            != ("streaming_complete" if item["repeat"] == 1 else "original_capped")
                        ),
                        receipt=receipt,
                    )
                )
                if item["repeat"] == 0:
                    result["original" if item["arm"] == "original_capped" else "streaming"] = (
                        observed
                    )
            else:
                result["invariance"].append(observed)
            if observed.get("window_reference"):
                load_ref = observed["window_reference"]
                e.verify(load_ref)
                refs[load_ref["path"]] = load_ref
        responses.append(result)
        if (index + 1) % 20 == 0:
            e.progress("source_reduction", index + 1, 190 - index - 1)
    coverage = source.reduce(dict(data, responses=responses))
    if any(coverage[k] != natural[k] for k in coverage):
        raise ValueError("complete_source_reduction")
    cost = values["8661"]
    cost_work = load(cost["authority_reference"])
    rebuilt = installed.reduce(cost_work)
    if any(
        rebuilt[k] != cost[k]
        for k in ("per_session_rows", "complete_operation_spans", "paired_throughput_intervals")
    ):
        raise ValueError("installed_cost_reduction")
    cost_refs: list[Json] = []
    cells = installed_cells(cost, cost_refs)
    for ref in cost_refs:
        e.verify(ref)
        refs[ref["path"]] = ref
    reset_value = values["8664"]
    reset_work = load(reset_value["work_reference"])
    episodes = [load(ref) for ref in reset_work["references"]]
    reset = transfer.reduce(episodes)
    if any(reset[k] != reset_value[k] for k in reset):
        raise ValueError("reset_callback_reduction")
    historical = load(e.reference(e.ROOT / "results/experiment_8650_v742_arc_reset_transfer.json"))
    historical_work = load(historical["work_reference"])
    historical_errors = set()
    for ref in historical_work["references"]:
        historical_errors.update(load(ref)["recorder_errors"])
    reset_summary = dict(
        episode_count=len(episodes),
        authentic_reset_count=sum(r["authentic_origin_count"] for r in reset["per_game_results"]),
        missing_cause_count=len(reset["missing_cause_rows"]),
        recorder_error_count=len(reset["recorder_error_rows"]),
        callback_count=len(reset["callback_sequence_rows"]),
        complete_horizon_count=reset["complete_horizon_count"],
        qualified=bool(reset["reset_transfer_rows_ready_score"]),
        per_game_results=reset["per_game_results"],
        observer_issues=[
            dict(
                episode_id=ep["unit"]["episode_id"],
                count=len(ep["observer_issues"]),
                reasons=sorted({r["reason"] for r in ep["observer_issues"]}),
            )
            for ep in episodes
        ],
        historical_transfer_loaded=True,
        historical_missing_causes=len(historical["missing_cause_rows"]),
        historical_recorder_errors=sorted(historical_errors),
    )
    unions = dict(
        learning=learning_audit,
        streaming=original_union(values["8656"], refs),
        streaming_semantic_identity=values["8656"]["semantic_source_identity_rows"],
        streaming_driver_reference=values["8656"]["historical_driver_reference"],
        streaming_source_roles=values["8656"]["child_source_roles"],
        installed=original_union(values["8660"], refs),
        installed_cleanup=values["8660"]["cleanup_rows"],
        historical_cleanup=values["8660"]["historical_8646"],
    )
    summary = dict(
        learning=dict(
            current_producer_present=False,
            current_benefit=None,
            historical_brier_rows=len(losses),
            historical_brier_mean=sum(losses) / len(losses),
            historical_feedback_count=len(feedback_losses),
            historical_retention_maximum=maximum,
            qualified=False,
        ),
        coverage=dict(
            recovered_count=coverage["recovered_count"],
            response_count=len(coverage["per_response_results"]),
            question_clusters=coverage["question_cluster_intervals"]["question_clusters"],
            source_windows=sum(r["window_count"] for r in coverage["source_window_rows"]),
            common_parity=coverage["common_parity"],
            semantic_truth_established=False,
        ),
        installed=dict(
            session_count=len(rebuilt["per_session_rows"]),
            operation_cells=cells,
            paired_throughput_intervals=rebuilt["paired_throughput_intervals"],
            packaging_noninferiority_score=cost["packaging_noninferiority_score"],
            nfr01_score=cost["nfr01_score"],
        ),
        reset=reset_summary,
        hardware={
            k: values["8665"][k]
            for k in (
                "missing_cost_rows",
                "board_status_rows",
                "physical_hardware_ready_score",
                "complete_boundary_rows",
            )
        },
    )
    for value in values.values():
        e.closure(
            {
                k: value[k]
                for k in (
                    "code_config_hashes",
                    "semantic_source_identity_rows",
                    "child_source_roles",
                    "original_node_union",
                    "cleanup_rows",
                    "historical_8646",
                )
                if k in value
            },
            raw,
            refs,
        )
    e.closure(
        {
            k: protocol[k]
            for k in (
                "head_checkpoint_reference",
                "learning_data_reference",
                "cohort_partition_reference",
                "v743_activation_reference",
                "v743_design_reference",
            )
        },
        raw,
        refs,
    )
    return summary, unions, list(refs.values())
