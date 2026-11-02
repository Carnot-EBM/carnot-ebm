"""REQ-REPORT-8626: fixed original operands retain their scientific limits."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from unittest.mock import patch

from coverage.parser import PythonParser

from carnot.reporting import arc_observer_8624 as arc
from carnot.reporting import v739_capstone_samples as historical
from carnot.reporting import v740_capstone_evidence as e

Json = dict[str, Any]


def producer(number: int) -> Json:
    """Read the one original producer rather than a copied capstone summary."""
    paths = list((e.ROOT / "results").glob(f"experiment_{number}_v740*.json"))
    if len(paths) != 1:
        raise ValueError("original_producer_identity")
    return dict(e.read(e.reference(paths[0])))


def reduce() -> Json:
    """Recompute fixed samples without training, serving or treating repeats as questions."""
    sealed = e.read(e.methods.protocol()["inherited_v739_protocol_reference"])
    with patch.object(historical.e.methods, "protocol", return_value=sealed):
        result = historical.reduce()
    learning = e.read(e.methods.protocol()["original_producers"]["8582"])
    ref = learning["released_feedback_reference"]
    if e.methods.sha256_file(Path(ref["path"])) != ref["sha256"]:
        raise ValueError("released_feedback_hash")
    feedback = json.loads(Path(ref["path"]).read_bytes())
    released = []
    for row in feedback[:2]:
        primitive = row["primitive"]
        original = e.read(primitive)["updates"][int(primitive["json_pointer"].rsplit("/", 1)[1])]
        p, y = original["probability_before_update"], original["target"]
        loss = -y * math.log(p) - (1 - y) * math.log1p(-p)
        released.append(
            dict(
                feedback_id=row["feedback_id"],
                recomputed_log_loss=loss,
                matches_original=math.isclose(loss, row["log_loss"], abs_tol=1e-15)
                and all(
                    row[k] == original[k]
                    for k in [
                        "feedback_id",
                        "target",
                        "original_target",
                        "before_head_hash",
                        "after_head_hash",
                        "parameter_write",
                    ]
                ),
            )
        )
    result["released_feedback"] = dict(
        source=ref,
        count=len(feedback),
        fixed_sample=feedback[:2],
        reductions=released,
        matches_original=all(r["matches_original"] for r in released),
        qualified=False,
        missing_current_reason="Exp8618 pre-gated; no qualified V740 learning history",
    )
    extraction = producer(8620)
    rows = extraction["extraction_rows"]
    usable = [r for r in rows if r["fixed_scores"] is not None and r["binary_target"] in (0, 1)]
    support = dict(
        intended_clusters=len({r["question_hash"] for r in rows}),
        usable_clusters=len({r["question_hash"] for r in usable}),
        usable_rows=len(usable),
        class_clusters=[
            len({r["question_hash"] for r in usable if r["binary_target"] == c}) for c in (0, 1)
        ],
        selective_coverage=len(usable) / len(rows),
    )
    result["triviaplus_coverage"] = dict(
        support,
        matches_original=support == extraction["usable_question_support"],
        source_faithfulness_only=True,
        factual_correctness_established=False,
        qwen_result=False,
        prior_exposure="unknown",
    )
    cost = producer(8622)
    clocks = []
    for arm in ["legacy_python", "legacy_native", "lean_python", "lean_native"]:
        row = next(
            r for r in cost["rows"] if r["arm"] == arm and r["length"] == 16 and r["batch"] == 1
        )
        primitive = e.read(row["primitive_reference"])
        spans = primitive["operations"]
        total = (primitive["end_ns"] - primitive["start_ns"]) / 1e9
        partition = sum(s["end_ns"] - s["start_ns"] for s in spans) / 1e9
        clocks.append(
            dict(
                arm=arm,
                unit_id=row["unit_id"],
                source=row["primitive_reference"],
                total_s=total,
                partition_s=partition,
                matches_original=total == partition == row["total_session_wall_s"]
                and all(a["end_ns"] == b["start_ns"] for a, b in zip(spans, spans[1:])),
            )
        )
    result["complete_lean_cost"] = dict(
        rows=clocks,
        primary_contrast=cost["primary_contrast"],
        lean_service_benefit_score=cost["lean_service_benefit_score"],
        nfr01_score=cost["nfr01_score"],
    )
    deadline = producer(8616)
    diagnosis = deadline["statement_diagnosis"]
    source = diagnosis["source_reference"]
    if e.methods.sha256_file(Path(source["path"])) != source["sha256"]:
        raise ValueError("deadline_source_hash")
    parser = PythonParser(filename=source["path"])
    parser.parse_source()
    raw = e.read(diagnosis["raw_coverage_reference"])
    runner = raw["files"]["python/carnot/reporting/causal_brier_runner_8582.py"]
    result["source_bound_deadline"] = dict(
        source=source,
        continuation94_first_line=parser.first_line(94),
        executable_statement_count=len(parser.statements),
        raw_executed_count=len(runner["executed_lines"]),
        raw_missing_lines=runner["missing_lines"],
        current_qualified=bool(deadline["learning_execution_ready_score"]),
        current_gate_failures=deadline["gate_check_summary"],
        historical_disqualification_preserved=True,
    )
    observer = producer(8624)
    primitive = e.read(e.read(observer["work_reference"])["primitive_reference"])
    windows: list[Json] = []
    for item in primitive["history"]["inputs"]:
        identity = {k: item["unit"][k] for k in ["game", "mode", "seed", "episode_id"]}
        windows.extend(
            dict(r, **identity) for r in arc.reduce_stream(item["steps"], item["events"])
        )
    result["current_arc_horizons"] = dict(
        rows=windows,
        matches_original=windows == observer["horizon_status_rows"],
        fixture_solve_credit=0,
        supervisor_priority_change=0,
    )
    return dict(result)
