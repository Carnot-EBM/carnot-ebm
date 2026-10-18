"""REQ-REPORT-8247: retain finished outcomes without upgrading failed science.

Saved source bytes preserve every original arm and missing mask. The capstone
counts reconciled tasks separately from producer execution and measured benefit.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import yaml

from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar, validate_primary
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v711_capstone_evidence import bind as bind, operand as operand
from carnot.verify import delayed_benefit_audit_8241 as delayed
from carnot.verify import margin_decision_audit_8239 as margin

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8247_v712_capstone"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_v712_capstone_8247.py"
MILESTONE = "2026.10.712"
DESIGN = "openspec/change-proposals/research-roadmap-vNEXT.md"
HISTORY = "results/experiment_8233_v711_capstone.json"
OWNED = [
    "python/carnot/reporting/v712_capstone_evidence.py",
    "python/carnot/reporting/v712_capstone.py",
    CLI,
]
COUNTS = ["intended_count", "completed_count", "failed_count", "censored_count", "excluded_count"]
AUDITS = {
    8239: "scripts/experiments/experiment_8239_v712_margin_decision_audit.py",
    8241: "scripts/experiments/experiment_8241_v712_delayed_benefit_audit.py",
    8242: "scripts/experiments/experiment_8242_v712_independent_concurrent_service.py",
    8243: "scripts/experiments/experiment_8243_v712_arc_supervisor_frontier.py",
    8244: "scripts/experiments/experiment_8244_v712_kv260_decision_boundary.py",
    8246: "scripts/experiments/experiment_8246_v712_gatemate_change_ledger.py",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real boundaries so a silent child cannot hide pending work."""
    print(f"[exp8247] phase={phase} completed={completed} pending={pending}", flush=True)


def resolve(task: Json, identity: int, root: Path) -> Path:
    """Alternate filenames authorize only the scheduled conductor gate receipt."""
    declared = root / str(task["deliverable"])
    if declared.is_file():
        return declared
    for path in sorted((root / "results").glob(f"experiment_{identity}_*.json")):
        try:
            value = json.loads(path.read_bytes())
            gates = value["gates_evaluated"]
            if (
                value.get("experiment") == identity
                and value.get("task_id", task["id"]) == task["id"]
                and value.get("schema") == "blocked_gate_check_v1"
                and value.get("blocked_at_layer") == "conductor_pre_gate"
                and gates
                and all(
                    any(
                        g["upstream"] == s["upstream"]
                        and g["artifact_field"] == s["artifact_field"]
                        and g["op"] == s["op"]
                        and g["expected"] == s["value"]
                        for s in task.get("gated_on", [])
                    )
                    for g in gates
                )
            ):
                return path
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return declared


def disposition(task: Json, identity: int, root: Path, raw: Path, work: Json) -> Json:
    """Authentication failures leave a missing disposition, never fabricated rows."""
    path = resolve(task, identity, root)
    row: Json = dict(
        experiment_id=identity,
        task_id=task["id"],
        unit_id=task["id"],
        arm="task_disposition",
        condition="terminal_accounting",
        metric="authenticated_disposition",
        numerator=0,
        denominator=1,
        status="completed",
        completed=True,
        missing=False,
        producer_executed=False,
        eligible=False,
        excluded=True,
        failed=False,
        censored=False,
        path=str(path),
        sha256=None,
        source_counts={k: None for k in COUNTS},
    )
    try:
        copy = bind(path, raw / "custody", work["references"])
        row["sha256"] = sha256_file(copy)
        value = json.loads(copy.read_bytes())
        if not isinstance(value, dict):
            raise ValueError("object_schema")
        if value.get("schema") == "blocked_gate_check_v1":
            if (
                value.get("experiment") != identity
                or value.get("task_id", task["id"]) != task["id"]
                or value.get("blocked_at_layer") != "conductor_pre_gate"
                or not value.get("gates_evaluated")
                or not all(
                    any(
                        g["upstream"] == s["upstream"]
                        and g["artifact_field"] == s["artifact_field"]
                        and g["op"] == s["op"]
                        and g["expected"] == s["value"]
                        for s in task.get("gated_on", [])
                    )
                    for g in value["gates_evaluated"]
                )
            ):
                raise ValueError("conductor_identity")
            row.update(
                disposition="conductor_pre_gate",
                verdict_class="blocked",
                honest_verdict="complete_blocked_conductor_pre_gate",
                numerator=1,
            )
            for g in value["gates_evaluated"]:
                if not g["passed"]:
                    work["failures"].append(
                        operand(
                            g["upstream"],
                            g["artifact_path"],
                            g["artifact_sha256"],
                            g["artifact_field"],
                            g["expected"],
                            g["actual"],
                            g["op"],
                        )
                    )
        else:
            validate_primary(value, path)
            if (
                value["task_id"] != task["id"]
                or not isinstance(value.get("rows"), list)
                or not all(type(value.get(k)) is int for k in COUNTS)
                or not all(
                    type(value.get(k)) is bool
                    for k in ["required_checks_passed", "flagged_adversarial"]
                )
            ):
                raise ValueError("primary_schema_or_task_identity")
            side = path.parent / "raw" / path.stem / "validators" / (row["sha256"][7:] + ".json")
            report = read_bound_sidecar(path, side)
            bind(side, raw / "custody", work["references"])
            if report["report"]["passed"] is not True:
                raise ValueError("terminal_report_failed")
            eligible = (
                value["required_checks_passed"]
                and not value["flagged_adversarial"]
                and value["verdict_class"] in {"positive", "circular_positive", "null"}
            )
            row.update(
                disposition="producer_terminal",
                producer_executed=True,
                numerator=1,
                honest_verdict=value["honest_verdict"],
                verdict_class=value["verdict_class"],
                eligible=eligible,
                excluded=not eligible,
                failed=value["verdict_class"] == "disqualified",
                source_counts={k: value[k] for k in COUNTS},
                source_rows_reference=dict(
                    path=str(copy),
                    sha256=row["sha256"],
                    artifact_field="rows",
                    row_count=len(value["rows"]),
                ),
            )
            work["primaries"][task["id"]] = value
            for check in value.get("gate_check_summary", []):
                if isinstance(check, dict) and check.get("passed") is False:
                    work["failures"].append(
                        operand(
                            task["id"],
                            check.get("path", str(path)),
                            check.get("hash", row["sha256"]),
                            check["artifact_field"],
                            check["expected"],
                            check["observed"],
                            check.get("op", "=="),
                        )
                    )
            if not eligible:
                work["failures"].append(
                    operand(
                        task["id"],
                        str(path),
                        row["sha256"],
                        "qualified_current_evidence",
                        True,
                        dict(
                            verdict_class=value["verdict_class"],
                            required_checks_passed=value["required_checks_passed"],
                        ),
                    )
                )
    except (OSError, ValueError, KeyError, TypeError) as error:
        row.update(
            missing=True,
            disposition="unavailable_output",
            verdict_class="blocked",
            honest_verdict="complete_blocked_upstream_evidence",
        )
        work["failures"].append(
            operand(
                task["id"],
                str(path),
                row["sha256"],
                "authenticated_current_output",
                True,
                str(error),
            )
        )
    return row


def science(work: Json, raw: Path) -> None:
    """Rebuild current costs from evaluator primitives rather than headline fields."""
    for index, name in [(5, "H1"), (7, "H2")]:
        row = work["dispositions"][index]
        progress("before_benchmark_" + name, 0, 1)
        try:
            value = work["primaries"][row["task_id"]]
            if not row["eligible"]:
                raise ValueError("eligible_current_audit_required")
            if name == "H1":
                ref = value["measurement_reference"]
                copy = bind(Path(ref["path"]), raw / "custody", work["references"], ref["sha256"])
                result = margin.reduce(json.loads(copy.read_bytes())["evidence"])
                if result["rows"] != value["rows"] or result["H1"] != value["H1"]:
                    raise ValueError("H1_primitive_drift")
            else:
                inputs = (
                    Path(value["terminal_validation_sidecar_path"]).parent / "audit_inputs.json"
                )
                copy = bind(inputs, raw / "custody", work["references"])
                data = json.loads(copy.read_bytes())
                for role in ["stream", "retention"]:
                    bind(Path(data[role + "_label_path"]), raw / "custody", work["references"])
                rebuilt = delayed.reconstruct(data, raw / "h2_reconstruction")
                result = delayed.statistics(rebuilt["rows"])
                if rebuilt["rows"] != value["rows"] or result != value["H2"]:
                    raise ValueError("H2_primitive_drift")
            work[name] = result
        except (OSError, ValueError, KeyError, TypeError) as error:
            work[name] = {}
            work["failures"].append(
                operand(
                    row["task_id"],
                    row["path"],
                    row["sha256"],
                    name + "_qualified_primitives",
                    True,
                    str(error),
                )
            )
        progress("after_benchmark_" + name, 1, 0)


def measure(root: Path, raw: Path) -> Json:
    """Bind activation and history before enumerating every current disposition."""
    work: Json = dict(
        tasks=[],
        references=[],
        primaries={},
        dispositions=[],
        failures=[],
        H1={},
        H2={},
        consumer_manifest=[],
        history={},
        activation_snapshot={},
    )
    design = bind(root / DESIGN, raw / "custody", work["references"])
    _, tasks = parse_design(design.read_text(), milestone=MILESTONE)
    if len(tasks) != 14 or [t["id"].split("-")[0] for t in tasks] != [
        f"exp{i}" for i in range(8234, 8248)
    ]:
        raise ValueError("exact_fourteen_task_contract")
    work["tasks"] = tasks
    for name in ["research-roadmap.yaml", HISTORY]:
        path = root / name
        try:
            copy = bind(path, raw / "custody", work["references"])
            value = (
                yaml.safe_load(copy.read_bytes())
                if name.endswith(".yaml")
                else json.loads(copy.read_bytes())
            )
            if name.endswith(".yaml"):
                if value["tasks"] != tasks or value["milestone"] != MILESTONE:
                    raise ValueError("full_active_task_object_drift")
                work["activation_snapshot"] = work["references"][-1]
            else:
                validate_primary(value, path)
                work["history"] = dict(
                    path=str(path),
                    sha256=sha256_file(copy),
                    rows=value["rows"],
                    actual_executed_task_count=value["actual_executed_task_count"],
                    conductor_pre_gate_count=value["conductor_pre_gate_count"],
                    required_checks_passed=value["required_checks_passed"],
                    upstream_validation_failures=value.get("upstream_validation_failures", []),
                )
        except (OSError, ValueError, KeyError, TypeError) as error:
            work["failures"].append(
                operand(
                    "activation_or_history",
                    str(path),
                    None,
                    "authenticated_authority_or_history",
                    True,
                    str(error),
                )
            )
    for index, task in enumerate(tasks[:-1]):
        progress("before_input", index, 13 - index)
        row = disposition(task, 8234 + index, root, raw, work)
        work["dispositions"].append(row)
        work["consumer_manifest"].append(
            dict(task_id=task["id"], path=row["path"], sha256=row["sha256"])
        )
        progress("after_input", index + 1, 12 - index)
    science(work, raw)
    return work


def reduce(work: Json, receipts: list[Json]) -> Json:
    """Owned completion grants readiness while qualified nulls remain terminal."""
    if len(work["dispositions"]) != 13 or len(work["tasks"]) != 14:
        raise ValueError("fourteen_disposition_roster")
    rows = deepcopy(work["dispositions"])
    passed = bool(receipts) and all(r["passed"] and r.get("normal_exit", True) for r in receipts)
    signal = bool(
        work["H1"].get("H1", {}).get("passed")
        or (work["H2"].get("passed") and work["H2"].get("retention_passed"))
    )
    kind = (
        "disqualified"
        if not passed
        else "blocked"
        if work["failures"]
        else "circular_positive"
        if signal
        else "null"
    )
    verdict = (
        "complete_"
        + kind
        + (
            "_owned_validation"
            if not passed
            else "_upstream_evidence"
            if work["failures"]
            else "_v712_capstone"
        )
    )
    rows.append(
        dict(
            experiment_id=8247,
            task_id=work["tasks"][-1]["id"],
            unit_id=work["tasks"][-1]["id"],
            arm="task_disposition",
            condition="terminal_accounting",
            metric="owned_execution_readiness",
            numerator=int(passed),
            denominator=1,
            status="completed",
            completed=True,
            missing=False,
            producer_executed=True,
            eligible=passed,
            excluded=not passed,
            failed=not passed,
            censored=False,
            honest_verdict=verdict,
            verdict_class=kind,
            disposition="self_owned_completion",
        )
    )

    def primary(index: int) -> Json:
        return dict(work["primaries"].get(work["tasks"][index]["id"], {}))

    h1, h2 = work["H1"], work["H2"]
    signals = [
        int(bool(h1.get("H1", {}).get("passed"))),
        int(bool(h2.get("passed") and h2.get("retention_passed"))),
    ]
    qualified = [
        all(
            r["passed"]
            for r in work.get("branch_audit_receipts", [])
            if r["name"] == "upstream_audit_" + str(identity)
        )
        for identity in [8239, 8241]
    ]
    signals = [score if qualified[i] else 0 for i, score in enumerate(signals)]
    branches = [
        dict(
            status="completed_signal"
            if signals[i]
            else "completed_null"
            if stats and qualified[i]
            else "blocked",
            audit_qualified=qualified[i],
            alpha=0.025,
            statistics=stats or None,
            intended_count=[128, 192][i],
        )
        for i, stats in enumerate([h1, h2])
    ]
    branches[1]["retention_intended_count"] = 64
    service, arc = primary(8), primary(9)
    return dict(
        honest_verdict=verdict,
        verdict_class=kind,
        rows=rows,
        task_dispositions=rows,
        intended_count=14,
        completed_count=14,
        failed_count=sum(r["failed"] for r in rows),
        censored_count=0,
        excluded_count=sum(r["excluded"] for r in rows),
        independent_count=0,
        actual_executed_task_count=sum(r["producer_executed"] for r in rows),
        upstream_executed_task_count=sum(r["producer_executed"] for r in rows[:-1]),
        conductor_pre_gate_count=sum(r["disposition"] == "conductor_pre_gate" for r in rows),
        missing_output_count=sum(r["missing"] for r in rows),
        capstone_execution_ready_score=int(passed),
        science_ready_score=0,
        required_checks_passed=passed,
        flagged_adversarial=False,
        H1=branches[0],
        H2=branches[1],
        h1_development_signal_score=signals[0],
        h2_development_signal_score=signals[1],
        verifier_is_oracle=True,
        exposure_scope="exposed development; upstream cached candidates",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        multiplicity=dict(alpha_per_hypothesis=dict(H1=0.025, H2=0.025), alpha_transfer=False),
        gate_check_summary=work["failures"],
        task_contract=work["tasks"],
        canonical_tasks_sha256=canonical_hash(work["tasks"]),
        activation_snapshot=work["activation_snapshot"],
        historical_v711=work["history"],
        validation_repairs=[
            dict(
                experiment_id=8235 + i,
                scientific_benefit_measured=False,
                qualification=primary(i + 1).get("honest_verdict"),
                historical_results_rewritten=False,
            )
            for i in range(2)
        ],
        three_prd_gaps=[
            dict(
                gap=name,
                requirements=req,
                closed=False,
                moved=bool(signals[i]) if i < 2 else False,
                remaining=remaining,
            )
            for i, (name, req, remaining) in enumerate(
                [
                    (
                        "useful_verified_decisions",
                        ["FR-06", "FR-12"],
                        "Different extraction or independently sourced cohort with decision benefit.",
                    ),
                    (
                        "later_learning_and_retention",
                        ["FR-11"],
                        "Later decision gain and independent retention under unchanged delayed protocol.",
                    ),
                    (
                        "request_scale_deployment",
                        ["FR-05", "FR-08", "NFR-01"],
                        "More independent cold-inclusive sweeps and matched Rust/Python10x.",
                    ),
                ]
            )
        ],
        board_obligations=[
            dict(
                board=board,
                experiment_id=8234 + i,
                path=rows[i]["path"],
                sha256=rows[i]["sha256"],
                honest_verdict=rows[i]["honest_verdict"],
                verdict_class=rows[i]["verdict_class"],
                qualified=rows[i]["eligible"],
                current_board_execution=primary(i).get("current_device_execution_count", 0),
                benefit_score=0,
                obligation=primary(i).get(key, {}),
            )
            for i, board, key in [
                (10, "KV260", "kv260_obligation"),
                (11, "PolarFire", "polarfire_obligation"),
                (12, "GateMate", "gatemate_obligation"),
            ]
        ],
        retirements=[
            dict(
                task_id=work["tasks"][5]["id"],
                scope="exact margin-only objective on exposed cohort",
                decision="retire" if h1 and not signals[0] else "await_named_evidence",
                reopening_condition="Materially different extraction signal or independently sourced cohort; coefficient and threshold renames do not reopen.",
            ),
            dict(
                task_id=work["tasks"][7]["id"],
                scope="unchanged delayed protocol; separate from static mechanism",
                decision="terminal_null" if h2 and not signals[1] else "await_named_evidence",
                reopening_condition="New eligible later decision gain and sealed retention evidence under the unchanged protocol.",
            ),
        ],
        request_accounting=dict(
            intended_generation_calls=96,
            observed_measurement_calls=service.get("completed_count"),
            paired_source_count=service.get("paired_source_count"),
            valid_completion_counts=service.get("valid_completion_counts"),
            cold_costs=service.get("cold_costs"),
            cold_inclusive_makespans=service.get("cold_inclusive_makespans"),
            sweep_makespans=service.get("sweep_makespans"),
            throughput_population_interval=service.get("throughput_population_interval"),
            latency_intervals=service.get("latency_intervals"),
            independent_deployment=False,
            sweep_support_limit="Two sweeps; request pairs and repeats do not establish deployment populations.",
            nfr01_met=False,
            nfr01_status="unmet_actual_matched_Rust_Python10x_absent",
        ),
        arc_evidence={
            k: arc.get(k)
            for k in [
                "new_outcome_count",
                "shared_arms",
                "arm_overlap_games",
                "credited_new_levels",
                "current_game_execution_count",
                "reopen_condition",
            ]
        },
        acceptance_gates=dict(
            owned_validation=passed, independent_science=False, terminal_accounting=True
        ),
    )
