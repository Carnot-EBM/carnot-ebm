"""REQ-REPORT-8233: freeze current inputs without upgrading historical failures.

Every scheduled task retains a disposition. Saved byte copies let a later replay
distinguish missing evidence from measured zero without rerunning producers.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar, validate_primary
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v710_contract_replay import snapshot
from carnot.verify import utility_audit_8224 as audit

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8233_v711_capstone"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_v711_capstone_8233.py"
MILESTONE = "2026.10.711"
DESIGN = "openspec/change-proposals/research-roadmap-vNEXT.md"
BLOCKED = {
    8226: "results/experiment_8226_learning_audit.json",
    8228: "results/experiment_8228_concurrent_service.json",
}
OWNED = [
    "python/carnot/reporting/v711_capstone_evidence.py",
    "python/carnot/reporting/v711_capstone.py",
    CLI,
]
COUNTS = ["intended_count", "completed_count", "failed_count", "censored_count", "excluded_count"]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real boundaries so input walks never conceal unfinished work."""
    print(f"[exp8233] phase={phase} completed={completed} pending={pending}", flush=True)


def bind(path: Path, raw: Path, refs: list[Json], digest: str | None = None) -> Path:
    """Copy once and check expected hashes before consuming imported bytes."""
    ref = snapshot(path, raw, str(len(refs)))
    refs.append(ref)
    if not ref["exists"] or (digest is not None and digest != ref["sha256"]):
        raise ValueError("input_bytes_unavailable_or_changed:" + str(path))
    return Path(ref["snapshot_path"])


def operand(
    task: str, path: str, digest: Any, field: str, expected: Any, observed: Any, op: str = "=="
) -> Json:
    """Name exact failed operands; absence is not a measured zero."""
    return dict(
        upstream=task,
        upstream_id=task,
        path=path,
        hash=digest,
        artifact_field=field,
        op=op,
        expected=expected,
        observed=observed,
        passed=False,
    )


def measure(root: Path, raw: Path) -> Json:
    """Use a small current manifest; producer checks remain their own dispositions."""
    refs: list[Json] = []
    frozen = bind(root / DESIGN, raw / "custody", refs)
    _, tasks = parse_design(frozen.read_text(), milestone=MILESTONE)
    if len(tasks) != 14 or [t["id"].split("-")[0] for t in tasks] != [
        f"exp{i}" for i in range(8220, 8234)
    ]:
        raise ValueError("exact_fourteen_task_contract")
    work: Json = dict(
        tasks=tasks,
        references=refs,
        primaries={},
        dispositions=[],
        failures=[],
        H1={},
        H2={},
        consumer_manifest=[],
        historical_limitations=[],
    )
    for index, task in enumerate(tasks[:-1]):
        identity = 8220 + index
        path = root / task["deliverable"]
        if not path.is_file() and identity in BLOCKED:
            path = root / BLOCKED[identity]
        progress("before_input", index, 13 - index)
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
            copy = bind(path, raw / "custody", refs)
            row["sha256"] = sha256_file(copy)
            value = json.loads(copy.read_bytes())
            if not isinstance(value, dict):
                raise ValueError("object_schema")
            if value.get("schema") == "blocked_gate_check_v1":
                if (
                    value.get("experiment") != identity
                    or value.get("blocked_at_layer") != "conductor_pre_gate"
                ):
                    raise ValueError("conductor_block_schema")
                row.update(
                    disposition="conductor_pre_gate",
                    verdict_class="blocked",
                    honest_verdict="complete_blocked_conductor_pre_gate",
                    numerator=1,
                )
                for gate in value["gates_evaluated"]:
                    if not gate["passed"]:
                        work["failures"].append(
                            operand(
                                gate["upstream"],
                                gate["artifact_path"],
                                gate["artifact_sha256"],
                                gate["artifact_field"],
                                gate["expected"],
                                gate["actual"],
                                gate["op"],
                            )
                        )
            else:
                validate_primary(value, path)
                if value["task_id"] != task["id"]:
                    raise ValueError("exact_task_identity")
                if (
                    not isinstance(value.get("rows"), list)
                    or not all(type(value.get(k)) is int for k in COUNTS)
                    or not all(
                        type(value.get(k)) is bool
                        for k in ["required_checks_passed", "flagged_adversarial"]
                    )
                ):
                    raise ValueError("current_primary_schema")
                side = (
                    path.parent / "raw" / path.stem / "validators" / (row["sha256"][7:] + ".json")
                )
                report = read_bound_sidecar(path, side)
                bind(side, raw / "custody", refs)
                if report["report"]["passed"] is not True:
                    raise ValueError("terminal_report_failed")
                eligible = (
                    value.get("required_checks_passed") is True
                    and value.get("flagged_adversarial") is False
                    and value["verdict_class"] in {"positive", "circular_positive", "null"}
                )
                row.update(
                    disposition="producer_terminal",
                    producer_executed=True,
                    honest_verdict=value["honest_verdict"],
                    verdict_class=value["verdict_class"],
                    source_counts={k: value.get(k) for k in COUNTS},
                    numerator=1,
                    eligible=eligible,
                    excluded=not eligible,
                    failed=value["verdict_class"] == "disqualified",
                    source_rows_reference=dict(
                        path=str(copy),
                        sha256=row["sha256"],
                        artifact_field="rows",
                        row_count=len(value["rows"]),
                        principle="The frozen primary preserves every original arm, unit, numerator, denominator and missing mask, including disqualified rows.",
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
                                str(
                                    check.get(
                                        "artifact_field",
                                        check.get("field", check.get("check", "upstream_check")),
                                    )
                                ),
                                check.get("expected"),
                                check.get("observed"),
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
                                required_checks_passed=value.get("required_checks_passed"),
                            ),
                        )
                    )
        except (OSError, ValueError, KeyError, TypeError) as error:
            row.update(
                missing=True,
                honest_verdict="complete_blocked_upstream_evidence",
                verdict_class="blocked",
                disposition="unavailable_output",
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
        work["dispositions"].append(row)
        work["consumer_manifest"].append(
            dict(
                task_id=task["id"],
                path=str(path),
                sha256=row["sha256"],
                schema=value.get("schema") if not row["missing"] else None,
            )
        )
        progress("after_input", index + 1, 12 - index)
    current = work["primaries"].get(tasks[0]["id"], {})
    work["historical_limitations"] = current.get("historical_dispositions", [])
    work["unexecuted_v710_design_entries"] = current.get("unexecuted_v710_design_entries", [])
    if current.get("authority_snapshots"):
        ref = current["authority_snapshots"]["design"]
        copied = bind(Path(ref["snapshot_path"]), raw / "custody", refs, ref["sha256"])
        _, original_tasks = parse_design(copied.read_text(), milestone=MILESTONE)
        if original_tasks != tasks:
            raise ValueError("frozen_current_contract_drift")
    row = work["dispositions"][4]
    progress("before_benchmark_H1", 0, 128)
    try:
        value = work["primaries"][tasks[4]["id"]]
        if not row["eligible"]:
            raise ValueError("eligible_current_audit_required")
        ref = value["measurement_reference"]
        copied = bind(Path(ref["path"]), raw / "custody", refs, ref["sha256"])
        measured = json.loads(copied.read_bytes())
        work["H1"] = audit.reduce(measured["evidence"])
        if work["H1"]["rows"] != value["rows"] or work["H1"]["H1"] != value["H1"]:
            raise ValueError("current_H1_primitive_drift")
        work["h1_primitive_reference"] = refs[-1]
    except (OSError, ValueError, KeyError, TypeError) as error:
        work["H1"] = {}
        work["failures"].append(
            operand(
                row["task_id"],
                row["path"],
                row["sha256"],
                "H1_qualified_primitive_rows",
                True,
                str(error),
            )
        )
    progress("after_benchmark_H1", 128 if work["H1"] else 0, 0)
    # No current audit ran after the learning gate failed. Reusing older learning
    # statistics would falsely turn that absence into a current result.
    h2 = work["dispositions"][6]
    work["failures"].append(
        operand(h2["task_id"], h2["path"], h2["sha256"], "H2_qualified_current_audit", True, None)
    )
    return work


def reduce(work: Json, receipts: list[Json]) -> Json:
    """Complete disposition accounting grants no independent scientific credit."""
    if len(work["dispositions"]) != 13 or len(work["tasks"]) != 14:
        raise ValueError("fourteen_disposition_roster")
    rows = deepcopy(work["dispositions"])
    passed = bool(receipts) and all(r["passed"] for r in receipts)
    blocked = bool(work["failures"])
    verdict = "complete_blocked_upstream_evidence" if blocked else "complete_null_v711_capstone"
    kind = "blocked" if blocked else "null"
    if not passed:
        verdict, kind = "complete_disqualified_owned_validation", "disqualified"
    rows.append(
        dict(
            experiment_id=8233,
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
    values = work["primaries"]

    def primary(index: int) -> Json:
        return dict(values.get(work["tasks"][index]["id"], {}))

    h1 = work["H1"]
    boards = [
        dict(
            board=board,
            experiment_id=8220 + index,
            path=rows[index]["path"],
            sha256=rows[index]["sha256"],
            honest_verdict=rows[index]["honest_verdict"],
            verdict_class=rows[index]["verdict_class"],
            qualified=rows[index]["eligible"],
            current_board_execution=False,
            benefit_score=0,
            obligation=primary(index).get(key, {}),
        )
        for index, board, key in [
            (10, "KV260", "kv260_obligation"),
            (11, "PolarFire", "polarfire_obligation"),
            (12, "GateMate", "gatemate_obligation"),
        ]
    ]
    retirements = []
    for task, row in zip(work["tasks"][:-1], rows[:-1]):
        for prior in task.get("prior_failures", []):
            same = prior["verdict"] == row["honest_verdict"]
            retirements.append(
                dict(
                    task_id=task["id"],
                    prior=prior,
                    current_verdict=row["honest_verdict"],
                    same_verdict=same,
                    retire_method_family=False,
                    decision="retire"
                    if same and prior["retire_if_same_verdict"]
                    else "await_named_evidence"
                    if row["verdict_class"] in {"blocked", "disqualified"}
                    else "continue",
                    scope=task["title"],
                    reopening_condition=(
                        "New authenticated environment outcomes after the Exp8229 frontier, with supported arm comparisons and cross-game overlap; another unchanged frontier retires immediately."
                        if row["experiment_id"] == 8229
                        else "Dated cable, port and power receipts that change the0xffffffff operand before any bounded physical GateMate attempt."
                        if row["experiment_id"] == 8232
                        else "Authenticated independent device operation, transfer and whole-request spans bound to the published state hashes; host timing cannot reopen device benefit."
                        if row["experiment_id"] in {8230, 8231}
                        else "Passing current Exp8227 server qualification followed by96 independent Exp8228 requests and four cold starts under the frozen protocol."
                        if row["experiment_id"] in {8227, 8228}
                        else "Named current sealed source or retention evidence with changed mechanism, passing owned checks, lower97.5% cost gain>.02 and all frozen Brier/false-accept controls; renaming tasks cannot reopen this scope."
                    ),
                )
            )
    gaps = [
        dict(
            gap="useful_verified_decisions",
            requirements=["FR-06", "FR-12"],
            moved=bool(h1.get("h1_development_signal_score")),
            closed=False,
            disposition="completed_null" if h1 else "blocked",
            evidence=rows[4]["path"],
            remaining="Independent source labels and action benefit over equally patched simple heads.",
        ),
        dict(
            gap="later_learning_and_retention",
            requirements=["FR-11"],
            moved=False,
            closed=False,
            disposition="blocked",
            evidence=rows[6]["path"],
            remaining="Qualified later cost benefit and sealed independent retention vs frozen and global-only.",
        ),
        dict(
            gap="request_scale_deployment",
            requirements=["FR-05", "FR-08", "NFR-01"],
            moved=False,
            closed=False,
            disposition="blocked",
            evidence=rows[8]["path"],
            remaining="Independent cold-inclusive request evidence and the original10x Rust target.",
        ),
    ]
    arc = primary(9)
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
        missing_output_count=sum(r["missing"] for r in rows),
        conductor_pre_gate_count=sum(r["disposition"] == "conductor_pre_gate" for r in rows),
        capstone_execution_ready_score=int(passed),
        science_ready_score=0,
        required_checks_passed=passed,
        flagged_adversarial=False,
        H1=dict(
            status="completed_signal"
            if h1.get("h1_development_signal_score")
            else "completed_null"
            if h1
            else "blocked",
            alpha=0.025,
            intended_count=128,
            statistics=h1,
        ),
        H2=dict(
            status="blocked",
            alpha=0.025,
            intended_count=192,
            retention_intended_count=64,
            statistics=None,
            trajectory_qualified=rows[5]["eligible"],
            audit_qualified=rows[6]["eligible"],
        ),
        h1_development_signal_score=h1.get("h1_development_signal_score", 0),
        h2_development_signal_score=0,
        multiplicity=dict(
            family=["H1", "H2"], alpha_per_hypothesis=dict(H1=0.025, H2=0.025), alpha_transfer=False
        ),
        verifier_is_oracle=True,
        exposure_scope="exposed development; administrative custody and cached candidates",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        gate_check_summary=work["failures"],
        three_prd_gaps=gaps,
        board_obligations=boards,
        retirements=retirements,
        historical_limitations=work["historical_limitations"],
        unexecuted_v710_design_entries=work["unexecuted_v710_design_entries"],
        task_contract=work["tasks"],
        canonical_tasks_sha256=canonical_hash(work["tasks"]),
        arc_evidence=dict(
            new_outcome_count=arc.get("new_outcome_count"),
            shared_arms=arc.get("shared_arms"),
            arm_overlap_games=arc.get("arm_overlap_games"),
            new_solve_credit=0,
            credited_new_levels=0,
            solve_claims=[],
            current_game_execution_count=0,
        ),
        request_accounting=dict(
            intended_generation_calls=96,
            observed_measurement_calls=None,
            canary_source_counts=primary(7).get("completed_count"),
            benchmark_executed=rows[8]["producer_executed"],
            measurement_disposition=rows[8]["honest_verdict"],
            cold_starts_intended=4,
            missing_metric_status="unmeasured",
            independent_deployment=False,
            nfr01_met=False,
            nfr01_required_ratio=10,
            rust_service_held_fixed=True,
            planned_slots=[
                dict(
                    unit_id=f"sweep{s}-{arm}-slot{i:02}",
                    sweep=s,
                    arm=arm,
                    source_slot=i,
                    status="unexecuted_obligation",
                    numerator=None,
                    denominator=1,
                    missing=True,
                    metric="whole_request_duration_s",
                )
                for s in range(2)
                for arm in ["serial", "concurrent"]
                for i in range(24)
            ],
        ),
        upstream_validation_failures=[
            dict(task_id=task, receipt=r)
            for task, primary in values.items()
            for r in primary.get("validation_receipts", [])
            if r.get("passed") is False
            and r.get("scope", r.get("classification")) != "repository_health"
        ],
        upstream_repository_health=[
            dict(
                task_id=task,
                evidence=primary.get("repository_health", primary.get("global_health", {})),
            )
            for task, primary in values.items()
        ],
        acceptance_gates=dict(
            owned_validation=dict(
                passed=passed, principle="Only measured owned checks establish readiness."
            ),
            independent_science=dict(
                passed=False, principle="Exposed development does not establish generalization."
            ),
        ),
    )
