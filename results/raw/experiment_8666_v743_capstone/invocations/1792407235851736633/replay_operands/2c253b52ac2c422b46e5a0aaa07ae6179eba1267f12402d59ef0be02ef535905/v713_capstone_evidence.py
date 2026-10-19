"""REQ-REPORT-8261: reduce authenticated outcomes without inventing missing science.

Source copies preserve each producer's rows, including missing slots. Task counts
describe completed accounting; they do not count independent scientific sources.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import yaml

from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar, validate_primary
from carnot.reporting.roadmap_contract import TABLE_FIELDS, parse_design
from carnot.reporting.v710_contract_replay import require_reference, snapshot
from carnot.reporting.v711_capstone_evidence import operand
from carnot.reporting.v712_capstone_evidence import COUNTS as COUNTS, resolve

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8261_v713_capstone"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_v713_capstone_8261.py"
MILESTONE = "2026.10.713"
DESIGN = "openspec/change-proposals/research-roadmap-vNEXT.md"
HISTORY = "results/experiment_8247_v712_capstone.json"
OWNED = [
    "python/carnot/reporting/v713_capstone_evidence.py",
    "python/carnot/reporting/v713_capstone.py",
    CLI,
]
NEXT_EVIDENCE = {
    8248: "Retain the unchanged registered protocol and qualify current capture operands.",
    8249: "Pass the failed owned validation operand before allowing live capture.",
    8250: "Qualified view kernel followed by authenticated bounded canary ledgers.",
    8251: "Complete registered fit/tune three-view capture with acquisition costs.",
    8252: "Frozen equal-information fits on qualified measured intervention features.",
    8253: "Qualified held stream/retention views with sealed labels and no replacements.",
    8254: "Registered H1 lower gain above .02 with support, controls and oracle headroom.",
    8255: "Qualified delayed constraint-admission trajectory on current stream features.",
    8256: "Registered H2 later decision gain and independently sealed retention.",
    8257: "New authenticated ARC outcomes with cross-game arm overlap; no unchanged game rerun.",
    8258: "Complete current acquisition costs and measured compatible kernels plus matched Rust/Python10x.",
    8259: "Preserve qualified device parity; new decision/learning benefit requires independent evidence.",
    8260: "Dated cable/port/power change, then GM1Ax detection and flashed n16 sample/hash smoke.",
    8261: "Qualify missing science operands and close each PRD benefit gap independently.",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual boundaries so pending work remains visible to the conductor."""
    print(f"[exp8261] phase={phase} completed={completed} pending={pending}", flush=True)


def read(ref: Json) -> Any:
    """Read immutable source bytes; a missing source remains an unavailable operand."""
    require_reference(ref)
    if not ref["exists"]:
        raise ValueError("missing_input:" + ref["path"])
    return json.loads(Path(ref["snapshot_path"]).read_bytes())


def audit_plan(root: Path) -> list[Json]:
    """Freeze each science audit's own CLI instead of replaying a fitted producer."""
    return [
        dict(
            name="science_audit_" + str(identity),
            experiment_id=identity,
            input_schema=schema,
            argv=[
                str(ROOT / ".venv/bin/python"),
                "-u",
                str(ROOT / f"scripts/experiments/experiment_{identity}_v713_{name}.py"),
                "--cold-replay",
                str(root / f"results/experiment_{identity}_v713_{name}.json"),
            ],
            deadline=180,
            expected=0,
            scope="upstream",
        )
        for identity, name, schema in [
            (8254, "intervention_benefit_audit", "registered V713 H1 source intervention audit"),
            (
                8256,
                "constraint_learning_audit",
                "registered V713 H2 delayed constraint/retention audit",
            ),
        ]
    ]


def outcome(task: Json, identity: int, item: Json) -> tuple[Json, list[Json], Json]:
    """Rebuild a disposition from saved bytes, keeping zero distinct from absence."""
    ref = item["reference"]
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
        path=ref["path"],
        sha256=ref["sha256"],
        source_counts={k: None for k in COUNTS},
    )
    failures: list[Json] = []
    source: Json = {}
    try:
        value = read(ref)
        if value.get("schema") == "blocked_gate_check_v1":
            if (
                value.get("experiment") != identity
                or value.get("task_id", task["id"]) != task["id"]
            ):
                raise ValueError("conductor_identity")
            if value["blocked_at_layer"] != "conductor_pre_gate" or not value["gates_evaluated"]:
                raise ValueError("conductor_schema")
            row.update(
                disposition="conductor_pre_gate",
                verdict_class="blocked",
                honest_verdict="complete_blocked_conductor_pre_gate",
                numerator=1,
            )
            if value.get("task_id") != task["id"]:
                failures.append(
                    operand(
                        task["id"],
                        ref["path"],
                        ref["sha256"],
                        "task_id",
                        task["id"],
                        value.get("task_id"),
                    )
                )
            for gate in value["gates_evaluated"]:
                if not any(
                    gate["upstream"] == g["upstream"]
                    and gate["artifact_field"] == g["artifact_field"]
                    and gate["op"] == g["op"]
                    and gate["expected"] == g["value"]
                    for g in task.get("gated_on", [])
                ):
                    raise ValueError("conductor_gate_contract")
                if not gate["passed"]:
                    failures.append(
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
            validate_primary(value, Path(ref["path"]))
            if (
                value["task_id"] != task["id"]
                or not isinstance(value["rows"], list)
                or not all(type(value[k]) is int for k in COUNTS)
                or not all(
                    type(value[k]) is bool
                    for k in ["required_checks_passed", "flagged_adversarial"]
                )
            ):
                raise ValueError("primary_schema")
            side = read(item["sidecar"])
            if item["sidecar"].get("binding_error"):
                raise ValueError("terminal_binding_error")
            if side["primary_sha256"] != ref["sha256"] or side["report"]["passed"] is not True:
                raise ValueError("terminal_sidecar")
            source = value
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
                    path=ref["snapshot_path"],
                    sha256=ref["sha256"],
                    artifact_field="rows",
                    row_count=len(value["rows"]),
                ),
            )
            if not eligible:
                failures.append(
                    operand(
                        task["id"],
                        ref["path"],
                        ref["sha256"],
                        "qualified_current_evidence",
                        True,
                        dict(
                            verdict_class=value["verdict_class"],
                            required_checks_passed=value["required_checks_passed"],
                        ),
                    )
                )
            for gate in value.get("gate_check_summary", []):
                if isinstance(gate, dict) and gate.get("passed") is False:
                    failures.append(
                        dict(
                            gate,
                            upstream=task["id"],
                            path=gate.get("path", ref["path"]),
                            hash=gate.get("hash", ref["sha256"]),
                        )
                    )
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        row.update(
            missing=True,
            disposition="unavailable_output",
            verdict_class="blocked",
            honest_verdict="complete_blocked_upstream_evidence",
        )
        failures.append(
            operand(
                task["id"],
                ref["path"],
                ref["sha256"],
                "authenticated_current_output",
                True,
                None if not ref["exists"] else str(error),
            )
        )
    return row, failures, source


def measure(root: Path, raw: Path) -> Json:
    """Freeze authority, source primaries and qualified sidecars before reducing them."""
    refs = [
        snapshot(root / name, raw / "custody", str(i))
        for i, name in enumerate([DESIGN, "research-roadmap.yaml", HISTORY])
    ]
    _, tasks = parse_design(Path(refs[0]["snapshot_path"]).read_text(), milestone=MILESTONE)
    if len(tasks) != 14 or [t["id"].split("-")[0] for t in tasks] != [
        f"exp{i}" for i in range(8248, 8262)
    ]:
        raise ValueError("exact_fourteen_task_contract")
    historical_path = root / HISTORY
    historical_digest = refs[2]["sha256"] or "sha256:missing"
    historical_side = snapshot(
        historical_path.parent
        / "raw"
        / historical_path.stem
        / "validators"
        / (historical_digest[7:] + ".json"),
        raw / "custody",
        str(len(refs)),
    )
    refs.append(historical_side)
    inputs = []
    for index, task in enumerate(tasks[:-1]):
        progress("before_input", index, 13 - index)
        path = resolve(task, 8248 + index, root)
        ref = snapshot(path, raw / "custody", str(len(refs)))
        refs.append(ref)
        side: Json = dict(exists=False, path=str(path), sha256=None)
        if ref["exists"]:
            side_path = (
                path.parent / "raw" / path.stem / "validators" / (ref["sha256"][7:] + ".json")
            )
            side = snapshot(side_path, raw / "custody", str(len(refs)))
            refs.append(side)
            if side["exists"]:
                try:
                    read_bound_sidecar(path, side_path)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    side["binding_error"] = str(error)
        inputs.append(dict(reference=ref, sidecar=side))
        progress("after_input", index + 1, 12 - index)
    return dict(references=refs, tasks=tasks, inputs=inputs, history_sidecar=historical_side)


def reduce(work: Json, receipts: list[Json]) -> Json:
    """Owned validation grants execution credit; missing science grants no benefit."""
    if len(work["inputs"]) != 13 or len(work["tasks"]) != 14:
        raise ValueError("fourteen_disposition_roster")
    for ref in work["references"]:
        require_reference(ref)
    if any(
        item["reference"] not in work["references"]
        or (item["sidecar"].get("exists") and item["sidecar"] not in work["references"])
        for item in work["inputs"]
    ):
        raise ValueError("input_reference_drift")
    table, tasks = parse_design(
        Path(work["references"][0]["snapshot_path"]).read_text(), milestone=MILESTONE
    )
    if tasks != work["tasks"]:
        raise ValueError("contract_primitive_drift")
    failures = []
    try:
        activation_ref = work["references"][1]
        if not activation_ref["exists"]:
            raise ValueError("activation_missing")
        active = yaml.safe_load(Path(activation_ref["snapshot_path"]).read_bytes())
        authority = (
            active["milestone"] == MILESTONE
            and active["tasks"] == tasks
            and len(table) == 14
            and all(
                shown["order"] == i + 1 and all(shown[k] == task[k] for k in TABLE_FIELDS)
                for i, (shown, task) in enumerate(zip(table, tasks))
            )
        )
        if not authority:
            raise ValueError("full_task_authority_drift")
    except (OSError, ValueError, KeyError, TypeError) as error:
        authority = False
        failures.append(
            operand(
                "activation",
                work["references"][1]["path"],
                work["references"][1]["sha256"],
                "full_task_object_equality",
                True,
                str(error),
            )
        )
    try:
        history = read(work["references"][2])
        validate_primary(history, Path(work["references"][2]["path"]))
        side = read(work["history_sidecar"])
        if (
            side["primary_sha256"] != work["references"][2]["sha256"]
            or side["report"]["passed"] is not True
        ):
            raise ValueError("historical_terminal_sidecar")
    except (OSError, ValueError, KeyError, TypeError) as error:
        history = {}
        ref = work["references"][2]
        failures.append(
            operand(
                "historical_v712",
                ref["path"],
                ref["sha256"],
                "authenticated_historical_primary",
                True,
                str(error),
            )
        )
    rows, sources = [], {}
    for i, (task, item) in enumerate(zip(tasks[:-1], work["inputs"])):
        row, row_failures, source = outcome(task, 8248 + i, item)
        rows.append(row)
        failures.extend(row_failures)
        sources[8248 + i] = source
    digest = canonical_hash(tasks).removeprefix("sha256:")
    if sources[8248].get("canonical_tasks_sha256") != digest:
        failures.append(
            operand(
                tasks[0]["id"],
                rows[0]["path"],
                rows[0]["sha256"],
                "canonical_tasks_sha256",
                digest,
                sources[8248].get("canonical_tasks_sha256"),
            )
        )
    owned = [r for r in receipts if r.get("scope") != "preconditions"]
    tool_failures = [r for r in receipts if r.get("scope") == "preconditions" and not r["passed"]]
    for failed in tool_failures:
        failures.append(
            operand(
                "required_tools",
                failed["stdout_path"],
                failed["stdout_sha256"],
                "required_tools.normal_exit",
                0,
                failed["actual_exit"],
            )
        )
    passed = bool(owned) and all(r["passed"] and r.get("normal_exit", True) for r in owned)
    kind = "disqualified" if not passed else "blocked" if failures else "null"
    verdict = (
        "complete_"
        + kind
        + (
            "_owned_validation"
            if not passed
            else "_upstream_evidence"
            if failures
            else "_v713_capstone"
        )
    )
    ready = int(passed and authority and not tool_failures)
    rows.append(
        dict(
            experiment_id=8261,
            task_id=tasks[-1]["id"],
            unit_id=tasks[-1]["id"],
            arm="task_disposition",
            condition="terminal_accounting",
            metric="owned_execution_readiness",
            numerator=ready,
            denominator=1,
            status="completed",
            completed=True,
            missing=False,
            producer_executed=True,
            eligible=bool(ready),
            excluded=not ready,
            failed=not passed,
            censored=False,
            honest_verdict=verdict,
            verdict_class=kind,
            disposition="self_owned_completion",
        )
    )
    branches = [
        dict(
            status="blocked_unmeasured",
            alpha=0.025,
            statistics=None,
            intended_count=n,
            completed_count=None,
            audit_experiment_id=identity,
            source_information_gain=None,
            energy_specific_gain=None,
            later_constraint_use=None,
            retention_gain=None,
            failed_operand="eligible_registered_audit_primitives",
        )
        for identity, n in [(8254, 128), (8256, 96)]
    ]
    branches[1]["retention_intended_count"] = 32
    return dict(
        honest_verdict=verdict,
        verdict_class=kind,
        rows=rows,
        task_dispositions=deepcopy(rows),
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
        capstone_execution_ready_score=ready,
        science_ready_score=0,
        required_checks_passed=passed,
        flagged_adversarial=False,
        H1=branches[0],
        H2=branches[1],
        h1_development_signal_score=0,
        h2_development_signal_score=0,
        evidence_improved=bool(
            sources[8259].get("required_checks_passed")
            and sources[8259].get("host_parity")
            and sources[8259].get("current_device_execution_count")
        ),
        learning_improved=False,
        evidence_improvement_scope="PolarFire owned qualification and oracle-defined real-device parity; no V713 scientific benefit demonstrated",
        verifier_is_oracle=True,
        exposure_scope="exposed development; upstream cached evidence",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        multiplicity=dict(alpha_per_hypothesis=dict(H1=0.025, H2=0.025), alpha_transfer=False),
        gate_check_summary=failures,
        task_contract=tasks,
        canonical_tasks_sha256=digest,
        full_task_authority_equal=authority,
        activation_snapshot=work["references"][1],
        historical_v712={
            k: history.get(k)
            for k in [
                "honest_verdict",
                "verdict_class",
                "rows",
                "paper_ready",
                "gate_check_summary",
                "actual_executed_task_count",
                "retirements",
            ]
        },
        three_prd_gaps=[
            dict(gap=name, requirements=req, closed=False, moved=False, remaining=next_evidence)
            for name, req, next_evidence in [
                (
                    "useful_verified_decisions",
                    ["FR-06", "FR-12"],
                    "Qualified H1 source-information and energy-specific cost gain.",
                ),
                (
                    "later_learning_and_retention",
                    ["FR-11"],
                    "Qualified later constraint gain and sealed retention.",
                ),
                (
                    "request_scale_deployment",
                    ["FR-05", "FR-08", "NFR-01"],
                    "Complete acquisition ledger and actual matched Rust/Python10x.",
                ),
            ]
        ],
        board_obligations=[
            dict(
                board=board,
                experiment_id=i,
                path=rows[i - 8248]["path"],
                sha256=rows[i - 8248]["sha256"],
                honest_verdict=rows[i - 8248]["honest_verdict"],
                verdict_class=rows[i - 8248]["verdict_class"],
                benefit_score=0,
                evidence={k: sources[i].get(k) for k in keys},
                obligation=sources[i].get(key),
            )
            for i, board, key, keys in [
                (
                    8258,
                    "KV260",
                    "kv260_obligation",
                    [
                        "ideal_whole_request_bound",
                        "phase_cost_rows",
                        "current_device_execution_count",
                        "nfr01_met",
                        "primitive_reference",
                    ],
                ),
                (
                    8259,
                    "PolarFire",
                    "polarfire_obligation",
                    [
                        "current_device_execution_count",
                        "polarfire_workload_validated",
                        "host_parity",
                        "board_cpu_seconds",
                        "transfer_seconds",
                        "board_output_hashes",
                        "host_expected_hashes",
                        "board_reference",
                        "host_reference",
                    ],
                ),
                (
                    8260,
                    "GateMate",
                    "gatemate_obligation",
                    [
                        "physical_change_evidence",
                        "current_jtag_retry_count",
                        "current_device_execution_count",
                        "physical_change_frontier",
                        "replay_input_reference",
                    ],
                ),
            ]
        ],
        arc_evidence={
            k: sources[8257].get(k)
            for k in [
                "new_outcome_count",
                "credited_new_levels",
                "current_game_execution_count",
                "shared_arms",
                "arm_overlap_games",
                "reopen_condition",
            ]
        },
        acquisition_accounting=dict(
            status="unavailable_capture_operands",
            capture_experiments=[8250, 8251, 8253],
            model_loads=None,
            generation_calls=None,
            complete_acquisition_cost_s=None,
            observed_live_capture_ledger_count=0,
            current_capstone_calls=0,
            historical_provenance_excluded=True,
            missing_is_not_measured_zero=True,
        ),
        retirements=[
            dict(
                task_id=task["id"],
                predecessors=task.get("prior_failures", []),
                decision="carry_anti_churn_obligation"
                if i in [8257, 8258, 8259, 8260]
                else "await_eligible_evidence",
                reopening_condition=NEXT_EVIDENCE[i],
                informative_registered_null=False,
                learnable_control_passed=None,
                oracle_action_headroom=None,
            )
            for i, task in zip(range(8248, 8262), tasks)
        ],
        acceptance_gates=dict(
            owned_validation=passed,
            full_task_authority=authority,
            terminal_accounting=True,
            independent_science=False,
        ),
    )
