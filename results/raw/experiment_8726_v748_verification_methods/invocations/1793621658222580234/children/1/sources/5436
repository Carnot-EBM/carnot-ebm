"""REQ-REPORT-8275: bind source bytes so missing work cannot become a finding.

Task rows count reconciled dispositions. Original scientific rows and denominators
remain in immutable producer copies, independently of this administrative count.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any

from carnot.reporting import v713_capstone_evidence as prior
from carnot.reporting.v713_capstone_evidence import read as read
from carnot.reporting import v714_coverage_custody as custody
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v710_contract_replay import require_reference, snapshot

Json = dict[str, Any]
ROOT = prior.ROOT
NAME = "experiment_8275_v714_capstone"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_v714_capstone_8275.py"
MILESTONE = "2026.10.714"
DESIGN = custody.DESIGN
HISTORY = "results/experiment_8261_v713_capstone.json"
POLARFIRE = "results/experiment_8259_v713_polarfire_dispatch_qualification.json"
POLARFIRE_PIN = "sha256:9933e509b6d05112843986d8271455df2194dc74f4000a2e68abbf52e0d04589"
COUNTS = prior.COUNTS
AUTHORITIES = [
    DESIGN,
    custody.STAGED,
    custody.ACTIVE,
    custody.PROTOCOL,
    custody.HISTORY,
    "research-complete.yaml",
    "ops/conductor-log.md",
]
OWNED = [
    "python/carnot/reporting/v714_capstone_evidence.py",
    "python/carnot/reporting/v714_capstone.py",
    CLI,
]
NEXT = [
    "Authenticated durable coverage and full current authority agreement.",
    "Qualified token-exact focal requests and causal typed-action controls.",
    "CUDA availability followed by bounded authenticated Qwen canary requests.",
    "Qualified original fit128 three-view capture and complete acquisition clocks.",
    "Qualified original tune64 three-view capture and complete acquisition clocks.",
    "Frozen equal-information heads from qualified fit and tune features.",
    "Original reserved128 views and sealed decisions, without replacement sources.",
    "Qualified H1 lower cost gain above .02 with support, controls and oracle headroom.",
    "Causal delayed constraint admission on the original96 distinct-source stream.",
    "Qualified H2 later gain and separate32-source retention at alpha .025.",
    "New authenticated ARC supervisor outcomes with cross-game arm overlap.",
    "Useful k<=5 workload, authenticated SSH fabric transcript and full request timing.",
    "Dated physical change, GM1Ax0x20000001, flashed n16 tile and device hash parity.",
    "Qualified original science primitives and independently closed PRD benefit gaps.",
]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real boundaries so the conductor can observe unfinished work."""
    print(f"[exp8275] phase={phase} completed={completed} pending={pending}", flush=True)


def bind(path: Path, raw: Path, refs: list[Json]) -> Json:
    """Copy actual bytes and their qualified sidecar before importing any values."""
    ref = snapshot(path, raw / "custody", str(len(refs)))
    refs.append(ref)
    digest = ref["sha256"] or "sha256:missing"
    side_path = path.parent / "raw" / path.stem / "validators" / (digest[7:] + ".json")
    side = snapshot(side_path, raw / "custody", str(len(refs)))
    refs.append(side)
    if side["exists"]:
        try:
            read_bound_sidecar(path, side_path)
        except (OSError, ValueError, KeyError, TypeError) as error:
            side["binding_error"] = str(error)
    return dict(reference=ref, sidecar=side)


def measure(root: Path, raw: Path) -> Json:
    """Freeze all fourteen scheduled identities and independent historical evidence."""
    refs = [snapshot(root / p, raw / "custody", str(i)) for i, p in enumerate(AUTHORITIES)]
    _, tasks = parse_design((ROOT / DESIGN).read_text(), milestone=MILESTONE)
    if refs[0]["exists"]:
        _, tasks = parse_design(Path(refs[0]["snapshot_path"]).read_text(), milestone=MILESTONE)
    if len(tasks) != 14 or [t["id"].split("-")[0] for t in tasks] != [
        f"exp{i}" for i in range(8262, 8276)
    ]:
        raise ValueError("exact_fourteen_task_contract")
    inputs = []
    for i, task in enumerate(tasks[:-1]):
        progress("before_input", i, 13 - i)
        inputs.append(bind(prior.resolve(task, 8262 + i, root), raw, refs))
        progress("after_input", i + 1, 12 - i)
    history = bind(root / HISTORY, raw, refs)
    polar = bind(root / POLARFIRE, raw, refs)
    polar_refs = []
    if polar["reference"]["exists"]:
        value = prior.read(polar["reference"])
        paths = [value.get("terminal_validation_sidecar_path", "absent-terminal")]
        paths += [
            value.get(k, {}).get("path", "absent-" + k)
            for k in ["board_reference", "host_reference"]
        ]
        side = prior.read(polar["sidecar"]) if polar["sidecar"]["exists"] else {}
        for receipt in side.get("report", {}).get("receipts", []):
            paths += [receipt[s + "_path"] for s in ["stdout", "stderr"]]
        for p in paths:
            ref = snapshot(Path(p), raw / "custody", str(len(refs)))
            refs.append(ref)
            polar_refs.append(ref)
    return dict(
        references=refs,
        tasks=tasks,
        inputs=inputs,
        history=history,
        polar=polar,
        polar_refs=polar_refs,
    )


def graduation(work: Json) -> tuple[Json, list[Json]]:
    """Graduate only the pinned real Linux CPU dispatch, never inferred acceleration."""
    item, failures = work["polar"], []
    ref = item["reference"]
    try:
        value, side = prior.read(ref), prior.read(item["sidecar"])
        if ref["sha256"] != POLARFIRE_PIN or item["sidecar"].get("binding_error"):
            raise ValueError("unchanged_primary_sha256")
        checks = dict(
            required_checks_passed=value["required_checks_passed"] is True,
            flagged_adversarial=value["flagged_adversarial"] is False,
            polarfire_workload_validated=value["polarfire_workload_validated"] is True,
            actual_device_dispatch=value["current_device_execution_count"] == 1,
            output_hash_parity=value["host_parity"] is True
            and value["board_output_hashes"] == value["host_expected_hashes"],
            terminal_sidecar=side["primary_sha256"] == ref["sha256"]
            and side["report"]["passed"] is True,
        )
        for r in work["polar_refs"]:
            require_reference(r)
            if not r["exists"]:
                raise ValueError("terminal_primitive_missing:" + r["path"])
        by_path = {r["path"]: r for r in work["polar_refs"]}
        terminal = prior.read(by_path[value["terminal_validation_sidecar_path"]])
        checks["terminal_primary_binding"] = (
            terminal["publication"]["primary_sha256"] == ref["sha256"]
            and terminal["normal_process_exit"] is True
        )
        checks["actual_board_primitive"] = (
            prior.read(by_path[value["board_reference"]["path"]])["executed"] is True
        )
        for k in ["board_reference", "host_reference"]:
            checks[k] = by_path[value[k]["path"]]["sha256"] == value[k]["sha256"]
        receipts = side["report"]["receipts"]
        checks["adversarial_sidecar"] = any(
            r["name"] == "adversarial" and r["actual_exit"] == 0 and r["passed"] for r in receipts
        )
        for r in receipts:
            for stream in ["stdout", "stderr"]:
                if by_path[r[stream + "_path"]]["sha256"] != r[stream + "_sha256"]:
                    raise ValueError("terminal_stream_hash")
        for field, passed in checks.items():
            if not passed:
                failures.append(
                    prior.operand("exp8259", ref["path"], ref["sha256"], field, True, False)
                )
    except (OSError, ValueError, KeyError, TypeError) as error:
        failures.append(
            prior.operand("exp8259", ref["path"], ref["sha256"], str(error), True, None)
        )
    return dict(
        graduated=not failures,
        scope="board-local PolarFire Linux CPU dispatch and output hash parity",
        fabric_acceleration=False,
        scientific_benefit=False,
        current_device_executions=0,
        retained_obligations=[
            "durable activation and restart",
            "transfer-inclusive performance",
            "independent benefit",
        ],
    ), failures


def reduce(work: Json, receipts: list[Json]) -> Json:
    """Current administrative completion grants no unmeasured scientific benefit."""
    refs, tasks = work["references"], work["tasks"]
    for ref in refs:
        require_reference(ref)
    if len(tasks) != 14 or len(work["inputs"]) != 13:
        raise ValueError("fourteen_disposition_roster")
    failures: list[Json] = []
    authority: Json = dict(activated=False, gate_check_summary=[])
    try:
        paths = [Path(r.get("snapshot_path", "/tmp/absent-v714-authority")) for r in refs[:3]]
        with tempfile.TemporaryDirectory(prefix="carnot8275-authority-") as directory:
            authority = custody.authority.assess_authorities(
                *paths, Path(directory), milestone=MILESTONE, first_id=8262, count=14
            )
        if parse_design(paths[0].read_text(), milestone=MILESTONE)[1] != tasks:
            raise ValueError("contract_primitive_drift")
    except OSError as error:
        failures.append(
            prior.operand(
                "authority",
                refs[0]["path"],
                refs[0]["sha256"],
                "authority_available",
                True,
                str(error),
            )
        )
    failures.extend(authority["gate_check_summary"])
    if refs[3]["sha256"] != custody.PIN:
        failures.append(
            prior.operand(
                "protocol",
                refs[3]["path"],
                refs[3]["sha256"],
                "science_protocol_sha256",
                custody.PIN,
                refs[3]["sha256"],
            )
        )
    rows, sources = [], {}
    for i, (task, item) in enumerate(zip(tasks[:-1], work["inputs"])):
        if item["reference"] not in refs or item["sidecar"] not in refs:
            raise ValueError("input_reference_drift")
        row, errors, source = prior.outcome(task, 8262 + i, item)
        row.update(
            producer_honest_verdict=source.get("honest_verdict"),
            evidence_type=row["disposition"],
            declared_path=task["deliverable"],
            source_independent_count=source.get("independent_count"),
        )
        if row["disposition"] == "conductor_pre_gate":
            gate_value = prior.read(item["reference"])
            if gate_value.get("task_id") is None:
                errors = [g for g in errors if g.get("artifact_field") != "task_id"]
            row["identity_basis"] = "exact_experiment_id_and_bound_conductor_gate_receipt"
            for gate in gate_value["gates_evaluated"]:
                path = Path(gate["artifact_path"])
                bound: Json = next((r for r in refs if r["path"] == str(path)), {})
                if not bound.get("exists") or bound.get("sha256") != gate["artifact_sha256"]:
                    errors.append(
                        prior.operand(
                            gate["upstream"],
                            str(path),
                            gate["artifact_sha256"],
                            "bound_conductor_gate_receipt",
                            True,
                            False,
                        )
                    )
        rows.append(row)
        failures.extend(errors)
        sources[8262 + i] = source
    digest = canonical_hash(tasks)[7:]
    if sources[8262].get("canonical_tasks_sha256") != digest:
        failures.append(
            prior.operand(
                tasks[0]["id"],
                rows[0]["path"],
                rows[0]["sha256"],
                "canonical_tasks_sha256",
                digest,
                sources[8262].get("canonical_tasks_sha256"),
            )
        )
    history_task = dict(id="exp8261-capstone")
    _, history_errors, history = prior.outcome(history_task, 8261, work["history"])
    failures.extend(history_errors)
    polar, polar_errors = graduation(work)
    failures.extend(polar_errors)
    owned = [r for r in receipts if r.get("scope") not in {"preconditions", "upstream"}]
    tools = [r for r in receipts if r.get("scope") == "preconditions"]
    passed = bool(owned) and all(r["passed"] and r.get("normal_exit", True) for r in owned)
    for r in receipts:
        if not r["passed"]:
            failures.append(
                prior.operand(
                    r.get("scope", "owned"),
                    r["stdout_path"],
                    r["stdout_sha256"],
                    r["name"] + ".normal_exit",
                    0,
                    r["actual_exit"],
                )
            )
    ready = int(passed and authority["activated"] and all(r["passed"] for r in tools))
    kind = "disqualified" if not passed else "blocked" if failures else "null"
    verdict = (
        "complete_"
        + kind
        + (
            "_owned_validation"
            if not passed
            else "_upstream_evidence"
            if failures
            else "_v714_capstone"
        )
    )
    rows.append(
        dict(
            experiment_id=8275,
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
            producer_honest_verdict=verdict,
            disposition="self_owned_completion",
            evidence_type="self_owned_completion",
        )
    )
    protocol = prior.read(refs[3]) if refs[3]["exists"] else {}
    branches = [
        dict(
            status="blocked_unmeasured",
            alpha=0.025,
            intended_count=n,
            completed_count=None,
            statistics=None,
            source_information_gain=None,
            energy_specific_gain=None,
            later_constraint_use=None,
            retention_gain=None,
            audit_experiment_id=i,
            failed_operand="eligible_registered_audit_primitives",
            frozen_registration=protocol.get(h),
            shared_fallback=protocol.get("action_rule"),
            selected_comparator=None,
        )
        for h, i, n in [("H1", 8269, 128), ("H2", 8271, 96)]
    ]
    branches[1]["retention_intended_count"] = 32
    accounting = [
        dict(
            experiment_id=i,
            role=role,
            intended_source_count=n,
            maximum_generation_calls=limit,
            imported_canary_counted_once=True,
            counts=sources[i].get("model_invocation_counts"),
            current_phase_spans=sources[i].get("phase_spans"),
            availability="available"
            if sources[i].get("model_invocation_counts") is not None
            else "unavailable",
            imported_calls_are_current_capstone_calls=False,
        )
        for i, role, n, limit in [
            (8264, "canary", 12, 36),
            (8265, "fit", 128, 384),
            (8266, "tune", 64, 192),
            (8268, "evaluation", 128, 384),
        ]
    ]
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
        pre_gate_count=sum(r["disposition"] == "conductor_pre_gate" for r in rows),
        missing_output_count=sum(r["missing"] for r in rows),
        required_checks_passed=passed,
        flagged_adversarial=False,
        capstone_execution_ready_score=ready,
        science_ready_score=0,
        h1_development_signal_score=0,
        h2_development_signal_score=0,
        H1=branches[0],
        H2=branches[1],
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        verifier_is_oracle=True,
        exposure_scope="exposed development; reused sources and oracle mechanics",
        gate_check_summary=failures,
        canonical_tasks_sha256=digest,
        full_task_authority_equal=authority["activated"],
        activation_snapshot=refs[2],
        task_contract=tasks,
        historical_v713={
            k: history.get(k)
            for k in [
                "rows",
                "actual_executed_task_count",
                "conductor_pre_gate_count",
                "missing_output_count",
                "H1",
                "H2",
                "retirements",
                "gate_check_summary",
            ]
        },
        archive_lag=dict(
            planning_archive_stopped_at="V712",
            archive_snapshot=refs[5],
            historical_authority=refs[4],
            conductor_log=refs[6],
            archive_is_current_execution=False,
        ),
        polarfire_graduation=polar,
        polarfire_terminal_evidence_hashes=[
            work["polar"]["reference"],
            work["polar"]["sidecar"],
            *work["polar_refs"],
        ],
        evidence_improved=bool(
            sources[8262].get("coverage_custody_ready_score")
            or sources[8263].get("view_kernel_ready_score")
        ),
        evidence_improvement_scope="durable coverage custody and token-exact typed-action mechanics only; no measured new source-information gain",
        learning_improved=False,
        live_call_accounting=accounting,
        current_capture_cost_scope=dict(
            status="unavailable",
            complete_acquisition_cost_s=None,
            complete_update_cost_s=None,
            historical_provenance_excluded=True,
            missing_is_not_measured_zero=True,
        ),
        board_obligations=[
            dict(
                board=b,
                evidence=sources[i],
                obligation=sources[i].get(k),
                sha256=rows[i - 8262]["sha256"],
                path=rows[i - 8262]["path"],
                benefit_score=0,
            )
            for i, b, k in [
                (8273, "KV260", "kv260_obligation"),
                (8274, "GateMate", "gatemate_obligation"),
            ]
        ]
        + [
            dict(
                board="PolarFire",
                evidence=polar,
                obligation=polar["retained_obligations"],
                benefit_score=0,
            )
        ],
        arc_evidence=sources[8272],
        three_prd_gaps=[
            dict(gap=g, requirements=req, closed=False, remaining=condition)
            for g, req, condition in [
                ("useful_verified_decisions", ["FR-06", "FR-12"], NEXT[7]),
                ("later_learning_and_retention", ["FR-11"], NEXT[9]),
                ("request_scale_deployment", ["FR-05", "FR-08", "NFR-01"], NEXT[11]),
            ]
        ],
        retirements=[
            dict(
                task_id=t["id"],
                predecessors=t.get("prior_failures", []),
                decision="carry_anti_churn_obligation"
                if i in [8272, 8273, 8274]
                else "await_eligible_evidence",
                informative_registered_null=False,
                limitation_of_hypothesis=False,
                actual_typed_action_positive_control=sources[8263].get("positive_control_passed"),
                permissible_action_oracle_headroom=None,
                reopening_condition=NEXT[i - 8262],
            )
            for i, t in zip(range(8262, 8276), tasks)
        ],
        acceptance_gates=dict(
            owned_validation=passed,
            full_task_authority=authority["activated"],
            terminal_accounting=True,
            independent_science=False,
        ),
    )
