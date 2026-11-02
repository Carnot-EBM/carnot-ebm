"""REQ-REPORT-8289: preserve source scope while reducing current task evidence.

Task accounting counts completed reconciliation, not scientific observations.
The existing readers authenticate authority and terminal bytes; missing units
retain their intended denominators and never gain a producer verdict.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from typing import Any

from carnot.reporting import current_contract_readiness_8276 as contract
from carnot.reporting import v713_capstone_evidence as prior
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.reporting.v710_contract_replay import require_reference, snapshot

Json = dict[str, Any]
ROOT = prior.ROOT
NAME = "experiment_8289_v715_capstone"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_v715_capstone_8289.py"
MILESTONE = "2026.10.715"
DESIGN, ACTIVE, STAGED = contract.DESIGN, contract.ACTIVE, contract.STAGED
PROTOCOL, EXECUTION = contract.PROTOCOL, contract.EXECUTION
MODEL_SPECS: list[Json] = []
POLAR_PIN = "sha256:9933e509b6d05112843986d8271455df2194dc74f4000a2e68abbf52e0d04589"
OWNED = [
    "python/carnot/reporting/v715_capstone_evidence.py",
    "python/carnot/reporting/v715_capstone.py",
    CLI,
]
HISTORY = [
    "results/experiment_8261_v713_capstone.json",
    "results/experiment_8275_v714_capstone.json",
    "results/experiment_8259_v713_polarfire_dispatch_qualification.json",
    "results/experiment_8264_v714_evidence_view_canary.json",
    "openspec/change-proposals/research-roadmap-v714-preserved-20261008.md",
    "research-complete.yaml",
    "ops/conductor-log.md",
    "ops/exclusion_manifest.yaml",
    "ops/arc_solve_registry.yaml",
    "research-hardware-wishlist.md",
    "results/experiment_8250_evidence_view_canary.json",
]
NEXT = {
    8276: "Current exact authority and independently qualified component receipts.",
    8277: "A fresh permitted lease with authenticated native CUDA load and cleanup.",
    8278: "A qualified backend and bounded complete three-view canary ledger.",
    8279: "Complete registered fit capture with cold-inclusive acquisition costs.",
    8280: "Complete registered calibration/selection capture without replacement units.",
    8281: "Frozen equal-information heads fit on qualified intervention features.",
    8282: "Qualified held stream and sealed retention captures with original slots.",
    8283: "Independent supported H1 gain with typed-action control and oracle headroom.",
    8284: "Admitted constraints used later on a distinct source under frozen costs.",
    8285: "Independent H2 later decision gain and sealed retention.",
    8286: "Authenticated new ARC outcomes and cross-game arm overlap; repair failed owned check.",
    8287: "Complete current clocks and matched Rust/Python10x; useful k<=5 SSH fabric evidence.",
    8288: "Dated physical setup change, valid GM1Ax IDCODE, flashed n16 sample/hash smoke.",
    8289: "Qualified current science operands and independent closure of each PRD gap.",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual phase boundaries so the conductor can observe unfinished work."""
    print(f"[exp8289] phase={phase} completed={completed} pending={pending}", flush=True)


def bind(path: Path, raw: Path, refs: list[Json]) -> Json:
    """Freeze a primary and its bound checks before granting any evidence credit."""
    ref = snapshot(path, raw / "custody", str(len(refs)))
    refs.append(ref)
    side: Json = dict(exists=False, path=str(path), sha256=None)
    if ref["exists"]:
        side_path = path.parent / "raw" / path.stem / "validators" / (ref["sha256"][7:] + ".json")
        try:
            value = prior.read(ref)
            terminal = Path(value["terminal_validation_sidecar_path"])
            terminal_value = json.loads(terminal.read_bytes())
            side_path = (
                terminal
                if "primary_sha256" in terminal_value
                else Path(terminal_value["publication"]["sidecar_path"])
            )
            refs.append(snapshot(terminal, raw / "custody", str(len(refs))))
            bound = read_bound_sidecar(path, side_path)
            if bound.get("primary_path") != str(path.absolute()):
                raise ValueError("terminal_primary_path")
        except (OSError, ValueError, KeyError, TypeError) as error:
            side["binding_error"] = str(error)
        saved = snapshot(side_path, raw / "custody", str(len(refs)))
        refs.append(saved)
        side = dict(saved, **{k: v for k, v in side.items() if k == "binding_error"})
    return dict(reference=ref, sidecar=side)


def measure(root: Path, raw: Path) -> Json:
    """Authenticate authority and all sources, including independent blocked reports."""
    authority = contract.authority_work(root, raw / "contract")
    tasks = authority["tasks"]
    if len(tasks) != 14 or [t["id"].split("-")[0] for t in tasks] != [
        f"exp{i}" for i in range(8276, 8290)
    ]:
        raise ValueError("exact_fourteen_task_contract")
    refs = authority["refs"]
    inputs = []
    for index, task in enumerate(tasks[:-1]):
        progress("before_input", index, 13 - index)
        path = prior.resolve(task, 8276 + index, root)
        inputs.append(bind(path, raw, refs))
        progress("after_input", index + 1, 12 - index)
    history = {Path(name).stem: bind(root / name, raw, refs) for name in HISTORY[:4]}
    for name in [EXECUTION, *HISTORY[4:]]:
        refs.append(snapshot(root / name, raw / "custody", str(len(refs))))
    polar = history[Path(HISTORY[2]).stem]
    if polar["reference"]["exists"]:
        for key in ["board_reference", "host_reference", "input_reference"]:
            ref = prior.read(polar["reference"]).get(key)
            if ref:
                saved = snapshot(Path(ref["path"]), raw / "custody", str(len(refs)))
                refs.append(dict(saved, expected_sha256=ref["sha256"]))
    return dict(authority=authority, tasks=tasks, inputs=inputs, history=history, references=refs)


def outcome(task: Json, identity: int, item: Json) -> tuple[Json, list[Json], Json]:
    """Reuse qualified schema checks and distinguish a receipt from producer execution."""
    row, failures, source = prior.outcome(task, identity, item)
    if row["disposition"] == "conductor_pre_gate":
        failures = [g for g in failures if g["artifact_field"] != "task_id"]
        row["evidence_type"] = "bound_conductor_pre_gate_receipt"
        row["honest_verdict"] = None
    else:
        row["evidence_type"] = (
            "producer_primary" if row["producer_executed"] else "absent_or_unauthenticated_primary"
        )
    if row["missing"]:
        row["honest_verdict"] = None
    return row, failures, source


def reduce(work: Json, receipts: list[Json]) -> Json:
    """Reconstruct all claims from frozen bytes so rehashed summaries cannot pass.

    Owned checks qualify execution only. Scientific fields remain unavailable when
    the registered audit cannot supply authenticated independent primitives.
    """
    for ref in work["references"]:
        require_reference(ref)
    authority = work["authority"]
    paths = [
        Path(r.get("snapshot_path", "/tmp/absent8289-" + str(i)))
        for i, r in enumerate(authority["refs"][:3])
    ]
    with tempfile.TemporaryDirectory(prefix="carnot8289-authority-") as directory:
        checked = contract.assess(paths, Path(directory))
    tasks = prior.parse_design(paths[0].read_text(), milestone=MILESTONE)[1]
    if tasks != work["tasks"] or len(work["inputs"]) != 13:
        raise ValueError("contract_primitive_drift")
    failures = deepcopy(authority["failures"])
    activated = (
        checked["activated"] and checked["canonical_tasks_sha256"] == canonical_hash(tasks)[7:]
    )
    if not activated:
        failures.append(
            prior.operand(
                "activation",
                authority["refs"][2]["path"],
                authority["refs"][2]["sha256"],
                "full_task_authority_equal",
                True,
                False,
            )
        )
    rows, sources = [], {}
    for identity, task, item in zip(range(8276, 8289), tasks[:-1], work["inputs"]):
        if item["reference"] not in work["references"]:
            raise ValueError("input_reference_drift")
        row, failed, source = outcome(task, identity, item)
        if row["disposition"] == "conductor_pre_gate":
            for gate in prior.read(item["reference"])["gates_evaluated"]:
                upstream: Json = next(
                    (r for r in work["references"] if r["path"] == gate["artifact_path"]), {}
                )
                if upstream.get("sha256") != gate["artifact_sha256"]:
                    failed.append(
                        prior.operand(
                            gate["upstream"],
                            gate["artifact_path"],
                            upstream.get("sha256"),
                            "conductor_receipt_bound_sha256",
                            gate["artifact_sha256"],
                            upstream.get("sha256"),
                        )
                    )
                    row.update(
                        missing=True,
                        disposition="unavailable_output",
                        numerator=0,
                        evidence_type="unbound_conductor_receipt",
                    )
        rows.append(row)
        failures.extend(failed)
        sources[identity] = source
    history = {}
    for name, item in sorted(work["history"].items()):
        ref = item["reference"]
        identity = int(name.split("_")[1])
        task = dict(id=f"exp{identity}-historical")
        if ref["exists"]:
            task["id"] = prior.read(ref).get("task_id", task["id"])
        _, failed, value = outcome(task, identity, item)
        history[identity] = value
        failures.extend(failed)
    polar = history[8259]
    polar_item = work["history"][Path(HISTORY[2]).stem]
    terminal_checks = (
        prior.read(polar_item["sidecar"]).get("report", {}).get("receipts", []) if polar else []
    )
    terminal_ok = {
        r.get("name") for r in terminal_checks if r.get("passed") and r.get("normal_exit")
    } >= {"cold_replay", "adversarial", "strict_rows"}
    primitive_ok = all(
        r["exists"] and r["sha256"] == r["expected_sha256"]
        for r in work["references"]
        if "expected_sha256" in r
    )
    graduated = bool(
        polar.get("required_checks_passed") is True
        and polar_item["reference"]["sha256"] == POLAR_PIN
        and terminal_ok
        and polar.get("flagged_adversarial") is False
        and polar.get("polarfire_workload_validated") is True
        and polar.get("current_device_execution_count", 0) > 0
        and polar.get("host_parity") is True
        and primitive_ok
        and polar.get("board_output_hashes") == polar.get("host_expected_hashes")
        and polar.get("board_output_hashes")
    )
    if not graduated:
        item = work["history"][Path(HISTORY[2]).stem]["reference"]
        failures.append(
            prior.operand(
                "exp8259",
                item["path"],
                item["sha256"],
                "authenticated_board_local_CPU_dispatch_and_hash_parity",
                True,
                False,
            )
        )
    for receipt in receipts:
        if not receipt["passed"] and receipt.get("scope") != "repository_health":
            failures.append(
                prior.operand(
                    receipt["name"],
                    receipt["stdout_path"],
                    receipt["stdout_sha256"],
                    "normal_exit",
                    receipt["expected_exit"],
                    receipt["actual_exit"],
                )
            )
    owned = [r for r in receipts if r.get("scope") == "owned"]
    passed = bool(owned) and all(r["passed"] and r.get("normal_exit", True) for r in owned)
    tools = all(r["passed"] for r in receipts if r.get("scope") == "preconditions")
    ready = int(passed and activated and tools)
    kind = "disqualified" if not passed else "blocked"
    verdict = "complete_" + kind + ("_owned_validation" if not passed else "_upstream_evidence")
    rows.append(
        dict(
            experiment_id=8289,
            task_id=tasks[-1]["id"],
            unit_id=tasks[-1]["id"],
            arm="task_disposition",
            condition="terminal_accounting",
            status="completed",
            metric="owned_execution_readiness",
            numerator=ready,
            denominator=1,
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
            evidence_type="current_capstone",
        )
    )
    protocol = prior.read(authority["refs"][3])
    branches = [
        dict(
            status="blocked_unmeasured",
            alpha=0.025,
            statistics=None,
            intended_count=n,
            completed_count=None,
            audit_experiment_id=i,
            registered_protocol=protocol[k],
            shared_fallback="retain intended slots; paired fallback gain zero only for measured qualified arms",
            selected_comparator=None,
            source_information_gain=None,
            energy_specific_gain=None,
            later_constraint_use=None,
            retention_gain=None,
            failed_operand="eligible_independent_audit_and_primitives",
        )
        for k, i, n in [("H1", 8283, 128), ("H2", 8285, 96)]
    ]
    branches[1]["retention_intended_count"] = 32
    accounting = [
        dict(
            experiment_id=i,
            path=rows[i - 8276]["path"],
            sha256=rows[i - 8276]["sha256"],
            evidence_type=rows[i - 8276]["evidence_type"],
            counts=sources[i].get("model_invocation_counts"),
            spans=sources[i].get("load_spans", sources[i].get("service_phase_spans")),
            acquisition_cost_s=None,
            current_capture_status="unavailable"
            if not sources[i]
            else "authenticated_blocked_no_capture",
            imported_canary_counted_once=(i == 8278),
        )
        for i in [8277, 8278, 8279, 8280, 8282]
    ]
    boards = [
        dict(
            board=b,
            experiment_id=i,
            benefit_score=0,
            obligation=s.get(key),
            historical_transcript_hashes=s.get("frontier_hashes", s.get("board_reference", {})),
            evidence=s,
            terminal_condition=condition,
        )
        for b, i, s, key, condition in [
            (
                "PolarFire",
                8259,
                polar,
                "polarfire_obligation",
                "Authenticated board-local Linux CPU state/query dispatch and output hash parity; no fabric acceleration.",
            ),
            (
                "KV260",
                8287,
                sources[8287],
                "kv260_obligation",
                "Useful implemented k<=5 workload through ssh kria with authenticated fabric transcript and transfer-inclusive clocks; no host storage prerequisite.",
            ),
            (
                "GateMate",
                8288,
                sources[8288],
                "gatemate_obligation",
                "Dated cable/port/power/board/DirtyJTAG change, valid GM1Ax IDCODE, flash then n16 sample/hash parity; no unchanged probe.",
            ),
        ]
    ]
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
        pre_gate_count=sum(r["disposition"] == "conductor_pre_gate" for r in rows),
        missing_output_count=sum(r["missing"] for r in rows),
        capstone_execution_ready_score=ready,
        science_ready_score=0,
        required_checks_passed=passed,
        flagged_adversarial=False,
        H1=branches[0],
        H2=branches[1],
        h1_development_signal_score=0,
        h2_development_signal_score=0,
        verifier_is_oracle=True,
        exposure_scope="exposed development",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        gate_check_summary=failures,
        task_contract=tasks,
        full_task_authority_equal=bool(activated),
        canonical_tasks_sha256=canonical_hash(tasks)[7:],
        activation_snapshot=authority["refs"][2],
        historical_v713=history[8261],
        historical_v714=history[8275],
        historical_CUDA_block=history[8264],
        archive_lag=dict(
            planning_archive_stopped_at="V713",
            v714_authorities="preserved design, active roadmap at planning, primaries and conductor log",
            historical_executed_count=7,
            historical_missing_count=7,
        ),
        evidence_improved=False,
        learning_improved=False,
        evidence_improvement_scope="V715 authority mechanics qualify; backend and science remain blocked. PolarFire graduation is inherited V713 CPU evidence.",
        polarfire_graduation=dict(
            graduated=graduated,
            inherited=True,
            scope="board-local Linux CPU",
            new_scientific_benefit=False,
        ),
        polarfire_terminal_evidence_hashes=[
            r for r in work["references"] if "8259" in r["path"] or "expected_sha256" in r
        ],
        board_obligations=boards,
        arc_evidence=sources[8286],
        live_call_accounting=accounting,
        current_capture_cost_scope=dict(
            status="unavailable",
            complete_acquisition_update_cost_s=None,
            imported_calls_are_new_execution=False,
            current_capstone_calls=0,
        ),
        three_prd_gaps=[
            dict(gap=g, requirements=req, closed=False, moved=False, remaining=NEXT[i])
            for g, req, i in [
                ("useful_verified_decisions", ["FR-06", "FR-12"], 8283),
                ("later_learning_and_retention", ["FR-11"], 8285),
                ("request_scale_deployment", ["FR-05", "FR-08", "NFR-01"], 8287),
            ]
        ],
        retirements=[
            dict(
                task_id=t["id"],
                predecessors=t.get("prior_failures", []),
                decision="carry_anti_churn_obligation" if i >= 8286 else "await_eligible_evidence",
                informative_registered_null=False,
                learnable_control_passed=None,
                oracle_action_headroom=None,
                reopening_condition=NEXT[i],
            )
            for i, t in zip(range(8276, 8290), tasks)
        ],
        acceptance_gates=dict(
            owned_validation=passed,
            full_task_authority=bool(activated),
            terminal_accounting=True,
            independent_science=False,
        ),
    )
