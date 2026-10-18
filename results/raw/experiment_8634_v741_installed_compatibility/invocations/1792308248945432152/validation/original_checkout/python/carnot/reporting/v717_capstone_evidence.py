"""REQ-REPORT-8317: keep administrative completion separate from scientific support.

Qualified readers preserve original bytes and failure scope. Fresh producer
replays reconstruct available primitives; missing work never becomes a zero.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.reporting import v717_contract_methods as contract
from carnot.reporting import v716_capstone_evidence as prior
from carnot.reporting import v709_execution as supervisor
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v710_contract_replay import require_reference, snapshot

Json = dict[str, Any]
ROOT, DESIGN, STAGED, ACTIVE, PROTOCOL = (
    contract.ROOT,
    contract.DESIGN,
    contract.STAGED,
    contract.ACTIVE,
    contract.PROTOCOL,
)
NAME = "experiment_8317_v717_capstone"
TASK, MILESTONE = "exp8317-capstone", "2026.10.717"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v717_capstone_8317.py"
OWNED = [
    "python/carnot/reporting/v717_capstone_evidence.py",
    "python/carnot/reporting/v717_capstone.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
HISTORY = [
    "results/experiment_8303_v716_capstone.json",
    "results/experiment_8259_v713_polarfire_dispatch_qualification.json",
]
failure = contract.failure


def read(ref: Json) -> Json:
    """Malformed external bytes remain an operand failure for the qualified reader."""
    try:
        return dict(prior.prior.read(ref)) if ref["exists"] else {}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real boundaries so the supervisor can observe unfinished work."""
    print(f"[exp8317] phase={phase} completed={completed} pending={pending}", flush=True)


def authority(refs: list[Json], raw: Path) -> Json:
    """Authenticate complete task objects separately from the actual activation copy."""
    if not all(refs[i]["exists"] for i in [0, 2]):
        return dict(
            activated=False,
            gate_check_summary=[
                failure(Path(r["path"]), "authority_available", True, None)
                for r in [refs[0], refs[2]]
                if not r["exists"]
            ],
        )
    return dict(
        contract.authority(
            [Path(r.get("snapshot_path", raw / f"absent-{i}")) for i, r in enumerate(refs[:3])], raw
        )
    )


def measure(root: Path, raw: Path) -> Json:
    """Freeze exact dependencies, their checks and primitive operands before replay."""
    progress("measurement_before", 0, 14)
    start, wall = time.monotonic_ns(), time.time_ns()
    refs = [
        snapshot(root / p, raw / "custody", str(i))
        for i, p in enumerate([DESIGN, STAGED, ACTIVE, PROTOCOL])
    ]
    design = refs[0]
    if not design["exists"]:
        design = snapshot(ROOT / DESIGN, raw / "custody", "identity_contract")
        refs.append(design)
    tasks = parse_design(Path(design["snapshot_path"]).read_text(), milestone=MILESTONE)[1]
    if [t["id"].split("-")[0] for t in tasks] != [f"exp{i}" for i in range(8304, 8318)]:
        raise ValueError("exact_fourteen_task_contract")
    inputs = []
    for index, task in enumerate(tasks[:-1]):
        progress("input_before", index, 13 - index)
        declared = root / task["deliverable"]
        path = prior.prior.resolve(task, 8304 + index, root)
        if path != declared:
            refs.append(snapshot(declared, raw / "custody", str(len(refs))))
        inputs.append(prior.bind(path, raw, refs))
        progress("input_after", index + 1, 12 - index)
    history = {Path(p).stem: prior.bind(root / p, raw, refs) for p in HISTORY}
    previous = read(history[Path(HISTORY[0]).stem]["reference"])
    for task in previous.get("task_contract", []):
        if task["id"] == "exp8302-gatemate-physical-delta":
            path = root / task["deliverable"]
            history[path.stem] = prior.bind(path, raw, refs)
    for ref in previous.get("source_artifact_hashes", []):
        if Path(ref["path"]).name == "experiment_8289_v715_capstone.json":
            path = root / "results" / Path(ref["path"]).name
            history[path.stem] = prior.bind(path, raw, refs)
            history[path.stem]["reference"]["expected_sha256"] = ref["sha256"]
    for item in [*inputs, *history.values()]:
        if not item["reference"]["exists"]:
            continue
        source = read(item["reference"])
        report = read(item["sidecar"]).get("report", {})
        for receipt in report.get("receipts", report.get("checks", [])):
            for stream in ["stdout", "stderr"]:
                if receipt.get(stream + "_path"):
                    refs.append(
                        dict(
                            snapshot(
                                Path(receipt[stream + "_path"]), raw / "custody", str(len(refs))
                            ),
                            expected_sha256=receipt[stream + "_sha256"],
                            bound_owner=item["reference"]["path"],
                        )
                    )
        for key in [
            "measurement_reference",
            "primitive_reference",
            "work_reference",
            "replay_input_reference",
            "board_reference",
            "host_reference",
            "input_reference",
        ]:
            operand = source.get(key)
            if isinstance(operand, dict) and operand.get("path"):
                refs.append(
                    dict(
                        snapshot(Path(operand["path"]), raw / "custody", str(len(refs))),
                        expected_sha256=operand["sha256"],
                    )
                )
        if source.get("activation_snapshot"):
            operand = source["activation_snapshot"]
            refs.append(
                dict(
                    snapshot(Path(operand["snapshot_path"]), raw / "custody", str(len(refs))),
                    expected_sha256=operand["sha256"],
                    evidence_role="historical_activation",
                )
            )
    for p in ["ops/exclusion_manifest.yaml", "ops/arc_solve_registry.yaml"]:
        refs.append(snapshot(root / p, raw / "custody", str(len(refs))))
    plan = []
    for identity, task, item in zip(range(8304, 8317), tasks[:-1], inputs):
        ref = item["reference"]
        script = ROOT / "scripts/experiments" / (Path(task["deliverable"]).stem + ".py")
        if not ref["exists"] or not script.is_file():
            continue
        source = read(ref)
        if not source:
            continue
        if source.get("schema") == "blocked_gate_check_v1":
            continue
        changed = dict(source, intended_count=-1)
        changed["reproducibility_checksum"] = canonical_hash(
            {k: v for k, v in changed.items() if k != "reproducibility_checksum"}
        )
        negative = raw / f"upstream_{identity}_rehashed.json"
        atomic_json(negative, changed)
        for label, path, expected in [
            ("replay", Path(ref["snapshot_path"]), 0),
            ("tamper", negative, 1),
        ]:
            plan.append(
                dict(
                    name=f"branch_{identity}_{label}",
                    argv=[
                        str(ROOT / ".venv/bin/python"),
                        "-u",
                        str(script),
                        "--cold-replay",
                        str(path),
                    ],
                    expected=expected,
                    deadline=180,
                    scope="upstream",
                )
            )
    gate_plan = dict(
        name="publication_gate",
        argv=[str(ROOT / ".venv/bin/python"), "-u", "scripts/publication_gate.py", "--json"],
        expected=0,
        deadline=90,
        scope="publication",
    )
    atomic_json(raw / "branch_manifest.json", dict(commands=plan, publication=gate_plan))
    audits = supervisor.execute(plan, raw / "branches")
    gate = supervisor.execute([gate_plan], raw / "publication")[0]
    work = dict(
        tasks=tasks,
        inputs=inputs,
        history=history,
        references=refs,
        failures=[],
        root=str(root),
        audits=audits,
        publication=gate,
        started_monotonic_ns=start,
        started_wall_ns=wall,
        ended_monotonic_ns=time.monotonic_ns(),
    )
    atomic_json(raw / "measurement.json", work)
    progress("measurement_after", 13, 1)
    return work


def reduce(work: Json, receipts: list[Json]) -> Json:
    """Rebuild all fourteen dispositions without treating missing work as a null."""
    for ref in work["references"]:
        require_reference(ref)
    design = next(r for r in work["references"] if r["exists"] and r["path"].endswith(DESIGN))
    tasks = parse_design(Path(design["snapshot_path"]).read_text(), milestone=MILESTONE)[1]
    if tasks != work["tasks"] or len(work["inputs"]) != 13:
        raise ValueError("contract_primitive_drift")
    with tempfile.TemporaryDirectory(prefix="exp8317-authority-") as directory:
        checked = authority(work["references"], Path(directory))
    rows, sources, failures = [], {}, list(checked["gate_check_summary"]) + work["failures"]
    for ref in work["references"]:
        if "expected_sha256" in ref and (
            not ref["exists"] or ref["sha256"] != ref["expected_sha256"]
        ):
            failures.append(
                prior.prior.operand(
                    ref["path"],
                    ref["path"],
                    ref["sha256"],
                    "primitive_or_validation_sha256",
                    ref["expected_sha256"],
                    ref["sha256"],
                )
            )
    for identity, task, item in zip(range(8304, 8317), tasks[:-1], work["inputs"]):
        if item["reference"] not in work["references"]:
            raise ValueError("input_reference_drift")
        row, failed, source = prior.outcome(task, identity, item)
        if row["disposition"] == "conductor_pre_gate":
            for gate in prior.prior.read(item["reference"])["gates_evaluated"]:
                bound: Json = next(
                    (r for r in work["references"] if r["path"] == gate["artifact_path"]), {}
                )
                if bound.get("sha256") != gate["artifact_sha256"]:
                    row.update(missing=True, disposition="unbound_pre_gate", numerator=0)
                    failed.append(
                        prior.prior.operand(
                            gate["upstream"],
                            gate["artifact_path"],
                            bound.get("sha256"),
                            "conductor_receipt_bound_sha256",
                            gate["artifact_sha256"],
                            bound.get("sha256"),
                        )
                    )
        row["source_sample_size"] = {
            k: source.get(k) for k in ["independent_count", "sample_size_budget"]
        }
        rows.append(row)
        failures.extend(failed)
        sources[identity] = source
    history = {}
    for name, item in work["history"].items():
        identity = int(name.split("_")[1])
        source = read(item["reference"])
        _, failed, authenticated = prior.outcome(
            dict(id=source.get("task_id", "historical")), identity, item
        )
        history[identity] = authenticated
        failures.extend(failed)
    polar_item, polar = work["history"][Path(HISTORY[1]).stem], history[8259]
    side = prior.prior.read(polar_item["sidecar"]) if polar else {}
    terminal = side.get("report", {}).get("receipts", [])
    primitive_ok = all(
        r["exists"] and r["sha256"] == r["expected_sha256"]
        for r in work["references"]
        if "expected_sha256" in r and ("8259" in r["path"] or "8259" in r.get("bound_owner", ""))
    )
    graduated = bool(
        polar
        and polar_item["reference"]["sha256"] == prior.qualified.POLAR_PIN
        and polar.get("polarfire_workload_validated") is True
        and polar.get("required_checks_passed") is True
        and polar.get("flagged_adversarial") is False
        and polar.get("current_device_execution_count", 0) > 0
        and polar.get("host_parity") is True
        and polar.get("board_output_hashes") == polar.get("host_expected_hashes")
        and polar.get("board_output_hashes")
        and primitive_ok
        and {r["name"] for r in terminal if r["passed"] and r["normal_exit"]}
        >= {"cold_replay", "adversarial", "strict_rows"}
    )
    if not graduated:
        ref = polar_item["reference"]
        failures.append(
            prior.prior.operand(
                "exp8259",
                ref["path"],
                ref["sha256"],
                "authenticated_board_local_CPU_dispatch_and_hash_parity",
                True,
                False,
            )
        )
    owned = [r for r in receipts if r.get("scope") == "owned"]
    passed = bool(owned) and all(r["passed"] and r.get("normal_exit", True) for r in owned)
    passed = passed and not any(r["disposition"] == "owned_reader_exception" for r in rows)
    passed = passed and not any(
        r["name"].endswith("_replay")
        and not r["passed"]
        and sources[int(r["name"].split("_")[1])].get("verdict_class") != "disqualified"
        for r in work["audits"]
        if r["name"].startswith("branch_")
    )
    tools = work.get("owned_validation_complete", True)
    ready = int(passed and checked["activated"] and tools)
    for receipt in [*receipts, *work["audits"]]:
        if not receipt["passed"]:
            failures.append(
                prior.prior.operand(
                    receipt["name"],
                    receipt.get("stdout_path"),
                    receipt.get("stdout_sha256"),
                    "normal_exit",
                    receipt.get("expected_exit"),
                    receipt.get("actual_exit"),
                )
            )
    replayed = {r["name"] for r in work["audits"] if r["passed"] and r["normal_exit"]}
    science = bool(
        all(rows[i - 8304]["eligible"] for i in range(8308, 8313))
        and {"branch_8311_replay", "branch_8312_replay"} <= replayed
    )
    kind = "disqualified" if not passed else "blocked" if not science or not tools else "null"
    verdict = (
        "complete_"
        + kind
        + (
            "_owned_validation"
            if not passed
            else "_required_resources"
            if not tools
            else "_upstream_evidence"
            if not science
            else "_unsupported_local_learning"
        )
    )
    rows.append(
        dict(
            experiment_id=8317,
            task_id=TASK,
            unit_id=TASK,
            arm="task_disposition",
            condition="terminal_accounting",
            metric="owned_execution_readiness",
            status="completed",
            completed=True,
            missing=False,
            producer_executed=True,
            eligible=bool(ready),
            excluded=not ready,
            failed=not passed,
            censored=False,
            numerator=ready,
            denominator=1,
            honest_verdict=verdict,
            verdict_class=kind,
            disposition="self_owned_completion",
            evidence_type="current_capstone",
        )
    )
    protocol = prior.prior.read(work["references"][3]) if work["references"][3]["exists"] else {}
    branches = [
        dict(
            status="qualified_development_audit" if science else "blocked_unmeasured",
            alpha=0.025,
            interval_scope="descriptive exposed development only",
            statistics=sources[i].get("statistics") if science else None,
            intended_count=n,
            completed_count=sources[i].get("completed_count") if science else None,
            registered_protocol=protocol.get(key),
            audit_experiment_id=i,
        )
        for key, i, n in [("H1", 8311, 128), ("H2", 8312, 88)]
    ]
    branches[1].update(
        stream_intended_count=96, retention_intended_count=32, retention_windows=[0, 32, 64, 96]
    )
    next_conditions = [
        "Authenticate exact complete current authorities and the frozen protocol.",
        "Preserve all original slots and labels; no replacement or recapture.",
        "Repair owned validation and independently replay exact numerical, issue/release and crash rows.",
        "Authenticate a real driver/device/lease correction then context and copy parity; root cause remains unproved.",
        "Produce frozen equal-information heads with qualified fit controls and complete original support.",
        "Seal every intended reserved prediction before opening evaluator targets.",
        "Measure all delayed local update arms with exact sparse/dense and crash parity.",
        "H1: >=80 sources, >=8 per class, mean gain>=.02, nominal lower bound>0 and Brier degradation<=.01.",
        "H2: >=64 later sources, >=8 per class, mean>=.02, lower bound>0 and qualified fixed retention.",
        "Current changed runtime and context receipts before one bounded canary; no H1 evidence.",
        "New authenticated ARC outcomes and >=3 overlapping games with >=5 firings in each shared arm cell.",
        "Qualified complete local CPU costs and an implemented compatible k<=5 SSH fabric workload.",
        "Dated operator physical change, GM1Ax IDCODE0x20000001, flash and n16 device sample/hash smoke.",
        "New qualified upstream science bytes before another capstone attempt; no unchanged missingness retry.",
    ]
    retirements = []
    for task, row, condition in zip(tasks, rows, next_conditions):
        matched = [
            p
            for p in task["prior_failures"]
            if p["retire_if_same_verdict"] is True and p["verdict"] == row["honest_verdict"]
        ]
        retirements.append(
            dict(
                task_id=task["id"],
                predecessors=deepcopy(task["prior_failures"]),
                same_verdict_entries=matched,
                prior_evidence=[
                    dict(
                        task_id=p["experiment_id"],
                        authenticated=any(
                            s.get("task_id") == p["experiment_id"]
                            and s.get("honest_verdict") == p["verdict"]
                            for s in history.values()
                        ),
                        references=[
                            item["reference"]
                            for item in work["history"].values()
                            if read(item["reference"]).get("task_id") == p["experiment_id"]
                        ],
                    )
                    for p in matched
                ],
                decision="retire_exact_repeated_scope" if matched else "await_eligible_evidence",
                scope="unchanged evidence/probe only; unmeasured scientific hypothesis remains open",
                permanent=bool(matched),
                reopening_condition=condition,
            )
        )
    canary = sources[8313]
    local = int(
        bool(
            sources[8306].get("local_kernel_ready_score") == 1
            and rows[2]["eligible"]
            and "branch_8306_replay" in replayed
        )
    )
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
        required_checks_passed=passed,
        flagged_adversarial=False,
        capstone_execution_ready_score=ready,
        current_contract_ready_score=ready,
        science_ready_score=int(science),
        H1=branches[0],
        H2=branches[1],
        h1_development_signal_score=0,
        h2_development_signal_score=0,
        local_mechanics_ready_score=local,
        local_sentence_decisions_changed=None
        if not science
        else sources[8311].get("decisions_changed"),
        local_update_decisions_changed=None
        if not science
        else sources[8312].get("decisions_changed"),
        verifier_is_oracle=False,
        exposure_scope="exposed_cached_development",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        gate_check_summary=failures,
        task_contract=tasks,
        canonical_tasks_sha256=canonical_hash(tasks)[7:],
        full_task_authority_equal=checked["activated"],
        activation_snapshot=work["references"][2],
        branch_replay_receipts=work["audits"],
        historical_v716={
            k: history[8303].get(k)
            for k in [
                "honest_verdict",
                "actual_executed_task_count",
                "pre_gate_count",
                "missing_output_count",
                "activation_snapshot",
            ]
        },
        polarfire_graduation=dict(
            graduated=graduated,
            inherited=True,
            scope="board-local Linux CPU",
            fpga_fabric_acceleration=False,
            new_scientific_benefit=False,
        ),
        polarfire_terminal_evidence_hashes=[r for r in work["references"] if "8259" in r["path"]],
        board_obligations=[
            dict(
                board=b,
                experiment_id=i,
                obligation=s.get(key),
                met=graduated if i == 8259 else False,
                next_evidence_condition=condition,
            )
            for b, i, s, key, condition in [
                (
                    "PolarFire",
                    8259,
                    polar,
                    "polarfire_obligation",
                    "Preserve authenticated board-local Linux CPU parity.",
                ),
                ("KV260", 8315, sources[8315], "kv260_obligation", next_conditions[11]),
                ("GateMate", 8316, sources[8316], "gatemate_obligation", next_conditions[12]),
            ]
        ],
        arc_evidence={
            k: sources[8314].get(k)
            for k in [
                "honest_verdict",
                "new_outcome_count",
                "credited_new_levels",
                "arm_overlap_games",
                "current_game_execution_count",
            ]
        },
        live_call_accounting=dict(
            current_capstone_calls=0,
            canary=dict(
                producer_executed=rows[9]["producer_executed"],
                evidence_type=rows[9]["evidence_type"],
                counts=canary.get("model_invocation_counts"),
                tokens=canary.get("token_counts"),
                h1_evidence=False,
            ),
            historical_receipts_counted_once=True,
            imported_calls_are_current_execution=False,
        ),
        historical_model_provenance=sources[8305].get("historical_model_provenance", []),
        historical_capture_counts=sources[8305].get("historical_capture_counts"),
        custody_support=sources[8305].get("class_support_by_role"),
        parked_v713_source_deletion=dict(
            status="parked_unmeasured", measured=False, canary_is_h1=False
        ),
        retirements=retirements,
        next_evidence_conditions=next_conditions,
        three_prd_gaps=[
            dict(gap=g, requirements=req, closed=False, next_evidence_condition=c)
            for g, req, c in [
                ("useful_verified_decisions", ["FR-06", "FR-12"], next_conditions[7]),
                ("later_learning_and_retention", ["FR-11"], next_conditions[8]),
                ("request_scale_deployment", ["FR-05", "FR-08", "NFR-01"], next_conditions[11]),
            ]
        ],
        acceptance_gates=dict(
            owned_validation=passed,
            full_task_authority=checked["activated"],
            terminal_accounting=True,
            independent_science=science,
        ),
        sample_size_budget=dict(
            task_slots=14,
            independent_scientific_observations=0,
            H1=128,
            H2_later=88,
            stream=96,
            retention=32,
        ),
    )


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Bind reduction, commands and clocks to immutable code and primitive bytes."""
    atomic_json(raw / "measurement.json", work)
    value = reduce(work, receipts)
    code = [
        snapshot(ROOT / p, raw / "code", str(i))
        for i, p in enumerate(
            [
                *OWNED,
                TEST,
                "python/carnot/reporting/v717_contract_runner.py",
                "python/carnot/reporting/v717_contract_methods.py",
                "python/carnot/reporting/v716_capstone_evidence.py",
                "python/carnot/reporting/primary_publication.py",
                "python/carnot/reporting/roadmap_contract.py",
                "python/carnot/reporting/v709_execution.py",
                "scripts/experiment_template.py",
                "scripts/publication_gate.py",
                "scripts/adversarial_verify.py",
                "scripts/verdict_row_consistency_lint.py",
                "AGENTS.md",
                "CODEX.md",
                "CLAUDE.md",
                "ops/e2e-test-plan.md",
                "openspec/capabilities/research-reporting/spec.md",
                "openspec/capabilities/verification/spec.md",
            ]
        )
    ]
    gate = work["publication"]
    publication = json.loads(Path(gate["stdout_path"]).read_bytes())
    value.update(
        experiment_id=8317,
        task_id=TASK,
        milestone=MILESTONE,
        run_date="20261008",
        schema="carnot.v717.capstone.v1",
        random_seed=7178317,
        no_model_load=True,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        call_ledger=[],
        duration_s=(work["ended_monotonic_ns"] - work["started_monotonic_ns"]) / 1e9,
        preconditions_checked=work["references"],
        validation_receipts=receipts,
        invocation_argv=work.get("invocation_argv", []),
        execution_manifest_reference=work.get("execution_manifest_reference"),
        owned_coverage_reference=work.get("owned_coverage_reference"),
        phase_spans=[
            dict(
                phase="aggregation",
                started_monotonic_ns=work["started_monotonic_ns"],
                ended_monotonic_ns=work["ended_monotonic_ns"],
                duration_s=(work["ended_monotonic_ns"] - work["started_monotonic_ns"]) / 1e9,
            ),
            *[
                dict(
                    phase=r["name"],
                    started_monotonic_ns=r["started_monotonic_ns"],
                    ended_monotonic_ns=r["ended_monotonic_ns"],
                    duration_s=r["duration_s"],
                )
                for r in [*receipts, *work["audits"], gate]
            ],
        ],
        measurement_clocks={
            k: work[k] for k in ["started_monotonic_ns", "ended_monotonic_ns", "started_wall_ns"]
        },
        source_artifact_hashes=work["references"],
        code_config_hashes=code,
        work_reference=dict(
            path=str(raw / "measurement.json"), sha256=sha256_file(raw / "measurement.json")
        ),
        raw_shard_hashes=[
            dict(path=str(p), sha256=sha256_file(p))
            for p in sorted(raw.rglob("*"))
            if p.is_file() and p.name != "measurement.json"
        ],
        terminal_validation_sidecar_path=str(
            output.parent / "raw" / output.stem / "terminal_validation.json"
        ),
        publication_gate_receipt=gate,
        paper_ready=publication["paper_ready"],
        unmet_gates=publication["unmet_gates"],
        **{k.lower(): publication["gates"][k] for k in ["G1", "G2", "G3", "G4"]},
        cited_upstream_artifacts=[
            dict(
                path=i["reference"]["path"],
                sha256=i["reference"]["sha256"],
                task_id=t["id"],
                fields_imported=[
                    "rows",
                    "honest_verdict",
                    "verdict_class",
                    "model_invocation_counts",
                    "required_checks_passed",
                    "sample_size_budget",
                    "gate_check_summary",
                ],
            )
            for t, i in zip(work["tasks"][:-1], work["inputs"])
        ],
        methodology_note="Spline energy is the logistic re-expression of the same basis. No architecture-specific truth or generalization follows. Fixture mechanics, unavailable science and external blocks retain separate scope.",
        external_publication_authorized=False,
    )
    value["field_principles"] = {
        k: "Bind actual invocation bytes and scope; missing observations remain unavailable; audit readiness grants no scientific benefit."
        for k in value
    }
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """A fresh reader recomputes primitives instead of trusting a rehashed aggregate."""
    try:
        value = json.loads(path.read_bytes())
        checksum = value.pop("reproducibility_checksum")
        if checksum != canonical_hash(value):
            return False
        if (value["experiment_id"], value["task_id"], value["milestone"]) != (
            8317,
            TASK,
            MILESTONE,
        ):
            return False
        for ref in [
            value["work_reference"],
            *value["source_artifact_hashes"],
            *value["code_config_hashes"],
            *value["raw_shard_hashes"],
        ]:
            require_reference(ref)
        work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
        if work["references"] != value["source_artifact_hashes"]:
            return False
        if any(value[k] != v for k, v in reduce(work, value["validation_receipts"]).items()):
            return False
        if value["MODEL_SPECS"] or value["model_invocation_counts"] != ZERO_INVOCATION_COUNTS:
            return False
        for receipt in [*value["validation_receipts"], *work["audits"], work["publication"]]:
            for stream in ["stdout", "stderr"]:
                if sha256_file(Path(receipt[stream + "_path"])) != receipt[stream + "_sha256"]:
                    return False
        gate = json.loads(Path(work["publication"]["stdout_path"]).read_bytes())
        expected = dict(
            paper_ready=gate["paper_ready"],
            unmet_gates=gate["unmet_gates"],
            **{k.lower(): gate["gates"][k] for k in ["G1", "G2", "G3", "G4"]},
        )
        return all(value[k] == v for k, v in expected.items())
    except (OSError, ValueError, KeyError, TypeError):
        return False
