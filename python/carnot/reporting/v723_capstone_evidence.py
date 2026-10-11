"""REQ-REPORT-8401: preserve scheduled units without inventing scientific evidence."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import importlib
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from typing import Any, Iterator
from unittest.mock import patch

from carnot.reporting import v721_capstone_evidence as base
from carnot.reporting import v722_capstone_evidence as previous
from carnot.reporting import v723_contract_methods as contract
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.evidence_features_custody_7980 import checked, reference

Json = dict[str, Any]
ROOT, DESIGN, STAGED, ACTIVE = contract.ROOT, contract.DESIGN, contract.STAGED, contract.ACTIVE
PROTOCOL, PROTOCOL_PIN = contract.PROTOCOL, contract.PROTOCOL_PIN
NAME, TASK, MILESTONE = "experiment_8401_v723_capstone", "exp8401-capstone", "2026.10.723"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v723_capstone_8401.py"
OWNED = [
    "python/carnot/reporting/v723_capstone_evidence.py",
    "python/carnot/reporting/v723_capstone.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
freeze, load, gate = base.freeze, base.load, base.gate
HOST_MEMORY = base.memory
RECEIPTS = {
    8392: "results/experiment_8392_continuous_direct_learning.json",
    8393: "results/experiment_8393_python_transaction_cost.json",
    8395: "results/experiment_8395_label_criterion_audit.json",
    8397: "results/experiment_8397_bounded_qwen_canary.json",
}
CASCADE = "| 2026-10-10 23:05 UTC | Compare native and Python transactions with matche | GATE_BLOCK | Pre-emptive skip: upstream retired (exp8393-python-transaction-cost) |"
READERS = {
    8388: "v723_contract_methods",
    8389: "human_label_custody_8389",
    8399: "board_operation_boundary_8399",
    8400: "gatemate_continuity_8400",
}
NEXT = [
    "Qualify historical readers only with changed scoped repair and fresh receipts; preserve missing V722 design and original failures.",
    "Release at least80 disjoint independent human-labeled question clusters, eight per class, with sealed target and extraction criteria.",
    "Repair owned direct-state qualification and cold replay exact pending, acknowledgment, reader and fsync controls before training.",
    "Repair native coverage topology and fresh source-bound replay before granting native parity readiness.",
    "After direct state qualifies, measure all96 delayed slots including22 missing, three arms, seeds11/22/33 and retention0/32/64/96.",
    "After direct state qualifies, measure complete scoring, updates, serialization, transfer, fsync and response costs for every request.",
    "After Python costs and native qualification, measure matched complete native transactions and all intended operations.",
    "After sealed labels qualify, measure independent target errors and extraction validity without oracle substitution.",
    "Repair owned CUDA causal qualification; require hash-bound environment change and a successful bounded context/copy probe.",
    "After typed reader, runtime change and CUDA context qualify, run the bounded source-independent Qwen canary.",
    "Repair owned ARC validation; require authenticated adapter-withheld cross-game supervisor outcomes with overlapping arm support.",
    "Supply complete native/Python operation costs; retain KV260 quadratic Ising k<=5 and PolarFire board-local Linux CPU scope.",
    "Recover both exact source hashes and dated physical cable/port/power change, correct IDCODE0x20000001, n16 flash and device smoke.",
    "Continue only changed qualified evidence for the three separate PRD gaps; no identical blocked retry or external publication.",
]


def memory() -> Json:
    """VmHWM resets at exec, so earlier parent allocations cannot inflate a fresh worker."""
    status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines())
    return dict(
        current_rss_mb=int(status["VmRSS"].split()[0]) / 1024,
        peak_rss_mb=int(status["VmHWM"].split()[0]) / 1024,
        prior_exec_lifetime_peak_mb=HOST_MEMORY()["peak_rss_mb"],
        measurement_source="/proc/self/status",
    )


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush counts so the conductor can distinguish bounded work from a stall."""
    print(f"[exp8401] phase={phase} completed={completed} pending={pending}", flush=True)


@contextmanager
def binding() -> Iterator[None]:
    """Reuse the shipped workers while giving each fresh process this task's identity."""
    with patch.multiple(
        base,
        contract=contract,
        DESIGN=DESIGN,
        STAGED=STAGED,
        ACTIVE=ACTIVE,
        NAME=NAME,
        TASK=TASK,
        MILESTONE=MILESTONE,
        CLI=CLI,
        TEST=TEST,
        OWNED=OWNED,
        NEXT=NEXT,
        progress=progress,
        outcome=outcome,
        reduce=reduce,
        memory=memory,
    ):
        yield


def outcome(task: Json, primary: Json, raw: Path, sealed: list[Json] | None = None) -> Json:
    """Seal one source closure; a blocked producer may still have an honest qualified reader."""
    with base.frozen_inputs(sealed, raw / "sealed_reads") if sealed else nullcontext():
        result = previous.outcome(task, primary, raw, sealed)
    value = load(primary)
    if value.get("blocked_at_layer") == "conductor_pre_gate":
        observed = value["gates_evaluated"]
        declared = task["gated_on"]
        if [(g["upstream"], g["artifact_field"], g["op"], g["expected"]) for g in observed] != [
            (g["upstream"], g["artifact_field"], g["op"], g["value"]) for g in declared
        ]:
            raise ValueError("declared_gate_contract")
        for g in observed:
            ref = next((r for r in result["closure"] if r["path"] == g["artifact_path"]), None)
            if ref is None:
                ref = freeze(Path(g["artifact_path"]), raw / "pregates", g["artifact_sha256"])
                result["closure"].append(ref)
            actual = load(ref)
            if actual.get(g["artifact_field"]) != g["actual"]:
                raise ValueError("gate_observed_value")
    number = int(task["id"][3:7])
    if result["producer_executed"]:
        result["selected"].update(
            {
                k: v
                for k, v in value.items()
                if k.endswith("_score")
                or k.startswith("coverage_")
                or k in ["owned_coverage", "adversarial_findings"]
                or k
                in [
                    "rows",
                    "intended_count",
                    "completed_count",
                    "failed_count",
                    "censored_count",
                    "excluded_count",
                    "independent_count",
                    "sample_size_budget",
                    "exposure_scope",
                    "verifier_is_oracle",
                    "acceptance_gates",
                    "sealed_label_criteria",
                    "coverage_topology",
                    "required_checks_passed",
                    "flagged_adversarial",
                ]
            }
        )
        if (
            value["required_checks_passed"]
            and value["flagged_adversarial"] is False
            and number in READERS
        ):
            source = ROOT / "python/carnot/reporting" / (READERS[number] + ".py")
            reader_reference = next(
                r
                for r in value["code_config_hashes"]
                if r.get("original_path", r.get("source_path", r["path"])) == str(source)
            )
            pin = reader_reference["sha256"]
            for ref in result["closure"]:
                if ref["sha256"] == pin:
                    ref["source_path"] = str(source)
            checked(dict(path=str(source), sha256=pin))
            with TemporaryDirectory(prefix="exp8401-reader-", dir="/var/tmp") as directory:
                with base.frozen_inputs(result["closure"], Path(directory)):
                    passed = importlib.import_module("carnot.reporting." + READERS[number]).replay(
                        Path(primary["snapshot_path"])
                    )
            result["branch_replay"] = dict(
                status="cold_replay_passed" if passed else "disqualified_cold_replay",
                passed=bool(passed),
            )
    return dict(result)


def slot(task: Json, primary: Json, cascade: str | None, summary: Json | None = None) -> Json:
    """An explicit receipt is scheduling evidence; absent measurements stay null."""
    if primary["exists"]:
        state = summary or outcome(task, primary, Path(primary["snapshot_path"]).parent)
        disposition = "pre_gate_receipt" if not state["producer_executed"] else "executed_producer"
    else:
        state = previous.absent(task, primary, cascade)
        disposition = "cascade_skip" if cascade else "absent_primary"
    return dict(
        state,
        disposition=disposition,
        task_id=task["id"],
        experiment_id=int(task["id"][3:7]),
        unit_id=task["id"],
        task=task,
        primary_reference=primary,
        metrics=state["selected"] if state["producer_executed"] else None,
        missing_reason=None if state["producer_executed"] else disposition,
        continuation_condition=NEXT[int(task["id"][3:7]) - 8388],
    )


def invoke(
    task: Json, primary: Json, raw: Path, closure: list[Json] | None = None
) -> tuple[Json, Json]:
    """Every branch has its own deadline, log hashes and fresh memory budget."""
    with binding():
        summary, receipt = base.invoke(task, primary, raw, closure)
    return dict(summary), dict(receipt)


def worker_process(request: Path, output: Path) -> int:
    """Authentication failures retain the same measured1500/500 MiB bounds."""
    with binding():
        return int(base.worker_process(request, output))


def measure(root: Path, raw: Path) -> Json:
    """Bind all scheduled inputs once; future producers remain dependencies."""
    progress("preconditions_before", 0, 14)
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    start, before = time.monotonic_ns(), memory()
    auth = contract.authority(root, raw / "authority")
    tasks = auth["tasks"]
    if (
        [t["id"].split("-")[0] for t in tasks] != [f"exp{i}" for i in range(8388, 8402)]
        or tasks[-1]["id"] != TASK
        or tasks[-1].get("gated_on")
    ):
        raise ValueError("exact_fourteen_authority")
    refs = [freeze(root / p, raw / "authority_custody") for p in [DESIGN, STAGED, ACTIVE]]
    protocol = freeze(root / PROTOCOL, raw / "custody", PROTOCOL_PIN)
    mounts = [line.split() for line in Path("/proc/self/mountinfo").read_text().splitlines()]
    mount = max(
        (m for m in mounts if raw.resolve().is_relative_to(Path(m[4]))), key=lambda m: len(m[4])
    )
    available = (
        int(
            next(
                line.split()[1]
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            )
        )
        * 1024
    )
    checks = [
        gate(str(root / DESIGN), "authority.activated", True, auth["activated"]),
        gate(str(root / PROTOCOL), "protocol.sha256", PROTOCOL_PIN, protocol["sha256"]),
        gate(
            str(raw),
            "private_disk_backed_scratch",
            True,
            mount[mount.index("-") + 1] not in ["tmpfs", "ramfs"]
            and raw.stat().st_mode & 0o077 == 0,
        ),
        gate(
            str(raw), "disk_available_at_least_1GiB", True, shutil.disk_usage(raw).free >= 1024**3
        ),
        gate("/proc/meminfo", "available_memory_at_least_1GiB", True, available >= 1024**3),
    ]
    checks.extend(
        gate(
            str(ROOT / ".venv/bin" / tool),
            "required_tool_executable",
            True,
            os.access(ROOT / ".venv/bin" / tool, os.X_OK),
        )
        for tool in ["python", "pytest", "coverage", "ruff", "mypy"]
    )
    conductor = freeze(root / "ops/conductor-log.md", raw / "custody")
    lines = Path(conductor["snapshot_path"]).read_text().splitlines() if conductor["exists"] else []
    slots, inputs, receipts = [], [], []
    for index, task in enumerate(tasks[:-1]):
        progress("producer_before", index, 13 - index)
        number = int(task["id"][3:7])
        primary = freeze(root / RECEIPTS.get(number, task["deliverable"]), raw / "custody")
        summary = None
        if primary["exists"]:
            summary, receipt = invoke(task, primary, raw / "branches" / task["id"])
            receipts.append(receipt)
        row = slot(task, primary, CASCADE if number == 8394 and CASCADE in lines else None, summary)
        slots.append(row)
        inputs.append(dict(task=task, primary=primary, summary=row))
        progress("producer_after", index + 1, 12 - index)
    history, extra = [], [conductor, freeze(root / "ops/exclusion_manifest.yaml", raw / "custody")]
    for name in [
        "results/experiment_8361_v721_utility_audit_qualification.json",
        "results/experiment_8381_v722_logit_policy_certificate.json",
        "results/experiment_8387_v722_capstone.json",
    ]:
        path = root / name
        if path.exists():
            item = base.historical(path, raw / "history")
            if item["experiment_id"] == 8361:
                from carnot.reporting.roadmap_contract import parse_design

                design = (
                    root / "openspec/change-proposals/research-roadmap-v721-preserved-20261010.md"
                )
                prior = parse_design(design.read_text(), milestone="2026.10.721")[1][1]
                summary, receipt = invoke(prior, item["reference"], raw / "utility")
                receipts.append(receipt)
                item.update(
                    selected=summary["selected"],
                    refs=summary["closure"],
                    utility_replay=summary["branch_replay"],
                )
                extra.append(freeze(design, raw / "custody"))
            history.append(item)
    after = memory()
    work = dict(
        root=str(root),
        tasks=tasks,
        authority_refs=refs,
        canonical_tasks_sha256=canonical_hash(tasks),
        protocol_reference=protocol,
        slots=slots,
        inputs=inputs,
        history=history,
        extra_refs=extra,
        checks=checks,
        branch_receipts=receipts,
        started_monotonic_ns=start,
        ended_monotonic_ns=time.monotonic_ns(),
        memory_measurements=dict(
            parent_before=before,
            parent_after=after,
            parent_growth_mb=max(0, after["current_rss_mb"] - before["current_rss_mb"]),
            parent_measurement_scope="current_RSS; prior_host_peak_is_not_current_parent_usage",
            peak_bound_mb=1500,
            growth_bound_mb=500,
            workers=[r.get("memory", dict(passed=True)) for r in slots],
        ),
    )
    atomic_json(raw / "measurement.json", work)
    progress("measurement_after", 13, 1)
    return work


def reduce(work: Json, receipts: list[Json]) -> Json:
    """Qualification and scientific benefit are separate reductions over all fourteen units."""
    with binding():
        auth = base.authority(work)
    if work["tasks"] != auth["tasks"] or work["canonical_tasks_sha256"] != canonical_hash(
        auth["tasks"]
    ):
        raise ValueError("full_task_authority")
    owned = bool(receipts) and all(r["passed"] for r in receipts if r.get("scope") != "global")
    owned &= (
        work["memory_measurements"]["parent_growth_mb"] <= 500
        and work["memory_measurements"]["parent_after"]["current_rss_mb"] <= 1500
    )
    owned &= all(r.get("memory", dict(passed=True))["passed"] for r in work["slots"]) and all(
        r["passed"] for r in work["branch_receipts"]
    )
    rows = [
        {
            k: v
            for k, v in row.items()
            if k not in ["task", "closure", "selected", "checks", "memory"]
        }
        for row in work["slots"]
    ]
    kind = (
        "disqualified"
        if not owned
        else "blocked"
        if not all(g["passed"] for g in work["checks"]) or any(not r["eligible"] for r in rows)
        else "null"
    )
    rows.append(
        dict(
            experiment_id=8401,
            task_id=TASK,
            unit_id=TASK,
            disposition="current_capstone",
            producer_executed=True,
            eligible=owned,
            honest_verdict="complete_" + kind + "_v723_capstone",
            verdict_class=kind,
            primary_reference=None,
            metrics=None,
            missing_reason=None,
            continuation_condition=NEXT[-1],
        )
    )
    utility: Json = next(
        (
            h.get("selected", {})
            for h in work["history"]
            if h["experiment_id"] == 8361
            and h["authenticated"]
            and h.get("utility_replay", {}).get("passed")
        ),
        {},
    )
    historical = [
        dict(
            task_id=h["task_id"],
            honest_verdict=h["honest_verdict"],
            verdict_class=h["verdict_class"],
            authenticated=h["authenticated"],
            readiness_imported=False,
            reference=h["reference"],
        )
        for h in work["history"]
    ]
    g = work["checks"] + [g for r in work["slots"] for g in r["checks"] if not g["passed"]]
    g.extend(
        gate(
            r.get("stdout_path", "owned_validation"), r["name"], True, False, r.get("stdout_sha256")
        )
        for r in receipts
        if not r["passed"] and r.get("scope") != "global"
    )
    gates = [
        dict(
            check=a.get("check", a.get("artifact_field")),
            upstream=a.get("upstream", a.get("upstream_id", TASK)),
            path=a.get("path", a.get("artifact_path")),
            sha256=a.get("sha256", a.get("artifact_hash")),
            artifact_field=a.get("artifact_field"),
            operator=a.get("op", "=="),
            expected=a.get("expected"),
            observed=a.get("observed", a.get("actual")),
            passed=a["passed"],
        )
        for a in g
    ]
    return dict(
        rows=rows,
        task_dispositions=rows,
        honest_verdict=rows[-1]["honest_verdict"],
        verdict_class=kind,
        completed_count=sum(r["eligible"] and r["verdict_class"] != "disqualified" for r in rows),
        failed_count=sum(r["verdict_class"] == "disqualified" for r in rows),
        excluded_count=sum(r["disposition"] in ["pre_gate_receipt", "cascade_skip"] for r in rows),
        censored_count=sum(
            not r["eligible"]
            and r["verdict_class"] != "disqualified"
            and r["disposition"] not in ["pre_gate_receipt", "cascade_skip"]
            for r in rows
        ),
        required_checks_passed=owned,
        flagged_adversarial=not owned,
        capstone_execution_ready_score=int(owned),
        science_ready_score=int(bool(utility)),
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        gate_check_summary=gates,
        actual_executed_task_count=sum(r["producer_executed"] for r in rows),
        pre_gate_count=sum(r["disposition"] == "pre_gate_receipt" for r in rows),
        cascade_skip_count=sum(r["disposition"] == "cascade_skip" for r in rows),
        missing_output_count=sum(not r["primary_reference"]["exists"] for r in rows[:-1]),
        H1=utility.get("H1", dict(qualified=False, intended_count=128)),
        H2=utility.get("H2", dict(qualified=False, intended_count=88)),
        historical_dispositions=historical,
        qualified_scope_decision=utility.get("qualified_scope_decision", {}),
        retirements=[],
        next_evidence_conditions=NEXT,
        three_prd_gaps=[
            dict(gap=name, closed=False, next_evidence_condition=NEXT[index])
            for name, index in [
                ("independent_target_and_extraction_validity", 7),
                ("actual_causal_learning_and_recovery", 4),
                ("complete_native_performance", 6),
            ]
        ],
    )


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """The established terminal schema binds this milestone to exact primitive operands."""
    with binding():
        value = base.build(work, receipts, raw, output)
    slots = work["slots"]
    qualification = (
        load(slots[0]["primary_reference"]) if slots[0]["primary_reference"]["exists"] else {}
    )
    health = qualification.get("repository_health", [])
    for receipt in health + qualification.get("historical_reproduction_receipts", []):
        for stream in ["stdout", "stderr"]:
            checked(
                dict(path=str(ROOT / receipt[stream + "_path"]), sha256=receipt[stream + "_sha256"])
            )
    historical_capstone: Json = next(
        (load(h["reference"]) for h in work["history"] if h["experiment_id"] == 8387), {}
    )
    failures = []
    for receipt in historical_capstone.get("validation_receipts", []):
        if receipt["name"] == "full_contract_mutations":
            checked(dict(path=receipt["stdout_path"], sha256=receipt["stdout_sha256"]))
            failures = [
                line
                for line in Path(receipt["stdout_path"]).read_text().splitlines()
                if line.startswith("FAILED ")
            ]
    rows = value["rows"]
    excluded = sum(r["disposition"] in ["pre_gate_receipt", "cascade_skip"] for r in rows)
    failed = sum(r["verdict_class"] == "disqualified" for r in rows)
    completed = sum(r["eligible"] and r["verdict_class"] != "disqualified" for r in rows)
    value.update(
        experiment_id=8401,
        run_date="20261011",
        random_seed=7238401,
        completed_count=completed,
        failed_count=failed,
        excluded_count=excluded,
        censored_count=14 - completed - failed - excluded,
        cascade_skip_count=value["cascade_skip_count"],
        repository_health=dict(
            status="authenticated_unchanged_baseline_reused; no_current_global_pass_claimed",
            source=slots[0]["primary_reference"],
            receipts=health,
            current_suite_launched=False,
        ),
        historical_replay_qualification=dict(
            source=slots[0]["primary_reference"],
            historical_replay_ready_score=qualification.get("historical_replay_ready_score"),
            receipts=qualification.get("historical_reproduction_receipts", []),
            original_v722_failures=failures,
            original_v722_verdict=historical_capstone.get("honest_verdict"),
            original_v722_disqualification_preserved=True,
        ),
        dyadic_logit_certificate=dict(
            reference=next(
                (h["reference"] for h in work["history"] if h["experiment_id"] == 8381), None
            ),
            separate_numerical_policy=True,
            deployed=False,
            semantic_benefit=False,
        ),
        memory_bounds=dict(child_peak_mb=1500, growth_mb=500, parent_current_rss_mb=1500),
        methodology_note="Read-only aggregation binds fourteen full-task slots, exact conductor receipts and immutable source closures. Fresh bounded workers preserve original disqualifications and readiness. Qualified exposed-development H1/H2 remain closed under unchanged thresholds. Independent human targets, actual causal learning/recovery and complete native performance remain separate open gaps. The dyadic-logit policy is separate and undeployed. Current LLM calls are zero; no weights or device state change.",
    )
    value["source_artifact_hashes"].extend(work["extra_refs"])
    value["cited_upstream_artifacts"].extend(
        dict(r, fields_imported=["sealed conductor, exclusion prefix or historical authority"])
        for r in work["extra_refs"]
    )
    value["cited_upstream_artifacts"].extend(
        dict(row["primary_reference"], fields_imported=list(row["metrics"]))
        for row in slots
        if row["metrics"] is not None
    )
    value["field_principles"].update(
        {
            k: "Bind "
            + k
            + " to frozen source bytes and independently recomputed accounting; preserve missing evidence and original failures."
            for k in value
        }
    )
    value["field_principles"].update(
        repository_health="Authenticate prior global-health logs separately; an unchanged timeout is no current repository-wide pass.",
        retirements="No exact prior scope equality was qualified; no retirement or manifest append is authorized by these observations.",
        historical_replay_qualification="Exp8388 receipts qualify only their tested scope; retain both original named V722 assertion failures.",
    )
    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    return dict(value)


def replay(path: Path) -> bool:
    """Cold primitive reads reject fabricated rows, gates, receipts and rehashed summaries."""
    try:
        value = json.loads(path.read_bytes())
        if value["reproducibility_checksum"] != canonical_hash(
            {k: v for k, v in value.items() if k != "reproducibility_checksum"}
        ):
            return False
        work = load(value["work_reference"])
        for ref in value["source_artifact_hashes"] + value["code_config_hashes"]:
            if ref["exists"]:
                checked(dict(path=ref["snapshot_path"], sha256=ref["sha256"]))
        for receipt in value["validation_receipts"] + work["branch_receipts"]:
            for stream in ["stdout", "stderr"]:
                checked(dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"]))
            original = json.loads(
                Path(receipt["stdout_path"]).with_suffix(".receipt.json").read_bytes()
            )
            if any(
                original.get(k) != receipt.get(k)
                for k in ["argv", "exit_code", "passed", "stdout_sha256", "stderr_sha256"]
            ):
                return False
        lines = (
            Path(work["extra_refs"][0]["snapshot_path"]).read_text().splitlines()
            if work["extra_refs"][0]["exists"]
            else []
        )
        with TemporaryDirectory(prefix="exp8401-cold-", dir="/var/tmp") as directory:
            raw = Path(directory)
            actual = build(
                work, value["validation_receipts"], raw, Path(value["publication_output"])
            )
            for key in ["work_reference", "raw_shard_hashes"]:
                actual[key] = value[key]
            actual["reproducibility_checksum"] = canonical_hash(
                {k: v for k, v in actual.items() if k != "reproducibility_checksum"}
            )
            if actual != value:
                return False
            for h in work["history"]:
                primary, terminal, side = [load(r) for r in h["refs"][:3]]
                if h["honest_verdict"] != primary["honest_verdict"] or h["authenticated"] != (
                    terminal["publication"]["primary_sha256"]
                    == h["reference"]["sha256"]
                    == side["primary_sha256"]
                    and side["report"]["passed"] is True
                ):
                    return False
                if h["experiment_id"] == 8361 and base.utility(primary, h["refs"]) != {
                    k: h["selected"][k]
                    for k in ["H1", "H2", "original_dispositions", "qualified_scope_decision"]
                }:
                    return False
            for index, row in enumerate(work["slots"]):
                if row["task"] != work["tasks"][index]:
                    return False
                primary = row["primary_reference"]
                summary = None
                if primary["exists"]:
                    summary, receipt = invoke(
                        row["task"], primary, raw / str(index), row["closure"]
                    )
                    if not receipt["passed"]:
                        return False
                actual = slot(
                    row["task"],
                    primary,
                    CASCADE if index == 6 and CASCADE in lines else None,
                    summary,
                )
                if any(actual.get(k) != row.get(k) for k in row if k not in ["memory", "closure"]):
                    return False
            return True
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False
