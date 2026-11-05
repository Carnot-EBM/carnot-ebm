"""REQ-REPORT-8852: reconcile thirteen slots without repairing missing authority."""

from copy import deepcopy
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from carnot.reporting import v756_capstone as legacy
from carnot.reporting import v756_capstone_report as previous_report
from carnot.reporting import v757_branch_contract as original_authority
from carnot.reporting.current_work_receipt import atomic_json as atomic_json, canonical_hash
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
ROOT, MILESTONE = original_authority.ROOT, original_authority.MILESTONE
NAME, TASK = "experiment_8852_v757_capstone", "exp8852-capstone"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/v757_capstone.py",
    "python/carnot/reporting/v757_capstone_reduction.py",
    "python/carnot/reporting/v757_capstone_runner.py",
    CLI,
]
TEST = "tests/python/test_v757_capstone_8852.py"
authority = SimpleNamespace(**vars(original_authority), HISTORY=original_authority.CAPSTONE)
shared, reference, checksum, retain = (
    legacy.shared,
    legacy.reference,
    legacy.checksum,
    legacy.retain,
)
retirement = legacy.retirement
report = namespace(previous_report, "capstone8852_report")


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush completed counts so a long custody check remains visible."""
    print(f"[exp8852] phase={phase} completed={completed} pending={pending}", flush=True)


def freeze(root: Path, raw: Path, run_date: str) -> Json:
    """A pinned queue permits accounting, but never supplies absent design bytes."""
    progress("custody_before", 0, 13)
    queue_path = root / authority.ACTIVE
    if not queue_path.is_file():
        raise ValueError("registered_queue_missing")
    tasks = yaml.safe_load(queue_path.read_bytes())["tasks"]
    if canonical_hash(tasks) != "sha256:" + authority.PIN:
        raise ValueError("registered_queue_drift")
    preconditions = legacy.preflight(root, raw)
    preconditions.update(
        time_budget_s=1800,
        installed_dependencies={
            p: importlib.metadata.version(p)
            for p in ["coverage", "pytest", "pydantic", "PyYAML", "numpy", "scipy"]
        },
    )
    raw.chmod(0o700)
    contract = authority.authority(root)
    contract["tasks"] = tasks
    for name in [authority.DESIGN, authority.ACTIVE, authority.STAGED]:
        if (root / name).is_file():
            target = raw / "authority_view" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((root / name).read_bytes())
    snapshots = [
        retain(root / p, raw / "authority")
        for p in [
            authority.DESIGN,
            authority.ACTIVE,
            authority.STAGED,
            "ops/conductor-log.md",
            "ops/exclusion_manifest.yaml",
            "ops/north-star.md",
            "research-references.md",
            *authority.PINS,
        ]
    ]
    snapshots += [
        reference(raw / "authority_view" / name)
        for name in [authority.DESIGN, authority.ACTIVE, authority.STAGED]
        if (raw / "authority_view" / name).is_file()
    ]
    for path, pin in authority.PINS.items():
        if reference(root / path)["sha256"] != "sha256:" + pin:
            raise ValueError("historical_authority_drift:" + path)
    log = (root / "ops/conductor-log.md").read_text().splitlines()
    slots = []
    from carnot.reporting.v757_capstone_reduction import reduce

    for i, task in enumerate(tasks[:-1]):
        progress("slot_before", i, 13 - i)
        row = shared.outcome(task, root, raw, log)
        checked = legacy.admit(root / task["deliverable"], task["id"])
        row.update(admitted=checked["admitted"], admission_errors=checked["errors"])
        if row["disposition"] == "admitted_primary" and not row["admitted"]:
            row["disposition"] = "rejected_producer"
        row["snapshots"] += [
            retain(Path(r["path"]), raw / "operands") for r in checked["references"]
        ]
        try:
            row["reduction"] = reduce(row["producer"]) if row["admitted"] else {}
        except ValueError as error:
            row.update(admitted=False, disposition="rejected_producer", reduction={})
            row["admission_errors"].append(str(error))
        row["primary_verdict_class"] = row["verdict_class"]
        if not row["admitted"] and row["producer"].get("task_id"):
            row["verdict_class"] = "disqualified"
        row["producer"] = shared.producer_summary(row["producer"])
        slots.append(row)
        progress("slot_after", i + 1, 12 - i)
    edges = []
    by_id = {t["id"]: t for t in tasks}
    for task in tasks:
        for g in task["gated_on"]:
            edge = shared.gate(
                root / by_id[g["upstream"]]["deliverable"],
                g["upstream"],
                g["artifact_field"],
                g["value"],
            )
            upstream = next(r for r in slots if r["task_id"] == g["upstream"])
            if edge["passed"] and not upstream["admitted"]:
                edge.update(passed=False, missing_reason="rejected_producer")
            edges.append(dict(edge, downstream=task["id"], op=g["op"]))
    dependencies = legacy.DEPENDENCIES + [
        "python/carnot/reporting/v756_capstone.py",
        "python/carnot/reporting/v756_capstone_report.py",
        "python/carnot/reporting/v756_capstone_runner.py",
        "python/carnot/reporting/v757_branch_contract.py",
        "python/carnot/reporting/future_benefit_audit_8846.py",
        "python/carnot/reporting/selective_availability_8844.py",
        "python/carnot/reporting/indexed_cost_8847.py",
        "python/carnot/reporting/memory_size_cost_8848.py",
    ]
    work = dict(
        contract=contract,
        slots=slots,
        edges=edges,
        snapshots=snapshots,
        historical=json.loads((root / authority.HISTORY).read_bytes()),
        preconditions=preconditions,
        run_date=run_date,
        raw=str(raw.absolute()),
        code=[reference(ROOT / p) for p in OWNED + [TEST] + dependencies],
    )
    atomic_json(raw / "work.json", work)
    progress("custody_after", 12, 1)
    return work


def build(work: Json, validation: Json) -> Json:
    """Reuse the ledger schema while deduplicating reused scientific populations."""
    report.c = SimpleNamespace(
        TASK=TASK,
        ROOT=ROOT,
        MILESTONE=MILESTONE,
        authority=authority,
        reference=reference,
        retirement=retirement,
        shared=shared,
        checksum=checksum,
    )
    value: Json = report.build(work, validation)
    owned = value["required_checks_passed"]
    verdict = "blocked" if owned else "disqualified"
    honest = (
        "complete_blocked_incomplete_v757_design_authority"
        if owned and not work["contract"]["ready"]
        else f"complete_{verdict}_v757_capstone"
    )
    ready = owned and all(r["accounted"] for r in work["slots"])
    self_row = value["rows"][-1]
    self_row.update(
        experiment_id=8852,
        primary_honest_verdict=honest,
        metric=int(ready),
        publication_evidence=dict(
            validation=reference(Path(work["raw"]) / "validation.json"),
            method="primary_publication hash-bound validator sidecar",
        ),
    )
    reductions = [r["reduction"] for r in work["slots"] if r["admitted"]]
    stream_ids = {s for r in reductions for s in r["stream_ids"]}
    world_ids = {s for r in reductions for s in r["world_ids"]}
    draw_ids = {s for r in reductions for s in r["draw_ids"]}
    audit = next((r["registered_estimands"] for r in reductions if r["experiment_id"] == 8846), {})
    value.update(
        experiment_id=8852,
        random_seed=7578852,
        honest_verdict=honest,
        execution_venue="host",
        reconciliation_ready_score=int(ready),
        independent_stream_count=len(stream_ids),
        independent_world_count=len(world_ids),
        independent_draw_count=len(draw_ids),
        continuation=audit.get(
            "next_head_recommendation", dict(next_head=None, reason="audit_not_admitted")
        ),
        actual_llm_call_count=0,
        live_arc_outcome_count=0,
        distinct_population_support=max((r["population_support"] for r in reductions), default=0),
        calibration_draw_count=len({s for s in draw_ids if "/calibration/" in s}),
        design_authority_ready=work["contract"]["ready"],
        publication_authorized=False,
        activation_authorized=False,
    )
    gates = value["historical_publication_gates"]
    value.update(paper_ready=gates["paper_ready"], unmet_gates=gates["unmet_gates"])
    if not work["contract"]["ready"]:
        value["gate_check_summary"].append(
            dict(
                work["contract"]["failure"],
                passed=False,
                missing_reason=work["contract"]["failure"]["observed"],
            )
        )
    value["acceptance_gates"]["complete_accounting"]["passed"] = bool(ready)
    for recommendation in value["retirement_recommendations"]:
        if (
            recommendation["retire_exact_scope"]
            and recommendation["task_id"] == "exp8851-hardware-operation-budget"
        ):
            recommendation.update(
                scope="Unchanged physical hardware acceleration claim",
                diagnosed_cause="No new authenticated fabric kernel or physical acceleration measurement",
                evidence_needed_to_reopen="Authenticated supported fabric kernel and complete host/transfer/device timing against CPU",
            )
    for ref in work["snapshots"]:
        if ref["sha256"] is None:
            value["gate_check_summary"].append(
                dict(
                    ref,
                    check="required_authority",
                    upstream=TASK,
                    artifact_field="authority_source",
                    expected="present",
                    observed="missing_file",
                    missing_reason="missing_file",
                    passed=False,
                )
            )
    value["methodology_note"] = (
        "Complete custody is terminal even when design authority is truncated. "
        "Constructed stream improvements do not pass retention or categorical noninferiority. "
        "Finite-population draws do not establish unseen-structure transfer. "
        "Preserve ordered-check harm, old learning nulls, runtime blocks and board limits; "
        "indexed exact-oracle cost benefit is circular_positive, never natural generalization."
    )
    value["field_principles"] = dict.fromkeys(
        [*value, "reproducibility_checksum"],
        "Frozen rows permit independent reduction; reused arms, seeds and audits do not add populations. "
        "Accounting readiness grants no scientific benefit or external-action authority.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Reauthenticate bound bytes and recompute every admitted reduction in a fresh reader."""
    from carnot.reporting.v757_capstone_reduction import reduce

    try:
        value = json.loads(path.read_bytes())
        for ref in (
            value["raw_shard_hashes"] + value["code_config_hashes"] + [value["work_reference"]]
        ):
            if ref["sha256"] and reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
                raise ValueError("bound_operand_drift")
        work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
        if canonical_hash(work["contract"]["tasks"]) != "sha256:" + authority.PIN:
            raise ValueError("registered_queue_drift")
        view = Path(work["raw"]) / "authority_view"
        frozen_contract = authority.authority(view)
        frozen_contract["tasks"] = work["contract"]["tasks"]
        if "failure" in frozen_contract:
            frozen_contract["failure"].update(
                path=str(ROOT / authority.DESIGN),
                sha256=reference(view / authority.DESIGN)["sha256"],
            )
        if frozen_contract != work["contract"]:
            raise ValueError("design_authority_drift")
        for slot in work["slots"]:
            admitted = legacy.admit(Path(slot["primary_reference"]["path"]), slot["task_id"])
            reduced = {}
            if admitted["admitted"]:
                try:
                    reduced = reduce(admitted["value"])
                except ValueError:
                    admitted["admitted"] = False
            if admitted["admitted"] != slot["admitted"]:
                raise ValueError("producer_admission_drift")
            if slot["admitted"] and reduced != slot["reduction"]:
                raise ValueError("producer_reduction_drift")
            if shared.producer_summary(admitted["value"]) != slot["producer"]:
                raise ValueError("producer_summary_drift")
        passed = value == build(work, value["validation_state"])
        return dict(passed=passed, reason="replay_passed" if passed else "semantic_drift")
    except (OSError, ValueError, KeyError, TypeError):
        return dict(passed=False, reason="invalid_or_changed_evidence")
