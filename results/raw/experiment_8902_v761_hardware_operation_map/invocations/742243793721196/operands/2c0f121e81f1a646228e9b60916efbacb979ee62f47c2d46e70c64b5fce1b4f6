"""REQ-REPORT-8865: reconcile every slot while retaining failed scientific claims."""

from copy import deepcopy
import importlib.metadata
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any

from carnot.reporting import v756_capstone as legacy
from carnot.reporting import v756_capstone_report as previous_report
from carnot.reporting import v758_branch_contract as authority
from carnot.reporting.current_work_receipt import (
    atomic_json as atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.primary_publication import validate_primary
from carnot.reporting.v748_collection_contract import namespace
from carnot.reporting.v758_capstone_reduction import reduce
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]
ROOT, MILESTONE = authority.ROOT, authority.MILESTONE
NAME, TASK = "experiment_8865_v758_capstone", "exp8865-capstone"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [f"python/carnot/reporting/v758_capstone{s}.py" for s in ["", "_reduction", "_runner"]] + [
    CLI
]
TEST = "tests/python/test_v758_capstone_8865.py"
shared = namespace(legacy.shared, "capstone8865_custody")
checksum = legacy.checksum
_digests: dict[tuple[str, int, int, int], str] = {}
retirement = legacy.retirement
report = namespace(previous_report, "capstone8865_report")


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Show real phase boundaries so long custody checks remain observable."""
    print(f"[exp8865] phase={phase} completed={completed} pending={pending}", flush=True)


def reference(path: Path) -> Json:
    """Hash each unchanged file once per reader; changed metadata invalidates the cache."""
    path = path.absolute()
    digest = None
    if path.is_file():
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if key not in _digests:
            _digests[key] = sha256_file(path)
        digest = _digests[key]
    return dict(path=str(path), sha256=digest)


def retain(path: Path, directory: Path) -> Json:
    """Copy each content-addressed operand once and reject changes to an existing copy."""
    ref = reference(path)
    if not ref["sha256"]:
        return ref
    target = directory / ref["sha256"][7:] / path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copyfile(path, target)
    if reference(target)["sha256"] != ref["sha256"]:
        raise ValueError("immutable_operand_conflict")
    return dict(reference(target), source_path=ref["path"])


def admit(path: Path, task_id: str) -> Json:
    """Apply the existing admission contract once per distinct operand, including logs."""
    value: Json = {}
    refs: list[Json] = []
    try:
        value = json.loads(path.read_bytes())
        validate_primary(value, path)
        coverage, receipts = (
            value.get("owned_statement_coverage", {}),
            value.get("validation_receipts", []),
        )
        if (
            value.get("task_id") != task_id
            or value.get("required_checks_passed") is not True
            or value.get("flagged_adversarial")
            or value["verdict_class"] == "disqualified"
            or not coverage.get("complete")
            or coverage.get("percent") != 100
            or not receipts
            or any(not r["passed"] for r in receipts if r.get("scope") != "repository_health")
        ):
            raise ValueError("producer_identity_or_owned_validation")
        refs = shared.references(value)
        refs += [
            dict(path=str(ROOT / r["log_path"]), sha256=r["log_sha256"])
            for r in receipts
            if "log_path" in r
        ]
        refs = list({(r["path"], r["sha256"]): r for r in refs}.values())
        if not refs:
            raise ValueError("producer_custody_missing")
        for i, ref in enumerate(refs):
            if reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
                raise ValueError("producer_operand_drift:" + ref["path"])
            if i % 5000 == 0:
                progress("operand_authentication", i, len(refs) - i)
        return dict(admitted=True, errors=[], references=refs, value=value)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return dict(admitted=False, errors=[str(error)], references=refs, value=value)


shared.admit, shared.reference, shared.retain = admit, reference, retain


def freeze(root: Path, raw: Path, run_date: str) -> Json:
    """Freeze full authority and raw producer bytes before admitting any conclusion."""
    _digests.clear()
    progress("preconditions_before", 0, 13)
    pre = legacy.preflight(root, raw)
    raw.chmod(0o700)
    pre.update(
        time_budget_s=4200,
        installed_dependencies={
            p: importlib.metadata.version(p)
            for p in ["coverage", "pytest", "PyYAML", "numpy", "scipy"]
        },
    )
    for path, pin in authority.PINS.items():
        if reference(root / path)["sha256"] != "sha256:" + pin:
            raise ValueError("historical_authority_drift:" + path)
    contract = authority.authority(root)
    if not contract["tasks"]:
        contract["tasks"] = json.loads(
            (root / "results/experiment_8853_v758_branch_contract.json").read_bytes()
        )["full_task_objects"]
    sources = [
        authority.DESIGN,
        authority.ACTIVE,
        authority.STAGED,
        "ops/conductor-log.md",
        "ops/exclusion_manifest.yaml",
        "research-complete.yaml",
        "ops/north-star.md",
        *authority.PINS,
    ]
    snapshots = [retain(root / p, raw / "authority") for p in sources]
    for name in [authority.DESIGN, authority.ACTIVE, authority.STAGED]:
        if (root / name).is_file():
            target = raw / "authority_view" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((root / name).read_bytes())
    historical = json.loads((root / authority.CAPSTONE).read_bytes())
    for task in contract["tasks"]:
        for prior in task["prior_failures"]:
            number = prior["experiment_id"][3:].split("-")[0]
            for path in sorted((root / "results").glob(f"experiment_{number}_*.json")):
                value = json.loads(path.read_bytes())
                if value.get("task_id") == prior["experiment_id"]:
                    ref = retain(path, raw / "history")
                    snapshots.append(ref)
                    historical["rows"].append(
                        dict(
                            task_id=prior["experiment_id"],
                            primary_honest_verdict=value["honest_verdict"],
                            primary_reference=ref,
                        )
                    )
    slots = []
    log = (root / "ops/conductor-log.md").read_text().splitlines()
    for i, task in enumerate(contract["tasks"][:-1]):
        progress("slot_before", i, 13 - i)
        row = shared.outcome(task, root, raw, log)
        checked = admit(root / task["deliverable"], task["id"])
        row.update(admitted=checked["admitted"], admission_errors=checked["errors"])
        row["snapshots"] += [
            retain(Path(r["path"]), raw / "operands") for r in checked["references"]
        ]
        row["snapshots"] = list({(r["path"], r["sha256"]): r for r in row["snapshots"]}.values())
        row["upstream_compute"] = {
            k: row["producer"].get(k)
            for k in ["model_invocation_counts", "numerical_training_calls"]
        }
        try:
            row["reduction"] = reduce(row["producer"]) if row["admitted"] else {}
        except ValueError as error:
            row.update(admitted=False, reduction={})
            row["admission_errors"].append(str(error))
        if row["disposition"] == "admitted_primary" and not row["admitted"]:
            row["disposition"] = "rejected_producer"
        row["producer"] = shared.producer_summary(row["producer"])
        # Missing evidence is still an accounted terminal observation.
        row.update(accounted=True, metric=1, status="completed")
        slots.append(row)
        progress("slot_after", i + 1, 12 - i)
    by_id = {t["id"]: t for t in contract["tasks"]}
    edges = []
    for task in contract["tasks"]:
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
    work = dict(
        contract=contract,
        slots=slots,
        edges=edges,
        snapshots=snapshots,
        historical=historical,
        preconditions=pre,
        run_date=run_date,
        raw=str(raw.absolute()),
        code=[
            reference(ROOT / p)
            for p in OWNED
            + [TEST]
            + legacy.DEPENDENCIES
            + [
                "python/carnot/reporting/v756_capstone.py",
                "python/carnot/reporting/v756_capstone_report.py",
                "python/carnot/reporting/v756_capstone_runner.py",
                "python/carnot/reporting/v748_collection_contract.py",
                "scripts/experiment_template.py",
                "python/carnot/reporting/v758_branch_contract.py",
            ]
        ],
    )
    atomic_json(raw / "work.json", work)
    progress("custody_after", 12, 1)
    return work


def build(work: Json, validation: Json) -> Json:
    """Complete accounting can be ready even while independent science remains blocked."""
    report.c = SimpleNamespace(
        TASK=TASK,
        ROOT=ROOT,
        MILESTONE=MILESTONE,
        authority=SimpleNamespace(PIN=authority.PIN, HISTORY=authority.CAPSTONE),
        reference=reference,
        retirement=retirement,
        shared=shared,
        checksum=checksum,
    )
    value: Json = report.build(work, validation)
    owned = value["required_checks_passed"]
    ready = owned and work["contract"]["ready"] and all(r["accounted"] for r in work["slots"])
    blocked = not work["contract"]["ready"] or any(
        not r["admitted"] or r["verdict_class"] in {"blocked", "disqualified"}
        for r in work["slots"]
    )
    verdict = "disqualified" if not owned else "blocked" if blocked else "null"
    honest = f"complete_{verdict}_v758_capstone"
    self_row = value["rows"][-1]
    self_row.update(
        experiment_id=8865,
        primary_honest_verdict=honest,
        verdict_class=verdict,
        metric=int(ready),
        publication_evidence=dict(
            validation=reference(Path(work["raw"]) / "validation.json"),
            method="primary_publication hash-bound validator sidecar",
        ),
    )
    reductions = [s["reduction"] for s in work["slots"] if s["admitted"]]
    streams = {x for r in reductions for x in r["stream_ids"]}
    worlds = {x for r in reductions for x in r["world_ids"]}
    draws = {x for r in reductions for x in r["draw_ids"]}
    decisions = [
        (
            "Keep CPU roots independent; leave CUDA generation blocked",
            "Causal hash-bound CUDA repair and measured offload",
        ),
        (
            "Keep the sealed six-arm linear protocol",
            "A preregistered changed mechanism before another study",
        ),
        (
            "Use the ceiling only as a population diagnostic",
            "A new sealed deployment rule and independent certification",
        ),
        (
            "Accept causal integrity separately from benefit",
            "New future predictions and untouched retention after a changed mechanism",
        ),
        (
            "Keep certificate validity separate from utility",
            "A changed protocol with fresh disjoint certification labels",
        ),
        (
            "Keep the indexed structured API opt-in",
            "Exact parity and full installed boundaries on any changed implementation",
        ),
        (
            "Retain the split-protocol utility null",
            "A changed protocol and positive preregistered paired coverage interval",
        ),
        (
            "Retain the learning conjunction null",
            "Combined must beat prefix-shuffled feedback and pass all other frozen endpoints",
        ),
        (
            "Retain the installed speed-benefit null",
            "Complete mixture speed interval above one and the size16 latency gate",
        ),
        (
            "Retain complete durable CPU costs without acceleration claims",
            "Matched complete windows, preparation, transfer and durable-tail timings",
        ),
        (
            "Keep the Qwen canary blocked without retrying the model",
            "Admitted changed CUDA repair plus installed readiness;24 calls remain a canary",
        ),
        (
            "Keep hardware mapping on CPU; defer purchases",
            "Supported kernel parity and measured host, transfer, device and durability costs",
        ),
    ]
    for branch, (decision, evidence) in zip(value["branch_conclusions"], decisions, strict=True):
        branch.update(next_decision=decision, evidence_needed_to_reopen=evidence)
    gates = value["historical_publication_gates"]
    value.update(
        experiment_id=8865,
        honest_verdict=honest,
        verdict_class=verdict,
        random_seed=7588865,
        execution_venue="host",
        reconciliation_ready_score=int(ready),
        independent_stream_count=len(streams),
        independent_world_count=len(worlds),
        independent_draw_count=len(draws),
        cpu_request_count=sum(r["cpu_request_count"] for r in reductions),
        actual_llm_call_count=0,
        live_arc_outcome_count=0,
        paper_ready=gates["paper_ready"],
        unmet_gates=gates["unmet_gates"],
        historical_v757=dict(
            reference=next(
                r
                for r in work["snapshots"]
                if r.get("source_path", "").endswith(authority.CAPSTONE)
            ),
            honest_verdict=work["historical"]["honest_verdict"],
            verdict_class=work["historical"]["verdict_class"],
            disqualified_task_ids=[
                r["task_id"]
                for r in work["historical"]["rows"]
                if r.get("verdict_class") == "disqualified"
            ],
        ),
        capstone_next_decision="Keep independent branches separate; do not activate or publish externally",
    )
    value["acceptance_gates"]["complete_accounting"]["passed"] = bool(ready)
    if not work["contract"]["ready"]:
        value["gate_check_summary"].append(
            dict(
                check="full_contract_authority",
                upstream=TASK,
                artifact_field="canonical_tasks_sha256",
                expected=authority.PIN,
                observed="missing_or_rejected_authority",
                passed=False,
                missing_reason="rejected_producer",
                **reference(Path(work["raw"]) / "work.json"),
            )
        )
    value["methodology_note"] = (
        "Constructed exact-oracle evidence is circular_positive, not natural generalization. "
        "Integrity, future gain, addition specificity, true-label specificity and retention are separate. "
        "Population certification does not imply protocol utility. Installed Python CPU costs prove "
        "neither native10x nor model accuracy; operation counts are not timing fractions. "
        "No new model, physical hardware or live ARC work occurred. External absence is blocked."
    )
    value["raw_shard_hashes"] = list(
        {(r["path"], r["sha256"]): r for r in value["raw_shard_hashes"]}.values()
    )
    value = normalize_artifact_for_template_write(value)
    value["field_principles"] = dict.fromkeys(
        [*value, "reproducibility_checksum"],
        "Frozen operands support independent reduction. Administrative readiness does not grant scientific benefit or external authority.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Rebuild from preserved authorities so mutable roadmap rollover cannot rewrite history."""
    try:
        _digests.clear()
        value = json.loads(path.read_bytes())
        work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
        if value != build(work, value["validation_state"]):
            raise ValueError("semantic_drift")
        for ref in (
            value["raw_shard_hashes"] + value["code_config_hashes"] + [value["work_reference"]]
        ):
            if ref["sha256"] and reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
                raise ValueError("bound_operand_drift")
        if canonical_hash(work["contract"]["tasks"]) != "sha256:" + authority.PIN:
            raise ValueError("registered_queue_drift")
        frozen_contract = authority.authority(Path(work["raw"]) / "authority_view")
        if not frozen_contract["tasks"]:
            frozen_contract["tasks"] = work["contract"]["tasks"]
        if frozen_contract != work["contract"]:
            raise ValueError("frozen_authority_drift")
        for slot in work["slots"]:
            checked = admit(Path(slot["primary_reference"]["path"]), slot["task_id"])
            rebuilt = {}
            if checked["admitted"]:
                try:
                    rebuilt = reduce(checked["value"])
                except ValueError:
                    checked["admitted"] = False
            if checked["admitted"] != slot["admitted"] or rebuilt != slot["reduction"]:
                raise ValueError("producer_admission_or_reduction_drift")
            if shared.producer_summary(checked["value"]) != slot["producer"]:
                raise ValueError("producer_summary_drift")
            if slot["verdict_class"] != checked["value"].get("verdict_class", "blocked") or slot[
                "primary_honest_verdict"
            ] != checked["value"].get("honest_verdict"):
                raise ValueError("producer_verdict_drift")
        expected_edges = []
        for task in work["contract"]["tasks"]:
            for gate in task["gated_on"]:
                slot = next(r for r in work["slots"] if r["task_id"] == gate["upstream"])
                expected = shared.gate(
                    Path(slot["primary_reference"]["path"]),
                    gate["upstream"],
                    gate["artifact_field"],
                    gate["value"],
                )
                if expected["passed"] and not slot["admitted"]:
                    expected.update(passed=False, missing_reason="rejected_producer")
                expected_edges.append(dict(expected, downstream=task["id"], op=gate["op"]))
        fields = [
            "upstream",
            "artifact_field",
            "expected",
            "observed",
            "passed",
            "missing_reason",
            "downstream",
            "op",
            "sha256",
        ]
        if [{k: r[k] for k in fields} for r in expected_edges] != [
            {k: r[k] for k in fields} for r in work["edges"]
        ]:
            raise ValueError("gate_edge_drift")
        return dict(passed=True, reason="replay_passed")
    except (OSError, ValueError, KeyError, TypeError):
        return dict(passed=False, reason="invalid_or_changed_evidence")
