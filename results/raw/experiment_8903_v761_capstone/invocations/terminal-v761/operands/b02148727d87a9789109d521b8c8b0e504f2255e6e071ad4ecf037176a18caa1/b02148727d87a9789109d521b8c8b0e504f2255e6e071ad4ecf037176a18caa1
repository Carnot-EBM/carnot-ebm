"""REQ-REPORT-8826: retain exact custody and separate measurements from generalization."""

from collections import defaultdict
import json
from pathlib import Path
import shutil
from typing import Any

from carnot.reporting import v755_branch_contract as authority
from carnot.reporting.current_work_receipt import atomic_json as atomic_json, canonical_hash
from carnot.reporting.primary_publication import validate_primary
from carnot.reporting.consensus_selective_energy_8819 import summarize

Json = dict[str, Any]
ROOT, MILESTONE = authority.ROOT, authority.MILESTONE
NAME, TASK = "experiment_8826_v755_capstone", "exp8826-capstone"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [f"python/carnot/reporting/v755_capstone{s}.py" for s in ["", "_report", "_runner"]] + [CLI]
TEST = "tests/python/test_v755_capstone_8826.py"
reference = authority.reference


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush boundaries so the parent can distinguish completed work from a stalled reader."""
    print(f"[exp8826] phase={phase} completed={completed} pending={pending}", flush=True)


def checksum(value: Json) -> str:
    """Exclude the checksum itself so replay can compare all other semantic fields."""
    return canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"})


def references(value: Json) -> list[Json]:
    """Producer schemas use both lists and dictionaries for the same custody records."""
    result = []
    for key in ["raw_shard_hashes", "code_config_hashes", "source_artifact_hashes"]:
        refs = value.get(key, [])
        result.extend(refs.values() if isinstance(refs, dict) else refs)
    for receipt in value.get("validation_receipts", []):
        for stream in ["stdout", "stderr"]:
            if stream + "_path" in receipt:
                result.append(
                    dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
                )
    return result


def admit(path: Path, task_id: str) -> Json:
    """Readiness admits evidence only when identity, owned checks and bound bytes agree."""
    value: Json = {}
    refs: list[Json] = []
    try:
        value = json.loads(path.read_bytes())
        validate_primary(value, path)
        coverage = value.get("owned_statement_coverage", {})
        receipts = value.get("validation_receipts", [])
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
        refs = references(value)
        if not refs:
            raise ValueError("producer_custody_missing")
        for ref in refs:
            if reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
                raise ValueError("producer_operand_drift:" + ref["path"])
        return dict(admitted=True, errors=[], references=refs, value=value)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return dict(admitted=False, errors=[str(error)], references=refs, value=value)


def gate(path: Path, upstream: str, field: str, expected: Any) -> Json:
    """A missing operand is a different observation from a measured or declared zero."""
    ref = reference(path)
    value = json.loads(path.read_bytes()) if ref["sha256"] else {}
    observed = value.get(field)
    reason = (
        "missing_file"
        if not ref["sha256"]
        else "missing_field"
        if field not in value
        else "observed_zero"
        if observed == 0
        else None
    )
    return dict(
        ref,
        check="registered_gate",
        upstream=upstream,
        artifact_field=field,
        expected=expected,
        observed=observed,
        passed=type(observed) is type(expected) and observed == expected,
        missing_reason=reason,
    )


def retain(path: Path, directory: Path) -> Json:
    """Content-addressed copies preserve rejected bytes without overwriting earlier observations."""
    ref = reference(path)
    if ref["sha256"]:
        target = directory / ref["sha256"][7:] / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        return dict(reference(target), source_path=ref["path"])
    return ref


def outcome(task: Json, root: Path, raw: Path, log: list[str]) -> Json:
    """Scheduler receipts never supply a scientific primary verdict for an unexecuted task."""
    path = root / task["deliverable"]
    siblings = sorted(
        (root / "results").glob("experiment_" + task["id"][3:].split("-")[0] + "_*.json")
    )
    primary = (
        admit(path, task["id"])
        if path.is_file()
        else dict(admitted=False, value={}, references=[], errors=["missing_file"])
    )
    value = primary["value"]
    scheduler = []
    for candidate in siblings:
        content = value if candidate == path else json.loads(candidate.read_bytes())
        if (
            content.get("blocked_at_layer") == "conductor_pre_gate"
            and content.get("title") == task["title"]
        ):
            scheduler.append(
                dict(reference=retain(candidate, raw / "inputs"), gates=content["gates_evaluated"])
            )
    lines = [
        line
        for line in log
        if "GATE_BLOCK" in line and task["title"].startswith(line.split("|")[2].strip())
    ]
    disposition = (
        "quarantined_evidence"
        if value.get("flagged_adversarial")
        else "admitted_primary"
        if primary["admitted"]
        else "rejected_producer"
        if value.get("task_id")
        else "conductor_pre_gate"
        if scheduler
        else "conductor_cascade_skip"
        if lines
        else "missing_file"
    )
    snapshots = [retain(p, raw / "inputs") for p in siblings]
    snapshots += [
        retain(Path(r["path"]), raw / "operands")
        for r in primary["references"]
        if Path(r["path"]).is_file()
    ]
    accounted = bool(value.get("task_id") == task["id"] or scheduler or lines)
    return dict(
        task_id=task["id"],
        experiment_id=int(task["id"][3:].split("-")[0]),
        unit_id=task["id"],
        cluster_id="administrative_task_slots",
        arm="scheduled_task",
        seed=None,
        metric=int(accounted),
        denominator=1,
        status="completed" if accounted else "censored",
        missing_reason=None if accounted else "missing_file",
        accounted=accounted,
        disposition=disposition,
        admitted=primary["admitted"],
        primary_honest_verdict=value.get("honest_verdict") if value.get("task_id") else None,
        verdict_class=value.get("verdict_class", "blocked"),
        primary_reference=retain(path, raw / "inputs"),
        scheduler_receipts=scheduler,
        conductor_lines=lines,
        admission_errors=primary["errors"],
        snapshots=snapshots,
        producer=value,
    )


def reduce(value: Json) -> Json:
    """Count units separately from sibling seeds, retaining only registered estimands."""
    rows = [r for r in value.get("rows", []) if r["status"] == "completed"]
    groups: dict[str, list[Json]] = defaultdict(list)
    for row in rows:
        groups[row["arm"]].append(row)
    measured = value.get("experiment_id") in {8815, 8818, 8819}
    result = dict(
        constructed_measurement_count=len(rows) if measured else 0,
        independent_constructed_unit_count=len({r["cluster_id"] for r in rows}) if measured else 0,
        natural_generalization_count=0,
        arms={
            a: dict(row_count=len(g), denominator=sum(r["denominator"] for r in g))
            for a, g in groups.items()
        },
        exact_oracle_verdict="circular_positive" if value.get("verifier_is_oracle") else None,
        registered_estimands={},
    )
    if value.get("experiment_id") == 8819:
        result["registered_estimands"] = summarize(
            value["per_candidate_rows"], value["selected_threshold_rows"]
        )
        result["exact_oracle_verdict"] = "circular_positive"
    if value.get("experiment_id") == 8818:
        from carnot.reporting.evidence_decisive_8818 import reduce as evidence_reduce

        result["registered_estimands"] = evidence_reduce(
            rows, value["comparator_selection"]["comparator"] or "joint_symbolic"
        )
    return result


def producer_summary(value: Json) -> Json:
    """Keep the accounting ledger small while immutable primaries retain every measured row."""
    keys = {
        "experiment_id",
        "honest_verdict",
        "verdict_class",
        "exposure_scope",
        "acceptance_gates",
        "gate_check_summary",
        "verifier_is_oracle",
        "task_id",
    }
    return dict(
        {k: v for k, v in value.items() if k in keys or k.endswith("_score")},
        reported_completed_rows=len(value.get("rows", [])),
    )


def freeze(root: Path, raw: Path, run_date: str) -> Json:
    """Freeze current authority and historical bytes before reducing any scientific rows."""
    progress("custody_before", 0, 13)
    raw.mkdir(parents=True, exist_ok=True)
    contract = authority.authority(root)
    if not contract["ready"]:
        raise ValueError("v755_full_object_authority")
    sources = [
        authority.DESIGN,
        authority.ACTIVE,
        "ops/conductor-log.md",
        "ops/exclusion_manifest.yaml",
    ]
    sources += list(authority.HISTORY_PINS)
    snapshots = [retain(root / p, raw / "authority") for p in sources]
    for name, pin in authority.HISTORY_PINS.items():
        if reference(root / name)["sha256"] != "sha256:" + pin:
            raise ValueError("historical_authority_drift:" + name)
    log = (root / "ops/conductor-log.md").read_text().splitlines()
    rows = []
    for index, task in enumerate(contract["tasks"][:-1]):
        progress("slot_before", index, 13 - index)
        row = outcome(task, root, raw, log)
        row["reduction"] = reduce(row["producer"]) if row["admitted"] else {}
        row["producer"] = producer_summary(row["producer"])
        rows.append(row)
        progress("slot_after", index + 1, 12 - index)
    edges = []
    by_id = {t["id"]: t for t in contract["tasks"]}
    for task in contract["tasks"]:
        for registered in task["gated_on"]:
            edge = gate(
                root / by_id[registered["upstream"]]["deliverable"],
                registered["upstream"],
                registered["artifact_field"],
                registered["value"],
            )
            edge.update(downstream=task["id"], op=registered["op"])
            edges.append(edge)
    historical = json.loads((root / authority.HISTORY).read_bytes())
    history_rows = {r["task_id"]: r for r in historical["rows"]}
    priors = []
    for task in contract["tasks"]:
        for prior in task["prior_failures"]:
            number = prior["experiment_id"][3:].split("-")[0]
            paths = sorted((root / "results").glob(f"experiment_{number}_*.json"))
            actual = history_rows.get(prior["experiment_id"], {})
            bound = actual.get("primary_reference", {})
            matches = [
                p
                for p in paths
                if str(p.absolute()) == bound.get("path")
                and reference(p)["sha256"] == bound.get("sha256")
            ]
            if prior["experiment_id"] == "exp8813-capstone":
                matches, actual = [root / authority.HISTORY], historical
            ref = retain(matches[0], raw / "history") if matches else dict(path=None, sha256=None)
            priors.append(
                dict(
                    task_id=task["id"],
                    registration=prior,
                    reference=ref,
                    actual_honest_verdict=actual.get("honest_verdict"),
                    actual_verdict_class=actual.get("verdict_class"),
                )
            )
    work = dict(
        contract=contract,
        slots=rows,
        edges=edges,
        priors=priors,
        snapshots=snapshots,
        historical_publication_gates=historical["publication_gates"],
        run_date=run_date,
        raw=str(raw.absolute()),
        code=[reference(ROOT / p) for p in OWNED + [TEST]],
    )
    atomic_json(raw / "work.json", work)
    progress("custody_after", 12, 1)
    return work


from carnot.reporting.v755_capstone_report import build as build, replay as replay
