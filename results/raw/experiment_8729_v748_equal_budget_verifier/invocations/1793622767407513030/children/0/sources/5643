"""REQ-REPORT-8708: authenticate original bytes before interpreting branch outcomes.

Scheduling dispositions are administrative evidence. They never supply a missing
scientific observation or repair a producer's historical validation failure.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

from carnot.reporting import hardware_operation_inputs_8707 as authority
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import validate_primary

Json = dict[str, Any]
__all__ = ["authority", "atomic_json", "sha256_file"]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK, MILESTONE = "experiment_8708_v746_capstone", "exp8708-capstone", "2026.11.746"
CLI = f"scripts/experiments/{NAME}.py"
TESTS = ["tests/python/test_v746_capstone_8708.py"]
OWNED = [f"python/carnot/reporting/v746_capstone{s}.py" for s in ("_inputs", "_report", "")] + [CLI]
MODEL_SPECS: list[Json] = []
DESIGN = "openspec/change-proposals/research-roadmap-v746-preserved-20261101.md"
PINS = {
    8681: "fdaa9313e6466b2bc0c3e9e72516d4d17286de1a5148a090e41cd6bfe5b3e284",
    8691: "89a9805fe678365ca347958c497f90229a6daa6bd051294ae294998f2d8d9366",
    8692: "bb960180068d3f6960df74f7f0dfeb440ed64ac4951071674e820a0d912eaa33",
    8693: "1edc62c1f83d68185ba712fa181275ab711ba9b7e51a1ac684e8e6d4b1004006",
    8695: "6aabdc1d57654175f8ec01af407c16855d3c24214665bcbdc2d0016428ea4b7d",
    8696: "fb110af2636a044d398028c1a63fe003c9b95e02e4962580e6d42228e9ccb515",
    8697: "1e9e81e361ffdbe70fae3c8e61460776abb74834b313fe575dd1ae20e6f39525",
    8698: "2d32e3129192287b372ea342b3e56b9139d18ea5a7eae23347151cf822df4d8a",
    8701: "fdcc5b61255ddceef2547d5c02b075ec20b5cafe0763703afd5dcebfd42bf194",
    8702: "879b1c1ebe5aa1996cd34ba4b635b6d30da6dbd4ee58e5fe4cbf8ef68b790e36",
    8705: "62e8795168a812f13e0e81002f56e7676fe25e275c0e5a5cf9a81827635be091",
    8706: "306fde7e7afccef1f9a079ead7a1bb26226008e0126bfccbc837a5afe172e0f5",
    8707: "21802928248cb86fe4dcc2cdb4ea6ad8bf403d0884d3c4e049f13e8906f53cb6",
    8694: "f9551a44d8d29f01cbb344d96647808253fd524262e582f7a4782e8e97e30947",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real counts so the parent supervisor can observe bounded CPU work."""
    print(f"[exp8708] phase={phase} completed={completed} pending={pending}", flush=True)


def reference(path: Path) -> Json:
    """A missing file has no hash; it is a different operand from measured zero."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path) if path.is_file() else None)


def verify(ref: Json) -> None:
    """Verify binary and textual evidence without interpreting a substituted file."""
    if (
        reference(Path(ref["path"])) != {k: ref[k] for k in ("path", "sha256")}
        or ref["sha256"] is None
    ):
        raise ValueError("changed_or_missing_bytes:" + ref["path"])


def load(ref: Json) -> Json:
    """Parse only authenticated JSON, including an independently sealed primitive."""
    verify(ref)
    return dict(json.loads(Path(ref["path"]).read_bytes()))


def digest(value: Json) -> str:
    """Exclude the checksum itself so a replay can recompute the same content hash."""
    return str(canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"}))


def freeze(path: Path, raw: Path) -> Json:
    """Keep an immutable byte copy beside the original reference before measurement."""
    ref = reference(path)
    if ref["sha256"] is not None:
        target = raw / ref["sha256"][7:]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        ref["snapshot_reference"] = reference(target)
    return ref


def observe(ref: Json, upstream: str, field: str, expected: Any = 1) -> Json:
    """Keep missing file, missing field, zero and changed bytes as distinct failures."""
    return dict(authority.gate(ref, field, expected, upstream))


def locate(task: Json, root: Path = ROOT) -> Path:
    """Read the declared deliverable first, then exact old conductor conventions."""
    declared = root / task["deliverable"]
    number, slug = task["id"].split("-", 1)
    fallback = root / "results" / f"experiment_{number[3:]}_{slug.replace('-', '_')}.json"
    bare = root / "results" / f"experiment_{number[3:]}.json"
    found = list((root / "results").glob(f"experiment_{number[3:]}*.json"))
    if len(found) > 1 or any(p not in {declared, fallback, bare} for p in found):
        raise ValueError("ambiguous_primary:" + task["id"])
    return next((p for p in (declared, fallback, bare) if p.is_file()), declared)


def authenticate(path: Path, task: Json) -> Json:
    """Terminal acceptance binds primitive bytes; a failed producer still supplies no readiness."""
    refs, errors = [], []
    try:
        ref = reference(path)
        number = int(task["id"].split("-")[0][3:])
        if number in PINS and ref["sha256"] != "sha256:" + PINS[number]:
            raise ValueError("independent_producer_pin")
        value = load(ref)
        validate_primary(value, path)
        if value["task_id"] != task["id"]:
            raise ValueError("producer_task_identity")
        refs += authority.terminal(ref, value)
        primitive = value["primitive_reference"]
        work = load(primitive)
        refs.append(primitive)
        for field in ("raw_refs", "raw_references", "upstream_refs"):
            for operand in work.get(field, []):
                verify(operand)
                refs.append(operand)
        if "validation_reference" in value:
            validation = load(value["validation_reference"])
            if validation.get("receipts") != value["validation_receipts"]:
                raise ValueError("producer_validation_drift")
            refs.append(value["validation_reference"])
        if not value["validation_receipts"]:
            raise ValueError("missing_owned_validation")
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append(str(error))
    return dict(authenticated=not errors, references=refs, errors=errors)


def resolve(task: Json, raw: Path, lines: list[str], root: Path = ROOT) -> Json:
    """Record each scheduled slot, including a literal skip with no scientific verdict."""
    path = locate(task, root)
    declared, ref = freeze(root / task["deliverable"], raw), freeze(path, raw)
    value = load(ref) if ref["sha256"] else {}
    skipped = value.get("blocked_at_layer") == "conductor_pre_gate"
    logs = [
        line for line in lines if task["title"][:48].strip() in line and "| GATE_BLOCK |" in line
    ]
    executed = bool(value) and not skipped
    auth = (
        authenticate(path, task)
        if executed
        else dict(authenticated=False, references=[], errors=[])
    )
    qualified = bool(
        executed
        and auth["authenticated"]
        and value.get("required_checks_passed") is True
        and value.get("flagged_adversarial") is False
        and all(r.get("passed") is True for r in value.get("validation_receipts", []))
    )
    verdict = (
        (
            "disqualified"
            if value.get("flagged_adversarial") or value.get("required_checks_passed") is False
            else value.get("verdict_class")
        )
        if executed
        else "blocked"
    )
    if executed and not auth["authenticated"]:
        verdict = "disqualified"
    return dict(
        experiment_id=int(task["id"].split("-")[0][3:]),
        task_id=task["id"],
        unit_id=task["id"],
        arm="scheduled_slot",
        status="completed",
        absolute_metric=int(qualified),
        censored=False,
        excluded=False,
        effective_independent_groups=0,
        producer_executed=executed,
        producer_qualified=qualified,
        scientific_result_measured=qualified
        and verdict in {"positive", "circular_positive", "null"},
        scientific_verdict=value.get("honest_verdict") if executed else None,
        honest_verdict=value.get("honest_verdict"),
        verdict_class=verdict,
        flagged_adversarial=value.get("flagged_adversarial", False),
        adversarial_findings=value.get("adversarial_findings", []),
        disposition="producer"
        if executed
        else "conductor_pre_gate"
        if skipped
        else "conductor_cascade_skip"
        if logs
        else "missing_disposition",
        conductor_disposition=logs,
        declared_reference=declared,
        primary_reference=ref,
        terminal_authenticated=auth["authenticated"],
        authentication_errors=auth["errors"],
        evidence_references=auth["references"],
        missing_reason=None if executed else "producer_not_executed",
        failed_or_missing_gate_operands=value.get("gate_check_summary", []),
    )


def prior_rows(tasks: list[Json], raw: Path, root: Path = ROOT) -> list[Json]:
    """Compare actual prior terminal verdicts; a skip or changed scope is never an exact repeat."""
    rows = []
    for task in tasks:
        for registration in task.get("prior_failures", []):
            identity = registration["experiment_id"]
            number = identity.split("-")[0][3:]
            found = list((root / "results").glob(f"experiment_{number}_*.json"))
            ref = (
                freeze(found[0], raw)
                if len(found) == 1
                else reference(root / "results" / f"experiment_{number}.json")
            )
            value = load(ref) if ref["sha256"] else {}
            auth = (
                authenticate(Path(ref["path"]), dict(id=identity))
                if value.get("primitive_reference")
                else dict(
                    authenticated=False, references=[], errors=["no_terminal_scientific_artifact"]
                )
            )
            rows.append(
                dict(
                    task_id=task["id"],
                    registration=registration,
                    primary_reference=ref,
                    actual_terminal_verdict=value.get("honest_verdict"),
                    terminal_authenticated=auth["authenticated"],
                    references=auth["references"],
                    registered_verdict_matches=value.get("honest_verdict")
                    == registration["verdict"],
                    prior_scope=value.get("scientific_scope", value.get("methodology_note")),
                    current_scope=task["title"],
                    flagged_adversarial=value.get("flagged_adversarial"),
                    required_checks_passed=value.get("required_checks_passed"),
                )
            )
    return rows


def projection(value: Json) -> Json:
    """Keep consumed summaries while full upstream numerical rows remain hash-bound on disk."""
    fields = {
        "historical_model_provenance",
        "historical_failure_reproduction",
        "complete_horizon_count",
        "policy_decision",
        "generalization_floor_decision",
        "rows",
        "complete_operation_cost_rows",
        "accumulator_bound_rows",
        "boundary_rows",
        "decay_error_rows",
        "restart_rows",
        "board_status_rows",
        "operation_support_rows",
        "complete_cost_rows",
        "conditional_acceleration_bounds",
    }
    return {k: v for k, v in value.items() if k in fields or k.endswith("_score")}


def original_inputs(root: Path = ROOT) -> Json:
    """Authenticate registered methods directly; external missing bytes remain blocked operands."""
    expected = dict(
        path=str(root / authority.PROTOCOL), sha256="sha256:" + authority.PINS[authority.PROTOCOL]
    )
    actual = reference(Path(expected["path"]))
    protocol = load(expected) if actual == expected else {}
    registered = [
        expected,
        dict(
            path=str(
                root / "openspec/change-proposals/research-roadmap-v745-preserved-20261101.md"
            ),
            sha256="sha256:01fb99526e816d6ff017196bcd17aad215562ef217bcdf113746fcbfc8e38a63",
        ),
    ]
    keys = [
        "head_checkpoint_reference",
        "learning_data_reference",
        "cohort_partition_reference",
        "test_inventory_reference",
    ]
    checks = [
        dict(
            upstream="original_protocol",
            check="registered_reference",
            **actual,
            field=k,
            operator="present",
            expected="registered path and hash",
            observed=None,
            operand_state="missing_field",
            passed=False,
        )
        for k in keys
        if k not in protocol
    ]
    registered += [protocol[k] for k in keys if k in protocol] + protocol.get(
        "learning_label_references", []
    )
    refs = []
    for ref in registered:
        observed = reference(Path(ref["path"]))
        refs.append(observed)
        checks.append(
            dict(
                upstream="original_methods",
                check="registered_bytes",
                **observed,
                field="bytes_sha256",
                operator="==",
                expected=ref["sha256"],
                observed=observed["sha256"],
                operand_state="missing_file"
                if observed["sha256"] is None
                else "observed"
                if observed["sha256"] == ref["sha256"]
                else "changed_bytes",
                passed=observed["sha256"] == ref["sha256"],
            )
        )
    return dict(
        protocol=protocol,
        references=refs,
        gate_check_summary=checks,
        ready=all(r["passed"] for r in checks),
    )
