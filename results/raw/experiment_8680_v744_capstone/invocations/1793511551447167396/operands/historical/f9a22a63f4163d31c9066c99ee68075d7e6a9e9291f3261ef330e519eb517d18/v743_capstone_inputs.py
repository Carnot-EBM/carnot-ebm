"""REQ-REPORT-8666: retain original bytes before interpreting scheduled outcomes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting import v743_contract_methods as methods
from carnot.reporting.current_work_receipt import atomic_json as atomic_json
from carnot.reporting.current_work_receipt import sha256_file as sha256_file
from carnot.reporting.v742_capstone_evidence import freeze as freeze
from carnot.reporting.v742_capstone_evidence import failure_evidence as failure_evidence
from carnot.reporting.primary_publication import read_bound_sidecar

Json = dict[str, Any]
__all__ = ["methods"]
ROOT: Path = methods.ROOT
NAME, TASK = "experiment_8666_v743_capstone", "exp8666-capstone"
CLI = f"scripts/experiments/{NAME}.py"
TESTS = ["tests/python/test_v743_capstone_8666.py"]
OWNED = [
    f"python/carnot/reporting/v743_capstone{s}.py" for s in ("_inputs", "_reduce", "_report", "")
] + [CLI]
MODEL_SPECS: list[Json] = []
reference, read, checksum = methods.reference, methods.read, methods.checksum


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush boundaries so the conductor can distinguish CPU work from silence."""
    print(f"[exp8666] phase={phase} completed={completed} pending={pending}", flush=True)


def verify(ref: Json) -> None:
    """A copied operand cannot replace the independently named source bytes."""
    if not Path(ref["path"]).is_file():
        if ref.get("exists", True):
            raise ValueError("missing_operand:" + ref["path"])
        return
    if sha256_file(Path(ref["path"])) != ref["sha256"]:
        raise ValueError("original_hash:" + ref["path"])
    if "snapshot_path" in ref and sha256_file(Path(ref["snapshot_path"])) != ref["sha256"]:
        raise ValueError("snapshot_hash:" + ref["path"])


def observe(path: Path, upstream: str, field: str, expected: Any = 1) -> Json:
    """Distinguish a missing producer, an omitted obligation and an observed zero."""
    exists = path.is_file()
    data = json.loads(path.read_bytes()) if exists else {}
    value = data.get(field)
    state = (
        "missing_file"
        if not exists
        else "missing_field"
        if field not in data
        else "observed_zero"
        if value == 0
        else "observed_value"
    )
    return dict(
        upstream=upstream,
        check=field,
        path=str(path),
        sha256=sha256_file(path) if exists else None,
        field=field,
        operator="==",
        expected=expected,
        observed=value,
        operand_state=state,
        passed=exists and field in data and value == expected,
    )


def locate(task: Json) -> Path:
    """Only a declared producer or the conductor's exact task slug identifies a slot."""
    declared = ROOT / task["deliverable"]
    number, slug = task["id"].split("-", 1)
    fallback = ROOT / "results" / f"experiment_{number[3:]}_{slug.replace('-', '_')}.json"
    found = list((ROOT / "results").glob(f"experiment_{number[3:]}_*.json"))
    if len(found) > 1 or any(p not in {declared, fallback} for p in found):
        raise ValueError("unexpected_primary_identity")
    return declared if declared.is_file() or not fallback.is_file() else fallback


def gates(value: Json) -> dict[str, bool]:
    """Qualification and scientific acceptance remain separately named gates."""
    items = value.get("acceptance_gates", {})
    if isinstance(items, list):
        return {r["name"]: r["passed"] is True for r in items}
    return {
        k: (v.get("passed") is True if isinstance(v, dict) else v is True) for k, v in items.items()
    }


def authority(raw: Path) -> Json:
    """Use the original sealed activation even when the next queue has drained."""
    protocol = methods.protocol()
    for key, name in [
        ("v743_design_reference", methods.DESIGN),
        ("v743_activation_reference", methods.ACTIVE),
    ]:
        ref = protocol[key]
        verify(ref)
        target = raw / name
        target.parent.mkdir(parents=True, exist_ok=True)
        original = Path(ref["path"]).read_bytes()
        if target.exists() and target.read_bytes() != original:
            raise ValueError("original_authority_drift")
        if not target.exists():
            target.write_bytes(original)
    return dict(
        methods.authority(raw),
        authority_root=str(raw),
        authority_scope="original_sealed_V743_activation",
    )


def resolve(task: Json, raw: Path, lines: list[str]) -> Json:
    """A gate receipt and a cascade log retain their identities as scheduling evidence."""
    declared = freeze(ROOT / task["deliverable"], raw)
    ref = freeze(locate(task), raw)
    value = read(ref) if ref["exists"] else {}
    gate_block = value.get("blocked_at_layer") == "conductor_pre_gate"
    producer = bool(value) and not gate_block
    evidence = []
    sealed = False
    if producer:
        side = Path(value["terminal_validation_sidecar_path"])
        terminal = json.loads(side.read_bytes())
        bound = Path(terminal.get("publication", terminal).get("sidecar_path", str(side)))
        if "candidate_reference" in terminal:
            candidate = terminal["candidate_reference"]
            verify(candidate)
            if candidate["sha256"] != ref["sha256"] or terminal["passed"] is not True:
                raise ValueError("terminal_candidate_identity")
            bound = (
                Path(ref["path"]).parent
                / "raw"
                / Path(ref["path"]).stem
                / "validators"
                / (ref["sha256"].split(":")[-1] + ".json")
            )
        sealed = read_bound_sidecar(Path(ref["path"]), bound)["report"]["passed"] is True
        evidence = [freeze(side, raw), freeze(bound, raw)]
    accepted = gates(value)
    owned = value.get("required_checks_passed") is True
    eligible = (
        producer
        and sealed
        and owned
        and value.get("flagged_adversarial") is False
        and value.get("verdict_class") in {"positive", "circular_positive", "null"}
    )
    if eligible and value["verdict_class"] in {"positive", "circular_positive"}:
        eligible = bool(accepted) and all(accepted.values())
    logs = [
        line for line in lines if task["title"][:48].strip() in line and "| GATE_BLOCK |" in line
    ]
    if not value and not logs:
        raise ValueError("missing_slot_disposition:" + task["id"])
    return dict(
        experiment_id=int(task["id"][3:7]),
        task_id=task["id"],
        unit_id=task["id"],
        arm="scheduled_slot",
        status="completed",
        censored=False,
        excluded=False,
        absolute_metric=int(eligible),
        effective_independent_groups=0,
        eligible=eligible,
        producer_executed=producer,
        declared_reference=declared,
        primary_reference=ref,
        evidence_references=evidence,
        honest_verdict=value.get("honest_verdict", "complete_blocked_conductor_cascade_skip"),
        verdict_class=value.get("verdict_class", "blocked"),
        disposition="producer"
        if producer
        else "conductor_pre_gate"
        if gate_block
        else "conductor_cascade_skip",
        missing_reason=None if producer else "producer_not_executed",
        acceptance_gates=accepted,
        required_checks_passed=owned,
        terminal_sealed=sealed,
        original_gate_check_summary=value.get("gate_check_summary"),
        failed_owned_checks=failure_evidence(value),
        conductor_receipt=logs,
    )


def closure(value: Any, raw: Path, refs: dict[str, Json]) -> None:
    """Copy only explicit path/hash operands, never inferred filenames or directories."""
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and str(value.get("sha256", "")).startswith(
            "sha256:"
        ):
            path = Path(value["path"])
            if path.is_file() and str(path) not in refs:
                refs[str(path)] = freeze(path, raw)
                refs[str(path)]["declared_source_sha256"] = value["sha256"]
                refs[str(path)]["declared_source_matches"] = (
                    refs[str(path)]["sha256"] == value["sha256"]
                )
        for child in value.values():
            closure(child, raw, refs)
    elif isinstance(value, list):
        for child in value:
            closure(child, raw, refs)
