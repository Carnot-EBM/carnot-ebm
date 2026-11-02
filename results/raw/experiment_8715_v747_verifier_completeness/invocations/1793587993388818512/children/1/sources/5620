"""REQ-REPORT-8694: preserve producer bytes and literal scheduling evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting import v745_contract_methods as methods
from carnot.reporting import v744_capstone_inputs as previous
from carnot.reporting.primary_publication import read_bound_sidecar, validate_primary

Json = dict[str, Any]
__all__ = ["methods"]
ROOT: Path = methods.ROOT
NAME, TASK = "experiment_8694_v745_capstone", "exp8694-capstone"
CLI = f"scripts/experiments/{NAME}.py"
TESTS = ["tests/python/test_v745_capstone_8694.py"]
OWNED = [
    f"python/carnot/reporting/v745_capstone{s}.py" for s in ("_inputs", "_reduce", "_report", "")
] + [CLI]
MODEL_SPECS: list[Json] = []
reference, read, checksum = methods.reference, methods.read, methods.checksum
atomic_json, sha256_file = methods.atomic_json, methods.sha256_file
freeze, verify = previous.freeze, previous.verify
HISTORY = "results/experiment_8680_v744_capstone.json"
HISTORY_PIN = "sha256:fee16c8c2496f1afa5f6dc0b547655dd483cd8ffc71380d3372a446e3fe425c3"


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real phase counts so short CPU reductions remain supervised."""
    print(f"[exp8694] phase={phase} completed={completed} pending={pending}", flush=True)


def protocol() -> Json:
    """A separately fixed task digest prevents coordinated roadmap rewriting."""
    return dict(methods.protocol())


def locate(task: Json) -> Path:
    """Prefer the exact deliverable; accept only the conductor's exact fallback slug."""
    declared = ROOT / task["deliverable"]
    number, slug = task["id"].split("-", 1)
    fallback = ROOT / "results" / f"experiment_{number[3:]}_{slug.replace('-', '_')}.json"
    found = list((ROOT / "results").glob(f"experiment_{number[3:]}_*.json"))
    if len(found) > 1 or any(p not in {declared, fallback} for p in found):
        raise ValueError("unexpected_primary_identity")
    return declared if declared.is_file() or not fallback.is_file() else fallback


def custody(path: Path, task: Json) -> Json:
    """Terminal acceptance authenticates bytes while historical stamps still veto readiness."""
    refs: list[Json] = []
    try:
        value = read(reference(path))
        validate_primary(value, path)
        if value["task_id"] != task["id"]:
            raise ValueError("declared_producer_identity")
        side = Path(value["terminal_validation_sidecar_path"])
        refs.append(reference(side))
        terminal = read(refs[-1])
        publication = terminal.get("publication", terminal)
        bound = Path(publication["sidecar_path"])
        refs.append(reference(bound))
        checked = read_bound_sidecar(path, bound)
        sealed = bool(
            publication["primary_sha256"] == sha256_file(path)
            and checked["report"]["passed"] is True
        )
        return dict(sealed=sealed, references=refs, error=None)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return dict(sealed=False, references=refs, error=str(error))


def observe(
    path: Path, task: Json, field: str, expected: Any = 1, expected_hash: str | None = None
) -> Json:
    """Missing, zero, drift and a flagged producer are different failed operands."""
    row = previous.observe(path, task["id"], field, expected, expected_hash=expected_hash)
    auth = custody(path, task) if path.is_file() else dict(sealed=False)
    value = json.loads(path.read_bytes()) if path.is_file() else {}
    row.update(
        producer_terminal_authenticated=auth["sealed"],
        producer_flagged=value.get("flagged_adversarial"),
    )
    row["passed"] = bool(
        row["passed"]
        and auth["sealed"]
        and value.get("required_checks_passed") is True
        and value.get("flagged_adversarial") is False
    )
    return dict(row)


def resolve(task: Json, raw: Path, lines: list[str]) -> Json:
    """An artifact-free skip has a literal log disposition and no invented verdict."""
    declared = freeze(ROOT / task["deliverable"], raw)
    ref = freeze(locate(task), raw)
    value = read(ref) if ref["exists"] else {}
    logs = [
        line for line in lines if task["title"][:48].strip() in line and "| GATE_BLOCK |" in line
    ]
    gate_block = value.get("blocked_at_layer") == "conductor_pre_gate"
    producer = bool(value) and not gate_block
    auth = (
        custody(Path(ref["path"]), task)
        if producer
        else dict(sealed=False, references=[], error=None)
    )
    if not value and not logs:
        raise ValueError("missing_slot_disposition:" + task["id"])
    qualified = bool(
        producer
        and auth["sealed"]
        and value.get("required_checks_passed") is True
        and value.get("flagged_adversarial") is False
    )
    eligible = qualified and value.get("verdict_class") in {"positive", "circular_positive", "null"}
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
        producer_executed=producer,
        eligible=eligible,
        producer_qualified=qualified,
        scientific_result_measured=eligible,
        declared_reference=declared,
        primary_reference=ref,
        evidence_references=auth["references"],
        terminal_sealed=auth["sealed"],
        terminal_error=auth["error"],
        honest_verdict=value.get("honest_verdict"),
        verdict_class=value.get("verdict_class"),
        flagged_adversarial=value.get("flagged_adversarial"),
        adversarial_findings=value.get("adversarial_findings", []),
        failed_or_missing_gate_operands=value.get("gate_check_summary", []),
        disposition="producer"
        if producer
        else "conductor_pre_gate"
        if gate_block
        else "conductor_cascade_skip",
        conductor_disposition="\n".join(logs),
        missing_reason=None if producer else "producer_not_executed",
    )
