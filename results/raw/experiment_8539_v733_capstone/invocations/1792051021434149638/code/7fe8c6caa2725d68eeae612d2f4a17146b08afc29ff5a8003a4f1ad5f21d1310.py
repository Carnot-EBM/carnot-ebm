"""Publish one reader-visible primary while keeping evidence below raw/.

REQ-REPORT-7928: the existing readers rank only top-level JSON mtimes. A
producer lock prevents competing writers from exposing two primary identities.
"""

from collections.abc import Callable, Mapping
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import re
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from scripts.conductor_gates import _find_artifact_by_task_id, evaluate_gates
from scripts.in_process_doc_reconcile import classify_artifact, find_artifact


def validate_primary(value: Any, output: Path) -> None:
    """Reject ambiguous identity and unsafe readiness before readers see bytes."""
    match = re.fullmatch(r"experiment_(\d+)_[A-Za-z0-9_-]+\.json", output.name)
    if match is None:
        raise ValueError("primary_name")
    if not isinstance(value, dict):
        raise ValueError("primary_object")
    identity = int(match.group(1))
    if value.get("experiment_id") != identity or not str(value.get("task_id", "")).startswith(
        f"exp{identity}-"
    ):
        raise ValueError("producer_identity")
    if not str(value.get("honest_verdict", "")).startswith("complete_") or value.get(
        "verdict_class"
    ) not in {"positive", "circular_positive", "null", "blocked", "disqualified"}:
        raise ValueError("terminal_verdict")
    if value.get("artifact_resolution_ready_score", 0) and (
        value.get("flagged_adversarial") is not False
        or value["verdict_class"] in {"blocked", "disqualified"}
    ):
        raise ValueError("unsafe_readiness")


def publish_primary(
    output: Path, value: Mapping[str, Any], validator: Callable[[Path], dict[str, Any]]
) -> dict[str, Any]:
    """Validate a locked candidate and replace only its checked bytes atomically.

    Validator reports use hash-specific paths so an older reader can detect a
    replaced primary. Historical siblings cause rejection and remain untouched.
    """
    output = output.absolute()
    validate_primary(value, output)
    raw = output.parent / "raw" / output.stem
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "publication.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        siblings = list(output.parent.glob(f"experiment_{value['experiment_id']}_*.json"))
        if any(path != output for path in siblings):
            raise ValueError("conflicting_primary")
        if output.exists():
            old = json.loads(output.read_text())
            if (old.get("experiment_id"), old.get("task_id")) != (
                value["experiment_id"],
                value["task_id"],
            ):
                raise ValueError("conflicting_identity")
        candidate = raw / "terminal_candidate.json"
        atomic_json(candidate, value)
        digest = sha256_file(candidate)
        report = validator(candidate)
        if report.get("passed") is not True:
            raise ValueError("candidate_rejected")
        if sha256_file(candidate) != digest:
            raise ValueError("candidate_changed")
        sidecar = raw / "validators" / (digest.split(":")[1] + ".json")
        atomic_json(
            sidecar, {"primary_path": str(output), "primary_sha256": digest, "report": report}
        )
        temporary = output.with_name("." + output.name + ".checked")
        with temporary.open("wb") as stream:
            stream.write(candidate.read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output)
        return {"primary_path": str(output), "primary_sha256": digest, "sidecar_path": str(sidecar)}


def read_bound_sidecar(primary: Path, sidecar: Path) -> dict[str, Any]:
    """Reject reports from other locations or earlier primary bytes."""
    raw = primary.parent / "raw" / primary.stem
    if not sidecar.resolve().is_relative_to(raw.resolve()):
        raise ValueError("sidecar_location")
    value: dict[str, Any] = json.loads(sidecar.read_text())
    if value.get("primary_sha256") != sha256_file(primary):
        raise ValueError("stale_primary_hash")
    return value


def reader_receipt(
    task_id: str,
    results: Path,
    *,
    field: str = "artifact_resolution_ready_score",
    expected: Any = 1,
    op: str = "==",
) -> dict[str, Any]:
    """Read through both live consumers and record exact selected identity.

    The same scalar gate remains in use; this function reports its outcome and
    does not substitute a different resolver or change imported consumer code.
    """
    task = {
        "gated_on": [{"upstream": task_id, "artifact_field": field, "op": op, "value": expected}]
    }
    selected = _find_artifact_by_task_id(task_id, results)
    checked = evaluate_gates(task, results)
    document = find_artifact(task_id, results)
    status = None
    if document is not None:
        try:
            status = classify_artifact(json.loads(document.read_text()))
        except (ValueError, OSError):
            status = "unreadable"
    return {
        "task_id": task_id,
        "passed": checked.passed and selected == document and selected is not None,
        "gate_path": str(selected) if selected else None,
        "gate_sha256": sha256_file(selected) if selected else None,
        "document_path": str(document) if document else None,
        "document_sha256": sha256_file(document) if document else None,
        "document_status": status,
        "gates": [asdict(row) for row in checked.gates_evaluated],
        "summary": checked.summary,
    }
