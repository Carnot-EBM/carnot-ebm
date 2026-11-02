"""REQ-REPORT-8680: original authority and original bytes constrain every claim."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import v744_contract_methods as methods
from carnot.reporting import v743_capstone_inputs as previous
from carnot.reporting.current_work_receipt import atomic_json as atomic_json
from carnot.reporting.current_work_receipt import sha256_file as sha256_file

Json = dict[str, Any]
__all__ = ["methods"]
ROOT = methods.ROOT
NAME, TASK = "experiment_8680_v744_capstone", "exp8680-capstone"
CLI = f"scripts/experiments/{NAME}.py"
TESTS = ["tests/python/test_v744_capstone_8680.py"]
OWNED = [
    f"python/carnot/reporting/v744_capstone{s}.py" for s in ("_inputs", "_reduce", "_report", "")
] + [CLI]
MODEL_SPECS: list[Json] = []
reference, read, checksum = methods.reference, methods.read, methods.checksum
freeze, verify, locate = previous.freeze, previous.verify, previous.locate


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush phase boundaries so the conductor can supervise real progress."""
    print(f"[exp8680] phase={phase} completed={completed} pending={pending}", flush=True)


def observe(
    path: Path, upstream: str, field: str, expected: Any = 1, *, expected_hash: str | None = None
) -> Json:
    """Changed source bytes cannot satisfy a field gate under an earlier identity."""
    row: Json = dict(previous.observe(path, upstream, field, expected))
    row["expected_hash"] = expected_hash
    if path.is_file() and expected_hash is not None and row["sha256"] != expected_hash:
        row.update(operand_state="changed_bytes", passed=False)
    return row


def protocol() -> Json:
    """Authenticate the sealed registration without requiring mutable live agent bytes."""
    value: Json = read(dict(path=str(ROOT / methods.PROTOCOL), sha256=methods.PROTOCOL_PIN))
    if methods.tasks_digest(value["task_map_v744"]) != methods.TASKS_PIN:
        raise ValueError("independent_task_authority")
    for key in (
        "v744_design_reference",
        "v744_activation_reference",
        "head_checkpoint_reference",
        "cohort_partition_reference",
    ):
        verify(value[key])
    return value


def authority(raw: Path, p: Json) -> Json:
    """The independently pinned complete design remains authority after scheduling drains."""
    for key, name in [
        ("v744_design_reference", methods.DESIGN),
        ("v744_activation_reference", methods.ACTIVE),
    ]:
        ref = p[key]
        verify(ref)
        target = raw / name
        target.parent.mkdir(parents=True, exist_ok=True)
        data = Path(ref["path"]).read_bytes()
        if target.exists() and target.read_bytes() != data:
            raise ValueError("sealed_authority_changed")
        target.write_bytes(data)
    with patch.object(methods, "protocol", lambda: p):
        result: Json = methods.authority(raw)
    return dict(result, authority_root=str(raw))


def resolve(task: Json, raw: Path, lines: list[str]) -> Json:
    """Literal producer verdicts and stamps survive separate scheduling normalization."""
    row: Json = previous.resolve(task, raw, lines)
    value = read(row["primary_reference"]) if row["primary_reference"]["exists"] else {}
    row.update(
        literal_verdict_class=value.get("verdict_class"),
        flagged_adversarial=value.get("flagged_adversarial"),
        adversarial_findings=value.get("adversarial_findings", []),
        failed_or_missing_gate_operands=value.get("gate_check_summary", []),
        scientific_result_measured=row["producer_executed"] and row["eligible"],
    )
    return row


def decisions(p: Json, rows: list[Json], prior: dict[str, Json]) -> list[Json]:
    """An unexecuted hypothesis cannot be retired by a repeated scheduling receipt."""
    result = []
    for task, row in zip(p["task_map_v744"], rows, strict=True):
        history = []
        for registration in task.get("prior_failures", []):
            actual = prior.get(registration["experiment_id"], {})
            literal = actual.get("honest_verdict")
            repeat = literal == registration["verdict"] == row["honest_verdict"]
            history.append(
                dict(
                    registration,
                    literal_verdict=literal,
                    exact_authenticated_repeat=repeat,
                    retire_scope=bool(repeat and registration["retire_if_same_verdict"]),
                )
            )
        scientific = row["scientific_result_measured"]
        decision = (
            "close_registered_diagnostic"
            if scientific and row["experiment_id"] in {8677, 8678}
            else "retain_qualified_scope"
            if scientific
            else "require_changed_operands"
        )
        result.append(
            dict(
                task_id=task["id"],
                decision=decision,
                actual_prior_outcomes=history,
                retire_exact_scope=any(r["retire_scope"] for r in history),
                retired_scientific_hypothesis=False,
                reopening_operands=[dict(g) for g in task.get("gated_on", [])],
                scope=task["title"],
                producer_executed=row["producer_executed"],
            )
        )
    return result
