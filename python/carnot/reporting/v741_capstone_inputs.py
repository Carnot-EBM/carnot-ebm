"""REQ-REPORT-8639: retain producer gates before assigning scientific eligibility."""

from pathlib import Path
from typing import Any

from carnot.reporting import v741_contract_methods as methods
from carnot.reporting.primary_publication import read_bound_sidecar, reader_receipt
from carnot.reporting.v735_capstone_evidence import freeze
from scripts.recurring_blocker_ledger import normalise

Json = dict[str, Any]


def locate(task: Json) -> Path:
    """Conductor receipts can use shortened names and lack task_id fields."""
    paths = list((methods.ROOT / "results").glob(f"experiment_{task['id'][3:7]}_*.json"))
    if len(paths) > 1:
        raise ValueError("ambiguous_original_primary")
    return Path(paths[0] if paths else methods.ROOT / task["deliverable"])


def resolve(task: Json, raw: Path, lines: list[str]) -> Json:
    """An authenticated scheduling receipt is complete without executed science."""
    ref = freeze(locate(task), raw / task["id"])
    activations = [
        i for i, line in enumerate(lines) if f"Milestone {methods.MILESTONE} activated" in line
    ]
    current = lines[activations[-1] + 1 :] if activations else []
    conductor = [
        line
        for line in current
        if task["title"][:48].strip() in line
        and any(status in line for status in ["| GATE_BLOCK |", "| OK |", "| FLAGGED |"])
    ]
    row: Json = dict(
        experiment_id=int(task["id"][3:7]),
        task_id=task["id"],
        unit_id=task["id"],
        primary_reference=ref,
        producer_executed=False,
        eligible=False,
        status="completed"
        if ref["exists"] or any("| GATE_BLOCK |" in line for line in conductor)
        else "missing",
        censored=False,
        excluded=False,
        absolute_metric=0,
        effective_independent_groups=0,
        evidence_references=[],
        checks=[],
        conductor_receipt=conductor,
        missing_reason=None,
    )
    if not ref["exists"]:
        row.update(
            verdict_class="blocked",
            honest_verdict="complete_blocked_missing_primary",
            disposition="conductor_cascade_skip",
            missing_reason="producer_primary_absent",
        )
        row["checks"] = [
            methods.gate(Path(ref["path"]), "producer_primary_exists", True, None, task["id"])
        ]
        return row
    value = methods.read(ref)
    row["original_honest_verdict"] = value["honest_verdict"]
    row["checks"] = value["gate_check_summary"]
    if value.get("blocked_at_layer") == "conductor_pre_gate":
        upstream = dict(path=value["failed_evidence_path"], sha256=value["failed_evidence_sha256"])
        methods.read(upstream)
        row["checks"] = [
            methods.gate(
                Path(upstream["path"]),
                value["failed_field"],
                value["failed_expected"],
                value["failed_observed"],
                value["failed_upstream"],
            )
        ]
        row.update(
            verdict_class="blocked",
            honest_verdict="complete_blocked_conductor_pre_gate",
            disposition="conductor_pre_gate",
            missing_reason="producer_not_executed",
            evidence_references=[upstream],
        )
        return row
    refs = [
        value[k]
        for k in ["primitive_reference", "work_reference", "authority_reference"]
        if k in value
    ]
    authenticated = True
    for operand in refs:
        actual = methods.sha256_file(Path(operand["path"]))
        if actual != operand["sha256"]:
            authenticated = False
            row["checks"] += [
                methods.gate(
                    Path(operand["path"]),
                    "original_operand_sha256",
                    operand["sha256"],
                    actual,
                    task["id"],
                )
            ]
    terminal = methods.reference(Path(value["terminal_validation_sidecar_path"]))
    side = methods.read(terminal)
    bound_path = Path(side.get("publication", side).get("sidecar_path", terminal["path"]))
    sealed = False
    try:
        bound = read_bound_sidecar(Path(ref["path"]), bound_path)
        sealed = bound["primary_sha256"] == ref["sha256"] and bound["report"]["passed"] is True
    except (OSError, ValueError, KeyError):
        pass
    gates = value["acceptance_gates"]
    accepted = all(gates.values()) if isinstance(gates, dict) else all(r["passed"] for r in gates)
    consumer = reader_receipt(
        task["id"],
        methods.ROOT / "results",
        field="required_checks_passed",
        expected=value["required_checks_passed"],
    )
    eligible = bool(
        authenticated
        and sealed
        and consumer["passed"]
        and value["required_checks_passed"]
        and not value["flagged_adversarial"]
        and accepted
        and value["verdict_class"] in {"positive", "circular_positive", "null"}
    )
    row.update(
        producer_executed=True,
        eligible=eligible,
        verdict_class=value["verdict_class"],
        honest_verdict=value["honest_verdict"],
        acceptance_gates=gates,
        original_required_checks_passed=value["required_checks_passed"],
        upstream_flagged_adversarial=value["flagged_adversarial"],
        terminal_sealed=sealed,
        downstream_consumer=consumer,
        evidence_references=[methods.reference(Path(r["path"])) for r in refs]
        + [terminal, methods.reference(bound_path)],
        disposition="qualified_producer" if eligible else "quarantined_producer",
    )
    if not sealed:
        row["checks"] += [
            methods.gate(Path(ref["path"]), "original_terminal_seal", True, False, task["id"])
        ]
    return row


def bindings(tasks: list[Json]) -> list[Json]:
    """Read exact declared field spelling; distinguish absent paths, fields and zeros."""
    by_id = {task["id"]: task for task in tasks}
    result = []
    for task in tasks:
        for requirement in task.get("gated_on", []):
            path = locate(by_id[requirement["upstream"]])
            exists = path.is_file()
            ref = methods.reference(path) if exists else dict(path=str(path), sha256=None)
            value = methods.read(ref) if exists else {}
            field = requirement["artifact_field"]
            observed = value.get(field)
            result.append(
                dict(
                    methods.gate(
                        path, field, requirement["value"], observed, requirement["upstream"]
                    ),
                    downstream=task["id"],
                    passed=exists and field in value and observed == requirement["value"],
                    declared_operator=requirement["op"],
                    operand_state="missing_path"
                    if not exists
                    else "missing_field"
                    if field not in value
                    else "observed_zero"
                    if observed == 0
                    else "observed_value",
                )
            )
    return result


def cause(row: Json) -> str:
    """Failure identity includes actual operands so milestone suffixes cannot hide it."""
    checks = row.get("checks") or row.get("gate_check_summary") or []
    failed = [r for r in checks if isinstance(r, dict) and not r.get("passed", False)]
    return str(
        normalise(
            " ".join(
                str(r.get("field", r.get("artifact_field", r.get("check", "")))) for r in failed
            )
            or row.get("missing_reason")
            or row.get(
                "original_honest_verdict",
                row.get("honest_verdict", "complete_blocked_unrecorded_prior_verdict"),
            )
        )
    )
