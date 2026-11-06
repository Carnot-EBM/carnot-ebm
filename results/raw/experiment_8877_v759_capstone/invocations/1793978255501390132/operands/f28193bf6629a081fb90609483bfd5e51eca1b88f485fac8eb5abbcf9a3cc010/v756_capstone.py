"""REQ-REPORT-8839: separate current accounting from admitted scientific benefit."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np

from carnot.reporting import v755_capstone as shared
from carnot.reporting import v756_branch_contract as authority
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
)

Json = dict[str, Any]
ROOT, MILESTONE = authority.ROOT, authority.MILESTONE
NAME, TASK = "experiment_8839_v756_capstone", "exp8839-capstone"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [f"python/carnot/reporting/v756_capstone{s}.py" for s in ["", "_report", "_runner"]] + [CLI]
TEST = "tests/python/test_v756_capstone_8839.py"
DEPENDENCIES = [
    "python/carnot/reporting/" + name + ".py"
    for name in [
        "v755_capstone",
        "v755_branch_runner",
        "v756_branch_contract",
        "current_work_receipt",
        "primary_publication",
        "roadmap_contract",
        "learning_signal_runner_8815",
        "selective_methods_8829",
        "ordered_evidence_cost_8831",
        "complete_adaptation_cost_8835",
    ]
] + [
    "scripts/publication_gate.py",
    "scripts/adversarial_verify.py",
    "scripts/verdict_row_consistency_lint.py",
]
reference, checksum, retain = shared.reference, shared.checksum, shared.retain


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush boundaries so completed accounting remains visible to the supervisor."""
    print(f"[exp8839] phase={phase} completed={completed} pending={pending}", flush=True)


def admit(path: Path, task_id: str) -> Json:
    """Validate receipt logs too; a producer's own success flag cannot authenticate its bytes."""
    result = shared.admit(path, task_id)
    refs = shared.references(result["value"])
    for receipt in result["value"].get("validation_receipts", []):
        if "log_path" in receipt:
            refs.append(dict(path=str(ROOT / receipt["log_path"]), sha256=receipt["log_sha256"]))
    result["references"] = refs
    for ref in refs:
        if reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
            result["admitted"] = False
            result["errors"].append("operand_or_validation_log_drift:" + ref["path"])
    return result


def preflight(root: Path, raw: Path) -> Json:
    """Check internal resources before any row reduction; absence is never synthetic data."""
    progress("preconditions_before")
    raw.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(raw).free
    if free < 100_000_000:
        raise ValueError("insufficient_disk")
    if not (root / ".venv/bin/python").is_file():
        raise ValueError("repository_virtualenv_missing")
    probe = raw / "write_check"
    probe.write_bytes(b"private output writable")
    probe.unlink()
    authority.history(root)
    progress("preconditions_after")
    return dict(
        repository_python=reference(root / ".venv/bin/python"),
        executable=sys.executable,
        numpy_version=np.__version__,
        free_bytes=free,
        private_output_writable=True,
        history_hashes_checked=True,
        required_operands_checked_before_reduction=True,
    )


def reduce(value: Json) -> Json:
    """Recompute registered quantities from rows, keeping reused worlds and CPU costs separate."""
    rows = [r for r in value.get("rows", []) if r["status"] == "completed"]
    number = value["experiment_id"]
    measured = number in {8828, 8829, 8831, 8832, 8833, 8834}
    estimands: Json = {}
    if number == 8828:
        from carnot.reporting.learning_signal_runner_8815 import reduce as learning_reduce

        evidence = json.loads(Path(value["raw_shard_hashes"][0]["path"]).read_bytes())
        rebuilt = learning_reduce(evidence)
        if rebuilt["rows"] != rows:
            raise ValueError("learning_row_drift")
        estimands = {
            k: rebuilt[k]
            for k in ["objective_selection", "empirical_feature_floor", "mechanism_supported"]
        }
    if number == 8829:
        from carnot.reporting.selective_methods_8829 import diagnose, select

        calibration = [r for r in rows if r["split"] == "calibration"]
        estimands = dict(
            finite_support_rows=diagnose(calibration), schedule_selection=select(calibration)
        )
        if estimands["finite_support_rows"] != value["finite_support_rows"]:
            raise ValueError("support_row_drift")
    if number == 8831:
        from carnot.reporting.ordered_evidence_cost_8831 import reduce as ordered_reduce

        estimands = ordered_reduce(rows, value["selected_schedule"])
    if number == 8835:
        from carnot.reporting.complete_adaptation_cost_8835 import reduce_costs

        estimands = reduce_costs(rows)
    if number == 8838:
        estimands = dict(
            derived_operation_rows=len(rows),
            sampling_call_count=sum(
                r.get("operation_counts", {}).get("sampler_calls", 0) for r in rows
            ),
            scientific_measurements_inherited=False,
        )
    return dict(
        constructed_measurement_count=len(rows) if measured else 0,
        independent_constructed_unit_count=len({r["cluster_id"] for r in rows}) if measured else 0,
        constructed_unit_ids=sorted({r["cluster_id"] for r in rows}) if measured else [],
        natural_generalization_count=0,
        cpu_cost_window_count=len(rows) if number == 8835 else 0,
        exact_oracle_verdict="circular_positive" if value.get("verifier_is_oracle") else None,
        registered_estimands=estimands,
    )


def retirement(prior: Json, row: Json, actual_prior: str | None) -> Json:
    """Exact repeated verdicts can retire a scope; a class or environmental skip cannot."""
    same = bool(
        actual_prior
        and actual_prior == prior["verdict"] == row["primary_honest_verdict"]
        and row["verdict_class"] in {"null", "blocked", "disqualified"}
    )
    retire = same and prior["retire_if_same_verdict"] is True
    return dict(
        prior_verdict=actual_prior,
        registered_prior_verdict=prior["verdict"],
        new_verdict=row["primary_honest_verdict"],
        diagnosed_cause=prior["verdict"],
        changed_mechanism=prior["addressed_by"],
        observed_gate=row.get("admission_errors", []),
        retire_exact_scope=retire,
        recommendation="retire_exact_scope" if retire else "require_changed_evidence",
        evidence_needed_to_reopen=prior["addressed_by"],
        broader_hypothesis_retired=False,
        manifest_edited=False,
    )


def freeze(root: Path, raw: Path, run_date: str) -> Json:
    """Freeze current objects, rejected bytes and historical authority before reducing admitted rows."""
    contract = authority.authority(root)
    if not contract["ready"]:
        raise ValueError("v756_full_object_authority")
    preconditions = preflight(root, raw)
    sources = [
        authority.DESIGN,
        authority.ACTIVE,
        "ops/conductor-log.md",
        "ops/exclusion_manifest.yaml",
        *authority.HISTORY_PINS,
    ]
    snapshots = [retain(root / p, raw / "authority") for p in sources]
    log = (root / "ops/conductor-log.md").read_text().splitlines()
    slots = []
    # Shared outcome parsing keeps old scheduler schemas intact; admission is checked again here.
    for index, task in enumerate(contract["tasks"][:-1]):
        progress("slot_before", index, 13 - index)
        row = shared.outcome(task, root, raw, log)
        checked = admit(root / task["deliverable"], task["id"])
        row["admitted"], row["admission_errors"] = checked["admitted"], checked["errors"]
        if not checked["admitted"] and row["disposition"] == "admitted_primary":
            row["disposition"] = "rejected_producer"
        row["snapshots"] += [
            retain(Path(r["path"]), raw / "operands") for r in checked["references"]
        ]
        row["reduction"] = reduce(row["producer"]) if row["admitted"] else {}
        row["producer"] = shared.producer_summary(row["producer"])
        slots.append(row)
        progress("slot_after", index + 1, 12 - index)
    by_id = {t["id"]: t for t in contract["tasks"]}
    edges = [
        dict(
            shared.gate(
                root / by_id[g["upstream"]]["deliverable"],
                g["upstream"],
                g["artifact_field"],
                g["value"],
            ),
            downstream=t["id"],
            op=g["op"],
        )
        for t in contract["tasks"]
        for g in t["gated_on"]
    ]
    history = json.loads((root / authority.HISTORY).read_bytes())
    work = dict(
        contract=contract,
        slots=slots,
        edges=edges,
        snapshots=snapshots,
        historical=history,
        preconditions=preconditions,
        run_date=run_date,
        raw=str(raw.absolute()),
        code=[reference(ROOT / p) for p in OWNED + [TEST] + DEPENDENCIES],
    )
    atomic_json(raw / "work.json", work)
    progress("custody_after", 12, 1)
    return work


from carnot.reporting.v756_capstone_report import build as build, replay as replay
