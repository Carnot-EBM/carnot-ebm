"""REQ-REPORT-8511: original evidence seals keep unavailable studies from becoming results."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

import yaml

from carnot.reporting import v730_capstone_evidence as old
from carnot.reporting import v731_contract_methods as methods
from carnot.reporting.current_work_receipt import atomic_json, sha256_file

Json = dict[str, Any]
ROOT = old.ROOT
NAME, TASK, MILESTONE = "experiment_8511_v731_capstone", "exp8511-capstone", "2026.10.731"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v731_capstone_8511.py"
OWNED = [f"python/carnot/reporting/v731_capstone{s}.py" for s in ["_evidence", "_report", ""]] + [
    CLI
]
MODEL_SPECS: list[Json] = []
PREDECESSOR_PINS = old.PREDECESSOR_PINS
PINS = {
    8499: "3430e5fe6ec35f6a3327d86145f83b58169d2e5528b9df2d93900c82fec45b39",
    8509: "cf0536bf85f9d22dd26c836e8ef67945f061290a35ecda5b7cbaede65ba8a1da",
    8510: "71504b2e6351aa7bb112f92a59bb071b3be2f898b3e81a292f89dc3173bd731c",
}
READERS = {
    8499: "reporting.v731_contract_methods",
    8509: "reporting.arc_redirect_effect_8509",
    8510: "reporting.board_operation_boundary_8510",
}
PREGATE_PINS = {
    8500: "f556c15ef08ad72c37bdbb4e329a99f2a88610678c1277b6c29387ab6b6645b2",
    8503: "49de95317ef068c8ac8699cfe7d6ca6e0a8173903eafbfbb8284457feaf84f4c",
    8506: "1920dbf7655b008190a31399144425fce772bef3246616ad2de4ded460e34be3",
}
checksum, reference, freeze, load, slot, memory = (
    old.checksum,
    old.reference,
    old.freeze,
    old.load,
    old.slot,
    old.memory,
)
SELECT = old.SELECT | {
    "effect_counts",
    "support_rows",
    "arm_support_rows",
    "model_free_readiness",
    "retention_simultaneous_upper95",
    "cost_boundary",
    "operation_rows",
    "censoring_counts",
    "effect_support_by_arm",
    "resolved_redirect_count",
    "mode_counts",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Actual counts distinguish slow bounded readers from abandoned subprocesses."""
    if phase.startswith("start_"):
        pending = 13
    if phase == "terminal_publication":
        completed = 13
    print(f"[exp8511] phase={phase} completed={completed} pending={pending}", flush=True)


def gate(ref: Json, field: str, expected: Any, observed: Any, upstream: str = TASK) -> Json:
    """Every failure names its exact operand, including missing values rather than zero."""
    return dict(old.base.gate(ref, field, expected, observed, upstream))


def outcome(task: Json, primary: Json, raw: Path, closure: Any = None) -> Json:
    """Qualification means replayable bytes; a qualified block still grants no benefit."""
    number = int(task["id"][3:7])
    if number < 8499:
        result = dict(old.outcome(task, primary, raw, closure))
        source = load(primary)
        result.update(
            original_run_date=source.get("run_date"),
            upstream_flagged_adversarial=source.get("flagged_adversarial"),
            original_required_checks_passed=source.get("required_checks_passed"),
        )
        return result
    value = load(primary)
    if number in PREGATE_PINS:
        with patch.dict(old.base.PINS, PREGATE_PINS):
            return dict(old.base.outcome(task, primary, raw, closure))
    if primary["sha256"] != "sha256:" + PINS[number]:
        raise ValueError("primary_pin")
    old.base.validate_primary(value, Path(primary["path"]))
    if value["task_id"] != task["id"]:
        raise ValueError("task_identity")
    terminal = load(reference(Path(value["terminal_validation_sidecar_path"])))
    side = load(reference(Path(terminal["publication"]["sidecar_path"])))
    sealed = (
        terminal["publication"]["primary_sha256"] == primary["sha256"] == side["primary_sha256"]
        and side["report"]["passed"] is True
    )
    reader = importlib.import_module("carnot." + READERS[number])
    replayed = bool(reader.replay(value if number == 8509 else Path(primary["snapshot_path"])))
    qualified = bool(
        sealed
        and replayed
        and value["required_checks_passed"]
        and not value["flagged_adversarial"]
        and value["verdict_class"] != "disqualified"
    )
    return dict(
        producer_executed=True,
        eligible=qualified,
        selected={k: v for k, v in value.items() if k in SELECT or k.endswith("_ready_score")},
        checks=[]
        if qualified
        else [
            gate(
                primary,
                "producer_terminal_replay",
                True,
                dict(sealed=sealed, replayed=replayed),
                task["id"],
            )
        ],
        disposition="qualified_producer" if qualified else "disqualified_producer",
        honest_verdict=value["honest_verdict"],
        original_run_date=value.get("run_date"),
        upstream_flagged_adversarial=value.get("flagged_adversarial"),
        verdict_class=value["verdict_class"],
        source_gates=value.get("gate_check_summary", []),
        historical_model_provenance=value.get("historical_model_provenance", []),
        upstream_adversarial_findings=value.get("adversarial_findings", []),
        primitive_reference=value.get("primitive_reference", value.get("work_reference")),
        branch_replay=dict(
            status="replayed" if qualified else "disqualified_upstream_evidence", passed=replayed
        ),
    )


def worker_process(request: Path, output: Path) -> int:
    """Shipped workers measure fresh-process peak and growth after reader imports."""
    with patch.multiple(old, READERS=old.READERS | READERS, outcome=outcome, progress=progress):
        return int(old.worker_process(request, output))


def invoke(task: Json, primary: Json, raw: Path) -> tuple[Json, Json]:
    """A fresh worker isolates each branch and retains its real command and log hashes."""
    with patch.multiple(old.workers, CLI=CLI, progress=progress):
        result, receipt = old.workers.invoke(task, primary, raw)
    return dict(result), dict(receipt)


def measure(root: Path, raw: Path, private: Path) -> Json:
    """Read only exact declared paths, with original authorities for historical branches."""
    start, before = time.monotonic_ns(), memory()
    progress("preconditions", 0, 13)
    authority = methods.authority(root, raw / "authority")
    checks = list(authority["gate_check_summary"])
    resources = dict(
        old.history.previous.resources(private),
        task_cap_s=4800,
        measurement_cap_s=3600,
        validation_reserve_s=900,
        optional_cuda_required=False,
        optional_boards_required=False,
    )
    if not (
        resources["disk_backed"]
        and resources["memory_available_bytes"] >= 1500 * 1024**2
        and resources["free_disk_bytes"] >= 1024**3
        and resources["tools_available"]
    ):
        checks.append(gate(dict(path=str(private)), "required_resources", True, resources))
    tasks = authority["tasks"] or yaml.safe_load((ROOT / methods.ACTIVE).read_bytes())["tasks"]
    if (
        [t["id"].split("-")[0] for t in tasks] != [f"exp{i}" for i in range(8499, 8512)]
        or tasks[-1].get("requires")
        or tasks[-1].get("gated_on")
    ):
        raise ValueError("exact_unconditional_thirteen_tasks")
    protocol = methods.protocol()
    log = freeze(root / "ops/conductor-log.md", raw / "scheduling")
    lines = Path(log["snapshot_path"]).read_text().splitlines() if log["exists"] else []
    activation = next(
        (i for i, line in enumerate(lines) if "Milestone 2026.10.731 activated" in line), len(lines)
    )
    lines = lines[activation:]
    historical_tasks, refs = [], [log]
    for family, inputs in dict(
        protocol["historical_authorities"], V730=protocol["v730_authority_references"]
    ).items():
        sealed = [freeze(Path(r["path"]), raw / "authorities" / family) for r in inputs]
        refs.extend(sealed)
        if any(
            r["sha256"] != original["sha256"] for r, original in zip(sealed, inputs, strict=True)
        ):
            checks.append(
                gate(
                    sealed[1],
                    "original_authority_pin",
                    inputs[1]["sha256"],
                    sealed[1]["sha256"],
                    family,
                )
            )
            continue
        historical_tasks += [
            t
            for t in yaml.safe_load(Path(sealed[1]["snapshot_path"]).read_bytes())["tasks"]
            if int(t["id"][3:7]) in [8432, 8435, 8436, 8437, 8464, 8465, 8466, *range(8485, 8498)]
            and (root / t["deliverable"]).is_file()
        ]
    slots: list[Json] = []
    historical: list[Json] = []
    receipts: list[Json] = []
    roster = [*tasks[:-1], *historical_tasks]
    for index, task in enumerate(roster):
        progress("branch_before_" + task["id"], index, len(roster) - index)
        if (time.monotonic_ns() - start) / 1e9 >= 3600:
            raise TimeoutError("measurement_deadline")
        number = int(task["id"][3:7])
        primary = freeze(root / task["deliverable"], raw / "primaries" / str(number))
        refs.append(primary)
        if not primary["exists"] and number in PREGATE_PINS:
            receipt_path = root / task["deliverable"].replace("_v731_", "_")
            if receipt_path.is_file():
                primary = freeze(receipt_path, raw / "pregates" / str(number))
                refs.append(primary)
        summary = None
        if primary["exists"]:
            summary, receipt = invoke(task, primary, raw / "branches" / task["id"])
            receipts.append(receipt)
            refs.append(freeze(raw / "branches" / task["id"] / "summary.json", raw / "summaries"))
        cascade = next(
            (
                line
                for line in lines
                if "GATE_BLOCK" in line and line.split("|")[2].strip() == task["title"][:50].strip()
            ),
            None,
        )
        row = slot(task, primary, cascade, summary)
        (slots if number >= 8499 else historical).append(row)
        checks.extend(row["checks"])
        progress("branch_after_" + task["id"], index + 1, len(roster) - index - 1)
    refs.extend(
        freeze(ROOT / name, raw / "code")
        for name in [
            *OWNED,
            TEST,
            "python/carnot/reporting/primary_publication.py",
            "scripts/publication_gate.py",
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
            "ops/exclusion_manifest.yaml",
            "openspec/capabilities/research-reporting/spec.md",
            "openspec/capabilities/verification/spec.md",
            "docs/research-notes/v731-capstone.md",
            methods.PROTOCOL,
        ]
    )
    predecessors = [
        freeze(Path(r["path"]), raw / "predecessors") for r in protocol["v730_fact_references"]
    ]
    after = memory()
    work = dict(
        root=str(root),
        raw=str(raw),
        tasks=tasks,
        authority=authority,
        contract_digest=methods.TASKS_PIN,
        slots=slots,
        historical_slots=historical,
        refs=refs,
        predecessors=predecessors,
        branch_receipts=receipts,
        checks=checks,
        preconditions=resources,
        invocation_argv=list(__import__("sys").argv),
        memory=dict(
            before=before,
            after=after,
            peak_bound_mb=1500,
            growth_bound_mb=500,
            passed=after["peak_rss_mb"] <= 1500
            and after["current_rss_mb"] - before["current_rss_mb"] <= 500,
        ),
        phase_spans=[
            dict(
                name="sealed_branch_reduction",
                started_monotonic_ns=start,
                ended_monotonic_ns=time.monotonic_ns(),
            )
        ],
        started_monotonic_ns=start,
        ended_monotonic_ns=time.monotonic_ns(),
        adversarial_findings=[],
    )
    atomic_json(raw / "measurement.json", work)
    return work


def replay(path: Path) -> bool:
    """Rebuild rows and reports against independent seals, so repaired hashes cannot hide drift."""
    from carnot.reporting.v731_capstone_report import build

    try:
        value = json.loads(path.read_bytes())
        work = load(value["primitive_reference"])
        if (
            value["reproducibility_checksum"] != checksum(value)
            or value["task_count"] != 13
            or methods.tasks_digest(work["tasks"]) != methods.TASKS_PIN
            or work["contract_digest"] != methods.TASKS_PIN
        ):
            return False
        if [r["sha256"] for r in work["predecessors"]] != [
            r["sha256"] for r in methods.protocol()["v730_fact_references"]
        ]:
            return False
        for ref in [*work["refs"], *work["predecessors"], *work.get("coverage_shards", [])]:
            if (
                ref.get("exists", True)
                and sha256_file(Path(ref.get("snapshot_path", ref["path"]))) != ref["sha256"]
            ):
                return False
        with TemporaryDirectory(prefix="exp8511-authority-", dir="/var/tmp") as directory:
            cold = Path(directory)
            for role, name in [
                ("design", methods.DESIGN),
                ("active", methods.ACTIVE),
                ("staged", methods.STAGED),
            ]:
                ref = work["authority"]["authority_snapshots"][role]
                if ref["exists"]:
                    source = Path(ref["snapshot_path"])
                    if sha256_file(source) != ref["sha256"]:
                        return False
                    target = cold / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
            checked = methods.authority(cold, cold / "checked")
            if (
                checked["activated"] != work["authority"]["activated"]
                or checked["tasks"] != work["authority"]["tasks"]
            ):
                return False
        for row in [*work["slots"], *work["historical_slots"]]:
            if (
                row["experiment_id"] >= 8499
                and row["task"] != work["tasks"][row["experiment_id"] - 8499]
            ):
                return False
            if row["primary_reference"]["exists"]:
                with TemporaryDirectory(prefix="exp8511-cold-", dir="/var/tmp") as directory:
                    actual, receipt = invoke(row["task"], row["primary_reference"], Path(directory))
                if not receipt["passed"] or any(
                    actual[k] != row[k]
                    for k in actual
                    if k not in {"memory", "reader_initialization"}
                ):
                    return False
            elif row != slot(row["task"], row["primary_reference"], row["cascade_log_line"]):
                return False
        for receipt in [*work["branch_receipts"], *value["validation_receipts"]]:
            for stream in ["stdout", "stderr"]:
                if sha256_file(Path(receipt[stream + "_path"])) != receipt[stream + "_sha256"]:
                    return False
        return bool(
            value
            == build(
                work,
                value["validation_receipts"],
                Path(work["raw"]),
                Path(value["publication_output"]),
            )
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False
