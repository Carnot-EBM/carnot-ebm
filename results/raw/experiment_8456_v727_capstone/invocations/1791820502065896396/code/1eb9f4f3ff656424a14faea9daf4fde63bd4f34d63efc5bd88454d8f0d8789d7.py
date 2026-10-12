"""REQ-REPORT-8456: narrow adapters preserve original assertions and missing science."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import time
from typing import Any, Iterator
from unittest.mock import patch

import yaml

from carnot.reporting import v726_capstone_evidence as old
from carnot.reporting import v726_capstone_report as report
from carnot.reporting import v727_contract_methods as methods
from carnot.reporting import v727_historical_inputs as history
from carnot.reporting.board_operation_boundary_8454 import resources
from carnot.reporting.current_work_receipt import atomic_json

Json = dict[str, Any]
ROOT = old.ROOT
NAME, TASK, MILESTONE = "experiment_8456_v727_capstone", "exp8456-capstone", "2026.10.727"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v727_capstone_8456.py"
OWNED = [
    "python/carnot/reporting/v727_capstone_evidence.py",
    "python/carnot/reporting/v727_capstone_report.py",
    "python/carnot/reporting/v727_capstone.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
PINS = {
    8443: "1caeb7a6168293843ad22462043829eaef58d48c44f9cf3958a5b88a1c2a2f56",
    8444: "0ca099017115a65ae661b064e51d8c489a38b45e8a5605aab74dbb5294f2bb1c",
    8447: "6e616c652bb7543c5b00b204627918a7556c04aba6ee1568d606cd7351035b4c",
    8449: "c56b50d2cf994a78fe4921f550b5d59d7b29d8ae33ce921a6766f345d6bc109a",
    8450: "7058c19171f766746fd609d019c4523919aaa0a1412f342e70b485b3ae466d3d",
    8453: "15ef5947f31881e2ce5e01a675a76a3538fdb38760ebb27360fc4227492b3155",
    8454: "0db32319eb4a5f348ca7a1983a71a96c3a066ce72692e5e0e2935c123af05cc3",
    8455: "9a015a784d28cc35553781655c62ab7f328c4ed01aa0bc8c39a66a87a679ce48",
}
PREDECESSOR_PINS = {
    8415: "sha256:125505046358d3debe531593528da1cfd75d4cbc7d100caef850292b3b6cd9d0",
    8359: "sha256:e597e603096c0fc652d709c4aad783f64ef2f2d9cb0c1abe4cf97bb5851d74fb",
    8442: "sha256:48090c853dbbe4cb0e6fb067ac6d9c2255b04f4f4eb6757340d862067c3e05e7",
}
READERS = {
    8443: "reporting.v727_contract_methods",
    8449: "reporting.persistence_localization_8449",
    8453: "reporting.arc_seal_and_support_8453",
    8454: "reporting.board_operation_boundary_8454",
    8455: "reporting.gatemate_obligation_8455",
}
SELECT = old.SELECT | {
    "fifo_minus_uniform_brier",
    "kernel_s",
    "persistence_stage_s",
    "conversion_s",
    "unattributed_s",
    "arithmetic_10x_feasible_under_fixed_service",
    "every_primary_repeat_rules_out_arithmetic_10x",
    "complete_wall_s",
    "stage_rows",
    "paired_repeat_rows",
    "warmups",
    "custody_recovery",
    "summary_comparison",
    "board_operation_rows",
    "board_terminal_states",
    "kv260_supported_operations",
    "polarfire_graduation",
    "physical_change_receipt",
    "required_idcode",
}
checksum, reference, freeze, load, slot, memory = (
    old.checksum,
    old.reference,
    old.freeze,
    old.load,
    old.slot,
    old.memory,
)
workers = old.workers
_OUTCOME, _WORKER, _REPLAY = old.outcome, old.worker_process, old.replay


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush counts because bounded historical replay can keep its logs private."""
    print(f"[exp8456] phase={phase} completed={completed} pending={pending}", flush=True)


def gate(ref: Json, field: str, expected: Any, observed: Any, upstream: str = TASK) -> Json:
    """Keep the original explicit missing-operand gate shape for both consumers."""
    return dict(old.gate(ref, field, expected, observed, upstream))


@contextmanager
def binding() -> Iterator[None]:
    """Only invocation constants change; historical reducers keep their assertions."""
    from carnot.reporting.v727_capstone_report import build

    with (
        patch.multiple(
            old,
            NAME=NAME,
            TASK=TASK,
            MILESTONE=MILESTONE,
            CLI=CLI,
            TEST=TEST,
            OWNED=OWNED,
            methods=methods,
            PINS=PINS,
            READERS=READERS,
            SELECT=SELECT,
            progress=progress,
        ),
        patch.object(report, "build", build),
    ):
        yield


def outcome(task: Json, primary: Json, raw: Path, closure: Any = None) -> Json:
    """Historical science reads its sealed original authority, never current vNEXT."""
    number = int(task["id"][3:7])
    if number < 8443:
        sealed = history.custody(ROOT, raw / "original_authority")
        with workers.frozen_inputs(sealed, raw / "scratch"), patch.object(old, "SELECT", SELECT):
            return dict(_OUTCOME(task, primary, raw, closure))
    with binding():
        return dict(_OUTCOME(task, primary, raw, closure))


def worker_process(request: Path, output: Path) -> int:
    """Shipped fresh process memory bounds apply to each eligible current branch."""
    if int(json.loads(request.read_bytes())["task"]["id"][3:7]) < 8443:
        with patch.multiple(old, outcome=outcome, progress=progress):
            return int(_WORKER(request, output))
    with binding(), patch.object(old, "outcome", outcome):
        return int(_WORKER(request, output))


def invoke(task: Json, primary: Json, raw: Path) -> tuple[Json, Json]:
    """Each branch runs in its own bounded child with durable command and streams."""
    with patch.multiple(workers, CLI=CLI, progress=progress):
        result, receipt = workers.invoke(task, primary, raw)
    return dict(result), dict(receipt)


def measure(root: Path, raw: Path, private: Path) -> Json:
    """Current operands alone gate CPU accounting; unavailable devices remain optional."""
    start, before = time.monotonic_ns(), memory()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    progress("preconditions", 0, 14)
    authority = methods.authority(root, raw / "authority")
    checks = list(authority["gate_check_summary"])
    resource = dict(
        resources(private), optional_cuda_required=False, optional_boards_required=False
    )
    if not (
        resource["disk_backed"]
        and resource["memory_available_bytes"] >= 1500 * 1024**2
        and resource["free_disk_bytes"] >= 1024**3
        and resource["tools_available"]
    ):
        checks.append(gate(dict(path=str(private)), "required_resources", True, resource))
    tasks = authority["tasks"] or yaml.safe_load((ROOT / methods.ACTIVE).read_bytes())["tasks"]
    if (
        [t["id"].split("-")[0] for t in tasks] != [f"exp{i}" for i in range(8443, 8457)]
        or tasks[-1].get("gated_on")
        or tasks[-1].get("requires")
    ):
        raise ValueError("exact_unconditional_fourteen_tasks")
    log = freeze(root / "ops/conductor-log.md", raw / "scheduling")
    lines = Path(log["snapshot_path"]).read_text().splitlines() if log["exists"] else []
    refs: list[Json] = [log]
    slots: list[Json] = []
    receipts: list[Json] = []
    historical: list[Json] = []
    historical_tasks = yaml.safe_load((ROOT / history.ACTIVE).read_bytes())["tasks"]
    roster = [
        *tasks[:-1],
        *[
            t
            for t in historical_tasks
            if int(t["id"][3:7]) in [8432, 8435, 8436, 8437] and (root / t["deliverable"]).is_file()
        ],
    ]
    for index, task in enumerate(roster):
        progress("branch_before_" + task["id"], index, len(roster) - index)
        if (time.monotonic_ns() - start) / 1e9 >= 3600:
            raise TimeoutError("measurement_deadline")
        number = int(task["id"][3:7])
        source_root = root
        candidates = list((source_root / "results").glob(f"experiment_{number}_*.json"))
        primary = freeze(
            candidates[0] if len(candidates) == 1 else source_root / task["deliverable"],
            raw / "primaries" / str(number),
        )
        refs.append(primary)
        summary = None
        if primary["exists"]:
            summary, receipt = invoke(task, primary, raw / "branches" / task["id"])
            receipts.append(receipt)
            refs.append(freeze(raw / "branches" / task["id"] / "summary.json", raw / "summaries"))
            if summary.get("primitive_reference"):
                refs.append(
                    freeze(Path(summary["primitive_reference"]["path"]), raw / "primitives")
                )
        cascade = next(
            (line for line in lines if "GATE_BLOCK" in line and task["title"][:48] in line), None
        )
        row = slot(task, primary, cascade, summary)
        (slots if number >= 8443 else historical).append(row)
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
            methods.PROTOCOL,
        ]
    )
    predecessors = [
        freeze(next((ROOT / "results").glob(f"experiment_{n}_*.json")), raw / "predecessors")
        for n in [8415, 8359, 8442]
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
        preconditions=resource,
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
    """Cold children rebuild historical science as well as current slot primitives."""
    try:
        value = json.loads(path.read_bytes())
        work = load(value["primitive_reference"])
        if any(
            r["sha256"] != PREDECESSOR_PINS[int(Path(r["path"]).name.split("_")[1])]
            for r in work["predecessors"]
        ):
            return False
        for row in work["historical_slots"]:
            from tempfile import TemporaryDirectory

            with TemporaryDirectory(prefix="exp8456-history-cold-", dir="/var/tmp") as directory:
                actual, receipt = invoke(row["task"], row["primary_reference"], Path(directory))
            if not receipt["passed"] or any(
                actual[k] != row[k] for k in actual if k not in {"memory", "reader_initialization"}
            ):
                return False
        with binding():
            return bool(_REPLAY(path))
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False
