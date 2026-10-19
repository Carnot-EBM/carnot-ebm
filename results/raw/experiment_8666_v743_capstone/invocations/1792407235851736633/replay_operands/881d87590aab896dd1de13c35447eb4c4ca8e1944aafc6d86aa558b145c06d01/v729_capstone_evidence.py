"""REQ-REPORT-8484: missing upstream science stays missing under sealed producer authority."""

from __future__ import annotations

from contextlib import contextmanager, ExitStack
import importlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Any, Iterator
from unittest.mock import patch

import yaml

from carnot.reporting import v726_capstone_evidence as base
from carnot.reporting import v726_capstone_report as report
from carnot.reporting import v728_capstone_evidence as history
from carnot.reporting import v729_contract_methods as methods
from carnot.reporting.current_work_receipt import atomic_json

Json = dict[str, Any]
ROOT = base.ROOT
NAME, TASK, MILESTONE = "experiment_8484_v729_capstone", "exp8484-capstone", "2026.10.729"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v729_capstone_8484.py"
OWNED = [f"python/carnot/reporting/v729_capstone{s}.py" for s in ["_evidence", "_report", ""]] + [
    CLI
]
MODEL_SPECS: list[Json] = []
PINS = {
    8471: "f892ea7702ac99a7f087527a18a21070fe8ec9fb85fae1097446eaeb611d2030",
    8482: "d3a1ed3b0378f0075f29536e8f53eb87d7df0aeb2d923431d343caff0949a9e2",
    8483: "db07f56c8d99337e6a21d3efd5e83a39df59eeca8fcaf8d9ccb8dd6ade2d43ea",
}
PREGATE_PINS = {
    8472: "25e4b3b925158ad4962974dab478866933bc645b5766da344f2c578dc36afa52",
    8475: "615293924a90e6e4416b5828adda13ac7dc7936ea9d2e697f433906986dc5d71",
    8478: "0fac3e56b1e2f18215b04ff8c3acc7da4cfbcbf07101e722124f555a91cb051b",
}
READERS = {
    8471: "v729_contract_methods",
    8482: "arc_panel_qualification_8482",
    8483: "board_operation_boundary_8483",
}
PREDECESSOR_PINS = {
    8415: history.previous.PREDECESSOR_PINS[8415],
    8359: history.previous.PREDECESSOR_PINS[8359],
    8442: "sha256:48090c853dbbe4cb0e6fb067ac6d9c2255b04f4f4eb6757340d862067c3e05e7",
    8470: "sha256:2e60de60e9f4b0085ab30026d51d0f1a73084c6a1db292ea908e395f6c9d64e7",
}
SELECT = history.SELECT | {
    "board_operation_rows",
    "append_cost_boundary",
    "board_terminal_states",
    "deployment_ready",
    "historical_authority_limits",
    "historical_family_receipts",
}
checksum, reference, freeze, load, slot, memory, gate = (
    base.checksum,
    base.reference,
    base.freeze,
    base.load,
    base.slot,
    base.memory,
    base.gate,
)
workers = base.workers


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed counts make long child work visible to the operator and conductor."""
    print(f"[exp8484] phase={phase} completed={completed} pending={pending}", flush=True)


@contextmanager
def binding() -> Iterator[None]:
    """Reuse immutable validation mechanics while restoring every original module afterwards."""
    from carnot.reporting.v729_capstone_report import build

    with (
        patch.multiple(
            base,
            NAME=NAME,
            TASK=TASK,
            MILESTONE=MILESTONE,
            CLI=CLI,
            TEST=TEST,
            OWNED=OWNED,
            methods=methods,
            progress=progress,
        ),
        patch.object(report, "build", build),
    ):
        yield


def outcome(task: Json, primary: Json, raw: Path, closure: Any = None) -> Json:
    """A current receipt must replay current code; historical branches retain original authority."""
    number = int(task["id"][3:7])
    if number < 8471:
        return dict(history.outcome(task, primary, raw, closure))
    value = load(primary)
    if primary["sha256"] != "sha256:" + (PINS | PREGATE_PINS)[number]:
        raise ValueError("primary_pin")
    if value.get("blocked_at_layer") == "conductor_pre_gate":
        with patch.dict(base.PINS, PINS | PREGATE_PINS):
            return dict(base.outcome(task, primary, raw, closure))
    base.validate_primary(value, Path(primary["path"]))
    if value["task_id"] != task["id"]:
        raise ValueError("task_identity")
    terminal = load(reference(Path(value["terminal_validation_sidecar_path"])))
    side = load(reference(Path(terminal["publication"]["sidecar_path"])))
    sealed = (
        terminal["publication"]["primary_sha256"] == primary["sha256"] == side["primary_sha256"]
        and side["report"]["passed"] is True
    )
    reader = importlib.import_module("carnot.reporting." + READERS[number])
    with ExitStack() as stack:
        if number == 8483:
            stack.enter_context(
                patch.object(reader.old.old.historical, "TemporaryDirectory", history.disk_scratch)
            )
        replayed = bool(reader.replay(value if number == 8482 else Path(primary["snapshot_path"])))
    qualified = (
        sealed
        and replayed
        and value["required_checks_passed"]
        and not value["flagged_adversarial"]
        and value["verdict_class"] != "disqualified"
    )
    return dict(
        producer_executed=True,
        eligible=bool(qualified),
        selected={k: v for k, v in value.items() if k in SELECT or k.endswith("_ready_score")},
        checks=[]
        if qualified
        else [
            gate(
                primary,
                "producer_terminal_replay",
                True,
                dict(
                    sealed=sealed,
                    replayed=replayed,
                    required_checks_passed=value["required_checks_passed"],
                ),
                task["id"],
            )
        ],
        disposition="qualified_producer" if qualified else "disqualified_producer",
        honest_verdict=value["honest_verdict"],
        verdict_class=value["verdict_class"],
        source_gates=value.get("gate_check_summary", []),
        historical_model_provenance=methods.history.model_provenance(value, primary),
        upstream_adversarial_findings=value.get("adversarial_findings", []),
        primitive_reference=value.get("primitive_reference", value.get("work_reference")),
        branch_replay=dict(
            status="replayed" if qualified else "disqualified_upstream_evidence", passed=replayed
        ),
    )


def worker_process(request: Path, output: Path) -> int:
    """Fresh exec bounds include reader imports and every primitive replay allocation."""
    args = json.loads(request.read_bytes())
    number = int(args["task"]["id"][3:7])
    if number < 8471:
        return int(history.worker_process(request, output))
    if number in READERS:
        progress("reader_dependency_import_before")
        importlib.import_module("carnot.reporting." + READERS[number])
        progress("reader_dependency_import_after")
    with patch.multiple(workers, outcome=outcome, memory=memory, progress=progress):
        return int(workers.worker_process(request, output))


def invoke(task: Json, primary: Json, raw: Path) -> tuple[Json, Json]:
    """Keep genuine child commands, exits, durations and log hashes for scientific reductions."""
    with patch.multiple(workers, CLI=CLI, progress=progress):
        result, receipt = workers.invoke(task, primary, raw)
    return dict(result), dict(receipt)


def measure(root: Path, raw: Path, private: Path) -> Json:
    """Only supplied CPU operands enter this invocation; unavailable devices never gate it."""
    start, before = time.monotonic_ns(), memory()
    progress("preconditions", 0, 14)
    authority = methods.authority(root, raw / "authority")
    checks = list(authority["gate_check_summary"])
    resource = dict(
        history.previous.resources(private),
        task_cap_s=4800,
        measurement_cap_s=3600,
        validation_reserve_s=900,
        optional_cuda_required=False,
        optional_boards_required=False,
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
        [t["id"].split("-")[0] for t in tasks] != [f"exp{i}" for i in range(8471, 8485)]
        or tasks[-1].get("gated_on")
        or tasks[-1].get("requires")
    ):
        raise ValueError("exact_unconditional_fourteen_tasks")
    log = freeze(root / "ops/conductor-log.md", raw / "scheduling")
    lines = Path(log["snapshot_path"]).read_text().splitlines() if log["exists"] else []
    activation = next(
        (i for i, line in enumerate(lines) if "Milestone 2026.10.729 activated" in line), len(lines)
    )
    lines = lines[activation:]
    historical_tasks = []
    refs: list[Json] = [log]
    for family, inputs in methods.protocol()["historical_authorities"].items():
        sealed = [freeze(Path(r["path"]), raw / "historical_authorities" / family) for r in inputs]
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
            if int(t["id"][3:7]) in [8432, 8435, 8436, 8437, 8464, 8465, 8466]
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
        candidates = list((root / "results").glob(f"experiment_{number}_*.json"))
        primary = freeze(
            candidates[0] if len(candidates) == 1 else root / task["deliverable"],
            raw / "primaries" / str(number),
        )
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
                if "GATE_BLOCK" in line and (task["id"] in line or task["title"][:48] in line)
            ),
            None,
        )
        row = slot(task, primary, cascade, summary)
        (slots if number >= 8471 else historical).append(row)
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
        for n in PREDECESSOR_PINS
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
    """Rebuild original primitives and authority, so repaired aggregate hashes cannot qualify."""
    try:
        value = json.loads(path.read_bytes())
        work = load(value["primitive_reference"])
        if methods.tasks_digest(work["tasks"]) != methods.TASKS_PIN:
            return False
        if any(
            r["sha256"] != PREDECESSOR_PINS[int(Path(r["path"]).name.split("_")[1])]
            for r in work["predecessors"]
        ):
            return False
        for row in work["historical_slots"]:
            with TemporaryDirectory(prefix="exp8484-history-cold-", dir="/var/tmp") as directory:
                actual, receipt = invoke(row["task"], row["primary_reference"], Path(directory))
            if not receipt["passed"] or any(
                actual[k] != row[k] for k in actual if k not in {"memory", "reader_initialization"}
            ):
                return False
        with binding():
            return bool(base.replay(path))
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False
