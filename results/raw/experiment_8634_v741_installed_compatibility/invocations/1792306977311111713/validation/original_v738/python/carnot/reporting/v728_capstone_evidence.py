"""REQ-REPORT-8470: sealed primitive replay keeps missing effects separate from zeros."""

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
from carnot.reporting import v727_capstone_evidence as previous
from carnot.reporting import v728_contract_methods as methods
from carnot.reporting import v726_contract_methods as historical_methods
from carnot.reporting.current_work_receipt import atomic_json

Json = dict[str, Any]
ROOT = base.ROOT
NAME, TASK, MILESTONE = "experiment_8470_v728_capstone", "exp8470-capstone", "2026.10.728"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v728_capstone_8470.py"
OWNED = [
    f"python/carnot/reporting/v728_capstone{suffix}.py" for suffix in ["_evidence", "_report", ""]
] + [CLI]
MODEL_SPECS: list[Json] = []
PINS = {
    8457: "7a4d4935da70d990d1821bc23f7b539a9bd854a649fe4c439642ea6680fb5459",
    8458: "96602b67ce48980cfddaba1f94aa5a3c40d8aa554874885ce451859e114dbbc9",
    8459: "2fdafadba557f62216a185357b74179a5c3bd044d09ca074b50069e46de97175",
    8461: "e4adbdbbb533054cb6a5496fcd0761bed7dac0d0fd4f99e4a60741c40f48cf16",
    8462: "b063f44fea0b591439dd5adaf317e72f3469b96cac568096aca981839868990c",
    8464: "759bd38012061675801027e8d053b6df2ebe06970453bceede8c822648629f77",
    8465: "6becdfa9334f6f6ab5b1695eb025ce0cb09ff09f838f0e6c415edb97c580a086",
    8466: "98705463e6d4a85cdddadd122821dae0365700e97459437813b56507cfab718f",
    8467: "f3e13267d5c2aa1027e18ce83ff08c28f147c6d4c2738631cd7f6dee2b3998fc",
    8468: "4a23ce380ecb60a4f584f0cc87904999605049034a24c5f20e9e7b3bd4d2e195",
    8469: "b0de8b3269e7e6e29becea03e27230eb09847035952314aa905bec0adf5fa53a",
}
READERS = {
    8457: "reporting.v728_contract_methods",
    8458: "reporting.v728_source_control_panel",
    8461: "reporting.objective_checkpoint_8461",
    8464: "reporting.persistence_localization_8464",
    8465: "reporting.journal_equivalence_8465",
    8466: "reporting.journal_service_cost_8466",
    8468: "reporting.arc_generalization_panel_8468",
    8469: "reporting.board_operation_boundary_8469",
}
SELECT = previous.SELECT | {
    "pooled_speedup",
    "speedup_ci95",
    "p95_ratio",
    "cell_results",
    "durability_scope",
    "accounting_equality",
    "equivalence_rows",
    "crash_rows",
    "gradient_checks",
    "derivative_rows",
    "restart_checks",
    "process_exit_rows",
    "transition_controls",
    "gatemate_obligation_rows",
    "last_authenticated_idcode",
    "optional_cost_evidence",
    "panel_execution_ready_score",
    "per_episode_rows",
    "paired_progress_rows",
    "refinement_status",
    "retention_rows",
    "NFR_01_status",
    "journal_speed_benefit_score",
    "tenfold_service_score",
}
checksum, reference, freeze, load, slot, memory = (
    base.checksum,
    base.reference,
    base.freeze,
    base.load,
    base.slot,
    base.memory,
)
workers = base.workers
HISTORICAL_PINS, HISTORICAL_READERS = dict(base.PINS), dict(base.READERS)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual status so a waiting conductor can distinguish progress from abandonment."""
    print(f"[exp8470] phase={phase} completed={completed} pending={pending}", flush=True)


def gate(ref: Json, field: str, expected: Any, observed: Any, upstream: str = TASK) -> Json:
    """Keep missing operands explicit, including their original path and byte hash."""
    return dict(base.gate(ref, field, expected, observed, upstream))


def disk_scratch(*args: Any, **kwargs: Any) -> Any:
    """Keep original board reconstruction writes outside sealed results and the test guard."""
    return TemporaryDirectory(*args, **dict(kwargs, dir="/var/tmp"))


@contextmanager
def binding() -> Iterator[None]:
    """Change invocation constants only; shipped replay and memory assertions stay intact."""
    from carnot.reporting.v728_capstone_report import build

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
            PINS=PINS,
            READERS=READERS,
            SELECT=SELECT,
            progress=progress,
        ),
        patch.object(report, "build", build),
    ):
        yield


def outcome(task: Json, primary: Json, raw: Path, closure: Any = None) -> Json:
    """Replay original primitives, retaining disqualification without inventing a measured null."""
    number = int(task["id"][3:7])
    if number < 8457:
        with patch.multiple(
            base,
            methods=historical_methods,
            PINS=HISTORICAL_PINS,
            READERS=HISTORICAL_READERS,
            SELECT=previous.SELECT,
        ):
            return dict(previous.outcome(task, primary, raw, closure))
    value = load(primary)
    if primary["sha256"] != "sha256:" + PINS[number]:
        raise ValueError("primary_pin")
    if value.get("blocked_at_layer") == "conductor_pre_gate":
        with binding():
            return dict(previous._OUTCOME(task, primary, raw, closure))
    base.validate_primary(value, Path(primary["path"]))
    if value["task_id"] != task["id"]:
        raise ValueError("task_identity")
    terminal = load(reference(Path(value["terminal_validation_sidecar_path"])))
    side = load(reference(Path(terminal["publication"]["sidecar_path"])))
    sealed = (
        terminal["publication"]["primary_sha256"] == primary["sha256"]
        and side["primary_sha256"] == primary["sha256"]
        and side["report"]["passed"] is True
    )
    reader = importlib.import_module("carnot." + READERS[number])
    with ExitStack() as stack:
        if number == 8469:
            stack.enter_context(
                patch.object(reader.old.historical, "TemporaryDirectory", disk_scratch)
            )
        replayed = bool(reader.replay(value if number == 8468 else Path(primary["snapshot_path"])))
    qualified = (
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
        historical_model_provenance=value.get("historical_model_provenance", []),
        upstream_adversarial_findings=value.get("adversarial_findings", []),
        primitive_reference=value.get("primitive_reference", value.get("work_reference")),
        branch_replay=dict(
            status="replayed" if qualified else "disqualified_upstream_evidence", passed=replayed
        ),
    )


def worker_process(request: Path, output: Path) -> int:
    """The shipped fresh-process worker enforces both peak and growth memory bounds."""
    args = json.loads(request.read_bytes())
    number = int(args["task"]["id"][3:7])
    startup = memory()
    readers = {**HISTORICAL_READERS, **READERS}
    if number in readers:
        progress("reader_dependency_import_before")
        importlib.import_module("carnot." + readers[number])
        progress("reader_dependency_import_after")
    initialized = memory()
    with patch.multiple(workers, outcome=outcome, memory=memory, progress=progress):
        code = int(workers.worker_process(request, output))
    value = json.loads(output.read_bytes())
    value["reader_initialization"] = dict(
        before=startup,
        after=initialized,
        scope="Python dependency imports; no model weights loaded",
    )
    atomic_json(output, value)
    return code


def invoke(task: Json, primary: Json, raw: Path) -> tuple[Json, Json]:
    """Retain real child commands, exit status and hashes rather than summarizing execution."""
    with patch.multiple(workers, CLI=CLI, progress=progress):
        result, receipt = workers.invoke(task, primary, raw)
    return dict(result), dict(receipt)


def measure(root: Path, raw: Path, private: Path) -> Json:
    """Only required CPU operands gate reconciliation; devices never gate these readers."""
    start, before = time.monotonic_ns(), memory()
    progress("preconditions", 0, 14)
    authority = methods.authority(root, raw / "authority")
    checks = list(authority["gate_check_summary"])
    resource = dict(
        previous.resources(private),
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
        [t["id"].split("-")[0] for t in tasks] != [f"exp{i}" for i in range(8457, 8471)]
        or tasks[-1].get("gated_on")
        or tasks[-1].get("requires")
    ):
        raise ValueError("exact_unconditional_fourteen_tasks")
    log = freeze(root / "ops/conductor-log.md", raw / "scheduling")
    lines = Path(log["snapshot_path"]).read_text().splitlines() if log["exists"] else []
    activation = next(
        (i for i, line in enumerate(lines) if "Milestone 2026.10.728 activated" in line), len(lines)
    )
    lines = lines[activation:]
    refs: list[Json] = [log]
    slots: list[Json] = []
    receipts: list[Json] = []
    historical: list[Json] = []
    old_tasks = yaml.safe_load((ROOT / previous.history.ACTIVE).read_bytes())["tasks"]
    roster = [
        *tasks[:-1],
        *[
            t
            for t in old_tasks
            if int(t["id"][3:7]) in [8432, 8435, 8436, 8437] and (root / t["deliverable"]).is_file()
        ],
    ]
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
            if summary.get("primitive_reference"):
                refs.append(
                    freeze(Path(summary["primitive_reference"]["path"]), raw / "primitives")
                )
        cascade = next(
            (line for line in lines if "GATE_BLOCK" in line and task["title"][:48] in line), None
        )
        row = slot(task, primary, cascade, summary)
        (slots if number >= 8457 else historical).append(row)
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
        for n in [8415, 8359, 8442, 8456]
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
    """Fresh workers rebuild primitives; local rehashing cannot replace independent producer pins."""
    try:
        value = json.loads(path.read_bytes())
        work = load(value["primitive_reference"])
        if methods.tasks_digest(work["tasks"]) != methods.TASKS_PIN:
            return False
        pins = {
            **previous.PREDECESSOR_PINS,
            8456: "sha256:717bab8d23534353315f45c55c990255e9dd79a1db5ffc4ea3e93d9974aa073c",
        }
        for ref in work["predecessors"]:
            if ref["sha256"] != pins[int(Path(ref["path"]).name.split("_")[1])]:
                return False
        for row in work["historical_slots"]:
            with TemporaryDirectory(prefix="exp8470-history-cold-", dir="/var/tmp") as directory:
                actual, receipt = invoke(row["task"], row["primary_reference"], Path(directory))
            if not receipt["passed"] or any(
                actual[k] != row[k] for k in actual if k not in {"memory", "reader_initialization"}
            ):
                return False
        with binding():
            return bool(previous._REPLAY(path))
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False
