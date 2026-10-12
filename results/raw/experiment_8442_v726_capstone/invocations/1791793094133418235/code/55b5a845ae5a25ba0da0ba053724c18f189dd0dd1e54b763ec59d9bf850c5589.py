"""REQ-REPORT-8442: immutable producer evidence keeps missing science distinct from zero."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import v726_contract_methods as methods
from carnot.reporting import v721_capstone_evidence as workers
from carnot.reporting.v723_capstone_evidence import memory
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import validate_primary

Json = dict[str, Any]
ROOT = methods.ROOT
NAME, TASK, MILESTONE = "experiment_8442_v726_capstone", "exp8442-capstone", "2026.10.726"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v726_capstone_8442.py"
OWNED = [
    "python/carnot/reporting/v726_capstone_evidence.py",
    "python/carnot/reporting/v726_capstone_report.py",
    "python/carnot/reporting/v726_capstone.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
PINS = dict(
    zip(
        range(8429, 8442),
        [
            "dfad99775207ce8073b35c7d5efbc0cbefa70ad1ddadadafa584157add3a3095",
            "c5783590d946a9500c335a7d39e0fb154dd5943c0efc2c1355dde885c3712989",
            "38db38433d5e2803a7d4b008c88fba5a1a91138abceb42e52f3dadf5b2f10b7f",
            "100efbb17df04e60271df4f8c7e574e1ad03948d1b0d4888625fd6b35993e4b2",
            "5cbdfe3e9b9498c1430fa125e53b635c72b0979bed8522d4fa3183b7991a704c",
            "e3bced9f078ce48912394c9047c2d94d44c7cefdc5df1519530b25924caba51c",
            "e2e58cbf680becad2f1572629337a58d747911492c24fa43697a8d867f4b0a51",
            "4cf40daca5cf3dd1bc27d5e3159e52d8887d2fa6f65e7b2ae144e039df14ac7b",
            "01c273a37be2749e94e07589d9224ea2c8bc9a7207420e4a75cb0719d4aa9a6f",
            "8806ae5c1cebf915fd7178f86949439c70866bd0a8a98a6644814d14d670f223",
            "816c38f530d7e5ab7cbb7f7a706858f2808733e2b531913df49049f57d5ac986",
            "f1fe951405765fc39318a440f90c3baa2cb34d713f75945de638aa3b74a887bb",
            "2c9f2e6b573db75a5db9466668b88f30f89a8f6be1a4765dd2a802be076759d1",
        ],
        strict=True,
    )
)
READERS = dict(
    zip(
        [8429, 8430, 8431, 8432, 8433, 8434, 8435, 8436, 8437, 8439, 8440, 8441],
        [
            "reporting.v726_contract_methods",
            "reporting.v726_feature_panel",
            "verify.v726_criterion_energy_fit",
            "reporting.v726_criterion_decision_audit",
            "reporting.block_checkpoint_qualification_8433",
            "reporting.capacity_learning_report_8434",
            "reporting.capacity_benefit_report_8435",
            "reporting.transaction_attribution_8436",
            "reporting.acceleration_ceiling_8437",
            "reporting.arc_supervisor_support_8439",
            "reporting.board_reader_qualification_8440",
            "reporting.gatemate_obligation_8441",
        ],
        strict=True,
    )
)
SELECT = {
    "scientific_outcome",
    "scientific_verdict_class",
    "brier_gain_vs_logistic",
    "brier_gain_ci95",
    "decision_loss_delta",
    "A1_metrics",
    "A2_sensitivity",
    "exploratory_benefit_score",
    "gain_vs_frozen",
    "conditional_ci95",
    "exploratory_learning_benefit_score",
    "per_window_retention",
    "retention_simultaneous_upper95",
    "support_adequate",
    "later_question_count",
    "coefficient_change_count",
    "fraction_accelerable",
    "target_comparison",
    "observed_primary_range",
    "lower_bound_nonkernel_removal",
    "arm_refinement_authorized",
    "overlap_by_arm",
    "per_game_results",
    "censoring_counts",
    "kv260_operation_mapping",
    "board_operation_mapping",
    "missing_source_hashes",
    "exact_missing_hashes",
    "physical_obligations",
    "native_proof_mismatch",
    "native_guard_diagnostic",
    "historical_missing_after",
    "support_status",
    "operation_mapping",
    "board_obligations",
    "supported_operations",
    "unsupported_operations",
    "coverage_missing_before_after",
    "eligible_operation_count",
    "obligation_rows",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush counts so bounded child work cannot look like an abandoned task."""
    print(f"[exp8442] phase={phase} completed={completed} pending={pending}", flush=True)


def checksum(value: Json) -> str:
    """A checksum detects changes but independent producer pins establish authenticity."""
    return str(canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"}))


def reference(path: Path) -> Json:
    """Exact bytes bind reductions and their child logs to this invocation."""
    return dict(path=str(path), sha256=sha256_file(path))


def freeze(path: Path, raw: Path) -> Json:
    """Copy individual bounded files rather than a potentially unbounded history tree."""
    if path.is_file() and path.stat().st_size > 128 * 1024**2:
        raise ValueError("operand_size_budget")
    return dict(workers.freeze(path, raw))


def load(ref: Json) -> Json:
    """A locally repaired hash cannot replace the independent producer pin."""
    path = Path(str(ref.get("snapshot_path", ref["path"])))
    if sha256_file(path) != ref["sha256"]:
        raise ValueError("operand_hash")
    return dict(json.loads(path.read_bytes()))


def gate(ref: Json, field: str, expected: Any, observed: Any, upstream: str = TASK) -> Json:
    """Explicit operands prevent a missing measurement from being reported as zero."""
    return dict(
        check=field,
        upstream=upstream,
        path=ref["path"],
        hash=ref.get("sha256"),
        field=field,
        operator="==",
        expected=expected,
        observed=observed,
        passed=False,
    )


def outcome(task: Json, primary: Json, raw: Path, closure: Any = None) -> Json:
    """Shipped reducers reconstruct primitives before any imported science is qualified."""
    value = load(primary)
    number = int(task["id"][3:7])
    pre = value.get("blocked_at_layer") == "conductor_pre_gate"
    if primary["sha256"] != "sha256:" + PINS[number]:
        raise ValueError("primary_pin")
    if pre:
        return dict(
            producer_executed=False,
            eligible=False,
            selected={},
            checks=[],
            disposition="pre_gate_receipt",
            honest_verdict=value["honest_verdict"],
            verdict_class="blocked",
            branch_replay=dict(status="pre_gate_receipt", passed=True),
            source_gates=[
                dict(
                    check=g["artifact_field"],
                    upstream=g["upstream"],
                    path=g["artifact_path"],
                    hash=g["artifact_sha256"],
                    field=g["artifact_field"],
                    operator=g["op"],
                    expected=g["expected"],
                    observed=g["actual"],
                    passed=False,
                )
                for g in value.get("gates_evaluated", [])
                if not g["passed"]
            ],
        )
    validate_primary(value, Path(primary["path"]))
    if value["task_id"] != task["id"]:
        raise ValueError("task_identity")
    terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
    sealed = terminal["publication"]["primary_sha256"] == primary["sha256"]
    replayed = bool(
        importlib.import_module("carnot." + READERS[number]).replay(
            Path(str(primary.get("snapshot_path", primary["path"])))
        )
    )
    qualified = (
        sealed and replayed and value["required_checks_passed"] and not value["flagged_adversarial"]
    )
    checks = (
        []
        if qualified
        else [
            gate(
                primary,
                "producer_terminal_replay",
                True,
                dict(sealed=sealed, replayed=replayed),
                task["id"],
            )
        ]
    )
    selected = {k: v for k, v in value.items() if k in SELECT or k.endswith("_ready_score")}
    return dict(
        producer_executed=True,
        eligible=qualified,
        selected=selected,
        checks=checks,
        disposition="qualified_producer"
        if qualified and value["verdict_class"] != "disqualified"
        else "disqualified_producer",
        honest_verdict=value["honest_verdict"],
        verdict_class=value["verdict_class"],
        source_gates=value.get("gate_check_summary", []),
        historical_model_provenance=value.get("historical_model_provenance", []),
        primitive_reference=value.get("primitive_reference", value.get("work_reference")),
        branch_replay=dict(
            status="replayed" if replayed else "disqualified_cold_replay", passed=replayed
        ),
    )


def slot(task: Json, primary: Json, cascade: str | None, summary: Json | None = None) -> Json:
    """Scheduling status does not substitute for an executed scientific experiment."""
    state = summary or dict(
        producer_executed=False,
        eligible=False,
        selected={},
        checks=[gate(primary, "producer.exists", True, None, task["id"])],
        disposition="cascade_skip" if cascade else "absent_artifact",
        honest_verdict=None,
        verdict_class="blocked",
        branch_replay=dict(status="absent_input", passed=False),
    )
    accounted = bool(primary["exists"] or cascade)
    return dict(
        state,
        experiment_id=int(task["id"][3:7]),
        task_id=task["id"],
        task=task,
        unit_id=task["id"],
        family=task["id"],
        arm="scheduling_slot",
        seed=None,
        primary_reference=primary,
        cascade_log_line=cascade,
        status="completed" if accounted else "censored",
        censored=not accounted,
        excluded=False,
        absolute_metric=int(accounted),
        raw_numerator=int(accounted),
        raw_denominator=1,
        effective_independent_groups=0,
        missing_reason=None if state["eligible"] else state["branch_replay"]["status"],
    )


def worker_process(request: Path, output: Path) -> int:
    """Fresh exec memory sampling excludes allocations from earlier parent phases."""
    args = json.loads(request.read_bytes())
    number = int(args["task"]["id"][3:7])
    startup = memory()
    if number in READERS:
        progress("reader_dependency_import_before")
        importlib.import_module("carnot." + READERS[number])
        progress("reader_dependency_import_after")
    initialized = memory()
    with patch.multiple(workers, outcome=outcome, memory=memory, progress=progress):
        code = int(workers.worker_process(request, output))
    value = json.loads(output.read_bytes())
    value["reader_initialization"] = dict(
        before=startup,
        after=initialized,
        scope="Python dependency imports before primitive replay; no model weights loaded",
    )
    atomic_json(output, value)
    return code


def measure(root: Path, raw: Path, private: Path) -> Json:
    """Only current branch operands are required; optional devices never gate CPU work."""
    start, before = time.monotonic_ns(), memory()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    progress("preconditions", 0, 14)
    authority = methods.authority(root, raw / "authority")
    resources = dict(
        methods.legacy.resources(private),
        free_disk_bytes=shutil.disk_usage(private).free,
        tools={
            n: (ROOT / ".venv/bin" / n).is_file()
            for n in ["python", "pytest", "coverage", "ruff", "mypy"]
        },
        task_cap_s=4800,
        validation_reserve_s=900,
        measurement_cap_s=3900,
        optional_cuda_required=False,
        historical_v722_required=False,
    )
    checks = list(authority["gate_check_summary"])
    if not (
        resources["private_disk_backed_scratch"]
        and resources["available_memory_bytes"] >= resources["minimum_memory_bytes"]
        and resources["free_disk_bytes"] >= 1024**3
        and all(resources["tools"].values())
    ):
        checks.append(gate(dict(path=str(private)), "required_resources", True, resources))
    tasks = (
        authority["tasks"]
        or methods.parse_design((ROOT / methods.DESIGN).read_text(), milestone=MILESTONE)[1]
    )
    log = freeze(root / "ops/conductor-log.md", raw / "scheduling")
    log_lines = Path(log["snapshot_path"]).read_text().splitlines() if log["exists"] else []
    slots, receipts, refs = [], [], [log]
    for index, task in enumerate(tasks[:-1]):
        progress("branch_before_" + task["id"], index, 13 - index)
        candidates = list((root / "results").glob(f"experiment_{8429 + index}_*.json"))
        path = candidates[0] if len(candidates) == 1 else root / task["deliverable"]
        primary = freeze(path, raw / "primaries" / str(8429 + index))
        refs.append(primary)
        summary = None
        if primary["exists"]:
            with patch.multiple(workers, CLI=CLI, progress=progress):
                summary, receipt = workers.invoke(task, primary, raw / "branches" / task["id"])
            receipts.append(receipt)
            refs.append(freeze(raw / "branches" / task["id"] / "summary.json", raw / "summaries"))
            if summary.get("primitive_reference"):
                refs.append(
                    freeze(Path(summary["primitive_reference"]["path"]), raw / "primitives")
                )
        cascade = next(
            (line for line in log_lines if "GATE_BLOCK" in line and task["id"] in line), None
        )
        current = slot(task, primary, cascade, summary)
        slots.append(current)
        checks.extend(current["checks"])
        progress("branch_after_" + task["id"], index + 1, 12 - index)
    after = memory()
    refs.extend(
        freeze(ROOT / name, raw / "code")
        for name in [
            *OWNED,
            TEST,
            "python/carnot/reporting/primary_publication.py",
            "scripts/publication_gate.py",
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
        ]
    )
    predecessors = [
        freeze(next((ROOT / "results").glob(f"experiment_{n}_*.json")), raw / "predecessors")
        for n in [8415, 8359, 8416, 8317]
    ]
    work = dict(
        root=str(root),
        raw=str(raw),
        tasks=tasks,
        authority=authority,
        contract_digest=methods.TASKS_PIN,
        slots=slots,
        refs=refs,
        predecessors=predecessors,
        branch_receipts=receipts,
        checks=checks,
        preconditions=resources,
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
    """Independent producer pins reject primitive substitutions even after local rehashing."""
    from carnot.reporting.v726_capstone_report import build

    try:
        value = json.loads(path.read_bytes())
        if value["reproducibility_checksum"] != checksum(value) or value["task_count"] != 14:
            return False
        work = load(value["primitive_reference"])
        for ref in [*work["refs"], *work["predecessors"], *work.get("coverage_shards", [])]:
            if ref.get("exists", True):
                if sha256_file(Path(str(ref.get("snapshot_path", ref["path"])))) != ref["sha256"]:
                    return False

        with TemporaryDirectory(prefix="exp8442-authority-", dir="/var/tmp") as directory:
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
        if work["contract_digest"] != methods.TASKS_PIN or work["tasks"][-1]["id"] != TASK:
            return False
        for index, row in enumerate(work["slots"]):
            if row["task"] != work["tasks"][index]:
                return False
            ref = row["primary_reference"]
            if ref["exists"]:
                with TemporaryDirectory(prefix="exp8442-branch-cold-", dir="/var/tmp") as folder:
                    with patch.multiple(workers, CLI=CLI, progress=progress):
                        actual, receipt = workers.invoke(row["task"], ref, Path(folder))
                if not receipt["passed"] or not actual["memory"]["passed"]:
                    return False
                if any(
                    actual[k] != row[k]
                    for k in actual
                    if k not in {"memory", "reader_initialization"}
                ):
                    return False
            elif row != slot(row["task"], ref, row["cascade_log_line"]):
                return False
        for receipt in [*work["branch_receipts"], *value["validation_receipts"]]:
            for stream in ["stdout", "stderr"]:
                if sha256_file(Path(receipt[stream + "_path"])) != receipt[stream + "_sha256"]:
                    return False
            saved = json.loads(
                Path(receipt["stdout_path"]).with_suffix(".receipt.json").read_bytes()
            )
            if any(saved[k] != receipt.get(k) for k in saved if k != "passed"):
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
    except (OSError, ValueError, KeyError, TypeError, IndexError, StopIteration):
        return False
