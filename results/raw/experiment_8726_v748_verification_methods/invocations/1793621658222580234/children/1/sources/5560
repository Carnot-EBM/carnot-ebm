"""REQ-REPORT-8564: original seals keep scheduling separate from scientific success."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import v734_capstone_evidence as previous
from carnot.reporting import v735_contract_methods as methods
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import (
    read_bound_sidecar,
    reader_receipt,
    validate_primary,
)

Json = dict[str, Any]
ROOT = methods.ROOT
NAME, TASK, MILESTONE = "experiment_8564_v735_capstone", "exp8564-capstone", "2026.10.735"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v735_capstone_8564.py"
OWNED = [
    f"python/carnot/reporting/v735_capstone{s}.py" for s in ["_evidence", "_samples", "_report", ""]
] + [CLI]
MODEL_SPECS: list[Json] = []
PINS = dict(
    zip(
        [8552, 8553, 8554, 8555, 8556, 8558, 8559, 8560, 8561, 8562, 8563],
        [
            "f8ec5af49484f9abda497bfc89aef8a65e4463b0fd32acf1028fae9ca3c3dab7",
            "582e1402c6ae68f2fae60dfc0e3564935facf5abc0bcadf76c43ffc164793230",
            "11366167c60a52398a2d0a4d88c9fbfe879292b445492990438985adb081e9bd",
            "e690ae89a0c2219d9ebea8558fca68168c07157190bef61ebe31313d4afee2f2",
            "a5a562dfc1a34e778cf4c8fa283971bf7cf234c4f5ee8ceba7d2505c3db4ad74",
            "efbf55b9e5cbdbd84c3e26040d53b7c76226fee6667595b1176c63d73538310d",
            "d08c9868dc613dd1b8f7ee7e53a2e6e5497d16a2526dcf7e85e13e35eff7862c",
            "3f243b03b74f0a037bfd8b0c5542be4adf9223679941a8d2cd699bf4cfb7b792",
            "f0813491f06f827bac29b75f8c722136b214963b7be1882627b44467d0204ba8",
            "d7b9674a865bbc40c1fccf8009ca607c8e3363080eecadde22394246f8382745",
            "8afa61147f5868f76af1338c09fae6a72f4965464a19ede36acbefd8fdaacfba",
        ],
        strict=True,
    )
)
PRIOR = dict(
    path=str(ROOT / "results/experiment_8551_v734_capstone.json"),
    sha256="sha256:a262176cb8348efdac189ea971ac0b97d034f59a94b9960d4338cd38d802526d",
)
PREGATE_PATHS = {
    n: f"results/experiment_{n}_{s}.json"
    for n, s in [
        (8556, "causal_brier_stream"),
        (8560, "qwen_evidence_canary"),
        (8562, "arc_capability_panel"),
    ]
}
checksum, reference, freeze, load, slot, memory, resources = (
    previous.checksum,
    previous.reference,
    previous.freeze,
    previous.load,
    previous.slot,
    previous.memory,
    previous.resources,
)
child = previous.child


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual counts so CPU reconciliation remains visible without model activity."""
    print(f"[exp8564] phase={phase} completed={completed} pending={pending}", flush=True)


def gate(ref: Json, field: str, expected: Any, observed: Any, upstream: str = TASK) -> Json:
    """Attribute current missing-resource and failure checks to this invocation."""
    return dict(previous.gate(ref, field, expected, observed, upstream))


def outcome(task: Json, primary: Json, raw: Path) -> Json:
    """Reuse terminal readers while adapting the release producer's original primitive key."""
    number, value = int(task["id"][3:7]), load(primary)
    if primary["sha256"] != "sha256:" + PINS[number]:
        raise ValueError("primary_pin")
    if value.get("blocked_at_layer") == "conductor_pre_gate":
        with patch.dict(previous.PINS, PINS):
            result = previous.outcome(task, primary, raw)
        return dict(result, original_run_date=value["run_date"])
    validate_primary(value, Path(primary["path"]))
    if value["task_id"] != task["id"]:
        raise ValueError("task_identity")
    primitive = value.get(
        "primitive_reference", value.get("work_reference", value.get("source_reference"))
    )
    if sha256_file(Path(primitive["path"])) != primitive["sha256"]:
        raise ValueError("primitive_hash")
    terminal_ref = reference(Path(value["terminal_validation_sidecar_path"]))
    terminal = load(terminal_ref)["publication"]
    bound_ref = reference(Path(terminal["sidecar_path"]))
    binding_error = None
    try:
        bound = read_bound_sidecar(Path(primary["path"]), Path(bound_ref["path"]))
        sealed = (
            terminal["primary_sha256"] == primary["sha256"] == bound["primary_sha256"]
            and bound["report"]["passed"] is True
        )
    except ValueError as error:
        binding_error, sealed = str(error), False
    consumer = reader_receipt(
        task["id"],
        Path(primary["path"]).parent,
        field="required_checks_passed",
        expected=value["required_checks_passed"],
    )
    eligible = bool(
        sealed
        and consumer["passed"]
        and value["required_checks_passed"]
        and not value["flagged_adversarial"]
        and value["verdict_class"] != "disqualified"
    )
    return dict(
        producer_executed=True,
        eligible=eligible,
        selected={
            k: v
            for k, v in value.items()
            if k.endswith("_score")
            or k
            in {
                "admission_decision",
                "complete_throughput_ratio",
                "throughput_ratio_ci95",
                "paired_repeat_rows",
                "operation_shares",
                "per_batch_rows",
            }
        },
        checks=[]
        if eligible
        else [
            gate(
                primary,
                "upstream_required_checks",
                True,
                dict(
                    sealed=sealed,
                    binding_error=binding_error,
                    required_checks_passed=value["required_checks_passed"],
                    flagged_adversarial=value["flagged_adversarial"],
                ),
                task["id"],
            )
        ],
        disposition="qualified_producer" if eligible else "disqualified_producer",
        honest_verdict=value["honest_verdict"],
        verdict_class=value["verdict_class"],
        source_gates=value["gate_check_summary"],
        historical_model_provenance=value["historical_model_provenance"],
        upstream_adversarial_findings=value["adversarial_findings"],
        upstream_failed_receipts=[r for r in value["validation_receipts"] if not r["passed"]],
        upstream_flagged_adversarial=value["flagged_adversarial"],
        original_required_checks_passed=value["required_checks_passed"],
        original_run_date=value["run_date"],
        primitive_reference=primitive,
        evidence_references=[primitive, terminal_ref, bound_ref],
        downstream_consumer=consumer,
        branch_replay=dict(
            status="authenticated_terminal_and_primitive_seals", passed=bool(sealed)
        ),
    )


def worker_process(request: Path, output: Path) -> int:
    """The bounded shipped sample child isolates historical operand allocations."""
    from carnot.reporting.v735_capstone_samples import samples

    with patch.multiple(previous, TASK=TASK, samples=samples, progress=progress):
        return int(previous.worker_process(request, output))


def sample_child(raw: Path) -> tuple[Json, Json]:
    """A new interpreter recomputes a small diagnostic without rerunning experiments."""
    with patch.multiple(previous, TASK=TASK, CLI=CLI, child=child):
        diagnostic, receipt = previous.sample_child(raw)
        return dict(diagnostic), dict(receipt)


def measure(root: Path, raw: Path, private: Path) -> Json:
    """Authenticate current authority and inputs before bounded deterministic CPU work."""
    start, before = time.monotonic_ns(), memory()
    progress("preconditions", 0, 12)
    authority = methods.authority(root, raw / "authority")
    tasks = authority["tasks"] or methods.protocol()["task_map_v735"]
    if (
        methods.tasks_digest(tasks) != methods.TASKS_PIN
        or tasks[-1].get("requires")
        or tasks[-1].get("gated_on")
    ):
        raise ValueError("exact_unconditional_thirteen_tasks")
    probe = private / "exp8564-write-probe"
    probe.write_bytes(b"disk-backed scratch")
    resource = dict(
        resources(private),
        scratch_write_read_passed=probe.read_bytes() == b"disk-backed scratch",
        optional_cuda_required=False,
        optional_boards_required=False,
        task_cap_s=4800,
        measurement_cap_s=3600,
        validation_reserve_s=900,
    )
    checks = list(authority["gate_check_summary"])
    if not (
        resource["disk_backed"]
        and resource["memory_available_bytes"] >= 1500 * 1024**2
        and resource["free_disk_bytes"] >= 1024**3
        and resource["tools_available"]
    ):
        checks.append(gate(dict(path=str(private)), "required_resources", True, resource))
    log = freeze(root / "ops/conductor-log.md", raw / "scheduling")
    lines = Path(log["snapshot_path"]).read_text().splitlines() if log["exists"] else []
    activation = next(
        (i for i, line in enumerate(lines) if "Milestone 2026.10.735 activated" in line), len(lines)
    )
    lines = lines[activation:]
    slots, refs = [], [log]
    for index, task in enumerate(tasks[:-1]):
        progress("branch_before_" + task["id"], index, 12 - index)
        if (time.monotonic_ns() - start) / 1e9 >= 3600:
            raise TimeoutError("measurement_deadline")
        number = int(task["id"][3:7])
        primary = freeze(root / task["deliverable"], raw / "primaries" / str(number))
        refs.append(primary)
        if (
            not primary["exists"]
            and number in PREGATE_PATHS
            and (root / PREGATE_PATHS[number]).is_file()
        ):
            primary = freeze(root / PREGATE_PATHS[number], raw / "pregates" / str(number))
            refs.append(primary)
        summary = outcome(task, primary, raw) if primary["exists"] else None
        cascade = next(
            (
                line
                for line in lines
                if "GATE_BLOCK" in line and line.split("|")[2].strip() == task["title"][:50].strip()
            ),
            None,
        )
        row = slot(task, primary, cascade, summary)
        slots.append(row)
        checks.extend(row["checks"])
        refs.extend(row.get("evidence_references", []))
        progress("branch_after_" + task["id"], index + 1, 11 - index)
    predecessor = freeze(Path(PRIOR["path"]), raw / "predecessors")
    if predecessor["sha256"] != PRIOR["sha256"]:
        raise ValueError("original_v734_pin")
    progress("deterministic_sample_before", 12, 1)
    diagnostic, sample_receipt = sample_child(raw)
    if (
        not diagnostic["learning"]["matches_original"]
        or not diagnostic["arc"]["matches_original"]
        or not diagnostic["native_cost"]["qualified"]
    ):
        checks.append(gate(predecessor, "sample_reconciliation", True, False))
        sample_receipt = dict(sample_receipt, passed=False)
    progress("deterministic_sample_after", 13, 0)
    refs.extend(
        freeze(ROOT / name, raw / "code")
        for name in [
            *OWNED,
            TEST,
            methods.PROTOCOL,
            "python/carnot/reporting/primary_publication.py",
            "scripts/publication_gate.py",
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
            "ops/exclusion_manifest.yaml",
            "docs/research-notes/v735-capstone.md",
        ]
    )
    after, end = memory(), time.monotonic_ns()
    work = dict(
        root=str(root),
        raw=str(raw),
        tasks=tasks,
        authority=authority,
        contract_digest=methods.TASKS_PIN,
        slots=slots,
        refs=refs,
        predecessors=[predecessor],
        branch_receipts=[sample_receipt],
        checks=checks,
        preconditions=resource,
        samples=diagnostic,
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
                name="terminal_evidence_reduction",
                started_monotonic_ns=start,
                ended_monotonic_ns=end,
            )
        ],
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
        adversarial_findings=[],
    )
    atomic_json(raw / "measurement.json", work)
    return work


def replay(path: Path) -> bool:
    """Rebuild original rows and authority so repaired envelope hashes cannot invent claims."""
    from carnot.reporting.v735_capstone_report import build

    try:
        value = json.loads(path.read_bytes())
        work = load(value["primitive_reference"])
        if (
            value["reproducibility_checksum"] != checksum(value)
            or value["task_count"] != 13
            or methods.tasks_digest(work["tasks"]) != methods.TASKS_PIN
            or work["contract_digest"] != methods.TASKS_PIN
            or work["predecessors"][0]["sha256"] != PRIOR["sha256"]
        ):
            return False
        for ref in [*work["refs"], *work["predecessors"], *work.get("coverage_shards", [])]:
            if (
                ref.get("exists", True)
                and sha256_file(Path(ref.get("snapshot_path", ref["path"]))) != ref["sha256"]
            ):
                return False
        with TemporaryDirectory(prefix="exp8564-authority-", dir="/var/tmp") as directory:
            cold = Path(directory)
            for role, name in [
                ("design", methods.DESIGN),
                ("active", methods.ACTIVE),
                ("staged", methods.STAGED),
            ]:
                ref = work["authority"]["authority_snapshots"][role]
                if ref["exists"]:
                    target = cold / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(Path(ref["snapshot_path"]).read_bytes())
                    if sha256_file(target) != ref["sha256"]:
                        return False
            actual = methods.authority(cold, cold / "checked")
            if (
                actual["activated"] != work["authority"]["activated"]
                or actual["tasks"] != work["authority"]["tasks"]
            ):
                return False
        for row, task in zip(work["slots"], work["tasks"][:-1], strict=True):
            summary = (
                outcome(task, row["primary_reference"], Path(work["raw"]))
                if row["primary_reference"]["exists"]
                else None
            )
            if row != slot(task, row["primary_reference"], row["cascade_log_line"], summary):
                return False
        with TemporaryDirectory(prefix="exp8564-sample-", dir="/var/tmp") as directory:
            diagnostic, _ = sample_child(Path(directory))
        if work["samples"] != diagnostic:
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
