"""REQ-REPORT-8345: reconcile compact references without retaining source trees.

Each worker reads one authenticated artifact. Its rows remain in the sealed
source, so historical repetition cannot inflate the parent or sample counts.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import resource
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

import psutil

from carnot.reporting import v716_capstone_evidence as reader
from carnot.reporting import v718_capstone_evidence as legacy
from carnot.reporting import v718_capstone_reduction as synthesis
from carnot.reporting import v719_contract_replay as contract
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v709_execution import child, execute
from carnot.reporting.v710_contract_replay import require_reference, snapshot

Json = dict[str, Any]
ROOT, DESIGN, ACTIVE, STAGED, PROTOCOL = (
    contract.ROOT,
    contract.DESIGN,
    contract.ACTIVE,
    contract.STAGED,
    contract.PROTOCOL,
)
NAME, TASK, MILESTONE = "experiment_8345_v719_capstone", "exp8345-capstone", contract.MILESTONE
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v719_capstone_8345.py"
OWNED = [
    "python/carnot/reporting/v719_capstone_evidence.py",
    "python/carnot/reporting/v719_capstone_reduction.py",
    "python/carnot/reporting/v719_capstone.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
PROTOCOL_PIN, failure, read = legacy.PROTOCOL_PIN, legacy.failure, legacy.read


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real boundaries so pending work can be observed without padding time."""
    print(f"[exp8345] phase={phase} completed={completed} pending={pending}", flush=True)


def memory() -> Json:
    """Record both retained resident pages and the process lifetime high water mark."""
    return dict(
        current_rss_mb=psutil.Process().memory_info().rss / 1048576,
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    )


def operand(task: Json, item: Json) -> Json:
    """Use the qualified outcome reader and discard repeated nested source payloads."""
    before = memory()
    require_reference(item["reference"])
    require_reference(item["sidecar"])
    row, failures, source = reader.outcome(task, int(task["id"][3:7]), item)
    keys = [
        "model_invocation_counts",
        "historical_model_provenance",
        "static_ready_score",
        "heads_ready_score",
        "predictions_ready_score",
        "prediction_parity",
        "label_access_ledger",
        "sample_size_budget",
        "local_kernel_ready_score",
        "runtime_reader_ready_score",
        "changed_runtime_ready_score",
        "polarfire_graduation",
        "kv260_obligation",
        "gatemate_obligation",
        "historical_dispositions",
        "first_reduction_mismatch",
        "honest_verdict",
        "verdict_class",
    ]
    # Historical synthesis is referenced separately; importing its nested objects
    # would reproduce the retained-memory defect without adding observations.
    selected = {k: source[k] for k in keys if k in source and k != "historical_dispositions"}
    refs = [
        source[k]
        for k in ["work_reference", "measurement_reference", "replay_input_reference"]
        if source.get(k)
    ]
    if row["missing"] or row["disposition"] == "conductor_pre_gate":
        row["honest_verdict"] = None
    gates = (
        read(item["reference"]).get("gates_evaluated", [])
        if row["disposition"] == "conductor_pre_gate"
        else []
    )
    del source
    gc.collect()
    after = memory()
    return dict(
        row=row,
        failures=failures,
        selected=selected,
        primitive_references=refs,
        conductor_gates=gates,
        memory=dict(
            before=before,
            after=after,
            growth_mb=after["peak_rss_mb"] - before["peak_rss_mb"],
            passed=after["peak_rss_mb"] - before["peak_rss_mb"] <= 500,
        ),
    )


def worker(request: Path, output: Path) -> int:
    """A bounded child returns a compact reduction and exposes its own RSS growth."""
    progress("worker_before")
    value = json.loads(request.read_bytes())
    result = operand(value["task"], value["item"])
    atomic_json(output, result)
    progress("worker_after", 1, 0)
    return int(not result["memory"]["passed"])


def compact(task: Json, item: Json, raw: Path, name: str) -> tuple[Json, Json | None]:
    """Cold readers use the same single-artifact child as the initial measurement."""
    if not item["reference"]["exists"]:
        return operand(task, item), None
    request, output = raw / (name + "_request.json"), raw / (name + "_summary.json")
    atomic_json(request, dict(task=task, item=item))
    receipt = child(
        name,
        [
            str(ROOT / ".venv/bin/python"),
            "-u",
            str(ROOT / CLI),
            "--worker-request",
            str(request),
            "--worker-output",
            str(output),
        ],
        raw / "logs",
        deadline=180,
        heartbeat=20,
    )
    if not output.is_file():
        raise ValueError("owned_worker_failed: " + str(receipt))
    return dict(json.loads(output.read_bytes())), receipt


def authority(work: Json) -> Json:
    """Full task objects, visible order and digest are checked from immutable copies."""
    with TemporaryDirectory(prefix="exp8345-authority-") as directory:
        root = Path(directory)
        for ref, name in zip(work["references"][:3], [DESIGN, STAGED, ACTIVE], strict=True):
            if ref["exists"]:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(Path(ref["snapshot_path"]).read_bytes())
        try:
            return dict(contract.authority(root, root / "assessment"))
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            return dict(
                activated=False,
                tasks=work["tasks"],
                gate_check_summary=[
                    failure(Path(work["root"]) / DESIGN, "authority_available", True, None)
                ],
            )


def measure(root: Path, raw: Path) -> Json:
    """Freeze exact declared primaries; future paths are dependencies, never fixtures."""
    progress("measurement_before", 0, 14)
    start, wall, before = time.monotonic_ns(), time.time_ns(), memory()
    refs = [
        snapshot(root / name, raw / "custody", str(i))
        for i, name in enumerate([DESIGN, STAGED, ACTIVE, PROTOCOL])
    ]
    design = refs[0] if refs[0]["exists"] else snapshot(ROOT / DESIGN, raw / "custody", "identity")
    tasks = parse_design(Path(design["snapshot_path"]).read_text(), milestone=MILESTONE)[1]
    if [t["id"].split("-")[0] for t in tasks] != [f"exp{i}" for i in range(8332, 8346)]:
        raise ValueError("exact_fourteen_task_contract")
    inputs, audits, plan = [], [], []
    for index, task in enumerate(tasks[:-1]):
        progress("input_before", index, 13 - index)
        path = reader.prior.resolve(task, 8332 + index, root)
        declared = root / task["deliverable"]
        if path != declared:
            refs.append(snapshot(declared, raw / "custody", str(len(refs))))
        item = reader.bind(path, raw, refs)
        summary, receipt = compact(task, item, raw / "operands", f"input_{8332 + index}")
        item["summary"] = summary
        item["summary_sha256"] = canonical_hash(summary)
        inputs.append(item)
        if receipt is not None:
            audits.append(receipt)
        for gate in summary["conductor_gates"]:
            refs.append(
                dict(
                    snapshot(Path(gate["artifact_path"]), raw / "custody", str(len(refs))),
                    expected_sha256=gate["artifact_sha256"],
                )
            )
        for ref in summary["primitive_references"]:
            refs.append(
                dict(
                    snapshot(Path(ref["path"]), raw / "custody", str(len(refs))),
                    expected_sha256=ref["sha256"],
                )
            )
        script = ROOT / "scripts/experiments" / (Path(task["deliverable"]).stem + ".py")
        if summary["row"]["producer_executed"] and script.is_file():
            plan.append(
                dict(
                    name=f"branch_{8332 + index}_replay",
                    argv=[
                        str(ROOT / ".venv/bin/python"),
                        "-u",
                        str(script),
                        "--cold-replay",
                        item["reference"]["snapshot_path"],
                    ],
                    expected=0,
                    deadline=180,
                    scope="owned",
                )
            )
        progress("input_after", index + 1, 12 - index)
    history = {}
    for name, version in [
        ("research-roadmap-v718-preserved-20261009.md", "2026.10.718"),
        ("research-roadmap-v717-preserved-20261008.md", "2026.10.717"),
    ]:
        path = root / "openspec/change-proposals" / name
        refs.append(snapshot(path, raw / "custody", str(len(refs))))
        if path.is_file():
            for task in parse_design(path.read_text(), milestone=version)[1]:
                if any(
                    p["experiment_id"] == task["id"] for t in tasks for p in t["prior_failures"]
                ):
                    history[task["id"]] = reader.bind(root / task["deliverable"], raw, refs)
    fixtures = []
    for name in [
        "results/experiment_8317_v717_capstone.json",
        "results/experiment_8331_v718_capstone.json",
        "tests/python/test_v718_capstone_8331.py",
        "python/carnot/testing/pytest_memory_watchdog.py",
        "ops/exclusion_manifest.yaml",
    ]:
        ref = snapshot(root / name, raw / "custody", str(len(refs)))
        refs.append(ref)
        fixtures.append(ref)
    publication = dict(
        name="publication_gate",
        argv=[str(ROOT / ".venv/bin/python"), "-u", "scripts/publication_gate.py", "--json"],
        expected=0,
        deadline=90,
        scope="publication",
    )
    atomic_json(raw / "branch_manifest.json", dict(commands=plan, publication=publication))
    audits.extend(execute(plan, raw / "branches"))
    gate = execute([publication], raw / "publication")[0]
    after = memory()
    work = dict(
        root=str(root),
        tasks=tasks,
        inputs=inputs,
        history=history,
        references=refs,
        failures=[],
        audits=audits,
        publication=gate,
        historical_fixture_hashes=fixtures,
        started_monotonic_ns=start,
        started_wall_ns=wall,
        ended_monotonic_ns=time.monotonic_ns(),
        memory_measurements=dict(
            parent_before=before,
            parent_after=after,
            parent_growth_mb=after["peak_rss_mb"] - before["peak_rss_mb"],
            workers=[i["summary"]["memory"] for i in inputs],
            retained_payload_bytes=len(json.dumps(inputs).encode()),
        ),
    )
    suite = os.environ.get("CARNOT_8345_REPOSITORY_SUITE_RECEIPT")
    if suite:
        ref = snapshot(Path(suite), raw / "custody", str(len(refs)))
        refs.append(ref)
        work["repository_suite_attempt"] = read(ref)
    atomic_json(raw / "measurement.json", work)
    progress("measurement_after", 13, 1)
    return work


def reduce(work: Json, receipts: list[Json]) -> Json:
    """Delegate scientific disposition to the compact reducer rather than old synthesis."""
    from carnot.reporting.v719_capstone_reduction import reduce as reduction

    return reduction(work, receipts)


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Reuse artifact normalization while binding the new invocation and memory scope."""
    if "owned_parent_after" not in work["memory_measurements"]:
        work["memory_measurements"]["owned_parent_after"] = memory()
    for receipt in receipts:
        if receipt.get("name") == "v718_mixed_history":
            lines = Path(receipt["stdout_path"]).read_text().splitlines()
            measurements = [
                json.loads(line.split("=", 1)[1])
                for line in lines
                if line.startswith("memory_receipt=")
            ]
            work["memory_measurements"]["historical_workers"] = measurements
    with (
        patch.object(synthesis, "reduce", reduce),
        patch.object(legacy, "OWNED", OWNED),
        patch.object(legacy, "TEST", TEST),
    ):
        value = dict(legacy.build(work, receipts, raw, output))
    value.update(
        experiment_id=8345,
        task_id=TASK,
        milestone=MILESTONE,
        schema="carnot.v719.capstone.v1",
        random_seed=7198345,
        historical_fixture_hashes=work["historical_fixture_hashes"],
        memory_measurements=work["memory_measurements"],
        memory_bounds=dict(growth_mb=500, retained_payload_bytes=8_000_000),
        finding_consumer_qualification="owned test_finding_policy before publication",
        repository_suite_attempt=work.get("repository_suite_attempt"),
    )
    for field in value:
        value["field_principles"].setdefault(
            field,
            "Bind current execution to exact bytes; historical synthesis and missing science confer no new observations.",
        )
    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    return value


def replay(path: Path) -> bool:
    """Fresh reconstruction rejects rehashed claims and changed compact primitives."""
    try:
        value = json.loads(path.read_bytes())
        for ref in [
            value["work_reference"],
            *value["source_artifact_hashes"],
            *value["code_config_hashes"],
        ]:
            require_reference(ref)
        work = read(value["work_reference"])
        if work["frozen_validation_receipts"] != value["validation_receipts"]:
            return False
        rebuilt = build(
            work,
            value["validation_receipts"],
            Path(value["work_reference"]["path"]).parent,
            Path(value["publication_output"]),
        )
        if rebuilt != value:
            return False
        for receipt in value["validation_receipts"] + work["audits"] + [work["publication"]]:
            for stream in ["stdout", "stderr"]:
                require_reference(
                    dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
                )
        with TemporaryDirectory(prefix="exp8345-replay-") as directory:
            for index, (task, item) in enumerate(
                zip(work["tasks"][:-1], work["inputs"], strict=True)
            ):
                actual, receipt = compact(task, item, Path(directory), f"replay_{index}")
                if (receipt is not None and not receipt["passed"]) or {
                    k: v for k, v in actual.items() if k != "memory"
                } != {k: v for k, v in item["summary"].items() if k != "memory"}:
                    return False
        return True
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return False
