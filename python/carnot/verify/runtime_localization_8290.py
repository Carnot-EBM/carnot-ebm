"""REQ-REPORT-8290 / REQ-VERIFY-8290: localize execution without changing science.

The existing supervisor owns checks and publication. This adapter supplies
current authority and separate CUDA primitives; no weights or tokens are used.
"""

from __future__ import annotations

from contextlib import contextmanager, ExitStack
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any, Iterator
from unittest.mock import patch

from carnot.gpu_lease_phase_journal import GpuLease, LeaseError
from carnot.reporting import current_contract_readiness_8276 as prior
from carnot.reporting import v714_coverage_runner as runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v685_authority_lifecycle import assess_authorities
from carnot.reporting.v709_execution import child
from carnot.reporting.v710_contract_replay import snapshot, require_reference
from carnot.verify import cuda_primitive_8290 as primitive

Json = dict[str, Any]
ROOT = prior.ROOT
NAME = "experiment_8290_v716_runtime_localization"
TASK = "exp8290-runtime-localization"
MILESTONE = "2026.10.716"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_runtime_localization_8290.py"
OWNED = [
    "python/carnot/verify/runtime_localization_8290.py",
    "python/carnot/verify/cuda_primitive_8290.py",
    CLI,
]
REUSED = (
    prior.REUSED
    + prior.OWNED
    + [
        "python/carnot/gpu_lease_phase_journal.py",
        "python/carnot/verify/lease_backend_8277.py",
        "AGENTS.md",
        "CODEX.md",
        "CLAUDE.md",
        "ops/e2e-test-plan.md",
        "ops/exclusion_manifest.yaml",
        "research-references.md",
        "openspec/capabilities/research-reporting/spec.md",
        "openspec/capabilities/verification/spec.md",
    ]
)
DESIGN, ACTIVE, STAGED, PROTOCOL, PIN = (
    prior.DESIGN,
    prior.ACTIVE,
    prior.STAGED,
    prior.PROTOCOL,
    prior.PIN,
)
EXECUTION = "openspec/change-proposals/v716-evidence-execution-contract.json"
MODEL_SPECS: list[Json] = []
SCORES = [
    "current_contract_ready_score",
    "coverage_custody_ready_score",
    "view_kernel_ready_score",
    "admission_kernel_ready_score",
]
failure = prior.prior.failure


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush boundaries so supervision can distinguish progress from a stall."""
    print(f"[exp8290] phase={phase} completed={completed} pending={pending}", flush=True)


@contextmanager
def bindings() -> Iterator[None]:
    """Supply current producer constants without editing the qualified readers."""
    with ExitStack() as stack:
        for key in ["NAME", "TASK", "MILESTONE", "CLI", "TEST", "OWNED", "REUSED", "EXECUTION"]:
            stack.enter_context(patch.object(prior, key, globals()[key]))
        stack.enter_context(patch.object(prior, "assess", assess))
        yield


def assess(paths: list[Path], raw: Path) -> Json:
    """Use the original reader with the explicit current milestone and range."""
    reader = raw / "reader_design.md"
    reader.parent.mkdir(parents=True, exist_ok=True)
    reader.write_text(
        paths[0].read_text().replace("Canonical task digest:", "Canonical full-task SHA256:")
    )
    return dict(
        assess_authorities(
            reader, *paths[1:3], raw / "assessment", milestone=MILESTONE, first_id=8290, count=14
        )
    )


def authority_work(root: Path, raw: Path) -> Json:
    """Snapshot the full execution binding while retaining the pinned science."""
    with bindings():
        return dict(prior.authority_work(root, raw))


def manifest(private: Path, candidate: Path) -> Json:
    """Freeze scoped owned coverage and unchanged private component checks first."""
    with bindings():
        return dict(prior.manifest(private, candidate))


def diagnostic(raw: Path, uuid: str, binary: Path) -> Json:
    """Lease the permitted GPU once; every configuration runs each layer once."""
    raw.mkdir(parents=True, exist_ok=True)
    result: Json = dict(
        rows=[],
        checks=[],
        lease_receipt={},
        cleanup_receipt={},
        device_inventory=[],
        runtime_binding={},
        phase_spans=[],
    )
    inventory = child(
        "inventory",
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        raw,
        deadline=15,
        scope="observation",
    )
    result["inventory_receipt"] = inventory
    result["device_inventory"] = [
        dict(
            index=c[0].strip(),
            uuid=c[1].strip(),
            name=c[2].strip(),
            driver_version=c[3].strip(),
            memory_used_mb=int(c[4]),
            memory_free_mb=int(c[5]),
        )
        for line in Path(inventory["stdout_path"]).read_text().splitlines()
        if len(c := line.split(",")) == 6
    ]
    result["device_nodes"] = primitive.node_access(sorted(Path("/dev").glob("nvidia*")))
    result["driver_module_version"] = (
        Path("/proc/driver/nvidia/version").read_text()
        if Path("/proc/driver/nvidia/version").exists()
        else None
    )
    runtimes = sorted(
        (ROOT / ".venv/lib").glob("python*/site-packages/nvidia/cuda_runtime/lib/libcudart.so*")
    )
    runtime = str(runtimes[0]) if runtimes else "libcudart.so.12"
    result["runtime_binding"] = dict(
        permitted_uuid=uuid,
        inherited={k: os.environ[k] for k in primitive.VARIABLES if k in os.environ},
        explicit_uuid={"CUDA_VISIBLE_DEVICES": uuid},
        driver_library="libcuda.so.1",
        runtime_library=runtime,
        binary=str(binary),
        binary_sha256=sha256_file(binary) if binary.is_file() else None,
        native_library_hashes=[
            dict(path=str(p), sha256=sha256_file(p))
            for p in sorted(binary.parent.glob("lib*.so*"))
            if p.is_file()
        ],
        adapter_authorized=False,
    )
    lease: Any = None
    started = time.monotonic_ns()
    try:
        if not uuid or not any(r["uuid"] == uuid for r in result["device_inventory"]):
            raise ValueError("permitted_uuid_inventory_missing")
        lease = GpuLease.acquire(
            runtime_dir="/tmp/carnot-gpu-leases",
            task_id=TASK,
            device_uuid=uuid,
            expected_model="no_model_load:cuda_byte_copy",
            vram_before_mb=0,
            ttl_s=420,
        )
        result["lease_receipt"] = lease.owner_receipt()
        lease.transition("admitted")
        for binding in ["inherited", "explicit_uuid"]:
            for layer in ["driver", "runtime", "native"]:
                progress("cuda_matrix", len(result["rows"]), 6 - len(result["rows"]))
                command = [
                    str(ROOT / ".venv/bin/python"),
                    "-u",
                    str(ROOT / CLI),
                    "--cuda-probe",
                    layer,
                    "--uuid",
                    uuid,
                    "--binary",
                    str(binary),
                ]
                if layer == "runtime":
                    command += ["--library", runtime]
                if binding == "explicit_uuid":
                    command = ["/usr/bin/env", "CUDA_VISIBLE_DEVICES=" + uuid, *command]
                remaining = 360 - (time.monotonic_ns() - started) / 1e9
                receipt = child(
                    binding + "_" + layer,
                    command,
                    raw,
                    deadline=min(59, max(0.01, remaining)),
                    heartbeat=20,
                    scope="external_cuda",
                )
                text = Path(receipt["stdout_path"]).read_text().splitlines()
                value = (
                    json.loads(text[-1]) if receipt["passed"] and text else dict(error="child_exit")
                )
                result["rows"].append(
                    dict(binding=binding, layer=layer, receipt=receipt, primitive=value)
                )
        lease.transition("terminal_blocked")
    except (OSError, ValueError, LeaseError) as error:
        result["checks"].append(failure(raw, "permitted_gpu_lease", True, str(error)))
    finally:
        if lease:
            if lease.document["phase"] not in ("terminal_blocked", "terminal_complete"):
                lease.transition("terminal_blocked")
            result["cleanup_receipt"] = lease.release()
            result["lease_receipt"]["release"] = result["cleanup_receipt"]
        result["phase_spans"] = [
            dict(
                phase="cuda_matrix",
                started_monotonic_ns=started,
                ended_monotonic_ns=time.monotonic_ns(),
                duration_s=(time.monotonic_ns() - started) / 1e9,
            )
        ]
    return result


def measure(root: Path, raw: Path) -> Json:
    """Re-execute qualified controls and preserve old CUDA failures as imports."""
    progress("before_current_components")
    with bindings():
        work = prior.measure(root, raw)
    historical = {}
    for exp, version, suffix in [
        (8264, "v714", "evidence_view_canary"),
        (8276, "v715", "current_contract_readiness"),
        (8277, "v715", "lease_backend_qualification"),
        (8289, "v715", "capstone"),
    ]:
        historical[exp] = prior.bind_terminal(
            root, f"results/experiment_{exp}_{version}_{suffix}.json", work, raw, "history"
        )
    for item in historical[8289].get("task_dispositions", []):
        path = (
            Path(item["path"])
            if item.get("path")
            else root / "results/experiment_8289_v715_capstone.json"
        )
        ref = snapshot(path, raw / "history", str(item["experiment_id"]))
        work["refs"].append(dict(ref, fields_imported=["historical disposition"]))
        source = json.loads(Path(ref["snapshot_path"]).read_bytes()) if ref["exists"] else {}
        row = dict(
            task_id=item["task_id"],
            path=str(path),
            sha256=ref["sha256"],
            disposition=(
                "conductor_pre_gate"
                if item.get("evidence_type") == "conductor_pre_gate"
                else "producer_terminal"
            )
            if source
            else "cascade_skip_absent_primary",
            source_counts={
                k: source.get(k)
                for k in [
                    "intended_count",
                    "completed_count",
                    "failed_count",
                    "censored_count",
                    "excluded_count",
                ]
            },
        )
        if source:
            row.update(
                honest_verdict=source.get("honest_verdict"),
                verdict_class=source.get("verdict_class"),
            )
        work["history"].append(row)
    runtime_import = historical[8277]
    for receipt in [
        runtime_import.get("backend_identity", {}).get("enumeration_receipt", {}),
        historical[8264].get("cuda_preflight_receipt", {}),
    ]:
        for prefix in ["stdout", "stderr"]:
            if prefix + "_path" in receipt:
                path = Path(receipt[prefix + "_path"])
                ref = snapshot(path, raw / "historical_streams", prefix)
                work["refs"].append(dict(ref, fields_imported=["historical CUDA stream"]))
                if ref["sha256"] != receipt[prefix + "_sha256"]:
                    work["failures"].append(
                        dict(
                            failure(
                                path,
                                "historical_stream_hash",
                                receipt[prefix + "_sha256"],
                                ref["sha256"],
                            ),
                            component="history",
                        )
                    )
    # The old preflight is retained through the authenticated canary primary too.
    work["historical_runtime"] = {str(k): v for k, v in historical.items()}
    uuid = runtime_import.get("gpu_lease_receipt", {}).get("device_uuid", "")
    binary = Path(
        runtime_import.get("backend_identity", {}).get(
            "path", Path.home() / ".cache/llama.cpp-master/build/bin/llama-server"
        )
    )
    work["diagnostic"] = diagnostic(raw / "cuda", uuid, binary)
    work["duration_s"] += work["diagnostic"]["phase_spans"][0]["duration_s"]
    progress("after_current_components_and_cuda", 6)
    return dict(work)


def cuda_ready(diag: Json) -> bool:
    """Require context parity, runtime and native agreement under one binding."""
    for binding in ["inherited", "explicit_uuid"]:
        rows = {r["layer"]: r for r in diag["rows"] if r["binding"] == binding}
        if set(rows) != {"driver", "runtime", "native"}:
            continue
        if (
            all(r["receipt"]["passed"] and r["receipt"]["normal_exit"] for r in rows.values())
            and rows["driver"]["primitive"].get("context_copy_ready") is True
            and rows["runtime"]["primitive"].get("api_returns", {}).get("cudaGetDeviceCount") == 0
            and (rows["runtime"]["primitive"].get("device_count") or 0) > 0
            and rows["native"]["primitive"].get("native_compatible") is True
            and diag["cleanup_receipt"].get("released") is True
        ):
            return True
    return False


def reduce(work: Json, receipts: list[Json], coverage_ok: bool, cold_ok: bool) -> Json:
    """Readiness fields depend on their own primitives, never a stale headline."""
    with bindings():
        value = prior.reduce(work, receipts, coverage_ok, cold_ok)
    for index, row in enumerate(value["rows"][:14]):
        row["unit_id"] = f"exp{8290 + index}"
    diag = work["diagnostic"]
    ready = cuda_ready(diag) and value["required_checks_passed"]
    failed = list(diag["checks"])
    for row in diag["rows"]:
        p = row["primitive"]
        field = (
            "context_copy_ready"
            if row["layer"] == "driver"
            else "native_compatible"
            if row["layer"] == "native"
            else "cudaGetDeviceCount"
        )
        expected = 0 if row["layer"] == "runtime" else True
        observed = (
            p.get("api_returns", {}).get(field) if row["layer"] == "runtime" else p.get(field)
        )
        if observed != expected:
            failed.append(
                dict(
                    failure(Path(row["receipt"]["stdout_path"]), field, expected, observed),
                    binding=row["binding"],
                    api_returns=p.get("api_returns", {}),
                )
            )
    value["gate_check_summary"].extend(failed)
    verdict = (
        "disqualified"
        if not value["required_checks_passed"]
        else "blocked"
        if value["gate_check_summary"] or not ready
        else "circular_positive"
    )
    value.update(
        experiment_id=8290,
        task_id=TASK,
        milestone=MILESTONE,
        verdict_class=verdict,
        honest_verdict="complete_"
        + verdict
        + "_"
        + ("cuda_runtime" if verdict == "blocked" else "runtime_localization"),
        cuda_context_ready_score=int(ready),
        cuda_probe_rows=diag["rows"],
        device_inventory=diag["device_inventory"],
        lease_receipt=diag["lease_receipt"],
        cleanup_receipt=diag["cleanup_receipt"],
        failure_layer=[
            dict(
                binding=r["binding"],
                layer=r["layer"],
                failed_api={k: v for k, v in r["primitive"].get("api_returns", {}).items() if v},
                error=r["primitive"].get("error"),
                native_stderr=r["primitive"].get("stderr"),
            )
            for r in diag["rows"]
            if not (
                r["primitive"].get("context_copy_ready")
                or r["primitive"].get("native_compatible")
                or (
                    r["layer"] == "runtime"
                    and r["primitive"].get("api_returns", {}).get("cudaGetDeviceCount") == 0
                )
            )
        ],
        root_cause_status="unproved",
        no_model_load=True,
    )
    value["acceptance_gates"]["cuda_context"] = ready
    return dict(value)


def build(
    work: Json,
    raw: Path,
    receipts: list[Json],
    binding: Json,
    cold: list[Json],
    scratch: Path,
    health: Json,
) -> Json:
    """Freeze runtime bindings alongside existing durable coverage and protocol evidence."""
    with bindings():
        value = prior.build(work, raw, receipts, binding, cold, scratch, health)
    path = raw / "runtime_binding.json"
    atomic_json(path, work["diagnostic"]["runtime_binding"])
    value.update(
        inference_substrate="deterministic_runtime_receipt_validation_no_llm",
        runtime_binding_path=str(path),
        runtime_binding_sha256=sha256_file(path),
        historical_runtime_imports=work.get("historical_runtime", {}),
    )
    value["raw_shard_hashes"].append(dict(path=str(path), sha256=sha256_file(path)))
    value["phase_spans"].extend(work["diagnostic"]["phase_spans"])
    value["field_principles"].update(
        {
            k: "Current CUDA primitives remain separate from imported science; readiness grants no benefit."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    return dict(value)


def replay(path: Path) -> Json:
    """Reconstruct gates from frozen primitives after scratch and live authority change."""
    value = json.loads(path.read_bytes())
    checksum = value.pop("reproducibility_checksum")
    if checksum != canonical_hash(value):
        raise ValueError("candidate_checksum")
    if (value["experiment_id"], value["task_id"], value["milestone"], value["run_date"]) != (
        8290,
        TASK,
        MILESTONE,
        "20261008",
    ):
        raise ValueError("invocation_identity")
    for ref in [
        value["work_reference"],
        value["component_primitive_reference"],
        *value["source_artifact_hashes"],
        *value["code_config_hashes"],
        *value["raw_shard_hashes"],
    ]:
        require_reference(ref)
    for receipt in value["validation_receipts"] + value["cold_replay_rows"]:
        for prefix in ["stdout", "stderr"]:
            if sha256_file(Path(receipt[prefix + "_path"])) != receipt[prefix + "_sha256"]:
                raise ValueError("validation_stream_hash")
    work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
    for row in work["diagnostic"]["rows"]:
        receipt = row["receipt"]
        for prefix in ["stdout", "stderr"]:
            if sha256_file(Path(receipt[prefix + "_path"])) != receipt[prefix + "_sha256"]:
                raise ValueError("cuda_stream_hash")
        if receipt["passed"]:
            actual = json.loads(Path(receipt["stdout_path"]).read_text().splitlines()[-1])
            if actual != row["primitive"]:
                raise ValueError("cuda_primitive_drift")
    measured = (
        prior.custody.replay(value["coverage_command_receipt"])
        if value["coverage_command_receipt"]
        else {}
    )
    if (
        measured != value["owned_statement_counts"]
        or not value["scratch_removed"]
        or Path(value["scratch_path"]).exists()
    ):
        raise ValueError("durable_coverage_reduction")
    rebuilt = reduce(
        work,
        value["validation_receipts"],
        bool(measured),
        bool(value["cold_replay_rows"]) and all(r["passed"] for r in value["cold_replay_rows"]),
    )
    rebuilt.pop("component_primitive_reference")
    if any(value[k] != v for k, v in rebuilt.items()):
        raise ValueError("primitive_readiness_drift")
    refs = work["refs"]
    with TemporaryDirectory(prefix="carnot8290-replay-") as directory:
        if refs[0]["exists"]:
            paths = [
                Path(r.get("snapshot_path", Path(directory) / f"absent-{i}"))
                for i, r in enumerate(refs[:3])
            ]
            actual = assess(paths, Path(directory))
            for key in ["activated", "planning_matched", "canonical_tasks_sha256", "contract_rows"]:
                if actual[key] != work["contract"][key]:
                    raise ValueError("authority_reduction_drift")
            if parse_design(paths[0].read_text(), milestone=MILESTONE)[1] != work["tasks"]:
                raise ValueError("full_task_primitive_drift")
    for row in work["history"]:
        ref = next(r for r in refs if r["path"] == row["path"])
        source = json.loads(Path(ref["snapshot_path"]).read_bytes()) if ref["exists"] else {}
        if row.get("honest_verdict") != source.get("honest_verdict") or row["source_counts"] != {
            k: source.get(k) for k in row["source_counts"]
        }:
            raise ValueError("historical_primitive_drift")
    if not prior.protocol.replay(Path(value["component_primitive_reference"]["path"])):
        raise ValueError("protocol_primitive_drift")
    if (
        value["MODEL_SPECS"]
        or value["model_invocation_counts"] != prior.protocol.ZERO_INVOCATION_COUNTS
    ):
        raise ValueError("current_model_provenance")
    return dict(
        passed=True,
        owned_statement_counts=measured,
        cuda_context_ready_score=value["cuda_context_ready_score"],
    )


def main(argv: list[str] | None = None) -> int:
    """Dispatch only diagnostic or existing qualified controls before publication."""
    args = list(sys.argv[1:] if argv is None else argv)
    if "--cuda-probe" in args:
        return primitive.main(args)
    if "--scripted-peer" in args or "--typed-roster" in args:
        return int(prior.protocol.main(args))
    progress("start")
    with (
        bindings(),
        patch.object(runner, "q", sys.modules[__name__]),
        patch.object(runner, "manifest", manifest),
        patch.object(runner, "replay", replay),
        patch.object(runner, "build", build),
    ):
        return int(runner.main(args))
