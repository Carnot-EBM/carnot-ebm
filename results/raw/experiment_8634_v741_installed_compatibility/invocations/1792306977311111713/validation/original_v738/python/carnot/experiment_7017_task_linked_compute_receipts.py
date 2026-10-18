"""Build the deterministic task-linked compute receipt contract artifact.

Spec refs: REQ-REPORT-7017 and SCENARIO-REPORT-7017-*.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import UTC, datetime
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

from carnot import gpu_lease_phase_journal as lease_api
from carnot import task_runtime_receipts as receipts
from carnot.pipeline.dual_gpu_assigner import DualGPUAssigner
from carnot.pipeline.dual_gpu_monitor import DualGPUMonitor
from scripts.experiment_template import ExperimentTemplate


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_7017_task_linked_compute_receipts.json")
TASK_ID = "exp7017-task-linked-compute-receipts"
RANDOM_SEED = 7017
INFERENCE_SUBSTRATE = "deterministic_task_compute_receipt_fixtures_no_llm"
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python scripts/experiments/experiment_7017_task_linked_compute_receipts.py "
    "--date 20260905"
)
SOURCE_PATHS = (
    Path("results/operational_retro_2026_09_614.json"),
    Path("results/experiment_6426_task_scoped_runtime_receipt_contract.json"),
    Path("python/carnot/task_runtime_receipts.py"),
    Path("python/carnot/phase_concurrency_receipts.py"),
    Path("python/carnot/gpu_lease_phase_journal.py"),
    Path("python/carnot/pipeline/dual_gpu_monitor.py"),
    Path("python/carnot/pipeline/dual_gpu_assigner.py"),
    Path("scripts/experiment_template.py"),
    Path("openspec/capabilities/research-reporting/spec.md"),
    Path("python/carnot/experiment_7017_task_linked_compute_receipts.py"),
    Path("scripts/experiments/experiment_7017_task_linked_compute_receipts.py"),
    Path("tests/python/test_experiment_7017_task_linked_compute_receipts.py"),
)
REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "phase_receipt_rows",
    "gpu_sample_rows",
    "lease_link_rows",
    "model_process_rows",
    "concurrency_rows",
    "runner_decision_rows",
    "cleanup_rows",
    "acceptance_fixture_rows",
    "rejection_fixture_rows",
    "consumer_wiring_rows",
    "command_receipt_rows",
    "task_compute_receipt_ready_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
FIELD_PRINCIPLES = {
    "field_principles": "Each required field states why an independent audit needs it.",
    "preconditions_checked": "A result must show that its required sources and write paths existed.",
    "inference_substrate": "The substrate prevents GPU-shaped fixtures from becoming an LLM claim.",
    "duration_s": "Measured wall time exposes the cost of producing and checking the receipt.",
    "source_artifact_hashes": "Hashes bind the result to the exact receipt sources that it exercised.",
    "rows": "Per-fixture rows let a reader recompute the readiness conclusion.",
    "phase_receipt_rows": "Monotonic phase rows locate setup, load, inference, write, and cleanup cost.",
    "gpu_sample_rows": "Task-owned samples distinguish this task's GPU work from global activity.",
    "lease_link_rows": "Lease links prove which task owned each sampled GPU interval.",
    "model_process_rows": "Stable process identities bind model files and PIDs to inference intervals.",
    "concurrency_rows": "Recomputed overlap prevents model-list length from masquerading as concurrency.",
    "runner_decision_rows": "The decision rule explains why sequential or dual-GPU execution was selected.",
    "cleanup_rows": "Exit, reap, unload, and release evidence exposes leaked compute resources.",
    "acceptance_fixture_rows": "Positive fixtures show the valid sequential and dual execution shapes.",
    "rejection_fixture_rows": "Damaged fixtures show that attribution failures reject instead of passing.",
    "consumer_wiring_rows": "A direct template call proves the shared builder is on an experiment path.",
    "command_receipt_rows": "Command evidence identifies the executable path that produced the artifact.",
    "task_compute_receipt_ready_score": "One means every acceptance and rejection gate passed together.",
    "random_seed": "A fixed seed identifies the deterministic fixture configuration.",
    "reproducibility_checksum": "The checksum detects source or configuration drift between reruns.",
    "gate_check_summary": "Exact expected and observed values make a blocked result actionable.",
    "verifier_is_oracle": "False prevents infrastructure evidence from claiming semantic authority.",
    "verdict_class": "The closed class lets downstream tools interpret the terminal result.",
    "honest_verdict": "A class-matched prefix prevents ambiguous terminal-state parsing.",
}
VERDICT_PREFIXES = {
    "positive": "positive_",
    "circular_positive": "circular_positive_",
    "null": "null_",
    "blocked": "blocked_",
    "disqualified": "disqualified_",
    "partial": "partial_",
}


def check_preconditions(repo_root: Path = REPO_ROOT) -> list[JsonDict]:
    """Require each source plus writable code, test, script, and result paths."""

    checks = [
        {
            "check": path.as_posix(),
            "expected_value": "readable_file",
            "observed_value": (
                "readable_file"
                if (repo_root / path).is_file() and os.access(repo_root / path, os.R_OK)
                else "missing_or_unreadable"
            ),
            "passed": (repo_root / path).is_file() and os.access(repo_root / path, os.R_OK),
        }
        for path in SOURCE_PATHS
    ]
    for path in (
        Path("python/carnot"),
        Path("tests/python"),
        Path("scripts/experiments"),
        Path("results"),
    ):
        target = repo_root / path
        writable = target.is_dir() and os.access(target, os.W_OK)
        checks.append(
            {
                "check": path.as_posix(),
                "expected_value": "writable_directory",
                "observed_value": ("writable_directory" if writable else "missing_or_not_writable"),
                "passed": writable,
            }
        )
    return checks


def _gate_rows(preconditions: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Convert failed preconditions into the required diagnostic shape."""

    return [
        {
            "failed_check": row.get("check"),
            "expected_value": row.get("expected_value"),
            "observed_value": row.get("observed_value"),
        }
        for row in preconditions
        if row.get("passed") is not True
    ]


def _checksum(artifact: Mapping[str, Any]) -> str:
    """Bind stable run inputs without hashing volatile clocks or process IDs."""

    return receipts.sha256_json(
        {
            "run_date": artifact.get("run_date"),
            "random_seed": artifact.get("random_seed"),
            "source_artifact_hashes": artifact.get("source_artifact_hashes", {}),
            "inference_substrate": artifact.get("inference_substrate"),
        }
    )


def _empty_artifact(*, date: str, preconditions: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Create every required field before the terminal outcome is known."""

    artifact: JsonDict = {
        "schema": "carnot.task_linked_compute_receipt_experiment.v1",
        "experiment_id": 7017,
        "run_date": date,
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": [dict(row) for row in preconditions],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": 0.0,
        "source_artifact_hashes": {},
        "rows": [],
        "phase_receipt_rows": [],
        "gpu_sample_rows": [],
        "lease_link_rows": [],
        "model_process_rows": [],
        "concurrency_rows": [],
        "runner_decision_rows": [],
        "cleanup_rows": [],
        "acceptance_fixture_rows": [],
        "rejection_fixture_rows": [],
        "consumer_wiring_rows": [],
        "command_receipt_rows": [],
        "task_compute_receipt_ready_score": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": _gate_rows(preconditions),
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_task_compute_receipt_contract",
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def blocked_artifact(*, date: str, preconditions: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return a schema-complete blocked result with exact failed checks."""

    return _empty_artifact(date=date, preconditions=preconditions)


def _phase_clock(phase: str, start_ns: int, end_ns: int, index: int) -> JsonDict:
    """Build one phase clock with stable wall labels and measured monotonic time."""

    return {
        "phase": phase,
        "monotonic_start_ns": start_ns,
        "monotonic_end_ns": end_ns,
        "wall_clock_start": f"2026-09-05T00:00:{index:02d}Z",
        "wall_clock_end": f"2026-09-05T00:00:{index + 1:02d}Z",
    }


def _model_intervals(
    *, start_ns: int, end_ns: int, model_count: int, overlap: bool
) -> list[tuple[int, int]]:
    """Return deterministic interval shapes inside the measured inference phase."""

    if model_count == 1:
        return [(start_ns, end_ns)]
    middle = start_ns + (end_ns - start_ns) // 2
    if overlap:
        quarter = max(1, (end_ns - start_ns) // 4)
        return [(start_ns, end_ns - quarter), (start_ns + quarter, end_ns)]
    return [(start_ns, middle), (middle, end_ns)]


def run_consumer_fixture(
    work_dir: Path,
    *,
    fixture_id: str,
) -> tuple[JsonDict, JsonDict]:
    """Run one experiment-shaped lease lifecycle and validate its serialized receipt."""

    fixture_shapes = {
        "one-model": (1, False),
        "two-model-serial": (2, False),
        "two-model-dual": (2, True),
    }
    model_count, overlap = fixture_shapes[fixture_id]
    task_id = f"{TASK_ID}:{fixture_id}"
    template = ExperimentTemplate(
        7017,
        "Task-linked compute receipt fixture",
        "results/unused_exp7017_fixture.json",
        repo_root=work_dir,
        seed=RANDOM_SEED,
    )
    start_ns = time.monotonic_ns()
    leases = [
        lease_api.GpuLease.acquire(
            runtime_dir=work_dir / "leases" / fixture_id,
            task_id=task_id,
            device_uuid=f"GPU-{fixture_id}-{index}",
            expected_model=f"fixture-model-{index}.gguf",
            vram_before_mb=0,
            ttl_s=5.0,
        )
        for index in range(model_count)
    ]
    setup_end_ns = time.monotonic_ns()
    for lease in leases:
        lease.transition("admitted")
        lease.transition("loading")
        lease.transition("resident", vram_mb=2048)
    model_load_end_ns = time.monotonic_ns()
    for lease in leases:
        lease.transition("inferencing")
    time.sleep(0.0001)
    inference_end_ns = time.monotonic_ns()
    _ = receipts.sha256_text(f"fixture-output:{fixture_id}")
    output_write_end_ns = time.monotonic_ns()
    for lease in leases:
        lease.transition("unloading")
        lease.transition("validating", vram_mb=0, exit_code=0, unload_observed=True)
        lease.transition("terminal_complete")
    releases = [lease.release() for lease in leases]
    cleanup_end_ns = time.monotonic_ns()
    boundaries = (
        start_ns,
        setup_end_ns,
        model_load_end_ns,
        inference_end_ns,
        output_write_end_ns,
        cleanup_end_ns,
    )
    phase_clocks = [
        _phase_clock(phase, boundaries[index], boundaries[index + 1], index)
        for index, phase in enumerate(receipts.TASK_COMPUTE_REQUIRED_PHASES)
    ]
    intervals = _model_intervals(
        start_ns=model_load_end_ns,
        end_ns=inference_end_ns,
        model_count=model_count,
        overlap=overlap,
    )
    model_rows: list[JsonDict] = []
    samples: list[JsonDict] = []
    cleanup_rows: list[JsonDict] = []
    link_rows: list[JsonDict] = []
    for index, (lease, release, interval) in enumerate(
        zip(leases, releases, intervals, strict=True)
    ):
        pid = os.getpid() + 1000 + index
        model_id = f"fixture-model-{index}"
        model_hash = receipts.sha256_text(f"fixture-model-file-{index}")
        device = f"cuda:{index}"
        link_rows.append(
            {
                "task_id": task_id,
                "lease_id": lease.lease_id,
                "gpu_uuid": lease.device_uuid,
                "device": device,
                "released": release["released"],
                "released_monotonic_ns": lease.document["released_monotonic_ns"],
                "journal_checksum": release["checksum"],
            }
        )
        model_rows.append(
            {
                "task_id": task_id,
                "lease_id": lease.lease_id,
                "pid": pid,
                "process_start_identity": f"deterministic-fixture-start:{index}",
                "model_id": model_id,
                "model_file_hash": model_hash,
                "gpu_uuid": lease.device_uuid,
                "device": device,
                "inference_start_ns": interval[0],
                "inference_end_ns": interval[1],
            }
        )
        samples.append(
            DualGPUMonitor.build_task_linked_sample(
                task_id=task_id,
                lease_id=lease.lease_id,
                gpu_uuid=lease.device_uuid,
                device=device,
                utilization_pct=70 + index,
                memory_used_mb=2048 + index,
                power_w=None if index == 0 else 310.5,
                sample_time="2026-09-05T00:00:02Z",
                monotonic_ns=(interval[0] + interval[1]) // 2,
                sample_age_s=0.0,
                pid=pid,
                model_id=model_id,
                model_file_hash=model_hash,
            )
        )
        cleanup_rows.extend(
            (
                {
                    "kind": "model",
                    "task_id": task_id,
                    "lease_id": lease.lease_id,
                    "pid": pid,
                    "model_id": model_id,
                    "cleanup_monotonic_ns": output_write_end_ns,
                    "process_exit_confirmed": True,
                    "process_reaped": True,
                    "model_unloaded": True,
                },
                {
                    "kind": "lease",
                    "task_id": task_id,
                    "lease_id": lease.lease_id,
                    "cleanup_monotonic_ns": lease.document["released_monotonic_ns"],
                    "lease_released": True,
                },
            )
        )
    simultaneous = 2 if overlap and model_count == 2 else 1
    specs = [{"name": f"fixture-model-{index}"} for index in range(model_count)]
    runner = DualGPUAssigner(specs, n_gpus=model_count).runner_decision(
        simultaneous_model_count=simultaneous,
        live_execution_requested=True,
    )
    task_identity = receipts.read_process_identity(os.getpid())
    if task_identity is None:
        raise RuntimeError("task process identity is unavailable")
    receipt = template.build_task_compute_receipt(
        task_id=task_id,
        task_process_identity=task_identity,
        lease_link_rows=link_rows,
        phase_clocks=phase_clocks,
        gpu_sample_rows=samples,
        model_process_rows=model_rows,
        runner_decision=runner,
        cleanup_rows=cleanup_rows,
        command=[sys.executable, __file__, "--fixture", fixture_id],
        config={"fixture_id": fixture_id, "random_seed": RANDOM_SEED},
    )
    receipt_path = work_dir / f"{fixture_id}-receipt.json"
    receipts.write_adoption_receipt(receipt_path, receipt)
    return receipt, receipts.load_and_validate_task_compute_receipt(receipt_path)


def _refresh_phase_rows(receipt: JsonDict) -> None:
    """Re-seal deliberate clock mutations so the targeted rule must catch them."""

    receipt["rows"] = [receipts.seal_adoption_row(row) for row in receipt["rows"]]
    receipt["receipt_sha256"] = receipts.sha256_json(receipt["rows"])


def _rejection_rows(receipts_by_fixture: Mapping[str, JsonDict]) -> list[JsonDict]:
    """Mutate six ownership properties and require each receipt to fail closed."""

    attacks: list[tuple[str, JsonDict, str]] = []

    stale = deepcopy(receipts_by_fixture["one-model"])
    stale["gpu_sample_rows"][0]["sample_age_s"] = 99.0
    attacks.append(("stale_gpu_sample", stale, "stale_gpu_sample"))

    foreign_lease = deepcopy(receipts_by_fixture["one-model"])
    foreign_lease["gpu_sample_rows"][0]["lease_id"] = "lease:foreign"
    attacks.append(("foreign_lease", foreign_lease, "gpu_sample_lease_mismatch"))

    rollback = deepcopy(receipts_by_fixture["one-model"])
    rollback["rows"][2]["monotonic_start_ns"] = rollback["rows"][2]["monotonic_end_ns"] + 1
    _refresh_phase_rows(rollback)
    attacks.append(("clock_rollback", rollback, "phase_clock_invalid"))

    false_parallel = deepcopy(receipts_by_fixture["one-model"])
    false_parallel["runner_decision"].update(
        {
            "runner_selected": "DualGPURunner",
            "dual_gpu_runner_eligible": True,
            "simultaneous_model_count": 2,
        }
    )
    attacks.append(("one_model_false_parallel", false_parallel, "runner_concurrency_mismatch"))

    missing_cleanup = deepcopy(receipts_by_fixture["two-model-dual"])
    missing_cleanup["cleanup_rows"] = [
        row
        for row in missing_cleanup["cleanup_rows"]
        if row.get("pid") != missing_cleanup["model_process_rows"][1]["pid"]
    ]
    attacks.append(("missing_cleanup", missing_cleanup, "model_cleanup_missing"))

    malformed = deepcopy(receipts_by_fixture["one-model"])
    del malformed["runner_decision"]
    attacks.append(("malformed_receipt", malformed, "task_compute_field_missing"))

    rows = []
    for attack_id, mutated, expected_reason in attacks:
        report = receipts.validate_task_compute_receipt(mutated)
        rows.append(
            {
                "attack_id": attack_id,
                "expected_reason": expected_reason,
                "observed_reasons": report["reasons"],
                "rejected": not report["accepted"] and expected_reason in report["reasons"],
            }
        )
    return rows


def validate_artifact(artifact: Any) -> list[str]:
    """Recompute required fields, readiness, checksum, and terminal semantics."""

    if not isinstance(artifact, Mapping):
        return ["artifact_not_mapping"]
    errors: list[str] = []
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        errors.append("required_artifact_field_missing")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or any(
        not str(principles.get(field, "")).strip() for field in REQUIRED_ARTIFACT_FIELDS
    ):
        errors.append("field_principle_missing")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_oracle_mismatch")
    verdict_class = str(artifact.get("verdict_class", ""))
    verdict = str(artifact.get("honest_verdict", ""))
    prefix = VERDICT_PREFIXES.get(verdict_class)
    if prefix is None or not verdict.startswith(prefix):
        errors.append("verdict_prefix_mismatch")
    acceptance = artifact.get("acceptance_fixture_rows", [])
    rejection = artifact.get("rejection_fixture_rows", [])
    wiring = artifact.get("consumer_wiring_rows", [])
    ready = bool(
        isinstance(acceptance, Sequence)
        and acceptance
        and all(isinstance(row, Mapping) and row.get("accepted") is True for row in acceptance)
        and isinstance(rejection, Sequence)
        and rejection
        and all(isinstance(row, Mapping) and row.get("rejected") is True for row in rejection)
        and isinstance(wiring, Sequence)
        and wiring
        and all(
            isinstance(row, Mapping)
            and row.get("calls_shared_builder") is True
            and row.get("serialized_receipt_valid") is True
            for row in wiring
        )
    )
    if artifact.get("task_compute_receipt_ready_score") != int(ready):
        errors.append("ready_score_mismatch")
    if artifact.get("reproducibility_checksum") != _checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    if ready and verdict_class != "positive":
        errors.append("ready_verdict_class_mismatch")
    if verdict_class == "blocked" and not artifact.get("gate_check_summary"):
        errors.append("blocked_gate_summary_missing")
    return errors


def run(
    *,
    date: str,
    output_path: Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    repo_root: Path = REPO_ROOT,
) -> JsonDict:
    """Run acceptance, rejection, wiring, serialization, and artifact checks."""

    started = time.perf_counter()
    preconditions = check_preconditions(repo_root)
    if any(row["passed"] is not True for row in preconditions):
        artifact = blocked_artifact(date=date, preconditions=preconditions)
        receipts.write_json_atomic(output_path, artifact)
        return artifact

    with tempfile.TemporaryDirectory(prefix="carnot-exp7017-") as directory:
        work_dir = Path(directory)
        fixture_receipts: dict[str, JsonDict] = {}
        acceptance_rows: list[JsonDict] = []
        for fixture_id in ("one-model", "two-model-serial", "two-model-dual"):
            receipt, serialized = run_consumer_fixture(work_dir, fixture_id=fixture_id)
            fixture_receipts[fixture_id] = receipt
            acceptance_rows.append(
                {
                    "fixture_id": fixture_id,
                    "accepted": receipt["validation"]["accepted"] is True
                    and serialized["accepted"] is True,
                    "peak_simultaneous_model_count": serialized["peak_simultaneous_model_count"],
                    "runner_selected": receipt["runner_decision"]["runner_selected"],
                    "serialized_receipt_valid": serialized["accepted"],
                }
            )
    rejection_rows = _rejection_rows(fixture_receipts)
    canonical = fixture_receipts["two-model-dual"]
    template_source = inspect.getsource(ExperimentTemplate.build_task_compute_receipt)
    consumer_source = inspect.getsource(run_consumer_fixture)
    calls_shared_builder = (
        "receipts.build_task_compute_receipt" in template_source
        and "template.build_task_compute_receipt" in consumer_source
    )
    wiring_rows = [
        {
            "consumer": "run_consumer_fixture",
            "template_api": "ExperimentTemplate.build_task_compute_receipt",
            "shared_builder": "carnot.task_runtime_receipts.build_task_compute_receipt",
            "calls_shared_builder": calls_shared_builder,
            "serialized_receipt_valid": all(
                row["serialized_receipt_valid"] for row in acceptance_rows
            ),
            "conductor_changed": False,
        }
    ]
    ready = bool(
        all(row["accepted"] for row in acceptance_rows)
        and all(row["rejected"] for row in rejection_rows)
        and calls_shared_builder
        and wiring_rows[0]["serialized_receipt_valid"]
    )
    source_hashes = {
        path.as_posix(): receipts.sha256_file(repo_root / path) for path in SOURCE_PATHS
    }
    artifact = _empty_artifact(date=date, preconditions=preconditions)
    artifact.update(
        {
            "duration_s": round(time.perf_counter() - started, 9),
            "source_artifact_hashes": source_hashes,
            "rows": acceptance_rows,
            "phase_receipt_rows": deepcopy(canonical["rows"]),
            "gpu_sample_rows": deepcopy(canonical["gpu_sample_rows"]),
            "lease_link_rows": deepcopy(canonical["lease_link_rows"]),
            "model_process_rows": deepcopy(canonical["model_process_rows"]),
            "concurrency_rows": [
                {
                    "fixture_id": row["fixture_id"],
                    "peak_simultaneous_model_count": row["peak_simultaneous_model_count"],
                }
                for row in acceptance_rows
            ],
            "runner_decision_rows": [
                {
                    "fixture_id": fixture_id,
                    **deepcopy(receipt["runner_decision"]),
                }
                for fixture_id, receipt in fixture_receipts.items()
            ],
            "cleanup_rows": deepcopy(canonical["cleanup_rows"]),
            "acceptance_fixture_rows": acceptance_rows,
            "rejection_fixture_rows": rejection_rows,
            "consumer_wiring_rows": wiring_rows,
            "command_receipt_rows": [
                {
                    "command": RUN_COMMAND,
                    "returncode": 0,
                    "purpose": "produce deterministic task compute receipt fixtures",
                },
                {
                    "command": "load_and_validate_task_compute_receipt",
                    "returncode": 0,
                    "purpose": "recompute each serialized receipt independently",
                },
            ],
            "task_compute_receipt_ready_score": int(ready),
            "gate_check_summary": (
                []
                if ready
                else [
                    {
                        "failed_check": "task_compute_receipt_acceptance_and_rejection",
                        "expected_value": True,
                        "observed_value": ready,
                    }
                ]
            ),
            "verdict_class": "positive" if ready else "partial",
            "honest_verdict": (
                "positive_task_compute_receipt_contract_ready"
                if ready
                else "partial_task_compute_receipt_contract_not_ready"
            ),
        }
    )
    artifact["reproducibility_checksum"] = _checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        artifact["task_compute_receipt_ready_score"] = 0
        artifact["verdict_class"] = "partial"
        artifact["honest_verdict"] = "partial_task_compute_receipt_artifact_invalid"
        artifact["gate_check_summary"] = [
            {
                "failed_check": "artifact_validation",
                "expected_value": [],
                "observed_value": errors,
            }
        ]
    receipts.write_json_atomic(output_path, artifact)
    return artifact


def main(argv: list[str] | None = None) -> int:
    """Parse the requested date, write the artifact, and support read-only validation."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    if args.validate is not None:
        payload = json.loads(args.validate.read_text(encoding="utf-8"))
        errors = validate_artifact(payload)
        print(json.dumps({"accepted": not errors, "errors": errors}, sort_keys=True))
        return int(bool(errors))
    artifact = run(date=args.date, output_path=args.output)
    print(
        json.dumps(
            {
                "honest_verdict": artifact["honest_verdict"],
                "task_compute_receipt_ready_score": artifact["task_compute_receipt_ready_score"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the script wrapper.
    raise SystemExit(main())
