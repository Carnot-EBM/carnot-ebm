"""Build the V606 task-owned runtime receipt adoption artifact.

Spec refs: REQ-REPORT-6924 and SCENARIO-REPORT-6924-*.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

from carnot import task_runtime_receipts as receipts
from scripts.experiment_template import ExperimentTemplate


JsonDict = dict[str, Any]
TASK_ID = "exp6924-task-runtime-receipt-adoption"
RESULT_RELATIVE_PATH = Path("results/experiment_6924_task_runtime_receipt_adoption.json")
INFERENCE_SUBSTRATE = "deterministic_runtime_receipt_fixture_no_llm"
RANDOM_SEED = 6924
REPO_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "adoption_api_rows",
    "cpu_fixture_rows",
    "optional_gpu_fixture_rows",
    "task_phase_timing_rows",
    "runner_selection_rows",
    "process_lineage_rows",
    "task_gpu_telemetry_rows",
    "model_concurrency_rows",
    "server_lifecycle_rows",
    "teardown_rows",
    "fresh_process_recheck_rows",
    "forged_receipt_rejection_rows",
    "random_seed",
    "reproducibility_checksum",
    "task_runtime_receipt_adoption_ready_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": "Each required output states why later audits need it.",
    "preconditions_checked": "A complete result first proves its source and write paths exist.",
    "inference_substrate": "The substrate prevents a deterministic fixture from becoming an LLM claim.",
    "duration_s": "Measured wall time bounds the cost of the adoption check.",
    "source_artifact_hashes": "Hashes bind this result to the helper, template, and examples it audited.",
    "rows": "The existing phase rows are the canonical receipt evidence.",
    "adoption_api_rows": "The exact call pattern makes later compute-task adoption mechanical.",
    "cpu_fixture_rows": "CPU evidence keeps the infrastructure usable without CUDA.",
    "optional_gpu_fixture_rows": "GPU evidence runs only when this task owns a device.",
    "task_phase_timing_rows": "Monotonic intervals support independent ordering and duration checks.",
    "runner_selection_rows": "Runner receipts distinguish selected execution code from labels.",
    "process_lineage_rows": "Kernel identities bind child work to the task process.",
    "task_gpu_telemetry_rows": "PID and UUID samples bind VRAM, utilization, and offload to GPU work.",
    "model_concurrency_rows": "Lifecycle intervals distinguish sequential and concurrent model use.",
    "server_lifecycle_rows": "Start and teardown events expose the complete server lifetime.",
    "teardown_rows": "Exit and reap evidence prevents a successful row from hiding a leaked server.",
    "fresh_process_recheck_rows": "A separate interpreter recomputes claims from serialized rows.",
    "forged_receipt_rejection_rows": "Mutations prove critical ownership failures reject the receipt.",
    "random_seed": "A fixed seed identifies the deterministic fixture configuration.",
    "reproducibility_checksum": "The checksum binds stable inputs that should reproduce the verdict.",
    "task_runtime_receipt_adoption_ready_score": "One means the reusable CPU path passed fresh validation.",
    "gate_check_summary": "A blocked verdict names expected and observed prerequisite values.",
    "verifier_is_oracle": "False keeps this infrastructure check outside scientific authority.",
    "verdict_class": "The class separates advisory readiness from a positive science result.",
    "honest_verdict": "A terminal complete prefix makes conductor interpretation unambiguous.",
}

SOURCE_PATHS = (
    Path("python/carnot/task_runtime_receipts.py"),
    Path("scripts/experiment_template.py"),
    Path("results/experiment_6915_qualified_relation_event_bank.json"),
    Path("results/experiment_6920_sota_exact_guided_relation_generation.json"),
)


def check_preconditions(repo_root: Path = REPO_ROOT) -> list[JsonDict]:
    """Check every required source and both writable development locations."""

    checks = [
        {
            "check": path.as_posix(),
            "expected_value": "existing_file",
            "observed_value": "existing_file" if (repo_root / path).is_file() else "missing",
            "passed": (repo_root / path).is_file(),
        }
        for path in SOURCE_PATHS
    ]
    writable_paths = (
        Path("tests/python"),
        Path("openspec/capabilities/research-reporting/spec.md"),
    )
    for path in writable_paths:
        target = repo_root / path
        writable = target.exists() and os.access(target, os.W_OK)
        checks.append(
            {
                "check": path.as_posix(),
                "expected_value": "writable",
                "observed_value": "writable" if writable else "not_writable_or_missing",
                "passed": writable,
            }
        )
    return checks


def _source_hashes(repo_root: Path) -> JsonDict:
    """Hash each required source without interpreting its reported claims."""

    return {path.as_posix(): receipts.sha256_file(repo_root / path) for path in SOURCE_PATHS}


def _runner_selection() -> JsonDict:
    """Describe the exact Python runner used by the bounded CPU fixture."""

    return {
        "runner_id": "cpython-bounded-local-subprocess",
        "binary_path": sys.executable,
        "binary_sha256": receipts.sha256_file(sys.executable),
        "substrate": "cpu",
        "selected": True,
    }


def _run_cpu_fixture(receipt_path: Path) -> tuple[JsonDict, list[JsonDict]]:
    """Run one bounded child and record its full task-owned lifecycle."""

    template = ExperimentTemplate(
        6924,
        "Task-owned runtime receipt adoption fixture",
        "results/unused_exp6924_fixture.json",
        repo_root=receipt_path.parent,
        seed=RANDOM_SEED,
    )
    model_identity = {
        "model_id": "deterministic-cpu-model-fixture",
        "model_sha256": receipts.sha256_text("deterministic-cpu-model-fixture-v1"),
        "model_identity_bound": True,
    }
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(0.05); print('receipt-fixture')"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    with template.task_runtime_receipts(
        receipt_path,
        task_id=TASK_ID,
        control_id="cpu-fixture",
        runner_selection=_runner_selection(),
        model_identity=model_identity,
        device_ids=["CPU"],
        model_count=1,
        concurrency_group="exp6924-cpu-fixture",
        config={"fixture": "deterministic_cpu", "random_seed": RANDOM_SEED},
    ) as runtime:
        with runtime.phase("queue_wait"):
            pass
        with runtime.phase(
            "model_load",
            model_id=str(model_identity["model_id"]),
            child_pids=[process.pid],
            server_lifecycle={
                "event": "started",
                "server_id": "cpu-fixture-server",
                "pid": process.pid,
            },
        ):
            pass
        with runtime.phase(
            "generation",
            model_id=str(model_identity["model_id"]),
            child_pids=[process.pid],
        ) as state:
            stdout, stderr = process.communicate(timeout=2)
            state["raw_output_bytes"] = stdout
            state["exit_status"] = {
                "returncode": process.returncode,
                "timed_out": False,
                "signal": None,
                "stderr_sha256": receipts.sha256_bytes(stderr),
            }
        with runtime.phase("exact_verification", model_id=str(model_identity["model_id"])):
            pass
        with runtime.phase(
            "teardown",
            model_id=str(model_identity["model_id"]),
            server_lifecycle={
                "event": "teardown",
                "server_id": "cpu-fixture-server",
                "pid": process.pid,
                "process_exit_confirmed": process.returncode == 0,
                "process_reaped": process.poll() is not None,
                "vram_after_teardown_mb": 0,
            },
        ):
            pass
        with runtime.phase("artifact_write"):
            pass
    return json.loads(receipt_path.read_text(encoding="utf-8")), template._phase_timings


def _fresh_process_recheck(receipt_path: Path, task_pid: int) -> JsonDict:
    """Ask a new interpreter to recompute every acceptance property from disk."""

    program = (
        "import json,sys; "
        "from carnot.task_runtime_receipts import load_and_validate_adoption_receipt as check; "
        "print(json.dumps(check(sys.argv[1], expected_task_id=sys.argv[2], "
        "expected_task_pid=int(sys.argv[3])), sort_keys=True))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program, str(receipt_path), TASK_ID, str(task_pid)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if completed.returncode != 0:
        return {
            "accepted": False,
            "returncode": completed.returncode,
            "stderr": completed.stderr.strip(),
        }
    report = json.loads(completed.stdout)
    report["returncode"] = completed.returncode
    return report


def _forgery_checks(rows: list[JsonDict], task_pid: int) -> list[JsonDict]:
    """Mutate critical fields and prove each changed receipt fails closed."""

    attacks: list[tuple[str, list[JsonDict]]] = []

    pid_rows = deepcopy(rows)
    pid_rows[0]["parent_pid"] = task_pid + 1
    pid_rows[0] = receipts.seal_adoption_row(pid_rows[0])
    attacks.append(("pid_mismatch", pid_rows))

    lineage_rows = deepcopy(rows)
    generation = next(row for row in lineage_rows if row["phase"] == "generation")
    generation["process_lineage"][0]["chain"][-1]["pid"] = task_pid + 1
    generation = receipts.seal_adoption_row(generation)
    lineage_rows[lineage_rows.index(next(row for row in lineage_rows if row["phase"] == "generation"))] = generation
    attacks.append(("cross_process_lineage", lineage_rows))

    teardown_rows = [deepcopy(row) for row in rows if row["phase"] != "teardown"]
    attacks.append(("missing_teardown", teardown_rows))

    gpu_rows = [deepcopy(next(row for row in rows if row["phase"] == "generation"))]
    gpu_row = gpu_rows[0]
    child_pid = int(gpu_row["child_pids"][0])
    sample_clock = (gpu_row["monotonic_start_ns"] + gpu_row["monotonic_end_ns"]) // 2
    gpu_row["device_ids"] = ["GPU-task-owned-fixture"]
    gpu_row["gpu_samples"] = [
        {
            "pid": child_pid,
            "device_uuid": "GPU-forged-cross-device",
            "pid_memory_mb": 1024,
            "device_memory_used_mb": 2048,
            "utilization_pct": 50,
            "offload_layers": 16,
            "monotonic_ns": sample_clock,
            "sample_age_s": 0.0,
        }
    ]
    gpu_rows[0] = receipts.seal_adoption_row(gpu_row)
    attacks.append(("gpu_uuid_mismatch", gpu_rows))

    results: list[JsonDict] = []
    for attack_id, mutated in attacks:
        report = receipts.validate_adoption_rows(
            mutated,
            expected_task_id=TASK_ID,
            expected_task_pid=task_pid,
        )
        results.append(
            {
                "attack_id": attack_id,
                "rejected": not report["accepted"],
                "reasons": report["reasons"],
            }
        )
    return results


def _optional_gpu_fixture() -> list[JsonDict]:
    """Run no GPU work unless the scheduler explicitly grants a device UUID."""

    granted_uuid = os.environ.get("CARNOT_TASK_GPU_UUID", "").strip()
    if not granted_uuid:
        return [{"status": "not_run_no_task_owned_gpu", "gpu_uuid": None}]
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    visible = {line.strip() for line in completed.stdout.splitlines() if line.strip()}
    return [
        {
            "status": "complete" if completed.returncode == 0 and granted_uuid in visible else "not_run_no_task_owned_gpu",
            "gpu_uuid": granted_uuid,
            "bounded_subprocess_returncode": completed.returncode,
        }
    ]


def _gate_rows(preconditions: list[JsonDict]) -> list[JsonDict]:
    """Convert failed checks to the exact blocked-artifact shape."""

    return [
        {
            "failed_check": row["check"],
            "expected_value": row["expected_value"],
            "observed_value": row["observed_value"],
        }
        for row in preconditions
        if not row["passed"]
    ]


def _empty_artifact(date: str, preconditions: list[JsonDict]) -> JsonDict:
    """Create all required fields before success or blocked evidence is known."""

    return {
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": 0.0,
        "source_artifact_hashes": {},
        "rows": [],
        "adoption_api_rows": [],
        "cpu_fixture_rows": [],
        "optional_gpu_fixture_rows": [],
        "task_phase_timing_rows": [],
        "runner_selection_rows": [],
        "process_lineage_rows": [],
        "task_gpu_telemetry_rows": [],
        "model_concurrency_rows": [],
        "server_lifecycle_rows": [],
        "teardown_rows": [],
        "fresh_process_recheck_rows": [],
        "forged_receipt_rejection_rows": [],
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": receipts.sha256_json(
            {"date": date, "random_seed": RANDOM_SEED}
        )[7:23],
        "task_runtime_receipt_adoption_ready_score": 0,
        "gate_check_summary": _gate_rows(preconditions),
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "complete_blocked_task_runtime_receipt_adoption",
        "schema": receipts.ADOPTION_SCHEMA_VERSION,
        "experiment_id": 6924,
        "run_date": date,
        "status": "complete_blocked",
    }


def blocked_artifact(*, date: str, preconditions: list[JsonDict]) -> JsonDict:
    """Return a complete blocked artifact with explicit prerequisite mismatches."""

    return _empty_artifact(date, preconditions)


def run(
    *,
    date: str,
    output_path: Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    repo_root: Path = REPO_ROOT,
) -> JsonDict:
    """Run the CPU adoption fixture, fresh recheck, and forgery checks."""

    started = time.perf_counter()
    preconditions = check_preconditions(repo_root)
    if any(not row["passed"] for row in preconditions):
        artifact = blocked_artifact(date=date, preconditions=preconditions)
        receipts.write_json_atomic(output_path, artifact)
        return artifact

    with tempfile.TemporaryDirectory(prefix="carnot-exp6924-") as directory:
        receipt_path = Path(directory) / "cpu-runtime-receipt.json"
        payload, template_timings = _run_cpu_fixture(receipt_path)
        fresh = _fresh_process_recheck(receipt_path, os.getpid())

    rows = payload["rows"]
    forged = _forgery_checks(rows, os.getpid())
    ready = bool(
        payload["validation"]["accepted"]
        and fresh.get("accepted")
        and all(row["rejected"] for row in forged)
    )
    artifact = _empty_artifact(date, preconditions)
    source_hashes = _source_hashes(repo_root)
    artifact.update(
        {
            "duration_s": round(time.perf_counter() - started, 9),
            "source_artifact_hashes": source_hashes,
            "rows": rows,
            "adoption_api_rows": [
                {
                    "api": "ExperimentTemplate.task_runtime_receipts",
                    "call_pattern": (
                        "with tmpl.task_runtime_receipts(path, task_id=TASK_ID, "
                        "control_id=CONTROL_ID, runner_selection=runner, "
                        "model_identity=model, device_ids=device_ids, model_count=len(models)) "
                        "as runtime:\n    with runtime.phase('generation', model_id=model_id, "
                        "child_pids=[server.pid]) as phase:\n        phase['raw_output_bytes'] = output\n"
                        "        phase['exit_status'] = exit_status\n"
                        "        phase['gpu_samples'] = pid_uuid_vram_utilization_offload_samples"
                    ),
                    "teardown_pattern": (
                        "with runtime.phase('teardown', server_lifecycle={'event': 'teardown', "
                        "'server_id': server_id, 'pid': server.pid, "
                        "'process_exit_confirmed': True, 'process_reaped': True}): pass"
                    ),
                }
            ],
            "cpu_fixture_rows": deepcopy(rows),
            "optional_gpu_fixture_rows": _optional_gpu_fixture(),
            "task_phase_timing_rows": template_timings,
            "runner_selection_rows": [deepcopy(rows[0]["runner_selection"])],
            "process_lineage_rows": [
                {"phase": row["phase"], "lineage": deepcopy(row["process_lineage"])}
                for row in rows
                if row["process_lineage"]
            ],
            "task_gpu_telemetry_rows": [
                {"phase": row["phase"], **deepcopy(sample)}
                for row in rows
                for sample in row["gpu_samples"]
            ],
            "model_concurrency_rows": [
                {
                    "declared_model_count": rows[0]["model_lifecycle"]["model_count"],
                    "recomputed_peak_model_concurrency": fresh.get("peak_model_concurrency"),
                    "concurrency_group": rows[0]["concurrency_group"],
                    "mode": rows[0]["model_lifecycle"]["concurrency_mode"],
                }
            ],
            "server_lifecycle_rows": [
                deepcopy(row["server_lifecycle"])
                for row in rows
                if row["server_lifecycle"]
            ],
            "teardown_rows": [
                deepcopy(row["server_lifecycle"])
                for row in rows
                if row["server_lifecycle"].get("event") == "teardown"
            ],
            "fresh_process_recheck_rows": [fresh],
            "forged_receipt_rejection_rows": forged,
            "reproducibility_checksum": receipts.sha256_json(
                {"date": date, "random_seed": RANDOM_SEED, "source_artifact_hashes": source_hashes}
            )[7:23],
            "task_runtime_receipt_adoption_ready_score": int(ready),
            "gate_check_summary": [] if ready else [
                {
                    "failed_check": "fresh_process_runtime_receipt_validation",
                    "expected_value": True,
                    "observed_value": bool(fresh.get("accepted")),
                }
            ],
            "verdict_class": "null" if ready else "blocked",
            "honest_verdict": (
                "complete_null_task_runtime_receipt_adoption_ready"
                if ready
                else "complete_blocked_task_runtime_receipt_adoption"
            ),
            "status": "complete",
        }
    )
    receipts.write_json_atomic(output_path, artifact)
    return artifact


def main(argv: list[str] | None = None) -> int:
    """Parse the dated experiment command and write its terminal artifact."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    args = parser.parse_args(argv)
    artifact = run(date=args.date, output_path=args.output)
    print(json.dumps({"honest_verdict": artifact["honest_verdict"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
