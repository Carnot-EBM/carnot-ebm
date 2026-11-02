"""Configure existing qualification callables for REQ-REPORT-7916-V687.

An isolated library namespace gives the current producer its own configuration.
The historical module and its checkpoint authority keep their original bytes.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import time
from types import ModuleType
from typing import Any

from carnot.reporting.current_work_receipt import build_current_work_receipt
from carnot.verify import training_qualification_7904 as prior

ROOT = prior.ROOT
MODULE = "python/carnot/verify/training_qualification_7916.py"
CLI = "scripts/experiments/experiment_7916_v687_training_qualification.py"
OWNED = (MODULE, CLI, *prior.OWNED)
INCLUDES = ",".join("*/" + name for name in OWNED)
FIXTURE_DATE = "20260929"
HISTORY = ROOT / "results/experiment_7904_v686_training_qualification.json"


def historical() -> dict[str, Any]:
    """Authenticate the old failures so passing today cannot erase yesterday's evidence."""
    if not HISTORY.is_file():
        raise prior.custody.InputBlocked(
            [prior.custody.operand(HISTORY, "artifact", "exists", True, None)]
        )
    artifact = json.loads(HISTORY.read_text())
    failures = [row for row in artifact["validation_receipts"] if not row["passed"]]
    for row in failures:
        log = Path(row["log_path"])
        actual = prior.sha256_file(log) if log.is_file() else None
        if actual != row["log_sha256"]:
            operand = prior.custody.operand(
                HISTORY, "historical_required_log", "==", row["log_sha256"], actual
            )
            operand["upstream_id"] = "exp7904-training-qualification"
            raise prior.custody.InputBlocked([operand])
    return {
        "historical_required_failures": [
            *artifact["historical_required_failures"],
            *[{"experiment_id": 7904, **row} for row in failures],
        ],
        "repository_health": {
            **artifact["repository_health"],
            "historical_exp7904_verdict": artifact["honest_verdict"],
            "historical_exp7904_path": str(HISTORY),
            "historical_exp7904_sha256": prior.sha256_file(HISTORY),
            "historical_exp7904_covered_statements": sum(
                row["covered_lines"] for row in artifact["coverage_statement_counts"].values()
            ),
            "historical_exp7904_coverage": artifact["coverage_statement_counts"],
            "retire_if_same_verdict": True,
        },
    }


def runtime() -> ModuleType:
    """Reuse the library in a private namespace instead of modifying its shared globals."""
    spec = importlib.util.spec_from_file_location(
        "carnot.verify._training_qualification_7916", ROOT / prior.MODULE
    )
    assert spec is not None and spec.loader is not None
    library = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(library)
    library.MODULE, library.CLI, library.OWNED = MODULE, CLI, OWNED
    library.INCLUDES = INCLUDES
    library.DEPENDENCIES = tuple(
        dict.fromkeys(
            (
                *prior.DEPENDENCIES,
                MODULE,
                CLI,
                "python/carnot/reporting/current_work_receipt.py",
                "python/carnot/reporting/experiment_7303_validation_scope.py",
                "scripts/adversarial_verify.py",
                "scripts/verdict_row_consistency_lint.py",
                "scripts/check_spec_coverage.py",
            )
        )
    )
    library.TESTS = (
        "tests/python/test_training_qualification_7916.py",
        "tests/python/test_experiment_7868_v683_intervention_protocol.py",
        *prior.TESTS,
    )
    original_base, original_freeze, original_publish = library.base, library.freeze, library.publish
    original_execute = library.execute
    started_ns = time.monotonic_ns()

    def progress(phase: str, units: int = 0) -> None:
        """Show the executing producer and real elapsed work at each callable boundary."""
        elapsed = (time.monotonic_ns() - started_ns) / 1e9
        print(f"[exp7916] phase={phase} elapsed_s={elapsed:.3f} completed={units}", flush=True)

    def execute(
        commands: list[prior.CommandSpec],
        raw: Path,
        heartbeat_s: float = 30,
        extra_env: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """Measure child entrypoints with the same includes as their owning unit run."""
        return list(
            original_execute(
                commands,
                raw,
                heartbeat_s,
                {
                    **(extra_env or {}),
                    "CARNOT7916_COVERAGE": "1",
                },
            )
        )

    def base(identity: int, date: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
        """Assign evidence to the current producer before any candidate is written."""
        result: dict[str, Any] = original_base(identity, date, sources)
        result.update(
            {
                "experiment_id": 7916,
                "task_id": "exp7916-training-qualification",
                "milestone": "2026.09.687",
                "run_date": date,
                "execution_date": date,
                "historical_fixture_date": FIXTURE_DATE,
                "inference_substrate": "aggregation_from_upstream_artifacts",
                "claim_scope": "private_fixture_oracle_agreement; natural_cohorts=exposed_development",
                "methodology": "Current callable fixture qualification only. Fresh private CPU heads test mechanics. Natural training belongs to Exp7918.",
            }
        )
        return result

    def freeze(private: Path, raw: Path, date: str) -> tuple[Path, list[prior.CommandSpec]]:
        """Keep the fixture's fixed date separate from the date of current execution."""
        manifest, commands = original_freeze(private, raw, FIXTURE_DATE)
        frozen = json.loads(manifest.read_text())
        frozen.update({"execution_date": date, "historical_fixture_date": FIXTURE_DATE})
        prior.atomic_json(manifest, frozen)
        return Path(manifest), list(commands)

    def publish(
        result: dict[str, Any], output: Path, raw: Path, recheck: Path | None = None
    ) -> bool:
        """Seal current timing and primitive compatibility rows before terminal readers run."""
        ended_ns = time.monotonic_ns()
        duration = (ended_ns - started_ns) / 1e9
        result["preconditions_checked"]["natural_fitting_deferred_to"] = 7918
        result["phase_spans"] = [
            {
                "phase": "current_qualification",
                "start_s": 0.0,
                "end_s": duration,
                "duration_s": duration,
            }
        ]
        result["duration_s"] = duration
        result["current_work_receipt"] = build_current_work_receipt(
            run_id="exp7916-training-qualification",
            owner_pid=library.os.getpid(),
            events=[],
            inference_substrate=result["inference_substrate"],
            inference_substrate_details={
                "private_cpu_heads": result["trained_head_specs"],
                "natural_training": False,
            },
            inference_substrate_class="no_model_load",
            execution_venue="host",
            started_monotonic_ns=started_ns,
            ended_monotonic_ns=ended_ns,
            phase_spans=result["phase_spans"],
            small_ebm_training={
                "performed": bool(result["trained_head_specs"]),
                "scope": "private_cpu_fixture",
            },
        )
        for key in result:
            result["field_principles"].setdefault(
                key, "Keep current evidence bound to its producer, exact bytes and measured scope."
            )
        return bool(original_publish(result, output, raw, recheck))

    library.base, library.freeze, library.publish, library.history = (
        base,
        freeze,
        publish,
        historical,
    )
    library.progress, library.execute = progress, execute
    return library
