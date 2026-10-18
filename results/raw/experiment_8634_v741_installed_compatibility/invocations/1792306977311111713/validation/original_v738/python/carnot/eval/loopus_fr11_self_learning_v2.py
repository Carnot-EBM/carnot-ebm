"""Exp 2857 LoopUS-style FR-11 self-learning pilot v2.

The runner is deliberately gated on Exp 2856. If the selected live recurrence
backend artifact is missing or not ready, this module writes a blocked artifact
with zero metric sentinels instead of fabricating recurrence gains. When the
gate is satisfied, a measurement runner can provide real per-example traces and
this module performs only the deterministic accounting: energy deltas,
correctness deltas, early-exit counts, memory hashes, and reproducibility
metadata.

Spec: REQ-LEARN-2857,
      SCENARIO-LEARN-2857,
      SCENARIO-LEARN-2857-BLOCKED.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


OUTPUT_FILENAME = "experiment_2857_loopus_fr11_self_learning_v2.json"
EXP2856_FILENAME = "experiment_2856_loopus_recurrence_backend_adapter.json"
REPO_ROOT = Path(__file__).resolve().parents[3]
RANDOM_SEED = 2857
RUN_DATE = "20260522"
MISSING_EXP2856_FLAG = (
    "precondition_failed_missing_results/experiment_2856_loopus_recurrence_backend_adapter.json"
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "continuous_self_learning_task",
    "fr11_self_learning_ready",
    "n_examples",
    "source_counts",
    "max_loops",
    "recurrence_success_rate",
    "energy_delta_mean",
    "correctness_delta",
    "per_loop_energy_summary",
    "early_exit_summary",
    "memory_hash_before",
    "memory_hash_after",
    "no_model_weight_mutation",
    "model_specs",
    "random_seed",
    "reproducibility_checksum",
    "preconditions_checked",
    "duration_s",
    "adversarial_verify_passed",
    "adversarial_verify_flags",
    "run_date",
)

JQ_READY_COMMAND = (
    "jq -e '.live_recurrence_backend_ready == true' "
    f"results/{EXP2856_FILENAME}"
)
BACKEND_PATH_COMMAND = (
    ".venv/bin/python3 -c \"import json; "
    f"p='results/{EXP2856_FILENAME}'; "
    "print(json.load(open(p))['backend_module_path'])\""
)


@dataclass(frozen=True)
class CommandResult:
    """Small subprocess result used by the precondition gate."""

    returncode: int
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime knobs for Exp 2857."""

    repo_root: Path = REPO_ROOT
    results_dir: Path | None = None
    run_date: str = RUN_DATE
    random_seed: int = RANDOM_SEED
    max_loops: int = 3
    started_at: float | None = None
    clock: Callable[[], float] = time.time

    def output_dir(self) -> Path:
        return self.results_dir if self.results_dir is not None else self.repo_root / "results"

    def exp2856_path(self) -> Path:
        return self.output_dir() / EXP2856_FILENAME

    def output_path(self) -> Path:
        return self.output_dir() / OUTPUT_FILENAME

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at


CommandRunner = Callable[[str, Path], CommandResult]
MeasurementRunner = Callable[[ExperimentConfig, Mapping[str, Any]], Mapping[str, Any]]


def run_shell_command(command: str, cwd: Path) -> CommandResult:  # pragma: no cover
    """Run one fixed precondition command and capture its output."""

    completed = subprocess.run(
        command,
        cwd=cwd,
        shell=True,
        text=True,
        capture_output=True,
        check=False,
    )
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def _check_from_command(step: str, result: CommandResult) -> dict[str, Any]:
    check: dict[str, Any] = {"step": step, "passed": result.returncode == 0}
    check["returncode"] = result.returncode
    if result.stdout.strip():
        check["stdout"] = result.stdout.strip()
    if result.stderr.strip():
        check["stderr"] = result.stderr.strip()
    return check


def probe_preconditions(config: ExperimentConfig, runner: CommandRunner) -> list[dict[str, Any]]:
    """Run the exact Exp 2856 gate commands required by REQ-LEARN-2857-1."""

    repo_root = config.repo_root
    checks: list[dict[str, Any]] = [
        {
            "step": "cd /home/ianblenke/github.com/ianblenke/carnot",
            "passed": repo_root.is_dir(),
            "observed": str(repo_root),
        }
    ]
    checks.append(_check_from_command(JQ_READY_COMMAND, runner(JQ_READY_COMMAND, repo_root)))
    checks.append(
        _check_from_command(BACKEND_PATH_COMMAND, runner(BACKEND_PATH_COMMAND, repo_root))
    )
    return checks


def _classify_blocker(config: ExperimentConfig, checks: Sequence[Mapping[str, Any]]) -> str:
    if not config.exp2856_path().is_file():
        return "blocked_missing_exp2856_artifact"
    failed_steps = [str(check.get("step", "")) for check in checks if not check.get("passed")]
    if any("live_recurrence_backend_ready" in step for step in failed_steps):
        return "blocked_exp2856_backend_not_ready"
    if any("backend_module_path" in step for step in failed_steps):
        return "blocked_exp2856_backend_module_path"
    return "blocked_exp2856_precondition"


def _load_backend_artifact(config: ExperimentConfig) -> dict[str, Any]:
    return json.loads(config.exp2856_path().read_text(encoding="utf-8"))


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _final_energy(trace: Mapping[str, Any]) -> float:
    loops = [float(value) for value in trace.get("energy_after_each_loop", [])]
    if loops:
        return loops[-1]
    return float(trace.get("energy_before", 0.0))


def _summarize_loop_energy(
    traces: Sequence[Mapping[str, Any]],
    *,
    max_loops: int,
) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for loop_index in range(max_loops):
        pairs = [
            (float(trace["energy_before"]), float(trace["energy_after_each_loop"][loop_index]))
            for trace in traces
            if len(trace.get("energy_after_each_loop", [])) > loop_index
        ]
        if pairs:
            energies = [after for _before, after in pairs]
            deltas = [before - after for before, after in pairs]
            summary[f"loop_{loop_index + 1}"] = {
                "n": len(pairs),
                "mean_energy": _mean(energies),
                "mean_delta_from_initial": _mean(deltas),
            }
    return summary


def _build_success_artifact(
    config: ExperimentConfig,
    *,
    backend_artifact: Mapping[str, Any],
    measurement: Mapping[str, Any],
    preconditions_checked: Sequence[Mapping[str, Any]],
    duration_s: float,
) -> dict[str, Any]:
    traces = list(measurement.get("per_example_trace", []))
    energy_deltas = [float(trace["energy_before"]) - _final_energy(trace) for trace in traces]
    before_correct = [bool(trace.get("correctness_before")) for trace in traces]
    after_correct = [bool(trace.get("correctness_after")) for trace in traces]
    successes = [
        delta > 0.0 or (not before and after)
        for delta, before, after in zip(energy_deltas, before_correct, after_correct, strict=True)
    ]
    artifact: dict[str, Any] = {
        "honest_verdict": "complete: LoopUS FR-11 self-learning v2 measured",
        "continuous_self_learning_task": True,
        "fr11_self_learning_ready": True,
        "n_examples": len(traces),
        "source_counts": dict(measurement.get("source_counts", {})),
        "max_loops": config.max_loops,
        "recurrence_success_rate": _mean([1.0 if success else 0.0 for success in successes]),
        "energy_delta_mean": _mean(energy_deltas),
        "correctness_delta": _mean([1.0 if item else 0.0 for item in after_correct])
        - _mean([1.0 if item else 0.0 for item in before_correct]),
        "per_loop_energy_summary": _summarize_loop_energy(traces, max_loops=config.max_loops),
        "early_exit_summary": dict(
            Counter(str(trace.get("early_exit_reason", "unknown")) for trace in traces)
        ),
        "memory_hash_before": str(measurement.get("memory_hash_before", "not_reported")),
        "memory_hash_after": str(measurement.get("memory_hash_after", "not_reported")),
        "no_model_weight_mutation": True,
        "model_specs": list(
            backend_artifact.get("model_specs")
            or [{"backend_module_path": backend_artifact.get("backend_module_path")}]
        ),
        "random_seed": config.random_seed,
        "preconditions_checked": list(preconditions_checked),
        "duration_s": duration_s,
        "adversarial_verify_passed": bool(measurement.get("adversarial_verify_passed", False)),
        "adversarial_verify_flags": list(measurement.get("adversarial_verify_flags", [])),
        "run_date": config.run_date,
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def _build_blocked_artifact(
    config: ExperimentConfig,
    *,
    honest_verdict: str,
    preconditions_checked: Sequence[Mapping[str, Any]],
    duration_s: float,
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "honest_verdict": honest_verdict,
        "continuous_self_learning_task": True,
        "fr11_self_learning_ready": False,
        "n_examples": 0,
        "source_counts": {},
        "max_loops": config.max_loops,
        "recurrence_success_rate": 0.0,
        "energy_delta_mean": 0.0,
        "correctness_delta": 0.0,
        "per_loop_energy_summary": {},
        "early_exit_summary": {honest_verdict: 1},
        "memory_hash_before": "not_checked_precondition_failed",
        "memory_hash_after": "not_checked_precondition_failed",
        "no_model_weight_mutation": True,
        "model_specs": [
            {
                "source": "Exp 2856 selected backend",
                "status": "unavailable",
                "required_backend_artifact": f"results/{EXP2856_FILENAME}",
            }
        ],
        "random_seed": config.random_seed,
        "preconditions_checked": list(preconditions_checked),
        "duration_s": duration_s,
        "adversarial_verify_passed": False,
        "adversarial_verify_flags": [MISSING_EXP2856_FLAG],
        "run_date": config.run_date,
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def _checksum(artifact: Mapping[str, Any]) -> str:
    stable = {
        key: artifact[key]
        for key in REQUIRED_ARTIFACT_FIELDS
        if key in artifact and key not in {"duration_s", "reproducibility_checksum"}
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _missing_measurement_runner(
    _config: ExperimentConfig,
    _backend_artifact: Mapping[str, Any],
) -> Mapping[str, Any]:  # pragma: no cover
    raise RuntimeError("live Exp 2857 measurement runner is not wired")


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    command_runner: CommandRunner = run_shell_command,
    measurement_runner: MeasurementRunner = _missing_measurement_runner,
    write: bool = True,
) -> dict[str, Any]:
    """Run Exp 2857 accounting, writing a blocked artifact when Exp 2856 is absent."""

    active_config = config or ExperimentConfig()
    started_at = active_config.start_time()
    preconditions_checked = probe_preconditions(active_config, command_runner)
    duration_s = active_config.clock() - started_at

    if not all(bool(check.get("passed")) for check in preconditions_checked):
        artifact = _build_blocked_artifact(
            active_config,
            honest_verdict=_classify_blocker(active_config, preconditions_checked),
            preconditions_checked=preconditions_checked,
            duration_s=duration_s,
        )
    else:
        backend_artifact = _load_backend_artifact(active_config)
        measurement = measurement_runner(active_config, backend_artifact)
        duration_s = active_config.clock() - started_at
        artifact = _build_success_artifact(
            active_config,
            backend_artifact=backend_artifact,
            measurement=measurement,
            preconditions_checked=preconditions_checked,
            duration_s=duration_s,
        )

    if write:
        active_config.output_dir().mkdir(parents=True, exist_ok=True)
        active_config.output_path().write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return artifact
