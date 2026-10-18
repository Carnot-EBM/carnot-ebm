"""Exp 2844 LoopUS-style external recurrence for FR-11.

The pilot treats Carnot energy feedback as an outside loop controller: generate
an answer, score it, feed localized violations into the next attempt, and stop
when the answer passes or energy stops improving. The code never mutates model
weights. If a live recurrence backend is not attached, it writes a blocked
artifact instead of inventing energy or correctness deltas.

Spec: REQ-LEARN-2844,
      SCENARIO-LEARN-2844,
      SCENARIO-LEARN-2844-BLOCKED.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


OUTPUT_FILENAME = "experiment_2844_loopus_fr11_self_learning_pilot.json"
EXP2836_FILENAME = "experiment_2836_sota_runtime_preflight.json"
PRIMARY_SOTA_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
LEGACY_CPU_SMOKE_ONLY = ("Qwen3.5-0.8B", "gemma-4-E4B-it")
REPO_ROOT = Path(__file__).resolve().parents[3]

FIELD_PRINCIPLES = {
    "honest_verdict": 'MUST start with "complete:" / "success:" or "blocked_".',
    "continuous_self_learning_task": "Satisfies research-program FR-11 milestone mandate.",
    "n_examples": "Pilot sample-size transparency; blocked runs report 0 measured examples.",
    "mean_energy_delta_loop0_to_final": "Measures energy descent; blocked runs use 0.0 sentinel.",
    "correctness_delta": "Measures whether recurrence improves outcomes; 0.0 sentinel when blocked.",
    "early_exit_rate": "LoopUS-style adaptive stopping signal.",
    "per_example_trace": "Audit trail for self-learning behavior.",
    "model_specs": "Mandated SOTA GGUF recorded.",
    "preconditions_checked": "Explains blocks honestly.",
    "duration_s": "Real compute wall-time; no sleep padding.",
}


@dataclass(frozen=True)
class PreconditionCheck:
    """One prerequisite checked before any recurrence measurement."""

    resource: str
    available: bool
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {"resource": self.resource, "available": self.available, "detail": self.detail}


@dataclass(frozen=True)
class PilotExample:
    """One FoVer or MBPP item selected for the recurrence pilot."""

    corpus: str
    example_id: str
    prompt: str
    reference: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class GenerationResult:
    """Generated answer plus its local token cost estimate."""

    text: str
    token_cost: int


@dataclass(frozen=True)
class CandidateScore:
    """Carnot energy score and localized violation feedback for one answer."""

    energy: float
    correct: bool
    localized_violations: list[str]


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime knobs for Exp 2844."""

    repo_root: Path = REPO_ROOT
    results_dir: Path | None = None
    exp2836_path: Path | None = None
    run_date: str = "20260522"
    random_seed: int = 20260522
    n_fover: int = 25
    n_mbpp: int = 25
    max_loops: int = 3
    convergence_threshold: float = 0.01
    started_at: float | None = None
    clock: Callable[[], float] = time.time

    def output_dir(self) -> Path:
        return self.results_dir if self.results_dir is not None else self.repo_root / "results"

    def preflight_path(self) -> Path:
        return self.exp2836_path or self.output_dir() / EXP2836_FILENAME

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    @property
    def requested_n_examples(self) -> int:
        return self.n_fover + self.n_mbpp


class LiveRecurrenceBackendUnavailable(RuntimeError):
    """Raised when no real generator/scorer backend is attached for Exp 2844."""


def load_exp2836_preflight(path: Path) -> dict[str, Any]:
    """Read Exp 2836 preflight evidence when present."""

    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def model_specs_from_exp2836(preflight: dict[str, Any]) -> dict[str, object]:
    """Normalize Exp 2836's selected Python and SOTA GGUF path for Exp 2844."""

    smoke_rows = [
        dict(row)
        for row in preflight.get("smoke_load_results", [])
        if row.get("load_success") and row.get("headline_usable") and row.get("model_path")
    ]
    selected = smoke_rows[0] if smoke_rows else {}
    raw_specs = dict(preflight.get("model_specs") or {})
    return {
        "headline_required_any_of": list(raw_specs.get("primary") or PRIMARY_SOTA_MODEL_IDS),
        "legacy_cpu_smoke_only": list(
            raw_specs.get("legacy_cpu_smoke_only") or LEGACY_CPU_SMOKE_ONLY
        ),
        "sota_runtime_ready": bool(preflight.get("sota_runtime_ready")),
        "selected_python": preflight.get("selected_python"),
        "selected_model_path": selected.get("model_path"),
        "selected_model_hf_id": selected.get("hf_id"),
        "no_model_weight_mutation": True,
    }


def _load_fover_rows(repo_root: Path) -> list[dict[str, Any]]:
    path = repo_root / "data" / "fover_corpus_v4.json"
    if path.is_file():
        rows = json.loads(path.read_text(encoding="utf-8"))
        return list(rows) if isinstance(rows, list) else []
    jsonl = repo_root / "data" / "fover_corpus.jsonl"
    if not jsonl.is_file():
        return []
    return [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line]


def _load_mbpp_rows(limit: int) -> list[dict[str, Any]]:  # pragma: no cover - host/HF dependent.
    from datasets import load_dataset

    rows = load_dataset(
        "google-research-datasets/mbpp",
        "sanitized",
        split=f"test[:{max(1, int(limit))}]",
    )
    return [dict(row) for row in rows]


def _select_rows(rows: Sequence[dict[str, Any]], n_rows: int, seed: int) -> list[dict[str, Any]]:
    if len(rows) < n_rows:
        raise ValueError(f"needed {n_rows} rows, found {len(rows)}")
    rng = random.Random(seed)
    return [rows[index] for index in rng.sample(range(len(rows)), n_rows)]


def select_mixed_examples(
    repo_root: Path,
    *,
    n_fover: int,
    n_mbpp: int,
    seed: int,
) -> list[PilotExample]:
    """Select a deterministic 25 FoVer + 25 MBPP pilot set.

    Spec: REQ-LEARN-2844-1
    """

    fover = [
        PilotExample(
            "fover",
            str(row.get("question_id", index)),
            str(row.get("step_text", "")),
            str(row.get("label", "")),
            {"label": row.get("label"), "confidence": row.get("confidence")},
        )
        for index, row in enumerate(_select_rows(_load_fover_rows(repo_root), n_fover, seed))
    ]
    mbpp = [
        PilotExample(
            "mbpp",
            str(row.get("task_id", index)),
            str(row.get("prompt", "")),
            str(row.get("code", "")),
            {"test_list": list(row.get("test_list") or [])},
        )
        for index, row in enumerate(_select_rows(_load_mbpp_rows(n_mbpp + 100), n_mbpp, seed + 1))
    ]
    mixed = [*fover, *mbpp]
    random.Random(seed + 2).shuffle(mixed)
    return mixed


def probe_preconditions(
    config: ExperimentConfig,
    model_specs: dict[str, object],
) -> list[PreconditionCheck]:
    """Check all resources needed before the live recurrence loop starts."""

    model_path = str(model_specs.get("selected_model_path") or "")
    checks = [
        PreconditionCheck(
            "exp2836_artifact",
            config.preflight_path().is_file(),
            str(config.preflight_path()) if config.preflight_path().is_file() else "missing",
        ),
        PreconditionCheck(
            "exp2836_sota_runtime_ready",
            bool(model_specs.get("sota_runtime_ready")),
            f"sota_runtime_ready={model_specs.get('sota_runtime_ready')}",
        ),
        PreconditionCheck(
            "exp2836_selected_python",
            bool(model_specs.get("selected_python")),
            str(model_specs.get("selected_python") or "missing"),
        ),
        PreconditionCheck(
            "mandated_sota_model_path",
            bool(model_path and Path(model_path).is_file()),
            model_path or "missing",
        ),
    ]
    try:
        fover_count = len(_load_fover_rows(config.repo_root))
        checks.append(
            PreconditionCheck(
                "fover_dataset",
                fover_count >= config.n_fover,
                f"rows={fover_count}; required={config.n_fover}",
            )
        )
    except Exception as exc:
        checks.append(PreconditionCheck("fover_dataset", False, f"{type(exc).__name__}: {exc}"))
    try:
        mbpp_count = len(_load_mbpp_rows(config.n_mbpp))
        checks.append(
            PreconditionCheck(
                "mbpp_dataset",
                mbpp_count >= config.n_mbpp,
                f"rows={mbpp_count}; required={config.n_mbpp}",
            )
        )
    except Exception as exc:
        checks.append(PreconditionCheck("mbpp_dataset", False, f"{type(exc).__name__}: {exc}"))
    try:
        from carnot.verify.sc_energy_verifier import SCEnergyVerifier

        verifier = SCEnergyVerifier(model_name="deterministic", hidden_dim=8)
        checks.append(PreconditionCheck("carnot_energy_feedback", True, verifier.name))
    except Exception as exc:
        checks.append(
            PreconditionCheck("carnot_energy_feedback", False, f"{type(exc).__name__}: {exc}")
        )
    return checks


def _feedback_text(violations: Sequence[str]) -> str:
    if not violations:
        return ""
    return "Repair only these localized Carnot energy violations: " + "; ".join(violations)


def _round_float(value: float) -> float:
    return round(float(value), 12)


def run_recurrence_pilot(
    examples: Sequence[PilotExample],
    *,
    generate: Callable[[PilotExample, int, str], GenerationResult],
    score: Callable[[PilotExample, str], CandidateScore],
    max_loops: int = 3,
    convergence_threshold: float = 0.01,
) -> dict[str, object]:
    """Run external recurrence and compute energy/correctness deltas.

    Spec: REQ-LEARN-2844-3, REQ-LEARN-2844-4
    """

    traces: list[dict[str, object]] = []
    initial_correct = 0
    final_correct = 0
    energy_deltas: list[float] = []
    early_exits = 0
    total_token_cost = 0

    for example in examples:
        loops: list[dict[str, object]] = []
        feedback = ""
        previous_energy: float | None = None
        exit_reason = "max_loops"
        for loop_index in range(max_loops):
            generated = generate(example, loop_index, feedback)
            scored = score(example, generated.text)
            total_token_cost += int(generated.token_cost)
            loops.append(
                {
                    "loop_index": loop_index,
                    "energy": _round_float(scored.energy),
                    "correct": bool(scored.correct),
                    "localized_feedback": list(scored.localized_violations),
                    "token_cost": int(generated.token_cost),
                }
            )
            if loop_index == 0 and scored.correct:
                initial_correct += 1
            if scored.correct:
                exit_reason = "answer_passed"
                break
            if previous_energy is not None and previous_energy - scored.energy < convergence_threshold:
                exit_reason = "energy_converged"
                break
            previous_energy = scored.energy
            feedback = _feedback_text(scored.localized_violations)

        if len(loops) < max_loops:
            early_exits += 1
        first_energy = float(loops[0]["energy"])
        final_energy = float(loops[-1]["energy"])
        final_is_correct = bool(loops[-1]["correct"])
        final_correct += int(final_is_correct)
        energy_deltas.append(first_energy - final_energy)
        traces.append(
            {
                "corpus": example.corpus,
                "example_id": example.example_id,
                "loops": loops,
                "final_correct": final_is_correct,
                "early_exit_reason": exit_reason,
                "token_cost": sum(int(loop["token_cost"]) for loop in loops),
            }
        )

    n_examples = len(examples)
    if n_examples == 0:
        return {
            "n_examples": 0,
            "mean_energy_delta_loop0_to_final": 0.0,
            "correctness_delta": 0.0,
            "early_exit_rate": 0.0,
            "per_example_trace": [],
            "total_token_cost": 0,
        }
    return {
        "n_examples": n_examples,
        "mean_energy_delta_loop0_to_final": _round_float(sum(energy_deltas) / n_examples),
        "correctness_delta": _round_float((final_correct - initial_correct) / n_examples),
        "early_exit_rate": _round_float(early_exits / n_examples),
        "per_example_trace": traces,
        "total_token_cost": total_token_cost,
    }


def default_measurement_runner(
    _config: ExperimentConfig,
    _model_specs: dict[str, object],
) -> dict[str, object]:
    """Default path blocks because no live recurrence generator is wired here."""

    raise LiveRecurrenceBackendUnavailable(
        "live SOTA GGUF recurrence backend is not configured in this process"
    )


def _blocked_verdict(checks: Sequence[PreconditionCheck]) -> str | None:
    verdicts = {
        "exp2836_artifact": "blocked_exp2836_missing",
        "exp2836_sota_runtime_ready": "blocked_sota_runtime_not_ready",
        "exp2836_selected_python": "blocked_selected_python_missing",
        "mandated_sota_model_path": "blocked_model_path",
        "fover_dataset": "blocked_fover_dataset",
        "mbpp_dataset": "blocked_mbpp_dataset",
        "carnot_energy_feedback": "blocked_verifier_ensemble",
        "live_recurrence_backend": "blocked_live_recurrence_backend",
    }
    for check in checks:
        if not check.available:
            return verdicts.get(check.resource, f"blocked_{check.resource}")
    return None


def _reproducibility_checksum(config: ExperimentConfig, model_specs: dict[str, object]) -> str:
    payload = {
        "run_date": config.run_date,
        "random_seed": config.random_seed,
        "n_fover": config.n_fover,
        "n_mbpp": config.n_mbpp,
        "max_loops": config.max_loops,
        "convergence_threshold": config.convergence_threshold,
        "model_specs": model_specs,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _base_artifact(
    *,
    config: ExperimentConfig,
    checks: Sequence[PreconditionCheck],
    model_specs: dict[str, object],
    duration_s: float,
) -> dict[str, object]:
    return {
        "artifact": "experiment_2844_loopus_fr11_self_learning_pilot",
        "schema": "carnot.loopus_fr11_self_learning_pilot.v1",
        "run_date": config.run_date,
        "random_seed": config.random_seed,
        "requested_n_examples": config.requested_n_examples,
        "max_loops": config.max_loops,
        "convergence_threshold": config.convergence_threshold,
        "continuous_self_learning_task": True,
        "model_specs": model_specs,
        "preconditions_checked": [check.as_dict() for check in checks],
        "duration_s": _round_float(duration_s),
        "reproducibility_checksum": _reproducibility_checksum(config, model_specs),
        "field_principles": FIELD_PRINCIPLES,
    }


def _blocked_artifact(
    *,
    config: ExperimentConfig,
    checks: Sequence[PreconditionCheck],
    model_specs: dict[str, object],
    duration_s: float,
) -> dict[str, object]:
    failed = [check for check in checks if not check.available]
    artifact = _base_artifact(
        config=config, checks=checks, model_specs=model_specs, duration_s=duration_s
    )
    artifact.update(
        {
            "honest_verdict": _blocked_verdict(checks) or "blocked_unknown_resource",
            "blocked_resources": [check.resource for check in failed],
            "n_examples": 0,
            "mean_energy_delta_loop0_to_final": 0.0,
            "correctness_delta": 0.0,
            "early_exit_rate": 0.0,
            "per_example_trace": [],
            "total_token_cost": 0,
            "methodology_note": (
                "Blocked before recurrence measurement. Delta fields are zero sentinels "
                "with n_examples=0, not measured improvements."
            ),
        }
    )
    return artifact


def _success_artifact(
    *,
    config: ExperimentConfig,
    checks: Sequence[PreconditionCheck],
    model_specs: dict[str, object],
    duration_s: float,
    pilot: dict[str, object],
) -> dict[str, object]:
    artifact = _base_artifact(
        config=config, checks=checks, model_specs=model_specs, duration_s=duration_s
    )
    artifact.update(
        {
            "honest_verdict": "complete: LoopUS-style FR-11 external recurrence measured",
            "n_examples": int(pilot["n_examples"]),
            "mean_energy_delta_loop0_to_final": float(
                pilot["mean_energy_delta_loop0_to_final"]
            ),
            "correctness_delta": float(pilot["correctness_delta"]),
            "early_exit_rate": float(pilot["early_exit_rate"]),
            "per_example_trace": list(pilot["per_example_trace"]),
            "total_token_cost": int(pilot.get("total_token_cost", 0)),
            "methodology_note": (
                "Recurrence used external Carnot energy feedback only; no model weights "
                "were modified."
            ),
        }
    )
    return artifact


def write_artifact(results_dir: Path, artifact: dict[str, object]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / OUTPUT_FILENAME).write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    precondition_probe: Callable[[ExperimentConfig, dict[str, object]], list[PreconditionCheck]]
    = probe_preconditions,
    measurement_runner: Callable[[ExperimentConfig, dict[str, object]], dict[str, object]]
    = default_measurement_runner,
    write: bool = True,
) -> dict[str, object]:
    """Run Exp 2844 or write an honest blocked artifact.

    Spec: REQ-LEARN-2844-2
    """

    config = config or ExperimentConfig()
    start = config.start_time()
    preflight = load_exp2836_preflight(config.preflight_path())
    model_specs = model_specs_from_exp2836(preflight)
    checks = precondition_probe(config, model_specs)
    verdict = _blocked_verdict(checks)
    if verdict is None:
        try:
            pilot = measurement_runner(config, model_specs)
            artifact = _success_artifact(
                config=config,
                checks=checks,
                model_specs=model_specs,
                duration_s=config.clock() - start,
                pilot=pilot,
            )
        except LiveRecurrenceBackendUnavailable as exc:
            checks = [*checks, PreconditionCheck("live_recurrence_backend", False, str(exc))]
            artifact = _blocked_artifact(
                config=config,
                checks=checks,
                model_specs=model_specs,
                duration_s=config.clock() - start,
            )
    else:
        artifact = _blocked_artifact(
            config=config,
            checks=checks,
            model_specs=model_specs,
            duration_s=config.clock() - start,
        )
    if write:
        write_artifact(config.output_dir(), artifact)
    return artifact
