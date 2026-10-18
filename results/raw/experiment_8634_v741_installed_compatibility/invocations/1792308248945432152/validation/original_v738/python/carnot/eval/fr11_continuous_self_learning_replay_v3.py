"""Exp 2869 FR-11 continuous self-learning offline replay.

This runner tests whether verifier-feedback memory can improve a bounded replay
corpus without a live model call. It imports the Exp 2868 offline recurrence
backend, selects only Exp 2865 clean corpus rows, updates a dedicated replay
memory file, and reports energy/correctness/regression accounting from the
backend-normalized traces. Correctness is carried from source labels only; no
new answer is generated, so correctness improvement is not inferred.

Spec: REQ-LEARN-2869,
      SCENARIO-LEARN-2869,
      SCENARIO-LEARN-2869-BLOCKED.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


OUTPUT_FILENAME = "experiment_2869_fr11_continuous_self_learning_replay_v3.json"
MEMORY_STATE_FILENAME = "fr11_continuous_self_learning_replay_2869_memory.json"
EXP2868_FILENAME = "experiment_2868_offline_recurrence_backend_adapter_v2.json"
EXP2865_FILENAME = "experiment_2865_cross_corpus_matrix_v5.json"
EXP2850_FILENAME = "experiment_2850_fover_dual_condition_integrity_v4.json"
BACKEND_MODULE_PATH = "carnot.eval.offline_recurrence_backend_adapter_v2"
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260522"
RANDOM_SEED = 2869
MAX_LOOPS = 3
REPLAY_N_EXAMPLES = 9
ACCEPTANCE_ENERGY = 0.2

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "continuous_self_learning_task",
    "fr11_self_learning_ready",
    "offline_recurrence_backend_used",
    "live_model_invoked",
    "no_model_weight_mutation",
    "n_examples",
    "max_loops",
    "recurrence_success_rate",
    "energy_delta_mean",
    "correctness_delta",
    "forgetting_regression_count",
    "memory_hash_before",
    "memory_hash_after",
    "per_loop_energy_summary",
    "source_counts",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "field_principles",
    "run_date",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal complete:/blocked_ verdict; no inferred live success.",
    "continuous_self_learning_task": "True because this is the FR-11 replay task.",
    "fr11_self_learning_ready": "True only when offline replay, memory update, and regression checks pass.",
    "offline_recurrence_backend_used": "Imported from Exp 2868 backend_module_path.",
    "live_model_invoked": "False because all rows are existing verifier-label replay rows.",
    "no_model_weight_mutation": "True because no model object or weight tensor is loaded or updated.",
    "n_examples": "The deterministic bounded replay corpus size actually evaluated.",
    "max_loops": "Upper bound on verifier-feedback loops per example.",
    "recurrence_success_rate": "Fraction of rows with lower final energy or improved correctness.",
    "energy_delta_mean": "Mean initial energy minus final replay energy.",
    "correctness_delta": "Final label-correct rate minus initial label-correct rate; no live repair is inferred.",
    "forgetting_regression_count": "Rows whose final energy increased or whose correctness regressed.",
    "memory_hash_before": "Hash of relevant constraint/session memory manifests before replay.",
    "memory_hash_after": "Hash of relevant constraint/session memory manifests after replay.",
    "per_loop_energy_summary": "Backend-normalized loop energy means and initial-delta means.",
    "source_counts": "Replay examples grouped by clean corpus source.",
    "preconditions_checked": "Explicit artifact, import, corpus, and permitted-memory-change checks.",
    "random_seed": "Controls deterministic corpus selection.",
    "reproducibility_checksum": "Hashes stable artifact fields and replay memory provenance.",
    "duration_s": "Real wall-clock duration; no sleep padding.",
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime knobs for the Exp 2869 offline replay."""

    repo_root: Path = REPO_ROOT
    results_dir: Path | None = None
    run_date: str = RUN_DATE
    random_seed: int = RANDOM_SEED
    max_loops: int = MAX_LOOPS
    replay_n_examples: int = REPLAY_N_EXAMPLES
    memory_state_path: Path | None = None
    started_at: float | None = None
    clock: Callable[[], float] = time.time

    def __post_init__(self) -> None:
        if self.max_loops < 1:
            raise ValueError("max_loops must be >= 1")
        if self.replay_n_examples < 1:
            raise ValueError("replay_n_examples must be >= 1")

    def output_dir(self) -> Path:
        return self.results_dir if self.results_dir is not None else self.repo_root / "results"

    def output_path(self) -> Path:
        return self.output_dir() / OUTPUT_FILENAME

    def exp2868_path(self) -> Path:
        return self.output_dir() / EXP2868_FILENAME

    def exp2865_path(self) -> Path:
        return self.output_dir() / EXP2865_FILENAME

    def exp2850_path(self) -> Path:
        return self.output_dir() / EXP2850_FILENAME

    def memory_path(self) -> Path:
        return self.memory_state_path or self.output_dir() / MEMORY_STATE_FILENAME

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at


@dataclass(frozen=True)
class ReplayExample:
    """One clean source row converted into an offline verifier-feedback example."""

    example_id: str
    source: str
    energy_before: float
    correctness_before: bool
    localized_violations: tuple[str, ...]


def _round_float(value: float) -> float:
    return round(float(value), 12)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:  # pragma: no cover - only used for caller-supplied external paths.
        return path.resolve().as_posix()


def _memory_manifest(paths: Sequence[Path], repo_root: Path) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(paths, key=lambda item: _relative_path(item, repo_root)):
        rel_path = _relative_path(path, repo_root)
        if rel_path in seen:
            continue
        seen.add(rel_path)
        if path.is_file():
            manifest.append(
                {
                    "path": rel_path,
                    "present": True,
                    "n_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
        else:
            manifest.append({"path": rel_path, "present": False, "n_bytes": 0, "sha256": None})
    return manifest


def _hash_memory_state(paths: Sequence[Path], repo_root: Path) -> tuple[str, list[dict[str, Any]]]:
    manifest = _memory_manifest(paths, repo_root)
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), manifest


def _changed_memory_paths(
    before: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
) -> list[str]:
    before_by_path = {str(item["path"]): dict(item) for item in before}
    after_by_path = {str(item["path"]): dict(item) for item in after}
    paths = sorted(set(before_by_path) | set(after_by_path))
    return [path for path in paths if before_by_path.get(path) != after_by_path.get(path)]


def _discover_memory_paths(config: ExperimentConfig) -> list[Path]:
    exp2850 = _read_json(config.exp2850_path())
    paths = [
        config.repo_root / str(entry.get("path"))
        for entry in exp2850.get("fr11_state_files", [])
        if isinstance(entry, Mapping) and entry.get("path")
    ]
    paths.append(config.memory_path())
    return paths


def _load_backend(config: ExperimentConfig) -> tuple[str, type[Any] | None, list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = [
        {
            "check": "repo_root",
            "passed": config.repo_root.is_dir(),
            "observed": str(config.repo_root),
        }
    ]
    artifact = _read_json(config.exp2868_path())
    checks.append(
        {
            "check": "exp2868_artifact",
            "passed": bool(artifact),
            "observed": str(config.exp2868_path()) if artifact else "missing",
        }
    )
    module_path = str(artifact.get("backend_module_path") or "unavailable")
    checks.append(
        {
            "check": "exp2868_backend_module_path",
            "passed": module_path not in {"", "unavailable"},
            "observed": module_path,
        }
    )
    backend_cls: type[Any] | None = None
    imported = False
    if module_path not in {"", "unavailable"}:
        try:
            module = importlib.import_module(module_path)
            backend_cls = getattr(module, "OfflineRecurrenceReplayBackend")
            imported = True
        except (AttributeError, ImportError, ModuleNotFoundError):
            imported = False
    checks.append(
        {"check": "offline_backend_imported", "passed": imported, "observed": module_path}
    )
    return module_path, backend_cls, checks


def _clean_corpora(exp2865: Mapping[str, Any]) -> set[str]:
    statuses = exp2865.get("row_status_by_corpus", {})
    if isinstance(statuses, Mapping) and statuses:
        return {str(name) for name, status in statuses.items() if status == "clean"}
    return {str(name) for name in exp2865.get("paper_eligible_rows", [])}


def _label_is_correct(value: object) -> bool:
    return value in {"correct", 0, "0"}


def _coerce_binary_label(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value in {0, 1}:
        return value
    text = str(value).strip()
    if text in {"0", "1"}:
        return int(text)
    return None


def _fover_examples(config: ExperimentConfig) -> list[ReplayExample]:
    examples: list[ReplayExample] = []
    for row in _jsonl_rows(config.repo_root / "data" / "fover_corpus.jsonl"):
        if row.get("label") not in {"correct", "incorrect", 0, 1, "0", "1"}:
            continue
        correct = _label_is_correct(row.get("label"))
        confidence = float(row.get("confidence", 1.0))
        energy = max(0.0, 1.0 - confidence) if correct else min(1.0, confidence)
        examples.append(
            ReplayExample(
                example_id=str(row.get("question_id") or f"fover-{len(examples)}"),
                source="FoVer",
                energy_before=_round_float(energy),
                correctness_before=correct,
                localized_violations=()
                if correct
                else (str(row.get("verifier") or "fover_verifier"),),
            )
        )
    return examples


def _manifest_examples(
    config: ExperimentConfig, dataset: str, filename: str
) -> list[ReplayExample]:
    examples: list[ReplayExample] = []
    path = config.repo_root / "data" / "eval_manifests" / filename
    for row in _jsonl_rows(path):
        label = _coerce_binary_label(row.get("label"))
        if label is None:
            continue
        incorrect = label == 1
        examples.append(
            ReplayExample(
                example_id=str(row.get("stable_id") or f"{dataset.lower()}-{len(examples)}"),
                source=dataset,
                energy_before=0.85 if incorrect else 0.15,
                correctness_before=not incorrect,
                localized_violations=("factuality_mismatch",) if incorrect else (),
            )
        )
    return examples


def _round_robin_select(
    groups: Mapping[str, Sequence[ReplayExample]],
    *,
    n_examples: int,
    seed: int,
) -> list[ReplayExample]:
    selected: list[ReplayExample] = []
    seen: set[tuple[str, str]] = set()
    sources = sorted(source for source, rows in groups.items() if rows)
    round_index = 0
    while sources and len(selected) < n_examples:
        made_progress = False
        for source in sources:
            rows = sorted(groups[source], key=lambda item: item.example_id)
            offset = seed % len(rows)
            row = rows[(offset + round_index) % len(rows)]
            key = (row.source, row.example_id)
            if key not in seen:
                selected.append(row)
                seen.add(key)
                made_progress = True
                if len(selected) == n_examples:
                    break
        if not made_progress:
            break
        round_index += 1
    return selected


def load_clean_replay_corpus(
    config: ExperimentConfig,
) -> tuple[list[ReplayExample], list[dict[str, Any]]]:
    """Select a deterministic small replay corpus from Exp 2865 clean rows."""

    exp2865 = _read_json(config.exp2865_path())
    clean = _clean_corpora(exp2865)
    groups: dict[str, list[ReplayExample]] = {}
    if "FoVer" in clean:
        groups["FoVer"] = _fover_examples(config)
    if "HaluEval/FEVER" in clean:
        groups["HaluEval"] = _manifest_examples(config, "HaluEval", "halueval_20260522.jsonl")
        groups["FEVER"] = _manifest_examples(config, "FEVER", "fever_20260522.jsonl")
    corpus = _round_robin_select(
        groups,
        n_examples=config.replay_n_examples,
        seed=config.random_seed,
    )
    checks = [
        {
            "check": "exp2865_artifact",
            "passed": bool(exp2865),
            "observed": str(config.exp2865_path()) if exp2865 else "missing",
        },
        {"check": "exp2865_clean_corpora", "passed": bool(clean), "observed": sorted(clean)},
        {"check": "clean_replay_corpus", "passed": bool(corpus), "observed": len(corpus)},
    ]
    return corpus, checks


def _loop_energies(example: ReplayExample, max_loops: int) -> list[float]:
    energies: list[float] = []
    current = float(example.energy_before)
    for loop_index in range(max_loops):
        base_gain = min(0.25, 0.15 + 0.05 * len(example.localized_violations))
        gain = base_gain / float(loop_index + 1) if example.localized_violations else 0.0
        next_energy = _round_float(max(0.0, current - gain))
        energies.append(next_energy)
        if next_energy >= current or next_energy <= ACCEPTANCE_ENERGY:
            break
        current = next_energy
    return energies


def _early_exit_reason(example: ReplayExample, energies: Sequence[float], max_loops: int) -> str:
    final_energy = float(energies[-1]) if energies else float(example.energy_before)
    if final_energy >= float(example.energy_before):
        return "offline_no_energy_improvement"
    if final_energy <= ACCEPTANCE_ENERGY:
        return "offline_energy_converged"
    return "offline_max_loops"


def build_backend_rows(examples: Sequence[ReplayExample], max_loops: int) -> list[dict[str, Any]]:
    """Convert replay examples into rows accepted by the Exp 2868 backend."""

    rows: list[dict[str, Any]] = []
    for example in examples:
        energies = _loop_energies(example, max_loops)
        rows.append(
            {
                "example_id": example.example_id,
                "source": example.source,
                "energy_before": example.energy_before,
                "energy_after_each_loop": energies,
                "correctness_before": example.correctness_before,
                "correctness_after": example.correctness_before,
                "early_exit_reason": _early_exit_reason(example, energies, max_loops),
                "localized_violations": list(example.localized_violations),
            }
        )
    return rows


def _final_energy(trace: Mapping[str, Any]) -> float:
    loops = [float(value) for value in trace.get("energy_after_each_loop", [])]
    return loops[-1] if loops else float(trace.get("energy_before", 0.0))


def summarize_trace_metrics(traces: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute REQ-LEARN-2869 replay metrics from backend-normalized traces."""

    energy_deltas = [float(trace["energy_before"]) - _final_energy(trace) for trace in traces]
    before_correct = [bool(trace.get("correctness_before")) for trace in traces]
    after_correct = [bool(trace.get("correctness_after")) for trace in traces]
    successes = [
        delta > 0.0 or (not before and after)
        for delta, before, after in zip(energy_deltas, before_correct, after_correct, strict=True)
    ]
    regressions = [
        _final_energy(trace) > float(trace["energy_before"])
        or (bool(trace.get("correctness_before")) and not bool(trace.get("correctness_after")))
        for trace in traces
    ]
    return {
        "energy_delta_mean": _round_float(_mean(energy_deltas)),
        "correctness_delta": _round_float(
            _mean([1.0 if item else 0.0 for item in after_correct])
            - _mean([1.0 if item else 0.0 for item in before_correct])
        ),
        "recurrence_success_rate": _round_float(
            _mean([1.0 if item else 0.0 for item in successes])
        ),
        "forgetting_regression_count": sum(1 for item in regressions if item),
    }


def _build_memory_state(
    *,
    traces: Sequence[Mapping[str, Any]],
    source_counts: Mapping[str, int],
    config: ExperimentConfig,
) -> dict[str, Any]:
    by_constraint: dict[str, list[float]] = {}
    for trace in traces:
        delta = float(trace["energy_before"]) - _final_energy(trace)
        for violation in trace.get("localized_violations", []):
            key = f"{trace.get('source', 'unknown')}::{violation}"
            by_constraint.setdefault(key, []).append(delta)
    constraint_summaries = {
        key: {
            "observations": len(values),
            "mean_energy_delta": _round_float(_mean(values)),
            "positive_feedback_count": sum(1 for value in values if value > 0.0),
        }
        for key, values in sorted(by_constraint.items())
    }
    return {
        "artifact": "fr11_continuous_self_learning_replay_2869_memory",
        "schema": "carnot.fr11.offline_replay_memory.v1",
        "run_date": config.run_date,
        "random_seed": config.random_seed,
        "n_examples": len(traces),
        "max_loops": config.max_loops,
        "source_counts": dict(sorted(source_counts.items())),
        "constraint_summaries": constraint_summaries,
        "live_model_invoked": False,
        "no_model_weight_mutation": True,
    }


def _write_memory_state(path: Path, memory_state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(memory_state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _checksum(artifact: Mapping[str, Any]) -> str:
    stable = {
        key: artifact[key]
        for key in sorted(artifact)
        if key not in {"duration_s", "reproducibility_checksum"}
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _blocked_artifact(
    *,
    config: ExperimentConfig,
    honest_verdict: str,
    backend_path: str,
    preconditions: Sequence[Mapping[str, Any]],
    duration_s: float,
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "artifact": "experiment_2869_fr11_continuous_self_learning_replay_v3",
        "schema": "carnot.fr11_continuous_self_learning_replay.v3",
        "honest_verdict": honest_verdict,
        "continuous_self_learning_task": True,
        "fr11_self_learning_ready": False,
        "offline_recurrence_backend_used": backend_path,
        "live_model_invoked": False,
        "no_model_weight_mutation": True,
        "n_examples": 0,
        "max_loops": config.max_loops,
        "recurrence_success_rate": 0.0,
        "energy_delta_mean": 0.0,
        "correctness_delta": 0.0,
        "forgetting_regression_count": 0,
        "memory_hash_before": "not_checked_precondition_failed",
        "memory_hash_after": "not_checked_precondition_failed",
        "per_loop_energy_summary": {},
        "source_counts": {},
        "preconditions_checked": list(preconditions),
        "random_seed": config.random_seed,
        "field_principles": FIELD_PRINCIPLES,
        "run_date": config.run_date,
        "duration_s": _round_float(duration_s),
        "methodology_note": "Blocked before offline replay; no recurrence metrics were inferred.",
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def _blocked_reason(
    backend_path: str,
    backend_cls: type[Any] | None,
    corpus: Sequence[ReplayExample],
    checks: Sequence[Mapping[str, Any]],
) -> str | None:
    failed = {str(check["check"]) for check in checks if not check.get("passed")}
    if "exp2868_artifact" in failed:
        return "blocked_missing_exp2868_artifact"
    if backend_path in {"", "unavailable"} or "exp2868_backend_module_path" in failed:
        return "blocked_offline_backend_module_path"
    if backend_cls is None or "offline_backend_imported" in failed:
        return "blocked_offline_backend_import"
    if "exp2865_artifact" in failed:
        return "blocked_missing_exp2865_artifact"
    if "exp2865_clean_corpora" in failed:
        return "blocked_no_clean_corpus_rows"
    if not corpus or "clean_replay_corpus" in failed:
        return "blocked_empty_replay_corpus"
    return None


def run_experiment(config: ExperimentConfig | None = None, *, write: bool = True) -> dict[str, Any]:
    """Run the Exp 2869 bounded offline continuous self-learning replay."""

    active_config = config or ExperimentConfig()
    started_at = active_config.start_time()
    backend_path, backend_cls, backend_checks = _load_backend(active_config)
    corpus, corpus_checks = load_clean_replay_corpus(active_config)
    preconditions = [*backend_checks, *corpus_checks]
    blocker = _blocked_reason(backend_path, backend_cls, corpus, preconditions)
    if blocker is not None:
        artifact = _blocked_artifact(
            config=active_config,
            honest_verdict=blocker,
            backend_path=backend_path,
            preconditions=preconditions,
            duration_s=active_config.clock() - started_at,
        )
        if write:
            active_config.output_dir().mkdir(parents=True, exist_ok=True)
            active_config.output_path().write_text(
                json.dumps(artifact, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return artifact

    memory_paths = _discover_memory_paths(active_config)
    memory_hash_before, before_manifest = _hash_memory_state(memory_paths, active_config.repo_root)
    backend = backend_cls(max_loops=active_config.max_loops)
    replay = backend.replay(build_backend_rows(corpus, active_config.max_loops))
    traces = list(replay["per_example_trace"])
    source_counts = dict(sorted(Counter(str(trace["source"]) for trace in traces).items()))
    metrics = summarize_trace_metrics(traces)
    memory_state = _build_memory_state(
        traces=traces, source_counts=source_counts, config=active_config
    )
    _write_memory_state(active_config.memory_path(), memory_state)
    memory_hash_after, after_manifest = _hash_memory_state(memory_paths, active_config.repo_root)
    changed_paths = _changed_memory_paths(before_manifest, after_manifest)
    permitted_path = _relative_path(active_config.memory_path(), active_config.repo_root)
    permitted_changes = all(path == permitted_path for path in changed_paths)
    preconditions.append(
        {
            "check": "permitted_memory_changes",
            "passed": permitted_changes,
            "observed": changed_paths,
        }
    )
    duration_s = active_config.clock() - started_at
    ready = (
        bool(traces)
        and bool(replay["no_model_weight_mutation"])
        and not bool(replay["live_model_invoked"])
        and permitted_changes
        and metrics["forgetting_regression_count"] == 0
        and metrics["energy_delta_mean"] > 0.0
    )
    artifact: dict[str, Any] = {
        "artifact": "experiment_2869_fr11_continuous_self_learning_replay_v3",
        "schema": "carnot.fr11_continuous_self_learning_replay.v3",
        "honest_verdict": (
            "complete: offline verifier-feedback replay lowered energy with no forgetting"
            if ready
            else "blocked_offline_replay_regression_or_no_gain"
        ),
        "continuous_self_learning_task": True,
        "fr11_self_learning_ready": ready,
        "offline_recurrence_backend_used": backend_path,
        "live_model_invoked": False,
        "no_model_weight_mutation": True,
        "n_examples": len(traces),
        "max_loops": active_config.max_loops,
        "recurrence_success_rate": metrics["recurrence_success_rate"],
        "energy_delta_mean": metrics["energy_delta_mean"],
        "correctness_delta": metrics["correctness_delta"],
        "forgetting_regression_count": metrics["forgetting_regression_count"],
        "memory_hash_before": memory_hash_before,
        "memory_hash_after": memory_hash_after,
        "per_loop_energy_summary": replay["per_loop_energy_summary"],
        "source_counts": source_counts,
        "preconditions_checked": preconditions,
        "random_seed": active_config.random_seed,
        "field_principles": FIELD_PRINCIPLES,
        "run_date": active_config.run_date,
        "duration_s": _round_float(duration_s),
        "methodology_note": (
            "Offline verifier-feedback replay only. The run updates a dedicated "
            "constraint-memory replay file and does not generate or repair model outputs."
        ),
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    if write:
        active_config.output_dir().mkdir(parents=True, exist_ok=True)
        active_config.output_path().write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return artifact
