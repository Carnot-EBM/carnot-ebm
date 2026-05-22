"""Exp 2868 offline recurrence backend adapter.

This module exists because the live LoopUS recurrence backend is not ready yet,
but downstream FR-11 work still needs a stable backend artifact it can import.
The adapter only replays verifier trace rows that already exist on disk or are
passed in by tests. It never calls a generator, never invokes a live LLM, and
never receives or mutates model weights.

Spec: REQ-LEARN-2868,
      SCENARIO-LEARN-2868.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


OUTPUT_FILENAME = "experiment_2868_offline_recurrence_backend_adapter_v2.json"
BACKEND_MODULE_PATH = "carnot.eval.offline_recurrence_backend_adapter_v2"
RUN_DATE = "20260522"
RANDOM_SEED = 2868
REPO_ROOT = Path(__file__).resolve().parents[3]

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "offline_recurrence_backend_ready",
    "live_recurrence_backend_ready",
    "backend_module_path",
    "backend_api",
    "replay_smoke_passed",
    "live_model_invoked",
    "no_model_weight_mutation",
    "preconditions_checked",
    "tests_run",
    "random_seed",
    "reproducibility_checksum",
    "field_principles",
    "run_date",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal complete:/blocked_ verdict; no inferred live success.",
    "offline_recurrence_backend_ready": "True only when trace replay normalizes at least one row.",
    "live_recurrence_backend_ready": "Always false; live repair remains a separate runtime task.",
    "backend_module_path": "Stable import path downstream experiments can gate on.",
    "backend_api": "Documents the replay-only callable and required row/output fields.",
    "replay_smoke_passed": "True only when the deterministic smoke replay completed.",
    "live_model_invoked": "False because the adapter consumes verifier traces only.",
    "no_model_weight_mutation": "True because no model object or weight tensor is accepted.",
    "reproducibility_checksum": "Hashes stable inputs, replay summaries, and API metadata.",
    "duration_s": "Real wall-clock duration; no sleep padding.",
}


@dataclass(frozen=True)
class PreconditionCheck:
    """One small fact checked before declaring the offline adapter usable."""

    resource: str
    available: bool
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {"resource": self.resource, "available": self.available, "detail": self.detail}


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime knobs for writing the Exp 2868 adapter artifact."""

    repo_root: Path = REPO_ROOT
    results_dir: Path | None = None
    run_date: str = RUN_DATE
    random_seed: int = RANDOM_SEED
    smoke_n_rows: int = 4
    started_at: float | None = None
    clock: Callable[[], float] = time.time

    def output_dir(self) -> Path:
        return self.results_dir if self.results_dir is not None else self.repo_root / "results"

    def output_path(self) -> Path:
        return self.output_dir() / OUTPUT_FILENAME

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at


def _round_float(value: float) -> float:
    return round(float(value), 12)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def backend_api() -> dict[str, object]:
    """Return the public API contract embedded into the Exp 2868 artifact."""

    return {
        "callable": "OfflineRecurrenceReplayBackend.replay",
        "module": BACKEND_MODULE_PATH,
        "input": {
            "type": "iterable[Mapping[str, object]]",
            "required_one_of": [
                "energy_before + revised_energy",
                "energy_before + energy_after_each_loop",
                "verifier_scores",
            ],
            "optional_fields": [
                "example_id",
                "case_id",
                "question_id",
                "source",
                "corpus",
                "correctness_before",
                "correctness_after",
                "localized_violations",
            ],
        },
        "output": {
            "fields": [
                "backend_module_path",
                "backend_api",
                "n_rows",
                "per_example_trace",
                "per_loop_energy_summary",
                "energy_delta_mean",
                "replay_smoke_passed",
                "live_model_invoked",
                "no_model_weight_mutation",
            ],
        },
        "live_model_invoked": False,
        "mutates_model_weights": False,
    }


class OfflineRecurrenceReplayBackend:
    """Replay verifier rows as recurrence energy summaries.

    The backend treats each row as already-produced verifier evidence. When the
    row has explicit `energy_after_each_loop` values, they are replayed as-is.
    When it has a single `revised_energy`, that becomes the one offline loop.
    When it only has verifier scores, the mean score is the initial energy and
    the lowest score is the revised summary. That gives downstream code a stable
    accounting surface without pretending a live repair was generated.
    """

    def __init__(self, *, max_loops: int = 3) -> None:
        if max_loops < 1:
            raise ValueError("max_loops must be >= 1")
        self.max_loops = int(max_loops)

    def replay(self, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        """Normalize verifier trace rows and aggregate deterministic energy summaries.

        Spec: REQ-LEARN-2868-1, REQ-LEARN-2868-2, SCENARIO-LEARN-2868
        """

        traces = [self._normalize_row(row, index) for index, row in enumerate(list(rows))]
        energy_deltas = [
            float(trace["energy_before"]) - float(trace["energy_after_each_loop"][-1])
            for trace in traces
            if trace["energy_after_each_loop"]
        ]
        return {
            "backend_module_path": BACKEND_MODULE_PATH,
            "backend_api": backend_api(),
            "n_rows": len(traces),
            "per_example_trace": traces,
            "per_loop_energy_summary": _summarize_loop_energy(traces),
            "energy_delta_mean": _round_float(_mean(energy_deltas)),
            "replay_smoke_passed": bool(traces),
            "live_model_invoked": False,
            "no_model_weight_mutation": True,
        }

    def _normalize_row(self, row: Mapping[str, Any], index: int) -> dict[str, Any]:
        scores = _verifier_scores(row)
        energy_before = _energy_before(row, scores)
        energy_after_each_loop = _energy_after_each_loop(row, scores, self.max_loops)
        if not energy_after_each_loop:
            energy_after_each_loop = [energy_before]
        return {
            "example_id": _example_id(row, index),
            "source": str(row.get("source") or row.get("corpus") or "verifier_trace"),
            "energy_before": _round_float(energy_before),
            "energy_after_each_loop": [_round_float(value) for value in energy_after_each_loop],
            "correctness_before": _correctness(row, "correctness_before"),
            "correctness_after": _correctness(row, "correctness_after"),
            "early_exit_reason": _early_exit_reason(row, energy_before, energy_after_each_loop),
            "localized_violations": _localized_violations(row, scores),
        }


def _example_id(row: Mapping[str, Any], index: int) -> str:
    for key in ("example_id", "case_id", "question_id", "trace_id", "id"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return f"trace-{index}"


def _verifier_scores(row: Mapping[str, Any]) -> dict[str, float]:
    value = row.get("verifier_scores")
    if not isinstance(value, Mapping):
        return {}
    scores: dict[str, float] = {}
    for key, raw_score in value.items():
        scores[str(key)] = float(raw_score)
    return scores


def _first_float(row: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return float(value)
    return None


def _energy_before(row: Mapping[str, Any], scores: Mapping[str, float]) -> float:
    direct = _first_float(row, ("energy_before", "initial_energy", "base_energy"))
    if direct is not None:
        return direct
    if scores:
        return _mean(list(scores.values()))
    raise ValueError("verifier trace row needs energy_before or verifier_scores")


def _energy_after_each_loop(
    row: Mapping[str, Any],
    scores: Mapping[str, float],
    max_loops: int,
) -> list[float]:
    loop_values = row.get("energy_after_each_loop")
    if isinstance(loop_values, Sequence) and not isinstance(loop_values, str | bytes):
        return [float(value) for value in list(loop_values)[:max_loops]]
    direct = _first_float(row, ("revised_energy", "energy_after", "final_energy"))
    if direct is not None:
        return [direct]
    if scores:
        return [min(scores.values())]
    raise ValueError("verifier trace row needs revised energy or verifier_scores")


def _correctness(row: Mapping[str, Any], key: str) -> bool:
    if key in row:
        return bool(row[key])
    label = row.get("label")
    if label in {"correct", 0, "0"}:
        return True
    if label in {"incorrect", 1, "1"}:
        return False
    return False


def _early_exit_reason(
    row: Mapping[str, Any],
    energy_before: float,
    energy_after_each_loop: Sequence[float],
) -> str:
    value = row.get("early_exit_reason")
    if value is not None:
        return str(value)
    if energy_after_each_loop and float(energy_after_each_loop[-1]) < float(energy_before):
        return "offline_energy_replayed"
    return "offline_no_energy_improvement"


def _localized_violations(
    row: Mapping[str, Any],
    scores: Mapping[str, float],
) -> list[str]:
    value = row.get("localized_violations")
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [str(item) for item in value]
    if not scores:
        return []
    max_score = max(scores.values())
    return sorted(name for name, score in scores.items() if score == max_score)


def _summarize_loop_energy(
    traces: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float | int]]:
    max_observed_loops = max(
        (len(trace.get("energy_after_each_loop", [])) for trace in traces),
        default=0,
    )
    summary: dict[str, dict[str, float | int]] = {}
    for loop_index in range(max_observed_loops):
        pairs = [
            (float(trace["energy_before"]), float(trace["energy_after_each_loop"][loop_index]))
            for trace in traces
            if len(trace.get("energy_after_each_loop", [])) > loop_index
        ]
        energies = [after for _before, after in pairs]
        deltas = [before - after for before, after in pairs]
        summary[f"loop_{loop_index + 1}"] = {
            "n": len(pairs),
            "mean_energy": _round_float(_mean(energies)),
            "mean_delta_from_initial": _round_float(_mean(deltas)),
        }
    return summary


def _label_is_correct(label: object) -> bool:
    return label in {"correct", 0, "0"}


def load_smoke_trace_rows(repo_root: Path, *, n_rows: int, seed: int) -> list[dict[str, Any]]:
    """Load a tiny deterministic FoVer-derived replay subset.

    The FoVer corpus rows are existing verifier traces: they already contain a
    verifier label and confidence. The offline replay converts that evidence
    into one synthetic energy-summary row per trace without calling a model or
    changing correctness labels.
    """

    path = Path(repo_root) / "data" / "fover_corpus.jsonl"
    if not path.is_file():
        return []
    valid_rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("label") in {"correct", "incorrect", 0, 1, "0", "1"}:
            valid_rows.append(row)
    if not valid_rows:
        return []
    positives = [row for row in valid_rows if not _label_is_correct(row.get("label"))]
    negatives = [row for row in valid_rows if _label_is_correct(row.get("label"))]
    positive_offset = seed % len(positives) if positives else 0
    negative_offset = seed % len(negatives) if negatives else 0
    selected: list[dict[str, Any]] = []
    while len(selected) < n_rows:
        if positives and len(selected) < n_rows:
            selected.append(positives[(positive_offset + len(selected)) % len(positives)])
        if negatives and len(selected) < n_rows:
            selected.append(negatives[(negative_offset + len(selected)) % len(negatives)])
    traces: list[dict[str, Any]] = []
    for row in selected:
        confidence = float(row.get("confidence", 1.0))
        label_correct = _label_is_correct(row.get("label"))
        energy_before = max(0.0, 1.0 - confidence) if label_correct else confidence
        revised_energy = max(0.0, energy_before - min(0.25, 0.25 * confidence))
        traces.append(
            {
                "example_id": str(row.get("question_id", len(traces))),
                "source": str(row.get("source") or "fover"),
                "energy_before": _round_float(energy_before),
                "revised_energy": _round_float(revised_energy),
                "correctness_before": label_correct,
                "correctness_after": label_correct,
                "localized_violations": []
                if label_correct
                else [str(row.get("verifier") or "fover_verifier")],
            }
        )
    return traces


def probe_preconditions(
    config: ExperimentConfig,
    trace_rows: Sequence[Mapping[str, Any]],
) -> list[PreconditionCheck]:
    """Return the small offline checks needed before artifact success."""

    return [
        PreconditionCheck("repo_root", Path(config.repo_root).is_dir(), str(config.repo_root)),
        PreconditionCheck(
            "offline_backend_module_path",
            bool(BACKEND_MODULE_PATH),
            BACKEND_MODULE_PATH,
        ),
        PreconditionCheck(
            "verifier_trace_rows",
            len(trace_rows) > 0,
            f"rows={len(trace_rows)}",
        ),
    ]


def _checksum(artifact: Mapping[str, Any]) -> str:
    stable = {
        key: artifact[key]
        for key in sorted(artifact)
        if key not in {"duration_s", "reproducibility_checksum"}
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    trace_rows: Sequence[Mapping[str, Any]] | None = None,
    tests_run: Sequence[str] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Write the Exp 2868 offline backend artifact.

    Spec: REQ-LEARN-2868-3
    """

    active_config = config or ExperimentConfig()
    started_at = active_config.start_time()
    rows = (
        list(trace_rows)
        if trace_rows is not None
        else load_smoke_trace_rows(
            active_config.repo_root,
            n_rows=active_config.smoke_n_rows,
            seed=active_config.random_seed,
        )
    )
    checks = probe_preconditions(active_config, rows)
    checks_passed = all(check.available for check in checks)
    replay = (
        OfflineRecurrenceReplayBackend().replay(rows)
        if checks_passed
        else {
            "backend_module_path": None,
            "backend_api": backend_api(),
            "n_rows": 0,
            "per_example_trace": [],
            "per_loop_energy_summary": {},
            "energy_delta_mean": 0.0,
            "replay_smoke_passed": False,
            "live_model_invoked": False,
            "no_model_weight_mutation": True,
        }
    )
    duration_s = active_config.clock() - started_at
    ready = bool(replay["replay_smoke_passed"])
    artifact: dict[str, Any] = {
        "artifact": "experiment_2868_offline_recurrence_backend_adapter_v2",
        "schema": "carnot.offline_recurrence_backend_adapter.v2",
        "honest_verdict": (
            "complete: offline recurrence replay backend ready"
            if ready
            else "blocked_no_verifier_trace_rows"
        ),
        "offline_recurrence_backend_ready": ready,
        "live_recurrence_backend_ready": False,
        "backend_module_path": replay["backend_module_path"],
        "backend_api": replay["backend_api"],
        "replay_smoke_passed": ready,
        "live_model_invoked": False,
        "no_model_weight_mutation": True,
        "preconditions_checked": [check.as_dict() for check in checks],
        "tests_run": list(tests_run or []),
        "random_seed": active_config.random_seed,
        "field_principles": FIELD_PRINCIPLES,
        "run_date": active_config.run_date,
        "duration_s": _round_float(duration_s),
        "n_rows": replay["n_rows"],
        "energy_delta_mean": replay["energy_delta_mean"],
        "per_loop_energy_summary": replay["per_loop_energy_summary"],
        "per_example_trace": replay["per_example_trace"],
        "methodology_note": (
            "Offline replay over existing verifier traces only. Revised energies are "
            "deterministic summary values, not live LLM repair outputs."
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
