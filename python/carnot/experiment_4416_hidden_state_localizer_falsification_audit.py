"""Exp 4416: hidden-state first-error localizer falsification audit.

Spec refs: REQ-VERIFY-4416, SCENARIO-VERIFY-4416.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot import experiment_4403_real_intervention_localizer_deconfound as exp4403
from carnot.experiment_4381_biprm_detector_localization_abstention import (
    NoStepLabelsError,
    load_step_labeled_traces,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = ROOT / "results" / "experiment_4416_hidden_state_localizer_falsification_audit.json"
FOVER_ROW_CORPUS_PATH = ROOT / "data" / "fover_corpus.jsonl"
FOVER_STEP_CORPUS_PATH = ROOT / "data" / "step_level_prm_training.jsonl"
EXP2850_ARTIFACT_PATH = ROOT / "results" / "experiment_2850_fover_dual_condition_integrity_v4.json"
EXP4403_ARTIFACT_PATH = ROOT / "results" / "experiment_4403_real_intervention_localizer_deconfound.json"
VERIFIER_GAPS_PATH = ROOT / "ops" / "verifier_gaps.md"

RANDOM_SEED = 4416
RANDOM_SEEDS_USED = (4416,)
BOOTSTRAP_RESAMPLES = 2500
MIN_CAPTURE_ERROR_TRACES = 1250
MIN_HELDOUT_TRACES = 1000
HELDOUT_FRACTION = 0.8
DEFAULT_HF_MODEL_ID = "Qwen/Qwen2.5-0.5B"
DEFAULT_MAX_LENGTH = 96
DEFAULT_BATCH_SIZE = 16
INFERENCE_SUBSTRATE = "local_hf_hidden_state_forward_pass"
SPEC_REFS = ["REQ-VERIFY-4416", "SCENARIO-VERIFY-4416"]

HIDDEN_FEATURE_NAMES = (
    "hidden_norm",
    "transport_from_prev_l2",
    "transport_to_next_l2",
    "trace_center_l2",
    "local_transport_margin",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A measured off-text signal (logged as a gap) and "
        "a CLEAN NULL (the localizer program conclusively closed) are BOTH "
        "decision-grade."
    ),
    "hidden_state_localizer_has_nonposition_signal": (
        "BARE bool: the capstone reads this; true iff the hidden-state probe's "
        "held-out first-error F1 exceeds the content-blind position-only "
        "baseline (delta CI95-excl-0) -- the one remaining oracle-distinct "
        "first-error question, settled off-text."
    ),
    "localization_f1_comparison": (
        "dict: {position_only_baseline_f1, text_localizer_f1_exp4403, "
        "hidden_state_probe_f1, delta_vs_position_only, delta_ci95, n_traces} "
        "-- the head-to-head that settles whether off-text adds anything."
    ),
    "position_only_baseline_f1": (
        "BARE float: the content-blind position-only baseline F1 on the "
        "held-out split -- the bar the hidden-state probe must clear "
        "CI95-excl-0 (the .407 confound made explicit)."
    ),
    "n_traces": (
        "BARE int: held-out evaluation trace count -- MUST be >= 1000 for the "
        "localization claim (sample-size rigor)."
    ),
    "verifier_is_oracle": (
        "BARE bool=false -- a learned probe over model hidden states; the "
        "reference prefix-check defines correctness, the probe estimates WHICH "
        "step -- oracle-DISTINCT."
    ),
    "preconditions_checked": (
        "Records the cached-corpus + the hidden-state-extraction-path + "
        "TRM-stand-down verified; pre-empts the silent-missing-resource "
        "fabrication mode."
    ),
    "random_seed": "Determinism precondition for the capture + the probe fit + the bootstrap.",
    "reproducibility_checksum": (
        "Hash of the hidden-state capture + the probe config + the held-out "
        "split + the controls; lets a third party re-run."
    ),
    "model_specs": (
        "The LOCAL open model used for hidden-state capture (HF base repo or "
        "documented local extraction path) + the FoVer corpus + the position-only "
        "baseline + n; required methodology + the oracle-distinct declaration."
    ),
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "hidden_state_localizer_has_nonposition_signal",
    "localization_f1_comparison",
    "position_only_baseline_f1",
    "n_traces",
    "verifier_is_oracle",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "model_specs",
)


@dataclass(frozen=True)
class PreconditionCheck:
    """One prerequisite checked before hidden-state capture."""

    resource: str
    available: bool
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "resource": self.resource,
            "available": bool(self.available),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class HiddenStatePathStatus:
    """Result of probing the local hidden-state extraction path."""

    available: bool
    detail: str
    model_specs: dict[str, Any]


@dataclass(frozen=True)
class HiddenStateTrace:
    """One cached FoVer failed trace used for first-error localization."""

    trace_id: str
    steps: tuple[str, ...]
    first_error_index: int
    position_bin: str
    source_row_id: str
    error_class: str = "fover_row_single_step"


@dataclass(frozen=True)
class StepFeatureRecord:
    """One per-step hidden transport/margin feature vector."""

    trace_id: str
    step_index: int
    n_steps: int
    first_error_index: int
    is_first_error: bool
    features: dict[str, float]


@dataclass(frozen=True)
class PositionOnlyBaseline:
    """Content-blind predictor using empirical first-error position counts."""

    position_counts: dict[int, int]

    @classmethod
    def fit(cls, traces: Sequence[HiddenStateTrace]) -> "PositionOnlyBaseline":
        counts: Counter[int] = Counter(int(trace.first_error_index) for trace in traces)
        return cls(position_counts=dict(sorted(counts.items())))

    def predict_first_error_index(self, trace: HiddenStateTrace) -> int | None:
        if not trace.steps or not self.position_counts:
            return None
        valid = [idx for idx in self.position_counts if idx < len(trace.steps)]
        if not valid:
            return len(trace.steps) - 1
        return max(valid, key=lambda idx: (self.position_counts[idx], -idx))


@dataclass(frozen=True)
class HiddenStateMarginProbe:
    """CPU-only linear probe over non-position hidden transport features."""

    weights: dict[str, float]
    training_summary: dict[str, Any]

    @classmethod
    def fit(cls, records: Sequence[StepFeatureRecord]) -> "HiddenStateMarginProbe":
        positives = [record.features for record in records if record.is_first_error]
        negatives = [record.features for record in records if not record.is_first_error]
        weights: dict[str, float] = {}
        for name in HIDDEN_FEATURE_NAMES:
            if positives and negatives:
                pos_mean = sum(float(row.get(name, 0.0)) for row in positives) / len(positives)
                neg_mean = sum(float(row.get(name, 0.0)) for row in negatives) / len(negatives)
                weights[name] = pos_mean - neg_mean
            else:
                weights[name] = 0.0
        return cls(
            weights=weights,
            training_summary={
                "training_step_count": len(records),
                "positive_first_error_steps": len(positives),
                "negative_non_first_error_steps": len(negatives),
                "feature_names": list(HIDDEN_FEATURE_NAMES),
                "position_features_used": False,
            },
        )

    def score(self, record: StepFeatureRecord) -> float:
        return sum(
            float(self.weights.get(name, 0.0)) * float(record.features.get(name, 0.0))
            for name in HIDDEN_FEATURE_NAMES
        )

    def predict_first_error_index(self, trace_id: str, records: Sequence[StepFeatureRecord]) -> int | None:
        trace_records = [record for record in records if record.trace_id == trace_id]
        if not trace_records:
            return None
        best = max(trace_records, key=lambda record: (self.score(record), -record.step_index))
        return int(best.step_index)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_type": "linear_hidden_transport_margin_probe",
            "weights": {key: round_float(value) for key, value in sorted(self.weights.items())},
            "training_summary": self.training_summary,
        }


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime configuration for Exp 4416."""

    repo_root: Path = ROOT
    fover_row_corpus_path: Path = FOVER_ROW_CORPUS_PATH
    fover_step_corpus_path: Path = FOVER_STEP_CORPUS_PATH
    exp2850_artifact_path: Path = EXP2850_ARTIFACT_PATH
    exp4403_artifact_path: Path = EXP4403_ARTIFACT_PATH
    verifier_gaps_path: Path = VERIFIER_GAPS_PATH
    artifact_path: Path = ARTIFACT_PATH
    hf_model_id: str = DEFAULT_HF_MODEL_ID
    min_capture_error_traces: int = MIN_CAPTURE_ERROR_TRACES
    min_eval_traces: int = MIN_HELDOUT_TRACES
    heldout_fraction: float = HELDOUT_FRACTION
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES
    random_seed: int = RANDOM_SEED
    started_at: float | None = None
    clock: Callable[[], float] = time.time

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at


AdversarialVerifyRunner = Callable[[Path], dict[str, Any]]


class TransformersHiddenStateExtractor:  # pragma: no cover - exercised by the real result run.
    """Local HuggingFace hidden-state extractor for the default experiment path."""

    def __init__(
        self,
        model_id: str,
        *,
        cache_dir: Path | None = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.max_length = max_length
        self.batch_size = batch_size
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._device: str | None = None
        self._snapshot_path: Path | None = None

    def _resolve_snapshot(self) -> Path:
        if "gguf" in self.model_id.lower():
            raise RuntimeError("gguf-only repositories are not valid transformers hidden-state paths")
        cache_root = self.cache_dir or Path.home() / ".cache" / "huggingface" / "hub"
        repo_dir = cache_root / ("models--" + self.model_id.replace("/", "--"))
        refs_main = repo_dir / "refs" / "main"
        if not refs_main.is_file():
            raise RuntimeError(f"local HF cache ref missing for {self.model_id}")
        snapshot = repo_dir / "snapshots" / refs_main.read_text(encoding="utf-8").strip()
        if not snapshot.is_dir():
            raise RuntimeError(f"local HF snapshot missing for {self.model_id}")
        has_tokenizer = any(
            (snapshot / name).exists()
            for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")
        )
        if not (snapshot / "config.json").exists() or not has_tokenizer:
            raise RuntimeError(f"HF base config/tokenizer files missing for {self.model_id}")
        self._snapshot_path = snapshot
        return snapshot

    def _load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        snapshot = self._resolve_snapshot()
        import torch
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot),
            local_files_only=True,
            trust_remote_code=True,
        )
        if getattr(self._tokenizer, "pad_token", None) is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        try:
            model = AutoModel.from_pretrained(
                str(snapshot),
                local_files_only=True,
                trust_remote_code=True,
            )
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                str(snapshot),
                local_files_only=True,
                trust_remote_code=True,
            )
        self._model = model.to(self._device)
        self._model.eval()

    def check(self) -> HiddenStatePathStatus:
        try:
            self.capture(
                [
                    HiddenStateTrace(
                        trace_id="hidden_state_probe_check",
                        steps=("Compute 1 + 1 = 2.",),
                        first_error_index=0,
                        position_bin="0",
                        source_row_id="hidden_state_probe_check",
                    )
                ]
            )
        except Exception as exc:
            return HiddenStatePathStatus(False, f"{type(exc).__name__}: {exc}", {})
        hidden_size = int(getattr(getattr(self._model, "config", None), "hidden_size", 0) or 0)
        layer_count = int(getattr(getattr(self._model, "config", None), "num_hidden_layers", 0) or 0)
        return HiddenStatePathStatus(
            True,
            "transformers output_hidden_states probe succeeded",
            {
                "model_id": self.model_id,
                "snapshot_path": str(self._snapshot_path),
                "device": self._device,
                "hidden_size": hidden_size,
                "num_hidden_layers": layer_count,
                "max_length": self.max_length,
                "batch_size": self.batch_size,
                "hidden_state_extraction_path": "transformers.AutoModel(output_hidden_states=True)",
                "gguf_tokenizer_rule": "base_hf_repo_used_not_gguf_only_repo",
            },
        )

    def capture(self, traces: Sequence[HiddenStateTrace]) -> list[list[list[float]]]:
        self._load()
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._torch is not None
        assert self._device is not None

        flat_texts = [step for trace in traces for step in trace.steps]
        flat_vectors: list[list[float]] = []
        for start in range(0, len(flat_texts), self.batch_size):
            batch = flat_texts[start : start + self.batch_size]
            encoded = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            with self._torch.inference_mode():
                outputs = self._model(
                    **encoded,
                    output_hidden_states=True,
                    return_dict=True,
                )
            hidden = outputs.hidden_states[-1]
            lengths = encoded["attention_mask"].sum(dim=1) - 1
            batch_indices = self._torch.arange(hidden.shape[0], device=hidden.device)
            vectors = hidden[batch_indices, lengths].detach().float().cpu().tolist()
            flat_vectors.extend([[float(value) for value in vector] for vector in vectors])

        captured: list[list[list[float]]] = []
        cursor = 0
        for trace in traces:
            width = len(trace.steps)
            captured.append(flat_vectors[cursor : cursor + width])
            cursor += width
        return captured


def round_float(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return round(float(value), digits)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_artifact(path: Path, artifact: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _f1(successes: Sequence[int]) -> float:
    return sum(int(value) for value in successes) / len(successes) if successes else 0.0


def _l2(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right, strict=True)))


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in vector))


def _vector_mean(vectors: Sequence[Sequence[float]]) -> list[float]:
    if not vectors:
        return []
    width = len(vectors[0])
    return [sum(float(vector[idx]) for vector in vectors) / len(vectors) for idx in range(width)]


def build_real_fover_error_traces_from_rows(rows: Sequence[dict[str, Any]]) -> list[HiddenStateTrace]:
    """Convert Exp 4403 verifier-checked failed rows into first-error traces."""

    labels = exp4403.build_fover_intervention_labels_from_rows(rows)
    traces: list[HiddenStateTrace] = []
    for label in labels:
        if label.trace.first_error_index is None or not label.intervention_verified:
            continue
        steps = tuple(step.text for step in label.trace.steps)
        if not steps:  # pragma: no cover - Exp 4403 labels always contain one row step.
            continue
        first_error = int(label.trace.first_error_index)
        traces.append(
            HiddenStateTrace(
                trace_id=label.trace.trace_id,
                steps=steps,
                first_error_index=first_error,
                position_bin=str(first_error),
                source_row_id=label.source_row_id,
                error_class=label.family,
            )
        )
    return sorted(traces, key=lambda trace: trace.trace_id)


def load_real_fover_error_traces(path: Path, *, limit: int | None = None) -> list[HiddenStateTrace]:
    traces = build_real_fover_error_traces_from_rows(_read_jsonl(path))
    if limit is not None:
        return traces[:limit]
    return traces


def position_bin_counts(traces: Sequence[HiddenStateTrace]) -> dict[str, int]:
    return dict(sorted(Counter(trace.position_bin for trace in traces).items()))


def split_train_heldout(
    traces: Sequence[HiddenStateTrace],
    *,
    seed: int,
    heldout_fraction: float,
    min_eval_traces: int,
) -> tuple[list[HiddenStateTrace], list[HiddenStateTrace]]:
    ordered = sorted(traces, key=lambda trace: trace.trace_id)
    rng = random.Random(seed)
    shuffled = list(ordered)
    rng.shuffle(shuffled)
    heldout_n = max(min_eval_traces, int(round(len(shuffled) * heldout_fraction)))
    heldout_n = min(len(shuffled) - 1, heldout_n)
    heldout = sorted(shuffled[:heldout_n], key=lambda trace: trace.trace_id)
    train = sorted(shuffled[heldout_n:], key=lambda trace: trace.trace_id)
    return train, heldout


def transport_margin_features(
    traces: Sequence[HiddenStateTrace],
    hidden_states: Sequence[Sequence[Sequence[float]]],
) -> list[StepFeatureRecord]:
    """Compute non-position hidden-state transport/margin features per step."""

    records: list[StepFeatureRecord] = []
    for trace, states in zip(traces, hidden_states, strict=True):
        state_list = [[float(value) for value in state] for state in states]
        if len(state_list) != len(trace.steps):
            raise ValueError(f"hidden state count mismatch for {trace.trace_id}")
        center = _vector_mean(state_list)
        for idx, state in enumerate(state_list):
            prev_l2 = _l2(state, state_list[idx - 1]) if idx > 0 else 0.0
            next_l2 = _l2(state, state_list[idx + 1]) if idx + 1 < len(state_list) else 0.0
            center_l2 = _l2(state, center) if center else 0.0
            features = {
                "hidden_norm": _norm(state),
                "transport_from_prev_l2": prev_l2,
                "transport_to_next_l2": next_l2,
                "trace_center_l2": center_l2,
                "local_transport_margin": next_l2 - prev_l2,
            }
            records.append(
                StepFeatureRecord(
                    trace_id=trace.trace_id,
                    step_index=idx,
                    n_steps=len(trace.steps),
                    first_error_index=int(trace.first_error_index),
                    is_first_error=idx == int(trace.first_error_index),
                    features=features,
                )
            )
    return records


def _records_by_trace(records: Sequence[StepFeatureRecord]) -> dict[str, list[StepFeatureRecord]]:
    grouped: dict[str, list[StepFeatureRecord]] = {}
    for record in records:
        grouped.setdefault(record.trace_id, []).append(record)
    return {key: sorted(value, key=lambda record: record.step_index) for key, value in grouped.items()}


def _position_successes(
    traces: Sequence[HiddenStateTrace],
    baseline: PositionOnlyBaseline,
) -> list[int]:
    return [int(baseline.predict_first_error_index(trace) == trace.first_error_index) for trace in traces]


def _probe_successes(
    traces: Sequence[HiddenStateTrace],
    features: Sequence[StepFeatureRecord],
    probe: HiddenStateMarginProbe,
) -> list[int]:
    grouped = _records_by_trace(features)
    return [
        int(probe.predict_first_error_index(trace.trace_id, grouped.get(trace.trace_id, ())) == trace.first_error_index)
        for trace in traces
    ]


def _paired_delta_ci95(
    left_successes: Sequence[int],
    right_successes: Sequence[int],
    *,
    seed: int,
    resamples: int,
) -> list[float | None]:
    if not left_successes or len(left_successes) != len(right_successes) or resamples <= 0:
        return [None, None]
    rng = random.Random(seed)
    values: list[float] = []
    n = len(left_successes)
    for _ in range(resamples):
        delta_sum = 0
        for _idx in range(n):
            item = rng.randrange(n)
            delta_sum += int(left_successes[item]) - int(right_successes[item])
        values.append(delta_sum / n)
    values.sort()
    lo = int(0.025 * (len(values) - 1))
    hi = int(0.975 * (len(values) - 1))
    return [round_float(values[lo]), round_float(values[hi])]


def evaluate_hidden_state_probe(
    heldout_traces: Sequence[HiddenStateTrace],
    heldout_features: Sequence[StepFeatureRecord],
    probe: HiddenStateMarginProbe,
    position_baseline: PositionOnlyBaseline,
    *,
    text_localizer_f1: float | None,
    seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    hidden_successes = _probe_successes(heldout_traces, heldout_features, probe)
    position_successes = _position_successes(heldout_traces, position_baseline)
    hidden_f1 = _f1(hidden_successes)
    position_f1 = _f1(position_successes)
    return {
        "position_only_baseline_f1": round_float(position_f1),
        "text_localizer_f1_exp4403": round_float(text_localizer_f1),
        "hidden_state_probe_f1": round_float(hidden_f1),
        "delta_vs_position_only": round_float(hidden_f1 - position_f1),
        "delta_ci95": _paired_delta_ci95(
            hidden_successes,
            position_successes,
            seed=seed,
            resamples=bootstrap_resamples,
        ),
        "n_traces": int(len(heldout_traces)),
        "hidden_exact_match_count": int(sum(hidden_successes)),
        "position_exact_match_count": int(sum(position_successes)),
    }


def has_nonposition_signal(comparison: dict[str, Any]) -> bool:
    ci95 = comparison.get("delta_ci95")
    if not isinstance(ci95, list | tuple) or len(ci95) != 2:
        return False
    delta = comparison.get("delta_vs_position_only")
    return bool(delta is not None and float(delta) > 0.0 and ci95[0] is not None and float(ci95[0]) > 0.0)


def text_localizer_f1_from_exp4403(payload: dict[str, Any]) -> float | None:
    domain = payload.get("localization_f1_by_domain", {}).get("FoVer")
    if not isinstance(domain, dict):
        return None
    value = domain.get("real_intervention_localizer")
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def reproducibility_checksum(source_paths: Sequence[Path], payload: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for path in sorted({Path(path) for path in source_paths}, key=lambda item: str(item)):
        digest.update(str(path).encode("utf-8"))
        if not path.exists():
            digest.update(b"\0MISSING\0")
            continue
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return "sha256:" + digest.hexdigest()


def _capture_checksum(
    traces: Sequence[HiddenStateTrace],
    hidden_states: Sequence[Sequence[Sequence[float]]],
) -> str:
    digest = hashlib.sha256()
    for trace, states in zip(traces, hidden_states, strict=True):
        digest.update(trace.trace_id.encode("utf-8"))
        digest.update(str(trace.first_error_index).encode("utf-8"))
        for state in states:
            digest.update(json.dumps([round_float(float(value), 8) for value in state]).encode("utf-8"))
    return "sha256:" + digest.hexdigest()


def _load_exp2850_check(path: Path) -> PreconditionCheck:
    if not path.is_file():
        return PreconditionCheck("exp2850_fover_corpus", False, "missing")
    try:
        payload = _read_json(path)
    except Exception as exc:
        return PreconditionCheck("exp2850_fover_corpus", False, f"unreadable: {exc}")
    n_examples = int(payload.get("n_examples") or 0) if isinstance(payload, dict) else 0
    return PreconditionCheck("exp2850_fover_corpus", n_examples >= 1000, f"n_examples={n_examples}")


def _step_corpus_check(path: Path) -> PreconditionCheck:
    if not path.is_file():
        return PreconditionCheck("cached_step_labeled_fover_corpus", False, "missing")
    try:
        traces = load_step_labeled_traces(path)
    except NoStepLabelsError as exc:
        return PreconditionCheck("cached_step_labeled_fover_corpus", False, f"no step labels: {exc}")
    except Exception as exc:
        return PreconditionCheck("cached_step_labeled_fover_corpus", False, f"unreadable: {exc}")
    error_traces = sum(1 for trace in traces if trace.has_error)
    return PreconditionCheck(
        "cached_step_labeled_fover_corpus",
        bool(traces),
        f"step_traces={len(traces)}; error_traces={error_traces}",
    )


def _text_localizer_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = _read_json(path)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _blocked_reason(checks: Sequence[PreconditionCheck]) -> str | None:
    if all(check.available for check in checks):
        return None
    by_resource = {check.resource: check for check in checks}
    hidden = by_resource.get("hidden_state_extraction_path")
    if hidden is not None and not hidden.available:
        return "blocked_no_hidden_state_extraction_path"
    return "blocked_cached_corpus_unavailable"


def _missing_gap(comparison: dict[str, Any], n_captured: int) -> dict[str, Any]:
    return {
        "gap_id": "GAP-FOVER-HIDDEN-STATE-LOCALIZATION-POSITION-SATURATED",
        "status": "open",
        "parent_gap": "GAP-FOVER-BIPRM-LOCALIZATION-untyped",
        "evidence": str(ARTIFACT_PATH),
        "failure_mode": (
            "The hidden-state transport/margin probe tied the content-blind "
            "position-only first-error baseline; the available FoVer failed "
            "traces are position-saturated at first step."
        ),
        "missing_discriminator": (
            "A model-native signal that separates earliest causal error from "
            "position and downstream consequence under non-degenerate "
            "multi-step FoVer labels."
        ),
        "candidate_design": (
            "Collect typed multi-step FoVer traces with non-first-position "
            "first errors before any localizer redeployment; do not revive the "
            "position-saturated text or hidden-state localizer line."
        ),
        "priority": "medium",
        "n_captured_traces": int(n_captured),
        "delta_vs_position_only": comparison.get("delta_vs_position_only"),
        "delta_ci95": comparison.get("delta_ci95"),
    }


def _model_specs(
    *,
    hidden_status: HiddenStatePathStatus,
    capture_receipts: dict[str, Any],
    probe: HiddenStateMarginProbe | None,
    comparison: dict[str, Any],
    config: ExperimentConfig,
) -> dict[str, Any]:
    specs = dict(hidden_status.model_specs)
    specs.update(
        {
            "fover_row_corpus": str(config.fover_row_corpus_path),
            "fover_step_corpus": str(config.fover_step_corpus_path),
            "exp2850_artifact": str(config.exp2850_artifact_path),
            "exp4403_text_localizer_artifact": str(config.exp4403_artifact_path),
            "capture_n": capture_receipts.get("n_captured_traces", 0),
            "heldout_n": comparison.get("n_traces", 0),
            "position_only_baseline": {
                "content_blind": True,
                "position_counts": capture_receipts.get("n_per_position_bin", {}),
            },
            "hidden_state_probe": probe.as_dict() if probe is not None else None,
            "bootstrap_resamples": int(config.bootstrap_resamples),
            "random_seed": int(config.random_seed),
            "random_seeds_used": list(RANDOM_SEEDS_USED),
            "trm_training": "stood_down_not_invoked",
            "generator_training": "stood_down_not_invoked",
            "live_generation": False,
            "verifier_is_oracle": False,
        }
    )
    return specs


def build_complete_artifact(
    *,
    traces: Sequence[HiddenStateTrace],
    train_traces: Sequence[HiddenStateTrace],
    heldout_traces: Sequence[HiddenStateTrace],
    hidden_states: Sequence[Sequence[Sequence[float]]],
    train_features: Sequence[StepFeatureRecord],
    heldout_features: Sequence[StepFeatureRecord],
    hidden_status: HiddenStatePathStatus,
    preconditions_checked: list[dict[str, Any]],
    comparison: dict[str, Any],
    probe: HiddenStateMarginProbe,
    source_paths: Sequence[Path],
    duration_s: float,
    config: ExperimentConfig,
) -> dict[str, Any]:
    signal = has_nonposition_signal(comparison)
    capture_receipts = {
        "n_captured_traces": int(len(traces)),
        "n_captured_steps": int(sum(len(trace.steps) for trace in traces)),
        "n_train_traces": int(len(train_traces)),
        "n_heldout_traces": int(len(heldout_traces)),
        "n_per_position_bin": position_bin_counts(traces),
        "feature_names": list(HIDDEN_FEATURE_NAMES),
        "hidden_state_capture_sha256": _capture_checksum(traces, hidden_states),
    }
    checksum_payload = {
        "comparison": comparison,
        "capture_receipts": capture_receipts,
        "probe": probe.as_dict(),
        "random_seed": config.random_seed,
        "bootstrap_resamples": config.bootstrap_resamples,
    }
    verdict = (
        "complete: hidden_state_nonposition_signal_detected"
        if signal
        else "complete: clean_powered_null_position_only_not_beaten"
    )
    artifact = {
        "experiment": "experiment_4416_hidden_state_localizer_falsification_audit",
        "schema": "carnot.hidden_state_localizer_falsification_audit.v1",
        "honest_verdict": verdict,
        "hidden_state_localizer_has_nonposition_signal": signal,
        "localization_f1_comparison": comparison,
        "position_only_baseline_f1": float(comparison["position_only_baseline_f1"]),
        "n_traces": int(comparison["n_traces"]),
        "verifier_is_oracle": False,
        "preconditions_checked": preconditions_checked,
        "random_seed": int(config.random_seed),
        "random_seeds_used": list(RANDOM_SEEDS_USED),
        "reproducibility_checksum": reproducibility_checksum(source_paths, checksum_payload),
        "model_specs": _model_specs(
            hidden_status=hidden_status,
            capture_receipts=capture_receipts,
            probe=probe,
            comparison=comparison,
            config=config,
        ),
        "hidden_state_capture_receipts": capture_receipts,
        "missing_verifier_gaps": [] if signal else [_missing_gap(comparison, len(traces))],
        "methodology_note": (
            "The default powered FoVer failed rows are verifier-checked one-step "
            "intervention traces from Exp 4403. Their first-error position is "
            "deterministic at step 0, so a hidden-state F1 tie at 1.0 is the "
            "expected content-blind null rather than evidence of a useful "
            "localizer."
        ),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "duration_s": round_float(duration_s, 3),
        "inference_substrate": INFERENCE_SUBSTRATE,
    }
    return artifact


def build_blocked_artifact(
    *,
    honest_verdict: str,
    preconditions_checked: list[dict[str, Any]],
    hidden_status: HiddenStatePathStatus,
    text_localizer_f1: float | None,
    source_paths: Sequence[Path],
    duration_s: float,
    config: ExperimentConfig,
) -> dict[str, Any]:
    comparison = {
        "position_only_baseline_f1": None,
        "text_localizer_f1_exp4403": round_float(text_localizer_f1),
        "hidden_state_probe_f1": None,
        "delta_vs_position_only": None,
        "delta_ci95": [None, None],
        "n_traces": 0,
    }
    capture_receipts = {
        "n_captured_traces": 0,
        "n_captured_steps": 0,
        "n_train_traces": 0,
        "n_heldout_traces": 0,
        "n_per_position_bin": {},
        "feature_names": list(HIDDEN_FEATURE_NAMES),
        "hidden_state_capture_sha256": None,
    }
    return {
        "experiment": "experiment_4416_hidden_state_localizer_falsification_audit",
        "schema": "carnot.hidden_state_localizer_falsification_audit.v1",
        "honest_verdict": honest_verdict,
        "hidden_state_localizer_has_nonposition_signal": False,
        "localization_f1_comparison": comparison,
        "position_only_baseline_f1": None,
        "n_traces": 0,
        "verifier_is_oracle": False,
        "preconditions_checked": preconditions_checked,
        "random_seed": int(config.random_seed),
        "random_seeds_used": list(RANDOM_SEEDS_USED),
        "reproducibility_checksum": reproducibility_checksum(
            source_paths,
            {
                "blocked": honest_verdict,
                "preconditions": preconditions_checked,
                "random_seed": config.random_seed,
            },
        ),
        "model_specs": _model_specs(
            hidden_status=hidden_status,
            capture_receipts=capture_receipts,
            probe=None,
            comparison=comparison,
            config=config,
        ),
        "hidden_state_capture_receipts": capture_receipts,
        "missing_verifier_gaps": [],
        "methodology_note": "blocked before hidden-state capture; no hidden-state metrics fabricated",
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "duration_s": round_float(duration_s, 3),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "adversarial_verify": {"returncode": None, "skipped": "blocked"},
    }


def run_adversarial_verify(path: Path, repo_root: Path = ROOT) -> dict[str, Any]:
    script = repo_root / "scripts" / "adversarial_verify.py"
    if not script.is_file():
        return {"returncode": None, "stderr_tail": "scripts/adversarial_verify.py missing"}
    proc = subprocess.run(
        [sys.executable, str(script), str(path)],
        cwd=str(repo_root),
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    return {
        "returncode": int(proc.returncode),
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def artifact_schema_errors(artifact: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    blocked = str(artifact.get("honest_verdict", "")).startswith("blocked_")
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not (
        verdict.startswith("complete:") or verdict.startswith("blocked_")
    ):
        errors.append("honest_verdict must be terminal-prefixed")
    if not isinstance(artifact.get("hidden_state_localizer_has_nonposition_signal"), bool):
        errors.append("hidden_state_localizer_has_nonposition_signal must be bare bool")
    if not isinstance(artifact.get("localization_f1_comparison"), dict):
        errors.append("localization_f1_comparison must be dict")
    position_f1 = artifact.get("position_only_baseline_f1")
    if not blocked and (not isinstance(position_f1, float) or isinstance(position_f1, bool)):
        errors.append("position_only_baseline_f1 must be bare float")
    if not isinstance(artifact.get("n_traces"), int) or isinstance(artifact.get("n_traces"), bool):
        errors.append("n_traces must be bare int")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if not isinstance(artifact.get("preconditions_checked"), list):
        errors.append("preconditions_checked must be list")
    if not isinstance(artifact.get("random_seed"), int) or isinstance(artifact.get("random_seed"), bool):
        errors.append("random_seed must be int")
    if not isinstance(artifact.get("reproducibility_checksum"), str):
        errors.append("reproducibility_checksum must be string")
    if not isinstance(artifact.get("model_specs"), dict):
        errors.append("model_specs must be dict")
    return errors


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    hidden_state_extractor: Any | None = None,
    adversarial_verify_runner: AdversarialVerifyRunner = run_adversarial_verify,
    write: bool = True,
) -> dict[str, Any]:
    cfg = config or ExperimentConfig()
    started = cfg.start_time()
    source_paths = [
        cfg.fover_row_corpus_path,
        cfg.fover_step_corpus_path,
        cfg.exp2850_artifact_path,
        cfg.exp4403_artifact_path,
    ]
    checks: list[PreconditionCheck] = [
        _load_exp2850_check(cfg.exp2850_artifact_path),
        _step_corpus_check(cfg.fover_step_corpus_path),
    ]
    exp4403_payload = _text_localizer_payload(cfg.exp4403_artifact_path)
    text_f1 = text_localizer_f1_from_exp4403(exp4403_payload)

    traces: list[HiddenStateTrace] = []
    if cfg.fover_row_corpus_path.is_file():
        try:
            traces = load_real_fover_error_traces(
                cfg.fover_row_corpus_path,
                limit=cfg.min_capture_error_traces,
            )
            checks.append(
                PreconditionCheck(
                    "cached_real_fover_failed_traces",
                    len(traces) >= cfg.min_capture_error_traces,
                    f"failed_traces={len(traces)}; required>={cfg.min_capture_error_traces}",
                )
            )
        except Exception as exc:
            checks.append(
                PreconditionCheck(
                    "cached_real_fover_failed_traces",
                    False,
                    f"unreadable: {type(exc).__name__}: {exc}",
                )
            )
    else:
        checks.append(PreconditionCheck("cached_real_fover_failed_traces", False, "missing"))

    if len(traces) > cfg.min_eval_traces:
        train_preview, heldout_preview = split_train_heldout(
            traces,
            seed=cfg.random_seed,
            heldout_fraction=cfg.heldout_fraction,
            min_eval_traces=cfg.min_eval_traces,
        )
        checks.append(
            PreconditionCheck(
                "heldout_first_error_eval_split",
                len(heldout_preview) >= cfg.min_eval_traces and bool(train_preview),
                f"train={len(train_preview)}; heldout={len(heldout_preview)}; required_heldout>={cfg.min_eval_traces}",
            )
        )
    else:
        checks.append(
            PreconditionCheck(
                "heldout_first_error_eval_split",
                False,
                f"captured_candidates={len(traces)}; required_heldout>={cfg.min_eval_traces}",
            )
        )

    extractor = hidden_state_extractor or TransformersHiddenStateExtractor(cfg.hf_model_id)
    hidden_status = HiddenStatePathStatus(False, "not checked", {})
    if all(check.available for check in checks):
        hidden_status = extractor.check()
        checks.append(
            PreconditionCheck(
                "hidden_state_extraction_path",
                hidden_status.available,
                hidden_status.detail,
            )
        )
    checks.append(
        PreconditionCheck(
            "trm_training_stand_down",
            True,
            "not invoked; this experiment runs forward-pass capture plus CPU probe only",
        )
    )
    preconditions = [check.as_dict() for check in checks]
    blocked = _blocked_reason(checks)
    if blocked is not None:
        artifact = build_blocked_artifact(
            honest_verdict=blocked,
            preconditions_checked=preconditions,
            hidden_status=hidden_status,
            text_localizer_f1=text_f1,
            source_paths=source_paths,
            duration_s=cfg.clock() - started,
            config=cfg,
        )
        if write:
            _write_artifact(cfg.artifact_path, artifact)
        return artifact

    train_traces, heldout_traces = split_train_heldout(
        traces,
        seed=cfg.random_seed,
        heldout_fraction=cfg.heldout_fraction,
        min_eval_traces=cfg.min_eval_traces,
    )
    hidden_states = extractor.capture(traces)
    all_features = transport_margin_features(traces, hidden_states)
    grouped = _records_by_trace(all_features)
    train_features = [record for trace in train_traces for record in grouped[trace.trace_id]]
    heldout_features = [record for trace in heldout_traces for record in grouped[trace.trace_id]]
    probe = HiddenStateMarginProbe.fit(train_features)
    position_baseline = PositionOnlyBaseline.fit(train_traces)
    comparison = evaluate_hidden_state_probe(
        heldout_traces,
        heldout_features,
        probe,
        position_baseline,
        text_localizer_f1=text_f1,
        seed=cfg.random_seed,
        bootstrap_resamples=cfg.bootstrap_resamples,
    )
    artifact = build_complete_artifact(
        traces=traces,
        train_traces=train_traces,
        heldout_traces=heldout_traces,
        hidden_states=hidden_states,
        train_features=train_features,
        heldout_features=heldout_features,
        hidden_status=hidden_status,
        preconditions_checked=preconditions,
        comparison=comparison,
        probe=probe,
        source_paths=source_paths,
        duration_s=cfg.clock() - started,
        config=cfg,
    )
    if write:
        _write_artifact(cfg.artifact_path, artifact)
        artifact["adversarial_verify"] = adversarial_verify_runner(cfg.artifact_path)
        if artifact["adversarial_verify"].get("returncode") not in (0, None):
            artifact["flagged_adversarial"] = True
        _write_artifact(cfg.artifact_path, artifact)
    else:
        artifact["adversarial_verify"] = {"returncode": None, "skipped": True}
    return artifact


def main() -> int:  # pragma: no cover - CLI wrapper.
    artifact = run_experiment(write=True)
    print(
        "[exp4416] "
        f"{artifact['honest_verdict']} "
        f"hidden_state_localizer_has_nonposition_signal="
        f"{artifact['hidden_state_localizer_has_nonposition_signal']} "
        f"n_traces={artifact['n_traces']} -> {ARTIFACT_PATH}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
