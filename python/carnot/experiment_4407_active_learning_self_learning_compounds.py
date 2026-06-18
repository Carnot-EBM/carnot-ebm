"""Exp 4407: active-learning first-error localizer compounding.

Spec refs: REQ-VERIFY-4407, SCENARIO-VERIFY-4407.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot import experiment_4392_verifiable_process_data_localizer as exp4392
from carnot import experiment_4393_localizer_skeptic_proof as exp4393
from carnot import experiment_4403_real_intervention_localizer_deconfound as exp4403


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = ROOT / "results" / "experiment_4407_active_learning_self_learning_compounds.json"
EXP4403_ARTIFACT_PATH = ROOT / "results" / "experiment_4403_real_intervention_localizer_deconfound.json"
FOVER_ROW_CORPUS_PATH = ROOT / "data" / "fover_corpus.jsonl"
FOVER_STEP_CORPUS_PATH = ROOT / "data" / "step_level_prm_training.jsonl"
EXP4381_ARTIFACT_PATH = ROOT / "results" / "experiment_4381_biprm_detector_localization_abstention.json"
VERIFIER_REGISTRY_PATH = ROOT / "ops" / "verifier_registry.yaml"
VERIFIER_GAPS_PATH = ROOT / "ops" / "verifier_gaps.md"

RANDOM_SEED = 4407
RANDOM_SEEDS_USED = (4407,)
BOOTSTRAP_RESAMPLES = 2500
MIN_LABEL_COUNT = 1000
MIN_HELD_OUT_TRACES = 30
HELDOUT_FAMILY = "gsm8k"
HEADROOM_EPSILON = 0.02
INFERENCE_SUBSTRATE = exp4392.INFERENCE_SUBSTRATE
SPEC_REFS = ["REQ-VERIFY-4407", "SCENARIO-VERIFY-4407"]

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "localizer_compounds",
    "active_vs_random_learning_curve",
    "compounding_delta_ci95",
    "verifier_is_oracle",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "model_specs",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A win (active selection compounds beyond random) "
        "and a clean null (the localizer signal is saturated/position-bound) "
        "are BOTH decision-grade."
    ),
    "localizer_compounds": (
        "BARE bool: the capstone reads this; true iff held-out first-error F1 "
        "under ACTIVE selection rises with corpus size beyond a RANDOM-selection "
        "baseline (delta CI95-excl-0) with positive-control headroom AND the "
        "position-only control stays beaten -- the self-learning reading on the "
        "axis that now has real headroom (NEW mechanism vs the .406 size-only "
        "saturated null)."
    ),
    "active_vs_random_learning_curve": (
        "list of {corpus_size, f1_active, f1_random, "
        "f1_positive_control_ceiling, position_only_floor} -- the "
        "ACTIVE-vs-RANDOM curve; ACTIVE rising above RANDOM and the "
        "position-only floor is the compounding signal, both flat = saturated."
    ),
    "compounding_delta_ci95": (
        "Bootstrap CI95 of the ACTIVE-minus-RANDOM held-out-F1 delta at the "
        "largest corpus size -- excluding 0 is the compounding claim."
    ),
    "verifier_is_oracle": "BARE bool=false -- the learned localizer is oracle-distinct.",
    "preconditions_checked": (
        "Records the localizer/corpus + TRM-stand-down verified; pre-empts the "
        "silent-missing-resource fabrication mode."
    ),
    "random_seed": (
        "Determinism precondition for the active/random selection + the "
        "localizer fit + the bootstrap."
    ),
    "reproducibility_checksum": (
        "Hash of the corpus + the active-selection config + the curve; lets a "
        "third party re-run."
    ),
    "model_specs": (
        "The localizer + the FoVer corpus + the active-selection config + the "
        "positive control + n; required methodology + the oracle-distinct "
        "declaration."
    ),
}


@dataclass(frozen=True)
class TraceLabel:
    """One labeled trace available to the active-learning selector."""

    trace: exp4392.ProcessTrace
    family: str
    position_bin: str
    source: str


@dataclass(frozen=True)
class PreconditionCheck:
    """One prerequisite checked before curve measurement starts."""

    resource: str
    available: bool
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {"resource": self.resource, "available": bool(self.available), "detail": self.detail}


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime configuration for Exp 4407."""

    repo_root: Path = ROOT
    exp4403_artifact_path: Path = EXP4403_ARTIFACT_PATH
    fover_row_corpus_path: Path = FOVER_ROW_CORPUS_PATH
    fover_step_corpus_path: Path = FOVER_STEP_CORPUS_PATH
    exp4381_artifact_path: Path = EXP4381_ARTIFACT_PATH
    verifier_registry_path: Path = VERIFIER_REGISTRY_PATH
    verifier_gaps_path: Path = VERIFIER_GAPS_PATH
    artifact_path: Path = ARTIFACT_PATH
    heldout_family: str = HELDOUT_FAMILY
    min_label_count: int = MIN_LABEL_COUNT
    min_held_out_traces: int = MIN_HELD_OUT_TRACES
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES
    random_seed: int = RANDOM_SEED
    started_at: float | None = None
    clock: Callable[[], float] = time.time

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at


RegistryChecker = Callable[[Path], bool]
AdversarialVerifyRunner = Callable[[Path], dict[str, Any]]


def _round_float(value: float | None, digits: int = 6) -> float | None:
    return exp4392.round_float(value, digits=digits)


def _write_artifact(path: Path, artifact: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json_dict(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _has_exp4403_localizer(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    localizer = payload.get("model_specs", {}).get("localizer")
    return isinstance(localizer, dict) and isinstance(localizer.get("weights"), dict)


def _baseline_available(path: Path) -> bool:
    payload = _read_json_dict(path)
    if not payload:
        return False
    directions = payload.get("localization_f1_by_direction")
    if isinstance(directions, dict):
        fused = directions.get("bidirectional_fusion")
        if isinstance(fused, dict) and fused.get("f1") is not None:
            return True
    return bool(payload.get("model_specs"))


def _position_bin(trace: exp4392.ProcessTrace) -> str:
    return "clean" if trace.first_error_index is None else str(int(trace.first_error_index))


def labels_from_exp4403_rows(rows: Sequence[dict[str, Any]]) -> list[TraceLabel]:
    labels = exp4403.build_fover_intervention_labels_from_rows(rows)
    return [
        TraceLabel(
            trace=label.trace,
            family=label.family,
            position_bin=label.position_bin,
            source="exp4403_real_intervention",
        )
        for label in labels
    ]


def labels_from_fover_step_rows(rows: Sequence[dict[str, Any]]) -> list[TraceLabel]:
    traces = exp4392.load_fover_real_traces_from_rows(rows)
    return [
        TraceLabel(
            trace=trace,
            family=trace.error_class or trace.source_domain,
            position_bin=_position_bin(trace),
            source="real_fover_step_fallback",
        )
        for trace in traces
    ]


def _error_labels(labels: Sequence[TraceLabel]) -> list[TraceLabel]:
    return [label for label in labels if label.trace.first_error_index is not None]


def _label_count_detail(labels: Sequence[TraceLabel]) -> str:
    errors = _error_labels(labels)
    families = Counter(label.family for label in errors)
    positions = Counter(label.position_bin for label in errors)
    return (
        f"labels={len(labels)}; error_labels={len(errors)}; "
        f"families={dict(sorted(families.items()))}; positions={dict(sorted(positions.items()))}"
    )


def split_train_heldout(
    labels: Sequence[TraceLabel],
    *,
    heldout_family: str,
    seed: int,
) -> tuple[list[TraceLabel], list[TraceLabel]]:
    train = [label for label in labels if label.family != heldout_family]
    heldout = [label for label in labels if label.family == heldout_family]
    if train and heldout:
        return train, heldout
    ordered = sorted(labels, key=lambda label: label.trace.trace_id)
    rng = random.Random(seed)
    shuffled = list(ordered)
    rng.shuffle(shuffled)
    cut = max(1, len(shuffled) // 4)
    return shuffled[cut:], shuffled[:cut]


def _prefix_sizes(total: int, *, min_prefix_size: int = 1) -> list[int]:
    if total <= 0:
        return []
    fractions = (0.10, 0.25, 0.50, 1.0)
    sizes = {
        min(total, max(1, int(round(total * fraction)), min_prefix_size))
        for fraction in fractions
    }
    sizes.add(total)
    return sorted(sizes)


def _train_localizer(labels: Sequence[TraceLabel]) -> exp4392.LocalizerModel:
    return exp4392.train_contrastive_localizer([label.trace for label in labels])


def _first_error_successes(
    labels: Sequence[TraceLabel],
    predict: Callable[[exp4392.ProcessTrace], int | None],
) -> list[int]:
    successes: list[int] = []
    for label in labels:
        if label.trace.first_error_index is None:
            continue
        successes.append(int(predict(label.trace) == label.trace.first_error_index))
    return successes


def _f1(successes: Sequence[int]) -> float:
    return sum(int(value) for value in successes) / len(successes) if successes else 0.0


def first_error_f1(labels: Sequence[TraceLabel], model: exp4392.LocalizerModel) -> float:
    return _f1(_first_error_successes(labels, model.predict_first_error_index))


def position_only_f1(labels: Sequence[TraceLabel], train_labels: Sequence[TraceLabel]) -> float:
    baseline = exp4393.PositionOnlyBaseline.fit([label.trace for label in train_labels])
    return _f1(_first_error_successes(labels, baseline.predict_first_error_index))


def _random_select(labels: Sequence[TraceLabel], size: int, *, seed: int) -> list[TraceLabel]:
    ordered = sorted(labels, key=lambda label: label.trace.trace_id)
    rng = random.Random(seed)
    shuffled = list(ordered)
    rng.shuffle(shuffled)
    return shuffled[: min(size, len(shuffled))]


def active_select_labels(labels: Sequence[TraceLabel], size: int) -> list[TraceLabel]:
    pool = sorted(labels, key=lambda label: label.trace.trace_id)
    selected: list[TraceLabel] = []
    remaining = list(pool)
    while remaining and len(selected) < size:
        model = _train_localizer(selected)
        position_counts = Counter(label.position_bin for label in selected)
        choice = min(
            remaining,
            key=lambda label: (
                position_counts[label.position_bin],
                model.confidence_margin(label.trace),
                label.trace.trace_id,
            ),
        )
        selected.append(choice)
        remaining.remove(choice)
    return selected


def bootstrap_active_minus_random_ci95(
    held_out: Sequence[TraceLabel],
    active_model: exp4392.LocalizerModel | None,
    random_model: exp4392.LocalizerModel | None,
    *,
    seed: int,
    resamples: int,
) -> list[float | None]:
    if active_model is None or random_model is None or resamples <= 0:
        return [None, None]
    rows: list[tuple[int, int]] = []
    for label in held_out:
        if label.trace.first_error_index is None:
            continue
        rows.append(
            (
                int(active_model.predict_first_error_index(label.trace) == label.trace.first_error_index),
                int(random_model.predict_first_error_index(label.trace) == label.trace.first_error_index),
            )
        )
    if not rows:
        return [None, None]
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(resamples):
        delta_sum = 0
        for _idx in range(len(rows)):
            active_ok, random_ok = rows[rng.randrange(len(rows))]
            delta_sum += active_ok - random_ok
        values.append(delta_sum / len(rows))
    values.sort()
    lo = int(0.025 * (len(values) - 1))
    hi = int(0.975 * (len(values) - 1))
    return [_round_float(values[lo]), _round_float(values[hi])]


def build_active_vs_random_curve(
    train_pool: Sequence[TraceLabel],
    held_out: Sequence[TraceLabel],
    *,
    corpus_sizes: Sequence[int] | None = None,
    seed: int,
) -> tuple[list[dict[str, Any]], list[exp4392.LocalizerModel], list[exp4392.LocalizerModel], dict[str, Any]]:
    error_pool = _error_labels(train_pool)
    sizes = list(corpus_sizes) if corpus_sizes is not None else _prefix_sizes(len(error_pool))
    positive_control_model = _train_localizer(error_pool)
    positive_ceiling = first_error_f1(held_out, positive_control_model)
    curve: list[dict[str, Any]] = []
    active_models: list[exp4392.LocalizerModel] = []
    random_models: list[exp4392.LocalizerModel] = []
    for raw_size in sizes:
        size = min(max(1, int(raw_size)), len(error_pool))
        active_selected = active_select_labels(error_pool, size)
        random_selected = _random_select(error_pool, size, seed=seed)
        active_model = _train_localizer(active_selected)
        random_model = _train_localizer(random_selected)
        active_models.append(active_model)
        random_models.append(random_model)
        curve.append(
            {
                "corpus_size": int(size),
                "f1_active": float(_round_float(first_error_f1(held_out, active_model)) or 0.0),
                "f1_random": float(_round_float(first_error_f1(held_out, random_model)) or 0.0),
                "f1_positive_control_ceiling": float(_round_float(positive_ceiling) or 0.0),
                "position_only_floor": float(_round_float(position_only_f1(held_out, active_selected)) or 0.0),
            }
        )
    positive_control = {
        "positive_control_passed": bool(
            curve and positive_ceiling > float(curve[-1]["f1_random"]) + HEADROOM_EPSILON
        ),
        "ceiling_held_out_f1": float(_round_float(positive_ceiling) or 0.0),
        "headroom_over_random_at_largest_size": float(
            _round_float(positive_ceiling - float(curve[-1]["f1_random"])) if curve else 0.0
        ),
        "minimum_nontrivial_headroom": float(HEADROOM_EPSILON),
    }
    return curve, active_models, random_models, positive_control


def summarize_compounding(
    learning_curve: Sequence[dict[str, Any]],
    *,
    compounding_delta_ci95: Sequence[float | None],
) -> dict[str, bool]:
    if not learning_curve:
        return {
            "localizer_compounds": False,
            "active_rises_beyond_random": False,
            "delta_ci95_excludes_zero": False,
            "positive_control_headroom": False,
            "position_only_control_beaten": False,
        }
    first = learning_curve[0]
    final = learning_curve[-1]
    active_gain = float(final["f1_active"]) - float(first["f1_active"])
    random_gain = float(final["f1_random"]) - float(first["f1_random"])
    active_rises = active_gain > random_gain and float(final["f1_active"]) > float(final["f1_random"])
    ci_positive = bool(compounding_delta_ci95[0] is not None and float(compounding_delta_ci95[0]) > 0.0)
    headroom = float(final["f1_positive_control_ceiling"]) > float(final["f1_random"]) + HEADROOM_EPSILON
    position_beaten = float(final["f1_active"]) > float(final["position_only_floor"]) + HEADROOM_EPSILON
    compounds = bool(active_rises and ci_positive and headroom and position_beaten)
    return {
        "localizer_compounds": compounds,
        "active_rises_beyond_random": bool(active_rises),
        "delta_ci95_excludes_zero": bool(ci_positive),
        "positive_control_headroom": bool(headroom),
        "position_only_control_beaten": bool(position_beaten),
    }


def _checksum(source_paths: Sequence[Path], payload: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for path in sorted({Path(path) for path in source_paths}, key=lambda item: str(item)):
        digest.update(str(path).encode("utf-8"))
        if not path.exists():
            digest.update(b"\0MISSING\0")
        else:
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))
    return "sha256:" + digest.hexdigest()


def _split_spec(
    *,
    corpus_source: str,
    train_labels: Sequence[TraceLabel],
    held_out_labels: Sequence[TraceLabel],
    heldout_family: str,
) -> dict[str, Any]:
    return {
        "split_axis": "template_family_holdout",
        "corpus_source": corpus_source,
        "heldout_family": heldout_family,
        "train_trace_count": int(len(train_labels)),
        "train_error_trace_count": int(len(_error_labels(train_labels))),
        "held_out_trace_count": int(len(held_out_labels)),
        "held_out_error_trace_count": int(len(_error_labels(held_out_labels))),
        "train_position_distribution": dict(sorted(Counter(label.position_bin for label in _error_labels(train_labels)).items())),
        "held_out_position_distribution": dict(sorted(Counter(label.position_bin for label in _error_labels(held_out_labels)).items())),
    }


def _model_specs(
    *,
    active_models: Sequence[exp4392.LocalizerModel],
    random_models: Sequence[exp4392.LocalizerModel],
    positive_control: dict[str, Any],
    split_spec: dict[str, Any],
    corpus_source: str,
    source_paths: Sequence[Path],
    bootstrap_resamples: int,
) -> dict[str, Any]:
    return {
        "localizer": "contrastive_feature_difference_localizer_refit_per_selection",
        "corpus_source": corpus_source,
        "fover_row_corpus": str(source_paths[1]) if len(source_paths) > 1 else str(FOVER_ROW_CORPUS_PATH),
        "fover_step_corpus": str(source_paths[2]) if len(source_paths) > 2 else str(FOVER_STEP_CORPUS_PATH),
        "exp4403_artifact": str(source_paths[0]) if source_paths else str(EXP4403_ARTIFACT_PATH),
        "active_selection_config": {
            "selection_rule": "min(position_count, confidence_margin, trace_id)",
            "uncertainty_signal": "smallest localizer confidence margin",
            "diversity_signal": "under-represented first-error position_bin",
            "random_baseline": "uniform shuffled trace addition with fixed seed",
            "corpus_sizes": [model.training_summary.get("training_trace_count") for model in active_models],
        },
        "positive_control": positive_control,
        "split": split_spec,
        "active_models": [model.as_dict() for model in active_models],
        "random_models": [model.as_dict() for model in random_models],
        "bootstrap_resamples": int(bootstrap_resamples),
        "trm_training": "stood_down_not_invoked",
        "generator_training": "stood_down_not_invoked",
        "live_generation": False,
        "verifier_is_oracle": False,
    }


def _missing_gap(artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "gap_id": "GAP-4407-ACTIVE-LOCALIZER-POSITION-BOUND",
        "status": "open",
        "evidence": "results/experiment_4407_active_learning_self_learning_compounds.json",
        "failure_mode": artifact["honest_verdict"],
        "missing_discriminator": (
            "Non-degenerate multi-position real intervention labels with content "
            "features that beat a position-only first-error baseline."
        ),
        "candidate_design": (
            "Collect multi-step FoVer interventions with non-empty suffix redirects "
            "and typed first-error families before retrying active trace selection."
        ),
        "priority": "high",
    }


def append_missing_verifier_gap(path: Path, gap: dict[str, Any]) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if gap["gap_id"] in existing:
        return
    text = "\n".join(
        [
            f"### {gap['gap_id']}: Exp 4407 active localizer residual",
            f"- status: {gap['status']}",
            f"- evidence: `{gap['evidence']}`.",
            f"- failure mode: {gap['failure_mode']}",
            f"- missing discriminator: {gap['missing_discriminator']}",
            f"- candidate design: {gap['candidate_design']}",
            f"- priority: {gap['priority']}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(existing.rstrip() + "\n\n" + text, encoding="utf-8")


def build_complete_artifact(
    *,
    active_vs_random_learning_curve: Sequence[dict[str, Any]],
    active_models: Sequence[exp4392.LocalizerModel],
    random_models: Sequence[exp4392.LocalizerModel],
    positive_control: dict[str, Any],
    compounding_delta_ci95: Sequence[float | None],
    preconditions_checked: list[dict[str, Any]],
    source_paths: Sequence[Path],
    split_spec: dict[str, Any],
    duration_s: float,
    bootstrap_resamples: int,
    random_seed: int,
    corpus_source: str,
) -> dict[str, Any]:
    summary = summarize_compounding(
        active_vs_random_learning_curve,
        compounding_delta_ci95=compounding_delta_ci95,
    )
    if summary["localizer_compounds"]:
        verdict = "success: active_selection_localizer_compounds_beyond_random"
    elif active_vs_random_learning_curve:
        verdict = "complete: clean_null_position_bound_or_saturated"
    else:
        verdict = "complete: clean_null_active_not_beyond_random"
    checksum_payload = {
        "curve": list(active_vs_random_learning_curve),
        "positive_control": positive_control,
        "compounding_delta_ci95": list(compounding_delta_ci95),
        "split_spec": split_spec,
        "corpus_source": corpus_source,
        "random_seed": random_seed,
    }
    artifact = {
        "experiment": "experiment_4407_active_learning_self_learning_compounds",
        "schema": "carnot.active_learning_localizer_compounds.v1",
        "honest_verdict": verdict,
        "localizer_compounds": bool(summary["localizer_compounds"]),
        "active_vs_random_learning_curve": [dict(point) for point in active_vs_random_learning_curve],
        "compounding_delta_ci95": list(compounding_delta_ci95),
        "positive_control_passed": bool(positive_control.get("positive_control_passed")),
        "verifier_is_oracle": False,
        "preconditions_checked": preconditions_checked,
        "random_seed": int(random_seed),
        "random_seeds_used": list(RANDOM_SEEDS_USED),
        "bootstrap_resamples": int(bootstrap_resamples),
        "reproducibility_checksum": _checksum(source_paths, checksum_payload),
        "model_specs": _model_specs(
            active_models=active_models,
            random_models=random_models,
            positive_control=positive_control,
            split_spec=split_spec,
            corpus_source=corpus_source,
            source_paths=source_paths,
            bootstrap_resamples=bootstrap_resamples,
        ),
        "gate_summary": summary,
        "missing_verifier_gaps": [],
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "methodology_note": (
            "CPU-only cached active-learning selection over first-error labels. "
            "The ACTIVE arm uses localizer uncertainty plus first-error-position "
            "diversity; RANDOM is fixed-seed uniform trace addition. No TRM "
            "training, generator training, live LLM inference, or quota-bearing "
            "call is invoked."
        ),
        "duration_s": _round_float(duration_s, digits=3),
        "inference_substrate": INFERENCE_SUBSTRATE,
    }
    if not artifact["localizer_compounds"]:
        artifact["missing_verifier_gaps"] = [_missing_gap(artifact)]
    return artifact


def build_blocked_artifact(
    *,
    preconditions_checked: list[dict[str, Any]],
    source_paths: Sequence[Path],
    duration_s: float,
    random_seed: int,
) -> dict[str, Any]:
    return {
        "experiment": "experiment_4407_active_learning_self_learning_compounds",
        "schema": "carnot.active_learning_localizer_compounds.v1",
        "honest_verdict": "blocked_no_localizer_or_corpus",
        "localizer_compounds": False,
        "active_vs_random_learning_curve": [],
        "compounding_delta_ci95": [None, None],
        "positive_control_passed": False,
        "verifier_is_oracle": False,
        "preconditions_checked": preconditions_checked,
        "random_seed": int(random_seed),
        "random_seeds_used": list(RANDOM_SEEDS_USED),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "reproducibility_checksum": _checksum(
            source_paths,
            {"blocked": "blocked_no_localizer_or_corpus", "random_seed": random_seed},
        ),
        "model_specs": {
            "blocked_reason": "no usable Exp 4403 localizer/corpus or REAL FoVer fallback corpus",
            "trm_training": "stood_down_not_invoked",
            "generator_training": "stood_down_not_invoked",
            "live_generation": False,
            "verifier_is_oracle": False,
        },
        "missing_verifier_gaps": [],
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "methodology_note": "blocked before curve measurement; no active/random metrics fabricated",
        "duration_s": _round_float(duration_s, digits=3),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "adversarial_verify": {"skipped": "blocked"},
    }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def artifact_schema_errors(artifact: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(("success:", "complete:", "blocked_")):
        errors.append("honest_verdict must be terminal-prefixed")
    if type(artifact.get("localizer_compounds")) is not bool:
        errors.append("localizer_compounds must be a bare bool")
    curve = artifact.get("active_vs_random_learning_curve")
    if not isinstance(curve, list):
        errors.append("active_vs_random_learning_curve must be a list")
    else:
        for point in curve:
            if not isinstance(point, dict):
                errors.append("curve points must be objects")
                continue
            for field in ("corpus_size", "f1_active", "f1_random", "f1_positive_control_ceiling", "position_only_floor"):
                if field not in point:
                    errors.append(f"curve point missing {field}")
            if type(point.get("corpus_size")) is not int:
                errors.append("corpus_size must be a bare int")
    ci95 = artifact.get("compounding_delta_ci95")
    if not (isinstance(ci95, list) and len(ci95) == 2):
        errors.append("compounding_delta_ci95 must be a two-element list")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be the bare bool false")
    if not isinstance(artifact.get("preconditions_checked"), list):
        errors.append("preconditions_checked must be a list")
    if type(artifact.get("random_seed")) is not int:
        errors.append("random_seed must be a bare int")
    if not isinstance(artifact.get("reproducibility_checksum"), str):
        errors.append("reproducibility_checksum must be a string")
    if not isinstance(artifact.get("model_specs"), dict):
        errors.append("model_specs must be an object")
    principles = artifact.get("field_principles")
    if not isinstance(principles, dict):
        errors.append("field_principles must be an object")
    else:
        for field in REQUIRED_ARTIFACT_FIELDS:
            if principles.get(field) != FIELD_PRINCIPLES[field]:
                errors.append(f"field_principles mismatch for {field}")
    if artifact.get("localizer_compounds") is True:
        if not ci95 or ci95[0] is None or float(ci95[0]) <= 0.0:
            errors.append("localizer_compounds requires positive compounding_delta_ci95")
        if isinstance(curve, list) and curve:
            final = curve[-1] if isinstance(curve[-1], dict) else {}
            if not _is_number(final.get("f1_active")) or not _is_number(final.get("f1_random")):
                errors.append("localizer_compounds requires numeric active/random F1")
            elif float(final["f1_active"]) <= float(final["f1_random"]):
                errors.append("localizer_compounds requires active final F1 above random")
    return errors


def check_preconditions(
    config: ExperimentConfig,
    *,
    registry_checker: RegistryChecker,
) -> tuple[list[PreconditionCheck], str | None, list[TraceLabel]]:
    exp4403_payload = _read_json_dict(config.exp4403_artifact_path)
    exp4403_localizer = _has_exp4403_localizer(exp4403_payload)
    labels: list[TraceLabel] = []
    fallback_labels: list[TraceLabel] = []
    if config.fover_row_corpus_path.is_file():
        labels = labels_from_exp4403_rows(_read_jsonl(config.fover_row_corpus_path))
    if config.fover_step_corpus_path.is_file():
        fallback_labels = labels_from_fover_step_rows(_read_jsonl(config.fover_step_corpus_path))
    primary_available = exp4403_localizer and len(_error_labels(labels)) >= config.min_label_count
    fallback_available = len(_error_labels(fallback_labels)) >= config.min_label_count
    try:
        registry_ok = bool(registry_checker(config.verifier_registry_path))
    except Exception as exc:
        registry_ok = False
        registry_detail = f"registry check failed: {exc}"
    else:
        registry_detail = "fover_production_ensemble present" if registry_ok else "missing fover_production_ensemble"
    checks = [
        PreconditionCheck(
            "exp4403_real_intervention_localizer",
            exp4403_localizer,
            "fitted localizer present" if exp4403_localizer else "missing fitted localizer",
        ),
        PreconditionCheck(
            "exp4403_real_intervention_corpus",
            len(_error_labels(labels)) >= config.min_label_count,
            _label_count_detail(labels) if labels else "missing",
        ),
        PreconditionCheck(
            "real_fover_first_error_fallback_corpus",
            fallback_available,
            _label_count_detail(fallback_labels) if fallback_labels else "missing",
        ),
        PreconditionCheck(
            "exp4381_ensemble_baseline",
            _baseline_available(config.exp4381_artifact_path),
            f"baseline={exp4392.ENSEMBLE_BASELINE_F1:.3f}" if _baseline_available(config.exp4381_artifact_path) else "missing",
        ),
        PreconditionCheck("verifier_registry", registry_ok, registry_detail),
        PreconditionCheck("trm_training_stand_down", True, "not invoked; cached contrastive localizer fit only"),
    ]
    if primary_available:
        return checks, "exp4403_real_intervention", labels
    if fallback_available:
        return checks, "real_fover_first_error_fallback", fallback_labels
    return checks, None, []


def run_adversarial_verify(path: Path, repo_root: Path = ROOT) -> dict[str, Any]:  # pragma: no cover
    script = repo_root / "scripts" / "adversarial_verify.py"
    if not script.is_file():
        return {"returncode": None, "stderr": "scripts/adversarial_verify.py missing"}
    proc = subprocess.run(
        [sys.executable, str(script), str(path)],
        cwd=str(repo_root),
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    return {
        "returncode": int(proc.returncode),
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    registry_checker: RegistryChecker = exp4392._registry_has_fover_ensemble,
    adversarial_verify_runner: AdversarialVerifyRunner = run_adversarial_verify,
    write: bool = True,
) -> dict[str, Any]:
    cfg = config or ExperimentConfig()
    started = cfg.start_time()
    source_paths = [
        cfg.exp4403_artifact_path,
        cfg.fover_row_corpus_path,
        cfg.fover_step_corpus_path,
        cfg.exp4381_artifact_path,
        cfg.verifier_registry_path,
    ]
    checks, corpus_source, labels = check_preconditions(cfg, registry_checker=registry_checker)
    preconditions = [check.as_dict() for check in checks]
    if corpus_source is None:
        artifact = build_blocked_artifact(
            preconditions_checked=preconditions,
            source_paths=source_paths,
            duration_s=cfg.clock() - started,
            random_seed=cfg.random_seed,
        )
        if write:
            _write_artifact(cfg.artifact_path, artifact)
        return artifact
    train_labels, held_out_labels = split_train_heldout(
        labels,
        heldout_family=cfg.heldout_family,
        seed=cfg.random_seed,
    )
    if len(_error_labels(held_out_labels)) < cfg.min_held_out_traces:
        artifact = build_blocked_artifact(
            preconditions_checked=preconditions
            + [
                PreconditionCheck(
                    "heldout_family_eval_split",
                    False,
                    f"heldout_error_traces={len(_error_labels(held_out_labels))}; required>={cfg.min_held_out_traces}",
                ).as_dict()
            ],
            source_paths=source_paths,
            duration_s=cfg.clock() - started,
            random_seed=cfg.random_seed,
        )
        if write:
            _write_artifact(cfg.artifact_path, artifact)
        return artifact
    curve, active_models, random_models, positive_control = build_active_vs_random_curve(
        train_labels,
        held_out_labels,
        seed=cfg.random_seed,
    )
    delta_ci95 = bootstrap_active_minus_random_ci95(
        held_out_labels,
        active_models[-1] if active_models else None,
        random_models[-1] if random_models else None,
        seed=cfg.random_seed,
        resamples=cfg.bootstrap_resamples,
    )
    split_spec = _split_spec(
        corpus_source=corpus_source,
        train_labels=train_labels,
        held_out_labels=held_out_labels,
        heldout_family=cfg.heldout_family,
    )
    artifact = build_complete_artifact(
        active_vs_random_learning_curve=curve,
        active_models=active_models,
        random_models=random_models,
        positive_control=positive_control,
        compounding_delta_ci95=delta_ci95,
        preconditions_checked=preconditions,
        source_paths=source_paths,
        split_spec=split_spec,
        duration_s=cfg.clock() - started,
        bootstrap_resamples=cfg.bootstrap_resamples,
        random_seed=cfg.random_seed,
        corpus_source=corpus_source,
    )
    if write:
        _write_artifact(cfg.artifact_path, artifact)
        artifact["adversarial_verify"] = adversarial_verify_runner(cfg.artifact_path)
        if artifact["adversarial_verify"].get("returncode") not in (0, None):
            artifact["flagged_adversarial"] = True
        _write_artifact(cfg.artifact_path, artifact)
        for gap in artifact.get("missing_verifier_gaps", []):
            append_missing_verifier_gap(cfg.verifier_gaps_path, gap)
    else:
        artifact["adversarial_verify"] = {"returncode": None, "skipped": True}
    return artifact


def main() -> int:  # pragma: no cover
    artifact = run_experiment(write=True)
    print(
        "[exp4407] "
        f"{artifact['honest_verdict']} "
        f"localizer_compounds={artifact['localizer_compounds']} "
        f"curve_points={len(artifact['active_vs_random_learning_curve'])} -> {ARTIFACT_PATH}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
