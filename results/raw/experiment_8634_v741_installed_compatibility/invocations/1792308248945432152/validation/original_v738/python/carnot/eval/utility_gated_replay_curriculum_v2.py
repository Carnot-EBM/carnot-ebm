"""Exp 2954 utility-gated replay policy for FR-11.

This module stays at the scheduler and memory layer. It reads checked-in
experiment artifacts, derives deterministic replay slices, and evaluates
whether a replay-weight update would improve held-out repair utility without
degrading a stable forgetting guard. No live model is invoked and no model
weights are changed.

Spec: REQ-LEARN-2954, SCENARIO-LEARN-2954,
SCENARIO-LEARN-2954-ROLLBACK.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260524"
OUTPUT_FILENAME = "experiment_2954_fr11_utility_gated_replay_curriculum_v2.json"
ARTIFACT = "experiment_2954_fr11_utility_gated_replay_curriculum_v2"
SCHEMA = "carnot.fr11.utility_gated_replay_curriculum.v2"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

EXP2947_REL_PATH = Path("results/experiment_2947_fr11_continuation_replay_curriculum_v1.json")
EXP2946_REL_PATH = Path("results/experiment_2946_sota_code_generation_continuation_v1.json")
EXP2940_REL_PATH = Path("results/experiment_2940_verifier_ensemble_auprc_code_corpora_v1.json")

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "continuous_self_learning_task",
    "self_learning_utility_artifact_ready",
    "source_artifacts",
    "replay_policies_compared",
    "heldout_utility_baseline",
    "heldout_utility_after",
    "heldout_utility_delta",
    "self_learning_utility_positive",
    "forgetting_guard_metric_before",
    "forgetting_guard_metric_after",
    "forgetting_guard_passed",
    "rollback_triggered",
    "update_rule",
    "inference_substrate",
    "duration_s",
)

TAXONOMY_ORDER = (
    "syntax_repair",
    "runtime_repair",
    "extraction_repair",
    "verified_pass",
)

EXP2947_BUCKET_TO_TAXONOMY = {
    "structural_memory_bootstrap": "syntax_repair",
    "process_reward_replay": "runtime_repair",
    "continuation_boundary_replay": "extraction_repair",
    "retention_guard_replay": "verified_pass",
}

ROW_STATUS_TO_ENERGY = {
    "candidate_passed": 0.0,
    "candidate_failed": 1.0,
    "candidate_syntax_failed": 2.0,
    "candidate_extraction_failed": 3.0,
}

ROW_STATUS_TO_TAXONOMY = {
    "candidate_passed": "verified_pass",
    "candidate_failed": "runtime_repair",
    "candidate_syntax_failed": "syntax_repair",
    "candidate_extraction_failed": "extraction_repair",
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime paths and clock used by the deterministic artifact builder."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    exp2947_path: Path = EXP2947_REL_PATH
    exp2946_path: Path = EXP2946_REL_PATH
    exp2940_path: Path = EXP2940_REL_PATH
    tests_run: Sequence[str] = field(default_factory=tuple)
    started_at: float | None = None
    clock: Callable[[], float] = time.time

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / OUTPUT_FILENAME


@dataclass(frozen=True)
class ReplayExample:
    """One candidate row converted into replay-taxonomy scoring evidence."""

    stable_id: str
    split: str
    taxonomy: str
    status_energy: float
    utility_signal: float
    forgetting_signal: float
    row_status: str
    random_seed: int


def build_artifact(config: ExperimentConfig | None = None) -> JsonDict:
    """Build the Exp 2954 artifact from checked-in upstream summaries."""

    config = config or ExperimentConfig()
    started = config.start_time()
    source_artifacts = _base_source_artifacts(config)
    missing_sources = _missing_required_sources(source_artifacts)
    if missing_sources:
        return _blocked_artifact(
            config,
            started,
            source_artifacts,
            "blocked_missing_upstream_artifact",
            [f"source:{source}" for source in missing_sources],
        )

    exp2947 = _read_json(_repo_path(config.repo_root, config.exp2947_path))
    exp2946 = _read_json(_repo_path(config.repo_root, config.exp2946_path))
    exp2940 = _read_json(_repo_path(config.repo_root, config.exp2940_path))
    protocol_rel = Path(str(exp2946.get("protocol_artifact_path", "")))
    protocol_source = _source_artifact(
        config.repo_root,
        protocol_rel,
        "exp2946_protocol",
        "repair_taxonomy_rows",
        ("candidate_results", "per_task_results"),
    )
    source_artifacts = [*source_artifacts, protocol_source]
    if not protocol_source["present"]:
        return _blocked_artifact(
            config,
            started,
            source_artifacts,
            "blocked_missing_exp2946_protocol_artifact",
            ["source:exp2946_protocol"],
        )

    protocol = _read_json(_repo_path(config.repo_root, protocol_rel))
    examples = replay_examples_from_rows(protocol.get("candidate_results", ()), exp2940)
    slices = split_examples(examples)
    if not all(slices.values()):
        return _blocked_artifact(
            config,
            started,
            source_artifacts,
            "blocked_empty_replay_slices",
            ["candidate_results"],
        )

    observed_taxonomies = _observed_taxonomies(examples)
    flat_weights = normalize_weights(dict.fromkeys(observed_taxonomies, 1.0))
    nonuniform_weights = nonuniform_weights_from_exp2947(exp2947, observed_taxonomies)
    guard_taxonomies = tuple(
        taxonomy
        for taxonomy in observed_taxonomies
        if any(example.taxonomy == taxonomy and example.forgetting_signal > 0.0 for example in slices["forgetting_guard"])
    )
    target_weights = target_weights_from_training(
        slices["train_replay"],
        baseline_weights=nonuniform_weights,
        guard_taxonomies=guard_taxonomies,
    )
    candidate_weights = blend_weights(nonuniform_weights, target_weights, learning_rate=0.65)
    decision = evaluate_policy_update(
        baseline_weights=nonuniform_weights,
        candidate_weights=candidate_weights,
        heldout_examples=slices["heldout_utility"],
        guard_examples=slices["forgetting_guard"],
    )
    flat_utility = policy_utility(flat_weights, slices["heldout_utility"])

    utility_positive = bool(decision["utility_improved"] and decision["forgetting_guard_passed"])
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "honest_verdict": _verdict(utility_positive, bool(decision["rollback_triggered"])),
        "continuous_self_learning_task": True,
        "self_learning_utility_artifact_ready": True,
        "source_artifacts": source_artifacts,
        "replay_policies_compared": [
            _policy_row("flat_replay", flat_weights, flat_utility, None, False),
            _policy_row(
                "nonuniform_replay_exp2947",
                nonuniform_weights,
                decision["heldout_utility_baseline"],
                decision["forgetting_guard_metric_before"],
                False,
            ),
            _policy_row(
                "utility_gated_replay",
                decision["accepted_weights"],
                decision["heldout_utility_after"],
                decision["forgetting_guard_metric_after"],
                bool(decision["utility_improved"] and decision["forgetting_guard_passed"]),
            ),
        ],
        "heldout_utility_baseline": decision["heldout_utility_baseline"],
        "heldout_utility_after": decision["heldout_utility_after"],
        "heldout_utility_delta": decision["heldout_utility_delta"],
        "self_learning_utility_positive": utility_positive,
        "forgetting_guard_metric_before": decision["forgetting_guard_metric_before"],
        "forgetting_guard_metric_after": decision["forgetting_guard_metric_after"],
        "forgetting_guard_passed": decision["forgetting_guard_passed"],
        "rollback_triggered": decision["rollback_triggered"],
        "update_rule": _update_rule(guard_taxonomies, exp2940),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": _elapsed(config, started),
        "slice_manifest": _slice_manifest(slices),
        "final_replay_weights": decision["accepted_weights"],
        "candidate_replay_weights": candidate_weights,
        "source_replay_distribution": exp2947.get("replay_count_distribution", {}),
        "soft_radial_projection_note": (
            "Soft-Radial Projection is future design context for differentiable "
            "constraint-preserving adaptation; it is not implemented or scored "
            "in this scheduler-only artifact."
        ),
        "model_weights_mutated": False,
        "live_model_invoked": False,
        "tests_run": list(config.tests_run),
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
        "missing_fields": [],
    }


def write_artifact(config: ExperimentConfig | None = None) -> JsonDict:
    """Build and persist the Exp 2954 artifact under ``results/``."""

    config = config or ExperimentConfig()
    artifact = build_artifact(config)
    output_path = config.artifact_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def replay_examples_from_rows(rows: object, exp2940: Mapping[str, Any]) -> tuple[ReplayExample, ...]:
    """Convert Exp 2946 candidate rows into verifier-energy replay examples."""

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return ()
    ppv = _positive_float(_mapping(exp2940.get("max_f1_operating_point")).get("ppv"), default=1.0)
    examples: list[ReplayExample] = []
    stable_ids = sorted({str(_mapping(row).get("stable_id", "")) for row in rows if _mapping(row).get("stable_id")})
    split_by_id = {
        stable_id: _split_name(index)
        for index, stable_id in enumerate(stable_ids)
    }
    for row_obj in rows:
        row = _mapping(row_obj)
        stable_id = str(row.get("stable_id", ""))
        row_status = str(row.get("row_status", ""))
        if stable_id not in split_by_id or row_status not in ROW_STATUS_TO_TAXONOMY:
            continue
        energy = ROW_STATUS_TO_ENERGY[row_status]
        taxonomy = ROW_STATUS_TO_TAXONOMY[row_status]
        repair_signal = energy / 3.0
        utility_signal = ppv * (0.10 if taxonomy == "verified_pass" else repair_signal)
        forgetting_signal = 1.0 if taxonomy == "verified_pass" else 0.0
        examples.append(
            ReplayExample(
                stable_id=stable_id,
                split=split_by_id[stable_id],
                taxonomy=taxonomy,
                status_energy=energy,
                utility_signal=_round(utility_signal),
                forgetting_signal=forgetting_signal,
                row_status=row_status,
                random_seed=int(row.get("random_seed", 0) or 0),
            )
        )
    return tuple(examples)


def split_examples(examples: Sequence[ReplayExample]) -> dict[str, tuple[ReplayExample, ...]]:
    """Group examples by the deterministic split assigned from stable task IDs."""

    return {
        "train_replay": tuple(example for example in examples if example.split == "train_replay"),
        "heldout_utility": tuple(example for example in examples if example.split == "heldout_utility"),
        "forgetting_guard": tuple(example for example in examples if example.split == "forgetting_guard"),
    }


def normalize_weights(weights: Mapping[str, float]) -> dict[str, float]:
    """Normalize positive replay weights into a deterministic probability table."""

    positive = {name: float(value) for name, value in sorted(weights.items()) if float(value) > 0.0}
    total = sum(positive.values())
    if total <= 0.0:
        raise ValueError("at least one positive replay weight is required")
    return {name: _round(value / total) for name, value in positive.items()}


def nonuniform_weights_from_exp2947(
    exp2947: Mapping[str, Any],
    observed_taxonomies: Sequence[str],
) -> dict[str, float]:
    """Map Exp 2947 replay buckets onto repair taxonomies."""

    distribution = _mapping(exp2947.get("replay_count_distribution"))
    mapped = {taxonomy: 0.0 for taxonomy in observed_taxonomies}
    for bucket, count in distribution.items():
        taxonomy = EXP2947_BUCKET_TO_TAXONOMY.get(str(bucket))
        if taxonomy in mapped:
            mapped[taxonomy] += _positive_float(count)
    if sum(mapped.values()) <= 0.0:
        mapped = dict.fromkeys(observed_taxonomies, 1.0)
    return normalize_weights(mapped)


def target_weights_from_training(
    examples: Sequence[ReplayExample],
    *,
    baseline_weights: Mapping[str, float],
    guard_taxonomies: Sequence[str],
) -> dict[str, float]:
    """Build candidate weights from train-slice utility while preserving guard mass."""

    baseline = normalize_weights(baseline_weights)
    utility_totals = {taxonomy: 0.0 for taxonomy in baseline}
    for example in examples:
        if example.taxonomy in utility_totals:
            utility_totals[example.taxonomy] += example.utility_signal
    guard_mass = sum(baseline.get(taxonomy, 0.0) for taxonomy in guard_taxonomies)
    candidate = {taxonomy: baseline[taxonomy] for taxonomy in guard_taxonomies if taxonomy in baseline}
    non_guard = {
        taxonomy: signal
        for taxonomy, signal in utility_totals.items()
        if taxonomy not in candidate and signal > 0.0
    }
    if not non_guard:
        return baseline
    scaled = normalize_weights(non_guard)
    for taxonomy, value in scaled.items():
        candidate[taxonomy] = value * max(0.0, 1.0 - guard_mass)
    return normalize_weights(candidate)


def blend_weights(
    baseline_weights: Mapping[str, float],
    target_weights: Mapping[str, float],
    *,
    learning_rate: float,
) -> dict[str, float]:
    """Move partway from baseline to target so one slice cannot dominate."""

    baseline = normalize_weights(baseline_weights)
    target = normalize_weights(target_weights)
    names = sorted(set(baseline) | set(target))
    blended = {
        name: (1.0 - learning_rate) * baseline.get(name, 0.0) + learning_rate * target.get(name, 0.0)
        for name in names
    }
    return normalize_weights(blended)


def evaluate_policy_update(
    *,
    baseline_weights: Mapping[str, float],
    candidate_weights: Mapping[str, float],
    heldout_examples: Sequence[ReplayExample],
    guard_examples: Sequence[ReplayExample],
) -> JsonDict:
    """Apply the utility gate and return accepted weights or a rollback decision."""

    baseline = normalize_weights(baseline_weights)
    candidate = normalize_weights(candidate_weights)
    heldout_before = policy_utility(baseline, heldout_examples)
    heldout_after = policy_utility(candidate, heldout_examples)
    guard_before = forgetting_guard_metric(baseline, guard_examples)
    guard_after = forgetting_guard_metric(candidate, guard_examples)
    utility_improved = heldout_after > heldout_before
    guard_passed = guard_after >= guard_before
    rollback = utility_improved and not guard_passed
    accepted = candidate if utility_improved and guard_passed else baseline
    return {
        "heldout_utility_baseline": heldout_before,
        "heldout_utility_after": heldout_after,
        "heldout_utility_delta": _round(heldout_after - heldout_before),
        "utility_improved": utility_improved,
        "forgetting_guard_metric_before": guard_before,
        "forgetting_guard_metric_after": guard_after,
        "forgetting_guard_passed": guard_passed,
        "rollback_triggered": rollback,
        "accepted_weights": accepted,
    }


def policy_utility(weights: Mapping[str, float], examples: Sequence[ReplayExample]) -> float:
    """Score how much replay mass covers verifier-grounded held-out utility."""

    normalized = normalize_weights(weights)
    total_signal = sum(example.utility_signal for example in examples)
    if total_signal <= 0.0:
        return 0.0
    score = sum(normalized.get(example.taxonomy, 0.0) * example.utility_signal for example in examples)
    return _round(score / total_signal)


def forgetting_guard_metric(weights: Mapping[str, float], examples: Sequence[ReplayExample]) -> float:
    """Measure replay mass retained for stable verified-pass examples."""

    normalized = normalize_weights(weights)
    total_signal = sum(example.forgetting_signal for example in examples)
    if total_signal <= 0.0:
        return 1.0
    score = sum(
        normalized.get(example.taxonomy, 0.0) * example.forgetting_signal for example in examples
    )
    return _round(score / total_signal)


def _base_source_artifacts(config: ExperimentConfig) -> list[JsonDict]:
    return [
        _source_artifact(
            config.repo_root,
            config.exp2947_path,
            "exp2947",
            "nonuniform_replay_pilot",
            ("replay_count_distribution", "curriculum_signal_scores", "honest_verdict"),
        ),
        _source_artifact(
            config.repo_root,
            config.exp2946_path,
            "exp2946",
            "continuation_summary",
            ("protocol_artifact_path", "pass_at_1", "pass_at_k", "honest_verdict"),
        ),
        _source_artifact(
            config.repo_root,
            config.exp2940_path,
            "exp2940",
            "verifier_energy_reward_signal",
            ("code_status_energy_values", "max_f1_operating_point", "code_status_energy_definition"),
        ),
    ]


def _source_artifact(
    repo_root: Path,
    rel_path: Path,
    experiment_id: str,
    role: str,
    fields_imported: Sequence[str],
) -> JsonDict:
    path = _repo_path(repo_root, rel_path)
    present = path.is_file()
    return {
        "experiment_id": experiment_id,
        "path": rel_path.as_posix(),
        "role": role,
        "required": True,
        "present": present,
        "fields_imported": list(fields_imported) if present else [],
        "sha256": _sha256(path) if present else None,
    }


def _blocked_artifact(
    config: ExperimentConfig,
    started: float,
    source_artifacts: list[JsonDict],
    verdict: str,
    missing_fields: list[str],
) -> JsonDict:
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "honest_verdict": verdict,
        "continuous_self_learning_task": True,
        "self_learning_utility_artifact_ready": False,
        "source_artifacts": source_artifacts,
        "replay_policies_compared": [],
        "heldout_utility_baseline": 0.0,
        "heldout_utility_after": 0.0,
        "heldout_utility_delta": 0.0,
        "self_learning_utility_positive": False,
        "forgetting_guard_metric_before": 0.0,
        "forgetting_guard_metric_after": 0.0,
        "forgetting_guard_passed": False,
        "rollback_triggered": False,
        "update_rule": {},
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": _elapsed(config, started),
        "missing_fields": missing_fields,
        "model_weights_mutated": False,
        "live_model_invoked": False,
        "tests_run": list(config.tests_run),
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
    }


def _policy_row(
    name: str,
    weights: Mapping[str, float],
    heldout_utility: float,
    guard_metric: float | None,
    accepted: bool,
) -> JsonDict:
    return {
        "policy_name": name,
        "weights": dict(weights),
        "heldout_utility": heldout_utility,
        "forgetting_guard_metric": guard_metric,
        "accepted": accepted,
    }


def _update_rule(guard_taxonomies: Sequence[str], exp2940: Mapping[str, Any]) -> JsonDict:
    operating_point = _mapping(exp2940.get("max_f1_operating_point"))
    return {
        "name": "verifier_weighted_utility_gate_v2",
        "baseline_policy": "nonuniform_replay_exp2947",
        "candidate_policy": "train_slice_verifier_energy_weighted_taxonomy_replay",
        "acceptance_rule": "accept only when held-out utility improves and forgetting guard does not degrade",
        "rollback_rule": "restore baseline replay weights when forgetting_guard_metric_after < before",
        "reward_signal": (
            "Exp2946 row_status mapped to Exp2940 code-status energy and repair taxonomy; "
            "signals are scaled by Exp2940 max-F1 PPV."
        ),
        "guard_taxonomies": list(guard_taxonomies),
        "learning_rate": 0.65,
        "exp2940_max_f1_threshold": operating_point.get("threshold"),
        "exp2940_max_f1_ppv": operating_point.get("ppv"),
    }


def _slice_manifest(slices: Mapping[str, Sequence[ReplayExample]]) -> JsonDict:
    return {
        name: {
            "candidate_count": len(examples),
            "stable_task_count": len({example.stable_id for example in examples}),
            "taxonomy_counts": _taxonomy_counts(examples),
        }
        for name, examples in slices.items()
    }


def _taxonomy_counts(examples: Sequence[ReplayExample]) -> dict[str, int]:
    counts = {taxonomy: 0 for taxonomy in TAXONOMY_ORDER}
    for example in examples:
        counts[example.taxonomy] = counts.get(example.taxonomy, 0) + 1
    return {taxonomy: count for taxonomy, count in counts.items() if count}


def _observed_taxonomies(examples: Sequence[ReplayExample]) -> tuple[str, ...]:
    observed = {example.taxonomy for example in examples}
    return tuple(taxonomy for taxonomy in TAXONOMY_ORDER if taxonomy in observed)


def _missing_required_sources(source_artifacts: Sequence[JsonDict]) -> list[str]:
    return [
        str(source["experiment_id"])
        for source in source_artifacts
        if source.get("required") is True and source.get("present") is not True
    ]


def _split_name(stable_index: int) -> str:
    remainder = stable_index % 5
    if remainder <= 2:
        return "train_replay"
    if remainder == 3:
        return "heldout_utility"
    return "forgetting_guard"


def _verdict(utility_positive: bool, rollback_triggered: bool) -> str:
    if rollback_triggered:
        return "complete: utility_candidate_rolled_back_by_forgetting_guard"
    if utility_positive:
        return "complete: utility_gated_replay_improved_heldout_without_forgetting"
    return "complete: utility_gated_replay_no_positive_heldout_gain"


def _read_json(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _repo_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _positive_float(value: object, *, default: float = 0.0) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return default


def _elapsed(config: ExperimentConfig, started: float) -> float:
    return _round(config.clock() - started)


def _round(value: float) -> float:
    return round(float(value), 12)
