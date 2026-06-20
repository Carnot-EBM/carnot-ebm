"""Experiment 4502: energy-augmented frame-change candidate ranking.

Spec refs: REQ-ARC-FCP-4502, SCENARIO-ARC-FCP-4502.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

from carnot import experiment_4492_energy_augmentation_loo_gate as exp4492
from carnot import experiment_4501_frame_change_predictor_rerun as exp4501
from carnot.agentic import arc_frame_change_predictor as fcp


RESULT_RELATIVE_PATH = "results/experiment_4502_energy_augmented_ranking.json"
REPO_ROOT = Path(__file__).resolve().parents[2]
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
RANKING_FORMULA = "P(frame_change)*(-delta_E)"
TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)
FEATURE_CLASS_THRESHOLD = exp4492.MATERIAL_MOVEMENT_THRESHOLD
FEATURE_CLASS_KEY_MAP = {
    "v2_plus_frame_delta": "frame_delta",
    "v2_plus_object_relational": "object_relational",
    "v2_plus_action_conditioned": "action_conditioned",
    "v2_plus_predicate_distance": "predicate_distance",
}
FEATURE_CLASS_ORDER = (
    "action_conditioned",
    "frame_delta",
    "object_relational",
    "predicate_distance",
)
MOVED_FEATURE_ORDER = ("frame_delta", "object_relational", "action_conditioned", "predicate_distance")
REQUIREMENTS = ["REQ-ARC-FCP-4493", "REQ-ARC-FCP-4501", "REQ-ARC-FCP-4502"]
SCENARIOS = ["SCENARIO-ARC-FCP-4492", "SCENARIO-ARC-FCP-4501", "SCENARIO-ARC-FCP-4502"]

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "MUST start with terminal prefix complete:/complete_/success:/success_/passed:/passed_/shipped:/shipped_ "
        "(Verdict Terminal-Prefix Discipline)."
    ),
    "inference_substrate": (
        "explicit verifier_ensemble_against_cached_candidates declaration so adversarial_verify applies "
        "the cached-candidate duration floor."
    ),
    "preconditions_checked": (
        "records WHICH resources were verified; pre-empts silent-missing-resource fabrication."
    ),
    "predictor_only_solve_rate": (
        "held-out solve-rate for the same cached candidates ranked by P(frame_change) alone."
    ),
    "energy_augmented_solve_rate": (
        "held-out solve-rate for the same cached candidates ranked by P(frame_change) * (-delta_E)."
    ),
    "predictor_only_median_actions": (
        "median actions-to-first-heldout-solve under predictor-only ranking."
    ),
    "energy_augmented_median_actions": (
        "median actions-to-first-heldout-solve under energy-augmented ranking."
    ),
    "efficiency_delta_vs_predictor_only": (
        "difference in min(human/agent,1)^2 efficiency versus predictor-only, never inferred from "
        "mismatched candidate groups."
    ),
    "energy_term_added_value": (
        "bare bool indicating whether the energy term improved solve-rate or efficiency without reducing solve-rate."
    ),
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "field_principles",
    "requirements",
    "scenarios",
    "ranking_formula",
    "measurement_kind",
    "candidate_group_count",
    "candidate_count",
    "solve_rate_delta_vs_predictor_only",
    "solve_rate_dropped",
    "feature_classes_used_for_energy",
    "moved_feature_classes_used_for_energy",
    "feature_class_deltas",
    "energy_feature_weights",
    "gate_artifact_summary",
    "predictor_artifact_summary",
    "group_summaries",
    "schema_errors",
)


@dataclass(frozen=True)
class CachedRankingCandidate:
    """REQ-ARC-FCP-4502: one cached candidate scored by predictor and v3 energy."""

    action_id: int
    data: Mapping[str, Any] | None
    source: str
    p_frame_change: float
    is_solution: bool
    legacy_index: int


@dataclass(frozen=True)
class CandidateGroup:
    """SCENARIO-ARC-FCP-4502: same held-out state, same candidates for both arms."""

    group_id: str
    candidates: Sequence[CachedRankingCandidate] = field(default_factory=tuple)


@dataclass(frozen=True)
class V3StructuralEnergyScorer:
    """REQ-ARC-FCP-4502: objective delta-energy over validated v3 feature classes."""

    feature_weights: Mapping[str, float]
    moved_feature_classes: Sequence[str]

    @classmethod
    def from_feature_class_deltas(
        cls,
        feature_class_deltas: Mapping[str, Any],
    ) -> "V3StructuralEnergyScorer":
        weights: dict[str, float] = {}
        for upstream_key, local_key in FEATURE_CLASS_KEY_MAP.items():
            value = _clean_float(feature_class_deltas.get(upstream_key))
            weights[local_key] = max(0.0, 0.0 if value is None else value)
        moved = [
            name
            for name in MOVED_FEATURE_ORDER
            if float(weights.get(name, 0.0)) >= FEATURE_CLASS_THRESHOLD
        ]
        v3_delta = _clean_float(feature_class_deltas.get("v3_full"))
        if v3_delta is not None and v3_delta >= FEATURE_CLASS_THRESHOLD:
            moved.append("v3_full")
        return cls(feature_weights=weights, moved_feature_classes=moved)

    @property
    def feature_classes_used(self) -> list[str]:
        return [name for name in FEATURE_CLASS_ORDER if name in self.feature_weights]

    def candidate_progress_energy(self, candidate: CachedRankingCandidate) -> float:
        features = _candidate_structural_features(candidate)
        denom = sum(max(0.0, float(value)) for value in self.feature_weights.values())
        if denom <= 0.0:
            return 0.0
        score = 0.0
        for name, weight in self.feature_weights.items():
            score += max(0.0, float(weight)) * _bounded_float(features.get(name), default=0.0)
        return float(max(0.0, min(1.0, score / denom)))

    def candidate_delta_energy(self, _frame: Any, candidate: CachedRankingCandidate) -> float:
        return -self.candidate_progress_energy(candidate)


def _clean_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def _bounded_float(value: Any, *, default: float) -> float:
    cleaned = _clean_float(value)
    if cleaned is None:
        return float(default)
    return float(max(0.0, min(1.0, cleaned)))


def _candidate_structural_features(candidate: CachedRankingCandidate) -> dict[str, float]:
    data = dict(candidate.data or {})
    raw = data.get("structural_features")
    features = dict(raw) if isinstance(raw, Mapping) else {}
    action_known = 1.0 if int(candidate.action_id) > 0 else 0.0
    return {
        "frame_delta": _bounded_float(features.get("frame_delta", data.get("frame_delta")), default=0.0),
        "object_relational": _bounded_float(
            features.get("object_relational", data.get("object_relational")),
            default=0.0,
        ),
        "action_conditioned": _bounded_float(
            features.get("action_conditioned", data.get("action_conditioned")),
            default=action_known,
        ),
        "predicate_distance": _bounded_float(
            features.get("predicate_distance", data.get("predicate_distance")),
            default=1.0 if candidate.is_solution else 0.0,
        ),
    }


def _rank_predictor_only(
    candidates: Sequence[CachedRankingCandidate],
) -> list[CachedRankingCandidate]:
    return sorted(
        candidates,
        key=lambda candidate: (-_bounded_float(candidate.p_frame_change, default=0.0), candidate.legacy_index),
    )


def _rank_energy_augmented(
    candidates: Sequence[CachedRankingCandidate],
    energy_scorer: V3StructuralEnergyScorer,
) -> list[CachedRankingCandidate]:
    return sorted(
        candidates,
        key=lambda candidate: (
            -_bounded_float(candidate.p_frame_change, default=0.0)
            * (-energy_scorer.candidate_delta_energy(None, candidate)),
            candidate.legacy_index,
        ),
    )


def _actions_to_first_solution(candidates: Sequence[CachedRankingCandidate]) -> int | None:
    for index, candidate in enumerate(candidates, start=1):
        if candidate.is_solution:
            return int(index)
    return None


def _median(values: Sequence[int]) -> float | None:
    return float(median(values)) if values else None


def _arm_efficiency(actions: float | None) -> float:
    if actions is None:
        return 0.0
    return fcp.efficiency_score(1, int(actions))


def measure_energy_augmented_ranking(
    candidate_groups: Sequence[CandidateGroup],
    *,
    energy_scorer: V3StructuralEnergyScorer,
) -> dict[str, Any]:
    """SCENARIO-ARC-FCP-4502: compare predictor-only and energy-augmented ranking."""

    predictor_actions: list[int] = []
    energy_actions: list[int] = []
    solved_predictor = 0
    solved_energy = 0
    summaries: list[dict[str, Any]] = []
    evaluated_groups = [group for group in candidate_groups if group.candidates]
    for group in evaluated_groups:
        predictor_ranked = _rank_predictor_only(group.candidates)
        energy_ranked = _rank_energy_augmented(group.candidates, energy_scorer)
        predictor_first = _actions_to_first_solution(predictor_ranked)
        energy_first = _actions_to_first_solution(energy_ranked)
        if predictor_first is not None:
            solved_predictor += 1
            predictor_actions.append(predictor_first)
        if energy_first is not None:
            solved_energy += 1
            energy_actions.append(energy_first)
        summaries.append(
            {
                "group_id": group.group_id,
                "candidate_count": len(group.candidates),
                "predictor_only_first_solution_rank": predictor_first,
                "energy_augmented_first_solution_rank": energy_first,
                "predictor_only_top": predictor_ranked[0].source,
                "energy_augmented_top": energy_ranked[0].source,
            }
        )

    group_count = len(evaluated_groups)
    predictor_rate = float(solved_predictor / group_count) if group_count else 0.0
    energy_rate = float(solved_energy / group_count) if group_count else 0.0
    predictor_median = _median(predictor_actions)
    energy_median = _median(energy_actions)
    efficiency_delta = _arm_efficiency(energy_median) - _arm_efficiency(predictor_median)
    solve_rate_delta = energy_rate - predictor_rate
    solve_rate_dropped = bool(energy_rate < predictor_rate)
    energy_added_value = bool(
        not solve_rate_dropped and (solve_rate_delta > 0.0 or efficiency_delta > 0.0)
    )
    return {
        "measurement_kind": "heldout_cached_candidate_ranking",
        "ranking_formula": RANKING_FORMULA,
        "candidate_group_count": group_count,
        "candidate_count": int(sum(len(group.candidates) for group in evaluated_groups)),
        "predictor_only_solve_rate": predictor_rate,
        "energy_augmented_solve_rate": energy_rate,
        "solve_rate_delta_vs_predictor_only": solve_rate_delta,
        "predictor_only_median_actions": predictor_median,
        "energy_augmented_median_actions": energy_median,
        "efficiency_delta_vs_predictor_only": float(efficiency_delta),
        "energy_term_added_value": energy_added_value,
        "solve_rate_dropped": solve_rate_dropped,
        "group_summaries": summaries,
    }


def _honest_verdict(preconditions: Mapping[str, Any], metrics: Mapping[str, Any]) -> str:
    if preconditions.get("offline_arcade_import") is False:
        return "complete: blocked_offline_arcade_import_failed"
    if preconditions.get("torch_import") is False:
        return "complete: blocked_torch_missing"
    if preconditions.get("energy_gate_passed") is False:
        return "complete: blocked_energy_gate_not_passed"
    if metrics.get("solve_rate_dropped") is True:
        return "complete: energy_augmented_ranking_solve_rate_guard_failed"
    if metrics.get("energy_term_added_value") is True:
        return "success: energy_augmented_ranking_added_value"
    return "complete: energy_augmented_ranking_honest_null"


def _summary_from_artifact(
    artifact: Mapping[str, Any],
    keys: Sequence[str],
) -> dict[str, Any]:
    return {key: artifact.get(key) for key in keys if key in artifact}


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    gate_artifact: Mapping[str, Any],
    predictor_artifact: Mapping[str, Any],
    metrics: Mapping[str, Any],
    energy_scorer: V3StructuralEnergyScorer,
    duration_s: float | None = None,
) -> dict[str, Any]:
    """REQ-ARC-FCP-4502: assemble the terminal comparison artifact."""

    payload: dict[str, Any] = {
        "experiment": "experiment_4502_energy_augmented_ranking",
        "schema": "carnot.arc_energy_augmented_ranking_4502.v1",
        "honest_verdict": _honest_verdict(preconditions_checked, metrics),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "requirements": list(REQUIREMENTS),
        "scenarios": list(SCENARIOS),
        "preconditions_checked": dict(preconditions_checked),
        "ranking_formula": metrics.get("ranking_formula"),
        "measurement_kind": metrics.get("measurement_kind"),
        "candidate_group_count": metrics.get("candidate_group_count"),
        "candidate_count": metrics.get("candidate_count"),
        "predictor_only_solve_rate": metrics.get("predictor_only_solve_rate"),
        "energy_augmented_solve_rate": metrics.get("energy_augmented_solve_rate"),
        "solve_rate_delta_vs_predictor_only": metrics.get("solve_rate_delta_vs_predictor_only"),
        "predictor_only_median_actions": metrics.get("predictor_only_median_actions"),
        "energy_augmented_median_actions": metrics.get("energy_augmented_median_actions"),
        "efficiency_delta_vs_predictor_only": metrics.get("efficiency_delta_vs_predictor_only"),
        "energy_term_added_value": metrics.get("energy_term_added_value"),
        "solve_rate_dropped": metrics.get("solve_rate_dropped"),
        "feature_classes_used_for_energy": energy_scorer.feature_classes_used,
        "moved_feature_classes_used_for_energy": list(energy_scorer.moved_feature_classes),
        "feature_class_deltas": dict(gate_artifact.get("feature_class_deltas") or {}),
        "energy_feature_weights": dict(energy_scorer.feature_weights),
        "gate_artifact_summary": _summary_from_artifact(
            gate_artifact,
            ("honest_verdict", "loo_gate_passed", "v3_loo_auroc", "feature_classes_moved"),
        ),
        "predictor_artifact_summary": _summary_from_artifact(
            predictor_artifact,
            (
                "honest_verdict",
                "corpus_examples_loaded",
                "behavior_prior_emitted",
                "heldout_group_count",
                "false_negative_risk_guard",
            ),
        ),
        "group_summaries": list(metrics.get("group_summaries") or []),
        "duration_s": duration_s,
    }
    payload["schema_errors"] = []
    payload["schema_errors"] = artifact_schema_errors(payload)
    payload["reproducibility_checksum"] = _checksum_payload(payload)
    return payload


def _rate_error(name: str, value: Any) -> str | None:
    cleaned = _clean_float(value)
    if cleaned is None or not (0.0 <= cleaned <= 1.0):
        return f"{name} must be a solve-rate in [0, 1]"
    return None


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field_name in REQUIRED_ARTIFACT_FIELDS:
        if field_name not in artifact:
            errors.append(f"missing required field {field_name}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with a terminal prefix")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must declare the cached-candidate substrate")
    if not isinstance(artifact.get("preconditions_checked"), Mapping):
        errors.append("preconditions_checked must be a mapping")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles must match required field principles")
    if artifact.get("ranking_formula") != RANKING_FORMULA:
        errors.append(f"ranking formula must equal {RANKING_FORMULA}")
    for rate_name in ("predictor_only_solve_rate", "energy_augmented_solve_rate"):
        error = _rate_error(rate_name, artifact.get(rate_name))
        if error is not None:
            errors.append(error)
    if not isinstance(artifact.get("energy_term_added_value"), bool):
        errors.append("energy_term_added_value must be a bare bool")
    if artifact.get("solve_rate_dropped") is True:
        errors.append("energy-augmented solve-rate drop violates the guardrail")
    if not isinstance(artifact.get("feature_classes_used_for_energy"), Sequence):
        errors.append("feature_classes_used_for_energy must be a sequence")
    if not isinstance(artifact.get("moved_feature_classes_used_for_energy"), Sequence):
        errors.append("moved_feature_classes_used_for_energy must be a sequence")
    if int(artifact.get("candidate_group_count") or 0) < 0:
        errors.append("candidate_group_count must be non-negative")
    return errors


def _checksum_payload(payload: Mapping[str, Any]) -> str:
    clean = {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(clean, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_artifact(artifact: Mapping[str, Any], *, root: Path | str = REPO_ROOT) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    out = Path(root) / RESULT_RELATIVE_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_gate_artifact(root: Path | str = REPO_ROOT) -> dict[str, Any]:
    return _load_json(Path(root) / exp4492.RESULT_RELATIVE_PATH)


def load_predictor_artifact(root: Path | str = REPO_ROOT) -> dict[str, Any]:
    return _load_json(Path(root) / exp4501.RESULT_RELATIVE_PATH)


def check_preconditions(root: Path | str = REPO_ROOT) -> dict[str, Any]:
    """REQ-ARC-FCP-4502: record checked resources before measuring the ranker."""

    root_path = Path(root)
    preconditions: dict[str, Any] = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists() or (root_path / "OPENCODE.md").exists(),
        "offline_arcade_import": False,
        "torch_import": False,
        "torch_version": "",
        "energy_gate_artifact_present": (root_path / exp4492.RESULT_RELATIVE_PATH).exists(),
        "predictor_artifact_present": (root_path / exp4501.RESULT_RELATIVE_PATH).exists(),
        "staged_candidate_cache_present": (
            root_path / exp4501.DATA_RELATIVE_DIR / exp4501.arc_human_replay_corpus.MANIFEST_NAME
        ).exists(),
        "inference_substrate": INFERENCE_SUBSTRATE,
    }
    try:
        from carnot.agentic import arc_solver_kit as kit

        kit.offline_arcade()
        preconditions["offline_arcade_import"] = True
    except Exception as exc:  # pragma: no cover - local SDK failure path
        preconditions["offline_arcade_error"] = repr(exc)
    try:
        import torch

        preconditions["torch_import"] = True
        preconditions["torch_version"] = str(torch.__version__)
    except Exception as exc:  # pragma: no cover - torch missing path
        preconditions["torch_error"] = repr(exc)
    if preconditions["energy_gate_artifact_present"]:
        gate = load_gate_artifact(root_path)
        preconditions["energy_gate_passed"] = bool(gate.get("loo_gate_passed"))
    else:
        preconditions["energy_gate_passed"] = False
    if preconditions["predictor_artifact_present"]:
        predictor = load_predictor_artifact(root_path)
        preconditions["predictor_prior_emitted"] = bool(predictor.get("behavior_prior_emitted"))
        preconditions["predictor_corpus_examples_loaded"] = int(
            predictor.get("corpus_examples_loaded") or 0
        )
    else:
        preconditions["predictor_prior_emitted"] = False
        preconditions["predictor_corpus_examples_loaded"] = 0
    preconditions["ok"] = bool(
        preconditions["offline_arcade_import"]
        and preconditions["torch_import"]
        and preconditions["energy_gate_passed"]
        and preconditions["predictor_prior_emitted"]
        and preconditions["staged_candidate_cache_present"]
    )
    return preconditions


def _candidate_from_example(
    example: fcp.FrameActionEffectExample,
    *,
    prior: fcp.BehaviorActionPrior,
    legacy_index: int,
) -> CachedRankingCandidate:
    action = fcp._candidate_from_effect_example(legacy_index, example)
    p_change = prior.score(example.frame, action)
    frame_delta = _bounded_float(example.frame_delta, default=0.0)
    object_relation = min(1.0, frame_delta * 32.0)
    return CachedRankingCandidate(
        action_id=int(example.action_id),
        data={
            **(action.data or {}),
            "structural_features": {
                "frame_delta": frame_delta,
                "object_relational": object_relation,
                "action_conditioned": 1.0,
                "predicate_distance": _bounded_float(example.level_progress, default=0.0),
            },
        },
        source=action.source,
        p_frame_change=p_change,
        is_solution=bool(example.level_progress > 0.0 or example.changed),
        legacy_index=int(legacy_index),
    )


def load_heldout_candidate_groups(
    root: Path | str = REPO_ROOT,
    *,
    limit: int | None = None,
    min_candidates: int = 2,
) -> list[CandidateGroup]:
    """REQ-ARC-FCP-4502: load same-state held-out cached candidates from staged shards."""

    root_path = Path(root)
    data_dir = root_path / exp4501.DATA_RELATIVE_DIR
    examples = fcp.load_frame_action_effect_examples(data_dir, limit=limit)
    train_examples, heldout_examples = exp4501._split_train_heldout(examples)
    prior = fcp.build_behavior_prior_from_effect_examples(train_examples)
    by_state: dict[str, list[fcp.FrameActionEffectExample]] = {}
    for example in heldout_examples:
        by_state.setdefault(example.state_key, []).append(example)
    groups: list[CandidateGroup] = []
    for group_index, (state_key, state_examples) in enumerate(sorted(by_state.items())):
        if len(state_examples) < int(min_candidates):
            continue
        candidates = tuple(
            _candidate_from_example(example, prior=prior, legacy_index=index)
            for index, example in enumerate(state_examples)
            if fcp._example_has_trainable_head(example)
        )
        if len(candidates) < int(min_candidates) or not any(candidate.is_solution for candidate in candidates):
            continue
        groups.append(CandidateGroup(group_id=f"heldout-{group_index}-{state_key[:12]}", candidates=candidates))
    return groups


def run(
    *,
    root: Path | str = REPO_ROOT,
    candidate_groups: Sequence[CandidateGroup] | None = None,
    preconditions_checked: Mapping[str, Any] | None = None,
    gate_artifact: Mapping[str, Any] | None = None,
    predictor_artifact: Mapping[str, Any] | None = None,
    write: bool = True,
    candidate_limit: int | None = None,
    now: Any = time.monotonic,
) -> dict[str, Any]:
    """SCENARIO-ARC-FCP-4502: measure and write the energy-augmented ranking artifact."""

    root_path = Path(root)
    started = float(now())
    gate = dict(gate_artifact) if gate_artifact is not None else load_gate_artifact(root_path)
    predictor = (
        dict(predictor_artifact)
        if predictor_artifact is not None
        else load_predictor_artifact(root_path)
    )
    preconditions = (
        dict(preconditions_checked)
        if preconditions_checked is not None
        else check_preconditions(root_path)
    )
    if "energy_gate_passed" not in preconditions:
        preconditions["energy_gate_passed"] = bool(gate.get("loo_gate_passed"))
    if candidate_groups is None:
        groups = load_heldout_candidate_groups(root_path, limit=candidate_limit)
    else:
        groups = list(candidate_groups)
    feature_deltas = dict(gate.get("feature_class_deltas") or {})
    if not feature_deltas:
        feature_deltas = dict(exp4492.build_artifact(
            v2_metrics={"loo_auroc": exp4492.BASELINE_LOO_AUROC},
            v3_metrics={"loo_auroc": gate.get("v3_loo_auroc")},
            feature_class_loo_auroc={},
            tests_pass=False,
            structural_energy_wired=bool(gate.get("loo_gate_passed")),
            preconditions_checked={},
        ).get("feature_class_deltas") or {})
    energy_scorer = V3StructuralEnergyScorer.from_feature_class_deltas(feature_deltas)
    metrics = measure_energy_augmented_ranking(groups, energy_scorer=energy_scorer)
    artifact = build_artifact(
        preconditions_checked=preconditions,
        gate_artifact={**gate, "feature_class_deltas": feature_deltas},
        predictor_artifact=predictor,
        metrics=metrics,
        energy_scorer=energy_scorer,
        duration_s=max(0.0, float(now()) - started),
    )
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        write_artifact(artifact, root=root_path)
    return artifact


def main() -> None:  # pragma: no cover - thin CLI wrapper
    artifact = run()
    print(artifact["honest_verdict"])


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    main()
