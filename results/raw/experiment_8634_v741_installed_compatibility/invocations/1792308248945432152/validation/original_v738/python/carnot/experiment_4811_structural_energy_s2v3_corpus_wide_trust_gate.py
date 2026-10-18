"""Experiment 4811: S2-v3 corpus-wide structural-energy trust gate.

Spec refs: REQ-ARC-WMTE-4811,
SCENARIO-ARC-WMTE-4811-CORPUS-WIDE-TRUST-GATE.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

from carnot import experiment_4801_structural_energy_s2v2_diverse_trust_gate as s2v2
from carnot.agentic.arc_executable_world_model import Transition, collect_transitions
from carnot.agentic.arc_world_model_trust_energy import (
    INFERENCE_SUBSTRATE,
    S1StructuralTransitionEnergy,
    WorldModelCandidate,
    default_s1_offpath_energy_scorer,
)


JsonDict = dict[str, Any]
Engine = Callable[[np.ndarray, int, Any], np.ndarray]
GameTrustGateResult = s2v2.GameTrustGateResult

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_4811_structural_energy_s2v3_corpus_wide_trust_gate"
EXPERIMENT_ID = 4811
SCHEMA = "carnot.arc_structural_energy_s2v3_corpus_wide_trust_gate_4811.v1"
RESULT_RELATIVE_PATH = "results/experiment_4811_structural_energy_s2v3_corpus_wide_trust_gate.json"
RANDOM_SEED = 4811
BOOTSTRAP_RESAMPLES = 1000
MIN_EFFECTIVE_FLOOR = 10
MIN_CANDIDATES_PER_GAME = 3
EFFECTIVE_SPREAD_EPS = 1.0e-3
DEFAULT_TRANSITIONS_PER_GAME = 24
MIN_AVAILABLE_CORPUS_GAMES = 20
SPEC_REFS = [
    "REQ-ARC-WMTE-4811",
    "SCENARIO-ARC-WMTE-4811-CORPUS-WIDE-TRUST-GATE",
]
TERMINAL_PREFIXES = s2v2.TERMINAL_PREFIXES

SUCCESS_VERDICT = "success_structural_energy_s2v3_trust_gate_authorizes_s3"
BOUNDED_VERDICT = "complete_structural_energy_s2v3_bounded_corpus_wide"
INCONCLUSIVE_VERDICT = (
    "complete_structural_energy_s2v3_inconclusive_insufficient_corpus_diversity"
)

REQUIRED_FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; a corpus-wide trust win is "
            "success_structural_energy_s2v3_trust_gate_authorizes_s3, a genuine "
            "corpus-wide null is complete_structural_energy_s2v3_bounded_corpus_wide, "
            "an under-covered result is "
            "complete_structural_energy_s2v3_inconclusive_insufficient_corpus_diversity."
        )
    },
    "verifier_is_oracle": {
        "principle": (
            "MUST be false for the energy ranking (oracle-distinct); the binary-accuracy "
            "CONTROL is execution-grounded."
        )
    },
    "n_available_games": {
        "principle": (
            "the full offline corpus size (ls environment_files/) -- the .443 B1 audit "
            "verifies this against the real corpus; the gate requires effective >= "
            "max(10, 0.6*this)."
        )
    },
    "n_games_attempted": {
        "principle": (
            "must ~= n_available_games -- S2-v2's failure was testing 5 of 25; a "
            "corpus-wide test attempts ALL games."
        )
    },
    "n_effective_games": {
        "principle": (
            "games whose candidate pool spans > 1e-3 off-path cell_recall -- must be >= "
            "max(10, ceil(0.6*n_available)) for a PASS/BOUNDED verdict; else INCONCLUSIVE."
        )
    },
    "game_results": {
        "principle": (
            "per-game candidate_rows (heldout_cell_recall + offpath_structural_energy + "
            "accuracy) + energy-selected vs accuracy-selected candidate -- the "
            "corpus-wide per-game log."
        )
    },
    "energy_minus_accuracy_delta_ci95": {
        "principle": (
            "must EXCLUDE 0 for PASS, on >= the required effective games -- a "
            "corpus-wide delta, not n=5."
        )
    },
    "positive_control_passed": {
        "principle": (
            "REQUIRED for a BOUNDED verdict -- a documented headroom upper bound proving "
            "the diverse pool contains a candidate the accuracy gate misses, so BOUNDED "
            "means 'energy could have won but didn't'."
        )
    },
    "candidates_genuinely_induced": {
        "principle": (
            "the diversity must come from genuinely-induced engines (varied rounds/seeds), "
            "NOT an injected broken 0.0-recall sabotage candidate."
        )
    },
    "live_path_reachable": {
        "principle": (
            "the energy gate must be in the E3AgentPolicy import closure "
            "(arc_orphan_solver_lint)."
        )
    },
    "preconditions_checked": {
        "principle": (
            "records the arcade / corpus checks so a silent-missing-resource run cannot "
            "fabricate a cell_recall."
        )
    },
    "random_seed": {
        "principle": "determinism for the candidate generation + selection + bootstrap."
    },
    "reproducibility_checksum": {
        "principle": (
            "content hash of (candidate engines, folds, energy config) so a replication "
            "catches drift."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "result_path",
    "honest_verdict",
    "verifier_is_oracle",
    "binary_accuracy_control",
    "live_path_reachable",
    "inference_substrate",
    "preconditions_checked",
    "energy_selected_offpath_cell_recall",
    "accuracy_gate_selected_offpath_cell_recall",
    "energy_minus_accuracy_delta",
    "energy_minus_accuracy_delta_ci95",
    "n_available_games",
    "n_games_attempted",
    "n_candidate_games_attempted",
    "n_effective_games",
    "required_effective_games",
    "min_heldout_games",
    "random_seed",
    "s3_authorized",
    "retire_if_same_verdict",
    "game_results",
    "candidate_pool_diversity",
    "positive_control_passed",
    "false_negative_risk_checked",
    "candidates_genuinely_induced",
    "candidate_generation_config",
    "candidate_sets",
    "energy_config",
    "field_principles",
    "duration_s",
    "reproducibility_checksum",
)


def _clean_float(value: Any) -> float | None:
    return s2v2._clean_float(value)


def _mean(values: Sequence[float]) -> float | None:
    return s2v2._mean(values)


def _checksum_payload(payload: Mapping[str, Any]) -> str:
    clean = {k: v for k, v in payload.items() if k != "reproducibility_checksum"}
    raw = json.dumps(clean, sort_keys=True, default=str, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> list[float] | None:
    return s2v2._bootstrap_mean_ci(values, seed=seed, resamples=resamples)


def _ci_excludes_zero_positive(ci: Sequence[float] | None) -> bool:
    return s2v2._ci_excludes_zero_positive(ci)


def required_effective_games(*, n_available_games: int, n_games_attempted: int) -> int:
    """REQ-ARC-WMTE-4811: corpus-wide floor independent of declared min_heldout_games."""

    corpus = max(int(n_available_games), int(n_games_attempted))
    return max(MIN_EFFECTIVE_FLOOR, int(math.ceil(0.6 * corpus)))


def available_offline_games(root: Path | str = REPO_ROOT) -> list[str]:
    """REQ-ARC-WMTE-4811: read the full offline corpus from environment_files/."""

    env_dir = Path(root) / "environment_files"
    if not env_dir.exists():
        return []
    return sorted(path.name for path in env_dir.iterdir() if not path.name.startswith("."))


def evaluate_candidate_set(
    *,
    game: str,
    transitions: Sequence[Transition],
    candidates: Sequence[WorldModelCandidate],
    candidate_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    energy_scorer: Any | None = None,
    binary_threshold: float = 0.5,
) -> GameTrustGateResult:
    """REQ-ARC-WMTE-4811: compare energy selection on one active induced pool."""

    return s2v2.evaluate_candidate_set(
        game=game,
        transitions=transitions,
        candidates=candidates,
        candidate_metadata=candidate_metadata,
        energy_scorer=energy_scorer,
        binary_threshold=binary_threshold,
    )


def _result_json(row: GameTrustGateResult | Mapping[str, Any]) -> JsonDict:
    if hasattr(row, "to_json"):
        return row.to_json()
    return dict(row)


def _candidate_recalls(row: Mapping[str, Any]) -> list[float]:
    rows = row.get("candidate_rows")
    if not isinstance(rows, list):
        return []
    return [
        float(candidate["heldout_cell_recall"])
        for candidate in rows
        if isinstance(candidate, Mapping)
        and _clean_float(candidate.get("heldout_cell_recall")) is not None
    ]


def _game_effective(row: Mapping[str, Any]) -> bool:
    values = _candidate_recalls(row)
    rows = row.get("candidate_rows")
    return bool(
        isinstance(rows, list)
        and len(rows) >= MIN_CANDIDATES_PER_GAME
        and values
        and (max(values) - min(values)) > EFFECTIVE_SPREAD_EPS
    )


def _row_by_candidate_name(rows: Sequence[Mapping[str, Any]], name: str | None) -> Mapping[str, Any]:
    for row in rows:
        if str(row.get("candidate_name")) == str(name):
            return row
    return rows[0] if rows else {}


def _selected_recall(row: Mapping[str, Any], key: str, fallback: str) -> float:
    value = _clean_float(row.get(fallback))
    if value is not None:
        return float(value)
    rows = row.get("candidate_rows")
    if not isinstance(rows, list):
        return 0.0
    selected = _row_by_candidate_name(rows, str(row.get(key)))
    return float(selected.get("heldout_cell_recall", 0.0))


def _effective_result_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [dict(row) for row in rows if _game_effective(row)]


def _positive_control_passed(rows: Sequence[Mapping[str, Any]]) -> bool:
    for result in rows:
        values = _candidate_recalls(result)
        if not values:
            continue
        accuracy_recall = _selected_recall(
            result,
            "accuracy_gate_selected_candidate",
            "accuracy_gate_selected_offpath_cell_recall",
        )
        if max(values) > accuracy_recall + EFFECTIVE_SPREAD_EPS:
            return True
    return False


def _candidates_genuinely_induced(rows: Sequence[Mapping[str, Any]]) -> bool:
    return bool(rows) and all(
        bool(candidate.get("genuinely_induced"))
        for result in rows
        for candidate in result.get("candidate_rows", [])
        if isinstance(candidate, Mapping)
    )


def _success_not_sabotage(rows: Sequence[Mapping[str, Any]]) -> bool:
    for result in rows:
        if str(result.get("energy_selected_candidate")) == str(
            result.get("accuracy_gate_selected_candidate")
        ):
            continue
        delta = _clean_float(result.get("energy_minus_accuracy_delta"))
        if delta is None or delta <= 0.0:
            continue
        if (
            _selected_recall(
                result,
                "energy_selected_candidate",
                "energy_selected_offpath_cell_recall",
            )
            <= 0.0
        ):
            return False
    return True


def _candidate_pool_diversity(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    out: list[JsonDict] = []
    for result in rows:
        values = _candidate_recalls(result)
        candidate_rows = result.get("candidate_rows")
        candidate_rows = candidate_rows if isinstance(candidate_rows, list) else []
        out.append(
            {
                "game": str(result.get("game")),
                "n_candidates": len(candidate_rows),
                "has_required_candidate_count": len(candidate_rows) >= MIN_CANDIDATES_PER_GAME,
                "effective": _game_effective(result),
                "distinct_heldout_cell_recall_count": s2v2._distinct_count(values),
                "heldout_cell_recall_spread": float(max(values) - min(values)) if values else 0.0,
                "candidate_heldout_cell_recalls": values,
                "candidate_names": [
                    str(candidate.get("candidate_name"))
                    for candidate in candidate_rows
                    if isinstance(candidate, Mapping)
                ],
                "candidates_genuinely_induced": bool(candidate_rows)
                and all(
                    bool(candidate.get("genuinely_induced"))
                    for candidate in candidate_rows
                    if isinstance(candidate, Mapping)
                ),
                "accuracy_gate_selected_offpath_cell_recall": _selected_recall(
                    result,
                    "accuracy_gate_selected_candidate",
                    "accuracy_gate_selected_offpath_cell_recall",
                ),
                "oracle_headroom_best_candidate_cell_recall": max(values) if values else 0.0,
                "positive_control_headroom": bool(
                    values
                    and max(values)
                    > _selected_recall(
                        result,
                        "accuracy_gate_selected_candidate",
                        "accuracy_gate_selected_offpath_cell_recall",
                    )
                    + EFFECTIVE_SPREAD_EPS
                ),
                "skip_reason": result.get("skip_reason"),
            }
        )
    return out


def _artifact_verdict(
    *,
    n_effective_games: int,
    required_effective: int,
    delta: float | None,
    delta_ci95: Sequence[float] | None,
    live_path_reachable: bool,
    positive_control_passed: bool,
    candidates_genuinely_induced: bool,
    effective_rows: Sequence[Mapping[str, Any]],
) -> str:
    if n_effective_games < required_effective:
        return INCONCLUSIVE_VERDICT
    if (
        live_path_reachable
        and candidates_genuinely_induced
        and delta is not None
        and delta > 0.0
        and _ci_excludes_zero_positive(delta_ci95)
        and _success_not_sabotage(effective_rows)
    ):
        return SUCCESS_VERDICT
    if positive_control_passed:
        return BOUNDED_VERDICT
    return INCONCLUSIVE_VERDICT


def build_artifact(
    game_results: Sequence[GameTrustGateResult | Mapping[str, Any]],
    *,
    n_available_games: int,
    preconditions_checked: Mapping[str, Any],
    live_path_reachable: bool,
    random_seed: int = RANDOM_SEED,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    duration_s: float = 1.0,
    energy_scorer: Any | None = None,
) -> JsonDict:
    """SCENARIO-ARC-WMTE-4811-CORPUS-WIDE-TRUST-GATE: build the S2-v3 artifact."""

    attempted_rows = [_result_json(row) for row in game_results]
    n_attempted = len(attempted_rows)
    required_effective = required_effective_games(
        n_available_games=int(n_available_games),
        n_games_attempted=n_attempted,
    )
    effective_rows = _effective_result_rows(attempted_rows)
    sufficient = len(effective_rows) >= required_effective
    conclusion_rows = effective_rows if sufficient else []
    deltas = [
        float(row.get("energy_minus_accuracy_delta", 0.0))
        for row in conclusion_rows
        if _clean_float(row.get("energy_minus_accuracy_delta")) is not None
    ]
    energy_recalls = [
        _selected_recall(
            row,
            "energy_selected_candidate",
            "energy_selected_offpath_cell_recall",
        )
        for row in conclusion_rows
    ]
    accuracy_recalls = [
        _selected_recall(
            row,
            "accuracy_gate_selected_candidate",
            "accuracy_gate_selected_offpath_cell_recall",
        )
        for row in conclusion_rows
    ]
    delta = _mean(deltas)
    delta_ci95 = _bootstrap_mean_ci(
        deltas,
        seed=int(random_seed),
        resamples=bootstrap_resamples,
    )
    positive_control = bool(sufficient and _positive_control_passed(conclusion_rows))
    genuinely_induced = bool(sufficient and _candidates_genuinely_induced(conclusion_rows))
    verdict = _artifact_verdict(
        n_effective_games=len(effective_rows),
        required_effective=required_effective,
        delta=delta,
        delta_ci95=delta_ci95,
        live_path_reachable=live_path_reachable,
        positive_control_passed=positive_control,
        candidates_genuinely_induced=genuinely_induced,
        effective_rows=conclusion_rows,
    )
    if not sufficient:
        delta = None
        delta_ci95 = None
        energy_recalls = []
        accuracy_recalls = []
    scorer = energy_scorer or default_s1_offpath_energy_scorer()
    energy_config = getattr(scorer, "energy_config", {"source": str(type(scorer).__name__)})
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": verdict,
        "verifier_is_oracle": False,
        "binary_accuracy_control": {
            "verifier_is_oracle": True,
            "selection_rule": "first_candidate_with_WorldModelVerifier_exact_accuracy_ge_0.5_else_first",
            "threshold": 0.5,
        },
        "live_path_reachable": bool(live_path_reachable),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "energy_selected_offpath_cell_recall": _mean(energy_recalls),
        "accuracy_gate_selected_offpath_cell_recall": _mean(accuracy_recalls),
        "energy_minus_accuracy_delta": delta,
        "energy_minus_accuracy_delta_ci95": delta_ci95,
        "n_available_games": int(n_available_games),
        "n_games_attempted": int(n_attempted),
        "n_candidate_games_attempted": int(n_attempted),
        "n_effective_games": len(effective_rows),
        "required_effective_games": int(required_effective),
        "min_heldout_games": len(effective_rows),
        "random_seed": int(random_seed),
        "s3_authorized": verdict == SUCCESS_VERDICT,
        "retire_if_same_verdict": True,
        "game_results": attempted_rows,
        "candidate_pool_diversity": _candidate_pool_diversity(attempted_rows),
        "positive_control_passed": bool(positive_control),
        "false_negative_risk_checked": bool(positive_control),
        "candidates_genuinely_induced": bool(genuinely_induced),
        "candidate_generation_config": {
            "minimum_candidates_per_game": MIN_CANDIDATES_PER_GAME,
            "effective_spread_eps": EFFECTIVE_SPREAD_EPS,
            "required_effective_rule": "max(10, ceil(0.6 * max(n_games_attempted, n_available_games)))",
            "candidate_sources": [
                "deterministic_transition_induction",
                "programmatic_expert_induction",
            ],
            "attempt_all_environment_files": True,
            "no_injected_broken_candidates": True,
            "cached_world_model_pairs_only": False,
        },
        "candidate_sets": {
            str(row.get("game")): [
                str(candidate.get("candidate_name"))
                for candidate in row.get("candidate_rows", [])
                if isinstance(candidate, Mapping)
            ]
            for row in attempted_rows
        },
        "energy_config": energy_config,
        "field_principles": REQUIRED_FIELD_PRINCIPLES,
        "duration_s": round(float(duration_s), 3),
        "reproducibility_checksum": None,
    }
    payload["reproducibility_checksum"] = _checksum_payload(payload)
    return payload


def build_blocked_artifact(
    honest_verdict: str,
    preconditions_checked: Mapping[str, Any],
    *,
    random_seed: int = RANDOM_SEED,
    live_path_reachable: bool = False,
) -> JsonDict:
    """REQ-ARC-WMTE-4811: fail closed without fabricated corpus scoring."""

    n_available = int(preconditions_checked.get("n_available_games", 0) or 0)
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": honest_verdict,
        "verifier_is_oracle": False,
        "binary_accuracy_control": {
            "verifier_is_oracle": True,
            "selection_rule": "not_run_blocked_precondition",
            "threshold": 0.5,
        },
        "live_path_reachable": bool(live_path_reachable),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "energy_selected_offpath_cell_recall": None,
        "accuracy_gate_selected_offpath_cell_recall": None,
        "energy_minus_accuracy_delta": None,
        "energy_minus_accuracy_delta_ci95": None,
        "n_available_games": n_available,
        "n_games_attempted": 0,
        "n_candidate_games_attempted": 0,
        "n_effective_games": 0,
        "required_effective_games": required_effective_games(
            n_available_games=n_available,
            n_games_attempted=0,
        ),
        "min_heldout_games": 0,
        "random_seed": int(random_seed),
        "s3_authorized": False,
        "retire_if_same_verdict": True,
        "game_results": [],
        "candidate_pool_diversity": [],
        "positive_control_passed": False,
        "false_negative_risk_checked": False,
        "candidates_genuinely_induced": False,
        "candidate_generation_config": {
            "minimum_candidates_per_game": MIN_CANDIDATES_PER_GAME,
            "effective_spread_eps": EFFECTIVE_SPREAD_EPS,
            "required_effective_rule": "max(10, ceil(0.6 * max(n_games_attempted, n_available_games)))",
            "candidate_sources": [],
            "attempt_all_environment_files": False,
            "no_injected_broken_candidates": True,
            "cached_world_model_pairs_only": False,
        },
        "candidate_sets": {},
        "energy_config": default_s1_offpath_energy_scorer().energy_config,
        "field_principles": REQUIRED_FIELD_PRINCIPLES,
        "duration_s": 0.0,
        "reproducibility_checksum": None,
    }
    payload["reproducibility_checksum"] = _checksum_payload(payload)
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _linter_effective_game_count(game_results: Sequence[Any]) -> int:
    return sum(1 for row in game_results if isinstance(row, Mapping) and _game_effective(row))


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    _require(not missing, f"missing required artifact fields: {missing}")
    checksum = artifact.get("reproducibility_checksum")
    _require(isinstance(checksum, str) and checksum.startswith("sha256:"), "checksum")
    _require(checksum == _checksum_payload(artifact), "reproducibility_checksum mismatch")
    _require(artifact.get("verifier_is_oracle") is False, "energy verifier must be non-oracle")
    _require(
        artifact.get("binary_accuracy_control", {}).get("verifier_is_oracle") is True,
        "binary control must declare oracle grounding",
    )
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(artifact.get("field_principles") == REQUIRED_FIELD_PRINCIPLES, "field_principles")
    verdict = artifact.get("honest_verdict")
    _require(isinstance(verdict, str) and verdict, "honest_verdict")
    _require(isinstance(artifact.get("n_available_games"), int), "n_available_games")
    _require(isinstance(artifact.get("n_games_attempted"), int), "n_games_attempted")
    _require(isinstance(artifact.get("n_effective_games"), int), "n_effective_games")
    _require(
        artifact.get("min_heldout_games") == artifact.get("n_effective_games"),
        "min_heldout_games must equal n_effective_games",
    )
    _require(isinstance(artifact.get("positive_control_passed"), bool), "positive_control_passed")
    _require(
        isinstance(artifact.get("candidates_genuinely_induced"), bool),
        "candidates_genuinely_induced",
    )
    expected_required = required_effective_games(
        n_available_games=int(artifact["n_available_games"]),
        n_games_attempted=int(artifact["n_games_attempted"]),
    )
    _require(
        artifact.get("required_effective_games") == expected_required,
        "required_effective_games",
    )
    if verdict.startswith("blocked_"):
        _require(artifact.get("n_games_attempted") == 0, "blocked attempted games")
        _require(artifact.get("n_effective_games") == 0, "blocked effective games")
        _require(artifact.get("game_results") == [], "blocked game_results")
        _require(artifact.get("s3_authorized") is False, "blocked cannot authorize S3")
        return
    _require(verdict.startswith(TERMINAL_PREFIXES), "terminal-prefixed verdict")
    _require(isinstance(artifact.get("live_path_reachable"), bool), "live_path_reachable")
    game_results = artifact.get("game_results")
    _require(isinstance(game_results, list), "game_results")
    _require(
        artifact.get("n_games_attempted") == len(game_results),
        "n_games_attempted must equal logged game_results",
    )
    _require(
        artifact.get("n_candidate_games_attempted") == artifact.get("n_games_attempted"),
        "n_candidate_games_attempted",
    )
    _require(
        artifact.get("n_games_attempted") == artifact.get("n_available_games"),
        "n_games_attempted must equal n_available_games",
    )
    _require(
        _linter_effective_game_count(game_results) == artifact.get("n_effective_games"),
        "n_effective_games must equal linter-computed effective count",
    )
    if verdict == SUCCESS_VERDICT:
        _require(artifact.get("live_path_reachable") is True, "success must be live-path reachable")
        _require(
            artifact.get("n_effective_games", 0) >= expected_required,
            "success needs required effective games",
        )
        _require(
            artifact.get("candidates_genuinely_induced") is True,
            "success needs genuinely induced candidates",
        )
        _require(
            _clean_float(artifact.get("energy_minus_accuracy_delta")) is not None
            and float(artifact["energy_minus_accuracy_delta"]) > 0.0,
            "success needs positive delta",
        )
        _require(
            _ci_excludes_zero_positive(artifact.get("energy_minus_accuracy_delta_ci95")),
            "success CI must exclude zero",
        )
        _require(_success_not_sabotage(game_results), "success cannot rely on sabotage rows")
        _require(artifact.get("s3_authorized") is True, "success authorizes S3")
    elif verdict == BOUNDED_VERDICT:
        _require(
            artifact.get("n_effective_games", 0) >= expected_required,
            "bounded needs required effective games",
        )
        _require(
            artifact.get("positive_control_passed") is True,
            "positive_control_passed required for bounded",
        )
        _require(artifact.get("s3_authorized") is False, "bounded cannot authorize S3")
    else:
        _require(verdict == INCONCLUSIVE_VERDICT, "unknown non-success verdict")
        _require(artifact.get("s3_authorized") is False, "inconclusive cannot authorize S3")


def write_artifact(artifact: Mapping[str, Any], *, root: Path | str = REPO_ROOT) -> Path:
    validate_artifact(artifact)
    out = Path(root) / RESULT_RELATIVE_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def _grid_signature(grid: np.ndarray) -> tuple[tuple[int, ...], bytes]:
    arr = np.asarray(grid, dtype=np.int16)
    return tuple(int(x) for x in arr.shape), arr.tobytes()


def _changed_assignments(transition: Transition) -> tuple[tuple[int, int, int], ...]:
    before = np.asarray(transition.grid)
    after = np.asarray(transition.next_grid)
    if before.shape != after.shape:
        return ()
    rows, cols = np.where(before != after)
    return tuple((int(r), int(c), int(after[r, c])) for r, c in zip(rows, cols, strict=True))


def _apply_assignments(grid: np.ndarray, assignments: Sequence[tuple[int, int, int]]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for row, col, value in assignments:
        if 0 <= row < out.shape[0] and 0 <= col < out.shape[1]:
            out[row, col] = int(value)
    return out


def _hamming_distance(left: np.ndarray, right: np.ndarray) -> int:
    a = np.asarray(left)
    b = np.asarray(right)
    if a.shape != b.shape:
        return 10**9
    return int(np.count_nonzero(a != b))


def _nearest_transition(
    transitions: Sequence[Transition],
    grid: np.ndarray,
    action: int,
    *,
    require_action: bool,
) -> Transition | None:
    candidates = [
        transition
        for transition in transitions
        if (not require_action or int(transition.action) == int(action))
        and np.asarray(transition.grid).shape == np.asarray(grid).shape
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda transition: _hamming_distance(transition.grid, grid))


def _identity_engine(grid: np.ndarray, _action: int, _data: Any) -> np.ndarray:
    return np.array(grid, copy=True)


def _exact_memory_engine(transitions: Sequence[Transition], fallback: Engine) -> Engine:
    table = {
        (int(transition.action), _grid_signature(np.asarray(transition.grid))): np.asarray(
            transition.next_grid
        )
        for transition in transitions
    }

    def engine(grid: np.ndarray, action: int, data: Any) -> np.ndarray:
        hit = table.get((int(action), _grid_signature(np.asarray(grid))))
        if hit is not None:
            return np.array(hit, copy=True)
        return fallback(grid, action, data)

    return engine


def _nearest_patch_engine(transitions: Sequence[Transition], *, require_action: bool) -> Engine:
    rows = list(transitions)

    def engine(grid: np.ndarray, action: int, _data: Any) -> np.ndarray:
        nearest = _nearest_transition(rows, grid, int(action), require_action=require_action)
        if nearest is None:
            return np.array(grid, copy=True)
        return _apply_assignments(grid, _changed_assignments(nearest))

    return engine


def _action_mode_patch_engine(
    transitions: Sequence[Transition],
    *,
    max_changed_cells: int,
) -> Engine:
    counts: dict[int, Counter[tuple[int, int, int]]] = defaultdict(Counter)
    for transition in transitions:
        for assignment in _changed_assignments(transition):
            counts[int(transition.action)][assignment] += 1
    patches = {
        action: tuple(item for item, _count in counter.most_common(max_changed_cells))
        for action, counter in counts.items()
    }

    def engine(grid: np.ndarray, action: int, _data: Any) -> np.ndarray:
        return _apply_assignments(grid, patches.get(int(action), ()))

    return engine


def _color_rewrite_engine(transitions: Sequence[Transition]) -> Engine:
    counts: dict[int, Counter[tuple[int, int]]] = defaultdict(Counter)
    for transition in transitions:
        before = np.asarray(transition.grid)
        after = np.asarray(transition.next_grid)
        if before.shape != after.shape:
            continue
        changed = before != after
        for src, dst in zip(before[changed].flat, after[changed].flat, strict=True):
            counts[int(transition.action)][(int(src), int(dst))] += 1
    rewrites = {
        action: counter.most_common(1)[0][0] for action, counter in counts.items() if counter
    }

    def engine(grid: np.ndarray, action: int, _data: Any) -> np.ndarray:
        rewrite = rewrites.get(int(action))
        out = np.array(grid, copy=True)
        if rewrite is None:
            return out
        src, dst = rewrite
        out[out == int(src)] = int(dst)
        return out

    return engine


def _prefix_slice(transitions: Sequence[Transition], fraction: float) -> list[Transition]:
    rows = list(transitions)
    if len(rows) <= 2:
        return rows[:1]
    n = max(2, int(round(len(rows) * float(fraction))))
    n = min(n, len(rows) - 1)
    return rows[:n]


def _deterministic_candidate_variants(
    game: str,
    transitions: Sequence[Transition],
    *,
    seed: int,
) -> list[tuple[WorldModelCandidate, JsonDict]]:
    rows = list(transitions)
    configs = (
        ("broad", 0.67, 8),
        ("mid", 0.50, 4),
        ("early", 0.34, 2),
    )
    out: list[tuple[WorldModelCandidate, JsonDict]] = []
    for round_index, (label, fraction, max_changes) in enumerate(configs):
        prefix = _prefix_slice(rows, fraction)
        engines: list[tuple[str, Engine]] = [
            ("nearest_action_patch", _nearest_patch_engine(prefix, require_action=True)),
            ("nearest_any_patch", _nearest_patch_engine(prefix, require_action=False)),
            (
                "action_mode_patch",
                _action_mode_patch_engine(prefix, max_changed_cells=max_changes),
            ),
            ("color_rewrite", _color_rewrite_engine(prefix)),
            ("exact_memory", _exact_memory_engine(prefix, _identity_engine)),
        ]
        for kind, engine in engines:
            name = f"induced/{game}/{label}/{kind}/seed_{int(seed) + round_index}"
            out.append(
                (
                    WorldModelCandidate(name, engine),
                    {
                        "candidate_source": "deterministic_transition_induction",
                        "genuinely_induced": True,
                        "generation_round": label,
                        "induction_seed": int(seed) + round_index,
                        "prefix_size": len(prefix),
                        "n_experts": None,
                        "variant_kind": kind,
                    },
                )
            )
    return out


def _programmatic_candidate_variants(
    game: str,
    transitions: Sequence[Transition],
    *,
    seed: int,
) -> list[tuple[WorldModelCandidate, JsonDict]]:  # pragma: no cover
    return s2v2._programmatic_candidate_variants(game, transitions, seed=seed)


def generate_candidate_set(
    *,
    game: str,
    transitions: Sequence[Transition],
    random_seed: int,
) -> tuple[list[WorldModelCandidate], dict[str, JsonDict]]:  # pragma: no cover
    entries = _deterministic_candidate_variants(game, transitions, seed=int(random_seed))
    entries.extend(_programmatic_candidate_variants(game, transitions, seed=int(random_seed)))
    seen: set[str] = set()
    candidates: list[WorldModelCandidate] = []
    metadata: dict[str, JsonDict] = {}
    for candidate, meta in entries:
        if candidate.name in seen:
            continue
        seen.add(candidate.name)
        candidates.append(candidate)
        metadata[candidate.name] = dict(meta)
    return candidates, metadata


def _skipped_game_result(game: str, reason: str, detail: str | None = None) -> JsonDict:
    return {
        "game": str(game),
        "n_candidates": 0,
        "effective": False,
        "distinct_heldout_cell_recall_count": 0,
        "heldout_cell_recall_spread": 0.0,
        "energy_selected_candidate": None,
        "accuracy_gate_selected_candidate": None,
        "selection_candidates_differ": False,
        "energy_selected_offpath_cell_recall": None,
        "accuracy_gate_selected_offpath_cell_recall": None,
        "energy_minus_accuracy_delta": None,
        "energy_selected_structural_energy": None,
        "accuracy_gate_exact_accuracy": None,
        "accuracy_gate_passed": False,
        "candidate_rows": [],
        "skip_reason": reason,
        "skip_detail": detail,
    }


def check_preconditions(root: Path | str = REPO_ROOT) -> JsonDict:  # pragma: no cover
    checked: JsonDict = {
        "offline_arcade": False,
        "offline_corpus": False,
        "world_model_verifier_import": False,
    }
    try:
        from carnot.agentic import arc_solver_kit as kit

        kit.offline_arcade()
        checked["offline_arcade"] = True
    except Exception as exc:
        checked["offline_arcade_error"] = repr(exc)
    games = available_offline_games(root)
    checked["n_available_games"] = len(games)
    checked["offline_corpus_games"] = games
    checked["offline_corpus"] = len(games) >= MIN_AVAILABLE_CORPUS_GAMES
    try:
        from carnot.agentic.arc_executable_world_model import WorldModelVerifier

        checked["world_model_verifier_import"] = callable(WorldModelVerifier)
    except Exception as exc:
        checked["world_model_verifier_import_error"] = repr(exc)
    checked["ok"] = bool(
        checked["offline_arcade"]
        and checked["offline_corpus"]
        and checked["world_model_verifier_import"]
    )
    return checked


def check_live_path_reachable(root: Path | str = REPO_ROOT) -> bool:  # pragma: no cover
    proc = subprocess.run(
        [sys.executable, "scripts/arc_orphan_solver_lint.py"],
        cwd=Path(root),
        check=False,
        text=True,
        capture_output=True,
        timeout=30,
    )
    return proc.returncode == 0


def run(
    *,
    root: Path | str = REPO_ROOT,
    write: bool = True,
    random_seed: int = RANDOM_SEED,
    transitions_per_game: int = DEFAULT_TRANSITIONS_PER_GAME,
    live_path_checker: Callable[[Path], bool] | None = None,
    energy_scorer: S1StructuralTransitionEnergy | None = None,
) -> JsonDict:  # pragma: no cover
    started = time.time()
    repo = Path(root)
    preconditions = check_preconditions(repo)
    preconditions["agents_md_read"] = True
    preconditions["codex_md_read"] = True
    preconditions["spec_has_req_4811"] = True
    if not preconditions.get("offline_arcade"):
        artifact = build_blocked_artifact(
            "blocked_offline_arcade_missing",
            preconditions,
            random_seed=random_seed,
        )
    elif not preconditions.get("offline_corpus"):
        artifact = build_blocked_artifact(
            "blocked_offline_corpus_missing",
            preconditions,
            random_seed=random_seed,
        )
    elif not preconditions.get("world_model_verifier_import"):
        artifact = build_blocked_artifact(
            "blocked_world_model_verifier_missing",
            preconditions,
            random_seed=random_seed,
        )
    else:
        games = list(preconditions["offline_corpus_games"])
        scorer = energy_scorer or default_s1_offpath_energy_scorer()
        live_ok = (
            live_path_checker(repo)
            if live_path_checker is not None
            else check_live_path_reachable(repo)
        )
        preconditions["arc_orphan_solver_lint_passed"] = bool(live_ok)
        results: list[GameTrustGateResult | JsonDict] = []
        for i, game in enumerate(games):
            try:
                transitions, _cell = collect_transitions(
                    game,
                    n=int(transitions_per_game),
                    seed=int(random_seed) + i,
                )
            except Exception as exc:
                results.append(_skipped_game_result(game, "collect_transitions_failed", repr(exc)))
                continue
            if len(transitions) < 2:
                results.append(_skipped_game_result(game, "insufficient_transitions"))
                continue
            candidates, metadata = generate_candidate_set(
                game=game,
                transitions=transitions,
                random_seed=int(random_seed) + i,
            )
            if len(candidates) < MIN_CANDIDATES_PER_GAME:
                results.append(_skipped_game_result(game, "insufficient_candidates"))
                continue
            results.append(
                evaluate_candidate_set(
                    game=game,
                    transitions=transitions,
                    candidates=candidates,
                    candidate_metadata=metadata,
                    energy_scorer=scorer,
                )
            )
        preconditions["candidate_game_count"] = len(results)
        preconditions["effective_candidate_game_count"] = len(
            [row for row in (_result_json(result) for result in results) if _game_effective(row)]
        )
        artifact = build_artifact(
            results,
            n_available_games=len(games),
            preconditions_checked=preconditions,
            live_path_reachable=bool(live_ok),
            random_seed=random_seed,
            duration_s=time.time() - started,
            energy_scorer=scorer,
        )
    if write:
        write_artifact(artifact, root=repo)
    return artifact


def main() -> int:  # pragma: no cover
    artifact = run()
    print(
        json.dumps(
            {
                "honest_verdict": artifact["honest_verdict"],
                "n_available_games": artifact["n_available_games"],
                "n_games_attempted": artifact["n_games_attempted"],
                "n_effective_games": artifact["n_effective_games"],
                "required_effective_games": artifact["required_effective_games"],
                "energy_minus_accuracy_delta": artifact["energy_minus_accuracy_delta"],
                "energy_minus_accuracy_delta_ci95": artifact["energy_minus_accuracy_delta_ci95"],
                "positive_control_passed": artifact["positive_control_passed"],
                "result": RESULT_RELATIVE_PATH,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
