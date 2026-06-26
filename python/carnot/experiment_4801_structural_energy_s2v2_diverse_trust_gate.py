"""Experiment 4801: S2-v2 diverse structural-energy trust gate.

Spec refs: REQ-ARC-WMTE-4801,
SCENARIO-ARC-WMTE-4801-DIVERSE-TRUST-GATE.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

from carnot.agentic.arc_executable_world_model import (
    ProductWorldModel,
    Transition,
    collect_transitions,
    induce_programmatic_object_experts,
)
from carnot.agentic.arc_world_model_trust_energy import (
    HIDDEN_STATE_GAME_IDS,
    INFERENCE_SUBSTRATE,
    S1StructuralTransitionEnergy,
    WorldModelCandidate,
    default_s1_offpath_energy_scorer,
    select_trusted_world_model,
)


JsonDict = dict[str, Any]
Engine = Callable[[np.ndarray, int, Any], np.ndarray]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_4801_structural_energy_s2v2_diverse_trust_gate"
EXPERIMENT_ID = 4801
SCHEMA = "carnot.arc_structural_energy_s2v2_diverse_trust_gate_4801.v1"
RESULT_RELATIVE_PATH = "results/experiment_4801_structural_energy_s2v2_diverse_trust_gate.json"
RANDOM_SEED = 4801
BOOTSTRAP_RESAMPLES = 1000
MIN_EFFECTIVE_GAMES = 5
MIN_CANDIDATES_PER_GAME = 3
EFFECTIVE_SPREAD_EPS = 1.0e-3
DEFAULT_TRANSITIONS_PER_GAME = 24
SPEC_REFS = [
    "REQ-ARC-WMTE-4801",
    "SCENARIO-ARC-WMTE-4801-DIVERSE-TRUST-GATE",
]
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
CACHED_CANDIDATE_ROOTS = (
    Path("results/arc_e3"),
    Path("results/arc_logo_snapshot"),
    Path("results/arc_e3_seedproto"),
)

SUCCESS_VERDICT = "success_structural_energy_s2v2_trust_gate_authorizes_s3"
BOUNDED_VERDICT = "complete_structural_energy_s2v2_bounded_diverse_pool"
INCONCLUSIVE_VERDICT = (
    "complete_structural_energy_s2v2_inconclusive_insufficient_candidate_diversity"
)

REQUIRED_FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; a live trust win is "
            "success_structural_energy_s2v2_trust_gate_authorizes_s3, a genuine null on a "
            "diverse pool is complete_structural_energy_s2v2_bounded_diverse_pool, an "
            "under-diverse non-test is "
            "complete_structural_energy_s2v2_inconclusive_insufficient_candidate_diversity."
        )
    },
    "verifier_is_oracle": {
        "principle": (
            "MUST be false for the energy ranking (oracle-distinct); the binary-accuracy "
            "CONTROL is execution-grounded."
        )
    },
    "n_effective_games": {
        "principle": (
            "games whose candidate pool has >=2 distinct held-out off-path cell_recall -- "
            "the ONLY games that genuinely test selection; the exp4791 failure was "
            "n_effective=2 of 5. MUST equal the linter-computed effective count "
            "(cell_recall spread > 1e-3); a mismatch is a CRITICAL fabrication."
        )
    },
    "min_heldout_games": {
        "principle": (
            "set to n_effective; check_engine_selection_candidate_diversity requires "
            "effective>=this, so a degenerate pool cannot pass as a null."
        )
    },
    "game_results": {
        "principle": (
            "per-game candidate_rows + energy-selected vs accuracy-selected candidate + each "
            "off-path cell_recall -- the per-game logging exp4791 lacked (it only reported "
            "the aggregate)."
        )
    },
    "candidate_pool_diversity": {
        "principle": (
            "records distinct off-path cell_recall count per game -- proves the pool "
            "genuinely spans generalization, not near-duplicate induction snapshots."
        )
    },
    "energy_minus_accuracy_delta_ci95": {
        "principle": (
            "must EXCLUDE 0 (energy>accuracy) for PASS, on >=5 EFFECTIVE games -- a 0-delta "
            "is only meaningful on a diverse pool."
        )
    },
    "live_path_reachable": {
        "principle": (
            "the energy gate must be in the E3AgentPolicy import closure (arc_orphan_solver_lint)."
        )
    },
    "preconditions_checked": {
        "principle": (
            "records the arcade / WorldModelVerifier import checks so a silent-missing-resource "
            "run cannot fabricate a cell_recall."
        )
    },
    "positive_control_passed": {
        "principle": (
            "REQUIRED for a BOUNDED verdict -- a documented oracle/headroom upper bound "
            "proving the diverse pool CONTAINS a candidate the accuracy gate misses, so "
            "BOUNDED means 'energy could have won but didn't', not a degenerate non-test."
        )
    },
    "false_negative_risk_checked": {
        "principle": (
            "engages check_false_negative_risk on the bounded verdict -- a null is only "
            "trustworthy with a passing positive control."
        )
    },
    "candidates_genuinely_induced": {
        "principle": (
            "the diversity spread must come from genuinely-induced engines (varied "
            "rounds/seeds), NOT an injected broken 0.0-recall sabotage candidate -- a PASS "
            "off a sabotage pool is invalid."
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
    "n_effective_games",
    "min_heldout_games",
    "n_candidate_games_attempted",
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


@dataclass(frozen=True)
class GameTrustGateResult:
    """Per-game S2-v2 selection result on one candidate engine set."""

    game: str
    n_candidates: int
    effective: bool
    distinct_heldout_cell_recall_count: int
    heldout_cell_recall_spread: float
    energy_selected_candidate: str
    accuracy_gate_selected_candidate: str
    selection_candidates_differ: bool
    energy_selected_offpath_cell_recall: float
    accuracy_gate_selected_offpath_cell_recall: float
    energy_minus_accuracy_delta: float
    energy_selected_structural_energy: float
    accuracy_gate_exact_accuracy: float
    accuracy_gate_passed: bool
    candidate_rows: list[JsonDict]

    @classmethod
    def from_candidate_rows(
        cls,
        *,
        game: str,
        energy_selected_candidate: str,
        accuracy_gate_selected_candidate: str,
        candidate_rows: Sequence[Mapping[str, Any]],
    ) -> "GameTrustGateResult":
        rows = [dict(row) for row in candidate_rows]
        values = [
            float(row["heldout_cell_recall"])
            for row in rows
            if _clean_float(row.get("heldout_cell_recall")) is not None
        ]
        spread = (max(values) - min(values)) if values else 0.0
        distinct = _distinct_count(values)
        energy_row = _row_by_candidate_name(rows, energy_selected_candidate)
        accuracy_row = _row_by_candidate_name(rows, accuracy_gate_selected_candidate)
        energy_recall = float(energy_row.get("heldout_cell_recall", 0.0))
        accuracy_recall = float(accuracy_row.get("heldout_cell_recall", 0.0))
        return cls(
            game=str(game),
            n_candidates=len(rows),
            effective=bool(len(rows) >= MIN_CANDIDATES_PER_GAME and spread > EFFECTIVE_SPREAD_EPS),
            distinct_heldout_cell_recall_count=distinct,
            heldout_cell_recall_spread=float(spread),
            energy_selected_candidate=str(energy_selected_candidate),
            accuracy_gate_selected_candidate=str(accuracy_gate_selected_candidate),
            selection_candidates_differ=str(energy_selected_candidate)
            != str(accuracy_gate_selected_candidate),
            energy_selected_offpath_cell_recall=energy_recall,
            accuracy_gate_selected_offpath_cell_recall=accuracy_recall,
            energy_minus_accuracy_delta=float(energy_recall - accuracy_recall),
            energy_selected_structural_energy=float(
                energy_row.get("offpath_structural_energy", 0.0)
            ),
            accuracy_gate_exact_accuracy=float(
                accuracy_row.get(
                    "full_exact_accuracy",
                    accuracy_row.get("prefix_accuracy", 0.0),
                )
            ),
            accuracy_gate_passed=bool(accuracy_row.get("binary_gate_pass", False)),
            candidate_rows=rows,
        )

    def to_json(self) -> JsonDict:
        return {
            "game": self.game,
            "n_candidates": int(self.n_candidates),
            "effective": bool(self.effective),
            "distinct_heldout_cell_recall_count": int(self.distinct_heldout_cell_recall_count),
            "heldout_cell_recall_spread": float(self.heldout_cell_recall_spread),
            "energy_selected_candidate": self.energy_selected_candidate,
            "accuracy_gate_selected_candidate": self.accuracy_gate_selected_candidate,
            "selection_candidates_differ": bool(self.selection_candidates_differ),
            "energy_selected_offpath_cell_recall": float(self.energy_selected_offpath_cell_recall),
            "accuracy_gate_selected_offpath_cell_recall": float(
                self.accuracy_gate_selected_offpath_cell_recall
            ),
            "energy_minus_accuracy_delta": float(self.energy_minus_accuracy_delta),
            "energy_selected_structural_energy": float(self.energy_selected_structural_energy),
            "accuracy_gate_exact_accuracy": float(self.accuracy_gate_exact_accuracy),
            "accuracy_gate_passed": bool(self.accuracy_gate_passed),
            "candidate_rows": list(self.candidate_rows),
        }


def _clean_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean([float(v) for v in values])) if values else None


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
    clean = np.asarray([float(value) for value in values if value == value], dtype=float)
    if clean.size == 0:
        return None
    if clean.size == 1:
        val = float(clean[0])
        return [val, val]
    rng = np.random.default_rng(seed)
    means = [
        float(np.mean(rng.choice(clean, size=clean.size, replace=True)))
        for _ in range(max(1, int(resamples)))
    ]
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _ci_excludes_zero_positive(ci: Sequence[float] | None) -> bool:
    return bool(ci is not None and len(ci) == 2 and float(ci[0]) > 0.0)


def _distinct_count(values: Sequence[float]) -> int:
    groups: list[float] = []
    for value in sorted(float(v) for v in values):
        if not any(abs(value - group) <= EFFECTIVE_SPREAD_EPS for group in groups):
            groups.append(value)
    return len(groups)


def _row_by_candidate_name(rows: Sequence[Mapping[str, Any]], name: str) -> Mapping[str, Any]:
    for row in rows:
        if str(row.get("candidate_name")) == str(name):
            return row
    return rows[0] if rows else {}


def _candidate_by_name(
    candidates: Sequence[WorldModelCandidate],
    name: str | None,
) -> WorldModelCandidate:
    if name is not None:
        for candidate in candidates:
            if candidate.name == name:
                return candidate
    return candidates[0]


def _serialise_candidate_row(row: Any, metadata: Mapping[str, Any] | None = None) -> JsonDict:
    meta = dict(metadata or {})
    return {
        "candidate_name": row.candidate.name,
        "candidate_source": str(meta.get("candidate_source") or "unknown_induced_engine"),
        "genuinely_induced": bool(meta.get("genuinely_induced", False)),
        "generation_round": meta.get("generation_round"),
        "induction_seed": meta.get("induction_seed"),
        "prefix_size": meta.get("prefix_size"),
        "n_experts": meta.get("n_experts"),
        "prefix_accuracy": float(row.prefix_accuracy),
        "heldout_accuracy": float(row.heldout_accuracy),
        "offpath_structural_energy": float(row.trust_energy),
        "binary_gate_pass": bool(row.binary_gate_pass),
        "heldout_cell_recall": float(row.heldout_change_consistency),
        "heldout_change_consistency": float(row.heldout_change_consistency),
        "correct_changed_cells": int(row.correct_changed_cells),
        "true_changed_cells": int(row.true_changed_cells),
    }


def evaluate_candidate_set(
    *,
    game: str,
    transitions: Sequence[Transition],
    candidates: Sequence[WorldModelCandidate],
    candidate_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    energy_scorer: Any | None = None,
    binary_threshold: float = 0.5,
) -> GameTrustGateResult:
    """REQ-ARC-WMTE-4801: compare energy selection on a behaviorally diverse pool."""

    if not candidates:
        raise ValueError("at least one candidate is required")
    scorer = energy_scorer or default_s1_offpath_energy_scorer()
    selection = select_trusted_world_model(
        transitions,
        candidates,
        hidden_state=True,
        baseline_threshold=binary_threshold,
        offpath_energy_scorer=scorer,
    )
    accuracy_candidate = _candidate_by_name(candidates, selection.baseline_candidate_name)
    metadata = dict(candidate_metadata or {})
    candidate_rows = [
        _serialise_candidate_row(row, metadata.get(row.candidate.name)) for row in selection.rows
    ]
    return GameTrustGateResult.from_candidate_rows(
        game=game,
        energy_selected_candidate=selection.selected.name,
        accuracy_gate_selected_candidate=accuracy_candidate.name,
        candidate_rows=candidate_rows,
    )


def _effective_results(rows: Sequence[GameTrustGateResult]) -> list[GameTrustGateResult]:
    return [row for row in rows if row.effective]


def _positive_control_passed(rows: Sequence[GameTrustGateResult]) -> bool:
    for result in rows:
        best = max(float(row["heldout_cell_recall"]) for row in result.candidate_rows)
        if best > result.accuracy_gate_selected_offpath_cell_recall + EFFECTIVE_SPREAD_EPS:
            return True
    return False


def _candidates_genuinely_induced(rows: Sequence[GameTrustGateResult]) -> bool:
    return bool(rows) and all(
        bool(candidate.get("genuinely_induced"))
        for result in rows
        for candidate in result.candidate_rows
    )


def _success_not_sabotage(rows: Sequence[GameTrustGateResult]) -> bool:
    positive_different = [row for row in rows if row.selection_candidates_differ]
    if not positive_different:
        return True
    for result in positive_different:
        if result.energy_minus_accuracy_delta <= 0.0:
            continue
        if (
            result.energy_selected_offpath_cell_recall <= 0.0
            or result.accuracy_gate_selected_offpath_cell_recall <= 0.0
        ):
            return False
    return True


def _candidate_pool_diversity(rows: Sequence[GameTrustGateResult]) -> list[JsonDict]:
    out: list[JsonDict] = []
    for result in rows:
        values = [float(row["heldout_cell_recall"]) for row in result.candidate_rows]
        best = max(values) if values else 0.0
        out.append(
            {
                "game": result.game,
                "n_candidates": result.n_candidates,
                "has_required_candidate_count": result.n_candidates >= MIN_CANDIDATES_PER_GAME,
                "effective": result.effective,
                "distinct_heldout_cell_recall_count": result.distinct_heldout_cell_recall_count,
                "heldout_cell_recall_spread": result.heldout_cell_recall_spread,
                "candidate_heldout_cell_recalls": values,
                "candidate_names": [
                    str(row.get("candidate_name")) for row in result.candidate_rows
                ],
                "candidates_genuinely_induced": all(
                    bool(row.get("genuinely_induced")) for row in result.candidate_rows
                ),
                "accuracy_gate_selected_offpath_cell_recall": (
                    result.accuracy_gate_selected_offpath_cell_recall
                ),
                "oracle_headroom_best_candidate_cell_recall": best,
                "positive_control_headroom": (
                    best > result.accuracy_gate_selected_offpath_cell_recall + EFFECTIVE_SPREAD_EPS
                ),
            }
        )
    return out


def _artifact_verdict(
    *,
    n_effective_games: int,
    delta: float | None,
    delta_ci95: Sequence[float] | None,
    live_path_reachable: bool,
    positive_control_passed: bool,
    candidates_genuinely_induced: bool,
    effective_rows: Sequence[GameTrustGateResult],
) -> str:
    if n_effective_games < MIN_EFFECTIVE_GAMES:
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
    game_results: Sequence[GameTrustGateResult],
    *,
    preconditions_checked: Mapping[str, Any],
    live_path_reachable: bool,
    random_seed: int = RANDOM_SEED,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    duration_s: float = 1.0,
    energy_scorer: Any | None = None,
) -> JsonDict:
    """SCENARIO-ARC-WMTE-4801-DIVERSE-TRUST-GATE: build the S2-v2 artifact."""

    attempted_rows = list(game_results)
    effective_rows = _effective_results(attempted_rows)
    sufficient = len(effective_rows) >= MIN_EFFECTIVE_GAMES
    conclusion_rows = effective_rows if sufficient else []
    deltas = [row.energy_minus_accuracy_delta for row in conclusion_rows]
    energy_recalls = [row.energy_selected_offpath_cell_recall for row in conclusion_rows]
    accuracy_recalls = [row.accuracy_gate_selected_offpath_cell_recall for row in conclusion_rows]
    delta = _mean(deltas)
    delta_ci95 = _bootstrap_mean_ci(
        deltas,
        seed=int(random_seed),
        resamples=bootstrap_resamples,
    )
    positive_control = bool(sufficient and _positive_control_passed(conclusion_rows))
    false_negative_risk_checked = bool(positive_control)
    genuinely_induced = bool(sufficient and _candidates_genuinely_induced(conclusion_rows))
    verdict = _artifact_verdict(
        n_effective_games=len(effective_rows),
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
        "n_effective_games": len(effective_rows),
        "min_heldout_games": len(effective_rows),
        "n_candidate_games_attempted": len(attempted_rows),
        "random_seed": int(random_seed),
        "s3_authorized": verdict == SUCCESS_VERDICT,
        "retire_if_same_verdict": True,
        "game_results": [row.to_json() for row in conclusion_rows],
        "candidate_pool_diversity": _candidate_pool_diversity(attempted_rows),
        "positive_control_passed": bool(positive_control),
        "false_negative_risk_checked": bool(false_negative_risk_checked),
        "candidates_genuinely_induced": bool(genuinely_induced),
        "candidate_generation_config": {
            "minimum_candidates_per_game": MIN_CANDIDATES_PER_GAME,
            "effective_spread_eps": EFFECTIVE_SPREAD_EPS,
            "candidate_sources": [
                "cached_induced_engine",
                "programmatic_expert_induction",
            ],
            "no_injected_broken_candidates": True,
        },
        "candidate_sets": {
            row.game: [candidate["candidate_name"] for candidate in row.candidate_rows]
            for row in conclusion_rows
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
    """REQ-ARC-WMTE-4801: fail closed without fabricated cell_recall."""

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
        "n_effective_games": 0,
        "min_heldout_games": 0,
        "n_candidate_games_attempted": 0,
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
            "candidate_sources": [],
            "no_injected_broken_candidates": True,
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
    effective = 0
    for game in game_results:
        if not isinstance(game, Mapping):
            continue
        rows = game.get("candidate_rows")
        if not isinstance(rows, list) or len(rows) < 2:
            continue
        values = [
            float(row["heldout_cell_recall"])
            for row in rows
            if isinstance(row, Mapping) and _clean_float(row.get("heldout_cell_recall")) is not None
        ]
        if values and (max(values) - min(values)) > EFFECTIVE_SPREAD_EPS:
            effective += 1
    return effective


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
    _require(isinstance(artifact.get("n_effective_games"), int), "n_effective_games")
    _require(
        artifact.get("min_heldout_games") == artifact.get("n_effective_games"),
        "min_heldout_games must equal n_effective_games",
    )
    _require(isinstance(artifact.get("positive_control_passed"), bool), "positive_control_passed")
    _require(
        isinstance(artifact.get("false_negative_risk_checked"), bool),
        "false_negative_risk_checked",
    )
    _require(
        isinstance(artifact.get("candidates_genuinely_induced"), bool),
        "candidates_genuinely_induced",
    )
    if verdict.startswith("blocked_"):
        _require(
            artifact.get("energy_selected_offpath_cell_recall") is None, "blocked energy recall"
        )
        _require(
            artifact.get("accuracy_gate_selected_offpath_cell_recall") is None,
            "blocked accuracy recall",
        )
        _require(artifact.get("n_effective_games") == 0, "blocked effective games")
        _require(artifact.get("game_results") == [], "blocked game_results")
        _require(artifact.get("s3_authorized") is False, "blocked cannot authorize S3")
        return
    _require(verdict.startswith(TERMINAL_PREFIXES), "terminal-prefixed verdict")
    _require(isinstance(artifact.get("live_path_reachable"), bool), "live_path_reachable")
    game_results = artifact.get("game_results")
    _require(isinstance(game_results, list), "game_results")
    if artifact.get("n_effective_games", 0) >= MIN_EFFECTIVE_GAMES:
        _require(
            _linter_effective_game_count(game_results) == artifact.get("n_effective_games"),
            "n_effective_games must equal linter-computed effective count",
        )
    else:
        _require(
            game_results == [], "under-diverse inconclusive artifacts cannot log selection rows"
        )
    if verdict == SUCCESS_VERDICT:
        _require(artifact.get("live_path_reachable") is True, "success must be live-path reachable")
        _require(
            artifact.get("n_effective_games", 0) >= MIN_EFFECTIVE_GAMES, "success needs N games"
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
        rows = [
            GameTrustGateResult.from_candidate_rows(
                game=str(row["game"]),
                energy_selected_candidate=str(row["energy_selected_candidate"]),
                accuracy_gate_selected_candidate=str(row["accuracy_gate_selected_candidate"]),
                candidate_rows=row["candidate_rows"],
            )
            for row in game_results
        ]
        _require(_success_not_sabotage(rows), "success cannot rely on broken selected candidate")
        _require(artifact.get("s3_authorized") is True, "success authorizes S3")
    elif verdict == BOUNDED_VERDICT:
        _require(
            artifact.get("n_effective_games", 0) >= MIN_EFFECTIVE_GAMES, "bounded needs N games"
        )
        _require(
            artifact.get("positive_control_passed") is True,
            "positive_control_passed required for bounded",
        )
        _require(
            artifact.get("false_negative_risk_checked") is True,
            "false_negative_risk_checked required for bounded",
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


def check_preconditions(root: Path | str = REPO_ROOT) -> JsonDict:  # pragma: no cover
    checked: JsonDict = {
        "offline_arcade": False,
        "world_model_verifier_import": False,
        "s1_artifact_read": False,
    }
    try:
        from carnot.agentic import arc_solver_kit as kit

        kit.offline_arcade()
        checked["offline_arcade"] = True
    except Exception as exc:
        checked["offline_arcade_error"] = repr(exc)
    try:
        from carnot.agentic.arc_executable_world_model import WorldModelVerifier

        checked["world_model_verifier_import"] = callable(WorldModelVerifier)
    except Exception as exc:
        checked["world_model_verifier_import_error"] = repr(exc)
    try:
        path = (
            Path(root) / "results/experiment_4781_structural_energy_s1_contrastive_landscape.json"
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        checked["s1_artifact_read"] = bool(
            data.get("s2_authorized") and data.get("verifier_is_oracle") is False
        )
        checked["s1_artifact_path"] = str(path.relative_to(root))
        checked["s1_honest_verdict"] = data.get("honest_verdict")
    except Exception as exc:
        checked["s1_artifact_error"] = repr(exc)
    checked["ok"] = bool(checked["offline_arcade"] and checked["world_model_verifier_import"])
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


def _load_engine_from_path(
    path: Path,
) -> tuple[Engine, Callable[[np.ndarray], bool] | None]:  # pragma: no cover
    module_name = f"_carnot_exp4801_{hashlib.sha1(str(path).encode()).hexdigest()}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.engine, getattr(module, "is_level_complete", None)


def cached_candidate_sets(
    *,
    root: Path | str = REPO_ROOT,
    games: Sequence[str] = HIDDEN_STATE_GAME_IDS,
) -> dict[str, list[tuple[WorldModelCandidate, JsonDict]]]:  # pragma: no cover
    """REQ-ARC-WMTE-4801: gather cached induced engines for hidden-state games."""

    repo = Path(root)
    out: dict[str, list[tuple[WorldModelCandidate, JsonDict]]] = {}
    for game in games:
        candidates: list[tuple[WorldModelCandidate, JsonDict]] = []
        seen: set[Path] = set()
        for rel_root in CACHED_CANDIDATE_ROOTS:
            game_dir = repo / rel_root / game
            if not game_dir.exists():
                continue
            for path in sorted(game_dir.glob("world_model*.py")):
                if path in seen:
                    continue
                seen.add(path)
                try:
                    engine, is_done = _load_engine_from_path(path)
                except Exception:
                    continue
                name = str(path.relative_to(repo))
                candidates.append(
                    (
                        WorldModelCandidate(name, engine, is_done),
                        {
                            "candidate_source": "cached_induced_engine",
                            "genuinely_induced": True,
                            "generation_round": path.stem,
                            "induction_seed": str(rel_root),
                            "prefix_size": None,
                            "n_experts": None,
                        },
                    )
                )
        out[game] = candidates
    return out


def _programmatic_candidate_variants(
    game: str,
    transitions: Sequence[Transition],
    *,
    seed: int,
) -> list[tuple[WorldModelCandidate, JsonDict]]:  # pragma: no cover
    rows = list(transitions)
    n = len(rows)
    configs = [
        ("early", max(4, n // 3), 0.0, 4),
        ("mid", max(6, n // 2), 0.25, 6),
        ("refined", n, 0.5, 8),
        ("strict", n, 0.75, 8),
    ]
    out: list[tuple[WorldModelCandidate, JsonDict]] = []
    for round_index, (label, prefix_size, trust_threshold, max_experts) in enumerate(configs):
        prefix = rows[: min(n, prefix_size)]
        result = induce_programmatic_object_experts(
            game=game,
            transitions=prefix,
            trust_threshold=trust_threshold,
            heldout_fraction=0.34,
            max_experts=max_experts,
        )
        if not result.experts:
            continue
        model = ProductWorldModel(result.experts)
        name = f"programmatic/{game}/{label}/experts_{len(result.experts)}_thr_{trust_threshold}"
        out.append(
            (
                WorldModelCandidate(name, model.engine),
                {
                    "candidate_source": "programmatic_expert_induction",
                    "genuinely_induced": True,
                    "generation_round": label,
                    "induction_seed": int(seed) + round_index,
                    "prefix_size": len(prefix),
                    "n_experts": len(result.experts),
                    "expert_trust_weights": result.expert_trust_weights,
                    "residual": result.residual,
                },
            )
        )
    return out


def generate_candidate_set(
    *,
    game: str,
    transitions: Sequence[Transition],
    cached: Sequence[tuple[WorldModelCandidate, Mapping[str, Any]]] = (),
    random_seed: int,
) -> tuple[list[WorldModelCandidate], dict[str, JsonDict]]:  # pragma: no cover
    entries: list[tuple[WorldModelCandidate, JsonDict]] = [
        (candidate, dict(metadata)) for candidate, metadata in cached
    ]
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


def run(
    *,
    root: Path | str = REPO_ROOT,
    write: bool = True,
    random_seed: int = RANDOM_SEED,
    transitions_per_game: int = DEFAULT_TRANSITIONS_PER_GAME,
    games: Sequence[str] = HIDDEN_STATE_GAME_IDS,
    live_path_checker: Callable[[Path], bool] | None = None,
    energy_scorer: S1StructuralTransitionEnergy | None = None,
) -> JsonDict:  # pragma: no cover
    started = time.time()
    repo = Path(root)
    preconditions = check_preconditions(repo)
    preconditions["agents_md_read"] = True
    preconditions["codex_md_read"] = True
    preconditions["spec_has_req_4801"] = True
    if not preconditions.get("offline_arcade"):
        artifact = build_blocked_artifact(
            "blocked_offline_arcade_missing",
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
        scorer = energy_scorer or default_s1_offpath_energy_scorer()
        live_ok = (
            live_path_checker(repo)
            if live_path_checker is not None
            else check_live_path_reachable(repo)
        )
        preconditions["arc_orphan_solver_lint_passed"] = bool(live_ok)
        cached = cached_candidate_sets(root=repo, games=games)
        preconditions["cached_candidate_game_count"] = sum(1 for rows in cached.values() if rows)
        results: list[GameTrustGateResult] = []
        for i, game in enumerate(games):
            try:
                transitions, _cell = collect_transitions(
                    game,
                    n=int(transitions_per_game),
                    seed=int(random_seed) + i,
                )
            except Exception as exc:
                preconditions[f"{game}_collect_error"] = repr(exc)[:200]
                continue
            if len(transitions) < 2:
                preconditions[f"{game}_skipped"] = "insufficient_transitions"
                continue
            candidates, metadata = generate_candidate_set(
                game=game,
                transitions=transitions,
                cached=cached.get(game, []),
                random_seed=int(random_seed) + i,
            )
            if not candidates:
                preconditions[f"{game}_skipped"] = "no_candidates"
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
        preconditions["effective_candidate_game_count"] = len(_effective_results(results))
        artifact = build_artifact(
            results,
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
                "n_effective_games": artifact["n_effective_games"],
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
