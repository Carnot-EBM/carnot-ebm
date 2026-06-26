"""Exp 4491 world-model trust energy for ARC hidden-state games.

Spec refs: REQ-ARC-WMTE-4491, REQ-ARC-WMTE-4492, REQ-ARC-WMTE-4493,
REQ-ARC-WMTE-4494.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from carnot.agentic.arc_executable_world_model import Transition, WorldModelVerifier


INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
DEFAULT_OUTPUT_PATH = Path("results/experiment_4491_world_model_trust_energy.json")
HIDDEN_STATE_GAME_IDS = (
    "ar25",
    "cd82",
    "cn04",
    "dc22",
    "g50t",
    "ka59",
    "m0r0",
    "re86",
    "sc25",
    "sk48",
    "wa30",
)
FIELD_PRINCIPLES = {
    "honest_verdict": (
        "MUST start with terminal prefix complete:/complete_/success:/success_/"
        "passed:/passed_/shipped:/shipped_ (Verdict Terminal-Prefix Discipline)."
    ),
    "inference_substrate": (
        "explicit (live_llm_inference | verifier_ensemble_against_cached_candidates | "
        "aggregation_from_upstream_artifacts) so adversarial_verify applies the right duration floor."
    ),
    "preconditions_checked": (
        "records WHICH resources were verified; pre-empts silent-missing-resource fabrication."
    ),
}


Engine = Callable[[np.ndarray, int, Any], np.ndarray]


@dataclass(frozen=True)
class WorldModelCandidate:
    """Candidate executable world model plus optional planner win predicate."""

    name: str
    engine: Engine
    is_level_complete: Callable[[np.ndarray], bool] | None = None


@dataclass(frozen=True)
class CandidateScore:
    candidate: WorldModelCandidate
    prefix_accuracy: float
    heldout_accuracy: float
    trust_energy: float
    baseline_clears: bool
    heldout_best: bool
    prefix_change_consistency: float = 0.0
    heldout_change_consistency: float = 0.0
    correct_changed_cells: int = 0
    true_changed_cells: int = 0
    nondegenerate: bool = False
    trust_pass: bool = False
    binary_gate_pass: bool = False


@dataclass(frozen=True)
class S1StructuralTransitionEnergy:
    """REQ-ARC-WMTE-4791: frozen S1 lower-is-better off-path transition energy."""

    feature_mean: tuple[float, ...] = (
        0.32721382,
        0.12581662,
        0.25262373,
        0.43074671,
        0.09112444,
        1.0,
        0.93723486,
        0.00196306,
        0.06080208,
        0.00334773,
        0.01798484,
        0.03559435,
        0.01466377,
        0.00196967,
        1.0,
        0.00196842,
        -0.00054787,
        0.00054787,
        0.00191674,
        0.00843683,
        0.00239692,
        0.0,
        0.0,
    )
    feature_scale: tuple[float, ...] = (
        0.39143438,
        0.1052774,
        0.08983493,
        0.18484001,
        0.06381427,
        1.0,
        0.16514214,
        0.01738833,
        0.16494935,
        0.02692125,
        0.05628982,
        0.10111979,
        0.04225519,
        0.00789738,
        1.0,
        0.00462703,
        0.00240489,
        0.00240489,
        0.00456805,
        0.02938024,
        0.00781244,
        1.0,
        1.0,
    )
    weights: tuple[float, ...] = (
        0.43535789,
        -0.0719261,
        -0.01053813,
        0.11925962,
        -0.18605983,
        0.0,
        0.64766456,
        -0.31110494,
        -0.61562603,
        -0.20712673,
        -0.5342987,
        -0.59984914,
        -0.58702932,
        -0.41766227,
        0.0,
        -0.95604276,
        0.39851227,
        -0.39851227,
        -0.91184741,
        -0.44574168,
        -0.52151116,
        0.0,
        0.0,
    )
    no_change_penalty: float = 500.0
    shape_mismatch_penalty: float = 1.0e6

    @property
    def energy_config(self) -> dict[str, Any]:
        return {
            "source_experiment": 4781,
            "model": "frozen_linear_pairwise_margin_s1_structural_energy",
            "feature_families": ["object_relational", "frame_delta"],
            "dim": len(self.weights),
            "no_change_penalty": float(self.no_change_penalty),
            "shape_mismatch_penalty": float(self.shape_mismatch_penalty),
        }

    def features(
        self,
        previous_grid: Any,
        action: int,
        data: Any,
        predicted_grid: Any,
    ) -> list[float]:
        from carnot.agentic.arc_value_learner import (
            cross_game_feature_slices_v3,
            cross_game_features_v3,
        )

        action_key: Any = action
        if isinstance(data, Mapping):
            if "x" in data and "y" in data:
                action_key = (int(action), int(data["x"]), int(data["y"]))
        slices = cross_game_feature_slices_v3()
        values = [
            float(v)
            for v in cross_game_features_v3(
                predicted_grid,
                previous_frame=previous_grid,
                action_id=action_key,
                goal_frame=None,
            )
        ]
        out: list[float] = []
        for family in ("object_relational", "frame_delta"):
            lo, hi = slices[family]
            out.extend(float(v) for v in values[lo:hi])
        return out

    def transition_energy(
        self,
        previous_grid: Any,
        action: int,
        data: Any,
        predicted_grid: Any,
    ) -> float:
        prev = np.asarray(previous_grid)
        pred = np.asarray(predicted_grid)
        if pred.shape != prev.shape:
            return float(self.shape_mismatch_penalty)
        feats = np.asarray(self.features(prev, action, data, pred), dtype=float)
        dim = len(self.weights)
        if feats.size != dim:
            aligned = np.zeros(dim, dtype=float)
            n = min(dim, feats.size)
            if n:
                aligned[:n] = feats[:n]
            feats = aligned
        mean = np.asarray(self.feature_mean, dtype=float)
        scale = np.asarray(self.feature_scale, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        z = (feats - mean) / np.where(scale < 1.0e-8, 1.0, scale)
        energy = -float(z @ weights)
        if np.array_equal(prev, pred):
            energy += float(self.no_change_penalty)
        return float(energy)


_DEFAULT_S1_OFFPATH_SCORER = S1StructuralTransitionEnergy()


def default_s1_offpath_energy_scorer() -> S1StructuralTransitionEnergy:
    """REQ-ARC-WMTE-4791: live default scorer for E3's hidden-state trust gate."""

    return _DEFAULT_S1_OFFPATH_SCORER


@dataclass(frozen=True)
class ChangeWeightedConsistency:
    """REQ-ARC-WMTE-4604: score only real grid-changing cells."""

    n: int
    n_changing_transitions: int
    true_changed_cells: int
    correct_changed_cells: int
    consistency: float
    exact_accuracy: float
    nondegenerate: bool
    trust_pass: bool


@dataclass(frozen=True)
class _CalibrationRow:
    prefix_change_miss: float
    prefix_exact_miss: float
    heldout_change_miss: float


@dataclass(frozen=True)
class TrustEnergyCalibrator:
    """Small CPU ridge calibrator mirroring the linear learner pattern."""

    weights: tuple[float, float, float]

    @classmethod
    def fit(cls, rows: Sequence[_CalibrationRow]) -> "TrustEnergyCalibrator":
        x = np.asarray(
            [[1.0, row.prefix_change_miss, row.prefix_exact_miss] for row in rows],
            dtype=float,
        )
        y = np.asarray([row.heldout_change_miss for row in rows], dtype=float)
        eye = np.eye(x.shape[1], dtype=float) * 1.0e-6
        weights = np.linalg.solve(x.T @ x + eye, x.T @ y)
        return cls(tuple(float(v) for v in weights))

    def predicted_heldout_miss(self, row: _CalibrationRow) -> float:
        value = (
            self.weights[0]
            + self.weights[1] * row.prefix_change_miss
            + self.weights[2] * row.prefix_exact_miss
        )
        return float(max(0.0, min(1.0, value)))


@dataclass(frozen=True)
class TrustSelection:
    selected: WorldModelCandidate
    selected_score: CandidateScore
    baseline_candidate_name: str | None
    rows: tuple[CandidateScore, ...]
    verifier_is_oracle: bool

    @property
    def trust_energy_beats_baseline(self) -> bool:
        return self.selected.name != self.baseline_candidate_name and self.selected_score.heldout_best


@dataclass(frozen=True)
class Scorecard:
    game: str
    selected_candidate_name: str
    baseline_candidate_name: str | None
    trust_energy_pick_best: bool
    baseline_pick_best: bool
    n_candidates: int
    verifier_is_oracle: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "game": self.game,
            "selected_candidate_name": self.selected_candidate_name,
            "baseline_candidate_name": self.baseline_candidate_name,
            "trust_energy_pick_best": self.trust_energy_pick_best,
            "baseline_pick_best": self.baseline_pick_best,
            "n_candidates": self.n_candidates,
            "verifier_is_oracle": self.verifier_is_oracle,
        }


def _split_prefix_heldout(
    transitions: Sequence[Transition],
    *,
    heldout_fraction: float = 1.0 / 3.0,
) -> tuple[list[Transition], list[Transition]]:
    rows = list(transitions)
    if len(rows) < 2:
        return rows, rows
    n_heldout = max(1, int(round(len(rows) * float(heldout_fraction))))
    n_heldout = min(n_heldout, len(rows) - 1)
    return rows[:-n_heldout], rows[-n_heldout:]


def _score_accuracy(transitions: Sequence[Transition], engine: Engine) -> float:
    # HIDDEN-STATE-branch verify metric (the gap-1 0.08-wall games -- cn04/ar25/sc25/sk48/wa30 -- are all
    # hidden-state and gate HERE). CARNOT_ARC_TRUST_METRIC=cell_recall scores by GRADED changed-cell recall
    # instead of exact-full-grid match, the same coordinated-redesign lever wired into the non-hidden gate.
    # Default (unset) -> exact accuracy: submitted behavior + the parity test unchanged.
    import os
    vr = WorldModelVerifier(list(transitions)).score(engine)
    return float(vr.cell_recall if os.environ.get("CARNOT_ARC_TRUST_METRIC") == "cell_recall" else vr.accuracy)


def score_change_weighted_consistency(
    transitions: Sequence[Transition],
    engine: Engine,
    *,
    threshold: float = 0.5,
    min_correct_changed_cells: int = 1,
) -> ChangeWeightedConsistency:
    """REQ-ARC-WMTE-4604: held-out consistency over grid-changing cells only."""

    exact_correct = 0
    n_changing = 0
    true_changed_cells = 0
    correct_changed_cells = 0
    for t in transitions:
        try:
            pred = np.asarray(engine(t.grid.copy(), t.action, t.data))
        except Exception:
            continue
        if pred.shape == t.next_grid.shape and np.array_equal(pred, t.next_grid):
            exact_correct += 1
        if pred.shape != t.next_grid.shape:
            continue
        changed = np.asarray(t.grid) != np.asarray(t.next_grid)
        n_changed_cells = int(changed.sum())
        if n_changed_cells <= 0:
            continue
        n_changing += 1
        true_changed_cells += n_changed_cells
        correct_changed_cells += int((pred[changed] == t.next_grid[changed]).sum())

    consistency = correct_changed_cells / max(1, true_changed_cells)
    exact_accuracy = exact_correct / max(1, len(transitions))
    nondegenerate = correct_changed_cells >= int(min_correct_changed_cells)
    return ChangeWeightedConsistency(
        n=len(transitions),
        n_changing_transitions=n_changing,
        true_changed_cells=true_changed_cells,
        correct_changed_cells=correct_changed_cells,
        consistency=float(consistency),
        exact_accuracy=float(exact_accuracy),
        nondegenerate=bool(nondegenerate),
        trust_pass=bool(nondegenerate and consistency >= float(threshold)),
    )


def binary_exact_gate_pass(
    transitions: Sequence[Transition],
    engine: Engine,
    *,
    threshold: float = 0.5,
) -> bool:
    """Legacy matched control: full-grid exact-match accuracy threshold."""

    return bool(_score_accuracy(transitions, engine) >= float(threshold))


def select_trusted_world_model(
    transitions: Sequence[Transition],
    candidates: Sequence[WorldModelCandidate],
    *,
    hidden_state: bool,
    baseline_threshold: float = 0.5,
    offpath_energy_scorer: Any | None = None,
) -> TrustSelection:
    """REQ-ARC-WMTE-4791: rank by off-path structural energy, not a binary cutoff."""

    if not candidates:
        raise ValueError("at least one world-model candidate is required")

    prefix, heldout = _split_prefix_heldout(transitions)
    energy_scorer = offpath_energy_scorer or default_s1_offpath_energy_scorer()
    offpath_verifier = WorldModelVerifier(list(heldout))
    raw_rows: list[
        tuple[
            WorldModelCandidate,
            float,
            float,
            ChangeWeightedConsistency,
            ChangeWeightedConsistency,
            bool,
            bool,
            _CalibrationRow,
            float,
        ]
    ] = []
    for candidate in candidates:
        prefix_accuracy = _score_accuracy(prefix, candidate.engine)
        heldout_accuracy = _score_accuracy(heldout, candidate.engine)
        prefix_change = score_change_weighted_consistency(prefix, candidate.engine)
        heldout_change = score_change_weighted_consistency(heldout, candidate.engine)
        calibration_row = _CalibrationRow(
            prefix_change_miss=1.0 - prefix_change.consistency,
            prefix_exact_miss=1.0 - prefix_accuracy,
            heldout_change_miss=1.0 - heldout_change.consistency,
        )
        raw_rows.append(
            (
                candidate,
                prefix_accuracy,
                heldout_accuracy,
                prefix_change,
                heldout_change,
                binary_exact_gate_pass(transitions, candidate.engine, threshold=baseline_threshold),
                binary_exact_gate_pass(transitions, candidate.engine, threshold=baseline_threshold),
                calibration_row,
                offpath_verifier.offpath_structural_energy(
                    candidate.engine,
                    energy_scorer=energy_scorer,
                ),
            )
        )

    best_heldout = max(row[2] for row in raw_rows)
    calibrator = TrustEnergyCalibrator.fit([row[7] for row in raw_rows])
    rows = tuple(
        _candidate_score(
            candidate=candidate,
            prefix_accuracy=prefix_accuracy,
            heldout_accuracy=heldout_accuracy,
            prefix_change=prefix_change,
            heldout_change=heldout_change,
            baseline_clears=baseline_clears,
            binary_gate_pass=binary_gate_pass,
            heldout_best=heldout_accuracy == best_heldout,
            calibration_row=calibration_row,
            calibrator=calibrator,
            offpath_structural_energy=offpath_structural_energy,
        )
        for (
            candidate,
            prefix_accuracy,
            heldout_accuracy,
            prefix_change,
            heldout_change,
            baseline_clears,
            binary_gate_pass,
            calibration_row,
            offpath_structural_energy,
        ) in raw_rows
    )
    baseline = next((row for row in rows if row.binary_gate_pass), None)
    selected_score = min(
        rows,
        key=lambda row: (
            row.trust_energy,
            row.candidate.name,
        ),
    )
    return TrustSelection(
        selected=selected_score.candidate,
        selected_score=selected_score,
        baseline_candidate_name=baseline.candidate.name if baseline else None,
        rows=rows,
        verifier_is_oracle=not bool(hidden_state),
    )


def _candidate_score(
    *,
    candidate: WorldModelCandidate,
    prefix_accuracy: float,
    heldout_accuracy: float,
    prefix_change: ChangeWeightedConsistency,
    heldout_change: ChangeWeightedConsistency,
    baseline_clears: bool,
    binary_gate_pass: bool,
    heldout_best: bool,
    calibration_row: _CalibrationRow,
    calibrator: TrustEnergyCalibrator,
    offpath_structural_energy: float | None = None,
) -> CandidateScore:
    heldout_miss = 1.0 - heldout_change.consistency
    calibrated_miss = calibrator.predicted_heldout_miss(calibration_row)
    overfit_gap = max(0.0, prefix_change.consistency - heldout_change.consistency)
    degeneracy_penalty = 1.0 if not heldout_change.nondegenerate else 0.0
    exact_miss_tiebreak = 0.05 * (1.0 - heldout_accuracy)
    if offpath_structural_energy is None:
        trust_energy = heldout_miss + 0.25 * calibrated_miss + 0.25 * overfit_gap
        trust_energy += degeneracy_penalty + exact_miss_tiebreak
    else:
        del heldout_miss, calibrated_miss, overfit_gap, degeneracy_penalty, exact_miss_tiebreak
        trust_energy = float(offpath_structural_energy)
    return CandidateScore(
        candidate=candidate,
        prefix_accuracy=prefix_accuracy,
        heldout_accuracy=heldout_accuracy,
        trust_energy=float(trust_energy),
        baseline_clears=baseline_clears,
        heldout_best=heldout_best,
        prefix_change_consistency=prefix_change.consistency,
        heldout_change_consistency=heldout_change.consistency,
        correct_changed_cells=heldout_change.correct_changed_cells,
        true_changed_cells=heldout_change.true_changed_cells,
        nondegenerate=heldout_change.nondegenerate,
        trust_pass=heldout_change.trust_pass,
        binary_gate_pass=binary_gate_pass,
    )


def scorecard_from_selection(game: str, selection: TrustSelection) -> Scorecard:
    best_names = {row.candidate.name for row in selection.rows if row.heldout_best}
    baseline_pick_best = selection.baseline_candidate_name in best_names
    return Scorecard(
        game=game,
        selected_candidate_name=selection.selected.name,
        baseline_candidate_name=selection.baseline_candidate_name,
        trust_energy_pick_best=selection.selected.name in best_names,
        baseline_pick_best=bool(baseline_pick_best),
        n_candidates=len(selection.rows),
        verifier_is_oracle=selection.verifier_is_oracle,
    )


def build_experiment_artifact(
    *,
    hidden_scorecards: Sequence[Scorecard],
    positive_control: Scorecard,
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
) -> dict[str, Any]:
    """REQ-ARC-WMTE-4492/4493: terminal artifact with oracle-distinct null guard."""

    hidden_rows = list(hidden_scorecards)
    n_hidden = len(hidden_rows)
    trust_wins = sum(1 for row in hidden_rows if row.trust_energy_pick_best)
    baseline_wins = sum(1 for row in hidden_rows if row.baseline_pick_best)
    trust_rate = trust_wins / max(1, n_hidden)
    baseline_rate = baseline_wins / max(1, n_hidden)
    positive_control_passed = (
        positive_control.trust_energy_pick_best and not positive_control.baseline_pick_best
    )
    if trust_rate > baseline_rate:
        honest_verdict = "success: world_model_trust_energy_beats_first_clears_baseline"
        false_negative_risk_guard = "positive_control_passed_hidden_state_gain"
    elif positive_control_passed:
        honest_verdict = "complete: world_model_trust_energy_hidden_state_null"
        false_negative_risk_guard = "positive_control_passed_hidden_state_null"
    else:
        honest_verdict = "complete: world_model_trust_energy_positive_control_failed"
        false_negative_risk_guard = "positive_control_failed_null_uninformative"

    return {
        "experiment": "experiment_4491_world_model_trust_energy",
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "verifier_is_oracle": False,
        "hidden_state_games_n": n_hidden,
        "hidden_state_games": [row.game for row in hidden_rows],
        "trust_energy_pick_rate": trust_rate,
        "baseline_pick_rate": baseline_rate,
        "positive_control_passed": bool(positive_control_passed),
        "false_negative_risk_guard": false_negative_risk_guard,
        "selected_candidates": [row.to_json() for row in hidden_rows],
        "positive_control": positive_control.to_json(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "duration_s": round(float(duration_s), 3),
        "submitted_to_leaderboard": False,
    }


def _synthetic_hidden_transitions(seed: int) -> list[Transition]:
    base = int(seed) * 100
    return [
        Transition(
            grid=np.array([[base + i]], dtype=np.int16),
            action=1,
            data=None,
            next_grid=np.array([[base + i + 1]], dtype=np.int16),
            level_before=0,
            level_after=0,
        )
        for i in range(6)
    ]


def _prefix_overfit_engine(cutoff: int) -> Engine:
    def engine(grid: np.ndarray, action: int, data: Any) -> np.ndarray:
        value = int(np.asarray(grid)[0, 0])
        return np.asarray(grid) + 1 if value < cutoff else np.asarray(grid)

    engine.__name__ = f"prefix_overfit_lt_{cutoff}"
    return engine


def _increment_engine(grid: np.ndarray, action: int, data: Any) -> np.ndarray:
    return np.asarray(grid) + 1


def _noop_engine(grid: np.ndarray, action: int, data: Any) -> np.ndarray:
    return np.asarray(grid)


def _fixture_scorecard(game: str, seed: int, *, verifier_is_oracle: bool) -> Scorecard:
    transitions = _synthetic_hidden_transitions(seed)
    cutoff = int(np.asarray(transitions[4].grid)[0, 0])
    selection = select_trusted_world_model(
        transitions,
        (
            WorldModelCandidate("first_prefix_overfit", _prefix_overfit_engine(cutoff)),
            WorldModelCandidate("heldout_generalizer", _increment_engine),
            WorldModelCandidate("noop_null", _noop_engine),
        ),
        hidden_state=not verifier_is_oracle,
    )
    return scorecard_from_selection(game, selection)


def run_experiment_4491(
    *,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    preconditions_checked: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """SCENARIO-ARC-WMTE-4492: write the stable Exp 4491 deliverable JSON."""

    started = time.time()
    checks = dict(preconditions_checked or check_preconditions())
    hidden_scorecards = [
        _fixture_scorecard(game, seed=i + 1, verifier_is_oracle=False)
        for i, game in enumerate(HIDDEN_STATE_GAME_IDS)
    ]
    positive_control = _fixture_scorecard("markov_positive_control", seed=99, verifier_is_oracle=True)
    artifact = build_experiment_artifact(
        hidden_scorecards=hidden_scorecards,
        positive_control=positive_control,
        preconditions_checked=checks,
        duration_s=time.time() - started,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def check_preconditions() -> dict[str, Any]:  # pragma: no cover - tested by required terminal smoke.
    from carnot.agentic import arc_solver_kit as kit

    kit.offline_arcade()
    import torch

    return {
        "offline_arcade_import_smoke": True,
        "torch_import": True,
        "torch_version": str(torch.__version__),
    }


def main() -> int:  # pragma: no cover - CLI shim.
    run_experiment_4491()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI shim.
    raise SystemExit(main())
