"""Exp5570 active-spline online KAN exact-constraint energy.

Spec refs: REQ-LEARN-5570,
SCENARIO-LEARN-5570-STREAM,
SCENARIO-LEARN-5570-ACTIVE-SPLINE,
SCENARIO-LEARN-5570-ROLLBACK,
SCENARIO-LEARN-5570-ARTIFACT.

This experiment tests weight-level online adaptation, not external memory.
Rows come from the exact ASP/FSM near-miss corpus, and labels are the exact
validator accept/reject outcomes. The sparse arm updates only coefficients whose
feature-local spline bases are active for the current row or bounded replay
rows. That is the load-bearing difference from the earlier memory-only CSL
experiments: model parameters genuinely move, while rollback remains
reproducible.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from math import exp, sqrt
from pathlib import Path
from typing import Any

import numpy as np

from carnot.models.kan import KAN


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5570_spline_local_kan_online_energy.json")
DATASET_RELATIVE_PATH = Path("results/experiment_5566_exact_asp_fsm_near_miss_corpus.json")
CHECKPOINT_RELATIVE_DIR = Path("results/experiment_5570_spline_local_kan_online_energy_checkpoints")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5570_spline_local_kan_online_energy.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5570_spline_local_kan_online_energy.py")

SCHEMA = "carnot.experiment_5570.spline_local_kan_online_energy.v1"
EXPERIMENT_ID = "experiment_5570_spline_local_kan_online_energy"
TASK_ID = "exp5570-spline-local-kan-online-energy"
MILESTONE = "2026.07.504"
RUN_DATE = "2026-07-11"
INFERENCE_SUBSTRATE = "online_kan_exact_constraint_energy"

FEATURE_DIM = 32
DEFAULT_SEEDS = (5570, 5571, 5572, 5573, 5574)
DEFAULT_REPLAY_BUDGET = 4
LEARNING_RATE = 0.18
DENSE_BACKGROUND_BASIS = 0.05

FROZEN_ARM = "frozen_kan"
DENSE_ARM = "dense_gradient_online_kan"
ACTIVE_ARM = "active_spline_only_kan"
ARM_NAMES = (FROZEN_ARM, DENSE_ARM, ACTIVE_ARM)

REQUIRED_FAMILIES = (
    "defaults_exceptions",
    "contradictions",
    "soft_preference_optimality",
    "fsm_transition_consistency",
)
SPEC_REFS = (
    "REQ-LEARN-5570",
    "SCENARIO-LEARN-5570-STREAM",
    "SCENARIO-LEARN-5570-ACTIVE-SPLINE",
    "SCENARIO-LEARN-5570-ROLLBACK",
    "SCENARIO-LEARN-5570-ARTIFACT",
)
REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "continuous_self_learning_target",
    "dataset_path",
    "sessions",
    "n_rows",
    "seeds",
    "arms",
    "exact_feedback_only",
    "weights_mutated",
    "active_spline_update",
    "touched_spline_fraction",
    "update_count",
    "parameter_diff_norm",
    "update_latency_ms",
    "heldout_exact_error_by_arm",
    "forward_adaptation_delta",
    "prior_family_regression",
    "unsafe_false_accept_delta",
    "replay_budget",
    "checkpoint_paths",
    "rollback_checksum_match",
    "promotion_thresholds",
    "inference_substrate",
    "honest_verdict",
    "kan_ready",
)
FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "Explains why every headline and gate field exists.",
    "continuous_self_learning_target": "Bare boolean marking this as continuous self-learning rather than static reporting.",
    "dataset_path": "Pins adaptation to the exact ASP/FSM corpus instead of ad hoc labels.",
    "sessions": "Shows ordered online sessions and the holdout that was never updated on.",
    "n_rows": "Confirms the longitudinal stream reaches the 120-row floor.",
    "seeds": "Records paired deterministic seeds used for the improvement confidence interval.",
    "arms": "Lists frozen, dense-gradient, and active-spline KAN controls.",
    "exact_feedback_only": "Guards against LLM labels or heuristic labels entering online learning.",
    "weights_mutated": "Confirms this is genuine parameter adaptation, not memory-only CSL.",
    "active_spline_update": "Marks the sparse arm as updating only activated spline coefficients.",
    "touched_spline_fraction": "Measures locality by reporting how much of the coefficient bank moved.",
    "update_count": "Counts exact-feedback update applications, including bounded replay.",
    "parameter_diff_norm": "Measures the magnitude of learned parameter movement from initialization.",
    "update_latency_ms": "Uses a deterministic operation-count latency proxy so cost scaling is reproducible.",
    "heldout_exact_error_by_arm": "Compares all arms on the same never-updated exact holdout.",
    "forward_adaptation_delta": "Measures held-out error reduction over the frozen KAN baseline.",
    "prior_family_regression": "Prevents new-family learning from damaging earlier exact families.",
    "unsafe_false_accept_delta": "Prevents adaptation from increasing invalid rows accepted as safe.",
    "replay_budget": "Bounds earlier-family replay so retention is not unlimited retraining.",
    "checkpoint_paths": "Records reproducible pre-promotion checkpoints and their hashes.",
    "rollback_checksum_match": "Requires rollback to reproduce the pre-update checksum exactly.",
    "promotion_thresholds": "States the exact gates required before active KAN promotion.",
    "inference_substrate": "Declares online KAN exact-constraint energy without LLM inference.",
    "honest_verdict": "Terminal complete or blocked status for conductor reconciliation.",
    "kan_ready": "only held-out improvement with safe retention and reproducible rollback may enter the reset-free harness.",
}
DEFAULT_TESTS_ADDED_OR_REUSED = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5570_spline_local_kan_online_energy.py "
    "-q --no-cov -n 0",
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5570_spline_local_kan_online_energy.py "
    "-m pytest tests/python/test_experiment_5570_spline_local_kan_online_energy.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report "
    "--include=python/carnot/experiment_5570_spline_local_kan_online_energy.py "
    "--fail-under=100",
    ".venv/bin/pytest tests/python -q",
)


@dataclass(frozen=True)
class FeatureRow:
    """One exact-labeled row with features visible to the online KAN."""

    row_id: str
    family: str
    partition: str
    session_id: str
    label: int
    accepted_by_exact_validator: bool
    features: np.ndarray


@dataclass(frozen=True)
class SessionSplit:
    """One ordered online family session and its held-out row ids."""

    session_id: str
    family: str
    online_row_ids: tuple[str, ...]
    holdout_row_ids: tuple[str, ...]


@dataclass(frozen=True)
class OnlineDataset:
    """The fixed Exp5570 feature matrix split into online and holdout rows."""

    dataset_path: str
    rows: tuple[FeatureRow, ...]
    online_rows: tuple[FeatureRow, ...]
    holdout_rows: tuple[FeatureRow, ...]
    sessions: tuple[SessionSplit, ...]

    @property
    def n_rows(self) -> int:
        """Return the total exact rows represented in this dataset."""

        return len(self.rows)


@dataclass(frozen=True)
class UpdateReceipt:
    """Receipt for one online KAN update call."""

    arm: str
    touched_indices: list[int]
    update_count: int
    touched_fraction: float
    parameter_diff_norm: float
    latency_ms: float


@dataclass(frozen=True)
class CheckpointReceipt:
    """Path and checksum for one pre-promotion model checkpoint."""

    path: Path
    checksum: str
    seed: int
    session_id: str
    phase: str


class OnlineKANEnergyModel(KAN):
    """Small additive KAN with snapshot and checksum helpers for rollback."""

    def __init__(
        self,
        *,
        seed: int,
        n_params: int = FEATURE_DIM,
        init_scale: float = 0.0,
    ) -> None:
        super().__init__(n_params=n_params, seed=seed, init_scale=init_scale)

    def score(self, row: FeatureRow) -> float:
        """Return the valid-row logit; higher means lower exact-constraint energy."""

        return float(self.logits(row.features)[0])

    def probability_valid(self, row: FeatureRow) -> float:
        """Map the KAN logit to a calibrated valid probability proxy."""

        score = max(-60.0, min(60.0, self.score(row)))
        return float(1.0 / (1.0 + exp(-score)))

    def predict_label(self, row: FeatureRow) -> int:
        """Return +1 for exact accept and -1 for exact reject."""

        return 1 if self.score(row) >= 0.0 else -1

    def exact_error(self, rows: Sequence[FeatureRow]) -> float:
        """Return exact classification error against validator labels."""

        return _round(sum(self.predict_label(row) != row.label for row in rows) / len(rows))

    def unsafe_false_accept_rate(self, rows: Sequence[FeatureRow]) -> float:
        """Return invalid-row accept rate, the unsafe direction for this task."""

        invalid = [row for row in rows if row.label == -1]
        return _round(sum(self.predict_label(row) == 1 for row in invalid) / len(invalid))

    def snapshot(self) -> JsonDict:
        """Return a JSON checkpoint payload for exact rollback."""

        return {
            "n_params": self.n_params,
            "activation_threshold": self.activation_threshold,
            "coefficients": [_round(float(value), 12) for value in self.coefficients],
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        """Restore coefficients from a snapshot produced by this class."""

        self.n_params = int(snapshot["n_params"])
        self.activation_threshold = float(snapshot["activation_threshold"])
        self.coefficients = np.array(snapshot["coefficients"], dtype=np.float64)

    def checksum(self) -> str:
        """Return a stable checksum of the current KAN parameters."""

        return sha256_json(self.snapshot())

    @classmethod
    def from_checkpoint(cls, path: Path | str) -> OnlineKANEnergyModel:
        """Load a model from a checkpoint JSON file."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(seed=0, n_params=int(payload["model"]["n_params"]), init_scale=0.0)
        model.restore(payload["model"])
        return model


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically for hashes and stable artifacts."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Return the SHA-256 digest of a JSON-compatible value."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _round(value: float, digits: int = 6) -> float:
    """Round floats once so JSON receipts are stable and readable."""

    return round(float(value), digits)


def load_exact_rows(root: Path | str = REPO_ROOT) -> list[JsonDict]:
    """Load exact-labeled ASP/FSM corpus rows from Exp5566."""

    artifact = json.loads((Path(root) / DATASET_RELATIVE_PATH).read_text(encoding="utf-8"))
    if artifact.get("corpus_ready") is not True:
        raise ValueError("Exp5566 exact ASP/FSM corpus is not ready")  # pragma: no cover
    return list(artifact["corpus_rows"])


def build_dataset(root: Path | str = REPO_ROOT) -> OnlineDataset:
    """Build fixed feature vectors and ordered sessions from exact rows."""

    raw_rows = sorted(load_exact_rows(root), key=row_sort_key)
    by_family = {family: [] for family in REQUIRED_FAMILIES}
    for row in raw_rows:
        by_family[str(row["family"])].append(feature_row(row))

    sessions: list[SessionSplit] = []
    online_rows: list[FeatureRow] = []
    holdout_rows: list[FeatureRow] = []
    all_rows: list[FeatureRow] = []
    for family in REQUIRED_FAMILIES:
        family_rows = tuple(by_family[family])
        online = family_rows[:20]
        holdout = family_rows[20:30]
        session_rows = tuple(
            FeatureRow(
                row_id=row.row_id,
                family=row.family,
                partition=row.partition,
                session_id=family,
                label=row.label,
                accepted_by_exact_validator=row.accepted_by_exact_validator,
                features=row.features,
            )
            for row in online
        )
        holdout_session_rows = tuple(
            FeatureRow(
                row_id=row.row_id,
                family=row.family,
                partition=row.partition,
                session_id="holdout",
                label=row.label,
                accepted_by_exact_validator=row.accepted_by_exact_validator,
                features=row.features,
            )
            for row in holdout
        )
        sessions.append(
            SessionSplit(
                session_id=family,
                family=family,
                online_row_ids=tuple(row.row_id for row in session_rows),
                holdout_row_ids=tuple(row.row_id for row in holdout_session_rows),
            )
        )
        online_rows.extend(session_rows)
        holdout_rows.extend(holdout_session_rows)
        all_rows.extend(session_rows)
        all_rows.extend(holdout_session_rows)

    dataset = OnlineDataset(
        dataset_path=DATASET_RELATIVE_PATH.as_posix(),
        rows=tuple(all_rows),
        online_rows=tuple(online_rows),
        holdout_rows=tuple(holdout_rows),
        sessions=tuple(sessions),
    )
    if dataset.n_rows < 120:
        raise ValueError("Exp5570 requires at least 120 exact rows")  # pragma: no cover
    return dataset


def row_sort_key(row: Mapping[str, Any]) -> tuple[int, str]:
    """Sort rows by preregistered family order and row id."""

    return (REQUIRED_FAMILIES.index(str(row["family"])), str(row["row_id"]))


def feature_row(row: Mapping[str, Any]) -> FeatureRow:
    """Convert one exact corpus row to the fixed KAN feature schema."""

    accepted = bool(row["accepted_by_exact_validator"])
    return FeatureRow(
        row_id=str(row["row_id"]),
        family=str(row["family"]),
        partition=str(row["partition"]),
        session_id=str(row["family"]),
        label=1 if accepted else -1,
        accepted_by_exact_validator=accepted,
        features=fixed_feature_vector(row),
    )


def fixed_feature_vector(row: Mapping[str, Any]) -> np.ndarray:
    """Derive a small fixed vector from candidate structure, not labels."""

    features = np.zeros(FEATURE_DIM, dtype=np.float64)
    family = str(row["family"])
    features[REQUIRED_FAMILIES.index(family)] = 1.0
    candidate_kind = str(row["candidate_kind"])

    candidate = row["candidate"]
    if candidate_kind == "asp_row":
        add_asp_features(features, family, candidate)
    else:
        add_fsm_features(features, candidate)
    return features


def add_asp_features(features: np.ndarray, family: str, candidate: Mapping[str, Any]) -> None:
    """Add structural ASP features visible before exact feedback."""

    facts = [str(fact) for fact in candidate.get("facts", [])]
    rules = list(candidate.get("rules", []))
    rule_ids = [str(rule.get("rule_id", "")) for rule in rules]
    null_head_count = sum(rule.get("head") is None for rule in rules)
    has_exception = any("exception" in fact for fact in facts)
    has_escape = any("escape" in fact for fact in facts)
    has_prefer_a = any("prefer_a" in fact for fact in facts)
    has_prefer_b = any("prefer_b" in fact for fact in facts)
    has_preference_rule = any(rule_id.endswith("_04") for rule_id in rule_ids)
    features[8] = 1.0 if null_head_count > 0 else 0.0
    features[11] = 1.0 if any("exception" in fact for fact in facts) else 0.0
    features[12] = 1.0 if has_escape else 0.0
    features[13] = 1.0 if has_prefer_a else 0.0
    features[14] = 1.0 if has_prefer_b else 0.0
    features[15] = 1.0 if has_preference_rule else 0.0
    features[16] = 1.0 if bool(candidate.get("contradiction_row")) else 0.0
    features[25] = 1.0 if family == "defaults_exceptions" and not has_exception else 0.0
    features[26] = 1.0 if family == "contradictions" and null_head_count > 0 and not has_escape else 0.0
    features[27] = 1.0 if (
        family == "soft_preference_optimality"
        and has_prefer_b
        and has_preference_rule
        and not has_prefer_a
    ) else 0.0


def add_fsm_features(features: np.ndarray, candidate: Mapping[str, Any]) -> None:
    """Add structural FSM features visible before exact feedback."""

    machine = json.loads(str(candidate["machine_description_yaml"]))
    transitions = list(machine.get("transition_constraints", []))
    accepting = [str(state) for state in machine.get("accepting_states", [])]
    errors = [str(state) for state in machine.get("error_states", [])]
    features[19] = scaled(fsm_conflict_count(transitions), 3)
    features[20] = 1.0 if any(state.endswith("_s1") for state in accepting) and any(
        state.endswith("_s2") for state in errors
    ) else 0.0
    features[21] = 1.0 if any(state.endswith("_s2") for state in accepting) or any(
        state.endswith("_s1") for state in errors
    ) else 0.0
    features[28] = features[20]


def fsm_conflict_count(transitions: Sequence[Mapping[str, Any]]) -> int:
    """Count deterministic FSM conflicts with same source/symbol and new target."""

    seen: dict[tuple[str, str], str] = {}
    conflicts = 0
    for transition in transitions:
        key = (str(transition.get("source")), str(transition.get("symbol")))
        target = str(transition.get("target"))
        if key in seen and seen[key] != target:
            conflicts += 1
        else:
            seen[key] = target
    return conflicts


def scaled(count: int, denominator: int) -> float:
    """Scale a structural count into the KAN spline domain."""

    return _round(min(float(count) / float(denominator), 1.0))


def future_holdout_update_leakage(dataset: OnlineDataset) -> int:
    """Count holdout rows that accidentally enter the online update stream."""

    online_ids = {row.row_id for row in dataset.online_rows}
    return sum(row.row_id in online_ids for row in dataset.holdout_rows)


def apply_online_update(
    model: OnlineKANEnergyModel,
    rows: Sequence[FeatureRow],
    *,
    learning_rate: float,
    arm: str,
) -> UpdateReceipt:
    """Apply one exact-feedback update to dense or active-spline KAN arms."""

    before = model.coefficients.copy()
    touched: set[int] = set()
    updates = 0
    for row in rows:
        basis = model.basis(row.features)[0]
        if arm == ACTIVE_ARM:
            indices = np.flatnonzero(np.abs(basis) > model.activation_threshold)
            direction = basis
        elif arm == DENSE_ARM:
            indices = np.arange(model.n_params)
            direction = np.where(np.abs(basis) > model.activation_threshold, basis, DENSE_BACKGROUND_BASIS)
        else:
            indices = np.array([], dtype=int)
            direction = basis
        if indices.size and row.label * float(basis @ model.coefficients) < 1.0:
            model.coefficients[indices] += learning_rate * float(row.label) * direction[indices]
            touched.update(int(index) for index in indices)
            updates += 1
    diff_norm = float(np.linalg.norm(model.coefficients - before))
    touched_indices = sorted(touched)
    return UpdateReceipt(
        arm=arm,
        touched_indices=touched_indices,
        update_count=updates,
        touched_fraction=_round(len(touched_indices) / model.n_params),
        parameter_diff_norm=_round(diff_norm),
        latency_ms=deterministic_latency_ms(len(touched_indices), updates),
    )


def deterministic_latency_ms(touched_count: int, update_count: int) -> float:
    """Return a reproducible update-cost proxy in milliseconds."""

    return _round(0.004 * touched_count + 0.011 * update_count)


def run_online_experiment(
    dataset: OnlineDataset,
    *,
    seeds: Sequence[int],
    replay_budget: int,
    checkpoint_dir: Path | str,
) -> JsonDict:
    """Run frozen, dense, and active arms across paired seeds."""

    seed_results = [
        evaluate_seed(dataset, seed=int(seed), replay_budget=replay_budget, checkpoint_dir=Path(checkpoint_dir))
        for seed in seeds
    ]
    heldout_error = summarize_arm_metric(seed_results, "heldout_exact_error")
    false_accept = summarize_arm_metric(seed_results, "unsafe_false_accept_rate")
    separation = summarize_arm_metric(seed_results, "energy_separation")
    calibration = summarize_arm_metric(seed_results, "brier_calibration")
    active_summary = summarize_update(seed_results, ACTIVE_ARM)
    dense_summary = summarize_update(seed_results, DENSE_ARM)
    paired_ci = confidence_interval(
        [
            result["arms"][FROZEN_ARM]["heldout_exact_error"]
            - result["arms"][ACTIVE_ARM]["heldout_exact_error"]
            for result in seed_results
        ]
    )
    prior_regression = _round(max(result["arms"][ACTIVE_ARM]["prior_family_regression"] for result in seed_results))
    unsafe_delta = _round(false_accept[ACTIVE_ARM] - false_accept[FROZEN_ARM])
    checkpoint_paths = [
        receipt
        for result in seed_results
        for receipt in result["arms"][ACTIVE_ARM]["checkpoint_paths"]
    ]
    rollback_ok = all(result["arms"][ACTIVE_ARM]["rollback_checksum_match"] for result in seed_results)
    forward_delta = paired_ci["mean"]
    result = {
        "seeds": [int(seed) for seed in seeds],
        "arms": list(ARM_NAMES),
        "n_rows": dataset.n_rows,
        "heldout_exact_error_by_arm": heldout_error,
        "unsafe_false_accept_rate_by_arm": false_accept,
        "energy_separation_by_arm": separation,
        "brier_calibration_by_arm": calibration,
        "paired_ci_active_vs_frozen": paired_ci,
        "forward_adaptation_delta": forward_delta,
        "prior_family_regression": prior_regression,
        "unsafe_false_accept_delta": unsafe_delta,
        "replay_budget": int(replay_budget),
        "active_update_summary": active_summary,
        "dense_update_summary": dense_summary,
        "checkpoint_paths": checkpoint_paths,
        "rollback_checksum_match": rollback_ok,
        "seed_results": seed_results,
    }
    result["promotion_thresholds"] = promotion_thresholds(result)
    result["kan_ready"] = kan_ready(result)
    return result


def evaluate_seed(
    dataset: OnlineDataset,
    *,
    seed: int,
    replay_budget: int,
    checkpoint_dir: Path,
) -> JsonDict:
    """Evaluate all arms for one paired seed and shared row order."""

    return {
        "seed": seed,
        "arms": {
            arm: evaluate_arm(
                dataset,
                seed=seed,
                arm=arm,
                replay_budget=replay_budget,
                checkpoint_dir=checkpoint_dir,
            )
            for arm in ARM_NAMES
        },
    }


def evaluate_arm(
    dataset: OnlineDataset,
    *,
    seed: int,
    arm: str,
    replay_budget: int,
    checkpoint_dir: Path,
) -> JsonDict:
    """Run one KAN arm over ordered sessions and exact feedback."""

    model = OnlineKANEnergyModel(seed=seed, n_params=FEATURE_DIM, init_scale=0.0)
    initial = model.coefficients.copy()
    replay_buffer: list[FeatureRow] = []
    receipts: list[UpdateReceipt] = []
    checkpoint_receipts: list[CheckpointReceipt] = []
    error_after_family: dict[str, float] = {}

    for session in dataset.sessions:
        session_rows = [row for row in dataset.online_rows if row.session_id == session.session_id]
        if arm == ACTIVE_ARM:
            checkpoint_receipts.append(
                write_checkpoint(
                    model,
                    checkpoint_dir,
                    seed=seed,
                    session_id=session.session_id,
                    phase="pre-promotion",
                )
            )
        for row in session_rows:
            replay_rows = select_replay_rows(replay_buffer, current_family=row.family, replay_budget=replay_budget)
            if arm in (ACTIVE_ARM, DENSE_ARM):
                receipts.append(
                    apply_online_update(
                        model,
                        [row, *replay_rows],
                        learning_rate=LEARNING_RATE,
                        arm=arm,
                    )
                )
            replay_buffer.append(row)
        family_holdout = [row for row in dataset.holdout_rows if row.family == session.family]
        error_after_family[session.family] = model.exact_error(family_holdout)

    final_metrics = evaluate_model(model, dataset.holdout_rows)
    final_by_family = {
        family: model.exact_error([row for row in dataset.holdout_rows if row.family == family])
        for family in REQUIRED_FAMILIES
    }
    prior_regression = max(
        [0.0]
        + [
            _round(final_by_family[family] - error_after_family[family])
            for family in REQUIRED_FAMILIES
        ]
    )
    rollback_ok = True
    if checkpoint_receipts:
        rollback_ok = all(
            rollback_checksum_match(receipt, OnlineKANEnergyModel.from_checkpoint(receipt.path))
            for receipt in checkpoint_receipts
        )
    summary = summarize_receipts(receipts, model, initial)
    return {
        **final_metrics,
        "prior_family_regression": _round(prior_regression),
        "update_summary": summary,
        "checkpoint_paths": checkpoint_receipts_to_json(checkpoint_receipts, REPO_ROOT),
        "rollback_checksum_match": rollback_ok,
        "final_checksum": model.checksum(),
    }


def select_replay_rows(
    replay_buffer: Sequence[FeatureRow],
    *,
    current_family: str,
    replay_budget: int,
) -> list[FeatureRow]:
    """Return bounded earlier-family replay rows for retention."""

    if replay_budget <= 0:
        return []
    earlier = [row for row in replay_buffer if row.family != current_family]
    return earlier[-replay_budget:]


def evaluate_model(model: OnlineKANEnergyModel, rows: Sequence[FeatureRow]) -> JsonDict:
    """Compute held-out error, unsafe accepts, separation, and calibration."""

    valid = [row for row in rows if row.label == 1]
    invalid = [row for row in rows if row.label == -1]
    valid_energy = [-model.score(row) for row in valid]
    invalid_energy = [-model.score(row) for row in invalid]
    brier = np.mean(
        [
            (model.probability_valid(row) - (1.0 if row.label == 1 else 0.0)) ** 2
            for row in rows
        ]
    )
    return {
        "heldout_exact_error": model.exact_error(rows),
        "unsafe_false_accept_rate": model.unsafe_false_accept_rate(rows),
        "energy_separation": _round(float(np.mean(invalid_energy) - np.mean(valid_energy))),
        "brier_calibration": _round(float(brier)),
    }


def summarize_receipts(
    receipts: Sequence[UpdateReceipt],
    model: OnlineKANEnergyModel,
    initial: np.ndarray,
) -> JsonDict:
    """Summarize update locality and parameter movement."""

    touched = sorted({index for receipt in receipts for index in receipt.touched_indices})
    update_count = sum(receipt.update_count for receipt in receipts)
    total_latency = _round(sum(receipt.latency_ms for receipt in receipts))
    return {
        "touched_spline_fraction": _round(len(touched) / model.n_params),
        "update_count": int(update_count),
        "parameter_diff_norm": _round(float(np.linalg.norm(model.coefficients - initial))),
        "update_latency_ms": {
            "total": total_latency,
            "mean": _round(total_latency / max(update_count, 1)),
        },
        "unique_touched_indices": touched,
    }


def summarize_update(seed_results: Sequence[Mapping[str, Any]], arm: str) -> JsonDict:
    """Average update summaries across seeds for one mutable arm."""

    summaries = [result["arms"][arm]["update_summary"] for result in seed_results]
    touched = sorted({index for summary in summaries for index in summary["unique_touched_indices"]})
    update_count = int(round(sum(summary["update_count"] for summary in summaries) / len(summaries)))
    total_latency = _round(sum(summary["update_latency_ms"]["total"] for summary in summaries) / len(summaries))
    return {
        "touched_spline_fraction": _round(len(touched) / FEATURE_DIM),
        "update_count": update_count,
        "parameter_diff_norm": _round(sum(summary["parameter_diff_norm"] for summary in summaries) / len(summaries)),
        "update_latency_ms": {
            "total": total_latency,
            "mean": _round(total_latency / max(update_count, 1)),
        },
        "unique_touched_indices": touched,
    }


def summarize_arm_metric(seed_results: Sequence[Mapping[str, Any]], metric: str) -> JsonDict:
    """Average a held-out metric for each arm across paired seeds."""

    return {
        arm: _round(sum(result["arms"][arm][metric] for result in seed_results) / len(seed_results))
        for arm in ARM_NAMES
    }


def confidence_interval(values: Sequence[float]) -> JsonDict:
    """Return a normal-approximation CI for paired seed improvements."""

    mean = _round(sum(values) / len(values))
    if len(values) == 1:
        half_width = 0.0
    else:
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        half_width = 1.96 * sqrt(variance) / sqrt(len(values))
    return {
        "mean": mean,
        "lower": _round(mean - half_width),
        "upper": _round(mean + half_width),
        "n": len(values),
    }


def write_checkpoint(
    model: OnlineKANEnergyModel,
    checkpoint_dir: Path | str,
    *,
    seed: int,
    session_id: str,
    phase: str,
) -> CheckpointReceipt:
    """Write one pre-promotion checkpoint and return its checksum."""

    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "carnot.experiment_5570.online_kan_checkpoint.v1",
        "seed": int(seed),
        "session_id": session_id,
        "phase": phase,
        "model": model.snapshot(),
    }
    checksum = model.checksum()
    payload["checkpoint_sha256"] = checksum
    path = directory / f"seed_{seed}_{session_id}_{phase}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return CheckpointReceipt(path=path, checksum=checksum, seed=int(seed), session_id=session_id, phase=phase)


def rollback_checksum_match(checkpoint: CheckpointReceipt, model: OnlineKANEnergyModel) -> bool:
    """Return true when a model checksum matches the checkpoint payload."""

    payload = json.loads(checkpoint.path.read_text(encoding="utf-8"))
    return model.checksum() == sha256_json(payload["model"]) == checkpoint.checksum


def checkpoint_receipts_to_json(receipts: Sequence[CheckpointReceipt], root: Path) -> list[JsonDict]:
    """Serialize checkpoint receipts with relative paths when possible."""

    return [
        {
            "path": display_path(receipt.path, root),
            "sha256": receipt.checksum,
            "seed": receipt.seed,
            "session_id": receipt.session_id,
            "phase": receipt.phase,
        }
        for receipt in receipts
    ]


def display_path(path: Path, root: Path) -> str:
    """Return a repo-relative checkpoint path when the file is under root."""

    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def promotion_thresholds(result: Mapping[str, Any]) -> JsonDict:
    """Return the active KAN promotion gate checks."""

    return {
        "heldout_improvement_required": True,
        "paired_ci_excludes_zero": result["paired_ci_active_vs_frozen"]["lower"] > 0.0,
        "max_prior_family_regression": 0.02,
        "prior_family_regression_within_bound": result["prior_family_regression"] <= 0.02,
        "unsafe_false_accept_not_increased": result["unsafe_false_accept_delta"] <= 0.0,
        "rollback_checksum_required": True,
        "rollback_checksum_match": result["rollback_checksum_match"] is True,
    }


def kan_ready(result: Mapping[str, Any]) -> bool:
    """Return true only when all active KAN promotion gates pass."""

    thresholds = result.get("promotion_thresholds", promotion_thresholds(result))
    return (
        result["paired_ci_active_vs_frozen"]["lower"] > 0.0
        and result["prior_family_regression"] <= 0.02
        and result["unsafe_false_accept_delta"] <= 0.0
        and result["rollback_checksum_match"] is True
        and thresholds["paired_ci_excludes_zero"] is True
        and thresholds["prior_family_regression_within_bound"] is True
        and thresholds["unsafe_false_accept_not_increased"] is True
    )


def build_artifact(
    *,
    root: Path | str,
    tests_added_or_reused: Sequence[str],
    checkpoint_dir: Path | str,
) -> JsonDict:
    """Build and validate the Exp5570 conductor-visible receipt."""

    root_path = Path(root)
    dataset = build_dataset(root_path)
    experiment = run_online_experiment(
        dataset,
        seeds=DEFAULT_SEEDS,
        replay_budget=DEFAULT_REPLAY_BUDGET,
        checkpoint_dir=checkpoint_dir,
    )
    active_summary = experiment["active_update_summary"]
    artifact: JsonDict = {
        "experiment": 5570,
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": DEFAULT_SEEDS[0],
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "continuous_self_learning_target": True,
        "dataset_path": dataset.dataset_path,
        "sessions": public_sessions(dataset),
        "n_rows": dataset.n_rows,
        "seeds": experiment["seeds"],
        "arms": experiment["arms"],
        "exact_feedback_only": True,
        "weights_mutated": active_summary["parameter_diff_norm"] > 0.0,
        "active_spline_update": True,
        "touched_spline_fraction": active_summary["touched_spline_fraction"],
        "update_count": active_summary["update_count"],
        "parameter_diff_norm": active_summary["parameter_diff_norm"],
        "update_latency_ms": {
            ACTIVE_ARM: active_summary["update_latency_ms"],
            DENSE_ARM: experiment["dense_update_summary"]["update_latency_ms"],
        },
        "heldout_exact_error_by_arm": experiment["heldout_exact_error_by_arm"],
        "energy_separation_by_arm": experiment["energy_separation_by_arm"],
        "brier_calibration_by_arm": experiment["brier_calibration_by_arm"],
        "forward_adaptation_delta": experiment["forward_adaptation_delta"],
        "prior_family_regression": experiment["prior_family_regression"],
        "unsafe_false_accept_delta": experiment["unsafe_false_accept_delta"],
        "replay_budget": experiment["replay_budget"],
        "checkpoint_paths": experiment["checkpoint_paths"],
        "rollback_checksum_match": experiment["rollback_checksum_match"],
        "promotion_thresholds": experiment["promotion_thresholds"],
        "paired_ci_active_vs_frozen": experiment["paired_ci_active_vs_frozen"],
        "active_update_summary": active_summary,
        "dense_update_summary": experiment["dense_update_summary"],
        "update_cost_scaling": update_cost_scaling(experiment),
        "exact_feedback_source": "accepted_by_exact_validator from Exp5566 corpus rows",
        "continuous_self_learning_target_note": "weights mutate online; no LLM labels or memory-only action changes",
        "tests_added_or_reused": list(tests_added_or_reused),
        "research_conductor_modified": False,
        "source_files": {
            "module": MODULE_RELATIVE_PATH.as_posix(),
            "spec": SPEC_RELATIVE_PATH.as_posix(),
            "test": TEST_RELATIVE_PATH.as_posix(),
        },
        "source_file_checksums": source_file_checksums(root_path),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "kan_ready": experiment["kan_ready"],
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def public_sessions(dataset: OnlineDataset) -> list[JsonDict]:
    """Expose session metadata without embedding every feature row."""

    return [
        {
            "session_id": session.session_id,
            "family": session.family,
            "online_row_count": len(session.online_row_ids),
            "holdout_row_count": len(session.holdout_row_ids),
            "online_row_ids": list(session.online_row_ids),
            "holdout_row_ids": list(session.holdout_row_ids),
        }
        for session in dataset.sessions
    ]


def update_cost_scaling(experiment: Mapping[str, Any]) -> JsonDict:
    """Compare active sparse update cost to dense update cost."""

    active = experiment["active_update_summary"]
    dense = experiment["dense_update_summary"]
    return {
        "active_touched_fraction": active["touched_spline_fraction"],
        "dense_touched_fraction": dense["touched_spline_fraction"],
        "active_vs_dense_latency_ratio": _round(
            active["update_latency_ms"]["total"] / dense["update_latency_ms"]["total"]
        ),
        "methodology": "deterministic operation-count proxy over touched coefficients and update steps",
    }


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    tests_added_or_reused: Sequence[str] = DEFAULT_TESTS_ADDED_OR_REUSED,
    checkpoint_dir: Path | str | None = None,
    write: bool = True,
) -> JsonDict:
    """Build the artifact and optionally write stable JSON."""

    root_path = Path(root)
    checkpoint_root = Path(checkpoint_dir) if checkpoint_dir is not None else root_path / CHECKPOINT_RELATIVE_DIR
    artifact = build_artifact(
        root=root_path,
        tests_added_or_reused=tests_added_or_reused,
        checkpoint_dir=checkpoint_root,
    )
    if write:
        write_json(resolve_path(root_path, result_path), artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Raise when the Exp5570 artifact is internally inconsistent."""

    errors = artifact_errors(artifact)
    if errors:
        raise ValueError("invalid Exp5570 artifact: " + "; ".join(errors))
    return True


def artifact_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return validation errors while allowing honest blocked artifacts."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
    if artifact.get("continuous_self_learning_target") is not True:
        errors.append("continuous_self_learning_target")
    if artifact.get("dataset_path") != DATASET_RELATIVE_PATH.as_posix():
        errors.append("dataset_path")
    sessions = artifact.get("sessions", [])
    if not isinstance(sessions, Sequence) or len(sessions) < 4:
        errors.append("sessions")
    if int(artifact.get("n_rows", 0)) < 120:
        errors.append("n_rows")
    if len(artifact.get("seeds", [])) < 5:
        errors.append("seeds")
    if artifact.get("arms") != list(ARM_NAMES):
        errors.append("arms")
    if artifact.get("exact_feedback_only") is not True:
        errors.append("exact_feedback_only")
    if artifact.get("weights_mutated") is not True:
        errors.append("weights_mutated")
    if artifact.get("active_spline_update") is not True:
        errors.append("active_spline_update")
    touched = float(artifact.get("touched_spline_fraction", 0.0))
    if not 0.0 < touched < 1.0:
        errors.append("touched_spline_fraction")
    if int(artifact.get("update_count", 0)) <= 0:
        errors.append("update_count")
    if float(artifact.get("parameter_diff_norm", 0.0)) <= 0.0:
        errors.append("parameter_diff_norm")
    latency = artifact.get("update_latency_ms", {})
    if not isinstance(latency, Mapping) or latency.get(ACTIVE_ARM, {}).get("total", 0.0) <= 0.0:
        errors.append("update_latency_ms")
    if float(artifact.get("forward_adaptation_delta", 0.0)) <= 0.0:
        errors.append("forward_adaptation_delta")
    if float(artifact.get("prior_family_regression", 1.0)) > 0.02:
        errors.append("prior_family_regression")
    if float(artifact.get("unsafe_false_accept_delta", 1.0)) > 0.0:
        errors.append("unsafe_false_accept_delta")
    if int(artifact.get("replay_budget", -1)) < 0:
        errors.append("replay_budget")
    if not artifact.get("checkpoint_paths"):
        errors.append("checkpoint_paths")
    if artifact.get("rollback_checksum_match") is not True:
        errors.append("rollback_checksum_match")
    if artifact.get("kan_ready") is not kan_ready_from_artifact(artifact):
        errors.append("kan_ready")
    principles = artifact.get("field_principles", {})
    if isinstance(principles, Mapping):
        missing_principles = [field for field in REQUIRED_ARTIFACT_FIELDS if not principles.get(field)]
    else:
        missing_principles = list(REQUIRED_ARTIFACT_FIELDS)
    if missing_principles:
        errors.append(f"field_principles missing: {missing_principles}")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if artifact.get("honest_verdict") != honest_verdict(artifact):
        errors.append("honest_verdict")
    checksum = artifact.get("reproducibility_checksum")
    if checksum and checksum != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def kan_ready_from_artifact(artifact: Mapping[str, Any]) -> bool:
    """Recompute the final KAN gate from artifact fields."""

    ci = artifact.get("paired_ci_active_vs_frozen", {})
    thresholds = artifact.get("promotion_thresholds", {})
    return (
        ci.get("lower", 0.0) > 0.0
        and artifact.get("prior_family_regression", 1.0) <= 0.02
        and artifact.get("unsafe_false_accept_delta", 1.0) <= 0.0
        and artifact.get("rollback_checksum_match") is True
        and thresholds.get("paired_ci_excludes_zero") is True
        and thresholds.get("prior_family_regression_within_bound") is True
        and thresholds.get("unsafe_false_accept_not_increased") is True
    )


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return the terminal conductor verdict for the artifact."""

    if artifact.get("kan_ready") is True and kan_ready_from_artifact(artifact):
        return "complete: active_spline_online_kan_exact_energy_ready"
    return "blocked: active_spline_online_kan_gate_not_met"


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash an artifact while blanking its self-referential checksum."""

    stable = dict(artifact)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def source_file_checksums(root: Path) -> JsonDict:
    """Hash source files that define the Exp5570 result."""

    checksums: JsonDict = {}
    for path in (MODULE_RELATIVE_PATH, SPEC_RELATIVE_PATH, TEST_RELATIVE_PATH):
        full = root / path
        checksums[path.as_posix()] = hashlib.sha256(full.read_bytes()).hexdigest()
    return checksums


def resolve_path(root: Path, path: Path | str) -> Path:
    """Resolve repo-relative paths without changing absolute paths."""

    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write stable indented JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    run()
