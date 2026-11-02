"""Seal a causal prediction-time ledger for four-expert mixture replay.

The ledger stores the numeric operands that Exp7440 did not retain. Its fixture
is an exact arithmetic oracle, so readiness proves mechanics only. It does not
establish online benefit or authorize a production policy change.

Spec refs: REQ-AUTO-7450 and SCENARIO-AUTO-7450-01 through -06.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)


JsonDict = dict[str, Any]
RUN_DATE = "20260920"
MILESTONE = "2026.09.653"
EXPERIMENT_ID = "exp7450-v653-prediction-ledger"
SCHEMA = "carnot.exp7450.v653.prediction_ledger.v1"
PREDICTION_SCHEMA = "carnot.exp7450.prediction_event.v1"
FEEDBACK_SCHEMA = "carnot.exp7450.feedback_event.v1"
STATE_SCHEMA = "carnot.exp7450.initial_numeric_state.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7450_v653_prediction_ledger.json")
RAW_DIR = Path("results/raw/experiment_7450_v653_prediction_ledger")
MODULE_PATH = Path("python/carnot/experiment_7450_v653_prediction_ledger.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7450_v653_prediction_ledger.py")
TEST_PATH = Path("tests/python/test_experiment_7450_v653_prediction_ledger.py")
SPEC_PATH = Path("openspec/capabilities/autoresearch/spec.md")
UPSTREAM_PATH = Path("results/experiment_7440_v652_mixture_learning.json")
AUDIT_PATH = Path("results/experiment_7441_v652_decision_audit.json")
PROTOTYPE_PROTOCOL_PATH = Path("results/raw/experiment_7438_v652_mixture_prototype/protocol.json")

EXPERT_NAMES = (
    "frozen_spline",
    "adaptive_spline",
    "frozen_gibbs",
    "adaptive_gibbs",
)
LEARNED_ARM = "learned_mixture"
NO_FEEDBACK_ARM = "no_feedback_frozen_prior_mixture"
ETA = 1.0
FIXED_SHARE = 0.01
PROBABILITY_CLIP = 1e-6
TRAINING_SEEDS = (65_201, 65_202, 65_203, 65_204, 65_205)
ORDERS = ("hash_order", "domain_blocked_shift_order")
DELAYS = (0, 8)
ARMS = (
    LEARNED_ARM,
    "equal_weight_adaptive_mixture",
    "frozen_spline",
    "adaptive_spline",
    NO_FEEDBACK_ARM,
    "shuffled_labels_permutation_1",
    "shuffled_labels_permutation_2",
)
PRIMARY_COMPARATORS = (
    "frozen_spline",
    "adaptive_spline",
    "equal_weight_adaptive_mixture",
)
SERVICE_STAGES = ("read", "predict", "persist", "reveal", "update")
INFERENCE_SUBSTRATE = "prediction_ledger_numeric_replay"
INFERENCE_SUBSTRATE_CLASS = "no_model_load"
EXECUTION_VENUE = "host"

INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/experiment_7427_v651_randomized_feedback.py"),
    Path("python/carnot/experiment_7438_v652_mixture_prototype.py"),
    Path("python/carnot/experiment_7440_v652_mixture_learning.py"),
    Path("python/carnot/experiment_7441_v652_decision_audit.py"),
    SPEC_PATH,
    UPSTREAM_PATH,
    AUDIT_PATH,
    PROTOTYPE_PROTOCOL_PATH,
)

VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)
AFFECTED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "cold_artifact_replay",
    "independent_raw_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
REQUIRED_MUTATIONS = (
    "changed_expert_prediction",
    "deleted_checkpoint",
    "future_label_access",
    "duplicated_feedback",
    "reordered_updates",
)


def utc_now() -> str:  # pragma: no cover - real execution boundary.
    """Return an aware UTC timestamp for the current run boundary."""

    return datetime.now(UTC).isoformat()


def progress(  # pragma: no cover - public progress boundary.
    started: float, phase: str, event: str, **details: Any
) -> None:
    """Emit a flushed line before and after each potentially long operation."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7450] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _load_object(path: Path) -> JsonDict:
    """Read one JSON object and return an empty object for absent or invalid bytes."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _precondition(
    check: str,
    upstream: str,
    path: str,
    field: str,
    expected: Any,
    observed: Any,
    *,
    operator: str = "==",
    passed: bool | None = None,
) -> JsonDict:
    """Keep missing, null, zero, false, and changed prerequisites distinct."""

    return {
        "check": check,
        "upstream": upstream,
        "path": path,
        "field": field,
        "operator": operator,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected if passed is None else bool(passed),
    }


def collect_preconditions(root: Path) -> tuple[list[JsonDict], JsonDict, JsonDict]:
    """Authenticate source bytes plus the original producer flags and defects."""

    checks: list[JsonDict] = []
    hashes: JsonDict = {}
    for relative in INPUT_PATHS:
        path = root / relative
        observed = "readable_nonempty_bytes" if path.is_file() and path.stat().st_size else None
        checks.append(
            _precondition(
                f"source_bytes:{relative.as_posix()}",
                relative.as_posix(),
                relative.as_posix(),
                "bytes",
                "readable_nonempty_bytes",
                observed,
            )
        )
        if observed is not None:
            hashes[relative.as_posix()] = {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "original_verdict_class": None,
                "original_flagged_adversarial": None,
            }

    upstream = _load_object(root / UPSTREAM_PATH)
    audit = _load_object(root / AUDIT_PATH)
    expected_upstream = {
        "schema": "carnot.exp7440.v652.mixture_learning.v1",
        "experiment_id": "exp7440-v652-mixture-learning",
        "milestone": "2026.09.652",
        "online_capture_complete_score": 1,
        "online_value_score": 0,
        "verdict_class": "null",
        "flagged_adversarial": False,
    }
    for field, expected in expected_upstream.items():
        checks.append(
            _precondition(
                f"exp7440:{field}",
                "exp7440-v652-mixture-learning",
                UPSTREAM_PATH.as_posix(),
                field,
                expected,
                upstream.get(field),
            )
        )
    defects = (audit.get("gate_check_summary") or {}).get("evidence_defects")
    expected_defects = [
        "online:missing_expert_predictions",
        "online:weight_update_replay_incomplete",
    ]
    checks.append(
        _precondition(
            "exp7441:original_online_defects",
            "exp7441-v652-decision-audit",
            AUDIT_PATH.as_posix(),
            "gate_check_summary.evidence_defects",
            expected_defects,
            defects,
        )
    )
    checks.append(
        _precondition(
            "exp7441:flagged_adversarial",
            "exp7441-v652-decision-audit",
            AUDIT_PATH.as_posix(),
            "flagged_adversarial",
            False,
            audit.get("flagged_adversarial"),
        )
    )
    for relative, value in ((UPSTREAM_PATH, upstream), (AUDIT_PATH, audit)):
        if relative.as_posix() in hashes:
            hashes[relative.as_posix()].update(
                {
                    "original_honest_verdict": value.get("honest_verdict"),
                    "original_verdict_class": value.get("verdict_class"),
                    "original_flagged_adversarial": value.get("flagged_adversarial"),
                    "source_receipt_class": "historical_numeric_replay",
                }
            )
    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    checks.append(
        _precondition(
            "driving_requirement",
            SPEC_PATH.as_posix(),
            SPEC_PATH.as_posix(),
            "REQ-*",
            "REQ-AUTO-7450",
            "REQ-AUTO-7450" if "REQ-AUTO-7450" in spec_text else None,
        )
    )
    exclusion = (
        (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
        if (root / "ops/exclusion_manifest.yaml").is_file()
        else ""
    )
    checks.append(
        _precondition(
            "current_task_not_quarantined",
            "ops/exclusion_manifest.yaml",
            "ops/exclusion_manifest.yaml",
            EXPERIMENT_ID,
            False,
            "experiment_id: 7450" in exclusion,
        )
    )
    return checks, hashes, upstream


def build_replay_protocol() -> JsonDict:
    """Freeze the Exp7440 comparison protocol without running its full stream."""

    protocol: JsonDict = {
        "schema": "carnot.exp7450.exp7440_replay_protocol.v1",
        "source_experiment_id": "exp7440-v652-mixture-learning",
        "orders": list(ORDERS),
        "delays": list(DELAYS),
        "training_seeds": list(TRAINING_SEEDS),
        "arms": list(ARMS),
        "primary_comparators": list(PRIMARY_COMPARATORS),
        "reveal_schedule": {
            "full_block": "uniform_eight_of_32",
            "target_fraction": 0.25,
            "partial_final_block": "floor(n/4)",
        },
        "moving_block_lengths": [32, 64],
        "bootstrap_draws": 10_000,
        "bootstrap_seed": 6_520_440,
        "labels": "authenticated_archived_human_binary_labels_unchanged",
        "success_bars": {
            "simultaneous_log_loss_upper_delta": "<0_against_each_primary_in_every_cell",
            "brier_noninferiority_margin": 0.001,
            "harmful_action_rate": "no_increase_where_defined",
            "coverage": "no_loss",
            "label_cost": "no_increase",
            "negative_controls": "no_feedback_equality_and_two_shuffled_labels",
        },
        "full_scientific_stream_executed": False,
        "claim_boundary": "ledger_mechanics_only",
    }
    protocol["protocol_hash"] = canonical_hash(protocol)
    return protocol


def _clip_probability(value: float) -> float:
    """Clip one finite probability under the inherited Exp7438 rule."""

    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("probability must be finite")
    return min(max(numeric, PROBABILITY_CLIP), 1.0 - PROBABILITY_CLIP)


def _loss(label: int, probability: float) -> float:
    """Compute binary log loss from one saved prediction-time probability."""

    if label not in {0, 1} or isinstance(label, bool):
        raise ValueError("label must be binary")
    probability = _clip_probability(probability)
    return -(label * math.log(probability) + (1 - label) * math.log1p(-probability))


def _state_hash(arm: str, log_weights: Mapping[str, float], feedback_count: int) -> str:
    """Bind the numeric mixture state without any mutable expert objects."""

    return canonical_hash(
        {
            "arm": arm,
            "expert_order": list(EXPERT_NAMES),
            "log_weights": {name: float(log_weights[name]) for name in EXPERT_NAMES},
            "feedback_count": int(feedback_count),
        }
    )


def prediction_event_hash(value: Mapping[str, Any]) -> str:
    """Hash every immutable prediction field except the hash slot itself."""

    return canonical_hash(
        {key: deepcopy(item) for key, item in value.items() if key != "event_hash"}
    )


def feedback_event_hash(value: Mapping[str, Any]) -> str:
    """Hash every feedback and update operand except the hash slot itself."""

    return canonical_hash(
        {key: deepcopy(item) for key, item in value.items() if key != "event_hash"}
    )


@dataclass(frozen=True)
class PredictionEvent:
    """Hold one immutable label-free prediction event in fixed expert order."""

    event_id: str
    group_id: str
    arm: str
    order: str
    delay: int
    seed: int
    request_order: int
    expert_probabilities: tuple[tuple[str, float], ...]
    mixture_weights: tuple[tuple[str, float], ...]
    mixture_prediction: float
    expert_checkpoint_hashes: tuple[tuple[str, str], ...]
    label_propensity: float
    pre_feedback_state_hash: str
    event_hash: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PredictionEvent:
        """Build the frozen value only after its prediction hash verifies."""

        if prediction_event_hash(value) != value.get("event_hash"):
            raise ValueError("prediction event hash mismatch")
        return cls(
            event_id=str(value["event_id"]),
            group_id=str(value["group_id"]),
            arm=str(value["arm"]),
            order=str(value["order"]),
            delay=int(value["delay"]),
            seed=int(value["seed"]),
            request_order=int(value["request_order"]),
            expert_probabilities=tuple(
                (name, float(value["expert_probabilities"][name])) for name in EXPERT_NAMES
            ),
            mixture_weights=tuple(
                (name, float(value["mixture_weights"][name])) for name in EXPERT_NAMES
            ),
            mixture_prediction=float(value["mixture_prediction"]),
            expert_checkpoint_hashes=tuple(
                (name, str(value["expert_checkpoint_hashes"][name])) for name in EXPERT_NAMES
            ),
            label_propensity=float(value["label_propensity"]),
            pre_feedback_state_hash=str(value["pre_feedback_state_hash"]),
            event_hash=str(value["event_hash"]),
        )

    def to_dict(self) -> JsonDict:
        """Return the canonical persisted mapping without adding a label field."""

        return {
            "schema": PREDICTION_SCHEMA,
            "row_type": "prediction_event",
            "event_id": self.event_id,
            "group_id": self.group_id,
            "arm": self.arm,
            "order": self.order,
            "delay": self.delay,
            "seed": self.seed,
            "request_order": self.request_order,
            "expert_probabilities": dict(self.expert_probabilities),
            "mixture_weights": dict(self.mixture_weights),
            "mixture_prediction": self.mixture_prediction,
            "expert_checkpoint_hashes": dict(self.expert_checkpoint_hashes),
            "label_propensity": self.label_propensity,
            "pre_feedback_state_hash": self.pre_feedback_state_hash,
            "event_hash": self.event_hash,
        }


def _write_jsonl_event(path: Path, value: Mapping[str, Any]) -> None:
    """Append and fsync one event so a label cannot overtake its prediction."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl(path: Path) -> list[JsonDict]:
    """Read one durable event stream and require each row to be a JSON object."""

    rows: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("ledger row must be an object")
        rows.append(value)
    return rows


def _checkpoint_sidecars(root: Path, relative_dir: Path) -> list[JsonDict]:
    """Write four small analytic checkpoint identities, not fitted model state."""

    output: list[JsonDict] = []
    for index, name in enumerate(EXPERT_NAMES):
        relative = relative_dir / "checkpoints" / f"{name}.json"
        path = root / relative
        atomic_json(
            path,
            {
                "schema": "carnot.exp7450.analytic_checkpoint.v1",
                "expert": name,
                "seed": TRAINING_SEEDS[0],
                "fixture_coefficient": (index + 1) / 10.0,
                "source_receipt_class": "analytic_fixture_not_training",
            },
        )
        output.append(
            {
                "expert": name,
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "source_receipt_class": "analytic_fixture_not_training",
            }
        )
    return output


def _initial_state(checkpoints: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Create enough code-free state for a cold reader to start from bytes."""

    logs = {name: math.log(0.25) for name in EXPERT_NAMES}
    arms = {
        arm: {
            "initial_log_weights": deepcopy(logs),
            "initial_state_hash": _state_hash(arm, logs, 0),
        }
        for arm in (LEARNED_ARM, NO_FEEDBACK_ARM)
    }
    manifest: JsonDict = {
        "schema": STATE_SCHEMA,
        "expert_order": list(EXPERT_NAMES),
        "eta": ETA,
        "fixed_share": FIXED_SHARE,
        "arms": arms,
        "checkpoints": [deepcopy(dict(row)) for row in checkpoints],
    }
    manifest["initial_state_hash"] = canonical_hash(manifest)
    return manifest


class LedgerCollector:
    """Persist predictions before accepting labels and retain numeric state only."""

    def __init__(self, path: Path, initial_state: Mapping[str, Any]) -> None:
        self.path = path
        self.initial_state = deepcopy(dict(initial_state))
        self.log_weights = {
            arm: {
                name: float(initial_state["arms"][arm]["initial_log_weights"][name])
                for name in EXPERT_NAMES
            }
            for arm in (LEARNED_ARM, NO_FEEDBACK_ARM)
        }
        self.feedback_counts = {LEARNED_ARM: 0, NO_FEEDBACK_ARM: 0}
        self.predictions: dict[str, JsonDict] = {}
        self.feedback_hashes: set[str] = set()
        self.events: list[JsonDict] = []
        self.stage_duration_ns = {name: 0 for name in SERVICE_STAGES}

    def persist_prediction(
        self,
        *,
        event_id: str,
        group_id: str,
        arm: str,
        order: str,
        delay: int,
        seed: int,
        request_order: int,
        expert_probabilities: Mapping[str, float],
        label_propensity: float,
    ) -> JsonDict:
        """Compute and durably append one label-free prediction event."""

        if event_id in self.predictions or arm not in self.log_weights:
            raise ValueError("prediction identity and arm must be new and registered")
        if order not in ORDERS or delay not in DELAYS or seed not in TRAINING_SEEDS:
            raise ValueError("prediction must use the frozen replay protocol")
        if set(expert_probabilities) != set(EXPERT_NAMES):
            raise ValueError("four named expert probabilities are required")
        if not 0.0 < float(label_propensity) <= 1.0:
            raise ValueError("label propensity must be positive")
        predict_started = time.perf_counter_ns()
        probabilities = {
            name: _clip_probability(expert_probabilities[name]) for name in EXPERT_NAMES
        }
        weights = {name: math.exp(self.log_weights[arm][name]) for name in EXPERT_NAMES}
        weight_sum = math.fsum(weights.values())
        weights = {name: weights[name] / weight_sum for name in EXPERT_NAMES}
        mixture = math.fsum(weights[name] * probabilities[name] for name in EXPERT_NAMES)
        checkpoint_hashes = {
            str(row["expert"]): str(row["sha256"]) for row in self.initial_state["checkpoints"]
        }
        value: JsonDict = {
            "schema": PREDICTION_SCHEMA,
            "row_type": "prediction_event",
            "event_id": event_id,
            "group_id": group_id,
            "arm": arm,
            "order": order,
            "delay": int(delay),
            "seed": int(seed),
            "request_order": int(request_order),
            "expert_probabilities": probabilities,
            "mixture_weights": weights,
            "mixture_prediction": mixture,
            "expert_checkpoint_hashes": checkpoint_hashes,
            "label_propensity": float(label_propensity),
            "pre_feedback_state_hash": _state_hash(
                arm, self.log_weights[arm], self.feedback_counts[arm]
            ),
        }
        value["event_hash"] = prediction_event_hash(value)
        self.stage_duration_ns["predict"] += time.perf_counter_ns() - predict_started
        persist_started = time.perf_counter_ns()
        _write_jsonl_event(self.path, value)
        self.stage_duration_ns["persist"] += time.perf_counter_ns() - persist_started
        self.predictions[event_id] = deepcopy(value)
        self.events.append(deepcopy(value))
        return deepcopy(value)

    def reveal(
        self,
        prediction_event_hash_value: str,
        *,
        label: int,
        reveal_order: int,
        label_origin: str,
    ) -> JsonDict:
        """Reference one durable prediction and record its exact numeric update."""

        if label not in {0, 1} or isinstance(label, bool):
            raise ValueError("label must be binary")
        matching = [
            row
            for row in self.predictions.values()
            if row["event_hash"] == prediction_event_hash_value
        ]
        if len(matching) != 1:
            raise ValueError("feedback must reference one persisted prediction")
        prediction = matching[0]
        if prediction_event_hash_value in self.feedback_hashes:
            raise ValueError("feedback already committed")
        if reveal_order < int(prediction["request_order"]) + int(prediction["delay"]):
            raise ValueError("feedback is not visible yet")
        reveal_started = time.perf_counter_ns()
        probabilities = prediction["expert_probabilities"]
        losses = {name: _loss(label, probabilities[name]) for name in EXPERT_NAMES}
        self.stage_duration_ns["reveal"] += time.perf_counter_ns() - reveal_started
        update_started = time.perf_counter_ns()
        arm = str(prediction["arm"])
        old = deepcopy(self.log_weights[arm])
        penalized = {name: old[name] - ETA * losses[name] for name in EXPERT_NAMES}
        maximum = max(penalized.values())
        exponentials = {name: math.exp(penalized[name] - maximum) for name in EXPERT_NAMES}
        sum_exp = math.fsum(exponentials.values())
        posterior = {name: exponentials[name] / sum_exp for name in EXPERT_NAMES}
        shared = {
            name: (1.0 - FIXED_SHARE) * posterior[name] + FIXED_SHARE / len(EXPERT_NAMES)
            for name in EXPERT_NAMES
        }
        new = {name: math.log(shared[name]) for name in EXPERT_NAMES}
        feedback_sequence = self.feedback_counts[arm]
        child_hash = _state_hash(arm, new, feedback_sequence + 1)
        value: JsonDict = {
            "schema": FEEDBACK_SCHEMA,
            "row_type": "feedback_event",
            "feedback_id": f"feedback-{prediction['event_id']}",
            "feedback_sequence": feedback_sequence,
            "event_id": prediction["event_id"],
            "group_id": prediction["group_id"],
            "arm": arm,
            "order": prediction["order"],
            "delay": prediction["delay"],
            "seed": prediction["seed"],
            "prediction_event_hash": prediction_event_hash_value,
            "reveal_order": int(reveal_order),
            "label_origin": label_origin,
            "label": int(label),
            "per_expert_loss": losses,
            "old_log_weights": old,
            "numeric_update": {
                "eta": ETA,
                "penalized_log_weights": penalized,
                "posterior_before_share": posterior,
                "fixed_share": FIXED_SHARE,
                "weights_after_share": shared,
            },
            "normalizer": {
                "maximum": maximum,
                "sum_exp": sum_exp,
                "log_normalizer": maximum + math.log(sum_exp),
            },
            "new_log_weights": new,
            "parent_state_hash": prediction["pre_feedback_state_hash"],
            "child_state_hash": child_hash,
        }
        value["event_hash"] = feedback_event_hash(value)
        self.stage_duration_ns["update"] += time.perf_counter_ns() - update_started
        persist_started = time.perf_counter_ns()
        _write_jsonl_event(self.path, value)
        self.stage_duration_ns["persist"] += time.perf_counter_ns() - persist_started
        self.log_weights[arm] = new
        self.feedback_counts[arm] += 1
        self.feedback_hashes.add(prediction_event_hash_value)
        self.events.append(deepcopy(value))
        return deepcopy(value)

    def state_hash(self, arm: str) -> str:
        """Return the current numeric state hash for one registered arm."""

        return _state_hash(arm, self.log_weights[arm], self.feedback_counts[arm])


def _mapping_close(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Compare one complete expert mapping with strict keys and tight floats."""

    return set(left) == set(right) == set(EXPERT_NAMES) and all(
        math.isclose(float(left[name]), float(right[name]), rel_tol=0.0, abs_tol=1e-12)
        for name in EXPERT_NAMES
    )


def cold_replay(value: Mapping[str, Any], *, root: Path) -> JsonDict:
    """Replay raw events with independent scalar math and no producer update call."""

    errors: list[str] = []
    initial = value.get("initial_state_manifest") or {}
    events = value.get("events") or []
    predictions_projection = value.get("prediction_events") or []
    feedback_projection = value.get("feedback_events") or []
    if [row for row in events if row.get("row_type") == "prediction_event"] != list(
        predictions_projection
    ) or [row for row in events if row.get("row_type") == "feedback_event"] != list(
        feedback_projection
    ):
        errors.append("event_projection_mismatch")
    checkpoints = initial.get("checkpoints") or []
    checkpoint_hashes: dict[str, str] = {}
    for row in checkpoints:
        path = root / str(row.get("path") or "")
        if not path.is_file():
            errors.append("checkpoint_missing")
            continue
        observed = sha256_file(path)
        if observed != row.get("sha256"):
            errors.append("checkpoint_hash_mismatch")
        checkpoint_hashes[str(row.get("expert"))] = observed
    arms = initial.get("arms") or {}
    log_weights: dict[str, dict[str, float]] = {}
    feedback_counts: dict[str, int] = {}
    for arm in (LEARNED_ARM, NO_FEEDBACK_ARM):
        arm_state = arms.get(arm) or {}
        try:
            logs = {name: float(arm_state["initial_log_weights"][name]) for name in EXPERT_NAMES}
        except (KeyError, TypeError, ValueError):
            errors.append("initial_state_invalid")
            logs = {name: math.log(0.25) for name in EXPERT_NAMES}
        log_weights[arm] = logs
        feedback_counts[arm] = 0
        if arm_state.get("initial_state_hash") != _state_hash(arm, logs, 0):
            errors.append("initial_state_hash_mismatch")
    predictions: dict[str, Mapping[str, Any]] = {}
    committed: set[str] = set()
    pending_hashes: list[str] = []
    last_feedback_sequence = -1
    last_reveal_order = -1
    updates_replayed = 0
    maximum_numeric_gap = 0.0
    for row in events:
        row_type = row.get("row_type")
        if row_type == "prediction_event":
            if prediction_event_hash(row) != row.get("event_hash"):
                errors.append("prediction_event_hash_mismatch")
            event_id = str(row.get("event_id"))
            if event_id in predictions:
                errors.append("duplicate_prediction")
            predictions[event_id] = row
            pending_hashes.append(str(row.get("event_hash")))
            arm = str(row.get("arm"))
            if arm not in log_weights:
                errors.append("prediction_arm_invalid")
                continue
            if row.get("pre_feedback_state_hash") != _state_hash(
                arm, log_weights[arm], feedback_counts[arm]
            ):
                errors.append("prediction_pre_state_mismatch")
            if set(row.get("expert_probabilities") or {}) != set(EXPERT_NAMES):
                errors.append("expert_probabilities_incomplete")
            if row.get("expert_checkpoint_hashes") != checkpoint_hashes:
                errors.append("prediction_checkpoint_hash_mismatch")
            if any(key in row for key in ("label", "loss", "per_expert_loss")):
                errors.append("prediction_contains_label")
            try:
                expected_mixture = math.fsum(
                    float(row["mixture_weights"][name])
                    * _clip_probability(float(row["expert_probabilities"][name]))
                    for name in EXPERT_NAMES
                )
                if not math.isclose(
                    expected_mixture,
                    float(row.get("mixture_prediction")),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    errors.append("mixture_prediction_mismatch")
            except (KeyError, TypeError, ValueError):
                errors.append("prediction_numeric_invalid")
        elif row_type == "feedback_event":
            if feedback_event_hash(row) != row.get("event_hash"):
                errors.append("feedback_event_hash_mismatch")
            sequence = int(row.get("feedback_sequence", -1))
            reveal_order = int(row.get("reveal_order", -1))
            if sequence != last_feedback_sequence + 1 or reveal_order < last_reveal_order:
                errors.append("feedback_order_invalid")
            last_feedback_sequence = sequence
            last_reveal_order = reveal_order
            prediction_hash = str(row.get("prediction_event_hash"))
            prediction = next(
                (
                    item
                    for item in predictions.values()
                    if item.get("event_hash") == prediction_hash
                ),
                None,
            )
            if prediction is None:
                errors.append("feedback_prediction_missing")
                continue
            if prediction_hash in committed:
                errors.append("duplicate_feedback")
                continue
            if reveal_order < int(prediction["request_order"]) + int(prediction["delay"]):
                errors.append("feedback_before_reveal")
            arm = str(prediction["arm"])
            label = row.get("label")
            try:
                losses = {
                    name: _loss(int(label), float(prediction["expert_probabilities"][name]))
                    for name in EXPERT_NAMES
                }
                old = log_weights[arm]
                penalized = {name: old[name] - ETA * losses[name] for name in EXPERT_NAMES}
                maximum = max(penalized.values())
                exponentials = {name: math.exp(penalized[name] - maximum) for name in EXPERT_NAMES}
                denominator = math.fsum(exponentials.values())
                posterior = {name: exponentials[name] / denominator for name in EXPERT_NAMES}
                shared = {
                    name: (1.0 - FIXED_SHARE) * posterior[name] + FIXED_SHARE / len(EXPERT_NAMES)
                    for name in EXPERT_NAMES
                }
                new = {name: math.log(shared[name]) for name in EXPERT_NAMES}
            except (KeyError, TypeError, ValueError):
                errors.append("feedback_numeric_invalid")
                continue
            if not _mapping_close(row.get("per_expert_loss") or {}, losses):
                errors.append("expert_loss_mismatch")
            if not _mapping_close(row.get("old_log_weights") or {}, old):
                errors.append("old_log_weight_mismatch")
            update = row.get("numeric_update") or {}
            if not _mapping_close(update.get("penalized_log_weights") or {}, penalized):
                errors.append("numeric_update_mismatch")
            if not _mapping_close(update.get("posterior_before_share") or {}, posterior):
                errors.append("numeric_update_mismatch")
            if not _mapping_close(update.get("weights_after_share") or {}, shared):
                errors.append("numeric_update_mismatch")
            normalizer = row.get("normalizer") or {}
            if not math.isclose(
                float(normalizer.get("sum_exp", math.nan)),
                denominator,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                errors.append("normalizer_mismatch")
            if not _mapping_close(row.get("new_log_weights") or {}, new):
                errors.append("new_log_weight_mismatch")
            maximum_numeric_gap = max(
                maximum_numeric_gap,
                max(
                    abs(float((row.get("new_log_weights") or {}).get(name, math.inf)) - new[name])
                    for name in EXPERT_NAMES
                ),
            )
            parent = _state_hash(arm, old, feedback_counts[arm])
            child = _state_hash(arm, new, feedback_counts[arm] + 1)
            if row.get("parent_state_hash") != parent:
                errors.append("feedback_parent_state_mismatch")
            if row.get("child_state_hash") != child:
                errors.append("feedback_child_state_mismatch")
            log_weights[arm] = new
            feedback_counts[arm] += 1
            committed.add(prediction_hash)
            if prediction_hash in pending_hashes:
                pending_hashes.remove(prediction_hash)
            updates_replayed += 1
        else:
            errors.append("event_type_invalid")
    final_states = {
        arm: _state_hash(arm, log_weights[arm], feedback_counts[arm])
        for arm in (LEARNED_ARM, NO_FEEDBACK_ARM)
    }
    return {
        "valid": not errors,
        "errors": list(dict.fromkeys(errors)),
        "prediction_count": len(predictions),
        "feedback_count": len(committed),
        "updates_replayed": updates_replayed,
        "maximum_numeric_gap": maximum_numeric_gap,
        "pending_prediction_hashes": pending_hashes,
        "final_state_hashes": final_states,
    }


def _ledger_value(
    *,
    initial: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    ledger_path: Path,
    root: Path,
) -> JsonDict:
    """Build one projection-rich ledger whose raw event stream remains authoritative."""

    return {
        "schema": "carnot.exp7450.prediction_ledger.v1",
        "initial_state_manifest": deepcopy(dict(initial)),
        "events": deepcopy(list(events)),
        "prediction_events": [
            deepcopy(dict(row)) for row in events if row.get("row_type") == "prediction_event"
        ],
        "feedback_events": [
            deepcopy(dict(row)) for row in events if row.get("row_type") == "feedback_event"
        ],
        "ledger_path": ledger_path.relative_to(root).as_posix(),
        "ledger_sha256": sha256_file(ledger_path),
    }


def build_fixture_ledger(root: Path, *, relative_dir: Path) -> JsonDict:
    """Collect a compact analytic trajectory, one crash point, and one fixed arm."""

    checkpoints = _checkpoint_sidecars(root, relative_dir)
    initial = _initial_state(checkpoints)
    ledger_path = root / relative_dir / "events.jsonl"
    ledger_path.unlink(missing_ok=True)
    collector = LedgerCollector(ledger_path, initial)
    first = collector.persist_prediction(
        event_id="prediction-000",
        group_id="group-000",
        arm=LEARNED_ARM,
        order="hash_order",
        delay=0,
        seed=TRAINING_SEEDS[0],
        request_order=0,
        expert_probabilities={
            "frozen_spline": 0.90,
            "adaptive_spline": 0.80,
            "frozen_gibbs": 0.65,
            "adaptive_gibbs": 0.55,
        },
        label_propensity=0.25,
    )
    collector.reveal(first["event_hash"], label=1, reveal_order=0, label_origin="analytic_fixture")
    second = collector.persist_prediction(
        event_id="prediction-001",
        group_id="group-001",
        arm=LEARNED_ARM,
        order="hash_order",
        delay=8,
        seed=TRAINING_SEEDS[0],
        request_order=1,
        expert_probabilities={
            "frozen_spline": 0.25,
            "adaptive_spline": 0.35,
            "frozen_gibbs": 0.45,
            "adaptive_gibbs": 0.60,
        },
        label_propensity=0.25,
    )
    prefix_events = deepcopy(collector.events)
    collector.reveal(second["event_hash"], label=0, reveal_order=9, label_origin="analytic_fixture")
    no_feedback_initial = collector.state_hash(NO_FEEDBACK_ARM)
    collector.persist_prediction(
        event_id="prediction-no-feedback",
        group_id="group-control",
        arm=NO_FEEDBACK_ARM,
        order="hash_order",
        delay=0,
        seed=TRAINING_SEEDS[0],
        request_order=2,
        expert_probabilities={name: 0.5 for name in EXPERT_NAMES},
        label_propensity=0.25,
    )
    ledger = _ledger_value(
        initial=initial, events=collector.events, ledger_path=ledger_path, root=root
    )
    prefix = {
        **deepcopy(ledger),
        "events": prefix_events,
        "prediction_events": [
            deepcopy(row) for row in prefix_events if row.get("row_type") == "prediction_event"
        ],
        "feedback_events": [
            deepcopy(row) for row in prefix_events if row.get("row_type") == "feedback_event"
        ],
    }
    prefix_replay = cold_replay(prefix, root=root)
    full_replay = cold_replay(ledger, root=root)
    ledger.update(
        {
            "persisted_before_reveal": read_jsonl(ledger_path)[0]["row_type"] == "prediction_event",
            "crash_recovery": {
                "prediction_persisted_before_crash": second["event_hash"]
                in prefix_replay["pending_prediction_hashes"],
                "prediction_hash_preserved": second["event_hash"]
                == ledger["prediction_events"][1]["event_hash"],
                "recovered_final_state_hash": full_replay["final_state_hashes"][LEARNED_ARM],
                "uninterrupted_final_state_hash": collector.state_hash(LEARNED_ARM),
                "passed": full_replay["final_state_hashes"][LEARNED_ARM]
                == collector.state_hash(LEARNED_ARM),
            },
            "no_feedback_control": {
                "prediction_count": 1,
                "feedback_count": 0,
                "initial_state_hash": no_feedback_initial,
                "final_state_hash": collector.state_hash(NO_FEEDBACK_ARM),
                "passed": no_feedback_initial == collector.state_hash(NO_FEEDBACK_ARM),
            },
            "service_stage_receipts": [
                {
                    "stage": name,
                    "duration_ns": collector.stage_duration_ns[name],
                    "qualified": True,
                    "scope": "analytic_ledger_prototype",
                }
                for name in SERVICE_STAGES
            ],
        }
    )
    return ledger


def mutate_fixture(value: Mapping[str, Any], mutation: str) -> JsonDict:
    """Plant one registered causal defect without altering unrelated operands."""

    changed = deepcopy(dict(value))
    events = changed["events"]
    if mutation == "changed_expert_prediction":
        prediction = next(row for row in events if row["row_type"] == "prediction_event")
        prediction["expert_probabilities"][EXPERT_NAMES[0]] = 0.01
    elif mutation == "future_label_access":
        feedback = next(row for row in events if row["row_type"] == "feedback_event")
        prediction = next(
            row for row in events if row.get("event_hash") == feedback["prediction_event_hash"]
        )
        feedback["reveal_order"] = prediction["request_order"] + prediction["delay"] - 1
    elif mutation == "duplicated_feedback":
        events.append(deepcopy(next(row for row in events if row["row_type"] == "feedback_event")))
    elif mutation == "reordered_updates":
        positions = [
            index for index, row in enumerate(events) if row["row_type"] == "feedback_event"
        ]
        events[positions[0]], events[positions[1]] = events[positions[1]], events[positions[0]]
    elif mutation == "deleted_checkpoint":
        changed["initial_state_manifest"]["checkpoints"][0]["path"] = "missing.json"
    else:
        raise ValueError(f"unknown mutation:{mutation}")
    changed["prediction_events"] = [
        deepcopy(row) for row in events if row.get("row_type") == "prediction_event"
    ]
    changed["feedback_events"] = [
        deepcopy(row) for row in events if row.get("row_type") == "feedback_event"
    ]
    return changed


def run_mutation_controls(value: Mapping[str, Any], *, root: Path) -> list[JsonDict]:
    """Prove that every registered defect is rejected by the cold reader."""

    expected = {
        "changed_expert_prediction": "prediction_event_hash_mismatch",
        "deleted_checkpoint": "checkpoint_missing",
        "future_label_access": "feedback_before_reveal",
        "duplicated_feedback": "duplicate_feedback",
        "reordered_updates": "feedback_order_invalid",
    }
    rows: list[JsonDict] = []
    for mutation in REQUIRED_MUTATIONS:
        replay = cold_replay(mutate_fixture(value, mutation), root=root)
        error = expected[mutation]
        rows.append(
            {
                "attack": mutation,
                "expected": error,
                "observed": error if error in replay["errors"] else replay["errors"],
                "passed": error in replay["errors"],
            }
        )
    return rows


def _unit_rows() -> list[JsonDict]:
    """Name every frozen Exp7440 unit and mark the bounded prototype subset."""

    exercised = {
        ("hash_order", 0, TRAINING_SEEDS[0], LEARNED_ARM),
        ("hash_order", 8, TRAINING_SEEDS[0], LEARNED_ARM),
        ("hash_order", 0, TRAINING_SEEDS[0], NO_FEEDBACK_ARM),
    }
    return [
        {
            "order": order,
            "delay": delay,
            "seed": seed,
            "arm": arm,
            "condition": "ledger_prototype_not_scientific_stream",
            "attempted": (order, delay, seed, arm) in exercised,
            "completed": (order, delay, seed, arm) in exercised,
            "failed": False,
            "censored": False,
            "unstarted": (order, delay, seed, arm) not in exercised,
            "status": "prototype_complete"
            if (order, delay, seed, arm) in exercised
            else "unstarted_by_registered_prototype_scope",
        }
        for order in ORDERS
        for delay in DELAYS
        for seed in TRAINING_SEEDS
        for arm in ARMS
    ]


def _validation_passed(receipts: Sequence[Mapping[str, Any]], *, candidate: bool) -> bool:
    """Require the exact affected set and, for final output, all terminal readers."""

    required = set(
        AFFECTED_CHECK_NAMES if candidate else (*AFFECTED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)
    )
    observed = [str(row.get("name")) for row in receipts if row.get("required") is True]
    return (
        set(observed) == required
        and len(observed) == len(required)
        and all(
            row.get("passed") is True and row.get("exit_code") == 0
            for row in receipts
            if row.get("required") is True
        )
    )


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
) -> JsonDict:
    """Keep gate operands as bare scalars and explain the reason separately."""

    return {
        "check": check,
        "category": category,
        "op": "==",
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
        "principle": principle,
    }


def independent_reduce(value: Mapping[str, Any], *, root: Path) -> JsonDict:
    """Reduce readiness directly from raw ledger rows and mutation outcomes."""

    ledger = value.get("ledger") or value
    replay = cold_replay(ledger, root=root)
    mutations = value.get("mutation_control_rows")
    if not isinstance(mutations, list):
        mutations = run_mutation_controls(ledger, root=root)
    crash = ledger.get("crash_recovery") or {}
    no_feedback = ledger.get("no_feedback_control") or {}
    stages = ledger.get("service_stage_receipts") or value.get("service_stage_receipts") or []
    checks = {
        "cold_replay": replay["valid"] is True and replay["maximum_numeric_gap"] <= 1e-12,
        "mutation_controls": len(mutations) == len(REQUIRED_MUTATIONS)
        and all(row.get("passed") is True for row in mutations),
        "crash_recovery": crash.get("passed") is True,
        "no_feedback_state_fixed": no_feedback.get("passed") is True,
        "five_service_stages": [row.get("stage") for row in stages] == list(SERVICE_STAGES)
        and all(row.get("qualified") is True for row in stages),
    }
    return {
        "prediction_ledger_ready_score": int(all(checks.values())),
        "checks": checks,
        "cold_replay": replay,
        "mutation_control_rows": deepcopy(mutations),
    }


def _field_principles() -> JsonDict:
    """Explain required fields without wrapping their machine-readable values."""

    return {
        "schema": "A versioned identity prevents a newer reader from guessing old semantics.",
        "run_date": "The scheduled date stays separate from measured UTC and monotonic clocks.",
        "preconditions_checked": "Exact upstream values prevent missing evidence from becoming synthetic success.",
        "MODEL_SPECS": "An empty list states that this task planned no current LLM work.",
        "model_invoked": "Archived model-shaped rows do not count as a current invocation.",
        "invocation_counts": "Balanced zero counters make no-model execution independently checkable.",
        "inference_substrate": "The string names numeric ledger replay rather than model inference.",
        "inference_substrate_class": "No model load avoids applying a generation-time duration floor.",
        "execution_venue": "Host execution is separate from compute class and archived device evidence.",
        "duration_s": "Measured phase time is never padded to satisfy a plausibility rule.",
        "phase_spans": "Offsets bind progress boundaries, checkpoints, and completed units.",
        "random_seed": "Fit, stream, and resampling seeds stay frozen; projection is explicitly absent.",
        "reproducibility_checksum": "The checksum binds code, protocol, inputs, ledger, and validation scope.",
        "source_artifact_hashes": "Byte hashes retain original upstream classes and flags without rehabilitation.",
        "rows": "Every registered unit is present, including units intentionally left unstarted.",
        "sample_size_budget": "Planned, attempted, complete, failed, censored, and unstarted counts remain distinct.",
        "acceptance_gate_results": "Validity, completion, and safety gates retain their own categories.",
        "gate_check_summary": "A failed gate names its exact field and observed value.",
        "verifier_is_oracle": "Exact synthetic arithmetic is circular and cannot establish online value.",
        "honest_verdict": "The terminal prefix reports completed mechanics without a benefit claim.",
        "verdict_class": "The closed class is circular_positive for an exact analytic fixture.",
        "flagged_adversarial": "A critical verifier finding would disqualify readiness.",
        "validation_receipts": "Exact arguments, environments, exits, durations, and log hashes bind each reader.",
        "field_principles": "The artifact carries the reason for each ordinary field.",
        "promotion_score": "This milestone cannot change rollout or generator weights.",
        "prediction_ledger_ready_score": "One requires replay plus every causal mutation control.",
        "event_schema": "Saved values replace references to mutable expert objects.",
        "initial_state_manifest": "A cold reader can reconstruct state without producer update logic.",
        "protocol_hash": "The comparison protocol is bound before any future stream label is consumed.",
    }


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Bind stable evidence while excluding clocks and the checksum slot itself."""

    excluded = {
        "reproducibility_checksum",
        "started_at_utc",
        "completed_at_utc",
        "started_monotonic_ns",
        "completed_monotonic_ns",
        "duration_s",
        "phase_spans",
        "clock_identity",
    }
    stable = {key: deepcopy(item) for key, item in value.items() if key not in excluded}
    stable["validation_receipts"] = [
        {
            key: deepcopy(row.get(key))
            for key in (
                "name",
                "command_argv",
                "command_environment",
                "scope",
                "exit_code",
                "passed",
                "required",
            )
        }
        for row in value.get("validation_receipts", [])
    ]
    return canonical_hash(stable)


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name the first failed gate without hiding later failures."""

    failed = [row for row in gates if row.get("passed") is not True]
    first = failed[0] if failed else {}
    return {
        "required_checks_passed": not failed,
        "failed_required_checks": [row.get("check") for row in failed],
        "blocked_upstream": None,
        "blocked_path": None,
        "blocked_check": first.get("check"),
        "blocked_field": first.get("check"),
        "blocked_expected": first.get("expected"),
        "blocked_observed": first.get("observed"),
    }


def _build_artifact(
    root: Path,
    *,
    ledger: Mapping[str, Any],
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    phase_spans: Sequence[Mapping[str, Any]],
    started_at: str,
    started_ns: int,
    ended_ns: int,
    candidate: bool,
) -> JsonDict:
    """Assemble one schema-complete ledger-readiness record from raw evidence."""

    mutations = run_mutation_controls(ledger, root=root)
    protocol = build_replay_protocol()
    rows = _unit_rows()
    attempted = sum(row["attempted"] for row in rows)
    required_validation = _validation_passed(validation_receipts, candidate=candidate)
    reduction_input = {
        "ledger": deepcopy(dict(ledger)),
        "mutation_control_rows": mutations,
        "service_stage_receipts": deepcopy(ledger["service_stage_receipts"]),
    }
    reduced = independent_reduce(reduction_input, root=root)
    gates = [
        _gate(
            "required_validation",
            "validation",
            True,
            required_validation,
            required_validation,
            "Only the frozen affected checks and terminal readers authorize publication.",
        ),
        _gate(
            "independent_cold_replay",
            "validity",
            1,
            reduced["prediction_ledger_ready_score"],
            reduced["prediction_ledger_ready_score"] == 1,
            "A cold reader must reconstruct every numeric transition from saved operands.",
        ),
        _gate(
            "promotion_disabled",
            "safety",
            0,
            0,
            True,
            "A circular fixture cannot authorize rollout or generator changes.",
        ),
    ]
    ready = int(all(row["passed"] for row in gates))
    ended_at = utc_now()
    value: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "complete_circular_positive_prediction_ledger_ready"
        if ready
        else "complete_disqualified_prediction_ledger_invalid",
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "completed_at_utc": ended_at,
        "started_monotonic_ns": started_ns,
        "completed_monotonic_ns": ended_ns,
        "clock_identity": {
            "boot_id": _boot_id(),
            "segment_id": canonical_hash([started_at, started_ns, ended_ns]),
        },
        "preconditions_checked": deepcopy(list(preconditions)),
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_details": {
            "current_work": "analytic_numeric_ledger_and_validation",
            "model_device": None,
            "historical_evidence_class": "hash_bound_typed_sidecars",
            "full_scientific_stream_executed": False,
        },
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "duration_s": max(0.0, (ended_ns - started_ns) / 1e9),
        "model_duration_s": 0.0,
        "numeric_computation_duration_s": sum(
            int(row["duration_ns"]) for row in ledger["service_stage_receipts"]
        )
        / 1e9,
        "validation_duration_s": sum(
            float(row.get("duration_s") or 0.0) for row in validation_receipts
        ),
        "phase_spans": deepcopy(list(phase_spans)),
        "random_seed": {
            "fit_seeds": list(TRAINING_SEEDS),
            "projection_seed": None,
            "projection_seed_reason": "No random projection is used by the ledger fixture.",
            "stream_seed": 6_520_440,
            "resampling_seed": 6_520_440,
        },
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": rows,
        "sample_size_budget": {
            "planned": len(rows),
            "attempted": attempted,
            "completed": attempted,
            "failed": 0,
            "censored": 0,
            "unstarted": len(rows) - attempted,
            "independent_units": "source_groups_in_future_full_stream",
            "stopping_rule": "complete the bounded analytic ledger fixture only",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "verifier_is_oracle": True,
        "honest_verdict": "complete_circular_positive_prediction_ledger_ready"
        if ready
        else "complete_disqualified_prediction_ledger_invalid",
        "verdict_class": "circular_positive" if ready else "disqualified",
        "flagged_adversarial": False,
        "validation_receipts": deepcopy(list(validation_receipts)),
        "validation_manifest": {
            "test_paths": list(VALIDATION_MANIFEST.test_paths),
            "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
            "static_paths": list(VALIDATION_MANIFEST.static_paths),
            "affected_check_names": list(AFFECTED_CHECK_NAMES),
            "terminal_check_names": list(TERMINAL_CHECK_NAMES),
        },
        "field_principles": _field_principles(),
        "promotion_score": 0,
        "prediction_ledger_ready_score": ready,
        "event_schema": {
            "prediction": PREDICTION_SCHEMA,
            "feedback": FEEDBACK_SCHEMA,
            "prediction_label_fields_forbidden": ["label", "loss", "per_expert_loss"],
            "feedback_references": "prediction_event_hash",
        },
        "initial_state_manifest": deepcopy(ledger["initial_state_manifest"]),
        "protocol_hash": protocol["protocol_hash"],
        "replay_protocol": protocol,
        "ledger": deepcopy(dict(ledger)),
        "service_stage_receipts": deepcopy(ledger["service_stage_receipts"]),
        "mutation_control_rows": mutations,
        "independent_reduction": reduced,
        "small_ebm_training": {
            "performed_currently": False,
            "historical_receipt_class": "small_ebm_training",
            "current_fixture_class": "analytic_fixture_not_training",
        },
        "candidate_artifact": candidate,
    }
    value["reproducibility_checksum"] = artifact_checksum(value)
    return value


def _boot_id() -> str:  # pragma: no cover - host identity varies by process.
    """Read the Linux boot identity, with an explicit unavailable fallback."""

    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    except OSError:
        return "unavailable"


def build_fixture_artifact(
    root: Path,
    *,
    validation_receipts: Sequence[Mapping[str, Any]],
    relative_dir: Path,
) -> JsonDict:
    """Build deterministic private evidence for reducer and validator tests."""

    ledger = build_fixture_ledger(root, relative_dir=relative_dir)
    return _build_artifact(
        root,
        ledger=ledger,
        preconditions=[],
        source_hashes={},
        validation_receipts=validation_receipts,
        phase_spans=[],
        started_at="2026-09-20T00:00:00+00:00",
        started_ns=1,
        ended_ns=2,
        candidate=False,
    )


def validate_artifact(value: Mapping[str, Any], *, root: Path) -> list[str]:
    """Cold-check identity, declarations, raw replay, validation, and checksum."""

    errors: list[str] = []
    if value.get("schema") != SCHEMA or value.get("experiment_id") != EXPERIMENT_ID:
        errors.append("artifact_identity_invalid")
    if value.get("milestone") != MILESTONE or value.get("run_date") != RUN_DATE:
        errors.append("artifact_schedule_invalid")
    if value.get("MODEL_SPECS") != [] or value.get("model_invoked") is not False:
        errors.append("current_model_declaration_invalid")
    if value.get("invocation_counts") != ZERO_INVOCATION_COUNTS:
        errors.append("current_invocation_counts_invalid")
    if value.get("inference_substrate_class") != INFERENCE_SUBSTRATE_CLASS:
        errors.append("inference_substrate_class_invalid")
    if value.get("execution_venue") != EXECUTION_VENUE:
        errors.append("execution_venue_invalid")
    if value.get("promotion_score") != 0 or value.get("verifier_is_oracle") is not True:
        errors.append("claim_boundary_invalid")
    try:
        reduced = independent_reduce(value, root=root)
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        errors.append(f"independent_reduction_failed:{error}")
    else:
        if reduced != value.get("independent_reduction"):
            errors.append("independent_reduction_mismatch")
        if reduced.get("prediction_ledger_ready_score") != value.get(
            "prediction_ledger_ready_score"
        ):
            errors.append("prediction_ledger_ready_score_mismatch")
    candidate = value.get("candidate_artifact") is True
    if not _validation_passed(value.get("validation_receipts") or [], candidate=candidate):
        errors.append("required_validation_incomplete")
    if value.get("protocol_hash") != (value.get("replay_protocol") or {}).get("protocol_hash"):
        errors.append("protocol_hash_mismatch")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def build_blocked_artifact(failed: Mapping[str, Any]) -> JsonDict:
    """Publish unchanged external absence as a terminal, specific blocker."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": f"blocked_{failed.get('check')}",
        "run_date": RUN_DATE,
        "preconditions_checked": [deepcopy(dict(failed))],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "rows": [],
        "acceptance_gate_results": [],
        "gate_check_summary": {
            "upstream": failed.get("upstream"),
            "path": failed.get("path"),
            "check": failed.get("check"),
            "field": failed.get("field"),
            "expected": failed.get("expected"),
            "observed": failed.get("observed"),
        },
        "verifier_is_oracle": True,
        "honest_verdict": f"blocked_{failed.get('check')}",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "validation_receipts": [],
        "field_principles": _field_principles(),
        "promotion_score": 0,
        "prediction_ledger_ready_score": 0,
    }


def _span(
    phase: str, phase_started: float, run_started: float, units: int
) -> JsonDict:  # pragma: no cover - authentic timing receipt.
    """Close one monotonic phase and retain its completed-unit checkpoint."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "started_offset_s": phase_started - run_started,
        "ended_offset_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
        "checkpoint": f"{phase}:{units}",
    }


def _terminal_commands(candidate: Path) -> list[PlannedCommand]:  # pragma: no cover
    """Build the fresh replay, reduction, adversarial, and strict readers."""

    common = (".venv/bin/python", "-u", WRAPPER_PATH.as_posix(), "--date", RUN_DATE)
    return [
        PlannedCommand(
            validation_scope.CommandSpec(
                "cold_artifact_replay", (*common, "--cold-replay", str(candidate)), "candidate"
            ),
            "completion",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "independent_raw_reduction",
                (*common, "--independent-reduce", str(candidate)),
                "candidate",
            ),
            "completion",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "adversarial_verify",
                (".venv/bin/python", "-u", "scripts/adversarial_verify.py", str(candidate)),
                "candidate",
            ),
            "safety",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "verdict_row_consistency_strict",
                (
                    ".venv/bin/python",
                    "-u",
                    "scripts/verdict_row_consistency_lint.py",
                    "--strict",
                    str(candidate),
                ),
                "candidate",
            ),
            "completion",
            True,
        ),
    ]


def run_experiment(
    root: Path, run_date: str, *, output_path: Path = RESULT_PATH
) -> JsonDict:  # pragma: no cover - declared capability E2E.
    """Authenticate, collect, validate, replay cold, and atomically publish."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    run_started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_at = utc_now()
    spans: list[JsonDict] = []

    progress(run_started, "preconditions", "start")
    phase_started = time.monotonic()
    preconditions, source_hashes, _upstream = collect_preconditions(root)
    spans.append(_span("preconditions", phase_started, run_started, len(preconditions)))
    progress(run_started, "preconditions", "end", completed=len(preconditions))
    failed = next((row for row in preconditions if row.get("passed") is not True), None)
    if failed is not None:
        blocked = build_blocked_artifact(failed)
        progress(run_started, "publish", "before_atomic_write", status="blocked")
        atomic_json(root / output_path, blocked)
        progress(run_started, "publish", "after_atomic_write", status="blocked")
        return blocked

    progress(run_started, "collection", "before_benchmark", planned_events=5)
    phase_started = time.monotonic()
    ledger = build_fixture_ledger(root, relative_dir=RAW_DIR / "ledger")
    spans.append(_span("collection", phase_started, run_started, len(ledger["events"])))
    progress(run_started, "collection", "after_benchmark", completed=len(ledger["events"]))
    for row in ledger["initial_state_manifest"]["checkpoints"]:
        source_hashes[row["path"]] = {
            "path": row["path"],
            "sha256": row["sha256"],
            "source_receipt_class": row["source_receipt_class"],
            "original_flagged_adversarial": None,
        }
    source_hashes[ledger["ledger_path"]] = {
        "path": ledger["ledger_path"],
        "sha256": ledger["ledger_sha256"],
        "source_receipt_class": "current_prediction_feedback_ledger",
        "original_flagged_adversarial": None,
    }
    for relative in (MODULE_PATH, WRAPPER_PATH, TEST_PATH, SPEC_PATH):
        source_hashes[relative.as_posix()] = {
            "path": relative.as_posix(),
            "sha256": sha256_file(root / relative),
            "original_flagged_adversarial": None,
        }

    private_root = Path(tempfile.mkdtemp(prefix="exp7450-validation-", dir="/tmp"))
    commands = build_command_plan(root, VALIDATION_MANIFEST, private_root)
    plan_errors = validate_command_plan(root, VALIDATION_MANIFEST, commands)
    progress(run_started, "affected_validation", "before_subprocesses", planned=len(commands))
    phase_started = time.monotonic()
    affected = (
        []
        if plan_errors
        else run_categorized_commands(
            root,
            [PlannedCommand(command, "required_validation", True) for command in commands],
            log_dir=root / RAW_DIR / "validation/affected",
        )
    )
    affected_reduction = reduce_affected_receipts(root, VALIDATION_MANIFEST, affected)
    spans.append(_span("affected_validation", phase_started, run_started, len(affected)))
    progress(
        run_started,
        "affected_validation",
        "after_subprocesses",
        completed=len(affected),
        passed=affected_reduction["passed"] and not plan_errors,
    )
    candidate = _build_artifact(
        root,
        ledger=ledger,
        preconditions=preconditions,
        source_hashes=source_hashes,
        validation_receipts=affected,
        phase_spans=spans,
        started_at=started_at,
        started_ns=started_ns,
        ended_ns=time.monotonic_ns(),
        candidate=True,
    )
    candidate_path = root / RAW_DIR / "measured_terminal_candidate.json"
    atomic_json(candidate_path, candidate)

    terminal_commands = _terminal_commands(candidate_path)
    progress(run_started, "terminal_validation", "before_subprocesses", planned=4)
    phase_started = time.monotonic()
    terminal = run_categorized_commands(
        root, terminal_commands, log_dir=root / RAW_DIR / "validation/terminal"
    )
    spans.append(_span("terminal_validation", phase_started, run_started, len(terminal)))
    progress(
        run_started,
        "terminal_validation",
        "after_subprocesses",
        completed=len(terminal),
        passed=all(row.get("passed") is True for row in terminal),
    )
    final = _build_artifact(
        root,
        ledger=ledger,
        preconditions=preconditions,
        source_hashes=source_hashes,
        validation_receipts=[*affected, *terminal],
        phase_spans=spans,
        started_at=started_at,
        started_ns=started_ns,
        ended_ns=time.monotonic_ns(),
        candidate=False,
    )
    errors = validate_artifact(final, root=root)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    progress(run_started, "publish", "before_atomic_write", path=output_path)
    atomic_json(root / output_path, final)
    progress(run_started, "publish", "after_atomic_write", status=final["status"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed date plus fresh-process replay and reduction modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the declared collector or one strict fresh-process terminal reader."""

    args = parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = args.root.resolve()
    if args.cold_replay is not None:
        value = _load_object(args.cold_replay)
        errors = validate_artifact(value, root=root) if value else ["artifact_unreadable"]
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_reduce is not None:
        value = _load_object(args.independent_reduce)
        try:
            reduced = independent_reduce(value, root=root)
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
            print(json.dumps({"error": str(error)}, sort_keys=True), flush=True)
            return 1
        print(json.dumps(reduced, sort_keys=True), flush=True)
        return 0
    run_experiment(root, args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - thin module CLI.
    raise SystemExit(main())
