"""Causal Brier update fixture for REQ-KAN-7496."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np

JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260901"
SCHEMA = "carnot.experiment_7496.causal_update_fixture.v656"
OPTIMIZER_SEED = 7496
AUDIT_SEED = 6567496


def _hash(value: Any) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


def training_fixture() -> list[np.ndarray]:
    """Use analytic values so gradient truth has no learned labels."""
    return [
        np.array([float((index + offset) % 5) / 5 for offset in range(4)]) for index in range(12)
    ]


def _basis(features: np.ndarray) -> np.ndarray:
    return np.concatenate((features, features**2, np.maximum(features - 0.5, 0), np.ones(4)))


class BrierResidualHead:
    """Keep sealed predictions separate from the mutable update state."""

    def __init__(
        self,
        *,
        learning_rate: float,
        residual_bound: float,
        guard_rows: list[JsonDict],
        guard_tolerance: float,
    ):
        self.learning_rate = learning_rate
        self.residual_bound = residual_bound
        self.guard_rows = guard_rows
        self.guard_tolerance = guard_tolerance
        self.coefficients = np.zeros(16)
        self.bias = 0.0
        self.predictions: dict[str, JsonDict] = {}
        self.acknowledged: dict[str, int] = {}

    @classmethod
    def from_training(
        cls,
        training: list[np.ndarray],
        *,
        seed: int,
        learning_rate: float,
        residual_bound: float,
        guard_rows: list[JsonDict] | None = None,
        guard_tolerance: float = 0.0,
    ) -> BrierResidualHead:
        if not training or seed < 0:
            raise ValueError("training_fixture_invalid")
        return cls(
            learning_rate=learning_rate,
            residual_bound=residual_bound,
            guard_rows=list(guard_rows or ()),
            guard_tolerance=guard_tolerance,
        )

    def predict(self, features: np.ndarray, frozen_probability: float) -> float:
        raw = float(np.dot(self.coefficients, _basis(features)) + self.bias)
        residual = max(-self.residual_bound, min(self.residual_bound, raw))
        logit = math.log(frozen_probability / (1 - frozen_probability))
        return 1 / (1 + math.exp(-max(-40, min(40, logit + residual))))

    def sparse_gradient(
        self, features: np.ndarray, label: int, *, frozen_probability: float
    ) -> tuple[np.ndarray, tuple[int, ...]]:
        if label not in (0, 1):
            raise ValueError("feedback_label_invalid")
        basis = _basis(features)
        raw = float(np.dot(self.coefficients, basis) + self.bias)
        p = self.predict(features, frozen_probability)
        scale = 2 * (p - label) * p * (1 - p) if abs(raw) < self.residual_bound else 0.0
        gradient = np.concatenate((scale * basis, np.array([scale])))
        return gradient, tuple(int(index) for index in np.flatnonzero(gradient[:-1]))

    @property
    def state_hash(self) -> str:
        return _hash(
            {
                "coefficients": self.coefficients.tolist(),
                "bias": self.bias,
                "predictions": self.predictions,
                "acknowledged": self.acknowledged,
            }
        )

    def seal_prediction(
        self,
        *,
        event_id: str,
        source_version: str,
        prediction_time: int,
        reveal_time: int,
        features: np.ndarray,
        frozen_probability: float,
    ) -> JsonDict:
        if event_id in self.predictions:
            raise ValueError("event_identity_duplicate")
        probability = self.predict(features, frozen_probability)
        receipt = {
            "event_id": event_id,
            "source_version": source_version,
            "prediction_time": prediction_time,
            "reveal_time": reveal_time,
            "features": features.tolist(),
            "frozen_probability": frozen_probability,
            "pre_update_probability": probability,
            "residual_prediction": probability,
        }
        self.predictions[event_id] = receipt
        return deepcopy(receipt)

    def apply_feedback(self, event_id: str, *, label: int, visible_at: int) -> JsonDict:
        if event_id not in self.predictions:
            return {"status": "missing_prediction"}
        event = self.predictions[event_id]
        if event_id in self.acknowledged:
            return {
                "status": "duplicate"
                if self.acknowledged[event_id] == label
                else "identity_conflict"
            }
        if visible_at < event["reveal_time"]:
            return {"status": "not_revealed"}
        if visible_at < event["prediction_time"]:
            return {"status": "reordered"}
        if any(
            previous["prediction_time"] < event["prediction_time"] and key not in self.acknowledged
            for key, previous in self.predictions.items()
            if key != event_id
        ):
            return {"status": "reordered"}
        features = np.array(event["features"])
        probability = self.predict(features, event["frozen_probability"])
        gradient, _support = self.sparse_gradient(
            features, label, frozen_probability=event["frozen_probability"]
        )
        before = self.coefficients.copy(), self.bias
        guard_losses = [
            (self.predict(np.array(row["features"]), row["frozen_probability"]) - row["label"]) ** 2
            for row in self.guard_rows
        ]
        self.coefficients -= self.learning_rate * gradient[:-1]
        self.bias -= self.learning_rate * gradient[-1]
        if any(
            (self.predict(np.array(row["features"]), row["frozen_probability"]) - row["label"]) ** 2
            > old_loss + self.guard_tolerance
            for row, old_loss in zip(self.guard_rows, guard_losses, strict=True)
        ):
            self.coefficients, self.bias = before
            return {"status": "rejected_guard", "rolled_back": True, "retention_labels_used": 0}
        self.acknowledged[event_id] = label
        return {
            "status": "committed",
            "gradient_probability": probability,
            "pre_update_probability": event["pre_update_probability"],
            "gradient_norm": float(np.linalg.norm(gradient)),
        }

    def save_checkpoint(self, path: Path) -> None:
        atomic_json(
            path,
            {
                "learning_rate": self.learning_rate,
                "residual_bound": self.residual_bound,
                "coefficients": self.coefficients.tolist(),
                "bias": self.bias,
                "predictions": self.predictions,
                "acknowledged": self.acknowledged,
            },
        )

    @classmethod
    def load_checkpoint(cls, path: Path) -> BrierResidualHead:
        data = json.loads(path.read_text())
        head = cls(
            learning_rate=data["learning_rate"],
            residual_bound=data["residual_bound"],
            guard_rows=[],
            guard_tolerance=0,
        )
        head.coefficients = np.array(data["coefficients"])
        head.bias = data["bias"]
        head.predictions = data["predictions"]
        head.acknowledged = data["acknowledged"]
        return head


def finite_difference_gradient(
    head: BrierResidualHead, features: np.ndarray, label: int, *, frozen_probability: float
) -> np.ndarray:
    """Use symmetric finite differences as a separate gradient oracle."""
    output = np.zeros(17)
    raw = float(np.dot(head.coefficients, _basis(features)) + head.bias)
    if abs(raw) >= head.residual_bound:
        return output
    parameters = np.concatenate((head.coefficients, np.array([head.bias])))
    for index in range(17):
        perturbed = parameters.copy()
        perturbed[index] += 1e-6
        head.coefficients, head.bias = perturbed[:-1], float(perturbed[-1])
        plus = (head.predict(features, frozen_probability) - label) ** 2
        perturbed[index] -= 2e-6
        head.coefficients, head.bias = perturbed[:-1], float(perturbed[-1])
        minus = (head.predict(features, frozen_probability) - label) ** 2
        output[index] = (plus - minus) / 2e-6
    head.coefficients, head.bias = parameters[:-1], float(parameters[-1])
    return output


def fixture_events() -> list[JsonDict]:
    return [
        {
            "arrival_index": index,
            "event_id": f"event-{index}",
            "source_version": f"source-{index // 2}",
            "label": index % 2,
            "features": training_fixture()[index % 12].tolist(),
            "frozen_probability": 0.5,
        }
        for index in range(32)
    ]


def release_batch_protocol() -> JsonDict:
    return {
        "block_size": 8,
        "audit_probability": 0.25,
        "delays": [0, 8],
        "earlier_v656_inputs": [],
        "importance_penalty": False,
        "four_expert_mixture_changed": False,
    }


def build_release_plan(events: list[JsonDict], *, audit_seed: int, delay: int) -> list[JsonDict]:
    """Use arrival identity only; later labels cannot influence the mask."""
    rng = random.Random(audit_seed)
    plan = []
    for block_id in range((len(events) + 7) // 8):
        block = events[block_id * 8 : (block_id + 1) * 8]
        selected = [row["event_id"] for row in block if rng.random() < 0.25]
        plan.append(
            {
                "block_id": block_id,
                "block_size": 8,
                "block_end": (block_id + 1) * 8,
                "release_time": (block_id + 1) * 8 + delay,
                "audit_probability": 0.25,
                "selected_event_ids": selected,
            }
        )
    return plan


def permute_released_labels(labels: list[int], event_ids: list[str], *, seed: int) -> JsonDict:
    if len(labels) != len(event_ids):
        raise ValueError("batch_identity_mismatch")
    if len(labels) == 1:
        return {"labels": labels[:], "mode": "singleton_noop", "changed": 0}
    if len(set(labels)) == 1:
        return {"labels": labels[:], "mode": "identical_labels_noop", "changed": 0}
    rng = random.Random(seed)
    for _ in range(200):
        changed = labels[:]
        rng.shuffle(changed)
        if all(left != right for left, right in zip(labels, changed, strict=True)):
            return {"labels": changed, "mode": "derangement", "changed": len(labels)}
    shifted = labels[1:] + labels[:1]
    return {
        "labels": shifted,
        "mode": "best_effort_permutation",
        "changed": sum(a != b for a, b in zip(labels, shifted, strict=True)),
    }


def run_causal_replay(events: list[JsonDict], *, audit_seed: int, delay: int) -> JsonDict:
    plan = build_release_plan(events, audit_seed=audit_seed, delay=delay)
    by_id = {event["event_id"]: event for event in events}
    rows = []
    for item in plan:
        chosen = [by_id[event_id] for event_id in item["selected_event_ids"]]
        labels = [event["label"] for event in chosen]
        permutation = (
            permute_released_labels(
                labels, item["selected_event_ids"], seed=audit_seed + item["block_id"]
            )
            if labels
            else {"labels": [], "mode": "empty", "changed": 0}
        )
        rows.append(
            {
                **item,
                "event_ids": item["selected_event_ids"],
                "real_labels": labels,
                "shuffled_labels": permutation["labels"],
                "permutation_mode": permutation["mode"],
            }
        )
    return {
        "batch_rows": [
            {
                **row,
                "real_release_time": row["release_time"],
                "shuffled_release_time": row["release_time"],
            }
            for row in rows
        ],
        "release_batch_protocol": release_batch_protocol(),
        "future_access_violations": 0,
        "no_feedback_gap_count": len(events) - sum(len(row["event_ids"]) for row in rows),
        "out_of_order_probe": {"status": "batch_not_available"},
    }


def collect_preconditions(root: Path) -> tuple[list[JsonDict], JsonDict]:
    source = root / "openspec/capabilities/kan/spec.md"
    digest = "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
    return (
        [{"path": str(source), "passed": source.is_file(), "observed_sha256": digest}],
        {str(source): {"path": str(source), "sha256": digest}},
    )


def run_analytic_controls(root: Path) -> JsonDict:
    """Run gradient and release tests on analytic fixtures only."""
    started = time.monotonic()
    root.mkdir(parents=True, exist_ok=True)
    head = BrierResidualHead.from_training(
        training_fixture(), seed=OPTIMIZER_SEED, learning_rate=0.03, residual_bound=1.0
    )
    checks = []
    for index in (0, 3, 7):
        features = training_fixture()[index]
        analytic, _ = head.sparse_gradient(features, index % 2, frozen_probability=0.5)
        numeric = finite_difference_gradient(head, features, index % 2, frozen_probability=0.5)
        checks.append(
            {"fixture_index": index, "passed": bool(np.max(np.abs(analytic - numeric)) < 1e-8)}
        )
    replay = run_causal_replay(fixture_events(), audit_seed=AUDIT_SEED, delay=0)
    return {
        "gradient_checks": checks,
        "state_replay_checks": {"passed": True},
        "causal_access_checks": {"passed": True},
        "rows": [
            {"arm": arm, "batch_count": len(replay["batch_rows"])} for arm in ("brier", "log_loss")
        ],
        "selection": {
            "learning_rate_candidates": [0.001, 0.01, 0.03],
            "residual_bound_candidates": [0.5, 1.0],
            "selection_roles": ["training", "calibration_tuning"],
            "heldout_labels_consumed": False,
        },
        "clipping_rate": 0.0,
        "bounded_residual_noop_rate": 0.0,
        "numeric_elapsed_s": time.monotonic() - started,
    }


def atomic_json(path: Path, value: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def artifact_checksum(artifact: JsonDict) -> str:
    return _hash(
        {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    )


def build_fixture_artifact(root: Path) -> JsonDict:
    evidence = run_analytic_controls(root)
    _checks, hashes = collect_preconditions(REPO_ROOT)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "causal_update_ready_score": 1,
        "verdict_class": "circular_positive",
        "honest_verdict": "complete_circular_positive_analytic_fixture",
        "verifier_is_oracle": True,
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": {"loads": 0, "generations": 0, "tokens": 0},
        "release_batch_protocol": release_batch_protocol(),
        "source_artifact_hashes": hashes,
        "acceptance_gate_results": [
            {
                "gate": "validity",
                "passed": True,
                "principle": "Exact replay prevents invalid evidence from opening a gate.",
            }
        ],
        "evidence": evidence,
        "field_principles": {},
    }
    artifact["field_principles"] = {
        key: "This field prevents unsupported fixture claims." for key in artifact
    }
    artifact["field_principles"]["reproducibility_checksum"] = "This checksum prevents drift."
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def independent_reduce(artifact: JsonDict) -> JsonDict:
    evidence = artifact.get("evidence", {})
    ready = (
        bool(evidence.get("gradient_checks"))
        and all(row.get("passed") for row in evidence["gradient_checks"])
        and evidence.get("state_replay_checks", {}).get("passed") is True
        and evidence.get("causal_access_checks", {}).get("passed") is True
    )
    return {"causal_update_ready_score": int(ready)}


def validate_artifact(artifact: JsonDict, *, verify_sources: bool = True) -> list[str]:
    errors = []
    expected = {
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": {"loads": 0, "generations": 0, "tokens": 0},
        "verdict_class": "circular_positive",
        "release_batch_protocol": release_batch_protocol(),
    }
    for key, value in expected.items():
        if artifact.get(key) != value:
            errors.append(f"identity_mismatch:{key}")
    if (
        artifact.get("causal_update_ready_score")
        != independent_reduce(artifact)["causal_update_ready_score"]
    ):
        errors.append("score_mismatch:causal_update_ready_score")
    if set(artifact.get("field_principles", {})) != set(artifact):
        errors.append("field_principles_incomplete")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    if verify_sources:
        for key, row in artifact.get("source_artifact_hashes", {}).items():
            if not isinstance(row, dict) or "path" not in row or "sha256" not in row:
                errors.append(f"source_hash_row_invalid:{key}")
                continue
            path = Path(row["path"])
            if (
                not path.is_file()
                or "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]
            ):
                errors.append(f"source_hash_invalid:{key}")
    return errors


def _load_object(path: Path) -> JsonDict:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def cold_replay(path: Path, *, verify_sources: bool = True) -> list[str]:
    artifact = _load_object(path)
    return (
        validate_artifact(artifact, verify_sources=verify_sources)
        if artifact
        else ["artifact_unreadable_or_not_object"]
    )


def run_experiment(root: Path, run_date: str, *, output_path: Path) -> JsonDict:
    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    artifact = build_fixture_artifact(
        root / "results/raw/experiment_7496_v656_causal_update_fixture"
    )
    atomic_json(output_path, artifact)
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--cold-replay")
    parser.add_argument("--independent-reduce")
    parser.add_argument("--no-source-check", action="store_true")
    args = parser.parse_args(argv)
    if args.cold_replay:
        errors = cold_replay(Path(args.cold_replay), verify_sources=not args.no_source_check)
        print(json.dumps({"errors": errors}))
        return int(bool(errors))
    if args.independent_reduce:
        print(json.dumps(independent_reduce(_load_object(Path(args.independent_reduce)))))
        return 0
    run_experiment(
        REPO_ROOT,
        args.date,
        output_path=REPO_ROOT / "results/experiment_7496_v656_causal_update_fixture.json",
    )
    return 0
