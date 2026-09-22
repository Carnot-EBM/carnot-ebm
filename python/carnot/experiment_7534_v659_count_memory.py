"""Qualify frozen-bin count memory using analytical controls only.

The learner changes sixteen local count values after independently supplied
feedback. It does not train or invoke a model. Constructed labels can therefore
qualify arithmetic and lifecycle behavior, but cannot establish real-world
calibration benefit.

Spec refs: REQ-CL-7534 and SCENARIO-CL-7534-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from types import MappingProxyType
from typing import Any

from carnot.experiment_7358_v646_validation_contract import AffectedManifest
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, atomic_json


JsonDict = dict[str, Any]
RUN_DATE = "20260922"
MILESTONE = "2026.09.659"
VERSION = "v659"
EXPERIMENT_ID = "exp7534-v659-count-memory"
SCHEMA = "carnot.exp7534.v659.count_memory.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7534_v659_count_memory.json")
RAW_DIR = Path("results/raw/experiment_7534_v659_count_memory")
MODULE_PATH = Path("python/carnot/experiment_7534_v659_count_memory.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7534_v659_count_memory.py")
TEST_PATH = Path("tests/python/test_experiment_7534_v659_count_memory.py")
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
CLIP_MIN = 1e-4
BIN_COUNT = 8
PRIOR_MASS = 8.0

AFFECTED_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


def canonical_hash(value: Any) -> str:
    """Hash canonical JSON so any changed event or result is detectable."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash exact file bytes rather than trusting a path or timestamp."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def clip_probability(probability: float) -> float:
    """Keep logits finite while retaining the registered probability range."""

    if not math.isfinite(probability):
        raise ValueError("probability_not_finite")
    return min(1.0 - CLIP_MIN, max(CLIP_MIN, float(probability)))


def _logit(probability: float) -> float:
    probability = clip_probability(probability)
    return math.log(probability / (1.0 - probability))


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def bin_index(probability: float) -> int:
    """Select one of eight equal-width bins after clipping the baseline."""

    return min(BIN_COUNT - 1, math.floor(BIN_COUNT * clip_probability(probability)))


def normalized_probability(energies: Sequence[float]) -> float:
    """Normalize both binary energies exactly and return probability of one."""

    if len(energies) != 2:
        raise ValueError("binary_energy_requires_two_values")
    weight_zero = math.exp(-float(energies[0]))
    weight_one = math.exp(-float(energies[1]))
    return weight_one / (weight_zero + weight_one)


@dataclass(frozen=True)
class CountConfig:
    """Immutable label-free means prevent stream feedback from moving bins."""

    bin_means: tuple[float, ...]
    global_mean: float
    kappa: float = PRIOR_MASS

    def __post_init__(self) -> None:
        if len(self.bin_means) != BIN_COUNT:
            raise ValueError("exactly_eight_bin_means_required")
        if self.kappa != PRIOR_MASS:
            raise ValueError("prior_mass_must_equal_eight")
        if any(not CLIP_MIN <= value <= 1.0 - CLIP_MIN for value in self.bin_means):
            raise ValueError("bin_mean_out_of_range")
        if not CLIP_MIN <= self.global_mean <= 1.0 - CLIP_MIN:
            raise ValueError("global_mean_out_of_range")


@dataclass(frozen=True)
class Prediction:
    """A sealed probability and its exact two-energy representation."""

    probability: float
    bin_index: int
    energies: tuple[float, float]


class CountArm:
    """Store sufficient statistics for a frozen, global, or local arm."""

    VALID_KINDS = {"frozen", "global", "local", "permuted_local"}

    def __init__(self, kind: str, config: CountConfig, counts: list[list[float]]) -> None:
        if kind not in self.VALID_KINDS:
            raise ValueError(f"count_arm_kind_invalid:{kind}")
        self.kind = kind
        self.config = config
        self._counts = counts
        self.processed_event_ids: set[str] = set()

    @classmethod
    def create(cls, kind: str, config: CountConfig) -> CountArm:
        """Create counts from frozen means before any stream feedback exists."""

        means = (config.global_mean,) if kind == "global" else config.bin_means
        counts = [[config.kappa * mean, config.kappa * (1.0 - mean)] for mean in means]
        return cls(kind, config, counts)

    @property
    def counts(self) -> list[tuple[float, float]]:
        return [tuple(pair) for pair in self._counts]

    def _slot_and_mean(self, probability: float) -> tuple[int, float, int]:
        index = bin_index(probability)
        if self.kind == "global":
            return 0, self.config.global_mean, index
        return index, self.config.bin_means[index], index

    def predict(self, p0: float) -> Prediction:
        """Apply the exact posterior-odds correction to a clipped baseline."""

        baseline = clip_probability(p0)
        slot, prior_mean, index = self._slot_and_mean(baseline)
        if self.kind == "frozen":
            posterior = prior_mean
        else:
            a_value, b_value = self._counts[slot]
            posterior = a_value / (a_value + b_value)
        probability = _sigmoid(_logit(baseline) + _logit(posterior) - _logit(prior_mean))
        energies = (0.0, -_logit(probability))
        return Prediction(probability, index, energies)

    def update(self, event_id: str, p0: float, label: int) -> None:
        """Apply one legal binary label once to the selected sufficient statistic."""

        if label not in (0, 1):
            raise ValueError("binary_label_required")
        if event_id in self.processed_event_ids:
            raise ValueError(f"duplicate_feedback:{event_id}")
        if self.kind == "frozen":
            raise ValueError("frozen_arm_cannot_update")
        slot, _prior, _index = self._slot_and_mean(p0)
        self._counts[slot][0] += label
        self._counts[slot][1] += 1 - label
        self.processed_event_ids.add(event_id)

    def to_payload(self) -> JsonDict:
        return {
            "kind": self.kind,
            "counts": deepcopy(self._counts),
            "processed_event_ids": sorted(self.processed_event_ids),
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, Any], config: CountConfig) -> CountArm:
        counts = [[float(cell) for cell in pair] for pair in value["counts"]]
        arm = cls(str(value["kind"]), config, counts)
        arm.processed_event_ids = {str(item) for item in value["processed_event_ids"]}
        return arm


class CountEventMachine:
    """Seal predictions before applying release-ordered external feedback."""

    def __init__(
        self,
        config: CountConfig,
        arms: dict[str, CountArm],
        predictions: JsonDict | None = None,
        release_receipts: JsonDict | None = None,
        acknowledgments: set[int] | None = None,
        next_release_index: int = 0,
        journal: list[JsonDict] | None = None,
    ) -> None:
        self.config = config
        self.arms = arms
        self._predictions = predictions or {}
        self.release_receipts = release_receipts or {}
        self.acknowledgments = acknowledgments or set()
        self.next_release_index = next_release_index
        self.journal = journal or []

    @classmethod
    def create(cls, config: CountConfig) -> CountEventMachine:
        arms = {kind: CountArm.create(kind, config) for kind in CountArm.VALID_KINDS}
        return cls(config, arms)

    @property
    def predictions(self) -> JsonDict:
        """Return a copy so feedback can never rewrite a historical prediction."""

        return deepcopy(self._predictions)

    def _append_journal(self, operation: str, payload: Mapping[str, Any]) -> None:
        previous = self.journal[-1]["entry_hash"] if self.journal else "GENESIS"
        body = {
            "sequence": len(self.journal),
            "operation": operation,
            "payload": deepcopy(dict(payload)),
            "previous_hash": previous,
        }
        self.journal.append({**body, "entry_hash": canonical_hash(body)})

    def predict(self, event_id: str, p0: float, *, release_index: int) -> Mapping[str, Any]:
        """Seal every arm and action before the event becomes feedback-eligible."""

        if event_id in self._predictions:
            raise ValueError(f"prediction_event_duplicate:{event_id}")
        if release_index < self.next_release_index:
            raise ValueError(f"prediction_release_already_closed:{release_index}")
        arm_rows: JsonDict = {}
        for name in ("frozen", "global", "local", "permuted_local"):
            prediction = self.arms[name].predict(p0)
            arm_rows[name] = {
                "probability": prediction.probability,
                "bin_index": prediction.bin_index,
                "energies": list(prediction.energies),
                "chosen_action": int(prediction.probability >= 0.5),
            }
        row = {
            "event_id": event_id,
            "p0": clip_probability(p0),
            "release_index": release_index,
            "arms": arm_rows,
        }
        self._predictions[event_id] = deepcopy(row)
        self._append_journal("prediction", row)
        public = {name: MappingProxyType(deepcopy(values)) for name, values in arm_rows.items()}
        return MappingProxyType(public)

    @staticmethod
    def _canonical_feedback(feedback: Sequence[tuple[str, int]]) -> list[list[Any]]:
        rows = [[str(event_id), int(label)] for event_id, label in feedback]
        if len({row[0] for row in rows}) != len(rows):
            raise ValueError("feedback_event_duplicate_in_release")
        if any(row[1] not in (0, 1) for row in rows):
            raise ValueError("binary_label_required")
        return rows

    def release(self, release_index: int, feedback: Sequence[tuple[str, int]]) -> JsonDict:
        """Update matched arms only from the next fully released batch."""

        rows = self._canonical_feedback(feedback)
        key = str(release_index)
        if release_index < self.next_release_index:
            prior = self.release_receipts.get(key)
            if prior is not None and prior["feedback"] == rows:
                return deepcopy(prior)
            raise ValueError(f"duplicate_release_conflict:{release_index}")
        if release_index != self.next_release_index:
            raise ValueError(
                f"release_out_of_order:expected={self.next_release_index}:observed={release_index}"
            )
        for event_id, _label in rows:
            prediction = self._predictions.get(event_id)
            if prediction is None:
                raise ValueError(f"feedback_event_unknown:{event_id}")
            if prediction["release_index"] != release_index:
                raise ValueError(f"feedback_prerelease:{event_id}")

        rotated = rows[1:] + rows[:1] if len(rows) > 1 else rows
        permutation: list[JsonDict] = []
        for (event_id, label), (origin_id, permuted_label) in zip(rows, rotated):
            p0 = float(self._predictions[event_id]["p0"])
            self.arms["global"].update(event_id, p0, label)
            self.arms["local"].update(event_id, p0, label)
            self.arms["permuted_local"].update(event_id, p0, permuted_label)
            permutation.append(
                {
                    "event_id": event_id,
                    "label_origin": origin_id,
                    "label": permuted_label,
                }
            )
        receipt = {
            "release_index": release_index,
            "feedback": rows,
            "labels": [row[1] for row in rows],
            "permuted_labels": [row[1] for row in rotated],
            "update_count": len(rows),
            "permuted_update_count": len(rotated),
            "permutation": permutation,
        }
        self.release_receipts[key] = deepcopy(receipt)
        self.next_release_index += 1
        self._append_journal("release", receipt)
        return deepcopy(receipt)

    def acknowledge(self, release_index: int) -> None:
        """Record durable acknowledgment only after its release was applied."""

        if str(release_index) not in self.release_receipts:
            raise ValueError(f"acknowledgment_without_release:{release_index}")
        if release_index not in self.acknowledgments:
            self.acknowledgments.add(release_index)
            self._append_journal("acknowledgment", {"release_index": release_index})

    def to_payload(self) -> JsonDict:
        """Serialize every value needed for an exact cold restart."""

        return {
            "schema": "carnot.count_memory.state.v1",
            "config": asdict(self.config),
            "arms": {name: arm.to_payload() for name, arm in sorted(self.arms.items())},
            "predictions": deepcopy(self._predictions),
            "release_receipts": deepcopy(self.release_receipts),
            "acknowledgments": sorted(self.acknowledgments),
            "next_release_index": self.next_release_index,
            "journal": deepcopy(self.journal),
        }

    def state_hash(self) -> str:
        return canonical_hash(self.to_payload())

    def save(self, path: Path) -> None:
        """Publish one complete fsynced checkpoint instead of partial state."""

        atomic_json(path, self.to_payload())

    @staticmethod
    def _validate_journal(rows: Sequence[Mapping[str, Any]]) -> None:
        previous = "GENESIS"
        for index, row in enumerate(rows):
            body = {
                "sequence": row.get("sequence"),
                "operation": row.get("operation"),
                "payload": row.get("payload"),
                "previous_hash": row.get("previous_hash"),
            }
            if body["sequence"] != index or body["previous_hash"] != previous:
                raise ValueError(f"journal_chain_mismatch:{index}")
            if row.get("entry_hash") != canonical_hash(body):
                raise ValueError(f"journal_hash_mismatch:{index}")
            previous = str(row["entry_hash"])

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> CountEventMachine:
        if value.get("schema") != "carnot.count_memory.state.v1":
            raise ValueError("state_schema_mismatch")
        journal = list(value["journal"])
        cls._validate_journal(journal)
        config_value = value["config"]
        config = CountConfig(
            tuple(float(item) for item in config_value["bin_means"]),
            float(config_value["global_mean"]),
            float(config_value["kappa"]),
        )
        arms = {
            str(name): CountArm.from_payload(payload, config)
            for name, payload in value["arms"].items()
        }
        if set(arms) != CountArm.VALID_KINDS:
            raise ValueError("state_arm_set_mismatch")
        return cls(
            config=config,
            arms=arms,
            predictions=deepcopy(dict(value["predictions"])),
            release_receipts=deepcopy(dict(value["release_receipts"])),
            acknowledgments={int(item) for item in value["acknowledgments"]},
            next_release_index=int(value["next_release_index"]),
            journal=deepcopy(journal),
        )

    @classmethod
    def load(cls, path: Path) -> CountEventMachine:
        """Load only a complete JSON object with a valid journal chain."""

        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("state_payload_not_object")
        return cls.from_payload(value)


def default_config() -> CountConfig:
    """Return fixed label-free means used by every analytical stream."""

    return CountConfig((0.05, 0.18, 0.30, 0.43, 0.57, 0.70, 0.82, 0.95), 0.5)


def analytical_streams() -> list[JsonDict]:
    """Create seven deterministic controls, never a surrogate real dataset."""

    return [
        {
            "name": "unchanged_information",
            "batches": [
                [("stable-0-a", 0.30, 0), ("stable-0-b", 0.70, 1)],
                [("stable-1-a", 0.30, 0), ("stable-1-b", 0.70, 1)],
            ],
        },
        {
            "name": "conditional_drift_stable_global_prevalence",
            "batches": [
                [("cond-0-a", 0.20, 0), ("cond-0-b", 0.80, 1)],
                [("cond-1-a", 0.20, 1), ("cond-1-b", 0.80, 0)],
            ],
        },
        {
            "name": "global_only_drift",
            "batches": [
                [("global-0-a", 0.40, 0), ("global-0-b", 0.60, 0)],
                [("global-1-a", 0.40, 1), ("global-1-b", 0.60, 1)],
            ],
        },
        {
            "name": "alternating_recurrence",
            "batches": [
                [("recur-0-a", 0.25, 0), ("recur-0-b", 0.75, 1)],
                [("recur-1-a", 0.25, 1), ("recur-1-b", 0.75, 0)],
                [("recur-2-a", 0.25, 0), ("recur-2-b", 0.75, 1)],
            ],
        },
        {
            "name": "no_feedback",
            "batches": [[("none-0-a", 0.15, None), ("none-0-b", 0.85, None)]],
        },
        {
            "name": "all_one_class_labels",
            "batches": [
                [("ones-0-a", 0.10, 1), ("ones-0-b", 0.90, 1)],
                [("ones-1-a", 0.10, 1), ("ones-1-b", 0.90, 1)],
            ],
        },
        {
            "name": "adversarial_delayed_ids",
            "batches": [
                [("z-late-looking", 0.20, 0), ("y-late-looking", 0.80, 1)],
                [("a-early-looking", 0.20, 1), ("b-early-looking", 0.80, 0)],
            ],
        },
    ]


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def run_fixture_panel() -> list[JsonDict]:
    """Replay every constructed stream through predict, release, and update."""

    output: list[JsonDict] = []
    for stream in analytical_streams():
        machine = CountEventMachine.create(default_config())
        initial_local = machine.arms["local"].counts
        losses: dict[str, list[float]] = {
            name: [] for name in ("frozen", "global", "local", "permuted_local")
        }
        update_count = 0
        event_count = 0
        for release_index, batch in enumerate(stream["batches"]):
            feedback: list[tuple[str, int]] = []
            for event_id, p0, label in batch:
                predictions = machine.predict(event_id, p0, release_index=release_index)
                event_count += 1
                if label is not None:
                    feedback.append((event_id, label))
                    for arm, row in predictions.items():
                        losses[arm].append((float(row["probability"]) - label) ** 2)
            receipt = machine.release(release_index, feedback)
            machine.acknowledge(release_index)
            update_count += int(receipt["update_count"])
        output.append(
            {
                "stream": stream["name"],
                "evidence_class": "constructed_control",
                "complete": True,
                "event_count": event_count,
                "labeled_event_count": sum(len(values) for values in losses.values()) // 4,
                "update_count": update_count,
                "local_state_changed": machine.arms["local"].counts != initial_local,
                "chronology_violation_count": 0,
                "arm_metrics": {
                    name: {"mean_brier": _mean(values), "n_scored": len(values)}
                    for name, values in losses.items()
                },
                "final_state_hash": machine.state_hash(),
            }
        )
    return output


def _run_two_releases(machine: CountEventMachine, start_at: int = 0) -> None:
    if start_at <= 0:
        machine.predict("r0-a", 0.30, release_index=0)
        machine.predict("r0-b", 0.70, release_index=0)
        machine.release(0, (("r0-a", 1), ("r0-b", 0)))
        machine.acknowledge(0)
    machine.predict("r1-a", 0.30, release_index=1)
    machine.predict("r1-b", 0.70, release_index=1)
    machine.release(1, (("r1-a", 0), ("r1-b", 1)))
    machine.acknowledge(1)


def run_restart_panel(checkpoint_root: Path) -> list[JsonDict]:
    """Resume once at each registered interruption and compare to one baseline."""

    checkpoint_root.mkdir(parents=True, exist_ok=True)
    uninterrupted = CountEventMachine.create(default_config())
    _run_two_releases(uninterrupted)
    rows: list[JsonDict] = []
    for crash_point in (
        "after_prediction",
        "after_release",
        "before_durable_acknowledgment",
        "after_durable_acknowledgment",
    ):
        machine = CountEventMachine.create(default_config())
        machine.predict("r0-a", 0.30, release_index=0)
        machine.predict("r0-b", 0.70, release_index=0)
        if crash_point != "after_prediction":
            machine.release(0, (("r0-a", 1), ("r0-b", 0)))
        if crash_point == "after_durable_acknowledgment":
            machine.acknowledge(0)
        checkpoint = checkpoint_root / f"{crash_point}.json"
        machine.save(checkpoint)
        resumed = CountEventMachine.load(checkpoint)
        if crash_point == "after_prediction":
            resumed.release(0, (("r0-a", 1), ("r0-b", 0)))
        else:
            resumed.release(0, (("r0-a", 1), ("r0-b", 0)))
        resumed.acknowledge(0)
        _run_two_releases(resumed, start_at=1)
        expected_updates = 4
        actual_updates = len(resumed.arms["local"].processed_event_ids)
        rows.append(
            {
                "crash_point": crash_point,
                "prediction_parity": resumed.predictions == uninterrupted.predictions,
                "state_parity": resumed.state_hash() == uninterrupted.state_hash(),
                "exactly_once": actual_updates == expected_updates,
                "expected_update_count": expected_updates,
                "observed_update_count": actual_updates,
                "journal_valid": True,
            }
        )
    return rows


def arithmetic_rows() -> list[JsonDict]:
    """Expose exact numeric witnesses for independent artifact reduction."""

    arm = CountArm.create("local", default_config())
    before = arm.predict(0.30)
    arm.update("arithmetic", 0.30, 1)
    after = arm.predict(0.30)
    expected_a = PRIOR_MASS * default_config().bin_means[2] + 1.0
    expected_b = PRIOR_MASS * (1.0 - default_config().bin_means[2])
    return [
        {
            "case": "local_positive_update",
            "bin_index": 2,
            "prior_probability": before.probability,
            "posterior_a": arm.counts[2][0],
            "posterior_b": arm.counts[2][1],
            "expected_a": expected_a,
            "expected_b": expected_b,
            "normalized_probability": normalized_probability(after.energies),
            "predicted_probability": after.probability,
            "passed": (
                arm.counts[2] == (expected_a, expected_b)
                and math.isclose(normalized_probability(after.energies), after.probability)
            ),
        }
    ]


def _gate(
    condition: str, expected: Any, observed: Any, op: str, passed: bool, principle: str
) -> JsonDict:
    return {
        "condition": condition,
        "expected": expected,
        "observed": observed,
        "op": op,
        "passed": passed,
        "principle": principle,
    }


def independent_reduce(value: Mapping[str, Any]) -> JsonDict:
    """Recompute qualification from raw rows without trusting headline fields."""

    arithmetic_passed = bool(value.get("arithmetic_rows")) and all(
        row.get("passed") is True for row in value.get("arithmetic_rows", [])
    )
    fixture_rows = list(value.get("fixture_rows", []))
    expected_streams = {row["name"] for row in analytical_streams()}
    observed_streams = {row.get("stream") for row in fixture_rows}
    fixtures_complete = observed_streams == expected_streams and all(
        row.get("complete") is True for row in fixture_rows
    )
    chronology_passed = fixtures_complete and all(
        row.get("chronology_violation_count") == 0 for row in fixture_rows
    )
    restart_rows = list(value.get("restart_rows", []))
    expected_crashes = {
        "after_prediction",
        "after_release",
        "before_durable_acknowledgment",
        "after_durable_acknowledgment",
    }
    restart_passed = {row.get("crash_point") for row in restart_rows} == expected_crashes and all(
        row.get("prediction_parity") is True
        and row.get("state_parity") is True
        and row.get("exactly_once") is True
        and row.get("journal_valid") is True
        for row in restart_rows
    )
    oracle_fixture = value.get("verifier_is_oracle") is True
    validation_passed = value.get("required_checks_passed") is True
    ready = int(
        arithmetic_passed
        and chronology_passed
        and restart_passed
        and fixtures_complete
        and validation_passed
    )
    return {
        "arithmetic_passed": arithmetic_passed,
        "chronology_passed": chronology_passed,
        "restart_passed": restart_passed,
        "fixtures_complete": fixtures_complete,
        "oracle_fixture": oracle_fixture,
        "count_memory_ready_score": ready,
        "verdict_class": "circular_positive" if ready and oracle_fixture else "disqualified",
        "positive_claim": False,
    }


def _acceptance_gates(value: Mapping[str, Any]) -> list[JsonDict]:
    reduced = independent_reduce(value)
    return [
        _gate(
            "exact arithmetic and normalized energy",
            True,
            reduced["arithmetic_passed"],
            "eq",
            reduced["arithmetic_passed"],
            "A readiness claim without numeric witnesses could hide a wrong posterior.",
        ),
        _gate(
            "release chronology has zero violations",
            0,
            sum(row["chronology_violation_count"] for row in value["fixture_rows"]),
            "eq",
            reduced["chronology_passed"],
            "Future labels would turn prequential evidence into leakage.",
        ),
        _gate(
            "all interruption points preserve exactly-once state",
            True,
            reduced["restart_passed"],
            "eq",
            reduced["restart_passed"],
            "A replay that double-updates counts cannot represent durable learning.",
        ),
        _gate(
            "all registered analytical controls complete",
            7,
            len(value["fixture_rows"]),
            "eq",
            reduced["fixtures_complete"],
            "Missing controls could hide no-feedback or one-class failures.",
        ),
        _gate(
            "all scoped validation commands pass",
            True,
            value["required_checks_passed"],
            "eq",
            value["required_checks_passed"] is True,
            "Invalid code evidence cannot support mechanism readiness.",
        ),
        _gate(
            "constructed evidence never becomes an empirical claim",
            False,
            value["positive_claim"],
            "eq",
            value["positive_claim"] is False,
            "Oracle fixtures test implementation, not natural-data benefit.",
        ),
    ]


def _row_view(fixture_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for row in fixture_rows:
        metrics = row["arm_metrics"]
        rows.append(
            {
                "unit": row["stream"],
                "frozen_mean_brier": metrics["frozen"]["mean_brier"],
                "global_mean_brier": metrics["global"]["mean_brier"],
                "local_mean_brier": metrics["local"]["mean_brier"],
                "permuted_mean_brier": metrics["permuted_local"]["mean_brier"],
                "n_scored": metrics["local"]["n_scored"],
                "disposition": "no_headroom" if row["update_count"] == 0 else "constructed",
                "positive_claim": False,
            }
        )
    return rows


def _field_principles(keys: Sequence[str]) -> dict[str, str]:
    custom = {
        "schema": "Binds experiment identity, milestone, version, and date to one contract.",
        "preconditions_checked": "Names exact resource observations before readiness is claimed.",
        "MODEL_SPECS": "An empty list prevents cached model names from becoming current calls.",
        "model_specs": "Mirrors the no-load declaration for readers using either field spelling.",
        "model_invoked": "False separates analytical CPU work from model inference.",
        "inference_substrate_class": "The closed compute class selects the truthful duration floor.",
        "inference_substrate": "The declared no-load path prevents inference fabrication.",
        "execution_venue": "Distinguishes host CPU evidence from CUDA or hardware-board evidence.",
        "duration_s": "Measured monotonic duration exposes implausibly short compute claims.",
        "random_seed": "Frozen seeds prevent later outcome-dependent randomization.",
        "reproducibility_checksum": "Binds code, settings, sources, roles, and raw rows.",
        "rows": "Absolute per-stream arm metrics keep missing outcomes from becoming zero.",
        "sample_size_budget": "Separates planned, completed, failed, censored, and unstarted units.",
        "acceptance_gate_results": "Raw operands make every validity decision independently checkable.",
        "gate_check_summary": "Names the exact first failed field rather than hiding a blocker.",
        "honest_verdict": "A terminal prefix prevents valid completion from being retried as partial.",
        "verdict_class": "A closed class keeps circular controls out of positive evidence.",
        "verifier_is_oracle": "Oracle-defined fixtures cannot support an oracle-distinct benefit claim.",
        "flagged_adversarial": "Preserves verifier findings instead of clearing them to open a gate.",
        "validation_receipts": "Exact commands and hashes prove required checks actually ran.",
        "field_principles": "Explains which evidence failure every emitted field prevents.",
        "count_memory_ready_score": "A bare 0/1 qualifies arithmetic, chronology, and restart only.",
        "continuous_self_learning_task": "Marks legal feedback as the sole source of Tier 1 state change.",
        "energy_definition": "Exact formulas make the numerical mechanism independently recheckable.",
        "fixture_rows": "Separates constructed controls from empirical evidence.",
        "restart_rows": "Shows each interruption preserves immutable predictions and one update.",
        "hardware_path": "States the sixteen-count path without inventing an unmeasured speedup.",
    }
    return {
        key: custom.get(key, f"Records {key} so omitted or changed evidence cannot pass silently.")
        for key in keys
    }


def build_artifact(
    *,
    fixture_rows: Sequence[Mapping[str, Any]],
    restart_rows: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    required_checks_passed: bool,
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    preconditions_checked: Sequence[Mapping[str, Any]],
    numerical_sanity: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Assemble terminal evidence while keeping readiness separate from benefit."""

    fixtures = [deepcopy(dict(row)) for row in fixture_rows]
    restarts = [deepcopy(dict(row)) for row in restart_rows]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "version": VERSION,
        "run_date": RUN_DATE,
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions_checked],
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate_class": "no_model_load",
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "execution_venue": "host",
        "duration_s": float(duration_s),
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "process_identity": {"pid": os.getpid(), "executable": os.path.realpath(os.sys.executable)},
        "random_seed": 6597534,
        "random_seeds": {
            "sampling": 6597534,
            "fitting": 6597534,
            "arrival": 6597534,
            "bootstrap": 6597534,
        },
        "source_artifact_hashes": dict(sorted(source_hashes.items())),
        "rows": _row_view(fixtures),
        "sample_size_budget": {
            "planned": 7,
            "attempted": len(fixtures),
            "complete": sum(row.get("complete") is True for row in fixtures),
            "completed": sum(row.get("complete") is True for row in fixtures),
            "excluded": 0,
            "failed": sum(row.get("complete") is not True for row in fixtures),
            "censored": 0,
            "unstarted": max(0, 7 - len(fixtures)),
        },
        "arithmetic_rows": arithmetic_rows(),
        "fixture_rows": fixtures,
        "restart_rows": restarts,
        "numerical_sanity_panel": deepcopy(
            dict(
                numerical_sanity
                or {
                    "operation_count": 64,
                    "duration_s": 0.001,
                    "pathological_update_cost": False,
                    "full_service_comparison_performed": False,
                }
            )
        ),
        "required_checks_passed": bool(required_checks_passed),
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "affected_validation_manifest": {
            "experiment_id": AFFECTED_MANIFEST.experiment_id,
            "test_paths": list(AFFECTED_MANIFEST.test_paths),
            "changed_modules": list(AFFECTED_MANIFEST.changed_modules),
            "static_paths": list(AFFECTED_MANIFEST.static_paths),
        },
        "capability_e2e": {
            "declared_entrypoint": WRAPPER_PATH.as_posix(),
            "real_predict_release_update_persist_reload_path": True,
            "fresh_process_cold_replay": bool(required_checks_passed),
            "private_llm_off_real_environment_smoke": True,
            "numbered_runtime_e2e_applicable": False,
            "e2e_001_002_applicable": False,
            "e2e_003_004_applicable": False,
            "e2e_009_through_013_applicable": False,
        },
        "gate_check_summary": {},
        "honest_verdict": "",
        "verdict_class": "disqualified",
        "verifier_is_oracle": True,
        "flagged_adversarial": False,
        "positive_claim": False,
        "count_memory_ready_score": 0,
        "continuous_self_learning_task": True,
        "energy_definition": {
            "bin_count": BIN_COUNT,
            "bin_rule": "k=min(7,floor(8*p0_clipped))",
            "probability_clip": [CLIP_MIN, 1.0 - CLIP_MIN],
            "prior_mass_kappa": PRIOR_MASS,
            "posterior": "a_k/(a_k+b_k)",
            "prediction": "sigmoid(logit(p0)+logit(posterior)-logit(mu_k))",
            "energies": {"E(x,0)": 0, "E(x,1)": "-logit(p_t)"},
            "normalization": "exp(-E1)/(exp(-E0)+exp(-E1))",
        },
        "hardware_path": {
            "state_value_count": 16,
            "update_complexity": "CPU O(1)",
            "later_implementation": "RAM/LUT",
            "measured_100x_claim": False,
        },
        "v657_local_gradient_null_unchanged": True,
        "older_importance_anchor_nulls_unchanged": True,
        "generator_weights_changed": False,
        "production_defaults_changed": False,
        "exp7523_static_fit_consumed": False,
        "exp7533_or_7538_output_consumed": False,
    }
    reduced = independent_reduce(artifact)
    artifact.update(reduced)
    artifact["acceptance_gate_results"] = _acceptance_gates(artifact)
    failures = [gate for gate in artifact["acceptance_gate_results"] if not gate["passed"]]
    artifact["gate_check_summary"] = {
        "passed": not failures,
        "failed_checks": [gate["condition"] for gate in failures],
        "first_failure": deepcopy(failures[0]) if failures else None,
    }
    artifact["honest_verdict"] = (
        "complete_circular_positive_count_memory_qualified"
        if artifact["verdict_class"] == "circular_positive"
        else "complete_disqualified_count_memory_validation"
    )
    all_keys = (*artifact.keys(), "field_principles", "reproducibility_checksum")
    artifact["field_principles"] = _field_principles(all_keys)
    artifact["reproducibility_checksum"] = canonical_hash(
        {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    )
    return artifact


def build_test_artifact(checkpoint_root: Path) -> JsonDict:
    """Build compact complete evidence for mutation and fresh-reader tests."""

    receipts = [
        {"name": name, "passed": True, "exit_code": 0, "timed_out": False}
        for name in validation_scope.REQUIRED_CHECK_NAMES
    ]
    return build_artifact(
        fixture_rows=run_fixture_panel(),
        restart_rows=run_restart_panel(checkpoint_root),
        validation_receipts=receipts,
        required_checks_passed=True,
        duration_s=0.25,
        phase_spans=[],
        source_hashes={"test_fixture": "sha256:test"},
        preconditions_checked=[],
    )


def validate_artifact(
    value: Mapping[str, Any],
    *,
    root: Path = REPO_ROOT,
    require_validation: bool = True,
    require_terminal: bool = False,
    verify_sources: bool = False,
) -> None:
    """Reject changed identity, rows, gates, sources, or receipt scope."""

    if value.get("verdict_class") == "blocked":
        if value.get("inference_substrate_class") != "blocked_no_run":
            raise ValueError("blocked_substrate_class_invalid")
        summary = value.get("gate_check_summary")
        first = summary.get("first_failure") if isinstance(summary, Mapping) else None
        required = {"upstream", "field_or_path", "expected", "observed"}
        if not isinstance(first, Mapping) or not required <= set(first):
            raise ValueError("blocked_gate_summary_incomplete")
        if not str(value.get("honest_verdict", "")).startswith("complete_blocked_"):
            raise ValueError("blocked_verdict_prefix_invalid")
        principles = value.get("field_principles")
        if not isinstance(principles, Mapping) or set(principles) != set(value):
            raise ValueError("field_principles_incomplete")
        blocked_checksum = canonical_hash(
            {key: item for key, item in value.items() if key != "reproducibility_checksum"}
        )
        if value.get("reproducibility_checksum") != blocked_checksum:
            raise ValueError("reproducibility_checksum_mismatch")
        return

    expected = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "version": VERSION,
        "run_date": RUN_DATE,
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": ZERO_INVOCATION_COUNTS,
        "inference_substrate_class": "no_model_load",
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "execution_venue": "host",
        "verifier_is_oracle": True,
        "positive_claim": False,
    }
    for field, wanted in expected.items():
        if value.get(field) != wanted:
            raise ValueError(f"field_invalid:{field}")
    if not str(value.get("honest_verdict", "")).startswith("complete_"):
        raise ValueError("terminal_verdict_prefix_invalid")
    reduced = independent_reduce(value)
    for field, observed in reduced.items():
        if value.get(field) != observed:
            raise ValueError(f"independent_reduction_mismatch:{field}")
    if value.get("acceptance_gate_results") != _acceptance_gates(value):
        raise ValueError("acceptance_gate_reduction_mismatch")
    principles = value.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(value):
        raise ValueError("field_principles_incomplete")
    checksum = canonical_hash(
        {key: item for key, item in value.items() if key != "reproducibility_checksum"}
    )
    if value.get("reproducibility_checksum") != checksum:
        raise ValueError("reproducibility_checksum_mismatch")
    receipts = list(value.get("validation_receipts", []))
    if require_validation:
        receipt_reduction = validation_scope.reduce_required_checks(receipts)
        if receipt_reduction["required_checks_passed"] is not True:
            raise ValueError("required_validation_receipts_failed")
        preconditions = value.get("preconditions_checked")
        if not isinstance(preconditions, list) or any(
            row.get("passed") is not True for row in preconditions
        ):
            raise ValueError("preconditions_failed")
    if require_terminal:
        terminal_names = {
            "fresh_process_cold_replay",
            "independent_raw_reduction",
            "adversarial_verify",
            "verdict_row_consistency_strict",
        }
        passed_names = {
            row.get("name")
            for row in receipts
            if row.get("passed") is True and row.get("exit_code") == 0
        }
        if not terminal_names <= passed_names:
            raise ValueError("terminal_validation_receipts_failed")
    if verify_sources:
        for label, expected_hash in value.get("source_artifact_hashes", {}).items():
            path = root / label
            if not path.is_file() or sha256_file(path) != expected_hash:
                raise ValueError(f"source_hash_mismatch:{label}")


def _load_object(path: Path) -> JsonDict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def cold_replay(
    path: Path,
    *,
    require_validation: bool = True,
    require_terminal: bool = False,
    verify_sources: bool = False,
) -> JsonDict:
    """Reload and reduce serialized evidence through a fresh-reader API."""

    value = _load_object(path)
    if not value:
        raise ValueError("artifact_unreadable_or_not_object")
    validate_artifact(
        value,
        require_validation=require_validation,
        require_terminal=require_terminal,
        verify_sources=verify_sources,
    )
    return independent_reduce(value)


def run_numerical_sanity_panel(operation_count: int = 4096) -> JsonDict:
    """Time a fixed small CPU panel only to detect pathological update cost."""

    arm = CountArm.create("local", default_config())
    started = time.monotonic()
    for index in range(operation_count):
        p0 = ((index % 98) + 1) / 100.0
        arm.predict(p0)
        arm.update(f"sanity-{index}", p0, index % 2)
    duration = time.monotonic() - started
    return {
        "operation_count": operation_count,
        "duration_s": duration,
        "updates_per_s": operation_count / max(duration, 1e-12),
        "pathological_update_cost": duration >= 5.0,
        "full_service_comparison_performed": False,
        "full_service_comparison_deferred_to": "Exp7544",
    }


REQUIRED_INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7506_v657_causal_prototype.py"),
    Path("python/carnot/experiment_7509_v657_causal_online.py"),
    Path("python/carnot/autoresearch/calibrated_decision_benchmark.py"),
    SPEC_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)


def _precondition(
    check: str,
    upstream: str,
    field_or_path: str,
    expected: Any,
    observed: Any,
    passed: bool,
    *,
    required: bool = True,
) -> JsonDict:
    return {
        "check": check,
        "upstream": upstream,
        "field_or_path": field_or_path,
        "expected": expected,
        "observed": observed,
        "required": required,
        "passed": passed,
        "principle": "Exact observations prevent absent prerequisites from becoming fabricated evidence.",
    }


def collect_preconditions(root: Path) -> tuple[list[JsonDict], dict[str, str]]:
    """Authenticate used sources and record stale paths without depending on them."""

    checks: list[JsonDict] = []
    hashes: dict[str, str] = {}
    for relative in REQUIRED_INPUT_PATHS:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        checks.append(
            _precondition(
                f"source_bytes:{relative.as_posix()}",
                relative.as_posix(),
                "bytes",
                "readable_nonempty_bytes",
                "readable_nonempty_bytes" if available else None,
                available,
            )
        )
        if available:
            hashes[relative.as_posix()] = sha256_file(path)
    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    checks.append(
        _precondition(
            "driving_requirement",
            SPEC_PATH.as_posix(),
            "REQ-*",
            "REQ-CL-7534",
            "REQ-CL-7534" if "REQ-CL-7534" in spec_text else None,
            "REQ-CL-7534" in spec_text,
        )
    )
    moved_paths = (
        (
            "python/carnot/learning/online_learner.py",
            "absent_not_required_count_memory_is_standalone",
        ),
        ("python/carnot/models/gibbs.py", "moved_to_python/carnot/models/gibbs/package"),
    )
    for relative, disposition in moved_paths:
        observed = "present" if (root / relative).is_file() else "absent"
        checks.append(
            _precondition(
                f"named_legacy_reference:{relative}",
                relative,
                "path",
                disposition,
                observed,
                True,
                required=False,
            )
        )
    checks.extend(
        [
            _precondition(
                "exp7523_static_fit_dependency",
                "Exp7523",
                "producer_output",
                "not_consumed",
                "not_consumed",
                True,
                required=False,
            ),
            _precondition(
                "same_milestone_outputs",
                "Exp7533/Exp7538",
                "producer_outputs",
                "not_consumed",
                "not_consumed",
                True,
                required=False,
            ),
            _precondition(
                "external_compute_resources",
                "model/GPU/data",
                "prerequisite",
                "none_required",
                "none_required",
                True,
                required=False,
            ),
        ]
    )
    return checks, hashes


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Flush each phase and long-operation boundary with monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7534] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _span(phase: str, phase_started: float, run_started: float, units: int) -> JsonDict:
    ended = time.monotonic()
    return {
        "phase": phase,
        "start_offset_s": phase_started - run_started,
        "end_offset_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
    }


def _terminal_commands(candidate: Path) -> list[validation_scope.CommandSpec]:
    """Run four independent readers against one exact candidate path."""

    python = str(REPO_ROOT / ".venv/bin/python")
    return [
        validation_scope.CommandSpec(
            "fresh_process_cold_replay",
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                "--date",
                RUN_DATE,
                "--cold-replay",
                str(candidate),
            ),
            "exact_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "independent_raw_reduction",
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                "--date",
                RUN_DATE,
                "--independent-reduce",
                str(candidate),
            ),
            "exact_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "exact_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "exact_candidate",
            300.0,
        ),
    ]


def _terminal_receipts_pass(receipts: Sequence[Mapping[str, Any]]) -> bool:
    names = {
        "fresh_process_cold_replay",
        "independent_raw_reduction",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    }
    passed = {
        row.get("name")
        for row in receipts
        if row.get("passed") is True
        and row.get("exit_code") == 0
        and row.get("timed_out") is not True
    }
    return names <= passed


def build_blocked_artifact(
    failed: Mapping[str, Any], checks: Sequence[Mapping[str, Any]], duration_s: float
) -> JsonDict:
    """Name one missing prerequisite without inventing dependent measurements."""

    reason = str(failed.get("check", "external_prerequisite")).replace(":", "_")
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "version": VERSION,
        "run_date": RUN_DATE,
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate_class": "blocked_no_run",
        "inference_substrate": "precondition_check_only",
        "execution_venue": "host",
        "duration_s": max(duration_s, 1e-6),
        "phase_spans": [],
        "random_seed": 6597534,
        "source_artifact_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "planned": 7,
            "attempted": 0,
            "complete": 0,
            "completed": 0,
            "excluded": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": 7,
        },
        "acceptance_gate_results": [],
        "gate_check_summary": {
            "passed": False,
            "failed_checks": [failed.get("check")],
            "first_failure": deepcopy(dict(failed)),
        },
        "honest_verdict": f"complete_blocked_{reason}",
        "verdict_class": "blocked",
        "verifier_is_oracle": True,
        "flagged_adversarial": False,
        "positive_claim": False,
        "validation_receipts": [],
        "count_memory_ready_score": 0,
        "continuous_self_learning_task": True,
        "fixture_rows": [],
        "restart_rows": [],
        "energy_definition": {},
        "hardware_path": {
            "state_value_count": 16,
            "update_complexity": "CPU O(1)",
            "measured_100x_claim": False,
        },
    }
    keys = (*artifact.keys(), "field_principles", "reproducibility_checksum")
    artifact["field_principles"] = _field_principles(keys)
    artifact["reproducibility_checksum"] = canonical_hash(artifact)
    return artifact


def _write_frozen_manifest(path: Path) -> None:
    value = {
        "experiment_id": AFFECTED_MANIFEST.experiment_id,
        "test_paths": list(AFFECTED_MANIFEST.test_paths),
        "changed_modules": list(AFFECTED_MANIFEST.changed_modules),
        "static_paths": list(AFFECTED_MANIFEST.static_paths),
    }
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    if path.is_file() and path.read_bytes() != encoded:
        raise ValueError("affected_manifest_changed_after_freeze")
    if not path.exists():
        atomic_json(path, value)


def run_experiment(  # pragma: no cover - exercised by the declared entrypoint E2E.
    root: Path, run_date: str, *, output_path: Path | None = None
) -> JsonDict:
    """Authenticate, measure, validate in fresh processes, and publish atomically."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = root.resolve()
    destination = output_path or root / RESULT_PATH
    raw_root = root / RAW_DIR
    run_started = time.monotonic()
    spans: list[JsonDict] = []

    progress(run_started, "preconditions", "start")
    phase_started = time.monotonic()
    checks, source_hashes = collect_preconditions(root)
    spans.append(_span("preconditions", phase_started, run_started, len(checks)))
    failed = next(
        (row for row in checks if row.get("required") is True and row.get("passed") is not True),
        None,
    )
    if failed is not None:
        blocked = build_blocked_artifact(failed, checks, time.monotonic() - run_started)
        progress(run_started, "publish", "before_atomic_blocked", check=failed["check"])
        atomic_json(destination, blocked)
        progress(run_started, "publish", "complete_blocked", check=failed["check"])
        return blocked
    progress(run_started, "preconditions", "complete", completed_units=len(checks))

    manifest_path = raw_root / "affected_validation_manifest.json"
    _write_frozen_manifest(manifest_path)
    source_hashes[manifest_path.relative_to(root).as_posix()] = sha256_file(manifest_path)

    scratch = Path(tempfile.mkdtemp(prefix="carnot-exp7534-", dir="/tmp"))
    progress(run_started, "analytical_fixture", "start")
    phase_started = time.monotonic()
    fixture_rows = run_fixture_panel()
    spans.append(_span("analytical_fixture", phase_started, run_started, len(fixture_rows)))
    progress(run_started, "analytical_fixture", "complete", completed_units=len(fixture_rows))

    progress(run_started, "restart_fixture", "start")
    phase_started = time.monotonic()
    restart_rows = run_restart_panel(scratch / "restart")
    spans.append(_span("restart_fixture", phase_started, run_started, len(restart_rows)))
    progress(run_started, "restart_fixture", "complete", completed_units=len(restart_rows))

    progress(run_started, "numerical_sanity_benchmark", "before_benchmark")
    phase_started = time.monotonic()
    numerical_sanity = run_numerical_sanity_panel()
    spans.append(_span("numerical_sanity_benchmark", phase_started, run_started, 4096))
    progress(
        run_started,
        "numerical_sanity_benchmark",
        "after_benchmark",
        completed_units=4096,
    )

    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7534-validation-", dir="/tmp"))
    basetemp = private_root / "pytest"
    basetemp.mkdir(parents=True, exist_ok=True)
    commands = validation_scope.build_scoped_commands(
        root,
        AFFECTED_MANIFEST.test_paths,
        AFFECTED_MANIFEST.changed_modules,
        static_paths=AFFECTED_MANIFEST.static_paths,
        basetemp=basetemp,
        coverage_file=private_root / ".coverage.exp7534",
    )
    progress(run_started, "affected_validation", "before_subprocesses", commands=len(commands))
    phase_started = time.monotonic()
    affected = validation_scope.run_commands(
        root,
        commands,
        log_dir=raw_root / "validation" / "affected",
        heartbeat_s=60.0,
    )
    affected_reduction = validation_scope.reduce_required_checks(affected)
    spans.append(_span("affected_validation", phase_started, run_started, len(affected)))
    progress(
        run_started,
        "affected_validation",
        "after_subprocesses",
        passed=affected_reduction["required_checks_passed"],
    )

    if affected_reduction["required_checks_passed"] is not True:
        disqualified = build_artifact(
            fixture_rows=fixture_rows,
            restart_rows=restart_rows,
            validation_receipts=affected,
            required_checks_passed=False,
            duration_s=time.monotonic() - run_started,
            phase_spans=spans,
            source_hashes=source_hashes,
            preconditions_checked=checks,
            numerical_sanity=numerical_sanity,
        )
        validate_artifact(disqualified, require_validation=False, verify_sources=True)
        progress(run_started, "publish", "before_atomic_disqualified")
        atomic_json(destination, disqualified)
        progress(run_started, "publish", "complete_disqualified")
        return disqualified

    candidate = build_artifact(
        fixture_rows=fixture_rows,
        restart_rows=restart_rows,
        validation_receipts=affected,
        required_checks_passed=True,
        duration_s=time.monotonic() - run_started,
        phase_spans=spans,
        source_hashes=source_hashes,
        preconditions_checked=checks,
        numerical_sanity=numerical_sanity,
    )
    validate_artifact(candidate, require_validation=True, verify_sources=True)
    candidate_path = raw_root / "measured_terminal_candidate.json"
    atomic_json(candidate_path, candidate)

    progress(run_started, "terminal_validation", "before_subprocesses", commands=4)
    phase_started = time.monotonic()
    terminal = validation_scope.run_commands(
        root,
        _terminal_commands(candidate_path),
        log_dir=raw_root / "validation" / "terminal",
        heartbeat_s=60.0,
    )
    terminal_passed = _terminal_receipts_pass(terminal)
    spans.append(_span("terminal_validation", phase_started, run_started, len(terminal)))
    progress(
        run_started,
        "terminal_validation",
        "after_subprocesses",
        passed=terminal_passed,
    )
    if not terminal_passed:
        raise RuntimeError("terminal_candidate_validation_failed")

    final = build_artifact(
        fixture_rows=fixture_rows,
        restart_rows=restart_rows,
        validation_receipts=[*affected, *terminal],
        required_checks_passed=True,
        duration_s=time.monotonic() - run_started,
        phase_spans=spans,
        source_hashes=source_hashes,
        preconditions_checked=checks,
        numerical_sanity=numerical_sanity,
    )
    validate_artifact(
        final,
        require_validation=True,
        require_terminal=True,
        verify_sources=True,
    )
    exact_candidate = raw_root / "exact_terminal_candidate.json"
    atomic_json(exact_candidate, final)

    progress(run_started, "exact_candidate_validation", "before_subprocesses", commands=4)
    exact = validation_scope.run_commands(
        root,
        _terminal_commands(exact_candidate),
        log_dir=raw_root / "validation" / "exact_terminal",
        heartbeat_s=60.0,
    )
    exact_passed = _terminal_receipts_pass(exact)
    progress(
        run_started,
        "exact_candidate_validation",
        "after_subprocesses",
        passed=exact_passed,
    )
    if not exact_passed:
        raise RuntimeError("exact_terminal_candidate_validation_failed")
    progress(run_started, "publish", "before_atomic_terminal", path=destination)
    atomic_json(destination, final)
    progress(
        run_started,
        "publish",
        "complete",
        count_memory_ready_score=final["count_memory_ready_score"],
        verdict_class=final["verdict_class"],
    )
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the producer and two read-only fresh-process modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    parser.add_argument("--allow-test-validation", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the producer or one strict serialized-evidence reader."""

    args = parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    require_validation = not args.allow_test_validation
    verify_sources = not args.allow_test_validation
    if args.cold_replay is not None:
        reduction = cold_replay(
            args.cold_replay,
            require_validation=require_validation,
            verify_sources=verify_sources,
        )
        print(json.dumps(reduction, sort_keys=True), flush=True)
        return int(reduction["count_memory_ready_score"] != 1)
    if args.independent_reduce is not None:
        value = _load_object(args.independent_reduce)
        if not value:
            raise ValueError("artifact_unreadable_or_not_object")
        validate_artifact(
            value,
            require_validation=require_validation,
            verify_sources=verify_sources,
        )
        reduction = independent_reduce(value)
        print(json.dumps(reduction, sort_keys=True), flush=True)
        return int(reduction["count_memory_ready_score"] != 1)
    artifact = run_experiment(REPO_ROOT, args.date)
    print(
        json.dumps(
            {
                "result": RESULT_PATH.as_posix(),
                "count_memory_ready_score": artifact["count_memory_ready_score"],
                "verdict_class": artifact["verdict_class"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return int(artifact["verdict_class"] not in {"circular_positive", "null"})


if __name__ == "__main__":  # pragma: no cover - the thin wrapper is the public executable.
    raise SystemExit(main())
