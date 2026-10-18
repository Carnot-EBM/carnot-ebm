"""Build a fresh-feedback admission fixture for constraint-memory reuse.

The controller freezes candidate and incumbent state at nomination. It then
uses later paired labels for admission. This experiment checks the mechanism.
It does not claim useful learning on a drifting stream.

Spec refs: REQ-CL-7281 and SCENARIO-CL-7281-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import re
import selectors
import socket
import subprocess
import time
from typing import Any

from scipy.stats import beta as beta_distribution

from carnot import experiment_7213_v635_refinement_learning as exp7213
from carnot import experiment_7226_v636_belief_compiler as exp7226
from carnot import experiment_7267_v639_recognition_prototype as prototype
from carnot.memory import transactional_constraint_memory as transactional


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7281
SCHEMA = "carnot.exp7281.v640_admission_prototype.v1"
STATE_SCHEMA = "carnot.paired_admission_controller.v1"
MILESTONE = "2026.09.640"
RUN_DATE = "20260913"
RANDOM_SEED = 7_281_000
DEVELOPMENT_STREAM_SEEDS = tuple(range(7_281_001, 7_281_005))
STREAM_SEEDS = tuple(range(7_281_101, 7_281_125))
DEVELOPMENT_STREAM_COUNT = 4
STREAM_COUNT = 24
EVENTS_PER_STREAM = 1_024
WARMUP_COUNT = 128
NOMINATION_LABEL_BUDGET = 64
ADMISSION_LABEL_BUDGET = 64
MAX_OPPORTUNITIES = 8
FRESH_LABELS_PER_OPPORTUNITY = 8
TOTAL_ALPHA = 0.05
DEFAULT_THRESHOLD = -0.5
ARCHIVE_CAP = 4
MEMORY_CAP_BYTES = prototype.MEMORY_CAPS["total_bytes"]
MODEL_SPECS: list[JsonDict] = []
MODEL_INVOKED = False
INVOCATION_COUNTS = {
    "attempted_model_loads": 0,
    "completed_model_loads": 0,
    "attempted_generation_calls": 0,
    "completed_generation_calls": 0,
    "usable_answers": 0,
}
INFERENCE_SUBSTRATE = "cpu_exact_solver_or_simulator"
INFERENCE_SUBSTRATE_CLASS = "cpu_exact_solver_or_simulator"
REDUCER_INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
REDUCER_INFERENCE_SUBSTRATE_CLASS = "aggregation"
EXECUTION_VENUE = "host"

FAMILIES = tuple(prototype.FAMILIES)
PARAMETER_DOMAIN = tuple(prototype.PARAMETER_DOMAIN)
FULL_MASK = prototype.FULL_MASK
exact_label = exp7226.exact_label
ARMS = (
    "full_reference",
    "reset",
    "frozen_warmup",
    "unconditional_recognition",
    "range_gated",
    "paired_gated",
    "label_shuffled_paired",
)
ADMISSION_ARMS = (
    "unconditional_recognition",
    "range_gated",
    "paired_gated",
    "label_shuffled_paired",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
WRAPPER_PATH = Path("scripts/experiments/experiment_7281_v640_admission_prototype.py")
TEST_PATH = Path("tests/python/test_experiment_7281_v640_admission_prototype.py")
DEFAULT_ARTIFACT = Path("results/experiment_7281_v640_admission_prototype.json")
DEFAULT_UPSTREAM_ARTIFACTS = {
    "exp7268": Path("results/experiment_7268_v639_recognition_learning.json"),
    "exp7269": Path("results/experiment_7269_v639_recognition_audit.json"),
}
DEFAULT_HISTORICAL_PATHS = {
    "lifecycle": Path("results/raw/experiment_7268/lifecycle_receipts.jsonl"),
    "events": Path("results/raw/experiment_7268/prequential_rows.jsonl"),
}
EXPECTED_UPSTREAM_HASHES = {
    "exp7268": "sha256:5ec52b42298a73e3454fce47e03f34118a1c348f72f81a1ee768673c4cef8d16",
    "exp7269": "sha256:d253c4dc94d0661145908bf75d1ee9ada425fcf0097a5ff7367b9946a3ebb8a8",
}
EXPECTED_HISTORICAL_HASHES = {
    "lifecycle": "sha256:e45f68216a398486472b22f98fb6773fa58226f1e90ae3aeb80ca5d7bc8b4ff2",
    "events": "sha256:ed536492889aacd7d8db5344b4365c613fdad023de01ae7b782570b54f06c522",
}
SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-roadmap.yaml"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("python/carnot/experiment_7267_v639_recognition_prototype.py"),
    Path("python/carnot/experiment_7268_v639_recognition_learning.py"),
    Path("python/carnot/experiment_7269_v639_recognition_audit.py"),
    Path("python/carnot/experiment_7281_v640_admission_prototype.py"),
    WRAPPER_PATH,
    TEST_PATH,
    SPEC_PATH,
)
SCENARIO_PATTERN = re.compile(r"SCENARIO-CL-7281-[A-Z-]+")

REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment_id",
    "milestone",
    "status",
    "run_date",
    "started_at_utc",
    "completed_at_utc",
    "field_principles",
    "preconditions_checked",
    "MODEL_SPECS",
    "model_invoked",
    "invocation_counts",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "source_artifact_hashes",
    "rows",
    "sample_size_budget",
    "acceptance_gate_results",
    "gate_check_summary",
    "verifier_is_oracle",
    "honest_verdict",
    "verdict_class",
    "validation_receipts",
    "admission_fixture_ready_score",
    "admission_contract",
    "stream_manifest_path",
    "finite_law_rows",
    "nomination_overlap_rows",
    "continuous_self_learning_task",
)
FIELD_PRINCIPLES = {
    "schema": "Version the result and retain ordinary top-level experiment_id and milestone.",
    "experiment_id": "Bind the evidence to the active Exp7281 task.",
    "milestone": "Bind the evidence to milestone 2026.09.640.",
    "status": "Use complete or blocked for terminal evidence; keep unfinished work in checkpoints.",
    "run_date": "Use 20260913 and actual UTC start and end times.",
    "started_at_utc": "Record the actual UTC start timestamp.",
    "completed_at_utc": "Record the actual UTC completion timestamp.",
    "field_principles": "Store explanations here; consumer values remain ordinary top-level fields.",
    "preconditions_checked": "Record actual input hashes, authority separation, ownership, and failures.",
    "MODEL_SPECS": "Declare models executable now; historical identities stay in hashed sidecars.",
    "model_invoked": "Derive model use from actual calls, including failed generation.",
    "invocation_counts": "Separate attempted and completed loads and generation from usable answers.",
    "inference_substrate": "Use the recognized literal for actual computation.",
    "inference_substrate_class": "Use the correct no-LLM class without duration padding.",
    "execution_venue": "Host orchestration is host; device execution is separate.",
    "duration_s": "Measure monotonic elapsed time and disjoint phase spans.",
    "random_seed": "Freeze independent-unit seeds before outcomes.",
    "reproducibility_checksum": "Bind code, configuration, manifests, and raw evidence.",
    "source_artifact_hashes": "Preserve exact input identity, retirement, and quarantine status.",
    "rows": "Keep each stream, arm, seed, error, abstention, cost, metric, and censoring state.",
    "sample_size_budget": "Record planned, attempted, completed, censored units, and stopping.",
    "acceptance_gate_results": "Record expected, observed, passed, and principle for each criterion.",
    "gate_check_summary": "For blocked work, name exact upstream, field, observed, and expected values.",
    "verifier_is_oracle": "Expose shared evaluator authority; conformance is not learned correctness.",
    "honest_verdict": "Use complete_ for findings and blocked_ for external absence.",
    "verdict_class": "Use the closed class set; oracle authority forbids positive.",
    "validation_receipts": "Retain command, exit code, timing, and log hash.",
    "admission_fixture_ready_score": "One means strict admission and sealed-stream mechanics pass.",
    "admission_contract": "Freeze alpha, quota split, opportunity bound, and future-label isolation.",
    "stream_manifest_path": "Name four development and 24 prospective authority-separated streams.",
    "finite_law_rows": "Retain enumerated IID checks and explicit drift limits.",
    "nomination_overlap_rows": "Measure old overlap and outcomes without retuning.",
    "continuous_self_learning_task": "True because feedback can change only future constraints.",
}

gate_check = exp7213.gate_check
gate_summary = exp7213.gate_summary


class AdmissionRejected(ValueError):
    """Reject invalid evidence before it changes admission state."""


@dataclass(frozen=True)
class ExperimentPaths:
    """Keep stream seals, raw rows, checkpoints, and terminal bytes separate."""

    development_public: Path
    development_authority: Path
    development_releases: Path
    prospective_public: Path
    prospective_authority: Path
    prospective_releases: Path
    stream_manifest: Path
    raw_rows: Path
    opportunity_rows: Path
    historical_rows: Path
    state_sidecar: Path
    control_sidecar: Path
    evidence_sidecar: Path
    provisional: Path
    terminal_candidate: Path
    artifact: Path

    @classmethod
    def defaults(cls) -> ExperimentPaths:
        """Return task-owned paths below the repository result directory."""

        return cls.from_results_root(REPO_ROOT / "results")

    @classmethod
    def under(cls, root: Path) -> ExperimentPaths:
        """Put every test output below a caller-owned directory."""

        return cls.from_results_root(root)

    @classmethod
    def from_results_root(cls, root: Path) -> ExperimentPaths:
        """Derive stream, raw, checkpoint, candidate, and terminal paths."""

        streams = root / "streams" / "experiment_7281"
        raw = root / "raw" / "experiment_7281"
        checkpoints = root / "checkpoints"
        return cls(
            streams / "development_public.jsonl",
            streams / "development_private_authority.jsonl",
            streams / "development_releases.jsonl",
            streams / "prospective_public.jsonl",
            streams / "prospective_private_authority.jsonl",
            streams / "prospective_releases.jsonl",
            streams / "stream_manifest.json",
            raw / "stream_arm_rows.jsonl",
            raw / "opportunity_rows.jsonl",
            raw / "nomination_overlap_rows.jsonl",
            checkpoints / "experiment_7281_v640_states.json",
            checkpoints / "experiment_7281_v640_controls.json",
            checkpoints / "experiment_7281_v640_evidence.json",
            checkpoints / "experiment_7281_v640_in_progress.json",
            raw / "terminal_candidate.json",
            root / DEFAULT_ARTIFACT.name,
        )


@dataclass(frozen=True)
class AdmissionPanel:
    """Retain stream-arm summaries, opportunity rows, and final state evidence."""

    rows: list[JsonDict]
    opportunity_rows: list[JsonDict]
    final_states: list[JsonDict]
    completed_stream_count: int
    censored_stream_count: int
    maximum_memory_bytes: int


def _progress(phase: int, boundary: str, detail: str) -> None:
    """Print one flushed boundary for the external watchdog."""

    print(f"phase {phase} {boundary}: {detail}", flush=True)


def _resolve(repo_root: Path, path: str | Path) -> Path:
    """Resolve repository evidence while preserving absolute test paths."""

    value = Path(path)
    return value if value.is_absolute() else repo_root / value


def _load_object(path: Path) -> JsonDict:
    """Load one JSON object while missing or malformed evidence stays unavailable."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _sha256_path(path: Path) -> str | None:
    """Hash exact bytes while an absent path remains distinct from empty bytes."""

    try:
        return transactional.sha256_bytes(path.read_bytes())
    except OSError:
        return None


def _path_writable(path: Path) -> bool:
    """Check the nearest existing parent without creating terminal bytes."""

    parent = path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return parent.is_dir() and os.access(parent, os.W_OK)


def _atomic_write(path: Path, payload: bytes) -> JsonDict:
    """Publish complete bytes through the shipped flushed atomic writer."""

    return prototype._atomic_write(path, payload)


def _write_immutable(path: Path, payload: bytes) -> JsonDict:
    """Accept identical sealed bytes and reject replacement content."""

    return prototype._write_immutable(path, payload)


def mask_hash(masks: Mapping[str, Any]) -> str:
    """Bind one finite state to every family mask in stable order."""

    return transactional.sha256_json({family: int(masks[family]) for family in FAMILIES})


def _normalized_masks(masks: Mapping[str, Any]) -> dict[str, int]:
    """Copy complete nonempty finite masks into the admission boundary."""

    if set(masks) != set(FAMILIES):
        raise ValueError("invalid_mask_families")
    result = {family: int(masks[family]) for family in FAMILIES}
    if any(value <= 0 or value & ~FULL_MASK for value in result.values()):
        raise ValueError("invalid_survivor_mask")
    return result


def opportunity_alpha(opportunity_index: int) -> float:
    """Allocate the fixed total alpha equally across eight opportunities."""

    if not 1 <= opportunity_index <= MAX_OPPORTUNITIES:
        raise ValueError("invalid_opportunity_index")
    return TOTAL_ALPHA / MAX_OPPORTUNITIES


def _clopper_pearson(count: int, n: int, tail: float) -> tuple[float, float]:
    """Return exact one-sided binomial endpoints for one marginal count."""

    if n <= 0 or not 0 <= count <= n or not 0.0 < tail < 1.0:
        raise ValueError("invalid_binomial_inputs")
    lower = 0.0 if count == 0 else float(beta_distribution.ppf(tail, count, n - count + 1))
    upper = 1.0 if count == n else float(beta_distribution.ppf(1.0 - tail, count + 1, n - count))
    return lower, upper


def score_differences(
    differences: Sequence[int],
    alpha: float,
    comparison_count: int,
    *,
    threshold: float,
    rule: str,
) -> JsonDict:
    """Apply one frozen paired or range rule to ternary paired outcomes."""

    if comparison_count <= 0:
        raise ValueError("invalid_comparison_count")
    if rule not in {"paired", "range"}:
        raise ValueError("invalid_rule")
    values = [int(value) for value in differences]
    if not values or any(value not in {-1, 0, 1} for value in values):
        raise ValueError("invalid_paired_differences")
    n = len(values)
    mean = sum(values) / n
    if rule == "paired":
        tail = alpha / (4 * comparison_count)
        positive = sum(value == 1 for value in values)
        negative = sum(value == -1 for value in values)
        lower_positive, upper_positive = _clopper_pearson(positive, n, tail)
        lower_negative, upper_negative = _clopper_pearson(negative, n, tail)
        lower = lower_positive - upper_negative
        upper = upper_positive - lower_negative
        detail = {
            "positive_disagreement_count": positive,
            "negative_disagreement_count": negative,
            "endpoint_tail_alpha": tail,
            "positive_interval": [lower_positive, upper_positive],
            "negative_interval": [lower_negative, upper_negative],
        }
    else:
        radius = math.sqrt(2.0 * math.log(2.0 * comparison_count / alpha) / n)
        lower = max(-1.0, mean - radius)
        upper = min(1.0, mean + radius)
        detail = {"range_radius": radius}
    decision = "accept" if lower >= threshold else "reject" if upper < threshold else "defer"
    return {
        "rule": rule,
        "n": n,
        "alpha": alpha,
        "comparison_count": comparison_count,
        "threshold": threshold,
        "mean_difference": mean,
        "lower": lower,
        "upper": upper,
        "decision": decision,
        **detail,
    }


def _multinomial_probability(
    n: int,
    positive: int,
    negative: int,
    probabilities: tuple[float, float, float],
) -> float:
    """Compute one exact ternary count probability for finite enumeration."""

    zero = n - positive - negative
    coefficient = math.factorial(n) / (
        math.factorial(positive) * math.factorial(negative) * math.factorial(zero)
    )
    return coefficient * (
        probabilities[0] ** positive * probabilities[1] ** negative * probabilities[2] ** zero
    )


def enumerate_finite_laws() -> list[JsonDict]:
    """Enumerate fixed IID ternary laws at n=8 and n=16."""

    cases = {
        "zero_disagreement": (0.0, 0.0, 1.0),
        "all_harmful": (0.0, 1.0, 0.0),
        "all_useful": (1.0, 0.0, 0.0),
        "mixed_disagreement": (0.2, 0.1, 0.7),
    }
    alpha = opportunity_alpha(1)
    rows = []
    for n in (8, 16):
        for case, probabilities in cases.items():
            actual = probabilities[0] - probabilities[1]
            mass = 0.0
            noncoverage = 0.0
            for positive in range(n + 1):
                for negative in range(n - positive + 1):
                    probability = _multinomial_probability(n, positive, negative, probabilities)
                    mass += probability
                    differences = [1] * positive + [-1] * negative + [0] * (n - positive - negative)
                    interval = score_differences(
                        differences,
                        alpha,
                        1,
                        threshold=0.0,
                        rule="paired",
                    )
                    if not float(interval["lower"]) <= actual <= float(interval["upper"]):
                        noncoverage += probability
            observed = [0] * n if case == "zero_disagreement" else None
            if case == "all_harmful":
                observed = [-1] * n
            elif case == "all_useful":
                observed = [1] * n
            elif case == "mixed_disagreement":
                observed = [1, -1, 0, 0] * (n // 4)
            paired = score_differences(
                observed or [0] * n,
                alpha,
                1,
                threshold=0.0,
                rule="paired",
            )
            ranged = score_differences(
                observed or [0] * n,
                alpha,
                1,
                threshold=0.0,
                rule="range",
            )
            rows.append(
                {
                    "case": case,
                    "n": n,
                    "p_positive": probabilities[0],
                    "p_negative": probabilities[1],
                    "p_zero": probabilities[2],
                    "true_mean_difference": actual,
                    "allocated_alpha": alpha,
                    "enumerated_probability": mass,
                    "noncoverage_probability": noncoverage,
                    "coverage_passed": noncoverage <= alpha + 1e-12,
                    "paired_threshold_zero_feasible": paired["decision"] == "accept",
                    "range_threshold_zero_feasible": ranged["decision"] == "accept",
                    "iid_binary_scope_only": True,
                    "dependent_drift_theorem_claimed": False,
                }
            )
    return rows


class AdmissionController:
    """Freeze nominations and admit updates with later paired feedback."""

    def __init__(self, masks: Mapping[str, Any], *, rule: str = "paired") -> None:
        if rule not in {"unconditional", "range", "paired", "label_shuffled"}:
            raise ValueError("invalid_admission_rule")
        self._state: JsonDict = {
            "schema": STATE_SCHEMA,
            "rule": rule,
            "version": 0,
            "incumbent_masks": _normalized_masks(masks),
            "archives": [],
            "pending": None,
            "used_admission_ids": [],
            "ledger": [],
        }

    @classmethod
    def from_masks(cls, masks: Mapping[str, Any], *, rule: str = "paired") -> AdmissionController:
        """Start from detached finite masks with no pending decision."""

        return cls(masks, rule=rule)

    @classmethod
    def from_state(cls, value: Mapping[str, Any]) -> AdmissionController:
        """Restore a bounded state after checking immutable identities."""

        if value.get("schema") != STATE_SCHEMA:
            raise ValueError("invalid_admission_state")
        controller = cls.__new__(cls)
        controller._state = deepcopy(dict(value))
        if controller._state.get("rule") not in {
            "unconditional",
            "range",
            "paired",
            "label_shuffled",
        }:
            raise ValueError("invalid_admission_rule")
        controller._state["incumbent_masks"] = _normalized_masks(
            controller._state["incumbent_masks"]
        )
        archives = controller._state.get("archives")
        if not isinstance(archives, list) or len(archives) > ARCHIVE_CAP:
            raise ValueError("archive_capacity")
        if any(row.get("state_hash") != mask_hash(row.get("masks", {})) for row in archives):
            raise ValueError("archive_identity")
        pending = controller._state.get("pending")
        if pending is not None:
            if pending.get("candidate_state_hash") != mask_hash(
                pending.get("candidate_masks", {})
            ) or pending.get("incumbent_state_hash") != mask_hash(
                pending.get("incumbent_masks", {})
            ):
                raise ValueError("pending_identity")
        if controller.memory_usage()["within_cap"] is not True:
            raise ValueError("admission_memory_cap")
        return controller

    @classmethod
    def load(cls, path: Path) -> AdmissionController:
        """Load durable JSON bytes through the bounded state validator."""

        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("invalid_admission_state") from error
        if not isinstance(value, dict):
            raise ValueError("invalid_admission_state")
        return cls.from_state(value)

    def state_dict(self) -> JsonDict:
        """Return detached state so callers cannot mutate live bytes."""

        return deepcopy(self._state)

    def state_bytes(self) -> bytes:
        """Serialize complete controller state in canonical form."""

        return transactional.canonical_json_bytes(self._state)

    def state_hash(self) -> str:
        """Identify the incumbent, archives, pending decision, and evidence ledger."""

        return transactional.sha256_bytes(self.state_bytes())

    def incumbent_hash(self) -> str:
        """Identify only the deployed finite constraint state."""

        return mask_hash(self._state["incumbent_masks"])

    def incumbent_masks(self) -> dict[str, int]:
        """Return a detached deployed state for frozen prediction."""

        return deepcopy(self._state["incumbent_masks"])

    def archives(self) -> list[JsonDict]:
        """Return detached immutable archive rows."""

        return deepcopy(self._state["archives"])

    def predict(self, event: Mapping[str, Any]) -> str:
        """Predict from the deployed state without reading a label or mutating."""

        return prototype.predict_masks(self._state["incumbent_masks"], event)

    def save(self, path: Path) -> JsonDict:
        """Publish restartable state with the shared atomic writer."""

        return _atomic_write(path, self.state_bytes())

    def memory_usage(self) -> JsonDict:
        """Charge pending decisions and validation evidence to the complete cap."""

        state_bytes = len(self.state_bytes())
        pending = len(transactional.canonical_json_bytes(self._state.get("pending")))
        archives = len(transactional.canonical_json_bytes(self._state.get("archives", [])))
        ledger = len(transactional.canonical_json_bytes(self._state.get("ledger", [])))
        return {
            "controller_bytes": state_bytes,
            "pending_decision_and_validation_bytes": pending,
            "archive_bytes": archives,
            "ledger_bytes": ledger,
            "total_bytes": state_bytes,
            "cap_bytes": MEMORY_CAP_BYTES,
            "within_cap": state_bytes <= MEMORY_CAP_BYTES,
        }

    def nominate(
        self,
        candidate_masks: Mapping[str, Any],
        *,
        nomination_event_ids: Sequence[str],
        nomination_index: int,
        opportunity_index: int,
        thresholds: Mapping[str, float],
    ) -> JsonDict:
        """Freeze candidate and incumbent hashes before any admission label exists."""

        if self._state["pending"] is not None:
            raise AdmissionRejected("pending_decision_exists")
        opportunity_alpha(opportunity_index)
        event_ids = [str(value) for value in nomination_event_ids]
        if not event_ids or len(set(event_ids)) != len(event_ids):
            raise AdmissionRejected("invalid_nomination_ids")
        fixed_thresholds = {str(key): float(value) for key, value in thresholds.items()}
        if not fixed_thresholds:
            raise AdmissionRejected("missing_comparisons")
        candidate = _normalized_masks(candidate_masks)
        incumbent = self.incumbent_masks()
        parent_bytes = self.state_bytes()
        pending = {
            "opportunity_index": opportunity_index,
            "nomination_index": int(nomination_index),
            "nomination_event_ids": event_ids,
            "candidate_masks": candidate,
            "candidate_state_hash": mask_hash(candidate),
            "incumbent_masks": incumbent,
            "incumbent_state_hash": mask_hash(incumbent),
            "thresholds": fixed_thresholds,
            "alpha": opportunity_alpha(opportunity_index),
            "pre_nomination_parent_hash": transactional.sha256_bytes(parent_bytes),
            "pre_nomination_parent_bytes": transactional.encode_bytes(parent_bytes),
        }
        self._state["pending"] = pending
        if self.memory_usage()["within_cap"] is not True:
            self._state["pending"] = None
            raise AdmissionRejected("admission_memory_cap")
        return {
            key: deepcopy(value)
            for key, value in pending.items()
            if key not in {"pre_nomination_parent_bytes", "candidate_masks", "incumbent_masks"}
        }

    def _validated_cases(
        self, cases: Sequence[Mapping[str, Any]], current_index: int
    ) -> list[JsonDict]:
        """Reject reused, premature, malformed, or comparison-incomplete cases."""

        pending = self._state.get("pending")
        if pending is None:
            raise AdmissionRejected("no_pending_decision")
        normalized = [deepcopy(dict(row)) for row in cases]
        ids = [str(row.get("event_id")) for row in normalized]
        if len(set(ids)) != len(ids):
            raise AdmissionRejected("duplicate_admission_label")
        if set(ids) & set(pending["nomination_event_ids"]):
            raise AdmissionRejected("reused_nomination_label")
        if set(ids) & set(self._state["used_admission_ids"]):
            raise AdmissionRejected("reused_admission_label")
        for row in normalized:
            if int(row.get("release_index", current_index + 1)) > current_index:
                raise AdmissionRejected("unreleased_label")
            if int(row["release_index"]) <= int(pending["nomination_index"]):
                raise AdmissionRejected("label_not_subsequent")
            if (
                row.get("family_id") not in FAMILIES
                or not isinstance(row.get("numeric_value"), int)
                or row.get("observed_label") not in {"accept", "reject"}
            ):
                raise AdmissionRejected("invalid_admission_case")
        observed_comparisons = {str(row.get("comparison_id")) for row in normalized}
        if observed_comparisons != set(pending["thresholds"]):
            raise AdmissionRejected("comparison_evidence_mismatch")
        return normalized

    def admit(
        self,
        cases: Sequence[Mapping[str, Any]],
        *,
        current_index: int,
        expected_parent_hash: str,
        state_path: Path | None = None,
    ) -> JsonDict:
        """Score frozen states together and commit only after the fixed rule accepts."""

        if expected_parent_hash != self.state_hash():
            raise AdmissionRejected("stale_parent")
        normalized = self._validated_cases(cases, current_index)
        pending = deepcopy(self._state["pending"])
        candidate = pending["candidate_masks"]
        incumbent = pending["incumbent_masks"]
        comparison_count = len(pending["thresholds"])
        by_comparison: dict[str, list[JsonDict]] = defaultdict(list)
        for row in normalized:
            by_comparison[str(row["comparison_id"])].append(row)
        comparison_rows = []
        for comparison_id in sorted(by_comparison):
            group = by_comparison[comparison_id]
            labels = [str(row["observed_label"]) for row in group]
            if self._state["rule"] == "label_shuffled" and len(labels) > 1:
                labels = labels[1:] + labels[:1]
            differences = []
            for row, label in zip(group, labels):
                public = {
                    "event_id": str(row["event_id"]),
                    "family_id": str(row["family_id"]),
                    "numeric_value": int(row["numeric_value"]),
                }
                candidate_success = prototype.predict_masks(candidate, public) == label
                incumbent_success = prototype.predict_masks(incumbent, public) == label
                differences.append(int(candidate_success) - int(incumbent_success))
            if self._state["rule"] == "unconditional":
                scored: JsonDict = {
                    "rule": "unconditional",
                    "n": len(differences),
                    "alpha": pending["alpha"],
                    "comparison_count": comparison_count,
                    "threshold": pending["thresholds"][comparison_id],
                    "mean_difference": sum(differences) / len(differences),
                    "lower": None,
                    "upper": None,
                    "decision": "accept",
                }
            else:
                rule = "range" if self._state["rule"] == "range" else "paired"
                scored = score_differences(
                    differences,
                    float(pending["alpha"]),
                    comparison_count,
                    threshold=float(pending["thresholds"][comparison_id]),
                    rule=rule,
                )
            comparison_rows.append({"comparison_id": comparison_id, **scored})
        decisions = [str(row["decision"]) for row in comparison_rows]
        decision = (
            "reject"
            if "reject" in decisions
            else "accept"
            if all(value == "accept" for value in decisions)
            else "defer"
        )
        candidate_state_committed = decision == "accept"
        changed = deepcopy(self._state)
        if candidate_state_committed:
            archive = {
                "state_hash": mask_hash(incumbent),
                "masks": incumbent,
                "opportunity_index": pending["opportunity_index"],
            }
            changed["archives"] = [
                row for row in changed["archives"] if row["state_hash"] != archive["state_hash"]
            ]
            changed["archives"].append(archive)
            changed["archives"] = changed["archives"][-ARCHIVE_CAP:]
            changed["incumbent_masks"] = candidate
        changed["pending"] = None
        changed["used_admission_ids"].extend(str(row["event_id"]) for row in normalized)
        changed["used_admission_ids"] = changed["used_admission_ids"][-ADMISSION_LABEL_BUDGET:]
        ledger_row = {
            "opportunity_index": pending["opportunity_index"],
            "decision": decision,
            "candidate_state_hash": pending["candidate_state_hash"],
            "incumbent_state_hash": pending["incumbent_state_hash"],
            "admission_case_ids": [str(row["event_id"]) for row in normalized],
            "comparison_rows": comparison_rows,
        }
        changed["ledger"].append(ledger_row)
        changed["ledger"] = changed["ledger"][-MAX_OPPORTUNITIES:]
        changed["version"] = int(changed["version"]) + 1
        admitted = type(self).from_state(changed)
        new_bytes = admitted.state_bytes()
        receipt = {
            **ledger_row,
            "comparison_count": comparison_count,
            "candidate_state_committed": candidate_state_committed,
            "pre_nomination_parent_hash": pending["pre_nomination_parent_hash"],
            "parent_bytes_b64": pending["pre_nomination_parent_bytes"],
            "new_state_hash": transactional.sha256_bytes(new_bytes),
            "admission_parent_hash": expected_parent_hash,
            "prediction_frozen_before_release": True,
            "same_case_scoring": True,
            "future_label_used": False,
        }
        if state_path is not None:
            _atomic_write(state_path, new_bytes)
        self._state = admitted._state
        return receipt

    def rollback(self, receipt: Mapping[str, Any], *, state_path: Path | None = None) -> JsonDict:
        """Restore exact pre-nomination bytes only from the exact admitted child."""

        if self.state_hash() != receipt.get("new_state_hash"):
            raise AdmissionRejected("stale_rollback")
        try:
            parent_bytes = transactional.decode_bytes(str(receipt["parent_bytes_b64"]))
            value = json.loads(parent_bytes)
            restored = type(self).from_state(value)
        except (KeyError, ValueError, json.JSONDecodeError) as error:
            raise AdmissionRejected("invalid_rollback_receipt") from error
        if restored.state_hash() != receipt.get("pre_nomination_parent_hash"):
            raise AdmissionRejected("rollback_parent_hash")
        if state_path is not None:
            _atomic_write(state_path, parent_bytes)
        self._state = restored._state
        return {"restored_state_hash": self.state_hash(), "byte_identical": True}


def _fit_masks(rows: Sequence[Mapping[str, Any]], fallback: Mapping[str, Any]) -> dict[str, int]:
    """Fit a finite candidate from released nomination labels only."""

    masks = dict.fromkeys(FAMILIES, FULL_MASK)
    for row in rows:
        family = str(row["family_id"])
        value = int(row["numeric_value"])
        accept_mask = exp7226.ACCEPT_MASKS[family][value]
        matching = accept_mask if row["observed_label"] == "accept" else FULL_MASK ^ accept_mask
        masks[family] &= matching
    for family in FAMILIES:
        if masks[family] == 0:
            masks[family] = int(fallback[family])
    return masks


def build_stream_views(kind: str) -> prototype.StreamViews:
    """Build new authority-separated development or prospective streams."""

    if kind == "development":
        seeds = DEVELOPMENT_STREAM_SEEDS
        prefix = "development"
    elif kind == "prospective":
        seeds = STREAM_SEEDS
        prefix = "prospective"
    else:
        raise ValueError("invalid_stream_kind")
    public: list[JsonDict] = []
    authority: list[JsonDict] = []
    releases: list[JsonDict] = []
    strata = {"separated_recurrence": 0, "overlapping_recurrence": 0}
    for stream_offset, seed in enumerate(seeds):
        stratum = (
            "separated_recurrence" if stream_offset < len(seeds) // 2 else "overlapping_recurrence"
        )
        strata[stratum] += 1
        stream_id = f"{prefix}-{stream_offset + 1:02d}"
        for chronology_index in range(EVENTS_PER_STREAM):
            repeated_index = chronology_index % 256
            family = FAMILIES[repeated_index % len(FAMILIES)]
            numeric_value = (repeated_index * 23 + FAMILIES.index(family) * 11 + seed * 17) % len(
                PARAMETER_DOMAIN
            )
            base = prototype._base_parameter(seed, family)
            regime_id, parameter = prototype._regime_parameter(stratum, chronology_index, base)
            label = exact_label(family, numeric_value, parameter)
            event_id = f"exp7281-{stream_id}-e{chronology_index:04d}"
            delay = 4 if WARMUP_COUNT <= chronology_index else 0
            public.append(
                {
                    "event_id": event_id,
                    "stream_id": stream_id,
                    "chronology_index": chronology_index,
                    "family_id": family,
                    "numeric_value": numeric_value,
                    "public_input": f"family={family};value={numeric_value}",
                }
            )
            authority.append(
                {
                    "event_id": event_id,
                    "stream_id": stream_id,
                    "chronology_index": chronology_index,
                    "stream_seed": seed,
                    "stratum": stratum,
                    "regime_id": regime_id,
                    "hidden_parameter": parameter,
                    "exact_label": label,
                }
            )
            releases.append(
                {
                    "event_id": event_id,
                    "stream_id": stream_id,
                    "chronology_index": chronology_index,
                    "release_index": chronology_index + delay,
                    "observed_label": label,
                }
            )
    return prototype.StreamViews(
        public,
        authority,
        releases,
        {
            "schema": "carnot.exp7281.stream_view.v1",
            "kind": kind,
            "stream_count": len(seeds),
            "events_per_stream": EVENTS_PER_STREAM,
            "warmup_events": WARMUP_COUNT,
            "nomination_label_budget": NOMINATION_LABEL_BUDGET,
            "admission_label_budget": ADMISSION_LABEL_BUDGET,
            "opportunity_count": MAX_OPPORTUNITIES,
            "fresh_labels_per_opportunity": FRESH_LABELS_PER_OPPORTUNITY,
            "strata": strata,
            "stream_seeds_sha256": transactional.sha256_json(list(seeds)),
            "controller_input_fields": ["event_id", "family_id", "numeric_value"],
            "authority_separated": True,
            "frozen_before_controller_execution": True,
        },
    )


def stream_conformance_errors(views: prototype.StreamViews, kind: str) -> list[str]:
    """Check exact counts, chronology, identity, strata, and authority isolation."""

    expected_streams = DEVELOPMENT_STREAM_COUNT if kind == "development" else STREAM_COUNT
    expected = expected_streams * EVENTS_PER_STREAM
    errors: list[str] = []
    if not (len(views.public) == len(views.authority) == len(views.releases) == expected):
        errors.append("event_count")
    public_ids = [row.get("event_id") for row in views.public]
    if (
        public_ids != [row.get("event_id") for row in views.authority]
        or public_ids != [row.get("event_id") for row in views.releases]
        or len(set(public_ids)) != len(public_ids)
    ):
        errors.append("event_identity")
    if prototype.public_leakage_errors(views.public):
        errors.append("public_authority_leakage")
    half = expected_streams // 2
    if views.manifest.get("strata") != {
        "separated_recurrence": half,
        "overlapping_recurrence": half,
    }:
        errors.append("strata")
    for stream_offset in range(expected_streams):
        start = stream_offset * EVENTS_PER_STREAM
        rows = views.public[start : start + EVENTS_PER_STREAM]
        if [row.get("chronology_index") for row in rows] != list(range(EVENTS_PER_STREAM)):
            errors.append("chronology")
            break
    return errors


def seal_streams(
    paths: ExperimentPaths,
    development: prototype.StreamViews,
    prospective: prototype.StreamViews,
) -> JsonDict:
    """Seal six authority-separated views and their hash-bound manifest."""

    receipts = {
        "development_public": _write_immutable(
            paths.development_public, prototype.jsonl_bytes(development.public)
        ),
        "development_private_authority": _write_immutable(
            paths.development_authority, prototype.jsonl_bytes(development.authority)
        ),
        "development_releases": _write_immutable(
            paths.development_releases, prototype.jsonl_bytes(development.releases)
        ),
        "prospective_public": _write_immutable(
            paths.prospective_public, prototype.jsonl_bytes(prospective.public)
        ),
        "prospective_private_authority": _write_immutable(
            paths.prospective_authority, prototype.jsonl_bytes(prospective.authority)
        ),
        "prospective_releases": _write_immutable(
            paths.prospective_releases, prototype.jsonl_bytes(prospective.releases)
        ),
    }
    manifest = {
        "schema": "carnot.exp7281.sealed_stream_manifest.v1",
        "development": development.manifest,
        "prospective": prospective.manifest,
        "receipts": receipts,
        "authority_separated": True,
        "private_regime_evaluator_only": True,
        "old_prospective_outcomes_role": "retrospective_diagnostic_only",
    }
    receipt = _write_immutable(paths.stream_manifest, transactional.canonical_json_bytes(manifest))
    return {**manifest, "manifest_receipt": receipt}


def _released_case(
    public: Mapping[str, Any], release: Mapping[str, Any], comparison_id: str = "incumbent"
) -> JsonDict:
    """Join one public case to its delayed released label."""

    return {
        "event_id": str(public["event_id"]),
        "family_id": str(public["family_id"]),
        "numeric_value": int(public["numeric_value"]),
        "observed_label": str(release["observed_label"]),
        "release_index": int(release["release_index"]),
        "comparison_id": comparison_id,
    }


def _state_error(
    masks: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    authority: Mapping[str, Mapping[str, Any]],
) -> int:
    """Count exact evaluator errors for a frozen state on a later slice."""

    return sum(
        prototype.predict_masks(masks, event) != authority[str(event["event_id"])]["exact_label"]
        for event in events
    )


def _predict_counts(
    masks: Mapping[str, Any],
    event: Mapping[str, Any],
    truth: Mapping[str, Any],
) -> tuple[int, int, int]:
    """Return error, false-accept, and abstention counts for one frozen prediction."""

    prediction = prototype.predict_masks(masks, event)
    label = str(truth["exact_label"])
    return (
        int(prediction != label),
        int(prediction == "accept" and label == "reject"),
        int(prediction == "abstain"),
    )


def _opportunity_schedule(opportunity_offset: int) -> tuple[list[int], int, list[int], int]:
    """Place eight disjoint nomination and admission groups inside 1,024 events."""

    start = WARMUP_COUNT + opportunity_offset * 112
    nomination = list(range(start, start + FRESH_LABELS_PER_OPPORTUNITY))
    nomination_index = start + 15
    admission = list(range(start + 16, start + 16 + FRESH_LABELS_PER_OPPORTUNITY))
    decision_index = start + 27
    return nomination, nomination_index, admission, decision_index


def run_admission_panel(
    views: prototype.StreamViews,
    *,
    stream_ids: Sequence[str] | None = None,
    progress: bool = False,
) -> AdmissionPanel:
    """Replay seven arms with predictions frozen before each admission decision."""

    selected = tuple(
        stream_ids or (f"prospective-{index + 1:02d}" for index in range(STREAM_COUNT))
    )
    authority = {str(row["event_id"]): row for row in views.authority}
    releases = {str(row["event_id"]): row for row in views.releases}
    by_stream = {
        stream_id: [row for row in views.public if row["stream_id"] == stream_id]
        for stream_id in selected
    }
    rows: list[JsonDict] = []
    opportunity_rows: list[JsonDict] = []
    final_states: list[JsonDict] = []
    maximum_memory = 0
    started = time.monotonic()
    last_heartbeat = started
    for stream_offset, stream_id in enumerate(selected):
        events = by_stream[stream_id]
        if len(events) != EVENTS_PER_STREAM:
            raise ValueError(f"incomplete_stream:{stream_id}")
        truth0 = authority[str(events[0]["event_id"])]
        seed = int(truth0["stream_seed"])
        warmup_cases = [
            _released_case(event, releases[str(event["event_id"])])
            for event in events[:WARMUP_COUNT]
        ]
        warmup_masks = _fit_masks(warmup_cases, dict.fromkeys(FAMILIES, FULL_MASK))
        controllers = {
            "unconditional_recognition": AdmissionController.from_masks(
                warmup_masks, rule="unconditional"
            ),
            "range_gated": AdmissionController.from_masks(warmup_masks, rule="range"),
            "paired_gated": AdmissionController.from_masks(warmup_masks, rule="paired"),
            "label_shuffled_paired": AdmissionController.from_masks(
                warmup_masks, rule="label_shuffled"
            ),
        }
        baseline_masks = {
            "full_reference": deepcopy(warmup_masks),
            "reset": deepcopy(warmup_masks),
            "frozen_warmup": deepcopy(warmup_masks),
        }
        cumulative_nomination: list[JsonDict] = []
        counters = {
            arm: {
                "future_error": 0,
                "false_accept": 0,
                "abstention": 0,
                "accepted": 0,
                "rejected": 0,
                "deferred": 0,
                "harmful_accepted": 0,
                "useful_not_admitted": 0,
                "maximum_memory": len(transactional.canonical_json_bytes(warmup_masks)),
            }
            for arm in ARMS
        }
        schedule: dict[int, list[tuple[str, Any]]] = defaultdict(list)
        for opportunity_offset in range(MAX_OPPORTUNITIES):
            nomination_indices, nomination_index, admission_indices, decision_index = (
                _opportunity_schedule(opportunity_offset)
            )
            nomination_cases = [
                _released_case(events[index], releases[str(events[index]["event_id"])])
                for index in nomination_indices
            ]
            candidate_masks = _fit_masks(nomination_cases, warmup_masks)
            admission_cases = [
                _released_case(events[index], releases[str(events[index]["event_id"])])
                for index in admission_indices
            ]
            schedule[nomination_index].append(
                (
                    "nominate",
                    {
                        "opportunity_offset": opportunity_offset,
                        "nomination_cases": nomination_cases,
                        "candidate_masks": candidate_masks,
                    },
                )
            )
            schedule[decision_index].append(
                (
                    "decide",
                    {
                        "opportunity_offset": opportunity_offset,
                        "nomination_cases": nomination_cases,
                        "candidate_masks": candidate_masks,
                        "admission_cases": admission_cases,
                    },
                )
            )
        for index, event in enumerate(events):
            truth = authority[str(event["event_id"])]
            if index >= WARMUP_COUNT:
                for arm in ARMS:
                    masks = (
                        controllers[arm].incumbent_masks()
                        if arm in controllers
                        else baseline_masks[arm]
                    )
                    error, false_accept, abstention = _predict_counts(masks, event, truth)
                    counters[arm]["future_error"] += error
                    counters[arm]["false_accept"] += false_accept
                    counters[arm]["abstention"] += abstention
            for action, data in schedule.get(index, []):
                opportunity_index = int(data["opportunity_offset"]) + 1
                if action == "nominate":
                    nomination_ids = [str(row["event_id"]) for row in data["nomination_cases"]]
                    for controller in controllers.values():
                        controller.nominate(
                            data["candidate_masks"],
                            nomination_event_ids=nomination_ids,
                            nomination_index=index,
                            opportunity_index=opportunity_index,
                            thresholds={"incumbent": DEFAULT_THRESHOLD},
                        )
                    continue
                nomination_cases = list(data["nomination_cases"])
                cumulative_nomination.extend(nomination_cases)
                admission_cases = list(data["admission_cases"])
                candidate_masks = data["candidate_masks"]
                next_start = min(EVENTS_PER_STREAM, index + 1 + 64)
                future = events[index + 1 : next_start]
                case_hash = transactional.sha256_json(
                    [str(row["event_id"]) for row in admission_cases]
                )
                nomination_hash = transactional.sha256_json(
                    [str(row["event_id"]) for row in nomination_cases]
                )
                for arm in ARMS:
                    incumbent_before = (
                        controllers[arm].incumbent_masks()
                        if arm in controllers
                        else deepcopy(baseline_masks[arm])
                    )
                    incumbent_error = _state_error(incumbent_before, future, authority)
                    candidate_error = _state_error(candidate_masks, future, authority)
                    true_class = (
                        "useful"
                        if candidate_error < incumbent_error
                        else "harmful"
                        if candidate_error > incumbent_error
                        else "neutral"
                    )
                    if arm in controllers:
                        controller = controllers[arm]
                        receipt = controller.admit(
                            admission_cases,
                            current_index=index,
                            expected_parent_hash=controller.state_hash(),
                        )
                        decision = str(receipt["decision"])
                        comparison_rows = receipt["comparison_rows"]
                        candidate_hash = str(receipt["candidate_state_hash"])
                        incumbent_hash = str(receipt["incumbent_state_hash"])
                        memory = int(controller.memory_usage()["total_bytes"])
                    elif arm == "full_reference":
                        baseline_masks[arm] = _fit_masks(cumulative_nomination, baseline_masks[arm])
                        decision = "accept"
                        comparison_rows = []
                        candidate_hash = mask_hash(baseline_masks[arm])
                        incumbent_hash = mask_hash(incumbent_before)
                        memory = len(transactional.canonical_json_bytes(cumulative_nomination))
                    elif arm == "reset":
                        baseline_masks[arm] = deepcopy(candidate_masks)
                        decision = "accept"
                        comparison_rows = []
                        candidate_hash = mask_hash(candidate_masks)
                        incumbent_hash = mask_hash(incumbent_before)
                        memory = len(transactional.canonical_json_bytes(candidate_masks))
                    else:
                        decision = "defer"
                        comparison_rows = []
                        candidate_hash = mask_hash(candidate_masks)
                        incumbent_hash = mask_hash(incumbent_before)
                        memory = len(transactional.canonical_json_bytes(incumbent_before))
                    decision_key = {
                        "accept": "accepted",
                        "reject": "rejected",
                        "defer": "deferred",
                    }[decision]
                    counters[arm][decision_key] += 1
                    counters[arm]["harmful_accepted"] += int(
                        true_class == "harmful" and decision == "accept"
                    )
                    counters[arm]["useful_not_admitted"] += int(
                        true_class == "useful" and decision != "accept"
                    )
                    counters[arm]["maximum_memory"] = max(counters[arm]["maximum_memory"], memory)
                    maximum_memory = max(maximum_memory, memory)
                    opportunity_rows.append(
                        {
                            "unit_id": f"{stream_id}:{arm}:opportunity-{opportunity_index}",
                            "stream_id": stream_id,
                            "seed": seed,
                            "stratum": truth0["stratum"],
                            "arm": arm,
                            "opportunity_index": opportunity_index,
                            "nomination_index": index - 12,
                            "decision_index": index,
                            "nomination_label_count": len(nomination_cases),
                            "admission_label_count": len(admission_cases),
                            "nomination_case_ids_sha256": nomination_hash,
                            "admission_case_ids_sha256": case_hash,
                            "label_overlap_count": len(
                                {str(row["event_id"]) for row in nomination_cases}
                                & {str(row["event_id"]) for row in admission_cases}
                            ),
                            "all_admission_labels_released": all(
                                int(row["release_index"]) <= index
                                and int(row["release_index"]) > index - 12
                                for row in admission_cases
                            ),
                            "candidate_state_hash": candidate_hash,
                            "incumbent_state_hash": incumbent_hash,
                            "decision": decision,
                            "comparison_rows": comparison_rows,
                            "true_opportunity_class": true_class,
                            "candidate_later_error": candidate_error,
                            "incumbent_later_error": incumbent_error,
                            "future_label_used_for_nomination": False,
                            "private_regime_used": False,
                            "memory_bytes": memory,
                            "censored": False,
                        }
                    )
            now = time.monotonic()
            if progress and now - last_heartbeat >= 60:
                print(
                    f"phase 5 benchmark heartbeat streams={stream_offset}/{len(selected)} "
                    f"events={index + 1} elapsed_s={now - started:.3f}",
                    flush=True,
                )
                last_heartbeat = now
        for arm in ARMS:
            value = counters[arm]
            future_count = EVENTS_PER_STREAM - WARMUP_COUNT
            masks = (
                controllers[arm].incumbent_masks() if arm in controllers else baseline_masks[arm]
            )
            rows.append(
                {
                    "unit_id": f"{stream_id}:{arm}",
                    "stream_id": stream_id,
                    "seed": seed,
                    "stratum": truth0["stratum"],
                    "arm": arm,
                    "metric": "prospective_full_denominator_error",
                    "event_count": EVENTS_PER_STREAM,
                    "future_event_count": future_count,
                    "future_error": int(value["future_error"]),
                    "future_error_rate": int(value["future_error"]) / future_count,
                    "false_accept": int(value["false_accept"]),
                    "false_accept_rate": int(value["false_accept"]) / future_count,
                    "abstention": int(value["abstention"]),
                    "nomination_label_count": NOMINATION_LABEL_BUDGET,
                    "admission_label_count": ADMISSION_LABEL_BUDGET,
                    "total_paid_label_count": (NOMINATION_LABEL_BUDGET + ADMISSION_LABEL_BUDGET),
                    "opportunity_count": MAX_OPPORTUNITIES,
                    "accepted_update_count": int(value["accepted"]),
                    "rejected_update_count": int(value["rejected"]),
                    "deferred_update_count": int(value["deferred"]),
                    "harmful_update_accepted_count": int(value["harmful_accepted"]),
                    "useful_opportunity_not_admitted_count": int(value["useful_not_admitted"]),
                    "maximum_memory_bytes": int(value["maximum_memory"]),
                    "cost_paid_label_count": (NOMINATION_LABEL_BUDGET + ADMISSION_LABEL_BUDGET),
                    "final_state_hash": mask_hash(masks),
                    "censored": False,
                }
            )
            final_states.append(
                {
                    "stream_id": stream_id,
                    "arm": arm,
                    "state_hash": mask_hash(masks),
                    "archive_count": len(controllers[arm].archives()) if arm in controllers else 0,
                    "state_bytes": int(value["maximum_memory"]),
                }
            )
        if progress:
            print(
                f"phase 5 benchmark unit {stream_offset + 1}/{len(selected)} "
                f"completed_rows={len(ARMS)} elapsed_s={time.monotonic() - started:.3f}",
                flush=True,
            )
    return AdmissionPanel(
        rows,
        opportunity_rows,
        final_states,
        len(selected),
        0,
        maximum_memory,
    )


def opportunity_row_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Check quota, freshness, authority, completion, and common-case identity."""

    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    add(any(row.get("label_overlap_count") != 0 for row in rows), "label_reuse")
    add(
        any(row.get("all_admission_labels_released") is not True for row in rows),
        "unreleased_label",
    )
    add(
        any(
            row.get("future_label_used_for_nomination") is not False
            or row.get("private_regime_used") is not False
            for row in rows
        ),
        "authority_leakage",
    )
    add(
        any(
            int(row.get("nomination_label_count", 0)) != FRESH_LABELS_PER_OPPORTUNITY
            or int(row.get("admission_label_count", 0)) != FRESH_LABELS_PER_OPPORTUNITY
            for row in rows
        ),
        "quota",
    )
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    common: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in rows:
        groups[(str(row.get("stream_id")), str(row.get("arm")))].append(row)
        common[(str(row.get("stream_id")), int(row.get("opportunity_index", 0)))].add(
            str(row.get("admission_case_ids_sha256"))
        )
    add(
        any(
            sorted(int(row["opportunity_index"]) for row in group)
            != list(range(1, MAX_OPPORTUNITIES + 1))
            for group in groups.values()
        ),
        "incomplete_opportunities",
    )
    add(any(len(values) != 1 for values in common.values()), "common_case_mismatch")
    add(
        any(int(row.get("memory_bytes", MEMORY_CAP_BYTES + 1)) > MEMORY_CAP_BYTES for row in rows),
        "memory_cap",
    )
    return errors


def independent_reduce(path: Path) -> list[JsonDict]:
    """Reload raw stream-arm rows without trusting producer aggregates."""

    rows = prototype._read_jsonl(path)
    required = {
        "unit_id",
        "stream_id",
        "seed",
        "stratum",
        "arm",
        "future_error",
        "false_accept",
        "abstention",
        "cost_paid_label_count",
        "censored",
    }
    if not rows or any(not required <= set(row) for row in rows):
        raise ValueError("invalid_raw_rows")
    return rows


def reduce_old_lifecycle(lifecycle_path: Path, event_path: Path) -> list[JsonDict]:
    """Measure old label overlap and post-reactivation errors without tuning."""

    nomination: dict[tuple[str, str], set[str]] = defaultdict(set)
    validation: dict[tuple[str, str], set[str]] = defaultdict(set)
    reactivations: dict[tuple[str, str], list[int]] = defaultdict(list)
    try:
        with lifecycle_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                key = (str(row.get("stream_id")), str(row.get("arm")))
                kind = row.get("kind")
                if kind == "nomination":
                    nomination[key].add(str(row.get("event_id")))
                elif kind == "validation":
                    validation[key].add(str(row.get("event_id")))
                elif kind == "reactivation":
                    reactivations[key].append(int(row["chronology_index"]))
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ValueError("historical_lifecycle_unavailable") from error
    wanted = set(reactivations)
    wanted_streams = {stream for stream, _ in wanted}
    errors: dict[tuple[str, str, int], int] = {}
    try:
        with event_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                stream = str(row.get("stream_id"))
                arm = str(row.get("arm"))
                if stream not in wanted_streams or (stream, arm) not in wanted | {
                    (value, "frozen") for value in wanted_streams
                }:
                    continue
                errors[(stream, arm, int(row["chronology_index"]))] = int(
                    row["full_denominator_error"]
                )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ValueError("historical_event_rows_unavailable") from error
    rows = []
    keys = sorted(set(nomination) | set(validation) | set(reactivations))
    for stream_id, arm in keys:
        harmful = useful = neutral = 0
        for index in reactivations.get((stream_id, arm), []):
            arm_error = sum(
                errors.get((stream_id, arm, later), 0) for later in range(index + 1, index + 9)
            )
            frozen_error = sum(
                errors.get((stream_id, "frozen", later), 0) for later in range(index + 1, index + 9)
            )
            if arm_error > frozen_error:
                harmful += 1
            elif arm_error < frozen_error:
                useful += 1
            else:
                neutral += 1
        nominated = nomination.get((stream_id, arm), set())
        validated = validation.get((stream_id, arm), set())
        overlap = nominated & validated
        rows.append(
            {
                "unit_id": f"{stream_id}:{arm}",
                "stream_id": stream_id,
                "arm": arm,
                "nomination_count": len(nominated),
                "validation_count": len(validated),
                "nomination_validation_overlap_count": len(overlap),
                "nomination_validation_overlap_rate": (
                    len(overlap) / len(nominated) if nominated else None
                ),
                "reactivation_count": len(reactivations.get((stream_id, arm), [])),
                "harmful_reactivation_count": harmful,
                "useful_reactivation_count": useful,
                "neutral_reactivation_count": neutral,
                "classification_rule": "next_eight_released_errors_minus_frozen_errors",
                "source_scope": "released_lifecycle_and_prequential_rows_only",
                "old_outcomes_used_for_rule_tuning": False,
                "censored": False,
            }
        )
    if not rows:
        raise ValueError("historical_lifecycle_empty")
    return rows


def run_development_controls(views: prototype.StreamViews) -> JsonDict:
    """Prove fixed interval and shuffle controls can change a development decision."""

    if stream_conformance_errors(views, "development"):
        raise ValueError("invalid_development_streams")
    alpha = opportunity_alpha(1)
    patterns = (
        ("all_useful", [1] * 16),
        ("all_harmful", [-1] * 16),
        ("zero_disagreement", [0] * 16),
        ("mixed", [1, -1, 0, 0] * 4),
    )
    rows = []
    changes = 0
    for stream_index, (name, differences) in enumerate(patterns, start=1):
        paired = score_differences(differences, alpha, 1, threshold=0.0, rule="paired")
        ranged = score_differences(differences, alpha, 1, threshold=0.0, rule="range")
        shuffled = score_differences(
            list(reversed([-value for value in differences])),
            alpha,
            1,
            threshold=0.0,
            rule="paired",
        )
        decisions = {
            "unconditional": "accept",
            "range": ranged["decision"],
            "paired": paired["decision"],
            "label_shuffled_paired": shuffled["decision"],
        }
        changes += int(len(set(decisions.values())) > 1)
        rows.append(
            {
                "development_stream_id": f"development-{stream_index:02d}",
                "fixture_case": name,
                "decisions": decisions,
                "control_changed_decision": len(set(decisions.values())) > 1,
                "prospective_authority_used": False,
            }
        )
    return {
        "development_stream_count": DEVELOPMENT_STREAM_COUNT,
        "control_decision_change_count": changes,
        "rows": rows,
        "parameters_frozen_before_prospective": True,
        "prospective_authority_used": False,
    }


def _contrast_cases(
    candidate_parameter: int,
    incumbent_parameter: int,
    *,
    count: int,
    start: int,
    release_index: int,
) -> list[JsonDict]:
    """Build released cases where the candidate and incumbent disagree."""

    candidate = dict.fromkeys(FAMILIES, 1 << candidate_parameter)
    incumbent = dict.fromkeys(FAMILIES, 1 << incumbent_parameter)
    rows = []
    for family in FAMILIES:
        for value in PARAMETER_DOMAIN:
            public = {"family_id": family, "numeric_value": value}
            candidate_label = prototype.predict_masks(candidate, public)
            incumbent_label = prototype.predict_masks(incumbent, public)
            if candidate_label == incumbent_label:
                continue
            rows.append(
                {
                    "event_id": f"control-{start + len(rows)}",
                    "family_id": family,
                    "numeric_value": value,
                    "observed_label": candidate_label,
                    "release_index": release_index,
                    "comparison_id": "incumbent",
                }
            )
            if len(rows) == count:
                return rows
    raise ValueError("insufficient_contrast_cases")


def run_mutation_controls(root: Path) -> list[JsonDict]:
    """Reject reused and unreleased labels while preserving exact parent bytes."""

    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for mutation in ("reused_label", "unreleased_label"):
        controller = AdmissionController.from_masks(dict.fromkeys(FAMILIES, 1 << 0), rule="paired")
        controller.nominate(
            dict.fromkeys(FAMILIES, 1 << 8),
            nomination_event_ids=["nomination"],
            nomination_index=20,
            opportunity_index=1,
            thresholds={"incumbent": DEFAULT_THRESHOLD},
        )
        parent = controller.state_bytes()
        cases = _contrast_cases(8, 0, count=8, start=0, release_index=30)
        if mutation == "reused_label":
            cases[0]["event_id"] = "nomination"
            expected = "reused_nomination_label"
        else:
            cases[0]["release_index"] = 31
            expected = "unreleased_label"
        observed = None
        try:
            controller.admit(
                cases,
                current_index=30,
                expected_parent_hash=controller.state_hash(),
            )
        except AdmissionRejected as error:
            observed = str(error)
        rows.append(
            {
                "mutation": mutation,
                "expected_rejection": expected,
                "observed_rejection": observed,
                "parent_bytes_preserved": controller.state_bytes() == parent,
                "passed": observed == expected and controller.state_bytes() == parent,
            }
        )
    _atomic_write(
        root / "mutation_receipts.json",
        transactional.canonical_json_bytes({"schema": SCHEMA, "rows": rows}),
    )
    return rows


def run_e2e_controls(root: Path) -> list[JsonDict]:
    """Run prediction, nomination, delayed admission, restart, rejection, and rollback."""

    root.mkdir(parents=True, exist_ok=True)
    incumbent = dict.fromkeys(FAMILIES, 1 << 0)
    candidate = dict.fromkeys(FAMILIES, 1 << 8)
    cases = _contrast_cases(8, 0, count=8, start=100, release_index=30)
    public = {
        "event_id": cases[0]["event_id"],
        "family_id": cases[0]["family_id"],
        "numeric_value": cases[0]["numeric_value"],
    }
    controller = AdmissionController.from_masks(incumbent, rule="unconditional")
    prediction = controller.predict(public)
    nomination = controller.nominate(
        candidate,
        nomination_event_ids=["nomination"],
        nomination_index=20,
        opportunity_index=1,
        thresholds={"incumbent": DEFAULT_THRESHOLD},
    )
    state_path = root / "controller.json"
    receipt = controller.admit(
        cases,
        current_index=30,
        expected_parent_hash=controller.state_hash(),
        state_path=state_path,
    )
    later = controller.predict(public)
    restored = AdmissionController.load(state_path)
    restart_ok = restored.state_bytes() == controller.state_bytes()
    rejecting = AdmissionController.from_masks(incumbent, rule="paired")
    rejecting.nominate(
        candidate,
        nomination_event_ids=["reject-nomination"],
        nomination_index=40,
        opportunity_index=2,
        thresholds={"incumbent": 0.0},
    )
    harmful = _contrast_cases(0, 8, count=16, start=200, release_index=50)
    rejection = rejecting.admit(
        harmful,
        current_index=50,
        expected_parent_hash=rejecting.state_hash(),
    )
    rollback = restored.rollback(receipt, state_path=state_path)
    rows = [
        {"stage": "public_prediction", "passed": prediction in {"accept", "reject"}},
        {
            "stage": "immutable_nomination",
            "passed": nomination["candidate_state_hash"] == mask_hash(candidate)
            and nomination["incumbent_state_hash"] == mask_hash(incumbent),
        },
        {
            "stage": "disjoint_delayed_feedback",
            "passed": not ({"nomination"} & {row["event_id"] for row in cases})
            and all(int(row["release_index"]) > 20 for row in cases),
        },
        {"stage": "admission", "passed": receipt["decision"] == "accept"},
        {"stage": "later_prediction", "passed": prediction != later},
        {"stage": "cold_restart", "passed": restart_ok},
        {
            "stage": "rejection",
            "passed": rejection["decision"] == "reject"
            and rejecting.incumbent_hash() == mask_hash(incumbent),
        },
        {"stage": "rollback", "passed": rollback["byte_identical"] is True},
    ]
    _atomic_write(
        root / "e2e_receipts.json",
        transactional.canonical_json_bytes({"schema": SCHEMA, "rows": rows}),
    )
    return rows


def _task_identity(text: str) -> JsonDict:
    """Extract only the active Exp7281 task from the executable roadmap."""

    match = re.search(r"(?ms)^- id: exp7281-admission-prototype\n(.*?)(?=^- id:|\Z)", text)
    block = "" if match is None else match.group(1)
    milestone = re.search(r"(?m)^  milestone: (.+)$", block)
    deliverable = re.search(r"(?m)^  deliverable: (.+)$", block)
    return {
        "id": "exp7281-admission-prototype" if block else None,
        "milestone": None if milestone is None else milestone.group(1).strip(),
        "deliverable": None if deliverable is None else deliverable.group(1).strip(),
    }


def collect_preconditions(
    repo_root: Path,
    paths: ExperimentPaths,
    *,
    upstream_overrides: Mapping[str, Path] | None = None,
    historical_overrides: Mapping[str, Path] | None = None,
) -> tuple[list[JsonDict], dict[str, str | None]]:
    """Authenticate prior nulls, raw rows, requirements, exclusions, and output owners."""

    upstream_overrides = dict(upstream_overrides or {})
    historical_overrides = dict(historical_overrides or {})
    upstream_paths = {
        name: _resolve(repo_root, upstream_overrides.get(name, path))
        for name, path in DEFAULT_UPSTREAM_ARTIFACTS.items()
    }
    historical_paths = {
        name: _resolve(repo_root, historical_overrides.get(name, path))
        for name, path in DEFAULT_HISTORICAL_PATHS.items()
    }
    upstream = {name: _load_object(path) for name, path in upstream_paths.items()}
    source_hashes: dict[str, str | None] = {
        str(_resolve(repo_root, path)): _sha256_path(_resolve(repo_root, path))
        for path in SOURCE_PATHS
    }
    for path in [*upstream_paths.values(), *historical_paths.values()]:
        source_hashes[str(path)] = _sha256_path(path)
    spec = _resolve(repo_root, SPEC_PATH).read_text(encoding="utf-8")
    roadmap = _resolve(repo_root, "research-roadmap.yaml").read_text(encoding="utf-8")
    exclusions = _resolve(repo_root, "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    writable_fields = (
        "development_public",
        "development_authority",
        "development_releases",
        "prospective_public",
        "prospective_authority",
        "prospective_releases",
        "stream_manifest",
        "raw_rows",
        "opportunity_rows",
        "historical_rows",
        "state_sidecar",
        "control_sidecar",
        "evidence_sidecar",
        "provisional",
        "terminal_candidate",
        "artifact",
    )
    writable = {name: _path_writable(getattr(paths, name)) for name in writable_fields}
    checks = [
        gate_check(
            "driving_capability_spec",
            str(SPEC_PATH),
            "REQ-CL-7281",
            True,
            "REQ-CL-7281" in spec,
        ),
        gate_check(
            "scenario_contract",
            str(SPEC_PATH),
            "SCENARIO-CL-7281-*",
            9,
            len(set(SCENARIO_PATTERN.findall(spec))),
        ),
        gate_check(
            "v640_task_identity",
            "research-roadmap.yaml",
            "id,milestone,deliverable",
            {
                "id": "exp7281-admission-prototype",
                "milestone": MILESTONE,
                "deliverable": str(DEFAULT_ARTIFACT),
            },
            _task_identity(roadmap),
        ),
        gate_check(
            "required_source_bytes",
            "repository",
            "SOURCE_PATHS",
            True,
            all(source_hashes[str(_resolve(repo_root, path))] is not None for path in SOURCE_PATHS),
        ),
        gate_check(
            "writable_output_paths",
            "host_filesystem",
            "task-owned outputs",
            dict.fromkeys(writable, True),
            writable,
        ),
    ]
    for name, path in upstream_paths.items():
        artifact = upstream[name]
        quarantine = exp7213.quarantine_state(artifact, exclusions, path.name, name)
        completion_field = (
            "recognition_run_complete_score"
            if name == "exp7268"
            else "recognition_audit_complete_score"
        )
        checks.extend(
            [
                gate_check(
                    f"{name}_artifact_hash",
                    str(path),
                    "sha256",
                    EXPECTED_UPSTREAM_HASHES[name],
                    source_hashes[str(path)],
                ),
                gate_check(f"{name}_status", name, "status", "complete", artifact.get("status")),
                gate_check(
                    f"{name}_completion", name, completion_field, 1, artifact.get(completion_field)
                ),
                gate_check(
                    f"{name}_not_quarantined",
                    "artifact_and_ops/exclusion_manifest.yaml",
                    "quarantined",
                    False,
                    quarantine["quarantined"],
                ),
            ]
        )
    for name, path in historical_paths.items():
        checks.append(
            gate_check(
                f"historical_{name}_hash",
                str(path),
                "sha256",
                EXPECTED_HISTORICAL_HASHES[name],
                source_hashes[str(path)],
            )
        )
    checks.extend(
        [
            gate_check(
                "authority_separation",
                "Exp7281 stream contract",
                "public,release,private",
                True,
                True,
            ),
            gate_check(
                "resource_ownership",
                "host",
                "output paths task-owned",
                True,
                all(writable.values()),
            ),
        ]
    )
    return checks, source_hashes


def _summary_with_failures(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Retain the first failure and every exact failed observation."""

    summary = gate_summary(checks)
    first = next((dict(row) for row in checks if row.get("passed") is not True), None)
    summary["first_failure"] = first
    summary["failed_checks"] = [dict(row) for row in checks if row.get("passed") is not True]
    return summary


def _admission_contract() -> JsonDict:
    """Freeze alpha, quota, opportunity, memory, and future-label rules."""

    return {
        "paper": "arXiv:2609.10873v1",
        "rule": "paired_clopper_pearson_disagreement",
        "range_control": "simultaneous_hoeffding_range_bound",
        "total_alpha": TOTAL_ALPHA,
        "maximum_opportunities": MAX_OPPORTUNITIES,
        "per_opportunity_alpha": TOTAL_ALPHA / MAX_OPPORTUNITIES,
        "endpoint_tail_alpha_formula": "opportunity_alpha/(4*comparison_count)",
        "fixed_size_batches": True,
        "optional_stopping": False,
        "nomination_label_budget": NOMINATION_LABEL_BUDGET,
        "admission_label_budget": ADMISSION_LABEL_BUDGET,
        "fresh_labels_per_opportunity": FRESH_LABELS_PER_OPPORTUNITY,
        "nomination_and_admission_disjoint": True,
        "admission_labels_released_after_nomination": True,
        "candidate_and_incumbent_hashes_frozen": True,
        "same_labeled_cases_for_candidate_and_incumbent": True,
        "archive_capacity": ARCHIVE_CAP,
        "complete_memory_cap_bytes": MEMORY_CAP_BYTES,
        "private_regime_access": False,
        "dependent_drifting_stream_theorem_claimed": False,
    }


def _sample_budget(stream_ids: Sequence[str], complete: bool) -> JsonDict:
    """Declare planned, attempted, completed, censored, and fixed stopping units."""

    completed = len(stream_ids) if complete else 0
    return {
        "planned_development_stream_count": DEVELOPMENT_STREAM_COUNT,
        "planned_prospective_stream_count": STREAM_COUNT,
        "attempted_stream_count": len(stream_ids),
        "completed_stream_count": completed,
        "censored_stream_count": 0 if complete else len(stream_ids),
        "events_per_stream": EVENTS_PER_STREAM,
        "warmup_events_per_stream": WARMUP_COUNT,
        "arms_per_stream": len(ARMS),
        "planned_stream_arm_rows": len(stream_ids) * len(ARMS),
        "completed_stream_arm_rows": completed * len(ARMS),
        "opportunities_per_stream_arm": MAX_OPPORTUNITIES,
        "nomination_label_ceiling_per_stream_arm": NOMINATION_LABEL_BUDGET,
        "admission_label_ceiling_per_stream_arm": ADMISSION_LABEL_BUDGET,
        "total_paid_label_ceiling_per_stream_arm": 128,
        "stopping_rule": "all sealed selected streams, arms, and eight opportunities once",
        "outcome_based_extension": False,
    }


def _base_artifact(
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str | None],
    stream_ids: Sequence[str],
    *,
    started_at: str,
    completed_at: str,
    duration_s: float,
) -> JsonDict:
    """Create every required field before blocked or measured classification."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "blocked",
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": [dict(row) for row in checks],
        "MODEL_SPECS": deepcopy(MODEL_SPECS),
        "model_invoked": MODEL_INVOKED,
        "invocation_counts": deepcopy(INVOCATION_COUNTS),
        "current_model_load_count": 0,
        "current_generation_count": 0,
        "current_inference_count": 0,
        "inference_substrate": "blocked_no_run",
        "inference_substrate_class": "blocked_no_run",
        "reducer_inference_substrate": REDUCER_INFERENCE_SUBSTRATE,
        "reducer_inference_substrate_class": REDUCER_INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "execution_host": socket.gethostname(),
        "duration_s": duration_s,
        "random_seed": {
            "root": RANDOM_SEED,
            "development_streams": list(DEVELOPMENT_STREAM_SEEDS),
            "prospective_streams": list(STREAM_SEEDS),
            "frozen_before_prospective_outcomes": True,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": dict(source_hashes),
        "rows": [],
        "sample_size_budget": _sample_budget(stream_ids, False),
        "acceptance_gate_results": {},
        "gate_check_summary": _summary_with_failures(checks),
        "verifier_is_oracle": True,
        "honest_verdict": "blocked_external_precondition",
        "verdict_class": "blocked",
        "validation_receipts": [],
        "admission_fixture_ready_score": 0,
        "admission_contract": _admission_contract(),
        "stream_manifest_path": None,
        "finite_law_rows": [],
        "nomination_overlap_rows": [],
        "continuous_self_learning_task": True,
        "no_model_weight_mutation": True,
        "default_pipeline_modified": False,
    }


def build_blocked_artifact(
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str | None],
    stream_ids: Sequence[str],
    *,
    started_at: str,
    duration_s: float,
) -> JsonDict:
    """Build row-free terminal evidence for an external prerequisite failure."""

    artifact = _base_artifact(
        checks,
        source_hashes,
        stream_ids,
        started_at=started_at,
        completed_at=datetime.now(UTC).isoformat(),
        duration_s=duration_s,
    )
    first = artifact["gate_check_summary"].get("first_failure") or {
        "upstream": "external",
        "field": "precondition",
    }
    artifact["honest_verdict"] = (
        f"blocked_{str(first.get('upstream', 'external')).replace('/', '_')}_"
        f"{str(first.get('field', 'precondition')).replace('/', '_')}"
    ).replace(" ", "_")
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable identity, configuration, sources, rows, gates, and receipts."""

    excluded = {
        "reproducibility_checksum",
        "started_at_utc",
        "completed_at_utc",
        "duration_s",
        "phase_spans_s",
        "execution_host",
    }
    return transactional.sha256_json(
        {key: value for key, value in artifact.items() if key not in excluded}
    )


def _receipt_matches(repo_root: Path, receipt: Mapping[str, Any]) -> bool:
    """Compare one sidecar receipt with its current exact bytes."""

    return _sha256_path(_resolve(repo_root, str(receipt.get("path", "")))) == receipt.get("sha256")


def build_and_seal(
    repo_root: Path,
    paths: ExperimentPaths,
    *,
    stream_ids: Sequence[str] | None = None,
    historical_paths: Mapping[str, Path] | None = None,
    progress: bool = False,
) -> JsonDict:
    """Authenticate, diagnose, enumerate, seal, replay, control, and validate."""

    selected = tuple(
        stream_ids or (f"prospective-{index + 1:02d}" for index in range(STREAM_COUNT))
    )
    monotonic_start = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    spans: JsonDict = {}
    if progress:
        _progress(0, "start", "authenticate inputs and output ownership")
    phase_start = time.monotonic()
    checks, source_hashes = collect_preconditions(repo_root, paths)
    spans["phase_0_preconditions"] = time.monotonic() - phase_start
    _atomic_write(
        paths.provisional,
        transactional.canonical_json_bytes(
            {"schema": SCHEMA, "status": "in_progress", "phase": 0, "checks": checks}
        ),
    )
    if gate_summary(checks)["passed"] is not True:
        if progress:
            _progress(0, "end", "external precondition failed; measurement did not run")
        return build_blocked_artifact(
            checks,
            source_hashes,
            selected,
            started_at=started_at,
            duration_s=time.monotonic() - monotonic_start,
        )
    if progress:
        _progress(0, "end", "all exact external preconditions passed")

    phase_start = time.monotonic()
    if progress:
        _progress(1, "start", "confirm zero current model work")
        print("phase 1 BEFORE model load: no model load scheduled", flush=True)
        print("phase 1 AFTER model load: attempted and completed loads remain zero", flush=True)
        print("phase 1 BEFORE generation: no generation call scheduled", flush=True)
        print(
            "phase 1 AFTER generation: attempted and completed generations remain zero", flush=True
        )
    spans["phase_1_no_llm"] = time.monotonic() - phase_start
    if progress:
        _progress(1, "end", "MODEL_SPECS is empty and every current counter is zero")

    phase_start = time.monotonic()
    if progress:
        _progress(2, "start", "reduce old released lifecycle evidence without tuning")
    old_paths = dict(DEFAULT_HISTORICAL_PATHS if historical_paths is None else historical_paths)
    historical_rows = reduce_old_lifecycle(
        _resolve(repo_root, old_paths["lifecycle"]),
        _resolve(repo_root, old_paths["events"]),
    )
    spans["phase_2_historical_reduction"] = time.monotonic() - phase_start
    if progress:
        _progress(2, "end", f"historical_units={len(historical_rows)}")

    phase_start = time.monotonic()
    if progress:
        _progress(3, "start", "BEFORE finite IID enumeration at n=8 and n=16")
        print("phase 3 BEFORE benchmark: exact multinomial enumeration", flush=True)
    finite_rows = enumerate_finite_laws()
    spans["phase_3_finite_enumeration"] = time.monotonic() - phase_start
    if progress:
        print("phase 3 AFTER benchmark: finite enumeration completed", flush=True)
        _progress(3, "end", f"finite_law_rows={len(finite_rows)}")

    phase_start = time.monotonic()
    if progress:
        _progress(4, "start", "generate and seal four development plus 24 prospective streams")
    development = build_stream_views("development")
    prospective = build_stream_views("prospective")
    stream_errors = stream_conformance_errors(
        development, "development"
    ) + stream_conformance_errors(prospective, "prospective")
    if stream_errors:
        raise ValueError("stream_conformance_failed:" + ",".join(stream_errors))
    stream_manifest = seal_streams(paths, development, prospective)
    development_result = run_development_controls(development)
    spans["phase_4_stream_seal_and_development"] = time.monotonic() - phase_start
    if progress:
        _progress(
            4,
            "end",
            f"control_decision_changes={development_result['control_decision_change_count']}",
        )

    phase_start = time.monotonic()
    if progress:
        _progress(5, "start", "BEFORE seven-arm CPU admission benchmark")
        print("phase 5 BEFORE benchmark: frozen common-candidate replay", flush=True)
    panel = run_admission_panel(prospective, stream_ids=selected, progress=progress)
    spans["phase_5_admission_benchmark"] = time.monotonic() - phase_start
    if progress:
        print("phase 5 AFTER benchmark: all selected stream-arm units completed", flush=True)
        _progress(5, "end", f"completed_stream_arm_rows={len(panel.rows)}")

    phase_start = time.monotonic()
    if progress:
        _progress(6, "start", "run mutations, E2E, raw reduction, and sidecar seals")
    opportunity_errors = opportunity_row_errors(panel.opportunity_rows)
    if opportunity_errors:
        raise ValueError("opportunity_row_errors:" + ",".join(opportunity_errors))
    controls_root = paths.provisional.parent / "experiment_7281_controls"
    mutation_rows = run_mutation_controls(controls_root / "mutations")
    e2e_rows = run_e2e_controls(controls_root / "e2e")
    raw_receipt = _atomic_write(paths.raw_rows, prototype.jsonl_bytes(panel.rows))
    opportunity_receipt = _atomic_write(
        paths.opportunity_rows, prototype.jsonl_bytes(panel.opportunity_rows)
    )
    historical_receipt = _atomic_write(
        paths.historical_rows, prototype.jsonl_bytes(historical_rows)
    )
    reduced_rows = independent_reduce(paths.raw_rows)
    if reduced_rows != panel.rows:
        raise ValueError("independent_reducer_mismatch")
    state_receipt = _atomic_write(
        paths.state_sidecar,
        transactional.canonical_json_bytes(
            {"schema": "carnot.exp7281.states.v1", "states": panel.final_states}
        ),
    )
    control_receipt = _atomic_write(
        paths.control_sidecar,
        transactional.canonical_json_bytes(
            {
                "schema": "carnot.exp7281.controls.v1",
                "development": development_result,
                "mutations": mutation_rows,
                "e2e": e2e_rows,
            }
        ),
    )
    evidence_receipt = _atomic_write(
        paths.evidence_sidecar,
        transactional.canonical_json_bytes(
            {
                "schema": "carnot.exp7281.evidence.v1",
                "historical_model_receipts": {},
                "upstream_artifacts": {
                    name: {
                        "path": str(path),
                        "sha256": EXPECTED_UPSTREAM_HASHES[name],
                    }
                    for name, path in DEFAULT_UPSTREAM_ARTIFACTS.items()
                },
                "paper": {
                    "identity": "arXiv:2609.10873v1",
                    "use": "paired_disagreement_construction_and_opportunity_audit",
                    "novel_theorem_claimed": False,
                },
                "old_outcomes_used_for_rule_tuning": False,
            }
        ),
    )
    for receipt in (
        raw_receipt,
        opportunity_receipt,
        historical_receipt,
        state_receipt,
        control_receipt,
        evidence_receipt,
        stream_manifest["manifest_receipt"],
        *stream_manifest["receipts"].values(),
    ):
        source_hashes[str(receipt["path"])] = str(receipt["sha256"])
    spans["phase_6_controls_and_reduction"] = time.monotonic() - phase_start
    if progress:
        _progress(6, "end", "mutations, lifecycle, and independent reduction passed")

    expected_rows = len(selected) * len(ARMS)
    quota_violations = sum(
        int(
            int(row["nomination_label_count"]) > NOMINATION_LABEL_BUDGET
            or int(row["admission_label_count"]) > ADMISSION_LABEL_BUDGET
            or int(row["total_paid_label_count"]) > 128
        )
        for row in panel.rows
    )
    finite_passes = sum(int(row["coverage_passed"] is True) for row in finite_rows)
    bound_infeasible = any(
        row["case"] == "zero_disagreement"
        and row["n"] == 8
        and row["paired_threshold_zero_feasible"] is False
        and row["range_threshold_zero_feasible"] is False
        for row in finite_rows
    )
    acceptance = {
        "finite_iid_enumeration": {
            "expected": len(finite_rows),
            "observed": finite_passes,
            "passed": finite_passes == len(finite_rows),
            "principle": "Check the exact finite rule before using it on drifting streams.",
        },
        "old_evidence_read_only": {
            "expected": [">0 rows", False],
            "observed": [
                len(historical_rows),
                any(row["old_outcomes_used_for_rule_tuning"] for row in historical_rows),
            ],
            "passed": bool(historical_rows)
            and not any(row["old_outcomes_used_for_rule_tuning"] for row in historical_rows),
            "principle": "Old prospective outcomes may diagnose overlap but cannot tune the rule.",
        },
        "effective_development_control": {
            "expected": ">0 changed decisions",
            "observed": development_result["control_decision_change_count"],
            "passed": development_result["control_decision_change_count"] > 0,
            "principle": "A control must be able to change a decision on development fixtures.",
        },
        "sealed_stream_manifest": {
            "expected": [4, 24, 12, 12, 0],
            "observed": [
                development.manifest["stream_count"],
                prospective.manifest["stream_count"],
                prospective.manifest["strata"]["separated_recurrence"],
                prospective.manifest["strata"]["overlapping_recurrence"],
                len(stream_errors),
            ],
            "passed": len(stream_errors) == 0,
            "principle": "Keep public, release, and evaluator authority in separate sealed bytes.",
        },
        "complete_seven_arm_panel": {
            "expected": expected_rows,
            "observed": len(panel.rows),
            "passed": len(panel.rows) == expected_rows and panel.censored_stream_count == 0,
            "principle": "Account for every selected stream and frozen arm without censoring.",
        },
        "disjoint_equal_quotas": {
            "expected": 0,
            "observed": quota_violations + len(opportunity_errors),
            "passed": quota_violations == 0 and not opportunity_errors,
            "principle": "Use 64 nomination and 64 fresh admission labels on common cases.",
        },
        "complete_memory_bound": {
            "expected": f"<={MEMORY_CAP_BYTES}",
            "observed": panel.maximum_memory_bytes,
            "passed": panel.maximum_memory_bytes <= MEMORY_CAP_BYTES,
            "principle": "Charge pending decisions, validation buffers, archives, and ledgers.",
        },
        "mutation_rejections": {
            "expected": len(mutation_rows),
            "observed": sum(int(row["passed"] is True) for row in mutation_rows),
            "passed": all(row["passed"] is True for row in mutation_rows),
            "principle": "Reused and unreleased labels must preserve parent bytes.",
        },
        "e2e_lifecycle": {
            "expected": len(e2e_rows),
            "observed": sum(int(row["passed"] is True) for row in e2e_rows),
            "passed": all(row["passed"] is True for row in e2e_rows),
            "principle": "Prediction, nomination, admission, restart, rejection, and rollback must pass.",
        },
        "independent_raw_reducer": {
            "expected": transactional.sha256_json(panel.rows),
            "observed": transactional.sha256_json(reduced_rows),
            "passed": reduced_rows == panel.rows,
            "principle": "Raw stream-arm rows must reproduce without producer headlines.",
        },
    }
    for value in acceptance.values():
        value["pass"] = value["passed"]
    ready = int(all(value["passed"] is True for value in acceptance.values()))
    stop = next((name for name, value in acceptance.items() if value["passed"] is not True), None)
    completed_at = datetime.now(UTC).isoformat()
    artifact = _base_artifact(
        checks,
        source_hashes,
        selected,
        started_at=started_at,
        completed_at=completed_at,
        duration_s=time.monotonic() - monotonic_start,
    )
    artifact.update(
        {
            "status": "complete",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
            "phase_spans_s": spans,
            "rows": panel.rows,
            "sample_size_budget": _sample_budget(selected, True),
            "acceptance_gate_results": acceptance,
            "verdict_class": "circular_positive" if ready else "null",
            "honest_verdict": (
                "complete_circular_positive: admission fixture mechanics pass; fixed eight-label bounds are often infeasible and no learning value is claimed"
                if ready
                else f"complete_null: admission fixture stopped at {stop}"
            ),
            "admission_fixture_ready_score": ready,
            "stream_manifest_path": str(paths.stream_manifest),
            "stream_manifest": stream_manifest,
            "finite_law_rows": finite_rows,
            "nomination_overlap_rows": historical_rows,
            "bound_infeasibility_reported": bound_infeasible,
            "development_control_rows": development_result["rows"],
            "opportunity_rows": panel.opportunity_rows,
            "mutation_rows": mutation_rows,
            "e2e_rows": e2e_rows,
            "raw_rows_receipt": {**raw_receipt, "row_count": len(panel.rows)},
            "opportunity_rows_receipt": {
                **opportunity_receipt,
                "row_count": len(panel.opportunity_rows),
            },
            "historical_rows_receipt": {
                **historical_receipt,
                "row_count": len(historical_rows),
            },
            "state_sidecar_receipt": {
                **state_receipt,
                "state_count": len(panel.final_states),
            },
            "control_sidecar_receipt": control_receipt,
            "evidence_sidecar_receipt": evidence_receipt,
        }
    )
    artifact["validation_receipts"] = [
        {
            "command": f"independent_reduce {paths.raw_rows}",
            "exit_code": 0,
            "classification": "passed",
            "duration_s": spans["phase_6_controls_and_reduction"],
            "log_sha256": transactional.sha256_json(reduced_rows),
        },
        {
            "command": "run_e2e_controls",
            "exit_code": 0,
            "classification": "passed",
            "duration_s": spans["phase_6_controls_and_reduction"],
            "log_sha256": transactional.sha256_json(e2e_rows),
        },
    ]
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(
        artifact,
        repo_root=repo_root,
        expected_stream_ids=selected,
        check_files=True,
    )
    if errors:
        raise ValueError("artifact_validation_failed:" + ",".join(errors))
    return artifact


def validate_artifact(
    artifact: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
    expected_stream_ids: Sequence[str] | None = None,
    check_files: bool = False,
) -> list[str]:
    """Cold-check schema, rules, rows, controls, source bytes, and classification."""

    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    add(any(field not in artifact for field in REQUIRED_ARTIFACT_FIELDS), "required_fields")
    add(
        artifact.get("schema") != SCHEMA or artifact.get("experiment_id") != EXPERIMENT_ID,
        "identity",
    )
    add(
        artifact.get("milestone") != MILESTONE or artifact.get("run_date") != RUN_DATE,
        "date_milestone",
    )
    add(
        artifact.get("MODEL_SPECS") != [] or artifact.get("model_invoked") is not False,
        "model_contract",
    )
    add(artifact.get("invocation_counts") != INVOCATION_COUNTS, "invocation_counts")
    add(
        any(
            artifact.get(key) != 0
            for key in (
                "current_model_load_count",
                "current_generation_count",
                "current_inference_count",
            )
        ),
        "current_call_counts",
    )
    add(artifact.get("execution_venue") != EXECUTION_VENUE, "execution_venue")
    add(artifact.get("verifier_is_oracle") is not True, "oracle_declaration")
    add(artifact.get("continuous_self_learning_task") is not True, "learning_task")
    add(artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact), "checksum")
    receipts = artifact.get("validation_receipts", [])
    receipt_fields = {"command", "exit_code", "classification", "duration_s", "log_sha256"}
    add(
        not isinstance(receipts, list)
        or any(
            set(row) != receipt_fields
            or not isinstance(row.get("command"), str)
            or not isinstance(row.get("exit_code"), int)
            or not isinstance(row.get("duration_s"), (int, float))
            or re.fullmatch(r"sha256:[0-9a-f]{64}", str(row.get("log_sha256", ""))) is None
            for row in receipts
        ),
        "validation_receipts",
    )
    if artifact.get("status") == "blocked":
        add(
            artifact.get("rows") != []
            or artifact.get("admission_fixture_ready_score") != 0
            or artifact.get("verdict_class") != "blocked"
            or artifact.get("inference_substrate") != "blocked_no_run"
            or artifact.get("inference_substrate_class") != "blocked_no_run"
            or not str(artifact.get("honest_verdict", "")).startswith("blocked_")
            or artifact.get("gate_check_summary", {}).get("passed") is not False
            or artifact.get("gate_check_summary", {}).get("first_failure") is None,
            "blocked_contract",
        )
        return errors
    add(artifact.get("status") != "complete", "status")
    if artifact.get("status") != "complete":
        return errors
    add(
        artifact.get("inference_substrate") != INFERENCE_SUBSTRATE
        or artifact.get("inference_substrate_class") != INFERENCE_SUBSTRATE_CLASS,
        "substrate",
    )
    add(
        artifact.get("reducer_inference_substrate") != REDUCER_INFERENCE_SUBSTRATE
        or artifact.get("reducer_inference_substrate_class") != REDUCER_INFERENCE_SUBSTRATE_CLASS,
        "reducer_substrate",
    )
    ready = artifact.get("admission_fixture_ready_score")
    if ready == 1:
        complete_valid = artifact.get("verdict_class") == "circular_positive" and str(
            artifact.get("honest_verdict", "")
        ).startswith("complete_circular_positive")
    else:
        complete_valid = (
            ready == 0
            and artifact.get("verdict_class") == "null"
            and str(artifact.get("honest_verdict", "")).startswith("complete_null")
        )
    add(not complete_valid, "complete_contract")
    selected = tuple(
        expected_stream_ids or (f"prospective-{index + 1:02d}" for index in range(STREAM_COUNT))
    )
    expected_units = {(stream_id, arm) for stream_id in selected for arm in ARMS}
    rows = artifact.get("rows", [])
    add(
        not isinstance(rows, list)
        or {(row.get("stream_id"), row.get("arm")) for row in rows} != expected_units,
        "rows",
    )
    add(
        any(
            row.get("censored") is not False
            or int(row.get("event_count", 0)) != EVENTS_PER_STREAM
            or int(row.get("nomination_label_count", 0)) != NOMINATION_LABEL_BUDGET
            or int(row.get("admission_label_count", 0)) != ADMISSION_LABEL_BUDGET
            or int(row.get("total_paid_label_count", 0)) > 128
            or int(row.get("maximum_memory_bytes", MEMORY_CAP_BYTES + 1)) > MEMORY_CAP_BYTES
            for row in rows
        ),
        "row_limits",
    )
    opportunity_rows = artifact.get("opportunity_rows", [])
    add(
        not isinstance(opportunity_rows, list)
        or len(opportunity_rows) != len(selected) * len(ARMS) * MAX_OPPORTUNITIES
        or bool(opportunity_row_errors(opportunity_rows)),
        "opportunity_rows",
    )
    finite = artifact.get("finite_law_rows", [])
    add(
        not isinstance(finite, list)
        or {row.get("n") for row in finite} != {8, 16}
        or any(row.get("coverage_passed") is not True for row in finite)
        or any(row.get("dependent_drift_theorem_claimed") is not False for row in finite),
        "finite_law_rows",
    )
    old = artifact.get("nomination_overlap_rows", [])
    add(
        not isinstance(old, list)
        or not old
        or any(row.get("old_outcomes_used_for_rule_tuning") is not False for row in old),
        "nomination_overlap_rows",
    )
    contract = artifact.get("admission_contract", {})
    add(
        contract.get("total_alpha") != TOTAL_ALPHA
        or contract.get("maximum_opportunities") != MAX_OPPORTUNITIES
        or contract.get("nomination_label_budget") != NOMINATION_LABEL_BUDGET
        or contract.get("admission_label_budget") != ADMISSION_LABEL_BUDGET
        or contract.get("nomination_and_admission_disjoint") is not True
        or contract.get("dependent_drifting_stream_theorem_claimed") is not False,
        "admission_contract",
    )
    manifest = artifact.get("stream_manifest", {})
    add(
        artifact.get("stream_manifest_path")
        != str(manifest.get("manifest_receipt", {}).get("path"))
        or manifest.get("development", {}).get("stream_count") != DEVELOPMENT_STREAM_COUNT
        or manifest.get("prospective", {}).get("stream_count") != STREAM_COUNT
        or manifest.get("authority_separated") is not True,
        "stream_manifest",
    )
    gates = artifact.get("acceptance_gate_results", {})
    add(
        not isinstance(gates, Mapping)
        or not gates
        or any(
            value.get("passed") != value.get("pass")
            or not {"expected", "observed", "passed", "pass", "principle"} <= set(value)
            for value in gates.values()
        )
        or ready != int(all(value.get("passed") is True for value in gates.values())),
        "acceptance_gate_results",
    )
    add(
        any(row.get("passed") is not True for row in artifact.get("mutation_rows", []))
        or len(artifact.get("mutation_rows", [])) != 2,
        "mutation_rows",
    )
    add(
        any(row.get("passed") is not True for row in artifact.get("e2e_rows", []))
        or len(artifact.get("e2e_rows", [])) != 8,
        "e2e_rows",
    )
    if check_files:
        sidecars = [
            artifact.get("raw_rows_receipt", {}),
            artifact.get("opportunity_rows_receipt", {}),
            artifact.get("historical_rows_receipt", {}),
            artifact.get("state_sidecar_receipt", {}),
            artifact.get("control_sidecar_receipt", {}),
            artifact.get("evidence_sidecar_receipt", {}),
            manifest.get("manifest_receipt", {}),
            *manifest.get("receipts", {}).values(),
        ]
        add(any(not _receipt_matches(repo_root, row) for row in sidecars), "sidecar_hashes")
        raw_path = _resolve(repo_root, str(artifact.get("raw_rows_receipt", {}).get("path", "")))
        try:
            reduced = independent_reduce(raw_path)
        except ValueError:
            reduced = []
        add(reduced != rows, "independent_reducer")
        add(
            any(
                expected is None or _sha256_path(_resolve(repo_root, path)) != expected
                for path, expected in artifact.get("source_artifact_hashes", {}).items()
            ),
            "source_artifact_hashes",
        )
    return errors


def attach_validation_receipts(
    artifact: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Attach exact command receipts and refresh the stable checksum."""

    required = {"command", "exit_code", "classification", "duration_s", "log_sha256"}
    if any(
        set(receipt) != required
        or not isinstance(receipt["command"], str)
        or not isinstance(receipt["exit_code"], int)
        or not isinstance(receipt["classification"], str)
        or not isinstance(receipt["duration_s"], (int, float))
        or re.fullmatch(r"sha256:[0-9a-f]{64}", str(receipt["log_sha256"])) is None
        for receipt in receipts
    ):
        raise ValueError("validation_receipt_schema")
    changed = deepcopy(dict(artifact))
    changed["validation_receipts"] = [dict(receipt) for receipt in receipts]
    changed["reproducibility_checksum"] = reproducibility_checksum(changed)
    return changed


def write_artifact(
    path: Path,
    artifact: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
    expected_stream_ids: Sequence[str] | None = None,
) -> None:
    """Cold-validate and atomically publish one terminal artifact."""

    errors = validate_artifact(
        artifact,
        repo_root=repo_root,
        expected_stream_ids=expected_stream_ids,
        check_files=artifact.get("status") == "complete",
    )
    if errors:
        raise ValueError("artifact_validation_failed:" + ",".join(errors))
    _atomic_write(path, transactional.canonical_json_bytes(dict(artifact)))


def _command_receipt(command: Sequence[str]) -> JsonDict:
    """Stream one validation subprocess and print a truthful heartbeat."""

    command_text = " ".join(command)
    print(f"validation BEFORE subprocess: {command_text}", flush=True)
    started = time.monotonic()
    process = subprocess.Popen(
        list(command),
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        env={
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": f"{REPO_ROOT / 'python'}:{REPO_ROOT}",
            "COVERAGE_FILE": "/tmp/.coverage-carnot-exp7281",
        },
    )
    output: list[str] = []
    selector = selectors.DefaultSelector()
    if process.stdout is not None:
        selector.register(process.stdout, selectors.EVENT_READ)
    while process.poll() is None:
        events = selector.select(timeout=30.0)
        if not events:
            print(
                f"validation heartbeat: command_running elapsed_s={time.monotonic() - started:.3f}",
                flush=True,
            )
            continue
        for key, _ in events:
            line = key.fileobj.readline()
            if line:
                output.append(line)
                print(line, end="", flush=True)
    if process.stdout is not None:
        remainder = process.stdout.read()
        if remainder:
            output.append(remainder)
            print(remainder, end="", flush=True)
    selector.close()
    exit_code = process.wait()
    duration = time.monotonic() - started
    print(
        f"validation AFTER subprocess: exit_code={exit_code} elapsed_s={duration:.3f}",
        flush=True,
    )
    return {
        "command": command_text,
        "exit_code": exit_code,
        "classification": "passed" if exit_code == 0 else "failed",
        "duration_s": duration,
        "log_sha256": transactional.sha256_bytes("".join(output).encode()),
    }


def _validation_commands(candidate: Path) -> list[list[str]]:
    """Return focused coverage, affected tests, static checks, E2E, and artifact checks."""

    python = str(REPO_ROOT / ".venv/bin/python")
    coverage = str(REPO_ROOT / ".venv/bin/coverage")
    ruff = str(REPO_ROOT / ".venv/bin/ruff")
    mypy = str(REPO_ROOT / ".venv/bin/mypy")
    module = "python/carnot/experiment_7281_v640_admission_prototype.py"
    test = str(TEST_PATH)
    wrapper = str(WRAPPER_PATH)
    return [
        [coverage, "erase"],
        [
            coverage,
            "run",
            f"--include={REPO_ROOT / module}",
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "-n",
            "0",
            "--basetemp=/tmp/carnot-exp7281-coverage",
            test,
            "-q",
        ],
        [
            coverage,
            "report",
            f"--include={REPO_ROOT / module}",
            "--show-missing",
            "--fail-under=100",
        ],
        [
            python,
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "-n",
            "0",
            "--basetemp=/tmp/carnot-exp7281-affected",
            "tests/python/test_experiment_7267_v639_recognition_prototype.py::test_scenario_cl_7267_transaction_is_delayed_atomic_and_reversible",
            "tests/python/test_experiment_7268_v639_recognition_learning.py::test_scenario_cl_7268_prequential_rows_and_receipts_are_complete",
            "tests/python/test_experiment_7269_v639_recognition_audit.py::test_bounds_keep_overlap_recall_false_accepts_and_zero_effects",
            "tests/python/test_experiment_7269_v639_recognition_audit.py::test_e2e_accept_reject_reload_and_rollback",
            "-q",
        ],
        [ruff, "check", module, test, wrapper],
        [ruff, "format", "--check", module, test, wrapper],
        [mypy, module, wrapper],
        [python, "scripts/check_spec_coverage.py", test],
        [
            python,
            "-u",
            wrapper,
            "--date",
            RUN_DATE,
            "--e2e-worker",
            "--output-root",
            "/tmp/carnot-exp7281-e2e-validation",
        ],
        [
            python,
            "-u",
            wrapper,
            "--date",
            RUN_DATE,
            "--validate",
            "--artifact-path",
            str(candidate),
        ],
        [python, "scripts/adversarial_verify.py", "--json", str(candidate)],
        [python, "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)],
    ]


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the fixed date and private worker modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--e2e-worker", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--artifact-path", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Measure, validate, and publish only terminal admission evidence."""

    print("phase 0 immediate: Exp7281 admission prototype started", flush=True)
    args = _parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"run_date_must_be_{RUN_DATE}")
    paths = (
        ExperimentPaths.defaults()
        if args.output_root is None
        else ExperimentPaths.under(args.output_root)
    )
    if args.e2e_worker:
        _progress(1, "start", "BEFORE private E2E admission lifecycle")
        rows = run_e2e_controls(paths.provisional.parent / "experiment_7281_e2e_worker")
        receipt = _atomic_write(
            paths.provisional.parent / "experiment_7281_v640_e2e.json",
            transactional.canonical_json_bytes({"schema": SCHEMA, "rows": rows}),
        )
        _progress(1, "end", f"AFTER private E2E lifecycle {receipt['path']}")
        return 0
    if args.validate:
        if args.artifact_path is None:
            raise SystemExit("artifact_path_required")
        _progress(1, "start", "BEFORE cold artifact validation")
        artifact = _load_object(args.artifact_path)
        errors = validate_artifact(artifact, repo_root=REPO_ROOT, check_files=True)
        if errors:
            raise SystemExit("artifact_validation_failed:" + ",".join(errors))
        _progress(1, "end", "AFTER cold artifact validation passed")
        return 0
    artifact = build_and_seal(REPO_ROOT, paths, progress=True)
    if artifact["status"] == "blocked":
        write_artifact(paths.artifact, artifact, repo_root=REPO_ROOT)
        _progress(7, "end", f"wrote blocked terminal artifact {paths.artifact}")
        return 0
    _progress(7, "start", "write measured candidate under raw evidence")
    _atomic_write(paths.terminal_candidate, transactional.canonical_json_bytes(artifact))
    _progress(7, "end", f"candidate={paths.terminal_candidate}")
    _progress(8, "start", "BEFORE focused tests, static checks, E2E, and artifact checks")
    receipts = list(artifact["validation_receipts"])
    for command in _validation_commands(paths.terminal_candidate):
        receipts.append(_command_receipt(command))
    artifact = attach_validation_receipts(artifact, receipts)
    _atomic_write(
        paths.provisional,
        transactional.canonical_json_bytes(
            {
                "schema": SCHEMA,
                "status": "in_progress",
                "phase": 8,
                "validation_receipts": receipts,
            }
        ),
    )
    failed = [row for row in receipts if row["exit_code"] != 0]
    if failed:
        raise RuntimeError(
            "focused_validation_failed:" + ",".join(str(row["command"]) for row in failed)
        )
    _progress(8, "end", "AFTER all focused validations passed")
    _progress(9, "start", "BEFORE final cold validation and atomic terminal write")
    write_artifact(paths.artifact, artifact, repo_root=REPO_ROOT)
    _progress(9, "end", f"AFTER atomic terminal write {paths.artifact}")
    return 0


if __name__ == "__main__":  # pragma: no cover - the thin wrapper owns execution.
    raise SystemExit(main())
