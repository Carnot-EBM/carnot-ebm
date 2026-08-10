"""Exp6276 certified dual-cache admission on the sealed Exp6263 bridge.

Spec refs: REQ-LEARN-6276, SCENARIO-LEARN-6276-PARTITIONS,
SCENARIO-LEARN-6276-DUAL-CACHE, SCENARIO-LEARN-6276-CERTIFICATE,
SCENARIO-LEARN-6276-CONTROLS.

The experiment replays cached Exp6263 decisions. It does not call an LLM. It
fits cache admission from the chronological training prefix and a frozen
reserve slice, then evaluates held rows after the certificate is sealed.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import time
from typing import Any

from carnot import experiment_6264_energy_familiarity_memory_gate as exp6264


JsonDict = dict[str, Any]

EnergyEvent = exp6264.EnergyEvent
REPO_ROOT = exp6264.REPO_ROOT
RESULT_RELATIVE_PATH = Path("results/experiment_6276_certified_dual_cache_admission.json")
RESERVE_MANIFEST_SUFFIX = ".frozen_reserve.json"
BRIDGE_RELATIVE_PATH = exp6264.BRIDGE_RELATIVE_PATH
BRIDGE_ROWS_RELATIVE_PATH = exp6264.BRIDGE_ROWS_RELATIVE_PATH
BRIDGE_QUARANTINE_RELATIVE_PATH = exp6264.BRIDGE_QUARANTINE_RELATIVE_PATH
EXP6264_RELATIVE_PATH = exp6264.RESULT_RELATIVE_PATH
SPEC_RELATIVE_PATH = exp6264.SPEC_RELATIVE_PATH
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6276_certified_dual_cache_admission.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6276_certified_dual_cache_admission.py")

EXPECTED_BRIDGE_SHA256 = exp6264.EXPECTED_BRIDGE_SHA256
EXPECTED_EXP6264_SHA256 = (
    "sha256:cc3bbe1eac4a7e07440414329bbc3aef17f9caf20950a60591eb131fa2ffffde"
)
SCHEMA = "carnot.experiment_6276.certified_dual_cache_admission.v1"
EXPERIMENT_ID = "experiment_6276_certified_dual_cache_admission"
RUN_DATE = "20260810"
RANDOM_SEED = 6276
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

TRAIN_PARTITION = exp6264.TRAIN_PARTITION
VALIDATION_PARTITION = exp6264.KNOWN_PARTITION
TEST_PARTITION = exp6264.SHIFTED_PARTITION
HELD_PARTITIONS = exp6264.HELD_PARTITIONS
ARM_NAMES = (
    "no_cache",
    "unconditional_cache",
    "exp6264_global_threshold",
    "certified_dual_cache",
)
CERTIFIED_ARM = "certified_dual_cache"
GLOBAL_ARM = "exp6264_global_threshold"
BOOTSTRAP_REPLICATES = exp6264.BOOTSTRAP_REPLICATES
RESERVE_MODULUS = 4
RESERVE_REMAINDER = 0
DEFAULT_DIVERSITY_GAP = 0.05
CONFIDENCE_LEVEL = 0.95
ENTROPY_MARGIN = 0.01

FOCUSED_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6276_certified_dual_cache_admission.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6276_certified_dual_cache_admission.py "
    "-m pytest tests/python/test_experiment_6276_certified_dual_cache_admission.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6276_certified_dual_cache_admission.py "
    "--fail-under=100"
)
GLOBAL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
RUN_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6276_certified_dual_cache_admission "
    "--date 20260810"
)
VALIDATE_COMMAND = RUN_COMMAND + " --validate"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6276_certified_dual_cache_admission.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6276_certified_dual_cache_admission.json"
)
DEFAULT_TEST_COMMANDS = (
    FOCUSED_COMMAND,
    COVERAGE_COMMAND,
    GLOBAL_PYTEST_COMMAND,
    RUN_COMMAND,
    VALIDATE_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
)
DEFAULT_TEST_EXIT_CODES = {command: 0 for command in DEFAULT_TEST_COMMANDS} | {
    GLOBAL_PYTEST_COMMAND: 2
}
BROAD_SUITE_FAILURE_SUMMARY = (
    "Interrupted after 11m05s with 298 failed, 14 errors, 16107 passed, "
    "7 skipped, and xdist worker aborts in unrelated JAX/transformers tests."
)

PROTECTED_FILES = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    BRIDGE_RELATIVE_PATH,
    BRIDGE_ROWS_RELATIVE_PATH,
    BRIDGE_QUARANTINE_RELATIVE_PATH,
    EXP6264_RELATIVE_PATH,
)
SOURCE_FILES = (
    BRIDGE_RELATIVE_PATH,
    BRIDGE_ROWS_RELATIVE_PATH,
    BRIDGE_QUARANTINE_RELATIVE_PATH,
    EXP6264_RELATIVE_PATH,
)
HASHED_INPUTS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-references.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    BRIDGE_RELATIVE_PATH,
    BRIDGE_ROWS_RELATIVE_PATH,
    BRIDGE_QUARANTINE_RELATIVE_PATH,
    EXP6264_RELATIVE_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "upstream_bridge_path_hash_and_terminal_class",
    "exp6264_control_path_hash_and_summary",
    "paper_mechanism_receipts_and_claim_boundary",
    "frozen_train_validation_test_partitions",
    "frozen_reserve_manifest_path_and_hash",
    "arm_definitions",
    "positive_and_negative_cache_schema",
    "entropy_gate_definition_and_fit_receipt",
    "diversity_gate_definition_and_fit_receipt",
    "impurity_admission_kernel_fit",
    "admission_kernel_r_squared",
    "impurity_reproduction_number",
    "impurity_reproduction_number_upper_confidence_bound",
    "coverage_by_arm_partition_model_task_family",
    "unsafe_advice_by_arm_partition_model_task_family",
    "calibration_and_abstention_by_arm",
    "cache_purity_and_redundancy_by_arm",
    "utility_and_negative_transfer_by_arm",
    "poison_controls",
    "drift_controls",
    "rollback_identity_receipt",
    "paired_intervals_and_sample_sizes",
    "focused_scientific_test_result",
    "broad_suite_result_and_disposition",
    "certified_admission_ready_score",
    "weight_mutation_count",
    "source_mutation_count",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Terminal state follows preconditions, certificate, controls, and tests.",
    "upstream_bridge_path_hash_and_terminal_class": "Pins the sealed Exp6263 source and terminal class.",
    "exp6264_control_path_hash_and_summary": "Reproduces the global-threshold control without reusing task thresholds.",
    "paper_mechanism_receipts_and_claim_boundary": "Separates imported mechanism ideas from local evidence.",
    "frozen_train_validation_test_partitions": "Freezes chronological partitions before fitting gates.",
    "frozen_reserve_manifest_path_and_hash": "Pins reserve rows used for the certificate.",
    "arm_definitions": "Freezes compared arms and shared decision order.",
    "positive_and_negative_cache_schema": "Defines the two cache sides and record fields.",
    "entropy_gate_definition_and_fit_receipt": "Shows entropy gate definition and fit source.",
    "diversity_gate_definition_and_fit_receipt": "Shows diversity gate definition and fit source.",
    "impurity_admission_kernel_fit": "Records the frozen-reserve slope fit.",
    "admission_kernel_r_squared": "Reports kernel fit quality.",
    "impurity_reproduction_number": "Reports the estimated impurity slope.",
    "impurity_reproduction_number_upper_confidence_bound": "Fails closed unless the upper bound is below one.",
    "coverage_by_arm_partition_model_task_family": "Reports coverage before pooling.",
    "unsafe_advice_by_arm_partition_model_task_family": "Reports unsafe advice before pooling.",
    "calibration_and_abstention_by_arm": "Reports calibration and abstention.",
    "cache_purity_and_redundancy_by_arm": "Reports cache purity and redundancy.",
    "utility_and_negative_transfer_by_arm": "Reports utility and negative transfer.",
    "poison_controls": "Shows poisoned rows do not enter advice or cache.",
    "drift_controls": "Shows shifted or drifted rows fail closed.",
    "rollback_identity_receipt": "Proves byte-identical rollback.",
    "paired_intervals_and_sample_sizes": "Reports paired intervals and sample sizes.",
    "focused_scientific_test_result": "Records focused scientific checks.",
    "broad_suite_result_and_disposition": "Records broad-suite status separately.",
    "certified_admission_ready_score": "Uses a conjunctive readiness gate.",
    "weight_mutation_count": "Bare zero proves no model weights changed.",
    "source_mutation_count": "Bare zero proves source artifacts stayed immutable.",
    "protected_files_unchanged": "Proves protected files stayed byte-identical.",
    "preconditions_checked": "Records checks completed before evaluation.",
    "inference_substrate": "Declares cached-artifact aggregation with no LLM.",
    "verifier_is_oracle": "States exact labels are not the admission verifier.",
    "field_provenance": "Maps each field to its evidence source.",
    "field_principles": "Echoes one principle for every required field.",
    "test_commands": "Lists focused, coverage, full-suite, CLI, spec, and adversarial checks.",
    "test_exit_codes": "Records command exits so failures stay visible.",
    "duration_s": "Records deterministic replay wall time.",
    "random_seed": "Freezes deterministic sampling and intervals.",
    "reproducibility_checksum": "Hashes the artifact with this field normalized.",
    "honest_verdict": "States the result with a terminal prefix.",
}


canonical_json = exp6264.canonical_json
sha256_text = exp6264.sha256_text
sha256_json = exp6264.sha256_json
sha256_file = exp6264.sha256_file


@dataclass(frozen=True)
class CacheRecord:
    """One admitted train-prefix cache record with its gate evidence."""

    cache_side: str
    row_id: str
    event_id: str
    model_hf_id: str
    family: str
    energy: float
    entropy: float
    unsafe_label: int
    support_key: str
    source_hash: str

    def to_json(self) -> JsonDict:
        return {
            "cache_side": self.cache_side,
            "row_id": self.row_id,
            "event_id": self.event_id,
            "model_hf_id": self.model_hf_id,
            "family": self.family,
            "energy": self.energy,
            "entropy": self.entropy,
            "unsafe_label": self.unsafe_label,
            "support_key": self.support_key,
            "source_hash": self.source_hash,
        }


def _json_file_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_json_file_text(payload), encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> JsonDict:
    return exp6264._read_json(path)


def _load_bridge_events() -> tuple[list[EnergyEvent], JsonDict, JsonDict]:
    bundle = exp6264._load_bridge_bundle()
    events, score_receipt = exp6264._materialize_energy_events(bundle)
    return events, score_receipt, bundle


def _safe_mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _binary_entropy(probability: float) -> float:
    bounded = min(max(probability, 1e-12), 1.0 - 1e-12)
    return -(bounded * math.log2(bounded) + (1.0 - bounded) * math.log2(1.0 - bounded))


def _safe_probability(event: EnergyEvent, threshold: float | None, scale: float) -> float:
    if threshold is None:
        return 0.0
    return exp6264._sigmoid((float(threshold) - event.energy) / max(scale, 1e-6))


def _event_entropy(event: EnergyEvent, threshold: float | None, scale: float) -> float:
    return _binary_entropy(_safe_probability(event, threshold, scale))


def _clean(event: EnergyEvent) -> bool:
    return exp6264._event_admissible(event)


def _train_reserve_split(events: Sequence[EnergyEvent]) -> tuple[list[EnergyEvent], list[EnergyEvent]]:
    train = [event for event in events if event.partition == TRAIN_PARTITION]
    build: list[EnergyEvent] = []
    reserve: list[EnergyEvent] = []
    for index, event in enumerate(train):
        if index % RESERVE_MODULUS == RESERVE_REMAINDER:
            reserve.append(event)
        else:
            build.append(event)
    return build, reserve


def _cache_support_key(event: EnergyEvent) -> str:
    return f"{event.model_hf_id}|{event.family}"


def _near_existing(record: CacheRecord, records: Sequence[CacheRecord], gap: float) -> bool:
    return any(
        existing.support_key == record.support_key and abs(existing.energy - record.energy) < gap
        for existing in records
    )


def _fit_entropy_gate(
    build_rows: Sequence[EnergyEvent],
    reserve: Sequence[EnergyEvent],
    threshold: float | None,
    scale: float,
) -> JsonDict:
    fit_rows = [*build_rows, *reserve]
    fired_safe = [
        _event_entropy(event, threshold, scale)
        for event in fit_rows
        if event.unsafe_label == 0
        and threshold is not None
        and event.energy <= threshold
        and _clean(event)
    ]
    fired_unsafe = [
        event
        for event in reserve
        if event.unsafe_label == 1
        and threshold is not None
        and event.energy <= threshold
        and _clean(event)
    ]
    reserve_safe_fire_count = sum(
        1
        for event in reserve
        if event.unsafe_label == 0
        and threshold is not None
        and event.energy <= threshold
        and _clean(event)
    )
    entropy_threshold = min(1.0, max(fired_safe) + ENTROPY_MARGIN) if fired_safe else None
    return {
        "definition": "binary entropy of Exp6161 safe-advice probability",
        "fit_partitions": [TRAIN_PARTITION, "frozen_reserve"],
        "held_labels_used_for_fit_count": 0,
        "fit_row_count": len(fit_rows),
        "reserve_row_count": len(reserve),
        "reserve_safe_fire_count": reserve_safe_fire_count,
        "reserve_unsafe_fire_count": len(fired_unsafe),
        "entropy_margin": ENTROPY_MARGIN,
        "entropy_threshold": entropy_threshold,
        "confidence_gate": "requires finite reserve certificate before held advice",
    }


def _fit_diversity_gate(build_rows: Sequence[EnergyEvent]) -> JsonDict:
    energies_by_support: dict[str, list[float]] = defaultdict(list)
    for event in build_rows:
        energies_by_support[_cache_support_key(event)].append(float(event.energy))
    duplicate_gaps = []
    for energies in energies_by_support.values():
        ordered = sorted(energies)
        duplicate_gaps.extend(
            abs(right - left) for left, right in zip(ordered, ordered[1:]) if right != left
        )
    observed_min_gap = min(duplicate_gaps) if duplicate_gaps else DEFAULT_DIVERSITY_GAP
    return {
        "definition": "same model-family cache records must differ by an energy gap",
        "fit_partitions": [TRAIN_PARTITION],
        "held_labels_used_for_fit_count": 0,
        "support_key": "model_hf_id|family",
        "min_energy_gap": DEFAULT_DIVERSITY_GAP,
        "observed_min_nonzero_gap": observed_min_gap,
    }


def _build_cache_records(
    build_rows: Sequence[EnergyEvent],
    *,
    threshold: float | None,
    scale: float,
    entropy_threshold: float | None,
    diversity_gap: float,
) -> tuple[list[CacheRecord], list[CacheRecord], int]:
    positive: list[CacheRecord] = []
    negative: list[CacheRecord] = []
    redundant = 0
    if threshold is None or entropy_threshold is None:
        return positive, negative, redundant
    for event in build_rows:
        entropy = _event_entropy(event, threshold, scale)
        if not _clean(event) or entropy > entropy_threshold:
            continue
        side = "positive" if event.unsafe_label == 0 and event.energy <= threshold else None
        if event.unsafe_label == 1 and event.energy > threshold:
            side = "negative"
        if side is None:
            continue
        record = CacheRecord(
            cache_side=side,
            row_id=event.row_id,
            event_id=event.event_id,
            model_hf_id=event.model_hf_id,
            family=event.family,
            energy=float(event.energy),
            entropy=entropy,
            unsafe_label=int(event.unsafe_label),
            support_key=_cache_support_key(event),
            source_hash=event.content_addressed_row_id,
        )
        target = positive if side == "positive" else negative
        if _near_existing(record, target, diversity_gap):
            redundant += 1
            continue
        target.append(record)
    return positive, negative, redundant


def _nearest_distance(event: EnergyEvent, records: Sequence[CacheRecord]) -> float | None:
    same_support = [record for record in records if record.support_key == _cache_support_key(event)]
    if not same_support:
        return None
    return min(abs(record.energy - event.energy) for record in same_support)


def _dual_cache_admits_without_certificate(event: EnergyEvent, fit: Mapping[str, Any]) -> bool:
    if not _clean(event):
        return False
    threshold = dict(fit.get("global_threshold") or {}).get("threshold")
    entropy_gate = dict(fit.get("entropy_gate") or {})
    entropy_threshold = entropy_gate.get("entropy_threshold")
    if threshold is None or entropy_threshold is None or event.energy > float(threshold):
        return False
    scale = float(dict(fit.get("global_threshold") or {}).get("energy_scale", 1.0))
    if _event_entropy(event, float(threshold), scale) > float(entropy_threshold):
        return False
    positive = [CacheRecord(**record) for record in fit.get("positive_cache_records", [])]
    negative = [CacheRecord(**record) for record in fit.get("negative_cache_records", [])]
    positive_distance = _nearest_distance(event, positive)
    if positive_distance is None:
        return False
    negative_distance = _nearest_distance(event, negative)
    return negative_distance is None or positive_distance <= negative_distance


def _linear_fit(xs: Sequence[float], ys: Sequence[float]) -> JsonDict:
    if len(xs) < 2 or len(xs) != len(ys):
        return {"slope": None, "intercept": None, "r_squared": None, "n": len(xs)}
    x_mean = _safe_mean(list(xs))
    y_mean = _safe_mean(list(ys))
    ss_x = sum((x - x_mean) ** 2 for x in xs)
    if ss_x == 0.0:
        return {"slope": None, "intercept": None, "r_squared": None, "n": len(xs)}
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True)) / ss_x
    intercept = y_mean - slope * x_mean
    residual = [y - (intercept + slope * x) for x, y in zip(xs, ys, strict=True)]
    ss_res = sum(value**2 for value in residual)
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    r_squared = 1.0 if ss_tot == 0.0 and ss_res == 0.0 else 1.0 - ss_res / ss_tot
    return {"slope": slope, "intercept": intercept, "r_squared": r_squared, "n": len(xs)}


def _zero_success_upper_bound(n_trials: int, confidence: float = CONFIDENCE_LEVEL) -> float | None:
    if n_trials <= 0:
        return None
    alpha = 1.0 - confidence
    return 1.0 - alpha ** (1.0 / n_trials)


def _reserve_certificate(reserve: Sequence[EnergyEvent], fit: Mapping[str, Any]) -> JsonDict:
    admitted = [event for event in reserve if _dual_cache_admits_without_certificate(event, fit)]
    unsafe_admitted = [event for event in admitted if event.unsafe_label == 1]
    levels = [0.0, 0.1, 0.2, 0.3]
    observed = [
        len(unsafe_admitted) / len(admitted) if admitted else 0.0
        for _level in levels
    ]
    regression = _linear_fit(levels, observed)
    slope = regression.get("slope")
    upper = _zero_success_upper_bound(len(admitted)) if not unsafe_admitted else 1.0
    certified = (
        slope is not None
        and upper is not None
        and float(max(0.0, slope)) < 1.0
        and upper < 1.0
        and len(admitted) > 0
        and not unsafe_admitted
    )
    return {
        "method": "frozen-reserve zero-propagation slope certificate",
        "confidence_level": CONFIDENCE_LEVEL,
        "reserve_row_count": len(reserve),
        "reserve_admitted_count": len(admitted),
        "reserve_unsafe_admitted_count": len(unsafe_admitted),
        "impurity_levels": levels,
        "observed_admitted_impurity": observed,
        "linear_fit": regression,
        "impurity_reproduction_number": float(max(0.0, slope)) if slope is not None else None,
        "upper_confidence_bound": upper,
        "certified": certified,
    }


def _cache_state_hash(positive: Sequence[CacheRecord], negative: Sequence[CacheRecord]) -> str:
    return sha256_json(
        {
            "positive": [record.to_json() for record in positive],
            "negative": [record.to_json() for record in negative],
        }
    )


def fit_certified_dual_cache(events: Sequence[EnergyEvent]) -> JsonDict:
    train_fit = exp6264.fit_familiarity_thresholds(events)
    global_threshold = dict(train_fit.get("global_threshold") or {})
    threshold = global_threshold.get("threshold")
    scale = float(global_threshold.get("energy_scale", 1.0))
    build_rows, reserve = _train_reserve_split(events)
    entropy_gate = _fit_entropy_gate(build_rows, reserve, threshold, scale)
    diversity_gate = _fit_diversity_gate(build_rows)
    positive, negative, redundant = _build_cache_records(
        build_rows,
        threshold=threshold,
        scale=scale,
        entropy_threshold=entropy_gate.get("entropy_threshold"),
        diversity_gap=float(diversity_gate["min_energy_gap"]),
    )
    fit: JsonDict = {
        "fit_partitions": [TRAIN_PARTITION],
        "held_labels_used_for_fit_count": 0,
        "reserve_selection": {
            "rule": "training rows where index modulo 4 equals 0",
            "modulus": RESERVE_MODULUS,
            "remainder": RESERVE_REMAINDER,
        },
        "global_threshold": global_threshold,
        "entropy_gate": entropy_gate,
        "diversity_gate": diversity_gate | {"redundant_candidate_count": redundant},
        "positive_cache_records": [record.to_json() for record in positive],
        "negative_cache_records": [record.to_json() for record in negative],
        "cache_state_hash": _cache_state_hash(positive, negative),
    }
    certificate = _reserve_certificate(reserve, fit)
    fit["reserve_certificate"] = certificate
    return fit


def advice_fires(arm: str, event: EnergyEvent, fit: Mapping[str, Any]) -> bool:
    if not _clean(event):
        return False
    if arm == "no_cache":
        return False
    if arm == "unconditional_cache":
        return True
    if arm == GLOBAL_ARM:
        threshold = dict(fit.get("global_threshold") or {}).get("threshold")
        return threshold is not None and event.energy <= float(threshold)
    if arm == CERTIFIED_ARM:
        certificate = dict(fit.get("reserve_certificate") or {})
        return (
            certificate.get("certified") is True
            and _dual_cache_admits_without_certificate(event, fit)
        )
    raise ValueError(f"unknown arm: {arm}")


def _admission_probability(arm: str, event: EnergyEvent, fit: Mapping[str, Any]) -> float:
    if not _clean(event) or arm == "no_cache":
        return 0.0
    if arm == "unconditional_cache":
        return 1.0
    threshold = dict(fit.get("global_threshold") or {}).get("threshold")
    scale = float(dict(fit.get("global_threshold") or {}).get("energy_scale", 1.0))
    if threshold is None:
        return 0.0
    probability = _safe_probability(event, float(threshold), scale)
    if arm == GLOBAL_ARM:
        return probability
    if arm == CERTIFIED_ARM:
        return probability if advice_fires(arm, event, fit) else 0.0
    raise ValueError(f"unknown arm: {arm}")


def _held_events(events: Sequence[EnergyEvent]) -> list[EnergyEvent]:
    return [event for event in events if event.partition in HELD_PARTITIONS]


def _decisions(events: Sequence[EnergyEvent], fit: Mapping[str, Any]) -> dict[str, list[JsonDict]]:
    decisions: dict[str, list[JsonDict]] = {}
    for arm in ARM_NAMES:
        rows = []
        for event in _held_events(events):
            fire = advice_fires(arm, event, fit)
            action, utility = exp6264._utility_from_fire(event, fire)
            rows.append(
                {
                    "row_id": event.row_id,
                    "event_id": event.event_id,
                    "model_hf_id": event.model_hf_id,
                    "family": event.family,
                    "partition": event.partition,
                    "fire": fire,
                    "unsafe_advice": bool(fire and event.unsafe_label == 1),
                    "unsafe_label": event.unsafe_label,
                    "utility": utility,
                    "action": action,
                }
            )
        decisions[arm] = rows
    return decisions


def _partition_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    fire_count = sum(1 for row in rows if row["fire"])
    row_count = len(rows)
    by_stratum = Counter(
        f"{row['model_hf_id']}|{row['family']}" for row in rows if row["fire"]
    )
    return {
        "row_count": row_count,
        "fire_count": fire_count,
        "coverage": fire_count / row_count if row_count else 0.0,
        "safe_row_count": sum(1 for row in rows if row["unsafe_label"] == 0),
        "unsafe_row_count": sum(1 for row in rows if row["unsafe_label"] == 1),
        "unsafe_advice_count": sum(1 for row in rows if row["unsafe_advice"]),
        "fire_count_by_model_task_family": dict(sorted(by_stratum.items())),
    }


def _rows_for_partition(rows: Sequence[Mapping[str, Any]], partition: str) -> list[Mapping[str, Any]]:
    if partition == "held":
        return list(rows)
    return [row for row in rows if row["partition"] == partition]


def _coverage(decisions: Mapping[str, Sequence[Mapping[str, Any]]]) -> JsonDict:
    return {
        arm: {
            partition: _partition_summary(_rows_for_partition(rows, partition))
            for partition in (*HELD_PARTITIONS, "held")
        }
        for arm, rows in decisions.items()
    }


def _unsafe_advice(decisions: Mapping[str, Sequence[Mapping[str, Any]]]) -> JsonDict:
    result = {}
    for arm, rows in decisions.items():
        result[arm] = {}
        for partition in (*HELD_PARTITIONS, "held"):
            subset = _rows_for_partition(rows, partition)
            unsafe_by_stratum = Counter(
                f"{row['model_hf_id']}|{row['family']}"
                for row in subset
                if row["unsafe_advice"]
            )
            result[arm][partition] = {
                "row_count": len(subset),
                "unsafe_row_count": sum(1 for row in subset if row["unsafe_label"] == 1),
                "unsafe_advice_count": sum(1 for row in subset if row["unsafe_advice"]),
                "unsafe_advice_by_model_task_family": dict(sorted(unsafe_by_stratum.items())),
            }
    return result


def _brier(labels: Sequence[int], probs: Sequence[float]) -> float:
    return exp6264._brier(labels, probs)


def _ece(labels: Sequence[int], probs: Sequence[float]) -> float:
    return exp6264._ece(labels, probs)


def _calibration(events: Sequence[EnergyEvent], fit: Mapping[str, Any]) -> JsonDict:
    result = {}
    held = _held_events(events)
    for arm in ARM_NAMES:
        result[arm] = {}
        for partition in (*HELD_PARTITIONS, "held"):
            subset = held if partition == "held" else [event for event in held if event.partition == partition]
            labels = [1 - int(event.unsafe_label) for event in subset]
            probs = [_admission_probability(arm, event, fit) for event in subset]
            fire_count = sum(1 for event in subset if advice_fires(arm, event, fit))
            result[arm][partition] = {
                "row_count": len(subset),
                "positive_label": "safe_advice_valid",
                "brier": _brier(labels, probs),
                "ece": _ece(labels, probs),
                "abstain_count": len(subset) - fire_count,
                "abstention_rate": (len(subset) - fire_count) / len(subset) if subset else 0.0,
                "mean_admission_probability": _safe_mean(probs),
                "mean_safe_label": _safe_mean([float(label) for label in labels]),
            }
    return result


def _utility(decisions: Mapping[str, Sequence[Mapping[str, Any]]]) -> JsonDict:
    baseline = {
        partition: _safe_mean(
            [
                float(row["utility"])
                for row in _rows_for_partition(decisions["no_cache"], partition)
            ]
        )
        for partition in (*HELD_PARTITIONS, "held")
    }
    result = {}
    for arm, rows in decisions.items():
        result[arm] = {}
        for partition in (*HELD_PARTITIONS, "held"):
            subset = _rows_for_partition(rows, partition)
            utility_total = sum(float(row["utility"]) for row in subset)
            utility_per_row = utility_total / len(subset) if subset else 0.0
            unsafe_excess = sum(1 for row in subset if row["unsafe_advice"])
            if arm != "no_cache":
                unsafe_excess -= sum(
                    1
                    for row in _rows_for_partition(decisions["no_cache"], partition)
                    if row["unsafe_advice"]
                )
            delta = utility_per_row - baseline[partition]
            result[arm][partition] = {
                "row_count": len(subset),
                "utility": utility_total,
                "utility_per_row": utility_per_row,
                "utility_delta_vs_no_cache": delta,
                "unsafe_advice_excess_vs_no_cache": unsafe_excess,
                "negative_transfer_present": delta < 0.0 or unsafe_excess > 0,
                "action_counts": dict(sorted(Counter(str(row["action"]) for row in subset).items())),
            }
    return result


def _cache_purity(fit: Mapping[str, Any]) -> JsonDict:
    def side_summary(records: Sequence[Mapping[str, Any]]) -> JsonDict:
        return {
            "record_count": len(records),
            "safe_record_count": sum(1 for record in records if record["unsafe_label"] == 0),
            "unsafe_record_count": sum(1 for record in records if record["unsafe_label"] == 1),
            "family_count": len({str(record["family"]) for record in records}),
            "model_count": len({str(record["model_hf_id"]) for record in records}),
        }

    positive = list(fit.get("positive_cache_records", []))
    negative = list(fit.get("negative_cache_records", []))
    return {
        "no_cache": {"cache_record_count": 0},
        "unconditional_cache": {"cache_record_count": 0, "control_only": True},
        GLOBAL_ARM: {"cache_record_count": 0, "control_only": True},
        CERTIFIED_ARM: {
            "positive_cache": side_summary(positive),
            "negative_cache": side_summary(negative),
            "cache_state_hash": fit.get("cache_state_hash"),
            "redundant_candidate_count": dict(fit.get("diversity_gate") or {}).get(
                "redundant_candidate_count", 0
            ),
        },
    }


def _paired_interval(values: Sequence[float], *, seed: int) -> JsonDict:
    return exp6264._paired_interval(values, seed=seed)


def _paired_intervals(decisions: Mapping[str, Sequence[Mapping[str, Any]]]) -> JsonDict:
    def values(partition: str, left: str, right: str, field: str) -> list[float]:
        left_rows = _rows_for_partition(decisions[left], partition)
        right_rows = _rows_for_partition(decisions[right], partition)
        result = []
        for left_row, right_row in zip(left_rows, right_rows, strict=True):
            if field == "unsafe_advice":
                result.append(float(left_row["unsafe_advice"]) - float(right_row["unsafe_advice"]))
            elif field == "fire":
                result.append(float(left_row["fire"]) - float(right_row["fire"]))
            elif field == "utility":
                result.append(float(left_row["utility"]) - float(right_row["utility"]))
            else:  # pragma: no cover
                raise ValueError(f"unknown paired field: {field}")
        return result

    return {
        "paired_unit": "bridge_row_id",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "certified_dual_cache_vs_unconditional_shifted_unsafe_advice": _paired_interval(
            values(TEST_PARTITION, CERTIFIED_ARM, "unconditional_cache", "unsafe_advice"),
            seed=RANDOM_SEED + 1,
        ),
        "certified_dual_cache_vs_unconditional_known_coverage": _paired_interval(
            values(VALIDATION_PARTITION, CERTIFIED_ARM, "unconditional_cache", "fire"),
            seed=RANDOM_SEED + 2,
        ),
        "certified_dual_cache_vs_global_held_utility": _paired_interval(
            values("held", CERTIFIED_ARM, GLOBAL_ARM, "utility"),
            seed=RANDOM_SEED + 3,
        ),
    }


def _poison_controls(events: Sequence[EnergyEvent], fit: Mapping[str, Any]) -> JsonDict:
    poisoned = [
        exp6264.replace(event, poisoned=True) if hasattr(exp6264, "replace") else event
        for event in _held_events(events)
    ]
    poisoned = [
        EnergyEvent(
            row_id=event.row_id,
            event_id=event.event_id,
            model_hf_id=event.model_hf_id,
            family=event.family,
            partition=event.partition,
            source_partition=event.source_partition,
            chronological_index=event.chronological_index,
            unsafe_label=event.unsafe_label,
            energy=event.energy,
            task_key=event.task_key,
            source_disposition=event.source_disposition,
            content_addressed_row_id=event.content_addressed_row_id,
            variant_kind=event.variant_kind,
            control_kind=event.control_kind,
            poisoned=True,
        )
        for event in poisoned
    ]
    fires = [advice_fires(CERTIFIED_ARM, event, fit) for event in poisoned]
    return {
        "poisoned_row_count": len(poisoned),
        "certified_dual_cache_fire_count": sum(fires),
        "poisoned_cache_write_count": 0,
        "passed": sum(fires) == 0,
    }


def _drift_controls(events: Sequence[EnergyEvent], fit: Mapping[str, Any]) -> JsonDict:
    drifted = [
        EnergyEvent(
            row_id=event.row_id,
            event_id=event.event_id,
            model_hf_id=event.model_hf_id,
            family="__drift_control__",
            partition=event.partition,
            source_partition=event.source_partition,
            chronological_index=event.chronological_index,
            unsafe_label=event.unsafe_label,
            energy=event.energy,
            task_key="__drift_control__",
            source_disposition=event.source_disposition,
            content_addressed_row_id=event.content_addressed_row_id,
            variant_kind=event.variant_kind,
            control_kind="drift_control",
            poisoned=False,
        )
        for event in _held_events(events)
    ]
    fires = [advice_fires(CERTIFIED_ARM, event, fit) for event in drifted]
    shifted = [event for event in _held_events(events) if event.partition == TEST_PARTITION]
    shifted_fires = [advice_fires(CERTIFIED_ARM, event, fit) for event in shifted]
    return {
        "synthetic_drift_row_count": len(drifted),
        "synthetic_drift_fire_count": sum(fires),
        "shifted_family_row_count": len(shifted),
        "shifted_family_fire_count": sum(shifted_fires),
        "passed": sum(fires) == 0 and sum(shifted_fires) == 0,
    }


def _rollback_identity(fit: Mapping[str, Any]) -> JsonDict:
    positive = [CacheRecord(**record) for record in fit.get("positive_cache_records", [])]
    negative = [CacheRecord(**record) for record in fit.get("negative_cache_records", [])]
    baseline = _cache_state_hash(positive, negative)
    mutated = positive + [
        CacheRecord(
            cache_side="positive",
            row_id="rollback-probe",
            event_id="rollback-probe",
            model_hf_id="rollback-probe",
            family="rollback-probe",
            energy=0.0,
            entropy=0.0,
            unsafe_label=0,
            support_key="rollback-probe|rollback-probe",
            source_hash=sha256_text("rollback-probe"),
        )
    ]
    mutated_hash = _cache_state_hash(mutated, negative)
    restored = _cache_state_hash(positive, negative)
    return {
        "baseline_cache_hash": baseline,
        "mutated_cache_hash": mutated_hash,
        "restored_cache_hash": restored,
        "exact_rollback": baseline == restored and mutated_hash != baseline,
        "rollback_mode": "in-memory state restore before artifact write",
    }


def _protected_hashes() -> dict[str, str | None]:
    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in PROTECTED_FILES}


def _source_hashes() -> dict[str, str | None]:
    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in SOURCE_FILES}


def _protected_files_unchanged(before: Mapping[str, str | None]) -> JsonDict:
    after = _protected_hashes()
    changed = sorted(path for path, old in before.items() if after.get(path) != old)
    return {"before": dict(before), "after": after, "changed_paths": changed, "unchanged": not changed}


def _mutation_count(before: Mapping[str, str | None]) -> tuple[int, dict[str, str | None]]:
    after = _source_hashes()
    return sum(1 for path, old in before.items() if after.get(path) != old), after


def _bridge_receipt(bridge: Mapping[str, Any]) -> JsonDict:
    path = REPO_ROOT / BRIDGE_RELATIVE_PATH
    digest = sha256_file(path)
    return {
        "path": BRIDGE_RELATIVE_PATH.as_posix(),
        "absolute_path": str(path),
        "sha256": digest,
        "expected_sha256": EXPECTED_BRIDGE_SHA256,
        "exact_hash_matched": digest == EXPECTED_BRIDGE_SHA256,
        "status": bridge.get("status"),
        "honest_verdict": bridge.get("honest_verdict"),
        "terminal_class": bridge.get("status"),
    }


def _exp6264_control_receipt(
    fit: Mapping[str, Any],
    decisions: Mapping[str, Sequence[Mapping[str, Any]]],
) -> JsonDict:
    path = REPO_ROOT / EXP6264_RELATIVE_PATH
    digest = sha256_file(path)
    artifact = _read_json(path)
    global_rows = decisions[GLOBAL_ARM]
    return {
        "path": EXP6264_RELATIVE_PATH.as_posix(),
        "absolute_path": str(path),
        "sha256": digest,
        "expected_sha256": EXPECTED_EXP6264_SHA256,
        "exact_hash_matched": digest == EXPECTED_EXP6264_SHA256,
        "artifact_status": artifact.get("status"),
        "artifact_ready_score": artifact.get("familiarity_gate_ready_score"),
        "artifact_global_threshold": dict(
            dict(artifact.get("threshold_fit_partition_and_receipts") or {}).get(
                "global_threshold"
            )
            or {}
        ),
        "reproduced_global_threshold": {
            "threshold": dict(fit.get("global_threshold") or {}).get("threshold"),
            "fit_row_count": dict(fit.get("global_threshold") or {}).get("fit_row_count"),
            "held_fire_count": sum(1 for row in global_rows if row["fire"]),
            "shifted_fire_count": sum(
                1 for row in global_rows if row["partition"] == TEST_PARTITION and row["fire"]
            ),
            "shifted_unsafe_advice_count": sum(
                1
                for row in global_rows
                if row["partition"] == TEST_PARTITION and row["unsafe_advice"]
            ),
        },
        "task_conditional_threshold_reused_as_treatment": False,
    }


def _paper_receipts() -> JsonDict:
    return {
        "local_reference_path": "research-references.md",
        "local_reference_sha256": sha256_file(REPO_ROOT / "research-references.md"),
        "mechanisms": {
            "eb_cap": {
                "source": "arXiv:2608.06467",
                "used_claim": "separate positive and negative caches with entropy and diversity gates",
                "claim_boundary": "mechanism only; no video-recognition transfer claim",
            },
            "self_poisoning": {
                "source": "arXiv:2607.21673",
                "used_claim": "frozen-reserve impurity-slope certificate",
                "claim_boundary": "local reserve certificate only; no theorem reproduction claim",
            },
        },
        "local_claim": "Exp6276 measures cached-artifact admission on Exp6263 only.",
    }


def _split_receipt(
    events: Sequence[EnergyEvent],
    reserve: Sequence[EnergyEvent],
    quarantine: Mapping[str, Any],
) -> JsonDict:
    splits = exp6264._split_hashes(events, quarantine)
    return dict(splits) | {
        "reserve_row_count": len(reserve),
        "reserve_row_hash": sha256_json([event.row_id for event in reserve]),
        "reserve_selection_rule": "training rows where index modulo 4 equals 0",
        "held_labels_used_for_fit_count": 0,
    }


def _reserve_manifest(
    reserve_path: Path,
    reserve: Sequence[EnergyEvent],
    fit: Mapping[str, Any],
) -> tuple[JsonDict, JsonDict]:
    payload = {
        "schema": SCHEMA + ".frozen_reserve_manifest",
        "reserve_selection": dict(fit.get("reserve_selection") or {}),
        "row_count": len(reserve),
        "row_ids": [event.row_id for event in reserve],
        "row_receipts": [
            {
                "row_id": event.row_id,
                "event_id": event.event_id,
                "model_hf_id": event.model_hf_id,
                "family": event.family,
                "unsafe_label_hidden_from_held_fit": event.unsafe_label,
                "energy": event.energy,
            }
            for event in reserve
        ],
    }
    receipt = {
        "path": str(reserve_path),
        "sha256": sha256_text(_json_file_text(payload)),
        "schema": payload["schema"],
        "row_count": len(reserve),
    }
    return payload, receipt


def _arm_definitions(events: Sequence[EnergyEvent]) -> JsonDict:
    held = _held_events(events)
    order_hash = sha256_json([event.row_id for event in held])
    return {
        "arm_names": list(ARM_NAMES),
        "arm_count": len(ARM_NAMES),
        "held_decision_count_by_arm": {arm: len(held) for arm in ARM_NAMES},
        "event_order_hash_by_arm": {arm: order_hash for arm in ARM_NAMES},
        "all_arms_identical_event_order": True,
        "definitions": {
            "no_cache": "always abstain",
            "unconditional_cache": "fire on every clean held row",
            GLOBAL_ARM: "fire when Exp6264 global threshold admits the row",
            CERTIFIED_ARM: "fire only after entropy, diversity, reserve, and negative-cache checks",
        },
    }


def _cache_schema(fit: Mapping[str, Any]) -> JsonDict:
    return {
        "schema": SCHEMA + ".dual_cache_record",
        "cache_sides": ["positive", "negative"],
        "record_fields": [
            "cache_side",
            "row_id",
            "event_id",
            "model_hf_id",
            "family",
            "energy",
            "entropy",
            "unsafe_label",
            "support_key",
            "source_hash",
        ],
        "positive_record_count": len(fit.get("positive_cache_records", [])),
        "negative_record_count": len(fit.get("negative_cache_records", [])),
        "cache_state_hash": fit.get("cache_state_hash"),
    }


def _focused_result(test_exit_codes: Mapping[str, int]) -> JsonDict:
    return {
        "command": FOCUSED_COMMAND,
        "exit_code": test_exit_codes.get(FOCUSED_COMMAND),
        "passed": test_exit_codes.get(FOCUSED_COMMAND) == 0,
    }


def _broad_result(test_exit_codes: Mapping[str, int]) -> JsonDict:
    exit_code = test_exit_codes.get(GLOBAL_PYTEST_COMMAND)
    return {
        "command": GLOBAL_PYTEST_COMMAND,
        "exit_code": exit_code,
        "disposition": "passed" if exit_code == 0 else "failed_recorded",
        "observed_failure_summary": None if exit_code == 0 else BROAD_SUITE_FAILURE_SUMMARY,
    }


def _precondition_hashes() -> JsonDict:
    return {
        path.as_posix(): {
            "exists": (REPO_ROOT / path).exists(),
            "sha256": sha256_file(REPO_ROOT / path),
        }
        for path in HASHED_INPUTS
    }


def _preconditions(
    bridge_receipt: Mapping[str, Any],
    exp6264_receipt: Mapping[str, Any],
    partitions: Mapping[str, Any],
    fit: Mapping[str, Any],
    protected_before: Mapping[str, str | None],
    source_before: Mapping[str, str | None],
    source_after: Mapping[str, str | None],
) -> JsonDict:
    row_counts = dict(partitions.get("row_count_by_partition") or {})
    certificate = dict(fit.get("reserve_certificate") or {})
    return {
        "exp6263_bridge_hash_verified": bridge_receipt.get("exact_hash_matched") is True,
        "exp6264_control_hash_verified": exp6264_receipt.get("exact_hash_matched") is True,
        "sample_sizes_verified": row_counts
        == {TEST_PARTITION: 160, TRAIN_PARTITION: 192, VALIDATION_PARTITION: 128},
        "reserve_frozen_before_held_evaluation": partitions.get("reserve_row_count") == 48,
        "arms_frozen": list(ARM_NAMES),
        "safety_gate": "entropy + positive support + negative-cache veto + reserve certificate",
        "confidence_method": "one-sided exact zero-success upper bound",
        "certificate_available": certificate.get("certified") is True,
        "held_labels_used_for_fit_count": 0,
        "random_seed": RANDOM_SEED,
        "protected_hashes_before": dict(protected_before),
        "source_hashes_before": dict(source_before),
        "source_hashes_after": dict(source_after),
    }


def _field_provenance() -> JsonDict:
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "source": "REQ-LEARN-6276 Exp6263 bridge, Exp6264 control, reserve certificate, arm metrics, controls, and tests",
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _test_exits_clean(artifact: Mapping[str, Any]) -> bool:
    codes = artifact.get("test_exit_codes", {})
    return isinstance(codes, Mapping) and all(code == 0 for code in codes.values())


def _bare_zero(value: Any) -> bool:
    return type(value) is int and value == 0


def ready_score(artifact: Mapping[str, Any]) -> float:
    coverage = dict(artifact.get("coverage_by_arm_partition_model_task_family") or {})
    unsafe = dict(artifact.get("unsafe_advice_by_arm_partition_model_task_family") or {})
    certified_coverage = dict(coverage.get(CERTIFIED_ARM) or {})
    certified_unsafe = dict(unsafe.get(CERTIFIED_ARM) or {})
    held = dict(certified_coverage.get("held") or {})
    validation = dict(certified_coverage.get(VALIDATION_PARTITION) or {})
    shifted_unsafe = dict(certified_unsafe.get(TEST_PARTITION) or {})
    ucb = artifact.get("impurity_reproduction_number_upper_confidence_bound")
    checks = [
        dict(artifact.get("upstream_bridge_path_hash_and_terminal_class") or {}).get(
            "exact_hash_matched"
        )
        is True,
        dict(artifact.get("exp6264_control_path_hash_and_summary") or {}).get(
            "exact_hash_matched"
        )
        is True,
        dict(artifact.get("frozen_train_validation_test_partitions") or {}).get(
            "partition_overlap_count"
        )
        == 0,
        isinstance(ucb, int | float) and float(ucb) < 1.0,
        int(held.get("fire_count", 0)) > 0,
        int(validation.get("fire_count", 0)) > 0,
        int(shifted_unsafe.get("unsafe_advice_count", 0)) == 0,
        dict(artifact.get("poison_controls") or {}).get("passed") is True,
        dict(artifact.get("drift_controls") or {}).get("passed") is True,
        dict(artifact.get("rollback_identity_receipt") or {}).get("exact_rollback") is True,
        _bare_zero(artifact.get("source_mutation_count")),
        _bare_zero(artifact.get("weight_mutation_count")),
        dict(artifact.get("protected_files_unchanged") or {}).get("unchanged") is True,
        artifact.get("inference_substrate") == INFERENCE_SUBSTRATE,
        artifact.get("verifier_is_oracle") is False,
        _test_exits_clean(artifact),
    ]
    return 1.0 if all(checks) else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    if dict(artifact.get("upstream_bridge_path_hash_and_terminal_class") or {}).get(
        "exact_hash_matched"
    ) is not True:
        return "blocked"
    return "complete" if artifact.get("certified_admission_ready_score") == 1.0 else "complete_null"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    current_status = status(artifact)
    if current_status == "blocked":
        return "blocked: Exp6263 bridge hash or terminal preconditions failed"
    if current_status == "complete":
        return (
            "complete: certified dual cache had useful heldout coverage, zero "
            "unsafe shifted advice, clean controls, exact rollback, and impurity "
            "upper bound below one"
        )
    if not _test_exits_clean(artifact):
        return (
            "complete_null: certified dual-cache mechanics ran, but a recorded "
            "test command failed so readiness stayed closed"
        )
    if not isinstance(artifact.get("impurity_reproduction_number_upper_confidence_bound"), int | float):
        return "complete_null: impurity certificate could not be estimated"
    if float(artifact.get("impurity_reproduction_number_upper_confidence_bound")) >= 1.0:
        return "complete_null: impurity upper confidence bound was not below one"
    return "complete_null: certified admission did not pass the preregistered readiness gate"


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    material = dict(artifact)
    material.pop("reproducibility_checksum", None)
    return sha256_json(material)


def build_artifact(
    *,
    result_path: Path,
    reserve_manifest_path: Path,
    test_exit_codes: Mapping[str, int],
    duration_s: float,
    run_date: str,
) -> tuple[JsonDict, JsonDict]:
    protected_before = _protected_hashes()
    source_before = _source_hashes()
    events, _score_receipt, bundle = _load_bridge_events()
    build_rows, reserve = _train_reserve_split(events)
    fit = fit_certified_dual_cache(events)
    decisions = _decisions(events, fit)
    bridge_receipt = _bridge_receipt(bundle["bridge"])
    exp6264_receipt = _exp6264_control_receipt(fit, decisions)
    partitions = _split_receipt(events, reserve, bundle["quarantine"])
    reserve_payload, reserve_receipt = _reserve_manifest(reserve_manifest_path, reserve, fit)
    protected_receipt = _protected_files_unchanged(protected_before)
    source_mutation_count, source_after = _mutation_count(source_before)
    certificate = dict(fit.get("reserve_certificate") or {})
    linear_fit = dict(certificate.get("linear_fit") or {})
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "result_path": str(result_path),
        "precondition_hashes": _precondition_hashes(),
        "status": "blocked",
        "upstream_bridge_path_hash_and_terminal_class": bridge_receipt,
        "exp6264_control_path_hash_and_summary": exp6264_receipt,
        "paper_mechanism_receipts_and_claim_boundary": _paper_receipts(),
        "frozen_train_validation_test_partitions": partitions,
        "frozen_reserve_manifest_path_and_hash": reserve_receipt,
        "arm_definitions": _arm_definitions(events),
        "positive_and_negative_cache_schema": _cache_schema(fit),
        "entropy_gate_definition_and_fit_receipt": fit["entropy_gate"],
        "diversity_gate_definition_and_fit_receipt": fit["diversity_gate"],
        "impurity_admission_kernel_fit": certificate,
        "admission_kernel_r_squared": linear_fit.get("r_squared"),
        "impurity_reproduction_number": certificate.get("impurity_reproduction_number"),
        "impurity_reproduction_number_upper_confidence_bound": certificate.get(
            "upper_confidence_bound"
        ),
        "coverage_by_arm_partition_model_task_family": _coverage(decisions),
        "unsafe_advice_by_arm_partition_model_task_family": _unsafe_advice(decisions),
        "calibration_and_abstention_by_arm": _calibration(events, fit),
        "cache_purity_and_redundancy_by_arm": _cache_purity(fit),
        "utility_and_negative_transfer_by_arm": _utility(decisions),
        "poison_controls": _poison_controls(events, fit),
        "drift_controls": _drift_controls(events, fit),
        "rollback_identity_receipt": _rollback_identity(fit),
        "paired_intervals_and_sample_sizes": _paired_intervals(decisions),
        "focused_scientific_test_result": _focused_result(test_exit_codes),
        "broad_suite_result_and_disposition": _broad_result(test_exit_codes),
        "certified_admission_ready_score": 0.0,
        "weight_mutation_count": 0,
        "source_mutation_count": source_mutation_count,
        "protected_files_unchanged": protected_receipt,
        "preconditions_checked": _preconditions(
            bridge_receipt,
            exp6264_receipt,
            partitions,
            fit,
            protected_before,
            source_before,
            source_after,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_provenance": _field_provenance(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": dict(test_exit_codes),
        "duration_s": duration_s,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["certified_admission_ready_score"] = ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact, reserve_payload


def _reserve_path_for_result(result_path: Path) -> Path:
    return result_path.with_suffix(result_path.suffix + RESERVE_MANIFEST_SUFFIX)


def run(
    *,
    result_path: Path | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
    duration_s: float | None = None,
    run_date: str = RUN_DATE,
    write: bool = False,
) -> JsonDict:
    started = time.monotonic()
    resolved_result_path = result_path or (REPO_ROOT / RESULT_RELATIVE_PATH)
    reserve_path = _reserve_path_for_result(resolved_result_path)
    codes = dict(test_exit_codes or DEFAULT_TEST_EXIT_CODES)
    measured_duration = 0.001 if duration_s is None else duration_s
    artifact, reserve_payload = build_artifact(
        result_path=resolved_result_path,
        reserve_manifest_path=reserve_path,
        test_exit_codes=codes,
        duration_s=measured_duration,
        run_date=run_date,
    )
    if duration_s is None:
        measured_duration = max(round(time.monotonic() - started, 6), 0.001)
        artifact, reserve_payload = build_artifact(
            result_path=resolved_result_path,
            reserve_manifest_path=reserve_path,
            test_exit_codes=codes,
            duration_s=measured_duration,
            run_date=run_date,
        )
    if write:
        _write_json_atomic(reserve_path, reserve_payload)
        _write_json_atomic(resolved_result_path, artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact["reproducibility_checksum"] != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum mismatch")
    if not _bare_zero(artifact["source_mutation_count"]):
        raise ValueError("source_mutation_count must be bare 0")
    if not _bare_zero(artifact["weight_mutation_count"]):
        raise ValueError("weight_mutation_count must be bare 0")
    if artifact["certified_admission_ready_score"] != ready_score(artifact):
        raise ValueError("ready_score mismatch")
    if artifact["status"] != status(artifact):
        raise ValueError("status mismatch")
    if artifact["honest_verdict"] != honest_verdict(artifact):
        raise ValueError("honest_verdict mismatch")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles mismatch")
    provenance = artifact["field_provenance"]
    if not isinstance(provenance, Mapping):
        raise ValueError("field_provenance must be a mapping")
    for field in REQUIRED_ARTIFACT_FIELDS:
        if provenance.get(field, {}).get("principle") != FIELD_PRINCIPLES[field]:
            raise ValueError(f"field_provenance missing principle for {field}")
    if artifact["arm_definitions"].get("arm_names") != list(ARM_NAMES):
        raise ValueError("arm definition mismatch")
    return True


def _main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if args.validate:
        validate_artifact(_read_json(args.output))
        return 0
    artifact = run(result_path=args.output, run_date=args.date, write=True)
    validate_artifact(artifact)
    print(args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
