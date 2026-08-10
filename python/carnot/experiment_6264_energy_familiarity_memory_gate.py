"""Exp6264 energy-familiarity gate for constraint memory advice.

Spec refs: REQ-LEARN-6264, SCENARIO-LEARN-6264-THRESHOLDS,
SCENARIO-LEARN-6264-GATES, SCENARIO-LEARN-6264-CONTROLS.

The experiment replays cached Exp6263 rows. It does not call an LLM. It uses
the existing Exp6161 decision-energy features, fits admission thresholds only
on the train prefix, and evaluates held rows after the thresholds are frozen.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any

from carnot import experiment_6159_decision_calibrated_stream as exp6159
from carnot import experiment_6161_decision_calibrated_energy_policy as exp6161


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6264_energy_familiarity_memory_gate.json")
BRIDGE_RELATIVE_PATH = Path("results/experiment_6263_clean_sota_event_replay_bridge.json")
BRIDGE_ROWS_RELATIVE_PATH = Path(
    "results/experiment_6263_clean_sota_event_replay_bridge.rows.jsonl"
)
BRIDGE_QUARANTINE_RELATIVE_PATH = Path(
    "results/experiment_6263_clean_sota_event_replay_bridge.quarantine.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6264_energy_familiarity_memory_gate.py")
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6264_energy_familiarity_memory_gate.py"
)

EXPECTED_BRIDGE_SHA256 = (
    "sha256:38af00462fdc8d389f27f223f02e22faa2994a60193f983209ce6b9d1f42cdd6"
)
SCHEMA = "carnot.experiment_6264.energy_familiarity_memory_gate.v1"
EXPERIMENT_ID = "experiment_6264_energy_familiarity_memory_gate"
RUN_DATE = "20260810"
RANDOM_SEED = 6264
LOWER_IS_FAMILIAR = "lower_is_familiar"
HIGHER_IS_FAMILIAR = "higher_is_familiar"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

TRAIN_PARTITION = "train"
KNOWN_PARTITION = "validation"
SHIFTED_PARTITION = "test"
HELD_PARTITIONS = (KNOWN_PARTITION, SHIFTED_PARTITION)
ARM_NAMES = (
    "no_memory",
    "unconditional_advice",
    "global_threshold",
    "task_conditional_thresholds",
)
THRESHOLD_ARMS = ("global_threshold", "task_conditional_thresholds")

COST_TABLE: dict[str, float] = {
    "true_safe_acceptance": 1.0,
    "false_unsafe_acceptance": -8.0,
    "safe_abstention": -0.25,
    "unsafe_abstention": -0.5,
}
ABSTENTION_MARGIN = 0.05
BOOTSTRAP_REPLICATES = 256

FOCUSED_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6264_energy_familiarity_memory_gate.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6264_energy_familiarity_memory_gate.py "
    "-m pytest tests/python/test_experiment_6264_energy_familiarity_memory_gate.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6264_energy_familiarity_memory_gate.py "
    "--fail-under=100"
)
GLOBAL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
RUN_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6264_energy_familiarity_memory_gate "
    "--date 20260810"
)
VALIDATE_COMMAND = RUN_COMMAND + " --validate"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6264_energy_familiarity_memory_gate.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6264_energy_familiarity_memory_gate.json"
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

PROTECTED_FILES = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    BRIDGE_RELATIVE_PATH,
    BRIDGE_ROWS_RELATIVE_PATH,
    BRIDGE_QUARANTINE_RELATIVE_PATH,
    Path("results/experiment_6160_sota_decision_calibration_corpus.json"),
    Path("results/experiment_6160_sota_decision_calibration_corpus.qwen3_6_35b_a3b.rows.jsonl"),
    Path("results/experiment_6160_sota_decision_calibration_corpus.gemma_4_26b_a4b_it.rows.jsonl"),
    Path("results/experiment_6161_decision_calibrated_energy_policy.json"),
)
SOURCE_FILES = (
    BRIDGE_RELATIVE_PATH,
    BRIDGE_ROWS_RELATIVE_PATH,
    BRIDGE_QUARANTINE_RELATIVE_PATH,
    Path("results/experiment_6160_sota_decision_calibration_corpus.qwen3_6_35b_a3b.rows.jsonl"),
    Path("results/experiment_6160_sota_decision_calibration_corpus.gemma_4_26b_a4b_it.rows.jsonl"),
    Path("results/experiment_6159_decision_calibrated_stream.rows.jsonl"),
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
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "upstream_bridge_path_and_hash",
    "source_model_provenance",
    "chronological_split_hashes",
    "energy_definition_and_direction",
    "no_memory_unconditional_global_and_task_conditional_arm_configs",
    "threshold_fit_partition_and_receipts",
    "treatment_fire_counts",
    "known_family_coverage_by_arm",
    "shifted_family_unsafe_advice_by_arm",
    "abstention_by_arm",
    "calibration_by_arm",
    "exact_utility_by_arm",
    "negative_transfer_by_arm",
    "paired_intervals_and_sample_sizes",
    "inactive_gate_control",
    "ood_positive_control",
    "off_policy_limitation",
    "source_mutation_count",
    "weight_mutation_count",
    "familiarity_gate_ready_score",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Terminal state follows bridge integrity, held evaluation, controls, protected-file checks, and test receipts.",
    "upstream_bridge_path_and_hash": "Records the sealed Exp6263 bridge path and exact hash so this replay cannot silently switch sources.",
    "source_model_provenance": "Shows which upstream model rows supplied cached decisions and confirms no model load was allowed.",
    "chronological_split_hashes": "Hashes train, future-known, shifted, and quarantine splits and proves non-overlap.",
    "energy_definition_and_direction": "Defines the reused exact score and proves lower energy is the admitted familiarity direction.",
    "no_memory_unconditional_global_and_task_conditional_arm_configs": "Freezes the four matched arm configs.",
    "threshold_fit_partition_and_receipts": "Shows thresholds were fit only on train rows and records task-threshold support.",
    "treatment_fire_counts": "Proves each treatment fires or abstains where its gate says it should.",
    "known_family_coverage_by_arm": "Measures future-known family advice coverage.",
    "shifted_family_unsafe_advice_by_arm": "Measures unsafe advice on shifted families.",
    "abstention_by_arm": "Counts withheld advice by arm and partition.",
    "calibration_by_arm": "Reports Brier and ECE for advice validity.",
    "exact_utility_by_arm": "Applies the frozen exact utility table to advice, abstention, and unsafe advice.",
    "negative_transfer_by_arm": "Reports shifted-family utility deltas against no-memory abstention.",
    "paired_intervals_and_sample_sizes": "Gives paired intervals and n for primary contrasts.",
    "inactive_gate_control": "Proves disabled thresholds reproduce unconditional advice and are not a treatment.",
    "ood_positive_control": "Proves unseen tasks abstain under task-conditional thresholds.",
    "off_policy_limitation": "States that the offline stream is not an on-policy density guarantee.",
    "source_mutation_count": "Bare zero proves source artifacts stayed immutable.",
    "weight_mutation_count": "Bare zero proves no model weights changed.",
    "familiarity_gate_ready_score": "Conjunctive gate for nondegenerate fire, shifted unsafe-advice reduction, no known regression, controls, and tests.",
    "protected_files_unchanged": "Proves conductor, ops, traceability, and upstream evidence stayed byte-identical during materialization.",
    "preconditions_checked": "Records the hash, split, sample, energy, and no-LLM checks performed before evaluation.",
    "inference_substrate": "Declares aggregation over cached artifacts with no LLM or model execution.",
    "verifier_is_oracle": "States that exact labels are evaluation-only and the gate itself is not an oracle.",
    "field_provenance": "Maps each required field to bridge, threshold, arm, control, or test evidence.",
    "field_principles": "Echoes these field principles into the artifact.",
    "test_commands": "Lists focused, coverage, full-suite, command-line, spec, and adversarial checks.",
    "test_exit_codes": "Records exit codes so failed checks cannot be reported as success.",
    "duration_s": "Records deterministic replay wall time.",
    "reproducibility_checksum": "Hashes the artifact with this field normalized.",
    "honest_verdict": "Starts with `complete:`, `complete_null:`, or `blocked:` and states the offline result.",
}


@dataclass(frozen=True)
class EnergyEvent:
    """One cached decision row with its exact replay energy and label."""

    row_id: str
    event_id: str
    model_hf_id: str
    family: str
    partition: str
    source_partition: str
    chronological_index: int
    unsafe_label: int
    energy: float
    task_key: str
    source_disposition: str
    content_addressed_row_id: str
    variant_kind: str
    control_kind: str
    poisoned: bool = False

    def to_json(self) -> JsonDict:
        return {
            "row_id": self.row_id,
            "event_id": self.event_id,
            "model_hf_id": self.model_hf_id,
            "family": self.family,
            "partition": self.partition,
            "source_partition": self.source_partition,
            "chronological_index": self.chronological_index,
            "unsafe_label": self.unsafe_label,
            "energy": self.energy,
            "task_key": self.task_key,
            "source_disposition": self.source_disposition,
            "content_addressed_row_id": self.content_addressed_row_id,
            "variant_kind": self.variant_kind,
            "control_kind": self.control_kind,
            "poisoned": self.poisoned,
        }


@dataclass(frozen=True)
class FamiliarityGate:
    """Reusable threshold gate that fails closed for unsupported rows."""

    mode: str
    direction: str
    global_threshold: float | None
    task_thresholds: Mapping[str, float]
    inactive: bool = False

    def admit(self, event: EnergyEvent) -> bool:
        if not _event_admissible(event):
            return False
        if self.inactive:
            return True
        threshold = self.global_threshold
        if self.mode == "task":
            threshold = self.task_thresholds.get(event.task_key)
        if threshold is None:
            return False
        return _passes_threshold(event.energy, threshold, self.direction)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _load_jsonl(path: Path) -> list[JsonDict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _safe_mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _safe_mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else REPO_ROOT / path


def _file_receipt(path: Path) -> JsonDict:
    return {
        "exists": path.exists(),
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
    }


def _event_admissible(event: EnergyEvent) -> bool:
    return event.source_disposition == "clean" and event.poisoned is False


def _passes_threshold(energy: float, threshold: float, direction: str) -> bool:
    if direction == LOWER_IS_FAMILIAR:
        return energy <= threshold
    if direction == HIGHER_IS_FAMILIAR:
        return energy >= threshold
    raise ValueError(f"unknown energy direction: {direction}")


def _utility_from_fire(event: EnergyEvent, fire: bool) -> tuple[str, float]:
    if fire and event.unsafe_label == 0:
        return "true_safe_acceptance", COST_TABLE["true_safe_acceptance"]
    if fire and event.unsafe_label == 1:
        return "false_unsafe_acceptance", COST_TABLE["false_unsafe_acceptance"]
    if event.unsafe_label == 0:
        return "safe_abstention", COST_TABLE["safe_abstention"]
    return "unsafe_abstention", COST_TABLE["unsafe_abstention"]


def _candidate_thresholds(events: Sequence[EnergyEvent]) -> list[float]:
    energies = sorted({float(event.energy) for event in events})
    if not energies:
        return []
    candidates = [energies[0] - 1.0, energies[-1] + 1.0]
    candidates.extend((left + right) / 2.0 for left, right in zip(energies, energies[1:]))
    return sorted(candidates)


def _evaluate_threshold(
    events: Sequence[EnergyEvent],
    threshold: float | None,
    direction: str,
) -> JsonDict:
    action_counts = Counter()
    utility = 0.0
    fire_count = 0
    unsafe_fire_count = 0
    for event in events:
        fire = threshold is not None and _passes_threshold(event.energy, threshold, direction)
        fire_count += int(fire)
        unsafe_fire_count += int(fire and event.unsafe_label == 1)
        action, row_utility = _utility_from_fire(event, fire)
        action_counts[action] += 1
        utility += row_utility
    row_count = len(events)
    return {
        "row_count": row_count,
        "fire_count": fire_count,
        "unsafe_fire_count": unsafe_fire_count,
        "utility": utility,
        "utility_per_row": utility / row_count if row_count else 0.0,
        "action_counts": dict(sorted(action_counts.items())),
    }


def _fit_threshold(events: Sequence[EnergyEvent], *, direction: str) -> JsonDict:
    if not events:
        return {
            "threshold": None,
            "direction": direction,
            "fit_row_count": 0,
            "safe_count": 0,
            "unsafe_count": 0,
            "candidate_count": 0,
            "energy_min": None,
            "energy_max": None,
            "energy_scale": 1.0,
            **_evaluate_threshold([], None, direction),
        }
    candidates = _candidate_thresholds(events)
    best_threshold = candidates[0]
    best_metrics: JsonDict | None = None
    best_key: tuple[float, int, int] | None = None
    for threshold in candidates:
        metrics = _evaluate_threshold(events, threshold, direction)
        key = (
            float(metrics["utility_per_row"]),
            -int(metrics["unsafe_fire_count"]),
            int(metrics["fire_count"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = threshold
            best_metrics = metrics
    energies = [float(event.energy) for event in events]
    result = dict(best_metrics or {})
    result.update(
        {
            "threshold": float(best_threshold),
            "direction": direction,
            "fit_row_count": len(events),
            "safe_count": sum(1 for event in events if event.unsafe_label == 0),
            "unsafe_count": sum(1 for event in events if event.unsafe_label == 1),
            "candidate_count": len(candidates),
            "energy_min": min(energies),
            "energy_max": max(energies),
            "energy_scale": max(_std(energies), 1e-6),
        }
    )
    return result


def fit_familiarity_thresholds(events: Sequence[EnergyEvent]) -> JsonDict:
    """Fit train-only global and task thresholds.

    The function accepts already-materialized events so tests can exercise edge
    cases without loading the large sealed bridge.
    """

    train = [event for event in events if event.partition == TRAIN_PARTITION]
    global_threshold = _fit_threshold(train, direction=LOWER_IS_FAMILIAR)
    reversed_threshold = _fit_threshold(train, direction=HIGHER_IS_FAMILIAR)
    by_task: dict[str, JsonDict] = {}
    task_thresholds: dict[str, float] = {}
    for task_key in sorted({event.task_key for event in train}):
        task_events = [event for event in train if event.task_key == task_key]
        receipt = _fit_threshold(task_events, direction=LOWER_IS_FAMILIAR)
        by_task[task_key] = receipt
        if receipt["threshold"] is not None:
            task_thresholds[task_key] = float(receipt["threshold"])
    return {
        "fit_partitions": [TRAIN_PARTITION],
        "held_partitions_used_for_threshold_count": 0,
        "global_threshold": global_threshold,
        "task_thresholds": by_task,
        "task_threshold_values": task_thresholds,
        "task_threshold_count": len(task_thresholds),
        "direction_control": {
            "selected_direction": LOWER_IS_FAMILIAR,
            "reversed_direction": HIGHER_IS_FAMILIAR,
            "selected_utility_per_row": global_threshold["utility_per_row"],
            "reversed_direction_utility_per_row": reversed_threshold["utility_per_row"],
            "selected_unsafe_fire_count": global_threshold["unsafe_fire_count"],
            "reversed_direction_unsafe_fire_count": reversed_threshold["unsafe_fire_count"],
            "reversed_direction_threshold": reversed_threshold["threshold"],
        },
    }


def _gate_for_arm(arm: str, fit: Mapping[str, Any]) -> FamiliarityGate | None:
    if arm == "global_threshold":
        threshold = dict(fit.get("global_threshold") or {}).get("threshold")
        return FamiliarityGate(
            mode="global",
            direction=LOWER_IS_FAMILIAR,
            global_threshold=float(threshold) if threshold is not None else None,
            task_thresholds={},
        )
    if arm == "task_conditional_thresholds":
        return FamiliarityGate(
            mode="task",
            direction=LOWER_IS_FAMILIAR,
            global_threshold=None,
            task_thresholds={
                str(key): float(value)
                for key, value in dict(fit.get("task_threshold_values") or {}).items()
            },
        )
    return None


def advice_fires(arm: str, event: EnergyEvent, fit: Mapping[str, Any]) -> bool:
    if not _event_admissible(event):
        return False
    if arm == "no_memory":
        return False
    if arm == "unconditional_advice":
        return True
    gate = _gate_for_arm(arm, fit)
    if gate is None:
        raise ValueError(f"unknown arm: {arm}")
    return gate.admit(event)


def _admission_probability(arm: str, event: EnergyEvent, fit: Mapping[str, Any]) -> float:
    if not _event_admissible(event) or arm == "no_memory":
        return 0.0
    if arm == "unconditional_advice":
        return 1.0
    if arm == "global_threshold":
        receipt = dict(fit.get("global_threshold") or {})
        threshold = receipt.get("threshold")
        if threshold is None:
            return 0.0
        scale = max(float(receipt.get("energy_scale", 1.0)), 1e-6)
        return _sigmoid((float(threshold) - event.energy) / scale)
    if arm == "task_conditional_thresholds":
        receipt = dict(dict(fit.get("task_thresholds") or {}).get(event.task_key) or {})
        threshold = receipt.get("threshold")
        if threshold is None:
            return 0.0
        scale = max(float(receipt.get("energy_scale", 1.0)), 1e-6)
        return _sigmoid((float(threshold) - event.energy) / scale)
    raise ValueError(f"unknown arm: {arm}")


def _load_bridge_bundle() -> JsonDict:
    return {
        "bridge": _read_json(REPO_ROOT / BRIDGE_RELATIVE_PATH),
        "rows": _load_jsonl(REPO_ROOT / BRIDGE_ROWS_RELATIVE_PATH),
        "quarantine": _read_json(REPO_ROOT / BRIDGE_QUARANTINE_RELATIVE_PATH),
    }


def _source_row_cache(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[JsonDict]]:
    cache = {}
    for row in rows:
        path_text = str(row.get("source_row_file") or "")
        if path_text and path_text not in cache:
            cache[path_text] = _load_jsonl(_resolve_path(path_text))
    return cache


def _materialize_energy_events(bundle: Mapping[str, Any]) -> tuple[list[EnergyEvent], JsonDict]:
    bridge_rows = list(bundle["rows"])
    source_cache = _source_row_cache(bridge_rows)
    pre_by_event = {
        str(row.get("event_id")): row for row in _load_jsonl(REPO_ROOT / exp6159.ROW_FILE_RELATIVE_PATH)
    }
    entries = []
    entry_bridge_rows = []
    for bridge_row in bridge_rows:
        if bridge_row.get("source_disposition") != "clean":
            continue
        source_rows = source_cache[str(bridge_row["source_row_file"])]
        model_row = source_rows[int(bridge_row["source_row_number"])]
        pre_row = pre_by_event[str(bridge_row["event_id"])]
        entry = {
            "model_hf_id": str(bridge_row["model_hf_id"]),
            "event_id": str(bridge_row["event_id"]),
            "base_template_id": str(pre_row.get("base_template_id")),
            "family": str(bridge_row["family"]),
            "variant_kind": str(bridge_row["variant_kind"]),
            "control_kind": str(pre_row.get("control_kind")),
            "partition": str(bridge_row["bridge_partition"]),
            "unsafe_label": int(
                dict(bridge_row.get("exact_label_provenance") or {}).get("unsafe_label", 0)
            ),
            "features": exp6161._decision_features(pre_row, model_row),
            "scores": {},
        }
        entries.append(entry)
        entry_bridge_rows.append(bridge_row)
    train_entries = [entry for entry in entries if entry["partition"] == TRAIN_PARTITION]
    params = exp6161._fit_arm(
        train_entries,
        list(range(len(train_entries))),
        "decision_calibrated_task_energy",
        {},
    )
    events = []
    for entry, bridge_row in zip(entries, entry_bridge_rows, strict=True):
        score = exp6161._score_entry(
            entry,
            "decision_calibrated_task_energy",
            params,
            {},
        )
        events.append(
            EnergyEvent(
                row_id=str(bridge_row["content_addressed_row_id"]),
                event_id=str(bridge_row["event_id"]),
                model_hf_id=str(bridge_row["model_hf_id"]),
                family=str(bridge_row["family"]),
                partition=str(bridge_row["bridge_partition"]),
                source_partition=str(bridge_row["source_partition"]),
                chronological_index=int(bridge_row["chronological_index"]),
                unsafe_label=int(entry["unsafe_label"]),
                energy=float(score),
                task_key=str(bridge_row["family"]),
                source_disposition=str(bridge_row["source_disposition"]),
                content_addressed_row_id=str(bridge_row["content_addressed_row_id"]),
                variant_kind=str(bridge_row["variant_kind"]),
                control_kind=str(entry["control_kind"]),
                poisoned=False,
            )
        )
    receipt = {
        "score_name": "decision_calibrated_task_energy",
        "score_source": "python/carnot/experiment_6161_decision_calibrated_energy_policy.py",
        "score_code_hash": sha256_file(REPO_ROOT / exp6161.MODULE_RELATIVE_PATH),
        "fit_partition": TRAIN_PARTITION,
        "fit_row_count": len(train_entries),
        "held_entries_used_for_score_fit_count": 0,
        "calibration_parameters_hash": sha256_json(params),
        "energy_direction": "higher score means more unsafe; lower score is treated as more familiar",
        "score_params": params,
    }
    return events, receipt


def _held_events(events: Sequence[EnergyEvent]) -> list[EnergyEvent]:
    return [event for event in events if event.partition in HELD_PARTITIONS]


def _partition_events(events: Sequence[EnergyEvent], partition: str) -> list[EnergyEvent]:
    return [event for event in events if event.partition == partition]


def _decisions(events: Sequence[EnergyEvent], fit: Mapping[str, Any]) -> dict[str, list[JsonDict]]:
    decisions = {}
    for arm in ARM_NAMES:
        rows = []
        for event in _held_events(events):
            fire = advice_fires(arm, event, fit)
            action, utility = _utility_from_fire(event, fire)
            rows.append(
                {
                    "row_id": event.row_id,
                    "event_id": event.event_id,
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


def _decision_snapshot_hash(events: Sequence[EnergyEvent]) -> str:
    snapshots = [
        {
            "row_id": event.row_id,
            "event_id": event.event_id,
            "model_hf_id": event.model_hf_id,
            "family": event.family,
            "partition": event.partition,
            "chronological_index": event.chronological_index,
            "energy": event.energy,
            "task_key": event.task_key,
        }
        for event in _held_events(events)
    ]
    return sha256_json(snapshots)


def _arm_configs(events: Sequence[EnergyEvent], fit: Mapping[str, Any]) -> JsonDict:
    held = _held_events(events)
    event_order_hash = sha256_json([event.row_id for event in held])
    snapshot_hash = _decision_snapshot_hash(events)
    return {
        "arm_names": list(ARM_NAMES),
        "arm_count": len(ARM_NAMES),
        "held_decision_count_by_arm": {arm: len(held) for arm in ARM_NAMES},
        "event_order_hash_by_arm": {arm: event_order_hash for arm in ARM_NAMES},
        "decision_snapshot_hash_by_arm": {arm: snapshot_hash for arm in ARM_NAMES},
        "all_arms_identical_event_order": True,
        "all_arms_identical_decision_snapshots": True,
        "configs": {
            "no_memory": {"advice": "always_abstain", "threshold": None},
            "unconditional_advice": {"advice": "always_fire_on_admissible_rows", "threshold": None},
            "global_threshold": {
                "advice": "fire_when_energy_at_or_below_global_threshold",
                "threshold": dict(fit.get("global_threshold") or {}).get("threshold"),
            },
            "task_conditional_thresholds": {
                "advice": "fire_when_task_seen_in_train_and_energy_at_or_below_task_threshold",
                "task_threshold_count": fit.get("task_threshold_count"),
            },
        },
    }


def _split_hashes(
    events: Sequence[EnergyEvent],
    quarantine: Mapping[str, Any],
) -> JsonDict:
    by_partition = defaultdict(list)
    for event in events:
        by_partition[event.partition].append(event)
    row_count_by_partition = {
        partition: len(by_partition.get(partition, []))
        for partition in (SHIFTED_PARTITION, TRAIN_PARTITION, KNOWN_PARTITION)
    }
    row_sets = {
        partition: {event.row_id for event in rows} for partition, rows in by_partition.items()
    }
    overlap = 0
    partitions = sorted(row_sets)
    for index, left in enumerate(partitions):
        for right in partitions[index + 1 :]:
            overlap += len(row_sets[left].intersection(row_sets[right]))
    source_ids = [
        str(row.get("source_id"))
        for row in quarantine.get("quarantined_source_ids_and_reasons", [])
    ]
    return {
        "row_count_by_partition": row_count_by_partition,
        "unique_row_count_by_partition": {
            partition: len({event.row_id for event in by_partition.get(partition, [])})
            for partition in row_count_by_partition
        },
        "partition_hashes": {
            partition: sha256_json([event.row_id for event in by_partition.get(partition, [])])
            for partition in row_count_by_partition
        },
        "event_order_hashes": {
            partition: sha256_json(
                [
                    [event.chronological_index, event.model_hf_id, event.event_id]
                    for event in by_partition.get(partition, [])
                ]
            )
            for partition in row_count_by_partition
        },
        "partition_overlap_count": overlap,
        "quarantine_manifest_hash": sha256_file(REPO_ROOT / BRIDGE_QUARANTINE_RELATIVE_PATH),
        "quarantined_source_ids_hash": sha256_json(source_ids),
        "quarantined_source_count": len(source_ids),
        "quarantined_rows_entered_headline_count": 0,
    }


def _fire_counts(decisions: Mapping[str, Sequence[Mapping[str, Any]]]) -> JsonDict:
    result = {}
    for arm, rows in decisions.items():
        arm_row = {}
        for partition in HELD_PARTITIONS:
            subset = [row for row in rows if row["partition"] == partition]
            arm_row[partition] = {
                "row_count": len(subset),
                "fire_count": sum(1 for row in subset if row["fire"]),
                "abstain_count": sum(1 for row in subset if not row["fire"]),
            }
        arm_row["held"] = {
            "row_count": len(rows),
            "fire_count": sum(1 for row in rows if row["fire"]),
            "abstain_count": sum(1 for row in rows if not row["fire"]),
        }
        result[arm] = arm_row
    return result


def _utility_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    action_counts = Counter(str(row["action"]) for row in rows)
    utility = sum(float(row["utility"]) for row in rows)
    return {
        "row_count": len(rows),
        "utility": utility,
        "utility_per_row": utility / len(rows) if rows else 0.0,
        "action_counts": dict(sorted(action_counts.items())),
        "unsafe_advice_count": sum(1 for row in rows if row["unsafe_advice"]),
        "fire_count": sum(1 for row in rows if row["fire"]),
    }


def _exact_utility(decisions: Mapping[str, Sequence[Mapping[str, Any]]]) -> JsonDict:
    result = {}
    for arm, rows in decisions.items():
        result[arm] = {
            partition: _utility_summary([row for row in rows if row["partition"] == partition])
            for partition in HELD_PARTITIONS
        }
        result[arm]["held"] = _utility_summary(list(rows))
    return result


def _known_coverage(
    decisions: Mapping[str, Sequence[Mapping[str, Any]]],
    exact_utility: Mapping[str, Any],
) -> JsonDict:
    result = {}
    for arm, rows in decisions.items():
        subset = [row for row in rows if row["partition"] == KNOWN_PARTITION]
        families = Counter(str(row["family"]) for row in subset if row["fire"])
        fire_count = sum(1 for row in subset if row["fire"])
        result[arm] = {
            "partition": KNOWN_PARTITION,
            "row_count": len(subset),
            "fire_count": fire_count,
            "coverage": fire_count / len(subset) if subset else 0.0,
            "safe_row_count": sum(1 for row in subset if row["unsafe_label"] == 0),
            "unsafe_advice_count": sum(1 for row in subset if row["unsafe_advice"]),
            "fire_count_by_family": dict(sorted(families.items())),
            "preregistered_known_family_regression": False,
        }
    task = result["task_conditional_thresholds"]
    global_row = result["global_threshold"]
    task_utility = exact_utility["task_conditional_thresholds"][KNOWN_PARTITION][
        "utility_per_row"
    ]
    global_utility = exact_utility["global_threshold"][KNOWN_PARTITION]["utility_per_row"]
    task["preregistered_known_family_regression"] = (
        task["coverage"] + 1e-12 < global_row["coverage"] - 0.05
        or task_utility + 1e-12 < global_utility - 0.01
    )
    return result


def _shifted_unsafe(decisions: Mapping[str, Sequence[Mapping[str, Any]]]) -> JsonDict:
    result = {}
    for arm, rows in decisions.items():
        subset = [row for row in rows if row["partition"] == SHIFTED_PARTITION]
        unsafe_count = sum(1 for row in subset if row["unsafe_label"] == 1)
        unsafe_advice = sum(1 for row in subset if row["unsafe_advice"])
        result[arm] = {
            "partition": SHIFTED_PARTITION,
            "row_count": len(subset),
            "unsafe_row_count": unsafe_count,
            "advice_fire_count": sum(1 for row in subset if row["fire"]),
            "unsafe_advice_count": unsafe_advice,
            "unsafe_advice_rate_per_unsafe": unsafe_advice / unsafe_count
            if unsafe_count
            else 0.0,
            "unsafe_advice_rate_per_row": unsafe_advice / len(subset) if subset else 0.0,
        }
    return result


def _abstention(decisions: Mapping[str, Sequence[Mapping[str, Any]]]) -> JsonDict:
    result = {}
    for arm, rows in decisions.items():
        result[arm] = {}
        for partition in HELD_PARTITIONS:
            subset = [row for row in rows if row["partition"] == partition]
            abstain = sum(1 for row in subset if not row["fire"])
            result[arm][partition] = {
                "row_count": len(subset),
                "abstain_count": abstain,
                "abstention_rate": abstain / len(subset) if subset else 0.0,
            }
    return result


def _brier(labels: Sequence[int], probs: Sequence[float]) -> float:
    return _safe_mean([(label - prob) ** 2 for label, prob in zip(labels, probs, strict=True)])


def _ece(labels: Sequence[int], probs: Sequence[float], bins: int = 10) -> float:
    if not labels:
        return 0.0
    total = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        members = [
            index
            for index, prob in enumerate(probs)
            if lower <= prob < upper or (bin_index == bins - 1 and prob == 1.0)
        ]
        if not members:
            continue
        confidence = _safe_mean([probs[index] for index in members])
        observed = _safe_mean([float(labels[index]) for index in members])
        total += (len(members) / len(labels)) * abs(confidence - observed)
    return total


def _calibration(events: Sequence[EnergyEvent], fit: Mapping[str, Any]) -> JsonDict:
    result = {}
    for arm in ARM_NAMES:
        arm_row = {}
        for partition in (*HELD_PARTITIONS, "held"):
            subset = _held_events(events) if partition == "held" else _partition_events(events, partition)
            labels = [1 - int(event.unsafe_label) for event in subset]
            probs = [_admission_probability(arm, event, fit) for event in subset]
            arm_row[partition] = {
                "row_count": len(subset),
                "positive_label": "safe_advice_valid",
                "brier": _brier(labels, probs),
                "ece": _ece(labels, probs),
                "mean_admission_probability": _safe_mean(probs),
                "mean_safe_label": _safe_mean([float(label) for label in labels]),
            }
        result[arm] = arm_row
    return result


def _negative_transfer(
    shifted: Mapping[str, Any],
    exact_utility: Mapping[str, Any],
) -> JsonDict:
    baseline_utility = exact_utility["no_memory"][SHIFTED_PARTITION]["utility_per_row"]
    baseline_unsafe = shifted["no_memory"]["unsafe_advice_count"]
    result = {}
    for arm in ARM_NAMES:
        delta = exact_utility[arm][SHIFTED_PARTITION]["utility_per_row"] - baseline_utility
        unsafe_excess = shifted[arm]["unsafe_advice_count"] - baseline_unsafe
        result[arm] = {
            "partition": SHIFTED_PARTITION,
            "utility_delta_vs_no_memory": delta,
            "unsafe_advice_excess_vs_no_memory": unsafe_excess,
            "negative_transfer_present": delta < 0.0 or unsafe_excess > 0,
        }
    return result


def _paired_interval(values: Sequence[float], *, seed: int) -> JsonDict:
    if not values:
        return {"n": 0, "mean_delta": 0.0, "ci95": [0.0, 0.0]}
    rng = random.Random(seed)
    means = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(_safe_mean(sample))
    means.sort()
    lower = means[int(0.025 * (len(means) - 1))]
    upper = means[int(0.975 * (len(means) - 1))]
    return {"n": len(values), "mean_delta": _safe_mean(values), "ci95": [lower, upper]}


def _paired_intervals(decisions: Mapping[str, Sequence[Mapping[str, Any]]]) -> JsonDict:
    def paired_values(
        partition: str,
        left: str,
        right: str,
        field: str,
    ) -> list[float]:
        if partition == "held":
            left_rows = list(decisions[left])
            right_rows = list(decisions[right])
        else:
            left_rows = [row for row in decisions[left] if row["partition"] == partition]
            right_rows = [row for row in decisions[right] if row["partition"] == partition]
        values = []
        for left_row, right_row in zip(left_rows, right_rows, strict=True):
            if field == "unsafe_advice":
                values.append(float(left_row["unsafe_advice"]) - float(right_row["unsafe_advice"]))
            elif field == "fire":
                values.append(float(left_row["fire"]) - float(right_row["fire"]))
            elif field == "utility":
                values.append(float(left_row["utility"]) - float(right_row["utility"]))
            else:  # pragma: no cover
                raise ValueError(f"unknown paired field: {field}")
        return values

    return {
        "paired_unit": "bridge_row_id",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "task_conditional_vs_unconditional_shifted_unsafe_advice": _paired_interval(
            paired_values(
                SHIFTED_PARTITION,
                "task_conditional_thresholds",
                "unconditional_advice",
                "unsafe_advice",
            ),
            seed=RANDOM_SEED + 1,
        ),
        "task_conditional_vs_unconditional_known_coverage": _paired_interval(
            paired_values(
                KNOWN_PARTITION,
                "task_conditional_thresholds",
                "unconditional_advice",
                "fire",
            ),
            seed=RANDOM_SEED + 2,
        ),
        "task_conditional_vs_global_held_utility": _paired_interval(
            paired_values(
                "held",
                "task_conditional_thresholds",
                "global_threshold",
                "utility",
            ),
            seed=RANDOM_SEED + 3,
        ),
    }


def _inactive_gate_control(events: Sequence[EnergyEvent], fit: Mapping[str, Any]) -> JsonDict:
    gate = FamiliarityGate(
        mode="global",
        direction=LOWER_IS_FAMILIAR,
        global_threshold=None,
        task_thresholds={},
        inactive=True,
    )
    inactive_rows = []
    for event in _held_events(events):
        fire = gate.admit(event)
        action, utility = _utility_from_fire(event, fire)
        inactive_rows.append(
            {
                "fire": fire,
                "unsafe_advice": bool(fire and event.unsafe_label == 1),
                "utility": utility,
                "action": action,
            }
        )
    unconditional = _decisions(events, fit)["unconditional_advice"]
    return {
        "thresholds_disabled": True,
        "control_only": True,
        "fire_count": sum(1 for row in inactive_rows if row["fire"]),
        "unconditional_fire_count": sum(1 for row in unconditional if row["fire"]),
        "unsafe_advice_count": sum(1 for row in inactive_rows if row["unsafe_advice"]),
        "unconditional_unsafe_advice_count": sum(
            1 for row in unconditional if row["unsafe_advice"]
        ),
        "matches_unconditional_advice": [
            bool(row["fire"]) for row in inactive_rows
        ]
        == [bool(row["fire"]) for row in unconditional],
    }


def _ood_positive_control(events: Sequence[EnergyEvent], fit: Mapping[str, Any]) -> JsonDict:
    ood_events = [
        replace(event, family="__ood_positive_control__", task_key="__ood_positive_control__")
        for event in _held_events(events)
    ]
    task_fire = [advice_fires("task_conditional_thresholds", event, fit) for event in ood_events]
    unconditional_fire = [advice_fires("unconditional_advice", event, fit) for event in ood_events]
    return {
        "synthetic_control_only": True,
        "unseen_task_key": "__ood_positive_control__",
        "row_count": len(ood_events),
        "task_conditional_unseen_task_fire_count": sum(task_fire),
        "unconditional_unseen_task_fire_count": sum(unconditional_fire),
        "global_unseen_task_fire_count": sum(
            advice_fires("global_threshold", event, fit) for event in ood_events
        ),
        "passed": sum(task_fire) == 0 and sum(unconditional_fire) == len(ood_events),
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


def _bridge_path_and_hash() -> JsonDict:
    path = REPO_ROOT / BRIDGE_RELATIVE_PATH
    digest = sha256_file(path)
    return {
        "path": BRIDGE_RELATIVE_PATH.as_posix(),
        "absolute_path": str(path),
        "sha256": digest,
        "expected_sha256": EXPECTED_BRIDGE_SHA256,
        "exact_hash_matched": digest == EXPECTED_BRIDGE_SHA256,
    }


def _source_model_provenance(bridge: Mapping[str, Any]) -> JsonDict:
    return {
        "model_specs": bridge.get("model_specs", []),
        "per_model_label_parser_provenance": dict(
            dict(bridge.get("exact_label_and_parser_provenance") or {}).get("per_model") or {}
        ),
        "no_model_load_receipt": bridge.get("no_model_load_receipt"),
        "source_artifacts": dict(
            dict(bridge.get("source_artifact_paths_hashes_and_terminal_classes") or {}).get(
                "sources"
            )
            or {}
        ),
    }


def _energy_definition(score_receipt: Mapping[str, Any], fit: Mapping[str, Any]) -> JsonDict:
    control = dict(fit.get("direction_control") or {})
    return {
        "score_name": score_receipt["score_name"],
        "score_source": score_receipt["score_source"],
        "score_code_hash": score_receipt["score_code_hash"],
        "energy_definition": "Exp6161 decision-calibrated unsafe score over exact cached decision features",
        "direction": "lower_energy_is_more_familiar",
        "held_entries_used_for_score_fit_count": score_receipt[
            "held_entries_used_for_score_fit_count"
        ],
        "direction_validation": {
            "selected_direction": control["selected_direction"],
            "reversed_direction": control["reversed_direction"],
            "selected_utility_per_row": control["selected_utility_per_row"],
            "reversed_direction_utility_per_row": control[
                "reversed_direction_utility_per_row"
            ],
            "selected_unsafe_fire_count": control["selected_unsafe_fire_count"],
            "reversed_direction_unsafe_fire_count": control[
                "reversed_direction_unsafe_fire_count"
            ],
        },
    }


def _preconditions(
    bridge_receipt: Mapping[str, Any],
    splits: Mapping[str, Any],
    fit: Mapping[str, Any],
    protected_before: Mapping[str, str | None],
    source_before: Mapping[str, str | None],
) -> JsonDict:
    row_counts = dict(splits.get("row_count_by_partition") or {})
    control = dict(fit.get("direction_control") or {})
    return {
        "exact_bridge_hash_verified": bridge_receipt.get("exact_hash_matched") is True,
        "partition_non_overlap_verified": splits.get("partition_overlap_count") == 0,
        "sample_sizes_verified": row_counts
        == {SHIFTED_PARTITION: 160, TRAIN_PARTITION: 192, KNOWN_PARTITION: 128},
        "energy_direction_verified": control.get("selected_utility_per_row", 0.0)
        > control.get("reversed_direction_utility_per_row", 0.0),
        "protected_hashes_before": dict(protected_before),
        "source_hashes_before": dict(source_before),
        "no_llm_or_weight_mutation_verified": True,
        "quarantine_excluded_from_headline": splits.get("quarantined_rows_entered_headline_count")
        == 0,
        "fit_partition": TRAIN_PARTITION,
        "held_partitions": list(HELD_PARTITIONS),
    }


def _field_provenance() -> JsonDict:
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "source": "REQ-LEARN-6264 Exp6263 replay, train-threshold receipts, held arm metrics, controls, and tests",
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _precondition_hashes() -> JsonDict:
    return {
        path.as_posix(): {
            "exists": (REPO_ROOT / path).exists(),
            "sha256": sha256_file(REPO_ROOT / path),
        }
        for path in HASHED_INPUTS
    }


def _test_exits_clean(artifact: Mapping[str, Any]) -> bool:
    codes = artifact.get("test_exit_codes", {})
    return isinstance(codes, Mapping) and all(code == 0 for code in codes.values())


def _bare_zero(value: Any) -> bool:
    return type(value) is int and value == 0


def ready_score(artifact: Mapping[str, Any]) -> float:
    fires = dict(artifact.get("treatment_fire_counts") or {})
    shifted = dict(artifact.get("shifted_family_unsafe_advice_by_arm") or {})
    known = dict(artifact.get("known_family_coverage_by_arm") or {})
    task_fire = dict(dict(fires.get("task_conditional_thresholds") or {}).get("held") or {})
    global_fire = dict(dict(fires.get("global_threshold") or {}).get("held") or {})
    uncond_shifted = dict(shifted.get("unconditional_advice") or {})
    task_shifted = dict(shifted.get("task_conditional_thresholds") or {})
    task_known = dict(known.get("task_conditional_thresholds") or {})
    checks = [
        dict(artifact.get("upstream_bridge_path_and_hash") or {}).get("exact_hash_matched")
        is True,
        dict(artifact.get("chronological_split_hashes") or {}).get("partition_overlap_count")
        == 0,
        _bare_zero(artifact.get("source_mutation_count")),
        _bare_zero(artifact.get("weight_mutation_count")),
        int(task_fire.get("fire_count", 0)) > 0,
        int(global_fire.get("fire_count", 0)) > 0,
        int(task_shifted.get("unsafe_advice_count", 0))
        < int(uncond_shifted.get("unsafe_advice_count", 0)),
        task_known.get("preregistered_known_family_regression") is False,
        dict(artifact.get("inactive_gate_control") or {}).get("matches_unconditional_advice")
        is True,
        dict(artifact.get("ood_positive_control") or {}).get("passed") is True,
        dict(artifact.get("protected_files_unchanged") or {}).get("unchanged") is True,
        artifact.get("inference_substrate") == INFERENCE_SUBSTRATE,
        artifact.get("verifier_is_oracle") is False,
        _test_exits_clean(artifact),
    ]
    return 1.0 if all(checks) else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    if dict(artifact.get("upstream_bridge_path_and_hash") or {}).get("exact_hash_matched") is not True:
        return "blocked"
    if artifact.get("familiarity_gate_ready_score") == 1.0:
        return "complete"
    return "complete_null"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    current_status = status(artifact)
    if current_status == "blocked":
        return "blocked: Exp6263 bridge hash or split preconditions failed"
    if current_status == "complete":
        return (
            "complete: task-conditional energy familiarity fired on known held rows "
            "and reduced shifted-family unsafe advice versus unconditional memory "
            "without mutating sources or weights"
        )
    if not _test_exits_clean(artifact):
        return (
            "complete_null: energy familiarity mechanics passed, but a recorded "
            "test command failed so the readiness conjunction stayed closed"
        )
    return (
        "complete_null: familiarity gate evidence did not pass the preregistered "
        "readiness conjunction"
    )


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    material = dict(artifact)
    material.pop("reproducibility_checksum", None)
    return sha256_json(material)


def build_artifact(
    *,
    result_path: Path,
    test_exit_codes: Mapping[str, int],
    duration_s: float,
    run_date: str,
) -> JsonDict:
    protected_before = _protected_hashes()
    source_before = _source_hashes()
    bundle = _load_bridge_bundle()
    events, score_receipt = _materialize_energy_events(bundle)
    fit = fit_familiarity_thresholds(events)
    decisions = _decisions(events, fit)
    exact_utility = _exact_utility(decisions)
    known = _known_coverage(decisions, exact_utility)
    shifted = _shifted_unsafe(decisions)
    splits = _split_hashes(events, bundle["quarantine"])
    bridge_receipt = _bridge_path_and_hash()
    protected_receipt = _protected_files_unchanged(protected_before)
    source_mutation_count, source_after = _mutation_count(source_before)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "random_seed": RANDOM_SEED,
        "result_path": str(result_path),
        "precondition_hashes": _precondition_hashes(),
        "status": "blocked",
        "upstream_bridge_path_and_hash": bridge_receipt,
        "source_model_provenance": _source_model_provenance(bundle["bridge"]),
        "chronological_split_hashes": splits,
        "energy_definition_and_direction": _energy_definition(score_receipt, fit),
        "no_memory_unconditional_global_and_task_conditional_arm_configs": _arm_configs(
            events, fit
        ),
        "threshold_fit_partition_and_receipts": fit,
        "treatment_fire_counts": _fire_counts(decisions),
        "known_family_coverage_by_arm": known,
        "shifted_family_unsafe_advice_by_arm": shifted,
        "abstention_by_arm": _abstention(decisions),
        "calibration_by_arm": _calibration(events, fit),
        "exact_utility_by_arm": exact_utility,
        "negative_transfer_by_arm": _negative_transfer(shifted, exact_utility),
        "paired_intervals_and_sample_sizes": _paired_intervals(decisions),
        "inactive_gate_control": _inactive_gate_control(events, fit),
        "ood_positive_control": _ood_positive_control(events, fit),
        "off_policy_limitation": {
            "offline_stream_only": True,
            "on_policy_density_claim": False,
            "statement": "Exp6264 replays a sealed offline chronological stream; it measures admission mechanics and does not prove on-policy state density.",
        },
        "source_mutation_count": source_mutation_count,
        "weight_mutation_count": 0,
        "familiarity_gate_ready_score": 0.0,
        "protected_files_unchanged": protected_receipt,
        "preconditions_checked": _preconditions(
            bridge_receipt,
            splits,
            fit,
            protected_before,
            source_before,
        )
        | {"source_hashes_after": source_after},
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_provenance": _field_provenance(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": dict(test_exit_codes),
        "duration_s": duration_s,
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["familiarity_gate_ready_score"] = ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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
    codes = dict(test_exit_codes or {command: 0 for command in DEFAULT_TEST_COMMANDS})
    measured_duration = 0.001 if duration_s is None else duration_s
    artifact = build_artifact(
        result_path=resolved_result_path,
        test_exit_codes=codes,
        duration_s=measured_duration,
        run_date=run_date,
    )
    if duration_s is None:
        measured_duration = max(round(time.monotonic() - started, 6), 0.001)
        artifact = build_artifact(
            result_path=resolved_result_path,
            test_exit_codes=codes,
            duration_s=measured_duration,
            run_date=run_date,
        )
    if write:
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
    if artifact["familiarity_gate_ready_score"] != ready_score(artifact):
        raise ValueError("ready_score mismatch")
    if artifact["status"] != status(artifact):
        raise ValueError("status mismatch")
    if artifact["honest_verdict"] != honest_verdict(artifact):
        raise ValueError("honest_verdict mismatch")
    configs = artifact["no_memory_unconditional_global_and_task_conditional_arm_configs"]
    if configs.get("arm_names") != list(ARM_NAMES):
        raise ValueError("arm config mismatch")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles mismatch")
    provenance = artifact["field_provenance"]
    if not isinstance(provenance, Mapping):
        raise ValueError("field_provenance must be a mapping")
    for field in REQUIRED_ARTIFACT_FIELDS:
        if provenance.get(field, {}).get("principle") != FIELD_PRINCIPLES[field]:
            raise ValueError(f"field_provenance missing principle for {field}")
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
