"""Compare grouped fixed-point proposals with a matched flat recurrent control.

Spec refs: REQ-VERIFY-6788 and SCENARIO-VERIFY-6788-*.

Both proposal arms start from the same legal variable-state projection. The
flat arm pools that state without reading graph edges or group adjacency. The
Exp6786 exact authority receives candidates only after proposal finishes. This
separation makes an exact-valid gain evidence about proposal quality, not an
exact solver hidden inside the proposal path.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any
import zlib

import torch
from torch import nn

from carnot import durable_row_checkpoint as checkpointing
from carnot import experiment_6786_constraint_dependency_hard_negative_fixture as exact_fixture
from carnot import experiment_6787_group_aware_soft_fixed_point as grouped_source


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_6788_soft_fixed_point_structural_control_ab"
SCHEMA = "carnot.experiment_6788.soft_fixed_point_structural_control_ab.v1"
ROW_SCHEMA = "carnot.experiment_6788.paired_proposal_row.v1"
RUN_DATE = "20260830"
RANDOM_SEED = 6_788_000
INFERENCE_SUBSTRATE = "paired_cpu_neural_proposal_with_independent_exact_evaluation_no_llm"
SOURCE_6786_RELATIVE_PATH = Path(
    "results/experiment_6786_constraint_dependency_hard_negative_fixture.json"
)
SOURCE_6787_RELATIVE_PATH = Path("results/experiment_6787_group_aware_soft_fixed_point.json")
RESULT_RELATIVE_PATH = Path("results") / f"{EXPERIMENT_ID}.json"
CHECKPOINT_RELATIVE_PATH = Path("results/checkpoints") / f"{EXPERIMENT_ID}.json"
EXPECTED_EXP6786_HASH = "sha256:f3780c85e29cda8dbd897b6c43a0ce3c938252625e823e54107f918d2052514a"
EXPECTED_EXP6787_HASH = "sha256:161214c61401ebdb6a5ec11ea02eb520c341bc14f7831379d19651511f32a37d"
EXPECTED_EXP6786_MANIFEST_HASH = (
    "sha256:99c6ea4200db1d22092951d9e46d1f8d1b598e3a8815d74449c828d427bbd7b9"
)
GROUPED_ARM = "grouped_fixed_point"
FLAT_ARM = "flat_recurrent_control"
ARMS = (GROUPED_ARM, FLAT_ARM)
FROZEN_HYPERPARAMETERS = deepcopy(grouped_source.FROZEN_HYPERPARAMETERS)
FROZEN_SEEDS = [int(seed) for seed in FROZEN_HYPERPARAMETERS["seeds"]]
OUTPUT_UNIT_COUNT = 64
PLANNED_ROW_COUNT = OUTPUT_UNIT_COUNT * len(FROZEN_SEEDS) * len(ARMS)
CPU_WALL_BUDGET_S = 60.0
RUNTIME_ESTIMATE_MULTIPLIER = 1.25
PARAMETER_MATCH_TOLERANCE = 0.05
SUPPORT_CONTRACTION_MARGIN = 0.05
BOOTSTRAP_RESAMPLES = 2_000
BOOTSTRAP_SEED = 6_788_900
FROZEN_SPLITS = {
    "train": {"topology_family": "directed_implication_chain", "unit_count": 32},
    "development": {"topology_family": "directed_implication_star", "unit_count": 32},
    "held_topology_test": {
        "topology_family": "directed_implication_cycle",
        "unit_count": 32,
    },
}
VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
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
STANDARD_ARTIFACT_FIELDS = ("schema", "experiment_id", "run_date", "status")
REQUIRED_ARTIFACT_FIELDS = STANDARD_ARTIFACT_FIELDS + (
    "field_principles",
    "inference_substrate",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "source_artifact_hashes",
    "frozen_manifest",
    "arm_definitions",
    "parameter_counts_by_arm",
    "optimization_steps_by_arm",
    "candidate_budget_by_arm",
    "rows",
    "metrics_by_arm",
    "metrics_by_topology",
    "paired_exact_valid_delta",
    "paired_exact_valid_delta_ci95",
    "dependency_violation_delta",
    "distance_to_valid_delta",
    "hard_negative_auroc_by_arm",
    "unique_valid_support_by_arm",
    "support_contraction",
    "paired_key_count",
    "fixed_point_comparison_completed",
    "checkpoint_receipt",
    "cold_recompute_agreement",
    "decision_gates",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
FIELD_PRINCIPLES: dict[str, str] = {
    "schema": "A versioned schema makes incompatible comparison artifacts fail closed.",
    "experiment_id": "A stable ID binds the evidence to the planned structural control.",
    "run_date": "The execution date distinguishes this frozen run from later replays.",
    "status": "Status separates a complete comparison from a complete blocked run.",
    "field_principles": "One-line purposes make every required field auditable.",
    "inference_substrate": "The declaration separates CPU proposal from later exact scoring.",
    "duration_s": "Measured wall time proves the paired workload ran inside its budget.",
    "random_seed": "The master seed anchors paired bootstrap and model seed rotation.",
    "reproducibility_checksum": "A stable hash detects source, row, or headline drift.",
    "source_artifact_hashes": "Exact hashes bind both inputs and their upstream authorities.",
    "frozen_manifest": "The manifest freezes units, pairs, seeds, budgets, and decision margins.",
    "arm_definitions": "Arm descriptions expose the single intended structural difference.",
    "parameter_counts_by_arm": "Counts prove capacity differs by no more than five percent.",
    "optimization_steps_by_arm": "Equal update totals prevent extra fitting from favoring one arm.",
    "candidate_budget_by_arm": "Equal candidate totals prevent search width from causing a gain.",
    "rows": "Every unit-seed-arm cell preserves proposal and exact post-check evidence.",
    "metrics_by_arm": "Arm summaries expose validity, violations, distance, convergence, and time.",
    "metrics_by_topology": "Family summaries prevent pooled results from hiding topology effects.",
    "paired_exact_valid_delta": "The paired grouped-minus-flat rate is the primary effect.",
    "paired_exact_valid_delta_ci95": "A unit-clustered interval measures primary-effect uncertainty.",
    "dependency_violation_delta": "The paired delta tests the failure class topology should reduce.",
    "distance_to_valid_delta": "The paired delta shows whether failures move closer to validity.",
    "hard_negative_auroc_by_arm": "AUROC tests discrimination after local constraints already pass.",
    "unique_valid_support_by_arm": "Unique valid hashes detect a gain caused by mode contraction.",
    "support_contraction": "The frozen margin prevents a narrow support set from counting as a win.",
    "paired_key_count": "The count proves each unit-seed comparison has both arms.",
    "fixed_point_comparison_completed": "This exact field authorizes Exp6789 consumption.",
    "checkpoint_receipt": "The receipt proves every planned cell reached durable parent storage.",
    "cold_recompute_agreement": "A fresh process must reproduce all row-derived aggregates.",
    "decision_gates": "Named gates make positive and null decisions mechanically inspectable.",
    "gate_check_summary": "Each precondition records its expected and observed value.",
    "verifier_is_oracle": "False states that the exact authority evaluates but never proposes.",
    "verdict_class": "A closed class keeps the terminal result machine-readable.",
    "honest_verdict": "A terminal prefix reports the measured result without hiding a null.",
}


def canonical_json(value: Any) -> str:
    """Serialize stable content so hashes do not depend on key order."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    """Return the algorithm-labelled hash of stable JSON content."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str | None:
    """Hash exact file bytes, or return no hash when the file is absent."""

    if not path.is_file():
        return None
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def load_json_object(path: Path) -> JsonDict:
    """Load one JSON object and reject arrays or scalar roots."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _gate(check: str, expected: Any, observed: Any, passed: bool | None = None) -> JsonDict:
    """Record one precondition with its expected and observed values."""

    return {
        "check": check,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": bool(observed == expected if passed is None else passed),
    }


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Collect failures without discarding later diagnostic checks."""

    copied = [deepcopy(dict(check)) for check in checks]
    failed = [check for check in copied if not check["passed"]]
    return {
        "all_passed": not failed,
        "checks": copied,
        "failed_checks": [check["check"] for check in failed],
        "first_failure": failed[0] if failed else None,
    }


@dataclass(frozen=True)
class ExperimentContext:
    """Keep legal proposal inputs separate from exact post-check authority."""

    proposal_units: list[JsonDict]
    exact_units: dict[str, JsonDict]
    hard_negatives: dict[str, JsonDict]


def build_context(
    source_6786: Mapping[str, Any], source_6787: Mapping[str, Any]
) -> ExperimentContext:
    """Project proposal inputs and retain exact data in a separate object."""

    proposal_units = grouped_source.project_units(source_6786)
    violations = grouped_source.audit_feature_contract(proposal_units)
    if violations:
        raise ValueError(f"oracle feature refusal: {violations[0]}")
    exact_units = {
        str(unit["unit_id"]): deepcopy(dict(unit))
        for unit in source_6786.get("frozen_manifest", {}).get("units", [])
    }
    hard_negatives = {
        str(row["unit_id"]): deepcopy(dict(row))
        for row in source_6786.get("rows", [])
        if row.get("negative_class") == "hard_cross_dependency_failure"
    }
    unit_ids = {str(unit["unit_id"]) for unit in proposal_units}
    if unit_ids != set(exact_units) or unit_ids != set(hard_negatives):
        raise ValueError("proposal and exact authority unit IDs differ")
    if source_6787.get("feature_allowlist") != list(grouped_source.FEATURE_ALLOWLIST):
        raise ValueError("Exp6787 legal feature allowlist drifted")
    return ExperimentContext(proposal_units, exact_units, hard_negatives)


def _manifest_hash_is_valid(source_6786: Mapping[str, Any]) -> bool:
    manifest = deepcopy(source_6786.get("frozen_manifest", {}))
    declared = manifest.pop("manifest_sha256", None)
    return declared == sha256_json(manifest) == EXPECTED_EXP6786_MANIFEST_HASH


def _observed_split_contract(source_6786: Mapping[str, Any]) -> JsonDict:
    observed: JsonDict = {}
    for split in grouped_source.REQUIRED_SPLITS:
        item = source_6786.get("split_by_topology", {}).get(split, {})
        families = item.get("topology_families", [])
        observed[split] = {
            "topology_family": families[0] if len(families) == 1 else families,
            "unit_count": item.get("unit_count"),
        }
    return observed


def _upstream_source_hash_observations(repo_root: Path, source_6786: Mapping[str, Any]) -> JsonDict:
    observed: JsonDict = {}
    for relative in source_6786.get("source_artifact_hashes", {}):
        path = Path(str(relative))
        resolved = path if path.is_absolute() else repo_root / path
        observed[str(relative)] = sha256_file(resolved)
    return observed


def evaluate_preconditions(
    *,
    repo_root: Path = REPO_ROOT,
    source_6786_path: Path | None = None,
    source_6787_path: Path | None = None,
    expected_6786_hash: str = EXPECTED_EXP6786_HASH,
    expected_6787_hash: str = EXPECTED_EXP6787_HASH,
    cpu_wall_budget_s: float = CPU_WALL_BUDGET_S,
) -> JsonDict:
    """Check readiness, hashes, splits, seeds, features, and CPU row budget."""

    path_6786 = source_6786_path or repo_root / SOURCE_6786_RELATIVE_PATH
    path_6787 = source_6787_path or repo_root / SOURCE_6787_RELATIVE_PATH
    exists_6786 = path_6786.is_file()
    exists_6787 = path_6787.is_file()
    source_6786 = load_json_object(path_6786) if exists_6786 else {}
    source_6787 = load_json_object(path_6787) if exists_6787 else {}
    actual_6786_hash = sha256_file(path_6786)
    actual_6787_hash = sha256_file(path_6787)
    projected = grouped_source.project_units(source_6786) if exists_6786 else []
    feature_violations = (
        sorted(
            set(source_6786.get("future_feature_violations", []))
            | set(grouped_source.audit_feature_contract(projected))
        )
        if exists_6786
        else ["source artifact unavailable"]
    )
    declared_upstream = source_6786.get("source_artifact_hashes", {})
    observed_upstream = _upstream_source_hash_observations(repo_root, source_6786)
    source_duration = source_6787.get("duration_s")
    source_rows = source_6787.get("rows", [])
    estimated_runtime = (
        round(
            float(source_duration)
            / len(source_rows)
            * PLANNED_ROW_COUNT
            * RUNTIME_ESTIMATE_MULTIPLIER,
            6,
        )
        if isinstance(source_duration, (int, float))
        and source_duration > 0
        and isinstance(source_rows, list)
        and source_rows
        else None
    )
    budget_observation = {
        "planned_row_count": PLANNED_ROW_COUNT,
        "estimated_runtime_s": estimated_runtime,
        "cpu_wall_budget_s": cpu_wall_budget_s,
    }
    seeds = source_6787.get("frozen_hyperparameters", {}).get("seeds")
    checks = [
        _gate("exp6786_artifact_exists", True, exists_6786),
        _gate("exp6787_artifact_exists", True, exists_6787),
        _gate(
            "constraint_group_fixture_ready",
            True,
            source_6786.get("constraint_group_fixture_ready") if exists_6786 else None,
        ),
        _gate(
            "soft_fixed_point_proposer_ready",
            True,
            source_6787.get("soft_fixed_point_proposer_ready") if exists_6787 else None,
        ),
        _gate("exp6786_artifact_hash", expected_6786_hash, actual_6786_hash),
        _gate("exp6787_artifact_hash", expected_6787_hash, actual_6787_hash),
        _gate(
            "exp6787_source_hash",
            expected_6786_hash,
            source_6787.get("source_artifact_hash") if exists_6787 else None,
        ),
        _gate("source_artifact_hashes", declared_upstream, observed_upstream),
        _gate("frozen_manifest_hash", True, _manifest_hash_is_valid(source_6786)),
        _gate("frozen_splits", FROZEN_SPLITS, _observed_split_contract(source_6786)),
        _gate("five_frozen_seeds", FROZEN_SEEDS, seeds),
        _gate("legal_feature_contract", [], feature_violations),
        _gate(
            "planned_rows_fit_cpu_wall_budget",
            f"estimated runtime <= {cpu_wall_budget_s}",
            budget_observation,
            estimated_runtime is not None and estimated_runtime <= cpu_wall_budget_s,
        ),
    ]
    summary = _gate_summary(checks)
    summary.update(
        {
            "planned_row_count": PLANNED_ROW_COUNT,
            "observed_seeds": deepcopy(seeds),
            "oracle_feature_violations": feature_violations,
            "runtime_budget": budget_observation,
            "source_artifact_hashes": {
                str(SOURCE_6786_RELATIVE_PATH): actual_6786_hash,
                str(SOURCE_6787_RELATIVE_PATH): actual_6787_hash,
                **observed_upstream,
            },
        }
    )
    return summary


@dataclass(frozen=True)
class ArmStep:
    """Expose the shared state and optional structural messages for one step."""

    variable_state: torch.Tensor
    group_messages: torch.Tensor | None
    dependency_messages: torch.Tensor | None
    aggregated_dependency_messages: torch.Tensor | None


class FlatRecurrentControl(nn.Module):
    """Use pooled flat state statistics without any graph edge or adjacency."""

    def __init__(self, *, hidden_width: int, seed: int) -> None:
        super().__init__()
        if hidden_width <= 0:
            raise ValueError("hidden_width must be positive")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.update_network = nn.Sequential(
                nn.Linear(8, hidden_width),
                nn.Tanh(),
                nn.Linear(hidden_width, 2),
            )
            self.update_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.update_network = self.update_network.to(dtype=torch.float64, device="cpu")

    def recurrent_step(self, state: torch.Tensor, features: Mapping[str, Any]) -> ArmStep:
        """Update each state from exchangeable global pools, not topology."""

        expected_shape = (len(features["local_groups"]), 2)
        if tuple(state.shape) != expected_shape:
            raise ValueError(f"variable state shape must be {expected_shape}")
        count = state.shape[0]
        mean = state.mean(dim=0, keepdim=True).expand(count, -1)
        maximum = state.max(dim=0, keepdim=True).values.expand(count, -1)
        minimum = state.min(dim=0, keepdim=True).values.expand(count, -1)
        network_input = torch.cat((state, mean, maximum, minimum), dim=1)
        delta = self.update_network(network_input)
        proposal = torch.softmax(
            torch.log(state.clamp_min(1e-12)) + torch.sigmoid(self.update_scale) * delta, dim=1
        )
        updated = 0.5 * state + 0.5 * proposal
        return ArmStep(updated, None, None, None)


def trainable_parameter_count(model: nn.Module) -> int:
    """Count only parameters that the shared optimizer can change."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_arm_models(seed: int) -> dict[str, nn.Module]:
    """Build separate arm instances with the same parameter initialization."""

    width = int(FROZEN_HYPERPARAMETERS["hidden_width"])
    return {
        GROUPED_ARM: grouped_source.GroupAwareSoftFixedPoint(hidden_width=width, seed=seed),
        FLAT_ARM: FlatRecurrentControl(hidden_width=width, seed=seed),
    }


def parameter_match_fraction(models: Mapping[str, nn.Module]) -> float:
    """Return the relative parameter-count difference between paired arms."""

    grouped_count = trainable_parameter_count(models[GROUPED_ARM])
    flat_count = trainable_parameter_count(models[FLAT_ARM])
    return abs(grouped_count - flat_count) / max(grouped_count, flat_count, 1)


def recurrent_step(
    model: nn.Module,
    state: torch.Tensor,
    features: Mapping[str, Any],
    *,
    arm: str,
) -> ArmStep:
    """Normalize both model-specific step records into one telemetry shape."""

    if arm == GROUPED_ARM:
        step = model.recurrent_step(state, features)  # type: ignore[attr-defined]
        return ArmStep(
            step.variable_state,
            step.group_messages,
            step.dependency_messages,
            step.aggregated_dependency_messages,
        )
    if arm == FLAT_ARM:
        return model.recurrent_step(state, features)  # type: ignore[attr-defined,no-any-return]
    raise ValueError(f"unknown arm: {arm}")


def _model_for_arm(arm: str, seed: int) -> nn.Module:
    models = build_arm_models(seed)
    if arm not in models:
        raise ValueError(f"unknown arm: {arm}")
    return models[arm]


def _rotated_units(units: Sequence[Mapping[str, Any]], seed: int) -> list[Mapping[str, Any]]:
    ordered = sorted(units, key=lambda unit: str(unit["unit_id"]))
    rotation = seed % len(ordered)
    return ordered[rotation:] + ordered[:rotation]


def fit_arm(
    train_units: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    seed: int,
    hyperparameters: Mapping[str, Any],
) -> tuple[nn.Module, JsonDict]:
    """Fit one arm with the frozen train-only schedule and update count."""

    if not train_units or any(unit.get("split") != "train" for unit in train_units):
        raise ValueError("fit_arm requires nonempty train split units only")
    violations = grouped_source.audit_feature_contract(train_units)
    if violations:
        raise ValueError(f"oracle feature refusal: {violations[0]}")
    grouped_source._configure_torch(seed)
    model = _model_for_arm(arm, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(hyperparameters["learning_rate"]))
    ordered_units = _rotated_units(train_units, seed)
    loss_history: list[float] = []
    model.train()
    for _ in range(int(hyperparameters["training_steps"])):
        optimizer.zero_grad()
        unit_losses: list[torch.Tensor] = []
        for unit in ordered_units:
            features = unit["proposal_features"]
            state = grouped_source.initial_variable_state(features, seed=seed)
            residual_terms: list[torch.Tensor] = []
            for _ in range(int(hyperparameters["training_unroll_steps"])):
                step = recurrent_step(model, state, features, arm=arm)
                residual_terms.append(torch.mean((step.variable_state - state) ** 2))
                state = step.variable_state
            certainty = torch.mean(state[:, 0] * state[:, 1])
            structural_term = (
                grouped_source._dependency_violation(state, features)
                if arm == GROUPED_ARM
                else torch.mean((state - state.mean(dim=0, keepdim=True)) ** 2)
            )
            unit_losses.append(
                torch.stack(residual_terms).mean() + 0.2 * certainty + 0.2 * structural_term
            )
        loss = torch.stack(unit_losses).mean()
        loss.backward()
        optimizer.step()
        loss_history.append(round(float(loss.detach().item()), 10))
    return model, {
        "arm": arm,
        "seed": seed,
        "optimizer": hyperparameters["optimizer"],
        "learning_rate": hyperparameters["learning_rate"],
        "optimizer_update_count": int(hyperparameters["training_steps"]),
        "training_unroll_steps": int(hyperparameters["training_unroll_steps"]),
        "initialization_schedule": "shared_seed_then_seed_rotated_train_unit_order",
        "train_unit_ids": [str(unit["unit_id"]) for unit in ordered_units],
        "train_splits_seen": sorted({str(unit["split"]) for unit in ordered_units}),
        "loss_history": loss_history,
        "final_loss": loss_history[-1],
    }


def run_arm_fixed_point(
    model: nn.Module,
    unit: Mapping[str, Any],
    *,
    arm: str,
    seed: int,
    iteration_cap: int,
    convergence_tolerance: float,
) -> JsonDict:
    """Run bounded recurrence while keeping structural-message presence explicit."""

    if iteration_cap <= 0:
        raise ValueError("iteration_cap must be positive")
    if convergence_tolerance < 0:
        raise ValueError("convergence_tolerance must be non-negative")
    features = unit["proposal_features"]
    state = grouped_source.initial_variable_state(features, seed=seed)
    receipts: list[JsonDict] = []
    dependency_receipts: list[JsonDict] = []
    residual = math.inf
    stop_reason = "iteration_cap"
    final_step: ArmStep | None = None
    model.eval()
    with torch.no_grad():
        for iteration in range(1, iteration_cap + 1):
            step = recurrent_step(model, state, features, arm=arm)
            residual = float(torch.max(torch.abs(step.variable_state - state)).item())
            values = [step.variable_state]
            values.extend(
                value
                for value in (
                    step.group_messages,
                    step.dependency_messages,
                    step.aggregated_dependency_messages,
                )
                if value is not None
            )
            finite = all(torch.isfinite(value).all().item() for value in values)
            if step.group_messages is not None and step.aggregated_dependency_messages is not None:
                # The task needs auditable presence, not a second copy of every tensor.
                # Hashing each iteration keeps topology use inspectable and makes the
                # row checkpoint small enough for one atomic write per cell.
                receipts.append(
                    {
                        "iteration": iteration,
                        "group_count": len(features["local_groups"]),
                        "local_group_messages_sha256": sha256_json(
                            grouped_source._rounded(step.group_messages)
                        ),
                        "aggregated_dependency_messages_sha256": sha256_json(
                            grouped_source._rounded(step.aggregated_dependency_messages)
                        ),
                        "state_residual": round(residual, 10),
                    }
                )
            state = step.variable_state
            final_step = step
            if not finite:
                stop_reason = "non_finite"
                break
            if residual <= convergence_tolerance:
                stop_reason = "converged"
                break
    assert final_step is not None
    if final_step.dependency_messages is not None:
        dependency_receipts = [
            {
                "dependency_id": edge["dependency_id"],
                "message": grouped_source._rounded(final_step.dependency_messages[index]),
            }
            for index, edge in enumerate(features["dependency_edges"])
        ]
    return {
        "iterations": iteration,
        "state_residual": round(residual, 10),
        "stop_reason": stop_reason,
        "finite_values": bool(torch.isfinite(state).all().item()),
        "variable_state": grouped_source._rounded(state),
        "variable_state_tensor": state,
        "group_message_presence": final_step.group_messages is not None,
        "group_message_receipts": receipts,
        "dependency_messages": dependency_receipts,
    }


def _assignment_score(
    state: torch.Tensor, unit: Mapping[str, Any], assignment: Mapping[str, Any]
) -> float:
    """Score an assignment from proposal probabilities without exact feedback."""

    values: list[float] = []
    for index, group in enumerate(unit["proposal_features"]["local_groups"]):
        first, second = [str(variable) for variable in group["variables"]]
        selected = 0 if assignment.get(first) == 1 else 1 if assignment.get(second) == 1 else None
        values.append(float(state[index, selected]) if selected is not None else 0.0)
    return round(sum(values) / len(values), 10)


def propose_raw_row(
    model: nn.Module,
    unit: Mapping[str, Any],
    *,
    arm: str,
    seed: int,
    parameter_count: int,
    optimizer_update_count: int,
    hyperparameters: Mapping[str, Any],
) -> JsonDict:
    """Create candidates and proposal telemetry before any exact authority runs."""

    violations = grouped_source.audit_feature_contract([unit])
    if violations:
        raise ValueError(f"oracle feature refusal: {violations[0]}")
    started = time.perf_counter()
    fixed_point = run_arm_fixed_point(
        model,
        unit,
        arm=arm,
        seed=seed,
        iteration_cap=int(hyperparameters["iteration_cap"]),
        convergence_tolerance=float(hyperparameters["convergence_tolerance"]),
    )
    state = fixed_point.pop("variable_state_tensor")
    candidates = grouped_source.decode_candidates(
        state,
        unit,
        seed=seed,
        threshold=float(hyperparameters["decoding_threshold"]),
        candidate_count=int(hyperparameters["candidate_count"]),
    )
    for candidate in candidates:
        candidate["proposal_score"] = _assignment_score(state, unit, candidate["assignment"])
    variable_ids = [str(value) for value in unit["variable_ids"]]
    paired_key = f"{unit['unit_id']}|seed-{seed}"
    return {
        "schema": ROW_SCHEMA,
        "row_id": f"{paired_key}|{arm}",
        "paired_key": paired_key,
        "unit_id": unit["unit_id"],
        "graph_id": unit["graph_id"],
        "split": unit["split"],
        "topology_family": unit["topology_family"],
        "difficulty_stratum": unit["difficulty_stratum"],
        "arm": arm,
        "random_seed": seed,
        "parameter_count": parameter_count,
        "optimizer_update_count": optimizer_update_count,
        "candidate_budget": int(hyperparameters["candidate_count"]),
        "variable_ids": variable_ids,
        "iterations": fixed_point["iterations"],
        "state_residual": fixed_point["state_residual"],
        "stop_reason": fixed_point["stop_reason"],
        "finite_values": fixed_point["finite_values"],
        "variable_state": fixed_point["variable_state"],
        "group_message_presence": fixed_point["group_message_presence"],
        "group_message_receipts": fixed_point["group_message_receipts"],
        "dependency_messages": fixed_point["dependency_messages"],
        "candidates": candidates,
        "raw_candidate_vectors": [
            [int(candidate["assignment"][variable]) for variable in variable_ids]
            for candidate in candidates
        ],
        "candidate_hashes": [candidate["candidate_hash"] for candidate in candidates],
        "runtime_s": round(time.perf_counter() - started, 6),
    }


def _selected_group_state(group: Mapping[str, Any], assignment: Mapping[str, Any]) -> int | None:
    first, second = [str(variable) for variable in group["variables"]]
    if assignment.get(first) == 1 and assignment.get(second) == 0:
        return 0
    if assignment.get(first) == 0 and assignment.get(second) == 1:
        return 1
    return None


def _distance_to_assignment(
    unit: Mapping[str, Any], left: Mapping[str, Any], right: Mapping[str, Any]
) -> int:
    """Count group selections that differ between two complete assignments."""

    distance = 0
    for group in unit["graph"]["local_groups"]:
        if _selected_group_state(group, left) != _selected_group_state(group, right):
            distance += 1
    return distance


def _nearest_valid_distance(unit: Mapping[str, Any], assignment: Mapping[str, Any]) -> int | None:
    exact_assignments = unit.get("exact_assignments", [])
    if not exact_assignments:
        return None
    return min(_distance_to_assignment(unit, assignment, valid) for valid in exact_assignments)


def binary_auroc(
    positive_scores: Sequence[float], negative_scores: Sequence[float]
) -> float | None:
    """Compute exact pairwise AUROC with half credit for tied scores."""

    if not positive_scores or not negative_scores:
        return None
    wins = 0.0
    comparisons = 0
    for positive in positive_scores:
        for negative in negative_scores:
            comparisons += 1
            wins += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return round(wins / comparisons, 10)


def attach_exact_outcomes(
    raw_row: Mapping[str, Any],
    exact_unit: Mapping[str, Any],
    hard_negative: Mapping[str, Any],
) -> JsonDict:
    """Append independent exact outcomes after proposal bytes are complete."""

    row = deepcopy(dict(raw_row))
    before_hashes = deepcopy(row["candidate_hashes"])
    outcomes: list[JsonDict] = []
    for candidate in row["candidates"]:
        assignment = candidate["assignment"]
        receipt = exact_fixture.evaluate_candidate(exact_unit, assignment)
        outcomes.append(
            {
                "candidate_index": candidate["candidate_index"],
                "candidate_hash": candidate["candidate_hash"],
                "assignment": deepcopy(assignment),
                "proposal_score": candidate["proposal_score"],
                "exact_valid": receipt["exact_valid"],
                "local_checks_passed": receipt["local_checks_passed"],
                "failed_local_group_ids": receipt["failed_local_group_ids"],
                "failed_dependency_ids": receipt["failed_dependency_ids"],
                "dependency_violation_count": len(receipt["failed_dependency_ids"]),
                "distance_to_nearest_valid": _nearest_valid_distance(exact_unit, assignment),
            }
        )
    state = torch.tensor(row["variable_state"], dtype=torch.float64)
    positive_scores = [
        _assignment_score(state, {"proposal_features": exact_unit["graph"]}, assignment)
        for assignment in exact_unit.get("exact_assignments", [])
    ]
    negative_assignment = hard_negative["candidate_assignment"]
    negative_scores = [
        _assignment_score(state, {"proposal_features": exact_unit["graph"]}, negative_assignment)
    ]
    valid_count = sum(bool(outcome["exact_valid"]) for outcome in outcomes)
    dependency_fail_count = sum(bool(outcome["failed_dependency_ids"]) for outcome in outcomes)
    row.update(
        {
            "exact_outcomes": outcomes,
            "exact_valid_candidate_count": valid_count,
            "exact_valid_rate": round(valid_count / len(outcomes), 10),
            "cross_dependency_violation_count": dependency_fail_count,
            "cross_dependency_violation_rate": round(dependency_fail_count / len(outcomes), 10),
            "nearest_valid_distance": min(
                int(outcome["distance_to_nearest_valid"])
                for outcome in outcomes
                if outcome["distance_to_nearest_valid"] is not None
            ),
            "hard_negative_discrimination": {
                "positive_count": len(positive_scores),
                "hard_negative_count": len(negative_scores),
                "hard_negative_row_id": hard_negative["row_id"],
                "positive_score_mean": round(sum(positive_scores) / len(positive_scores), 10),
                "hard_negative_score": negative_scores[0],
                "auroc": binary_auroc(positive_scores, negative_scores),
            },
            "exact_evaluation_receipt": {
                "checker": "experiment_6786.evaluate_candidate",
                "evaluated_after_proposal": True,
                "candidate_hashes_before": before_hashes,
                "candidate_hashes_after": deepcopy(row["candidate_hashes"]),
                "model_feedback_applied": False,
            },
        }
    )
    return row


def arm_definitions() -> JsonDict:
    """Describe the controlled difference and every matched resource."""

    common = {
        "legal_observations": list(grouped_source.FEATURE_ALLOWLIST),
        "flattened_observation": "seeded two-state vector for every ordered local domain",
        "optimizer": FROZEN_HYPERPARAMETERS["optimizer"],
        "training_steps_per_seed": FROZEN_HYPERPARAMETERS["training_steps"],
        "training_unroll_steps": FROZEN_HYPERPARAMETERS["training_unroll_steps"],
        "iteration_cap": FROZEN_HYPERPARAMETERS["iteration_cap"],
        "candidate_count_per_cell": FROZEN_HYPERPARAMETERS["candidate_count"],
        "seeds": deepcopy(FROZEN_SEEDS),
        "initialization_schedule": "shared_seed_then_seed_rotated_train_unit_order",
        "cpu_wall_time_envelope_s": CPU_WALL_BUDGET_S,
    }
    return {
        GROUPED_ARM: {
            **deepcopy(common),
            "model": "Exp6787 group-aware soft fixed-point recurrence",
            "reads_group_adjacency": True,
            "reads_dependency_edges": True,
            "emits_group_messages": True,
        },
        FLAT_ARM: {
            **deepcopy(common),
            "model": "exchangeable pooled flat recurrent control",
            "reads_group_adjacency": False,
            "reads_dependency_edges": False,
            "emits_group_messages": False,
        },
    }


def frozen_manifest(
    output_units: Sequence[Mapping[str, Any]], *, seeds: Sequence[int] = FROZEN_SEEDS
) -> JsonDict:
    """Freeze row identities and all paired resource and inference choices."""

    row_ids = [
        f"{unit['unit_id']}|seed-{seed}|{arm}"
        for seed in seeds
        for unit in output_units
        for arm in ARMS
    ]
    paired_keys = [f"{unit['unit_id']}|seed-{seed}" for seed in seeds for unit in output_units]
    manifest: JsonDict = {
        "schema": "carnot.experiment_6788.frozen_manifest.v1",
        "source_manifest_hash": EXPECTED_EXP6786_MANIFEST_HASH,
        "unit_ids": [str(unit["unit_id"]) for unit in output_units],
        "split_by_unit_id": {str(unit["unit_id"]): str(unit["split"]) for unit in output_units},
        "topology_by_unit_id": {
            str(unit["unit_id"]): str(unit["topology_family"]) for unit in output_units
        },
        "seeds": [int(seed) for seed in seeds],
        "arms": list(ARMS),
        "paired_keys": paired_keys,
        "row_ids": row_ids,
        "planned_row_count": len(row_ids),
        "candidate_count_per_cell": FROZEN_HYPERPARAMETERS["candidate_count"],
        "training_steps_per_seed": FROZEN_HYPERPARAMETERS["training_steps"],
        "iteration_cap": FROZEN_HYPERPARAMETERS["iteration_cap"],
        "parameter_match_tolerance": PARAMETER_MATCH_TOLERANCE,
        "support_contraction_margin": SUPPORT_CONTRACTION_MARGIN,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "cpu_wall_budget_s": CPU_WALL_BUDGET_S,
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    return manifest


def _encode_checkpoint_payload(row: Mapping[str, Any]) -> JsonDict:
    """Compress one row so repeated atomic checkpoint writes stay CPU-bounded."""

    raw = canonical_json(row).encode("utf-8")
    return {
        "encoding": "zlib_base64_canonical_json",
        "row_sha256": sha256_json(row),
        "data": base64.b64encode(zlib.compress(raw, level=9)).decode("ascii"),
    }


def _decode_checkpoint_payload(payload: Mapping[str, Any]) -> JsonDict:
    """Restore a row and verify the hash before aggregate or resume use."""

    if payload.get("encoding") != "zlib_base64_canonical_json":
        return deepcopy(dict(payload))
    raw = zlib.decompress(base64.b64decode(str(payload["data"]))).decode("utf-8")
    row = json.loads(raw)
    if not isinstance(row, dict) or sha256_json(row) != payload.get("row_sha256"):
        raise ValueError("checkpoint row payload hash mismatch")
    return row


def _decoded_checkpoint_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Expose decoded row payloads while retaining durable envelope receipts."""

    decoded: list[JsonDict] = []
    for envelope in rows:
        item = deepcopy(dict(envelope))
        item["payload"] = _decode_checkpoint_payload(item["payload"])
        decoded.append(item)
    return decoded


def execute_cells(
    *,
    context: ExperimentContext,
    output_units: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    checkpoint_path: Path,
    manifest: Mapping[str, Any],
    stop_after_new_rows: int | None = None,
) -> JsonDict:
    """Fit paired seeds and append only pending cells to durable parent storage."""

    store = checkpointing.DurableRowCheckpoint(checkpoint_path, manifest)
    pending = set(store.pending(manifest["row_ids"]))
    train_units = [unit for unit in context.proposal_units if unit["split"] == "train"]
    new_row_count = 0
    training_receipts: list[JsonDict] = []
    for seed in seeds:
        seed_prefix = f"seed-{seed}|"
        if not any(seed_prefix in row_id for row_id in pending):
            continue
        models: dict[str, nn.Module] = {}
        receipts: dict[str, JsonDict] = {}
        for arm in ARMS:
            model, receipt = fit_arm(
                train_units,
                arm=arm,
                seed=int(seed),
                hyperparameters=FROZEN_HYPERPARAMETERS,
            )
            models[arm] = model
            receipts[arm] = receipt
            training_receipts.append(receipt)
        for unit in output_units:
            for arm in ARMS:
                row_id = f"{unit['unit_id']}|seed-{seed}|{arm}"
                if row_id not in pending:
                    continue
                raw = propose_raw_row(
                    models[arm],
                    unit,
                    arm=arm,
                    seed=int(seed),
                    parameter_count=trainable_parameter_count(models[arm]),
                    optimizer_update_count=receipts[arm]["optimizer_update_count"],
                    hyperparameters=FROZEN_HYPERPARAMETERS,
                )
                payload = attach_exact_outcomes(
                    raw,
                    context.exact_units[str(unit["unit_id"])],
                    context.hard_negatives[str(unit["unit_id"])],
                )
                envelope = checkpointing.complete_row_envelope(
                    row_id=row_id,
                    manifest_hash=store.manifest_hash,
                    payload=_encode_checkpoint_payload(payload),
                    attempt=1,
                    start_receipt={"unit_id": unit["unit_id"], "seed": seed, "arm": arm},
                    end_receipt={
                        "candidate_hashes": payload["candidate_hashes"],
                        "exact_evaluation_complete": True,
                    },
                )
                store.append(envelope)
                pending.remove(row_id)
                new_row_count += 1
                if stop_after_new_rows is not None and new_row_count >= stop_after_new_rows:
                    return {
                        "rows": _decoded_checkpoint_rows(store.rows),
                        "new_row_count": new_row_count,
                        "pending_row_ids": store.pending(manifest["row_ids"]),
                        "training_receipts": training_receipts,
                        "manifest_hash": store.manifest_hash,
                    }
    return {
        "rows": _decoded_checkpoint_rows(store.rows),
        "new_row_count": new_row_count,
        "pending_row_ids": store.pending(manifest["row_ids"]),
        "training_receipts": training_receipts,
        "manifest_hash": store.manifest_hash,
    }


def row_attribution_errors(
    rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> list[str]:
    """Check exact row identity, paired arms, and per-cell budget attribution."""

    errors: list[str] = []
    row_ids = [str(row.get("row_id")) for row in rows]
    expected_ids = [str(row_id) for row_id in manifest["row_ids"]]
    if len(row_ids) != len(set(row_ids)):
        errors.append("duplicate row IDs")
    if set(row_ids) != set(expected_ids):
        errors.append("row IDs do not match frozen manifest")
    pairs: dict[str, set[str]] = {}
    for row in rows:
        paired_key = str(row.get("paired_key"))
        pairs.setdefault(paired_key, set()).add(str(row.get("arm")))
        if row.get("row_id") != f"{paired_key}|{row.get('arm')}":
            errors.append(f"row identity mismatch: {row.get('row_id')}")
        if row.get("candidate_budget") != manifest["candidate_count_per_cell"]:
            errors.append(f"candidate budget mismatch: {row.get('row_id')}")
    expected_pairs = set(manifest["paired_keys"])
    if set(pairs) != expected_pairs:
        errors.append("paired keys do not match frozen manifest")
    if any(arms != set(ARMS) for arms in pairs.values()) or len(rows) != len(expected_pairs) * 2:
        errors.append("each paired key must contain both paired arms")
    return errors


def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 10) if values else None


def percentile(values: Sequence[float], quantile: float) -> float:
    """Return a linearly interpolated quantile from deterministic bootstrap values."""

    if not values:
        raise ValueError("percentile requires a nonempty value sequence")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 10)
    weight = position - lower
    return round(ordered[lower] * (1.0 - weight) + ordered[upper] * weight, 10)


def _arm_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    candidate_total = sum(int(row["candidate_budget"]) for row in rows)
    valid_total = sum(int(row["exact_valid_candidate_count"]) for row in rows)
    dependency_total = sum(int(row["cross_dependency_violation_count"]) for row in rows)
    valid_hashes = {
        str(outcome["candidate_hash"])
        for row in rows
        for outcome in row["exact_outcomes"]
        if outcome["exact_valid"]
    }
    aurocs = [
        float(row["hard_negative_discrimination"]["auroc"])
        for row in rows
        if row["hard_negative_discrimination"]["auroc"] is not None
    ]
    return {
        "row_count": len(rows),
        "candidate_count": candidate_total,
        "exact_valid_candidate_count": valid_total,
        "exact_valid_rate": round(valid_total / candidate_total, 10) if candidate_total else None,
        "cross_dependency_violation_count": dependency_total,
        "cross_dependency_violation_rate": (
            round(dependency_total / candidate_total, 10) if candidate_total else None
        ),
        "mean_nearest_valid_distance": _mean(
            [float(row["nearest_valid_distance"]) for row in rows]
        ),
        "converged_count": sum(row["stop_reason"] == "converged" for row in rows),
        "convergence_rate": (
            round(sum(row["stop_reason"] == "converged" for row in rows) / len(rows), 10)
            if rows
            else None
        ),
        "mean_iterations": _mean([float(row["iterations"]) for row in rows]),
        "mean_state_residual": _mean([float(row["state_residual"]) for row in rows]),
        "finite_value_failure_count": sum(not row["finite_values"] for row in rows),
        "runtime_total_s": round(sum(float(row["runtime_s"]) for row in rows), 6),
        "hard_negative_auroc": _mean(aurocs),
        "hard_negative_auroc_defined_row_count": len(aurocs),
        "unique_valid_support": len(valid_hashes),
    }


def _paired_values(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    by_pair: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        by_pair.setdefault(str(row["paired_key"]), {})[str(row["arm"])] = row
    values: list[JsonDict] = []
    for paired_key, arms in sorted(by_pair.items()):
        if set(arms) != set(ARMS):
            continue
        grouped = arms[GROUPED_ARM]
        flat = arms[FLAT_ARM]
        values.append(
            {
                "paired_key": paired_key,
                "unit_id": grouped["unit_id"],
                "topology_family": grouped["topology_family"],
                "exact_valid_delta": round(
                    float(grouped["exact_valid_rate"]) - float(flat["exact_valid_rate"]), 10
                ),
                "dependency_violation_delta": round(
                    float(grouped["cross_dependency_violation_rate"])
                    - float(flat["cross_dependency_violation_rate"]),
                    10,
                ),
                "distance_to_valid_delta": round(
                    float(grouped["nearest_valid_distance"])
                    - float(flat["nearest_valid_distance"]),
                    10,
                ),
            }
        )
    return values


def _effect_means(values: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "exact_valid_delta": _mean([float(value["exact_valid_delta"]) for value in values]),
        "dependency_violation_delta": _mean(
            [float(value["dependency_violation_delta"]) for value in values]
        ),
        "distance_to_valid_delta": _mean(
            [float(value["distance_to_valid_delta"]) for value in values]
        ),
    }


def _bootstrap_effects(
    paired: Sequence[Mapping[str, Any]], *, resamples: int, seed: int
) -> tuple[JsonDict, dict[str, JsonDict]]:
    families: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    for value in paired:
        families.setdefault(str(value["topology_family"]), {}).setdefault(
            str(value["unit_id"]), []
        ).append(value)
    point_by_family = {
        family: _effect_means([item for values in units.values() for item in values])
        for family, units in families.items()
    }
    point = {
        key: _mean([float(effect[key]) for effect in point_by_family.values()])
        for key in (
            "exact_valid_delta",
            "dependency_violation_delta",
            "distance_to_valid_delta",
        )
    }
    draws = {key: [] for key in point}
    family_draws = {family: {key: [] for key in point} for family in sorted(families)}
    generator = random.Random(seed)
    for _ in range(resamples):
        sampled_family_effects: dict[str, JsonDict] = {}
        for family in sorted(families):
            units = families[family]
            unit_ids = sorted(units)
            sampled = [generator.choice(unit_ids) for _ in unit_ids]
            sampled_values = [value for unit_id in sampled for value in units[unit_id]]
            family_effect = _effect_means(sampled_values)
            sampled_family_effects[family] = family_effect
            for key in draws:
                family_draws[family][key].append(float(family_effect[key]))
        for key in draws:
            draws[key].append(
                sum(float(effect[key]) for effect in sampled_family_effects.values())
                / len(sampled_family_effects)
            )
    ci = {
        key: {
            "lower": percentile(draws[key], 0.025) if draws[key] else point[key],
            "upper": percentile(draws[key], 0.975) if draws[key] else point[key],
            "confidence_level": 0.95,
            "resamples": resamples,
            "resampling_unit": "unit_inside_topology_family",
            "family_aggregation": "equal_weight_mean",
        }
        for key in draws
    }
    family_ci = {
        family: {
            key: {
                "lower": percentile(values, 0.025) if values else point_by_family[family][key],
                "upper": percentile(values, 0.975) if values else point_by_family[family][key],
                "confidence_level": 0.95,
                "resamples": resamples,
                "resampling_unit": "unit",
            }
            for key, values in effect_draws.items()
        }
        for family, effect_draws in family_draws.items()
    }
    return {"point": point, "ci": ci}, family_ci


def aggregate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> JsonDict:
    """Derive arm, topology, support, convergence, and paired bootstrap metrics."""

    metrics_by_arm = {arm: _arm_metrics([row for row in rows if row["arm"] == arm]) for arm in ARMS}
    paired = _paired_values(rows)
    bootstrap, family_ci = _bootstrap_effects(
        paired, resamples=bootstrap_resamples, seed=bootstrap_seed
    )
    metrics_by_topology: JsonDict = {}
    for family in sorted({str(row["topology_family"]) for row in rows}):
        family_rows = [row for row in rows if row["topology_family"] == family]
        family_values = [value for value in paired if value["topology_family"] == family]
        effects = _effect_means(family_values)
        metrics_by_topology[family] = {
            "split": sorted({str(row["split"]) for row in family_rows})[0],
            "unit_count": len({str(row["unit_id"]) for row in family_rows}),
            "paired_key_count": len(family_values),
            "metrics_by_arm": {
                arm: _arm_metrics([row for row in family_rows if row["arm"] == arm]) for arm in ARMS
            },
            "paired_exact_valid_delta": effects["exact_valid_delta"],
            "paired_exact_valid_delta_ci95": family_ci[family]["exact_valid_delta"],
            "dependency_violation_delta": effects["dependency_violation_delta"],
            "distance_to_valid_delta": effects["distance_to_valid_delta"],
        }
    grouped_support = int(metrics_by_arm[GROUPED_ARM]["unique_valid_support"])
    flat_support = int(metrics_by_arm[FLAT_ARM]["unique_valid_support"])
    contraction = max(0.0, (flat_support - grouped_support) / flat_support) if flat_support else 0.0
    return {
        "metrics_by_arm": metrics_by_arm,
        "metrics_by_topology": metrics_by_topology,
        "paired_exact_valid_delta": bootstrap["point"]["exact_valid_delta"],
        "paired_exact_valid_delta_ci95": bootstrap["ci"]["exact_valid_delta"],
        "dependency_violation_delta": bootstrap["point"]["dependency_violation_delta"],
        "distance_to_valid_delta": bootstrap["point"]["distance_to_valid_delta"],
        "hard_negative_auroc_by_arm": {
            arm: metrics_by_arm[arm]["hard_negative_auroc"] for arm in ARMS
        },
        "unique_valid_support_by_arm": {
            arm: metrics_by_arm[arm]["unique_valid_support"] for arm in ARMS
        },
        "support_contraction": round(contraction, 10),
        "paired_key_count": len(paired),
    }


def _aggregation_row(row: Mapping[str, Any]) -> JsonDict:
    """Keep only row fields that the cold aggregate is allowed to consume."""

    keys = (
        "row_id",
        "paired_key",
        "unit_id",
        "split",
        "topology_family",
        "arm",
        "candidate_budget",
        "exact_valid_candidate_count",
        "cross_dependency_violation_count",
        "exact_valid_rate",
        "cross_dependency_violation_rate",
        "nearest_valid_distance",
        "stop_reason",
        "iterations",
        "state_residual",
        "finite_values",
        "runtime_s",
        "hard_negative_discrimination",
        "exact_outcomes",
    )
    return {key: deepcopy(row[key]) for key in keys}


def cold_recompute_payload(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Build the row-only payload consumed by a fresh aggregate process."""

    return {
        "rows": [_aggregation_row(row) for row in rows],
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def _cold_recompute_worker() -> int:
    """Read row-only JSON and emit independently rebuilt aggregate metrics."""

    payload = json.load(sys.stdin)
    aggregates = aggregate_rows(
        payload["rows"],
        bootstrap_resamples=int(payload["bootstrap_resamples"]),
        bootstrap_seed=int(payload["bootstrap_seed"]),
    )
    print(
        canonical_json(
            {
                "aggregates": aggregates,
                "aggregate_hash": sha256_json(aggregates),
                "worker_pid": os.getpid(),
            }
        )
    )
    return 0


def run_cold_recompute(rows: Sequence[Mapping[str, Any]], *, repo_root: Path) -> JsonDict:
    """Rebuild aggregate evidence in a fresh interpreter with no model objects."""

    environment = os.environ.copy()
    python_path = str(repo_root / "python")
    environment["PYTHONPATH"] = (
        python_path
        if not environment.get("PYTHONPATH")
        else f"{python_path}{os.pathsep}{environment['PYTHONPATH']}"
    )
    process = subprocess.run(
        [sys.executable, "-m", __name__, "--cold-recompute-worker"],
        input=canonical_json(cold_recompute_payload(rows)),
        text=True,
        capture_output=True,
        cwd=repo_root,
        env=environment,
        timeout=120,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(f"cold aggregate failed: {process.stderr.strip()}")
    return json.loads(process.stdout)


AGGREGATE_FIELDS = (
    "metrics_by_arm",
    "metrics_by_topology",
    "paired_exact_valid_delta",
    "paired_exact_valid_delta_ci95",
    "dependency_violation_delta",
    "distance_to_valid_delta",
    "hard_negative_auroc_by_arm",
    "unique_valid_support_by_arm",
    "support_contraction",
    "paired_key_count",
)


def _artifact_aggregates(artifact: Mapping[str, Any]) -> JsonDict:
    return {key: deepcopy(artifact.get(key)) for key in AGGREGATE_FIELDS}


def headline_consistency_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Recompute point headlines from rows and bind bootstrap fields to the cold hash."""

    rows = artifact.get("rows", [])
    if not rows:
        return []
    point = aggregate_rows(rows, bootstrap_resamples=0, bootstrap_seed=BOOTSTRAP_SEED)
    errors: list[str] = []
    for key in (
        "metrics_by_arm",
        "paired_exact_valid_delta",
        "dependency_violation_delta",
        "distance_to_valid_delta",
        "hard_negative_auroc_by_arm",
        "unique_valid_support_by_arm",
        "support_contraction",
    ):
        if artifact.get(key) != point[key]:
            errors.append(f"headline metrics do not match rows: {key}")
    for family, metrics in point["metrics_by_topology"].items():
        actual = artifact.get("metrics_by_topology", {}).get(family, {})
        for key in (
            "split",
            "unit_count",
            "paired_key_count",
            "metrics_by_arm",
            "paired_exact_valid_delta",
            "dependency_violation_delta",
            "distance_to_valid_delta",
        ):
            if actual.get(key) != metrics[key]:
                errors.append(f"headline metrics do not match rows: {family}.{key}")
    declared_hash = artifact.get("cold_recompute_agreement", {}).get("producer_aggregate_hash")
    if declared_hash != sha256_json(_artifact_aggregates(artifact)):
        errors.append("cold aggregate hash does not match headline fields")
    return errors


def _without_runtime(value: Any) -> Any:
    """Remove measured timing and process identity from stable replay material."""

    excluded = {
        "duration_s",
        "runtime_s",
        "runtime_total_s",
        "producer_pid",
        "worker_pid",
        "checkpoint_path",
    }
    if isinstance(value, Mapping):
        return {
            str(key): _without_runtime(item)
            for key, item in value.items()
            if key not in excluded and key != "reproducibility_checksum"
        }
    if isinstance(value, list):
        return [_without_runtime(item) for item in value]
    return value


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable source, row, and decision evidence while excluding wall time."""

    return sha256_json(_without_runtime(artifact))


def _empty_metrics_by_arm() -> JsonDict:
    return {
        arm: {
            "row_count": 0,
            "candidate_count": 0,
            "exact_valid_candidate_count": 0,
            "exact_valid_rate": None,
            "cross_dependency_violation_count": 0,
            "cross_dependency_violation_rate": None,
            "mean_nearest_valid_distance": None,
            "converged_count": 0,
            "convergence_rate": None,
            "mean_iterations": None,
            "mean_state_residual": None,
            "finite_value_failure_count": 0,
            "runtime_total_s": 0.0,
            "hard_negative_auroc": None,
            "hard_negative_auroc_defined_row_count": 0,
            "unique_valid_support": 0,
        }
        for arm in ARMS
    }


def _blocked_artifact(
    *, run_date: str, duration_s: float, preconditions: Mapping[str, Any]
) -> JsonDict:
    """Return the complete schema with no fallback rows after a failed gate."""

    first = preconditions.get("first_failure") or {"check": "preconditions", "observed": None}
    parameter_count = grouped_source.expected_parameter_count(
        int(FROZEN_HYPERPARAMETERS["hidden_width"])
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "status": "complete_blocked_fixed_point_control_ab",
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(preconditions.get("source_artifact_hashes", {})),
        "frozen_manifest": {
            "schema": "carnot.experiment_6788.frozen_manifest.v1",
            "source_manifest_hash": EXPECTED_EXP6786_MANIFEST_HASH,
            "seeds": deepcopy(FROZEN_SEEDS),
            "arms": list(ARMS),
            "planned_row_count": PLANNED_ROW_COUNT,
            "candidate_count_per_cell": FROZEN_HYPERPARAMETERS["candidate_count"],
            "cpu_wall_budget_s": CPU_WALL_BUDGET_S,
            "support_contraction_margin": SUPPORT_CONTRACTION_MARGIN,
        },
        "arm_definitions": arm_definitions(),
        "parameter_counts_by_arm": {arm: parameter_count for arm in ARMS},
        "optimization_steps_by_arm": {
            arm: len(FROZEN_SEEDS) * int(FROZEN_HYPERPARAMETERS["training_steps"]) for arm in ARMS
        },
        "candidate_budget_by_arm": {
            arm: OUTPUT_UNIT_COUNT
            * len(FROZEN_SEEDS)
            * int(FROZEN_HYPERPARAMETERS["candidate_count"])
            for arm in ARMS
        },
        "rows": [],
        "metrics_by_arm": _empty_metrics_by_arm(),
        "metrics_by_topology": {},
        "paired_exact_valid_delta": None,
        "paired_exact_valid_delta_ci95": None,
        "dependency_violation_delta": None,
        "distance_to_valid_delta": None,
        "hard_negative_auroc_by_arm": {arm: None for arm in ARMS},
        "unique_valid_support_by_arm": {arm: 0 for arm in ARMS},
        "support_contraction": None,
        "paired_key_count": 0,
        "fixed_point_comparison_completed": False,
        "checkpoint_receipt": {
            "attempted": False,
            "planned_row_count": PLANNED_ROW_COUNT,
            "completed_row_count": 0,
            "complete": False,
        },
        "cold_recompute_agreement": {
            "attempted": False,
            "agreement": False,
            "producer_aggregate_hash": None,
            "replay_aggregate_hash": None,
        },
        "decision_gates": {
            "positive": False,
            "preconditions_passed": False,
        },
        "gate_check_summary": deepcopy(dict(preconditions)),
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": (
            f"complete_blocked_fixed_point_control_ab: {first['check']} observed "
            f"{first.get('observed')}"
        ),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _validate_run_date(run_date: str) -> None:
    if len(run_date) != 8 or not run_date.isdigit():
        raise ValueError("run date must use YYYYMMDD")


def _comparison_gates(
    aggregates: Mapping[str, Any],
    parameter_counts: Mapping[str, int],
    optimization_steps: Mapping[str, int],
    candidate_budgets: Mapping[str, int],
    *,
    duration_s: float,
) -> JsonDict:
    grouped_metrics = aggregates["metrics_by_arm"][GROUPED_ARM]
    flat_metrics = aggregates["metrics_by_arm"][FLAT_ARM]
    parameter_fraction = abs(
        int(parameter_counts[GROUPED_ARM]) - int(parameter_counts[FLAT_ARM])
    ) / max(int(parameter_counts[GROUPED_ARM]), int(parameter_counts[FLAT_ARM]), 1)
    parameter_matching = parameter_fraction <= PARAMETER_MATCH_TOLERANCE
    compute_matching = bool(
        len(set(optimization_steps.values())) == 1
        and len(set(candidate_budgets.values())) == 1
        and duration_s <= CPU_WALL_BUDGET_S
        and all(
            float(grouped_metrics[key]) == float(flat_metrics[key])
            for key in ("row_count", "candidate_count")
        )
    )
    convergence_harm = bool(
        float(grouped_metrics["convergence_rate"]) < float(flat_metrics["convergence_rate"])
        or int(grouped_metrics["finite_value_failure_count"])
        > int(flat_metrics["finite_value_failure_count"])
        or float(grouped_metrics["mean_iterations"]) > float(flat_metrics["mean_iterations"])
    )
    exact_lcb = float(aggregates["paired_exact_valid_delta_ci95"]["lower"])
    support_ok = float(aggregates["support_contraction"]) <= SUPPORT_CONTRACTION_MARGIN
    positive = bool(
        exact_lcb > 0.0
        and parameter_matching
        and compute_matching
        and support_ok
        and not convergence_harm
    )
    return {
        "grouped_minus_flat_exact_valid_lcb_gt_zero": exact_lcb > 0.0,
        "paired_exact_valid_delta_lcb": exact_lcb,
        "parameter_matching_within_five_percent": parameter_matching,
        "parameter_difference_fraction": round(parameter_fraction, 10),
        "compute_matching": compute_matching,
        "support_within_frozen_margin": support_ok,
        "support_contraction_margin": SUPPORT_CONTRACTION_MARGIN,
        "no_convergence_harm": not convergence_harm,
        "positive": positive,
    }


def build_artifact(
    *,
    run_date: str = RUN_DATE,
    repo_root: Path = REPO_ROOT,
    source_6786_path: Path | None = None,
    source_6787_path: Path | None = None,
    checkpoint_path: Path | None = None,
    duration_s: float | None = None,
) -> JsonDict:
    """Run every paired cell or return the complete no-fallback blocked schema."""

    _validate_run_date(run_date)
    started = time.monotonic()
    path_6786 = source_6786_path or repo_root / SOURCE_6786_RELATIVE_PATH
    path_6787 = source_6787_path or repo_root / SOURCE_6787_RELATIVE_PATH
    preconditions = evaluate_preconditions(
        repo_root=repo_root,
        source_6786_path=path_6786,
        source_6787_path=path_6787,
    )
    measured = round(time.monotonic() - started, 6) if duration_s is None else duration_s
    if not preconditions["all_passed"]:
        artifact = _blocked_artifact(
            run_date=run_date, duration_s=measured, preconditions=preconditions
        )
        errors = validate_artifact(artifact)
        if errors:  # pragma: no cover - construction and validation share one contract.
            raise ValueError("; ".join(errors))
        return artifact

    source_6786 = load_json_object(path_6786)
    source_6787 = load_json_object(path_6787)
    context = build_context(source_6786, source_6787)
    output_units = [
        unit for unit in context.proposal_units if unit["split"] in grouped_source.OUTPUT_SPLITS
    ]
    manifest = frozen_manifest(output_units)
    checkpoint = checkpoint_path or repo_root / CHECKPOINT_RELATIVE_PATH
    execution = execute_cells(
        context=context,
        output_units=output_units,
        seeds=FROZEN_SEEDS,
        checkpoint_path=checkpoint,
        manifest=manifest,
    )
    rows = [deepcopy(envelope["payload"]) for envelope in execution["rows"]]
    attribution_errors = row_attribution_errors(rows, manifest)
    aggregates = aggregate_rows(rows)
    cold = run_cold_recompute(rows, repo_root=repo_root)
    producer_aggregate_hash = sha256_json(aggregates)
    cold_agreement = {
        "attempted": True,
        "agreement": cold["aggregates"] == aggregates,
        "fresh_process": cold["worker_pid"] != os.getpid(),
        "producer_aggregate_hash": producer_aggregate_hash,
        "replay_aggregate_hash": cold["aggregate_hash"],
        "producer_pid": os.getpid(),
        "worker_pid": cold["worker_pid"],
    }
    parameter_counts = {
        arm: grouped_source.expected_parameter_count(int(FROZEN_HYPERPARAMETERS["hidden_width"]))
        for arm in ARMS
    }
    optimization_steps = {
        arm: len(FROZEN_SEEDS) * int(FROZEN_HYPERPARAMETERS["training_steps"]) for arm in ARMS
    }
    candidate_budgets = {
        arm: OUTPUT_UNIT_COUNT * len(FROZEN_SEEDS) * int(FROZEN_HYPERPARAMETERS["candidate_count"])
        for arm in ARMS
    }
    measured = round(time.monotonic() - started, 6) if duration_s is None else duration_s
    gates = _comparison_gates(
        aggregates,
        parameter_counts,
        optimization_steps,
        candidate_budgets,
        duration_s=measured,
    )
    completed = bool(
        not attribution_errors
        and not execution["pending_row_ids"]
        and len(rows) == PLANNED_ROW_COUNT
        and cold_agreement["agreement"]
        and cold_agreement["fresh_process"]
    )
    gates["preconditions_passed"] = True
    gates["all_rows_attributable"] = not attribution_errors
    gates["cold_recompute_agreement"] = cold_agreement["agreement"]
    gates["positive"] = bool(gates["positive"] and completed)
    verdict_class = "positive" if gates["positive"] else "null" if completed else "partial"
    honest_verdict = (
        "complete: grouped fixed-point proposal has a positive paired structural effect"
        if verdict_class == "positive"
        else (
            "complete: paired fixed-point comparison finished; grouped lower confidence bound "
            "did not exceed zero or a secondary safety gate failed"
            if verdict_class == "null"
            else "complete_partial_fixed_point_control_ab: not every planned cell cold-recomputed"
        )
    )
    gate_summary = deepcopy(preconditions)
    gate_summary["attribution_errors"] = attribution_errors
    gate_summary["comparison_checks"] = deepcopy(gates)
    artifact = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "status": "complete" if completed else "complete_partial_fixed_point_control_ab",
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": measured,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(preconditions["source_artifact_hashes"]),
        "frozen_manifest": manifest,
        "arm_definitions": arm_definitions(),
        "parameter_counts_by_arm": parameter_counts,
        "optimization_steps_by_arm": optimization_steps,
        "candidate_budget_by_arm": candidate_budgets,
        "rows": rows,
        **{key: aggregates[key] for key in AGGREGATE_FIELDS},
        "fixed_point_comparison_completed": completed,
        "checkpoint_receipt": {
            "attempted": True,
            "checkpoint_path": str(checkpoint),
            "manifest_hash": execution["manifest_hash"],
            "planned_row_count": PLANNED_ROW_COUNT,
            "completed_row_count": len(rows),
            "new_row_count": execution["new_row_count"],
            "pending_row_ids": execution["pending_row_ids"],
            "complete": not execution["pending_row_ids"],
            "payload_hashes": [envelope["payload_hash"] for envelope in execution["rows"]],
        },
        "cold_recompute_agreement": cold_agreement,
        "decision_gates": gates,
        "gate_check_summary": gate_summary,
        "verifier_is_oracle": False,
        "verdict_class": verdict_class,
        "honest_verdict": honest_verdict,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:  # pragma: no cover - construction and validation share one contract.
        raise ValueError("; ".join(errors))
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return schema, attribution, headline, and terminal-state errors."""

    errors: list[str] = []
    if set(artifact) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("required field set mismatch")
    if set(artifact.get("field_principles", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field principle coverage mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference substrate mismatch")
    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or duration < 0:
        errors.append("duration_s must be non-negative")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random seed mismatch")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        errors.append("verdict class is outside the closed enum")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest verdict lacks a terminal prefix")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must remain false")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility checksum mismatch")

    blocked = artifact.get("verdict_class") == "blocked"
    if blocked:
        if artifact.get("status") != "complete_blocked_fixed_point_control_ab":
            errors.append("blocked artifact status mismatch")
        if artifact.get("rows") != []:
            errors.append("blocked artifact must not contain rows")
        if artifact.get("fixed_point_comparison_completed") is not False:
            errors.append("blocked artifact cannot be complete")
        if artifact.get("gate_check_summary", {}).get("all_passed") is not False:
            errors.append("blocked artifact must name failed preconditions")
        return errors

    completed = artifact.get("fixed_point_comparison_completed") is True
    if completed != (artifact.get("status") == "complete"):
        errors.append("complete artifact completion flag mismatch")
    if completed:
        if len(artifact.get("rows", [])) != PLANNED_ROW_COUNT:
            errors.append("complete artifact row count mismatch")
        manifest = artifact.get("frozen_manifest", {})
        attribution = row_attribution_errors(artifact.get("rows", []), manifest)
        if attribution:
            errors.append(f"row attribution failed: {attribution[0]}")
        if artifact.get("gate_check_summary", {}).get("all_passed") is not True:
            errors.append("complete artifact has failed preconditions")
        if artifact.get("cold_recompute_agreement", {}).get("agreement") is not True:
            errors.append("complete artifact lacks cold recompute agreement")
        expected_parameters = {
            arm: grouped_source.expected_parameter_count(
                int(FROZEN_HYPERPARAMETERS["hidden_width"])
            )
            for arm in ARMS
        }
        if artifact.get("parameter_counts_by_arm") != expected_parameters:
            errors.append("parameter counts do not match frozen architectures")
        errors.extend(headline_consistency_errors(artifact))
        positive = artifact.get("verdict_class") == "positive"
        if positive != bool(artifact.get("decision_gates", {}).get("positive")):
            errors.append("positive verdict does not match decision gates")
    return errors


def write_outputs(
    *,
    run_date: str = RUN_DATE,
    repo_root: Path = REPO_ROOT,
    source_6786_path: Path | None = None,
    source_6787_path: Path | None = None,
    artifact_path: Path = RESULT_RELATIVE_PATH,
    checkpoint_path: Path = CHECKPOINT_RELATIVE_PATH,
) -> JsonDict:
    """Write one validated artifact and retain its task-owned checkpoint."""

    output = artifact_path if artifact_path.is_absolute() else repo_root / artifact_path
    checkpoint = checkpoint_path if checkpoint_path.is_absolute() else repo_root / checkpoint_path
    artifact = build_artifact(
        run_date=run_date,
        repo_root=repo_root,
        source_6786_path=source_6786_path,
        source_6787_path=source_6787_path,
        checkpoint_path=checkpoint,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpointing.atomic_write_json(output, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the frozen experiment CLI and print its terminal verdict."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--source-6786", type=Path, default=SOURCE_6786_RELATIVE_PATH)
    parser.add_argument("--source-6787", type=Path, default=SOURCE_6787_RELATIVE_PATH)
    parser.add_argument("--artifact-path", type=Path, default=RESULT_RELATIVE_PATH)
    parser.add_argument("--checkpoint-path", type=Path, default=CHECKPOINT_RELATIVE_PATH)
    parser.add_argument("--cold-recompute-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.cold_recompute_worker:
        return _cold_recompute_worker()
    source_6786 = (
        args.source_6786 if args.source_6786.is_absolute() else REPO_ROOT / args.source_6786
    )
    source_6787 = (
        args.source_6787 if args.source_6787.is_absolute() else REPO_ROOT / args.source_6787
    )
    artifact = write_outputs(
        run_date=args.date,
        repo_root=REPO_ROOT,
        source_6786_path=source_6786,
        source_6787_path=source_6787,
        artifact_path=args.artifact_path,
        checkpoint_path=args.checkpoint_path,
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the repository entry point.
    raise SystemExit(main())
