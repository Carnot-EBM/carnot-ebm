"""Build bounded soft fixed-point proposals from Exp6786 graph structure.

Spec refs: REQ-VERIFY-6787 and SCENARIO-VERIFY-6787-*.

The model uses only the proposal projection declared by Exp6786. The exact
assignments and checker receipts stay in the source artifact for later audits,
but this module never passes them to training, recurrence, stopping, or decode.
The result measures mechanism readiness only. It does not measure correctness.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch
from torch import nn


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_6787_group_aware_soft_fixed_point"
SCHEMA = "carnot.experiment_6787.group_aware_soft_fixed_point.v1"
ROW_SCHEMA = "carnot.experiment_6787.soft_fixed_point_row.v1"
RUN_DATE = "20260830"
RANDOM_SEED = 6_787_000
INFERENCE_SUBSTRATE = "cpu_pytorch_soft_fixed_point_proposal_no_llm"
SOURCE_ARTIFACT_RELATIVE_PATH = Path(
    "results/experiment_6786_constraint_dependency_hard_negative_fixture.json"
)
RESULT_RELATIVE_PATH = Path("results") / f"{EXPERIMENT_ID}.json"
EXPECTED_SOURCE_ARTIFACT_HASH = (
    "sha256:f3780c85e29cda8dbd897b6c43a0ce3c938252625e823e54107f918d2052514a"
)
REQUIRED_SPLITS = ("train", "development", "held_topology_test")
OUTPUT_SPLITS = ("development", "held_topology_test")
FEATURE_ALLOWLIST = (
    "schema",
    "graph_id",
    "topology_family",
    "difficulty_stratum",
    "variables",
    "local_groups",
    "dependency_edges",
)
FEATURE_DENYLIST = (
    "exact_assignments",
    "exact_valid",
    "future_receipts",
    "solver_conflicts",
    "oracle_residuals",
    "post_action_outcomes",
    "exact_certificate",
    "exact_label",
    "ground_truth_assignment",
    "target_assignment",
    "target_assignments",
    "exact_checker_feedback",
    "cold_replay_receipt",
    "group_receipts",
    "dependency_receipts",
    "failed_cross_dependencies",
    "post_action_outcome",
    "oracle_residual",
    "future_rows",
)
FROZEN_HYPERPARAMETERS: JsonDict = {
    "seeds": [6_787_001, 6_787_002, 6_787_003, 6_787_004, 6_787_005],
    "training_steps": 6,
    "optimizer": "Adam",
    "learning_rate": 0.02,
    "hidden_width": 8,
    "training_unroll_steps": 3,
    "iteration_cap": 6,
    "convergence_tolerance": 1e-4,
    "decoding_threshold": 0.5,
    "candidate_count": 3,
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
    "source_artifact_hash",
    "frozen_hyperparameters",
    "trainable_parameter_count",
    "feature_allowlist",
    "feature_denylist",
    "oracle_feature_violations",
    "training_receipts",
    "rows",
    "convergence_by_split",
    "finite_value_failures",
    "deterministic_replay_agreement",
    "candidate_hashes",
    "soft_fixed_point_proposer_ready",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
FIELD_PRINCIPLES: dict[str, str] = {
    "schema": "A versioned schema makes incompatible proposal artifacts fail closed.",
    "experiment_id": "A stable ID binds the result to the planned bounded mechanism.",
    "run_date": "The execution date distinguishes this frozen run from later replays.",
    "status": "Status separates a completed proposal run from a complete blocked run.",
    "field_principles": "One-line purposes make every required field auditable.",
    "inference_substrate": "The CPU PyTorch declaration prevents an LLM or solver claim.",
    "duration_s": "Measured wall time shows that fitting and proposal actually ran.",
    "random_seed": "The master seed anchors the five preregistered model seeds.",
    "reproducibility_checksum": "A stable hash detects input, telemetry, or candidate drift.",
    "source_artifact_hash": "The exact byte hash prevents silent Exp6786 fixture changes.",
    "frozen_hyperparameters": "Frozen settings prevent held topology results from tuning the run.",
    "trainable_parameter_count": "The count bounds the size of the proposal mechanism.",
    "feature_allowlist": "The allowlist defines every field permitted to reach the proposer.",
    "feature_denylist": "The denylist names oracle and future fields that must stay outside it.",
    "oracle_feature_violations": "An empty list proves the proposal projection passed its audit.",
    "training_receipts": "Per-seed receipts prove that fitting saw train topology units only.",
    "rows": "One unit-seed row preserves convergence, messages, and candidate assignments.",
    "convergence_by_split": "Split aggregates expose convergence without a correctness claim.",
    "finite_value_failures": "Any NaN or infinity makes mechanism readiness fail closed.",
    "deterministic_replay_agreement": "A fresh process must reproduce candidate hashes exactly.",
    "candidate_hashes": "Content hashes bind every decoded assignment without oracle scoring.",
    "soft_fixed_point_proposer_ready": "This gate authorizes Exp6788 mechanism consumption only.",
    "gate_check_summary": "Each precondition records its expected and observed value.",
    "verifier_is_oracle": "False states that this proposer has no correctness authority.",
    "verdict_class": "A closed class keeps the terminal mechanism result machine-readable.",
    "honest_verdict": "A terminal prefix reports readiness without superiority or correctness.",
}


def canonical_json(value: Any) -> str:
    """Serialize stable content so hashes do not depend on JSON formatting."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    """Return an algorithm-labelled hash of canonical JSON."""

    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def sha256_file(path: Path) -> str | None:
    """Return an algorithm-labelled file hash, or no hash when the file is absent."""

    if not path.is_file():
        return None
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def load_json_object(path: Path) -> JsonDict:
    """Load a JSON object and reject roots that cannot satisfy the artifact contract."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _gate(check: str, expected: Any, observed: Any, passed: bool | None = None) -> JsonDict:
    """Build one inspectable precondition row."""

    return {
        "check": check,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected if passed is None else bool(passed),
    }


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name all failures and the first failure without discarding later evidence."""

    copied = [deepcopy(dict(check)) for check in checks]
    failed = [str(check["check"]) for check in copied if check.get("passed") is not True]
    first = next((check for check in copied if check.get("passed") is not True), None)
    return {
        "all_passed": not failed,
        "checks": copied,
        "failed_checks": failed,
        "first_failure": first,
    }


def _walk_keys(value: Any, prefix: str) -> list[tuple[str, str]]:
    """Return nested key paths so forbidden fields cannot hide in group records."""

    found: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            found.append((path, str(key)))
            found.extend(_walk_keys(nested, path))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            found.extend(_walk_keys(nested, f"{prefix}[{index}]"))
    return found


def audit_feature_contract(units: Sequence[Mapping[str, Any]]) -> list[str]:
    """Report every non-allowlisted top-level key or nested oracle key."""

    violations: list[str] = []
    denied = set(FEATURE_DENYLIST)
    allowed = set(FEATURE_ALLOWLIST)
    for unit in units:
        unit_id = str(unit.get("unit_id", "unknown-unit"))
        features = unit.get("proposal_features", {})
        if not isinstance(features, Mapping):
            violations.append(f"{unit_id}.proposal_features")
            continue
        for key in features:
            if key not in allowed:
                violations.append(f"{unit_id}.proposal_features.{key}")
        for path, key in _walk_keys(features, f"{unit_id}.proposal_features"):
            if key in denied and path not in violations:
                violations.append(path)
    return sorted(violations)


def project_units(source: Mapping[str, Any]) -> list[JsonDict]:
    """Copy graph structure into a schema that contains no exact checker outputs."""

    projected: list[JsonDict] = []
    for unit in source.get("frozen_manifest", {}).get("units", []):
        graph = unit["graph"]
        features: JsonDict = {
            "schema": "carnot.experiment_6786.proposal_features.v1",
            "graph_id": unit["graph_id"],
            "topology_family": unit["topology_family"],
            "difficulty_stratum": unit["difficulty_stratum"],
            "variables": deepcopy(graph["variables"]),
            "local_groups": [
                {
                    "group_id": group["group_id"],
                    "group_type": group["group_type"],
                    "variables": deepcopy(group["variables"]),
                }
                for group in graph["local_groups"]
            ],
            "dependency_edges": deepcopy(graph["dependency_edges"]),
        }
        projected.append(
            {
                "unit_id": unit["unit_id"],
                "graph_id": unit["graph_id"],
                "graph_serialization": canonical_json(features),
                "split": unit["split"],
                "topology_family": unit["topology_family"],
                "difficulty_stratum": unit["difficulty_stratum"],
                "group_ids": [group["group_id"] for group in features["local_groups"]],
                "variable_ids": deepcopy(features["variables"]),
                "dependency_ids": [edge["dependency_id"] for edge in features["dependency_edges"]],
                "proposal_features": features,
            }
        )
    return projected


def evaluate_preconditions(
    source_artifact_path: Path,
    *,
    expected_hash: str = EXPECTED_SOURCE_ARTIFACT_HASH,
) -> JsonDict:
    """Check exact input bytes, readiness, split denominators, and feature isolation."""

    exists = source_artifact_path.is_file()
    actual_hash = sha256_file(source_artifact_path)
    source = load_json_object(source_artifact_path) if exists else {}
    projected = project_units(source) if exists else []
    source_rows = source.get("rows", []) if exists else []
    row_features = [
        {"unit_id": row.get("unit_id"), "proposal_features": row.get("proposal_features")}
        for row in source_rows
    ]
    feature_violations = sorted(
        set(source.get("future_feature_violations", []))
        | set(audit_feature_contract(projected))
        | set(audit_feature_contract(row_features))
    )
    split_observations: dict[str, JsonDict] = {}
    for split in REQUIRED_SPLITS:
        declared = source.get("split_by_topology", {}).get(split, {}).get("unit_count")
        projected_count = sum(unit.get("split") == split for unit in projected)
        split_observations[split] = {
            "declared_unit_count": declared,
            "projected_unit_count": projected_count,
        }
    checks = [
        _gate("exp6786_artifact_exists", True, exists),
        _gate(
            "constraint_group_fixture_ready",
            True,
            source.get("constraint_group_fixture_ready") if exists else None,
        ),
        _gate("exact_artifact_hash_agreement", expected_hash, actual_hash),
        *[
            _gate(
                f"{split}_split_nonempty",
                "declared and projected unit counts > 0",
                split_observations[split],
                bool(
                    isinstance(split_observations[split]["declared_unit_count"], int)
                    and split_observations[split]["declared_unit_count"] > 0
                    and split_observations[split]["projected_unit_count"] > 0
                ),
            )
            for split in REQUIRED_SPLITS
        ],
        _gate("feature_denylist_clean", [], feature_violations),
    ]
    summary = _gate_summary(checks)
    summary["oracle_feature_violations"] = feature_violations
    return summary


def _configure_torch(seed: int) -> None:
    """Use one CPU thread and deterministic kernels for byte-stable replay."""

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def expected_parameter_count(hidden_width: int) -> int:
    """Compute the fixed two-layer update-network size without building a model."""

    if hidden_width <= 0:
        raise ValueError("hidden_width must be positive")
    return (8 * hidden_width + hidden_width) + (hidden_width * 2 + 2) + 1


@dataclass(frozen=True)
class RecurrentStep:
    """Keep variable state and both message types separate for inspection."""

    variable_state: torch.Tensor
    group_messages: torch.Tensor
    dependency_messages: torch.Tensor
    aggregated_dependency_messages: torch.Tensor


class GroupAwareSoftFixedPoint(nn.Module):
    """Small recurrent update whose topology comes only from declared groups."""

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

    @staticmethod
    def _group_messages(state: torch.Tensor) -> torch.Tensor:
        """Sharpen each two-variable one-hot group with a soft conjunction."""

        messages = state * (1.0 - state.flip(dims=(1,)))
        return messages / messages.sum(dim=1, keepdim=True).clamp_min(1e-12)

    @staticmethod
    def _dependency_messages(
        state: torch.Tensor, features: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build edge messages and aggregate incoming messages with soft OR."""

        groups = features["local_groups"]
        group_index = {str(group["group_id"]): index for index, group in enumerate(groups)}
        edges = features["dependency_edges"]
        messages: list[torch.Tensor] = []
        incoming: list[list[torch.Tensor]] = [[] for _ in groups]
        incoming_degree = torch.zeros((len(groups), 1), dtype=torch.float64)
        outgoing_degree = torch.zeros((len(groups), 1), dtype=torch.float64)
        for edge in edges:
            source_index = group_index[str(edge["source_group"])]
            target_index = group_index[str(edge["target_group"])]
            source = state[source_index]
            relation = str(edge["relation_type"])
            if relation == "implies_selected_one":
                message = torch.stack((1.0 - source[1], torch.ones_like(source[1])))
            elif relation == "implies_selected_zero":
                message = torch.stack((torch.ones_like(source[0]), 1.0 - source[0]))
            else:
                raise ValueError(f"unknown dependency relation: {relation}")
            message = message / message.sum().clamp_min(1e-12)
            messages.append(message)
            incoming[target_index].append(message)
            incoming_degree[target_index, 0] += 1.0
            outgoing_degree[source_index, 0] += 1.0
        edge_messages = (
            torch.stack(messages)
            if messages
            else torch.empty((0, 2), dtype=torch.float64, device="cpu")
        )
        aggregated: list[torch.Tensor] = []
        for group_messages in incoming:
            if group_messages:
                stacked = torch.stack(group_messages)
                combined = 1.0 - torch.prod(1.0 - stacked, dim=0)
                aggregated.append(combined / combined.sum().clamp_min(1e-12))
            else:
                aggregated.append(torch.full((2,), 0.5, dtype=torch.float64))
        maximum_degree = max(1, len(edges))
        degrees = torch.cat(
            (incoming_degree / maximum_degree, outgoing_degree / maximum_degree), dim=1
        )
        return edge_messages, torch.stack(aggregated), degrees

    def recurrent_step(self, state: torch.Tensor, features: Mapping[str, Any]) -> RecurrentStep:
        """Apply one differentiable update without consulting any exact authority."""

        expected_shape = (len(features["local_groups"]), 2)
        if tuple(state.shape) != expected_shape:
            raise ValueError(f"variable state shape must be {expected_shape}")
        group_messages = self._group_messages(state)
        dependency_messages, aggregated, degrees = self._dependency_messages(state, features)
        network_input = torch.cat((state, group_messages, aggregated, degrees), dim=1)
        delta = self.update_network(network_input)
        scale = torch.sigmoid(self.update_scale)
        proposal_logits = (
            torch.log(group_messages.clamp_min(1e-12))
            + 0.5 * torch.log(aggregated.clamp_min(1e-12))
            + scale * delta
        )
        proposal = torch.softmax(proposal_logits, dim=1)
        updated = 0.5 * state + 0.5 * proposal
        return RecurrentStep(updated, group_messages, dependency_messages, aggregated)


def trainable_parameter_count(model: nn.Module) -> int:
    """Count only parameters that the frozen optimizer can change."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def initial_variable_state(features: Mapping[str, Any], *, seed: int) -> torch.Tensor:
    """Create identity-bound soft states without target assignments or random globals."""

    rows: list[list[float]] = []
    graph_id = str(features["graph_id"])
    for group in features["local_groups"]:
        material = f"{seed}|{graph_id}|{group['group_id']}".encode()
        digest = hashlib.sha256(material).digest()
        left = int.from_bytes(digest[:8], "big") / float(2**64)
        right = int.from_bytes(digest[8:16], "big") / float(2**64)
        rows.append([left - 0.5, right - 0.5])
    return torch.softmax(torch.tensor(rows, dtype=torch.float64), dim=1)


def _rounded(values: torch.Tensor) -> list[float] | list[list[float]]:
    """Convert a tensor to stable JSON numbers while retaining useful telemetry."""

    value = values.detach().cpu().tolist()
    if value and isinstance(value[0], list):
        return [[round(float(item), 10) for item in row] for row in value]
    return [round(float(item), 10) for item in value]


def run_fixed_point(
    model: GroupAwareSoftFixedPoint,
    unit: Mapping[str, Any],
    *,
    seed: int,
    iteration_cap: int,
    convergence_tolerance: float,
) -> JsonDict:
    """Iterate until residual convergence or the frozen cap, with all receipts."""

    if iteration_cap <= 0:
        raise ValueError("iteration_cap must be positive")
    if convergence_tolerance < 0:
        raise ValueError("convergence_tolerance must be non-negative")
    features = unit["proposal_features"]
    state = initial_variable_state(features, seed=seed)
    receipts: list[JsonDict] = []
    stop_reason = "iteration_cap"
    residual = math.inf
    final_step: RecurrentStep | None = None
    model.eval()
    with torch.no_grad():
        for iteration in range(1, iteration_cap + 1):
            step = model.recurrent_step(state, features)
            residual = float(torch.max(torch.abs(step.variable_state - state)).item())
            finite = all(
                torch.isfinite(value).all().item()
                for value in (
                    step.variable_state,
                    step.group_messages,
                    step.dependency_messages,
                    step.aggregated_dependency_messages,
                )
            )
            for group_index, group in enumerate(features["local_groups"]):
                receipts.append(
                    {
                        "iteration": iteration,
                        "group_id": group["group_id"],
                        "variables": deepcopy(group["variables"]),
                        "variable_state": _rounded(state[group_index]),
                        "local_group_message": _rounded(step.group_messages[group_index]),
                        "dependency_message": _rounded(
                            step.aggregated_dependency_messages[group_index]
                        ),
                        "updated_state": _rounded(step.variable_state[group_index]),
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
    final_finite = all(
        torch.isfinite(value).all().item()
        for value in (
            state,
            final_step.group_messages,
            final_step.dependency_messages,
        )
    )
    dependency_receipts = [
        {
            "dependency_id": edge["dependency_id"],
            "source_group": edge["source_group"],
            "target_group": edge["target_group"],
            "relation_type": edge["relation_type"],
            "message": _rounded(final_step.dependency_messages[index]),
        }
        for index, edge in enumerate(features["dependency_edges"])
    ]
    return {
        "iterations": iteration,
        "state_residual": round(residual, 10),
        "stop_reason": stop_reason,
        "finite_values": bool(final_finite),
        "variable_state": _rounded(state),
        "variable_state_tensor": state,
        "dependency_messages": dependency_receipts,
        "group_message_receipts": receipts,
    }


def _dependency_violation(state: torch.Tensor, features: Mapping[str, Any]) -> torch.Tensor:
    """Return a soft structural loss derived only from named edge relations."""

    group_index = {
        str(group["group_id"]): index for index, group in enumerate(features["local_groups"])
    }
    losses: list[torch.Tensor] = []
    for edge in features["dependency_edges"]:
        source = state[group_index[str(edge["source_group"])]]
        target = state[group_index[str(edge["target_group"])]]
        if edge["relation_type"] == "implies_selected_one":
            losses.append(source[1] * target[0])
        elif edge["relation_type"] == "implies_selected_zero":
            losses.append(source[0] * target[1])
        else:
            raise ValueError(f"unknown dependency relation: {edge['relation_type']}")
    return torch.stack(losses).mean() if losses else torch.zeros((), dtype=torch.float64)


def fit_seed(
    train_units: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    hyperparameters: Mapping[str, Any],
) -> tuple[GroupAwareSoftFixedPoint, JsonDict]:
    """Fit one model on train topology structure without labels or checker feedback."""

    if not train_units or any(unit.get("split") != "train" for unit in train_units):
        raise ValueError("fit_seed requires nonempty train split units only")
    violations = audit_feature_contract(train_units)
    if violations:
        raise ValueError(f"oracle feature refusal: {violations[0]}")
    _configure_torch(seed)
    model = GroupAwareSoftFixedPoint(hidden_width=int(hyperparameters["hidden_width"]), seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(hyperparameters["learning_rate"]))
    loss_history: list[float] = []
    model.train()
    for _step_index in range(int(hyperparameters["training_steps"])):
        optimizer.zero_grad()
        unit_losses: list[torch.Tensor] = []
        for unit in train_units:
            features = unit["proposal_features"]
            state = initial_variable_state(features, seed=seed)
            residual_terms: list[torch.Tensor] = []
            for _ in range(int(hyperparameters["training_unroll_steps"])):
                step = model.recurrent_step(state, features)
                residual_terms.append(torch.mean((step.variable_state - state) ** 2))
                state = step.variable_state
            certainty = torch.mean(state[:, 0] * state[:, 1])
            dependency = _dependency_violation(state, features)
            unit_losses.append(
                torch.stack(residual_terms).mean() + 0.2 * certainty + 0.2 * dependency
            )
        loss = torch.stack(unit_losses).mean()
        loss.backward()
        optimizer.step()
        loss_history.append(round(float(loss.detach().item()), 10))
    receipt = {
        "seed": seed,
        "optimizer": hyperparameters["optimizer"],
        "learning_rate": hyperparameters["learning_rate"],
        "training_steps": hyperparameters["training_steps"],
        "training_unroll_steps": hyperparameters["training_unroll_steps"],
        "train_unit_ids": [str(unit["unit_id"]) for unit in train_units],
        "train_splits_seen": sorted({str(unit["split"]) for unit in train_units}),
        "train_topology_families": sorted({str(unit["topology_family"]) for unit in train_units}),
        "loss_history": loss_history,
        "final_loss": loss_history[-1],
    }
    return model, receipt


def decode_candidates(
    state: torch.Tensor,
    unit: Mapping[str, Any],
    *,
    seed: int,
    threshold: float,
    candidate_count: int,
) -> list[JsonDict]:
    """Decode bounded one-hot assignments without checking any candidate."""

    groups = unit["proposal_features"]["local_groups"]
    if candidate_count <= 0 or candidate_count > len(groups) + 1:
        raise ValueError("candidate_count must be between 1 and group_count + 1")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("decoding threshold must be in [0, 1]")
    probabilities = state.detach().cpu()
    base_states = [int(float(row[1]) >= threshold) for row in probabilities]
    uncertain = sorted(
        range(len(groups)),
        key=lambda index: (
            abs(float(probabilities[index, 1]) - threshold),
            sha256_json([seed, groups[index]["group_id"]]),
        ),
    )
    candidates: list[JsonDict] = []
    for candidate_index in range(candidate_count):
        states = list(base_states)
        changed_group = None
        if candidate_index:
            flip_index = uncertain[candidate_index - 1]
            states[flip_index] = 1 - states[flip_index]
            changed_group = groups[flip_index]["group_id"]
        assignment: JsonDict = {}
        for group, selected in zip(groups, states, strict=True):
            assignment[str(group["variables"][0])] = int(selected == 0)
            assignment[str(group["variables"][1])] = int(selected == 1)
        candidates.append(
            {
                "candidate_index": candidate_index,
                "decode_rule": "threshold_then_uncertainty_flip",
                "decoding_threshold": threshold,
                "changed_group": changed_group,
                "assignment": assignment,
                "candidate_hash": sha256_json(assignment),
            }
        )
    return candidates


def propose_unit(
    model: GroupAwareSoftFixedPoint,
    unit: Mapping[str, Any],
    *,
    seed: int,
    hyperparameters: Mapping[str, Any],
) -> JsonDict:
    """Emit one unit-seed row with recurrence and decode telemetry."""

    violations = audit_feature_contract([unit])
    if violations:
        raise ValueError(f"oracle feature refusal: {violations[0]}")
    started = time.perf_counter()
    fixed_point = run_fixed_point(
        model,
        unit,
        seed=seed,
        iteration_cap=int(hyperparameters["iteration_cap"]),
        convergence_tolerance=float(hyperparameters["convergence_tolerance"]),
    )
    candidates = decode_candidates(
        fixed_point.pop("variable_state_tensor"),
        unit,
        seed=seed,
        threshold=float(hyperparameters["decoding_threshold"]),
        candidate_count=int(hyperparameters["candidate_count"]),
    )
    return {
        "schema": ROW_SCHEMA,
        "row_id": f"{unit['unit_id']}|seed-{seed}",
        "unit_id": unit["unit_id"],
        "graph_id": unit["graph_id"],
        "graph_serialization": unit["graph_serialization"],
        "group_ids": deepcopy(unit["group_ids"]),
        "variable_ids": deepcopy(unit["variable_ids"]),
        "dependency_ids": deepcopy(unit["dependency_ids"]),
        "split": unit["split"],
        "topology_family": unit["topology_family"],
        "difficulty_stratum": unit["difficulty_stratum"],
        "random_seed": seed,
        "iterations": fixed_point["iterations"],
        "state_residual": fixed_point["state_residual"],
        "stop_reason": fixed_point["stop_reason"],
        "finite_values": fixed_point["finite_values"],
        "variable_state": fixed_point["variable_state"],
        "dependency_messages": fixed_point["dependency_messages"],
        "group_message_receipts": fixed_point["group_message_receipts"],
        "candidates": candidates,
        "candidate_hashes": [candidate["candidate_hash"] for candidate in candidates],
        "runtime_s": round(time.perf_counter() - started, 6),
    }


def _produce_seed_rows(
    units: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    hyperparameters: Mapping[str, Any],
) -> tuple[list[JsonDict], JsonDict, int]:
    """Fit one seed and propose for every non-train unit."""

    train_units = [unit for unit in units if unit["split"] == "train"]
    output_units = [unit for unit in units if unit["split"] in OUTPUT_SPLITS]
    model, training_receipt = fit_seed(train_units, seed=seed, hyperparameters=hyperparameters)
    rows = [
        propose_unit(model, unit, seed=seed, hyperparameters=hyperparameters)
        for unit in output_units
    ]
    return rows, training_receipt, trainable_parameter_count(model)


def replay_seed(
    source_artifact_path: Path,
    *,
    seed: int,
    expected_hash: str,
) -> JsonDict:
    """Re-fit one seed and return only ordered candidate hashes for replay."""

    preconditions = evaluate_preconditions(source_artifact_path, expected_hash=expected_hash)
    if not preconditions["all_passed"]:
        first = preconditions["first_failure"]
        raise ValueError(f"replay precondition failed: {first['check']}")
    units = project_units(load_json_object(source_artifact_path))
    rows, _receipt, _count = _produce_seed_rows(
        units, seed=seed, hyperparameters=FROZEN_HYPERPARAMETERS
    )
    candidate_hashes = [
        {"row_id": row["row_id"], "candidate_hashes": row["candidate_hashes"]} for row in rows
    ]
    return {
        "seed": seed,
        "candidate_hashes": candidate_hashes,
        "candidate_hashes_sha256": sha256_json(candidate_hashes),
        "worker_pid": os.getpid(),
    }


def _replay_worker() -> int:
    """Read a replay job on stdin and emit one JSON receipt on stdout."""

    payload = json.load(sys.stdin)
    receipt = replay_seed(
        Path(payload["source_artifact_path"]),
        seed=int(payload["seed"]),
        expected_hash=str(payload["expected_hash"]),
    )
    print(canonical_json(receipt))
    return 0


def run_fresh_replay(
    source_artifact_path: Path,
    *,
    seed: int,
    expected_hash: str,
    repo_root: Path,
) -> JsonDict:
    """Run one seed in a new interpreter so in-process caches cannot help it."""

    payload = canonical_json(
        {
            "source_artifact_path": str(source_artifact_path),
            "seed": seed,
            "expected_hash": expected_hash,
        }
    )
    environment = os.environ.copy()
    python_path = str(repo_root / "python")
    environment["PYTHONPATH"] = (
        python_path
        if not environment.get("PYTHONPATH")
        else f"{python_path}{os.pathsep}{environment['PYTHONPATH']}"
    )
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    environment["PYTHONHASHSEED"] = str(seed)
    process = subprocess.run(
        [sys.executable, "-m", __name__, "--replay-worker"],
        input=payload,
        text=True,
        capture_output=True,
        cwd=repo_root,
        env=environment,
        timeout=120,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(f"fresh replay failed: {process.stderr.strip()}")
    return json.loads(process.stdout)


def _convergence_by_split(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Aggregate stopping telemetry without introducing an accuracy metric."""

    result: JsonDict = {}
    for split in OUTPUT_SPLITS:
        split_rows = [row for row in rows if row["split"] == split]
        result[split] = {
            "row_count": len(split_rows),
            "converged_count": sum(row["stop_reason"] == "converged" for row in split_rows),
            "iteration_cap_count": sum(row["stop_reason"] == "iteration_cap" for row in split_rows),
            "non_finite_count": sum(not row["finite_values"] for row in split_rows),
            "mean_iterations": round(
                sum(int(row["iterations"]) for row in split_rows) / len(split_rows), 10
            )
            if split_rows
            else None,
            "mean_state_residual": round(
                sum(float(row["state_residual"]) for row in split_rows) / len(split_rows), 10
            )
            if split_rows
            else None,
        }
    return result


def _candidate_hash_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Copy the ordered candidate digest list used by replay and downstream consumers."""

    return [
        {"row_id": row["row_id"], "candidate_hashes": deepcopy(row["candidate_hashes"])}
        for row in rows
    ]


def _stable_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Remove measured per-row runtime before reproducibility hashing."""

    return [{key: value for key, value in row.items() if key != "runtime_s"} for row in rows]


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash all stable mechanism evidence while excluding wall-clock and process IDs."""

    replay = artifact.get("deterministic_replay_agreement", {})
    stable_replay = {
        key: replay.get(key)
        for key in (
            "attempted",
            "seed",
            "agreement",
            "fresh_process",
            "producer_hash",
            "replay_hash",
            "row_count",
            "mismatches",
        )
    }
    material = {
        "schema": artifact.get("schema"),
        "experiment_id": artifact.get("experiment_id"),
        "run_date": artifact.get("run_date"),
        "source_artifact_hash": artifact.get("source_artifact_hash"),
        "frozen_hyperparameters": artifact.get("frozen_hyperparameters"),
        "trainable_parameter_count": artifact.get("trainable_parameter_count"),
        "feature_allowlist": artifact.get("feature_allowlist"),
        "feature_denylist": artifact.get("feature_denylist"),
        "oracle_feature_violations": artifact.get("oracle_feature_violations"),
        "training_receipts": artifact.get("training_receipts"),
        "rows": _stable_rows(artifact.get("rows", [])),
        "convergence_by_split": artifact.get("convergence_by_split"),
        "finite_value_failures": artifact.get("finite_value_failures"),
        "deterministic_replay_agreement": stable_replay,
        "candidate_hashes": artifact.get("candidate_hashes"),
        "soft_fixed_point_proposer_ready": artifact.get("soft_fixed_point_proposer_ready"),
        "failed_checks": artifact.get("gate_check_summary", {}).get("failed_checks"),
        "verifier_is_oracle": artifact.get("verifier_is_oracle"),
        "verdict_class": artifact.get("verdict_class"),
    }
    return sha256_json(material)


def _blocked_artifact(
    *,
    run_date: str,
    duration_s: float,
    source_hash: str | None,
    gate_summary: Mapping[str, Any],
) -> JsonDict:
    """Return the full schema even when authority fails before training."""

    first = gate_summary.get("first_failure") or {
        "check": "all_preconditions",
        "observed": None,
    }
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "status": "complete_blocked_soft_fixed_point",
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "source_artifact_hash": source_hash,
        "frozen_hyperparameters": deepcopy(FROZEN_HYPERPARAMETERS),
        "trainable_parameter_count": expected_parameter_count(
            int(FROZEN_HYPERPARAMETERS["hidden_width"])
        ),
        "feature_allowlist": list(FEATURE_ALLOWLIST),
        "feature_denylist": list(FEATURE_DENYLIST),
        "oracle_feature_violations": deepcopy(
            list(gate_summary.get("oracle_feature_violations", []))
        ),
        "training_receipts": [],
        "rows": [],
        "convergence_by_split": {
            split: {
                "row_count": 0,
                "converged_count": 0,
                "iteration_cap_count": 0,
                "non_finite_count": 0,
                "mean_iterations": None,
                "mean_state_residual": None,
            }
            for split in OUTPUT_SPLITS
        },
        "finite_value_failures": [],
        "deterministic_replay_agreement": {
            "attempted": False,
            "seed": FROZEN_HYPERPARAMETERS["seeds"][0],
            "agreement": False,
            "fresh_process": False,
            "producer_hash": None,
            "replay_hash": None,
            "row_count": 0,
            "mismatches": [],
        },
        "candidate_hashes": [],
        "soft_fixed_point_proposer_ready": False,
        "gate_check_summary": deepcopy(dict(gate_summary)),
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": (
            f"complete_blocked_soft_fixed_point: {first['check']} observed {first.get('observed')}"
        ),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _validate_run_date(run_date: str) -> None:
    """Reject ambiguous dates before reading or writing experiment evidence."""

    if len(run_date) != 8 or not run_date.isdigit():
        raise ValueError("run date must use YYYYMMDD")


def build_artifact(
    *,
    run_date: str = RUN_DATE,
    source_artifact_path: Path | None = None,
    repo_root: Path = REPO_ROOT,
    duration_s: float | None = None,
) -> JsonDict:
    """Build the ready mechanism artifact, or a full block on failed authority."""

    _validate_run_date(run_date)
    started = time.monotonic()
    source_path = source_artifact_path or repo_root / SOURCE_ARTIFACT_RELATIVE_PATH
    preconditions = evaluate_preconditions(source_path)
    measured = round(time.monotonic() - started, 6) if duration_s is None else duration_s
    if not preconditions["all_passed"]:
        artifact = _blocked_artifact(
            run_date=run_date,
            duration_s=measured,
            source_hash=sha256_file(source_path),
            gate_summary=preconditions,
        )
        errors = validate_artifact(artifact)
        if errors:  # pragma: no cover - construction and validation share one contract.
            raise ValueError("; ".join(errors))
        return artifact

    source = load_json_object(source_path)
    units = project_units(source)
    all_rows: list[JsonDict] = []
    training_receipts: list[JsonDict] = []
    parameter_counts: list[int] = []
    for seed in FROZEN_HYPERPARAMETERS["seeds"]:
        rows, receipt, parameter_count = _produce_seed_rows(
            units,
            seed=int(seed),
            hyperparameters=FROZEN_HYPERPARAMETERS,
        )
        all_rows.extend(rows)
        training_receipts.append(receipt)
        parameter_counts.append(parameter_count)

    replay_seed_value = int(FROZEN_HYPERPARAMETERS["seeds"][0])
    producer_seed_rows = [row for row in all_rows if row["random_seed"] == replay_seed_value]
    producer_hashes = _candidate_hash_rows(producer_seed_rows)
    replay = run_fresh_replay(
        source_path,
        seed=replay_seed_value,
        expected_hash=EXPECTED_SOURCE_ARTIFACT_HASH,
        repo_root=repo_root,
    )
    replay_hashes = replay["candidate_hashes"]
    mismatches = [
        producer["row_id"]
        for producer, repeated in zip(producer_hashes, replay_hashes, strict=True)
        if producer != repeated
    ]
    replay_agreement = {
        "attempted": True,
        "seed": replay_seed_value,
        "agreement": not mismatches and producer_hashes == replay_hashes,
        "fresh_process": replay["worker_pid"] != os.getpid(),
        "producer_hash": sha256_json(producer_hashes),
        "replay_hash": replay["candidate_hashes_sha256"],
        "row_count": len(replay_hashes),
        "mismatches": mismatches,
        "producer_pid": os.getpid(),
        "worker_pid": replay["worker_pid"],
    }
    finite_failures = [str(row["row_id"]) for row in all_rows if not row["finite_values"]]
    candidate_hashes = _candidate_hash_rows(all_rows)
    expected_rows = len(FROZEN_HYPERPARAMETERS["seeds"]) * sum(
        unit["split"] in OUTPUT_SPLITS for unit in units
    )
    ready = bool(
        len(all_rows) == expected_rows
        and not finite_failures
        and replay_agreement["agreement"]
        and replay_agreement["fresh_process"]
        and len(set(parameter_counts)) == 1
    )
    measured = round(time.monotonic() - started, 6) if duration_s is None else duration_s
    artifact = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "status": "complete" if ready else "complete_blocked_soft_fixed_point",
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": measured,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "source_artifact_hash": sha256_file(source_path),
        "frozen_hyperparameters": deepcopy(FROZEN_HYPERPARAMETERS),
        "trainable_parameter_count": parameter_counts[0],
        "feature_allowlist": list(FEATURE_ALLOWLIST),
        "feature_denylist": list(FEATURE_DENYLIST),
        "oracle_feature_violations": [],
        "training_receipts": training_receipts,
        "rows": all_rows,
        "convergence_by_split": _convergence_by_split(all_rows),
        "finite_value_failures": finite_failures,
        "deterministic_replay_agreement": replay_agreement,
        "candidate_hashes": candidate_hashes,
        "soft_fixed_point_proposer_ready": ready,
        "gate_check_summary": preconditions,
        "verifier_is_oracle": False,
        "verdict_class": "positive" if ready else "blocked",
        "honest_verdict": (
            "complete: bounded group-aware soft fixed-point proposer is mechanism-ready; "
            "candidate correctness and superiority were not tested"
            if ready
            else "complete_blocked_soft_fixed_point: deterministic replay or finite-value gate failed"
        ),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:  # pragma: no cover - construction and validation share one contract.
        raise ValueError("; ".join(errors))
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return all schema and authority errors without altering measured evidence."""

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

    ready = artifact.get("soft_fixed_point_proposer_ready") is True
    if ready and artifact.get("status") == "complete_blocked_soft_fixed_point":
        errors.append("blocked artifact cannot be ready")
    if ready:
        if artifact.get("status") != "complete":
            errors.append("ready artifact status mismatch")
        if artifact.get("source_artifact_hash") != EXPECTED_SOURCE_ARTIFACT_HASH:
            errors.append("ready artifact source hash mismatch")
        if artifact.get("frozen_hyperparameters") != FROZEN_HYPERPARAMETERS:
            errors.append("ready artifact hyperparameters drifted")
        if artifact.get("trainable_parameter_count") != expected_parameter_count(
            int(FROZEN_HYPERPARAMETERS["hidden_width"])
        ):
            errors.append("ready artifact parameter count mismatch")
        if artifact.get("oracle_feature_violations") != []:
            errors.append("ready artifact contains oracle feature violations")
        if artifact.get("finite_value_failures") != []:
            errors.append("ready artifact contains finite-value failures")
        if artifact.get("gate_check_summary", {}).get("all_passed") is not True:
            errors.append("ready artifact has failed preconditions")
        if artifact.get("deterministic_replay_agreement", {}).get("agreement") is not True:
            errors.append("ready artifact lacks deterministic replay")
        expected_row_count = len(FROZEN_HYPERPARAMETERS["seeds"]) * 64
        if len(artifact.get("rows", [])) != expected_row_count:
            errors.append("ready artifact row count mismatch")
        if len(artifact.get("training_receipts", [])) != len(FROZEN_HYPERPARAMETERS["seeds"]):
            errors.append("ready artifact training receipt count mismatch")
        if artifact.get("candidate_hashes") != _candidate_hash_rows(artifact.get("rows", [])):
            errors.append("ready artifact candidate hash index mismatch")
        for row in artifact.get("rows", []):
            expected_receipts = int(row.get("iterations", 0)) * len(row.get("group_ids", []))
            if len(row.get("group_message_receipts", [])) != expected_receipts:
                errors.append(f"row group receipt count mismatch: {row.get('row_id')}")
                break
    else:
        if artifact.get("status") != "complete_blocked_soft_fixed_point":
            errors.append("blocked artifact status mismatch")
        if artifact.get("rows") != []:
            errors.append("blocked artifact must not contain rows")
        if artifact.get("candidate_hashes") != []:
            errors.append("blocked artifact must not contain candidate hashes")
        if artifact.get("verdict_class") != "blocked":
            errors.append("blocked artifact must use blocked verdict class")
    return errors


def write_outputs(
    *,
    run_date: str = RUN_DATE,
    source_artifact_path: Path | None = None,
    artifact_path: Path = RESULT_RELATIVE_PATH,
    repo_root: Path = REPO_ROOT,
    duration_s: float | None = None,
) -> JsonDict:
    """Write one validated artifact to an explicit or repository-relative path."""

    source_path = source_artifact_path or repo_root / SOURCE_ARTIFACT_RELATIVE_PATH
    output_path = artifact_path if artifact_path.is_absolute() else repo_root / artifact_path
    artifact = build_artifact(
        run_date=run_date,
        source_artifact_path=source_path,
        repo_root=repo_root,
        duration_s=duration_s,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the bounded experiment CLI and print its terminal verdict."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--source-artifact", type=Path, default=SOURCE_ARTIFACT_RELATIVE_PATH)
    parser.add_argument("--artifact-path", type=Path, default=RESULT_RELATIVE_PATH)
    parser.add_argument("--replay-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.replay_worker:
        return _replay_worker()
    source_path = (
        args.source_artifact
        if args.source_artifact.is_absolute()
        else REPO_ROOT / args.source_artifact
    )
    artifact = write_outputs(
        run_date=args.date,
        source_artifact_path=source_path,
        artifact_path=args.artifact_path,
        repo_root=REPO_ROOT,
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the repository entry point.
    raise SystemExit(main())
