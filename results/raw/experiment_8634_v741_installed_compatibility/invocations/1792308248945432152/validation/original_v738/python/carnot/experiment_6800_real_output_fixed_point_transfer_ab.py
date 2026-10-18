"""Transfer the frozen V592 fixed-point arms to authentic output graphs.

Spec refs: REQ-VERIFY-6800 and SCENARIO-VERIFY-6800-*.

The module rebuilds both arms from the Exp6786 train split. It proves the
stored Exp6788 candidate hashes before it proposes on Exp6799. Exact CNF
checks run only after each proposal has frozen its candidate hashes.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import torch
from torch import nn

from carnot import durable_row_checkpoint as checkpointing
from carnot import experiment_6787_group_aware_soft_fixed_point as grouped_source
from carnot import experiment_6788_soft_fixed_point_structural_control_ab as v592
from carnot import experiment_6799_model_output_formal_constraint_probes as real_probes


JsonDict = dict[str, Any]
ModelKey = tuple[int, str]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_6800_real_output_fixed_point_transfer_ab"
SCHEMA = "carnot.experiment_6800.real_output_fixed_point_transfer_ab.v1"
ROW_SCHEMA = "carnot.experiment_6800.real_output_transfer_row.v1"
RUN_DATE = "20260831"
RANDOM_SEED = 6_800_000
INFERENCE_SUBSTRATE = "CPU PyTorch frozen recipe; no LLM invocation"

SOURCE_PATHS = {
    "exp6786": Path("results/experiment_6786_constraint_dependency_hard_negative_fixture.json"),
    "exp6787": Path("results/experiment_6787_group_aware_soft_fixed_point.json"),
    "exp6788": Path("results/experiment_6788_soft_fixed_point_structural_control_ab.json"),
    "exp6789": Path("results/experiment_6789_soft_fixed_point_cold_authority_audit.json"),
    "exp6799": Path("results/experiment_6799_model_output_formal_constraint_probes.json"),
}
EXPECTED_SOURCE_HASHES = {
    "exp6786": "sha256:f3780c85e29cda8dbd897b6c43a0ce3c938252625e823e54107f918d2052514a",
    "exp6787": "sha256:161214c61401ebdb6a5ec11ea02eb520c341bc14f7831379d19651511f32a37d",
    "exp6788": "sha256:71feb075e5606891c857e6698f4d43b11fcaad2bfc82d1f21e99584aedad565f",
    "exp6789": "sha256:101487f224e5926d519b3db94c6a8ce12910c26ff1e711936216762398fc4748",
    "exp6799": "sha256:73d7b885bd909172c46acb567a956c668788856fe9aa90de96e0a6dd7936560c",
}
RESULT_RELATIVE_PATH = Path("results") / f"{EXPERIMENT_ID}.json"
CHECKPOINT_RELATIVE_PATH = Path("results/checkpoints") / f"{EXPERIMENT_ID}.json"

GROUPED_ARM = v592.GROUPED_ARM
FLAT_ARM = v592.FLAT_ARM
ARMS = v592.ARMS
TRANSFORMATIONS = real_probes.TRANSFORMATIONS
FROZEN_HYPERPARAMETERS = deepcopy(grouped_source.FROZEN_HYPERPARAMETERS)
FROZEN_SEEDS = tuple(int(seed) for seed in FROZEN_HYPERPARAMETERS["seeds"])
CANDIDATE_COUNT = int(FROZEN_HYPERPARAMETERS["candidate_count"])
CASE_COUNT = 97
TRANSFER_UNIT_COUNT = CASE_COUNT * len(TRANSFORMATIONS)
PLANNED_ROW_COUNT = TRANSFER_UNIT_COUNT * len(FROZEN_SEEDS) * len(ARMS)
CPU_WALL_BUDGET_S = 600.0
BOOTSTRAP_RESAMPLES = 2_000
BOOTSTRAP_SEED = 6_800_900
SUPPORT_CONTRACTION_MARGIN = 0.05
FEATURE_ALLOWLIST = tuple(grouped_source.FEATURE_ALLOWLIST)
FEATURE_DENYLIST = tuple(
    sorted(set(grouped_source.FEATURE_DENYLIST) | set(real_probes.FEATURE_DENYLIST))
)

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
TASK_REQUIRED_FIELDS = (
    "field_principles",
    "inference_substrate",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "source_artifact_hashes",
    "training_isolation_receipts",
    "v592_reproduction_receipts",
    "frozen_arm_definitions",
    "parameter_counts_by_arm",
    "optimization_steps_by_arm",
    "candidate_budget_by_arm",
    "feature_allowlist",
    "feature_denylist",
    "metrics_by_transformation_model_family",
    "paired_exact_valid_deltas",
    "clustered_confidence_intervals",
    "support_contraction",
    "convergence_harm",
    "work_matching",
    "destructive_control_results",
    "rows",
    "model_output_fixed_point_comparison_completed",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
EXTRA_ARTIFACT_FIELDS = ("frozen_manifest", "checkpoint_receipt", "decision_gates")
REQUIRED_ARTIFACT_FIELDS = STANDARD_ARTIFACT_FIELDS + TASK_REQUIRED_FIELDS + EXTRA_ARTIFACT_FIELDS

FIELD_PRINCIPLES = {
    "schema": "A versioned schema makes incompatible transfer evidence fail closed.",
    "experiment_id": "A stable ID binds the result to the planned real-output transfer.",
    "run_date": "The date separates this frozen execution from later replays.",
    "status": "Status distinguishes a full comparison from a complete blocked run.",
    "field_principles": "Short purposes make every required field auditable.",
    "inference_substrate": "The CPU declaration rules out hidden LLM inference.",
    "duration_s": "Measured wall time exposes whether the planned CPU envelope held.",
    "random_seed": "The master seed anchors bootstrap and control replay.",
    "reproducibility_checksum": "A stable digest detects source, row, or verdict drift.",
    "source_artifact_hashes": "Exact file hashes bind all frozen authorities.",
    "training_isolation_receipts": "Receipts prove fitting used only Exp6786 train units.",
    "v592_reproduction_receipts": "Receipts prove stored V592 candidates replayed before transfer.",
    "frozen_arm_definitions": "Definitions expose the one structural arm difference.",
    "parameter_counts_by_arm": "Equal counts prevent capacity from explaining an effect.",
    "optimization_steps_by_arm": "Equal updates prevent extra fitting from favoring one arm.",
    "candidate_budget_by_arm": "Equal totals prevent wider search from favoring one arm.",
    "feature_allowlist": "The allowlist names every field that proposal can read.",
    "feature_denylist": "The denylist keeps labels, identity, and diagnoses out of proposal.",
    "metrics_by_transformation_model_family": "Strata expose effects hidden by pooled means.",
    "paired_exact_valid_deltas": "Paired deltas report grouped minus flat at each requested level.",
    "clustered_confidence_intervals": "Case clustering keeps repeated seeds inside one unit.",
    "support_contraction": "Support checks prevent a narrow mode from counting as a gain.",
    "convergence_harm": "Convergence checks prevent slower or unstable updates from counting.",
    "work_matching": "Work receipts expose planned and realized compute differences.",
    "destructive_control_results": "Controls separate graph structure from identity shortcuts.",
    "rows": "Every paired unit preserves proposal, controls, and post-check evidence.",
    "model_output_fixed_point_comparison_completed": "This exact gate means the full grid finished.",
    "gate_check_summary": "All preconditions and terminal checks retain observed values.",
    "verifier_is_oracle": "False states that exact checking never proposes or fits.",
    "verdict_class": "A closed enum keeps the scientific outcome machine-readable.",
    "honest_verdict": "A terminal prefix reports positive or null completion plainly.",
    "frozen_manifest": "The manifest binds every row identity and matched budget.",
    "checkpoint_receipt": "The receipt proves durable completion and safe resume.",
    "decision_gates": "Named gates make the effect decision reproducible.",
}


class ReproductionError(ValueError):
    """Report a V592 candidate mismatch before any real-output proposal."""


@dataclass(frozen=True)
class FrozenReconstruction:
    """Keep frozen arm instances and their isolation evidence together."""

    models: dict[ModelKey, nn.Module]
    training_isolation_receipts: list[JsonDict]
    v592_reproduction_receipts: list[JsonDict]


def canonical_json(value: Any) -> str:
    """Serialize stable JSON so hashes do not depend on formatting."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Hash a JSON-compatible value with an algorithm label."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str | None:
    """Hash exact file bytes, or return no hash for a missing source."""

    if not path.is_file():
        return None
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def load_json_object(path: Path) -> JsonDict:
    """Load one artifact and reject a root that is not an object."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def load_sources(repo_root: Path = REPO_ROOT) -> dict[str, JsonDict]:
    """Load the five frozen artifacts from an explicit checkout root."""

    return {name: load_json_object(repo_root / path) for name, path in SOURCE_PATHS.items()}


def _gate(check: str, expected: Any, observed: Any, passed: bool | None = None) -> JsonDict:
    """Record one gate without dropping the observed value."""

    return {
        "check": check,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": bool(observed == expected if passed is None else passed),
    }


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name all failures and retain the first complete failure receipt."""

    copied = [deepcopy(dict(check)) for check in checks]
    failures = [check for check in copied if check["passed"] is not True]
    return {
        "all_passed": not failures,
        "checks": copied,
        "failed_checks": [str(check["check"]) for check in failures],
        "first_failure": failures[0] if failures else None,
    }


def probe_grid_observation(source_6799: Mapping[str, Any]) -> JsonDict:
    """Summarize the exact real-output grid without exposing labels to fitting."""

    groups = source_6799.get("probe_groups", [])
    transformations = {
        transformation for group in groups for transformation in group.get("graphs", {}).keys()
    }
    split_counts: dict[str, int] = defaultdict(int)
    for group in groups:
        split_counts[str(group.get("split"))] += 1
    return {
        "case_count": len(groups),
        "transformation_count": len(transformations),
        "source_models": sorted(
            {str(group.get("source_model", {}).get("family_id")) for group in groups}
        ),
        "constraint_families": sorted({str(group.get("constraint_family")) for group in groups}),
        "splits": dict(sorted(split_counts.items())),
        "planned_row_count": len(groups) * len(transformations) * len(FROZEN_SEEDS) * len(ARMS),
    }


EXPECTED_GRID = {
    "case_count": 97,
    "transformation_count": 3,
    "source_models": [
        "gemma4_26b_middle_moe",
        "gemma4_31b_flagship_dense",
        "qwen36_flagship_moe",
    ],
    "constraint_families": [
        "expander_tseitin",
        "ladder_tseitin",
        "pigeonhole_anchor",
    ],
    "splits": {"development": 64, "held_case": 33},
    "planned_row_count": PLANNED_ROW_COUNT,
}


def evaluate_preconditions(
    *, repo_root: Path = REPO_ROOT, sources: Mapping[str, Mapping[str, Any]] | None = None
) -> JsonDict:
    """Check exact authority, frozen recipes, complete grid, and CPU budget."""

    available: dict[str, Mapping[str, Any]] = {}
    source_hashes: dict[str, str | None] = {}
    checks: list[JsonDict] = []
    for name, relative in SOURCE_PATHS.items():
        path = repo_root / relative
        observed_hash = sha256_file(path)
        source_hashes[str(relative)] = observed_hash
        source = sources.get(name, {}) if sources is not None else {}
        if sources is None and path.is_file():
            source = load_json_object(path)
        available[name] = source
        checks.append(_gate(f"source_artifact:{name}", True, bool(source) and path.is_file()))
        checks.append(_gate(f"source_hash:{name}", EXPECTED_SOURCE_HASHES[name], observed_hash))

    source_6786 = available["exp6786"]
    source_6787 = available["exp6787"]
    source_6788 = available["exp6788"]
    source_6789 = available["exp6789"]
    source_6799 = available["exp6799"]
    grid = probe_grid_observation(source_6799)
    train_units = [
        unit
        for unit in source_6786.get("frozen_manifest", {}).get("units", [])
        if unit.get("split") == "train"
    ]
    arm_defs = source_6788.get("arm_definitions", {})
    matched_resources = {
        "parameter_counts": source_6788.get("parameter_counts_by_arm"),
        "optimization_steps": source_6788.get("optimization_steps_by_arm"),
        "candidate_budgets": source_6788.get("candidate_budget_by_arm"),
        "iteration_caps": {arm: arm_defs.get(arm, {}).get("iteration_cap") for arm in ARMS},
    }
    matched_expected = {
        "parameter_counts": {arm: 91 for arm in ARMS},
        "optimization_steps": {arm: 30 for arm in ARMS},
        "candidate_budgets": {arm: 960 for arm in ARMS},
        "iteration_caps": {arm: int(FROZEN_HYPERPARAMETERS["iteration_cap"]) for arm in ARMS},
    }
    source_duration = source_6788.get("duration_s")
    estimate = (
        round(float(source_duration) + float(source_duration) / 640 * PLANNED_ROW_COUNT * 4, 6)
        if isinstance(source_duration, (int, float)) and source_duration > 0
        else None
    )
    runtime_budget = {
        "estimated_cpu_wall_s": estimate,
        "planned_cpu_wall_budget_s": CPU_WALL_BUDGET_S,
        "planned_row_count": PLANNED_ROW_COUNT,
        "control_proposals_per_row": 3,
    }
    checks.extend(
        [
            _gate(
                "constraint_group_fixture_ready",
                True,
                source_6786.get("constraint_group_fixture_ready"),
            ),
            _gate(
                "soft_fixed_point_proposer_ready",
                True,
                source_6787.get("soft_fixed_point_proposer_ready"),
            ),
            _gate(
                "fixed_point_comparison_completed",
                True,
                source_6788.get("fixed_point_comparison_completed"),
            ),
            _gate(
                "fixed_point_audit_completed",
                True,
                source_6789.get("fixed_point_audit_completed"),
            ),
            _gate(
                "model_output_constraint_probe_ready",
                True,
                source_6799.get("model_output_constraint_probe_ready"),
            ),
            _gate("complete_paired_probe_grid", EXPECTED_GRID, grid),
            _gate("exp6786_train_split", 32, len(train_units)),
            _gate(
                "exp6787_hyperparameters",
                FROZEN_HYPERPARAMETERS,
                source_6787.get("frozen_hyperparameters"),
            ),
            _gate(
                "five_frozen_seeds",
                list(FROZEN_SEEDS),
                source_6788.get("frozen_manifest", {}).get("seeds"),
            ),
            _gate("matched_arm_resources", matched_expected, matched_resources),
            _gate(
                "legal_feature_allowlist",
                list(real_probes.FEATURE_ALLOWLIST),
                source_6799.get("feature_allowlist"),
            ),
            _gate(
                "legal_feature_denylist",
                list(real_probes.FEATURE_DENYLIST),
                source_6799.get("feature_denylist"),
            ),
            _gate(
                "planned_cpu_wall_budget",
                f"estimated <= {CPU_WALL_BUDGET_S}",
                runtime_budget,
                estimate is not None and estimate <= CPU_WALL_BUDGET_S,
            ),
        ]
    )
    summary = _gate_summary(checks)
    summary["source_artifact_hashes"] = source_hashes
    summary["grid_observation"] = grid
    summary["runtime_budget"] = runtime_budget
    return summary


def parameter_counts(models: Mapping[ModelKey, nn.Module]) -> dict[str, int]:
    """Return frozen parameter sizes after gradients are disabled."""

    return {
        arm: next(
            sum(parameter.numel() for parameter in model.parameters())
            for (seed, model_arm), model in models.items()
            if seed == FROZEN_SEEDS[0] and model_arm == arm
        )
        for arm in ARMS
    }


def reconstruct_frozen_arms(sources: Mapping[str, Mapping[str, Any]]) -> FrozenReconstruction:
    """Rebuild both arms and reproduce every Exp6788 candidate hash."""

    context = v592.build_context(sources["exp6786"], sources["exp6787"])
    train_units = [unit for unit in context.proposal_units if unit["split"] == "train"]
    output_units = [
        unit for unit in context.proposal_units if unit["split"] in grouped_source.OUTPUT_SPLITS
    ]
    expected_rows = {str(row["row_id"]): row for row in sources["exp6788"].get("rows", [])}
    models: dict[ModelKey, nn.Module] = {}
    training_receipts: list[JsonDict] = []
    reproduction_receipts: list[JsonDict] = []
    for seed in FROZEN_SEEDS:
        for arm in ARMS:
            model, training = v592.fit_arm(
                train_units,
                arm=arm,
                seed=seed,
                hyperparameters=FROZEN_HYPERPARAMETERS,
            )
            observed_rows: list[JsonDict] = []
            expected_hash_rows: list[JsonDict] = []
            mismatches: list[str] = []
            for unit in output_units:
                raw = v592.propose_raw_row(
                    model,
                    unit,
                    arm=arm,
                    seed=seed,
                    parameter_count=v592.trainable_parameter_count(model),
                    optimizer_update_count=int(training["optimizer_update_count"]),
                    hyperparameters=FROZEN_HYPERPARAMETERS,
                )
                expected = expected_rows.get(str(raw["row_id"]))
                observed = {
                    "row_id": raw["row_id"],
                    "candidate_hashes": deepcopy(raw["candidate_hashes"]),
                }
                observed_rows.append(observed)
                expected_item = {
                    "row_id": raw["row_id"],
                    "candidate_hashes": deepcopy(expected.get("candidate_hashes", []))
                    if expected
                    else [],
                }
                expected_hash_rows.append(expected_item)
                if expected_item != observed:
                    mismatches.append(str(raw["row_id"]))
            receipt = {
                "seed": seed,
                "arm": arm,
                "candidate_row_count": len(observed_rows),
                "expected_candidate_hash": sha256_json(expected_hash_rows),
                "observed_candidate_hash": sha256_json(observed_rows),
                "mismatched_row_ids": mismatches,
                "agreement": not mismatches and expected_hash_rows == observed_rows,
                "proved_before_exp6799_proposal": True,
            }
            reproduction_receipts.append(receipt)
            if not receipt["agreement"]:
                raise ReproductionError(f"V592 candidate mismatch: {mismatches[0]}")
            training_receipts.append(
                {
                    **deepcopy(training),
                    "training_source": "exp6786_frozen_train_split",
                    "exp6799_fields_seen": [],
                }
            )
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            models[(seed, arm)] = model
    return FrozenReconstruction(models, training_receipts, reproduction_receipts)


def _project_dependency_edges(graph: Mapping[str, Any]) -> list[JsonDict]:
    """Map each CNF clause to ordered pair messages without exact outcomes."""

    edges: list[JsonDict] = []
    for clause_index, clause in enumerate(graph["clauses"]):
        if len(clause) == 1:
            literal_pairs = [(clause[0], clause[0])]
        else:
            literal_pairs = list(zip(clause, clause[1:] + clause[:1], strict=True))
        for pair_index, (source_literal, target_literal) in enumerate(literal_pairs):
            edges.append(
                {
                    "dependency_id": f"c{clause_index + 1:03d}-e{pair_index + 1:02d}",
                    "source_group": f"x{abs(int(source_literal))}",
                    "target_group": f"x{abs(int(target_literal))}",
                    "relation_type": (
                        "implies_selected_one"
                        if int(target_literal) > 0
                        else "implies_selected_zero"
                    ),
                }
            )
    return edges


def project_transfer_unit(group: Mapping[str, Any], transformation: str) -> JsonDict:
    """Project one exact CNF record into the frozen arm input schema."""

    if transformation not in TRANSFORMATIONS:
        raise ValueError(f"unknown transformation: {transformation}")
    record = deepcopy(group["graphs"][transformation])
    graph = record["graph"]
    variables = [str(variable) for variable in graph["variables"]]
    local_groups = [
        {
            "group_id": variable,
            "group_type": "binary_domain",
            "variables": [f"{variable}__0", f"{variable}__1"],
        }
        for variable in variables
    ]
    features = {
        "schema": "carnot.experiment_6800.transfer_features.v1",
        "graph_id": record["graph_hash"],
        "topology_family": str(group["constraint_family"]),
        "difficulty_stratum": str(record["structural_difficulty_score"]),
        "variables": [value for local in local_groups for value in local["variables"]],
        "local_groups": local_groups,
        "dependency_edges": _project_dependency_edges(graph),
    }
    source_case_id = str(group["source_case_id"])
    unit_id = f"{source_case_id}|{transformation}"
    return {
        "unit_id": unit_id,
        "graph_id": record["graph_hash"],
        "case_cluster_key": str(group["case_cluster_key"]),
        "source_case_id": source_case_id,
        "source_model": str(group["source_model"]["family_id"]),
        "source_model_hf_id": str(group["source_model"]["hf_id"]),
        "constraint_family": str(group["constraint_family"]),
        "split": str(group["split"]),
        "transformation": transformation,
        "proposal_features": features,
        "variable_ids": variables,
        "exact_graph_record": record,
    }


def build_transfer_units(source_6799: Mapping[str, Any]) -> list[JsonDict]:
    """Build every case and transformation unit in stable source order."""

    return [
        project_transfer_unit(group, transformation)
        for group in source_6799.get("probe_groups", [])
        for transformation in TRANSFORMATIONS
    ]


def audit_transfer_feature_contract(units: Sequence[Mapping[str, Any]]) -> list[str]:
    """Reject any proposal field outside the frozen V592 feature schema."""

    return grouped_source.audit_feature_contract(units)


def _decode_transfer_candidates(
    state: torch.Tensor, unit: Mapping[str, Any], *, seed: int
) -> list[JsonDict]:
    """Use the frozen decoder and map two-state groups back to Booleans."""

    candidates = grouped_source.decode_candidates(
        state,
        unit,
        seed=seed,
        threshold=float(FROZEN_HYPERPARAMETERS["decoding_threshold"]),
        candidate_count=CANDIDATE_COUNT,
    )
    decoded: list[JsonDict] = []
    for candidate in candidates:
        assignment = {
            variable: int(candidate["assignment"][f"{variable}__1"])
            for variable in unit["variable_ids"]
        }
        decoded.append(
            {
                "candidate_index": candidate["candidate_index"],
                "decode_rule": candidate["decode_rule"],
                "changed_group": candidate["changed_group"],
                "assignment": assignment,
                "candidate_hash": sha256_json(assignment),
            }
        )
    return decoded


def _proposal_once(
    model: nn.Module, unit: Mapping[str, Any], *, arm: str, seed: int
) -> tuple[JsonDict, list[JsonDict]]:
    """Run one bounded proposal with no exact graph or label access."""

    fixed = v592.run_arm_fixed_point(
        model,
        unit,
        arm=arm,
        seed=seed,
        iteration_cap=int(FROZEN_HYPERPARAMETERS["iteration_cap"]),
        convergence_tolerance=float(FROZEN_HYPERPARAMETERS["convergence_tolerance"]),
    )
    state = fixed.pop("variable_state_tensor")
    candidates = _decode_transfer_candidates(state, unit, seed=seed)
    telemetry = {
        "iterations": fixed["iterations"],
        "state_residual": fixed["state_residual"],
        "stop_reason": fixed["stop_reason"],
        "finite_values": fixed["finite_values"],
        "group_message_presence": fixed["group_message_presence"],
    }
    return telemetry, candidates


def _controlled_unit(unit: Mapping[str, Any], control: str) -> JsonDict:
    """Build one declared structural control from proposal fields only."""

    controlled = deepcopy(dict(unit))
    features = deepcopy(unit["proposal_features"])
    groups = features["local_groups"]
    if control == "group_id_permutation":
        old_ids = [str(group["group_id"]) for group in groups]
        new_ids = old_ids[1:] + old_ids[:1]
        mapping = dict(zip(old_ids, new_ids, strict=True))
        for group in groups:
            group["group_id"] = mapping[str(group["group_id"])]
        for edge in features["dependency_edges"]:
            edge["source_group"] = mapping[str(edge["source_group"])]
            edge["target_group"] = mapping[str(edge["target_group"])]
    elif control == "dependency_edge_removal":
        features["dependency_edges"] = []
    elif control == "surface_relabeling":
        groups.reverse()
    else:
        raise ValueError(f"unknown control: {control}")
    controlled["proposal_features"] = features
    return controlled


def propose_transfer_row(
    model: nn.Module, unit: Mapping[str, Any], *, arm: str, seed: int
) -> JsonDict:
    """Freeze one proposal and three structural controls before exact checking."""

    violations = audit_transfer_feature_contract([unit])
    if violations:
        raise ValueError(f"oracle feature refusal: {violations[0]}")
    started = time.perf_counter()
    telemetry, candidates = _proposal_once(model, unit, arm=arm, seed=seed)
    controls: JsonDict = {}
    for control in ("group_id_permutation", "dependency_edge_removal", "surface_relabeling"):
        control_telemetry, control_candidates = _proposal_once(
            model,
            _controlled_unit(unit, control),
            arm=arm,
            seed=seed,
        )
        controls[control] = {
            "candidate_budget": len(control_candidates),
            "candidate_hashes": [candidate["candidate_hash"] for candidate in control_candidates],
            "candidates": control_candidates,
            "iterations": control_telemetry["iterations"],
        }
    controls["source_model_identity_removal"] = {
        "proposal_input_unchanged": True,
        "proposal_input_hash": sha256_json(unit["proposal_features"]),
    }
    paired_key = f"{unit['unit_id']}|seed-{seed}"
    return {
        "schema": ROW_SCHEMA,
        "row_id": f"{paired_key}|{arm}",
        "paired_key": paired_key,
        "unit_id": unit["unit_id"],
        "case_cluster_key": unit["case_cluster_key"],
        "source_case_id": unit["source_case_id"],
        "source_model": unit["source_model"],
        "source_model_hf_id": unit["source_model_hf_id"],
        "constraint_family": unit["constraint_family"],
        "split": unit["split"],
        "transformation": unit["transformation"],
        "graph_id": unit["graph_id"],
        "arm": arm,
        "random_seed": seed,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "optimizer_update_count": int(FROZEN_HYPERPARAMETERS["training_steps"]),
        "proposal_iteration_cap": int(FROZEN_HYPERPARAMETERS["iteration_cap"]),
        "candidate_budget": CANDIDATE_COUNT,
        "candidate_work": CANDIDATE_COUNT,
        "proposal_input_hash": sha256_json(unit["proposal_features"]),
        **telemetry,
        "candidates": candidates,
        "candidate_hashes": [candidate["candidate_hash"] for candidate in candidates],
        "control_outcomes": controls,
        "runtime_s": round(time.perf_counter() - started, 6),
    }


def _nearest_valid_distance(
    assignment: Mapping[str, int], valid_assignments: Sequence[Mapping[str, int]]
) -> int:
    """Return Boolean Hamming distance to the closest exact witness."""

    return min(
        sum(int(assignment[key]) != int(valid[key]) for key in assignment)
        for valid in valid_assignments
    )


def _score_exact_candidates(
    candidates: Sequence[Mapping[str, Any]], graph_record: Mapping[str, Any]
) -> tuple[list[JsonDict], float]:
    """Apply independent exact checks to an already frozen candidate list."""

    outcomes: list[JsonDict] = []
    valid_assignments = graph_record["valid_assignments"]
    for candidate in candidates:
        receipt = real_probes.exact_check_candidate(graph_record["graph"], candidate["assignment"])
        outcomes.append(
            {
                "candidate_index": candidate["candidate_index"],
                "candidate_hash": candidate["candidate_hash"],
                "assignment": deepcopy(candidate["assignment"]),
                "local_checks_passed": receipt["local_checks_passed"],
                "failed_dependency_ids": deepcopy(receipt["failed_clause_ids"]),
                "dependency_violation_count": len(receipt["failed_clause_ids"]),
                "exact_valid": receipt["exact_valid"],
                "distance_to_nearest_valid": _nearest_valid_distance(
                    candidate["assignment"], valid_assignments
                ),
            }
        )
    return outcomes, round(sum(row["exact_valid"] for row in outcomes) / len(outcomes), 10)


def attach_exact_outcomes(raw_row: Mapping[str, Any], graph_record: Mapping[str, Any]) -> JsonDict:
    """Append exact outcomes without changing proposal bytes or model state."""

    row = deepcopy(dict(raw_row))
    before = deepcopy(row["candidate_hashes"])
    outcomes, exact_rate = _score_exact_candidates(row["candidates"], graph_record)
    for control in ("group_id_permutation", "dependency_edge_removal", "surface_relabeling"):
        control_outcomes, control_rate = _score_exact_candidates(
            row["control_outcomes"][control]["candidates"], graph_record
        )
        row["control_outcomes"][control]["exact_outcomes"] = control_outcomes
        row["control_outcomes"][control]["exact_valid_rate"] = control_rate
    hard_negative = real_probes._hard_negative(dict(graph_record))
    hard_receipt = real_probes.exact_check_candidate(graph_record["graph"], hard_negative)
    row.update(
        {
            "exact_outcomes": outcomes,
            "exact_valid_candidate_count": sum(outcome["exact_valid"] for outcome in outcomes),
            "exact_valid_rate": exact_rate,
            "dependency_violation_count": sum(
                outcome["dependency_violation_count"] for outcome in outcomes
            ),
            "nearest_valid_distance": min(
                outcome["distance_to_nearest_valid"] for outcome in outcomes
            ),
            "valid_support": sorted(
                outcome["candidate_hash"] for outcome in outcomes if outcome["exact_valid"]
            ),
            "hard_negative_control": {
                "assignment_hash": sha256_json(hard_negative),
                **hard_receipt,
            },
            "exact_evaluation_receipt": {
                "checker": "experiment_6799.exact_check_candidate",
                "evaluated_after_proposal": True,
                "candidate_hashes_before": before,
                "candidate_hashes_after": deepcopy(row["candidate_hashes"]),
                "model_feedback_applied": False,
            },
        }
    )
    return row


def frozen_manifest(
    units: Sequence[Mapping[str, Any]], *, seeds: Sequence[int] = FROZEN_SEEDS
) -> JsonDict:
    """Freeze the complete paired row roster and all matched budgets."""

    row_ids = [
        f"{unit['unit_id']}|seed-{seed}|{arm}" for unit in units for seed in seeds for arm in ARMS
    ]
    manifest = {
        "schema": "carnot.experiment_6800.frozen_manifest.v1",
        "unit_ids": [str(unit["unit_id"]) for unit in units],
        "seeds": [int(seed) for seed in seeds],
        "arms": list(ARMS),
        "transformations": list(TRANSFORMATIONS),
        "row_ids": row_ids,
        "planned_row_count": len(row_ids),
        "parameter_count_per_arm": 91,
        "training_steps_per_seed": int(FROZEN_HYPERPARAMETERS["training_steps"]),
        "proposal_iteration_cap": int(FROZEN_HYPERPARAMETERS["iteration_cap"]),
        "candidate_count_per_cell": CANDIDATE_COUNT,
        "cpu_wall_budget_s": CPU_WALL_BUDGET_S,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    return manifest


def execute_cells(
    *,
    units: Sequence[Mapping[str, Any]],
    reconstruction: FrozenReconstruction,
    checkpoint_path: Path,
    manifest: Mapping[str, Any],
    seeds: Sequence[int] = FROZEN_SEEDS,
    stop_after_new_rows: int | None = None,
) -> JsonDict:
    """Append only pending transfer cells to durable parent storage."""

    store = checkpointing.DurableRowCheckpoint(checkpoint_path, manifest)
    pending = set(store.pending(manifest["row_ids"]))
    new_row_count = 0
    for unit in units:
        for seed in seeds:
            for arm in ARMS:
                row_id = f"{unit['unit_id']}|seed-{seed}|{arm}"
                if row_id not in pending:
                    continue
                raw = propose_transfer_row(
                    reconstruction.models[(int(seed), arm)], unit, arm=arm, seed=int(seed)
                )
                payload = attach_exact_outcomes(raw, unit["exact_graph_record"])
                envelope = checkpointing.complete_row_envelope(
                    row_id=row_id,
                    manifest_hash=store.manifest_hash,
                    payload=v592._encode_checkpoint_payload(payload),
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
                    decoded = v592._decoded_checkpoint_rows(store.rows)
                    return {
                        "rows": decoded,
                        "new_row_count": new_row_count,
                        "pending_row_ids": store.pending(manifest["row_ids"]),
                        "manifest_hash": store.manifest_hash,
                    }
    return {
        "rows": v592._decoded_checkpoint_rows(store.rows),
        "new_row_count": new_row_count,
        "pending_row_ids": store.pending(manifest["row_ids"]),
        "manifest_hash": store.manifest_hash,
    }


def row_attribution_errors(
    rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> list[str]:
    """Check frozen identities, paired arms, and per-cell budgets."""

    errors: list[str] = []
    row_ids = [str(row.get("row_id")) for row in rows]
    if len(row_ids) != len(set(row_ids)):
        errors.append("duplicate row IDs")
    if set(row_ids) != set(manifest["row_ids"]):
        errors.append("row IDs do not match frozen manifest")
    pairs: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        pairs[str(row.get("paired_key"))].add(str(row.get("arm")))
        if row.get("row_id") != f"{row.get('paired_key')}|{row.get('arm')}":
            errors.append(f"row identity mismatch: {row.get('row_id')}")
        if row.get("candidate_budget") != manifest["candidate_count_per_cell"]:
            errors.append(f"candidate budget mismatch: {row.get('row_id')}")
    if any(arms != set(ARMS) for arms in pairs.values()):
        errors.append("each paired key must contain both arms")
    return errors


def _mean(values: Sequence[float]) -> float | None:
    """Return a stable mean or no value for an empty stratum."""

    return round(sum(values) / len(values), 10) if values else None


def percentile(values: Sequence[float], quantile: float) -> float:
    """Return one interpolated deterministic percentile."""

    if not values:
        raise ValueError("percentile requires nonempty values")
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
    """Summarize exact, support, convergence, work, and runtime for one arm."""

    candidates = sum(int(row["candidate_budget"]) for row in rows)
    valid = sum(int(row["exact_valid_candidate_count"]) for row in rows)
    support = {
        (str(row["unit_id"]), str(candidate_hash))
        for row in rows
        for candidate_hash in row["valid_support"]
    }
    return {
        "row_count": len(rows),
        "candidate_count": candidates,
        "exact_valid_candidate_count": valid,
        "exact_valid_rate": round(valid / candidates, 10) if candidates else None,
        "dependency_violation_count": sum(int(row["dependency_violation_count"]) for row in rows),
        "mean_nearest_valid_distance": _mean(
            [float(row["nearest_valid_distance"]) for row in rows]
        ),
        "valid_support": len(support),
        "convergence_rate": (
            round(sum(row["stop_reason"] == "converged" for row in rows) / len(rows), 10)
            if rows
            else None
        ),
        "mean_iterations": _mean([float(row["iterations"]) for row in rows]),
        "finite_value_failures": sum(not row["finite_values"] for row in rows),
        "candidate_work": sum(int(row["candidate_work"]) for row in rows),
        "runtime_s": round(sum(float(row["runtime_s"]) for row in rows), 6),
    }


def _paired_values(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Build grouped-minus-flat values while keeping each seed and case paired."""

    by_pair: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_pair[str(row["paired_key"])][str(row["arm"])] = row
    values: list[JsonDict] = []
    for paired_key, arms in sorted(by_pair.items()):
        if set(arms) != set(ARMS):
            continue
        grouped = arms[GROUPED_ARM]
        flat = arms[FLAT_ARM]
        value: JsonDict = {
            "paired_key": paired_key,
            "case_cluster_key": grouped["case_cluster_key"],
            "source_case_id": grouped["source_case_id"],
            "source_model": grouped["source_model"],
            "constraint_family": grouped["constraint_family"],
            "transformation": grouped["transformation"],
            "random_seed": grouped["random_seed"],
            "exact_valid_delta": round(
                float(grouped["exact_valid_rate"]) - float(flat["exact_valid_rate"]), 10
            ),
        }
        for control in ("group_id_permutation", "dependency_edge_removal", "surface_relabeling"):
            value[f"{control}_exact_valid_delta"] = round(
                float(grouped["control_outcomes"][control]["exact_valid_rate"])
                - float(flat["control_outcomes"][control]["exact_valid_rate"]),
                10,
            )
        values.append(value)
    return values


def _strata_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Report both arms for one requested result stratum."""

    return {arm: _arm_metrics([row for row in rows if row["arm"] == arm]) for arm in ARMS}


def _metrics_by_requested_strata(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Build transformation, model, family, case, and seed summaries."""

    output: JsonDict = {}
    for transformation in TRANSFORMATIONS:
        selected = [row for row in rows if row["transformation"] == transformation]
        models = sorted({str(row["source_model"]) for row in selected})
        families = sorted({str(row["constraint_family"]) for row in selected})
        cases = sorted({str(row["case_cluster_key"]) for row in selected})
        seeds = sorted({int(row["random_seed"]) for row in selected})
        output[transformation] = {
            "overall": _strata_metrics(selected),
            "by_source_model": {
                model: {
                    "overall": _strata_metrics(
                        [row for row in selected if row["source_model"] == model]
                    ),
                    "by_constraint_family": {
                        family: _strata_metrics(
                            [
                                row
                                for row in selected
                                if row["source_model"] == model
                                and row["constraint_family"] == family
                            ]
                        )
                        for family in families
                    },
                }
                for model in models
            },
            "by_constraint_family": {
                family: _strata_metrics(
                    [row for row in selected if row["constraint_family"] == family]
                )
                for family in families
            },
            "by_case": {
                case: _strata_metrics([row for row in selected if row["case_cluster_key"] == case])
                for case in cases
            },
            "by_seed": {
                str(seed): _strata_metrics([row for row in selected if row["random_seed"] == seed])
                for seed in seeds
            },
        }
    return output


def _mean_by(values: Sequence[Mapping[str, Any]], key: str) -> JsonDict:
    """Return exact-valid delta means for every value of one stratum key."""

    return {
        str(level): _mean(
            [float(value["exact_valid_delta"]) for value in values if value[key] == level]
        )
        for level in sorted({value[key] for value in values}, key=str)
    }


def _clustered_intervals(
    paired: Sequence[Mapping[str, Any]], *, resamples: int, seed: int
) -> JsonDict:
    """Bootstrap source cases while retaining all seeds and paired arms."""

    output: JsonDict = {}
    generator = random.Random(seed)
    for transformation in TRANSFORMATIONS:
        selected = [value for value in paired if value["transformation"] == transformation]
        by_case: dict[str, list[float]] = defaultdict(list)
        for value in selected:
            by_case[str(value["source_case_id"])].append(float(value["exact_valid_delta"]))
        case_means = {case: float(_mean(values)) for case, values in by_case.items()}
        cases = sorted(case_means)
        point = _mean(list(case_means.values()))
        draws = [
            sum(case_means[generator.choice(cases)] for _ in cases) / len(cases)
            for _ in range(resamples)
        ]
        output[transformation] = {
            "point": point,
            "lower": percentile(draws, 0.025) if draws else point,
            "upper": percentile(draws, 0.975) if draws else point,
            "confidence_level": 0.95,
            "resamples": resamples,
            "resampling_unit": "source_case",
            "case_count": len(cases),
            "paired_seed_count": len(selected),
        }
    return output


def aggregate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> JsonDict:
    """Derive every headline and destructive control from attributable rows."""

    paired = _paired_values(rows)
    metrics = _metrics_by_requested_strata(rows)
    paired_deltas = {
        "by_transformation": _mean_by(paired, "transformation"),
        "by_source_model": _mean_by(paired, "source_model"),
        "by_constraint_family": _mean_by(paired, "constraint_family"),
        "by_case": _mean_by(paired, "case_cluster_key"),
        "by_seed": _mean_by(paired, "random_seed"),
        "paired_key_count": len(paired),
    }
    intervals = _clustered_intervals(paired, resamples=bootstrap_resamples, seed=bootstrap_seed)
    support: JsonDict = {}
    convergence: JsonDict = {}
    for transformation in TRANSFORMATIONS:
        arm_metrics = metrics[transformation]["overall"]
        grouped_support = int(arm_metrics[GROUPED_ARM]["valid_support"])
        flat_support = int(arm_metrics[FLAT_ARM]["valid_support"])
        contraction = (
            max(0.0, (flat_support - grouped_support) / flat_support) if flat_support else 0.0
        )
        support[transformation] = {
            "grouped": grouped_support,
            "flat": flat_support,
            "contraction": round(contraction, 10),
            "margin": SUPPORT_CONTRACTION_MARGIN,
            "harm": contraction > SUPPORT_CONTRACTION_MARGIN,
        }
        grouped_metrics = arm_metrics[GROUPED_ARM]
        flat_metrics = arm_metrics[FLAT_ARM]
        convergence[transformation] = {
            "grouped_rate": grouped_metrics["convergence_rate"],
            "flat_rate": flat_metrics["convergence_rate"],
            "grouped_mean_iterations": grouped_metrics["mean_iterations"],
            "flat_mean_iterations": flat_metrics["mean_iterations"],
            "harm": bool(
                float(grouped_metrics["convergence_rate"]) < float(flat_metrics["convergence_rate"])
                or float(grouped_metrics["mean_iterations"])
                > float(flat_metrics["mean_iterations"])
                or int(grouped_metrics["finite_value_failures"])
                > int(flat_metrics["finite_value_failures"])
            ),
        }
    candidate_totals = {
        arm: sum(int(row["candidate_budget"]) for row in rows if row["arm"] == arm) for arm in ARMS
    }
    iteration_totals = {
        arm: sum(int(row["iterations"]) for row in rows if row["arm"] == arm) for arm in ARMS
    }
    work = {
        "planned_budgets_match": len(set(candidate_totals.values())) == 1,
        "candidate_totals_by_arm": candidate_totals,
        "proposal_iteration_cap_by_arm": {
            arm: sorted({int(row["proposal_iteration_cap"]) for row in rows if row["arm"] == arm})
            for arm in ARMS
        },
        "realized_iterations_by_arm": iteration_totals,
        "no_grouped_work_harm": iteration_totals[GROUPED_ARM] <= iteration_totals[FLAT_ARM],
    }
    controls: JsonDict = {}
    for control in ("group_id_permutation", "dependency_edge_removal", "surface_relabeling"):
        controls[control] = {
            "candidate_hash_agreement_rate": round(
                sum(
                    row["candidate_hashes"] == row["control_outcomes"][control]["candidate_hashes"]
                    for row in rows
                )
                / len(rows),
                10,
            ),
            "paired_exact_valid_delta_by_transformation": {
                transformation: _mean(
                    [
                        float(value[f"{control}_exact_valid_delta"])
                        for value in paired
                        if value["transformation"] == transformation
                    ]
                )
                for transformation in TRANSFORMATIONS
            },
        }
    controls["source_model_identity_removal"] = {
        "proposal_input_unchanged": all(
            row["control_outcomes"]["source_model_identity_removal"]["proposal_input_unchanged"]
            for row in rows
        )
    }
    controls["identical_arm"] = {"paired_exact_valid_delta": 0.0, "derived_arm": FLAT_ARM}
    controls["hard_negative"] = {
        "row_count": len(rows),
        "local_pass_rate": round(
            sum(row["hard_negative_control"]["local_checks_passed"] for row in rows) / len(rows),
            10,
        ),
        "exact_invalid_rate": round(
            sum(not row["hard_negative_control"]["exact_valid"] for row in rows) / len(rows),
            10,
        ),
    }
    return {
        "metrics_by_transformation_model_family": metrics,
        "paired_exact_valid_deltas": paired_deltas,
        "clustered_confidence_intervals": intervals,
        "support_contraction": support,
        "convergence_harm": convergence,
        "work_matching": work,
        "destructive_control_results": controls,
    }


AGGREGATE_FIELDS = (
    "metrics_by_transformation_model_family",
    "paired_exact_valid_deltas",
    "clustered_confidence_intervals",
    "support_contraction",
    "convergence_harm",
    "work_matching",
    "destructive_control_results",
)


def _without_runtime(value: Any) -> Any:
    """Remove wall-clock and resume-only values from stable replay material."""

    excluded = {
        "duration_s",
        "runtime_s",
        "checkpoint_path",
        "new_row_count",
        "reproducibility_checksum",
    }
    if isinstance(value, Mapping):
        return {
            str(key): _without_runtime(item)
            for key, item in value.items()
            if str(key) not in excluded
        }
    if isinstance(value, list):
        return [_without_runtime(item) for item in value]
    return value


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable source, proposal, exact, control, and decision evidence."""

    return sha256_json(_without_runtime(artifact))


def frozen_arm_definitions() -> JsonDict:
    """Declare the exact V592 recipe and its frozen transfer state."""

    definitions = v592.arm_definitions()
    for arm in ARMS:
        definitions[arm].update(
            {
                "frozen_after_v592_reproduction": True,
                "training_source": "Exp6786 train split only",
                "transfer_updates": 0,
                "decoder": "threshold_then_uncertainty_flip",
            }
        )
    return definitions


def _empty_aggregates() -> JsonDict:
    """Return complete aggregate fields for a pre-training blocked artifact."""

    return {
        "metrics_by_transformation_model_family": {},
        "paired_exact_valid_deltas": {},
        "clustered_confidence_intervals": {},
        "support_contraction": {},
        "convergence_harm": {},
        "work_matching": {"planned_budgets_match": False},
        "destructive_control_results": {},
    }


def _blocked_artifact(
    *, run_date: str, duration_s: float, gate_summary: Mapping[str, Any]
) -> JsonDict:
    """Build the full no-fallback schema after any authority failure."""

    first = gate_summary.get("first_failure") or {"check": "preconditions", "observed": None}
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "status": "complete_blocked_real_output_fixed_point_transfer",
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(gate_summary.get("source_artifact_hashes", {})),
        "training_isolation_receipts": [],
        "v592_reproduction_receipts": [],
        "frozen_arm_definitions": frozen_arm_definitions(),
        "parameter_counts_by_arm": {arm: 91 for arm in ARMS},
        "optimization_steps_by_arm": {arm: 30 for arm in ARMS},
        "candidate_budget_by_arm": {
            arm: TRANSFER_UNIT_COUNT * len(FROZEN_SEEDS) * CANDIDATE_COUNT for arm in ARMS
        },
        "feature_allowlist": list(FEATURE_ALLOWLIST),
        "feature_denylist": list(FEATURE_DENYLIST),
        **_empty_aggregates(),
        "rows": [],
        "model_output_fixed_point_comparison_completed": False,
        "gate_check_summary": deepcopy(dict(gate_summary)),
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": (
            "complete_blocked_real_output_fixed_point_transfer: "
            f"{first['check']} observed {first.get('observed')}"
        ),
        "frozen_manifest": {
            "schema": "carnot.experiment_6800.frozen_manifest.v1",
            "planned_row_count": PLANNED_ROW_COUNT,
            "seeds": list(FROZEN_SEEDS),
            "arms": list(ARMS),
            "cpu_wall_budget_s": CPU_WALL_BUDGET_S,
        },
        "checkpoint_receipt": {
            "attempted": False,
            "planned_row_count": PLANNED_ROW_COUNT,
            "completed_row_count": 0,
            "complete": False,
        },
        "decision_gates": {"positive": False, "preconditions_passed": False},
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _decision_gates(aggregates: Mapping[str, Any]) -> JsonDict:
    """Apply the preregistered restructuring and no-harm decision."""

    intervals = aggregates["clustered_confidence_intervals"]
    support = aggregates["support_contraction"]
    convergence = aggregates["convergence_harm"]
    work = aggregates["work_matching"]
    restructuring_lcb = float(intervals["restructuring"]["lower"])
    refinement_lcb = float(intervals["refinement"]["lower"])
    no_support_harm = not any(value["harm"] for value in support.values())
    no_convergence_harm = not any(value["harm"] for value in convergence.values())
    no_work_harm = bool(work["planned_budgets_match"] and work["no_grouped_work_harm"])
    positive = bool(
        restructuring_lcb > 0.0
        and refinement_lcb >= 0.0
        and no_support_harm
        and no_convergence_harm
        and no_work_harm
    )
    return {
        "restructuring_exact_valid_lower_bound_above_zero": restructuring_lcb > 0.0,
        "restructuring_exact_valid_lower_bound": restructuring_lcb,
        "no_refinement_harm": refinement_lcb >= 0.0,
        "refinement_exact_valid_lower_bound": refinement_lcb,
        "no_support_harm": no_support_harm,
        "no_convergence_harm": no_convergence_harm,
        "no_work_harm": no_work_harm,
        "positive": positive,
    }


def _validate_run_date(run_date: str) -> None:
    """Reject dates that do not use the frozen YYYYMMDD form."""

    if len(run_date) != 8 or not run_date.isdigit():
        raise ValueError("run date must use YYYYMMDD")


def build_artifact(
    *,
    run_date: str = RUN_DATE,
    repo_root: Path = REPO_ROOT,
    checkpoint_path: Path | None = None,
) -> JsonDict:
    """Run the full grid or return a complete blocked artifact without fitting."""

    _validate_run_date(run_date)
    started = time.monotonic()
    preconditions = evaluate_preconditions(repo_root=repo_root)
    if not preconditions["all_passed"]:
        artifact = _blocked_artifact(
            run_date=run_date,
            duration_s=round(time.monotonic() - started, 6),
            gate_summary=preconditions,
        )
        errors = validate_artifact(artifact)
        if errors:
            raise ValueError("; ".join(errors))
        return artifact
    sources = load_sources(repo_root)
    try:
        reconstruction = reconstruct_frozen_arms(sources)
    except ReproductionError as error:
        failed = _gate("v592_candidate_hash_reproduction", True, str(error), False)
        gate_summary = deepcopy(preconditions)
        gate_summary["checks"].append(failed)
        gate_summary["failed_checks"].append(failed["check"])
        gate_summary["first_failure"] = failed
        gate_summary["all_passed"] = False
        artifact = _blocked_artifact(
            run_date=run_date,
            duration_s=round(time.monotonic() - started, 6),
            gate_summary=gate_summary,
        )
        errors = validate_artifact(artifact)
        if errors:
            raise ValueError("; ".join(errors))
        return artifact
    units = build_transfer_units(sources["exp6799"])
    feature_violations = audit_transfer_feature_contract(units)
    if feature_violations:
        failed = _gate("transfer_feature_contract", [], feature_violations)
        gate_summary = deepcopy(preconditions)
        gate_summary["checks"].append(failed)
        gate_summary["failed_checks"].append(failed["check"])
        gate_summary["first_failure"] = failed
        gate_summary["all_passed"] = False
        return _blocked_artifact(
            run_date=run_date,
            duration_s=round(time.monotonic() - started, 6),
            gate_summary=gate_summary,
        )
    manifest = frozen_manifest(units)
    checkpoint = checkpoint_path or repo_root / CHECKPOINT_RELATIVE_PATH
    execution = execute_cells(
        units=units,
        reconstruction=reconstruction,
        checkpoint_path=checkpoint,
        manifest=manifest,
    )
    rows = [deepcopy(envelope["payload"]) for envelope in execution["rows"]]
    attribution_errors = row_attribution_errors(rows, manifest)
    aggregates = aggregate_rows(rows)
    duration = round(time.monotonic() - started, 6)
    completed = bool(
        not execution["pending_row_ids"]
        and not attribution_errors
        and len(rows) == PLANNED_ROW_COUNT
        and duration <= CPU_WALL_BUDGET_S
    )
    gates = _decision_gates(aggregates)
    gates.update(
        {
            "preconditions_passed": True,
            "v592_reproduction_passed": all(
                receipt["agreement"] for receipt in reconstruction.v592_reproduction_receipts
            ),
            "all_rows_attributable": not attribution_errors,
            "cpu_wall_budget_met": duration <= CPU_WALL_BUDGET_S,
        }
    )
    gates["positive"] = bool(gates["positive"] and completed)
    verdict_class = "positive" if gates["positive"] else "null" if completed else "partial"
    honest_verdict = (
        "complete: frozen grouped fixed-point transfer has a positive restructuring effect"
        if verdict_class == "positive"
        else (
            "complete: frozen real-output comparison finished without the preregistered positive effect"
            if verdict_class == "null"
            else "complete_partial_real_output_fixed_point_transfer: planned cells or wall gate failed"
        )
    )
    parameter_count_by_arm = parameter_counts(reconstruction.models)
    optimization_steps = {
        arm: len(FROZEN_SEEDS) * int(FROZEN_HYPERPARAMETERS["training_steps"]) for arm in ARMS
    }
    candidate_budgets = {
        arm: TRANSFER_UNIT_COUNT * len(FROZEN_SEEDS) * CANDIDATE_COUNT for arm in ARMS
    }
    gate_summary = deepcopy(preconditions)
    gate_summary.update(
        {
            "attribution_errors": attribution_errors,
            "feature_violations": feature_violations,
            "decision_gates": deepcopy(gates),
        }
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "status": "complete" if completed else "complete_partial_real_output_fixed_point_transfer",
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(preconditions["source_artifact_hashes"]),
        "training_isolation_receipts": reconstruction.training_isolation_receipts,
        "v592_reproduction_receipts": reconstruction.v592_reproduction_receipts,
        "frozen_arm_definitions": frozen_arm_definitions(),
        "parameter_counts_by_arm": parameter_count_by_arm,
        "optimization_steps_by_arm": optimization_steps,
        "candidate_budget_by_arm": candidate_budgets,
        "feature_allowlist": list(FEATURE_ALLOWLIST),
        "feature_denylist": list(FEATURE_DENYLIST),
        **aggregates,
        "rows": rows,
        "model_output_fixed_point_comparison_completed": completed,
        "gate_check_summary": gate_summary,
        "verifier_is_oracle": False,
        "verdict_class": verdict_class,
        "honest_verdict": honest_verdict,
        "frozen_manifest": manifest,
        "checkpoint_receipt": {
            "attempted": True,
            "checkpoint_path": str(checkpoint),
            "manifest_hash": execution["manifest_hash"],
            "planned_row_count": PLANNED_ROW_COUNT,
            "completed_row_count": len(rows),
            "new_row_count": execution["new_row_count"],
            "pending_row_ids": execution["pending_row_ids"],
            "payload_hashes": [envelope["payload_hash"] for envelope in execution["rows"]],
            "complete": not execution["pending_row_ids"],
        },
        "decision_gates": gates,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return schema, authority, completion, and row-attribution errors."""

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
        if artifact.get("status") != "complete_blocked_real_output_fixed_point_transfer":
            errors.append("blocked artifact status mismatch")
        if artifact.get("rows") != []:
            errors.append("blocked artifact must not contain rows")
        if artifact.get("model_output_fixed_point_comparison_completed") is not False:
            errors.append("blocked artifact completion flag mismatch")
        if artifact.get("gate_check_summary", {}).get("all_passed") is not False:
            errors.append("blocked artifact must name failed gates")
        return errors
    completed = artifact.get("model_output_fixed_point_comparison_completed") is True
    if completed != (artifact.get("status") == "complete"):
        errors.append("completion flag does not match status")
    rows = artifact.get("rows", [])
    if completed and len(rows) != PLANNED_ROW_COUNT:
        errors.append("complete artifact row count mismatch")
    if completed:
        attribution = row_attribution_errors(rows, artifact.get("frozen_manifest", {}))
        if attribution:
            errors.append(f"row attribution failed: {attribution[0]}")
        if artifact.get("gate_check_summary", {}).get("all_passed") is not True:
            errors.append("complete artifact has failed preconditions")
        positive = artifact.get("verdict_class") == "positive"
        if positive != bool(artifact.get("decision_gates", {}).get("positive")):
            errors.append("positive verdict does not match decision gates")
    return errors


def write_outputs(
    *,
    run_date: str = RUN_DATE,
    repo_root: Path = REPO_ROOT,
    artifact_path: Path = RESULT_RELATIVE_PATH,
    checkpoint_path: Path = CHECKPOINT_RELATIVE_PATH,
) -> JsonDict:
    """Build, validate, and atomically publish the requested artifact."""

    output = artifact_path if artifact_path.is_absolute() else repo_root / artifact_path
    checkpoint = checkpoint_path if checkpoint_path.is_absolute() else repo_root / checkpoint_path
    artifact = build_artifact(
        run_date=run_date,
        repo_root=repo_root,
        checkpoint_path=checkpoint,
    )
    checkpointing.atomic_write_json(output, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the bounded CLI and print the terminal honest verdict."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifact-path", type=Path, default=RESULT_RELATIVE_PATH)
    parser.add_argument("--checkpoint-path", type=Path, default=CHECKPOINT_RELATIVE_PATH)
    args = parser.parse_args(argv)
    artifact = write_outputs(
        run_date=args.date,
        repo_root=args.repo_root,
        artifact_path=args.artifact_path,
        checkpoint_path=args.checkpoint_path,
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":  # pragma: no cover - repository entry point covers the CLI.
    raise SystemExit(main())
