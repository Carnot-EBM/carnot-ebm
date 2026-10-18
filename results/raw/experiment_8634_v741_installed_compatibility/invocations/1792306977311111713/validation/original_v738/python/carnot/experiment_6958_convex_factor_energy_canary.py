"""Bounded input-convex factor energy canary for structured mappings.

The experiment asks a deliberately small question: when mapping defects are
represented as local, nonnegative factor variables, does constraining a neural
energy to be convex in those variables give a useful and reproducible ranking
signal?  It borrows that architectural shape from arXiv:2605.23395, but it is
not a paper reproduction.  The data, task, and metrics are Carnot's bounded
Exp6955 reformulation fixture.

The exact fixture is used only to select mappings known to have an unchanged
valid version and to evaluate ordering.  Model inputs contain no exact label or
solver status.  They contain five explainable pre-oracle channels: variable
coverage, mapped domains, affine witness consistency, objective direction,
and objective-order witness consistency.  A transparent sum of those channels
is retained as a non-learned upper control.

Spec: REQ-ENERGY-6958 and SCENARIO-ENERGY-6958-*.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as torch_functional


JsonDict = dict[str, Any]

EXPERIMENT_ID = 6958
RANDOM_SEED = 69_580
SPEC_PATH = Path("openspec/capabilities/energy-verification/spec.md")
FIXTURE_PATH = Path("results/experiment_6955_reformulation_fixture.json")
CORPUS_PATH = Path("results/checkpoints/experiment_6955_reformulation_fixture_corpus.json")
OUTPUT_PATH = Path("results/experiment_6958_convex_factor_energy_canary.json")
CHECKPOINT_DIR = Path("results/checkpoints/experiment_6958_convex_factor_energy_canary")
REPLAY_MANIFEST_PATH = CHECKPOINT_DIR / "replay_manifest.json"

INFERENCE_SUBSTRATE = "small_cpu_input_convex_factor_energy_training"
SCHEMA_VERSION = "carnot.exp6958.convex_factor_energy_canary.v1"
REPLAY_SCHEMA_VERSION = "carnot.exp6958.convex_factor_replay.v1"

FEATURE_NAMES = (
    "variable_coverage",
    "domain_correspondence",
    "affine_consistency",
    "objective_direction",
    "order_witness",
)
FEATURE_DIM = len(FEATURE_NAMES)
CORRUPTION_KINDS = FEATURE_NAMES
FORBIDDEN_FEATURE_FIELDS = (
    "expected_label",
    "claimed_relation",
    "exact_label",
    "enumeration_label",
    "z3_label",
    "solver_status",
    "authorities_agree",
    "quarantined",
    "verdict",
)

ARM_CONVEX = "input_convex_factor_sum"
ARM_MLP = "unconstrained_mlp_factor_sum"
ARM_LINEAR = "linear_factor_score"
ARM_HAND = "exact_hand_penalties"
ARM_SHUFFLED = "shuffled_label_convex_energy"
ARMS = (ARM_CONVEX, ARM_MLP, ARM_LINEAR, ARM_HAND, ARM_SHUFFLED)
LEARNED_ARMS = (ARM_CONVEX, ARM_MLP, ARM_LINEAR, ARM_SHUFFLED)
CONVEX_ARMS = (ARM_CONVEX, ARM_SHUFFLED)
EVALUATION_SPLITS = ("train", "calibration", "held_out", "prospective_family")

DEFAULT_SEEDS = (RANDOM_SEED, RANDOM_SEED + 1, RANDOM_SEED + 2)
DEFAULT_EPOCHS = 60
DEFAULT_CONVEXITY_SAMPLES = 12
DEFAULT_BOOTSTRAP_SAMPLES = 2_000

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "feature_rows",
    "corruption_rows",
    "split_rows",
    "arm_rows",
    "seed_rows",
    "training_rows",
    "convex_weight_rows",
    "jensen_rows",
    "finite_difference_rows",
    "projection_rows",
    "optimizer_rows",
    "ordering_rows",
    "calibration_rows",
    "size_transfer_rows",
    "latency_rows",
    "baseline_rows",
    "shuffled_label_rows",
    "paired_metric_rows",
    "confidence_interval_rows",
    "label_isolation_rows",
    "fresh_process_replay_rows",
    "checkpoint_paths",
    "random_seed",
    "reproducibility_checksum",
    "convex_factor_run_complete_score",
    "convex_factor_positive_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "field_principles": "A reason beside every field makes the evidence contract auditable.",
    "preconditions_checked": "Fail-closed dependency receipts prevent synthetic success after a missing input.",
    "inference_substrate": "The fixed substrate bounds the claim to small CPU training.",
    "duration_s": "Measured wall time distinguishes executed fitting from a schema-only receipt.",
    "source_artifact_hashes": "Content hashes bind results to the exact fixture, code, tests, and interfaces.",
    "rows": "Paired unit rows let a reader rebuild the headline without trusting aggregates.",
    "feature_rows": "Visible factor tensors make the learned input inspectable.",
    "corruption_rows": "Frozen corruption provenance prevents adaptive negative construction.",
    "split_rows": "Template identities expose cross-split leakage.",
    "arm_rows": "One row per control states its capacity and scientific role.",
    "seed_rows": "Every registered start remains in the denominator, including failures.",
    "training_rows": "Per-epoch losses prove matched optimizer work.",
    "convex_weight_rows": "Minimum constrained weights directly audit the architectural guarantee.",
    "jensen_rows": "Jensen tests probe the defining input-convex inequality.",
    "finite_difference_rows": "Directional second differences probe local convex curvature.",
    "projection_rows": "Projection receipts show negative convex-path weights cannot persist.",
    "optimizer_rows": "All projected starts expose convergence and deterministic failures.",
    "ordering_rows": "Pairwise valid-before-corrupt rows are the exact ranking endpoint.",
    "calibration_rows": "Calibration separates score ordering from threshold quality.",
    "size_transfer_rows": "Factor-count bands test small-to-large composition without refitting.",
    "latency_rows": "CPU timings bound the cost of the canary method.",
    "baseline_rows": "Linear and exact controls reveal learnability and available headroom.",
    "shuffled_label_rows": "A frozen label-shuffle control detects success unrelated to supervision.",
    "paired_metric_rows": "Same-pair deltas prevent unpaired aggregate comparisons.",
    "confidence_interval_rows": "Paired uncertainty prevents a point-estimate gate flip.",
    "label_isolation_rows": "Input audits prove exact labels and solver outcomes never enter features.",
    "fresh_process_replay_rows": "Independent checkpoint loading catches in-memory-only reproducibility.",
    "checkpoint_paths": "Durable learned parameters make the claimed scores replayable.",
    "random_seed": "A declared seed fixes corruptions, starts, fitting, and bootstrap draws.",
    "reproducibility_checksum": "A timing-free digest detects scientific payload drift.",
    "convex_factor_run_complete_score": "Completion requires every arm, seed, split, band, and replay row.",
    "convex_factor_positive_score": "The positive gate requires convexity and strictly positive paired intervals.",
    "gate_check_summary": "Expected and observed gate values make blocks and nulls diagnosable.",
    "verifier_is_oracle": "False keeps exact evaluation authority distinct from the learned method.",
    "verdict_class": "A closed class prevents prose from disguising a null or blocked run.",
    "honest_verdict": "A machine-readable terminal summary prevents ambiguous conductor handling.",
}


def canonical_json(value: Any) -> bytes:
    """Return stable JSON bytes used for hashes and frozen shuffle identities."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256_bytes(value: bytes) -> str:
    """Prefix a SHA-256 digest so hash algorithms remain explicit in artifacts."""

    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def sha256_path(path: Path) -> str | None:
    """Hash a file, returning null when a precondition target does not exist."""

    target = Path(path)
    return sha256_bytes(target.read_bytes()) if target.is_file() else None


def _fraction(value: Any) -> Fraction:
    """Read Exp6955's exact rational strings without introducing float drift."""

    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        return Fraction(int(value))
    return Fraction(str(value))


def _fraction_text(value: Fraction) -> str:
    """Serialize a rational using the same integer-or-fraction convention as Exp6955."""

    return (
        str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
    )


@dataclass
class Candidate:
    """One mapping candidate and only the metadata needed to keep it paired.

    ``factors`` is the sole model input.  The supervision bit and corruption
    name live outside it so the label-isolation audit can inspect the exact
    payload passed to each model.
    """

    candidate_id: str
    pair_id: str
    family: str
    generator_template: str
    split: str
    n_variables: int
    factors: list[list[float]]
    is_corrupt: int
    corruption_kind: str | None
    mapping_hash: str
    corruption_hash: str | None

    @property
    def factor_count(self) -> int:
        """Count real factors before batching adds masked padding."""

        return len(self.factors)


def _candidate_id(pair_id: str, variant: str) -> str:
    """Use an opaque ID so target words such as ``valid`` never become features."""

    return sha256_bytes(canonical_json({"pair_id": pair_id, "variant": variant}))


def corrupt_mapping(pair: Mapping[str, Any], kind: str) -> JsonDict:
    """Apply exactly one registered, deterministic structural mapping corruption."""

    if kind not in CORRUPTION_KINDS:
        raise ValueError(f"unknown_corruption:{kind}")
    mapping = deepcopy(pair["mapping"])
    if kind == "variable_coverage":
        mapping["variables"].pop(0)
    elif kind == "domain_correspondence":
        row = mapping["domain_clauses"][0]
        field = next(
            (
                name
                for name in ("target_lower", "target_upper", "source_lower", "source_upper")
                if row.get(name) is not None
            ),
            "target_lower",
        )
        row[field] = "0" if row.get(field) is None else _fraction_text(_fraction(row[field]) + 1)
    elif kind == "affine_consistency":
        row = mapping["variables"][0]
        row["offset"] = _fraction_text(_fraction(row["offset"]) + 1)
    elif kind == "objective_direction":
        objective = mapping["objective"]
        objective["target_direction"] = "max" if objective["target_direction"] == "min" else "min"
    else:
        objective = mapping["objective"]
        objective["scale"] = _fraction_text(-_fraction(objective["scale"]))
    return mapping


def _mapping_row(mapping: Mapping[str, Any], source_name: str) -> Mapping[str, Any] | None:
    """Return the unique row for a source variable; duplicates are coverage failures."""

    matches = [row for row in mapping.get("variables", []) if row.get("source") == source_name]
    return matches[0] if len(matches) == 1 else None


def _domain_row(mapping: Mapping[str, Any], source_name: str) -> Mapping[str, Any] | None:
    """Return the unique declared domain correspondence for one source variable."""

    matches = [row for row in mapping.get("domain_clauses", []) if row.get("source") == source_name]
    return matches[0] if len(matches) == 1 else None


def _same_fraction(left: Any, right: Any) -> bool:
    """Compare rational fields exactly and treat malformed values as mismatches."""

    if left is None or right is None:
        return left is None and right is None
    try:
        return _fraction(left) == _fraction(right)
    except (ValueError, ZeroDivisionError):
        return False


def encode_candidate(pair: Mapping[str, Any], mapping: Mapping[str, Any]) -> list[list[float]]:
    """Encode mapping structure and supplied witnesses into five factor channels.

    Each source variable contributes one local factor.  Objective direction and
    objective-order evidence contribute two global factors.  The result can
    grow with the number of variables, while a shared factor network keeps the
    learned parameter count fixed.
    """

    source = pair["source"]
    target = pair["target"]
    target_variables = {row["name"]: row for row in target["variables"]}
    source_names = [row["name"] for row in source["variables"]]
    target_names = [row["name"] for row in target["variables"]]
    mapped_sources = [row.get("source") for row in mapping.get("variables", [])]
    mapped_targets = [row.get("target") for row in mapping.get("variables", [])]
    witness = pair["objective_order_witness"]
    factors: list[list[float]] = []

    for source_variable in source["variables"]:
        source_name = source_variable["name"]
        row = _mapping_row(mapping, source_name)
        coverage = float(
            row is None
            or mapped_sources.count(source_name) != 1
            or row.get("target") not in target_names
            or mapped_targets.count(row.get("target")) != 1
            or set(mapped_sources) != set(source_names)
            or set(mapped_targets) != set(target_names)
        )

        domain = 1.0
        affine = 1.0
        if row is not None and row.get("target") in target_variables:
            target_variable = target_variables[str(row["target"])]
            declared = _domain_row(mapping, source_name)
            if declared is not None:
                domain = float(
                    declared.get("target") != row.get("target")
                    or not _same_fraction(
                        declared.get("source_lower"), source_variable["domain"]["lower"]
                    )
                    or not _same_fraction(
                        declared.get("source_upper"), source_variable["domain"]["upper"]
                    )
                    or not _same_fraction(
                        declared.get("target_lower"), target_variable["domain"]["lower"]
                    )
                    or not _same_fraction(
                        declared.get("target_upper"), target_variable["domain"]["upper"]
                    )
                )
            try:
                scale = _fraction(row["scale"])
                offset = _fraction(row["offset"])
                target_name = str(row["target"])
                affine_mismatch = []
                for side in ("left", "right"):
                    source_value = _fraction(witness[f"source_{side}"][source_name])
                    target_value = _fraction(witness[f"target_{side}"][target_name])
                    affine_mismatch.append(target_value != scale * source_value + offset)
                affine = float(any(affine_mismatch))
            except (KeyError, ValueError, ZeroDivisionError):
                affine = 1.0
        factors.append([coverage, domain, affine, 0.0, 0.0])

    objective = mapping.get("objective", {})
    direction_violation = 1.0
    order_violation = 1.0
    try:
        scale = _fraction(objective["scale"])
        source_direction = str(objective["source_direction"])
        target_direction = str(objective["target_direction"])
        sign_relation_ok = (scale > 0 and source_direction == target_direction) or (
            scale < 0 and source_direction != target_direction
        )
        direction_violation = float(
            scale == 0
            or source_direction != source["objective"]["direction"]
            or target_direction != target["objective"]["direction"]
            or not sign_relation_ok
        )
        offset = _fraction(objective["offset"])
        residuals = []
        for source_value, target_value in zip(
            witness["source_values"], witness["target_values"], strict=True
        ):
            left = _fraction(source_value)
            right = _fraction(target_value)
            residuals.append(
                abs(float(right - (scale * left + offset))) / (1.0 + abs(float(right)))
            )
        order_violation = max(residuals, default=1.0)
    except (KeyError, ValueError, ZeroDivisionError):
        direction_violation = 1.0
        order_violation = 1.0
    factors.append([0.0, 0.0, 0.0, direction_violation, 0.0])
    factors.append([0.0, 0.0, 0.0, 0.0, order_violation])
    return factors


def attach_fixture_witnesses(
    pairs: Sequence[Mapping[str, Any]], fixture: Mapping[str, Any]
) -> list[JsonDict]:
    """Join replayable Exp6955 witness rows onto its formulation checkpoint.

    Exp6955 intentionally separates large serialized formulations from its
    summary evidence rows.  Joining by the frozen pair ID preserves that
    separation and fails closed if either required witness row is absent.
    """

    feasibility = {
        str(row["pair_id"]): row.get("witness") for row in fixture["feasibility_witness_rows"]
    }
    objective = {str(row["pair_id"]): row.get("witness") for row in fixture["objective_order_rows"]}
    hydrated = []
    for source_pair in pairs:
        pair_id = str(source_pair["pair_id"])
        if pair_id not in feasibility or pair_id not in objective:
            raise ValueError(f"missing_fixture_witness_row:{pair_id}")
        pair = deepcopy(source_pair)
        pair["feasibility_witness"] = feasibility[pair_id]
        pair["objective_order_witness"] = objective[pair_id]
        hydrated.append(pair)
    return hydrated


def freeze_candidates(pairs: Sequence[Mapping[str, Any]], seed: int) -> list[Candidate]:
    """Freeze paired candidates before fitting, using only exact-equivalent bases.

    Non-equivalent Exp6955 rows have no valid mapping by definition, so using
    them as the supposedly valid member of a ranking pair would corrupt the
    endpoint.  Their generator templates still remain represented by the
    equivalent rows in each split.
    """

    equivalents = sorted(
        (row for row in pairs if row.get("expected_label") == "equivalent"),
        key=lambda row: str(row["pair_id"]),
    )
    rotation = seed % len(CORRUPTION_KINDS)
    candidates: list[Candidate] = []
    for index, pair in enumerate(equivalents):
        kind = CORRUPTION_KINDS[(index + rotation) % len(CORRUPTION_KINDS)]
        valid_mapping = deepcopy(pair["mapping"])
        corrupted_mapping = corrupt_mapping(pair, kind)
        for variant, mapping, label, corruption_kind in (
            ("a", valid_mapping, 0, None),
            ("b", corrupted_mapping, 1, kind),
        ):
            mapping_hash = sha256_bytes(canonical_json(mapping))
            corruption_hash = (
                sha256_bytes(
                    canonical_json(
                        {
                            "pair_id": pair["pair_id"],
                            "kind": corruption_kind,
                            "mapping_hash": mapping_hash,
                        }
                    )
                )
                if corruption_kind is not None
                else None
            )
            candidates.append(
                Candidate(
                    candidate_id=_candidate_id(str(pair["pair_id"]), variant),
                    pair_id=str(pair["pair_id"]),
                    family=str(pair["family"]),
                    generator_template=str(pair["generator_template"]),
                    split=str(pair["split"]),
                    n_variables=len(pair["source"]["variables"]),
                    factors=encode_candidate(pair, mapping),
                    is_corrupt=label,
                    corruption_kind=corruption_kind,
                    mapping_hash=mapping_hash,
                    corruption_hash=corruption_hash,
                )
            )
    return candidates


def feature_payload(candidate: Candidate) -> JsonDict:
    """Expose the literal tensor payload audited for label and solver leakage."""

    return {
        "factors": candidate.factors,
        "mask": [1.0] * candidate.factor_count,
        "factor_count": candidate.factor_count,
    }


def audit_split_isolation(candidates: Sequence[Candidate]) -> list[JsonDict]:
    """Prove each Exp6955 generator template remains assigned to one split."""

    grouped: dict[str, set[str]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.generator_template, set()).add(candidate.split)
    rows = [
        {
            "generator_template": template,
            "split": sorted(splits)[0] if len(splits) == 1 else None,
            "observed_splits": sorted(splits),
            "crosses_split": len(splits) != 1,
        }
        for template, splits in sorted(grouped.items())
    ]
    if any(row["crosses_split"] for row in rows):
        raise ValueError("generator_template_crosses_split")
    return rows


def batch_tensors(candidates: Sequence[Candidate]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-size factor lists and return an explicit zero/one mask."""

    if not candidates:
        raise ValueError("empty_candidate_batch")
    maximum = max(candidate.factor_count for candidate in candidates)
    factors = torch.zeros((len(candidates), maximum, FEATURE_DIM), dtype=torch.float64)
    mask = torch.zeros((len(candidates), maximum), dtype=torch.float64)
    for index, candidate in enumerate(candidates):
        count = candidate.factor_count
        factors[index, :count] = torch.tensor(candidate.factors, dtype=torch.float64)
        mask[index, :count] = 1.0
    return factors, mask


class ConvexFactorEnergy(nn.Module):
    """A small monotone input-convex network shared across mapping factors.

    Softplus is convex and nondecreasing.  Every input, hidden-to-hidden, and
    output weight is projected nonnegative, so composing these layers preserves
    convexity in each factor tensor.  Summing masked factors preserves it again.
    Biases may have either sign because affine shifts do not affect convexity.
    """

    def __init__(self, hidden_dim: int = 4) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_weight = nn.Parameter(
            torch.rand(hidden_dim, FEATURE_DIM, dtype=torch.float64) * 0.2
        )
        self.input_bias = nn.Parameter(torch.zeros(hidden_dim, dtype=torch.float64))
        self.hidden_weight = nn.Parameter(
            torch.rand(hidden_dim, hidden_dim, dtype=torch.float64) * 0.2
        )
        self.skip_weight = nn.Parameter(
            torch.rand(hidden_dim, FEATURE_DIM, dtype=torch.float64) * 0.2
        )
        self.hidden_bias = nn.Parameter(torch.zeros(hidden_dim, dtype=torch.float64))
        self.output_weight = nn.Parameter(torch.rand(1, hidden_dim, dtype=torch.float64) * 0.2)
        self.linear_weight = nn.Parameter(torch.rand(1, FEATURE_DIM, dtype=torch.float64) * 0.2)
        self.output_bias = nn.Parameter(torch.zeros(1, dtype=torch.float64))

    def constrained_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return exactly the paths whose sign is load-bearing for convexity."""

        return (
            self.input_weight,
            self.hidden_weight,
            self.skip_weight,
            self.output_weight,
            self.linear_weight,
        )

    def forward(self, factors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Score each factor independently and sum only real, unpadded factors."""

        factors = factors.to(dtype=self.input_weight.dtype)
        mask = mask.to(dtype=self.input_weight.dtype)
        first = torch_functional.softplus(
            torch_functional.linear(factors, self.input_weight, self.input_bias)
        )
        second = torch_functional.softplus(
            torch_functional.linear(first, self.hidden_weight, self.hidden_bias)
            + torch_functional.linear(factors, self.skip_weight)
        )
        factor_energy = (
            torch_functional.linear(second, self.output_weight, self.output_bias)
            + torch_functional.linear(factors, self.linear_weight)
        ).squeeze(-1)
        return (factor_energy * mask).sum(dim=-1)


class UnconstrainedFactorEnergy(ConvexFactorEnergy):
    """Parameter-matched nonlinear control whose weight signs are unrestricted."""

    def __init__(self, hidden_dim: int = 4) -> None:
        nn.Module.__init__(self)
        self.hidden_dim = hidden_dim
        self.input_weight = nn.Parameter(
            torch.randn(hidden_dim, FEATURE_DIM, dtype=torch.float64) * 0.2
        )
        self.input_bias = nn.Parameter(torch.zeros(hidden_dim, dtype=torch.float64))
        self.hidden_weight = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim, dtype=torch.float64) * 0.2
        )
        self.skip_weight = nn.Parameter(
            torch.randn(hidden_dim, FEATURE_DIM, dtype=torch.float64) * 0.2
        )
        self.hidden_bias = nn.Parameter(torch.zeros(hidden_dim, dtype=torch.float64))
        self.output_weight = nn.Parameter(torch.randn(1, hidden_dim, dtype=torch.float64) * 0.2)
        self.linear_weight = nn.Parameter(torch.randn(1, FEATURE_DIM, dtype=torch.float64) * 0.2)
        self.output_bias = nn.Parameter(torch.zeros(1, dtype=torch.float64))

    def constrained_parameters(self) -> tuple[nn.Parameter, ...]:
        """The control has no projected parameters; signed geometry is intentional."""

        return ()


class LinearFactorEnergy(nn.Module):
    """A learned affine score shared across factors and summed with the mask."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(FEATURE_DIM, 1, dtype=torch.float64)

    def forward(self, factors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply the same linear score to every factor before masked summation."""

        factors = factors.to(dtype=self.linear.weight.dtype)
        mask = mask.to(dtype=self.linear.weight.dtype)
        values = self.linear(factors).squeeze(-1)
        return (values * mask).sum(dim=-1)


class HandPenaltyEnergy(nn.Module):
    """Transparent non-learned control: sum every nonnegative violation channel."""

    def forward(self, factors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return the exact hand penalty without trainable parameters."""

        return (factors.sum(dim=-1) * mask).sum(dim=-1)


def make_model(arm: str, seed: int) -> nn.Module:
    """Construct one deterministically initialized CPU model for an arm."""

    torch.manual_seed(seed)
    if arm in CONVEX_ARMS:
        return ConvexFactorEnergy()
    if arm == ARM_MLP:
        return UnconstrainedFactorEnergy()
    if arm == ARM_LINEAR:
        return LinearFactorEnergy()
    if arm == ARM_HAND:
        return HandPenaltyEnergy()
    raise ValueError(f"unknown_arm:{arm}")


def model_parameter_count(model: nn.Module) -> int:
    """Count trainable scalar parameters for matched-capacity receipts."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def project_nonnegative_(model: nn.Module) -> None:
    """Project every declared convex path onto the nonnegative orthant in place."""

    if not isinstance(model, ConvexFactorEnergy) or isinstance(model, UnconstrainedFactorEnergy):
        return
    with torch.no_grad():
        for parameter in model.constrained_parameters():
            parameter.clamp_(min=0.0)


def minimum_convex_weight(model: nn.Module) -> float | None:
    """Return the smallest load-bearing convex-path weight, or null when inapplicable."""

    if not isinstance(model, ConvexFactorEnergy) or isinstance(model, UnconstrainedFactorEnergy):
        return None
    values = [float(parameter.detach().min()) for parameter in model.constrained_parameters()]
    return min(values)


def training_labels(candidates: Sequence[Candidate], seed: int, shuffled: bool) -> torch.Tensor:
    """Freeze supervision, with the shuffled control using a seeded fixed permutation."""

    labels = torch.tensor([candidate.is_corrupt for candidate in candidates], dtype=torch.float64)
    if shuffled:
        generator = torch.Generator(device="cpu").manual_seed(seed + 8_117)
        labels = labels[torch.randperm(len(labels), generator=generator)]
    return labels


@dataclass
class FitResult:
    """Keep a fitted model, its full optimizer trace, and budget metadata together."""

    model: nn.Module
    training_rows: list[JsonDict]
    metadata: JsonDict


def fit_arm(
    arm: str,
    candidates: Sequence[Candidate],
    seed: int,
    epochs: int,
) -> FitResult:
    """Fit one learned arm with one full-batch optimizer call per epoch."""

    if arm not in LEARNED_ARMS:
        raise ValueError(f"arm_is_not_learned:{arm}")
    model = make_model(arm, seed)
    factors, mask = batch_tensors(candidates)
    labels = training_labels(candidates, seed, shuffled=arm == ARM_SHUFFLED)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    training_rows: list[JsonDict] = []
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        energies = model(factors, mask)
        loss = torch_functional.binary_cross_entropy_with_logits(energies, labels)
        loss.backward()
        optimizer.step()
        project_nonnegative_(model)
        training_rows.append(
            {
                "arm": arm,
                "seed": seed,
                "epoch": epoch + 1,
                "optimizer_call": epoch + 1,
                "loss": float(loss.detach()),
                "finite": math.isfinite(float(loss.detach())),
            }
        )
    metadata = {
        "arm": arm,
        "seed": seed,
        "examples": len(candidates),
        "epochs": epochs,
        "optimizer_calls": epochs,
        "starts": 1,
        "parameter_count": model_parameter_count(model),
        "terminal": True,
        "failure": None,
    }
    return FitResult(model=model, training_rows=training_rows, metadata=metadata)


def ordering_credit(valid_energy: float, corrupt_energy: float, tolerance: float = 1e-12) -> float:
    """Award one for correct order, zero for reversal, and half for a true tie."""

    delta = corrupt_energy - valid_energy
    if delta > tolerance:
        return 1.0
    if delta < -tolerance:
        return 0.0
    return 0.5


def score_candidates(model: nn.Module, candidates: Sequence[Candidate]) -> dict[str, float]:
    """Evaluate a candidate list in one CPU batch and return ID-keyed energies."""

    factors, mask = batch_tensors(candidates)
    with torch.no_grad():
        energies = model(factors, mask)
    return {
        candidate.candidate_id: float(energy)
        for candidate, energy in zip(candidates, energies, strict=True)
    }


def ordering_rows(
    model: nn.Module, candidates: Sequence[Candidate], arm: str, seed: int, split: str
) -> list[JsonDict]:
    """Build one exact valid-versus-corrupt ordering row per mapping pair."""

    selected = [candidate for candidate in candidates if candidate.split == split]
    energies = score_candidates(model, selected)
    grouped: dict[str, list[Candidate]] = {}
    for candidate in selected:
        grouped.setdefault(candidate.pair_id, []).append(candidate)
    rows: list[JsonDict] = []
    for pair_id, pair_candidates in sorted(grouped.items()):
        valid = next(candidate for candidate in pair_candidates if candidate.is_corrupt == 0)
        corrupt = next(candidate for candidate in pair_candidates if candidate.is_corrupt == 1)
        valid_energy = energies[valid.candidate_id]
        corrupt_energy = energies[corrupt.candidate_id]
        credit = ordering_credit(valid_energy, corrupt_energy)
        rows.append(
            {
                "arm": arm,
                "seed": seed,
                "split": split,
                "pair_id": pair_id,
                "family": valid.family,
                "n_variables": valid.n_variables,
                "factor_count": valid.factor_count,
                "valid_candidate_id": valid.candidate_id,
                "corrupt_candidate_id": corrupt.candidate_id,
                "valid_energy": valid_energy,
                "corrupt_energy": corrupt_energy,
                "energy_margin": corrupt_energy - valid_energy,
                "ordering_credit": credit,
                "outcome": "win" if credit == 1.0 else "loss" if credit == 0.0 else "tie",
                "terminal": True,
                "failure": None,
            }
        )
    return rows


def _property_inputs(seed: int, samples: int) -> list[tuple[torch.Tensor, torch.Tensor, float]]:
    """Create deterministic interior points shared by all geometry checks."""

    generator = torch.Generator(device="cpu").manual_seed(seed + 19_771)
    rows = []
    for index in range(samples):
        x = torch.rand((1, 4, FEATURE_DIM), generator=generator, dtype=torch.float64)
        y = torch.rand((1, 4, FEATURE_DIM), generator=generator, dtype=torch.float64)
        theta = 0.2 + 0.6 * ((index + 1) / (samples + 1))
        rows.append((x, y, theta))
    return rows


def convexity_checks(model: nn.Module, seed: int, samples: int) -> dict[str, list[JsonDict]]:
    """Measure Jensen, finite-difference, and monotone-gradient inequalities."""

    mask = torch.ones((1, 4), dtype=torch.float64)
    jensen: list[JsonDict] = []
    finite: list[JsonDict] = []
    gradient: list[JsonDict] = []
    for index, (x, y, theta) in enumerate(_property_inputs(seed, samples)):
        with torch.no_grad():
            left = float(model(theta * x + (1.0 - theta) * y, mask))
            right = theta * float(model(x, mask)) + (1.0 - theta) * float(model(y, mask))
        margin = right - left
        jensen.append(
            {
                "sample": index,
                "lhs": left,
                "rhs": right,
                "margin": margin,
                "passed": margin >= -1e-9,
            }
        )

        midpoint = (x + y) / 2.0
        direction = y - x
        step = 0.05
        with torch.no_grad():
            second = float(
                model(midpoint + step * direction, mask)
                + model(midpoint - step * direction, mask)
                - 2.0 * model(midpoint, mask)
            )
        finite.append(
            {"sample": index, "directional_second_difference": second, "passed": second >= -1e-9}
        )

        gx_input = x.detach().clone().requires_grad_(True)
        gy_input = y.detach().clone().requires_grad_(True)
        gx = torch.autograd.grad(model(gx_input, mask).sum(), gx_input)[0]
        gy = torch.autograd.grad(model(gy_input, mask).sum(), gy_input)[0]
        monotone_inner_product = float(((gy - gx) * (y - x)).sum().detach())
        gradient.append(
            {
                "sample": index,
                "monotone_inner_product": monotone_inner_product,
                "passed": monotone_inner_product >= -1e-9,
            }
        )
    return {"jensen": jensen, "finite_difference": finite, "gradient": gradient}


def projected_optimize(
    model: nn.Module,
    start: torch.Tensor,
    steps: int = 120,
    learning_rate: float = 4.0,
) -> JsonDict:
    """Minimize factor values inside [0,1] and retain every registered start."""

    value = start.detach().clone().to(dtype=torch.float64)
    mask = torch.ones((1, value.shape[0]), dtype=torch.float64)
    initial = float(model(value.unsqueeze(0), mask).detach())
    finite = True
    for _ in range(steps):
        variable = value.detach().clone().requires_grad_(True)
        energy = model(variable.unsqueeze(0), mask).sum()
        gradient = torch.autograd.grad(energy, variable)[0]
        with torch.no_grad():
            value = (variable - learning_rate * gradient).clamp(0.0, 1.0)
        finite = (
            finite and bool(torch.isfinite(value).all()) and math.isfinite(float(energy.detach()))
        )
    final = float(model(value.unsqueeze(0), mask).detach())
    return {
        "initial_energy": initial,
        "final_energy": final,
        "steps": steps,
        "minimum_coordinate": float(value.min()),
        "maximum_coordinate": float(value.max()),
        "converged": finite and final <= initial + 1e-10,
        "terminal": True,
        "failure": None if finite else "non_finite_optimization",
    }


def bootstrap_ci(
    values: Sequence[float], seed: int, samples: int
) -> tuple[float | None, float | None]:
    """Return a deterministic paired percentile CI, preserving empty cells as null."""

    if not values:
        return None, None
    generator = random.Random(seed)
    means = []
    for _ in range(samples):
        means.append(sum(values[generator.randrange(len(values))] for _ in values) / len(values))
    means.sort()
    low_index = max(0, math.floor(0.025 * (samples - 1)))
    high_index = min(samples - 1, math.ceil(0.975 * (samples - 1)))
    return means[low_index], means[high_index]


def positive_gate(
    completion: bool,
    convexity_passed: bool,
    intervals: Sequence[tuple[float | None, float | None]],
) -> bool:
    """Apply the strict positive rule; a lower bound equal to zero does not pass."""

    return bool(
        completion
        and convexity_passed
        and len(intervals) == 2
        and all(low is not None and high is not None and low > 0.0 for low, high in intervals)
    )


def _threshold(energies: Sequence[float], labels: Sequence[int]) -> float:
    """Choose a deterministic calibration threshold from observed energy midpoints."""

    unique = sorted(set(energies))
    candidates = (
        [unique[0] - 1.0]
        + [(left + right) / 2.0 for left, right in zip(unique, unique[1:], strict=False)]
        + [unique[-1] + 1.0]
    )
    scored = []
    for threshold in candidates:
        accuracy = sum(
            int((energy > threshold) == bool(label))
            for energy, label in zip(energies, labels, strict=True)
        ) / len(labels)
        scored.append((accuracy, -abs(threshold), -threshold, threshold))
    return max(scored)[-1]


def _classification_metrics(
    energies: Sequence[float], labels: Sequence[int], threshold: float
) -> tuple[float, float, float]:
    """Report accuracy, Brier score, and five-bin expected calibration error."""

    probabilities = [
        1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, value - threshold)))) for value in energies
    ]
    accuracy = sum(
        int((value > threshold) == bool(label))
        for value, label in zip(energies, labels, strict=True)
    ) / len(labels)
    brier = sum(
        (probability - label) ** 2 for probability, label in zip(probabilities, labels, strict=True)
    ) / len(labels)
    ece = 0.0
    for bin_index in range(5):
        lower = bin_index / 5.0
        upper = (bin_index + 1) / 5.0
        members = [
            (probability, label)
            for probability, label in zip(probabilities, labels, strict=True)
            if (lower <= probability <= upper if bin_index == 4 else lower <= probability < upper)
        ]
        if members:
            confidence = sum(row[0] for row in members) / len(members)
            frequency = sum(row[1] for row in members) / len(members)
            ece += len(members) / len(labels) * abs(confidence - frequency)
    return accuracy, brier, ece


def _gate(check: str, expected: Any, observed: Any) -> JsonDict:
    """Create one machine-readable equality gate row."""

    return {
        "check": check,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
    }


def _summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Retain all checks and separately list exact failed observations."""

    rows = [dict(row) for row in checks]
    return {
        "passed": all(row["passed"] for row in rows),
        "checks": rows,
        "failed_checks": [row for row in rows if not row["passed"]],
    }


def check_preconditions(
    repo_root: Path,
    fixture_path: Path,
    corpus_path: Path,
    checkpoint_dir: Path,
) -> JsonDict:
    """Check the complete fixture, frozen splits, CPU autograd, and writable target."""

    root = Path(repo_root)
    fixture_target = Path(fixture_path)
    corpus_target = Path(corpus_path)
    checks: list[JsonDict] = []
    try:
        fixture = json.loads(fixture_target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fixture = {}
        fixture_error = type(exc).__name__
    else:
        fixture_error = None
    try:
        corpus = json.loads(corpus_target.read_text(encoding="utf-8"))
        pairs = corpus.get("pairs", [])
    except (OSError, json.JSONDecodeError) as exc:
        pairs = []
        corpus_error = type(exc).__name__
    else:
        corpus_error = None

    row_count = len(fixture.get("rows", []))
    mapping_pair_ids = {str(row.get("pair_id")) for row in fixture.get("mapping_rows", [])}
    feasibility_pair_ids = {
        str(row.get("pair_id")) for row in fixture.get("feasibility_witness_rows", [])
    }
    objective_pair_ids = {
        str(row.get("pair_id")) for row in fixture.get("objective_order_rows", [])
    }
    fixture_pair_ids = {str(row.get("pair_id")) for row in fixture.get("rows", [])}
    checks.extend(
        [
            _gate("fixture_readable", True, fixture_error is None),
            _gate("corpus_readable", True, corpus_error is None),
            _gate(
                "reformulation_fixture_ready_score",
                1,
                fixture.get("reformulation_fixture_ready_score"),
            ),
            _gate("mapping_rows_complete", row_count, len(fixture.get("mapping_rows", []))),
            _gate(
                "feasibility_witness_rows_complete",
                row_count,
                len(fixture.get("feasibility_witness_rows", [])),
            ),
            _gate(
                "objective_order_rows_complete",
                row_count,
                len(fixture.get("objective_order_rows", [])),
            ),
            _gate("checkpoint_pair_count", row_count, len(pairs)),
        ]
    )
    split_rows = fixture.get("split_rows", [])
    template_splits: dict[str, set[str]] = {}
    for row in split_rows:
        template_splits.setdefault(str(row.get("generator_template")), set()).add(
            str(row.get("split"))
        )
    observed_splits = sorted({str(row.get("split")) for row in split_rows})
    checks.extend(
        [
            _gate("fixed_split_roster", sorted(EVALUATION_SPLITS), observed_splits),
            _gate(
                "template_cross_split_count",
                0,
                sum(len(value) != 1 for value in template_splits.values()),
            ),
            _gate(
                "mapping_payloads_present",
                row_count,
                sum(
                    isinstance(row.get("mapping"), dict) for row in fixture.get("mapping_rows", [])
                ),
            ),
            _gate(
                "witness_payloads_present",
                sorted(fixture_pair_ids),
                sorted(feasibility_pair_ids),
            ),
            _gate(
                "order_witness_payloads_present",
                sorted(fixture_pair_ids),
                sorted(objective_pair_ids),
            ),
            _gate("mapping_pair_ids_complete", sorted(fixture_pair_ids), sorted(mapping_pair_ids)),
        ]
    )

    try:
        probe = torch.tensor([1.0], dtype=torch.float64, device="cpu", requires_grad=True)
        (probe.square().sum()).backward()
        torch_observed = probe.grad is not None and probe.device.type == "cpu"
    except RuntimeError:
        torch_observed = False
    checks.append(_gate("pytorch_cpu_autograd", True, torch_observed))

    writable = False
    target_dir = Path(checkpoint_dir)
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target_dir, prefix=".exp6958-", delete=True):
            writable = True
    except OSError:
        writable = False
    checks.append(_gate("writable_checkpoints", True, writable))
    result = _summary(checks)
    result["fixture_path"] = str(fixture_target)
    result["corpus_path"] = str(corpus_target)
    result["checkpoint_dir"] = str(target_dir)
    result["repo_root"] = str(root)
    return result


def source_artifact_hashes(repo_root: Path, fixture_path: Path, corpus_path: Path) -> JsonDict:
    """Bind the run to the actual current interfaces, including corrected paths."""

    root = Path(repo_root)
    sources = {
        "spec": root / SPEC_PATH,
        "module": root / "python/carnot/experiment_6958_convex_factor_energy_canary.py",
        "wrapper": root / "scripts/experiments/experiment_6958_convex_factor_energy_canary.py",
        "tests": root / "tests/python/test_experiment_6958_convex_factor_energy_canary.py",
        "core_energy_interface": root / "python/carnot/core/energy.py",
        "constraint_interface": root / "python/carnot/verify/constraint.py",
        "exp6955_fixture": Path(fixture_path),
        "exp6955_corpus": Path(corpus_path),
    }
    return {name: sha256_path(path) for name, path in sources.items()}


def _empty_row_fields() -> JsonDict:
    """Return every required list field so a blocked artifact remains schema-complete."""

    fields = (
        "rows",
        "feature_rows",
        "corruption_rows",
        "split_rows",
        "arm_rows",
        "seed_rows",
        "training_rows",
        "convex_weight_rows",
        "jensen_rows",
        "finite_difference_rows",
        "projection_rows",
        "optimizer_rows",
        "ordering_rows",
        "calibration_rows",
        "size_transfer_rows",
        "latency_rows",
        "baseline_rows",
        "shuffled_label_rows",
        "paired_metric_rows",
        "confidence_interval_rows",
        "label_isolation_rows",
        "fresh_process_replay_rows",
        "checkpoint_paths",
    )
    return {field: [] for field in fields}


def build_blocked_artifact(
    date: str,
    duration_s: float,
    preconditions: Mapping[str, Any],
    hashes: Mapping[str, Any],
) -> JsonDict:
    """Emit the exact blocked contract without attempting any model fitting."""

    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_date": date,
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": dict(preconditions),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "source_artifact_hashes": dict(hashes),
        **_empty_row_fields(),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "convex_factor_run_complete_score": 0,
        "convex_factor_positive_score": 0,
        "gate_check_summary": {
            "passed": False,
            "checks": list(preconditions["checks"]),
            "failed_checks": list(preconditions["failed_checks"]),
        },
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_convex_factor_energy_canary",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _save_checkpoint(path: Path, arm: str, seed: int, model: nn.Module) -> None:
    """Atomically persist only task-owned tensors and minimal reconstruction metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}-", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(
            {
                "schema_version": REPLAY_SCHEMA_VERSION,
                "arm": arm,
                "seed": seed,
                "state_dict": model.state_dict(),
            },
            temporary,
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_checkpoint(path: Path, arm: str, seed: int) -> nn.Module:
    """Reconstruct one task-owned model and reject mismatched checkpoint metadata."""

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != REPLAY_SCHEMA_VERSION:
        raise ValueError("checkpoint_schema_mismatch")
    if payload.get("arm") != arm or payload.get("seed") != seed:
        raise ValueError("checkpoint_binding_mismatch")
    model = make_model(arm, seed)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def _manifest_candidates(candidates: Sequence[Candidate]) -> list[JsonDict]:
    """Serialize held-out factor tensors and pairing metadata for child replay."""

    return [
        {
            "candidate_id": candidate.candidate_id,
            "pair_id": candidate.pair_id,
            "split": candidate.split,
            "is_corrupt": candidate.is_corrupt,
            "factors": candidate.factors,
        }
        for candidate in candidates
        if candidate.split in {"held_out", "prospective_family"}
    ]


def _replay_candidate(row: Mapping[str, Any]) -> Candidate:
    """Rehydrate only the fields needed for scoring and exact pair ordering."""

    return Candidate(
        candidate_id=str(row["candidate_id"]),
        pair_id=str(row["pair_id"]),
        family="replay",
        generator_template="replay",
        split=str(row["split"]),
        n_variables=max(0, len(row["factors"]) - 2),
        factors=[[float(value) for value in factor] for factor in row["factors"]],
        is_corrupt=int(row["is_corrupt"]),
        corruption_kind=None,
        mapping_hash="replay",
        corruption_hash=None,
    )


def replay_manifest(path: Path) -> list[JsonDict]:
    """Load checkpoints from a serialized manifest and recompute replay metrics."""

    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != REPLAY_SCHEMA_VERSION:
        raise ValueError("replay_manifest_schema")
    rows = []
    for entry in manifest.get("models", []):
        checkpoint_path = Path(entry["checkpoint_path"])
        observed_hash = sha256_path(checkpoint_path)
        if observed_hash != entry["checkpoint_sha256"]:
            raise ValueError("checkpoint_hash_mismatch")
        arm = str(entry["arm"])
        seed = int(entry["seed"])
        model = _load_checkpoint(checkpoint_path, arm, seed)
        candidates = [_replay_candidate(row) for row in entry["candidates"]]
        energies = score_candidates(model, candidates)
        replay_ordering = []
        for split in ("held_out", "prospective_family"):
            selected = [candidate for candidate in candidates if candidate.split == split]
            grouped: dict[str, list[Candidate]] = {}
            for candidate in selected:
                grouped.setdefault(candidate.pair_id, []).append(candidate)
            ordering_scores = []
            for pair_candidates in grouped.values():
                valid = next(row for row in pair_candidates if row.is_corrupt == 0)
                corrupt = next(row for row in pair_candidates if row.is_corrupt == 1)
                ordering_scores.append(
                    ordering_credit(energies[valid.candidate_id], energies[corrupt.candidate_id])
                )
            replay_ordering.append(
                {
                    "split": split,
                    "pair_count": len(ordering_scores),
                    "exact_ordering_accuracy": (
                        sum(ordering_scores) / len(ordering_scores) if ordering_scores else None
                    ),
                }
            )
        rows.append(
            {
                "arm": arm,
                "seed": seed,
                "checkpoint_sha256": observed_hash,
                "energy_checksum": sha256_bytes(canonical_json(energies)),
                "ordering": replay_ordering,
            }
        )
    return rows


def _write_json(path: Path, value: Any) -> None:
    """Atomically replace JSON so interrupted writes cannot masquerade as artifacts."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=target.parent, prefix=f".{target.name}-", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, target)


def _fresh_process_replay(repo_root: Path, manifest_path: Path) -> list[JsonDict]:
    """Launch a clean interpreter whose only scientific input is the replay manifest."""

    output_path = Path(manifest_path).with_suffix(".child.json")
    env = dict(os.environ)
    python_root = str(Path(repo_root) / "python")
    env["PYTHONPATH"] = python_root + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "carnot.experiment_6958_convex_factor_energy_canary",
            "--replay-manifest",
            str(manifest_path),
            "--replay-output",
            str(output_path),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"fresh_process_replay_failed:{completed.stderr[-500:]}")
    rows = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise RuntimeError("fresh_process_replay_not_rows")
    return rows


def _feature_rows(candidates: Sequence[Candidate]) -> list[JsonDict]:
    """Record model inputs without adjoining supervision or corruption metadata."""

    return [
        {
            "candidate_id": candidate.candidate_id,
            "pair_id": candidate.pair_id,
            "family": candidate.family,
            "generator_template": candidate.generator_template,
            "split": candidate.split,
            "n_variables": candidate.n_variables,
            **feature_payload(candidate),
        }
        for candidate in candidates
    ]


def _label_isolation_rows(candidates: Sequence[Candidate]) -> list[JsonDict]:
    """Audit the exact feature payload for every forbidden oracle field name."""

    rows = []
    for candidate in candidates:
        payload = feature_payload(candidate)
        encoded = canonical_json(payload).decode("ascii")
        found = [
            field for field in FORBIDDEN_FEATURE_FIELDS if field in encoded or field in payload
        ]
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "feature_hash": sha256_bytes(canonical_json(payload)),
                "forbidden_fields_found": found,
                "passed": not found,
            }
        )
    return rows


def _checkpoint_text(path: Path, repo_root: Path) -> str:
    """Prefer portable relative paths while allowing temporary test targets."""

    return str(path.relative_to(repo_root)) if path.is_relative_to(repo_root) else str(path)


def build_artifact(
    date: str,
    repo_root: Path,
    fixture_path: Path,
    corpus_path: Path,
    checkpoint_dir: Path,
    replay_manifest_path: Path,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    epochs: int = DEFAULT_EPOCHS,
    convexity_samples: int = DEFAULT_CONVEXITY_SAMPLES,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
) -> JsonDict:
    """Run all arms, property checks, paired metrics, checkpoints, and replay."""

    started = time.perf_counter()
    root = Path(repo_root)
    fixture_target = Path(fixture_path)
    corpus_target = Path(corpus_path)
    checkpoint_target = Path(checkpoint_dir)
    preconditions = check_preconditions(root, fixture_target, corpus_target, checkpoint_target)
    hashes = source_artifact_hashes(root, fixture_target, corpus_target)
    if not preconditions["passed"]:
        return build_blocked_artifact(date, time.perf_counter() - started, preconditions, hashes)

    fixture = json.loads(fixture_target.read_text(encoding="utf-8"))
    raw_pairs = json.loads(corpus_target.read_text(encoding="utf-8"))["pairs"]
    pairs = attach_fixture_witnesses(raw_pairs, fixture)
    candidates = freeze_candidates(pairs, RANDOM_SEED)
    split_rows = audit_split_isolation(candidates)
    feature_rows = _feature_rows(candidates)
    label_isolation_rows = _label_isolation_rows(candidates)
    corruption_rows = [
        {
            "candidate_id": candidate.candidate_id,
            "pair_id": candidate.pair_id,
            "corruption_kind": candidate.corruption_kind,
            "corruption_hash": candidate.corruption_hash,
            "mapping_hash": candidate.mapping_hash,
            "frozen_before_fitting": True,
        }
        for candidate in candidates
        if candidate.is_corrupt
    ]

    # Only mappings with at most two source variables enter fitting.  Larger
    # Boolean-cardinality templates therefore remain a genuine composition-size
    # transfer cell rather than leaking into parameter updates.
    train_candidates = [
        candidate
        for candidate in candidates
        if candidate.split == "train" and candidate.n_variables <= 2
    ]
    models: dict[tuple[str, int], nn.Module] = {}
    training_rows: list[JsonDict] = []
    seed_rows: list[JsonDict] = []
    checkpoint_paths: list[str] = []
    manifest_models: list[JsonDict] = []

    for arm in ARMS:
        for seed in seeds:
            if arm in LEARNED_ARMS:
                fitted = fit_arm(arm, train_candidates, seed, epochs)
                model = fitted.model
                training_rows.extend(fitted.training_rows)
                seed_rows.append(dict(fitted.metadata))
                checkpoint_path = checkpoint_target / f"{arm}_seed_{seed}.pt"
                _save_checkpoint(checkpoint_path, arm, seed, model)
                checkpoint_hash = sha256_path(checkpoint_path)
                checkpoint_paths.append(_checkpoint_text(checkpoint_path, root))
                manifest_models.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "checkpoint_path": str(checkpoint_path.resolve()),
                        "checkpoint_sha256": checkpoint_hash,
                        "candidates": _manifest_candidates(candidates),
                    }
                )
            else:
                model = make_model(arm, seed)
                seed_rows.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "examples": len(train_candidates),
                        "epochs": 0,
                        "optimizer_calls": 0,
                        "starts": 1,
                        "parameter_count": 0,
                        "terminal": True,
                        "failure": None,
                        "training_status": "not_applicable_non_learned_control",
                    }
                )
                training_rows.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "epoch": None,
                        "optimizer_call": None,
                        "loss": None,
                        "finite": None,
                        "status": "not_applicable_non_learned_control",
                    }
                )
            model.eval()
            models[(arm, seed)] = model

    arm_rows = []
    for arm in ARMS:
        parameter_counts = sorted({model_parameter_count(models[(arm, seed)]) for seed in seeds})
        arm_rows.append(
            {
                "arm": arm,
                "learned": arm in LEARNED_ARMS,
                "convex_by_construction": arm in CONVEX_ARMS or arm in {ARM_LINEAR, ARM_HAND},
                "parameter_counts": parameter_counts,
                "examples": len(train_candidates),
                "epochs": epochs if arm in LEARNED_ARMS else 0,
                "optimizer_calls": epochs if arm in LEARNED_ARMS else 0,
                "seed_count": len(seeds),
                "terminal": True,
                "scientific_role": (
                    "learned_headline"
                    if arm == ARM_CONVEX
                    else "transparent_non_learned_upper_control"
                    if arm == ARM_HAND
                    else "learned_control"
                    if arm in {ARM_MLP, ARM_LINEAR}
                    else "shuffled_supervision_control"
                ),
            }
        )

    convex_weight_rows: list[JsonDict] = []
    projection_rows: list[JsonDict] = []
    jensen_rows: list[JsonDict] = []
    finite_difference_rows: list[JsonDict] = []
    optimizer_rows: list[JsonDict] = []
    for arm in ARMS:
        for seed in seeds:
            model = models[(arm, seed)]
            if arm in CONVEX_ARMS:
                minimum_before = minimum_convex_weight(model)
                project_nonnegative_(model)
                minimum_after = minimum_convex_weight(model)
                convex_weight_rows.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "minimum_weight": minimum_after,
                        "nonnegative": minimum_after is not None and minimum_after >= 0.0,
                    }
                )
                projection_rows.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "minimum_before": minimum_before,
                        "minimum_after": minimum_after,
                        "passed": minimum_after is not None and minimum_after >= 0.0,
                    }
                )
            checks = convexity_checks(model, seed, convexity_samples)
            jensen_rows.extend({"arm": arm, "seed": seed, **row} for row in checks["jensen"])
            finite_difference_rows.extend(
                {"arm": arm, "seed": seed, "check_kind": "directional_second_difference", **row}
                for row in checks["finite_difference"]
            )
            finite_difference_rows.extend(
                {"arm": arm, "seed": seed, "check_kind": "gradient_monotonicity", **row}
                for row in checks["gradient"]
            )

    for seed in seeds:
        model = models[(ARM_CONVEX, seed)]
        for start_index, fill in enumerate((0.0, 0.5, 1.0)):
            result = projected_optimize(
                model, torch.full((4, FEATURE_DIM), fill, dtype=torch.float64), steps=120
            )
            optimizer_rows.append(
                {
                    "arm": ARM_CONVEX,
                    "seed": seed,
                    "start": start_index,
                    "start_fill": fill,
                    **result,
                }
            )

    all_ordering_rows: list[JsonDict] = []
    calibration_rows: list[JsonDict] = []
    size_transfer_rows: list[JsonDict] = []
    latency_rows: list[JsonDict] = []
    for arm in ARMS:
        for seed in seeds:
            model = models[(arm, seed)]
            for split in EVALUATION_SPLITS:
                all_ordering_rows.extend(ordering_rows(model, candidates, arm, seed, split))

            calibration_candidates = [row for row in candidates if row.split == "calibration"]
            held_candidates = [row for row in candidates if row.split == "held_out"]
            calibration_energies_by_id = score_candidates(model, calibration_candidates)
            held_energies_by_id = score_candidates(model, held_candidates)
            calibration_energies = [
                calibration_energies_by_id[row.candidate_id] for row in calibration_candidates
            ]
            calibration_labels = [row.is_corrupt for row in calibration_candidates]
            threshold = _threshold(calibration_energies, calibration_labels)
            calibration_accuracy, calibration_brier, calibration_ece = _classification_metrics(
                calibration_energies, calibration_labels, threshold
            )
            held_energies = [held_energies_by_id[row.candidate_id] for row in held_candidates]
            held_labels = [row.is_corrupt for row in held_candidates]
            held_accuracy, held_brier, held_ece = _classification_metrics(
                held_energies, held_labels, threshold
            )
            calibration_rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "threshold": threshold,
                    "calibration_accuracy": calibration_accuracy,
                    "calibration_brier": calibration_brier,
                    "calibration_ece": calibration_ece,
                    "held_out_accuracy": held_accuracy,
                    "held_out_brier": held_brier,
                    "held_out_ece": held_ece,
                    "terminal": True,
                }
            )

            arm_ordering = [
                row for row in all_ordering_rows if row["arm"] == arm and row["seed"] == seed
            ]
            for band, predicate in (
                ("smaller_or_equal_two_variables", lambda count: count <= 2),
                ("larger_than_two_variables", lambda count: count > 2),
            ):
                selected = [row for row in arm_ordering if predicate(row["n_variables"])]
                size_transfer_rows.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "size_band": band,
                        "trained_in_band": band == "smaller_or_equal_two_variables",
                        "pair_count": len(selected),
                        "exact_ordering_accuracy": (
                            sum(row["ordering_credit"] for row in selected) / len(selected)
                            if selected
                            else None
                        ),
                        "terminal": bool(selected),
                        "failure": None if selected else "empty_size_band",
                    }
                )

            latency_candidates = held_candidates
            factors, mask = batch_tensors(latency_candidates)
            repeats = 20
            latency_started = time.perf_counter()
            with torch.no_grad():
                for _ in range(repeats):
                    model(factors, mask)
            elapsed = time.perf_counter() - latency_started
            latency_rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "examples": len(latency_candidates) * repeats,
                    "total_s": elapsed,
                    "microseconds_per_candidate": elapsed
                    * 1_000_000
                    / (len(latency_candidates) * repeats),
                    "terminal": True,
                }
            )

    baseline_rows = [
        {
            "arm": arm,
            "seed": seed,
            "held_out_exact_ordering_accuracy": sum(
                row["ordering_credit"]
                for row in all_ordering_rows
                if row["arm"] == arm and row["seed"] == seed and row["split"] == "held_out"
            )
            / sum(
                1
                for row in all_ordering_rows
                if row["arm"] == arm and row["seed"] == seed and row["split"] == "held_out"
            ),
            "role": "transparent_upper_control" if arm == ARM_HAND else "learned_control",
        }
        for arm in (ARM_LINEAR, ARM_HAND, ARM_MLP)
        for seed in seeds
    ]
    shuffled_label_rows = []
    for seed in seeds:
        normal = training_labels(train_candidates, seed, shuffled=False)
        shuffled = training_labels(train_candidates, seed, shuffled=True)
        shuffled_label_rows.append(
            {
                "arm": ARM_SHUFFLED,
                "seed": seed,
                "label_count": len(shuffled),
                "unchanged_positions": int((normal == shuffled).sum()),
                "shuffled_label_hash": sha256_bytes(canonical_json(shuffled.tolist())),
                "frozen_before_optimizer": True,
                "held_out_exact_ordering_accuracy": next(
                    row["held_out_exact_ordering_accuracy"]
                    for row in [
                        {
                            "held_out_exact_ordering_accuracy": sum(
                                item["ordering_credit"]
                                for item in all_ordering_rows
                                if item["arm"] == ARM_SHUFFLED
                                and item["seed"] == seed
                                and item["split"] == "held_out"
                            )
                            / sum(
                                1
                                for item in all_ordering_rows
                                if item["arm"] == ARM_SHUFFLED
                                and item["seed"] == seed
                                and item["split"] == "held_out"
                            )
                        }
                    ]
                ),
            }
        )

    held_by_key = {
        (row["arm"], row["seed"], row["pair_id"]): row
        for row in all_ordering_rows
        if row["split"] == "held_out"
    }
    paired_metric_rows: list[JsonDict] = []
    confidence_interval_rows: list[JsonDict] = []
    intervals: list[tuple[float | None, float | None]] = []
    for comparison_index, control in enumerate((ARM_MLP, ARM_LINEAR)):
        deltas = []
        for seed in seeds:
            pair_ids = sorted(
                row["pair_id"]
                for row in all_ordering_rows
                if row["arm"] == ARM_CONVEX and row["seed"] == seed and row["split"] == "held_out"
            )
            for pair_id in pair_ids:
                convex_row = held_by_key[(ARM_CONVEX, seed, pair_id)]
                control_row = held_by_key[(control, seed, pair_id)]
                delta = convex_row["ordering_credit"] - control_row["ordering_credit"]
                deltas.append(delta)
                paired_metric_rows.append(
                    {
                        "comparison": f"{ARM_CONVEX}_minus_{control}",
                        "seed": seed,
                        "pair_id": pair_id,
                        "convex_ordering_credit": convex_row["ordering_credit"],
                        "control_ordering_credit": control_row["ordering_credit"],
                        "paired_delta": delta,
                    }
                )
        low, high = bootstrap_ci(deltas, RANDOM_SEED + 100 * comparison_index, bootstrap_samples)
        intervals.append((low, high))
        confidence_interval_rows.append(
            {
                "comparison": f"{ARM_CONVEX}_minus_{control}",
                "paired_unit_count": len(deltas),
                "mean_delta": sum(deltas) / len(deltas) if deltas else None,
                "ci95_lower": low,
                "ci95_upper": high,
                "strictly_above_zero": low is not None and low > 0.0,
                "bootstrap_samples": bootstrap_samples,
            }
        )

    replay_payload = {"schema_version": REPLAY_SCHEMA_VERSION, "models": manifest_models}
    _write_json(Path(replay_manifest_path), replay_payload)
    expected_replay = replay_manifest_from_path(Path(replay_manifest_path))
    child_replay = _fresh_process_replay(root, Path(replay_manifest_path))
    child_by_key = {(row["arm"], row["seed"]): row for row in child_replay}
    fresh_process_replay_rows = [
        {
            "arm": row["arm"],
            "seed": row["seed"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "energy_checksum": row["energy_checksum"],
            "ordering": row["ordering"],
            "replay_matches": child_by_key.get((row["arm"], row["seed"])) == row,
        }
        for row in expected_replay
    ]
    convex_geometry_rows = [
        row for row in jensen_rows + finite_difference_rows if row["arm"] == ARM_CONVEX
    ]
    convexity_passed = bool(convex_geometry_rows) and all(
        row["passed"] for row in convex_geometry_rows
    )
    convexity_passed = convexity_passed and all(
        row["passed"] for row in projection_rows if row["arm"] == ARM_CONVEX
    )
    nonlinear_match = all(
        model_parameter_count(models[(ARM_CONVEX, seed)])
        == model_parameter_count(models[(ARM_MLP, seed)])
        for seed in seeds
    )
    completion_checks = [
        _gate("arm_count", len(ARMS), len(arm_rows)),
        _gate("seed_row_count", len(ARMS) * len(seeds), len(seed_rows)),
        _gate("all_seed_rows_terminal", True, all(row["terminal"] for row in seed_rows)),
        _gate("nonlinear_parameter_match", True, nonlinear_match),
        _gate("label_isolation", True, all(row["passed"] for row in label_isolation_rows)),
        _gate("split_isolation", True, all(not row["crosses_split"] for row in split_rows)),
        _gate(
            "ordering_split_coverage",
            sorted(EVALUATION_SPLITS),
            sorted({row["split"] for row in all_ordering_rows}),
        ),
        _gate(
            "size_band_terminal_count",
            len(ARMS) * len(seeds) * 2,
            sum(row["terminal"] for row in size_transfer_rows),
        ),
        _gate(
            "optimizer_terminal_count",
            len(seeds) * 3,
            sum(row["terminal"] for row in optimizer_rows),
        ),
        _gate("checkpoint_count", len(LEARNED_ARMS) * len(seeds), len(checkpoint_paths)),
        _gate(
            "fresh_process_replay_count",
            len(LEARNED_ARMS) * len(seeds),
            sum(row["replay_matches"] for row in fresh_process_replay_rows),
        ),
    ]
    completion_summary = _summary(completion_checks)
    complete = bool(completion_summary["passed"])
    positive = positive_gate(complete, convexity_passed, intervals)
    verdict_class = "positive" if positive else "null" if complete else "partial"
    honest_verdict = (
        "complete_positive_convex_factor_energy_canary"
        if positive
        else "complete_null_convex_factor_energy_canary"
        if complete
        else "partial_convex_factor_energy_canary"
    )

    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_date": date,
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": time.perf_counter() - started,
        "source_artifact_hashes": hashes,
        "rows": all_ordering_rows,
        "feature_rows": feature_rows,
        "corruption_rows": corruption_rows,
        "split_rows": split_rows,
        "arm_rows": arm_rows,
        "seed_rows": seed_rows,
        "training_rows": training_rows,
        "convex_weight_rows": convex_weight_rows,
        "jensen_rows": jensen_rows,
        "finite_difference_rows": finite_difference_rows,
        "projection_rows": projection_rows,
        "optimizer_rows": optimizer_rows,
        "ordering_rows": all_ordering_rows,
        "calibration_rows": calibration_rows,
        "size_transfer_rows": size_transfer_rows,
        "latency_rows": latency_rows,
        "baseline_rows": baseline_rows,
        "shuffled_label_rows": shuffled_label_rows,
        "paired_metric_rows": paired_metric_rows,
        "confidence_interval_rows": confidence_interval_rows,
        "label_isolation_rows": label_isolation_rows,
        "fresh_process_replay_rows": fresh_process_replay_rows,
        "checkpoint_paths": checkpoint_paths,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "convex_factor_run_complete_score": int(complete),
        "convex_factor_positive_score": int(positive),
        "gate_check_summary": completion_summary,
        "verifier_is_oracle": False,
        "verdict_class": verdict_class,
        "honest_verdict": honest_verdict,
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def replay_manifest_from_path(path: Path) -> list[JsonDict]:
    """Named indirection keeps the parent replay call distinct from its manifest value."""

    return replay_manifest(path)


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash deterministic scientific content while excluding measured timing fields."""

    payload = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "latency_rows", "reproducibility_checksum"}
    }
    return sha256_bytes(canonical_json(payload))


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Reject missing fields, principle gaps, checksum drift, or score/verdict conflicts."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing_artifact_fields:{','.join(missing)}")
    if set(artifact["field_principles"]) != set(REQUIRED_ARTIFACT_FIELDS):
        raise ValueError("field_principles_mismatch")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate_mismatch")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle_mismatch")
    if (
        artifact["convex_factor_positive_score"]
        and not artifact["convex_factor_run_complete_score"]
    ):
        raise ValueError("positive_without_completion")
    expected_class = (
        "blocked"
        if artifact["honest_verdict"] == "blocked_convex_factor_energy_canary"
        else "positive"
        if artifact["convex_factor_positive_score"]
        else "null"
        if artifact["convex_factor_run_complete_score"]
        else "partial"
    )
    if artifact["verdict_class"] != expected_class:
        raise ValueError("verdict_class_score_mismatch")
    if payload_checksum(artifact) != artifact["reproducibility_checksum"]:
        raise ValueError("reproducibility_checksum_mismatch")


def run(
    date: str,
    repo_root: Path | None = None,
    output_path: Path | None = None,
) -> JsonDict:
    """Build, validate, and atomically write the canonical Exp6958 artifact."""

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    output = Path(output_path) if output_path is not None else root / OUTPUT_PATH
    artifact = build_artifact(
        date=date,
        repo_root=root,
        fixture_path=root / FIXTURE_PATH,
        corpus_path=root / CORPUS_PATH,
        checkpoint_dir=root / CHECKPOINT_DIR,
        replay_manifest_path=root / REPLAY_MANIFEST_PATH,
    )
    validate_artifact(artifact)
    _write_json(output, artifact)
    return artifact


def main(
    argv: Sequence[str] | None = None,
) -> int:  # pragma: no cover - exercised by required E2E command.
    """Expose normal execution and the private fresh-process replay surface."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260904")
    parser.add_argument("--replay-manifest", type=Path)
    parser.add_argument("--replay-output", type=Path)
    args = parser.parse_args(argv)
    if args.replay_manifest is not None:
        if args.replay_output is None:
            parser.error("--replay-output is required with --replay-manifest")
        _write_json(args.replay_output, replay_manifest(args.replay_manifest))
        return 0
    artifact = run(args.date)
    print(
        json.dumps(
            {
                field: artifact[field]
                for field in (
                    "convex_factor_run_complete_score",
                    "convex_factor_positive_score",
                    "verdict_class",
                    "honest_verdict",
                )
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - module execution is the replay boundary.
    raise SystemExit(main())
