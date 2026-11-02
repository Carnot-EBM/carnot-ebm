"""Build a bounded ConvergeFlow-inspired feasible-token canary.

The canary uses fixed toy embeddings and exact token sets. It does not train a
predictor. It also does not reproduce the paper's language model. The exact
set supplies the treatment constraint and the validity oracle, so validity is
circular evidence unless distortion and cost also support a separate claim.

Spec refs: REQ-REPORT-6596, SCENARIO-REPORT-6596-FIXTURES,
SCENARIO-REPORT-6596-CONTROLS, SCENARIO-REPORT-6596-ROWS,
SCENARIO-REPORT-6596-EXACT, SCENARIO-REPORT-6596-ROBUSTNESS,
SCENARIO-REPORT-6596-ATTACKS, SCENARIO-REPORT-6596-ATOMIC.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import platform
import shutil
import tempfile
import time
from typing import Any

import numpy as np

from carnot.task_runtime_receipts import canonical_json, sha256_file, sha256_json


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260825"
SCHEMA_VERSION = "carnot.experiment_6596.convergeflow_feasible_token_canary.v1"
INFERENCE_SUBSTRATE = "toy_convex_hull_feasible_token_flow_no_llm"
RESULT_RELATIVE_PATH = Path("results/experiment_6596_convergeflow_feasible_token_canary.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6596_convergeflow_feasible_token_canary.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6596_convergeflow_feasible_token_canary.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")

PROTECTED_RELATIVE_PATHS = (
    Path("research-roadmap.yaml"),
    Path("scripts/research_conductor.py"),
)
SOURCE_MODULE_PATHS = (
    Path("python/carnot/phase3/continuous_ebm.py"),
    Path("python/carnot/samplers/continuous_latent.py"),
    Path("python/carnot/samplers/flow_sampling.py"),
    Path("python/carnot/models/pinet_layer.py"),
    Path("python/carnot/verify/xgrammar_abs_contract_decoder_adapter.py"),
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
)

ARXIV_ABS_URL = "https://arxiv.org/abs/2608.23551"
ARXIV_SOURCE_URL = "https://arxiv.org/e-print/2608.23551v1"
ARXIV_SOURCE_SHA256 = "sha256:9cdd1d29979905614f0843c50b5536395ead6e542debc929c69047e1069464bb"
OFFICIAL_CODE_URL = "https://github.com/Na-Li66/ConvergeFlow"
METHOD_CONTRACT = {
    "source": "arXiv:2608.23551v1",
    "imported_mechanism": (
        "use positive embedding weights multiplied by the exact Gaussian corruption "
        "kernel, so every data prediction is in the convex hull of fixed embeddings"
    ),
    "integration_rule": (
        "first-order data-prediction flow update on a fixed geometric noise schedule"
    ),
    "paper_assumptions_not_proved_by_canary": [
        "positive base weights for every token",
        "Lipschitz log base weights along the trajectory",
        "a terminal time grid that satisfies the paper's refinement condition",
        "fixed distinct token embeddings",
    ],
    "bounded_departures": [
        "the canary has one toy token position rather than a token sequence",
        "the canary uses a fixed analytic predictor rather than a trained network",
        "the treatment hull contains exact feasible tokens rather than the full vocabulary",
    ],
    "non_claims": [
        "not a language model",
        "not language-model training",
        "not an OpenWebText reproduction",
        "not a proof of the paper's theorem for Carnot",
        "not an oracle-distinct validity result",
    ],
}

HELD_SEEDS = (11, 23, 37, 53, 71)
INTEGRATION_STEPS = 32
ENDPOINT_SIGMA = 1e-6
CONVERGENCE_TOLERANCE = 5e-5
BASE_WEIGHT_TEMPERATURE = 0.75
FLOAT_TOLERANCE = 1e-12
RANDOM_SEED = 6596

ARMS = (
    "unconstrained_flow",
    "nearest_token_rounding",
    "convex_hull_predictor_projection",
)
ATTACK_IDS = (
    "feasible_set_leakage",
    "nearest_token_treatment_projection",
    "hidden_endpoint_snap_cost",
    "empty_feasible_set_acceptance",
    "post_outcome_step_tuning",
    "dropped_nonconvergence",
    "one_seed_promotion",
    "aggregate_only_output",
)
PER_UNIT_METRIC_FIELDS = (
    "valid_endpoint",
    "exact_constraint_result",
    "steps",
    "path_length",
    "endpoint_distortion",
    "convergence",
    "failure",
    "wall_time_s",
    "projection_calls",
    "endpoint_snap_operations",
    "charged_work_units",
)

VALIDATION_COMMANDS = (
    ".venv/bin/pytest -n 0 -o addopts= "
    "tests/python/test_experiment_6596_convergeflow_feasible_token_canary.py -q",
    ".venv/bin/coverage run --rcfile=/dev/null --branch "
    "--include=python/carnot/experiment_6596_convergeflow_feasible_token_canary.py "
    "-m pytest -n 0 -o addopts= "
    "tests/python/test_experiment_6596_convergeflow_feasible_token_canary.py -q",
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6596_convergeflow_feasible_token_canary.py "
    "--show-missing --fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/ruff check "
    "python/carnot/experiment_6596_convergeflow_feasible_token_canary.py "
    "tests/python/test_experiment_6596_convergeflow_feasible_token_canary.py",
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6596_convergeflow_feasible_token_canary.py",
    ".venv/bin/python scripts/artifact_convention_audit.py --recent 1 --dry-run",
    ".venv/bin/python scripts/verdict_row_consistency_lint.py --strict "
    "results/experiment_6596_convergeflow_feasible_token_canary.json",
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6596_convergeflow_feasible_token_canary.json",
    ".venv/bin/pytest -n 0 -o addopts= tests/python/test_e2e_clarav.py -q",
)
DEFAULT_TESTS_RUN = tuple(
    {"command": command, "exit_code": None, "duration_s": 0.0, "state": "pending"}
    for command in VALIDATION_COMMANDS
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "gate_check_summary",
    "per_unit_rows",
    "method_source_receipt",
    "embedding_and_constraint_receipts",
    "arm_definition_rows",
    "exact_endpoint_check_rows",
    "robustness_summary",
    "distortion_and_cost_summary",
    "convergeflow_canary_ready_score",
    "attack_rows",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES = {
    "status": (
        "The canary ends with complete fixture evidence or a named numerical or precondition block."
    ),
    "honest_verdict": (
        "The verdict is limited to toy feasible-token flow and cannot claim "
        "language-model reproduction."
    ),
    "verdict_class": (
        "Use the closed enum; exact-set-defined validity is circular_positive at best."
    ),
    "gate_check_summary": (
        "Any block names the source, module, fixture, numerical, or resource check and "
        "observed value."
    ),
    "per_unit_rows": (
        "Every geometry, constraint, error, seed, and arm carries validity, distortion, "
        "convergence, cost, and failure metrics."
    ),
    "method_source_receipt": (
        "The paper mechanism, assumptions, bounded import, and non-claims bind by source."
    ),
    "embedding_and_constraint_receipts": (
        "Fixed embeddings, feasible sets, exact automata, starts, and perturbations bind every row."
    ),
    "arm_definition_rows": (
        "Unconstrained, nearest-token, and convex-hull arms differ only by preregistered "
        "mechanisms."
    ),
    "exact_endpoint_check_rows": (
        "Independent exact feasible-set checks decide endpoint validity."
    ),
    "robustness_summary": "Clean and held predictor-error results remain separate.",
    "distortion_and_cost_summary": (
        "Validity cannot hide path distortion, projection work, steps, or wall time."
    ),
    "convergeflow_canary_ready_score": (
        "This binary field records full fixture and control replay, not a language-model claim."
    ),
    "attack_rows": (
        "Leakage, control contamination, hidden snap, emptiness, tuning, row drop, seed, "
        "and aggregate attacks fail closed."
    ),
    "preconditions_checked": (
        "Sources, modules, embeddings, constraints, seeds, errors, steps, tolerances, and "
        "protected files are explicit."
    ),
    "protected_files_unchanged": (
        "Both protected orchestration files retain their original hashes."
    ),
    "inference_substrate": (
        "The task declares deterministic continuous token-flow fixtures with no LLM."
    ),
    "verifier_is_oracle": (
        "The exact feasible set defines validity, so validity-only gains are circular."
    ),
    "field_provenance": (
        "Every field points to fixture rows, exact checks, and reducer functions."
    ),
    "duration_s": "Monotonic duration exposes skipped perturbations or controls.",
    "tests_run": "Focused numerical and E2E commands include exits and durations.",
    "reproducibility_checksum": "A final content hash protects the canary.",
}

VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}


class EmptyFeasibleSetError(ValueError):
    """Signal that a convex predictor has no legal embedding to use."""


@dataclass(frozen=True)
class ErrorLevel:
    """One preregistered predictor-error magnitude and evaluation role."""

    error_id: str
    magnitude: float
    predictor_condition: str


ERROR_LEVELS = (
    ErrorLevel("clean", 0.0, "clean"),
    ErrorLevel("held_error_0p25", 0.25, "held_perturbed"),
    ErrorLevel("held_error_0p65", 0.65, "held_perturbed"),
)


@dataclass(frozen=True)
class GeometryFixture:
    """One fixed vocabulary geometry and its exact feasible token set."""

    geometry_id: str
    constraint_id: str
    dimension: int
    token_ids: tuple[str, ...]
    embeddings: tuple[tuple[float, ...], ...]
    feasible_token_ids: tuple[str, ...]
    expected_failure: str | None

    def embedding_array(self) -> np.ndarray:
        """Return the frozen embedding rows as a float64 matrix."""

        return np.asarray(self.embeddings, dtype=np.float64)

    def feasible_embeddings(self) -> np.ndarray:
        """Return only exact feasible rows without changing their fixed order."""

        index = {token_id: position for position, token_id in enumerate(self.token_ids)}
        return np.asarray(
            [self.embeddings[index[token_id]] for token_id in self.feasible_token_ids],
            dtype=np.float64,
        ).reshape((-1, self.dimension))


def artifact_checksum(payload: Mapping[str, Any]) -> str:
    """Hash all artifact content except the field that stores the hash."""

    return sha256_json(
        {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    )


def build_fixtures() -> tuple[GeometryFixture, ...]:
    """Return three ordinary geometries and one expected empty-set failure."""

    return (
        GeometryFixture(
            geometry_id="axis_square_2d",
            constraint_id="axis_square_exact_set",
            dimension=2,
            token_ids=("sq_nw", "sq_ne", "sq_se", "sq_sw", "sq_center"),
            embeddings=((-1.0, 1.0), (1.0, 1.0), (1.0, -1.0), (-1.0, -1.0), (0.0, 0.0)),
            feasible_token_ids=("sq_nw", "sq_se", "sq_sw"),
            expected_failure=None,
        ),
        GeometryFixture(
            geometry_id="skew_pentagon_2d",
            constraint_id="skew_pentagon_exact_set",
            dimension=2,
            token_ids=("sk_a", "sk_b", "sk_c", "sk_d", "sk_e"),
            embeddings=((-1.2, 0.1), (-0.3, 1.1), (1.0, 0.7), (1.2, -0.7), (-0.4, -1.0)),
            feasible_token_ids=("sk_a", "sk_c", "sk_e"),
            expected_failure=None,
        ),
        GeometryFixture(
            geometry_id="tetra_center_3d",
            constraint_id="tetra_center_exact_set",
            dimension=3,
            token_ids=("te_a", "te_b", "te_c", "te_d", "te_center"),
            embeddings=(
                (1.0, 1.0, 1.0),
                (1.0, -1.0, -1.0),
                (-1.0, 1.0, -1.0),
                (-1.0, -1.0, 1.0),
                (0.0, 0.0, 0.0),
            ),
            feasible_token_ids=("te_a", "te_b", "te_c"),
            expected_failure=None,
        ),
        GeometryFixture(
            geometry_id="empty_set_control_2d",
            constraint_id="empty_exact_set",
            dimension=2,
            token_ids=("em_a", "em_b", "em_c"),
            embeddings=((-1.0, 0.0), (0.0, 1.0), (1.0, 0.0)),
            feasible_token_ids=(),
            expected_failure="empty_feasible_set",
        ),
    )


def _salt(text: str) -> int:
    return int(sha256_json(text)[-8:], 16)


def _target_token(fixture: GeometryFixture, seed: int) -> str:
    candidates = fixture.feasible_token_ids or fixture.token_ids
    return candidates[(seed + _salt(fixture.geometry_id)) % len(candidates)]


def _embedding_for_token(fixture: GeometryFixture, token_id: str) -> np.ndarray:
    return fixture.embedding_array()[fixture.token_ids.index(token_id)]


def _start_state(fixture: GeometryFixture, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + _salt(fixture.geometry_id))
    return rng.normal(0.0, 0.85, size=fixture.dimension)


def _perturbation(fixture: GeometryFixture, seed: int, error: ErrorLevel) -> np.ndarray:
    if error.magnitude == 0.0:
        return np.zeros(fixture.dimension, dtype=np.float64)
    rng = np.random.default_rng(seed + _salt(f"{fixture.geometry_id}:predictor_error"))
    direction = rng.normal(0.0, 1.0, size=fixture.dimension)
    norm = float(np.linalg.norm(direction))
    if norm <= FLOAT_TOLERANCE:  # pragma: no cover - a normal vector is nonzero almost surely.
        direction[0] = 1.0
        norm = 1.0
    return error.magnitude * direction / norm


def _integration_sigmas() -> np.ndarray:
    return np.geomspace(1.0, ENDPOINT_SIGMA, INTEGRATION_STEPS + 1)


def _exact_automaton_definition(fixture: GeometryFixture) -> JsonDict:
    feasible = set(fixture.feasible_token_ids)
    return {
        "schema": "carnot.exact_single_token_set_dfa.v1",
        "start_state": "start",
        "accept_state": "accept",
        "reject_state": "reject",
        "consumes_exactly_one_token": True,
        "alphabet": list(fixture.token_ids),
        "transitions": [
            {
                "from_state": "start",
                "symbol": token_id,
                "to_state": "accept" if token_id in feasible else "reject",
            }
            for token_id in fixture.token_ids
        ],
    }


def _validate_fixture(fixture: GeometryFixture) -> None:
    if len(set(fixture.token_ids)) != len(fixture.token_ids):
        raise ValueError("token ids must be unique")
    embedding = fixture.embedding_array()
    if embedding.shape != (len(fixture.token_ids), fixture.dimension):
        raise ValueError("embedding shape must match vocabulary and dimension")
    if not set(fixture.feasible_token_ids) <= set(fixture.token_ids):
        raise ValueError("feasible token must belong to the vocabulary")
    if not fixture.feasible_token_ids and fixture.expected_failure != "empty_feasible_set":
        raise ValueError("empty feasible set must declare its expected failure")


def build_embedding_and_constraint_receipts(
    fixtures: Sequence[GeometryFixture],
) -> JsonDict:
    """Freeze embeddings, exact sets, starts, errors, and their row hashes."""

    for fixture in fixtures:
        _validate_fixture(fixture)
    fixture_rows = []
    start_rows = []
    perturbation_rows = []
    for fixture in fixtures:
        automaton = _exact_automaton_definition(fixture)
        fixture_record = json.loads(canonical_json(asdict(fixture)))
        fixture_payload = {
            **fixture_record,
            "embedding_sha256": sha256_json(fixture.embeddings),
            "feasible_set_sha256": sha256_json(fixture.feasible_token_ids),
            "exact_automaton_definition": automaton,
            "exact_automaton_sha256": sha256_json(automaton),
        }
        fixture_payload["fixture_sha256"] = sha256_json(fixture_payload)
        fixture_rows.append(fixture_payload)
        for seed in HELD_SEEDS:
            start = _start_state(fixture, seed)
            start_hash = sha256_json(start.tolist())
            start_rows.append(
                {
                    "geometry_id": fixture.geometry_id,
                    "constraint_id": fixture.constraint_id,
                    "seed": seed,
                    "start": start.tolist(),
                    "start_hash": start_hash,
                    "membership": "held",
                }
            )
            for error in ERROR_LEVELS:
                perturbation = _perturbation(fixture, seed, error)
                perturbation_rows.append(
                    {
                        "geometry_id": fixture.geometry_id,
                        "constraint_id": fixture.constraint_id,
                        "seed": seed,
                        "error_id": error.error_id,
                        "predictor_condition": error.predictor_condition,
                        "error_magnitude": error.magnitude,
                        "observed_error_magnitude": float(np.linalg.norm(perturbation)),
                        "perturbation": perturbation.tolist(),
                        "perturbation_hash": sha256_json(perturbation.tolist()),
                        "start_hash": start_hash,
                        "membership": "held" if error.magnitude else "clean_control",
                    }
                )
    ordinary = [fixture for fixture in fixtures if fixture.expected_failure is None]
    expected_failures = [fixture for fixture in fixtures if fixture.expected_failure]
    return {
        "fixture_count": len(fixtures),
        "ordinary_fixture_count": len(ordinary),
        "expected_failure_fixture_count": len(expected_failures),
        "fixture_rows": fixture_rows,
        "start_rows": start_rows,
        "perturbation_rows": perturbation_rows,
        "fixture_matrix_sha256": sha256_json(fixture_rows),
        "start_matrix_sha256": sha256_json(start_rows),
        "perturbation_matrix_sha256": sha256_json(perturbation_rows),
        "all_fixture_hashes_present": all(
            str(row["fixture_sha256"]).startswith("sha256:") for row in fixture_rows
        ),
        "all_ordinary_sets_nontrivial": all(
            0 < len(fixture.feasible_token_ids) < len(fixture.token_ids) for fixture in ordinary
        ),
        "empty_set_fails_closed": bool(expected_failures)
        and all(
            not fixture.feasible_token_ids and fixture.expected_failure == "empty_feasible_set"
            for fixture in expected_failures
        ),
        "held_seeds": list(HELD_SEEDS),
        "error_levels": [asdict(error) for error in ERROR_LEVELS],
        "integration": {
            "steps": INTEGRATION_STEPS,
            "noise_sigmas": _integration_sigmas().tolist(),
            "grid_sha256": sha256_json(_integration_sigmas().tolist()),
            "endpoint_sigma": ENDPOINT_SIGMA,
        },
        "tolerances": {
            "endpoint_convergence": CONVERGENCE_TOLERANCE,
            "float_comparison": FLOAT_TOLERANCE,
        },
    }


def convex_hull_predictor(
    state: np.ndarray,
    raw_prediction: np.ndarray,
    feasible_embeddings: np.ndarray,
    *,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the positive Gaussian-weighted feasible embedding barycenter.

    The base weights use the same perturbed raw predictor as both controls.
    The Gaussian factor is the bounded import from ConvergeFlow. A tiny floor
    keeps every floating-point weight positive when the kernel is very sharp.
    """

    embeddings = np.asarray(feasible_embeddings, dtype=np.float64)
    if embeddings.ndim != 2:
        raise ValueError("feasible embeddings must be a two-dimensional matrix")
    if embeddings.shape[0] == 0:
        raise EmptyFeasibleSetError("empty feasible set has no convex hull")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    state_array = np.asarray(state, dtype=np.float64)
    raw_array = np.asarray(raw_prediction, dtype=np.float64)
    if state_array.shape != (embeddings.shape[1],) or raw_array.shape != state_array.shape:
        raise ValueError("state, raw predictor, and embedding shape must agree")
    base_log_weight = -np.sum((embeddings - raw_array) ** 2, axis=1) / (
        2.0 * BASE_WEIGHT_TEMPERATURE**2
    )
    alpha = 1.0 - sigma
    kernel_log_weight = -np.sum((state_array - alpha * embeddings) ** 2, axis=1) / (2.0 * sigma**2)
    log_weight = base_log_weight + kernel_log_weight
    shifted = log_weight - float(np.max(log_weight))
    weights = np.exp(shifted)
    weights = np.maximum(weights, np.finfo(np.float64).tiny)
    weights /= float(np.sum(weights))
    return weights @ embeddings, weights


def _nearest_token(fixture: GeometryFixture, endpoint: np.ndarray) -> tuple[str, np.ndarray, float]:
    embedding = fixture.embedding_array()
    distances = np.linalg.norm(embedding - endpoint, axis=1)
    index = int(np.argmin(distances))
    return fixture.token_ids[index], embedding[index].copy(), float(distances[index])


def _exact_endpoint_result(fixture: GeometryFixture, endpoint: np.ndarray) -> JsonDict:
    token_id, token_embedding, distance = _nearest_token(fixture, endpoint)
    converged = distance <= CONVERGENCE_TOLERANCE
    nonempty = bool(fixture.feasible_token_ids)
    accepted = nonempty and converged and token_id in fixture.feasible_token_ids
    return {
        "checker": "exact_single_token_set_membership",
        "constraint_id": fixture.constraint_id,
        "nearest_token_id": token_id,
        "nearest_token_embedding": token_embedding.tolist(),
        "distance_to_nearest_token": distance,
        "convergence_tolerance": CONVERGENCE_TOLERANCE,
        "converged_to_token": converged,
        "feasible_set_nonempty": nonempty,
        "token_in_exact_feasible_set": token_id in fixture.feasible_token_ids,
        "accepted": accepted,
        "verifier_is_oracle": True,
    }


def run_arm(
    fixture: GeometryFixture,
    seed: int,
    error: ErrorLevel,
    arm: str,
) -> JsonDict:
    """Run one fixed integration unit and return all charged diagnostics."""

    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    started = time.perf_counter()
    start = _start_state(fixture, seed)
    target_token_id = _target_token(fixture, seed)
    target_embedding = _embedding_for_token(fixture, target_token_id)
    perturbation = _perturbation(fixture, seed, error)
    raw_prediction = target_embedding + perturbation
    sigmas = _integration_sigmas()
    grid_hash = sha256_json(sigmas.tolist())
    row_id = f"{fixture.geometry_id}:{fixture.constraint_id}:{error.error_id}:{seed}:{arm}"
    if fixture.expected_failure:
        exact = _exact_endpoint_result(fixture, start)
        return {
            "row_id": row_id,
            "geometry_id": fixture.geometry_id,
            "constraint_id": fixture.constraint_id,
            "error_id": error.error_id,
            "error_magnitude": error.magnitude,
            "predictor_condition": error.predictor_condition,
            "seed": seed,
            "arm": arm,
            "expected_fixture_failure": True,
            "target_token_id": target_token_id,
            "start": start.tolist(),
            "start_hash": sha256_json(start.tolist()),
            "perturbation": perturbation.tolist(),
            "perturbation_hash": sha256_json(perturbation.tolist()),
            "raw_predictor": raw_prediction.tolist(),
            "raw_predictor_hash": sha256_json(raw_prediction.tolist()),
            "matched_predictor_error_magnitude": float(np.linalg.norm(perturbation)),
            "integration_grid_hash": grid_hash,
            "continuous_endpoint": start.tolist(),
            "endpoint": start.tolist(),
            "endpoint_hash": sha256_json(start.tolist()),
            "decoded_token_id": exact["nearest_token_id"],
            "valid_endpoint": False,
            "exact_constraint_result": exact,
            "steps": 0,
            "predictor_evaluations": 0,
            "projection_calls": 0,
            "projection_candidate_evaluations": 0,
            "feasible_set_access_count": 0,
            "endpoint_snap_operations": 0,
            "endpoint_snap_candidate_evaluations": 0,
            "continuous_path_length": 0.0,
            "endpoint_snap_distance": 0.0,
            "path_length": 0.0,
            "endpoint_distortion": float(np.linalg.norm(start - target_embedding)),
            "convergence": False,
            "failure": "empty_feasible_set",
            "charged_work_units": 0,
            "wall_time_s": max(time.perf_counter() - started, 1e-9),
        }

    state = start.copy()
    continuous_path_length = 0.0
    projection_calls = 0
    projection_candidates = 0
    feasible_access = 0
    for step in range(INTEGRATION_STEPS):
        sigma = float(sigmas[step])
        sigma_next = float(sigmas[step + 1])
        predictor = raw_prediction
        if arm == "convex_hull_predictor_projection":
            predictor, _ = convex_hull_predictor(
                state,
                raw_prediction,
                fixture.feasible_embeddings(),
                sigma=sigma,
            )
            projection_calls += 1
            projection_candidates += len(fixture.feasible_token_ids)
            feasible_access += 1
        ratio = sigma_next / sigma
        next_state = ratio * state + (1.0 - ratio) * predictor
        continuous_path_length += float(np.linalg.norm(next_state - state))
        state = next_state

    continuous_endpoint = state.copy()
    endpoint = continuous_endpoint.copy()
    snap_operations = 0
    snap_candidates = 0
    snap_distance = 0.0
    if arm == "nearest_token_rounding":
        _, endpoint, snap_distance = _nearest_token(fixture, continuous_endpoint)
        snap_operations = 1
        snap_candidates = len(fixture.token_ids)
    exact = _exact_endpoint_result(fixture, endpoint)
    charged_work = INTEGRATION_STEPS + projection_candidates + snap_candidates
    return {
        "row_id": row_id,
        "geometry_id": fixture.geometry_id,
        "constraint_id": fixture.constraint_id,
        "error_id": error.error_id,
        "error_magnitude": error.magnitude,
        "predictor_condition": error.predictor_condition,
        "seed": seed,
        "arm": arm,
        "expected_fixture_failure": False,
        "target_token_id": target_token_id,
        "start": start.tolist(),
        "start_hash": sha256_json(start.tolist()),
        "perturbation": perturbation.tolist(),
        "perturbation_hash": sha256_json(perturbation.tolist()),
        "raw_predictor": raw_prediction.tolist(),
        "raw_predictor_hash": sha256_json(raw_prediction.tolist()),
        "matched_predictor_error_magnitude": float(np.linalg.norm(perturbation)),
        "integration_grid_hash": grid_hash,
        "continuous_endpoint": continuous_endpoint.tolist(),
        "endpoint": endpoint.tolist(),
        "endpoint_hash": sha256_json(endpoint.tolist()),
        "decoded_token_id": exact["nearest_token_id"],
        "valid_endpoint": exact["accepted"],
        "exact_constraint_result": exact,
        "steps": INTEGRATION_STEPS,
        "predictor_evaluations": INTEGRATION_STEPS,
        "projection_calls": projection_calls,
        "projection_candidate_evaluations": projection_candidates,
        "feasible_set_access_count": feasible_access,
        "endpoint_snap_operations": snap_operations,
        "endpoint_snap_candidate_evaluations": snap_candidates,
        "continuous_path_length": continuous_path_length,
        "endpoint_snap_distance": snap_distance,
        "path_length": continuous_path_length + snap_distance,
        "endpoint_distortion": float(np.linalg.norm(endpoint - target_embedding)),
        "convergence": exact["converged_to_token"],
        "failure": None,
        "charged_work_units": charged_work,
        "wall_time_s": max(time.perf_counter() - started, 1e-9),
    }


def evaluate_rows(fixtures: Sequence[GeometryFixture]) -> list[JsonDict]:
    """Evaluate the complete frozen geometry, error, seed, and arm product."""

    return [
        run_arm(fixture, seed, error, arm)
        for fixture in fixtures
        for error in ERROR_LEVELS
        for seed in HELD_SEEDS
        for arm in ARMS
    ]


def build_exact_endpoint_check_rows(
    rows: Sequence[Mapping[str, Any]],
    fixtures: Sequence[GeometryFixture],
) -> list[JsonDict]:
    """Recompute exact validity from endpoint coordinates, not claimed validity."""

    by_id = {fixture.geometry_id: fixture for fixture in fixtures}
    checks = []
    for row in rows:
        fixture = by_id[str(row["geometry_id"])]
        endpoint = np.asarray(row["endpoint"], dtype=np.float64)
        result = _exact_endpoint_result(fixture, endpoint)
        checks.append(
            {
                "row_id": row["row_id"],
                "geometry_id": fixture.geometry_id,
                "constraint_id": fixture.constraint_id,
                "error_id": row["error_id"],
                "seed": row["seed"],
                "arm": row["arm"],
                "expected_fixture_failure": bool(fixture.expected_failure),
                "endpoint_hash": sha256_json(endpoint.tolist()),
                "feasible_set_sha256": sha256_json(fixture.feasible_token_ids),
                **result,
            }
        )
    return checks


def _condition_groups(
    rows: Sequence[Mapping[str, Any]],
) -> list[tuple[str, str, list[Mapping[str, Any]]]]:
    groups = []
    for condition in ("clean", "held_perturbed"):
        for arm in ARMS:
            selected = [
                row for row in rows if row["predictor_condition"] == condition and row["arm"] == arm
            ]
            groups.append((condition, arm, selected))
    return groups


def build_robustness_summary(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Keep clean and held predictor-error validity outcomes separate."""

    summaries = []
    for condition, arm, selected in _condition_groups(rows):
        ordinary = [row for row in selected if not row["expected_fixture_failure"]]
        error_breakdown = []
        for error in ERROR_LEVELS:
            if error.predictor_condition != condition:
                continue
            subset = [row for row in ordinary if row["error_id"] == error.error_id]
            error_breakdown.append(
                {
                    "error_id": error.error_id,
                    "error_magnitude": error.magnitude,
                    "row_count": len(subset),
                    "validity_rate": (
                        sum(bool(row["valid_endpoint"]) for row in subset) / len(subset)
                        if subset
                        else 0.0
                    ),
                    "convergence_rate": (
                        sum(bool(row["convergence"]) for row in subset) / len(subset)
                        if subset
                        else 0.0
                    ),
                }
            )
        summaries.append(
            {
                "predictor_condition": condition,
                "arm": arm,
                "row_count": len(selected),
                "ordinary_row_count": len(ordinary),
                "expected_failure_count": sum(
                    bool(row["expected_fixture_failure"]) for row in selected
                ),
                "valid_count": sum(bool(row["valid_endpoint"]) for row in ordinary),
                "validity_rate": (
                    sum(bool(row["valid_endpoint"]) for row in ordinary) / len(ordinary)
                    if ordinary
                    else 0.0
                ),
                "convergence_rate": (
                    sum(bool(row["convergence"]) for row in ordinary) / len(ordinary)
                    if ordinary
                    else 0.0
                ),
                "failure_count": sum(row["failure"] is not None for row in selected),
                "error_breakdown": error_breakdown,
            }
        )
    return summaries


def build_distortion_and_cost_summary(
    rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Reduce distortion and charged cost without removing failed fixtures."""

    summaries = []
    for condition, arm, selected in _condition_groups(rows):
        summaries.append(
            {
                "predictor_condition": condition,
                "arm": arm,
                "row_count": len(selected),
                "mean_endpoint_distortion": (
                    float(np.mean([row["endpoint_distortion"] for row in selected]))
                    if selected
                    else None
                ),
                "mean_path_length": (
                    float(np.mean([row["path_length"] for row in selected])) if selected else None
                ),
                "total_steps": sum(int(row["steps"]) for row in selected),
                "total_predictor_evaluations": sum(
                    int(row["predictor_evaluations"]) for row in selected
                ),
                "total_projection_calls": sum(int(row["projection_calls"]) for row in selected),
                "total_projection_candidate_evaluations": sum(
                    int(row["projection_candidate_evaluations"]) for row in selected
                ),
                "total_endpoint_snap_operations": sum(
                    int(row["endpoint_snap_operations"]) for row in selected
                ),
                "total_endpoint_snap_candidate_evaluations": sum(
                    int(row["endpoint_snap_candidate_evaluations"]) for row in selected
                ),
                "total_charged_work_units": sum(int(row["charged_work_units"]) for row in selected),
                "failure_count": sum(row["failure"] is not None for row in selected),
                "wall_time_s": float(sum(float(row["wall_time_s"]) for row in selected)),
            }
        )
    return summaries


def build_arm_definition_rows() -> list[JsonDict]:
    """Declare the only preregistered differences among the three arms."""

    common = {
        "integration_steps": INTEGRATION_STEPS,
        "integration_grid_sha256": sha256_json(_integration_sigmas().tolist()),
        "raw_predictor": "target embedding plus the frozen matched perturbation",
        "starting_state": "identical within geometry and seed across errors and arms",
    }
    return [
        {
            "arm": "unconstrained_flow",
            **common,
            "predictor_control": "none",
            "feasible_set_access": "none",
            "endpoint_rule": "continuous endpoint; no snap",
            "charged_extra_work": "none",
        },
        {
            "arm": "nearest_token_rounding",
            **common,
            "predictor_control": "none",
            "feasible_set_access": "none",
            "endpoint_rule": "one nearest-neighbor snap over the full vocabulary",
            "charged_extra_work": "all full-vocabulary endpoint distance evaluations",
        },
        {
            "arm": "convex_hull_predictor_projection",
            **common,
            "predictor_control": "positive Gaussian-weighted feasible embedding barycenter",
            "feasible_set_access": "once per predictor evaluation",
            "endpoint_rule": "continuous endpoint; no snap",
            "charged_extra_work": "every feasible embedding weight evaluation",
        },
    ]


def build_attack_rows() -> list[JsonDict]:
    """Record the detector that makes each preregistered shortcut fail closed."""

    detectors = {
        "feasible_set_leakage": "control rows require feasible_set_access_count=0",
        "nearest_token_treatment_projection": "nearest rows require projection_calls=0",
        "hidden_endpoint_snap_cost": (
            "nearest path equals continuous path plus snap and full vocabulary work is charged"
        ),
        "empty_feasible_set_acceptance": (
            "every empty-set row and independent exact check must reject"
        ),
        "post_outcome_step_tuning": "every ordinary row uses the frozen integration step count",
        "dropped_nonconvergence": "the exact geometry-error-seed-arm key product must match",
        "one_seed_promotion": "the exact key product includes all five held seeds",
        "aggregate_only_output": "row reducers and independent exact checks must recompute",
    }
    return [
        {
            "attack_id": attack_id,
            "detector": detectors[attack_id],
            "detected": True,
            "failed_closed": True,
        }
        for attack_id in ATTACK_IDS
    ]


def _expected_unit_keys() -> set[tuple[str, str, str, int, str]]:
    return {
        (fixture.geometry_id, fixture.constraint_id, error.error_id, seed, arm)
        for fixture in build_fixtures()
        for error in ERROR_LEVELS
        for seed in HELD_SEEDS
        for arm in ARMS
    }


def _row_keys(rows: Sequence[Mapping[str, Any]]) -> set[tuple[str, str, str, int, str]]:
    return {
        (
            str(row["geometry_id"]),
            str(row["constraint_id"]),
            str(row["error_id"]),
            int(row["seed"]),
            str(row["arm"]),
        )
        for row in rows
    }


def compute_ready_score(
    rows: Sequence[Mapping[str, Any]],
    exact_checks: Sequence[Mapping[str, Any]],
) -> float:
    """Return one only for complete replay, including expected failures."""

    if len(rows) != len(_expected_unit_keys()) or _row_keys(rows) != _expected_unit_keys():
        return 0.0
    checks_by_id = {str(check["row_id"]): check for check in exact_checks}
    if len(checks_by_id) != len(rows):
        return 0.0
    for row in rows:
        check = checks_by_id.get(str(row["row_id"]))
        if check is None or bool(check["accepted"]) != bool(row["valid_endpoint"]):
            return 0.0
        if row["expected_fixture_failure"]:
            if row["failure"] != "empty_feasible_set" or row["valid_endpoint"]:
                return 0.0
            continue
        if row["failure"] is not None or int(row["steps"]) != INTEGRATION_STEPS:
            return 0.0
        if row["arm"] in {"unconstrained_flow", "nearest_token_rounding"}:
            if int(row["feasible_set_access_count"]) != 0:
                return 0.0
        if row["arm"] == "nearest_token_rounding" and (
            int(row["projection_calls"]) != 0 or int(row["endpoint_snap_operations"]) != 1
        ):
            return 0.0
        if row["arm"] == "convex_hull_predictor_projection" and (
            int(row["projection_calls"]) != INTEGRATION_STEPS
            or int(row["feasible_set_access_count"]) != INTEGRATION_STEPS
        ):
            return 0.0
    return 1.0


def _protected_hashes(repo_root: Path) -> dict[str, str | None]:
    return {path.as_posix(): sha256_file(repo_root / path) for path in PROTECTED_RELATIVE_PATHS}


def _protected_receipt(
    before: Mapping[str, str | None], after: Mapping[str, str | None]
) -> JsonDict:
    rows = [
        {
            "path": path,
            "before_sha256": before.get(path),
            "after_sha256": after.get(path),
            "unchanged": before.get(path) is not None and before.get(path) == after.get(path),
        }
        for path in sorted(set(before) | set(after))
    ]
    return {"rows": rows, "all_unchanged": all(row["unchanged"] for row in rows)}


def resource_receipt(repo_root: Path) -> JsonDict:
    """Record CPU, RAM, and disk values used by this CPU-only canary."""

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        total_pages = os.sysconf("SC_PHYS_PAGES")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
        ram_total = int(page_size * total_pages)
        ram_available = int(page_size * available_pages)
    except (OSError, ValueError):
        ram_total = None
        ram_available = None
    disk = shutil.disk_usage(repo_root)
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "logical_cpu_count": os.cpu_count(),
        "ram_total_bytes": ram_total,
        "ram_available_bytes": ram_available,
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
        "requires_gpu": False,
        "llm_calls": 0,
    }


def _method_source_receipt() -> JsonDict:
    return {
        "paper": "ConvergeFlow: Language Flow with Provable Convergence to Token Embeddings",
        "authors": ["Na Li", "Yuchen Jiao", "Changxiao Cai", "Gen Li"],
        "version": "arXiv:2608.23551v1",
        "submitted_utc": "2026-08-24T17:54:14Z",
        "abstract_url": ARXIV_ABS_URL,
        "source_url": ARXIV_SOURCE_URL,
        "source_sha256": ARXIV_SOURCE_SHA256,
        "source_bytes": 202808,
        "equation_refs": ["Eq. 21", "Eq. 22", "Eq. 26", "Theorem 1"],
        "method_contract": dict(METHOD_CONTRACT),
        "method_contract_sha256": sha256_json(METHOD_CONTRACT),
        "official_code_check": {
            "url": OFFICIAL_CODE_URL,
            "checked_on": RUN_DATE,
            "available": False,
            "observed": "HTTP 404 from the paper-linked repository URL",
            "code_imported": False,
        },
    }


def _preconditions(
    repo_root: Path,
    fixtures: Sequence[GeometryFixture],
    receipts: Mapping[str, Any],
    overrides: Mapping[str, bool] | None,
) -> list[JsonDict]:
    module_hashes = {path.as_posix(): sha256_file(repo_root / path) for path in SOURCE_MODULE_PATHS}
    protected = _protected_hashes(repo_root)
    checks: list[tuple[str, bool, Any, Any]] = [
        (
            "method_source_hash",
            len(ARXIV_SOURCE_SHA256) == 71 and ARXIV_SOURCE_SHA256.startswith("sha256:"),
            "SHA-256 receipt for arXiv:2608.23551v1",
            ARXIV_SOURCE_SHA256,
        ),
        (
            "source_modules",
            all(module_hashes.values()),
            "all source module hashes present",
            module_hashes,
        ),
        (
            "embedding_geometries",
            int(receipts["ordinary_fixture_count"]) >= 3,
            ">=3 ordinary fixed geometries",
            receipts["ordinary_fixture_count"],
        ),
        (
            "nontrivial_feasible_sets",
            bool(receipts["all_ordinary_sets_nontrivial"]),
            True,
            receipts["all_ordinary_sets_nontrivial"],
        ),
        (
            "empty_set_failure_fixture",
            bool(receipts["empty_set_fails_closed"]),
            True,
            receipts["empty_set_fails_closed"],
        ),
        ("held_seed_count", len(HELD_SEEDS) >= 5, ">=5", len(HELD_SEEDS)),
        (
            "predictor_error_levels",
            len(ERROR_LEVELS) >= 3 and ERROR_LEVELS[0].magnitude == 0.0,
            "one clean and at least two held errors",
            [asdict(error) for error in ERROR_LEVELS],
        ),
        (
            "integration_budget",
            INTEGRATION_STEPS > 0 and len(_integration_sigmas()) == INTEGRATION_STEPS + 1,
            INTEGRATION_STEPS,
            len(_integration_sigmas()) - 1,
        ),
        ("cpu_resources", (os.cpu_count() or 0) >= 1, ">=1 logical CPU", os.cpu_count()),
        ("protected_files_present", all(protected.values()), "both present", protected),
        (
            "fixture_count",
            len(fixtures) == int(receipts["fixture_count"]),
            len(fixtures),
            receipts["fixture_count"],
        ),
    ]
    override_map = dict(overrides or {})
    rows = []
    for check_id, passed, expected, observed in checks:
        if check_id in override_map:
            passed = bool(override_map[check_id])
            observed = bool(override_map[check_id])
        rows.append(
            {
                "check_id": check_id,
                "expected_value": expected,
                "observed_value": observed,
                "passed": bool(passed),
            }
        )
    rows.append(
        {
            "check_id": "resources",
            "expected_value": "deterministic CPU execution with local RAM and disk receipts",
            "observed_value": resource_receipt(repo_root),
            "passed": True,
        }
    )
    return rows


def _gate_summary(preconditions: Sequence[Mapping[str, Any]]) -> JsonDict:
    failed = [dict(row) for row in preconditions if not row["passed"]]
    return {
        "blocked": bool(failed),
        "all_preconditions_passed": not failed,
        "failed_checks": failed,
        "principle": FIELD_PRINCIPLES["gate_check_summary"],
    }


def _field_provenance(repo_root: Path) -> dict[str, JsonDict]:
    sources = {
        "module": MODULE_RELATIVE_PATH.as_posix(),
        "module_sha256": sha256_file(repo_root / MODULE_RELATIVE_PATH),
        "spec": SPEC_RELATIVE_PATH.as_posix(),
        "spec_sha256": sha256_file(repo_root / SPEC_RELATIVE_PATH),
    }
    origins = {
        "status": "preconditions_checked and compute_ready_score",
        "honest_verdict": "oracle boundary and complete fixture reducers",
        "verdict_class": "oracle-bound verdict reducer",
        "gate_check_summary": "preconditions_checked reducer",
        "per_unit_rows": "evaluate_rows frozen Cartesian product",
        "method_source_receipt": "arXiv source and bounded METHOD_CONTRACT",
        "embedding_and_constraint_receipts": "frozen fixture, start, and error rows",
        "arm_definition_rows": "build_arm_definition_rows",
        "exact_endpoint_check_rows": "independent endpoint and exact-set replay",
        "robustness_summary": "build_robustness_summary(per_unit_rows)",
        "distortion_and_cost_summary": "build_distortion_and_cost_summary(per_unit_rows)",
        "convergeflow_canary_ready_score": "compute_ready_score(rows, exact checks)",
        "attack_rows": "required attack detector matrix",
        "preconditions_checked": "source, module, fixture, seed, error, grid, resource checks",
        "protected_files_unchanged": "before and after SHA-256 receipts",
        "inference_substrate": "task constant",
        "verifier_is_oracle": "exact feasible-set authority boundary",
        "field_provenance": "required field provenance reducer",
        "duration_s": "monotonic CLI or supplied focused-test duration",
        "tests_run": "task validation command receipts",
        "reproducibility_checksum": "artifact_checksum excluding itself",
    }
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "satisfied_by": origins[field],
            **sources,
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def build_report(
    repo_root: Path = REPO_ROOT,
    *,
    date: str = RUN_DATE,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    precondition_overrides: Mapping[str, bool] | None = None,
) -> JsonDict:
    """Build one terminal artifact from the complete frozen fixture matrix."""

    started = time.perf_counter()
    protected_before = _protected_hashes(repo_root)
    fixtures = build_fixtures()
    receipts = build_embedding_and_constraint_receipts(fixtures)
    preconditions = _preconditions(repo_root, fixtures, receipts, precondition_overrides)
    payload: JsonDict = {
        "schema": SCHEMA_VERSION,
        "experiment_id": 6596,
        "run_date": date,
        "spec_refs": [
            "REQ-REPORT-6596",
            "SCENARIO-REPORT-6596-FIXTURES",
            "SCENARIO-REPORT-6596-CONTROLS",
            "SCENARIO-REPORT-6596-ROWS",
            "SCENARIO-REPORT-6596-EXACT",
            "SCENARIO-REPORT-6596-ROBUSTNESS",
            "SCENARIO-REPORT-6596-ATTACKS",
            "SCENARIO-REPORT-6596-ATOMIC",
        ],
        "method_source_receipt": _method_source_receipt(),
        "embedding_and_constraint_receipts": receipts,
        "arm_definition_rows": build_arm_definition_rows(),
        "preconditions_checked": preconditions,
        "gate_check_summary": _gate_summary(preconditions),
        "protected_files_unchanged": _protected_receipt(
            protected_before, _protected_hashes(repo_root)
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": _field_provenance(repo_root),
        "held_seeds": list(HELD_SEEDS),
        "error_levels": [asdict(error) for error in ERROR_LEVELS],
        "integration_steps": INTEGRATION_STEPS,
        "tolerances": {
            "endpoint_convergence": CONVERGENCE_TOLERANCE,
            "float_comparison": FLOAT_TOLERANCE,
        },
        "resources": resource_receipt(repo_root),
        "random_seed": RANDOM_SEED,
        "duration_s": float(duration_s) if duration_s is not None else 0.0,
        "tests_run": [dict(row) for row in (tests_run or DEFAULT_TESTS_RUN)],
        "attack_rows": build_attack_rows(),
    }
    if any(not row["passed"] for row in preconditions):
        payload.update(
            {
                "status": "blocked_precondition",
                "honest_verdict": (
                    "blocked_precondition: toy feasible-token canary did not start because a "
                    "named source, module, fixture, numerical, or resource check failed"
                ),
                "verdict_class": "blocked",
                "per_unit_rows": [],
                "exact_endpoint_check_rows": [],
                "robustness_summary": [],
                "distortion_and_cost_summary": [],
                "convergeflow_canary_ready_score": 0.0,
            }
        )
    else:
        rows = evaluate_rows(fixtures)
        exact_checks = build_exact_endpoint_check_rows(rows, fixtures)
        ready = compute_ready_score(rows, exact_checks)
        has_oracle_success = any(
            row["arm"] == "convex_hull_predictor_projection"
            and not row["expected_fixture_failure"]
            and row["valid_endpoint"]
            for row in rows
        )
        verdict_class = "circular_positive" if ready == 1.0 and has_oracle_success else "null"
        payload.update(
            {
                "status": "complete_fixture_evidence",
                "honest_verdict": (
                    "complete: the toy convex-hull feasible-token flow canary replayed all "
                    "fixtures, controls, held errors, exact checks, distortion, and charged "
                    "cost; exact-set validity is circular and this does not reproduce a "
                    "language model"
                    if ready == 1.0
                    else "complete: the toy feasible-token flow canary returned null because "
                    "one or more frozen fixture, control, error, or exact replay checks failed; "
                    "this does not reproduce a language model"
                ),
                "verdict_class": verdict_class,
                "per_unit_rows": rows,
                "exact_endpoint_check_rows": exact_checks,
                "robustness_summary": build_robustness_summary(rows),
                "distortion_and_cost_summary": build_distortion_and_cost_summary(rows),
                "convergeflow_canary_ready_score": ready,
            }
        )
    payload["duration_s"] = (
        float(duration_s) if duration_s is not None else max(time.perf_counter() - started, 1e-9)
    )
    payload["protected_files_unchanged"] = _protected_receipt(
        protected_before, _protected_hashes(repo_root)
    )
    payload["reproducibility_checksum"] = artifact_checksum(payload)
    return payload


def validate_report(payload: Mapping[str, Any], repo_root: Path = REPO_ROOT) -> list[str]:
    """Validate required fields and recompute every decision-bearing reducer."""

    errors = [
        f"missing required field: {field}"
        for field in REQUIRED_ARTIFACT_FIELDS
        if field not in payload
    ]
    if errors:
        return errors
    if payload["inference_substrate"] != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if payload["verifier_is_oracle"] is not True:
        errors.append("verifier_is_oracle must be true")
    if payload["verdict_class"] not in VERDICT_CLASSES:
        errors.append("verdict_class outside closed enum")
    if set(payload["field_provenance"]) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance coverage mismatch")
    protected_receipt = payload["protected_files_unchanged"]
    if not protected_receipt.get("all_unchanged"):
        errors.append("protected files changed")
    current_protected = _protected_hashes(repo_root)
    for row in protected_receipt.get("rows", []):
        if current_protected.get(row["path"]) != row["after_sha256"]:
            errors.append("protected file current hash mismatch")
            break
    attacks = payload["attack_rows"]
    if {row.get("attack_id") for row in attacks} != set(ATTACK_IDS) or not all(
        row.get("detected") and row.get("failed_closed") for row in attacks
    ):
        errors.append("attack_rows incomplete")

    blocked = payload["verdict_class"] == "blocked"
    if blocked:
        if not payload["gate_check_summary"].get("failed_checks"):
            errors.append("blocked report lacks failed gate detail")
        if payload["per_unit_rows"]:
            errors.append("blocked report fabricated per_unit_rows")
        if payload["convergeflow_canary_ready_score"] != 0.0:
            errors.append("blocked report has nonzero ready score")
    else:
        rows = payload["per_unit_rows"]
        keys = _row_keys(rows)
        if keys != _expected_unit_keys() or len(rows) != len(keys):
            errors.append("per_unit_rows key coverage mismatch")
        if any(
            row["arm"] == "nearest_token_rounding"
            and (row["projection_calls"] != 0 or row["feasible_set_access_count"] != 0)
            for row in rows
        ):
            errors.append("nearest-token control contaminated")
        if any(
            row["arm"] == "unconstrained_flow" and row["feasible_set_access_count"] != 0
            for row in rows
        ):
            errors.append("feasible-set leakage into unconstrained control")
        if any(
            row["arm"] == "nearest_token_rounding"
            and not row["expected_fixture_failure"]
            and (
                row["endpoint_snap_operations"] != 1
                or row["endpoint_snap_candidate_evaluations"]
                != len(
                    next(
                        fixture.token_ids
                        for fixture in build_fixtures()
                        if fixture.geometry_id == row["geometry_id"]
                    )
                )
                or not math.isclose(
                    float(row["path_length"]),
                    float(row["continuous_path_length"]) + float(row["endpoint_snap_distance"]),
                    rel_tol=0.0,
                    abs_tol=FLOAT_TOLERANCE,
                )
            )
            for row in rows
        ):
            errors.append("endpoint snap cost hidden")
        if any(
            row["expected_fixture_failure"]
            and (row["valid_endpoint"] or row["failure"] != "empty_feasible_set")
            for row in rows
        ):
            errors.append("empty feasible set accepted")
        if any(
            not row["expected_fixture_failure"] and row["steps"] != INTEGRATION_STEPS
            for row in rows
        ):
            errors.append("post-outcome integration tuning")
        fixtures = build_fixtures()
        exact_checks = build_exact_endpoint_check_rows(rows, fixtures)
        if payload["exact_endpoint_check_rows"] != exact_checks:
            errors.append("exact_endpoint_check_rows mismatch")
        robustness = build_robustness_summary(rows)
        if payload["robustness_summary"] != robustness:
            errors.append("robustness_summary mismatch")
        cost = build_distortion_and_cost_summary(rows)
        if payload["distortion_and_cost_summary"] != cost:
            errors.append("distortion_and_cost_summary mismatch")
        ready = compute_ready_score(rows, exact_checks)
        if payload["convergeflow_canary_ready_score"] != ready:
            errors.append("ready score mismatch")
        has_oracle_success = any(
            row["arm"] == "convex_hull_predictor_projection"
            and not row["expected_fixture_failure"]
            and row["valid_endpoint"]
            for row in rows
        )
        expected_class = "circular_positive" if ready == 1.0 and has_oracle_success else "null"
        if payload["verdict_class"] == "positive":
            errors.append("oracle-defined validity cannot be positive")
        elif payload["verdict_class"] != expected_class:
            errors.append("verdict_class mismatch")
    if payload["reproducibility_checksum"] != artifact_checksum(payload):
        errors.append("reproducibility_checksum mismatch")
    return errors


def atomic_write_report(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
) -> JsonDict:
    """Validate, file-sync, atomically replace, and directory-sync one JSON."""

    errors = validate_report(payload, repo_root)
    if errors:
        raise ValueError("; ".join(errors))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():  # pragma: no cover - replacement failure cleanup.
            temporary.unlink()
    return {
        "path": str(target),
        "file_fsync": True,
        "atomic_replace": True,
        "directory_fsync": True,
        "output_sha256": sha256_file(target),
    }


def existing_test_receipts(path: str | Path) -> list[JsonDict]:
    """Preserve measured validation receipts when the terminal command reruns."""

    source = Path(path)
    if not source.is_file():
        return [dict(row) for row in DEFAULT_TESTS_RUN]
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [dict(row) for row in DEFAULT_TESTS_RUN]
    rows = payload.get("tests_run")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        return [dict(row) for row in DEFAULT_TESTS_RUN]
    return [dict(row) for row in rows]


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    output = REPO_ROOT / RESULT_RELATIVE_PATH
    if args.validate:
        if not output.is_file():
            print(f"missing: {output}")
            return 1
        payload = json.loads(output.read_text(encoding="utf-8"))
        errors = validate_report(payload)
        if errors:
            print("\n".join(errors))
            return 1
        print(f"valid: {output}")
        return 0
    started = time.perf_counter()
    report = build_report(
        REPO_ROOT,
        date=args.date,
        tests_run=existing_test_receipts(output),
    )
    report["duration_s"] = max(time.perf_counter() - started, 1e-9)
    report["reproducibility_checksum"] = artifact_checksum(report)
    receipt = atomic_write_report(output, report)
    print(
        canonical_json(
            {"status": report["status"], "verdict_class": report["verdict_class"], **receipt}
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
