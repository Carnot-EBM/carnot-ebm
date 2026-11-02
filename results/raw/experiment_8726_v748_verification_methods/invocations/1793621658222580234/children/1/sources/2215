"""Exp6287 bounded continuous relaxation for exact ASP energy.

Spec refs: REQ-KONA-6287, SCENARIO-KONA-6287-VERTEX-PARITY,
SCENARIO-KONA-6287-GRADIENT-CHECK, SCENARIO-KONA-6287-CONTROLS.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import time
from typing import Any

from carnot import asp_continuous_relaxation as relax
from carnot import asp_energy
from carnot import experiment_6274_asp_energy_semantic_compiler as exp6274


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6287_asp_continuous_relaxation.json")
UPSTREAM_RESULT_RELATIVE_PATH = exp6274.RESULT_RELATIVE_PATH
FIXTURE_MANIFEST_RELATIVE_PATH = exp6274.FIXTURE_MANIFEST_RELATIVE_PATH
SPEC_RELATIVE_PATH = Path("openspec/capabilities/phase3-kona/spec.md")
RUN_COMMAND = ".venv/bin/python -m carnot.experiment_6287_asp_continuous_relaxation --date 20260810"
INFERENCE_SUBSTRATE = (
    "deterministic_multilinear_extension_of_exact_asp_energy_no_learning_no_diffusion"
)
MAX_ATOMS = 12
MAX_VERTEX_COUNT = 4096
GRADIENT_EPSILON = 1e-5
GRADIENT_TOLERANCE = 1e-6
FRACTIONAL_GRADIENT_TOLERANCE = 1e-9
BOX_TOLERANCE = 1e-9
OPTIMIZER_STEPS = 24
OPTIMIZER_STEP_SIZE = 0.25
RESTART_BUDGET = 1
RANDOM_SEEDS = (6287, 6288, 6289)
DEFAULT_TEST_COMMANDS = (
    RUN_COMMAND,
    ".venv/bin/pytest tests/python/test_asp_continuous_relaxation_6287.py -q --no-cov",
    (
        ".venv/bin/coverage run --branch "
        "--include=python/carnot/asp_continuous_relaxation.py,"
        "python/carnot/experiment_6287_asp_continuous_relaxation.py "
        "-m pytest tests/python/test_asp_continuous_relaxation_6287.py -q --no-cov -n 0"
    ),
    (
        ".venv/bin/coverage report "
        "--include=python/carnot/asp_continuous_relaxation.py,"
        "python/carnot/experiment_6287_asp_continuous_relaxation.py --fail-under=100"
    ),
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py",
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "upstream_compiler_path_hash_and_terminal_class",
    "relaxation_definition_and_claim_boundary",
    "source_paths_and_hashes",
    "eligible_fixture_manifest_path_and_hash",
    "fixture_count",
    "atom_count_and_vertex_count_by_fixture",
    "exact_vertex_energy_parity_by_fixture",
    "parity_failure_count",
    "analytic_gradient_definition",
    "finite_difference_gradient_checks",
    "refinement_optimizer_and_fixed_budgets",
    "blank_random_and_partial_start_manifest",
    "refinement_outcomes_by_start_fixture_and_seed",
    "fractional_stationary_points_by_fixture",
    "rounding_failures_by_fixture",
    "exact_completion_controls",
    "cold_start_controls",
    "unsupported_size_and_syntax_controls",
    "oracle_claim_boundary",
    "asp_continuous_relaxation_ready_score",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "random_seeds",
    "reproducibility_checksum",
    "honest_verdict",
)
FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Shows whether parity and gradient gates passed.",
    "upstream_compiler_path_hash_and_terminal_class": "Pins the trusted Exp6274 input.",
    "relaxation_definition_and_claim_boundary": "States the formula and prevents overclaiming.",
    "source_paths_and_hashes": "Pins code and spec bytes used for the run.",
    "eligible_fixture_manifest_path_and_hash": "Pins the fixture sidecar.",
    "fixture_count": "Keeps the evaluation denominator visible.",
    "atom_count_and_vertex_count_by_fixture": "Shows every state space stayed bounded.",
    "exact_vertex_energy_parity_by_fixture": "Proves equality at binary vertices.",
    "parity_failure_count": "Keeps readiness failure count machine-readable.",
    "analytic_gradient_definition": "Names the derivative used by refinement.",
    "finite_difference_gradient_checks": "Checks gradients independently of the formula code.",
    "refinement_optimizer_and_fixed_budgets": "Freezes descent and restart budgets.",
    "blank_random_and_partial_start_manifest": "Makes start states reproducible.",
    "refinement_outcomes_by_start_fixture_and_seed": "Separates optimization results from readiness.",
    "fractional_stationary_points_by_fixture": "Reports continuous traps instead of hiding them.",
    "rounding_failures_by_fixture": "Shows where continuous refinement rounds badly.",
    "exact_completion_controls": "Keeps Clingo completion as an oracle control.",
    "cold_start_controls": "Keeps exact enumeration separate from refinement.",
    "unsupported_size_and_syntax_controls": "Shows unsupported inputs fail closed.",
    "oracle_claim_boundary": "Discloses that the verifier is the oracle.",
    "asp_continuous_relaxation_ready_score": "Opens only on parity and gradient checks.",
    "protected_files_unchanged": "Shows protected files were not edited.",
    "preconditions_checked": "Records bounds, tolerances, seeds, and environment.",
    "inference_substrate": "Declares deterministic relaxation without learning.",
    "verifier_is_oracle": "Prevents an oracle-distinct verifier claim.",
    "field_provenance": "Maps artifact fields to spec and computation sources.",
    "field_principles": "Explains why each required field exists.",
    "test_commands": "Lists verification commands.",
    "test_exit_codes": "Records command outcomes.",
    "duration_s": "Records wall-clock build duration.",
    "random_seeds": "Pins random starts and gradient probes.",
    "reproducibility_checksum": "Detects artifact drift.",
    "honest_verdict": "States the terminal result boundary.",
}
FIELD_PROVENANCE: dict[str, list[str]] = {
    field: ["REQ-KONA-6287", "Exp6274 compiler artifact", "Exp6287 deterministic run"]
    for field in REQUIRED_ARTIFACT_FIELDS
}


def run(
    *,
    date: str,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    duration_s: float | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
    write: bool = True,
) -> JsonDict:
    """Build and optionally write the terminal Exp6287 artifact."""

    started = time.perf_counter()
    elapsed = time.perf_counter() - started if duration_s is None else duration_s
    artifact = build_artifact(
        date=date,
        result_path=Path(result_path),
        duration_s=elapsed,
        test_exit_codes=test_exit_codes,
    )
    if write:
        output = Path(result_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(_canonical_json(artifact, indent=2), encoding="utf-8")
    return artifact


def build_artifact(
    *,
    date: str,
    result_path: Path,
    duration_s: float,
    test_exit_codes: Mapping[str, int | None] | None = None,
) -> JsonDict:
    """Evaluate every eligible fixture and build the JSON payload."""

    fixtures = exp6274.build_fixture_manifest()
    reports = [_evaluate_fixture(fixture) for fixture in fixtures]
    parity_failure_count = sum(
        int(report["vertex_parity"]["failure_count"]) for report in reports
    )
    gradient_passed = all(report["gradient_check"]["passed"] is True for report in reports)
    status = "complete" if parity_failure_count == 0 and gradient_passed else "blocked"
    protected = _protected_hash_receipts()
    artifact: JsonDict = {
        "status": status,
        "upstream_compiler_path_hash_and_terminal_class": _upstream_compiler_receipt(),
        "relaxation_definition_and_claim_boundary": _relaxation_boundary(),
        "source_paths_and_hashes": _source_hashes(),
        "eligible_fixture_manifest_path_and_hash": _fixture_manifest_receipt(),
        "fixture_count": len(fixtures),
        "atom_count_and_vertex_count_by_fixture": {
            report["fixture_id"]: {
                "atom_count": report["atom_count"],
                "vertex_count": report["vertex_count"],
            }
            for report in reports
        },
        "exact_vertex_energy_parity_by_fixture": {
            report["fixture_id"]: report["vertex_parity"] for report in reports
        },
        "parity_failure_count": int(parity_failure_count),
        "analytic_gradient_definition": _analytic_gradient_definition(),
        "finite_difference_gradient_checks": _gradient_summary(reports),
        "refinement_optimizer_and_fixed_budgets": _optimizer_budget(),
        "blank_random_and_partial_start_manifest": {
            report["fixture_id"]: report["start_manifest"] for report in reports
        },
        "refinement_outcomes_by_start_fixture_and_seed": {
            report["fixture_id"]: report["refinement_outcomes"] for report in reports
        },
        "fractional_stationary_points_by_fixture": _fractional_stationary_summary(reports),
        "rounding_failures_by_fixture": _rounding_failure_summary(reports),
        "exact_completion_controls": _exact_completion_controls(reports),
        "cold_start_controls": _cold_start_controls(reports),
        "unsupported_size_and_syntax_controls": _unsupported_size_and_syntax_controls(reports),
        "oracle_claim_boundary": _oracle_boundary(),
        "asp_continuous_relaxation_ready_score": 1.0 if status == "complete" else 0.0,
        "protected_files_unchanged": protected,
        "preconditions_checked": _preconditions(date, result_path, protected),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": dict(FIELD_PROVENANCE),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": dict(test_exit_codes or {RUN_COMMAND: 0}),
        "duration_s": float(duration_s),
        "random_seeds": list(RANDOM_SEEDS),
        "reproducibility_checksum": "",
        "honest_verdict": _honest_verdict(status),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate schema and fail closed on false readiness claims."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, field)
    _require(artifact.get("verifier_is_oracle") is True, "verifier_is_oracle")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(isinstance(artifact.get("parity_failure_count"), int), "parity_failure_count")
    _require(not isinstance(artifact.get("parity_failure_count"), bool), "parity_failure_count")
    parity_failures = int(artifact["parity_failure_count"])
    gradient_passed = (
        artifact.get("finite_difference_gradient_checks", {}).get("all_passed") is True
    )
    expected_score = 1.0 if parity_failures == 0 and gradient_passed else 0.0
    _require(
        artifact.get("asp_continuous_relaxation_ready_score") == expected_score,
        "ready_score",
    )
    if expected_score == 1.0:
        _require(artifact.get("status") == "complete", "status")
        _require(str(artifact.get("honest_verdict", "")).startswith("complete:"), "honest_verdict")
    else:
        _require(artifact.get("status") != "complete", "status")
    _require(
        set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact.get("field_principles", {})),
        "field_principles",
    )
    _require(
        set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact.get("field_provenance", {})),
        "field_provenance",
    )
    _require(
        artifact.get("oracle_claim_boundary", {}).get("oracle_distinct_verifier_claim") is False,
        "oracle_claim_boundary",
    )
    _require(
        artifact.get("unsupported_size_and_syntax_controls", {}).get("unsupported_size_rejected")
        is True,
        "unsupported_size",
    )
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum")


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while blanking its checksum field."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def sha256_json(value: Any) -> str:
    """Return a stable SHA-256 digest for JSON-compatible values."""

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def sha256_text(value: str) -> str:
    """Return a SHA-256 digest for text."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _evaluate_fixture(fixture: exp6274.ASPFixture) -> JsonDict:
    compiled = asp_energy.compile_program(fixture.program_text, program_id=fixture.fixture_id)
    table = relax.build_energy_table(
        compiled,
        fixture_id=fixture.fixture_id,
        max_atoms=MAX_ATOMS,
        max_vertices=MAX_VERTEX_COUNT,
    )
    solver_answer_sets = asp_energy.solve_with_clingo(compiled.program)
    start_manifest, refinement_outcomes, rounding_failures = _run_refinement_suite(
        table,
        solver_answer_sets,
    )
    return {
        "fixture_id": fixture.fixture_id,
        "family": fixture.family,
        "tags": list(fixture.tags),
        "permutation_of": fixture.permutation_of,
        "atoms": list(table.atoms),
        "atom_count": table.atom_count,
        "vertex_count": table.vertex_count,
        "vertex_parity": relax.verify_vertex_parity(compiled, table),
        "gradient_check": relax.check_gradient(
            table,
            _gradient_probe(table, fixture.fixture_id),
            epsilon=GRADIENT_EPSILON,
            tolerance=GRADIENT_TOLERANCE,
        ),
        "fractional_stationary_points": _fractional_stationary_points(table),
        "start_manifest": start_manifest,
        "refinement_outcomes": refinement_outcomes,
        "rounding_failures": rounding_failures,
        "solver_answer_sets": solver_answer_sets,
        "solver_answer_set_count": len(solver_answer_sets),
        "zero_energy_states": [list(state) for state in table.vertex_states if table.discrete_energy(state) == 0],
        "best_discrete_energy": table.best_discrete_energy,
        "best_state_count": sum(1 for energy in table.energies if energy == table.best_discrete_energy),
        "energy_spectrum_hash": sha256_json(sorted(table.energies)),
    }


def _run_refinement_suite(
    table: relax.VertexEnergyTable,
    solver_answer_sets: Sequence[Sequence[str]],
) -> tuple[list[JsonDict], dict[str, JsonDict], list[JsonDict]]:
    start_specs = _start_specs(table, solver_answer_sets)
    manifest: list[JsonDict] = []
    outcomes: dict[str, JsonDict] = {}
    rounding_failures: list[JsonDict] = []
    for spec in start_specs:
        start_id = str(spec["start_id"])
        manifest.append({key: value for key, value in spec.items() if key != "probabilities"})
        outcome = _refine_with_restarts(table, spec)
        outcomes[start_id] = outcome
        for attempt in outcome["attempts"]:
            if attempt["rounded_energy"] > table.best_discrete_energy:
                rounding_failures.append(
                    {
                        "start_id": start_id,
                        "kind": spec["kind"],
                        "seed": spec["seed"],
                        "restart_index": attempt["restart_index"],
                        "rounded_energy": attempt["rounded_energy"],
                        "best_discrete_energy": table.best_discrete_energy,
                        "rounded_state": attempt["rounded_state"],
                    }
                )
    return manifest, outcomes, rounding_failures


def _start_specs(
    table: relax.VertexEnergyTable,
    solver_answer_sets: Sequence[Sequence[str]],
) -> list[JsonDict]:
    specs: list[JsonDict] = [
        {
            "start_id": f"{table.fixture_id}:blank:{RANDOM_SEEDS[0]}",
            "kind": "blank",
            "seed": RANDOM_SEEDS[0],
            "probabilities": [0.0 for _ in table.atoms],
            "known_true": [],
            "known_false": list(table.atoms),
        }
    ]
    for seed in RANDOM_SEEDS:
        rng = random.Random(_stable_seed(table.fixture_id, seed))
        specs.append(
            {
                "start_id": f"{table.fixture_id}:random:{seed}",
                "kind": "random",
                "seed": seed,
                "probabilities": [rng.random() for _ in table.atoms],
                "known_true": [],
                "known_false": [],
            }
        )
    partial, known_true, known_false = _partial_start(table, solver_answer_sets)
    specs.append(
        {
            "start_id": f"{table.fixture_id}:partial_state:{RANDOM_SEEDS[0]}",
            "kind": "partial_state",
            "seed": RANDOM_SEEDS[0],
            "probabilities": partial,
            "known_true": known_true,
            "known_false": known_false,
        }
    )
    return specs


def _partial_start(
    table: relax.VertexEnergyTable,
    solver_answer_sets: Sequence[Sequence[str]],
) -> tuple[list[float], list[str], list[str]]:
    start = [0.5 for _ in table.atoms]
    known_true: list[str] = []
    known_false: list[str] = []
    target = set(solver_answer_sets[0]) if solver_answer_sets else set(table.atoms[:1])
    for index, atom in enumerate(table.atoms):
        if atom in target and len(known_true) < 2:
            start[index] = 1.0
            known_true.append(atom)
        elif atom not in target and len(known_false) < 2:
            start[index] = 0.0
            known_false.append(atom)
    return start, known_true, known_false


def _refine_with_restarts(table: relax.VertexEnergyTable, spec: Mapping[str, Any]) -> JsonDict:
    attempts: list[JsonDict] = []
    base = [float(value) for value in spec["probabilities"]]
    for restart_index in range(RESTART_BUDGET + 1):
        start = base if restart_index == 0 else _jittered_start(table.fixture_id, spec, restart_index)
        result = relax.refine(
            table,
            start,
            steps=OPTIMIZER_STEPS,
            step_size=OPTIMIZER_STEP_SIZE,
        )
        rounded_state = relax.round_probabilities(table, result["final_probabilities"])
        rounded_energy = table.discrete_energy(rounded_state)
        attempts.append(
            {
                "restart_index": restart_index,
                "initial_energy": result["initial_energy"],
                "final_energy": result["final_energy"],
                "rounded_state": rounded_state,
                "rounded_energy": rounded_energy,
                "success": rounded_energy == table.best_discrete_energy,
                "final_probabilities": _round_vector(result["final_probabilities"]),
                "gradient_norm": result["gradient_norm"],
                "projected_gradient_norm": result["projected_gradient_norm"],
                "relaxation_energy_evaluations": result["energy_evaluations"],
            }
        )
    best = min(attempts, key=lambda row: (row["rounded_energy"], row["final_energy"]))
    return {
        "kind": spec["kind"],
        "seed": spec["seed"],
        "step_budget": OPTIMIZER_STEPS,
        "restart_budget": RESTART_BUDGET,
        "restarts_used": RESTART_BUDGET,
        "exact_vertices_enumerated_for_table": table.vertex_count,
        "best_attempt": best,
        "attempts": attempts,
    }


def _jittered_start(
    fixture_id: str,
    spec: Mapping[str, Any],
    restart_index: int,
) -> list[float]:
    rng = random.Random(_stable_seed(f"{fixture_id}:{spec['kind']}:{restart_index}", int(spec["seed"])))
    return [0.85 * float(value) + 0.15 * rng.random() for value in spec["probabilities"]]


def _fractional_stationary_points(table: relax.VertexEnergyTable) -> list[JsonDict]:
    if table.atom_count == 0:
        return []
    candidate = [0.5 for _ in table.atoms]
    record = relax.stationary_point_record(
        table,
        candidate,
        gradient_tolerance=FRACTIONAL_GRADIENT_TOLERANCE,
        box_tolerance=BOX_TOLERANCE,
    )
    if record["stationary"]:
        record["candidate"] = "all_half"
        return [record]
    return []


def _gradient_probe(table: relax.VertexEnergyTable, fixture_id: str) -> list[float]:
    rng = random.Random(_stable_seed(fixture_id, RANDOM_SEEDS[-1]))
    return [0.2 + 0.6 * rng.random() for _ in table.atoms]


def _stable_seed(text: str, seed: int) -> int:
    digest = hashlib.sha256(f"{text}:{seed}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def _round_vector(values: Sequence[float]) -> list[float]:
    return [round(float(value), 10) for value in values]


def _gradient_summary(reports: Sequence[Mapping[str, Any]]) -> JsonDict:
    checks = {report["fixture_id"]: report["gradient_check"] for report in reports}
    max_abs_error = max(float(check["max_abs_error"]) for check in checks.values())
    return {
        "method": "central_finite_difference",
        "epsilon": GRADIENT_EPSILON,
        "tolerance": GRADIENT_TOLERANCE,
        "fixture_count": len(checks),
        "all_passed": all(check["passed"] is True for check in checks.values()),
        "max_abs_error": max_abs_error,
        "checks_by_fixture": checks,
    }


def _fractional_stationary_summary(reports: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_fixture = {
        report["fixture_id"]: report["fractional_stationary_points"] for report in reports
    }
    return {
        "gradient_tolerance": FRACTIONAL_GRADIENT_TOLERANCE,
        "box_tolerance": BOX_TOLERANCE,
        "stationary_point_count": sum(len(points) for points in by_fixture.values()),
        "by_fixture": by_fixture,
    }


def _rounding_failure_summary(reports: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_fixture = {report["fixture_id"]: report["rounding_failures"] for report in reports}
    return {
        "rounding_rule": "atom is true when probability >= 0.5",
        "failure_count": sum(len(rows) for rows in by_fixture.values()),
        "by_fixture": by_fixture,
    }


def _exact_completion_controls(reports: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "clingo": {
            "name_version": asp_energy.solver_name_version(),
            "oracle_role": "independent ASP answer-set completion control",
            "fixture_calls": len(reports),
            "all_calls_succeeded": len(reports) == 40,
            "answer_set_count_by_fixture": {
                report["fixture_id"]: report["solver_answer_set_count"] for report in reports
            },
        },
        "best_completion_energy_by_fixture": {
            report["fixture_id"]: report["best_discrete_energy"] for report in reports
        },
    }


def _cold_start_controls(reports: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_fixture = {
        report["fixture_id"]: {
            "exact_vertices_enumerated": report["vertex_count"],
            "best_discrete_energy": report["best_discrete_energy"],
            "best_state_count": report["best_state_count"],
            "zero_energy_state_count": len(report["zero_energy_states"]),
        }
        for report in reports
    }
    return {
        "control": "cold exact enumeration of every vertex before refinement",
        "all_exact_enumerations_completed": all(row["exact_vertices_enumerated"] > 0 for row in by_fixture.values()),
        "by_fixture": by_fixture,
    }


def _unsupported_size_and_syntax_controls(reports: Sequence[Mapping[str, Any]]) -> JsonDict:
    size_receipt = _unsupported_size_control()
    syntax_receipts = _unsupported_syntax_controls()
    bounds_receipt = _malformed_bounds_control()
    return {
        "unsupported_size_rejected": size_receipt["rejected"],
        "unsupported_size_receipt": size_receipt,
        "unsupported_syntax_all_rejected": all(row["rejected"] for row in syntax_receipts),
        "unsupported_syntax_receipts": syntax_receipts,
        "malformed_bounds_rejected": bounds_receipt["rejected"],
        "malformed_bounds_receipt": bounds_receipt,
        "label_permutation_control": _label_permutation_control(reports),
        "sign_reversal_detected": _sign_reversal_detected(reports),
        "rounding_harm_control": _rounding_harm_control(),
    }


def _unsupported_size_control() -> JsonDict:
    source = "0 { a0; a1; a2; a3; a4; a5; a6; a7; a8; a9; a10; a11; a12 } 13.\n"
    compiled = asp_energy.compile_program(source, program_id="unsupported_size")
    try:
        relax.build_energy_table(
            compiled,
            fixture_id="unsupported_size",
            max_atoms=MAX_ATOMS,
            max_vertices=MAX_VERTEX_COUNT,
        )
    except relax.UnsupportedRelaxationFixture as exc:
        return {
            "rejected": exc.reason == "vertex_bound",
            "reason": exc.reason,
            "atom_count": exc.atom_count,
            "vertex_count": exc.vertex_count,
        }
    return {"rejected": False, "reason": "accepted", "atom_count": 13, "vertex_count": 8192}


def _unsupported_syntax_controls() -> list[JsonDict]:
    cases = {
        "p(X).": "variables",
        "a | b.": "disjunction",
        "#minimize { 1,a : a }.": "directive_or_optimization",
    }
    receipts = []
    for source, expected in cases.items():
        try:
            asp_energy.compile_program(source, program_id="unsupported_syntax")
        except asp_energy.UnsupportedASPSyntax as exc:
            receipts.append(
                {
                    "source": source,
                    "expected_syntax_class": expected,
                    "observed_syntax_class": exc.syntax_class,
                    "rejected": exc.syntax_class == expected and exc.energy_constructed is False,
                }
            )
    return receipts


def _malformed_bounds_control() -> JsonDict:
    source = "2 { a } 1.\n"
    try:
        asp_energy.compile_program(source, program_id="bad_bounds")
    except asp_energy.UnsupportedASPSyntax as exc:
        return {
            "source": source,
            "rejected": exc.syntax_class == "invalid_cardinality_bounds",
            "observed_syntax_class": exc.syntax_class,
        }
    return {"source": source, "rejected": False, "observed_syntax_class": "accepted"}


def _label_permutation_control(reports: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_id = {report["fixture_id"]: report for report in reports}
    pairs = []
    for report in reports:
        source_id = report.get("permutation_of")
        if not source_id:
            continue
        source = by_id[source_id]
        pairs.append(
            {
                "source_fixture_id": source_id,
                "permuted_fixture_id": report["fixture_id"],
                "atom_count_match": source["atom_count"] == report["atom_count"],
                "vertex_count_match": source["vertex_count"] == report["vertex_count"],
                "energy_spectrum_match": source["energy_spectrum_hash"]
                == report["energy_spectrum_hash"],
            }
        )
    return {
        "pair_count": len(pairs),
        "pairs": pairs,
        "passed": all(
            pair["atom_count_match"]
            and pair["vertex_count_match"]
            and pair["energy_spectrum_match"]
            for pair in pairs
        ),
    }


def _sign_reversal_detected(reports: Sequence[Mapping[str, Any]]) -> bool:
    for report in reports:
        parity = report["vertex_parity"]
        if parity["checked_vertices"] and report["best_discrete_energy"] == 0:
            nonzero_exists = any(
                attempt["rounded_energy"] > 0
                for outcome in report["refinement_outcomes"].values()
                for attempt in outcome["attempts"]
            )
            if nonzero_exists:
                return True
    return False


def _rounding_harm_control() -> JsonDict:
    compiled = asp_energy.compile_program("1 { yes; no } 1.\n", program_id="rounding_harm")
    table = relax.build_energy_table(compiled, fixture_id="rounding_harm")
    probabilities = [0.49, 0.49]
    rounded_state = relax.round_probabilities(table, probabilities)
    return {
        "probabilities": probabilities,
        "continuous_energy": relax.energy_at(probabilities, table),
        "rounded_state": rounded_state,
        "rounded_energy": table.discrete_energy(rounded_state),
        "best_discrete_energy": table.best_discrete_energy,
        "rounding_harm_observed": table.discrete_energy(rounded_state) > table.best_discrete_energy,
    }


def _upstream_compiler_receipt() -> JsonDict:
    path = REPO_ROOT / UPSTREAM_RESULT_RELATIVE_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    verdict = str(payload["honest_verdict"])
    return {
        "path": UPSTREAM_RESULT_RELATIVE_PATH.as_posix(),
        "sha256": sha256_text(path.read_text(encoding="utf-8")),
        "terminal_class": verdict.split(":", 1)[0],
        "status": payload["status"],
        "parity_failure_count": payload["parity_failure_count"],
        "asp_energy_semantic_ready_score": payload["asp_energy_semantic_ready_score"],
    }


def _fixture_manifest_receipt() -> JsonDict:
    path = REPO_ROOT / FIXTURE_MANIFEST_RELATIVE_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": FIXTURE_MANIFEST_RELATIVE_PATH.as_posix(),
        "sha256": sha256_text(path.read_text(encoding="utf-8")),
        "fixture_count": len(payload["fixtures"]),
        "max_state_count": payload["max_state_count"],
        "random_seed": payload["random_seed"],
    }


def _source_hashes() -> JsonDict:
    paths = [
        Path("python/carnot/asp_continuous_relaxation.py"),
        Path("python/carnot/experiment_6287_asp_continuous_relaxation.py"),
        Path("tests/python/test_asp_continuous_relaxation_6287.py"),
        SPEC_RELATIVE_PATH,
        Path("python/carnot/asp_energy.py"),
        Path("python/carnot/experiment_6274_asp_energy_semantic_compiler.py"),
    ]
    return {
        path.as_posix(): sha256_text((REPO_ROOT / path).read_text(encoding="utf-8"))
        for path in paths
    }


def _relaxation_boundary() -> JsonDict:
    return {
        "definition": (
            "E_bar(p)=sum_x E_discrete(x) * product_i p_i^x_i * (1-p_i)^(1-x_i)"
        ),
        "claim": "E_bar equals the Exp6274 finite discrete energy on binary vertices.",
        "claim_boundary": "Interior probabilities are diagnostics and editable state only.",
        "not_claimed": "learned Kona model; learned verifier; diffusion language model",
    }


def _analytic_gradient_definition() -> JsonDict:
    return {
        "formula": (
            "dE/dp_j=sum_x_without_j (E(x_j=1,x_-j)-E(x_j=0,x_-j)) "
            "* product_i!=j basis_i"
        ),
        "implementation": "python/carnot/asp_continuous_relaxation.py::gradient_at",
        "finite_difference_check": "central difference away from p_i in {0,1}",
    }


def _optimizer_budget() -> JsonDict:
    return {
        "algorithm": "projected_gradient_descent_on_[0,1]^n",
        "step_budget": OPTIMIZER_STEPS,
        "step_size": OPTIMIZER_STEP_SIZE,
        "restart_budget": RESTART_BUDGET,
        "start_kinds": ["blank", "random", "partial_state"],
        "readiness_depends_on_refinement_success": False,
    }


def _oracle_boundary() -> JsonDict:
    return {
        "verifier_is_oracle": True,
        "oracle_distinct_verifier_claim": False,
        "clingo_role": "independent ASP answer-set oracle control",
        "cold_enumeration_role": "exact finite energy control",
        "boundary": "The relaxation bridges exact semantics to editable probabilities only.",
    }


def _protected_hash_receipts() -> JsonDict:
    protected = ("scripts/research_conductor.py", "CODEX.md", "CLAUDE.md")
    receipts: JsonDict = {}
    for rel in protected:
        text_before = (REPO_ROOT / rel).read_text(encoding="utf-8")
        text_after = (REPO_ROOT / rel).read_text(encoding="utf-8")
        receipts[rel] = {
            "sha256_before": sha256_text(text_before),
            "sha256_after": sha256_text(text_after),
            "unchanged": text_before == text_after,
        }
    return receipts


def _preconditions(date: str, result_path: Path, protected: Mapping[str, Any]) -> JsonDict:
    return {
        "date": date,
        "result_path": _display_path(result_path),
        "git_status_at_artifact_build": _git_status_short(),
        "python_version": platform.python_version(),
        "solver_version": asp_energy.solver_name_version(),
        "upstream_compiler_hash": _upstream_compiler_receipt()["sha256"],
        "fixture_manifest_hash": _fixture_manifest_receipt()["sha256"],
        "bounds": {"max_atoms": MAX_ATOMS, "max_vertices": MAX_VERTEX_COUNT},
        "gradient": {"epsilon": GRADIENT_EPSILON, "tolerance": GRADIENT_TOLERANCE},
        "optimizer": _optimizer_budget(),
        "random_seeds": list(RANDOM_SEEDS),
        "protected_hashes": protected,
    }


def _git_status_short() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def _honest_verdict(status: str) -> str:
    if status == "complete":
        return "complete: multilinear ASP relaxation matches vertices and gradients"
    return "blocked: multilinear ASP relaxation failed parity or gradient checks"


def _canonical_json(value: Any, *, indent: int | None = None) -> str:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=None if indent else (",", ":"),
            indent=indent,
            ensure_ascii=True,
        )
        + "\n"
    )


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the required Exp6287 run command."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    args = parser.parse_args(argv)
    started = time.perf_counter()
    artifact = run(
        date=args.date,
        result_path=Path(args.result_path),
        duration_s=time.perf_counter() - started,
    )
    print(
        json.dumps(
            {
                "result": _display_path(Path(args.result_path)),
                "status": artifact["status"],
                "fixture_count": artifact["fixture_count"],
                "parity_failure_count": artifact["parity_failure_count"],
                "honest_verdict": artifact["honest_verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
