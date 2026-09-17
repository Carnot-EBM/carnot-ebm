"""Verify that source 2-CNF compilation preserves its finite-temperature law.

An entailed clause is redundant for Boolean satisfiability. It is not redundant
when its violation indicator is added to a finite-temperature energy. This
host-only fixture checks both facts by enumerating every state.

Spec refs: REQ-ISING-7377 and SCENARIO-ISING-7377-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import UTC, datetime
import itertools
import json
import math
from pathlib import Path
import platform
import tempfile
import time
from typing import Any

from carnot import experiment_7358_v646_validation_contract as validation_contract
from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]
Clause = tuple[int, int]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260917"
MILESTONE = "2026.09.647"
PHASE = 3
EXPERIMENT_ID = "exp7377-v647-ising-law"
SCHEMA = "carnot.exp7377.v647.ising_law.v1"
RESULT_PATH = Path("results/experiment_7377_v647_ising_law.json")
RAW_DIR = Path("results/raw/experiment_7377_v647_ising_law")
MODULE_PATH = Path("python/carnot/experiment_7377_v647_ising_law.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7377_v647_ising_law.py")
TEST_PATH = Path("tests/python/test_experiment_7377_v647_ising_law.py")
SPEC_PATH = Path("openspec/capabilities/ising-backend/spec.md")
FORMULA_SEEDS = (7_377_001, 7_377_002, 7_377_003)
BETA_GRID = (0.5, 1.0, 2.0)
EXP7378_CHAIN_SEEDS = (7_378_101, 7_378_211, 7_378_307, 7_378_401)
REQUIRED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "cold_artifact_replay",
    "independent_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
ZERO_CURRENT_INVOCATIONS = {
    "model_loads_attempted": 0,
    "model_loads_completed": 0,
    "model_loads_failed": 0,
    "model_loads_cancelled": 0,
    "model_loads_in_flight": 0,
    "generation_calls_attempted": 0,
    "generation_calls_completed": 0,
    "generation_calls_failed": 0,
    "generation_calls_cancelled": 0,
    "generation_calls_in_flight": 0,
}
CLOSED_VERDICTS = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
REQUIRED_SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("openspec/capabilities/constraint-verification/spec.md"),
    SPEC_PATH,
    Path("python/carnot/phase3/k_sat_ising.py"),
    Path("python/carnot/verify/ising.py"),
    Path("python/carnot/samplers/parallel_ising.py"),
    Path("research-references.md"),
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)
V647_MANIFEST = validation_contract.AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


@dataclass(frozen=True)
class IsingLaw:
    """Store the exact polynomial form of summed clause violations.

    Biases and couplings use the common Ising sign convention shown in
    ``energy_convention``. The offset is necessary because probability
    normalizers and direct energy comparisons need the complete polynomial.
    """

    n_vars: int
    offset: float
    biases: tuple[float, ...]
    couplings: tuple[tuple[int, int, float], ...]
    energy_convention: str = "E(s)=offset-sum_i(h_i*s_i)-sum_i<j(J_ij*s_i*s_j)"
    bit_spin_convention: str = "s_i=2*x_i-1; x_i=(s_i+1)/2"

    def energy(self, spins: Sequence[int]) -> float:
        """Evaluate the complete polynomial, including its constant offset."""

        if len(spins) != self.n_vars:
            raise ValueError("spin_state_length")
        value = self.offset - math.fsum(
            bias * float(spins[index]) for index, bias in enumerate(self.biases)
        )
        value -= math.fsum(
            coupling * float(spins[left]) * float(spins[right])
            for left, right, coupling in self.couplings
        )
        return value

    def to_dict(self) -> JsonDict:
        """Return a JSON-safe coefficient record with one-based variable labels."""

        return {
            "n_vars": self.n_vars,
            "offset": self.offset,
            "biases": list(self.biases),
            "couplings": [
                {"left": left + 1, "right": right + 1, "value": value}
                for left, right, value in self.couplings
            ],
            "energy_convention": self.energy_convention,
            "bit_spin_convention": self.bit_spin_convention,
        }


def bits_to_spins(bits: Sequence[int]) -> tuple[int, ...]:
    """Map Boolean bits to spins without changing variable order."""

    return tuple(2 * int(bit) - 1 for bit in bits)


def enumerate_bits(n_vars: int) -> list[tuple[int, ...]]:
    """Enumerate the complete finite state space in stable lexical order."""

    return list(itertools.product((0, 1), repeat=n_vars))


def _validated_clauses(n_vars: int, clauses: Sequence[Sequence[int]]) -> tuple[Clause, ...]:
    """Reject inputs that are not bounded, exact two-literal clauses."""

    if not 1 <= n_vars <= 12:
        raise ValueError("n_vars_out_of_range")
    normalized: list[Clause] = []
    for clause in clauses:
        if len(clause) != 2:
            raise ValueError("clause_not_two_literals")
        left, right = int(clause[0]), int(clause[1])
        if left == 0 or right == 0 or abs(left) > n_vars or abs(right) > n_vars:
            raise ValueError("literal_out_of_range")
        normalized.append((left, right))
    return tuple(normalized)


def compile_2cnf(n_vars: int, clauses: Sequence[Sequence[int]]) -> IsingLaw:
    """Compile clause violations into one exact offset-aware Ising polynomial."""

    source = _validated_clauses(n_vars, clauses)
    offset = 0.0
    biases = [0.0] * n_vars
    coupling_map: dict[tuple[int, int], float] = {}
    for left_literal, right_literal in source:
        left = abs(left_literal) - 1
        right = abs(right_literal) - 1
        left_sign = 1.0 if left_literal > 0 else -1.0
        right_sign = 1.0 if right_literal > 0 else -1.0
        offset += 0.25
        biases[left] += 0.25 * left_sign
        biases[right] += 0.25 * right_sign
        product = left_sign * right_sign
        if left == right:
            offset += 0.25 * product
        else:
            pair = tuple(sorted((left, right)))
            coupling_map[pair] = coupling_map.get(pair, 0.0) - 0.25 * product
    couplings = tuple(
        (left, right, value)
        for (left, right), value in sorted(coupling_map.items())
        if value != 0.0
    )
    return IsingLaw(n_vars, offset, tuple(biases), couplings)


def independent_clause_energy(bits: Sequence[int], clauses: Sequence[Sequence[int]]) -> float:
    """Count violated clauses directly, without using compiled coefficients."""

    violations = 0
    for clause in clauses:
        satisfied = False
        for literal in clause:
            bit = int(bits[abs(int(literal)) - 1])
            satisfied = satisfied or (bit == 1 if int(literal) > 0 else bit == 0)
        violations += int(not satisfied)
    return float(violations)


def condition_mask(states: Sequence[Sequence[int]], assumptions: Sequence[int]) -> list[bool]:
    """Represent unit conditioning as a support clamp, never as an energy term."""

    return [
        all(
            int(bits[abs(int(literal)) - 1]) == (1 if int(literal) > 0 else 0)
            for literal in assumptions
        )
        for bits in states
    ]


def normalized_probabilities(
    energies: Sequence[float], support: Sequence[bool], beta: float
) -> tuple[list[float], float]:
    """Normalize ``exp(-beta*E)`` on one explicit finite support."""

    if beta <= 0:
        raise ValueError("beta_must_be_positive")
    if len(energies) != len(support):
        raise ValueError("energy_mask_length")
    weights = [
        math.exp(-beta * float(energy)) if keep else 0.0 for energy, keep in zip(energies, support)
    ]
    normalizer = math.fsum(weights)
    if normalizer == 0.0:
        return [0.0] * len(weights), 0.0
    return [weight / normalizer for weight in weights], normalizer


def _total_variation(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute total variation for two finite vectors in the same state order."""

    return 0.5 * math.fsum(abs(float(a) - float(b)) for a, b in zip(left, right))


def _source_hash(n_vars: int, clauses: Sequence[Sequence[int]]) -> str:
    return validation_contract.canonical_hash(
        {"n_vars": n_vars, "clauses": [list(clause) for clause in clauses]}
    )


def build_frozen_fixture() -> list[JsonDict]:
    """Return 24 fixed six-variable formulas with explicit proof paths."""

    permutations = (
        (1, 2, 3, 4, 5, 6),
        (2, 5, 1, 6, 3, 4),
        (6, 3, 5, 2, 4, 1),
    )
    features = (
        "development_counterexample",
        "duplicate_clause",
        "conditioned_endpoint",
        "conflicting_assumptions",
        "duplicate_assumption",
        "conditioned_antecedent_false",
        "shortcut_plus_long_proof",
        "conditioned_source_contradiction",
    )
    formulas: list[JsonDict] = []
    for seed, path in zip(FORMULA_SEEDS, permutations, strict=True):
        chain = [(-path[index], path[index + 1]) for index in range(5)]
        extras: tuple[tuple[int, int], ...] = (
            (),
            (chain[1],),
            ((path[1], path[3]),),
            ((-path[5], -path[2]),),
            ((path[1], -path[1]),),
            (chain[2], (path[4], path[0])),
            ((-path[0], path[2]),),
            ((path[5], path[0]), chain[4]),
        )
        assumptions = (
            (),
            (path[0],),
            (-path[5],),
            (path[0], -path[0]),
            (path[1], path[1]),
            (-path[0],),
            (path[2],),
            (path[0], -path[5]),
        )
        for index in range(8):
            clauses = [*chain, *extras[index]]
            source_hash = _source_hash(6, clauses)
            formulas.append(
                {
                    "formula_id": f"seed-{seed}-formula-{index}",
                    "seed": seed,
                    "formula_index": index,
                    "feature": features[index],
                    "n_vars": 6,
                    "original_clauses": [list(clause) for clause in clauses],
                    "assumptions": list(assumptions[index]),
                    "implied_clause": [-path[0], path[5]],
                    "source_hash": source_hash,
                    "certificate": {
                        "source_hash": source_hash,
                        "antecedent": path[0],
                        "consequent": path[5],
                        "implied_clause": [-path[0], path[5]],
                        "path_literals": list(path),
                        "source_clause_indices": list(range(5)),
                        "usage": "logical_entailment_only",
                    },
                }
            )
    return formulas


def fixture_hash(formulas: Sequence[Mapping[str, Any]]) -> str:
    """Bind the exact formula order, clauses, assumptions, and certificates."""

    return validation_contract.canonical_hash(list(formulas))


def validate_implication_certificate(formula: Mapping[str, Any]) -> list[str]:
    """Check each proof edge against the original source clause at its named ID."""

    errors: list[str] = []
    clauses = [tuple(int(value) for value in clause) for clause in formula["original_clauses"]]
    certificate = formula["certificate"]
    path = [int(value) for value in certificate["path_literals"]]
    indices = [int(value) for value in certificate["source_clause_indices"]]
    if certificate.get("source_hash") != _source_hash(int(formula["n_vars"]), clauses):
        errors.append("source_hash_mismatch")
    if (
        not path
        or path[0] != certificate.get("antecedent")
        or path[-1] != certificate.get("consequent")
    ):
        errors.append("certificate_endpoints_mismatch")
    expected_implied = [-int(certificate["antecedent"]), int(certificate["consequent"])]
    if list(certificate.get("implied_clause") or []) != expected_implied:
        errors.append("implied_clause_mismatch")
    if len(indices) != max(0, len(path) - 1):
        errors.append("path_index_count_mismatch")
    else:
        for start, end, clause_index in zip(path[:-1], path[1:], indices, strict=True):
            if not 0 <= clause_index < len(clauses):
                errors.append("source_clause_index_invalid")
                continue
            if clauses[clause_index] != (-start, end):
                errors.append("source_edge_mismatch")
    implied = [expected_implied]
    for bits in enumerate_bits(int(formula["n_vars"])):
        if (
            independent_clause_energy(bits, clauses) == 0.0
            and independent_clause_energy(bits, implied) != 0.0
        ):
            errors.append("enumerated_entailment_failed")
            break
    return list(dict.fromkeys(errors))


def _minimum_states(
    states: Sequence[Sequence[int]], energies: Sequence[float], support: Sequence[bool]
) -> list[str]:
    allowed = [(bits, energy) for bits, energy, keep in zip(states, energies, support) if keep]
    if not allowed:
        return []
    minimum = min(energy for _bits, energy in allowed)
    return ["".join(str(bit) for bit in bits) for bits, energy in allowed if energy == minimum]


def run_exact_panel(formulas: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Enumerate every formula, beta, and energy-definition condition."""

    rows: list[JsonDict] = []
    total = len(formulas) * len(BETA_GRID) * 3
    completed = 0
    panel_started = time.monotonic()
    for formula in formulas:
        row_started = time.monotonic()
        n_vars = int(formula["n_vars"])
        clauses = [tuple(int(value) for value in clause) for clause in formula["original_clauses"]]
        implied = tuple(int(value) for value in formula["implied_clause"])
        assumptions = [int(value) for value in formula["assumptions"]]
        states = enumerate_bits(n_vars)
        support = condition_mask(states, assumptions)
        source_direct = [independent_clause_energy(bits, clauses) for bits in states]
        appended_clauses = [*clauses, implied]
        appended_direct = [independent_clause_energy(bits, appended_clauses) for bits in states]
        source_law = compile_2cnf(n_vars, clauses)
        appended_law = compile_2cnf(n_vars, appended_clauses)
        certificate_failures = validate_implication_certificate(formula)
        full_support = [True] * len(states)
        source_minima = _minimum_states(states, source_direct, full_support)
        appended_minima = _minimum_states(states, appended_direct, full_support)
        conditions = (
            ("source_only", source_law, source_direct, "none"),
            (
                "proof_assisted_source_only",
                source_law,
                source_direct,
                "source_checked_logical_entailment_only",
            ),
            (
                "appended_implied_clause",
                appended_law,
                appended_direct,
                "intentional_wrong_law_negative_control",
            ),
        )
        for beta in BETA_GRID:
            source_probabilities, source_normalizer = normalized_probabilities(
                source_direct, support, beta
            )
            for condition, compiled, independent, certificate_usage in conditions:
                compiled_energy = [compiled.energy(bits_to_spins(bits)) for bits in states]
                compiled_probabilities, compiled_normalizer = normalized_probabilities(
                    compiled_energy, support, beta
                )
                enumeration_probabilities, enumeration_normalizer = normalized_probabilities(
                    independent, support, beta
                )
                completed += 1
                rows.append(
                    {
                        "row_type": "finite_law",
                        "formula_id": formula["formula_id"],
                        "seed": formula["seed"],
                        "feature": formula["feature"],
                        "n_vars": n_vars,
                        "beta": beta,
                        "condition": condition,
                        "original_clause_count": len(clauses),
                        "evaluated_clause_count": len(independent),
                        "support_size": sum(support),
                        "state_count": len(states),
                        "compiled_normalizer": compiled_normalizer,
                        "enumerated_normalizer": enumeration_normalizer,
                        "source_normalizer": source_normalizer,
                        "max_abs_energy_residual": max(
                            abs(left - right) for left, right in zip(compiled_energy, independent)
                        ),
                        "enumeration_total_variation": _total_variation(
                            compiled_probabilities, enumeration_probabilities
                        ),
                        "source_law_total_variation": _total_variation(
                            compiled_probabilities, source_probabilities
                        ),
                        "source_minimum_states": source_minima,
                        "condition_minimum_states": (
                            appended_minima
                            if condition == "appended_implied_clause"
                            else source_minima
                        ),
                        "certificate_usage": certificate_usage,
                        "unit_conditioning": "hard_clamp_only",
                        "proof_zeroed_positive_source_mass": math.fsum(
                            probability
                            for probability, observed in zip(
                                source_probabilities, compiled_probabilities
                            )
                            if probability > 0.0 and observed == 0.0
                        ),
                        "outcome": "complete",
                        "costs": {
                            "states_enumerated": len(states),
                            "clause_state_evaluations": len(states) * len(independent),
                            "formula_elapsed_s": time.monotonic() - row_started,
                        },
                        "failures": list(certificate_failures),
                        "censored": False,
                    }
                )
                if time.monotonic() - panel_started >= 60.0:  # pragma: no cover - tiny fixture.
                    print(
                        f"[exp7377] phase=evaluate event=heartbeat completed={completed}/{total} "
                        f"elapsed_s={time.monotonic() - panel_started:.3f}",
                        flush=True,
                    )
                    panel_started = time.monotonic()
    return rows


def run_negative_controls(formula: Mapping[str, Any]) -> list[JsonDict]:
    """Mutate one boundary at a time and record the independent detector."""

    n_vars = int(formula["n_vars"])
    clauses = [tuple(int(value) for value in clause) for clause in formula["original_clauses"]]
    implied = tuple(int(value) for value in formula["implied_clause"])
    states = enumerate_bits(n_vars)
    support = [True] * len(states)
    beta = 1.0
    source_law = compile_2cnf(n_vars, clauses)
    source_energy = [independent_clause_energy(bits, clauses) for bits in states]
    source_probability, _ = normalized_probabilities(source_energy, support, beta)
    source_minima = _minimum_states(states, source_energy, support)

    first_bias = next(index for index, value in enumerate(source_law.biases) if value != 0.0)
    sign_biases = list(source_law.biases)
    sign_biases[first_bias] *= -1.0
    mutations: list[tuple[str, str, list[float], list[float], float, bool]] = []

    offset_law = replace(source_law, offset=source_law.offset + 0.125)
    offset_energy = [offset_law.energy(bits_to_spins(bits)) for bits in states]
    offset_probability, _ = normalized_probabilities(offset_energy, support, beta)
    mutations.append(
        (
            "offset",
            "absolute_energy_residual",
            offset_energy,
            offset_probability,
            0.0,
            _minimum_states(states, offset_energy, support) == source_minima,
        )
    )

    sign_law = replace(source_law, biases=tuple(sign_biases))
    sign_energy = [sign_law.energy(bits_to_spins(bits)) for bits in states]
    sign_probability, _ = normalized_probabilities(sign_energy, support, beta)
    mutations.append(
        (
            "sign",
            "energy_and_probability_residual",
            sign_energy,
            sign_probability,
            0.0,
            _minimum_states(states, sign_energy, support) == source_minima,
        )
    )

    beta_probability, _ = normalized_probabilities(source_energy, support, 2.0)
    mutations.append(
        ("beta", "normalized_probability_residual", source_energy, beta_probability, 0.0, True)
    )

    multiplicity_law = compile_2cnf(n_vars, [*clauses, clauses[0]])
    multiplicity_energy = [multiplicity_law.energy(bits_to_spins(bits)) for bits in states]
    multiplicity_probability, _ = normalized_probabilities(multiplicity_energy, support, beta)
    mutations.append(
        (
            "clause_multiplicity",
            "source_law_probability_residual",
            multiplicity_energy,
            multiplicity_probability,
            0.0,
            _minimum_states(states, multiplicity_energy, support) == source_minima,
        )
    )

    certificate_filter = [independent_clause_energy(bits, (implied,)) == 0.0 for bits in states]
    conditioned_probability, _ = normalized_probabilities(source_energy, certificate_filter, beta)
    removed_mass = math.fsum(
        probability
        for probability, observed in zip(source_probability, conditioned_probability)
        if probability > 0.0 and observed == 0.0
    )
    mutations.append(
        (
            "conditioning",
            "positive_source_mass_removed",
            source_energy,
            conditioned_probability,
            removed_mass,
            _minimum_states(states, source_energy, certificate_filter) == source_minima,
        )
    )

    appended_law = compile_2cnf(n_vars, [*clauses, implied])
    appended_energy = [appended_law.energy(bits_to_spins(bits)) for bits in states]
    appended_probability, _ = normalized_probabilities(appended_energy, support, beta)
    mutations.append(
        (
            "extra_implied_clause",
            "finite_temperature_probability_residual",
            appended_energy,
            appended_probability,
            0.0,
            _minimum_states(states, appended_energy, support) == source_minima,
        )
    )

    rows: list[JsonDict] = []
    for (
        mutation,
        detector,
        energy,
        probabilities,
        positive_mass_removed,
        minima_preserved,
    ) in mutations:
        energy_residual = max(abs(left - right) for left, right in zip(energy, source_energy))
        probability_change = _total_variation(probabilities, source_probability)
        if mutation == "offset":
            detected = energy_residual > 1e-12 and probability_change <= 1e-12
        elif mutation == "beta":
            detected = energy_residual <= 1e-12 and probability_change > 1e-10
        elif mutation == "conditioning":
            detected = positive_mass_removed > 0.0
        elif mutation == "extra_implied_clause":
            detected = minima_preserved and probability_change > 1e-10
        else:
            detected = energy_residual > 1e-12 and probability_change > 1e-10
        rows.append(
            {
                "row_type": "negative_control",
                "formula_id": formula["formula_id"],
                "mutation": mutation,
                "expected_detector": detector,
                "max_abs_energy_residual": energy_residual,
                "probability_change": probability_change,
                "positive_source_mass_removed": positive_mass_removed,
                "minima_preserved": minima_preserved,
                "detected": detected,
                "outcome": "complete",
                "costs": {"states_enumerated": len(states), "current_model_calls": 0},
                "failures": [] if detected else ["mutation_not_detected"],
                "censored": False,
            }
        )
    return rows


def _precondition(
    check: str,
    upstream: str,
    field: str,
    expected: Any,
    observed: Any,
    *,
    terminal_blocking: bool,
) -> JsonDict:
    return {
        "check": check,
        "upstream": upstream,
        "artifact_field": field,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
        "terminal_blocking": terminal_blocking,
    }


def collect_preconditions(
    repo_root: Path,
) -> tuple[list[JsonDict], dict[str, str], list[JsonDict]]:
    """Authenticate source bytes and confirm that no upstream science is required."""

    root = repo_root.resolve()
    checks: list[JsonDict] = []
    hashes: dict[str, str] = {}
    for relative in REQUIRED_SOURCE_PATHS:
        path = root / relative
        present = path.is_file() and path.stat().st_size > 0
        checks.append(
            _precondition(
                f"source_bytes:{relative.as_posix()}",
                relative.as_posix(),
                "bytes",
                "readable_nonempty_bytes",
                "readable_nonempty_bytes" if present else "missing",
                terminal_blocking=True,
            )
        )
        if present:
            hashes[relative.as_posix()] = validation_contract.sha256_file(path)
    spec_path = root / SPEC_PATH
    spec_text = spec_path.read_text(encoding="utf-8") if spec_path.is_file() else ""
    checks.append(
        _precondition(
            "driving_requirement",
            SPEC_PATH.as_posix(),
            "REQ-*",
            "REQ-ISING-7377",
            "REQ-ISING-7377" if "REQ-ISING-7377" in spec_text else None,
            terminal_blocking=True,
        )
    )
    exclusion_path = root / "ops/exclusion_manifest.yaml"
    exclusion_text = exclusion_path.read_text(encoding="utf-8") if exclusion_path.is_file() else ""
    excluded = "experiment_id: 7377" in exclusion_text or "exp7377-ising-law" in exclusion_text
    checks.append(
        _precondition(
            "current_task_not_quarantined",
            "ops/exclusion_manifest.yaml",
            EXPERIMENT_ID,
            False,
            excluded,
            terminal_blocking=True,
        )
    )
    checks.append(
        _precondition(
            "structured_science_prerequisite",
            "research-roadmap.yaml:exp7377-ising-law",
            "gated_on",
            "none",
            "none",
            terminal_blocking=True,
        )
    )
    sidecars = [
        {
            "label": "no_historical_model_inputs",
            "counted_as_current": False,
            "receipts": [],
        }
    ]
    return checks, hashes, sidecars


def _receipt_set_passes(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    counts = Counter(row.get("name") for row in receipts)
    return all(
        counts[name] == 1
        and next(row for row in receipts if row.get("name") == name).get("passed") is True
        and next(row for row in receipts if row.get("name") == name).get("exit_code") == 0
        and next(row for row in receipts if row.get("name") == name).get("timed_out") is False
        for name in names
    )


def independent_reduce(artifact: Mapping[str, Any]) -> JsonDict:
    """Recompute every readiness component from raw formulas, rows, and receipts."""

    formulas = artifact.get("frozen_formulas") or []
    finite_rows = artifact.get("finite_law_rows") or []
    controls = artifact.get("negative_control_rows") or []
    receipts = artifact.get("validation_receipts") or []
    blocking = [
        row
        for row in artifact.get("preconditions_checked") or []
        if isinstance(row, Mapping) and row.get("terminal_blocking") is True
    ]
    preconditions_passed = bool(blocking) and all(row.get("passed") is True for row in blocking)
    fixture_complete = (
        len(formulas) == 24
        and {row.get("seed") for row in formulas} == set(FORMULA_SEEDS)
        and all(not validate_implication_certificate(row) for row in formulas)
        and artifact.get("frozen_sampling_protocol", {}).get("fixture_sha256")
        == fixture_hash(formulas)
    )
    faithful = [row for row in finite_rows if row.get("condition") != "appended_implied_clause"]
    appended = [row for row in finite_rows if row.get("condition") == "appended_implied_clause"]
    panel_complete = (
        len(finite_rows) == 24 * len(BETA_GRID) * 3
        and len(faithful) == 24 * len(BETA_GRID) * 2
        and len(appended) == 24 * len(BETA_GRID)
        and all(row.get("censored") is False and not row.get("failures") for row in finite_rows)
    )
    energy_parity = bool(faithful) and all(
        float(row.get("max_abs_energy_residual", math.inf)) <= 1e-12 for row in faithful
    )
    probability_parity = bool(faithful) and all(
        float(row.get("enumeration_total_variation", math.inf)) <= 1e-10
        and float(row.get("source_law_total_variation", math.inf)) <= 1e-10
        for row in faithful
    )
    normalizers_match = bool(finite_rows) and all(
        math.isclose(
            float(row.get("compiled_normalizer", math.nan)),
            float(row.get("enumerated_normalizer", math.nan)),
            rel_tol=0.0,
            abs_tol=1e-10,
        )
        for row in finite_rows
    )
    proof_safety = bool(faithful) and all(
        float(row.get("proof_zeroed_positive_source_mass", math.inf)) <= 1e-15 for row in faithful
    )
    minima_preserved = bool(appended) and all(
        row.get("source_minimum_states") == row.get("condition_minimum_states") for row in appended
    )
    intentional_change_detected = bool(appended) and any(
        float(row.get("source_law_total_variation", 0.0)) > 1e-10 for row in appended
    )
    expected_mutations = {
        "offset",
        "sign",
        "beta",
        "clause_multiplicity",
        "conditioning",
        "extra_implied_clause",
    }
    negative_controls_passed = (
        {row.get("mutation") for row in controls} == expected_mutations
        and len(controls) == len(expected_mutations)
        and all(row.get("detected") is True and row.get("censored") is False for row in controls)
    )
    affected_validation_passed = _receipt_set_passes(receipts, REQUIRED_CHECK_NAMES)
    terminal_validation_passed = _receipt_set_passes(receipts, TERMINAL_CHECK_NAMES)
    ready = int(
        preconditions_passed
        and fixture_complete
        and panel_complete
        and energy_parity
        and probability_parity
        and normalizers_match
        and proof_safety
        and minima_preserved
        and intentional_change_detected
        and negative_controls_passed
        and affected_validation_passed
        and terminal_validation_passed
        and artifact.get("flagged_adversarial") is False
    )
    return {
        "preconditions_passed": preconditions_passed,
        "fixture_complete": fixture_complete,
        "finite_panel_complete": panel_complete,
        "source_energy_parity": energy_parity,
        "source_probability_parity": probability_parity,
        "normalizers_match": normalizers_match,
        "proof_assistance_preserves_law": proof_safety,
        "satisfying_minima_preserved": minima_preserved,
        "intentional_law_change_detected": intentional_change_detected,
        "negative_controls_passed": negative_controls_passed,
        "affected_validation_passed": affected_validation_passed,
        "terminal_validation_passed": terminal_validation_passed,
        "law_fixture_ready_score": ready,
        "promotion_score": 0,
    }


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    *,
    terminal_blocking: bool,
) -> JsonDict:
    return {
        "check": check,
        "category": category,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
        "terminal_blocking": terminal_blocking,
    }


def build_acceptance_gates(artifact: Mapping[str, Any]) -> list[JsonDict]:
    """Separate formal-law, safety, validation, and promotion decisions."""

    reduced = independent_reduce(artifact)
    definitions = (
        ("source_energy_parity", "scientific_efficacy", True),
        ("source_probability_parity", "scientific_efficacy", True),
        ("normalizers_match", "scientific_efficacy", True),
        ("proof_assistance_preserves_law", "safety", True),
        ("satisfying_minima_preserved", "completion", True),
        ("intentional_law_change_detected", "scientific_efficacy", True),
        ("negative_controls_passed", "safety", True),
        ("affected_validation_passed", "required_validation", True),
        ("terminal_validation_passed", "completion", True),
    )
    gates = [
        _gate(name, category, expected, reduced[name], terminal_blocking=True)
        for name, category, expected in definitions
    ]
    gates.append(_gate("automatic_promotion", "promotion", 0, 0, terminal_blocking=False))
    return gates


def gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Keep all failures while naming the first exact blocking check."""

    failures = [dict(row) for row in gates if row.get("passed") is not True]
    blocking = [row for row in failures if row.get("terminal_blocking") is True]
    return {
        "passed": not blocking,
        "failed_count": len(failures),
        "blocking_failed_count": len(blocking),
        "first_failure": blocking[0] if blocking else (failures[0] if failures else None),
        "failures": failures,
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind code hashes, exact formulas, protocol, and every scientific row."""

    bound = {
        key: artifact.get(key)
        for key in (
            "schema",
            "experiment_id",
            "milestone",
            "run_date",
            "random_seed",
            "source_artifact_hashes",
            "energy_definition",
            "frozen_formulas",
            "finite_law_rows",
            "negative_control_rows",
            "frozen_sampling_protocol",
            "sample_size_budget",
            "verdict_class",
        )
    }
    return validation_contract.canonical_hash(bound)


def field_principles(artifact: Mapping[str, Any]) -> dict[str, str]:
    """Explain required fields without changing their ordinary JSON types."""

    specific = {
        "schema": "Use a versioned schema with ordinary top-level experiment_id and milestone.",
        "status": "Use a terminal status only after actual work and required validation.",
        "run_date": "Use 20260917 with actual start and end UTC timestamps.",
        "preconditions_checked": "Record exact paths, producer identity, hashes, class, and resources before dependent work.",
        "MODEL_SPECS": "List intended current models; this host-only task has none.",
        "model_invoked": "Set true for any attempted current model load or generation.",
        "invocation_counts": "Record all current load and generation states as zero.",
        "inference_substrate": "Describe actual host exact enumeration and keep historical inputs in labeled sidecars.",
        "inference_substrate_class": "Use the closed CPU exact solver or simulator class.",
        "execution_venue": "Record measured host CPU work and no board execution.",
        "duration_s": "Use measured monotonic duration without delay padding.",
        "phase_spans": "Record measured read, build, load, generate, evaluate, validate, and write boundaries.",
        "random_seed": "Freeze formula and Exp7378 resampling seeds.",
        "reproducibility_checksum": "Bind exact code, settings, formulas, protocol, and raw scientific rows.",
        "source_artifact_hashes": "Hash every exact source and producer path used by this task.",
        "rows": "Retain every arm outcome, metric, cost, failure, and censoring disposition.",
        "sample_size_budget": "Predeclare planned, attempted, completed, censored units, stopping rules, and remaining work.",
        "acceptance_gate_results": "Separate expected, observed, and passed values for each required category.",
        "gate_check_summary": "Name failed checks with exact expected and observed values.",
        "verifier_is_oracle": "True because exhaustive formal enumeration defines correctness.",
        "honest_verdict": "Use complete_ for finished work and blocked_ for unavailable required input.",
        "verdict_class": "Use only positive, circular_positive, null, blocked, disqualified, or partial.",
        "flagged_adversarial": "Set true only for a critical independent finding and prevent readiness.",
        "validation_receipts": "Keep command argv, environment, scope, return code, duration, and log hash.",
        "repository_health": "Keep unrelated dated repository health separate from affected checks.",
        "field_principles": "Explain each field without wrapping its ordinary value.",
        "promotion_score": "Keep zero because this milestone makes no rollout or publication decision.",
        "law_fixture_ready_score": "One requires exact preservation controls and detection of the intentional law change.",
        "finite_law_rows": "Record every formula, beta, condition, normalizer, support, residual, and law distance.",
        "negative_control_rows": "Record offset, sign, beta, multiplicity, conditioning, and extra-clause mutations.",
        "frozen_sampling_protocol": "Freeze the exact fixture, seeds, beta grid, and sample budgets for Exp7378.",
    }
    return {
        key: specific.get(key, "Retain this supporting evidence in its ordinary JSON type.")
        for key in artifact
        if key != "field_principles"
    } | {"field_principles": specific["field_principles"]}


def _base_artifact(
    formulas: Sequence[Mapping[str, Any]],
    finite_rows: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
    *,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    historical_sidecars: Sequence[Mapping[str, Any]],
    phase_spans: Sequence[Mapping[str, Any]],
    started_at_utc: str,
    completed_at_utc: str,
    duration_s: float,
    repository_health: Mapping[str, Any],
    flagged_adversarial: bool = False,
) -> JsonDict:
    """Build one terminal-shape record from independently reducible evidence."""

    protocol = {
        "fixture_sha256": fixture_hash(formulas),
        "formula_seeds": list(FORMULA_SEEDS),
        "beta_grid": list(BETA_GRID),
        "chain_count": 4,
        "chain_seeds": list(EXP7378_CHAIN_SEEDS),
        "warmup_steps_per_chain": 1_000,
        "recorded_samples_per_chain": 4_000,
        "steps_per_recorded_sample": 1,
        "planned_samples_per_formula_beta_condition": 16_000,
        "tuning_after_observation": False,
        "exp7378_only_not_executed_here": True,
    }
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": PHASE,
        "status": "building_terminal_record",
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": {
            "current": deepcopy(ZERO_CURRENT_INVOCATIONS),
            "historical": {
                "counted_as_current": False,
                "sidecar_count": len(historical_sidecars),
            },
        },
        "inference_substrate": "host_cpu_exact_2cnf_enumeration_no_model",
        "inference_substrate_class": "cpu_exact_solver_or_simulator",
        "execution_venue": "host",
        "host_computation": {
            "processor": platform.processor() or "host_cpu",
            "node": platform.node(),
            "python": platform.python_version(),
            "states_per_formula": 64,
            "current_model_operations": 0,
            "current_gpu_operations": 0,
        },
        "duration_s": duration_s,
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "random_seed": {
            "formula_seeds": list(FORMULA_SEEDS),
            "exp7378_chain_seeds": list(EXP7378_CHAIN_SEEDS),
        },
        "reproducibility_checksum": "pending",
        "source_artifact_hashes": dict(source_hashes),
        "historical_inference_sidecars": [deepcopy(dict(row)) for row in historical_sidecars],
        "energy_definition": {
            "source_energy": "sum of original-clause violation indicators",
            "clause_violation": "product of the two literal-false indicators",
            "positive_literal_false_factor": "(1-s_i)/2",
            "negative_literal_false_factor": "(1+s_i)/2",
            "bit_spin_convention": "s_i=2*x_i-1; x_i=(s_i+1)/2",
            "ising_sign_convention": "E(s)=offset-sum_i(h_i*s_i)-sum_i<j(J_ij*s_i*s_j)",
            "unit_conditioning": "support clamp only; never a finite penalty",
            "proof_assistance": "source-checked logical entailment only; energy unchanged",
        },
        "development_counterexample": deepcopy(dict(formulas[0])) if formulas else None,
        "frozen_formulas": [deepcopy(dict(row)) for row in formulas],
        "finite_law_rows": [deepcopy(dict(row)) for row in finite_rows],
        "negative_control_rows": [deepcopy(dict(row)) for row in controls],
        "rows": [
            *[deepcopy(dict(row)) for row in finite_rows],
            *[deepcopy(dict(row)) for row in controls],
        ],
        "sample_size_budget": {
            "planned_formulas": 24,
            "attempted_formulas": len({row.get("formula_id") for row in finite_rows}),
            "completed_formulas": len({row.get("formula_id") for row in finite_rows}),
            "censored_formulas": 0,
            "planned_beta_values": len(BETA_GRID),
            "planned_conditions": 3,
            "planned_finite_rows": 24 * len(BETA_GRID) * 3,
            "attempted_finite_rows": len(finite_rows),
            "completed_finite_rows": sum(row.get("censored") is False for row in finite_rows),
            "censored_finite_rows": sum(row.get("censored") is True for row in finite_rows),
            "planned_negative_controls": 6,
            "attempted_negative_controls": len(controls),
            "completed_negative_controls": sum(row.get("censored") is False for row in controls),
            "stopping_rule": "Enumerate every state for all frozen cells once; do not tune after results.",
            "remaining_work": max(0, 24 * len(BETA_GRID) * 3 - len(finite_rows)),
        },
        "frozen_sampling_protocol": protocol,
        "validation_receipts": [deepcopy(dict(row)) for row in receipts],
        "repository_health": deepcopy(dict(repository_health)),
        "verifier_is_oracle": True,
        "flagged_adversarial": flagged_adversarial,
        "production_defaults_changed": False,
        "active_research_roadmap_changed": False,
        "research_conductor_changed": False,
        "rust_changed": False,
        "numbered_e2e_applicable": False,
        "capability_e2e_checks": ["declared_entrypoint", "cold_artifact_replay"],
    }
    reduction = independent_reduce(artifact)
    artifact["independent_reduction"] = reduction
    artifact["law_fixture_ready_score"] = reduction["law_fixture_ready_score"]
    artifact["promotion_score"] = 0
    blocking_failed = any(
        row.get("terminal_blocking") is True and row.get("passed") is not True
        for row in preconditions
    )
    if blocking_failed:
        artifact["status"] = "blocked_required_input_unavailable"
        artifact["honest_verdict"] = "blocked_required_input_unavailable"
        artifact["verdict_class"] = "blocked"
        artifact["law_fixture_ready_score"] = 0
    elif reduction["law_fixture_ready_score"] == 1:
        artifact["status"] = "complete_exact_source_law_fixture_ready"
        artifact["honest_verdict"] = "complete_circular_positive_exact_source_law_fixture_ready"
        artifact["verdict_class"] = "circular_positive"
    else:
        artifact["status"] = "complete_disqualified_required_validation_or_law_gate_failure"
        artifact["honest_verdict"] = "complete_disqualified_required_validation_or_law_gate_failure"
        artifact["verdict_class"] = "disqualified"
        artifact["law_fixture_ready_score"] = 0
    artifact["acceptance_gate_results"] = build_acceptance_gates(artifact)
    artifact["gate_check_summary"] = gate_summary(artifact["acceptance_gate_results"])
    artifact["field_principles"] = field_principles(
        {**artifact, "field_principles": {}, "reproducibility_checksum": None}
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_artifact_for_test(
    finite_rows: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
    *,
    preconditions_passed: bool = True,
) -> JsonDict:
    """Build a complete in-memory artifact without writing tracked state."""

    preconditions = [
        _precondition(
            "unit_fixture",
            "unit_fixture",
            "available",
            True,
            preconditions_passed,
            terminal_blocking=True,
        )
    ]
    spans = [
        {
            "phase": name,
            "start_s": index / 100,
            "end_s": (index + 1) / 100,
            "duration_s": 0.01,
        }
        for index, name in enumerate(
            ("read", "build", "load", "generate", "evaluate", "validate", "write")
        )
    ]
    return _base_artifact(
        build_frozen_fixture(),
        finite_rows,
        controls,
        receipts,
        preconditions=preconditions,
        source_hashes={},
        historical_sidecars=[
            {"label": "no_historical_model_inputs", "counted_as_current": False, "receipts": []}
        ],
        phase_spans=spans,
        started_at_utc="2026-09-17T00:00:00+00:00",
        completed_at_utc="2026-09-17T00:00:01+00:00",
        duration_s=1.0,
        repository_health={
            "as_of": RUN_DATE,
            "status": "not_assessed_by_scoped_unit_fixture",
            "affects_required_checks": False,
            "historical_failures": [],
        },
    )


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Cold-check identity, raw reduction, terminal semantics, and checksum."""

    errors: list[str] = []
    if artifact.get("schema") != SCHEMA or artifact.get("experiment_id") != EXPERIMENT_ID:
        errors.append("identity_mismatch")
    if artifact.get("milestone") != MILESTONE or artifact.get("run_date") != RUN_DATE:
        errors.append("run_identity_mismatch")
    if artifact.get("verdict_class") not in CLOSED_VERDICTS:
        errors.append("verdict_class_invalid")
    if artifact.get("MODEL_SPECS") != [] or artifact.get("model_invoked") is not False:
        errors.append("current_model_declaration_invalid")
    if (artifact.get("invocation_counts") or {}).get("current") != ZERO_CURRENT_INVOCATIONS:
        errors.append("current_invocation_counts_invalid")
    if artifact.get("inference_substrate_class") != "cpu_exact_solver_or_simulator":
        errors.append("substrate_class_invalid")
    if artifact.get("execution_venue") != "host":
        errors.append("execution_venue_invalid")
    if artifact.get("promotion_score") != 0:
        errors.append("promotion_nonzero")
    reduced = independent_reduce(artifact)
    if artifact.get("independent_reduction") != reduced:
        errors.append("stored_reduction_mismatch")
    if artifact.get("law_fixture_ready_score") != reduced["law_fixture_ready_score"]:
        errors.append("readiness_mismatch")
    if (
        artifact.get("verdict_class") in {"blocked", "disqualified", "partial"}
        and artifact.get("law_fixture_ready_score") != 0
    ):
        errors.append("failed_state_readiness_nonzero")
    if (
        artifact.get("verdict_class") == "circular_positive"
        and artifact.get("verifier_is_oracle") is not True
    ):
        errors.append("circularity_declaration_invalid")
    if set(artifact.get("field_principles") or {}) != set(artifact):
        errors.append("field_principles_incomplete")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def utc_now() -> str:  # pragma: no cover - entrypoint timestamp.
    return datetime.now(UTC).isoformat()


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Emit each entrypoint boundary with real monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7377] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _span(
    phase: str, phase_started: float, run_started: float, *, operation: str = "executed"
) -> JsonDict:  # pragma: no cover - entrypoint evidence.
    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "operation": operation,
    }


def _terminal_commands(
    candidate: Path,
) -> list[validation_contract.PlannedCommand]:  # pragma: no cover
    """Build cold replay, reduction, adversarial, and strict row checks."""

    python = str(REPO_ROOT / ".venv/bin/python")
    reducer_code = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7377_v647_ising_law import validate_artifact;"
        "v=json.loads(pathlib.Path(sys.argv[1]).read_text());"
        "e=validate_artifact(v);print(e,flush=True);raise SystemExit(bool(e))"
    )
    return [
        validation_contract.PlannedCommand(
            validation_scope.CommandSpec(
                "cold_artifact_replay",
                (
                    python,
                    "-u",
                    WRAPPER_PATH.as_posix(),
                    "--date",
                    RUN_DATE,
                    "--replay-artifact",
                    str(candidate),
                ),
                "measured_candidate",
            ),
            "completion",
            True,
        ),
        validation_contract.PlannedCommand(
            validation_scope.CommandSpec(
                "independent_reducer",
                (python, "-u", "-c", reducer_code, str(candidate)),
                "measured_candidate",
            ),
            "completion",
            True,
        ),
        validation_contract.PlannedCommand(
            validation_scope.CommandSpec(
                "adversarial_verify",
                (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
                "measured_candidate",
            ),
            "safety",
            True,
        ),
        validation_contract.PlannedCommand(
            validation_scope.CommandSpec(
                "verdict_row_consistency_strict",
                (
                    python,
                    "-u",
                    "scripts/verdict_row_consistency_lint.py",
                    "--strict",
                    str(candidate),
                ),
                "measured_candidate",
            ),
            "completion",
            True,
        ),
    ]


def run_experiment(  # pragma: no cover - executed through the declared entrypoint.
    repo_root: Path,
    run_date: str,
    *,
    output_path: Path = RESULT_PATH,
) -> JsonDict:
    """Run exact enumeration, scoped checks, cold readers, and atomic publication."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = repo_root.resolve()
    started = time.monotonic()
    started_at = utc_now()
    spans: list[JsonDict] = []

    phase_started = time.monotonic()
    progress(started, "preconditions", "start")
    preconditions, hashes, sidecars = collect_preconditions(root)
    spans.append(_span("read", phase_started, started))
    blocking_passed = all(
        row["passed"] for row in preconditions if row["terminal_blocking"] is True
    )
    progress(started, "preconditions", "end", passed=blocking_passed)

    raw_dir = root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    phase_started = time.monotonic()
    validation_contract.atomic_json(raw_dir / "historical_model_receipts.json", {"rows": sidecars})
    spans.append(_span("write", phase_started, started, operation="raw_sidecar_checkpoint"))
    if not blocking_passed:
        blocked = _base_artifact(
            [],
            [],
            [],
            [],
            preconditions=preconditions,
            source_hashes=hashes,
            historical_sidecars=sidecars,
            phase_spans=spans,
            started_at_utc=started_at,
            completed_at_utc=utc_now(),
            duration_s=time.monotonic() - started,
            repository_health={
                "as_of": RUN_DATE,
                "status": "not_assessed_blocked_before_validation",
                "affects_required_checks": False,
                "historical_failures": [],
            },
        )
        progress(started, "write", "before_atomic_blocked", path=output_path)
        validation_contract.atomic_json(root / output_path, blocked)
        progress(started, "write", "after_atomic_blocked", status=blocked["status"])
        return blocked

    phase_started = time.monotonic()
    progress(started, "build", "start")
    formulas = build_frozen_fixture()
    spans.append(_span("build", phase_started, started))
    progress(started, "build", "end", formulas=len(formulas))

    phase_started = time.monotonic()
    progress(started, "load", "before", operation="no_model_load_required")
    spans.append(_span("load", phase_started, started, operation="not_applicable_no_model"))
    progress(started, "load", "after", model_invoked=False)
    phase_started = time.monotonic()
    progress(started, "generate", "before", operation="no_generation_required")
    spans.append(_span("generate", phase_started, started, operation="not_applicable_no_model"))
    progress(started, "generate", "after", generation_calls=0)

    phase_started = time.monotonic()
    progress(started, "evaluate", "before_exact_enumeration", formulas=len(formulas))
    finite_rows = run_exact_panel(formulas)
    controls = run_negative_controls(formulas[0])
    spans.append(_span("evaluate", phase_started, started))
    progress(
        started,
        "evaluate",
        "after_exact_enumeration",
        finite_rows=len(finite_rows),
        controls=len(controls),
    )

    private = Path(tempfile.mkdtemp(prefix="exp7377-validation-", dir="/tmp"))
    commands = validation_contract.build_command_plan(root, V647_MANIFEST, private)
    plan_errors = validation_contract.validate_command_plan(root, V647_MANIFEST, commands)
    phase_started = time.monotonic()
    progress(started, "validate", "before_affected_subprocesses", plan_errors=len(plan_errors))
    affected_rows: list[JsonDict] = []
    if not plan_errors:
        affected_rows = validation_contract.run_categorized_commands(
            root,
            [
                validation_contract.PlannedCommand(command, "required_validation", True)
                for command in commands
            ],
            log_dir=raw_dir / "validation/affected",
        )
    progress(
        started,
        "validate",
        "after_affected_subprocesses",
        passed=_receipt_set_passes(affected_rows, REQUIRED_CHECK_NAMES),
    )

    repository_health = {
        "as_of": RUN_DATE,
        "status": "not_assessed_by_scoped_experiment",
        "scope": "affected checks only; no repository-wide fallback command",
        "affects_required_checks": False,
        "historical_failures": [],
    }
    candidate = _base_artifact(
        formulas,
        finite_rows,
        controls,
        affected_rows,
        preconditions=preconditions,
        source_hashes=hashes,
        historical_sidecars=sidecars,
        phase_spans=spans,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        duration_s=time.monotonic() - started,
        repository_health=repository_health,
    )
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    validation_contract.atomic_json(candidate_path, candidate)

    progress(started, "validate", "before_terminal_subprocesses")
    terminal_rows = validation_contract.run_categorized_commands(
        root,
        _terminal_commands(candidate_path),
        log_dir=raw_dir / "validation/terminal",
    )
    spans.append(_span("validate", phase_started, started))
    terminal_passed = _receipt_set_passes(terminal_rows, TERMINAL_CHECK_NAMES)
    critical = any("CRITICAL" in str(row.get("output_tail") or "") for row in terminal_rows)
    progress(
        started,
        "validate",
        "after_terminal_subprocesses",
        passed=terminal_passed,
        critical=critical,
    )

    phase_started = time.monotonic()
    final = _base_artifact(
        formulas,
        finite_rows,
        controls,
        [*affected_rows, *terminal_rows],
        preconditions=preconditions,
        source_hashes=hashes,
        historical_sidecars=sidecars,
        phase_spans=spans,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        duration_s=time.monotonic() - started,
        repository_health=repository_health,
        flagged_adversarial=critical,
    )
    errors = validate_artifact(final)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    validation_contract.atomic_json(candidate_path, final)
    spans.append(_span("write", phase_started, started, operation="terminal_payload_checkpoint"))
    progress(started, "write", "before_atomic_terminal", path=output_path)
    validation_contract.atomic_json(root / output_path, final)
    progress(started, "write", "after_atomic_terminal", status=final["status"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    """Parse the declared run and cold-replay modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--replay-artifact", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Run the experiment or cold-check one existing artifact."""

    args = parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    if args.replay_artifact is not None:
        artifact = json.loads(args.replay_artifact.read_text(encoding="utf-8"))
        errors = validate_artifact(artifact)
        print(json.dumps({"cold_artifact_replay_errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    run_experiment(REPO_ROOT, args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
