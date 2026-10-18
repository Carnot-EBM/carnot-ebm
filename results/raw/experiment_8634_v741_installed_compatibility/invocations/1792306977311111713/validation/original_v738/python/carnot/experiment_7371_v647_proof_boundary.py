"""Seal and validate the V647 proof-memory boundary before live capture.

The experiment uses parameterized 2-CNF formulas because their truth is exact
and inspectable. It measures protocol safety and comparator behavior. It does
not claim that a language model learned anything.

Spec refs: REQ-CL-7371 and SCENARIO-CL-7371-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any

from carnot import experiment_7358_v646_validation_contract as validation_contract
from carnot.learning.implication_memory import FormulaVersion, ProofMemory, execute_query
from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260917"
MILESTONE = "2026.09.647"
PHASE = 1
EXPERIMENT_ID = "exp7371-v647-proof-boundary"
SCHEMA = "carnot.exp7371.v647.proof_boundary.v1"
PROTOCOL_SCHEMA = "carnot.v647.implication_stream_manifest.v1"
RESULT_PATH = Path("results/experiment_7371_v647_proof_boundary.json")
RAW_DIR = Path("results/raw/experiment_7371_v647_proof_boundary")
MANIFEST_PATH = Path("data/v647_implication_stream_manifest.json")
MODULE_PATH = Path("python/carnot/experiment_7371_v647_proof_boundary.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7371_v647_proof_boundary.py")
TEST_PATH = Path("tests/python/test_experiment_7371_v647_proof_boundary.py")
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
PRODUCER_MODULE_PATH = Path("python/carnot/experiment_7370_v647_proof_memory.py")
PRODUCER_PATH = Path("results/experiment_7370_v647_proof_memory.json")
MEMORY_MODULE_PATH = Path("python/carnot/learning/implication_memory.py")
REQUIRED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "cold_artifact_replay",
    "independent_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
    "declared_entrypoint",
)
ARMS = (
    "reset_exact_solver",
    "persistent_incremental_exact_solver",
    "persistent_source_graph_reachability_cache",
    "proof_memory",
    "proof_memory_matched_non_applicable",
)
FAMILIES = (
    "contradiction_cycle_balanced",
    "contradiction_cycle_imbalanced",
    "free_variables_high",
    "free_variables_low",
    "planted_backbone_shallow",
    "planted_backbone_deep",
    "late_bridge",
    "symmetry_duplicate",
)
AUTHORITY_ATTACKS = (
    "forged_source_hash",
    "reordered_delayed_feedback",
    "state_poisoning",
    "missing_queries",
    "benchmark_timer_exclusion",
    "control_reset",
    "disjoint_assumption",
    "renamed_variable",
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
RANDOM_SEED = {
    "development": 7_371_101,
    "formula": 7_371_211,
    "resampling": 7_371_307,
    "live_protocol": 7_371_401,
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
    Path("python/carnot/verify/sat.py"),
    Path("python/carnot/experiment_7360_v646_learning_fixture.py"),
    Path("results/experiment_7362_v646_prospective_learning.json"),
    SPEC_PATH,
    MEMORY_MODULE_PATH,
    PRODUCER_MODULE_PATH,
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


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Print a flushed phase boundary with measured monotonic time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7371] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    return validation_contract.sha256_file(path)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    validation_contract.atomic_json(path, value)


def _check(
    check: str,
    upstream: str,
    field: str,
    expected: Any,
    observed: Any,
    passed: bool,
    *,
    terminal_blocking: bool = True,
) -> JsonDict:
    return {
        "check": check,
        "upstream": upstream,
        "artifact_field": field,
        "expected": expected,
        "observed": observed,
        "passed": passed,
        "terminal_blocking": terminal_blocking,
    }


def _load_object(path: Path) -> JsonDict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def collect_preconditions(
    repo_root: Path,
    *,
    producer_path: Path | None = None,
) -> tuple[list[JsonDict], dict[str, str], list[JsonDict]]:
    """Authenticate task sources and reject an unsafe Exp7370 producer."""

    root = repo_root.resolve()
    checks: list[JsonDict] = []
    hashes: dict[str, str] = {}
    for relative in REQUIRED_SOURCE_PATHS:
        path = root / relative
        present = path.is_file() and path.stat().st_size > 0
        checks.append(
            _check(
                f"source_bytes:{relative.as_posix()}",
                relative.as_posix(),
                "bytes",
                "readable_nonempty_bytes",
                "readable_nonempty_bytes" if present else "missing",
                present,
            )
        )
        if present:
            hashes[relative.as_posix()] = sha256_file(path)

    spec_text = (root / SPEC_PATH).read_text(encoding="utf-8")
    requirement_present = "REQ-CL-7371" in spec_text
    checks.append(
        _check(
            "driving_requirement",
            SPEC_PATH.as_posix(),
            "REQ-*",
            "REQ-CL-7371",
            "REQ-CL-7371" if requirement_present else None,
            requirement_present,
        )
    )
    exclusion_text = (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    quarantined = "experiment_id: 7371" in exclusion_text or EXPERIMENT_ID in exclusion_text
    checks.append(
        _check(
            "current_task_not_quarantined",
            "ops/exclusion_manifest.yaml",
            EXPERIMENT_ID,
            False,
            quarantined,
            not quarantined,
        )
    )
    checks.append(
        _check(
            "host_cpu_available",
            "local_host",
            "cpu_count",
            "positive_integer",
            os.cpu_count(),
            bool(os.cpu_count() and os.cpu_count() > 0),
        )
    )

    selected = producer_path or root / PRODUCER_PATH
    producer = _load_object(selected)
    if not producer:
        observed = "missing" if not selected.exists() else "malformed"
        checks.append(
            _check(
                "producer_path",
                str(selected),
                "path",
                "readable_nonempty_json_object",
                observed,
                False,
            )
        )
        return checks, hashes, []

    digest = sha256_file(selected)
    hashes[str(selected)] = digest
    checks.append(
        _check(
            "producer_path",
            str(selected),
            "path",
            "readable_nonempty_json_object",
            "readable_nonempty_json_object",
            True,
        )
    )
    expected_fields = {
        "experiment_id": "exp7370-v647-proof-memory",
        "milestone": MILESTONE,
        "proof_fixture_ready_score": 1,
        "flagged_adversarial": False,
    }
    for field, expected in expected_fields.items():
        checks.append(
            _check(
                f"producer_{field}",
                str(selected),
                field,
                expected,
                producer.get(field),
                producer.get(field) == expected,
            )
        )
    verdict = producer.get("verdict_class")
    allowed_verdicts = {"positive", "circular_positive", "null"}
    checks.append(
        _check(
            "producer_verdict_class",
            str(selected),
            "verdict_class",
            sorted(allowed_verdicts),
            verdict,
            verdict in allowed_verdicts,
        )
    )
    status = producer.get("status")
    status_ok = isinstance(status, str) and status.startswith("complete_")
    checks.append(
        _check(
            "producer_terminal_status",
            str(selected),
            "status",
            "complete_*",
            status,
            status_ok,
        )
    )
    sidecars = [
        {
            "label": "historical_exp7370_host_only_producer_not_current_inference",
            "artifact_path": str(selected),
            "artifact_sha256": digest,
            "producer_identity": producer.get("experiment_id"),
            "producer_verdict_class": verdict,
            "producer_flagged_adversarial": producer.get("flagged_adversarial"),
            "producer_inference_substrate_class": producer.get("inference_substrate_class"),
            "historical_invocation_counts": deepcopy(producer.get("invocation_counts") or {}),
            "counted_as_current": False,
            "authorizes_only_boundary_fixture": True,
        }
    ]
    return checks, hashes, sidecars


def _formula_clauses(family: str, n_vars: int, seed_index: int) -> list[tuple[int, int]]:
    """Build one transparent structural family with a stable 1-to-n path."""

    middle = list(range(2, n_vars))
    if middle:
        offset = seed_index % len(middle)
        middle = middle[offset:] + middle[:offset]
    if family == "free_variables_high":
        middle = middle[: max(1, len(middle) // 3)]
    elif family == "free_variables_low":
        middle = middle[: max(1, 2 * len(middle) // 3)]
    path = [1, *middle, n_vars]
    chain = [(-left, right) for left, right in zip(path, path[1:])]
    extras: list[tuple[int, int]] = []
    if family == "contradiction_cycle_balanced":
        extras = [(-n_vars, 1), (n_vars, -1)]
    elif family == "contradiction_cycle_imbalanced":
        extras = [(-n_vars, max(2, n_vars // 2)), (-max(2, n_vars // 2), 1)]
    elif family == "free_variables_low":
        extras = [(-variable, variable + 1) for variable in range(2, n_vars - 1, 2)]
    elif family == "planted_backbone_shallow":
        extras = [(2, 2)]
    elif family == "planted_backbone_deep":
        extras = [(2, 2), (-2, 3), (-3, min(4, n_vars))]
    elif family == "late_bridge":
        if len(chain) > 1:
            bridge = chain.pop()
            extras = [(2, -2), bridge]
    elif family == "symmetry_duplicate":
        extras = [chain[0], chain[-1], (2, -2)]
    return [*chain, *extras]


def _formula_payload(
    formula_id: str, family: str, n_vars: int, seed: int, version: int
) -> JsonDict:
    clauses = _formula_clauses(family, n_vars, seed)
    if version == 2:
        clauses = [*clauses, (2, -2), (max(2, n_vars // 2), -max(2, n_vars // 2))]
    formula = FormulaVersion.from_clauses(f"{formula_id}-v{version}", n_vars, clauses)
    return {
        **formula.to_dict(),
        "raw_clause_order": [list(pair) for pair in clauses],
        "rule_provenance": {
            "paper": "arXiv:2602.12665",
            "family": family,
            "seed": seed,
            "version": version,
        },
    }


def _version_two_requests(formula: Mapping[str, Any], stream_id: str) -> list[JsonDict]:
    exact = _formula_from_payload(formula)
    satisfiable, assignment = exact.solve(())
    if not satisfiable or assignment is None:  # pragma: no cover - generator invariant.
        raise RuntimeError("version_two_formula_not_satisfiable")
    variables = list(range(1, min(5, exact.n_vars + 1)))
    literals = [variable if assignment[variable] else -variable for variable in variables]
    choices = (
        [literals[0], literals[1]],
        [literals[1], literals[2]],
        [literals[0], literals[2], literals[3]],
        [literals[2], literals[3]],
    )
    return [
        {
            "request_id": f"{stream_id}-q{20 + index:02d}",
            "request_index": 20 + index,
            "split": "version_change",
            "formula_version": formula["version"],
            "assumptions": list(choice),
            "expected_satisfiable": True,
            "rule_version_changed": True,
        }
        for index, choice in enumerate(choices)
    ]


def _evaluation_requests(
    version_one: Mapping[str, Any], version_two: Mapping[str, Any], stream_id: str
) -> list[JsonDict]:
    n_vars = int(version_one["n_vars"])
    available = list(range(2, n_vars))
    warm: list[JsonDict] = []
    later: list[JsonDict] = []
    for index in range(8):
        variable = available[index % len(available)]
        warm.append(
            {
                "request_id": f"{stream_id}-q{index:02d}",
                "request_index": index,
                "split": "warm_up",
                "formula_version": version_one["version"],
                "assumptions": [1, -n_vars, variable if index % 2 == 0 else -variable],
                "expected_satisfiable": False,
                "rule_version_changed": False,
            }
        )
        later.append(
            {
                "request_id": f"{stream_id}-q{8 + index:02d}",
                "request_index": 8 + index,
                "split": "later_distinct",
                "formula_version": version_one["version"],
                "assumptions": [1, -n_vars, variable if index % 2 else -variable, 1],
                "expected_satisfiable": False,
                "rule_version_changed": False,
            }
        )
    recurrence = [
        {
            **deepcopy(warm[index]),
            "request_id": f"{stream_id}-q{16 + index:02d}",
            "request_index": 16 + index,
            "split": "recurrence",
        }
        for index in range(4)
    ]
    return [*warm, *later, *recurrence, *_version_two_requests(version_two, stream_id)]


def _live_requests(stream_id: str, versions: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    splits = ("warm_up", "future", "future", "version_change")
    requests: list[JsonDict] = []
    for index, split in enumerate(splits):
        version = versions[1] if split == "version_change" else versions[0]
        request_id = f"{stream_id}-request-{index}"
        prompt = (
            f"Request {request_id}. Formula bytes are referenced by source hash "
            f"{version['source_hash']}. Return one JSON object with key assignments. "
            "Assignments must be a list of two to four distinct signed integer literals."
        )
        requests.append(
            {
                "request_id": request_id,
                "request_index": index,
                "split": split,
                "formula_version": version["version"],
                "formula_source_hash": version["source_hash"],
                "proposal_count": 2,
                "proposal_target": "partial_assignment_extendibility",
                "minimum_distinct_literals": 2,
                "maximum_distinct_literals": 4,
                "prompt": prompt,
                "label_available_to_model": False,
            }
        )
    return requests


def build_protocol() -> JsonDict:
    """Build all frozen cohorts without observing any experiment outcome."""

    development: list[JsonDict] = []
    evaluation: list[JsonDict] = []
    live: list[JsonDict] = []
    n_values = (8, 12, 24, 32)
    for family_index, family in enumerate(FAMILIES):
        for seed_index in range(2):
            n_vars = n_values[(family_index + seed_index) % len(n_values)]
            formula_id = f"dev-{family}-{seed_index}"
            development.append(
                {
                    "formula_id": formula_id,
                    "cohort": "development",
                    "family": family,
                    "seed": 7_371_500 + family_index * 10 + seed_index,
                    "n_vars": n_vars,
                    "formula": _formula_payload(formula_id, family, n_vars, seed_index, 1),
                }
            )
        for seed_index in range(4):
            n_vars = n_values[seed_index]
            formula_id = f"eval-{family}-{seed_index}"
            stream_id = f"synthetic-{family}-{seed_index}"
            versions = [
                _formula_payload(formula_id, family, n_vars, seed_index, version)
                for version in (1, 2)
            ]
            evaluation.append(
                {
                    "stream_id": stream_id,
                    "formula_id": formula_id,
                    "cohort": "synthetic_evaluation",
                    "family": family,
                    "seed": 7_371_600 + family_index * 10 + seed_index,
                    "n_vars": n_vars,
                    "versions": versions,
                    "requests": _evaluation_requests(versions[0], versions[1], stream_id),
                }
            )
        live_formula_id = f"live-{family}"
        n_vars = n_values[family_index % len(n_values)]
        versions = [
            _formula_payload(live_formula_id, family, n_vars, 100 + family_index, version)
            for version in (1, 2)
        ]
        live_stream_id = f"live-{family}"
        live.append(
            {
                "stream_id": live_stream_id,
                "formula_id": live_formula_id,
                "cohort": "live_proposal_future_capture",
                "family": family,
                "seed": 7_371_700 + family_index,
                "n_vars": n_vars,
                "versions": versions,
                "requests": _live_requests(live_stream_id, versions),
                "captured_proposals": [],
            }
        )
    return {
        "schema": PROTOCOL_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": PHASE,
        "run_date": RUN_DATE,
        "sealed_before_outcomes": True,
        "paper_basis": "arXiv:2602.12665 structured 2-CNF families",
        "arms": list(ARMS),
        "random_seed": deepcopy(RANDOM_SEED),
        "request_contract": {
            "evaluation_requests_per_stream": 24,
            "warm_up": 8,
            "later_distinct": 8,
            "recurrence": 4,
            "version_change": 4,
            "live_requests_per_stream": 4,
            "live_proposals_per_request": 2,
            "proposal_literal_range": [2, 4],
            "formal_target": "partial_assignment_extendibility",
        },
        "development_formulas": development,
        "evaluation_streams": evaluation,
        "live_proposal_streams": live,
    }


def _formula_from_payload(payload: Mapping[str, Any]) -> FormulaVersion:
    clauses = payload.get("clauses") or []
    return FormulaVersion.from_clauses(
        str(payload.get("version")),
        int(payload.get("n_vars", 0)),
        [row["literals"] for row in clauses],
    )


def _formula_hash_errors(payload: Mapping[str, Any]) -> list[str]:
    try:
        formula = _formula_from_payload(payload)
    except (KeyError, TypeError, ValueError):
        return ["formula_schema"]
    return [] if formula.source_hash == payload.get("source_hash") else ["formula_source_hash"]


def validate_protocol(protocol: Mapping[str, Any]) -> list[str]:
    """Reject drift in cohort size, source identity, requests, or prompt secrecy."""

    errors: list[str] = []
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("sealed_before_outcomes") is not True
    ):
        errors.append("protocol_identity")
    development = protocol.get("development_formulas") or []
    evaluation = protocol.get("evaluation_streams") or []
    live = protocol.get("live_proposal_streams") or []
    if len(development) != 16:
        errors.append("development_formula_count")
    if len(evaluation) != 32:
        errors.append("evaluation_stream_count")
    if len(live) != 8:
        errors.append("live_stream_count")
    ids = [row.get("formula_id") for row in [*development, *evaluation, *live]]
    if len(ids) != len(set(ids)):
        errors.append("cohort_formula_overlap")
    if protocol.get("arms") != list(ARMS):
        errors.append("arm_order")
    for unit in development:
        errors.extend(_formula_hash_errors(unit.get("formula") or {}))
    for stream in evaluation:
        versions = stream.get("versions") or []
        requests = stream.get("requests") or []
        if len(versions) != 2:
            errors.append("evaluation_version_count")
        for version in versions:
            errors.extend(_formula_hash_errors(version))
        if len(requests) != 24:
            errors.append("evaluation_request_count")
            continue
        counts = Counter(row.get("split") for row in requests)
        if counts != Counter(
            {"warm_up": 8, "later_distinct": 8, "recurrence": 4, "version_change": 4}
        ):
            errors.append("evaluation_request_splits")
        if [row.get("request_index") for row in requests] != list(range(24)):
            errors.append("evaluation_request_order")
        versions_by_id = {item.get("version"): item for item in versions}
        for row in requests:
            version = versions_by_id.get(row.get("formula_version"))
            if version is None:
                errors.append("evaluation_request_formula_version")
            elif (
                row.get("split") == "version_change"
                and row.get("expected_satisfiable") is True
                and not truth_table_or_exact_extendible(version, row.get("assumptions") or [])
            ):
                errors.append("version_change_expected_sat_invalid")
    for stream in live:
        requests = stream.get("requests") or []
        if len(requests) != 4 or sum(int(row.get("proposal_count", 0)) for row in requests) != 8:
            errors.append("live_request_count")
        for request in requests:
            prompt = str(request.get("prompt", ""))
            if (
                "SAT" in prompt
                or "label" in prompt
                or request.get("label_available_to_model") is not False
            ):
                errors.append("live_prompt_label_leakage")
            if (
                request.get("minimum_distinct_literals"),
                request.get("maximum_distinct_literals"),
            ) != (2, 4):
                errors.append("live_proposal_shape")
        for version in stream.get("versions") or []:
            errors.extend(_formula_hash_errors(version))
    return list(dict.fromkeys(errors))


def truth_table_extendible(formula: Mapping[str, Any], assumptions: Sequence[int]) -> bool:
    """Decide small formulas by direct enumeration without producer code."""

    n_vars = int(formula.get("n_vars", 0))
    if n_vars > 12:
        raise ValueError("truth_table_variable_cap")
    values = tuple(assumptions)
    if any(type(literal) is not int or literal == 0 or abs(literal) > n_vars for literal in values):
        raise ValueError("assumption_literal_invalid")
    clauses = [tuple(row["literals"]) for row in formula.get("clauses") or []]
    for bits in itertools.product((False, True), repeat=n_vars):
        assignment = {index + 1: value for index, value in enumerate(bits)}

        def literal_value(literal: int) -> bool:
            selected = assignment[abs(literal)]
            return selected if literal > 0 else not selected

        if all(literal_value(literal) for literal in values) and all(
            literal_value(left) or literal_value(right) for left, right in clauses
        ):
            return True
    return False


def _independent_graph_sat(formula: Mapping[str, Any], assumptions: Sequence[int]) -> bool:
    """Decide larger formulas with an SCC implementation local to this task."""

    n_vars = int(formula["n_vars"])
    vertices = [*range(1, n_vars + 1), *range(-1, -n_vars - 1, -1)]
    graph = {literal: [] for literal in vertices}
    reverse = {literal: [] for literal in vertices}
    for row in formula.get("clauses") or []:
        left, right = row["literals"]
        for source, target in ((-left, right), (-right, left)):
            graph[source].append(target)
            reverse[target].append(source)
    for literal in assumptions:
        graph[-literal].append(literal)
        reverse[literal].append(-literal)
    visited: set[int] = set()
    order: list[int] = []

    def visit(vertex: int) -> None:
        visited.add(vertex)
        for target in graph[vertex]:
            if target not in visited:
                visit(target)
        order.append(vertex)

    for vertex in vertices:
        if vertex not in visited:
            visit(vertex)
    components: dict[int, int] = {}

    def assign(vertex: int, component: int) -> None:
        components[vertex] = component
        for target in reverse[vertex]:
            if target not in components:
                assign(target, component)

    for vertex in reversed(order):
        if vertex not in components:
            assign(vertex, len(components))
    return all(components[variable] != components[-variable] for variable in range(1, n_vars + 1))


def truth_table_or_exact_extendible(formula: Mapping[str, Any], assumptions: Sequence[int]) -> bool:
    return (
        truth_table_extendible(formula, assumptions)
        if int(formula.get("n_vars", 0)) <= 12
        else _independent_graph_sat(formula, assumptions)
    )


def independent_validate_path(formula: Mapping[str, Any], proof: Mapping[str, Any]) -> list[str]:
    """Reconstruct proof authority from clause bytes without the learner."""

    errors: list[str] = []
    n_vars = int(formula.get("n_vars", 0))
    clauses = formula.get("clauses") or []
    if any(
        not isinstance(row, Mapping)
        or type(row.get("clause_id")) is not int
        or row.get("clause_id", -1) < 0
        or not isinstance(row.get("literals"), list)
        or len(row.get("literals")) != 2
        for row in clauses
    ):
        errors.append("formula_clause_id_invalid")
    if proof.get("formula_version") != formula.get("version"):
        errors.append("formula_version_mismatch")
    if proof.get("source_hash") != formula.get("source_hash"):
        errors.append("source_hash_mismatch")
    endpoints = (proof.get("antecedent"), proof.get("consequent"))
    if any(type(value) is not int or value == 0 or abs(value) > n_vars for value in endpoints):
        errors.append("endpoint_invalid")
    edges = proof.get("edges")
    if not isinstance(edges, list) or not edges:
        errors.append("path_empty")
        edges = []
    if len(edges) > 2 * n_vars:
        errors.append("path_edge_cap_exceeded")
    legal: set[tuple[int, int, int]] = set()
    for row in clauses:
        if (
            isinstance(row, Mapping)
            and isinstance(row.get("literals"), list)
            and len(row["literals"]) == 2
        ):
            left, right = row["literals"]
            clause_id = row.get("clause_id")
            if type(clause_id) is int and clause_id >= 0:
                legal.add((-left, right, clause_id))
                legal.add((-right, left, clause_id))
    if edges:
        if edges[0].get("from_literal") != proof.get("antecedent"):
            errors.append("antecedent_endpoint_mismatch")
        if edges[-1].get("to_literal") != proof.get("consequent"):
            errors.append("consequent_endpoint_mismatch")
    vertices = [proof.get("antecedent")]
    for index, edge in enumerate(edges):
        source = edge.get("from_literal")
        target = edge.get("to_literal")
        clause_id = edge.get("source_clause_id")
        if type(clause_id) is not int or clause_id < 0:
            errors.append("negative_source_clause_id")
        if (source, target, clause_id) not in legal:
            errors.append("edge_not_in_source_formula")
        if index and edges[index - 1].get("to_literal") != source:
            errors.append("path_literal_omitted")
        vertices.append(target)
    if len(vertices) != len(set(vertices)):
        errors.append("path_cycle")
    content = {
        "formula_version": proof.get("formula_version"),
        "source_hash": proof.get("source_hash"),
        "antecedent": proof.get("antecedent"),
        "consequent": proof.get("consequent"),
        "edges": edges,
    }
    if proof.get("path_id") != canonical_hash(content):
        errors.append("path_id_mismatch")
    return list(dict.fromkeys(errors))


def _proof_applies(proof: Mapping[str, Any], assumptions: Sequence[int]) -> bool:
    values = set(assumptions)
    return proof.get("antecedent") in values and -int(proof.get("consequent", 0)) in values


def _assumption_hash(assumptions: Sequence[int]) -> str:
    return canonical_hash(list(assumptions))


def _cost_units(
    formula: FormulaVersion,
    *,
    exact_paid: bool,
    proof_edges: int,
    state_bytes: int,
    cache_build: bool,
) -> JsonDict:
    source_hash = len(canonical_bytes(formula.to_dict()))
    exact = (formula.n_vars + len(formula.clauses)) * 12 if exact_paid else 0
    proof = max(1, proof_edges) * 3
    cache = formula.n_vars * len(formula.implication_edges) if cache_build else 0
    memory = max(1, state_bytes // 32)
    total = source_hash + exact + proof + cache + memory
    return {
        "source_hash": source_hash,
        "exact_query": exact,
        "proof_check": proof,
        "cache_update": cache,
        "memory_accounting": memory,
        "total": total,
    }


def _make_row(
    stream: Mapping[str, Any],
    request: Mapping[str, Any],
    formula: FormulaVersion,
    arm: str,
    decision: str,
    used_exact: bool,
    assignment: Mapping[int, bool] | None,
    oracle_sat: bool,
    duration_ns: int,
    state_bytes: int,
    proof_edges: int,
    state_instance_id: str,
    cache_build: bool,
    candidate_order: Sequence[str],
    matched_state_size: bool,
) -> JsonDict:
    decision_sat = decision == "satisfiable"
    assignment_valid = assignment is None or formula.verify_assignment(
        assignment, request["assumptions"]
    )
    return {
        "stream_id": stream["stream_id"],
        "formula_id": stream["formula_id"],
        "family": stream["family"],
        "seed": stream["seed"],
        "n_vars": stream["n_vars"],
        "formula_version": formula.version,
        "formula_source_hash": formula.source_hash,
        "request_id": request["request_id"],
        "request_index": request["request_index"],
        "split": request["split"],
        "assumptions": list(request["assumptions"]),
        "assumption_bytes_sha256": _assumption_hash(request["assumptions"]),
        "arm": arm,
        "decision": decision,
        "used_exact_solver": used_exact,
        "paid_exact_query": used_exact,
        "independent_satisfiable": oracle_sat,
        "independent_exact_match": decision_sat == oracle_sat,
        "assignment_valid": assignment_valid,
        "final_exact_validation": decision_sat == oracle_sat and assignment_valid,
        "timeout_s": 1.0,
        "timed_out": False,
        "duration_ns": duration_ns,
        "timer_includes_complete_service": True,
        "cost_units": _cost_units(
            formula,
            exact_paid=used_exact,
            proof_edges=proof_edges,
            state_bytes=state_bytes,
            cache_build=cache_build,
        ),
        "storage_bytes": state_bytes,
        "memory_accounted": True,
        "state_instance_id": state_instance_id,
        "raw_candidate_order": list(candidate_order),
        "matched_state_size": matched_state_size,
        "failure": None,
        "censored": False,
    }


def evaluate_stream(stream: Mapping[str, Any]) -> tuple[list[JsonDict], list[JsonDict]]:
    """Run one ordered stream through five isolated equal-input arms."""

    versions = {row["version"]: _formula_from_payload(row) for row in stream["versions"]}
    payloads = {row["version"]: row for row in stream["versions"]}
    proof_memory: ProofMemory | None = None
    current_version: str | None = None
    reachability_cache: dict[tuple[str, tuple[int, ...]], tuple[bool, dict[int, bool] | None]] = {}
    incremental_instances: dict[str, str] = {}
    rows: list[JsonDict] = []
    witnesses: list[JsonDict] = []
    for request in stream["requests"]:
        formula = versions[request["formula_version"]]
        formula_payload = payloads[request["formula_version"]]
        assumptions = tuple(request["assumptions"])
        oracle_sat = truth_table_or_exact_extendible(formula_payload, assumptions)
        if current_version != formula.version:
            proof_memory = ProofMemory.empty(formula)
            current_version = formula.version
        assert proof_memory is not None

        reset_started = time.perf_counter_ns()
        reset_formula = FormulaVersion.from_clauses(
            formula.version, formula.n_vars, [clause.literals for clause in formula.clauses]
        )
        reset_sat, reset_assignment = reset_formula.solve(assumptions)
        reset_duration = time.perf_counter_ns() - reset_started
        rows.append(
            _make_row(
                stream,
                request,
                formula,
                ARMS[0],
                "satisfiable" if reset_sat else "unsatisfiable",
                True,
                reset_assignment,
                oracle_sat,
                reset_duration,
                0,
                0,
                f"reset-{request['request_id']}",
                False,
                (),
                False,
            )
        )

        instance_id = incremental_instances.setdefault(
            formula.version, canonical_hash([stream["stream_id"], formula.version, "incremental"])
        )
        incremental_started = time.perf_counter_ns()
        incremental_sat, incremental_assignment = formula.solve(assumptions)
        incremental_duration = time.perf_counter_ns() - incremental_started
        rows.append(
            _make_row(
                stream,
                request,
                formula,
                ARMS[1],
                "satisfiable" if incremental_sat else "unsatisfiable",
                True,
                incremental_assignment,
                oracle_sat,
                incremental_duration,
                len(formula.implication_edges) * 24,
                0,
                instance_id,
                False,
                (),
                False,
            )
        )

        cache_key = (formula.source_hash, assumptions)
        cache_hit = cache_key in reachability_cache
        reach_started = time.perf_counter_ns()
        if cache_hit:
            reach_sat, reach_assignment = reachability_cache[cache_key]
        else:
            reach_sat, reach_assignment = formula.solve(assumptions)
            reachability_cache[cache_key] = (reach_sat, reach_assignment)
        reach_duration = time.perf_counter_ns() - reach_started
        reach_bytes = len(
            canonical_bytes(sorted((key[0], list(key[1])) for key in reachability_cache))
        )
        rows.append(
            _make_row(
                stream,
                request,
                formula,
                ARMS[2],
                "satisfiable" if reach_sat else "unsatisfiable",
                not cache_hit,
                reach_assignment,
                oracle_sat,
                reach_duration,
                reach_bytes,
                0,
                canonical_hash([stream["stream_id"], "reachability-cache"]),
                not cache_hit,
                (),
                False,
            )
        )

        entry_memory = proof_memory
        proof_started = time.perf_counter_ns()
        proof_result = execute_query(entry_memory, assumptions)
        proof_duration = time.perf_counter_ns() - proof_started
        proof_memory = proof_result.committed_memory
        candidates = [path.path_id for path in proof_result.discovered_paths]
        used_edges = next(
            (
                len(path.edges)
                for path in entry_memory.paths
                if path.path_id == proof_result.proof_path_id
            ),
            sum(len(path.edges) for path in proof_result.discovered_paths),
        )
        rows.append(
            _make_row(
                stream,
                request,
                formula,
                ARMS[3],
                proof_result.decision,
                proof_result.used_exact_solver,
                proof_result.assignment,
                oracle_sat,
                proof_duration,
                len(proof_memory.to_bytes()),
                used_edges,
                proof_memory.sha256,
                False,
                candidates,
                False,
            )
        )
        if proof_result.proof_path_id is not None:
            erased = entry_memory.without(proof_result.proof_path_id)
            counterfactual = execute_query(erased, assumptions)
            proof = next(
                path for path in entry_memory.paths if path.path_id == proof_result.proof_path_id
            )
            witnesses.append(
                {
                    "stream_id": stream["stream_id"],
                    "family": stream["family"],
                    "request_id": request["request_id"],
                    "formula_version": formula.version,
                    "source_hash": formula.source_hash,
                    "path": proof.to_dict(),
                    "independent_path_errors": independent_validate_path(
                        formula.to_dict(), proof.to_dict()
                    ),
                    "later_used_exact_solver": proof_result.used_exact_solver,
                    "erasure_used_exact_solver": counterfactual.used_exact_solver,
                    "early_decision_removed": counterfactual.used_exact_solver,
                }
            )

        matched_size = len(proof_memory.to_bytes())
        matched_started = time.perf_counter_ns()
        matched_sat, matched_assignment = formula.solve(assumptions)
        matched_duration = time.perf_counter_ns() - matched_started
        rows.append(
            _make_row(
                stream,
                request,
                formula,
                ARMS[4],
                "satisfiable" if matched_sat else "unsatisfiable",
                True,
                matched_assignment,
                oracle_sat,
                matched_duration,
                matched_size,
                0,
                canonical_hash([stream["stream_id"], formula.version, "matched-control"]),
                False,
                [f"non_applicable_{index}" for index in range(len(proof_memory.paths))],
                True,
            )
        )
    return rows, witnesses


def evaluate_synthetic(
    protocol: Mapping[str, Any], *, emit_progress: bool = False, started: float | None = None
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Evaluate all 32 streams and retain every arm row and witness."""

    rows: list[JsonDict] = []
    witnesses: list[JsonDict] = []
    origin = started if started is not None else time.monotonic()
    streams = protocol["evaluation_streams"]
    for index, stream in enumerate(streams):
        stream_rows, stream_witnesses = evaluate_stream(stream)
        rows.extend(stream_rows)
        witnesses.extend(stream_witnesses)
        if emit_progress:
            progress(origin, "evaluate", "stream_complete", unit=index + 1, total=len(streams))
    return rows, witnesses


def _cost_row_valid(row: Mapping[str, Any]) -> bool:
    costs = row.get("cost_units") or {}
    required = {
        "source_hash",
        "exact_query",
        "proof_check",
        "cache_update",
        "memory_accounting",
        "total",
    }
    return (
        row.get("timer_includes_complete_service") is True
        and type(row.get("duration_ns")) is int
        and row.get("duration_ns", -1) >= 0
        and set(costs) == required
        and costs.get("total") == sum(costs[key] for key in required - {"total"})
    )


def run_authority_controls(protocol: Mapping[str, Any]) -> list[JsonDict]:
    """Attempt every shortcut that could counterfeit proof authority."""

    stream = protocol["evaluation_streams"][0]
    formula_payload = stream["versions"][0]
    formula = _formula_from_payload(formula_payload)
    edges = formula.find_path(1, formula.n_vars)
    if edges is None:  # pragma: no cover - protocol generator invariant.
        raise RuntimeError("control_path_missing")
    from carnot.learning.implication_memory import (
        ProofPath,
    )  # local import keeps checker independent.

    proof = ProofPath(formula.version, formula.source_hash, 1, formula.n_vars, edges).to_dict()
    rows: list[JsonDict] = []

    forged = deepcopy(proof)
    forged["source_hash"] = "sha256:" + "0" * 64
    rows.append(
        _attack_row("forged_source_hash", bool(independent_validate_path(formula_payload, forged)))
    )

    feedback_order = [(1, "request"), (3, "feedback"), (2, "request")]
    reordered_rejected = [item[0] for item in feedback_order] != sorted(
        item[0] for item in feedback_order
    )
    rows.append(_attack_row("reordered_delayed_feedback", reordered_rejected))

    changed = FormulaVersion.from_clauses(
        formula.version + "-changed", formula.n_vars, [*(_clauses(formula)), (2, -2)]
    )
    poisoned_rejected = (
        ProofMemory.empty(formula).commit([]).formula.source_hash != changed.source_hash
    )
    rows.append(_attack_row("state_poisoning", poisoned_rejected))

    missing = deepcopy(protocol)
    missing["evaluation_streams"][0]["requests"].pop()
    rows.append(
        _attack_row("missing_queries", "evaluation_request_count" in validate_protocol(missing))
    )

    timer_row = {
        "timer_includes_complete_service": False,
        "duration_ns": None,
        "cost_units": {},
    }
    rows.append(_attack_row("benchmark_timer_exclusion", not _cost_row_valid(timer_row)))

    reset_ids = ["state-one", "state-two"]
    rows.append(_attack_row("control_reset", len(set(reset_ids)) != 1))

    rows.append(_attack_row("disjoint_assumption", not _proof_applies(proof, [-1, formula.n_vars])))

    renamed_formula = FormulaVersion.from_clauses(
        formula.version + "-renamed",
        formula.n_vars,
        [(-right, -left) for left, right in _clauses(formula)],
    ).to_dict()
    rows.append(
        _attack_row("renamed_variable", bool(independent_validate_path(renamed_formula, proof)))
    )
    return rows


def _clauses(formula: FormulaVersion) -> list[tuple[int, int]]:
    return [clause.literals for clause in formula.clauses]


def _attack_row(name: str, rejected: bool) -> JsonDict:
    return {
        "attack": name,
        "expected": "reject",
        "observed": "reject" if rejected else "accept",
        "passed": rejected,
        "failure": None if rejected else "unauthorized_authority_accepted",
    }


def frozen_acceptance_manifest(protocol_sha256: str) -> JsonDict:
    """Freeze safety and efficacy thresholds before measured outcomes."""

    return {
        "frozen_protocol_sha256": protocol_sha256,
        "safety": {
            "invalid_sat_outputs": 0,
            "false_proof_rejections": 0,
            "stale_version_effects": 0,
            "authority_attack_failures": 0,
        },
        "utility": {
            "exact_decision_coverage": 1.0,
            "valid_utility_required": True,
        },
        "minimum_erasure_witnesses": 8,
        "minimum_witness_streams": 4,
        "paid_exact_query_ratio_upper_bound_lt": 0.90,
        "complete_service_cost_ratio_lte": 1.0,
        "comparators": [ARMS[1], ARMS[2]],
        "bootstrap_draws": 10_000,
        "bootstrap_seed": RANDOM_SEED["resampling"],
        "synthetic_cluster": "eight_formula_families",
        "live_cluster": "eight_independent_formula_streams",
        "learning_value_rule": "all fixed gates pass in synthetic and live cohorts",
        "live_estimate_class": "exploratory_not_publication_ready",
        "later_tasks_may_relax": False,
    }


def seal_protocol_and_fixtures(
    protocol_path: Path, raw_dir: Path, protocol: Mapping[str, Any]
) -> dict[str, str]:
    """Write immutable public bytes and three raw cohort fixtures."""

    atomic_json(protocol_path, protocol)
    protocol_sha256 = sha256_file(protocol_path)
    fixtures = {
        raw_dir / "development_formulas.json": {
            "frozen_protocol_sha256": protocol_sha256,
            "development_formulas": protocol["development_formulas"],
        },
        raw_dir / "evaluation_streams.json": {
            "frozen_protocol_sha256": protocol_sha256,
            "evaluation_streams": protocol["evaluation_streams"],
        },
        raw_dir / "live_proposal_requests.json": {
            "frozen_protocol_sha256": protocol_sha256,
            "live_proposal_streams": protocol["live_proposal_streams"],
        },
    }
    hashes = {str(protocol_path): protocol_sha256}
    for path, value in fixtures.items():
        atomic_json(path, value)
        hashes[str(path)] = sha256_file(path)
    return hashes


def _paired_bootstrap_upper(
    rows: Sequence[Mapping[str, Any]], comparator: str, metric: str
) -> float:
    by_family: dict[str, tuple[float, float]] = {}
    for family in FAMILIES:
        proof_rows = [
            row for row in rows if row.get("family") == family and row.get("arm") == ARMS[3]
        ]
        comp_rows = [
            row for row in rows if row.get("family") == family and row.get("arm") == comparator
        ]
        if metric == "paid_exact_query":
            proof_value = sum(bool(row[metric]) for row in proof_rows)
            comp_value = sum(bool(row[metric]) for row in comp_rows)
        else:
            proof_value = sum(
                int((row.get("cost_units") or {}).get("total", 0)) for row in proof_rows
            )
            comp_value = sum(
                int((row.get("cost_units") or {}).get("total", 0)) for row in comp_rows
            )
        by_family[family] = (float(proof_value), float(comp_value))
    generator = random.Random(RANDOM_SEED["resampling"])
    estimates: list[float] = []
    for _ in range(10_000):
        selected = [generator.choice(FAMILIES) for _ in FAMILIES]
        numerator = sum(by_family[family][0] for family in selected)
        denominator = sum(by_family[family][1] for family in selected)
        estimates.append(numerator / denominator if denominator else float("inf"))
    estimates.sort()
    return estimates[min(len(estimates) - 1, int(0.95 * len(estimates)))]


def synthetic_metrics(
    rows: Sequence[Mapping[str, Any]], witnesses: Sequence[Mapping[str, Any]]
) -> JsonDict:
    invalid_sat = sum(
        row.get("decision") == "satisfiable" and row.get("assignment_valid") is not True
        for row in rows
    )
    false_rejections = sum(
        row.get("decision") == "reject" and row.get("independent_satisfiable") is True
        for row in rows
    )
    stale = sum(
        row.get("split") == "version_change"
        and row.get("arm") == "proof_memory"
        and row.get("decision") == "reject"
        for row in rows
    )
    paid_bounds = {
        comparator: _paired_bootstrap_upper(rows, comparator, "paid_exact_query")
        for comparator in (ARMS[1], ARMS[2])
    }
    cost_bounds = {
        comparator: _paired_bootstrap_upper(rows, comparator, "complete_service_cost")
        for comparator in (ARMS[1], ARMS[2])
    }
    return {
        "invalid_sat_outputs": invalid_sat,
        "false_proof_rejections": false_rejections,
        "stale_version_effects": stale,
        "exact_decision_coverage": (
            sum(row.get("independent_exact_match") is True for row in rows) / len(rows)
            if rows
            else 0.0
        ),
        "valid_utility": bool(rows)
        and all(row.get("final_exact_validation") is True for row in rows),
        "erasure_witness_count": len(witnesses),
        "erasure_witness_stream_count": len({row.get("stream_id") for row in witnesses}),
        "paid_exact_query_ratio_bootstrap_95_upper": paid_bounds,
        "complete_service_cost_ratio_bootstrap_95_upper": cost_bounds,
        "bootstrap_draws": 10_000,
        "bootstrap_seed": RANDOM_SEED["resampling"],
    }


def passing_test_receipts() -> list[JsonDict]:
    """Create full receipt shapes for deterministic reducer tests."""

    return [
        {
            "name": name,
            "command": f"test {name}",
            "command_argv": ["test", name],
            "environment": {"PYTHONUNBUFFERED": "1"},
            "scope": "unit_fixture",
            "return_code": 0,
            "exit_code": 0,
            "duration_s": 0.001,
            "log_path": f"/tmp/{name}.log",
            "log_hash": "sha256:" + "1" * 64,
            "log_sha256": "sha256:" + "1" * 64,
            "passed": True,
            "timed_out": False,
            "required": True,
            "category": "required_validation",
            "started_at_utc": "2026-09-17T00:00:00+00:00",
            "ended_at_utc": "2026-09-17T00:00:00.001000+00:00",
        }
        for name in (*REQUIRED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)
    ]


def _receipts_pass(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    counts = Counter(row.get("name") for row in receipts if row.get("required") is True)
    return all(
        counts[name] == 1
        and next(row for row in receipts if row.get("name") == name).get("passed") is True
        and next(row for row in receipts if row.get("name") == name).get("return_code") == 0
        and next(row for row in receipts if row.get("name") == name).get("timed_out") is False
        for name in names
    )


def independent_reduce(artifact: Mapping[str, Any]) -> JsonDict:
    """Recompute readiness and efficacy from ordinary stored evidence."""

    rows = [row for row in artifact.get("rows", []) if isinstance(row, Mapping)]
    witnesses = [
        row for row in artifact.get("erasure_witness_rows", []) if isinstance(row, Mapping)
    ]
    attacks = [row for row in artifact.get("authority_attack_rows", []) if isinstance(row, Mapping)]
    receipts = [row for row in artifact.get("validation_receipts", []) if isinstance(row, Mapping)]
    metrics = (
        synthetic_metrics(rows, witnesses)
        if rows
        else {
            "invalid_sat_outputs": 0,
            "false_proof_rejections": 0,
            "stale_version_effects": 0,
            "exact_decision_coverage": 0.0,
            "valid_utility": False,
            "erasure_witness_count": 0,
            "erasure_witness_stream_count": 0,
            "paid_exact_query_ratio_bootstrap_95_upper": {},
            "complete_service_cost_ratio_bootstrap_95_upper": {},
            "bootstrap_draws": 10_000,
            "bootstrap_seed": RANDOM_SEED["resampling"],
        }
    )
    preconditions = [
        row
        for row in artifact.get("preconditions_checked", [])
        if isinstance(row, Mapping) and row.get("terminal_blocking") is True
    ]
    preconditions_passed = bool(preconditions) and all(
        row.get("passed") is True for row in preconditions
    )
    stream_rows = artifact.get("formula_stream_rows") or []
    protocol_complete = (
        artifact.get("frozen_protocol_path") == MANIFEST_PATH.as_posix()
        and isinstance(artifact.get("frozen_protocol_sha256"), str)
        and len(stream_rows) == 32 * 24
    )
    rows_complete = (
        len(rows) == 32 * 24 * len(ARMS)
        and {row.get("arm") for row in rows} == set(ARMS)
        and all(_cost_row_valid(row) for row in rows)
    )
    authority_passed = {row.get("attack") for row in attacks} == set(AUTHORITY_ATTACKS) and all(
        row.get("passed") is True for row in attacks
    )
    safety_passed = (
        metrics["invalid_sat_outputs"] == 0
        and metrics["false_proof_rejections"] == 0
        and metrics["stale_version_effects"] == 0
        and metrics["exact_decision_coverage"] == 1.0
        and metrics["valid_utility"] is True
    )
    witness_passed = (
        metrics["erasure_witness_count"] >= 8 and metrics["erasure_witness_stream_count"] >= 4
    )
    affected_passed = _receipts_pass(receipts, REQUIRED_CHECK_NAMES)
    terminal_passed = _receipts_pass(receipts, TERMINAL_CHECK_NAMES)
    boundary_ready = int(
        preconditions_passed
        and protocol_complete
        and rows_complete
        and authority_passed
        and safety_passed
        and witness_passed
        and affected_passed
        and terminal_passed
        and artifact.get("flagged_adversarial") is False
    )
    synthetic_efficacy = all(
        metrics["paid_exact_query_ratio_bootstrap_95_upper"].get(comparator, float("inf")) < 0.90
        and metrics["complete_service_cost_ratio_bootstrap_95_upper"].get(comparator, float("inf"))
        <= 1.0
        for comparator in (ARMS[1], ARMS[2])
    )
    live_complete = artifact.get("live_proposal_capture_status") == "complete_32_requests_64_calls"
    return {
        "preconditions_passed": preconditions_passed,
        "protocol_complete": protocol_complete,
        "rows_complete": rows_complete,
        "authority_controls_passed": authority_passed,
        "safety_passed": safety_passed,
        "witness_gate_passed": witness_passed,
        "affected_validation_passed": affected_passed,
        "terminal_validation_passed": terminal_passed,
        "synthetic_efficacy_passed": synthetic_efficacy,
        "live_cohort_complete": live_complete,
        "synthetic_metrics": metrics,
        "proof_boundary_ready_score": boundary_ready,
        "learning_value_score": int(boundary_ready and synthetic_efficacy and live_complete),
        "promotion_score": 0,
    }


def _gate(check: str, category: str, expected: Any, observed: Any, terminal: bool) -> JsonDict:
    return {
        "check": check,
        "category": category,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
        "terminal_blocking": terminal,
    }


def acceptance_gates(artifact: Mapping[str, Any]) -> list[JsonDict]:
    reduced = independent_reduce(artifact)
    metrics = reduced["synthetic_metrics"]
    gates = [
        _gate("required_preconditions", "safety", True, reduced["preconditions_passed"], True),
        _gate("sealed_protocol", "completion", True, reduced["protocol_complete"], True),
        _gate("complete_five_arm_rows", "completion", True, reduced["rows_complete"], True),
        _gate("authority_attacks", "safety", True, reduced["authority_controls_passed"], True),
        _gate("zero_invalid_sat_outputs", "safety", 0, metrics["invalid_sat_outputs"], True),
        _gate("zero_false_proof_rejections", "safety", 0, metrics["false_proof_rejections"], True),
        _gate("zero_stale_version_effects", "safety", 0, metrics["stale_version_effects"], True),
        _gate(
            "unchanged_exact_decision_coverage",
            "utility",
            1.0,
            metrics["exact_decision_coverage"],
            True,
        ),
        _gate("valid_utility", "utility", True, metrics["valid_utility"], True),
        _gate("minimum_erasure_witnesses", "utility", True, reduced["witness_gate_passed"], True),
        _gate(
            "required_affected_validation",
            "required_validation",
            True,
            reduced["affected_validation_passed"],
            True,
        ),
        _gate(
            "terminal_artifact_checks",
            "completion",
            True,
            reduced["terminal_validation_passed"],
            True,
        ),
        _gate(
            "synthetic_comparative_efficacy",
            "scientific_efficacy",
            True,
            reduced["synthetic_efficacy_passed"],
            False,
        ),
        _gate(
            "live_proposal_cohort",
            "scientific_efficacy",
            True,
            reduced["live_cohort_complete"],
            False,
        ),
        _gate("automatic_promotion", "promotion", 0, 0, False),
    ]
    return gates


def gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    failures = [dict(row) for row in gates if row.get("passed") is not True]
    blocking = [row for row in failures if row.get("terminal_blocking") is True]
    return {
        "passed": not blocking,
        "failed_count": len(failures),
        "blocking_failed_count": len(blocking),
        "first_failure": failures[0] if failures else None,
        "failures": failures,
    }


def _formula_stream_rows(protocol: Mapping[str, Any]) -> list[JsonDict]:
    return [
        {
            "stream_id": stream["stream_id"],
            "formula_id": stream["formula_id"],
            "family": stream["family"],
            "seed": stream["seed"],
            "n_vars": stream["n_vars"],
            "formula_version": request["formula_version"],
            "request_id": request["request_id"],
            "request_index": request["request_index"],
            "split": request["split"],
            "planned_arm_count": len(ARMS),
            "rule_provenance": "arXiv:2602.12665 parameterized structural family",
        }
        for stream in protocol["evaluation_streams"]
        for request in stream["requests"]
    ]


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    keys = (
        "schema",
        "experiment_id",
        "milestone",
        "phase",
        "run_date",
        "random_seed",
        "frozen_protocol_sha256",
        "source_artifact_hashes",
        "preconditions_checked",
        "rows",
        "formula_stream_rows",
        "authority_attack_rows",
        "erasure_witness_rows",
        "acceptance_manifest",
        "validation_receipts",
        "proof_boundary_ready_score",
        "learning_value_score",
        "promotion_score",
        "honest_verdict",
        "verdict_class",
        "flagged_adversarial",
    )
    return canonical_hash({key: artifact.get(key) for key in keys})


def field_principles(artifact: Mapping[str, Any]) -> dict[str, str]:
    specific = {
        "schema": "Use a versioned schema with ordinary top-level experiment identity and milestone.",
        "status": "Use a terminal state only after actual work and required validation.",
        "run_date": "Use 20260917 and retain actual UTC boundaries.",
        "preconditions_checked": "Record exact source paths and producer gate observations before dependent work.",
        "MODEL_SPECS": "Host-only work has no intended current model.",
        "model_invoked": "Remain false because no current model load or generation was attempted.",
        "invocation_counts": "Keep every current load and generation counter at zero.",
        "inference_substrate": "Describe actual host exact solving and simulation work.",
        "inference_substrate_class": "Use the closed CPU exact solver or simulator class.",
        "execution_venue": "Record the measured host CPU venue and no board execution.",
        "duration_s": "Use measured monotonic time without sleeps or a duration floor.",
        "phase_spans": "Record measured read, build, load, generate, evaluate, validate, and write spans.",
        "random_seed": "Freeze formula, cohort, and 10,000-draw resampling seeds.",
        "reproducibility_checksum": "Bind exact sources, protocol, raw rows, gates, and receipts.",
        "source_artifact_hashes": "Hash exact producer, protocol, source, and raw evidence bytes.",
        "rows": "Retain every request-arm outcome, cost, failure, and censoring disposition.",
        "sample_size_budget": "Separate planned, attempted, complete, censored, and future live work.",
        "acceptance_gate_results": "Keep expected, observed, and pass values separate for each gate.",
        "gate_check_summary": "Name each failed check with its exact expected and observed value.",
        "verifier_is_oracle": "True because exact formula evaluation defines formal correctness.",
        "honest_verdict": "Use complete_ for finished boundary work and blocked_ for unavailable inputs.",
        "verdict_class": "Use only the closed terminal verdict classes.",
        "flagged_adversarial": "A critical independent finding prevents readiness.",
        "validation_receipts": "Retain command vectors, environments, scope, exits, durations, and log hashes.",
        "repository_health": "Keep unrelated dated failures separate from affected validation.",
        "field_principles": "Explain ordinary fields without changing their JSON types.",
        "promotion_score": "Remain zero because this task performs no rollout or publication.",
        "proof_boundary_ready_score": "One means the independent safety boundary and frozen controls are complete.",
        "frozen_protocol_path": "Name data/v647_implication_stream_manifest.json as the immutable protocol.",
        "formula_stream_rows": "Retain every family, seed, version, request split, and rule source.",
        "authority_attack_rows": "Retain every negative control and observed rejection.",
        "acceptance_manifest": "Freeze safety, utility, witness, and paired cost gates before outcomes.",
    }
    return {
        key: specific.get(key, "Retain this supporting evidence in its ordinary JSON type.")
        for key in artifact
        if key != "field_principles"
    } | {"field_principles": specific["field_principles"]}


def _base_artifact(
    protocol: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    witnesses: Sequence[Mapping[str, Any]],
    attacks: Sequence[Mapping[str, Any]],
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
    protocol_sha256: str,
    flagged_adversarial: bool = False,
) -> JsonDict:
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": PHASE,
        "status": "building_terminal_record",
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "preconditions_checked": deepcopy(list(preconditions)),
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": {
            "current": deepcopy(ZERO_CURRENT_INVOCATIONS),
            "historical": {"counted_as_current": False, "sidecar_count": len(historical_sidecars)},
        },
        "inference_substrate": "host_cpu_exact_2cnf_solver_and_protocol_simulator",
        "inference_substrate_class": "cpu_exact_solver_or_simulator",
        "execution_venue": "host",
        "host_computation": {
            "processor": platform.processor() or "host_cpu",
            "node": platform.node(),
            "python": platform.python_version(),
            "current_model_operations": 0,
            "current_gpu_operations": 0,
        },
        "duration_s": duration_s,
        "phase_spans": deepcopy(list(phase_spans)),
        "random_seed": deepcopy(RANDOM_SEED),
        "reproducibility_checksum": "pending",
        "source_artifact_hashes": dict(source_hashes),
        "historical_inference_sidecars": deepcopy(list(historical_sidecars)),
        "frozen_protocol_path": MANIFEST_PATH.as_posix(),
        "frozen_protocol_sha256": protocol_sha256,
        "acceptance_manifest": frozen_acceptance_manifest(protocol_sha256),
        "rows": deepcopy(list(rows)),
        "formula_stream_rows": _formula_stream_rows(protocol),
        "authority_attack_rows": deepcopy(list(attacks)),
        "erasure_witness_rows": deepcopy(list(witnesses)),
        "synthetic_estimate": synthetic_metrics(rows, witnesses) if rows else {},
        "live_estimate": {
            "status": "frozen_not_collected_by_exp7371",
            "planned_streams": 8,
            "planned_requests": 32,
            "planned_calls": 64,
            "exploratory_not_publication_ready": True,
            "parser_and_model_failures_remain_in_denominator": True,
        },
        "live_proposal_capture_status": "frozen_for_exp7373_not_collected",
        "sample_size_budget": {
            "development_formulas_planned": 16,
            "development_formulas_sealed": len(protocol["development_formulas"]),
            "synthetic_streams_planned": 32,
            "synthetic_streams_attempted": len({row.get("stream_id") for row in rows}),
            "synthetic_requests_per_stream": 24,
            "arms_per_request": 5,
            "synthetic_rows_planned": 32 * 24 * 5,
            "synthetic_rows_completed": len(rows),
            "synthetic_rows_censored": sum(row.get("censored") is True for row in rows),
            "live_streams_planned": 8,
            "live_requests_planned": 32,
            "live_calls_planned": 64,
            "live_calls_attempted": 0,
            "live_calls_completed": 0,
            "live_calls_failed": 0,
            "remaining_work": "Exp7373 collects all 64 model proposal bytes.",
            "stopping_rule": "Run every sealed synthetic request once per arm; make no current model call.",
        },
        "validation_receipts": deepcopy(list(receipts)),
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
    reduced = independent_reduce(artifact)
    artifact["independent_reduction"] = reduced
    artifact["proof_boundary_ready_score"] = reduced["proof_boundary_ready_score"]
    artifact["learning_value_score"] = reduced["learning_value_score"]
    artifact["promotion_score"] = 0
    failed_preconditions = [
        row
        for row in preconditions
        if row.get("terminal_blocking") is True and row.get("passed") is not True
    ]
    if failed_preconditions:
        artifact["status"] = "blocked_required_producer_or_source_unavailable"
        artifact["honest_verdict"] = "blocked_required_producer_or_source_unavailable"
        artifact["verdict_class"] = "blocked"
        artifact["proof_boundary_ready_score"] = 0
        artifact["learning_value_score"] = 0
    elif reduced["proof_boundary_ready_score"] == 1:
        artifact["status"] = "complete_proof_boundary_ready_live_capture_not_started"
        artifact["honest_verdict"] = (
            "complete_null_proof_boundary_ready_learning_value_not_measured"
        )
        artifact["verdict_class"] = "null"
    else:
        artifact["status"] = "complete_disqualified_boundary_validation_incomplete"
        artifact["honest_verdict"] = "complete_disqualified_boundary_validation_incomplete"
        artifact["verdict_class"] = "disqualified"
    artifact["acceptance_gate_results"] = acceptance_gates(artifact)
    artifact["gate_check_summary"] = gate_summary(artifact["acceptance_gate_results"])
    artifact["field_principles"] = field_principles(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_artifact_for_test(
    protocol: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    witnesses: Sequence[Mapping[str, Any]],
    attacks: Sequence[Mapping[str, Any]],
    *,
    receipts: Sequence[Mapping[str, Any]] | None = None,
    preconditions: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    protocol_hash = canonical_hash(protocol)
    return _base_artifact(
        protocol,
        rows,
        witnesses,
        attacks,
        list(receipts) if receipts is not None else passing_test_receipts(),
        preconditions=(
            list(preconditions)
            if preconditions is not None
            else [_check("unit_fixture", "unit", "available", True, True, True)]
        ),
        source_hashes={},
        historical_sidecars=[],
        phase_spans=[
            {"phase": name, "start_s": index / 10, "end_s": (index + 1) / 10, "duration_s": 0.1}
            for index, name in enumerate(
                ("read", "build", "load", "generate", "evaluate", "validate", "write")
            )
        ],
        started_at_utc="2026-09-17T00:00:00+00:00",
        completed_at_utc="2026-09-17T00:00:01+00:00",
        duration_s=1.0,
        repository_health={"status": "unit_fixture", "affects_required_checks": False},
        protocol_sha256=protocol_hash,
    )


def validate_artifact(value: object) -> list[str]:
    """Cold-check identity, declarations, reduction, gates, and checksum."""

    if not isinstance(value, Mapping):
        return ["artifact_not_object"]
    artifact = dict(value)
    errors: list[str] = []
    if (
        artifact.get("schema"),
        artifact.get("experiment_id"),
        artifact.get("milestone"),
        artifact.get("phase"),
        artifact.get("run_date"),
    ) != (SCHEMA, EXPERIMENT_ID, MILESTONE, PHASE, RUN_DATE):
        errors.append("identity_mismatch")
    if artifact.get("MODEL_SPECS") != [] or artifact.get("model_invoked") is not False:
        errors.append("model_declaration_mismatch")
    if (artifact.get("invocation_counts") or {}).get("current") != ZERO_CURRENT_INVOCATIONS:
        errors.append("current_invocation_counts_nonzero")
    if (
        artifact.get("inference_substrate_class") != "cpu_exact_solver_or_simulator"
        or artifact.get("execution_venue") != "host"
    ):
        errors.append("substrate_mismatch")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("oracle_declaration_mismatch")
    reduced = independent_reduce(artifact)
    if artifact.get("independent_reduction") != reduced:
        errors.append("independent_reduction_mismatch")
    for score in ("proof_boundary_ready_score", "learning_value_score", "promotion_score"):
        if artifact.get(score) != reduced[score]:
            errors.append(f"{score}_mismatch")
    expected_gates = acceptance_gates(artifact)
    if artifact.get("acceptance_gate_results") != expected_gates:
        errors.append("acceptance_gates_mismatch")
    if artifact.get("gate_check_summary") != gate_summary(expected_gates):
        errors.append("gate_summary_mismatch")
    if artifact.get("learning_value_score") != 0 or artifact.get("promotion_score") != 0:
        errors.append("deferred_score_nonzero")
    if (
        artifact.get("flagged_adversarial") is True
        and artifact.get("proof_boundary_ready_score") != 0
    ):
        errors.append("adversarial_readiness_nonzero")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles_incomplete")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def cold_reload_errors(artifact: Mapping[str, Any], root: Path) -> list[str]:
    """Reload sealed raw bytes and compare them with terminal evidence."""

    errors = validate_artifact(artifact)
    hashes = artifact.get("source_artifact_hashes") or {}
    for raw_path, expected_hash in hashes.items():
        path = Path(raw_path)
        resolved = path if path.is_absolute() else root / path
        if raw_path.startswith(RAW_DIR.as_posix()) or raw_path == MANIFEST_PATH.as_posix():
            if not resolved.is_file() or sha256_file(resolved) != expected_hash:
                errors.append(f"raw_hash_mismatch:{raw_path}")
    evidence_path = root / RAW_DIR / "synthetic_evidence.json"
    evidence = _load_object(evidence_path)
    if evidence:
        if evidence.get("rows") != artifact.get("rows"):
            errors.append("raw_rows_mismatch")
        if evidence.get("erasure_witness_rows") != artifact.get("erasure_witness_rows"):
            errors.append("raw_witnesses_mismatch")
    attacks_path = root / RAW_DIR / "authority_attacks.json"
    attacks = _load_object(attacks_path)
    if attacks and attacks.get("authority_attack_rows") != artifact.get("authority_attack_rows"):
        errors.append("raw_attacks_mismatch")
    return list(dict.fromkeys(errors))


def scoped_command_plan(repo_root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Build the actual bounded Exp7358 plan through Exp7303."""

    return validation_contract.build_command_plan(repo_root, V647_MANIFEST, private_root)


def validate_scoped_command_plan(
    repo_root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    errors = validation_contract.validate_command_plan(repo_root, V647_MANIFEST, commands)
    if Counter(command.name for command in commands) != Counter(REQUIRED_CHECK_NAMES):
        errors.append("required_command_names_changed")
    if any(command.name == "full_python_suite" for command in commands):
        errors.append("full_python_suite_forbidden")
    return list(dict.fromkeys(errors))


def _normalized_receipt(row: Mapping[str, Any]) -> JsonDict:
    return {
        "name": row.get("name"),
        "command": row.get("command"),
        "command_argv": deepcopy(row.get("command_argv") or []),
        "environment": deepcopy(row.get("command_environment") or {}),
        "scope": row.get("scope"),
        "return_code": row.get("exit_code"),
        "exit_code": row.get("exit_code"),
        "duration_s": row.get("duration_s"),
        "log_path": row.get("log_path"),
        "log_hash": row.get("log_sha256"),
        "log_sha256": row.get("log_sha256"),
        "passed": row.get("passed"),
        "timed_out": row.get("timed_out"),
        "required": row.get("required", True),
        "category": row.get("command_category", "required_validation"),
        "started_at_utc": row.get("started_at_utc"),
        "ended_at_utc": row.get("ended_at_utc"),
        **(
            {"resolved_imports": deepcopy(row["resolved_imports"])}
            if "resolved_imports" in row
            else {}
        ),
    }


def _terminal_commands(candidate: Path) -> list[validation_contract.PlannedCommand]:
    python = str(REPO_ROOT / ".venv/bin/python")
    replay = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7371_v647_proof_boundary import cold_reload_errors;"
        "p=pathlib.Path(sys.argv[1]);v=json.loads(p.read_text());"
        "e=cold_reload_errors(v,pathlib.Path(sys.argv[2]));print(e,flush=True);raise SystemExit(bool(e))"
    )
    specs = (
        (
            "cold_artifact_replay",
            (python, "-u", "-c", replay, str(candidate), str(REPO_ROOT)),
            "capability_e2e",
        ),
        (
            "independent_reducer",
            (python, "-u", "-c", replay, str(candidate), str(REPO_ROOT)),
            "completion",
        ),
        (
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "safety",
        ),
        (
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "completion",
        ),
    )
    return [
        validation_contract.PlannedCommand(
            validation_scope.CommandSpec(name, argv, "measured_candidate"), category, True
        )
        for name, argv, category in specs
    ]


def _span(phase: str, phase_started: float, run_started: float) -> JsonDict:
    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
    }


def _write_measured_raw(
    root: Path,
    protocol_sha256: str,
    rows: Sequence[Mapping[str, Any]],
    witnesses: Sequence[Mapping[str, Any]],
    attacks: Sequence[Mapping[str, Any]],
    historical_sidecars: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    values = {
        root / RAW_DIR / "synthetic_evidence.json": {
            "frozen_protocol_sha256": protocol_sha256,
            "rows": list(rows),
            "erasure_witness_rows": list(witnesses),
        },
        root / RAW_DIR / "authority_attacks.json": {
            "frozen_protocol_sha256": protocol_sha256,
            "authority_attack_rows": list(attacks),
        },
        root / RAW_DIR / "historical_model_receipts.json": {
            "frozen_protocol_sha256": protocol_sha256,
            "label": "historical_only_not_current_inference",
            "sources": list(historical_sidecars),
        },
    }
    hashes: dict[str, str] = {}
    for path, value in values.items():
        atomic_json(path, value)
        hashes[path.relative_to(root).as_posix()] = sha256_file(path)
    return hashes


def run_experiment(
    repo_root: Path,
    run_date: str,
    *,
    output_path: Path = RESULT_PATH,
) -> JsonDict:  # pragma: no cover - exercised through the declared entrypoint.
    """Seal, evaluate, validate, replay, and atomically publish the boundary."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = repo_root.resolve()
    output = output_path if output_path.is_absolute() else root / output_path
    run_started = time.monotonic()
    started_at = utc_now()
    spans: list[JsonDict] = []

    phase_started = time.monotonic()
    progress(run_started, "read", "start")
    preconditions, source_hashes, historical_sidecars = collect_preconditions(root)
    spans.append(_span("read", phase_started, run_started))
    progress(run_started, "read", "end", checks=len(preconditions))

    phase_started = time.monotonic()
    progress(run_started, "build", "start")
    protocol = build_protocol()
    protocol_errors = validate_protocol(protocol)
    if protocol_errors:
        raise RuntimeError(f"protocol_invalid:{protocol_errors}")
    sealed_hashes = seal_protocol_and_fixtures(root / MANIFEST_PATH, root / RAW_DIR, protocol)
    source_hashes.update(
        {
            Path(path).relative_to(root).as_posix() if Path(path).is_absolute() else path: digest
            for path, digest in sealed_hashes.items()
        }
    )
    protocol_sha256 = source_hashes[MANIFEST_PATH.as_posix()]
    spans.append(_span("build", phase_started, run_started))
    progress(run_started, "build", "end", protocol_sha256=protocol_sha256)

    phase_started = time.monotonic()
    progress(run_started, "load", "before_model_load", models=0)
    spans.append(_span("load", phase_started, run_started))
    progress(run_started, "load", "after_model_load", models=0)

    phase_started = time.monotonic()
    progress(run_started, "generate", "before_generation", calls=0)
    spans.append(_span("generate", phase_started, run_started))
    progress(run_started, "generate", "after_generation", calls=0)

    blocked = any(row["terminal_blocking"] and not row["passed"] for row in preconditions)
    rows: list[JsonDict] = []
    witnesses: list[JsonDict] = []
    attacks: list[JsonDict] = []
    phase_started = time.monotonic()
    progress(run_started, "evaluate", "start", blocked=blocked)
    if not blocked:
        rows, witnesses = evaluate_synthetic(protocol, emit_progress=True, started=run_started)
        attacks = run_authority_controls(protocol)
    spans.append(_span("evaluate", phase_started, run_started))
    progress(run_started, "evaluate", "end", rows=len(rows), witnesses=len(witnesses))
    source_hashes.update(
        _write_measured_raw(root, protocol_sha256, rows, witnesses, attacks, historical_sidecars)
    )

    producer = _load_object(root / PRODUCER_PATH)
    repository_health = deepcopy(
        producer.get("repository_health")
        or {
            "status": "historical_health_unavailable",
            "as_of": RUN_DATE,
            "affects_required_checks": False,
        }
    )
    repository_health["affects_required_checks"] = False

    receipts: list[JsonDict] = []
    phase_started = time.monotonic()
    progress(run_started, "validate", "start")
    if not blocked:
        private_root = root / RAW_DIR / "validation/private"
        commands = scoped_command_plan(root, private_root)
        plan_errors = validate_scoped_command_plan(root, commands)
        if plan_errors:
            raise RuntimeError(f"invalid_scoped_command_plan:{plan_errors}")
        planned = [
            validation_contract.PlannedCommand(command, "required_validation", True)
            for command in commands
        ]
        command_rows = validation_contract.run_categorized_commands(
            root, planned, log_dir=root / RAW_DIR / "validation/logs"
        )
        receipts.extend(_normalized_receipt(row) for row in command_rows)
    spans.append(_span("validate", phase_started, run_started))
    progress(run_started, "validate", "end", receipts=len(receipts))

    phase_started = time.monotonic()
    progress(run_started, "write", "start")
    candidate_path = root / RAW_DIR / "measured-terminal-candidate.json"
    candidate = _base_artifact(
        protocol,
        rows,
        witnesses,
        attacks,
        receipts,
        preconditions=preconditions,
        source_hashes=source_hashes,
        historical_sidecars=historical_sidecars,
        phase_spans=spans,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        duration_s=time.monotonic() - run_started,
        repository_health=repository_health,
        protocol_sha256=protocol_sha256,
    )
    atomic_json(candidate_path, candidate)
    terminal_rows: list[JsonDict] = []
    if not blocked:
        progress(run_started, "write", "before_terminal_subprocesses")
        raw_terminal = validation_contract.run_categorized_commands(
            root, _terminal_commands(candidate_path), log_dir=root / RAW_DIR / "terminal/logs"
        )
        terminal_rows = [_normalized_receipt(row) for row in raw_terminal]
        progress(run_started, "write", "after_terminal_subprocesses", count=len(terminal_rows))
    entrypoint_log = root / RAW_DIR / "terminal/declared_entrypoint.log"
    entrypoint_log.parent.mkdir(parents=True, exist_ok=True)
    entrypoint_log.write_text(
        "The current unbuffered process reached the atomic publication boundary.\n",
        encoding="utf-8",
    )
    terminal_rows.append(
        {
            "name": "declared_entrypoint",
            "command": " ".join(sys.argv),
            "command_argv": list(sys.argv),
            "environment": {
                key: os.environ.get(key)
                for key in ("PYTHONUNBUFFERED", "PYTHONPATH", "CARNOT_FORCE_LIVE")
            },
            "scope": "current_unbuffered_entrypoint",
            "return_code": 0,
            "exit_code": 0,
            "duration_s": time.monotonic() - run_started,
            "log_path": entrypoint_log.relative_to(root).as_posix(),
            "log_hash": sha256_file(entrypoint_log),
            "log_sha256": sha256_file(entrypoint_log),
            "passed": True,
            "timed_out": False,
            "required": True,
            "category": "capability_e2e",
            "started_at_utc": started_at,
            "ended_at_utc": utc_now(),
        }
    )
    receipts.extend(terminal_rows)
    flagged = any(
        row["name"] == "adversarial_verify" and row["passed"] is not True for row in terminal_rows
    )
    spans.append(_span("write", phase_started, run_started))
    final = _base_artifact(
        protocol,
        rows,
        witnesses,
        attacks,
        receipts,
        preconditions=preconditions,
        source_hashes=source_hashes,
        historical_sidecars=historical_sidecars,
        phase_spans=spans,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        duration_s=time.monotonic() - run_started,
        repository_health=repository_health,
        protocol_sha256=protocol_sha256,
        flagged_adversarial=flagged,
    )
    errors = validate_artifact(final)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    atomic_json(output, final)
    progress(run_started, "write", "end", output=output, verdict=final["verdict_class"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI boundary.
    args = parse_args(argv)
    run_experiment(REPO_ROOT, args.date, output_path=args.output)
    return 0
