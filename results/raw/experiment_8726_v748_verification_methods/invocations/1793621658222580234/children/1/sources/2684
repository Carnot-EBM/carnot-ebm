"""Build an independent exact chronological constraint stream.

The stream uses local SAT, graph-coloring, arithmetic, and finite-trace
fixtures. Exact solvers create the hidden sidecar. The readable decision view
does not contain labels or label-binding hashes, so a later learner can open
feedback only after its decision is durable.

Spec refs: REQ-CL-7105 and SCENARIO-CL-7105-*.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

from carnot.reporting.ltlzinc_temporal_continual_learning_adapter import (
    verify_temporal_case,
)
from carnot.verify.exact_repair_panel_manifest_v11 import arithmetic_exact_cases
from carnot.verify.sat import parse_dimacs
from carnot.verify.z3_math_verifier import Z3MathVerifier


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7105
SCHEMA = "carnot.exp7105.v623_exact_constraint_stream.v1"
RUN_DATE = "20260907"
RANDOM_SEED = 7_105_202_609_07
EXPECTED_EVENT_COUNT = 144
MIN_GROUP_COUNT = 12
MIN_HARD_ROWS = 48
MIN_DECOY_ROWS = 36
MIN_REUSE_ROWS = 108
MIN_RETENTION_ROWS = 36
REQUIRED_FAMILIES = ("sat", "graph_coloring", "arithmetic", "temporal")
HARDNESS_STRATA = ("easy", "medium", "hard")
INFERENCE_SUBSTRATE = (
    "deterministic exact local constraint generation and fresh-process witness replay"
)
INFERENCE_SUBSTRATE_CLASS = "no_model_load"
EXECUTION_VENUE = "host"

DEFAULT_STREAM_PATH = Path("results/streams/experiment_7105_v623_exact_constraint_stream.jsonl")
DEFAULT_DECISION_PATH = Path("results/streams/experiment_7105_v623_decision_view.jsonl")
DEFAULT_LABEL_PATH = Path("results/streams/experiment_7105_v623_label_view.jsonl")
DEFAULT_ARTIFACT_PATH = Path("results/experiment_7105_v623_exact_constraint_stream.json")

SOURCE_MODULE_PATHS = (
    Path("python/carnot/verify/sat.py"),
    Path("tests/python/test_verify_sat.py"),
    Path("python/carnot/verify/graph_coloring.py"),
    Path("tests/python/test_verify_graph_coloring.py"),
    Path("python/carnot/verify/exact_repair_panel_manifest_v11.py"),
    Path("python/carnot/verify/z3_math_verifier.py"),
    Path("python/carnot/reporting/ltlzinc_temporal_continual_learning_adapter.py"),
    Path("tests/python/test_ltlzinc_temporal_continual_learning_adapter.py"),
)

FROZEN_GROUP_IDS = (
    "sat_implication_chain",
    "sat_exactly_one",
    "sat_xor",
    "graph_triangle",
    "graph_even_cycle",
    "graph_wheel",
    "arithmetic_triangle_rule",
    "arithmetic_delta_rule",
    "arithmetic_box_rule",
    "temporal_always",
    "temporal_eventually",
    "temporal_until",
)
FROZEN_FAMILY_IDS = REQUIRED_FAMILIES
FROZEN_PROTECTED_GROUP_IDS = (
    "sat_implication_chain",
    "graph_triangle",
    "arithmetic_triangle_rule",
    "temporal_always",
)
FROZEN_SLICE_DEFINITIONS = (
    {"slice_id": "early", "start": 0, "stop": 48, "event_count": 48},
    {"slice_id": "middle", "start": 48, "stop": 96, "event_count": 48},
    {"slice_id": "late", "start": 96, "stop": 144, "event_count": 48},
)
FROZEN_CAPACITY_SCHEDULE = (
    {"slice_id": "early", "memory_capacity": 12},
    {"slice_id": "middle", "memory_capacity": 18},
    {"slice_id": "late", "memory_capacity": 24},
)
FROZEN_PRIMARY_COMPARISONS = (
    "structure_memory_vs_no_memory",
    "structure_memory_vs_fifo",
)

FORBIDDEN_DECISION_FIELDS = {
    "exact_label",
    "hidden_label",
    "future_label",
    "label",
    "witness",
    "counterexample",
    "candidate_valid",
    "solver_receipt",
    "label_content_hash",
    "canonical_content_hash",
}

REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment_id",
    "run_date",
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "duration_s",
    "source_artifact_hashes",
    "stream_path",
    "decision_view_path",
    "label_view_path",
    "stream_hash",
    "decision_view_hash",
    "label_view_hash",
    "event_count",
    "group_count",
    "family_count",
    "frozen_group_ids",
    "frozen_family_ids",
    "frozen_protected_group_ids",
    "frozen_capacity_schedule",
    "frozen_slice_definitions",
    "frozen_primary_comparisons",
    "rows",
    "event_rows",
    "group_rows",
    "family_rows",
    "hardness_rows",
    "reuse_rows",
    "decoy_rows",
    "retention_probe_rows",
    "chronology_rows",
    "witness_replay_rows",
    "uniqueness_rows",
    "conflict_rows",
    "leakage_rows",
    "mutation_attack_rows",
    "exact_constraint_stream_ready_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "schema": "A versioned schema makes incompatible stream readers fail closed.",
    "experiment_id": "A stable ID prevents another task from supplying these rows.",
    "run_date": "The fixed execution date identifies the sealed construction run.",
    "field_principles": "A reason for each field keeps the evidence contract reviewable.",
    "preconditions_checked": "Exact preflight rows stop incomplete fixture support from running.",
    "inference_substrate": "The substrate states that local exact solvers, not a model, made labels.",
    "inference_substrate_class": "The compute class prevents deterministic replay from becoming a model claim.",
    "execution_venue": "The venue distinguishes this host replay from remote or hardware execution.",
    "duration_s": "Measured wall time proves that construction and replay executed.",
    "source_artifact_hashes": "Source hashes bind each local fixture and solver to the stream.",
    "stream_path": "The full immutable path gives auditors one canonical event sequence.",
    "decision_view_path": "A separate decision path keeps future exact outcomes unavailable.",
    "label_view_path": "A separate label path permits feedback release after each decision.",
    "stream_hash": "The full-stream hash detects any event or witness change.",
    "decision_view_hash": "The decision hash freezes all learner-visible inputs before feedback.",
    "label_view_hash": "The label hash freezes exact feedback without exposing it in decisions.",
    "event_count": "The exact denominator prevents a small failed branch from entering Exp7106.",
    "group_count": "Independent groups prevent repeated rows from posing as broad learning support.",
    "family_count": "Family breadth tests reuse across distinct exact constraint mechanisms.",
    "frozen_group_ids": "A fixed group roster prevents outcome-driven regrouping.",
    "frozen_family_ids": "A fixed family roster prevents outcome-driven family selection.",
    "frozen_protected_group_ids": "Protected groups make later retention regression measurable.",
    "frozen_capacity_schedule": "A prior capacity schedule keeps later memory comparisons causal.",
    "frozen_slice_definitions": "Fixed time slices prevent outcome-driven early or late boundaries.",
    "frozen_primary_comparisons": "Prior comparisons prevent selective reporting after Exp7106.",
    "rows": "Complete raw rows keep the scientific denominator available to generic linters.",
    "event_rows": "Event rows preserve every decision, label, witness, role, and seal.",
    "group_rows": "Group summaries prove every group contains the required learning structure.",
    "family_rows": "Family summaries expose exact coverage and group balance.",
    "hardness_rows": "Hardness counts prove the stream has a predeclared difficult slice.",
    "reuse_rows": "Reuse rows identify the structures that later memory may lawfully exploit.",
    "decoy_rows": "Decoy rows measure whether superficial matches cause harmful retrieval.",
    "retention_probe_rows": "Retention rows preserve old structures for forgetting checks.",
    "chronology_rows": "Chronology rows prove stable order and delayed feedback.",
    "witness_replay_rows": "Replay rows bind every hidden label to exact executable evidence.",
    "uniqueness_rows": "Uniqueness checks prevent duplicate IDs, decisions, or content.",
    "conflict_rows": "Conflict checks prevent one decision or group from carrying two meanings.",
    "leakage_rows": "Leakage checks prove the learner-visible view excludes hidden outcomes.",
    "mutation_attack_rows": "Mutation attacks show that each protected seal field is effective.",
    "exact_constraint_stream_ready_score": "One requires every count, coverage, replay, leakage, and mutation gate.",
    "random_seed": "One seed fixes candidate order and the label-sidecar permutation.",
    "reproducibility_checksum": "A timing-free digest detects drift in scientific content.",
    "gate_check_summary": "Exact failed values make every blocked no-run diagnosable.",
    "verifier_is_oracle": "True discloses that exact solvers construct the fixture labels.",
    "verdict_class": "A closed verdict separates circular fixture readiness from model quality.",
    "honest_verdict": "A terminal prefix gives automation a stable and bounded conclusion.",
}


class ImmutableSealError(RuntimeError):
    """Report an attempt to replace an existing seal with different bytes."""


@dataclass(frozen=True)
class StreamPaths:
    """Keep all stream outputs explicit so tests can use private directories."""

    stream: Path
    decisions: Path
    labels: Path
    artifact: Path

    @classmethod
    def defaults(cls) -> StreamPaths:
        """Return the repository-relative paths required by the task command."""

        return cls(
            DEFAULT_STREAM_PATH, DEFAULT_DECISION_PATH, DEFAULT_LABEL_PATH, DEFAULT_ARTIFACT_PATH
        )

    @classmethod
    def under(cls, root: Path) -> StreamPaths:
        """Return private paths below one root for isolated tests and replays."""

        return cls(
            root / "stream.jsonl",
            root / "decisions.jsonl",
            root / "labels.jsonl",
            root / "artifact.json",
        )


def canonical_json(value: Any) -> bytes:
    """Serialize JSON once so all processes calculate the same content bytes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_bytes(value: bytes) -> str:
    """Return one named SHA-256 digest for bytes."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash canonical JSON instead of formatting-dependent source text."""

    return sha256_bytes(canonical_json(value))


def sha256_path(path: Path) -> str | None:
    """Hash one source or keep a missing path visible as a failed value."""

    return sha256_bytes(path.read_bytes()) if path.is_file() else None


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Encode immutable JSONL with one canonical row on each line."""

    return b"".join(canonical_json(row) + b"\n" for row in rows)


def write_immutable_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    """Create a seal once and reject any later run with different bytes."""

    payload = _jsonl_bytes(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ImmutableSealError(f"immutable_seal_mismatch:{path}") from None
    return sha256_bytes(payload)


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Publish the terminal artifact only after complete bytes exist beside it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
    os.replace(temporary, path)


def gate_check(check: str, expected: Any, observed: Any, passed: bool | None = None) -> JsonDict:
    """Record both sides of one exact gate comparison."""

    return {
        "check": check,
        "expected_value": deepcopy(expected),
        "observed_value": deepcopy(observed),
        "passed": observed == expected if passed is None else bool(passed),
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Keep all checks and expose the first failed comparison directly."""

    copied = [dict(row) for row in checks]
    failed = [row for row in copied if row.get("passed") is not True]
    first = failed[0] if failed else {}
    return {
        "checks": copied,
        "passed": not failed,
        "failed_check": first.get("check"),
        "expected_value": deepcopy(first.get("expected_value")),
        "observed_value": deepcopy(first.get("observed_value")),
    }


def group_definitions() -> list[JsonDict]:
    """Freeze twelve group templates from existing local exact fixture families."""

    arithmetic_sources = arithmetic_exact_cases()[:3]
    rows = [
        {
            "group_id": "sat_implication_chain",
            "constraint_family": "sat",
            "template": "implication_chain",
            "fixture_source": "test_verify_sat.py",
        },
        {
            "group_id": "sat_exactly_one",
            "constraint_family": "sat",
            "template": "exactly_one",
            "fixture_source": "test_verify_sat.py",
        },
        {
            "group_id": "sat_xor",
            "constraint_family": "sat",
            "template": "xor",
            "fixture_source": "test_verify_sat.py",
        },
        {
            "group_id": "graph_triangle",
            "constraint_family": "graph_coloring",
            "template": "triangle",
            "fixture_source": "test_verify_graph_coloring.py",
        },
        {
            "group_id": "graph_even_cycle",
            "constraint_family": "graph_coloring",
            "template": "even_cycle",
            "fixture_source": "test_verify_graph_coloring.py",
        },
        {
            "group_id": "graph_wheel",
            "constraint_family": "graph_coloring",
            "template": "wheel",
            "fixture_source": "test_verify_graph_coloring.py",
        },
        {
            "group_id": "arithmetic_triangle_rule",
            "constraint_family": "arithmetic",
            "template": "triangle",
            "fixture_source": arithmetic_sources[0]["case_id"],
        },
        {
            "group_id": "arithmetic_delta_rule",
            "constraint_family": "arithmetic",
            "template": "delta",
            "fixture_source": arithmetic_sources[1]["case_id"],
        },
        {
            "group_id": "arithmetic_box_rule",
            "constraint_family": "arithmetic",
            "template": "box",
            "fixture_source": arithmetic_sources[2]["case_id"],
        },
        {
            "group_id": "temporal_always",
            "constraint_family": "temporal",
            "template": "always",
            "fixture_source": "ltlzinc_temporal_cases",
        },
        {
            "group_id": "temporal_eventually",
            "constraint_family": "temporal",
            "template": "eventually",
            "fixture_source": "ltlzinc_temporal_cases",
        },
        {
            "group_id": "temporal_until",
            "constraint_family": "temporal",
            "template": "until",
            "fixture_source": "ltlzinc_temporal_cases",
        },
    ]
    return rows


def _candidate_count(local_index: int) -> int:
    """Use a visible candidate count to define easy, medium, and hard rows."""

    return 2 if local_index < 4 else 3 if local_index < 8 else 4


def derive_hardness(decision_input: Mapping[str, Any]) -> str:
    """Derive difficulty only from prompt-visible structural features."""

    count = int(decision_input["difficulty_features"]["candidate_count"])
    return "easy" if count == 2 else "medium" if count == 3 else "hard"


def _sat_problem(template: str, local_index: int) -> JsonDict:
    """Return one CNF template using the same signed-literal form as DIMACS."""

    del local_index
    if template == "implication_chain":
        clauses = [[-1, 2], [-2, 3], [1]]
    elif template == "exactly_one":
        clauses = [[1, 2, 3], [-1, -2], [-1, -3], [-2, -3]]
    else:
        clauses = [[1, 2], [-1, -2], [2, 3], [-2, -3]]
    dimacs = "p cnf 3 %d\n%s\n" % (
        len(clauses),
        "\n".join(" ".join(str(value) for value in clause) + " 0" for clause in clauses),
    )
    return {"n_vars": 3, "clauses": clauses, "dimacs": dimacs}


def _sat_valid(problem: Mapping[str, Any], assignment: Sequence[bool]) -> tuple[bool, JsonDict]:
    """Evaluate every parsed CNF clause exactly on a Boolean assignment."""

    clauses, n_vars = parse_dimacs(str(problem["dimacs"]))
    failed: list[int] = []
    for index, clause in enumerate(clauses):
        if not any(
            (not assignment[var]) if negated else assignment[var]
            for var, negated in clause.literals
        ):
            failed.append(index)
    return not failed and len(assignment) == n_vars, {"failed_clause_indices": failed}


def _graph_problem(template: str, local_index: int) -> JsonDict:
    """Return graph shapes used by the existing exact graph-coloring tests."""

    del local_index
    if template == "triangle":
        return {"n_nodes": 3, "n_colors": 3, "edges": [[0, 1], [1, 2], [0, 2]]}
    if template == "even_cycle":
        return {"n_nodes": 4, "n_colors": 2, "edges": [[0, 1], [1, 2], [2, 3], [3, 0]]}
    return {
        "n_nodes": 5,
        "n_colors": 3,
        "edges": [[0, 1], [1, 2], [2, 3], [3, 0], [4, 0], [4, 1], [4, 2], [4, 3]],
    }


def _graph_valid(problem: Mapping[str, Any], colors: Sequence[int]) -> tuple[bool, JsonDict]:
    """Check the discrete edge and range constraints without numeric tolerance."""

    n_colors = int(problem["n_colors"])
    range_failures = [index for index, color in enumerate(colors) if color < 0 or color >= n_colors]
    edge_failures = [edge for edge in problem["edges"] if colors[edge[0]] == colors[edge[1]]]
    valid = len(colors) == int(problem["n_nodes"]) and not range_failures and not edge_failures
    return valid, {"conflicting_edges": edge_failures, "out_of_range_nodes": range_failures}


def _arithmetic_problem(template: str, local_index: int, group_index: int) -> JsonDict:
    """Vary operands while retaining three exact local worksheet rules."""

    left = 2 + ((local_index + group_index) % 7)
    right = 1 + ((2 * local_index + group_index) % 6)
    rules = {
        "triangle": "a*b+a",
        "delta": "a*b-b",
        "box": "(a+b)*2",
    }
    return {"rule_name": template, "rule_expression": rules[template], "a": left, "b": right}


def _arithmetic_target(problem: Mapping[str, Any]) -> int:
    """Apply one declared integer rule without evaluating arbitrary source text."""

    left, right = int(problem["a"]), int(problem["b"])
    if problem["rule_name"] == "triangle":
        return left * right + left
    if problem["rule_name"] == "delta":
        return left * right - right
    return (left + right) * 2


def _arithmetic_valid(problem: Mapping[str, Any], value: int) -> tuple[bool, JsonDict]:
    """Cross-check exact integer evaluation with the local symbolic verifier."""

    expected = _arithmetic_target(problem)
    expression = (
        str(problem["rule_expression"])
        .replace("a", str(problem["a"]))
        .replace("b", str(problem["b"]))
    )
    claim = f"{expression} = {value}"
    verifier_score = Z3MathVerifier().score(claim)
    valid = value == expected and verifier_score == 0.0
    return valid, {
        "expected_value": expected,
        "observed_value": value,
        "verifier_score": verifier_score,
    }


def _temporal_problem(template: str, local_index: int) -> JsonDict:
    """Return one bounded temporal formula with a visible finite horizon."""

    horizon = 2 + (local_index % 4)
    return {
        "temporal_operator": template,
        "signal": "goal" if template == "until" else "signal",
        "guard_signal": "safe" if template == "until" else None,
        "horizon": horizon,
    }


def _temporal_valid(
    problem: Mapping[str, Any], trace: Sequence[Mapping[str, bool]]
) -> tuple[bool, JsonDict]:
    """Run the existing finite-trace authority and expose violated positions."""

    case = {
        "temporal_operator": problem["temporal_operator"],
        "signal": problem["signal"],
        "guard_signal": problem["guard_signal"],
        "trace": trace,
    }
    valid = verify_temporal_case(case)
    signal = str(problem["signal"])
    false_positions = [
        index for index, step in enumerate(trace) if not bool(step.get(signal, False))
    ]
    return valid, {"false_signal_positions": false_positions, "horizon": len(trace)}


def _all_candidate_payloads(
    definition: Mapping[str, Any], local_index: int, group_index: int
) -> tuple[JsonDict, list[Any]]:
    """Enumerate exact candidates and retain enough invalid controls for difficulty."""

    family = str(definition["constraint_family"])
    template = str(definition["template"])
    if family == "sat":
        problem_row = _sat_problem(template, local_index)
        payloads = [{"assignment": list(values)} for values in product((False, True), repeat=3)]
    elif family == "graph_coloring":
        problem_row = _graph_problem(template, local_index)
        payloads = [
            {"colors": list(values)}
            for values in product(
                range(int(problem_row["n_colors"])), repeat=int(problem_row["n_nodes"])
            )
        ]
    elif family == "arithmetic":
        problem_row = _arithmetic_problem(template, local_index, group_index)
        target = _arithmetic_target(problem_row)
        payloads = [{"value": target + offset} for offset in (0, -1, 1, -2, 2, 3)]
    else:
        problem_row = _temporal_problem(template, local_index)
        horizon = int(problem_row["horizon"])
        payloads = []
        for variant in range(8):
            trace: list[JsonDict] = []
            for position in range(horizon):
                if template == "always":
                    step = {"signal": variant == 0 or position != variant % horizon}
                elif template == "eventually":
                    step = {"signal": variant == 0 and position == (local_index % horizon)}
                else:
                    goal_at = horizon - 1
                    step = {
                        "goal": variant == 0 and position == goal_at,
                        "safe": variant == 0 or position != max(0, goal_at - 1),
                    }
                step[f"surface_{variant}"] = position % 2 == 0
                trace.append(step)
            payloads.append({"trace": trace})
    return problem_row, payloads


def _candidate_valid(
    family: str, problem_row: Mapping[str, Any], payload: Mapping[str, Any]
) -> tuple[bool, JsonDict]:
    """Dispatch one candidate to its deterministic exact family solver."""

    if family == "sat":
        return _sat_valid(problem_row, payload["assignment"])
    if family == "graph_coloring":
        return _graph_valid(problem_row, payload["colors"])
    if family == "arithmetic":
        return _arithmetic_valid(problem_row, int(payload["value"]))
    return _temporal_valid(problem_row, payload["trace"])


def _choose_candidates(
    event_id: str,
    family: str,
    problem_row: Mapping[str, Any],
    payloads: Sequence[Mapping[str, Any]],
    count: int,
    local_index: int,
) -> list[JsonDict]:
    """Choose one valid candidate plus exact invalid controls in label-blind order."""

    classified = [
        (payload, _candidate_valid(family, problem_row, payload)[0]) for payload in payloads
    ]
    valid = [payload for payload, passed in classified if passed]
    invalid = [payload for payload, passed in classified if not passed]
    if not valid or len(invalid) < count - 1:
        raise ValueError(f"insufficient_exact_candidates:{event_id}")
    chosen_valid = valid[local_index % len(valid)]
    start = (local_index * 3) % len(invalid)
    chosen_invalid = [invalid[(start + index) % len(invalid)] for index in range(count - 1)]
    chosen = [chosen_valid, *chosen_invalid]
    chosen.sort(
        key=lambda payload: sha256_json(
            {"event_id": event_id, "payload": payload, "seed": RANDOM_SEED}
        )
    )
    return [
        {"candidate_id": f"{event_id}:candidate:{index}", "payload": deepcopy(payload)}
        for index, payload in enumerate(chosen)
    ]


def solve_decision(
    family: str,
    problem_row: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[str, JsonDict]:
    """Return the sole valid candidate and a counterexample for every invalid one."""

    results: list[JsonDict] = []
    for candidate in candidates:
        valid, detail = _candidate_valid(family, problem_row, candidate["payload"])
        results.append(
            {
                "candidate_id": candidate["candidate_id"],
                "valid": valid,
                "counterexample": detail,
            }
        )
    valid_rows = [row for row in results if row["valid"]]
    if len(valid_rows) != 1:
        raise ValueError(f"exact_candidate_count:{len(valid_rows)}")
    label = str(valid_rows[0]["candidate_id"])
    payload = next(deepcopy(row["payload"]) for row in candidates if row["candidate_id"] == label)
    witness = {
        "authority": f"deterministic_{family}_solver",
        "valid_candidate_id": label,
        "valid_candidate_payload": payload,
        "candidate_results": results,
    }
    return label, witness


def _decision_signature(row: Mapping[str, Any]) -> str:
    """Identify one visible decision while ignoring its event and chronology names."""

    return sha256_json(
        {
            "constraint_family": row["constraint_family"],
            "decision_visible_input": row["decision_visible_input"],
            "candidate_payloads": [candidate["payload"] for candidate in row["candidate_set"]],
        }
    )


def _decision_payload(row: Mapping[str, Any]) -> JsonDict:
    """Project only fields that a later learner may read before feedback."""

    return {
        "event_id": row["event_id"],
        "constraint_family": row["constraint_family"],
        "group_id": row["group_id"],
        "chronology_index": row["chronology_index"],
        "reuse_or_decoy_status": row["reuse_or_decoy_status"],
        "hardness_stratum": row["hardness_stratum"],
        "feedback_release_point": row["feedback_release_point"],
        "protected_retention_probe": row["protected_retention_probe"],
        "protected_group": row["protected_group"],
        "matched_reusable_event_id": row["matched_reusable_event_id"],
        "structure_key": row["structure_key"],
        "decision_signature": row["decision_signature"],
        "decision_visible_input": deepcopy(row["decision_visible_input"]),
        "candidate_set": deepcopy(row["candidate_set"]),
    }


def _label_payload(row: Mapping[str, Any]) -> JsonDict:
    """Project exact feedback into the sealed label sidecar."""

    return {
        "event_id": row["event_id"],
        "feedback_release_point": row["feedback_release_point"],
        "exact_label": row["exact_label"],
        "witness": deepcopy(row["witness"]),
    }


def seal_event(row: Mapping[str, Any]) -> JsonDict:
    """Recompute every event seal after removing any stored digest values."""

    sealed = deepcopy(dict(row))
    for key in (
        "decision_signature",
        "decision_content_hash",
        "label_content_hash",
        "canonical_content_hash",
    ):
        sealed.pop(key, None)
    sealed["decision_signature"] = _decision_signature(sealed)
    sealed["decision_content_hash"] = sha256_json(_decision_payload(sealed))
    sealed["label_content_hash"] = sha256_json(_label_payload(sealed))
    sealed["canonical_content_hash"] = sha256_json(
        {
            "decision_content_hash": sealed["decision_content_hash"],
            "label_content_hash": sealed["label_content_hash"],
        }
    )
    return sealed


def build_events() -> list[JsonDict]:
    """Build exactly 144 interleaved events before any learning arm can run."""

    definitions = group_definitions()
    events: list[JsonDict] = []
    for local_index in range(12):
        for group_index, definition in enumerate(definitions):
            chronology_index = local_index * len(definitions) + group_index
            event_id = f"v623-{chronology_index:03d}-{definition['group_id']}"
            problem_row, payloads = _all_candidate_payloads(definition, local_index, group_index)
            candidate_count = _candidate_count(local_index)
            candidates = _choose_candidates(
                event_id,
                str(definition["constraint_family"]),
                problem_row,
                payloads,
                candidate_count,
                local_index,
            )
            constraint_count = len(
                problem_row.get("clauses", problem_row.get("edges", [problem_row]))
            )
            decision_input = {
                "task": "select_the_only_candidate_that_satisfies_all_declared_constraints",
                "fixture_source": definition["fixture_source"],
                "template": definition["template"],
                "surface_nonce": f"visible-variant-{local_index:02d}",
                "problem": problem_row,
                "difficulty_features": {
                    "candidate_count": candidate_count,
                    "constraint_count": constraint_count,
                },
            }
            label, witness = solve_decision(
                str(definition["constraint_family"]), problem_row, candidates
            )
            decoy = local_index in {2, 6, 10}
            retention = local_index in {3, 7, 11}
            matched_local = {2: 1, 6: 5, 10: 9}.get(local_index)
            matched_event = (
                f"v623-{matched_local * len(definitions) + group_index:03d}-{definition['group_id']}"
                if matched_local is not None
                else None
            )
            row = {
                "event_id": event_id,
                "constraint_family": definition["constraint_family"],
                "group_id": definition["group_id"],
                "chronology_index": chronology_index,
                "reuse_or_decoy_status": "decoy" if decoy else "reusable",
                "hardness_stratum": derive_hardness(decision_input),
                "feedback_release_point": chronology_index + 1,
                "protected_retention_probe": retention,
                "protected_group": definition["group_id"] in FROZEN_PROTECTED_GROUP_IDS,
                "matched_reusable_event_id": matched_event,
                "structure_key": (
                    f"{definition['group_id']}:matched-decoy:{local_index // 4}"
                    if decoy
                    else f"{definition['group_id']}:reusable-core"
                ),
                "decision_visible_input": decision_input,
                "candidate_set": candidates,
                "exact_label": label,
                "witness": witness,
            }
            events.append(seal_event(row))
    return events


def decision_view(events: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return chronological learner-visible rows with decision seals only."""

    rows: list[JsonDict] = []
    for event in events:
        row = _decision_payload(event)
        row["decision_content_hash"] = event["decision_content_hash"]
        rows.append(row)
    return rows


def label_blind_event_order(events: Sequence[Mapping[str, Any]]) -> list[str]:
    """Order label rows by seeded event identity without reading a label."""

    return [
        str(row["event_id"])
        for row in sorted(
            events,
            key=lambda item: sha256_json({"event_id": item["event_id"], "seed": RANDOM_SEED}),
        )
    ]


def label_view(events: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return sealed feedback rows in a label-blind storage permutation."""

    by_id = {str(row["event_id"]): row for row in events}
    rows: list[JsonDict] = []
    for event_id in label_blind_event_order(events):
        event = by_id[event_id]
        row = _label_payload(event)
        row["label_content_hash"] = event["label_content_hash"]
        rows.append(row)
    return rows


def _nested_keys(value: Any) -> set[str]:
    """Collect nested dictionary keys for explicit leakage checks."""

    if isinstance(value, Mapping):
        return {str(key) for key in value} | set().union(
            *(_nested_keys(item) for item in value.values()), set()
        )
    if isinstance(value, list):
        return set().union(*(_nested_keys(item) for item in value), set())
    return set()


def leakage_errors(decisions: Sequence[Mapping[str, Any]]) -> list[str]:
    """Name every hidden or label-binding field in learner-visible rows."""

    errors: list[str] = []
    for index, row in enumerate(decisions):
        for key in sorted(_nested_keys(row) & FORBIDDEN_DECISION_FIELDS):
            errors.append(f"decision_row_{index}:{key}")
    return errors


def replay_event(event: Mapping[str, Any]) -> JsonDict:
    """Recompute one exact label and witness from decision-visible content."""

    family = str(event["constraint_family"])
    problem_row = event["decision_visible_input"]["problem"]
    label, witness = solve_decision(family, problem_row, event["candidate_set"])
    passed = label == event.get("exact_label") and witness == event.get("witness")
    return {
        "event_id": event["event_id"],
        "constraint_family": family,
        "label_matches": label == event.get("exact_label"),
        "witness_matches": witness == event.get("witness"),
        "witness_hash": sha256_json(witness),
        "passed": passed,
    }


def stream_conformance_errors(
    events: Sequence[Mapping[str, Any]],
    *,
    decisions: Sequence[Mapping[str, Any]] | None = None,
    labels: Sequence[Mapping[str, Any]] | None = None,
) -> list[str]:
    """Return stable failures for identity, coverage, chronology, replay, and leakage."""

    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    rows = [dict(row) for row in events]
    event_ids = [str(row.get("event_id")) for row in rows]
    content_hashes = [str(row.get("canonical_content_hash")) for row in rows]
    group_ids = {str(row.get("group_id")) for row in rows}
    families = {str(row.get("constraint_family")) for row in rows}
    hardness = Counter(str(row.get("hardness_stratum")) for row in rows)
    roles = Counter(str(row.get("reuse_or_decoy_status")) for row in rows)
    retention_count = sum(row.get("protected_retention_probe") is True for row in rows)
    add(len(rows) != EXPECTED_EVENT_COUNT, "event_count_not_144")
    add(len(group_ids) < MIN_GROUP_COUNT, "group_count_below_12")
    add(families != set(REQUIRED_FAMILIES), "family_coverage_incomplete")
    add(set(hardness) != set(HARDNESS_STRATA), "hardness_coverage_incomplete")
    add(hardness["hard"] < MIN_HARD_ROWS, "insufficient_hard_rows")
    add(roles["decoy"] < MIN_DECOY_ROWS, "insufficient_decoy_rows")
    add(roles["reusable"] < MIN_REUSE_ROWS, "insufficient_reuse_rows")
    add(retention_count < MIN_RETENTION_ROWS, "insufficient_retention_probe_rows")
    add(len(set(event_ids)) != len(event_ids), "duplicate_event_id")
    add(len(set(content_hashes)) != len(content_hashes), "duplicate_event_content")
    add(
        [row.get("chronology_index") for row in rows] != list(range(len(rows))),
        "unstable_chronology_order",
    )
    add(
        any(
            row.get("feedback_release_point") != int(row.get("chronology_index", -1)) + 1
            for row in rows
        ),
        "feedback_released_before_decision",
    )
    add(
        any(
            row.get("hardness_stratum") != derive_hardness(row["decision_visible_input"])
            for row in rows
        ),
        "hardness_not_visible_derived",
    )
    add(any(seal_event(row) != row for row in rows), "event_hash_mismatch")

    group_families: dict[str, set[str]] = defaultdict(set)
    decision_labels: dict[str, set[str]] = defaultdict(set)
    by_event_id = {str(row.get("event_id")): row for row in rows}
    for row in rows:
        group_families[str(row.get("group_id"))].add(str(row.get("constraint_family")))
        decision_labels[str(row.get("decision_signature"))].add(str(row.get("exact_label")))
    add(any(len(values) > 1 for values in group_families.values()), "group_family_collision")
    add(any(len(values) > 1 for values in decision_labels.values()), "contradictory_exact_labels")

    for group_id in group_ids:
        group = [row for row in rows if row.get("group_id") == group_id]
        add(len(group) != 12, "group_event_count_not_12")
        add(
            not any(row.get("reuse_or_decoy_status") == "reusable" for row in group),
            "group_missing_reuse",
        )
        add(
            not any(row.get("reuse_or_decoy_status") == "decoy" for row in group),
            "group_missing_decoy",
        )
        add(
            not any(row.get("hardness_stratum") == "hard" for row in group),
            "group_missing_hard_slice",
        )
        add(
            not any(row.get("protected_retention_probe") is True for row in group),
            "group_missing_retention_probe",
        )
    for row in rows:
        if row.get("reuse_or_decoy_status") != "decoy":
            continue
        matched = by_event_id.get(str(row.get("matched_reusable_event_id")))
        add(matched is None, "decoy_missing_match")
        if matched is not None:
            add(
                matched.get("chronology_index", -1) >= row.get("chronology_index", -1),
                "decoy_match_not_prior",
            )
            add(matched.get("group_id") != row.get("group_id"), "decoy_match_group_mismatch")
            add(
                matched.get("hardness_stratum") != row.get("hardness_stratum"),
                "decoy_match_hardness_mismatch",
            )

    replay_failed = False
    for row in rows:
        try:
            replay_failed = replay_failed or replay_event(row)["passed"] is not True
        except (KeyError, TypeError, ValueError, IndexError):
            replay_failed = True
    add(replay_failed, "witness_replay_mismatch")

    visible = decision_view(rows) if decisions is None else [dict(row) for row in decisions]
    add(len(visible) != len(rows) or bool(leakage_errors(visible)), "future_label_leakage")
    if labels is not None:
        add([dict(row) for row in labels] != label_view(rows), "label_view_mismatch")
    return errors


def replay_projection(events: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return the timing-free hashes a clean interpreter must reproduce."""

    replays = [replay_event(row) for row in events]
    return {
        "event_count": len(events),
        "stream_hash": sha256_bytes(_jsonl_bytes(events)),
        "decision_view_hash": sha256_bytes(_jsonl_bytes(decision_view(events))),
        "label_view_hash": sha256_bytes(_jsonl_bytes(label_view(events))),
        "witness_replay_hash": sha256_json(replays),
        "witness_hashes": [row["witness_hash"] for row in replays],
    }


def fresh_process_replay(repo_root: Path) -> JsonDict:
    """Regenerate all rows in a clean interpreter with only local package code."""

    environment = dict(os.environ)
    python_root = str(repo_root / "python")
    environment["PYTHONPATH"] = python_root + os.pathsep + environment.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, "-m", "carnot.experiment_7105_v623_exact_constraint_stream", "--replay"],
        cwd=repo_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise ValueError("fresh_process_replay_not_object")
    return value


def run_mutation_attacks(events: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Attack one byte, label, order index, and group assignment independently."""

    attacks: list[tuple[str, list[JsonDict]]] = []
    byte_rows = deepcopy(list(events))
    byte_rows[0]["decision_visible_input"]["surface_nonce"] += "x"
    attacks.append(("one_byte", byte_rows))
    label_rows = deepcopy(list(events))
    alternatives = [row["candidate_id"] for row in label_rows[0]["candidate_set"]]
    label_rows[0]["exact_label"] = next(
        value for value in alternatives if value != label_rows[0]["exact_label"]
    )
    attacks.append(("one_label", label_rows))
    order_rows = deepcopy(list(events))
    order_rows[0]["chronology_index"] = 1
    attacks.append(("one_order_index", order_rows))
    group_rows = deepcopy(list(events))
    group_rows[0]["group_id"] = FROZEN_GROUP_IDS[1]
    attacks.append(("one_group_assignment", group_rows))
    results: list[JsonDict] = []
    for name, rows in attacks:
        errors = stream_conformance_errors(rows)
        results.append(
            {"attack": name, "seal_invalidated": bool(errors), "detected_errors": errors}
        )
    return results


def _path_writable(path: Path) -> bool:
    """Probe a destination directory without changing the requested output path."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=".exp7105-write-probe-", dir=path.parent)
        os.close(descriptor)
        Path(name).unlink()
    except OSError:
        return False
    return True


def source_artifact_hashes(repo_root: Path) -> JsonDict:
    """Bind each existing local fixture and exact solver source to the artifact."""

    return {str(path): sha256_path(repo_root / path) for path in SOURCE_MODULE_PATHS}


def collect_preconditions(repo_root: Path, paths: StreamPaths) -> list[JsonDict]:
    """Check group breadth, local sources, deterministic replay, and destinations."""

    definitions = group_definitions()
    sources = source_artifact_hashes(repo_root)
    try:
        first = build_events()
        second = build_events()
        deterministic = canonical_json(first) == canonical_json(second)
        replay_ok = all(replay_event(row)["passed"] for row in first)
    except (KeyError, TypeError, ValueError, IndexError):
        deterministic = False
        replay_ok = False
    destinations = {
        "stream": _path_writable(paths.stream),
        "decision_view": _path_writable(paths.decisions),
        "label_view": _path_writable(paths.labels),
        "artifact": _path_writable(paths.artifact),
    }
    return [
        gate_check(
            "independent_exact_group_count", ">=12", len(definitions), len(definitions) >= 12
        ),
        gate_check(
            "constraint_family_count",
            4,
            len({row["constraint_family"] for row in definitions}),
        ),
        gate_check(
            "local_fixture_sources_readable",
            True,
            all(value is not None for value in sources.values()),
        ),
        gate_check("deterministic_local_generation", True, deterministic),
        gate_check("exact_label_and_witness_replay", True, replay_ok),
        gate_check(
            "stream_and_artifact_paths_writable", {key: True for key in destinations}, destinations
        ),
    ]


def _summary_rows(events: Sequence[Mapping[str, Any]], fresh: Mapping[str, Any]) -> JsonDict:
    """Reduce raw events into every required coverage and integrity table."""

    groups: list[JsonDict] = []
    for group_id in FROZEN_GROUP_IDS:
        rows = [row for row in events if row["group_id"] == group_id]
        groups.append(
            {
                "group_id": group_id,
                "constraint_family": rows[0]["constraint_family"] if rows else None,
                "event_count": len(rows),
                "reuse_count": sum(row["reuse_or_decoy_status"] == "reusable" for row in rows),
                "decoy_count": sum(row["reuse_or_decoy_status"] == "decoy" for row in rows),
                "hard_count": sum(row["hardness_stratum"] == "hard" for row in rows),
                "retention_probe_count": sum(
                    row["protected_retention_probe"] is True for row in rows
                ),
                "protected_group": group_id in FROZEN_PROTECTED_GROUP_IDS,
            }
        )
    families = [
        {
            "family_id": family,
            "event_count": sum(row["constraint_family"] == family for row in events),
            "group_ids": sorted(
                {row["group_id"] for row in events if row["constraint_family"] == family}
            ),
            "hardness_strata": sorted(
                {row["hardness_stratum"] for row in events if row["constraint_family"] == family}
            ),
        }
        for family in REQUIRED_FAMILIES
    ]
    hardness = [
        {
            "hardness_stratum": name,
            "event_count": sum(row["hardness_stratum"] == name for row in events),
        }
        for name in HARDNESS_STRATA
    ]
    reuse = [
        {
            "event_id": row["event_id"],
            "group_id": row["group_id"],
            "structure_key": row["structure_key"],
        }
        for row in events
        if row["reuse_or_decoy_status"] == "reusable"
    ]
    decoys = [
        {
            "event_id": row["event_id"],
            "group_id": row["group_id"],
            "matched_reusable_event_id": row["matched_reusable_event_id"],
            "hardness_stratum": row["hardness_stratum"],
        }
        for row in events
        if row["reuse_or_decoy_status"] == "decoy"
    ]
    retention = [
        {
            "event_id": row["event_id"],
            "group_id": row["group_id"],
            "protected_group": row["protected_group"],
        }
        for row in events
        if row["protected_retention_probe"] is True
    ]
    chronology = [
        {
            "event_id": row["event_id"],
            "chronology_index": row["chronology_index"],
            "feedback_release_point": row["feedback_release_point"],
            "feedback_after_decision": row["feedback_release_point"] == row["chronology_index"] + 1,
        }
        for row in events
    ]
    replays = [replay_event(row) for row in events]
    child_hashes = list(fresh.get("witness_hashes", []))
    for index, row in enumerate(replays):
        row["fresh_process_match"] = (
            index < len(child_hashes) and row["witness_hash"] == child_hashes[index]
        )
    uniqueness = [
        {
            "check": "event_ids",
            "observed": len({row["event_id"] for row in events}),
            "expected": len(events),
            "passed": len({row["event_id"] for row in events}) == len(events),
        },
        {
            "check": "canonical_content_hashes",
            "observed": len({row["canonical_content_hash"] for row in events}),
            "expected": len(events),
            "passed": len({row["canonical_content_hash"] for row in events}) == len(events),
        },
        {
            "check": "decision_signatures",
            "observed": len({row["decision_signature"] for row in events}),
            "expected": len(events),
            "passed": len({row["decision_signature"] for row in events}) == len(events),
        },
    ]
    group_family_conflicts = sum(
        len({row["constraint_family"] for row in events if row["group_id"] == group_id}) > 1
        for group_id in {row["group_id"] for row in events}
    )
    decision_labels: dict[str, set[str]] = defaultdict(set)
    for row in events:
        decision_labels[str(row["decision_signature"])].add(str(row["exact_label"]))
    conflicts = [
        {
            "check": "group_family_collisions",
            "conflict_count": group_family_conflicts,
            "passed": group_family_conflicts == 0,
        },
        {
            "check": "contradictory_exact_labels",
            "conflict_count": sum(len(values) > 1 for values in decision_labels.values()),
            "passed": all(len(values) == 1 for values in decision_labels.values()),
        },
    ]
    leakage = [
        {
            "event_id": row["event_id"],
            "forbidden_fields_present": sorted(_nested_keys(row) & FORBIDDEN_DECISION_FIELDS),
            "passed": not (_nested_keys(row) & FORBIDDEN_DECISION_FIELDS),
        }
        for row in decision_view(events)
    ]
    return {
        "group_rows": groups,
        "family_rows": families,
        "hardness_rows": hardness,
        "reuse_rows": reuse,
        "decoy_rows": decoys,
        "retention_probe_rows": retention,
        "chronology_rows": chronology,
        "witness_replay_rows": replays,
        "uniqueness_rows": uniqueness,
        "conflict_rows": conflicts,
        "leakage_rows": leakage,
    }


def _empty_rows() -> JsonDict:
    """Return every row table so a blocked no-run keeps the full schema."""

    return {
        key: []
        for key in (
            "rows",
            "event_rows",
            "group_rows",
            "family_rows",
            "hardness_rows",
            "reuse_rows",
            "decoy_rows",
            "retention_probe_rows",
            "chronology_rows",
            "witness_replay_rows",
            "uniqueness_rows",
            "conflict_rows",
            "leakage_rows",
            "mutation_attack_rows",
        )
    }


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash scientific content while excluding duration and the digest itself."""

    stable = deepcopy(dict(artifact))
    stable["duration_s"] = None
    stable["reproducibility_checksum"] = None
    return sha256_json(stable)


def build_blocked_artifact(
    checks: Sequence[Mapping[str, Any]],
    paths: StreamPaths,
    *,
    run_date: str,
    duration_s: float,
) -> JsonDict:
    """Emit every required field when a precondition prevents construction."""

    summary = gate_summary(checks)
    failed = str(summary.get("failed_check") or "unknown_precondition")
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": [dict(row) for row in checks],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": EXECUTION_VENUE,
        "duration_s": duration_s,
        "source_artifact_hashes": {},
        "stream_path": str(paths.stream),
        "decision_view_path": str(paths.decisions),
        "label_view_path": str(paths.labels),
        "stream_hash": None,
        "decision_view_hash": None,
        "label_view_hash": None,
        "event_count": 0,
        "group_count": 0,
        "family_count": 0,
        "frozen_group_ids": list(FROZEN_GROUP_IDS),
        "frozen_family_ids": list(FROZEN_FAMILY_IDS),
        "frozen_protected_group_ids": list(FROZEN_PROTECTED_GROUP_IDS),
        "frozen_capacity_schedule": deepcopy(FROZEN_CAPACITY_SCHEDULE),
        "frozen_slice_definitions": deepcopy(FROZEN_SLICE_DEFINITIONS),
        "frozen_primary_comparisons": list(FROZEN_PRIMARY_COMPARISONS),
        "exact_constraint_stream_ready_score": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": None,
        "gate_check_summary": summary,
        "verifier_is_oracle": True,
        "verdict_class": "blocked",
        "honest_verdict": f"complete_blocked_exact_constraint_stream:{failed}",
        **_empty_rows(),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def build_and_seal(
    repo_root: Path,
    paths: StreamPaths,
    *,
    run_date: str,
    duration_s: float | None = None,
) -> JsonDict:
    """Check preconditions, seal three views, and reduce the terminal artifact."""

    started = time.monotonic()
    checks = collect_preconditions(repo_root, paths)
    if not all(row["passed"] for row in checks):
        elapsed = time.monotonic() - started if duration_s is None else duration_s
        return build_blocked_artifact(checks, paths, run_date=run_date, duration_s=elapsed)
    events = build_events()
    decisions = decision_view(events)
    labels = label_view(events)
    conformance = stream_conformance_errors(events, decisions=decisions, labels=labels)
    parent_projection = replay_projection(events)
    child_projection = fresh_process_replay(repo_root)
    fresh_match = child_projection == parent_projection
    mutation_rows = run_mutation_attacks(events)
    ready = int(
        not conformance and fresh_match and all(row["seal_invalidated"] for row in mutation_rows)
    )
    stream_hash = write_immutable_jsonl(paths.stream, events)
    decision_hash = write_immutable_jsonl(paths.decisions, decisions)
    label_hash = write_immutable_jsonl(paths.labels, labels)
    summaries = _summary_rows(events, child_projection)
    elapsed = time.monotonic() - started if duration_s is None else duration_s
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": checks,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "duration_s": elapsed,
        "source_artifact_hashes": source_artifact_hashes(repo_root),
        "stream_path": str(paths.stream),
        "decision_view_path": str(paths.decisions),
        "label_view_path": str(paths.labels),
        "stream_hash": stream_hash,
        "decision_view_hash": decision_hash,
        "label_view_hash": label_hash,
        "event_count": len(events),
        "group_count": len({row["group_id"] for row in events}),
        "family_count": len({row["constraint_family"] for row in events}),
        "frozen_group_ids": list(FROZEN_GROUP_IDS),
        "frozen_family_ids": list(FROZEN_FAMILY_IDS),
        "frozen_protected_group_ids": list(FROZEN_PROTECTED_GROUP_IDS),
        "frozen_capacity_schedule": deepcopy(FROZEN_CAPACITY_SCHEDULE),
        "frozen_slice_definitions": deepcopy(FROZEN_SLICE_DEFINITIONS),
        "frozen_primary_comparisons": list(FROZEN_PRIMARY_COMPARISONS),
        "rows": deepcopy(events),
        "event_rows": deepcopy(events),
        **summaries,
        "mutation_attack_rows": mutation_rows,
        "exact_constraint_stream_ready_score": ready,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": None,
        "gate_check_summary": gate_summary(checks),
        "verifier_is_oracle": True,
        "verdict_class": "circular_positive" if ready else "disqualified",
        "honest_verdict": (
            "complete: exact constraint stream ready; exact solvers measure fixture integrity only"
            if ready
            else "complete_disqualified_exact_constraint_stream"
        ),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def validate_artifact(
    artifact: Mapping[str, Any],
    *,
    repo_root: Path | None = None,
    check_files: bool = False,
) -> list[str]:
    """Recompute schema, rows, hashes, readiness, verdict, and checksum."""

    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    add(bool(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)), "required_fields_missing")
    if errors:
        return errors
    add(
        set(artifact["field_principles"]) != set(REQUIRED_ARTIFACT_FIELDS),
        "field_principles_mismatch",
    )
    add(artifact["schema"] != SCHEMA, "schema_mismatch")
    add(artifact["experiment_id"] != EXPERIMENT_ID, "experiment_id_mismatch")
    add(
        type(artifact["exact_constraint_stream_ready_score"]) is not int,
        "ready_score_not_bare_integer",
    )
    add(artifact["verifier_is_oracle"] is not True, "oracle_declaration_mismatch")
    add(artifact["execution_venue"] != EXECUTION_VENUE, "execution_venue_mismatch")
    blocked = artifact["verdict_class"] == "blocked"
    if blocked:
        add(
            artifact["inference_substrate_class"] != "blocked_no_run",
            "blocked_substrate_class_mismatch",
        )
        add(artifact["exact_constraint_stream_ready_score"] != 0, "ready_score_mismatch")
        add(
            artifact["gate_check_summary"].get("passed") is not False,
            "blocked_gate_summary_mismatch",
        )
        add(
            not str(artifact["honest_verdict"]).startswith("complete_blocked_"),
            "honest_verdict_mismatch",
        )
    else:
        events = artifact["event_rows"]
        decisions = decision_view(events)
        labels = label_view(events)
        conformance = stream_conformance_errors(events, decisions=decisions, labels=labels)
        expected_ready = int(
            not conformance
            and len(artifact["mutation_attack_rows"]) == 4
            and all(row.get("seal_invalidated") is True for row in artifact["mutation_attack_rows"])
            and all(
                row.get("passed") is True and row.get("fresh_process_match") is True
                for row in artifact["witness_replay_rows"]
            )
        )
        add(artifact["rows"] != events, "rows_event_rows_mismatch")
        add(artifact["event_count"] != len(events), "event_count_mismatch")
        add(
            artifact["group_count"] != len({row["group_id"] for row in events}),
            "group_count_mismatch",
        )
        add(
            artifact["family_count"] != len({row["constraint_family"] for row in events}),
            "family_count_mismatch",
        )
        add(artifact["stream_hash"] != sha256_bytes(_jsonl_bytes(events)), "stream_hash_mismatch")
        add(
            artifact["decision_view_hash"] != sha256_bytes(_jsonl_bytes(decisions)),
            "decision_view_hash_mismatch",
        )
        add(
            artifact["label_view_hash"] != sha256_bytes(_jsonl_bytes(labels)),
            "label_view_hash_mismatch",
        )
        add(
            artifact["exact_constraint_stream_ready_score"] != expected_ready,
            "ready_score_mismatch",
        )
        expected_class = "circular_positive" if expected_ready else "disqualified"
        add(artifact["verdict_class"] != expected_class, "verdict_class_mismatch")
        add(
            expected_ready == 1 and not str(artifact["honest_verdict"]).startswith("complete:"),
            "honest_verdict_mismatch",
        )
        add(
            artifact["inference_substrate_class"] != INFERENCE_SUBSTRATE_CLASS,
            "substrate_class_mismatch",
        )
        if check_files:
            file_rows = (
                (Path(str(artifact["stream_path"])), artifact["stream_hash"]),
                (Path(str(artifact["decision_view_path"])), artifact["decision_view_hash"]),
                (Path(str(artifact["label_view_path"])), artifact["label_view_hash"]),
            )
            add(
                any(sha256_path(path) != expected for path, expected in file_rows),
                "sealed_file_hash_mismatch",
            )
    if repo_root is not None and not blocked:
        add(
            artifact["source_artifact_hashes"] != source_artifact_hashes(repo_root),
            "source_hash_mismatch",
        )
    add(artifact["reproducibility_checksum"] != payload_checksum(artifact), "checksum_mismatch")
    return errors


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the public construction command and private replay mode."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--stream-path", type=Path, default=DEFAULT_STREAM_PATH)
    parser.add_argument("--decision-view-path", type=Path, default=DEFAULT_DECISION_PATH)
    parser.add_argument("--label-view-path", type=Path, default=DEFAULT_LABEL_PATH)
    parser.add_argument("--artifact-path", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--replay", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Construct the terminal artifact or print the private replay projection."""

    args = _parse_args(argv)
    if args.replay:
        print(json.dumps(replay_projection(build_events()), sort_keys=True))
        return 0
    repo_root = Path(__file__).resolve().parents[2]
    paths = StreamPaths(
        args.stream_path, args.decision_view_path, args.label_view_path, args.artifact_path
    )
    artifact = build_and_seal(repo_root, paths, run_date=str(args.date))
    validation_errors = validate_artifact(
        artifact, repo_root=repo_root, check_files=artifact["verdict_class"] != "blocked"
    )
    if validation_errors:
        raise ValueError("artifact_validation_failed:" + ",".join(validation_errors))
    write_json_atomic(paths.artifact, artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the required command wrapper.
    raise SystemExit(main())
