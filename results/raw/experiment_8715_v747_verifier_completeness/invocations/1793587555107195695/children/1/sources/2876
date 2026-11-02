"""Measure a bounded exact-feedback finite-bias acquisition prototype.

The learner sees public variable domains and Boolean answers only. The exact
evaluator owns private rules and uses existential partial-assignment semantics.
This is a small CPU experiment, not the paper's neural-oracle method.

Spec refs: REQ-CL-7351 and SCENARIO-CL-7351-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import itertools
import json
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any

from carnot.experiment_7330_v644_public_learner import (
    canonical_bytes,
    sha256_bytes,
    sha256_file,
    sha256_json,
)
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    REQUIRED_CHECK_NAMES,
    build_scoped_commands,
    run_commands,
    run_scoped_validation,
)


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260916"
MILESTONE = "2026.09.645"
EXPERIMENT_ID = 7351
SCHEMA = "carnot.experiment_7351.v645_acquisition_prototype.v1"
RESULT_PATH = Path("results/experiment_7351_v645_acquisition_prototype.json")
RAW_PATH = Path("results/raw/experiment_7351_v645_acquisition_prototype")
MODULE_PATH = Path("python/carnot/experiment_7351_v645_acquisition_prototype.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7351_v645_acquisition_prototype.py")
TEST_PATH = Path("tests/python/test_experiment_7351_v645_acquisition_prototype.py")
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
FIXTURE_PATH = Path("results/experiment_7344_v645_executor_fixture.json")
QUERY_CAP = 256
STATE_CAP_BYTES = 69_632
FUTURE_REQUESTS = 12
DEVELOPMENT_SEED = 7_351_101
EVALUATION_SEED = 7_351_211
RESAMPLING_SEED = 7_351_303
ARMS = (
    "conservative_acquisition",
    "finite_bias_elimination",
    "exact_plan_cache_reset",
)
TERMINAL_CHECK_NAMES = (
    "independent_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
ZERO_INVOCATION_COUNTS = {
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
SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    SPEC_PATH,
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7330_v644_public_learner.py"),
    Path("scripts/experiments/experiment_7330_v644_private_executor.py"),
    Path("python/carnot/experiment_7323_v643_addition_prototype.py"),
    Path("results/experiment_7340_v644_native_cost.json"),
    FIXTURE_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)


class AcquisitionError(RuntimeError):
    """Stop on unsupported, malformed, over-budget, or oversized learner work."""


def progress(phase: str, event: str, detail: str = "") -> None:
    """Flush phase boundaries so long work never looks abandoned."""

    suffix = f" {detail}" if detail else ""
    print(f"[exp7351] phase={phase} event={event}{suffix}", flush=True)


def make_query(context_id: str, assignments: Mapping[str, int]) -> JsonDict:
    """Reuse the public plan shape while naming relation contexts as requests."""

    return {
        "request_id": str(context_id),
        "assignments": {name: value for name, value in sorted(assignments.items())},
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish complete evidence with fsync and a same-directory rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():  # pragma: no cover - only an interrupted rename leaves it.
            temporary.unlink()


def _table(relation: str) -> set[tuple[int, int]]:
    values = range(3)
    if relation == "unconstrained":
        return set(itertools.product(values, repeat=2))
    if relation == "eq":
        return {(left, right) for left in values for right in values if left == right}
    if relation == "neq":
        return {(left, right) for left in values for right in values if left != right}
    if relation == "next":
        return {(left, (left + 1) % 3) for left in values}
    if relation == "prev":
        return {(left, (left - 1) % 3) for left in values}
    if relation == "eq_or_next":
        return _table("eq") | _table("next")
    if relation == "eq_or_prev":
        return _table("eq") | _table("prev")
    raise AcquisitionError(f"unknown_relation:{relation}")


def finite_bias() -> list[JsonDict]:
    """Return seven distinct semantic tables, including no binary constraint."""

    aliases = {
        "unconstrained": ["unconstrained"],
        "eq": ["eq", "eq_or_next&eq_or_prev"],
        "neq": ["neq"],
        "next": ["next", "neq&eq_or_next"],
        "prev": ["prev", "neq&eq_or_prev"],
        "eq_or_next": ["eq_or_next"],
        "eq_or_prev": ["eq_or_prev"],
    }
    return [
        {
            "relation_id": relation,
            "allowed_pairs": [list(pair) for pair in sorted(_table(relation))],
            "syntactic_aliases": values,
        }
        for relation, values in aliases.items()
    ]


def validate_public_context(context: Mapping[str, Any]) -> None:
    """Accept only the declared domain-three public acquisition language."""

    required = {"context_id", "version_token", "variables", "domain", "future_requests"}
    if not required <= set(context):
        raise AcquisitionError("context_fields")
    if context["domain"] != [0, 1, 2]:
        raise AcquisitionError("unsupported_domain")
    variables = context["variables"]
    if not isinstance(variables, list) or not 2 <= len(variables) <= 8:
        raise AcquisitionError("variable_count")
    if len(set(variables)) != len(variables) or any(
        not isinstance(name, str) for name in variables
    ):
        raise AcquisitionError("variable_identity")
    future = context["future_requests"]
    if not isinstance(future, list) or len(future) != FUTURE_REQUESTS:
        raise AcquisitionError("future_request_count")
    if any(
        not isinstance(row, Mapping)
        or not isinstance(row.get("request_id"), str)
        or row.get("value_order") != [0, 1, 2]
        for row in future
    ):
        raise AcquisitionError("future_request_shape")


class ExactRelationEvaluator:
    """Keep private relations behind exact existential Boolean feedback."""

    def __init__(self, context: Mapping[str, Any], private_record: Mapping[str, Any]) -> None:
        self.context = deepcopy(dict(context))
        self._record = deepcopy(dict(private_record))
        self.variables = tuple(str(value) for value in context["variables"])
        self.domain = tuple(int(value) for value in context["domain"])
        self._valid_assignments = tuple(
            assignment
            for values in itertools.product(self.domain, repeat=len(self.variables))
            if self._allows(dict(zip(self.variables, values, strict=True)))
            for assignment in (dict(zip(self.variables, values, strict=True)),)
        )

    def _relation_allows(self, relation: str, left: int, right: int) -> bool:
        if relation == "outside_sum_even":
            return (left + right) % 2 == 0
        if set(self.domain) != {0, 1, 2}:
            if relation == "eq":
                return left == right
            if relation == "neq":
                return left != right
            raise AcquisitionError(f"unsupported_private_relation:{relation}")
        return (left, right) in _table(relation)

    def _allows(self, assignment: Mapping[str, int]) -> bool:
        for constraint in self._record.get("constraints", []):
            left, right = constraint["scope"]
            if any(
                not self._relation_allows(str(relation), assignment[left], assignment[right])
                for relation in constraint["relations"]
            ):
                return False
        for constraint in self._record.get("ternary_constraints", []):
            values = [assignment[name] for name in constraint["scope"]]
            if constraint["relation"] == "not_all_equal" and len(set(values)) == 1:
                return False
        return True

    def check(self, query: Mapping[str, Any]) -> bool:
        """Accept a partial query exactly when a private valid extension exists."""

        if not isinstance(query, Mapping) or query.get("request_id") != self.context["context_id"]:
            return False
        assignments = query.get("assignments")
        if not isinstance(assignments, Mapping) or not assignments:
            return False
        if not set(assignments) <= set(self.variables):
            return False
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value not in self.domain
            for value in assignments.values()
        ):
            return False
        return any(
            all(candidate[name] == value for name, value in assignments.items())
            for candidate in self._valid_assignments
        )

    def audit_projection(self, pair: Sequence[str]) -> set[tuple[int, int]]:
        """Enumerate a private projection for audit only, never learner feedback."""

        left, right = pair
        return {
            (left_value, right_value)
            for left_value, right_value in itertools.product(self.domain, repeat=2)
            if self.check(
                make_query(
                    str(self.context["context_id"]),
                    {left: left_value, right: right_value},
                )
            )
        }

    def first_witness(self, accepted: bool) -> JsonDict:
        """Create evaluator-only positive or negative full-assignment evidence."""

        for values in itertools.product(self.domain, repeat=len(self.variables)):
            assignment = dict(zip(self.variables, values, strict=True))
            if self._allows(assignment) is accepted:
                return make_query(str(self.context["context_id"]), assignment)
        raise AcquisitionError("missing_witness")


class ChargedExactOracle:
    """Charge exact calls, retain timing, and never cache a final authority check."""

    def __init__(
        self,
        evaluator: ExactRelationEvaluator,
        query_cap: int = QUERY_CAP,
        *,
        exact_cache: dict[str, bool] | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.query_cap = int(query_cap)
        self.exact_cache = exact_cache
        self.call_count = 0
        self.attempt_count = 0
        self.cache_hits = 0
        self.evaluator_time_s = 0.0
        self.receipts: list[JsonDict] = []

    def query(self, plan: Mapping[str, Any], reason: str, *, allow_cache: bool = False) -> bool:
        self.attempt_count += 1
        key = sha256_json(
            {
                "version_token": self.evaluator.context["version_token"],
                "assignments": plan.get("assignments"),
            }
        )
        cached = allow_cache and reason != "future_final" and self.exact_cache is not None
        started = time.perf_counter()
        if cached and key in self.exact_cache:
            accepted = self.exact_cache[key]
            external = False
            self.cache_hits += 1
        else:
            if self.call_count >= self.query_cap:
                raise AcquisitionError("query_cap")
            accepted = self.evaluator.check(plan)
            self.call_count += 1
            external = True
            if cached:
                assert self.exact_cache is not None
                self.exact_cache[key] = accepted
        elapsed = time.perf_counter() - started
        if external:
            self.evaluator_time_s += elapsed
        self.receipts.append(
            {
                "sequence": self.attempt_count,
                "charged_call_index": self.call_count,
                "reason": reason,
                "query_hash": sha256_json(plan),
                "accepted": bool(accepted),
                "external_call": external,
                "duration_s": elapsed,
            }
        )
        return bool(accepted)


class FiniteBiasEliminationLearner:
    """Eliminate semantic tables without receiving a target relation or scope."""

    def __init__(self, version_token: str, *, state_cap_bytes: int = STATE_CAP_BYTES) -> None:
        self.state_cap_bytes = int(state_cap_bytes)
        self._state: JsonDict = {
            "schema": "carnot.exp7351.finite-bias-state.v1",
            "active_version_token": str(version_token),
            "commits": [],
            "uncertainty": [],
        }

    def state_bytes(self) -> bytes:
        return canonical_bytes(self._state)

    def activate_version(self, version_token: str) -> None:
        self._state["active_version_token"] = str(version_token)

    def committed_relations(self) -> list[JsonDict]:
        token = self._state["active_version_token"]
        return deepcopy([row for row in self._state["commits"] if row["version_token"] == token])

    def uncertainties(self) -> list[JsonDict]:
        token = self._state["active_version_token"]
        return deepcopy(
            [row for row in self._state["uncertainty"] if row["version_token"] == token]
        )

    def _bounded_update(self, key: str, row: JsonDict) -> None:
        previous = deepcopy(self._state)
        self._state[key] = [*self._state[key], row]
        if len(self.state_bytes()) > self.state_cap_bytes:
            self._state = previous
            raise AcquisitionError("state_cap")

    @staticmethod
    def _adaptive_tuple(
        candidates: Sequence[Mapping[str, Any]], remaining: Sequence[tuple[int, int]]
    ) -> tuple[int, int]:
        def score(pair: tuple[int, int]) -> tuple[int, int]:
            accepted = sum(
                pair in {tuple(value) for value in row["allowed_pairs"]} for row in candidates
            )
            return min(accepted, len(candidates) - accepted), -remaining.index(pair)

        return max(remaining, key=score)

    def acquire(
        self,
        context: Mapping[str, Any],
        oracle: ChargedExactOracle,
        *,
        mode: str,
        future_reserve: int,
    ) -> JsonDict:
        """Query every public pair and commit only a uniquely closed semantic table."""

        validate_public_context(context)
        if mode not in {"adaptive", "conservative"}:
            raise AcquisitionError("acquisition_mode")
        variables = list(context["variables"])
        pairs = [
            tuple(row) for row in context.get("pair_order", itertools.combinations(variables, 2))
        ]
        tuples = [
            tuple(row) for row in context.get("tuple_order", itertools.product(range(3), repeat=2))
        ]
        compilation_time = 0.0
        before = len(self.committed_relations())
        for pair in pairs:
            if oracle.query_cap - oracle.call_count <= future_reserve:
                self._bounded_update(
                    "uncertainty",
                    {
                        "version_token": self._state["active_version_token"],
                        "pair": list(pair),
                        "reason": "query_budget",
                        "surviving_relation_ids": [],
                    },
                )
                break
            candidates = finite_bias()
            remaining = list(tuples)
            witnesses: list[str] = []
            while remaining and (mode == "conservative" or len(candidates) > 1):
                if oracle.query_cap - oracle.call_count <= future_reserve:
                    break
                probe = (
                    remaining[0]
                    if mode == "conservative"
                    else self._adaptive_tuple(candidates, remaining)
                )
                remaining.remove(probe)
                accepted = oracle.query(
                    make_query(
                        str(context["context_id"]),
                        {pair[0]: probe[0], pair[1]: probe[1]},
                    ),
                    "acquisition_pair",
                )
                witnesses.append(str(oracle.receipts[-1]["query_hash"]))
                candidates = [
                    row
                    for row in candidates
                    if ((probe in {tuple(value) for value in row["allowed_pairs"]}) is accepted)
                ]
                if not candidates:
                    break
            started = time.perf_counter()
            if len(candidates) == 1:
                candidate = candidates[0]
                if candidate["relation_id"] != "unconstrained":
                    row = {
                        "version_token": self._state["active_version_token"],
                        "scope": list(pair),
                        "relation_id": candidate["relation_id"],
                        "allowed_pairs": candidate["allowed_pairs"],
                        "syntactic_aliases": candidate["syntactic_aliases"],
                        "closed_against": len(finite_bias()) - 1,
                        "witness_hashes": witnesses,
                    }
                    if not any(
                        prior["version_token"] == row["version_token"]
                        and prior["scope"] == row["scope"]
                        for prior in self._state["commits"]
                    ):
                        self._bounded_update("commits", row)
            else:
                self._bounded_update(
                    "uncertainty",
                    {
                        "version_token": self._state["active_version_token"],
                        "pair": list(pair),
                        "reason": "out_of_bias" if not candidates else "witnesses_open",
                        "surviving_relation_ids": [row["relation_id"] for row in candidates],
                    },
                )
            compilation_time += time.perf_counter() - started
        return {
            "committed_count": len(self.committed_relations()) - before,
            "uncertainty_count": len(self.uncertainties()),
            "compilation_time_s": compilation_time,
            "state_bytes": len(self.state_bytes()),
        }

    def propose(self, context: Mapping[str, Any], request: Mapping[str, Any]) -> JsonDict:
        """Enumerate public values and use active committed tables only for ordering."""

        variables = list(context["variables"])
        order = list(request["value_order"])
        commits = self.committed_relations()
        for values in itertools.product(order, repeat=len(variables)):
            assignment = dict(zip(variables, values, strict=True))
            if all(
                (assignment[row["scope"][0]], assignment[row["scope"][1]])
                in {tuple(pair) for pair in row["allowed_pairs"]}
                for row in commits
            ):
                return make_query(str(context["context_id"]), assignment)
        raise AcquisitionError("no_public_candidate")


def _public_context(
    context_id: str,
    variable_count: int,
    version_token: str,
    category: str,
    generator: random.Random,
    *,
    domain: Sequence[int] = (0, 1, 2),
) -> JsonDict:
    variables = [f"v{index}" for index in range(variable_count)]
    pairs = [list(pair) for pair in itertools.combinations(variables, 2)]
    tuples = [list(pair) for pair in itertools.product(domain, repeat=2)]
    generator.shuffle(pairs)
    generator.shuffle(tuples)
    return {
        "context_id": context_id,
        "version_token": version_token,
        "category": category,
        "variable_count": variable_count,
        "variables": variables,
        "domain": list(domain),
        "pair_order": pairs,
        "tuple_order": tuples,
        "search_seed": generator.randrange(1, 2**31),
        "future_requests": [
            {"request_id": f"{context_id}-future-{index:02d}", "value_order": list(domain)}
            for index in range(FUTURE_REQUESTS)
        ],
    }


def _supported_record(context: Mapping[str, Any], *, multiple: bool) -> JsonDict:
    variables = list(context["variables"])
    constraints = []
    for index in range(0, len(variables) - 1, 2):
        if index == 0:
            relations = ["neq", "eq_or_next"] if multiple else ["next"]
        else:
            relations = ["eq_or_next", "eq_or_prev"] if multiple else ["eq_or_next"]
        constraints.append({"scope": variables[index : index + 2], "relations": relations})
    return {
        "context_id": context["context_id"],
        "version_token": context["version_token"],
        "kind": "supported_binary",
        "constraints": constraints,
        "ternary_constraints": [],
    }


def _attach_witnesses(context: Mapping[str, Any], record: JsonDict) -> JsonDict:
    evaluator = ExactRelationEvaluator(context, record)
    return {
        **record,
        "positive_witness": evaluator.first_witness(True),
        "negative_witness": evaluator.first_witness(False),
    }


def build_manifests(public_path: Path, private_path: Path) -> JsonDict:
    """Seal public panels and evaluator-only rules before any acquisition outcome."""

    development_rng = random.Random(DEVELOPMENT_SEED)
    evaluation_rng = random.Random(EVALUATION_SEED)
    development: list[JsonDict] = []
    records: dict[str, JsonDict] = {}
    categories = [
        *("one_constraint_per_scope" for _ in range(4)),
        *("multiple_relations_one_scope" for _ in range(3)),
        *("out_of_bias_ternary" for _ in range(2)),
        "malformed_queries",
        "version_change",
        "version_change",
    ]
    for index, category in enumerate(categories):
        token = "opaque-development-version-a" if index != 11 else "opaque-development-version-b"
        context = _public_context(f"development-{index:02d}", 4, token, category, development_rng)
        if category == "out_of_bias_ternary":
            record = {
                "context_id": context["context_id"],
                "version_token": token,
                "kind": "unsupported_ternary",
                "constraints": [],
                "ternary_constraints": [
                    {"scope": context["variables"][:3], "relation": "not_all_equal"}
                ],
            }
        else:
            record = _supported_record(
                context, multiple=category == "multiple_relations_one_scope" or index == 11
            )
        development.append(context)
        records[str(context["context_id"])] = _attach_witnesses(context, record)

    evaluation: list[JsonDict] = []
    arm_rotation: list[JsonDict] = []
    for variable_count in (4, 6, 8):
        for local_index in range(10):
            context_id = f"evaluation-{variable_count}-{local_index:02d}"
            context = _public_context(
                context_id,
                variable_count,
                f"opaque-evaluation-{variable_count}-{local_index:02d}",
                "supported",
                evaluation_rng,
            )
            evaluation.append(context)
            records[context_id] = _attach_witnesses(
                context, _supported_record(context, multiple=local_index % 3 == 0)
            )
            offset = len(evaluation) % len(ARMS)
            arm_rotation.append(
                {"context_id": context_id, "arms": list(ARMS[offset:] + ARMS[:offset])}
            )

    challenges: list[JsonDict] = []
    for index in range(12):
        category = (
            "unsupported_ternary"
            if index < 4
            else "unsupported_domain"
            if index < 8
            else "outside_binary_bias"
        )
        domain = (0, 1, 2, 3) if category == "unsupported_domain" else (0, 1, 2)
        context_id = f"challenge-{index:02d}"
        context = _public_context(
            context_id,
            4,
            f"opaque-challenge-{index:02d}",
            category,
            evaluation_rng,
            domain=domain,
        )
        if category == "unsupported_ternary":
            record = {
                "context_id": context_id,
                "version_token": context["version_token"],
                "kind": "unsupported_ternary",
                "constraints": [],
                "ternary_constraints": [
                    {"scope": context["variables"][:3], "relation": "not_all_equal"}
                ],
            }
        else:
            relation = "eq" if category == "unsupported_domain" else "outside_sum_even"
            record = {
                "context_id": context_id,
                "version_token": context["version_token"],
                "kind": category,
                "constraints": [{"scope": context["variables"][:2], "relations": [relation]}],
                "ternary_constraints": [],
            }
        challenges.append(context)
        records[context_id] = _attach_witnesses(context, record)

    method = {
        "supported_arity": 2,
        "supported_domain": [0, 1, 2],
        "bias": finite_bias(),
        "query_cap_per_arm": QUERY_CAP,
        "state_cap_bytes": STATE_CAP_BYTES,
        "future_requests_per_context": FUTURE_REQUESTS,
        "arms": list(ARMS),
        "partial_semantics": "accepted iff an exact satisfying full extension exists",
    }
    public: JsonDict = {
        "schema": "carnot.exp7351.public-acquisition-manifest.v1",
        "sealed_before_evaluation_opened": True,
        "method_frozen_before_evaluation_opened": True,
        "method_checksum": sha256_json(method),
        "development_seed": DEVELOPMENT_SEED,
        "evaluation_seed": EVALUATION_SEED,
        "resampling_seed": RESAMPLING_SEED,
        "query_cap_per_arm": QUERY_CAP,
        "state_cap_bytes": STATE_CAP_BYTES,
        "future_requests_per_context": FUTURE_REQUESTS,
        "development_contexts": development,
        "evaluation_contexts": evaluation,
        "unsupported_challenges": challenges,
        "arm_rotation": arm_rotation,
    }
    public["manifest_hash"] = sha256_json(public)
    _atomic_json(public_path, public)
    private: JsonDict = {
        "schema": "carnot.exp7351.evaluator-private-manifest.v1",
        "public_manifest_sha256": sha256_file(public_path),
        "evaluator_records": records,
        "record_count": len(records),
    }
    private["manifest_hash"] = sha256_json(private)
    _atomic_json(private_path, private)
    return {
        "public_manifest_sha256": sha256_file(public_path),
        "private_manifest_sha256": sha256_file(private_path),
        "development_context_count": len(development),
        "evaluation_context_count": len(evaluation),
        "challenge_context_count": len(challenges),
        "method_checksum": public["method_checksum"],
    }


def manifest_errors(
    public: Mapping[str, Any], private: Mapping[str, Any], public_path: Path
) -> list[str]:
    """Recompute panel, seal, and public/private boundary invariants."""

    errors: list[str] = []
    if len(public.get("development_contexts", [])) != 12:
        errors.append("development_count")
    evaluation = public.get("evaluation_contexts", [])
    if len(evaluation) != 30 or Counter(row.get("variable_count") for row in evaluation) != {
        4: 10,
        6: 10,
        8: 10,
    }:
        errors.append("evaluation_panel")
    if len(public.get("unsupported_challenges", [])) != 12:
        errors.append("challenge_count")
    public_frozen = deepcopy(dict(public))
    public_hash = public_frozen.pop("manifest_hash", None)
    if public_hash != sha256_json(public_frozen):
        errors.append("public_manifest_hash")
    private_frozen = deepcopy(dict(private))
    private_hash = private_frozen.pop("manifest_hash", None)
    if private_hash != sha256_json(private_frozen):
        errors.append("private_manifest_hash")
    if private.get("public_manifest_sha256") != sha256_file(public_path):
        errors.append("manifest_binding")
    public_text = json.dumps(public, sort_keys=True)
    if any(marker in public_text for marker in ('"constraints"', '"scope"', '"witness"')):
        errors.append("private_data_in_public_manifest")
    if len(private.get("evaluator_records", {})) != 54:
        errors.append("private_record_count")
    return sorted(set(errors))


def run_development_controls(public: Mapping[str, Any], private: Mapping[str, Any]) -> JsonDict:
    """Audit supported, unsupported, malformed, and version-change development cases."""

    rows: list[JsonDict] = []
    wrong = 0
    audit_queries = 0
    malformed_rejections = 0
    records = private["evaluator_records"]
    for context in public["development_contexts"]:
        record = records[context["context_id"]]
        evaluator = ExactRelationEvaluator(context, record)
        positive = evaluator.check(record["positive_witness"])
        negative = not evaluator.check(record["negative_witness"])
        learner = FiniteBiasEliminationLearner(str(context["version_token"]))
        oracle = ChargedExactOracle(evaluator)
        learner.acquire(context, oracle, mode="adaptive", future_reserve=FUTURE_REQUESTS)
        context_wrong = 0
        for commit in learner.committed_relations():
            projection = evaluator.audit_projection(commit["scope"])
            audit_queries += len(context["domain"]) ** 2
            if projection != {tuple(pair) for pair in commit["allowed_pairs"]}:
                context_wrong += 1
        if context["category"] == "malformed_queries":
            malformed = (
                {},
                {"request_id": "wrong", "assignments": {"v0": 0}},
                make_query(str(context["context_id"]), {}),
                make_query(str(context["context_id"]), {"missing": 0}),
            )
            malformed_rejections += sum(not evaluator.check(query) for query in malformed)
        wrong += context_wrong
        rows.append(
            {
                "context_id": context["context_id"],
                "category": context["category"],
                "positive_witness_accepted": positive,
                "negative_witness_rejected": negative,
                "wrong_admission_count": context_wrong,
                "committed_relation_count": len(learner.committed_relations()),
                "uncertainty_count": len(learner.uncertainties()),
                "learner_query_count": oracle.call_count,
                "audit_query_count": len(learner.committed_relations())
                * len(context["domain"]) ** 2,
            }
        )
    versions = [
        row for row in public["development_contexts"] if row["category"] == "version_change"
    ]
    first, second = versions
    first_evaluator = ExactRelationEvaluator(first, records[first["context_id"]])
    version_learner = FiniteBiasEliminationLearner(str(first["version_token"]))
    version_learner.acquire(
        first,
        ChargedExactOracle(first_evaluator),
        mode="adaptive",
        future_reserve=FUTURE_REQUESTS,
    )
    old_count = len(version_learner.committed_relations())
    version_learner.activate_version(str(second["version_token"]))
    version_isolation = old_count > 0 and not version_learner.committed_relations()
    counts = dict(
        sorted(Counter(row["category"] for row in public["development_contexts"]).items())
    )
    return {
        "context_count": len(rows),
        "rows": rows,
        "category_counts": counts,
        "wrong_admission_count": wrong,
        "positive_witness_failures": sum(not row["positive_witness_accepted"] for row in rows),
        "negative_witness_failures": sum(not row["negative_witness_rejected"] for row in rows),
        "malformed_query_rejection_count": malformed_rejections,
        "version_isolation_passed": version_isolation,
        "audit_query_count": audit_queries,
        "audit_cost_is_separate": True,
        "passed": wrong == 0
        and all(
            row["positive_witness_accepted"] and row["negative_witness_rejected"] for row in rows
        )
        and malformed_rejections == 4
        and version_isolation,
    }


def _candidate_assignments(context: Mapping[str, Any]) -> list[JsonDict]:
    variables = list(context["variables"])
    domain = list(context["domain"])
    patterns = [
        [domain[0]] * len(variables),
        [domain[index % min(2, len(domain))] for index in range(len(variables))],
        [domain[-1]] * len(variables),
    ]
    fallback_seed = int(sha256_json(context["context_id"])[-8:], 16)
    generator = random.Random(int(context.get("search_seed", fallback_seed)))
    for _index in range(min(QUERY_CAP, len(domain) ** len(variables))):
        patterns.append([generator.choice(domain) for _name in variables])
    seen: set[tuple[int, ...]] = set()
    return [
        make_query(str(context["context_id"]), dict(zip(variables, values, strict=True)))
        for values in patterns
        if not (tuple(values) in seen or seen.add(tuple(values)))
    ]


def _utility(context: Mapping[str, Any], plan: Mapping[str, Any] | None) -> float:
    if plan is None:
        return 0.0
    maximum = max(context["domain"])
    if maximum == 0:
        return 1.0
    values = plan["assignments"].values()
    return sum(1.0 - int(value) / maximum for value in values) / len(context["variables"])


def run_arm(
    context: Mapping[str, Any],
    private_record: Mapping[str, Any],
    arm: str,
    *,
    rotation_index: int,
) -> JsonDict:
    """Run one arm through acquisition, future proposals, fallback, and final checks."""

    if arm not in ARMS:
        raise AcquisitionError("arm")
    started = time.perf_counter()
    evaluator = ExactRelationEvaluator(context, private_record)
    exact_cache: dict[str, bool] | None = {} if arm == "exact_plan_cache_reset" else None
    oracle = ChargedExactOracle(evaluator, exact_cache=exact_cache)
    learner: FiniteBiasEliminationLearner | None = None
    acquisition: JsonDict = {"compilation_time_s": 0.0, "state_bytes": 0}
    unsupported = private_record.get("kind") != "supported_binary" or context["domain"] != [0, 1, 2]
    acquisition_started = time.perf_counter()
    if arm != "exact_plan_cache_reset" and context["domain"] == [0, 1, 2]:
        learner = FiniteBiasEliminationLearner(str(context["version_token"]))
        acquisition = learner.acquire(
            context,
            oracle,
            mode="conservative" if arm == "conservative_acquisition" else "adaptive",
            future_reserve=FUTURE_REQUESTS,
        )
    acquisition_time = time.perf_counter() - acquisition_started
    future_started = time.perf_counter()
    returned: list[JsonDict] = []
    ordering_changes = 0
    exact_final_checks = 0
    infeasible = 0
    candidates = _candidate_assignments(context)
    for future in context["future_requests"]:
        if oracle.call_count >= QUERY_CAP:
            break
        unfiltered = candidates[0]
        proposal = learner.propose(context, future) if learner is not None else unfiltered
        ordering_changes += int(proposal["assignments"] != unfiltered["assignments"])
        accepted = oracle.query(
            proposal,
            "future_candidate" if arm == "exact_plan_cache_reset" else "future_final",
            allow_cache=arm == "exact_plan_cache_reset",
        )
        if arm != "exact_plan_cache_reset":
            exact_final_checks += 1
        selected: JsonDict | None = proposal if accepted else None
        if selected is None:
            for candidate in candidates:
                if oracle.call_count >= QUERY_CAP:
                    break
                if candidate["assignments"] == proposal["assignments"]:
                    continue
                if oracle.query(
                    candidate,
                    "future_fallback",
                    allow_cache=arm == "exact_plan_cache_reset",
                ):
                    selected = candidate
                    break
        if selected is not None and arm == "exact_plan_cache_reset":
            if oracle.call_count >= QUERY_CAP:
                selected = None
            else:
                final_accepted = oracle.query(selected, "future_final", allow_cache=False)
                exact_final_checks += 1
                if not final_accepted:
                    infeasible += 1
                    selected = None
        if selected is not None:
            returned.append(selected)
    future_time = time.perf_counter() - future_started
    state_bytes = (
        len(learner.state_bytes()) if learner is not None else len(canonical_bytes(exact_cache))
    )
    commits = learner.committed_relations() if learner is not None else []
    full_cost = time.perf_counter() - started
    return {
        "panel": "challenge" if unsupported else "evaluation",
        "context_id": context["context_id"],
        "variable_count": len(context["variables"]),
        "arm": arm,
        "rotation_index": rotation_index,
        "query_count": oracle.call_count,
        "query_attempt_count": oracle.attempt_count,
        "cache_hit_count": oracle.cache_hits,
        "acquisition_time_s": acquisition_time,
        "future_verification_time_s": future_time,
        "evaluator_time_s": oracle.evaluator_time_s,
        "compilation_time_s": float(acquisition["compilation_time_s"]),
        "full_cost_s": full_cost,
        "future_request_count": len(context["future_requests"]),
        "completed_future_requests": len(returned),
        "coverage": len(returned),
        "utility": sum(_utility(context, plan) for plan in returned),
        "returned_infeasible_count": infeasible,
        "exact_final_check_count": exact_final_checks,
        "ordering_change_count": ordering_changes,
        "committed_relation_count": len(commits),
        "uncertainty_count": len(learner.uncertainties()) if learner is not None else 0,
        "maximum_state_bytes": state_bytes,
        "unsupported": unsupported,
        "fallback_or_abstention": unsupported,
        "censored": len(returned) != len(context["future_requests"]),
    }


def run_panel(public: Mapping[str, Any], private: Mapping[str, Any]) -> list[JsonDict]:
    """Run all frozen evaluation and challenge units with visible unit progress."""

    records = private["evaluator_records"]
    rotation = {row["context_id"]: row["arms"] for row in public["arm_rotation"]}
    rows: list[JsonDict] = []
    contexts = [*public["evaluation_contexts"], *public["unsupported_challenges"]]
    for context_index, context in enumerate(contexts):
        arms = rotation.get(context["context_id"], list(ARMS))
        for rotation_index, arm in enumerate(arms):
            rows.append(
                run_arm(
                    context,
                    records[context["context_id"]],
                    arm,
                    rotation_index=rotation_index,
                )
            )
        progress(
            "evaluation",
            "unit_complete",
            f"contexts={context_index + 1}/{len(contexts)} rows={len(rows)}",
        )
    return rows


def _paired_cost_ci(rows: Sequence[Mapping[str, Any]], seed: int) -> JsonDict:
    by_context: dict[str, dict[str, float]] = {}
    for row in rows:
        by_context.setdefault(str(row["context_id"]), {})[str(row["arm"])] = float(
            row["full_cost_s"]
        )
    pairs = [
        (values["finite_bias_elimination"], values["conservative_acquisition"])
        for values in by_context.values()
        if {"finite_bias_elimination", "conservative_acquisition"} <= set(values)
    ]
    observed = sum(left for left, _right in pairs) / sum(right for _left, right in pairs)
    generator = random.Random(seed)
    samples = []
    for _index in range(2_000):
        draw = [pairs[generator.randrange(len(pairs))] for _pair in pairs]
        samples.append(sum(left for left, _right in draw) / sum(right for _left, right in draw))
    samples.sort()
    return {
        "estimate": observed,
        "lower": samples[int(0.025 * len(samples))],
        "upper": samples[int(0.975 * len(samples))],
        "context_clusters": len(pairs),
        "resamples": len(samples),
    }


def reduce_rows(rows: Sequence[Mapping[str, Any]], *, resampling_seed: int) -> JsonDict:
    """Independently reduce complete rows into the frozen safety and value gates."""

    evaluation = [row for row in rows if row.get("panel") == "evaluation"]
    finite = [row for row in evaluation if row.get("arm") == "finite_bias_elimination"]
    conservative = [row for row in evaluation if row.get("arm") == "conservative_acquisition"]
    ci = _paired_cost_ci(evaluation, resampling_seed)
    infeasible = sum(int(row.get("returned_infeasible_count", 0)) for row in rows)
    finite_coverage = sum(int(row["coverage"]) for row in finite)
    conservative_coverage = sum(int(row["coverage"]) for row in conservative)
    finite_utility = sum(float(row["utility"]) for row in finite)
    conservative_utility = sum(float(row["utility"]) for row in conservative)
    ordering = sum(int(row["ordering_change_count"]) for row in finite)
    budgets = all(
        int(row["query_count"]) <= QUERY_CAP and int(row["maximum_state_bytes"]) <= STATE_CAP_BYTES
        for row in rows
    )
    value = (
        ci["upper"] < 0.90
        and infeasible == 0
        and finite_coverage >= conservative_coverage
        and finite_utility + 1e-12 >= conservative_utility
        and ordering >= 1
        and budgets
    )
    return {
        "row_count": len(rows),
        "evaluation_row_count": len(evaluation),
        "challenge_row_count": len(rows) - len(evaluation),
        "paired_context_clustered_ci95": ci,
        "returned_infeasible_count": infeasible,
        "finite_bias_coverage": finite_coverage,
        "conservative_coverage": conservative_coverage,
        "finite_bias_utility": finite_utility,
        "conservative_utility": conservative_utility,
        "finite_bias_ordering_change_count": ordering,
        "resource_bounds_passed": budgets,
        "cost_value_gate_passed": value,
        "query_count_is_diagnostic_only": True,
        "rows_sha256": sha256_json(list(rows)),
    }


def precondition_row(
    check: str,
    upstream: str,
    field: str,
    expected: Any,
    observed: Any,
    available: bool,
) -> JsonDict:
    """Record the exact source, field, expectation, observation, and decision."""

    return {
        "check": check,
        "upstream": upstream,
        "artifact_field": field,
        "expected_value": expected,
        "observed_value": observed,
        "available": bool(available),
        "blocking": True,
        "principle": "Dependent work starts only from authenticated eligible inputs.",
    }


def collect_preconditions(root: Path) -> tuple[list[JsonDict], dict[str, str]]:
    """Authenticate exact sources and the same-milestone qualified fixture first."""

    checks: list[JsonDict] = []
    hashes: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        if available:
            hashes[str(relative)] = sha256_file(path)
        checks.append(
            precondition_row(
                f"source_bytes:{relative}",
                str(relative),
                "bytes",
                "readable_nonempty_bytes",
                path.stat().st_size if available else "missing_or_empty",
                available,
            )
        )
    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    checks.append(
        precondition_row(
            "driving_requirement",
            str(SPEC_PATH),
            "REQ-*",
            "REQ-CL-7351",
            "REQ-CL-7351" if "REQ-CL-7351" in spec_text else "missing",
            "REQ-CL-7351" in spec_text,
        )
    )
    exclusion = (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    unlisted = "experiment_7351_v645_acquisition_prototype" not in exclusion
    checks.append(
        precondition_row(
            "current_task_not_quarantined",
            "ops/exclusion_manifest.yaml",
            "experiment_7351_v645_acquisition_prototype",
            True,
            unlisted,
            unlisted,
        )
    )
    try:
        fixture = json.loads((root / FIXTURE_PATH).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        fixture = {}
    fixture_expectations = {
        "experiment_id": "exp7344-executor-fixture",
        "milestone": MILESTONE,
        "status": "complete",
        "executor_fixture_ready_score": 1,
        "flagged_adversarial": False,
        "required_checks_passed": True,
    }
    for field, expected in fixture_expectations.items():
        observed = fixture.get(field, "missing")
        checks.append(
            precondition_row(
                f"fixture_{field}",
                str(FIXTURE_PATH),
                field,
                expected,
                observed,
                observed == expected,
            )
        )
    gate_passed = fixture.get("gate_check_summary", {}).get("passed") is True
    checks.append(
        precondition_row(
            "fixture_gate_summary",
            str(FIXTURE_PATH),
            "gate_check_summary.passed",
            True,
            fixture.get("gate_check_summary", {}).get("passed", "missing"),
            gate_passed,
        )
    )
    return checks, hashes


def _field_principles() -> JsonDict:
    return {
        "schema": "Version the record and retain ordinary top-level experiment_id and milestone.",
        "status": "Write a terminal result only after actual work and affected checks.",
        "run_date": "Use 20260916; record real UTC timestamps as well.",
        "preconditions_checked": "Record each actual input/resource check before dependent work.",
        "MODEL_SPECS": "List actual intended model identities; this run intends no model work.",
        "model_invoked": "True for any attempted current model load or generation, including failures.",
        "invocation_counts": "Separate attempted, completed, failed, cancelled and in-flight operations.",
        "inference_substrate": "Declare actual computation; historical model receipts are not current inference.",
        "inference_substrate_class": "Use the closed duration class matching the actual run.",
        "execution_venue": "Use host; this milestone makes no new board-execution claim.",
        "duration_s": "Measure monotonic time; never wait merely to pass a duration floor.",
        "phase_spans": "Measure disjoint load, generation, evaluation, test and write spans.",
        "random_seed": "Freeze development, evaluation and resampling seeds before outcomes.",
        "reproducibility_checksum": "Bind code, settings, inputs, evaluator identity and raw evidence.",
        "source_artifact_hashes": "Authenticate exact producers and current same-milestone paths.",
        "rows": "Keep every comparative unit, arm, metric, cost, failure and censoring disposition.",
        "sample_size_budget": "Record planned, attempted, completed and censored units and stopping rules.",
        "acceptance_gate_results": "Each gate records expected, observed, passed and its principle.",
        "gate_check_summary": "Every blocked_* names upstream, failed check, exact artifact field, expected and observed value.",
        "verifier_is_oracle": "True when the executor defines correctness; separate code does not remove circularity.",
        "honest_verdict": "Completed work starts complete_ or complete:; external absence starts blocked_ with its failed check.",
        "verdict_class": "Closed enum: positive | circular_positive | null | blocked | disqualified | partial.",
        "flagged_adversarial": "Set false only after current verification; a critical finding prevents promotion.",
        "validation_receipts": "Retain exact command, scope, exit code, elapsed time and log hash, including failures.",
        "repository_health": "Preserve dated unrelated failures separately from affected required validation.",
        "field_principles": "Explain fields separately; do not wrap numeric gates or ordinary dictionaries.",
        "acquisition_prototype_ready_score": "A sound bounded prototype permits measurement, not a speed claim.",
        "acquisition_protocol": "Seal arity, finite bias, exact oracle semantics, caps and controls.",
        "acquisition_manifest_path": "New evaluation contexts remain unopened until the method is frozen.",
        "unsupported_controls": "Record unsupported scope without converting it to binary certainty.",
    }


def _gate(expected: Any, observed: Any, passed: bool, principle: str) -> JsonDict:
    return {
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
        "principle": principle,
    }


def _repository_health(root: Path) -> JsonDict:
    try:
        fixture = json.loads((root / FIXTURE_PATH).read_text(encoding="utf-8"))
        history = fixture.get("repository_health", {}).get("historical_failures", [])
    except (OSError, json.JSONDecodeError):
        history = []
    return {
        "status": "degraded_open" if history else "healthy",
        "affects_required_checks": False,
        "historical_failures": history,
        "current_observation": None,
    }


def base_artifact(checks: Sequence[Mapping[str, Any]], hashes: Mapping[str, str]) -> JsonDict:
    """Create one schema-complete record before scientific work or classification."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": 3,
        "status": "partial",
        "run_date": RUN_DATE,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "completed_at_utc": None,
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "cpu_exact_solver_or_simulator",
        "inference_substrate_class": "cpu_exact_solver_or_simulator",
        "execution_venue": "host",
        "duration_s": 0.0,
        "phase_spans": [],
        "random_seed": {
            "development": DEVELOPMENT_SEED,
            "evaluation": EVALUATION_SEED,
            "resampling": RESAMPLING_SEED,
            "sealed_before_outcomes": True,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": dict(hashes),
        "rows": [],
        "sample_size_budget": {},
        "acceptance_gate_results": {},
        "gate_check_summary": {},
        "verifier_is_oracle": True,
        "honest_verdict": "partial_work_in_progress",
        "verdict_class": "partial",
        "flagged_adversarial": False,
        "validation_receipts": [],
        "required_checks_passed": False,
        "repository_health": _repository_health(REPO_ROOT),
        "field_principles": _field_principles(),
        "acquisition_prototype_ready_score": 0,
        "acquisition_value_score": 0,
        "promotion_score": 0,
        "acquisition_protocol": {
            "supported_arity": 2,
            "supported_domain": [0, 1, 2],
            "finite_bias": finite_bias(),
            "target_scopes_known": False,
            "oracle": "exact Boolean existential partial-assignment evaluator",
            "neural_oracle": False,
            "query_cap_per_arm": QUERY_CAP,
            "state_cap_bytes": STATE_CAP_BYTES,
            "future_requests_per_context": FUTURE_REQUESTS,
            "arms": list(ARMS),
        },
        "acquisition_manifest_path": str(RAW_PATH / "public/acquisition_manifest.json"),
        "unsupported_controls": {
            "planned": 12,
            "binary_certainty_authorized": False,
            "disposition": "abstain_or_exact_fallback",
        },
        "raw_evidence_paths": {},
        "development_controls": {},
        "independent_reduction": {},
        "production_default_changed": False,
        "publication_surface_changed": False,
        "research_roadmap_changed": False,
    }


def _first_failed(checks: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    return next((row for row in checks if row.get("available") is not True), None)


def build_blocked_artifact(
    checks: Sequence[Mapping[str, Any]], hashes: Mapping[str, str]
) -> JsonDict:
    """Return a row-free terminal record when an external prerequisite is absent."""

    artifact = base_artifact(checks, hashes)
    failure = _first_failed(checks)
    if failure is None:
        raise AcquisitionError("blocked_without_failure")
    artifact.update(
        {
            "status": "blocked",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "honest_verdict": f"blocked_{failure['check']}",
            "verdict_class": "blocked",
            "gate_check_summary": {
                "passed": False,
                "failed_check": failure["check"],
                "upstream": failure["upstream"],
                "artifact_field": failure["artifact_field"],
                "expected_value": failure["expected_value"],
                "observed_value": failure["observed_value"],
            },
            "sample_size_budget": {
                "planned_contexts": 42,
                "attempted_contexts": 0,
                "completed_contexts": 0,
                "censored_contexts": 0,
                "stopping_rule": "External prerequisite failure stops dependent work.",
            },
        }
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _span(phase: str, start: float, run_start: float, units: int) -> JsonDict:
    ended = time.monotonic()
    return {
        "phase": phase,
        "start_elapsed_s": start - run_start,
        "end_elapsed_s": ended - run_start,
        "duration_s": ended - start,
        "completed_units": units,
        "pending_operations": [],
    }


def build_artifact(root: Path, raw_dir: Path) -> JsonDict:
    """Run preconditions, freeze manifests, audit development, and measure panels."""

    run_start = time.monotonic()
    phase_start = time.monotonic()
    checks, hashes = collect_preconditions(root)
    if _first_failed(checks) is not None:
        return build_blocked_artifact(checks, hashes)
    artifact = base_artifact(checks, hashes)
    artifact["phase_spans"].append(_span("preconditions", phase_start, run_start, len(checks)))
    public_path = raw_dir / "public/acquisition_manifest.json"
    private_path = raw_dir / "evaluator/private_manifest.json"
    phase_start = time.monotonic()
    progress("manifest", "start")
    manifest_receipt = build_manifests(public_path, private_path)
    public = json.loads(public_path.read_text(encoding="utf-8"))
    private = json.loads(private_path.read_text(encoding="utf-8"))
    errors = manifest_errors(public, private, public_path)
    artifact["phase_spans"].append(_span("manifest", phase_start, run_start, 54))
    progress("manifest", "end", f"errors={len(errors)}")
    artifact["acquisition_manifest_path"] = str(public_path)
    artifact["source_artifact_hashes"].update(
        {
            str(public_path): sha256_file(public_path),
            str(private_path): sha256_file(private_path),
        }
    )
    sidecar = raw_dir / "sidecars/model_shaped_evidence.json"
    _atomic_json(
        sidecar,
        {
            "schema": "carnot.exp7351.model-shaped-evidence.v1",
            "current_model_work": False,
            "historical_fixture": str(FIXTURE_PATH),
            "historical_fixture_sha256": sha256_file(root / FIXTURE_PATH),
            "scripted_model_shaped_evidence_used": False,
            "disposition": "hash-bound context only; not current inference",
        },
    )
    artifact["source_artifact_hashes"][str(sidecar)] = sha256_file(sidecar)
    phase_start = time.monotonic()
    progress("development", "start")
    controls = run_development_controls(public, private)
    artifact["phase_spans"].append(_span("development", phase_start, run_start, 12))
    progress("development", "end", f"passed={controls['passed']}")
    phase_start = time.monotonic()
    progress("evaluation", "start", "contexts=42 arms=3")
    rows = run_panel(public, private) if controls["passed"] and not errors else []
    artifact["phase_spans"].append(_span("evaluation", phase_start, run_start, len(rows)))
    progress("evaluation", "end", f"rows={len(rows)}")
    rows_path = raw_dir / "evidence/comparative_rows.json"
    _atomic_json(rows_path, {"schema": "carnot.exp7351.rows.v1", "rows": rows})
    reduction = reduce_rows(rows, resampling_seed=RESAMPLING_SEED) if rows else {}
    artifact.update(
        {
            "status": "complete",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "duration_s": time.monotonic() - run_start,
            "rows": rows,
            "development_controls": controls,
            "independent_reduction": reduction,
            "manifest_receipt": manifest_receipt,
            "manifest_errors": errors,
            "raw_evidence_paths": {
                "comparative_rows": str(rows_path),
                "public_manifest": str(public_path),
                "private_manifest": str(private_path),
                "model_shaped_sidecar": str(sidecar),
            },
            "sample_size_budget": {
                "development": {"planned": 12, "attempted": 12, "completed": 12, "censored": 0},
                "evaluation": {"planned": 30, "attempted": 30, "completed": 30, "censored": 0},
                "unsupported_challenges": {
                    "planned": 12,
                    "attempted": 12,
                    "completed": 12,
                    "censored": sum(row["censored"] for row in rows if row["panel"] == "challenge"),
                },
                "arms_per_context": 3,
                "future_requests_per_arm": FUTURE_REQUESTS,
                "query_cap_per_arm": QUERY_CAP,
                "state_cap_bytes": STATE_CAP_BYTES,
                "stopping_rule": "Run every sealed context and arm once; do not extend from outcomes.",
            },
        }
    )
    apply_terminal_state(artifact, require_terminal=False)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _has_receipts(artifact: Mapping[str, Any], names: Sequence[str]) -> bool:
    receipts = artifact.get("validation_receipts", [])
    return all(
        sum(
            row.get("name") == name
            and row.get("passed") is True
            and row.get("exit_code") == 0
            and row.get("timed_out") is not True
            for row in receipts
        )
        == 1
        for name in names
    )


def apply_terminal_state(artifact: JsonDict, *, require_terminal: bool) -> None:
    """Derive readiness and value from current rows and current validation only."""

    failed_precondition = _first_failed(artifact.get("preconditions_checked", []))
    if failed_precondition is not None:
        artifact.update(
            build_blocked_artifact(
                artifact["preconditions_checked"], artifact["source_artifact_hashes"]
            )
        )
        return
    controls = artifact.get("development_controls", {})
    reduction = artifact.get("independent_reduction", {})
    manifests = artifact.get("manifest_errors", [])
    scoped = artifact.get("required_checks_passed") is True and _has_receipts(
        artifact, REQUIRED_CHECK_NAMES
    )
    terminal = _has_receipts(artifact, TERMINAL_CHECK_NAMES) if require_terminal else True
    adversarial = artifact.get("flagged_adversarial") is False
    gates = {
        "qualified_upstream": _gate(
            True,
            True,
            True,
            "The same-milestone executor fixture must be terminal and readiness-qualified.",
        ),
        "sealed_manifests": _gate(
            [], manifests, not manifests, "Public and private seals must replay."
        ),
        "development_controls": _gate(
            {"passed": True, "wrong_admissions": 0},
            {
                "passed": controls.get("passed"),
                "wrong_admissions": controls.get("wrong_admission_count"),
            },
            controls.get("passed") is True and controls.get("wrong_admission_count") == 0,
            "Exhaustive small-domain audits permit no wrong binary admission.",
        ),
        "resource_bounds": _gate(
            True,
            reduction.get("resource_bounds_passed"),
            reduction.get("resource_bounds_passed") is True,
            "Every arm must respect the query and durable-state caps.",
        ),
        "no_infeasible_output": _gate(
            0,
            reduction.get("returned_infeasible_count"),
            reduction.get("returned_infeasible_count") == 0,
            "Every returned future plan needs an exact final acceptance.",
        ),
        "cost_value": _gate(
            True,
            reduction.get("cost_value_gate_passed"),
            reduction.get("cost_value_gate_passed") is True,
            "Full acquisition and future-verification time governs value.",
        ),
        "affected_validation": _gate(True, scoped, scoped, "Current affected checks must pass."),
        "terminal_validation": _gate(
            True, terminal, terminal, "Independent and strict readers must pass."
        ),
        "adversarial_clear": _gate(
            False,
            artifact.get("flagged_adversarial"),
            adversarial,
            "A critical finding prevents readiness.",
        ),
    }
    artifact["acceptance_gate_results"] = gates
    readiness_names = (
        "qualified_upstream",
        "sealed_manifests",
        "development_controls",
        "resource_bounds",
        "no_infeasible_output",
        "affected_validation",
        "terminal_validation",
        "adversarial_clear",
    )
    ready = all(gates[name]["passed"] for name in readiness_names)
    value = ready and gates["cost_value"]["passed"]
    artifact["acquisition_prototype_ready_score"] = int(ready)
    artifact["acquisition_value_score"] = int(value)
    artifact["promotion_score"] = 0
    failed = next(((name, row) for name, row in gates.items() if not row["passed"]), None)
    artifact["gate_check_summary"] = {
        "passed": failed is None,
        "check_count": len(gates),
        "failed_check_count": sum(not row["passed"] for row in gates.values()),
        "failed_check": failed[0] if failed else None,
        "upstream": EXPERIMENT_ID if failed is None else failed[0],
        "artifact_field": "acceptance_gate_results"
        if failed is None
        else f"acceptance_gate_results.{failed[0]}",
        "expected_value": True,
        "observed_value": True if failed is None else failed[1]["observed"],
    }
    artifact["status"] = "complete"
    if not ready and scoped:
        artifact["verdict_class"] = "null"
        artifact["honest_verdict"] = (
            "complete_null: bounded prototype controls passed but the frozen value gate failed"
        )
    elif not ready:
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = (
            "complete_disqualified: current required validation or prototype safety failed"
        )
    elif value:
        artifact["verdict_class"] = "circular_positive"
        artifact["honest_verdict"] = (
            "complete_circular_positive_acquisition_cost_gate_passed_under_exact_evaluator_authority"
        )
    else:
        artifact["verdict_class"] = "null"
        artifact["honest_verdict"] = (
            "complete_null: prototype ready but acquisition cost-value gate failed"
        )


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind scientific inputs and rows while excluding host timing metadata."""

    keys = (
        "schema",
        "experiment_id",
        "milestone",
        "run_date",
        "MODEL_SPECS",
        "inference_substrate",
        "random_seed",
        "source_artifact_hashes",
        "rows",
        "sample_size_budget",
        "acquisition_protocol",
        "development_controls",
        "independent_reduction",
        "acceptance_gate_results",
        "verifier_is_oracle",
    )
    stable = deepcopy({key: artifact.get(key) for key in keys})
    for row in stable.get("rows", []):
        for timing in (
            "acquisition_time_s",
            "future_verification_time_s",
            "evaluator_time_s",
            "compilation_time_s",
            "full_cost_s",
        ):
            row.pop(timing, None)
    return sha256_json(stable)


def independent_reduce(artifact: Mapping[str, Any]) -> list[str]:
    """Reload raw comparative rows and rebuild all deterministic aggregates."""

    errors: list[str] = []
    path_value = artifact.get("raw_evidence_paths", {}).get("comparative_rows")
    if path_value is None:
        return errors
    try:
        raw = json.loads(Path(path_value).read_text(encoding="utf-8"))
        rows = raw["rows"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return ["raw_rows"]
    if rows != artifact.get("rows"):
        errors.append("raw_rows")
    if rows:
        reduction = reduce_rows(rows, resampling_seed=RESAMPLING_SEED)
        if reduction != artifact.get("independent_reduction"):
            errors.append("independent_reduction")
    return sorted(set(errors))


def validate_artifact(artifact: Mapping[str, Any], *, require_validation: bool = True) -> list[str]:
    """Cold-check identity, inference, safety, readiness, and evidence replay."""

    errors: list[str] = []
    if artifact.get("schema") != SCHEMA or artifact.get("experiment_id") != EXPERIMENT_ID:
        errors.append("identity")
    if artifact.get("MODEL_SPECS") != [] or artifact.get("model_invoked") is not False:
        errors.append("model_contract")
    if artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS:
        errors.append("invocation_counts")
    if artifact.get("inference_substrate") != "cpu_exact_solver_or_simulator":
        errors.append("substrate")
    if artifact.get("inference_substrate_class") != "cpu_exact_solver_or_simulator":
        errors.append("substrate_class")
    if artifact.get("execution_venue") != "host":
        errors.append("venue")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class")
    if artifact.get("promotion_score") != 0:
        errors.append("promotion")
    if any(row.get("returned_infeasible_count", 0) for row in artifact.get("rows", [])):
        errors.append("infeasible_output")
    if artifact.get("verdict_class") in {"blocked", "disqualified"} and (
        artifact.get("acquisition_prototype_ready_score") or artifact.get("acquisition_value_score")
    ):
        errors.append("failed_readiness")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum")
    errors.extend(independent_reduce(artifact))
    if require_validation:
        if artifact.get("required_checks_passed") is not True:
            errors.append("required_validation")
        if not _has_receipts(artifact, (*REQUIRED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)):
            errors.append("validation_receipts")
    return sorted(set(errors))


def scoped_command_plan(root: Path, basetemp: Path, coverage_file: Path) -> list[CommandSpec]:
    """Build the exact Exp7303 plan after creating private output parents."""

    basetemp.mkdir(parents=True, exist_ok=True)
    coverage_file.parent.mkdir(parents=True, exist_ok=True)
    commands = build_scoped_commands(
        root,
        [str(TEST_PATH)],
        [str(MODULE_PATH)],
        static_paths=[str(WRAPPER_PATH)],
        basetemp=basetemp,
        coverage_file=coverage_file,
    )
    if any(argument in {"tests", "tests/python", "."} for row in commands for argument in row.argv):
        raise AcquisitionError("broad_scoped_command")
    return commands


def run_affected_validation(root: Path, raw_dir: Path) -> JsonDict:
    """Use the shipped scoped runner with explicit files and changed modules."""

    basetemp = Path("/tmp/carnot-exp7351-v645-scoped")
    coverage_file = Path("/tmp/carnot-exp7351-v645-coverage/.coverage")
    scoped_command_plan(root, basetemp, coverage_file)
    return run_scoped_validation(
        root,
        [str(TEST_PATH)],
        [str(MODULE_PATH)],
        static_paths=[str(WRAPPER_PATH)],
        basetemp=basetemp,
        coverage_file=coverage_file,
        log_dir=raw_dir / "validation/scoped",
        historical_failures=_repository_health(root)["historical_failures"],
    )


def run_terminal_validation(root: Path, candidate: Path, raw_dir: Path) -> list[JsonDict]:
    """Run independent reduction and both required strict terminal readers."""

    python = str(root / ".venv/bin/python")
    reducer = (
        "import json,sys;from pathlib import Path;"
        "from carnot.experiment_7351_v645_acquisition_prototype import independent_reduce;"
        "value=json.loads(Path(sys.argv[1]).read_text());"
        "errors=independent_reduce(value);print(errors,flush=True);raise SystemExit(bool(errors))"
    )
    commands = [
        CommandSpec(
            "independent_reducer", (python, "-u", "-c", reducer, str(candidate)), "raw_rows"
        ),
        CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "terminal_candidate",
        ),
        CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "terminal_candidate",
        ),
    ]
    return run_commands(root, commands, log_dir=raw_dir / "validation/terminal")


def run_full_python_suite(root: Path, raw_dir: Path) -> JsonDict:
    """Run the mandated repository suite once and retain its exact health receipt."""

    command = CommandSpec(
        "full_python_suite",
        (str(root / ".venv/bin/pytest"), "tests/python", "-q"),
        "repository_wide_required_observation",
        timeout_s=4_000.0,
    )
    return run_commands(root, [command], log_dir=raw_dir / "validation/full_suite")[0]


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--validate", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build, validate, classify, and atomically publish the terminal artifact."""

    args = _parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"Exp7351 requires --date {RUN_DATE}")
    if args.validate is not None:
        value = json.loads(args.validate.read_text(encoding="utf-8"))
        errors = validate_artifact(value, require_validation=True)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    root = REPO_ROOT
    output_root = args.output_root.resolve() if args.output_root else root
    result_path = output_root / RESULT_PATH
    raw_dir = output_root / RAW_PATH
    run_started = time.monotonic()
    progress("preconditions", "start")
    artifact = build_artifact(root, raw_dir)
    progress("prototype", "end", f"status={artifact['status']}")
    if artifact["status"] == "blocked":
        _atomic_json(result_path, artifact)
        progress("terminal_write", "end", f"path={result_path}")
        return 0
    candidate = raw_dir / "terminal_candidate.json"
    _atomic_json(candidate, artifact)
    test_started = time.monotonic()
    progress("scoped_validation", "start")
    validation = run_affected_validation(root, raw_dir)
    artifact.update(validation)
    progress("scoped_validation", "end", f"passed={artifact['required_checks_passed']}")
    progress("full_python_suite", "start")
    full_suite = run_full_python_suite(root, raw_dir)
    artifact["repository_health"] = {
        **artifact["repository_health"],
        "current_observation": full_suite,
        "status": "healthy" if full_suite["passed"] else "degraded_open",
        "affects_required_checks": not full_suite["passed"],
    }
    if not full_suite["passed"]:
        artifact["required_checks_passed"] = False
    progress("full_python_suite", "end", f"exit={full_suite['exit_code']}")
    artifact["phase_spans"].append(_span("test", test_started, run_started, 9))
    apply_terminal_state(artifact, require_terminal=False)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    _atomic_json(candidate, artifact)
    progress("terminal_validation", "start")
    terminal = run_terminal_validation(root, candidate, raw_dir)
    artifact["validation_receipts"].extend(terminal)
    if any(row["name"] == "adversarial_verify" and not row["passed"] for row in terminal):
        artifact["flagged_adversarial"] = True
    progress("terminal_validation", "end", f"units={len(terminal)}")
    apply_terminal_state(artifact, require_terminal=True)
    artifact["duration_s"] = time.monotonic() - run_started
    artifact["completed_at_utc"] = datetime.now(UTC).isoformat()
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(
        artifact,
        require_validation=artifact["verdict_class"] not in {"blocked", "disqualified"},
    )
    if errors:
        raise AcquisitionError("terminal_artifact_invalid:" + ",".join(errors))
    progress("terminal_write", "start")
    _atomic_json(result_path, artifact)
    progress("terminal_write", "end", f"path={result_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - the thin script is the supported entrypoint.
    raise SystemExit(main())
