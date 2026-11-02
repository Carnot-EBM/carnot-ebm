"""Build the bounded V643 pairwise and capacity addition prototype.

The learner sees public tasks and Boolean schedule outcomes. The executor keeps
its rule map private. This split matters because a rejected compound schedule
does not identify which hidden rule caused the rejection.

Spec refs: REQ-CL-7323 and SCENARIO-CL-7323-*.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import itertools
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Sequence

from carnot.memory import transactional_constraint_memory as transactional
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    REQUIRED_CHECK_NAMES,
    run_commands,
    run_scoped_validation,
)


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_NAME = "experiment_7323_v643_addition_prototype.json"
SCHEMA = "carnot.experiment_7323.v643_addition_prototype.v1"
QUERY_BUDGET = 24
STATE_CAP_BYTES = 69_632
SEPARATION_DOMAIN = tuple(range(16))
CAPACITY_DOMAIN = tuple(range(1, 7))
DEVELOPMENT_SEED = 7_323_101
EVALUATION_SEED = 7_323_211
CHALLENGE_SEED = 7_323_307
PRIMARY_ARMS = (
    "persistent_structural_acquisition",
    "reset_each_request_acquisition",
    "exact_plan_cache_reset_learner",
    "frozen_after_four_request_warmup",
)
DIAGNOSTIC_ARM = "label_shuffled_diagnostic"
ALL_ARMS = (*PRIMARY_ARMS, DIAGNOSTIC_ARM)
REQUIRED_SOURCE_PATHS = (
    "openspec/capabilities/continuous-learning/spec.md",
    "research-references.md",
    "ops/exclusion_manifest.yaml",
    "ops/e2e-test-plan.md",
    "python/carnot/memory/transactional_constraint_memory.py",
    "python/carnot/reporting/experiment_7303_validation_scope.py",
    "python/carnot/experiment_7323_v643_addition_prototype.py",
    "scripts/experiments/experiment_7323_v643_addition_prototype.py",
    "tests/python/test_experiment_7323_v643_addition_prototype.py",
)


class AdditionRejected(RuntimeError):
    """Reject an unsafe or over-budget structural memory change."""


def canonical_bytes(value: Any) -> bytes:
    """Use one byte form for cache, state, stream, and artifact identities."""

    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    """Prefix hashes so evidence identities are not confused with raw text."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a JSON-compatible value after canonical serialization."""

    return sha256_bytes(canonical_bytes(value))


def sha256_file(path: Path) -> str:
    """Authenticate the exact source bytes used by this run."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _pair_key(left: str, right: str) -> str:
    """Give an unordered activity pair one stable public name."""

    return "|".join(sorted((left, right)))


def _plan(request_id: str, assignments: Mapping[str, int]) -> JsonDict:
    """Keep the public request identity inside exact plan bytes."""

    return {
        "request_id": request_id,
        "assignments": {name: int(slot) for name, slot in sorted(assignments.items())},
    }


def _utility(request: Mapping[str, Any], plan: Mapping[str, Any] | None) -> float:
    """Reward earlier allowed slots without reading private constraints."""

    if plan is None:
        return 0.0
    assignments = plan["assignments"]
    penalties = []
    for name in request["activities"]:
        slots = list(request["allowed_slots"][name])
        width = max(slots) - min(slots) + 1
        penalties.append((int(assignments[name]) - min(slots)) / width)
    return 1.0 - sum(penalties) / len(penalties)


class BooleanScheduleExecutor:
    """Return only schedule feasibility while retaining private rules internally."""

    def __init__(self, version: str, private_rules: Mapping[str, Any]) -> None:
        self.version = str(version)
        self._rules = deepcopy(dict(private_rules))
        self.identity = sha256_json({"version": self.version, "rules": self._rules})

    def check(self, request: Mapping[str, Any], plan: Mapping[str, Any]) -> bool:
        """Evaluate one public plan without explaining a hidden failure."""

        if plan.get("request_id") != request.get("request_id"):
            return False
        assignments = plan.get("assignments")
        if not isinstance(assignments, Mapping) or not assignments:
            return False
        activities = set(request["activities"])
        if not set(assignments).issubset(activities):
            return False
        for name, slot in assignments.items():
            if slot not in request["allowed_slots"][name]:
                return False
        for key, minimum in self._rules["pairwise_separation"].items():
            left, right = key.split("|")
            if left in assignments and right in assignments:
                if abs(int(assignments[left]) - int(assignments[right])) < int(minimum):
                    return False
        counts: dict[int, int] = {}
        for slot in assignments.values():
            counts[int(slot)] = counts.get(int(slot), 0) + 1
        return max(counts.values()) <= int(self._rules["capacity"])

    def optimal_utility(self, request: Mapping[str, Any]) -> float:
        """Score attainable reward after execution for utility normalization."""

        names = list(request["activities"])
        best = 0.0
        for values in itertools.product(*(request["allowed_slots"][name] for name in names)):
            candidate = _plan(str(request["request_id"]), dict(zip(names, values, strict=True)))
            if self.check(request, candidate):
                best = max(best, _utility(request, candidate))
        return best


class ChargedOracle:
    """Enforce one request budget and retain receipts for every attempted query."""

    def __init__(
        self,
        executor: BooleanScheduleExecutor,
        request: Mapping[str, Any],
        *,
        exact_cache: dict[str, bool] | None = None,
    ) -> None:
        self.executor = executor
        self.request = deepcopy(dict(request))
        self.exact_cache = exact_cache
        self.call_count = 0
        self.attempt_count = 0
        self.failed_call_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.receipts: list[JsonDict] = []

    def query(self, plan: Mapping[str, Any], reason: str, *, allow_cache: bool = False) -> bool:
        """Charge external work; a final check can never use cached authority."""

        self.attempt_count += 1
        plan_bytes = canonical_bytes(plan)
        cache_key = sha256_json({"version": self.executor.version, "plan": json.loads(plan_bytes)})
        cache_allowed = allow_cache and reason != "final" and self.exact_cache is not None
        if cache_allowed and cache_key in self.exact_cache:
            accepted = bool(self.exact_cache[cache_key])
            self.cache_hits += 1
            external_call = False
        else:
            if self.call_count >= QUERY_BUDGET:
                raise AdditionRejected("query_budget")
            accepted = self.executor.check(self.request, plan)
            self.call_count += 1
            self.failed_call_count += int(not accepted)
            external_call = True
            if cache_allowed:
                self.cache_misses += 1
                assert self.exact_cache is not None
                self.exact_cache[cache_key] = accepted
        self.receipts.append(
            {
                "sequence": self.attempt_count,
                "charged_call_index": self.call_count,
                "reason": reason,
                "version": self.executor.version,
                "plan_hash": sha256_bytes(plan_bytes),
                "accepted": accepted,
                "external_call": external_call,
                "cache_hit": not external_call,
            }
        )
        return accepted


def make_atom(
    kind: str,
    version: str,
    payload: Mapping[str, Any],
    witness: Mapping[str, Any],
    query_receipts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Create one hash-named atom from evidence the learner is allowed to see."""

    body = {
        "kind": kind,
        "version": version,
        "payload": deepcopy(dict(payload)),
        "witness": deepcopy(dict(witness)),
        "query_receipts": [deepcopy(dict(row)) for row in query_receipts],
    }
    body["atom_id"] = sha256_json(body)
    return body


def _atom_allows_plan(atom: Mapping[str, Any], plan: Mapping[str, Any]) -> bool:
    """Evaluate a learned atom without consulting the hidden executor."""

    assignments = plan["assignments"]
    payload = atom["payload"]
    if atom["kind"] == "pairwise_separation":
        left, right = payload["pair"]
        if left not in assignments or right not in assignments:
            return True
        return abs(int(assignments[left]) - int(assignments[right])) >= int(payload["minimum"])
    counts: dict[int, int] = {}
    for slot in assignments.values():
        counts[int(slot)] = counts.get(int(slot), 0) + 1
    return max(counts.values(), default=0) <= int(payload["maximum"])


class StructuralAdditionLearner:
    """Learn conservative atoms from public requests and charged Boolean receipts."""

    def __init__(
        self,
        executor_version: str,
        *,
        memory_cap_bytes: int = STATE_CAP_BYTES,
        frozen: bool = False,
    ) -> None:
        self.memory_cap_bytes = int(memory_cap_bytes)
        self.frozen = bool(frozen)
        self._state: JsonDict = {
            "schema": "carnot.v643.structural-memory.v1",
            "active_version": str(executor_version),
            "atoms": [],
            "uncertain": [],
            "accepted_plans": [],
            "pair_candidates": {},
            "capacity_candidates": {},
        }

    def state_bytes(self) -> bytes:
        """Charge every durable learner field under one canonical state cap."""

        return canonical_bytes(self._state)

    def state_hash(self) -> str:
        """Bind later predictions to the exact durable learner bytes."""

        return sha256_bytes(self.state_bytes())

    def activate_version(self, version: str) -> None:
        """Flush active authority while retaining bounded same-version archives."""

        self._state["active_version"] = str(version)

    def active_atoms(self) -> list[JsonDict]:
        """Return only atoms authenticated for the currently visible executor."""

        version = self._state["active_version"]
        return deepcopy([row for row in self._state["atoms"] if row["version"] == version])

    def uncertain_candidates(self) -> list[JsonDict]:
        """Expose unresolved compound causes instead of inventing a prohibition."""

        return deepcopy(list(self._state["uncertain"]))

    def pair_hypotheses(self, pair: tuple[str, str]) -> tuple[int, ...]:
        """Return the finite survivor domain for one pair in the active version."""

        key = f"{self._state['active_version']}:{_pair_key(*pair)}"
        return tuple(self._state["pair_candidates"].get(key, SEPARATION_DOMAIN))

    def _capacity_hypotheses(self) -> tuple[int, ...]:
        version = self._state["active_version"]
        return tuple(self._state["capacity_candidates"].get(version, CAPACITY_DOMAIN))

    def _check_state_cap(self, previous: JsonDict) -> None:
        if len(self.state_bytes()) > self.memory_cap_bytes:
            self._state = previous
            raise AdditionRejected("persistent_state_cap")

    def retain_accepted_plan(self, plan: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
        """Keep bounded accepted witnesses so later atoms cannot erase successes."""

        previous = deepcopy(self._state)
        self._state["accepted_plans"] = [
            *self._state["accepted_plans"][-15:],
            {
                "version": self._state["active_version"],
                "plan": deepcopy(dict(plan)),
                "receipt": deepcopy(dict(receipt)),
            },
        ]
        self._check_state_cap(previous)

    def admit_atom(self, atom: Mapping[str, Any], *, current_query_index: int) -> JsonDict:
        """Admit one isolated atom only after chronology, version, and replay checks."""

        candidate = deepcopy(dict(atom))
        if candidate.get("version") != self._state["active_version"]:
            raise AdditionRejected("version_mismatch")
        receipts = candidate.get("query_receipts")
        if not isinstance(receipts, list) or not receipts:
            raise AdditionRejected("missing_query_receipts")
        if any(
            int(row.get("sequence", current_query_index + 1)) > current_query_index
            for row in receipts
        ):
            raise AdditionRejected("early_evidence")
        if candidate.get("kind") not in {"pairwise_separation", "capacity"}:
            raise AdditionRejected("unsupported_atom")
        if any(
            row["version"] == candidate["version"] and not _atom_allows_plan(candidate, row["plan"])
            for row in self._state["accepted_plans"]
        ):
            raise AdditionRejected("contradicts_retained_acceptance")
        existing = next(
            (row for row in self._state["atoms"] if row["atom_id"] == candidate.get("atom_id")),
            None,
        )
        if existing is not None:
            return deepcopy(existing)
        previous = deepcopy(self._state)
        self._state["atoms"].append(candidate)
        self._state["atoms"].sort(key=lambda row: row["atom_id"])
        self._check_state_cap(previous)
        return deepcopy(candidate)

    def import_certified_atoms(self, atoms: Sequence[Mapping[str, Any]]) -> None:
        """Restore records already certified by transactional memory."""

        previous = deepcopy(self._state)
        for atom in atoms:
            candidate = deepcopy(dict(atom))
            if candidate.get("atom_id") != sha256_json(
                {
                    key: candidate[key]
                    for key in ("kind", "version", "payload", "witness", "query_receipts")
                }
            ):
                self._state = previous
                raise AdditionRejected("atom_hash_mismatch")
            if not any(row["atom_id"] == candidate["atom_id"] for row in self._state["atoms"]):
                self._state["atoms"].append(candidate)
        self._state["atoms"].sort(key=lambda row: row["atom_id"])
        self._check_state_cap(previous)

    def propose(self, request: Mapping[str, Any]) -> JsonDict:
        """Enumerate the small public domain using admitted atoms only."""

        started = time.monotonic()
        names = list(request["activities"])
        atoms = self.active_atoms()
        selected: JsonDict | None = None
        for values in itertools.product(*(request["allowed_slots"][name] for name in names)):
            candidate = _plan(str(request["request_id"]), dict(zip(names, values, strict=True)))
            if all(_atom_allows_plan(atom, candidate) for atom in atoms):
                selected = candidate
                break
        elapsed = time.monotonic() - started
        if elapsed > 2.0:
            raise AdditionRejected("solver_time_limit")
        if selected is None:
            return {"plan": None, "solve_duration_s": elapsed, "influenced_atom_ids": []}
        unconstrained = _plan(
            str(request["request_id"]),
            {name: int(request["allowed_slots"][name][0]) for name in names},
        )
        influenced = [
            atom["atom_id"] for atom in atoms if not _atom_allows_plan(atom, unconstrained)
        ]
        return {
            "plan": selected,
            "solve_duration_s": elapsed,
            "influenced_atom_ids": influenced,
        }

    def _record_uncertainty(self, plan: Mapping[str, Any]) -> int:
        assignments = plan["assignments"]
        previous = deepcopy(self._state)
        row = {
            "version": self._state["active_version"],
            "compound_plan_hash": sha256_json(plan),
            "candidate_pairs": [
                _pair_key(left, right)
                for left, right in itertools.combinations(sorted(assignments), 2)
            ],
            "capacity_candidate": True,
            "resolved_atom_ids": [],
        }
        self._state["uncertain"] = [*self._state["uncertain"][-15:], row]
        self._check_state_cap(previous)
        return len(self._state["uncertain"]) - 1

    def localize_rejection(
        self,
        request: Mapping[str, Any],
        plan: Mapping[str, Any],
        oracle: ChargedOracle,
    ) -> list[JsonDict]:
        """Use charged subplans to isolate conservative pair and capacity atoms."""

        if self.frozen:
            return []
        uncertainty_index = self._record_uncertainty(plan)
        assignments = plan["assignments"]
        pair_results: dict[str, tuple[bool, JsonDict, int]] = {}
        for left, right in itertools.combinations(sorted(assignments), 2):
            pair_plan = _plan(
                str(request["request_id"]),
                {left: int(assignments[left]), right: int(assignments[right])},
            )
            accepted = oracle.query(pair_plan, "localization_pair")
            gap = abs(int(assignments[left]) - int(assignments[right]))
            pair_results[_pair_key(left, right)] = (accepted, deepcopy(oracle.receipts[-1]), gap)
        accepted_same_slot = any(
            accepted and gap == 0 for accepted, _receipt, gap in pair_results.values()
        )
        admitted: list[JsonDict] = []
        for key, (accepted, receipt, gap) in pair_results.items():
            state_key = f"{self._state['active_version']}:{key}"
            candidates = list(self._state["pair_candidates"].get(state_key, SEPARATION_DOMAIN))
            candidates = (
                [value for value in candidates if value <= gap]
                if accepted
                else [value for value in candidates if value > gap]
            )
            self._state["pair_candidates"][state_key] = candidates
            if not accepted and candidates and (gap > 0 or accepted_same_slot):
                atom = make_atom(
                    "pairwise_separation",
                    str(self._state["active_version"]),
                    {"pair": key.split("|"), "minimum": min(candidates)},
                    receipt,
                    [receipt],
                )
                admitted.append(self.admit_atom(atom, current_query_index=oracle.attempt_count))
        groups: dict[int, list[str]] = {}
        for name, slot in assignments.items():
            groups.setdefault(int(slot), []).append(str(name))
        for names in groups.values():
            for triple in itertools.combinations(sorted(names), 3):
                keys = [_pair_key(left, right) for left, right in itertools.combinations(triple, 2)]
                if all(pair_results[key][0] for key in keys):
                    probe = _plan(
                        str(request["request_id"]),
                        {name: int(assignments[name]) for name in triple},
                    )
                    accepted = oracle.query(probe, "localization_capacity")
                    receipt = deepcopy(oracle.receipts[-1])
                    candidates = list(self._capacity_hypotheses())
                    candidates = (
                        [value for value in candidates if value >= 3]
                        if accepted
                        else [value for value in candidates if 2 <= value < 3]
                    )
                    self._state["capacity_candidates"][self._state["active_version"]] = candidates
                    if not accepted and candidates:
                        atom = make_atom(
                            "capacity",
                            str(self._state["active_version"]),
                            {"maximum": max(candidates)},
                            receipt,
                            [*[pair_results[key][1] for key in keys], receipt],
                        )
                        admitted.append(
                            self.admit_atom(atom, current_query_index=oracle.attempt_count)
                        )
                    break
            else:
                continue
            break
        self._state["uncertain"][uncertainty_index]["resolved_atom_ids"] = [
            atom["atom_id"] for atom in admitted
        ]
        return admitted

    def run_request(
        self,
        request: Mapping[str, Any],
        oracle: ChargedOracle,
        *,
        allow_cache: bool = False,
    ) -> JsonDict:
        """Optimize, learn from bounded failures, and final-check any returned plan."""

        atoms_before = {row["atom_id"] for row in self.active_atoms()}
        returned: JsonDict | None = None
        final_accepted = False
        solve_time = 0.0
        influenced: list[str] = []
        for attempt in range(3):
            decision = self.propose(request)
            solve_time += float(decision["solve_duration_s"])
            plan = decision["plan"]
            influenced = list(decision["influenced_atom_ids"])
            if plan is None:
                break
            accepted = oracle.query(
                plan,
                "main" if attempt == 0 else "main_retry",
                allow_cache=allow_cache,
            )
            if accepted:
                final_accepted = oracle.query(plan, "final", allow_cache=False)
                if final_accepted:
                    returned = deepcopy(plan)
                    self.retain_accepted_plan(plan, oracle.receipts[-1])
                break
            if self.frozen:
                break
            self.localize_rejection(request, plan, oracle)
        atoms_after = self.active_atoms()
        new_atoms = [row for row in atoms_after if row["atom_id"] not in atoms_before]
        return {
            "returned_plan": returned,
            "final_accepted": final_accepted,
            "oracle_calls": oracle.call_count,
            "oracle_attempts": oracle.attempt_count,
            "failed_calls": oracle.failed_call_count,
            "cache_hits": oracle.cache_hits,
            "cache_misses": oracle.cache_misses,
            "final_checks": sum(row["reason"] == "final" for row in oracle.receipts),
            "new_atoms": deepcopy(new_atoms),
            "active_atom_count": len(atoms_after),
            "influenced_atom_ids": influenced,
            "solve_duration_s": solve_time,
            "state_bytes": len(self.state_bytes()),
            "state_hash": self.state_hash(),
        }


def _public_request(environment_id: str, index: int) -> JsonDict:
    """Create a distinct public task without embedding private rule changes."""

    offset = index % 2
    base = [offset, offset + 1, offset + 2, offset + 3]
    return {
        "request_id": f"{environment_id}-request-{index:02d}",
        "activities": ["a", "b", "c", "d"],
        "allowed_slots": {name: list(base) for name in ("a", "b", "c", "d")},
        "public_task_revision": index,
    }


def _version_schedule(environment_id: str, stratum: str) -> list[str]:
    """Expose announced authority changes without exposing their rules."""

    first = f"{environment_id}-executor-a"
    second = f"{environment_id}-executor-b"
    if stratum in {"development", "stationary"}:
        return [first] * 24
    if stratum == "announced_version_change":
        return [first] * 8 + [second] * 16
    return [first] * 8 + [second] * 8 + [first] * 8


def _private_rules(version: str) -> JsonDict:
    """Derive evaluator-only rules from the authenticated version suffix."""

    if version.endswith("-b"):
        return {"pairwise_separation": {"a|b": 1, "c|d": 2}, "capacity": 3}
    return {"pairwise_separation": {"a|b": 2}, "capacity": 2}


def _environment(environment_id: str, stratum: str) -> JsonDict:
    versions = _version_schedule(environment_id, stratum)
    public = [_public_request(environment_id, index) for index in range(24)]
    private_view = [
        {"request_id": row["request_id"], "version": version, "rules": _private_rules(version)}
        for row, version in zip(public, versions, strict=True)
    ]
    return {
        "environment_id": environment_id,
        "stratum": stratum,
        "public_requests": public,
        "observable_executor_versions": versions,
        "public_view_hash": sha256_json(public),
        "private_executor_view_hash": sha256_json(private_view),
    }


def build_stream_manifest() -> JsonDict:
    """Seal independent development, held-out, and hidden-change views."""

    development = [_environment(f"development-{index:02d}", "development") for index in range(8)]
    strata = [
        *("stationary" for _ in range(8)),
        *("announced_version_change" for _ in range(8)),
        *("return_to_known_version" for _ in range(8)),
    ]
    held_out = [
        _environment(f"held-out-{index:02d}", stratum) for index, stratum in enumerate(strata)
    ]
    challenge_private = {
        "seed": CHALLENGE_SEED,
        "observable_version": "challenge-executor-v1",
        "change_index": 12,
        "before": {"pairwise_separation": {"a|b": 1}, "capacity": 3},
        "after": {"pairwise_separation": {"a|b": 3}, "capacity": 2},
    }
    manifest = {
        "sealed_before_held_out_execution": True,
        "development_seed": DEVELOPMENT_SEED,
        "evaluation_seed": EVALUATION_SEED,
        "challenge_seed": CHALLENGE_SEED,
        "development": {"environments": development},
        "held_out": {
            "environments": held_out,
            "strata": {
                "stationary": 8,
                "announced_version_change": 8,
                "return_to_known_version": 8,
            },
            "executed": False,
        },
        "challenge": {
            "disjoint": True,
            "observable_version_changes": 0,
            "private_view_hash": sha256_json(challenge_private),
            "expected_outcome": "stable_version_assumption_limit_exposed",
        },
    }
    manifest["manifest_hash"] = sha256_json(manifest)
    return manifest


def stream_conformance_errors(manifest: Mapping[str, Any]) -> list[str]:
    """Reject count, public-shape, private-leakage, and seal errors."""

    errors: list[str] = []
    if len(manifest["development"]["environments"]) != 8:
        errors.append("development_count")
    if len(manifest["held_out"]["environments"]) != 24:
        errors.append("held_out_count")
    for split in ("development", "held_out"):
        for environment in manifest[split]["environments"]:
            requests = environment["public_requests"]
            if len(requests) != 24:
                errors.append("request_count")
            if any(len(row["activities"]) > 6 for row in requests):
                errors.append("activity_limit")
            if any(len(slots) > 4 for row in requests for slots in row["allowed_slots"].values()):
                errors.append("slot_limit")
            if any("private" in key for row in requests for key in row):
                errors.append("private_view_leak")
    frozen = deepcopy(dict(manifest))
    observed_hash = frozen.pop("manifest_hash", None)
    if observed_hash != sha256_json(frozen):
        errors.append("manifest_hash")
    return sorted(set(errors))


def run_unannounced_change_challenge() -> JsonDict:
    """Show that unchanged public version bytes cannot authorize drift recovery."""

    request = _public_request("challenge", 12)
    learner = StructuralAdditionLearner("challenge-executor-v1")
    before = BooleanScheduleExecutor(
        "challenge-executor-v1", {"pairwise_separation": {"a|b": 1}, "capacity": 3}
    )
    learner.run_request(request, ChargedOracle(before, request))
    after = BooleanScheduleExecutor(
        "challenge-executor-v1", {"pairwise_separation": {"a|b": 3}, "capacity": 2}
    )
    changed_request = _public_request("challenge", 13)
    stale_plan = learner.propose(changed_request)["plan"]
    changed_oracle = ChargedOracle(after, changed_request)
    exposed = not changed_oracle.query(stale_plan, "main")
    return {
        "control": "unannounced_change",
        "passed": exposed,
        "assumption_limit_exposed": exposed,
        "learned_recovery_claim": False,
        "observable_version_unchanged": True,
    }


def _executor_for(environment: Mapping[str, Any], request_index: int) -> BooleanScheduleExecutor:
    version = str(environment["observable_executor_versions"][request_index])
    return BooleanScheduleExecutor(version, _private_rules(version))


def run_development_panel(manifest: Mapping[str, Any]) -> JsonDict:
    """Run all preregistered arms on development streams only."""

    rows: list[JsonDict] = []
    causal_count = 0
    for environment in manifest["development"]["environments"]:
        for arm in ALL_ARMS:
            persistent: StructuralAdditionLearner | None = None
            exact_cache: dict[str, bool] | None = (
                {} if arm == "exact_plan_cache_reset_learner" else None
            )
            atom_sources: dict[str, str] = {}
            for request_index, request in enumerate(environment["public_requests"]):
                executor = _executor_for(environment, request_index)
                reset = arm in {"reset_each_request_acquisition", "exact_plan_cache_reset_learner"}
                if persistent is None or reset:
                    learner = StructuralAdditionLearner(executor.version)
                    if not reset:
                        persistent = learner
                else:
                    learner = persistent
                    learner.activate_version(executor.version)
                learner.frozen = arm == "frozen_after_four_request_warmup" and request_index >= 4
                oracle = ChargedOracle(executor, request, exact_cache=exact_cache)
                result = learner.run_request(
                    request,
                    oracle,
                    allow_cache=arm == "exact_plan_cache_reset_learner",
                )
                for atom in result["new_atoms"]:
                    atom_sources[atom["atom_id"]] = str(request["request_id"])
                later_influence = sum(
                    atom_sources.get(atom_id) not in {None, request["request_id"]}
                    for atom_id in result["influenced_atom_ids"]
                )
                if arm == "persistent_structural_acquisition":
                    causal_count += later_influence
                returned = result["returned_plan"]
                feasible = returned is not None and executor.check(request, returned)
                attainable = executor.optimal_utility(request)
                memory_bytes = len(learner.state_bytes()) + len(canonical_bytes(exact_cache or {}))
                rows.append(
                    {
                        "environment_id": environment["environment_id"],
                        "stratum": environment["stratum"],
                        "request_id": request["request_id"],
                        "request_index": request_index,
                        "arm": arm,
                        "diagnostic_only": arm == DIAGNOSTIC_ARM,
                        "executor_version": executor.version,
                        "oracle_calls": result["oracle_calls"],
                        "oracle_attempts": result["oracle_attempts"],
                        "failed_calls": result["failed_calls"],
                        "cache_hits": result["cache_hits"],
                        "cache_misses": result["cache_misses"],
                        "final_checks": result["final_checks"],
                        "returned": returned is not None,
                        "abstained": returned is None,
                        "returned_infeasible": returned is not None and not feasible,
                        "utility": _utility(request, returned),
                        "attainable_reward": attainable,
                        "utility_fraction": _utility(request, returned) / attainable
                        if attainable
                        else 0.0,
                        "active_atom_count": result["active_atom_count"],
                        "new_atom_count": len(result["new_atoms"]),
                        "later_distinct_causal_uses": later_influence,
                        "state_bytes": memory_bytes,
                        "solve_duration_s": result["solve_duration_s"],
                        "censored": False,
                    }
                )
    summaries: list[JsonDict] = []
    for arm in ALL_ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        summaries.append(
            {
                "arm": arm,
                "comparative_units": len(arm_rows),
                "total_oracle_calls": sum(row["oracle_calls"] for row in arm_rows),
                "returned_infeasible_count": sum(row["returned_infeasible"] for row in arm_rows),
                "abstention_count": sum(row["abstained"] for row in arm_rows),
                "mean_utility_fraction": sum(row["utility_fraction"] for row in arm_rows)
                / len(arm_rows),
                "maximum_state_bytes": max(row["state_bytes"] for row in arm_rows),
                "censored_count": sum(row["censored"] for row in arm_rows),
            }
        )
    return {
        "rows": rows,
        "arm_summaries": summaries,
        "causal_later_distinct_request_count": causal_count,
        "label_shuffled_is_diagnostic_only": True,
    }


class AdditionMemoryAdapter:
    """Persist opted-in atoms through the shipped transactional memory class."""

    def __init__(
        self,
        state_dir: Path | str,
        *,
        enabled: bool = False,
        memory_cap_bytes: int = STATE_CAP_BYTES,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.enabled = bool(enabled)
        self.memory_cap_bytes = int(memory_cap_bytes)
        self._memory: transactional.TransactionalConstraintMemory | None = None
        if self.enabled:
            self._memory = transactional.TransactionalConstraintMemory(
                self.state_dir / "transactional"
            )

    def _atoms(self) -> list[JsonDict]:
        if self._memory is None:
            return []
        atoms = []
        for record in self._memory.records():
            if str(record.get("key", "")).startswith("v643:") and record.get("certified") is True:
                atoms.append(json.loads(str(record["repair"])))
        return atoms

    def _learner(self, version: str) -> StructuralAdditionLearner:
        learner = StructuralAdditionLearner(version, memory_cap_bytes=self.memory_cap_bytes)
        learner.import_certified_atoms(self._atoms())
        learner.activate_version(version)
        return learner

    def predict(self, request: Mapping[str, Any], executor: BooleanScheduleExecutor) -> JsonDict:
        """Return abstention by default or a proposal from same-version atoms."""

        if not self.enabled:
            return {"status": "abstain", "plan": None, "active_atom_count": 0}
        learner = self._learner(executor.version)
        decision = learner.propose(request)
        return {
            "status": "proposed" if decision["plan"] is not None else "abstain",
            "plan": decision["plan"],
            "active_atom_count": len(learner.active_atoms()),
            "influenced_atom_ids": decision["influenced_atom_ids"],
        }

    def _commit_atom(self, atom: Mapping[str, Any], boundary_index: int) -> JsonDict:
        if self._memory is None:
            raise AdditionRejected("adapter_disabled")
        atom_text = canonical_bytes(atom).decode("utf-8")
        event_id = f"v643-{str(atom['atom_id']).removeprefix('sha256:')}"
        event = {
            "event_id": event_id,
            "kind": "reusable_repair",
            "family": "v643_structural_atom",
            "scope": atom["version"],
            "facts": {"constraint_family": atom["kind"], "scope": atom["version"]},
            "exact_label": True,
            "certified_repair": atom_text,
            "target_key": None,
        }
        proposal = {
            "key": f"v643:{atom['version']}:{atom['atom_id']}",
            "scope": atom["version"],
            "repair": atom_text,
            "source_event_id": event_id,
            "evidence_hash": transactional.event_evidence_hash(event),
            "future_use_eligible": True,
            "expires_after": 1_000_000,
        }
        proposal["content_hash"] = transactional.sha256_json(
            {key: proposal[key] for key in ("key", "scope", "repair")}
        )
        decision = self._memory.admit(proposal, event, boundary_index=boundary_index)
        if not decision["admitted"]:
            raise AdditionRejected(str(decision["reason"]))
        if len(self._memory.state_bytes()) > self.memory_cap_bytes:
            self._memory.rollback(decision["commit_receipt"])
            raise AdditionRejected("persistent_state_cap")
        return decision

    def execute(self, request: Mapping[str, Any], executor: BooleanScheduleExecutor) -> JsonDict:
        """Learn from charged execution and commit only newly justified atoms."""

        if not self.enabled:
            return {"status": "abstain", "returned_plan": None, "final_checks": 0}
        learner = self._learner(executor.version)
        oracle = ChargedOracle(executor, request)
        result = learner.run_request(request, oracle)
        commits = [
            self._commit_atom(atom, boundary_index=oracle.attempt_count)
            for atom in result["new_atoms"]
        ]
        return {**result, "status": "complete", "commit_count": len(commits)}


def run_adapter_e2e(state_dir: Path) -> JsonDict:
    """Exercise prediction, feedback, commit, restart, crash, and invalidation."""

    request = _public_request("adapter", 0)
    executor = BooleanScheduleExecutor(
        "adapter-executor-v1", {"pairwise_separation": {"a|b": 2}, "capacity": 2}
    )
    unconstrained = StructuralAdditionLearner(executor.version).propose(request)["plan"]
    adapter = AdditionMemoryAdapter(state_dir, enabled=True)
    result = adapter.execute(request, executor)
    later_request = _public_request("adapter", 1)
    later = adapter.predict(later_request, executor)
    later_changed = later["plan"]["assignments"] != unconstrained["assignments"]
    assert adapter._memory is not None
    committed_bytes = adapter._memory.state_bytes()
    restarted = AdditionMemoryAdapter(state_dir, enabled=True)
    restart_bytes = restarted._memory.state_bytes() if restarted._memory is not None else b""

    crash_request = {
        "request_id": "crash-probe",
        "activities": ["a", "b"],
        "allowed_slots": {"a": [0], "b": [0]},
    }
    crash_executor = BooleanScheduleExecutor(
        "crash-probe-v1", {"pairwise_separation": {"a|b": 1}, "capacity": 2}
    )
    crash_oracle = ChargedOracle(crash_executor, crash_request)
    crash_plan = _plan("crash-probe", {"a": 0, "b": 0})
    crash_oracle.query(crash_plan, "localization_pair")
    crash_atom = make_atom(
        "pairwise_separation",
        crash_executor.version,
        {"pair": ["a", "b"], "minimum": 1},
        crash_oracle.receipts[-1],
        [crash_oracle.receipts[-1]],
    )
    assert restarted._memory is not None
    restarted._memory.crash_stage = "after_rename"
    try:
        restarted._commit_atom(crash_atom, boundary_index=1)
    except transactional.CrashInjected:
        pass
    crash_reloaded = AdditionMemoryAdapter(state_dir, enabled=True)
    crash_parity = any(atom["atom_id"] == crash_atom["atom_id"] for atom in crash_reloaded._atoms())

    changed_executor = BooleanScheduleExecutor(
        "adapter-executor-v2", {"pairwise_separation": {"a|b": 3}, "capacity": 2}
    )
    invalidated = crash_reloaded.predict(later_request, changed_executor)["active_atom_count"] == 0
    checks = {
        "charged_final_check": result["final_checks"] == 1,
        "later_distinct_prediction_changed": later_changed,
        "commit_persisted": bool(result["commit_count"]) and committed_bytes == restart_bytes,
        "crash_reload_parity": crash_parity,
        "version_invalidated": invalidated,
    }
    return {**checks, "passed": all(checks.values())}


def _control_row(name: str, passed: bool, observed: Any) -> JsonDict:
    return {"control": name, "passed": bool(passed), "observed": observed}


def run_development_controls(state_root: Path) -> list[JsonDict]:
    """Run the preregistered counterexamples before held-out execution."""

    request = {
        "request_id": "controls",
        "activities": ["a", "b", "c", "d"],
        "allowed_slots": {name: [0, 1, 2, 3] for name in "abcd"},
    }
    executor = BooleanScheduleExecutor(
        "controls-v1", {"pairwise_separation": {"a|b": 2}, "capacity": 2}
    )
    learner = StructuralAdditionLearner(executor.version)
    oracle = ChargedOracle(executor, request)
    plan = learner.propose(request)["plan"]
    rejected = not oracle.query(plan, "main")
    no_early_atom = learner.active_atoms() == []
    learner.localize_rejection(request, plan, oracle)
    compound = rejected and no_early_atom and bool(learner.uncertain_candidates())

    accepted_learner = StructuralAdditionLearner("controls-v1")
    accepted_learner.retain_accepted_plan(
        _plan("accepted", {"a": 0, "b": 0}), {"accepted": True, "sequence": 1}
    )
    contradiction_atom = make_atom(
        "pairwise_separation",
        "controls-v1",
        {"pair": ["a", "b"], "minimum": 1},
        {"accepted": False, "sequence": 2},
        [{"accepted": False, "sequence": 2}],
    )
    contradiction = False
    try:
        accepted_learner.admit_atom(contradiction_atom, current_query_index=2)
    except AdditionRejected as error:
        contradiction = str(error) == "contradicts_retained_acceptance"
    early = False
    try:
        accepted_learner.admit_atom(contradiction_atom, current_query_index=1)
    except AdditionRejected as error:
        early = str(error) == "early_evidence"
    stale = deepcopy(contradiction_atom)
    stale["version"] = "controls-v2"
    mismatch = False
    try:
        accepted_learner.admit_atom(stale, current_query_index=2)
    except AdditionRejected as error:
        mismatch = str(error) == "version_mismatch"
    tiny = StructuralAdditionLearner("controls-v1", memory_cap_bytes=400)
    exhausted = False
    try:
        tiny.admit_atom(contradiction_atom, current_query_index=2)
    except AdditionRejected as error:
        exhausted = str(error) == "persistent_state_cap"
    state_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="adapter-control-", dir=state_root) as directory:
        e2e = run_adapter_e2e(Path(directory))
    challenge = run_unannounced_change_challenge()
    return [
        _control_row("compound_conflict", compound, len(learner.active_atoms())),
        _control_row("contradictory_feedback", contradiction, contradiction),
        _control_row("early_labels", early, early),
        _control_row("version_mismatch", mismatch, mismatch),
        _control_row("byte_exhaustion", exhausted, exhausted),
        _control_row("crash_reload", e2e["passed"], e2e),
        _control_row("unannounced_change", challenge["passed"], challenge),
    ]


def collect_preconditions(
    repo_root: Path, *, overrides: Mapping[str, Any] | None = None
) -> tuple[list[JsonDict], JsonDict]:
    """Authenticate owned inputs without treating historical diagnostics as gates."""

    root = repo_root.resolve()
    hashes = {
        relative: sha256_file(root / relative)
        for relative in REQUIRED_SOURCE_PATHS
        if (root / relative).is_file()
    }
    spec_text = (
        (root / REQUIRED_SOURCE_PATHS[0]).read_text(encoding="utf-8")
        if (root / REQUIRED_SOURCE_PATHS[0]).is_file()
        else ""
    )
    exclusion_text = (
        (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
        if (root / "ops/exclusion_manifest.yaml").is_file()
        else ""
    )
    observed: JsonDict = {
        "required_sources_available": len(hashes) == len(REQUIRED_SOURCE_PATHS),
        "capability_requirement_present": "REQ-CL-7323" in spec_text,
        "scenario_contract_present": spec_text.count("SCENARIO-CL-7323-") >= 7,
        "current_task_not_quarantined": "experiment_7323_v643_addition_prototype"
        not in exclusion_text,
        "no_same_milestone_dependencies": [],
        "executor_available": True,
        "host_execution_venue": os.name,
    }
    observed.update(dict(overrides or {}))
    metadata = {
        "required_sources_available": ("repository_worktree", "paths", len(REQUIRED_SOURCE_PATHS)),
        "capability_requirement_present": (REQUIRED_SOURCE_PATHS[0], "REQ-CL-7323", True),
        "scenario_contract_present": (REQUIRED_SOURCE_PATHS[0], "SCENARIO-CL-7323-*", True),
        "current_task_not_quarantined": ("ops/exclusion_manifest.yaml", "experiment_7323", True),
        "no_same_milestone_dependencies": ("milestone_2026.09.643", "dependencies", []),
        "executor_available": ("host_cpu_executor", "available", True),
        "host_execution_venue": ("host", "os.name", "posix"),
    }
    checks = []
    for name, value in observed.items():
        upstream, field, expected = metadata[name]
        expected_value = expected
        if name == "required_sources_available":
            value = len(hashes)
        passed = value == expected_value
        checks.append(
            {
                "upstream": upstream,
                "check": name,
                "field": field,
                "expected_value": expected_value,
                "observed_value": value,
                "passed": passed,
                "principle": "External absence or identity failure stops current execution.",
            }
        )
    return checks, hashes


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Preserve the first exact failed check and every observed value."""

    failures = [deepcopy(dict(row)) for row in checks if row.get("passed") is not True]
    return {
        "passed": not failures,
        "check_count": len(checks),
        "failed_check_count": len(failures),
        "first_failure": failures[0] if failures else None,
        "checks": [deepcopy(dict(row)) for row in checks],
    }


def repository_health() -> JsonDict:
    """Retain the dated unrelated full-suite failure without waiving current checks."""

    return {
        "status": "degraded_open",
        "affects_required_checks": False,
        "historical_failures": [
            {
                "source_experiment_id": 7312,
                "date": "2026-09-14",
                "command": ".venv/bin/pytest tests/python -q",
                "exit_code": 2,
                "duration_s": 3155.878384954063,
                "log_sha256": "sha256:08528e7d4283aefee9ab4e4a1b580cd873793bc1fbf73fb32af5f080c3a0ed75",
                "resolved": False,
                "classification": "unrelated_repository_wide_collection_failures",
            }
        ],
    }


def _acceptance_contract() -> JsonDict:
    return {
        "sealed_before_held_out_execution": True,
        "total_oracle_calls_per_request": {"operator": "<=", "threshold": QUERY_BUDGET},
        "oracle_call_ratio_vs_reset": {"paired_stream_ci95_upper": "<0.90"},
        "oracle_call_ratio_vs_cache": {"paired_stream_ci95_upper": "<0.90"},
        "utility_difference": {"paired_stream_ci95_lower": ">=-0.02 attainable_reward"},
        "feasibility_coverage_difference": {"paired_stream_ci95_lower": ">=-0.02"},
        "returned_infeasible_plans": {"operator": "==", "threshold": 0},
        "stale_version_atoms": {"operator": "==", "threshold": 0},
        "cold_restart_parity": {"operator": "==", "threshold": True},
        "causal_later_distinct_request_count": {"operator": ">=", "threshold": 1},
        "stopping_rule": "execute all 24 sealed held-out streams once in a later efficacy task",
    }


def _field_principles() -> JsonDict:
    return {
        "schema": "Version the record while keeping ordinary experiment identity fields.",
        "status": "Publish only a terminal current-work state.",
        "run_date": "Bind the declared execution date while timestamps retain actual UTC.",
        "preconditions_checked": "Make every input and exact failure inspectable.",
        "MODEL_SPECS": "List current executable model identities only.",
        "model_invoked": "Reveal every attempted model load or generation.",
        "invocation_counts": "Separate attempted and terminal model operations.",
        "inference_substrate": "Name the computation that actually produced rows.",
        "inference_substrate_class": "Use the closed substrate class for automation.",
        "execution_venue": "Distinguish host execution from historical board evidence.",
        "duration_s": "Report measured elapsed time without a synthetic floor.",
        "phase_spans": "Keep disjoint work spans and pending operations auditable.",
        "random_seed": "Freeze independent development and evaluation identities before results.",
        "reproducibility_checksum": "Bind code, views, settings, evaluator identity, and rows.",
        "source_artifact_hashes": "Authenticate producers without granting historical readiness authority.",
        "rows": "Retain every comparative development request and arm.",
        "sample_size_budget": "Separate planned, attempted, complete, and censored units.",
        "acceptance_gate_results": "Keep expected, observed, pass state, and reason together.",
        "gate_check_summary": "Preserve exact upstream, field, expected value, and observed value.",
        "verifier_is_oracle": "Shared authority forbids a positive scientific class.",
        "honest_verdict": "Use a terminal prefix and state the measured scope.",
        "verdict_class": "Use the closed terminal evidence enum.",
        "validation_receipts": "Retain commands, scopes, exits, elapsed times, and log hashes.",
        "repository_health": "Do not convert unrelated historical failures into passing checks.",
        "field_principles": "Explain why required evidence fields exist.",
        "addition_fixture_ready_score": "Credit mechanics, isolation, sealed splits, and controls only.",
        "continuous_self_learning_task": "Identify persistent structural updates from charged feedback.",
        "constraint_language": "State supported pair and capacity rules and unsupported cases.",
        "stream_manifest": "Seal public, private, version, seed, and split identities.",
        "learning_acceptance_contract": "Freeze efficacy and safety thresholds before evaluation.",
        "hardware_path": "Keep measured CPU work separate from future acceleration options.",
    }


def _gate(expected: Any, observed: Any, passed: bool | None, principle: str) -> JsonDict:
    return {"expected": expected, "observed": observed, "passed": passed, "principle": principle}


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Exclude host timing while binding every scientific decision and row."""

    keys = (
        "schema",
        "experiment_id",
        "milestone",
        "run_date",
        "MODEL_SPECS",
        "inference_substrate",
        "random_seed",
        "source_artifact_hashes",
        "constraint_language",
        "stream_manifest",
        "learning_acceptance_contract",
        "development_controls",
        "rows",
        "arm_summaries",
        "sample_size_budget",
        "acceptance_gate_results",
        "verifier_is_oracle",
    )
    return sha256_json({key: artifact.get(key) for key in keys})


def _base_artifact(checks: Sequence[Mapping[str, Any]], hashes: Mapping[str, Any]) -> JsonDict:
    return {
        "schema": SCHEMA,
        "experiment_id": 7323,
        "milestone": "2026.09.643",
        "phase": 3,
        "status": "complete",
        "run_date": "20260915",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "completed_at_utc": None,
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": {
            "loads": {"attempted": 0, "completed": 0, "failed": 0, "cancelled": 0, "in_flight": 0},
            "generations": {
                "attempted": 0,
                "completed": 0,
                "failed": 0,
                "cancelled": 0,
                "in_flight": 0,
            },
        },
        "inference_substrate": "cpu_exact_solver_or_simulator",
        "inference_substrate_class": "cpu_exact_solver_or_simulator",
        "execution_venue": "host",
        "duration_s": 0.0,
        "phase_spans": [],
        "random_seed": {
            "development": DEVELOPMENT_SEED,
            "evaluation": EVALUATION_SEED,
            "unannounced_challenge": CHALLENGE_SEED,
            "sealed_before_results": True,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": {
            "current_sources": deepcopy(dict(hashes)),
            "historical_diagnostics": {
                "results/experiment_7312_v642_factor_audit.json": {
                    "sha256": sha256_file(
                        REPO_ROOT / "results/experiment_7312_v642_factor_audit.json"
                    ),
                    "readiness_authority": False,
                    "disposition": "retired_factor_mechanism_diagnostic_only",
                },
                "results/experiment_7199_v634_bounded_acquisition.json": {
                    "sha256": sha256_file(
                        REPO_ROOT / "results/experiment_7199_v634_bounded_acquisition.json"
                    ),
                    "readiness_authority": False,
                    "disposition": "minimal_core_precedent_diagnostic_only",
                },
            },
        },
        "rows": [],
        "sample_size_budget": {},
        "acceptance_gate_results": {},
        "gate_check_summary": gate_summary(checks),
        "verifier_is_oracle": True,
        "honest_verdict": "",
        "verdict_class": "partial",
        "validation_receipts": [],
        "required_checks_passed": False,
        "repository_health": repository_health(),
        "field_principles": _field_principles(),
        "addition_fixture_ready_score": 0,
        "addition_promotion_score": 0,
        "continuous_self_learning_task": True,
        "constraint_language": {
            "name": "finite_pairwise_minimum_separation_and_global_concurrent_capacity",
            "pairwise_separation_hypotheses": list(SEPARATION_DOMAIN),
            "capacity_hypotheses": list(CAPACITY_DOMAIN),
            "public_request_limit": {"activities": 6, "allowed_slots_per_activity": 4},
            "executor_response": "one Boolean for the entire proposed schedule",
            "supported": "integer slots, selected unordered pairs, one global same-slot capacity",
            "unsupported": [
                "hidden drift without an authenticated version change",
                "soft constraints",
                "activity durations",
                "multiple resource capacities",
                "unbounded time domains",
            ],
        },
        "stream_manifest": {},
        "learning_acceptance_contract": _acceptance_contract(),
        "hardware_path": {
            "measured_now": "CPU bitsets, bounded counters, exact integer enumeration",
            "future": "Sparse integer comparisons and reductions permit Rust, GPU/NPU batching, or FPGA evaluation.",
            "acceleration_claimed": False,
        },
    }


def build_blocked_artifact(
    checks: Sequence[Mapping[str, Any]], hashes: Mapping[str, Any]
) -> JsonDict:
    """Write a schema-complete row-free terminal record for external failure."""

    artifact = _base_artifact(checks, hashes)
    failure = artifact["gate_check_summary"]["first_failure"]
    artifact.update(
        {
            "status": "blocked",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "honest_verdict": f"blocked_{failure['check']}: expected {failure['expected_value']!r}; observed {failure['observed_value']!r}",
            "verdict_class": "blocked",
            "sample_size_budget": {
                "development": {
                    "planned_environments": 8,
                    "attempted_environments": 0,
                    "complete_environments": 0,
                    "censored_environments": 0,
                },
                "held_out": {
                    "planned_environments": 24,
                    "attempted_environments": 0,
                    "complete_environments": 0,
                    "censored_environments": 0,
                },
                "stopping_rule": "external failure is terminal blocked",
            },
        }
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_artifact(repo_root: Path, state_root: Path, *, progress: bool = True) -> JsonDict:
    """Build the complete prototype without running held-out efficacy streams."""

    started = time.monotonic()
    checks, hashes = collect_preconditions(repo_root)
    if not gate_summary(checks)["passed"]:
        return build_blocked_artifact(checks, hashes)
    artifact = _base_artifact(checks, hashes)
    spans = []

    phase_started = time.monotonic()
    if progress:
        print("[exp7323] phase=stream_seal event=start", flush=True)
    manifest = build_stream_manifest()
    spans.append(
        {
            "phase": "stream_seal",
            "start_s": phase_started - started,
            "end_s": time.monotonic() - started,
            "units": 32,
            "checkpoint": "manifest_hash",
            "pending_operations": [],
        }
    )
    if progress:
        print("[exp7323] phase=stream_seal event=end units=32", flush=True)

    phase_started = time.monotonic()
    if progress:
        print("[exp7323] phase=development_controls event=start", flush=True)
    controls = run_development_controls(state_root)
    spans.append(
        {
            "phase": "development_controls",
            "start_s": phase_started - started,
            "end_s": time.monotonic() - started,
            "units": len(controls),
            "checkpoint": "control_rows",
            "pending_operations": [],
        }
    )
    if progress:
        print(f"[exp7323] phase=development_controls event=end units={len(controls)}", flush=True)

    phase_started = time.monotonic()
    if progress:
        print("[exp7323] phase=development_panel event=start units=960", flush=True)
    panel = run_development_panel(manifest)
    spans.append(
        {
            "phase": "development_panel",
            "start_s": phase_started - started,
            "end_s": time.monotonic() - started,
            "units": len(panel["rows"]),
            "checkpoint": "development_rows",
            "pending_operations": [],
        }
    )
    if progress:
        print(f"[exp7323] phase=development_panel event=end units={len(panel['rows'])}", flush=True)

    controls_passed = all(row["passed"] for row in controls)
    conformance = stream_conformance_errors(manifest)
    safety = sum(row["returned_infeasible"] for row in panel["rows"]) == 0
    budgets = all(
        row["oracle_calls"] <= QUERY_BUDGET and row["state_bytes"] <= STATE_CAP_BYTES
        for row in panel["rows"]
    )
    causal = panel["causal_later_distinct_request_count"] >= 1
    artifact.update(
        {
            "stream_manifest": manifest,
            "development_controls": controls,
            "rows": panel["rows"],
            "arm_summaries": panel["arm_summaries"],
            "sample_size_budget": {
                "development": {
                    "planned_environments": 8,
                    "attempted_environments": 8,
                    "complete_environments": 8,
                    "censored_environments": 0,
                    "requests_per_environment": 24,
                    "arms": 5,
                },
                "held_out": {
                    "planned_environments": 24,
                    "attempted_environments": 0,
                    "complete_environments": 0,
                    "censored_environments": 0,
                    "requests_per_environment": 24,
                    "arms": 4,
                },
                "stopping_rule": "prototype stops before sealed held-out efficacy execution",
            },
            "acceptance_gate_results": {
                "prototype_controls": _gate(
                    True,
                    controls_passed,
                    controls_passed,
                    "All development attacks must have their declared outcome.",
                ),
                "sealed_stream_conformance": _gate(
                    [],
                    conformance,
                    not conformance,
                    "Public and private views need exact independent seals.",
                ),
                "development_returned_infeasible": _gate(
                    0,
                    sum(row["returned_infeasible"] for row in panel["rows"]),
                    safety,
                    "A final Boolean check owns every returned plan.",
                ),
                "development_resource_bounds": _gate(
                    True, budgets, budgets, "No request or durable state can exceed its frozen cap."
                ),
                "development_causal_later_use": _gate(
                    ">=1",
                    panel["causal_later_distinct_request_count"],
                    causal,
                    "Learning must affect a later distinct request.",
                ),
                "held_out_oracle_ratio_vs_reset": _gate(
                    "paired CI95 upper<0.90",
                    "not_executed",
                    None,
                    "The efficacy task must measure independent held-out streams.",
                ),
                "held_out_oracle_ratio_vs_cache": _gate(
                    "paired CI95 upper<0.90",
                    "not_executed",
                    None,
                    "Exact caching is a separate comparator.",
                ),
                "held_out_utility": _gate(
                    "paired CI95 lower>=-0.02 attainable reward",
                    "not_executed",
                    None,
                    "Call savings cannot erase attainable utility.",
                ),
                "held_out_coverage": _gate(
                    "paired CI95 lower>=-0.02",
                    "not_executed",
                    None,
                    "Savings cannot come from extra abstention.",
                ),
            },
            "addition_fixture_ready_score": int(
                controls_passed and not conformance and safety and budgets and causal
            ),
            "addition_promotion_score": 0,
            "honest_verdict": "complete: bounded addition fixture ready under shared Boolean executor authority; held-out efficacy not executed",
            "verdict_class": "circular_positive",
            "phase_spans": spans,
            "duration_s": time.monotonic() - started,
            "completed_at_utc": datetime.now(UTC).isoformat(),
        }
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(
    artifact: Mapping[str, Any], *, require_validation: bool = False
) -> list[str]:
    """Cold-check contracts without converting missing evidence into success."""

    errors: list[str] = []
    if artifact.get("schema") != SCHEMA or artifact.get("experiment_id") != 7323:
        errors.append("identity")
    if artifact.get("MODEL_SPECS") != [] or artifact.get("model_invoked") is not False:
        errors.append("model_contract")
    if artifact.get("inference_substrate") != "cpu_exact_solver_or_simulator":
        errors.append("substrate")
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
    rows = artifact.get("rows", [])
    if artifact.get("status") == "blocked" and rows:
        errors.append("blocked_rows")
    if any(int(row.get("oracle_calls", QUERY_BUDGET + 1)) > QUERY_BUDGET for row in rows):
        errors.append("row_query_budget")
    if any(int(row.get("state_bytes", STATE_CAP_BYTES + 1)) > STATE_CAP_BYTES for row in rows):
        errors.append("row_state_cap")
    if any(row.get("returned_infeasible") is True for row in rows):
        errors.append("returned_infeasible")
    if artifact.get("addition_promotion_score") != 0:
        errors.append("premature_promotion")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum")
    if require_validation:
        if artifact.get("required_checks_passed") is not True:
            errors.append("required_validation")
        names = {
            row.get("name")
            for row in artifact.get("validation_receipts", [])
            if row.get("passed") is True
        }
        if not set(REQUIRED_CHECK_NAMES).issubset(names):
            errors.append("validation_receipts")
    ready = artifact.get("addition_fixture_ready_score") == 1
    if ready and artifact.get("verdict_class") != "circular_positive":
        errors.append("fixture_ready_class")
    if artifact.get("verdict_class") in {"blocked", "disqualified"} and ready:
        errors.append("failed_readiness")
    return sorted(set(errors))


def _atomic_write(path: Path, artifact: Mapping[str, Any]) -> None:
    """Publish exact terminal bytes with fsync and one same-directory rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(artifact, sort_keys=True, separators=(",", ":")) + "\n"
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def passing_receipt(name: str) -> JsonDict:
    """Provide a compact test seam for an already completed child command."""

    return {
        "name": name,
        "command": name,
        "scope": "repository_health_observation",
        "exit_code": 0,
        "duration_s": 0.0,
        "log_sha256": "sha256:" + "0" * 64,
        "passed": True,
        "timed_out": False,
        "output_tail": "",
    }


def run_full_python_suite(repo_root: Path, log_dir: Path) -> JsonDict:
    """Run the mandated suite once and preserve failures as repository health."""

    command = CommandSpec(
        "full_python_suite",
        (str(repo_root / ".venv/bin/pytest"), "tests/python", "-q"),
        "repository_wide_health_observation",
        timeout_s=4_000.0,
    )
    return run_commands(repo_root, [command], log_dir=log_dir)[0]


def run_terminal_validators(repo_root: Path, candidate: Path, log_dir: Path) -> list[JsonDict]:
    """Run current adversarial and strict row checks against measured rows."""

    commands = [
        CommandSpec(
            "adversarial_verify",
            (
                str(repo_root / ".venv/bin/python"),
                "-u",
                "scripts/adversarial_verify.py",
                str(candidate),
            ),
            "measured_terminal_candidate",
        ),
        CommandSpec(
            "verdict_row_consistency_strict",
            (
                str(repo_root / ".venv/bin/python"),
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "measured_terminal_candidate",
        ),
    ]
    return run_commands(repo_root, commands, log_dir=log_dir)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--validate", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build, validate, classify, and atomically publish the terminal artifact."""

    args = _parse_args(argv)
    if args.date != "20260915":
        raise SystemExit("Exp7323 requires --date 20260915")
    if args.validate is not None:
        errors = validate_artifact(
            json.loads(args.validate.read_text(encoding="utf-8")), require_validation=True
        )
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    root = REPO_ROOT
    output_root = args.output_root.resolve() if args.output_root else root
    result_path = output_root / "results" / RESULT_NAME
    raw_root = output_root / "results/raw/experiment_7323_v643_addition_prototype"
    state_root = raw_root / "state"
    started = time.monotonic()
    print("[exp7323] phase=preconditions event=start", flush=True)
    artifact = build_artifact(root, state_root, progress=True)
    print(f"[exp7323] phase=prototype event=end status={artifact['status']}", flush=True)
    if artifact["status"] == "blocked":
        _atomic_write(result_path, artifact)
        print(f"[exp7323] phase=terminal_write event=end path={result_path}", flush=True)
        return 0

    candidate = raw_root / "terminal_candidate.json"
    _atomic_write(candidate, artifact)
    print("[exp7323] phase=scoped_validation event=start", flush=True)
    scoped_basetemp = Path("/tmp/carnot-exp7323-scoped")
    scoped_basetemp.mkdir(parents=True, exist_ok=True)
    validation = run_scoped_validation(
        root,
        ["tests/python/test_experiment_7323_v643_addition_prototype.py"],
        ["python/carnot/experiment_7323_v643_addition_prototype.py"],
        static_paths=["scripts/experiments/experiment_7323_v643_addition_prototype.py"],
        basetemp=scoped_basetemp,
        coverage_file=raw_root / ".coverage",
        log_dir=raw_root / "validation/scoped",
        historical_failures=repository_health()["historical_failures"],
    )
    print(
        f"[exp7323] phase=scoped_validation event=end passed={validation['required_checks_passed']}",
        flush=True,
    )
    artifact.update(validation)
    _atomic_write(candidate, artifact)

    print("[exp7323] phase=terminal_validators event=start", flush=True)
    terminal_receipts = run_terminal_validators(root, candidate, raw_root / "validation/terminal")
    artifact["validation_receipts"].extend(terminal_receipts)
    print(
        f"[exp7323] phase=terminal_validators event=end units={len(terminal_receipts)}", flush=True
    )

    print("[exp7323] phase=full_python_suite event=start", flush=True)
    full_receipt = run_full_python_suite(root, raw_root / "validation/full_suite")
    print(
        f"[exp7323] phase=full_python_suite event=end exit={full_receipt['exit_code']}", flush=True
    )
    artifact["repository_health"] = {
        **repository_health(),
        "current_observation": full_receipt,
        "status": "healthy" if full_receipt["passed"] else "degraded_open",
        "affects_required_checks": (
            not full_receipt["passed"]
            and "test_experiment_7323_v643_addition_prototype.py"
            in full_receipt.get("output_tail", "")
        ),
    }
    affected_failure = artifact["repository_health"]["affects_required_checks"]
    current_failures = [row for row in artifact["validation_receipts"] if not row.get("passed")]
    if not artifact["required_checks_passed"] or current_failures or affected_failure:
        artifact["addition_fixture_ready_score"] = 0
        artifact["addition_promotion_score"] = 0
        artifact["verdict_class"] = "disqualified"
        failed_names = [str(row.get("name")) for row in current_failures]
        if affected_failure:
            failed_names.append("affected_full_python_suite")
        artifact["honest_verdict"] = (
            "complete_disqualified: current affected validation failed: " + ",".join(failed_names)
        )
    artifact["duration_s"] = time.monotonic() - started
    artifact["completed_at_utc"] = datetime.now(UTC).isoformat()
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(
        artifact, require_validation=artifact["verdict_class"] != "disqualified"
    )
    if errors:
        raise RuntimeError("terminal_artifact_invalid:" + ",".join(errors))
    print("[exp7323] phase=terminal_write event=start", flush=True)
    _atomic_write(result_path, artifact)
    print(f"[exp7323] phase=terminal_write event=end path={result_path}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - the thin script is the supported entrypoint.
    raise SystemExit(main())
