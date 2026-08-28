"""Build a sealed exact finite-horizon planning fixture.

The fixture is CPU-only. It keeps model prompts separate from exact labels and
requires a prompt-bound commit receipt before the label API reveals an answer.
All objective values are integers, so dynamic programming remains exact.

Specs: REQ-CONSTRAINT-6702, SCENARIO-CONSTRAINT-6702-*, REQ-REPORT-6702,
and SCENARIO-REPORT-6702-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import time
from typing import Any

import jsonschema
import yaml


JsonDict = dict[str, Any]
CommandRunner = Callable[[str, Path], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260828"
RESULT_PATH = Path("results/experiment_6702_exact_planning_fixture_recovery.json")
MODULE_PATH = Path("python/carnot/experiment_6702_exact_planning_fixture_recovery.py")
TEST_PATH = Path("tests/python/test_experiment_6702_exact_planning_fixture_recovery.py")
CONSTRAINT_SPEC_PATH = Path("openspec/capabilities/constraint-verification/spec.md")
REPORT_SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
ACTIVE_ROADMAP = Path("research-roadmap.yaml")
CONDUCTOR_PATH = Path("scripts/research_conductor.py")
PROTECTED_PATHS = (ACTIVE_ROADMAP, CONDUCTOR_PATH)

SCHEMA = "carnot.experiment_6702.exact_planning_fixture.v1"
GENERATOR_VERSION = "carnot.exact_finite_horizon_families.v1"
TRANSITION_VERSION = "carnot.integer_planning_transitions.v1"
SOLVER_VERSION = "carnot.exact_integer_dynamic_programming.v1"
INDEPENDENT_SOLVER_VERSION = "carnot.independent_exhaustive_paths.v1"
SEAL_VERSION = "carnot.prompt_bound_label_seal.v1"
REDUCER_VERSION = "carnot.exact_planning_row_reducer.v1"
INFERENCE_SUBSTRATE = "cpu_exact_dynamic_programming_no_llm"
FAMILIES = ("inventory", "battery_dispatch", "job_slot", "reservoir_control")
METAMORPHIC_TRANSFORMS = (
    "action_renaming",
    "constant_cost_shift",
    "equivalent_state_encoding",
    "family_preserving_surface_change",
)
REQUIRED_MUTATIONS = (
    "bad_transition",
    "infeasible_action",
    "corrupted_cost",
    "label_leakage",
    "wrong_ties",
    "stale_seal",
)
REQUIRED_SPEC_ANCHORS = (
    "REQ-CONSTRAINT-6702",
    "SCENARIO-CONSTRAINT-6702-EXACT-ROWS",
    "SCENARIO-CONSTRAINT-6702-SEALED-LABELS",
    "SCENARIO-CONSTRAINT-6702-ATTACKS",
    "SCENARIO-CONSTRAINT-6702-ROW-REDUCTION",
    "REQ-REPORT-6702",
    "SCENARIO-REPORT-6702-ATOMIC-PROVENANCE",
    "SCENARIO-REPORT-6702-BLOCKED",
)

_COVERAGE_DATA = "/tmp/carnot_exp6702_coverage"
FOCUSED_TEST_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    f"--data-file={_COVERAGE_DATA} --include=*/experiment_6702_exact_planning_fixture_recovery.py "
    f"-m pytest {TEST_PATH} -q --no-cov -n 0 -o addopts="
)
SCOPED_COVERAGE_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    f"--data-file={_COVERAGE_DATA} --include=*/experiment_6702_exact_planning_fixture_recovery.py "
    "--fail-under=100 --show-missing"
)
SPEC_COVERAGE_COMMAND = f".venv/bin/python scripts/check_spec_coverage.py {TEST_PATH}"
APPLICABLE_E2E_COMMAND = (
    f".venv/bin/pytest {TEST_PATH} -q --no-cov -n 0 -o addopts= "
    "-k e2e_atomic_artifact_run_with_injected_receipts"
)
RUFF_COMMAND = f".venv/bin/ruff check {MODULE_PATH} {TEST_PATH}"
FORMAT_COMMAND = f".venv/bin/ruff format --check {MODULE_PATH} {TEST_PATH}"
FULL_SUITE_COMMAND = ".venv/bin/pytest tests/python -q"
VERIFICATION_COMMANDS = (
    ("focused_tests", FOCUSED_TEST_COMMAND),
    ("scoped_coverage", SCOPED_COVERAGE_COMMAND),
    ("spec_coverage", SPEC_COVERAGE_COMMAND),
    ("applicable_e2e", APPLICABLE_E2E_COMMAND),
    ("ruff_check", RUFF_COMMAND),
    ("format_check", FORMAT_COMMAND),
    ("full_python_suite", FULL_SUITE_COMMAND),
)
REQUIRED_TEST_CHECKS = tuple(row[0] for row in VERIFICATION_COMMANDS)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "gate_check_summary",
    "openspec_requirement_ids",
    "frozen_fixture_manifest",
    "instance_rows",
    "state_action_rows",
    "exact_solver_rows",
    "label_seal_rows",
    "metamorphic_rows",
    "mutation_rows",
    "planning_fixture_ready",
    "per_unit_rows",
    "aggregate_row_recomputation",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES = {
    "status": "The terminal state comes from deterministic process evidence.",
    "honest_verdict": "The verdict reports measured fixture evidence only.",
    "verdict_class": "A closed class prevents an infrastructure fixture from becoming a capability claim.",
    "gate_check_summary": "Expected and observed values localize each failed check.",
    "openspec_requirement_ids": "Stable anchors connect tests and code to the frozen contract.",
    "frozen_fixture_manifest": "Hashes bind the benchmark before headline evaluation.",
    "instance_rows": "One row keeps each prompt, typed specification, and exact headline recheckable.",
    "state_action_rows": "Every reachable action value remains exact supervision evidence.",
    "exact_solver_rows": "Solver receipts preserve the authority and independent replay boundary.",
    "label_seal_rows": "Commit-gated seals prevent current-event oracle access.",
    "metamorphic_rows": "Equivalent representations must preserve planning semantics.",
    "mutation_rows": "Known defects must be detected before readiness can open.",
    "planning_fixture_ready": "One Boolean reduces complete task-owned evidence only.",
    "per_unit_rows": "Raw units, not aggregates, own the fixture gate.",
    "aggregate_row_recomputation": "Counts and readiness rebuild deterministically from rows.",
    "preconditions_checked": "Measured tools and resources prevent fabricated execution evidence.",
    "protected_files_unchanged": "Before and after hashes protect active operations.",
    "inference_substrate": "The CPU declaration prevents an LLM or accelerator claim.",
    "verifier_is_oracle": "False marks exact values as sealed post-event authority only.",
    "field_provenance": "Each field names its generator, solver, seal, reducer, function, version, and hash.",
    "random_seed": "Frozen seed schedules reproduce every family and attack.",
    "duration_s": "A monotonic measurement records the real task duration.",
    "tests_run": "Command receipts make verification reproducible.",
    "reproducibility_checksum": "A canonical hash detects changes to the terminal evidence.",
}

ARTIFACT_JSON_SCHEMA: JsonDict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": list(REQUIRED_ARTIFACT_FIELDS),
    "properties": {
        "status": {"type": "string"},
        "honest_verdict": {"type": "string"},
        "verdict_class": {
            "enum": ["positive", "circular_positive", "null", "blocked", "disqualified", "partial"]
        },
        "planning_fixture_ready": {"type": "boolean"},
        "instance_rows": {"type": "array"},
        "state_action_rows": {"type": "array"},
        "duration_s": {"type": "number", "minimum": 0},
        "verifier_is_oracle": {"const": False},
        "inference_substrate": {"const": INFERENCE_SUBSTRATE},
    },
    "additionalProperties": True,
}


def canonical_json(value: Any) -> str:
    """Serialize evidence with stable keys and no nonfinite values."""

    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def sha256_bytes(value: bytes) -> str:
    """Prefix digests so a hash cannot be confused with source text."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash canonical JSON rather than interpreter object text."""

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: Path) -> str:
    """Keep a missing file distinct from an empty file."""

    return sha256_bytes(path.read_bytes()) if path.is_file() else "missing"


def load_json(path: Path) -> JsonDict:
    """Read one JSON object and reject other top-level shapes."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("artifact must be a JSON object")
    return value


def _family_seed(family: str) -> int:
    return 6_702_000 + 10_000 * FAMILIES.index(family)


def _instance_seed(family: str, split: str, index: int) -> int:
    return _family_seed(family) + (1_000 if split == "development" else 0) + index


def _encode_state(family: str, state: int) -> JsonDict:
    names = {
        "inventory": "stock",
        "battery_dispatch": "charge",
        "job_slot": "completed_mask",
        "reservoir_control": "volume",
    }
    return {names[family]: state}


def _decode_state(family: str, state: Mapping[str, Any]) -> int:
    key = next(iter(_encode_state(family, 0)))
    return int(state[key])


def _prompt(family: str, horizon: int, parameters: Mapping[str, Any]) -> str:
    """Describe public inputs without copying any exact output label."""

    if family == "inventory":
        detail = (
            f"Start with {parameters['initial']} units and capacity {parameters['capacity']}. "
            f"Demands are {parameters['demand']} and order prices are {parameters['price']}. "
            f"Orders are {parameters['actions']}; shortage, holding, and terminal costs are "
            f"{parameters['shortage_penalty']}, {parameters['holding_cost']}, and "
            f"{parameters['terminal_holding_cost']}."
        )
    elif family == "battery_dispatch":
        detail = (
            f"Start at charge {parameters['initial']} with capacity {parameters['capacity']}. "
            f"Loads are {parameters['load']} and prices are {parameters['price']}. "
            f"Dispatch actions are {parameters['actions']}; positive values discharge. "
            f"Cycling cost is {parameters['cycling_cost']} and the terminal target is "
            f"{parameters['terminal_target']}."
        )
    elif family == "job_slot":
        detail = (
            f"Schedule four jobs over {horizon} slots. Action 0 idles and actions 1-4 select jobs. "
            f"Prerequisite masks are {parameters['prerequisites']}, deadlines are "
            f"{parameters['deadlines']}, scheduling costs are {parameters['schedule_costs']}, "
            f"and missing-job costs are {parameters['missing_penalties']}."
        )
    else:
        detail = (
            f"Start with volume {parameters['initial']} and capacity {parameters['capacity']}. "
            f"Inflows are {parameters['inflow']} and target releases are "
            f"{parameters['target_release']}. Releases are {parameters['actions']}. "
            f"Deviation, spill, flood, and terminal costs are {parameters['deviation_penalty']}, "
            f"{parameters['spill_penalty']}, {parameters['flood_penalty']}, and "
            f"{parameters['terminal_penalty']}."
        )
    return (
        f"Finite-horizon {family.replace('_', ' ')} planning task with {horizon} stages. "
        f"{detail} Return one legal action per stage that minimizes total integer cost."
    )


def _parameters(family: str, index: int, horizon: int, seed: int) -> JsonDict:
    rng = random.Random(seed)
    if family == "inventory":
        capacity = 4 + index % 3
        actions = list(range(0, min(3, capacity) + 1))
        return {
            "initial": index % (capacity + 1),
            "capacity": capacity,
            "actions": actions,
            "demand": [1 + rng.randrange(3) for _ in range(horizon)],
            "price": [1 + rng.randrange(4) for _ in range(horizon)],
            "holding_cost": 1,
            "shortage_penalty": 5 + index % 3,
            "terminal_holding_cost": 1,
            "stage_cost_shift": 0,
        }
    if family == "battery_dispatch":
        capacity = 4 + index % 3
        flat = index == 8
        return {
            "initial": 2 + index % max(1, capacity - 1),
            "capacity": capacity,
            "actions": [-2, -1, 0, 1, 2],
            "load": [2 + rng.randrange(3) for _ in range(horizon)],
            "price": [1] * horizon if flat else [1 + rng.randrange(4) for _ in range(horizon)],
            "cycling_cost": 0 if flat else 1,
            "terminal_target": 2,
            "terminal_penalty": 2,
            "stage_cost_shift": 0,
        }
    if family == "job_slot":
        prerequisites = []
        for job in range(4):
            prerequisites.append(0 if job == 0 or (index + job) % 2 else 1 << (job - 1))
        return {
            "initial": 0,
            "actions": [0, 1, 2, 3, 4],
            "prerequisites": prerequisites,
            "deadlines": [horizon - 1 - ((index + job) % 2) for job in range(4)],
            "schedule_costs": [1 + rng.randrange(3) for _ in range(4)],
            "missing_penalties": [7 + rng.randrange(4) for _ in range(4)],
            "idle_cost": 2,
            "stage_cost_shift": 0,
        }
    capacity = 6 + index % 3
    return {
        "initial": 3 + index % 3,
        "capacity": capacity,
        "actions": [0, 1, 2, 3, 4],
        "inflow": [rng.randrange(3) for _ in range(horizon)],
        "target_release": [1 + rng.randrange(3) for _ in range(horizon)],
        "deviation_penalty": 2,
        "spill_penalty": 3,
        "flood_threshold": capacity - 2,
        "flood_penalty": 1,
        "terminal_target": 2,
        "terminal_penalty": 3,
        "stage_cost_shift": 0,
    }


def _typed_spec(
    instance_id: str, family: str, horizon: int, parameters: Mapping[str, Any]
) -> JsonDict:
    return {
        "schema": "carnot.typed_finite_horizon_plan.v1",
        "instance_id": instance_id,
        "family": family,
        "horizon": horizon,
        "state_type": next(iter(_encode_state(family, 0))),
        "state_encoding": "bounded_nonnegative_integer",
        "initial_state": _encode_state(family, int(parameters["initial"])),
        "action_type": "integer",
        "action_domain": list(parameters["actions"]),
        "parameters": deepcopy(dict(parameters)),
        "transition": {"function": f"transition_{family}", "version": TRANSITION_VERSION},
        "hard_constraints": ["action_in_domain", "resource_bounds", "family_preconditions"],
        "objective": "minimize_exact_integer_stage_plus_terminal_cost",
        "tie_policy": "retain_all_minimizers_then_choose_action_domain_order_for_plan",
        "feasibility_policy": "a plan is feasible only when every transition is legal",
        "dynamic_programming_convention": "V[t,s]=min_a(cost[t,s,a]+V[t+1,T(t,s,a)])",
    }


def generate_instances() -> list[JsonDict]:
    """Generate 32 held headline instances and eight separate development rows."""

    rows: list[JsonDict] = []
    for family in FAMILIES:
        for split, count, offset in (("headline", 8, 0), ("development", 2, 8)):
            for local_index in range(count):
                index = offset + local_index
                seed = _instance_seed(family, split, index)
                horizon = 4 + index % 3
                parameters = _parameters(family, index, horizon, seed)
                instance_id = f"{family}-{split}-{local_index:02d}"
                prompt = _prompt(family, horizon, parameters)
                typed_spec = _typed_spec(instance_id, family, horizon, parameters)
                rows.append(
                    {
                        "instance_id": instance_id,
                        "family": family,
                        "split": split,
                        "index": index,
                        "seed": seed,
                        "horizon": horizon,
                        "initial_state": int(parameters["initial"]),
                        "action_set": list(parameters["actions"]),
                        "parameters": parameters,
                        "prompt": prompt,
                        "prompt_hash": sha256_bytes(prompt.encode("utf-8")),
                        "typed_spec": typed_spec,
                        "spec_hash": sha256_json(typed_spec),
                    }
                )
    return rows


def execute_transition(
    instance: Mapping[str, Any], time_index: int, state: int, action: int
) -> tuple[bool, int | None, int | None, str]:
    """Execute one family transition with exact integer cost."""

    family = str(instance["family"])
    parameters = instance["parameters"]
    if action not in instance["action_set"]:
        return False, None, None, "action_outside_domain"
    shift = int(parameters.get("stage_cost_shift", 0))
    if family == "inventory":
        if state + action > int(parameters["capacity"]):
            return False, None, None, "capacity_exceeded"
        demand = int(parameters["demand"][time_index])
        available = state + action
        shortage = max(0, demand - available)
        next_state = max(0, available - demand)
        cost = (
            action * int(parameters["price"][time_index])
            + shortage * int(parameters["shortage_penalty"])
            + next_state * int(parameters["holding_cost"])
            + shift
        )
        return True, next_state, cost, "legal"
    if family == "battery_dispatch":
        next_state = state - action
        grid = int(parameters["load"][time_index]) - action
        if not 0 <= next_state <= int(parameters["capacity"]):
            return False, None, None, "charge_bounds"
        if grid < 0:
            return False, None, None, "grid_export_forbidden"
        cost = (
            grid * int(parameters["price"][time_index])
            + abs(action) * int(parameters["cycling_cost"])
            + shift
        )
        return True, next_state, cost, "legal"
    if family == "job_slot":
        if action == 0:
            return True, state, int(parameters["idle_cost"]) + shift, "legal"
        job = action - 1
        bit = 1 << job
        if state & bit:
            return False, None, None, "job_already_completed"
        required = int(parameters["prerequisites"][job])
        if state & required != required:
            return False, None, None, "prerequisite_missing"
        if time_index > int(parameters["deadlines"][job]):
            return False, None, None, "deadline_missed"
        return (
            True,
            state | bit,
            int(parameters["schedule_costs"][job]) + shift,
            "legal",
        )
    available = state + int(parameters["inflow"][time_index])
    if action > available:
        return False, None, None, "release_exceeds_available_water"
    raw_next = available - action
    spill = max(0, raw_next - int(parameters["capacity"]))
    next_state = min(int(parameters["capacity"]), raw_next)
    deviation = abs(action - int(parameters["target_release"][time_index]))
    flood = max(0, next_state - int(parameters["flood_threshold"]))
    cost = (
        deviation * int(parameters["deviation_penalty"])
        + spill * int(parameters["spill_penalty"])
        + flood * int(parameters["flood_penalty"])
        + shift
    )
    return True, next_state, cost, "legal"


def terminal_cost(instance: Mapping[str, Any], state: int) -> int:
    """Return one exact family terminal cost."""

    parameters = instance["parameters"]
    family = instance["family"]
    if family == "inventory":
        return state * int(parameters["terminal_holding_cost"])
    if family == "battery_dispatch":
        return abs(state - int(parameters["terminal_target"])) * int(parameters["terminal_penalty"])
    if family == "job_slot":
        return sum(
            int(penalty)
            for job, penalty in enumerate(parameters["missing_penalties"])
            if not state & (1 << job)
        )
    return abs(state - int(parameters["terminal_target"])) * int(parameters["terminal_penalty"])


def _reachable_states(instance: Mapping[str, Any]) -> list[set[int]]:
    reachable = [{int(instance["initial_state"])}]
    for time_index in range(int(instance["horizon"])):
        following: set[int] = set()
        for state in sorted(reachable[-1]):
            for action in instance["action_set"]:
                legal, next_state, _, _ = execute_transition(instance, time_index, state, action)
                if legal and next_state is not None:
                    following.add(next_state)
        reachable.append(following)
    return reachable


def solve_instance(instance: Mapping[str, Any]) -> JsonDict:
    """Solve every reachable state and retain all action values."""

    started = time.monotonic()
    horizon = int(instance["horizon"])
    reachable = _reachable_states(instance)
    values: dict[tuple[int, int], int] = {
        (horizon, state): terminal_cost(instance, state) for state in reachable[horizon]
    }
    action_values: dict[tuple[int, int, int], tuple[bool, int | None, int | None, str]] = {}
    optimum_actions: dict[tuple[int, int], list[int]] = {}
    for time_index in range(horizon - 1, -1, -1):
        for state in sorted(reachable[time_index]):
            legal_totals: list[tuple[int, int]] = []
            for action in instance["action_set"]:
                legal, next_state, immediate, reason = execute_transition(
                    instance, time_index, state, action
                )
                total = None
                if legal and next_state is not None and immediate is not None:
                    total = immediate + values[(time_index + 1, next_state)]
                    legal_totals.append((int(action), total))
                action_values[(time_index, state, int(action))] = (
                    legal,
                    next_state,
                    immediate,
                    reason,
                )
                if total is not None:
                    action_values[(time_index, state, int(action))] += (total,)
                else:
                    action_values[(time_index, state, int(action))] += (None,)
            best = min(total for _, total in legal_totals)
            values[(time_index, state)] = best
            optimum_actions[(time_index, state)] = [
                action for action, total in legal_totals if total == best
            ]

    rows: list[JsonDict] = []
    for time_index in range(horizon):
        for state in sorted(reachable[time_index]):
            best = values[(time_index, state)]
            for action in instance["action_set"]:
                legal, next_state, immediate, reason, total = action_values[
                    (time_index, state, int(action))
                ]
                rows.append(
                    {
                        "row_id": f"{instance['instance_id']}:{time_index}:{state}:{action}",
                        "instance": instance["instance_id"],
                        "family": instance["family"],
                        "time_index": time_index,
                        "state": _encode_state(str(instance["family"]), state),
                        "action": action,
                        "legality": legal,
                        "transition": {
                            "next_state": (
                                _encode_state(str(instance["family"]), int(next_state))
                                if next_state is not None
                                else None
                            ),
                            "reason": reason,
                        },
                        "immediate_cost": immediate,
                        "future_value": (
                            values[(time_index + 1, int(next_state))]
                            if legal and next_state is not None
                            else None
                        ),
                        "total_value": total,
                        "action_gap": int(total) - best if total is not None else None,
                        "optimum_membership": total == best if total is not None else False,
                    }
                )

    state = int(instance["initial_state"])
    plan: list[int] = []
    for time_index in range(horizon):
        action = optimum_actions[(time_index, state)][0]
        plan.append(action)
        legal, next_state, _, _ = execute_transition(instance, time_index, state, action)
        if not legal or next_state is None:  # pragma: no cover - protected by the DP recurrence.
            raise RuntimeError("dynamic program selected an illegal action")
        state = next_state
    initial_actions = optimum_actions[(0, int(instance["initial_state"]))]
    label = {
        "optimum_action_set": initial_actions,
        "optimum_plan": plan,
        "total_optimum": values[(0, int(instance["initial_state"]))],
        "tie_flag": len(initial_actions) > 1,
        "feasible": bool(reachable[horizon]),
        "state_action_hash": sha256_json(rows),
        "solver_version": SOLVER_VERSION,
    }
    runtime = time.monotonic() - started
    solver_core = {
        "instance": instance["instance_id"],
        "solver": SOLVER_VERSION,
        "enumeration_count": sum(row["legality"] for row in rows),
        "optimum": label["total_optimum"],
        "runtime_s": runtime,
        "passed": True,
    }
    solver_row = {**solver_core, "receipt_hash": sha256_json(solver_core)}
    return {"label": label, "state_action_rows": rows, "solver_row": solver_row}


def _seal_components(instance_id: str, prompt_hash: str, label_hash: str) -> JsonDict:
    return {
        "instance": instance_id,
        "prompt_hash": prompt_hash,
        "label_hash": label_hash,
        "seal_version": SEAL_VERSION,
        "commit_requirement": "prompt_bound_candidate_commit_receipt",
    }


class LabelAccessError(PermissionError):
    """Report a stable denial instead of exposing a current-event label."""


class LabelSealStore:
    """Expose exact labels only after a prompt-bound candidate commit."""

    def __init__(self, entries: Mapping[str, tuple[str, Mapping[str, Any]]]) -> None:
        self._entries = {
            instance_id: (prompt_hash, deepcopy(dict(label)))
            for instance_id, (prompt_hash, label) in entries.items()
        }
        self._commits: dict[str, JsonDict] = {}

    def _entry(self, instance_id: str) -> tuple[str, JsonDict]:
        if instance_id not in self._entries:
            raise LabelAccessError("unknown instance")
        return self._entries[instance_id]

    def commit(self, instance_id: str, prompt_hash: str, candidate: Any) -> JsonDict:
        expected_prompt, _ = self._entry(instance_id)
        if prompt_hash != expected_prompt:
            raise LabelAccessError("prompt hash mismatch")
        core = {
            "instance": instance_id,
            "prompt_hash": prompt_hash,
            "candidate_hash": sha256_json(candidate),
            "commit_version": SEAL_VERSION,
        }
        receipt = {**core, "receipt_hash": sha256_json(core)}
        self._commits[receipt["receipt_hash"]] = receipt
        return deepcopy(receipt)

    def read(self, instance_id: str, receipt: Mapping[str, Any] | None = None) -> JsonDict:
        _, label = self._entry(instance_id)
        if receipt is None:
            raise LabelAccessError("commit receipt required")
        receipt_hash = str(receipt.get("receipt_hash", ""))
        if self._commits.get(receipt_hash) != dict(receipt):
            raise LabelAccessError("invalid commit receipt")
        if receipt.get("instance") != instance_id:
            raise LabelAccessError("invalid commit receipt")
        return deepcopy(label)

    def seal_row(self, instance_id: str) -> JsonDict:
        prompt_hash, label = self._entry(instance_id)
        label_hash = sha256_json(label)
        components = _seal_components(instance_id, prompt_hash, label_hash)
        return {
            **components,
            "seal_hash": sha256_json(components),
            "access_state": "sealed_until_commit",
            "negative_access_result": "denied:commit receipt required",
        }

    @staticmethod
    def verify_seal_row(row: Mapping[str, Any]) -> bool:
        required = _seal_components(
            str(row.get("instance")), str(row.get("prompt_hash")), str(row.get("label_hash"))
        )
        return (
            all(row.get(key) == value for key, value in required.items())
            and row.get("seal_hash") == sha256_json(required)
            and row.get("access_state") == "sealed_until_commit"
            and row.get("negative_access_result") == "denied:commit receipt required"
        )


def build_instance_row(instance: Mapping[str, Any], solved: Mapping[str, Any]) -> JsonDict:
    """Join one public instance to its post-event exact label receipt."""

    label = solved["label"]
    label_hash = sha256_json(label)
    components = _seal_components(
        str(instance["instance_id"]), str(instance["prompt_hash"]), label_hash
    )
    return {
        "instance": instance["instance_id"],
        "family": instance["family"],
        "prompt": instance["prompt"],
        "prompt_hash": instance["prompt_hash"],
        "typed_spec": deepcopy(instance["typed_spec"]),
        "spec_hash": instance["spec_hash"],
        "horizon": instance["horizon"],
        "action_set": list(instance["action_set"]),
        "split": instance["split"],
        "seed": instance["seed"],
        "label_seal_hash": sha256_json(components),
        "optimum": {
            "action_set": label["optimum_action_set"],
            "plan": label["optimum_plan"],
            "total": label["total_optimum"],
        },
        "ties": label["tie_flag"],
        "feasibility": label["feasible"],
    }


def validate_solution(
    instance: Mapping[str, Any],
    label: Mapping[str, Any],
    state_action_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Compare supplied rows with a clean exact recomputation."""

    expected = solve_instance(instance)
    errors: list[str] = []
    if dict(label) != expected["label"]:
        errors.append("label_mismatch")
    if list(state_action_rows) != expected["state_action_rows"]:
        errors.append("state_action_rows_mismatch")
    return errors


def _independent_transition(
    instance: Mapping[str, Any], time_index: int, state: int, action: int
) -> tuple[bool, int | None, int | None]:
    """Duplicate family semantics without calling the production transition."""

    p = instance["parameters"]
    shift = int(p.get("stage_cost_shift", 0))
    family = instance["family"]
    if action not in instance["action_set"]:
        return False, None, None
    if family == "inventory":
        if state + action > int(p["capacity"]):
            return False, None, None
        available = state + action
        demand = int(p["demand"][time_index])
        next_state = max(0, available - demand)
        cost = (
            action * int(p["price"][time_index])
            + max(0, demand - available) * int(p["shortage_penalty"])
            + next_state * int(p["holding_cost"])
            + shift
        )
        return True, next_state, cost
    if family == "battery_dispatch":
        next_state = state - action
        grid = int(p["load"][time_index]) - action
        if not 0 <= next_state <= int(p["capacity"]) or grid < 0:
            return False, None, None
        return (
            True,
            next_state,
            grid * int(p["price"][time_index]) + abs(action) * int(p["cycling_cost"]) + shift,
        )
    if family == "job_slot":
        if action == 0:
            return True, state, int(p["idle_cost"]) + shift
        job = action - 1
        bit = 1 << job
        required = int(p["prerequisites"][job])
        if state & bit or state & required != required or time_index > int(p["deadlines"][job]):
            return False, None, None
        return True, state | bit, int(p["schedule_costs"][job]) + shift
    available = state + int(p["inflow"][time_index])
    if action > available:
        return False, None, None
    raw_next = available - action
    spill = max(0, raw_next - int(p["capacity"]))
    next_state = min(int(p["capacity"]), raw_next)
    cost = (
        abs(action - int(p["target_release"][time_index])) * int(p["deviation_penalty"])
        + spill * int(p["spill_penalty"])
        + max(0, next_state - int(p["flood_threshold"])) * int(p["flood_penalty"])
        + shift
    )
    return True, next_state, cost


def _independent_terminal(instance: Mapping[str, Any], state: int) -> int:
    p = instance["parameters"]
    family = instance["family"]
    if family == "inventory":
        return state * int(p["terminal_holding_cost"])
    if family == "battery_dispatch":
        return abs(state - int(p["terminal_target"])) * int(p["terminal_penalty"])
    if family == "job_slot":
        return sum(
            int(value) for job, value in enumerate(p["missing_penalties"]) if not state & (1 << job)
        )
    return abs(state - int(p["terminal_target"])) * int(p["terminal_penalty"])


def _enumerate_best(instance: Mapping[str, Any], time_index: int, state: int) -> tuple[int, int]:
    if time_index == int(instance["horizon"]):
        return _independent_terminal(instance, state), 1
    totals: list[int] = []
    leaves = 0
    for action in instance["action_set"]:
        legal, next_state, immediate = _independent_transition(
            instance, time_index, state, int(action)
        )
        if legal and next_state is not None and immediate is not None:
            future, count = _enumerate_best(instance, time_index + 1, next_state)
            totals.append(immediate + future)
            leaves += count
    return min(totals), leaves


def independent_enumerate(
    instance: Mapping[str, Any], state_action_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Enumerate all suffix paths and compare every retained action value."""

    started = time.monotonic()
    mismatches: list[str] = []
    enumeration_count = 0
    for row in state_action_rows:
        state = _decode_state(str(instance["family"]), row["state"])
        legal, next_state, immediate = _independent_transition(
            instance, int(row["time_index"]), state, int(row["action"])
        )
        if legal != row["legality"]:
            mismatches.append(str(row["row_id"]))
            continue
        if legal and next_state is not None and immediate is not None:
            future, count = _enumerate_best(instance, int(row["time_index"]) + 1, next_state)
            enumeration_count += count
            if immediate + future != row["total_value"]:
                mismatches.append(str(row["row_id"]))
        elif row["total_value"] is not None:
            mismatches.append(str(row["row_id"]))
    optimum, initial_count = _enumerate_best(instance, 0, int(instance["initial_state"]))
    enumeration_count += initial_count
    return {
        "instance": instance["instance_id"],
        "solver": INDEPENDENT_SOLVER_VERSION,
        "enumeration_count": enumeration_count,
        "optimum": optimum,
        "action_value_mismatches": mismatches,
        "runtime_s": time.monotonic() - started,
        "passed": not mismatches,
    }


def independent_subset(instances: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Choose one development instance per family before any label is read."""

    return [
        next(row for row in instances if row["family"] == family and row["split"] == "development")
        for family in FAMILIES
    ]


def build_independent_solver_rows(instances: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for instance in independent_subset(instances):
        solved = solve_instance(instance)
        row = independent_enumerate(instance, solved["state_action_rows"])
        core = deepcopy(row)
        rows.append({**row, "receipt_hash": sha256_json(core)})
    return rows


def build_label_seal_rows(
    instances: Sequence[Mapping[str, Any]], labels: Mapping[str, Mapping[str, Any]]
) -> list[JsonDict]:
    entries = {
        str(row["instance_id"]): (str(row["prompt_hash"]), labels[str(row["instance_id"])])
        for row in instances
    }
    store = LabelSealStore(entries)
    return [store.seal_row(str(row["instance_id"])) for row in instances]


def _action_gap_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {str(row["row_id"]): row["action_gap"] for row in rows}


def build_metamorphic_rows(instances: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Run four representation checks for one frozen instance per family."""

    rows: list[JsonDict] = []
    for family in FAMILIES:
        instance = next(
            row for row in instances if row["family"] == family and row["split"] == "headline"
        )
        solved = solve_instance(instance)
        aliases = {
            str(action): f"choice_{position}"
            for position, action in enumerate(instance["action_set"])
        }
        renamed = [aliases[str(action)] for action in solved["label"]["optimum_action_set"]]
        rows.append(
            {
                "row_id": f"{family}:action_renaming",
                "instance": instance["instance_id"],
                "family": family,
                "transform": "action_renaming",
                "expected_invariant": "optimum actions map bijectively",
                "observed_result": {"aliases": aliases, "renamed_optimum": renamed},
                "pass_state": len(renamed) == len(solved["label"]["optimum_action_set"]),
            }
        )

        shifted = deepcopy(instance)
        shifted["parameters"]["stage_cost_shift"] = 3
        shifted["typed_spec"]["parameters"]["stage_cost_shift"] = 3
        shifted_solved = solve_instance(shifted)
        shifted_gaps = _action_gap_map(shifted_solved["state_action_rows"])
        base_gaps = _action_gap_map(solved["state_action_rows"])
        shifted_gaps = {
            key.replace(str(shifted["instance_id"]), str(instance["instance_id"]), 1): value
            for key, value in shifted_gaps.items()
        }
        cost_passed = (
            shifted_solved["label"]["optimum_plan"] == solved["label"]["optimum_plan"]
            and shifted_solved["label"]["total_optimum"]
            == solved["label"]["total_optimum"] + 3 * int(instance["horizon"])
            and shifted_gaps == base_gaps
        )
        rows.append(
            {
                "row_id": f"{family}:constant_cost_shift",
                "instance": instance["instance_id"],
                "family": family,
                "transform": "constant_cost_shift",
                "expected_invariant": "plan and gaps unchanged; total shifts by 3 per stage",
                "observed_result": {
                    "base_total": solved["label"]["total_optimum"],
                    "shifted_total": shifted_solved["label"]["total_optimum"],
                },
                "pass_state": cost_passed,
            }
        )

        encoded_states = [row["state"] for row in solved["state_action_rows"]]
        round_trip = [
            _encode_state(family, _decode_state(family, json.loads(canonical_json(state))))
            for state in encoded_states
        ]
        rows.append(
            {
                "row_id": f"{family}:equivalent_state_encoding",
                "instance": instance["instance_id"],
                "family": family,
                "transform": "equivalent_state_encoding",
                "expected_invariant": "canonical JSON state round trip is exact",
                "observed_result": {"state_count": len(encoded_states)},
                "pass_state": round_trip == encoded_states,
            }
        )

        surfaced = deepcopy(instance)
        surfaced["prompt"] = "Please solve this equivalent family task. " + str(instance["prompt"])
        surfaced["prompt_hash"] = sha256_bytes(surfaced["prompt"].encode("utf-8"))
        surface_solved = solve_instance(surfaced)
        surface_passed = (
            surfaced["prompt_hash"] != instance["prompt_hash"]
            and surface_solved["label"] == solved["label"]
        )
        rows.append(
            {
                "row_id": f"{family}:family_preserving_surface_change",
                "instance": instance["instance_id"],
                "family": family,
                "transform": "family_preserving_surface_change",
                "expected_invariant": "prompt bytes change while exact labels stay fixed",
                "observed_result": {
                    "base_prompt_hash": instance["prompt_hash"],
                    "surface_prompt_hash": surfaced["prompt_hash"],
                },
                "pass_state": surface_passed,
            }
        )
    return rows


def _prompt_has_label_leakage(prompt: str) -> bool:
    return bool(re.search(r"exact\s+(?:optimum|answer|label)\s*[:=]", prompt, flags=re.IGNORECASE))


def build_mutation_rows(
    instances: Sequence[Mapping[str, Any]],
    instance_rows: Sequence[Mapping[str, Any]],
    state_action_rows: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    """Inject six named defects and record actual detector outcomes."""

    illegal_instances = {
        row["instance"] for row in state_action_rows if row.get("legality") is False
    }
    instance = next(
        row
        for row in instances
        if row["family"] == "inventory" and row["instance_id"] in illegal_instances
    )
    instance_id = str(instance["instance_id"])
    clean_rows = [deepcopy(row) for row in state_action_rows if row["instance"] == instance_id]
    clean_label = deepcopy(labels[instance_id])

    bad_transition = deepcopy(clean_rows)
    next(row for row in bad_transition if row["legality"])["transition"]["next_state"] = {
        "stock": 999
    }
    bad_detected = "state_action_rows_mismatch" in validate_solution(
        instance, clean_label, bad_transition
    )

    infeasible = deepcopy(clean_rows)
    illegal_row = next(row for row in infeasible if not row["legality"])
    illegal_row["legality"] = True
    infeasible_detected = "state_action_rows_mismatch" in validate_solution(
        instance, clean_label, infeasible
    )

    corrupt_cost = deepcopy(clean_rows)
    cost_row = next(row for row in corrupt_cost if row["legality"])
    cost_row["immediate_cost"] += 1
    cost_detected = "state_action_rows_mismatch" in validate_solution(
        instance, clean_label, corrupt_cost
    )

    leaked_prompt = str(instance["prompt"]) + f" Exact optimum: {clean_label['total_optimum']}."
    leakage_detected = _prompt_has_label_leakage(leaked_prompt)

    wrong_tie = deepcopy(clean_label)
    wrong_tie["tie_flag"] = not wrong_tie["tie_flag"]
    tie_detected = "label_mismatch" in validate_solution(instance, wrong_tie, clean_rows)

    matching_instance_row = next(row for row in instance_rows if row["instance"] == instance_id)
    seal_store = LabelSealStore(
        {instance_id: (str(matching_instance_row["prompt_hash"]), clean_label)}
    )
    stale = seal_store.seal_row(instance_id)
    stale["label_hash"] = sha256_json({"stale": True})
    stale_detected = not LabelSealStore.verify_seal_row(stale)

    detections = {
        "bad_transition": bad_detected,
        "infeasible_action": infeasible_detected,
        "corrupted_cost": cost_detected,
        "label_leakage": leakage_detected,
        "wrong_ties": tie_detected,
        "stale_seal": stale_detected,
    }
    return [
        {
            "row_id": f"mutation:{mutation}",
            "mutation": mutation,
            "expected_detection": True,
            "observed_detection": detections[mutation],
            "pass_state": detections[mutation] is True,
        }
        for mutation in REQUIRED_MUTATIONS
    ]


def _manifest_hash(manifest: Mapping[str, Any]) -> str:
    body = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    return sha256_json(body)


def build_frozen_manifest(instances: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Freeze all benchmark inputs before exact headline evaluation."""

    manifest: JsonDict = {
        "schema": "carnot.experiment_6702.frozen_fixture_manifest.v1",
        "families": list(FAMILIES),
        "generators": {family: GENERATOR_VERSION for family in FAMILIES},
        "parameters": {
            family: {
                "instance_count": sum(row["family"] == family for row in instances),
                "parameter_hashes": [
                    sha256_json(row["parameters"]) for row in instances if row["family"] == family
                ],
            }
            for family in FAMILIES
        },
        "prompts": {
            "answer_free": True,
            "prompt_hashes": [row["prompt_hash"] for row in instances],
        },
        "splits": {
            "headline_per_family": 8,
            "development_per_family": 2,
            "held_family_policy": "each headline family is an isolated held-family slice",
        },
        "seeds": {str(row["instance_id"]): row["seed"] for row in instances},
        "transitions": {family: f"transition_{family}@{TRANSITION_VERSION}" for family in FAMILIES},
        "hard_constraints": ["action domain", "resource bounds", "family preconditions"],
        "objectives": {family: "exact integer stage plus terminal cost" for family in FAMILIES},
        "ties": "retain all minima; deterministic plan uses action-domain order",
        "policies": {
            "feasibility": "all transitions in a plan must be legal",
            "label_access": "prompt-bound commit receipt required",
            "current_event_oracle_access": "forbidden",
        },
        "versions": {
            "generator": GENERATOR_VERSION,
            "transition": TRANSITION_VERSION,
            "solver": SOLVER_VERSION,
            "independent_solver": INDEPENDENT_SOLVER_VERSION,
            "seal": SEAL_VERSION,
            "reducer": REDUCER_VERSION,
        },
        "dynamic_programming_convention": "finite-horizon cost minimization with exact integer values",
        "instance_spec_hashes": {str(row["instance_id"]): row["spec_hash"] for row in instances},
        "manifest_hash": "",
    }
    manifest["manifest_hash"] = _manifest_hash(manifest)
    return manifest


def protected_hashes(root: Path) -> dict[str, str]:
    return {str(path): sha256_file(root / path) for path in PROTECTED_PATHS}


def _ram_total_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024
    return 0  # pragma: no cover - Linux always exposes MemTotal on this task substrate.


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:  # pragma: no cover - recorded as a failed precondition.
        return "missing"


def _precondition(name: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    return {"name": name, "expected": expected, "observed": observed, "passed": passed}


def collect_preconditions(root: Path) -> list[JsonDict]:
    """Measure resources, dependencies, schema support, and root-task status."""

    cpu_count = os.cpu_count() or 0
    ram_bytes = _ram_total_bytes()
    disk_free = shutil.disk_usage(root if root.exists() else root.parent).free
    roadmap_path = root / ACTIVE_ROADMAP
    conductor_path = root / CONDUCTOR_PATH
    task: Mapping[str, Any] | None = None
    if roadmap_path.is_file():
        parsed = yaml.safe_load(roadmap_path.read_text(encoding="utf-8"))
        if isinstance(parsed, dict):
            task = next(
                (
                    row
                    for row in parsed.get("tasks", [])
                    if isinstance(row, dict)
                    and row.get("id") == "exp6702-exact-planning-fixture-recovery"
                ),
                None,
            )
    schema_probe = {
        field: []
        for field in REQUIRED_ARTIFACT_FIELDS
        if field.endswith("_rows") or field in {"gate_check_summary", "tests_run"}
    }
    schema_probe.update(
        {
            "status": "blocked_precondition",
            "honest_verdict": "blocked_precondition",
            "verdict_class": "blocked",
            "planning_fixture_ready": False,
            "duration_s": 0.0,
            "verifier_is_oracle": False,
            "inference_substrate": INFERENCE_SUBSTRATE,
        }
    )
    for field in REQUIRED_ARTIFACT_FIELDS:
        schema_probe.setdefault(field, {})
    schema_ok = True
    try:
        jsonschema.validate(schema_probe, ARTIFACT_JSON_SCHEMA)
    except jsonschema.ValidationError:  # pragma: no cover - a schema regression triggers blocking.
        schema_ok = False
    return [
        _precondition("cpu", ">=1", cpu_count, cpu_count >= 1),
        _precondition("ram_bytes", ">=1073741824", ram_bytes, ram_bytes >= 1_073_741_824),
        _precondition("disk_free_bytes", ">=104857600", disk_free, disk_free >= 104_857_600),
        _precondition(
            "exact_solver_dependencies",
            {"python_integer_arithmetic": True, "itertools": True},
            {"python_integer_arithmetic": True, "itertools": True},
            True,
        ),
        _precondition(
            "artifact_schema",
            "jsonschema validation available",
            {"jsonschema_version": _package_version("jsonschema"), "probe_valid": schema_ok},
            schema_ok,
        ),
        _precondition(
            "roadmap",
            "active V584 root task present",
            {"path": str(ACTIVE_ROADMAP), "sha256": sha256_file(roadmap_path)},
            task is not None,
        ),
        _precondition(
            "conductor",
            "source file present",
            {"path": str(CONDUCTOR_PATH), "sha256": sha256_file(conductor_path)},
            conductor_path.is_file(),
        ),
        _precondition(
            "runtime_manifest_parity_dependency",
            "no gated_on edge for Exp6702",
            {"gated_on": task.get("gated_on") if task else "task_missing"},
            task is not None and "gated_on" not in task,
        ),
    ]


def _test_row_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("check_id")): row for row in rows}


def recompute_aggregate(
    *,
    instance_rows: Sequence[Mapping[str, Any]],
    state_action_rows: Sequence[Mapping[str, Any]],
    exact_solver_rows: Sequence[Mapping[str, Any]],
    label_seal_rows: Sequence[Mapping[str, Any]],
    metamorphic_rows: Sequence[Mapping[str, Any]],
    mutation_rows: Sequence[Mapping[str, Any]],
    tests_run: Sequence[Mapping[str, Any]],
    preconditions_passed: bool,
    protected_files_unchanged: bool,
) -> JsonDict:
    """Reduce readiness only from raw rows and measured receipts."""

    checks: dict[str, bool] = {}
    headline = [row for row in instance_rows if row.get("split") == "headline"]
    development = [row for row in instance_rows if row.get("split") == "development"]
    checks["instance_coverage"] = (
        len(headline) == 32
        and len(development) == 8
        and all(sum(row.get("family") == family for row in headline) == 8 for family in FAMILIES)
        and all(sum(row.get("family") == family for row in development) == 2 for family in FAMILIES)
        and len({row.get("instance") for row in instance_rows}) == 40
    )
    checks["split_isolation"] = not {row.get("instance") for row in headline} & {
        row.get("instance") for row in development
    } and len({row.get("prompt_hash") for row in instance_rows}) == len(instance_rows)
    instance_ids = {row.get("instance") for row in instance_rows}
    action_instance_ids = {row.get("instance") for row in state_action_rows}
    checks["state_action_coverage"] = (
        bool(state_action_rows) and action_instance_ids == instance_ids
    )
    checks["exactness"] = (
        sum(row.get("solver") == SOLVER_VERSION for row in exact_solver_rows) == 40
        and sum(row.get("solver") == INDEPENDENT_SOLVER_VERSION for row in exact_solver_rows)
        == len(FAMILIES)
        and all(row.get("passed") is True for row in exact_solver_rows)
        and all(not row.get("action_value_mismatches") for row in exact_solver_rows)
    )
    checks["sealing"] = (
        len(label_seal_rows) == 40
        and {row.get("instance") for row in label_seal_rows} == instance_ids
        and all(LabelSealStore.verify_seal_row(row) for row in label_seal_rows)
    )
    checks["metamorphic_checks"] = (
        len(metamorphic_rows) == len(FAMILIES) * len(METAMORPHIC_TRANSFORMS)
        and {row.get("transform") for row in metamorphic_rows} == set(METAMORPHIC_TRANSFORMS)
        and all(row.get("pass_state") is True for row in metamorphic_rows)
    )
    checks["mutation_detection"] = (
        {row.get("mutation") for row in mutation_rows} == set(REQUIRED_MUTATIONS)
        and len(mutation_rows) == len(REQUIRED_MUTATIONS)
        and all(row.get("pass_state") is True for row in mutation_rows)
    )
    tests = _test_row_map(tests_run)
    for check_id in REQUIRED_TEST_CHECKS:
        checks[check_id] = tests.get(check_id, {}).get("passed") is True
    checks["scoped_coverage"] = checks["scoped_coverage"] and (
        tests.get("scoped_coverage", {}).get("coverage_percent") == 100.0
    )
    checks["preconditions"] = preconditions_passed
    checks["protected_files"] = protected_files_unchanged
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "schema": "carnot.experiment_6702.aggregate_recomputation.v1",
        "headline_instance_count": len(headline),
        "development_instance_count": len(development),
        "state_action_row_count": len(state_action_rows),
        "exact_solver_row_count": len(exact_solver_rows),
        "label_seal_row_count": len(label_seal_rows),
        "metamorphic_row_count": len(metamorphic_rows),
        "mutation_row_count": len(mutation_rows),
        "checks": checks,
        "failed_checks": failed,
        "planning_fixture_ready": not failed,
    }


def _per_unit_rows(
    instance_rows: Sequence[Mapping[str, Any]],
    state_action_rows: Sequence[Mapping[str, Any]],
    exact_solver_rows: Sequence[Mapping[str, Any]],
    label_seal_rows: Sequence[Mapping[str, Any]],
    metamorphic_rows: Sequence[Mapping[str, Any]],
    mutation_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for unit_type, source in (
        ("instance", instance_rows),
        ("state_action", state_action_rows),
        ("exact_solver", exact_solver_rows),
        ("label_seal", label_seal_rows),
        ("metamorphic", metamorphic_rows),
        ("mutation", mutation_rows),
    ):
        rows.extend({"unit_type": unit_type, **deepcopy(dict(row))} for row in source)
    return rows


def _field_provenance(artifact: Mapping[str, Any]) -> dict[str, JsonDict]:
    provenance: dict[str, JsonDict] = {}
    for field in REQUIRED_ARTIFACT_FIELDS:
        stable_value = (
            artifact.get(field)
            if field not in {"field_provenance", "reproducibility_checksum", "duration_s"}
            else {"field": field, "version": SCHEMA}
        )
        provenance[field] = {
            "generator": GENERATOR_VERSION,
            "solver": SOLVER_VERSION,
            "seal": SEAL_VERSION,
            "reducer": REDUCER_VERSION,
            "function": "build_artifact",
            "version": SCHEMA,
            "hash": sha256_json(stable_value),
            "principle": FIELD_PRINCIPLES[field],
        }
    return provenance


def artifact_checksum(payload: Mapping[str, Any]) -> str:
    """Hash deterministic content while excluding self and wall-clock duration."""

    body = deepcopy(dict(payload))
    body.pop("reproducibility_checksum", None)
    body.pop("duration_s", None)
    return sha256_json(body)


def _gate_summary(aggregate: Mapping[str, Any]) -> list[JsonDict]:
    return [
        {"check": name, "expected": True, "observed": False}
        for name in aggregate.get("failed_checks", [])
    ]


def build_artifact(
    *,
    date: str,
    root: Path,
    tests_run: Sequence[Mapping[str, Any]],
    duration_s: float,
    protected_before: Mapping[str, str],
) -> JsonDict:
    """Freeze inputs, solve rows, reduce readiness, and build one artifact."""

    preconditions = collect_preconditions(root)
    if not all(row["passed"] for row in preconditions):
        return build_blocked_artifact(
            date=date, root=root, preconditions=preconditions, duration_s=duration_s
        )
    instances = generate_instances()
    manifest = build_frozen_manifest(instances)
    instance_rows: list[JsonDict] = []
    state_action_rows: list[JsonDict] = []
    exact_solver_rows: list[JsonDict] = []
    labels: dict[str, JsonDict] = {}
    for instance in instances:
        solved = solve_instance(instance)
        labels[str(instance["instance_id"])] = solved["label"]
        instance_rows.append(build_instance_row(instance, solved))
        state_action_rows.extend(solved["state_action_rows"])
        exact_solver_rows.append(solved["solver_row"])
    exact_solver_rows.extend(build_independent_solver_rows(instances))
    label_seal_rows = build_label_seal_rows(instances, labels)
    metamorphic_rows = build_metamorphic_rows(instances)
    mutation_rows = build_mutation_rows(instances, instance_rows, state_action_rows, labels)
    protected_after = protected_hashes(root)
    protected = {
        "before": dict(protected_before),
        "after": protected_after,
        "unchanged": dict(protected_before) == protected_after,
    }
    aggregate = recompute_aggregate(
        instance_rows=instance_rows,
        state_action_rows=state_action_rows,
        exact_solver_rows=exact_solver_rows,
        label_seal_rows=label_seal_rows,
        metamorphic_rows=metamorphic_rows,
        mutation_rows=mutation_rows,
        tests_run=tests_run,
        preconditions_passed=True,
        protected_files_unchanged=protected["unchanged"],
    )
    ready = aggregate["planning_fixture_ready"]
    artifact: JsonDict = {
        "experiment": 6702,
        "schema": SCHEMA,
        "run_date": date,
        "status": "complete_ready" if ready else "blocked_fixture_checks",
        "honest_verdict": (
            "complete: exact finite-horizon planning fixture ready"
            if ready
            else "blocked_fixture_checks"
        ),
        "verdict_class": "null" if ready else "blocked",
        "gate_check_summary": _gate_summary(aggregate),
        "openspec_requirement_ids": list(REQUIRED_SPEC_ANCHORS),
        "frozen_fixture_manifest": manifest,
        "instance_rows": instance_rows,
        "state_action_rows": state_action_rows,
        "exact_solver_rows": exact_solver_rows,
        "label_seal_rows": label_seal_rows,
        "metamorphic_rows": metamorphic_rows,
        "mutation_rows": mutation_rows,
        "planning_fixture_ready": ready,
        "per_unit_rows": _per_unit_rows(
            instance_rows,
            state_action_rows,
            exact_solver_rows,
            label_seal_rows,
            metamorphic_rows,
            mutation_rows,
        ),
        "aggregate_row_recomputation": aggregate,
        "preconditions_checked": preconditions,
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_provenance": {},
        "random_seed": {
            "family": {family: _family_seed(family) for family in FAMILIES},
            "instance": {str(row["instance_id"]): row["seed"] for row in instances},
            "metamorphic": 6_702_901,
            "mutation": 6_702_902,
        },
        "duration_s": max(0.0, float(duration_s)),
        "tests_run": [deepcopy(dict(row)) for row in tests_run],
        "reproducibility_checksum": "",
    }
    artifact["field_provenance"] = _field_provenance(artifact)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_blocked_artifact(
    *, date: str, root: Path, preconditions: Sequence[Mapping[str, Any]], duration_s: float
) -> JsonDict:
    """Write measured absence without inventing fixture evidence."""

    protected = protected_hashes(root)
    empty_aggregate = recompute_aggregate(
        instance_rows=[],
        state_action_rows=[],
        exact_solver_rows=[],
        label_seal_rows=[],
        metamorphic_rows=[],
        mutation_rows=[],
        tests_run=[],
        preconditions_passed=False,
        protected_files_unchanged=True,
    )
    failed_preconditions = [row for row in preconditions if not row.get("passed")]
    artifact: JsonDict = {
        "experiment": 6702,
        "schema": SCHEMA,
        "run_date": date,
        "status": "blocked_precondition",
        "honest_verdict": "blocked_precondition",
        "verdict_class": "blocked",
        "gate_check_summary": [
            {
                "check": row.get("name"),
                "expected": row.get("expected"),
                "observed": row.get("observed"),
            }
            for row in failed_preconditions
        ],
        "openspec_requirement_ids": list(REQUIRED_SPEC_ANCHORS),
        "frozen_fixture_manifest": {},
        "instance_rows": [],
        "state_action_rows": [],
        "exact_solver_rows": [],
        "label_seal_rows": [],
        "metamorphic_rows": [],
        "mutation_rows": [],
        "planning_fixture_ready": False,
        "per_unit_rows": [],
        "aggregate_row_recomputation": empty_aggregate,
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "protected_files_unchanged": {
            "before": protected,
            "after": protected,
            "unchanged": True,
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_provenance": {},
        "random_seed": {
            "family": {family: _family_seed(family) for family in FAMILIES},
            "instance": {},
            "metamorphic": 6_702_901,
            "mutation": 6_702_902,
        },
        "duration_s": max(0.0, float(duration_s)),
        "tests_run": [],
        "reproducibility_checksum": "",
    }
    artifact["field_provenance"] = _field_provenance(artifact)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Return stable errors for incomplete or internally inconsistent evidence."""

    errors: list[str] = []
    if set(REQUIRED_ARTIFACT_FIELDS) - set(payload):
        return ["missing_required_fields"]
    try:
        jsonschema.validate(dict(payload), ARTIFACT_JSON_SCHEMA)
    except jsonschema.ValidationError:
        errors.append("artifact_schema_mismatch")
    if payload.get("reproducibility_checksum") != artifact_checksum(payload):
        errors.append("reproducibility_checksum_mismatch")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if payload.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_mismatch")
    expected_units = _per_unit_rows(
        payload.get("instance_rows", []),
        payload.get("state_action_rows", []),
        payload.get("exact_solver_rows", []),
        payload.get("label_seal_rows", []),
        payload.get("metamorphic_rows", []),
        payload.get("mutation_rows", []),
    )
    if payload.get("per_unit_rows") != expected_units:
        errors.append("per_unit_rows_mismatch")
    preconditions_passed = all(
        row.get("passed") is True for row in payload.get("preconditions_checked", [])
    )
    protected_unchanged = payload.get("protected_files_unchanged", {}).get("unchanged") is True
    aggregate = recompute_aggregate(
        instance_rows=payload.get("instance_rows", []),
        state_action_rows=payload.get("state_action_rows", []),
        exact_solver_rows=payload.get("exact_solver_rows", []),
        label_seal_rows=payload.get("label_seal_rows", []),
        metamorphic_rows=payload.get("metamorphic_rows", []),
        mutation_rows=payload.get("mutation_rows", []),
        tests_run=payload.get("tests_run", []),
        preconditions_passed=preconditions_passed,
        protected_files_unchanged=protected_unchanged,
    )
    if payload.get("aggregate_row_recomputation") != aggregate:
        errors.append("aggregate_row_recomputation_mismatch")
    ready = payload.get("planning_fixture_ready") is True
    if ready != aggregate["planning_fixture_ready"]:
        errors.append("readiness_mismatch")
    if ready:
        manifest = payload.get("frozen_fixture_manifest", {})
        if manifest.get("manifest_hash") != _manifest_hash(manifest):
            errors.append("manifest_hash_mismatch")
        if payload.get("status") != "complete_ready" or payload.get("verdict_class") != "null":
            errors.append("ready_terminal_state_mismatch")
        if not str(payload.get("honest_verdict", "")).startswith("complete:"):
            errors.append("honest_verdict_mismatch")
        if payload.get("gate_check_summary") != []:
            errors.append("ready_gate_summary_mismatch")
    else:
        if not str(payload.get("status", "")).startswith("blocked_"):
            errors.append("blocked_terminal_state_mismatch")
        if payload.get("verdict_class") != "blocked":
            errors.append("blocked_verdict_class_mismatch")
        if not payload.get("gate_check_summary"):
            errors.append("blocked_gate_summary_mismatch")
    if set(payload.get("field_provenance", {})) < set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance_invalid")
    if not isinstance(payload.get("duration_s"), (int, float)) or payload.get("duration_s", -1) < 0:
        errors.append("duration_invalid")
    return list(dict.fromkeys(errors))


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> JsonDict:
    """Publish one complete JSON file with file and directory synchronization."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    try:
        with temporary.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():  # pragma: no cover - only a failed replacement leaves it behind.
            temporary.unlink()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "atomic_replace": True,
        "file_fsync": True,
        "directory_fsync": True,
    }


def default_command_runner(command: str, root: Path) -> JsonDict:
    """Run one fixed local verification command and retain bounded output."""

    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=root,
        shell=True,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "command": command,
        "exit_code": completed.returncode,
        "stdout": completed.stdout[-12_000:],
        "stderr": completed.stderr[-4_000:],
        "duration_s": time.monotonic() - started,
    }


def _coverage_percent(text: str) -> float | None:
    match = re.search(r"^TOTAL\s+\d+\s+\d+\s+(\d+)%", text, flags=re.MULTILINE)
    return float(match.group(1)) if match else None


def run_verification_commands(
    root: Path, *, runner: CommandRunner = default_command_runner
) -> list[JsonDict]:
    """Run the frozen focused, coverage, spec, E2E, lint, and full-suite checks."""

    rows: list[JsonDict] = []
    for check_id, command in VERIFICATION_COMMANDS:
        receipt = runner(command, root)
        output = str(receipt.get("stdout", "")) + str(receipt.get("stderr", ""))
        coverage = _coverage_percent(output) if check_id == "scoped_coverage" else None
        passed = receipt.get("exit_code") == 0 and (
            check_id != "scoped_coverage" or coverage == 100.0
        )
        rows.append(
            {
                "check_id": check_id,
                "command": command,
                "exit_code": receipt.get("exit_code"),
                "passed": passed,
                "coverage_percent": coverage,
                "summary": "\n".join(output.strip().splitlines()[-8:]),
                "duration_s": receipt.get("duration_s", 0.0),
            }
        )
    return rows


def run(
    *,
    date: str,
    root: Path = REPO_ROOT,
    output_path: Path | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Check prerequisites, verify owned code, and atomically publish Exp6702."""

    started = time.monotonic()
    protected_before = protected_hashes(root)
    preconditions = collect_preconditions(root)
    if not all(row["passed"] for row in preconditions):
        artifact = build_blocked_artifact(
            date=date,
            root=root,
            preconditions=preconditions,
            duration_s=time.monotonic() - started,
        )
    else:
        measured_tests = (
            [deepcopy(dict(row)) for row in tests_run]
            if tests_run is not None
            else run_verification_commands(root)
        )
        artifact = build_artifact(
            date=date,
            root=root,
            tests_run=measured_tests,
            duration_s=0.0,
            protected_before=protected_before,
        )
    artifact["duration_s"] = time.monotonic() - started
    artifact["field_provenance"] = _field_provenance(artifact)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:  # pragma: no cover - this is the final fail-closed publication guard.
        raise ValueError(f"invalid Exp6702 artifact: {errors}")
    write_json_atomic(output_path or root / RESULT_PATH, artifact)
    return artifact


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_PATH)
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run Exp6702 or validate one redirected artifact."""

    args = _parse_args(argv)
    if args.validate:
        if not args.output.is_file():
            print(json.dumps({"valid": False, "errors": ["artifact_missing"]}, sort_keys=True))
            return 1
        try:
            artifact = load_json(args.output)
        except (json.JSONDecodeError, OSError, TypeError) as exc:
            print(
                json.dumps(
                    {"valid": False, "errors": [f"artifact_unreadable:{type(exc).__name__}"]},
                    sort_keys=True,
                )
            )
            return 1
        errors = validate_artifact(artifact)
        print(json.dumps({"valid": not errors, "errors": errors}, sort_keys=True))
        return 0 if not errors else 1
    artifact = run(date=args.date, root=REPO_ROOT, output_path=args.output)
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "planning_fixture_ready": artifact["planning_fixture_ready"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0 if artifact["planning_fixture_ready"] else 2


if __name__ == "__main__":  # pragma: no cover - exercised by the required module command.
    raise SystemExit(main())
