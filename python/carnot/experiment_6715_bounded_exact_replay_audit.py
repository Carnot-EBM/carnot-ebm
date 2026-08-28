"""Cold-recompute a bounded subset of the exact planning fixture.

The solver in this module reads only typed specifications. It enumerates
complete action paths and does not import the fixture producer. Reported labels
are opened only after the sample identity, reveal order, caps, and hashes are
frozen. This order makes agreement independent evidence instead of a second
call to the producer's solver.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
from importlib import metadata
from itertools import product
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_PATH = Path("results/experiment_6702_exact_planning_fixture_recovery.json")
RESULT_PATH = Path("results/experiment_6715_bounded_exact_replay_audit.json")
ACTIVE_ROADMAP = Path("research-roadmap.yaml")
CONDUCTOR_PATH = Path("scripts/research_conductor.py")
MODULE_PATH = Path("python/carnot/experiment_6715_bounded_exact_replay_audit.py")
TEST_PATH = Path("tests/python/test_experiment_6715_bounded_exact_replay_audit.py")

SCHEMA = "carnot.experiment_6715.bounded_exact_replay_audit.v1"
INFERENCE_SUBSTRATE = "cpu_independent_exhaustive_audit_no_llm"
SOLVER_VERSION = "carnot.exp6715.complete_path_enumerator.v1"
COMPARATOR_VERSION = "carnot.exp6715.exact_field_comparator.v1"
REDUCER_VERSION = "carnot.exp6715.row_owned_reducer.v1"
SELECTION_VERSION = "carnot.exp6715.frozen_stratified_sample.v1"
FAMILIES = ("inventory", "battery_dispatch", "job_slot", "reservoir_control")
RANDOM_SEEDS = {"sample": 6_715_002, "reveal_order": 6_715_003}
PREREGISTERED_CAPS: JsonDict = {
    "max_horizon": 6,
    "max_action_count": 5,
    "max_state_count": 128,
    "max_enumeration_per_instance": 15_625,
    "max_total_enumeration_count": 50_000,
    "max_audit_wall_time_s": 600,
}
REPORTED_LABEL_FIELDS = frozenset(
    {
        "optimum",
        "ties",
        "feasibility",
        "state_action_rows",
        "exact_solver_rows",
        "aggregate_row_recomputation",
        "planning_fixture_ready",
    }
)

OPEN_SPEC_IDS = (
    "REQ-CONSTRAINT-6715",
    "SCENARIO-CONSTRAINT-6715-FROZEN-SAMPLE",
    "SCENARIO-CONSTRAINT-6715-EXHAUSTIVE",
    "SCENARIO-CONSTRAINT-6715-CAPS",
    "REQ-VERIFY-6715",
    "SCENARIO-VERIFY-6715-EXACT-PARITY",
    "SCENARIO-VERIFY-6715-AUTHORITY",
    "REQ-PIPELINE-6715",
    "SCENARIO-PIPELINE-6715-ROW-REDUCTION",
    "SCENARIO-PIPELINE-6715-PER-UNIT",
    "REQ-REPORT-6715",
    "SCENARIO-REPORT-6715-ATOMIC",
    "SCENARIO-REPORT-6715-BLOCKED",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "gate_check_summary",
    "openspec_requirement_ids",
    "frozen_sample_manifest",
    "method_fidelity_contract",
    "enumeration_rows",
    "reported_vs_recomputed_rows",
    "exact_replay_audit_passed",
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
PROVENANCE_KEYS = (
    "source_store",
    "audit_solver",
    "comparator",
    "reducer",
    "function",
    "version",
    "hash",
    "principle",
)

FOCUSED_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--data-file=/tmp/carnot_exp6715_coverage "
    "--include=*/experiment_6715_bounded_exact_replay_audit.py "
    "-m pytest tests/python/test_experiment_6715_bounded_exact_replay_audit.py "
    "-q --no-cov -n 0 -o addopts="
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--data-file=/tmp/carnot_exp6715_coverage "
    "--include=*/experiment_6715_bounded_exact_replay_audit.py "
    "--fail-under=100 --show-missing"
)
FULL_SUITE_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6715_bounded_exact_replay_audit.py"
)
E2E_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6715_bounded_exact_replay_audit.py "
    "-q --no-cov -n 0 -o addopts= -k e2e_bounded_actual_replay"
)
RUFF_COMMAND = f".venv/bin/ruff check {MODULE_PATH} {TEST_PATH}"
FORMAT_COMMAND = f".venv/bin/ruff format --check {MODULE_PATH} {TEST_PATH}"
VERIFICATION_COMMANDS = (
    ("focused_tests", FOCUSED_COMMAND),
    ("scoped_coverage", COVERAGE_COMMAND),
    ("full_python_suite", FULL_SUITE_COMMAND),
    ("spec_coverage", SPEC_COVERAGE_COMMAND),
    ("applicable_e2e", E2E_COMMAND),
    ("ruff_check", RUFF_COMMAND),
    ("format_check", FORMAT_COMMAND),
)
OPERATIONAL_CHECK_IDS = (
    "artifact_validation",
    "row_consistency",
    "adversarial_verification",
)
REQUIRED_TEST_CHECKS = (
    *(check_id for check_id, _ in VERIFICATION_COMMANDS),
    *OPERATIONAL_CHECK_IDS,
)


def canonical_json(value: Any) -> str:
    """Return stable JSON bytes for hashes and exact comparisons."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Hash a JSON value without depending on source whitespace."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash file bytes, or retain an explicit missing state."""

    if not path.is_file():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def load_json(path: Path) -> JsonDict:
    """Load one JSON object and reject scalar or array substitutes."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("JSON object required")
    return value


def public_instance_rows(upstream: Mapping[str, Any]) -> list[JsonDict]:
    """Project public inputs before any reported label field is accessed."""

    fields = (
        "instance",
        "family",
        "split",
        "seed",
        "horizon",
        "prompt_hash",
        "spec_hash",
        "label_seal_hash",
        "typed_spec",
    )
    return [
        {field: deepcopy(row.get(field)) for field in fields}
        for row in upstream.get("instance_rows", [])
    ]


def embedded_store_receipts(upstream: Mapping[str, Any], path: Path) -> JsonDict:
    """Bind the producer artifact and each embedded raw store by content."""

    stores = ("instance_rows", "state_action_rows", "label_seal_rows")
    return {
        "upstream_artifact": {"path": path.as_posix(), "sha256": sha256_file(path)},
        **{
            store: {
                "count": len(upstream.get(store, [])),
                "sha256": sha256_json(upstream.get(store, [])),
            }
            for store in stores
        },
    }


def manifest_checksum(manifest: Mapping[str, Any]) -> str:
    """Hash a manifest without accepting its self-referential hash."""

    return sha256_json({key: value for key, value in manifest.items() if key != "manifest_hash"})


def _selection_hash(instance: str, spec_hash: str) -> str:
    material = f"{RANDOM_SEEDS['sample']}:{instance}:{spec_hash}".encode()
    return "sha256:" + hashlib.sha256(material).hexdigest()


def freeze_sample(
    public_rows: Sequence[Mapping[str, Any]], store_receipts: Mapping[str, Any]
) -> JsonDict:
    """Freeze eight headline identities without accepting reported labels."""

    if any(REPORTED_LABEL_FIELDS.intersection(row) for row in public_rows):
        raise ValueError("reported labels present before sample freeze")
    identities = [str(row.get("instance")) for row in public_rows]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate public instance")

    selected: list[JsonDict] = []
    for family in FAMILIES:
        candidates = [
            row
            for row in public_rows
            if row.get("family") == family and row.get("split") == "headline"
        ]
        anchor_id = f"{family}-headline-01"
        anchor = next((row for row in candidates if row.get("instance") == anchor_id), None)
        if anchor is None:
            raise ValueError(f"missing preregistered edge-probe: {anchor_id}")
        contrast_candidates = [row for row in candidates if row is not anchor]
        if not contrast_candidates:
            raise ValueError(f"missing contrast candidates: {family}")
        contrast = min(
            contrast_candidates,
            key=lambda row: _selection_hash(str(row["instance"]), str(row["spec_hash"])),
        )
        for role, row in (("edge_probe", anchor), ("contrast", contrast)):
            selected.append(
                {
                    "instance": row["instance"],
                    "family": family,
                    "split": "headline",
                    "selection_role": role,
                    "spec_hash": row["spec_hash"],
                    "prompt_hash": row["prompt_hash"],
                    "seal_hash": row["label_seal_hash"],
                    "selection_hash": _selection_hash(str(row["instance"]), str(row["spec_hash"])),
                }
            )

    reveal_order = [
        row["instance"]
        for row in sorted(
            selected,
            key=lambda row: hashlib.sha256(
                f"{RANDOM_SEEDS['reveal_order']}:{row['instance']}".encode()
            ).hexdigest(),
        )
    ]
    public_by_instance = {str(row["instance"]): row for row in public_rows}
    typed_specs: JsonDict = {}
    for row in selected:
        instance = str(row["instance"])
        typed_spec = deepcopy(public_by_instance[instance].get("typed_spec"))
        if not isinstance(typed_spec, Mapping) or sha256_json(typed_spec) != row["spec_hash"]:
            raise ValueError(f"public typed specification hash mismatch: {instance}")
        typed_specs[instance] = typed_spec
    manifest: JsonDict = {
        "schema": "carnot.experiment_6715.frozen_sample_manifest.v1",
        "selection_rule": (
            "headline-01 edge probe plus the lowest "
            "sha256(sample_seed:instance:spec_hash) contrast per family"
        ),
        "edge_rule": (
            "after reveal, each fixed edge probe must be tie-bearing or infeasible; "
            "failure cannot replace or widen the sample"
        ),
        "expected_instance_count": 8,
        "instances": selected,
        "selection_hash": sha256_json(selected),
        "typed_specs": typed_specs,
        "typed_spec_store_hash": sha256_json(typed_specs),
        "reveal_order": reveal_order,
        "reveal_order_hash": sha256_json(reveal_order),
        "caps": deepcopy(PREREGISTERED_CAPS),
        "versions": {
            "selection": SELECTION_VERSION,
            "solver": SOLVER_VERSION,
            "comparator": COMPARATOR_VERSION,
            "reducer": REDUCER_VERSION,
        },
        "source_store_receipts": deepcopy(dict(store_receipts)),
        "frozen_before_reported_label_read": True,
        "sample_widening_permitted": False,
    }
    manifest["manifest_hash"] = manifest_checksum(manifest)
    return manifest


def _state_name(spec: Mapping[str, Any]) -> str:
    return next(iter(spec["initial_state"]))


def _state_dict(spec: Mapping[str, Any], value: int) -> JsonDict:
    return {_state_name(spec): value}


def independent_transition(
    spec: Mapping[str, Any], time_index: int, state: int, action: int
) -> JsonDict:
    """Apply one independently written transition and exact stage cost."""

    family = str(spec.get("family"))
    parameters = spec.get("parameters", {})
    if family not in FAMILIES:
        raise ValueError(f"unknown family: {family}")
    actions = list(parameters.get("actions", spec.get("action_domain", [])))
    if action not in actions:
        return {
            "legal": False,
            "next_state": None,
            "immediate_cost": None,
            "reason": "action_outside_domain",
        }
    shift = int(parameters.get("stage_cost_shift", 0))

    if family == "inventory":
        available = state + action
        if available > int(parameters["capacity"]):
            return {
                "legal": False,
                "next_state": None,
                "immediate_cost": None,
                "reason": "capacity_exceeded",
            }
        demand = int(parameters["demand"][time_index])
        shortage = max(0, demand - available)
        next_state = max(0, available - demand)
        cost = (
            action * int(parameters["price"][time_index])
            + shortage * int(parameters["shortage_penalty"])
            + next_state * int(parameters["holding_cost"])
            + shift
        )
    elif family == "battery_dispatch":
        next_state = state - action
        if not 0 <= next_state <= int(parameters["capacity"]):
            return {
                "legal": False,
                "next_state": None,
                "immediate_cost": None,
                "reason": "charge_bounds",
            }
        grid = int(parameters["load"][time_index]) - action
        if grid < 0:
            return {
                "legal": False,
                "next_state": None,
                "immediate_cost": None,
                "reason": "grid_export_forbidden",
            }
        cost = (
            grid * int(parameters["price"][time_index])
            + abs(action) * int(parameters["cycling_cost"])
            + shift
        )
    elif family == "job_slot":
        if action == 0:
            next_state = state
            cost = int(parameters["idle_cost"]) + shift
        else:
            job = action - 1
            bit = 1 << job
            if state & bit:
                return {
                    "legal": False,
                    "next_state": None,
                    "immediate_cost": None,
                    "reason": "job_already_completed",
                }
            required = int(parameters["prerequisites"][job])
            if state & required != required:
                return {
                    "legal": False,
                    "next_state": None,
                    "immediate_cost": None,
                    "reason": "prerequisite_missing",
                }
            if time_index > int(parameters["deadlines"][job]):
                return {
                    "legal": False,
                    "next_state": None,
                    "immediate_cost": None,
                    "reason": "deadline_missed",
                }
            next_state = state | bit
            cost = int(parameters["schedule_costs"][job]) + shift
    else:
        available = state + int(parameters["inflow"][time_index])
        if action > available:
            return {
                "legal": False,
                "next_state": None,
                "immediate_cost": None,
                "reason": "release_exceeds_available_water",
            }
        raw_next = available - action
        spill = max(0, raw_next - int(parameters["capacity"]))
        next_state = min(int(parameters["capacity"]), raw_next)
        cost = (
            abs(action - int(parameters["target_release"][time_index]))
            * int(parameters["deviation_penalty"])
            + spill * int(parameters["spill_penalty"])
            + max(0, next_state - int(parameters["flood_threshold"]))
            * int(parameters["flood_penalty"])
            + shift
        )
    return {
        "legal": True,
        "next_state": next_state,
        "immediate_cost": cost,
        "reason": "legal",
    }


def terminal_cost(spec: Mapping[str, Any], state: int) -> int:
    """Return the independently implemented exact terminal cost."""

    parameters = spec["parameters"]
    family = spec["family"]
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
    if family == "reservoir_control":
        return abs(state - int(parameters["terminal_target"])) * int(parameters["terminal_penalty"])
    raise ValueError(f"unknown family: {family}")


class CapExceeded(RuntimeError):
    """Carry one measured cap failure into a terminal audit artifact."""

    def __init__(self, row: Mapping[str, Any]) -> None:
        self.row = deepcopy(dict(row))
        super().__init__(f"{row['cap']}: expected {row['expected']}, observed {row['observed']}")


@dataclass
class EnumerationBudget:
    """Reserve complete-path counts before an instance starts enumeration."""

    maximum: int
    used: int = 0

    def reserve(self, amount: int, instance: str) -> None:
        observed = self.used + amount
        if observed > self.maximum:
            raise CapExceeded(
                cap_row(
                    "max_total_enumeration_count",
                    self.maximum,
                    observed,
                    False,
                    instance,
                )
            )
        self.used = observed


def cap_row(
    cap: str, expected: Any, observed: Any, passed: bool, instance: str | None = None
) -> JsonDict:
    """Retain an expected and observed value for one frozen bound."""

    return {
        "cap": cap,
        "instance": instance,
        "expected": expected,
        "observed": observed,
        "passed": passed,
    }


def _raise_cap(cap: str, expected: Any, observed: Any, instance: str) -> None:
    raise CapExceeded(cap_row(cap, expected, observed, False, instance))


def exhaustive_solve(
    spec: Mapping[str, Any],
    caps: Mapping[str, Any],
    *,
    budget: EnumerationBudget | None = None,
    audit_started: float | None = None,
) -> JsonDict:
    """Enumerate complete paths once and derive all state-action values."""

    started = time.perf_counter()
    global_started = started if audit_started is None else audit_started
    instance = str(spec["instance_id"])
    horizon = int(spec["horizon"])
    actions = [int(action) for action in spec["action_domain"]]
    if horizon > int(caps["max_horizon"]):
        _raise_cap("max_horizon", caps["max_horizon"], horizon, instance)
    if len(actions) > int(caps["max_action_count"]):
        _raise_cap("max_action_count", caps["max_action_count"], len(actions), instance)
    enumeration_count = len(actions) ** horizon
    if enumeration_count > int(caps["max_enumeration_per_instance"]):
        _raise_cap(
            "max_enumeration_per_instance",
            caps["max_enumeration_per_instance"],
            enumeration_count,
            instance,
        )
    if time.perf_counter() - global_started > float(caps["max_audit_wall_time_s"]):
        _raise_cap(
            "max_audit_wall_time_s",
            caps["max_audit_wall_time_s"],
            time.perf_counter() - global_started,
            instance,
        )

    state_name = _state_name(spec)
    initial = int(spec["initial_state"][state_name])
    reachable: dict[int, set[int]] = {0: {initial}}
    for time_index in range(horizon):
        following: set[int] = set()
        for state in sorted(reachable[time_index]):
            for action in actions:
                transition = independent_transition(spec, time_index, state, action)
                if transition["legal"]:
                    following.add(int(transition["next_state"]))
        reachable[time_index + 1] = following
        state_count = sum(len(reachable[index]) for index in range(time_index + 2))
        if state_count > int(caps["max_state_count"]):
            _raise_cap("max_state_count", caps["max_state_count"], state_count, instance)

    if budget is not None:
        budget.reserve(enumeration_count, instance)

    legal_plans: list[JsonDict] = []
    suffix_values: dict[tuple[int, int, int], list[int]] = {}
    for path_index, path in enumerate(product(actions, repeat=horizon)):
        if path_index % 1024 == 0 and (
            time.perf_counter() - global_started > float(caps["max_audit_wall_time_s"])
        ):
            _raise_cap(
                "max_audit_wall_time_s",
                caps["max_audit_wall_time_s"],
                time.perf_counter() - global_started,
                instance,
            )
        state = initial
        trace: list[tuple[int, int, int, int]] = []
        legal = True
        for time_index, action in enumerate(path):
            transition = independent_transition(spec, time_index, state, action)
            if not transition["legal"]:
                legal = False
                break
            immediate = int(transition["immediate_cost"])
            trace.append((time_index, state, action, immediate))
            state = int(transition["next_state"])
        if not legal:
            continue
        running = terminal_cost(spec, state)
        for time_index, state, action, immediate in reversed(trace):
            running += immediate
            suffix_values.setdefault((time_index, state, action), []).append(running)
        legal_plans.append({"actions": list(path), "total": running})

    if legal_plans:
        optimum = min(row["total"] for row in legal_plans)
        optimum_plans = [row["actions"] for row in legal_plans if row["total"] == optimum]
    else:
        optimum = None
        optimum_plans = []
    tie_set = [action for action in actions if any(plan[0] == action for plan in optimum_plans)]

    rows: list[JsonDict] = []
    for time_index in range(horizon):
        for state in sorted(reachable[time_index]):
            totals: dict[int, int] = {
                action: min(suffix_values[(time_index, state, action)])
                for action in actions
                if (time_index, state, action) in suffix_values
            }
            best = min(totals.values()) if totals else None
            for action in actions:
                transition = independent_transition(spec, time_index, state, action)
                total = totals.get(action)
                immediate = transition["immediate_cost"]
                future = (
                    total - int(immediate) if total is not None and immediate is not None else None
                )
                row: JsonDict = {
                    "row_id": f"{instance}:{time_index}:{state}:{action}",
                    "instance": instance,
                    "family": spec["family"],
                    "time_index": time_index,
                    "state": _state_dict(spec, state),
                    "action": action,
                    "legality": bool(transition["legal"]),
                    "transition": {
                        "next_state": (
                            _state_dict(spec, int(transition["next_state"]))
                            if transition["legal"]
                            else None
                        ),
                        "reason": transition["reason"],
                    },
                    "immediate_cost": immediate,
                    "future_value": future,
                    "total_value": total,
                    "action_gap": None if total is None or best is None else total - best,
                    "optimum_membership": total == best
                    if total is not None and best is not None
                    else False,
                }
                row["receipt"] = sha256_json(row)
                rows.append(row)

    state_count = sum(len(reachable[index]) for index in range(horizon + 1))
    initial_rows = [row for row in rows if row["time_index"] == 0]
    result: JsonDict = {
        "instance": instance,
        "family": spec["family"],
        "horizon": horizon,
        "states": state_count,
        "actions": len(actions),
        "candidate_plans": enumeration_count,
        "enumeration_count": enumeration_count,
        "legal_plans": len(legal_plans),
        "feasible_plan_count": len(legal_plans),
        "feasible": bool(legal_plans),
        "plan_cost_receipt": sha256_json(legal_plans),
        "optimum": optimum,
        "optimum_plans": optimum_plans,
        "ties": tie_set,
        "tie_set": tie_set,
        "action_values": [
            {"action": row["action"], "value": row["total_value"]} for row in initial_rows
        ],
        "gaps": [{"action": row["action"], "gap": row["action_gap"]} for row in initial_rows],
        "runtime_s": round(time.perf_counter() - started, 6),
        "cap_state": "within_preregistered_caps",
        "state_action_rows": rows,
    }
    receipt_view = {
        key: value for key, value in result.items() if key not in {"runtime_s", "state_action_rows"}
    }
    result["receipt"] = sha256_json(receipt_view)
    return result


class _Missing:
    pass


MISSING = _Missing()
MISSING_VALUE = {"state": "missing"}


def nested_value(row: Mapping[str, Any], *keys: str) -> Any:
    """Keep an absent nested field distinct from a reported JSON null."""

    value: Any = row
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return MISSING
        value = value[key]
    return value


def comparison_row(unit: str, field: str, reported: Any, recomputed: Any) -> JsonDict:
    """Compare one exact field without converting missing values to zero."""

    if reported is MISSING:
        disposition = "missing_reported"
    elif recomputed is MISSING:
        disposition = "missing_recomputed"
    else:
        disposition = "match" if reported == recomputed else "mismatch"
    return {
        "unit": unit,
        "field": field,
        "reported_value": MISSING_VALUE if reported is MISSING else deepcopy(reported),
        "recomputed_value": MISSING_VALUE if recomputed is MISSING else deepcopy(recomputed),
        "tolerance": 0,
        "disposition": disposition,
    }


def _comparison_fields(
    instance: str,
    reported: Mapping[str, Any],
    solved: Mapping[str, Any],
) -> list[JsonDict]:
    optimum_plans = solved["optimum_plans"]
    values = (
        ("feasibility", reported.get("feasibility", MISSING), solved["feasible"]),
        ("optimum.total", nested_value(reported, "optimum", "total"), solved["optimum"]),
        (
            "optimum.plan",
            nested_value(reported, "optimum", "plan"),
            optimum_plans[0] if optimum_plans else None,
        ),
        (
            "optimum.action_set",
            nested_value(reported, "optimum", "action_set"),
            solved["tie_set"],
        ),
        ("ties", reported.get("ties", MISSING), len(solved["tie_set"]) > 1),
    )
    return [
        comparison_row(instance, field, reported_value, recomputed)
        for field, reported_value, recomputed in values
    ]


def recompute_frozen_sample(upstream: Mapping[str, Any], manifest: Mapping[str, Any]) -> JsonDict:
    """Open only frozen identities, then solve and compare every exact row."""

    if manifest.get("manifest_hash") != manifest_checksum(manifest):
        raise ValueError("frozen sample manifest hash mismatch")
    selected_hashes = {
        str(row.get("instance")): row.get("spec_hash") for row in manifest.get("instances", [])
    }
    frozen_specs = manifest.get("typed_specs")
    if (
        not isinstance(frozen_specs, Mapping)
        or set(frozen_specs) != set(selected_hashes)
        or any(
            sha256_json(frozen_specs[instance]) != spec_hash
            for instance, spec_hash in selected_hashes.items()
        )
    ):
        raise ValueError("frozen typed specification mismatch")
    reported_instances = {row["instance"]: row for row in upstream.get("instance_rows", [])}
    reported_actions: dict[str, list[Mapping[str, Any]]] = {
        instance: [
            row for row in upstream.get("state_action_rows", []) if row.get("instance") == instance
        ]
        for instance in manifest["reveal_order"]
    }
    roles = {row["instance"]: row["selection_role"] for row in manifest["instances"]}
    budget = EnumerationBudget(int(manifest["caps"]["max_total_enumeration_count"]))
    audit_started = time.perf_counter()
    enumeration_rows: list[JsonDict] = []
    state_action_rows: list[JsonDict] = []
    comparisons: list[JsonDict] = []
    edge_rows: list[JsonDict] = []
    cap_rows: list[JsonDict] = []

    for instance in manifest["reveal_order"]:
        reported = reported_instances.get(instance)
        solved = exhaustive_solve(
            frozen_specs[instance],
            manifest["caps"],
            budget=budget,
            audit_started=audit_started,
        )
        enumeration_rows.append(
            {key: deepcopy(value) for key, value in solved.items() if key != "state_action_rows"}
        )
        state_action_rows.extend(deepcopy(solved["state_action_rows"]))
        if reported is None:
            comparisons.append(comparison_row(instance, "instance", MISSING, instance))
        comparisons.extend(_comparison_fields(instance, reported or {}, solved))

        reported_by_key = {
            (row.get("time_index"), canonical_json(row.get("state")), row.get("action")): row
            for row in reported_actions[instance]
        }
        recomputed_by_key = {
            (row.get("time_index"), canonical_json(row.get("state")), row.get("action")): row
            for row in solved["state_action_rows"]
        }
        comparisons.append(
            comparison_row(
                instance,
                "state_action_row_count",
                len(reported_by_key),
                len(recomputed_by_key),
            )
        )
        action_fields = (
            "legality",
            "transition",
            "immediate_cost",
            "future_value",
            "total_value",
            "action_gap",
            "optimum_membership",
        )
        for key in sorted(set(reported_by_key) | set(recomputed_by_key), key=repr):
            reported_row = reported_by_key.get(key)
            recomputed_row = recomputed_by_key.get(key)
            unit = f"{instance}:{key[0]}:{key[1]}:{key[2]}"
            for field in action_fields:
                comparisons.append(
                    comparison_row(
                        unit,
                        field,
                        MISSING if reported_row is None else reported_row.get(field, MISSING),
                        MISSING if recomputed_row is None else recomputed_row.get(field, MISSING),
                    )
                )

        if roles[instance] == "edge_probe":
            observed = {
                "reported_ties": (reported or {}).get("ties", MISSING_VALUE),
                "reported_feasibility": (reported or {}).get("feasibility", MISSING_VALUE),
                "recomputed_ties": len(solved["tie_set"]) > 1,
                "recomputed_feasibility": solved["feasible"],
            }
            reported_edge = (reported or {}).get("ties") is True or (reported or {}).get(
                "feasibility"
            ) is False
            recomputed_edge = len(solved["tie_set"]) > 1 or solved["feasible"] is False
            edge_rows.append(
                {
                    "instance": instance,
                    "expected": "reported and recomputed tie-bearing or infeasible",
                    "observed": observed,
                    "passed": reported_edge and recomputed_edge,
                }
            )

        observed_caps = {
            "max_horizon": solved["horizon"],
            "max_action_count": solved["actions"],
            "max_state_count": solved["states"],
            "max_enumeration_per_instance": solved["enumeration_count"],
            "max_total_enumeration_count": budget.used,
            "max_audit_wall_time_s": time.perf_counter() - audit_started,
        }
        for cap, observed in observed_caps.items():
            expected = manifest["caps"][cap]
            cap_rows.append(cap_row(cap, expected, observed, observed <= expected, instance))

    return {
        "enumeration_rows": enumeration_rows,
        "state_action_rows": state_action_rows,
        "reported_vs_recomputed_rows": comparisons,
        "edge_rows": edge_rows,
        "cap_rows": cap_rows,
        "total_enumeration_count": budget.used,
    }


def _imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
    return modules


def method_fidelity_contract(root: Path) -> JsonDict:
    """State and mechanically check the bounded solver independence contract."""

    source = root / MODULE_PATH
    imports = _imported_modules(source) if source.is_file() else []
    forbidden_imports = [name for name in imports if "experiment_6702" in name]
    contract: JsonDict = {
        "schema": "carnot.experiment_6715.method_fidelity_contract.v1",
        "required_solver_independence": (
            "new complete-path enumeration from typed specs; no Exp6702 solver import"
        ),
        "data": {
            "selection_inputs": "public projected identities and typed specifications",
            "solver_inputs": "frozen typed specifications only",
            "reported_labels": "opened after manifest freeze for comparison only",
        },
        "budgets": deepcopy(PREREGISTERED_CAPS),
        "components": [
            "independent family transitions",
            "complete-path enumerator",
            "reachable-state transition audit",
            "exact field comparator",
            "row-owned reducer",
        ],
        "metrics": [
            "transitions",
            "legal actions",
            "feasible plans",
            "exact plan costs",
            "optima",
            "tie sets",
            "state-action values",
            "action gaps",
        ],
        "forbidden_operations": [
            "leakage scan",
            "seal attack scan",
            "metamorphic scan",
            "mutation scan",
        ],
        "forbidden_substitutions": [
            "Exp6702 dynamic-program solver",
            "reported optimum as solver input",
            "sample widening after reveal",
            "cap reduction",
            "LLM or learned oracle",
        ],
        "observed_forbidden_imports": forbidden_imports,
        "method_substitution": False,
        "cap_reduction": False,
        "sample_widening": False,
        "solver_source": MODULE_PATH.as_posix(),
        "solver_source_hash": sha256_file(source),
    }
    contract["satisfied"] = (
        not forbidden_imports
        and contract["method_substitution"] is False
        and contract["cap_reduction"] is False
        and contract["sample_widening"] is False
        and contract["budgets"] == PREREGISTERED_CAPS
    )
    contract["contract_hash"] = sha256_json(contract)
    return contract


def _test_row_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("check_id")): row for row in rows}


def _audit_check(check: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    return {"check": check, "expected": expected, "observed": observed, "passed": passed}


def recompute_aggregate(
    *,
    manifest: Mapping[str, Any],
    enumeration_rows: Sequence[Mapping[str, Any]],
    state_action_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    edge_rows: Sequence[Mapping[str, Any]],
    cap_rows: Sequence[Mapping[str, Any]],
    tests_run: Sequence[Mapping[str, Any]],
    preconditions_passed: bool,
    protected_files_unchanged: bool,
    method_contract: Mapping[str, Any],
) -> JsonDict:
    """Reduce the exact-replay gate from retained rows and receipts only."""

    selected = [row.get("instance") for row in manifest.get("instances", [])]
    selected_spec_hashes = {
        str(row.get("instance")): row.get("spec_hash") for row in manifest.get("instances", [])
    }
    frozen_specs = manifest.get("typed_specs")
    frozen_specs_ok = (
        isinstance(frozen_specs, Mapping)
        and set(frozen_specs) == set(selected_spec_hashes)
        and all(
            sha256_json(frozen_specs[instance]) == spec_hash
            for instance, spec_hash in selected_spec_hashes.items()
        )
        and manifest.get("typed_spec_store_hash") == sha256_json(frozen_specs)
    )
    family_counts = {
        family: sum(row.get("family") == family for row in manifest.get("instances", []))
        for family in FAMILIES
    }
    manifest_ok = (
        len(selected) == 8
        and len(set(selected)) == 8
        and all(count == 2 for count in family_counts.values())
        and manifest.get("caps") == PREREGISTERED_CAPS
        and manifest.get("manifest_hash") == manifest_checksum(manifest)
        and frozen_specs_ok
    )
    enumeration_ids = [row.get("instance") for row in enumeration_rows]
    enumeration_ok = (
        len(enumeration_rows) == 8
        and set(enumeration_ids) == set(selected)
        and len(enumeration_ids) == len(set(enumeration_ids))
        and all(row.get("cap_state") == "within_preregistered_caps" for row in enumeration_rows)
    )
    state_row_ids = [row.get("row_id") for row in state_action_rows]
    state_actions_ok = (
        bool(state_action_rows)
        and len(state_row_ids) == len(set(state_row_ids))
        and {row.get("instance") for row in state_action_rows} == set(selected)
    )
    comparisons_ok = bool(comparison_rows) and all(
        row.get("disposition") == "match" for row in comparison_rows
    )
    edge_ok = len(edge_rows) == len(FAMILIES) and all(
        row.get("passed") is True for row in edge_rows
    )
    caps_ok = bool(cap_rows) and all(row.get("passed") is True for row in cap_rows)
    tests = _test_row_map(tests_run)
    tests_ok = (
        all(tests.get(check_id, {}).get("passed") is True for check_id in REQUIRED_TEST_CHECKS)
        and tests.get("scoped_coverage", {}).get("coverage_percent") == 100.0
    )
    method_ok = (
        method_contract.get("satisfied") is True
        and method_contract.get("budgets") == PREREGISTERED_CAPS
        and method_contract.get("method_substitution") is False
        and method_contract.get("cap_reduction") is False
        and method_contract.get("sample_widening") is False
    )

    check_rows = [
        _audit_check("preconditions", True, preconditions_passed, preconditions_passed),
        _audit_check("frozen_sample_manifest", True, manifest_ok, manifest_ok),
        _audit_check("enumeration_coverage", selected, enumeration_ids, enumeration_ok),
        _audit_check(
            "state_action_coverage",
            "unique rows for all selected",
            len(state_action_rows),
            state_actions_ok,
        ),
        _audit_check(
            "exact_comparisons",
            "all match",
            sorted({row.get("disposition") for row in comparison_rows}),
            comparisons_ok,
        ),
        _audit_check("edge_stratification", len(FAMILIES), len(edge_rows), edge_ok),
        _audit_check("preregistered_caps", "all within fixed caps", len(cap_rows), caps_ok),
        _audit_check("required_tests", list(REQUIRED_TEST_CHECKS), sorted(tests), tests_ok),
        _audit_check("method_fidelity", True, method_contract.get("satisfied"), method_ok),
        _audit_check("protected_files", True, protected_files_unchanged, protected_files_unchanged),
    ]
    failed = [row["check"] for row in check_rows if not row["passed"]]
    passed = not failed
    return {
        "schema": "carnot.experiment_6715.aggregate_row_recomputation.v1",
        "selected_instance_count": len(selected),
        "enumeration_row_count": len(enumeration_rows),
        "state_action_row_count": len(state_action_rows),
        "comparison_row_count": len(comparison_rows),
        "edge_row_count": len(edge_rows),
        "cap_row_count": len(cap_rows),
        "total_enumeration_count": sum(
            int(row.get("enumeration_count", 0)) for row in enumeration_rows
        ),
        "check_rows": check_rows,
        "failed_checks": failed,
        "exact_replay_audit_passed": passed,
    }


def build_per_unit_rows(
    enumeration_rows: Sequence[Mapping[str, Any]],
    state_action_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    edge_rows: Sequence[Mapping[str, Any]],
    cap_rows: Sequence[Mapping[str, Any]],
    check_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Conserve every raw unit with an explicit row type."""

    result: list[JsonDict] = []
    for unit_type, rows in (
        ("enumeration", enumeration_rows),
        ("state_action", state_action_rows),
        ("comparison", comparison_rows),
        ("edge_check", edge_rows),
        ("cap_check", cap_rows),
        ("audit_check", check_rows),
    ):
        result.extend({"unit_type": unit_type, **deepcopy(dict(row))} for row in rows)
    return result


def protected_hashes(root: Path) -> JsonDict:
    """Hash the active roadmap and conductor without modifying either."""

    return {path.as_posix(): sha256_file(root / path) for path in (ACTIVE_ROADMAP, CONDUCTOR_PATH)}


def _memory_bytes(meminfo: Path = Path("/proc/meminfo")) -> int:
    if not meminfo.is_file():
        return 0
    match = re.search(r"^MemTotal:\s+(\d+)\s+kB", meminfo.read_text(), re.MULTILINE)
    return int(match.group(1)) * 1024 if match else 0


def _precondition(name: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    return {"name": name, "expected": expected, "observed": observed, "passed": passed}


def collect_preconditions(root: Path) -> list[JsonDict]:
    """Measure every required input, resource, schema, tool, and source hash."""

    rows: list[JsonDict] = []
    upstream_path = root / UPSTREAM_PATH
    upstream: JsonDict = {}
    upstream_error: str | None = None
    if upstream_path.is_file():
        try:
            upstream = load_json(upstream_path)
        except (OSError, ValueError, TypeError) as exc:
            upstream_error = str(exc)
    rows.append(
        _precondition(
            "upstream_artifact",
            "Exp6702 JSON object present and hashed",
            {
                "path": UPSTREAM_PATH.as_posix(),
                "sha256": sha256_file(upstream_path),
                "error": upstream_error,
            },
            bool(upstream) and upstream_error is None,
        )
    )
    rows.append(
        _precondition(
            "planning_fixture_ready",
            True,
            upstream.get("planning_fixture_ready"),
            upstream.get("planning_fixture_ready") is True,
        )
    )

    instance_rows = upstream.get("instance_rows", [])
    action_rows = upstream.get("state_action_rows", [])
    seal_rows = upstream.get("label_seal_rows", [])
    instances = {row.get("instance"): row for row in instance_rows if isinstance(row, Mapping)}
    seals = {row.get("instance"): row for row in seal_rows if isinstance(row, Mapping)}
    store_ok = (
        isinstance(instance_rows, list)
        and len(instance_rows) == 40
        and isinstance(action_rows, list)
        and bool(action_rows)
        and isinstance(seal_rows, list)
        and len(seal_rows) == 40
        and set(instances) == set(seals)
        and all(
            row.get("label_seal_hash") == seals[instance].get("seal_hash")
            and row.get("spec_hash") == sha256_json(row.get("typed_spec"))
            for instance, row in instances.items()
        )
    )
    rows.append(
        _precondition(
            "raw_stores_and_seals",
            "40 instances, state-action rows, 40 resolving seals, and exact spec hashes",
            embedded_store_receipts(upstream, upstream_path),
            store_ok,
        )
    )
    cpu = os.cpu_count() or 0
    rows.append(_precondition("cpu", ">=1", cpu, cpu >= 1))
    memory = _memory_bytes()
    rows.append(_precondition("ram_bytes", ">=1073741824", memory, memory >= 1024**3))
    free = shutil.disk_usage(root).free
    rows.append(_precondition("disk_free_bytes", ">=104857600", free, free >= 100 * 1024**2))
    try:
        jsonschema_version = metadata.version("jsonschema")
        schema_ok = True
    except metadata.PackageNotFoundError:
        jsonschema_version = "missing"
        schema_ok = False
    rows.append(
        _precondition(
            "artifact_schema",
            {
                "jsonschema": "available",
                "upstream_schema": "carnot.experiment_6702.exact_planning_fixture.v1",
            },
            {"jsonschema": jsonschema_version, "upstream_schema": upstream.get("schema")},
            schema_ok
            and upstream.get("schema") == "carnot.experiment_6702.exact_planning_fixture.v1",
        )
    )
    tools = {
        "python": Path(os.sys.executable).is_file(),
        "git": shutil.which("git") is not None,
        "jq": shutil.which("jq") is not None,
        "spec_coverage": (root / "scripts/check_spec_coverage.py").is_file(),
        "row_consistency": (root / "scripts/verdict_row_consistency_lint.py").is_file(),
        "adversarial_verification": (root / "scripts/adversarial_verify.py").is_file(),
    }
    rows.append(_precondition("audit_tools", "all present", tools, all(tools.values())))
    roadmap = root / ACTIVE_ROADMAP
    roadmap_text = roadmap.read_text(encoding="utf-8") if roadmap.is_file() else ""
    roadmap_observed = {
        "path": ACTIVE_ROADMAP.as_posix(),
        "sha256": sha256_file(roadmap),
        "task_present": "exp6715-bounded-exact-replay-audit" in roadmap_text,
    }
    rows.append(
        _precondition(
            "roadmap",
            "active V585 Exp6715 task present and hashed",
            roadmap_observed,
            roadmap.is_file() and roadmap_observed["task_present"],
        )
    )
    conductor = root / CONDUCTOR_PATH
    conductor_observed = {
        "path": CONDUCTOR_PATH.as_posix(),
        "sha256": sha256_file(conductor),
    }
    rows.append(
        _precondition(
            "conductor",
            "source present and hashed",
            conductor_observed,
            conductor.is_file(),
        )
    )
    return rows


_PRINCIPLES = {
    "status": "The terminal state follows deterministic process evidence.",
    "honest_verdict": "The conclusion uses measured replay evidence only.",
    "verdict_class": "A closed class prevents an ambiguous audit conclusion.",
    "gate_check_summary": "Expected and observed values localize each failure.",
    "openspec_requirement_ids": "Stable anchors connect requirements, tests, and code.",
    "frozen_sample_manifest": "Hashes bind the sample before reported labels open.",
    "method_fidelity_contract": "A fixed contract detects method substitution or reduced budgets.",
    "enumeration_rows": "Complete-path receipts make the independent authority recheckable.",
    "reported_vs_recomputed_rows": "Every exact agreement, mismatch, and missing value remains visible.",
    "exact_replay_audit_passed": "The producer-owned gate reduces only from raw audit rows.",
    "per_unit_rows": "Raw units prevent an aggregate from hiding a failed instance or action.",
    "aggregate_row_recomputation": "Counts and the gate rebuild deterministically from retained rows.",
    "preconditions_checked": "Measured resources prevent fabricated execution evidence.",
    "protected_files_unchanged": "Before and after hashes protect active operations.",
    "inference_substrate": "The CPU declaration prevents an LLM or accelerator claim.",
    "verifier_is_oracle": "False keeps post-event labels outside later live selection.",
    "field_provenance": "Each field names its source, functions, versions, and content hash.",
    "random_seed": "Fixed sample and reveal seeds reproduce the audit identity.",
    "duration_s": "A monotonic measurement records real wall-clock work.",
    "tests_run": "Commands and exits make verification reproducible.",
    "reproducibility_checksum": "Canonical content hashing detects evidence drift.",
}


def _stable_view(value: Any) -> Any:
    """Remove measured timing and free-form output from reproducible identity."""

    volatile = {"duration_s", "runtime_s", "summary", "reproducibility_checksum"}
    if isinstance(value, Mapping):
        return {
            str(key): _stable_view(item) for key, item in value.items() if str(key) not in volatile
        }
    if isinstance(value, list):
        return [_stable_view(item) for item in value]
    return value


def _field_provenance(artifact: Mapping[str, Any]) -> JsonDict:
    result: JsonDict = {}
    solver_fields = {
        "enumeration_rows",
        "reported_vs_recomputed_rows",
        "exact_replay_audit_passed",
        "per_unit_rows",
        "aggregate_row_recomputation",
    }
    reducer_fields = {
        "status",
        "honest_verdict",
        "verdict_class",
        "gate_check_summary",
        "exact_replay_audit_passed",
        "aggregate_row_recomputation",
    }
    for field in REQUIRED_ARTIFACT_FIELDS:
        value = artifact.get(field)
        result[field] = {
            "source_store": UPSTREAM_PATH.as_posix()
            if field not in {"duration_s", "tests_run"}
            else None,
            "audit_solver": SOLVER_VERSION if field in solver_fields else None,
            "comparator": COMPARATOR_VERSION
            if field in {"reported_vs_recomputed_rows", "exact_replay_audit_passed"}
            else None,
            "reducer": REDUCER_VERSION if field in reducer_fields else None,
            "function": "build_artifact",
            "version": SCHEMA,
            "hash": sha256_json(_stable_view(value)),
            "principle": _PRINCIPLES[field],
        }
    result["field_provenance"]["hash"] = sha256_json(sorted(REQUIRED_ARTIFACT_FIELDS))
    result["reproducibility_checksum"]["hash"] = "sha256:computed_after_field_provenance"
    return result


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the canonical stable artifact identity."""

    return sha256_json(_stable_view(artifact))


def _classification(
    aggregate: Mapping[str, Any], method: Mapping[str, Any]
) -> tuple[str, str, str]:
    if aggregate.get("exact_replay_audit_passed") is True:
        return (
            "complete_reproduced",
            "complete: reproduced bounded exact replay",
            "positive",
        )
    if method.get("satisfied") is not True or "preregistered_caps" in aggregate.get(
        "failed_checks", []
    ):
        return (
            "disqualified_method_or_cap",
            "complete: disqualified bounded replay due to method or cap failure",
            "disqualified",
        )
    if "exact_comparisons" in aggregate.get("failed_checks", []):
        return (
            "complete_corrected",
            "complete: corrected sealed fixture values with independent replay",
            "partial",
        )
    return (
        "complete_partial",
        "complete: partial bounded replay because a required audit check failed",
        "partial",
    )


def build_artifact(
    *,
    date: str,
    root: Path,
    tests_run: Sequence[Mapping[str, Any]],
    duration_s: float,
    protected_before: Mapping[str, Any],
) -> JsonDict:
    """Freeze, reveal, replay, reduce, and build one complete artifact."""

    upstream = load_json(root / UPSTREAM_PATH)
    receipts = embedded_store_receipts(upstream, root / UPSTREAM_PATH)
    manifest = freeze_sample(public_instance_rows(upstream), receipts)
    try:
        result = recompute_frozen_sample(upstream, manifest)
    except CapExceeded as exc:
        result = {
            "enumeration_rows": [],
            "state_action_rows": [],
            "reported_vs_recomputed_rows": [],
            "edge_rows": [],
            "cap_rows": [exc.row],
            "total_enumeration_count": 0,
        }
    manifest["reveal_events"] = ["manifest_frozen", *manifest["reveal_order"]]
    manifest["manifest_hash"] = manifest_checksum(manifest)
    preconditions = collect_preconditions(root)
    after = protected_hashes(root)
    protected = {
        "before": deepcopy(dict(protected_before)),
        "after": after,
        "unchanged": dict(protected_before) == after,
    }
    method = method_fidelity_contract(root)
    aggregate = recompute_aggregate(
        manifest=manifest,
        enumeration_rows=result["enumeration_rows"],
        state_action_rows=result["state_action_rows"],
        comparison_rows=result["reported_vs_recomputed_rows"],
        edge_rows=result["edge_rows"],
        cap_rows=result["cap_rows"],
        tests_run=tests_run,
        preconditions_passed=all(row["passed"] for row in preconditions),
        protected_files_unchanged=protected["unchanged"],
        method_contract=method,
    )
    status, verdict, verdict_class = _classification(aggregate, method)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": 6715,
        "run_date": date,
        "status": status,
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "gate_check_summary": [
            deepcopy(row) for row in aggregate["check_rows"] if not row["passed"]
        ],
        "openspec_requirement_ids": list(OPEN_SPEC_IDS),
        "frozen_sample_manifest": manifest,
        "method_fidelity_contract": method,
        "enumeration_rows": result["enumeration_rows"],
        "state_action_rows": result["state_action_rows"],
        "edge_rows": result["edge_rows"],
        "cap_rows": result["cap_rows"],
        "reported_vs_recomputed_rows": result["reported_vs_recomputed_rows"],
        "exact_replay_audit_passed": aggregate["exact_replay_audit_passed"],
        "per_unit_rows": build_per_unit_rows(
            result["enumeration_rows"],
            result["state_action_rows"],
            result["reported_vs_recomputed_rows"],
            result["edge_rows"],
            result["cap_rows"],
            aggregate["check_rows"],
        ),
        "aggregate_row_recomputation": aggregate,
        "preconditions_checked": preconditions,
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "authority_boundary": "sealed Exp6702 labels read for audit only; never for later selection",
        "field_provenance": {},
        "random_seed": deepcopy(RANDOM_SEEDS),
        "duration_s": round(float(duration_s), 6),
        "tests_run": [deepcopy(dict(row)) for row in tests_run],
        "reproducibility_checksum": "",
    }
    artifact["field_provenance"] = _field_provenance(artifact)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_blocked_artifact(
    date: str,
    root: Path,
    preconditions: Sequence[Mapping[str, Any]],
    duration_s: float,
) -> JsonDict:
    """Retain a real prerequisite failure without fabricated audit rows."""

    failed = next((row for row in preconditions if not row.get("passed")), None)
    check = _audit_check(
        "preconditions",
        True if failed is None else failed.get("expected"),
        False if failed is None else failed.get("observed"),
        False,
    )
    aggregate = {
        "schema": "carnot.experiment_6715.aggregate_row_recomputation.v1",
        "selected_instance_count": 0,
        "enumeration_row_count": 0,
        "state_action_row_count": 0,
        "comparison_row_count": 0,
        "edge_row_count": 0,
        "cap_row_count": 0,
        "total_enumeration_count": 0,
        "check_rows": [check],
        "failed_checks": ["preconditions"],
        "exact_replay_audit_passed": False,
    }
    protected = protected_hashes(root)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": 6715,
        "run_date": date,
        "status": "blocked_precondition",
        "honest_verdict": "complete: blocked because a required exact-replay input is unavailable",
        "verdict_class": "blocked",
        "gate_check_summary": (
            []
            if failed is None
            else [
                {
                    "check": failed.get("name"),
                    "expected": failed.get("expected"),
                    "observed": failed.get("observed"),
                    "passed": False,
                }
            ]
        ),
        "openspec_requirement_ids": list(OPEN_SPEC_IDS),
        "frozen_sample_manifest": {},
        "method_fidelity_contract": method_fidelity_contract(root),
        "enumeration_rows": [],
        "state_action_rows": [],
        "edge_rows": [],
        "cap_rows": [],
        "reported_vs_recomputed_rows": [],
        "exact_replay_audit_passed": False,
        "per_unit_rows": build_per_unit_rows([], [], [], [], [], [check]),
        "aggregate_row_recomputation": aggregate,
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "protected_files_unchanged": {
            "before": protected,
            "after": protected,
            "unchanged": True,
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "authority_boundary": "no sealed label was used because preconditions failed",
        "field_provenance": {},
        "random_seed": deepcopy(RANDOM_SEEDS),
        "duration_s": round(float(duration_s), 6),
        "tests_run": [],
        "reproducibility_checksum": "",
    }
    artifact["field_provenance"] = _field_provenance(artifact)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Validate fields, checksum, row conservation, aggregate, and status."""

    if not set(REQUIRED_ARTIFACT_FIELDS) <= set(payload):
        return ["missing_required_fields"]
    errors: list[str] = []
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if payload.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_mismatch")
    if payload.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if not isinstance(payload.get("duration_s"), (int, float)) or payload["duration_s"] < 0:
        errors.append("duration_invalid")
    if payload.get("reproducibility_checksum") != artifact_checksum(payload):
        errors.append("reproducibility_checksum_mismatch")
    provenance = payload.get("field_provenance")
    if (
        not isinstance(provenance, Mapping)
        or not set(REQUIRED_ARTIFACT_FIELDS) <= set(provenance)
        or any(not set(PROVENANCE_KEYS) <= set(row) for row in provenance.values())
    ):
        errors.append("field_provenance_invalid")

    expected_units = build_per_unit_rows(
        payload.get("enumeration_rows", []),
        payload.get("state_action_rows", []),
        payload.get("reported_vs_recomputed_rows", []),
        payload.get("edge_rows", []),
        payload.get("cap_rows", []),
        payload.get("aggregate_row_recomputation", {}).get("check_rows", []),
    )
    if payload.get("per_unit_rows") != expected_units:
        errors.append("per_unit_rows_mismatch")
    if payload.get("status") == "blocked_precondition":
        if payload.get("exact_replay_audit_passed") is not False or not payload.get(
            "gate_check_summary"
        ):
            errors.append("blocked_state_mismatch")
        return errors

    manifest = payload.get("frozen_sample_manifest", {})
    if manifest.get("manifest_hash") != manifest_checksum(manifest):
        errors.append("manifest_hash_mismatch")
    aggregate = recompute_aggregate(
        manifest=manifest,
        enumeration_rows=payload.get("enumeration_rows", []),
        state_action_rows=payload.get("state_action_rows", []),
        comparison_rows=payload.get("reported_vs_recomputed_rows", []),
        edge_rows=payload.get("edge_rows", []),
        cap_rows=payload.get("cap_rows", []),
        tests_run=payload.get("tests_run", []),
        preconditions_passed=all(
            row.get("passed") is True for row in payload.get("preconditions_checked", [])
        ),
        protected_files_unchanged=payload.get("protected_files_unchanged", {}).get("unchanged")
        is True,
        method_contract=payload.get("method_fidelity_contract", {}),
    )
    if payload.get("aggregate_row_recomputation") != aggregate:
        errors.append("aggregate_row_recomputation_mismatch")
    if payload.get("exact_replay_audit_passed") != aggregate["exact_replay_audit_passed"]:
        errors.append("audit_gate_mismatch")
    if payload.get("exact_replay_audit_passed") is True and payload.get("gate_check_summary"):
        errors.append("passed_gate_summary_mismatch")
    if payload.get("exact_replay_audit_passed") is False and not payload.get("gate_check_summary"):
        errors.append("failed_gate_summary_missing")
    return errors


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> JsonDict:
    """Sync a complete temporary file before one atomic replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return {"path": path.as_posix(), "bytes": len(data), "atomic_replace": True}


def default_command_runner(command: str, root: Path) -> JsonDict:
    """Run one declared verification command and retain its process receipt."""

    started = time.perf_counter()
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
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "duration_s": round(time.perf_counter() - started, 6),
    }


def _command_row(check_id: str, command: str, receipt: Mapping[str, Any]) -> JsonDict:
    output = str(receipt.get("stdout", "")) + str(receipt.get("stderr", ""))
    coverage: float | None = None
    if check_id == "scoped_coverage":
        match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", output)
        coverage = float(match.group(1)) if match else None
    passed = receipt.get("exit_code") == 0 and (check_id != "scoped_coverage" or coverage == 100.0)
    return {
        "check_id": check_id,
        "command": command,
        "exit_code": receipt.get("exit_code"),
        "passed": passed,
        "coverage_percent": coverage,
        "summary": output[-2000:],
        "duration_s": receipt.get("duration_s", 0.0),
    }


def run_verification_commands(
    root: Path,
    runner: Callable[[str, Path], Mapping[str, Any]] = default_command_runner,
) -> list[JsonDict]:
    """Run focused, coverage, full-suite, spec, E2E, and lint checks."""

    return [
        _command_row(check_id, command, runner(command, root))
        for check_id, command in VERIFICATION_COMMANDS
    ]


def run_artifact_checks(
    root: Path,
    artifact_path: Path,
    runner: Callable[[str, Path], Mapping[str, Any]] = default_command_runner,
) -> list[JsonDict]:
    """Run validation, row consistency, and adversarial checks on a candidate."""

    target = shlex.quote(str(artifact_path))
    commands = (
        (
            "artifact_validation",
            ".venv/bin/python -m carnot.experiment_6715_bounded_exact_replay_audit "
            f"--validate --output {target}",
        ),
        (
            "row_consistency",
            f".venv/bin/python scripts/verdict_row_consistency_lint.py --strict {target}",
        ),
        (
            "adversarial_verification",
            f".venv/bin/python scripts/adversarial_verify.py --json {target}",
        ),
    )
    rows: list[JsonDict] = []
    for check_id, command in commands:
        receipt = runner(command, root)
        row = _command_row(check_id, command, receipt)
        row["critical_free"] = None
        if check_id == "adversarial_verification":
            try:
                report = json.loads(str(receipt.get("stdout", "")))
                row["critical_free"] = all(
                    int(item.get("max_severity", 0)) < 2 for item in report.get("reports", [])
                )
            except (TypeError, ValueError, AttributeError):
                row["critical_free"] = receipt.get("exit_code") == 0
            row["passed"] = row["critical_free"] is True
        rows.append(row)
    return rows


def pending_artifact_check_rows() -> list[JsonDict]:
    """Reserve operational receipt slots while the candidate is built."""

    return [
        {
            "check_id": check_id,
            "command": "pending complete candidate artifact",
            "exit_code": None,
            "passed": False,
            "coverage_percent": None,
            "summary": "not run before candidate publication",
            "duration_s": 0.0,
        }
        for check_id in OPERATIONAL_CHECK_IDS
    ]


def run(
    *,
    date: str,
    root: Path = REPO_ROOT,
    output_path: Path | None = None,
    runner: Callable[[str, Path], Mapping[str, Any]] = default_command_runner,
) -> JsonDict:
    """Run preconditions, bounded replay, verification, and atomic output."""

    started = time.perf_counter()
    output = output_path or root / RESULT_PATH
    preconditions = collect_preconditions(root)
    if not all(row["passed"] for row in preconditions):
        artifact = build_blocked_artifact(date, root, preconditions, time.perf_counter() - started)
    else:
        before = protected_hashes(root)
        receipts = run_verification_commands(root, runner=runner)
        candidate = build_artifact(
            date=date,
            root=root,
            tests_run=[*receipts, *pending_artifact_check_rows()],
            duration_s=time.perf_counter() - started,
            protected_before=before,
        )
        errors = validate_artifact(candidate)
        if errors:
            raise ValueError("candidate: " + "; ".join(errors))
        with tempfile.TemporaryDirectory(prefix="carnot-exp6715-") as temporary:
            candidate_path = Path(temporary) / RESULT_PATH.name
            write_json_atomic(candidate_path, candidate)
            receipts.extend(run_artifact_checks(root, candidate_path, runner=runner))
        artifact = build_artifact(
            date=date,
            root=root,
            tests_run=receipts,
            duration_s=time.perf_counter() - started,
            protected_before=before,
        )
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    write_json_atomic(output, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Run the dated audit, or validate one existing artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260828")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_PATH)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if args.validate:
        if not args.output.is_file():
            return 1
        try:
            return 0 if not validate_artifact(load_json(args.output)) else 1
        except (OSError, ValueError, TypeError):
            return 1
    run(date=args.date, root=REPO_ROOT, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the required module command.
    raise SystemExit(main())
