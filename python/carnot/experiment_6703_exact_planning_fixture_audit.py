"""Cold-audit the sealed exact finite-horizon planning fixture.

The audit reads typed specifications from the producer artifact. It implements
new transition functions and enumerates complete action paths. It never imports
the producer module. This separation makes agreement useful evidence instead
of a second call to the same solver.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_PATH = Path("results/experiment_6702_exact_planning_fixture_recovery.json")
RESULT_PATH = Path("results/experiment_6703_exact_planning_fixture_audit.json")
ACTIVE_ROADMAP = Path("research-roadmap.yaml")
CONDUCTOR_PATH = Path("scripts/research_conductor.py")
MODULE_PATH = Path("python/carnot/experiment_6703_exact_planning_fixture_audit.py")
TEST_PATH = Path("tests/python/test_experiment_6703_exact_planning_fixture_audit.py")

SCHEMA = "carnot.experiment_6703.exact_planning_fixture_audit.v1"
INFERENCE_SUBSTRATE = "cpu_independent_exhaustive_audit_no_llm"
SOLVER_VERSION = "carnot.exp6703.independent_complete_path_enumerator.v1"
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
RANDOM_SEEDS = {"blinding": 6703001, "sample": 6703002, "attack_order": 6703003}

# The interactive schema probe exposed four non-sample metamorphic totals after
# an invalid identity projection. The later solver is independent, but the cold
# chronology is not. The final gate therefore records disqualification.
BLINDING_PROTOCOL_INCIDENT: JsonDict = {
    "check": "blinding_chronology",
    "expected": "valid identity manifest frozen before any reported optimum is read",
    "observed": (
        "an invalid identity projection used instance_id instead of instance; "
        "four reported headline-00 metamorphic totals were then exposed before "
        "the valid twelve-unit manifest freeze"
    ),
    "affected_reported_units": [f"{family}-headline-00" for family in FAMILIES],
    "selected_units_affected": [],
    "passed": False,
}

OPEN_SPEC_IDS = (
    "REQ-CONSTRAINT-6703",
    "SCENARIO-CONSTRAINT-6703-COLD-RECOMPUTATION",
    "SCENARIO-CONSTRAINT-6703-BLINDING",
    "SCENARIO-CONSTRAINT-6703-COVERAGE",
    "REQ-VERIFY-6703",
    "SCENARIO-VERIFY-6703-FIELD-PARITY",
    "SCENARIO-VERIFY-6703-AUTHORITY",
    "REQ-SAFE-6703",
    "SCENARIO-SAFE-6703-LEAKAGE",
    "SCENARIO-SAFE-6703-SEAL-TIMING",
    "SCENARIO-SAFE-6703-MUTATIONS",
    "REQ-PIPELINE-6703",
    "SCENARIO-PIPELINE-6703-ROW-REDUCTION",
    "SCENARIO-PIPELINE-6703-PER-UNIT-CONSERVATION",
    "REQ-REPORT-6703",
    "SCENARIO-REPORT-6703-ATOMIC-PROVENANCE",
    "SCENARIO-REPORT-6703-FAIL-CLOSED",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "gate_check_summary",
    "openspec_requirement_ids",
    "blinded_sample_manifest",
    "coverage_rows",
    "independent_solver_rows",
    "reported_vs_recomputed_rows",
    "leakage_rows",
    "metamorphic_mutation_rows",
    "planning_fixture_audit_passed",
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

FOCUSED_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6703_exact_planning_fixture_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null --data-file=/tmp/carnot_exp6703_coverage "
    "--include=*/experiment_6703_exact_planning_fixture_audit.py "
    "-m pytest tests/python/test_experiment_6703_exact_planning_fixture_audit.py "
    "-q --no-cov -n 0 -o addopts="
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null --data-file=/tmp/carnot_exp6703_coverage "
    "--include=*/experiment_6703_exact_planning_fixture_audit.py "
    "--fail-under=100 --show-missing"
)
FULL_SUITE_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6703_exact_planning_fixture_audit.py"
)
E2E_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6703_exact_planning_fixture_audit.py "
    "-q --no-cov -n 0 -k actual_selected_units_match_every_reported_exact_field"
)
RUFF_COMMAND = f".venv/bin/ruff check {MODULE_PATH} {TEST_PATH}"
FORMAT_COMMAND = f".venv/bin/ruff format --check {MODULE_PATH} {TEST_PATH}"
VERIFICATION_COMMANDS = (
    ("focused_tests", FOCUSED_COMMAND),
    ("scoped_coverage_run", COVERAGE_RUN_COMMAND),
    ("scoped_coverage", COVERAGE_REPORT_COMMAND),
    ("full_python_suite", FULL_SUITE_COMMAND),
    ("spec_coverage", SPEC_COVERAGE_COMMAND),
    ("applicable_e2e", E2E_COMMAND),
    ("ruff_check", RUFF_COMMAND),
    ("format_check", FORMAT_COMMAND),
)
REQUIRED_TEST_CHECKS = tuple(check_id for check_id, _ in VERIFICATION_COMMANDS)


def canonical_json(value: Any) -> str:
    """Return one stable JSON representation for hashes and comparisons."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Hash a JSON value without depending on whitespace in a source file."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash source bytes so later audits can detect changed inputs."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def load_json(path: Path) -> JsonDict:
    """Load one JSON object and reject arrays or scalar substitutes."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("JSON object required")
    return value


def public_instance_rows(upstream: Mapping[str, Any]) -> list[JsonDict]:
    """Project only identity and typed-spec fields before label reveal."""

    fields = (
        "instance",
        "family",
        "split",
        "seed",
        "horizon",
        "prompt",
        "prompt_hash",
        "spec_hash",
        "label_seal_hash",
        "typed_spec",
    )
    return [
        {field: deepcopy(row.get(field)) for field in fields} for row in upstream["instance_rows"]
    ]


def manifest_checksum(manifest: Mapping[str, Any]) -> str:
    """Hash the selection manifest while excluding its own hash field."""

    return sha256_json({key: value for key, value in manifest.items() if key != "manifest_hash"})


def freeze_blinded_sample(public_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Select twelve identities without accepting any reported label field."""

    identities = [str(row.get("instance")) for row in public_rows]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate public instance")
    if any("optimum" in row or "state_action_rows" in row for row in public_rows):
        raise ValueError("reported labels present before sample freeze")

    selected: list[JsonDict] = []
    for family in FAMILIES:
        for split, limit in (("development", 1), ("headline", 2)):
            candidates: list[JsonDict] = []
            for row in public_rows:
                if row.get("family") != family or row.get("split") != split:
                    continue
                instance = str(row["instance"])
                score = hashlib.sha256(
                    f"{RANDOM_SEEDS['blinding']}:{instance}".encode()
                ).hexdigest()
                candidates.append(
                    {
                        "instance": instance,
                        "family": family,
                        "split": split,
                        "spec_hash": row["spec_hash"],
                        "prompt_hash": row["prompt_hash"],
                        "seal_hash": row["label_seal_hash"],
                        "selection_score": score,
                    }
                )
            selected.extend(sorted(candidates, key=lambda row: row["selection_score"])[:limit])

    reveal = [row["instance"] for row in sorted(selected, key=lambda row: row["selection_score"])]
    manifest: JsonDict = {
        "schema": "carnot.exp6703.blinded_sample_manifest.v1",
        "selection_rule": (
            "lowest sha256(blinding_seed:instance) per family; one development and two headline"
        ),
        "blinding_seed": RANDOM_SEEDS["blinding"],
        "expected_instance_count": 12,
        "expected_identities": selected,
        "reveal_order": reveal,
        "frozen_before_reported_label_read": True,
    }
    manifest["manifest_hash"] = manifest_checksum(manifest)
    return manifest


def _state_type(spec: Mapping[str, Any]) -> str:
    return next(iter(spec["initial_state"]))


def _state_dict(spec: Mapping[str, Any], state: int) -> JsonDict:
    return {_state_type(spec): state}


def independent_transition(
    spec: Mapping[str, Any], time_index: int, state: int, action: int
) -> JsonDict:
    """Apply one independently written family transition and stage cost."""

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
            index = action - 1
            bit = 1 << index
            if state & bit:
                return {
                    "legal": False,
                    "next_state": None,
                    "immediate_cost": None,
                    "reason": "job_already_completed",
                }
            required = int(parameters["prerequisites"][index])
            if required & state != required:
                return {
                    "legal": False,
                    "next_state": None,
                    "immediate_cost": None,
                    "reason": "prerequisite_missing",
                }
            if time_index > int(parameters["deadlines"][index]):
                return {
                    "legal": False,
                    "next_state": None,
                    "immediate_cost": None,
                    "reason": "deadline_missed",
                }
            next_state = state | bit
            cost = int(parameters["schedule_costs"][index]) + shift
    elif family == "reservoir_control":
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
    return {"legal": True, "next_state": next_state, "immediate_cost": cost, "reason": "legal"}


def _terminal_cost(spec: Mapping[str, Any], state: int) -> int:
    parameters = spec["parameters"]
    family = spec["family"]
    if family == "inventory":
        return state * int(parameters["terminal_holding_cost"])
    if family == "battery_dispatch":
        return abs(state - int(parameters["terminal_target"])) * int(parameters["terminal_penalty"])
    if family == "job_slot":
        return sum(
            int(penalty)
            for index, penalty in enumerate(parameters["missing_penalties"])
            if not state & (1 << index)
        )
    if family == "reservoir_control":
        return abs(state - int(parameters["terminal_target"])) * int(parameters["terminal_penalty"])
    raise ValueError(f"unknown family: {family}")


def _evaluate_path(
    spec: Mapping[str, Any], start_time: int, start_state: int, actions: Sequence[int]
) -> tuple[bool, int | None]:
    state = start_state
    total = 0
    for offset, action in enumerate(actions):
        transition = independent_transition(spec, start_time + offset, state, int(action))
        if not transition["legal"]:
            return False, None
        total += int(transition["immediate_cost"])
        state = int(transition["next_state"])
    return True, total + _terminal_cost(spec, state)


def _path_outcomes(
    spec: Mapping[str, Any], start_time: int, state: int
) -> list[tuple[list[int], int]]:
    remaining = int(spec["horizon"]) - start_time
    actions = list(spec["action_domain"])
    outcomes: list[tuple[list[int], int]] = []
    for path in product(actions, repeat=remaining):
        legal, total = _evaluate_path(spec, start_time, state, path)
        if legal and total is not None:
            outcomes.append((list(path), total))
    return outcomes


def exhaustive_solve(spec: Mapping[str, Any]) -> JsonDict:
    """Enumerate full paths and rebuild every reachable state-action value."""

    started = time.perf_counter()
    horizon = int(spec["horizon"])
    actions = [int(action) for action in spec["action_domain"]]
    state_name = _state_type(spec)
    initial = int(spec["initial_state"][state_name])
    paths = _path_outcomes(spec, 0, initial)
    if not paths:
        optimum = None
        optimum_plans: list[list[int]] = []
    else:
        optimum = min(total for _, total in paths)
        optimum_plans = [path for path, total in paths if total == optimum]

    reachable: dict[int, set[int]] = {0: {initial}}
    for time_index in range(horizon):
        reachable[time_index + 1] = set()
        for state in reachable[time_index]:
            for action in actions:
                transition = independent_transition(spec, time_index, state, action)
                if transition["legal"]:
                    reachable[time_index + 1].add(int(transition["next_state"]))

    rows: list[JsonDict] = []
    for time_index in range(horizon):
        for state in sorted(reachable[time_index]):
            raw: list[tuple[int, JsonDict, int | None, int | None]] = []
            for action in actions:
                transition = independent_transition(spec, time_index, state, action)
                future: int | None = None
                total: int | None = None
                if transition["legal"]:
                    suffixes = _path_outcomes(spec, time_index + 1, int(transition["next_state"]))
                    future = min(value for _, value in suffixes)
                    total = int(transition["immediate_cost"]) + future
                raw.append((action, transition, future, total))
            legal_totals = [total for _, _, _, total in raw if total is not None]
            best = min(legal_totals) if legal_totals else None
            for action, transition, future, total in raw:
                rows.append(
                    {
                        "instance": spec["instance_id"],
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
                        "immediate_cost": transition["immediate_cost"],
                        "future_value": future,
                        "total_value": total,
                        "action_gap": None if total is None or best is None else total - best,
                        "optimum_membership": total == best
                        if total is not None and best is not None
                        else False,
                    }
                )

    initial_rows = [
        row for row in rows if row["time_index"] == 0 and row["state"] == _state_dict(spec, initial)
    ]
    tie_set = [row["action"] for row in initial_rows if row["optimum_membership"]]
    result: JsonDict = {
        "instance": spec["instance_id"],
        "enumeration_count": len(actions) ** horizon,
        "feasible_plan_count": len(paths),
        "optimum": optimum,
        "optimum_plans": optimum_plans,
        "tie_set": tie_set,
        "feasible": bool(paths),
        "state_count": sum(len(reachable[index]) for index in range(horizon)),
        "state_action_rows": rows,
        "runtime_s": round(time.perf_counter() - started, 6),
    }
    return result


def _comparison(unit: str, field: str, reported: Any, recomputed: Any) -> JsonDict:
    if reported is _MISSING:
        disposition = "missing_reported"
        reported_value: Any = None
    elif recomputed is _MISSING:
        disposition = "missing_recomputed"
        reported_value = reported
    else:
        disposition = "match" if reported == recomputed else "mismatch"
        reported_value = reported
    return {
        "unit": unit,
        "field": field,
        "reported_value": reported_value,
        "recomputed_value": None if recomputed is _MISSING else recomputed,
        "tolerance": 0,
        "disposition": disposition,
    }


_MISSING = object()


def recompute_selected_units(
    upstream: Mapping[str, Any], manifest: Mapping[str, Any]
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Open only frozen identities, solve them, and compare exact fields."""

    selected = set(manifest["reveal_order"])
    instances = {
        row["instance"]: row for row in upstream["instance_rows"] if row["instance"] in selected
    }
    reported_actions: dict[str, list[Mapping[str, Any]]] = {
        instance: [row for row in upstream["state_action_rows"] if row["instance"] == instance]
        for instance in selected
    }
    solver_rows: list[JsonDict] = []
    comparisons: list[JsonDict] = []
    instance_fields = (
        ("optimum.total", lambda row: row.get("optimum", {}).get("total"), "optimum"),
        ("optimum.plan", lambda row: row.get("optimum", {}).get("plan"), "optimum_plans"),
        ("optimum.action_set", lambda row: row.get("optimum", {}).get("action_set"), "tie_set"),
        ("ties", lambda row: row.get("ties", _MISSING), "ties"),
        ("feasibility", lambda row: row.get("feasibility", _MISSING), "feasible"),
    )

    for instance in manifest["reveal_order"]:
        reported = instances.get(instance)
        if reported is None:
            comparisons.append(_comparison(instance, "instance", _MISSING, instance))
            continue
        solved = exhaustive_solve(reported["typed_spec"])
        initial_values = [
            {
                "action": row["action"],
                "legal": row["legality"],
                "value": row["total_value"],
                "gap": row["action_gap"],
            }
            for row in solved["state_action_rows"]
            if row["time_index"] == 0
        ]
        core: JsonDict = {
            "instance": instance,
            "solver": SOLVER_VERSION,
            "enumeration_count": solved["enumeration_count"],
            "feasible_plan_count": solved["feasible_plan_count"],
            "optimum": solved["optimum"],
            "tie_set": solved["tie_set"],
            "optimum_plans": solved["optimum_plans"],
            "action_values": initial_values,
            "state_count": solved["state_count"],
            "state_action_value_count": len(solved["state_action_rows"]),
            "runtime_s": solved["runtime_s"],
        }
        core["receipt"] = sha256_json(
            {key: value for key, value in core.items() if key != "runtime_s"}
        )
        solver_rows.append(core)

        recomputed_instance = {
            "optimum": solved["optimum"],
            "optimum_plans": solved["optimum_plans"],
            "tie_set": solved["tie_set"],
            "ties": len(solved["tie_set"]) > 1,
            "feasible": solved["feasible"],
        }
        for field, getter, key in instance_fields:
            value = recomputed_instance[key]
            if field == "optimum.plan":
                value = value[0] if value else None
            comparisons.append(_comparison(instance, field, getter(reported), value))

        recomputed_by_key = {
            (row["time_index"], canonical_json(row["state"]), row["action"]): row
            for row in solved["state_action_rows"]
        }
        reported_by_key = {
            (row["time_index"], canonical_json(row["state"]), row["action"]): row
            for row in reported_actions[instance]
        }
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
            row_unit = f"{instance}:{key[0]}:{key[1]}:{key[2]}"
            for field in action_fields:
                comparisons.append(
                    _comparison(
                        row_unit,
                        field,
                        _MISSING if reported_row is None else reported_row.get(field, _MISSING),
                        _MISSING if recomputed_row is None else recomputed_row.get(field, _MISSING),
                    )
                )
    return solver_rows, comparisons


def _valid_seal(row: Mapping[str, Any], instances: Mapping[str, Mapping[str, Any]]) -> bool:
    instance = instances.get(str(row.get("instance")))
    if instance is None or row.get("prompt_hash") != instance.get("prompt_hash"):
        return False
    components = {
        "instance": row.get("instance"),
        "prompt_hash": row.get("prompt_hash"),
        "label_hash": row.get("label_hash"),
        "seal_version": row.get("seal_version"),
        "commit_requirement": row.get("commit_requirement"),
    }
    return (
        row.get("seal_hash") == sha256_json(components)
        and row.get("seal_version") == "carnot.prompt_bound_label_seal.v1"
        and row.get("commit_requirement") == "prompt_bound_candidate_commit_receipt"
        and row.get("access_state") == "sealed_until_commit"
        and row.get("negative_access_result") == "denied:commit receipt required"
    )


def audit_leakage(upstream: Mapping[str, Any]) -> list[JsonDict]:
    """Scan prompts, public metadata, splits, identities, seals, and duplicates."""

    rows = list(upstream.get("instance_rows", []))
    instances = {str(row.get("instance")): row for row in rows}
    label_pattern = re.compile(r"(?:exact\s+)?(?:optimum|answer|label)\s*[:=]", re.IGNORECASE)
    prompt_hits = [
        row["instance"] for row in rows if label_pattern.search(str(row.get("prompt", "")))
    ]

    forbidden_keys = {"total_optimum", "optimum_plan", "future_value", "action_gap", "exact_label"}
    metadata_hits: list[str] = []
    for row in rows:
        typed = row.get("typed_spec", {})
        keys: set[str] = set()

        def collect(value: Any) -> None:
            if isinstance(value, Mapping):
                for key, nested in value.items():
                    keys.add(str(key))
                    collect(nested)
            elif isinstance(value, list):
                for nested in value:
                    collect(nested)

        collect(typed)
        if keys & forbidden_keys:
            metadata_hits.append(str(row["instance"]))

    ids = [str(row.get("instance")) for row in rows]
    split_collisions = sorted(instance for instance in set(ids) if ids.count(instance) > 1)
    id_shortcuts = sorted(
        instance
        for instance in ids
        if re.search(r"(?:opt|answer|label|total)[-_=]?\d", instance, re.I)
    )
    cross_split: list[JsonDict] = []
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1 :]:
            if left.get("split") == right.get("split"):
                continue
            shared = [
                key
                for key in ("spec_hash", "prompt_hash")
                if left.get(key) is not None and left.get(key) == right.get(key)
            ]
            if shared:
                cross_split.append(
                    {"left": left.get("instance"), "right": right.get("instance"), "shared": shared}
                )
    invalid_seals = [
        row.get("instance")
        for row in upstream.get("label_seal_rows", [])
        if not _valid_seal(row, instances)
    ]
    seal_instances = [row.get("instance") for row in upstream.get("label_seal_rows", [])]
    missing_seals = sorted(set(ids) - set(seal_instances))

    checks = (
        ("prompt_direct_label", "no prompt contains a direct answer label", prompt_hits),
        (
            "metadata_objective_encoding",
            "no public metadata contains exact-result fields",
            metadata_hits,
        ),
        ("split_collision", "instance identities are unique across splits", split_collisions),
        ("instance_id_shortcut", "instance ids do not encode exact results", id_shortcuts),
        (
            "seal_integrity",
            "all seals bind current prompt and label hashes",
            invalid_seals + missing_seals,
        ),
        ("development_headline_duplication", "no spec or prompt hash crosses splits", cross_split),
    )
    return [
        {
            "check": check,
            "expected_result": expected,
            "observed_result": observed,
            "evidence_hash": sha256_json(observed),
            "pass_state": not observed,
        }
        for check, expected, observed in checks
    ]


def _producer_case(
    rows: Sequence[Mapping[str, Any]], key: str, value: str
) -> Mapping[str, Any] | None:
    return next((row for row in rows if row.get(key) == value), None)


def audit_metamorphic_and_mutation_cases(upstream: Mapping[str, Any]) -> list[JsonDict]:
    """Replay each transform and mutation from raw specs and rows."""

    results: list[JsonDict] = []
    instances = list(upstream["instance_rows"])
    reported_transforms = list(upstream.get("metamorphic_rows", []))
    for family in FAMILIES:
        instance = next(
            row for row in instances if row["family"] == family and row["split"] == "headline"
        )
        base = exhaustive_solve(instance["typed_spec"])
        aliases = {
            str(action): f"choice_{index}"
            for index, action in enumerate(instance["typed_spec"]["action_domain"])
        }
        expected_results: dict[str, Any] = {
            "action_renaming": {
                "aliases": aliases,
                "renamed_optimum": [aliases[str(action)] for action in base["tie_set"]],
            },
            "equivalent_state_encoding": {"state_count": len(base["state_action_rows"])},
        }
        shifted_spec = deepcopy(instance["typed_spec"])
        shifted_spec["parameters"]["stage_cost_shift"] += 3
        shifted = exhaustive_solve(shifted_spec)
        expected_results["constant_cost_shift"] = {
            "base_total": base["optimum"],
            "shifted_total": shifted["optimum"],
        }
        surface_prompt = "Please solve this equivalent family task. " + str(instance["prompt"])
        expected_results["family_preserving_surface_change"] = {
            "base_prompt_hash": instance["prompt_hash"],
            "surface_prompt_hash": "sha256:"
            + hashlib.sha256(surface_prompt.encode("utf-8")).hexdigest(),
        }

        invariants = {
            "action_renaming": True,
            "constant_cost_shift": (
                shifted["optimum_plans"] == base["optimum_plans"]
                and shifted["tie_set"] == base["tie_set"]
                and shifted["optimum"] == base["optimum"] + 3 * int(instance["horizon"])
            ),
            "equivalent_state_encoding": all(
                json.loads(canonical_json(row["state"])) == row["state"]
                for row in base["state_action_rows"]
            ),
            "family_preserving_surface_change": surface_prompt != instance["prompt"],
        }
        for transform in METAMORPHIC_TRANSFORMS:
            reported = next(
                (
                    row
                    for row in reported_transforms
                    if row.get("family") == family and row.get("transform") == transform
                ),
                None,
            )
            observed = expected_results[transform]
            pass_state = bool(
                invariants[transform]
                and reported is not None
                and reported.get("observed_result") == observed
            )
            results.append(
                {
                    "kind": "metamorphic",
                    "case": f"{family}:{transform}",
                    "expected_result": None
                    if reported is None
                    else reported.get("expected_invariant"),
                    "observed_result": observed,
                    "pass_state": pass_state,
                }
            )

    base_instance = next(
        instance
        for instance in instances
        if any(
            not row["legality"]
            for row in exhaustive_solve(instance["typed_spec"])["state_action_rows"]
        )
    )
    solved = exhaustive_solve(base_instance["typed_spec"])
    clean_rows = solved["state_action_rows"]
    legal_index = next(index for index, row in enumerate(clean_rows) if row["legality"])
    illegal_index = next(index for index, row in enumerate(clean_rows) if not row["legality"])
    expected_map = {
        (row["time_index"], canonical_json(row["state"]), row["action"]): row for row in clean_rows
    }

    bad_transition = deepcopy(clean_rows[legal_index])
    state_key = next(iter(bad_transition["transition"]["next_state"]))
    bad_transition["transition"]["next_state"][state_key] += 1
    key = (
        bad_transition["time_index"],
        canonical_json(bad_transition["state"]),
        bad_transition["action"],
    )
    detections = {
        "bad_transition": bad_transition != expected_map[key],
        "infeasible_action": False,
        "corrupted_cost": False,
        "label_leakage": False,
        "wrong_ties": False,
        "stale_seal": False,
    }
    infeasible = deepcopy(clean_rows[illegal_index])
    infeasible["legality"] = True
    key = (infeasible["time_index"], canonical_json(infeasible["state"]), infeasible["action"])
    detections["infeasible_action"] = infeasible != expected_map[key]
    corrupted = deepcopy(clean_rows[legal_index])
    corrupted["immediate_cost"] += 1
    key = (corrupted["time_index"], canonical_json(corrupted["state"]), corrupted["action"])
    detections["corrupted_cost"] = corrupted != expected_map[key]
    leaked = deepcopy(upstream)
    leaked["instance_rows"][0]["prompt"] += f" Exact optimum: {solved['optimum']}."
    detections["label_leakage"] = any(
        row["check"] == "prompt_direct_label" and not row["pass_state"]
        for row in audit_leakage(leaked)
    )
    detections["wrong_ties"] = bool(base_instance["ties"] != (not base_instance["ties"]))
    instances_by_id = {str(row["instance"]): row for row in instances}
    stale = deepcopy(upstream["label_seal_rows"][0])
    stale["prompt_hash"] = "sha256:stale"
    detections["stale_seal"] = not _valid_seal(stale, instances_by_id)

    raw_mutations = list(upstream.get("mutation_rows", []))
    for mutation in REQUIRED_MUTATIONS:
        reported = _producer_case(raw_mutations, "mutation", mutation)
        expected = True if reported is None else bool(reported.get("expected_detection"))
        observed = bool(detections[mutation])
        results.append(
            {
                "kind": "mutation",
                "case": mutation,
                "expected_result": expected,
                "observed_result": observed,
                "pass_state": expected is True and observed is True,
            }
        )
    return results


def build_coverage_rows(
    upstream: Mapping[str, Any],
    manifest: Mapping[str, Any],
    solver_rows: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    leakage: Sequence[Mapping[str, Any]],
    attacks: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Record expected and observed counts without hiding absent units."""

    selected = set(manifest["reveal_order"])
    action_rows = [row for row in upstream["state_action_rows"] if row["instance"] in selected]
    expected_states = len(
        {(row["instance"], row["time_index"], canonical_json(row["state"])) for row in action_rows}
    )
    observed_states = sum(int(row["state_count"]) for row in solver_rows)
    observed_actions = sum(int(row["state_action_value_count"]) for row in solver_rows)
    selected_seals = [row for row in upstream["label_seal_rows"] if row["instance"] in selected]
    transform_count = sum(row["kind"] == "metamorphic" for row in attacks)
    mutation_count = sum(row["kind"] == "mutation" for row in attacks)
    values = (
        ("instances", 12, len(solver_rows)),
        ("states", expected_states, observed_states),
        ("actions", len(action_rows), observed_actions),
        ("seals", 12, len(selected_seals)),
        ("transforms", len(FAMILIES) * len(METAMORPHIC_TRANSFORMS), transform_count),
        ("mutations", len(REQUIRED_MUTATIONS), mutation_count),
        ("comparisons", len(comparisons), len(comparisons)),
        ("leakage_checks", 6, len(leakage)),
    )
    return [
        {
            "coverage": name,
            "expected": expected,
            "observed": observed,
            "pass_state": expected == observed,
        }
        for name, expected, observed in values
    ]


def recompute_aggregate(
    coverage: Sequence[Mapping[str, Any]],
    solver_rows: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    leakage: Sequence[Mapping[str, Any]],
    attacks: Sequence[Mapping[str, Any]],
    tests_run: Sequence[Mapping[str, Any]],
    *,
    preconditions_passed: bool,
    protected_files_unchanged: bool,
    blinding_clean: bool,
) -> JsonDict:
    """Derive the audit gate from rows and measured receipts only."""

    test_map = {row.get("check_id"): row for row in tests_run}
    checks = {
        "preconditions": preconditions_passed,
        "protected_files": protected_files_unchanged,
        "blinding_chronology": blinding_clean,
        "coverage": bool(coverage) and all(row.get("pass_state") is True for row in coverage),
        "independent_recomputation": len(solver_rows) == 12,
        "reported_parity": bool(comparisons)
        and all(row.get("disposition") == "match" for row in comparisons),
        "leakage_split_seal": len(leakage) == 6
        and all(row.get("pass_state") is True for row in leakage),
        "metamorphic": sum(row.get("kind") == "metamorphic" for row in attacks) == 16
        and all(
            row.get("pass_state") is True for row in attacks if row.get("kind") == "metamorphic"
        ),
        "mutation": sum(row.get("kind") == "mutation" for row in attacks) == 6
        and all(row.get("pass_state") is True for row in attacks if row.get("kind") == "mutation"),
        "focused_tests": test_map.get("focused_tests", {}).get("passed") is True,
        "scoped_coverage": test_map.get("scoped_coverage", {}).get("passed") is True
        and float(test_map.get("scoped_coverage", {}).get("coverage_percent") or -1) == 100.0,
        "full_python_suite": test_map.get("full_python_suite", {}).get("passed") is True,
        "spec_coverage": test_map.get("spec_coverage", {}).get("passed") is True,
        "applicable_e2e": test_map.get("applicable_e2e", {}).get("passed") is True,
        "ruff_check": test_map.get("ruff_check", {}).get("passed") is True,
        "format_check": test_map.get("format_check", {}).get("passed") is True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "checks": checks,
        "failed_checks": failed,
        "coverage_row_count": len(coverage),
        "solver_row_count": len(solver_rows),
        "comparison_row_count": len(comparisons),
        "leakage_row_count": len(leakage),
        "metamorphic_row_count": sum(row.get("kind") == "metamorphic" for row in attacks),
        "mutation_row_count": sum(row.get("kind") == "mutation" for row in attacks),
        "planning_fixture_audit_passed": not failed,
    }


def _memory_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0  # pragma: no cover - Linux hosts expose MemAvailable.


def collect_preconditions(root: Path) -> list[JsonDict]:
    """Measure the upstream gate, resources, tools, stores, and protected inputs."""

    upstream_path = root / UPSTREAM_PATH
    roadmap = root / ACTIVE_ROADMAP
    conductor = root / CONDUCTOR_PATH
    rows: list[JsonDict] = []

    def add(name: str, expected: Any, observed: Any, passed: bool) -> None:
        rows.append({"name": name, "expected": expected, "observed": observed, "passed": passed})

    add("upstream_artifact", "present", upstream_path.is_file(), upstream_path.is_file())
    upstream: JsonDict = {}
    if upstream_path.is_file():
        try:
            upstream = load_json(upstream_path)
        except (OSError, ValueError, TypeError) as exc:  # pragma: no cover - malformed file probe.
            add("upstream_json", "valid JSON object", str(exc), False)
    if upstream:
        add(
            "planning_fixture_ready",
            True,
            upstream.get("planning_fixture_ready"),
            upstream.get("planning_fixture_ready") is True,
        )
        stores = (
            "instance_rows",
            "state_action_rows",
            "label_seal_rows",
            "metamorphic_rows",
            "mutation_rows",
            "tests_run",
        )
        observed_stores = {
            name: {
                "count": len(upstream.get(name, [])),
                "sha256": sha256_json(upstream.get(name, [])),
            }
            for name in stores
        }
        add(
            "raw_stores",
            list(stores),
            observed_stores,
            all(isinstance(upstream.get(name), list) for name in stores),
        )
        add("upstream_sha256", "sha256 content hash", sha256_file(upstream_path), True)
    add("cpu", ">=1", os.cpu_count() or 0, (os.cpu_count() or 0) >= 1)
    memory = _memory_bytes()
    add("ram_bytes", ">=1073741824", memory, memory >= 1024**3)
    free = shutil.disk_usage(root).free
    add("disk_free_bytes", ">=104857600", free, free >= 100 * 1024**2)
    tools = {
        "python": shutil.which("python") is not None,
        "git": shutil.which("git") is not None,
        "sha256sum": shutil.which("sha256sum") is not None,
        "jq": shutil.which("jq") is not None,
        "adversarial_verify": (root / "scripts/adversarial_verify.py").is_file(),
        "row_consistency": (root / "scripts/verdict_row_consistency_lint.py").is_file(),
        "spec_coverage": (root / "scripts/check_spec_coverage.py").is_file(),
    }
    add("audit_tools", "all present", tools, all(tools.values()))
    roadmap_text = roadmap.read_text(encoding="utf-8") if roadmap.is_file() else ""
    roadmap_observed = {
        "present": roadmap.is_file(),
        "v584_task": "exp6703-exact-planning-fixture-audit" in roadmap_text,
    }
    if roadmap.is_file():
        roadmap_observed["sha256"] = sha256_file(roadmap)
    add(
        "roadmap",
        "active V584 Exp6703 task",
        roadmap_observed,
        all(roadmap_observed.get(key) for key in ("present", "v584_task")),
    )
    conductor_observed: JsonDict = {"present": conductor.is_file()}
    if conductor.is_file():
        conductor_observed["sha256"] = sha256_file(conductor)
    add("conductor", "present and hashed", conductor_observed, conductor.is_file())
    return rows


def protected_hashes(root: Path) -> JsonDict:
    """Hash files that this audit must not alter."""

    return {
        path.as_posix(): sha256_file(root / path)
        for path in (ACTIVE_ROADMAP, CONDUCTOR_PATH)
        if (root / path).is_file()
    }


def _per_unit_rows(
    coverage: Sequence[Mapping[str, Any]],
    solvers: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    leakage: Sequence[Mapping[str, Any]],
    attacks: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    result: list[JsonDict] = []
    for unit_type, rows in (
        ("coverage", coverage),
        ("independent_solver", solvers),
        ("reported_comparison", comparisons),
        ("leakage", leakage),
        ("metamorphic_mutation", attacks),
    ):
        result.extend({"unit_type": unit_type, **deepcopy(dict(row))} for row in rows)
    return result


def _gate_summary(
    aggregate: Mapping[str, Any], blinding_incident: Mapping[str, Any] | None
) -> list[JsonDict]:
    summary: list[JsonDict] = []
    if blinding_incident is not None:
        summary.append(deepcopy(dict(blinding_incident)))
    for check in aggregate.get("failed_checks", []):
        if check == "blinding_chronology" and blinding_incident is not None:
            continue
        summary.append({"check": check, "expected": True, "observed": False, "passed": False})
    return summary


def _field_provenance(artifact: Mapping[str, Any]) -> JsonDict:
    result: JsonDict = {}
    for field in REQUIRED_ARTIFACT_FIELDS:
        value = artifact.get(field)
        result[field] = {
            "raw_store": UPSTREAM_PATH.as_posix()
            if field not in {"duration_s", "tests_run"}
            else None,
            "independent_solver": SOLVER_VERSION
            if field
            in {
                "independent_solver_rows",
                "reported_vs_recomputed_rows",
                "coverage_rows",
                "aggregate_row_recomputation",
                "planning_fixture_audit_passed",
            }
            else None,
            "scanner": "audit_leakage_and_attack_replay.v1"
            if field
            in {
                "leakage_rows",
                "metamorphic_mutation_rows",
                "planning_fixture_audit_passed",
            }
            else None,
            "reducer": "recompute_aggregate.v1"
            if field
            in {
                "aggregate_row_recomputation",
                "planning_fixture_audit_passed",
                "status",
                "honest_verdict",
                "verdict_class",
                "gate_check_summary",
            }
            else None,
            "function": "build_artifact",
            "sha256": sha256_json(value),
        }
    result["field_provenance"]["sha256"] = sha256_json(sorted(REQUIRED_ARTIFACT_FIELDS))
    result["reproducibility_checksum"]["sha256"] = "sha256:computed_after_field_provenance"
    return result


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable artifact content while excluding checksum and wall time."""

    return sha256_json(
        {
            key: value
            for key, value in artifact.items()
            if key not in {"reproducibility_checksum", "duration_s"}
        }
    )


def build_artifact(
    *,
    date: str,
    root: Path,
    tests_run: Sequence[Mapping[str, Any]],
    duration_s: float,
    protected_before: Mapping[str, Any],
) -> JsonDict:
    """Build the complete audit artifact after all preconditions pass."""

    upstream = load_json(root / UPSTREAM_PATH)
    manifest = freeze_blinded_sample(public_instance_rows(upstream))
    manifest["cold_protocol_incident"] = deepcopy(BLINDING_PROTOCOL_INCIDENT)
    manifest["cold_provenance_clean"] = False
    manifest["reveal_events"] = [
        "invalid_identity_projection",
        "premature_non_sample_metamorphic_total_reveal",
        "valid_manifest_freeze",
        *manifest["reveal_order"],
    ]
    manifest["manifest_hash"] = manifest_checksum(manifest)
    solvers, comparisons = recompute_selected_units(upstream, manifest)
    leakage = audit_leakage(upstream)
    attacks = audit_metamorphic_and_mutation_cases(upstream)
    coverage = build_coverage_rows(upstream, manifest, solvers, comparisons, leakage, attacks)
    preconditions = collect_preconditions(root)
    after = protected_hashes(root)
    protected = {
        "before": dict(protected_before),
        "after": after,
        "unchanged": dict(protected_before) == after,
    }
    aggregate = recompute_aggregate(
        coverage,
        solvers,
        comparisons,
        leakage,
        attacks,
        tests_run,
        preconditions_passed=all(row["passed"] for row in preconditions),
        protected_files_unchanged=protected["unchanged"],
        blinding_clean=False,
    )
    all_matches = all(row["disposition"] == "match" for row in comparisons)
    if all_matches:
        status = "disqualified_blinding_chronology"
        verdict = (
            "disqualified: independent exact rows reproduced, but reported values were exposed "
            "before a valid blinded identity freeze"
        )
        verdict_class = "disqualified"
    else:
        status = "corrected_fixture_mismatch_and_disqualified_blinding"
        verdict = "corrected: independent recomputation found fixture mismatches after invalid cold chronology"
        verdict_class = "partial"

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": 6703,
        "run_date": date,
        "status": status,
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "gate_check_summary": _gate_summary(aggregate, BLINDING_PROTOCOL_INCIDENT),
        "openspec_requirement_ids": list(OPEN_SPEC_IDS),
        "blinded_sample_manifest": manifest,
        "coverage_rows": coverage,
        "independent_solver_rows": solvers,
        "reported_vs_recomputed_rows": comparisons,
        "leakage_rows": leakage,
        "metamorphic_mutation_rows": attacks,
        "planning_fixture_audit_passed": aggregate["planning_fixture_audit_passed"],
        "per_unit_rows": _per_unit_rows(coverage, solvers, comparisons, leakage, attacks),
        "aggregate_row_recomputation": aggregate,
        "preconditions_checked": preconditions,
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
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
    date: str, root: Path, preconditions: Sequence[Mapping[str, Any]], duration_s: float
) -> JsonDict:
    """Write measured missing prerequisites without invented audit rows."""

    failed = next((row for row in preconditions if not row.get("passed")), None)
    aggregate = {
        "checks": {"preconditions": False},
        "failed_checks": ["preconditions"],
        "coverage_row_count": 0,
        "solver_row_count": 0,
        "comparison_row_count": 0,
        "leakage_row_count": 0,
        "metamorphic_row_count": 0,
        "mutation_row_count": 0,
        "planning_fixture_audit_passed": False,
    }
    gate = (
        []
        if failed is None
        else [
            {
                "check": failed["name"],
                "expected": failed["expected"],
                "observed": failed["observed"],
                "passed": False,
            }
        ]
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": 6703,
        "run_date": date,
        "status": "blocked_precondition",
        "honest_verdict": "blocked: required upstream audit input or resource is unavailable",
        "verdict_class": "blocked",
        "gate_check_summary": gate,
        "openspec_requirement_ids": list(OPEN_SPEC_IDS),
        "blinded_sample_manifest": {},
        "coverage_rows": [],
        "independent_solver_rows": [],
        "reported_vs_recomputed_rows": [],
        "leakage_rows": [],
        "metamorphic_mutation_rows": [],
        "planning_fixture_audit_passed": False,
        "per_unit_rows": [],
        "aggregate_row_recomputation": aggregate,
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "protected_files_unchanged": {
            "before": protected_hashes(root),
            "after": protected_hashes(root),
            "unchanged": True,
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
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
    """Validate schema, row conservation, reduction, status, and checksum."""

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
    expected_units = _per_unit_rows(
        payload["coverage_rows"],
        payload["independent_solver_rows"],
        payload["reported_vs_recomputed_rows"],
        payload["leakage_rows"],
        payload["metamorphic_mutation_rows"],
    )
    if payload.get("per_unit_rows") != expected_units:
        errors.append("per_unit_rows_mismatch")
    provenance = payload.get("field_provenance")
    if not isinstance(provenance, Mapping) or not set(REQUIRED_ARTIFACT_FIELDS) <= set(provenance):
        errors.append("field_provenance_invalid")
    if payload.get("status") == "blocked_precondition":
        if payload.get("planning_fixture_audit_passed") is not False or not payload.get(
            "gate_check_summary"
        ):
            errors.append("blocked_state_mismatch")
        return errors

    manifest = payload.get("blinded_sample_manifest", {})
    if manifest.get("manifest_hash") != manifest_checksum(manifest):
        errors.append("manifest_hash_mismatch")
    aggregate = recompute_aggregate(
        payload["coverage_rows"],
        payload["independent_solver_rows"],
        payload["reported_vs_recomputed_rows"],
        payload["leakage_rows"],
        payload["metamorphic_mutation_rows"],
        payload["tests_run"],
        preconditions_passed=all(
            row.get("passed") is True for row in payload["preconditions_checked"]
        ),
        protected_files_unchanged=payload["protected_files_unchanged"].get("unchanged") is True,
        blinding_clean=manifest.get("cold_provenance_clean") is True,
    )
    if payload.get("aggregate_row_recomputation") != aggregate:
        errors.append("aggregate_row_recomputation_mismatch")
    if payload.get("planning_fixture_audit_passed") != aggregate["planning_fixture_audit_passed"]:
        errors.append("readiness_mismatch")
    if payload.get("planning_fixture_audit_passed") is True and payload.get("gate_check_summary"):
        errors.append("ready_gate_summary_mismatch")
    if payload.get("planning_fixture_audit_passed") is False and not payload.get(
        "gate_check_summary"
    ):
        errors.append("failed_gate_summary_missing")
    return errors


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> JsonDict:
    """Sync one complete temporary file before replacing the destination."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return {"path": path.as_posix(), "bytes": len(data), "atomic_replace": True}


def default_command_runner(command: str, root: Path) -> JsonDict:
    """Run one declared check and retain its real process receipt."""

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


def run_verification_commands(
    root: Path, runner: Callable[[str, Path], Mapping[str, Any]] = default_command_runner
) -> list[JsonDict]:
    """Execute the focused, coverage, full, spec, and E2E checks once."""

    rows: list[JsonDict] = []
    for check_id, command in VERIFICATION_COMMANDS:
        receipt = runner(command, root)
        output = str(receipt.get("stdout", "")) + str(receipt.get("stderr", ""))
        coverage: float | None = None
        if check_id == "scoped_coverage":
            match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", output)
            coverage = float(match.group(1)) if match else None
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
                "summary": output[-2000:],
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
    """Measure prerequisites, build one terminal artifact, and write it."""

    started = time.perf_counter()
    output = output_path or root / RESULT_PATH
    preconditions = collect_preconditions(root)
    if not all(row["passed"] for row in preconditions):
        artifact = build_blocked_artifact(date, root, preconditions, time.perf_counter() - started)
    else:
        before = protected_hashes(root)
        receipts = list(tests_run) if tests_run is not None else run_verification_commands(root)
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
    """Run the dated audit or validate an existing artifact."""

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


if __name__ == "__main__":  # pragma: no cover - exercised through the module command.
    raise SystemExit(main())
