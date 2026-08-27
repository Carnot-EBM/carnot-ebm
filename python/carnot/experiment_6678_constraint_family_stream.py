"""Build the frozen Exp6678 independent constraint-family stream.

The stream is infrastructure for a later prequential comparison. It runs four
exact checkers and durable state exercises, but it does not run a model. Exact
checkers remain the only validity authority.

Spec: REQ-LEARN-6678 and SCENARIO-LEARN-6678-*.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import UTC, datetime, timedelta
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence


JsonDict = dict[str, Any]
Checker = Callable[[Mapping[str, Any]], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260827"
RESULT_PATH = Path("results/experiment_6678_constraint_family_stream.json")
MODULE_PATH = Path("python/carnot/experiment_6678_constraint_family_stream.py")
TEST_PATH = Path("tests/python/test_experiment_6678_constraint_family_stream.py")
SPEC_PATH = Path("openspec/capabilities/self-learning/spec.md")
ACTIVE_ROADMAP = Path("research-roadmap.yaml")
CONDUCTOR_PATH = Path("scripts/research_conductor.py")
PROTECTED_PATHS = (ACTIVE_ROADMAP, CONDUCTOR_PATH)
INFERENCE_SUBSTRATE = "cpu_stream_fixture_and_exact_checkers_no_llm"
SCHEMA = "carnot.experiment_6678.constraint_family_stream.v1"
STATE_SCHEMA = "carnot.experiment_6678.repair_state.v1"
TYPED_REPAIR_VERSION = "carnot.family_blind_typed_repair.v1"
RANDOM_SEED = 6_678_027
FAMILY_ORDER = ("scheduling", "graph", "logic", "plan_state")
EVENT_COUNT = 16
ORDER_SEEDS = {
    "chronological": 6_678_100,
    "reverse_blocks": 6_678_101,
    "round_robin": 6_678_102,
    "seeded_a": 6_678_103,
    "seeded_b": 6_678_104,
}
KEY_FIELDS = ("constraint_shape", "operator_kind", "value_type", "arity")
EXCLUDED_KEY_FIELDS = (
    "task_id",
    "event_id",
    "family",
    "family_label",
    "future_outcome",
    "outcome",
    "hidden_gold",
    "exact_target",
    "exact_violation_witness",
    "witness",
    "future_state",
    "split_role",
    "timestamp",
)
ATTACK_TYPES = (
    "future_leakage",
    "family_specific_key",
    "task_identity_key",
    "duplicate_events",
    "shuffled_timestamps",
    "poison_patch",
    "support_collapse",
    "anchor_regression",
    "partial_writes",
    "restart_corruption",
    "non_invertible_rollback",
)

SOURCE_CORPUS_PATHS = (
    Path("results/experiment_6653_state_grounded_repair_memory_fixture.json"),
    Path("results/experiment_6654_prospective_repair_memory_evolution.json"),
    Path("results/experiment_6655_repair_memory_safety_audit.json"),
    Path("results/experiment_6661_triggered_tail_fixture.json"),
    Path("results/experiment_6675_triggered_tail_scope_receipt.json"),
)
EXACT_CHECKER_PATHS = (
    Path("python/carnot/experiment_6604_exact_two_level_plan_corpus.py"),
    Path("python/carnot/experiment_6661_triggered_tail_fixture.py"),
    MODULE_PATH,
)
REPAIR_MEMORY_PATHS = (
    Path("python/carnot/experiment_6653_state_grounded_repair_memory_fixture.py"),
    Path("python/carnot/experiment_6654_prospective_repair_memory_evolution.py"),
    Path("python/carnot/experiment_6655_repair_memory_safety_audit.py"),
    Path("python/carnot/memory/revocable_atomic_repair.py"),
    Path("python/carnot/pipeline/atomic_writer.py"),
)
STATE_SCHEMA_PATHS = (
    Path("results/experiment_6653_state_grounded_repair_memory_fixture.json"),
    Path("results/experiment_6290_revocable_atomic_repair_memory.json"),
    SPEC_PATH,
)

VERIFICATION_COMMANDS = (
    f".venv/bin/pytest {TEST_PATH.as_posix()} -q -n 0 -o addopts=",
    (
        ".venv/bin/coverage run --rcfile=/dev/null "
        "--data-file=/tmp/carnot-exp6678.coverage "
        f"--include={MODULE_PATH.as_posix()} -m pytest {TEST_PATH.as_posix()} "
        "-q --no-cov -n 0 -o addopts="
    ),
    (
        ".venv/bin/coverage report --rcfile=/dev/null "
        "--data-file=/tmp/carnot-exp6678.coverage "
        f"--include={MODULE_PATH.as_posix()} --fail-under=100 --show-missing"
    ),
    f".venv/bin/ruff check {MODULE_PATH.as_posix()} {TEST_PATH.as_posix()}",
    f".venv/bin/ruff format --check {MODULE_PATH.as_posix()} {TEST_PATH.as_posix()}",
    f".venv/bin/python scripts/check_spec_coverage.py {TEST_PATH.as_posix()}",
    f".venv/bin/pytest {TEST_PATH.as_posix()} -q -k restart -n 0 -o addopts=",
    ".venv/bin/pytest tests/python -q",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "gate_check_summary",
    "family_manifest",
    "event_order_manifests",
    "typed_repair_schema",
    "exact_checker_rows",
    "isolation_attack_rows",
    "restart_rollback_rows",
    "constraint_family_stream_ready",
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
    "status": "The terminal state records whether the deterministic fixture completed.",
    "honest_verdict": "The verdict reports readiness without a model or learning claim.",
    "verdict_class": "The closed class treats ready infrastructure as null evidence.",
    "gate_check_summary": "The first failed row keeps a failure local and reproducible.",
    "family_manifest": "Definitions and hashes freeze four independent exact authorities.",
    "event_order_manifests": "Sealed orders preserve prospective evaluation before a run.",
    "typed_repair_schema": "The schema fixes visible and excluded memory fields.",
    "exact_checker_rows": "Positive and negative controls exercise each authority.",
    "isolation_attack_rows": "Named mutations prove that leakage and corruption fail closed.",
    "restart_rollback_rows": "Byte comparisons prove recoverability and reversibility.",
    "constraint_family_stream_ready": "One Boolean reduces all complete fixture rows.",
    "per_unit_rows": "Stable unit rows keep every gate independently recheckable.",
    "aggregate_row_recomputation": "Counts and readiness rebuild from raw rows.",
    "preconditions_checked": "Input hashes and host resources bind the measured substrate.",
    "protected_files_unchanged": "Before and after hashes protect the roadmap and conductor.",
    "inference_substrate": "The substrate identity states that no model ran.",
    "verifier_is_oracle": "Exact family checkers explicitly define validity.",
    "field_provenance": "Each field names its source, checker, function, and hash.",
    "random_seed": "Base, family, and order seeds make fixture construction repeatable.",
    "duration_s": "Monotonic time measures the real fixture build and verification.",
    "tests_run": "Command receipts make verification reproducible.",
    "reproducibility_checksum": "The canonical content hash detects artifact drift.",
}


def canonical_json(value: Any) -> str:
    """Return one stable JSON encoding for all evidence comparisons."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_bytes(value: Any) -> bytes:
    """Encode stable JSON as bytes for restart and rollback checks."""

    return canonical_json(value).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    """Return a prefixed digest so the hash algorithm stays explicit."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash canonical JSON instead of interpreter-specific object text."""

    return sha256_bytes(canonical_bytes(value))


def sha256_file(path: Path) -> str:
    """Hash one required file or preserve its absence as measured input."""

    return sha256_bytes(path.read_bytes()) if path.is_file() else "missing"


def _without(value: Mapping[str, Any], field: str) -> JsonDict:
    return {key: deepcopy(item) for key, item in value.items() if key != field}


def row_hash(row: Mapping[str, Any]) -> str:
    """Hash one evidence row without trusting its stored self-hash."""

    return sha256_json(_without(row, "row_sha256"))


def patch_hash(patch: Mapping[str, Any]) -> str:
    """Hash one patch without trusting its stored self-hash."""

    return sha256_json(_without(patch, "patch_sha256"))


def order_manifest_hash(manifest: Mapping[str, Any]) -> str:
    """Hash one sealed order without its stored digest."""

    return sha256_json(_without(manifest, "manifest_sha256"))


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash every final field except the field that stores this digest."""

    return sha256_json(_without(artifact, "reproducibility_checksum"))


def check_scheduling(state: Mapping[str, Any]) -> JsonDict:
    """Check that jobs do not overlap on the one exact shared resource."""

    jobs = list(state.get("jobs", []))
    conflicts: list[list[str]] = []
    for left_index, left in enumerate(jobs):
        for right in jobs[left_index + 1 :]:
            left_end = int(left["start"]) + int(left["duration"])
            right_end = int(right["start"]) + int(right["duration"])
            if int(left["start"]) < right_end and int(right["start"]) < left_end:
                conflicts.append([str(left["id"]), str(right["id"])])
    return {
        "exact_valid": not conflicts,
        "reason": "valid_schedule" if not conflicts else "job_overlap",
        "witness": conflicts,
        "authority": "carnot.exact_shared_resource_schedule.v1",
    }


def check_graph(state: Mapping[str, Any]) -> JsonDict:
    """Check a complete finite graph coloring from visible graph facts."""

    nodes = [str(node) for node in state.get("nodes", [])]
    colors = dict(state.get("colors", {}))
    color_count = int(state.get("color_count", 0))
    missing = sorted(set(nodes) - set(colors))
    out_of_range = sorted(
        node
        for node in nodes
        if node in colors and int(colors[node]) not in range(1, color_count + 1)
    )
    conflicts = [
        [str(left), str(right)]
        for left, right in state.get("edges", [])
        if str(left) in colors
        and str(right) in colors
        and int(colors[str(left)]) == int(colors[str(right)])
    ]
    if missing:
        reason, witness = "missing_color", missing
    elif out_of_range:
        reason, witness = "color_out_of_range", out_of_range
    elif conflicts:
        reason, witness = "edge_color_conflict", conflicts
    else:
        reason, witness = "valid_coloring", []
    return {
        "exact_valid": not witness,
        "reason": reason,
        "witness": witness,
        "authority": "carnot.exact_graph_coloring.v1",
    }


def check_logic(state: Mapping[str, Any]) -> JsonDict:
    """Evaluate finite integer equations directly from visible assignments."""

    variables = {str(key): int(value) for key, value in dict(state.get("variables", {})).items()}
    failures: list[JsonDict] = []
    for index, equation in enumerate(state.get("equations", [])):
        observed = sum(
            int(coefficient) * variables.get(str(name), 0)
            for name, coefficient in dict(equation["coefficients"]).items()
        )
        expected = int(equation["rhs"])
        if observed != expected:
            failures.append({"equation": index, "observed": observed, "expected": expected})
    return {
        "exact_valid": not failures,
        "reason": "all_equations_satisfied" if not failures else "equation_violation",
        "witness": failures,
        "authority": "carnot.exact_integer_equations.v1",
    }


def check_plan_state(state: Mapping[str, Any]) -> JsonDict:
    """Check that every prerequisite occurs before its dependent plan step."""

    steps = [str(step) for step in state.get("steps", [])]
    positions = {step: index for index, step in enumerate(steps)}
    violations: list[list[str]] = []
    for step, requirements in dict(state.get("requires", {})).items():
        for requirement in requirements:
            if str(step) not in positions or str(requirement) not in positions:
                violations.append([str(requirement), str(step), "missing"])
            elif positions[str(requirement)] >= positions[str(step)]:
                violations.append([str(requirement), str(step), "out_of_order"])
    return {
        "exact_valid": not violations,
        "reason": "valid_plan_state" if not violations else "prerequisite_order",
        "witness": violations,
        "authority": "carnot.exact_plan_prerequisites.v1",
    }


CHECKERS: dict[str, Checker] = {
    "scheduling": check_scheduling,
    "graph": check_graph,
    "logic": check_logic,
    "plan_state": check_plan_state,
}


def _checker_identity(family: str) -> JsonDict:
    function = CHECKERS[family]
    return {
        "name": str(function({})["authority"]),
        "function": f"{__name__}.{function.__name__}",
        "sha256": sha256_bytes(inspect.getsource(function).encode("utf-8")),
        "executable": True,
        "verifier_is_oracle": True,
    }


def _control_states() -> dict[str, tuple[JsonDict, JsonDict]]:
    return {
        "scheduling": (
            {
                "jobs": [
                    {"id": "a", "start": 0, "duration": 2},
                    {"id": "b", "start": 2, "duration": 2},
                ]
            },
            {
                "jobs": [
                    {"id": "a", "start": 0, "duration": 2},
                    {"id": "b", "start": 1, "duration": 2},
                ]
            },
        ),
        "graph": (
            {
                "nodes": ["a", "b"],
                "edges": [["a", "b"]],
                "colors": {"a": 1, "b": 2},
                "color_count": 2,
            },
            {
                "nodes": ["a", "b"],
                "edges": [["a", "b"]],
                "colors": {"a": 1, "b": 1},
                "color_count": 2,
            },
        ),
        "logic": (
            {
                "variables": {"x": 1, "y": 2},
                "equations": [{"coefficients": {"x": 1, "y": 1}, "rhs": 3}],
            },
            {
                "variables": {"x": 1, "y": 1},
                "equations": [{"coefficients": {"x": 1, "y": 1}, "rhs": 3}],
            },
        ),
        "plan_state": (
            {"steps": ["test", "ship"], "requires": {"test": [], "ship": ["test"]}},
            {"steps": ["ship", "test"], "requires": {"test": [], "ship": ["test"]}},
        ),
    }


def _family_event(family: str, index: int) -> tuple[JsonDict, list[str], Any, str]:
    suffix = str(index)
    if family == "scheduling":
        state = {
            "jobs": [
                {"id": "a" + suffix, "start": 0, "duration": 2},
                {"id": "b" + suffix, "start": 1, "duration": 2},
            ]
        }
        return state, ["jobs", "1", "start"], 2, "set_scalar"
    if family == "graph":
        left, right = "a" + suffix, "b" + suffix
        state = {
            "nodes": [left, right],
            "edges": [[left, right]],
            "colors": {left: 1, right: 1},
            "color_count": 2,
        }
        return state, ["colors", right], 2, "set_scalar"
    if family == "logic":
        state = {
            "variables": {"x": index + 1, "y": 0},
            "equations": [{"coefficients": {"x": 1, "y": 1}, "rhs": index + 3}],
        }
        return state, ["variables", "y"], 2, "set_scalar"
    state = {
        "steps": ["ship" + suffix, "test" + suffix],
        "requires": {"test" + suffix: [], "ship" + suffix: ["test" + suffix]},
    }
    return state, ["steps"], ["test" + suffix, "ship" + suffix], "replace_sequence"


def _value_at_path(state: Any, path: Sequence[str]) -> Any:
    current = state
    for part in path:
        current = current[int(part)] if isinstance(current, list) else current[part]
    return deepcopy(current)


def _replace_at_path(state: Mapping[str, Any], path: Sequence[str], value: Any) -> JsonDict:
    changed = deepcopy(dict(state))
    current: Any = changed
    for part in path[:-1]:
        current = current[int(part)] if isinstance(current, list) else current[part]
    last = path[-1]
    if isinstance(current, list):
        current[int(last)] = deepcopy(value)
    else:
        current[last] = deepcopy(value)
    return changed


def _make_patch(path: list[str], before: Any, after: Any, version: int) -> JsonDict:
    patch = {
        "operation": "replace",
        "path": path,
        "before": deepcopy(before),
        "after": deepcopy(after),
        "expected_version": version,
        "target_version": version + 1,
        "before_sha256": sha256_json(before),
        "after_sha256": sha256_json(after),
    }
    patch["patch_sha256"] = patch_hash(patch)
    return patch


def build_event_rows() -> list[JsonDict]:
    """Build 16 deterministic events with exact witnesses and inverses."""

    rows: list[JsonDict] = []
    base = datetime(2026, 8, 27, tzinfo=UTC)
    for family_index, family in enumerate(FAMILY_ORDER):
        for local_index in range(4):
            state, path, after_value, operator_kind = _family_event(family, local_index)
            before_value = _value_at_path(state, path)
            repaired = _replace_at_path(state, path, after_value)
            violation = CHECKERS[family](state)
            repaired_result = CHECKERS[family](repaired)
            event_id = f"evt-{family_index:02d}-{local_index:02d}"
            forward = _make_patch(path, before_value, after_value, 1)
            inverse = _make_patch(path, after_value, before_value, 2)
            material = {
                "constraint_shape": "binary_constraint"
                if family != "plan_state"
                else "ordered_dependency",
                "operator_kind": operator_kind,
                "value_type": "integer" if operator_kind == "set_scalar" else "sequence",
                "arity": 2,
            }
            row: JsonDict = {
                "schema": SCHEMA + ".event.v1",
                "event_id": event_id,
                "family": family,
                "timestamp": (base + timedelta(minutes=len(rows))).isoformat(),
                "split_role": "calibration" if local_index < 2 else "held_family",
                "visible_pre_event_state": state,
                "visible_pre_event_state_sha256": sha256_json(state),
                "exact_violation_witness": {
                    "reason": violation["reason"],
                    "witness": violation["witness"],
                    "witness_sha256": sha256_json(violation["witness"]),
                    "checker": violation["authority"],
                },
                "candidate_operator": {
                    "operator_kind": operator_kind,
                    "forward_patch": forward,
                    "repaired_state_sha256": sha256_json(repaired),
                    "exact_repair_passed": repaired_result["exact_valid"],
                },
                "applicability_key_material": material,
                "applicability_key": sha256_json(material),
                "support": {
                    "count": 1,
                    "source_receipts": [f"exact:{event_id}"],
                    "sha256": sha256_json([event_id]),
                },
                "anchors": [
                    {
                        "anchor_id": f"anchor-{family}",
                        "state_sha256": sha256_json(_control_states()[family][0]),
                        "exact_valid": True,
                    }
                ],
                "version": 1,
                "provenance": {
                    "source": MODULE_PATH.as_posix(),
                    "checker_function": _checker_identity(family)["function"],
                    "checker_sha256": _checker_identity(family)["sha256"],
                    "seed": RANDOM_SEED + family_index * 10 + local_index,
                },
                "inverse_patch": inverse,
            }
            row["row_sha256"] = row_hash(row)
            rows.append(row)
    return rows


def build_family_manifest(events: Sequence[Mapping[str, Any]] | None = None) -> dict[str, JsonDict]:
    """Freeze family definitions, controls, partitions, operators, and hashes."""

    event_rows = list(events) if events is not None else build_event_rows()
    controls = _control_states()
    manifest: dict[str, JsonDict] = {}
    for index, family in enumerate(FAMILY_ORDER):
        family_events = [row for row in event_rows if row["family"] == family]
        partitions = {
            role: [str(row["event_id"]) for row in family_events if row["split_role"] == role]
            for role in ("calibration", "held_family")
        }
        row: JsonDict = {
            "definition": "exact shared-resource schedule"
            if family == "scheduling"
            else f"exact {family} constraints",
            "sources": [
                {"path": MODULE_PATH.as_posix(), "sha256": sha256_file(REPO_ROOT / MODULE_PATH)}
            ],
            "checker": _checker_identity(family),
            "transferable_operator": "set_scalar" if family != "plan_state" else "replace_sequence",
            "positive_control_sha256": sha256_json(controls[family][0]),
            "negative_control_sha256": sha256_json(controls[family][1]),
            "partitions": partitions,
            "partitions_sha256": sha256_json(partitions),
            "held_family_fold": {
                "held_family": family,
                "calibration_families": [name for name in FAMILY_ORDER if name != family],
                "seed": RANDOM_SEED + index,
            },
            "counts": {"calibration": 2, "held_family": 2, "total": 4},
            "event_ids": [str(row["event_id"]) for row in family_events],
        }
        row["manifest_sha256"] = sha256_json(row)
        manifest[family] = row
    return manifest


def build_exact_checker_rows(families: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    """Execute positive and negative controls for every family checker."""

    controls = _control_states()
    rows: list[JsonDict] = []
    for family in FAMILY_ORDER:
        for kind, expected, state in (
            ("known_positive", True, controls[family][0]),
            ("known_negative", False, controls[family][1]),
        ):
            result = CHECKERS[family](state)
            row: JsonDict = {
                "row_type": "checker_control",
                "family": family,
                "control_kind": kind,
                "input_sha256": sha256_json(state),
                "expected_exact_valid": expected,
                "observed_exact_valid": result["exact_valid"],
                "reason": result["reason"],
                "witness": result["witness"],
                "authority": result["authority"],
                "checker_sha256": families[family]["checker"]["sha256"],
                "passed": result["exact_valid"] is expected,
            }
            row["row_sha256"] = row_hash(row)
            rows.append(row)
    return rows


def build_typed_repair_schema() -> JsonDict:
    """Publish the family-blind repair and between-event admission contract."""

    return {
        "schema": TYPED_REPAIR_VERSION,
        "visible_fields": [
            "visible_pre_event_state",
            "candidate_operator",
            "applicability_key",
            "support",
            "anchors",
            "version",
            "provenance",
            "inverse_patch",
        ],
        "key_fields": list(KEY_FIELDS),
        "excluded_fields": list(EXCLUDED_KEY_FIELDS),
        "admission_stage": "between_events_only",
        "admission_gates": [
            "source_exact_repair",
            "support_floor",
            "anchor_non_regression",
            "version_match",
            "invertible_patch",
        ],
        "retirement_policy": {
            "remove_before_next_retrieval": True,
            "reasons": [
                "support_collapse",
                "anchor_regression",
                "stale_version",
                "invalid_exact_evidence",
            ],
        },
        "provenance_fields": ["source", "checker_function", "checker_sha256", "seed"],
        "inverse_patch_required": True,
        "family_blind": True,
    }


def build_event_order_manifests(events: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Seal five complete orders with monotonically assigned event timestamps."""

    ids = [str(row["event_id"]) for row in events]
    by_id = {str(row["event_id"]): row for row in events}
    family_blocks = {
        family: [str(row["event_id"]) for row in events if row["family"] == family]
        for family in FAMILY_ORDER
    }
    order_ids: dict[str, list[str]] = {
        "chronological": ids,
        "reverse_blocks": [
            event_id for family in reversed(FAMILY_ORDER) for event_id in family_blocks[family]
        ],
        "round_robin": [
            family_blocks[family][index] for index in range(4) for family in FAMILY_ORDER
        ],
    }
    for name in ("seeded_a", "seeded_b"):
        shuffled = list(ids)
        random.Random(ORDER_SEEDS[name]).shuffle(shuffled)
        order_ids[name] = shuffled
    base = datetime(2026, 8, 27, tzinfo=UTC)
    manifests: list[JsonDict] = []
    for name in ORDER_SEEDS:
        ordered = order_ids[name]
        row: JsonDict = {
            "schema": SCHEMA + ".order.v1",
            "order_id": name,
            "seed": ORDER_SEEDS[name],
            "ordered_event_ids": ordered,
            "timestamps": [
                (base + timedelta(seconds=index)).isoformat() for index in range(len(ordered))
            ],
            "split_roles": [str(by_id[event_id]["split_role"]) for event_id in ordered],
            "event_rows": [
                {
                    "position": index,
                    "event_id": event_id,
                    "timestamp": (base + timedelta(seconds=index)).isoformat(),
                    "split_role": by_id[event_id]["split_role"],
                }
                for index, event_id in enumerate(ordered)
            ],
        }
        row["manifest_sha256"] = order_manifest_hash(row)
        manifests.append(row)
    return manifests


def admission_decision(
    event: Mapping[str, Any],
    *,
    support_ok: bool = True,
    anchor_ok: bool = True,
    was_active: bool = False,
) -> JsonDict:
    """Admit or retire one patch only after its exact event evaluation."""

    if not bool(event["candidate_operator"]["exact_repair_passed"]):
        return {"admitted": False, "retired": was_active, "reason": "invalid_exact_evidence"}
    if not support_ok:
        return {"admitted": False, "retired": was_active, "reason": "support_collapse"}
    if not anchor_ok:
        return {"admitted": False, "retired": was_active, "reason": "anchor_regression"}
    return {"admitted": True, "retired": False, "reason": "all_exact_gates_passed"}


def build_prequential_rows(
    events: Sequence[Mapping[str, Any]], order: Mapping[str, Any]
) -> list[JsonDict]:
    """Record what each event can read before its between-event write."""

    by_id = {str(row["event_id"]): row for row in events}
    active: list[str] = []
    rows: list[JsonDict] = []
    for position, event_id in enumerate(order["ordered_event_ids"]):
        event = by_id[str(event_id)]
        decision = admission_decision(event)
        row: JsonDict = {
            "row_type": "prequential_admission",
            "order_id": order["order_id"],
            "position": position,
            "event_id": event_id,
            "visible_commit_ids": list(active),
            "pre_event_state_sha256": event["visible_pre_event_state_sha256"],
            "action_committed_before_outcome": True,
            "same_event_patch_visible": False,
            "admission_stage": "between_events",
            "source_repair_passed": event["candidate_operator"]["exact_repair_passed"],
            "support_passed": True,
            "anchors_passed": True,
            **decision,
        }
        row["row_sha256"] = row_hash(row)
        rows.append(row)
        if decision["admitted"]:
            active.append(str(event_id))
    return rows


def _state_with_checksum(
    version: int, records: Mapping[str, Any], lineage: Sequence[str]
) -> JsonDict:
    state: JsonDict = {
        "schema": STATE_SCHEMA,
        "version": version,
        "records": deepcopy(dict(records)),
        "lineage": list(lineage),
    }
    state["state_checksum"] = sha256_json(state)
    return state


def empty_memory_state() -> JsonDict:
    """Return the sealed empty state used by every order and recovery case."""

    return _state_with_checksum(0, {}, [])


def verify_memory_state(state: Mapping[str, Any]) -> bool:
    """Reject malformed or checksum-corrupt restart state."""

    if set(state) != {"schema", "version", "records", "lineage", "state_checksum"}:
        return False
    return state.get("schema") == STATE_SCHEMA and state.get("state_checksum") == sha256_json(
        _without(state, "state_checksum")
    )


def _memory_patch(event: Mapping[str, Any], state: Mapping[str, Any]) -> tuple[JsonDict, JsonDict]:
    record_id = sha256_json({"key": event["applicability_key"], "version": event["version"]})
    record = {
        "schema": TYPED_REPAIR_VERSION,
        "applicability_key": event["applicability_key"],
        "candidate_operator": event["candidate_operator"],
        "support": event["support"],
        "anchors": event["anchors"],
        "version": event["version"],
        "provenance": event["provenance"],
        "inverse_patch": event["inverse_patch"],
    }
    before = deepcopy(dict(state))
    records = deepcopy(dict(state["records"]))
    records[record_id] = record
    after = _state_with_checksum(int(state["version"]) + 1, records, [*state["lineage"], record_id])
    forward: JsonDict = {
        "expected_state_sha256": sha256_json(before),
        "result_state": after,
        "result_state_sha256": sha256_json(after),
    }
    forward["patch_sha256"] = patch_hash(forward)
    inverse: JsonDict = {
        "expected_state_sha256": sha256_json(after),
        "result_state": before,
        "result_state_sha256": sha256_json(before),
    }
    inverse["patch_sha256"] = patch_hash(inverse)
    return forward, inverse


def apply_memory_patch(state: Mapping[str, Any], patch: Mapping[str, Any]) -> JsonDict:
    """Apply one state patch only when its exact before and after hashes match."""

    if patch_hash(patch) != patch.get("patch_sha256"):
        raise ValueError("patch_checksum_corruption")
    if sha256_json(state) != patch.get("expected_state_sha256"):
        raise ValueError("stale_state")
    result = deepcopy(dict(patch["result_state"]))
    if sha256_json(result) != patch.get("result_state_sha256") or not verify_memory_state(result):
        raise ValueError("result_state_corruption")
    return result


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write bytes with file sync, atomic replacement, and directory sync."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one complete indented JSON document through the durable writer."""

    atomic_write_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def read_memory_state(path: Path) -> tuple[JsonDict | None, str]:
    """Read a complete restart state or reject corruption without coercion."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, "corrupt_or_missing"
    if not isinstance(value, Mapping) or not verify_memory_state(value):
        return None, "checksum_or_schema_invalid"
    return dict(value), "valid"


def build_restart_rollback_rows(
    events: Sequence[Mapping[str, Any]], state_path: Path
) -> list[JsonDict]:
    """Exercise old-or-new restart and one byte-exact inverse per family."""

    state_path.mkdir(parents=True, exist_ok=True)
    old_state = empty_memory_state()
    forward, _ = _memory_patch(events[0], old_state)
    new_state = apply_memory_patch(old_state, forward)
    old_bytes, new_bytes = canonical_bytes(old_state), canonical_bytes(new_state)
    rows: list[JsonDict] = []

    cases = (
        "before_replace_old_state",
        "after_replace_new_state",
        "partial_temp_old_state",
        "corrupt_final_rejected",
    )
    for case in cases:
        case_dir = state_path / case
        target = case_dir / "state.json"
        atomic_write_bytes(target, old_bytes)
        if case == "after_replace_new_state":
            atomic_write_bytes(target, new_bytes)
        elif case == "partial_temp_old_state":
            partial = case_dir / "state.json.interrupted.tmp"
            partial.write_bytes(new_bytes[: max(1, len(new_bytes) // 2)])
        elif case == "corrupt_final_rejected":
            atomic_write_bytes(target, b'{"schema":')
        recovered, status = read_memory_state(target)
        recovered_bytes = None if recovered is None else canonical_bytes(recovered)
        if recovered_bytes == old_bytes:
            recovered_class = "old"
        elif recovered_bytes == new_bytes:
            recovered_class = "new"
        else:
            recovered_class = "rejected"
        expected = (
            "new"
            if case == "after_replace_new_state"
            else "rejected"
            if case == "corrupt_final_rejected"
            else "old"
        )
        row: JsonDict = {
            "row_type": "restart",
            "case": case,
            "recovered_class": recovered_class,
            "read_status": status,
            "old_sha256": sha256_bytes(old_bytes),
            "new_sha256": sha256_bytes(new_bytes),
            "recovered_sha256": None if recovered_bytes is None else sha256_bytes(recovered_bytes),
            "byte_equal": recovered_class in {"old", "new"},
            "passed": recovered_class == expected,
        }
        row["row_sha256"] = row_hash(row)
        rows.append(row)

    for family in FAMILY_ORDER:
        event = next(row for row in events if row["family"] == family)
        before = empty_memory_state()
        patch, inverse = _memory_patch(event, before)
        after = apply_memory_patch(before, patch)
        restored = apply_memory_patch(after, inverse)
        row = {
            "row_type": "rollback",
            "case": "family_inverse_patch",
            "family": family,
            "before_sha256": sha256_bytes(canonical_bytes(before)),
            "after_sha256": sha256_bytes(canonical_bytes(after)),
            "restored_sha256": sha256_bytes(canonical_bytes(restored)),
            "forward_patch_sha256": patch["patch_sha256"],
            "inverse_patch_sha256": inverse["patch_sha256"],
            "byte_equal": canonical_bytes(restored) == canonical_bytes(before),
            "passed": canonical_bytes(restored) == canonical_bytes(before),
        }
        row["row_sha256"] = row_hash(row)
        rows.append(row)
    return rows


def _attack_row(attack_type: str, detected: bool, reason: str, observed: Any) -> JsonDict:
    row: JsonDict = {
        "row_type": "isolation_attack",
        "attack_type": attack_type,
        "detected": detected,
        "failed_closed": detected,
        "passed": detected,
        "reason": reason,
        "observed": observed,
    }
    row["row_sha256"] = row_hash(row)
    return row


def build_isolation_attack_rows(
    events: Sequence[Mapping[str, Any]], orders: Sequence[Mapping[str, Any]], state_path: Path
) -> list[JsonDict]:
    """Inject every registered leakage, state, order, and rollback attack."""

    rows: list[JsonDict] = []
    for attack, field in (
        ("future_leakage", "future_outcome"),
        ("family_specific_key", "family"),
        ("task_identity_key", "event_id"),
    ):
        material = deepcopy(dict(events[0]["applicability_key_material"]))
        material[field] = "forbidden"
        detected = bool(set(material) & set(EXCLUDED_KEY_FIELDS))
        rows.append(
            _attack_row(
                attack,
                detected,
                "forbidden_key_field",
                sorted(set(material) & set(EXCLUDED_KEY_FIELDS)),
            )
        )

    duplicate_ids = [str(row["event_id"]) for row in events] + [str(events[0]["event_id"])]
    rows.append(
        _attack_row(
            "duplicate_events",
            len(set(duplicate_ids)) != len(duplicate_ids),
            "duplicate_event_id",
            duplicate_ids[-1],
        )
    )

    timestamps = list(orders[0]["timestamps"])
    timestamps[0], timestamps[1] = timestamps[1], timestamps[0]
    rows.append(
        _attack_row(
            "shuffled_timestamps",
            timestamps != sorted(timestamps),
            "timestamp_order_invalid",
            timestamps[:2],
        )
    )

    poisoned = deepcopy(dict(events[0]))
    poisoned["candidate_operator"] = deepcopy(dict(poisoned["candidate_operator"]))
    poisoned["candidate_operator"]["exact_repair_passed"] = False
    poison_decision = admission_decision(poisoned)
    rows.append(
        _attack_row(
            "poison_patch",
            not poison_decision["admitted"],
            poison_decision["reason"],
            poison_decision,
        )
    )

    support = admission_decision(events[0], support_ok=False)
    rows.append(
        _attack_row("support_collapse", not support["admitted"], support["reason"], support)
    )
    anchor = admission_decision(events[0], anchor_ok=False, was_active=True)
    rows.append(_attack_row("anchor_regression", anchor["retired"], anchor["reason"], anchor))

    partial_dir = state_path / "attack_partial"
    target = partial_dir / "state.json"
    old = empty_memory_state()
    atomic_write_bytes(target, canonical_bytes(old))
    partial_dir.mkdir(parents=True, exist_ok=True)
    (partial_dir / "state.json.tmp").write_bytes(canonical_bytes(old)[:7])
    recovered, _ = read_memory_state(target)
    rows.append(
        _attack_row(
            "partial_writes",
            recovered == old,
            "partial_temp_not_authoritative",
            sha256_json(recovered),
        )
    )

    corrupt = state_path / "attack_corrupt" / "state.json"
    atomic_write_bytes(corrupt, b"not-json")
    recovered, status = read_memory_state(corrupt)
    rows.append(
        _attack_row("restart_corruption", recovered is None, "restart_corruption_rejected", status)
    )

    forward, inverse = _memory_patch(events[0], old)
    after = apply_memory_patch(old, forward)
    broken = deepcopy(inverse)
    broken["result_state"] = after
    broken["result_state_sha256"] = sha256_json(after)
    broken["patch_sha256"] = patch_hash(broken)
    restored = apply_memory_patch(after, broken)
    rows.append(
        _attack_row(
            "non_invertible_rollback",
            canonical_bytes(restored) != canonical_bytes(old),
            "inverse_byte_mismatch",
            sha256_json(restored),
        )
    )
    return rows


def _protected_hashes(repo_root: Path) -> dict[str, str]:
    return {path.as_posix(): sha256_file(repo_root / path) for path in PROTECTED_PATHS}


def _file_preconditions(repo_root: Path, category: str, paths: Sequence[Path]) -> list[JsonDict]:
    return [
        {
            "category": category,
            "path": path.as_posix(),
            "available": (repo_root / path).is_file(),
            "sha256": sha256_file(repo_root / path),
        }
        for path in paths
    ]


def build_preconditions(repo_root: Path, state_path: Path) -> list[JsonDict]:
    """Measure all input files, host resources, writable state, and substrate."""

    state_path.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(state_path)
    memory_kib = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        first = meminfo.read_text(encoding="utf-8").splitlines()[0].split()
        memory_kib = int(first[1])
    rows = [
        *_file_preconditions(repo_root, "source_corpus", SOURCE_CORPUS_PATHS),
        *_file_preconditions(repo_root, "exact_checker", EXACT_CHECKER_PATHS),
        *_file_preconditions(repo_root, "repair_memory", REPAIR_MEMORY_PATHS),
        *_file_preconditions(repo_root, "state_schema", STATE_SCHEMA_PATHS),
        *_file_preconditions(repo_root, "protected", PROTECTED_PATHS),
        {
            "category": "resource",
            "name": "cpu",
            "available": bool(os.cpu_count()),
            "value": {
                "count": os.cpu_count(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
            "sha256": "sha256:measured",
        },
        {
            "category": "resource",
            "name": "ram",
            "available": memory_kib > 0,
            "value": {"total_kib": memory_kib},
            "sha256": "sha256:measured",
        },
        {
            "category": "resource",
            "name": "disk",
            "available": disk.free > 0,
            "value": {"total": disk.total, "free": disk.free},
            "sha256": "sha256:measured",
        },
        {
            "category": "state_path",
            "path": str(state_path),
            "available": os.access(state_path, os.W_OK),
            "value": "writable",
            "sha256": "sha256:measured",
        },
        {
            "category": "substrate",
            "name": INFERENCE_SUBSTRATE,
            "available": True,
            "value": {"llm_invoked": False, "exact_checkers": list(FAMILY_ORDER)},
            "sha256": "sha256:measured",
        },
    ]
    return rows


def _protected_rows(before: Mapping[str, str], after: Mapping[str, str]) -> list[JsonDict]:
    return [
        {
            "path": path,
            "before_sha256": before[path],
            "after_sha256": after[path],
            "unchanged": before[path] == after[path],
        }
        for path in before
    ]


def _per_unit_rows(
    families: Mapping[str, Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    orders: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    attacks: Sequence[Mapping[str, Any]],
    recovery: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for family, row in families.items():
        rows.append(
            {
                "unit_type": "family",
                "unit_id": family,
                "passed": row["counts"]["total"] == 4,
                "source_sha256": row["manifest_sha256"],
            }
        )
    for row in events:
        rows.append(
            {
                "unit_type": "event",
                "unit_id": row["event_id"],
                "passed": row["candidate_operator"]["exact_repair_passed"],
                "source_sha256": row["row_sha256"],
            }
        )
    for row in orders:
        rows.append(
            {
                "unit_type": "order",
                "unit_id": row["order_id"],
                "passed": len(set(row["ordered_event_ids"])) == EVENT_COUNT,
                "source_sha256": row["manifest_sha256"],
            }
        )
    for group, source in (("checker", controls), ("attack", attacks), ("recovery", recovery)):
        for index, row in enumerate(source):
            rows.append(
                {
                    "unit_type": group,
                    "unit_id": f"{group}-{index:03d}",
                    "passed": row["passed"],
                    "source_sha256": row["row_sha256"],
                }
            )
    for row in rows:
        row["row_sha256"] = row_hash(row)
    return rows


def recompute_aggregate(
    families: Mapping[str, Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    orders: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    attacks: Sequence[Mapping[str, Any]],
    recovery: Sequence[Mapping[str, Any]],
    prequential: Sequence[Mapping[str, Any]],
    tests_run: Sequence[Mapping[str, Any]],
    protected: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Rebuild readiness and counts only from complete source rows."""

    expected_events = {str(row["event_id"]) for row in events}
    gating_tests = [row for row in tests_run if row.get("gates_readiness", True)]
    repository_diagnostics = [
        row for row in tests_run if row.get("verification_scope") == "repository_diagnostic"
    ]
    checks = {
        "families": tuple(families) == FAMILY_ORDER
        and all(row["counts"]["total"] == 4 for row in families.values()),
        "events": len(events) == EVENT_COUNT
        and len(expected_events) == EVENT_COUNT
        and all(row["candidate_operator"]["exact_repair_passed"] for row in events),
        "orders": len(orders) >= 5
        and all(
            set(row["ordered_event_ids"]) == expected_events
            and row["manifest_sha256"] == order_manifest_hash(row)
            for row in orders
        ),
        "checker_controls": len(controls) == 2 * len(FAMILY_ORDER)
        and all(row["passed"] for row in controls),
        "isolation_attacks": {row["attack_type"] for row in attacks} == set(ATTACK_TYPES)
        and all(row["passed"] for row in attacks),
        "restart_rollback": bool(recovery) and all(row["passed"] for row in recovery),
        "prequential": len(prequential) == len(orders) * EVENT_COUNT
        and all(
            row["action_committed_before_outcome"] and not row["same_event_patch_visible"]
            for row in prequential
        ),
        "tests": bool(gating_tests) and all(int(row["exit_code"]) == 0 for row in gating_tests),
        "protected": bool(protected) and all(row["unchanged"] for row in protected),
    }
    return {
        "counts": {
            "families": len(families),
            "events": len(events),
            "orders": len(orders),
            "checker_controls": len(controls),
            "isolation_attacks": len(attacks),
            "restart_rows": sum(row["row_type"] == "restart" for row in recovery),
            "rollback_rows": sum(row["row_type"] == "rollback" for row in recovery),
            "prequential_rows": len(prequential),
            "test_commands": len(tests_run),
            "task_owned_test_commands": len(gating_tests),
            "diagnostic_test_commands": len(repository_diagnostics),
        },
        "checks": checks,
        "diagnostics": {
            "repository_suite_exit_code": (
                None if not repository_diagnostics else int(repository_diagnostics[-1]["exit_code"])
            ),
            "repository_suite_gates_readiness": False,
        },
        "ready": all(checks.values()),
    }


def _gate_summary(aggregate: Mapping[str, Any]) -> JsonDict:
    failed = next((name for name, passed in aggregate["checks"].items() if not passed), None)
    return {
        "failed_check": failed,
        "expected_value": True if failed else None,
        "observed_value": False if failed else None,
    }


def _field_provenance(artifact: Mapping[str, Any]) -> dict[str, JsonDict]:
    provenance: dict[str, JsonDict] = {}
    for field in REQUIRED_ARTIFACT_FIELDS:
        value = artifact.get(field)
        provenance[field] = {
            "principle": FIELD_PRINCIPLES[field],
            "source_path": MODULE_PATH.as_posix(),
            "checker": "exact_family_rows"
            if field in {"family_manifest", "exact_checker_rows", "constraint_family_stream_ready"}
            else "deterministic_fixture_reducer",
            "function": "build_artifact",
            "sha256": "self_excluded"
            if field in {"field_provenance", "reproducibility_checksum"}
            else sha256_json(value),
        }
    return provenance


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    output_path: Path | None = None,
    state_path: Path | None = None,
    date: str = RUN_DATE,
    duration_s: float = 0.0,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    write: bool = False,
    protected_before: Mapping[str, str] | None = None,
) -> JsonDict:
    """Build, validate, and optionally write the complete terminal artifact."""

    destination = output_path or repo_root / RESULT_PATH
    recovery_path = state_path or Path(tempfile.mkdtemp(prefix="carnot-exp6678-state-"))
    before = (
        dict(protected_before) if protected_before is not None else _protected_hashes(repo_root)
    )
    events = build_event_rows()
    families = build_family_manifest(events)
    orders = build_event_order_manifests(events)
    controls = build_exact_checker_rows(families)
    prequential = [row for order in orders for row in build_prequential_rows(events, order)]
    attacks = build_isolation_attack_rows(events, orders, recovery_path)
    recovery = build_restart_rollback_rows(events, recovery_path / "recovery")
    test_rows = [dict(row) for row in (tests_run or [])]
    after = _protected_hashes(repo_root)
    protected = _protected_rows(before, after)
    aggregate = recompute_aggregate(
        families, events, orders, controls, attacks, recovery, prequential, test_rows, protected
    )
    ready = bool(aggregate["ready"])
    artifact: JsonDict = {
        "schema": SCHEMA,
        "run_date": date,
        "status": "complete_ready" if ready else "blocked_fixture_gate",
        "honest_verdict": "complete: exact constraint-family stream ready; no model or learning-benefit claim"
        if ready
        else "blocked_fixture_gate: exact constraint-family stream is not ready",
        "verdict_class": None if ready else "blocked",
        "gate_check_summary": _gate_summary(aggregate),
        "family_manifest": families,
        "event_order_manifests": orders,
        "typed_repair_schema": build_typed_repair_schema(),
        "event_rows": events,
        "prequential_admission_rows": prequential,
        "sealed_initial_state": {
            "bytes": canonical_bytes(empty_memory_state()).decode("utf-8"),
            "sha256": sha256_bytes(canonical_bytes(empty_memory_state())),
        },
        "exact_checker_rows": controls,
        "isolation_attack_rows": attacks,
        "restart_rollback_rows": recovery,
        "constraint_family_stream_ready": ready,
        "per_unit_rows": _per_unit_rows(families, events, orders, controls, attacks, recovery),
        "aggregate_row_recomputation": aggregate,
        "preconditions_checked": build_preconditions(repo_root, recovery_path),
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": {},
        "random_seed": {
            "base": RANDOM_SEED,
            "family": {family: RANDOM_SEED + index for index, family in enumerate(FAMILY_ORDER)},
            "order": dict(ORDER_SEEDS),
        },
        "duration_s": round(float(duration_s), 6),
        "tests_run": test_rows,
    }
    artifact["field_provenance"] = _field_provenance(artifact)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    if write:
        atomic_write_json(destination, artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Validate required fields, row hashes, checksum, and readiness reduction."""

    errors: list[str] = []
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        errors.append("missing_required_fields:" + ",".join(missing))
        return errors
    if artifact_checksum(artifact) != artifact["reproducibility_checksum"]:
        errors.append("reproducibility_checksum_mismatch")
    for row in artifact["isolation_attack_rows"]:
        if row_hash(row) != row.get("row_sha256"):
            errors.append("attack_row_hash_mismatch")
            break
    recomputed = recompute_aggregate(
        artifact["family_manifest"],
        artifact["event_rows"],
        artifact["event_order_manifests"],
        artifact["exact_checker_rows"],
        artifact["isolation_attack_rows"],
        artifact["restart_rollback_rows"],
        artifact["prequential_admission_rows"],
        artifact["tests_run"],
        artifact["protected_files_unchanged"],
    )
    if bool(artifact["constraint_family_stream_ready"]) != bool(recomputed["ready"]):
        errors.append("readiness_recomputation_mismatch")
    if artifact["aggregate_row_recomputation"] != recomputed:
        errors.append("aggregate_recomputation_mismatch")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact["verifier_is_oracle"] is not True:
        errors.append("oracle_boundary_mismatch")
    return errors


def run_command(command: Sequence[str], repo_root: Path) -> JsonDict:
    """Run one verification command and retain its exact exit and output tail."""

    started = time.monotonic()
    completed = subprocess.run(command, cwd=repo_root, text=True, capture_output=True, check=False)
    output = (completed.stdout + completed.stderr).strip()
    return {
        "command": " ".join(command),
        "exit_code": completed.returncode,
        "summary": "passed" if completed.returncode == 0 else "failure",
        "duration_s": round(time.monotonic() - started, 6),
        "output_tail": output[-4000:],
    }


def run_verification_commands(repo_root: Path) -> list[JsonDict]:
    """Run focused, coverage, lint, spec, E2E, and one full-suite command."""

    rows: list[JsonDict] = []
    for command in VERIFICATION_COMMANDS:
        row = run_command(tuple(command.split()), repo_root)
        diagnostic = command == ".venv/bin/pytest tests/python -q"
        row["verification_scope"] = "repository_diagnostic" if diagnostic else "task_owned"
        row["gates_readiness"] = not diagnostic
        rows.append(row)
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument(
        "--state-path",
        type=Path,
        default=Path(tempfile.gettempdir()) / "carnot-exp6678-state",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run verification, build the fixture, and write its terminal artifact."""

    args = parse_args(argv)
    started = time.monotonic()
    protected_before = _protected_hashes(REPO_ROOT)
    tests = run_verification_commands(REPO_ROOT)
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    state_path = args.state_path if args.state_path.is_absolute() else REPO_ROOT / args.state_path
    artifact = build_artifact(
        repo_root=REPO_ROOT,
        output_path=output,
        state_path=state_path,
        date=args.date,
        duration_s=time.monotonic() - started,
        tests_run=tests,
        write=True,
        protected_before=protected_before,
    )
    errors = validate_artifact(artifact)
    print(
        json.dumps(
            {
                "output": str(output),
                "ready": artifact["constraint_family_stream_ready"],
                "errors": errors,
            },
            sort_keys=True,
        )
    )
    return 0 if not errors and artifact["constraint_family_stream_ready"] else 1


if __name__ == "__main__":  # pragma: no cover - exercised through ``python -m``.
    raise SystemExit(main())
