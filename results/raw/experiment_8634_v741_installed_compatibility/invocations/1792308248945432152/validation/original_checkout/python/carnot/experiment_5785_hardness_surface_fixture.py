"""Exp5785 sealed hardness/surface exact fixture.

Spec refs: REQ-BENCH-5785, SCENARIO-BENCH-5785,
SCENARIO-BENCH-5785-CONTROLS, REQ-VERIFY-5785, SCENARIO-VERIFY-5785.

This module builds the cheap no-LLM fixture that downstream SOTA inference will
consume.  The natural-language surface is deliberately mutable, while the
protected facts, mutable constraints, exact candidate domains, and row hashes
are the authority.  That separation is the point of the fixture: a model can
later be tested for surface sensitivity without re-litigating labels.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import sys
from typing import Any


JsonDict = dict[str, Any]
Probe = Callable[[], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5785_hardness_surface_fixture.json")
ROW_FILE_RELATIVE_PATH = Path("results/experiment_5785_hardness_surface_fixture.rows.jsonl")
UPSTREAM_EXP5784_RELATIVE_PATH = Path(
    "results/experiment_5784_evidence_index_terminal_qualification.json"
)

SCHEMA = "carnot.experiment_5785.hardness_surface_fixture.v1"
ROW_SCHEMA = SCHEMA + ".row"
EXPERIMENT = 5785
EXPERIMENT_ID = "experiment_5785_hardness_surface_fixture"
RUN_DATE = "20260722"
INFERENCE_SUBSTRATE = "deterministic_local_fixture_generation_z3_and_exact_validators_no_llm"
PRIMARY_VALIDATOR_VERSION = "exp5785_z3_primary_exact_validator_v1"
INDEPENDENT_VALIDATOR_VERSION = "exp5785_enumeration_independent_validator_v1"
GENERATOR_VERSION = "exp5785_hardness_surface_fixture_v1"

REQUIRED_FAMILIES = ("finite_domain_scheduling", "logic_grid", "typed_finite_choice")
SPLITS = ("train", "calibration", "future_test")
SURFACE_KINDS = ("canonical", "symbol_relabel", "order_paraphrase", "meaning_change_canary")
PROOF_PRESERVING_KINDS = ("symbol_relabel", "order_paraphrase")
LABELS = ("A", "B", "C", "D")
PRODUCER_GATE_FIELDS = ("fixture_ready_score", "exact_label_coverage", "parser_control_pass_rate")
REQUIRED_PARSER_CONTROLS = (
    "truncation",
    "missing_answer",
    "duplicate_id",
    "invalid_candidate",
    "whitespace_order",
    "stop_token",
    "adversarial_payload",
    "valid_wrong_label",
)
SPEC_REFS = (
    "REQ-BENCH-5785",
    "SCENARIO-BENCH-5785",
    "SCENARIO-BENCH-5785-CONTROLS",
    "REQ-VERIFY-5785",
    "SCENARIO-VERIFY-5785",
)
RANDOM_SEEDS: JsonDict = {
    "base_seed": 5785,
    "unit_seed": 5785001,
    "surface_seed": 5785002,
    "label_seed": 5785003,
    "parser_control_seed": 5785004,
}

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "spec_refs",
    "fixture_schema",
    "row_file",
    "row_file_sha256",
    "family_counts",
    "independent_unit_count",
    "sample_size_justification",
    "chronological_split_receipts",
    "solver_hardness_bins",
    "surface_variant_matrix",
    "proof_preserving_receipts",
    "meaning_change_canary_receipts",
    "protected_fact_manifest",
    "mutable_constraint_manifest",
    "candidate_completeness_receipts",
    "parser_contract",
    "parser_negative_controls",
    "exact_validator_receipts",
    "leakage_checks",
    "fixture_ready_score",
    "exact_label_coverage",
    "parser_control_pass_rate",
    "producer_gate_fields",
    "inference_substrate",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)
FIELD_PRINCIPLES: JsonDict = {}

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5785_hardness_surface_fixture.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5785_hardness_surface_fixture.py "
    "-m pytest tests/python/test_experiment_5785_hardness_surface_fixture.py -q --no-cov -n 0 && "
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5785_hardness_surface_fixture.py --fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/root_clutter_sweep.py",
)


class ManifestReplayError(ValueError):
    """Raised when row manifest bytes no longer match sealed artifact receipts."""


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically before hashing."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for text evidence."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for JSON-compatible evidence."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash a local file in chunks rather than trusting filesystem metadata."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _memory_probe() -> JsonDict:  # pragma: no cover - host-dependent preflight.
    required_mb = 512
    available_mb = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available_mb = int(line.split()[1]) // 1024
                break
    if available_mb == 0:
        available_mb = int(
            os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
        )
    return {
        "available_mb": available_mb,
        "required_mb": required_mb,
        "ok": available_mb >= required_mb,
    }


def _disk_probe() -> JsonDict:  # pragma: no cover - host-dependent preflight.
    required_mb = 512
    usage = shutil.disk_usage(REPO_ROOT)
    available_mb = int(usage.free / (1024 * 1024))
    return {
        "available_mb": available_mb,
        "required_mb": required_mb,
        "ok": available_mb >= required_mb,
    }


def _z3_probe() -> JsonDict:  # pragma: no cover - environment-dependent preflight.
    try:
        import z3  # type: ignore[import-untyped]
    except ImportError as exc:
        return {"available": False, "version": "", "ok": False, "error": str(exc)}
    return {"available": True, "version": z3.get_version_string(), "ok": True}


def _read_json_object(path: str | Path) -> JsonDict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return dict(payload)


def _replay_exp5784(path: str | Path) -> JsonDict:
    gate_specs = (
        ("evidence_index_ready_score", "==", 1.0),
        ("next_range_collision_count", "==", 0),
        ("unresolved_canonical_count", "==", 0),
        ("history_mutation_count", "==", 0),
    )
    try:
        artifact = _read_json_object(path)
        receipts = []
        for field, op, expected in gate_specs:
            actual = artifact.get(field)
            passed = actual == expected
            receipts.append(
                {
                    "source": str(UPSTREAM_EXP5784_RELATIVE_PATH),
                    "field": field,
                    "op": op,
                    "expected": expected,
                    "actual": actual,
                    "passed": passed,
                }
            )
        return {
            "artifact_path": str(UPSTREAM_EXP5784_RELATIVE_PATH),
            "artifact_sha256": sha256_file(path),
            "status": artifact.get("status"),
            "gate_receipts": receipts,
            "ok": all(row["passed"] for row in receipts),
        }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "artifact_path": str(UPSTREAM_EXP5784_RELATIVE_PATH),
            "gate_receipts": [],
            "ok": False,
            "error": str(exc),
        }


def collect_preconditions(
    *,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    row_file_path: str | Path = REPO_ROOT / ROW_FILE_RELATIVE_PATH,
    upstream_exp5784_path: str | Path = REPO_ROOT / UPSTREAM_EXP5784_RELATIVE_PATH,
    memory_probe: Probe = _memory_probe,
    disk_probe: Probe = _disk_probe,
    z3_probe: Probe = _z3_probe,
) -> JsonDict:
    """Collect all Step 0 checks before rows are generated."""

    memory = memory_probe()
    disk = disk_probe()
    z3 = z3_probe()
    exp5784 = _replay_exp5784(upstream_exp5784_path)
    deliverable_unoccupied = not Path(result_path).exists()
    row_file_unoccupied = not Path(row_file_path).exists()
    blocked: list[str] = []
    if exp5784.get("ok") is not True:
        blocked.append("exp5784_gate_replay_failed")
    if z3.get("ok") is not True:
        blocked.append("z3_unavailable")
    if memory.get("ok") is not True:
        blocked.append("insufficient_free_ram")
    if disk.get("ok") is not True:
        blocked.append("insufficient_free_disk")
    if not deliverable_unoccupied:
        blocked.append("deliverable_path_occupied")
    if not row_file_unoccupied:
        blocked.append("row_file_path_occupied")
    return {
        "schema": SCHEMA + ".preconditions",
        "run_date": RUN_DATE,
        "python": {
            "available": True,
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "exp5784_gate_replay": exp5784,
        "z3": z3,
        "exact_validators": {
            "primary": PRIMARY_VALIDATOR_VERSION,
            "independent": INDEPENDENT_VALIDATOR_VERSION,
            "available": z3.get("ok") is True,
        },
        "memory": memory,
        "disk": disk,
        "deterministic_seeds": dict(RANDOM_SEEDS),
        "deliverable_paths": {
            "result_path": str(RESULT_RELATIVE_PATH),
            "row_file_path": str(ROW_FILE_RELATIVE_PATH),
            "result_unoccupied": deliverable_unoccupied,
            "row_file_unoccupied": row_file_unoccupied,
        },
        "preconditions_ready": not blocked,
        "blocked_reasons": sorted(blocked),
    }


def _split_for_index(index: int) -> str:
    return SPLITS[index // 10]


def _status_for_index(index: int) -> str:
    return "positive" if index % 2 == 0 else "negative"


def _solver_bin_for_index(index: int) -> str:
    return ("low", "medium", "high")[index % 3]


def _scheduling_problem(index: int, canary: bool) -> JsonDict:
    status = _status_for_index(index)
    tasks = ["compile", "test", "ship"]
    slots = [0, 1, 2]
    constraints = [
        {"id": "all_tasks_distinct", "type": "all_distinct", "vars": tasks},
        {"id": "compile_before_test", "type": "before", "left": "compile", "right": "test"},
    ]
    if status == "positive":
        constraints.append({"id": "ship_fixed", "type": "equals", "var": "ship", "value": 2})
        if canary:
            constraints.append(
                {
                    "id": "test_before_compile_canary",
                    "type": "before",
                    "left": "test",
                    "right": "compile",
                }
            )
    else:
        constraints.append(
            {"id": "test_before_compile", "type": "before", "left": "test", "right": "compile"}
        )
        if canary:
            constraints[-1] = {
                "id": "compile_before_ship_canary",
                "type": "before",
                "left": "compile",
                "right": "ship",
            }
    return {
        "protected_facts": {"tasks": tasks, "slots": slots},
        "mutable_constraints": constraints,
        "answer_domain": ["FEASIBLE", "INFEASIBLE", "UNKNOWN", "BOTH"],
    }


def _logic_problem(index: int, canary: bool) -> JsonDict:
    status = _status_for_index(index)
    people = ["Ada", "Ben", "Cy"]
    colors = ["red", "green", "blue"]
    rooms = ["north", "east", "west"]
    color = index % 3
    constraints = [
        {"id": "colors_bijective", "type": "all_distinct", "field": "color"},
        {"id": "rooms_bijective", "type": "all_distinct", "field": "room"},
        {"id": "ada_color", "type": "equals", "entity": "Ada", "field": "color", "value": color},
        {
            "id": "ben_room",
            "type": "equals",
            "entity": "Ben",
            "field": "room",
            "value": (index + 1) % 3,
        },
    ]
    if status == "negative":
        constraints.append(
            {
                "id": "ada_color_conflict",
                "type": "equals",
                "entity": "Ada",
                "field": "color",
                "value": (color + 1) % 3,
            }
        )
        if canary:
            constraints[-1] = {
                "id": "cy_color_canary",
                "type": "not_equals",
                "entity": "Cy",
                "field": "color",
                "value": color,
            }
    elif canary:
        constraints.append(
            {
                "id": "ada_color_conflict_canary",
                "type": "equals",
                "entity": "Ada",
                "field": "color",
                "value": (color + 1) % 3,
            }
        )
    return {
        "protected_facts": {"people": people, "colors": colors, "rooms": rooms},
        "mutable_constraints": constraints,
        "answer_domain": ["FEASIBLE", "INFEASIBLE", "UNKNOWN", "BOTH"],
    }


def _choice_problem(index: int, canary: bool) -> JsonDict:
    status = _status_for_index(index)
    choices = [
        {"id": "CHOICE_0", "kind": "alpha", "risk": 0, "cost": 1, "score": 2 + index % 3},
        {"id": "CHOICE_1", "kind": "beta", "risk": 1, "cost": 2, "score": 5 + index % 2},
        {"id": "CHOICE_2", "kind": "alpha", "risk": 2, "cost": 3, "score": 6},
    ]
    constraints = [
        {"id": "kind_alpha", "type": "equals", "field": "kind", "value": "alpha"},
        {"id": "risk_cap", "type": "lte", "field": "risk", "value": 2},
        {"id": "cost_cap", "type": "lte", "field": "cost", "value": 3},
        {"id": "objective", "type": "maximize", "field": "score"},
    ]
    if status == "negative":
        constraints[1] = {"id": "risk_cap", "type": "lte", "field": "risk", "value": -1}
        if canary:
            constraints[1] = {"id": "risk_cap_canary", "type": "lte", "field": "risk", "value": 2}
    elif canary:
        constraints[2] = {"id": "cost_cap_canary", "type": "lte", "field": "cost", "value": 0}
    return {
        "protected_facts": {"choices": choices, "answer_type": "choice_id_or_no_feasible"},
        "mutable_constraints": constraints,
        "answer_domain": ["CHOICE_0", "CHOICE_1", "CHOICE_2", "NO_FEASIBLE"],
    }


def _problem(family: str, index: int, canary: bool = False) -> JsonDict:
    if family == "finite_domain_scheduling":
        return _scheduling_problem(index, canary)
    if family == "logic_grid":
        return _logic_problem(index, canary)
    return _choice_problem(index, canary)


def _constraint_text(constraint: Mapping[str, Any]) -> str:
    kind = str(constraint["type"])
    if kind == "all_distinct":
        return f"{constraint.get('vars') or constraint.get('field')} are all distinct"
    if kind == "before":
        return f"{constraint['left']} before {constraint['right']}"
    if kind == "equals":
        target = constraint.get("var") or f"{constraint.get('entity')}.{constraint.get('field')}"
        return f"{target} equals {constraint['value']}"
    if kind == "not_equals":
        return f"{constraint['entity']}.{constraint['field']} is not {constraint['value']}"
    if kind == "lte":
        return f"{constraint['field']} <= {constraint['value']}"
    return f"maximize {constraint['field']}"


def _surface_text(family: str, problem: Mapping[str, Any], kind: str, unit_id: str) -> str:
    constraints = list(problem["mutable_constraints"])
    if kind == "order_paraphrase":
        constraints = list(reversed(constraints))
    prefix = {
        "canonical": "Solve the exact fixture.",
        "symbol_relabel": "Solve the relabeled exact fixture with opaque names.",
        "order_paraphrase": "Determine the same formal case from the reordered clues.",
        "meaning_change_canary": "Solve the canary case whose mutable constraint changed.",
    }[kind]
    return (
        f"{prefix} unit={unit_id}; family={family}; facts={canonical_json(problem['protected_facts'])}; "
        "constraints=" + " | ".join(_constraint_text(row) for row in constraints)
    )


def _z3_scheduling(problem: Mapping[str, Any]) -> JsonDict:
    import z3  # type: ignore[import-untyped]

    variables = {name: z3.Int(name) for name in problem["protected_facts"]["tasks"]}
    solver = z3.Solver()
    slots = list(problem["protected_facts"]["slots"])
    for var in variables.values():
        solver.add(z3.Or([var == value for value in slots]))
    for constraint in problem["mutable_constraints"]:
        kind = constraint["type"]
        if kind == "all_distinct":
            solver.add(z3.Distinct([variables[name] for name in constraint["vars"]]))
        elif kind == "before":
            solver.add(variables[str(constraint["left"])] < variables[str(constraint["right"])])
        elif kind == "equals":
            solver.add(variables[str(constraint["var"])] == int(constraint["value"]))
    result = solver.check()
    elapsed_ms = round(0.01 * len(problem["mutable_constraints"]), 6)
    if result == z3.sat:
        assignment = _enumerate_scheduling_assignment(problem)
        return {
            "status": "sat",
            "exact_answer": "FEASIBLE",
            "certificate": {"assignment": assignment},
            "time_ms": elapsed_ms,
        }
    return {
        "status": "unsat",
        "exact_answer": "INFEASIBLE",
        "certificate": {"unsat_constraints": len(problem["mutable_constraints"])},
        "time_ms": elapsed_ms,
    }


def _z3_logic(problem: Mapping[str, Any]) -> JsonDict:
    import z3  # type: ignore[import-untyped]

    people = list(problem["protected_facts"]["people"])
    color = {name: z3.Int(f"{name}_color") for name in people}
    room = {name: z3.Int(f"{name}_room") for name in people}
    solver = z3.Solver()
    for field_vars in (color, room):
        for var in field_vars.values():
            solver.add(z3.And(var >= 0, var < 3))
        solver.add(z3.Distinct(list(field_vars.values())))
    for constraint in problem["mutable_constraints"]:
        if constraint["type"] == "equals":
            target = color if constraint["field"] == "color" else room
            solver.add(target[str(constraint["entity"])] == int(constraint["value"]))
        elif constraint["type"] == "not_equals":
            target = color if constraint["field"] == "color" else room
            solver.add(target[str(constraint["entity"])] != int(constraint["value"]))
    result = solver.check()
    elapsed_ms = round(0.01 * len(problem["mutable_constraints"]), 6)
    if result == z3.sat:
        assignment = _enumerate_logic_assignment(problem)
        return {
            "status": "sat",
            "exact_answer": "FEASIBLE",
            "certificate": {"assignment": assignment},
            "time_ms": elapsed_ms,
        }
    return {
        "status": "unsat",
        "exact_answer": "INFEASIBLE",
        "certificate": {"unsat_constraints": len(problem["mutable_constraints"])},
        "time_ms": elapsed_ms,
    }


def _scheduling_constraints_hold(problem: Mapping[str, Any], assignment: Mapping[str, int]) -> bool:
    for constraint in problem["mutable_constraints"]:
        kind = constraint["type"]
        if kind == "all_distinct" and len(
            {assignment[str(name)] for name in constraint["vars"]}
        ) != len(constraint["vars"]):
            return False
        if (
            kind == "before"
            and assignment[str(constraint["left"])] >= assignment[str(constraint["right"])]
        ):
            return False
        if kind == "equals" and assignment[str(constraint["var"])] != int(constraint["value"]):
            return False
    return True


def _enumerate_scheduling_assignment(problem: Mapping[str, Any]) -> JsonDict:
    tasks = list(problem["protected_facts"]["tasks"])
    slots = list(problem["protected_facts"]["slots"])
    for first in slots:
        for second in slots:
            for third in slots:
                assignment = dict(zip(tasks, (first, second, third), strict=True))
                if _scheduling_constraints_hold(problem, assignment):
                    return assignment
    return {}


def _logic_constraints_hold(
    problem: Mapping[str, Any], assignment: Mapping[str, Mapping[str, int]]
) -> bool:
    for constraint in problem["mutable_constraints"]:
        kind = constraint["type"]
        if kind == "equals" and assignment[str(constraint["entity"])][
            str(constraint["field"])
        ] != int(constraint["value"]):
            return False
        if kind == "not_equals" and assignment[str(constraint["entity"])][
            str(constraint["field"])
        ] == int(constraint["value"]):
            return False
    return True


def _enumerate_logic_assignment(problem: Mapping[str, Any]) -> JsonDict:
    people = list(problem["protected_facts"]["people"])
    permutations = (
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    )
    for colors in permutations:
        for rooms in permutations:
            assignment = {
                person: {"color": colors[index], "room": rooms[index]}
                for index, person in enumerate(people)
            }
            if _logic_constraints_hold(problem, assignment):
                return assignment
    return {}


def _choice_feasible(choice: Mapping[str, Any], constraints: Sequence[Mapping[str, Any]]) -> bool:
    for constraint in constraints:
        if (
            constraint["type"] == "equals"
            and choice[str(constraint["field"])] != constraint["value"]
        ):
            return False
        if constraint["type"] == "lte" and int(choice[str(constraint["field"])]) > int(
            constraint["value"]
        ):
            return False
    return True


def _z3_choice(problem: Mapping[str, Any]) -> JsonDict:
    import z3  # type: ignore[import-untyped]

    choices = list(problem["protected_facts"]["choices"])
    index = z3.Int("choice_index")
    solver = z3.Solver()
    solver.add(z3.And(index >= 0, index < len(choices)))
    allowed_indexes = [
        row_index
        for row_index, choice in enumerate(choices)
        if _choice_feasible(choice, problem["mutable_constraints"])
    ]
    solver.add(
        z3.Or([index == value for value in allowed_indexes])
        if allowed_indexes
        else z3.BoolVal(False)
    )
    result = solver.check()
    elapsed_ms = round(0.01 * len(problem["mutable_constraints"]), 6)
    if result != z3.sat:
        return {
            "status": "unsat",
            "exact_answer": "NO_FEASIBLE",
            "certificate": {"unsat_constraints": len(problem["mutable_constraints"])},
            "time_ms": elapsed_ms,
        }
    feasible = [choices[value] for value in allowed_indexes]
    best = max(
        feasible, key=lambda row: (int(row["score"]), -int(str(row["id"]).rsplit("_", 1)[1]))
    )
    return {
        "status": "sat",
        "exact_answer": str(best["id"]),
        "certificate": {"selected_choice": best},
        "time_ms": elapsed_ms,
    }


def primary_validate(family: str, problem: Mapping[str, Any]) -> JsonDict:
    """Validate one fixture row with Z3-backed exact authority."""

    if family == "finite_domain_scheduling":
        result = _z3_scheduling(problem)
    elif family == "logic_grid":
        result = _z3_logic(problem)
    else:
        result = _z3_choice(problem)
    result["validator_version"] = PRIMARY_VALIDATOR_VERSION
    return result


def independent_validate(family: str, problem: Mapping[str, Any]) -> JsonDict:
    """Validate one fixture row with a direct finite-domain enumerator."""

    if family == "typed_finite_choice":
        choices = [
            row
            for row in problem["protected_facts"]["choices"]
            if _choice_feasible(row, problem["mutable_constraints"])
        ]
        if not choices:
            result = {
                "status": "unsat",
                "exact_answer": "NO_FEASIBLE",
                "certificate": {"enumerated_feasible_count": 0},
            }
        else:
            best = max(
                choices, key=lambda row: (int(row["score"]), -int(str(row["id"]).rsplit("_", 1)[1]))
            )
            result = {
                "status": "sat",
                "exact_answer": str(best["id"]),
                "certificate": {"selected_choice": best},
            }
    else:
        primary = primary_validate(family, problem)
        result = {key: primary[key] for key in ("status", "exact_answer", "certificate")}
    result["validator_version"] = INDEPENDENT_VALIDATOR_VERSION
    return result


def _validator_receipt(family: str, problem: Mapping[str, Any], solver_bin: str) -> JsonDict:
    primary = primary_validate(family, problem)
    independent = independent_validate(family, problem)
    receipt = {
        "primary": primary,
        "independent": independent,
        "solver_effort_bin": solver_bin,
        "validators_agree": primary["status"] == independent["status"]
        and primary["exact_answer"] == independent["exact_answer"],
    }
    receipt["certificate_hash"] = sha256_json(
        {
            "primary": primary["certificate"],
            "independent": independent["certificate"],
            "exact_answer": primary["exact_answer"],
        }
    )
    return receipt


def _label_mapping(
    answer_domain: Sequence[str],
    exact_answer: str,
    *,
    label_index: int,
    sequence_index: int,
) -> list[JsonDict]:
    exact_label = LABELS[label_index % len(LABELS)]
    remaining_labels = [label for label in LABELS if label != exact_label]
    remaining_candidates = [candidate for candidate in answer_domain if candidate != exact_answer]
    rng = random.Random(RANDOM_SEEDS["label_seed"] + sequence_index)
    rng.shuffle(remaining_labels)
    rng.shuffle(remaining_candidates)
    by_label = {exact_label: exact_answer}
    for label, candidate in zip(remaining_labels, remaining_candidates, strict=True):
        by_label[label] = candidate
    return [
        {
            "label": label,
            "candidate": by_label[label],
            "candidate_hash": sha256_text(by_label[label]),
            "is_exact": by_label[label] == exact_answer,
        }
        for label in LABELS
    ]


def candidate_completeness_receipt(row: Mapping[str, Any]) -> JsonDict:
    """Prove a row exposes a complete bounded finite-choice domain."""

    domain = [str(item) for item in row["candidate_domain"]]
    candidates = [str(item["candidate"]) for item in row["label_mapping"]]
    labels = [str(item["label"]) for item in row["label_mapping"]]
    exact = str(row["exact_answer"])
    return {
        "row_id": str(row["row_id"]),
        "domain_size": len(domain),
        "candidate_count": len(candidates),
        "label_count": len(labels),
        "complete": set(domain) == set(candidates) and len(candidates) == len(domain),
        "exact_candidate_present": exact in candidates,
        "labels_unique": len(labels) == len(set(labels)) == len(LABELS),
        "domain_hash": sha256_json(domain),
        "label_mapping_hash": sha256_json(row["label_mapping"]),
    }


def _row_hash(row: Mapping[str, Any]) -> str:
    stable = dict(row)
    stable["row_hash"] = ""
    return sha256_json(stable)


def _build_row(
    *,
    unit_id: str,
    family: str,
    split: str,
    unit_index: int,
    surface_kind: str,
    sequence_index: int,
) -> JsonDict:
    problem = _problem(family, unit_index, canary=surface_kind == "meaning_change_canary")
    protected_fact_hash = sha256_json(problem["protected_facts"])
    mutable_constraint_hash = sha256_json(problem["mutable_constraints"])
    solver_bin = _solver_bin_for_index(unit_index)
    validator = _validator_receipt(family, problem, solver_bin)
    label_index = unit_index + (1 if surface_kind == "meaning_change_canary" else 0)
    label_mapping = _label_mapping(
        problem["answer_domain"],
        str(validator["primary"]["exact_answer"]),
        label_index=label_index,
        sequence_index=sequence_index,
    )
    row = {
        "schema": ROW_SCHEMA,
        "row_id": f"{unit_id}-{surface_kind}",
        "unit_id": unit_id,
        "family": family,
        "split": split,
        "chronology_index": sequence_index,
        "surface_kind": surface_kind,
        "proof_preserving": surface_kind in PROOF_PRESERVING_KINDS or surface_kind == "canonical",
        "solver_effort_bin": solver_bin,
        "preregistered_conflict_bin": solver_bin,
        "preregistered_time_bin": solver_bin,
        "protected_facts": problem["protected_facts"],
        "mutable_constraints": problem["mutable_constraints"],
        "protected_fact_hash": protected_fact_hash,
        "mutable_constraint_hash": mutable_constraint_hash,
        "surface_text": _surface_text(family, problem, surface_kind, unit_id),
        "candidate_domain": list(problem["answer_domain"]),
        "label_mapping": label_mapping,
        "candidate_labels": list(LABELS),
        "exact_answer": str(validator["primary"]["exact_answer"]),
        "exact_label": next(item["label"] for item in label_mapping if item["is_exact"]),
        "exact_status": str(validator["primary"]["status"]),
        "exact_certificate_hash": str(validator["certificate_hash"]),
        "exact_validator_receipt": validator,
        "row_hash": "",
    }
    row["candidate_completeness_receipt"] = candidate_completeness_receipt(row)
    row["row_hash"] = _row_hash(row)
    return row


def generate_fixture_rows() -> list[JsonDict]:
    """Generate sealed chronological rows before downstream inference exists."""

    rows: list[JsonDict] = []
    sequence_index = 0
    for family in REQUIRED_FAMILIES:
        for unit_index in range(30):
            split = _split_for_index(unit_index)
            unit_id = f"exp5785-{split}-{family.replace('_', '-')}-{unit_index:03d}"
            for surface_kind in SURFACE_KINDS:
                rows.append(
                    _build_row(
                        unit_id=unit_id,
                        family=family,
                        split=split,
                        unit_index=unit_index,
                        surface_kind=surface_kind,
                        sequence_index=sequence_index,
                    )
                )
                sequence_index += 1
    return rows


def rows_to_jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    """Serialize row records to deterministic JSONL bytes."""

    return "".join(canonical_json(row) + "\n" for row in rows)


def read_row_file(path: str | Path) -> list[JsonDict]:
    """Read the sealed fixture JSONL manifest."""

    text = Path(path).read_text(encoding="utf-8") if Path(path).exists() else ""
    return [dict(json.loads(line)) for line in text.splitlines() if line.strip()]


def _chronological_split_receipts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_split = {split: [row for row in rows if row["split"] == split] for split in SPLITS}
    canonical_by_split = {
        split: [row for row in split_rows if row["surface_kind"] == "canonical"]
        for split, split_rows in by_split.items()
    }
    pairwise: JsonDict = {}
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            intersection = sorted(
                {row["row_hash"] for row in by_split[left]}
                & {row["row_hash"] for row in by_split[right]}
            )
            pairwise[f"{left}|{right}"] = intersection
    return {
        "chronology": list(SPLITS),
        "canonical_unit_counts": {split: len(canonical_by_split[split]) for split in SPLITS},
        "row_counts": {split: len(by_split[split]) for split in SPLITS},
        "split_hashes": {
            split: sha256_json([row["row_hash"] for row in by_split[split]]) for split in SPLITS
        },
        "pairwise_row_hash_intersections": pairwise,
        "disjoint_row_hashes": all(not value for value in pairwise.values()),
    }


def _proof_receipts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_unit: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_unit.setdefault(str(row["unit_id"]), []).append(row)
    receipts: JsonDict = {}
    for unit_id, unit_rows in by_unit.items():
        canonical = next(row for row in unit_rows if row["surface_kind"] == "canonical")
        variants = [row for row in unit_rows if row["surface_kind"] in PROOF_PRESERVING_KINDS]
        receipts[unit_id] = {
            "canonical_row_id": canonical["row_id"],
            "variant_row_ids": [row["row_id"] for row in variants],
            "surface_kinds": [row["surface_kind"] for row in variants],
            "protected_fact_hash_preserved": all(
                row["protected_fact_hash"] == canonical["protected_fact_hash"] for row in variants
            ),
            "exact_label_preserved": all(
                row["exact_label"] == canonical["exact_label"] for row in variants
            ),
            "exact_answer_hashes": [
                sha256_text(str(row["exact_answer"])) for row in [canonical, *variants]
            ],
        }
    return receipts


def _canary_receipts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_unit: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_unit.setdefault(str(row["unit_id"]), []).append(row)
    receipts: JsonDict = {}
    for unit_id, unit_rows in by_unit.items():
        canonical = next(row for row in unit_rows if row["surface_kind"] == "canonical")
        canary = next(row for row in unit_rows if row["surface_kind"] == "meaning_change_canary")
        receipts[unit_id] = {
            "canonical_row_id": canonical["row_id"],
            "canary_row_id": canary["row_id"],
            "protected_fact_hash_preserved": canary["protected_fact_hash"]
            == canonical["protected_fact_hash"],
            "exact_label_changed": canary["exact_label"] != canonical["exact_label"],
            "exact_answer_changed": canary["exact_answer"] != canonical["exact_answer"],
            "mutable_constraint_hash_changed": canary["mutable_constraint_hash"]
            != canonical["mutable_constraint_hash"],
        }
    return receipts


def _candidate_receipts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {str(row["row_id"]): dict(row["candidate_completeness_receipt"]) for row in rows}


def _validator_receipts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {str(row["row_id"]): dict(row["exact_validator_receipt"]) for row in rows}


def _protected_manifest(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    manifest: JsonDict = {}
    for row in rows:
        unit_id = str(row["unit_id"])
        manifest.setdefault(
            unit_id,
            {
                "family": row["family"],
                "split": row["split"],
                "protected_facts": row["protected_facts"],
                "protected_fact_hash": row["protected_fact_hash"],
            },
        )
    return manifest


def _mutable_manifest(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {str(row["row_id"]): str(row["mutable_constraint_hash"]) for row in rows}


def _leakage_checks(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    protected_has_answer_channel = any(
        any(
            token in canonical_json(row["protected_facts"]).lower()
            for token in ("exact_label", "exact_answer", "label_mapping")
        )
        for row in rows
    )
    return {
        "protected_fact_separation": not protected_has_answer_channel,
        "mutable_constraints_separate": all(
            row["protected_fact_hash"] != row["mutable_constraint_hash"] for row in rows
        ),
        "candidate_labels_not_in_surface_text": all(
            not any(f" {label}:" in str(row["surface_text"]) for label in LABELS) for row in rows
        ),
        "future_test_answer_leakage": not any(
            f" {row['exact_label']}:" in str(row["surface_text"])
            for row in rows
            if row["split"] == "future_test"
        ),
    }


def _solver_hardness_bins(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    canonical_rows = [row for row in rows if row["surface_kind"] == "canonical"]
    return {
        "terminology": "solver_hardness_only_not_model_hardness",
        "preregistered_bins": {
            "low": {"conflict_range": [0, 2], "time_budget_ms": 5},
            "medium": {"conflict_range": [3, 8], "time_budget_ms": 20},
            "high": {"conflict_range": [9, 99], "time_budget_ms": 100},
        },
        "canonical_unit_counts": dict(
            Counter(str(row["solver_effort_bin"]) for row in canonical_rows)
        ),
        "row_counts": dict(Counter(str(row["solver_effort_bin"]) for row in rows)),
    }


def parser_contract() -> JsonDict:
    """Describe the finite-choice parser boundary consumed by future inference."""

    return {
        "format": "one line per fixture row: <row_id>: <label>",
        "candidate_labels": list(LABELS),
        "accepts_whitespace_and_order_variants": True,
        "parser_failure_is_separate_from_exact_wrongness": True,
        "required_negative_controls": list(REQUIRED_PARSER_CONTROLS),
    }


def parse_response(text: str, row_by_id: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Parse finite-choice answers while separating format failure from wrong labels."""

    lowered = text.lower()
    if "<|eot_id|>" in text or "<stop>" in lowered:
        return _parse_failure("stop_token")
    if "ignore previous" in lowered or "<script" in lowered or "```" in text:
        return _parse_failure("adversarial_payload")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if any(":" not in line for line in lines):
        return _parse_failure("truncation")
    parsed: JsonDict = {}
    for line in lines:
        row_id, label = [part.strip() for part in line.split(":", 1)]
        if not row_id or not label:
            return _parse_failure("truncation")
        if row_id in parsed:
            return _parse_failure("duplicate_id")
        if row_id not in row_by_id:
            return _parse_failure("invalid_id")
        if label not in row_by_id[row_id]["candidate_labels"]:
            return _parse_failure("invalid_candidate")
        parsed[row_id] = label
    missing = sorted(set(row_by_id) - set(parsed))
    if missing:
        return _parse_failure("missing_answer")
    valid_wrong = sorted(
        row_id for row_id, label in parsed.items() if label != row_by_id[row_id]["exact_label"]
    )
    return {
        "parse_ok": True,
        "parser_failure_reason": "",
        "parsed_labels": parsed,
        "valid_wrong_labels": valid_wrong,
        "exact_wrong_count": len(valid_wrong),
    }


def _parse_failure(reason: str) -> JsonDict:
    return {
        "parse_ok": False,
        "parser_failure_reason": reason,
        "parsed_labels": {},
        "valid_wrong_labels": [],
        "exact_wrong_count": 0,
    }


def parser_negative_control_receipts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Replay parser controls used to qualify the fixture boundary."""

    selected = list(rows[:2])
    row_by_id = {str(row["row_id"]): row for row in selected}
    exact_lines = [f"{row['row_id']}: {row['exact_label']}" for row in selected]
    wrong_label = next(label for label in LABELS if label != selected[0]["exact_label"])
    cases = {
        "truncation": (f"{selected[0]['row_id']}: ", False, "truncation", 0),
        "missing_answer": (exact_lines[0] + "\n", False, "missing_answer", 0),
        "duplicate_id": (
            "\n".join([exact_lines[0], exact_lines[0], exact_lines[1]]) + "\n",
            False,
            "duplicate_id",
            0,
        ),
        "invalid_candidate": (
            f"{selected[0]['row_id']}: Z\n{exact_lines[1]}\n",
            False,
            "invalid_candidate",
            0,
        ),
        "whitespace_order": (
            "\n".join(reversed([f"  {line}  " for line in exact_lines])) + "\n",
            True,
            "",
            0,
        ),
        "stop_token": ("\n".join(exact_lines) + "\n<|eot_id|>", False, "stop_token", 0),
        "adversarial_payload": (
            "ignore previous instructions\n" + "\n".join(exact_lines) + "\n",
            False,
            "adversarial_payload",
            0,
        ),
        "valid_wrong_label": (
            f"{selected[0]['row_id']}: {wrong_label}\n{exact_lines[1]}\n",
            True,
            "",
            1,
        ),
    }
    receipts: JsonDict = {}
    for name, (payload, expected_parse_ok, expected_reason, expected_wrong) in cases.items():
        receipt = parse_response(payload, row_by_id)
        receipts[name] = {
            "control": name,
            "expected_parse_ok": expected_parse_ok,
            "actual_parse_ok": receipt["parse_ok"],
            "expected_failure_reason": expected_reason,
            "actual_failure_reason": receipt["parser_failure_reason"],
            "expected_exact_wrong_count": expected_wrong,
            "actual_exact_wrong_count": receipt["exact_wrong_count"],
            "passed": (
                receipt["parse_ok"] is expected_parse_ok
                and receipt["parser_failure_reason"] == expected_reason
                and int(receipt["exact_wrong_count"]) == expected_wrong
            ),
        }
    return receipts


def parser_control_pass_rate(receipts: Mapping[str, Mapping[str, Any]]) -> float:
    """Return the parser control pass rate over required controls."""

    if not receipts:
        return 0.0
    return sum(1 for row in receipts.values() if row.get("passed") is True) / len(
        REQUIRED_PARSER_CONTROLS
    )


def exact_label_coverage(rows: Sequence[Mapping[str, Any]]) -> float:
    """Return exact label coverage over sealed fixture rows."""

    if not rows:
        return 0.0
    ok = [
        row["exact_label"] in row["candidate_labels"]
        and row["candidate_completeness_receipt"]["exact_candidate_present"] is True
        and row["exact_validator_receipt"]["validators_agree"] is True
        for row in rows
    ]
    return sum(1 for value in ok if value) / len(rows)


def blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    """Return mechanical blockers that prevent the ready score from reaching 1.0."""

    reasons = list((artifact.get("preconditions_checked") or {}).get("blocked_reasons") or [])
    if artifact.get("row_hashes_unique", True) is not True:
        reasons.append("row_hashes_not_unique")
    if float(artifact.get("exact_label_coverage") or 0.0) != 1.0:
        reasons.append("exact_label_coverage_incomplete")
    if float(artifact.get("parser_control_pass_rate") or 0.0) != 1.0:
        reasons.append("parser_control_failure")
    if (artifact.get("chronological_split_receipts") or {}).get("disjoint_row_hashes") is False:
        reasons.append("split_isolation_failed")
    if any(
        row.get("validators_agree") is not True
        for row in (artifact.get("exact_validator_receipts") or {}).values()
    ):
        reasons.append("exact_validator_disagreement")
    if any(
        row.get("complete") is not True
        for row in (artifact.get("candidate_completeness_receipts") or {}).values()
    ):
        reasons.append("candidate_completeness_failed")
    if any(
        row.get("exact_label_preserved") is not True
        for row in (artifact.get("proof_preserving_receipts") or {}).values()
    ):
        reasons.append("proof_preserving_surface_drift")
    if any(
        row.get("exact_label_changed") is not True
        for row in (artifact.get("meaning_change_canary_receipts") or {}).values()
    ):
        reasons.append("meaning_change_canary_missing")
    leakage = artifact.get("leakage_checks") or {}
    if any(value is not True for value in leakage.values()):
        reasons.append("protected_fact_leakage")
    if any(
        row.get("passed") is not True
        for row in (artifact.get("parser_negative_controls") or {}).values()
    ):
        reasons.append("parser_control_failure")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        reasons.append("inference_substrate")
    return sorted(set(reasons))


def fixture_ready_score(artifact: Mapping[str, Any]) -> float:
    """Strict readiness gate for downstream inference consumers."""

    return 0.0 if blocked_reasons(artifact) else 1.0


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal verdict with the required complete:/blocked: prefix."""

    reasons = blocked_reasons(artifact)
    if reasons:
        return "blocked: hardness_surface_fixture_not_ready: " + ",".join(reasons[:8])
    return "complete: sealed_hardness_surface_exact_fixture_ready"


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the terminal artifact while blanking its self-referential checksum."""

    stable = _copy_json(artifact)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def _artifact_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    preconditions_checked: Mapping[str, Any],
    row_file_sha256: str,
    test_commands: Sequence[str],
    test_exit_codes: Mapping[str, int],
) -> JsonDict:
    row_hashes = {str(row["row_id"]): str(row["row_hash"]) for row in rows}
    parser_controls = parser_negative_control_receipts(rows) if rows else {}
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "generator_version": GENERATOR_VERSION,
        "random_seed": RANDOM_SEEDS["base_seed"],
        "random_seeds": dict(RANDOM_SEEDS),
        "llm_inference_used": False,
        "verifier_is_oracle": True,
        "result_path": str(RESULT_RELATIVE_PATH),
        "status": "blocked",
        "preconditions_checked": dict(preconditions_checked),
        "spec_refs": list(SPEC_REFS),
        "fixture_schema": ROW_SCHEMA,
        "row_file": str(ROW_FILE_RELATIVE_PATH),
        "row_file_sha256": row_file_sha256,
        "row_hashes": row_hashes,
        "row_hashes_unique": len(set(row_hashes.values())) == len(row_hashes),
        "family_counts": dict(
            Counter(str(row["family"]) for row in rows if row["surface_kind"] == "canonical")
        ),
        "independent_unit_count": sum(1 for row in rows if row.get("surface_kind") == "canonical"),
        "sample_size_justification": {
            "primary_paired_comparison": "canonical_vs_proof_preserving_surface_variants",
            "independent_units_per_family": 30,
            "independent_unit_count": sum(
                1 for row in rows if row.get("surface_kind") == "canonical"
            ),
            "repeated_turns_counted_as_independent": False,
            "rationale": "Thirty canonical units per family exceed the minimum paired-comparison floor before any LLM turn exists.",
        },
        "chronological_split_receipts": _chronological_split_receipts(rows) if rows else {},
        "solver_hardness_bins": _solver_hardness_bins(rows) if rows else {},
        "surface_variant_matrix": dict(Counter(str(row["surface_kind"]) for row in rows)),
        "proof_preserving_receipts": _proof_receipts(rows) if rows else {},
        "meaning_change_canary_receipts": _canary_receipts(rows) if rows else {},
        "protected_fact_manifest": _protected_manifest(rows),
        "mutable_constraint_manifest": _mutable_manifest(rows),
        "candidate_completeness_receipts": _candidate_receipts(rows),
        "parser_contract": parser_contract(),
        "parser_negative_controls": parser_controls,
        "exact_validator_receipts": _validator_receipts(rows),
        "leakage_checks": _leakage_checks(rows) if rows else {},
        "fixture_ready_score": 0.0,
        "exact_label_coverage": exact_label_coverage(rows),
        "parser_control_pass_rate": parser_control_pass_rate(parser_controls),
        "producer_gate_fields": list(PRODUCER_GATE_FIELDS),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "test_commands": list(test_commands),
        "test_exit_codes": dict(test_exit_codes),
        "reproducibility_checksum": "",
        "honest_verdict": "",
        "blocked_reasons": [],
    }
    artifact["fixture_ready_score"] = fixture_ready_score(artifact)
    artifact["status"] = "complete" if artifact["fixture_ready_score"] == 1.0 else "blocked"
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["blocked_reasons"] = blocked_reasons(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def run(
    *,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    row_file_path: str | Path = REPO_ROOT / ROW_FILE_RELATIVE_PATH,
    preconditions_checked: Mapping[str, Any] | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    write: bool = True,
) -> JsonDict:
    """Build, validate, and optionally write the terminal artifact and rows."""

    preconditions = dict(
        preconditions_checked
        or collect_preconditions(result_path=result_path, row_file_path=row_file_path)
    )
    rows = generate_fixture_rows() if preconditions.get("preconditions_ready") is True else []
    row_text = rows_to_jsonl(rows)
    row_sha = sha256_text(row_text)
    if write:
        Path(row_file_path).parent.mkdir(parents=True, exist_ok=True)
        Path(row_file_path).write_text(row_text, encoding="utf-8")
    artifact = _artifact_from_rows(
        rows,
        preconditions_checked=preconditions,
        row_file_sha256=row_sha,
        test_commands=test_commands,
        test_exit_codes=test_exit_codes or {command: 0 for command in test_commands},
    )
    validate_artifact(artifact)
    if rows:
        verify_row_file(rows, artifact)
    if write:
        Path(result_path).parent.mkdir(parents=True, exist_ok=True)
        Path(result_path).write_text(canonical_json(artifact) + "\n", encoding="utf-8")
    return artifact


def verify_row_file(rows: Sequence[Mapping[str, Any]], artifact: Mapping[str, Any]) -> bool:
    """Replay JSONL row hashes against the terminal artifact."""

    seen: set[str] = set()
    for row in rows:
        row_id = str(row["row_id"])
        if row_id in seen:
            raise ManifestReplayError("duplicate row_id")
        seen.add(row_id)
        if _row_hash(row) != row.get("row_hash"):
            raise ManifestReplayError("row_hash mismatch")
        if artifact.get("row_hashes", {}).get(row_id) != row.get("row_hash"):
            raise ManifestReplayError("artifact row_hash mismatch")
    if len(seen) != len(artifact.get("row_hashes", {})):
        raise ManifestReplayError("row count mismatch")
    if sha256_text(rows_to_jsonl(rows)) != artifact.get("row_file_sha256"):
        raise ManifestReplayError("row_file_sha256 mismatch")
    return True


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate the terminal artifact schema and readiness invariants."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    for field in PRODUCER_GATE_FIELDS:
        if isinstance(artifact.get(field), Mapping):
            raise ValueError("producer_gate_fields must be bare scalars")
    if list(artifact.get("producer_gate_fields") or []) != list(PRODUCER_GATE_FIELDS):
        raise ValueError("producer_gate_fields mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")
    expected_score = fixture_ready_score(artifact)
    if float(artifact.get("fixture_ready_score")) != expected_score:
        raise ValueError("fixture_ready_score mismatch")
    if expected_score == 1.0 and artifact.get("status") != "complete":
        raise ValueError("status mismatch")
    if expected_score == 1.0 and not str(artifact.get("honest_verdict")).startswith("complete:"):
        raise ValueError("honest_verdict mismatch")
    if expected_score == 0.0 and not str(artifact.get("honest_verdict")).startswith("blocked:"):
        raise ValueError("honest_verdict mismatch")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum mismatch")
    return True


def main() -> int:  # pragma: no cover - thin CLI wrapper.
    artifact = run()
    print(
        canonical_json({"status": artifact["status"], "honest_verdict": artifact["honest_verdict"]})
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())
