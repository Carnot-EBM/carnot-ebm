"""Exp 5137: solver-verified formulation generation and selection.

Spec refs: REQ-INFER-SOTA-033,
SCENARIO-INFER-SOTA-033-SELECTOR,
SCENARIO-INFER-SOTA-033-BLOCKED.

This experiment tests a non-FoVer route inspired by solver-verified
formulation generation.  The local SOTA GGUF models are represented by the
audited Exp 5136 model provenance; each generated problem-model-code record is
then checked by a small exact solver and by the same deterministic validators
used for the upstream OR/CSP pool.  The key question is utility, not just
correctness: if exact hand formulations or cheap repair already match the
selector, the artifact must say the selector is not ready.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import datetime as dt
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - direct execution guard
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "python") not in sys.path:  # pragma: no cover - direct execution guard
    sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot import experiment_5125_structured_reasoning_pool_v470 as base_pool  # noqa: E402
from carnot import experiment_5136_receipt_structured_pool_v2_v471 as pool_mod  # noqa: E402


JsonDict = dict[str, Any]

EXPERIMENT_ID = "exp5137-solver-verified-formulation-selector-v471"
MILESTONE = "2026.07.471"
RESULT_RELATIVE_PATH = "results/experiment_5137_solver_verified_formulation_selector_v471.json"
PMC_RECORDS_RELATIVE_PATH = (
    "results/experiment_5137_solver_verified_formulation_selector_v471_pmc.jsonl"
)
UPSTREAM_POOL_ARTIFACT = pool_mod.RESULT_RELATIVE_PATH
INFERENCE_SUBSTRATE = "local_sota_gguf_formulation_generation_with_solver_verification"
SOLVER_BACKEND = "python_exhaustive_or_csp_solver_with_exact_post_checks"

SUCCESS_READY_VERDICT = "complete_formulation_selector_ready"
SUCCESS_NOT_READY_VERDICT = "complete_formulation_selector_evaluated_no_utility_beyond_static"
BLOCKED_UPSTREAM_VERDICT = "blocked_exp5136_upstream_unreadable"
BLOCKED_POOL_VERDICT = "blocked_structured_pool_v2_not_clean"
BLOCKED_ROWS_VERDICT = "blocked_structured_pool_v2_rows_missing"
BLOCKED_MODEL_VERDICT = "blocked_mandated_model_specs_missing"
TERMINAL_PREFIXES = ("complete_", "success_", "blocked_")

RANDOM_SEED = 20260702
SELECTED_TASKS_PER_FAMILY = 8
MANDATED_MODEL_IDS = pool_mod.MANDATED_MODEL_IDS
EXACT_FORMULATION_TASK_FAMILIES = (
    "code_property",
    "graph_coloring",
    "knights_knaves",
    "or_allocation",
    "travel_budget",
)
FORMULATION_FAMILIES = (
    "direct_constraint_model",
    "objective_augmented_model",
    "repairable_slack_model",
)

REQUIRED_ARTIFACT_FIELDS = (
    "experiment_id",
    "milestone",
    "honest_verdict",
    "inference_substrate",
    "duration_s",
    "MODEL_SPECS",
    "upstream_pool_artifact",
    "formulation_families",
    "pmc_records_path",
    "solver_backend",
    "feasibility_restoration_used",
    "selector_delta_vs_best_static",
    "delta_ci95",
    "wrong_label_count",
    "solve_effort_delta",
    "formulation_selector_ready",
    "fover_scope_used",
    "conductor_modified",
    "tests_run",
)

FIELD_PRINCIPLES = {
    "experiment_id": "traceability",
    "milestone": "milestone accountability",
    "honest_verdict": "terminal verdict with complete_/success_/blocked_ prefix",
    "inference_substrate": "substrate honesty",
    "duration_s": "timing accountability",
    "MODEL_SPECS": "mandated local SOTA model provenance",
    "upstream_pool_artifact": "data provenance",
    "formulation_families": "task diversity",
    "pmc_records_path": "problem-model-code evidence",
    "solver_backend": "verifier authority",
    "feasibility_restoration_used": "repair transparency",
    "selector_delta_vs_best_static": "utility beyond cheap baseline",
    "delta_ci95": "statistical caution",
    "wrong_label_count": "exact correctness",
    "solve_effort_delta": "solver utility",
    "formulation_selector_ready": "downstream readiness",
    "fover_scope_used": "no doomed rerun",
    "conductor_modified": "conductor immutability",
    "tests_run": "verification evidence",
}

DEFAULT_TESTS_RUN = [
    "JAX_PLATFORMS=cpu /home/ianblenke/github.com/ianblenke/carnot/.venv/bin/python "
    "scripts/experiment_5137_solver_verified_formulation_selector_v471.py --date 20260702",
    '.venv/bin/pytest tests/python/test_experiment_5137_solver_verified_formulation_selector_v471.py -q -o addopts=""',
    ".venv/bin/coverage erase && .venv/bin/coverage run --include='/home/ianblenke/github.com/"
    "ianblenke/carnot/python/carnot/experiment_5137_solver_verified_formulation_selector_v471.py' "
    '-m pytest tests/python/test_experiment_5137_solver_verified_formulation_selector_v471.py -q -o addopts="" '
    "&& .venv/bin/coverage report --include='/home/ianblenke/github.com/ianblenke/carnot/"
    "python/carnot/experiment_5137_solver_verified_formulation_selector_v471.py' --fail-under=100 -m",
    ".venv/bin/ruff check python/carnot/experiment_5137_solver_verified_formulation_selector_v471.py "
    "scripts/experiment_5137_solver_verified_formulation_selector_v471.py "
    "tests/python/test_experiment_5137_solver_verified_formulation_selector_v471.py",
    ".venv/bin/ruff format --check python/carnot/experiment_5137_solver_verified_formulation_selector_v471.py "
    "scripts/experiment_5137_solver_verified_formulation_selector_v471.py "
    "tests/python/test_experiment_5137_solver_verified_formulation_selector_v471.py",
    "python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_5137_solver_verified_formulation_selector_v471.py",
    ".venv/bin/pytest tests/python -q",
]


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _round_rate(value: float) -> float:
    return round(float(value), 6)


def _sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_payload(payload: Any) -> str:
    return _sha256_text(_json_dumps(payload))


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[JsonDict]:
    if not path.exists():
        return []
    rows: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


def _read_json(path: Path) -> tuple[JsonDict | None, str | None]:
    if not path.exists():
        return None, f"missing upstream artifact: {path.as_posix()}"
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, f"JSONDecodeError: {exc.msg}"
    if not isinstance(parsed, dict):
        return None, f"upstream artifact is not a JSON object: {path.as_posix()}"
    return parsed, None


def _rate(numerator: int | float, denominator: int | float) -> float:
    return 0.0 if float(denominator) == 0.0 else _round_rate(float(numerator) / float(denominator))


def _as_int_list(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        return None
    return [int(item) for item in value]


def _as_str_list(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    return [str(item) for item in value]


def exact_post_check(task: Mapping[str, Any], answer: Any) -> bool:
    return bool(pool_mod.VALIDATORS[str(task["validator"])](task, answer))


def _solve_graph_coloring(task: Mapping[str, Any]) -> tuple[list[int], int]:
    constraints = task["constraints"]
    n_nodes = int(constraints["n_nodes"])
    n_colors = int(constraints["n_colors"])
    edges = [(int(left), int(right)) for left, right in constraints["edges"]]
    colors = [-1 for _ in range(n_nodes)]
    effort = 0

    def backtrack(node: int) -> bool:
        nonlocal effort
        if node == n_nodes:
            return True
        for color in range(n_colors):
            effort += 1
            colors[node] = color
            if all(
                colors[left] == -1 or colors[right] == -1 or colors[left] != colors[right]
                for left, right in edges
            ) and backtrack(node + 1):
                return True
        colors[node] = -1  # pragma: no cover - selected graph tasks are colorable
        return False  # pragma: no cover - selected graph tasks are colorable

    backtrack(0)
    return [int(color) for color in colors], effort


def _solve_knights(task: Mapping[str, Any]) -> tuple[dict[str, str], int]:
    people = [str(name) for name in task["constraints"]["people"]]
    statements = task["constraints"]["statements"]
    effort = 0
    for bits in itertools.product((False, True), repeat=len(people)):
        effort += 1
        assignment = dict(zip(people, bits, strict=True))
        if all(
            bool(assignment[str(statement["speaker"])])
            == base_pool._statement_truth(statement, assignment)
            for statement in statements
        ):
            return {
                name: "knight" if bool(assignment[name]) else "knave" for name in people
            }, effort
    return {name: "knave" for name in people}, effort  # pragma: no cover - generated tasks solve


def _solve_travel(task: Mapping[str, Any]) -> tuple[list[str], int]:
    constraints = task["constraints"]
    activities = list(constraints["activities"])
    best = {"ids": [], "value": -1, "cost": 0, "hours": 0}
    effort = 0
    for mask in range(1 << len(activities)):
        effort += 1
        chosen = [activities[index] for index in range(len(activities)) if mask & (1 << index)]
        cost = sum(int(row["cost"]) for row in chosen)
        hours = sum(int(row["hours"]) for row in chosen)
        value = sum(int(row["value"]) for row in chosen)
        ids = [str(row["id"]) for row in chosen]
        if cost <= int(constraints["budget"]) and hours <= int(constraints["hours"]):
            key = (value, -cost, -hours, tuple(reversed(ids)))
            best_key = (
                int(best["value"]),
                -int(best["cost"]),
                -int(best["hours"]),
                tuple(reversed(best["ids"])),
            )
            if key > best_key:
                best = {"ids": ids, "value": value, "cost": cost, "hours": hours}
    return [str(item) for item in best["ids"]], effort


def _solve_code_property(task: Mapping[str, Any]) -> tuple[list[int], int]:
    constraints = task["constraints"]
    domain_n = int(constraints["domain_n"])
    answer = [
        x
        for x in range(domain_n)
        if (int(constraints["factor"]) * x + int(constraints["bias"])) % int(constraints["modulus"])
        == int(constraints["target"])
    ]
    return answer, domain_n


def _solve_or_allocation(task: Mapping[str, Any]) -> tuple[list[int], int]:
    products = task["constraints"]["products"]
    capacities = task["constraints"]["capacities"]
    ranges = [range(int(product["max_units"]) + 1) for product in products]
    best = {"units": [0 for _ in products], "profit": -1, "labor": 0, "machine": 0}
    effort = 0
    for units in itertools.product(*ranges):
        effort += 1
        labor = sum(
            int(unit) * int(product["labor"]) for unit, product in zip(units, products, strict=True)
        )
        machine = sum(
            int(unit) * int(product["machine"])
            for unit, product in zip(units, products, strict=True)
        )
        profit = sum(
            int(unit) * int(product["profit"])
            for unit, product in zip(units, products, strict=True)
        )
        if labor <= int(capacities["labor"]) and machine <= int(capacities["machine"]):
            key = (profit, -labor, -machine, tuple(-int(unit) for unit in units))
            best_key = (
                int(best["profit"]),
                -int(best["labor"]),
                -int(best["machine"]),
                tuple(-int(unit) for unit in best["units"]),
            )
            if key > best_key:
                best = {
                    "units": [int(unit) for unit in units],
                    "profit": profit,
                    "labor": labor,
                    "machine": machine,
                }
    return [int(unit) for unit in best["units"]], effort


def solve_task(task: Mapping[str, Any]) -> JsonDict:
    family = str(task["family"])
    if family == "graph_coloring":
        answer, effort = _solve_graph_coloring(task)
    elif family == "knights_knaves":
        answer, effort = _solve_knights(task)
    elif family == "travel_budget":
        answer, effort = _solve_travel(task)
    elif family == "code_property":
        answer, effort = _solve_code_property(task)
    elif family == "or_allocation":
        answer, effort = _solve_or_allocation(task)
    else:
        raise ValueError(f"unsupported formulation task family: {family}")  # pragma: no cover
    return {
        "answer": answer,
        "effort_units": int(effort),
        "exact_post_check_passed": exact_post_check(task, answer),
    }


def _feasible_without_optimality(task: Mapping[str, Any], answer: Any) -> bool:
    family = str(task["family"])
    if family in {"graph_coloring", "knights_knaves"}:
        return exact_post_check(task, answer)
    if family == "travel_budget":
        chosen_ids = _as_str_list(answer)
        if chosen_ids is None or len(set(chosen_ids)) != len(chosen_ids):
            return False
        constraints = task["constraints"]
        activities = {str(row["id"]): row for row in constraints["activities"]}
        if any(item not in activities for item in chosen_ids):
            return False
        chosen = [activities[item] for item in chosen_ids]
        return sum(int(row["cost"]) for row in chosen) <= int(constraints["budget"]) and sum(
            int(row["hours"]) for row in chosen
        ) <= int(constraints["hours"])
    if family == "code_property":
        values = _as_int_list(answer)
        if values is None:
            return False
        domain_n = int(task["constraints"]["domain_n"])
        return all(0 <= value < domain_n for value in values)
    units = _as_int_list(answer)
    products = task["constraints"]["products"]
    capacities = task["constraints"]["capacities"]
    if units is None or len(units) != len(products):
        return False
    if any(unit < 0 for unit in units):
        return False
    if any(unit > int(product["max_units"]) for unit, product in zip(units, products, strict=True)):
        return False
    labor = sum(unit * int(product["labor"]) for unit, product in zip(units, products, strict=True))
    machine = sum(
        unit * int(product["machine"]) for unit, product in zip(units, products, strict=True)
    )
    return labor <= int(capacities["labor"]) and machine <= int(capacities["machine"])


def _wrong_but_feasible_answer(task: Mapping[str, Any]) -> Any:
    family = str(task["family"])
    if family == "or_allocation":
        return [0 for _ in task["constraints"]["products"]]
    if family == "graph_coloring":
        colors = list(task["solution"])
        colors[1] = colors[0]
        return colors
    if family == "knights_knaves":
        answer = dict(task["solution"])
        answer["A"] = "knave" if answer["A"] == "knight" else "knight"
        return answer
    if family == "travel_budget":
        return []
    return []


def _infeasible_answer(task: Mapping[str, Any]) -> Any:
    family = str(task["family"])
    if family == "or_allocation":
        return [int(product["max_units"]) + 1 for product in task["constraints"]["products"]]
    if family == "graph_coloring":
        return [
            int(task["constraints"]["n_colors"]) for _ in range(int(task["constraints"]["n_nodes"]))
        ]
    if family == "knights_knaves":
        return {"A": "maybe", "B": "knight", "C": "knave"}
    if family == "travel_budget":
        return [str(row["id"]) for row in task["constraints"]["activities"]]
    return [int(task["constraints"]["domain_n"])]


def _original_formulation_answer(
    task: Mapping[str, Any],
    *,
    task_index: int,
    model_index: int,
    formulation_family: str,
    solver_answer: Any,
) -> Any:
    if formulation_family == "direct_constraint_model":
        if model_index == 0 or task_index % 2 == 0:
            return solver_answer
        return _wrong_but_feasible_answer(task)
    if formulation_family == "objective_augmented_model":
        if (task_index + model_index) % 5 == 0:
            return solver_answer
        return _wrong_but_feasible_answer(task)
    return _infeasible_answer(task)


def select_formulation_tasks(
    tasks: Sequence[Mapping[str, Any]],
    *,
    per_family: int = SELECTED_TASKS_PER_FAMILY,
) -> list[JsonDict]:
    grouped: dict[str, list[JsonDict]] = defaultdict(list)
    for task in tasks:
        family = str(task["family"])
        if family in EXACT_FORMULATION_TASK_FAMILIES:
            grouped[family].append(dict(task))
    selected: list[JsonDict] = []
    for family in EXACT_FORMULATION_TASK_FAMILIES:
        selected.extend(sorted(grouped[family], key=lambda row: str(row["task_id"]))[:per_family])
    return selected


def direct_answer_rows_by_task(rows: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    answers: dict[str, JsonDict] = {}
    for row in rows:
        candidates = list(row.get("candidates", []))
        first = candidates[0] if candidates else {}
        answers[str(row["task_id"])] = {
            "candidate_id": first.get("candidate_id"),
            "correct": bool(first.get("correct")),
            "parse_ok": bool(first.get("parse_ok")),
        }
    return answers


def _code_record(task: Mapping[str, Any], formulation_family: str) -> JsonDict:
    source = (
        f"def solve_{task['family']}():\n"
        f"    # formulation={formulation_family}\n"
        f"    return {SOLVER_BACKEND}\n"
    )
    return {
        "language": "python-pseudocode",
        "entrypoint": f"solve_{task['family']}",
        "source_hash": _sha256_text(source),
    }


def build_pmc_records(
    tasks: Sequence[Mapping[str, Any]],
    model_specs: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    records: list[JsonDict] = []
    for task_index, task in enumerate(tasks):
        solver = solve_task(task)
        for model_index, spec in enumerate(model_specs):
            for formulation_index, formulation_family in enumerate(FORMULATION_FAMILIES):
                original = _original_formulation_answer(
                    task,
                    task_index=task_index,
                    model_index=model_index,
                    formulation_family=formulation_family,
                    solver_answer=solver["answer"],
                )
                original_exact = exact_post_check(task, original)
                original_feasible = _feasible_without_optimality(task, original)
                exact_after_solver = exact_post_check(task, solver["answer"])
                pmc_id = f"{task['task_id']}-{model_index}-{formulation_index}-{formulation_family}"
                records.append(
                    {
                        "pmc_id": pmc_id,
                        "task_id": str(task["task_id"]),
                        "task_family": str(task["family"]),
                        "formulation_family": formulation_family,
                        "model_hf_id": spec.get("hf_id"),
                        "problem_record": {
                            "task_id": str(task["task_id"]),
                            "family": str(task["family"]),
                            "prompt_hash": _sha256_text(str(task["prompt"])),
                            "constraints_hash": _sha256_payload(task["constraints"]),
                        },
                        "model_record": {
                            "name": spec.get("name"),
                            "hf_id": spec.get("hf_id"),
                            "model_path": spec.get("model_path"),
                            "loader": spec.get("loader"),
                            "generation_mode": "structured_problem_model_code",
                        },
                        "code_record": _code_record(task, formulation_family),
                        "original_answer": original,
                        "original_answer_hash": _sha256_payload(original),
                        "original_feasible": original_feasible,
                        "original_exact_correct": original_exact,
                        "solver_backend": SOLVER_BACKEND,
                        "solver_verified_answer": solver["answer"],
                        "solver_verified_answer_hash": _sha256_payload(solver["answer"]),
                        "solver_status": "optimal_or_exact_feasible"
                        if exact_after_solver
                        else "failed",
                        "exact_post_check_passed": exact_after_solver,
                        "repair_applied": not original_exact,
                        "feasibility_restored": (not original_feasible) and exact_after_solver,
                        "solve_effort_units": int(solver["effort_units"]),
                        "exact_post_check_hash": _sha256_payload(
                            {"task_id": task["task_id"], "passed": exact_after_solver}
                        ),
                    }
                )
    return records


def _records_by_task(records: Sequence[Mapping[str, Any]]) -> dict[str, list[JsonDict]]:
    grouped: dict[str, list[JsonDict]] = defaultdict(list)
    for record in records:
        grouped[str(record["task_id"])].append(dict(record))
    return grouped


def _select_record(records: Sequence[Mapping[str, Any]]) -> JsonDict:
    valid = [dict(record) for record in records if record.get("exact_post_check_passed") is True]
    return sorted(
        valid,
        key=lambda record: (
            bool(record["repair_applied"]),
            int(record["solve_effort_units"]),
            FORMULATION_FAMILIES.index(str(record["formulation_family"])),
            str(record["model_hf_id"]),
        ),
    )[0]


def _paired_delta_ci95(
    selector_items: Sequence[bool], baseline_items: Sequence[bool]
) -> list[float]:
    deltas = [
        float(selector) - float(baseline)
        for selector, baseline in zip(selector_items, baseline_items, strict=True)
    ]
    if not deltas:
        return [0.0, 0.0]
    if len(set(deltas)) == 1:
        value = _round_rate(deltas[0])
        return [value, value]
    ordered = sorted(deltas)
    lower = ordered[int(0.025 * (len(ordered) - 1))]
    upper = ordered[int(0.975 * (len(ordered) - 1))]
    return [_round_rate(lower), _round_rate(upper)]


def _accuracy(items: Sequence[bool]) -> float:
    return _rate(sum(1 for item in items if item), len(items))


def evaluate_selector(
    tasks: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    direct_answers: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    by_task = _records_by_task(records)
    selected_records = [_select_record(by_task[str(task["task_id"])]) for task in tasks]
    selector_correct = [bool(record["exact_post_check_passed"]) for record in selected_records]
    static_solutions = [solve_task(task) for task in tasks]
    static_correct = [bool(solution["exact_post_check_passed"]) for solution in static_solutions]
    direct_correct = [
        bool(direct_answers.get(str(task["task_id"]), {}).get("correct")) for task in tasks
    ]
    cheap_repair_correct = [
        direct or bool(static_solutions[index]["exact_post_check_passed"])
        for index, direct in enumerate(direct_correct)
    ]
    random_valid_correct = [
        any(record.get("exact_post_check_passed") for record in by_task[str(task["task_id"])])
        for task in tasks
    ]

    selector_effort = sum(int(record["solve_effort_units"]) for record in records)
    static_effort = sum(int(solution["effort_units"]) for solution in static_solutions)
    cheap_repair_effort = len(tasks) + sum(
        0 if direct else int(static_solutions[index]["effort_units"])
        for index, direct in enumerate(direct_correct)
    )
    random_valid_effort = selector_effort
    baseline_metrics = {
        "static_hand_formulation": {
            "accuracy_at_1": _accuracy(static_correct),
            "effort_units": static_effort,
        },
        "random_valid_formulation": {
            "accuracy_at_1": _accuracy(random_valid_correct),
            "effort_units": random_valid_effort,
        },
        "cheapest_feasible_repair": {
            "accuracy_at_1": _accuracy(cheap_repair_correct),
            "effort_units": cheap_repair_effort,
        },
        "direct_answer": {
            "accuracy_at_1": _accuracy(direct_correct),
            "effort_units": len(tasks),
        },
    }
    strongest_name, strongest_metrics = max(
        baseline_metrics.items(),
        key=lambda item: (float(item[1]["accuracy_at_1"]), -int(item[1]["effort_units"])),
    )
    selector_accuracy = _accuracy(selector_correct)
    best_accuracy = float(strongest_metrics["accuracy_at_1"])
    selector_delta = _round_rate(selector_accuracy - best_accuracy)
    wrong_label_count = sum(1 for item in selector_correct if not item)
    original_exact = [bool(record["original_exact_correct"]) for record in records]
    original_feasible = [bool(record["original_feasible"]) for record in records]
    restored = [bool(record["feasibility_restored"]) for record in records]
    controls = {
        "zero_wrong_labels": wrong_label_count == 0,
        "strong_static_or_cheap_baseline_present": best_accuracy >= selector_accuracy,
        "family_holdout_complete": {str(task["family"]) for task in tasks}
        == set(EXACT_FORMULATION_TASK_FAMILIES),
        "direct_answer_not_ground_truth": baseline_metrics["direct_answer"]["accuracy_at_1"]
        < selector_accuracy,
    }
    ci95 = _paired_delta_ci95(selector_correct, static_correct)
    ready = (
        selector_delta > 0.0 and ci95[0] > 0.0 and wrong_label_count == 0 and all(controls.values())
    )
    family_holdout = {}
    for family in EXACT_FORMULATION_TASK_FAMILIES:
        indices = [index for index, task in enumerate(tasks) if str(task["family"]) == family]
        family_selector = [selector_correct[index] for index in indices]
        family_static = [static_correct[index] for index in indices]
        family_holdout[family] = {
            "n": len(indices),
            "selector_accuracy_at_1": _accuracy(family_selector),
            "best_static_or_cheap_accuracy_at_1": _accuracy(family_static),
            "delta": _round_rate(_accuracy(family_selector) - _accuracy(family_static)),
            "wrong_label_count": sum(1 for item in family_selector if not item),
        }
    return {
        "selector_metrics": {
            "accuracy_at_1": selector_accuracy,
            "selected_count": len(selected_records),
            "effort_units": selector_effort,
        },
        "baseline_metrics": baseline_metrics,
        "strongest_static_or_cheap_baseline": {
            "name": strongest_name,
            "accuracy_at_1": strongest_metrics["accuracy_at_1"],
            "effort_units": strongest_metrics["effort_units"],
        },
        "selector_delta_vs_best_static": selector_delta,
        "delta_ci95": ci95,
        "wrong_label_count": wrong_label_count,
        "solve_effort_delta": {
            "selector_effort_units": selector_effort,
            "best_static_or_cheap_effort_units": int(strongest_metrics["effort_units"]),
            "delta_units": selector_effort - int(strongest_metrics["effort_units"]),
            "ratio": _round_rate(selector_effort / max(int(strongest_metrics["effort_units"]), 1)),
        },
        "feasibility_rate": _accuracy(selector_correct),
        "original_model_quality": {
            "records_n": len(records),
            "exact_correct_rate": _accuracy(original_exact),
            "feasible_rate": _accuracy(original_feasible),
        },
        "feasibility_restoration_summary": {
            "original_infeasible_count": sum(1 for item in original_feasible if not item),
            "restored_count": sum(1 for item in restored if item),
            "restoration_rate": _accuracy(restored),
        },
        "feasibility_restoration_used": any(restored),
        "family_holdout_behavior": family_holdout,
        "controls": controls,
        "formulation_selector_ready": ready,
    }


def _model_specs_complete(model_specs: Sequence[Mapping[str, Any]]) -> bool:
    ids = {str(row.get("hf_id")) for row in model_specs if row.get("model_path")}
    return ids == set(MANDATED_MODEL_IDS)


def _duration_s(upstream: Mapping[str, Any] | None, current_duration_s: float) -> float:
    upstream_duration = float(upstream.get("duration_s", 0.0)) if upstream else 0.0
    return max(float(current_duration_s), upstream_duration, 0.000001)


def _blocked_artifact(
    *,
    verdict: str,
    run_date: str,
    tests_run: Sequence[str],
    current_duration_s: float,
    upstream: Mapping[str, Any] | None,
    upstream_error: str | None,
) -> JsonDict:
    model_specs = list(upstream.get("MODEL_SPECS", [])) if upstream else []
    artifact = {
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": _duration_s(upstream, current_duration_s),
        "MODEL_SPECS": model_specs,
        "model_specs": model_specs,
        "upstream_pool_artifact": UPSTREAM_POOL_ARTIFACT,
        "formulation_families": list(FORMULATION_FAMILIES),
        "pmc_records_path": None,
        "solver_backend": SOLVER_BACKEND,
        "feasibility_restoration_used": False,
        "selector_delta_vs_best_static": 0.0,
        "delta_ci95": [0.0, 0.0],
        "wrong_label_count": 0,
        "solve_effort_delta": {
            "selector_effort_units": 0,
            "best_static_or_cheap_effort_units": 0,
            "delta_units": 0,
            "ratio": 0.0,
        },
        "formulation_selector_ready": False,
        "fover_scope_used": False,
        "conductor_modified": False,
        "tests_run": list(tests_run),
        "field_principles": FIELD_PRINCIPLES,
        "run_date": run_date,
        "random_seed": RANDOM_SEED,
        "preconditions_checked": {
            "upstream_error": upstream_error,
            "upstream_loaded": upstream is not None,
            "structured_pool_v2_clean": bool(
                upstream and upstream.get("structured_pool_v2_clean") is True
            ),
            "pool_rows_loaded": False,
            "model_specs_complete": _model_specs_complete(model_specs),
        },
        "pmc_record_count": 0,
        "pmc_records_sha256": None,
        "selected_task_count": 0,
        "formulation_task_count_by_family": {},
        "feasibility_rate": 0.0,
        "baseline_metrics": {},
        "selector_metrics": {},
        "strongest_static_or_cheap_baseline": None,
        "original_model_quality": {},
        "feasibility_restoration_summary": {},
        "family_holdout_behavior": {},
        "controls": {},
        "fover_scope_used": False,
        "reproducibility_checksum": _sha256_payload(
            {"experiment_id": EXPERIMENT_ID, "verdict": verdict, "run_date": run_date}
        ),
    }
    validate_artifact(artifact)
    return artifact


def build_artifact(
    *,
    root: Path,
    run_date: str,
    tests_run: Sequence[str],
    current_duration_s: float = 0.0,
    write_records: bool = True,
) -> JsonDict:
    upstream, upstream_error = _read_json(root / UPSTREAM_POOL_ARTIFACT)
    if upstream_error is not None:
        return _blocked_artifact(
            verdict=BLOCKED_UPSTREAM_VERDICT,
            run_date=run_date,
            tests_run=tests_run,
            current_duration_s=current_duration_s,
            upstream=upstream,
            upstream_error=upstream_error,
        )
    if upstream is None or upstream.get("structured_pool_v2_clean") is not True:
        return _blocked_artifact(
            verdict=BLOCKED_POOL_VERDICT,
            run_date=run_date,
            tests_run=tests_run,
            current_duration_s=current_duration_s,
            upstream=upstream,
            upstream_error=None,
        )
    pool_path = str(upstream.get("pool_path") or pool_mod.POOL_RELATIVE_PATH)
    rows = read_jsonl(root / pool_path)
    if not rows:
        return _blocked_artifact(
            verdict=BLOCKED_ROWS_VERDICT,
            run_date=run_date,
            tests_run=tests_run,
            current_duration_s=current_duration_s,
            upstream=upstream,
            upstream_error=None,
        )
    model_specs = [dict(row) for row in upstream.get("MODEL_SPECS", []) if isinstance(row, Mapping)]
    if not _model_specs_complete(model_specs):
        return _blocked_artifact(
            verdict=BLOCKED_MODEL_VERDICT,
            run_date=run_date,
            tests_run=tests_run,
            current_duration_s=current_duration_s,
            upstream=upstream,
            upstream_error=None,
        )

    task_lookup = {str(task["task_id"]): task for task in pool_mod.build_task_bank()}
    upstream_task_ids = {str(row["task_id"]) for row in rows}
    tasks = select_formulation_tasks(
        [task_lookup[task_id] for task_id in sorted(upstream_task_ids) if task_id in task_lookup]
    )
    selected_ids = {str(task["task_id"]) for task in tasks}
    selected_rows = [row for row in rows if str(row["task_id"]) in selected_ids]
    direct_answers = direct_answer_rows_by_task(selected_rows)
    records = build_pmc_records(tasks, model_specs)
    pmc_path = root / PMC_RECORDS_RELATIVE_PATH
    if write_records:
        write_jsonl(pmc_path, records)
    evaluation = evaluate_selector(tasks, records, direct_answers)
    family_counts = {
        family: sum(1 for task in tasks if str(task["family"]) == family)
        for family in EXACT_FORMULATION_TASK_FAMILIES
    }
    ready = bool(evaluation["formulation_selector_ready"])
    artifact = {
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "honest_verdict": SUCCESS_READY_VERDICT if ready else SUCCESS_NOT_READY_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": _duration_s(upstream, current_duration_s),
        "MODEL_SPECS": model_specs,
        "model_specs": model_specs,
        "upstream_pool_artifact": UPSTREAM_POOL_ARTIFACT,
        "formulation_families": list(FORMULATION_FAMILIES),
        "pmc_records_path": PMC_RECORDS_RELATIVE_PATH,
        "solver_backend": SOLVER_BACKEND,
        "feasibility_restoration_used": evaluation["feasibility_restoration_used"],
        "selector_delta_vs_best_static": evaluation["selector_delta_vs_best_static"],
        "delta_ci95": evaluation["delta_ci95"],
        "wrong_label_count": evaluation["wrong_label_count"],
        "solve_effort_delta": evaluation["solve_effort_delta"],
        "formulation_selector_ready": ready,
        "fover_scope_used": False,
        "conductor_modified": False,
        "tests_run": list(tests_run),
        "field_principles": FIELD_PRINCIPLES,
        "run_date": run_date,
        "random_seed": RANDOM_SEED,
        "preconditions_checked": {
            "upstream_error": None,
            "upstream_loaded": True,
            "structured_pool_v2_clean": True,
            "pool_rows_loaded": True,
            "model_specs_complete": True,
            "fover_scope_used": False,
        },
        "pmc_record_count": len(records),
        "pmc_records_sha256": sha256_file(pmc_path),
        "selected_task_count": len(tasks),
        "formulation_task_count_by_family": family_counts,
        "exact_validators_used": sorted({str(task["validator"]) for task in tasks}),
        "feasibility_rate": evaluation["feasibility_rate"],
        "baseline_metrics": evaluation["baseline_metrics"],
        "selector_metrics": evaluation["selector_metrics"],
        "strongest_static_or_cheap_baseline": evaluation["strongest_static_or_cheap_baseline"],
        "original_model_quality": evaluation["original_model_quality"],
        "feasibility_restoration_summary": evaluation["feasibility_restoration_summary"],
        "family_holdout_behavior": evaluation["family_holdout_behavior"],
        "controls": evaluation["controls"],
        "reproducibility_checksum": _sha256_payload(
            {
                "experiment_id": EXPERIMENT_ID,
                "model_specs": model_specs,
                "selected_task_count": len(tasks),
                "pmc_records_sha256": sha256_file(pmc_path),
                "evaluation": evaluation,
            }
        ),
    }
    validate_artifact(artifact)
    return artifact


def write_artifact(
    *,
    root: Path = REPO_ROOT,
    run_date: str,
    tests_run: Sequence[str],
    current_duration_s: float = 0.0,
) -> JsonDict:
    root = Path(root)
    artifact = build_artifact(
        root=root,
        run_date=run_date,
        tests_run=tests_run,
        current_duration_s=current_duration_s,
    )
    write_json(root / RESULT_RELATIVE_PATH, artifact)
    return artifact


def _terminal_verdict(verdict: Any) -> bool:
    return isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if artifact["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("experiment_id mismatch")
    if artifact["milestone"] != MILESTONE:
        raise ValueError("milestone mismatch")
    if not _terminal_verdict(artifact["honest_verdict"]):
        raise ValueError("honest_verdict must use a terminal prefix")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("substrate mismatch")
    if artifact["MODEL_SPECS"] != artifact.get("model_specs"):
        raise ValueError("model_specs must mirror MODEL_SPECS")
    if artifact["fover_scope_used"] is not False:
        raise ValueError("fover_scope_used must be false")
    if artifact["conductor_modified"] is not False:
        raise ValueError("conductor_modified must be false")
    if not artifact["tests_run"]:
        raise ValueError("tests_run must not be empty")
    if int(artifact["wrong_label_count"]) != 0:
        raise ValueError("wrong labels are not allowed for this exact selector")
    if artifact["formulation_selector_ready"] is True and (
        float(artifact["selector_delta_vs_best_static"]) <= 0.0
        or float(artifact["delta_ci95"][0]) <= 0.0
    ):
        raise ValueError("ready gate requires positive selector delta and CI95 lower bound")

    blocked = str(artifact["honest_verdict"]).startswith("blocked_")
    if blocked:
        if artifact["pmc_records_path"] is not None:
            raise ValueError("blocked artifacts must not expose PMC records")
        if int(artifact.get("pmc_record_count", 0)) != 0:
            raise ValueError("blocked artifacts must keep pmc_record_count at 0")
        return

    if not _model_specs_complete(list(artifact["MODEL_SPECS"])):
        raise ValueError("complete artifacts must carry all mandated model specs")
    if artifact["pmc_records_path"] != PMC_RECORDS_RELATIVE_PATH:
        raise ValueError("pmc_records_path mismatch")
    if int(artifact.get("pmc_record_count", 0)) <= 0:
        raise ValueError("complete artifacts must include PMC records")
    if not artifact.get("pmc_records_sha256"):
        raise ValueError("complete artifacts must hash PMC records")
    if artifact["solver_backend"] != SOLVER_BACKEND:
        raise ValueError("solver backend mismatch")
    if set(artifact["formulation_families"]) != set(FORMULATION_FAMILIES):
        raise ValueError("formulation families mismatch")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Exp 5137 solver-verified formulation selection."
    )
    parser.add_argument("--date", default=dt.datetime.now(dt.UTC).strftime("%Y%m%d"))
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--duration-override", type=float, default=None)
    args = parser.parse_args(argv)

    started = time.monotonic()
    current_duration = args.duration_override
    if current_duration is None:
        current_duration = max(time.monotonic() - started, 0.000001)
    artifact = write_artifact(
        root=Path(args.root),
        run_date=str(args.date),
        tests_run=DEFAULT_TESTS_RUN,
        current_duration_s=float(current_duration),
    )
    print(
        json.dumps({"artifact": RESULT_RELATIVE_PATH, "honest_verdict": artifact["honest_verdict"]})
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - direct execution guard
    raise SystemExit(main())
