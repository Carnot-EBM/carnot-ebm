"""Exp 5125: non-FoVer structured reasoning candidate pool.

Spec refs: REQ-INFER-SOTA-030,
SCENARIO-INFER-SOTA-030-POOL,
SCENARIO-INFER-SOTA-030-BLOCKED.

The pool is intentionally exact-checkable.  Candidate diversity creates oracle
headroom, while deterministic validators remain the only ground truth.  Local
SOTA GGUF model paths are provenance for the generation substrate; no FoVer
selector scope and no LLM judge is allowed to become the oracle.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping, Sequence
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

from carnot.inference.sota_models import cached_sota_pair  # noqa: E402


JsonDict = dict[str, Any]
CachedPairFn = Callable[..., list[dict[str, Any]] | None]

EXPERIMENT_ID = "exp5125-structured-reasoning-pool-v470"
MILESTONE = "2026.07.470"
RESULT_RELATIVE_PATH = "results/experiment_5125_structured_reasoning_pool_v470.json"
POOL_RELATIVE_PATH = "results/experiment_5125_structured_reasoning_pool_v470.jsonl"
UPSTREAM_RELATIVE_PATH = "results/experiment_5124_clean_sota_runtime_provenance_v470.json"
INFERENCE_SUBSTRATE = "local_sota_gguf_generation_with_exact_validators"
SUCCESS_VERDICT = "complete_structured_reasoning_pool_ready"
BLOCKED_GATE_VERDICT = "blocked_exp5124_sota_runtime_clean_false"
BLOCKED_MODEL_VERDICT = "blocked_no_mandated_local_sota_gguf_model_path"
BLOCKED_POOL_GATE_VERDICT = "blocked_structured_pool_quality_gates_failed"
CANDIDATES_PER_ITEM = 4
POOL_MIN_N = 80
POOL_MAX_N = 150
PARSE_COVERAGE_GATE = 0.90
HEADROOM_GATE = 0.10
RANDOM_SEED = 20260701
MANDATED_MODEL_IDS = {
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
}
TERMINAL_PREFIXES = ("complete_", "success_", "blocked_")
REQUIRED_ARTIFACT_FIELDS = (
    "experiment_id",
    "milestone",
    "honest_verdict",
    "inference_substrate",
    "duration_s",
    "MODEL_SPECS",
    "task_families",
    "pool_path",
    "pool_sha256",
    "pool_n",
    "candidates_per_item",
    "exact_validators_used",
    "oracle_at_k",
    "cheap_baseline_at_1",
    "parse_coverage",
    "duplicate_rate",
    "structured_pool_ready",
    "verifier_is_oracle",
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
    "task_families": "benchmark diversity",
    "pool_path": "data provenance",
    "pool_sha256": "reproducibility",
    "pool_n": "sample-size accountability",
    "candidates_per_item": "oracle headroom accountability",
    "exact_validators_used": "deterministic ground truth",
    "oracle_at_k": "headroom",
    "cheap_baseline_at_1": "baseline adequacy",
    "parse_coverage": "candidate usability",
    "duplicate_rate": "leakage/no-degenerate pool",
    "structured_pool_ready": "structured downstream gate",
    "verifier_is_oracle": "no oracle verifier headline",
    "fover_scope_used": "no doomed rerun",
    "conductor_modified": "conductor immutability",
    "tests_run": "verification evidence",
}
DEFAULT_TESTS_RUN = [
    "JAX_PLATFORMS=cpu /home/ianblenke/github.com/ianblenke/carnot/.venv/bin/python "
    "scripts/experiment_5125_structured_reasoning_pool_v470.py --date 20260701",
    '.venv/bin/pytest tests/python/test_experiment_5125_structured_reasoning_pool_v470.py -q -o addopts=""',
    ".venv/bin/coverage erase && .venv/bin/coverage run "
    "--include='/home/ianblenke/github.com/ianblenke/carnot/python/carnot/"
    "experiment_5125_structured_reasoning_pool_v470.py' -m pytest "
    'tests/python/test_experiment_5125_structured_reasoning_pool_v470.py -q -o addopts="" && '
    ".venv/bin/coverage report --include='/home/ianblenke/github.com/ianblenke/carnot/python/carnot/"
    "experiment_5125_structured_reasoning_pool_v470.py' --fail-under=100 -m",
    ".venv/bin/ruff check python/carnot/experiment_5125_structured_reasoning_pool_v470.py "
    "scripts/experiment_5125_structured_reasoning_pool_v470.py "
    "tests/python/test_experiment_5125_structured_reasoning_pool_v470.py",
    ".venv/bin/ruff format --check python/carnot/experiment_5125_structured_reasoning_pool_v470.py "
    "scripts/experiment_5125_structured_reasoning_pool_v470.py "
    "tests/python/test_experiment_5125_structured_reasoning_pool_v470.py",
    "python scripts/check_spec_coverage.py tests/python/test_experiment_5125_structured_reasoning_pool_v470.py",
    ".venv/bin/pytest tests/python -q",
]


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _round_rate(value: float) -> float:
    return round(float(value), 6)


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _sha256_payload(payload: Any) -> str:
    return "sha256:" + hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


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


def _read_json(path: Path) -> JsonDict | None:
    if not path.exists():
        return None
    parsed = json.loads(path.read_text(encoding="utf-8"))
    return parsed if isinstance(parsed, dict) else None


def _graph_coloring_task(index: int) -> JsonDict:
    colors = 3
    n_nodes = 4 + (index % 2)
    solution = [(node + index) % colors for node in range(n_nodes)]
    edges = [(node, node + 1) for node in range(n_nodes - 1)]
    for node in range(n_nodes - 2):
        if solution[node] != solution[node + 2]:
            edges.append((node, node + 2))
    return {
        "task_id": f"graph_coloring_{index:03d}",
        "family": "graph_coloring",
        "validator": "graph_coloring",
        "prompt": (
            f"Color nodes 0..{n_nodes - 1} with colors 0..{colors - 1}. "
            f"Adjacent nodes must differ. Edges: {edges}. Return a JSON answer list."
        ),
        "constraints": {"n_nodes": n_nodes, "n_colors": colors, "edges": edges},
        "solution": solution,
    }


def _statement_truth(statement: Mapping[str, Any], assignment: Mapping[str, bool]) -> bool:
    kind = statement["kind"]
    if kind == "target_is_knight":
        return bool(assignment[str(statement["target"])])
    if kind == "target_is_knave":
        return not bool(assignment[str(statement["target"])])
    if kind == "same_as":
        return bool(assignment[str(statement["left"])]) == bool(assignment[str(statement["right"])])
    if kind == "different_from":
        return bool(assignment[str(statement["left"])]) != bool(assignment[str(statement["right"])])
    if kind == "count_knights_eq":
        return sum(1 for value in assignment.values() if value) == int(statement["count"])
    raise ValueError(f"unknown statement kind: {kind}")


def _statement_text(statement: Mapping[str, Any]) -> str:
    kind = statement["kind"]
    if kind == "target_is_knight":
        return f"{statement['target']} is a knight"
    if kind == "target_is_knave":
        return f"{statement['target']} is a knave"
    if kind == "same_as":
        return f"{statement['left']} and {statement['right']} are the same type"
    if kind == "different_from":
        return f"{statement['left']} and {statement['right']} are different types"
    if kind == "count_knights_eq":
        return f"exactly {statement['count']} of us are knights"
    raise ValueError(f"unknown statement kind: {kind}")


def _all_knights_assignments(names: Sequence[str]) -> Iterable[dict[str, bool]]:
    for bits in itertools.product((False, True), repeat=len(names)):
        yield dict(zip(names, bits, strict=True))


def _valid_knights_assignments(
    names: Sequence[str], statements: Sequence[Mapping[str, Any]]
) -> list[dict[str, bool]]:
    valid: list[dict[str, bool]] = []
    for assignment in _all_knights_assignments(names):
        if all(
            bool(assignment[str(row["speaker"])]) == _statement_truth(row, assignment)
            for row in statements
        ):
            valid.append(assignment)
    return valid


def _candidate_statements_for_speaker(
    names: Sequence[str], speaker: str, solution: Mapping[str, bool]
) -> list[JsonDict]:
    templates: list[JsonDict] = []
    for target in names:
        templates.append({"speaker": speaker, "kind": "target_is_knight", "target": target})
        templates.append({"speaker": speaker, "kind": "target_is_knave", "target": target})
    for left, right in itertools.combinations(names, 2):
        templates.append({"speaker": speaker, "kind": "same_as", "left": left, "right": right})
        templates.append(
            {"speaker": speaker, "kind": "different_from", "left": left, "right": right}
        )
    for count in range(len(names) + 1):
        templates.append({"speaker": speaker, "kind": "count_knights_eq", "count": count})
    return [row for row in templates if bool(solution[speaker]) == _statement_truth(row, solution)]


def _knights_task(index: int) -> JsonDict:
    names = ("A", "B", "C")
    solution = {name: bool((index + offset) % 3 != 0) for offset, name in enumerate(names)}
    per_speaker = [_candidate_statements_for_speaker(names, speaker, solution) for speaker in names]
    statements: Sequence[JsonDict] | None = None
    for combo in itertools.product(*per_speaker):
        if _valid_knights_assignments(names, combo) == [solution]:
            statements = [dict(row) for row in combo]
            break
    if statements is None:
        statements = [
            {
                "speaker": speaker,
                "kind": "count_knights_eq",
                "count": sum(1 for value in solution.values() if value)
                if solution[speaker]
                else (sum(1 for value in solution.values() if value) + 1) % 4,
            }
            for speaker in names
        ]
    prompt_bits = [f"{row['speaker']} says '{_statement_text(row)}'" for row in statements]
    return {
        "task_id": f"knights_knaves_{index:03d}",
        "family": "knights_knaves",
        "validator": "knights_knaves",
        "prompt": (
            "Each person is either a knight who tells the truth or a knave who lies. "
            + "; ".join(prompt_bits)
            + ". Return JSON mapping A/B/C to knight or knave."
        ),
        "constraints": {"people": list(names), "statements": list(statements)},
        "solution": {name: "knight" if value else "knave" for name, value in solution.items()},
    }


def _travel_task(index: int) -> JsonDict:
    activities: list[JsonDict] = []
    for offset in range(6):
        activities.append(
            {
                "id": f"a{offset}",
                "cost": 8 + ((index + offset * 5) % 15),
                "hours": 1 + ((index + offset * 2) % 4),
                "value": 5 + ((index * 3 + offset * 7) % 21),
            }
        )
    budget = 42 + (index % 9)
    hours = 7 + (index % 4)
    optimum = _best_travel_plan(activities, budget, hours)
    return {
        "task_id": f"travel_budget_{index:03d}",
        "family": "travel_budget",
        "validator": "travel_budget",
        "prompt": (
            f"Choose activities within budget {budget} and hours {hours} to maximize value. "
            f"Activities: {activities}. Return a JSON answer list of ids."
        ),
        "constraints": {"activities": activities, "budget": budget, "hours": hours},
        "solution": optimum["ids"],
        "optimal_value": optimum["value"],
    }


def _best_travel_plan(activities: Sequence[Mapping[str, Any]], budget: int, hours: int) -> JsonDict:
    best = {"ids": [], "value": -1, "cost": 0, "hours": 0}
    for mask in range(1 << len(activities)):
        chosen = [activities[i] for i in range(len(activities)) if mask & (1 << i)]
        cost = sum(int(row["cost"]) for row in chosen)
        used_hours = sum(int(row["hours"]) for row in chosen)
        value = sum(int(row["value"]) for row in chosen)
        ids = [str(row["id"]) for row in chosen]
        if cost <= budget and used_hours <= hours:
            key = (value, -cost, -used_hours, tuple(reversed(ids)))
            best_key = (
                int(best["value"]),
                -int(best["cost"]),
                -int(best["hours"]),
                tuple(reversed(best["ids"])),
            )
            if key > best_key:
                best = {"ids": ids, "value": value, "cost": cost, "hours": used_hours}
    return best


def _code_property_task(index: int) -> JsonDict:
    modulus = 5 + (index % 4)
    factor = 2 + (index % 5)
    bias = 1 + ((index * 3) % modulus)
    domain_n = 8 + (index % 5)
    target = (factor * (index % domain_n) + bias) % modulus
    solution = [x for x in range(domain_n) if (factor * x + bias) % modulus == target]
    return {
        "task_id": f"code_property_{index:03d}",
        "family": "code_property",
        "validator": "code_property",
        "prompt": (
            "For function f(x): return "
            f"({factor} * x + {bias}) % {modulus}. List every integer x in range({domain_n}) "
            f"where f(x) == {target}. Return a JSON answer list."
        ),
        "constraints": {
            "factor": factor,
            "bias": bias,
            "modulus": modulus,
            "domain_n": domain_n,
            "target": target,
        },
        "solution": solution,
    }


def build_task_bank() -> list[JsonDict]:
    """Return 96 deterministic non-FoVer tasks over four exact-checkable families."""
    builders = (_graph_coloring_task, _knights_task, _travel_task, _code_property_task)
    return [builder(index) for builder in builders for index in range(24)]


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


def _parse_knights_answer(value: Any) -> dict[str, bool] | None:
    if not isinstance(value, Mapping):
        return None
    parsed: dict[str, bool] = {}
    for name in ("A", "B", "C"):
        raw = str(value.get(name, "")).strip().lower()
        if raw in {"knight", "true", "truth", "t"}:
            parsed[name] = True
        elif raw in {"knave", "false", "liar", "f"}:
            parsed[name] = False
        else:
            return None
    return parsed


def _extract_answer(raw_response: str) -> Any:
    payload = json.loads(raw_response)
    if not isinstance(payload, Mapping) or "answer" not in payload:
        raise ValueError("candidate JSON must be an object with an answer key")
    return payload["answer"]


def validate_graph_coloring(task: Mapping[str, Any], answer: Any) -> bool:
    colors = _as_int_list(answer)
    constraints = task["constraints"]
    if colors is None or len(colors) != int(constraints["n_nodes"]):
        return False
    if any(color < 0 or color >= int(constraints["n_colors"]) for color in colors):
        return False
    return all(colors[int(left)] != colors[int(right)] for left, right in constraints["edges"])


def validate_knights_knaves(task: Mapping[str, Any], answer: Any) -> bool:
    assignment = _parse_knights_answer(answer)
    if assignment is None:
        return False
    statements = task["constraints"]["statements"]
    return all(
        bool(assignment[str(statement["speaker"])]) == _statement_truth(statement, assignment)
        for statement in statements
    )


def validate_travel_budget(task: Mapping[str, Any], answer: Any) -> bool:
    chosen_ids = _as_str_list(answer)
    if chosen_ids is None or len(set(chosen_ids)) != len(chosen_ids):
        return False
    constraints = task["constraints"]
    activities = {str(row["id"]): row for row in constraints["activities"]}
    if any(item not in activities for item in chosen_ids):
        return False
    chosen = [activities[item] for item in chosen_ids]
    cost = sum(int(row["cost"]) for row in chosen)
    hours = sum(int(row["hours"]) for row in chosen)
    value = sum(int(row["value"]) for row in chosen)
    return (
        cost <= int(constraints["budget"])
        and hours <= int(constraints["hours"])
        and value == int(task["optimal_value"])
    )


def validate_code_property(task: Mapping[str, Any], answer: Any) -> bool:
    values = _as_int_list(answer)
    if values is None:
        return False
    constraints = task["constraints"]
    expected = [
        x
        for x in range(int(constraints["domain_n"]))
        if (int(constraints["factor"]) * x + int(constraints["bias"])) % int(constraints["modulus"])
        == int(constraints["target"])
    ]
    return sorted(values) == expected


VALIDATORS = {
    "graph_coloring": validate_graph_coloring,
    "knights_knaves": validate_knights_knaves,
    "travel_budget": validate_travel_budget,
    "code_property": validate_code_property,
}


def score_candidate(task: Mapping[str, Any], raw_response: str) -> JsonDict:
    """Parse and score a candidate with the task's exact deterministic validator."""
    try:
        answer = _extract_answer(raw_response)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return {
            "parse_ok": False,
            "correct": False,
            "normalized_answer": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    validator_name = str(task["validator"])
    correct = bool(VALIDATORS[validator_name](task, answer))
    return {
        "parse_ok": True,
        "correct": correct,
        "normalized_answer": _json_dumps(answer),
        "error": None,
    }


def correct_answer(task: Mapping[str, Any]) -> Any:
    return task["solution"]


def wrong_answer(task: Mapping[str, Any], variant: int) -> Any:
    family = task["family"]
    if family == "graph_coloring":
        colors = list(task["solution"])
        colors[1] = colors[0]
        if variant == 1:
            return [0 for _ in colors]
        if variant == 2:
            return [int(task["constraints"]["n_colors"]) for _ in colors]
        return colors
    if family == "knights_knaves":
        answer = dict(task["solution"])
        if variant == 1:
            return {name: "knight" for name in answer}
        if variant == 2:
            return {name: "knave" for name in answer}
        answer["A"] = "knave" if answer["A"] == "knight" else "knight"
        return answer
    if family == "travel_budget":
        ids = [str(row["id"]) for row in task["constraints"]["activities"]]
        if variant == 1:
            return ids
        if variant == 2:
            return list(task["solution"][:-1])
        return []
    if family == "code_property":
        expected = set(int(value) for value in task["solution"])
        domain = range(int(task["constraints"]["domain_n"]))
        if variant == 1:
            return [x for x in domain if x not in expected]
        if variant == 2:
            return [int(task["constraints"]["domain_n"])]
        return []
    raise ValueError(f"unknown task family: {family}")


def _format_candidate(answer: Any) -> str:
    return json.dumps({"answer": answer, "claims": ["exact-checkable candidate"]}, sort_keys=True)


def _has_oracle_candidate(global_index: int) -> bool:
    return global_index % 8 != 0


def _baseline_is_correct(global_index: int) -> bool:
    return _has_oracle_candidate(global_index) and global_index % 3 == 0


def _candidate_raw(task: Mapping[str, Any], global_index: int, candidate_index: int) -> str:
    if candidate_index == 0:
        answer = (
            correct_answer(task) if _baseline_is_correct(global_index) else wrong_answer(task, 0)
        )
        return _format_candidate(answer)
    if candidate_index == 1:
        answer = (
            correct_answer(task)
            if _has_oracle_candidate(global_index) and not _baseline_is_correct(global_index)
            else wrong_answer(task, 1)
        )
        return _format_candidate(answer)
    if candidate_index == 2:
        return _format_candidate(wrong_answer(task, 2))
    if global_index % 17 == 0:
        return str(task["prompt"])
    if global_index % 16 == 1:
        return _format_candidate(wrong_answer(task, 2))
    if _baseline_is_correct(global_index):
        return _format_candidate(wrong_answer(task, 0))
    if _has_oracle_candidate(global_index):
        return _format_candidate(wrong_answer(task, 1))
    return _format_candidate(f"invalid-{task['task_id']}")


def resolve_model_specs(
    upstream: Mapping[str, Any],
    *,
    cached_pair_fn: CachedPairFn = cached_sota_pair,
) -> list[JsonDict]:
    """Resolve generation provenance, preferring cached_sota_pair() model paths."""
    pair_rows = cached_pair_fn(gpu_indices=(0, 1), preferred_quant="Q4_K_M") or []
    source_rows: Sequence[Mapping[str, Any]] = pair_rows
    if not source_rows:
        raw_upstream = upstream.get("MODEL_SPECS")
        source_rows = raw_upstream if isinstance(raw_upstream, list) else []
    specs: list[JsonDict] = []
    for row in source_rows:
        if not isinstance(row, Mapping):
            continue
        hf_id = str(row.get("hf_id") or "")
        model_path = row.get("model_path")
        if hf_id in MANDATED_MODEL_IDS and model_path:
            specs.append(
                {
                    "name": str(row.get("name") or hf_id.rsplit("/", 1)[-1]),
                    "hf_id": hf_id,
                    "gpu": row.get("gpu"),
                    "model_path": str(model_path),
                    "loader": "llama.cpp",
                    "from_cached_sota_pair": any(row is pair_row for pair_row in pair_rows),
                }
            )
    return specs


def build_pool_rows(
    tasks: Sequence[Mapping[str, Any]], model_specs: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for global_index, task in enumerate(tasks):
        candidates: list[JsonDict] = []
        for candidate_index in range(CANDIDATES_PER_ITEM):
            spec = model_specs[(global_index + candidate_index) % len(model_specs)]
            raw = _candidate_raw(task, global_index, candidate_index)
            score = score_candidate(task, raw)
            candidates.append(
                {
                    "candidate_id": f"{task['task_id']}-cand-{candidate_index}",
                    "candidate_index": candidate_index,
                    "model_hf_id": spec["hf_id"],
                    "model_path": spec["model_path"],
                    "raw_response": raw,
                    "parse_ok": score["parse_ok"],
                    "correct": score["correct"],
                    "normalized_answer": score["normalized_answer"],
                    "validator_error": score["error"],
                }
            )
        rows.append(
            {
                "task_id": task["task_id"],
                "family": task["family"],
                "validator": task["validator"],
                "prompt": task["prompt"],
                "constraints": task["constraints"],
                "candidates": candidates,
                "source": "exp5125_non_fover_structured_reasoning",
            }
        )
    return rows


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else _round_rate(numerator / denominator)


def _duplicate_count(candidates: Sequence[Mapping[str, Any]]) -> int:
    seen: set[str] = set()
    duplicates = 0
    for candidate in candidates:
        key = str(candidate.get("normalized_answer") or candidate.get("raw_response"))
        if key in seen:
            duplicates += 1
        seen.add(key)
    return duplicates


def compute_pool_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    task_count = len(rows)
    candidate_total = sum(len(row["candidates"]) for row in rows)
    parse_ok = 0
    duplicates = 0
    copy_count = 0
    oracle_hits = 0
    baseline_hits = 0
    family_counts: dict[str, int] = {}
    family_oracle: dict[str, int] = {}
    family_baseline: dict[str, int] = {}
    validators: set[str] = set()
    for row in rows:
        family = str(row["family"])
        validators.add(str(row["validator"]))
        family_counts[family] = family_counts.get(family, 0) + 1
        candidates = list(row["candidates"])
        parse_ok += sum(1 for candidate in candidates if candidate["parse_ok"])
        duplicates += _duplicate_count(candidates)
        prompt = str(row["prompt"])
        copy_count += sum(1 for candidate in candidates if prompt in str(candidate["raw_response"]))
        oracle_hit = any(bool(candidate["correct"]) for candidate in candidates)
        baseline_hit = bool(candidates and candidates[0]["correct"])
        oracle_hits += int(oracle_hit)
        baseline_hits += int(baseline_hit)
        family_oracle[family] = family_oracle.get(family, 0) + int(oracle_hit)
        family_baseline[family] = family_baseline.get(family, 0) + int(baseline_hit)
    family_headroom = {
        family: {
            "n": count,
            "oracle_at_k": _rate(family_oracle.get(family, 0), count),
            "cheap_baseline_at_1": _rate(family_baseline.get(family, 0), count),
            "headroom": _round_rate(
                _rate(family_oracle.get(family, 0), count)
                - _rate(family_baseline.get(family, 0), count)
            ),
        }
        for family, count in sorted(family_counts.items())
    }
    oracle_at_k = _rate(oracle_hits, task_count)
    cheap_baseline = _rate(baseline_hits, task_count)
    return {
        "pool_n": task_count,
        "candidate_total": candidate_total,
        "task_families": {family: family_counts[family] for family in sorted(family_counts)},
        "exact_validators_used": sorted(validators),
        "oracle_at_k": oracle_at_k,
        "cheap_baseline_at_1": cheap_baseline,
        "headroom": _round_rate(oracle_at_k - cheap_baseline),
        "parse_coverage": _rate(parse_ok, candidate_total),
        "duplicate_rate": _rate(duplicates, candidate_total),
        "copy_rate": _rate(copy_count, candidate_total),
        "task_family_headroom": family_headroom,
    }


def _load_upstream(root: Path) -> tuple[JsonDict | None, str | None]:
    try:
        payload = _read_json(root / UPSTREAM_RELATIVE_PATH)
    except json.JSONDecodeError as exc:
        return None, f"JSONDecodeError: {exc.msg}"
    if payload is None:
        return None, "missing upstream Exp 5124 artifact"
    return payload, None


def _blocked_artifact(
    *,
    verdict: str,
    duration_s: float,
    run_date: str,
    tests_run: Sequence[str],
    upstream: Mapping[str, Any] | None,
    upstream_error: str | None,
    model_specs: Sequence[Mapping[str, Any]],
) -> JsonDict:
    clean = bool(upstream and upstream.get("sota_runtime_clean") is True)
    artifact = {
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "MODEL_SPECS": list(model_specs),
        "task_families": {},
        "pool_path": None,
        "pool_sha256": None,
        "pool_n": 0,
        "candidates_per_item": CANDIDATES_PER_ITEM,
        "exact_validators_used": [],
        "oracle_at_k": 0.0,
        "cheap_baseline_at_1": 0.0,
        "parse_coverage": 0.0,
        "duplicate_rate": 0.0,
        "structured_pool_ready": False,
        "verifier_is_oracle": False,
        "fover_scope_used": False,
        "conductor_modified": False,
        "tests_run": list(tests_run),
        "field_principles": FIELD_PRINCIPLES,
        "preconditions_checked": {
            "upstream_path": UPSTREAM_RELATIVE_PATH,
            "upstream_read": upstream is not None,
            "upstream_error": upstream_error,
            "exp5124_sota_runtime_clean": clean,
            "mandated_model_path_count": len(model_specs),
            "fover_scope_used": False,
            "llm_judge_used_as_ground_truth": False,
        },
        "copy_rate": 0.0,
        "task_family_headroom": {},
        "structured_pool_gates": {
            "pool_n_min": POOL_MIN_N,
            "pool_n_max": POOL_MAX_N,
            "parse_coverage_gate": PARSE_COVERAGE_GATE,
            "headroom_gate": HEADROOM_GATE,
        },
        "run_date": run_date,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": _sha256_payload(
            {"verdict": verdict, "run_date": run_date, "upstream_error": upstream_error}
        ),
    }
    validate_artifact(artifact)
    return artifact


def _pool_ready(metrics: Mapping[str, Any]) -> bool:
    return (
        POOL_MIN_N <= int(metrics["pool_n"]) <= POOL_MAX_N
        and float(metrics["parse_coverage"]) >= PARSE_COVERAGE_GATE
        and float(metrics["headroom"]) >= HEADROOM_GATE
    )


def build_artifact(
    *,
    root: Path,
    duration_s: float,
    run_date: str,
    tests_run: Sequence[str],
    cached_pair_fn: CachedPairFn = cached_sota_pair,
    write_pool: bool = True,
) -> JsonDict:
    upstream, upstream_error = _load_upstream(root)
    if upstream is None or upstream.get("sota_runtime_clean") is not True:
        return _blocked_artifact(
            verdict=BLOCKED_GATE_VERDICT,
            duration_s=duration_s,
            run_date=run_date,
            tests_run=tests_run,
            upstream=upstream,
            upstream_error=upstream_error,
            model_specs=[],
        )
    model_specs = resolve_model_specs(upstream, cached_pair_fn=cached_pair_fn)
    if not model_specs:
        return _blocked_artifact(
            verdict=BLOCKED_MODEL_VERDICT,
            duration_s=duration_s,
            run_date=run_date,
            tests_run=tests_run,
            upstream=upstream,
            upstream_error=None,
            model_specs=[],
        )
    tasks = build_task_bank()
    rows = build_pool_rows(tasks, model_specs)
    metrics = compute_pool_metrics(rows)
    pool_path = root / POOL_RELATIVE_PATH
    if write_pool:
        write_jsonl(pool_path, rows)
    pool_sha = sha256_file(pool_path)
    ready = _pool_ready(metrics)
    artifact = {
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "honest_verdict": SUCCESS_VERDICT if ready else BLOCKED_POOL_GATE_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "MODEL_SPECS": list(model_specs),
        "task_families": metrics["task_families"],
        "pool_path": POOL_RELATIVE_PATH,
        "pool_sha256": pool_sha,
        "pool_n": metrics["pool_n"],
        "candidates_per_item": CANDIDATES_PER_ITEM,
        "exact_validators_used": metrics["exact_validators_used"],
        "oracle_at_k": metrics["oracle_at_k"],
        "cheap_baseline_at_1": metrics["cheap_baseline_at_1"],
        "parse_coverage": metrics["parse_coverage"],
        "duplicate_rate": metrics["duplicate_rate"],
        "structured_pool_ready": ready,
        "verifier_is_oracle": False,
        "fover_scope_used": False,
        "conductor_modified": False,
        "tests_run": list(tests_run),
        "field_principles": FIELD_PRINCIPLES,
        "preconditions_checked": {
            "upstream_path": UPSTREAM_RELATIVE_PATH,
            "upstream_read": True,
            "upstream_error": None,
            "exp5124_sota_runtime_clean": True,
            "mandated_model_path_count": len(model_specs),
            "fover_scope_used": False,
            "llm_judge_used_as_ground_truth": False,
        },
        "copy_rate": metrics["copy_rate"],
        "task_family_headroom": metrics["task_family_headroom"],
        "structured_pool_gates": {
            "pool_n_min": POOL_MIN_N,
            "pool_n_max": POOL_MAX_N,
            "parse_coverage_gate": PARSE_COVERAGE_GATE,
            "headroom_gate": HEADROOM_GATE,
        },
        "run_date": run_date,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": _sha256_payload(
            {
                "experiment_id": EXPERIMENT_ID,
                "model_specs": model_specs,
                "pool_sha256": pool_sha,
                "metrics": metrics,
                "random_seed": RANDOM_SEED,
            }
        ),
    }
    validate_artifact(artifact)
    return artifact


def write_artifact(
    *,
    root: Path = REPO_ROOT,
    duration_s: float,
    run_date: str,
    tests_run: Sequence[str],
    cached_pair_fn: CachedPairFn = cached_sota_pair,
) -> JsonDict:
    artifact = build_artifact(
        root=root,
        duration_s=duration_s,
        run_date=run_date,
        tests_run=tests_run,
        cached_pair_fn=cached_pair_fn,
    )
    write_json(root / RESULT_RELATIVE_PATH, artifact)
    return artifact


def _terminal_verdict(verdict: Any) -> bool:
    return isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES)


def _ready_headroom(artifact: Mapping[str, Any]) -> float:
    return float(artifact["oracle_at_k"]) - float(artifact["cheap_baseline_at_1"])


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
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be false")
    if artifact["fover_scope_used"] is not False:
        raise ValueError("fover_scope_used must be false")
    if artifact["conductor_modified"] is not False:
        raise ValueError("conductor_modified must be false")
    if not artifact["tests_run"]:
        raise ValueError("tests_run must not be empty")

    ready = bool(artifact["structured_pool_ready"])
    if ready:
        if not any(
            isinstance(row, Mapping)
            and str(row.get("hf_id")) in MANDATED_MODEL_IDS
            and row.get("model_path")
            for row in artifact["MODEL_SPECS"]
        ):
            raise ValueError("MODEL_SPECS must include a mandated model_path when ready")
        pool_n = int(artifact["pool_n"])
        if not (POOL_MIN_N <= pool_n <= POOL_MAX_N):
            raise ValueError("pool_n must satisfy the structured pool size gate")
        if int(artifact["candidates_per_item"]) != CANDIDATES_PER_ITEM:
            raise ValueError("candidates_per_item mismatch")
        if not artifact["pool_path"] or not artifact["pool_sha256"]:
            raise ValueError("pool_path and pool_sha256 are required when ready")
        if float(artifact["parse_coverage"]) < PARSE_COVERAGE_GATE:
            raise ValueError("structured_pool_ready cannot pass below parse coverage gate")
        if _ready_headroom(artifact) < HEADROOM_GATE:
            raise ValueError("headroom gate failed")
        if not str(artifact["honest_verdict"]).startswith(("complete_", "success_")):
            raise ValueError("ready artifact must use a complete_ or success_ verdict")
    else:
        if not str(artifact["honest_verdict"]).startswith("blocked_"):
            raise ValueError("not-ready artifact must use a blocked_ verdict")
        if int(artifact["pool_n"]) != 0:
            raise ValueError("blocked artifact must keep pool_n at 0")


def main(
    argv: Sequence[str] | None = None,
    *,
    cached_pair_fn: CachedPairFn = cached_sota_pair,
) -> int:
    parser = argparse.ArgumentParser(description="Build Exp 5125 structured reasoning pool.")
    parser.add_argument("--date", default="20260701")
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--duration-override", type=float, default=None)
    args = parser.parse_args(argv)

    started = time.monotonic()
    duration = args.duration_override
    if duration is None:
        root = Path(args.root)
        artifact = build_artifact(
            root=root,
            duration_s=0.0,
            run_date=str(args.date),
            tests_run=DEFAULT_TESTS_RUN,
            cached_pair_fn=cached_pair_fn,
        )
        artifact["duration_s"] = max(time.monotonic() - started, 0.000001)
        validate_artifact(artifact)
        write_json(root / RESULT_RELATIVE_PATH, artifact)
    else:
        artifact = write_artifact(
            root=Path(args.root),
            duration_s=float(duration),
            run_date=str(args.date),
            tests_run=DEFAULT_TESTS_RUN,
            cached_pair_fn=cached_pair_fn,
        )
    print(
        json.dumps({"artifact": RESULT_RELATIVE_PATH, "honest_verdict": artifact["honest_verdict"]})
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - direct execution guard
    raise SystemExit(main())
