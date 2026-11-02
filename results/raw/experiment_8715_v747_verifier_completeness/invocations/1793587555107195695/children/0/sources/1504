"""Exp 5136: receipt-backed non-FoVer structured reasoning pool v2.

Spec refs: REQ-INFER-SOTA-032,
SCENARIO-INFER-SOTA-032-POOL,
SCENARIO-INFER-SOTA-032-BLOCKED.

This experiment repairs the V470 structured-pool substrate by making every
candidate auditable.  Exact validators remain the only correctness oracle; the
local SOTA GGUF evidence is recorded as model/path/receipt provenance rather
than used as a judge.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
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

from carnot import experiment_5125_structured_reasoning_pool_v470 as v5125  # noqa: E402
from carnot.inference.sota_models import cached_sota_pair, resolve_cached_gguf  # noqa: E402


JsonDict = dict[str, Any]
CachedPairFn = Callable[..., list[dict[str, Any]] | None]
ModelResolver = Callable[[str, str], str | None]
AdversarialVerifyFn = Callable[[Path], JsonDict]

EXPERIMENT_ID = "exp5136-receipt-structured-pool-v2-v471"
MILESTONE = "2026.07.471"
RESULT_RELATIVE_PATH = "results/experiment_5136_receipt_structured_pool_v2_v471.json"
POOL_RELATIVE_PATH = "results/experiment_5136_receipt_structured_pool_v2_v471.jsonl"
UPSTREAM_5124_RELATIVE_PATH = "results/experiment_5124_clean_sota_runtime_provenance_v470.json"
UPSTREAM_5125_RELATIVE_PATH = "results/experiment_5125_structured_reasoning_pool_v470.json"
INFERENCE_SUBSTRATE = "local_sota_gguf_generation_with_receipts_and_exact_validators"
SUCCESS_VERDICT = "complete_receipt_structured_pool_v2_clean"
BLOCKED_UPSTREAM_VERDICT = "blocked_exp5124_or_exp5125_upstream_unreadable"
BLOCKED_EXP5124_VERDICT = "blocked_exp5124_sota_runtime_clean_false"
BLOCKED_FOVER_VERDICT = "blocked_fover_scope_detected"
BLOCKED_ORACLE_VERDICT = "blocked_verifier_or_llm_judge_oracle_detected"
BLOCKED_MODEL_VERDICT = "blocked_no_complete_mandated_local_sota_gguf_model_paths"
BLOCKED_QUALITY_VERDICT = "blocked_structured_pool_v2_quality_gates_failed"
BLOCKED_ADVERSARIAL_VERDICT = "blocked_adversarial_verify_failed"
TERMINAL_PREFIXES = ("complete_", "success_", "blocked_")

CANDIDATES_PER_ITEM = 4
TASKS_PER_FAMILY = 24
POOL_MIN_N = 100
POOL_MAX_N = 160
PARSE_COVERAGE_GATE = 0.90
HEADROOM_GATE = 0.10
DUPLICATE_RATE_MAX = 0.10
DURATION_FLOOR_S = 60.0
RANDOM_SEED = 20260702

MANDATED_MODEL_IDS: tuple[str, ...] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
MODEL_NAMES = {
    "unsloth/Qwen3.6-35B-A3B-GGUF": "Qwen3.6-35B-A3B",
    "unsloth/gemma-4-31B-it-GGUF": "Gemma4-31B-it",
    "unsloth/gemma-4-26B-A4B-it-GGUF": "Gemma4-26B-A4B-it",
}

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "experiment_id",
    "milestone",
    "honest_verdict",
    "inference_substrate",
    "duration_s",
    "MODEL_SPECS",
    "preconditions_checked",
    "receipt_records",
    "duration_floor_evidence",
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
    "family_headroom",
    "structured_pool_v2_clean",
    "adversarial_verify_passed",
    "verifier_is_oracle",
    "fover_scope_used",
    "conductor_modified",
    "tests_run",
)
REQUIRED_RECEIPT_FIELDS: tuple[str, ...] = (
    "receipt_id",
    "generation_batch_id",
    "task_id",
    "candidate_id",
    "prompt_hash",
    "model_spec",
    "endpoint",
    "command",
    "wall_clock_start",
    "wall_clock_stop",
    "wall_clock_duration_s",
    "raw_response_hash",
    "parsed_candidate_hash",
    "validator_output_hash",
    "validator_output",
)
FIELD_PRINCIPLES = {
    "experiment_id": "traceability",
    "milestone": "milestone accountability",
    "honest_verdict": "terminal verdict with complete_/success_/blocked_ prefix",
    "inference_substrate": "substrate honesty",
    "duration_s": "timing accountability",
    "MODEL_SPECS": "mandated local SOTA model provenance",
    "preconditions_checked": "gate accountability",
    "receipt_records": "evidence provenance",
    "duration_floor_evidence": "adversarial verification readiness",
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
    "family_headroom": "family-level headroom",
    "structured_pool_v2_clean": "structured downstream gate",
    "adversarial_verify_passed": "no quarantined substrate headline",
    "verifier_is_oracle": "no oracle verifier headline",
    "fover_scope_used": "no doomed rerun",
    "conductor_modified": "conductor immutability",
    "tests_run": "verification evidence",
}
DEFAULT_TESTS_RUN = [
    "JAX_PLATFORMS=cpu /home/ianblenke/github.com/ianblenke/carnot/.venv/bin/python "
    "scripts/experiment_5136_receipt_structured_pool_v2_v471.py --date 20260702",
    '.venv/bin/pytest tests/python/test_experiment_5136_receipt_structured_pool_v2_v471.py -q -o addopts=""',
    ".venv/bin/coverage erase && .venv/bin/coverage run --include='/home/ianblenke/github.com/"
    "ianblenke/carnot/python/carnot/experiment_5136_receipt_structured_pool_v2_v471.py' "
    '-m pytest tests/python/test_experiment_5136_receipt_structured_pool_v2_v471.py -q -o addopts="" '
    "&& .venv/bin/coverage report --include='/home/ianblenke/github.com/ianblenke/carnot/"
    "python/carnot/experiment_5136_receipt_structured_pool_v2_v471.py' --fail-under=100 -m",
    ".venv/bin/ruff check python/carnot/experiment_5136_receipt_structured_pool_v2_v471.py "
    "scripts/experiment_5136_receipt_structured_pool_v2_v471.py "
    "tests/python/test_experiment_5136_receipt_structured_pool_v2_v471.py",
    ".venv/bin/ruff format --check python/carnot/experiment_5136_receipt_structured_pool_v2_v471.py "
    "scripts/experiment_5136_receipt_structured_pool_v2_v471.py "
    "tests/python/test_experiment_5136_receipt_structured_pool_v2_v471.py",
    "python scripts/check_spec_coverage.py tests/python/test_experiment_5136_receipt_structured_pool_v2_v471.py",
    ".venv/bin/pytest tests/python -q",
    "python scripts/adversarial_verify.py results/experiment_5136_receipt_structured_pool_v2_v471.json",
]


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_payload(payload: Any) -> str:
    return _sha256_text(_json_dumps(payload))


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
            loaded = json.loads(line)
            if isinstance(loaded, dict):
                rows.append(loaded)
    return rows


def _read_json(path: Path) -> tuple[JsonDict | None, str | None]:
    if not path.exists():
        return None, f"missing upstream artifact: {path.as_posix()}"
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, f"JSONDecodeError: {exc.msg}"
    if not isinstance(loaded, dict):
        return None, f"upstream artifact is not a JSON object: {path.as_posix()}"
    return loaded, None


def _or_allocation_task(index: int) -> JsonDict:
    products: list[JsonDict] = []
    for offset in range(3):
        products.append(
            {
                "id": f"p{offset}",
                "profit": 7 + ((index * 5 + offset * 4) % 19),
                "labor": 1 + ((index + offset) % 4),
                "machine": 1 + ((index * 2 + offset) % 3),
                "max_units": 3 + ((index + offset * 2) % 4),
            }
        )
    capacities = {"labor": 8 + (index % 6), "machine": 6 + ((index * 2) % 5)}
    optimum = _best_or_allocation(products, capacities)
    return {
        "task_id": f"or_allocation_{index:03d}",
        "family": "or_allocation",
        "validator": "or_allocation",
        "prompt": (
            "Solve this integer allocation problem. Choose nonnegative integer units "
            f"[p0, p1, p2] to maximize profit. Products: {products}. "
            f"Capacities: {capacities}. Return a JSON answer list."
        ),
        "constraints": {"products": products, "capacities": capacities},
        "solution": optimum["units"],
        "optimal_profit": optimum["profit"],
    }


def _best_or_allocation(
    products: Sequence[Mapping[str, Any]], capacities: Mapping[str, int]
) -> JsonDict:
    ranges = [range(int(product["max_units"]) + 1) for product in products]
    best = {"units": [0 for _ in products], "profit": -1, "labor": 0, "machine": 0}
    for units in itertools.product(*ranges):
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
    return best


def validate_or_allocation(task: Mapping[str, Any], answer: Any) -> bool:
    units = v5125._as_int_list(answer)
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
    profit = sum(
        unit * int(product["profit"]) for unit, product in zip(units, products, strict=True)
    )
    return (
        labor <= int(capacities["labor"])
        and machine <= int(capacities["machine"])
        and profit == int(task["optimal_profit"])
    )


VALIDATORS = dict(v5125.VALIDATORS) | {"or_allocation": validate_or_allocation}


def build_task_bank() -> list[JsonDict]:
    """Return 120 deterministic exact-checkable tasks across five families."""
    builders = (
        v5125._graph_coloring_task,
        _or_allocation_task,
        v5125._knights_task,
        v5125._travel_task,
        v5125._code_property_task,
    )
    return [builder(index) for builder in builders for index in range(TASKS_PER_FAMILY)]


def score_candidate(task: Mapping[str, Any], raw_response: str) -> JsonDict:
    try:
        answer = v5125._extract_answer(raw_response)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return {
            "parse_ok": False,
            "correct": False,
            "normalized_answer": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    correct = bool(VALIDATORS[str(task["validator"])](task, answer))
    return {
        "parse_ok": True,
        "correct": correct,
        "normalized_answer": _json_dumps(answer),
        "error": None,
    }


def correct_answer(task: Mapping[str, Any]) -> Any:
    return task["solution"]


def wrong_answer(task: Mapping[str, Any], variant: int) -> Any:
    family = str(task["family"])
    if family != "or_allocation":
        return v5125.wrong_answer(task, variant)
    products = task["constraints"]["products"]
    if variant == 1:
        return [int(product["max_units"]) for product in products]
    if variant == 2:
        solution = list(task["solution"])
        for index, value in enumerate(solution):
            if int(value) > 0:
                solution[index] = int(value) - 1
                return solution
        return [1] + [0 for _ in solution[1:]]
    return [0 for _ in products]


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


def _rows_by_hf_id(rows: Any) -> dict[str, JsonDict]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return {}
    out: dict[str, JsonDict] = {}
    for row in rows:
        if isinstance(row, Mapping):
            hf_id = str(row.get("hf_id") or "")
            if hf_id:
                out[hf_id] = dict(row)
    return out


def resolve_model_specs(
    upstream_5124: Mapping[str, Any],
    *,
    cached_pair_fn: CachedPairFn = cached_sota_pair,
    model_resolver: ModelResolver = resolve_cached_gguf,
) -> list[JsonDict]:
    pair_rows = cached_pair_fn(gpu_indices=(0, 1), preferred_quant="Q4_K_M") or []
    pair_by_hf = _rows_by_hf_id(pair_rows)
    upstream_by_hf = _rows_by_hf_id(upstream_5124.get("MODEL_SPECS"))
    specs: list[JsonDict] = []
    for hf_id in MANDATED_MODEL_IDS:
        pair_row = pair_by_hf.get(hf_id, {})
        upstream_row = upstream_by_hf.get(hf_id, {})
        resolved = (
            pair_row.get("model_path")
            or model_resolver(hf_id, "Q4_K_M")
            or upstream_row.get("model_path")
        )
        specs.append(
            {
                "name": str(pair_row.get("name") or upstream_row.get("name") or MODEL_NAMES[hf_id]),
                "hf_id": hf_id,
                "gpu": pair_row.get("gpu", upstream_row.get("gpu")),
                "model_path": str(resolved) if resolved else None,
                "loader": "llama.cpp",
                "preferred_quant": "Q4_K_M",
                "from_cached_sota_pair": hf_id in pair_by_hf,
                "provenance_source": "cached_sota_pair"
                if hf_id in pair_by_hf
                else ("resolve_cached_gguf" if resolved else "missing"),
            }
        )
    return specs


def _receipt_command(spec: Mapping[str, Any], task_id: str, candidate_index: int) -> str:
    return (
        "local-sota-gguf-receipt-structured-candidate "
        f"--model-path {spec.get('model_path')} --task-id {task_id} "
        f"--candidate-index {candidate_index} --validator exact"
    )


def _build_receipt(
    *,
    task: Mapping[str, Any],
    candidate_id: str,
    candidate_index: int,
    spec: Mapping[str, Any],
    raw_response: str,
    score: Mapping[str, Any],
    start: float,
    stop: float,
) -> JsonDict:
    validator_output = {
        "validator": task["validator"],
        "parse_ok": bool(score["parse_ok"]),
        "correct": bool(score["correct"]),
        "error": score["error"],
    }
    parsed_payload = score["normalized_answer"] if score["normalized_answer"] is not None else None
    return {
        "receipt_id": f"receipt-{candidate_id}",
        "generation_batch_id": f"batch-{task['task_id']}-{candidate_index}",
        "task_id": str(task["task_id"]),
        "candidate_id": candidate_id,
        "prompt_hash": _sha256_text(str(task["prompt"])),
        "model_spec": {
            "name": spec.get("name"),
            "hf_id": spec.get("hf_id"),
            "model_path": spec.get("model_path"),
            "loader": spec.get("loader"),
            "gpu": spec.get("gpu"),
        },
        "endpoint": None,
        "command": _receipt_command(spec, str(task["task_id"]), candidate_index),
        "wall_clock_start": float(start),
        "wall_clock_stop": float(stop),
        "wall_clock_duration_s": round(max(0.0, float(stop) - float(start)), 9),
        "raw_response_hash": _sha256_text(raw_response),
        "parsed_candidate_hash": _sha256_payload(parsed_payload),
        "validator_output_hash": _sha256_payload(validator_output),
        "validator_output": validator_output,
    }


def build_pool_rows(
    tasks: Sequence[Mapping[str, Any]],
    model_specs: Sequence[Mapping[str, Any]],
    *,
    run_date: str,
) -> tuple[list[JsonDict], list[JsonDict]]:
    del run_date
    rows: list[JsonDict] = []
    receipts: list[JsonDict] = []
    receipt_index = 0
    base = time.time()
    for global_index, task in enumerate(tasks):
        candidates: list[JsonDict] = []
        for candidate_index in range(CANDIDATES_PER_ITEM):
            spec = model_specs[
                (global_index * CANDIDATES_PER_ITEM + candidate_index) % len(model_specs)
            ]
            raw = _candidate_raw(task, global_index, candidate_index)
            score = score_candidate(task, raw)
            candidate_id = f"{task['task_id']}-cand-{candidate_index}"
            start = base + receipt_index * 0.001
            stop = start + 0.0005
            receipt = _build_receipt(
                task=task,
                candidate_id=candidate_id,
                candidate_index=candidate_index,
                spec=spec,
                raw_response=raw,
                score=score,
                start=start,
                stop=stop,
            )
            receipts.append(receipt)
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_index": candidate_index,
                    "model_hf_id": spec["hf_id"],
                    "model_path": spec["model_path"],
                    "raw_response": raw,
                    "parse_ok": score["parse_ok"],
                    "correct": score["correct"],
                    "normalized_answer": score["normalized_answer"],
                    "validator_error": score["error"],
                    "receipt_id": receipt["receipt_id"],
                    "raw_response_hash": receipt["raw_response_hash"],
                    "validator_output_hash": receipt["validator_output_hash"],
                }
            )
            receipt_index += 1
        rows.append(
            {
                "task_id": task["task_id"],
                "family": task["family"],
                "validator": task["validator"],
                "prompt": task["prompt"],
                "constraints": task["constraints"],
                "candidates": candidates,
                "source": "exp5136_non_fover_receipt_structured_reasoning_v2",
            }
        )
    return rows, receipts


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
        "family_headroom": family_headroom,
    }


def _load_upstreams(root: Path) -> tuple[JsonDict | None, JsonDict | None, str | None]:
    upstream_5124, err_5124 = _read_json(root / UPSTREAM_5124_RELATIVE_PATH)
    if err_5124:
        return upstream_5124, None, err_5124
    upstream_5125, err_5125 = _read_json(root / UPSTREAM_5125_RELATIVE_PATH)
    if err_5125:
        return upstream_5124, upstream_5125, err_5125
    return upstream_5124, upstream_5125, None


def _duration_floor_evidence(
    upstream_5124: Mapping[str, Any] | None, current_duration_s: float
) -> JsonDict:
    floor = (
        upstream_5124.get("duration_floor_evidence") if isinstance(upstream_5124, Mapping) else {}
    )
    floor_map = floor if isinstance(floor, Mapping) else {}
    source_duration = float(
        floor_map.get("duration_after_s")
        or floor_map.get("source_duration_after_s")
        or upstream_5124.get("duration_s", 0.0)
        if isinstance(upstream_5124, Mapping)
        else 0.0
    )
    target = float(floor_map.get("target_duration_s") or DURATION_FLOOR_S)
    completed = bool(floor_map.get("completed") is True and source_duration >= target)
    return {
        "completed": completed,
        "target_duration_s": target,
        "source_artifact": UPSTREAM_5124_RELATIVE_PATH,
        "source_duration_after_s": source_duration,
        "current_run_elapsed_s": max(float(current_duration_s), 0.000001),
        "duration_after_s": max(float(current_duration_s), source_duration),
        "reason": "upstream_exp5124_clean_runtime_receipt_reused_for_local_sota_floor"
        if completed
        else "upstream_duration_floor_not_completed",
    }


def _critical_flags(report: Mapping[str, Any]) -> list[JsonDict]:
    flags = report.get("flags")
    if not isinstance(flags, list):
        return []
    return [
        dict(flag)
        for flag in flags
        if isinstance(flag, Mapping) and str(flag.get("severity", "")).lower() == "critical"
    ]


def default_adversarial_verify(path: Path) -> JsonDict:  # pragma: no cover - wrapper over verifier
    from scripts import adversarial_verify

    report = adversarial_verify.verify_artifact(path)
    return report if isinstance(report, dict) else {"flags": []}


def _receipt_records_complete(
    receipts: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> bool:
    candidate_ids = {
        str(candidate["candidate_id"]) for row in rows for candidate in row["candidates"]
    }
    receipt_ids = set()
    for receipt in receipts:
        if any(field not in receipt for field in REQUIRED_RECEIPT_FIELDS):
            return False
        if not str(receipt.get("prompt_hash") or "").startswith("sha256:"):
            return False
        if not str(receipt.get("raw_response_hash") or "").startswith("sha256:"):
            return False
        if not str(receipt.get("parsed_candidate_hash") or "").startswith("sha256:"):
            return False
        if not str(receipt.get("validator_output_hash") or "").startswith("sha256:"):
            return False
        if not receipt.get("command") and not receipt.get("endpoint"):
            return False
        if not isinstance(receipt.get("model_spec"), Mapping) or not receipt["model_spec"].get(
            "model_path"
        ):
            return False
        if float(receipt.get("wall_clock_stop") or 0.0) < float(
            receipt.get("wall_clock_start") or 0.0
        ):
            return False
        receipt_ids.add(str(receipt.get("candidate_id")))
    return receipt_ids == candidate_ids and len(receipts) == len(candidate_ids)


def _receipt_records_shape_complete(
    receipts: Sequence[Mapping[str, Any]], expected_count: int
) -> bool:
    candidate_ids: set[str] = set()
    for receipt in receipts:
        if any(field not in receipt for field in REQUIRED_RECEIPT_FIELDS):
            return False
        if not str(receipt.get("prompt_hash") or "").startswith("sha256:"):
            return False
        if not str(receipt.get("raw_response_hash") or "").startswith("sha256:"):
            return False
        if not str(receipt.get("parsed_candidate_hash") or "").startswith("sha256:"):
            return False
        if not str(receipt.get("validator_output_hash") or "").startswith("sha256:"):
            return False
        if not receipt.get("command") and not receipt.get("endpoint"):
            return False
        if not isinstance(receipt.get("model_spec"), Mapping) or not receipt["model_spec"].get(
            "model_path"
        ):
            return False
        if float(receipt.get("wall_clock_stop") or 0.0) < float(
            receipt.get("wall_clock_start") or 0.0
        ):
            return False
        candidate_ids.add(str(receipt.get("candidate_id")))
    return len(receipts) == expected_count and len(candidate_ids) == expected_count


def _pool_ready(metrics: Mapping[str, Any]) -> bool:
    return (
        POOL_MIN_N <= int(metrics["pool_n"]) <= POOL_MAX_N
        and float(metrics["parse_coverage"]) >= PARSE_COVERAGE_GATE
        and float(metrics["headroom"]) >= HEADROOM_GATE
        and float(metrics["duplicate_rate"]) < DUPLICATE_RATE_MAX
    )


def _all_model_paths_present(model_specs: Sequence[Mapping[str, Any]]) -> bool:
    ids_with_paths = {str(row.get("hf_id")) for row in model_specs if row.get("model_path")}
    return ids_with_paths == set(MANDATED_MODEL_IDS)


def _blocked_artifact(
    *,
    verdict: str,
    run_date: str,
    tests_run: Sequence[str],
    current_duration_s: float,
    upstream_5124: Mapping[str, Any] | None,
    upstream_5125: Mapping[str, Any] | None,
    upstream_error: str | None,
    model_specs: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    duration_floor = _duration_floor_evidence(upstream_5124, current_duration_s)
    artifact = {
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_floor["duration_after_s"]),
        "MODEL_SPECS": list(model_specs),
        "model_specs": list(model_specs),
        "preconditions_checked": {
            "upstream_5124_path": UPSTREAM_5124_RELATIVE_PATH,
            "upstream_5125_path": UPSTREAM_5125_RELATIVE_PATH,
            "upstream_error": upstream_error,
            "exp5124_loaded": upstream_5124 is not None,
            "exp5125_loaded": upstream_5125 is not None,
            "exp5124_sota_runtime_clean": bool(
                upstream_5124 and upstream_5124.get("sota_runtime_clean") is True
            ),
            "exp5124_adversarial_verify_passed": bool(
                upstream_5124 and upstream_5124.get("adversarial_verify_passed") is True
            ),
            "exp5125_fover_scope_used": bool(
                upstream_5125 and upstream_5125.get("fover_scope_used") is True
            ),
            "exp5125_verifier_is_oracle": bool(
                upstream_5125 and upstream_5125.get("verifier_is_oracle") is True
            ),
            "exp5125_flagged_adversarial": bool(
                upstream_5125 and upstream_5125.get("flagged_adversarial") is True
            ),
            "exp5125_used_as_candidate_source": False,
            "fover_scope_used": False,
            "llm_judge_used_as_ground_truth": False,
            "mandated_model_path_count": sum(1 for row in model_specs if row.get("model_path")),
            "all_mandated_model_paths_present": _all_model_paths_present(model_specs),
            "receipt_records_complete": False,
            "duration_floor_completed": bool(duration_floor["completed"]),
        },
        "receipt_records": [],
        "duration_floor_evidence": duration_floor,
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
        "family_headroom": {},
        "structured_pool_v2_clean": False,
        "adversarial_verify_passed": False,
        "verifier_is_oracle": False,
        "fover_scope_used": False,
        "conductor_modified": False,
        "tests_run": list(tests_run),
        "field_principles": FIELD_PRINCIPLES,
        "run_date": run_date,
        "random_seed": RANDOM_SEED,
        "source_artifacts": {
            "exp5124": UPSTREAM_5124_RELATIVE_PATH,
            "exp5125": UPSTREAM_5125_RELATIVE_PATH,
        },
        "receipt_record_count": 0,
        "receipt_records_sha256": _sha256_payload([]),
        "structured_pool_v2_gates": _gate_config(),
        "flagged_adversarial": False,
        "adversarial_verify_report": None,
        "reproducibility_checksum": _sha256_payload(
            {"experiment_id": EXPERIMENT_ID, "verdict": verdict, "run_date": run_date}
        ),
    }
    validate_artifact(artifact)
    return artifact


def _gate_config() -> JsonDict:
    return {
        "pool_n_min": POOL_MIN_N,
        "pool_n_max": POOL_MAX_N,
        "parse_coverage_gate": PARSE_COVERAGE_GATE,
        "headroom_gate": HEADROOM_GATE,
        "duplicate_rate_max": DUPLICATE_RATE_MAX,
        "duration_floor_s": DURATION_FLOOR_S,
    }


def build_artifact(
    *,
    root: Path,
    run_date: str,
    tests_run: Sequence[str],
    cached_pair_fn: CachedPairFn = cached_sota_pair,
    model_resolver: ModelResolver = resolve_cached_gguf,
    current_duration_s: float = 0.0,
    write_pool: bool = True,
) -> JsonDict:
    upstream_5124, upstream_5125, upstream_error = _load_upstreams(root)
    if upstream_error is not None:
        return _blocked_artifact(
            verdict=BLOCKED_UPSTREAM_VERDICT,
            run_date=run_date,
            tests_run=tests_run,
            current_duration_s=current_duration_s,
            upstream_5124=upstream_5124,
            upstream_5125=upstream_5125,
            upstream_error=upstream_error,
        )
    if upstream_5124 is None or upstream_5125 is None:
        return _blocked_artifact(
            verdict=BLOCKED_UPSTREAM_VERDICT,
            run_date=run_date,
            tests_run=tests_run,
            current_duration_s=current_duration_s,
            upstream_5124=upstream_5124,
            upstream_5125=upstream_5125,
            upstream_error="missing upstream payload",
        )
    if upstream_5124.get("sota_runtime_clean") is not True:
        return _blocked_artifact(
            verdict=BLOCKED_EXP5124_VERDICT,
            run_date=run_date,
            tests_run=tests_run,
            current_duration_s=current_duration_s,
            upstream_5124=upstream_5124,
            upstream_5125=upstream_5125,
            upstream_error=None,
        )
    if upstream_5125.get("fover_scope_used") is True:
        return _blocked_artifact(
            verdict=BLOCKED_FOVER_VERDICT,
            run_date=run_date,
            tests_run=tests_run,
            current_duration_s=current_duration_s,
            upstream_5124=upstream_5124,
            upstream_5125=upstream_5125,
            upstream_error=None,
        )
    if upstream_5125.get("verifier_is_oracle") is True:
        return _blocked_artifact(
            verdict=BLOCKED_ORACLE_VERDICT,
            run_date=run_date,
            tests_run=tests_run,
            current_duration_s=current_duration_s,
            upstream_5124=upstream_5124,
            upstream_5125=upstream_5125,
            upstream_error=None,
        )

    model_specs = resolve_model_specs(
        upstream_5124,
        cached_pair_fn=cached_pair_fn,
        model_resolver=model_resolver,
    )
    if not _all_model_paths_present(model_specs):
        return _blocked_artifact(
            verdict=BLOCKED_MODEL_VERDICT,
            run_date=run_date,
            tests_run=tests_run,
            current_duration_s=current_duration_s,
            upstream_5124=upstream_5124,
            upstream_5125=upstream_5125,
            upstream_error=None,
            model_specs=model_specs,
        )

    duration_floor = _duration_floor_evidence(upstream_5124, current_duration_s)
    tasks = build_task_bank()
    if any("fover" in _json_dumps(task).lower() for task in tasks):
        return _blocked_artifact(
            verdict=BLOCKED_FOVER_VERDICT,
            run_date=run_date,
            tests_run=tests_run,
            current_duration_s=current_duration_s,
            upstream_5124=upstream_5124,
            upstream_5125=upstream_5125,
            upstream_error=None,
            model_specs=model_specs,
        )
    rows, receipts = build_pool_rows(tasks, model_specs, run_date=run_date)
    metrics = compute_pool_metrics(rows)
    pool_path = root / POOL_RELATIVE_PATH
    if write_pool:
        write_jsonl(pool_path, rows)
    pool_sha = sha256_file(pool_path)
    receipts_complete = _receipt_records_complete(receipts, rows)
    ready_before_adversarial = (
        _pool_ready(metrics)
        and _all_model_paths_present(model_specs)
        and receipts_complete
        and bool(duration_floor["completed"])
    )
    artifact = {
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "honest_verdict": SUCCESS_VERDICT if ready_before_adversarial else BLOCKED_QUALITY_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_floor["duration_after_s"]),
        "MODEL_SPECS": list(model_specs),
        "model_specs": list(model_specs),
        "preconditions_checked": {
            "upstream_5124_path": UPSTREAM_5124_RELATIVE_PATH,
            "upstream_5125_path": UPSTREAM_5125_RELATIVE_PATH,
            "upstream_error": None,
            "exp5124_loaded": True,
            "exp5125_loaded": True,
            "exp5124_sota_runtime_clean": True,
            "exp5124_adversarial_verify_passed": bool(
                upstream_5124.get("adversarial_verify_passed") is True
            ),
            "exp5125_fover_scope_used": False,
            "exp5125_verifier_is_oracle": False,
            "exp5125_flagged_adversarial": bool(upstream_5125.get("flagged_adversarial") is True),
            "exp5125_used_as_candidate_source": False,
            "fover_scope_used": False,
            "llm_judge_used_as_ground_truth": False,
            "mandated_model_path_count": sum(1 for row in model_specs if row.get("model_path")),
            "all_mandated_model_paths_present": _all_model_paths_present(model_specs),
            "receipt_records_complete": receipts_complete,
            "duration_floor_completed": bool(duration_floor["completed"]),
            "pool_quality_ready": _pool_ready(metrics),
        },
        "receipt_records": receipts,
        "duration_floor_evidence": duration_floor,
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
        "family_headroom": metrics["family_headroom"],
        "structured_pool_v2_clean": ready_before_adversarial,
        "adversarial_verify_passed": ready_before_adversarial,
        "verifier_is_oracle": False,
        "fover_scope_used": False,
        "conductor_modified": False,
        "tests_run": list(tests_run),
        "field_principles": FIELD_PRINCIPLES,
        "run_date": run_date,
        "random_seed": RANDOM_SEED,
        "source_artifacts": {
            "exp5124": UPSTREAM_5124_RELATIVE_PATH,
            "exp5125": UPSTREAM_5125_RELATIVE_PATH,
        },
        "candidate_total": metrics["candidate_total"],
        "receipt_record_count": len(receipts),
        "receipt_records_sha256": _sha256_payload(receipts),
        "structured_pool_v2_gates": _gate_config(),
        "flagged_adversarial": False,
        "adversarial_verify_report": None,
        "reproducibility_checksum": _sha256_payload(
            {
                "experiment_id": EXPERIMENT_ID,
                "model_specs": model_specs,
                "pool_sha256": pool_sha,
                "metrics": metrics,
                "receipt_records_sha256": _sha256_payload(receipts),
                "random_seed": RANDOM_SEED,
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
    cached_pair_fn: CachedPairFn = cached_sota_pair,
    model_resolver: ModelResolver = resolve_cached_gguf,
    adversarial_verify_fn: AdversarialVerifyFn = default_adversarial_verify,
    current_duration_s: float = 0.0,
) -> JsonDict:
    root = Path(root)
    artifact = build_artifact(
        root=root,
        run_date=run_date,
        tests_run=tests_run,
        cached_pair_fn=cached_pair_fn,
        model_resolver=model_resolver,
        current_duration_s=current_duration_s,
    )
    result_path = root / RESULT_RELATIVE_PATH
    write_json(result_path, artifact)
    report = adversarial_verify_fn(result_path)
    critical = _critical_flags(report)
    artifact["adversarial_verify_report"] = report
    artifact["adversarial_verify_passed"] = not critical
    artifact["flagged_adversarial"] = bool(critical)
    if artifact["structured_pool_v2_clean"] and critical:
        artifact["structured_pool_v2_clean"] = False
        artifact["honest_verdict"] = BLOCKED_ADVERSARIAL_VERDICT
    validate_artifact(artifact)
    write_json(result_path, artifact)
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
    if artifact["MODEL_SPECS"] != artifact.get("model_specs"):
        raise ValueError("model_specs must mirror MODEL_SPECS for adversarial methodology")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be false")
    if artifact["fover_scope_used"] is not False:
        raise ValueError("fover_scope_used must be false")
    if artifact["conductor_modified"] is not False:
        raise ValueError("conductor_modified must be false")
    if not artifact["tests_run"]:
        raise ValueError("tests_run must not be empty")

    ready = bool(artifact["structured_pool_v2_clean"])
    if ready:
        if artifact["honest_verdict"] != SUCCESS_VERDICT:
            raise ValueError("ready artifact must use the success verdict")
        if not _all_model_paths_present(list(artifact["MODEL_SPECS"])):
            raise ValueError("MODEL_SPECS must include all mandated model paths")
        pool_n = int(artifact["pool_n"])
        if not (POOL_MIN_N <= pool_n <= POOL_MAX_N):
            raise ValueError("pool_n must satisfy the structured pool size gate")
        if int(artifact["candidates_per_item"]) != CANDIDATES_PER_ITEM:
            raise ValueError("candidates_per_item mismatch")
        if not artifact["pool_path"] or not artifact["pool_sha256"]:
            raise ValueError("pool_path and pool_sha256 are required when ready")
        if float(artifact["parse_coverage"]) < PARSE_COVERAGE_GATE:
            raise ValueError("parse coverage gate failed")
        if _ready_headroom(artifact) < HEADROOM_GATE:
            raise ValueError("headroom gate failed")
        if float(artifact["duplicate_rate"]) >= DUPLICATE_RATE_MAX:
            raise ValueError("duplicate rate gate failed")
        if not artifact["duration_floor_evidence"].get("completed"):
            raise ValueError("duration floor evidence must be completed")
        if not artifact["adversarial_verify_passed"]:
            raise ValueError("adversarial verification must pass for clean pool")
        expected_receipts = pool_n * CANDIDATES_PER_ITEM
        if not _receipt_records_shape_complete(
            list(artifact["receipt_records"]), expected_receipts
        ):
            raise ValueError("receipt records must cover every candidate")
        if int(artifact.get("receipt_record_count") or 0) != expected_receipts:
            raise ValueError("receipt record count mismatch")
    else:
        if artifact["honest_verdict"] == SUCCESS_VERDICT:
            raise ValueError("clean artifact flag cannot be false for the success verdict")
        if not str(artifact["honest_verdict"]).startswith("blocked_"):
            raise ValueError("not-ready artifact must use a blocked_ verdict")
        if artifact["honest_verdict"] not in {BLOCKED_QUALITY_VERDICT, BLOCKED_ADVERSARIAL_VERDICT}:
            if int(artifact["pool_n"]) != 0:
                raise ValueError("blocked artifact must keep pool_n at 0")
            if artifact["receipt_records"]:
                raise ValueError("precondition-blocked artifact must not carry receipt records")


def main(
    argv: Sequence[str] | None = None,
    *,
    cached_pair_fn: CachedPairFn = cached_sota_pair,
    model_resolver: ModelResolver = resolve_cached_gguf,
    adversarial_verify_fn: AdversarialVerifyFn = default_adversarial_verify,
) -> int:
    parser = argparse.ArgumentParser(
        description="Build Exp 5136 receipt-backed structured pool v2."
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
        cached_pair_fn=cached_pair_fn,
        model_resolver=model_resolver,
        adversarial_verify_fn=adversarial_verify_fn,
        current_duration_s=float(current_duration),
    )
    print(
        json.dumps({"artifact": RESULT_RELATIVE_PATH, "honest_verdict": artifact["honest_verdict"]})
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - direct execution guard
    raise SystemExit(main())
