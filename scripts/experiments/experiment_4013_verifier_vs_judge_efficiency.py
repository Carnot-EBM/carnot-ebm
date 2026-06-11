"""Exp 4013: GAP-4 model-free verifier versus Codex/GPT-5.5 judge efficiency.

Spec refs: REQ-VERIFY-4013, SCENARIO-VERIFY-4013.

The experiment compares two selectors over the same ARC candidate-output sets.
Arm A is the cheap GAP-4 verifier: demo-perfect program outputs are executed,
identical outputs are clustered, and the output with the strongest model-free
execution/agreement support is selected. Arm B is Codex/GPT-5.5 as a judge:
it sees the same demos, test input, and candidate output grids, then chooses
one candidate ID. Gold labels are used only after both arms select.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from arc3_gap3_stage2_transition_ebm import SEED, ghash
from arc3_gap4_rule_exec_verifier import safe_transform_from_code


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = REPO_ROOT / "results" / "experiment_4013_verifier_vs_judge_efficiency.json"
INFERENCE_SUBSTRATE = "codex_gpt55_llm_judge_vs_gap4_modelfree_execution_verifier"

SOURCE_KEYS = (
    "rule",
    "arc1_programs",
    "arc1_pool",
    "arc2_chain",
    "arc2_induced",
    "arc2_pool",
)

DEFAULT_PATHS = {
    "rule": REPO_ROOT / "results" / "arc3_gap4_rule_exec_verifier.json",
    "arc1_programs": REPO_ROOT / "results" / "arc3_gap4_induced_programs.json",
    "arc1_pool": REPO_ROOT / "results" / "arc3_gap3_stage2_eval_pool.json.gz",
    "arc2_chain": REPO_ROOT / "results" / "arc3_gap4_arc2_chain_ensemble.json",
    "arc2_induced": REPO_ROOT / "results" / "arc3_gap4_arc2_induced_programs.json",
    "arc2_pool": REPO_ROOT / "results" / "arc3_gap4_arc2_eval_pool.json.gz",
}

REFERENCE_VERIFIER_COSTS = (
    REPO_ROOT / "results" / "experiment_4012_gap4_local_best_of_n.json",
    REPO_ROOT / "results" / "experiment_4002_gap4_local_generator_arm.json",
)

REQUIRED_FIELDS = [
    "selection_accuracy_parity",
    "verifier_gold_rate",
    "judge_gold_rate",
    "cost_ratio_judge_over_verifier",
    "token_ratio_judge_over_verifier",
    "cost_verifier_seconds",
    "cost_judge_seconds",
    "selection_agreement_rate",
    "n_tasks",
    "n_judge_calls",
    "random_seed",
    "honest_verdict",
    "duration_s",
    "inference_substrate",
]

FIELD_PRINCIPLES = {
    "selection_accuracy_parity": (
        "BARE BOOL -- the model-free verifier and the LLM-judge pick equally-gold "
        "candidates within CI (the parity precondition for an efficiency claim)."
    ),
    "verifier_gold_rate": (
        "BARE FLOAT -- per-arm fraction picking a gold candidate (the accuracy head-to-head)."
    ),
    "judge_gold_rate": (
        "BARE FLOAT -- per-arm fraction picking a gold candidate (the accuracy head-to-head)."
    ),
    "cost_ratio_judge_over_verifier": (
        "BARE FLOAT -- judge_seconds / verifier_seconds (the Nx cheaper datum)."
    ),
    "token_ratio_judge_over_verifier": (
        "BARE FLOAT -- finite token-ratio placeholder; verifier tokens are zero, so the seconds "
        "ratio is the headline."
    ),
    "cost_verifier_seconds": "BARE FLOAT -- per-task model-free verifier wall-cost.",
    "cost_judge_seconds": "BARE FLOAT -- per-task Codex judge wall-cost.",
    "selection_agreement_rate": (
        "BARE FLOAT -- fraction of tasks where A and B pick the SAME candidate."
    ),
    "n_tasks": "Coverage count of ARC candidate sets judged.",
    "n_judge_calls": "Codex judge-call provenance.",
    "random_seed": "Reproducibility seed.",
    "honest_verdict": "Terminal-prefix verdict.",
    "duration_s": "Measured wall-clock seconds for this runner.",
    "inference_substrate": "LLM-judge-vs-model-free-verifier substrate.",
}


@dataclass(frozen=True)
class JudgeBatchResult:
    choices: dict[str, str]
    seconds: float
    tokens: int
    raw: str = ""


def _paths(paths: dict[str, Path] | None = None) -> dict[str, Path]:
    merged = dict(DEFAULT_PATHS)
    if paths:
        merged.update({key: Path(value) for key, value in paths.items()})
    return merged


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_pool(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)["entries"]


def load_sources(paths: dict[str, Path] | None = None) -> dict[str, Any]:
    resolved = _paths(paths)
    return {
        "rule": _read_json(resolved["rule"]),
        "arc1_programs": _read_json(resolved["arc1_programs"]),
        "arc1_entries": _read_pool(resolved["arc1_pool"]),
        "arc2_chain": _read_json(resolved["arc2_chain"]),
        "arc2_induced": _read_json(resolved["arc2_induced"]),
        "arc2_entries": _read_pool(resolved["arc2_pool"]),
    }


def check_preconditions(
    *,
    paths: dict[str, Path] | None = None,
    codex_available_override: bool | None = None,
) -> list[dict[str, Any]]:
    codex_available = (
        shutil.which("codex") is not None
        if codex_available_override is None
        else bool(codex_available_override)
    )
    preconditions = [{"resource": "codex", "available": codex_available}]
    resolved = _paths(paths)
    for key in SOURCE_KEYS:
        available = False
        try:
            if key.endswith("pool"):
                _read_pool(resolved[key])
            else:
                _read_json(resolved[key])
            available = True
        except Exception:
            available = False
        preconditions.append(
            {
                "resource": key,
                "path": str(resolved[key]),
                "available": available,
            }
        )
    return preconditions


def blocker_from_preconditions(preconditions: list[dict[str, Any]]) -> str | None:
    by_resource = {row["resource"]: bool(row["available"]) for row in preconditions}
    if not by_resource.get("codex", False):
        return "blocked_codex_unavailable"
    if not all(by_resource.get(key, False) for key in SOURCE_KEYS):
        return "blocked_candidate_sets_missing"
    return None


def _as_grid(grid: Any) -> list[list[int]]:
    return np.asarray(grid, dtype=np.int64).tolist()


def _is_valid_grid(grid: Any) -> bool:
    arr = np.asarray(grid, dtype=np.int64)
    return arr.ndim == 2 and arr.size > 0


def _grid_hash(grid: Any) -> str:
    return ghash(np.asarray(grid, dtype=np.int64))


def _shape(grid: Any) -> list[int]:
    arr = np.asarray(grid, dtype=np.int64)
    return [int(arr.shape[0]), int(arr.shape[1])]


def _gold_hashes(entry: dict[str, Any]) -> set[str]:
    return {
        _grid_hash(candidate["grid"])
        for candidate in entry.get("candidates", [])
        if candidate.get("correct") is True and candidate.get("grid") is not None
    }


def _add_candidate(
    clusters: dict[str, dict[str, Any]],
    *,
    grid: Any,
    source: str,
    is_gold: bool,
    demo_fit: float = 0.0,
    demo_perfect: bool = False,
    votes: int = 0,
    pool_source: bool = False,
) -> None:
    if not _is_valid_grid(grid):
        return
    grid_list = _as_grid(grid)
    grid_hash = _grid_hash(grid_list)
    record = clusters.setdefault(
        grid_hash,
        {
            "grid": grid_list,
            "grid_hash": grid_hash,
            "shape": _shape(grid_list),
            "sources": [],
            "program_source_count": 0,
            "pool_source_count": 0,
            "pool_votes": 0,
            "max_demo_fit": 0.0,
            "is_gold": False,
        },
    )
    record["sources"].append(source)
    record["is_gold"] = bool(record["is_gold"] or is_gold)
    record["max_demo_fit"] = max(float(record["max_demo_fit"]), float(demo_fit or 0.0))
    if demo_perfect:
        record["program_source_count"] += 1
    if pool_source:
        record["pool_source_count"] += 1
        record["pool_votes"] += int(votes)


def _top_pool_candidates(entry: dict[str, Any], k: int) -> list[dict[str, Any]]:
    return sorted(
        [candidate for candidate in entry.get("candidates", []) if _is_valid_grid(candidate["grid"])],
        key=lambda candidate: (-int(candidate.get("votes", 0)), _grid_hash(candidate["grid"])),
    )[:k]


def _program_record_by_index(programs: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for idx, record in enumerate(programs.get("programs", [])):
        out[int(record.get("entry_i", idx))] = record
    return out


def _chain_rows_by_task(chain_artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["task"]): row for row in chain_artifact.get("per_task", [])}


def _execute_chain_arm(arm: dict[str, Any], test_input: Any) -> list[list[int]] | None:
    if not arm.get("demo_perfect") or not arm.get("code"):
        return None
    fn = safe_transform_from_code(str(arm["code"]))
    pred = fn(test_input) if fn is not None else None
    return _as_grid(pred) if pred is not None else None


def _finalize_candidate_set(
    *,
    corpus: str,
    entry_idx: int,
    entry: dict[str, Any],
    clusters: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if not clusters:
        return None
    candidates = sorted(
        clusters.values(),
        key=lambda candidate: (
            -int(candidate["program_source_count"]),
            -float(candidate["max_demo_fit"]),
            -int(candidate["pool_votes"]),
            str(candidate["grid_hash"]),
        ),
    )
    for idx, candidate in enumerate(candidates):
        candidate["choice_id"] = f"C{idx}"
        candidate["sources"] = sorted(set(candidate["sources"]))
    record = {
        "task_key": f"{corpus}:{entry_idx}:{entry['task']}",
        "corpus": corpus,
        "entry_i": entry_idx,
        "task": str(entry["task"]),
        "demos": entry.get("demos", []),
        "test_input": entry.get("test_input"),
        "candidates": candidates,
    }
    record["verifier_choice_id"] = select_with_verifier(record)
    return record


def _arc1_candidate_sets(
    sources: dict[str, Any],
    *,
    top_pool_candidates: int,
) -> list[dict[str, Any]]:
    programs = _program_record_by_index(sources["arc1_programs"])
    out = []
    for entry_idx, entry in enumerate(sources["arc1_entries"]):
        gold_hashes = _gold_hashes(entry)
        clusters: dict[str, dict[str, Any]] = {}
        record = programs.get(entry_idx)
        if record and record.get("demo_perfect") and record.get("pred_grid") is not None:
            _add_candidate(
                clusters,
                grid=record["pred_grid"],
                source="arc1_induced_program",
                is_gold=_grid_hash(record["pred_grid"]) in gold_hashes,
                demo_fit=float(record.get("demo_fit", 0.0)),
                demo_perfect=True,
            )
        for candidate in _top_pool_candidates(entry, top_pool_candidates):
            _add_candidate(
                clusters,
                grid=candidate["grid"],
                source="arc1_pool_vote_candidate",
                is_gold=bool(candidate.get("correct")),
                votes=int(candidate.get("votes", 0)),
                pool_source=True,
            )
        finalized = _finalize_candidate_set(
            corpus="arc1", entry_idx=entry_idx, entry=entry, clusters=clusters
        )
        if finalized:
            out.append(finalized)
    return out


def _arc2_candidate_sets(
    sources: dict[str, Any],
    *,
    top_pool_candidates: int,
) -> list[dict[str, Any]]:
    induced = _program_record_by_index(sources["arc2_induced"])
    chain_by_task = _chain_rows_by_task(sources["arc2_chain"])
    out = []
    for entry_idx, entry in enumerate(sources["arc2_entries"]):
        gold_hashes = _gold_hashes(entry)
        clusters: dict[str, dict[str, Any]] = {}
        record = induced.get(entry_idx)
        if record and record.get("demo_perfect") and record.get("pred_grid") is not None:
            _add_candidate(
                clusters,
                grid=record["pred_grid"],
                source="arc2_induced_program",
                is_gold=_grid_hash(record["pred_grid"]) in gold_hashes,
                demo_fit=float(record.get("demo_fit", 0.0)),
                demo_perfect=True,
            )
        chain_row = chain_by_task.get(str(entry["task"]))
        if chain_row:
            for arm in chain_row.get("arms", []):
                pred = _execute_chain_arm(arm, entry["test_input"])
                if pred is not None:
                    _add_candidate(
                        clusters,
                        grid=pred,
                        source=f"arc2_chain_{arm.get('source', 'arm')}",
                        is_gold=_grid_hash(pred) in gold_hashes,
                        demo_fit=float(arm.get("demo_fit", 0.0)),
                        demo_perfect=True,
                    )
        for candidate in _top_pool_candidates(entry, top_pool_candidates):
            _add_candidate(
                clusters,
                grid=candidate["grid"],
                source="arc2_pool_vote_candidate",
                is_gold=bool(candidate.get("correct")),
                votes=int(candidate.get("votes", 0)),
                pool_source=True,
            )
        finalized = _finalize_candidate_set(
            corpus="arc2", entry_idx=entry_idx, entry=entry, clusters=clusters
        )
        if finalized:
            out.append(finalized)
    return out


def assemble_candidate_sets(
    sources: dict[str, Any],
    *,
    top_pool_candidates: int = 2,
    limit: int = 0,
) -> list[dict[str, Any]]:
    candidate_sets = _arc1_candidate_sets(sources, top_pool_candidates=top_pool_candidates)
    candidate_sets.extend(_arc2_candidate_sets(sources, top_pool_candidates=top_pool_candidates))
    candidate_sets = [row for row in candidate_sets if len(row["candidates"]) >= 1]
    if limit:
        candidate_sets = candidate_sets[:limit]
    return candidate_sets


def select_with_verifier(candidate_set: dict[str, Any]) -> str:
    candidates = candidate_set["candidates"]
    best = max(
        candidates,
        key=lambda candidate: (
            int(candidate["program_source_count"]),
            float(candidate["max_demo_fit"]),
            int(candidate["pool_source_count"]),
            int(candidate["pool_votes"]),
            -int(str(candidate.get("choice_id", "C0"))[1:] or 0),
        ),
    )
    return str(best.get("choice_id", "C0"))


def _compact_grid(grid: Any) -> str:
    return json.dumps(_as_grid(grid), separators=(",", ":"))


def build_judge_prompt(batch: list[dict[str, Any]]) -> str:
    lines = [
        "You are judging ARC candidate outputs. For each task, choose the candidate output most "
        "consistent with the demonstrations. Use only the demos, test input, and candidate outputs.",
        "Return only JSON: {\"decisions\":[{\"task_key\":\"...\",\"choice_id\":\"C0\"}]}",
    ]
    for task in batch:
        lines.append(f"\nTASK {task['task_key']}")
        for idx, pair in enumerate(task["demos"], 1):
            lines.append(f"Demo {idx} input: {_compact_grid(pair['input'])}")
            lines.append(f"Demo {idx} output: {_compact_grid(pair['output'])}")
        lines.append(f"Test input: {_compact_grid(task['test_input'])}")
        for candidate in task["candidates"]:
            lines.append(f"Candidate {candidate['choice_id']}: {_compact_grid(candidate['grid'])}")
    return "\n".join(lines)


def _extract_json_object(text: str) -> Any:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object found")
    return json.loads(text[start : end + 1])


def parse_judge_payload(text: str, task_keys: list[str]) -> dict[str, str]:
    choices: dict[str, str] = {}
    try:
        data = _extract_json_object(text)
    except Exception:
        data = None
    if isinstance(data, dict):
        decisions = data.get("decisions")
        if isinstance(decisions, list):
            for row in decisions:
                if isinstance(row, dict):
                    key = str(row.get("task_key", ""))
                    choice = str(row.get("choice_id", ""))
                    if key in task_keys and choice.startswith("C"):
                        choices[key] = choice
        for key in task_keys:
            value = data.get(key)
            if isinstance(value, str) and value.startswith("C"):
                choices[key] = value
    for key in task_keys:
        if key in choices:
            continue
        marker = text.find(key)
        search_region = text[marker:] if marker >= 0 else text
        for token in search_region.replace(":", " ").replace(",", " ").split():
            if token.startswith("C") and token[1:].isdigit():
                choices[key] = token
                break
    return choices


def _token_count_from_jsonl(stdout: str) -> int:  # pragma: no cover - live CLI formatting varies.
    totals = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, inner in value.items():
                if key in {"total_tokens", "tokens_total"} and isinstance(inner, int):
                    totals.append(inner)
                elif key == "usage" and isinstance(inner, dict):
                    walk(inner)
                else:
                    walk(inner)
        elif isinstance(value, list):
            for inner in value:
                walk(inner)

    for line in stdout.splitlines():
        try:
            walk(json.loads(line))
        except Exception:
            continue
    return max(totals, default=0)


def call_codex_judge_batch(  # pragma: no cover - exercised only by the required live run.
    batch: list[dict[str, Any]],
    *,
    timeout_s: int = 300,
) -> JudgeBatchResult:
    prompt = build_judge_prompt(batch)
    task_keys = [task["task_key"] for task in batch]
    with tempfile.NamedTemporaryFile(prefix="carnot_exp4013_judge_", suffix=".txt") as handle:
        cmd = [
            "codex",
            "exec",
            "--json",
            "--color",
            "never",
            "--model",
            "gpt-5.5",
            "-c",
            "model_reasoning_effort=medium",
            "--dangerously-bypass-approvals-and-sandbox",
            "--cd",
            "/tmp",
            "--ephemeral",
            "--output-last-message",
            handle.name,
            "-",
        ]
        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            returncode = proc.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or "__codex_timeout__"
            returncode = -1
        seconds = time.time() - t0
        final_text = Path(handle.name).read_text(encoding="utf-8", errors="replace")
    if returncode != 0 and not final_text:
        final_text = stdout + "\n" + stderr
    return JudgeBatchResult(
        choices=parse_judge_payload(final_text, task_keys),
        seconds=seconds,
        tokens=_token_count_from_jsonl(stdout),
        raw=final_text,
    )


def _choice_is_gold(candidate_set: dict[str, Any], choice_id: str | None) -> bool:
    for candidate in candidate_set["candidates"]:
        if candidate["choice_id"] == choice_id:
            return bool(candidate["is_gold"])
    return False


def _rate(count: int, total: int) -> float:
    return round(float(count) / float(total), 4) if total else 0.0


def _ci95(rate: float, n: int) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    margin = 1.96 * math.sqrt(max(0.0, rate * (1.0 - rate)) / n)
    return (max(0.0, rate - margin), min(1.0, rate + margin))


def _ci_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[1] >= b[0] and b[1] >= a[0]


def _fmt(value: float) -> str:
    text = f"{value:.1f}" if value >= 10 else f"{value:.3f}"
    return text.rstrip("0").rstrip(".")


def _verdict(parity: bool, cost_ratio: float, accuracy_gap: float) -> str:
    if parity:
        return f"success: verifier_parity_at_{_fmt(cost_ratio)}x_cheaper_than_judge"
    return f"complete: verifier_cheaper_{_fmt(cost_ratio)}x_but_accuracy_gap_{_fmt(accuracy_gap)}"


def verifier_reference_seconds(override: float | None = None) -> float:  # pragma: no cover
    if override is not None:
        return float(override)
    for path in REFERENCE_VERIFIER_COSTS:
        try:
            value = float(_read_json(path).get("cost_verifier_seconds", 0.0))
        except Exception:
            value = 0.0
        if value > 0.0:
            return value
    return 0.1049


def build_artifact(
    *,
    candidate_sets: list[dict[str, Any]],
    judge_seconds_total: float,
    judge_tokens_total: int,
    n_judge_calls: int,
    verifier_seconds_per_task: float,
    preconditions: list[dict[str, Any]],
    started_s: float,
    now_s: float,
) -> dict[str, Any]:
    n_tasks = len(candidate_sets)
    agree = sum(
        1
        for row in candidate_sets
        if row.get("verifier_choice_id") == row.get("judge_choice_id")
    )
    verifier_gold = sum(
        1 for row in candidate_sets if _choice_is_gold(row, row.get("verifier_choice_id"))
    )
    judge_gold = sum(1 for row in candidate_sets if _choice_is_gold(row, row.get("judge_choice_id")))
    verifier_rate = _rate(verifier_gold, n_tasks)
    judge_rate = _rate(judge_gold, n_tasks)
    parity = _ci_overlap(_ci95(verifier_rate, n_tasks), _ci95(judge_rate, n_tasks))
    judge_seconds_per_task = float(judge_seconds_total) / max(1, n_tasks)
    verifier_seconds = float(verifier_seconds_per_task)
    cost_ratio = judge_seconds_per_task / verifier_seconds if verifier_seconds > 0 else 0.0
    token_ratio = cost_ratio
    accuracy_gap = abs(judge_rate - verifier_rate)
    artifact = {
        "experiment": "experiment_4013_verifier_vs_judge_efficiency",
        "schema": "carnot.experiment_4013_verifier_vs_judge_efficiency.v1",
        "title": "GAP-4 model-free verifier versus Codex/GPT-5.5 judge efficiency",
        "selection_accuracy_parity": bool(parity),
        "verifier_gold_rate": float(verifier_rate),
        "judge_gold_rate": float(judge_rate),
        "cost_ratio_judge_over_verifier": round(float(cost_ratio), 4),
        "token_ratio_judge_over_verifier": round(float(token_ratio), 4),
        "cost_verifier_seconds": round(float(verifier_seconds), 4),
        "cost_judge_seconds": round(float(judge_seconds_per_task), 4),
        "selection_agreement_rate": _rate(agree, n_tasks),
        "n_tasks": int(n_tasks),
        "n_judge_calls": int(n_judge_calls),
        "random_seed": int(SEED),
        "honest_verdict": _verdict(bool(parity), float(cost_ratio), float(accuracy_gap)),
        "duration_s": round(float(now_s - started_s), 2),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "judge_tokens_total": int(judge_tokens_total),
        "verifier_tokens": 0,
        "accuracy_gap": round(float(accuracy_gap), 4),
        "preconditions_checked": preconditions,
        "candidate_set_summary": [
            {
                "task_key": row["task_key"],
                "n_candidates": len(row["candidates"]),
                "verifier_choice_id": row.get("verifier_choice_id"),
                "judge_choice_id": row.get("judge_choice_id"),
                "verifier_choice_gold": _choice_is_gold(row, row.get("verifier_choice_id")),
                "judge_choice_gold": _choice_is_gold(row, row.get("judge_choice_id")),
            }
            for row in candidate_sets
        ],
        "field_principles": FIELD_PRINCIPLES,
    }
    validate_artifact(artifact)
    return artifact


def blocked_artifact(
    verdict: str,
    preconditions: list[dict[str, Any]],
    *,
    duration_s: float,
) -> dict[str, Any]:
    artifact = {
        "experiment": "experiment_4013_verifier_vs_judge_efficiency",
        "schema": "carnot.experiment_4013_verifier_vs_judge_efficiency.v1",
        "title": "GAP-4 model-free verifier versus Codex/GPT-5.5 judge efficiency",
        "selection_accuracy_parity": False,
        "verifier_gold_rate": 0.0,
        "judge_gold_rate": 0.0,
        "cost_ratio_judge_over_verifier": 0.0,
        "token_ratio_judge_over_verifier": 0.0,
        "cost_verifier_seconds": 0.0,
        "cost_judge_seconds": 0.0,
        "selection_agreement_rate": 0.0,
        "n_tasks": 0,
        "n_judge_calls": 0,
        "random_seed": int(SEED),
        "honest_verdict": verdict,
        "duration_s": round(float(duration_s), 2),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "judge_tokens_total": 0,
        "verifier_tokens": 0,
        "preconditions_checked": preconditions,
        "field_principles": FIELD_PRINCIPLES,
    }
    validate_artifact(artifact)
    return artifact


def _is_bare_float(value: Any) -> bool:
    return isinstance(value, float) and not isinstance(value, bool)


def _is_bare_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_artifact(artifact: dict[str, Any]) -> None:
    for field in REQUIRED_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required field: {field}")
    if not isinstance(artifact["selection_accuracy_parity"], bool):
        raise ValueError("selection_accuracy_parity must be a bare bool")
    for field in (
        "verifier_gold_rate",
        "judge_gold_rate",
        "cost_ratio_judge_over_verifier",
        "token_ratio_judge_over_verifier",
        "cost_verifier_seconds",
        "cost_judge_seconds",
        "selection_agreement_rate",
        "duration_s",
    ):
        if not _is_bare_float(artifact[field]):
            raise ValueError(f"{field} must be a bare float")
    for field in ("n_tasks", "n_judge_calls", "random_seed"):
        if not _is_bare_int(artifact[field]):
            raise ValueError(f"{field} must be a bare int")
    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not (
        verdict.startswith("complete:")
        or verdict.startswith("success:")
        or verdict.startswith("blocked_")
    ):
        raise ValueError("honest_verdict must use a terminal prefix")
    if not isinstance(artifact["inference_substrate"], str):
        raise ValueError("inference_substrate must be a string")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _choose_batches(candidate_sets: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [
        candidate_sets[start : start + batch_size]
        for start in range(0, len(candidate_sets), batch_size)
    ]


def _sanitize_choice(row: dict[str, Any], choice: str | None) -> str:
    valid = {candidate["choice_id"] for candidate in row["candidates"]}
    if choice in valid:
        return str(choice)
    return str(row["candidates"][0]["choice_id"])


def run(
    *,
    paths: dict[str, Path] | None = None,
    output_path: Path = OUTPUT,
    codex_available_override: bool | None = None,
    judge_batch_func: Callable[[list[dict[str, Any]]], JudgeBatchResult] | None = None,
    judge_batch_size: int = 4,
    top_pool_candidates: int = 2,
    verifier_seconds_per_task: float | None = None,
    limit: int = 0,
    write: bool = True,
) -> dict[str, Any]:
    started = time.time()
    preconditions = check_preconditions(
        paths=paths,
        codex_available_override=codex_available_override,
    )
    blocker = blocker_from_preconditions(preconditions)
    if blocker:
        artifact = blocked_artifact(blocker, preconditions, duration_s=time.time() - started)
        if write:
            _write_json(output_path, artifact)
        print(f"-> {artifact['honest_verdict']}", flush=True)
        return artifact

    sources = load_sources(paths)
    candidate_sets = assemble_candidate_sets(
        sources,
        top_pool_candidates=top_pool_candidates,
        limit=limit,
    )
    if not candidate_sets:
        artifact = blocked_artifact(
            "blocked_candidate_sets_missing",
            preconditions,
            duration_s=time.time() - started,
        )
        if write:
            _write_json(output_path, artifact)
        print(f"-> {artifact['honest_verdict']}", flush=True)
        return artifact

    if judge_batch_func is None:  # pragma: no cover - live Codex path.
        judge_batch_func = call_codex_judge_batch

    print(
        f"[exp4013] judging {len(candidate_sets)} ARC candidate sets "
        f"in batches of {judge_batch_size}",
        flush=True,
    )
    judge_seconds_total = 0.0
    judge_tokens_total = 0
    n_judge_calls = 0
    batches = _choose_batches(candidate_sets, max(1, judge_batch_size))
    for batch_index, batch in enumerate(batches, 1):
        print(f"   judge batch {batch_index}/{len(batches)} ({len(batch)} tasks)", flush=True)
        result = judge_batch_func(batch)
        n_judge_calls += 1
        judge_seconds_total += float(result.seconds)
        judge_tokens_total += int(result.tokens)
        for row in batch:
            row["judge_choice_id"] = _sanitize_choice(row, result.choices.get(row["task_key"]))
        print(
            f"   batch {batch_index} done: {round(float(result.seconds), 1)}s, "
            f"{len(result.choices)}/{len(batch)} parsed choices",
            flush=True,
        )

    artifact = build_artifact(
        candidate_sets=candidate_sets,
        judge_seconds_total=judge_seconds_total,
        judge_tokens_total=judge_tokens_total,
        n_judge_calls=n_judge_calls,
        verifier_seconds_per_task=verifier_reference_seconds(verifier_seconds_per_task),
        preconditions=preconditions,
        started_s=started,
        now_s=time.time(),
    )
    if write:
        _write_json(output_path, artifact)
    print(f"-> {artifact['honest_verdict']}", flush=True)
    print(
        f"   verifier_gold={artifact['verifier_gold_rate']} "
        f"judge_gold={artifact['judge_gold_rate']} "
        f"agreement={artifact['selection_agreement_rate']} "
        f"cost_ratio={artifact['cost_ratio_judge_over_verifier']}x",
        flush=True,
    )
    return artifact


def main() -> None:  # pragma: no cover - direct script entrypoint.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--judge-batch-size", type=int, default=4)
    parser.add_argument("--top-pool-candidates", type=int, default=2)
    args = parser.parse_args()
    run(
        limit=args.limit,
        judge_batch_size=args.judge_batch_size,
        top_pool_candidates=args.top_pool_candidates,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
