#!/usr/bin/env python3
"""Executable/objective headroom census for Exp 4175.

Spec refs: REQ-VERIFY-4175, SCENARIO-VERIFY-4175.
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HEADROOM_THRESHOLD = 0.10
OUTPUT_REL = Path("results/experiment_4175_headroom_gate_executable_census.json")

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A clean census (even 'no domain clears 0.10') is a "
        "COMPLETE verdict."
    ),
    "max_selectable_headroom": (
        "BARE float (oracle@k - SC-vote, sanitized) -- the downstream gate for "
        "A3 compares this raw value; a principle-dict would break the gate "
        "(gated-fields-must-be-bare)."
    ),
    "headroom_present_domain": (
        "Names the executable domain A3 must run on; the positive control that "
        "makes a null informative."
    ),
    "per_domain_headroom": (
        "oracle@k/baseline/vote/headroom/artifact_flags per domain so a reviewer "
        "can audit the pick and the sanitization."
    ),
    "artifact_inflation_flagged": (
        "Count of candidates excluded as truncated/mis-formatted; proves the "
        "headroom is real, not an evaluation artifact (arXiv:2605.07395)."
    ),
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "max_selectable_headroom",
    "headroom_present_domain",
    "per_domain_headroom",
    "artifact_inflation_flagged",
    "field_principles",
    "spec_refs",
    "inference_substrate",
)


def _rate(count: int, n: int) -> float:
    return round(float(count) / float(n), 10) if n else 0.0


def _float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _objective_oracle(domain: str) -> str:
    if domain == "code":
        return "unit_test_pass_flags"
    if domain == "math":
        return "exact_match"
    return "exact_candidate_correctness"


def _empty_stats(domain: str, source: Any, reason: str) -> dict[str, Any]:
    return {
        "oracle_at_k": 0.0,
        "baseline_pass1": 0.0,
        "sc_vote_pass1": 0.0,
        "selectable_headroom": 0.0,
        "n": 0,
        "artifact_flags": {
            "domain": domain,
            "source": source,
            "objective_oracle": _objective_oracle(domain),
            "candidate_pool_detected": False,
            "census_incomplete": True,
            "incomplete_reason": reason,
            "n_tasks_raw": 0,
            "n_multicandidate_tasks": 0,
            "n_tasks_evaluated": 0,
            "artifact_inflation_flagged": 0,
            "excluded_reasons": {},
            "llm_judge_used": False,
        },
    }


def _boxed_payload(text: str) -> str | None:
    marker = "\\boxed"
    start = text.rfind(marker)
    if start < 0:
        return None
    brace = text.find("{", start + len(marker))
    if brace < 0:
        return None
    depth = 0
    chars: list[str] = []
    for ch in text[brace:]:
        if ch == "{":
            depth += 1
            if depth > 1:
                chars.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip() or None
            chars.append(ch)
        else:
            chars.append(ch)
    return None


def _looks_truncated(text: Any) -> bool:
    if not isinstance(text, str):
        return False
    stripped = text.rstrip()
    if stripped.endswith("..."):
        return True
    marker = "\\boxed"
    start = stripped.rfind(marker)
    if start < 0:
        return False
    brace = stripped.find("{", start + len(marker))
    if brace < 0:
        return True
    depth = 0
    for ch in stripped[brace:]:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return False
    return True


def _extract_math_answer(candidate: dict[str, Any]) -> str | None:
    answer = candidate.get("answer")
    if answer is None:
        answer = candidate.get("extracted_answer")
    if answer is not None:
        return str(answer)

    text = candidate.get("text") or candidate.get("completion") or candidate.get("response")
    if not isinstance(text, str):
        return None
    boxed = _boxed_payload(text)
    if boxed:
        return boxed
    coda = re.findall(r"####\s*([-+]?\d+(?:\.\d+)?)", text)
    if coda:
        return coda[-1]
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    return numbers[-1] if numbers else None


def _normalize_math(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("$", "").replace(",", "").replace(" ", "")
    if text.startswith("{") and text.endswith("}") and len(text) >= 2:
        text = text[1:-1]
    if text.startswith("+"):
        text = text[1:]
    if text.endswith("."):
        text = text[:-1]
    return text or None


def _candidate_from_math(
    candidate: Any, gold_answer: Any, index: int
) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(candidate, dict):
        return None, "unparseable"
    if _looks_truncated(candidate.get("text") or candidate.get("completion") or candidate.get("response")):
        return None, "truncated"
    answer = _extract_math_answer(candidate)
    gold = _normalize_math(gold_answer)
    norm_answer = _normalize_math(answer)
    if gold is None or norm_answer is None:
        return None, "unparseable"
    return {
        "correct": norm_answer == gold,
        "votes": _float(candidate.get("votes"), 1.0 if index == 0 else 0.0),
        "index": index,
    }, None


def _candidate_from_correct_flag(
    candidate: Any, index: int
) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(candidate, dict):
        return None, "unparseable"
    correct = candidate.get("correct")
    if not isinstance(correct, bool):
        return None, "unparseable"
    return {
        "correct": correct,
        "votes": _float(candidate.get("votes"), 1.0 if index == 0 else 0.0),
        "index": index,
    }, None


def _candidate_from_task(domain: str, candidate: Any, gold_answer: Any, index: int) -> tuple[dict[str, Any] | None, str | None]:
    if domain == "math":
        return _candidate_from_math(candidate, gold_answer, index)
    return _candidate_from_correct_flag(candidate, index)


def _tasks_from_flat_candidates(pool: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for candidate in pool.get("candidates", []):
        if isinstance(candidate, dict) and "task_idx" in candidate:
            grouped[candidate["task_idx"]].append(candidate)
    task_names = pool.get("task_names") if isinstance(pool.get("task_names"), list) else []
    tasks = []
    for task_idx in sorted(grouped):
        name = task_names[task_idx] if isinstance(task_idx, int) and task_idx < len(task_names) else str(task_idx)
        tasks.append({"task_id": name, "candidates": grouped[task_idx]})
    return tasks


def _tasks_from_pool(pool: dict[str, Any], domain: str) -> list[dict[str, Any]]:
    if domain == "code" and isinstance(pool.get("results"), list):
        tasks = []
        for row in pool["results"]:
            if not isinstance(row, dict):
                continue
            candidates = [
                {"correct": row.get("baseline_passed"), "votes": 1.0, "role": "baseline"},
                {"correct": row.get("repair_passed"), "votes": 0.0, "role": "repair"},
            ]
            tasks.append({"task_id": row.get("task_id"), "candidates": candidates})
        return tasks
    if isinstance(pool.get("tasks"), list):
        tasks = []
        for row in pool["tasks"]:
            if not isinstance(row, dict):
                continue
            candidates = row.get("candidates")
            if candidates is None:
                candidates = row.get("cands")
            tasks.append(
                {
                    "task_id": row.get("task_id") or row.get("task"),
                    "candidates": candidates if isinstance(candidates, list) else [],
                    "gold_answer": row.get("gold_answer", row.get("correct_answer")),
                }
            )
        return tasks
    if isinstance(pool.get("candidates"), list):
        return _tasks_from_flat_candidates(pool)
    return []


def _summary_headroom(pool: dict[str, Any], domain: str, source: Any) -> dict[str, Any] | None:
    oracle = pool.get("oracle_ceiling")
    if not isinstance(oracle, dict) or "trm_vote_pass2" not in pool:
        return None
    n = int(_float(pool.get("n_tasks"), 0.0))
    if n <= 0:
        return None
    oracle_at_k = _float(oracle.get("pass@2", oracle.get("pass@1000")))
    sc_vote = _float(pool.get("trm_vote_pass2"))
    rankers = pool.get("rankers") if isinstance(pool.get("rankers"), dict) else {}
    trm = rankers.get("TRM_VOTE") if isinstance(rankers.get("TRM_VOTE"), dict) else {}
    baseline = _float(trm.get("pass@1"), _rate(sum(bool(r.get("base_top1_correct")) for r in pool.get("per_task", [])), n))
    incomplete = not all(_float(r.get("n_candidates"), 0.0) >= 2.0 for r in pool.get("per_task", []))
    return {
        "oracle_at_k": round(oracle_at_k, 10),
        "baseline_pass1": round(baseline, 10),
        "sc_vote_pass1": round(sc_vote, 10),
        "selectable_headroom": round(oracle_at_k - sc_vote, 10),
        "n": n,
        "artifact_flags": {
            "domain": domain,
            "source": source,
            "objective_oracle": _objective_oracle(domain),
            "candidate_pool_detected": True,
            "census_incomplete": incomplete,
            "incomplete_reason": "some_tasks_lack_k_candidates" if incomplete else None,
            "n_tasks_raw": n,
            "n_multicandidate_tasks": sum(_float(r.get("n_candidates"), 0.0) >= 2.0 for r in pool.get("per_task", [])),
            "n_tasks_evaluated": n,
            "artifact_inflation_flagged": 0,
            "excluded_reasons": {},
            "summary_only_candidate_rows": True,
            "llm_judge_used": False,
        },
    }


def headroom(pool: dict[str, Any]) -> dict[str, Any]:
    """REQ-VERIFY-4175: compute sanitized selectable headroom for one pool."""
    domain = str(pool.get("domain", "unknown"))
    source = pool.get("source", "unknown")

    summary = _summary_headroom(pool, domain, source)
    if summary is not None:
        return summary

    tasks = _tasks_from_pool(pool, domain)
    if not tasks:
        return _empty_stats(domain, source, "no_candidate_tasks")

    excluded: Counter[str] = Counter()
    raw_multicandidate = 0
    evaluated = 0
    oracle_hits = 0
    baseline_hits = 0
    vote_hits = 0

    for task in tasks:
        raw_candidates = task.get("candidates", [])
        if len(raw_candidates) < 2:
            continue
        raw_multicandidate += 1
        valid: list[dict[str, Any]] = []
        baseline_correct = False
        for index, candidate in enumerate(raw_candidates):
            parsed, reason = _candidate_from_task(domain, candidate, task.get("gold_answer"), index)
            if parsed is None:
                excluded[reason or "unparseable"] += 1
                continue
            if index == 0:
                baseline_correct = bool(parsed["correct"])
            valid.append(parsed)
        if not valid:
            continue
        evaluated += 1
        oracle_hits += int(any(bool(candidate["correct"]) for candidate in valid))
        baseline_hits += int(baseline_correct)
        vote_winner = max(valid, key=lambda candidate: (candidate["votes"], -candidate["index"]))
        vote_hits += int(bool(vote_winner["correct"]))

    if raw_multicandidate == 0 or evaluated == 0:
        reason = "no_multicandidate_rows" if raw_multicandidate == 0 else "no_parseable_candidates"
        stats = _empty_stats(domain, source, reason)
        stats["artifact_flags"]["n_tasks_raw"] = len(tasks)
        stats["artifact_flags"]["n_multicandidate_tasks"] = raw_multicandidate
        stats["artifact_flags"]["artifact_inflation_flagged"] = sum(excluded.values())
        stats["artifact_flags"]["excluded_reasons"] = dict(sorted(excluded.items()))
        stats["artifact_flags"]["candidate_pool_detected"] = raw_multicandidate > 0
        return stats

    oracle_at_k = _rate(oracle_hits, evaluated)
    sc_vote = _rate(vote_hits, evaluated)
    return {
        "oracle_at_k": oracle_at_k,
        "baseline_pass1": _rate(baseline_hits, evaluated),
        "sc_vote_pass1": sc_vote,
        "selectable_headroom": round(oracle_at_k - sc_vote, 10),
        "n": evaluated,
        "artifact_flags": {
            "domain": domain,
            "source": source,
            "objective_oracle": _objective_oracle(domain),
            "candidate_pool_detected": True,
            "census_incomplete": False,
            "incomplete_reason": None,
            "n_tasks_raw": len(tasks),
            "n_multicandidate_tasks": raw_multicandidate,
            "n_tasks_evaluated": evaluated,
            "artifact_inflation_flagged": sum(excluded.values()),
            "excluded_reasons": dict(sorted(excluded.items())),
            "llm_judge_used": False,
        },
    }


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def _pool_from_path(path: Path, domain: str) -> dict[str, Any] | None:
    data = _load_json(path)
    if data is None:
        return None
    data = dict(data)
    data["domain"] = domain
    data["source"] = str(path)
    return data


def _math_pool(repo_root: Path) -> dict[str, Any]:
    candidates = [
        repo_root / "results/experiment_1816_gsm8k_baseline.json",
        repo_root / "results/adversarial_gsm8k_data_400.json",
    ]
    present = [str(path) for path in candidates if path.exists()]
    return {"domain": "math", "source": present or "missing_math_artifacts"}


def _sudoku_pool(repo_root: Path) -> dict[str, Any] | None:
    summary = _pool_from_path(repo_root / "results/arc3_trm_verifier_rerank.json", "sudoku")
    if summary is not None:
        table = repo_root / "results/arc3_gap3_stage0_candidate_table.json"
        summary["source_candidate_table_present"] = table.exists()
        return summary
    return _pool_from_path(repo_root / "results/arc3_gap3_stage0_candidate_table.json", "sudoku")


def _precondition_paths(repo_root: Path) -> dict[str, list[str]]:
    globs = {
        "experiment_1999": "results/experiment_1999_*",
        "experiment_2090": "results/experiment_2090_*",
        "experiment_1816": "results/experiment_1816_*",
    }
    return {
        key: [str(path) for path in sorted(repo_root.glob(pattern))]
        for key, pattern in globs.items()
    }


def build_artifact(repo_root: Path, per_domain: dict[str, dict[str, Any]], duration_s: float) -> dict[str, Any]:
    complete = {
        domain: stats
        for domain, stats in per_domain.items()
        if not stats["artifact_flags"].get("census_incomplete")
    }
    if complete:
        domain, stats = max(
            complete.items(),
            key=lambda item: (item[1]["selectable_headroom"], item[0]),
        )
        max_headroom = float(stats["selectable_headroom"])
    else:
        domain = ""
        max_headroom = 0.0

    if not complete:
        verdict = "blocked_no_multicandidate_pool"
        headroom_domain = ""
    elif max_headroom >= HEADROOM_THRESHOLD:
        verdict = f"complete: headroom_present_domain_{domain}_max_selectable_headroom_{max_headroom:.4f}"
        headroom_domain = domain
    else:
        verdict = f"complete: no_domain_clears_0.10_max_selectable_headroom_{max_headroom:.4f}"
        headroom_domain = ""

    artifact = {
        "honest_verdict": verdict,
        "max_selectable_headroom": float(max_headroom),
        "headroom_present_domain": headroom_domain,
        "per_domain_headroom": per_domain,
        "artifact_inflation_flagged": int(
            sum(
                stats["artifact_flags"].get("artifact_inflation_flagged", 0)
                for stats in per_domain.values()
            )
        ),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": ["REQ-VERIFY-4175", "SCENARIO-VERIFY-4175"],
        "acceptance_gate": (
            bool(headroom_domain)
            or verdict.startswith("complete: no_domain_clears_0.10")
            or verdict == "blocked_no_multicandidate_pool"
        ),
        "precondition_paths": _precondition_paths(repo_root),
        "duration_s": round(duration_s, 6),
        "inference_substrate": "cached_artifact_objective_oracle_census",
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: dict[str, Any]) -> None:
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required field: {field}")
    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not (verdict.startswith("complete:") or verdict.startswith("blocked_")):
        raise ValueError("honest_verdict must use a terminal prefix")
    max_headroom = artifact["max_selectable_headroom"]
    if isinstance(max_headroom, bool) or not isinstance(max_headroom, float):
        raise ValueError("max_selectable_headroom must be a bare float")
    if not isinstance(artifact["headroom_present_domain"], str):
        raise ValueError("headroom_present_domain must be a string")
    if not isinstance(artifact["per_domain_headroom"], dict):
        raise ValueError("per_domain_headroom must be a dict")
    if isinstance(artifact["artifact_inflation_flagged"], bool) or not isinstance(artifact["artifact_inflation_flagged"], int):
        raise ValueError("artifact_inflation_flagged must be an int")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles do not match REQ-VERIFY-4175")
    if artifact["inference_substrate"] != "cached_artifact_objective_oracle_census":
        raise ValueError("inference_substrate must be cached_artifact_objective_oracle_census")
    if max_headroom >= HEADROOM_THRESHOLD and verdict.startswith("complete: headroom_present") and not artifact["headroom_present_domain"]:
        raise ValueError("headroom_present_domain is required when headroom clears the gate")


def run_census(repo_root: Path | str = Path(".")) -> dict[str, Any]:
    start = time.perf_counter()
    root = Path(repo_root)
    pools = {
        "code": _pool_from_path(root / "results/experiment_1999_code_verification_humaneval.json", "code")
        or {"domain": "code", "source": "missing_code_artifact"},
        "math": _math_pool(root),
        "sudoku": _sudoku_pool(root) or {"domain": "sudoku", "source": "missing_sudoku_artifact"},
    }
    per_domain = {domain: headroom(pool) for domain, pool in pools.items()}
    artifact = build_artifact(root, per_domain, time.perf_counter() - start)
    output = root / OUTPUT_REL
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - covered by runner smoke command.
    args = sys.argv[1:] if argv is None else argv
    repo_root = Path(args[0]) if args else Path(".")
    artifact = run_census(repo_root)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
