"""Exp 1274 certificate-memory replay evaluation.

This module turns verified certificate outputs, or verified FoVer labels when
certificates are unavailable, into a small replayable case-memory table. The
evaluation compares a no-memory routing baseline with a memory lookup keyed by
constraint pattern, verifier result, and repair hint.

Spec: REQ-LEARN-1274, SCENARIO-LEARN-1275.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXP1271_PATH = (
    REPO_ROOT / "results" / "experiment_1271_triggered_certificate_extraction_sota_gguf.json"
)
DEFAULT_FOVER_PATH = REPO_ROOT / "results" / "fover_corpus_v5.json"
DEFAULT_RESULT_PATH = (
    REPO_ROOT / "results" / "experiment_1274_online_self_learning_certificate_memory_v3.json"
)

EXPERIMENT_NAME = "1274_online_self_learning_certificate_memory_v3"
SCHEMA = "certificate_memory_replay_v3"
RUN_DATE = "20260504"
DEFAULT_BUILD_FRACTION = 0.60

Decision = Literal["accept", "repair"]

CERTIFICATE_KEYS = (
    "certificates",
    "certificate_outputs",
    "verification_certificates",
    "per_step_certificates",
)
REQUIRED_ARTIFACT_FIELDS = (
    "experiment",
    "schema",
    "run_date",
    "status",
    "source",
    "before_score",
    "after_score",
    "self_learning_delta_overall",
    "memory_entries",
    "skill_graph_candidate_count",
    "honest_verdict",
)


@dataclass(frozen=True)
class CertificateMemoryExample:
    """One verified case that can be stored and replayed as certificate memory."""

    example_id: str
    source: str
    question: str
    response: str
    is_correct: bool
    constraint_pattern: str
    verifier_result: Literal["passed", "failed"]
    repair_hint: str
    target_decision: Decision


def _utc_now() -> str:
    return _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_json_if_exists(path: Path | str | None) -> Any | None:
    if path is None:
        return None
    candidate = Path(path)
    return _read_json(candidate) if candidate.exists() else None


def _write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_in_progress_artifact(
    output_path: Path | str = DEFAULT_RESULT_PATH,
    *,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """Write the REQ-LEARN-1274 in-progress artifact skeleton."""

    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "schema": SCHEMA,
        "run_date": run_date,
        "status": "in_progress",
        "honest_verdict": "in_progress",
    }
    _write_json(output_path, artifact)
    return artifact


def infer_constraint_pattern(question: str, response: str) -> str:
    """Infer a coarse replay pattern from the question/response text."""

    text = f"{question} {response}".lower()
    patterns = (
        ((" plus ", " + ", "sum", "total"), "arithmetic:addition"),
        (("left over", "remaining", "still need", "budget"), "arithmetic:balance"),
        (("twice", "half", "times as long", "percent", "%"), "arithmetic:ratio"),
    )
    for needles, pattern in patterns:
        if any(needle in text for needle in needles):
            return pattern
    return "arithmetic:general"


def infer_repair_hint(is_correct: bool, constraint_pattern: str) -> str:
    """Choose the replay repair hint recorded with a verified example."""

    if is_correct:
        return "accept_verified_answer"
    if constraint_pattern.startswith("arithmetic:"):
        return "recompute_arithmetic_result"
    return "repair_constraint_violation"


def _normalise_verifier_result(value: Any, *, default_correct: bool | None = None) -> Literal["passed", "failed"]:
    raw = str(value).strip().lower()
    if raw in {"passed", "pass", "correct", "sat", "true", "verified", "1"}:
        return "passed"
    if raw in {"failed", "fail", "incorrect", "unsat", "false", "violated", "0"}:
        return "failed"
    return "passed" if bool(default_correct) else "failed"


def _target_decision(is_correct: bool) -> Decision:
    return "accept" if is_correct else "repair"


def _certificate_rows(payload: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None) -> list[Mapping[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, Mapping):
        if payload.get("status") == "blocked":
            return []
        for key in CERTIFICATE_KEYS:
            rows = payload.get(key)
            if rows:
                return [row for row in rows if isinstance(row, Mapping)]
        return []
    return [row for row in payload if isinstance(row, Mapping)]


def _examples_from_exp1271(payload: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None) -> list[CertificateMemoryExample]:
    examples: list[CertificateMemoryExample] = []
    for index, row in enumerate(_certificate_rows(payload)):
        question = str(row.get("question") or row.get("prompt") or "")
        response = str(row.get("response") or row.get("completion") or row.get("answer") or "")
        is_correct = bool(row.get("is_correct", row.get("verified", False)))
        verifier_result = _normalise_verifier_result(
            row.get("verifier_result", row.get("z3_verdict", row.get("label"))),
            default_correct=is_correct,
        )
        is_correct = verifier_result == "passed"
        pattern = str(row.get("constraint_pattern") or row.get("constraint_type") or "").strip()
        if not pattern:
            pattern = infer_constraint_pattern(question, response)
        repair_hint = str(row.get("repair_hint") or "").strip()
        if not repair_hint:
            repair_hint = infer_repair_hint(is_correct, pattern)
        examples.append(
            CertificateMemoryExample(
                example_id=str(row.get("id") or row.get("example_id") or f"exp1271_{index:04d}"),
                source="exp1271",
                question=question,
                response=response,
                is_correct=is_correct,
                constraint_pattern=pattern,
                verifier_result=verifier_result,
                repair_hint=repair_hint,
                target_decision=_target_decision(is_correct),
            )
        )
    return examples


def _examples_from_fover(payload: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> list[CertificateMemoryExample]:
    pairs: Sequence[Mapping[str, Any]]
    if isinstance(payload, Mapping):
        pairs = payload.get("pairs", [])  # type: ignore[assignment]
    else:
        pairs = payload

    examples: list[CertificateMemoryExample] = []
    for index, pair in enumerate(pairs):
        if not isinstance(pair, Mapping):
            continue
        question = str(pair.get("question", ""))
        response = str(pair.get("response", ""))
        is_correct = bool(pair.get("is_correct", False))
        pattern = infer_constraint_pattern(question, response)
        verifier_result: Literal["passed", "failed"] = "passed" if is_correct else "failed"
        examples.append(
            CertificateMemoryExample(
                example_id=str(pair.get("question_index", f"fover_{index:04d}")) + f":{index:04d}",
                source="fover_fallback",
                question=question,
                response=response,
                is_correct=is_correct,
                constraint_pattern=pattern,
                verifier_result=verifier_result,
                repair_hint=infer_repair_hint(is_correct, pattern),
                target_decision=_target_decision(is_correct),
            )
        )
    return examples


def load_certificate_examples(
    *,
    exp1271_path: Path | str | None = DEFAULT_EXP1271_PATH,
    fover_path: Path | str = DEFAULT_FOVER_PATH,
) -> tuple[list[CertificateMemoryExample], str]:
    """Load Exp 1271 certificates, falling back to verified FoVer pairs."""

    exp1271_payload = _read_json_if_exists(exp1271_path)
    exp1271_examples = _examples_from_exp1271(exp1271_payload)
    if exp1271_examples:
        return exp1271_examples, "exp1271"
    return _examples_from_fover(_read_json(fover_path)), "fover_fallback"


def memory_key(example: CertificateMemoryExample) -> tuple[str, str, str]:
    """Return the REQ-LEARN-1274 memory key for one example."""

    return (example.constraint_pattern, example.verifier_result, example.repair_hint)


def _majority_decision(counter: Counter[str]) -> Decision:
    if counter["repair"] > counter["accept"]:
        return "repair"
    return "accept"


def build_memory_table(examples: Sequence[CertificateMemoryExample]) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Build a case-memory table keyed by pattern, verifier result, and hint."""

    buckets: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    for example in examples:
        buckets[memory_key(example)][example.target_decision] += 1

    memory: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, counter in buckets.items():
        memory[key] = {
            "constraint_pattern": key[0],
            "verifier_result": key[1],
            "repair_hint": key[2],
            "evidence_count": int(sum(counter.values())),
            "decision_counts": dict(counter),
            "selected_decision": _majority_decision(counter),
        }
    return memory


def split_examples(
    examples: Sequence[CertificateMemoryExample],
    *,
    build_fraction: float = DEFAULT_BUILD_FRACTION,
) -> tuple[list[CertificateMemoryExample], list[CertificateMemoryExample]]:
    """Split examples into memory-build and replay-eval slices."""

    if not 0.0 < build_fraction < 1.0:
        raise ValueError("build_fraction must be between 0 and 1")
    items = list(examples)
    if len(items) < 2:
        return items, []
    split_index = max(1, min(len(items) - 1, int(len(items) * build_fraction)))
    return items[:split_index], items[split_index:]


def baseline_decision(example: CertificateMemoryExample) -> Decision:
    """Return the no-memory routing decision for one replay example."""

    if re.search(r"\banswer\s+is\s+42\b|\bfinal answer:\s*42\b", example.response, re.IGNORECASE):
        return "repair"
    return "accept"


def memory_augmented_decision(
    example: CertificateMemoryExample,
    memory: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> Decision:
    """Return the replay decision after consulting certificate memory."""

    entry = memory.get(memory_key(example))
    if entry is None:
        return baseline_decision(example)
    return "repair" if entry["selected_decision"] == "repair" else "accept"


def _score_decisions(
    examples: Sequence[CertificateMemoryExample],
    decisions: Sequence[Decision],
) -> float:
    if not examples:
        return 0.0
    correct = sum(
        1
        for example, decision in zip(examples, decisions, strict=True)
        if decision == example.target_decision
    )
    return round(float(correct / len(examples)), 6)


def build_skill_graph_candidates(
    memory: Mapping[tuple[str, str, str], Mapping[str, Any]],
    replay_stats: Mapping[tuple[str, str, str], Mapping[str, int]],
) -> list[dict[str, Any]]:
    """Emit replayable skill-graph candidates from memory entries with eval support."""

    candidates: list[dict[str, Any]] = []
    for key, entry in sorted(memory.items()):
        stats = replay_stats.get(key, {"count": 0, "success": 0})
        if not stats["count"]:
            continue
        success_rate = round(float(stats["success"] / stats["count"]), 6)
        candidates.append(
            {
                "contract": (
                    f"if constraint_pattern={entry['constraint_pattern']} and "
                    f"verifier_result={entry['verifier_result']} then "
                    f"route={entry['selected_decision']}"
                ),
                "evidence_count": int(entry["evidence_count"]),
                "replay_success_rate": success_rate,
                "demotion_condition": "demote if replay_success_rate < 0.50 over 20 future replays",
            }
        )
    return candidates


def run_replay_evaluation(
    examples: Sequence[CertificateMemoryExample],
    *,
    build_fraction: float = DEFAULT_BUILD_FRACTION,
) -> dict[str, Any]:
    """Measure before/after replay decision quality for certificate memory."""

    build_slice, eval_slice = split_examples(examples, build_fraction=build_fraction)
    memory = build_memory_table(build_slice)
    before_decisions = [baseline_decision(example) for example in eval_slice]
    after_decisions = [memory_augmented_decision(example, memory) for example in eval_slice]

    replay_stats: dict[tuple[str, str, str], dict[str, int]] = defaultdict(lambda: {"count": 0, "success": 0})
    for example, decision in zip(eval_slice, after_decisions, strict=True):
        key = memory_key(example)
        if key in memory:
            replay_stats[key]["count"] += 1
            replay_stats[key]["success"] += int(decision == example.target_decision)

    before_score = _score_decisions(eval_slice, before_decisions)
    after_score = _score_decisions(eval_slice, after_decisions)
    delta = round(after_score - before_score, 6)
    candidates = build_skill_graph_candidates(memory, replay_stats)
    return {
        "n_examples": len(examples),
        "n_memory_build_examples": len(build_slice),
        "n_replay_eval_examples": len(eval_slice),
        "before_score": before_score,
        "after_score": after_score,
        "self_learning_delta_overall": delta,
        "memory_entries": len(memory),
        "skill_graph_candidates": candidates,
        "skill_graph_candidate_count": len(candidates),
    }


def derive_honest_verdict(delta: float) -> str:
    """Classify the measured replay delta without suppressing regressions."""

    if delta > 0.0:
        return "certificate_memory_replay_improved"
    if delta < 0.0:
        return "certificate_memory_replay_regressed"
    return "certificate_memory_replay_neutral"


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Assert the final Exp 1274 artifact satisfies REQ-LEARN-1274."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    if artifact["status"] != "complete":
        raise AssertionError("final artifact status must be complete")
    if artifact["source"] not in {"exp1271", "fover_fallback"}:
        raise AssertionError("source must be exp1271 or fover_fallback")
    expected_delta = round(float(artifact["after_score"]) - float(artifact["before_score"]), 6)
    if float(artifact["self_learning_delta_overall"]) != expected_delta:
        raise AssertionError("self_learning_delta_overall must equal after_score - before_score")
    if int(artifact["memory_entries"]) < 0:
        raise AssertionError("memory_entries must be non-negative")
    candidates = artifact.get("skill_graph_candidates", [])
    if int(artifact["skill_graph_candidate_count"]) != len(candidates):
        raise AssertionError("skill_graph_candidate_count must match candidates")
    required_candidate_fields = {
        "contract",
        "evidence_count",
        "replay_success_rate",
        "demotion_condition",
    }
    for candidate in candidates:
        if not required_candidate_fields <= set(candidate):
            raise AssertionError("skill_graph_candidates missing required fields")


def run_experiment(
    *,
    exp1271_path: Path | str | None = DEFAULT_EXP1271_PATH,
    fover_path: Path | str = DEFAULT_FOVER_PATH,
    output_path: Path | str = DEFAULT_RESULT_PATH,
    run_date: str = RUN_DATE,
    project_root: str = "/home/ianblenke/github.com/ianblenke/carnot",
    build_fraction: float = DEFAULT_BUILD_FRACTION,
) -> dict[str, Any]:
    """Run Exp 1274 and persist the final certificate-memory replay artifact."""

    write_in_progress_artifact(output_path, run_date=run_date)
    started_at = _utc_now()
    examples, source = load_certificate_examples(exp1271_path=exp1271_path, fover_path=fover_path)
    evaluation = run_replay_evaluation(examples, build_fraction=build_fraction)
    finished_at = _utc_now()
    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "schema": SCHEMA,
        "run_date": run_date,
        "started_at": started_at,
        "finished_at": finished_at,
        "status": "complete",
        "source": source,
        "source_artifacts": {
            "exp1271": str(exp1271_path) if exp1271_path is not None else None,
            "fover_corpus": str(fover_path),
        },
        "project_root": project_root,
        "artifact_metadata": {"project_root": project_root, "run_date": run_date},
    }
    artifact.update(evaluation)
    artifact["honest_verdict"] = derive_honest_verdict(
        float(artifact["self_learning_delta_overall"])
    )
    validate_artifact(artifact)
    _write_json(output_path, artifact)
    return artifact


def main() -> int:  # pragma: no cover
    artifact = run_experiment()
    print(
        artifact["honest_verdict"],
        artifact["self_learning_delta_overall"],
        artifact["memory_entries"],
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
