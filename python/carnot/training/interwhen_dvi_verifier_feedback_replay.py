"""Exp 1288 InterWhen DVI verifier-feedback replay.

This module compares a frozen post-hoc replay policy with an online policy that
observes each verifier accept/reject decision before routing the next item.

Spec: REQ-LEARN-1288, SCENARIO-LEARN-1288.
"""

from __future__ import annotations

import datetime as _dt
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from carnot.training import certificate_memory_replay as cert


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOTA_PATHS = (
    REPO_ROOT / "results" / "experiment_1285_triggered_certificate_extraction_v2.json",
    REPO_ROOT / "results" / "experiment_1286_grad_beaver_nsvif_semantic_routing.json",
)
DEFAULT_EXP1274_PATH = (
    REPO_ROOT / "results" / "experiment_1274_online_self_learning_certificate_memory_v3.json"
)
DEFAULT_FOVER_PATH = REPO_ROOT / "results" / "fover_corpus_v5.json"
DEFAULT_RESULT_PATH = (
    REPO_ROOT / "results" / "experiment_1288_interwhen_dvi_verifier_feedback_replay.json"
)

EXPERIMENT_NAME = "1288_interwhen_dvi_verifier_feedback_replay"
SCHEMA = "interwhen_dvi_verifier_feedback_replay_v1"
RUN_DATE = "20260504"
DEFAULT_BUILD_FRACTION = 0.60

Decision = Literal["accept", "repair"]
VerifierResult = Literal["passed", "failed"]
PolicyKey = tuple[str, str, str]
PolicyCounters = dict[PolicyKey, Counter[str]]

SOTA_RECORD_KEYS = (
    "verification_certificates",
    "certificates",
    "certificate_outputs",
    "routing_records",
    "records",
)
REQUIRED_ARTIFACT_FIELDS = (
    "experiment",
    "schema",
    "run_date",
    "status",
    "source",
    "dvi_acceptance_delta",
    "online_acceptance_delta",
    "violation_delta",
    "self_learning_delta_overall",
    "self_verify_signal_used",
    "verification_gain",
    "reasoning_trace_length_delta",
    "claim_level_memory_entries",
    "memory_update_written",
    "headline_result_allowed",
    "honest_verdict",
    "replay_slices",
)


@dataclass(frozen=True)
class FeedbackReplayExample:
    """One chronological verifier-feedback case for online replay."""

    example_id: str
    source: str
    question: str
    response: str
    is_correct: bool
    constraint_pattern: str
    verifier_result: VerifierResult
    repair_hint: str
    target_decision: Decision
    reasoning_trace_length: int


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
    """Write the REQ-LEARN-1288 in-progress artifact skeleton."""

    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "schema": SCHEMA,
        "run_date": run_date,
        "status": "in_progress",
        "honest_verdict": "in_progress",
        "headline_result_allowed": False,
    }
    _write_json(output_path, artifact)
    return artifact


def _trace_length(response: str) -> int:
    words = [word for word in response.replace("\n", " ").split(" ") if word]
    return max(1, len(words))


def _to_feedback_example(
    example: cert.CertificateMemoryExample,
    *,
    source: str,
) -> FeedbackReplayExample:
    return FeedbackReplayExample(
        example_id=example.example_id,
        source=source,
        question=example.question,
        response=example.response,
        is_correct=example.is_correct,
        constraint_pattern=example.constraint_pattern,
        verifier_result=example.verifier_result,
        repair_hint=example.repair_hint,
        target_decision=example.target_decision,
        reasoning_trace_length=_trace_length(example.response),
    )


def _to_certificate_example(example: FeedbackReplayExample) -> cert.CertificateMemoryExample:
    return cert.CertificateMemoryExample(
        example_id=example.example_id,
        source=example.source,
        question=example.question,
        response=example.response,
        is_correct=example.is_correct,
        constraint_pattern=example.constraint_pattern,
        verifier_result=example.verifier_result,
        repair_hint=example.repair_hint,
        target_decision=example.target_decision,
    )


def _sota_rows(payload: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    if payload is None or payload.get("status") != "complete":
        return []
    for key in SOTA_RECORD_KEYS:
        rows = payload.get(key)
        if isinstance(rows, list) and rows:
            return [row for row in rows if isinstance(row, Mapping)]
    return []


def _examples_from_sota(payload: Mapping[str, Any] | None) -> list[FeedbackReplayExample]:
    examples: list[FeedbackReplayExample] = []
    for index, row in enumerate(_sota_rows(payload)):
        question = str(row.get("question") or row.get("prompt") or "")
        response = str(row.get("response") or row.get("completion") or row.get("answer") or "")
        verifier_result = cert._normalise_verifier_result(
            row.get("verifier_result", row.get("routing_decision", row.get("label"))),
            default_correct=bool(row.get("is_correct", row.get("verified", False))),
        )
        is_correct = verifier_result == "passed"
        pattern = str(row.get("constraint_pattern") or row.get("constraint_type") or "").strip()
        if not pattern:
            pattern = cert.infer_constraint_pattern(question, response)
        repair_hint = str(row.get("repair_hint") or "").strip()
        if not repair_hint:
            repair_hint = cert.infer_repair_hint(is_correct, pattern)
        target_decision: Decision = "accept" if is_correct else "repair"
        examples.append(
            FeedbackReplayExample(
                example_id=str(row.get("id") or row.get("example_id") or f"sota_{index:04d}"),
                source="sota_certificates",
                question=question,
                response=response,
                is_correct=is_correct,
                constraint_pattern=pattern,
                verifier_result=verifier_result,
                repair_hint=repair_hint,
                target_decision=target_decision,
                reasoning_trace_length=_trace_length(response),
            )
        )
    return examples


def load_feedback_examples(
    *,
    sota_paths: Sequence[Path | str] = DEFAULT_SOTA_PATHS,
    exp1274_path: Path | str | None = DEFAULT_EXP1274_PATH,
    fover_path: Path | str = DEFAULT_FOVER_PATH,
) -> tuple[list[FeedbackReplayExample], str]:
    """Load SOTA certificates when usable, otherwise Exp 1274/FoVer fallback cases."""

    for path in sota_paths:
        examples = _examples_from_sota(_read_json_if_exists(path))
        if examples:
            return examples, "sota_certificates"

    _read_json_if_exists(exp1274_path)
    certificate_examples, _source = cert.load_certificate_examples(
        exp1271_path=None,
        fover_path=fover_path,
    )
    return [
        _to_feedback_example(example, source="fover_fallback") for example in certificate_examples
    ], "fover_fallback"


def _split_examples(
    examples: Sequence[FeedbackReplayExample],
    *,
    build_fraction: float,
) -> tuple[list[FeedbackReplayExample], list[FeedbackReplayExample]]:
    if not 0.0 < build_fraction < 1.0:
        raise ValueError("build_fraction must be between 0 and 1")
    items = list(examples)
    if len(items) < 2:
        return items, []
    split_index = max(1, min(len(items) - 1, int(len(items) * build_fraction)))
    return items[:split_index], items[split_index:]


def _policy_key(example: FeedbackReplayExample) -> PolicyKey:
    return (example.constraint_pattern, example.verifier_result, example.repair_hint)


def _observe(policy: PolicyCounters, example: FeedbackReplayExample) -> None:
    policy.setdefault(_policy_key(example), Counter())[example.target_decision] += 1


def _build_policy(examples: Sequence[FeedbackReplayExample]) -> PolicyCounters:
    policy: PolicyCounters = {}
    for example in examples:
        _observe(policy, example)
    return policy


def _clone_policy(policy: PolicyCounters) -> PolicyCounters:
    return {key: Counter(counter) for key, counter in policy.items()}


def _selected_decision(counter: Counter[str]) -> Decision:
    return "repair" if counter["repair"] > counter["accept"] else "accept"


def _policy_decision(example: FeedbackReplayExample, policy: Mapping[PolicyKey, Counter[str]]) -> Decision:
    counter = policy.get(_policy_key(example))
    if counter is None:
        return cert.baseline_decision(_to_certificate_example(example))
    return _selected_decision(counter)


def _score_decisions(
    examples: Sequence[FeedbackReplayExample],
    decisions: Sequence[Decision],
) -> float:
    if not examples:
        return 0.0
    correct = sum(
        1
        for example, decision in zip(examples, decisions, strict=True)
        if decision == example.target_decision
    )
    return round(correct / len(examples), 6)


def _violation_rate(
    examples: Sequence[FeedbackReplayExample],
    decisions: Sequence[Decision],
) -> float:
    if not examples:
        return 0.0
    false_accepts = sum(
        1
        for example, decision in zip(examples, decisions, strict=True)
        if decision == "accept" and not example.is_correct
    )
    return round(false_accepts / len(examples), 6)


def _decision_trace_length(example: FeedbackReplayExample, decision: Decision) -> int:
    return example.reasoning_trace_length + int(decision == "repair")


def _mean_trace_length(
    examples: Sequence[FeedbackReplayExample],
    decisions: Sequence[Decision],
) -> float:
    if not examples:
        return 0.0
    total = sum(
        _decision_trace_length(example, decision)
        for example, decision in zip(examples, decisions, strict=True)
    )
    return round(total / len(examples), 6)


def _policy_state(policy: Mapping[PolicyKey, Counter[str]]) -> dict[str, Any]:
    support = sum(sum(counter.values()) for counter in policy.values())
    return {
        "memory_entries": len(policy),
        "support": support,
        "repair_routes": sum(1 for counter in policy.values() if _selected_decision(counter) == "repair"),
        "accept_routes": sum(1 for counter in policy.values() if _selected_decision(counter) == "accept"),
    }


def _clause_prediction_records(policy: Mapping[PolicyKey, Counter[str]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for key in sorted(policy):
        counter = policy[key]
        records.append(
            {
                "constraint_pattern": key[0],
                "verifier_result": key[1],
                "repair_hint": key[2],
                "selected_decision": _selected_decision(counter),
                "support": int(sum(counter.values())),
            }
        )
    return records


def compare_replay_modes(
    examples: Sequence[FeedbackReplayExample],
    *,
    build_fraction: float = DEFAULT_BUILD_FRACTION,
) -> dict[str, Any]:
    """Compare frozen post-hoc replay against online verifier-feedback replay."""

    build_slice, eval_slice = _split_examples(examples, build_fraction=build_fraction)
    frozen_policy = _build_policy(build_slice)
    online_policy = _clone_policy(frozen_policy)

    baseline_decisions = [cert.baseline_decision(_to_certificate_example(example)) for example in eval_slice]
    frozen_decisions = [_policy_decision(example, frozen_policy) for example in eval_slice]
    online_decisions: list[Decision] = []
    replay_slices: list[dict[str, Any]] = []

    for index, example in enumerate(eval_slice):
        before_state = _policy_state(online_policy)
        online_decision = _policy_decision(example, online_policy)
        _observe(online_policy, example)
        after_state = _policy_state(online_policy)
        online_decisions.append(online_decision)
        replay_slices.append(
            {
                "case_id": example.example_id,
                "chronological_index": index,
                "verifier_result": example.verifier_result,
                "target_decision": example.target_decision,
                "posthoc_decision": frozen_decisions[index],
                "online_decision": online_decision,
                "before_policy_state": before_state,
                "after_policy_state": after_state,
            }
        )

    baseline_score = _score_decisions(eval_slice, baseline_decisions)
    frozen_score = _score_decisions(eval_slice, frozen_decisions)
    online_score = _score_decisions(eval_slice, online_decisions)
    frozen_violation_rate = _violation_rate(eval_slice, frozen_decisions)
    online_violation_rate = _violation_rate(eval_slice, online_decisions)
    self_learning_delta = round(online_score - frozen_score, 6)
    trace_delta = round(
        _mean_trace_length(eval_slice, online_decisions)
        - _mean_trace_length(eval_slice, frozen_decisions),
        6,
    )

    return {
        "n_examples": len(examples),
        "n_memory_build_examples": len(build_slice),
        "n_replay_eval_examples": len(eval_slice),
        "baseline_acceptance_score": baseline_score,
        "posthoc_acceptance_score": frozen_score,
        "online_acceptance_score": online_score,
        "dvi_acceptance_delta": round(frozen_score - baseline_score, 6),
        "online_acceptance_delta": round(online_score - baseline_score, 6),
        "posthoc_violation_rate": frozen_violation_rate,
        "online_violation_rate": online_violation_rate,
        "violation_delta": round(online_violation_rate - frozen_violation_rate, 6),
        "self_learning_delta_overall": self_learning_delta,
        "self_verify_signal_used": bool(eval_slice),
        "verification_gain": self_learning_delta,
        "reasoning_trace_length_delta": trace_delta,
        "claim_level_memory_entries": len(online_policy),
        "clause_prediction_records": _clause_prediction_records(online_policy),
        "memory_update_written": bool(eval_slice),
        "replay_slices": replay_slices,
    }


def derive_honest_verdict(delta: float, *, headline_allowed: bool) -> str:
    """Classify online replay without hiding neutral or regressed outcomes."""

    if delta > 0.0:
        outcome = "improved"
    elif delta < 0.0:
        outcome = "regressed"
    else:
        outcome = "neutral"
    suffix = "headline_candidate" if headline_allowed else "non_headline"
    return f"online_verifier_feedback_{outcome}_{suffix}"


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Assert the final Exp 1288 artifact satisfies REQ-LEARN-1288."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    if artifact["status"] != "complete":
        raise AssertionError("final artifact status must be complete")
    if artifact["source"] not in {"sota_certificates", "fover_fallback"}:
        raise AssertionError("unsupported source")
    expected_gain = round(
        float(artifact["online_acceptance_score"]) - float(artifact["posthoc_acceptance_score"]),
        6,
    )
    if float(artifact["verification_gain"]) != expected_gain:
        raise AssertionError("verification_gain must equal online minus posthoc acceptance")
    if float(artifact["self_learning_delta_overall"]) != float(artifact["verification_gain"]):
        raise AssertionError("self_learning_delta_overall must equal verification_gain")
    if not isinstance(artifact["memory_update_written"], bool):
        raise AssertionError("memory_update_written must be boolean")
    if not isinstance(artifact["headline_result_allowed"], bool):
        raise AssertionError("headline_result_allowed must be boolean")
    if not isinstance(artifact["replay_slices"], list):
        raise AssertionError("replay_slices must be a list")
    if int(artifact["claim_level_memory_entries"]) < 0:
        raise AssertionError("claim_level_memory_entries must be non-negative")


def run_experiment(
    *,
    sota_paths: Sequence[Path | str] = DEFAULT_SOTA_PATHS,
    exp1274_path: Path | str | None = DEFAULT_EXP1274_PATH,
    fover_path: Path | str = DEFAULT_FOVER_PATH,
    output_path: Path | str = DEFAULT_RESULT_PATH,
    run_date: str = RUN_DATE,
    project_root: str = "/home/ianblenke/github.com/ianblenke/carnot",
    build_fraction: float = DEFAULT_BUILD_FRACTION,
) -> dict[str, Any]:
    """Run Exp 1288 and persist the final verifier-feedback replay artifact."""

    write_in_progress_artifact(output_path, run_date=run_date)
    started_at = _utc_now()
    examples, source = load_feedback_examples(
        sota_paths=sota_paths,
        exp1274_path=exp1274_path,
        fover_path=fover_path,
    )
    evaluation = compare_replay_modes(examples, build_fraction=build_fraction)
    finished_at = _utc_now()
    headline_allowed = source != "fover_fallback"
    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "schema": SCHEMA,
        "run_date": run_date,
        "started_at": started_at,
        "finished_at": finished_at,
        "status": "complete",
        "source": source,
        "source_artifacts": {
            "sota_paths": [str(path) for path in sota_paths],
            "exp1274": str(exp1274_path) if exp1274_path is not None else None,
            "fover_corpus": str(fover_path),
        },
        "project_root": project_root,
        "artifact_metadata": {"project_root": project_root, "run_date": run_date},
        "headline_result_allowed": headline_allowed,
    }
    artifact.update(evaluation)
    artifact["honest_verdict"] = derive_honest_verdict(
        float(artifact["self_learning_delta_overall"]),
        headline_allowed=headline_allowed,
    )
    validate_artifact(artifact)
    _write_json(output_path, artifact)
    return artifact


def main() -> int:  # pragma: no cover
    artifact = run_experiment()
    print(
        artifact["honest_verdict"],
        artifact["dvi_acceptance_delta"],
        artifact["memory_update_written"],
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
