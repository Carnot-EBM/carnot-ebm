"""Compare free and answer-span-anchored relation extraction prompts.

The live path reuses the shipped native CUDA server and lease machinery. The
pure functions freeze selection, parsing, reduction, and terminal validation so
the later audit can replay the capture without loading the model.

Spec refs: REQ-VERIFY-7416 and SCENARIO-VERIFY-7416-*.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Iterator

from carnot import experiment_7347_v645_plan_canary as native_runtime
from carnot import experiment_7400_v649_assignment_canary as canary
from carnot import experiment_7410_v650_source_corpus as source_corpus
from carnot import experiment_7412_v650_source_features as source_features
from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.inference.llama_server_supervisor import canonical_json
from carnot.reporting import current_work_receipt
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    REQUIRED_CHECK_NAMES,
)


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260919"
MILESTONE = "2026.09.650"
PHASE = 3
EXPERIMENT_ID = "exp7416-anchored-extraction"
TASK_ID = "experiment_7416_v650_anchored_extraction"
SCHEMA = "carnot.exp7416.v650.anchored_extraction.v1"
MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
MODEL_SPECS = [MODEL_ID]
QUANTIZATION = "Q4_K_M"
RANDOM_SEED = 6_501_601
MAX_GENERATED_TOKENS = 384
MAX_DEVELOPMENT_CALLS = 4
DEVELOPMENT_TOKEN_BUDGET = 64
MODEL_LOAD_TIMEOUT_S = 600.0
CAPTURE_TIMEOUT_S = 2400.0
VALIDATION_TIMEOUT_S = 900.0
REQUEST_TIMEOUT_S = 120.0
LEASE_WAIT_TIMEOUT_S = 180.0

MODULE_PATH = Path("python/carnot/experiment_7416_v650_anchored_extraction.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7416_v650_anchored_extraction.py")
TEST_PATH = Path("tests/python/test_experiment_7416_v650_anchored_extraction.py")
SPEC_PATH = Path("openspec/capabilities/constraint-verification/spec.md")
RESULT_PATH = Path("results/experiment_7416_v650_anchored_extraction.json")
RAW_DIR = Path("results/raw/experiment_7416_v650_anchored_extraction")
UPSTREAM_PATH = Path("results/experiment_7412_v650_source_features.json")
SOURCE_ARTIFACT_PATH = Path("results/experiment_7410_v650_source_corpus.json")
SOURCE_MANIFEST_PATH = Path("results/raw/experiment_7410_v650_source_corpus/corpus_manifest.json")
CHALLENGE_PATH = Path("results/raw/experiment_7412_v650_source_features/challenge_manifest.json")
EXPECTED_UPSTREAM_SHA256 = "sha256:b6ff5100e270646ad98961513956bcdc07073668e034f764d1bd64d20fb17874"
EXPECTED_SOURCE_SHA256 = "sha256:f56c1acc842959972ceb425777cda16ba979f6565aa512dbfd3bb0f6db247ffe"
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "sha256:be0f0b29d6216eaf9a2c1a8ddefd98a5761256c2c9a091038baaf901ab4c277a"
)
EXPECTED_CHALLENGE_SHA256 = (
    "sha256:832167dabdf0b76a0c760906c9a167fbbcd0a76e21db287887cf05657c64fb6a"
)
TERMINAL_CHECK_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_cold_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)

VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)

FIELD_PRINCIPLES: dict[str, str] = {
    "schema": "Use a versioned schema with ordinary top-level identity and terminal fields.",
    "run_date": "Use 20260919 and retain actual UTC boundaries.",
    "preconditions_checked": "Authenticate exact paths, hashes, eligibility, model, runner, and owned capacity before dependent work.",
    "MODEL_SPECS": "Name unsloth/Qwen3.8-27B-GGUF for every current attempted LLM call.",
    "model_invoked": "Set true for actual current attempted model work, including a failed load.",
    "invocation_counts": "Reduce owned current event dispositions without importing historical counts.",
    "inference_substrate": "Use a truthful string and keep device and software facts in a separate detail object.",
    "inference_substrate_class": "Name bounded generation, load-only work, or no-load work from actual current events.",
    "execution_venue": "Use the closed host string and keep hostname and device details elsewhere.",
    "duration_s": "Measure current monotonic task time without padding or historical time.",
    "phase_spans": "Keep real phase boundaries, checkpoints, completed units, and heartbeat records.",
    "random_seed": "Freeze selection, order, sampling, and any resampling seed before outcomes.",
    "reproducibility_checksum": "Bind code, prompts, inputs, events, raw bytes, reductions, and checks.",
    "source_artifact_hashes": "Hash exact source paths and preserve original artifact flags in sidecar metadata.",
    "rows": "Keep every assigned case, arm, condition, and terminal disposition.",
    "sample_size_budget": "Keep planned, attempted, completed, failed, censored, and unstarted counts with the fixed stop rule.",
    "acceptance_gate_results": "Separate completion, evidence, safety, validation, and scientific benefit checks.",
    "gate_check_summary": "Name the exact upstream, path, check, field, operator, expected value, and observed value.",
    "verifier_is_oracle": "Mark true because source-defined span proofs are correctness authority for those endpoints.",
    "honest_verdict": "Use complete_ for finished findings and blocked_ for unchanged missing prerequisites.",
    "verdict_class": "Use only positive, circular_positive, null, blocked, disqualified, or partial.",
    "flagged_adversarial": "Preserve critical findings because flagged science cannot supply readiness.",
    "validation_receipts": "Retain exact argv, environment, names, exits, durations, and hashed logs.",
    "field_principles": "Explain fields separately and keep gate values as ordinary scalars.",
    "promotion_score": "Keep zero because capture cannot change defaults, weights, publication, or rollout.",
    "extraction_capture_complete_score": "Measure accountable raw capture and checks separately from parsing or semantic quality.",
    "raw_capture_manifest": "Bind immutable request, response, timing, cancellation, and byte hashes for both arms.",
    "model_receipt": "Record actual model, runtime, device, lease, ownership, hash, and CUDA offload.",
    "extraction_rows": "Keep all 48 cases by both arms, including failed, censored, and unstarted calls.",
    "semantic_scope": "Keep real annotation alignment, constructed semantic checks, and offset validity distinct.",
}
REQUIRED_FIELDS = frozenset(FIELD_PRINCIPLES)

_CODE_FENCE = re.compile(r"^```", re.MULTILINE)
_WORD = re.compile(r"[\w]+", re.UNICODE)
_QUALIFIER_FAMILIES = {
    "negation",
    "comparator_reversal",
    "time_qualifier",
    "unit_mismatch",
    "count_mismatch",
    "omitted_condition",
    "coreference_ambiguity",
}


def utc_now() -> str:  # pragma: no cover - actual wall-clock boundary.
    """Return an actual UTC boundary while elapsed time remains monotonic."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush each phase, long operation, heartbeat, and checkpoint boundary."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7416] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def sha256_file(path: Path) -> str:
    """Hash exact bytes so a source or raw response cannot drift silently."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def canonical_hash(value: Any) -> str:
    """Hash stable JSON bytes for selection, prompts, rows, and artifacts."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish one complete JSON value only after its bytes reach local storage."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_object(path: Path) -> JsonDict:
    """Return one JSON object, or an empty object for unavailable bytes."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind the artifact without recursively hashing its checksum field."""

    value = deepcopy(dict(artifact))
    value.pop("reproducibility_checksum", None)
    return canonical_hash(value)


def compare(operator: str, observed: Any, expected: Any) -> bool:
    """Apply a declared gate operator without interpreting its prose."""

    if operator == "==":
        return observed == expected
    if operator == "in":
        return observed in expected
    if operator == ">=":
        return observed >= expected
    raise ValueError(f"unsupported_operator:{operator}")


def gate_row(
    check: str,
    upstream: str,
    artifact_field: str,
    operator: str,
    expected: Any,
    observed: Any,
    *,
    principle: str,
    category: str = "precondition",
) -> JsonDict:
    """Keep both operands and the reason for one independently computed gate."""

    try:
        passed = compare(operator, observed, expected)
    except (TypeError, ValueError):
        passed = False
    return {
        "category": category,
        "check": check,
        "upstream": upstream,
        "path": upstream,
        "artifact_field": artifact_field,
        "operator": operator,
        "expected_value": deepcopy(expected),
        "observed_value": deepcopy(observed),
        "passed": passed,
        "principle": principle,
    }


def gate_check_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name the first failure and keep every failed check name."""

    failures = [dict(row) for row in checks if row.get("passed") is not True]
    first = failures[0] if failures else None
    return {
        "all_passed": not failures,
        "failed_count": len(failures),
        "failed_checks": [row.get("check") for row in failures],
        "upstream": first.get("upstream") if first else EXPERIMENT_ID,
        "path": first.get("path") if first else RESULT_PATH.as_posix(),
        "check": first.get("check") if first else "all_required_checks",
        "artifact_field": first.get("artifact_field") if first else "gate_check_summary",
        "operator": first.get("operator") if first else "==",
        "expected_value": deepcopy(first.get("expected_value")) if first else True,
        "observed_value": deepcopy(first.get("observed_value")) if first else True,
        "passed": not failures,
    }


def zero_counts() -> JsonDict:
    """Return an explicit fresh zero-count ledger."""

    return deepcopy(current_work_receipt.ZERO_INVOCATION_COUNTS)


def substrate_class_from_counts(counts: Mapping[str, Any]) -> str:
    """Classify only current attempted loads and generations."""

    if int(counts.get("generation_calls_attempted", 0) or 0) > 0:
        return "model_bounded_generation"
    if int(counts.get("model_loads_attempted", 0) or 0) > 0:
        return "model_load_no_generation"
    return "no_model_load"


def select_official_cases(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Select 24 final-test sentences by seed and row identity only."""

    candidates = [
        deepcopy(dict(row))
        for row in rows
        if row.get("partition") == "final_test"
        and isinstance(row.get("sentence"), str)
        and str(row.get("sentence")).strip()
        and row.get("row_key")
    ]
    candidates.sort(
        key=lambda row: canonical_hash({"seed": RANDOM_SEED, "row_key": row["row_key"]})
    )
    if len(candidates) < 24:
        raise ValueError("fewer_than_24_official_test_sentences")
    return candidates[:24]


def build_cases(
    official_predictors: Sequence[Mapping[str, Any]],
    official_evaluators: Sequence[Mapping[str, Any]],
    challenge_manifest: Mapping[str, Any],
) -> list[JsonDict]:
    """Join authority only after predictor-only selection has finished."""

    if len(official_predictors) != 24:
        raise ValueError("official_case_count")
    official_authority = {str(row.get("row_key")): dict(row) for row in official_evaluators}
    cases: list[JsonDict] = []
    for predictor_value in official_predictors:
        predictor = dict(predictor_value)
        row_key = str(predictor.get("row_key") or "")
        authority = official_authority.get(row_key)
        if authority is None:
            raise ValueError(f"official_authority_missing:{row_key}")
        answer = str(predictor.get("answer") or "")
        sentence = str(predictor.get("sentence") or "")
        offset = answer.find(sentence)
        teacher_span = [offset, offset + len(sentence)] if offset >= 0 else None
        cases.append(
            {
                "case_id": f"official:{row_key}",
                "corpus": "official_test",
                "row_key": row_key,
                "group_id": predictor.get("group_id"),
                "question": str(predictor.get("question") or ""),
                "source_sentence": sentence,
                "answer": answer,
                "pair_id": None,
                "family": None,
                "case_kind": "real_machine_annotation",
                "teacher_answer_span": teacher_span,
                "authority": "machine_annotation_not_truth",
                "machine_label": authority.get("label"),
                "expected_scope": None,
                "expected_verdict": None,
                "source_relation": None,
            }
        )
    predictor_rows = challenge_manifest.get("predictor_records")
    evaluator_rows = challenge_manifest.get("evaluator_records")
    if not isinstance(predictor_rows, list) or not isinstance(evaluator_rows, list):
        raise ValueError("challenge_manifest_shape")
    if len(predictor_rows) != 24 or len(evaluator_rows) != 24:
        raise ValueError("challenge_case_count")
    challenge_authority = {
        str(row.get("case_id")): dict(row) for row in evaluator_rows if isinstance(row, Mapping)
    }
    for predictor_value in predictor_rows:
        if not isinstance(predictor_value, Mapping):
            raise ValueError("challenge_predictor_shape")
        predictor = dict(predictor_value)
        case_id = str(predictor.get("case_id") or "")
        authority = challenge_authority.get(case_id)
        if authority is None:
            raise ValueError(f"challenge_authority_missing:{case_id}")
        cases.append(
            {
                "case_id": case_id,
                "corpus": "constructed_challenge",
                "row_key": None,
                "group_id": None,
                "question": str(predictor.get("question") or ""),
                "source_sentence": str(predictor.get("source") or ""),
                "answer": str(predictor.get("answer") or ""),
                "pair_id": predictor.get("pair_id"),
                "family": predictor.get("family"),
                "case_kind": predictor.get("case_kind"),
                "teacher_answer_span": deepcopy(authority.get("answer_span")),
                "authority": authority.get("authority"),
                "machine_label": None,
                "expected_scope": authority.get("expected_scope"),
                "expected_verdict": authority.get("expected_verdict"),
                "source_relation": deepcopy(authority.get("source_relation")),
            }
        )
    if len(cases) != 48 or len({str(row["case_id"]) for row in cases}) != 48:
        raise ValueError("case_identity")
    return cases


def _prompt(case: Mapping[str, Any], arm: str) -> str:
    """Render one frozen prompt without authority or expected outcomes."""

    shared = (
        "Extract the factual relation claims made in ANSWER. Use QUESTION and "
        "SOURCE SENTENCE only as context. Return one JSON object and no markdown.\n"
        f"QUESTION:\n{case['question']}\n"
        f"SOURCE SENTENCE:\n{case['source_sentence']}\n"
        f"ANSWER:\n{case['answer']}\n"
    )
    if arm == "free":
        return shared + (
            'Schema: {"triples":[{"subject":"text","relation":"text",'
            '"object":"text","qualifiers":["explicit qualifier"]}]}. '
            "Decontextualize pronouns when the text supports it. Keep negation, time, units, "
            "counts, comparisons, and conditions as explicit qualifiers."
        )
    if arm == "anchored":
        return shared + (
            'Schema: {"triples":[{"subject":{"text":"exact ANSWER substring",'
            '"span":[start,end]},"relation":"text","object":{"text":"exact ANSWER '
            'substring","span":[start,end]},"qualifiers":[{"text":"exact ANSWER '
            'substring","span":[start,end]}]}]}. Spans are zero-based half-open character '
            "offsets into ANSWER. Preserve explicit negation, time, units, counts, comparisons, "
            "and conditions."
        )
    raise ValueError(f"unsupported_arm:{arm}")


def build_schedule(cases: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Build two fixed calls per case and alternate order by case hash."""

    schedule: list[JsonDict] = []
    for case_value in cases:
        case = dict(case_value)
        case_id = str(case.get("case_id") or "")
        first = "anchored" if int(canonical_hash(case_id)[-1], 16) % 2 else "free"
        arms = (first, "free" if first == "anchored" else "anchored")
        for arm_order, arm in enumerate(arms):
            call_index = len(schedule)
            prompt = _prompt(case, arm)
            schedule.append(
                {
                    "call_index": call_index,
                    "call_id": f"{case_id}:{arm}",
                    "request_id": f"{case_id}:{arm}",
                    "case_id": case_id,
                    "corpus": case.get("corpus"),
                    "pair_id": case.get("pair_id"),
                    "family": case.get("family"),
                    "case_kind": case.get("case_kind"),
                    "arm": arm,
                    "arm_order": arm_order,
                    "seed": RANDOM_SEED,
                    "max_new_tokens": MAX_GENERATED_TOKENS,
                    "question": case.get("question"),
                    "source_sentence": case.get("source_sentence"),
                    "answer": case.get("answer"),
                    "prompt": prompt,
                    "prompt_sha256": canonical_hash(prompt),
                    "teacher_answer_span": deepcopy(case.get("teacher_answer_span")),
                    "expected_scope": case.get("expected_scope"),
                    "expected_verdict": case.get("expected_verdict"),
                    "source_relation": deepcopy(case.get("source_relation")),
                }
            )
    return schedule


def _span(
    value: Any, answer: str, expected_text: str | None = None
) -> tuple[bool, list[int] | None]:
    """Check one half-open answer span and its optional exact text."""

    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    ):
        return False, None
    start, end = value
    valid = 0 <= start < end <= len(answer)
    if expected_text is not None:
        valid = valid and answer[start:end] == expected_text
    return valid, [start, end] if valid else None


def _free_argument(text: Any, answer: str) -> tuple[bool, list[int] | None]:
    """Locate one exact free-arm argument without changing model text."""

    if not isinstance(text, str) or not text:
        return False, None
    start = answer.find(text)
    return (start >= 0, [start, start + len(text)] if start >= 0 else None)


def parse_extraction(raw_reply: str, *, arm: str, answer: str) -> JsonDict:
    """Parse once and keep syntax, anchoring, and semantics as separate facts."""

    if arm not in {"free", "anchored"}:
        raise ValueError(f"unsupported_arm:{arm}")
    base: JsonDict = {
        "parse_valid": False,
        "parse_error": None,
        "decoded_triples": None,
        "triple_count": 0,
        "coverage_valid": False,
        "argument_anchoring_valid": False,
        "qualifier_anchoring_valid": False,
        "argument_spans": [],
        "qualifier_spans": [],
        "semantic_judgment": "unknown",
        "retry_count": 0,
        "repair_attempted": False,
    }
    if _CODE_FENCE.search(raw_reply):
        base["parse_error"] = "markdown_fence_forbidden"
        return base
    try:
        value = json.loads(raw_reply)
    except json.JSONDecodeError:
        base["parse_error"] = "json_decode_error"
        return base
    if not isinstance(value, dict) or set(value) != {"triples"}:
        base["parse_error"] = "top_level_schema"
        return base
    triples = value.get("triples")
    if not isinstance(triples, list):
        base["parse_error"] = "triples_not_list"
        return base
    argument_valid = True
    qualifier_valid = True
    argument_spans: list[list[int]] = []
    qualifier_spans: list[list[int]] = []
    for triple in triples:
        if not isinstance(triple, dict) or set(triple) != {
            "subject",
            "relation",
            "object",
            "qualifiers",
        }:
            base["parse_error"] = "triple_schema"
            return base
        if not isinstance(triple["relation"], str) or not isinstance(triple["qualifiers"], list):
            base["parse_error"] = "triple_field_type"
            return base
        if arm == "free":
            if not isinstance(triple["subject"], str) or not isinstance(triple["object"], str):
                base["parse_error"] = "free_argument_type"
                return base
            for text in (triple["subject"], triple["object"]):
                valid, span = _free_argument(text, answer)
                argument_valid = argument_valid and valid
                if span is not None:
                    argument_spans.append(span)
            for qualifier in triple["qualifiers"]:
                if not isinstance(qualifier, str):
                    base["parse_error"] = "free_qualifier_type"
                    return base
                valid, span = _free_argument(qualifier, answer)
                qualifier_valid = qualifier_valid and valid
                if span is not None:
                    qualifier_spans.append(span)
        else:
            for argument in (triple["subject"], triple["object"]):
                if not isinstance(argument, dict) or set(argument) != {"text", "span"}:
                    base["parse_error"] = "anchored_argument_schema"
                    return base
                if not isinstance(argument["text"], str):
                    base["parse_error"] = "anchored_argument_text_type"
                    return base
                valid, span = _span(argument["span"], answer, argument["text"])
                argument_valid = argument_valid and valid
                if span is not None:
                    argument_spans.append(span)
            for qualifier in triple["qualifiers"]:
                if not isinstance(qualifier, dict) or set(qualifier) != {"text", "span"}:
                    base["parse_error"] = "anchored_qualifier_schema"
                    return base
                if not isinstance(qualifier["text"], str):
                    base["parse_error"] = "anchored_qualifier_text_type"
                    return base
                valid, span = _span(qualifier["span"], answer, qualifier["text"])
                qualifier_valid = qualifier_valid and valid
                if span is not None:
                    qualifier_spans.append(span)
    base.update(
        {
            "parse_valid": True,
            "decoded_triples": deepcopy(triples),
            "triple_count": len(triples),
            "coverage_valid": bool(triples),
            "argument_anchoring_valid": bool(triples) and argument_valid,
            "qualifier_anchoring_valid": qualifier_valid,
            "argument_spans": argument_spans,
            "qualifier_spans": qualifier_spans,
        }
    )
    return base


def build_raw_capture_row(
    schedule: Mapping[str, Any],
    response: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
) -> JsonDict:
    """Bind exact transport bytes without invoking the extraction parser."""

    raw_request = deepcopy(dict(response.get("raw_request") or {}))
    raw_response = deepcopy(dict(response.get("raw_response") or {}))
    raw_reply = str(response.get("raw_reply") or "")
    request_bytes = canonical_json(raw_request).encode("utf-8")
    response_bytes = canonical_json(raw_response).encode("utf-8")
    terminal_state = str(
        response.get("terminal_state") or ("failed" if response.get("error") else "response")
    )
    attempted = response.get("attempted") is not False
    return {
        **deepcopy(dict(schedule)),
        "raw_request": raw_request,
        "raw_request_sha256": "sha256:" + hashlib.sha256(request_bytes).hexdigest(),
        "raw_request_bytes_b64": base64.b64encode(request_bytes).decode("ascii"),
        "raw_response": raw_response,
        "raw_response_sha256": "sha256:" + hashlib.sha256(response_bytes).hexdigest(),
        "raw_response_bytes_b64": base64.b64encode(response_bytes).decode("ascii"),
        "raw_reply": raw_reply,
        "raw_reply_sha256": canonical_hash(raw_reply),
        "raw_reply_bytes_b64": base64.b64encode(raw_reply.encode("utf-8")).decode("ascii"),
        "attempted": attempted,
        "terminal_state": terminal_state,
        "error": response.get("error"),
        "finish_reason": response.get("finish_reason"),
        "prompt_tokens": int(response.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(response.get("completion_tokens", 0) or 0),
        "latency_s": float(response.get("latency_s", 0.0) or 0.0),
        "runtime_identity_receipt": deepcopy(dict(runtime_identity)),
        "censored": attempted and terminal_state != "response",
        "parse_status": "not_parsed",
        "persisted_before_parse": True,
    }


def parse_capture_row(raw: Mapping[str, Any]) -> JsonDict:
    """Parse one already-persisted row exactly once."""

    row = deepcopy(dict(raw))
    if row.get("terminal_state") != "response":
        parsed = {
            "parse_valid": False,
            "parse_error": "no_terminal_response",
            "decoded_triples": None,
            "triple_count": 0,
            "coverage_valid": False,
            "argument_anchoring_valid": False,
            "qualifier_anchoring_valid": False,
            "argument_spans": [],
            "qualifier_spans": [],
            "semantic_judgment": "unknown",
            "retry_count": 0,
            "repair_attempted": False,
        }
    else:
        parsed = parse_extraction(
            str(row.get("raw_reply") or ""),
            arm=str(row.get("arm") or ""),
            answer=str(row.get("answer") or ""),
        )
    row.update(parsed)
    row["parse_status"] = "valid" if parsed["parse_valid"] else "invalid"
    teacher = row.get("teacher_answer_span")
    spans = list(parsed["argument_spans"])
    overlap: float | None = None
    if (
        isinstance(teacher, list)
        and len(teacher) == 2
        and all(isinstance(item, int) for item in teacher)
        and spans
    ):
        covered = sum(max(0, min(span[1], teacher[1]) - max(span[0], teacher[0])) for span in spans)
        extent = sum(span[1] - span[0] for span in spans)
        overlap = covered / extent if extent else None
    row["teacher_span_overlap"] = overlap
    row["qualifier_retained"] = _qualifier_retention(row)
    row["metric"] = "json_parse_validity"
    row["metric_value"] = int(parsed["parse_valid"])
    row["cost"] = {
        "prompt_tokens": row.get("prompt_tokens", 0),
        "completion_tokens": row.get("completion_tokens", 0),
        "latency_s": row.get("latency_s", 0.0),
        "usd": 0.0,
    }
    return row


def _qualifier_retention(row: Mapping[str, Any]) -> bool | None:
    """Check only visible controlled markers and leave broader meaning unknown."""

    family = row.get("family")
    if family not in _QUALIFIER_FAMILIES or row.get("parse_valid") is not True:
        return None
    answer = str(row.get("answer") or "").lower()
    rendered = json.dumps(row.get("decoded_triples"), sort_keys=True).lower()
    answer_tokens = _WORD.findall(answer)
    if family == "negation":
        required = {token for token in answer_tokens if token in {"not", "no", "never"}}
    elif family == "omitted_condition":
        required = {token for token in answer_tokens if token in {"if", "when", "unless"}}
    elif family == "unit_mismatch":
        required = {
            token
            for token in answer_tokens
            if token in {"kg", "kilogram", "kilograms", "pound", "pounds"}
        }
    elif family == "comparator_reversal":
        required = {token for token in answer_tokens if token in {"longer", "shorter"}}
    elif family in {"time_qualifier", "count_mismatch"}:
        required = {token for token in answer_tokens if token.isdigit()}
    else:
        required = set()
    return all(token in rendered for token in required) if required else None


def _metric(values: Sequence[Any]) -> JsonDict:
    """Reduce a Boolean endpoint with assigned calls as the denominator."""

    known = [value for value in values if isinstance(value, bool)]
    passed = sum(value is True for value in known)
    return {
        "numerator": passed,
        "denominator": len(values),
        "known": len(known),
        "unknown": len(values) - len(known),
        "rate_all_assigned": passed / len(values) if values else None,
        "rate_known": passed / len(known) if known else None,
    }


def reduce_extractions(
    rows: Sequence[Mapping[str, Any]], cases: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Reduce all assigned calls without dropping parse failures or censored rows."""

    values = [deepcopy(dict(row)) for row in rows]
    if len(cases) != 48 or len(values) != 96:
        raise ValueError("fixed_panel_shape")
    pair_ids = sorted(
        {
            str(case.get("pair_id"))
            for case in cases
            if case.get("corpus") == "constructed_challenge" and case.get("pair_id")
        }
    )
    semantic_pairs: list[JsonDict] = []
    for pair_id in pair_ids:
        pair_rows = [row for row in values if row.get("pair_id") == pair_id]
        semantic_pairs.append(
            {
                "pair_id": pair_id,
                "family": pair_rows[0].get("family") if pair_rows else None,
                "assigned_call_count": len(pair_rows),
                "case_count": len({row.get("case_id") for row in pair_rows}),
                "independent_unit_weight": 1,
                "semantic_judgment": "unknown",
                "qualifier_retention": _metric(
                    [row.get("qualifier_retained") for row in pair_rows]
                ),
            }
        )
    latencies = [float(row.get("latency_s", 0.0) or 0.0) for row in values]
    completed = sum(row.get("terminal_state") == "response" for row in values)
    failed = sum(row.get("terminal_state") in {"failed", "request_error"} for row in values)
    censored = sum(bool(row.get("censored")) for row in values)
    attempted = sum(row.get("attempted") is True for row in values)
    unstarted = len(values) - attempted
    teacher_values = [
        row.get("teacher_span_overlap") for row in values if row.get("corpus") == "official_test"
    ]
    teacher_known = [float(value) for value in teacher_values if isinstance(value, (int, float))]
    return {
        "extraction_rows": values,
        "planned_call_count": len(values),
        "assigned_case_count": len(cases),
        "attempted_call_count": attempted,
        "completed_call_count": completed,
        "failed_call_count": failed,
        "censored_call_count": censored,
        "unstarted_call_count": unstarted,
        "independent_constructed_pair_count": len(pair_ids),
        "semantic_pair_rows": semantic_pairs,
        "endpoint_metrics": {
            "json_parse_validity": _metric([row.get("parse_valid") for row in values]),
            "argument_anchoring": _metric([row.get("argument_anchoring_valid") for row in values]),
            "qualifier_retention": _metric([row.get("qualifier_retained") for row in values]),
            "teacher_span_overlap": {
                "denominator": len(teacher_values),
                "known": len(teacher_known),
                "unknown": len(teacher_values) - len(teacher_known),
                "mean_known": (sum(teacher_known) / len(teacher_known) if teacher_known else None),
            },
            "coverage": _metric([row.get("coverage_valid") for row in values]),
            "latency_s": {
                "denominator": len(values),
                "mean": sum(latencies) / len(latencies),
                "maximum": max(latencies, default=0.0),
                "total": sum(latencies),
            },
            "semantic_fidelity": {
                "denominator": len(values),
                "known": 0,
                "unknown": len(values),
                "status": "deferred_to_exp7417",
            },
        },
        "token_and_cost_totals": {
            "prompt_tokens": sum(int(row.get("prompt_tokens", 0) or 0) for row in values),
            "completion_tokens": sum(int(row.get("completion_tokens", 0) or 0) for row in values),
            "local_inference_usd": 0.0,
        },
    }


def _raw_manifest(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Describe immutable transport bytes for every planned call."""

    return [
        {
            "call_index": row.get("call_index"),
            "call_id": row.get("call_id"),
            "case_id": row.get("case_id"),
            "arm": row.get("arm"),
            "attempted": row.get("attempted"),
            "terminal_state": row.get("terminal_state"),
            "latency_s": row.get("latency_s"),
            "error": row.get("error"),
            "raw_request_sha256": row.get("raw_request_sha256"),
            "raw_response_sha256": row.get("raw_response_sha256"),
            "raw_reply_sha256": row.get("raw_reply_sha256"),
            "persisted_before_parse": row.get("persisted_before_parse"),
        }
        for row in rows
    ]


def _base_artifact() -> JsonDict:
    """Return the complete ordinary-field shape shared by all dispositions."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": PHASE,
        "run_date": RUN_DATE,
        "status": "unstarted",
        "started_at_utc": None,
        "completed_at_utc": None,
        "preconditions_checked": [],
        "MODEL_SPECS": deepcopy(MODEL_SPECS),
        "model_invoked": False,
        "invocation_counts": zero_counts(),
        "current_invocation_events": [],
        "current_run_id": f"{EXPERIMENT_ID}:unstarted",
        "current_owner_pid": os.getpid(),
        "event_count": 0,
        "event_sha256": canonical_hash([]),
        "inference_substrate": "no_model_load",
        "inference_substrate_details": {},
        "inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "duration_s": 0.0,
        "scientific_duration_s": 0.0,
        "validation_duration_s": 0.0,
        "phase_spans": [],
        "random_seed": {
            "selection": RANDOM_SEED,
            "arm_order": RANDOM_SEED,
            "sampling": RANDOM_SEED,
            "resampling": RANDOM_SEED,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "planned": 96,
            "attempted": 0,
            "completed": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": 96,
            "case_count": 48,
            "official_case_count": 24,
            "constructed_case_count": 24,
            "independent_constructed_pair_count": 8,
            "measured_calls_per_case": 2,
            "maximum_new_tokens_per_call": MAX_GENERATED_TOKENS,
            "development_call_limit": MAX_DEVELOPMENT_CALLS,
            "development_token_limit": DEVELOPMENT_TOKEN_BUDGET,
            "development_calls_used": 0,
            "capture_timeout_s": CAPTURE_TIMEOUT_S,
            "validation_timeout_s": VALIDATION_TIMEOUT_S,
            "lease_wait_timeout_s": LEASE_WAIT_TIMEOUT_S,
            "stop_rule": "stop_after_96_dispositions_or_capture_deadline_without_replacement",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": {},
        "verifier_is_oracle": True,
        "honest_verdict": "blocked_unstarted",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "validation_receipts": [],
        "repository_health": {
            "status": "not_checked",
            "unrelated_broad_suite_observations": [],
            "affects_required_checks": False,
        },
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "promotion_score": 0,
        "extraction_capture_complete_score": 0,
        "raw_capture_manifest": [],
        "model_receipt": {},
        "extraction_rows": [],
        "semantic_scope": {
            "real_annotation_alignment": "machine_annotation_not_truth",
            "constructed_semantic_checks": "source_defined_but_provisional_until_exp7417",
            "offset_validity": "exact_half_open_offsets_into_answer",
            "full_system_correctness_claimed": False,
            "repair_claimed": False,
        },
        "semantic_pair_rows": [],
        "endpoint_metrics": {},
        "token_and_cost_totals": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "local_inference_usd": 0.0,
        },
        "historical_receipt_sidecars": [],
    }


def build_blocked_artifact(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Publish unchanged external absence with no invented model work."""

    artifact = _base_artifact()
    summary = gate_check_summary(checks)
    artifact.update(
        {
            "status": "blocked_precondition",
            "preconditions_checked": [deepcopy(dict(row)) for row in checks],
            "gate_check_summary": summary,
            "acceptance_gate_results": [],
            "honest_verdict": f"blocked_{summary['check']}",
            "verdict_class": "blocked",
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _fixture_cases() -> list[JsonDict]:
    """Create deterministic public and constructed cases for cold unit tests."""

    predictors: list[JsonDict] = []
    evaluators: list[JsonDict] = []
    for index in range(24):
        sentence = f"Person {index} founded Place {index}."
        predictors.append(
            {
                "row_key": f"fixture-{index:02d}",
                "group_id": f"fixture-group-{index:02d}",
                "partition": "final_test",
                "question": "Who founded the place?",
                "answer": sentence,
                "sentence": sentence,
            }
        )
        evaluators.append({"row_key": f"fixture-{index:02d}", "label": index % 2})
    challenge_predictors: list[JsonDict] = []
    challenge_evaluators: list[JsonDict] = []
    families = (
        "negation",
        "subject_object_reversal",
        "comparator_reversal",
        "time_qualifier",
        "unit_mismatch",
        "count_mismatch",
        "omitted_condition",
        "coreference_ambiguity",
    )
    for pair_index, family in enumerate(families, start=1):
        pair_id = f"fixture-pair-{pair_index}"
        for member in ("base", "contrast", "control"):
            answer = f"Agent {pair_index} {member} Value {pair_index}."
            case_id = f"{pair_id}-{member}"
            challenge_predictors.append(
                {
                    "case_id": case_id,
                    "pair_id": pair_id,
                    "case_kind": member,
                    "family": family,
                    "question": "Extract the relation.",
                    "source": answer,
                    "answer": answer,
                }
            )
            challenge_evaluators.append(
                {
                    "case_id": case_id,
                    "answer_span": [0, len(answer)],
                    "authority": "constructed_source_defined_fixture",
                    "expected_scope": member,
                    "expected_verdict": "supported",
                    "source_relation": ["Agent", "value", str(pair_index)],
                }
            )
    return build_cases(
        predictors,
        evaluators,
        {
            "predictor_records": challenge_predictors,
            "evaluator_records": challenge_evaluators,
        },
    )


def build_artifact_for_test() -> JsonDict:
    """Build a deterministic complete artifact through production reducers."""

    cases = _fixture_cases()
    schedule = build_schedule(cases)
    rows: list[JsonDict] = []
    for call in schedule:
        answer = str(call["answer"])
        subject = answer.split()[0]
        object_text = answer.rstrip(".").split()[-1]
        if call["arm"] == "free":
            payload: JsonDict = {
                "triples": [
                    {
                        "subject": subject,
                        "relation": "mentions",
                        "object": object_text,
                        "qualifiers": [],
                    }
                ]
            }
        else:
            object_start = answer.rfind(object_text)
            payload = {
                "triples": [
                    {
                        "subject": {"text": subject, "span": [0, len(subject)]},
                        "relation": "mentions",
                        "object": {
                            "text": object_text,
                            "span": [object_start, object_start + len(object_text)],
                        },
                        "qualifiers": [],
                    }
                ]
            }
        reply = json.dumps(payload, sort_keys=True)
        response = {
            "raw_request": {"messages": [{"role": "user", "content": call["prompt"]}]},
            "raw_response": {"choices": [{"message": {"content": reply}}]},
            "raw_reply": reply,
            "attempted": True,
            "terminal_state": "response",
            "finish_reason": "stop",
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "latency_s": 0.2,
        }
        rows.append(parse_capture_row(build_raw_capture_row(call, response, {})))
    reduced = reduce_extractions(rows, cases)
    counts = zero_counts()
    counts.update(
        {
            "model_loads_attempted": 1,
            "model_loads_completed": 1,
            "generation_calls_attempted": 96,
            "generation_calls_completed": 96,
        }
    )
    receipts = [
        {"name": name, "passed": True, "exit_code": 0, "timed_out": False}
        for name in (*REQUIRED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)
    ]
    gates = [
        gate_row(
            "capture_accounted",
            EXPERIMENT_ID,
            "planned_call_count",
            "==",
            96,
            reduced["planned_call_count"],
            principle="All calls stay in the denominator.",
            category="completion",
        )
    ]
    artifact = _base_artifact()
    artifact.update(
        {
            "status": "complete_capture",
            "started_at_utc": "2026-09-19T00:00:00Z",
            "completed_at_utc": "2026-09-19T00:00:12Z",
            "MODEL_SPECS": deepcopy(MODEL_SPECS),
            "model_invoked": True,
            "invocation_counts": counts,
            "inference_substrate": "owned_native_cuda_llama_cpp_bounded_generation",
            "inference_substrate_details": {"gpu_name": "NVIDIA GeForce RTX 3090"},
            "inference_substrate_class": "model_bounded_generation",
            "duration_s": 12.0,
            "scientific_duration_s": 10.0,
            "validation_duration_s": 2.0,
            "source_artifact_hashes": {"fixture": {"sha256": canonical_hash("fixture")}},
            "rows": deepcopy(rows),
            "sample_size_budget": {
                **artifact["sample_size_budget"],
                "attempted": 96,
                "completed": 96,
                "failed": 0,
                "censored": 0,
                "unstarted": 0,
            },
            "acceptance_gate_results": gates,
            "gate_check_summary": gate_check_summary(gates),
            "honest_verdict": "complete_null_capture_ready_semantics_pending_exp7417",
            "verdict_class": "null",
            "validation_receipts": receipts,
            "repository_health": {
                "status": "healthy",
                "unrelated_broad_suite_observations": [],
                "affects_required_checks": False,
            },
            "extraction_capture_complete_score": 1,
            "raw_capture_manifest": _raw_manifest(rows),
            "model_receipt": {
                "owned_by_task": True,
                "all_layers_offloaded": True,
                "actual_offloaded_layers": 65,
                "lease_released": True,
            },
            **reduced,
        }
    )
    artifact["rows"] = deepcopy(artifact["extraction_rows"])
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _hash_matches(row: Mapping[str, Any], index: int) -> list[str]:
    """Recompute one row's transport hashes without trusting its parser fields."""

    errors: list[str] = []
    request = canonical_json(row.get("raw_request") or {}).encode("utf-8")
    response = canonical_json(row.get("raw_response") or {}).encode("utf-8")
    reply = str(row.get("raw_reply") or "")
    expected = {
        "raw_request_sha256": "sha256:" + hashlib.sha256(request).hexdigest(),
        "raw_response_sha256": "sha256:" + hashlib.sha256(response).hexdigest(),
        "raw_reply_sha256": canonical_hash(reply),
    }
    for field, value in expected.items():
        if row.get(field) != value:
            errors.append(f"{field.removesuffix('_sha256')}_hash_mismatch:{index}")
    return errors


def validate_artifact(
    value: Mapping[str, Any], *, require_terminal: bool = False, root: Path = REPO_ROOT
) -> list[str]:
    """Cold-check identity, rows, hashes, counts, gates, and completion claims."""

    del root
    errors = [f"missing_field:{field}" for field in REQUIRED_FIELDS if field not in value]
    expected = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "execution_venue": "host",
        "promotion_score": 0,
        "verifier_is_oracle": True,
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            errors.append(f"declaration_mismatch:{field}")
    if value.get("MODEL_SPECS") != MODEL_SPECS:
        errors.append("model_specs_invalid")
    if value.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if set(value.get("field_principles") or {}) != REQUIRED_FIELDS:
        errors.append("field_principles_mismatch")
    counts = value.get("invocation_counts") or {}
    if value.get("inference_substrate_class") != substrate_class_from_counts(counts):
        errors.append("substrate_class_mismatch")
    invoked = (
        int(counts.get("model_loads_attempted", 0) or 0)
        + int(counts.get("generation_calls_attempted", 0) or 0)
        > 0
    )
    if value.get("model_invoked") is not invoked:
        errors.append("model_invoked_mismatch")
    rows = value.get("extraction_rows") or []
    if value.get("rows") != rows:
        errors.append("rows_alias_mismatch")
    if value.get("verdict_class") == "blocked":
        if rows or value.get("extraction_capture_complete_score") != 0:
            errors.append("blocked_rows_or_score_invalid")
        if not str(value.get("honest_verdict") or "").startswith("blocked_"):
            errors.append("blocked_verdict_prefix_invalid")
    else:
        if not isinstance(rows, list) or len(rows) != 96:
            errors.append("extraction_row_count_mismatch")
        else:
            if len({row.get("call_id") for row in rows if isinstance(row, Mapping)}) != 96:
                errors.append("call_identity_mismatch")
            case_ids = {row.get("case_id") for row in rows if isinstance(row, Mapping)}
            if len(case_ids) != 48:
                errors.append("case_identity_mismatch")
            for index, row in enumerate(rows):
                if not isinstance(row, Mapping):
                    errors.append(f"row_shape:{index}")
                    continue
                errors.extend(_hash_matches(row, index))
                if row.get("persisted_before_parse") is not True:
                    errors.append(f"raw_not_persisted_before_parse:{index}")
        safe_rows = [row if isinstance(row, Mapping) else {} for row in rows]
        budget = value.get("sample_size_budget") or {}
        expected_budget = {
            "planned": 96,
            "attempted": sum(row.get("attempted") is True for row in safe_rows),
            "completed": sum(row.get("terminal_state") == "response" for row in safe_rows),
            "failed": sum(
                row.get("terminal_state") in {"failed", "request_error"} for row in safe_rows
            ),
            "censored": sum(bool(row.get("censored")) for row in safe_rows),
            "unstarted": sum(row.get("attempted") is not True for row in safe_rows),
        }
        if any(budget.get(field) != observed for field, observed in expected_budget.items()):
            errors.append("sample_size_budget_mismatch")
        if value.get("independent_constructed_pair_count") != 8:
            errors.append("constructed_pair_count_mismatch")
        if len(value.get("semantic_pair_rows") or []) != 8:
            errors.append("semantic_pair_rows_mismatch")
        manifest = value.get("raw_capture_manifest") or []
        if len(manifest) != len(rows):
            errors.append("raw_capture_manifest_count_mismatch")
    receipts = value.get("validation_receipts") or []
    if require_terminal and value.get("verdict_class") != "blocked":
        names = {
            row.get("name")
            for row in receipts
            if row.get("passed") is True
            and row.get("exit_code") == 0
            and row.get("timed_out") is not True
        }
        missing = set((*REQUIRED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)) - names
        if missing:
            errors.append("required_validation_receipts_missing:" + ",".join(sorted(missing)))
        expected_complete = int(
            not missing
            and len(rows) == 96
            and value.get("flagged_adversarial") is False
            and dict(value.get("model_receipt") or {}).get("all_layers_offloaded") is True
        )
        if value.get("extraction_capture_complete_score") != expected_complete:
            errors.append("extraction_capture_complete_score_mismatch")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def independent_reduce_artifact(
    artifact: Mapping[str, Any], *, root: Path = REPO_ROOT
) -> list[str]:
    """Reparse raw replies and compare every stored reduction field."""

    del root
    errors: list[str] = []
    rows = artifact.get("extraction_rows") or []
    if artifact.get("verdict_class") == "blocked" and rows == []:
        return validate_artifact(artifact, require_terminal=False)
    if not isinstance(rows, list) or len(rows) != 96:
        return ["independent_row_count"]
    for index, row_value in enumerate(rows):
        if not isinstance(row_value, Mapping):
            errors.append(f"independent_row_shape:{index}")
            continue
        row = dict(row_value)
        reparsed = parse_capture_row(
            {
                key: deepcopy(value)
                for key, value in row.items()
                if key
                not in {
                    "parse_valid",
                    "parse_error",
                    "decoded_triples",
                    "triple_count",
                    "coverage_valid",
                    "argument_anchoring_valid",
                    "qualifier_anchoring_valid",
                    "argument_spans",
                    "qualifier_spans",
                    "semantic_judgment",
                    "retry_count",
                    "repair_attempted",
                    "teacher_span_overlap",
                    "qualifier_retained",
                    "metric",
                    "metric_value",
                    "cost",
                }
            }
        )
        fields = (
            "parse_valid",
            "parse_error",
            "decoded_triples",
            "triple_count",
            "coverage_valid",
            "argument_anchoring_valid",
            "qualifier_anchoring_valid",
            "argument_spans",
            "qualifier_spans",
            "semantic_judgment",
            "retry_count",
            "repair_attempted",
            "teacher_span_overlap",
            "qualifier_retained",
        )
        if any(row.get(field) != reparsed.get(field) for field in fields):
            errors.append(f"independent_parse_mismatch:{index}")
    errors.extend(validate_artifact(artifact, require_terminal=False))
    return list(dict.fromkeys(errors))


def collect_preconditions(root: Path) -> tuple[list[JsonDict], JsonDict]:  # pragma: no cover
    """Authenticate exact sources and upstream eligibility before model work."""

    required = (
        Path("AGENTS.md"),
        Path("CLAUDE.md"),
        Path("CODEX.md"),
        Path("research-program.md"),
        Path("ops/exclusion_manifest.yaml"),
        Path("ops/e2e-test-plan.md"),
        Path("scripts/experiment_template.py"),
        Path("python/carnot/reporting/current_work_receipt.py"),
        Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
        Path("python/carnot/experiment_7358_v646_validation_contract.py"),
        Path("python/carnot/inference/sota_models.py"),
        Path("python/carnot/experiment_7400_v649_assignment_canary.py"),
        Path("python/carnot/experiment_7402_v649_proposal_capture.py"),
        SPEC_PATH,
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
        UPSTREAM_PATH,
        SOURCE_ARTIFACT_PATH,
        SOURCE_MANIFEST_PATH,
        CHALLENGE_PATH,
    )
    checks: list[JsonDict] = []
    for relative in required:
        path = root / relative
        observed = "readable_nonempty_bytes" if path.is_file() and path.stat().st_size else None
        checks.append(
            gate_row(
                f"source_bytes:{relative.as_posix()}",
                relative.as_posix(),
                "bytes",
                "==",
                "readable_nonempty_bytes",
                observed,
                principle="Required bytes must exist before dependent work.",
            )
        )
    spec = (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    checks.append(
        gate_row(
            "driving_requirement",
            SPEC_PATH.as_posix(),
            "REQ-*",
            "==",
            "REQ-VERIFY-7416",
            "REQ-VERIFY-7416" if "REQ-VERIFY-7416" in spec else None,
            principle="Implementation starts only after the capability requirement exists.",
        )
    )
    upstream = load_object(root / UPSTREAM_PATH)
    upstream_hash = sha256_file(root / UPSTREAM_PATH) if (root / UPSTREAM_PATH).is_file() else None
    upstream_fields = (
        ("upstream_identity", "experiment_id", "==", "exp7412-source-features"),
        ("source_feature_protocol_ready", "source_feature_protocol_ready_score", "==", 1),
        (
            "upstream_verdict_eligible",
            "verdict_class",
            "in",
            ["positive", "circular_positive", "null"],
        ),
        ("upstream_unflagged", "flagged_adversarial", "==", False),
    )
    for check, field, operator, expected in upstream_fields:
        checks.append(
            gate_row(
                check,
                UPSTREAM_PATH.as_posix(),
                field,
                operator,
                expected,
                upstream.get(field),
                principle="Only exact eligible Exp7412 evidence can authorize capture.",
            )
        )
    checks.extend(
        [
            gate_row(
                "upstream_artifact_hash",
                UPSTREAM_PATH.as_posix(),
                "sha256",
                "==",
                EXPECTED_UPSTREAM_SHA256,
                upstream_hash,
                principle="A changed producer cannot silently authorize this run.",
            ),
            gate_row(
                "source_artifact_hash",
                SOURCE_ARTIFACT_PATH.as_posix(),
                "sha256",
                "==",
                EXPECTED_SOURCE_SHA256,
                sha256_file(root / SOURCE_ARTIFACT_PATH)
                if (root / SOURCE_ARTIFACT_PATH).is_file()
                else None,
                principle="The real corpus identity must remain exact.",
            ),
            gate_row(
                "source_manifest_hash",
                SOURCE_MANIFEST_PATH.as_posix(),
                "sha256",
                "==",
                EXPECTED_SOURCE_MANIFEST_SHA256,
                sha256_file(root / SOURCE_MANIFEST_PATH)
                if (root / SOURCE_MANIFEST_PATH).is_file()
                else None,
                principle="Selection must use the authenticated predictor manifest.",
            ),
            gate_row(
                "challenge_manifest_hash",
                CHALLENGE_PATH.as_posix(),
                "sha256",
                "==",
                EXPECTED_CHALLENGE_SHA256,
                sha256_file(root / CHALLENGE_PATH) if (root / CHALLENGE_PATH).is_file() else None,
                principle="Constructed pairs must remain sealed before generation.",
            ),
        ]
    )
    cold_errors = source_features.validate_artifact(upstream) if upstream else ["missing"]
    checks.append(
        gate_row(
            "upstream_cold_validation",
            UPSTREAM_PATH.as_posix(),
            "validate_artifact.errors",
            "==",
            [],
            cold_errors,
            principle="The producer must replay before its score is trusted.",
        )
    )
    exclusion = (
        (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
        if (root / "ops/exclusion_manifest.yaml").is_file()
        else ""
    )
    excluded = "experiment_id: 7416" in exclusion or EXPERIMENT_ID in exclusion
    checks.append(
        gate_row(
            "current_task_not_quarantined",
            "ops/exclusion_manifest.yaml",
            EXPERIMENT_ID,
            "==",
            False,
            excluded,
            principle="A retired unchanged capture cannot run automatically.",
        )
    )
    context: JsonDict = {"upstream": upstream, "upstream_sha256": upstream_hash}
    try:
        reloaded = source_corpus.reload_corpus(root / SOURCE_MANIFEST_PATH.parent)
        readers = source_corpus.CorpusReaders(reloaded)
        selected = select_official_cases(readers.read_predictors("final_test"))
        official_labels = readers.read_labels("final_test", token=source_corpus.FINAL_TEST_TOKEN)
        challenge = source_features.load_challenge_manifest(root / CHALLENGE_PATH)
        cases = build_cases(selected, official_labels, challenge)
        schedule = build_schedule(cases)
        selection_error = None
    except (OSError, PermissionError, TypeError, ValueError) as exc:
        selected, official_labels, challenge, cases, schedule = [], [], {}, [], []
        selection_error = f"{type(exc).__name__}:{exc}"
    checks.append(
        gate_row(
            "frozen_extraction_panel",
            "authenticated_predictor_views",
            "sample_size_budget",
            "==",
            {"official": 24, "challenge": 24, "calls": 96, "error": None},
            {
                "official": len(selected),
                "challenge": len(cases) - len(selected),
                "calls": len(schedule),
                "error": selection_error,
            },
            principle="The fixed panel cannot replace hard cases after outcomes exist.",
        )
    )
    context.update(
        {
            "selected_official": selected,
            "official_labels": official_labels,
            "challenge": challenge,
            "cases": cases,
            "schedule": schedule,
        }
    )
    return checks, context


def _runtime_preconditions(
    root: Path, context: JsonDict, started: float
) -> list[JsonDict]:  # pragma: no cover - host and GPU dependent.
    """Reuse the qualified Qwen preflight and require one available RTX 3090."""

    checks = [
        row
        for row in canary._runtime_preconditions(root, context, started)
        if row.get("check") != "one_owned_rtx3090_slot"
    ]
    available = list(context.get("available_gpu_uuids") or [])
    checks.append(
        gate_row(
            "bounded_lease_wait",
            "runtime_policy",
            "lease_wait_timeout_s",
            "==",
            LEASE_WAIT_TIMEOUT_S,
            LEASE_WAIT_TIMEOUT_S,
            principle="Contention cannot consume an unbounded task window.",
        )
    )
    checks.append(
        gate_row(
            "one_owned_rtx3090_slot",
            "nvidia-smi_and_gpu_lease_journal",
            "available_rtx3090_slots",
            ">=",
            1,
            len(available),
            principle="Headline capture requires one task-owned RTX 3090 and no CPU fallback.",
        )
    )
    model_spec = dict(context.get("model_spec") or {})
    decoding = dict(model_spec.get("decoding") or {})
    decoding.update(
        {
            "max_new_tokens": MAX_GENERATED_TOKENS,
            "seed": RANDOM_SEED,
            "repair_attempts": 0,
            "retry_budget": 0,
        }
    )
    model_spec["decoding"] = decoding
    context["model_spec"] = model_spec
    return checks


def _runtime_raw_row(  # pragma: no cover - called by the live native loop.
    *,
    call_index: int,
    request: Mapping[str, Any],
    prompt: str,
    response: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
) -> JsonDict:
    """Adapt the native transport to raw-first persistence without parsing."""

    schedule = deepcopy(dict(request))
    schedule["call_index"] = call_index
    schedule["prompt"] = prompt
    return build_raw_capture_row(schedule, response, runtime_identity)


@contextmanager
def _shared_runtime_settings(
    recorder: canary.InvocationEventRecorder,
) -> Iterator[None]:  # pragma: no cover - live globals are restored in finally.
    """Apply fixed settings to the shipped owned native runner for one capture."""

    values = {
        "TASK_ID": TASK_ID,
        "RUN_DATE": RUN_DATE,
        "MAX_GENERATED_TOKENS": MAX_GENERATED_TOKENS,
        "MODEL_LOAD_TIMEOUT_S": MODEL_LOAD_TIMEOUT_S,
        "INFERENCE_WINDOW_TIMEOUT_S": CAPTURE_TIMEOUT_S,
        "REQUEST_TIMEOUT_S": REQUEST_TIMEOUT_S,
        "RANDOM_SEED": {
            "development": RANDOM_SEED,
            "evaluation": RANDOM_SEED,
            "resampling": RANDOM_SEED,
        },
        "render_public_prompt": lambda request: str(request["prompt"]),
        "build_call_row": _runtime_raw_row,
        "progress": recorder,
    }
    previous = {name: getattr(native_runtime, name) for name in values}
    try:
        for name, value in values.items():
            setattr(native_runtime, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(native_runtime, name, value)


def _capture_current(
    context: Mapping[str, Any], raw_dir: Path, started: float
) -> JsonDict:  # pragma: no cover - live model work.
    """Run 96 calls through one task-owned server and retain owned events."""

    recorder = canary.InvocationEventRecorder(
        f"{EXPERIMENT_ID}:{os.getpid()}:{time.monotonic_ns()}", os.getpid(), started
    )
    runtime_context = deepcopy(dict(context))
    runtime_context["selected_requests"] = deepcopy(list(context["schedule"]))
    with _shared_runtime_settings(recorder):
        capture = native_runtime._live_capture(runtime_context, raw_dir / "native")
    recorder.close(capture)
    capture["current_run_id"] = recorder.run_id
    capture["current_owner_pid"] = recorder.owner_pid
    capture["current_invocation_events"] = recorder.events
    return capture


def _span_receipt(
    spans: list[JsonDict],
    phase: str,
    phase_started: float,
    run_started: float,
    completed_units: int,
) -> None:  # pragma: no cover - actual timing evidence.
    """Close one real phase with a monotonic checkpoint."""

    ended = time.monotonic()
    spans.append(
        {
            "phase": phase,
            "start_s": round(phase_started - run_started, 6),
            "end_s": round(ended - run_started, 6),
            "duration_s": round(ended - phase_started, 6),
            "completed_units": completed_units,
            "heartbeat_times_s": [],
            "checkpoint_times_s": [round(ended - run_started, 6)],
            "checkpoint_at_utc": utc_now(),
        }
    )


def _run_affected_validation(
    root: Path, raw_dir: Path, started: float
) -> tuple[list[JsonDict], JsonDict]:  # pragma: no cover - subprocess orchestration.
    """Run the frozen Exp7358 plan through the streaming Exp7303 runner."""

    private = Path(tempfile.mkdtemp(prefix="exp7416-validation-", dir="/tmp"))
    commands = build_command_plan(root, VALIDATION_MANIFEST, private)
    plan_errors = validate_command_plan(root, VALIDATION_MANIFEST, commands)
    if plan_errors:
        return [], {"passed": False, "plan_errors": plan_errors}
    planned = [
        PlannedCommand(spec=command, category="required_affected_validation", required=True)
        for command in commands
    ]
    progress(started, "validation", "before_affected_commands", count=len(planned))
    receipts = run_categorized_commands(
        root, planned, log_dir=raw_dir / "validation/affected", heartbeat_s=60.0
    )
    progress(started, "validation", "after_affected_commands", count=len(receipts))
    reduced = reduce_affected_receipts(root, VALIDATION_MANIFEST, receipts)
    return receipts, {**reduced, "plan_errors": []}


def _terminal_commands(root: Path, candidate: Path) -> list[CommandSpec]:  # pragma: no cover
    """Build fresh replay, independent reduction, and unchanged strict readers."""

    python = str(root / ".venv/bin/python")
    reducer = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7416_v650_anchored_extraction import independent_reduce_artifact;"
        "value=json.loads(pathlib.Path(sys.argv[1]).read_text());"
        "errors=independent_reduce_artifact(value);"
        "print(json.dumps({'errors':errors},sort_keys=True),flush=True);"
        "raise SystemExit(bool(errors))"
    )
    return [
        CommandSpec(
            "declared_entrypoint_cold_replay",
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                "--date",
                RUN_DATE,
                "--validate",
                str(candidate),
            ),
            "measured_candidate",
        ),
        CommandSpec(
            "independent_cold_reducer",
            (python, "-u", "-c", reducer, str(candidate)),
            "measured_candidate",
        ),
        CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "measured_candidate",
        ),
        CommandSpec(
            "verdict_row_consistency_strict",
            (
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "measured_candidate",
        ),
    ]


def _receipts_pass(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    """Require one successful terminal receipt for every declared name."""

    by_name = {str(row.get("name")): row for row in receipts}
    return set(by_name) == set(names) and all(
        by_name[name].get("passed") is True
        and by_name[name].get("exit_code") == 0
        and by_name[name].get("timed_out") is not True
        for name in names
    )


def _source_hashes(
    root: Path,
    context: Mapping[str, Any],
    raw_files: Sequence[Mapping[str, Any]],
    sidecar: Mapping[str, Any],
) -> JsonDict:  # pragma: no cover - measured source evidence.
    """Bind current sources, upstream bytes, model, runner, sidecar, and rows."""

    paths = (
        Path("AGENTS.md"),
        Path("CLAUDE.md"),
        Path("CODEX.md"),
        Path("research-program.md"),
        Path("ops/exclusion_manifest.yaml"),
        Path("ops/e2e-test-plan.md"),
        SPEC_PATH,
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
        UPSTREAM_PATH,
        SOURCE_ARTIFACT_PATH,
        SOURCE_MANIFEST_PATH,
        CHALLENGE_PATH,
        Path("python/carnot/reporting/current_work_receipt.py"),
        Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
        Path("python/carnot/experiment_7358_v646_validation_contract.py"),
        Path("python/carnot/experiment_7400_v649_assignment_canary.py"),
        Path("python/carnot/experiment_7402_v649_proposal_capture.py"),
        Path("python/carnot/inference/sota_models.py"),
    )
    hashes: JsonDict = {}
    for relative in paths:
        path = root / relative
        if path.is_file():
            original = None
            if relative == UPSTREAM_PATH:
                original = dict(context.get("upstream") or {}).get("flagged_adversarial")
            hashes[relative.as_posix()] = {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "original_flagged_adversarial": original,
            }
    model_path = Path(str(dict(context.get("model_spec") or {}).get("path") or ""))
    if model_path.is_file():
        hashes["cached_model"] = {
            "path": str(model_path),
            "sha256": dict(context["model_spec"]).get("sha256"),
            "original_flagged_adversarial": None,
        }
    hashes["raw_calls"] = [deepcopy(dict(row)) for row in raw_files]
    hashes["historical_sidecar"] = deepcopy(dict(sidecar))
    return hashes


def _date_argument(value: str) -> str:
    """Reject execution outside the fixed V650 date."""

    if value != RUN_DATE:
        raise ValueError(f"date must be {RUN_DATE}")
    return value


def run_experiment(
    *, root: Path = REPO_ROOT, run_date: str = RUN_DATE, output_path: Path | None = None
) -> JsonDict:  # pragma: no cover - bounded live orchestration.
    """Authenticate, capture, validate, cold replay, and publish atomically."""

    started = time.monotonic()
    started_at = utc_now()
    output = output_path or root / RESULT_PATH
    raw_dir = root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    spans: list[JsonDict] = []

    phase_started = time.monotonic()
    progress(started, "preconditions", "start")
    checks, context = collect_preconditions(root)
    checks.insert(
        0,
        gate_row(
            "run_date",
            "execution_contract",
            "run_date",
            "==",
            RUN_DATE,
            run_date,
            principle="The fixed protocol uses one declared execution date.",
        ),
    )
    _span_receipt(spans, "preconditions_static", phase_started, started, len(checks))
    if not all(row["passed"] for row in checks):
        artifact = build_blocked_artifact(checks)
        artifact.update(
            {
                "started_at_utc": started_at,
                "completed_at_utc": utc_now(),
                "duration_s": round(time.monotonic() - started, 6),
                "phase_spans": spans,
            }
        )
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
        atomic_json(output, artifact)
        return artifact

    phase_started = time.monotonic()
    progress(started, "preconditions", "runtime_start")
    runtime_checks = _runtime_preconditions(root, context, started)
    checks.extend(runtime_checks)
    progress(
        started, "preconditions", "runtime_complete", passed=all(row["passed"] for row in checks)
    )
    _span_receipt(spans, "preconditions_runtime", phase_started, started, len(runtime_checks))
    if not all(row["passed"] for row in checks):
        artifact = build_blocked_artifact(checks)
        artifact.update(
            {
                "started_at_utc": started_at,
                "completed_at_utc": utc_now(),
                "duration_s": round(time.monotonic() - started, 6),
                "phase_spans": spans,
                "inference_substrate_details": deepcopy(context.get("model_spec") or {}),
            }
        )
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
        atomic_json(output, artifact)
        return artifact

    sidecar = current_work_receipt.write_immutable_sidecar(
        raw_dir / "historical_exp7412_source_features.json",
        scope="historical_model_receipts",
        payload={
            "artifact_path": UPSTREAM_PATH.as_posix(),
            "artifact_sha256": context["upstream_sha256"],
            "original_verdict_class": context["upstream"].get("verdict_class"),
            "original_flagged_adversarial": context["upstream"].get("flagged_adversarial"),
            "source_feature_protocol_ready_score": context["upstream"].get(
                "source_feature_protocol_ready_score"
            ),
            "authorizes_current_inference_counts": False,
        },
        root=root,
    )

    phase_started = time.monotonic()
    scientific_started = phase_started
    progress(started, "generation", "before_model_load_and_generation", planned_calls=96)
    capture = _capture_current(context, raw_dir / "owned_runtime", started)
    progress(
        started,
        "generation",
        "after_model_load_and_generation",
        returned_calls=len(capture.get("rows") or []),
    )
    _span_receipt(spans, "model_load_and_generation", phase_started, started, len(capture["rows"]))
    scientific_duration = time.monotonic() - scientific_started

    phase_started = time.monotonic()
    raw_files: list[JsonDict] = []
    parsed_rows: list[JsonDict] = []
    runtime_rows = list(capture.get("rows") or [])
    for index, schedule in enumerate(context["schedule"]):
        source = dict(runtime_rows[index]) if index < len(runtime_rows) else {}
        if source.get("raw_request_sha256") is None:
            source = build_raw_capture_row(schedule, source, capture.get("runtime_identity") or {})
        raw_path = raw_dir / "calls" / f"call_{index:03d}.json"
        atomic_json(raw_path, source)
        raw_files.append(
            {"path": raw_path.relative_to(root).as_posix(), "sha256": sha256_file(raw_path)}
        )
        parsed_rows.append(parse_capture_row(source))
        progress(
            started,
            "reduction",
            "checkpoint",
            completed=index + 1,
            total=96,
            terminal_state=parsed_rows[-1]["terminal_state"],
        )
    reduced = reduce_extractions(parsed_rows, context["cases"])
    current = canary.reduce_current_events(
        capture["current_invocation_events"],
        run_id=capture["current_run_id"],
        owner_pid=capture["current_owner_pid"],
    )
    provenance = dict(capture.get("gpu_receipts") or {}).get("provenance") or {}
    offload = canary.build_offload_receipt(
        dict(capture.get("runtime_identity") or {}),
        provenance,
        int(dict(context["model_spec"]).get("model_block_count", 0) or 0),
    )
    _span_receipt(spans, "reduction", phase_started, started, len(parsed_rows))

    validation_started = time.monotonic()
    affected_receipts, affected = _run_affected_validation(root, raw_dir, started)
    affected_ok = bool(affected.get("passed"))
    completion_observed = int(
        len(parsed_rows) == 96
        and len(raw_files) == 96
        and current["model_invoked"] is True
        and offload["all_layers_offloaded"] is True
        and affected_ok
    )
    gates = [
        gate_row(
            "all_planned_dispositions",
            EXPERIMENT_ID,
            "planned_call_count",
            "==",
            96,
            reduced["planned_call_count"],
            principle="Every planned call stays accountable.",
            category="completion",
        ),
        gate_row(
            "cuda_all_layer_offload",
            "owned_runtime_receipt",
            "all_layers_offloaded",
            "==",
            True,
            offload["all_layers_offloaded"],
            principle="CPU fallback cannot supply headline evidence.",
            category="evidence",
        ),
        gate_row(
            "affected_validation",
            "validation_receipts",
            "required_checks_passed",
            "==",
            True,
            affected_ok,
            principle="Genuine affected failures disqualify the capture.",
            category="validation",
        ),
    ]
    artifact = _base_artifact()
    artifact.update(
        {
            "status": "complete_capture_awaiting_terminal_readers",
            "started_at_utc": started_at,
            "completed_at_utc": utc_now(),
            "preconditions_checked": [deepcopy(dict(row)) for row in checks],
            **current,
            "current_invocation_events": deepcopy(capture["current_invocation_events"]),
            "current_run_id": capture["current_run_id"],
            "current_owner_pid": capture["current_owner_pid"],
            "inference_substrate": "owned_native_cuda_llama_cpp_bounded_generation",
            "inference_substrate_details": {
                "model": MODEL_ID,
                "quantization": QUANTIZATION,
                "runner": str(context.get("server_path") or ""),
                "gpu_uuid": dict(capture.get("runtime_identity") or {}).get("gpu_uuid"),
                "gpu_name": "NVIDIA GeForce RTX 3090",
                "embedded_tokenizer": True,
                "embedded_chat_template": True,
            },
            "inference_substrate_class": substrate_class_from_counts(current["invocation_counts"]),
            "scientific_duration_s": round(scientific_duration, 6),
            "phase_spans": spans,
            "historical_receipt_sidecars": [sidecar],
            "rows": deepcopy(parsed_rows),
            **reduced,
            "sample_size_budget": {
                **artifact["sample_size_budget"],
                "attempted": reduced["attempted_call_count"],
                "completed": reduced["completed_call_count"],
                "failed": reduced["failed_call_count"],
                "censored": reduced["censored_call_count"],
                "unstarted": reduced["unstarted_call_count"],
            },
            "acceptance_gate_results": gates,
            "gate_check_summary": gate_check_summary([*checks, *gates]),
            "honest_verdict": "complete_null_capture_ready_semantics_pending_exp7417",
            "verdict_class": "null",
            "validation_receipts": affected_receipts,
            "repository_health": {
                "status": "healthy" if affected_ok else "required_checks_failed",
                "unrelated_broad_suite_observations": [],
                "affects_required_checks": not affected_ok,
                "affected_reduction": affected,
            },
            "extraction_capture_complete_score": completion_observed,
            "raw_capture_manifest": _raw_manifest(parsed_rows),
            "model_receipt": {
                **deepcopy(dict(capture.get("runtime_identity") or {})),
                "model_load": deepcopy(capture.get("load_receipt") or {}),
                "gpu_provenance": deepcopy(provenance),
                **offload,
                "reservation": {
                    "lease_wait_timeout_s": LEASE_WAIT_TIMEOUT_S,
                    "waited_s": 0.0,
                    "policy": "single_read_only_capacity_check_then_owned_acquire",
                },
                "cleanup": deepcopy(dict(capture.get("gpu_receipts") or {}).get("cleanup") or {}),
                "lease_release": deepcopy(
                    dict(capture.get("gpu_receipts") or {}).get("lease_release") or {}
                ),
            },
        }
    )
    artifact["rows"] = deepcopy(artifact["extraction_rows"])
    artifact["source_artifact_hashes"] = _source_hashes(root, context, raw_files, sidecar)
    artifact["duration_s"] = round(time.monotonic() - started, 6)
    artifact["validation_duration_s"] = round(time.monotonic() - validation_started, 6)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    candidate = raw_dir / "measured_terminal_candidate.json"
    atomic_json(candidate, artifact)

    progress(started, "validation", "before_terminal_commands", candidate=candidate)
    terminal_receipts = run_categorized_commands(
        root,
        [
            PlannedCommand(
                spec=spec,
                category="safety" if spec.name == "adversarial_verify" else "completion",
                required=True,
            )
            for spec in _terminal_commands(root, candidate)
        ],
        log_dir=raw_dir / "validation/terminal",
        heartbeat_s=60.0,
    )
    progress(started, "validation", "after_terminal_commands", count=len(terminal_receipts))
    terminal_ok = _receipts_pass(terminal_receipts, TERMINAL_CHECK_NAMES)
    independent_ok = not independent_reduce_artifact(artifact, root=root)
    adversarial = next(
        (row for row in terminal_receipts if row.get("name") == "adversarial_verify"), {}
    )
    flagged = adversarial.get("passed") is not True
    _span_receipt(
        spans,
        "validation",
        validation_started,
        started,
        len(affected_receipts) + len(terminal_receipts),
    )
    final_complete = int(completion_observed and terminal_ok and independent_ok and not flagged)
    disqualified = not affected_ok or not terminal_ok or not independent_ok or flagged
    if disqualified:
        artifact.update(
            {
                "status": "complete_required_validation_failed",
                "verdict_class": "disqualified",
                "honest_verdict": "complete_disqualified_extraction_capture_required_check_failed",
            }
        )
    else:
        artifact["status"] = "complete_extraction_capture"
    artifact["flagged_adversarial"] = flagged
    artifact["extraction_capture_complete_score"] = final_complete
    artifact["validation_receipts"] = [*affected_receipts, *terminal_receipts]
    artifact["phase_spans"] = spans
    artifact["completed_at_utc"] = utc_now()
    artifact["duration_s"] = round(time.monotonic() - started, 6)
    artifact["validation_duration_s"] = round(time.monotonic() - validation_started, 6)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    errors = validate_artifact(artifact, require_terminal=True, root=root)
    if errors:
        artifact.update(
            {
                "status": "complete_internal_validation_failed",
                "verdict_class": "disqualified",
                "honest_verdict": "complete_disqualified_internal_artifact_validation_failed",
                "flagged_adversarial": True,
                "extraction_capture_complete_score": 0,
                "internal_validation_errors": errors,
            }
        )
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    progress(started, "write", "before_atomic_publish", artifact=output)
    atomic_json(output, artifact)
    progress(
        started,
        "write",
        "after_atomic_publish",
        artifact=output,
        verdict=artifact["honest_verdict"],
    )
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI.
    """Run live capture or cold-validate one measured candidate."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=_date_argument, default=RUN_DATE)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    started = time.monotonic()
    progress(started, "entrypoint", "start", date=args.date)
    if args.validate is not None:
        value = load_object(args.validate)
        errors = [
            *validate_artifact(value, require_terminal=False, root=REPO_ROOT),
            *independent_reduce_artifact(value, root=REPO_ROOT),
        ]
        errors = list(dict.fromkeys(errors))
        print(canonical_json({"errors": errors}), flush=True)
        return int(bool(errors))
    result = run_experiment(root=REPO_ROOT, run_date=args.date)
    print(
        canonical_json(
            {
                "artifact": str(REPO_ROOT / RESULT_PATH),
                "status": result["status"],
                "honest_verdict": result["honest_verdict"],
                "extraction_capture_complete_score": result["extraction_capture_complete_score"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
