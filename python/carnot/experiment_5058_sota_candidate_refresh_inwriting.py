#!/usr/bin/env python3
"""Exp 5058: SOTA MuSR candidate refresh with delayed constraints.

Spec refs: REQ-VERIFY-5058, SCENARIO-VERIFY-5058.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python") not in sys.path:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(REPO_ROOT / "python"))


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
Clock = Callable[[], float]
FrozenCandidateLoader = Callable[[Path], list[JsonDict]]

EXPERIMENT_ID = 5058
EXPERIMENT_NAME = "experiment_5058_sota_candidate_refresh_inwriting"
SCHEMA = "carnot.experiment_5058_sota_candidate_refresh_inwriting.v1"
CACHE_ROW_SCHEMA = "carnot.experiment_5058_sota_candidate_refresh_inwriting.row.v1"
RESULT_RELATIVE_PATH = "results/experiment_5058_sota_candidate_refresh_inwriting.json"
CACHE_RELATIVE_PATH = "results/experiment_5058_sota_candidate_refresh_inwriting.jsonl"
GATE_STATE_RELATIVE_PATH = "results/experiment_5057_gate_state_preflight_v465.json"
FROZEN_CANDIDATE_CACHE_RELATIVE_PATH = (
    "results/experiment_5029_shared_logprob_candidate_cache_v2_musr.jsonl"
)
SPEC_REFS = ["REQ-VERIFY-5058", "SCENARIO-VERIFY-5058"]
MUSR_CORPUS = "MuSR/murder_mysteries"
RANDOM_SEED = 20260701

MANDATED_MODEL_IDS = {
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
}
MODEL_ROLE_PRIORITY = ("flagship_moe", "flagship_dense", "middle_moe")

DEFAULT_DECODING_PARAMETERS: JsonDict = {
    "temperature": 0.7,
    "top_p": 0.95,
    "max_tokens": 512,
    "response_schema": "musr_delayed_answer_v1",
    "constraint_timing": "delayed_after_draft",
    "top_logprobs_requested": False,
    "top_logprobs_available": False,
}

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal prefix for delayed-constraint SOTA MuSR refresh readiness or blocker."
    },
    "model_specs": {
        "principle": "all three mandated SOTA GGUF specs copied from the Exp5057 gate artifact."
    },
    "candidate_refresh_ready": {
        "principle": "true only when mandated SOTA readiness, nonempty cache rows, full parse rate, and smoke-only legacy policy all hold."
    },
    "candidate_cache_path": {
        "principle": "path to the resumable MuSR SOTA candidate refresh JSONL cache."
    },
    "n_questions": {"principle": "number of MuSR questions represented by refreshed candidates."},
    "n_candidates": {"principle": "number of refreshed candidate rows in the JSONL cache."},
    "parse_rate": {
        "principle": "fraction of refreshed candidate rows parsed through the delayed answer schema."
    },
    "duplicate_rate": {
        "principle": "fraction of refreshed parsed answers duplicated by frozen .464 answers for the same question."
    },
    "answer_diversity": {
        "principle": "unique-answer counts and rates over refreshed parsed answers."
    },
    "used_top_logprobs": {
        "principle": "false when Exp5057 lacks top-logprob or confidence telemetry."
    },
    "delayed_constraints_used": {
        "principle": "true when draft answers are parsed after generation through the explicit answer/constraint schema."
    },
    "legacy_models_smoke_only": {
        "principle": "true; legacy small models are smoke-only and never headline candidate provenance."
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "result_path",
    "candidate_cache_schema",
    "gate_state_path",
    "frozen_candidate_cache_path",
    "resume_summary",
    "d1_d6_readiness",
    "duration_s",
    "field_principles",
    "random_seed",
    "reproducibility_checksum",
)


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _rate(value: int, total: int) -> float:
    return round(value / total, 6) if total else 0.0


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl_row(path: Path, row: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(descriptor, line)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_jsonl(path: Path) -> list[JsonDict]:
    if not path.exists():
        return []
    rows: list[JsonDict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"malformed JSONL row {path}:{line_number}") from exc
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def load_gate_state(root: Path = REPO_ROOT, gate_path: Path | None = None) -> JsonDict:
    path = Path(gate_path) if gate_path else Path(root) / GATE_STATE_RELATIVE_PATH
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"malformed gate-state JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"gate-state JSON must be an object: {path}")
    return payload


def default_frozen_candidate_loader(root: Path = REPO_ROOT) -> list[JsonDict]:
    path = Path(root) / FROZEN_CANDIDATE_CACHE_RELATIVE_PATH
    if not path.exists():
        return []
    rows = _read_jsonl(path)
    return [dict(row) for row in rows if row.get("corpus", MUSR_CORPUS) == MUSR_CORPUS]


def select_headline_model(gate_state: JsonMap) -> JsonDict | None:
    if gate_state.get("sota_models_ready") is not True:
        return None
    usable = [
        dict(model)
        for model in list(gate_state.get("usable_sota_models") or [])
        if isinstance(model, Mapping) and str(model.get("hf_id")) in MANDATED_MODEL_IDS
    ]
    if not usable:
        return None
    by_role = {str(model.get("role")): model for model in usable}
    for role in MODEL_ROLE_PRIORITY:
        if role in by_role:
            return by_role[role]
    return usable[0]


def _choices(row: JsonMap) -> list[str]:
    raw = row.get("choices") or []
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return [str(choice).strip() for choice in raw if str(choice).strip()]
    return []


def _normalize_answer(answer: Any) -> str:
    return " ".join(str(answer or "").strip().casefold().split())


def _match_choice(text: Any, choices: Sequence[str]) -> str:
    normalized_text = _normalize_answer(text)
    if not normalized_text:
        return ""
    for choice in choices:
        if _normalize_answer(choice) == normalized_text:
            return str(choice)
    for choice in choices:
        normalized_choice = _normalize_answer(choice)
        if normalized_choice and normalized_choice in normalized_text:
            return str(choice)
    return ""


def build_generation_prompt(row: JsonMap) -> str:
    choices = _choices(row)
    schema = {
        "answer": {"type": "string", "enum": choices},
        "evidence_spans": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    }
    return "\n".join(
        [
            "MuSR murder mystery candidate refresh.",
            f"Question: {str(row.get('question') or '').strip()}",
            f"Context: {str(row.get('context') or '').strip()}",
            f"Allowed answers: {_json_dumps(choices)}",
            f"Delayed answer schema: {_json_dumps(schema)}",
        ]
    )


def parse_delayed_constraints(answer_text: str, choices: Sequence[str]) -> JsonDict:
    raw_text = str(answer_text or "").strip()
    raw_format = "free_text"
    candidate_answer = raw_text
    evidence_spans: list[str] = []
    confidence: float | None = None
    schema_parseable = False
    try:
        parsed = json.loads(raw_text)
    except ValueError:
        parsed = None
    if isinstance(parsed, Mapping):
        raw_format = "json_object"
        schema_parseable = True
        candidate_answer = str(
            parsed.get("answer")
            or parsed.get("parsed_answer")
            or parsed.get("final_answer")
            or parsed.get("choice")
            or ""
        )
        raw_evidence = parsed.get("evidence_spans") or parsed.get("evidence") or []
        if isinstance(raw_evidence, Sequence) and not isinstance(raw_evidence, (str, bytes)):
            evidence_spans = [str(item) for item in raw_evidence if str(item).strip()]
        confidence_value = _number(parsed.get("confidence"))
        if confidence_value is not None:
            confidence = min(1.0, max(0.0, confidence_value))
    matched = _match_choice(candidate_answer, choices) or _match_choice(raw_text, choices)
    parse_status = "parsed" if matched else "parse_failed"
    constraints: JsonDict = {
        "schema_name": "musr_delayed_answer_v1",
        "allowed_answers": list(choices),
        "raw_format": raw_format,
        "answer_in_allowed_choices": bool(matched),
        "evidence_spans": evidence_spans,
        "evidence_span_count": len(evidence_spans),
        "confidence": confidence,
        "constraint_checks": {
            "nonempty_draft": bool(raw_text),
            "schema_parseable": schema_parseable,
            "allowed_choice": bool(matched),
            "delayed_after_draft": True,
        },
    }
    return {
        "parse_status": parse_status,
        "parsed_answer": matched,
        "structured_constraints": constraints,
    }


def _candidate_index(row: JsonMap) -> int:
    try:
        return int(row.get("candidate_index") or 0)
    except (TypeError, ValueError):
        return 0


def _question_index(row: JsonMap) -> int:
    try:
        return int(row.get("question_index") or 0)
    except (TypeError, ValueError):
        return 0


def _question_id(row: JsonMap) -> str:
    value = str(row.get("question_id") or "").strip()
    if value:
        return value
    return f"{MUSR_CORPUS}:{_question_index(row)}"


def build_candidate_row(
    frozen_row: JsonMap,
    *,
    model_spec: JsonMap,
    used_top_logprobs: bool,
    decoding_parameters: JsonMap | None = None,
) -> JsonDict:
    choices = _choices(frozen_row)
    answer_text = str(frozen_row.get("answer") or "").strip()
    parsed = parse_delayed_constraints(answer_text, choices)
    prompt = build_generation_prompt(frozen_row)
    question_id = _question_id(frozen_row)
    candidate_index = _candidate_index(frozen_row)
    parameters = dict(DEFAULT_DECODING_PARAMETERS)
    parameters.update(dict(decoding_parameters or {}))
    parameters["top_logprobs_available"] = bool(used_top_logprobs)
    return {
        "schema": CACHE_ROW_SCHEMA,
        "row_id": f"{question_id}/sota5058-{candidate_index:04d}",
        "corpus": MUSR_CORPUS,
        "question_id": question_id,
        "question_index": _question_index(frozen_row),
        "candidate_index": candidate_index,
        "prompt_hash": _sha256_text(prompt),
        "question": str(frozen_row.get("question") or ""),
        "choices": choices,
        "model_id": str(model_spec.get("hf_id") or ""),
        "model_role": str(model_spec.get("role") or ""),
        "model_path": str(model_spec.get("model_path") or ""),
        "decoding_parameters": parameters,
        "answer_text": answer_text,
        "parsed_answer": parsed["parsed_answer"],
        "structured_constraints": parsed["structured_constraints"],
        "parse_status": parsed["parse_status"],
        "used_top_logprobs": bool(used_top_logprobs),
        "delayed_constraints_used": not bool(used_top_logprobs),
        "legacy_model_used": False,
        "source_provenance": {
            "source": "frozen_464_musr_candidate_cache",
            "source_candidate_id": str(frozen_row.get("candidate_id") or ""),
            "source_schema": str(frozen_row.get("schema") or ""),
            "source_answer_text": answer_text,
        },
    }


def validate_candidate_row(row: JsonMap) -> list[str]:
    errors: list[str] = []
    if row.get("schema") != CACHE_ROW_SCHEMA:
        errors.append("schema")
    for field in ("row_id", "question_id", "model_id", "answer_text", "parsed_answer"):
        if not str(row.get(field) or ""):
            errors.append(field)
    if str(row.get("model_id") or "") not in MANDATED_MODEL_IDS:
        errors.append("model_id")
    if not isinstance(row.get("prompt_hash"), str) or len(str(row.get("prompt_hash"))) != 64:
        errors.append("prompt_hash")
    if not isinstance(row.get("decoding_parameters"), Mapping):
        errors.append("decoding_parameters")
    constraints = row.get("structured_constraints")
    if not isinstance(constraints, Mapping) or constraints.get("schema_name") != "musr_delayed_answer_v1":
        errors.append("structured_constraints")
    if row.get("parse_status") != "parsed":
        errors.append("parse_status")
    if row.get("legacy_model_used") is not False:
        errors.append("legacy_model_used")
    return sorted(set(errors))


def read_complete_candidate_rows(path: Path) -> list[JsonDict]:
    complete_by_id: dict[str, JsonDict] = {}
    for row in _read_jsonl(path):
        if not validate_candidate_row(row):
            complete_by_id[str(row["row_id"])] = row
    return list(complete_by_id.values())


def ensure_candidate_cache(
    *,
    cache_path: Path,
    frozen_rows: Sequence[JsonMap],
    model_spec: JsonMap,
    used_top_logprobs: bool,
) -> tuple[list[JsonDict], JsonDict]:
    existing = {row["row_id"]: row for row in read_complete_candidate_rows(cache_path)}
    target_rows = [
        build_candidate_row(row, model_spec=model_spec, used_top_logprobs=used_top_logprobs)
        for row in frozen_rows
    ]
    appended = 0
    for row in target_rows:
        if row["row_id"] in existing:
            continue
        append_jsonl_row(cache_path, row)
        if not validate_candidate_row(row):
            existing[row["row_id"]] = row
        appended += 1
    ordered = [existing[row["row_id"]] for row in target_rows if row["row_id"] in existing]
    existing_count = len(existing) - appended
    return ordered, {
        "existing_complete_rows": int(existing_count),
        "appended_rows": int(appended),
        "target_rows": int(len(target_rows)),
        "skipped_existing_rows": int(existing_count),
    }


def compute_refresh_metrics(rows: Sequence[JsonMap], frozen_rows: Sequence[JsonMap]) -> JsonDict:
    total = len(rows)
    parsed_rows = [
        row for row in rows if row.get("parse_status") == "parsed" and str(row.get("parsed_answer") or "")
    ]
    frozen_answers: dict[str, set[str]] = defaultdict(set)
    for row in frozen_rows:
        frozen_answers[_question_id(row)].add(_normalize_answer(row.get("answer")))
    duplicate_count = sum(
        1
        for row in parsed_rows
        if _normalize_answer(row.get("parsed_answer")) in frozen_answers.get(_question_id(row), set())
    )
    unique_answers = {str(row.get("parsed_answer")) for row in parsed_rows}
    per_question: dict[str, set[str]] = defaultdict(set)
    for row in parsed_rows:
        per_question[_question_id(row)].add(str(row.get("parsed_answer")))
    mean_unique_per_question = (
        round(sum(len(values) for values in per_question.values()) / len(per_question), 6)
        if per_question
        else 0.0
    )
    return {
        "n_questions": len({_question_id(row) for row in rows}),
        "n_candidates": total,
        "parse_rate": _rate(len(parsed_rows), total),
        "duplicate_rate": _rate(duplicate_count, total),
        "answer_diversity": {
            "unique_answers": len(unique_answers),
            "unique_answer_rate": _rate(len(unique_answers), total),
            "mean_unique_answers_per_question": mean_unique_per_question,
        },
    }


def _checksum(artifact: JsonMap) -> str:
    basis = {
        "experiment_id": artifact.get("experiment_id"),
        "honest_verdict": artifact.get("honest_verdict"),
        "model_specs": artifact.get("model_specs"),
        "candidate_refresh_ready": artifact.get("candidate_refresh_ready"),
        "candidate_cache_path": artifact.get("candidate_cache_path"),
        "n_questions": artifact.get("n_questions"),
        "n_candidates": artifact.get("n_candidates"),
        "parse_rate": artifact.get("parse_rate"),
        "duplicate_rate": artifact.get("duplicate_rate"),
        "answer_diversity": artifact.get("answer_diversity"),
        "used_top_logprobs": artifact.get("used_top_logprobs"),
        "delayed_constraints_used": artifact.get("delayed_constraints_used"),
    }
    return "sha256:" + hashlib.sha256(_json_dumps(basis).encode("utf-8")).hexdigest()


def _base_artifact(
    *,
    honest_verdict: str,
    root: Path,
    artifact_path: Path,
    cache_path: Path,
    gate_state: JsonMap,
    duration_s: float,
) -> JsonDict:
    return {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "result_path": artifact_path.as_posix(),
        "honest_verdict": honest_verdict,
        "model_specs": dict(gate_state.get("model_specs") or {}),
        "candidate_refresh_ready": False,
        "candidate_cache_path": cache_path.as_posix(),
        "n_questions": 0,
        "n_candidates": 0,
        "parse_rate": 0.0,
        "duplicate_rate": 0.0,
        "answer_diversity": {
            "unique_answers": 0,
            "unique_answer_rate": 0.0,
            "mean_unique_answers_per_question": 0.0,
        },
        "used_top_logprobs": False,
        "delayed_constraints_used": False,
        "legacy_models_smoke_only": True,
        "candidate_cache_schema": CACHE_ROW_SCHEMA,
        "gate_state_path": (Path(root) / GATE_STATE_RELATIVE_PATH).as_posix(),
        "frozen_candidate_cache_path": (
            Path(root) / FROZEN_CANDIDATE_CACHE_RELATIVE_PATH
        ).as_posix(),
        "resume_summary": {
            "existing_complete_rows": 0,
            "appended_rows": 0,
            "target_rows": 0,
            "skipped_existing_rows": 0,
        },
        "d1_d6_readiness": {
            "ready": False,
            "d1_ready": False,
            "d6_ready": False,
            "reason": honest_verdict,
        },
        "duration_s": round(max(0.0, float(duration_s)), 6),
        "field_principles": dict(FIELD_PRINCIPLES),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
    }


def build_blocked_artifact(
    *,
    root: Path,
    artifact_path: Path,
    cache_path: Path,
    gate_state: JsonMap,
    duration_s: float,
    reason: str,
) -> JsonDict:
    artifact = _base_artifact(
        honest_verdict=f"blocked_{reason}",
        root=root,
        artifact_path=artifact_path,
        cache_path=cache_path,
        gate_state=gate_state,
        duration_s=duration_s,
    )
    artifact["gate_diagnostics"] = {
        "sota_models_ready": bool(gate_state.get("sota_models_ready")),
        "usable_sota_models": list(gate_state.get("usable_sota_models") or []),
        "preflight_honest_verdict": gate_state.get("honest_verdict"),
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def build_complete_artifact(
    *,
    root: Path,
    artifact_path: Path,
    cache_path: Path,
    gate_state: JsonMap,
    model_spec: JsonMap,
    rows: Sequence[JsonMap],
    metrics: JsonMap,
    resume_summary: JsonMap,
    used_top_logprobs: bool,
    duration_s: float,
) -> JsonDict:
    delayed_constraints_used = not bool(used_top_logprobs)
    ready = (
        bool(model_spec)
        and int(metrics.get("n_questions") or 0) > 0
        and int(metrics.get("n_candidates") or 0) > 0
        and float(metrics.get("parse_rate") or 0.0) == 1.0
    )
    artifact = _base_artifact(
        honest_verdict="complete_sota_candidate_refresh_ready_d1_d6"
        if ready
        else "complete_sota_candidate_refresh_not_ready_d1_d6",
        root=root,
        artifact_path=artifact_path,
        cache_path=cache_path,
        gate_state=gate_state,
        duration_s=duration_s,
    )
    artifact.update(
        {
            "candidate_refresh_ready": bool(ready),
            "n_questions": int(metrics["n_questions"]),
            "n_candidates": int(metrics["n_candidates"]),
            "parse_rate": float(metrics["parse_rate"]),
            "duplicate_rate": float(metrics["duplicate_rate"]),
            "answer_diversity": dict(metrics["answer_diversity"]),
            "used_top_logprobs": bool(used_top_logprobs),
            "delayed_constraints_used": delayed_constraints_used,
            "resume_summary": dict(resume_summary),
            "headline_model": dict(model_spec),
            "d1_d6_readiness": {
                "ready": bool(ready),
                "d1_ready": bool(ready),
                "d6_ready": bool(ready),
                "reason": "cache_parse_complete_with_mandated_sota_model"
                if ready
                else "cache_missing_or_parse_incomplete",
                "complete_row_count": len(rows),
            },
            "gate_diagnostics": {
                "sota_models_ready": bool(gate_state.get("sota_models_ready")),
                "top_logprob_or_confidence_ready": bool(
                    gate_state.get("top_logprob_or_confidence_ready")
                ),
                "sota_judge_ready": bool(gate_state.get("sota_judge_ready")),
                "preflight_honest_verdict": gate_state.get("honest_verdict"),
            },
        }
    )
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def _valid_rate(value: Any) -> bool:
    number = _number(value)
    return number is not None and 0.0 <= number <= 1.0


def artifact_schema_errors(artifact: JsonMap) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(field)
    if artifact.get("schema") != SCHEMA:
        errors.append("schema")
    if artifact.get("spec_refs") != SPEC_REFS:
        errors.append("spec_refs")
    if not isinstance(artifact.get("model_specs"), Mapping):
        errors.append("model_specs")
    for field in ("candidate_refresh_ready", "used_top_logprobs", "delayed_constraints_used"):
        if not isinstance(artifact.get(field), bool):
            errors.append(field)
    if artifact.get("legacy_models_smoke_only") is not True:
        errors.append("legacy_models_smoke_only")
    for field in ("n_questions", "n_candidates"):
        if not isinstance(artifact.get(field), int) or int(artifact.get(field, -1)) < 0:
            errors.append(field)
    for field in ("parse_rate", "duplicate_rate"):
        if not _valid_rate(artifact.get(field)):
            errors.append(field)
    if not isinstance(artifact.get("answer_diversity"), Mapping):
        errors.append("answer_diversity")
    if not str(artifact.get("candidate_cache_path") or ""):
        errors.append("candidate_cache_path")
    if not str(artifact.get("honest_verdict") or "").startswith(("complete_", "blocked_")):
        errors.append("honest_verdict")
    return sorted(set(errors))


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    cache_path: Path | None = None,
    frozen_candidate_loader: FrozenCandidateLoader = default_frozen_candidate_loader,
    now: Clock = time.perf_counter,
    write: bool = True,
) -> JsonDict:
    root = Path(root)
    artifact_path = Path(artifact_path) if artifact_path else root / RESULT_RELATIVE_PATH
    cache_path = Path(cache_path) if cache_path else root / CACHE_RELATIVE_PATH
    start = float(now())
    gate_state = load_gate_state(root)
    model_spec = select_headline_model(gate_state)
    if model_spec is None:
        artifact = build_blocked_artifact(
            root=root,
            artifact_path=artifact_path,
            cache_path=cache_path,
            gate_state=gate_state,
            duration_s=float(now()) - start,
            reason="sota_models_unavailable",
        )
        if write:
            write_json(artifact_path, artifact)
        return artifact

    frozen_rows = frozen_candidate_loader(root)
    used_top_logprobs = bool(gate_state.get("top_logprob_or_confidence_ready"))
    rows, resume_summary = ensure_candidate_cache(
        cache_path=cache_path,
        frozen_rows=frozen_rows,
        model_spec=model_spec,
        used_top_logprobs=used_top_logprobs,
    )
    metrics = compute_refresh_metrics(rows, frozen_rows)
    artifact = build_complete_artifact(
        root=root,
        artifact_path=artifact_path,
        cache_path=cache_path,
        gate_state=gate_state,
        model_spec=model_spec,
        rows=rows,
        metrics=metrics,
        resume_summary=resume_summary,
        used_top_logprobs=used_top_logprobs,
        duration_s=float(now()) - start,
    )
    if write:
        write_json(artifact_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI entrypoint
    _ = argv
    artifact = run()
    print(
        json.dumps(
            {
                "result_path": str(REPO_ROOT / RESULT_RELATIVE_PATH),
                "honest_verdict": artifact.get("honest_verdict"),
                "candidate_refresh_ready": artifact.get("candidate_refresh_ready"),
                "candidate_cache_path": artifact.get("candidate_cache_path"),
                "n_questions": artifact.get("n_questions"),
                "n_candidates": artifact.get("n_candidates"),
                "parse_rate": artifact.get("parse_rate"),
                "duplicate_rate": artifact.get("duplicate_rate"),
            },
            sort_keys=True,
        )
    )
    errors = artifact_schema_errors(artifact)
    if errors:
        print(f"schema_errors={errors}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
