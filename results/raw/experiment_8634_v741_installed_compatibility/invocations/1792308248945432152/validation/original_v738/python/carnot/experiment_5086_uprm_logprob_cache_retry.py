"""Exp 5086: resumable uPRM token/step logprob cache retry.

Spec refs: REQ-VERIFY-5086, SCENARIO-VERIFY-5086.

The cache built here is intentionally row-oriented and resumable. A long live
llama.cpp scoring run can stop part way through without losing already scored
candidate telemetry, and a later run can skip rows whose hashes and required
token/step fields are already complete.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python") not in sys.path:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(REPO_ROOT / "python"))


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
Clock = Callable[[], float]
EndpointProbe = Callable[[str, float], JsonDict]
CandidateScorer = Callable[..., JsonDict]

EXPERIMENT_ID = 5086
EXPERIMENT_NAME = "experiment_5086_uprm_logprob_cache_retry"
SCHEMA = "carnot.experiment_5086_uprm_logprob_cache_retry.v1"
CACHE_ROW_SCHEMA = "carnot.experiment_5086_uprm_logprob_cache_retry.row.v1"
RESULT_RELATIVE_PATH = "results/experiment_5086_uprm_logprob_cache_retry_v467.json"
CACHE_RELATIVE_PATH = "results/experiment_5086_uprm_logprob_cache_retry_v467.jsonl"
EXP5085_RELATIVE_PATH = "results/experiment_5085_llamacpp_logprob_endpoint_bringup_v467.json"
EXP5058_CACHE_RELATIVE_PATH = "results/experiment_5058_sota_candidate_refresh_inwriting.jsonl"
EXP5029_CACHE_RELATIVE_PATH = "results/experiment_5029_shared_logprob_candidate_cache_v2_musr.jsonl"
SPEC_REFS = ["REQ-VERIFY-5086", "SCENARIO-VERIFY-5086"]
MUSR_CORPUS = "MuSR/murder_mysteries"
RANDOM_SEED = 20260701
DEFAULT_LIMIT_QUESTIONS = 200
DEFAULT_ENDPOINT_TIMEOUT_S = 30.0
TOP_LOGPROBS_REQUESTED = 20

MANDATED_MODEL_IDS: tuple[str, ...] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
ROLE_BY_HF_ID = {
    "unsloth/Qwen3.6-35B-A3B-GGUF": "flagship_moe",
    "unsloth/gemma-4-31B-it-GGUF": "flagship_dense",
    "unsloth/gemma-4-26B-A4B-it-GGUF": "middle_moe",
}

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal prefix distinguishes ready cache from endpoint/input blockers; no 0-row success is allowed."
    },
    "duration_s": {
        "principle": "wall-clock duration records whether this was live endpoint scoring or a blocked precondition pass."
    },
    "inference_substrate": {
        "principle": "live_llm_inference only when rows were scored through the Exp5085 endpoint; otherwise precondition_check_only."
    },
    "preconditions_checked": {
        "principle": "records Exp5085 artifact hash, endpoint URL, selected model path, cache input path, disk space, and live endpoint probe evidence before scoring."
    },
    "model_specs": {
        "principle": "all three mandated SOTA GGUF IDs plus resolved paths and the selected Exp5085 endpoint model."
    },
    "logprob_cache_ready": {
        "principle": "true only when every target candidate row has token logprobs from the endpoint."
    },
    "step_cache_ready": {
        "principle": "true only when every complete row has deterministic step boundaries with token spans."
    },
    "cache_path": {
        "principle": "resumable JSONL cache path consumed by downstream uPRM scoring."
    },
    "n_questions": {
        "principle": "number of MuSR questions represented by complete scored rows."
    },
    "n_candidates": {
        "principle": "number of target existing candidate strings selected for scoring."
    },
    "n_rows_complete": {
        "principle": "number of valid row-level token/step telemetry rows flushed to disk."
    },
    "parse_rate": {
        "principle": "complete valid rows divided by target candidates; catches partial runs honestly."
    },
    "top_logprob_coverage": {
        "principle": "fraction of complete rows whose endpoint response exposed top-logprob alternatives."
    },
    "endpoint_used": {
        "principle": "concrete Exp5085 endpoint route used for scoring or attempted during the blocker."
    },
    "flagged_adversarial": {
        "principle": "false for honest blocked/ready artifacts unless the runner itself detects inconsistent provenance."
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "result_path",
    "cache_row_schema",
    "field_principles",
    "random_seed",
    "reproducibility_checksum",
)


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_payload(payload: Any) -> str:
    return "sha256:" + hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _rate(value: int, total: int) -> float:
    return round(value / total, 6) if total else 0.0


def write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl_row(path: Path, row: JsonMap) -> None:
    """Append one cache row and fsync it so an interrupted run can resume."""

    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(descriptor, line)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_json(path: Path) -> JsonDict:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return loaded


def _read_jsonl(path: Path) -> list[JsonDict]:
    if not path.exists():
        return []
    rows: list[JsonDict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            loaded = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"malformed JSONL row {path}:{line_number}") from exc
        if isinstance(loaded, dict):
            rows.append(loaded)
    return rows


def _endpoint_route(endpoint: str) -> str:
    value = str(endpoint or "").strip().rstrip("/")
    if value.endswith("/completion") or value.endswith("/v1/completions"):
        return value
    return value + "/completion"


def _http_post_json(url: str, payload: JsonMap, timeout_s: float) -> tuple[int, Any]:
    data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    request = Request(url, data=data, method="POST", headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout_s) as response:
        status = int(getattr(response, "status", 0) or 0)
        raw = response.read().decode("utf-8", "replace")
    try:
        return status, json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return status, {"raw": raw}


def _http_error_detail(exc: BaseException) -> str:
    if isinstance(exc, HTTPError):
        try:
            body = exc.read().decode("utf-8", "replace")
        except Exception:
            body = ""
        return f"HTTPError {exc.code}: {body[:240]}" if body else f"HTTPError {exc.code}"
    if isinstance(exc, (URLError, OSError, TimeoutError)):
        return f"{type(exc).__name__}: {exc}"
    return f"{type(exc).__name__}: {exc}"


def _top_logprob_row(raw: Any) -> dict[str, float]:
    row: dict[str, float] = {}
    if isinstance(raw, Mapping):
        for token, logprob in raw.items():
            value = _number(logprob)
            if value is not None:
                row[str(token)] = value
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            token = item.get("token", item.get("text", item.get("content", "")))
            value = _number(item.get("logprob"))
            if value is not None:
                row[str(token)] = value
    return row


def _response_text(payload: Any) -> str:
    if not isinstance(payload, Mapping):
        return ""
    for key in ("content", "response", "text"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    choices = payload.get("choices")
    if isinstance(choices, Sequence) and choices and isinstance(choices[0], Mapping):
        choice = choices[0]
        text = choice.get("text")
        if isinstance(text, str):
            return text
        message = choice.get("message")
        if isinstance(message, Mapping) and isinstance(message.get("content"), str):
            return str(message["content"])
    return ""


def parse_logprob_payload(payload: JsonMap) -> JsonDict:
    """Extract text, token strings, chosen logprobs, and top-logprob rows."""

    text = _response_text(payload)
    tokens: list[str] = []
    token_logprobs: list[float] = []
    top_logprobs: list[dict[str, float]] = []

    probabilities = payload.get("completion_probabilities")
    if isinstance(probabilities, Sequence) and not isinstance(probabilities, (str, bytes)):
        for item in probabilities:
            if not isinstance(item, Mapping):
                continue
            token = item.get("token", item.get("content", item.get("text", "")))
            tokens.append(str(token))
            value = _number(item.get("logprob"))
            if value is not None:
                token_logprobs.append(value)
            row = _top_logprob_row(item.get("top_logprobs"))
            if row:
                top_logprobs.append(row)

    choices = payload.get("choices")
    if isinstance(choices, Sequence) and choices and isinstance(choices[0], Mapping):
        logprobs = choices[0].get("logprobs")
        if isinstance(logprobs, Mapping):
            if not tokens:
                tokens = [str(token) for token in logprobs.get("tokens") or []]
            for raw in logprobs.get("token_logprobs") or []:
                value = _number(raw)
                if value is not None:
                    token_logprobs.append(value)
            content = logprobs.get("content")
            if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
                for item in content:
                    if not isinstance(item, Mapping):
                        continue
                    value = _number(item.get("logprob"))
                    if value is not None:
                        token_logprobs.append(value)
                    row = _top_logprob_row(item.get("top_logprobs"))
                    if row:
                        top_logprobs.append(row)
            for raw_row in logprobs.get("top_logprobs") or []:
                row = _top_logprob_row(raw_row)
                if row:
                    top_logprobs.append(row)

    return {
        "completion_text": text,
        "tokens": tokens,
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs,
    }


def _limited_question_rows(rows: Sequence[JsonMap], limit_questions: int) -> list[JsonMap]:
    selected_questions: set[str] = set()
    selected_rows: list[JsonMap] = []
    for row in rows:
        question_id = str(row.get("question_id") or "")
        if not question_id:
            continue
        if question_id not in selected_questions and len(selected_questions) >= limit_questions:
            continue
        selected_questions.add(question_id)
        selected_rows.append(row)
    return selected_rows


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


def _fallback_by_key(fallback_rows: Sequence[JsonMap]) -> dict[tuple[str, int], JsonMap]:
    return {
        (str(row.get("question_id") or ""), _candidate_index(row)): row
        for row in fallback_rows
        if str(row.get("question_id") or "")
    }


def _candidate_from_5058(row: JsonMap, enrich: JsonMap | None, source_path: str) -> JsonDict:
    question_id = str(row.get("question_id") or "")
    candidate_index = _candidate_index(row)
    candidate_text = str(row.get("answer_text") or row.get("parsed_answer") or "").strip()
    return {
        "candidate_id": str(row.get("row_id") or f"{question_id}/sota5058-{candidate_index:04d}"),
        "source_candidate_id": str(row.get("row_id") or ""),
        "source_schema": str(row.get("schema") or ""),
        "source_cache_path": source_path,
        "candidate_source": "exp5058_sota_candidate_refresh",
        "corpus": MUSR_CORPUS,
        "question_id": question_id,
        "question_index": _question_index(row),
        "candidate_index": candidate_index,
        "question": str(row.get("question") or (enrich or {}).get("question") or ""),
        "context": str((enrich or {}).get("context") or row.get("context") or ""),
        "choices": list(row.get("choices") or (enrich or {}).get("choices") or []),
        "gold": str((enrich or {}).get("gold") or row.get("gold") or ""),
        "candidate_text": candidate_text,
    }


def _candidate_from_5029(row: JsonMap, source_path: str) -> JsonDict:
    question_id = str(row.get("question_id") or "")
    candidate_index = _candidate_index(row)
    return {
        "candidate_id": str(row.get("candidate_id") or f"{question_id}/cached-{candidate_index}"),
        "source_candidate_id": str(row.get("candidate_id") or ""),
        "source_schema": str(row.get("schema") or ""),
        "source_cache_path": source_path,
        "candidate_source": "exp5029_shared_logprob_candidate_cache_v2",
        "corpus": MUSR_CORPUS,
        "question_id": question_id,
        "question_index": _question_index(row),
        "candidate_index": candidate_index,
        "question": str(row.get("question") or ""),
        "context": str(row.get("context") or ""),
        "choices": list(row.get("choices") or []),
        "gold": str(row.get("gold") or ""),
        "candidate_text": str(row.get("answer") or "").strip(),
    }


def _valid_candidate(candidate: JsonMap) -> bool:
    return bool(
        str(candidate.get("question_id") or "")
        and str(candidate.get("candidate_id") or "")
        and str(candidate.get("candidate_text") or "").strip()
    )


def load_candidate_inputs_from_rows(
    *,
    primary_rows: Sequence[JsonMap],
    fallback_rows: Sequence[JsonMap],
    min_questions: int = DEFAULT_LIMIT_QUESTIONS,
    limit_questions: int = DEFAULT_LIMIT_QUESTIONS,
    primary_path: str = EXP5058_CACHE_RELATIVE_PATH,
    fallback_path: str = EXP5029_CACHE_RELATIVE_PATH,
) -> tuple[list[JsonDict], JsonDict]:
    """Select existing MuSR candidate strings, preferring Exp5058 over Exp5029."""

    fallback_map = _fallback_by_key(fallback_rows)
    primary_candidates = [
        _candidate_from_5058(
            row,
            fallback_map.get((str(row.get("question_id") or ""), _candidate_index(row))),
            primary_path,
        )
        for row in primary_rows
        if str(row.get("corpus") or MUSR_CORPUS) == MUSR_CORPUS
    ]
    primary_candidates = [row for row in primary_candidates if _valid_candidate(row)]
    primary_candidates = _limited_question_rows(
        sorted(primary_candidates, key=lambda row: (_question_index(row), _candidate_index(row))),
        limit_questions,
    )
    primary_questions = {str(row["question_id"]) for row in primary_candidates}
    if len(primary_questions) >= min_questions:
        source = "exp5058_enriched_by_exp5029" if fallback_map else "exp5058"
        return primary_candidates, {
            "available": True,
            "candidate_source": source,
            "cache_input_path": primary_path,
            "fallback_enrichment_path": fallback_path if fallback_map else None,
            "n_questions": len(primary_questions),
            "n_candidates": len(primary_candidates),
        }

    fallback_candidates = [
        _candidate_from_5029(row, fallback_path)
        for row in fallback_rows
        if str(row.get("corpus") or MUSR_CORPUS) == MUSR_CORPUS
    ]
    fallback_candidates = [row for row in fallback_candidates if _valid_candidate(row)]
    fallback_candidates = _limited_question_rows(
        sorted(fallback_candidates, key=lambda row: (_question_index(row), _candidate_index(row))),
        limit_questions,
    )
    fallback_questions = {str(row["question_id"]) for row in fallback_candidates}
    return fallback_candidates, {
        "available": len(fallback_questions) >= min_questions,
        "candidate_source": "exp5029_shared_logprob_candidate_cache_v2",
        "cache_input_path": fallback_path,
        "fallback_enrichment_path": None,
        "n_questions": len(fallback_questions),
        "n_candidates": len(fallback_candidates),
    }


def load_candidate_inputs(
    *,
    root: Path = REPO_ROOT,
    min_questions: int = DEFAULT_LIMIT_QUESTIONS,
    limit_questions: int = DEFAULT_LIMIT_QUESTIONS,
    primary_path: Path | None = None,
    fallback_path: Path | None = None,
) -> tuple[list[JsonDict], JsonDict]:
    root = Path(root)
    primary = Path(primary_path) if primary_path else root / EXP5058_CACHE_RELATIVE_PATH
    fallback = Path(fallback_path) if fallback_path else root / EXP5029_CACHE_RELATIVE_PATH
    primary_rows = _read_jsonl(primary)
    fallback_rows = _read_jsonl(fallback)
    return load_candidate_inputs_from_rows(
        primary_rows=primary_rows,
        fallback_rows=fallback_rows,
        min_questions=min_questions,
        limit_questions=limit_questions,
        primary_path=primary.as_posix(),
        fallback_path=fallback.as_posix(),
    )


def build_scoring_prompt(candidate: JsonMap) -> str:
    choices = ", ".join(str(choice) for choice in candidate.get("choices") or [])
    context = " ".join(str(candidate.get("context") or "").split())
    candidate_text = str(candidate.get("candidate_text") or "").strip()
    return "\n".join(
        [
            "MuSR uPRM token logprob scoring.",
            f"Question: {str(candidate.get('question') or '').strip()}",
            f"Context: {context}",
            f"Allowed choices: {choices}",
            "Candidate trajectory:",
            candidate_text,
            "End candidate trajectory.",
        ]
    )


def segment_candidate_steps(candidate_text: str) -> list[JsonDict]:
    """Split candidate text into deterministic step spans with character offsets."""

    text = str(candidate_text or "")
    line_steps: list[JsonDict] = []
    for match in re.finditer(r"[^\n]+", text):
        start, end = match.span()
        raw = match.group(0)
        stripped = raw.strip()
        if not stripped:
            continue
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw.rstrip())
        line_steps.append(
            {
                "text_start": start + leading,
                "text_end": start + trailing,
                "text": stripped,
            }
        )
    raw_steps = line_steps
    if len(raw_steps) <= 1:
        sentence_steps: list[JsonDict] = []
        for match in re.finditer(r"[^.!?]+[.!?]*", text):
            raw = match.group(0)
            stripped = raw.strip()
            if not stripped:
                continue
            leading = len(raw) - len(raw.lstrip())
            trailing = len(raw.rstrip())
            sentence_steps.append(
                {
                    "text_start": match.start() + leading,
                    "text_end": match.start() + trailing,
                    "text": stripped,
                }
            )
        raw_steps = sentence_steps
    if not raw_steps and text.strip():
        start = len(text) - len(text.lstrip())
        end = len(text.rstrip())
        raw_steps = [{"text_start": start, "text_end": end, "text": text.strip()}]
    return [
        {
            "step_index": index,
            "text_start": int(step["text_start"]),
            "text_end": int(step["text_end"]),
            "text_hash": _sha256_text(str(step["text"])),
        }
        for index, step in enumerate(raw_steps)
    ]


def _token_offsets(tokens: Sequence[str]) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        text = str(token)
        offsets.append((cursor, cursor + len(text)))
        cursor += len(text)
    return offsets


def _candidate_token_window(tokens: Sequence[str], candidate_text: str) -> tuple[int, int, int]:
    joined = "".join(str(token) for token in tokens)
    candidate = str(candidate_text or "")
    char_start = joined.find(candidate)
    if char_start < 0 and candidate.strip():
        char_start = joined.find(candidate.strip())
    if char_start < 0:
        return 0, 0, len(tokens)
    char_end = char_start + len(candidate.strip() if joined.find(candidate) < 0 else candidate)
    offsets = _token_offsets(tokens)
    overlapping = [
        index for index, (start, end) in enumerate(offsets) if end > char_start and start < char_end
    ]
    if not overlapping:
        return char_start, 0, len(tokens)
    return char_start, min(overlapping), max(overlapping) + 1


def _step_boundaries(
    *,
    candidate_text: str,
    tokens: Sequence[str],
    token_logprobs: Sequence[float],
) -> list[JsonDict]:
    steps = segment_candidate_steps(candidate_text)
    if not steps:
        return []
    candidate_char_start, candidate_token_start, candidate_token_end = _candidate_token_window(
        tokens, candidate_text
    )
    offsets = _token_offsets(tokens)
    boundaries: list[JsonDict] = []
    for step in steps:
        absolute_start = candidate_char_start + int(step["text_start"])
        absolute_end = candidate_char_start + int(step["text_end"])
        overlapping = [
            index
            for index, (start, end) in enumerate(offsets)
            if end > absolute_start and start < absolute_end
        ]
        if overlapping:
            token_start = min(overlapping)
            token_end = max(overlapping) + 1
        else:
            token_start = candidate_token_start
            token_end = candidate_token_end
        bounded_logprobs = [
            float(token_logprobs[index])
            for index in range(token_start, min(token_end, len(token_logprobs)))
        ]
        logprob_sum = sum(bounded_logprobs) if bounded_logprobs else None
        boundaries.append(
            {
                **step,
                "token_start": int(token_start),
                "token_end": int(token_end),
                "token_count": int(max(0, token_end - token_start)),
                "logprob_sum": round(float(logprob_sum), 12) if logprob_sum is not None else None,
            }
        )
    return boundaries


def build_cache_row(
    *,
    candidate: JsonMap,
    telemetry: JsonMap,
    prompt: str,
    endpoint_used: str,
    model_hf_id: str,
    gguf_path: str,
    random_seed: int,
) -> JsonDict:
    tokens = [str(token) for token in telemetry.get("tokens") or []]
    token_logprobs = [
        float(value) for value in telemetry.get("token_logprobs") or [] if _number(value) is not None
    ]
    top_logprobs = [
        {str(token): float(value) for token, value in row.items()}
        for row in telemetry.get("top_logprobs") or []
        if isinstance(row, Mapping)
    ]
    candidate_text = str(candidate.get("candidate_text") or "")
    candidate_char_start, candidate_token_start, candidate_token_end = _candidate_token_window(
        tokens, candidate_text
    )
    step_boundaries = _step_boundaries(
        candidate_text=candidate_text,
        tokens=tokens,
        token_logprobs=token_logprobs,
    )
    response_text = str(telemetry.get("completion_text") or "")
    row_id = (
        f"{candidate.get('question_id')}/candidate-{candidate.get('candidate_index')}"
        f"/{_sha256_text(str(candidate.get('candidate_id') or ''))[:12]}"
    )
    return {
        "schema": CACHE_ROW_SCHEMA,
        "row_id": row_id,
        "corpus": MUSR_CORPUS,
        "question_id": str(candidate.get("question_id") or ""),
        "question_index": int(candidate.get("question_index") or 0),
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "candidate_index": int(candidate.get("candidate_index") or 0),
        "source_candidate_id": str(candidate.get("source_candidate_id") or ""),
        "source_schema": str(candidate.get("source_schema") or ""),
        "source_cache_path": str(candidate.get("source_cache_path") or ""),
        "model_hf_id": str(model_hf_id),
        "gguf_path": str(gguf_path),
        "endpoint_used": _endpoint_route(endpoint_used),
        "prompt_hash": _sha256_text(prompt),
        "response_hash": _sha256_text(response_text or _json_dumps(telemetry)),
        "question": str(candidate.get("question") or ""),
        "context_hash": _sha256_text(str(candidate.get("context") or "")),
        "choices": list(candidate.get("choices") or []),
        "gold": str(candidate.get("gold") or ""),
        "candidate_text": candidate_text,
        "candidate_text_hash": _sha256_text(candidate_text),
        "random_seed": int(random_seed),
        "tokens": tokens,
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs,
        "token_count": int(len(tokens)),
        "candidate_token_span": {
            "char_start": int(candidate_char_start),
            "token_start": int(candidate_token_start),
            "token_end": int(candidate_token_end),
        },
        "step_boundaries": step_boundaries,
        "completion_text": response_text,
        "top_logprobs_available": bool(top_logprobs),
    }


def validate_cache_row(row: JsonMap) -> list[str]:
    errors: list[str] = []
    if row.get("schema") != CACHE_ROW_SCHEMA:
        errors.append("schema")
    for field in (
        "row_id",
        "question_id",
        "candidate_id",
        "model_hf_id",
        "gguf_path",
        "prompt_hash",
        "response_hash",
    ):
        if not str(row.get(field) or ""):
            errors.append(field)
    if str(row.get("model_hf_id") or "") not in MANDATED_MODEL_IDS:
        errors.append("model_hf_id")
    for field in ("prompt_hash", "response_hash"):
        if len(str(row.get(field) or "")) != 64:
            errors.append(field)
    if not isinstance(row.get("token_count"), int) or int(row.get("token_count") or 0) <= 0:
        errors.append("token_count")
    token_logprobs = row.get("token_logprobs")
    if not (
        isinstance(token_logprobs, list)
        and bool(token_logprobs)
        and all(_number(value) is not None for value in token_logprobs)
    ):
        errors.append("token_logprobs")
    if not isinstance(row.get("top_logprobs_available"), bool):
        errors.append("top_logprobs_available")
    if row.get("top_logprobs_available") is True and not isinstance(row.get("top_logprobs"), list):
        errors.append("top_logprobs")
    steps = row.get("step_boundaries")
    if not isinstance(steps, list) or not steps:
        errors.append("step_boundaries")
    else:
        for step in steps:
            if not isinstance(step, Mapping):
                errors.append("step_boundaries")
                continue
            if int(step.get("token_end", 0) or 0) < int(step.get("token_start", 0) or 0):
                errors.append("step_boundaries")
    return sorted(set(errors))


def read_complete_rows(path: Path) -> list[JsonDict]:
    rows_by_id: dict[str, JsonDict] = {}
    for row in _read_jsonl(path):
        if not validate_cache_row(row):
            rows_by_id[str(row["row_id"])] = row
    return list(rows_by_id.values())


def default_endpoint_probe(endpoint: str, timeout_s: float = DEFAULT_ENDPOINT_TIMEOUT_S) -> JsonDict:
    route = _endpoint_route(endpoint)
    payload = {
        "prompt": "Exp5086 endpoint probe. Return OK.",
        "n_predict": 4,
        "temperature": 0.0,
        "seed": RANDOM_SEED,
        "n_probs": TOP_LOGPROBS_REQUESTED,
        "top_k": TOP_LOGPROBS_REQUESTED,
    }
    try:
        status, raw = _http_post_json(route, payload, timeout_s)
        parsed = parse_logprob_payload(raw if isinstance(raw, Mapping) else {})
    except Exception as exc:
        return {
            "available": False,
            "endpoint_used": route,
            "detail": _http_error_detail(exc),
            "token_logprob_count": 0,
            "top_logprob_row_count": 0,
        }
    token_count = len(parsed["token_logprobs"])
    top_count = len(parsed["top_logprobs"])
    return {
        "available": bool(200 <= status < 300 and token_count > 0 and top_count > 0),
        "endpoint_used": route,
        "detail": "endpoint returned token logprobs and top-logprob rows"
        if token_count and top_count
        else "endpoint response lacked token logprobs or top-logprob rows",
        "status": status,
        "token_logprob_count": token_count,
        "top_logprob_row_count": top_count,
    }


def default_candidate_scorer(
    *,
    candidate: JsonMap,
    prompt: str,
    endpoint: str,
    seed: int,
) -> JsonDict:
    del candidate
    payload = {
        "prompt": prompt,
        "n_predict": 1,
        "temperature": 0.0,
        "cache_prompt": True,
        "seed": int(seed),
        "n_probs": TOP_LOGPROBS_REQUESTED,
        "top_k": TOP_LOGPROBS_REQUESTED,
        "echo": True,
    }
    status, raw = _http_post_json(_endpoint_route(endpoint), payload, DEFAULT_ENDPOINT_TIMEOUT_S)
    if not 200 <= status < 300:
        raise RuntimeError(f"endpoint returned HTTP status {status}")
    if not isinstance(raw, Mapping):
        raise RuntimeError("endpoint returned non-object payload")
    parsed = parse_logprob_payload(raw)
    if not parsed["token_logprobs"]:
        raise RuntimeError("endpoint scoring response lacked token_logprobs")
    if not parsed["top_logprobs"]:
        raise RuntimeError("endpoint scoring response lacked top_logprobs")
    return parsed


def _extract_gate_config(gate: JsonMap) -> JsonDict:
    sample = gate.get("sample_completion")
    sample_map = sample if isinstance(sample, Mapping) else {}
    endpoint = str(gate.get("endpoint_url") or "")
    if not endpoint and isinstance(gate.get("endpoint_summary"), Mapping):
        endpoint = str(gate["endpoint_summary"].get("selected_endpoint") or "")
    route = str(sample_map.get("route") or _endpoint_route(endpoint))
    selected_hf_id = str(sample_map.get("model_hf_id") or "")
    selected_path = str(sample_map.get("model_path") or "")
    resolved = {}
    model_specs = gate.get("model_specs")
    if isinstance(model_specs, Mapping) and isinstance(model_specs.get("resolved_models"), Mapping):
        resolved = dict(model_specs["resolved_models"])
    if not selected_hf_id or not selected_path:
        for role in ("middle_moe", "flagship_moe", "flagship_dense"):
            row = resolved.get(role)
            if not isinstance(row, Mapping):
                continue
            path = str(row.get("resolved_path") or "")
            hf_id = str(row.get("hf_id") or "")
            if path and hf_id:
                selected_hf_id = selected_hf_id or hf_id
                selected_path = selected_path or path
                break
    return {
        "endpoint_url": endpoint or route.rsplit("/completion", 1)[0],
        "endpoint_route": _endpoint_route(route),
        "selected_model_hf_id": selected_hf_id,
        "selected_model_path": selected_path,
        "resolved_models": resolved,
    }


def build_model_specs(gate: JsonMap) -> JsonDict:
    config = _extract_gate_config(gate)
    resolved_models = config["resolved_models"] if isinstance(config["resolved_models"], Mapping) else {}
    models: list[JsonDict] = []
    for hf_id in MANDATED_MODEL_IDS:
        role = ROLE_BY_HF_ID[hf_id]
        row = resolved_models.get(role)
        row_map = row if isinstance(row, Mapping) else {}
        resolved_path = str(row_map.get("resolved_path") or row_map.get("model_path") or "")
        models.append(
            {
                "role": role,
                "hf_id": hf_id,
                "preferred_quant": str(row_map.get("preferred_quant") or "Q4_K_M"),
                "resolved_path": resolved_path or None,
                "selected_for_endpoint": hf_id == config["selected_model_hf_id"],
            }
        )
    return {
        "mandatory_models": models,
        "selected_model_hf_id": config["selected_model_hf_id"],
        "selected_gguf_path": config["selected_model_path"],
        "endpoint_url": config["endpoint_url"],
        "endpoint_route": config["endpoint_route"],
        "telemetry_request": {
            "n_probs": TOP_LOGPROBS_REQUESTED,
            "top_logprobs_required": True,
            "echo_prompt_for_candidate_tokens": True,
        },
    }


def _disk_precondition(root: Path) -> JsonDict:
    usage = shutil.disk_usage(root)
    return {
        "available": usage.free > 0,
        "resource": "disk_space",
        "path": root.as_posix(),
        "free_bytes": int(usage.free),
        "free_gib": round(usage.free / (1024**3), 3),
        "detail": "disk space available" if usage.free > 0 else "no free disk space",
    }


def build_preconditions(
    *,
    root: Path,
    gate_path: Path,
    gate: JsonMap | None,
    gate_error: str | None,
    candidate_summary: JsonMap,
    endpoint_probe_result: JsonMap | None,
) -> JsonDict:
    config = _extract_gate_config(gate or {})
    artifact_hash = _sha256_file(gate_path)
    gate_ready = bool(gate and gate.get("logprob_endpoint_ready") is True)
    structured_gate_available = bool(
        gate_ready
        and config["endpoint_url"]
        and config["selected_model_path"]
        and config["selected_model_hf_id"] in MANDATED_MODEL_IDS
    )
    probe = dict(endpoint_probe_result or {})
    return {
        "exp5085_artifact": {
            "available": bool(gate is not None and artifact_hash),
            "resource": "exp5085_artifact",
            "path": gate_path.as_posix(),
            "sha256": artifact_hash,
            "honest_verdict": gate.get("honest_verdict") if gate else None,
            "logprob_endpoint_ready": gate.get("logprob_endpoint_ready") if gate else None,
            "detail": gate_error or "Exp5085 artifact loaded",
        },
        "structured_gate": {
            "available": structured_gate_available,
            "resource": "exp5085_structured_gate",
            "endpoint_url": config["endpoint_url"] or None,
            "selected_model_hf_id": config["selected_model_hf_id"] or None,
            "selected_model_path": config["selected_model_path"] or None,
            "detail": "Exp5085 structured gate records endpoint and selected model"
            if structured_gate_available
            else "Exp5085 gate missing readiness, endpoint, or selected mandated model",
        },
        "cache_input": {
            "available": bool(candidate_summary.get("available")),
            "resource": "existing_musr_candidate_cache",
            "path": candidate_summary.get("cache_input_path"),
            "fallback_enrichment_path": candidate_summary.get("fallback_enrichment_path"),
            "candidate_source": candidate_summary.get("candidate_source"),
            "n_questions": int(candidate_summary.get("n_questions") or 0),
            "n_candidates": int(candidate_summary.get("n_candidates") or 0),
            "detail": "existing MuSR candidate strings selected"
            if candidate_summary.get("available")
            else "not enough existing MuSR candidate strings",
        },
        "disk_space": _disk_precondition(root),
        "endpoint_live_probe": {
            "available": bool(probe.get("available")),
            "resource": "exp5085_endpoint_live_probe",
            "endpoint_used": probe.get("endpoint_used") or config["endpoint_route"] or None,
            "token_logprob_count": int(probe.get("token_logprob_count") or 0),
            "top_logprob_row_count": int(probe.get("top_logprob_row_count") or 0),
            "detail": str(probe.get("detail") or "endpoint probe not attempted"),
        },
    }


def _precondition_blocker(preconditions: JsonMap) -> str | None:
    if not preconditions["exp5085_artifact"]["available"]:
        return "blocked_uprm_logprob_cache_retry_exp5085_missing"
    if not preconditions["structured_gate"]["available"]:
        return "blocked_uprm_logprob_cache_retry_exp5085_not_ready"
    if not preconditions["cache_input"]["available"]:
        return "blocked_uprm_logprob_cache_retry_no_candidate_cache"
    if not preconditions["disk_space"]["available"]:
        return "blocked_uprm_logprob_cache_retry_disk_space"
    if not preconditions["endpoint_live_probe"]["available"]:
        return "blocked_uprm_logprob_cache_retry_endpoint_failed"
    return None


def _target_row_ids(candidates: Sequence[JsonMap]) -> set[str]:
    ids: set[str] = set()
    for candidate in candidates:
        ids.add(
            f"{candidate.get('question_id')}/candidate-{candidate.get('candidate_index')}"
            f"/{_sha256_text(str(candidate.get('candidate_id') or ''))[:12]}"
        )
    return ids


def _cache_metrics(cache_path: Path, candidates: Sequence[JsonMap]) -> JsonDict:
    target_ids = _target_row_ids(candidates)
    complete_rows = [row for row in read_complete_rows(cache_path) if str(row.get("row_id")) in target_ids]
    complete_ids = {str(row["row_id"]) for row in complete_rows}
    complete_questions = {str(row.get("question_id") or "") for row in complete_rows}
    n_candidates = len(candidates)
    top_rows = sum(1 for row in complete_rows if row.get("top_logprobs_available") is True)
    token_ready_rows = sum(1 for row in complete_rows if row.get("token_logprobs"))
    step_ready_rows = sum(1 for row in complete_rows if row.get("step_boundaries"))
    return {
        "complete_rows": complete_rows,
        "complete_ids": complete_ids,
        "n_rows_complete": len(complete_rows),
        "n_questions": len(complete_questions),
        "n_candidates": n_candidates,
        "parse_rate": _rate(len(complete_rows), n_candidates),
        "top_logprob_coverage": _rate(top_rows, len(complete_rows)),
        "logprob_cache_ready": bool(n_candidates > 0 and token_ready_rows == n_candidates),
        "step_cache_ready": bool(n_candidates > 0 and step_ready_rows == n_candidates),
    }


def _checksum_basis(artifact: JsonMap) -> str:
    basis = {
        "experiment_id": artifact.get("experiment_id"),
        "honest_verdict": artifact.get("honest_verdict"),
        "model_specs": artifact.get("model_specs"),
        "preconditions_checked": artifact.get("preconditions_checked"),
        "cache_path": artifact.get("cache_path"),
        "n_questions": artifact.get("n_questions"),
        "n_candidates": artifact.get("n_candidates"),
        "n_rows_complete": artifact.get("n_rows_complete"),
        "parse_rate": artifact.get("parse_rate"),
        "top_logprob_coverage": artifact.get("top_logprob_coverage"),
    }
    return _sha256_payload(basis)


def build_artifact(
    *,
    root: Path,
    artifact_path: Path,
    cache_path: Path,
    gate: JsonMap | None,
    preconditions_checked: JsonMap,
    candidates: Sequence[JsonMap],
    started_at: float,
    finished_at: float,
    resume_summary: JsonMap,
    row_errors: Sequence[JsonMap] = (),
    forced_verdict: str | None = None,
) -> JsonDict:
    metrics = _cache_metrics(cache_path, candidates)
    ready = bool(
        metrics["logprob_cache_ready"]
        and metrics["step_cache_ready"]
        and metrics["n_candidates"] > 0
        and metrics["n_rows_complete"] == metrics["n_candidates"]
    )
    if forced_verdict:
        verdict = forced_verdict
    elif ready:
        verdict = f"success_uprm_logprob_cache_retry_ready_n{metrics['n_questions']}"
    else:
        verdict = f"blocked_uprm_logprob_cache_retry_incomplete_n{metrics['n_questions']}"
    endpoint_used = (
        preconditions_checked.get("endpoint_live_probe", {}).get("endpoint_used")
        or build_model_specs(gate or {}).get("endpoint_route")
    )
    inference_substrate = "live_llm_inference" if ready else "precondition_check_only"
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "result_path": artifact_path.as_posix(),
        "cache_row_schema": CACHE_ROW_SCHEMA,
        "honest_verdict": verdict,
        "duration_s": round(max(0.0, float(finished_at) - float(started_at)), 6),
        "inference_substrate": inference_substrate,
        "preconditions_checked": dict(preconditions_checked),
        "model_specs": build_model_specs(gate or {}),
        "logprob_cache_ready": bool(ready and metrics["logprob_cache_ready"]),
        "step_cache_ready": bool(ready and metrics["step_cache_ready"]),
        "cache_path": cache_path.as_posix(),
        "n_questions": int(metrics["n_questions"]),
        "n_candidates": int(metrics["n_candidates"]),
        "n_rows_complete": int(metrics["n_rows_complete"]),
        "parse_rate": float(metrics["parse_rate"]),
        "top_logprob_coverage": float(metrics["top_logprob_coverage"]),
        "endpoint_used": str(endpoint_used or ""),
        "flagged_adversarial": False,
        "resume_summary": dict(resume_summary),
        "row_errors": list(row_errors),
        "field_principles": dict(FIELD_PRINCIPLES),
        "random_seed": RANDOM_SEED,
        "cache_input_summary": dict(preconditions_checked.get("cache_input", {})),
    }
    artifact["reproducibility_checksum"] = _checksum_basis(artifact)
    return artifact


def _load_gate(root: Path, gate_path: Path | None) -> tuple[JsonDict | None, Path, str | None]:
    path = Path(gate_path) if gate_path else root / EXP5085_RELATIVE_PATH
    try:
        return _read_json(path), path, None
    except Exception as exc:
        return None, path, f"{type(exc).__name__}: {exc}"


def artifact_schema_errors(artifact: JsonMap) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(field)
    if artifact.get("schema") != SCHEMA:
        errors.append("schema")
    if artifact.get("spec_refs") != SPEC_REFS:
        errors.append("spec_refs")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if not str(artifact.get("honest_verdict") or "").startswith(("success_", "blocked_")):
        errors.append("honest_verdict")
    for field in ("logprob_cache_ready", "step_cache_ready", "flagged_adversarial"):
        if not isinstance(artifact.get(field), bool):
            errors.append(field)
    for field in ("n_questions", "n_candidates", "n_rows_complete"):
        if not isinstance(artifact.get(field), int) or int(artifact.get(field, -1)) < 0:
            errors.append(field)
    for field in ("duration_s", "parse_rate", "top_logprob_coverage"):
        value = _number(artifact.get(field))
        if value is None or value < 0:
            errors.append(field)
    if not isinstance(artifact.get("preconditions_checked"), Mapping):
        errors.append("preconditions_checked")
    if not isinstance(artifact.get("model_specs"), Mapping):
        errors.append("model_specs")
    else:
        ids = {
            str(row.get("hf_id") or "")
            for row in artifact["model_specs"].get("mandatory_models", [])
            if isinstance(row, Mapping)
        }
        if set(MANDATED_MODEL_IDS) - ids:
            errors.append("model_specs")
    if not str(artifact.get("cache_path") or ""):
        errors.append("cache_path")
    if not str(artifact.get("endpoint_used") or ""):
        errors.append("endpoint_used")
    return sorted(set(errors))


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    cache_path: Path | None = None,
    gate_path: Path | None = None,
    min_questions: int = DEFAULT_LIMIT_QUESTIONS,
    limit_questions: int = DEFAULT_LIMIT_QUESTIONS,
    endpoint_probe: EndpointProbe = default_endpoint_probe,
    candidate_scorer: CandidateScorer = default_candidate_scorer,
    now: Clock = time.time,
    write: bool = True,
) -> JsonDict:
    root = Path(root)
    artifact_path = Path(artifact_path) if artifact_path else root / RESULT_RELATIVE_PATH
    cache_path = Path(cache_path) if cache_path else root / CACHE_RELATIVE_PATH
    started_at = float(now())
    gate, resolved_gate_path, gate_error = _load_gate(root, gate_path)
    candidates, candidate_summary = load_candidate_inputs(
        root=root,
        min_questions=min_questions,
        limit_questions=limit_questions,
    )
    config = _extract_gate_config(gate or {})
    endpoint_probe_result: JsonDict | None = None
    if gate and gate.get("logprob_endpoint_ready") is True and config.get("endpoint_url"):
        endpoint_probe_result = endpoint_probe(str(config["endpoint_url"]), DEFAULT_ENDPOINT_TIMEOUT_S)
    preconditions = build_preconditions(
        root=root,
        gate_path=resolved_gate_path,
        gate=gate,
        gate_error=gate_error,
        candidate_summary=candidate_summary,
        endpoint_probe_result=endpoint_probe_result,
    )
    blocker = _precondition_blocker(preconditions)
    if blocker is not None:
        artifact = build_artifact(
            root=root,
            artifact_path=artifact_path,
            cache_path=cache_path,
            gate=gate,
            preconditions_checked=preconditions,
            candidates=candidates,
            started_at=started_at,
            finished_at=float(now()),
            resume_summary={
                "existing_complete_rows": 0,
                "appended_rows": 0,
                "target_rows": len(candidates),
                "skipped_existing_rows": 0,
            },
            forced_verdict=blocker,
        )
        if write:
            write_json(artifact_path, artifact)
        return artifact

    endpoint_used = str(preconditions["endpoint_live_probe"]["endpoint_used"])
    model_specs = build_model_specs(gate or {})
    model_hf_id = str(model_specs.get("selected_model_hf_id") or "")
    gguf_path = str(model_specs.get("selected_gguf_path") or "")
    complete_ids = _cache_metrics(cache_path, candidates)["complete_ids"]
    existing_before = len(complete_ids)
    appended = 0
    row_errors: list[JsonDict] = []
    for candidate in candidates:
        row_id = (
            f"{candidate.get('question_id')}/candidate-{candidate.get('candidate_index')}"
            f"/{_sha256_text(str(candidate.get('candidate_id') or ''))[:12]}"
        )
        if row_id in complete_ids:
            continue
        seed = RANDOM_SEED + int(candidate.get("question_index") or 0) * 1000 + int(
            candidate.get("candidate_index") or 0
        )
        prompt = build_scoring_prompt(candidate)
        try:
            telemetry = candidate_scorer(
                candidate=candidate,
                prompt=prompt,
                endpoint=endpoint_used,
                seed=seed,
            )
            row = build_cache_row(
                candidate=candidate,
                telemetry=telemetry,
                prompt=prompt,
                endpoint_used=endpoint_used,
                model_hf_id=model_hf_id,
                gguf_path=gguf_path,
                random_seed=seed,
            )
            errors = validate_cache_row(row)
            if errors:
                raise ValueError(f"cache row schema errors: {errors}")
            append_jsonl_row(cache_path, row)
            complete_ids.add(row_id)
            appended += 1
        except Exception as exc:
            row_errors.append(
                {
                    "question_id": str(candidate.get("question_id") or ""),
                    "candidate_id": str(candidate.get("candidate_id") or ""),
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

    artifact = build_artifact(
        root=root,
        artifact_path=artifact_path,
        cache_path=cache_path,
        gate=gate,
        preconditions_checked=preconditions,
        candidates=candidates,
        started_at=started_at,
        finished_at=float(now()),
        resume_summary={
            "existing_complete_rows": existing_before,
            "appended_rows": appended,
            "target_rows": len(candidates),
            "skipped_existing_rows": existing_before,
        },
        row_errors=row_errors,
    )
    if write:
        write_json(artifact_path, artifact)
    return artifact


def main() -> int:  # pragma: no cover - exercised by direct operator command
    artifact = run()
    errors = artifact_schema_errors(artifact)
    print(f"{REPO_ROOT / RESULT_RELATIVE_PATH}: {artifact.get('honest_verdict')}")
    if errors:
        print(f"schema_errors={errors}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
