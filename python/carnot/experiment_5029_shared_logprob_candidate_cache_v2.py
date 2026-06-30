"""Exp 5029: rescore existing MuSR candidates with logprob telemetry.

Spec refs: REQ-VERIFY-5029, SCENARIO-VERIFY-5029.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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

from carnot import moat_benchmark_harness as harness  # noqa: E402


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
Clock = Callable[[], float]
CandidateLoader = Callable[[int], list[JsonDict]]
CandidateScorer = Callable[..., JsonDict]

EXPERIMENT_ID = 5029
EXPERIMENT_NAME = "experiment_5029_shared_logprob_candidate_cache_v2"
RESULT_RELATIVE_PATH = "results/experiment_5029_shared_logprob_candidate_cache_v2.json"
CACHE_RELATIVE_PATH = "results/experiment_5029_shared_logprob_candidate_cache_v2_musr.jsonl"
CACHE_ROW_SCHEMA = "carnot.shared_logprob_candidate_cache_v2.candidate_row.v1"
ARTIFACT_SCHEMA = "carnot.experiment_5029_shared_logprob_candidate_cache_v2.v1"
MODEL_HF_ID = "unsloth/gemma-4-12B-it-GGUF"
MODEL_NAME = "gemma-4-12B-it-GGUF"
CORPUS = harness.MUSR_CORPUS_NAME
SPEC_REFS = ["REQ-VERIFY-5029", "SCENARIO-VERIFY-5029"]
RANDOM_SEED = harness.DEFAULT_RANDOM_SEED
DEFAULT_LIMIT = 200
DEFAULT_SERVER_PORT = 8919
LOGPROBS_TOP_K = 200
DEFAULT_TIMEOUT_S = 120

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success_logprob_cache_rescored_musr_n<N>; a 0-row run "
            "is blocked_cache_zero_rows (the .462 failure mode -- not a silent pass)."
        )
    },
    "candidate_cache_built": {
        "principle": "true iff n_cached_rows>0 over >=200 MuSR questions (the field D2 gates on)."
    },
    "n_cached_rows": {
        "principle": (
            ">0 REQUIRED -- 0 was the .462 failure signature; the incremental-write fix "
            "guarantees rows persist as they are scored."
        )
    },
    "cache_jsonl_path": {
        "principle": "the resumable JSONL path D2 consumes for uPRM scoring."
    },
    "rescored_not_regenerated": {
        "principle": (
            "true -- the candidates are the EXISTING cached MuSR strings re-scored for "
            "logprobs, NOT freshly generated (the anti-0-rows fix)."
        )
    },
    "n_questions": {"principle": ">=200 (sample-size rigor)."},
    "candidates_per_question": {
        "principle": "the number of cached candidates re-scored per question."
    },
    "has_per_token_logprobs": {
        "principle": "true -- the +/- marker telemetry uPRM's first-error score needs."
    },
    "corpora_cached": {
        "principle": "MuSR (required) + any best-effort 2nd corpus (GPQA/MMLU-Pro-hard) for D4."
    },
    "model_specs": {
        "principle": (
            "gemma-4-12B-it-GGUF on the GPU-0 CUDA llama-server -- the scoring methodology stamp."
        )
    },
    "inference_substrate": {
        "principle": "live_llm_inference (live re-scoring with logprobs; >=60s floor)."
    },
    "random_seed": {"principle": "determinism for the scoring order."},
    "reproducibility_checksum": {
        "principle": (
            "content hash of (server, corpus, candidate set) so a replication catches drift."
        )
    },
    "preconditions_checked": {
        "principle": (
            "records cached-candidates/GGUF/logprob-server checks; a missing resource "
            "emits blocked_, never a fabricated cache."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "result_path",
    "duration_s",
    "field_principles",
)


@dataclass(frozen=True)
class PreconditionCheck:
    """One checked input that controls whether Exp 5029 may claim a cache."""

    resource: str
    available: bool
    detail: str
    path: str | None = None

    def as_dict(self) -> JsonDict:
        payload: JsonDict = {
            "resource": self.resource,
            "available": bool(self.available),
            "detail": self.detail,
        }
        if self.path is not None:
            payload["path"] = self.path
        return payload


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl_row(path: Path, row: JsonMap) -> None:
    """Append one candidate row and fsync immediately for capped-run survival."""

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
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _top_logprob_row(raw: Any) -> dict[str, float]:
    row: dict[str, float] = {}
    if isinstance(raw, Mapping):
        for token, logprob in raw.items():
            value = _number(logprob)
            if value is not None:
                row[str(token)] = value
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for item in raw:
            if isinstance(item, Mapping) and "token" in item:
                value = _number(item.get("logprob"))
                if value is not None:
                    row[str(item["token"])] = value
    return row


def _has_marker_pair(marker_row: Any) -> bool:
    if not isinstance(marker_row, Mapping):
        return False
    markers = {str(token).strip() for token in marker_row if _number(marker_row[token]) is not None}
    return {"-", "+"}.issubset(markers)


def _candidate_key(question_id: str, candidate_index: int | str) -> str:
    return f"{question_id}/cached-{int(candidate_index)}"


def _row_candidate_key(row: JsonMap) -> str:
    return _candidate_key(str(row.get("question_id") or ""), int(row.get("candidate_index") or 0))


def validate_candidate_row(row: JsonMap) -> list[str]:
    errors: list[str] = []
    if row.get("schema") != CACHE_ROW_SCHEMA:
        errors.append("schema")
    if row.get("corpus") != CORPUS:
        errors.append("corpus")
    if not str(row.get("question_id") or ""):
        errors.append("question_id")
    if not str(row.get("candidate_id") or ""):
        errors.append("candidate_id")
    if not str(row.get("answer") or "").strip():
        errors.append("answer")
    if row.get("rescored_not_regenerated") is not True:
        errors.append("rescored_not_regenerated")
    token_logprobs = row.get("token_logprobs")
    if not (
        isinstance(token_logprobs, list)
        and bool(token_logprobs)
        and all(_number(value) is not None for value in token_logprobs)
    ):
        errors.append("token_logprobs")
    marker_rows = row.get("uprm_marker_logprobs")
    if not (
        isinstance(marker_rows, list)
        and bool(marker_rows)
        and all(_has_marker_pair(marker_row) for marker_row in marker_rows)
    ):
        errors.append("uprm_marker_logprobs")
    return sorted(set(errors))


def read_complete_candidate_rows(path: Path) -> list[JsonDict]:
    complete_by_key: dict[str, JsonDict] = {}
    for row in _read_jsonl(path):
        if not validate_candidate_row(row):
            complete_by_key[_row_candidate_key(row)] = row
    return list(complete_by_key.values())


def cache_summary(path: Path, *, min_questions: int = DEFAULT_LIMIT) -> JsonDict:
    del min_questions
    rows = read_complete_candidate_rows(path)
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row.get("question_id"))] = counts.get(str(row.get("question_id")), 0) + 1
    min_candidates = min(counts.values(), default=0)
    max_candidates = max(counts.values(), default=0)
    return {
        "n_cached_rows": len(rows),
        "n_questions": len(counts),
        "min_candidates_per_question": min_candidates,
        "max_candidates_per_question": max_candidates,
        "candidates_per_question": min_candidates,
        "has_per_token_logprobs": bool(rows)
        and all(not validate_candidate_row(row) for row in rows),
        "corpora_cached": sorted({str(row.get("corpus")) for row in rows}),
        "candidate_count_summary": {
            "min": min_candidates,
            "max": max_candidates,
            "total": len(rows),
        },
    }


def parse_logprob_payload(payload: JsonMap) -> JsonDict:
    """Extract tokens, chosen-token logprobs, and top-logprob rows."""

    text = str(payload.get("content") or "")
    probabilities = payload.get("completion_probabilities")
    if isinstance(probabilities, Sequence) and not isinstance(probabilities, (str, bytes)):
        tokens: list[str] = []
        token_logprobs: list[float] = []
        top_logprobs: list[dict[str, float]] = []
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
        return {
            "completion_text": text,
            "tokens": tokens,
            "token_logprobs": token_logprobs,
            "top_logprobs": top_logprobs,
        }

    choices = payload.get("choices")
    if isinstance(choices, Sequence) and choices and isinstance(choices[0], Mapping):
        choice = choices[0]
        text = str(choice.get("text") or text)
        logprobs = choice.get("logprobs")
        if isinstance(logprobs, Mapping):
            token_logprobs = [
                value
                for raw in logprobs.get("token_logprobs") or []
                if (value := _number(raw)) is not None
            ]
            top_rows = [_top_logprob_row(row) for row in logprobs.get("top_logprobs") or []]
            return {
                "completion_text": text,
                "tokens": [str(token) for token in logprobs.get("tokens") or []],
                "token_logprobs": token_logprobs,
                "top_logprobs": [row for row in top_rows if row],
            }
    return {"completion_text": text, "tokens": [], "token_logprobs": [], "top_logprobs": []}


def _first_marker_top_logprobs(top_logprobs: Sequence[Mapping[str, float]]) -> dict[str, float] | None:
    for row in reversed(list(top_logprobs)):
        if _has_marker_pair(row):
            return {str(token): float(value) for token, value in row.items()}
    return None


def build_scoring_prompt(question: JsonMap, candidate: JsonMap) -> str:
    context = str(question.get("context") or "")[:3000]
    choices = list(question.get("choices") or [])
    return (
        "You are scoring an existing cached answer. Do not write a new answer. "
        "Reply with exactly '+' if the candidate answer is plausible for the "
        "question so far, or '-' if it is not.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question.get('question', '')}\n"
        f"CHOICES: {choices}\n\n"
        f"CANDIDATE ANSWER:\n{candidate.get('answer', '')}\n\n"
        "MARKER:"
    )


def llama_server_echo_completion(  # pragma: no cover - live HTTP boundary
    prompt: str,
    *,
    port: int,
    seed: int,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> JsonDict:
    import urllib.request

    payload: JsonDict = {
        "prompt": prompt,
        "n_predict": 1,
        "temperature": 0.0,
        "cache_prompt": True,
        "seed": int(seed),
        "logprobs": LOGPROBS_TOP_K,
        "echo": True,
        "stop": ["\n"],
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        loaded = json.load(response)
    if not isinstance(loaded, dict):
        raise RuntimeError("llama-server returned non-object payload")
    return loaded


def live_candidate_scorer(  # pragma: no cover - live HTTP boundary
    *,
    question: JsonMap,
    candidate: JsonMap,
    seed: int,
    server_port: int,
) -> JsonDict:
    payload = llama_server_echo_completion(
        build_scoring_prompt(question, candidate),
        port=server_port,
        seed=seed,
    )
    parsed = parse_logprob_payload(payload)
    marker_row = _first_marker_top_logprobs(parsed["top_logprobs"])
    if not parsed["token_logprobs"]:
        raise RuntimeError("scoring completion lacked token_logprobs")
    if marker_row is None:
        raise RuntimeError("scoring completion lacked '+'/'-' marker top_logprobs")
    return {**parsed, "marker_top_logprobs": marker_row}


def build_scored_candidate_row(
    *,
    question: JsonMap,
    candidate: JsonMap,
    telemetry: JsonMap,
    random_seed: int,
    server_port: int,
) -> JsonDict:
    question_id = str(question.get("question_id") or f"q{int(question.get('question_index', 0)):04d}")
    candidate_index = int(candidate.get("candidate_index", candidate.get("cache_index", 0)))
    token_logprobs = [
        float(value) for value in telemetry.get("token_logprobs", []) if _number(value) is not None
    ]
    marker_row = telemetry.get("marker_top_logprobs")
    if not _has_marker_pair(marker_row):
        marker_row = _first_marker_top_logprobs(
            [
                row
                for row in telemetry.get("top_logprobs", [])
                if isinstance(row, Mapping)
            ]
        )
    top_logprobs = [
        {str(token): float(value) for token, value in row.items()}
        for row in telemetry.get("top_logprobs", [])
        if isinstance(row, Mapping)
    ]
    mean_logprob = sum(token_logprobs) / len(token_logprobs) if token_logprobs else None
    return {
        "schema": CACHE_ROW_SCHEMA,
        "corpus": CORPUS,
        "question_id": question_id,
        "question_index": int(question.get("question_index", 0)),
        "question": str(question.get("question") or ""),
        "context": str(question.get("context") or ""),
        "choices": list(question.get("choices") or []),
        "gold": str(question.get("gold") or ""),
        "candidate_id": _candidate_key(question_id, candidate_index),
        "candidate_index": candidate_index,
        "answer": str(candidate.get("answer") or ""),
        "source_checkpoint_path": str(question.get("checkpoint_path") or ""),
        "source": "distributional_energy_verifier_musr_checkpoints",
        "rescored_not_regenerated": True,
        "scoring_model": MODEL_NAME,
        "scoring_server_url": f"http://127.0.0.1:{server_port}/completion",
        "random_seed": int(random_seed),
        "completion_text": str(telemetry.get("completion_text") or ""),
        "tokens": [str(token) for token in telemetry.get("tokens", [])],
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs,
        "mean_logprob": round(float(mean_logprob), 12) if mean_logprob is not None else None,
        "uprm_marker_logprobs": [dict(marker_row)] if isinstance(marker_row, Mapping) else [],
    }


def _load_checkpoint_json(path: Path) -> JsonDict:
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"checkpoint is not an object: {path}")
    return loaded


def _question_from_checkpoint(
    *,
    checkpoint_path: Path,
    question_index: int,
    corpus_row: JsonMap | None = None,
) -> JsonDict:
    checkpoint = _load_checkpoint_json(checkpoint_path)
    answers = checkpoint.get("answers")
    if not isinstance(answers, list):
        raise ValueError(f"checkpoint lacks answers: {checkpoint_path}")
    candidates = [
        {"candidate_index": index, "answer": str(answer)}
        for index, answer in enumerate(answers)
        if answer is not None and str(answer).strip()
    ]
    if not candidates:
        raise ValueError(f"checkpoint has no non-empty answers: {checkpoint_path}")
    row = dict(corpus_row or {})
    return {
        "question_id": str(row.get("row_id") or f"q{question_index:04d}"),
        "question_index": int(question_index),
        "corpus": CORPUS,
        "question": str(row.get("question") or ""),
        "context": str(row.get("context") or ""),
        "choices": list(row.get("choices") or []),
        "gold": str(row.get("gold") or checkpoint.get("gold") or ""),
        "checkpoint_path": checkpoint_path.as_posix(),
        "candidates": candidates,
    }


def load_cached_musr_candidate_questions(
    *,
    limit: int = DEFAULT_LIMIT,
    checkpoint_dir: Path = harness.DEFAULT_MUSR_CHECKPOINT_DIR,
) -> list[JsonDict]:
    """Load existing checkpoint answer strings, using corpus text when available."""

    paths = sorted(checkpoint_dir.glob("q*.json"))[:limit]
    corpus_rows: list[JsonDict] = []
    try:
        corpus_rows = harness.load_musr_murder_mysteries(limit=len(paths))
    except Exception:
        corpus_rows = []
    questions: list[JsonDict] = []
    for index, path in enumerate(paths):
        corpus_row = corpus_rows[index] if index < len(corpus_rows) else None
        questions.append(
            _question_from_checkpoint(
                checkpoint_path=path,
                question_index=index,
                corpus_row=corpus_row,
            )
        )
    return questions


def default_candidate_loader(limit: int) -> list[JsonDict]:  # pragma: no cover - host cache boundary
    return load_cached_musr_candidate_questions(limit=limit)


def _resolve_gemma_gguf() -> str | None:  # pragma: no cover - host cache probe
    from carnot.inference.sota_models import resolve_cached_gguf

    return resolve_cached_gguf(MODEL_HF_ID, preferred_quant="Q4_K_M")


def default_server_probe(  # pragma: no cover - live HTTP boundary
    port: int,
    timeout_s: int = 30,
) -> PreconditionCheck:
    try:
        payload = llama_server_echo_completion(
            "Candidate answer: yes\n\nMARKER:",
            port=port,
            seed=RANDOM_SEED,
            timeout_s=timeout_s,
        )
        parsed = parse_logprob_payload(payload)
        ok = bool(parsed["token_logprobs"]) and _first_marker_top_logprobs(
            parsed["top_logprobs"]
        ) is not None
    except Exception as exc:
        return PreconditionCheck("llama_server_logprobs", False, f"{type(exc).__name__}: {exc}")
    return PreconditionCheck(
        "llama_server_logprobs",
        ok,
        "server returns completion_probabilities with marker top_logprobs"
        if ok
        else "server response lacked token_logprobs or +/- marker top_logprobs",
        f"http://127.0.0.1:{port}/completion",
    )


def _candidate_set_checksum(questions: Sequence[JsonMap]) -> str:
    payload = [
        {
            "question_id": str(question.get("question_id") or ""),
            "gold": str(question.get("gold") or ""),
            "checkpoint_path": str(question.get("checkpoint_path") or ""),
            "candidates": [
                {
                    "candidate_index": int(candidate.get("candidate_index", 0)),
                    "answer": str(candidate.get("answer") or ""),
                }
                for candidate in question.get("candidates", [])
                if isinstance(candidate, Mapping)
            ],
        }
        for question in questions
    ]
    return "sha256:" + hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def reproducibility_checksum(
    *,
    server_url: str,
    candidate_set_sha256: str,
    corpora: Sequence[str],
) -> str:
    payload = {
        "server": server_url,
        "corpora": list(corpora),
        "candidate_set": candidate_set_sha256,
        "schema": CACHE_ROW_SCHEMA,
    }
    return "sha256:" + hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def discover_cached_second_corpora(root: Path, *, min_questions: int = DEFAULT_LIMIT) -> list[str]:
    candidates = {
        "GPQA": (
            root / "results" / "distributional_energy_verifier_gpqa_checkpoints",
            root / "results" / "gpqa_candidate_checkpoints",
        ),
        "MMLU-Pro-hard": (
            root / "results" / "distributional_energy_verifier_mmlu_pro_hard_checkpoints",
            root / "results" / "mmlu_pro_hard_candidate_checkpoints",
        ),
    }
    found: list[str] = []
    for corpus, paths in candidates.items():
        if any(path.is_dir() and len(list(path.glob("q*.json"))) >= min_questions for path in paths):
            found.append(corpus)
    return found


def check_preconditions(
    *,
    root: Path,
    gguf_resolver: Callable[[], str | None],
    server_probe: Callable[[int], PreconditionCheck],
    candidate_loader: CandidateLoader,
    min_questions: int,
    server_port: int,
) -> tuple[list[PreconditionCheck], Path | None, list[JsonDict]]:
    raw_gguf = gguf_resolver()
    gguf_path = Path(raw_gguf) if raw_gguf else None
    gguf_ok = bool(gguf_path and gguf_path.exists() and gguf_path.is_file())
    checks = [
        PreconditionCheck(
            "gemma_gguf_cache",
            gguf_ok,
            f"{MODEL_HF_ID} resolved" if gguf_ok else f"{MODEL_HF_ID} not resolved as a GGUF",
            gguf_path.as_posix() if gguf_path else None,
        ),
        server_probe(server_port),
    ]
    try:
        questions = candidate_loader(min_questions)
        valid_question_count = sum(1 for row in questions if row.get("candidates"))
        candidate_count = sum(len(row.get("candidates", [])) for row in questions)
        detail = (
            f"{valid_question_count} cached MuSR question(s), {candidate_count} candidate(s), "
            f"required >= {min_questions} questions"
        )
    except Exception as exc:
        questions = []
        valid_question_count = 0
        detail = f"{type(exc).__name__}: {exc}"
    checks.append(
        PreconditionCheck(
            "musr_candidate_checkpoints",
            valid_question_count >= min_questions,
            detail,
            (root / "results" / "distributional_energy_verifier_musr_checkpoints").as_posix(),
        )
    )
    return checks, gguf_path if gguf_ok else None, questions


def _first_missing(checks: Sequence[PreconditionCheck]) -> str | None:
    for check in checks:
        if not check.available:
            return check.resource
    return None


def build_artifact(
    *,
    honest_verdict: str,
    root: Path,
    artifact_path: Path,
    cache_path: Path,
    preconditions_checked: Sequence[JsonMap],
    gguf_path: Path | None,
    candidate_set_sha256: str,
    min_questions: int,
    started_at: float,
    finished_at: float,
    question_errors: Sequence[JsonMap] = (),
) -> JsonDict:
    summary = cache_summary(cache_path, min_questions=min_questions)
    second_corpora = discover_cached_second_corpora(root, min_questions=min_questions)
    corpora_cached = list(summary["corpora_cached"])
    for corpus in second_corpora:
        if corpus not in corpora_cached:
            corpora_cached.append(corpus)
    built = bool(
        summary["n_cached_rows"] > 0
        and summary["n_questions"] >= min_questions
        and summary["has_per_token_logprobs"]
        and CORPUS in summary["corpora_cached"]
    )
    if built:
        honest_verdict = f"success_logprob_cache_rescored_musr_n{summary['n_questions']}"
    elif honest_verdict == "success_pending_cache_summary" and summary["n_cached_rows"] == 0:
        honest_verdict = "blocked_cache_zero_rows"
    elif honest_verdict == "success_pending_cache_summary":
        honest_verdict = f"blocked_incomplete_musr_n{summary['n_questions']}"
    inference_substrate = (
        "live_llm_inference"
        if built or int(summary["n_cached_rows"]) > 0
        else "precondition_check_only"
    )
    preconditions = list(preconditions_checked)
    if question_errors:
        preconditions.append(
            {
                "resource": "question_scoring_errors",
                "available": False,
                "detail": f"{len(question_errors)} question(s) had scoring errors",
            }
        )
    return {
        "schema": ARTIFACT_SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "result_path": artifact_path.as_posix(),
        "honest_verdict": honest_verdict,
        "candidate_cache_built": built,
        "n_cached_rows": int(summary["n_cached_rows"]),
        "cache_jsonl_path": cache_path.as_posix(),
        "rescored_not_regenerated": True,
        "n_questions": int(summary["n_questions"] if built else summary["n_questions"]),
        "candidates_per_question": int(summary["candidates_per_question"]),
        "candidate_count_summary": dict(summary["candidate_count_summary"]),
        "has_per_token_logprobs": bool(summary["has_per_token_logprobs"] if built else False),
        "corpora_cached": corpora_cached if built else list(summary["corpora_cached"]),
        "model_specs": {
            "generator_model": MODEL_NAME,
            "generator_hf_id": MODEL_HF_ID,
            "gguf_path": gguf_path.as_posix() if gguf_path else None,
            "cuda_gpu": 0,
            "server_url": f"http://127.0.0.1:{DEFAULT_SERVER_PORT}/completion",
            "scoring_mode": "echo_logprobs_rescore_existing_candidates",
            "logprobs_requested": LOGPROBS_TOP_K,
            "requires_completion_probabilities": True,
            "requires_top_logprobs_for_markers": True,
        },
        "inference_substrate": inference_substrate,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": reproducibility_checksum(
            server_url=f"http://127.0.0.1:{DEFAULT_SERVER_PORT}/completion",
            candidate_set_sha256=candidate_set_sha256,
            corpora=corpora_cached,
        ),
        "preconditions_checked": preconditions,
        "preconditions_checked_summary": {
            "cached_candidates": any(
                check.get("resource") == "musr_candidate_checkpoints" and check.get("available")
                for check in preconditions_checked
            ),
            "gguf": gguf_path is not None,
            "logprob_server": any(
                check.get("resource") == "llama_server_logprobs" and check.get("available")
                for check in preconditions_checked
            ),
        },
        "question_errors": list(question_errors),
        "duration_s": round(float(finished_at) - float(started_at), 6),
        "field_principles": FIELD_PRINCIPLES,
    }


def artifact_schema_errors(artifact: JsonMap) -> list[str]:
    errors = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if artifact.get("spec_refs") != SPEC_REFS:
        errors.append("spec_refs")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    for field in ("candidate_cache_built", "has_per_token_logprobs", "rescored_not_regenerated"):
        if not isinstance(artifact.get(field), bool):
            errors.append(field)
    for field in ("n_questions", "candidates_per_question", "n_cached_rows"):
        if not isinstance(artifact.get(field), int):
            errors.append(field)
    if not isinstance(artifact.get("preconditions_checked"), list):
        errors.append("preconditions_checked")
    if not str(artifact.get("honest_verdict", "")).startswith(("blocked_", "success_")):
        errors.append("honest_verdict")
    n_cached_rows = artifact.get("n_cached_rows", 0)
    if (
        artifact.get("candidate_cache_built") is True
        and isinstance(n_cached_rows, int)
        and n_cached_rows <= 0
    ):
        errors.append("n_cached_rows")
    if artifact.get("candidate_cache_built") and artifact.get("rescored_not_regenerated") is not True:
        errors.append("rescored_not_regenerated")
    return sorted(set(errors))


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    cache_path: Path | None = None,
    gguf_resolver: Callable[[], str | None] = _resolve_gemma_gguf,
    server_probe: Callable[[int], PreconditionCheck] = default_server_probe,
    candidate_loader: CandidateLoader = default_candidate_loader,
    candidate_scorer: CandidateScorer = live_candidate_scorer,
    min_questions: int = DEFAULT_LIMIT,
    random_seed: int = RANDOM_SEED,
    server_port: int = DEFAULT_SERVER_PORT,
    now: Clock = time.time,
) -> JsonDict:
    root = Path(root)
    artifact_path = Path(artifact_path) if artifact_path else root / RESULT_RELATIVE_PATH
    cache_path = Path(cache_path) if cache_path else root / CACHE_RELATIVE_PATH
    started_at = float(now())
    checks, gguf_path, questions = check_preconditions(
        root=root,
        gguf_resolver=gguf_resolver,
        server_probe=server_probe,
        candidate_loader=candidate_loader,
        min_questions=min_questions,
        server_port=server_port,
    )
    preconditions = [check.as_dict() for check in checks]
    candidate_set_sha256 = _candidate_set_checksum(questions)
    missing = _first_missing(checks)
    if missing is not None:
        artifact = build_artifact(
            honest_verdict=f"blocked_{missing}",
            root=root,
            artifact_path=artifact_path,
            cache_path=cache_path,
            preconditions_checked=preconditions,
            gguf_path=gguf_path,
            candidate_set_sha256=candidate_set_sha256,
            min_questions=min_questions,
            started_at=started_at,
            finished_at=float(now()),
        )
        write_json(artifact_path, artifact)
        return artifact

    done_keys = {_row_candidate_key(row) for row in read_complete_candidate_rows(cache_path)}
    question_errors: list[JsonDict] = []
    for question in questions[:min_questions]:
        try:
            question_id = str(question.get("question_id") or "")
            for candidate in question.get("candidates", []):
                if not isinstance(candidate, Mapping):
                    continue
                candidate_index = int(candidate.get("candidate_index", 0))
                key = _candidate_key(question_id, candidate_index)
                if key in done_keys:
                    continue
                seed = random_seed + int(question.get("question_index", 0)) * 1000 + candidate_index
                telemetry = candidate_scorer(
                    question=question,
                    candidate=candidate,
                    seed=seed,
                    server_port=server_port,
                )
                row = build_scored_candidate_row(
                    question=question,
                    candidate=candidate,
                    telemetry=telemetry,
                    random_seed=seed,
                    server_port=server_port,
                )
                errors = validate_candidate_row(row)
                if errors:
                    raise ValueError(f"scored candidate row is malformed: {errors}")
                append_jsonl_row(cache_path, row)
                done_keys.add(key)
        except Exception as exc:
            question_errors.append(
                {
                    "question_id": str(question.get("question_id") or ""),
                    "question_index": int(question.get("question_index", 0)),
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

    artifact = build_artifact(
        honest_verdict="success_pending_cache_summary",
        root=root,
        artifact_path=artifact_path,
        cache_path=cache_path,
        preconditions_checked=preconditions,
        gguf_path=gguf_path,
        candidate_set_sha256=candidate_set_sha256,
        min_questions=min_questions,
        started_at=started_at,
        finished_at=float(now()),
        question_errors=question_errors,
    )
    write_json(artifact_path, artifact)
    return artifact


def main() -> int:  # pragma: no cover - exercised by requested entrypoint
    artifact = run()
    errors = artifact_schema_errors(artifact)
    path = REPO_ROOT / RESULT_RELATIVE_PATH
    print(f"{path}: {artifact.get('honest_verdict')}")
    if errors:
        print(f"schema_errors={errors}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
