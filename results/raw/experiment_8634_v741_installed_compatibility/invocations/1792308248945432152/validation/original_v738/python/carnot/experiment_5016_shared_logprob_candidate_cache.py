"""Exp 5016: build the shared MuSR candidate cache with logprob telemetry.

Spec refs: REQ-VERIFY-5016, SCENARIO-VERIFY-5016.
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

from carnot import experiment_5004_uprm_replication as uprm  # noqa: E402
from carnot import moat_benchmark_harness as harness  # noqa: E402
from carnot.moat_benchmark_harness import GenerationConfig  # noqa: E402


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
Clock = Callable[[], float]
CandidateRowBuilder = Callable[..., JsonDict]

EXPERIMENT_ID = 5016
EXPERIMENT_NAME = "experiment_5016_shared_logprob_candidate_cache"
RESULT_RELATIVE_PATH = "results/experiment_5016_shared_logprob_candidate_cache.json"
CACHE_RELATIVE_PATH = "results/experiment_5016_shared_logprob_candidate_cache_musr.jsonl"
CACHE_ROW_SCHEMA = "carnot.shared_logprob_candidate_cache.row.v1"
ARTIFACT_SCHEMA = "carnot.experiment_5016_shared_logprob_candidate_cache.v1"
MODEL_HF_ID = "unsloth/gemma-4-12B-it-GGUF"
MODEL_NAME = "gemma-4-12B-it-GGUF"
CORPUS = harness.MUSR_CORPUS_NAME
SPEC_REFS = ["REQ-VERIFY-5016", "SCENARIO-VERIFY-5016"]
RANDOM_SEED = harness.DEFAULT_RANDOM_SEED
DEFAULT_K = 5
DEFAULT_LIMIT = 200
DEFAULT_SERVER_PORT = 8919
GENERATION_LOGPROBS_TOP_K = 20
MARKER_LOGPROBS_TOP_K = 200

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal prefix; success_logprob_candidate_cache_built_musr_n<N>_k<K>."
    },
    "candidate_cache_built": {
        "principle": (
            "true iff >=200 MuSR questions x K>=5 candidates-with-logprobs were cached "
            "(the field D1/D2/D3 gate/precondition on)."
        )
    },
    "cache_jsonl_path": {
        "principle": (
            "the resumable JSONL path (results/...candidates...jsonl) D2 consumes for "
            "uPRM scoring + D1/D3/cascade reuse."
        )
    },
    "n_questions": {"principle": ">=200 (sample-size rigor)."},
    "candidates_per_question": {
        "principle": (
            "K>=5 -- enough for a genuine K-way SC vote AND a non-degenerate oracle@K."
        )
    },
    "has_per_token_logprobs": {
        "principle": (
            "true -- the +/- marker telemetry uPRM's first-error score needs "
            "(the .461 D2 blocker)."
        )
    },
    "corpora_cached": {
        "principle": "MuSR (required) + any best-effort 2nd corpus (GPQA/MMLU-Pro-hard) for D4."
    },
    "model_specs": {
        "principle": (
            "gemma-4-12B-it-GGUF on the GPU-0 CUDA llama-server -- the generation "
            "methodology stamp."
        )
    },
    "inference_substrate": {
        "principle": "live_llm_inference (live generation with logprobs; >=60s floor)."
    },
    "random_seed": {"principle": "determinism for sampling (vary by question index)."},
    "reproducibility_checksum": {
        "principle": (
            "content hash of (generator, corpus, K, seed) so a replication catches drift."
        )
    },
    "preconditions_checked": {
        "principle": (
            "records GGUF-cached/logprob-server/corpus checks; a missing resource emits "
            "blocked_, never a fabricated cache."
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
    """One checked input that decides whether cache generation may make claims."""

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


def _read_jsonl(path: Path) -> list[JsonDict]:
    if not path.exists():
        return []
    rows: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        loaded = json.loads(line)
        if isinstance(loaded, dict):
            rows.append(loaded)
    return rows


def append_jsonl_atomic(path: Path, row: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    descriptor = os.open(path, flags, 0o644)
    try:
        os.write(descriptor, line)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _has_marker_pair(marker_row: Any) -> bool:
    if not isinstance(marker_row, Mapping):
        return False
    markers = {str(token).strip() for token in marker_row if _finite_number(marker_row[token])}
    return {"-", "+"}.issubset(markers)


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def _candidate_has_logprobs(candidate: JsonMap) -> bool:
    token_logprobs = candidate.get("token_logprobs")
    marker_rows = candidate.get("uprm_marker_logprobs")
    return (
        isinstance(token_logprobs, list)
        and bool(token_logprobs)
        and all(_finite_number(value) for value in token_logprobs)
        and isinstance(marker_rows, list)
        and bool(marker_rows)
        and all(_has_marker_pair(row) for row in marker_rows)
    )


def _cache_key(row: JsonMap) -> str:
    return str(row.get("row_id") or row.get("q") or row.get("question") or "")


def validate_cache_row(row: JsonMap, *, k_candidates: int) -> list[str]:
    errors: list[str] = []
    if row.get("schema") != CACHE_ROW_SCHEMA:
        errors.append("schema")
    if row.get("corpus") != CORPUS:
        errors.append("corpus")
    if not _cache_key(row):
        errors.append("row_id")
    candidates = row.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < k_candidates:
        errors.append("candidates")
        candidates = candidates if isinstance(candidates, list) else []
    for index, candidate in enumerate(candidates[:k_candidates]):
        if not isinstance(candidate, Mapping):
            errors.append(f"candidate_{index}")
            continue
        if not isinstance(candidate.get("token_logprobs"), list) or not candidate.get(
            "token_logprobs"
        ):
            errors.append(f"candidate_{index}_token_logprobs")
        if not _candidate_has_logprobs(candidate):
            errors.append(f"candidate_{index}_uprm_marker_logprobs")
    return sorted(set(errors))


def read_complete_cache_rows(path: Path, *, k_candidates: int) -> list[JsonDict]:
    complete_by_key: dict[str, JsonDict] = {}
    for row in _read_jsonl(path):
        if not validate_cache_row(row, k_candidates=k_candidates):
            complete_by_key[_cache_key(row)] = row
    return list(complete_by_key.values())


def cache_summary(path: Path, *, k_candidates: int) -> JsonDict:
    rows = read_complete_cache_rows(path, k_candidates=k_candidates)
    min_candidates = min((len(row.get("candidates", [])) for row in rows), default=0)
    return {
        "n_cached_rows": len(rows),
        "n_questions": len(rows),
        "min_candidates_per_question": min_candidates,
        "has_per_token_logprobs": bool(rows)
        and all(
            _candidate_has_logprobs(candidate)
            for row in rows
            for candidate in list(row.get("candidates", []))[:k_candidates]
        ),
        "corpora_cached": sorted({str(row.get("corpus")) for row in rows}),
    }


def build_cache_row(
    *,
    row: JsonMap,
    row_index: int,
    candidates: Sequence[JsonMap],
    k_candidates: int,
    random_seed: int,
) -> JsonDict:
    return {
        "schema": CACHE_ROW_SCHEMA,
        "row_id": str(row.get("row_id", f"{CORPUS}:{row_index}")),
        "row_index": int(row_index),
        "corpus": CORPUS,
        "question": str(row.get("question", "")),
        "context": str(row.get("context", "")),
        "choices": list(row.get("choices") or []),
        "gold": str(row.get("gold", "")),
        "random_seed": int(random_seed),
        "candidates_per_question": int(k_candidates),
        "has_per_token_logprobs": all(_candidate_has_logprobs(candidate) for candidate in candidates),
        "candidates": [dict(candidate) for candidate in candidates],
    }


def reproducibility_checksum(*, k_candidates: int, random_seed: int, corpora: Sequence[str]) -> str:
    source = Path(__file__).read_text(encoding="utf-8")
    payload = {
        "generator_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "corpora": list(corpora),
        "k": int(k_candidates),
        "seed": int(random_seed),
    }
    return "sha256:" + hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def build_artifact(
    *,
    honest_verdict: str,
    root: Path,
    cache_path: Path,
    preconditions_checked: Sequence[JsonMap],
    gguf_path: Path | None,
    min_questions: int,
    k_candidates: int,
    started_at: float,
    finished_at: float,
) -> JsonDict:
    summary = cache_summary(cache_path, k_candidates=k_candidates)
    built = bool(
        summary["n_questions"] >= min_questions
        and k_candidates >= DEFAULT_K
        and summary["min_candidates_per_question"] >= k_candidates
        and summary["has_per_token_logprobs"]
        and CORPUS in summary["corpora_cached"]
    )
    if built:
        honest_verdict = (
            f"success_logprob_candidate_cache_built_musr_n{summary['n_questions']}_k{k_candidates}"
        )
    return {
        "schema": ARTIFACT_SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "result_path": (root / RESULT_RELATIVE_PATH).as_posix(),
        "honest_verdict": honest_verdict,
        "candidate_cache_built": built,
        "cache_jsonl_path": cache_path.as_posix(),
        "n_questions": int(summary["n_questions"] if built else 0),
        "candidates_per_question": int(k_candidates),
        "n_cached_rows": int(summary["n_cached_rows"] if built else summary["n_cached_rows"]),
        "has_per_token_logprobs": bool(summary["has_per_token_logprobs"] if built else False),
        "corpora_cached": list(summary["corpora_cached"] if built else []),
        "model_specs": {
            "generator_model": MODEL_NAME,
            "generator_hf_id": MODEL_HF_ID,
            "gguf_path": gguf_path.as_posix() if gguf_path else None,
            "cuda_gpu": 0,
            "server_url": f"http://127.0.0.1:{DEFAULT_SERVER_PORT}/completion",
            "requires_completion_probabilities": True,
            "requires_top_logprobs_for_markers": True,
            "logprobs_requested": {
                "generation": GENERATION_LOGPROBS_TOP_K,
                "uprm_markers": MARKER_LOGPROBS_TOP_K,
            },
        },
        "inference_substrate": "live_llm_inference" if built else "precondition_check_only",
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": reproducibility_checksum(
            k_candidates=k_candidates,
            random_seed=RANDOM_SEED,
            corpora=summary["corpora_cached"],
        ),
        "preconditions_checked": list(preconditions_checked),
        "duration_s": round(float(finished_at) - float(started_at), 6),
        "field_principles": FIELD_PRINCIPLES,
    }


def artifact_schema_errors(artifact: JsonMap) -> list[str]:
    errors = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if artifact.get("spec_refs") != SPEC_REFS:
        errors.append("spec_refs")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    for field in ("candidate_cache_built", "has_per_token_logprobs"):
        if not isinstance(artifact.get(field), bool):
            errors.append(field)
    for field in ("n_questions", "candidates_per_question", "n_cached_rows"):
        if not isinstance(artifact.get(field), int):
            errors.append(field)
    if not isinstance(artifact.get("preconditions_checked"), list):
        errors.append("preconditions_checked")
    if not str(artifact.get("honest_verdict", "")).startswith(("blocked_", "success_")):
        errors.append("honest_verdict")
    return sorted(set(errors))


def _resolve_gemma_gguf() -> str | None:  # pragma: no cover - host cache probe
    return uprm._resolve_gemma_gguf(MODEL_HF_ID)


def default_server_probe(port: int) -> PreconditionCheck:  # pragma: no cover - live HTTP boundary
    check = uprm.default_server_probe(port=port)
    return PreconditionCheck(check.resource, check.available, check.detail, check.path)


def default_corpus_loader(limit: int) -> list[JsonDict]:  # pragma: no cover - dataset cache boundary
    return harness.load_musr_murder_mysteries(limit=limit)


def _llama_generator(
    prompt: str, *, seed: int, config: GenerationConfig, server_port: int
) -> JsonDict:  # pragma: no cover - live HTTP boundary
    payload = uprm.llama_server_completion(
        prompt,
        port=server_port,
        seed=seed,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        logprobs=GENERATION_LOGPROBS_TOP_K,
        timeout_s=300,
        stop=["<|im_end|>", "<end_of_turn>", "<|endoftext|>"],
    )
    return uprm.parse_llama_completion_payload(payload)


def annotate_candidate_markers(
    row: JsonMap, candidate: JsonMap, *, server_port: int, seed: int
) -> JsonDict:  # pragma: no cover - live HTTP boundary
    steps = uprm.split_reasoning_steps(str(candidate.get("reasoning") or ""))
    if not steps:
        steps = [str(candidate.get("reasoning") or candidate.get("answer") or "")]
    annotated = dict(candidate)
    annotated["steps"] = steps
    marker_rows: list[dict[str, float]] = []
    for index, step in enumerate(steps):
        payload = uprm.llama_server_completion(
            uprm._marker_prompt(row, {**annotated, "steps": steps[: index + 1]}, step),
            port=server_port,
            seed=seed + index,
            max_tokens=1,
            temperature=0.0,
            logprobs=MARKER_LOGPROBS_TOP_K,
            timeout_s=120,
        )
        parsed = uprm.parse_llama_completion_payload(payload)
        if not parsed["top_logprobs"]:
            raise uprm.UprmScoringError("marker completion lacked top_logprobs")
        marker_row = parsed["top_logprobs"][0]
        if not _has_marker_pair(marker_row):
            raise uprm.UprmScoringError("marker top_logprobs lacked '+'/'-' alternatives")
        marker_rows.append(marker_row)
    annotated["uprm_marker_logprobs"] = marker_rows
    return annotated


def live_candidate_row_builder(
    *,
    row: JsonMap,
    row_index: int,
    k_candidates: int,
    random_seed: int,
    server_port: int,
) -> JsonDict:  # pragma: no cover - live HTTP boundary
    config = GenerationConfig(k=k_candidates, model=MODEL_NAME, gpu=0, max_tokens=512)

    def generator(prompt: str, *, seed: int, config: GenerationConfig) -> JsonDict:
        return _llama_generator(prompt, seed=seed, config=config, server_port=server_port)

    candidates = harness.generate_candidates_with_logprobs(
        row,
        generator=generator,
        config=config,
        seed=random_seed,
    )
    annotated = [
        annotate_candidate_markers(
            row,
            {
                **candidate,
                "answer": candidate.get("answer")
                or harness._match_choice(str(candidate.get("reasoning") or ""), row.get("choices", [])),
            },
            server_port=server_port,
            seed=random_seed + candidate_index * 100,
        )
        for candidate_index, candidate in enumerate(candidates)
    ]
    return build_cache_row(
        row=row,
        row_index=row_index,
        candidates=annotated,
        k_candidates=k_candidates,
        random_seed=random_seed,
    )


def check_preconditions(
    *,
    root: Path,
    gguf_resolver: Callable[[], str | None],
    server_probe: Callable[[int], PreconditionCheck],
    corpus_loader: Callable[[int], list[JsonDict]],
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
        )
    ]
    checks.append(server_probe(server_port))
    try:
        rows = corpus_loader(min_questions)
    except Exception as exc:
        rows = []
        detail = f"{type(exc).__name__}: {exc}"
    else:
        detail = f"{len(rows)} cached MuSR row(s), required >= {min_questions}"
    checks.append(PreconditionCheck("musr_corpus", len(rows) >= min_questions, detail, root.as_posix()))
    return checks, gguf_path if gguf_ok else None, rows


def _first_missing(checks: Sequence[PreconditionCheck]) -> str | None:
    for check in checks:
        if not check.available:
            return check.resource
    return None


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    gguf_resolver: Callable[[], str | None] = _resolve_gemma_gguf,
    server_probe: Callable[[int], PreconditionCheck] = default_server_probe,
    corpus_loader: Callable[[int], list[JsonDict]] = default_corpus_loader,
    candidate_row_builder: CandidateRowBuilder = live_candidate_row_builder,
    min_questions: int = DEFAULT_LIMIT,
    k_candidates: int = DEFAULT_K,
    random_seed: int = RANDOM_SEED,
    server_port: int = DEFAULT_SERVER_PORT,
    now: Clock = time.time,
) -> JsonDict:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    root = Path(root)
    cache_path = root / CACHE_RELATIVE_PATH
    artifact_path = Path(artifact_path) if artifact_path else root / RESULT_RELATIVE_PATH
    started_at = float(now())
    checks, gguf_path, corpus_rows = check_preconditions(
        root=root,
        gguf_resolver=gguf_resolver,
        server_probe=server_probe,
        corpus_loader=corpus_loader,
        min_questions=min_questions,
        server_port=server_port,
    )
    preconditions = [check.as_dict() for check in checks]
    missing = _first_missing(checks)
    if missing is not None:
        artifact = build_artifact(
            honest_verdict=f"blocked_{missing}",
            root=root,
            cache_path=cache_path,
            preconditions_checked=preconditions,
            gguf_path=gguf_path,
            min_questions=min_questions,
            k_candidates=k_candidates,
            started_at=started_at,
            finished_at=float(now()),
        )
        write_json(artifact_path, artifact)
        return artifact

    complete_keys = {
        _cache_key(row) for row in read_complete_cache_rows(cache_path, k_candidates=k_candidates)
    }
    try:
        for row_index, row in enumerate(corpus_rows[:min_questions]):
            if _cache_key(row) in complete_keys:
                continue
            cache_row = candidate_row_builder(
                row=row,
                row_index=row_index,
                k_candidates=k_candidates,
                random_seed=random_seed + row_index * 1000,
                server_port=server_port,
            )
            errors = validate_cache_row(cache_row, k_candidates=k_candidates)
            if errors:
                raise ValueError(f"generated cache row is malformed: {errors}")
            append_jsonl_atomic(cache_path, cache_row)
            complete_keys.add(_cache_key(cache_row))
    except Exception as exc:
        artifact = build_artifact(
            honest_verdict="blocked_generation_or_cache_error",
            root=root,
            cache_path=cache_path,
            preconditions_checked=preconditions
            + [
                {
                    "resource": "generation_or_cache_error",
                    "available": False,
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            ],
            gguf_path=gguf_path,
            min_questions=min_questions,
            k_candidates=k_candidates,
            started_at=started_at,
            finished_at=float(now()),
        )
        write_json(artifact_path, artifact)
        return artifact

    artifact = build_artifact(
        honest_verdict="success_pending_cache_summary",
        root=root,
        cache_path=cache_path,
        preconditions_checked=preconditions,
        gguf_path=gguf_path,
        min_questions=min_questions,
        k_candidates=k_candidates,
        started_at=started_at,
        finished_at=float(now()),
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
