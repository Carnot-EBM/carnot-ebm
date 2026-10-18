"""Exp 1490 local partial-trace failure localization audit.

The audit tests a narrow, reproducible proxy for Kona-style partial-trace
localization: start from existing local telemetry, inject a known wrong answer
span, then ask whether local energy/verifier features rank that span above the
clean spans.  It deliberately does not call Kona, decode new model samples, or
claim answer quality.  The result is a bounded diagnostic about whether Carnot's
available trace features can point at an injected failure.

Spec refs: REQ-KONA-036, SCENARIO-KONA-036.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260507"
TELEMETRY_MANIFEST_PATH = (
    PROJECT_ROOT / "results" / "live_sota_balanced_telemetry_manifest_1480.jsonl"
)
RESULT_PATH = (
    PROJECT_ROOT / "results" / "experiment_1490_kona_ebt_partial_trace_localization_audit.json"
)
AUDIT_NOTE_PATH = "docs/research-notes/kona_ebt_partial_trace_localization_audit.md"
MODEL_SPECS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "model_specs",
    "localization_audit_complete",
    "traces_evaluated",
    "injected_failures",
    "localization_top1_rate",
    "localization_top3_rate",
    "random_baseline_rate",
    "decoded_quality_claim_allowed",
    "kona_dependency_used",
    "audit_note_path",
    "tests_run",
    "honest_verdict",
)
VERIFIER_MISMATCH_PENALTY = 5.0


@dataclass(frozen=True)
class SpanScore:
    """One candidate span and its local score.

    `local_energy` is intentionally simple and auditable. Clean spans use the
    observed token surprisal from the existing trace. The injected span uses the
    top-k energy of the deterministic wrong alternative plus a verifier mismatch
    penalty because the answer slot no longer matches the known expected answer.
    """

    span_index: int
    clean_text: str
    injected_text: str | None
    token_surprisal: float
    verifier_energy: float
    local_energy: float
    span_length: int
    injected_failure: bool


@dataclass(frozen=True)
class TraceLocalizationResult:
    """Localization result for one telemetry trace after deterministic injection."""

    case_id: str
    injected_span_index: int
    localization_rank: int
    length_baseline_credit: float
    spans: tuple[SpanScore, ...]


@dataclass(frozen=True)
class LocalizationSummary:
    """Aggregate Exp 1490 localization metrics."""

    localization_audit_complete: bool
    traces_evaluated: int
    injected_failures: int
    localization_top1_rate: float
    localization_top3_rate: float
    random_baseline_rate: float
    random_top3_baseline_rate: float
    length_baseline_top1_rate: float
    results: tuple[TraceLocalizationResult, ...]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _has_usable_telemetry(row: dict[str, Any]) -> bool:
    token_texts = row.get("token_texts")
    token_logprobs = row.get("token_logprobs")
    top_logprobs = row.get("top_logprobs")
    return bool(
        row.get("correct") is True
        and row.get("format_valid") is True
        and row.get("expected_answer")
        and row.get("adversarial_wrong_answer")
        and row.get("token_logprobs_available") is True
        and row.get("topk_alternatives_available") is True
        and isinstance(token_texts, list)
        and isinstance(token_logprobs, list)
        and isinstance(top_logprobs, list)
        and len(token_texts) == len(token_logprobs) == len(top_logprobs)
        and len(token_texts) > 0
    )


def load_telemetry_rows(
    path: str | Path = TELEMETRY_MANIFEST_PATH,
    *,
    rows: Iterable[dict[str, Any]] | None = None,
    max_traces: int = 12,
) -> tuple[dict[str, Any], ...]:
    """Load bounded clean Exp 1480-style rows with local token telemetry."""
    source_rows = list(rows) if rows is not None else _read_jsonl(Path(path))
    selected: list[dict[str, Any]] = []
    for row in source_rows:
        if _has_usable_telemetry(row):
            selected.append(row)
        if len(selected) >= max_traces:
            break
    return tuple(selected)


def _injected_answer_atom(row: dict[str, Any], clean_text: str) -> str:
    wrong_answer = str(row["adversarial_wrong_answer"]).strip()
    width = max(1, len(clean_text.strip() or clean_text))
    return wrong_answer[-width:]


def _alternative_energy(top_logprobs: dict[str, float], injected_text: str) -> float:
    if injected_text in top_logprobs:
        return -float(top_logprobs[injected_text])
    return max(-float(value) for value in top_logprobs.values()) + 1.0


def _span_length(text: str) -> int:
    return max(1, len(text.strip() or text))


def _rank_desc(scores: Sequence[float]) -> dict[int, int]:
    ordered = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    return {index: rank for rank, index in enumerate(ordered, start=1)}


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _length_baseline_credit(spans: Sequence[SpanScore], injected_span_index: int) -> float:
    max_length = max(span.span_length for span in spans)
    longest_count = sum(1 for span in spans if span.span_length == max_length)
    return (1.0 / longest_count) if spans[injected_span_index].span_length == max_length else 0.0


def score_trace(row: dict[str, Any]) -> TraceLocalizationResult:
    """Inject one deterministic bad span and rank it against clean spans."""
    token_texts = [str(token) for token in row["token_texts"]]
    token_logprobs = [float(logprob) for logprob in row["token_logprobs"]]
    top_logprobs = [dict(item) for item in row["top_logprobs"]]
    injected_span_index = len(token_texts) - 1
    injected_text = _injected_answer_atom(row, token_texts[injected_span_index])
    expected_answer = str(row["expected_answer"]).strip()

    spans: list[SpanScore] = []
    for index, (clean_text, token_logprob, alternatives) in enumerate(
        zip(token_texts, token_logprobs, top_logprobs, strict=True)
    ):
        injected_failure = index == injected_span_index
        observed_surprisal = -token_logprob
        verifier_energy = (
            VERIFIER_MISMATCH_PENALTY
            if injected_failure and injected_text != expected_answer[-len(injected_text) :]
            else 0.0
        )
        token_energy = (
            _alternative_energy(alternatives, injected_text)
            if injected_failure
            else observed_surprisal
        )
        rendered_text = injected_text if injected_failure else clean_text
        spans.append(
            SpanScore(
                span_index=index,
                clean_text=clean_text,
                injected_text=injected_text if injected_failure else None,
                token_surprisal=observed_surprisal,
                verifier_energy=verifier_energy,
                local_energy=token_energy + verifier_energy,
                span_length=_span_length(rendered_text),
                injected_failure=injected_failure,
            )
        )

    ranks = _rank_desc([span.local_energy for span in spans])
    return TraceLocalizationResult(
        case_id=str(row.get("case_id", f"trace_{injected_span_index}")),
        injected_span_index=injected_span_index,
        localization_rank=ranks[injected_span_index],
        length_baseline_credit=_length_baseline_credit(spans, injected_span_index),
        spans=tuple(spans),
    )


def run_localization_audit(
    rows: Iterable[dict[str, Any]] | None = None,
    *,
    path: str | Path = TELEMETRY_MANIFEST_PATH,
    max_traces: int = 12,
) -> LocalizationSummary:
    """Run the bounded injected-failure localization audit."""
    telemetry_rows = load_telemetry_rows(path, rows=rows, max_traces=max_traces)
    results = tuple(score_trace(row) for row in telemetry_rows)
    span_counts = [len(result.spans) for result in results]
    top1_rate = _mean([float(result.localization_rank == 1) for result in results])
    top3_rate = _mean([float(result.localization_rank <= 3) for result in results])
    return LocalizationSummary(
        localization_audit_complete=bool(results),
        traces_evaluated=len(results),
        injected_failures=len(results),
        localization_top1_rate=top1_rate,
        localization_top3_rate=top3_rate,
        random_baseline_rate=_mean([1.0 / count for count in span_counts]),
        random_top3_baseline_rate=_mean([min(3, count) / count for count in span_counts]),
        length_baseline_top1_rate=_mean([result.length_baseline_credit for result in results]),
        results=results,
    )


def _honest_verdict(summary: LocalizationSummary) -> str:
    outcome = (
        "beats_random"
        if summary.localization_top1_rate > summary.random_baseline_rate
        else "not_above_random"
    )
    return (
        f"bounded_injected_failure_localization_{outcome}_no_decoded_quality_claim"
        if summary.localization_audit_complete
        else "blocked_no_usable_local_trace_telemetry"
    )


def build_artifact(
    *,
    rows: Iterable[dict[str, Any]] | None = None,
    path: str | Path = TELEMETRY_MANIFEST_PATH,
    tests_run: Sequence[str] = (),
    max_traces: int = 12,
) -> dict[str, Any]:
    """Build the complete Exp 1490 artifact from measured local telemetry."""
    summary = run_localization_audit(rows, path=path, max_traces=max_traces)
    return {
        "schema": "carnot.phase3.kona_partial_trace_localization_audit.v1",
        "experiment": "1490_kona_ebt_partial_trace_localization_audit",
        "run_date": RUN_DATE,
        "spec_refs": ["REQ-KONA-036", "SCENARIO-KONA-036"],
        "status": "complete" if summary.localization_audit_complete else "blocked",
        "model_specs": list(MODEL_SPECS),
        "localization_audit_complete": summary.localization_audit_complete,
        "traces_evaluated": summary.traces_evaluated,
        "injected_failures": summary.injected_failures,
        "localization_top1_rate": summary.localization_top1_rate,
        "localization_top3_rate": summary.localization_top3_rate,
        "random_baseline_rate": summary.random_baseline_rate,
        "random_top3_baseline_rate": summary.random_top3_baseline_rate,
        "length_baseline_top1_rate": summary.length_baseline_top1_rate,
        "decoded_quality_claim_allowed": False,
        "kona_dependency_used": False,
        "audit_note_path": AUDIT_NOTE_PATH,
        "tests_run": list(tests_run),
        "honest_verdict": _honest_verdict(summary),
        "source_trace_manifest": str(TELEMETRY_MANIFEST_PATH.relative_to(PROJECT_ROOT)),
        "source_artifacts": [
            "results/experiment_1450_ebt_nrgpt_local_microprototype_audit.json",
            "results/experiment_1480_live_sota_balanced_telemetry_v2.json",
            "results/live_sota_balanced_telemetry_manifest_1480.jsonl",
        ],
        "span_energy_definition": (
            "clean spans use observed token surprisal; injected answer spans use top-k "
            "wrong-answer energy plus a verifier mismatch penalty"
        ),
        "per_trace": [
            {
                "case_id": result.case_id,
                "injected_span_index": result.injected_span_index,
                "localization_rank": result.localization_rank,
                "span_count": len(result.spans),
                "length_baseline_credit": result.length_baseline_credit,
                "spans": [asdict(span) for span in result.spans],
            }
            for result in summary.results
        ],
    }


def write_experiment_artifact(
    path: str | Path = RESULT_PATH,
    *,
    rows: Iterable[dict[str, Any]] | None = None,
    telemetry_path: str | Path = TELEMETRY_MANIFEST_PATH,
    tests_run: Sequence[str] = (),
    max_traces: int = 12,
) -> dict[str, Any]:
    """Persist Exp 1490 after first writing the required bootstrap artifact."""
    result_path = Path(path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps({"status": "in_progress"}, indent=2) + "\n")
    artifact = build_artifact(
        rows=rows,
        path=telemetry_path,
        tests_run=tests_run,
        max_traces=max_traces,
    )
    result_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact
