"""Exp 1400 BiPRM R2L retrospective FoVer pivot probe.

This module ports BiPRM's right-to-left process-reward idea to Carnot's local
FoVer rows without training a new reward model.  The probe treats Carnot's
deterministic verifier energy as the reward-model surrogate: forward scoring
asks which step reduces energy when removed, while R2L scoring asks whether a
candidate step remains consistent when judged from later steps and the final
answer backward.

Spec: REQ-VERIFY-1400, SCENARIO-VERIFY-1400
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from carnot.verify.z3_math_verifier import Z3MathVerifier


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260506"
EXPERIMENT_ID = 1400
SCHEMA = "biprm_retrospective_verification_probe_v1"
R2L_UPDATE_RULE = "r_t^R2L = f_theta(s_t | q, s_>t)"
FUSION_RULE = "sigma_t = (t + 1) / (T + 1); score_t = sigma_t * forward_t + (1 - sigma_t) * r2l_t"

DEFAULT_OUTPUT_PATH = (
    REPO_ROOT / "results" / "experiment_1400_biprm_retrospective_verification_probe.json"
)
DEFAULT_FOVER_PATH = REPO_ROOT / "data" / "fover_corpus.jsonl"

PIVOT_CATEGORIES = (
    "arithmetic_error",
    "logical_fallacy",
    "missing_premise",
    "hallucination",
)

REQUIRED_ARTIFACT_FIELDS: set[str] = {
    "status",
    "corpus_cases_used",
    "forward_only_pivot_precision",
    "biprm_r2l_pivot_precision",
    "pivot_precision_delta",
    "retrospective_verification_viable",
    "pivotal_step_categories",
    "honest_verdict",
}

_NUMBER_RE = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?")
_ANSWER_MARKER_RE = re.compile(r"\b(?:answer|final|therefore the answer|result)\b", re.I)
_FINAL_ANSWER_MARKER_RE = re.compile(
    r"\b(?:answer|final answer|therefore,? the answer|result)\b", re.I
)
_MATH_OPERATOR_RE = re.compile(r"[+*/=]|\\times|\\cdot|\d\s*-\s*\d")
_PERCENT_OF_RE = re.compile(
    r"(?P<pct>\d+(?:\.\d+)?)\s*%\s+of\s+\$?(?P<base>[-+]?\d[\d,]*(?:\.\d+)?)",
    re.I,
)
_DECIMAL_TIMES_RE = re.compile(
    r"(?P<pct>0?\.\d+)\s*(?:\\times|times|\*)\s*\$?(?P<base>[-+]?\d[\d,]*(?:\.\d+)?)",
    re.I,
)
_REMAINING_RESULT_RE = re.compile(
    r"\b(?:remaining|left|remain)\b[^.\n=]*?(?:=|is|are)\s*\$?(?P<value>[-+]?\d[\d,]*(?:\.\d+)?)",
    re.I,
)
_STEP_PREFIX_RE = re.compile(r"^\s*(?:[-*]\s+|\d+[.)]\s+|Step\s+\d+\s*[:.)]\s*)", re.I)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")

_HALLUCINATION_MARKERS = (
    "---",
    "new problem",
    "a man has",
    "unrelated",
    "ignore the",
    "previous question",
)


@dataclass(frozen=True)
class FoVerRetrospectivePair:
    """One FoVer positive/negative pair normalized for pivot localization."""

    case_id: str
    positive_text: str
    negative_text: str
    steps: tuple[str, ...]
    gold_pivot_indices: tuple[int, ...]
    gold_pivot_category: str
    annotation_source: str

    @classmethod
    def from_texts(
        cls,
        *,
        case_id: str,
        positive_text: str,
        negative_text: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> "FoVerRetrospectivePair":
        """Build a pair and derive human or proxy pivot labels.

        FoVer v4/v5 rows in this repository label whole rejected steps but do
        not usually carry human important-step indices.  The fallback therefore
        derives a proxy pivot from the rejected step text itself, preferring
        arithmetic contradictions, then wrong-base/missing-premise evidence,
        then hallucinated unrelated text, then final-answer mismatch.
        """

        steps = tuple(split_reasoning_steps(negative_text))
        pivots, category, source = derive_pivot_labels(
            steps,
            positive_text=positive_text,
            metadata=metadata or {},
        )
        return cls(
            case_id=str(case_id),
            positive_text=positive_text,
            negative_text=negative_text,
            steps=steps,
            gold_pivot_indices=tuple(pivots),
            gold_pivot_category=category,
            annotation_source=source,
        )


@dataclass(frozen=True)
class ScoredPivotCase:
    """Pivot scores and selected pivots for one FoVer pair."""

    case_id: str
    forward_only_scores: tuple[float, ...]
    biprm_r2l_scores: tuple[float, ...]
    biprm_fused_scores: tuple[float, ...]
    forward_pivot_index: int
    biprm_pivot_index: int
    gold_pivot_indices: tuple[int, ...]
    gold_pivot_category: str
    annotation_source: str

    @property
    def forward_correct(self) -> bool:
        """Return whether the L2R baseline selected a proxy/human pivot."""

        return self.forward_pivot_index in self.gold_pivot_indices

    @property
    def biprm_correct(self) -> bool:
        """Return whether the R2L BiPRM score selected a proxy/human pivot."""

        return self.biprm_pivot_index in self.gold_pivot_indices


def utc_now_iso() -> str:
    """Return a compact UTC timestamp for experiment artifacts."""

    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    """Write a deterministic JSON object to disk."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def write_in_progress_artifact(
    path: Path | str = DEFAULT_OUTPUT_PATH,
    *,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """REQ-VERIFY-1400: write the bootstrap artifact before corpus loading."""

    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "run_date": run_date,
        "status": "in_progress",
        "started_at": utc_now_iso(),
        "corpus_cases_used": 0,
        "forward_only_pivot_precision": None,
        "biprm_r2l_pivot_precision": None,
        "pivot_precision_delta": None,
        "retrospective_verification_viable": False,
        "pivotal_step_categories": {category: 0 for category in PIVOT_CATEGORIES},
        "r2l_update_rule": R2L_UPDATE_RULE,
        "biprm_fusion_rule": FUSION_RULE,
        "fresh_llm_inference_used": False,
        "gpu_required": False,
        "honest_verdict": "in_progress",
    }
    write_json(path, artifact)
    return artifact


def load_fover_verified_pairs(
    path: Path | str = DEFAULT_FOVER_PATH,
    *,
    limit: int = 100,
) -> list[FoVerRetrospectivePair]:
    """Load same-question FoVer positive/negative rows as verified pairs."""

    rows = _read_rows(Path(path))
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for index, row in enumerate(rows):
        text = _row_text(row)
        label = _label_direction(row)
        if not text or label is None:
            continue
        case_id = str(row.get("question_id") or row.get("id") or row.get("case_id") or index)
        bucket = grouped.setdefault(case_id, {"positive": [], "negative": []})
        bucket["positive" if label else "negative"].append(dict(row))

    pairs: list[FoVerRetrospectivePair] = []
    for case_id in sorted(grouped, key=_natural_sort_key):
        positives = grouped[case_id]["positive"]
        negatives = grouped[case_id]["negative"]
        if not positives or not negatives:
            continue
        positive_text = _row_text(positives[0])
        for neg_index, negative in enumerate(negatives):
            suffix = "" if len(negatives) == 1 else f":{neg_index}"
            pairs.append(
                FoVerRetrospectivePair.from_texts(
                    case_id=f"{case_id}{suffix}",
                    positive_text=positive_text,
                    negative_text=_row_text(negative),
                    metadata=negative,
                )
            )
            if len(pairs) >= limit:
                return pairs
    return pairs


def split_reasoning_steps(text: str) -> list[str]:
    """Split FoVer reasoning text into stable, verifier-sized steps."""

    cleaned = str(text).replace("\\n", "\n")
    cleaned = cleaned.replace("\\(", " ").replace("\\)", " ")
    cleaned = cleaned.replace("\\[", " ").replace("\\]", " ")
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"(?m)^\s*(?=(?:Step\s*)?\d+[.)]\s+|Step\s+\d+\s*:)", "", cleaned)
    cleaned = re.sub(r"\n\s*[-*]\s+", "\n", cleaned)

    chunks: list[str] = []
    for raw_line in cleaned.splitlines():
        line = _STEP_PREFIX_RE.sub("", raw_line.strip(" \t-*"))
        if not line:
            continue
        chunks.extend(_split_sentence_like_step(line))

    if not chunks:
        stripped = cleaned.strip()
        return [stripped] if stripped else []
    return [chunk for chunk in chunks if chunk.strip()]


def derive_pivot_labels(
    steps: Sequence[str],
    *,
    positive_text: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> tuple[tuple[int, ...], str, str]:
    """Return gold pivot indices, category, and annotation source."""

    human = _human_pivots(metadata or {}, len(steps))
    if human:
        category = _metadata_category(metadata or {}) or classify_step_category(
            steps[human[0]],
            prior_steps=steps[: human[0]],
            future_steps=steps[human[0] + 1 :],
            positive_text=positive_text,
        )
        return tuple(human), category, "fover_metadata"

    if not steps:
        return (0,), "logical_fallacy", "fover_negative_proxy"

    categorized: dict[str, list[int]] = {category: [] for category in PIVOT_CATEGORIES}
    for index, step in enumerate(steps):
        category = classify_step_category(
            step,
            prior_steps=steps[:index],
            future_steps=steps[index + 1 :],
            positive_text=positive_text,
        )
        categorized[category].append(index)

    for category in ("arithmetic_error", "missing_premise", "hallucination", "logical_fallacy"):
        indices = categorized.get(category, [])
        if indices:
            return (indices[0],), category, "fover_negative_proxy"

    scores = [
        local_step_energy(
            step,
            prior_steps=steps[:index],
            future_steps=steps[index + 1 :],
            positive_text=positive_text,
        )
        for index, step in enumerate(steps)
    ]
    return (_top_index(scores),), "logical_fallacy", "fover_negative_proxy"


def classify_step_category(
    step: str,
    *,
    prior_steps: Sequence[str] = (),
    future_steps: Sequence[str] = (),
    positive_text: str = "",
) -> str:
    """Classify a candidate pivotal step into the required taxonomy."""

    arithmetic = _arithmetic_energy(step)
    if arithmetic >= 0.5:
        return "arithmetic_error"
    if _missing_premise_score(step, prior_steps=prior_steps, positive_text=positive_text) >= 0.5:
        return "missing_premise"
    if _hallucination_score(step, positive_text=positive_text) >= 0.5:
        return "hallucination"
    if (
        _reference_mismatch_score(step, future_steps=future_steps, positive_text=positive_text)
        >= 0.5
    ):
        return "logical_fallacy"
    return "logical_fallacy"


def trace_energy(steps: Sequence[str], *, positive_text: str = "") -> float:
    """Return verifier energy for a candidate reasoning trace."""

    if not steps:
        return 0.0

    total = 0.0
    for index, step in enumerate(steps):
        total += local_step_energy(
            step,
            prior_steps=steps[:index],
            future_steps=steps[index + 1 :],
            positive_text=positive_text,
        )

    final_step = steps[-1]
    if _FINAL_ANSWER_MARKER_RE.search(final_step):
        total += 1.5 * _final_answer_mismatch_score(final_step, positive_text)
    return float(max(0.0, total))


def local_step_energy(
    step: str,
    *,
    prior_steps: Sequence[str] = (),
    future_steps: Sequence[str] = (),
    positive_text: str = "",
) -> float:
    """Return local verifier energy for one candidate pivot step."""

    arithmetic = _arithmetic_energy(step)
    missing = _missing_premise_score(step, prior_steps=prior_steps, positive_text=positive_text)
    hallucination = _hallucination_score(step, positive_text=positive_text)
    reference = _reference_mismatch_score(
        step,
        future_steps=future_steps,
        positive_text=positive_text,
    )
    return float(arithmetic + 0.9 * missing + 0.7 * hallucination + 0.4 * reference)


def forward_only_pivot_scores(pair: FoVerRetrospectivePair) -> tuple[float, ...]:
    """Score each step by leave-one-out verifier-energy decrease."""

    steps = list(pair.steps)
    if not steps:
        return (0.0,)
    full_energy = trace_energy(steps, positive_text=pair.positive_text)
    scores: list[float] = []
    for index in range(len(steps)):
        without = steps[:index] + steps[index + 1 :]
        reduced = trace_energy(without, positive_text=pair.positive_text)
        scores.append(max(0.0, full_energy - reduced))
    return tuple(scores)


def biprm_r2l_pivot_scores(pair: FoVerRetrospectivePair) -> tuple[float, ...]:
    """Score candidate pivots from the answer backward using later steps."""

    steps = list(pair.steps)
    if not steps:
        return (0.0,)

    scores: list[float] = []
    for index, step in enumerate(steps):
        future = steps[index + 1 :]
        with_pivot = trace_energy([step] + future, positive_text=pair.positive_text)
        without_pivot = trace_energy(future, positive_text=pair.positive_text)
        energy_delta = max(0.0, with_pivot - without_pivot)
        local = local_step_energy(
            step,
            prior_steps=steps[:index],
            future_steps=future,
            positive_text=pair.positive_text,
        )
        retrospective = (
            energy_delta
            + 1.15 * local
            + _future_conflict_score(
                step,
                future_steps=future,
                positive_text=pair.positive_text,
            )
        )
        if _is_final_answer_symptom(step):
            retrospective *= 0.6
        scores.append(float(retrospective))
    return tuple(scores)


def score_pair(pair: FoVerRetrospectivePair) -> ScoredPivotCase:
    """Compute forward-only and BiPRM R2L pivot decisions for one pair."""

    forward = forward_only_pivot_scores(pair)
    r2l = biprm_r2l_pivot_scores(pair)
    count = max(len(forward), len(r2l), 1)
    fused: list[float] = []
    for index in range(count):
        f_score = forward[index] if index < len(forward) else 0.0
        r_score = r2l[index] if index < len(r2l) else 0.0
        sigma = (index + 1) / (count + 1)
        fused.append(float(sigma * f_score + (1.0 - sigma) * r_score))

    return ScoredPivotCase(
        case_id=pair.case_id,
        forward_only_scores=tuple(round(score, 6) for score in forward),
        biprm_r2l_scores=tuple(round(score, 6) for score in r2l),
        biprm_fused_scores=tuple(round(score, 6) for score in fused),
        forward_pivot_index=_top_index(forward),
        biprm_pivot_index=_top_index(fused),
        gold_pivot_indices=pair.gold_pivot_indices,
        gold_pivot_category=pair.gold_pivot_category,
        annotation_source=pair.annotation_source,
    )


def pivot_precision(scores: Sequence[ScoredPivotCase], *, method: str) -> float:
    """Return top-1 pivot identification precision for the selected method."""

    if not scores:
        return 0.0
    if method == "forward":
        hits = sum(1 for score in scores if score.forward_correct)
    elif method == "biprm_r2l":
        hits = sum(1 for score in scores if score.biprm_correct)
    else:
        raise ValueError(f"unsupported pivot precision method: {method}")
    return float(hits / len(scores))


def build_artifact(
    pairs: Sequence[FoVerRetrospectivePair],
    scores: Sequence[ScoredPivotCase],
    *,
    corpus_path: Path | str,
    started_at: str | None = None,
    duration_s: float = 0.0,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """Build the complete Exp 1400 result artifact."""

    if len(pairs) != len(scores):
        raise ValueError(f"pairs and scores length mismatch: {len(pairs)} vs {len(scores)}")

    forward_precision = round(pivot_precision(scores, method="forward"), 6)
    r2l_precision = round(pivot_precision(scores, method="biprm_r2l"), 6)
    delta = round(r2l_precision - forward_precision, 6)
    categories = {category: 0 for category in PIVOT_CATEGORIES}
    for pair in pairs:
        categories[pair.gold_pivot_category] = categories.get(pair.gold_pivot_category, 0) + 1

    viable = delta > 0.0
    human_cases = sum(1 for pair in pairs if pair.annotation_source == "fover_metadata")
    proxy_cases = len(pairs) - human_cases
    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "run_date": run_date,
        "status": "complete" if pairs else "blocked",
        "started_at": started_at or utc_now_iso(),
        "finished_at": utc_now_iso(),
        "duration_s": round(float(duration_s), 3),
        "title": "BiPRM R2L Retrospective Verification Probe on FoVer Pairs",
        "spec": ["REQ-VERIFY-1400", "SCENARIO-VERIFY-1400"],
        "corpus_path": str(corpus_path),
        "corpus_cases_used": len(pairs),
        "forward_only_pivot_precision": forward_precision,
        "biprm_r2l_pivot_precision": r2l_precision,
        "pivot_precision_delta": delta,
        "retrospective_verification_viable": bool(viable),
        "pivotal_step_categories": categories,
        "human_annotated_pivot_cases": human_cases,
        "proxy_pivot_cases": proxy_cases,
        "single_step_cases": sum(1 for pair in pairs if len(pair.steps) == 1),
        "r2l_update_rule": R2L_UPDATE_RULE,
        "biprm_fusion_rule": FUSION_RULE,
        "baseline_scoring_rule": "forward_only_score_i = E(s_1:T) - E(s_1:i-1, s_i+1:T)",
        "retrospective_scoring_rule": "r2l_score_i = E(s_i, s_>i) - E(s_>i), judged from final-answer context",
        "fresh_llm_inference_used": False,
        "gpu_required": False,
        "models_used": [],
        "sample_scored_cases": _sample_cases(pairs, scores),
        "honest_verdict": _honest_verdict(
            case_count=len(pairs),
            delta=delta,
            human_cases=human_cases,
        ),
    }

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required artifact fields: {sorted(missing)}")
    if artifact["pivot_precision_delta"] != round(
        artifact["biprm_r2l_pivot_precision"] - artifact["forward_only_pivot_precision"], 6
    ):
        raise ValueError("pivot_precision_delta must equal R2L precision minus forward precision")
    if artifact["retrospective_verification_viable"] != (artifact["pivot_precision_delta"] > 0):
        raise ValueError("retrospective_verification_viable must match the positive-delta gate")
    return artifact


def run_experiment(
    *,
    corpus_path: Path | str = DEFAULT_FOVER_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    limit: int = 100,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """Run the CPU-only BiPRM retrospective probe and persist the artifact."""

    write_in_progress_artifact(output_path, run_date=run_date)
    started_at = utc_now_iso()
    t0 = time.perf_counter()
    pairs = load_fover_verified_pairs(corpus_path, limit=limit)
    scores = [score_pair(pair) for pair in pairs]
    artifact = build_artifact(
        pairs,
        scores,
        corpus_path=corpus_path,
        started_at=started_at,
        duration_s=time.perf_counter() - t0,
        run_date=run_date,
    )
    write_json(output_path, artifact)
    return artifact


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
        return rows

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, Mapping)]
    if isinstance(payload, Mapping):
        for key in ("pairs", "rows", "items", "examples", "data", "records"):
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(row) for row in value if isinstance(row, Mapping)]
    return []


def _row_text(row: Mapping[str, Any]) -> str:
    return str(row.get("step_text") or row.get("response") or row.get("step") or "").strip()


def _label_direction(row: Mapping[str, Any]) -> bool | None:
    raw = row.get("label", row.get("is_correct", row.get("step_correct")))
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return bool(raw)
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in {"correct", "true", "yes", "1", "verified"}:
            return True
        if normalized in {"incorrect", "false", "wrong", "0", "rejected", "incoherent"}:
            return False
    return None


def _split_sentence_like_step(line: str) -> list[str]:
    if not line:
        return []
    parts = re.split(
        r"(?<=[.!?])\s+(?=(?:The|Therefore|Then|Next|Finally|Answer|Step|Compute|Now|First|Since|So)\b)",
        line,
    )
    if len(parts) == 1 and len(line) > 220:
        parts = re.split(r"(?<=[.!?])\s+", line)
    return [part.strip(" \t-*") for part in parts if part.strip(" \t-*")]


def _human_pivots(metadata: Mapping[str, Any], n_steps: int) -> tuple[int, ...]:
    keys = (
        "important_step_indices",
        "human_important_step_indices",
        "human_pivot_indices",
        "pivot_step_indices",
        "incorrect_step_indices",
        "error_step_indices",
    )
    for key in keys:
        value = metadata.get(key)
        indices = _coerce_indices(value, n_steps)
        if indices:
            return indices
    single_keys = (
        "important_step_index",
        "human_pivot_index",
        "pivot_step_index",
        "error_step_index",
    )
    for key in single_keys:
        indices = _coerce_indices(metadata.get(key), n_steps)
        if indices:
            return indices
    return ()


def _coerce_indices(value: Any, n_steps: int) -> tuple[int, ...]:
    if value is None:
        return ()
    raw_items: list[Any]
    if isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        raw_items = [value]

    parsed: list[int] = []
    for item in raw_items:
        try:
            parsed.append(int(item))
        except (TypeError, ValueError):
            continue
    if not parsed:
        return ()
    if parsed and min(parsed) >= 1 and max(parsed) <= n_steps:
        parsed = [item - 1 for item in parsed]
    return tuple(sorted({item for item in parsed if 0 <= item < n_steps}))


def _metadata_category(metadata: Mapping[str, Any]) -> str | None:
    raw = (
        metadata.get("pivot_category") or metadata.get("error_category") or metadata.get("category")
    )
    if not isinstance(raw, str):
        return None
    normalized = raw.strip().lower().replace(" ", "_")
    aliases = {
        "arithmetic": "arithmetic_error",
        "math_error": "arithmetic_error",
        "logical": "logical_fallacy",
        "logic": "logical_fallacy",
        "unsupported": "hallucination",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in PIVOT_CATEGORIES else None


def _arithmetic_energy(step: str) -> float:
    return float(Z3MathVerifier().verify_step(step))


def _missing_premise_score(
    step: str,
    *,
    prior_steps: Sequence[str] = (),
    positive_text: str = "",
) -> float:
    expected_bases = _expected_percentage_bases(prior_steps, positive_text)
    if not expected_bases:
        return 0.0

    bases = _percentage_bases(step)
    if not bases:
        return 0.0

    best = 0.0
    for base in bases:
        nearest = min(expected_bases, key=lambda expected: abs(expected - base))
        if abs(nearest - base) <= max(0.02, abs(nearest) * 0.01):
            continue
        distance = abs(nearest - base) / max(abs(nearest), abs(base), 1.0)
        best = max(best, min(1.0, 0.65 + distance))
    return float(best)


def _expected_percentage_bases(prior_steps: Sequence[str], positive_text: str) -> list[float]:
    bases: list[float] = []
    for step in prior_steps[-2:]:
        remaining = _remaining_result(step)
        if remaining is not None:
            bases.append(remaining)
        conclusion = _last_number(step)
        if conclusion is not None:
            bases.append(conclusion)

    for base in _percentage_bases(positive_text):
        bases.append(base)
    return _dedupe_close_numbers(bases)


def _percentage_bases(text: str) -> list[float]:
    bases: list[float] = []
    normalized = text.replace("\\times", " times ").replace("\\cdot", " times ")
    for match in _PERCENT_OF_RE.finditer(normalized):
        parsed = _parse_number(match.group("base"))
        if parsed is not None:
            bases.append(parsed)
    for match in _DECIMAL_TIMES_RE.finditer(normalized):
        parsed = _parse_number(match.group("base"))
        if parsed is not None:
            bases.append(parsed)
    return bases


def _remaining_result(text: str) -> float | None:
    match = _REMAINING_RESULT_RE.search(text)
    if not match:
        return None
    return _parse_number(match.group("value"))


def _hallucination_score(step: str, *, positive_text: str = "") -> float:
    lowered = step.lower()
    if any(marker in lowered for marker in _HALLUCINATION_MARKERS):
        return 1.0
    if not positive_text:
        return 0.0

    step_words = {word.lower() for word in _WORD_RE.findall(step) if len(word) > 3}
    positive_words = {word.lower() for word in _WORD_RE.findall(positive_text) if len(word) > 3}
    if len(step_words) < 5 or not positive_words:
        return 0.0
    overlap = len(step_words & positive_words) / max(len(step_words), 1)
    step_numbers = set(_numbers(step))
    positive_numbers = set(_numbers(positive_text))
    unsupported_numbers = step_numbers - positive_numbers
    if overlap < 0.2 and unsupported_numbers and not _MATH_OPERATOR_RE.search(step):
        return 0.8
    return 0.0


def _reference_mismatch_score(
    step: str,
    *,
    future_steps: Sequence[str] = (),
    positive_text: str = "",
) -> float:
    if not _ANSWER_MARKER_RE.search(step):
        return 0.0
    return _final_answer_mismatch_score(step, positive_text)


def _final_answer_mismatch_score(text: str, positive_text: str) -> float:
    predicted = _last_number(text)
    expected = _last_number(positive_text)
    if predicted is None or expected is None:
        return 0.0
    if math.isclose(predicted, expected, rel_tol=1e-6, abs_tol=1e-6):
        return 0.0
    return min(1.0, abs(predicted - expected) / max(abs(expected), abs(predicted), 1.0) + 0.25)


def _future_conflict_score(
    step: str,
    *,
    future_steps: Sequence[str],
    positive_text: str,
) -> float:
    score = _missing_premise_score(step, prior_steps=(), positive_text=positive_text)
    if future_steps and _is_final_answer_symptom(future_steps[-1]):
        score += 0.5 * _final_answer_mismatch_score(future_steps[-1], positive_text)
    return float(min(1.0, score))


def _is_final_answer_symptom(step: str) -> bool:
    return bool(_FINAL_ANSWER_MARKER_RE.search(step)) and not bool(_MATH_OPERATOR_RE.search(step))


def _numbers(text: str) -> list[float]:
    parsed: list[float] = []
    for match in _NUMBER_RE.finditer(str(text)):
        value = _parse_number(match.group(0))
        if value is not None:
            parsed.append(value)
    return parsed


def _last_number(text: str) -> float | None:
    nums = _numbers(text)
    return nums[-1] if nums else None


def _parse_number(raw: str) -> float | None:
    cleaned = str(raw).strip().replace("$", "").replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _dedupe_close_numbers(values: Sequence[float]) -> list[float]:
    deduped: list[float] = []
    for value in values:
        if not any(math.isclose(value, other, rel_tol=1e-6, abs_tol=1e-6) for other in deduped):
            deduped.append(float(value))
    return deduped


def _top_index(scores: Sequence[float]) -> int:
    if not scores:
        return 0
    return max(range(len(scores)), key=lambda index: (scores[index], -index))


def _sample_cases(
    pairs: Sequence[FoVerRetrospectivePair],
    scores: Sequence[ScoredPivotCase],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair, score in list(zip(pairs, scores, strict=False))[:limit]:
        rows.append(
            {
                "case_id": pair.case_id,
                "step_count": len(pair.steps),
                "gold_pivot_indices": list(pair.gold_pivot_indices),
                "gold_pivot_category": pair.gold_pivot_category,
                "forward_pivot_index": score.forward_pivot_index,
                "biprm_pivot_index": score.biprm_pivot_index,
                "forward_correct": score.forward_correct,
                "biprm_correct": score.biprm_correct,
                "annotation_source": pair.annotation_source,
            }
        )
    return rows


def _honest_verdict(*, case_count: int, delta: float, human_cases: int) -> str:
    if case_count == 0:
        return "blocked_no_local_fover_verified_pairs_found"
    annotation_note = (
        "human_pivot_annotations_present"
        if human_cases
        else "fover_negative_step_proxy_labels_used_no_human_pivots"
    )
    if delta > 0:
        return f"viable_positive_r2l_pivot_precision_delta_{annotation_note}"
    if delta == 0:
        return f"not_viable_no_r2l_pivot_precision_delta_{annotation_note}"
    return f"not_viable_negative_r2l_pivot_precision_delta_{annotation_note}"


def _natural_sort_key(value: str) -> tuple[int, str]:
    try:
        return (0, f"{int(value):012d}")
    except ValueError:
        return (1, value)
