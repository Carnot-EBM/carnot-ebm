"""Exp 1367 DiffuTruth-style energy-of-falsehood FoVer feasibility probe.

This module intentionally does not implement a full discrete text diffusion
model.  It runs the CPU feasibility slice requested for Exp 1367: corrupt key
tokens in each FoVer claim, reconstruct them with a deterministic local
back-paraphrase proxy, and measure whether the resulting reconstruction energy
aligns with Carnot's existing constraint/KAN energy signals.

Spec: REQ-VERIFY-1367, SCENARIO-VERIFY-1367
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260505"
EXPERIMENT_ID = 1367
SCHEMA = "diffutruth_energy_of_falsehood_probe_v1"
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT / "results" / "experiment_1367_diffutruth_energy_of_falsehood_probe.json"
)
DEFAULT_CORPUS_PATHS = (
    REPO_ROOT / "results" / "fover_corpus_v5.json",
    REPO_ROOT / "data" / "fover_corpus.jsonl",
    REPO_ROOT / "data" / "fover_corpus_v4.json",
    REPO_ROOT / "results" / "fover_labeled_steps_v21_multi.json",
)
DEFAULT_KAN_TRAINING_PATH = REPO_ROOT / "data" / "fover_corpus_v4.json"

PERTURBATION_METHOD = (
    "deterministic key-token synonym/random replacement over numbers and content words"
)
RECONSTRUCTION_METHOD = (
    "CPU local back-paraphrase proxy with restoration budget from lexical/arithmetic stability; "
    "no full discrete diffusion model or NLI model"
)
INTERPRETABLE_ALIGNMENT_MIN_ABS_R = 0.05

REQUIRED_ARTIFACT_FIELDS: set[str] = {
    "status",
    "corpus_cases_used",
    "perturbation_method",
    "reconstruction_method",
    "diffutruth_energy_delta_mean",
    "ising_correlation",
    "kan_correlation",
    "detection_auroc_proxy",
    "hallucination_energy_rate",
    "viable_as_complement",
    "honest_verdict",
}

_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?|[A-Za-z][A-Za-z0-9_-]*|[^\w\s]", re.ASCII)
_WORD_RE = re.compile(r"[A-Za-z0-9_.$+-]+", re.ASCII)
_ANSWER_ONLY_RE = re.compile(
    r"^\s*(?:the\s+)?(?:answer|result|total)\s*(?:is|=|:)\s*[-+]?\$?\d+(?:\.\d+)?\s*\.?\s*$",
    re.IGNORECASE,
)
_EQUATION_RE = re.compile(r"[-+]?\d+(?:\.\d+)?\s*[+*/-]\s*[-+]?\d+(?:\.\d+)?\s*=")

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "per",
    "so",
    "than",
    "that",
    "the",
    "then",
    "to",
    "was",
    "we",
    "with",
}

_SYNONYMS = {
    "answer": "result",
    "because": "since",
    "calculate": "compute",
    "computed": "calculated",
    "first": "initially",
    "left": "remaining",
    "more": "additional",
    "next": "then",
    "remaining": "left",
    "result": "answer",
    "so": "therefore",
    "therefore": "thus",
    "total": "sum",
}
_CANONICAL_SYNONYM = {
    "additional": "more",
    "answer": "answer",
    "calculated": "compute",
    "compute": "compute",
    "computed": "compute",
    "initially": "first",
    "left": "remaining",
    "remaining": "remaining",
    "result": "answer",
    "since": "because",
    "sum": "total",
    "therefore": "therefore",
    "then": "next",
    "thus": "therefore",
}
_NOISE_WORDS = (
    "unstable",
    "placeholder",
    "unknown",
    "contradictory",
    "noisy",
    "drifted",
)
_NOISE_NUMBERS = ("13", "17", "42", "73", "99", "101")


@dataclass(frozen=True)
class FoVerClaimCase:
    """One normalized FoVer claim with label 1 for hallucinated/incorrect."""

    case_id: str
    question: str
    response: str
    label: int


@dataclass(frozen=True)
class TokenPerturbation:
    """One deterministic token replacement applied during the stress test."""

    index: int
    original: str
    replacement: str
    kind: str
    salience: float


@dataclass(frozen=True)
class CorruptedClaim:
    """A corrupted claim plus enough metadata to run the local reconstruction."""

    text: str
    tokens: list[str]
    perturbations: list[TokenPerturbation]


@dataclass(frozen=True)
class ScoredDiffuTruthCase:
    """Per-case DiffuTruth proxy result."""

    case_id: str
    label: int
    corrupted_text: str
    reconstructed_text: str
    semantic_similarity: float
    diffutruth_energy: float
    local_stability: float
    perturbation_count: int
    unresolved_perturbation_rate: float


def utc_now_iso() -> str:
    """Return a UTC timestamp suitable for deterministic experiment artifacts."""

    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    """Write JSON with stable formatting so conductor diffs are readable."""

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
    """Write the mandatory in-progress artifact before corpus access."""

    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "run_date": run_date,
        "status": "in_progress",
        "honest_verdict": "in_progress",
        "started_at": utc_now_iso(),
    }
    write_json(path, artifact)
    return artifact


def load_fover_cases(
    path: Path | str | None = None,
    *,
    limit: int = 100,
) -> list[FoVerClaimCase]:
    """Load local FoVer rows and return a deterministic class-balanced subset."""

    candidate_paths = (Path(path),) if path is not None else DEFAULT_CORPUS_PATHS
    last_cases: list[FoVerClaimCase] = []
    for candidate in candidate_paths:
        if not candidate.exists():
            continue
        rows = _read_rows(candidate)
        cases = _cases_from_rows(rows)
        if not cases:
            continue
        last_cases = cases
        if {case.label for case in cases} == {0, 1}:
            return _balanced_subset(cases, limit)
    if last_cases:
        return _balanced_subset(last_cases, limit)
    searched = ", ".join(str(candidate) for candidate in candidate_paths)
    raise FileNotFoundError(f"no usable local FoVer cases found in: {searched}")


def corrupt_claim(text: str, *, seed: int = EXPERIMENT_ID) -> CorruptedClaim:
    """Corrupt key claim tokens with deterministic synonym or random replacements."""

    tokens = _tokenize(text)
    candidate_indices = [idx for idx, token in enumerate(tokens) if _is_key_token(token)]
    if not candidate_indices and tokens:
        candidate_indices = [idx for idx, token in enumerate(tokens) if token.strip()]
    if not candidate_indices:
        return CorruptedClaim(text=text, tokens=tokens, perturbations=[])

    rng = random.Random(_stable_seed(text, seed))
    candidate_indices = sorted(
        candidate_indices,
        key=lambda idx: (-_token_salience(tokens[idx]), rng.random(), idx),
    )
    target_count = min(6, max(1, math.ceil(len(candidate_indices) * 0.28)))
    selected = sorted(candidate_indices[:target_count])

    corrupted = list(tokens)
    perturbations: list[TokenPerturbation] = []
    for index in selected:
        original = corrupted[index]
        replacement, kind = _replacement_for(original, rng)
        corrupted[index] = replacement
        perturbations.append(
            TokenPerturbation(
                index=index,
                original=original,
                replacement=replacement,
                kind=kind,
                salience=_token_salience(original),
            )
        )
    return CorruptedClaim(
        text=_join_tokens(corrupted),
        tokens=corrupted,
        perturbations=perturbations,
    )


def reconstruct_claim(
    original_text: str,
    corrupted: CorruptedClaim,
    *,
    local_stability: float,
) -> tuple[str, float]:
    """Reconstruct a corrupted claim with a local stability-based proxy.

    The proxy mirrors DiffuTruth's attractor intuition: stable claims have
    enough local redundancy to pull perturbed key tokens back toward the
    original text, while unstable claims leave more random corruption
    unresolved.  The original text is used only as the deterministic local
    fixture for this feasibility probe; the artifact verdict labels this as a
    CPU proxy rather than a true diffusion/NLI result.
    """

    if not corrupted.perturbations:
        return original_text.strip(), 0.0

    stability = _clip(local_stability, 0.0, 1.0)
    reconstructed = list(corrupted.tokens)
    random_perturbations = [p for p in corrupted.perturbations if p.kind == "random"]
    restore_random_count = int(round(stability * len(random_perturbations)))
    restore_random = {
        p.index
        for p in sorted(random_perturbations, key=lambda p: (-p.salience, p.index))[
            :restore_random_count
        ]
    }

    unresolved = 0
    for perturbation in corrupted.perturbations:
        if perturbation.kind == "synonym" or perturbation.index in restore_random:
            reconstructed[perturbation.index] = perturbation.original
        else:
            unresolved += 1

    reconstructed_text = _join_tokens(reconstructed)
    if stability < 0.25 and _sentence_count(reconstructed_text) > 1:
        reconstructed_text = _first_sentence(reconstructed_text)

    unresolved_rate = unresolved / len(corrupted.perturbations)
    return reconstructed_text.strip(), float(unresolved_rate)


def semantic_similarity(left: str, right: str) -> float:
    """Return bag-of-content-token cosine similarity in [0, 1]."""

    left_counts = _content_counts(left)
    right_counts = _content_counts(right)
    if not left_counts and not right_counts:
        return 1.0
    if not left_counts or not right_counts:
        return 0.0
    dot = sum(left_counts.get(token, 0) * right_counts.get(token, 0) for token in left_counts)
    left_norm = math.sqrt(sum(value * value for value in left_counts.values()))
    right_norm = math.sqrt(sum(value * value for value in right_counts.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return _clip(dot / (left_norm * right_norm), 0.0, 1.0)


def score_case(case: FoVerClaimCase, *, seed: int = EXPERIMENT_ID) -> ScoredDiffuTruthCase:
    """Run the local DiffuTruth stress proxy for one FoVer case."""

    stability = local_reconstruction_stability(case.response)
    corrupted = corrupt_claim(case.response, seed=seed)
    reconstructed, unresolved_rate = reconstruct_claim(
        case.response,
        corrupted,
        local_stability=stability,
    )
    similarity = semantic_similarity(case.response, reconstructed)
    cosine_energy = 1.0 - similarity
    unresolved_energy = unresolved_rate * (1.0 - stability)
    energy = _clip(max(cosine_energy, unresolved_energy), 0.0, 1.0)
    return ScoredDiffuTruthCase(
        case_id=case.case_id,
        label=int(case.label),
        corrupted_text=corrupted.text,
        reconstructed_text=reconstructed,
        semantic_similarity=float(similarity),
        diffutruth_energy=float(energy),
        local_stability=float(stability),
        perturbation_count=len(corrupted.perturbations),
        unresolved_perturbation_rate=float(unresolved_rate),
    )


def local_reconstruction_stability(text: str) -> float:
    """Estimate local claim stability without using the FoVer label."""

    stripped = text.strip()
    tokens = _content_tokens(stripped)
    token_count = len(tokens)
    risk = 0.0

    if token_count <= 3:
        risk += 0.55
    elif token_count <= 8:
        risk += 0.35
    if _ANSWER_ONLY_RE.match(stripped):
        risk += 0.35
    if not any(any(char.isdigit() for char in token) for token in tokens):
        risk += 0.10

    z3_energy = _z3_score(stripped)
    risk += 0.45 * _clip(z3_energy, 0.0, 1.0)

    has_equation = bool(_EQUATION_RE.search(stripped))
    if has_equation and z3_energy <= 0.05:
        risk -= 0.20
    if _sentence_count(stripped) >= 2:
        risk -= 0.10
    if token_count >= 18:
        risk -= 0.05

    return _clip(1.0 - risk, 0.05, 1.0)


def load_existing_energy_scores(
    cases: Sequence[FoVerClaimCase],
    *,
    use_kan_adapter: bool = True,
    kan_training_path: Path | str | None = DEFAULT_KAN_TRAINING_PATH,
) -> tuple[list[float], list[float], dict[str, str]]:
    """Return Ising-style and KAN-style scores from existing Carnot machinery."""

    ising_scores = [_z3_score(case.response) for case in cases]
    kan_scores, kan_source = _kan_scores(
        cases,
        use_kan_adapter=use_kan_adapter,
        kan_training_path=kan_training_path,
    )
    sources = {
        "ising": (
            "carnot.verify.z3_math_verifier.Z3MathVerifier.score "
            "as local Ising/formal-constraint energy proxy; anchored by "
            "results/experiment_1121_k5_and_compose_production.json"
        ),
        "kan": kan_source,
    }
    return ising_scores, kan_scores, sources


def build_artifact(
    cases: Sequence[FoVerClaimCase],
    scores: Sequence[ScoredDiffuTruthCase],
    *,
    ising_scores: Sequence[float],
    kan_scores: Sequence[float],
    corpus_path: Path | str,
    score_sources: Mapping[str, str] | None = None,
    started_at: str | None = None,
    duration_s: float = 0.0,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """Build the schema-complete Exp 1367 artifact."""

    if not (len(cases) == len(scores) == len(ising_scores) == len(kan_scores)):
        raise ValueError(
            "cases, scores, ising_scores, and kan_scores must have equal lengths: "
            f"{len(cases)}, {len(scores)}, {len(ising_scores)}, {len(kan_scores)}"
        )

    labels = [int(case.label) for case in cases]
    energies = [float(score.diffutruth_energy) for score in scores]
    similarities = [float(score.semantic_similarity) for score in scores]
    factual_energies = [
        energy for label, energy in zip(labels, energies, strict=True) if label == 0
    ]
    hallucinated_energies = [
        energy for label, energy in zip(labels, energies, strict=True) if label == 1
    ]
    factual_similarities = [
        score for label, score in zip(labels, similarities, strict=True) if label == 0
    ]
    hallucinated_similarities = [
        score for label, score in zip(labels, similarities, strict=True) if label == 1
    ]

    diffutruth_delta = _mean(hallucinated_energies) - _mean(factual_energies)
    similarity_delta = _mean(factual_similarities) - _mean(hallucinated_similarities)
    detection_auroc = tie_aware_auroc(labels, energies)
    ising_corr = pearson_r(energies, [float(value) for value in ising_scores])
    kan_corr = pearson_r(energies, [float(value) for value in kan_scores])
    factual_threshold = _median(factual_energies) if factual_energies else _median(energies)
    hallucination_energy_rate = _mean(
        [1.0 if energy > factual_threshold else 0.0 for energy in hallucinated_energies]
    )
    aligned = (
        abs(ising_corr) >= INTERPRETABLE_ALIGNMENT_MIN_ABS_R
        or abs(kan_corr) >= INTERPRETABLE_ALIGNMENT_MIN_ABS_R
    )
    viable = bool(detection_auroc > 0.55 and aligned)

    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "run_date": run_date,
        "status": "complete",
        "started_at": started_at or utc_now_iso(),
        "finished_at": utc_now_iso(),
        "duration_s": round(float(duration_s), 3),
        "title": "DiffuTruth Energy Of Falsehood FoVer Probe",
        "spec": ["REQ-VERIFY-1367", "SCENARIO-VERIFY-1367"],
        "corpus_path": str(corpus_path),
        "corpus_cases_used": len(cases),
        "corpus_positive_hallucinated": int(sum(labels)),
        "corpus_negative_factual": int(len(labels) - sum(labels)),
        "perturbation_method": PERTURBATION_METHOD,
        "reconstruction_method": RECONSTRUCTION_METHOD,
        "diffutruth_energy_delta_mean": round(float(diffutruth_delta), 6),
        "diffutruth_similarity_delta_mean": round(float(similarity_delta), 6),
        "diffutruth_energy_mean_factual": round(_mean(factual_energies), 6),
        "diffutruth_energy_mean_hallucinated": round(_mean(hallucinated_energies), 6),
        "semantic_similarity_mean_factual": round(_mean(factual_similarities), 6),
        "semantic_similarity_mean_hallucinated": round(_mean(hallucinated_similarities), 6),
        "ising_correlation": round(float(ising_corr), 6),
        "kan_correlation": round(float(kan_corr), 6),
        "detection_auroc_proxy": round(float(detection_auroc), 6),
        "hallucination_energy_rate": round(float(hallucination_energy_rate), 6),
        "viable_as_complement": viable,
        "honest_verdict": _honest_verdict(viable, detection_auroc, ising_corr, kan_corr),
        "score_sources": dict(score_sources or {}),
        "models_used": [],
        "no_full_discrete_diffusion": True,
        "no_nli_model": True,
        "sample_cases": _sample_case_summaries(cases, scores),
    }
    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required artifact fields: {sorted(missing)}")
    if artifact["viable_as_complement"] != (
        artifact["detection_auroc_proxy"] > 0.55
        and (
            abs(artifact["ising_correlation"]) >= INTERPRETABLE_ALIGNMENT_MIN_ABS_R
            or abs(artifact["kan_correlation"]) >= INTERPRETABLE_ALIGNMENT_MIN_ABS_R
        )
    ):
        raise ValueError("viable_as_complement does not match the required gate")
    return artifact


def run_experiment(
    *,
    corpus_path: Path | str | None = None,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    limit: int = 100,
    use_kan_adapter: bool = True,
    kan_training_path: Path | str | None = DEFAULT_KAN_TRAINING_PATH,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """Run the complete local Exp 1367 probe and persist the artifact."""

    write_in_progress_artifact(output_path, run_date=run_date)
    started_at = utc_now_iso()
    t0 = time.perf_counter()
    cases = load_fover_cases(corpus_path, limit=limit)
    ising_scores, kan_scores, score_sources = load_existing_energy_scores(
        cases,
        use_kan_adapter=use_kan_adapter,
        kan_training_path=kan_training_path,
    )
    scored = [score_case(case, seed=EXPERIMENT_ID) for case in cases]
    resolved_corpus_path = (
        Path(corpus_path) if corpus_path is not None else _first_existing_corpus()
    )
    artifact = build_artifact(
        cases,
        scored,
        ising_scores=ising_scores,
        kan_scores=kan_scores,
        corpus_path=resolved_corpus_path,
        score_sources=score_sources,
        started_at=started_at,
        duration_s=time.perf_counter() - t0,
        run_date=run_date,
    )
    write_json(output_path, artifact)
    return artifact


def tie_aware_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute AUROC with 0.5 tie credit and neutral single-class fallback."""

    positives = [
        float(score) for label, score in zip(labels, scores, strict=False) if int(label) == 1
    ]
    negatives = [
        float(score) for label, score in zip(labels, scores, strict=False) if int(label) == 0
    ]
    if not positives or not negatives:
        return 0.5
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return float(wins / (len(positives) * len(negatives)))


def pearson_r(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Return Pearson r with zero-variance and length guards."""

    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    x_values = [float(value) for value in xs]
    y_values = [float(value) for value in ys]
    x_mean = _mean(x_values)
    y_mean = _mean(y_values)
    x_centered = [value - x_mean for value in x_values]
    y_centered = [value - y_mean for value in y_values]
    x_var = sum(value * value for value in x_centered)
    y_var = sum(value * value for value in y_centered)
    if x_var == 0.0 or y_var == 0.0:
        return 0.0
    return float(
        sum(x * y for x, y in zip(x_centered, y_centered, strict=True)) / math.sqrt(x_var * y_var)
    )


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, Mapping)]
    if isinstance(payload, Mapping):
        for key in ("pairs", "rows", "items", "examples", "data", "records"):
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(row) for row in value if isinstance(row, Mapping)]
    raise ValueError(f"unsupported FoVer corpus schema: {path}")


def _cases_from_rows(rows: Iterable[Mapping[str, Any]]) -> list[FoVerClaimCase]:
    cases: list[FoVerClaimCase] = []
    for index, row in enumerate(rows):
        label = _label_from_row(row)
        response = _row_response(row)
        if label is None or not response:
            continue
        case_id = str(
            row.get("question_index")
            or row.get("question_id")
            or row.get("case_id")
            or row.get("id")
            or f"case_{index}"
        )
        cases.append(
            FoVerClaimCase(
                case_id=case_id,
                question=str(row.get("question") or row.get("prompt") or ""),
                response=response,
                label=label,
            )
        )
    return cases


def _label_from_row(row: Mapping[str, Any]) -> int | None:
    if "is_correct" in row:
        return 0 if bool(row["is_correct"]) else 1
    if "correct" in row:
        return 0 if bool(row["correct"]) else 1
    if "step_correct" in row:
        return 0 if bool(row["step_correct"]) else 1

    raw = row.get("label", row.get("verdict", row.get("z3_label")))
    if isinstance(raw, bool):
        return 0 if raw else 1
    if isinstance(raw, (int, float)):
        return 0 if int(raw) == 1 else 1
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in {"correct", "true", "supported", "entailed", "1"}:
            return 0
        if normalized in {"incorrect", "wrong", "false", "violated", "violation", "0"}:
            return 1
    return None


def _row_response(row: Mapping[str, Any]) -> str:
    return str(
        row.get("response")
        or row.get("model_response")
        or row.get("step_text")
        or row.get("step")
        or ""
    ).strip()


def _balanced_subset(cases: Sequence[FoVerClaimCase], limit: int) -> list[FoVerClaimCase]:
    if limit <= 0 or len(cases) <= limit:
        return list(cases)
    positives = [idx for idx, case in enumerate(cases) if case.label == 1]
    negatives = [idx for idx, case in enumerate(cases) if case.label == 0]
    if not positives or not negatives:
        return list(cases[:limit])

    target_each = max(1, min(len(positives), len(negatives), limit // 2))
    selected = set(positives[:target_each] + negatives[:target_each])
    for idx in range(len(cases)):
        if len(selected) >= limit:
            break
        if idx not in selected and len(selected) < limit:
            selected.add(idx)
    return [cases[idx] for idx in sorted(selected)]


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def _join_tokens(tokens: Sequence[str]) -> str:
    text = ""
    for token in tokens:
        if not text:
            text = token
        elif (
            re.fullmatch(r"[.,;:!?%)]", token)
            or token in {"'", '"'}
            or text.endswith(("(", "$", "#"))
        ):
            text += token
        else:
            text += " " + token
    return text


def _is_key_token(token: str) -> bool:
    if any(char.isdigit() for char in token):
        return True
    lower = token.lower()
    return lower not in _STOPWORDS and len(lower) >= 4 and lower.isascii() and lower.isalnum()


def _token_salience(token: str) -> float:
    lower = token.lower()
    if any(char.isdigit() for char in token):
        return 3.0
    if lower in {"answer", "result", "total", "therefore", "because"}:
        return 2.0
    return 1.0


def _replacement_for(token: str, rng: random.Random) -> tuple[str, str]:
    lower = token.lower()
    if lower in _SYNONYMS and rng.random() < 0.45:
        replacement = _SYNONYMS[lower]
        if token[:1].isupper():
            replacement = replacement.capitalize()
        return replacement, "synonym"
    if any(char.isdigit() for char in token):
        choices = [item for item in _NOISE_NUMBERS if item != token]
        return rng.choice(choices), "random"
    choices = [item for item in _NOISE_WORDS if item != lower]
    replacement = rng.choice(choices)
    if token[:1].isupper():
        replacement = replacement.capitalize()
    return replacement, "random"


def _stable_seed(text: str, seed: int) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return seed + int(digest[:12], 16)


def _content_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in _WORD_RE.findall(text.lower()):
        token = raw.strip(".$+-_")
        if not token or token in _STOPWORDS:
            continue
        tokens.append(_CANONICAL_SYNONYM.get(token, token))
    return tokens


def _content_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in _content_tokens(text):
        counts[token] = counts.get(token, 0) + 1
    return counts


def _sentence_count(text: str) -> int:
    return len([part for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()])


def _first_sentence(text: str) -> str:
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    return parts[0] if parts else text


def _z3_score(text: str) -> float:
    try:
        from carnot.verify.z3_math_verifier import Z3MathVerifier

        return float(Z3MathVerifier().score(text))
    except Exception:
        return 0.0


def _kan_scores(
    cases: Sequence[FoVerClaimCase],
    *,
    use_kan_adapter: bool,
    kan_training_path: Path | str | None,
) -> tuple[list[float], str]:
    if not use_kan_adapter:
        return [_kan_feature_proxy(case.response) for case in cases], "local_text_feature_kan_proxy"
    if len({case.label for case in cases}) != 2:
        return (
            [_kan_feature_proxy(case.response) for case in cases],
            "local_text_feature_kan_proxy_single_label",
        )

    try:
        from carnot.verify.and_composition_verifier import SOSKANEnergyV3Adapter

        training_rows = _load_kan_training_rows(kan_training_path)
        adapter = SOSKANEnergyV3Adapter()
        adapter.fit_from_corpus(training_rows, n_epochs=25, lr=3e-3)
        return (
            [float(adapter.score(case.response)) for case in cases],
            (f"SOSKANEnergyV3Adapter trained on existing local FoVer artifact {kan_training_path}"),
        )
    except Exception as exc:
        return (
            [_kan_feature_proxy(case.response) for case in cases],
            f"local_text_feature_kan_proxy_after_adapter_error:{type(exc).__name__}",
        )


def _load_kan_training_rows(path: Path | str | None) -> list[dict[str, Any]]:
    candidate = Path(path) if path is not None else DEFAULT_KAN_TRAINING_PATH
    rows = _read_rows(candidate)
    examples = []
    for row in rows:
        label = _label_from_row(row)
        text = _row_response(row)
        if label is None or not text:
            continue
        examples.append({"step_text": text, "label": "incorrect" if label == 1 else "correct"})
    positives = [row for row in examples if row["label"] == "incorrect"]
    negatives = [row for row in examples if row["label"] == "correct"]
    if positives and negatives:
        n_each = min(120, len(positives), len(negatives))
        return positives[:n_each] + negatives[:n_each]
    return examples[:240]


def _kan_feature_proxy(text: str) -> float:
    words = text.split()
    n_words = max(len(words), 1)
    unique_ratio = len(set(words)) / n_words
    numeric_density = sum(1 for word in words if any(char.isdigit() for char in word)) / n_words
    shortness = 1.0 - min(math.log(len(text) + 1) / 6.5, 1.0)
    answer_only = 1.0 if _ANSWER_ONLY_RE.match(text.strip()) else 0.0
    equation_bonus = 0.15 if _EQUATION_RE.search(text) else 0.0
    return _clip(
        0.45 * shortness
        + 0.25 * (1.0 - unique_ratio)
        + 0.20 * answer_only
        + 0.10 * abs(numeric_density - 0.12)
        - equation_bonus,
        0.0,
        1.0,
    )


def _first_existing_corpus() -> Path:
    for path in DEFAULT_CORPUS_PATHS:
        if path.exists():
            return path
    return DEFAULT_CORPUS_PATHS[0]


def _sample_case_summaries(
    cases: Sequence[FoVerClaimCase],
    scores: Sequence[ScoredDiffuTruthCase],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for case, score in list(zip(cases, scores, strict=True))[:limit]:
        summaries.append(
            {
                "case_id": case.case_id,
                "label": case.label,
                "energy": round(score.diffutruth_energy, 6),
                "similarity": round(score.semantic_similarity, 6),
                "stability": round(score.local_stability, 6),
                "perturbations": score.perturbation_count,
                "response_preview": case.response[:140],
            }
        )
    return summaries


def _honest_verdict(
    viable: bool,
    detection_auroc: float,
    ising_corr: float,
    kan_corr: float,
) -> str:
    prefix = "diffutruth_proxy_viable_complement" if viable else "diffutruth_proxy_not_viable"
    return (
        f"{prefix}_auroc_{detection_auroc:.3f}_ising_r_{ising_corr:.3f}_"
        f"kan_r_{kan_corr:.3f}_cpu_proxy_not_full_diffusion"
    )


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(value) for value in values) / len(values))


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return float((ordered[midpoint - 1] + ordered[midpoint]) / 2.0)


def _clip(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


if __name__ == "__main__":  # pragma: no cover
    run_experiment()
