"""Eidoku-style CSP neuro-symbolic verification probe for Exp 1365.

The probe is deliberately grammar-free: it accepts free-text FoVer rows, builds
three deterministic proxy costs, and writes the required experiment artifact
without calling any generation model.

Spec: REQ-VERIFY-1365, SCENARIO-VERIFY-1365
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from carnot.embeddings.fast_embedding import HashEmbedding
from carnot.verify.z3_math_verifier import Z3MathVerifier

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260505"
EXPERIMENT_ID = 1365
SCHEMA = "eidoku_csp_neuro_symbolic_verification_probe_v1"
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT / "results" / "experiment_1365_eidoku_csp_neuro_symbolic_verification_probe.json"
)
DEFAULT_CORPUS_PATHS = (
    REPO_ROOT / "data" / "fover_corpus.jsonl",
    REPO_ROOT / "results" / "fover_labeled_steps_v21_multi.json",
    REPO_ROOT / "results" / "fover_corpus_v5.json",
    REPO_ROOT / "results" / "fover_corpus_v5_oracle.json",
)

REQUIRED_ARTIFACT_FIELDS: set[str] = {
    "status",
    "corpus_cases_used",
    "structural_violation_cost_mean",
    "geometric_consistency_rate",
    "symbolic_entailment_rate",
    "csp_feasibility_rate",
    "eidoku_auroc_proxy",
    "ising_correlation",
    "kan_correlation",
    "eidoku_csp_viable",
    "honest_verdict",
}

STRUCTURAL_COST_THRESHOLD = 0.35
GEOMETRIC_COST_THRESHOLD = 0.50
SYMBOLIC_COST_THRESHOLD = 0.25
GEOMETRIC_SIMILARITY_THRESHOLD = 0.05

_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_ASSIGN_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\$?([-+]?\d[\d,]*(?:\.\d+)?)")
_QUANTITY_RE = re.compile(
    r"(?:there\s+(?:are|is|were|was)\s+\$?([-+]?\d[\d,]*(?:\.\d+)?)\s+([A-Za-z][A-Za-z0-9_-]*))"
    r"|(?:\$?([-+]?\d[\d,]*(?:\.\d+)?)\s+([A-Za-z][A-Za-z0-9_-]*)\s+remain)",
    re.IGNORECASE,
)
_DEPENDENCY_MARKERS = (
    "then",
    "therefore",
    "so ",
    "next",
    "after",
    "remaining",
    "total",
    "result",
    "answer",
    "because",
    "since",
)
_STOPWORDS = {
    "and",
    "are",
    "because",
    "for",
    "from",
    "have",
    "into",
    "now",
    "number",
    "that",
    "the",
    "then",
    "there",
    "therefore",
    "this",
    "total",
    "with",
}


@dataclass(frozen=True)
class FoVerCSPCase:
    """One local FoVer case normalized for grammar-free CSP scoring."""

    case_id: str
    question: str
    response: str
    steps: list[str]
    label: int


@dataclass(frozen=True)
class StructuralCost:
    """Reasoning-step graph connectivity summary."""

    cost: float
    node_count: int
    edge_count: int
    largest_component: int


@dataclass(frozen=True)
class GeometricCost:
    """Consecutive-step embedding consistency summary."""

    cost: float
    consistency_rate: float
    mean_similarity: float
    pair_count: int


@dataclass(frozen=True)
class SymbolicCost:
    """Arithmetic entailment summary from Z3MathVerifier or fallback rules."""

    cost: float
    entailed: bool
    claim_count: int
    backend: str


@dataclass(frozen=True)
class ScoredCase:
    """All Eidoku proxy costs for one FoVer case."""

    case_id: str
    label: int
    structural_cost: float
    geometric_cost: float
    geometric_consistency_rate: float
    symbolic_cost: float
    symbolic_entailed: bool
    csp_feasible: bool
    csp_violation_score: float
    ising_score: float
    claim_count: int

    @property
    def geometric_consistent(self) -> bool:
        """Return whether this case passed the geometric cost threshold."""

        return self.geometric_cost <= GEOMETRIC_COST_THRESHOLD


def utc_now_iso() -> str:
    """Return a stable UTC timestamp string for artifacts."""

    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    """Write a deterministic JSON object."""

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
    """Write the mandatory in-progress Exp 1365 artifact before scoring starts."""

    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "run_date": run_date,
        "status": "in_progress",
        "honest_verdict": "in_progress",
        "no_sota_gguf_called": True,
        "started_at": utc_now_iso(),
    }
    write_json(path, artifact)
    return artifact


def load_fover_cases(
    path: Path | str | None = None,
    *,
    limit: int = 100,
) -> list[FoVerCSPCase]:
    """Load local FoVer cases and return a deterministic balanced subset.

    The loader understands the schemas already present in this repository:
    JSONL step rows, JSON arrays of step rows, and dict artifacts containing a
    ``pairs`` list. It never calls a model; it only reads checked-in/local
    corpus artifacts.
    """

    candidate_paths = (Path(path),) if path is not None else DEFAULT_CORPUS_PATHS
    last_cases: list[FoVerCSPCase] = []
    for candidate in candidate_paths:
        if not candidate.exists():
            continue
        rows = _read_rows(candidate)
        cases = _cases_from_rows(rows)
        if not cases:
            continue
        last_cases = cases
        labels = {case.label for case in cases}
        if labels == {0, 1}:
            return _balanced_subset(cases, limit)

    if last_cases:
        return _balanced_subset(last_cases, limit)
    searched = ", ".join(str(p) for p in candidate_paths)
    raise FileNotFoundError(f"no usable local FoVer cases found in: {searched}")


def extract_reasoning_steps(text: str) -> list[str]:
    """Split free-form reasoning text into stable step-like fragments."""

    cleaned = text.replace("\\n", "\n")
    cleaned = re.sub(r"</?think>", "\n", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+(?=(?:Step\s*)?\d+[.)]\s+|Step\s+\d+\s*:)", "\n", cleaned)
    cleaned = re.sub(r"\n\s*[-*]\s+", "\n", cleaned)

    chunks: list[str] = []
    for line in cleaned.splitlines():
        stripped = line.strip(" \t-*")
        if not stripped:
            continue
        if len(stripped) > 320:
            chunks.extend(_split_long_step(stripped))
        else:
            chunks.append(stripped)

    if not chunks:
        fallback = cleaned.strip()
        return [fallback] if fallback else []
    return chunks


def structural_graph_cost(steps: Sequence[str]) -> StructuralCost:
    """Build a dependency graph over reasoning steps and return connectivity cost."""

    nonempty_steps = [step for step in steps if step.strip()]
    n_steps = len(nonempty_steps)
    if n_steps == 0:
        return StructuralCost(cost=1.0, node_count=0, edge_count=0, largest_component=0)
    if n_steps == 1:
        return StructuralCost(cost=0.0, node_count=1, edge_count=0, largest_component=1)

    signatures = [_step_signature(step) for step in nonempty_steps]
    adjacency: list[set[int]] = [set() for _ in range(n_steps)]
    edge_count = 0
    for i in range(n_steps):
        for j in range(i + 1, n_steps):
            if _steps_have_dependency(
                nonempty_steps[i], nonempty_steps[j], signatures[i], signatures[j]
            ):
                adjacency[i].add(j)
                adjacency[j].add(i)
                edge_count += 1

    largest = _largest_component_size(adjacency)
    cost = 1.0 - (largest / n_steps)
    return StructuralCost(
        cost=float(max(0.0, min(1.0, cost))),
        node_count=n_steps,
        edge_count=edge_count,
        largest_component=largest,
    )


def geometric_step_cost(
    steps: Sequence[str],
    *,
    embedder: HashEmbedding | None = None,
    similarity_threshold: float = GEOMETRIC_SIMILARITY_THRESHOLD,
) -> GeometricCost:
    """Score feature-space consistency between consecutive reasoning steps."""

    nonempty_steps = [step for step in steps if step.strip()]
    if len(nonempty_steps) <= 1:
        return GeometricCost(cost=0.0, consistency_rate=1.0, mean_similarity=1.0, pair_count=0)

    local_embedder = embedder or HashEmbedding(embed_dim=96, seed=1365)
    vectors = local_embedder.encode_batch(nonempty_steps)
    similarities: list[float] = []
    for idx in range(len(nonempty_steps) - 1):
        sim = float(np.dot(vectors[idx], vectors[idx + 1]))
        similarities.append(max(-1.0, min(1.0, sim)))

    if not similarities:
        return GeometricCost(cost=0.0, consistency_rate=1.0, mean_similarity=1.0, pair_count=0)
    consistent = [sim >= similarity_threshold for sim in similarities]
    rate = sum(1 for item in consistent if item) / len(consistent)
    return GeometricCost(
        cost=float(1.0 - rate),
        consistency_rate=float(rate),
        mean_similarity=float(sum(similarities) / len(similarities)),
        pair_count=len(similarities),
    )


def symbolic_entailment_cost(
    text: str,
    *,
    verifier: Z3MathVerifier | None = None,
) -> SymbolicCost:
    """Verify extractable arithmetic claims with Z3MathVerifier when possible."""

    local_verifier = verifier or Z3MathVerifier()
    claim_fragments = _claim_fragments(local_verifier, text)
    claim_count = sum(count for _, count in claim_fragments)
    if claim_count > 0:
        weighted_score = 0.0
        for fragment, count in claim_fragments:
            weighted_score += float(local_verifier.score(fragment)) * count
        score = weighted_score / claim_count
        entailed = score <= SYMBOLIC_COST_THRESHOLD
        backend = "z3" if local_verifier.z3_available else "exact_arithmetic"
        return SymbolicCost(
            cost=float(max(0.0, min(1.0, score))),
            entailed=entailed,
            claim_count=claim_count,
            backend=backend,
        )

    fallback_entailed = _rule_based_entailment_fallback(text)
    return SymbolicCost(
        cost=0.25 if fallback_entailed else 0.5,
        entailed=fallback_entailed,
        claim_count=0,
        backend="conservative_rule_fallback",
    )


def score_case(
    case: FoVerCSPCase,
    *,
    embedder: HashEmbedding | None = None,
    verifier: Z3MathVerifier | None = None,
) -> ScoredCase:
    """Compute all three Eidoku proxy costs and the per-case CSP verdict."""

    structural = structural_graph_cost(case.steps)
    geometric = geometric_step_cost(case.steps, embedder=embedder)
    symbolic = symbolic_entailment_cost(case.response, verifier=verifier)
    feasible = (
        structural.cost <= STRUCTURAL_COST_THRESHOLD
        and geometric.cost <= GEOMETRIC_COST_THRESHOLD
        and symbolic.cost <= SYMBOLIC_COST_THRESHOLD
    )
    violation_score = (structural.cost + geometric.cost + symbolic.cost) / 3.0
    return ScoredCase(
        case_id=case.case_id,
        label=case.label,
        structural_cost=structural.cost,
        geometric_cost=geometric.cost,
        geometric_consistency_rate=geometric.consistency_rate,
        symbolic_cost=symbolic.cost,
        symbolic_entailed=symbolic.entailed,
        csp_feasible=feasible,
        csp_violation_score=float(violation_score),
        ising_score=structural.cost,
        claim_count=symbolic.claim_count,
    )


def build_artifact(
    cases: Sequence[FoVerCSPCase],
    scores: Sequence[ScoredCase],
    *,
    corpus_path: Path | str,
    ising_scores: Sequence[float] | None = None,
    kan_scores: Sequence[float] | None = None,
    kan_score_source: str = "geometric_hash_embedding_fallback",
    started_at: str | None = None,
    duration_s: float = 0.0,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """Build the schema-complete Exp 1365 artifact."""

    if len(cases) != len(scores):
        raise ValueError(f"cases and scores length mismatch: {len(cases)} vs {len(scores)}")

    labels = [int(case.label) for case in cases]
    csp_scores = [float(score.csp_violation_score) for score in scores]
    feasibility = [bool(score.csp_feasible) for score in scores]
    ising_values = (
        list(ising_scores) if ising_scores is not None else [s.ising_score for s in scores]
    )
    kan_values = list(kan_scores) if kan_scores is not None else [s.geometric_cost for s in scores]

    csp_feasibility_rate = _mean([1.0 if item else 0.0 for item in feasibility])
    eidoku_auroc = tie_aware_auroc(labels, csp_scores)
    viable = csp_feasibility_rate > 0.5 and eidoku_auroc > 0.55
    z3_probe = Z3MathVerifier()

    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "run_date": run_date,
        "status": "complete",
        "started_at": started_at or utc_now_iso(),
        "finished_at": utc_now_iso(),
        "duration_s": round(float(duration_s), 3),
        "title": "Eidoku CSP Neuro-Symbolic Verification Probe",
        "spec": ["REQ-VERIFY-1365", "SCENARIO-VERIFY-1365"],
        "corpus_path": str(corpus_path),
        "corpus_cases_used": len(cases),
        "corpus_positive_incorrect": int(sum(labels)),
        "corpus_negative_correct": int(len(labels) - sum(labels)),
        "structural_violation_cost_mean": round(_mean([s.structural_cost for s in scores]), 6),
        "geometric_consistency_rate": round(
            _mean([s.geometric_consistency_rate for s in scores]), 6
        ),
        "symbolic_entailment_rate": round(
            _mean([1.0 if s.symbolic_entailed else 0.0 for s in scores]), 6
        ),
        "csp_feasibility_rate": round(csp_feasibility_rate, 6),
        "eidoku_auroc_proxy": round(eidoku_auroc, 6),
        "ising_correlation": round(
            pearson_r([1.0 if item else 0.0 for item in feasibility], ising_values), 6
        ),
        "kan_correlation": round(
            pearson_r([1.0 if item else 0.0 for item in feasibility], kan_values), 6
        ),
        "eidoku_csp_viable": bool(viable),
        "honest_verdict": _honest_verdict(viable, csp_feasibility_rate, eidoku_auroc),
        "thresholds": {
            "structural_cost": STRUCTURAL_COST_THRESHOLD,
            "geometric_cost": GEOMETRIC_COST_THRESHOLD,
            "symbolic_cost": SYMBOLIC_COST_THRESHOLD,
            "geometric_similarity": GEOMETRIC_SIMILARITY_THRESHOLD,
        },
        "score_sources": {
            "structural": "reasoning_step_graph_connectivity",
            "ising": "structural_graph_connectivity_cost_as_ising_tier_proxy",
            "geometric": "carnot.embeddings.fast_embedding.HashEmbedding",
            "kan": kan_score_source,
            "symbolic": "carnot.verify.z3_math_verifier.Z3MathVerifier",
        },
        "z3_available": bool(z3_probe.z3_available),
        "no_sota_gguf_called": True,
        "models_used": [],
        "claim_extractable_case_rate": round(
            _mean([1.0 if s.claim_count else 0.0 for s in scores]), 6
        ),
    }
    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required artifact fields: {sorted(missing)}")
    if artifact["eidoku_csp_viable"] != (
        artifact["csp_feasibility_rate"] > 0.5 and artifact["eidoku_auroc_proxy"] > 0.55
    ):
        raise ValueError("eidoku_csp_viable does not match the required gate")
    return artifact


def run_experiment(
    *,
    corpus_path: Path | str | None = None,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    limit: int = 100,
    use_kan_adapter: bool = True,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """Run the complete local Exp 1365 probe and persist the artifact."""

    write_in_progress_artifact(output_path, run_date=run_date)
    started_at = utc_now_iso()
    t0 = time.perf_counter()
    cases = load_fover_cases(corpus_path, limit=limit)
    embedder = HashEmbedding(embed_dim=96, seed=1365)
    verifier = Z3MathVerifier()
    scored = [score_case(case, embedder=embedder, verifier=verifier) for case in cases]
    kan_scores, kan_source = _kan_scores(cases, scored, use_kan_adapter=use_kan_adapter)
    resolved_corpus_path = (
        Path(corpus_path) if corpus_path is not None else _first_existing_corpus()
    )
    artifact = build_artifact(
        cases,
        scored,
        corpus_path=resolved_corpus_path,
        ising_scores=[score.ising_score for score in scored],
        kan_scores=kan_scores,
        kan_score_source=kan_source,
        started_at=started_at,
        duration_s=time.perf_counter() - t0,
        run_date=run_date,
    )
    write_json(output_path, artifact)
    return artifact


def tie_aware_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute AUROC with 0.5 tie credit and a neutral single-class fallback."""

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
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


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


def _cases_from_rows(rows: Iterable[Mapping[str, Any]]) -> list[FoVerCSPCase]:
    cases: list[FoVerCSPCase] = []
    for index, row in enumerate(rows):
        label = _label_from_row(row)
        text = _row_text(row)
        if label is None or not text.strip():
            continue
        case_id = str(
            row.get("question_id")
            or row.get("case_id")
            or row.get("id")
            or row.get("question_index")
            or f"case_{index}"
        )
        cases.append(
            FoVerCSPCase(
                case_id=case_id,
                question=str(row.get("question") or row.get("prompt") or ""),
                response=text,
                steps=extract_reasoning_steps(text),
                label=label,
            )
        )
    return cases


def _label_from_row(row: Mapping[str, Any]) -> int | None:
    if "is_correct" in row:
        return 0 if bool(row["is_correct"]) else 1
    if "step_correct" in row:
        return 0 if bool(row["step_correct"]) else 1
    if "correct" in row:
        return 0 if bool(row["correct"]) else 1

    raw = row.get("label")
    if raw is None:
        raw = row.get("verdict") or row.get("z3_label")
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


def _row_text(row: Mapping[str, Any]) -> str:
    return str(row.get("response") or row.get("step_text") or row.get("step") or "").strip()


def _balanced_subset(cases: Sequence[FoVerCSPCase], limit: int) -> list[FoVerCSPCase]:
    if limit <= 0 or len(cases) <= limit:
        return list(cases)
    positives = [idx for idx, case in enumerate(cases) if case.label == 1]
    negatives = [idx for idx, case in enumerate(cases) if case.label == 0]
    if not positives or not negatives:
        return list(cases[:limit])

    target_pos = min(len(positives), max(1, limit // 2))
    target_neg = min(len(negatives), max(1, limit - target_pos))
    selected = set(positives[:target_pos] + negatives[:target_neg])
    for idx in range(len(cases)):
        if len(selected) >= limit:
            break
        selected.add(idx)
    return [cases[idx] for idx in sorted(selected)]


def _split_long_step(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\s{2,}", text)
    chunks = [part.strip() for part in parts if len(part.strip()) > 3]
    return chunks or [text]


def _step_signature(step: str) -> tuple[set[str], set[str]]:
    numbers = {_normal_number(match.group(0)) for match in _NUMBER_RE.finditer(step)}
    variables: set[str] = set()
    for match in _ASSIGN_RE.finditer(step):
        variables.add(match.group(1).lower())
    for match in _QUANTITY_RE.finditer(step):
        unit = match.group(2) or match.group(4)
        if unit:
            variables.add(unit.lower())
    for word in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", step.lower()):
        if word not in _STOPWORDS and any(ch.isdigit() for ch in step):
            variables.add(word)
    return numbers, variables


def _normal_number(raw: str) -> str:
    try:
        return f"{float(raw.replace(',', '')):.8g}"
    except ValueError:
        return raw.replace(",", "")


def _steps_have_dependency(
    left_step: str,
    right_step: str,
    left_signature: tuple[set[str], set[str]],
    right_signature: tuple[set[str], set[str]],
) -> bool:
    left_numbers, left_vars = left_signature
    right_numbers, right_vars = right_signature
    if left_numbers & right_numbers:
        return True
    if left_vars & right_vars:
        return True
    right_lower = f" {right_step.lower()} "
    return any(marker in right_lower for marker in _DEPENDENCY_MARKERS) and bool(
        left_numbers or left_vars or right_numbers or right_vars
    )


def _largest_component_size(adjacency: Sequence[set[int]]) -> int:
    seen: set[int] = set()
    largest = 0
    for start in range(len(adjacency)):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for nxt in adjacency[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        largest = max(largest, size)
    return largest


def _claim_count(verifier: Z3MathVerifier, text: str) -> int:
    try:
        equations = verifier._extract_equations(text)
        comparisons = verifier._extract_comparisons(text)
    except Exception:
        return 0
    return len(equations) + len(comparisons)


def _claim_fragments(verifier: Z3MathVerifier, text: str) -> list[tuple[str, int]]:
    fragments = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]
    if not fragments:
        fragments = [text]

    found: list[tuple[str, int]] = []
    for fragment in fragments:
        count = _claim_count(verifier, fragment)
        if count:
            found.append((fragment, count))

    if found:
        return found

    whole_count = _claim_count(verifier, text)
    return [(text, whole_count)] if whole_count else []


def _rule_based_entailment_fallback(text: str) -> bool:
    final = re.search(
        r"\b(?:answer|total|result)\s*(?:is|=|:)\s*\$?([-+]?\d[\d,]*(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if not final:
        return False
    final_value = _normal_number(final.group(1))
    prior_text = text[: final.start()]
    prior_numbers = {_normal_number(match.group(0)) for match in _NUMBER_RE.finditer(prior_text)}
    return final_value in prior_numbers


def _kan_scores(
    cases: Sequence[FoVerCSPCase],
    scored: Sequence[ScoredCase],
    *,
    use_kan_adapter: bool,
) -> tuple[list[float], str]:
    if not use_kan_adapter:
        return [score.geometric_cost for score in scored], "geometric_hash_embedding_fallback"
    if len({case.label for case in cases}) != 2:
        return [
            score.geometric_cost for score in scored
        ], "geometric_hash_embedding_single_label_fallback"

    try:
        from carnot.verify.and_composition_verifier import SOSKANEnergyV3Adapter

        adapter = SOSKANEnergyV3Adapter()
        examples = [
            {
                "step_text": case.response,
                "label": "incorrect" if case.label == 1 else "correct",
            }
            for case in cases
        ]
        adapter.fit_from_corpus(examples, n_epochs=25, lr=3e-3)
        return [float(adapter.score(case.response)) for case in cases], "SOSKANEnergyV3Adapter"
    except Exception:
        return [
            score.geometric_cost for score in scored
        ], "geometric_hash_embedding_kan_error_fallback"


def _first_existing_corpus() -> Path:
    for path in DEFAULT_CORPUS_PATHS:
        if path.exists():
            return path
    return DEFAULT_CORPUS_PATHS[0]


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(value) for value in values) / len(values))


def _honest_verdict(viable: bool, feasibility_rate: float, auroc: float) -> str:
    if viable:
        return (
            "eidoku_csp_viable_local_fover_probe_no_sota_generation_"
            f"feasibility_{feasibility_rate:.3f}_auroc_{auroc:.3f}"
        )
    return (
        "eidoku_csp_not_viable_local_fover_probe_no_sota_generation_"
        f"feasibility_{feasibility_rate:.3f}_auroc_{auroc:.3f}"
    )


if __name__ == "__main__":  # pragma: no cover
    run_experiment()
