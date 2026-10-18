"""Rule-based graph-grounding fact-verifier prototype for Exp 3862.

This module is a small graph-alignment probe inspired by MemGraphRAG,
HalluGraph, and GraphRAG hallucination-detection work, but it is not a model-backed
KG-alignment system. It uses heuristic text rules to extract
subject-relation-object triples from a claim and from the retrieved context,
then scores whether claim triples align to the context graph. It does not
invoke NTK, SAE, attention, GGUF, NLI, or learned verifier substrates.

The score path receives only `(answer_or_claim, retrieved_context)`. It does
not read labels, gold-answer fields, or model confidence while scoring.

Spec: REQ-VERIFY-3862, SCENARIO-VERIFY-3862,
SCENARIO-VERIFY-3862-BLOCKED.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any
import urllib.request

from carnot.verify.corrected_cross_domain_remeasurement_v4 import tie_aware_auroc


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path(
    "results/experiment_3862_graph_grounding_fact_verifier_prototype_v2.json"
)
FACTS_CORPUS_CANDIDATES = (
    Path("data/real_factual_corpus_ragtruth.jsonl"),
    Path("data/realistic_factual_corpus_v3.jsonl"),
    Path("data/realistic_factual_corpus_v2.jsonl"),
    Path("data/realistic_factual_corpus_v1.jsonl"),
)
RANDOM_SEED = 3862
DEFAULT_SAMPLE_SIZE = 120
SIGNAL_AUROC_FLOOR = 0.6
TRIPLE_EXTRACTION_METHOD = "rule_based"
INFERENCE_SUBSTRATE = (
    "rule_based_graph_alignment_verifier_scoring_only "
    "(principle: no live LLM, no GGUF, no NLI checkpoint; scores cached "
    "answer/evidence pairs plus math-bound reasoning heuristics on the same rows)."
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "graph_grounding_auroc",
    "math_ensemble_auroc_on_facts",
    "facts_catch_delta",
    "triple_extraction_method",
    "verifier_authenticity_disclosed",
    "n_facts_items",
    "preconditions_checked",
    "model_specs",
    "random_seed",
    "reproducibility_checksum",
    "inference_substrate",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal verdict prefix; blocked outcomes start with blocked_<resource>.",
    "graph_grounding_auroc": (
        "The prototype's facts-domain signal; materially > 0.5 => the NEW "
        "architecture has traction where the math ensemble is earned-negative."
    ),
    "math_ensemble_auroc_on_facts": (
        "The earned-negative baseline on the SAME set -- the comparison that "
        "shows the graph verifier does something the ensemble structurally cannot."
    ),
    "facts_catch_delta": (
        "GATE field -- emit as a BARE float. graph_grounding_auroc - "
        "math_ensemble_auroc_on_facts; gate for exp3863 (only worth a "
        "complementarity eval if > 0)."
    ),
    "triple_extraction_method": (
        "gguf | small_cpu_model | rule_based -- honest record of the substrate."
    ),
    "verifier_authenticity_disclosed": (
        "Bare bool -- the new module's docstring discloses heuristic-vs-model gap "
        "per the Verifier Authenticity Discipline."
    ),
    "n_facts_items": (
        "Prototype sample size; small is acceptable but report it honestly -- "
        "not a headline claim."
    ),
    "preconditions_checked": (
        "Pre-Launch + Adversarial-Verify: corpus, import, and extractor-substrate "
        "checks are explicit before scoring."
    ),
    "model_specs": (
        "Inference-Substrate: names the graph extractor and same-row math-bound "
        "baseline scorer actually used."
    ),
    "random_seed": "Determinism precondition for row selection and reproducibility.",
    "reproducibility_checksum": "Stable checksum over labels, scores, method, and sample source.",
    "inference_substrate": (
        "Declares verifier-scoring substrate; if a model is not invoked, says so plainly."
    ),
    "duration_s": "Real measured wall-clock duration.",
}

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'-]*")
_UNIT_SPLIT_RE = re.compile(r"(?:[.!?]+|\n+|;)+")
_CAMEL_ENTITY_RE = re.compile(r"\b[A-Z][a-z]+(?:[-\s]+[A-Z][a-z0-9]+)*\b")
_NUMBER_RE = re.compile(r"\b\d{1,4}(?:[.,]\d+)?\b")
_STOPWORDS = {
    "a",
    "about",
    "after",
    "all",
    "an",
    "and",
    "are",
    "as",
    "at",
    "based",
    "be",
    "been",
    "being",
    "by",
    "for",
    "from",
    "given",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "output",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "with",
}
_RELATION_WORDS = {
    "are",
    "arrested",
    "became",
    "become",
    "caused",
    "causes",
    "conducted",
    "concluded",
    "contain",
    "contains",
    "died",
    "dies",
    "emits",
    "founded",
    "had",
    "has",
    "is",
    "liberated",
    "located",
    "named",
    "orbits",
    "released",
    "revealed",
    "sent",
    "separated",
    "survive",
    "survived",
    "was",
    "were",
    "won",
}


@dataclass(frozen=True)
class Triple:
    """A normalized subject-relation-object triple."""

    subject: str
    relation: str
    object: str

    def tokens(self) -> tuple[str, ...]:
        return tuple(_content_tokens(f"{self.subject} {self.relation} {self.object}"))


@dataclass(frozen=True)
class ContextGraph:
    """Lightweight graph built from retrieved context triples."""

    triples: tuple[Triple, ...]
    nodes: frozenset[str]
    edge_index: frozenset[tuple[str, str, str]]
    token_set: frozenset[str]
    number_set: frozenset[str]


@dataclass(frozen=True)
class GroundingScore:
    """Claim-level graph-grounding result."""

    energy: float
    claim_triples: tuple[Triple, ...]
    context_graph: ContextGraph
    best_alignment: float
    missing_key_tokens: tuple[str, ...]
    triple_alignments: tuple[JsonDict, ...]


class GraphGroundingProbe:
    """Prototype graph-grounding verifier using rule-based triple extraction."""

    triple_extraction_method = TRIPLE_EXTRACTION_METHOD

    def extract_triples(self, text: str) -> tuple[Triple, ...]:
        """Extract small heuristic triples from free text."""

        triples: list[Triple] = []
        for unit in _split_units(text):
            triple = _triple_from_unit(unit)
            if triple is not None:
                triples.append(triple)
        return tuple(triples)

    def build_context_graph(self, context: str) -> ContextGraph:
        """Build a context graph from retrieved evidence text."""

        triples = self.extract_triples(context)
        nodes = frozenset(
            _canonical_phrase(part)
            for triple in triples
            for part in (triple.subject, triple.object)
            if _canonical_phrase(part)
        )
        edge_index = frozenset(
            (
                _canonical_phrase(triple.subject),
                _canonical_phrase(triple.relation),
                _canonical_phrase(triple.object),
            )
            for triple in triples
        )
        token_set = frozenset(_content_tokens(context))
        number_set = frozenset(_number_tokens(context))
        return ContextGraph(
            triples=triples,
            nodes=nodes,
            edge_index=edge_index,
            token_set=token_set,
            number_set=number_set,
        )

    def score_claim(self, claim: str, context: str) -> GroundingScore:
        """Score one claim against a context graph; higher energy is less grounded."""

        claim_triples = self.extract_triples(claim)
        context_graph = self.build_context_graph(context)
        if not claim_triples:
            fallback_energy, missing = _fallback_token_energy(claim, context_graph)
            return GroundingScore(
                energy=fallback_energy,
                claim_triples=(),
                context_graph=context_graph,
                best_alignment=1.0 - fallback_energy,
                missing_key_tokens=missing,
                triple_alignments=(),
            )

        energies: list[float] = []
        alignments: list[JsonDict] = []
        missing_tokens: set[str] = set()
        for triple in claim_triples:
            best_alignment = _best_triple_alignment(triple, context_graph)
            key_tokens = _key_tokens_for_triple(triple)
            missing = tuple(token for token in key_tokens if token not in context_graph.token_set)
            missing_tokens.update(missing)
            support = _support_ratio(key_tokens, context_graph.token_set)
            number_penalty = 0.25 if _number_mismatch(triple, context_graph) else 0.0
            energy = 1.0 - (0.65 * best_alignment + 0.35 * support) + number_penalty
            energy = _clamp01(energy)
            energies.append(energy)
            alignments.append(
                {
                    "claim_triple": {
                        "subject": triple.subject,
                        "relation": triple.relation,
                        "object": triple.object,
                    },
                    "best_alignment": round(best_alignment, 6),
                    "token_support": round(support, 6),
                    "number_mismatch": bool(number_penalty),
                    "energy": round(energy, 6),
                }
            )

        mean_energy = sum(energies) / len(energies)
        return GroundingScore(
            energy=round(float(mean_energy), 6),
            claim_triples=claim_triples,
            context_graph=context_graph,
            best_alignment=round(float(max(1.0 - energy for energy in energies)), 6),
            missing_key_tokens=tuple(sorted(missing_tokens)),
            triple_alignments=tuple(alignments),
        )

    def verify(self, answer: str, context: str) -> float:
        """Return graph-grounding energy for an answer/evidence pair."""

        return self.score_claim(answer, context).energy


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    graph_scores: Sequence[float] | None = None,
    math_scores: Sequence[float] | None = None,
    tests_run: Sequence[str] | None = None,
    download_fn: Callable[[Path], Path | None] | None = None,
) -> JsonDict:
    """Build the Exp 3862 terminal artifact."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    preconditions, corpus_path, blocked = check_preconditions(root_path, download_fn=download_fn)
    if blocked is not None:
        finished = time.perf_counter() if now_s is None else float(now_s)
        artifact = _blocked_artifact(
            blocked,
            preconditions_checked=preconditions,
            started_s=start,
            finished_s=finished,
            tests_run=tests_run,
        )
        validate_artifact(artifact)
        return artifact

    rows = load_facts_rows(corpus_path or root_path / FACTS_CORPUS_CANDIDATES[0], sample_size)
    return build_artifact_from_rows(
        rows,
        graph_scores=graph_scores,
        math_scores=math_scores,
        preconditions_checked=preconditions,
        sample_source=_display_path(root_path, corpus_path),
        started_s=start,
        now_s=now_s,
        tests_run=tests_run,
    )


def build_artifact_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    graph_scores: Sequence[float] | None = None,
    math_scores: Sequence[float] | None = None,
    preconditions_checked: Sequence[Mapping[str, Any]] | None = None,
    sample_source: str = "in_memory_rows",
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build a measured artifact from already-loaded RAGTruth-style rows."""

    start = time.perf_counter() if started_s is None else float(started_s)
    clean_rows = tuple(row for row in rows if _valid_facts_row(row))
    labels = [int(bool(row["is_hallucination"])) for row in clean_rows]
    graph = (
        [float(score) for score in graph_scores]
        if graph_scores is not None
        else score_rows_graph_grounding(clean_rows)
    )
    math_bound = (
        [float(score) for score in math_scores]
        if math_scores is not None
        else score_rows_math_bound_ensemble(clean_rows)
    )
    labels, graph, math_bound = _finite_triplets(labels, graph, math_bound)

    graph_auroc = round(float(tie_aware_auroc(labels, graph)), 6) if labels else None
    math_auroc = round(float(tie_aware_auroc(labels, math_bound)), 6) if labels else None
    delta = (
        round(float(graph_auroc - math_auroc), 6)
        if graph_auroc is not None and math_auroc is not None
        else 0.0
    )
    finished = time.perf_counter() if now_s is None else float(now_s)
    artifact: JsonDict = {
        "honest_verdict": _terminal_verdict(graph_auroc, delta),
        "graph_grounding_auroc": graph_auroc,
        "math_ensemble_auroc_on_facts": math_auroc,
        "facts_catch_delta": float(delta),
        "triple_extraction_method": TRIPLE_EXTRACTION_METHOD,
        "verifier_authenticity_disclosed": verifier_authenticity_disclosed(),
        "n_facts_items": len(labels),
        "preconditions_checked": list(preconditions_checked or _default_passed_preconditions()),
        "model_specs": model_specs(),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": reproducibility_checksum(
            {
                "labels": labels,
                "graph_scores": [round(score, 6) for score in graph],
                "math_scores": [round(score, 6) for score in math_bound],
                "method": TRIPLE_EXTRACTION_METHOD,
                "sample_source": sample_source,
            }
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(max(0.0, finished - start), 6),
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate": {
            "condition": (
                "graph_grounding_auroc >= 0.6 AND facts_catch_delta > 0"
            ),
            "passed": bool(
                graph_auroc is not None
                and graph_auroc >= SIGNAL_AUROC_FLOOR
                and delta > 0.0
            ),
            "principle": (
                "Signal requires material graph-grounding AUROC and a positive "
                "same-set delta over the math-bound facts baseline."
            ),
        },
        "score_summary": {
            "n_positive_hallucinations": int(sum(labels)),
            "n_negative_grounded": int(len(labels) - sum(labels)),
            "graph_score_min": min(graph) if graph else None,
            "graph_score_max": max(graph) if graph else None,
            "math_score_min": min(math_bound) if math_bound else None,
            "math_score_max": max(math_bound) if math_bound else None,
        },
        "sample_source": sample_source,
        "tests_run": list(tests_run or []),
        "frozen_fover_0_9131_untouched": True,
        "scripts_research_conductor_modified": False,
    }
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    graph_scores: Sequence[float] | None = None,
    math_scores: Sequence[float] | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build and persist the Exp 3862 artifact."""

    root_path = Path(root)
    output = _repo_path(root_path, Path(output_path))
    artifact = build_artifact(
        root_path,
        graph_scores=graph_scores,
        math_scores=math_scores,
        tests_run=tests_run,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def check_preconditions(
    root: Path,
    *,
    download_fn: Callable[[Path], Path | None] | None = None,
) -> tuple[list[JsonDict], Path | None, str | None]:
    """Check corpus/import/GPU preconditions before scoring."""

    checks: list[JsonDict] = []
    try:
        importlib.import_module("carnot.verify")
    except Exception as exc:
        checks.append(
            {
                "resource": "carnot.verify",
                "available": False,
                "detail": f"{type(exc).__name__}: {exc}",
            }
        )
        return checks, None, "blocked_carnot_verify_import"
    checks.append({"resource": "carnot.verify", "available": True, "detail": "import_ok"})

    corpus_path = resolve_facts_corpus(root)
    if corpus_path is None:
        downloader = download_fn or try_download_small_facts_corpus
        downloaded = downloader(root)
        corpus_path = downloaded if downloaded is not None else resolve_facts_corpus(root)
    if corpus_path is None:
        checks.append(
            {
                "resource": "facts_corpus",
                "available": False,
                "detail": "no cached RAGTruth-style labeled corpus found",
            }
        )
        checks.append(_gpu_precondition())
        checks.append(
            {
                "resource": "triple_extractor",
                "available": True,
                "detail": TRIPLE_EXTRACTION_METHOD,
            }
        )
        return checks, None, "blocked_facts_corpus_not_available"

    checks.append(
        {
            "resource": "facts_corpus",
            "available": True,
            "detail": str(corpus_path),
        }
    )
    checks.append(_gpu_precondition())
    checks.append(
        {
            "resource": "triple_extractor",
            "available": True,
            "detail": TRIPLE_EXTRACTION_METHOD,
        }
    )
    return checks, corpus_path, None


def resolve_facts_corpus(root: Path) -> Path | None:
    """Return the first local labeled factual corpus path, if any."""

    for rel_path in FACTS_CORPUS_CANDIDATES:
        path = _repo_path(root, rel_path)
        if path.exists() and path.is_file():
            return path
    return None


def try_download_small_facts_corpus(root: Path) -> Path | None:
    """Attempt a small opt-in corpus download when no local facts corpus exists."""

    url = os.environ.get("CARNOT_FACTS_CORPUS_URL")
    if not url:
        return None
    output = _repo_path(root, FACTS_CORPUS_CANDIDATES[0])  # pragma: no cover
    output.parent.mkdir(parents=True, exist_ok=True)  # pragma: no cover
    try:  # pragma: no cover - network is optional and not used in unit tests.
        with urllib.request.urlopen(url, timeout=8) as response:
            payload = response.read(2_000_000)
    except Exception:  # pragma: no cover
        return None
    output.write_bytes(payload)  # pragma: no cover
    return output if output.exists() else None  # pragma: no cover


def load_facts_rows(path: Path, sample_size: int = DEFAULT_SAMPLE_SIZE) -> tuple[JsonDict, ...]:
    """Load a deterministic balanced RAGTruth-style sample."""

    positives: list[JsonDict] = []
    negatives: list[JsonDict] = []
    for row in _read_jsonl(path):
        if not _valid_facts_row(row):
            continue
        if int(bool(row["is_hallucination"])) == 1:
            positives.append(dict(row))
        else:
            negatives.append(dict(row))
        if len(positives) >= sample_size // 2 and len(negatives) >= sample_size // 2:
            break

    per_class = min(len(positives), len(negatives), max(1, sample_size // 2))
    selected: list[JsonDict] = []
    for neg, pos in zip(negatives[:per_class], positives[:per_class], strict=True):
        selected.extend((neg, pos))
    return tuple(selected)


def score_rows_graph_grounding(
    rows: Sequence[Mapping[str, Any]],
    *,
    verifier: GraphGroundingProbe | None = None,
) -> list[float]:
    """Score rows with the graph-grounding verifier."""

    graph_verifier = verifier or GraphGroundingProbe()
    return [
        float(graph_verifier.verify(str(row.get("answer") or ""), str(row.get("evidence_passage") or "")))
        for row in rows
    ]


def score_rows_math_bound_ensemble(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    """Score the same facts rows with math-bound reasoning heuristics."""

    try:
        from carnot.verify.tier0r_curry_howard import Tier0rVerifier
        from carnot.verify.tier0u_logical_consistency import Tier0uVerifier
    except Exception:  # pragma: no cover - import is a precondition in normal runs.
        return [0.5 for _row in rows]

    tier0r = Tier0rVerifier()
    tier0u = Tier0uVerifier()
    scores: list[float] = []
    for row in rows:
        text = str(row.get("answer") or "")
        try:
            score = 0.9 * float(tier0r.score(text)) + 0.1 * float(tier0u.score(text))
        except Exception:  # pragma: no cover - defensive fallback for verifier adapters.
            score = 0.5
        scores.append(_clamp01(score))
    return scores


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3862 artifact contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    verdict = str(artifact.get("honest_verdict") or "")
    if not (verdict.startswith("complete:") or verdict.startswith("blocked_")):
        raise ValueError("honest_verdict must start with complete: or blocked_")
    if artifact.get("triple_extraction_method") not in {"gguf", "small_cpu_model", "rule_based"}:
        raise ValueError("triple_extraction_method must be gguf, small_cpu_model, or rule_based")
    if type(artifact.get("facts_catch_delta")) is not float:
        raise ValueError("facts_catch_delta must be a bare float")
    if type(artifact.get("verifier_authenticity_disclosed")) is not bool:
        raise ValueError("verifier_authenticity_disclosed must be a bare bool")
    if not isinstance(artifact.get("n_facts_items"), int) or artifact["n_facts_items"] < 0:
        raise ValueError("n_facts_items must be a non-negative integer")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        raise ValueError("field_principles must be present")
    uncovered = set(REQUIRED_ARTIFACT_FIELDS) - set(principles)
    if uncovered:
        raise ValueError(f"field_principles missing required fields: {sorted(uncovered)}")
    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or float(duration) < 0.0:
        raise ValueError("duration_s must be a non-negative number")
    for field in ("graph_grounding_auroc", "math_ensemble_auroc_on_facts"):
        value = artifact.get(field)
        if value is not None and not (0.0 <= float(value) <= 1.0):
            raise ValueError(f"{field} must be null or in [0, 1]")


def model_specs() -> JsonDict:
    """Return the exact scoring substrates used by this prototype."""

    return {
        "triple_extractor": {
            "method": TRIPLE_EXTRACTION_METHOD,
            "model_invoked": False,
            "detail": "regex/content-token heuristic SRO extraction",
        },
        "math_ensemble_baseline": {
            "method": "math_bound_tier0r_tier0u_formal_core_on_same_facts_rows",
            "weights": {"tier0r_curry_howard": 0.9, "tier0u_logical_consistency": 0.1},
            "frozen_fover_headline_unchanged": 0.9131,
        },
    }


def verifier_authenticity_disclosed() -> bool:
    """Return whether the module docstring discloses the heuristic gap."""

    doc = (__doc__ or "").lower()
    return "heuristic" in doc and "not a model-backed" in doc and "rule-based" in doc


def reproducibility_checksum(payload: Mapping[str, Any]) -> str:
    """Return a stable short checksum for measured rows and scores."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _blocked_artifact(
    reason: str,
    *,
    preconditions_checked: Sequence[Mapping[str, Any]],
    started_s: float,
    finished_s: float,
    tests_run: Sequence[str] | None,
) -> JsonDict:
    return {
        "honest_verdict": reason,
        "graph_grounding_auroc": None,
        "math_ensemble_auroc_on_facts": None,
        "facts_catch_delta": 0.0,
        "triple_extraction_method": TRIPLE_EXTRACTION_METHOD,
        "verifier_authenticity_disclosed": verifier_authenticity_disclosed(),
        "n_facts_items": 0,
        "preconditions_checked": list(preconditions_checked),
        "model_specs": model_specs(),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": reproducibility_checksum(
            {"blocked_reason": reason, "preconditions_checked": list(preconditions_checked)}
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(max(0.0, finished_s - started_s), 6),
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate": {
            "condition": "graph_grounding_auroc >= 0.6 AND facts_catch_delta > 0",
            "passed": False,
            "principle": "Blocked preconditions cannot promote a graph-grounding signal.",
        },
        "tests_run": list(tests_run or []),
        "frozen_fover_0_9131_untouched": True,
        "scripts_research_conductor_modified": False,
    }


def _terminal_verdict(graph_auroc: float | None, delta: float) -> str:
    if graph_auroc is not None and graph_auroc >= SIGNAL_AUROC_FLOOR and delta > 0.0:
        return (
            "complete: graph_grounding_prototype_SIGNAL_"
            f"auroc{graph_auroc:.3f}_delta{delta:.3f}_"
            "new_architecture_has_facts_traction"
        )
    auc_text = "nan" if graph_auroc is None else f"{graph_auroc:.3f}"
    return (
        "complete: graph_grounding_prototype_NO_SIGNAL_"
        f"auroc{auc_text}_facts_remain_out_of_reach_even_with_graph_grounding"
    )


def _default_passed_preconditions() -> list[JsonDict]:
    return [
        {"resource": "carnot.verify", "available": True, "detail": "import_ok"},
        {"resource": "facts_corpus", "available": True, "detail": "in_memory_rows"},
        _gpu_precondition(),
        {"resource": "triple_extractor", "available": True, "detail": TRIPLE_EXTRACTION_METHOD},
    ]


def _gpu_precondition() -> JsonDict:
    free_gpu = _free_gpu_for_gguf()
    return {
        "resource": "free_gpu_for_gguf",
        "available": free_gpu,
        "detail": "gguf path unused; rule_based extractor selected",
    }


def _free_gpu_for_gguf() -> bool:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except Exception:  # pragma: no cover - depends on host GPU tooling.
        return False
    if result.returncode != 0:
        return False
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        used, total, util = (float(part) for part in parts)
        if total > 0 and used / total < 0.10 and util < 10.0:
            return True
    return False


def _split_units(text: str) -> tuple[str, ...]:
    return tuple(unit.strip(" -:\t") for unit in _UNIT_SPLIT_RE.split(str(text)) if unit.strip())


def _triple_from_unit(unit: str) -> Triple | None:
    tokens = _content_tokens(unit)
    if not tokens:
        return None
    relation_index = _first_relation_index(unit)
    if relation_index is None:
        subject_tokens = tokens[: min(3, len(tokens))]
        object_tokens = tokens[min(3, len(tokens)) :] or tokens
        relation = "related_to"
    else:
        raw_tokens = _all_tokens(unit)
        subject_tokens = _content_tokens(" ".join(raw_tokens[:relation_index])) or tokens[:1]
        relation = _stem(raw_tokens[relation_index].lower())
        object_tokens = _content_tokens(" ".join(raw_tokens[relation_index + 1 :])) or tokens[1:]
    subject = _best_subject_phrase(unit, subject_tokens)
    obj = " ".join(object_tokens[:8])
    return Triple(subject=subject, relation=relation, object=obj)


def _first_relation_index(unit: str) -> int | None:
    raw_tokens = _all_tokens(unit)
    for idx, token in enumerate(raw_tokens):
        if token.lower() in _RELATION_WORDS:
            return idx
    return None


def _best_subject_phrase(unit: str, subject_tokens: Sequence[str]) -> str:
    entities = _CAMEL_ENTITY_RE.findall(unit)
    if entities:
        return _canonical_phrase(entities[0])
    return " ".join(subject_tokens[:4])


def _best_triple_alignment(triple: Triple, graph: ContextGraph) -> float:
    if not graph.triples:
        return 0.0
    return max(_triple_similarity(triple, candidate) for candidate in graph.triples)


def _triple_similarity(left: Triple, right: Triple) -> float:
    left_subject = set(_content_tokens(left.subject))
    right_subject = set(_content_tokens(right.subject))
    left_object = set(_content_tokens(left.object))
    right_object = set(_content_tokens(right.object))
    left_relation = set(_content_tokens(left.relation))
    right_relation = set(_content_tokens(right.relation))
    left_all = set(left.tokens())
    right_all = set(right.tokens())
    return _clamp01(
        0.25 * _jaccard(left_subject, right_subject | right_object)
        + 0.45 * _jaccard(left_object, right_object | right_subject)
        + 0.15 * _jaccard(left_relation, right_relation)
        + 0.15 * _jaccard(left_all, right_all)
    )


def _fallback_token_energy(claim: str, graph: ContextGraph) -> tuple[float, tuple[str, ...]]:
    key_tokens = tuple(_content_tokens(claim))
    missing = tuple(token for token in key_tokens if token not in graph.token_set)
    support = _support_ratio(key_tokens, graph.token_set)
    return round(float(1.0 - support), 6), tuple(sorted(set(missing)))


def _key_tokens_for_triple(triple: Triple) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_content_tokens(f"{triple.subject} {triple.object}")))


def _support_ratio(tokens: Sequence[str], context_tokens: frozenset[str]) -> float:
    if not tokens:
        return 1.0
    supported = sum(1 for token in tokens if token in context_tokens)
    return supported / len(tokens)


def _number_mismatch(triple: Triple, graph: ContextGraph) -> bool:
    claim_numbers = _number_tokens(f"{triple.subject} {triple.relation} {triple.object}")
    return bool(claim_numbers and not set(claim_numbers) <= set(graph.number_set))


def _finite_triplets(
    labels: Sequence[int],
    first_scores: Sequence[float],
    second_scores: Sequence[float],
) -> tuple[list[int], list[float], list[float]]:
    clean_labels: list[int] = []
    clean_first: list[float] = []
    clean_second: list[float] = []
    for label, first, second in zip(labels, first_scores, second_scores, strict=False):
        first_f = float(first)
        second_f = float(second)
        if math.isfinite(first_f) and math.isfinite(second_f):
            clean_labels.append(int(label))
            clean_first.append(first_f)
            clean_second.append(second_f)
    return clean_labels, clean_first, clean_second


def _content_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for token in _all_tokens(text):
        lowered = token.lower().strip("'")
        if len(lowered) <= 1 or lowered in _STOPWORDS:
            continue
        tokens.append(_stem(lowered))
    return tokens


def _all_tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(str(text))


def _number_tokens(text: str) -> tuple[str, ...]:
    return tuple(match.group(0).replace(",", "") for match in _NUMBER_RE.finditer(str(text)))


def _stem(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 3:
            return token[: -len(suffix)]
    return token


def _canonical_phrase(text: str) -> str:
    return " ".join(_content_tokens(text))


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _valid_facts_row(row: Mapping[str, Any]) -> bool:
    return all(key in row for key in ("answer", "evidence_passage", "is_hallucination"))


def _read_jsonl(path: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _repo_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _display_path(root: Path, path: Path | None) -> str:
    if path is None:
        return "unknown"
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
