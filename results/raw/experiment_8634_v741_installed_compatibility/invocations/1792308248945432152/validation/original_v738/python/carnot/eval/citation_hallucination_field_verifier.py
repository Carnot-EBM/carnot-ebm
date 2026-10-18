"""Exp 2932 local citation hallucination field verifier.

Spec: REQ-VERIFY-2932, SCENARIO-VERIFY-2932.

This module treats citations as compact, field-verifiable objects. A local
GGUF model produces short citation-bearing answers from a fixture context, but
the model never judges itself. The deterministic verifier extracts structured
fields, compares them against repository-local fixture truth, and classifies
each citation before aggregate metrics are computed.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot.inference.sota_models import cached_sota_pair, resolve_cached_gguf


JsonDict = dict[str, Any]
RUN_DATE = "20260523"
RANDOM_SEED = 2932
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_FILENAME = "experiment_2932_citation_hallucination_field_verifier_v1.json"
FIXTURE_FILENAME = "experiment_2932_citation_fixture_v1.json"
RAW_RESPONSE_DIRNAME = "citation_hallucination_field_verifier_2932_raw"
INFERENCE_SUBSTRATE = "live_llm_inference_plus_deterministic_verifier"
TAXONOMY_CLASSES = ("real", "potential/ambiguous", "hallucinated-field", "nonexistent-seed")
COMPARABLE_FIELDS = ("title", "authors", "year", "venue", "arxiv_id", "url")
MUTATION_FIELDS = ("title", "authors", "year", "venue", "arxiv_id", "url")
MANDATED_MODEL_IDS: tuple[str, ...] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
MANDATED_MODEL_SPECS: tuple[JsonDict, ...] = (
    {"name": "Qwen3.6-35B-A3B", "hf_id": MANDATED_MODEL_IDS[0], "gpu": 0},
    {"name": "Gemma4-31B-it", "hf_id": MANDATED_MODEL_IDS[1], "gpu": 0},
    {"name": "Gemma4-26B-A4B-it", "hf_id": MANDATED_MODEL_IDS[2], "gpu": 0},
)
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "honest_verdict",
    "citation_verifier_ready",
    "random_seed",
    "reproducibility_checksum",
    "model_specs",
    "models_used",
    "fixture_path",
    "n_citation_cases",
    "extraction_success_rate",
    "field_match_accuracy",
    "hallucination_detection_accuracy",
    "taxonomy_counts",
    "per_case_results",
    "raw_response_dir",
    "inference_substrate",
    "duration_s",
    "run_date",
)

CachedPairProvider = Callable[..., list[JsonDict] | None]
IndividualResolver = Callable[[str], str | None]
CollectModelOutputs = Callable[[JsonDict, list["CitationCase"], "ExperimentConfig"], JsonDict]


@dataclass(frozen=True)
class CitationFields:
    """Structured citation fields used by the deterministic verifier."""

    seed_id: str | None = None
    title: str | None = None
    authors: str | None = None
    year: int | None = None
    venue: str | None = None
    arxiv_id: str | None = None
    url: str | None = None


@dataclass(frozen=True)
class CitationCase:
    """One real, mutated, or nonexistent citation probe row."""

    case_id: str
    truth_seed_id: str | None
    expected_taxonomy: str
    citation: CitationFields
    question: str
    mutation_field: str | None
    source_document: str
    source_found: bool


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime knobs for Exp 2932 paths and local inference limits."""

    output_path: Path | None = None
    fixture_path: Path | None = None
    raw_response_dir: Path | None = None
    max_models: int = 1
    max_tokens: int = 160
    n_ctx: int = 2048
    random_seed: int = RANDOM_SEED
    started_at: float | None = None
    clock: Callable[[], float] = time.time
    monotonic_clock: Callable[[], float] = time.monotonic

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or REPO_ROOT / "results" / OUTPUT_FILENAME

    def fixture_file(self) -> Path:
        return self.fixture_path or REPO_ROOT / "results" / FIXTURE_FILENAME

    def response_dir(self) -> Path:
        return self.raw_response_dir or REPO_ROOT / "results" / RAW_RESPONSE_DIRNAME


def sha256_text(text: str) -> str:
    """Return the SHA-256 digest for a text blob."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_citation_fixture(
    research_references_path: Path | str = REPO_ROOT / "research-references.md",
) -> list[CitationCase]:
    """Build the fixed Exp 2932 citation fixture from repository references."""

    source_text = _read_text(Path(research_references_path))
    seeds = _seed_references()
    cases: list[CitationCase] = []
    for seed in seeds[:4]:
        cases.append(_case_from_seed(seed, "real", None, source_text))
        for field_name in MUTATION_FIELDS:
            cases.append(_case_from_seed(seed, "hallucinated-field", field_name, source_text))
    cases.append(_case_from_seed(seeds[4], "real", None, source_text))
    cases.append(_nonexistent_case(source_text))
    return cases


def prompt_for_case(case: CitationCase) -> str:
    """Return the local-GGUF prompt for one citation probe case."""

    citation_line = format_citation_line(
        {
            "seed_id": case.citation.seed_id,
            "title": case.citation.title,
            "authors": case.citation.authors,
            "year": case.citation.year,
            "venue": case.citation.venue,
            "arxiv_id": case.citation.arxiv_id,
            "url": case.citation.url,
        }
    )
    return (
        "You are answering from a closed local citation fixture. Do not use web search.\n"
        "Write one short sentence answering the question, then put exactly one citation "
        "line after it.\n"
        "Copy the citation fields from the fixture context. Final citation format:\n"
        "CITE[seed_id=<id>; title=<title>; authors=<authors>; year=<year>; "
        "venue=<venue>; arxiv_id=<arxiv_id>; url=<url>]\n"
        f"Question: {case.question}\n"
        f"Fixture citation context:\n{citation_line}\n"
        "Answer:\n"
    )


def format_citation_line(fields: Mapping[str, Any]) -> str:
    """Format citation fields in the parser's canonical line format."""

    parts = [
        ("seed_id", fields.get("seed_id")),
        ("title", fields.get("title")),
        ("authors", fields.get("authors")),
        ("year", fields.get("year")),
        ("venue", fields.get("venue")),
        ("arxiv_id", fields.get("arxiv_id")),
        ("url", fields.get("url")),
    ]
    body = "; ".join(f"{key}={'' if value is None else value}" for key, value in parts)
    return f"CITE[{body}]"


def extract_citations(text: str) -> list[CitationFields]:
    """Extract structured citation fields from raw model output."""

    citations: list[CitationFields] = []
    for obj in _json_objects_from_text(text):
        citations.extend(_citations_from_json_obj(obj))
    for body in re.findall(r"CITE\s*[\[\{](.*?)[\]\}]", text, flags=re.DOTALL | re.IGNORECASE):
        citations.append(_fields_from_mapping(_parse_key_value_body(body)))
    if citations:
        return citations

    match = re.search(r"([A-Z][A-Za-z-]+(?:\s+et\s+al\.)?)\s*\((20\d{2})\)", text)
    if match:
        return [
            CitationFields(
                authors=match.group(1).strip(),
                year=int(match.group(2)),
            )
        ]
    return []


def verify_citation(citation: CitationFields, fixture_cases: Sequence[CitationCase]) -> JsonDict:
    """Classify one extracted citation against fixture truth."""

    truth = _truth_by_seed(fixture_cases)
    matched_seed_id = _matched_seed_id(citation, truth)
    if matched_seed_id is None:
        if citation.seed_id or citation.arxiv_id or citation.url:
            taxonomy = "nonexistent-seed"
        else:
            taxonomy = "potential/ambiguous"
        return _verification_row(taxonomy, None, [], [], 0, 0)

    canonical = truth[matched_seed_id].citation
    matched_fields: list[str] = []
    mismatched_fields: list[str] = []
    missing_fields: list[str] = []
    for field_name in COMPARABLE_FIELDS:
        actual = getattr(citation, field_name)
        expected = getattr(canonical, field_name)
        if actual is None:
            missing_fields.append(field_name)
        elif _field_equal(field_name, actual, expected):
            matched_fields.append(field_name)
        else:
            mismatched_fields.append(field_name)

    if mismatched_fields:
        taxonomy = "hallucinated-field"
    elif missing_fields:
        taxonomy = "potential/ambiguous"
    else:
        taxonomy = "real"
    return _verification_row(
        taxonomy,
        matched_seed_id,
        matched_fields,
        mismatched_fields,
        len(matched_fields),
        len(COMPARABLE_FIELDS),
        missing_fields,
    )


def evaluate_raw_output(
    case: CitationCase,
    raw_output: str,
    fixture_cases: Sequence[CitationCase],
    *,
    generation_metadata: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Evaluate one model response against its expected fixture taxonomy."""

    metadata = dict(generation_metadata or {})
    citations = extract_citations(raw_output)
    if citations:
        verification = verify_citation(citations[0], fixture_cases)
        extraction_success = True
    else:
        verification = _verification_row("potential/ambiguous", None, [], [], 0, 0)
        extraction_success = False

    return {
        "case_id": case.case_id,
        "truth_seed_id": case.truth_seed_id,
        "expected_taxonomy": case.expected_taxonomy,
        "taxonomy": verification["taxonomy"],
        "taxonomy_correct": verification["taxonomy"] == case.expected_taxonomy,
        "extraction_success": extraction_success,
        "matched_seed_id": verification["matched_seed_id"],
        "matched_fields": verification["matched_fields"],
        "mismatched_fields": verification["mismatched_fields"],
        "missing_fields": verification["missing_fields"],
        "field_match_count": verification["field_match_count"],
        "field_comparison_count": verification["field_comparison_count"],
        "raw_output_sha256": sha256_text(raw_output),
        "raw_response_path": metadata.get("raw_response_path", ""),
        "model_hf_id": metadata.get("model_hf_id"),
        "generation_source": metadata.get("generation_source"),
    }


def aggregate_results(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compute aggregate Exp 2932 rates from per-case verification rows."""

    n_rows = len(rows)
    taxonomy_counter = Counter(str(row.get("taxonomy")) for row in rows)
    counts = {label: int(taxonomy_counter.get(label, 0)) for label in TAXONOMY_CLASSES}
    field_comparisons = sum(int(row.get("field_comparison_count") or 0) for row in rows)
    field_matches = sum(int(row.get("field_match_count") or 0) for row in rows)
    return {
        "extraction_success_rate": _safe_rate(
            sum(1 for row in rows if row.get("extraction_success") is True),
            n_rows,
        ),
        "field_match_accuracy": _safe_rate(field_matches, field_comparisons),
        "hallucination_detection_accuracy": _safe_rate(
            sum(1 for row in rows if row.get("taxonomy_correct") is True),
            n_rows,
        ),
        "taxonomy_counts": counts,
    }


def compute_reproducibility_checksum(
    *,
    fixture_cases: Sequence[CitationCase] | Sequence[Mapping[str, Any]],
    model_specs: Sequence[Mapping[str, Any]],
    raw_outputs: Sequence[str],
    random_seed: int = RANDOM_SEED,
) -> str:
    """Hash the fixture, model specs, raw output digests, and seed."""

    payload = {
        "fixture_cases": [_case_to_json(case) for case in fixture_cases],
        "model_specs": list(model_specs),
        "random_seed": random_seed,
        "raw_outputs": list(raw_outputs),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def collect_model_outputs_llamacpp(
    spec: Mapping[str, Any],
    cases: Sequence[CitationCase],
    config: ExperimentConfig,
    *,
    llama_cls: type[Any] | None = None,
    monotonic_clock: Callable[[], float] | None = None,
) -> JsonDict:
    """Run one llama.cpp GGUF model and persist raw citation responses."""

    if llama_cls is None:  # pragma: no cover - covered in live experiment runs.
        from llama_cpp import Llama

        llama_cls = Llama

    clock = monotonic_clock or config.monotonic_clock
    response_dir = config.response_dir()
    response_dir.mkdir(parents=True, exist_ok=True)
    start_all = clock()
    try:
        llm = llama_cls(
            model_path=str(spec["model_path"]),
            n_ctx=config.n_ctx,
            n_gpu_layers=-1,
            main_gpu=int(spec.get("gpu", 0)),
            seed=config.random_seed,
            verbose=False,
        )
    except Exception as exc:  # pragma: no cover - depends on local GGUF runtime.
        return {
            "summary": {
                "hf_id": spec.get("hf_id"),
                "model_name": spec.get("name"),
                "model_path": spec.get("model_path"),
                "model_used": False,
                "blocker": f"llamacpp_load_failed:{type(exc).__name__}:{exc}",
                "live_inference_duration_s": round(max(0.0, clock() - start_all), 6),
            },
            "rows": [],
        }

    rows: list[JsonDict] = []
    for index, case in enumerate(cases):
        prompt = prompt_for_case(case)
        started = clock()
        output = llm(
            prompt,
            max_tokens=config.max_tokens,
            temperature=0.0,
            top_p=1.0,
            repeat_penalty=1.0,
            stop=["\n\n\n"],
        )
        output_text = _llama_output_text(output)
        raw_path = response_dir / f"{case.case_id}_{spec.get('name', 'model')}.txt"
        raw_path.write_text(output_text, encoding="utf-8")
        rows.append(
            {
                "case_id": case.case_id,
                "model_hf_id": spec.get("hf_id"),
                "model_name": spec.get("name"),
                "model_path": spec.get("model_path"),
                "gpu_index": spec.get("gpu"),
                "prompt_hash": sha256_text(prompt),
                "per_case_seed": config.random_seed + index,
                "generation_source": "live_sota_llamacpp_citation",
                "output_text": output_text,
                "raw_response_path": str(raw_path),
                "raw_response_sha256": sha256_text(output_text),
                "elapsed_seconds": round(max(0.0, clock() - started), 6),
                "blocker": None,
            }
        )

    return {
        "summary": {
            "hf_id": spec.get("hf_id"),
            "model_name": spec.get("name"),
            "model_path": spec.get("model_path"),
            "model_used": True,
            "blocker": None,
            "live_inference_duration_s": round(max(0.0, clock() - start_all), 6),
        },
        "rows": rows,
    }


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    cached_pair_provider: CachedPairProvider = cached_sota_pair,
    individual_model_resolver: IndividualResolver = resolve_cached_gguf,
    collect_model_outputs_fn: CollectModelOutputs = collect_model_outputs_llamacpp,
) -> JsonDict:
    """Build the fixture, collect local GGUF outputs, verify, and write JSON."""

    cfg = config or ExperimentConfig()
    started = cfg.start_time()
    fixture_cases = build_citation_fixture()
    _write_json(cfg.fixture_file(), {"schema": "carnot.citation_fixture.v1", "cases": fixture_cases})
    model_specs = _select_model_specs(
        cached_pair_provider,
        individual_model_resolver,
        max_models=cfg.max_models,
    )
    if not model_specs:
        artifact = _blocked_artifact(cfg, fixture_cases, started, cfg.clock())
        _write_json(cfg.artifact_path(), artifact)
        return artifact

    collection_summaries: list[JsonDict] = []
    collected_rows: list[JsonDict] = []
    for spec in model_specs:
        collection = collect_model_outputs_fn(dict(spec), fixture_cases, cfg)
        collection_summaries.append(dict(collection.get("summary") or {}))
        collected_rows.extend(_list_of_mappings(collection.get("rows")))

    rows_by_case = {str(row.get("case_id")): row for row in collected_rows}
    per_case_results: list[JsonDict] = []
    for case in fixture_cases:
        row = rows_by_case.get(case.case_id, {})
        per_case_results.append(
            evaluate_raw_output(
                case,
                str(row.get("output_text") or ""),
                fixture_cases,
                generation_metadata=row,
            )
        )
    metrics = aggregate_results(per_case_results)
    models_used = [
        str(summary["hf_id"])
        for summary in collection_summaries
        if summary.get("model_used") is True and summary.get("hf_id")
    ]
    artifact = _complete_artifact(
        cfg,
        fixture_cases,
        model_specs,
        models_used,
        collection_summaries,
        per_case_results,
        metrics,
        started,
        cfg.clock(),
    )
    _write_json(cfg.artifact_path(), artifact)
    return artifact


def main() -> None:  # pragma: no cover - thin CLI wrapper.
    """Run Exp 2932 with the repository-default output paths."""

    path = run_experiment()["fixture_path"]
    print(f"wrote fixture-bound citation verifier artifact using {path}")


def _seed_references() -> list[CitationFields]:
    return [
        CitationFields(
            seed_id="ebt-2507-02092",
            title="Energy-Based Transformers are Scalable Learners and Thinkers",
            authors="Alex Gladstone et al.",
            year=2026,
            venue="ICLR 2026 Oral",
            arxiv_id="arXiv:2507.02092",
            url="https://arxiv.org/abs/2507.02092",
        ),
        CitationFields(
            seed_id="arm-ebm-2512-15605",
            title="Autoregressive Language Models are Secretly Energy-Based Models",
            authors="Mathieu Blondel et al.",
            year=2025,
            venue="arXiv preprint",
            arxiv_id="arXiv:2512.15605",
            url="https://arxiv.org/abs/2512.15605",
        ),
        CitationFields(
            seed_id="beaver-2512-05439",
            title="BEAVER: An Efficient Deterministic LLM Verifier",
            authors="BEAVER authors",
            year=2025,
            venue="ICLR 2026 VerifAI-2 workshop",
            arxiv_id="arXiv:2512.05439",
            url="https://arxiv.org/abs/2512.05439",
        ),
        CitationFields(
            seed_id="halluguard-2601-18753",
            title="HalluGuard: Demystifying Data-Driven and Reasoning-Driven Hallucinations in LLMs",
            authors="HalluGuard authors",
            year=2026,
            venue="ICLR 2026",
            arxiv_id="arXiv:2601.18753",
            url="https://arxiv.org/abs/2601.18753",
        ),
        CitationFields(
            seed_id="spilled-energy-2602-18671",
            title="Spilled Energy in Large Language Models",
            authors="Spilled Energy authors",
            year=2026,
            venue="ICLR 2026",
            arxiv_id="arXiv:2602.18671",
            url="https://arxiv.org/abs/2602.18671",
        ),
    ]


def _case_from_seed(
    seed: CitationFields,
    expected_taxonomy: str,
    mutation_field: str | None,
    source_text: str,
) -> CitationCase:
    citation = seed if mutation_field is None else _mutated_citation(seed, mutation_field)
    suffix = "real" if mutation_field is None else f"mutated-{mutation_field}"
    return CitationCase(
        case_id=f"{seed.seed_id}:{suffix}",
        truth_seed_id=seed.seed_id,
        expected_taxonomy=expected_taxonomy,
        citation=citation,
        question=(
            "Which fixture reference should be cited for Carnot's local verification "
            f"discussion of {seed.title}?"
        ),
        mutation_field=mutation_field,
        source_document="research-references.md",
        source_found=_source_contains_seed(source_text, seed),
    )


def _mutated_citation(seed: CitationFields, field_name: str) -> CitationFields:
    values = {field: getattr(seed, field) for field in COMPARABLE_FIELDS}
    replacements: dict[str, Any] = {
        "title": f"{seed.title} with Fabricated Appendix",
        "authors": "Imaginary Citation Collective",
        "year": int(seed.year or 0) + 7,
        "venue": "Journal of Synthetic References",
        "arxiv_id": "arXiv:2099.99999",
        "url": "https://example.invalid/nonexistent-citation",
    }
    values[field_name] = replacements[field_name]
    return CitationFields(seed_id=seed.seed_id, **values)


def _nonexistent_case(source_text: str) -> CitationCase:
    citation = CitationFields(
        seed_id="nonexistent-seed-2932",
        title="CiteTracer Energy Fields for Verified Citation Repair",
        authors="A. Fabricator et al.",
        year=2099,
        venue="Proceedings of Imaginary Verification",
        arxiv_id="arXiv:2099.29320",
        url="https://example.invalid/citetracer-energy-fields",
    )
    return CitationCase(
        case_id="nonexistent-seed-2932:fake",
        truth_seed_id=None,
        expected_taxonomy="nonexistent-seed",
        citation=citation,
        question="Which fixture reference supports the fabricated citation-repair claim?",
        mutation_field=None,
        source_document="research-references.md",
        source_found="CiteTracer Energy Fields for Verified Citation Repair" in source_text,
    )


def _source_contains_seed(source_text: str, seed: CitationFields) -> bool:
    return bool(seed.title and seed.title in source_text and _normalize_arxiv(seed.arxiv_id) in source_text)


def _json_objects_from_text(text: str) -> list[Any]:
    decoder = json.JSONDecoder()
    objects: list[Any] = []
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            obj, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        objects.append(obj)
    return objects


def _citations_from_json_obj(obj: Any) -> list[CitationFields]:
    if isinstance(obj, list):
        return [_fields_from_mapping(item) for item in obj if isinstance(item, Mapping)]
    if not isinstance(obj, Mapping):
        return []
    if isinstance(obj.get("citations"), list):
        return [
            _fields_from_mapping(item)
            for item in obj["citations"]
            if isinstance(item, Mapping)
        ]
    if any(key in obj for key in ("seed_id", "title", "authors", "author", "arxiv_id", "url")):
        return [_fields_from_mapping(obj)]
    return []


def _parse_key_value_body(body: str) -> JsonDict:
    parsed: JsonDict = {}
    for part in body.replace("\n", " ").split(";"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        parsed[key.strip().lower()] = value.strip()
    return parsed


def _fields_from_mapping(mapping: Mapping[str, Any]) -> CitationFields:
    return CitationFields(
        seed_id=_string_or_none(mapping.get("seed_id") or mapping.get("citation_id")),
        title=_string_or_none(mapping.get("title")),
        authors=_authors_value(mapping.get("authors") or mapping.get("author")),
        year=_year_value(mapping.get("year")),
        venue=_string_or_none(mapping.get("venue") or mapping.get("booktitle") or mapping.get("journal")),
        arxiv_id=_string_or_none(
            mapping.get("arxiv_id") or mapping.get("arxiv") or mapping.get("doi") or mapping.get("id")
        ),
        url=_string_or_none(mapping.get("url")),
    )


def _matched_seed_id(citation: CitationFields, truth: Mapping[str, CitationCase]) -> str | None:
    if citation.seed_id:
        return citation.seed_id if citation.seed_id in truth else None
    for seed_id, case in truth.items():
        canonical = case.citation
        if citation.arxiv_id and _normalize_arxiv(citation.arxiv_id) == _normalize_arxiv(
            canonical.arxiv_id
        ):
            return seed_id
        if citation.url and _normalize_url(citation.url) == _normalize_url(canonical.url):
            return seed_id
        if citation.title and _normalize_text(citation.title) == _normalize_text(canonical.title):
            return seed_id
    return None


def _truth_by_seed(fixture_cases: Sequence[CitationCase]) -> dict[str, CitationCase]:
    return {
        str(case.truth_seed_id): case
        for case in fixture_cases
        if case.expected_taxonomy == "real" and case.truth_seed_id is not None
    }


def _verification_row(
    taxonomy: str,
    matched_seed_id: str | None,
    matched_fields: Sequence[str],
    mismatched_fields: Sequence[str],
    field_match_count: int,
    field_comparison_count: int,
    missing_fields: Sequence[str] = (),
) -> JsonDict:
    return {
        "taxonomy": taxonomy,
        "matched_seed_id": matched_seed_id,
        "matched_fields": list(matched_fields),
        "mismatched_fields": list(mismatched_fields),
        "missing_fields": list(missing_fields),
        "field_match_count": int(field_match_count),
        "field_comparison_count": int(field_comparison_count),
    }


def _field_equal(field_name: str, actual: Any, expected: Any) -> bool:
    if field_name == "year":
        return _year_value(actual) == _year_value(expected)
    if field_name == "arxiv_id":
        return _normalize_arxiv(actual) == _normalize_arxiv(expected)
    if field_name == "url":
        return _normalize_url(actual) == _normalize_url(expected)
    return _normalize_text(actual) == _normalize_text(expected)


def _normalize_arxiv(value: Any) -> str:
    text = str(value or "").strip().lower()
    match = re.search(r"(\d{4}\.\d{4,5})", text)
    return match.group(1) if match else text


def _normalize_url(value: Any) -> str:
    return str(value or "").strip().lower().rstrip("/")


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _authors_value(value: Any) -> str | None:
    if isinstance(value, list):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    return _string_or_none(value)


def _year_value(value: Any) -> int | None:
    if value is None:
        return None
    match = re.search(r"(19|20)\d{2}", str(value))
    return int(match.group(0)) if match else None


def _safe_rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _llama_output_text(output: Any) -> str:
    if isinstance(output, Mapping):
        choices = output.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, Mapping):
                if choice.get("text") is not None:
                    return str(choice["text"]).strip()
                message = choice.get("message")
                if isinstance(message, Mapping) and message.get("content") is not None:
                    return str(message["content"]).strip()
    return str(output).strip()


def _select_model_specs(
    cached_pair_provider: CachedPairProvider,
    individual_model_resolver: IndividualResolver,
    *,
    max_models: int,
) -> list[JsonDict]:
    cached = cached_pair_provider(gpu_indices=(0, 1))
    selected: list[JsonDict] = []
    if cached:
        for spec in cached:
            if spec.get("hf_id") in MANDATED_MODEL_IDS and spec.get("model_path"):
                selected.append(dict(spec))
            if len(selected) >= max_models:
                return selected

    for spec in MANDATED_MODEL_SPECS:
        model_path = individual_model_resolver(str(spec["hf_id"]))
        if model_path is not None:
            selected.append({**spec, "model_path": model_path})
        if len(selected) >= max_models:
            break
    return selected


def _complete_artifact(
    config: ExperimentConfig,
    fixture_cases: Sequence[CitationCase],
    model_specs: Sequence[Mapping[str, Any]],
    models_used: Sequence[str],
    collection_summaries: Sequence[Mapping[str, Any]],
    per_case_results: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    started: float,
    finished: float,
) -> JsonDict:
    checksum = compute_reproducibility_checksum(
        fixture_cases=fixture_cases,
        model_specs=model_specs,
        raw_outputs=[str(row.get("raw_output_sha256") or "") for row in per_case_results],
        random_seed=config.random_seed,
    )
    ready = bool(models_used) and len(fixture_cases) >= 30
    return {
        "artifact": "experiment_2932_citation_hallucination_field_verifier_v1",
        "schema": "carnot.citation_hallucination_field_verifier.v1",
        "honest_verdict": _honest_verdict(metrics),
        "citation_verifier_ready": ready,
        "random_seed": config.random_seed,
        "reproducibility_checksum": checksum,
        "model_specs": [dict(spec) for spec in model_specs],
        "models_used": list(models_used),
        "model_attempts": [dict(summary) for summary in collection_summaries],
        "fixture_path": str(config.fixture_file()),
        "n_citation_cases": len(fixture_cases),
        "extraction_success_rate": metrics["extraction_success_rate"],
        "field_match_accuracy": metrics["field_match_accuracy"],
        "hallucination_detection_accuracy": metrics["hallucination_detection_accuracy"],
        "taxonomy_counts": dict(metrics["taxonomy_counts"]),
        "per_case_results": [dict(row) for row in per_case_results],
        "raw_response_dir": str(config.response_dir()),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(max(0.0, finished - started), 6),
        "run_date": RUN_DATE,
    }


def _honest_verdict(metrics: Mapping[str, Any]) -> str:
    if metrics["extraction_success_rate"] < 0.5:
        return "complete:low_extraction_success"
    if metrics["hallucination_detection_accuracy"] >= 0.8:
        return "complete:citation_field_verifier_ready"
    return "complete:citation_field_verifier_partial"


def _blocked_artifact(
    config: ExperimentConfig,
    fixture_cases: Sequence[CitationCase],
    started: float,
    finished: float,
) -> JsonDict:
    return {
        "artifact": "experiment_2932_citation_hallucination_field_verifier_v1",
        "schema": "carnot.citation_hallucination_field_verifier.v1",
        "honest_verdict": "blocked_sota_gguf_cache_missing",
        "citation_verifier_ready": False,
        "random_seed": config.random_seed,
        "reproducibility_checksum": compute_reproducibility_checksum(
            fixture_cases=fixture_cases,
            model_specs=[],
            raw_outputs=[],
            random_seed=config.random_seed,
        ),
        "model_specs": [],
        "models_used": [],
        "model_attempts": [],
        "fixture_path": str(config.fixture_file()),
        "n_citation_cases": len(fixture_cases),
        "extraction_success_rate": 0.0,
        "field_match_accuracy": 0.0,
        "hallucination_detection_accuracy": 0.0,
        "taxonomy_counts": {label: 0 for label in TAXONOMY_CLASSES},
        "per_case_results": [],
        "raw_response_dir": str(config.response_dir()),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(max(0.0, finished - started), 6),
        "run_date": RUN_DATE,
    }


def _case_to_json(case: CitationCase | Mapping[str, Any]) -> JsonDict:
    if isinstance(case, Mapping):
        return dict(case)
    return {
        "case_id": case.case_id,
        "truth_seed_id": case.truth_seed_id,
        "expected_taxonomy": case.expected_taxonomy,
        "citation": {
            "seed_id": case.citation.seed_id,
            "title": case.citation.title,
            "authors": case.citation.authors,
            "year": case.citation.year,
            "venue": case.citation.venue,
            "arxiv_id": case.citation.arxiv_id,
            "url": case.citation.url,
        },
        "question": case.question,
        "mutation_field": case.mutation_field,
        "source_document": case.source_document,
        "source_found": case.source_found,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = _json_ready(payload)
    path.write_text(json.dumps(serializable, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json_ready(value: Any) -> Any:
    if isinstance(value, CitationCase):
        return _case_to_json(value)
    if isinstance(value, CitationFields):
        return {
            "seed_id": value.seed_id,
            "title": value.title,
            "authors": value.authors,
            "year": value.year,
            "venue": value.venue,
            "arxiv_id": value.arxiv_id,
            "url": value.url,
        }
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _list_of_mappings(value: Any) -> list[JsonDict]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


if __name__ == "__main__":  # pragma: no cover
    main()
