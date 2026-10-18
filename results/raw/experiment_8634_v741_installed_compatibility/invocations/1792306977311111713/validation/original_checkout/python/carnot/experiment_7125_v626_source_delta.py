"""Build the V626 source, method-map, and local model-cache receipt.

The experiment aggregates public metadata and local file facts. It does not
load a model. A source page records what its publisher states, but it cannot
prove a Carnot execution. Spec refs: REQ-REPORT-7125 and
SCENARIO-REPORT-7125-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import date, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from carnot.experiment_artifacts import atomic_write_json, atomic_write_text
from carnot.inference.sota_models import cached_sota_pair
from carnot.paths import repo_root as find_repo_root


JsonDict = dict[str, Any]

RUN_DATE = "20260907"
SOURCE_CAPTURE_UTC = "2026-09-07T22:21:29Z"
RANDOM_SEED = 7_125_202_609_07
INFERENCE_SUBSTRATE = (
    "aggregation_from_external_primary_sources: source and cache delta; "
    "no model, hardware, product, or ARC execution invoked"
)
INFERENCE_SUBSTRATE_CLASS = "aggregation"
EXECUTION_VENUE = "host"
VENDOR_BOUNDARY = "vendor_claim_not_independent_evidence"
PRODUCT_BOUNDARY = "product_page_not_reproducible_evidence"

REFERENCE_PATH = Path("research-references.md")
RESULT_PATH = Path("results/experiment_7125_v626_source_delta.json")
DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "huggingface" / "hub"
PLANNER_START_MARKER = "<!-- V626-PLANNER-REFRESH-20260907-START -->"
PLANNER_END_MARKER = "<!-- V626-PLANNER-REFRESH-20260907-END -->"
REFERENCE_START_MARKER = "<!-- V626-EXECUTION-DELTA-20260907-START -->"
REFERENCE_END_MARKER = "<!-- V626-EXECUTION-DELTA-20260907-END -->"

REQUIRED_INPUT_PATHS = (
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    REFERENCE_PATH,
    Path("research-hardware-wishlist.md"),
    Path("research-roadmap.yaml"),
    Path("results/experiment_6461_v556_sota_source_and_benchmark_delta.json"),
    Path("results/experiment_7122_v625_sota_ingestion.json"),
    Path("scripts/experiments/experiment_7122_v625_sota_ingestion.py"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/inference/sota_models.py"),
    Path("scripts/adversarial_verify.py"),
    Path("openspec/capabilities/research-reporting/spec.md"),
)
SOURCE_ARTIFACT_PATHS = (
    *REQUIRED_INPUT_PATHS,
    Path("python/carnot/experiment_7125_v626_source_delta.py"),
    Path("scripts/experiments/experiment_7125_v626_source_delta.py"),
    Path("tests/python/test_experiment_7125_v626_source_delta.py"),
)

SOURCE_COLLECTION_FIELDS = (
    "arxiv_rows",
    "openreview_rows",
    "semantic_scholar_rows",
    "huggingface_rows",
    "github_rows",
    "extropic_rows",
    "logical_intelligence_rows",
    "kan_rows",
)
SOURCE_TYPES = (
    "peer_reviewed",
    "preprint",
    "vendor_claim",
    "product_page",
    "repository_discovery",
    "secondary_discovery",
)
MODEL_REPOSITORY_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
# The central registry orders Gemma 26B before Gemma 31B. This map keeps the
# report's mandated order while calling the registry helper with valid indexes.
SOTA_INDEX_TO_ID = {
    0: MODEL_REPOSITORY_IDS[0],
    1: MODEL_REPOSITORY_IDS[2],
    2: MODEL_REPOSITORY_IDS[1],
}
TARGET_TASK_IDS = (
    "exp7129-hardness-controlled-sota-constraint-bank",
    "exp7130-verifier-committed-uncertainty-routing",
    "exp7132-directional-memory-portability-audit",
    "exp7133-wcrg-multiscale-sampler-prototype",
)
VERDICT_CLASSES = (
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
)

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "run_date",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "arxiv_rows",
    "openreview_rows",
    "semantic_scholar_rows",
    "huggingface_rows",
    "github_rows",
    "extropic_rows",
    "logical_intelligence_rows",
    "kan_rows",
    "source_class_rows",
    "publication_date_rows",
    "model_repository_rows",
    "cached_model_rows",
    "task_method_map_rows",
    "deferred_rows",
    "source_pages_are_execution_oracles",
    "reference_append_marker",
    "v626_source_delta_complete_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
ROW_COLLECTION_FIELDS = (
    *SOURCE_COLLECTION_FIELDS,
    "source_class_rows",
    "publication_date_rows",
    "model_repository_rows",
    "cached_model_rows",
    "task_method_map_rows",
    "deferred_rows",
)
URL_COLLECTION_FIELDS = (
    *SOURCE_COLLECTION_FIELDS,
    "source_class_rows",
    "model_repository_rows",
    "task_method_map_rows",
    "deferred_rows",
)

FIELD_PRINCIPLES = {
    "field_principles": "A principle for each field makes the receipt auditable.",
    "preconditions_checked": "Explicit local gates prevent missing inputs from becoming invented evidence.",
    "run_date": "The fixed date closes the scientific publication window.",
    "inference_substrate": "The substrate limits the result to source and cache aggregation.",
    "inference_substrate_class": "The class distinguishes aggregation from a blocked no-run state.",
    "execution_venue": "The host venue prevents an unattached hardware claim.",
    "duration_s": "Measured elapsed time records that the refresh executed.",
    "source_artifact_hashes": "Hashes bind the refresh to its local planning inputs.",
    "rows": "Combined typed rows support an independent consistency check.",
    "arxiv_rows": "Primary paper rows preserve exact titles, dates, and bounded facts.",
    "openreview_rows": "Reviewed venue rows stay separate from preprints and discovery indexes.",
    "semantic_scholar_rows": "Citation queries are discovery receipts, not scientific authority.",
    "huggingface_rows": "Community paper pages remain below their primary sources.",
    "github_rows": "Repository availability does not prove method quality or local fitness.",
    "extropic_rows": "Vendor claims cannot prove local hardware execution, speed, or power.",
    "logical_intelligence_rows": "A product page cannot replace public weights or a runner.",
    "kan_rows": "KAN work stays visible without reopening a repeatedly negative branch.",
    "source_class_rows": "One canonical source identity prevents evidence-class drift and duplication.",
    "publication_date_rows": "Inclusive date checks exclude old and future scientific claims.",
    "model_repository_rows": "Remote metadata records repository state without downloading weights.",
    "cached_model_rows": "Exact local paths, revisions, sizes, and hashes support later reproduction.",
    "task_method_map_rows": "Only a method that changes a named V626 task receives a map.",
    "deferred_rows": "A reasoned deferral preserves useful leads without expanding scope.",
    "source_pages_are_execution_oracles": "False separates publisher receipts from local executions.",
    "reference_append_marker": "Marker counts and a hash make the dated append idempotent.",
    "v626_source_delta_complete_score": "One means all source, cache, map, and append checks agree.",
    "random_seed": "A fixed seed records deterministic row ordering.",
    "reproducibility_checksum": "A stable payload hash detects later evidence or verdict drift.",
    "gate_check_summary": "Exact expected and observed values make a block actionable.",
    "verifier_is_oracle": "False states that this audit does not certify Carnot correctness.",
    "verdict_class": "A closed class separates null, blocked, and disqualified outcomes.",
    "honest_verdict": "A class-aligned prefix gives automation an unambiguous outcome.",
}


def _copy(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [deepcopy(dict(row)) for row in rows]


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str | None:
    """Hash one readable file and preserve absence as a null fact."""

    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return "sha256:" + digest.hexdigest()
    except OSError:
        return None


def valid_direct_url(value: object) -> bool:
    """Accept a direct HTTPS URL with a network host."""

    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    return parsed.scheme == "https" and bool(parsed.netloc)


def publication_date_in_window(value: object, run_date: str) -> bool:
    """Accept scientific dates from 2025 through the fixed run day."""

    if not isinstance(value, str):
        return False
    try:
        observed = date.fromisoformat(value)
        upper = datetime.strptime(run_date, "%Y%m%d").date()
    except ValueError:
        return False
    return date(2025, 1, 1) <= observed <= upper


def deduplicate_source_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Keep one source identity and prefer the strongest evidence class."""

    priority = {
        "peer_reviewed": 0,
        "preprint": 1,
        "vendor_claim": 2,
        "product_page": 2,
        "repository_discovery": 3,
        "secondary_discovery": 4,
    }
    order: list[str] = []
    best: dict[str, JsonDict] = {}
    for raw in rows:
        row = deepcopy(dict(raw))
        source_id = str(row.get("source_id", ""))
        if source_id not in best:
            order.append(source_id)
            best[source_id] = row
        elif priority.get(str(row.get("source_type")), 99) < priority.get(
            str(best[source_id].get("source_type")), 99
        ):
            best[source_id] = row
    return [best[source_id] for source_id in order]


def arxiv_rows() -> list[JsonDict]:
    """Return the bounded query receipt and V626-relevant primary papers."""

    query = {
        "row_id": "arxiv-topic-sweep-20260907",
        "query_url": "https://export.arxiv.org/api/query",
        "query_families": [
            "EBM reasoning",
            "neural constraint satisfaction",
            "Ising",
            "hallucination mitigation",
            "KANs",
            "constrained decoding",
            "accelerated sampling",
            "continual learning",
        ],
        "date_start": "2025-01-01",
        "date_end": "2026-09-07",
        "available": True,
        "accessed_at": SOURCE_CAPTURE_UTC,
        "access_outcome": "http_200_complete",
        "terminal": True,
    }
    records = (
        (
            "2605.18871",
            "Distributional Energy-Based Models for Uncertainty-Aware Structured LLM Reasoning",
            "2026-05-15",
            "A scorer ensemble uses its mean for ranking and standard deviation for targeted regeneration or abstention, while deterministic penalties remain separate.",
            "No public code link appears in the arXiv record.",
            "preprint",
        ),
        (
            "2607.17047",
            "Solver-Hard Is Not Model-Hard: A Hardness-Controlled Diagnostic for LLM Constraint Reasoning",
            "2026-07-19",
            "Matched structural families separate solver effort from model accuracy, and proof-preserving relabeling measures surface sensitivity.",
            "Code, data, and deterministic receipts are public at lucky-verma/solver-hard-is-not-model-hard.",
            "preprint",
        ),
        (
            "2608.14569",
            "Position: Certified Correctness in Neural Constraint Reasoning Requires Symbolic Integration",
            "2026-06-02",
            "The paper argues for instance-level symbolic certification when checking is cheaper than solving under distribution shift.",
            "No public implementation link appears in the arXiv record.",
            "peer_reviewed",
        ),
        (
            "2609.04773",
            "Persistent Teacher Anchoring for Tool-Using Agents",
            "2026-09-04",
            "A proposed tool call reaches the environment only after verification of the complete turn.",
            "No public code link appears in the arXiv record.",
            "peer_reviewed",
        ),
        (
            "2609.05339",
            "Does Your Agent's Memory Survive a Model Upgrade? A Controlled Study of Memory Portability",
            "2026-09-04",
            "The study tests directional writer-reader swaps, fixed schemas, notes, embedding migration, and source-backed repair.",
            "No public code link appears in the arXiv record.",
            "preprint",
        ),
        (
            "2608.31114",
            "Overcoming critical slowing down in frustrated spin systems by learned multiscale sampling",
            "2026-08-31",
            "WCRG learns conditional wavelet distributions and samples recursively from coarse to fine scales.",
            "A related WCRG repository exists, but it is not identified as this paper's release.",
            "preprint",
        ),
        (
            "2607.00170",
            "Scaling Up Thermodynamic AI Models",
            "2026-06-30",
            "High-temperature Gibbs inference schedules are evaluated with both autocorrelation and compute cost.",
            "No public code link appears in the arXiv record.",
            "preprint",
        ),
    )
    rows = [query]
    for paper_id, title, published, fact, code_state, source_type in records:
        row: JsonDict = {
            "source_id": f"arxiv:{paper_id}",
            "title": title,
            "url": f"https://arxiv.org/abs/{paper_id}",
            "publication_date": published,
            "bounded_method_fact": fact,
            "public_code_state": code_state,
            "relevance": "A bounded V626 method or control; the task map decides promotion.",
            "source_type": source_type,
            "available": True,
            "accessed_at": SOURCE_CAPTURE_UTC,
            "access_outcome": "http_200_verified_primary",
            "terminal": True,
        }
        if paper_id == "2607.17047":
            row["public_code_url"] = (
                "https://github.com/lucky-verma/solver-hard-is-not-model-hard"
            )
        rows.append(row)
    return rows


def openreview_rows() -> list[JsonDict]:
    """Return current reviewed constraint and memory controls."""

    records = (
        (
            "oum1txoy1D",
            "FSNet: Feasibility-Seeking Neural Network for Constrained Optimization with Guarantees",
            "2025-09-18",
            "NeurIPS 2025 poster",
            "FSNet puts a differentiable feasibility-seeking step inside its solution procedure.",
        ),
        (
            "MSXbrNExax",
            "Benchmarking Continual Agent Memory for Online Learning, Transfer, and Forgetting",
            "2026-04-11",
            "Lifelong Agents at ICLR 2026 workshop poster",
            "AgentMemoryBench separates online, replay, transfer, repair, and forgetting measurements.",
        ),
    )
    return [
        {
            "source_id": f"openreview:{source_id}",
            "title": title,
            "url": f"https://openreview.net/forum?id={source_id}",
            "publication_date": published,
            "venue_state": venue,
            "bounded_method_fact": fact,
            "public_code_state": "A public project was observed, but no code is adopted by V626.",
            "relevance": "Control only; exact execution remains the authority.",
            "source_type": "peer_reviewed",
            "available": True,
            "accessed_at": SOURCE_CAPTURE_UTC,
            "access_outcome": "reachable_http_challenge_with_verified_index_metadata",
            "terminal": True,
        }
        for source_id, title, published, venue, fact in records
    ]


def semantic_scholar_rows() -> list[JsonDict]:
    """Record separate citation-query outcomes without promoting the index."""

    records = (
        (
            "ARXIV:2507.02092",
            None,
            [],
            False,
            "http_429_rate_limited",
        ),
        (
            "ARXIV:2512.15605",
            8,
            [
                "Distributional Energy-Based Models",
                "LoopUS",
                "Large Language Models Can Take False First Steps",
            ],
            True,
            "http_200_complete",
        ),
    )
    return [
        {
            "source_id": f"semantic-scholar:{paper_id}/citations",
            "paper_id": paper_id,
            "query_url": f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations?limit=100",
            "visible_response_rows": count,
            "relevant_title_samples": titles,
            "source_type": "secondary_discovery",
            "citation_count_claimed_as_authoritative": False,
            "relevance": "The citation index can discover controls but cannot prove a method claim.",
            "available": available,
            "accessed_at": SOURCE_CAPTURE_UTC,
            "access_outcome": outcome,
            "terminal": True,
        }
        for paper_id, count, titles, available, outcome in records
    ]


def huggingface_rows() -> list[JsonDict]:
    """Keep Hugging Face Papers as a secondary discovery surface."""

    return [
        {
            "source_id": f"arxiv:{paper_id}",
            "url": f"https://huggingface.co/papers/{paper_id}",
            "primary_source_url": f"https://arxiv.org/abs/{paper_id}",
            "source_type": "secondary_discovery",
            "community_summary_used_as_evidence": False,
            "relevance": "Discovery duplicate; the arXiv row owns the method fact.",
            "available": True,
            "accessed_at": SOURCE_CAPTURE_UTC,
            "access_outcome": "checked_secondary_index",
            "terminal": True,
        }
        for paper_id in (
            "2605.18871",
            "2607.17047",
            "2608.14569",
            "2609.04773",
            "2609.05339",
            "2608.31114",
        )
    ]


def github_rows() -> list[JsonDict]:
    """Return repository identities while excluding popularity as evidence."""

    records = (
        ("alexiglad/EBT", "19420cbeae655bbf11930219a675ade6897019e8", "2026-04-21T00:53:47Z", "Apache-2.0"),
        ("MOSSLab-MIT/FSNet", "826457df85302da8c7553977ce74a4ee18d1b362", "2026-06-09T21:48:12Z", "MIT"),
        ("lucky-verma/solver-hard-is-not-model-hard", "0762959ad1abe3b3aa9556512c92ef957c6dd1b9", "2026-07-18T11:14:23Z", "Apache-2.0"),
        ("Elempereur/WCRG", "d81bac563fb3825472e807bff9ae4ded81585f4a", "2023-05-29T08:50:04Z", None),
        ("extropic-ai/sparse-transformers", "13051e90df9669be5b8f9f34fb097329fa82f674", "2026-09-03T20:28:22Z", "Apache-2.0"),
    )
    return [
        {
            "source_id": f"github:{repository}",
            "url": f"https://github.com/{repository}",
            "default_branch_revision": revision,
            "pushed_at": pushed_at,
            "license": license_id,
            "source_type": "repository_discovery",
            "stars_used_as_evidence": False,
            "implementation_fitness_claimed": False,
            "relevance": "Code identity only; no repository proves local fitness.",
            "available": True,
            "accessed_at": SOURCE_CAPTURE_UTC,
            "access_outcome": "http_200_complete",
            "terminal": True,
        }
        for repository, revision, pushed_at, license_id in records
    ]


def extropic_rows() -> list[JsonDict]:
    """Keep Z1T inside its first-party vendor evidence boundary."""

    return [{
        "source_id": "extropic:z1t-2026-09-04",
        "title": "Z1T: Sparse Transformer-Like Models for Probabilistic Hardware",
        "url": "https://extropic.ai/writing/z1t",
        "publication_date": "2026-09-04",
        "bounded_method_fact": "The page states a degree-16 graph, a 50 MHz update clock, and an FPGA-plus-Z1 design.",
        "public_code_state": "The page links open weights and a sparse-transformer training repository.",
        "relevance": "Vendor boundary information only; no Z1 device is attached.",
        "source_type": "vendor_claim",
        "claim_boundary_label": VENDOR_BOUNDARY,
        "scientific_evidence_promoted": False,
        "hardware_execution_claimed": False,
        "available": True,
        "accessed_at": SOURCE_CAPTURE_UTC,
        "access_outcome": "http_200_complete",
        "terminal": True,
    }]


def logical_intelligence_rows() -> list[JsonDict]:
    """Keep Kona as a product comparator without reproducibility claims."""

    return [{
        "source_id": "logical-intelligence:kona-1.0",
        "title": "Kona 1.0",
        "url": "https://logicalintelligence.com/kona-ebms-energy-based-models",
        "publication_date": "2026-02-03",
        "bounded_method_fact": "The page describes a proprietary global energy layer beneath generators.",
        "public_code_state": "No public checkpoint, training recipe, or compatible runner was observed.",
        "relevance": "Architecture comparator only.",
        "source_type": "product_page",
        "claim_boundary_label": PRODUCT_BOUNDARY,
        "scientific_evidence_promoted": False,
        "reproducible_local_runner_observed": False,
        "available": True,
        "accessed_at": SOURCE_CAPTURE_UTC,
        "access_outcome": "http_200_complete",
        "terminal": True,
    }]


def kan_rows() -> list[JsonDict]:
    """Record the strongest current KAN audit as a deferred primary source."""

    return [{
        "source_id": "arxiv:2607.15525",
        "title": "Kolmogorov--Arnold Networks for Small Language Models",
        "url": "https://arxiv.org/abs/2607.15525",
        "publication_date": "2026-07-17",
        "bounded_method_fact": "The audit tests KAN-family replacements and reports no consistent benchmark, quality, or latency advantage over strong MLP baselines.",
        "public_code_state": "No public code link appears in the arXiv record.",
        "relevance": "It supports deferral of the KAN lineage for V626.",
        "source_type": "preprint",
        "available": True,
        "accessed_at": SOURCE_CAPTURE_UTC,
        "access_outcome": "http_200_verified_primary",
        "terminal": True,
    }]


def _apply_availability(rows: Sequence[Mapping[str, Any]], available: bool) -> list[JsonDict]:
    copied = _copy(rows)
    if available:
        return copied
    for row in copied:
        row["available"] = False
        row["access_outcome"] = "unavailable_at_execution"
    return copied


def source_collections(route_reachability: Mapping[str, bool]) -> dict[str, list[JsonDict]]:
    """Build every source collection and preserve route failures as rows."""

    builders = {
        "arxiv_rows": arxiv_rows,
        "openreview_rows": openreview_rows,
        "semantic_scholar_rows": semantic_scholar_rows,
        "huggingface_rows": huggingface_rows,
        "github_rows": github_rows,
        "extropic_rows": extropic_rows,
        "logical_intelligence_rows": logical_intelligence_rows,
        "kan_rows": kan_rows,
    }
    return {
        field: _apply_availability(builder(), route_reachability.get(field) is True)
        for field, builder in builders.items()
    }


def source_class_rows(collections: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[JsonDict]:
    """Return one canonical ledger with explicit claim boundaries."""

    candidates: list[JsonDict] = []
    for field in SOURCE_COLLECTION_FIELDS:
        rows = collections[field]
        if field == "arxiv_rows":
            rows = rows[1:]
        for raw in rows:
            row = deepcopy(dict(raw))
            source_type = str(row.get("source_type"))
            if "url" not in row and "query_url" in row:
                row["url"] = row["query_url"]
            boundaries = {
                "peer_reviewed": "Reviewed results do not prove a Carnot execution or result.",
                "preprint": "The paper's result does not establish the mapped Carnot outcome.",
                "vendor_claim": "Vendor statements do not prove local hardware execution, speed, or power.",
                "product_page": "The product page is not a public reproducible baseline.",
                "repository_discovery": "Repository availability does not prove scientific quality or local fitness.",
                "secondary_discovery": "A discovery index cannot override a primary source.",
            }
            row["claim_boundary"] = boundaries[source_type]
            candidates.append(row)
    return deduplicate_source_rows(candidates)


def publication_date_rows(collections: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[JsonDict]:
    """Return date checks for each source used as scientific work."""

    scientific = [
        row
        for field in ("arxiv_rows", "openreview_rows", "kan_rows")
        for row in collections[field]
        if row.get("source_type") in {"peer_reviewed", "preprint"}
    ]
    return [
        {
            "source_id": row["source_id"],
            "publication_date": row["publication_date"],
            "window_start": "2025-01-01",
            "window_end": "2026-09-07",
            "in_window": publication_date_in_window(row["publication_date"], RUN_DATE),
            "terminal": True,
        }
        for row in scientific
    ]


def model_repository_rows() -> list[JsonDict]:
    """Return the execution-time Hugging Face metadata snapshot."""

    records = (
        ("a483e9e6cbd595906af30beda3187c2663a1118c", "2026-04-20T12:42:25Z", 28),
        ("c1ac76e99d5513b141e8adde7288b85c3f9c32ec", "2026-07-17T13:09:42Z", 30),
        ("c099eb48e663fd284577b04978a94ffccb261841", "2026-07-17T13:00:49Z", 29),
    )
    return [
        {
            "hf_id": hf_id,
            "repository_url": f"https://huggingface.co/{hf_id}",
            "metadata_url": f"https://huggingface.co/api/models/{hf_id}?blobs=true",
            "revision": revision,
            "updated_at": updated_at,
            "remote_gguf_candidate_count": count,
            "access_outcome": "http_200_complete",
            "download_performed": False,
            "terminal": True,
        }
        for hf_id, (revision, updated_at, count) in zip(
            MODEL_REPOSITORY_IDS, records, strict=True
        )
    ]


def _language_model_gguf(path: Path) -> bool:
    name = path.name.lower()
    return (
        name.endswith(".gguf")
        and not name.startswith(("mmproj", "mtp-"))
        and "mmproj" not in name
        and all(parent.name.lower() != "mtp" for parent in path.parents)
    )


def _quantization(path: Path) -> str:
    match = re.search(r"(?:UD-)?(Q4_K_M)", path.name, flags=re.IGNORECASE)
    return match.group(1).upper() if match else "unknown"


def _preferred_cached_file(repo_dir: Path) -> Path | None:
    candidates = [
        path
        for path in repo_dir.glob("snapshots/**/*.gguf")
        if path.is_file() and _language_model_gguf(path)
    ]
    preferred = [path for path in candidates if "q4_k_m" in path.name.lower()]
    pool = preferred or candidates
    if not pool:
        return None
    main_ref = repo_dir / "refs" / "main"
    main_revision = main_ref.read_text(encoding="utf-8").strip() if main_ref.is_file() else ""
    return sorted(
        pool,
        key=lambda path: (
            main_revision not in path.parts,
            -path.stat().st_mtime_ns,
            path.as_posix(),
        ),
    )[0]


def _production_cached_paths() -> dict[str, Path]:
    """Call cached_sota_pair twice because its contract returns two models."""

    found: dict[str, Path] = {}
    for indexes in ((0, 2), (1, 0)):
        pair = cached_sota_pair(model_indices=indexes)
        for row in pair or []:
            hf_id = str(row.get("hf_id", ""))
            model_path = row.get("model_path")
            if hf_id in MODEL_REPOSITORY_IDS and isinstance(model_path, str):
                found[hf_id] = Path(model_path)
    return found


def _snapshot_revision(path: Path) -> str | None:
    parts = path.parts
    try:
        return parts[parts.index("snapshots") + 1]
    except (ValueError, IndexError):
        return None


def inspect_cached_models(cache_root: Path) -> list[JsonDict]:
    """Resolve one language-model GGUF per repository without downloads."""

    cache_root = Path(cache_root)
    production = cache_root == DEFAULT_CACHE_ROOT
    production_paths = _production_cached_paths() if production else {}
    rows: list[JsonDict] = []
    for hf_id in MODEL_REPOSITORY_IDS:
        repo_dir = cache_root / f"models--{hf_id.replace('/', '--')}"
        candidate = production_paths.get(hf_id)
        if not production:
            candidate = _preferred_cached_file(repo_dir) if repo_dir.is_dir() else None
        resolution_api = "cached_sota_pair" if production else "private_cache_fixture_scan"
        if candidate is None or not candidate.is_file() or not _language_model_gguf(candidate):
            rows.append(
                {
                    "hf_id": hf_id,
                    "cache_status": "missing",
                    "path": None,
                    "file_name": None,
                    "quantization": None,
                    "size_bytes": None,
                    "sha256": None,
                    "revision": None,
                    "resolution_api": resolution_api,
                    "download_performed": False,
                    "terminal": True,
                }
            )
            continue
        rows.append(
            {
                "hf_id": hf_id,
                "cache_status": "resolved",
                "path": str(candidate.absolute()),
                "file_name": candidate.name,
                "quantization": _quantization(candidate),
                "size_bytes": candidate.stat().st_size,
                "sha256": file_sha256(candidate),
                "revision": _snapshot_revision(candidate),
                "resolution_api": resolution_api,
                "download_performed": False,
                "terminal": True,
            }
        )
    return rows


def task_method_map_rows() -> list[JsonDict]:
    """Map five planner-selected methods to exact V626 tasks."""

    records = (
        (
            "hardness_controlled_constraint_diagnostics",
            "2607.17047",
            "2026-07-19",
            "Match structural families, measure exact solver effort, and add proof-preserving relabels.",
            "Stratifies the SOTA constraint bank without treating solver effort as model difficulty.",
            TARGET_TASK_IDS[0],
            "The paper does not prove local model hardness or a Carnot constraint result.",
        ),
        (
            "distributional_uncertainty_routing",
            "2605.18871",
            "2026-05-15",
            "Use scorer-ensemble spread to route a candidate to accept, retry, or abstain.",
            "Adds bounded uncertainty routing while exact feasibility remains authoritative.",
            TARGET_TASK_IDS[1],
            "Learned uncertainty cannot override an exact rejection or prove routing value.",
        ),
        (
            "persistent_verified_turn_commitment",
            "2609.04773",
            "2026-09-04",
            "Permit an action to execute only after verification of the complete proposed turn.",
            "Changes the commit boundary for model-generated constraint actions.",
            TARGET_TASK_IDS[1],
            "Teacher approval is not an exact constraint certificate or a Carnot result.",
        ),
        (
            "directional_fixed_schema_memory_portability",
            "2609.05339",
            "2026-09-04",
            "Test each writer-reader direction and retain source evidence for repair.",
            "Changes the three-family portability matrix and repair controls.",
            TARGET_TASK_IDS[2],
            "Two sub-10B models and synthetic histories do not prove Carnot portability.",
        ),
        (
            "corrected_learned_multiscale_proposals",
            "2608.31114",
            "2026-08-31",
            "Learn scale-conditional proposals and correct them against exact finite targets.",
            "Changes the small frustrated-Ising proposal prototype.",
            TARGET_TASK_IDS[3],
            "The paper does not prove Carnot parity, mixing, scaling, or hardware speed.",
        ),
    )
    return [
        {
            "method_id": method_id,
            "source_id": f"arxiv:{paper_id}",
            "source_url": f"https://arxiv.org/abs/{paper_id}",
            "publication_date": published,
            "bounded_method_fact": fact,
            "public_code_state": (
                "Public code and deterministic receipts are linked by arXiv."
                if paper_id == "2607.17047"
                else "No verified task-ready public implementation is adopted."
            ),
            "relevance": relevance,
            "target_task_id": target,
            "decision_changing": True,
            "already_in_v626_planner": True,
            "execution_delta_new": False,
            "claim_boundary": boundary,
            "carnot_result_claimed": False,
            "terminal": True,
        }
        for method_id, paper_id, published, fact, relevance, target, boundary in records
    ]


def deferred_rows() -> list[JsonDict]:
    """Preserve relevant controls that do not change the V626 contract."""

    records = (
        ("arxiv:2608.14569", "https://arxiv.org/abs/2608.14569", "symbolic certification", "It reinforces exact authority already built into Exp7130; no separate method cell is needed."),
        ("arxiv:2607.00170", "https://arxiv.org/abs/2607.00170", "matched-cost Gibbs scheduling", "Exp7134 already records autocorrelation and matched compute; the image-classification result adds no task."),
        ("arxiv:2603.20801", "https://arxiv.org/abs/2603.20801", "neural large-neighborhood search", "V626 first needs a frozen exact action bank, so neural destroy-repair search remains later work."),
        ("arxiv:2512.05439", "https://arxiv.org/abs/2512.05439", "BEAVER output-space bounds", "A finite GGUF sample is not a sound output-space bound, so this remains a future tail-risk control."),
        ("arxiv:2607.15525", "https://arxiv.org/abs/2607.15525", "KAN language-model audit", "No consistent benchmark or latency advantage reverses Carnot's earlier KAN negatives."),
        ("openreview:oum1txoy1D", "https://openreview.net/forum?id=oum1txoy1D", "FSNet feasibility seeking", "V626 already keeps exact feasibility before learned routing."),
        ("semantic-scholar:ARXIV:2507.02092/citations", "https://api.semanticscholar.org/graph/v1/paper/ARXIV:2507.02092/citations?limit=100", "EBT citation trail", "The endpoint was rate-limited, and a citation index cannot establish a new method claim."),
        ("semantic-scholar:ARXIV:2512.15605/citations", "https://api.semanticscholar.org/graph/v1/paper/ARXIV:2512.15605/citations?limit=100", "ARM-EBM citation trail", "The eight visible rows add no new exact local scorer or task."),
        ("extropic:z1t-2026-09-04", "https://extropic.ai/writing/z1t", "Z1T deployment", "No authenticated Z1 device is attached, so hardware execution and performance stay deferred."),
        ("logical-intelligence:kona-1.0", "https://logicalintelligence.com/kona-ebms-energy-based-models", "Kona architecture", "No public checkpoint, training recipe, or compatible local runner was observed."),
    )
    return [
        {
            "source_id": source_id,
            "url": url,
            "lead": lead,
            "target_task_id": None,
            "reason": reason,
            "terminal": True,
        }
        for source_id, url, lead, reason in records
    ]


def combined_rows(artifact: Mapping[str, Any]) -> list[JsonDict]:
    """Combine every typed row and retain its collection identity."""

    rows: list[JsonDict] = []
    for field in ROW_COLLECTION_FIELDS:
        value = artifact.get(field, [])
        if isinstance(value, list):
            rows.extend(
                {"collection": field, **deepcopy(row)}
                for row in value
                if isinstance(row, dict)
            )
    return rows


def http_reachable(url: str) -> bool:
    """Treat any explicit HTTP response as a completed route probe."""

    request = Request(url, headers={"User-Agent": "carnot-v626-source-audit/1.0"})
    try:
        with urlopen(request, timeout=10) as response:
            return int(response.status) > 0
    except HTTPError:
        return True
    except (OSError, URLError):
        return False


def probe_routes() -> dict[str, bool]:
    """Probe source routes without downloading model files."""

    urls = {
        "arxiv_rows": "https://export.arxiv.org/api/query?id_list=2605.18871&max_results=1",
        "openreview_rows": "https://openreview.net/forum?id=oum1txoy1D",
        "semantic_scholar_rows": "https://api.semanticscholar.org/graph/v1/paper/ARXIV:2512.15605",
        "huggingface_rows": "https://huggingface.co/api/daily_papers?limit=1",
        "github_rows": "https://api.github.com/repos/alexiglad/EBT",
        "extropic_rows": "https://extropic.ai/writing/z1t",
        "logical_intelligence_rows": "https://logicalintelligence.com/kona-ebms-energy-based-models",
        "kan_rows": "https://arxiv.org/abs/2607.15525",
    }
    return {name: http_reachable(url) for name, url in urls.items()}


def _readable_nonempty(path: Path) -> bool:
    try:
        return path.is_file() and bool(path.read_bytes())
    except OSError:
        return False


def _writable_target(path: Path) -> bool:
    parent = path.parent
    if not parent.is_dir() or not os.access(parent, os.W_OK):
        return False
    try:
        with tempfile.NamedTemporaryFile(dir=parent, prefix=".exp7125-", delete=True):
            return True
    except OSError:
        return False


def check_preconditions(
    root: Path,
    output_path: Path,
    cache_root: Path,
    *,
    route_reachability: Mapping[str, bool] | None = None,
) -> list[JsonDict]:
    """Check external routes and the local gates that can block execution."""

    routes = dict(probe_routes() if route_reachability is None else route_reachability)
    available = sorted(name for name in SOURCE_COLLECTION_FIELDS if routes.get(name) is True)
    unavailable = sorted(name for name in SOURCE_COLLECTION_FIELDS if routes.get(name) is not True)
    inputs = [path.as_posix() for path in REQUIRED_INPUT_PATHS if _readable_nonempty(root / path)]
    references = root / REFERENCE_PATH
    planner_present = False
    if _readable_nonempty(references):
        planner_text = references.read_text(encoding="utf-8")
        planner_present = PLANNER_START_MARKER in planner_text and PLANNER_END_MARKER in planner_text
    cache_ok = Path(cache_root).is_dir() and os.access(cache_root, os.R_OK)
    references_writable = _writable_target(references)
    artifact_writable = _writable_target(output_path)
    return [
        {
            "check": "network_access",
            "expected_value": "probe_each_route_and_record_unavailable",
            "observed_value": {"available": available, "unavailable": unavailable},
            "required_for_terminal_artifact": False,
            "passed": True,
        },
        {
            "check": "readable_planning_sources",
            "expected_value": [path.as_posix() for path in REQUIRED_INPUT_PATHS],
            "observed_value": inputs,
            "passed": len(inputs) == len(REQUIRED_INPUT_PATHS),
        },
        {
            "check": "v626_planner_marker",
            "expected_value": [PLANNER_START_MARKER, PLANNER_END_MARKER],
            "observed_value": [PLANNER_START_MARKER, PLANNER_END_MARKER] if planner_present else "missing",
            "passed": planner_present,
        },
        {
            "check": "model_cache_metadata",
            "expected_value": "readable_directory",
            "observed_value": "readable_directory" if cache_ok else "missing_or_unreadable",
            "passed": cache_ok,
        },
        {
            "check": "writable_reference_path",
            "expected_value": "writable_parent_directory",
            "observed_value": "writable_parent_directory" if references_writable else "not_writable",
            "passed": references_writable,
        },
        {
            "check": "writable_artifact_path",
            "expected_value": "writable_parent_directory",
            "observed_value": "writable_parent_directory" if artifact_writable else "not_writable",
            "passed": artifact_writable,
        },
    ]


def append_references_delta(
    path: Path,
    method_rows: Sequence[Mapping[str, Any]],
    cached_rows: Sequence[Mapping[str, Any]],
    run_date: str,
    external_rows: Sequence[Mapping[str, Any]] | None = None,
) -> bool:
    """Append one bounded V626 block and preserve repeated-call bytes."""

    text = path.read_text(encoding="utf-8")
    if REFERENCE_START_MARKER in text or REFERENCE_END_MARKER in text:
        return False
    lines = [
        "",
        f"## V626 execution delta - {datetime.strptime(run_date, '%Y%m%d').date().isoformat()}",
        "",
        REFERENCE_START_MARKER,
        "",
        "This execution refresh confirms five planner-selected method maps.",
        "The refresh found no post-planner method changed the V626 task contract.",
        "It does not report a Carnot result from any source claim.",
        "",
    ]
    for row in method_rows:
        lines.append(
            f"- **{row['method_id']}** - {row['source_url']}. V626 task: "
            f"`{row['target_task_id']}`. Boundary: {row['claim_boundary']}"
        )
    unavailable = [row for row in external_rows or [] if row.get("available") is False]
    lines.extend(["", "Execution-time external access limits:", ""])
    if unavailable:
        for row in unavailable:
            lines.append(
                f"- `{row['source_id']}` - {row['access_outcome']}. "
                "This discovery gap does not invalidate primary receipts."
            )
    else:
        lines.append("- No checked discovery route was unavailable.")
    lines.extend(["", "Execution-time local GGUF cache state:", ""])
    for row in cached_rows:
        if row.get("cache_status") == "resolved":
            state = (
                f"`{row['file_name']}`; {row['quantization']}; {row['size_bytes']} bytes; "
                f"revision {row['revision']}; {row['sha256']}"
            )
        else:
            state = "no local language-model GGUF candidate resolved"
        lines.append(f"- `{row['hf_id']}` - {state}. No download was performed.")
    lines.extend(["", REFERENCE_END_MARKER, ""])
    atomic_write_text(path, text.rstrip() + "\n" + "\n".join(lines), allow_override=False)
    return True


def _reference_marker_receipt(path: Path, appended: bool) -> JsonDict:
    text = path.read_text(encoding="utf-8")
    return {
        "path": REFERENCE_PATH.as_posix(),
        "start_marker": REFERENCE_START_MARKER,
        "end_marker": REFERENCE_END_MARKER,
        "start_count": text.count(REFERENCE_START_MARKER),
        "end_count": text.count(REFERENCE_END_MARKER),
        "post_write_sha256": file_sha256(path),
        "appended": appended,
    }


def _source_hashes(root: Path) -> list[JsonDict]:
    return [
        {"path": path.as_posix(), "sha256": digest, "terminal": True}
        for path in SOURCE_ARTIFACT_PATHS
        if (digest := file_sha256(root / path)) is not None
    ]


def _url_fields_valid(artifact: Mapping[str, Any]) -> bool:
    for field in URL_COLLECTION_FIELDS:
        for row in artifact.get(field, []):
            if not isinstance(row, Mapping):
                return False
            for key, value in row.items():
                if (key == "url" or key.endswith("_url")) and not valid_direct_url(value):
                    return False
    return True


def _coverage_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in SOURCE_COLLECTION_FIELDS:
        rows = artifact.get(field)
        if not isinstance(rows, list) or not rows:
            errors.append(f"missing source collection {field}")
        elif any(
            not isinstance(row, Mapping)
            or row.get("terminal") is not True
            or not isinstance(row.get("available"), bool)
            or not row.get("access_outcome")
            for row in rows
        ):
            errors.append(f"incomplete source receipt {field}")

    classified = artifact.get("source_class_rows")
    if not isinstance(classified, list):
        errors.append("source class rows missing")
        classified = []
    classes = {row.get("source_type") for row in classified if isinstance(row, Mapping)}
    if not set(SOURCE_TYPES).issubset(classes):
        errors.append("source type coverage mismatch")
    ids = [row.get("source_id") for row in classified if isinstance(row, Mapping)]
    if len(ids) != len(set(ids)) or len(ids) != len(classified):
        errors.append("source class canonical duplicate")
    if any(
        not isinstance(row, Mapping)
        or not row.get("claim_boundary")
        or row.get("terminal") is not True
        for row in classified
    ):
        errors.append("source class boundary or terminal receipt missing")
    if not _url_fields_valid(artifact):
        errors.append("direct URL field invalid")

    date_rows = artifact.get("publication_date_rows")
    if not isinstance(date_rows, list) or not date_rows:
        errors.append("publication date rows missing")
    elif any(
        not isinstance(row, Mapping)
        or row.get("in_window") is not True
        or not publication_date_in_window(row.get("publication_date"), RUN_DATE)
        for row in date_rows
    ):
        errors.append("scientific publication date outside window")

    repositories = artifact.get("model_repository_rows")
    if not isinstance(repositories, list) or [
        row.get("hf_id") for row in repositories if isinstance(row, Mapping)
    ] != list(MODEL_REPOSITORY_IDS):
        errors.append("model repository order mismatch")
    elif any(
        row.get("download_performed") is not False
        or not row.get("revision")
        or not row.get("updated_at")
        or not isinstance(row.get("remote_gguf_candidate_count"), int)
        or row.get("terminal") is not True
        for row in repositories
    ):
        errors.append("model repository metadata incomplete")

    cached = artifact.get("cached_model_rows")
    if not isinstance(cached, list) or [
        row.get("hf_id") for row in cached if isinstance(row, Mapping)
    ] != list(MODEL_REPOSITORY_IDS):
        errors.append("cached model coverage mismatch")
    else:
        for row in cached:
            if not isinstance(row, Mapping) or row.get("download_performed") is not False:
                errors.append("cached model download boundary mismatch")
                continue
            if row.get("resolution_api") not in {"cached_sota_pair", "private_cache_fixture_scan"}:
                errors.append("cached model resolver mismatch")
            if row.get("cache_status") == "resolved":
                if (
                    not str(row.get("path", "")).startswith("/")
                    or not str(row.get("file_name", "")).lower().endswith(".gguf")
                    or "mmproj" in str(row.get("file_name", "")).lower()
                    or not row.get("quantization")
                    or not isinstance(row.get("size_bytes"), int)
                    or row.get("size_bytes", 0) <= 0
                    or not str(row.get("sha256", "")).startswith("sha256:")
                    or not row.get("revision")
                ):
                    errors.append("resolved cached model facts incomplete")
            elif row.get("cache_status") == "missing":
                if any(
                    row.get(key) is not None
                    for key in ("path", "file_name", "quantization", "size_bytes", "sha256", "revision")
                ):
                    errors.append("missing cached model row has file facts")
            else:
                errors.append("cached model status invalid")

    if artifact.get("task_method_map_rows") != task_method_map_rows():
        errors.append("task method map mismatch")
    deferred = artifact.get("deferred_rows")
    if deferred != deferred_rows() or any(
        not row.get("reason") or row.get("target_task_id") is not None for row in deferred or []
    ):
        errors.append("deferred row mismatch")
    if artifact.get("source_pages_are_execution_oracles") is not False:
        errors.append("source page execution-oracle boundary mismatch")
    marker = artifact.get("reference_append_marker")
    if not isinstance(marker, Mapping) or (
        marker.get("start_marker") != REFERENCE_START_MARKER
        or marker.get("end_marker") != REFERENCE_END_MARKER
        or marker.get("start_count") != 1
        or marker.get("end_count") != 1
        or not str(marker.get("post_write_sha256", "")).startswith("sha256:")
        or not isinstance(marker.get("appended"), bool)
    ):
        errors.append("reference append marker mismatch")
    return list(dict.fromkeys(errors))


def completion_score(artifact: Mapping[str, Any]) -> int:
    """Return one only when each source, cache, map, and marker check passes."""

    return int(not _coverage_errors(artifact))


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable evidence while excluding elapsed and first-append state."""

    payload = deepcopy(dict(artifact))
    payload.pop("duration_s", None)
    payload.pop("reproducibility_checksum", None)
    marker = payload.get("reference_append_marker")
    if isinstance(marker, dict):
        marker["appended"] = False
    return _sha256_json(payload)


def _base_artifact(preconditions: Sequence[Mapping[str, Any]]) -> JsonDict:
    artifact: JsonDict = {
        field: [] for field in ROW_COLLECTION_FIELDS if field != "rows"
    }
    artifact.update(
        {
            "field_principles": deepcopy(FIELD_PRINCIPLES),
            "preconditions_checked": _copy(preconditions),
            "run_date": RUN_DATE,
            "inference_substrate": INFERENCE_SUBSTRATE,
            "inference_substrate_class": "blocked_no_run",
            "execution_venue": EXECUTION_VENUE,
            "duration_s": 0.0,
            "source_artifact_hashes": [],
            "rows": [],
            "source_pages_are_execution_oracles": False,
            "reference_append_marker": {},
            "v626_source_delta_complete_score": 0,
            "random_seed": RANDOM_SEED,
            "reproducibility_checksum": "",
            "gate_check_summary": {
                "failed_check": None,
                "expected_value": 1,
                "observed_value": 0,
                "passed": False,
            },
            "verifier_is_oracle": False,
            "verdict_class": "blocked",
            "honest_verdict": "blocked_v626_source_delta_precondition",
        }
    )
    return artifact


def recompute_artifact(artifact: JsonDict) -> JsonDict:
    """Recompute combined rows, score, verdict, diagnostic, and checksum."""

    artifact["rows"] = combined_rows(artifact)
    score = completion_score(artifact)
    artifact["v626_source_delta_complete_score"] = score
    if score == 1:
        has_new_delta = any(row.get("execution_delta_new") is True for row in artifact["task_method_map_rows"])
        verdict_class = "positive" if has_new_delta else "null"
        artifact["verdict_class"] = verdict_class
        artifact["honest_verdict"] = (
            "positive_v626_source_delta_complete_with_plan_change"
            if has_new_delta
            else "null_v626_source_delta_complete_no_post_planner_change"
        )
        failed_check = None
    else:
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = "disqualified_v626_source_delta_contract"
        failed_check = "source_delta_contract"
    artifact["gate_check_summary"] = {
        "failed_check": failed_check,
        "expected_value": 1,
        "observed_value": score,
        "passed": score == 1,
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def build_artifact(
    root: Path,
    run_date: str,
    *,
    cache_root: Path,
    output_path: Path = RESULT_PATH,
    route_reachability: Mapping[str, bool] | None = None,
    update_references: bool = False,
    duration_s: float | None = None,
) -> JsonDict:
    """Build one schema-complete null, blocked, or disqualified receipt."""

    started = time.monotonic()
    root = Path(root)
    cache_root = Path(cache_root)
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = root / output_path
    routes = dict(probe_routes() if route_reachability is None else route_reachability)
    preconditions = check_preconditions(
        root,
        output_path,
        cache_root,
        route_reachability=routes,
    )
    artifact = _base_artifact(preconditions)
    artifact["source_artifact_hashes"] = _source_hashes(root)
    failure = next((row for row in preconditions if row["passed"] is False), None)
    if failure is not None:
        artifact["gate_check_summary"] = {
            "failed_check": failure["check"],
            "expected_value": failure["expected_value"],
            "observed_value": failure["observed_value"],
            "passed": False,
        }
    else:
        collections = source_collections(routes)
        cached = inspect_cached_models(cache_root)
        semantic = collections["semantic_scholar_rows"]
        appended = False
        if update_references:
            appended = append_references_delta(
                root / REFERENCE_PATH,
                task_method_map_rows(),
                cached,
                run_date,
                semantic,
            )
        artifact.update(collections)
        artifact.update(
            {
                "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
                "source_class_rows": source_class_rows(collections),
                "publication_date_rows": publication_date_rows(collections),
                "model_repository_rows": model_repository_rows(),
                "cached_model_rows": cached,
                "task_method_map_rows": task_method_map_rows(),
                "deferred_rows": deferred_rows(),
                "reference_append_marker": _reference_marker_receipt(
                    root / REFERENCE_PATH, appended
                ),
                "source_artifact_hashes": _source_hashes(root),
            }
        )
        recompute_artifact(artifact)
    elapsed = duration_s if duration_s is not None else time.monotonic() - started
    artifact["duration_s"] = round(max(0.0, float(elapsed)), 6)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _terminal_errors(artifact: Mapping[str, Any]) -> list[str]:
    preconditions = artifact.get("preconditions_checked")
    if not isinstance(preconditions, list) or not preconditions:
        return ["preconditions_checked missing"]
    failure = next(
        (
            row
            for row in preconditions
            if isinstance(row, Mapping) and row.get("passed") is False
        ),
        None,
    )
    if failure is not None:
        expected_gate = {
            "failed_check": failure.get("check"),
            "expected_value": failure.get("expected_value"),
            "observed_value": failure.get("observed_value"),
            "passed": False,
        }
        errors = []
        if artifact.get("gate_check_summary") != expected_gate:
            errors.append("blocked gate_check_summary mismatch")
        if artifact.get("inference_substrate_class") != "blocked_no_run":
            errors.append("blocked inference_substrate_class mismatch")
        if artifact.get("v626_source_delta_complete_score") != 0:
            errors.append("blocked completion score must be zero")
        if artifact.get("verdict_class") != "blocked":
            errors.append("blocked verdict_class mismatch")
        if not str(artifact.get("honest_verdict", "")).startswith("blocked_"):
            errors.append("blocked honest_verdict prefix mismatch")
        return errors

    coverage_errors = _coverage_errors(artifact)
    expected_score = int(not coverage_errors)
    errors = list(coverage_errors)
    if artifact.get("rows") != combined_rows(artifact):
        errors.append("combined rows mismatch")
    if artifact.get("v626_source_delta_complete_score") != expected_score:
        errors.append("v626_source_delta_complete_score mismatch")
    has_new_delta = any(
        isinstance(row, Mapping) and row.get("execution_delta_new") is True
        for row in artifact.get("task_method_map_rows", [])
    )
    expected_class = ("positive" if has_new_delta else "null") if expected_score == 1 else "disqualified"
    if artifact.get("verdict_class") != expected_class:
        errors.append("verdict_class inconsistent with completion")
    if not str(artifact.get("honest_verdict", "")).startswith(f"{expected_class}_"):
        errors.append("honest_verdict prefix inconsistent with completion")
    expected_gate = {
        "failed_check": None if expected_score == 1 else "source_delta_contract",
        "expected_value": 1,
        "observed_value": expected_score,
        "passed": expected_score == 1,
    }
    if artifact.get("gate_check_summary") != expected_gate:
        errors.append("gate_check_summary inconsistent with completion")
    return errors


def validate_artifact(value: Mapping[str, Any] | str | Path) -> list[str]:
    """Validate field shape, evidence boundaries, state, and checksum."""

    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.is_file():
            return ["artifact_missing"]
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ["artifact_unreadable"]
        if not isinstance(loaded, dict):
            return ["artifact_not_object"]
        artifact = loaded
    elif isinstance(value, Mapping):
        artifact = deepcopy(dict(value))
    else:
        return ["artifact_not_object"]

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    extra = [field for field in artifact if field not in REQUIRED_ARTIFACT_FIELDS]
    if missing or extra:
        return [f"artifact fields mismatch missing={missing} extra={extra}"]

    errors: list[str] = []
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        errors.append("field_principles mismatch")
    if artifact["run_date"] != RUN_DATE:
        errors.append("run_date mismatch")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact["inference_substrate_class"] not in {INFERENCE_SUBSTRATE_CLASS, "blocked_no_run"}:
        errors.append("inference_substrate_class mismatch")
    if artifact["execution_venue"] != EXECUTION_VENUE:
        errors.append("execution_venue mismatch")
    if not isinstance(artifact["duration_s"], (int, float)) or artifact["duration_s"] < 0:
        errors.append("duration_s invalid")
    if artifact["random_seed"] != RANDOM_SEED:
        errors.append("random_seed mismatch")
    if artifact["verifier_is_oracle"] is not False:
        errors.append("verifier_is_oracle mismatch")
    if artifact["verdict_class"] not in VERDICT_CLASSES:
        errors.append("verdict_class invalid")
    errors.extend(_terminal_errors(artifact))
    if artifact["reproducibility_checksum"] != payload_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    return list(dict.fromkeys(errors))


def write_artifact(artifact: Mapping[str, Any], output_path: Path) -> Path:
    """Write only an artifact that passes the independent validator."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    atomic_write_json(Path(output_path), dict(artifact), allow_override=False)
    return Path(output_path)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fixed-date refresh or validate an existing artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)

    if args.validate is not None:
        errors = validate_artifact(args.validate)
        print(json.dumps({"valid": not errors, "errors": errors}, sort_keys=True))
        return int(bool(errors))
    try:
        datetime.strptime(args.date, "%Y%m%d")
    except ValueError:
        return 2
    if args.date != RUN_DATE:
        return 2

    root = find_repo_root()
    output_path = args.output if args.output.is_absolute() else root / args.output
    artifact = build_artifact(
        root,
        args.date,
        cache_root=args.cache_root,
        output_path=output_path,
        update_references=True,
    )
    errors = validate_artifact(artifact)
    if errors:
        print(json.dumps({"valid": False, "errors": errors}, sort_keys=True))
        return 1
    write_artifact(artifact, output_path)
    print(
        json.dumps(
            {
                "artifact": str(output_path),
                "v626_source_delta_complete_score": artifact[
                    "v626_source_delta_complete_score"
                ],
                "verdict_class": artifact["verdict_class"],
                "honest_verdict": artifact["honest_verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - the tested CLI wrapper calls main.
    raise SystemExit(main())
