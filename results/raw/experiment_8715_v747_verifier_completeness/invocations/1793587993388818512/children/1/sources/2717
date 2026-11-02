"""Build the V629 source, method-map, and local model-cache receipt.

The workflow aggregates public metadata and local file facts. It does not load,
download, or generate with a model. Source pages are method receipts. They are
not proof that Carnot ran or reproduced a method. Spec refs: REQ-REPORT-7152
and SCENARIO-REPORT-7152-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from carnot.experiment_7137_v627_source_delta import (
    file_sha256,
    publication_date_in_window,
    valid_direct_url,
)
from carnot.experiment_artifacts import atomic_write_json, atomic_write_text
from carnot.inference.gguf_metadata import read_gguf_metadata
from carnot.inference.sota_models import cached_sota_pair
from carnot.paths import repo_root as find_repo_root


JsonDict = dict[str, Any]
MetadataReader = Callable[[Path], Mapping[str, Any]]
ProgressCallback = Callable[[str], None]

RUN_DATE = "20260909"
SOURCE_CAPTURE_UTC = "2026-09-09T00:00:00Z"
RANDOM_SEED = 7_152_202_609_09
INFERENCE_SUBSTRATE = "aggregation_from_external_primary_sources: V629 source and cache delta"
INFERENCE_SUBSTRATE_CLASS = "aggregation"
EXECUTION_VENUE = "host"

REFERENCE_PATH = Path("research-references.md")
RESULT_PATH = Path("results/experiment_7152_v629_source_delta.json")
DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "huggingface" / "hub"
PLANNER_START_MARKER = "<!-- V629-PLANNER-REFRESH-20260908-START -->"
PLANNER_END_MARKER = "<!-- V629-PLANNER-REFRESH-20260908-END -->"
REFERENCE_START_MARKER = "<!-- V629-EXECUTION-DELTA-20260909-START -->"
REFERENCE_END_MARKER = "<!-- V629-EXECUTION-DELTA-20260909-END -->"

REQUIRED_INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    REFERENCE_PATH,
    Path("research-hardware-wishlist.md"),
    Path("research-roadmap.yaml"),
    Path("results/experiment_7149_v628_source_delta.json"),
    Path("scripts/experiments/experiment_7137_v627_source_delta.py"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/inference/sota_models.py"),
    Path("openspec/capabilities/research-reporting/spec.md"),
)
SOURCE_ARTIFACT_PATHS = (
    *REQUIRED_INPUT_PATHS,
    Path("python/carnot/experiment_7152_v629_source_delta.py"),
    Path("scripts/experiments/experiment_7152_v629_source_delta.py"),
    Path("tests/python/test_experiment_7152_v629_source_delta.py"),
)

SOURCE_COLLECTION_FIELDS = (
    "arxiv_rows",
    "openreview_rows",
    "semantic_scholar_rows",
    "huggingface_rows",
    "github_rows",
    "extropic_rows",
    "logical_intelligence_rows",
)
SOURCE_TYPES = (
    "author_venue_record",
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
TARGET_TASK_IDS = (
    "exp7154-qwen-dual-side-grounding-pilot",
    "exp7158-procedure-family-memory-csl",
    "exp7158-procedure-family-memory-csl",
    "exp7159-memory-drift-poison-cold-audit",
)
VERDICT_CLASSES = (
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
)

MODEL_ROUTE_NAMES = tuple(f"model_repository:{hf_id}" for hf_id in MODEL_REPOSITORY_IDS)
NETWORK_ROUTE_NAMES = (
    "arxiv_rows",
    "openreview_rows",
    "huggingface_rows",
    "github_rows",
    "extropic_rows",
    "logical_intelligence_rows",
    "semantic_scholar_ebt",
    "semantic_scholar_arm",
    *MODEL_ROUTE_NAMES,
)
PROBE_URLS = {
    "arxiv_rows": (
        "https://export.arxiv.org/api/query?search_query=all%3Aenergy-based%20OR%20"
        "all%3AIsing&start=0&max_results=1"
    ),
    "openreview_rows": "https://api2.openreview.net/notes?forum=w696Vhv5B2&limit=1",
    "huggingface_rows": "https://huggingface.co/api/daily_papers?limit=20",
    "github_rows": "https://github.com/trending?since=weekly",
    "extropic_rows": "https://extropic.ai/writing/z1t",
    "logical_intelligence_rows": (
        "https://logicalintelligence.com/kona-ebms-energy-based-models"
    ),
    "semantic_scholar_ebt": (
        "https://api.semanticscholar.org/graph/v1/paper/ARXIV:2507.02092/"
        "citations?limit=1"
    ),
    "semantic_scholar_arm": (
        "https://api.semanticscholar.org/graph/v1/paper/ARXIV:2512.15605/"
        "citations?limit=1"
    ),
    **{
        f"model_repository:{hf_id}": f"https://huggingface.co/api/models/{hf_id}?blobs=true"
        for hf_id in MODEL_REPOSITORY_IDS
    },
}

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
    *SOURCE_COLLECTION_FIELDS,
    "source_class_rows",
    "model_repository_rows",
    "cached_model_rows",
    "task_method_map_rows",
    "deferred_rows",
    "source_pages_are_execution_oracles",
    "reference_append_marker",
    "v629_source_delta_complete_score",
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
    "field_principles": "A principle for every field makes the receipt auditable.",
    "preconditions_checked": "Recorded local checks prevent invented evidence and precede network access.",
    "run_date": "The fixed date closes the allowed publication window.",
    "inference_substrate": "The substrate limits claims to source and cache aggregation.",
    "inference_substrate_class": "The class separates aggregation from a blocked no-run.",
    "execution_venue": "The host venue prevents an unattached hardware claim.",
    "duration_s": "Elapsed time shows that the refresh workflow executed.",
    "source_artifact_hashes": "Hashes bind the output to its local planning inputs.",
    "rows": "Combined typed rows support an independent consistency check.",
    "arxiv_rows": "Primary paper rows preserve dated method facts.",
    "openreview_rows": "Author venue records do not imply review acceptance.",
    "semantic_scholar_rows": "Citation routes remain secondary discovery receipts.",
    "huggingface_rows": "Community paper pages stay below primary sources.",
    "github_rows": "Repository discovery does not prove method fitness.",
    "extropic_rows": "Vendor writing does not prove local hardware execution.",
    "logical_intelligence_rows": "A product page cannot replace a public runner.",
    "source_class_rows": "One canonical identity prevents class drift and duplication.",
    "model_repository_rows": "Repository access records state without downloading weights.",
    "cached_model_rows": "Exact local file and template facts support reproduction.",
    "task_method_map_rows": "Only methods that change a V629 task receive a map.",
    "deferred_rows": "Deferrals preserve useful leads without expanding scope.",
    "source_pages_are_execution_oracles": "False separates receipts from executions.",
    "reference_append_marker": "Marker counts and a hash prove append idempotence.",
    "v629_source_delta_complete_score": "One means all source, cache, map, and append checks agree.",
    "random_seed": "The fixed seed records deterministic row ordering.",
    "reproducibility_checksum": "A stable payload hash detects later evidence drift.",
    "gate_check_summary": "Exact expected and observed values make a block actionable.",
    "verifier_is_oracle": "False states that this audit does not certify correctness.",
    "verdict_class": "A closed class separates null, blocked, and invalid outcomes.",
    "honest_verdict": "A class-aligned prefix gives automation an exact outcome.",
}


def _copy(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [deepcopy(dict(row)) for row in rows]


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _progress(phase: int, message: str) -> None:
    """Emit one unbuffered phase line so the conductor can observe progress."""

    print(f"[exp7152 phase {phase}] {message}", flush=True)


def deduplicate_source_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Keep one source identity and prefer the strongest evidence class."""

    priority = {
        "author_venue_record": 0,
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
    """Return the topic sweep receipt and bounded 2025-2026 paper facts."""

    papers = (
        (
            "2609.02217",
            "SkillGLoW: Procedural-Family Skill Consolidation for Self-Improving Agents on Long-Horizon Task Streams",
            "2026-09-02",
            "The method groups execution lessons into procedural families and gates library admission on execution checks.",
        ),
        (
            "2609.05261",
            "Trace2Tower: Transition-Aware EigenTrace Induction of Multi-Level Skills for LLM Agents",
            "2026-09-04",
            "The method converts traces into event evidence and builds multiple skill levels.",
        ),
        (
            "2609.02265",
            "CAPTURE: Disentangling Preference Drift from Memory Poisoning in Personalized LLM Agents",
            "2026-09-02",
            "The evaluation separates valid preference change from stale or poisoned memory.",
        ),
        (
            "2609.04582",
            "When Do Internal Probes Beat Reading the Answer?",
            "2026-09-04",
            "The paper separates concealed knowledge, threshold error, and absent knowledge.",
        ),
        (
            "2609.00838",
            "Probabilistic Model Checking of Autoregressive Neural Sequence Models",
            "2026-09-01",
            "The method extracts a token-level Markov chain and computes conservative reachability bounds.",
        ),
        (
            "2609.02195",
            "Memory as an Energy Landscape---Hopfield",
            "2026-09-02",
            "The review states the energy, basin, and capacity assumptions of associative memory.",
        ),
        (
            "2608.31114",
            "Overcoming critical slowing down in frustrated spin systems by learned multiscale sampling",
            "2026-08-31",
            "The sampler proposes configurations from learned coarse-to-fine conditional distributions.",
        ),
        (
            "2609.05025",
            "Leveraging Low-Level Symbolic Competences for Unsupervised Grounding in Hallucination Detection",
            "2026-09-04",
            "The method builds source relations and checks sampled responses with SQL execution.",
        ),
        (
            "2609.05281",
            "KanAdapter: A Kolmogorov-Arnold Network-based Plug-and-Play Module for Efficient Fine-tuning of Foundation Speech Models",
            "2026-09-04",
            "Trainable KAN branches adapt speech encoders under a parameter budget.",
        ),
        (
            "2608.26960",
            "Geo-LoRA: Geometry-Aware Subspace Evolution for Low-Rank Adaptation in Continual Learning",
            "2026-08-27",
            "Geometry constraints regulate trainable low-rank updates across tasks.",
        ),
    )
    rows: list[JsonDict] = [
        {
            "row_id": "arxiv-topic-sweep-20260909",
            "query_url": PROBE_URLS["arxiv_rows"],
            "query_families": [
                "EBM reasoning",
                "neural constraints",
                "Ising",
                "hallucination control",
                "KANs",
                "constrained generation",
                "hardware sampling",
                "continual learning",
            ],
            "date_start": "2025-01-01",
            "date_end": "2026-09-09",
            "available": True,
            "accessed_at": SOURCE_CAPTURE_UTC,
            "access_outcome": "execution_route_rechecked",
            "terminal": True,
        }
    ]
    for paper_id, title, published, fact in papers:
        rows.append(
            {
                "source_id": f"arxiv:{paper_id}",
                "title": title,
                "url": f"https://arxiv.org/abs/{paper_id}",
                "publication_date": published,
                "bounded_method_fact": fact,
                "source_type": "preprint",
                "abstract_result_claimed_as_carnot_evidence": False,
                "available": True,
                "accessed_at": SOURCE_CAPTURE_UTC,
                "access_outcome": "execution_route_rechecked",
                "terminal": True,
            }
        )
    return rows


def openreview_rows() -> list[JsonDict]:
    """Return the OptiVer author record without implying acceptance."""

    return [
        {
            "source_id": "openreview:w696Vhv5B2",
            "title": "OptiVer: Unleashing the Power of LLMs for Optimization Modeling via Dual-Side Verification",
            "url": "https://openreview.net/forum?id=w696Vhv5B2",
            "publication_date": "2025-09-20",
            "revised_date": "2026-02-11",
            "venue_state": "ICLR 2026 author submission record; acceptance is not claimed",
            "bounded_method_fact": "The method checks generated model structure and checks a solution against that model.",
            "source_type": "author_venue_record",
            "peer_reviewed_claimed": False,
            "available": True,
            "accessed_at": SOURCE_CAPTURE_UTC,
            "access_outcome": "execution_route_rechecked",
            "terminal": True,
        }
    ]


def semantic_scholar_rows() -> list[JsonDict]:
    """Keep EBT and ARM-EBM citation endpoints as discovery evidence."""

    records = (
        ("ARXIV:2507.02092", "semantic_scholar_ebt", 35, "2026-08-14"),
        ("ARXIV:2512.15605", "semantic_scholar_arm", 8, "2026-07-02"),
    )
    return [
        {
            "source_id": f"semantic-scholar:{paper_id}/citations",
            "paper_id": paper_id,
            "route_name": route,
            "query_url": PROBE_URLS[route].replace("limit=1", "limit=100"),
            "last_complete_visible_rows": count,
            "last_complete_newest_publication_date": newest,
            "fresh_count_claimed_as_authoritative": False,
            "source_type": "secondary_discovery",
            "available": True,
            "accessed_at": SOURCE_CAPTURE_UTC,
            "access_outcome": "execution_route_rechecked",
            "terminal": True,
        }
        for paper_id, route, count, newest in records
    ]


def huggingface_rows() -> list[JsonDict]:
    """Record the paper feed as discovery rather than method evidence."""

    return [
        {
            "source_id": "huggingface:daily-papers-20260909",
            "url": "https://huggingface.co/papers",
            "query_url": PROBE_URLS["huggingface_rows"],
            "decision_changing_method_count": 0,
            "source_type": "secondary_discovery",
            "community_summary_used_as_evidence": False,
            "available": True,
            "accessed_at": SOURCE_CAPTURE_UTC,
            "access_outcome": "execution_route_rechecked",
            "terminal": True,
        },
        {
            "source_id": "arxiv:2609.02217",
            "url": "https://huggingface.co/papers/2609.02217",
            "primary_source_url": "https://arxiv.org/abs/2609.02217",
            "source_type": "secondary_discovery",
            "community_summary_used_as_evidence": False,
            "available": True,
            "accessed_at": SOURCE_CAPTURE_UTC,
            "access_outcome": "execution_route_rechecked",
            "terminal": True,
        },
    ]


def github_rows() -> list[JsonDict]:
    """Record weekly discovery without using popularity as evidence."""

    return [
        {
            "source_id": "github:weekly-trending-20260909",
            "url": PROBE_URLS["github_rows"],
            "languages_checked": ["general", "python", "rust"],
            "decision_changing_repository_count": 0,
            "source_type": "repository_discovery",
            "stars_used_as_evidence": False,
            "implementation_fitness_claimed": False,
            "available": True,
            "accessed_at": SOURCE_CAPTURE_UTC,
            "access_outcome": "execution_route_rechecked",
            "terminal": True,
        }
    ]


def extropic_rows() -> list[JsonDict]:
    """Keep Z1T inside a first-party vendor-claim boundary."""

    return [
        {
            "source_id": "extropic:z1t-2026-09-04",
            "title": "Z1T: Sparse Transformer-Like Models for Probabilistic Hardware",
            "url": PROBE_URLS["extropic_rows"],
            "publication_date": "2026-09-04",
            "bounded_method_fact": "The vendor describes a degree-16 graph and separate FPGA and Z1 timing categories.",
            "source_type": "vendor_claim",
            "scientific_evidence_promoted": False,
            "hardware_execution_claimed": False,
            "available": True,
            "accessed_at": SOURCE_CAPTURE_UTC,
            "access_outcome": "execution_route_rechecked",
            "terminal": True,
        }
    ]


def logical_intelligence_rows() -> list[JsonDict]:
    """Keep Kona as a product comparator without a local result claim."""

    return [
        {
            "source_id": "logical-intelligence:kona-1.0",
            "title": "Kona 1.0",
            "url": PROBE_URLS["logical_intelligence_rows"],
            "publication_date": "2026-02-03",
            "bounded_method_fact": "The page describes a proprietary energy-based constraint layer.",
            "source_type": "product_page",
            "scientific_evidence_promoted": False,
            "reproducible_local_runner_observed": False,
            "available": True,
            "accessed_at": SOURCE_CAPTURE_UTC,
            "access_outcome": "execution_route_rechecked",
            "terminal": True,
        }
    ]


def _route_receipt(value: object) -> JsonDict:
    if isinstance(value, Mapping):
        return deepcopy(dict(value))
    available = value is True
    return {
        "available": available,
        "access_outcome": (
            "execution_route_rechecked" if available else "unavailable_at_execution"
        ),
        "http_status": None,
    }


def _apply_route(rows: Sequence[Mapping[str, Any]], route: object) -> list[JsonDict]:
    receipt = _route_receipt(route)
    copied = _copy(rows)
    for row in copied:
        row["available"] = receipt["available"]
        row["access_outcome"] = receipt["access_outcome"]
        if receipt.get("http_status") is not None:
            row["http_status"] = receipt["http_status"]
    return copied


def source_collections(route_reachability: Mapping[str, object]) -> dict[str, list[JsonDict]]:
    """Build each source collection and keep route failures as terminal rows."""

    collections = {
        "arxiv_rows": _apply_route(arxiv_rows(), route_reachability.get("arxiv_rows")),
        "openreview_rows": _apply_route(
            openreview_rows(), route_reachability.get("openreview_rows")
        ),
        "huggingface_rows": _apply_route(
            huggingface_rows(), route_reachability.get("huggingface_rows")
        ),
        "github_rows": _apply_route(github_rows(), route_reachability.get("github_rows")),
        "extropic_rows": _apply_route(
            extropic_rows(), route_reachability.get("extropic_rows")
        ),
        "logical_intelligence_rows": _apply_route(
            logical_intelligence_rows(), route_reachability.get("logical_intelligence_rows")
        ),
    }
    semantic = []
    for row in semantic_scholar_rows():
        semantic.extend(_apply_route([row], route_reachability.get(str(row["route_name"]))))
    collections["semantic_scholar_rows"] = semantic
    return {field: collections[field] for field in SOURCE_COLLECTION_FIELDS}


def source_class_rows(
    collections: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[JsonDict]:
    """Build one canonical source ledger with explicit claim boundaries."""

    boundaries = {
        "author_venue_record": "An author venue record does not prove acceptance or a Carnot result.",
        "preprint": "The paper does not establish the mapped Carnot outcome.",
        "vendor_claim": "Vendor statements do not prove local hardware results.",
        "product_page": "The product page is not a reproducible public baseline.",
        "repository_discovery": "Repository state does not prove scientific fitness.",
        "secondary_discovery": "A discovery index cannot override a primary source.",
    }
    candidates: list[JsonDict] = []
    for field in SOURCE_COLLECTION_FIELDS:
        for raw in collections[field]:
            if "source_type" not in raw:
                continue
            row = deepcopy(dict(raw))
            if "url" not in row and "query_url" in row:
                row["url"] = row["query_url"]
            row["claim_boundary"] = boundaries[str(row["source_type"])]
            candidates.append(row)
    return deduplicate_source_rows(candidates)


def model_repository_rows(route_reachability: Mapping[str, object]) -> list[JsonDict]:
    """Record repository API access without treating it as model execution."""

    rows = []
    for hf_id in MODEL_REPOSITORY_IDS:
        route = f"model_repository:{hf_id}"
        receipt = _route_receipt(route_reachability.get(route))
        rows.append(
            {
                "hf_id": hf_id,
                "repository_url": f"https://huggingface.co/{hf_id}",
                "metadata_url": PROBE_URLS[route],
                "available": receipt["available"],
                "access_outcome": receipt["access_outcome"],
                "http_status": receipt.get("http_status"),
                "repository_page_is_execution_evidence": False,
                "download_performed": False,
                "terminal": True,
            }
        )
    return rows


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
    """Call the two-model resolver twice to cover all three repositories."""

    found: dict[str, Path] = {}
    for indexes in ((0, 2), (1, 0)):
        pair = cached_sota_pair(model_indices=indexes, preferred_quant="Q4_K_M")
        for row in pair or []:
            hf_id = str(row.get("hf_id", ""))
            model_path = row.get("model_path")
            if hf_id in MODEL_REPOSITORY_IDS and isinstance(model_path, str):
                found[hf_id] = Path(model_path)
    return found


def _snapshot_revision(path: Path) -> str | None:
    try:
        return path.parts[path.parts.index("snapshots") + 1]
    except (ValueError, IndexError):
        return None


def inspect_cached_models(
    cache_root: Path,
    *,
    metadata_reader: MetadataReader | None = None,
    progress_callback: ProgressCallback | None = None,
) -> list[JsonDict]:
    """Resolve and hash one GGUF per repository without loading a model."""

    cache_root = Path(cache_root)
    reader = read_gguf_metadata if metadata_reader is None else metadata_reader
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
                    "candidate_path": None,
                    "path": None,
                    "file_name": None,
                    "quantization": None,
                    "size_bytes": None,
                    "sha256": None,
                    "revision": None,
                    "embedded_chat_template_available": None,
                    "metadata_inspection": "not_applicable_missing_candidate",
                    "metadata_error": None,
                    "tensor_payload_bytes_read": 0,
                    "resolution_api": resolution_api,
                    "model_loaded": False,
                    "generation_performed": False,
                    "download_performed": False,
                    "terminal": True,
                }
            )
            continue
        if progress_callback is not None:
            progress_callback(f"hash start {hf_id}")
        digest = file_sha256(candidate)
        if progress_callback is not None:
            progress_callback(f"hash end {hf_id}")
        template_available: bool | None = None
        metadata_outcome = "bounded_header_only"
        metadata_error: str | None = None
        tensor_payload_bytes_read = 0
        try:
            metadata = dict(reader(candidate))
            tokenizer = metadata.get("tokenizer_metadata", {})
            bounded = metadata.get("bounded_read_receipt", {})
            if isinstance(tokenizer, Mapping):
                template_available = bool(tokenizer.get("chat_template_present"))
            if isinstance(bounded, Mapping):
                tensor_payload_bytes_read = int(bounded.get("tensor_payload_bytes_read", 0))
        except (OSError, ValueError, TypeError) as exc:
            metadata_outcome = "bounded_header_error"
            metadata_error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "hf_id": hf_id,
                "cache_status": "resolved",
                "candidate_path": str(candidate.absolute()),
                "path": str(candidate.absolute()),
                "file_name": candidate.name,
                "quantization": _quantization(candidate),
                "size_bytes": candidate.stat().st_size,
                "sha256": digest,
                "revision": _snapshot_revision(candidate),
                "embedded_chat_template_available": template_available,
                "metadata_inspection": metadata_outcome,
                "metadata_error": metadata_error,
                "tensor_payload_bytes_read": tensor_payload_bytes_read,
                "resolution_api": resolution_api,
                "model_loaded": False,
                "generation_performed": False,
                "download_performed": False,
                "terminal": True,
            }
        )
    return rows


def task_method_map_rows() -> list[JsonDict]:
    """Map the four planner-selected methods to exact V629 tasks."""

    records = (
        (
            "optiver_dual_side_verification",
            "openreview:w696Vhv5B2",
            "https://openreview.net/forum?id=w696Vhv5B2",
            "2025-09-20",
            "Check generated source structure and exact solution output on the same frozen rows.",
            "The author record does not prove Carnot detection or repair value.",
        ),
        (
            "skillglow_procedure_family_consolidation",
            "arxiv:2609.02217",
            "https://arxiv.org/abs/2609.02217",
            "2026-09-02",
            "Derive de-instantiated procedure families and admit updates only after delayed exact feedback.",
            "The preprint does not prove Carnot continuous self-learning value.",
        ),
        (
            "trace2tower_deterministic_trace_projection",
            "arxiv:2609.05261",
            "https://arxiv.org/abs/2609.05261",
            "2026-09-04",
            "Project canonical events into procedure-family and strategy summaries without claiming spectral reproduction.",
            "The preprint does not prove Carnot memory transfer or spectral parity.",
        ),
        (
            "capture_drift_and_poison_controls",
            "arxiv:2609.02265",
            "https://arxiv.org/abs/2609.02265",
            "2026-09-02",
            "Replay valid drift, stale memory, poison, and negative-transfer controls in a cold process.",
            "The preprint does not prove Carnot poison resistance or transfer value.",
        ),
    )
    return [
        {
            "method_id": method_id,
            "source_id": source_id,
            "source_url": source_url,
            "publication_date": published,
            "bounded_method_fact": fact,
            "target_task_id": target,
            "decision_changing": True,
            "already_in_v629_planner": True,
            "execution_delta_new": False,
            "claim_boundary": boundary,
            "carnot_result_claimed": False,
            "terminal": True,
        }
        for (method_id, source_id, source_url, published, fact, boundary), target in zip(
            records, TARGET_TASK_IDS, strict=True
        )
    ]


def deferred_rows() -> list[JsonDict]:
    """Preserve relevant leads that do not change a V629 task."""

    records = (
        (
            "arxiv:2609.04582",
            "https://arxiv.org/abs/2609.04582",
            "internal probe calibration",
            "The local GGUF path lacks a validated hidden-state receipt and a disjoint calibration split.",
        ),
        (
            "arxiv:2609.00838",
            "https://arxiv.org/abs/2609.00838",
            "probabilistic sequence model checking",
            "The mandated GGUF path lacks validated token probabilities and a PRISM receipt.",
        ),
        (
            "arxiv:2609.02195",
            "https://arxiv.org/abs/2609.02195",
            "Hopfield memory review",
            "The review adds no bounded method beyond V629 exact memory controls.",
        ),
        (
            "arxiv:2608.31114",
            "https://arxiv.org/abs/2608.31114",
            "learned multiscale Ising sampling",
            "V629 already has a corrected exact-parity branch; the source adds no execution result.",
        ),
        (
            "arxiv:2609.05025",
            "https://arxiv.org/abs/2609.05025",
            "source-to-SQL grounding",
            "The V627 fixture already supplies this input; V629 tests OptiVer's added source-side check.",
        ),
        (
            "arxiv:2609.05281",
            "https://arxiv.org/abs/2609.05281",
            "KAN speech adaptation",
            "The trainable speech adapter does not reverse prior Carnot KAN results.",
        ),
        (
            "arxiv:2608.26960",
            "https://arxiv.org/abs/2608.26960",
            "Geo-LoRA continual adaptation",
            "The method changes model weights and does not supply frozen-GGUF external memory.",
        ),
        (
            "extropic:z1t-2026-09-04",
            PROBE_URLS["extropic_rows"],
            "Z1T hardware deployment",
            "No authenticated Z1 device is attached, so runtime, power, and speed stay deferred.",
        ),
        (
            "logical-intelligence:kona-1.0",
            PROBE_URLS["logical_intelligence_rows"],
            "Kona architecture",
            "No public checkpoint, training recipe, or compatible local runner was found.",
        ),
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
    """Combine typed rows while retaining each collection name."""

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


def probe_url(url: str) -> JsonDict:
    """Probe a public metadata route and preserve explicit HTTP limits."""

    request = Request(url, headers={"User-Agent": "carnot-v629-source-audit/1.0"})
    try:
        with urlopen(request, timeout=15) as response:
            response.read(4096)
            status = int(response.status)
        return {
            "available": 200 <= status < 400,
            "access_outcome": f"http_{status}",
            "http_status": status,
        }
    except HTTPError as exc:
        return {
            "available": False,
            "access_outcome": f"http_{exc.code}",
            "http_status": int(exc.code),
        }
    except (OSError, URLError) as exc:
        return {
            "available": False,
            "access_outcome": f"network_error:{type(exc).__name__}",
            "http_status": None,
        }


def probe_routes() -> dict[str, JsonDict]:
    """Probe each named source and repository route without model downloads."""

    receipts: dict[str, JsonDict] = {}
    for route in NETWORK_ROUTE_NAMES:
        print(f"[exp7152 network start] {route}", flush=True)
        receipts[route] = probe_url(PROBE_URLS[route])
        print(
            f"[exp7152 network end] {route} {receipts[route]['access_outcome']}",
            flush=True,
        )
    return receipts


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
        with tempfile.NamedTemporaryFile(dir=parent, prefix=".exp7152-", delete=True):
            return True
    except OSError:
        return False


def check_preconditions(root: Path, output_path: Path, cache_root: Path) -> list[JsonDict]:
    """Check every local blocker before the workflow can access a network route."""

    inputs = [path.as_posix() for path in REQUIRED_INPUT_PATHS if _readable_nonempty(root / path)]
    references = root / REFERENCE_PATH
    planner_present = False
    if _readable_nonempty(references):
        text = references.read_text(encoding="utf-8")
        planner_present = PLANNER_START_MARKER in text and PLANNER_END_MARKER in text
    reference_writable = _writable_target(references)
    output_writable = _writable_target(output_path)
    cache_readable = cache_root.is_dir() and os.access(cache_root, os.R_OK)
    return [
        {
            "check": "readable_planning_sources",
            "expected_value": [path.as_posix() for path in REQUIRED_INPUT_PATHS],
            "observed_value": inputs,
            "passed": len(inputs) == len(REQUIRED_INPUT_PATHS),
        },
        {
            "check": "v629_planner_marker",
            "expected_value": [PLANNER_START_MARKER, PLANNER_END_MARKER],
            "observed_value": (
                [PLANNER_START_MARKER, PLANNER_END_MARKER] if planner_present else "missing"
            ),
            "passed": planner_present,
        },
        {
            "check": "model_cache_metadata",
            "expected_value": "readable_directory",
            "observed_value": "readable_directory" if cache_readable else "missing_or_unreadable",
            "passed": cache_readable,
        },
        {
            "check": "writable_reference_path",
            "expected_value": "writable_parent_directory",
            "observed_value": "writable_parent_directory" if reference_writable else "not_writable",
            "passed": reference_writable,
        },
        {
            "check": "writable_artifact_path",
            "expected_value": "writable_parent_directory",
            "observed_value": "writable_parent_directory" if output_writable else "not_writable",
            "passed": output_writable,
        },
    ]


def append_references_delta(
    path: Path,
    method_rows: Sequence[Mapping[str, Any]],
    cached_rows: Sequence[Mapping[str, Any]],
    run_date: str,
    external_rows: Sequence[Mapping[str, Any]],
) -> bool:
    """Append one bounded V629 block and preserve repeated-call bytes."""

    text = path.read_text(encoding="utf-8")
    if REFERENCE_START_MARKER in text or REFERENCE_END_MARKER in text:
        return False
    lines = [
        "",
        f"## V629 execution delta - {datetime.strptime(run_date, '%Y%m%d').date().isoformat()}",
        "",
        REFERENCE_START_MARKER,
        "",
        "This execution refresh confirms four planner-selected method maps.",
        "The refresh found no post-planner method changed the V629 task contract.",
        "It does not report a Carnot result from any source claim.",
        "",
    ]
    for row in method_rows:
        lines.append(
            f"- **{row['method_id']}** - {row['source_url']}. V629 task: "
            f"`{row['target_task_id']}`. Boundary: {row['claim_boundary']}"
        )
    unavailable = [row for row in external_rows if row.get("available") is False]
    lines.extend(["", "Execution-time external access limits:", ""])
    if unavailable:
        for row in unavailable:
            identity = row.get("source_id", row.get("row_id", row.get("hf_id", "route")))
            lines.append(
                f"- `{identity}` - {row['access_outcome']}. "
                "This source gap does not replace a primary receipt."
            )
    else:
        lines.append("- No checked external route was unavailable.")
    lines.extend(["", "Execution-time local GGUF cache state:", ""])
    for row in cached_rows:
        if row.get("cache_status") == "resolved":
            template = row.get("embedded_chat_template_available")
            state = (
                f"`{row['file_name']}`; {row['quantization']}; {row['size_bytes']} bytes; "
                f"revision {row['revision']}; {row['sha256']}; embedded template {template}"
            )
        else:
            state = "no local language-model GGUF candidate resolved; embedded template unknown"
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


def _base_artifact(preconditions: Sequence[Mapping[str, Any]]) -> JsonDict:
    artifact: JsonDict = {field: [] for field in ROW_COLLECTION_FIELDS}
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
            "v629_source_delta_complete_score": 0,
            "random_seed": RANDOM_SEED,
            "reproducibility_checksum": "",
            "gate_check_summary": {
                "failed_check": "preconditions_not_checked",
                "expected_value": "all_local_prerequisites_present",
                "observed_value": "not_checked",
                "passed": False,
            },
            "verifier_is_oracle": False,
            "verdict_class": "blocked",
            "honest_verdict": "blocked_v629_source_delta_precondition",
        }
    )
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


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
    identities = [row.get("source_id") for row in classified if isinstance(row, Mapping)]
    if len(identities) != len(set(identities)) or len(identities) != len(classified):
        errors.append("source class canonical duplicate")
    if any(
        not isinstance(row, Mapping)
        or not row.get("claim_boundary")
        or row.get("terminal") is not True
        for row in classified
    ):
        errors.append("source class boundary or terminal receipt missing")
    if any(
        row.get("source_type") in {"author_venue_record", "preprint"}
        and not publication_date_in_window(row.get("publication_date"), RUN_DATE)
        for row in classified
        if isinstance(row, Mapping)
    ):
        errors.append("scientific publication date outside window")
    if not _url_fields_valid(artifact):
        errors.append("direct URL field invalid")

    repositories = artifact.get("model_repository_rows")
    if not isinstance(repositories, list) or [
        row.get("hf_id") for row in repositories if isinstance(row, Mapping)
    ] != list(MODEL_REPOSITORY_IDS):
        errors.append("model repository order mismatch")
    elif any(
        row.get("download_performed") is not False
        or row.get("terminal") is not True
        or not isinstance(row.get("available"), bool)
        or not row.get("access_outcome")
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
            if not isinstance(row, Mapping):
                errors.append("cached model row invalid")
                continue
            if any(
                row.get(key) is not False
                for key in ("download_performed", "model_loaded", "generation_performed")
            ):
                errors.append("cached model execution boundary mismatch")
            if row.get("candidate_path") != row.get("path"):
                errors.append("cached model candidate path mismatch")
            if row.get("cache_status") == "resolved":
                if (
                    not str(row.get("path", "")).startswith("/")
                    or not str(row.get("file_name", "")).lower().endswith(".gguf")
                    or "mmproj" in str(row.get("file_name", "")).lower()
                    or row.get("quantization") != "Q4_K_M"
                    or not isinstance(row.get("size_bytes"), int)
                    or row.get("size_bytes", 0) <= 0
                    or not str(row.get("sha256", "")).startswith("sha256:")
                    or not row.get("revision")
                    or not isinstance(row.get("embedded_chat_template_available"), bool)
                    or row.get("metadata_inspection") != "bounded_header_only"
                    or row.get("metadata_error") is not None
                    or row.get("tensor_payload_bytes_read") != 0
                    or row.get("resolution_api")
                    not in {"cached_sota_pair", "private_cache_fixture_scan"}
                ):
                    errors.append("resolved cached model facts incomplete")
            elif row.get("cache_status") == "missing":
                if any(
                    row.get(key) is not None
                    for key in (
                        "candidate_path",
                        "path",
                        "file_name",
                        "quantization",
                        "size_bytes",
                        "sha256",
                        "revision",
                        "embedded_chat_template_available",
                    )
                ):
                    errors.append("missing cached model row has file facts")
            else:
                errors.append("cached model status invalid")

    if artifact.get("task_method_map_rows") != task_method_map_rows():
        errors.append("task method map mismatch")
    if artifact.get("deferred_rows") != deferred_rows():
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
    hashes = artifact.get("source_artifact_hashes")
    if not isinstance(hashes, list) or not hashes or any(
        not isinstance(row, Mapping)
        or not row.get("path")
        or not str(row.get("sha256", "")).startswith("sha256:")
        for row in hashes
    ):
        errors.append("source artifact hashes incomplete")
    return list(dict.fromkeys(errors))


def completion_score(artifact: Mapping[str, Any]) -> int:
    """Return one only when source, cache, map, and append checks pass."""

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


def recompute_artifact(artifact: JsonDict) -> JsonDict:
    """Recompute combined rows, score, verdict, gate, and checksum."""

    artifact["rows"] = combined_rows(artifact)
    score = completion_score(artifact)
    artifact["v629_source_delta_complete_score"] = score
    if score == 1:
        changed = any(
            row.get("execution_delta_new") is True for row in artifact["task_method_map_rows"]
        )
        verdict = "positive" if changed else "null"
        artifact["verdict_class"] = verdict
        artifact["honest_verdict"] = (
            "positive_v629_source_delta_complete_with_plan_change"
            if changed
            else "null_v629_source_delta_complete_no_post_planner_change"
        )
        failed_check = None
    else:
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = "disqualified_v629_source_delta_contract"
        failed_check = "source_delta_contract"
    artifact["gate_check_summary"] = {
        "failed_check": failed_check,
        "expected_value": 1,
        "observed_value": score,
        "passed": score == 1,
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _checkpoint_artifact(artifact: Mapping[str, Any], output_path: Path) -> Path:
    """Write the current complete schema before the next fallible phase."""

    atomic_write_json(output_path, dict(artifact), allow_override=False)
    return output_path


def _set_blocked_gate(artifact: JsonDict, failure: Mapping[str, Any]) -> None:
    artifact["gate_check_summary"] = {
        "failed_check": failure["check"],
        "expected_value": failure["expected_value"],
        "observed_value": failure["observed_value"],
        "passed": False,
    }
    artifact["inference_substrate_class"] = "blocked_no_run"
    artifact["v629_source_delta_complete_score"] = 0
    artifact["verdict_class"] = "blocked"
    artifact["honest_verdict"] = f"blocked_v629_source_delta_{failure['check']}"


def build_artifact(
    root: Path,
    run_date: str,
    *,
    cache_root: Path,
    output_path: Path = RESULT_PATH,
    route_reachability: Mapping[str, object] | None = None,
    metadata_reader: MetadataReader | None = None,
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

    _progress(0, "schema checkpoint start")
    artifact = _base_artifact([])
    _checkpoint_artifact(artifact, output_path)
    _progress(0, "schema checkpoint end; local checks start")
    preconditions = check_preconditions(root, output_path, cache_root)
    artifact["preconditions_checked"] = preconditions
    artifact["source_artifact_hashes"] = _source_hashes(root)
    failure = next((row for row in preconditions if row["passed"] is False), None)
    if failure is not None:
        _set_blocked_gate(artifact, failure)
        elapsed = duration_s if duration_s is not None else time.monotonic() - started
        artifact["duration_s"] = round(max(0.0, float(elapsed)), 6)
        artifact["reproducibility_checksum"] = payload_checksum(artifact)
        _checkpoint_artifact(artifact, output_path)
        _progress(0, f"local checks blocked: {failure['check']}")
        return artifact

    artifact["inference_substrate_class"] = INFERENCE_SUBSTRATE_CLASS
    artifact["verdict_class"] = "partial"
    artifact["honest_verdict"] = "partial_v629_source_delta_running_after_preconditions"
    artifact["gate_check_summary"] = {
        "failed_check": None,
        "expected_value": "all_local_prerequisites_present",
        "observed_value": "all_local_prerequisites_present",
        "passed": True,
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    _checkpoint_artifact(artifact, output_path)
    _progress(1, "local checks recorded; running checkpoint written before network")

    _progress(2, "external source probe start")
    routes = dict(probe_routes() if route_reachability is None else route_reachability)
    _progress(2, "external source probe end")
    _progress(3, "source classification start")
    collections = source_collections(routes)
    repositories = model_repository_rows(routes)
    artifact.update(collections)
    artifact["source_class_rows"] = source_class_rows(collections)
    artifact["model_repository_rows"] = repositories
    _progress(3, "source classification end")

    _progress(4, "local cache hash and bounded metadata start")
    cached = inspect_cached_models(
        cache_root,
        metadata_reader=metadata_reader,
        progress_callback=lambda message: _progress(4, message),
    )
    artifact["cached_model_rows"] = cached
    _progress(4, "local cache hash and bounded metadata end")

    _progress(5, "V629 method mapping start")
    artifact["task_method_map_rows"] = task_method_map_rows()
    artifact["deferred_rows"] = deferred_rows()
    _progress(5, "V629 method mapping end")

    _progress(6, "dated reference append start")
    external_rows = [
        row
        for field in SOURCE_COLLECTION_FIELDS
        for row in collections[field]
        if row.get("available") is False
    ] + [row for row in repositories if row.get("available") is False]
    appended = False
    if update_references:
        appended = append_references_delta(
            root / REFERENCE_PATH,
            artifact["task_method_map_rows"],
            cached,
            run_date,
            external_rows,
        )
    artifact["reference_append_marker"] = _reference_marker_receipt(
        root / REFERENCE_PATH, appended
    )
    artifact["source_artifact_hashes"] = _source_hashes(root)
    _progress(6, "dated reference append end")

    _progress(7, "row consistency and checksum recompute start")
    recompute_artifact(artifact)
    _progress(7, "row consistency and checksum recompute end")
    elapsed = duration_s if duration_s is not None else time.monotonic() - started
    artifact["duration_s"] = round(max(0.0, float(elapsed)), 6)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    _progress(8, "final artifact checkpoint start")
    _checkpoint_artifact(artifact, output_path)
    _progress(8, "final artifact checkpoint end")
    return artifact


def _terminal_errors(artifact: Mapping[str, Any]) -> list[str]:
    preconditions = artifact.get("preconditions_checked")
    if not isinstance(preconditions, list) or not preconditions:
        return ["preconditions_checked missing"]
    failure = next(
        (row for row in preconditions if isinstance(row, Mapping) and row.get("passed") is False),
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
        if artifact.get("v629_source_delta_complete_score") != 0:
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
    if artifact.get("v629_source_delta_complete_score") != expected_score:
        errors.append("v629_source_delta_complete_score mismatch")
    changed = any(
        isinstance(row, Mapping) and row.get("execution_delta_new") is True
        for row in artifact.get("task_method_map_rows", [])
    )
    expected_class = ("positive" if changed else "null") if expected_score == 1 else "disqualified"
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
    if artifact["inference_substrate_class"] not in {
        INFERENCE_SUBSTRATE_CLASS,
        "blocked_no_run",
    }:
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
    atomic_write_json(output_path, dict(artifact), allow_override=False)
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
        print(json.dumps({"valid": not errors, "errors": errors}, sort_keys=True), flush=True)
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
        print(json.dumps({"valid": False, "errors": errors}, sort_keys=True), flush=True)
        return 1
    write_artifact(artifact, output_path)
    print(
        json.dumps(
            {
                "artifact": str(output_path),
                "v629_source_delta_complete_score": artifact[
                    "v629_source_delta_complete_score"
                ],
                "verdict_class": artifact["verdict_class"],
                "honest_verdict": artifact["honest_verdict"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - the CLI wrapper calls main.
    raise SystemExit(main())
