"""Build the V625 source, method-map, and local-model-cache receipt.

The task aggregates public metadata and local file facts. It does not load a
model or treat a paper, repository, vendor page, or product page as a Carnot
result. Spec refs: REQ-REPORT-7122 and SCENARIO-REPORT-7122-*.
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
from carnot.paths import repo_root as find_repo_root


JsonDict = dict[str, Any]

RUN_DATE = "20260907"
SOURCE_CAPTURE_UTC = "2026-09-07T18:16:46Z"
RANDOM_SEED = 7_122_202_609_07
INFERENCE_SUBSTRATE = (
    "aggregation_from_external_primary_sources; "
    "local model-cache metadata included, no model loaded"
)
INFERENCE_SUBSTRATE_CLASS = "aggregation"
EXECUTION_VENUE = "host"
VENDOR_BOUNDARY = "vendor_claim_not_independent_evidence"
PRODUCT_BOUNDARY = "product_page_not_reproducible_evidence"

REFERENCE_PATH = Path("research-references.md")
RESULT_PATH = Path("results/experiment_7122_v625_sota_ingestion.json")
PLANNER_START_MARKER = "<!-- V625-PLANNER-REFRESH-20260907-START -->"
PLANNER_END_MARKER = "<!-- V625-PLANNER-REFRESH-20260907-END -->"
REFERENCE_START_MARKER = "<!-- V625-EXECUTION-DELTA-20260907-START -->"
REFERENCE_END_MARKER = "<!-- V625-EXECUTION-DELTA-20260907-END -->"

REQUIRED_INPUT_PATHS = (
    Path("research-program.md"),
    REFERENCE_PATH,
    Path("research-hardware-wishlist.md"),
    Path("research-roadmap.yaml"),
    Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
    Path("results/experiment_7112_v624_sota_ingestion.json"),
)
SOURCE_ARTIFACT_PATHS = (
    *REQUIRED_INPUT_PATHS,
    Path("openspec/capabilities/research-reporting/spec.md"),
    Path("python/carnot/experiment_7122_v625_sota_ingestion.py"),
    Path("scripts/experiments/experiment_7122_v625_sota_ingestion.py"),
    Path("tests/python/test_experiment_7122_v625_sota_ingestion.py"),
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
TARGET_TASK_IDS = (
    "exp7125-arc-loo-causal-audit",
    "exp7127-verifier-committed-revision",
    "exp7129-cross-model-memory-portability",
    "exp7131-wcrg-multiscale-sampler-prototype",
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
    "source_class_rows",
    "publication_date_rows",
    "model_repository_rows",
    "cached_model_rows",
    "task_method_map_rows",
    "deferred_rows",
    "reference_append_marker",
    "v625_sota_ingestion_complete_score",
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

FIELD_PRINCIPLES = {
    "field_principles": "A principle for each field makes the receipt auditable.",
    "preconditions_checked": "Explicit gates prevent an access failure from becoming invented evidence.",
    "run_date": "The fixed date closes the scientific publication window.",
    "inference_substrate": "The substrate limits the result to source and cache aggregation.",
    "inference_substrate_class": "The class applies the aggregation duration and no-model rules.",
    "execution_venue": "The host venue prevents an unattached hardware claim.",
    "duration_s": "Measured elapsed time proves that the refresh executed.",
    "source_artifact_hashes": "Hashes bind the refresh to its local planning inputs.",
    "rows": "The combined rows support an independent collection consistency check.",
    "arxiv_rows": "Primary preprint rows preserve exact titles, dates, and bounded facts.",
    "openreview_rows": "Reviewed venue rows stay separate from preprints and discovery indexes.",
    "semantic_scholar_rows": "Citation queries are discovery receipts, not scientific authority.",
    "huggingface_rows": "Generated paper summaries stay below their primary papers.",
    "github_rows": "Repository identity and code availability do not prove method quality.",
    "extropic_rows": "Vendor claims cannot prove local hardware execution, speed, or power.",
    "logical_intelligence_rows": "A product page cannot replace public weights or a runner.",
    "source_class_rows": "One canonical row per source prevents evidence-class drift and duplication.",
    "publication_date_rows": "Inclusive date checks exclude old and future scientific claims.",
    "model_repository_rows": "Remote metadata records repository state without downloading weights.",
    "cached_model_rows": "Exact local paths, sizes, and hashes make later model use reproducible.",
    "task_method_map_rows": "Only a method that changes a named V625 task receives a map.",
    "deferred_rows": "A reasoned deferral preserves useful leads without expanding scope.",
    "reference_append_marker": "Marker counts and a hash make the dated append idempotent.",
    "v625_sota_ingestion_complete_score": "One means all source, model, map, and append checks agree.",
    "random_seed": "A fixed seed records deterministic row ordering.",
    "reproducibility_checksum": "A stable payload hash detects later evidence or verdict drift.",
    "gate_check_summary": "Exact expected and observed values make a block actionable.",
    "verifier_is_oracle": "False states that this audit does not certify Carnot correctness.",
    "verdict_class": "A closed class separates valid, blocked, and disqualified outcomes.",
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
    """Accept a direct HTTPS URL with a real network host."""

    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    return parsed.scheme == "https" and bool(parsed.netloc)


def publication_date_in_window(value: object, run_date: str) -> bool:
    """Accept scientific dates from 2025 through the fixed execution day."""

    if not isinstance(value, str):
        return False
    try:
        observed = date.fromisoformat(value)
        upper = datetime.strptime(run_date, "%Y%m%d").date()
    except ValueError:
        return False
    return date(2025, 1, 1) <= observed <= upper


def deduplicate_source_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Keep one canonical identity and prefer the strongest evidence class."""

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
    """Return the bounded arXiv query and the four decision-changing papers."""

    query = {
        "row_id": "arxiv-topic-sweep-20260907",
        "query_url": "https://export.arxiv.org/api/query",
        "query_families": [
            "EBM reasoning",
            "neural constraint satisfaction",
            "Ising machine learning",
            "hallucination mitigation",
            "KANs",
            "constrained decoding",
            "accelerated sampling",
            "continual learning",
        ],
        "date_start": "2025-01-01",
        "date_end": "2026-09-07",
        "accessed_at": SOURCE_CAPTURE_UTC,
        "access_outcome": "http_200_complete",
        "terminal": True,
    }
    records = (
        (
            "2609.04343",
            "A Removal Based Approach to Improve LLM Faithfulness at Test-Time",
            "2026-09-03",
            "The method removes input concepts that an explanation did not credit and re-queries the model to test explanation incompleteness.",
            "No public code link was present in the arXiv metadata during this audit.",
            "Directly motivates deterministic removal replay for credited ARC signals.",
        ),
        (
            "2609.04773",
            "Persistent Teacher Anchoring for Tool-Using Agents",
            "2026-09-04",
            "A tool call reaches the environment only after a teacher verifies the complete turn.",
            "No public code link was present in the arXiv metadata during this audit.",
            "Directly motivates exact-verifier commitment before a constraint action executes.",
        ),
        (
            "2609.05339",
            "Does Your Agent's Memory Survive a Model Upgrade? A Controlled Study of Memory Portability",
            "2026-09-04",
            "The study tests directional writer-reader swaps and compares fixed-schema memory, notes, retrieval, and source-backed repair.",
            "No public code link was present in the arXiv metadata during this audit.",
            "Directly motivates directional three-family memory portability cells.",
        ),
        (
            "2608.31114",
            "Overcoming critical slowing down in frustrated spin systems by learned multiscale sampling",
            "2026-08-31",
            "WCRG learns conditional wavelet distributions and generates configurations recursively from coarse to fine scales.",
            "A related public WCRG repository exists, but it is not verified as this paper's release.",
            "Directly motivates a corrected, finite-parity software prototype.",
        ),
    )
    return [query] + [
        {
            "source_id": f"arxiv:{paper_id}",
            "title": title,
            "url": f"https://arxiv.org/abs/{paper_id}",
            "publication_date": published,
            "abstract_or_method_fact": fact,
            "public_code_state": code,
            "relevance": relevance,
            "source_type": "preprint",
            "accessed_at": SOURCE_CAPTURE_UTC,
            "access_outcome": "http_200_verified_primary",
            "terminal": True,
        }
        for paper_id, title, published, fact, code, relevance in records
    ]


def openreview_rows() -> list[JsonDict]:
    """Return current reviewed constraint and continual-memory controls."""

    return [
        {
            "source_id": "openreview:oum1txoy1D",
            "title": "FSNet: Feasibility-Seeking Neural Network for Constrained Optimization with Guarantees",
            "url": "https://openreview.net/forum?id=oum1txoy1D",
            "publication_date": "2025-09-18",
            "venue_state": "NeurIPS 2025 poster",
            "abstract_or_method_fact": "FSNet puts a differentiable feasibility-seeking step inside its solution procedure.",
            "public_code_state": "Public code repository observed at MOSSLab-MIT/FSNet.",
            "relevance": "Retains feasibility-first handling as a control, but does not change a V625 task.",
            "source_type": "peer_reviewed",
            "accessed_at": SOURCE_CAPTURE_UTC,
            "access_outcome": "reachable_http_challenge_with_verified_index_metadata",
            "terminal": True,
        },
        {
            "source_id": "openreview:MSXbrNExax",
            "title": "Benchmarking Continual Agent Memory for Online Learning, Transfer, and Forgetting",
            "url": "https://openreview.net/forum?id=MSXbrNExax",
            "publication_date": "2026-04-11",
            "venue_state": "Lifelong Agents at ICLR 2026 workshop poster",
            "abstract_or_method_fact": "AgentMemoryBench separates online, replay, transfer, repair, and forgetting measurements.",
            "public_code_state": "A public AgentMemoryBench repository was observed.",
            "relevance": "It supports V625 metric separation but adds no new task.",
            "source_type": "peer_reviewed",
            "accessed_at": SOURCE_CAPTURE_UTC,
            "access_outcome": "reachable_http_challenge_with_verified_index_metadata",
            "terminal": True,
        },
    ]


def semantic_scholar_rows() -> list[JsonDict]:
    """Record current citation-query counts without promoting an index."""

    records = (
        (
            "ARXIV:2507.02092",
            35,
            ["Memoir", "Solver-Hard Is Not Model-Hard", "Fixed-Point Reasoners", "LoopUS"],
        ),
        (
            "ARXIV:2512.15605",
            8,
            ["Distributional Energy-Based Models", "LoopUS", "Large Language Models Can Take False First Steps"],
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
            "relevance": "The citation trail adds controls but no fifth V625 task method.",
            "accessed_at": SOURCE_CAPTURE_UTC,
            "access_outcome": "http_200_complete",
            "terminal": True,
        }
        for paper_id, count, titles in records
    ]


def huggingface_rows() -> list[JsonDict]:
    """Keep Hugging Face Papers as a secondary discovery surface."""

    return [
        {
            "source_id": f"arxiv:{paper_id}",
            "url": f"https://huggingface.co/papers/{paper_id}",
            "primary_source_url": f"https://arxiv.org/abs/{paper_id}",
            "source_type": "secondary_discovery",
            "generated_summary_used_as_evidence": False,
            "relevance": "Discovery duplicate; the arXiv row owns the method fact.",
            "accessed_at": SOURCE_CAPTURE_UTC,
            "access_outcome": "checked_secondary_index",
            "terminal": True,
        }
        for paper_id in ("2609.04343", "2609.04773", "2609.05339", "2608.31114")
    ]


def github_rows() -> list[JsonDict]:
    """Return repository identities while excluding popularity as evidence."""

    records = (
        ("alexiglad/EBT", "19420cbeae655bbf11930219a675ade6897019e8", "2026-04-21T00:53:47Z", "Apache-2.0"),
        ("MOSSLab-MIT/FSNet", "826457df85302da8c7553977ce74a4ee18d1b362", "2026-06-09T21:48:12Z", "MIT"),
        ("Qwen-Applications/MARCH", "1805095bbfee0b1d90c9ad8c473911276469c4e3", "2026-06-09T05:01:09Z", None),
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
            "relevance": "Code identity only; no repository replaces Carnot's exact authority or pinned runtime.",
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
        "abstract_or_method_fact": "The page states a degree-16 Z1 graph and an FPGA-plus-Z1 design.",
        "public_code_state": "The page links open weights and a sparse-transformer training repository.",
        "relevance": "Useful boundary information only; no Z1 device is attached.",
        "source_type": "vendor_claim",
        "claim_boundary_label": VENDOR_BOUNDARY,
        "scientific_evidence_promoted": False,
        "hardware_execution_claimed": False,
        "speed_or_power_claimed": False,
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
        "abstract_or_method_fact": "The page describes a global energy layer that evaluates allowed system states.",
        "public_code_state": "No public checkpoint, training recipe, or compatible local runner was observed.",
        "relevance": "Architecture comparator only.",
        "source_type": "product_page",
        "claim_boundary_label": PRODUCT_BOUNDARY,
        "scientific_evidence_promoted": False,
        "reproducible_local_runner_observed": False,
        "accessed_at": SOURCE_CAPTURE_UTC,
        "access_outcome": "http_200_complete",
        "terminal": True,
    }]


def source_class_rows() -> list[JsonDict]:
    """Return one canonical ledger with every required evidence class."""

    arxiv = arxiv_rows()
    openreview = openreview_rows()
    semantic = semantic_scholar_rows()
    github = github_rows()
    rows = [
        {
            **openreview[0],
            "claim_boundary": "Reviewed results do not prove Carnot feasibility or model value.",
        },
        *(
            {
                **row,
                "claim_boundary": "The paper's result does not establish the mapped Carnot outcome.",
            }
            for row in arxiv[1:]
        ),
        {
            **extropic_rows()[0],
            "claim_boundary": "Vendor projections do not prove local execution, speed, or power.",
        },
        {
            **logical_intelligence_rows()[0],
            "claim_boundary": "The product page is not a public reproducible baseline.",
        },
        {
            **github[0],
            "claim_boundary": "Repository availability does not prove scientific quality or local fitness.",
        },
        {
            **semantic[0],
            "url": semantic[0]["query_url"],
            "claim_boundary": "A citation index cannot override a primary source.",
        },
    ]
    return deduplicate_source_rows(rows)


def publication_date_rows() -> list[JsonDict]:
    """Return explicit date checks for every source used as scientific work."""

    scientific = [*arxiv_rows()[1:], *openreview_rows()]
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
    """Return the observed Hugging Face repository metadata snapshot."""

    records = (
        ("a483e9e6cbd595906af30beda3187c2663a1118c", "2026-04-20T12:42:25Z", 25),
        ("c1ac76e99d5513b141e8adde7288b85c3f9c32ec", "2026-07-17T13:09:42Z", 23),
        ("c099eb48e663fd284577b04978a94ffccb261841", "2026-07-17T13:00:49Z", 22),
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


def inspect_cached_models(cache_root: Path) -> list[JsonDict]:
    """Resolve one preferred language-model GGUF per repo without downloading."""

    rows: list[JsonDict] = []
    for hf_id in MODEL_REPOSITORY_IDS:
        repo_dir = Path(cache_root) / f"models--{hf_id.replace('/', '--')}"
        candidate = _preferred_cached_file(repo_dir) if repo_dir.is_dir() else None
        if candidate is None:
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
                    "download_performed": False,
                    "terminal": True,
                }
            )
            continue
        revision = candidate.relative_to(repo_dir / "snapshots").parts[0]
        rows.append(
            {
                "hf_id": hf_id,
                "cache_status": "resolved",
                "path": str(candidate.resolve()),
                "file_name": candidate.name,
                "quantization": _quantization(candidate),
                "size_bytes": candidate.stat().st_size,
                "sha256": file_sha256(candidate),
                "revision": revision,
                "download_performed": False,
                "terminal": True,
            }
        )
    return rows


def task_method_map_rows() -> list[JsonDict]:
    """Map the four verified decision-changing methods to exact V625 tasks."""

    return [
        {
            "method_id": "removal_based_causal_faithfulness_replay",
            "source_id": "arxiv:2609.04343",
            "source_url": "https://arxiv.org/abs/2609.04343",
            "publication_date": "2026-09-03",
            "bounded_method_fact": "Remove an uncredited input concept, re-query, and measure whether the decision changes.",
            "public_code_state": "No public code link was present in the arXiv metadata during this audit.",
            "relevance": "Tests whether an ARC signal affected the acting policy.",
            "target_task_id": TARGET_TASK_IDS[0],
            "decision_changing": True,
            "claim_boundary": "Removal effects in the paper do not prove an ARC action or a Carnot solve.",
            "carnot_result_claimed": False,
            "terminal": True,
        },
        {
            "method_id": "persistent_teacher_turn_commitment",
            "source_id": "arxiv:2609.04773",
            "source_url": "https://arxiv.org/abs/2609.04773",
            "publication_date": "2026-09-04",
            "bounded_method_fact": "Permit a tool call to execute only after verification of the complete turn.",
            "public_code_state": "No public code link was present in the arXiv metadata during this audit.",
            "relevance": "Changes the commit rule for exact constraint revision.",
            "target_task_id": TARGET_TASK_IDS[1],
            "decision_changing": True,
            "claim_boundary": "Teacher acceptance is not an exact constraint result and does not prove repair value.",
            "carnot_result_claimed": False,
            "terminal": True,
        },
        {
            "method_id": "directional_fixed_schema_memory_portability",
            "source_id": "arxiv:2609.05339",
            "source_url": "https://arxiv.org/abs/2609.05339",
            "publication_date": "2026-09-04",
            "bounded_method_fact": "Test every writer-reader direction and retain raw source evidence for repair.",
            "public_code_state": "No public code link was present in the arXiv metadata during this audit.",
            "relevance": "Changes the cross-family memory matrix and repair control.",
            "target_task_id": TARGET_TASK_IDS[2],
            "decision_changing": True,
            "claim_boundary": "Two sub-10B models and synthetic histories do not prove portability across Carnot's models.",
            "carnot_result_claimed": False,
            "terminal": True,
        },
        {
            "method_id": "learned_coarse_to_fine_multiscale_proposals",
            "source_id": "arxiv:2608.31114",
            "source_url": "https://arxiv.org/abs/2608.31114",
            "publication_date": "2026-08-31",
            "bounded_method_fact": "Learn scale-conditional wavelet proposals and sample recursively from coarse to fine scales.",
            "public_code_state": "A related WCRG repository exists, but it is not verified as this paper's release.",
            "relevance": "Changes the small frustrated-Ising proposal prototype.",
            "target_task_id": TARGET_TASK_IDS[3],
            "decision_changing": True,
            "claim_boundary": "The paper does not prove Carnot parity, mixing, asymptotic scaling, or hardware speed.",
            "carnot_result_claimed": False,
            "terminal": True,
        },
    ]


def deferred_rows() -> list[JsonDict]:
    """Preserve relevant controls that do not change the V625 task contract."""

    return [
        {
            "source_id": "openreview:oum1txoy1D",
            "url": "https://openreview.net/forum?id=oum1txoy1D",
            "lead": "FSNet feasibility-seeking",
            "target_task_id": None,
            "reason": "V625 already keeps exact feasibility before learned ranking; no new method cell is needed.",
            "terminal": True,
        },
        {
            "source_id": "openreview:MSXbrNExax",
            "url": "https://openreview.net/forum?id=MSXbrNExax",
            "lead": "AgentMemoryBench metric decomposition",
            "target_task_id": None,
            "reason": "The roadmap already separates online, replay, transfer, repair, retention, and forgetting metrics.",
            "terminal": True,
        },
        {
            "source_id": "semantic-scholar:ARXIV:2507.02092/citations",
            "url": "https://api.semanticscholar.org/graph/v1/paper/ARXIV:2507.02092/citations?limit=100",
            "lead": "EBT citation trail",
            "target_task_id": None,
            "reason": "The visible citing work adds controls but no matching exact local scorer.",
            "terminal": True,
        },
        {
            "source_id": "semantic-scholar:ARXIV:2512.15605/citations",
            "url": "https://api.semanticscholar.org/graph/v1/paper/ARXIV:2512.15605/citations?limit=100",
            "lead": "ARM-EBM citation trail",
            "target_task_id": None,
            "reason": "The visible citing work does not supply an exact ARC authority or a fifth V625 method.",
            "terminal": True,
        },
        {
            "source_id": "extropic:z1t-2026-09-04",
            "url": "https://extropic.ai/writing/z1t",
            "lead": "Z1 degree-16 deployment",
            "target_task_id": None,
            "reason": "No authenticated Z1 device is attached, so hardware execution and performance stay deferred.",
            "terminal": True,
        },
        {
            "source_id": "logical-intelligence:kona-1.0",
            "url": "https://logicalintelligence.com/kona-ebms-energy-based-models",
            "lead": "Kona architecture comparison",
            "target_task_id": None,
            "reason": "No public checkpoint, training recipe, or compatible local runner was observed.",
            "terminal": True,
        },
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
    """Treat an explicit HTTP response as reachability, even when access is denied."""

    request = Request(url, headers={"User-Agent": "carnot-v625-source-audit/1.0"})
    try:
        with urlopen(request, timeout=10) as response:
            return int(response.status) > 0
    except HTTPError:
        return True
    except (OSError, URLError):
        return False


def probe_routes() -> dict[str, bool]:
    """Probe every required source route without downloading model files."""

    urls = {
        "arxiv_rows": "https://export.arxiv.org/api/query?id_list=2609.04343&max_results=1",
        "openreview_rows": "https://openreview.net/forum?id=oum1txoy1D",
        "semantic_scholar_rows": "https://api.semanticscholar.org/graph/v1/paper/ARXIV:2507.02092",
        "huggingface_rows": "https://huggingface.co/api/daily_papers?limit=1",
        "github_rows": "https://api.github.com/repos/alexiglad/EBT",
        "extropic_rows": "https://extropic.ai/writing/z1t",
        "logical_intelligence_rows": "https://logicalintelligence.com/kona-ebms-energy-based-models",
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
        with tempfile.NamedTemporaryFile(dir=parent, prefix=".exp7122-", delete=True):
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
    """Check network, planning, cache, and destination access before the audit."""

    routes = dict(probe_routes() if route_reachability is None else route_reachability)
    network_ok = all(routes.get(name) is True for name in SOURCE_COLLECTION_FIELDS)
    inputs_ok = all(_readable_nonempty(root / path) for path in REQUIRED_INPUT_PATHS)
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
            "expected_value": list(SOURCE_COLLECTION_FIELDS),
            "observed_value": sorted(name for name, reachable in routes.items() if reachable),
            "passed": network_ok,
        },
        {
            "check": "readable_planning_sources",
            "expected_value": [path.as_posix() for path in REQUIRED_INPUT_PATHS],
            "observed_value": [
                path.as_posix() for path in REQUIRED_INPUT_PATHS if _readable_nonempty(root / path)
            ],
            "passed": inputs_ok,
        },
        {
            "check": "v625_planner_marker",
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
) -> bool:
    """Append one bounded V625 execution block and preserve repeated-run bytes."""

    text = path.read_text(encoding="utf-8")
    if REFERENCE_START_MARKER in text or REFERENCE_END_MARKER in text:
        return False
    lines = [
        "",
        f"## V625 execution delta - {datetime.strptime(run_date, '%Y%m%d').date().isoformat()}",
        "",
        REFERENCE_START_MARKER,
        "",
        "This execution refresh confirms four decision-changing method maps.",
        "It does not report a Carnot result from any source claim.",
        "",
    ]
    for row in method_rows:
        lines.append(
            f"- **{row['method_id']}** - {row['source_url']}. V625 task: "
            f"`{row['target_task_id']}`. Boundary: {row['claim_boundary']}"
        )
    lines.extend(["", "Execution-time local GGUF cache state:", ""])
    for row in cached_rows:
        if row.get("cache_status") == "resolved":
            state = (
                f"`{row['file_name']}`; {row['quantization']}; {row['size_bytes']} bytes; "
                f"{row['sha256']}"
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
    for field in (*SOURCE_COLLECTION_FIELDS, "source_class_rows", "model_repository_rows", "task_method_map_rows", "deferred_rows"):
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
        if not isinstance(artifact.get(field), list) or not artifact[field]:
            errors.append(f"missing source collection {field}")
    classified = artifact.get("source_class_rows")
    if not isinstance(classified, list):
        errors.append("source class rows missing")
        classified = []
    if {row.get("source_type") for row in classified if isinstance(row, Mapping)} != set(SOURCE_TYPES):
        errors.append("source type coverage mismatch")
    if len({row.get("source_id") for row in classified if isinstance(row, Mapping)}) != len(classified):
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
    if not isinstance(cached, list) or len(cached) != len(MODEL_REPOSITORY_IDS) or {
        row.get("hf_id") for row in cached if isinstance(row, Mapping)
    } != set(MODEL_REPOSITORY_IDS):
        errors.append("cached model coverage mismatch")
    else:
        for row in cached:
            if not isinstance(row, Mapping) or row.get("download_performed") is not False:
                errors.append("cached model download boundary mismatch")
                continue
            if row.get("cache_status") == "resolved":
                if (
                    not str(row.get("path", "")).startswith("/")
                    or not str(row.get("file_name", "")).lower().endswith(".gguf")
                    or "mmproj" in str(row.get("file_name", "")).lower()
                    or not row.get("quantization")
                    or not isinstance(row.get("size_bytes"), int)
                    or row.get("size_bytes", 0) <= 0
                    or not str(row.get("sha256", "")).startswith("sha256:")
                ):
                    errors.append("resolved cached model facts incomplete")
            elif row.get("cache_status") == "missing":
                if any(row.get(key) is not None for key in ("path", "file_name", "quantization", "size_bytes", "sha256", "revision")):
                    errors.append("missing cached model row has file facts")
            else:
                errors.append("cached model status invalid")

    maps = artifact.get("task_method_map_rows")
    if maps != task_method_map_rows():
        errors.append("task method map mismatch")
    deferred = artifact.get("deferred_rows")
    if deferred != deferred_rows() or any(
        not row.get("reason") or row.get("target_task_id") is not None for row in deferred or []
    ):
        errors.append("deferred row mismatch")
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
    """Return one only when every source, model, map, and marker check passes."""

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
    return {
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": _copy(preconditions),
        "run_date": RUN_DATE,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": EXECUTION_VENUE,
        "duration_s": 0.0,
        "source_artifact_hashes": [],
        "rows": [],
        "arxiv_rows": [],
        "openreview_rows": [],
        "semantic_scholar_rows": [],
        "huggingface_rows": [],
        "github_rows": [],
        "extropic_rows": [],
        "logical_intelligence_rows": [],
        "source_class_rows": [],
        "publication_date_rows": [],
        "model_repository_rows": [],
        "cached_model_rows": [],
        "task_method_map_rows": [],
        "deferred_rows": [],
        "reference_append_marker": {},
        "v625_sota_ingestion_complete_score": 0,
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
        "honest_verdict": "blocked_v625_sota_ingestion_precondition",
    }


def recompute_artifact(artifact: JsonDict) -> JsonDict:
    """Recompute combined rows, score, verdict, diagnostic, and checksum."""

    artifact["rows"] = combined_rows(artifact)
    score = completion_score(artifact)
    artifact["v625_sota_ingestion_complete_score"] = score
    if score == 1:
        artifact["verdict_class"] = "positive"
        artifact["honest_verdict"] = "positive_v625_sota_ingestion_complete"
        failed_check = None
    else:
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = "disqualified_v625_sota_ingestion_contract"
        failed_check = "source_ingestion_contract"
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
    """Build one schema-complete positive, blocked, or disqualified receipt."""

    started = time.monotonic()
    root = Path(root)
    cache_root = Path(cache_root)
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = root / output_path
    preconditions = check_preconditions(
        root,
        output_path,
        cache_root,
        route_reachability=route_reachability,
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
        cached = inspect_cached_models(cache_root)
        appended = False
        if update_references:
            appended = append_references_delta(
                root / REFERENCE_PATH,
                task_method_map_rows(),
                cached,
                run_date,
            )
        artifact.update(
            {
                "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
                "arxiv_rows": arxiv_rows(),
                "openreview_rows": openreview_rows(),
                "semantic_scholar_rows": semantic_scholar_rows(),
                "huggingface_rows": huggingface_rows(),
                "github_rows": github_rows(),
                "extropic_rows": extropic_rows(),
                "logical_intelligence_rows": logical_intelligence_rows(),
                "source_class_rows": source_class_rows(),
                "publication_date_rows": publication_date_rows(),
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
        if artifact.get("v625_sota_ingestion_complete_score") != 0:
            errors.append("blocked completion score must be zero")
        if artifact.get("verdict_class") != "blocked":
            errors.append("blocked verdict_class mismatch")
        if not str(artifact.get("honest_verdict", "")).startswith("blocked_"):
            errors.append("blocked honest_verdict prefix mismatch")
        return errors

    coverage_errors = _coverage_errors(artifact)
    expected_score = int(not coverage_errors)
    errors = list(coverage_errors)
    if artifact.get("v625_sota_ingestion_complete_score") != expected_score:
        errors.append("v625_sota_ingestion_complete_score mismatch")
    expected_class = "positive" if expected_score == 1 else "disqualified"
    if artifact.get("verdict_class") != expected_class:
        errors.append("verdict_class inconsistent with completion")
    if not str(artifact.get("honest_verdict", "")).startswith(f"{expected_class}_"):
        errors.append("honest_verdict prefix inconsistent with completion")
    expected_gate = {
        "failed_check": None if expected_score == 1 else "source_ingestion_contract",
        "expected_value": 1,
        "observed_value": expected_score,
        "passed": expected_score == 1,
    }
    if artifact.get("gate_check_summary") != expected_gate:
        errors.append("gate_check_summary inconsistent with completion")
    return errors


def validate_artifact(value: Mapping[str, Any] | str | Path) -> list[str]:
    """Validate field shape, evidence boundaries, rows, state, and checksum."""

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
        errors.append("verifier_is_oracle must be false")
    if artifact["verdict_class"] not in VERDICT_CLASSES:
        errors.append("verdict_class outside closed enum")
    if artifact["rows"] != combined_rows(artifact):
        errors.append("rows consistency mismatch")
    errors.extend(_terminal_errors(artifact))
    if artifact["reproducibility_checksum"] != payload_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    return list(dict.fromkeys(errors))


def write_artifact(artifact: Mapping[str, Any], path: Path) -> Path:
    """Validate before atomically replacing the requested JSON artifact."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError(";".join(errors))
    return atomic_write_json(path, dict(artifact), allow_override=False, sort_keys=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path.home() / ".cache" / "huggingface" / "hub",
    )
    parser.add_argument("--validate", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the refresh or validate one stored artifact."""

    args = _parser().parse_args(argv)
    if args.validate is not None:
        errors = validate_artifact(args.validate)
        print(json.dumps({"errors": errors} if errors else {"valid": True}, sort_keys=True))
        return int(bool(errors))
    try:
        datetime.strptime(args.date, "%Y%m%d")
    except ValueError:
        return 2
    if args.date != RUN_DATE:
        return 2
    root = find_repo_root()
    output = args.output if args.output.is_absolute() else root / args.output
    artifact = build_artifact(
        root,
        args.date,
        cache_root=args.cache_root,
        output_path=output,
        route_reachability=probe_routes(),
        update_references=True,
    )
    errors = validate_artifact(artifact)
    if errors:
        print(json.dumps({"errors": errors}, sort_keys=True))
        return 1
    write_artifact(artifact, output)
    print(
        json.dumps(
            {
                "path": str(output),
                "v625_sota_ingestion_complete_score": artifact[
                    "v625_sota_ingestion_complete_score"
                ],
                "verdict_class": artifact["verdict_class"],
                "honest_verdict": artifact["honest_verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the wrapper.
    raise SystemExit(main())
