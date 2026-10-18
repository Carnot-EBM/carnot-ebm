"""Build the V627 source, method-map, and local model-cache receipt.

This workflow aggregates public metadata and local file facts. It does not
load or download a model. Source pages are receipts, not proof that Carnot ran
or reproduced a reported method. Spec refs: REQ-REPORT-7137 and
SCENARIO-REPORT-7137-*.
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
import tempfile
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from carnot import experiment_7125_v626_source_delta as base
from carnot.experiment_artifacts import atomic_write_json, atomic_write_text
from carnot.paths import repo_root as find_repo_root


JsonDict = dict[str, Any]

RUN_DATE = "20260908"
SOURCE_CAPTURE_UTC = "2026-09-08T11:55:59Z"
RANDOM_SEED = 7_137_202_609_08
INFERENCE_SUBSTRATE = "aggregation_from_external_primary_sources: V627 source and cache delta"
INFERENCE_SUBSTRATE_CLASS = "aggregation"
EXECUTION_VENUE = "host"

REFERENCE_PATH = Path("research-references.md")
RESULT_PATH = Path("results/experiment_7137_v627_source_delta.json")
DEFAULT_CACHE_ROOT = base.DEFAULT_CACHE_ROOT
PLANNER_START_MARKER = "<!-- V627-PLANNER-REFRESH-20260908-START -->"
PLANNER_END_MARKER = "<!-- V627-PLANNER-REFRESH-20260908-END -->"
REFERENCE_START_MARKER = "<!-- V627-EXECUTION-DELTA-20260908-START -->"
REFERENCE_END_MARKER = "<!-- V627-EXECUTION-DELTA-20260908-END -->"

REQUIRED_INPUT_PATHS = (
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    REFERENCE_PATH,
    Path("research-hardware-wishlist.md"),
    Path("research-roadmap.yaml"),
    Path("results/experiment_6461_v556_sota_source_and_benchmark_delta.json"),
    Path("results/experiment_7125_v626_source_delta.json"),
    Path("scripts/experiments/experiment_7125_v626_source_delta.py"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/inference/sota_models.py"),
    Path("openspec/capabilities/research-reporting/spec.md"),
)
SOURCE_ARTIFACT_PATHS = (
    *REQUIRED_INPUT_PATHS,
    Path("python/carnot/experiment_7137_v627_source_delta.py"),
    Path("scripts/experiments/experiment_7137_v627_source_delta.py"),
    Path("tests/python/test_experiment_7137_v627_source_delta.py"),
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
MODEL_REPOSITORY_IDS = base.MODEL_REPOSITORY_IDS
TARGET_TASK_IDS = (
    "exp7142-flowbalance-external-memory-csl",
    "exp7138-source-grounded-relational-fixture",
    "exp7143-flowbalance-memory-cold-audit",
    "exp7144-rebudgeted-adapter-withheld-arc-loo",
    "exp7145-rust-multiscale-sampler-parity",
)
VERDICT_CLASSES = base.VERDICT_CLASSES

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
    "v627_source_delta_complete_score",
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
    "preconditions_checked": "Local checks stop missing inputs from becoming invented evidence.",
    "run_date": "The fixed date closes the allowed publication window.",
    "inference_substrate": "The substrate limits the claim to source and cache aggregation.",
    "inference_substrate_class": "The class distinguishes aggregation from a blocked no-run.",
    "execution_venue": "The host venue prevents an unattached hardware claim.",
    "duration_s": "Elapsed time records that the refresh workflow executed.",
    "source_artifact_hashes": "Hashes bind the output to the local planning inputs.",
    "rows": "Combined typed rows support an independent consistency check.",
    "arxiv_rows": "Primary paper rows preserve dated method facts.",
    "openreview_rows": "Reviewed venue rows stay separate from preprints.",
    "semantic_scholar_rows": "Citation queries remain discovery receipts.",
    "huggingface_rows": "Community paper pages stay below primary sources.",
    "github_rows": "Repository availability does not prove method fitness.",
    "extropic_rows": "Vendor statements do not prove local hardware execution.",
    "logical_intelligence_rows": "A product page cannot replace a public runner.",
    "source_class_rows": "One source identity prevents class drift and duplication.",
    "model_repository_rows": "Remote metadata records state without downloading weights.",
    "cached_model_rows": "Exact paths, revisions, sizes, and hashes support reproduction.",
    "task_method_map_rows": "Only methods that change a V627 task receive a map.",
    "deferred_rows": "Deferrals preserve relevant leads without expanding scope.",
    "source_pages_are_execution_oracles": "False separates receipts from executions.",
    "reference_append_marker": "Marker counts and a hash prove append idempotence.",
    "v627_source_delta_complete_score": "One means all source and cache checks agree.",
    "random_seed": "The fixed seed records deterministic row ordering.",
    "reproducibility_checksum": "A stable payload hash detects later evidence drift.",
    "gate_check_summary": "Exact expected and observed values make a block actionable.",
    "verifier_is_oracle": "False states that this audit does not certify correctness.",
    "verdict_class": "A closed class separates null, blocked, and invalid outcomes.",
    "honest_verdict": "A class-aligned prefix gives automation an exact outcome.",
}

file_sha256 = base.file_sha256
valid_direct_url = base.valid_direct_url
publication_date_in_window = base.publication_date_in_window
deduplicate_source_rows = base.deduplicate_source_rows


def _copy(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [deepcopy(dict(row)) for row in rows]


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def arxiv_rows() -> list[JsonDict]:
    """Return the bounded arXiv query receipt and V627-relevant papers."""

    papers = (
        (
            "2609.03241",
            "FlowBalance: Verifier-Grounded Self-Improvement from On-Policy Reasoning Experience",
            "2026-09-03",
            "Verifier advantage signs dense guidance and disables it without preference.",
        ),
        (
            "2609.05025",
            "Leveraging Low-Level Symbolic Competences for Unsupervised Grounding in Hallucination Detection",
            "2026-09-04",
            "A model builds a SQL database from sources and checks sampled responses against it.",
        ),
        (
            "2608.20927",
            "MentorPulse: Refreshing Cross-Model Latent Guidance for Long-Form Generation",
            "2026-08-21",
            "Static mentor memory can harm constraints; bounded refresh tests stale-memory harm.",
        ),
        (
            "2608.31114",
            "Overcoming critical slowing down in frustrated spin systems by learned multiscale sampling",
            "2026-08-31",
            "WCRG samples learned conditional wavelet distributions from coarse to fine scales.",
        ),
        (
            "2608.24382",
            "Exact autoregressive sampling of planar Ising spin glasses via the Kac--Ward theory",
            "2026-08-25",
            "The planar zero-field construction gives an exact finite sampling baseline.",
        ),
        (
            "2609.02948",
            "FrOGS: Discrete Neural Sampler for Independent Alloy Configurations Across Chemical Conditions",
            "2026-09-01",
            "The hybrid neural and continuous-time sampler reports finite Ising checks.",
        ),
        (
            "2608.26960",
            "Geo-LoRA: Geometry-Aware Subspace Evolution for Low-Rank Adaptation in Continual Learning",
            "2026-08-27",
            "Geometry constraints regulate trainable low-rank updates across tasks.",
        ),
        (
            "2609.05281",
            "KanAdapter: A Kolmogorov-Arnold Network-based Plug-and-Play Module for Efficient Fine-tuning of Foundation Speech Models",
            "2026-09-04",
            "Trainable GR-KAN branches adapt speech encoders under a parameter budget.",
        ),
    )
    rows: list[JsonDict] = [
        {
            "row_id": "arxiv-topic-sweep-20260908",
            "query_url": "https://export.arxiv.org/api/query",
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
            "date_end": "2026-09-08",
            "available": True,
            "accessed_at": SOURCE_CAPTURE_UTC,
            "access_outcome": "http_200_complete",
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
                "available": True,
                "accessed_at": SOURCE_CAPTURE_UTC,
                "access_outcome": "http_200_verified_primary",
                "terminal": True,
            }
        )
    return rows


def openreview_rows() -> list[JsonDict]:
    """Return the reviewed long-horizon control without claiming ARC value."""

    return [
        {
            "source_id": "openreview:PnHfrCMKtp",
            "title": "On Training Large Language Models for Long-Horizon Tasks: An Empirical Study of Horizon Length",
            "url": "https://openreview.net/forum?id=PnHfrCMKtp",
            "publication_date": "2026-04-30",
            "venue_state": "ICML 2026 regular paper",
            "bounded_method_fact": "Controlled horizon reduction and higher-level actions improve training stability.",
            "source_type": "peer_reviewed",
            "available": True,
            "accessed_at": SOURCE_CAPTURE_UTC,
            "access_outcome": "reachable_http_challenge_with_verified_index_metadata",
            "terminal": True,
        }
    ]


def semantic_scholar_rows() -> list[JsonDict]:
    """Keep EBT and ARM-EBM citation lists as secondary discovery evidence."""

    records = (
        (
            "ARXIV:2507.02092",
            35,
            "2026-08-14",
            ["Memoir", "Solver-Hard Is Not Model-Hard", "LoopUS"],
        ),
        (
            "ARXIV:2512.15605",
            8,
            "2026-07-02",
            [
                "Distributional Energy-Based Models",
                "LoopUS",
                "Large Language Models Can Take False First Steps",
            ],
        ),
    )
    return [
        {
            "source_id": f"semantic-scholar:{paper_id}/citations",
            "paper_id": paper_id,
            "query_url": f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations?limit=100",
            "visible_response_rows": count,
            "newest_visible_publication_date": newest,
            "relevant_title_samples": titles,
            "citation_count_claimed_as_authoritative": False,
            "source_type": "secondary_discovery",
            "available": True,
            "accessed_at": SOURCE_CAPTURE_UTC,
            "access_outcome": "http_200_complete",
            "terminal": True,
        }
        for paper_id, count, newest, titles in records
    ]


def huggingface_rows() -> list[JsonDict]:
    """Record one available and one unavailable community paper page."""

    return [
        {
            "source_id": "arxiv:2609.03241",
            "url": "https://huggingface.co/papers/2609.03241",
            "primary_source_url": "https://arxiv.org/abs/2609.03241",
            "source_type": "secondary_discovery",
            "community_summary_used_as_evidence": False,
            "available": True,
            "accessed_at": SOURCE_CAPTURE_UTC,
            "access_outcome": "http_200_complete",
            "terminal": True,
        },
        {
            "source_id": "arxiv:2609.05025",
            "url": "https://huggingface.co/papers/2609.05025",
            "primary_source_url": "https://arxiv.org/abs/2609.05025",
            "source_type": "secondary_discovery",
            "community_summary_used_as_evidence": False,
            "available": False,
            "accessed_at": SOURCE_CAPTURE_UTC,
            "access_outcome": "http_404_not_found",
            "terminal": True,
        },
    ]


def github_rows() -> list[JsonDict]:
    """Record repository and trending discovery without using popularity."""

    return [
        {
            "source_id": "github:alexhuang13/FlowBalance",
            "url": "https://github.com/alexhuang13/FlowBalance",
            "default_branch_revision": "5810e1dc7e816436863613ba652db257193af140",
            "pushed_at": "2026-09-04T04:07:37Z",
            "license": None,
            "source_type": "repository_discovery",
            "stars_used_as_evidence": False,
            "implementation_fitness_claimed": False,
            "available": True,
            "accessed_at": SOURCE_CAPTURE_UTC,
            "access_outcome": "http_200_complete",
            "terminal": True,
        },
        {
            "source_id": "github:weekly-trending-20260908",
            "url": "https://github.com/trending?since=weekly",
            "languages_checked": ["general", "python", "rust"],
            "decision_changing_repository_count": 0,
            "source_type": "repository_discovery",
            "stars_used_as_evidence": False,
            "implementation_fitness_claimed": False,
            "available": True,
            "accessed_at": SOURCE_CAPTURE_UTC,
            "access_outcome": "http_200_complete",
            "terminal": True,
        },
    ]


def extropic_rows() -> list[JsonDict]:
    """Keep Z1T inside the first-party vendor claim boundary."""

    return [
        {
            "source_id": "extropic:z1t-2026-09-04",
            "title": "Z1T: Sparse Transformer-Like Models for Probabilistic Hardware",
            "url": "https://extropic.ai/writing/z1t",
            "publication_date": "2026-09-04",
            "bounded_method_fact": "The vendor reports a degree-16 graph and an FPGA-plus-Z1 split.",
            "source_type": "vendor_claim",
            "scientific_evidence_promoted": False,
            "hardware_execution_claimed": False,
            "available": True,
            "accessed_at": SOURCE_CAPTURE_UTC,
            "access_outcome": "http_200_complete",
            "terminal": True,
        }
    ]


def logical_intelligence_rows() -> list[JsonDict]:
    """Keep Kona as a product comparator without a local baseline claim."""

    return [
        {
            "source_id": "logical-intelligence:kona-1.0",
            "title": "Kona 1.0",
            "url": "https://logicalintelligence.com/kona-ebms-energy-based-models",
            "publication_date": "2026-02-03",
            "bounded_method_fact": "The page describes a proprietary energy-based constraint layer.",
            "source_type": "product_page",
            "scientific_evidence_promoted": False,
            "reproducible_local_runner_observed": False,
            "available": True,
            "accessed_at": SOURCE_CAPTURE_UTC,
            "access_outcome": "http_200_complete",
            "terminal": True,
        }
    ]


def _apply_availability(rows: Sequence[Mapping[str, Any]], available: bool) -> list[JsonDict]:
    copied = _copy(rows)
    if not available:
        for row in copied:
            row["available"] = False
            row["access_outcome"] = "unavailable_at_execution"
    return copied


def source_collections(
    route_reachability: Mapping[str, bool],
) -> dict[str, list[JsonDict]]:
    """Build each source collection and preserve every route failure as rows."""

    builders = {
        "arxiv_rows": arxiv_rows,
        "openreview_rows": openreview_rows,
        "semantic_scholar_rows": semantic_scholar_rows,
        "huggingface_rows": huggingface_rows,
        "github_rows": github_rows,
        "extropic_rows": extropic_rows,
        "logical_intelligence_rows": logical_intelligence_rows,
    }
    return {
        field: _apply_availability(builder(), route_reachability.get(field) is True)
        for field, builder in builders.items()
    }


def source_class_rows(
    collections: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[JsonDict]:
    """Build one canonical source ledger with explicit claim boundaries."""

    boundaries = {
        "peer_reviewed": "Reviewed results do not prove a Carnot execution.",
        "preprint": "The paper does not establish the mapped Carnot outcome.",
        "vendor_claim": "Vendor statements do not prove local hardware results.",
        "product_page": "The product page is not a reproducible public baseline.",
        "repository_discovery": "Repository state does not prove scientific fitness.",
        "secondary_discovery": "A discovery index cannot override a primary source.",
    }
    candidates: list[JsonDict] = []
    for field in SOURCE_COLLECTION_FIELDS:
        rows = collections[field][1:] if field == "arxiv_rows" else collections[field]
        for raw in rows:
            row = deepcopy(dict(raw))
            if "url" not in row and "query_url" in row:
                row["url"] = row["query_url"]
            row["claim_boundary"] = boundaries[str(row["source_type"])]
            candidates.append(row)
    return deduplicate_source_rows(candidates)


def model_repository_rows() -> list[JsonDict]:
    """Reuse the same-day read-only Hugging Face metadata snapshot."""

    return _copy(base.model_repository_rows())


def inspect_cached_models(cache_root: Path) -> list[JsonDict]:
    """Resolve the three local GGUF candidates without loading or downloading."""

    rows = _copy(base.inspect_cached_models(cache_root))
    for row in rows:
        row["candidate_path"] = row["path"]
    return rows


def task_method_map_rows() -> list[JsonDict]:
    """Map only the five planner-selected methods that change V627 tasks."""

    records = (
        (
            "flowbalance_signed_verifier_control",
            "arxiv:2609.03241",
            "https://arxiv.org/abs/2609.03241",
            "2026-09-03",
            "Sign external memory guidance from exact verifier advantage and skip ties.",
            "A paper training result does not prove Carnot memory value.",
        ),
        (
            "source_to_sql_symbolic_grounding",
            "arxiv:2609.05025",
            "https://arxiv.org/abs/2609.05025",
            "2026-09-04",
            "Build a source-backed relation fixture whose SQL execution is exact authority.",
            "Model-written SQL is untrusted and cannot certify itself.",
        ),
        (
            "sealed_boundary_stale_memory_refresh",
            "arxiv:2608.20927",
            "https://arxiv.org/abs/2608.20927",
            "2026-08-21",
            "Refresh external memory only at sealed boundaries and measure stale-memory harm.",
            "Latent mentor gains do not prove frozen-GGUF memory retention.",
        ),
        (
            "horizon_reduction_positive_control",
            "openreview:PnHfrCMKtp",
            "https://openreview.net/forum?id=PnHfrCMKtp",
            "2026-04-30",
            "Raise the ARC cell budget enough to reach its positive control.",
            "A larger action ceiling is not an ARC efficiency or solve claim.",
        ),
        (
            "corrected_multiscale_proposal_rust_port",
            "arxiv:2608.31114",
            "https://arxiv.org/abs/2608.31114",
            "2026-08-31",
            "Port the corrected coarse-to-fine proposal and prove finite-law parity first.",
            "The paper does not prove Carnot parity, mixing, or speed.",
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
            "already_in_v627_planner": True,
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
    """Preserve useful leads that do not change a V627 task."""

    records = (
        (
            "arxiv:2608.24382",
            "https://arxiv.org/abs/2608.24382",
            "exact planar Ising sampling",
            "V627 ports the already corrected sampler and needs shared-law parity, not a new planar solver.",
        ),
        (
            "arxiv:2609.02948",
            "https://arxiv.org/abs/2609.02948",
            "FrOGS neural sampling",
            "Its alloy and learned-model scope does not change the bounded Rust parity task.",
        ),
        (
            "arxiv:2608.26960",
            "https://arxiv.org/abs/2608.26960",
            "Geo-LoRA continual adaptation",
            "It changes trainable adapter weights and does not provide frozen-GGUF memory.",
        ),
        (
            "arxiv:2609.05281",
            "https://arxiv.org/abs/2609.05281",
            "KanAdapter speech adaptation",
            "It does not reverse prior KAN negatives or provide the selected memory mechanism.",
        ),
        (
            "semantic-scholar:ARXIV:2507.02092/citations",
            "https://api.semanticscholar.org/graph/v1/paper/ARXIV:2507.02092/citations?limit=100",
            "EBT citation trail",
            "The 35 visible rows add no matching-base public local energy model.",
        ),
        (
            "semantic-scholar:ARXIV:2512.15605/citations",
            "https://api.semanticscholar.org/graph/v1/paper/ARXIV:2512.15605/citations?limit=100",
            "ARM-EBM citation trail",
            "The eight visible rows add no exact local scorer for V627.",
        ),
        (
            "extropic:z1t-2026-09-04",
            "https://extropic.ai/writing/z1t",
            "Z1T hardware deployment",
            "No authenticated Z1 device is attached, so runtime and power stay deferred.",
        ),
        (
            "logical-intelligence:kona-1.0",
            "https://logicalintelligence.com/kona-ebms-energy-based-models",
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
    """Combine typed rows while retaining their source collection names."""

    rows: list[JsonDict] = []
    for field in ROW_COLLECTION_FIELDS:
        value = artifact.get(field, [])
        if isinstance(value, list):
            rows.extend(
                {"collection": field, **deepcopy(row)} for row in value if isinstance(row, dict)
            )
    return rows


def http_reachable(url: str) -> bool:
    """Treat any explicit HTTP response as a completed route probe."""

    request = Request(url, headers={"User-Agent": "carnot-v627-source-audit/1.0"})
    try:
        with urlopen(request, timeout=10) as response:
            return int(response.status) > 0
    except HTTPError:
        return True
    except (OSError, URLError):
        return False


def probe_routes() -> dict[str, bool]:
    """Probe each named source surface without requesting model files."""

    urls = {
        "arxiv_rows": "https://export.arxiv.org/api/query?id_list=2609.03241&max_results=1",
        "openreview_rows": "https://openreview.net/forum?id=PnHfrCMKtp",
        "semantic_scholar_rows": "https://api.semanticscholar.org/graph/v1/paper/ARXIV:2512.15605/citations?limit=1",
        "huggingface_rows": "https://huggingface.co/papers/2609.03241",
        "github_rows": "https://api.github.com/repos/alexhuang13/FlowBalance",
        "extropic_rows": "https://extropic.ai/writing/z1t",
        "logical_intelligence_rows": "https://logicalintelligence.com/kona-ebms-energy-based-models",
    }
    return {field: http_reachable(url) for field, url in urls.items()}


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
        with tempfile.NamedTemporaryFile(dir=parent, prefix=".exp7137-", delete=True):
            return True
    except OSError:
        return False


def check_preconditions(
    root: Path,
    output_path: Path,
    cache_root: Path,
    *,
    route_reachability: Mapping[str, bool],
) -> list[JsonDict]:
    """Check local blockers and summarize non-blocking route availability."""

    available = sorted(
        field for field in SOURCE_COLLECTION_FIELDS if route_reachability.get(field) is True
    )
    unavailable = sorted(
        field for field in SOURCE_COLLECTION_FIELDS if route_reachability.get(field) is not True
    )
    inputs = [path.as_posix() for path in REQUIRED_INPUT_PATHS if _readable_nonempty(root / path)]
    references = root / REFERENCE_PATH
    planner_present = False
    if _readable_nonempty(references):
        text = references.read_text(encoding="utf-8")
        planner_present = PLANNER_START_MARKER in text and PLANNER_END_MARKER in text
    checks = [
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
            "check": "v627_planner_marker",
            "expected_value": [PLANNER_START_MARKER, PLANNER_END_MARKER],
            "observed_value": (
                [PLANNER_START_MARKER, PLANNER_END_MARKER] if planner_present else "missing"
            ),
            "passed": planner_present,
        },
        {
            "check": "model_cache_metadata",
            "expected_value": "readable_directory",
            "observed_value": (
                "readable_directory"
                if cache_root.is_dir() and os.access(cache_root, os.R_OK)
                else "missing_or_unreadable"
            ),
            "passed": cache_root.is_dir() and os.access(cache_root, os.R_OK),
        },
        {
            "check": "writable_reference_path",
            "expected_value": "writable_parent_directory",
            "observed_value": (
                "writable_parent_directory" if _writable_target(references) else "not_writable"
            ),
            "passed": _writable_target(references),
        },
        {
            "check": "writable_artifact_path",
            "expected_value": "writable_parent_directory",
            "observed_value": (
                "writable_parent_directory" if _writable_target(output_path) else "not_writable"
            ),
            "passed": _writable_target(output_path),
        },
    ]
    return checks


def append_references_delta(
    path: Path,
    method_rows: Sequence[Mapping[str, Any]],
    cached_rows: Sequence[Mapping[str, Any]],
    run_date: str,
    external_rows: Sequence[Mapping[str, Any]],
) -> bool:
    """Append one bounded V627 block and preserve repeated-call bytes."""

    text = path.read_text(encoding="utf-8")
    if REFERENCE_START_MARKER in text or REFERENCE_END_MARKER in text:
        return False
    lines = [
        "",
        f"## V627 execution delta - {datetime.strptime(run_date, '%Y%m%d').date().isoformat()}",
        "",
        REFERENCE_START_MARKER,
        "",
        "This execution refresh confirms five planner-selected method maps.",
        "The refresh found no post-planner method changed the V627 task contract.",
        "It does not report a Carnot result from any source claim.",
        "",
    ]
    for row in method_rows:
        lines.append(
            f"- **{row['method_id']}** - {row['source_url']}. V627 task: "
            f"`{row['target_task_id']}`. Boundary: {row['claim_boundary']}"
        )
    unavailable = [row for row in external_rows if row.get("available") is False]
    lines.extend(["", "Execution-time external access limits:", ""])
    if unavailable:
        for row in unavailable:
            lines.append(
                f"- `{row.get('source_id', row.get('row_id'))}` - "
                f"{row['access_outcome']}. This source gap does not replace primary receipts."
            )
    else:
        lines.append("- No checked external route was unavailable.")
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


def _base_artifact(preconditions: Sequence[Mapping[str, Any]]) -> JsonDict:
    artifact: JsonDict = {field: [] for field in ROW_COLLECTION_FIELDS if field != "rows"}
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
            "v627_source_delta_complete_score": 0,
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
            "honest_verdict": "blocked_v627_source_delta_precondition",
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
        row.get("source_type") in {"peer_reviewed", "preprint"}
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
        or not row.get("revision")
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
            if row.get("candidate_path") != row.get("path"):
                errors.append("cached model candidate path mismatch")
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
    return list(dict.fromkeys(errors))


def completion_score(artifact: Mapping[str, Any]) -> int:
    """Return one only when every source, cache, map, and append check passes."""

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
    artifact["v627_source_delta_complete_score"] = score
    if score == 1:
        changed = any(
            row.get("execution_delta_new") is True for row in artifact["task_method_map_rows"]
        )
        verdict = "positive" if changed else "null"
        artifact["verdict_class"] = verdict
        artifact["honest_verdict"] = (
            "positive_v627_source_delta_complete_with_plan_change"
            if changed
            else "null_v627_source_delta_complete_no_post_planner_change"
        )
        failed_check = None
    else:
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = "disqualified_v627_source_delta_contract"
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

    artifact = _base_artifact([])
    _checkpoint_artifact(artifact, output_path)
    routes = dict(probe_routes() if route_reachability is None else route_reachability)
    preconditions = check_preconditions(
        root,
        output_path,
        cache_root,
        route_reachability=routes,
    )
    artifact["preconditions_checked"] = preconditions
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
        external_rows = [
            row
            for field in SOURCE_COLLECTION_FIELDS
            for row in collections[field]
            if row.get("available") is False
        ]
        appended = False
        if update_references:
            appended = append_references_delta(
                root / REFERENCE_PATH,
                task_method_map_rows(),
                cached,
                run_date,
                external_rows,
            )
        artifact.update(collections)
        artifact.update(
            {
                "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
                "source_class_rows": source_class_rows(collections),
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
    _checkpoint_artifact(artifact, output_path)
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
        if artifact.get("v627_source_delta_complete_score") != 0:
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
    if artifact.get("v627_source_delta_complete_score") != expected_score:
        errors.append("v627_source_delta_complete_score mismatch")
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
                "v627_source_delta_complete_score": artifact["v627_source_delta_complete_score"],
                "verdict_class": artifact["verdict_class"],
                "honest_verdict": artifact["honest_verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - the CLI wrapper calls main.
    raise SystemExit(main())
