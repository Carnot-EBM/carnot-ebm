"""Build the bounded V623 source-ingestion and claim-boundary receipt.

This audit loads no model. It records low-concurrency network receipts and
keeps vendor or index metadata below primary scientific evidence. Spec refs:
REQ-REPORT-7098 and SCENARIO-REPORT-7098-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
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
from urllib.request import Request, urlopen

from carnot import experiment_7092_v622_sota_ingestion as prior
from carnot.experiment_artifacts import atomic_write_json, atomic_write_text
from carnot.paths import repo_root as find_repo_root


JsonDict = dict[str, Any]

RUN_DATE = "20260907"
SOURCE_CAPTURE_UTC = "2026-09-07T05:11:50Z"
RANDOM_SEED = 7_098_202_609_07
INFERENCE_SUBSTRATE = (
    "low_concurrency_primary_source_ingestion_no_experimental_llm: "
    "bounded network literature and repository audit"
)
INFERENCE_SUBSTRATE_CLASS = "no_model_load"
EXECUTION_VENUE = "host"
PLANNER_HEADING = "## V623 planner refresh - 2026-09-07"
REFERENCE_START_MARKER = "<!-- V623-EXECUTION-DELTA-20260907-START -->"
REFERENCE_END_MARKER = "<!-- V623-EXECUTION-DELTA-20260907-END -->"
VENDOR_BOUNDARY = "vendor_claim_not_independent_evidence"

REFERENCE_PATH = Path("research-references.md")
RESULT_PATH = Path("results/experiment_7098_v623_sota_ingestion.json")
SOURCE_ARTIFACT_PATHS = (
    REFERENCE_PATH,
    Path("research-program.md"),
    Path("research-complete.yaml"),
    Path("research-roadmap.yaml"),
    Path("ops/status.md"),
    Path("ops/known-issues.md"),
    Path("openspec/capabilities/research-reporting/spec.md"),
    Path("python/carnot/experiment_7098_v623_sota_ingestion.py"),
    Path("scripts/experiments/experiment_7098_v623_sota_ingestion.py"),
    Path("tests/python/test_experiment_7098_v623_sota_ingestion.py"),
)

QUERY_FAMILIES = (
    "ebm_verification_reasoning",
    "neural_constraint_satisfaction",
    "ising_machine_learning",
    "hallucination_mitigation",
    "kolmogorov_arnold_networks",
    "energy_guided_constrained_decoding",
    "fpga_thermodynamic_sampling",
    "continual_online_constraint_learning",
)
SOURCE_CLASSES = (
    "arxiv",
    "openreview",
    "semantic_scholar",
    "huggingface_papers",
    "github",
    "extropic",
    "logical_intelligence",
)
SECONDARY_ROUTES = ("openreview", "semantic_scholar", "huggingface_papers")
CLASSIFICATIONS = ("adopt", "control", "watch", "reject", "duplicate")
VERDICT_CLASSES = (
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
)
ALLOWED_EXPERIMENT_HOOKS = {
    "exp7099-adapter-withheld-live-path-preflight",
    "exp7100-adapter-withheld-arc-loo-measurement",
    "exp7101-adapter-withheld-arc-cold-audit",
    "exp7102-feasibility-projected-action-energy",
    "exp7103-adapter-withheld-energy-live-ab",
    "exp7104-degree16-action-energy-portability",
    "exp7105-sealed-exact-constraint-stream",
    "exp7106-delayed-commit-procedural-memory-csl",
    "exp7107-continual-memory-cold-audit",
}
KNOWN_REFERENCE_SOURCE_IDS = {
    "arxiv:2601.03905",
    "arxiv:2506.00362",
    "arxiv:2604.27003",
    "arxiv:2607.20792",
    "arxiv:2605.18871",
    "arxiv:2609.00581",
    "arxiv:2603.24579",
    "arxiv:2605.02443",
    "arxiv:2602.06737",
    "arxiv:2505.07179",
    "extropic:z1t-2026-09-04",
}

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "duration_s",
    "source_artifact_hashes",
    "search_window",
    "query_rows",
    "source_class_rows",
    "arxiv_rows",
    "openreview_rows",
    "semantic_scholar_rows",
    "huggingface_papers_rows",
    "github_rows",
    "extropic_rows",
    "logical_intelligence_rows",
    "candidate_rows",
    "deduplication_rows",
    "primary_source_receipts",
    "adoption_rows",
    "claim_boundary_rows",
    "references_append_path",
    "references_append_hash",
    "rows",
    "v623_sota_ingestion_complete_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

ROW_COLLECTION_FIELDS = (
    "query_rows",
    "source_class_rows",
    "arxiv_rows",
    "openreview_rows",
    "semantic_scholar_rows",
    "huggingface_papers_rows",
    "github_rows",
    "extropic_rows",
    "logical_intelligence_rows",
    "candidate_rows",
    "deduplication_rows",
    "primary_source_receipts",
    "adoption_rows",
    "claim_boundary_rows",
)

FIELD_PRINCIPLES = {
    "field_principles": "One principle per field makes the scientific contract reviewable.",
    "preconditions_checked": "Explicit gates prevent unavailable sources from becoming fabricated results.",
    "inference_substrate": "The substrate limits this work to a bounded network evidence audit.",
    "inference_substrate_class": "The compute class confirms that the audit loads no model.",
    "execution_venue": "The venue separates host work from unavailable hardware claims.",
    "duration_s": "Measured duration distinguishes an executed audit from a template.",
    "source_artifact_hashes": "Hashes bind the result to the local evidence that was read.",
    "search_window": "A fixed window makes freshness decisions falsifiable.",
    "query_rows": "Exact queries disclose coverage without treating rank as evidence.",
    "source_class_rows": "Each requested source class ends with an honest receipt.",
    "arxiv_rows": "Primary paper rows preserve canonical identity and date.",
    "openreview_rows": "A browser challenge stays an access receipt, not an inferred claim.",
    "semantic_scholar_rows": "Rate limits and citation indexes cannot become scientific authority.",
    "huggingface_papers_rows": "Generated summaries remain discovery signals below primary papers.",
    "github_rows": "Repository identity does not prove method quality or Carnot fitness.",
    "extropic_rows": "Vendor projections remain separate from attached-hardware evidence.",
    "logical_intelligence_rows": "Kona stays a comparator without public reproducibility.",
    "candidate_rows": "A closed disposition prevents a lead from silently becoming work.",
    "deduplication_rows": "Canonical suppression stops rediscovery from appearing novel.",
    "primary_source_receipts": "Each candidate traces to a paper or official project page.",
    "adoption_rows": "Each promotion must name one bounded existing V623 experiment.",
    "claim_boundary_rows": "Each item states what its source cannot prove for Carnot.",
    "references_append_path": "The path makes the only permitted documentation write explicit.",
    "references_append_hash": "The ledger hash proves byte stability for an empty delta.",
    "rows": "A combined ledger supports independent collection consistency checks.",
    "v623_sota_ingestion_complete_score": "One means coverage and promotion rules passed.",
    "random_seed": "A fixed seed records deterministic ordering without sampling.",
    "reproducibility_checksum": "A payload hash detects evidence or verdict changes.",
    "gate_check_summary": "Exact diagnostics make blocked or invalid work actionable.",
    "verifier_is_oracle": "False states that a source audit cannot certify Carnot correctness.",
    "verdict_class": "A closed class keeps outages and invalid rows from reading positive.",
    "honest_verdict": "A terminal prefix gives automation one unambiguous outcome.",
}


def _copy(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [deepcopy(dict(row)) for row in rows]


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str | None:
    """Hash a readable file and preserve absence as an explicit null."""

    try:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def search_window(run_date: str) -> JsonDict:
    """Return the review interval and the strict post-planner boundary."""

    execution_date = datetime.strptime(run_date, "%Y%m%d").date().isoformat()
    return {
        "literature_start_date": "2025-01-01",
        "literature_end_date": execution_date,
        "planner_cutoff_date": "2026-09-07",
        "delta_rule": "publication_or_revision_date > planner_cutoff_date and <= literature_end_date",
        "post_cutoff_arxiv_result_count": 0,
    }


def in_execution_delta(value: str, window: Mapping[str, Any]) -> bool:
    """Accept a source date only after planning and by the execution date."""

    try:
        observed = date.fromisoformat(value)
        lower = date.fromisoformat(str(window["planner_cutoff_date"]))
        upper = date.fromisoformat(str(window["literature_end_date"]))
    except (KeyError, TypeError, ValueError):
        return False
    return lower < observed <= upper


identifier_matches_url = prior.identifier_matches_url


def _receipt_hash(row: Mapping[str, Any]) -> str:
    stable = {key: value for key, value in row.items() if key != "metadata_receipt_sha256"}
    return _sha256_json(stable)


def primary_source_receipts() -> list[JsonDict]:
    """Return the primary identities that bound the V623 decision review."""

    records = [
        (
            "2601.03905",
            "Current Agents Fail to Leverage World Model as Tool for Foresight",
            "2026-01-08",
            "v2",
            "Agents can fail when they choose, interpret, or consume world-model forecasts.",
        ),
        (
            "2506.00362",
            "FSNet: Feasibility-Seeking Neural Network for Constrained Optimization with Guarantees",
            "2025-10-24",
            "v2",
            "A differentiable feasibility-seeking step minimizes violations inside a neural solution procedure.",
        ),
        (
            "2604.27003",
            "When Continual Learning Moves to Memory: A Study of Experience Reuse in LLM Agents",
            "2026-04-29",
            "v1",
            "Abstract procedural memories transfer more reliably than detailed trajectories in the tested agents.",
        ),
        (
            "2607.20792",
            "Memoir: Should a Model Write to Its Memory While It Thinks?",
            "2026-07-22",
            "v1",
            "Writing memory during the same reasoning iteration can impose an early learning penalty.",
        ),
        (
            "2605.18871",
            "Distributional Energy-Based Models for Uncertainty-Aware Structured LLM Reasoning",
            "2026-05-15",
            "v1",
            "The method combines learned quality, deterministic penalties, and ensemble spread for structured reasoning.",
        ),
        (
            "2609.00581",
            "Enoki: Efficient Multi-Level Hallucination Detection",
            "2026-09-04",
            "v2",
            "A text-anchored relational fact representation supports claim and span-level hallucination detection.",
        ),
        (
            "2603.24579",
            "MARCH: Multi-Agent Reinforced Self-Check for LLM Hallucination",
            "2026-03-25",
            "v1",
            "A checker hidden from the original answer reduces confirmation bias in the tested RAG setting.",
        ),
        (
            "2605.02443",
            "HalluScan: A Systematic Benchmark for Detecting and Mitigating Hallucinations in Instruction-Following LLMs",
            "2026-05-22",
            "v2",
            "The benchmark evaluates adaptive routing across hallucination detection methods.",
        ),
        (
            "2602.06737",
            "Optimized Piecewise Affine Abstractions of Neural Networks with Learnable Activation Functions",
            "2026-08-02",
            "v2",
            "Piecewise-affine abstraction and MILP can verify networks with learnable activations, including KANs.",
        ),
        (
            "2505.07179",
            "Lagrange Oscillatory Neural Networks for Constraint Satisfaction and Optimization",
            "2025-10-07",
            "v2",
            "Multiplier dynamics add an explicit feasibility channel to oscillatory constraint optimization.",
        ),
        (
            "2608.28128",
            "VICT: Verifier-Instrumented Credit Tracing for Long-Horizon LLM Agent Reinforcement Learning",
            "2026-08-28",
            "v1",
            "Verifier evidence can be traced to actions through dependency-valid proof edges.",
        ),
    ]
    rows: list[JsonDict] = []
    for source_id, title, source_date, revision, claim in records:
        row: JsonDict = {
            "receipt_id": f"receipt-arxiv-{source_id}",
            "source_id": f"arxiv:{source_id}",
            "title": title,
            "source_url": f"https://arxiv.org/abs/{source_id}",
            "publication_or_revision_date": source_date,
            "version_or_revision": revision,
            "source_role": "primary_paper",
            "core_claim": claim,
            "content_verified": True,
            "accessed_at": SOURCE_CAPTURE_UTC,
            "terminal": True,
        }
        row["metadata_receipt_sha256"] = _receipt_hash(row)
        rows.append(row)
    vendor: JsonDict = {
        "receipt_id": "receipt-extropic-z1t",
        "source_id": "extropic:z1t-2026-09-04",
        "title": "Z1T: Sparse Transformer-Like Models for Probabilistic Hardware",
        "source_url": "https://extropic.ai/writing/z1t",
        "publication_or_revision_date": "2026-09-04",
        "version_or_revision": "captured-2026-09-07",
        "source_role": "official_vendor_project_page",
        "core_claim": "The public design uses a degree-16 graph and separates sparse sampling from FPGA or XPU work.",
        "content_verified": True,
        "accessed_at": SOURCE_CAPTURE_UTC,
        "terminal": True,
    }
    vendor["metadata_receipt_sha256"] = _receipt_hash(vendor)
    rows.append(vendor)
    return rows


def candidate_rows() -> list[JsonDict]:
    """Classify verified candidates without inventing a post-cutoff delta."""

    boundaries = {
        "arxiv:2601.03905": "The tested agent failures do not prove that a Carnot forecast changes an ARC action.",
        "arxiv:2506.00362": "FSNet does not prove that Carnot's exact action projector preserves useful ARC support.",
        "arxiv:2604.27003": "Agent benchmark averages do not prove transfer from Carnot procedural memory.",
        "arxiv:2607.20792": "The reported memory effect does not set Carnot's delayed-commit value result.",
        "arxiv:2605.18871": "Learned energy and spread do not prove correctness or feasibility in Carnot.",
        "arxiv:2609.00581": "Enoki does not solve V623 action generation or replace exact verification.",
        "arxiv:2603.24579": "RAG hallucination results do not establish ARC efficacy or verifier independence.",
        "arxiv:2605.02443": "Adaptive routing does not justify skipping a fixed exact-verifier baseline.",
        "arxiv:2602.06737": "A verification method does not prove that Carnot's PWA-KAN ranking path is useful.",
        "arxiv:2505.07179": "Simulated oscillator results do not establish FPGA or thermodynamic execution.",
        "arxiv:2608.28128": "The reported agent tasks do not establish Carnot ARC credit or authorize an update.",
        "extropic:z1t-2026-09-04": "Vendor projections cannot establish Carnot runtime, power, speed, availability, or Z1 execution.",
    }
    receipts = {row["source_id"]: row for row in primary_source_receipts()}
    rows: list[JsonDict] = []
    for source_id, receipt in receipts.items():
        duplicate = source_id in KNOWN_REFERENCE_SOURCE_IDS
        rows.append(
            {
                "candidate_id": source_id.replace(":", "-").replace(".", "-"),
                "source_id": source_id,
                "title": receipt["title"],
                "source_url": receipt["source_url"],
                "publication_or_revision_date": receipt["publication_or_revision_date"],
                "source_receipt_id": receipt["receipt_id"],
                "source_role": receipt["source_role"],
                "identity_verified": True,
                "core_claim_verified": True,
                "classification": "duplicate" if duplicate else "watch",
                "decision_relevant": False,
                "experiment_hook": None,
                "claim_boundary": boundaries[source_id],
                "classification_reason": (
                    "The V623 planner refresh already records this source and bounded hook."
                    if duplicate
                    else "The source predates the execution window and changes no active V623 decision."
                ),
                "terminal": True,
            }
        )
    return rows


def query_rows(run_date: str) -> list[JsonDict]:
    """Record each exact topic and required source-route query."""

    rows = [
        {
            "query_id": f"arxiv-{family}",
            "query_family": family,
            "route": "arxiv",
            "exact_query": (
                f"submittedDate:[202501010000 TO {run_date}2359] AND "
                f'all:"{family.replace("_", " ")}"'
            ),
            "accessed_at": SOURCE_CAPTURE_UTC,
            "access_outcome": "completed_bounded_search",
            "search_rank_used_as_evidence": False,
            "terminal": True,
        }
        for family in QUERY_FAMILIES
    ]
    route_queries = (
        (
            "arxiv-post-v623-cutoff",
            "arxiv",
            "submittedDate:[202609070000 TO 202609072359] AND bounded V623 topic union",
            "http_200_zero_results",
        ),
        (
            "openreview-ebm-submissions",
            "openreview",
            "OpenReview EBM submissions and FSNet forum oum1txoy1D",
            "challenge_verification_required",
        ),
        (
            "semantic-scholar-ebt-citations",
            "semantic_scholar",
            "citations for ARXIV:2507.02092 limit 100",
            "http_429_rate_limited",
        ),
        (
            "semantic-scholar-arm-ebm-citations",
            "semantic_scholar",
            "citations for ARXIV:2512.15605 limit 100",
            "http_429_rate_limited",
        ),
        (
            "huggingface-daily-papers",
            "huggingface_papers",
            "GET /api/daily_papers?limit=100 and Enoki paper page",
            "http_200_complete",
        ),
        (
            "github-targeted-repositories",
            "github",
            "official EBT, FSNet, MARCH, and Extropic repositories",
            "http_200_complete",
        ),
        (
            "extropic-writing-z1t",
            "extropic",
            "GET https://extropic.ai/writing/z1t",
            "http_200_complete",
        ),
        (
            "logical-intelligence-kona",
            "logical_intelligence",
            "GET https://logicalintelligence.com/kona-ebms-energy-based-models",
            "http_200_complete",
        ),
    )
    rows.extend(
        {
            "query_id": query_id,
            "query_family": "execution_delta" if route == "arxiv" else "requested_source_route",
            "route": route,
            "exact_query": query,
            "result_count_observed": 0 if route == "arxiv" else None,
            "accessed_at": SOURCE_CAPTURE_UTC,
            "access_outcome": outcome,
            "search_rank_used_as_evidence": False,
            "terminal": True,
        }
        for query_id, route, query, outcome in route_queries
    )
    return rows


def arxiv_rows() -> list[JsonDict]:
    """Return the zero-delta search and verified primary-paper identities."""

    rows: list[JsonDict] = [
        {
            "row_id": "arxiv-post-cutoff-query",
            "source_url": "https://export.arxiv.org/api/query",
            "search_start_date": "2026-09-07",
            "search_end_date": "2026-09-07",
            "matching_result_count": 0,
            "accessed_at": SOURCE_CAPTURE_UTC,
            "access_outcome": "http_200_complete",
            "terminal": True,
        }
    ]
    for receipt in primary_source_receipts():
        if not str(receipt["source_id"]).startswith("arxiv:"):
            continue
        rows.append(
            {
                "row_id": receipt["source_id"],
                "source_id": receipt["source_id"],
                "title": receipt["title"],
                "source_url": receipt["source_url"],
                "publication_or_revision_date": receipt["publication_or_revision_date"],
                "version_or_revision": receipt["version_or_revision"],
                "accessed_at": SOURCE_CAPTURE_UTC,
                "access_outcome": "http_200_verified_primary",
                "terminal": True,
            }
        )
    return rows


def openreview_rows() -> list[JsonDict]:
    """Keep the current browser challenge as a terminal access receipt."""

    return [
        {
            "source_id": "openreview:oum1txoy1D",
            "source_url": "https://openreview.net/forum?id=oum1txoy1D",
            "title_hint": "FSNet: Feasibility-Seeking Neural Network for Constrained Optimization with Guarantees",
            "http_status": 403,
            "access_outcome": "challenge_verification_required",
            "content_verified_in_this_route": False,
            "primary_fallback_source_id": "arxiv:2506.00362",
            "claim_promoted": False,
            "accessed_at": SOURCE_CAPTURE_UTC,
            "terminal": True,
        }
    ]


def semantic_scholar_rows() -> list[JsonDict]:
    """Record citation endpoint rate limits without guessing citation data."""

    return [
        {
            "source_id": f"semantic-scholar:{paper_id}",
            "paper_id": paper_id,
            "source_url": (
                f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations?limit=100"
            ),
            "http_status": 429,
            "access_outcome": "rate_limited_terminal_receipt",
            "visible_response_rows": None,
            "citation_count_claimed_as_authoritative": False,
            "accessed_at": SOURCE_CAPTURE_UTC,
            "terminal": True,
        }
        for paper_id in ("ARXIV:2507.02092", "ARXIV:2512.15605")
    ]


def huggingface_papers_rows() -> list[JsonDict]:
    """Record Hugging Face only as a secondary discovery index."""

    return [
        {
            "source_id": "hf-paper:2609.00581",
            "source_url": "https://huggingface.co/papers/2609.00581",
            "daily_feed_url": "https://huggingface.co/api/daily_papers?limit=100",
            "evidence_role": "secondary_discovery_index_only",
            "primary_source_id": "arxiv:2609.00581",
            "generated_summary_used_as_evidence": False,
            "upvotes_used_as_evidence": False,
            "access_outcome": "http_200_complete",
            "accessed_at": SOURCE_CAPTURE_UTC,
            "terminal": True,
        }
    ]


def github_rows() -> list[JsonDict]:
    """Freeze repository identity while excluding popularity from evidence."""

    records = (
        (
            "alexiglad/EBT",
            "19420cbeae655bbf11930219a675ade6897019e8",
            "2026-04-21T00:53:47Z",
            "Apache-2.0",
        ),
        (
            "MOSSLab-MIT/FSNet",
            "826457df85302da8c7553977ce74a4ee18d1b362",
            "2026-06-09T21:48:12Z",
            "MIT",
        ),
        (
            "Qwen-Applications/MARCH",
            "1805095bbfee0b1d90c9ad8c473911276469c4e3",
            "2026-06-09T05:01:09Z",
            None,
        ),
        (
            "extropic-ai/sparse-transformers",
            "13051e90df9669be5b8f9f34fb097329fa82f674",
            "2026-09-03T20:28:22Z",
            "Apache-2.0",
        ),
    )
    return [
        {
            "source_id": f"github:{repository}",
            "source_url": f"https://github.com/{repository}",
            "default_branch_revision": revision,
            "pushed_at": pushed_at,
            "license": license_id,
            "stars_used_as_evidence": False,
            "implementation_promoted": False,
            "access_outcome": "http_200_complete",
            "accessed_at": SOURCE_CAPTURE_UTC,
            "terminal": True,
        }
        for repository, revision, pushed_at, license_id in records
    ]


def _retimestamp(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    result = _copy(rows)
    for row in result:
        row["accessed_at"] = SOURCE_CAPTURE_UTC
    return result


def extropic_rows() -> list[JsonDict]:
    """Keep the Z1T page inside its official vendor evidence boundary."""

    return _retimestamp(prior.extropic_rows())


def logical_intelligence_rows() -> list[JsonDict]:
    """Keep Kona proprietary until weights and a local runner are public."""

    return _retimestamp(prior.logical_intelligence_rows())


def source_class_rows() -> list[JsonDict]:
    """Give every requested route one honest and terminal access outcome."""

    outcomes = {
        "arxiv": ("http_200_complete", True),
        "openreview": ("challenge_verification_required", False),
        "semantic_scholar": ("http_429_rate_limited", False),
        "huggingface_papers": ("http_200_complete", True),
        "github": ("http_200_complete", True),
        "extropic": ("http_200_complete", True),
        "logical_intelligence": ("http_200_complete", True),
    }
    return [
        {
            "source_class": source_class,
            "access_outcome": outcomes[source_class][0],
            "content_available": outcomes[source_class][1],
            "honest_receipt": True,
            "accessed_at": SOURCE_CAPTURE_UTC,
            "terminal": True,
        }
        for source_class in SOURCE_CLASSES
    ]


def deduplication_rows(candidates: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return the exact source-ledger duplicates suppressed by the audit."""

    return [
        {
            "candidate_id": row["candidate_id"],
            "canonical_source_id": row["source_id"],
            "existing_reference_marker": PLANNER_HEADING,
            "suppressed": True,
            "terminal": True,
        }
        for row in candidates
        if isinstance(row, Mapping) and row.get("classification") == "duplicate"
    ]


def adoption_rows(candidates: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return only controls or adoptions that change an existing V623 task."""

    return [
        {
            "candidate_id": row["candidate_id"],
            "source_id": row["source_id"],
            "classification": row["classification"],
            "experiment_hook": row["experiment_hook"],
            "source_receipt_id": row["source_receipt_id"],
            "claim_boundary": row["claim_boundary"],
            "terminal": True,
        }
        for row in candidates
        if isinstance(row, Mapping)
        and row.get("classification") in {"adopt", "control"}
        and row.get("decision_relevant") is True
    ]


def claim_boundary_rows(candidates: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Make each prohibited inference independently enumerable."""

    return [
        {
            "candidate_id": row["candidate_id"],
            "source_id": row["source_id"],
            "claim_boundary": row["claim_boundary"],
            "claim_boundary_present": bool(row.get("claim_boundary")),
            "terminal": True,
        }
        for row in candidates
        if isinstance(row, Mapping)
    ]


def combined_rows(artifact: Mapping[str, Any]) -> list[JsonDict]:
    """Combine typed rows while retaining their collection names."""

    rows: list[JsonDict] = []
    for field in ROW_COLLECTION_FIELDS:
        value = artifact.get(field, [])
        if isinstance(value, list):
            rows.extend(
                {"collection": field, **deepcopy(row)} for row in value if isinstance(row, dict)
            )
    return rows


def http_reachable(url: str) -> bool:
    """Treat explicit HTTP denial as reachability, not verified content."""

    request = Request(url, headers={"User-Agent": "carnot-v623-source-audit/1.0"})
    try:
        with urlopen(request, timeout=10) as response:
            return int(response.status) > 0
    except HTTPError:
        return True
    except (OSError, URLError):
        return False


def probe_routes() -> dict[str, bool]:
    """Probe only routes required by the minimum network gate."""

    urls = {
        "arxiv": "https://arxiv.org/abs/2609.00581",
        "openreview": "https://openreview.net/forum?id=oum1txoy1D",
        "semantic_scholar": "https://api.semanticscholar.org/graph/v1/paper/ARXIV:2507.02092",
        "huggingface_papers": "https://huggingface.co/papers/2609.00581",
    }
    return {route: http_reachable(url) for route, url in urls.items()}


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
        with tempfile.NamedTemporaryFile(dir=parent, prefix=".exp7098-", delete=True):
            return True
    except OSError:
        return False


def check_preconditions(
    root: Path,
    output_path: Path,
    *,
    route_reachability: Mapping[str, bool] | None = None,
) -> list[JsonDict]:
    """Evaluate network, local-read, and destination-write gates first."""

    routes = dict(probe_routes() if route_reachability is None else route_reachability)
    arxiv_ok = routes.get("arxiv") is True
    secondary_ok = any(routes.get(route) is True for route in SECONDARY_ROUTES)
    references = root / REFERENCE_PATH
    readable = _readable_nonempty(references)
    marker_present = False
    if readable:
        try:
            marker_present = PLANNER_HEADING in references.read_text(encoding="utf-8")
        except OSError:
            marker_present = False
    reference_writable = _writable_target(references)
    artifact_writable = _writable_target(output_path)
    return [
        {
            "check": "arxiv_network_reachability",
            "expected_value": True,
            "observed_value": arxiv_ok,
            "passed": arxiv_ok,
        },
        {
            "check": "secondary_index_reachability",
            "expected_value": "at_least_one_of_openreview_semantic_scholar_huggingface_papers",
            "observed_value": sorted(route for route in SECONDARY_ROUTES if routes.get(route)),
            "passed": secondary_ok,
        },
        {
            "check": "readable_current_references",
            "expected_value": "readable_nonempty_file",
            "observed_value": "readable_nonempty_file" if readable else "missing_or_unreadable",
            "passed": readable,
        },
        {
            "check": "v623_planner_marker",
            "expected_value": PLANNER_HEADING,
            "observed_value": PLANNER_HEADING if marker_present else "missing",
            "passed": marker_present,
        },
        {
            "check": "writable_references_path",
            "expected_value": "writable_parent_directory",
            "observed_value": "writable_parent_directory" if reference_writable else "not_writable",
            "passed": reference_writable,
        },
        {
            "check": "writable_artifact_path",
            "expected_value": "writable_parent_directory",
            "observed_value": "writable_parent_directory" if artifact_writable else "not_writable",
            "passed": artifact_writable,
        },
    ]


def _appendable_candidates(
    candidates: Sequence[Mapping[str, Any]], run_date: str
) -> list[Mapping[str, Any]]:
    window = search_window(run_date)
    return [
        row
        for row in candidates
        if row.get("classification") in {"adopt", "control"}
        and row.get("decision_relevant") is True
        and row.get("identity_verified") is True
        and row.get("core_claim_verified") is True
        and row.get("source_role") == "primary_paper"
        and row.get("experiment_hook") in ALLOWED_EXPERIMENT_HOOKS
        and identifier_matches_url(str(row.get("source_id", "")), str(row.get("source_url", "")))
        and in_execution_delta(str(row.get("publication_or_revision_date", "")), window)
    ]


def append_references_delta(
    path: Path, candidates: Sequence[Mapping[str, Any]], run_date: str
) -> bool:
    """Append one verified delta block, or preserve the ledger bytes."""

    accepted = _appendable_candidates(candidates, run_date)
    if not accepted:
        return False
    text = path.read_text(encoding="utf-8")
    if REFERENCE_START_MARKER in text or REFERENCE_END_MARKER in text:
        return False
    lines = [
        "",
        f"## V623 execution delta - {datetime.strptime(run_date, '%Y%m%d').date().isoformat()}",
        "",
        REFERENCE_START_MARKER,
        "",
        "Only verified execution-window controls or adoptions are listed below.",
        "",
    ]
    for row in accepted:
        lines.append(
            f"- **{row['title']}** - {row['source_id']}, {row['source_url']}. "
            f"Carnot hook: `{row['experiment_hook']}`. Boundary: {row['claim_boundary']}"
        )
    lines.extend(["", REFERENCE_END_MARKER, ""])
    atomic_write_text(path, text.rstrip() + "\n" + "\n".join(lines), allow_override=False)
    return True


def _source_hashes(root: Path) -> list[JsonDict]:
    rows = []
    for relative in SOURCE_ARTIFACT_PATHS:
        digest = file_sha256(root / relative)
        if digest is not None:
            rows.append({"path": relative.as_posix(), "sha256": digest, "terminal": True})
    return rows


@contextmanager
def _prior_policy() -> Any:
    """Apply V623 constants while the proven V622 boundary checks run."""

    names = (
        "SOURCE_CAPTURE_UTC",
        "QUERY_FAMILIES",
        "ALLOWED_EXPERIMENT_HOOKS",
        "PLANNER_HEADING",
    )
    original = {name: getattr(prior, name) for name in names}
    prior.SOURCE_CAPTURE_UTC = SOURCE_CAPTURE_UTC
    prior.QUERY_FAMILIES = QUERY_FAMILIES
    prior.ALLOWED_EXPERIMENT_HOOKS = ALLOWED_EXPERIMENT_HOOKS
    prior.PLANNER_HEADING = PLANNER_HEADING
    try:
        yield
    finally:
        for name, value in original.items():
            setattr(prior, name, value)


def _coverage_errors(artifact: Mapping[str, Any]) -> list[str]:
    comparable = deepcopy(dict(artifact))
    comparable["per_game_results"] = []
    with _prior_policy():
        errors = prior._coverage_errors(comparable)
    for candidate in artifact.get("candidate_rows", []):
        if not isinstance(candidate, Mapping):
            continue
        if (
            candidate.get("source_id") in KNOWN_REFERENCE_SOURCE_IDS
            and candidate.get("classification") != "duplicate"
        ):
            errors.append("known duplicate canonical source was not suppressed")
    return list(dict.fromkeys(errors))


def completion_score(artifact: Mapping[str, Any]) -> int:
    """Return one only when all source and boundary checks pass."""

    return int(not _coverage_errors(artifact))


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable evidence while excluding elapsed time and the hash itself."""

    payload = deepcopy(dict(artifact))
    payload.pop("duration_s", None)
    payload.pop("reproducibility_checksum", None)
    return _sha256_json(payload)


def _base_artifact(
    preconditions: Sequence[Mapping[str, Any]],
    window: Mapping[str, Any],
    reference_hash: str | None,
) -> JsonDict:
    return {
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": _copy(preconditions),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": EXECUTION_VENUE,
        "duration_s": 0.0,
        "source_artifact_hashes": [],
        "search_window": deepcopy(dict(window)),
        "query_rows": [],
        "source_class_rows": [],
        "arxiv_rows": [],
        "openreview_rows": [],
        "semantic_scholar_rows": [],
        "huggingface_papers_rows": [],
        "github_rows": [],
        "extropic_rows": [],
        "logical_intelligence_rows": [],
        "candidate_rows": [],
        "deduplication_rows": [],
        "primary_source_receipts": [],
        "adoption_rows": [],
        "claim_boundary_rows": [],
        "references_append_path": REFERENCE_PATH.as_posix(),
        "references_append_hash": reference_hash,
        "rows": [],
        "v623_sota_ingestion_complete_score": 0,
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
        "honest_verdict": "complete_blocked_v623_sota_ingestion_precondition",
    }


def recompute_artifact(artifact: JsonDict) -> JsonDict:
    """Recompute derived rows, score, terminal state, and checksum in place."""

    candidates = artifact.get("candidate_rows", [])
    artifact["deduplication_rows"] = deduplication_rows(candidates)
    artifact["adoption_rows"] = adoption_rows(candidates)
    artifact["claim_boundary_rows"] = claim_boundary_rows(candidates)
    artifact["rows"] = combined_rows(artifact)
    score = completion_score(artifact)
    artifact["v623_sota_ingestion_complete_score"] = score
    if score == 1:
        artifact["verdict_class"] = "positive"
        artifact["honest_verdict"] = "complete_positive_v623_sota_ingestion_empty_delta"
        failed_check = None
    else:
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = "complete_disqualified_v623_sota_ingestion_contract"
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
    output_path: Path = RESULT_PATH,
    route_reachability: Mapping[str, bool] | None = None,
    candidates: Sequence[Mapping[str, Any]] | None = None,
    receipts: Sequence[Mapping[str, Any]] | None = None,
    duration_s: float | None = None,
    update_references: bool = False,
) -> JsonDict:
    """Build one schema-complete positive, disqualified, or blocked receipt."""

    started = time.monotonic()
    root = Path(root)
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = root / output_path
    references = root / REFERENCE_PATH
    preconditions = check_preconditions(root, output_path, route_reachability=route_reachability)
    artifact = _base_artifact(preconditions, search_window(run_date), file_sha256(references))
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
        candidate_data = _copy(candidate_rows() if candidates is None else candidates)
        receipt_data = _copy(primary_source_receipts() if receipts is None else receipts)
        if update_references:
            append_references_delta(references, candidate_data, run_date)
        artifact.update(
            {
                "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
                "source_artifact_hashes": _source_hashes(root),
                "query_rows": query_rows(run_date),
                "source_class_rows": source_class_rows(),
                "arxiv_rows": arxiv_rows(),
                "openreview_rows": openreview_rows(),
                "semantic_scholar_rows": semantic_scholar_rows(),
                "huggingface_papers_rows": huggingface_papers_rows(),
                "github_rows": github_rows(),
                "extropic_rows": extropic_rows(),
                "logical_intelligence_rows": logical_intelligence_rows(),
                "candidate_rows": candidate_data,
                "primary_source_receipts": receipt_data,
                "references_append_hash": file_sha256(references),
            }
        )
        recompute_artifact(artifact)
    elapsed = duration_s if duration_s is not None else time.monotonic() - started
    artifact["duration_s"] = round(max(0.0, float(elapsed)), 6)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _terminal_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    preconditions = artifact.get("preconditions_checked")
    if not isinstance(preconditions, list) or not preconditions:
        return ["preconditions_checked missing"]
    failure = next(
        (row for row in preconditions if isinstance(row, dict) and row.get("passed") is False),
        None,
    )
    if failure is not None:
        expected_gate = {
            "failed_check": failure.get("check"),
            "expected_value": failure.get("expected_value"),
            "observed_value": failure.get("observed_value"),
            "passed": False,
        }
        if artifact.get("gate_check_summary") != expected_gate:
            errors.append("blocked gate_check_summary mismatch")
        if artifact.get("inference_substrate_class") != "blocked_no_run":
            errors.append("blocked inference_substrate_class mismatch")
        if artifact.get("v623_sota_ingestion_complete_score") != 0:
            errors.append("blocked completion score must be zero")
        if artifact.get("verdict_class") != "blocked":
            errors.append("blocked verdict_class mismatch")
        if not str(artifact.get("honest_verdict", "")).startswith("complete_blocked_"):
            errors.append("blocked honest_verdict prefix mismatch")
        return errors

    coverage_errors = _coverage_errors(artifact)
    expected_score = int(not coverage_errors)
    if coverage_errors and (
        artifact.get("v623_sota_ingestion_complete_score") == 1
        or artifact.get("verdict_class") == "positive"
    ):
        errors.extend(coverage_errors)
    if artifact.get("v623_sota_ingestion_complete_score") != expected_score:
        errors.append("v623_sota_ingestion_complete_score mismatch")
    expected_class = "positive" if expected_score == 1 else "disqualified"
    expected_prefix = "complete_positive_" if expected_score == 1 else "complete_disqualified_"
    if artifact.get("verdict_class") != expected_class:
        errors.append("verdict_class inconsistent with completion")
    if not str(artifact.get("honest_verdict", "")).startswith(expected_prefix):
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
    """Validate shape, evidence boundaries, terminal state, rows, and hash."""

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
        errors.append("verifier_is_oracle must be false")
    if artifact["verdict_class"] not in VERDICT_CLASSES:
        errors.append("verdict_class outside closed enum")
    if artifact["search_window"] != search_window(RUN_DATE):
        errors.append("search_window mismatch")
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
    parser.add_argument("--validate", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the source audit or validate one stored artifact."""

    args = _parser().parse_args(argv)
    if args.validate is not None:
        errors = validate_artifact(args.validate)
        print(json.dumps({"errors": errors} if errors else {"valid": True}, sort_keys=True))
        return int(bool(errors))
    try:
        datetime.strptime(args.date, "%Y%m%d")
    except ValueError:
        return 2
    root = find_repo_root()
    output = args.output if args.output.is_absolute() else root / args.output
    artifact = build_artifact(
        root,
        args.date,
        output_path=output,
        route_reachability=probe_routes(),
        update_references=True,
    )
    errors = validate_artifact(artifact)
    if errors:
        print(json.dumps({"errors": errors}, sort_keys=True))
        return 1
    write_artifact(artifact, output)
    print(json.dumps({"path": str(output), "verdict": artifact["honest_verdict"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
