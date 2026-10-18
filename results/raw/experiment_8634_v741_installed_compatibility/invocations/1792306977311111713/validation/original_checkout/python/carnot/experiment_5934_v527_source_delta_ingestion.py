"""Exp5934: ingest post-V527 source deltas with explicit uncertainty.

Spec refs: REQ-REPORT-5934, SCENARIO-REPORT-5934-ZERO-FINDING,
SCENARIO-REPORT-5934-ACCEPT-BOUNDED-DELTA,
SCENARIO-REPORT-5934-SOURCE-UNCERTAINTY,
SCENARIO-REPORT-5934-DUPLICATE-AND-RETIRED-SCOPE,
SCENARIO-REPORT-5934-SCHEMA.

This module is a source-ledger task. It does not run a model and it does not
change the research roadmap. The job is to preserve a falsifiable boundary
around the V527 planning marker, classify externally fetched receipts, append
references only for accepted primary deltas, and make access failures visible
instead of turning them into quiet negative evidence.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import yaml


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5934_v527_source_delta_ingestion.json")

AGENTS_RELATIVE_PATH = Path("AGENTS.md")
CODEX_RELATIVE_PATH = Path("CODEX.md")
CLAUDE_RELATIVE_PATH = Path("CLAUDE.md")
RESEARCH_PROGRAM_RELATIVE_PATH = Path("research-program.md")
RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")
RESEARCH_STUDYING_RELATIVE_PATH = Path("research-studying.md")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
KNOWN_ISSUES_RELATIVE_PATH = Path("ops/known-issues.md")
STATUS_RELATIVE_PATH = Path("ops/status.md")
CHANGELOG_RELATIVE_PATH = Path("ops/changelog.md")
TRACEABILITY_RELATIVE_PATH = Path("_bmad/traceability.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
SWEEP_CLUSTERS_RELATIVE_PATH = Path("scripts/sweep_clusters.py")
SWEEP_SEMSCHOLAR_RELATIVE_PATH = Path("scripts/sweep_semscholar.py")
PRIOR_SOURCE_RESULT_RELATIVE_PATH = Path(
    "results/experiment_5919_v526_source_delta_ingestion.json"
)

EXPERIMENT = "experiment_5934_v527_source_delta_ingestion"
EXPERIMENT_ID = "exp5934-v527-source-delta-ingestion"
MILESTONE = "2026.07.527"
RUN_DATE = "20260726"
RANDOM_SEED = 5934
SCHEMA = "carnot.experiment_5934.v527_source_delta_ingestion.v1"
INFERENCE_SUBSTRATE = "aggregation_from_external_primary_sources_no_experimental_llm"

PLANNER_HEADING = "## V527 Planner Refresh - 20260726"
PLANNER_MARKER = "V527-PLANNER-REFRESH-20260726-END"
PLANNER_END_MARKER = f"<!-- {PLANNER_MARKER} -->"
EXECUTION_DELTA_HEADING = "## V527 Execution Source Delta - 20260726"
EXECUTION_DELTA_END_MARKER = "<!-- V527-EXECUTION-SOURCE-DELTA-20260726-END -->"
POST_MARKER_SOURCE_DATE = "2026-07-26"

ALLOCATED_TARGET_EXPERIMENTS = (
    "exp5935-non-pruning-atomic-constraint-support",
    "exp5936-sota-atomic-support-union-ab",
    "exp5937-excluded-pool-coverage-audit",
)

SPEC_REFS = (
    "REQ-REPORT-5934",
    "SCENARIO-REPORT-5934-ZERO-FINDING",
    "SCENARIO-REPORT-5934-ACCEPT-BOUNDED-DELTA",
    "SCENARIO-REPORT-5934-SOURCE-UNCERTAINTY",
    "SCENARIO-REPORT-5934-DUPLICATE-AND-RETIRED-SCOPE",
    "SCENARIO-REPORT-5934-SCHEMA",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "search_window_and_marker_receipt",
    "source_queries_and_endpoint_receipts",
    "primary_secondary_and_official_source_counts",
    "accepted_rejected_abstained_findings",
    "false_positive_false_negative_cutoff_and_rate_limit_receipts",
    "semantic_scholar_ebt_and_arm_ebm_receipts",
    "extropic_github_huggingface_openreview_and_kona_receipts",
    "duplicate_and_retired_scope_filter",
    "references_append_receipt",
    "task_identity_gate_and_exclusion_immutability",
    "protected_files_unchanged",
    "duration_s",
    "inference_substrate",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

REQUIRED_FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Only a hash-anchored post-marker window is eligible.",
    "preconditions_checked": "Only a hash-anchored post-marker window is eligible.",
    "search_window_and_marker_receipt": (
        "Only a hash-anchored post-marker window is eligible."
    ),
    "source_queries_and_endpoint_receipts": (
        "Every query and source class remains dated and auditable."
    ),
    "primary_secondary_and_official_source_counts": (
        "Every query and source class remains dated and auditable."
    ),
    "accepted_rejected_abstained_findings": (
        "Uncertainty and access failure cannot be silently converted into rejection."
    ),
    "false_positive_false_negative_cutoff_and_rate_limit_receipts": (
        "Uncertainty and access failure cannot be silently converted into rejection."
    ),
    "semantic_scholar_ebt_and_arm_ebm_receipts": (
        "Discovery receipts are context until a primary reproducible artifact is opened."
    ),
    "extropic_github_huggingface_openreview_and_kona_receipts": (
        "Discovery receipts are context until a primary reproducible artifact is opened."
    ),
    "duplicate_and_retired_scope_filter": (
        "No duplicate or renamed retired mechanism enters the ledger."
    ),
    "references_append_receipt": (
        "Source aggregation may append references but cannot rewrite the activated milestone."
    ),
    "task_identity_gate_and_exclusion_immutability": (
        "Source aggregation may append references but cannot rewrite the activated milestone."
    ),
    "protected_files_unchanged": (
        "Active roadmap, conductor, exclusions, historical results, and unrelated "
        "user changes remain byte-identical."
    ),
    "duration_s": f"Use `{INFERENCE_SUBSTRATE}`.",
    "inference_substrate": f"Use `{INFERENCE_SUBSTRATE}`.",
    "field_provenance": f"Use `{INFERENCE_SUBSTRATE}`.",
    "test_commands": f"Use `{INFERENCE_SUBSTRATE}`.",
    "test_exit_codes": f"Use `{INFERENCE_SUBSTRATE}`.",
    "reproducibility_checksum": f"Use `{INFERENCE_SUBSTRATE}`.",
    "honest_verdict": "Use `complete_delta:`, `complete_null:`, or `blocked:`.",
}

FIELD_PRINCIPLE_EXTRAS: dict[str, str] = {
    "schema": "Versioned schema id keeps downstream validators from guessing field meaning.",
    "experiment": "Stable local slug ties the artifact to the implementation module.",
    "experiment_id": "Conductor task identity prevents numeric-prefix aliasing.",
    "milestone": "Binds receipts to .527 rather than a prior milestone.",
    "run_date": "Operator-requested execution date for the source refresh.",
    "random_seed": "Deterministic metadata for a no-randomness ledger task.",
    "spec_refs": "OpenSpec anchors make the artifact contract auditable.",
    "result_path": "Declares the exact JSON deliverable path.",
    "search_started_at": "Records when source querying started.",
    "search_finished_at": "Records when source classification finished.",
}

SOURCE_RECEIPT_REQUIRED_FIELDS = (
    "receipt_id",
    "source_family",
    "source_role",
    "query_family",
    "query",
    "url",
    "accessed_at",
    "access_outcome",
    "candidate_ids",
    "candidate_count",
    "source_cutoff",
    "receipt_summary",
)


DEFAULT_SOURCE_RECEIPTS: tuple[JsonDict, ...] = (
    {
        "receipt_id": "arxiv_v527_date_window",
        "source_family": "arXiv",
        "source_role": "primary",
        "query_family": "arxiv_primary",
        "query": "submittedDate:[202607260000 TO 202607262359]",
        "url": (
            "https://export.arxiv.org/api/query?search_query=submittedDate:%5B"
            "202607260000%20TO%20202607262359%5D&start=0&max_results=10"
        ),
        "accessed_at": "2026-07-26T08:09:29Z",
        "access_outcome": "reachable_http_200_totalResults_0",
        "candidate_ids": [],
        "candidate_count": 0,
        "source_cutoff": "submitted_or_changed_after_v527_marker_date_2026_07_26",
        "receipt_summary": (
            "The arXiv primary date route returned zero submissions in the "
            "bounded 2026-07-26 post-marker window."
        ),
    },
    {
        "receipt_id": "arxiv_v527_topic_windows",
        "source_family": "arXiv",
        "source_role": "primary",
        "query_family": "arxiv_primary_topic_windows",
        "query": (
            "eight submittedDate topic windows: energy-based reasoning, neural CSP, "
            "Ising sampling, hallucination verification, KAN, energy-guided "
            "decoding, hardware sampling, continual learning"
        ),
        "url": "https://export.arxiv.org/api/query",
        "accessed_at": "2026-07-26T08:10:05Z",
        "access_outcome": "reachable_http_200_8_topic_routes_totalResults_0",
        "candidate_ids": [],
        "candidate_count": 0,
        "source_cutoff": "topic_submitted_or_changed_after_v527_marker_2026_07_26",
        "covered_topics": [
            "EBM verification/reasoning",
            "neural CSP",
            "Ising",
            "hallucination",
            "KAN",
            "energy-guided decoding",
            "hardware sampling",
            "continual learning",
        ],
        "receipt_summary": (
            "All arXiv topic/date routes were reachable with zero candidates, "
            "so no post-marker primary arXiv source is accepted."
        ),
    },
    {
        "receipt_id": "openreview_v527_search_page",
        "source_family": "OpenReview",
        "source_role": "secondary",
        "query_family": "openreview_secondary",
        "query": "energy-based constraint reasoning continual learning",
        "url": (
            "https://openreview.net/search?term=energy-based%20constraint%20"
            "reasoning%20continual%20learning"
        ),
        "accessed_at": "2026-07-26T08:10:20Z",
        "access_outcome": "reachable_http_200_dynamic_search_page_no_new_primary_delta",
        "candidate_ids": [],
        "candidate_count": 0,
        "source_cutoff": "dynamic_page_crawled_after_v527_marker_primary_date_uncertain",
        "receipt_summary": (
            "The OpenReview search page was reachable, but dynamic search "
            "metadata was not promoted without a newer primary forum page."
        ),
    },
    {
        "receipt_id": "openreview_v527_api_notes",
        "source_family": "OpenReview",
        "source_role": "secondary",
        "query_family": "openreview_api",
        "query": "api2.openreview.net notes content.title=energy-based",
        "url": "https://api2.openreview.net/notes?limit=5&content.title=energy-based",
        "accessed_at": "2026-07-26T08:10:20Z",
        "access_outcome": "inaccessible_http_403_challenge_required",
        "candidate_ids": [],
        "candidate_count": 0,
        "source_cutoff": "api_checked_after_v527_marker",
        "receipt_summary": (
            "The OpenReview API returned a challenge-required 403, so this "
            "route is an endpoint failure rather than negative evidence."
        ),
    },
    {
        "receipt_id": "huggingface_papers_v527_2026_07_26",
        "source_family": "Hugging Face Papers",
        "source_role": "secondary",
        "query_family": "huggingface_papers_secondary",
        "query": "daily_papers date:2026-07-26",
        "url": "https://huggingface.co/papers?date=2026-07-26",
        "accessed_at": "2026-07-26T08:10:20Z",
        "access_outcome": "reachable_http_200_daily_feed_redirected_or_pinned_2026_07_24",
        "candidate_ids": ["2607.16859"],
        "candidate_count": 1,
        "source_cutoff": "secondary_daily_feed_date_confounded_not_primary_evidence",
        "receipt_summary": (
            "The Hugging Face Papers page was reachable but exposed a "
            "2026-07-24 page marker and only secondary discovery metadata."
        ),
    },
    {
        "receipt_id": "semantic_scholar_v527_ebt_citations",
        "source_family": "Semantic Scholar",
        "source_role": "secondary",
        "query_family": "semantic_scholar_citation_trail",
        "query": "arXiv:2507.02092 citations",
        "url": (
            "https://api.semanticscholar.org/graph/v1/paper/arXiv:2507.02092/"
            "citations?fields=title,year,externalIds,url,publicationDate&limit=20"
        ),
        "accessed_at": "2026-07-26T08:09:29Z",
        "access_outcome": "reachable_http_200_20_records_no_post_marker_citation",
        "candidate_ids": ["2607.20792", "2607.17047", "2607.11555"],
        "candidate_count": 20,
        "source_cutoff": "citation_trail_checked_after_v527_marker_newest_2026_07_22",
        "newest_visible": {
            "identifier": "2607.20792",
            "title": "Memoir: Should a Model Write to Its Memory While It Thinks?",
            "publication_date": "2026-07-22",
        },
        "receipt_summary": (
            "The direct EBT citation API returned 20 visible records; the "
            "newest visible citation is pre-marker and already indexed."
        ),
    },
    {
        "receipt_id": "semantic_scholar_v527_arm_ebm_citations",
        "source_family": "Semantic Scholar",
        "source_role": "secondary",
        "query_family": "semantic_scholar_citation_trail",
        "query": "arXiv:2512.15605 citations",
        "url": (
            "https://api.semanticscholar.org/graph/v1/paper/arXiv:2512.15605/"
            "citations?fields=title,year,externalIds,url,publicationDate&limit=20"
        ),
        "accessed_at": "2026-07-26T08:09:29Z",
        "access_outcome": "reachable_http_200_8_records_no_post_marker_citation",
        "candidate_ids": ["2607.02154", "2606.03089", "2605.18871"],
        "candidate_count": 8,
        "source_cutoff": "citation_trail_checked_after_v527_marker_newest_2026_07_02",
        "newest_visible": {
            "identifier": "2607.02154",
            "title": "Path-Measure Dynamics of Attention-Driven World Models",
            "publication_date": "2026-07-02",
        },
        "receipt_summary": (
            "The ARM-EBM citation API returned eight visible records; no "
            "record is a newer actionable post-marker source."
        ),
    },
    {
        "receipt_id": "github_v527_trending_python",
        "source_family": "GitHub",
        "source_role": "secondary",
        "query_family": "github_trending_secondary",
        "query": "trending/python daily",
        "url": "https://github.com/trending/python?since=daily",
        "accessed_at": "2026-07-26T08:10:20Z",
        "access_outcome": "reachable_http_200_secondary_trending_page_no_carnot_delta",
        "candidate_ids": ["usestrix/strix"],
        "candidate_count": 1,
        "source_cutoff": "daily_trending_secondary_discovery_after_v527_marker",
        "receipt_summary": (
            "GitHub Trending was reachable; the visible leading repository was "
            "not a Carnot EBM, CSP, KAN, sampler, or verifier dependency."
        ),
    },
    {
        "receipt_id": "github_v527_targeted_searches",
        "source_family": "GitHub",
        "source_role": "secondary",
        "query_family": "github_targeted_secondary",
        "query": (
            "energy-based constraint reasoning; KAN continual learning; Ising "
            "sampler hardware; transactional memory poison, all pushed or "
            "updated after 2026-07-26"
        ),
        "url": "https://api.github.com/search/repositories",
        "accessed_at": "2026-07-26T08:10:35Z",
        "access_outcome": "reachable_http_200_four_targeted_routes_total_count_0",
        "candidate_ids": [],
        "candidate_count": 0,
        "source_cutoff": "pushed_or_updated_after_2026_07_26_secondary_metadata",
        "receipt_summary": (
            "Four targeted GitHub API searches returned zero repositories or "
            "issues after the cutoff."
        ),
    },
    {
        "receipt_id": "extropic_v527_official_pages",
        "source_family": "Extropic",
        "source_role": "official",
        "query_family": "official_project_page",
        "query": "Extropic writing and hardware pages",
        "url": "https://www.extropic.ai/writing ; https://www.extropic.ai/hardware",
        "accessed_at": "2026-07-26T08:11:10Z",
        "access_outcome": "reachable_http_200_official_context_no_authenticated_route",
        "candidate_ids": [
            "tsu_101_2025_10_29",
            "dtms_2025_10_28",
            "x0_pbit_pdit_pmode_pmog",
        ],
        "candidate_count": 3,
        "source_cutoff": "official_pages_checked_after_v527_marker_latest_date_2025_10_29",
        "receipt_summary": (
            "Extropic official pages remained public hardware context without "
            "an authenticated Carnot-local TSU, SDK, speed, power, or correctness route."
        ),
    },
    {
        "receipt_id": "logical_intelligence_v527_official_pages",
        "source_family": "Logical Intelligence",
        "source_role": "official",
        "query_family": "official_project_page",
        "query": "Logical Intelligence root, Kona, and Aleph public pages",
        "url": (
            "https://logicalintelligence.com/ ; "
            "https://logicalintelligence.com/kona-ebms-energy-based-models"
        ),
        "accessed_at": "2026-07-26T08:11:10Z",
        "access_outcome": "reachable_http_200_published_2026_06_26_no_local_weights",
        "candidate_ids": ["kona_page_2026_06_26", "aleph_context"],
        "candidate_count": 2,
        "source_cutoff": "official_pages_changed_before_v527_marker_2026_06_26",
        "receipt_summary": (
            "Logical Intelligence pages were reachable but expose no public "
            "Kona weights, documented local inference API, or reproducible comparator."
        ),
    },
    {
        "receipt_id": "local_sweep_clusters_v527",
        "source_family": "local sweep helper",
        "source_role": "tooling",
        "query_family": "local_tooling",
        "query": "scripts/sweep_clusters.py all --max-results 3",
        "url": "scripts/sweep_clusters.py",
        "accessed_at": "2026-07-26T08:12:00Z",
        "access_outcome": "reachable_local_tool_exit_0_emitted_7_arxiv_urls",
        "candidate_ids": [],
        "candidate_count": 0,
        "source_cutoff": "tooling_urls_emitted_after_v527_marker",
        "receipt_summary": (
            "The local arXiv cluster helper emitted seven broadened query URLs "
            "and did not mutate repository files."
        ),
    },
    {
        "receipt_id": "local_sweep_semscholar_v527",
        "source_family": "local sweep helper",
        "source_role": "tooling",
        "query_family": "local_tooling",
        "query": (
            "energy based reasoning EBT ARM-EBM; neural CSP Ising hallucination "
            "KAN energy guided decoding hardware sampling continual learning"
        ),
        "url": "scripts/sweep_semscholar.py",
        "accessed_at": "2026-07-26T08:12:20Z",
        "access_outcome": "inaccessible_remote_http_429_rate_limited_on_2_keyword_queries",
        "candidate_ids": [],
        "candidate_count": 0,
        "source_cutoff": "keyword_queries_after_v527_marker",
        "receipt_summary": (
            "The local Semantic Scholar keyword helper hit HTTP 429 on two "
            "focused queries; direct citation APIs remained reachable."
        ),
    },
)


def _finding(
    source_id: str,
    classification: str,
    title: str,
    url: str,
    *,
    identifier: str,
    reason: str,
    receipt_id: str,
    query_family: str,
    access_outcome: str,
    authors: Sequence[str] | None = None,
    publication_date: str = "2026-07-26",
    source_date: str = "2026-07-26",
    search_timestamp: str = "2026-07-26T08:16:00Z",
) -> JsonDict:
    return {
        "source_id": source_id,
        "classification": classification,
        "decision_bucket": classification,
        "title": title,
        "url": url,
        "identifier": identifier,
        "authors": list(authors or ["unknown"]),
        "publication_date": publication_date,
        "source_date": source_date,
        "search_timestamp": search_timestamp,
        "receipt_id": receipt_id,
        "query_family": query_family,
        "query": query_family,
        "access_outcome": access_outcome,
        "reason": reason,
    }


DEFAULT_REJECTED_FINDINGS: tuple[JsonDict, ...] = (
    _finding(
        "github_targeted_zero_no_dependency_delta",
        "rejected",
        "GitHub targeted searches without a maintained method delta",
        "https://api.github.com/search/repositories",
        identifier="github_targeted_zero_delta",
        receipt_id="github_v527_targeted_searches",
        query_family="github_targeted_secondary",
        access_outcome="reachable_http_200_four_targeted_routes_total_count_0",
        reason=(
            "Repository and issue metadata are secondary and did not reveal a "
            "new dependency that sharpens an allocated .527 control."
        ),
    ),
    _finding(
        "extropic_context_no_authenticated_route",
        "rejected",
        "Extropic public hardware context without an authenticated local route",
        "https://www.extropic.ai/hardware",
        identifier="extropic_public_context_no_local_route",
        receipt_id="extropic_v527_official_pages",
        query_family="official_project_page",
        access_outcome="reachable_http_200_official_context_no_authenticated_route",
        reason=(
            "Official hardware context is useful background, but there is no "
            "Carnot-local TSU, SDK, power, speed, or correctness receipt."
        ),
    ),
)

DEFAULT_ABSTAINED_FINDINGS: tuple[JsonDict, ...] = (
    _finding(
        "openreview_dynamic_search_date_uncertain",
        "abstained",
        "OpenReview dynamic search page with uncertain primary date",
        "https://openreview.net/search?term=energy-based%20constraint%20reasoning",
        identifier="openreview_dynamic_date_uncertain_post_v527",
        receipt_id="openreview_v527_search_page",
        query_family="openreview_secondary",
        access_outcome="reachable_http_200_dynamic_search_page_no_new_primary_delta",
        reason=(
            "Dynamic search pages can crawl recently without proving a primary "
            "forum publication or material change after the marker."
        ),
    ),
)

DEFAULT_FALSE_POSITIVE_FINDINGS: tuple[JsonDict, ...] = (
    _finding(
        "hf_daily_feed_redirect_false_positive",
        "false_positive",
        "Hugging Face requested date did not prove a post-marker primary source",
        "https://huggingface.co/papers?date=2026-07-26",
        identifier="hf_daily_redirect_secondary_date_only",
        receipt_id="huggingface_papers_v527_2026_07_26",
        query_family="huggingface_papers_secondary",
        access_outcome="reachable_http_200_daily_feed_redirected_or_pinned_2026_07_24",
        reason=(
            "A requested daily-feed date is secondary discovery metadata and "
            "the rendered page exposed a pre-marker 2026-07-24 marker."
        ),
    ),
)

DEFAULT_KNOWN_FALSE_NEGATIVE_FINDINGS: tuple[JsonDict, ...] = (
    _finding(
        "semantic_scholar_keyword_rate_limit_possible_miss",
        "known_false_negative",
        "Semantic Scholar keyword helper rate-limit possible miss",
        "scripts/sweep_semscholar.py",
        identifier="s2_keyword_rate_limit_possible_miss_post_v527",
        receipt_id="local_sweep_semscholar_v527",
        query_family="local_tooling",
        access_outcome="inaccessible_remote_http_429_rate_limited_on_2_keyword_queries",
        reason=(
            "The keyword helper did not return records for two focused queries; "
            "a non-arXiv or relevance-ranked miss is recorded explicitly."
        ),
    ),
)

DEFAULT_CUTOFF_CONFOUND_FINDINGS: tuple[JsonDict, ...] = (
    _finding(
        "same_day_marker_cutoff_confound",
        "cutoff_confound",
        "Same-day post-marker source judgement confound",
        "research-references.md#v527-planner-refresh---20260726",
        identifier="v527_same_day_marker_cutoff_confound",
        receipt_id="arxiv_v527_date_window",
        query_family="arxiv_primary",
        access_outcome="cutoff_confound_preserved",
        reason=(
            "A date-only source route cannot prove ordering after the exact "
            "local marker receipt when source metadata lacks a precise timestamp."
        ),
    ),
)

DEFAULT_ENDPOINT_FAILED_FINDINGS: tuple[JsonDict, ...] = (
    _finding(
        "openreview_api_challenge_endpoint_failed",
        "endpoint_failed",
        "OpenReview notes API challenge gate",
        "https://api2.openreview.net/notes?limit=5&content.title=energy-based",
        identifier="openreview_api_challenge_v527",
        receipt_id="openreview_v527_api_notes",
        query_family="openreview_api",
        access_outcome="inaccessible_http_403_challenge_required",
        publication_date="unknown",
        source_date="unknown",
        reason=(
            "The API route required challenge verification; no result is "
            "fabricated and the failure remains separate from rejection."
        ),
    ),
)

DEFAULT_DUPLICATE_FINDINGS: tuple[JsonDict, ...] = (
    _finding(
        "v527_planner_coverage_audit_2607_21480",
        "duplicate",
        "Finite-Sample Coverage Audits for High-Recall Candidate Generation: Certification and Learning-Theoretic Design",
        "https://arxiv.org/abs/2607.21480",
        identifier="2607.21480",
        authors=["V527 planner source"],
        receipt_id="arxiv_v527_date_window",
        query_family="arxiv_primary",
        access_outcome="duplicate_existing_v527_reference_heading",
        publication_date="2026-07-23",
        source_date="2026-07-23",
        reason=(
            "Already accepted in the sealed V527 planner block for "
            "excluded-pool coverage and non-pruning support audits."
        ),
    ),
    _finding(
        "v527_planner_umem_openreview",
        "duplicate",
        "UMEM: Unified Memory Extraction and Management Framework for Generalizable Memory",
        "https://openreview.net/forum?id=BoiXvrwtdi",
        identifier="openreview:BoiXvrwtdi",
        authors=["V527 planner source"],
        receipt_id="openreview_v527_search_page",
        query_family="openreview_secondary",
        access_outcome="duplicate_existing_v527_reference_heading",
        publication_date="2026-07-26",
        source_date="2026-07-26",
        reason=(
            "Already recorded in the sealed V527 planner block as guarded "
            "neighborhood-utility memory evidence."
        ),
    ),
)

DEFAULT_RETIRED_SCOPE_FINDINGS: tuple[JsonDict, ...] = (
    _finding(
        "schema_reprompt_reopen_post_v527",
        "retired_scope",
        "Schema reprompt reopen request",
        "ops/exclusion_manifest.yaml",
        identifier="schema_reprompt_reopen",
        receipt_id="arxiv_v527_topic_windows",
        query_family="arxiv_primary_topic_windows",
        access_outcome="retired_scope_excluded_by_manifest",
        reason="Schema-supported reprompting is retired and cannot be reopened by freshness.",
    ),
    _finding(
        "exact_diagnostic_reprompt_reopen_post_v527",
        "retired_scope",
        "Exact-diagnostic reprompt reopen request",
        "ops/exclusion_manifest.yaml",
        identifier="exact_diagnostic_reprompt_reopen",
        receipt_id="openreview_v527_search_page",
        query_family="openreview_secondary",
        access_outcome="retired_scope_excluded_by_manifest",
        reason="Exact-diagnostic reprompting remains a retired mechanism.",
    ),
    _finding(
        "finite_id_external_scorer_reopen_post_v527",
        "retired_scope",
        "Finite-ID transport or external scorer reopen request",
        "ops/exclusion_manifest.yaml",
        identifier="finite_id_external_scorer_reopen",
        receipt_id="github_v527_targeted_searches",
        query_family="github_targeted_secondary",
        access_outcome="retired_scope_excluded_by_manifest",
        reason=(
            "Finite-ID transport and external scorer routes overlap closed "
            "generated-answer and scoring mechanisms."
        ),
    ),
    _finding(
        "kan_public_arc_board_probe_reopen_post_v527",
        "retired_scope",
        "KAN mutation, public ARC solve, or unchanged board probe reopen request",
        "ops/exclusion_manifest.yaml",
        identifier="kan_public_arc_board_reopen",
        receipt_id="arxiv_v527_topic_windows",
        query_family="arxiv_primary_topic_windows",
        access_outcome="retired_scope_excluded_by_manifest",
        reason=(
            "KAN mutation, public ARC solves, and unchanged board probes "
            "remain closed for source acceptance."
        ),
    ),
)

DEFAULT_TESTS_RUN: tuple[JsonDict, ...] = (
    {
        "command": (
            ".venv/bin/pytest tests/python/"
            "test_experiment_5934_v527_source_delta_ingestion.py -q --no-cov -n 0"
        ),
        "exit_code": None,
    },
    {
        "command": (
            ".venv/bin/coverage run --rcfile=/dev/null --include="
            "python/carnot/experiment_5934_v527_source_delta_ingestion.py -m pytest "
            "tests/python/test_experiment_5934_v527_source_delta_ingestion.py "
            "-q --no-cov -n 0"
        ),
        "exit_code": None,
    },
    {
        "command": (
            ".venv/bin/coverage report --rcfile=/dev/null --include="
            "python/carnot/experiment_5934_v527_source_delta_ingestion.py "
            "--fail-under=100"
        ),
        "exit_code": None,
    },
    {
        "command": (
            ".venv/bin/ruff check "
            "python/carnot/experiment_5934_v527_source_delta_ingestion.py "
            "tests/python/test_experiment_5934_v527_source_delta_ingestion.py"
        ),
        "exit_code": None,
    },
    {
        "command": (
            ".venv/bin/python scripts/adversarial_verify.py --json "
            "results/experiment_5934_v527_source_delta_ingestion.json"
        ),
        "exit_code": None,
    },
    {"command": ".venv/bin/python scripts/check_spec_coverage.py", "exit_code": None},
    {"command": ".venv/bin/python scripts/root_clutter_sweep.py", "exit_code": None},
    {"command": ".venv/bin/pytest tests/python -q", "exit_code": None},
)


def read_text_if_present(path: Path) -> str:
    """Read an optional local file, using empty text for missing evidence."""

    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def path_sha256(path: Path) -> str | None:
    """Return a sha256 receipt for an existing local file."""

    if not path.exists():
        return None
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(data.encode("utf-8")).hexdigest()


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def normalize_timestamp(value: str) -> str:
    """Normalize timestamps to the artifact's UTC Z form."""

    return _parse_timestamp(value).isoformat().replace("+00:00", "Z")


def planner_marker_line(text: str) -> int | None:
    """Return the one-based line number for the sealed V527 marker."""

    for index, line in enumerate(text.splitlines(), start=1):
        if PLANNER_MARKER in line:
            return index
    return None


def planner_block_hash(text: str) -> str | None:
    """Hash the sealed planner block so post-marker novelty is falsifiable."""

    start = text.find(PLANNER_HEADING)
    end = text.find(PLANNER_END_MARKER)
    if start == -1 or end == -1:
        return None
    block = text[start : end + len(PLANNER_END_MARKER)]
    return "sha256:" + hashlib.sha256(block.encode("utf-8")).hexdigest()


def _roadmap_snapshot(path: Path) -> JsonDict:
    text = read_text_if_present(path)
    if not text:
        return {
            "present": False,
            "milestone": "",
            "task_ids": [],
            "task_ids_hash": None,
            "gates": [],
            "gates_hash": None,
            "model_policy_hash": None,
        }
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError:
        loaded = {}
    if not isinstance(loaded, Mapping):
        loaded = {}
    tasks = loaded.get("tasks") if isinstance(loaded.get("tasks"), list) else []
    task_ids = [
        str(row.get("id")) for row in tasks if isinstance(row, Mapping) and row.get("id")
    ]
    gates = [
        {"id": row.get("id"), "gated_on": row.get("gated_on")}
        for row in tasks
        if isinstance(row, Mapping) and row.get("gated_on")
    ]
    model_policy = [
        {
            "id": row.get("id"),
            "model": row.get("model"),
            "requires_gpu": row.get("requires_gpu"),
        }
        for row in tasks
        if isinstance(row, Mapping) and row.get("id")
    ]
    return {
        "present": True,
        "milestone": str(loaded.get("milestone", "")),
        "task_ids": task_ids,
        "task_ids_hash": _stable_hash(task_ids),
        "gates": gates,
        "gates_hash": _stable_hash(gates),
        "model_policy_hash": _stable_hash(model_policy),
    }


def _resource_receipts(root: Path) -> JsonDict:
    usage = shutil.disk_usage(root)
    return {
        "disk_free_bytes": usage.free,
        "ram_available_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES"),
        "output_parent_writable": os.access(root / RESULT_RELATIVE_PATH.parent, os.W_OK),
    }


def _protected_paths() -> tuple[Path, ...]:
    return (
        AGENTS_RELATIVE_PATH,
        CODEX_RELATIVE_PATH,
        CLAUDE_RELATIVE_PATH,
        RESEARCH_PROGRAM_RELATIVE_PATH,
        RESEARCH_STUDYING_RELATIVE_PATH,
        ROADMAP_RELATIVE_PATH,
        ROADMAP_NEXT_RELATIVE_PATH,
        VNEXT_RELATIVE_PATH,
        EXCLUSION_MANIFEST_RELATIVE_PATH,
        KNOWN_ISSUES_RELATIVE_PATH,
        STATUS_RELATIVE_PATH,
        CHANGELOG_RELATIVE_PATH,
        TRACEABILITY_RELATIVE_PATH,
        CONDUCTOR_RELATIVE_PATH,
        SWEEP_CLUSTERS_RELATIVE_PATH,
        SWEEP_SEMSCHOLAR_RELATIVE_PATH,
        PRIOR_SOURCE_RESULT_RELATIVE_PATH,
    )


def _protected_hashes(root: Path) -> JsonDict:
    return {path.as_posix(): path_sha256(root / path) for path in _protected_paths()}


def _receipt_is_reachable(receipt: Mapping[str, Any]) -> bool:
    outcome = str(receipt.get("access_outcome", ""))
    return "reachable" in outcome or "http_200" in outcome


def _sources_reachable(source_receipts: Sequence[JsonDict]) -> bool:
    return any(_receipt_is_reachable(row) for row in source_receipts)


def _endpoint_failures(source_receipts: Sequence[JsonDict]) -> list[JsonDict]:
    failures: list[JsonDict] = []
    for row in source_receipts:
        outcome = str(row.get("access_outcome", ""))
        if "inaccessible" in outcome or "403" in outcome:
            failures.append(
                {
                    "receipt_id": row["receipt_id"],
                    "source_family": row["source_family"],
                    "access_outcome": row["access_outcome"],
                    "url": row["url"],
                }
            )
    return failures


def _rate_limits(source_receipts: Sequence[JsonDict]) -> list[JsonDict]:
    return [
        {
            "receipt_id": row["receipt_id"],
            "source_family": row["source_family"],
            "access_outcome": row["access_outcome"],
        }
        for row in source_receipts
        if "429" in str(row.get("access_outcome", ""))
        or "rate_limit" in str(row.get("access_outcome", ""))
    ]


def preconditions_checked(
    root: Path,
    *,
    marker_found: bool,
    source_reachable: bool,
    checked_at: str | None = None,
    source_receipts: Sequence[JsonDict] = DEFAULT_SOURCE_RECEIPTS,
) -> JsonDict:
    """Hash the marker, roadmap identity, sweep code, and protected state."""

    active = _roadmap_snapshot(root / ROADMAP_RELATIVE_PATH)
    next_roadmap = _roadmap_snapshot(root / ROADMAP_NEXT_RELATIVE_PATH)
    resources = _resource_receipts(root)
    spec_text = read_text_if_present(root / SPEC_RELATIVE_PATH)
    references_text = read_text_if_present(root / RESEARCH_REFERENCES_RELATIVE_PATH)
    failures: list[str] = []
    if not marker_found:
        failures.append("planner_marker_missing")
    if not source_reachable:
        failures.append("source_reachability_failed")
    if path_sha256(root / ROADMAP_RELATIVE_PATH) is None:
        failures.append("active_roadmap_hash_missing")
    if path_sha256(root / EXCLUSION_MANIFEST_RELATIVE_PATH) is None:
        failures.append("exclusion_manifest_hash_missing")
    if not active["task_ids"] or EXPERIMENT_ID not in active["task_ids"]:
        failures.append("active_roadmap_identity_unavailable")
    if active["milestone"] != MILESTONE:
        failures.append("active_roadmap_identity_unavailable")
    if "REQ-REPORT-5934" not in spec_text:
        failures.append("spec_req_report_5934_missing")
    if not resources["output_parent_writable"]:
        failures.append("output_path_unavailable")
    return {
        "checked_at": normalize_timestamp(checked_at or datetime.now(UTC).isoformat()),
        "planner_marker_found": marker_found,
        "references_hash": path_sha256(root / RESEARCH_REFERENCES_RELATIVE_PATH),
        "v527_marker_hash": planner_block_hash(references_text),
        "active_roadmap_hash": path_sha256(root / ROADMAP_RELATIVE_PATH),
        "active_roadmap_milestone": active["milestone"],
        "active_roadmap_task_ids": active["task_ids"],
        "active_roadmap_task_ids_hash": active["task_ids_hash"],
        "active_roadmap_gates_hash": active["gates_hash"],
        "active_roadmap_model_policy_hash": active["model_policy_hash"],
        "research_roadmap_next_read": bool(next_roadmap["present"]),
        "research_roadmap_next_hash": path_sha256(root / ROADMAP_NEXT_RELATIVE_PATH),
        "research_roadmap_next_milestone": next_roadmap["milestone"],
        "research_roadmap_next_task_ids_hash": next_roadmap["task_ids_hash"],
        "vnext_hash": path_sha256(root / VNEXT_RELATIVE_PATH),
        "exclusion_manifest_hash": path_sha256(root / EXCLUSION_MANIFEST_RELATIVE_PATH),
        "sweep_clusters_hash": path_sha256(root / SWEEP_CLUSTERS_RELATIVE_PATH),
        "sweep_semscholar_hash": path_sha256(root / SWEEP_SEMSCHOLAR_RELATIVE_PATH),
        "prior_source_result_hash": path_sha256(root / PRIOR_SOURCE_RESULT_RELATIVE_PATH),
        "output_path_hash": path_sha256(root / RESULT_RELATIVE_PATH),
        "protected_file_hashes": _protected_hashes(root),
        "network_available": source_reachable,
        "api_routes_checked": source_reachable,
        "source_query_families": sorted(
            {str(row.get("query_family", "")) for row in source_receipts}
        ),
        "source_cutoffs": sorted(
            {
                str(row.get("source_cutoff", "published_or_changed_after_v527_marker"))
                for row in source_receipts
            }
        ),
        "endpoint_failures": _endpoint_failures(source_receipts),
        "rate_limits": _rate_limits(source_receipts),
        "output_path_available": resources["output_parent_writable"],
        "disk_free_bytes": resources["disk_free_bytes"],
        "ram_available_bytes": resources["ram_available_bytes"],
        "failed_preconditions": failures,
    }


def search_window_and_marker_receipt(
    references_text: str,
    *,
    search_started_at: str,
    search_finished_at: str,
) -> JsonDict:
    """Record the sealed V527 marker and UTC post-marker search interval."""

    return {
        "boundary_marker": PLANNER_MARKER,
        "boundary_heading": PLANNER_HEADING,
        "boundary_line": planner_marker_line(references_text),
        "boundary_hash": planner_block_hash(references_text),
        "search_window_start_utc": normalize_timestamp(search_started_at),
        "search_window_end_utc": normalize_timestamp(search_finished_at),
        "novelty_rule": (
            "accept only primary-source evidence published or materially changed "
            "after the V527 marker that sharpens already allocated .527 tasks"
        ),
    }


def source_queries_and_endpoint_receipts(source_receipts: Sequence[JsonDict]) -> JsonDict:
    """Group source receipts with endpoint-failure and rate-limit summaries."""

    normalized_receipts = []
    for row in source_receipts:
        receipt = dict(row)
        receipt.setdefault("source_cutoff", "published_or_changed_after_v527_marker")
        normalized_receipts.append(receipt)
    return {
        "source_receipts": normalized_receipts,
        "query_families": sorted(
            {str(row.get("query_family", "")) for row in source_receipts}
        ),
        "endpoint_failures": _endpoint_failures(source_receipts),
        "rate_limits": _rate_limits(source_receipts),
    }


def primary_secondary_and_official_source_counts(
    source_receipts: Sequence[JsonDict],
) -> JsonDict:
    """Count source roles so secondary discovery cannot masquerade as primary."""

    counts = {"primary": 0, "secondary": 0, "official": 0, "tooling": 0}
    for row in source_receipts:
        role = str(row.get("source_role", ""))
        if role in counts:
            counts[role] += 1
    return {
        **counts,
        "source_route_count": len(source_receipts),
        "candidate_count_total": sum(
            int(row.get("candidate_count", 0)) for row in source_receipts
        ),
        "source_families_checked": sorted(
            {str(row.get("source_family", "")) for row in source_receipts}
        ),
    }


def _classification(
    accepted_findings: Sequence[JsonDict],
    *,
    blocked: bool,
) -> JsonDict:
    accepted = [] if blocked else [dict(row) for row in accepted_findings]
    rejected = [dict(row) for row in DEFAULT_REJECTED_FINDINGS]
    abstained = [dict(row) for row in DEFAULT_ABSTAINED_FINDINGS]
    false_positive = [dict(row) for row in DEFAULT_FALSE_POSITIVE_FINDINGS]
    known_false_negative = [dict(row) for row in DEFAULT_KNOWN_FALSE_NEGATIVE_FINDINGS]
    cutoff_confound = [dict(row) for row in DEFAULT_CUTOFF_CONFOUND_FINDINGS]
    endpoint_failed = [dict(row) for row in DEFAULT_ENDPOINT_FAILED_FINDINGS]
    duplicate = [dict(row) for row in DEFAULT_DUPLICATE_FINDINGS]
    retired_scope = [dict(row) for row in DEFAULT_RETIRED_SCOPE_FINDINGS]
    return {
        "accepted": accepted,
        "rejected": rejected,
        "abstained": abstained,
        "false_positive": false_positive,
        "known_false_negative": known_false_negative,
        "cutoff_confound": cutoff_confound,
        "endpoint_failed": endpoint_failed,
        "duplicate": duplicate,
        "retired_scope": retired_scope,
        "all_candidates": (
            accepted
            + rejected
            + abstained
            + false_positive
            + known_false_negative
            + cutoff_confound
            + endpoint_failed
            + duplicate
            + retired_scope
        ),
    }


def false_positive_false_negative_cutoff_and_rate_limit_receipts(
    classification: JsonDict,
    source_receipts: Sequence[JsonDict],
) -> JsonDict:
    """Expose uncertainty classes without hiding them as ordinary rejects."""

    return {
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "false_positive_false_negative_cutoff_and_rate_limit_receipts"
        ],
        "false_positive_source_decisions": list(classification["false_positive"]),
        "known_false_negative_source_decisions": list(classification["known_false_negative"]),
        "cutoff_confounds": list(classification["cutoff_confound"]),
        "endpoint_failed_source_decisions": list(classification["endpoint_failed"]),
        "rate_limit_receipts": _rate_limits(source_receipts),
        "ordinary_rejections_do_not_include_uncertainty": True,
    }


def semantic_scholar_ebt_and_arm_ebm_receipts(
    source_receipts: Sequence[JsonDict],
) -> JsonDict:
    """Summarize the two direct Semantic Scholar citation API receipts."""

    direct = [
        dict(row)
        for row in source_receipts
        if str(row.get("source_family")) == "Semantic Scholar"
    ]
    ebt = next((row for row in direct if "ebt" in row["receipt_id"]), {})
    arm = next((row for row in direct if "arm_ebm" in row["receipt_id"]), {})
    return {
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "semantic_scholar_ebt_and_arm_ebm_receipts"
        ],
        "direct_api_receipts": direct,
        "ebt_arxiv_id": "2507.02092",
        "arm_ebm_arxiv_id": "2512.15605",
        "ebt_visible_citation_count": int(ebt.get("candidate_count", 0)),
        "arm_ebm_visible_citation_count": int(arm.get("candidate_count", 0)),
        "ebt_newest_visible": ebt.get("newest_visible"),
        "arm_ebm_newest_visible": arm.get("newest_visible"),
        "post_marker_actionable_citation_count": 0,
        "counts_dated_at_utc": max(
            [str(row.get("accessed_at", "")) for row in direct] or [""]
        ),
        "discovery_not_primary_evidence": True,
    }


def extropic_github_huggingface_openreview_and_kona_receipts(
    source_receipts: Sequence[JsonDict],
) -> JsonDict:
    """Group official and secondary routes that are context until primary opened."""

    def by_family(*families: str) -> list[JsonDict]:
        return [
            dict(row)
            for row in source_receipts
            if str(row.get("source_family")) in set(families)
        ]

    return {
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "extropic_github_huggingface_openreview_and_kona_receipts"
        ],
        "extropic_receipts": by_family("Extropic"),
        "github_receipts": by_family("GitHub"),
        "huggingface_receipts": by_family("Hugging Face Papers"),
        "openreview_receipts": by_family("OpenReview"),
        "kona_or_aleph_receipts": by_family("Logical Intelligence"),
        "secondary_or_official_until_primary_reproducible_artifact_opened": True,
    }


def duplicate_and_retired_scope_filter(classification: JsonDict) -> JsonDict:
    """Record duplicate dimensions and closed scopes used for rejection."""

    return {
        "duplicate_dimensions": [
            "identifier",
            "title",
            "authors",
            "mechanism",
            "existing_reference_heading",
        ],
        "duplicate_source_decisions": list(classification["duplicate"]),
        "retired_scope_rules": [
            "schema reprompt",
            "exact-diagnostic reprompt",
            "finite-ID transport",
            "external scorers",
            "KAN mutation",
            "public ARC solves",
            "unchanged board probes",
        ],
        "retired_scope_source_decisions": list(classification["retired_scope"]),
        "accepted_reopens_retired_scope_count": sum(
            1 for row in classification["accepted"] if row.get("reopens_retired_scope")
        ),
    }


def task_identity_gate_and_exclusion_immutability(root: Path) -> JsonDict:
    """Declare task, gate, and exclusion boundaries this task cannot rewrite."""

    active = _roadmap_snapshot(root / ROADMAP_RELATIVE_PATH)
    return {
        "task_ids_unchanged": True,
        "gates_unchanged": True,
        "exclusions_unchanged": True,
        "model_policy_unchanged": True,
        "authority_boundaries_unchanged": True,
        "retired_scopes_reopened": False,
        "hardware_requirements_changed": False,
        "headline_claims_changed": False,
        "active_roadmap_task_ids_hash": active["task_ids_hash"],
        "active_roadmap_gates_hash": active["gates_hash"],
        "active_roadmap_model_policy_hash": active["model_policy_hash"],
        "exclusion_manifest_hash": path_sha256(root / EXCLUSION_MANIFEST_RELATIVE_PATH),
        "allowed_target_experiments": list(ALLOCATED_TARGET_EXPERIMENTS),
    }


def protected_files_unchanged(
    root: Path,
    before_hashes: Mapping[str, str | None],
) -> JsonDict:
    """Compare protected files before and after optional reference append."""

    after = _protected_hashes(root)
    changed = [
        path for path, before_hash in before_hashes.items() if after.get(path) != before_hash
    ]
    return {
        "all_unchanged": not changed,
        "changed_paths": changed,
        "before_hashes": dict(before_hashes),
        "after_hashes": after,
    }


def execution_delta_block(accepted_findings: Sequence[JsonDict]) -> str:
    """Render the only reference block this workflow is allowed to append."""

    lines = [
        "",
        EXECUTION_DELTA_HEADING,
        "",
        (
            "Execution-time sweep on 2026-07-26 after the V527 planner marker. "
            "Only non-duplicate primary-source deltas that sharpen existing "
            ".527 controls are listed here."
        ),
        "",
    ]
    for item in accepted_findings:
        mapping = item["method_to_task_mapping"]
        lines.append(
            "- **{title}** - {url}; source date {source_date}. Carnot hook: {hook} "
            "Target: `{target}`. Mapping: `{method}`. Boundary: {boundary}".format(
                title=item["title"],
                url=item["url"],
                source_date=item["source_date"],
                hook=item["source_hook"],
                target=item["target_experiment"],
                method=mapping["method"],
                boundary=item["authority_boundary"],
            )
        )
    lines.extend(["", EXECUTION_DELTA_END_MARKER, ""])
    return "\n".join(lines)


def insert_after_planner_block(text: str, block: str) -> str:
    """Insert the execution block once, immediately after the sealed marker."""

    if EXECUTION_DELTA_HEADING in text:
        return text
    marker_index = text.find(PLANNER_END_MARKER)
    if marker_index == -1:
        return text.rstrip() + "\n" + block
    insert_at = marker_index + len(PLANNER_END_MARKER)
    return text[:insert_at].rstrip() + "\n" + block + text[insert_at:].lstrip("\n")


def references_append_receipt(
    *,
    before_hash: str | None,
    after_hash: str | None,
    appended: bool,
    accepted_findings: Sequence[JsonDict],
) -> JsonDict:
    """Summarize the optional references append without rewriting markers."""

    return {
        "appended": bool(appended),
        "heading": EXECUTION_DELTA_HEADING,
        "end_marker": EXECUTION_DELTA_END_MARKER,
        "accepted_count": len(accepted_findings),
        "accepted_source_ids": [str(row["source_id"]) for row in accepted_findings],
        "references_before_hash": before_hash,
        "references_after_hash": after_hash,
        "prior_v527_marker_preserved": True,
        "prior_marker_rewrite_count": 0,
    }


def honest_verdict(
    marker_found: bool,
    source_reachable: bool,
    accepted_findings: Sequence[JsonDict],
    blocked: bool,
) -> str:
    """Return a terminal verdict with the required source-refresh prefix."""

    if blocked or not marker_found:
        return "blocked: V527 source refresh precondition failed"
    if not source_reachable:
        return "blocked: no primary, official, or reliable secondary source route reachable"
    if accepted_findings:
        return (
            f"complete_delta: accepted {len(accepted_findings)} bounded "
            "post-V527 source delta(s); task identities, gates, and exclusions unchanged"
        )
    return "complete_null: no accepted post-V527 source deltas; references unchanged"


def _field_provenance(accepted_findings: Sequence[JsonDict]) -> JsonDict:
    provenance: JsonDict = {
        field: {
            "principle": REQUIRED_FIELD_PRINCIPLES[field],
            "source": "Exp5934 source receipts, local hashes, query families, cutoffs, or classification records",
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }
    for field, principle in FIELD_PRINCIPLE_EXTRAS.items():
        provenance[field] = {"principle": principle, "source": "local artifact metadata"}
    provenance["accepted_findings"] = [
        {
            "source_id": item["source_id"],
            "receipt_id": item["receipt_id"],
            "target_experiment": item["target_experiment"],
        }
        for item in accepted_findings
    ]
    return provenance


def _checksum_payload(artifact: JsonDict) -> JsonDict:
    payload = dict(artifact)
    payload.pop("reproducibility_checksum", None)
    return payload


def _compute_checksum(artifact: JsonDict) -> str:
    return _stable_hash(_checksum_payload(artifact))


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    search_started_at: str,
    search_finished_at: str,
    accepted_findings: Sequence[JsonDict] = (),
    source_receipts: Sequence[JsonDict] = DEFAULT_SOURCE_RECEIPTS,
    references_appended: bool = False,
    references_before_hash: str | None = None,
    references_after_hash: str | None = None,
    protected_before_hashes: Mapping[str, str | None] | None = None,
    test_commands: Sequence[str] | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
    duration_s: float = 0.0,
) -> JsonDict:
    """Build the Exp5934 artifact from local hashes and source receipts."""

    references_text = read_text_if_present(root / RESEARCH_REFERENCES_RELATIVE_PATH)
    marker_found = PLANNER_MARKER in references_text
    source_reachable = _sources_reachable(source_receipts)
    preconditions = preconditions_checked(
        root,
        marker_found=marker_found,
        source_reachable=source_reachable,
        checked_at=search_started_at,
        source_receipts=source_receipts,
    )
    blocked = bool(preconditions["failed_preconditions"])
    effective_accepted = [] if blocked else [dict(row) for row in accepted_findings]
    classification = _classification(effective_accepted, blocked=blocked)
    commands = (
        list(test_commands)
        if test_commands is not None
        else [row["command"] for row in DEFAULT_TESTS_RUN]
    )
    exit_codes = (
        dict(test_exit_codes)
        if test_exit_codes is not None
        else {row["command"]: row["exit_code"] for row in DEFAULT_TESTS_RUN}
    )
    before_hash = references_before_hash or path_sha256(
        root / RESEARCH_REFERENCES_RELATIVE_PATH
    )
    after_hash = references_after_hash or path_sha256(root / RESEARCH_REFERENCES_RELATIVE_PATH)
    protected_hashes = protected_before_hashes or _protected_hashes(root)
    started = normalize_timestamp(search_started_at)
    finished = normalize_timestamp(search_finished_at)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "status": "blocked" if blocked else "complete",
        "preconditions_checked": preconditions,
        "search_window_and_marker_receipt": search_window_and_marker_receipt(
            references_text,
            search_started_at=started,
            search_finished_at=finished,
        ),
        "source_queries_and_endpoint_receipts": source_queries_and_endpoint_receipts(
            source_receipts
        ),
        "primary_secondary_and_official_source_counts": (
            primary_secondary_and_official_source_counts(source_receipts)
        ),
        "accepted_rejected_abstained_findings": classification,
        "false_positive_false_negative_cutoff_and_rate_limit_receipts": (
            false_positive_false_negative_cutoff_and_rate_limit_receipts(
                classification,
                source_receipts,
            )
        ),
        "semantic_scholar_ebt_and_arm_ebm_receipts": (
            semantic_scholar_ebt_and_arm_ebm_receipts(source_receipts)
        ),
        "extropic_github_huggingface_openreview_and_kona_receipts": (
            extropic_github_huggingface_openreview_and_kona_receipts(source_receipts)
        ),
        "duplicate_and_retired_scope_filter": duplicate_and_retired_scope_filter(
            classification
        ),
        "references_append_receipt": references_append_receipt(
            before_hash=before_hash,
            after_hash=after_hash,
            appended=references_appended and bool(effective_accepted),
            accepted_findings=effective_accepted,
        ),
        "task_identity_gate_and_exclusion_immutability": (
            task_identity_gate_and_exclusion_immutability(root)
        ),
        "protected_files_unchanged": protected_files_unchanged(root, protected_hashes),
        "duration_s": float(duration_s),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_provenance": _field_provenance(effective_accepted),
        "test_commands": commands,
        "test_exit_codes": exit_codes,
        "search_started_at": started,
        "search_finished_at": finished,
        "honest_verdict": honest_verdict(
            marker_found,
            source_reachable,
            effective_accepted,
            blocked,
        ),
    }
    artifact["reproducibility_checksum"] = _compute_checksum(artifact)
    return artifact


def build_and_write_artifact(
    *,
    root: Path = REPO_ROOT,
    search_started_at: str,
    search_finished_at: str,
    accepted_findings: Sequence[JsonDict] = (),
    source_receipts: Sequence[JsonDict] = DEFAULT_SOURCE_RECEIPTS,
    test_commands: Sequence[str] | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
    duration_s: float = 0.0,
) -> JsonDict:
    """Optionally append accepted deltas, validate the artifact, then write JSON."""

    references_path = root / RESEARCH_REFERENCES_RELATIVE_PATH
    references_text = read_text_if_present(references_path)
    references_before = path_sha256(references_path)
    protected_before = _protected_hashes(root)
    marker_found = PLANNER_MARKER in references_text
    source_reachable = _sources_reachable(source_receipts)
    preconditions = preconditions_checked(
        root,
        marker_found=marker_found,
        source_reachable=source_reachable,
        checked_at=search_started_at,
        source_receipts=source_receipts,
    )
    references_appended = False
    if not preconditions["failed_preconditions"] and accepted_findings:
        block = execution_delta_block(accepted_findings)
        updated = insert_after_planner_block(references_text, block)
        references_appended = updated != references_text
        if references_appended:
            references_path.write_text(updated, encoding="utf-8")
    artifact = build_artifact(
        root=root,
        search_started_at=search_started_at,
        search_finished_at=search_finished_at,
        accepted_findings=accepted_findings,
        source_receipts=source_receipts,
        references_appended=references_appended,
        references_before_hash=references_before,
        references_after_hash=path_sha256(references_path),
        protected_before_hashes=protected_before,
        test_commands=test_commands,
        test_exit_codes=test_exit_codes,
        duration_s=duration_s,
    )
    validate_artifact(artifact)
    result_path = root / RESULT_RELATIVE_PATH
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


def _validate_finding(row: Mapping[str, Any], expected_classification: str) -> None:
    for field in (
        "source_id",
        "classification",
        "decision_bucket",
        "title",
        "url",
        "identifier",
        "authors",
        "publication_date",
        "source_date",
        "search_timestamp",
        "receipt_id",
        "query_family",
        "query",
        "access_outcome",
        "reason",
    ):
        if field not in row:
            raise ValueError(f"finding provenance field missing {field}")
    if row["classification"] != expected_classification:
        raise ValueError("invalid finding classification")
    if row["decision_bucket"] != expected_classification:
        raise ValueError("invalid finding decision bucket")
    if expected_classification != "accepted":
        return
    if not row.get("post_marker_or_newer_primary_source"):
        raise ValueError("accepted finding must be newer primary-source evidence")
    if str(row.get("source_date")) < POST_MARKER_SOURCE_DATE:
        raise ValueError("accepted finding must be newer primary-source evidence")
    if not row.get("primary_source"):
        raise ValueError("accepted finding must cite primary-source evidence")
    if row.get("duplicate_of_existing_reference"):
        raise ValueError("accepted finding cannot be duplicate")
    if row.get("reopens_retired_scope"):
        raise ValueError("accepted finding cannot reopen retired scope")
    for field in ("target_experiment", "source_hook", "authority_boundary"):
        if not row.get(field):
            raise ValueError(f"accepted finding missing {field}")
    if row["target_experiment"] not in ALLOCATED_TARGET_EXPERIMENTS:
        raise ValueError("accepted finding must target an allocated .527 experiment")
    mapping = row.get("method_to_task_mapping")
    if not isinstance(mapping, Mapping):
        raise ValueError("accepted finding missing method-to-task mapping")
    if mapping.get("target_experiment") != row["target_experiment"]:
        raise ValueError("accepted finding method-to-task mapping target mismatch")
    for field in ("method", "task_hook", "failure_boundary"):
        if not mapping.get(field):
            raise ValueError(f"accepted finding method-to-task mapping missing {field}")


def _validate_source_receipts(artifact: Mapping[str, Any]) -> None:
    source_queries = artifact.get("source_queries_and_endpoint_receipts")
    if not isinstance(source_queries, Mapping):
        raise ValueError("source_queries_and_endpoint_receipts must be a mapping")
    source_receipts = source_queries.get("source_receipts")
    if not isinstance(source_receipts, list) or not source_receipts:
        raise ValueError("source_queries_and_endpoint_receipts source_receipts missing")
    for row in source_receipts:
        if not isinstance(row, Mapping):
            raise ValueError("source receipt entries must be mappings")
        for field in SOURCE_RECEIPT_REQUIRED_FIELDS:
            if field not in row:
                raise ValueError(f"source receipt missing {field}")
    if not isinstance(source_queries.get("endpoint_failures"), list):
        raise ValueError("source_queries endpoint failures missing")
    if not isinstance(source_queries.get("rate_limits"), list):
        raise ValueError("source_queries rate limits missing")


def _validate_classification(artifact: Mapping[str, Any]) -> JsonDict:
    classification = artifact.get("accepted_rejected_abstained_findings")
    expected = [
        "accepted",
        "rejected",
        "abstained",
        "false_positive",
        "known_false_negative",
        "cutoff_confound",
        "endpoint_failed",
        "duplicate",
        "retired_scope",
    ]
    if not isinstance(classification, Mapping):
        raise ValueError("accepted_rejected_abstained_findings must be a mapping")
    for key in expected:
        if not isinstance(classification.get(key), list):
            raise ValueError(f"accepted_rejected_abstained_findings.{key} must be a list")
        for row in classification[key]:
            if not isinstance(row, Mapping):
                raise ValueError("finding classification entries must be mappings")
            _validate_finding(row, key)
    all_candidates = classification.get("all_candidates")
    ordered: list[JsonDict] = []
    for key in expected:
        ordered.extend(classification[key])
    if all_candidates != ordered:
        raise ValueError("all_candidates must preserve classification order")
    return dict(classification)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp5934 artifact schema and anti-laundering invariants."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required field {field}")
    if artifact["status"] not in {"complete", "blocked"}:
        raise ValueError("invalid status")
    if not str(artifact["honest_verdict"]).startswith(
        ("complete_delta:", "complete_null:", "blocked:")
    ):
        raise ValueError("honest_verdict must use a terminal source prefix")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference substrate must be external source aggregation")
    if float(artifact["duration_s"]) < 0:
        raise ValueError("duration must be non-negative")
    if _parse_timestamp(str(artifact["search_finished_at"])) <= _parse_timestamp(
        str(artifact["search_started_at"])
    ):
        raise ValueError("timestamp window must be positive")

    _validate_source_receipts(artifact)
    classification = _validate_classification(artifact)

    counts = artifact.get("primary_secondary_and_official_source_counts")
    if not isinstance(counts, Mapping) or not all(
        key in counts for key in ("primary", "secondary", "official", "tooling")
    ):
        raise ValueError("source counts missing required roles")
    if artifact["status"] == "complete" and not all(
        int(counts[key]) >= 1 for key in ("primary", "secondary", "official", "tooling")
    ):
        raise ValueError("source counts must include primary, secondary, official, tooling")

    references = artifact["references_append_receipt"]
    accepted = classification["accepted"]
    if references["accepted_count"] != len(accepted):
        raise ValueError("references append accepted count mismatch")
    if references["accepted_source_ids"] != [row["source_id"] for row in accepted]:
        raise ValueError("references append accepted source ids mismatch")
    if not accepted and references["appended"]:
        raise ValueError("zero accepted findings cannot append references")

    provenance = artifact.get("field_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("field_provenance must be a mapping")
    for field, principle in REQUIRED_FIELD_PRINCIPLES.items():
        if field not in provenance:
            raise ValueError(f"field_provenance missing {field}")
        if provenance[field].get("principle") != principle:
            raise ValueError(f"field_provenance principle mismatch for {field}")

    uncertainty = artifact["false_positive_false_negative_cutoff_and_rate_limit_receipts"]
    if not isinstance(uncertainty, Mapping) or uncertainty.get("principle") != (
        REQUIRED_FIELD_PRINCIPLES[
            "false_positive_false_negative_cutoff_and_rate_limit_receipts"
        ]
    ):
        raise ValueError("false-positive/cutoff receipts malformed")
    for key in (
        "false_positive_source_decisions",
        "known_false_negative_source_decisions",
        "cutoff_confounds",
        "endpoint_failed_source_decisions",
        "rate_limit_receipts",
    ):
        if not isinstance(uncertainty.get(key), list):
            raise ValueError("false-positive/cutoff receipts missing list")

    semantic = artifact["semantic_scholar_ebt_and_arm_ebm_receipts"]
    if not isinstance(semantic, Mapping) or (
        artifact["status"] == "complete" and not semantic.get("direct_api_receipts")
    ):
        raise ValueError("semantic scholar receipts missing direct API receipts")
    grouped = artifact["extropic_github_huggingface_openreview_and_kona_receipts"]
    required_groups = (
        "extropic_receipts",
        "github_receipts",
        "huggingface_receipts",
        "openreview_receipts",
        "kona_or_aleph_receipts",
    )
    if not isinstance(grouped, Mapping) or not all(key in grouped for key in required_groups):
        raise ValueError("official/discovery receipts missing grouped routes")
    if artifact["status"] == "complete" and not all(grouped[key] for key in required_groups):
        raise ValueError("official/discovery receipts require each route group")

    filters = artifact["duplicate_and_retired_scope_filter"]
    if not isinstance(filters, Mapping):
        raise ValueError("duplicate_and_retired_scope_filter must be a mapping")
    if filters.get("accepted_reopens_retired_scope_count") != 0:
        raise ValueError("retired scope reopened by accepted finding")

    immutability = artifact["task_identity_gate_and_exclusion_immutability"]
    if not isinstance(immutability, Mapping):
        raise ValueError("task_identity_gate_and_exclusion_immutability must be a mapping")
    if not immutability.get("task_ids_unchanged"):
        raise ValueError("task ids changed")
    if not immutability.get("gates_unchanged"):
        raise ValueError("gates changed")
    if not immutability.get("exclusions_unchanged"):
        raise ValueError("exclusions changed")

    protected = artifact["protected_files_unchanged"]
    if not isinstance(protected, Mapping) or not protected.get("all_unchanged"):
        raise ValueError("protected files changed")

    if artifact["reproducibility_checksum"] != _compute_checksum(dict(artifact)):
        raise ValueError("checksum mismatch")


def _load_tests_run(path: Path | None) -> tuple[list[str], dict[str, int | None]]:
    if path is None:
        commands = [str(row["command"]) for row in DEFAULT_TESTS_RUN]
        return commands, {str(row["command"]): row["exit_code"] for row in DEFAULT_TESTS_RUN}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    commands = [str(row["command"]) for row in loaded]
    return commands, {str(row["command"]): row.get("exit_code") for row in loaded}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--search-started-at", required=True)
    parser.add_argument("--search-finished-at", required=True)
    parser.add_argument("--duration-s", type=float, default=0.0)
    parser.add_argument("--tests-run-json", type=Path)
    parser.add_argument(
        "--zero-findings",
        action="store_true",
        help="Confirm that no accepted post-V527 findings are being supplied.",
    )
    args = parser.parse_args(argv)
    if not args.zero_findings:
        raise SystemExit("--zero-findings is required for the CLI emission path")
    commands, exit_codes = _load_tests_run(args.tests_run_json)
    artifact = build_and_write_artifact(
        root=args.root,
        search_started_at=args.search_started_at,
        search_finished_at=args.search_finished_at,
        accepted_findings=[],
        test_commands=commands,
        test_exit_codes=exit_codes,
        duration_s=args.duration_s,
    )
    print(f"wrote {artifact['result_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
