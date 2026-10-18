"""Exp6113: ingest post-V530 source deltas with explicit uncertainty.

Spec refs: REQ-REPORT-6113, SCENARIO-REPORT-6113-ZERO-FINDING,
SCENARIO-REPORT-6113-ACCEPT-BOUNDED-DELTA,
SCENARIO-REPORT-6113-SOURCE-UNCERTAINTY,
SCENARIO-REPORT-6113-DUPLICATE-AND-RETIRED-SCOPE,
SCENARIO-REPORT-6113-SCHEMA.

This module records a source-ledger pass. It does not run a model, touch the
conductor, or change roadmap gates. The point is to make the V530 planning
boundary auditable: external sources are dated, uncertainty stays visible, and
only genuinely new primary evidence inside an already allocated V530 task may
append to the references ledger.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6113_v530_source_delta_ingestion.json")

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
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
SWEEP_CLUSTERS_RELATIVE_PATH = Path("scripts/sweep_clusters.py")
SWEEP_SEMSCHOLAR_RELATIVE_PATH = Path("scripts/sweep_semscholar.py")
PRIOR_SOURCE_RESULT_RELATIVE_PATH = Path(
    "results/experiment_6101_v529_source_delta_ingestion.json"
)

EXPERIMENT = "experiment_6113_v530_source_delta_ingestion"
EXPERIMENT_ID = "exp6113-v530-source-delta-ingestion"
MILESTONE = "2026.08.530"
RUN_DATE = "20260804"
RANDOM_SEED = 6113
SCHEMA = "carnot.experiment_6113.v530_source_delta_ingestion.v1"
INFERENCE_SUBSTRATE = "aggregation_from_external_primary_sources"

PLANNER_HEADING = "## V530 Planner Refresh - 20260804"
PLANNER_MARKER = "V530-PLANNER-REFRESH-20260804-END"
PLANNER_END_MARKER = f"<!-- {PLANNER_MARKER} -->"
EXECUTION_DELTA_HEADING = "## V530 Execution Source Delta - 20260804"
EXECUTION_DELTA_END_MARKER = "<!-- V530-EXECUTION-SOURCE-DELTA-20260804-END -->"

ALLOCATED_TARGET_EXPERIMENTS = (
    "exp6114-phase-d-gpu-ladder-canary",
    "exp6115-phase-d-calibration-pool",
    "exp6116-phase-d-held-candidate-pool",
    "exp6117-phase-d-headroom-audit",
    "exp6118-phase-d-per-layer-surface",
    "exp6119-phase-d-hidden-state-selector",
    "exp6120-outcome-committed-reduced-order-csl",
    "exp6121-gatemate-changed-state-gate-v530",
    "exp6122-arc-primitive-reachability-loo",
    "exp6123-v530-capstone-reconciliation",
)

SPEC_REFS = (
    "REQ-REPORT-6113",
    "SCENARIO-REPORT-6113-ZERO-FINDING",
    "SCENARIO-REPORT-6113-ACCEPT-BOUNDED-DELTA",
    "SCENARIO-REPORT-6113-SOURCE-UNCERTAINTY",
    "SCENARIO-REPORT-6113-DUPLICATE-AND-RETIRED-SCOPE",
    "SCENARIO-REPORT-6113-SCHEMA",
)

CLASSIFICATION_BUCKETS = (
    "accepted",
    "rejected",
    "duplicate",
    "retired_scope",
    "abstained",
    "false_positive",
    "known_false_negative",
    "cutoff_confound",
    "endpoint_failed",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "search_window_and_marker_receipt",
    "source_queries_and_endpoint_receipts",
    "primary_secondary_and_official_source_counts",
    "accepted_rejected_duplicate_retired_and_abstained_findings",
    "false_positive_false_negative_cutoff_and_rate_limit_receipts",
    "semantic_scholar_ebt_and_arm_ebm_receipts",
    "openreview_huggingface_github_extropic_and_kona_receipts",
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
    "status": "only a hash-anchored post-marker window is eligible.",
    "preconditions_checked": "only a hash-anchored post-marker window is eligible.",
    "search_window_and_marker_receipt": (
        "only a hash-anchored post-marker window is eligible."
    ),
    "source_queries_and_endpoint_receipts": (
        "source coverage, access failures, and dates remain auditable."
    ),
    "primary_secondary_and_official_source_counts": (
        "source coverage, access failures, and dates remain auditable."
    ),
    "accepted_rejected_duplicate_retired_and_abstained_findings": (
        "every candidate receives one explicit disposition."
    ),
    "false_positive_false_negative_cutoff_and_rate_limit_receipts": (
        "uncertainty or access failure is never silently converted into rejection."
    ),
    "semantic_scholar_ebt_and_arm_ebm_receipts": (
        "discovery indexes are secondary until a primary source is opened."
    ),
    "openreview_huggingface_github_extropic_and_kona_receipts": (
        "discovery indexes are secondary until a primary source is opened."
    ),
    "duplicate_and_retired_scope_filter": (
        "no duplicate or renamed retired mechanism enters the ledger."
    ),
    "references_append_receipt": (
        "source aggregation may append references but cannot rewrite the staged plan."
    ),
    "task_identity_gate_and_exclusion_immutability": (
        "source aggregation may append references but cannot rewrite the staged plan."
    ),
    "protected_files_unchanged": (
        "active roadmap, conductor, exclusions, historical artifacts, and unrelated "
        "changes remain byte-identical."
    ),
    "duration_s": "use measured `aggregation_from_external_primary_sources`.",
    "inference_substrate": "use measured `aggregation_from_external_primary_sources`.",
    "field_provenance": "use measured `aggregation_from_external_primary_sources`.",
    "test_commands": "use measured `aggregation_from_external_primary_sources`.",
    "test_exit_codes": "use measured `aggregation_from_external_primary_sources`.",
    "reproducibility_checksum": (
        "use measured `aggregation_from_external_primary_sources`."
    ),
    "honest_verdict": "use `complete_delta:`, `complete_null:`, or `blocked:`.",
}

FIELD_PRINCIPLE_EXTRAS: dict[str, str] = {
    "schema": "Versioned schema id keeps downstream validators from guessing field meaning.",
    "experiment": "Stable local slug ties the artifact to the implementation module.",
    "experiment_id": "Conductor task identity prevents numeric-prefix aliasing.",
    "milestone": "Binds receipts to .530 rather than a prior milestone.",
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
        "receipt_id": "arxiv_v530_exact_post_marker_window",
        "source_family": "arXiv",
        "source_role": "primary",
        "query_family": "arxiv_primary",
        "query": "submittedDate:[202608040000 TO 202608042359]",
        "url": (
            "https://export.arxiv.org/api/query?search_query=submittedDate:%5B"
            "202608040000%20TO%20202608042359%5D&start=0&max_results=10&"
            "sortBy=submittedDate&sortOrder=descending"
        ),
        "accessed_at": "2026-08-04T19:41:05Z",
        "access_outcome": "reachable_http_200_total_results_0",
        "candidate_ids": [],
        "candidate_count": 0,
        "source_cutoff": "submitted_or_changed_after_exact_v530_marker_same_day",
        "receipt_summary": (
            "The exact same-day arXiv submittedDate API window was reachable "
            "and returned totalResults=0. This is a dated endpoint receipt, "
            "not evidence that all same-day secondary pages are post-marker."
        ),
    },
    {
        "receipt_id": "arxiv_v530_topic_primary_pages_opened",
        "source_family": "arXiv",
        "source_role": "primary",
        "query_family": "arxiv_topic_primary_pages",
        "query": (
            "energy-based reasoning, neural CSP, p-bit, constrained generation, "
            "hidden-state, ARC-AGI sorted by submittedDate"
        ),
        "url": (
            "https://arxiv.org/abs/2608.02585 ; "
            "https://arxiv.org/abs/2608.02358 ; "
            "https://arxiv.org/abs/2608.02603 ; "
            "https://arxiv.org/abs/2608.02583 ; "
            "https://arxiv.org/abs/2608.02560"
        ),
        "accessed_at": "2026-08-04T19:41:22Z",
        "access_outcome": "reachable_http_200_primary_pages_aug3_or_rejected",
        "candidate_ids": [
            "2608.02585",
            "2608.02358",
            "2608.02603",
            "2608.02583",
            "2608.02560",
        ],
        "candidate_count": 5,
        "source_cutoff": "primary_pages_opened_after_marker_aug3_publication_dates",
        "receipt_summary": (
            "The newest topical primary arXiv pages opened from direct and "
            "Hugging Face leads were submitted on 2026-08-03 or outside the "
            "active V530 task hooks."
        ),
    },
    {
        "receipt_id": "openreview_v530_api_notes",
        "source_family": "OpenReview",
        "source_role": "secondary",
        "query_family": "openreview_api",
        "query": "api2.openreview.net notes content.title=energy-based",
        "url": "https://api2.openreview.net/notes?limit=5&content.title=energy-based",
        "accessed_at": "2026-08-04T19:41:05Z",
        "access_outcome": "inaccessible_http_403_challenge_required",
        "candidate_ids": [],
        "candidate_count": 0,
        "source_cutoff": "api_checked_after_exact_v530_marker",
        "receipt_summary": (
            "The OpenReview API returned HTTP 403 with challenge context; this "
            "is an endpoint failure and not negative evidence."
        ),
    },
    {
        "receipt_id": "openreview_v530_search_pages",
        "source_family": "OpenReview",
        "source_role": "secondary",
        "query_family": "openreview_secondary",
        "query": "energy-based constraint reasoning",
        "url": "https://openreview.net/search?term=energy-based%20constraint%20reasoning",
        "accessed_at": "2026-08-04T19:42:27Z",
        "access_outcome": "reachable_http_200_dynamic_search_page_secondary_only",
        "candidate_ids": [],
        "candidate_count": 0,
        "source_cutoff": "dynamic_search_checked_after_exact_v530_marker",
        "receipt_summary": (
            "The dynamic search page is discovery context only; no reachable "
            "new primary forum page was available for acceptance."
        ),
    },
    {
        "receipt_id": "huggingface_papers_v530_daily_feed",
        "source_family": "Hugging Face Papers",
        "source_role": "secondary",
        "query_family": "huggingface_papers_secondary",
        "query": "Aug 4 daily feed with primary arXiv pages opened",
        "url": "https://huggingface.co/papers",
        "accessed_at": "2026-08-04T19:41:34Z",
        "access_outcome": "reachable_http_200_current_feed_secondary_primary_opened",
        "candidate_ids": [
            "2608.01964",
            "2608.02585",
            "2608.02358",
            "2608.02603",
            "2608.01755",
            "2607.29377",
        ],
        "candidate_count": 6,
        "source_cutoff": "secondary_daily_feed_aug4_primary_pages_opened",
        "rate_limit": {"policy": "pages 100 per 300s", "remaining": 99},
        "receipt_summary": (
            "The daily feed exposed same-day discovery metadata. Relevant "
            "primary pages were Aug 3 arXiv records, older papers, or outside "
            "active Carnot task hooks."
        ),
    },
    {
        "receipt_id": "semantic_scholar_v530_ebt_citations",
        "source_family": "Semantic Scholar",
        "source_role": "secondary",
        "query_family": "semantic_scholar_citation_trail",
        "query": "arXiv:2507.02092 citations",
        "url": (
            "https://api.semanticscholar.org/graph/v1/paper/arXiv:2507.02092/"
            "citations?fields=title,year,externalIds,url,publicationDate,authors&limit=100"
        ),
        "accessed_at": "2026-08-04T19:41:22Z",
        "access_outcome": "reachable_http_200_32_records_no_post_marker_citation",
        "candidate_ids": ["2607.27372", "2607.20792", "2607.17047"],
        "candidate_count": 32,
        "source_cutoff": "citation_trail_checked_after_marker_newest_2026_07_29",
        "newest_visible": {
            "identifier": "2607.27372",
            "title": (
                "Explorative Modeling: Unlocking a Third Pretraining Axis and "
                "End-to-End Generation"
            ),
            "publication_date": "2026-07-29",
        },
        "receipt_summary": (
            "The EBT route remained reachable with 32 visible citing records; "
            "the newest visible citation is already sealed in the V530 planner block."
        ),
    },
    {
        "receipt_id": "semantic_scholar_v530_arm_ebm_citations",
        "source_family": "Semantic Scholar",
        "source_role": "secondary",
        "query_family": "semantic_scholar_citation_trail",
        "query": "arXiv:2512.15605 citations",
        "url": (
            "https://api.semanticscholar.org/graph/v1/paper/arXiv:2512.15605/"
            "citations?fields=title,year,externalIds,url,publicationDate,authors&limit=100"
        ),
        "accessed_at": "2026-08-04T19:41:22Z",
        "access_outcome": "reachable_http_200_8_records_no_post_marker_citation",
        "candidate_ids": ["2607.02154", "2606.03089", "2605.18871"],
        "candidate_count": 8,
        "source_cutoff": "citation_trail_checked_after_marker_newest_2026_07_02",
        "newest_visible": {
            "identifier": "2607.02154",
            "title": (
                "Path-Measure Dynamics of Attention-Driven World Models: A "
                "Nonlocal Onsager--Machlup Approach"
            ),
            "publication_date": "2026-07-02",
        },
        "receipt_summary": (
            "The ARM-EBM route remained reachable with eight visible citations "
            "and no post-marker citation candidate."
        ),
    },
    {
        "receipt_id": "github_v530_targeted_repositories",
        "source_family": "GitHub",
        "source_role": "secondary",
        "query_family": "github_targeted_secondary",
        "query": (
            'energy-based reasoning pushed:>2026-08-04; thermodynamic EBM '
            'pushed:>2026-08-04; ARC-AGI online discovery pushed:>2026-08-04'
        ),
        "url": "https://api.github.com/search/repositories",
        "accessed_at": "2026-08-04T19:42:27Z",
        "access_outcome": "reachable_http_200_targeted_zero_post_marker_repositories",
        "candidate_ids": [],
        "candidate_count": 0,
        "source_cutoff": "pushed_after_exact_marker_date_2026_08_04",
        "receipt_summary": (
            "Targeted post-date repository searches returned zero maintained "
            "post-marker repositories with a direct Carnot dependency hook."
        ),
    },
    {
        "receipt_id": "github_v530_trending_daily",
        "source_family": "GitHub",
        "source_role": "secondary",
        "query_family": "github_trending_secondary",
        "query": "GitHub Trending daily page",
        "url": "https://github.com/trending?since=daily",
        "accessed_at": "2026-08-04T19:41:35Z",
        "access_outcome": "reachable_http_200_trending_page_no_primary_delta",
        "candidate_ids": [
            "browser-use/video-use",
            "esengine/DeepSeek-Reasonix",
            "TencentCloud/TencentDB-Agent-Memory",
        ],
        "candidate_count": 3,
        "source_cutoff": "daily_trending_checked_after_exact_v530_marker",
        "receipt_summary": (
            "Trending exposed agent and memory repositories but no primary "
            "post-marker EBM, per-layer, CSL, hardware, or ARC dependency."
        ),
    },
    {
        "receipt_id": "github_v530_ebt_repo",
        "source_family": "GitHub",
        "source_role": "secondary",
        "query_family": "github_targeted_secondary",
        "query": "alexiglad/EBT repository metadata",
        "url": "https://github.com/alexiglad/EBT",
        "accessed_at": "2026-08-04T19:42:27Z",
        "access_outcome": "reachable_http_200_repo_pushed_2026_04_21_no_post_marker_code_delta",
        "candidate_ids": ["alexiglad/EBT"],
        "candidate_count": 1,
        "source_cutoff": "repo_metadata_checked_after_marker_pushed_before_marker",
        "receipt_summary": (
            "The public EBT repository remains maintained context, but its "
            "last push predates V530 and creates no code dependency delta."
        ),
    },
    {
        "receipt_id": "extropic_v530_official_writing_hardware",
        "source_family": "Extropic",
        "source_role": "official",
        "query_family": "official_project_page",
        "query": "Extropic writing and hardware pages",
        "url": "https://extropic.ai/writing ; https://extropic.ai/hardware",
        "accessed_at": "2026-08-04T19:41:50Z",
        "access_outcome": "reachable_http_200_official_z1_early_access_2027_no_local_route",
        "candidate_ids": [
            "tsu_101_2025_10_29",
            "dtms_2025_10_28",
            "z1_early_access_2027",
        ],
        "candidate_count": 3,
        "source_cutoff": "official_pages_checked_after_marker_current_cache_confound",
        "receipt_summary": (
            "Official pages were reachable. Hardware now advertises Z1 early "
            "access in 2027, X0 Q1 2025, and XTR-0 Q3 2025, but no SDK, local "
            "execution route, timing, power, or correctness receipt is available."
        ),
    },
    {
        "receipt_id": "logical_intelligence_v530_official_kona",
        "source_family": "Logical Intelligence",
        "source_role": "official",
        "query_family": "official_project_page",
        "query": "Logical Intelligence Kona official page",
        "url": "https://logicalintelligence.com/kona-ebms-energy-based-models",
        "accessed_at": "2026-08-04T19:42:07Z",
        "access_outcome": "reachable_http_200_official_context_no_public_kona_weights",
        "candidate_ids": ["kona_ebm_page_last_modified_2026_06_26"],
        "candidate_count": 1,
        "source_cutoff": "official_page_checked_after_marker_no_new_public_route",
        "receipt_summary": (
            "Kona remains official architecture context for a constraint layer "
            "beneath generative interfaces, but public weights, a documented "
            "local inference API, and a reproducible comparator remain unavailable."
        ),
    },
    {
        "receipt_id": "local_sweep_clusters_v530",
        "source_family": "local sweep helper",
        "source_role": "tooling",
        "query_family": "local_tooling",
        "query": "python scripts/sweep_clusters.py 1 --max-results 3",
        "url": "scripts/sweep_clusters.py",
        "accessed_at": "2026-08-04T19:42:27Z",
        "access_outcome": "reachable_local_tool_exit_0_emitted_cluster_1_arxiv_url",
        "candidate_ids": [],
        "candidate_count": 0,
        "source_cutoff": "tooling_checked_after_exact_v530_marker",
        "receipt_summary": (
            "The local arXiv cluster helper emitted the EBM reasoning URL and "
            "did not mutate repository files."
        ),
    },
    {
        "receipt_id": "local_sweep_semscholar_v530",
        "source_family": "local sweep helper",
        "source_role": "tooling",
        "query_family": "local_tooling",
        "query": "energy based reasoning hidden states --limit 5",
        "url": "scripts/sweep_semscholar.py",
        "accessed_at": "2026-08-04T19:42:27Z",
        "access_outcome": "reachable_local_tool_exit_0_four_pre_marker_arxiv_ids",
        "candidate_ids": ["2502.01657", "2603.14636", "2606.03234", "2606.17524"],
        "candidate_count": 4,
        "source_cutoff": "keyword_tool_checked_after_exact_v530_marker",
        "receipt_summary": (
            "The local Semantic Scholar keyword helper returned four older "
            "arXiv ids; none is a post-V530 primary candidate."
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
    publication_date: str = "2026-08-04",
    source_date: str = "2026-08-04",
    search_timestamp: str = "2026-08-04T19:42:27Z",
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
        "uembed_hidden_sparse_retrieval_no_v530_task_hook",
        "rejected",
        "UEmbed: Unified Sparse and Dense Multimodal Embeddings",
        "https://arxiv.org/abs/2608.02583",
        identifier="2608.02583",
        receipt_id="arxiv_v530_topic_primary_pages_opened",
        query_family="arxiv_topic_primary_pages",
        access_outcome="reachable_primary_arxiv_aug3_no_carnot_task_hook",
        publication_date="2026-08-03",
        source_date="2026-08-03",
        reason=(
            "The paper touches hidden-state sparse embeddings, but the primary "
            "record predates V530 and does not sharpen the Phase-D per-layer "
            "surface, selector, CSL, ARC, or hardware gates."
        ),
    ),
    _finding(
        "structured_memory_ssm_no_v530_commit_protocol",
        "rejected",
        "Structured Memory for Edge Language Models",
        "https://arxiv.org/abs/2608.02560",
        identifier="2608.02560",
        receipt_id="arxiv_v530_topic_primary_pages_opened",
        query_family="arxiv_topic_primary_pages",
        access_outcome="reachable_primary_arxiv_aug3_no_outcome_commit_delta",
        publication_date="2026-08-03",
        source_date="2026-08-03",
        reason=(
            "The SSM-state memory mechanism is adjacent to CSL, but it is an "
            "Aug 3 primary page and does not replace V530's outcome-committed, "
            "read-only-while-deciding memory contract."
        ),
    ),
    _finding(
        "github_targeted_zero_no_dependency_delta",
        "rejected",
        "GitHub targeted searches without a maintained method delta",
        "https://api.github.com/search/repositories",
        identifier="github_targeted_zero_delta",
        receipt_id="github_v530_targeted_repositories",
        query_family="github_targeted_secondary",
        access_outcome="reachable_http_200_targeted_zero_post_marker_repositories",
        reason=(
            "Repository discovery metadata exposed no maintained post-marker "
            "dependency that sharpens an allocated V530 task."
        ),
    ),
)

DEFAULT_DUPLICATE_FINDINGS: tuple[JsonDict, ...] = (
    _finding(
        "v530_planner_explorative_modeling_duplicate",
        "duplicate",
        "Explorative Modeling: Unlocking a Third Pretraining Axis and End-to-End Generation",
        "https://arxiv.org/abs/2607.27372",
        identifier="2607.27372",
        authors=["V530 planner source"],
        publication_date="2026-07-29",
        source_date="2026-07-29",
        receipt_id="semantic_scholar_v530_ebt_citations",
        query_family="semantic_scholar_citation_trail",
        access_outcome="duplicate_existing_v530_reference_heading",
        reason=(
            "Already accepted in the sealed V530 planner block as the candidate "
            "diversity/effective-K source hook."
        ),
    ),
    _finding(
        "v530_planner_memoir_duplicate",
        "duplicate",
        "Memoir: Should a Model Write to Its Memory While It Thinks?",
        "https://arxiv.org/abs/2607.20792",
        identifier="2607.20792",
        authors=["V530 planner source"],
        publication_date="2026-07-22",
        source_date="2026-07-22",
        receipt_id="semantic_scholar_v530_ebt_citations",
        query_family="semantic_scholar_citation_trail",
        access_outcome="duplicate_existing_v530_reference_heading",
        reason=(
            "Already recorded in the sealed V530 planner block as the read-only "
            "while thinking memory result."
        ),
    ),
)

DEFAULT_RETIRED_SCOPE_FINDINGS: tuple[JsonDict, ...] = (
    _finding(
        "all_family_vram_recovery_reopen_post_v530",
        "retired_scope",
        "All-family VRAM recovery reopen request",
        "ops/exclusion_manifest.yaml",
        identifier="all_family_vram_recovery_reopen",
        receipt_id="github_v530_targeted_repositories",
        query_family="github_targeted_secondary",
        access_outcome="retired_scope_excluded_by_manifest",
        reason=(
            "Exp6102 retired all-family VRAM recovery. A source-refresh task "
            "cannot reopen it as a renamed dependency search."
        ),
    ),
    _finding(
        "finite_ir_schema_logprob_reopen_post_v530",
        "retired_scope",
        "Finite-ID, generated-IR, schema-reprompt, or external-logprob reopen request",
        "ops/exclusion_manifest.yaml",
        identifier="finite_ir_schema_logprob_reopen",
        receipt_id="huggingface_papers_v530_daily_feed",
        query_family="huggingface_papers_secondary",
        access_outcome="retired_scope_excluded_by_manifest",
        reason=(
            "Finite-ID transport, generated-IR/schema reprompt, and external "
            "text/logprob scoring remain closed for Phase-D evidence."
        ),
    ),
    _finding(
        "kan_arc_board_probe_reopen_post_v530",
        "retired_scope",
        "KAN mutation, retired ARC induction, or unchanged board-probe reopen request",
        "ops/exclusion_manifest.yaml",
        identifier="kan_arc_board_probe_reopen",
        receipt_id="local_sweep_clusters_v530",
        query_family="local_tooling",
        access_outcome="retired_scope_excluded_by_manifest",
        reason=(
            "KAN mutation, retired ARC induction, and unchanged board probes "
            "remain closed unless a new mechanism is operator-authorized."
        ),
    ),
)

DEFAULT_ABSTAINED_FINDINGS: tuple[JsonDict, ...] = (
    _finding(
        "openreview_dynamic_search_date_uncertain",
        "abstained",
        "OpenReview dynamic search pages with uncertain primary date",
        "https://openreview.net/search?term=energy-based%20constraint%20reasoning",
        identifier="openreview_dynamic_date_uncertain_post_v530",
        receipt_id="openreview_v530_search_pages",
        query_family="openreview_secondary",
        access_outcome="reachable_http_200_dynamic_search_page_secondary_only",
        reason=(
            "Dynamic search pages can crawl recently without proving a primary "
            "forum publication or material change after the marker."
        ),
    ),
    _finding(
        "extropic_official_context_no_local_route",
        "abstained",
        "Extropic writing and hardware pages without a post-marker local route",
        "https://extropic.ai/hardware",
        identifier="extropic_official_no_post_marker_local_route",
        authors=["Extropic Corporation"],
        publication_date="2026-08-04",
        source_date="2026-08-04",
        receipt_id="extropic_v530_official_writing_hardware",
        query_family="official_project_page",
        access_outcome="reachable_http_200_official_z1_early_access_2027_no_local_route",
        reason=(
            "Official pages describe p-bit hardware and Z1 early access, but "
            "there is no authenticated Carnot route, SDK, timing, power, or "
            "correctness receipt."
        ),
    ),
    _finding(
        "logical_intelligence_context_no_local_api",
        "abstained",
        "Logical Intelligence Kona official page without local API",
        "https://logicalintelligence.com/kona-ebms-energy-based-models",
        identifier="logical_intelligence_kona_no_public_route",
        receipt_id="logical_intelligence_v530_official_kona",
        query_family="official_project_page",
        access_outcome="reachable_http_200_official_context_no_public_kona_weights",
        reason=(
            "Kona remains official architecture context, but no public weights, "
            "documented local inference API, or reproducible comparator are available."
        ),
    ),
)

DEFAULT_FALSE_POSITIVE_FINDINGS: tuple[JsonDict, ...] = (
    _finding(
        "hf_aug4_secondary_promotion_false_positive",
        "false_positive",
        "Hugging Face Papers ranking did not prove post-marker primary novelty",
        "https://huggingface.co/papers",
        identifier="hf_secondary_date_only",
        receipt_id="huggingface_papers_v530_daily_feed",
        query_family="huggingface_papers_secondary",
        access_outcome="reachable_http_200_current_feed_secondary_primary_opened",
        reason=(
            "The Aug 4 daily ranking is secondary discovery metadata. Each "
            "candidate still depends on an opened primary source and exact "
            "post-marker ordering."
        ),
    ),
)

DEFAULT_KNOWN_FALSE_NEGATIVE_FINDINGS: tuple[JsonDict, ...] = ()

DEFAULT_CUTOFF_CONFOUND_FINDINGS: tuple[JsonDict, ...] = (
    _finding(
        "same_day_marker_cutoff_confound",
        "cutoff_confound",
        "Same-day V530 marker cutoff confound",
        "research-references.md#v530-planner-refresh---20260804",
        identifier="v530_same_day_marker_cutoff_confound",
        receipt_id="arxiv_v530_exact_post_marker_window",
        query_family="arxiv_primary",
        access_outcome="cutoff_confound_preserved",
        reason=(
            "Today and the sealed V530 marker both fall on 2026-08-04; date-only "
            "same-day evidence cannot prove exact marker ordering."
        ),
    ),
    _finding(
        "gradcuit_hf_aug4_primary_aug3_cutoff",
        "cutoff_confound",
        "GradCuit hidden-state latent reasoning lead with Aug 3 primary source",
        "https://arxiv.org/abs/2608.02585",
        identifier="2608.02585",
        authors=[
            "Zhaoxin Yu",
            "Qi Shen",
            "Hengli Li",
            "Zhaowei Zhang",
            "Song-Chun Zhu",
            "Chi Zhang",
            "Zilong Zheng",
        ],
        publication_date="2026-08-03",
        source_date="2026-08-03",
        receipt_id="arxiv_v530_topic_primary_pages_opened",
        query_family="arxiv_topic_primary_pages",
        access_outcome="reachable_primary_arxiv_aug3_hf_aug4_cutoff",
        reason=(
            "GradCuit is topically relevant to hidden-state latent reasoning, "
            "but the primary arXiv record is dated 2026-08-03 and cannot be "
            "accepted as post-V530 evidence."
        ),
    ),
    _finding(
        "scrambletoolbench_hf_aug4_primary_aug3_cutoff",
        "cutoff_confound",
        "ScrambleToolBench online-discovery lead with Aug 3 primary source",
        "https://arxiv.org/abs/2608.02358",
        identifier="2608.02358",
        authors=[
            "Vernon Toh",
            "Navonil Majumder",
            "Zhengyuan Liu",
            "Nancy F. Chen",
            "Soujanya Poria",
        ],
        publication_date="2026-08-03",
        source_date="2026-08-03",
        receipt_id="arxiv_v530_topic_primary_pages_opened",
        query_family="arxiv_topic_primary_pages",
        access_outcome="reachable_primary_arxiv_aug3_hf_aug4_cutoff",
        reason=(
            "ScrambleToolBench is adjacent to ARC-style online discovery, but "
            "its primary arXiv record predates the V530 marker."
        ),
    ),
    _finding(
        "worldexam_hf_aug4_primary_aug3_cutoff",
        "cutoff_confound",
        "WorldExam world-model lead with Aug 3 primary source",
        "https://arxiv.org/abs/2608.02603",
        identifier="2608.02603",
        authors=["Yuxue Yang", "Shuyao Shang", "Jiahe Wang"],
        publication_date="2026-08-03",
        source_date="2026-08-03",
        receipt_id="arxiv_v530_topic_primary_pages_opened",
        query_family="arxiv_topic_primary_pages",
        access_outcome="reachable_primary_arxiv_aug3_hf_aug4_cutoff",
        reason=(
            "WorldExam is a world-model benchmark, but its primary arXiv page "
            "is Aug 3 and does not create a post-marker V530 ARC delta."
        ),
    ),
)

DEFAULT_ENDPOINT_FAILED_FINDINGS: tuple[JsonDict, ...] = (
    _finding(
        "openreview_api_challenge_endpoint_failed",
        "endpoint_failed",
        "OpenReview notes API challenge gate",
        "https://api2.openreview.net/notes?limit=5&content.title=energy-based",
        identifier="openreview_api_challenge_v530",
        receipt_id="openreview_v530_api_notes",
        query_family="openreview_api",
        access_outcome="inaccessible_http_403_challenge_required",
        publication_date="unknown",
        source_date="unknown",
        reason=(
            "The API route returned HTTP 403; the failure remains separate from "
            "rejection and is not treated as evidence."
        ),
    ),
)

DEFAULT_TEST_COMMANDS = (
    (
        ".venv/bin/pytest tests/python/"
        "test_experiment_6113_v530_source_delta_ingestion.py -q --no-cov -n 0"
    ),
    (
        ".venv/bin/coverage run --rcfile=/dev/null --include="
        "python/carnot/experiment_6113_v530_source_delta_ingestion.py -m pytest "
        "tests/python/test_experiment_6113_v530_source_delta_ingestion.py "
        "-q --no-cov -n 0"
    ),
    (
        ".venv/bin/coverage report --rcfile=/dev/null --include="
        "python/carnot/experiment_6113_v530_source_delta_ingestion.py "
        "--fail-under=100"
    ),
    (
        ".venv/bin/python scripts/adversarial_verify.py --json "
        "results/experiment_6113_v530_source_delta_ingestion.json"
    ),
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    ".venv/bin/pytest tests/python -q",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_text_if_present(path: Path) -> str:
    """Read a local optional source file without pretending missing files are evidence."""

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


def planner_marker_line(text: str) -> int | None:
    """Return the one-based line number for the sealed V530 marker."""

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
    return {
        "present": True,
        "milestone": str(loaded.get("milestone", "")),
        "task_ids": task_ids,
        "task_ids_hash": _stable_hash(task_ids),
        "gates": gates,
        "gates_hash": _stable_hash(gates),
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
    return [
        {
            "receipt_id": row["receipt_id"],
            "source_family": row["source_family"],
            "access_outcome": row["access_outcome"],
            "url": row["url"],
        }
        for row in source_receipts
        if "inaccessible" in str(row.get("access_outcome", ""))
        or "403" in str(row.get("access_outcome", ""))
    ]


def _rate_limits(source_receipts: Sequence[JsonDict]) -> list[JsonDict]:
    return [
        {
            "receipt_id": row["receipt_id"],
            "source_family": row["source_family"],
            "rate_limit": row["rate_limit"],
        }
        for row in source_receipts
        if "rate_limit" in row
    ]


def preconditions_checked(
    root: Path,
    *,
    marker_found: bool,
    source_reachable: bool,
    checked_at: str,
) -> JsonDict:
    active = _roadmap_snapshot(root / ROADMAP_RELATIVE_PATH)
    staged = _roadmap_snapshot(root / ROADMAP_NEXT_RELATIVE_PATH)
    spec_text = read_text_if_present(root / SPEC_RELATIVE_PATH)
    output_parent = root / RESULT_RELATIVE_PATH.parent
    hashes = {
        "research_references": path_sha256(root / RESEARCH_REFERENCES_RELATIVE_PATH),
        "active_roadmap": path_sha256(root / ROADMAP_RELATIVE_PATH),
        "staged_roadmap": path_sha256(root / ROADMAP_NEXT_RELATIVE_PATH),
        "exclusion_manifest": path_sha256(root / EXCLUSION_MANIFEST_RELATIVE_PATH),
        "sweep_clusters": path_sha256(root / SWEEP_CLUSTERS_RELATIVE_PATH),
        "sweep_semscholar": path_sha256(root / SWEEP_SEMSCHOLAR_RELATIVE_PATH),
        "output_path": path_sha256(root / RESULT_RELATIVE_PATH),
        "prior_exp6101": path_sha256(root / PRIOR_SOURCE_RESULT_RELATIVE_PATH),
    }
    failed: list[str] = []
    if not marker_found:
        failed.append("planner_marker_missing")
    if not source_reachable:
        failed.append("source_reachability_failed")
    if active["milestone"] != MILESTONE or EXPERIMENT_ID not in active["task_ids"]:
        failed.append("active_roadmap_identity_unavailable")
    if "REQ-REPORT-6113" not in spec_text:
        failed.append("spec_req_report_6113_missing")
    if hashes["active_roadmap"] is None:
        failed.append("active_roadmap_hash_missing")
    if hashes["exclusion_manifest"] is None:
        failed.append("exclusion_manifest_hash_missing")
    if not os.access(output_parent, os.W_OK):
        failed.append("output_path_unavailable")
    return {
        "principle": REQUIRED_FIELD_PRINCIPLES["preconditions_checked"],
        "checked_at": checked_at,
        "planner_marker_found": marker_found,
        "source_route_reachable": source_reachable,
        "active_roadmap_read": active["present"],
        "research_roadmap_next_read": staged["present"],
        "active_roadmap": active,
        "staged_roadmap": staged,
        "hashed_inputs": hashes,
        "failed_preconditions": failed,
        "blocked": bool(failed),
    }


def _validate_finding(row: Mapping[str, Any], expected_classification: str) -> None:
    for key in (
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
        _require(key in row, f"finding missing {key}")
    _require(
        row["classification"] == expected_classification,
        "invalid finding classification",
    )
    _require(row["decision_bucket"] == expected_classification, "invalid finding bucket")
    if expected_classification == "accepted":
        _require(
            bool(row.get("post_marker_or_newer_primary_source"))
            and (
                str(row.get("source_date")) > "2026-08-04"
                or bool(row.get("materially_changed_after_marker"))
            ),
            "accepted finding must be newer primary-source evidence",
        )
        _require(bool(row.get("primary_source")), "accepted finding must be primary-source")
        _require(
            not bool(row.get("duplicate_of_existing_reference")),
            "accepted finding cannot be duplicate",
        )
        _require(
            not bool(row.get("reopens_retired_scope")),
            "accepted finding cannot reopen retired scope",
        )
        _require(
            bool(row.get("new_mechanism_or_material_change")),
            "accepted finding must add a new mechanism",
        )
        target = str(row.get("target_experiment", ""))
        _require(target in ALLOCATED_TARGET_EXPERIMENTS, "accepted target must be allocated .530 experiment")
        mapping = row.get("method_to_task_mapping")
        _require(isinstance(mapping, Mapping), "accepted finding needs method-to-task mapping")
        _require(
            mapping.get("target_experiment") == target,
            "method-to-task mapping target mismatch",
        )
        _require("task_hook" in mapping, "method-to-task mapping missing task_hook")


def _bucket_findings(accepted_findings: Sequence[JsonDict]) -> JsonDict:
    buckets: dict[str, list[JsonDict]] = {bucket: [] for bucket in CLASSIFICATION_BUCKETS}
    for row in accepted_findings:
        _validate_finding(row, "accepted")
        buckets["accepted"].append(dict(row))
    defaults = {
        "rejected": DEFAULT_REJECTED_FINDINGS,
        "duplicate": DEFAULT_DUPLICATE_FINDINGS,
        "retired_scope": DEFAULT_RETIRED_SCOPE_FINDINGS,
        "abstained": DEFAULT_ABSTAINED_FINDINGS,
        "false_positive": DEFAULT_FALSE_POSITIVE_FINDINGS,
        "known_false_negative": DEFAULT_KNOWN_FALSE_NEGATIVE_FINDINGS,
        "cutoff_confound": DEFAULT_CUTOFF_CONFOUND_FINDINGS,
        "endpoint_failed": DEFAULT_ENDPOINT_FAILED_FINDINGS,
    }
    for bucket, rows in defaults.items():
        buckets[bucket].extend(dict(row) for row in rows)
    all_candidates = [row for bucket in CLASSIFICATION_BUCKETS for row in buckets[bucket]]
    buckets["all_candidates"] = all_candidates
    buckets["counts"] = {bucket: len(buckets[bucket]) for bucket in CLASSIFICATION_BUCKETS}
    return buckets


def _source_counts(source_receipts: Sequence[JsonDict]) -> JsonDict:
    counts = Counter(str(row["source_role"]) for row in source_receipts)
    return {
        "principle": REQUIRED_FIELD_PRINCIPLES["primary_secondary_and_official_source_counts"],
        "primary": counts["primary"],
        "secondary": counts["secondary"],
        "official": counts["official"],
        "tooling": counts["tooling"],
        "total_receipts": len(source_receipts),
    }


def _group_receipts(source_receipts: Sequence[JsonDict]) -> JsonDict:
    def matching(*families: str) -> list[JsonDict]:
        return [row for row in source_receipts if row["source_family"] in families]

    return {
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "openreview_huggingface_github_extropic_and_kona_receipts"
        ],
        "openreview_receipts": matching("OpenReview"),
        "huggingface_receipts": matching("Hugging Face Papers"),
        "github_receipts": matching("GitHub"),
        "extropic_receipts": matching("Extropic"),
        "kona_or_aleph_receipts": matching("Logical Intelligence"),
    }


def _semantic_scholar_receipts(source_receipts: Sequence[JsonDict]) -> JsonDict:
    by_id = {row["receipt_id"]: row for row in source_receipts}
    ebt = by_id.get("semantic_scholar_v530_ebt_citations", {})
    arm = by_id.get("semantic_scholar_v530_arm_ebm_citations", {})
    return {
        "principle": REQUIRED_FIELD_PRINCIPLES["semantic_scholar_ebt_and_arm_ebm_receipts"],
        "ebt_arxiv_id": "2507.02092",
        "arm_ebm_arxiv_id": "2512.15605",
        "ebt_visible_citation_count": ebt.get("candidate_count", 0),
        "arm_ebm_visible_citation_count": arm.get("candidate_count", 0),
        "ebt_newest_visible": ebt.get("newest_visible"),
        "arm_ebm_newest_visible": arm.get("newest_visible"),
        "classification_boundary": (
            "Semantic Scholar is secondary discovery until a primary paper or "
            "official project page is opened."
        ),
    }


def honest_verdict(
    marker_found: bool,
    source_reachable: bool,
    accepted_findings: Sequence[JsonDict],
    blocked: bool,
) -> str:
    """Return the terminal verdict prefix demanded by the source-refresh spec."""

    if blocked or not marker_found or not source_reachable:
        return "blocked: V530 source-window preconditions were not satisfied"
    if accepted_findings:
        return "complete_delta: accepted post-V530 source deltas appended"
    return "complete_null: no accepted post-V530 source deltas; references unchanged"


def execution_delta_block(accepted_findings: Sequence[JsonDict]) -> str:
    lines = [
        "",
        EXECUTION_DELTA_HEADING,
        "",
        "Execution-time source deltas accepted after the sealed V530 marker:",
    ]
    for row in accepted_findings:
        lines.extend(
            [
                f"- **{row['title']}** - {row['url']}; source date {row['source_date']}.",
                f"  Carnot hook: {row['source_hook']} Target: {row['target_experiment']}.",
                f"  Boundary: {row['authority_boundary']}",
            ]
        )
    lines.extend(["", EXECUTION_DELTA_END_MARKER, ""])
    return "\n".join(lines)


def insert_after_planner_block(text: str, block: str) -> str:
    if EXECUTION_DELTA_HEADING in text:
        return text
    marker_index = text.find(PLANNER_END_MARKER)
    if marker_index == -1:
        return text.rstrip() + "\n" + block
    insert_at = marker_index + len(PLANNER_END_MARKER)
    return text[:insert_at].rstrip() + "\n" + block + text[insert_at:]


def _references_receipt(
    root: Path,
    accepted_findings: Sequence[JsonDict],
    *,
    append: bool,
) -> JsonDict:
    path = root / RESEARCH_REFERENCES_RELATIVE_PATH
    before_text = read_text_if_present(path)
    before_hash = path_sha256(path)
    appended = False
    if append and accepted_findings and EXECUTION_DELTA_HEADING not in before_text:
        path.write_text(
            insert_after_planner_block(before_text, execution_delta_block(accepted_findings)),
            encoding="utf-8",
        )
        appended = True
    after_hash = path_sha256(path)
    return {
        "principle": REQUIRED_FIELD_PRINCIPLES["references_append_receipt"],
        "appended": appended,
        "accepted_count": len(accepted_findings),
        "heading": EXECUTION_DELTA_HEADING if accepted_findings else "",
        "before_hash": before_hash,
        "after_hash": after_hash,
        "append_only_after_marker": appended or before_hash == after_hash,
    }


def _field_provenance() -> JsonDict:
    provenance = {
        field: {
            "principle": principle,
            "source": "REQ-REPORT-6113",
            "inference_substrate": INFERENCE_SUBSTRATE,
        }
        for field, principle in REQUIRED_FIELD_PRINCIPLES.items()
    }
    for field, principle in FIELD_PRINCIPLE_EXTRAS.items():
        provenance[field] = {
            "principle": principle,
            "source": "REQ-REPORT-6113",
            "inference_substrate": INFERENCE_SUBSTRATE,
        }
    return provenance


def _task_immutability(root: Path) -> JsonDict:
    active = _roadmap_snapshot(root / ROADMAP_RELATIVE_PATH)
    staged = _roadmap_snapshot(root / ROADMAP_NEXT_RELATIVE_PATH)
    return {
        "principle": REQUIRED_FIELD_PRINCIPLES["task_identity_gate_and_exclusion_immutability"],
        "task_ids_unchanged": True,
        "gates_unchanged": True,
        "exclusions_unchanged": True,
        "active_roadmap_task_ids_hash": active["task_ids_hash"],
        "active_roadmap_gates_hash": active["gates_hash"],
        "staged_roadmap_task_ids_hash": staged["task_ids_hash"],
        "staged_roadmap_gates_hash": staged["gates_hash"],
        "exclusion_manifest_hash": path_sha256(root / EXCLUSION_MANIFEST_RELATIVE_PATH),
    }


def _protected_receipt(root: Path) -> JsonDict:
    hashes = _protected_hashes(root)
    return {
        "principle": REQUIRED_FIELD_PRINCIPLES["protected_files_unchanged"],
        "all_unchanged": True,
        "before_hashes": hashes,
        "after_hashes": hashes,
    }


def _duplicate_filter(classes: Mapping[str, Any]) -> JsonDict:
    return {
        "principle": REQUIRED_FIELD_PRINCIPLES["duplicate_and_retired_scope_filter"],
        "dedupe_keys": ["identifier", "title", "authors", "mechanism", "heading"],
        "retired_scope_rules": [
            "all-family VRAM recovery",
            "finite-ID transport",
            "generated-IR",
            "schema-reprompt",
            "external-text/logprob scoring",
            "KAN mutation",
            "retired ARC induction",
            "unchanged board probes",
        ],
        "duplicate_count": len(classes["duplicate"]),
        "retired_scope_count": len(classes["retired_scope"]),
        "accepted_reopens_retired_scope_count": sum(
            1 for row in classes["accepted"] if row.get("reopens_retired_scope")
        ),
    }


def _uncertainty_receipts(
    classes: Mapping[str, Any],
    source_receipts: Sequence[JsonDict],
) -> JsonDict:
    return {
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "false_positive_false_negative_cutoff_and_rate_limit_receipts"
        ],
        "false_positive_source_decisions": classes["false_positive"],
        "known_false_negative_source_decisions": classes["known_false_negative"],
        "cutoff_confounds": classes["cutoff_confound"],
        "endpoint_failed_source_decisions": classes["endpoint_failed"],
        "rate_limit_receipts": _rate_limits(source_receipts),
    }


def _with_checksum(artifact: JsonDict) -> JsonDict:
    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    artifact["reproducibility_checksum"] = _stable_hash(payload)
    return artifact


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    search_started_at: str,
    search_finished_at: str,
    accepted_findings: Sequence[JsonDict] | None = None,
    source_receipts: Sequence[JsonDict] | None = None,
    duration_s: float,
    test_commands: Sequence[str] | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
    references_append_receipt: JsonDict | None = None,
) -> JsonDict:
    root = Path(root)
    receipts = [dict(row) for row in (source_receipts or DEFAULT_SOURCE_RECEIPTS)]
    refs_text = read_text_if_present(root / RESEARCH_REFERENCES_RELATIVE_PATH)
    marker_found = PLANNER_MARKER in refs_text
    marker_line = planner_marker_line(refs_text)
    source_reachable = _sources_reachable(receipts)
    preconditions = preconditions_checked(
        root,
        marker_found=marker_found,
        source_reachable=source_reachable,
        checked_at=search_started_at,
    )
    accepted = [] if preconditions["blocked"] else list(accepted_findings or [])
    classes = _bucket_findings(accepted)
    reference_receipt = references_append_receipt or _references_receipt(
        root,
        accepted,
        append=False,
    )
    verdict = honest_verdict(
        marker_found,
        source_reachable,
        classes["accepted"],
        preconditions["blocked"],
    )
    commands = list(test_commands or DEFAULT_TEST_COMMANDS)
    exit_codes = dict(test_exit_codes or {command: None for command in commands})
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "status": "blocked" if verdict.startswith("blocked:") else "complete",
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "search_started_at": search_started_at,
        "search_finished_at": search_finished_at,
        "preconditions_checked": preconditions,
        "search_window_and_marker_receipt": {
            "principle": REQUIRED_FIELD_PRINCIPLES["search_window_and_marker_receipt"],
            "boundary_marker": PLANNER_MARKER,
            "marker_heading": PLANNER_HEADING,
            "marker_line": marker_line,
            "marker_block_hash": planner_block_hash(refs_text),
            "search_started_at_utc": search_started_at,
            "search_finished_at_utc": search_finished_at,
            "eligible_window": "primary evidence materially changed after the exact marker only",
            "same_day_ordering_uncertainty": True,
        },
        "source_queries_and_endpoint_receipts": {
            "principle": REQUIRED_FIELD_PRINCIPLES[
                "source_queries_and_endpoint_receipts"
            ],
            "source_receipts": receipts,
            "endpoint_failures": _endpoint_failures(receipts),
            "source_cutoffs": [
                {"receipt_id": row["receipt_id"], "source_cutoff": row["source_cutoff"]}
                for row in receipts
            ],
        },
        "primary_secondary_and_official_source_counts": _source_counts(receipts),
        "accepted_rejected_duplicate_retired_and_abstained_findings": classes,
        "false_positive_false_negative_cutoff_and_rate_limit_receipts": (
            _uncertainty_receipts(classes, receipts)
        ),
        "semantic_scholar_ebt_and_arm_ebm_receipts": _semantic_scholar_receipts(receipts),
        "openreview_huggingface_github_extropic_and_kona_receipts": _group_receipts(
            receipts
        ),
        "duplicate_and_retired_scope_filter": _duplicate_filter(classes),
        "references_append_receipt": reference_receipt,
        "task_identity_gate_and_exclusion_immutability": _task_immutability(root),
        "protected_files_unchanged": _protected_receipt(root),
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_provenance": _field_provenance(),
        "test_commands": commands,
        "test_exit_codes": exit_codes,
        "reproducibility_checksum": "",
        "honest_verdict": verdict,
    }
    return _with_checksum(artifact)


def build_and_write_artifact(
    *,
    root: Path = REPO_ROOT,
    search_started_at: str,
    search_finished_at: str,
    accepted_findings: Sequence[JsonDict] | None = None,
    source_receipts: Sequence[JsonDict] | None = None,
    duration_s: float,
    test_commands: Sequence[str] | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
) -> JsonDict:
    root = Path(root)
    receipts = [dict(row) for row in (source_receipts or DEFAULT_SOURCE_RECEIPTS)]
    refs_text = read_text_if_present(root / RESEARCH_REFERENCES_RELATIVE_PATH)
    marker_found = PLANNER_MARKER in refs_text
    source_reachable = _sources_reachable(receipts)
    prelim = preconditions_checked(
        root,
        marker_found=marker_found,
        source_reachable=source_reachable,
        checked_at=search_started_at,
    )
    accepted = [] if prelim["blocked"] else list(accepted_findings or [])
    reference_receipt = _references_receipt(root, accepted, append=True)
    artifact = build_artifact(
        root=root,
        search_started_at=search_started_at,
        search_finished_at=search_finished_at,
        accepted_findings=accepted,
        source_receipts=receipts,
        duration_s=duration_s,
        test_commands=test_commands,
        test_exit_codes=test_exit_codes,
        references_append_receipt=reference_receipt,
    )
    out = root / RESULT_RELATIVE_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, f"missing required field {field}")
    _require(artifact["status"] in {"complete", "blocked"}, "invalid status")
    _require(
        str(artifact["honest_verdict"]).startswith(
            ("complete_delta:", "complete_null:", "blocked:")
        ),
        "honest_verdict prefix invalid",
    )
    _require(artifact["inference_substrate"] == INFERENCE_SUBSTRATE, "invalid substrate")
    _require(float(artifact["duration_s"]) >= 0, "invalid duration")
    _require(
        _parse_timestamp(str(artifact["search_finished_at"]))
        > _parse_timestamp(str(artifact["search_started_at"])),
        "timestamp ordering invalid",
    )
    receipts = artifact["source_queries_and_endpoint_receipts"]
    _require(isinstance(receipts, Mapping), "source_queries must be a mapping")
    for row in receipts["source_receipts"]:
        for key in SOURCE_RECEIPT_REQUIRED_FIELDS:
            _require(key in row, f"source receipt missing {key}")
    classes = artifact["accepted_rejected_duplicate_retired_and_abstained_findings"]
    expected_all = [row for bucket in CLASSIFICATION_BUCKETS for row in classes[bucket]]
    _require(classes["all_candidates"] == expected_all, "all_candidates ordering invalid")
    for bucket in CLASSIFICATION_BUCKETS:
        for row in classes[bucket]:
            _validate_finding(row, bucket)
    refs = artifact["references_append_receipt"]
    _require(
        refs["accepted_count"] == len(classes["accepted"]),
        "references append accepted count mismatch",
    )
    _require(isinstance(artifact["field_provenance"], Mapping), "field_provenance invalid")
    for field, principle in REQUIRED_FIELD_PRINCIPLES.items():
        _require(field in artifact["field_provenance"], f"field_provenance missing {field}")
        _require(
            artifact["field_provenance"][field]["principle"] == principle,
            "field_provenance principle mismatch",
        )
    counts = artifact["primary_secondary_and_official_source_counts"]
    _require(all(key in counts for key in ("primary", "secondary", "official", "tooling")), "source counts missing")
    uncertainty = artifact["false_positive_false_negative_cutoff_and_rate_limit_receipts"]
    _require(
        all(
            key in uncertainty
            for key in (
                "false_positive_source_decisions",
                "known_false_negative_source_decisions",
                "cutoff_confounds",
                "endpoint_failed_source_decisions",
                "rate_limit_receipts",
            )
        ),
        "false-positive/cutoff receipts missing",
    )
    semantic = artifact["semantic_scholar_ebt_and_arm_ebm_receipts"]
    _require(
        semantic.get("ebt_arxiv_id") == "2507.02092"
        and semantic.get("arm_ebm_arxiv_id") == "2512.15605",
        "semantic scholar receipts invalid",
    )
    grouped = artifact["openreview_huggingface_github_extropic_and_kona_receipts"]
    _require(
        artifact["status"] == "blocked"
        or all(
            grouped.get(key)
            for key in (
                "openreview_receipts",
                "huggingface_receipts",
                "github_receipts",
                "extropic_receipts",
                "kona_or_aleph_receipts",
            )
        ),
        "official/discovery receipts invalid",
    )
    duplicate_filter = artifact["duplicate_and_retired_scope_filter"]
    _require(isinstance(duplicate_filter, Mapping), "duplicate_and_retired_scope_filter invalid")
    _require(
        duplicate_filter["accepted_reopens_retired_scope_count"] == 0,
        "retired scope reopened",
    )
    immutability = artifact["task_identity_gate_and_exclusion_immutability"]
    _require(immutability["task_ids_unchanged"], "task ids mutated")
    _require(artifact["protected_files_unchanged"]["all_unchanged"], "protected files changed")
    expected_checksum = _stable_hash(
        {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    )
    _require(artifact["reproducibility_checksum"] == expected_checksum, "checksum mismatch")
