"""Exp5962: ingest post-V528 source deltas with explicit uncertainty.

Spec refs: REQ-REPORT-5962, SCENARIO-REPORT-5962-ZERO-FINDING,
SCENARIO-REPORT-5962-ACCEPT-BOUNDED-DELTA,
SCENARIO-REPORT-5962-SOURCE-UNCERTAINTY,
SCENARIO-REPORT-5962-DUPLICATE-AND-RETIRED-SCOPE,
SCENARIO-REPORT-5962-SCHEMA.

This module records a source-ledger pass. It does not run a model, touch the
conductor, or change roadmap gates. The purpose is to make the V528 planning
boundary falsifiable: external sources are dated, source uncertainty remains
visible, and only a genuinely new primary-source mechanism inside the active
roadmap may append to the references ledger.
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
RESULT_RELATIVE_PATH = Path("results/experiment_5962_v528_source_delta_ingestion.json")

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
    "results/experiment_5934_v527_source_delta_ingestion.json"
)

EXPERIMENT = "experiment_5962_v528_source_delta_ingestion"
EXPERIMENT_ID = "exp5962-v528-source-delta-ingestion"
MILESTONE = "2026.07.528"
RUN_DATE = "20260803"
RANDOM_SEED = 5962
SCHEMA = "carnot.experiment_5962.v528_source_delta_ingestion.v1"
INFERENCE_SUBSTRATE = "aggregation_from_external_primary_sources"

PLANNER_HEADING = "## V528 Planner Refresh - 20260726"
PLANNER_MARKER = "V528-PLANNER-REFRESH-20260726-END"
PLANNER_END_MARKER = f"<!-- {PLANNER_MARKER} -->"
EXECUTION_DELTA_HEADING = "## V528 Execution Source Delta - 20260803"
EXECUTION_DELTA_END_MARKER = "<!-- V528-EXECUTION-SOURCE-DELTA-20260803-END -->"
MARKER_DATE = "2026-07-26"

ALLOCATED_TARGET_EXPERIMENTS = (
    "exp5963-exact-atom-pair-fixture",
    "exp5964-sota-atom-compatibility-corpus",
    "exp5965-portable-atom-energy-ranker",
    "exp5966-discriminative-constraint-acquisition",
    "exp5967-delayed-commit-memory-fixture",
    "exp5968-delayed-commit-csl-prospective",
    "exp5969-csl-poison-drift-abi-audit",
    "exp5970-arc-strip-swap-sentinel",
    "exp5971-arc-strip-swap-battery",
    "exp5972-arc-llm-on-budget2000-feasibility",
)

SPEC_REFS = (
    "REQ-REPORT-5962",
    "SCENARIO-REPORT-5962-ZERO-FINDING",
    "SCENARIO-REPORT-5962-ACCEPT-BOUNDED-DELTA",
    "SCENARIO-REPORT-5962-SOURCE-UNCERTAINTY",
    "SCENARIO-REPORT-5962-DUPLICATE-AND-RETIRED-SCOPE",
    "SCENARIO-REPORT-5962-SCHEMA",
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
        "Acceptance requires a new mechanism or materially changed evidence "
        "relevant to the roadmap."
    ),
    "false_positive_false_negative_cutoff_and_rate_limit_receipts": (
        "Uncertainty and access failure cannot be silently converted into rejection."
    ),
    "semantic_scholar_ebt_and_arm_ebm_receipts": (
        "Discovery indexes are context until a primary reproducible source is opened."
    ),
    "openreview_huggingface_github_extropic_and_kona_receipts": (
        "Discovery indexes are context until a primary reproducible source is opened."
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
        "changes remain byte-identical."
    ),
    "duration_s": "Use measured `aggregation_from_external_primary_sources`.",
    "inference_substrate": "Use measured `aggregation_from_external_primary_sources`.",
    "field_provenance": "Use measured `aggregation_from_external_primary_sources`.",
    "test_commands": "Use measured `aggregation_from_external_primary_sources`.",
    "test_exit_codes": "Use measured `aggregation_from_external_primary_sources`.",
    "reproducibility_checksum": (
        "Use measured `aggregation_from_external_primary_sources`."
    ),
    "honest_verdict": "Use `complete_delta:`, `complete_null:`, or `blocked:`.",
}

FIELD_PRINCIPLE_EXTRAS: dict[str, str] = {
    "schema": "Versioned schema id keeps downstream validators from guessing field meaning.",
    "experiment": "Stable local slug ties the artifact to the implementation module.",
    "experiment_id": "Conductor task identity prevents numeric-prefix aliasing.",
    "milestone": "Binds receipts to .528 rather than a prior milestone.",
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
        "receipt_id": "arxiv_v528_date_window",
        "source_family": "arXiv",
        "source_role": "primary",
        "query_family": "arxiv_primary",
        "query": "submittedDate:[202607270000 TO 202608032359]",
        "url": (
            "https://export.arxiv.org/api/query?search_query=submittedDate:%5B"
            "202607270000%20TO%20202608032359%5D&start=0&max_results=5"
        ),
        "accessed_at": "2026-08-03T17:59:46Z",
        "access_outcome": "reachable_http_200_totalResults_5561",
        "candidate_ids": [
            "2607.29687",
            "2607.29686",
            "2607.29685",
            "2607.29684",
            "2607.29683",
        ],
        "candidate_count": 5561,
        "source_cutoff": "submitted_or_changed_after_v528_marker_date_2026_07_26",
        "receipt_summary": (
            "The broad arXiv date window was reachable; topical and citation "
            "filters were needed before any candidate could be source-classified."
        ),
    },
    {
        "receipt_id": "arxiv_v528_topic_windows",
        "source_family": "arXiv",
        "source_role": "primary",
        "query_family": "arxiv_primary_topic_windows",
        "query": (
            "post-marker topic windows for EBM verification/reasoning, neural CSP, "
            "p-bit hardware, hallucination/internal representations, KANs, "
            "energy-guided generation, continual learning, and ARC online discovery"
        ),
        "url": "https://export.arxiv.org/api/query",
        "accessed_at": "2026-08-03T17:59:46Z",
        "access_outcome": "reachable_http_200_topic_routes_candidate_count_6",
        "candidate_ids": [
            "2607.28674",
            "2607.23802",
            "2607.27201",
            "2607.27888",
            "2607.29592",
            "2607.27372",
        ],
        "candidate_count": 6,
        "source_cutoff": "submitted_or_changed_after_v528_marker_primary_pages_opened",
        "receipt_summary": (
            "Six topical or citation-led arXiv candidates were opened and "
            "classified; none created an accepted V528 task delta."
        ),
    },
    {
        "receipt_id": "openreview_v528_search_pages",
        "source_family": "OpenReview",
        "source_role": "secondary",
        "query_family": "openreview_secondary",
        "query": (
            "energy-based constraint reasoning continual learning; UNLS mixed "
            "integer Langevin sampler"
        ),
        "url": (
            "https://openreview.net/search?term=energy-based%20constraint%20"
            "reasoning%20continual%20learning"
        ),
        "accessed_at": "2026-08-03T18:01:14Z",
        "access_outcome": "reachable_http_200_dynamic_search_pages_no_new_primary_delta",
        "candidate_ids": [],
        "candidate_count": 0,
        "source_cutoff": "dynamic_search_checked_after_v528_marker_primary_date_uncertain",
        "receipt_summary": (
            "OpenReview search pages loaded, but dynamic search output was not "
            "promoted without a newer reachable forum page."
        ),
    },
    {
        "receipt_id": "openreview_v528_api_notes",
        "source_family": "OpenReview",
        "source_role": "secondary",
        "query_family": "openreview_api",
        "query": "api2.openreview.net notes content.title=energy-based",
        "url": "https://api2.openreview.net/notes?limit=5&content.title=energy-based",
        "accessed_at": "2026-08-03T18:01:14Z",
        "access_outcome": "inaccessible_http_403_challenge_required",
        "candidate_ids": [],
        "candidate_count": 0,
        "source_cutoff": "api_checked_after_v528_marker",
        "receipt_summary": (
            "The OpenReview API returned ChallengeRequiredError; this is an "
            "endpoint failure, not negative evidence."
        ),
    },
    {
        "receipt_id": "huggingface_papers_v528_2026_08_03",
        "source_family": "Hugging Face Papers",
        "source_role": "secondary",
        "query_family": "huggingface_papers_secondary",
        "query": "daily_papers date:2026-08-03",
        "url": "https://huggingface.co/papers",
        "accessed_at": "2026-08-03T18:00:00Z",
        "access_outcome": "reachable_http_200_daily_feed_aug_3_three_candidates_opened",
        "candidate_ids": ["2607.23802", "2607.27201", "2607.27888"],
        "candidate_count": 3,
        "source_cutoff": "secondary_daily_feed_aug_3_primary_pages_opened_for_candidates",
        "receipt_summary": (
            "The Aug. 3 Hugging Face daily feed was reachable and pointed to "
            "three opened arXiv pages; all remained rejected, cutoff-confounded, "
            "or retired-scope context."
        ),
    },
    {
        "receipt_id": "semantic_scholar_v528_ebt_citations",
        "source_family": "Semantic Scholar",
        "source_role": "secondary",
        "query_family": "semantic_scholar_citation_trail",
        "query": "arXiv:2507.02092 citations",
        "url": (
            "https://api.semanticscholar.org/graph/v1/paper/arXiv:2507.02092/"
            "citations?fields=title,year,externalIds,url,publicationDate,authors&limit=100"
        ),
        "accessed_at": "2026-08-03T18:00:46Z",
        "access_outcome": "reachable_http_200_31_records_one_post_marker_citation",
        "candidate_ids": ["2607.27372", "2607.20792", "2607.17047"],
        "candidate_count": 31,
        "source_cutoff": "citation_trail_checked_after_v528_marker_newest_2026_07_29",
        "newest_visible": {
            "identifier": "2607.27372",
            "title": "Explorative Modeling: Unlocking a Third Pretraining Axis and End-to-End Generation",
            "publication_date": "2026-07-29",
        },
        "receipt_summary": (
            "The EBT citation route exposed one new post-marker citation, "
            "arXiv:2607.27372, opened as primary but rejected as not a V528 task hook."
        ),
    },
    {
        "receipt_id": "semantic_scholar_v528_arm_ebm_citations",
        "source_family": "Semantic Scholar",
        "source_role": "secondary",
        "query_family": "semantic_scholar_citation_trail",
        "query": "arXiv:2512.15605 citations",
        "url": (
            "https://api.semanticscholar.org/graph/v1/paper/arXiv:2512.15605/"
            "citations?fields=title,year,externalIds,url,publicationDate,authors&limit=100"
        ),
        "accessed_at": "2026-08-03T18:00:46Z",
        "access_outcome": "reachable_http_200_8_records_no_post_marker_citation",
        "candidate_ids": ["2607.02154", "2606.03089", "2605.18871"],
        "candidate_count": 8,
        "source_cutoff": "citation_trail_checked_after_v528_marker_newest_2026_07_02",
        "newest_visible": {
            "identifier": "2607.02154",
            "title": "Path-Measure Dynamics of Attention-Driven World Models",
            "publication_date": "2026-07-02",
        },
        "receipt_summary": (
            "The ARM-EBM citation route remained unchanged after the V528 marker."
        ),
    },
    {
        "receipt_id": "github_v528_targeted_and_trending",
        "source_family": "GitHub",
        "source_role": "secondary",
        "query_family": "github_targeted_secondary",
        "query": (
            "GitHub trending/python daily plus six repository searches pushed "
            "after 2026-07-26 for EBM, exact atom compatibility, SARE, SpyRL, "
            "Extropic TSU, and Kona"
        ),
        "url": "https://api.github.com/search/repositories",
        "accessed_at": "2026-08-03T18:00:30Z",
        "access_outcome": "reachable_http_200_six_targeted_routes_total_count_0",
        "candidate_ids": [],
        "candidate_count": 0,
        "source_cutoff": "pushed_after_2026_07_26_secondary_metadata",
        "receipt_summary": (
            "GitHub Trending and targeted repository search exposed no maintained "
            "post-marker dependency delta for Carnot."
        ),
    },
    {
        "receipt_id": "extropic_v528_official_pages",
        "source_family": "Extropic",
        "source_role": "official",
        "query_family": "official_project_page",
        "query": "Extropic writing, hardware, and July 29 Z1/TSU letter-of-intent update",
        "url": (
            "https://extropic.ai/writing/thermodynamic-computing-chips-in-america ; "
            "https://extropic.ai/hardware"
        ),
        "accessed_at": "2026-08-03T18:00:00Z",
        "access_outcome": "reachable_http_200_official_z1_loi_no_authenticated_local_route",
        "candidate_ids": ["extropic_z1_loi_2026_07_29", "z1_hardware_page_2026"],
        "candidate_count": 2,
        "source_cutoff": "official_pages_checked_after_v528_marker_latest_2026_07_29",
        "receipt_summary": (
            "Extropic published a post-marker official Z1 funding/access update, "
            "but it does not provide Carnot-local TSU access, SDK, timing, power, "
            "or correctness receipts."
        ),
    },
    {
        "receipt_id": "logical_intelligence_v528_official_pages",
        "source_family": "Logical Intelligence",
        "source_role": "official",
        "query_family": "official_project_page",
        "query": "Logical Intelligence root, Kona, Aleph, and formal-verification pages",
        "url": (
            "https://logicalintelligence.com/ ; "
            "https://logicalintelligence.com/blog/energy-based-models-for-reasoning"
        ),
        "accessed_at": "2026-08-03T18:00:00Z",
        "access_outcome": "reachable_http_200_official_context_no_public_kona_weights",
        "candidate_ids": [
            "kona_1_0_public_context",
            "aleph_public_context",
            "formal_verification_blog",
        ],
        "candidate_count": 3,
        "source_cutoff": "official_pages_checked_after_v528_marker_no_new_local_route",
        "receipt_summary": (
            "Logical Intelligence pages remain official architecture context; "
            "public Kona weights, documented local API, and reproducible comparator "
            "remain unavailable."
        ),
    },
    {
        "receipt_id": "local_sweep_clusters_v528",
        "source_family": "local sweep helper",
        "source_role": "tooling",
        "query_family": "local_tooling",
        "query": "scripts/sweep_clusters.py all --max-results 3",
        "url": "scripts/sweep_clusters.py",
        "accessed_at": "2026-08-03T18:01:40Z",
        "access_outcome": "reachable_local_tool_exit_0_emitted_7_arxiv_urls",
        "candidate_ids": [],
        "candidate_count": 0,
        "source_cutoff": "tooling_urls_emitted_after_v528_marker",
        "receipt_summary": (
            "The local arXiv cluster helper emitted seven broadened query URLs "
            "and did not mutate repository files."
        ),
    },
    {
        "receipt_id": "local_sweep_semscholar_v528",
        "source_family": "local sweep helper",
        "source_role": "tooling",
        "query_family": "local_tooling",
        "query": (
            "step-aware reasoning energy hidden states; energy based exact atom "
            "compatibility; RLSVR self-verifiable rewards"
        ),
        "url": "scripts/sweep_semscholar.py",
        "accessed_at": "2026-08-03T18:02:10Z",
        "access_outcome": "inaccessible_remote_http_429_rate_limited_on_3_keyword_queries",
        "candidate_ids": [],
        "candidate_count": 0,
        "source_cutoff": "keyword_queries_after_v528_marker",
        "receipt_summary": (
            "The local Semantic Scholar keyword helper hit HTTP 429 on all three "
            "focused keyword queries; direct citation APIs remained reachable."
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
    publication_date: str = "2026-08-03",
    source_date: str = "2026-08-03",
    search_timestamp: str = "2026-08-03T18:01:30Z",
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
        "sare_hidden_state_no_current_backend_delta",
        "rejected",
        "How Hard Does It Think? Analyzing Step-Aware Reasoning Energy in LLM Chain-of-Thought Trajectories",
        "https://arxiv.org/abs/2607.28674",
        identifier="2607.28674",
        authors=["Hui Wei", "Junda Wu", "Julian McAuley"],
        publication_date="2026-07-28",
        source_date="2026-07-28",
        receipt_id="arxiv_v528_topic_windows",
        query_family="arxiv_primary_topic_windows",
        access_outcome="reachable_primary_arxiv_post_marker",
        reason=(
            "SARE is a post-marker primary internal-representation paper, but "
            "its CKA-over-intermediate-hidden-states mechanism is not available "
            "through Carnot's current GGUF compatibility path and does not create "
            "a bounded exact-atom task delta."
        ),
    ),
    _finding(
        "explorative_modeling_not_v528_task_hook",
        "rejected",
        "Explorative Modeling: Unlocking a Third Pretraining Axis and End-to-End Generation",
        "https://arxiv.org/abs/2607.27372",
        identifier="2607.27372",
        authors=["Alexi Gladstone", "Heng Ji", "Yilun Du"],
        publication_date="2026-07-29",
        source_date="2026-07-29",
        receipt_id="semantic_scholar_v528_ebt_citations",
        query_family="semantic_scholar_citation_trail",
        access_outcome="reachable_primary_arxiv_post_marker_via_ebt_citation",
        reason=(
            "The new EBT citation concerns generative pretraining and "
            "end-to-end generation, not deterministic exact-atom compatibility, "
            "delayed commit memory, or ARC budget/convention measurements."
        ),
    ),
    _finding(
        "mental_world_modeling_not_arc_online_delta",
        "rejected",
        "Mental World Modeling",
        "https://arxiv.org/abs/2607.27201",
        identifier="2607.27201",
        authors=["Hao Fei", "Yiran Zhao"],
        publication_date="2026-07-29",
        source_date="2026-07-29",
        receipt_id="huggingface_papers_v528_2026_08_03",
        query_family="huggingface_papers_secondary",
        access_outcome="reachable_primary_arxiv_from_hf_secondary",
        reason=(
            "The paper is world-model context but does not provide an ARC-AGI "
            "online discovery mechanism, action-budget gate, or local exact "
            "transition verifier for the active V528 roadmap."
        ),
    ),
    _finding(
        "tood_continual_learning_not_transactional_memory_delta",
        "rejected",
        "TOOD: Task-Aware Out-of-Distribution Score Calibration for Continual Learners",
        "https://arxiv.org/abs/2607.29592",
        identifier="2607.29592",
        authors=["Mostafa ElAraby", "Samer B. Nashed", "Liam Paull"],
        publication_date="2026-07-31",
        source_date="2026-07-31",
        receipt_id="arxiv_v528_topic_windows",
        query_family="arxiv_primary_topic_windows",
        access_outcome="reachable_primary_arxiv_post_marker",
        reason=(
            "TOOD is relevant continual-learning OOD calibration context, but "
            "V528's learning tasks are delayed transactional commit, poison "
            "quarantine, rollback, and immutable-weight memory rather than "
            "image-classification OOD score recalibration."
        ),
    ),
    _finding(
        "extropic_z1_loi_no_local_execution_route",
        "rejected",
        "Extropic Signs $75 Million Letter of Intent with U.S. Department of Commerce",
        "https://extropic.ai/writing/thermodynamic-computing-chips-in-america",
        identifier="extropic_z1_loi_2026_07_29",
        authors=["Extropic Corporation"],
        publication_date="2026-07-29",
        source_date="2026-07-29",
        receipt_id="extropic_v528_official_pages",
        query_family="official_project_page",
        access_outcome="reachable_http_200_official_z1_loi_no_authenticated_local_route",
        reason=(
            "The official Z1 update is materially changed hardware context, but "
            "it does not provide Carnot-local TSU access, SDK, timing, power, or "
            "correctness evidence."
        ),
    ),
    _finding(
        "github_targeted_zero_no_dependency_delta",
        "rejected",
        "GitHub targeted searches without a maintained method delta",
        "https://api.github.com/search/repositories",
        identifier="github_targeted_zero_delta",
        receipt_id="github_v528_targeted_and_trending",
        query_family="github_targeted_secondary",
        access_outcome="reachable_http_200_six_targeted_routes_total_count_0",
        reason=(
            "Repository discovery metadata exposed no maintained post-marker "
            "dependency that sharpens an allocated V528 control."
        ),
    ),
)

DEFAULT_ABSTAINED_FINDINGS: tuple[JsonDict, ...] = (
    _finding(
        "openreview_dynamic_search_date_uncertain",
        "abstained",
        "OpenReview dynamic search pages with uncertain primary date",
        "https://openreview.net/search?term=energy-based%20constraint%20reasoning",
        identifier="openreview_dynamic_date_uncertain_post_v528",
        receipt_id="openreview_v528_search_pages",
        query_family="openreview_secondary",
        access_outcome="reachable_http_200_dynamic_search_pages_no_new_primary_delta",
        reason=(
            "Dynamic search pages can crawl recently without proving a primary "
            "forum publication or material change after the marker."
        ),
    ),
    _finding(
        "logical_intelligence_context_no_local_api",
        "abstained",
        "Logical Intelligence Kona/Aleph official pages without local API",
        "https://logicalintelligence.com/",
        identifier="logical_intelligence_kona_no_public_route",
        receipt_id="logical_intelligence_v528_official_pages",
        query_family="official_project_page",
        access_outcome="reachable_http_200_official_context_no_public_kona_weights",
        reason=(
            "Official Kona/Aleph context remains architecture evidence, but no "
            "public weights, documented local inference API, or reproducible "
            "comparator are available to turn it into a V528 source delta."
        ),
    ),
)

DEFAULT_FALSE_POSITIVE_FINDINGS: tuple[JsonDict, ...] = (
    _finding(
        "hf_aug_3_secondary_promotion_false_positive",
        "false_positive",
        "Hugging Face Aug. 3 ranking did not prove post-marker primary novelty",
        "https://huggingface.co/papers",
        identifier="hf_aug_3_secondary_date_only",
        receipt_id="huggingface_papers_v528_2026_08_03",
        query_family="huggingface_papers_secondary",
        access_outcome="reachable_http_200_daily_feed_aug_3_three_candidates_opened",
        reason=(
            "The daily ranking date is secondary discovery metadata; candidate "
            "acceptance still depends on each opened primary arXiv page."
        ),
    ),
)

DEFAULT_KNOWN_FALSE_NEGATIVE_FINDINGS: tuple[JsonDict, ...] = (
    _finding(
        "semantic_scholar_keyword_rate_limit_possible_miss",
        "known_false_negative",
        "Semantic Scholar keyword helper rate-limit possible miss",
        "scripts/sweep_semscholar.py",
        identifier="s2_keyword_rate_limit_possible_miss_post_v528",
        receipt_id="local_sweep_semscholar_v528",
        query_family="local_tooling",
        access_outcome="inaccessible_remote_http_429_rate_limited_on_3_keyword_queries",
        reason=(
            "The keyword helper returned HTTP 429 for all three focused queries; "
            "a relevance-ranked miss is recorded explicitly instead of treated "
            "as no evidence."
        ),
    ),
)

DEFAULT_CUTOFF_CONFOUND_FINDINGS: tuple[JsonDict, ...] = (
    _finding(
        "same_day_marker_cutoff_confound",
        "cutoff_confound",
        "Same-day V528 marker cutoff confound",
        "research-references.md#v528-planner-refresh---20260726",
        identifier="v528_same_day_marker_cutoff_confound",
        publication_date="2026-07-26",
        source_date="2026-07-26",
        receipt_id="huggingface_papers_v528_2026_08_03",
        query_family="huggingface_papers_secondary",
        access_outcome="cutoff_confound_preserved",
        reason=(
            "RLSVR v1 was submitted on the same calendar day as the exact local "
            "marker; date-only discovery cannot prove marker ordering even though "
            "v2 was later opened and rejected as a roadmap mismatch."
        ),
    ),
)

DEFAULT_ENDPOINT_FAILED_FINDINGS: tuple[JsonDict, ...] = (
    _finding(
        "openreview_api_challenge_endpoint_failed",
        "endpoint_failed",
        "OpenReview notes API challenge gate",
        "https://api2.openreview.net/notes?limit=5&content.title=energy-based",
        identifier="openreview_api_challenge_v528",
        receipt_id="openreview_v528_api_notes",
        query_family="openreview_api",
        access_outcome="inaccessible_http_403_challenge_required",
        publication_date="unknown",
        source_date="unknown",
        reason=(
            "The API route required challenge verification; no result is fabricated "
            "and the failure remains separate from rejection."
        ),
    ),
)

DEFAULT_DUPLICATE_FINDINGS: tuple[JsonDict, ...] = (
    _finding(
        "v528_planner_hide_duplicate",
        "duplicate",
        "HIDE and Seek: Detecting Hallucinations in Language Models via Decoupled Representations",
        "https://arxiv.org/abs/2506.17748",
        identifier="2506.17748",
        authors=["V528 planner source"],
        publication_date="2025-06-17",
        source_date="2025-06-17",
        receipt_id="arxiv_v528_topic_windows",
        query_family="arxiv_primary_topic_windows",
        access_outcome="duplicate_existing_v528_reference_heading",
        reason=(
            "Already accepted in the sealed V528 planner block as the exact-atom "
            "representation compatibility hook."
        ),
    ),
    _finding(
        "v528_planner_ltla_duplicate",
        "duplicate",
        "Learning Tractable Distributions of Language Model Continuations",
        "https://arxiv.org/abs/2511.16054",
        identifier="2511.16054",
        authors=["V528 planner source"],
        publication_date="2025-11-16",
        source_date="2025-11-16",
        receipt_id="arxiv_v528_topic_windows",
        query_family="arxiv_primary_topic_windows",
        access_outcome="duplicate_existing_v528_reference_heading",
        reason="Already recorded in the sealed V528 planner block as guarded LTLA context.",
    ),
    _finding(
        "v528_planner_pal_duplicate",
        "duplicate",
        "A Probabilistic Neuro-symbolic Layer for Algebraic Constraint Satisfaction",
        "https://arxiv.org/abs/2503.19466",
        identifier="2503.19466",
        authors=["V528 planner source"],
        publication_date="2025-03-19",
        source_date="2025-03-19",
        receipt_id="arxiv_v528_topic_windows",
        query_family="arxiv_primary_topic_windows",
        access_outcome="duplicate_existing_v528_reference_heading",
        reason="Already recorded in the sealed V528 planner block as guarded PAL context.",
    ),
)

DEFAULT_RETIRED_SCOPE_FINDINGS: tuple[JsonDict, ...] = (
    _finding(
        "generated_ir_schema_reprompt_reopen_post_v528",
        "retired_scope",
        "Generated-IR or schema-reprompt reopen request",
        "ops/exclusion_manifest.yaml",
        identifier="generated_ir_schema_reprompt_reopen",
        receipt_id="arxiv_v528_topic_windows",
        query_family="arxiv_primary_topic_windows",
        access_outcome="retired_scope_excluded_by_manifest",
        reason=(
            "Generated-IR and schema-reprompt mechanisms remain closed; HIDE is "
            "only carried forward as discriminative exact-atom compatibility."
        ),
    ),
    _finding(
        "finite_id_external_logprob_final_embedding_reopen_post_v528",
        "retired_scope",
        "Finite-ID, external-text/logprob, or final-embedding MMLU reopen request",
        "ops/exclusion_manifest.yaml",
        identifier="finite_id_external_logprob_final_embedding_reopen",
        receipt_id="huggingface_papers_v528_2026_08_03",
        query_family="huggingface_papers_secondary",
        access_outcome="retired_scope_excluded_by_manifest",
        reason=(
            "Finite-ID transport, external generated-text/logprob scoring, and "
            "final-embedding MMLU scoring remain closed."
        ),
    ),
    _finding(
        "kan_public_arc_board_probe_reopen_post_v528",
        "retired_scope",
        "KAN mutation, public ARC solve, or unchanged board-probe reopen request",
        "ops/exclusion_manifest.yaml",
        identifier="kan_public_arc_board_probe_reopen",
        receipt_id="github_v528_targeted_and_trending",
        query_family="github_targeted_secondary",
        access_outcome="retired_scope_excluded_by_manifest",
        reason=(
            "KAN mutation, public ARC solves, and unchanged board probes remain "
            "closed for source acceptance."
        ),
    ),
    _finding(
        "broad_grpo_rlvr_reopen_post_v528",
        "retired_scope",
        "Broad GRPO/RLVR source rerun request",
        "https://arxiv.org/abs/2607.27888",
        identifier="2607.27888",
        authors=["Qiangqiang He", "Zhongheng Wu", "ZiJian Wang"],
        publication_date="2026-07-30",
        source_date="2026-07-30",
        receipt_id="huggingface_papers_v528_2026_08_03",
        query_family="huggingface_papers_secondary",
        access_outcome="reachable_primary_arxiv_from_hf_secondary",
        reason=(
            "CSCR is a GRPO/RLVR credit-assignment extension; V528 does not "
            "reopen broad policy-gradient or generated-answer RL lineages."
        ),
    ),
)

DEFAULT_TESTS_RUN: tuple[JsonDict, ...] = (
    {
        "command": (
            ".venv/bin/pytest tests/python/"
            "test_experiment_5962_v528_source_delta_ingestion.py -q --no-cov -n 0"
        ),
        "exit_code": None,
    },
    {
        "command": (
            ".venv/bin/coverage run --rcfile=/dev/null --include="
            "python/carnot/experiment_5962_v528_source_delta_ingestion.py -m pytest "
            "tests/python/test_experiment_5962_v528_source_delta_ingestion.py "
            "-q --no-cov -n 0"
        ),
        "exit_code": None,
    },
    {
        "command": (
            ".venv/bin/coverage report --rcfile=/dev/null --include="
            "python/carnot/experiment_5962_v528_source_delta_ingestion.py "
            "--fail-under=100"
        ),
        "exit_code": None,
    },
    {
        "command": (
            ".venv/bin/ruff check "
            "python/carnot/experiment_5962_v528_source_delta_ingestion.py "
            "tests/python/test_experiment_5962_v528_source_delta_ingestion.py"
        ),
        "exit_code": None,
    },
    {
        "command": (
            ".venv/bin/python scripts/adversarial_verify.py --json "
            "results/experiment_5962_v528_source_delta_ingestion.json"
        ),
        "exit_code": None,
    },
    {"command": ".venv/bin/python scripts/check_spec_coverage.py", "exit_code": None},
    {"command": ".venv/bin/python scripts/root_clutter_sweep.py", "exit_code": None},
    {"command": ".venv/bin/pytest tests/python -q", "exit_code": None},
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


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
    """Return the one-based line number for the sealed V528 marker."""

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
    if "REQ-REPORT-5962" not in spec_text:
        failures.append("spec_req_report_5962_missing")
    if not resources["output_parent_writable"]:
        failures.append("output_path_unavailable")
    return {
        "checked_at": normalize_timestamp(checked_at or datetime.now(UTC).isoformat()),
        "planner_marker_found": marker_found,
        "references_hash": path_sha256(root / RESEARCH_REFERENCES_RELATIVE_PATH),
        "v528_marker_hash": planner_block_hash(references_text),
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
                str(row.get("source_cutoff", "published_or_changed_after_v528_marker"))
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
    """Record the sealed V528 marker and UTC post-marker search interval."""

    return {
        "boundary_marker": PLANNER_MARKER,
        "boundary_heading": PLANNER_HEADING,
        "boundary_line": planner_marker_line(references_text),
        "boundary_hash": planner_block_hash(references_text),
        "search_window_start_utc": normalize_timestamp(search_started_at),
        "search_window_end_utc": normalize_timestamp(search_finished_at),
        "novelty_rule": (
            "accept only primary-source evidence published or materially changed "
            "after the V528 marker that supplies a new mechanism or material "
            "change for already allocated V528 tasks"
        ),
    }


def source_queries_and_endpoint_receipts(source_receipts: Sequence[JsonDict]) -> JsonDict:
    """Group source receipts with endpoint-failure and rate-limit summaries."""

    normalized_receipts = []
    for row in source_receipts:
        receipt = dict(row)
        receipt.setdefault("source_cutoff", "published_or_changed_after_v528_marker")
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


def openreview_huggingface_github_extropic_and_kona_receipts(
    source_receipts: Sequence[JsonDict],
) -> JsonDict:
    """Group official and secondary routes that are context until primary opened."""

    def by_family(*families: str) -> list[JsonDict]:
        wanted = set(families)
        return [
            dict(row) for row in source_receipts if str(row.get("source_family")) in wanted
        ]

    return {
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "openreview_huggingface_github_extropic_and_kona_receipts"
        ],
        "openreview_receipts": by_family("OpenReview"),
        "huggingface_receipts": by_family("Hugging Face Papers"),
        "github_receipts": by_family("GitHub"),
        "extropic_receipts": by_family("Extropic"),
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
            "generated-IR",
            "schema-reprompt",
            "finite-ID",
            "external-text/logprob scorer",
            "final-embedding MMLU",
            "KAN mutation",
            "public ARC solve",
            "unchanged board-probe",
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
            "Execution-time sweep on 2026-08-03 after the V528 planner marker. "
            "Only non-duplicate primary-source deltas that sharpen existing "
            ".528 controls are listed here."
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
        "prior_v528_marker_preserved": True,
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
        return "blocked: V528 source refresh precondition failed"
    if not source_reachable:
        return "blocked: no primary, official, or reliable secondary source route reachable"
    if accepted_findings:
        return (
            f"complete_delta: accepted {len(accepted_findings)} bounded "
            "post-V528 source delta(s); task identities, gates, and exclusions unchanged"
        )
    return "complete_null: no accepted post-V528 source deltas; references unchanged"


def _field_provenance(accepted_findings: Sequence[JsonDict]) -> JsonDict:
    provenance: JsonDict = {
        field: {
            "principle": REQUIRED_FIELD_PRINCIPLES[field],
            "source": (
                "Exp5962 source receipts, local hashes, query families, cutoffs, "
                "or classification records"
            ),
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
    """Build the Exp5962 artifact from local hashes and source receipts."""

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
        "openreview_huggingface_github_extropic_and_kona_receipts": (
            openreview_huggingface_github_extropic_and_kona_receipts(source_receipts)
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
    result_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def _finding_is_post_marker(row: Mapping[str, Any]) -> bool:
    return str(row.get("source_date", "")) > MARKER_DATE or bool(
        row.get("materially_changed_after_marker")
    )


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
        _require(field in row, f"finding provenance field missing {field}")
    _require(row["classification"] == expected_classification, "invalid finding classification")
    _require(row["decision_bucket"] == expected_classification, "invalid finding decision bucket")
    if expected_classification != "accepted":
        return
    _require(
        bool(row.get("post_marker_or_newer_primary_source")) and _finding_is_post_marker(row),
        "accepted finding must be newer primary-source evidence",
    )
    _require(bool(row.get("primary_source")), "accepted finding must cite primary-source evidence")
    _require(
        not row.get("duplicate_of_existing_reference"),
        "accepted finding cannot be duplicate",
    )
    _require(not row.get("reopens_retired_scope"), "accepted finding cannot reopen retired scope")
    _require(
        bool(row.get("new_mechanism_or_material_change")),
        "accepted finding must provide a new mechanism or material change",
    )
    for field in ("target_experiment", "source_hook", "authority_boundary"):
        _require(bool(row.get(field)), f"accepted finding missing {field}")
    _require(
        row["target_experiment"] in ALLOCATED_TARGET_EXPERIMENTS,
        "accepted finding must target an allocated .528 experiment",
    )
    mapping = row.get("method_to_task_mapping")
    _require(isinstance(mapping, Mapping), "accepted finding missing method-to-task mapping")
    _require(
        mapping.get("target_experiment") == row["target_experiment"],
        "accepted finding method-to-task mapping target mismatch",
    )
    for field in ("method", "task_hook", "failure_boundary"):
        _require(
            bool(mapping.get(field)),
            f"accepted finding method-to-task mapping missing {field}",
        )


def _validate_source_receipts(artifact: Mapping[str, Any]) -> None:
    source_queries = artifact.get("source_queries_and_endpoint_receipts")
    _require(isinstance(source_queries, Mapping), "source_queries_and_endpoint_receipts must be a mapping")
    source_receipts = source_queries.get("source_receipts")
    _require(
        isinstance(source_receipts, list) and bool(source_receipts),
        "source_queries_and_endpoint_receipts source_receipts missing",
    )
    for row in source_receipts:
        _require(isinstance(row, Mapping), "source receipt entries must be mappings")
        for field in SOURCE_RECEIPT_REQUIRED_FIELDS:
            _require(field in row, f"source receipt missing {field}")
    _require(isinstance(source_queries.get("endpoint_failures"), list), "source_queries endpoint failures missing")
    _require(isinstance(source_queries.get("rate_limits"), list), "source_queries rate limits missing")


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
    _require(isinstance(classification, Mapping), "accepted_rejected_abstained_findings must be a mapping")
    for key in expected:
        _require(isinstance(classification.get(key), list), f"accepted_rejected_abstained_findings.{key} must be a list")
        for row in classification[key]:
            _require(isinstance(row, Mapping), "finding classification entries must be mappings")
            _validate_finding(row, key)
    ordered: list[JsonDict] = []
    for key in expected:
        ordered.extend(classification[key])
    _require(classification.get("all_candidates") == ordered, "all_candidates must preserve classification order")
    return dict(classification)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp5962 artifact schema and anti-laundering invariants."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, f"missing required field {field}")
    _require(artifact["status"] in {"complete", "blocked"}, "invalid status")
    _require(
        str(artifact["honest_verdict"]).startswith(
            ("complete_delta:", "complete_null:", "blocked:")
        ),
        "honest_verdict must use a terminal source prefix",
    )
    _require(
        artifact["inference_substrate"] == INFERENCE_SUBSTRATE,
        "inference substrate must be external source aggregation",
    )
    _require(float(artifact["duration_s"]) >= 0, "duration must be non-negative")
    _require(
        _parse_timestamp(str(artifact["search_finished_at"]))
        > _parse_timestamp(str(artifact["search_started_at"])),
        "timestamp window must be positive",
    )
    _validate_source_receipts(artifact)
    classification = _validate_classification(artifact)

    counts = artifact.get("primary_secondary_and_official_source_counts")
    _require(
        isinstance(counts, Mapping)
        and all(key in counts for key in ("primary", "secondary", "official", "tooling")),
        "source counts missing required roles",
    )
    if artifact["status"] == "complete":
        _require(
            all(int(counts[key]) >= 1 for key in ("primary", "secondary", "official", "tooling")),
            "source counts must include primary, secondary, official, tooling",
        )

    references = artifact["references_append_receipt"]
    accepted = classification["accepted"]
    _require(
        references["accepted_count"] == len(accepted),
        "references append accepted count mismatch",
    )
    _require(
        references["accepted_source_ids"] == [row["source_id"] for row in accepted],
        "references append accepted source ids mismatch",
    )
    _require(
        bool(accepted) or not references["appended"],
        "zero accepted findings cannot append references",
    )

    provenance = artifact.get("field_provenance")
    _require(isinstance(provenance, Mapping), "field_provenance must be a mapping")
    for field, principle in REQUIRED_FIELD_PRINCIPLES.items():
        _require(field in provenance, f"field_provenance missing {field}")
        _require(
            provenance[field].get("principle") == principle,
            f"field_provenance principle mismatch for {field}",
        )

    uncertainty = artifact["false_positive_false_negative_cutoff_and_rate_limit_receipts"]
    _require(
        isinstance(uncertainty, Mapping)
        and uncertainty.get("principle")
        == REQUIRED_FIELD_PRINCIPLES[
            "false_positive_false_negative_cutoff_and_rate_limit_receipts"
        ],
        "false-positive/cutoff receipts malformed",
    )
    for key in (
        "false_positive_source_decisions",
        "known_false_negative_source_decisions",
        "cutoff_confounds",
        "endpoint_failed_source_decisions",
        "rate_limit_receipts",
    ):
        _require(isinstance(uncertainty.get(key), list), "false-positive/cutoff receipts missing list")

    semantic = artifact["semantic_scholar_ebt_and_arm_ebm_receipts"]
    _require(
        isinstance(semantic, Mapping)
        and (artifact["status"] != "complete" or bool(semantic.get("direct_api_receipts"))),
        "semantic scholar receipts missing direct API receipts",
    )
    grouped = artifact["openreview_huggingface_github_extropic_and_kona_receipts"]
    required_groups = (
        "openreview_receipts",
        "huggingface_receipts",
        "github_receipts",
        "extropic_receipts",
        "kona_or_aleph_receipts",
    )
    _require(
        isinstance(grouped, Mapping) and all(key in grouped for key in required_groups),
        "official/discovery receipts missing grouped routes",
    )
    if artifact["status"] == "complete":
        _require(
            all(grouped[key] for key in required_groups),
            "official/discovery receipts require each route group",
        )

    filters = artifact["duplicate_and_retired_scope_filter"]
    _require(isinstance(filters, Mapping), "duplicate_and_retired_scope_filter must be a mapping")
    _require(
        filters.get("accepted_reopens_retired_scope_count") == 0,
        "retired scope reopened by accepted finding",
    )

    immutability = artifact["task_identity_gate_and_exclusion_immutability"]
    _require(isinstance(immutability, Mapping), "task_identity_gate_and_exclusion_immutability must be a mapping")
    _require(bool(immutability.get("task_ids_unchanged")), "task ids changed")
    _require(bool(immutability.get("gates_unchanged")), "gates changed")
    _require(bool(immutability.get("exclusions_unchanged")), "exclusions changed")

    protected = artifact["protected_files_unchanged"]
    _require(
        isinstance(protected, Mapping) and bool(protected.get("all_unchanged")),
        "protected files changed",
    )
    _require(
        artifact["reproducibility_checksum"] == _compute_checksum(dict(artifact)),
        "checksum mismatch",
    )


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
        help="Confirm that no accepted post-V528 findings are being supplied.",
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
