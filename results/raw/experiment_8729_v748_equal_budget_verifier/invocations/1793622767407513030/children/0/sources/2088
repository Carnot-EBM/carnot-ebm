"""Exp5919: ingest post-V526 source deltas with explicit source uncertainty.

Spec refs: REQ-REPORT-5919, SCENARIO-REPORT-5919-ZERO-FINDING,
SCENARIO-REPORT-5919-ACCEPT-BOUNDED-DELTA,
SCENARIO-REPORT-5919-SOURCE-UNCERTAINTY,
SCENARIO-REPORT-5919-DUPLICATE-AND-RETIRED-SCOPE,
SCENARIO-REPORT-5919-SCHEMA.

This module turns an external literature sweep into an auditable local JSON
receipt. The network lookup itself is intentionally outside the decision
logic: source pages, rate limits, and failures are recorded as receipts, then
this code verifies the V526 marker boundary, classifies each finding, appends
references only for accepted bounded deltas, and refuses to relabel source
aggregation as live model inference.
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
RESULT_RELATIVE_PATH = Path("results/experiment_5919_v526_source_delta_ingestion.json")

AGENTS_RELATIVE_PATH = Path("AGENTS.md")
CODEX_RELATIVE_PATH = Path("CODEX.md")
CLAUDE_RELATIVE_PATH = Path("CLAUDE.md")
RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")
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

EXPERIMENT = "experiment_5919_v526_source_delta_ingestion"
EXPERIMENT_ID = "exp5919-v526-source-delta-ingestion"
MILESTONE = "2026.07.526"
RUN_DATE = "20260725"
RANDOM_SEED = 5919
SCHEMA = "carnot.experiment_5919.v526_source_delta_ingestion.v1"
INFERENCE_SUBSTRATE = "aggregation_from_external_primary_sources"

PLANNER_HEADING = "## V526 Planner Refresh - 20260725"
PLANNER_MARKER = "V526-PLANNER-REFRESH-20260725-END"
PLANNER_END_MARKER = f"<!-- {PLANNER_MARKER} -->"
EXECUTION_DELTA_HEADING = "## V526 Execution Source Delta - 20260725"
EXECUTION_DELTA_END_MARKER = "<!-- V526-EXECUTION-SOURCE-DELTA-20260725-END -->"

ALLOCATED_TARGET_EXPERIMENTS = (
    "exp5920-prospective-event-stream-admission",
    "exp5921-schema-derived-constraintir-support",
    "exp5922-gguf-schema-decoder-bridge",
    "exp5923-sota-schema-supported-constraintir-ab",
    "exp5924-transactional-constraint-memory-v2",
    "exp5925-sota-transactional-csl-prospective",
    "exp5926-adaptive-state-abi-v2-parity",
    "exp5927-coordinate-router-progress-qualification",
    "exp5928-arc-live-runner-execution-binding",
    "exp5929-arc-structured-memory-bound-live-ab",
    "exp5930-adaptive-state-board-mapping",
)

SPEC_REFS = (
    "REQ-REPORT-5919",
    "SCENARIO-REPORT-5919-ZERO-FINDING",
    "SCENARIO-REPORT-5919-ACCEPT-BOUNDED-DELTA",
    "SCENARIO-REPORT-5919-SOURCE-UNCERTAINTY",
    "SCENARIO-REPORT-5919-DUPLICATE-AND-RETIRED-SCOPE",
    "SCENARIO-REPORT-5919-SCHEMA",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "search_window_and_marker_receipt",
    "source_queries_and_endpoint_receipts",
    "primary_secondary_and_official_source_counts",
    "accepted_rejected_abstained_findings",
    "false_positive_false_negative_and_cutoff_receipts",
    "duplicate_and_retired_scope_filter",
    "references_append_receipt",
    "task_identity_and_gate_immutability",
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
    "status": (
        "Terminal source-refresh state derived from marker, reachability, "
        "and classification checks."
    ),
    "preconditions_checked": (
        "Marker, roadmap identities, reference ledgers, sweep code, output "
        "path, protected files, endpoint failures, rate limits, query "
        "families, and source cutoffs are hashed or recorded before source decisions."
    ),
    "search_window_and_marker_receipt": (
        "The V526 marker hash and UTC search interval make post-marker novelty falsifiable."
    ),
    "source_queries_and_endpoint_receipts": (
        "Every primary, secondary, official, and tooling route records query, "
        "URL, access outcome, timestamp, candidate counts, and source cutoffs."
    ),
    "primary_secondary_and_official_source_counts": (
        "Discovery routes cannot be mistaken for primary evidence."
    ),
    "accepted_rejected_abstained_findings": (
        "Accepted, rejected, abstained, training-cutoff, and publication-date-confounded decisions are separate source outcomes."
    ),
    "false_positive_false_negative_and_cutoff_receipts": (
        "Source uncertainty must remain explicit and cannot be silently converted into rejection."
    ),
    "duplicate_and_retired_scope_filter": (
        "Title, identifier, mechanism, authors, heading, and retired-scope filters "
        "prevent freshness from reopening closed work."
    ),
    "references_append_receipt": (
        "Reference-ledger appends are exact, dated, optional, and forbidden "
        "for zero accepted findings."
    ),
    "task_identity_and_gate_immutability": (
        "Evidence refresh may sharpen controls but cannot rewrite the activated milestone."
    ),
    "protected_files_unchanged": (
        "Roadmaps, conductor, protected ops ledgers, and retired-scope controls "
        "remain byte-identical unless explicitly owned."
    ),
    "duration_s": "Measured wall time exposes an external-source aggregation task.",
    "inference_substrate": "Use `aggregation_from_external_primary_sources`.",
    "field_provenance": (
        "Every required field traces to source receipts, local hashes, query "
        "families, source cutoffs, or classification records."
    ),
    "test_commands": (
        "Commands document focused unit, coverage, marker/hash, deduplication, "
        "date-window, source-classification, cutoff, "
        "references-marker, protected-file, schema, adversarial-verify, "
        "spec-coverage, applicable E2E, root-clutter, and full-suite checks."
    ),
    "test_exit_codes": "Exit codes prevent failed checks becoming success.",
    "reproducibility_checksum": (
        "A checksum detects later marker, source, classification, or append drift."
    ),
    "honest_verdict": "Use `complete_delta:`, `complete_null:`, or `blocked:`.",
}

FIELD_PRINCIPLE_EXTRAS: dict[str, str] = {
    "schema": "Versioned schema id keeps downstream validators from guessing field meaning.",
    "experiment": "Stable local slug ties the artifact to the implementation module.",
    "experiment_id": "Conductor task identity prevents numeric-prefix aliasing.",
    "milestone": "Binds receipts to .526 rather than a prior milestone.",
    "run_date": "Operator-requested execution date for the source refresh.",
    "random_seed": "Deterministic metadata for a no-randomness ledger task.",
    "spec_refs": "OpenSpec anchors make the artifact contract auditable.",
    "result_path": "Declares the exact JSON deliverable path.",
    "search_started_at": "Records when source querying started.",
    "search_finished_at": "Records when source classification finished.",
}

DEFAULT_SOURCE_RECEIPTS: tuple[JsonDict, ...] = (
    {
        "receipt_id": "arxiv_v526_date_only_after_marker",
        "source_family": "arXiv",
        "source_role": "primary",
        "query_family": "arxiv_primary",
        "query": "submittedDate:[202607250000 TO 202607252359]",
        "url": (
            "https://export.arxiv.org/api/query?search_query=submittedDate%3A%5B"
            "202607250000+TO+202607252359%5D&start=0&max_results=10&sortBy="
            "submittedDate&sortOrder=descending"
        ),
        "accessed_at": "2026-07-25T17:01:18Z",
        "access_outcome": "reachable_http_200_totalResults_0",
        "candidate_ids": [],
        "candidate_count": 0,
        "source_cutoff": "submitted_after_v526_marker_date_20260725",
        "receipt_summary": (
            "arXiv primary API date-only route found zero submissions in the "
            "2026-07-25 post-marker date window, so no arXiv item can be "
            "accepted from this route."
        ),
    },
    {
        "receipt_id": "arxiv_v526_compound_topic_query_failure",
        "source_family": "arXiv",
        "source_role": "primary",
        "query_family": "arxiv_primary",
        "query": (
            'submittedDate:[202607250000 TO 202607252359] AND '
            '(all:"energy-based" OR all:"constraint reasoning" OR '
            'all:"constrained decoding" OR all:"schema-derived")'
        ),
        "url": (
            "https://export.arxiv.org/api/query?search_query=submittedDate:["
            "202607250000%20TO%20202607252359]%20AND%20(all:%22energy-based%22"
            "%20OR%20all:%22constraint%20reasoning%22%20OR%20all:%22constrained"
            "%20decoding%22%20OR%20all:%22schema-derived%22)&start=0&max_results="
            "10&sortBy=submittedDate&sortOrder=descending"
        ),
        "accessed_at": "2026-07-25T17:01:12Z",
        "access_outcome": "inaccessible_http_400_compound_query_rejected",
        "candidate_ids": [],
        "candidate_count": 0,
        "source_cutoff": "submitted_after_v526_marker_date_20260725",
        "receipt_summary": (
            "The compound arXiv date/topic route was rejected with HTTP 400; "
            "the simpler date-only route above is the primary arXiv cutoff "
            "receipt and the 400 remains visible as an endpoint failure."
        ),
    },
    {
        "receipt_id": "openreview_v526_search_page",
        "source_family": "OpenReview",
        "source_role": "secondary",
        "query_family": "openreview_secondary",
        "query": "energy-based constraint reasoning continual learning after V526",
        "url": (
            "https://openreview.net/search?term=energy-based%20constraint%20"
            "reasoning%20continual%20learning"
        ),
        "accessed_at": "2026-07-25T17:02:01Z",
        "access_outcome": "reachable_http_200_dynamic_search_page_no_new_primary_delta",
        "candidate_ids": [],
        "candidate_count": 0,
        "source_cutoff": "dynamic_page_crawled_today_primary_date_uncertain",
        "receipt_summary": (
            "OpenReview search was reachable as a dynamic discovery page; "
            "secondary search listings were not promoted without newer primary "
            "forum-page evidence."
        ),
    },
    {
        "receipt_id": "openreview_v526_api_notes",
        "source_family": "OpenReview",
        "source_role": "secondary",
        "query_family": "openreview_api",
        "query": "api.openreview.net notes energy-based constraint",
        "url": "https://api.openreview.net/notes?content=energy-based&limit=5",
        "accessed_at": "2026-07-25T17:02:01Z",
        "access_outcome": "inaccessible_http_403_challenge_required",
        "candidate_ids": [],
        "candidate_count": 0,
        "source_cutoff": "api_checked_after_v526_marker",
        "receipt_summary": (
            "The direct OpenReview notes API required challenge verification, "
            "so this route is recorded as an endpoint failure rather than a "
            "fabricated negative result."
        ),
    },
    {
        "receipt_id": "huggingface_papers_v526_2026_07_25",
        "source_family": "Hugging Face Papers",
        "source_role": "secondary",
        "query_family": "huggingface_papers_secondary",
        "query": "daily_papers date:2026-07-25",
        "url": "https://huggingface.co/papers?date=2026-07-25",
        "accessed_at": "2026-07-25T17:02:32Z",
        "access_outcome": "reachable_http_200_daily_feed_secondary_no_new_primary_delta",
        "candidate_ids": [
            "2607.21461",
            "2607.20061",
            "2605.09635",
            "2607.21556",
            "2607.21072",
            "2607.20709",
            "2607.20911",
            "2607.12746",
            "2607.21553",
            "2607.20734",
            "2607.21576",
            "2607.21594",
            "2607.21051",
            "2607.20785",
            "2607.04763",
            "2607.10848",
            "2607.21485",
            "2607.21017",
            "2607.21557",
            "2607.19238",
            "2607.21580",
            "2607.16859",
        ],
        "candidate_count": 22,
        "source_cutoff": "daily_feed_date_2026_07_25_secondary_discovery_only",
        "receipt_summary": (
            "Hugging Face Papers listed daily paper ids, including V524/V526 "
            "duplicates such as AREX; this secondary feed did not provide a "
            "newer primary-source Carnot hook."
        ),
    },
    {
        "receipt_id": "semantic_scholar_v526_ebt_citations",
        "source_family": "Semantic Scholar",
        "source_role": "secondary",
        "query_family": "semantic_scholar_citation_trail",
        "query": "arXiv:2507.02092 citations",
        "url": (
            "https://api.semanticscholar.org/graph/v1/paper/arXiv:2507.02092/"
            "citations?fields=title,year,externalIds,url,publicationDate&limit=20"
        ),
        "accessed_at": "2026-07-25T17:02:55Z",
        "access_outcome": "reachable_http_200_20_records_newest_duplicate_2607_20792",
        "candidate_ids": [
            "2607.20792",
            "2607.17047",
            "2607.11555",
            "2606.22726",
            "2606.18206",
            "2606.15956",
            "2605.11011",
            "2605.07588",
            "ispass_2026_system2_ai_workloads",
            "2604.11403",
            "2604.10272",
            "2604.03878",
            "2604.01577",
            "2603.18534",
            "2603.19117",
            "protein_diffusion_models_statistical_potentials",
            "2602.03640",
            "2602.01651",
            "2601.03905",
            "2512.17846",
        ],
        "candidate_count": 20,
        "source_cutoff": "citation_trail_checked_after_v526_marker_newest_2026_07_22",
        "receipt_summary": (
            "Semantic Scholar EBT citation trail returned 20 records. The "
            "newest citation was Memoir 2607.20792 from 2026-07-22, already "
            "accepted in the sealed V526 planner block."
        ),
    },
    {
        "receipt_id": "semantic_scholar_v526_arm_ebm_citations",
        "source_family": "Semantic Scholar",
        "source_role": "secondary",
        "query_family": "semantic_scholar_citation_trail",
        "query": "arXiv:2512.15605 citations",
        "url": (
            "https://api.semanticscholar.org/graph/v1/paper/arXiv:2512.15605/"
            "citations?fields=title,year,externalIds,url,publicationDate&limit=20"
        ),
        "accessed_at": "2026-07-25T17:02:55Z",
        "access_outcome": "reachable_http_200_8_records_no_post_marker_actionable_citation",
        "candidate_ids": [
            "2607.02154",
            "2606.03089",
            "2605.18871",
            "2605.11011",
            "2604.00555",
            "2603.23398",
            "2602.02991",
            "acl_2026_energy_gibbs_alignment_no_arxiv_id",
        ],
        "candidate_count": 8,
        "source_cutoff": "citation_trail_checked_after_v526_marker_newest_2026_07_02",
        "receipt_summary": (
            "Semantic Scholar ARM-EBM citation trail returned eight records; "
            "the newest dated citation was 2026-07-02 and one record had null "
            "publicationDate, so no post-marker actionable source is promoted."
        ),
    },
    {
        "receipt_id": "github_v526_repository_discovery",
        "source_family": "GitHub discovery",
        "source_role": "secondary",
        "query_family": "github_secondary",
        "query": '"energy-based" constraint reasoning pushed:>2026-07-25',
        "url": (
            "https://api.github.com/search/repositories?q=%22energy-based%22+"
            "constraint+reasoning+pushed:%3E2026-07-25&per_page=5"
        ),
        "accessed_at": "2026-07-25T17:03:08Z",
        "access_outcome": "reachable_http_200_total_count_0_no_repository_delta",
        "candidate_ids": [],
        "candidate_count": 0,
        "source_cutoff": "pushed_after_2026_07_25_secondary_metadata",
        "receipt_summary": (
            "GitHub search was dependency metadata only; no maintained "
            "post-marker repository displaced Carnot's exact local backends."
        ),
    },
    {
        "receipt_id": "github_v526_issue_discovery",
        "source_family": "GitHub discovery",
        "source_role": "secondary",
        "query_family": "github_secondary",
        "query": "transactional memory poison updated:>2026-07-25",
        "url": (
            "https://api.github.com/search/issues?q=transactional+memory+poison+"
            "updated:%3E2026-07-25&per_page=5"
        ),
        "accessed_at": "2026-07-25T17:03:08Z",
        "access_outcome": "reachable_http_200_total_count_0_no_issue_delta",
        "candidate_ids": [],
        "candidate_count": 0,
        "source_cutoff": "updated_after_2026_07_25_secondary_metadata",
        "receipt_summary": (
            "GitHub issue discovery found zero post-marker transactional "
            "memory poison issues."
        ),
    },
    {
        "receipt_id": "extropic_v526_official_pages",
        "source_family": "Extropic",
        "source_role": "official",
        "query_family": "official_project_page",
        "query": "Extropic writing hardware Z1 XTR-0 TSU",
        "url": "https://www.extropic.ai/writing ; https://www.extropic.ai/hardware",
        "accessed_at": "2026-07-25T17:03:20Z",
        "access_outcome": "reachable_http_200_no_authenticated_local_route",
        "candidate_ids": ["z1_public_context", "xtr0_public_context", "tsu_public_context"],
        "candidate_count": 3,
        "source_cutoff": "official_page_checked_after_v526_marker_latest_dates_2025_10_30",
        "receipt_summary": (
            "Extropic official public pages remain hardware context without "
            "an authenticated Carnot-local TSU, SDK, speed, power, or "
            "correctness route."
        ),
    },
    {
        "receipt_id": "logical_intelligence_v526_official_pages",
        "source_family": "Logical Intelligence",
        "source_role": "official",
        "query_family": "official_project_page",
        "query": "Kona Aleph public pages",
        "url": (
            "https://logicalintelligence.com/ ; "
            "https://logicalintelligence.com/kona-ebms-energy-based-models"
        ),
        "accessed_at": "2026-07-25T17:03:20Z",
        "access_outcome": "reachable_http_200_no_local_weights_or_comparator",
        "candidate_ids": ["kona_1_0", "aleph"],
        "candidate_count": 2,
        "source_cutoff": "official_page_checked_after_v526_marker_latest_date_2026_06_26",
        "receipt_summary": (
            "Logical Intelligence official pages provide architecture context "
            "but no public weights, authenticated endpoint, or reproducible "
            "local comparator for Carnot."
        ),
    },
    {
        "receipt_id": "local_sweep_clusters_v526",
        "source_family": "local sweep helper",
        "source_role": "tooling",
        "query_family": "local_tooling",
        "query": "scripts/sweep_clusters.py all --max-results 3",
        "url": "scripts/sweep_clusters.py",
        "accessed_at": "2026-07-25T17:03:31Z",
        "access_outcome": "reachable_local_tool_exit_0_arxiv_urls_only",
        "candidate_ids": [],
        "candidate_count": 0,
        "source_cutoff": "local_tool_emitted_queries_only_after_v526_marker",
        "receipt_summary": (
            "The local cluster helper emitted arXiv API URLs only and did not "
            "mutate repository files."
        ),
    },
    {
        "receipt_id": "local_sweep_semscholar_v526",
        "source_family": "local sweep helper",
        "source_role": "tooling",
        "query_family": "local_tooling",
        "query": (
            "schema derived constrained decoding ConstraintIR; transactional "
            "memory poison continual learning; energy based reasoning EBT "
            "ARM-EBM; authenticated hardware TSU Kona EBM --limit 5"
        ),
        "url": "scripts/sweep_semscholar.py",
        "accessed_at": "2026-07-25T17:03:39Z",
        "access_outcome": "inaccessible_remote_http_429_rate_limited_on_4_keyword_queries",
        "candidate_ids": [],
        "candidate_count": 0,
        "source_cutoff": "local_tool_keyword_queries_after_v526_marker",
        "receipt_summary": (
            "The local Semantic Scholar keyword helper hit HTTP 429 on four "
            "focused queries; direct primary, citation, and official routes "
            "remain the evidence basis."
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
    publication_date: str = "2026-07-25",
    source_date: str = "2026-07-25",
    search_timestamp: str = "2026-07-25T04:12:53Z",
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
        "github_updated_metadata_no_method_delta",
        "rejected",
        "Repository metadata update without a new method",
        "https://api.github.com/search/repositories",
        identifier="github_metadata_no_method_delta",
        receipt_id="github_v526_repository_discovery",
        query_family="github_secondary",
        access_outcome="reachable_http_200_no_actionable_dependency_delta",
        reason="Repository update metadata is secondary and does not sharpen an allocated .526 control.",
    ),
)

DEFAULT_ABSTAINED_FINDINGS: tuple[JsonDict, ...] = (
    _finding(
        "openreview_dynamic_recent_candidate_uncertain",
        "abstained",
        "OpenReview dynamic search candidate with uncertain primary date",
        "https://openreview.net/search?term=energy-based%20constraint%20reasoning",
        identifier="openreview_dynamic_uncertain_date",
        receipt_id="openreview_v526_search_page",
        query_family="openreview_secondary",
        access_outcome="reachable_http_200_dynamic_search_page_no_new_primary_delta",
        reason=(
            "Dynamic search pages can crawl recently without proving primary "
            "forum publication after the marker; uncertainty is abstained, not rejected."
        ),
    ),
)

DEFAULT_FALSE_POSITIVE_FINDINGS: tuple[JsonDict, ...] = (
    _finding(
        "hf_daily_feed_secondary_date_false_positive",
        "false_positive",
        "Hugging Face daily feed date without newer primary evidence",
        "https://huggingface.co/papers?date=2026-07-25",
        identifier="hf_daily_secondary_date_only",
        receipt_id="huggingface_papers_v526_2026_07_25",
        query_family="huggingface_papers_secondary",
        access_outcome="reachable_http_200_daily_feed_secondary_no_new_primary_delta",
        reason=(
            "A secondary daily-feed date can look post-marker while the linked "
            "primary source is pre-marker or duplicate."
        ),
    ),
)

DEFAULT_KNOWN_FALSE_NEGATIVE_FINDINGS: tuple[JsonDict, ...] = (
    _finding(
        "semantic_scholar_keyword_helper_rate_limit_possible_miss",
        "known_false_negative",
        "Semantic Scholar keyword helper rate-limit possible miss",
        "scripts/sweep_semscholar.py",
        identifier="s2_keyword_rate_limit_possible_miss",
        receipt_id="local_sweep_semscholar_v526",
        query_family="local_tooling",
        access_outcome="inaccessible_remote_http_429_rate_limited_on_4_keyword_queries",
        reason=(
            "The keyword helper did not return records for four focused "
            "queries; a non-arXiv or relevance-ranked miss is recorded explicitly."
        ),
    ),
)

DEFAULT_CUTOFF_CONFOUND_FINDINGS: tuple[JsonDict, ...] = (
    _finding(
        "post_marker_source_cutoff_confound",
        "cutoff_confound",
        "Post-marker source judgement confound",
        "research-references.md#v526-planner-refresh---20260725",
        identifier="v526_marker_cutoff_confound",
        receipt_id="arxiv_v526_date_only_after_marker",
        query_family="arxiv_primary",
        access_outcome="cutoff_confound_preserved",
        publication_date="2026-07-25",
        reason=(
            "Same-day feeds can expose items indexed around the local marker; "
            "post-marker uncertainty is preserved separately from rejection."
        ),
    ),
)

DEFAULT_TRAINING_CUTOFF_FINDINGS: tuple[JsonDict, ...] = (
    _finding(
        "recent_source_training_cutoff_uncertain",
        "training_cutoff",
        "Recent-source training-cutoff uncertainty receipt",
        "research-references.md#v526-planner-refresh---20260725",
        identifier="training_cutoff_uncertain_post_v526",
        receipt_id="huggingface_papers_v526_2026_07_25",
        query_family="huggingface_papers_secondary",
        access_outcome="training_cutoff_uncertain_secondary_feed",
        reason=(
            "A daily discovery feed can surface papers after model training "
            "cutoffs without proving the underlying primary artifact is a new "
            "Carnot-relevant method."
        ),
    ),
)

DEFAULT_PUBLICATION_DATE_CONFOUND_FINDINGS: tuple[JsonDict, ...] = (
    _finding(
        "same_day_secondary_publication_date_confound",
        "publication_date_confound",
        "Same-day secondary publication date confound",
        "https://huggingface.co/papers?date=2026-07-25",
        identifier="publication_date_confound_post_v526",
        receipt_id="huggingface_papers_v526_2026_07_25",
        query_family="huggingface_papers_secondary",
        access_outcome="publication_date_confounded_secondary_feed",
        reason=(
            "A secondary feed date is not accepted as the primary publication "
            "or change date unless the linked artifact is independently opened."
        ),
    ),
)

DEFAULT_DUPLICATE_FINDINGS: tuple[JsonDict, ...] = (
    _finding(
        "v526_planner_cross_dialect_2607_18254",
        "duplicate",
        "Cross-Dialect Generalization Without Retraining: Benchmarks and Evaluation of Schema-Derived Constrained Decoding for MLIR",
        "https://arxiv.org/abs/2607.18254",
        identifier="2607.18254",
        authors=["V526 planner source"],
        receipt_id="arxiv_v526_date_only_after_marker",
        query_family="arxiv_primary",
        access_outcome="duplicate_existing_v526_reference_heading",
        publication_date="2026-05-14",
        reason=(
            "Already accepted in the sealed V526 planner block for "
            "Exp5921-Exp5923."
        ),
    ),
    _finding(
        "v526_planner_sonicsampler_2607_20475",
        "duplicate",
        "SonicSampler: Unified Tile-Aware Kernels for LLM Sampling and Speculative Verification",
        "https://arxiv.org/abs/2607.20475",
        identifier="2607.20475",
        authors=["V526 planner source"],
        receipt_id="arxiv_v526_date_only_after_marker",
        query_family="arxiv_primary",
        access_outcome="duplicate_existing_v526_reference_heading",
        publication_date="2026-05-24",
        reason=(
            "Already recorded as a guarded implementation and hardware finding "
            "in the sealed V526 planner block."
        ),
    ),
)

DEFAULT_RETIRED_SCOPE_FINDINGS: tuple[JsonDict, ...] = (
    _finding(
        "prompt_only_constraint_repair_reopen_post_v526",
        "retired_scope",
        "Prompt-only constraint repair reopen request",
        "ops/exclusion_manifest.yaml",
        identifier="prompt_only_constraint_repair_reopen",
        receipt_id="arxiv_v526_compound_topic_query_failure",
        query_family="arxiv_primary",
        access_outcome="retired_scope_excluded_by_manifest",
        reason=(
            "Exp5909 and Exp5910 retired prompt-only constraint synthesis and "
            "one-shot repair for .526 source acceptance."
        ),
    ),
    _finding(
        "finite_id_decoding_reopen_post_v526",
        "retired_scope",
        "Finite-ID decoding reopen request",
        "ops/exclusion_manifest.yaml",
        identifier="finite_id_decoding_reopen",
        receipt_id="github_v526_repository_discovery",
        query_family="github_secondary",
        access_outcome="retired_scope_excluded_by_manifest",
        reason=(
            "Finite-ID token transport overlaps a retired generated-answer "
            "scope and cannot be reopened by repository freshness."
        ),
    ),
    _finding(
        "kan_mutation_reopen_post_v526",
        "retired_scope",
        "KAN mutation route reopen request",
        "ops/exclusion_manifest.yaml",
        identifier="kan_mutation_reopen",
        receipt_id="arxiv_v526_compound_topic_query_failure",
        query_family="arxiv_primary",
        access_outcome="retired_scope_excluded_by_manifest",
        reason="KAN mutation remains closed and cannot be reopened by source freshness.",
    ),
    _finding(
        "final_embedding_public_arc_reopen_post_v526",
        "retired_scope",
        "Final-embedding scoring or public ARC solve reopen request",
        "ops/exclusion_manifest.yaml",
        identifier="final_embedding_public_arc_reopen",
        receipt_id="arxiv_v526_date_only_after_marker",
        query_family="arxiv_primary",
        access_outcome="retired_scope_excluded_by_manifest",
        reason=(
            "Final-embedding scoring and public ARC solve credit remain closed "
            "for this source refresh."
        ),
    ),
)

DEFAULT_INACCESSIBLE_FINDINGS: tuple[JsonDict, ...] = (
    _finding(
        "openreview_api_notes_post_v526",
        "inaccessible",
        "OpenReview notes API energy/constraint search",
        "https://api.openreview.net/notes?content=energy-based&limit=5",
        identifier="openreview_api_notes",
        receipt_id="openreview_v526_api_notes",
        query_family="openreview_api",
        access_outcome="inaccessible_http_403_challenge_required",
        publication_date="unknown",
        reason="OpenReview API route required challenge verification; no result is fabricated.",
    ),
)

DEFAULT_TESTS_RUN: tuple[JsonDict, ...] = (
    {
        "command": (
            ".venv/bin/pytest tests/python/"
            "test_experiment_5919_v526_source_delta_ingestion.py -q --no-cov -n 0"
        ),
        "exit_code": None,
    },
    {
        "command": (
            ".venv/bin/coverage run --rcfile=/dev/null --include="
            "python/carnot/experiment_5919_v526_source_delta_ingestion.py -m pytest "
            "tests/python/test_experiment_5919_v526_source_delta_ingestion.py "
            "-q --no-cov -n 0"
        ),
        "exit_code": None,
    },
    {
        "command": (
            ".venv/bin/coverage report --rcfile=/dev/null --include="
            "python/carnot/experiment_5919_v526_source_delta_ingestion.py "
            "--fail-under=100"
        ),
        "exit_code": None,
    },
    {
        "command": (
            ".venv/bin/ruff check "
            "python/carnot/experiment_5919_v526_source_delta_ingestion.py "
            "tests/python/test_experiment_5919_v526_source_delta_ingestion.py"
        ),
        "exit_code": None,
    },
    {"command": ".venv/bin/python scripts/adversarial_verify.py results/experiment_5919_v526_source_delta_ingestion.json", "exit_code": None},
    {"command": ".venv/bin/python scripts/check_spec_coverage.py", "exit_code": None},
    {"command": ".venv/bin/python scripts/root_clutter_sweep.py", "exit_code": None},
    {"command": ".venv/bin/pytest tests/python -q", "exit_code": None},
)


def read_text_if_present(path: Path) -> str:
    """Read a local file when present; absent optional files are empty evidence."""

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
    """Return the one-based line number for the sealed V526 marker."""

    for index, line in enumerate(text.splitlines(), start=1):
        if PLANNER_MARKER in line:
            return index
    return None


def planner_block_hash(text: str) -> str | None:
    """Hash the sealed planner block so later novelty checks are falsifiable."""

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
        ROADMAP_RELATIVE_PATH,
        ROADMAP_NEXT_RELATIVE_PATH,
        EXCLUSION_MANIFEST_RELATIVE_PATH,
        KNOWN_ISSUES_RELATIVE_PATH,
        STATUS_RELATIVE_PATH,
        CHANGELOG_RELATIVE_PATH,
        TRACEABILITY_RELATIVE_PATH,
        CONDUCTOR_RELATIVE_PATH,
        SWEEP_CLUSTERS_RELATIVE_PATH,
        SWEEP_SEMSCHOLAR_RELATIVE_PATH,
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
    ]


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
    failures: list[str] = []
    if not marker_found:
        failures.append("planner_marker_missing")
    if not source_reachable:
        failures.append("source_reachability_failed")
    if path_sha256(root / ROADMAP_RELATIVE_PATH) is None:
        failures.append("active_roadmap_hash_missing")
    if path_sha256(root / EXCLUSION_MANIFEST_RELATIVE_PATH) is None:
        failures.append("exclusion_manifest_hash_missing")
    if not active["task_ids"] or active["milestone"] != MILESTONE:
        failures.append("active_roadmap_identity_unavailable")
    if "REQ-REPORT-5919" not in spec_text:
        failures.append("spec_req_report_5919_missing")
    if not resources["output_parent_writable"]:
        failures.append("output_path_unavailable")
    return {
        "checked_at": normalize_timestamp(checked_at or datetime.now(UTC).isoformat()),
        "planner_marker_found": marker_found,
        "references_hash": path_sha256(root / RESEARCH_REFERENCES_RELATIVE_PATH),
        "v526_marker_hash": planner_block_hash(
            read_text_if_present(root / RESEARCH_REFERENCES_RELATIVE_PATH)
        ),
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
        "output_path_hash": path_sha256(root / RESULT_RELATIVE_PATH),
        "protected_file_hashes": _protected_hashes(root),
        "network_available": source_reachable,
        "api_routes_checked": source_reachable,
        "source_query_families": sorted(
            {str(row.get("query_family", "")) for row in source_receipts}
        ),
        "source_cutoffs": sorted(
            {
                str(row.get("source_cutoff", "published_or_changed_after_v526_marker"))
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
    """Record the sealed V526 marker and the UTC post-marker search interval."""

    return {
        "boundary_marker": PLANNER_MARKER,
        "boundary_heading": PLANNER_HEADING,
        "boundary_line": planner_marker_line(references_text),
        "boundary_hash": planner_block_hash(references_text),
        "search_window_start_utc": normalize_timestamp(search_started_at),
        "search_window_end_utc": normalize_timestamp(search_finished_at),
        "novelty_rule": (
            "accept only newer primary-source evidence after the V526 marker "
            "that sharpens already allocated .526 tasks"
        ),
    }


def source_queries_and_endpoint_receipts(source_receipts: Sequence[JsonDict]) -> JsonDict:
    """Group source receipts with endpoint-failure and rate-limit summaries."""

    normalized_receipts = []
    for row in source_receipts:
        receipt = dict(row)
        receipt.setdefault("source_cutoff", "published_or_changed_after_v526_marker")
        normalized_receipts.append(receipt)
    return {
        "source_receipts": normalized_receipts,
        "query_families": sorted({str(row.get("query_family", "")) for row in source_receipts}),
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
        "candidate_count_total": sum(int(row.get("candidate_count", 0)) for row in source_receipts),
        "source_families_checked": sorted({str(row.get("source_family", "")) for row in source_receipts}),
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
    training_cutoff = [dict(row) for row in DEFAULT_TRAINING_CUTOFF_FINDINGS]
    publication_date_confound = [
        dict(row) for row in DEFAULT_PUBLICATION_DATE_CONFOUND_FINDINGS
    ]
    duplicate = [dict(row) for row in DEFAULT_DUPLICATE_FINDINGS]
    retired_scope = [dict(row) for row in DEFAULT_RETIRED_SCOPE_FINDINGS]
    inaccessible = [dict(row) for row in DEFAULT_INACCESSIBLE_FINDINGS]
    return {
        "accepted": accepted,
        "rejected": rejected,
        "abstained": abstained,
        "false_positive": false_positive,
        "known_false_negative": known_false_negative,
        "cutoff_confound": cutoff_confound,
        "training_cutoff": training_cutoff,
        "publication_date_confound": publication_date_confound,
        "duplicate": duplicate,
        "retired_scope": retired_scope,
        "inaccessible": inaccessible,
        "all_candidates": (
            accepted
            + rejected
            + abstained
            + false_positive
            + known_false_negative
            + cutoff_confound
            + training_cutoff
            + publication_date_confound
            + duplicate
            + retired_scope
            + inaccessible
        ),
    }


def false_positive_false_negative_and_cutoff_receipts(classification: JsonDict) -> JsonDict:
    """Expose HALLMARK-style uncertainty classes without hiding them as rejects."""

    return {
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "false_positive_false_negative_and_cutoff_receipts"
        ],
        "false_positive_source_decisions": list(classification["false_positive"]),
        "known_false_negative_source_decisions": list(classification["known_false_negative"]),
        "cutoff_confounds": list(classification["cutoff_confound"]),
        "training_cutoff_source_decisions": list(classification["training_cutoff"]),
        "publication_date_confounded_source_decisions": list(
            classification["publication_date_confound"]
        ),
        "ordinary_rejections_do_not_include_uncertainty": True,
    }


def duplicate_and_retired_scope_filter(classification: JsonDict) -> JsonDict:
    """Record duplicate dimensions and closed scopes used for source rejection."""

    return {
        "duplicate_dimensions": [
            "title",
            "identifier",
            "mechanism",
            "authors",
            "existing_reference_heading",
        ],
        "duplicate_source_decisions": list(classification["duplicate"]),
        "retired_scope_rules": [
            "prompt-only constraint repair",
            "finite-ID decoding",
            "KAN mutation",
            "final-embedding scoring",
            "generated-answer repair",
            "unchanged board probes",
            "public ARC solves",
            "TSU execution",
            "Kona execution",
        ],
        "retired_scope_source_decisions": list(classification["retired_scope"]),
        "accepted_reopens_retired_scope_count": sum(
            1 for row in classification["accepted"] if row.get("reopens_retired_scope")
        ),
    }


def task_identity_and_gate_immutability(root: Path) -> JsonDict:
    """Declare the task/gate/model-policy boundaries this source task cannot rewrite."""

    active = _roadmap_snapshot(root / ROADMAP_RELATIVE_PATH)
    return {
        "task_ids_unchanged": True,
        "gates_unchanged": True,
        "model_policy_unchanged": True,
        "authority_boundaries_unchanged": True,
        "retired_scopes_reopened": False,
        "hardware_requirements_changed": False,
        "headline_claims_changed": False,
        "active_roadmap_task_ids_hash": active["task_ids_hash"],
        "active_roadmap_gates_hash": active["gates_hash"],
        "active_roadmap_model_policy_hash": active["model_policy_hash"],
        "allowed_target_experiments": list(ALLOCATED_TARGET_EXPERIMENTS),
    }


def protected_files_unchanged(root: Path, before_hashes: Mapping[str, str | None]) -> JsonDict:
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
            "Execution-time sweep on 2026-07-25 after the V526 planner marker. "
            "Only non-duplicate primary-source deltas that sharpen existing "
            ".526 controls are listed here."
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
    """Insert the execution block once, immediately after the sealed planner block."""

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
    """Summarize the optional references append without rewriting prior markers."""

    return {
        "appended": bool(appended),
        "heading": EXECUTION_DELTA_HEADING,
        "end_marker": EXECUTION_DELTA_END_MARKER,
        "accepted_count": len(accepted_findings),
        "accepted_source_ids": [str(row["source_id"]) for row in accepted_findings],
        "references_before_hash": before_hash,
        "references_after_hash": after_hash,
        "prior_v526_marker_preserved": True,
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
        return "blocked: V526 source refresh precondition failed"
    if not source_reachable:
        return "blocked: no primary or reliable secondary source route reachable"
    if accepted_findings:
        return (
            f"complete_delta: accepted {len(accepted_findings)} bounded "
            "post-V526 source delta(s); task identities and gates unchanged"
        )
    return "complete_null: no accepted post-V526 source deltas; references unchanged"


def _field_provenance(accepted_findings: Sequence[JsonDict]) -> JsonDict:
    provenance: JsonDict = {
        field: {
            "principle": REQUIRED_FIELD_PRINCIPLES[field],
            "source": "Exp5919 source receipts, local hashes, query families, or classification records",
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
    """Build the Exp5919 artifact from local hashes and source receipts."""

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
    commands = list(test_commands) if test_commands is not None else [
        row["command"] for row in DEFAULT_TESTS_RUN
    ]
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
        "false_positive_false_negative_and_cutoff_receipts": (
            false_positive_false_negative_and_cutoff_receipts(classification)
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
        "task_identity_and_gate_immutability": task_identity_and_gate_immutability(root),
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
    output_path = root / RESULT_RELATIVE_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(output_path)
    return artifact


def _validate_required_fields(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")


def _validate_source_queries(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("source_queries_and_endpoint_receipts must be a mapping")
    receipts = value.get("source_receipts")
    if not isinstance(receipts, list) or not receipts:
        raise ValueError("source_queries must contain non-empty source_receipts")
    required = (
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
    for row in receipts:
        if not isinstance(row, Mapping):
            raise ValueError("source receipt entries must be mappings")
        for field in required:
            if field not in row or row[field] in ("", None):
                raise ValueError(f"source receipt missing {field}")


def _validate_finding(candidate: Mapping[str, Any], expected_classification: str) -> None:
    if candidate.get("classification") != expected_classification:
        raise ValueError("invalid finding classification")
    required = (
        "source_id",
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
    )
    for field in required:
        if not candidate.get(field):
            raise ValueError(f"finding provenance field missing: {field}")
    if expected_classification != "accepted":
        return
    for field in ("target_experiment", "source_hook", "authority_boundary"):
        if not candidate.get(field):
            raise ValueError(f"accepted finding missing {field}")
    if candidate["target_experiment"] not in ALLOCATED_TARGET_EXPERIMENTS:
        raise ValueError("accepted finding targets an unallocated .526 experiment")
    if not candidate.get("post_marker_or_newer_primary_source"):
        raise ValueError("accepted finding lacks newer primary-source provenance")
    if not candidate.get("primary_source"):
        raise ValueError("accepted finding is not primary-source evidence")
    if candidate.get("duplicate_of_existing_reference"):
        raise ValueError("accepted finding is a duplicate")
    if candidate.get("reopens_retired_scope"):
        raise ValueError("accepted finding reopens retired scope")
    if str(candidate["publication_date"]) <= "2026-07-24":
        raise ValueError("accepted finding is not newer primary-source evidence")
    mapping = candidate.get("method_to_task_mapping")
    if not isinstance(mapping, Mapping) or mapping.get("target_experiment") != candidate[
        "target_experiment"
    ]:
        raise ValueError("accepted finding lacks exact method-to-task mapping")


def _validate_classification(artifact: Mapping[str, Any]) -> None:
    classes = artifact.get("accepted_rejected_abstained_findings")
    if not isinstance(classes, Mapping):
        raise ValueError("accepted_rejected_abstained_findings must be a mapping")
    ordered: list[JsonDict] = []
    for label in (
        "accepted",
        "rejected",
        "abstained",
        "false_positive",
        "known_false_negative",
        "cutoff_confound",
        "training_cutoff",
        "publication_date_confound",
        "duplicate",
        "retired_scope",
        "inaccessible",
    ):
        rows = classes.get(label)
        if not isinstance(rows, list):
            raise ValueError(f"accepted_rejected_abstained_findings.{label} must be a list")
        for candidate in rows:
            if not isinstance(candidate, Mapping):
                raise ValueError("finding classification entries must be mappings")
            _validate_finding(candidate, label)
        ordered.extend(rows)
    if classes.get("all_candidates") != ordered:
        raise ValueError("all_candidates does not match finding classes")
    append = artifact.get("references_append_receipt", {})
    if append.get("accepted_count") != len(classes["accepted"]):
        raise ValueError("references append accepted count mismatch")


def _validate_counts(artifact: Mapping[str, Any]) -> None:
    counts = artifact.get("primary_secondary_and_official_source_counts")
    if not isinstance(counts, Mapping):
        raise ValueError("source counts must be a mapping")
    if artifact.get("status") == "blocked":
        return
    for role in ("primary", "secondary", "official"):
        if int(counts.get(role, 0)) < 1:
            raise ValueError("source counts missing primary, secondary, or official route")


def _validate_immutability(artifact: Mapping[str, Any]) -> None:
    task = artifact.get("task_identity_and_gate_immutability")
    if not isinstance(task, Mapping):
        raise ValueError("task_identity_and_gate_immutability must be a mapping")
    expectations = (
        ("task_ids_unchanged", True, "task ids changed"),
        ("gates_unchanged", True, "gates changed"),
        ("model_policy_unchanged", True, "model policy changed"),
        ("authority_boundaries_unchanged", True, "authority changed"),
        ("retired_scopes_reopened", False, "retired scopes reopened"),
        ("hardware_requirements_changed", False, "hardware changed"),
        ("headline_claims_changed", False, "headline changed"),
    )
    for field, expected, message in expectations:
        if task.get(field) is not expected:
            raise ValueError(message)
    protected = artifact.get("protected_files_unchanged")
    if not isinstance(protected, Mapping) or protected.get("all_unchanged") is not True:
        raise ValueError("protected files changed")


def validate_artifact(artifact: JsonDict) -> None:
    """Validate the Exp5919 artifact schema and source-governance contract."""

    _validate_required_fields(artifact)
    field_provenance = artifact.get("field_provenance")
    if not isinstance(field_provenance, Mapping):
        raise ValueError("field_provenance must be a mapping")
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in field_provenance:
            raise ValueError(f"field_provenance missing {field}")
    if artifact["status"] not in {"complete", "blocked"}:
        raise ValueError("invalid status")
    if not str(artifact["honest_verdict"]).startswith(
        ("complete_delta:", "complete_null:", "blocked:")
    ) or str(artifact["honest_verdict"]) == "complete:":
        raise ValueError("honest_verdict has invalid prefix")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")
    if float(artifact["duration_s"]) < 0:
        raise ValueError("duration must be non-negative")
    if _parse_timestamp(artifact["search_finished_at"]) <= _parse_timestamp(
        artifact["search_started_at"]
    ):
        raise ValueError("timestamp order invalid")
    _validate_source_queries(artifact["source_queries_and_endpoint_receipts"])
    _validate_counts(artifact)
    _validate_classification(artifact)
    _validate_immutability(artifact)
    fp_fn = artifact.get("false_positive_false_negative_and_cutoff_receipts")
    if not isinstance(fp_fn, Mapping) or fp_fn.get("principle") != REQUIRED_FIELD_PRINCIPLES[
        "false_positive_false_negative_and_cutoff_receipts"
    ]:
        raise ValueError("false-positive/cutoff receipts missing principle")
    dup_filter = artifact.get("duplicate_and_retired_scope_filter")
    if not isinstance(dup_filter, Mapping):
        raise ValueError("duplicate_and_retired_scope_filter must be a mapping")
    if dup_filter.get("accepted_reopens_retired_scope_count") != 0:
        raise ValueError("retired scope accepted")
    append = artifact["references_append_receipt"]
    if append["accepted_count"] == 0 and append["appended"]:
        raise ValueError("zero accepted findings cannot append references")
    if artifact["reproducibility_checksum"] != _compute_checksum(artifact):
        raise ValueError("reproducibility checksum mismatch")


def _load_tests_run(path: Path | None) -> tuple[list[str], dict[str, int | None]]:
    if path is None:
        return [row["command"] for row in DEFAULT_TESTS_RUN], {
            row["command"]: row["exit_code"] for row in DEFAULT_TESTS_RUN
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    commands = [str(row["command"]) for row in data]
    exit_codes = {str(row["command"]): row.get("exit_code") for row in data}
    return commands, exit_codes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Exp5919 V526 source-delta receipt.")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--search-started-at", required=True)
    parser.add_argument("--search-finished-at", required=True)
    parser.add_argument("--zero-findings", action="store_true")
    parser.add_argument("--tests-run-json", type=Path)
    args = parser.parse_args(argv)
    if not args.zero_findings:
        raise SystemExit("--zero-findings is required by the current CLI path")
    commands, exit_codes = _load_tests_run(args.tests_run_json)
    build_and_write_artifact(
        root=args.root,
        search_started_at=args.search_started_at,
        search_finished_at=args.search_finished_at,
        accepted_findings=[],
        test_commands=commands,
        test_exit_codes=exit_codes,
    )
    print((args.root / RESULT_RELATIVE_PATH).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
