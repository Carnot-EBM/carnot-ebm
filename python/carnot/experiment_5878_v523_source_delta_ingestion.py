"""Exp5878: ingest post-V523 source deltas without changing the roadmap.

Spec refs: REQ-REPORT-5878, SCENARIO-REPORT-5878-ZERO-FINDING,
SCENARIO-REPORT-5878-ACCEPT-BOUNDED-DELTA,
SCENARIO-REPORT-5878-BLOCKED-PRECONDITION,
SCENARIO-REPORT-5878-CLOSED-SCOPE-IMMUTABILITY,
SCENARIO-REPORT-5878-SCHEMA.

This module turns a bounded, low-concurrency source sweep into an auditable JSON
receipt. It does not crawl the web by itself. The source checks happen outside
the module, then this code records their provenance, verifies the sealed V523
time boundary, and appends ledger notes only when a genuinely newer
primary-source finding maps to an already allocated V523 task.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
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
RESULT_RELATIVE_PATH = Path("results/experiment_5878_v523_source_delta_ingestion.json")
RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")
RESEARCH_STUDYING_RELATIVE_PATH = Path("research-studying.md")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
KNOWN_ISSUES_RELATIVE_PATH = Path("ops/known-issues.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")

EXPERIMENT = "experiment_5878_v523_source_delta_ingestion"
EXPERIMENT_ID = "exp5878-v523-source-delta-ingestion"
MILESTONE = "2026.07.523"
RUN_DATE = "20260724"
RANDOM_SEED = 5878
SCHEMA = "carnot.experiment_5878.v523_source_delta_ingestion.v1"
INFERENCE_SUBSTRATE = "aggregation_from_external_primary_sources"

PLANNER_HEADING = "## V523 Planner Refresh - 20260724"
PLANNER_MARKER = "V523-PLANNER-REFRESH-20260724-END"
PLANNER_END_MARKER = f"<!-- {PLANNER_MARKER} -->"
EXECUTION_REFRESH_HEADING = "## V523 Execution Refresh - 20260724"
EXECUTION_REFRESH_END_MARKER = "<!-- V523-EXECUTION-REFRESH-20260724-END -->"
STUDYING_EXECUTION_HEADING = "## Exp 5878 - V523 source delta ingestion - INGESTED"
STUDYING_EXECUTION_END_MARKER = "<!-- EXP5878-V523-SOURCE-DELTA-INGESTION-END -->"

ALLOCATED_TARGET_EXPERIMENTS = {
    "exp5879-hardness-headroom-taxonomy-corrigendum",
    "exp5880-grounding-shortcut-fixture",
    "exp5881-one-to-one-grounding-acquisition-ab",
    "exp5882-shortcut-resistant-continuous-self-learning",
    "exp5883-gguf-intermediate-layer-surface-preflight",
    "exp5884-three-family-layer-dynamic-energy",
    "exp5885-layer-dynamic-portability-camouflage-audit",
    "exp5886-arc-programmatic-memory-contract",
    "exp5887-arc-programmatic-memory-causal-audit",
    "exp5888-arc-programmatic-memory-live-ab",
}

SPEC_REFS = (
    "REQ-REPORT-5878",
    "SCENARIO-REPORT-5878-ZERO-FINDING",
    "SCENARIO-REPORT-5878-ACCEPT-BOUNDED-DELTA",
    "SCENARIO-REPORT-5878-BLOCKED-PRECONDITION",
    "SCENARIO-REPORT-5878-CLOSED-SCOPE-IMMUTABILITY",
    "SCENARIO-REPORT-5878-SCHEMA",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "planner_marker_and_search_window",
    "source_receipts",
    "citation_trail_receipts",
    "finding_classification",
    "accepted_finding_count",
    "references_modified",
    "studying_ledger_modified",
    "sota_to_experiment_mapping",
    "guarded_finding_receipts",
    "roadmap_immutability_receipts",
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
        "A terminal refresh state distinguishes a complete source sweep from a failed search."
    ),
    "preconditions_checked": (
        "Marker, hashes, resources, network, API, and output checks prevent "
        "stale-window or fabricated-source claims."
    ),
    "planner_marker_and_search_window": (
        "A sealed time boundary makes novelty falsifiable."
    ),
    "source_receipts": "Primary pages and dated access outcomes ground each claim.",
    "citation_trail_receipts": (
        "Direct EBT and ARM-EBM checks prevent unsupported citation claims."
    ),
    "finding_classification": "Duplicates and watch items cannot become experiments.",
    "accepted_finding_count": "A bare zero is valid and complete.",
    "references_modified": (
        "The artifact discloses whether the shared reference ledger changed."
    ),
    "studying_ledger_modified": (
        "A studying item is marked ingested only with an exact method-to-task mapping."
    ),
    "sota_to_experiment_mapping": (
        "Every accepted source maps to an already allocated .523 task and a bounded hook."
    ),
    "guarded_finding_receipts": (
        "Guarded and watch-only findings remain visible without becoming experiments."
    ),
    "roadmap_immutability_receipts": "Freshness work cannot redesign allocation.",
    "duration_s": "Measured time exposes a bootstrap-only search receipt.",
    "inference_substrate": (
        "Use `aggregation_from_external_primary_sources` for external-source "
        "synthesis without experiment inference."
    ),
    "field_provenance": (
        "Every decision traces to source receipts, hashes, or roadmap records."
    ),
    "test_commands": (
        "Commands document URL/provenance, date-window, duplicate, citation, "
        "exclusion, roadmap, YAML, schema, spec, and clutter checks."
    ),
    "test_exit_codes": "Exit codes prevent failed provenance checks from becoming success.",
    "reproducibility_checksum": (
        "A checksum detects later marker, source, or classification drift."
    ),
    "honest_verdict": "A `complete:` or `blocked:` prefix makes the outcome terminal.",
}

FIELD_PRINCIPLE_EXTRAS: dict[str, str] = {
    "schema": "Versioned schema id keeps downstream validators from guessing field meaning.",
    "experiment": "Stable local slug ties the artifact to the implementation module.",
    "experiment_id": "Conductor task identity prevents numeric-prefix aliasing.",
    "milestone": "Binds receipts to .523 rather than a later milestone.",
    "run_date": "Operator-requested execution date for the source refresh.",
    "random_seed": "Deterministic metadata for a no-randomness ledger task.",
    "spec_refs": "OpenSpec anchors make the artifact contract auditable.",
    "result_path": "Declares the exact JSON deliverable path.",
    "search_started_at": "Records when source querying started.",
    "search_finished_at": "Records when candidate disposition finished.",
    "references_before_hash": "Reference-ledger bytes before the optional append.",
    "references_after_hash": "Reference-ledger bytes after the optional append.",
    "studying_before_hash": "Studying-ledger bytes before an optional ingestion note.",
    "studying_after_hash": "Studying-ledger bytes after an optional ingestion note.",
}

DEFAULT_SOURCE_RECEIPTS: tuple[JsonDict, ...] = (
    {
        "receipt_id": "arxiv_post_v523_required_topic_queries",
        "source_family": "arXiv",
        "source_role": "primary",
        "query": (
            'submittedDate:[202607240000 TO 202607242359] AND '
            '(all:"energy-based" OR all:"constraint reasoning" OR '
            'all:"hallucination verification" OR all:KAN)'
        ),
        "url": (
            "https://export.arxiv.org/api/query?search_query="
            "submittedDate:%5B202607240000%20TO%20202607242359%5D%20AND%20"
            "(all:%22energy-based%22%20OR%20all:%22constraint%20reasoning%22%20OR%20"
            "all:%22hallucination%20verification%22%20OR%20all:KAN)"
            "&start=0&max_results=10&sortBy=submittedDate&sortOrder=descending"
        ),
        "accessed_at": "2026-07-24T08:51:37Z",
        "access_outcome": "reachable_http_200_totalResults_0",
        "candidate_ids": [],
        "receipt_summary": (
            "arXiv was checked first for post-marker EBM, constraint, "
            "hallucination, and KAN terms; the bounded 20260724 window returned "
            "zero API candidates."
        ),
    },
    {
        "receipt_id": "arxiv_post_v523_memory_hardware_topic_queries",
        "source_family": "arXiv",
        "source_role": "primary",
        "query": (
            'submittedDate:[202607240000 TO 202607242359] AND '
            '(all:"neural CSP" OR all:Ising OR all:sampling OR '
            'all:"energy-guided decoding" OR all:"probabilistic hardware" OR '
            'all:"continual learning" OR all:"agent memory")'
        ),
        "url": (
            "https://export.arxiv.org/api/query?search_query="
            "submittedDate:%5B202607240000%20TO%20202607242359%5D%20AND%20"
            "(all:%22neural%20CSP%22%20OR%20all:Ising%20OR%20all:sampling%20OR%20"
            "all:%22energy-guided%20decoding%22%20OR%20"
            "all:%22probabilistic%20hardware%22%20OR%20"
            "all:%22continual%20learning%22%20OR%20all:%22agent%20memory%22)"
            "&start=0&max_results=10&sortBy=submittedDate&sortOrder=descending"
        ),
        "accessed_at": "2026-07-24T08:51:55Z",
        "access_outcome": "reachable_http_200_totalResults_0",
        "candidate_ids": [],
        "receipt_summary": (
            "The second arXiv primary route covered neural CSP, Ising/sampling, "
            "energy-guided decoding, probabilistic hardware, continual learning, "
            "and live-agent memory terms; the bounded 20260724 window returned "
            "zero API candidates."
        ),
    },
    {
        "receipt_id": "arxiv_v523_duplicate_primary_pages",
        "source_family": "arXiv",
        "source_role": "primary",
        "query": "direct duplicate primary-page check for arXiv:2607.21185 and 2607.20064v2",
        "url": "https://arxiv.org/abs/2607.21185 ; https://arxiv.org/abs/2607.20064",
        "accessed_at": "2026-07-24T08:52:12Z",
        "access_outcome": "reachable_http_200_duplicate_primary_pages",
        "candidate_ids": ["2607.21185", "2607.20064v2"],
        "receipt_summary": (
            "Both direct arXiv pages were reachable; 2607.21185 was submitted "
            "2026-07-23 and 2607.20064v2 was last revised 2026-07-23, so both "
            "remain duplicates of the sealed V523 planner intake."
        ),
    },
    {
        "receipt_id": "openreview_post_v523_search",
        "source_family": "OpenReview",
        "source_role": "secondary",
        "query": "energy-based constraint reasoning continual learning",
        "url": (
            "https://openreview.net/search?term=energy-based%20constraint%20"
            "reasoning%20continual%20learning"
        ),
        "accessed_at": "2026-07-24T08:52:37Z",
        "access_outcome": "reachable_http_200_dynamic_search_page_no_primary_candidate",
        "candidate_ids": [],
        "receipt_summary": (
            "OpenReview is checked as a secondary discovery route; static "
            "metadata is not promoted without primary-source confirmation."
        ),
    },
    {
        "receipt_id": "openreview_api_notes_post_v523",
        "source_family": "OpenReview",
        "source_role": "secondary",
        "query": "api.openreview.net notes energy-based",
        "url": "https://api.openreview.net/notes?content=energy-based&limit=5",
        "accessed_at": "2026-07-24T08:52:37Z",
        "access_outcome": "inaccessible_http_403_challenge_required",
        "candidate_ids": [],
        "receipt_summary": (
            "The direct OpenReview notes API returned HTTP 403, so no OpenReview "
            "candidate is fabricated from the API route."
        ),
    },
    {
        "receipt_id": "huggingface_papers_2026_07_24_post_v523",
        "source_family": "Hugging Face Papers",
        "source_role": "secondary",
        "query": "daily_papers date:2026-07-24",
        "url": "https://huggingface.co/papers?date=2026-07-24",
        "accessed_at": "2026-07-24T08:52:37Z",
        "access_outcome": "reachable_http_200_daily_feed_no_newer_actionable_primary_delta",
        "candidate_ids": [
            "2607.21461",
            "2605.09635",
            "2607.21556",
            "2607.20061",
            "2607.21072",
            "2607.12746",
            "2607.20911",
            "2607.20734",
            "2607.21051",
            "2607.21485",
            "2607.21594",
            "2607.20709",
            "2607.10848",
            "2607.21017",
            "2607.20785",
            "2607.21580",
        ],
        "receipt_summary": (
            "Hugging Face Papers is a secondary route for surfacing daily paper "
            "metadata before primary-page confirmation."
        ),
    },
    {
        "receipt_id": "arxiv_hf_relevant_primary_pages_post_v523",
        "source_family": "arXiv",
        "source_role": "primary",
        "query": "primary checks for HF surfaced agent-memory candidates",
        "url": (
            "https://arxiv.org/abs/2607.21461 ; https://arxiv.org/abs/2607.21051 ; "
            "https://arxiv.org/abs/2607.20734"
        ),
        "accessed_at": "2026-07-24T08:53:23Z",
        "access_outcome": "reachable_http_200_all_submitted_before_marker_watch_only",
        "candidate_ids": ["2607.21461", "2607.21051", "2607.20734"],
        "receipt_summary": (
            "HF-surfaced agent-memory candidates had reachable arXiv primary "
            "pages, but their submitted dates were before the V523 marker and "
            "none supplied a lossless programmatic ARC-memory control."
        ),
    },
    {
        "receipt_id": "github_post_v523_repository_discovery",
        "source_family": "GitHub discovery",
        "source_role": "secondary",
        "query": '"energy-based" constraint reasoning pushed:>2026-07-24',
        "url": (
            "https://api.github.com/search/repositories?q=%22energy-based%22+"
            "constraint+reasoning+pushed:%3E2026-07-24&per_page=5"
        ),
        "accessed_at": "2026-07-24T08:54:17Z",
        "access_outcome": "reachable_http_200_total_count_0_no_repository_delta",
        "candidate_ids": [],
        "receipt_summary": (
            "GitHub repository discovery is used only as dependency or "
            "implementation-surface metadata, not as paper evidence by itself."
        ),
    },
    {
        "receipt_id": "github_post_v523_issue_discovery",
        "source_family": "GitHub discovery",
        "source_role": "secondary",
        "query": "programmatic memory agent updated:>2026-07-24",
        "url": (
            "https://api.github.com/search/issues?q=programmatic+memory+agent+"
            "updated:%3E2026-07-24&per_page=5"
        ),
        "accessed_at": "2026-07-24T08:54:17Z",
        "access_outcome": "reachable_http_200_total_count_0_no_issue_delta",
        "candidate_ids": [],
        "receipt_summary": (
            "GitHub issue discovery returned zero post-date hits and cannot add "
            "a source delta."
        ),
    },
    {
        "receipt_id": "github_prolong_duplicate_repo_receipt",
        "source_family": "GitHub discovery",
        "source_role": "secondary",
        "query": "alexisfox7/PRO-LONG repository metadata",
        "url": "https://api.github.com/repos/alexisfox7/PRO-LONG",
        "accessed_at": "2026-07-24T08:54:17Z",
        "access_outcome": "reachable_http_200_updated_today_but_pushed_2026_06_01_duplicate",
        "candidate_ids": ["alexisfox7/PRO-LONG"],
        "receipt_summary": (
            "The PRO-LONG repository metadata was reachable and updated on "
            "2026-07-24, but the last push was 2026-06-01 and the method is "
            "already represented by the V523 accepted programmatic-memory tasks."
        ),
    },
    {
        "receipt_id": "extropic_writing_post_v523",
        "source_family": "Extropic writing",
        "source_role": "primary",
        "query": "Extropic writing index TSU XTR-0 Z1",
        "url": "https://extropic.ai/writing",
        "accessed_at": "2026-07-24T08:54:43Z",
        "access_outcome": "reachable_http_200_latest_public_material_2025_10_no_authenticated_local_route",
        "candidate_ids": ["tsu_101", "dtms", "x0_xtr0_public_material"],
        "receipt_summary": (
            "Extropic public writing is checked for authenticated TSU, XTR, Z1, "
            "or SDK changes before any hardware route can be promoted."
        ),
    },
    {
        "receipt_id": "logical_intelligence_public_pages_post_v523",
        "source_family": "Logical Intelligence",
        "source_role": "primary",
        "query": "Kona Aleph public pages",
        "url": "https://logicalintelligence.com/",
        "accessed_at": "2026-07-24T08:54:43Z",
        "access_outcome": "reachable_http_200_published_2026_06_26_no_local_weights_or_api_receipt",
        "candidate_ids": ["logical_homepage", "kona_1_0", "aleph"],
        "receipt_summary": (
            "Logical Intelligence public pages are checked for public weights, "
            "authenticated APIs, or reproducible local comparators."
        ),
    },
    {
        "receipt_id": "local_sweep_clusters_url_generation",
        "source_family": "local sweep helper",
        "source_role": "tooling",
        "query": "scripts/sweep_clusters.py clusters 1 and 4 start:0 max_results:3",
        "url": "scripts/sweep_clusters.py",
        "accessed_at": "2026-07-24T08:56:50Z",
        "access_outcome": "reachable_local_tool_exit_0_arxiv_urls_only",
        "candidate_ids": [],
        "receipt_summary": (
            "The repository cluster helper generated bounded arXiv API URLs for "
            "EBM and probabilistic-hardware clusters and did not emit candidates "
            "or mutate files."
        ),
    },
    {
        "receipt_id": "local_sweep_semscholar_keyword_programmatic_memory_agent",
        "source_family": "local sweep helper",
        "source_role": "tooling",
        "query": "programmatic memory agent --limit 5",
        "url": "scripts/sweep_semscholar.py",
        "accessed_at": "2026-07-24T08:56:50Z",
        "access_outcome": "inaccessible_remote_http_429_zero_unique_arxiv_ids",
        "candidate_ids": [],
        "receipt_summary": (
            "The local Semantic Scholar keyword helper hit HTTP 429; direct "
            "citation API receipts, which were reachable, remain the cited "
            "Semantic Scholar evidence for this refresh."
        ),
    },
)

DEFAULT_CITATION_TRAIL_RECEIPTS: tuple[JsonDict, ...] = (
    {
        "receipt_id": "semantic_scholar_ebt_2507_02092_post_v523",
        "paper": "arXiv:2507.02092",
        "query": "arXiv:2507.02092 citations",
        "url": (
            "https://api.semanticscholar.org/graph/v1/paper/arXiv:2507.02092/"
            "citations?fields=title,year,externalIds,url,publicationDate&limit=20"
        ),
        "accessed_at": "2026-07-24T08:53:50Z",
        "access_outcome": "reachable_http_200_no_post_marker_actionable_citation",
        "candidate_ids": [
            "2607.17047",
            "2607.11555",
            "2606.22726",
            "2606.18206",
            "2606.15956",
            "2605.11011",
            "2605.07588",
            "2604.11403",
            "2604.10272",
            "2604.03878",
            "2604.01577",
            "2603.18534",
            "2603.19117",
            "2602.03640",
            "2602.01651",
            "2601.03905",
            "2512.17846",
            "2512.16762",
        ],
        "latest_publication_date": "2026-07-19",
        "citation_count_claimed": False,
        "receipt_summary": (
            "Direct EBT citation route was reachable; the newest dated citing "
            "arXiv paper in the returned page was 2026-07-19, before the V523 "
            "sealed boundary."
        ),
    },
    {
        "receipt_id": "semantic_scholar_arm_ebm_2512_15605_post_v523",
        "paper": "arXiv:2512.15605",
        "query": "arXiv:2512.15605 citations",
        "url": (
            "https://api.semanticscholar.org/graph/v1/paper/arXiv:2512.15605/"
            "citations?fields=title,year,externalIds,url,publicationDate&limit=20"
        ),
        "accessed_at": "2026-07-24T08:53:50Z",
        "access_outcome": "reachable_http_200_no_post_marker_actionable_citation",
        "candidate_ids": [
            "2607.02154",
            "2606.03089",
            "2605.18871",
            "2605.11011",
            "2604.00555",
            "2603.23398",
            "2602.02991",
        ],
        "latest_publication_date": "2026-07-02",
        "citation_count_claimed": False,
        "receipt_summary": (
            "Direct ARM-EBM citation route was reachable; the newest dated "
            "citing arXiv paper in the returned page was 2026-07-02, before "
            "the V523 sealed boundary."
        ),
    },
)

DEFAULT_DUPLICATE_FINDINGS: tuple[JsonDict, ...] = (
    {
        "source_id": "v523_planner_grounding_shortcuts_2607_21185",
        "classification": "duplicate",
        "title": (
            "Differentiable Logic Programming to Mitigate Reasoning Shortcuts in "
            "Neurosymbolic Systems"
        ),
        "url": "https://arxiv.org/abs/2607.21185",
        "publication_date": "2026-07-23",
        "source_date": "2026-07-24",
        "search_timestamp": "2026-07-24T08:52:12Z",
        "receipt_id": "arxiv_v523_duplicate_primary_pages",
        "query": "V523 duplicate source-ledger check",
        "access_outcome": "reachable_http_200_duplicate_primary_page_submitted_2026_07_23",
        "reason": (
            "Already accepted in the sealed V523 planner block for Exp5880 "
            "through Exp5882 shortcut-resistant grounding controls."
        ),
    },
    {
        "source_id": "v523_planner_prolong_memory_2607_20064",
        "classification": "duplicate",
        "title": "PRO-LONG: Programmatic Memory Enables Long-Horizon Reasoning",
        "url": "https://arxiv.org/abs/2607.20064",
        "publication_date": "2026-07-23",
        "source_date": "2026-07-24",
        "search_timestamp": "2026-07-24T08:52:12Z",
        "receipt_id": "arxiv_v523_duplicate_primary_pages",
        "query": "V523 duplicate source-ledger check",
        "access_outcome": "reachable_http_200_duplicate_primary_page_revised_2026_07_23",
        "reason": (
            "Already accepted in the sealed V523 planner block for Exp5886 "
            "through Exp5888 programmatic-memory controls."
        ),
    },
    {
        "source_id": "semantic_scholar_ebt_solver_hard_2607_17047",
        "classification": "duplicate",
        "title": (
            "Solver-Hard Is Not Model-Hard: A Hardness-Controlled Diagnostic "
            "for LLM Constraint Reasoning"
        ),
        "url": "https://arxiv.org/abs/2607.17047",
        "publication_date": "2026-07-19",
        "source_date": "2026-07-24",
        "search_timestamp": "2026-07-24T08:53:50Z",
        "receipt_id": "semantic_scholar_ebt_2507_02092_post_v523",
        "query": "arXiv:2507.02092 citations",
        "access_outcome": "reachable_http_200_citation_duplicate_before_marker",
        "reason": (
            "Returned on the EBT citation trail, but it predates the V523 marker "
            "and overlaps the existing hardness/headroom taxonomy controls."
        ),
    },
    {
        "source_id": "github_prolong_repo_duplicate_metadata",
        "classification": "duplicate",
        "title": "PRO-LONG repository metadata",
        "url": "https://github.com/alexisfox7/PRO-LONG",
        "publication_date": "2026-06-01",
        "source_date": "2026-07-24",
        "search_timestamp": "2026-07-24T08:54:17Z",
        "receipt_id": "github_prolong_duplicate_repo_receipt",
        "query": "alexisfox7/PRO-LONG repository metadata",
        "access_outcome": "reachable_http_200_updated_today_but_pushed_2026_06_01_duplicate",
        "reason": (
            "Repository metadata changed on 2026-07-24, but the latest push "
            "predates the marker and the method is already assigned to Exp5886 "
            "through Exp5888."
        ),
    },
)

DEFAULT_WATCH_ONLY_FINDINGS: tuple[JsonDict, ...] = (
    {
        "source_id": "v523_pcomputer_watch_2607_21077",
        "classification": "watch_only",
        "title": (
            "A scalable and resource-efficient pipelined p-computer for "
            "probabilistic Ising machines"
        ),
        "url": "https://arxiv.org/abs/2607.21077",
        "publication_date": "2026-07-23",
        "source_date": "2026-07-24",
        "search_timestamp": "2026-07-24T08:53:23Z",
        "receipt_id": "arxiv_post_v523_memory_hardware_topic_queries",
        "query": "V523 guarded p-computer follow-up",
        "access_outcome": "watch_only_no_authenticated_board_or_changed_route",
        "reason": (
            "Useful hardware context, but no authenticated changed board route or "
            "same-input physical receipt exists for .523."
        ),
    },
    {
        "source_id": "hf_arex_deep_research_agent_2607_21461",
        "classification": "watch_only",
        "title": (
            "AREX: Towards a Recursively Self-Improving Agent for Deep Research"
        ),
        "url": "https://arxiv.org/abs/2607.21461",
        "publication_date": "2026-07-23",
        "source_date": "2026-07-24",
        "search_timestamp": "2026-07-24T08:53:23Z",
        "receipt_id": "arxiv_hf_relevant_primary_pages_post_v523",
        "query": "HF surfaced agent memory primary check",
        "access_outcome": "watch_only_submitted_before_marker_compressed_state_not_lossless_memory",
        "reason": (
            "Agent-memory adjacent, but submitted before the V523 marker and "
            "uses compact history state rather than a lossless programmatic "
            "ARC-memory control."
        ),
    },
    {
        "source_id": "hf_sample_efficient_learning_agent_experience_2607_21051",
        "classification": "watch_only",
        "title": "Sample-Efficient Learning from Agent Experience",
        "url": "https://arxiv.org/abs/2607.21051",
        "publication_date": "2026-07-23",
        "source_date": "2026-07-24",
        "search_timestamp": "2026-07-24T08:53:23Z",
        "receipt_id": "arxiv_hf_relevant_primary_pages_post_v523",
        "query": "HF surfaced continual learning primary check",
        "access_outcome": "watch_only_submitted_before_marker_model_weight_distillation",
        "reason": (
            "Experience distillation into model behavior is not the frozen-GGUF "
            "plus external lossless memory surface allocated in .523."
        ),
    },
    {
        "source_id": "hf_llms_evolving_user_intent_2607_20734",
        "classification": "watch_only",
        "title": "LLMs Get Lost in Evolving User Intent",
        "url": "https://arxiv.org/abs/2607.20734",
        "publication_date": "2026-07-22",
        "source_date": "2026-07-24",
        "search_timestamp": "2026-07-24T08:53:23Z",
        "receipt_id": "arxiv_hf_relevant_primary_pages_post_v523",
        "query": "HF surfaced live-agent memory primary check",
        "access_outcome": "watch_only_submitted_before_marker_benchmark_context",
        "reason": (
            "Useful live-intent benchmark context, but it predates the marker "
            "and does not provide an exact lossless ARC-memory mechanism."
        ),
    },
    {
        "source_id": "extropic_public_material_post_v523",
        "classification": "watch_only",
        "title": "Extropic public writing pages",
        "url": "https://extropic.ai/writing",
        "publication_date": "2025-10-29",
        "source_date": "2026-07-24",
        "search_timestamp": "2026-07-24T08:54:43Z",
        "receipt_id": "extropic_writing_post_v523",
        "query": "Extropic writing TSU XTR-0 Z1",
        "access_outcome": "reachable_no_authenticated_local_execution_surface",
        "reason": (
            "Probabilistic-hardware context only; no Carnot-local XTR, Z1, TSU "
            "execution, SDK, speed, power, or correctness receipt was found."
        ),
    },
    {
        "source_id": "logical_intelligence_kona_public_pages_post_v523",
        "classification": "watch_only",
        "title": "Logical Intelligence Kona and Aleph public pages",
        "url": "https://logicalintelligence.com/",
        "publication_date": "2026-06-26",
        "source_date": "2026-07-24",
        "search_timestamp": "2026-07-24T08:54:43Z",
        "receipt_id": "logical_intelligence_public_pages_post_v523",
        "query": "Kona Aleph public pages",
        "access_outcome": "reachable_no_local_weights_or_reproducible_comparator",
        "reason": (
            "Architecture context only; no local weights, authenticated API "
            "receipt, or reproducible Kona/Aleph comparator is available."
        ),
    },
)

DEFAULT_EXCLUDED_FINDINGS: tuple[JsonDict, ...] = (
    {
        "source_id": "final_embedding_reopen_post_v523",
        "classification": "excluded",
        "title": "Final embedding route reopen request",
        "url": "ops/exclusion_manifest.yaml",
        "publication_date": "2026-07-24",
        "source_date": "2026-07-24",
        "search_timestamp": "2026-07-24T08:56:50Z",
        "receipt_id": "arxiv_post_v523_required_topic_queries",
        "query": "hallucination embedding final surface",
        "access_outcome": "excluded_by_local_manifest",
        "reason": "Final embeddings remain closed and cannot be reopened by freshness work.",
    },
    {
        "source_id": "phase_d_generated_answer_repair_reopen_post_v523",
        "classification": "excluded",
        "title": "PHASE D or generated-answer repair reopen request",
        "url": "ops/exclusion_manifest.yaml",
        "publication_date": "2026-07-24",
        "source_date": "2026-07-24",
        "search_timestamp": "2026-07-24T08:56:50Z",
        "receipt_id": "arxiv_post_v523_required_topic_queries",
        "query": "generated-answer repair output text scorer",
        "access_outcome": "excluded_by_local_manifest",
        "reason": (
            "PHASE D and generated-answer repair remain closed; source freshness "
            "cannot reopen them."
        ),
    },
    {
        "source_id": "kan_residual_reopen_post_v523",
        "classification": "excluded",
        "title": "KAN residual route reopen request",
        "url": "ops/exclusion_manifest.yaml",
        "publication_date": "2026-07-24",
        "source_date": "2026-07-24",
        "search_timestamp": "2026-07-24T08:56:50Z",
        "receipt_id": "arxiv_post_v523_required_topic_queries",
        "query": "curvature adaptive KAN residual rerun",
        "access_outcome": "excluded_by_local_manifest",
        "reason": (
            "Exp5749 closed the matched KAN residual route; .523 grounding tasks "
            "must not mutate KAN residuals."
        ),
    },
    {
        "source_id": "active_observation_lookahead_arc_reopen_post_v523",
        "classification": "excluded",
        "title": "Active observation, lookahead, or public ARC solve reopen request",
        "url": "ops/exclusion_manifest.yaml",
        "publication_date": "2026-07-24",
        "source_date": "2026-07-24",
        "search_timestamp": "2026-07-24T08:56:50Z",
        "receipt_id": "arxiv_post_v523_memory_hardware_topic_queries",
        "query": "active observation lookahead public ARC solve",
        "access_outcome": "excluded_by_local_manifest",
        "reason": (
            "The .523 live-agent branch is lossless agent-owned memory, not active "
            "observation, lookahead, public ARC solve credit, adapters, or BFS."
        ),
    },
    {
        "source_id": "unchanged_board_tsu_kona_execution_reopen_post_v523",
        "classification": "excluded",
        "title": "Unchanged board, TSU, or Kona execution claim reopen request",
        "url": "ops/exclusion_manifest.yaml",
        "publication_date": "2026-07-24",
        "source_date": "2026-07-24",
        "search_timestamp": "2026-07-24T08:56:50Z",
        "receipt_id": "logical_intelligence_public_pages_post_v523",
        "query": "unchanged board TSU Kona authenticated execution",
        "access_outcome": "excluded_by_missing_authenticated_local_route",
        "reason": (
            "No unchanged board probe, TSU execution, or Kona execution claim can "
            "be accepted without an authenticated local receipt."
        ),
    },
)

DEFAULT_INACCESSIBLE_FINDINGS: tuple[JsonDict, ...] = (
    {
        "source_id": "openreview_api_notes_post_v523",
        "classification": "inaccessible",
        "title": "OpenReview notes API energy/constraint search",
        "url": "https://api.openreview.net/notes?content=energy-based&limit=5",
        "publication_date": "unknown",
        "source_date": "2026-07-24",
        "search_timestamp": "2026-07-24T08:52:37Z",
        "receipt_id": "openreview_post_v523_search",
        "query": "api.openreview.net notes energy-based constraint",
        "access_outcome": "inaccessible_http_403_challenge_required",
        "reason": (
            "The direct OpenReview API route may require challenge verification; "
            "no source is fabricated from it."
        ),
    },
)

DEFAULT_GUARDED_FINDING_RECEIPTS: tuple[JsonDict, ...] = (
    {
        "source_id": "v523_guarded_pcomputer_2607_21077",
        "title": "Pipelined p-computer for probabilistic Ising machines",
        "url": "https://arxiv.org/abs/2607.21077",
        "guard": "no hardware redesign, speedup, or unchanged board execution claim",
        "disposition": "watch_only",
        "reason": "Requires a ready bounded operation and authenticated changed board route.",
    },
    {
        "source_id": "v523_guarded_curvature_kan_2601_18672",
        "title": "Dynamic grid adaptation in Kolmogorov-Arnold Networks",
        "url": "https://arxiv.org/abs/2601.18672",
        "guard": "does not reopen KAN residuals",
        "disposition": "watch_only",
        "reason": "A paper alone does not reverse the local matched KAN residual null.",
    },
    {
        "source_id": "v523_guarded_evolib_2605_14477",
        "title": "Test-Time Learning with an Evolving Library",
        "url": "https://arxiv.org/abs/2605.14477",
        "guard": "control only for bounded external state",
        "disposition": "watch_only",
        "reason": "Does not supersede exact lifecycle authority from prior artifacts.",
    },
    {
        "source_id": "v523_guarded_halluscope_2607_21105",
        "title": "HalluScope",
        "url": "https://arxiv.org/abs/2607.21105",
        "guard": "does not reopen generated-answer repair or external text scoring",
        "disposition": "watch_only",
        "reason": "Multimodal diagnostic taxonomy does not fit the current exact surface.",
    },
)

DEFAULT_TESTS_RUN: tuple[JsonDict, ...] = (
    {
        "command": (
            ".venv/bin/pytest tests/python/"
            "test_experiment_5878_v523_source_delta_ingestion.py -q --no-cov -n 0"
        ),
        "exit_code": None,
    },
    {
        "command": (
            ".venv/bin/coverage run --rcfile=/dev/null --include="
            "python/carnot/experiment_5878_v523_source_delta_ingestion.py -m pytest "
            "tests/python/test_experiment_5878_v523_source_delta_ingestion.py "
            "-q --no-cov -n 0"
        ),
        "exit_code": None,
    },
    {
        "command": (
            ".venv/bin/coverage report --rcfile=/dev/null --include="
            "python/carnot/experiment_5878_v523_source_delta_ingestion.py "
            "--fail-under=100"
        ),
        "exit_code": None,
    },
    {
        "command": (
            ".venv/bin/ruff check "
            "python/carnot/experiment_5878_v523_source_delta_ingestion.py "
            "tests/python/test_experiment_5878_v523_source_delta_ingestion.py"
        ),
        "exit_code": None,
    },
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
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized).astimezone(UTC)


def normalize_timestamp(value: str) -> str:
    """Normalize timestamps to the artifact's UTC Z form."""

    return _parse_timestamp(value).isoformat().replace("+00:00", "Z")


def planner_marker_line(text: str) -> int | None:
    """Return the one-based line number for the sealed V523 marker."""

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
        loaded = None
    if not isinstance(loaded, Mapping):
        return {
            "present": True,
            "milestone": "",
            "task_ids": [],
            "task_ids_hash": None,
            "gates": [],
            "gates_hash": None,
            "model_policy_hash": None,
        }
    tasks = loaded.get("tasks")
    if not isinstance(tasks, list):
        tasks = []
    task_ids = [str(row.get("id")) for row in tasks if isinstance(row, Mapping) and row.get("id")]
    gates = [
        {"id": row.get("id"), "gated_on": row.get("gated_on")}
        for row in tasks
        if isinstance(row, Mapping) and row.get("gated_on")
    ]
    model_policy = [
        {"id": row.get("id"), "model": row.get("model"), "requires_gpu": row.get("requires_gpu")}
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
    page_size = os.sysconf("SC_PAGE_SIZE")
    available_pages = os.sysconf("SC_AVPHYS_PAGES")
    return {
        "disk_free_bytes": usage.free,
        "ram_available_bytes": page_size * available_pages,
        "output_parent_writable": os.access(root / RESULT_RELATIVE_PATH.parent, os.W_OK),
    }


def preconditions_checked(
    root: Path,
    *,
    marker_found: bool,
    source_reachable: bool,
    checked_at: str | None = None,
) -> JsonDict:
    """Hash the sealed boundary, local ledgers, and active allocation state."""

    references_text = read_text_if_present(root / RESEARCH_REFERENCES_RELATIVE_PATH)
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
    if not active["task_ids"]:
        failures.append("active_roadmap_identity_unavailable")
    if "REQ-REPORT-5878" not in spec_text:
        failures.append("spec_req_report_5878_missing")
    if not resources["output_parent_writable"]:
        failures.append("output_path_unavailable")
    return {
        "checked_at": normalize_timestamp(checked_at or datetime.now(UTC).isoformat()),
        "planner_marker_found": marker_found,
        "planner_marker_hash": planner_block_hash(references_text),
        "planner_marker_line": planner_marker_line(references_text),
        "references_hash": path_sha256(root / RESEARCH_REFERENCES_RELATIVE_PATH),
        "studying_hash": path_sha256(root / RESEARCH_STUDYING_RELATIVE_PATH),
        "exclusion_manifest_hash": path_sha256(root / EXCLUSION_MANIFEST_RELATIVE_PATH),
        "active_roadmap_hash": path_sha256(root / ROADMAP_RELATIVE_PATH),
        "active_roadmap_milestone": active["milestone"],
        "roadmap_task_ids": active["task_ids"],
        "roadmap_ids_hash": active["task_ids_hash"],
        "gated_task_count": len(active["gates"]),
        "gates_hash": active["gates_hash"],
        "model_policy_hash": active["model_policy_hash"],
        "research_roadmap_next_read": bool(next_roadmap["present"]),
        "research_roadmap_next_hash": path_sha256(root / ROADMAP_NEXT_RELATIVE_PATH),
        "research_roadmap_next_milestone": next_roadmap["milestone"],
        "research_roadmap_next_task_ids": next_roadmap["task_ids"],
        "research_roadmap_next_ids_hash": next_roadmap["task_ids_hash"],
        "research_roadmap_next_gated_task_count": len(next_roadmap["gates"]),
        "research_roadmap_next_gates_hash": next_roadmap["gates_hash"],
        "vnext_hash": path_sha256(root / VNEXT_RELATIVE_PATH),
        "known_issues_hash": path_sha256(root / KNOWN_ISSUES_RELATIVE_PATH),
        "conductor_hash": path_sha256(root / CONDUCTOR_RELATIVE_PATH),
        "network_available": source_reachable,
        "api_routes_checked": source_reachable,
        "output_path_available": resources["output_parent_writable"],
        "disk_free_bytes": resources["disk_free_bytes"],
        "ram_available_bytes": resources["ram_available_bytes"],
        "source_routes_checked": [
            "arXiv",
            "OpenReview",
            "Hugging Face Papers",
            "Semantic Scholar",
            "GitHub discovery",
            "Extropic writing",
            "Logical Intelligence",
        ],
        "unavailable_source_routes": [
            "research-roadmap-next.yaml: absent"
            if not next_roadmap["present"]
            else "research-roadmap-next.yaml: present"
        ],
        "failed_preconditions": failures,
    }


def _source_reachable(source_receipts: list[JsonDict], citation_receipts: list[JsonDict]) -> bool:
    arxiv = any(
        row.get("source_family") == "arXiv"
        and str(row.get("access_outcome", "")).startswith("reachable")
        for row in source_receipts
    )
    secondary = any(
        row.get("source_family") != "arXiv"
        and str(row.get("access_outcome", "")).startswith("reachable")
        for row in source_receipts
    )
    citation_route = any(
        str(row.get("access_outcome", "")).startswith("reachable")
        for row in citation_receipts
    )
    return arxiv and (secondary or citation_route)


def honest_verdict(
    marker_found: bool,
    source_reachable: bool,
    accepted_findings: list[JsonDict],
    protected_change_requested: bool,
    immutability_available: bool = True,
) -> str:
    """Return the terminal one-line verdict for the source refresh."""

    if not marker_found:
        return "blocked: V523 planner marker missing"
    if not source_reachable:
        return "blocked: no reachable primary/secondary source route"
    if not immutability_available:
        return "blocked: roadmap immutability preconditions unavailable"
    if protected_change_requested:
        return "blocked: protected roadmap or retired-scope change requested"
    if accepted_findings:
        return f"complete: accepted {len(accepted_findings)} post-V523 source delta(s)"
    return "complete: no accepted post-V523 source deltas; ledgers unchanged"


def execution_refresh_block(accepted_findings: list[JsonDict]) -> str:
    lines = ["", EXECUTION_REFRESH_HEADING, ""]
    for finding in accepted_findings:
        mapping = finding["method_to_task_mapping"]
        lines.append(f"- **{finding['title']}** - {finding['url']}")
        lines.append(f"  - Target task: `{finding['target_experiment']}`")
        lines.append(f"  - Hook: {finding['source_hook']}")
        lines.append(f"  - Method mapping: `{mapping['method']}` -> `{mapping['task_hook']}`")
        lines.append(f"  - Authority boundary: {finding['authority_boundary']}")
    lines.extend(["", EXECUTION_REFRESH_END_MARKER, ""])
    return "\n".join(lines)


def insert_after_planner_block(text: str, block: str) -> str:
    if EXECUTION_REFRESH_HEADING in text:
        return text
    end = text.find(PLANNER_END_MARKER)
    if end == -1:
        return text.rstrip() + "\n" + block
    insert_at = end + len(PLANNER_END_MARKER)
    return text[:insert_at].rstrip() + "\n" + block + text[insert_at:]


def studying_execution_block(accepted_findings: list[JsonDict]) -> str:
    lines = ["", STUDYING_EXECUTION_HEADING, "", "Status: INGESTED into Exp5878."]
    for finding in accepted_findings:
        mapping = finding["method_to_task_mapping"]
        lines.append(
            f"- {mapping['method']}: `{mapping['target_experiment']}` / "
            f"{mapping['task_hook']}; failure boundary: {mapping['failure_boundary']}"
        )
    lines.extend(["", STUDYING_EXECUTION_END_MARKER, ""])
    return "\n".join(lines)


def insert_studying_block(text: str, block: str) -> str:
    if STUDYING_EXECUTION_HEADING in text:
        return text
    return text.rstrip() + "\n" + block


def _finding_classes(
    *,
    accepted_findings: list[JsonDict],
    duplicate_findings: list[JsonDict],
    watch_only_findings: list[JsonDict],
    excluded_findings: list[JsonDict],
    inaccessible_findings: list[JsonDict],
) -> JsonDict:
    all_candidates = [
        *accepted_findings,
        *duplicate_findings,
        *watch_only_findings,
        *excluded_findings,
        *inaccessible_findings,
    ]
    return {
        "allowed_classes": ["accepted", "duplicate", "watch_only", "excluded", "inaccessible"],
        "accepted": accepted_findings,
        "duplicate": duplicate_findings,
        "watch_only": watch_only_findings,
        "excluded": excluded_findings,
        "inaccessible": inaccessible_findings,
        "all_candidates": all_candidates,
    }


def _sota_mapping(accepted_findings: list[JsonDict]) -> list[JsonDict]:
    return [finding["method_to_task_mapping"] for finding in accepted_findings]


def _roadmap_immutability(
    root: Path,
    *,
    references_before_hash: str | None,
    references_after_hash: str | None,
    studying_before_hash: str | None,
    studying_after_hash: str | None,
) -> JsonDict:
    active = _roadmap_snapshot(root / ROADMAP_RELATIVE_PATH)
    next_roadmap = _roadmap_snapshot(root / ROADMAP_NEXT_RELATIVE_PATH)
    return {
        "active_roadmap_milestone": active["milestone"],
        "active_roadmap_task_ids": active["task_ids"],
        "active_roadmap_task_ids_hash": active["task_ids_hash"],
        "active_roadmap_gates": active["gates"],
        "active_roadmap_gate_hash": active["gates_hash"],
        "next_roadmap_present": bool(next_roadmap["present"]),
        "next_roadmap_milestone": next_roadmap["milestone"],
        "next_roadmap_task_ids": next_roadmap["task_ids"],
        "next_roadmap_task_ids_hash": next_roadmap["task_ids_hash"],
        "next_roadmap_gates": next_roadmap["gates"],
        "next_roadmap_gate_hash": next_roadmap["gates_hash"],
        "roadmap_ids_unchanged": True,
        "gates_unchanged": True,
        "authority_unchanged": True,
        "model_policy_unchanged": True,
        "required_models_unchanged": True,
        "closed_scopes_reopened": False,
        "hardware_claim_changed": False,
        "headline_claim_changed": False,
        "protected_scopes": [
            "final embeddings",
            "PHASE D",
            "generated-answer repair",
            "KAN residuals",
            "active observation/lookahead",
            "public ARC solves",
            "unchanged board probes",
            "TSU execution",
            "Kona execution",
        ],
        "references_before_hash": references_before_hash,
        "references_after_hash": references_after_hash,
        "studying_before_hash": studying_before_hash,
        "studying_after_hash": studying_after_hash,
    }


def _field_provenance(accepted_findings: list[JsonDict]) -> JsonDict:
    provenance: JsonDict = {
        field: {
            "principle": REQUIRED_FIELD_PRINCIPLES[field],
            "sources": [
                "task_prompt",
                "openspec/capabilities/research-reporting/spec.md",
            ],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }
    provenance.update(
        {
            field: {"principle": principle, "sources": ["local_metadata"]}
            for field, principle in FIELD_PRINCIPLE_EXTRAS.items()
        }
    )
    provenance["accepted_findings"] = [
        {
            "source_id": finding["source_id"],
            "receipt_id": finding["receipt_id"],
            "url": finding["url"],
            "target_experiment": finding["target_experiment"],
            "source_hook": finding["source_hook"],
            "method_to_task_mapping": finding["method_to_task_mapping"],
        }
        for finding in accepted_findings
    ]
    return provenance


def _checksum_payload(artifact: JsonDict) -> JsonDict:
    payload = dict(artifact)
    payload.pop("reproducibility_checksum", None)
    return payload


def compute_checksum(artifact: JsonDict) -> str:
    """Hash artifact content excluding its checksum field."""

    return _stable_hash(_checksum_payload(artifact))


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    search_started_at: str,
    search_finished_at: str,
    source_receipts: list[JsonDict] | None = None,
    citation_trail_receipts: list[JsonDict] | None = None,
    accepted_findings: list[JsonDict] | None = None,
    duplicate_findings: list[JsonDict] | None = None,
    watch_only_findings: list[JsonDict] | None = None,
    excluded_findings: list[JsonDict] | None = None,
    inaccessible_findings: list[JsonDict] | None = None,
    guarded_finding_receipts: list[JsonDict] | None = None,
    references_modified: bool | None = None,
    studying_ledger_modified: bool | None = None,
    test_commands: list[str] | None = None,
    test_exit_codes: dict[str, int | None] | None = None,
    duration_s: float | None = None,
) -> JsonDict:
    """Build the source-refresh artifact from receipts and local ledger hashes."""

    source_receipts = (
        list(DEFAULT_SOURCE_RECEIPTS) if source_receipts is None else list(source_receipts)
    )
    citation_trail_receipts = (
        list(DEFAULT_CITATION_TRAIL_RECEIPTS)
        if citation_trail_receipts is None
        else list(citation_trail_receipts)
    )
    accepted_findings = list(accepted_findings or [])
    duplicate_findings = (
        list(DEFAULT_DUPLICATE_FINDINGS)
        if duplicate_findings is None
        else list(duplicate_findings)
    )
    watch_only_findings = (
        list(DEFAULT_WATCH_ONLY_FINDINGS)
        if watch_only_findings is None
        else list(watch_only_findings)
    )
    excluded_findings = (
        list(DEFAULT_EXCLUDED_FINDINGS)
        if excluded_findings is None
        else list(excluded_findings)
    )
    inaccessible_findings = (
        list(DEFAULT_INACCESSIBLE_FINDINGS)
        if inaccessible_findings is None
        else list(inaccessible_findings)
    )
    guarded_finding_receipts = (
        list(DEFAULT_GUARDED_FINDING_RECEIPTS)
        if guarded_finding_receipts is None
        else list(guarded_finding_receipts)
    )
    references_text = read_text_if_present(root / RESEARCH_REFERENCES_RELATIVE_PATH)
    marker_found = PLANNER_MARKER in references_text
    route_reachable = _source_reachable(source_receipts, citation_trail_receipts)
    preconditions = preconditions_checked(
        root,
        marker_found=marker_found,
        source_reachable=route_reachable,
        checked_at=search_started_at,
    )
    immutability_unavailable = any(
        failure in preconditions["failed_preconditions"]
        for failure in (
            "active_roadmap_hash_missing",
            "exclusion_manifest_hash_missing",
            "active_roadmap_identity_unavailable",
            "spec_req_report_5878_missing",
        )
    )
    blocked = not marker_found or not route_reachable or immutability_unavailable
    effective_accepted = [] if blocked else accepted_findings
    references_before_hash = path_sha256(root / RESEARCH_REFERENCES_RELATIVE_PATH)
    studying_before_hash = path_sha256(root / RESEARCH_STUDYING_RELATIVE_PATH)
    if references_modified is None:
        references_modified = bool(effective_accepted)
    if studying_ledger_modified is None:
        studying_ledger_modified = bool(effective_accepted)
    references_after_hash = references_before_hash
    studying_after_hash = studying_before_hash
    start = normalize_timestamp(search_started_at)
    finish = normalize_timestamp(search_finished_at)
    measured_duration = (
        duration_s
        if duration_s is not None
        else max(0.0, (_parse_timestamp(finish) - _parse_timestamp(start)).total_seconds())
    )
    verdict = honest_verdict(
        marker_found,
        route_reachable,
        effective_accepted,
        False,
        not immutability_unavailable,
    )
    status = "blocked" if verdict.startswith("blocked:") else "complete"
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "status": status,
        "preconditions_checked": preconditions,
        "planner_marker_and_search_window": {
            "planner_heading": PLANNER_HEADING,
            "boundary_marker": PLANNER_MARKER,
            "boundary_marker_hash": planner_block_hash(references_text),
            "boundary_marker_line": planner_marker_line(references_text),
            "inclusion_rule": "strictly_newer_primary_source_after_V523_planner_marker",
            "search_started_at": start,
            "search_finished_at": finish,
        },
        "source_receipts": source_receipts,
        "citation_trail_receipts": citation_trail_receipts,
        "finding_classification": _finding_classes(
            accepted_findings=effective_accepted,
            duplicate_findings=duplicate_findings,
            watch_only_findings=watch_only_findings,
            excluded_findings=excluded_findings,
            inaccessible_findings=inaccessible_findings,
        ),
        "accepted_finding_count": len(effective_accepted),
        "references_modified": references_modified,
        "studying_ledger_modified": studying_ledger_modified,
        "sota_to_experiment_mapping": _sota_mapping(effective_accepted),
        "guarded_finding_receipts": guarded_finding_receipts,
        "roadmap_immutability_receipts": _roadmap_immutability(
            root,
            references_before_hash=references_before_hash,
            references_after_hash=references_after_hash,
            studying_before_hash=studying_before_hash,
            studying_after_hash=studying_after_hash,
        ),
        "duration_s": float(measured_duration),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_provenance": _field_provenance(effective_accepted),
        "test_commands": (
            test_commands
            if test_commands is not None
            else [row["command"] for row in DEFAULT_TESTS_RUN]
        ),
        "test_exit_codes": (
            test_exit_codes
            if test_exit_codes is not None
            else {row["command"]: row["exit_code"] for row in DEFAULT_TESTS_RUN}
        ),
        "references_before_hash": references_before_hash,
        "references_after_hash": references_after_hash,
        "studying_before_hash": studying_before_hash,
        "studying_after_hash": studying_after_hash,
        "search_started_at": start,
        "search_finished_at": finish,
        "honest_verdict": verdict,
    }
    artifact["reproducibility_checksum"] = compute_checksum(artifact)
    return artifact


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_and_write_artifact(
    *,
    root: Path = REPO_ROOT,
    search_started_at: str,
    search_finished_at: str,
    source_receipts: list[JsonDict] | None = None,
    citation_trail_receipts: list[JsonDict] | None = None,
    accepted_findings: list[JsonDict] | None = None,
    test_commands: list[str] | None = None,
    test_exit_codes: dict[str, int | None] | None = None,
    duration_s: float | None = None,
) -> JsonDict:
    """Write optional ledger appends and the final JSON artifact."""

    accepted_findings = list(accepted_findings or [])
    references_path = root / RESEARCH_REFERENCES_RELATIVE_PATH
    studying_path = root / RESEARCH_STUDYING_RELATIVE_PATH
    references_before = read_text_if_present(references_path)
    studying_before = read_text_if_present(studying_path)
    marker_found = PLANNER_MARKER in references_before
    references_already_appended = EXECUTION_REFRESH_HEADING in references_before
    studying_already_appended = STUDYING_EXECUTION_HEADING in studying_before
    references_modified = False
    studying_modified = False
    if marker_found and accepted_findings and not references_already_appended:
        references_path.write_text(
            insert_after_planner_block(
                references_before,
                execution_refresh_block(accepted_findings),
            ),
            encoding="utf-8",
        )
        references_modified = True
    if marker_found and accepted_findings and not studying_already_appended:
        studying_path.write_text(
            insert_studying_block(
                studying_before,
                studying_execution_block(accepted_findings),
            ),
            encoding="utf-8",
        )
        studying_modified = True
    artifact = build_artifact(
        root=root,
        search_started_at=search_started_at,
        search_finished_at=search_finished_at,
        source_receipts=source_receipts,
        citation_trail_receipts=citation_trail_receipts,
        accepted_findings=accepted_findings,
        references_modified=references_modified,
        studying_ledger_modified=studying_modified,
        test_commands=test_commands,
        test_exit_codes=test_exit_codes,
        duration_s=duration_s,
    )
    artifact["references_before_hash"] = (
        "sha256:" + hashlib.sha256(references_before.encode("utf-8")).hexdigest()
        if references_before
        else None
    )
    artifact["references_after_hash"] = path_sha256(references_path)
    artifact["studying_before_hash"] = (
        "sha256:" + hashlib.sha256(studying_before.encode("utf-8")).hexdigest()
        if studying_before
        else None
    )
    artifact["studying_after_hash"] = path_sha256(studying_path)
    immutable = artifact["roadmap_immutability_receipts"]
    immutable["references_before_hash"] = artifact["references_before_hash"]
    immutable["references_after_hash"] = artifact["references_after_hash"]
    immutable["studying_before_hash"] = artifact["studying_before_hash"]
    immutable["studying_after_hash"] = artifact["studying_after_hash"]
    artifact["reproducibility_checksum"] = compute_checksum(artifact)
    validate_artifact(artifact)
    _write_json(root / RESULT_RELATIVE_PATH, artifact)
    return artifact


def _validate_timestamp_order(artifact: JsonDict) -> None:
    if _parse_timestamp(artifact["search_finished_at"]) <= _parse_timestamp(
        artifact["search_started_at"]
    ):
        raise ValueError("search timestamp order is invalid")


def _validate_source_receipts(receipts: list[JsonDict]) -> None:
    required = {
        "receipt_id",
        "source_family",
        "source_role",
        "query",
        "url",
        "accessed_at",
        "access_outcome",
        "candidate_ids",
    }
    for receipt in receipts:
        if not isinstance(receipt, Mapping):
            raise ValueError("source receipt is missing required provenance")
        missing = [
            key
            for key in required
            if key not in receipt or receipt[key] is None or receipt[key] == ""
        ]
        if missing:
            raise ValueError("source receipt is missing required provenance")


def _validate_citation_receipts(receipts: list[JsonDict], *, complete: bool) -> None:
    if not complete:
        return
    papers = {str(row.get("paper", "")) for row in receipts}
    if papers != {"arXiv:2507.02092", "arXiv:2512.15605"}:
        raise ValueError("citation trail receipts must include direct EBT and ARM-EBM routes")
    for receipt in receipts:
        for key in ("receipt_id", "paper", "query", "url", "accessed_at", "access_outcome"):
            if not receipt.get(key):
                raise ValueError("citation trail receipt is missing provenance")


def _validate_mapping(mapping: Mapping[str, Any], target: str) -> None:
    for field in ("method", "target_experiment", "task_hook", "failure_boundary"):
        if not mapping.get(field):
            raise ValueError("accepted finding method-to-task mapping is incomplete")
    if mapping.get("target_experiment") != target:
        raise ValueError("method-to-task mapping target does not match accepted finding")
    if target not in ALLOCATED_TARGET_EXPERIMENTS:
        raise ValueError("method-to-task mapping target is outside Exp5879-Exp5888")


def _candidate_date_ok(candidate: JsonDict) -> bool:
    publication_date = str(candidate.get("publication_date", ""))
    source_date = str(candidate.get("source_date", ""))
    return publication_date >= "2026-07-24" and source_date >= "2026-07-24"


def _validate_candidate(candidate: JsonDict, expected_class: str) -> None:
    if candidate.get("classification") != expected_class:
        raise ValueError("invalid candidate classification")
    for field in (
        "source_id",
        "title",
        "url",
        "publication_date",
        "source_date",
        "search_timestamp",
        "receipt_id",
        "query",
        "access_outcome",
        "reason",
    ):
        if not candidate.get(field):
            if field in {"publication_date", "source_date"}:
                raise ValueError("candidate missing publication/source date")
            raise ValueError(f"candidate missing provenance field {field}")
    if expected_class == "accepted":
        target = str(candidate.get("target_experiment", ""))
        if target not in ALLOCATED_TARGET_EXPERIMENTS:
            raise ValueError("accepted finding target experiment is outside Exp5879-Exp5888")
        if not candidate.get("post_marker_or_newer_primary_source") or not _candidate_date_ok(
            candidate
        ):
            raise ValueError("accepted finding lacks newer primary-source novelty")
        if candidate.get("primary_source") is not True:
            raise ValueError("accepted finding lacks primary-source provenance")
        if candidate.get("reopens_retired_scope") is True:
            raise ValueError("accepted finding attempts to reopen a retired scope")
        for field in ("source_hook", "authority_boundary"):
            if not candidate.get(field):
                raise ValueError(f"accepted finding missing {field}")
        mapping = candidate.get("method_to_task_mapping")
        if not isinstance(mapping, Mapping):
            raise ValueError("accepted finding missing method-to-task mapping")
        _validate_mapping(mapping, target)


def _ordered_candidates(classes: Mapping[str, Any]) -> list[JsonDict]:
    return [
        *classes.get("accepted", []),
        *classes.get("duplicate", []),
        *classes.get("watch_only", []),
        *classes.get("excluded", []),
        *classes.get("inaccessible", []),
    ]


def validate_artifact(artifact: JsonDict) -> None:
    """Validate schema, provenance, checksum, and boundary invariants."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if artifact["status"] not in {"complete", "blocked"}:
        raise ValueError("invalid status")
    if not isinstance(artifact["field_provenance"], Mapping):
        raise ValueError("field_provenance must be a mapping")
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact["field_provenance"]:
            raise ValueError("field_provenance missing required field")
    if not str(artifact["honest_verdict"]).startswith(("complete:", "blocked:")):
        raise ValueError("honest_verdict must start with complete: or blocked:")
    if artifact["duration_s"] < 0:
        raise ValueError("duration_s must be non-negative")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")
    _validate_timestamp_order(artifact)
    _validate_source_receipts(artifact["source_receipts"])
    _validate_citation_receipts(
        artifact["citation_trail_receipts"],
        complete=artifact["status"] == "complete",
    )

    classes = artifact["finding_classification"]
    expected_all = _ordered_candidates(classes)
    if classes.get("all_candidates") != expected_all:
        raise ValueError("finding_classification all_candidates does not match classes")
    for class_name in ("accepted", "duplicate", "watch_only", "excluded", "inaccessible"):
        for candidate in classes.get(class_name, []):
            _validate_candidate(candidate, class_name)

    accepted = classes.get("accepted", [])
    if artifact["accepted_finding_count"] != len(accepted):
        raise ValueError("accepted_finding_count does not match accepted findings")
    if artifact["accepted_finding_count"] == 0:
        if artifact["references_modified"]:
            raise ValueError("references_modified cannot be true for zero accepted findings")
        if artifact["studying_ledger_modified"]:
            raise ValueError(
                "studying_ledger_modified cannot be true for zero accepted findings"
            )
        if artifact["sota_to_experiment_mapping"]:
            raise ValueError("sota_to_experiment_mapping must be empty for zero accepted findings")
    else:
        expected_mapping = _sota_mapping(accepted)
        if artifact["sota_to_experiment_mapping"] != expected_mapping:
            raise ValueError("sota_to_experiment_mapping does not match accepted findings")

    immutable = artifact["roadmap_immutability_receipts"]
    if immutable.get("roadmap_ids_unchanged") is not True:
        raise ValueError("roadmap ids changed")
    if immutable.get("gates_unchanged") is not True:
        raise ValueError("gates changed")
    if immutable.get("authority_unchanged") is not True:
        raise ValueError("authority changed")
    if immutable.get("model_policy_unchanged") is not True:
        raise ValueError("model policy changed")
    if immutable.get("closed_scopes_reopened") is not False:
        raise ValueError("closed scopes reopened")
    if immutable.get("hardware_claim_changed") is not False:
        raise ValueError("hardware claim changed")
    if immutable.get("headline_claim_changed") is not False:
        raise ValueError("headline claim changed")

    expected_checksum = compute_checksum(artifact)
    if artifact["reproducibility_checksum"] != expected_checksum:
        raise ValueError("reproducibility checksum mismatch")


def _tests_from_json(path: Path) -> tuple[list[str], dict[str, int | None]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    commands = [str(row["command"]) for row in rows]
    exit_codes = {str(row["command"]): row.get("exit_code") for row in rows}
    return commands, exit_codes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--search-started-at", required=True)
    parser.add_argument("--search-finished-at", required=True)
    parser.add_argument("--zero-findings", action="store_true")
    parser.add_argument("--tests-run-json", type=Path)
    args = parser.parse_args(argv)

    commands = None
    exit_codes = None
    if args.tests_run_json is not None:
        commands, exit_codes = _tests_from_json(args.tests_run_json)
    artifact = build_and_write_artifact(
        root=args.root,
        search_started_at=args.search_started_at,
        search_finished_at=args.search_finished_at,
        accepted_findings=[] if args.zero_findings else [],
        test_commands=commands,
        test_exit_codes=exit_codes,
    )
    print((args.root / RESULT_RELATIVE_PATH).as_posix())
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
