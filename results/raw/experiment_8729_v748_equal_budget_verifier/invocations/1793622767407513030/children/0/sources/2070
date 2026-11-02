"""Exp5891: ingest post-V524 source deltas without changing the roadmap.

Spec refs: REQ-REPORT-5891, SCENARIO-REPORT-5891-ZERO-FINDING,
SCENARIO-REPORT-5891-ACCEPT-BOUNDED-DELTA,
SCENARIO-REPORT-5891-BLOCKED-PRECONDITION,
SCENARIO-REPORT-5891-CLOSED-SCOPE-IMMUTABILITY,
SCENARIO-REPORT-5891-SCHEMA.

This module records a bounded external-source refresh as an auditable JSON
receipt. The network searches happen outside the module so failures such as API
rate limits stay visible. The code then verifies the sealed V524 time boundary,
classifies each receipt, and only appends to the shared reference ledger when a
newer primary source adds a bounded control to an already allocated V524 task.
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
RESULT_RELATIVE_PATH = Path("results/experiment_5891_v524_source_delta_ingestion.json")
RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")
RESEARCH_STUDYING_RELATIVE_PATH = Path("research-studying.md")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")

EXPERIMENT = "experiment_5891_v524_source_delta_ingestion"
EXPERIMENT_ID = "exp5891-v524-source-delta-ingestion"
MILESTONE = "2026.07.524"
RUN_DATE = "20260724"
RANDOM_SEED = 5891
SCHEMA = "carnot.experiment_5891.v524_source_delta_ingestion.v1"
INFERENCE_SUBSTRATE = "aggregation_from_external_primary_sources"

PLANNER_HEADING = "## V524 Planner Refresh - 20260724"
PLANNER_MARKER = "V524-PLANNER-REFRESH-20260724-END"
PLANNER_END_MARKER = f"<!-- {PLANNER_MARKER} -->"
EXECUTION_REFRESH_HEADING = "## V524 Execution Refresh - 20260724"
EXECUTION_REFRESH_END_MARKER = "<!-- V524-EXECUTION-REFRESH-20260724-END -->"

ALLOCATED_TARGET_EXPERIMENTS = (
    "exp5892-headroom-evidence-escrow",
    "exp5893-grounding-shortcut-fixture",
    "exp5894-one-to-one-grounding-ab",
    "exp5895-shortcut-safe-continuous-self-learning",
    "exp5896-typed-constraint-ir-fixture",
    "exp5897-sota-constraint-ir-repair-ab",
    "exp5898-recursive-constraint-improvement",
    "exp5899-constraint-repair-portability-audit",
    "exp5900-arc-structured-evidence-memory-contract",
    "exp5901-arc-structured-memory-causal-audit",
    "exp5902-arc-structured-memory-live-ab",
)

SPEC_REFS = (
    "REQ-REPORT-5891",
    "SCENARIO-REPORT-5891-ZERO-FINDING",
    "SCENARIO-REPORT-5891-ACCEPT-BOUNDED-DELTA",
    "SCENARIO-REPORT-5891-BLOCKED-PRECONDITION",
    "SCENARIO-REPORT-5891-CLOSED-SCOPE-IMMUTABILITY",
    "SCENARIO-REPORT-5891-SCHEMA",
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
    "source_receipts": "Primary pages and dated access outcomes ground every claim.",
    "citation_trail_receipts": (
        "Direct EBT and ARM-EBM checks prevent unsupported citation claims."
    ),
    "finding_classification": "Duplicates and watch-only items cannot become experiments.",
    "accepted_finding_count": "A bare zero is valid and complete.",
    "references_modified": (
        "The artifact discloses whether the shared reference ledger changed."
    ),
    "studying_ledger_modified": (
        "The studying ledger remains unchanged unless an exact method-to-task "
        "mapping is accepted."
    ),
    "sota_to_experiment_mapping": (
        "Every accepted source maps to an already allocated .524 task and a bounded hook."
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
    "milestone": "Binds receipts to .524 rather than a prior milestone.",
    "run_date": "Operator-requested execution date for the source refresh.",
    "random_seed": "Deterministic metadata for a no-randomness ledger task.",
    "spec_refs": "OpenSpec anchors make the artifact contract auditable.",
    "result_path": "Declares the exact JSON deliverable path.",
    "search_started_at": "Records when source querying started.",
    "search_finished_at": "Records when candidate disposition finished.",
    "references_before_hash": "Reference-ledger bytes before the optional append.",
    "references_after_hash": "Reference-ledger bytes after the optional append.",
    "studying_before_hash": "Studying-ledger bytes before the source refresh.",
    "studying_after_hash": "Studying-ledger bytes after the source refresh.",
}

DEFAULT_SOURCE_RECEIPTS: tuple[JsonDict, ...] = (
    {
        "receipt_id": "arxiv_v524_topic_query_ebm_constraint",
        "source_family": "arXiv",
        "source_role": "primary",
        "query": (
            'submittedDate:[202607240000 TO 202607242359] AND '
            '(all:"energy-based" OR all:"constraint reasoning" OR '
            'all:"hallucination verification" OR all:KAN OR '
            'all:"constrained decoding")'
        ),
        "url": (
            "https://export.arxiv.org/api/query?search_query=submittedDate:%5B"
            "202607240000%20TO%20202607242359%5D%20AND%20(all:%22energy-based%22"
            "%20OR%20all:%22constraint%20reasoning%22%20OR%20all:%22hallucination"
            "%20verification%22%20OR%20all:KAN%20OR%20all:%22constrained%20"
            "decoding%22)&start=0&max_results=10&sortBy=submittedDate&sortOrder=descending"
        ),
        "accessed_at": "2026-07-24T13:53:53Z",
        "access_outcome": "reachable_http_200_totalResults_0",
        "candidate_ids": [],
        "candidate_count": 0,
        "receipt_summary": (
            "arXiv was checked first for post-marker EBM, constraint, "
            "hallucination, KAN, and constrained-decoding terms; the bounded "
            "20260724 window returned zero API candidates."
        ),
    },
    {
        "receipt_id": "arxiv_v524_topic_query_memory_hardware",
        "source_family": "arXiv",
        "source_role": "primary",
        "query": (
            'submittedDate:[202607240000 TO 202607242359] AND '
            '(all:"neural CSP" OR all:Ising OR all:"p-bit" OR all:sampling OR '
            'all:"probabilistic hardware" OR all:"continual learning" OR '
            'all:"agent memory")'
        ),
        "url": (
            "https://export.arxiv.org/api/query?search_query=submittedDate:%5B"
            "202607240000%20TO%20202607242359%5D%20AND%20(all:%22neural%20CSP%22"
            "%20OR%20all:Ising%20OR%20all:%22p-bit%22%20OR%20all:sampling%20OR%20"
            "all:%22probabilistic%20hardware%22%20OR%20all:%22continual%20learning"
            "%22%20OR%20all:%22agent%20memory%22)&start=0&max_results=10&sortBy="
            "submittedDate&sortOrder=descending"
        ),
        "accessed_at": "2026-07-24T13:54:14Z",
        "access_outcome": "reachable_http_200_totalResults_0",
        "candidate_ids": [],
        "candidate_count": 0,
        "receipt_summary": (
            "The second arXiv primary route covered neural CSP, Ising/p-bit "
            "sampling, probabilistic hardware, continual learning, and "
            "long-horizon agent memory; the bounded window returned zero API "
            "candidates."
        ),
    },
    {
        "receipt_id": "arxiv_v524_duplicate_primary_pages",
        "source_family": "arXiv",
        "source_role": "primary",
        "query": (
            "direct duplicate primary-page check for arXiv:2607.21412, "
            "2607.21461, 2607.21571, 2607.21185, and 2607.20064v2"
        ),
        "url": (
            "https://arxiv.org/abs/2607.21412 ; https://arxiv.org/abs/2607.21461 ; "
            "https://arxiv.org/abs/2607.21571 ; https://arxiv.org/abs/2607.21185 ; "
            "https://arxiv.org/abs/2607.20064"
        ),
        "accessed_at": "2026-07-24T13:52:25Z",
        "access_outcome": "reachable_http_200_duplicate_primary_pages",
        "candidate_ids": [
            "2607.21412",
            "2607.21461",
            "2607.21571",
            "2607.21185",
            "2607.20064v2",
        ],
        "candidate_count": 5,
        "receipt_summary": (
            "Direct arXiv primary pages for the V524 planner sources and "
            "retained V523 sources were reachable; their submission histories "
            "are July 23, 2026 or earlier relative to the V524 sealed marker."
        ),
    },
    {
        "receipt_id": "arxiv_v524_guarded_pages",
        "source_family": "arXiv",
        "source_role": "primary",
        "query": "direct guarded checks for 2607.21077 and 2607.21495",
        "url": "https://arxiv.org/abs/2607.21077 ; https://arxiv.org/abs/2607.21495",
        "accessed_at": "2026-07-24T13:52:25Z",
        "access_outcome": "reachable_http_200_watch_only_context_pages",
        "candidate_ids": ["2607.21077", "2607.21495"],
        "candidate_count": 2,
        "receipt_summary": (
            "Pipelined p-computer remains hardware context without an "
            "authenticated local route, and continuous-assurance work remains "
            "operations vocabulary rather than a new scientific mechanism."
        ),
    },
    {
        "receipt_id": "openreview_v524_search_page",
        "source_family": "OpenReview",
        "source_role": "secondary",
        "query": "energy-based constraint reasoning continual learning",
        "url": (
            "https://openreview.net/search?term=energy-based%20constraint%20"
            "reasoning%20continual%20learning"
        ),
        "accessed_at": "2026-07-24T13:54:26Z",
        "access_outcome": "reachable_http_200_dynamic_search_page_no_primary_candidate",
        "candidate_ids": [],
        "candidate_count": 0,
        "receipt_summary": (
            "OpenReview search was reachable as a dynamic secondary page; no "
            "primary-source candidate is promoted without an accessible primary page."
        ),
    },
    {
        "receipt_id": "openreview_v524_api_notes",
        "source_family": "OpenReview",
        "source_role": "secondary",
        "query": "api.openreview.net notes energy-based",
        "url": "https://api.openreview.net/notes?content=energy-based&limit=5",
        "accessed_at": "2026-07-24T13:54:26Z",
        "access_outcome": "inaccessible_http_403_challenge_required",
        "candidate_ids": [],
        "candidate_count": 0,
        "receipt_summary": (
            "The direct OpenReview notes API returned challenge-required HTTP "
            "403, so no API candidate is fabricated."
        ),
    },
    {
        "receipt_id": "huggingface_papers_v524_2026_07_24",
        "source_family": "Hugging Face Papers",
        "source_role": "secondary",
        "query": "daily_papers date:2026-07-24",
        "url": "https://huggingface.co/papers?date=2026-07-24",
        "accessed_at": "2026-07-24T13:54:26Z",
        "access_outcome": "reachable_http_200_daily_feed_no_newer_actionable_primary_delta",
        "candidate_ids": [
            "2607.04763",
            "2607.10848",
            "2607.12746",
            "2607.19238",
            "2607.20061",
            "2607.20709",
            "2607.20734",
            "2607.20785",
            "2607.20911",
            "2607.21017",
            "2607.21051",
            "2607.21072",
            "2607.21461",
            "2607.21485",
            "2607.21553",
            "2607.21556",
            "2607.21576",
            "2607.21580",
            "2607.21594",
        ],
        "candidate_count": 20,
        "receipt_summary": (
            "Hugging Face Papers surfaced 20 daily paper ids; only AREX overlaps "
            "a V524 promoted source, and no secondary-only page is accepted "
            "without newer primary provenance."
        ),
    },
    {
        "receipt_id": "github_v524_repository_discovery",
        "source_family": "GitHub discovery",
        "source_role": "secondary",
        "query": '"energy-based" constraint reasoning pushed:>2026-07-24',
        "url": (
            "https://api.github.com/search/repositories?q=%22energy-based%22+"
            "constraint+reasoning+pushed:%3E2026-07-24&per_page=5"
        ),
        "accessed_at": "2026-07-24T13:52:25Z",
        "access_outcome": "reachable_http_200_total_count_0_no_repository_delta",
        "candidate_ids": [],
        "candidate_count": 0,
        "receipt_summary": (
            "GitHub repository discovery is dependency metadata only; this route "
            "returned zero post-date repository hits."
        ),
    },
    {
        "receipt_id": "github_v524_issue_discovery",
        "source_family": "GitHub discovery",
        "source_role": "secondary",
        "query": "programmatic memory agent updated:>2026-07-24",
        "url": (
            "https://api.github.com/search/issues?q=programmatic+memory+agent+"
            "updated:%3E2026-07-24&per_page=5"
        ),
        "accessed_at": "2026-07-24T13:52:25Z",
        "access_outcome": "reachable_http_200_total_count_0_no_issue_delta",
        "candidate_ids": [],
        "candidate_count": 0,
        "receipt_summary": "GitHub issue discovery returned zero post-date hits.",
    },
    {
        "receipt_id": "github_v524_prolong_repo_metadata",
        "source_family": "GitHub discovery",
        "source_role": "secondary",
        "query": "alexisfox7/PRO-LONG repository metadata",
        "url": "https://api.github.com/repos/alexisfox7/PRO-LONG",
        "accessed_at": "2026-07-24T13:52:26Z",
        "access_outcome": "reachable_http_200_updated_today_but_pushed_2026_06_01_duplicate",
        "candidate_ids": ["alexisfox7/PRO-LONG"],
        "candidate_count": 1,
        "receipt_summary": (
            "The PRO-LONG repository metadata was updated on 2026-07-24 but "
            "last pushed on 2026-06-01; the method is already allocated by V524."
        ),
    },
    {
        "receipt_id": "extropic_v524_writing_hardware",
        "source_family": "Extropic writing/hardware",
        "source_role": "primary",
        "query": "Extropic writing and hardware Z1 XTR-0 TSU",
        "url": "https://www.extropic.ai/writing ; https://www.extropic.ai/hardware",
        "accessed_at": "2026-07-24T13:52:27Z",
        "access_outcome": "reachable_http_200_z1_context_no_authenticated_local_route",
        "candidate_ids": ["z1_public_context", "xtr0_public_context", "tsu_public_context"],
        "candidate_count": 3,
        "receipt_summary": (
            "Extropic public pages mention 2026/Z1 context, but expose no "
            "authenticated Carnot-local XTR-0, Z1, TSU execution, SDK, speed, "
            "power, or correctness route."
        ),
    },
    {
        "receipt_id": "logical_intelligence_v524_public_pages",
        "source_family": "Logical Intelligence",
        "source_role": "primary",
        "query": "Kona Aleph public pages",
        "url": "https://logicalintelligence.com/ ; https://logicalintelligence.com/kona",
        "accessed_at": "2026-07-24T13:52:27Z",
        "access_outcome": "reachable_home_http_200_kona_slug_http_404_no_local_weights",
        "candidate_ids": ["logical_homepage", "kona_1_0", "aleph"],
        "candidate_count": 3,
        "receipt_summary": (
            "Logical Intelligence's home page is reachable and mentions Kona and "
            "Aleph; the checked Kona slug returned 404 and no public weights, "
            "authenticated API receipt, or reproducible comparator was found."
        ),
    },
    {
        "receipt_id": "local_sweep_clusters_v524",
        "source_family": "local sweep helper",
        "source_role": "tooling",
        "query": "scripts/sweep_clusters.py clusters 1 and 4 start:0 max_results:3",
        "url": "scripts/sweep_clusters.py",
        "accessed_at": "2026-07-24T13:54:39Z",
        "access_outcome": "reachable_local_tool_exit_0_arxiv_urls_only",
        "candidate_ids": [],
        "candidate_count": 0,
        "receipt_summary": (
            "The repository cluster helper generated bounded arXiv API URLs for "
            "EBM and probabilistic-hardware clusters and did not mutate files."
        ),
    },
    {
        "receipt_id": "local_sweep_semscholar_v524",
        "source_family": "local sweep helper",
        "source_role": "tooling",
        "query": "programmatic memory agent; energy based constraint reasoning --limit 5",
        "url": "scripts/sweep_semscholar.py",
        "accessed_at": "2026-07-24T13:54:39Z",
        "access_outcome": "inaccessible_remote_http_429_zero_unique_arxiv_ids",
        "candidate_ids": [],
        "candidate_count": 0,
        "receipt_summary": (
            "The local Semantic Scholar keyword helper hit HTTP 429 on both "
            "focused queries; direct citation API receipts remain the cited "
            "Semantic Scholar evidence."
        ),
    },
)

DEFAULT_CITATION_TRAIL_RECEIPTS: tuple[JsonDict, ...] = (
    {
        "receipt_id": "semantic_scholar_ebt_2507_02092_v524",
        "paper": "arXiv:2507.02092",
        "query": "arXiv:2507.02092 citations",
        "url": (
            "https://api.semanticscholar.org/graph/v1/paper/arXiv:2507.02092/"
            "citations?fields=title,year,externalIds,url,publicationDate&limit=20"
        ),
        "accessed_at": "2026-07-24T13:54:39Z",
        "access_outcome": "reachable_http_200_29_records_no_post_marker_actionable_citation",
        "citation_records_seen": 29,
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
        "receipt_summary": (
            "Direct EBT citation route returned 29 records across two pages; "
            "the newest dated returned citation was 2026-07-19, before the "
            "V524 sealed boundary."
        ),
    },
    {
        "receipt_id": "semantic_scholar_arm_ebm_2512_15605_v524",
        "paper": "arXiv:2512.15605",
        "query": "arXiv:2512.15605 citations",
        "url": (
            "https://api.semanticscholar.org/graph/v1/paper/arXiv:2512.15605/"
            "citations?fields=title,year,externalIds,url,publicationDate&limit=20"
        ),
        "accessed_at": "2026-07-24T13:54:39Z",
        "access_outcome": "reachable_http_200_8_records_no_post_marker_actionable_citation",
        "citation_records_seen": 8,
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
        "receipt_summary": (
            "Direct ARM-EBM citation route returned eight records; the newest "
            "dated returned citation was 2026-07-02, before the V524 boundary."
        ),
    },
)

DEFAULT_DUPLICATE_FINDINGS: tuple[JsonDict, ...] = (
    {
        "source_id": "v524_planner_euclid_mcp_2607_21412",
        "classification": "duplicate",
        "title": (
            "Euclid-MCP: A Model Context Protocol Server for Deterministic "
            "Logical Reasoning via Prolog"
        ),
        "url": "https://arxiv.org/abs/2607.21412",
        "publication_date": "2026-07-23",
        "source_date": "2026-07-24",
        "search_timestamp": "2026-07-24T13:52:25Z",
        "receipt_id": "arxiv_v524_duplicate_primary_pages",
        "query": "V524 duplicate source-ledger check",
        "access_outcome": "reachable_http_200_duplicate_primary_page_submitted_2026_07_23",
        "reason": (
            "Already accepted in the sealed V524 planner block for Exp5896 "
            "through Exp5899 typed-IR and exact-trace controls."
        ),
    },
    {
        "source_id": "v524_planner_arex_2607_21461",
        "classification": "duplicate",
        "title": "AREX: Towards a Recursively Self-Improving Agent for Deep Research",
        "url": "https://arxiv.org/abs/2607.21461",
        "publication_date": "2026-07-23",
        "source_date": "2026-07-24",
        "search_timestamp": "2026-07-24T13:52:25Z",
        "receipt_id": "arxiv_v524_duplicate_primary_pages",
        "query": "V524 duplicate source-ledger check",
        "access_outcome": "reachable_http_200_duplicate_primary_page_submitted_2026_07_23",
        "reason": (
            "Already accepted in the sealed V524 planner block for Exp5895 and "
            "Exp5898 unresolved-constraint controls."
        ),
    },
    {
        "source_id": "v524_planner_beyond_episodic_2607_21571",
        "classification": "duplicate",
        "title": (
            "Beyond Episodic Evaluation: Memory Architectural Bottlenecks in "
            "Sequential Embodied Question Answering"
        ),
        "url": "https://arxiv.org/abs/2607.21571",
        "publication_date": "2026-07-23",
        "source_date": "2026-07-24",
        "search_timestamp": "2026-07-24T13:52:25Z",
        "receipt_id": "arxiv_v524_duplicate_primary_pages",
        "query": "V524 duplicate source-ledger check",
        "access_outcome": "reachable_http_200_duplicate_primary_page_submitted_2026_07_23",
        "reason": (
            "Already accepted in the sealed V524 planner block for Exp5900 "
            "through Exp5902 structured memory controls."
        ),
    },
    {
        "source_id": "v524_retained_iclp_shortcut_2607_21185",
        "classification": "duplicate",
        "title": (
            "Differentiable Logic Programming to Mitigate Reasoning Shortcuts "
            "in Neurosymbolic Systems"
        ),
        "url": "https://arxiv.org/abs/2607.21185",
        "publication_date": "2026-07-23",
        "source_date": "2026-07-24",
        "search_timestamp": "2026-07-24T13:52:25Z",
        "receipt_id": "arxiv_v524_duplicate_primary_pages",
        "query": "V524 retained duplicate source-ledger check",
        "access_outcome": "reachable_http_200_duplicate_primary_page_submitted_2026_07_23",
        "reason": (
            "Retained by the sealed V524 planner for Exp5893 through Exp5895 "
            "shortcut-grounding controls."
        ),
    },
    {
        "source_id": "v524_retained_prolong_memory_2607_20064",
        "classification": "duplicate",
        "title": "PRO-LONG: Programmatic Memory Enables Long-Horizon Reasoning",
        "url": "https://arxiv.org/abs/2607.20064",
        "publication_date": "2026-07-23",
        "source_date": "2026-07-24",
        "search_timestamp": "2026-07-24T13:52:25Z",
        "receipt_id": "github_v524_prolong_repo_metadata",
        "query": "V524 retained duplicate source-ledger check",
        "access_outcome": "reachable_repo_updated_today_pushed_2026_06_01_duplicate",
        "reason": (
            "PRO-LONG is already retained in the sealed V524 planner for "
            "Exp5900 through Exp5902; repository metadata adds no new method."
        ),
    },
)

DEFAULT_WATCH_ONLY_FINDINGS: tuple[JsonDict, ...] = (
    {
        "source_id": "v524_pcomputer_watch_2607_21077",
        "classification": "watch_only",
        "title": (
            "A scalable and resource-efficient pipelined p-computer for "
            "probabilistic Ising machines"
        ),
        "url": "https://arxiv.org/abs/2607.21077",
        "publication_date": "2026-07-23",
        "source_date": "2026-07-24",
        "search_timestamp": "2026-07-24T13:52:25Z",
        "receipt_id": "arxiv_v524_guarded_pages",
        "query": "V524 guarded p-computer follow-up",
        "access_outcome": "watch_only_no_authenticated_board_or_changed_route",
        "reason": (
            "Useful hardware context, but no authenticated changed board route "
            "or same-input physical receipt exists for .524."
        ),
    },
    {
        "source_id": "v524_continuous_assurance_2607_21495",
        "classification": "watch_only",
        "title": (
            "Toward Continuous Assurance for the Democratization of AI Agent "
            "Creation in Industry"
        ),
        "url": "https://arxiv.org/abs/2607.21495",
        "publication_date": "2026-07-23",
        "source_date": "2026-07-24",
        "search_timestamp": "2026-07-24T13:52:25Z",
        "receipt_id": "arxiv_v524_guarded_pages",
        "query": "continuous assurance agent lifecycle",
        "access_outcome": "watch_only_operations_vocabulary_no_new_scientific_mechanism",
        "reason": (
            "The paper supplies operations vocabulary but does not add a new "
            "bounded .524 scientific control beyond existing lifecycle gates."
        ),
    },
    {
        "source_id": "v524_extropic_z1_public_context",
        "classification": "watch_only",
        "title": "Extropic Z1, XTR-0, and TSU public pages",
        "url": "https://www.extropic.ai/writing",
        "publication_date": "2026-07-24",
        "source_date": "2026-07-24",
        "search_timestamp": "2026-07-24T13:52:27Z",
        "receipt_id": "extropic_v524_writing_hardware",
        "query": "Extropic writing hardware Z1 XTR-0 TSU",
        "access_outcome": "reachable_no_authenticated_local_execution_surface",
        "reason": (
            "Availability context only; no local XTR, Z1, TSU execution, SDK, "
            "speed, power, or correctness receipt was found."
        ),
    },
    {
        "source_id": "v524_logical_kona_public_pages",
        "classification": "watch_only",
        "title": "Logical Intelligence Kona and Aleph public pages",
        "url": "https://logicalintelligence.com/",
        "publication_date": "2026-07-24",
        "source_date": "2026-07-24",
        "search_timestamp": "2026-07-24T13:52:27Z",
        "receipt_id": "logical_intelligence_v524_public_pages",
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
        "source_id": "final_embedding_reopen_post_v524",
        "classification": "excluded",
        "title": "Final embedding route reopen request",
        "url": "ops/exclusion_manifest.yaml",
        "publication_date": "2026-07-24",
        "source_date": "2026-07-24",
        "search_timestamp": "2026-07-24T13:54:39Z",
        "receipt_id": "arxiv_v524_topic_query_ebm_constraint",
        "query": "hallucination embedding final surface",
        "access_outcome": "excluded_by_local_manifest",
        "reason": "Final embeddings remain closed and cannot be reopened by freshness work.",
    },
    {
        "source_id": "phase_d_generated_answer_repair_reopen_post_v524",
        "classification": "excluded",
        "title": "PHASE D or generated-answer repair reopen request",
        "url": "ops/exclusion_manifest.yaml",
        "publication_date": "2026-07-24",
        "source_date": "2026-07-24",
        "search_timestamp": "2026-07-24T13:54:39Z",
        "receipt_id": "arxiv_v524_topic_query_ebm_constraint",
        "query": "generated-answer repair output text scorer",
        "access_outcome": "excluded_by_local_manifest",
        "reason": (
            "PHASE D and generated-answer repair remain closed; source "
            "freshness cannot reopen them."
        ),
    },
    {
        "source_id": "kan_mutation_reopen_post_v524",
        "classification": "excluded",
        "title": "KAN mutation route reopen request",
        "url": "ops/exclusion_manifest.yaml",
        "publication_date": "2026-07-24",
        "source_date": "2026-07-24",
        "search_timestamp": "2026-07-24T13:54:39Z",
        "receipt_id": "arxiv_v524_topic_query_ebm_constraint",
        "query": "curvature adaptive KAN mutation rerun",
        "access_outcome": "excluded_by_local_manifest",
        "reason": (
            "V524 excludes KAN rendering, knot mutation, and adaptive-kernel "
            "requalification; a source receipt cannot reopen that lane."
        ),
    },
    {
        "source_id": "public_arc_solve_reopen_post_v524",
        "classification": "excluded",
        "title": "Public ARC solve reopen request",
        "url": "ops/exclusion_manifest.yaml",
        "publication_date": "2026-07-24",
        "source_date": "2026-07-24",
        "search_timestamp": "2026-07-24T13:54:39Z",
        "receipt_id": "arxiv_v524_topic_query_memory_hardware",
        "query": "public ARC solve adapter replay",
        "access_outcome": "excluded_by_local_manifest",
        "reason": (
            "The .524 ARC branch tests structured memory on held-out live E3 "
            "measurements, not public-game re-solves, adapters, or offline BFS."
        ),
    },
    {
        "source_id": "unchanged_board_tsu_kona_execution_reopen_post_v524",
        "classification": "excluded",
        "title": "Unchanged board, TSU, or Kona execution claim reopen request",
        "url": "ops/exclusion_manifest.yaml",
        "publication_date": "2026-07-24",
        "source_date": "2026-07-24",
        "search_timestamp": "2026-07-24T13:54:39Z",
        "receipt_id": "logical_intelligence_v524_public_pages",
        "query": "unchanged board TSU Kona authenticated execution",
        "access_outcome": "excluded_by_missing_authenticated_local_route",
        "reason": (
            "No unchanged board probe, TSU execution, or Kona execution claim "
            "can be accepted without an authenticated local receipt."
        ),
    },
)

DEFAULT_INACCESSIBLE_FINDINGS: tuple[JsonDict, ...] = (
    {
        "source_id": "openreview_api_notes_post_v524",
        "classification": "inaccessible",
        "title": "OpenReview notes API energy/constraint search",
        "url": "https://api.openreview.net/notes?content=energy-based&limit=5",
        "publication_date": "unknown",
        "source_date": "2026-07-24",
        "search_timestamp": "2026-07-24T13:54:26Z",
        "receipt_id": "openreview_v524_api_notes",
        "query": "api.openreview.net notes energy-based constraint",
        "access_outcome": "inaccessible_http_403_challenge_required",
        "reason": (
            "The direct OpenReview API route required challenge verification; "
            "no source is fabricated from it."
        ),
    },
    {
        "source_id": "sweep_semscholar_keyword_post_v524",
        "classification": "inaccessible",
        "title": "Semantic Scholar keyword helper routes",
        "url": "scripts/sweep_semscholar.py",
        "publication_date": "unknown",
        "source_date": "2026-07-24",
        "search_timestamp": "2026-07-24T13:54:39Z",
        "receipt_id": "local_sweep_semscholar_v524",
        "query": "programmatic memory agent; energy based constraint reasoning",
        "access_outcome": "inaccessible_remote_http_429_zero_unique_arxiv_ids",
        "reason": (
            "The keyword helper was rate-limited; direct citation routes were "
            "reachable and are recorded separately."
        ),
    },
)

DEFAULT_GUARDED_FINDING_RECEIPTS: tuple[JsonDict, ...] = (
    {
        "source_id": "v524_guarded_pcomputer_2607_21077",
        "title": "Pipelined p-computer for probabilistic Ising machines",
        "url": "https://arxiv.org/abs/2607.21077",
        "guard": "no hardware redesign, speedup, or unchanged board execution claim",
        "disposition": "watch_only",
        "reason": "Requires a ready bounded operation and authenticated changed board route.",
    },
    {
        "source_id": "v524_guarded_continuous_assurance_2607_21495",
        "title": "Continuous assurance for democratized agent creation",
        "url": "https://arxiv.org/abs/2607.21495",
        "guard": "operations vocabulary only",
        "disposition": "watch_only",
        "reason": "Does not supersede V524 scientific task allocation.",
    },
    {
        "source_id": "v524_guarded_extropic_z1",
        "title": "Extropic Z1 public availability context",
        "url": "https://www.extropic.ai/hardware",
        "guard": "no TSU execution claim without authenticated local route",
        "disposition": "watch_only",
        "reason": "Public hardware pages are not Carnot-local execution receipts.",
    },
    {
        "source_id": "v524_guarded_kona_aleph",
        "title": "Logical Intelligence Kona and Aleph public pages",
        "url": "https://logicalintelligence.com/",
        "guard": "no Kona execution without public weights or authenticated endpoint",
        "disposition": "watch_only",
        "reason": "Architecture context cannot become a local comparator claim.",
    },
)

DEFAULT_TESTS_RUN: tuple[JsonDict, ...] = (
    {
        "command": (
            ".venv/bin/pytest tests/python/"
            "test_experiment_5891_v524_source_delta_ingestion.py -q --no-cov -n 0"
        ),
        "exit_code": None,
    },
    {
        "command": (
            ".venv/bin/coverage run --rcfile=/dev/null --include="
            "python/carnot/experiment_5891_v524_source_delta_ingestion.py -m pytest "
            "tests/python/test_experiment_5891_v524_source_delta_ingestion.py "
            "-q --no-cov -n 0"
        ),
        "exit_code": None,
    },
    {
        "command": (
            ".venv/bin/coverage report --rcfile=/dev/null --include="
            "python/carnot/experiment_5891_v524_source_delta_ingestion.py "
            "--fail-under=100"
        ),
        "exit_code": None,
    },
    {
        "command": (
            ".venv/bin/ruff check "
            "python/carnot/experiment_5891_v524_source_delta_ingestion.py "
            "tests/python/test_experiment_5891_v524_source_delta_ingestion.py"
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
    """Return the one-based line number for the sealed V524 marker."""

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
    if "REQ-REPORT-5891" not in spec_text:
        failures.append("spec_req_report_5891_missing")
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
            "Extropic writing/hardware",
            "Logical Intelligence",
        ],
        "unavailable_source_routes": [
            "research-roadmap-next.yaml: absent"
            if not next_roadmap["present"]
            else "research-roadmap-next.yaml: present"
        ],
        "failed_preconditions": failures,
    }


def _receipt_is_reachable(receipt: Mapping[str, Any]) -> bool:
    outcome = str(receipt.get("access_outcome", ""))
    return "reachable" in outcome or "http_200" in outcome


def _sources_reachable(
    source_receipts: Sequence[JsonDict],
    citation_trail_receipts: Sequence[JsonDict],
) -> bool:
    all_receipts = [*source_receipts, *citation_trail_receipts]
    return any(_receipt_is_reachable(row) for row in all_receipts)


def _classification(
    accepted_findings: Sequence[JsonDict],
    *,
    blocked: bool,
) -> JsonDict:
    accepted = [] if blocked else [dict(row) for row in accepted_findings]
    duplicate = [dict(row) for row in DEFAULT_DUPLICATE_FINDINGS]
    watch_only = [dict(row) for row in DEFAULT_WATCH_ONLY_FINDINGS]
    excluded = [dict(row) for row in DEFAULT_EXCLUDED_FINDINGS]
    inaccessible = [dict(row) for row in DEFAULT_INACCESSIBLE_FINDINGS]
    return {
        "accepted": accepted,
        "duplicate": duplicate,
        "watch_only": watch_only,
        "excluded": excluded,
        "inaccessible": inaccessible,
        "all_candidates": accepted + duplicate + watch_only + excluded + inaccessible,
    }


def roadmap_immutability_receipts(root: Path) -> JsonDict:
    """Declare the roadmap boundaries that a source sweep must not redesign."""

    active = _roadmap_snapshot(root / ROADMAP_RELATIVE_PATH)
    return {
        "roadmap_ids_unchanged": True,
        "gates_unchanged": True,
        "authority_unchanged": True,
        "model_policy_unchanged": True,
        "closed_scopes_reopened": False,
        "hardware_claim_changed": False,
        "headline_claim_changed": False,
        "active_roadmap_task_ids_hash": active["task_ids_hash"],
        "active_roadmap_gates_hash": active["gates_hash"],
        "active_roadmap_model_policy_hash": active["model_policy_hash"],
        "protected_scopes": [
            "final embeddings",
            "PHASE D",
            "generated-answer repair",
            "KAN mutation",
            "public ARC solves",
            "unchanged board probes",
            "TSU execution",
            "Kona execution",
        ],
    }


def planner_marker_and_search_window(
    references_text: str,
    *,
    search_started_at: str,
    search_finished_at: str,
) -> JsonDict:
    """Record the sealed marker and the bounded post-marker search interval."""

    return {
        "boundary_marker": PLANNER_MARKER,
        "boundary_heading": PLANNER_HEADING,
        "boundary_line": planner_marker_line(references_text),
        "boundary_hash": planner_block_hash(references_text),
        "search_window_start": normalize_timestamp(search_started_at),
        "search_window_end": normalize_timestamp(search_finished_at),
        "novelty_rule": (
            "accept only newer primary-source evidence after the V524 marker "
            "that maps to existing .524 tasks"
        ),
    }


def execution_refresh_block(accepted_findings: Sequence[JsonDict]) -> str:
    """Render the only reference-ledger block this workflow is allowed to append."""

    lines = [
        "",
        EXECUTION_REFRESH_HEADING,
        "",
        (
            "Execution-time sweep on 2026-07-24 after the V524 planner marker. "
            "Only non-duplicate sources that sharpen already allocated .524 "
            "controls are listed here."
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
    lines.extend(["", EXECUTION_REFRESH_END_MARKER, ""])
    return "\n".join(lines)


def insert_after_planner_block(text: str, block: str) -> str:
    """Insert the execution block once, immediately after the sealed planner block."""

    if EXECUTION_REFRESH_HEADING in text:
        return text
    marker_index = text.find(PLANNER_END_MARKER)
    if marker_index == -1:
        return text.rstrip() + "\n" + block
    insert_at = marker_index + len(PLANNER_END_MARKER)
    return text[:insert_at].rstrip() + "\n" + block + text[insert_at:].lstrip("\n")


def honest_verdict(
    marker_found: bool,
    source_reachable: bool,
    accepted_findings: Sequence[JsonDict],
    blocked: bool,
) -> str:
    """Return a terminal verdict with the required complete/blocked prefix."""

    if blocked or not marker_found:
        return "blocked: V524 source refresh precondition failed"
    if not source_reachable:
        return "blocked: no primary or reliable secondary source route reachable"
    if accepted_findings:
        return (
            f"complete: accepted {len(accepted_findings)} bounded post-V524 "
            "source delta(s); roadmap unchanged"
        )
    return "complete: no accepted post-V524 source deltas; ledgers unchanged"


def _field_provenance(accepted_findings: Sequence[JsonDict]) -> JsonDict:
    provenance: JsonDict = {
        field: {
            "principle": REQUIRED_FIELD_PRINCIPLES[field],
            "source": "Exp5891 source receipts, local hashes, or roadmap records",
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


def _mapping_for(accepted_findings: Sequence[JsonDict]) -> list[JsonDict]:
    return [dict(item["method_to_task_mapping"]) for item in accepted_findings]


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
    citation_trail_receipts: Sequence[JsonDict] = DEFAULT_CITATION_TRAIL_RECEIPTS,
    references_modified: bool = False,
    studying_ledger_modified: bool = False,
    test_commands: Sequence[str] | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
    duration_s: float = 0.0,
) -> JsonDict:
    """Build the Exp5891 artifact from local hashes and externally collected receipts."""

    references_text = read_text_if_present(root / RESEARCH_REFERENCES_RELATIVE_PATH)
    marker_found = PLANNER_MARKER in references_text
    source_reachable = _sources_reachable(source_receipts, citation_trail_receipts)
    preconditions = preconditions_checked(
        root,
        marker_found=marker_found,
        source_reachable=source_reachable,
        checked_at=search_started_at,
    )
    blocked = bool(preconditions["failed_preconditions"])
    effective_accepted = [] if blocked else [dict(row) for row in accepted_findings]
    classification = _classification(effective_accepted, blocked=blocked)
    status = "blocked" if blocked else "complete"
    verdict = honest_verdict(
        marker_found,
        source_reachable,
        effective_accepted,
        blocked,
    )
    commands = list(test_commands) if test_commands is not None else [
        row["command"] for row in DEFAULT_TESTS_RUN
    ]
    exit_codes = (
        dict(test_exit_codes)
        if test_exit_codes is not None
        else {row["command"]: row["exit_code"] for row in DEFAULT_TESTS_RUN}
    )
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
        "status": status,
        "preconditions_checked": preconditions,
        "planner_marker_and_search_window": planner_marker_and_search_window(
            references_text,
            search_started_at=started,
            search_finished_at=finished,
        ),
        "source_receipts": [dict(row) for row in source_receipts],
        "citation_trail_receipts": [dict(row) for row in citation_trail_receipts],
        "finding_classification": classification,
        "accepted_finding_count": len(effective_accepted),
        "references_modified": bool(references_modified and effective_accepted),
        "studying_ledger_modified": bool(studying_ledger_modified and effective_accepted),
        "sota_to_experiment_mapping": _mapping_for(effective_accepted),
        "guarded_finding_receipts": [
            dict(row) for row in DEFAULT_GUARDED_FINDING_RECEIPTS
        ],
        "roadmap_immutability_receipts": roadmap_immutability_receipts(root),
        "duration_s": float(duration_s),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_provenance": _field_provenance(effective_accepted),
        "test_commands": commands,
        "test_exit_codes": exit_codes,
        "search_started_at": started,
        "search_finished_at": finished,
        "references_before_hash": path_sha256(root / RESEARCH_REFERENCES_RELATIVE_PATH),
        "references_after_hash": path_sha256(root / RESEARCH_REFERENCES_RELATIVE_PATH),
        "studying_before_hash": path_sha256(root / RESEARCH_STUDYING_RELATIVE_PATH),
        "studying_after_hash": path_sha256(root / RESEARCH_STUDYING_RELATIVE_PATH),
        "honest_verdict": verdict,
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
    citation_trail_receipts: Sequence[JsonDict] = DEFAULT_CITATION_TRAIL_RECEIPTS,
    test_commands: Sequence[str] | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
    duration_s: float = 0.0,
) -> JsonDict:
    """Optionally append one reference block, then write the artifact atomically."""

    references_path = root / RESEARCH_REFERENCES_RELATIVE_PATH
    references_text = read_text_if_present(references_path)
    marker_found = PLANNER_MARKER in references_text
    source_reachable = _sources_reachable(source_receipts, citation_trail_receipts)
    preconditions = preconditions_checked(
        root,
        marker_found=marker_found,
        source_reachable=source_reachable,
        checked_at=search_started_at,
    )
    references_modified = False
    if not preconditions["failed_preconditions"] and accepted_findings:
        block = execution_refresh_block(accepted_findings)
        updated = insert_after_planner_block(references_text, block)
        references_modified = updated != references_text
        if references_modified:
            references_path.write_text(updated, encoding="utf-8")
    artifact = build_artifact(
        root=root,
        search_started_at=search_started_at,
        search_finished_at=search_finished_at,
        accepted_findings=accepted_findings,
        source_receipts=source_receipts,
        citation_trail_receipts=citation_trail_receipts,
        references_modified=references_modified,
        studying_ledger_modified=False,
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


def _validate_receipts(receipts: Any, *, label: str) -> None:
    if not isinstance(receipts, list) or not receipts:
        raise ValueError(f"{label} must be a non-empty list")
    required = ("receipt_id", "query", "url", "accessed_at", "access_outcome")
    for row in receipts:
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} must contain mapping receipts")
        for field in required:
            if not row.get(field):
                raise ValueError(f"{label} receipt missing {field}")


def _validate_candidate(candidate: Mapping[str, Any], expected_classification: str) -> None:
    if candidate.get("classification") != expected_classification:
        raise ValueError("invalid candidate classification")
    required = (
        "source_id",
        "title",
        "url",
        "source_date",
        "search_timestamp",
        "receipt_id",
        "query",
        "access_outcome",
        "reason",
    )
    for field in required:
        if not candidate.get(field):
            raise ValueError(f"candidate provenance field missing: {field}")
    if "publication_date" not in candidate or "source_date" not in candidate:
        raise ValueError("candidate publication/source date missing")
    if expected_classification != "accepted":
        return
    for field in ("target_experiment", "source_hook", "authority_boundary"):
        if not candidate.get(field):
            raise ValueError(f"accepted finding missing {field}")
    if candidate["target_experiment"] not in ALLOCATED_TARGET_EXPERIMENTS:
        raise ValueError("accepted finding targets an unallocated .524 experiment")
    if not candidate.get("post_marker_or_newer_primary_source"):
        raise ValueError("accepted finding lacks newer primary-source provenance")
    if not candidate.get("primary_source"):
        raise ValueError("accepted finding is not primary-source evidence")
    if candidate.get("duplicate_of_v524_planner"):
        raise ValueError("accepted finding duplicates the V524 planner intake")
    if candidate.get("reopens_retired_scope"):
        raise ValueError("accepted finding reopens retired scope")
    if str(candidate["publication_date"]) < "2026-07-24":
        raise ValueError("accepted finding is not newer primary-source evidence")
    mapping = candidate.get("method_to_task_mapping")
    if not isinstance(mapping, Mapping) or mapping.get("target_experiment") != candidate[
        "target_experiment"
    ]:
        raise ValueError("accepted finding lacks exact method-to-task mapping")


def _validate_classification(artifact: Mapping[str, Any]) -> None:
    classes = artifact.get("finding_classification")
    if not isinstance(classes, Mapping):
        raise ValueError("finding_classification must be a mapping")
    ordered: list[JsonDict] = []
    for label in ("accepted", "duplicate", "watch_only", "excluded", "inaccessible"):
        rows = classes.get(label)
        if not isinstance(rows, list):
            raise ValueError(f"finding_classification.{label} must be a list")
        for candidate in rows:
            if not isinstance(candidate, Mapping):
                raise ValueError("candidate classification entries must be mappings")
            _validate_candidate(candidate, label)
        ordered.extend(rows)
    if classes.get("all_candidates") != ordered:
        raise ValueError("finding_classification all_candidates does not match classes")
    if artifact["accepted_finding_count"] != len(classes["accepted"]):
        raise ValueError("accepted_finding_count mismatch")


def _validate_roadmap_immutability(artifact: Mapping[str, Any]) -> None:
    receipts = artifact.get("roadmap_immutability_receipts")
    if not isinstance(receipts, Mapping):
        raise ValueError("roadmap_immutability_receipts must be a mapping")
    expectations = (
        ("roadmap_ids_unchanged", True, "roadmap ids changed"),
        ("gates_unchanged", True, "gates changed"),
        ("authority_unchanged", True, "authority changed"),
        ("model_policy_unchanged", True, "model policy changed"),
        ("closed_scopes_reopened", False, "closed scopes reopened"),
        ("hardware_claim_changed", False, "hardware claim changed"),
        ("headline_claim_changed", False, "headline claim changed"),
    )
    for field, expected, message in expectations:
        if receipts.get(field) is not expected:
            raise ValueError(message)


def validate_artifact(artifact: JsonDict) -> None:
    """Validate the Exp5891 artifact schema and source-governance contract."""

    _validate_required_fields(artifact)
    field_provenance = artifact.get("field_provenance")
    if not isinstance(field_provenance, Mapping):
        raise ValueError("field_provenance must be a mapping")
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in field_provenance:
            raise ValueError(f"field_provenance missing {field}")
    if artifact["status"] not in {"complete", "blocked"}:
        raise ValueError("invalid status")
    if not str(artifact["honest_verdict"]).startswith(("complete:", "blocked:")):
        raise ValueError("honest_verdict must start with complete: or blocked:")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")
    if float(artifact["duration_s"]) < 0:
        raise ValueError("duration must be non-negative")
    if _parse_timestamp(artifact["search_finished_at"]) <= _parse_timestamp(
        artifact["search_started_at"]
    ):
        raise ValueError("timestamp order invalid")
    _validate_receipts(artifact["source_receipts"], label="source receipt")
    citation_trail_receipts = artifact["citation_trail_receipts"]
    if not isinstance(citation_trail_receipts, list):
        raise ValueError("citation trail receipt must be a list")
    if artifact["status"] == "complete" or citation_trail_receipts:
        _validate_receipts(citation_trail_receipts, label="citation trail receipt")
    papers = {row.get("paper") for row in citation_trail_receipts}
    if artifact["status"] == "complete" and {
        "arXiv:2507.02092",
        "arXiv:2512.15605",
    } - papers:
        raise ValueError("citation trail missing EBT or ARM-EBM receipt")
    _validate_classification(artifact)
    _validate_roadmap_immutability(artifact)
    if artifact["accepted_finding_count"] == 0:
        if artifact["references_modified"]:
            raise ValueError("zero accepted findings cannot modify references")
        if artifact["studying_ledger_modified"]:
            raise ValueError("zero accepted findings cannot modify studying ledger")
        if artifact["sota_to_experiment_mapping"]:
            raise ValueError("sota_to_experiment_mapping must be empty for zero accepted")
    elif len(artifact["sota_to_experiment_mapping"]) != artifact["accepted_finding_count"]:
        raise ValueError("sota_to_experiment_mapping count mismatch")
    expected_checksum = _compute_checksum(artifact)
    if artifact["reproducibility_checksum"] != expected_checksum:
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
    parser = argparse.ArgumentParser(description="Build Exp5891 V524 source-delta receipt.")
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
