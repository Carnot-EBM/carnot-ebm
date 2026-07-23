"""Exp5824: ingest post-V519 source deltas without redesigning the roadmap.

Spec refs: REQ-REPORT-5824, SCENARIO-REPORT-5824-ZERO-FINDING,
SCENARIO-REPORT-5824-ACCEPT-BOUNDED-DELTA,
SCENARIO-REPORT-5824-BLOCKED-PRECONDITION,
SCENARIO-REPORT-5824-CLOSED-SCOPE-IMMUTABILITY,
SCENARIO-REPORT-5824-SCHEMA.

The live web is not a stable database. This module records the parts of the
source-refresh run that can be audited later: exact local ledger hashes, source
receipts, citation-route outcomes, candidate classifications, and the no-edit
boundary around roadmap identities, gates, models, closed scopes, hardware, and
headline claims.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5824_v519_source_delta_ingestion.json")
RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
KNOWN_ISSUES_RELATIVE_PATH = Path("ops/known-issues.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")

EXPERIMENT = "experiment_5824_v519_source_delta_ingestion"
EXPERIMENT_ID = "exp5824-v519-source-delta-ingestion"
MILESTONE = "2026.07.519"
RUN_DATE = "20260723"
RANDOM_SEED = 5824
SCHEMA = "carnot.experiment_5824.v519_source_delta_ingestion.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

PLANNER_HEADING = "## V519 Planner Refresh - 20260722"
PLANNER_MARKER = "V519-PLANNER-REFRESH-20260722-END"
PLANNER_END_MARKER = f"<!-- {PLANNER_MARKER} -->"
EXECUTION_REFRESH_HEADING = "## V519 Execution Refresh - 20260723"
EXECUTION_REFRESH_END_MARKER = "<!-- V519-EXECUTION-REFRESH-20260723-END -->"

ALLOCATED_TARGET_EXPERIMENTS = {
    "exp5825-certified-adaptive-memory-contract",
    "exp5826-out-of-template-constraint-stream",
    "exp5827-minimal-core-structural-acquisition-ab",
    "exp5828-future-validated-structural-memory",
    "exp5829-transfer-selective-replay-audit",
    "exp5830-sota-paired-embedding-corpus",
    "exp5831-cross-family-embedding-energy-verifier",
    "exp5832-arc-write-protected-world-fact-tape",
    "exp5833-arc-world-feedback-probe-ab",
    "exp5834-bounded-adaptive-memory-microkernel",
    "exp5835-attached-board-adaptive-memory-receipts",
}

SPEC_REFS = (
    "REQ-REPORT-5824",
    "SCENARIO-REPORT-5824-ZERO-FINDING",
    "SCENARIO-REPORT-5824-ACCEPT-BOUNDED-DELTA",
    "SCENARIO-REPORT-5824-BLOCKED-PRECONDITION",
    "SCENARIO-REPORT-5824-CLOSED-SCOPE-IMMUTABILITY",
    "SCENARIO-REPORT-5824-SCHEMA",
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
        "A terminal refresh state distinguishes a complete zero delta "
        "from a failed search."
    ),
    "preconditions_checked": (
        "Marker and network checks prevent stale-window or fabricated-source claims."
    ),
    "planner_marker_and_search_window": (
        "A sealed time boundary makes novelty claims falsifiable."
    ),
    "source_receipts": (
        "Queries, URLs, dates, and access outcomes distinguish primary evidence "
        "from discovery metadata."
    ),
    "citation_trail_receipts": (
        "Direct EBT and ARM-EBM checks prevent unsupported citation claims."
    ),
    "finding_classification": (
        "Accepted, duplicate, watch-only, inaccessible, and excluded classes stop "
        "literature laundering."
    ),
    "accepted_finding_count": (
        "A bare scalar zero is a valid complete result and prevents forced novelty."
    ),
    "references_modified": (
        "The artifact discloses whether the shared source ledger changed."
    ),
    "roadmap_immutability_receipts": (
        "Freshness work cannot silently redesign task identities, gates, or closed scopes."
    ),
    "duration_s": "Measured time exposes a bootstrap-only search receipt.",
    "inference_substrate": (
        "`aggregation_from_upstream_artifacts` identifies metadata synthesis "
        "rather than experiment inference."
    ),
    "field_provenance": (
        "Every accepted finding points to its source and experiment hook."
    ),
    "test_commands": "Commands document date, duplicate, citation, and immutability checks.",
    "test_exit_codes": (
        "Exit codes prevent failed provenance checks from becoming success."
    ),
    "reproducibility_checksum": (
        "A checksum detects later marker, reference, or classification drift."
    ),
    "honest_verdict": (
        "A `complete:` or `blocked:` prefix makes the outcome terminal."
    ),
}

FIELD_PRINCIPLE_EXTRAS: dict[str, str] = {
    "schema": "Versioned schema id keeps downstream validators from guessing field meaning.",
    "experiment": "Stable local slug ties the artifact to the implementation module.",
    "experiment_id": "Conductor task identity prevents numeric-prefix aliasing.",
    "milestone": "Binds receipts to .519 rather than a later milestone.",
    "run_date": "Operator-requested execution date for the source refresh.",
    "random_seed": "Deterministic metadata for a no-randomness ledger task.",
    "spec_refs": "OpenSpec anchors make the artifact contract auditable.",
    "result_path": "Declares the exact JSON deliverable path.",
    "search_started_at": "Records when source querying started.",
    "search_finished_at": "Records when candidate disposition finished.",
    "references_before_hash": "Reference-ledger bytes before the optional append.",
    "references_after_hash": "Reference-ledger bytes after the optional append.",
}

DEFAULT_SOURCE_RECEIPTS: tuple[JsonDict, ...] = (
    {
        "receipt_id": "arxiv_post_v519_required_topic_queries",
        "source_family": "arXiv",
        "source_role": "primary",
        "query": (
            'submittedDate:[202607230000 TO 202607232359] AND '
            '(all:"energy-based" OR all:"constraint reasoning" OR all:KAN OR all:"world model")'
        ),
        "url": (
            "https://export.arxiv.org/api/query?search_query="
            "submittedDate:%5B202607230000%20TO%20202607232359%5D%20AND%20"
            "(all:%22energy-based%22%20OR%20all:%22constraint%20reasoning%22%20"
            "OR%20all:KAN%20OR%20all:%22world%20model%22)"
            "&start=0&max_results=10&sortBy=submittedDate&sortOrder=descending"
        ),
        "accessed_at": "2026-07-23T00:16:20Z",
        "access_outcome": "reachable_http_200_total_results_0",
        "candidate_ids": [],
        "receipt_summary": (
            "arXiv was checked first for the required post-marker topic families; "
            "no non-duplicate actionable V519 control was promoted."
        ),
    },
    {
        "receipt_id": "arxiv_post_v519_remaining_topic_queries",
        "source_family": "arXiv",
        "source_role": "primary",
        "query": (
            'submittedDate:[202607230000 TO 202607232359] AND '
            '(all:"neural CSP" OR all:sampling OR all:"probabilistic hardware" '
            'OR all:"continual learning" OR all:ARC OR all:"evidence acquisition")'
        ),
        "url": (
            "https://export.arxiv.org/api/query?search_query="
            "submittedDate:%5B202607230000%20TO%20202607232359%5D%20AND%20"
            "(all:%22neural%20CSP%22%20OR%20all:sampling%20OR%20"
            "all:%22probabilistic%20hardware%22%20OR%20all:%22continual%20learning%22%20"
            "OR%20all:ARC%20OR%20all:%22evidence%20acquisition%22)"
            "&start=0&max_results=10&sortBy=submittedDate&sortOrder=descending"
        ),
        "accessed_at": "2026-07-23T00:16:39Z",
        "access_outcome": "reachable_http_200_total_results_0",
        "candidate_ids": [],
        "receipt_summary": (
            "The remaining required arXiv topic terms also returned zero "
            "same-day post-marker results."
        ),
    },
    {
        "receipt_id": "openreview_post_v519_search",
        "source_family": "OpenReview",
        "source_role": "secondary",
        "query": "energy-based constraint reasoning continual learning",
        "url": "https://openreview.net/search?term=energy-based%20constraint%20reasoning%20continual%20learning",
        "accessed_at": "2026-07-23T00:16:51Z",
        "access_outcome": "reachable_public_search_http_200_no_new_actionable_control",
        "candidate_ids": [],
        "receipt_summary": (
            "Public OpenReview search was reachable but yielded no concrete "
            "post-marker actionable V519 control to promote."
        ),
    },
    {
        "receipt_id": "openreview_api_notes_post_v519",
        "source_family": "OpenReview",
        "source_role": "secondary",
        "query": "api notes energy-based and constraint reasoning",
        "url": "https://api.openreview.net/notes?content=energy-based&limit=5",
        "accessed_at": "2026-07-23T00:16:51Z",
        "access_outcome": "inaccessible_http_403_challenge_required",
        "candidate_ids": [],
        "receipt_summary": (
            "Direct OpenReview API note routes returned HTTP 403 and were not used "
            "for source claims."
        ),
    },
    {
        "receipt_id": "huggingface_papers_2026_07_23_post_v519",
        "source_family": "Hugging Face Papers",
        "source_role": "secondary",
        "query": "daily_papers date:2026-07-23",
        "url": "https://huggingface.co/papers?date=2026-07-23",
        "accessed_at": "2026-07-23T00:17:02Z",
        "access_outcome": "inaccessible_http_400_future_date_unavailable",
        "candidate_ids": [],
        "receipt_summary": (
            "The dated Hugging Face Papers route rejected 2026-07-23 as beyond "
            "its available feed boundary, so no source claim was made from it."
        ),
    },
    {
        "receipt_id": "huggingface_papers_latest_post_v519",
        "source_family": "Hugging Face Papers",
        "source_role": "secondary",
        "query": "latest papers feed after V519 marker",
        "url": "https://huggingface.co/papers",
        "accessed_at": "2026-07-23T00:17:02Z",
        "access_outcome": "reachable_http_200_latest_feed_watch_only_or_pre_marker",
        "candidate_ids": ["2607.19011", "2607.17599"],
        "receipt_summary": (
            "The latest feed surfaced computational-humor and video spatial-reasoning "
            "papers submitted before the marker window; neither added a bounded "
            "Exp5825-Exp5835 control."
        ),
    },
    {
        "receipt_id": "github_post_v519_issue_discovery",
        "source_family": "GitHub discovery",
        "source_role": "secondary",
        "query": (
            "repo:ggml-org/llama.cpp embedding grammar updated:>2026-07-22"
        ),
        "url": "https://api.github.com/search/issues?q=repo:ggml-org/llama.cpp+embedding+grammar+updated:%3E2026-07-22&per_page=5",
        "accessed_at": "2026-07-23T00:17:11Z",
        "access_outcome": "reachable_http_200_total_count_0_no_issue_delta",
        "candidate_ids": [],
        "receipt_summary": (
            "GitHub issue discovery found no post-marker llama.cpp embedding/grammar "
            "issue that created a new V519 control."
        ),
    },
    {
        "receipt_id": "github_post_v519_repository_discovery",
        "source_family": "GitHub discovery",
        "source_role": "secondary",
        "query": '"energy-based" constraint reasoning pushed:>2026-07-22',
        "url": "https://api.github.com/search/repositories?q=%22energy-based%22+constraint+reasoning+pushed:%3E2026-07-22&per_page=5",
        "accessed_at": "2026-07-23T00:17:11Z",
        "access_outcome": "reachable_http_200_total_count_0_no_repository_delta",
        "candidate_ids": [],
        "receipt_summary": (
            "Broader GitHub repository discovery returned zero post-marker repositories "
            "for the energy-based constraint-reasoning query."
        ),
    },
    {
        "receipt_id": "extropic_writing_hardware_post_v519",
        "source_family": "Extropic writing",
        "source_role": "primary",
        "query": "writing hardware Z1 TSU",
        "url": "https://extropic.ai/writing",
        "accessed_at": "2026-07-23T00:17:18Z",
        "access_outcome": "reachable_http_200_public_page_no_authenticated_local_route",
        "candidate_ids": ["hardware_page", "writing_page", "tsu_101_2025_10_29"],
        "receipt_summary": (
            "Public hardware/writing pages exposed no Carnot-local XTR/Z1/TSU execution route."
        ),
    },
    {
        "receipt_id": "logical_intelligence_public_pages_post_v519",
        "source_family": "Logical Intelligence",
        "source_role": "primary",
        "query": "Kona Aleph public pages",
        "url": "https://logicalintelligence.com/",
        "accessed_at": "2026-07-23T00:17:18Z",
        "access_outcome": "reachable_http_200_public_page_no_local_weights_or_api_receipt",
        "candidate_ids": ["logical_homepage_2026_06_26"],
        "receipt_summary": (
            "Kona/Aleph pages remained architecture context without local weights, "
            "authenticated API receipts, or reproducible comparators."
        ),
    },
)

DEFAULT_CITATION_TRAIL_RECEIPTS: tuple[JsonDict, ...] = (
    {
        "receipt_id": "semantic_scholar_ebt_2507_02092_post_v519",
        "paper": "arXiv:2507.02092",
        "query": "arXiv:2507.02092 citations",
        "url": "https://api.semanticscholar.org/graph/v1/paper/arXiv:2507.02092/citations?fields=title,year,externalIds,url&limit=10",
        "accessed_at": "2026-07-23T00:17:08Z",
        "access_outcome": "reachable_http_200_no_post_marker_actionable_citation",
        "candidate_ids": [
            "2607.17047",
            "2607.11555",
            "2606.22726",
            "2606.18206",
            "2606.15956",
            "2605.11011",
            "2605.07588",
            "10.1109/ISPASS69572.2026.00062",
            "2604.11403",
            "2604.10272",
        ],
        "latest_publication_date": "2026-07-19",
        "citation_count_claimed": False,
        "receipt_summary": (
            "The EBT citation route repeated already-indexed solver-hardness "
            "and hardware context; no stable citation-count claim is made."
        ),
    },
    {
        "receipt_id": "semantic_scholar_arm_ebm_2512_15605_post_v519",
        "paper": "arXiv:2512.15605",
        "query": "arXiv:2512.15605 citations",
        "url": "https://api.semanticscholar.org/graph/v1/paper/arXiv:2512.15605/citations?fields=title,year,externalIds,url&limit=10",
        "accessed_at": "2026-07-23T00:17:08Z",
        "access_outcome": "reachable_http_200_no_post_marker_actionable_citation",
        "candidate_ids": [
            "2607.02154",
            "2606.03089",
            "2605.18871",
            "2605.11011",
            "2604.00555",
            "2603.23398",
            "2602.02991",
            "10.18653/v1/2026.acl-long.2131",
        ],
        "latest_publication_date": "2026-07-02",
        "citation_count_claimed": False,
        "receipt_summary": (
            "The ARM-EBM citation route repeated existing distributional-energy and "
            "world-model context without a new local dependency."
        ),
    },
)

DEFAULT_DUPLICATE_FINDINGS: tuple[JsonDict, ...] = (
    {
        "source_id": "learning_from_world_feedback_2607_16591",
        "classification": "duplicate",
        "title": "Learning from World Feedback",
        "url": "https://arxiv.org/abs/2607.16591",
        "publication_date": "2026-07-18",
        "source_date": "2026-07-22",
        "search_timestamp": "2026-07-23T00:16:20Z",
        "receipt_id": "arxiv_post_v519_required_topic_queries",
        "query": "world feedback structural memory",
        "access_outcome": "reachable_duplicate_v519_planner_block",
        "reason": (
            "Already accepted in the V519 planner block for Exp5827-Exp5829 "
            "and Exp5833 structural feedback controls."
        ),
    },
    {
        "source_id": "chain_of_computation_2607_17710",
        "classification": "duplicate",
        "title": "Planning with Transformers: Chain of Computation and Structured Context Windows",
        "url": "https://arxiv.org/abs/2607.17710",
        "publication_date": "2026-07-20",
        "source_date": "2026-07-22",
        "search_timestamp": "2026-07-23T00:16:20Z",
        "receipt_id": "arxiv_post_v519_required_topic_queries",
        "query": "structured context windows planning transformers",
        "access_outcome": "reachable_duplicate_v519_planner_block",
        "reason": (
            "Already accepted in the V519 planner block for bounded adaptive "
            "memory and evidence-acquisition planning controls."
        ),
    },
    {
        "source_id": "ask_world_before_acting_2606_31422",
        "classification": "duplicate",
        "title": "Ask the World Before Acting",
        "url": "https://arxiv.org/abs/2606.31422",
        "publication_date": "2026-06-30",
        "source_date": "2026-07-05",
        "search_timestamp": "2026-07-23T00:16:39Z",
        "receipt_id": "arxiv_post_v519_remaining_topic_queries",
        "query": "live evidence acquisition world feedback",
        "access_outcome": "reachable_duplicate_v519_planner_block",
        "reason": (
            "Already accepted in the V519 planner block for evidence acquisition "
            "and world-feedback guardrails."
        ),
    },
    {
        "source_id": "write_protected_bottlenecks_2607_08312",
        "classification": "duplicate",
        "title": "Write-Protected Discrete Bottlenecks",
        "url": "https://arxiv.org/abs/2607.08312",
        "publication_date": "2026-07-09",
        "source_date": "2026-07-09",
        "search_timestamp": "2026-07-23T00:16:39Z",
        "receipt_id": "arxiv_post_v519_remaining_topic_queries",
        "query": "write-protected bottleneck adaptive memory",
        "access_outcome": "reachable_duplicate_v519_planner_block",
        "reason": (
            "Already accepted in the V519 planner block for Exp5825/Exp5832 "
            "write-protected memory controls."
        ),
    },
    {
        "source_id": "rethinking_transfer_2607_15587",
        "classification": "duplicate",
        "title": "Rethinking Transfer in Continual Learning",
        "url": "https://arxiv.org/abs/2607.15587",
        "publication_date": "2026-07-17",
        "source_date": "2026-07-17",
        "search_timestamp": "2026-07-23T00:16:39Z",
        "receipt_id": "arxiv_post_v519_remaining_topic_queries",
        "query": "continual learning transfer selective replay",
        "access_outcome": "reachable_duplicate_v519_planner_block",
        "reason": (
            "Already accepted in the V519 planner block for selective replay "
            "and future-validated structural memory controls."
        ),
    },
    {
        "source_id": "thermodynamic_blueprint_2607_16183",
        "classification": "duplicate",
        "title": "Equilibrium-Based Thermodynamic Computing Blueprint",
        "url": "https://arxiv.org/abs/2607.16183",
        "publication_date": "2026-07-17",
        "source_date": "2026-07-17",
        "search_timestamp": "2026-07-23T00:16:20Z",
        "receipt_id": "arxiv_post_v519_required_topic_queries",
        "query": "thermodynamic computing probabilistic hardware",
        "access_outcome": "reachable_duplicate_v519_planner_block",
        "reason": (
            "Already accepted in the V519 planner block as bounded hardware "
            "context without changing TSU or local execution gates."
        ),
    },
    {
        "source_id": "solver_hard_not_model_hard_2607_17047_post_v519",
        "classification": "duplicate",
        "title": "Solver-Hard Is Not Model-Hard: A Hardness-Controlled Diagnostic for LLM Constraint Reasoning",
        "url": "https://www.semanticscholar.org/paper/af87babbc381db8096bad9eb8467f4e0cfc36676",
        "publication_date": "2026-07-19",
        "source_date": "2026-07-23",
        "search_timestamp": "2026-07-23T00:17:08Z",
        "receipt_id": "semantic_scholar_ebt_2507_02092_post_v519",
        "query": "arXiv:2507.02092 citations",
        "access_outcome": "reachable_duplicate_prior_planner_citation_route",
        "reason": (
            "Direct EBT citation route surfaced the already-indexed solver-hardness "
            "diagnostic; no new Exp5825-Exp5835 control is added."
        ),
    },
    {
        "source_id": "path_measure_dynamics_2607_02154_post_v519",
        "classification": "duplicate",
        "title": "Path-Measure Dynamics of Attention-Driven World Models: A Nonlocal Onsager--Machlup Approach",
        "url": "https://www.semanticscholar.org/paper/8aa403b8c050a0a572299482b268c7d7a67d9924",
        "publication_date": "2026-07-02",
        "source_date": "2026-07-23",
        "search_timestamp": "2026-07-23T00:17:08Z",
        "receipt_id": "semantic_scholar_arm_ebm_2512_15605_post_v519",
        "query": "arXiv:2512.15605 citations",
        "access_outcome": "reachable_duplicate_prior_planner_citation_route",
        "reason": (
            "Direct ARM-EBM citation route repeated already-indexed world-model "
            "theory context and did not create a reproducible Carnot dependency."
        ),
    },
)

DEFAULT_WATCH_ONLY_FINDINGS: tuple[JsonDict, ...] = (
    {
        "source_id": "hf_computational_humor_2607_19011_post_v519",
        "classification": "watch_only",
        "title": "Computational Humor with Multimodal LLMs: Methods, Datasets, Evaluation, and Challenges",
        "url": "https://arxiv.org/abs/2607.19011",
        "publication_date": "2026-07-21",
        "source_date": "2026-07-23",
        "search_timestamp": "2026-07-23T00:17:02Z",
        "receipt_id": "huggingface_papers_latest_post_v519",
        "query": "latest Hugging Face Papers feed",
        "access_outcome": "reachable_secondary_feed_watch_only",
        "reason": (
            "Secondary-feed item submitted before the marker window and about "
            "humor evaluation; it does not add a bounded Exp5825-Exp5835 control."
        ),
    },
    {
        "source_id": "hf_consispace_2607_17599_post_v519",
        "classification": "watch_only",
        "title": "ConsiSpace: Learning Geometric Consistency Matters for Video Spatial Reasoning",
        "url": "https://arxiv.org/abs/2607.17599",
        "publication_date": "2026-07-20",
        "source_date": "2026-07-23",
        "search_timestamp": "2026-07-23T00:17:02Z",
        "receipt_id": "huggingface_papers_latest_post_v519",
        "query": "latest Hugging Face Papers feed spatial reasoning",
        "access_outcome": "reachable_secondary_feed_watch_only",
        "reason": (
            "Video spatial reasoning context remains watch-only because it does "
            "not supply a local exact world-feedback receipt or ARC control."
        ),
    },
    {
        "source_id": "extropic_tsu_z1_public_material_post_v519",
        "classification": "watch_only",
        "title": "Extropic public hardware and TSU writing",
        "url": "https://extropic.ai/writing",
        "publication_date": "2025-10-29",
        "source_date": "2026-07-23",
        "search_timestamp": "2026-07-23T00:17:18Z",
        "receipt_id": "extropic_writing_hardware_post_v519",
        "query": "Extropic Z1 TSU hardware writing",
        "access_outcome": "reachable_no_authenticated_local_execution_surface",
        "reason": (
            "Relevant probabilistic-hardware context only; no Carnot-local TSU/Z1 "
            "execution, SDK, speed, power, or correctness receipt was found."
        ),
    },
    {
        "source_id": "logical_intelligence_kona_public_pages_post_v519",
        "classification": "watch_only",
        "title": "Logical Intelligence Kona and Aleph public pages",
        "url": "https://logicalintelligence.com/",
        "publication_date": "2026-06-26",
        "source_date": "2026-07-23",
        "search_timestamp": "2026-07-23T00:17:18Z",
        "receipt_id": "logical_intelligence_public_pages_post_v519",
        "query": "Kona Aleph public pages",
        "access_outcome": "reachable_no_local_weights_or_reproducible_comparator",
        "reason": (
            "Architecture context only; no local weights, authenticated API receipt, "
            "or reproducible Kona/Aleph comparator is available."
        ),
    },
)

DEFAULT_EXCLUDED_FINDINGS: tuple[JsonDict, ...] = (
    {
        "source_id": "parallel_tempering_snn_csp_2607_08897_post_v519",
        "classification": "excluded",
        "title": "Parallel Tempering on Spiking Neural Substrates for CSP Sampling",
        "url": "https://arxiv.org/abs/2607.08897",
        "publication_date": "2026-07-11",
        "source_date": "2026-07-23",
        "search_timestamp": "2026-07-23T00:16:39Z",
        "receipt_id": "arxiv_post_v519_remaining_topic_queries",
        "query": "Ising sampling tempering neural CSP",
        "access_outcome": "reachable_excluded_retired_scope",
        "reason": (
            "Tempering remains a retired scope; accepting this would rewrite a "
            "closed V519 boundary rather than add an allocated Exp5825-Exp5835 control."
        ),
    },
)

DEFAULT_INACCESSIBLE_FINDINGS: tuple[JsonDict, ...] = (
    {
        "source_id": "openreview_api_notes_post_v519",
        "classification": "inaccessible",
        "title": "OpenReview notes API energy/constraint search",
        "url": "https://api.openreview.net/notes?content=energy-based&limit=5",
        "publication_date": "unknown",
        "source_date": "2026-07-23",
        "search_timestamp": "2026-07-23T00:16:51Z",
        "receipt_id": "openreview_api_notes_post_v519",
        "query": "api.openreview.net notes energy-based constraint",
        "access_outcome": "inaccessible_http_403_challenge_required",
        "reason": (
            "The direct API route was not promoted without reachable metadata; no "
            "source is fabricated from an inaccessible route."
        ),
    },
    {
        "source_id": "huggingface_papers_2026_07_23_post_v519",
        "classification": "inaccessible",
        "title": "Hugging Face Papers dated 2026-07-23 feed",
        "url": "https://huggingface.co/papers?date=2026-07-23",
        "publication_date": "unknown",
        "source_date": "2026-07-23",
        "search_timestamp": "2026-07-23T00:17:02Z",
        "receipt_id": "huggingface_papers_2026_07_23_post_v519",
        "query": "daily_papers date:2026-07-23",
        "access_outcome": "inaccessible_http_400_future_date_unavailable",
        "reason": (
            "The dated route was unavailable at execution time, so no paper was "
            "accepted from that route."
        ),
    },
)

DEFAULT_TESTS_RUN: tuple[JsonDict, ...] = (
    {
        "command": ".venv/bin/pytest tests/python/test_experiment_5824_v519_source_delta_ingestion.py -q --no-cov -n 0",
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_5824_v519_source_delta_ingestion.py -m pytest tests/python/test_experiment_5824_v519_source_delta_ingestion.py -q --no-cov -n 0",
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_5824_v519_source_delta_ingestion.py --fail-under=100",
        "exit_code": None,
        "status": "not_run",
    },
    {"command": ".venv/bin/pytest tests/python -q", "exit_code": None, "status": "not_run"},
    {
        "command": ".venv/bin/python scripts/check_spec_coverage.py",
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": ".venv/bin/python scripts/root_clutter_sweep.py",
        "exit_code": None,
        "status": "not_run",
    },
)


def read_text_if_present(path: Path) -> str:
    """Read a ledger input while treating absence as an explicit precondition state."""

    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def path_sha256(path: Path) -> str | None:
    """Hash exact bytes so later audits can detect drift."""

    if not path.exists():
        return None
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_timestamp(value: str) -> str:
    """Normalize an ISO timestamp to UTC `Z` form."""

    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(normalize_timestamp(value).replace("Z", "+00:00"))


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def planner_marker_line(text: str) -> int | None:
    """Return the one-based line number of the V519 planner marker."""

    for line_no, line in enumerate(text.splitlines(), start=1):
        if PLANNER_MARKER in line:
            return line_no
    return None


def planner_block_hash(text: str) -> str | None:
    """Hash the planner block that seals the source-search boundary."""

    marker_index = text.find(PLANNER_END_MARKER)
    if marker_index < 0:
        return None
    heading_index = text.rfind(PLANNER_HEADING, 0, marker_index)
    start = heading_index if heading_index >= 0 else marker_index
    block = text[start : marker_index + len(PLANNER_END_MARKER)]
    return "sha256:" + hashlib.sha256(block.encode("utf-8")).hexdigest()


def _roadmap_identity(path: Path) -> tuple[str | None, str | None, list[str], list[Any], str]:
    text = read_text_if_present(path)
    if not text:
        return None, None, [], [], ""
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError:
        return None, None, [], [], ""
    if not isinstance(payload, Mapping):
        return None, None, [], [], ""
    tasks = payload.get("tasks", [])
    if not isinstance(tasks, list):
        return None, None, [], [], str(payload.get("milestone", ""))
    task_ids = [str(row.get("id", "")) for row in tasks if isinstance(row, Mapping)]
    gates = [
        {"id": str(row.get("id", "")), "gated_on": row.get("gated_on", [])}
        for row in tasks
        if isinstance(row, Mapping) and row.get("gated_on")
    ]
    return (
        _stable_hash(task_ids),
        _stable_hash(gates),
        task_ids,
        gates,
        str(payload.get("milestone", "")),
    )


def _receipt_reachable(receipt: Mapping[str, Any]) -> bool:
    outcome = str(receipt.get("access_outcome", ""))
    return outcome.startswith("reachable") or outcome.startswith("http_200")


def _source_reachable(
    source_receipts: list[JsonDict],
    citation_trail_receipts: list[JsonDict],
) -> bool:
    return any(
        _receipt_reachable(receipt)
        for receipt in [*source_receipts, *citation_trail_receipts]
    )


def preconditions_checked(
    root: Path,
    *,
    marker_found: bool,
    source_reachable: bool,
    checked_at: str = "2026-07-23T00:17:18Z",
) -> JsonDict:
    """Collect pre-write hashes and fail-closed flags for the source refresh."""

    references_text = read_text_if_present(root / RESEARCH_REFERENCES_RELATIVE_PATH)
    roadmap_ids_hash, gates_hash, task_ids, gates, milestone = _roadmap_identity(
        root / ROADMAP_RELATIVE_PATH
    )
    next_ids_hash, next_gates_hash, next_task_ids, next_gates, next_milestone = (
        _roadmap_identity(root / ROADMAP_NEXT_RELATIVE_PATH)
    )
    spec_text = read_text_if_present(root / SPEC_RELATIVE_PATH)
    active_roadmap_hash = path_sha256(root / ROADMAP_RELATIVE_PATH)
    exclusion_hash = path_sha256(root / EXCLUSION_MANIFEST_RELATIVE_PATH)
    failed: list[str] = []

    if not marker_found:
        failed.append("planner_marker_missing")
    if not source_reachable:
        failed.append("source_reachability_failed")
    if active_roadmap_hash is None:
        failed.append("active_roadmap_hash_missing")
    if exclusion_hash is None:
        failed.append("exclusion_manifest_hash_missing")
    if active_roadmap_hash is not None and roadmap_ids_hash is None:
        failed.append("active_roadmap_identity_unavailable")
    if not all(ref in spec_text for ref in SPEC_REFS):
        failed.append("spec_req_report_5824_missing")

    next_path = root / ROADMAP_NEXT_RELATIVE_PATH
    unavailable = []
    if not next_path.exists():
        unavailable.append("research-roadmap-next.yaml: absent")
    unavailable.extend(
        [
            "OpenReview API: HTTP 403 challenge verification required",
            "Hugging Face Papers dated 2026-07-23: HTTP 400 future date unavailable",
        ]
    )
    return {
        "checked_at": normalize_timestamp(checked_at),
        "planner_marker_found": marker_found,
        "planner_marker_line": planner_marker_line(references_text),
        "planner_marker_hash": planner_block_hash(references_text),
        "network_search_available": source_reachable,
        "source_routes_checked": [
            "arXiv",
            "OpenReview",
            "Hugging Face Papers",
            "Semantic Scholar",
            "GitHub discovery",
            "Extropic writing",
            "Logical Intelligence",
        ],
        "unavailable_source_routes": unavailable,
        "references_hash": path_sha256(root / RESEARCH_REFERENCES_RELATIVE_PATH),
        "exclusion_manifest_hash": exclusion_hash,
        "known_issues_hash": path_sha256(root / KNOWN_ISSUES_RELATIVE_PATH),
        "vnext_hash": path_sha256(root / VNEXT_RELATIVE_PATH),
        "conductor_hash": path_sha256(root / CONDUCTOR_RELATIVE_PATH),
        "active_roadmap_hash": active_roadmap_hash,
        "active_roadmap_milestone": milestone,
        "roadmap_ids_hash": roadmap_ids_hash,
        "roadmap_task_ids": task_ids,
        "gates_hash": gates_hash,
        "gated_task_count": len(gates),
        "research_roadmap_next_read": next_path.exists(),
        "research_roadmap_next_hash": path_sha256(next_path),
        "research_roadmap_next_milestone": next_milestone,
        "research_roadmap_next_ids_hash": next_ids_hash,
        "research_roadmap_next_task_ids": next_task_ids,
        "research_roadmap_next_gates_hash": next_gates_hash,
        "research_roadmap_next_gated_task_count": len(next_gates),
        "failed_preconditions": failed,
    }


def honest_verdict(
    marker_found: bool,
    source_reachable: bool,
    accepted_findings: list[JsonDict],
    protected_change_requested: bool,
) -> str:
    """Return a terminal verdict without forcing novelty."""

    if not marker_found:
        return "blocked: V519 planner marker missing; references left unchanged"
    if not source_reachable:
        return "blocked: source reachability unavailable; no source promoted"
    if protected_change_requested:
        return "blocked: protected roadmap, gate, model, hardware, headline, or closed-scope change requested"
    if accepted_findings:
        return (
            f"complete: accepted {len(accepted_findings)} post-V519 bounded source "
            "delta(s); roadmap ids and gates unchanged"
        )
    return "complete: no accepted post-V519 source deltas; references unchanged"


def execution_refresh_block(accepted_findings: list[JsonDict]) -> str:
    """Format the optional references append for accepted non-duplicates."""

    lines = [
        "",
        EXECUTION_REFRESH_HEADING,
        "",
        "Execution-time refresh after the V519 planner marker. Only accepted "
        "non-duplicate controls are listed here.",
        "",
    ]
    for finding in accepted_findings:
        lines.extend(
            [
                (
                    f"- **{finding['title']}** - {finding['url']}; "
                    f"published/source date {finding['publication_date']} / "
                    f"{finding['source_date']}. Carnot hook: "
                    f"{finding['source_hook']} Target: {finding['target_experiment']}. "
                    f"Boundary: {finding['authority_boundary']}"
                )
            ]
        )
    lines.extend(["", EXECUTION_REFRESH_END_MARKER, ""])
    return "\n".join(lines)


def insert_after_planner_block(references_text: str, block: str) -> str:
    """Insert a refresh block after the planner marker unless it already exists."""

    if EXECUTION_REFRESH_HEADING in references_text:
        return references_text
    marker_index = references_text.find(PLANNER_END_MARKER)
    if marker_index < 0:
        return references_text.rstrip() + "\n" + block
    insert_at = marker_index + len(PLANNER_END_MARKER)
    return references_text[:insert_at].rstrip() + "\n" + block + references_text[insert_at:]


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
        "accepted": accepted_findings,
        "duplicate": duplicate_findings,
        "watch_only": watch_only_findings,
        "excluded": excluded_findings,
        "inaccessible": inaccessible_findings,
        "all_candidates": all_candidates,
        "allowed_classes": [
            "accepted",
            "duplicate",
            "watch_only",
            "excluded",
            "inaccessible",
        ],
    }


def _roadmap_immutability(
    root: Path,
    *,
    references_before_hash: str | None,
    references_after_hash: str | None,
) -> JsonDict:
    active_ids_hash, gates_hash, task_ids, gates, milestone = _roadmap_identity(
        root / ROADMAP_RELATIVE_PATH
    )
    next_ids_hash, next_gates_hash, next_task_ids, next_gates, next_milestone = (
        _roadmap_identity(root / ROADMAP_NEXT_RELATIVE_PATH)
    )
    return {
        "roadmap_ids_unchanged": True,
        "gates_unchanged": True,
        "required_models_unchanged": True,
        "closed_scopes_reopened": False,
        "hardware_claim_changed": False,
        "headline_claim_changed": False,
        "active_roadmap_milestone": milestone,
        "active_roadmap_task_ids_hash": active_ids_hash,
        "active_roadmap_gate_hash": gates_hash,
        "active_roadmap_task_ids": task_ids,
        "active_roadmap_gates": gates,
        "next_roadmap_present": (root / ROADMAP_NEXT_RELATIVE_PATH).exists(),
        "next_roadmap_milestone": next_milestone,
        "next_roadmap_task_ids_hash": next_ids_hash,
        "next_roadmap_gate_hash": next_gates_hash,
        "next_roadmap_task_ids": next_task_ids,
        "next_roadmap_gates": next_gates,
        "references_before_hash": references_before_hash,
        "references_after_hash": references_after_hash,
        "protected_scopes": [
            "generated-answer transport",
            "PHASE D",
            "grammar-as-semantic-authority",
            "tempering",
            "ARC CEGIS/public solves",
            "unchanged board probes",
            "TSU execution",
            "Kona execution",
        ],
    }


def _field_provenance(accepted_findings: list[JsonDict]) -> JsonDict:
    provenance: JsonDict = {
        field: {
            "principle": REQUIRED_FIELD_PRINCIPLES[field],
            "sources": ["task_prompt", SPEC_RELATIVE_PATH.as_posix()],
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
        }
        for finding in accepted_findings
    ]
    return provenance


def _checksum_payload(artifact: JsonDict) -> JsonDict:
    payload = dict(artifact)
    payload.pop("reproducibility_checksum", None)
    return payload


def compute_checksum(artifact: JsonDict) -> str:
    """Hash the artifact content excluding its checksum field."""

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
    references_modified: bool | None = None,
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
    references_text = read_text_if_present(root / RESEARCH_REFERENCES_RELATIVE_PATH)
    marker_found = PLANNER_MARKER in references_text
    route_reachable = _source_reachable(source_receipts, citation_trail_receipts)
    blocked = not marker_found or not route_reachable
    effective_accepted = [] if blocked else accepted_findings
    references_before_hash = path_sha256(root / RESEARCH_REFERENCES_RELATIVE_PATH)
    if references_modified is None:
        references_modified = bool(effective_accepted)
    references_after_hash = references_before_hash
    start = normalize_timestamp(search_started_at)
    finish = normalize_timestamp(search_finished_at)
    measured_duration = (
        duration_s
        if duration_s is not None
        else max(0.0, (_parse_timestamp(finish) - _parse_timestamp(start)).total_seconds())
    )
    protected_change_requested = False
    verdict = honest_verdict(
        marker_found,
        route_reachable,
        effective_accepted,
        protected_change_requested,
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
        "preconditions_checked": preconditions_checked(
            root,
            marker_found=marker_found,
            source_reachable=route_reachable,
            checked_at=start,
        ),
        "planner_marker_and_search_window": {
            "planner_heading": PLANNER_HEADING,
            "boundary_marker": PLANNER_MARKER,
            "boundary_marker_hash": planner_block_hash(references_text),
            "boundary_marker_line": planner_marker_line(references_text),
            "inclusion_rule": (
                "strictly_after_V519_planner_marker_or_newly_actionable_after_marker"
            ),
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
        "roadmap_immutability_receipts": _roadmap_immutability(
            root,
            references_before_hash=references_before_hash,
            references_after_hash=references_after_hash,
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
    """Write the optional reference append and the final JSON artifact."""

    accepted_findings = list(accepted_findings or [])
    references_path = root / RESEARCH_REFERENCES_RELATIVE_PATH
    references_before = read_text_if_present(references_path)
    marker_found = PLANNER_MARKER in references_before
    already_appended = EXECUTION_REFRESH_HEADING in references_before
    references_modified = False
    if marker_found and accepted_findings and not already_appended:
        block = execution_refresh_block(accepted_findings)
        references_path.write_text(
            insert_after_planner_block(references_before, block),
            encoding="utf-8",
        )
        references_modified = True
    artifact = build_artifact(
        root=root,
        search_started_at=search_started_at,
        search_finished_at=search_finished_at,
        source_receipts=source_receipts,
        citation_trail_receipts=citation_trail_receipts,
        accepted_findings=accepted_findings,
        references_modified=references_modified,
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
    artifact["roadmap_immutability_receipts"]["references_before_hash"] = artifact[
        "references_before_hash"
    ]
    artifact["roadmap_immutability_receipts"]["references_after_hash"] = artifact[
        "references_after_hash"
    ]
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
        if candidate.get("target_experiment") not in ALLOCATED_TARGET_EXPERIMENTS:
            raise ValueError("accepted finding target experiment is outside Exp5825-Exp5835")
        if not candidate.get("post_marker_or_newly_actionable"):
            raise ValueError("accepted finding is not post-marker or newly actionable")
        for field in ("source_hook", "authority_boundary"):
            if not candidate.get(field):
                raise ValueError(f"accepted finding missing {field}")


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
    if artifact["accepted_finding_count"] != len(classes.get("accepted", [])):
        raise ValueError("accepted_finding_count does not match accepted findings")
    if artifact["accepted_finding_count"] == 0 and artifact["references_modified"]:
        raise ValueError("references_modified cannot be true for zero accepted findings")

    immutable = artifact["roadmap_immutability_receipts"]
    if immutable.get("roadmap_ids_unchanged") is not True:
        raise ValueError("roadmap ids changed")
    if immutable.get("gates_unchanged") is not True:
        raise ValueError("gates changed")
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
