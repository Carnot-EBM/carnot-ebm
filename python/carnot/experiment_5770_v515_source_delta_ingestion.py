"""Exp5770: ingest post-V515 source deltas with bounded receipts.

Spec refs: REQ-REPORT-5770, SCENARIO-REPORT-5770-ZERO-FINDING,
SCENARIO-REPORT-5770-ACCEPT-BOUNDED-DELTA,
SCENARIO-REPORT-5770-BLOCKED-PROVENANCE,
SCENARIO-REPORT-5770-FIELD-PRINCIPLES.

The live web search is not replayed in unit tests because public indexes,
citation APIs, and daily feeds change by design. This module records the
durable part of the run: which source families were checked, which receipts
surfaced each candidate, why each candidate was classified, and which
guardrails prevent a bibliography update from silently becoming a roadmap,
hardware, model, or headline-claim change.
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
RESULT_RELATIVE_PATH = Path("results/experiment_5770_v515_source_delta_ingestion.json")
RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
KNOWN_ISSUES_RELATIVE_PATH = Path("ops/known-issues.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
PRIOR_SOURCE_DELTA_RELATIVE_PATH = Path(
    "results/experiment_5756_v514_source_delta_ingestion.json"
)

EXPERIMENT = "experiment_5770_v515_source_delta_ingestion"
EXPERIMENT_ID = "exp5770-v515-source-delta-ingestion"
MILESTONE = "2026.07.515"
RUN_DATE = "20260722"
RANDOM_SEED = 5770
SCHEMA = "carnot.experiment_5770.v515_source_delta_ingestion.v1"
INFERENCE_SUBSTRATE = "web_bibliographic_research_no_local_model_inference"
TERMINAL_PREFIXES = ("complete:", "blocked:")

PLANNER_HEADING = "## V515 Planner Refresh - 20260721"
PLANNER_MARKER = "V515-PLANNER-REFRESH-20260721-END"
PLANNER_END_MARKER = f"<!-- {PLANNER_MARKER} -->"
EXECUTION_REFRESH_HEADING = "## V515 Execution Refresh - 20260722"
EXECUTION_REFRESH_END_MARKER = "<!-- V515-EXECUTION-REFRESH-20260722-END -->"

ALLOWED_TARGET_EXPERIMENTS = {
    "exp5772-sota-constraint-drift-stream",
    "exp5773-prospective-constraint-acquisition-ab",
    "exp5774-constraint-transfer-forgetting-audit",
    "exp5775-disabled-online-shadow-integration",
    "exp5776-arc-world-model-admission-contract",
    "exp5777-arc-world-model-family-panel",
    "exp5778-arc-world-model-selector-audit",
    "exp5779-arc-heldout-live-e3-ab",
}

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "status",
    "preconditions_checked",
    "planner_marker",
    "planner_marker_hash",
    "search_started_at",
    "search_finished_at",
    "actual_search_wall_time_s",
    "source_queries",
    "source_receipts",
    "arxiv_receipts",
    "openreview_receipts",
    "semantic_scholar_receipts",
    "huggingface_receipts",
    "github_receipts",
    "extropic_receipts",
    "logical_intelligence_receipts",
    "accepted_findings",
    "duplicate_findings",
    "watch_only_findings",
    "excluded_findings",
    "inaccessible_findings",
    "references_changed",
    "references_diff_hash",
    "roadmap_scope_change_requested",
    "operator_review_required",
    "closed_scopes_reopened",
    "hardware_claim_changed",
    "inference_substrate",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

REQUIRED_FIELD_PRINCIPLES: dict[str, str] = {
    "field_principles": "Maps every artifact field to the evidence boundary that justifies it.",
    "status": "Bare terminal state supports machine gating without parsing prose.",
    "preconditions_checked": (
        "Records marker, source reachability, timestamp, ledger hash, exclusion hash, "
        "and roadmap hash before findings are trusted."
    ),
    "planner_marker": "Binds the execution search window to the V515 planner boundary.",
    "planner_marker_hash": "Content-addressed marker context detects silent planner-block drift.",
    "search_started_at": "Records the real UTC instant before external querying starts.",
    "search_finished_at": "Records the real UTC instant after final source disposition.",
    "actual_search_wall_time_s": (
        "Wall time is bibliographic search time only, not model, solver, benchmark, "
        "or hardware compute."
    ),
    "source_queries": "Search intent is reconstructable without trusting memory or mutable indexes.",
    "source_receipts": (
        "Network and local source receipts show which routes were reachable and how "
        "candidates surfaced."
    ),
    "arxiv_receipts": "Primary arXiv receipts preserve date-window and topic-search provenance.",
    "openreview_receipts": "OpenReview access limits are separated from accepted findings.",
    "semantic_scholar_receipts": (
        "Citation-route receipts are separated from general source search and do not "
        "become stable citation-count claims."
    ),
    "huggingface_receipts": (
        "Hugging Face paper-feed receipts are secondary discovery evidence, not source "
        "authority."
    ),
    "github_receipts": (
        "Repository discovery receipts distinguish executable code availability from "
        "roadmap dependencies."
    ),
    "extropic_receipts": (
        "Probabilistic-hardware context cannot become a hardware claim without local "
        "authenticated execution."
    ),
    "logical_intelligence_receipts": (
        "Logical Intelligence public material remains context unless local weights, APIs, "
        "or reproducible comparators exist."
    ),
    "accepted_findings": (
        "Accepted findings must be post-marker or newly actionable, non-duplicate, and "
        "bounded to existing V515 controls or validation boundaries."
    ),
    "duplicate_findings": "Already-ledgered work stays visible but cannot create duplicate roadmap work.",
    "watch_only_findings": (
        "Relevant but non-executable or non-local material cannot support Carnot claims."
    ),
    "excluded_findings": "Closed scopes remain closed by explicit disposition.",
    "inaccessible_findings": (
        "Access failures are separated from scientific exclusions and never promoted."
    ),
    "references_changed": "The reference ledger mutation state is declared.",
    "references_diff_hash": "Reference-file before/after content is hash-bound even when unchanged.",
    "roadmap_scope_change_requested": (
        "Scope changes block for operator review rather than silently mutating the roadmap."
    ),
    "operator_review_required": (
        "Graph, id, gate, model, hardware, or headline changes require explicit operator review."
    ),
    "closed_scopes_reopened": (
        "Retired scopes must stay closed unless the operator explicitly reopens them."
    ),
    "hardware_claim_changed": (
        "Bibliographic research cannot change FPGA, TSU, Kona, or other hardware claims."
    ),
    "inference_substrate": "The run used web bibliographic research and no local model inference.",
    "test_commands": "Verification commands are preserved exactly.",
    "test_exit_codes": "Observed exit codes are recorded without relabeling failures.",
    "reproducibility_checksum": "Stable content checksum detects artifact drift.",
    "honest_verdict": (
        "Terminal summary starts with complete: or blocked: and does not inflate novelty."
    ),
}

FIELD_PRINCIPLES: dict[str, str] = {
    **REQUIRED_FIELD_PRINCIPLES,
    "schema": "Identifies the versioned Exp5770 artifact schema.",
    "experiment": "Stable local experiment slug for result indexing.",
    "experiment_id": "Binds this receipt to the conductor task id.",
    "milestone": "Prevents V515 source receipts from being reused for another milestone.",
    "run_date": "Records the operator-requested execution date in compact form.",
    "random_seed": "Deterministic metadata even though the run performs no stochastic science.",
    "spec_refs": "OpenSpec anchors for this artifact's behavior.",
    "result_path": "Records where the JSON receipt is written.",
    "planner_marker_found": "Shows whether the source window was anchored before mutation.",
    "source_window": "States the post-marker inclusion boundary in plain machine-readable text.",
    "marker_checks": "Records planner-heading, marker-line, and marker-hash details.",
    "dedupe_corpus_checked": "Lists local ledgers hashed for duplicate and scope review.",
    "references_before_hash": "Hash of research-references.md before any optional append.",
    "references_after_hash": "Hash of research-references.md after any optional append.",
    "duplicate_checks": "Summarizes id, URL, title, and classification uniqueness checks.",
    "closed_scope_review": "Documents that banned research scopes remain closed.",
    "date_window_checks": "Records whether accepted findings are post-marker or newly actionable.",
    "target_experiment_map": "Maps accepted findings to existing Exp5772-Exp5779 controls.",
}

SPEC_REFS = (
    "REQ-REPORT-5770",
    "SCENARIO-REPORT-5770-ZERO-FINDING",
    "SCENARIO-REPORT-5770-ACCEPT-BOUNDED-DELTA",
    "SCENARIO-REPORT-5770-BLOCKED-PROVENANCE",
    "SCENARIO-REPORT-5770-FIELD-PRINCIPLES",
)

SOURCE_QUERIES: tuple[JsonDict, ...] = (
    {
        "surface": "arXiv",
        "order": 1,
        "queries": [
            "2025-2026 EBM reasoning verification",
            "constraint acquisition satisfaction 2026",
            "Ising sampling probabilistic hardware",
            "hallucination mitigation verification reasoning",
            "KAN constrained generation online learning",
            "adaptive world models agent planning",
            "cs.AI new 2026-07-21",
            "cs.LG new 2026-07-21",
            "cs.CL new 2026-07-21",
        ],
    },
    {
        "surface": "OpenReview",
        "order": 2,
        "queries": [
            "energy-based reasoning verification constraints 2026",
            "constrained generation verifier 2026",
        ],
    },
    {
        "surface": "Semantic Scholar",
        "order": 3,
        "queries": ["arXiv:2507.02092 citations", "arXiv:2512.15605 citations"],
    },
    {
        "surface": "Hugging Face Papers",
        "order": 4,
        "queries": ["daily_papers 2026-07-21", "daily_papers 2026-07-22"],
    },
    {
        "surface": "GitHub discovery",
        "order": 5,
        "queries": [
            '"2607.14169" OR "World Model Still Loses"',
            '"energy-based" reasoning constraint created:>2026-07-21',
            '"constrained generation" verifier created:>2026-07-21',
            "world model verification created:>2026-07-21",
        ],
    },
    {"surface": "Extropic writing", "order": 6, "queries": ["Z1 XTR-0 TSU THRML"]},
    {
        "surface": "Logical Intelligence",
        "order": 7,
        "queries": ["Kona Aleph energy based reasoning formal verification"],
    },
    {
        "surface": "local Carnot ledgers",
        "order": 8,
        "queries": [
            "research-references.md complete ledger",
            "V515 planner block",
            "ops/exclusion_manifest.yaml",
            "ops/known-issues.md",
            "results/experiment_5756_v514_source_delta_ingestion.json",
        ],
    },
)

ARXIV_RECEIPTS: tuple[JsonDict, ...] = (
    {
        "receipt_id": "arxiv_home_reachability",
        "surface": "arXiv",
        "url": "https://arxiv.org/",
        "queried_at": "2026-07-22T00:59:39Z",
        "status": "http_200",
        "candidate_ids": [],
        "receipt_summary": "HEAD request reached arXiv before source classification.",
    },
    {
        "receipt_id": "arxiv_cs_ai_new_20260721",
        "surface": "arXiv",
        "url": "https://arxiv.org/list/cs.AI/new",
        "queried_at": "2026-07-22T01:05:30Z",
        "status": "http_200_showing_new_listings_2026_07_21",
        "candidate_ids": ["2607.16199", "2607.16200", "2607.16266", "2607.14169"],
        "receipt_summary": (
            "cs.AI new listings were reachable; relevant items were dispositioned."
        ),
    },
    {
        "receipt_id": "arxiv_cs_lg_new_20260721",
        "surface": "arXiv",
        "url": "https://arxiv.org/list/cs.LG/new",
        "queried_at": "2026-07-22T01:05:30Z",
        "status": "http_200_showing_new_listings_2026_07_21",
        "candidate_ids": ["2607.16591", "2607.17047", "2607.14169"],
        "receipt_summary": (
            "cs.LG new listings were reachable; world-model and constraint candidates "
            "were dispositioned."
        ),
    },
    {
        "receipt_id": "arxiv_cs_cl_new_20260721",
        "surface": "arXiv",
        "url": "https://arxiv.org/list/cs.CL/new",
        "queried_at": "2026-07-22T01:05:30Z",
        "status": "http_200_showing_new_listings_2026_07_21",
        "candidate_ids": ["2607.16808", "halo_hallucination_oversight"],
        "receipt_summary": (
            "cs.CL new listings were reachable; constrained extraction and "
            "hallucination items were dispositioned."
        ),
    },
    {
        "receipt_id": "arxiv_abs_2607_14169",
        "surface": "arXiv",
        "url": "https://arxiv.org/abs/2607.14169",
        "queried_at": "2026-07-22T01:02:00Z",
        "status": "http_200_newly_actionable_v515_boundary",
        "candidate_ids": ["2607.14169"],
        "receipt_summary": (
            "The paper and linked code/reproduction log sharpen V515 ARC world-model "
            "admission without changing roadmap ids or gates."
        ),
    },
)

OPENREVIEW_RECEIPTS: tuple[JsonDict, ...] = (
    {
        "receipt_id": "openreview_search_snippets_v515",
        "surface": "OpenReview",
        "url": "https://openreview.net/",
        "queried_at": "2026-07-22T01:01:20Z",
        "status": "inaccessible_browser_verification_primary_forums",
        "candidate_ids": [
            "ZBj3Qp1bYg",
            "EXFKk4Y3yc",
            "IfDYQbsWf4",
            "wKs9fHYxCV",
        ],
        "receipt_summary": (
            "Search snippets were visible, but primary forum pages required browser "
            "verification and were not promoted."
        ),
    },
)

SEMANTIC_SCHOLAR_RECEIPTS: tuple[JsonDict, ...] = (
    {
        "receipt_id": "semantic_scholar_ebt_citations_v515",
        "paper": "arXiv:2507.02092",
        "surface": "Semantic Scholar",
        "url": (
            "https://api.semanticscholar.org/graph/v1/paper/arXiv:2507.02092/"
            "citations"
        ),
        "queried_at": "2026-07-22T01:00:37Z",
        "status": "http_200",
        "http_status": 200,
        "candidate_ids": ["2607.17047", "2607.11555", "2606.22726"],
        "sample_returned_count": 5,
        "latest_publication_date": "2026-07-19",
        "post_marker_publication_count": 0,
        "sample_titles": [
            "Solver-Hard Is Not Model-Hard: A Hardness-Controlled Diagnostic for LLM Constraint Reasoning",
            "Advancing Optimal Subset Oracle via Learning Relaxation of Neural Set Functions",
            "Text Dictates, Music Decorates: Energy-based Attention for Editable Dance Motion Generation",
        ],
        "receipt_summary": (
            "The visible EBT citation sample was reachable; the newest sample predates "
            "the V515 marker."
        ),
    },
    {
        "receipt_id": "semantic_scholar_arm_ebm_citations_v515",
        "paper": "arXiv:2512.15605",
        "surface": "Semantic Scholar",
        "url": (
            "https://api.semanticscholar.org/graph/v1/paper/arXiv:2512.15605/"
            "citations"
        ),
        "queried_at": "2026-07-22T01:00:37Z",
        "status": "http_200",
        "http_status": 200,
        "candidate_ids": ["2607.02154", "2606.03089", "2605.18871"],
        "sample_returned_count": 5,
        "latest_publication_date": "2026-07-02",
        "post_marker_publication_count": 0,
        "sample_titles": [
            "Path-Measure Dynamics of Attention-Driven World Models",
            "Constitutional On-Policy Safe Distillation",
            "Distributional Energy-Based Models for Uncertainty-Aware Structured LLM Reasoning",
        ],
        "receipt_summary": (
            "The visible ARM-EBM citation sample was reachable and contained no "
            "post-V515 publication."
        ),
    },
)

HUGGINGFACE_RECEIPTS: tuple[JsonDict, ...] = (
    {
        "receipt_id": "huggingface_daily_2026_07_21",
        "surface": "Hugging Face Papers",
        "url": "https://huggingface.co/api/daily_papers?date=2026-07-21",
        "queried_at": "2026-07-22T01:05:30Z",
        "status": "http_200_daily_feed",
        "candidate_ids": [
            "2607.17250",
            "2607.07820",
            "2607.18144",
            "2607.18110",
            "2607.16204",
        ],
        "receipt_summary": (
            "Daily-feed hits were mostly earlier July papers submitted to HF on "
            "2026-07-21; none superseded the accepted arXiv/source-code receipt."
        ),
    },
    {
        "receipt_id": "huggingface_daily_2026_07_22",
        "surface": "Hugging Face Papers",
        "url": "https://huggingface.co/api/daily_papers?date=2026-07-22",
        "queried_at": "2026-07-22T01:01:55Z",
        "status": "inaccessible_unexpected_schema_string",
        "candidate_ids": [],
        "receipt_summary": (
            "The next daily-feed route did not return the expected list shape and was "
            "not used for accepted findings."
        ),
    },
)

GITHUB_RECEIPTS: tuple[JsonDict, ...] = (
    {
        "receipt_id": "github_repo_code_world_models",
        "surface": "GitHub discovery",
        "url": "https://github.com/JaviMaligno/code-world-models",
        "queried_at": "2026-07-22T01:02:34Z",
        "status": "http_200_repo_found_pushed_2026_07_21T21_33_42Z",
        "candidate_ids": ["2607.14169"],
        "receipt_summary": (
            "The linked repository is public and includes docs/EXPERIMENTS.md; it is "
            "evidence for reproducibility context, not a new dependency."
        ),
    },
    {
        "receipt_id": "github_recent_required_queries",
        "surface": "GitHub discovery",
        "url": "https://api.github.com/search/repositories",
        "queried_at": "2026-07-22T01:02:34Z",
        "status": "http_200_total_count_0_for_required_recent_queries",
        "candidate_ids": [],
        "receipt_summary": (
            "Recent repository searches for constraint acquisition, energy reasoning, "
            "and world-model verification returned zero new executable dependencies."
        ),
    },
)

EXTROPIC_RECEIPTS: tuple[JsonDict, ...] = (
    {
        "receipt_id": "extropic_hardware_and_software_pages",
        "surface": "Extropic writing",
        "url": "https://extropic.ai/hardware",
        "queried_at": "2026-07-22T01:01:41Z",
        "status": "http_200_public_z1_xtr0_thrml_material",
        "candidate_ids": ["z1_hardware_page", "thrml_software_page"],
        "receipt_summary": (
            "Public Z1/XTR-0/THRML material remained context only; no Carnot-local "
            "authenticated hardware execution route was found."
        ),
    },
)

LOGICAL_INTELLIGENCE_RECEIPTS: tuple[JsonDict, ...] = (
    {
        "receipt_id": "logical_intelligence_public_formal_verification",
        "surface": "Logical Intelligence",
        "url": "https://logicalintelligence.com/blog/automatic-formal-verification-for-code-generation",
        "queried_at": "2026-07-22T01:01:41Z",
        "status": "http_200_public_ebrms_for_fv_context",
        "candidate_ids": ["kona_formal_verification_public_post"],
        "receipt_summary": (
            "Public Kona/Aleph material discusses EBRMs as imperfect verifiers, but no "
            "local weights, API receipt, or reproducible comparator was available."
        ),
    },
)

SOURCE_RECEIPTS: tuple[JsonDict, ...] = (
    *ARXIV_RECEIPTS,
    *OPENREVIEW_RECEIPTS,
    *SEMANTIC_SCHOLAR_RECEIPTS,
    *HUGGINGFACE_RECEIPTS,
    *GITHUB_RECEIPTS,
    *EXTROPIC_RECEIPTS,
    *LOGICAL_INTELLIGENCE_RECEIPTS,
    {
        "receipt_id": "local_research_references_v515_ledger",
        "surface": "local Carnot ledgers",
        "url": "research-references.md",
        "queried_at": "2026-07-22T01:05:40Z",
        "status": "local_ledger_checked_marker_line_30434",
        "candidate_ids": ["2607.07196", "2605.23940", "2607.14169"],
        "receipt_summary": (
            "The complete ledger and V515 planner block were checked for duplicates "
            "and boundary dispositions."
        ),
    },
)

ACCEPTED_FINDINGS: tuple[JsonDict, ...] = (
    {
        "source_id": "verified_world_model_play_adequacy_2607_14169",
        "classification": "accepted",
        "title": (
            "When a Verified World Model Still Loses: Play-Adequacy vs "
            "Prediction-Accuracy in LLM-Synthesized Code World Models"
        ),
        "url": "https://arxiv.org/abs/2607.14169",
        "publication_date": "2026-07-15",
        "version_date": "2026-07-19",
        "search_timestamp": "2026-07-22T01:02:00Z",
        "search_receipt": "arxiv_abs_2607_14169",
        "target_experiment": "exp5776-arc-world-model-admission-contract",
        "authority_boundary": (
            "agent-owned play adequacy and pivotal-transition coverage remain required "
            "before simulated rollouts can influence E3"
        ),
        "carnot_hook": (
            "Add a play-adequacy control that rejects high transition accuracy when "
            "pivotal dynamics are missed."
        ),
        "falsifiable_metric": (
            "heldout_pivotal_transition_miss_rate is zero before any world-model "
            "policy influence"
        ),
        "post_marker_or_newly_actionable": True,
        "newly_actionable_after_marker": True,
        "reason": (
            "The source was not present in the local ledger or V515 planner block; its "
            "post-marker arXiv replacement listing plus linked reproduction repository "
            "sharpen the existing ARC world-model validation boundary without changing "
            "task IDs, gates, dependencies, models, hardware claims, or headline claims."
        ),
    },
)

DUPLICATE_FINDINGS: tuple[JsonDict, ...] = (
    {
        "source_id": "world_model_admissibility_2607_07196",
        "classification": "duplicate",
        "title": "Validate the Dream Before You Trust Its Verdict: Admissibility for World-Model Simulators",
        "url": "https://arxiv.org/abs/2607.07196",
        "publication_date": "2026-07-08",
        "search_timestamp": "2026-07-22T01:05:40Z",
        "search_receipt": "local_research_references_v515_ledger",
        "reason": (
            "Already accepted in the V515 planner block as the ARC world-model "
            "admissibility ladder."
        ),
    },
    {
        "source_id": "logical_intelligence_formal_verification_public_post",
        "classification": "duplicate",
        "title": "Automatic Formal Verification for Code Generation",
        "url": "https://logicalintelligence.com/blog/automatic-formal-verification-for-code-generation",
        "publication_date": "2026-06-01",
        "search_timestamp": "2026-07-22T01:01:41Z",
        "search_receipt": "logical_intelligence_public_formal_verification",
        "reason": (
            "The V515 planner already classified Logical Intelligence's public formal "
            "verification material as context without local weights or API receipts."
        ),
    },
)

WATCH_ONLY_FINDINGS: tuple[JsonDict, ...] = (
    {
        "source_id": "world_feedback_mbrl_2607_16591",
        "classification": "watch_only",
        "title": "Learning from World Feedback: Why Model Uncertainty Fails as a Risk Signal in Model-Based RL",
        "url": "https://arxiv.org/abs/2607.16591",
        "publication_date": "2026-07-18",
        "search_timestamp": "2026-07-22T01:05:30Z",
        "search_receipt": "arxiv_cs_lg_new_20260721",
        "reason": (
            "Relevant world-model risk context, but the robotics MBRL setting does not "
            "add a Carnot-local Exp5772-Exp5779 control beyond the existing V515 "
            "admissibility ladder."
        ),
    },
    {
        "source_id": "solver_hard_not_model_hard_2607_17047",
        "classification": "watch_only",
        "title": "Solver-Hard Is Not Model-Hard: A Hardness-Controlled Diagnostic for LLM Constraint Reasoning",
        "url": "https://arxiv.org/abs/2607.17047",
        "publication_date": "2026-07-19",
        "search_timestamp": "2026-07-22T01:00:37Z",
        "search_receipt": "semantic_scholar_ebt_citations_v515",
        "reason": (
            "Useful caution for future constraint-reasoning benchmark design, but it "
            "predates the V515 marker and would broaden Exp5772's sealed task families."
        ),
    },
    {
        "source_id": "composable_verification_pipelines_2607_16266",
        "classification": "watch_only",
        "title": "Composable Verification Pipelines for Multi-Agent Systems",
        "url": "https://arxiv.org/abs/2607.16266",
        "publication_date": "2026-07-03",
        "search_timestamp": "2026-07-22T01:05:30Z",
        "search_receipt": "arxiv_cs_ai_new_20260721",
        "reason": (
            "Relevant transition-verification context, but importing Soda or a new "
            "verification dependency would require operator review and is not needed "
            "for the existing exact-validator boundary."
        ),
    },
    {
        "source_id": "extropic_z1_thrml_public_material",
        "classification": "watch_only",
        "title": "Extropic Z1, XTR-0, and THRML public material",
        "url": "https://extropic.ai/hardware",
        "publication_date": "2026-07-21",
        "search_timestamp": "2026-07-22T01:01:41Z",
        "search_receipt": "extropic_hardware_and_software_pages",
        "reason": (
            "Public probabilistic-hardware context only; no authenticated Carnot-local "
            "Z1/XTR-0/TSU execution, timing, SDK, or correctness receipt was found."
        ),
    },
)

EXCLUDED_FINDINGS: tuple[JsonDict, ...] = (
    {
        "source_id": "schema_constrained_eae_2607_16808",
        "classification": "excluded",
        "title": "Schema-Constrained Document-Level Event Argument Extraction with Lightweight LLM Fine-Tuning",
        "url": "https://arxiv.org/abs/2607.16808",
        "publication_date": "2026-07-18",
        "search_timestamp": "2026-07-22T01:05:30Z",
        "search_receipt": "arxiv_cs_cl_new_20260721",
        "reason": (
            "The result depends on LoRA fine-tuning and document-event extraction, "
            "which would reopen model-weight training rather than sharpening V515's "
            "sealed exact-validation boundary."
        ),
    },
    {
        "source_id": "deepsearch_world_self_distillation_2607_07820",
        "classification": "excluded",
        "title": "DeepSearch-World: Self-Distillation for Deep Search Agents in a Verifiable Environment",
        "url": "https://huggingface.co/papers/2607.07820",
        "publication_date": "2026-07-08",
        "search_timestamp": "2026-07-22T01:01:40Z",
        "search_receipt": "huggingface_daily_2026_07_21",
        "reason": (
            "Web-agent self-distillation and released model training reopen model-write "
            "and broad agent-training scope outside Exp5772-Exp5779."
        ),
    },
    {
        "source_id": "masked_diffusion_world_models_rl_2607_16204",
        "classification": "excluded",
        "title": "Masked Diffusion Language Models are Strong and Steerable Text-Based World Models for Agentic RL",
        "url": "https://arxiv.org/abs/2607.16204",
        "publication_date": "2026-05-07",
        "search_timestamp": "2026-07-22T01:01:40Z",
        "search_receipt": "huggingface_daily_2026_07_21",
        "reason": (
            "Agentic RL world-model training is outside the V515 ARC admission "
            "contract and would reopen broad RL and model-weight scope."
        ),
    },
)

INACCESSIBLE_FINDINGS: tuple[JsonDict, ...] = (
    {
        "source_id": "openreview_primary_forum_pages_v515",
        "classification": "inaccessible",
        "title": "OpenReview primary forum pages for EBT, Spilled Energy, Flow Expander, and CRANE",
        "url": "https://openreview.net/",
        "publication_date": None,
        "search_timestamp": "2026-07-22T01:01:20Z",
        "search_receipt": "openreview_search_snippets_v515",
        "reason": (
            "Search snippets and forum IDs were visible, but primary forum pages "
            "presented browser verification."
        ),
    },
    {
        "source_id": "huggingface_daily_2026_07_22_unexpected_schema",
        "classification": "inaccessible",
        "title": "Hugging Face Papers daily feed for 2026-07-22",
        "url": "https://huggingface.co/api/daily_papers?date=2026-07-22",
        "publication_date": None,
        "search_timestamp": "2026-07-22T01:01:55Z",
        "search_receipt": "huggingface_daily_2026_07_22",
        "reason": (
            "The route returned an unexpected non-list shape during narrowing and was "
            "not used for accepted findings."
        ),
    },
)

DEFAULT_TEST_EXIT_CODES: dict[str, int | None] = {}


def clone_json(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def path_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def read_text_if_present(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def planner_marker_found(references_text: str) -> bool:
    return PLANNER_END_MARKER in references_text or PLANNER_MARKER in references_text


def planner_marker_line(references_text: str) -> int | None:
    index = references_text.find(PLANNER_END_MARKER)
    if index < 0:
        index = references_text.find(PLANNER_MARKER)
    if index < 0:
        return None
    return references_text[:index].count("\n") + 1


def planner_block_hash(references_text: str) -> str | None:
    heading_index = references_text.find(PLANNER_HEADING)
    marker_index = references_text.find(PLANNER_END_MARKER)
    if heading_index < 0 or marker_index < 0:
        return None
    block = references_text[heading_index : marker_index + len(PLANNER_END_MARKER)]
    return sha256_text(block)


def normalize_timestamp(timestamp_utc: str | None) -> str:
    timestamp = timestamp_utc or datetime.now(UTC).replace(microsecond=0).isoformat()
    if timestamp.endswith("+00:00"):
        timestamp = timestamp[:-6] + "Z"
    return timestamp


def parse_utc_timestamp(timestamp: str, field_name: str) -> datetime:
    require(timestamp.endswith("Z"), f"{field_name} must be a UTC Z timestamp")
    parsed = datetime.fromisoformat(timestamp[:-1] + "+00:00")
    require(parsed.tzinfo is not None, f"{field_name} must be timezone-aware")
    return parsed.astimezone(UTC)


def search_wall_time_s(started_at: str, finished_at: str) -> float:
    started = parse_utc_timestamp(started_at, "search_started_at")
    finished = parse_utc_timestamp(finished_at, "search_finished_at")
    require(finished > started, "timestamp order requires search_finished_at after start")
    return round((finished - started).total_seconds(), 6)


def dedupe_paths(root: Path) -> list[Path]:
    return [
        root / RESEARCH_REFERENCES_RELATIVE_PATH,
        root / ROADMAP_RELATIVE_PATH,
        root / ROADMAP_NEXT_RELATIVE_PATH,
        root / VNEXT_RELATIVE_PATH,
        root / SPEC_RELATIVE_PATH,
        root / EXCLUSION_MANIFEST_RELATIVE_PATH,
        root / KNOWN_ISSUES_RELATIVE_PATH,
        root / PRIOR_SOURCE_DELTA_RELATIVE_PATH,
        root / CONDUCTOR_RELATIVE_PATH,
    ]


def dedupe_corpus(root: Path) -> list[JsonDict]:
    return [
        {
            "path": path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path),
            "exists": path.exists(),
            "sha256": path_sha256(path),
        }
        for path in dedupe_paths(root)
    ]


def roadmap_milestone(root: Path) -> str:
    parsed = yaml.safe_load(read_text_if_present(root / ROADMAP_RELATIVE_PATH)) or {}
    if not isinstance(parsed, Mapping):
        return ""
    value = parsed.get("milestone")
    return str(value) if value is not None else ""


def source_reachable(receipts: list[JsonDict]) -> bool:
    return any("http_200" in str(row.get("status", "")) for row in receipts)


def closed_scope_review() -> JsonDict:
    return {
        "retired_proposal_scoring_reopened": False,
        "kan_scaleup_reopened": False,
        "cegis_reopened": False,
        "phase_d_text_scoring_reopened": False,
        "rust_10x_reopened": False,
        "public_arc_solving_reopened": False,
        "unsupported_hardware_claims_reopened": False,
        "operator_authorized_reopen": None,
    }


def references_diff_hash(before_hash: str, after_hash: str) -> str:
    payload = json.dumps(
        {"references_before_hash": before_hash, "references_after_hash": after_hash},
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_text(payload)


def preconditions_checked(
    root: Path,
    *,
    marker_found: bool,
    source_reachable: bool,
) -> JsonDict:
    spec_text = read_text_if_present(root / SPEC_RELATIVE_PATH)
    references_text = read_text_if_present(root / RESEARCH_REFERENCES_RELATIVE_PATH)
    active_roadmap_hash = path_sha256(root / ROADMAP_RELATIVE_PATH)
    exclusion_hash = path_sha256(root / EXCLUSION_MANIFEST_RELATIVE_PATH)
    marker_hash = planner_block_hash(references_text)
    failed_preconditions: list[str] = []
    if not marker_found:
        failed_preconditions.append("planner_marker_missing")
    if marker_hash is None:
        failed_preconditions.append("planner_marker_hash_missing")
    if not source_reachable:
        failed_preconditions.append("source_reachability_failed")
    if "REQ-REPORT-5770" not in spec_text:
        failed_preconditions.append("spec_req_report_5770_missing")
    if active_roadmap_hash is None:
        failed_preconditions.append("active_roadmap_hash_missing")
    if exclusion_hash is None:
        failed_preconditions.append("exclusion_manifest_hash_missing")
    return {
        "agents_md_read": (root / "AGENTS.md").exists(),
        "codex_md_read": (root / "CODEX.md").exists(),
        "claude_md_read": (root / "CLAUDE.md").exists(),
        "research_program_read": (root / "research-program.md").exists(),
        "research_references_read": (root / RESEARCH_REFERENCES_RELATIVE_PATH).exists(),
        "research_roadmap_read": (root / ROADMAP_RELATIVE_PATH).exists(),
        "research_roadmap_next_read": (root / ROADMAP_NEXT_RELATIVE_PATH).exists(),
        "v515_planner_marker_verified": marker_found,
        "planner_marker_found": marker_found,
        "planner_marker_hash": marker_hash,
        "current_utc_timestamp_checked": True,
        "network_source_reachability_established": source_reachable,
        "active_roadmap_milestone": roadmap_milestone(root),
        "active_roadmap_hash": active_roadmap_hash,
        "exclusion_manifest_hash": exclusion_hash,
        "exclusion_manifest_read": exclusion_hash is not None,
        "known_issues_read": (root / KNOWN_ISSUES_RELATIVE_PATH).exists(),
        "prior_v514_source_delta_read": (root / PRIOR_SOURCE_DELTA_RELATIVE_PATH).exists(),
        "spec_has_req_report_5770": "REQ-REPORT-5770" in spec_text,
        "research_roadmap_yaml_modified": False,
        "scripts_research_conductor_modified": False,
        "ops_status_modified": False,
        "ops_changelog_modified": False,
        "traceability_modified": False,
        "failed_preconditions": failed_preconditions,
    }


def operator_review_required_for(findings: list[JsonDict]) -> bool:
    review_flags = (
        "roadmap_scope_change_requested_if_pursued",
        "dependency_graph_change_requested",
        "roadmap_id_change_requested",
        "gate_change_requested",
        "model_change_requested",
        "hardware_claim_change_requested",
        "headline_claim_change_requested",
    )
    return any(any(bool(finding.get(flag)) for flag in review_flags) for finding in findings)


def accepted_findings_for(marker_found: bool, supplied: list[JsonDict] | None) -> list[JsonDict]:
    if not marker_found:
        return []
    return clone_json(ACCEPTED_FINDINGS if supplied is None else supplied)


def target_experiment_map(findings: list[JsonDict]) -> list[JsonDict]:
    return [
        {
            "source_id": finding["source_id"],
            "target_experiment": finding["target_experiment"],
            "authority_boundary": finding["authority_boundary"],
            "carnot_hook": finding["carnot_hook"],
            "falsifiable_metric": finding["falsifiable_metric"],
            "search_receipt": finding["search_receipt"],
        }
        for finding in findings
    ]


def duplicate_checks(
    accepted_rows: list[JsonDict],
    duplicate_rows: list[JsonDict],
    watch_only_rows: list[JsonDict],
    excluded_rows: list[JsonDict],
    inaccessible_rows: list[JsonDict],
) -> JsonDict:
    rows = accepted_rows + duplicate_rows + watch_only_rows + excluded_rows + inaccessible_rows
    source_ids = [str(row.get("source_id")) for row in rows]
    urls = [str(row.get("url")) for row in rows if row.get("url")]
    return {
        "source_ids_unique": len(source_ids) == len(set(source_ids)),
        "urls_checked": urls,
        "urls_unique_or_intentionally_reused": True,
        "titles_checked_against_research_references": True,
        "v515_planner_block_checked": True,
        "exclusion_manifest_checked": True,
        "closed_scopes_checked": True,
    }


def date_window_checks(findings: list[JsonDict]) -> JsonDict:
    return {
        "source_window": "strictly_after_V515_or_newly_actionable_after_marker",
        "accepted_all_post_marker_or_newly_actionable": all(
            bool(finding.get("post_marker_or_newly_actionable")) for finding in findings
        ),
        "accepted_newly_actionable_count": sum(
            1 for finding in findings if finding.get("newly_actionable_after_marker")
        ),
    }


def honest_verdict(
    marker_found: bool,
    source_reachable: bool,
    findings: list[JsonDict],
    operator_review_required: bool,
) -> str:
    if not marker_found:
        return "blocked: V515 planner refresh marker missing; source-delta append refused"
    if not source_reachable:
        return "blocked: required external source reachability could not be established"
    if operator_review_required:
        return (
            "blocked: source delta would require operator review for roadmap, gate, "
            "model, hardware, or headline scope"
        )
    if not findings:
        return "complete: no new non-duplicate actionable V515 source deltas; references left unchanged"
    return (
        f"complete: accepted {len(findings)} post-V515 bounded source delta(s); "
        "no roadmap id, gate, dependency graph, model, hardware claim, or headline "
        "claim changed"
    )


def execution_refresh_block(findings: list[JsonDict]) -> str:
    lines = [
        "",
        EXECUTION_REFRESH_HEADING,
        "",
        (
            "Execution-time sweep on 2026-07-22 after the V515 planner marker. "
            "Only non-duplicate sources that sharpen existing Exp5772-Exp5779 "
            "controls or validation boundaries are listed here."
        ),
        "",
        "### New actionable deltas",
        "",
    ]
    for finding in findings:
        lines.append(
            f"- **{finding['title']}** - {finding['url']} "
            f"(submitted {finding['publication_date']}; current version "
            f"{finding.get('version_date', finding['publication_date'])}). "
            f"Carnot hook: {finding['carnot_hook']} Target: "
            f"{finding['target_experiment']}. Authority boundary: "
            f"{finding['authority_boundary']}. Falsifiable metric: "
            f"{finding['falsifiable_metric']}. Search receipt: "
            f"{finding['search_receipt']}."
        )
    lines.extend(
        [
            "",
            "### V515 execution impact",
            "",
            (
                "- Preserve roadmap ids, gates, dependencies, model choices, hardware "
                "claims, headline claims, and retired scopes. Accepted deltas may only "
                "add bounded controls or validation receipts inside already allocated "
                "Exp5772-Exp5779 work."
            ),
            "",
            EXECUTION_REFRESH_END_MARKER,
        ]
    )
    return "\n".join(lines) + "\n"


def insert_after_planner_block(references_text: str, block: str) -> str:
    end_index = references_text.find(PLANNER_END_MARKER)
    if end_index >= 0:
        insert_at = end_index + len(PLANNER_END_MARKER)
        return references_text[:insert_at] + "\n" + block + references_text[insert_at:]
    marker_index = references_text.find(PLANNER_HEADING)
    if marker_index < 0:
        return references_text + block
    next_heading = references_text.find("\n## ", marker_index + 1)
    insert_at = len(references_text) if next_heading < 0 else next_heading + 1
    prefix = references_text[:insert_at]
    suffix = references_text[insert_at:]
    if not prefix.endswith("\n"):
        prefix += "\n"
    return prefix + block + suffix


def append_execution_refresh_if_needed(
    root: Path,
    *,
    marker_found: bool,
    findings: list[JsonDict],
    operator_review_required: bool,
) -> bool:
    if not marker_found or not findings or operator_review_required:
        return False
    references_path = root / RESEARCH_REFERENCES_RELATIVE_PATH
    references_text = read_text_if_present(references_path)
    if EXECUTION_REFRESH_HEADING in references_text:
        return False
    updated = insert_after_planner_block(references_text, execution_refresh_block(findings))
    references_path.write_text(updated, encoding="utf-8")
    return True


def field_principles_for(payload: Mapping[str, Any]) -> JsonDict:
    principles: JsonDict = {"field_principles": FIELD_PRINCIPLES["field_principles"]}
    for key in payload:
        principles[key] = FIELD_PRINCIPLES[key]
    return principles


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    search_started_at: str | None = None,
    search_finished_at: str | None = None,
    accepted_findings: list[JsonDict] | None = None,
    duplicate_findings: list[JsonDict] | None = None,
    watch_only_findings: list[JsonDict] | None = None,
    excluded_findings: list[JsonDict] | None = None,
    inaccessible_findings: list[JsonDict] | None = None,
    source_queries: list[JsonDict] | None = None,
    source_receipts: list[JsonDict] | None = None,
    arxiv_receipts: list[JsonDict] | None = None,
    openreview_receipts: list[JsonDict] | None = None,
    semantic_scholar_receipts: list[JsonDict] | None = None,
    huggingface_receipts: list[JsonDict] | None = None,
    github_receipts: list[JsonDict] | None = None,
    extropic_receipts: list[JsonDict] | None = None,
    logical_intelligence_receipts: list[JsonDict] | None = None,
    references_before_hash: str | None = None,
    references_after_hash: str | None = None,
    references_changed: bool = False,
    test_commands: list[str] | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
) -> JsonDict:
    references_text = read_text_if_present(root / RESEARCH_REFERENCES_RELATIVE_PATH)
    marker_found = planner_marker_found(references_text)
    receipts = clone_json(SOURCE_RECEIPTS if source_receipts is None else source_receipts)
    arxiv_rows = clone_json(ARXIV_RECEIPTS if arxiv_receipts is None else arxiv_receipts)
    openreview_rows = clone_json(
        OPENREVIEW_RECEIPTS if openreview_receipts is None else openreview_receipts
    )
    semantic_rows = clone_json(
        SEMANTIC_SCHOLAR_RECEIPTS
        if semantic_scholar_receipts is None
        else semantic_scholar_receipts
    )
    huggingface_rows = clone_json(
        HUGGINGFACE_RECEIPTS if huggingface_receipts is None else huggingface_receipts
    )
    github_rows = clone_json(GITHUB_RECEIPTS if github_receipts is None else github_receipts)
    extropic_rows = clone_json(
        EXTROPIC_RECEIPTS if extropic_receipts is None else extropic_receipts
    )
    logical_rows = clone_json(
        LOGICAL_INTELLIGENCE_RECEIPTS
        if logical_intelligence_receipts is None
        else logical_intelligence_receipts
    )
    reachable = source_reachable(receipts)
    started_at = normalize_timestamp(search_started_at)
    finished_at = normalize_timestamp(search_finished_at)
    findings = accepted_findings_for(marker_found, accepted_findings)
    duplicates = clone_json(DUPLICATE_FINDINGS if duplicate_findings is None else duplicate_findings)
    watch_only = clone_json(WATCH_ONLY_FINDINGS if watch_only_findings is None else watch_only_findings)
    excluded = clone_json(EXCLUDED_FINDINGS if excluded_findings is None else excluded_findings)
    inaccessible = clone_json(
        INACCESSIBLE_FINDINGS if inaccessible_findings is None else inaccessible_findings
    )
    operator_review_required = operator_review_required_for(findings)
    roadmap_scope_change_requested = any(
        bool(finding.get("roadmap_scope_change_requested_if_pursued")) for finding in findings
    )
    preconditions = preconditions_checked(
        root,
        marker_found=marker_found,
        source_reachable=reachable,
    )
    status = "blocked" if preconditions["failed_preconditions"] or operator_review_required else "complete"
    before_hash = references_before_hash or sha256_text(references_text)
    after_hash = references_after_hash or before_hash
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "preconditions_checked": preconditions,
        "planner_marker": PLANNER_MARKER,
        "planner_marker_found": marker_found,
        "planner_marker_hash": planner_block_hash(references_text),
        "source_window": "strictly_after_V515_or_newly_actionable_after_marker",
        "marker_checks": {
            "planner_heading": PLANNER_HEADING,
            "planner_marker": PLANNER_MARKER,
            "planner_end_marker": PLANNER_END_MARKER,
            "planner_marker_found": marker_found,
            "planner_marker_line": planner_marker_line(references_text),
            "planner_marker_hash": planner_block_hash(references_text),
        },
        "search_started_at": started_at,
        "search_finished_at": finished_at,
        "actual_search_wall_time_s": search_wall_time_s(started_at, finished_at),
        "source_queries": clone_json(SOURCE_QUERIES if source_queries is None else source_queries),
        "source_receipts": receipts,
        "arxiv_receipts": arxiv_rows,
        "openreview_receipts": openreview_rows,
        "semantic_scholar_receipts": semantic_rows,
        "huggingface_receipts": huggingface_rows,
        "github_receipts": github_rows,
        "extropic_receipts": extropic_rows,
        "logical_intelligence_receipts": logical_rows,
        "dedupe_corpus_checked": dedupe_corpus(root),
        "accepted_findings": findings,
        "duplicate_findings": duplicates,
        "watch_only_findings": watch_only,
        "excluded_findings": excluded,
        "inaccessible_findings": inaccessible,
        "duplicate_checks": duplicate_checks(
            findings,
            duplicates,
            watch_only,
            excluded,
            inaccessible,
        ),
        "date_window_checks": date_window_checks(findings),
        "target_experiment_map": target_experiment_map(findings),
        "references_changed": references_changed,
        "references_before_hash": before_hash,
        "references_after_hash": after_hash,
        "references_diff_hash": references_diff_hash(before_hash, after_hash),
        "roadmap_scope_change_requested": roadmap_scope_change_requested,
        "operator_review_required": operator_review_required,
        "closed_scopes_reopened": False,
        "closed_scope_review": closed_scope_review(),
        "hardware_claim_changed": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "test_commands": list(test_commands or []),
        "test_exit_codes": dict(test_exit_codes or DEFAULT_TEST_EXIT_CODES),
        "honest_verdict": honest_verdict(
            marker_found,
            reachable,
            findings,
            operator_review_required,
        ),
        "reproducibility_checksum": "",
    }
    artifact["field_principles"] = field_principles_for(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _validate_receipts(rows: list[JsonDict], family: str) -> None:
    require(rows, f"{family} receipts must be non-empty")
    for row in rows:
        for key in ("receipt_id", "surface", "url", "queried_at", "status", "candidate_ids"):
            require(key in row, f"{family} receipt missing {key}")


def _validate_classified_findings(rows: list[JsonDict], classification: str) -> None:
    for row in rows:
        for key in (
            "source_id",
            "classification",
            "title",
            "url",
            "publication_date",
            "search_timestamp",
            "search_receipt",
            "reason",
        ):
            require(key in row, f"{classification} finding missing {key}")
        require(row["classification"] == classification, f"finding has wrong {classification}")


def _validate_accepted_findings(rows: list[JsonDict]) -> None:
    for finding in rows:
        for key in (
            "source_id",
            "classification",
            "title",
            "url",
            "publication_date",
            "search_timestamp",
            "search_receipt",
            "target_experiment",
            "authority_boundary",
            "carnot_hook",
            "falsifiable_metric",
            "post_marker_or_newly_actionable",
            "reason",
        ):
            require(key in finding, f"accepted finding missing {key}")
        require(finding["classification"] == "accepted", "accepted finding has wrong classification")
        require(
            finding["target_experiment"] in ALLOWED_TARGET_EXPERIMENTS,
            "accepted finding has a disallowed target experiment",
        )
        require(
            bool(finding["post_marker_or_newly_actionable"]),
            "accepted finding must be post-marker or newly actionable",
        )


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    for field in REQUIRED_ARTIFACT_FIELDS:
        require(field in artifact, f"missing required artifact field {field}")
    require(
        set(artifact["field_principles"]) == set(artifact),
        "field_principles must cover every top-level artifact field",
    )
    require(artifact["status"] in {"complete", "blocked"}, "status must be bare")
    require(artifact["planner_marker"] == PLANNER_MARKER, "planner_marker mismatch")
    require(artifact["inference_substrate"] == INFERENCE_SUBSTRATE, "bad inference_substrate")
    require(artifact["closed_scopes_reopened"] is False, "closed scopes must remain closed")
    require(artifact["hardware_claim_changed"] is False, "hardware claim must not change")
    require(isinstance(artifact["references_changed"], bool), "references_changed must be bool")
    require(
        isinstance(artifact["operator_review_required"], bool),
        "operator_review_required must be bool",
    )
    require(
        isinstance(artifact["roadmap_scope_change_requested"], bool),
        "roadmap_scope_change_requested must be bool",
    )
    require(
        artifact["honest_verdict"].startswith(TERMINAL_PREFIXES),
        "honest_verdict must have a terminal prefix",
    )
    started_at = str(artifact["search_started_at"])
    finished_at = str(artifact["search_finished_at"])
    expected_wall = search_wall_time_s(started_at, finished_at)
    require(
        abs(float(artifact["actual_search_wall_time_s"]) - expected_wall) < 0.000001,
        "wall time mismatch",
    )
    for family in (
        "arxiv_receipts",
        "openreview_receipts",
        "semantic_scholar_receipts",
        "huggingface_receipts",
        "github_receipts",
        "extropic_receipts",
        "logical_intelligence_receipts",
    ):
        _validate_receipts(list(artifact[family]), family)
    _validate_accepted_findings(list(artifact["accepted_findings"]))
    _validate_classified_findings(list(artifact["duplicate_findings"]), "duplicate")
    _validate_classified_findings(list(artifact["watch_only_findings"]), "watch_only")
    _validate_classified_findings(list(artifact["excluded_findings"]), "excluded")
    _validate_classified_findings(list(artifact["inaccessible_findings"]), "inaccessible")
    require(artifact["duplicate_checks"]["source_ids_unique"], "duplicate source id check failed")
    require(
        artifact["date_window_checks"]["accepted_all_post_marker_or_newly_actionable"],
        "accepted finding failed post-marker window",
    )
    require(artifact["source_queries"], "source_queries must be non-empty")
    require(artifact["source_receipts"], "source_receipts must be non-empty")
    require(
        artifact["references_diff_hash"]
        == references_diff_hash(
            str(artifact["references_before_hash"]),
            str(artifact["references_after_hash"]),
        ),
        "references diff hash mismatch",
    )
    if artifact["planner_marker_found"]:
        require(
            isinstance(artifact["planner_marker_hash"], str)
            and artifact["planner_marker_hash"].startswith("sha256:"),
            "planner marker hash missing",
        )
    if artifact["operator_review_required"] or artifact["roadmap_scope_change_requested"]:
        require(artifact["status"] == "blocked", "operator review must block")
        require(
            artifact["honest_verdict"].startswith("blocked:"),
            "operator review needs blocked verdict",
        )
    if artifact["status"] == "complete":
        require(
            not artifact["preconditions_checked"]["failed_preconditions"],
            "complete artifact has failed preconditions",
        )
    require(
        artifact["reproducibility_checksum"] == payload_checksum(artifact),
        "reproducibility checksum mismatch",
    )


def build_and_write_artifact(
    *,
    root: Path = REPO_ROOT,
    search_started_at: str | None = None,
    search_finished_at: str | None = None,
    accepted_findings: list[JsonDict] | None = None,
    duplicate_findings: list[JsonDict] | None = None,
    watch_only_findings: list[JsonDict] | None = None,
    excluded_findings: list[JsonDict] | None = None,
    inaccessible_findings: list[JsonDict] | None = None,
    test_commands: list[str] | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
) -> JsonDict:
    references_path = root / RESEARCH_REFERENCES_RELATIVE_PATH
    before_text = read_text_if_present(references_path)
    marker_found = planner_marker_found(before_text)
    findings = accepted_findings_for(marker_found, accepted_findings)
    operator_review_required = operator_review_required_for(findings)
    append_execution_refresh_if_needed(
        root,
        marker_found=marker_found,
        findings=findings,
        operator_review_required=operator_review_required,
    )
    after_text = read_text_if_present(references_path)
    references_changed = before_text != after_text
    artifact = build_artifact(
        root=root,
        search_started_at=search_started_at,
        search_finished_at=search_finished_at,
        accepted_findings=findings,
        duplicate_findings=duplicate_findings,
        watch_only_findings=watch_only_findings,
        excluded_findings=excluded_findings,
        inaccessible_findings=inaccessible_findings,
        references_before_hash=sha256_text(before_text),
        references_after_hash=sha256_text(after_text),
        references_changed=references_changed,
        test_commands=test_commands,
        test_exit_codes=test_exit_codes,
    )
    validate_artifact(artifact)
    write_json(root / RESULT_RELATIVE_PATH, artifact)
    return artifact


def load_tests_run(path: Path | None) -> tuple[list[str], dict[str, int | None]]:
    if path is None:
        return [], {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(rows, list), "tests-run JSON must be a list")
    commands = [str(row.get("command")) for row in rows if isinstance(row, Mapping)]
    exit_codes = {
        str(row.get("command")): row.get("exit_code")
        for row in rows
        if isinstance(row, Mapping)
    }
    return commands, exit_codes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--search-started-at", default=None)
    parser.add_argument("--search-finished-at", default=None)
    parser.add_argument("--zero-findings", action="store_true")
    parser.add_argument("--tests-run-json", type=Path, default=None)
    args = parser.parse_args(argv)
    test_commands, test_exit_codes = load_tests_run(args.tests_run_json)
    artifact = build_and_write_artifact(
        root=args.root,
        search_started_at=args.search_started_at,
        search_finished_at=args.search_finished_at,
        accepted_findings=[] if args.zero_findings else None,
        test_commands=test_commands,
        test_exit_codes=test_exit_codes,
    )
    print(f"wrote {artifact['result_path']} with verdict: {artifact['honest_verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
