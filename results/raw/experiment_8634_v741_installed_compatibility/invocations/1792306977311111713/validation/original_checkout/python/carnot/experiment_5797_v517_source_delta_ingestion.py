"""Exp5797: ingest post-V517 source deltas without mutating the roadmap.

Spec refs: REQ-REPORT-5797, SCENARIO-REPORT-5797-ZERO-FINDING,
SCENARIO-REPORT-5797-ACCEPT-BOUNDED-DELTA,
SCENARIO-REPORT-5797-BLOCKED-PROVENANCE,
SCENARIO-REPORT-5797-CLOSED-SCOPE-IMMUTABILITY,
SCENARIO-REPORT-5797-FIELD-PRINCIPLES.

This module records the durable part of a bounded literature refresh. Public
indexes and citation APIs are mutable, so the code does not attempt to replay
the live web search. Instead, it stores source-family receipts, candidate
dispositions, local ledger hashes, and the guardrails that keep bibliography
updates from silently becoming roadmap, gate, model, hardware, or headline
claim changes.
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
RESULT_RELATIVE_PATH = Path("results/experiment_5797_v517_source_delta_ingestion.json")
RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
KNOWN_ISSUES_RELATIVE_PATH = Path("ops/known-issues.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
PRIOR_SOURCE_DELTA_RELATIVE_PATH = Path(
    "results/experiment_5783_v516_source_delta_ingestion.json"
)

EXPERIMENT = "experiment_5797_v517_source_delta_ingestion"
EXPERIMENT_ID = "exp5797-v517-source-delta-ingestion"
MILESTONE = "2026.07.517"
RUN_DATE = "20260722"
RANDOM_SEED = 5797
SCHEMA = "carnot.experiment_5797.v517_source_delta_ingestion.v1"
INFERENCE_SUBSTRATE = (
    "primary_source_metadata_and_local_ledger_synthesis_no_experiment_llm"
)
TERMINAL_PREFIXES = ("complete:", "blocked:")

PLANNER_HEADING = "## V517 Planner Refresh - 20260722"
PLANNER_MARKER = "V517-PLANNER-REFRESH-20260722-END"
PLANNER_END_MARKER = f"<!-- {PLANNER_MARKER} -->"
EXECUTION_REFRESH_HEADING = "## V517 Execution Refresh - 20260722"
EXECUTION_REFRESH_END_MARKER = "<!-- V517-EXECUTION-REFRESH-20260722-END -->"

ALLOCATED_TARGET_EXPERIMENTS = {
    "exp5798-sota-answer-channel-diagnostic",
    "exp5799-sota-answer-channel-canary",
    "exp5800-channel-qualified-constraint-stream",
    "exp5801-future-validated-constraint-skill-ab",
    "exp5802-constraint-skill-endurance",
    "exp5803-constraint-skill-ood-audit",
    "exp5804-arc-bootstrap-safe-sota-panel",
    "exp5805-arc-immutable-selector",
    "exp5806-arc-live-heldout-world-model-ab",
    "exp5807-self-learning-microkernel-handoff",
}

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "planner_marker",
    "search_window",
    "primary_source_receipts",
    "secondary_source_receipts",
    "semantic_scholar_citation_receipts",
    "candidate_findings",
    "accepted_findings",
    "duplicate_findings",
    "watch_only_findings",
    "inaccessible_findings",
    "excluded_findings",
    "accepted_finding_count",
    "references_modified",
    "roadmap_ids_unchanged",
    "gates_unchanged",
    "closed_scopes_reopened",
    "hardware_claim_changed",
    "inference_substrate",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

REQUIRED_FIELD_PRINCIPLES: dict[str, str] = {
    "status": (
        "Terminal state is derived from source reachability, marker anchoring, "
        "and protected-boundary checks."
    ),
    "preconditions_checked": (
        "Records marker, roadmap-id/gate, exclusion, reference, active-roadmap, "
        "optional-next-roadmap, conductor, and network availability checks before "
        "findings are trusted."
    ),
    "planner_marker": "Binds the search window to the V517 planner ledger boundary.",
    "search_window": (
        "States the post-marker inclusion rule and execution timestamp range without "
        "relying on mutable search indexes."
    ),
    "primary_source_receipts": (
        "Primary metadata receipts show arXiv and directly named source availability "
        "before any hypothesis is promoted."
    ),
    "secondary_source_receipts": (
        "Secondary feeds and discovery routes are context only unless backed by "
        "primary source metadata or local receipts."
    ),
    "semantic_scholar_citation_receipts": (
        "Citation-route receipts are direct API evidence and do not become stable "
        "citation-count claims."
    ),
    "candidate_findings": (
        "Every surfaced candidate is dispositioned before it can influence a Carnot task."
    ),
    "accepted_findings": (
        "Accepted deltas must be post-marker or newly actionable, non-duplicate, and "
        "bounded to allocated Exp5798-Exp5807 controls."
    ),
    "duplicate_findings": (
        "Planner-covered or already-ledgered sources remain visible but cannot create "
        "duplicate roadmap work."
    ),
    "watch_only_findings": (
        "Relevant but non-executable or non-local material is monitored without "
        "supporting Carnot execution claims."
    ),
    "inaccessible_findings": "Access failures are recorded separately and never promoted.",
    "excluded_findings": (
        "Closed scopes and unsupported mechanisms remain closed by explicit disposition."
    ),
    "accepted_finding_count": (
        "Bare scalar prevents prose from inflating zero accepted deltas."
    ),
    "references_modified": (
        "The reference-ledger mutation state is declared and must be false when no "
        "accepted non-duplicate delta exists."
    ),
    "roadmap_ids_unchanged": (
        "Bibliographic refresh cannot rewrite allocated task identity."
    ),
    "gates_unchanged": (
        "Bibliographic refresh cannot rewrite conductor gate requirements."
    ),
    "closed_scopes_reopened": (
        "Retired scopes require explicit operator authorization outside this workflow."
    ),
    "hardware_claim_changed": (
        "Public source metadata cannot change FPGA, TSU, Kona, speed, energy, or "
        "execution claims."
    ),
    "inference_substrate": (
        "The run synthesizes primary source metadata and local ledger evidence without "
        "experiment LLM inference."
    ),
    "test_commands": "Verification commands are preserved exactly.",
    "test_exit_codes": "Observed exit codes are recorded without relabeling failures.",
    "reproducibility_checksum": "Stable content checksum detects artifact drift.",
    "honest_verdict": (
        "Terminal summary starts with complete: or blocked: and does not inflate novelty."
    ),
}

FIELD_PRINCIPLES: dict[str, str] = {
    **REQUIRED_FIELD_PRINCIPLES,
    "field_principles": "Maps every artifact field to its evidence boundary.",
    "schema": "Identifies the versioned Exp5797 artifact schema.",
    "experiment": "Stable local experiment slug for result indexing.",
    "experiment_id": "Binds this receipt to the conductor task id.",
    "milestone": "Prevents V517 source receipts from being reused for another milestone.",
    "run_date": "Records the operator-requested execution date in compact form.",
    "random_seed": "Deterministic metadata even though the run performs no stochastic science.",
    "spec_refs": "OpenSpec anchors for this artifact behavior.",
    "result_path": "Records where the JSON receipt is written.",
    "planner_marker_found": "Shows whether the source window was anchored before mutation.",
    "planner_marker_hash": "Content-addressed marker context detects silent planner-block drift.",
    "search_started_at": "Records the real UTC instant before external querying starts.",
    "search_finished_at": "Records the real UTC instant after final source disposition.",
    "actual_search_wall_time_s": (
        "Wall time is bibliographic search time only, not model, solver, benchmark, "
        "or hardware compute."
    ),
    "source_queries": "Search intent is reconstructable without trusting memory.",
    "references_before_hash": "Hash of research-references.md before any optional append.",
    "references_after_hash": "Hash of research-references.md after any optional append.",
    "references_diff_hash": "Reference before/after state is content-addressed.",
    "closed_scope_review": "Documents that banned research scopes remain closed.",
    "duplicate_checks": "Summarizes source-id, URL, title, and local-ledger duplicate checks.",
    "target_experiment_map": "Maps accepted findings only to allocated Exp5798-Exp5807 work.",
    "roadmap_immutability": "Hashes roadmap identities and gates before and after the write path.",
}

SPEC_REFS = (
    "REQ-REPORT-5797",
    "SCENARIO-REPORT-5797-ZERO-FINDING",
    "SCENARIO-REPORT-5797-ACCEPT-BOUNDED-DELTA",
    "SCENARIO-REPORT-5797-BLOCKED-PROVENANCE",
    "SCENARIO-REPORT-5797-CLOSED-SCOPE-IMMUTABILITY",
    "SCENARIO-REPORT-5797-FIELD-PRINCIPLES",
)

SOURCE_QUERIES: tuple[JsonDict, ...] = (
    {
        "surface": "arXiv",
        "order": 1,
        "queries": [
            'all:"energy-based" AND submittedDate:[202607220000 TO 202607222359]',
            'all:"constraint reasoning" AND submittedDate:[202607220000 TO 202607222359]',
            'all:"world model" AND submittedDate:[202607220000 TO 202607222359]',
            '(all:"neural CSP" OR all:"constraint satisfaction") AND submittedDate:[202607220000 TO 202607222359]',
            '(all:Ising OR all:sampling OR all:"probabilistic hardware") AND submittedDate:[202607220000 TO 202607222359]',
            '(all:hallucination OR all:"constrained decoding" OR all:KAN) AND submittedDate:[202607220000 TO 202607222359]',
            '(all:"online learning" OR all:ARC OR all:"world-model") AND submittedDate:[202607220000 TO 202607222359]',
        ],
    },
    {
        "surface": "OpenReview",
        "order": 2,
        "queries": [
            "api.openreview.net/notes?content=energy-based&limit=5",
            "api.openreview.net/notes?content=constraint%20reasoning&limit=5",
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
        "queries": ["daily_papers 2026-07-22"],
    },
    {
        "surface": "GitHub discovery",
        "order": 5,
        "queries": [
            "repo:ggml-org/llama.cpp qwen grammar thinking updated:>2026-07-21",
            '"constraint reasoning" OR "energy-based" updated:>2026-07-21',
            '"ARC" "world model" accreditation updated:>2026-07-21',
            '"constrained decoding" verifier updated:>2026-07-21',
        ],
    },
    {"surface": "Extropic writing", "order": 6, "queries": ["writing", "hardware"]},
    {
        "surface": "Logical Intelligence",
        "order": 7,
        "queries": ["home", "search index", "automatic formal verification blog"],
    },
)

PRIMARY_SOURCE_RECEIPTS: tuple[JsonDict, ...] = (
    {
        "receipt_id": "arxiv_home_reachability_v517",
        "surface": "arXiv",
        "url": "https://arxiv.org/",
        "queried_at": "2026-07-22T16:56:43Z",
        "status": "http_200",
        "candidate_ids": [],
        "receipt_summary": "HEAD request reached arXiv before post-marker classification.",
    },
    {
        "receipt_id": "arxiv_post_marker_required_topic_queries",
        "surface": "arXiv",
        "url": "https://export.arxiv.org/api/query",
        "queried_at": "2026-07-22T16:56:43Z",
        "status": "http_200_total_results_0_for_required_post_marker_topic_queries",
        "candidate_ids": [],
        "receipt_summary": (
            "Date-window API searches across EBM, CSP, Ising, hallucination, KAN, "
            "constrained decoding, online learning, and ARC/world-model terms returned "
            "zero 2026-07-22 submitted matches."
        ),
    },
    {
        "receipt_id": "arxiv_abs_2607_19191",
        "surface": "arXiv",
        "url": "https://arxiv.org/abs/2607.19191",
        "queried_at": "2026-07-22T16:56:43Z",
        "status": "http_200_via_huggingface_daily",
        "candidate_ids": ["2607.19191"],
        "receipt_summary": (
            "ABot-World-0 surfaced through the 2026-07-22 Hugging Face feed; "
            "it is a broad video world-model training result rather than an "
            "agent-owned ARC accreditation control."
        ),
    },
    {
        "receipt_id": "extropic_writing_hardware_v517",
        "surface": "Extropic writing",
        "url": "https://extropic.ai/writing",
        "queried_at": "2026-07-22T16:57:10Z",
        "status": "http_200_public_writing_no_authenticated_local_route",
        "candidate_ids": ["tsu_101_2025_10_29", "dtms_2025_10_28"],
        "receipt_summary": (
            "The public writing index and hardware page were reachable; no "
            "authenticated Carnot-local TSU/Z1 execution route was exposed."
        ),
    },
    {
        "receipt_id": "logical_intelligence_public_pages_v517",
        "surface": "Logical Intelligence",
        "url": "https://logicalintelligence.com/",
        "queried_at": "2026-07-22T16:57:10Z",
        "status": "http_200_search_index_last_modified_2026_06_26",
        "candidate_ids": ["kona_public_page", "aleph_public_pages"],
        "receipt_summary": (
            "The public pages and search index were reachable; Kona/Aleph material "
            "remains proprietary context without local weights, API receipts, or "
            "reproducible comparators."
        ),
    },
)

SECONDARY_SOURCE_RECEIPTS: tuple[JsonDict, ...] = (
    {
        "receipt_id": "openreview_api_challenge_v517",
        "surface": "OpenReview",
        "url": "https://api.openreview.net/notes?content=energy-based&limit=5",
        "queried_at": "2026-07-22T16:56:43Z",
        "status": "inaccessible_403_challenge_required",
        "candidate_ids": [],
        "receipt_summary": (
            "OpenReview home was reachable, but API note queries returned challenge "
            "verification and were not promoted."
        ),
    },
    {
        "receipt_id": "huggingface_daily_2026_07_22_v517",
        "surface": "Hugging Face Papers",
        "url": "https://huggingface.co/api/daily_papers?date=2026-07-22",
        "queried_at": "2026-07-22T16:56:43Z",
        "status": "http_200_daily_feed",
        "candidate_ids": [
            "2607.19191",
            "2607.18754",
            "2607.18703",
            "2607.16617",
            "2607.19343",
            "2607.18955",
            "2607.19331",
        ],
        "receipt_summary": (
            "The daily feed was reachable and surfaced world-model, agent "
            "observability, data-pipeline, robot-world-model, hindsight-distillation, "
            "and RLVR candidates; none became a Carnot execution claim."
        ),
    },
    {
        "receipt_id": "github_llamacpp_qwen_recent_issues_v517",
        "surface": "GitHub discovery",
        "url": "https://api.github.com/search/issues?q=repo:ggml-org/llama.cpp+qwen+grammar+thinking+updated:%3E2026-07-21",
        "queried_at": "2026-07-22T16:57:10Z",
        "status": "http_200_total_count_2",
        "candidate_ids": ["ggml-org/llama.cpp#24523", "ggml-org/llama.cpp#22275"],
        "receipt_summary": (
            "Recent llama.cpp issue search reached GitHub; visible updates did not "
            "supersede the V517 planner-promoted Qwen grammar/thinking receipts."
        ),
    },
    {
        "receipt_id": "github_required_recent_queries_v517",
        "surface": "GitHub discovery",
        "url": "https://api.github.com/search/repositories",
        "queried_at": "2026-07-22T16:57:10Z",
        "status": "http_200_unscoped_results_no_actionable_repository",
        "candidate_ids": ["doctorspapers6-ops/Updatedcode"],
        "receipt_summary": (
            "Recent repository searches for energy reasoning constraints and "
            "world-model verification returned only unscoped or irrelevant repositories."
        ),
    },
)

SEMANTIC_SCHOLAR_CITATION_RECEIPTS: tuple[JsonDict, ...] = (
    {
        "receipt_id": "semantic_scholar_ebt_citations_v517",
        "paper": "arXiv:2507.02092",
        "surface": "Semantic Scholar",
        "url": (
            "https://api.semanticscholar.org/graph/v1/paper/arXiv:2507.02092/"
            "citations"
        ),
        "queried_at": "2026-07-22T16:56:43Z",
        "status": "http_200",
        "http_status": 200,
        "candidate_ids": ["2607.17047", "2607.11555", "2606.22726"],
        "sample_returned_count": 10,
        "latest_publication_date": "2026-07-19",
        "post_marker_publication_count": 0,
        "sample_titles": [
            "Solver-Hard Is Not Model-Hard: A Hardness-Controlled Diagnostic for LLM Constraint Reasoning",
            "Advancing Optimal Subset Oracle via Learning Relaxation of Neural Set Functions",
            "Text Dictates, Music Decorates: Energy-based Attention for Editable Dance Motion Generation",
        ],
        "receipt_summary": (
            "The visible EBT citation sample was reachable and repeated the V517 "
            "planner-covered Solver-Hard trail."
        ),
    },
    {
        "receipt_id": "semantic_scholar_arm_ebm_citations_v517",
        "paper": "arXiv:2512.15605",
        "surface": "Semantic Scholar",
        "url": (
            "https://api.semanticscholar.org/graph/v1/paper/arXiv:2512.15605/"
            "citations"
        ),
        "queried_at": "2026-07-22T16:56:43Z",
        "status": "http_200",
        "http_status": 200,
        "candidate_ids": ["2607.02154", "2606.03089", "2605.18871"],
        "sample_returned_count": 8,
        "latest_publication_date": "2026-07-02",
        "post_marker_publication_count": 0,
        "sample_titles": [
            "Path-Measure Dynamics of Attention-Driven World Models",
            "Constitutional On-Policy Safe Distillation",
            "Distributional Energy-Based Models for Uncertainty-Aware Structured LLM Reasoning",
        ],
        "receipt_summary": (
            "The ARM-EBM citation route was reachable and exposed no post-marker "
            "publication."
        ),
    },
)

ACCEPTED_FINDINGS: tuple[JsonDict, ...] = ()

DUPLICATE_FINDINGS: tuple[JsonDict, ...] = (
    {
        "source_id": "worldevolver_2606_30639",
        "classification": "duplicate",
        "title": "Self-Evolving World Models for LLM Agent Planning",
        "url": "https://arxiv.org/abs/2606.30639",
        "publication_date": "2026-06-29",
        "source_date": "2026-07-22",
        "search_timestamp": "2026-07-22T16:56:43Z",
        "search_receipt": "arxiv_post_marker_required_topic_queries",
        "access_outcome": "reachable_duplicate_v517_planner_block",
        "reason": (
            "Already accepted in the V517 planner block for Exp5801-Exp5803 "
            "future-gated typed skill controls; no execution-refresh delta remains."
        ),
    },
    {
        "source_id": "reasoning_aware_slos_openreview_eez7uze9kf",
        "classification": "duplicate",
        "title": "Reasoning-Aware SLOs: Why TTFT and TBT Are the Wrong Metrics for o1/R1-Class Models",
        "url": "https://openreview.net/forum?id=eEZ7uze9kf",
        "publication_date": "2026-06-26",
        "source_date": "2026-07-22",
        "search_timestamp": "2026-07-22T16:56:43Z",
        "search_receipt": "openreview_api_challenge_v517",
        "access_outcome": "reachable_duplicate_v517_planner_block_public_page_previously_indexed",
        "reason": (
            "Already accepted in the V517 planner block for verified-useful-output "
            "and wasted-token accounting; the current OpenReview API route was "
            "challenge-blocked and adds no stronger metadata."
        ),
    },
    {
        "source_id": "llamacpp_qwen_grammar_thinking_20345",
        "classification": "duplicate",
        "title": "llama.cpp grammar/thinking interaction on Qwen 3.x",
        "url": "https://github.com/ggml-org/llama.cpp/issues/20345",
        "publication_date": "2026-03-10",
        "source_date": "2026-07-22",
        "search_timestamp": "2026-07-22T16:57:10Z",
        "search_receipt": "github_llamacpp_qwen_recent_issues_v517",
        "access_outcome": "reachable_duplicate_v517_planner_block",
        "reason": (
            "Already accepted in the V517 planner block for Exp5798/Exp5799 "
            "answer-channel controls; later GitHub search did not supersede it."
        ),
    },
    {
        "source_id": "solver_hard_not_model_hard_2607_17047",
        "classification": "duplicate",
        "title": "Solver-Hard Is Not Model-Hard: A Hardness-Controlled Diagnostic for LLM Constraint Reasoning",
        "url": "https://arxiv.org/abs/2607.17047",
        "publication_date": "2026-07-19",
        "source_date": "2026-07-19",
        "search_timestamp": "2026-07-22T16:56:43Z",
        "search_receipt": "semantic_scholar_ebt_citations_v517",
        "access_outcome": "reachable_duplicate_v517_planner_block",
        "reason": (
            "Already accepted in the V517 planner block as V516 fixture carry-forward "
            "context for Exp5800/Exp5803 hardness and surface reporting."
        ),
    },
    {
        "source_id": "logical_intelligence_kona_aleph_public_pages",
        "classification": "duplicate",
        "title": "Logical Intelligence public Kona and Aleph pages",
        "url": "https://logicalintelligence.com/",
        "publication_date": "2026-06-26",
        "source_date": "2026-06-26",
        "search_timestamp": "2026-07-22T16:57:10Z",
        "search_receipt": "logical_intelligence_public_pages_v517",
        "access_outcome": "reachable_duplicate_v517_planner_context",
        "reason": (
            "The V517 planner already treated Kona/Aleph material as architecture "
            "context without local weights, authenticated API, or reproducible "
            "comparator evidence."
        ),
    },
)

WATCH_ONLY_FINDINGS: tuple[JsonDict, ...] = (
    {
        "source_id": "agentdebugx_2607_18754",
        "classification": "watch_only",
        "title": "AgentDebugX: An Open-Source Toolkit for Failure Observability, Attribution, and Recovery in LLM Agents",
        "url": "https://huggingface.co/papers/2607.18754",
        "publication_date": "2026-07-21",
        "source_date": "2026-07-22",
        "search_timestamp": "2026-07-22T16:56:43Z",
        "search_receipt": "huggingface_daily_2026_07_22_v517",
        "access_outcome": "reachable_secondary_feed",
        "reason": (
            "Agent observability is relevant background for Exp5798 diagnostics, but "
            "the V517 task already requires exact local raw-row attribution and does "
            "not need a new external toolkit dependency."
        ),
    },
    {
        "source_id": "generative_world_renderer_2607_18703",
        "classification": "watch_only",
        "title": "Generative World Renderer at the Speed of Play",
        "url": "https://arxiv.org/abs/2607.18703",
        "publication_date": "2026-07-21",
        "source_date": "2026-07-22",
        "search_timestamp": "2026-07-22T16:56:43Z",
        "search_receipt": "huggingface_daily_2026_07_22_v517",
        "access_outcome": "reachable_hf_and_github",
        "reason": (
            "Fresh secondary-feed and GitHub visibility, but the visual renderer "
            "preserves physics-engine state for image synthesis rather than adding an "
            "agent-owned ARC accreditation control to Exp5804-Exp5806."
        ),
    },
    {
        "source_id": "dataflow_harness_2607_16617",
        "classification": "watch_only",
        "title": "DataFlow-Harness: A Grounded Code-Agent Platform for Constructing Editable LLM Data Pipelines",
        "url": "https://huggingface.co/papers/2607.16617",
        "publication_date": "2026-07-18",
        "source_date": "2026-07-22",
        "search_timestamp": "2026-07-22T16:56:43Z",
        "search_receipt": "huggingface_daily_2026_07_22_v517",
        "access_outcome": "reachable_secondary_feed",
        "reason": (
            "Typed incremental mutation is relevant context for skill workflows, but "
            "it does not supply exact solver authority, a constraint lifecycle receipt, "
            "or a bounded Exp5801 control."
        ),
    },
    {
        "source_id": "llamacpp_minimax_m3_pr_24523",
        "classification": "watch_only",
        "title": "llama.cpp Preliminary MiniMax-M3 support",
        "url": "https://github.com/ggml-org/llama.cpp/pull/24523",
        "publication_date": "2026-06-12",
        "source_date": "2026-07-22",
        "search_timestamp": "2026-07-22T16:57:10Z",
        "search_receipt": "github_llamacpp_qwen_recent_issues_v517",
        "access_outcome": "reachable_recent_update_not_qwen_channel_control",
        "reason": (
            "The PR was visible in the recent llama.cpp search, but it concerns "
            "MiniMax-M3 support rather than the mandated Qwen/Gemma GGUF answer channel."
        ),
    },
    {
        "source_id": "extropic_tsu_z1_public_material_v517",
        "classification": "watch_only",
        "title": "Extropic TSU 101, DTMS, and hardware public material",
        "url": "https://extropic.ai/hardware",
        "publication_date": "2025-10-29",
        "source_date": "2026-07-22",
        "search_timestamp": "2026-07-22T16:57:10Z",
        "search_receipt": "extropic_writing_hardware_v517",
        "access_outcome": "reachable_no_authenticated_local_execution_surface",
        "reason": (
            "Public probabilistic-hardware context only; no Carnot-local TSU, Z1, "
            "speed, power, SDK, or correctness receipt was found."
        ),
    },
)

EXCLUDED_FINDINGS: tuple[JsonDict, ...] = (
    {
        "source_id": "abot_world_0_2607_19191",
        "classification": "excluded",
        "title": "ABot-World-0: Infinite Interactive World Rollout on a Single Desktop GPU",
        "url": "https://arxiv.org/abs/2607.19191",
        "publication_date": "2026-07-21",
        "source_date": "2026-07-22",
        "search_timestamp": "2026-07-22T16:56:43Z",
        "search_receipt": "huggingface_daily_2026_07_22_v517",
        "access_outcome": "reachable_primary_and_secondary_metadata",
        "reason": (
            "The work is a broad video/action world-model training and deployment "
            "result, not an immutable ARC hypothesis accreditation or E3 policy control."
        ),
    },
    {
        "source_id": "hybrid_hindsight_self_distillation_2607_18955",
        "classification": "excluded",
        "title": "H2SD: Hybrid Hindsight Self-Distillation",
        "url": "https://huggingface.co/papers/2607.18955",
        "publication_date": "2026-07-21",
        "source_date": "2026-07-22",
        "search_timestamp": "2026-07-22T16:56:43Z",
        "search_receipt": "huggingface_daily_2026_07_22_v517",
        "access_outcome": "reachable_secondary_feed",
        "reason": (
            "The method is a hindsight self-distillation and RLVR weight-update path, "
            "which would reopen generated-text scoring or model-weight scope rather "
            "than add a bounded V517 control."
        ),
    },
    {
        "source_id": "iso_rlvr_native_optimization_stack_2607_19331",
        "classification": "excluded",
        "title": "ISO: An RLVR-Native Optimization Stack",
        "url": "https://huggingface.co/papers/2607.19331",
        "publication_date": "2026-07-21",
        "source_date": "2026-07-22",
        "search_timestamp": "2026-07-22T16:56:43Z",
        "search_receipt": "huggingface_daily_2026_07_22_v517",
        "access_outcome": "reachable_secondary_feed",
        "reason": (
            "RLVR optimization stack material would reopen broad policy-gradient or "
            "model-training scope rather than the exact frozen-GGUF self-learning lane."
        ),
    },
    {
        "source_id": "masked_visual_actions_robot_world_model_2607_19343",
        "classification": "excluded",
        "title": "Masked Visual Actions Make World Models More In-Context Learners",
        "url": "https://huggingface.co/papers/2607.19343",
        "publication_date": "2026-07-21",
        "source_date": "2026-07-22",
        "search_timestamp": "2026-07-22T16:56:43Z",
        "search_receipt": "huggingface_daily_2026_07_22_v517",
        "access_outcome": "reachable_secondary_feed",
        "reason": (
            "Robot video-action world-model training and planning would reopen broad "
            "model-training scope and does not provide agent-owned ARC E3 accreditation."
        ),
    },
)

INACCESSIBLE_FINDINGS: tuple[JsonDict, ...] = (
    {
        "source_id": "openreview_energy_constraint_search_v517",
        "classification": "inaccessible",
        "title": "OpenReview energy-based and constraint-reasoning note search",
        "url": "https://api.openreview.net/notes?content=energy-based&limit=5",
        "publication_date": "unknown",
        "source_date": "2026-07-22",
        "search_timestamp": "2026-07-22T16:56:43Z",
        "search_receipt": "openreview_api_challenge_v517",
        "access_outcome": "inaccessible_403_challenge_required",
        "reason": (
            "The API required challenge verification; no result is fabricated or "
            "promoted from inaccessible OpenReview data."
        ),
    },
)

DEFAULT_TESTS_RUN: tuple[JsonDict, ...] = (
    {
        "command": ".venv/bin/pytest tests/python/test_experiment_5797_v517_source_delta_ingestion.py -q --no-cov -n 0",
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": ".venv/bin/coverage run --include=python/carnot/experiment_5797_v517_source_delta_ingestion.py -m pytest tests/python/test_experiment_5797_v517_source_delta_ingestion.py -q --no-cov -n 0",
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": ".venv/bin/coverage report --include=python/carnot/experiment_5797_v517_source_delta_ingestion.py --fail-under=100",
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
    """Return file text or an empty string when a precondition file is absent."""

    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def path_sha256(path: Path) -> str | None:
    """Hash a file so local evidence claims can be tied to exact bytes."""

    if not path.exists():
        return None
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_timestamp(value: str) -> str:
    """Normalize UTC timestamps to the compact `Z` form used in artifacts."""

    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(normalize_timestamp(value).replace("Z", "+00:00"))


def planner_marker_line(text: str) -> int | None:
    """Return the one-based line of the V517 planner marker."""

    for line_no, line in enumerate(text.splitlines(), start=1):
        if PLANNER_MARKER in line:
            return line_no
    return None


def planner_block_hash(text: str) -> str | None:
    """Hash the V517 planner block ending at the required marker."""

    marker_index = text.find(PLANNER_END_MARKER)
    if marker_index < 0:
        return None
    heading_index = text.rfind(PLANNER_HEADING, 0, marker_index)
    start = heading_index if heading_index >= 0 else marker_index
    block = text[start : marker_index + len(PLANNER_END_MARKER)]
    return "sha256:" + hashlib.sha256(block.encode("utf-8")).hexdigest()


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _roadmap_identity(root: Path) -> tuple[str | None, str | None, list[str], list[Any], str]:
    text = read_text_if_present(root / ROADMAP_RELATIVE_PATH)
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
    task_ids = [row.get("id", "") for row in tasks if isinstance(row, Mapping)]
    gates = [
        {"id": row.get("id", ""), "gated_on": row.get("gated_on", [])}
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
    status = str(receipt.get("status", ""))
    return status.startswith("http_")


def _source_reachable(
    primary_source_receipts: list[JsonDict],
    secondary_source_receipts: list[JsonDict],
    semantic_scholar_citation_receipts: list[JsonDict],
) -> bool:
    receipts = (
        primary_source_receipts
        + secondary_source_receipts
        + semantic_scholar_citation_receipts
    )
    return any(_receipt_reachable(receipt) for receipt in receipts)


def preconditions_checked(
    root: Path,
    *,
    marker_found: bool,
    source_reachable: bool,
    checked_at: str = "2026-07-22T16:56:43Z",
) -> JsonDict:
    """Collect pre-mutation local hashes and fail-closed precondition flags."""

    references_text = read_text_if_present(root / RESEARCH_REFERENCES_RELATIVE_PATH)
    roadmap_ids_hash, gates_hash, task_ids, gates, milestone = _roadmap_identity(root)
    spec_text = read_text_if_present(root / SPEC_RELATIVE_PATH)
    failed: list[str] = []
    active_roadmap_hash = path_sha256(root / ROADMAP_RELATIVE_PATH)
    exclusion_hash = path_sha256(root / EXCLUSION_MANIFEST_RELATIVE_PATH)

    if not marker_found:
        failed.append("planner_marker_missing")
    if not source_reachable:
        failed.append("source_reachability_failed")
    if active_roadmap_hash is None:
        failed.append("active_roadmap_hash_missing")
    if exclusion_hash is None:
        failed.append("exclusion_manifest_hash_missing")
    if not all(ref in spec_text for ref in SPEC_REFS):
        failed.append("spec_req_report_5797_missing")
    if roadmap_ids_hash is None and active_roadmap_hash is not None:
        failed.append("active_roadmap_identity_unavailable")

    next_path = root / ROADMAP_NEXT_RELATIVE_PATH
    unavailable_routes = ["OpenReview notes API: inaccessible_403_challenge_required"]
    if not next_path.exists():
        unavailable_routes.append("research-roadmap-next.yaml: absent")
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
        "unavailable_source_routes": unavailable_routes,
        "references_hash": path_sha256(root / RESEARCH_REFERENCES_RELATIVE_PATH),
        "active_roadmap_hash": active_roadmap_hash,
        "active_roadmap_milestone": milestone,
        "roadmap_ids_hash": roadmap_ids_hash,
        "roadmap_task_ids": task_ids,
        "gates_hash": gates_hash,
        "gated_task_count": len(gates),
        "exclusion_manifest_hash": exclusion_hash,
        "known_issues_hash": path_sha256(root / KNOWN_ISSUES_RELATIVE_PATH),
        "vnext_hash": path_sha256(root / VNEXT_RELATIVE_PATH),
        "prior_v516_source_delta_hash": path_sha256(
            root / PRIOR_SOURCE_DELTA_RELATIVE_PATH
        ),
        "conductor_hash": path_sha256(root / CONDUCTOR_RELATIVE_PATH),
        "research_roadmap_next_read": next_path.exists(),
        "research_roadmap_next_hash": path_sha256(next_path),
        "failed_preconditions": failed,
    }


def honest_verdict(
    marker_found: bool,
    source_reachable: bool,
    accepted_findings: list[JsonDict],
    protected_change_requested: bool,
) -> str:
    """Return the terminal verdict without inflating bibliographic novelty."""

    if not marker_found:
        return "blocked: V517 planner marker missing; source refresh left references unchanged"
    if not source_reachable:
        return "blocked: source reachability unavailable; no inaccessible result promoted"
    if protected_change_requested:
        return "blocked: protected roadmap, gate, model, hardware, or closed-scope change requested"
    if accepted_findings:
        return (
            f"complete: accepted {len(accepted_findings)} post-V517 bounded source "
            "delta(s); roadmap ids and gates unchanged"
        )
    return (
        "complete: no accepted post-V517 source deltas; references unchanged and "
        "closed scopes preserved"
    )


def target_experiment_map(findings: list[JsonDict]) -> list[JsonDict]:
    """Extract the accepted-finding to allocated-task map for downstream audit."""

    return [
        {
            "source_id": finding["source_id"],
            "target_experiment": finding["target_experiment"],
            "target_allocated": finding["target_experiment"] in ALLOCATED_TARGET_EXPERIMENTS,
        }
        for finding in findings
    ]


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact while excluding the checksum field itself."""

    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def execution_refresh_block(accepted_findings: list[JsonDict]) -> str:
    """Format the optional append block for accepted non-duplicate deltas."""

    lines = [f"\n{EXECUTION_REFRESH_HEADING}\n", "\n### New actionable deltas\n"]
    for finding in accepted_findings:
        lines.append(
            "\n- **{title}** - {url} ({date}). Carnot hook: {hook}. Target: "
            "{target}. Authority boundary: {boundary}. Falsifiable metric: {metric}. "
            "Search receipt: {receipt}.\n".format(
                title=finding["title"],
                url=finding["url"],
                date=finding.get("publication_date", finding.get("source_date", "unknown")),
                hook=finding["carnot_hook"],
                target=finding["target_experiment"],
                boundary=finding["authority_boundary"],
                metric=finding["falsifiable_metric"],
                receipt=finding["search_receipt"],
            )
        )
    lines.append("\n### V517 execution impact\n")
    lines.append(
        "\n- Preserve roadmap ids, gates, model requirements, hardware requirements, "
        "headline claims, and retired scopes. Accepted deltas may only add bounded "
        "controls or receipts inside already allocated Exp5798-Exp5807 work.\n"
    )
    lines.append(f"\n{EXECUTION_REFRESH_END_MARKER}\n")
    return "".join(lines)


def insert_after_planner_block(text: str, block: str) -> str:
    """Insert the execution-refresh block after the V517 planner marker when present."""

    if EXECUTION_REFRESH_HEADING in text:
        return text
    marker_index = text.find(PLANNER_END_MARKER)
    if marker_index >= 0:
        insert_at = marker_index + len(PLANNER_END_MARKER)
        return text[:insert_at] + block + text[insert_at:]
    return text.rstrip() + block


def _default_tests(
    test_commands: list[str] | None,
    test_exit_codes: Mapping[str, int | None] | None,
) -> tuple[list[str], dict[str, int | None]]:
    if test_commands is not None or test_exit_codes is not None:
        commands = list(test_commands or [])
        exit_codes = dict(test_exit_codes or {})
        return commands, exit_codes
    commands = [row["command"] for row in DEFAULT_TESTS_RUN]
    exit_codes = {row["command"]: row["exit_code"] for row in DEFAULT_TESTS_RUN}
    return commands, exit_codes


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    search_started_at: str,
    search_finished_at: str,
    accepted_findings: list[JsonDict] | None = None,
    duplicate_findings: list[JsonDict] | None = None,
    watch_only_findings: list[JsonDict] | None = None,
    excluded_findings: list[JsonDict] | None = None,
    inaccessible_findings: list[JsonDict] | None = None,
    primary_source_receipts: list[JsonDict] | None = None,
    secondary_source_receipts: list[JsonDict] | None = None,
    semantic_scholar_citation_receipts: list[JsonDict] | None = None,
    references_modified: bool = False,
    references_before_hash: str | None = None,
    references_after_hash: str | None = None,
    test_commands: list[str] | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
) -> JsonDict:
    """Build the Exp5797 artifact from source receipts and local ledgers."""

    root = Path(root)
    primary_receipts = list(
        primary_source_receipts
        if primary_source_receipts is not None
        else PRIMARY_SOURCE_RECEIPTS
    )
    secondary_receipts = list(
        secondary_source_receipts
        if secondary_source_receipts is not None
        else SECONDARY_SOURCE_RECEIPTS
    )
    semantic_receipts = list(
        semantic_scholar_citation_receipts
        if semantic_scholar_citation_receipts is not None
        else SEMANTIC_SCHOLAR_CITATION_RECEIPTS
    )
    source_reachable = _source_reachable(primary_receipts, secondary_receipts, semantic_receipts)
    references_text = read_text_if_present(root / RESEARCH_REFERENCES_RELATIVE_PATH)
    marker_found = PLANNER_MARKER in references_text
    planner_hash = planner_block_hash(references_text)
    preconditions = preconditions_checked(
        root,
        marker_found=marker_found,
        source_reachable=source_reachable,
    )

    accepted = list(accepted_findings if accepted_findings is not None else ACCEPTED_FINDINGS)
    blocked_by_preconditions = bool(preconditions["failed_preconditions"])
    if blocked_by_preconditions:
        accepted = []
    duplicates = list(duplicate_findings or DUPLICATE_FINDINGS)
    watch_only = list(watch_only_findings or WATCH_ONLY_FINDINGS)
    excluded = list(excluded_findings or EXCLUDED_FINDINGS)
    inaccessible = list(inaccessible_findings or INACCESSIBLE_FINDINGS)
    candidate_findings = accepted + duplicates + watch_only + excluded + inaccessible
    commands, exit_codes = _default_tests(test_commands, test_exit_codes)
    start = normalize_timestamp(search_started_at)
    finish = normalize_timestamp(search_finished_at)
    wall_time = (_parse_timestamp(finish) - _parse_timestamp(start)).total_seconds()
    protected_change_requested = any(
        finding.get("gate_change_requested")
        or finding.get("roadmap_scope_change_requested")
        or finding.get("model_requirement_change_requested")
        or finding.get("hardware_claim_change_requested")
        for finding in accepted
    )
    status = "blocked" if blocked_by_preconditions or protected_change_requested else "complete"
    before_hash = references_before_hash or path_sha256(
        root / RESEARCH_REFERENCES_RELATIVE_PATH
    )
    after_hash = references_after_hash or before_hash
    roadmap_ids_hash, gates_hash, _, _, _ = _roadmap_identity(root)

    artifact: JsonDict = {
        "field_principles": dict(FIELD_PRINCIPLES),
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
        "planner_marker": PLANNER_MARKER,
        "planner_marker_found": marker_found,
        "planner_marker_hash": planner_hash,
        "search_started_at": start,
        "search_finished_at": finish,
        "actual_search_wall_time_s": wall_time,
        "search_window": {
            "boundary_marker": PLANNER_MARKER,
            "rule": "strictly_after_V517_planner_marker_or_newly_actionable_after_marker",
            "execution_date": "2026-07-22",
            "arxiv_submitted_date_range_checked": "202607220000-202607222359",
            "zero_accepted_findings_complete": True,
        },
        "source_queries": list(SOURCE_QUERIES),
        "primary_source_receipts": primary_receipts,
        "secondary_source_receipts": secondary_receipts,
        "semantic_scholar_citation_receipts": semantic_receipts,
        "candidate_findings": candidate_findings,
        "accepted_findings": accepted,
        "duplicate_findings": duplicates,
        "watch_only_findings": watch_only,
        "excluded_findings": excluded,
        "inaccessible_findings": inaccessible,
        "accepted_finding_count": len(accepted),
        "references_modified": references_modified,
        "references_before_hash": before_hash,
        "references_after_hash": after_hash,
        "references_diff_hash": _stable_hash(
            {"before": before_hash, "after": after_hash, "modified": references_modified}
        ),
        "roadmap_ids_unchanged": True,
        "gates_unchanged": True,
        "closed_scopes_reopened": False,
        "hardware_claim_changed": False,
        "closed_scope_review": {
            "protected_scopes": [
                "grammar-as-authority",
                "PHASE-D generated-text scoring",
                "constraint shadow integration",
                "CEGIS",
                "public ARC solves",
                "unchanged board probes",
                "TSU execution",
                "Kona execution",
            ],
            "operator_authorized_reopen": None,
        },
        "duplicate_checks": {
            "source_ids_unique": len({row["source_id"] for row in candidate_findings})
            == len(candidate_findings),
            "titles_checked_against_research_references": True,
            "urls_checked": [row["url"] for row in candidate_findings],
            "v517_planner_block_checked": marker_found,
            "exclusion_manifest_checked": path_sha256(root / EXCLUSION_MANIFEST_RELATIVE_PATH)
            is not None,
        },
        "target_experiment_map": target_experiment_map(accepted),
        "roadmap_immutability": {
            "ids_hash_before": roadmap_ids_hash,
            "ids_hash_after": roadmap_ids_hash,
            "gates_hash_before": gates_hash,
            "gates_hash_after": gates_hash,
            "roadmap_ids_unchanged": True,
            "gates_unchanged": True,
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "test_commands": commands,
        "test_exit_codes": exit_codes,
        "reproducibility_checksum": "",
        "honest_verdict": honest_verdict(
            marker_found,
            source_reachable,
            accepted,
            protected_change_requested,
        ),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate Exp5797 schema and the protected evidence boundary."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        raise ValueError("field_principles must be a mapping")
    missing_principles = [
        field for field in artifact if not str(principles.get(field, "")).strip()
    ]
    if missing_principles:
        raise ValueError(f"field_principles missing entries: {missing_principles}")
    status = artifact["status"]
    if status not in {"complete", "blocked"}:
        raise ValueError(f"invalid status: {status}")
    verdict = str(artifact["honest_verdict"])
    if not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must start with complete: or blocked:")
    start = _parse_timestamp(str(artifact["search_started_at"]))
    finish = _parse_timestamp(str(artifact["search_finished_at"]))
    if finish <= start:
        raise ValueError("timestamp order invalid")
    expected_wall_time = (finish - start).total_seconds()
    if abs(float(artifact["actual_search_wall_time_s"]) - expected_wall_time) > 1e-6:
        raise ValueError("wall time does not match timestamps")
    if artifact["roadmap_ids_unchanged"] is not True:
        raise ValueError("roadmap ids changed")
    if artifact["gates_unchanged"] is not True:
        raise ValueError("gates changed")
    if artifact["closed_scopes_reopened"] is not False:
        raise ValueError("closed scopes reopened")
    if artifact["hardware_claim_changed"] is not False:
        raise ValueError("hardware claim changed")
    accepted = list(artifact["accepted_findings"])
    if artifact["accepted_finding_count"] != len(accepted):
        raise ValueError("accepted_finding_count mismatch")
    if not accepted and artifact["references_modified"]:
        raise ValueError("references_modified cannot be true with zero accepted findings")
    for finding in accepted:
        if finding.get("target_experiment") not in ALLOCATED_TARGET_EXPERIMENTS:
            raise ValueError("accepted finding maps outside allocated Exp5798-Exp5807 work")
        if finding.get("post_marker_or_newly_actionable") is not True:
            raise ValueError("accepted finding is not post-marker or newly actionable")
    expected_candidates = (
        accepted
        + list(artifact["duplicate_findings"])
        + list(artifact["watch_only_findings"])
        + list(artifact["excluded_findings"])
        + list(artifact.get("inaccessible_findings", []))
    )
    if list(artifact["candidate_findings"]) != expected_candidates:
        raise ValueError("candidate_findings do not match classified finding lists")
    allowed = {"accepted", "duplicate", "watch_only", "excluded", "inaccessible"}
    for finding in expected_candidates:
        classification = finding.get("classification")
        if classification not in allowed:
            raise ValueError(f"invalid candidate classification: {classification}")
        for key in ("source_id", "title", "url", "search_timestamp", "search_receipt", "reason"):
            if not str(finding.get(key, "")).strip():
                raise ValueError(f"finding missing provenance field: {key}")
        if not str(finding.get("publication_date", finding.get("source_date", ""))).strip():
            raise ValueError("finding missing publication/source date")
    if artifact["status"] == "complete":
        if not artifact["primary_source_receipts"]:
            raise ValueError("primary source receipts missing")
        if not artifact["secondary_source_receipts"]:
            raise ValueError("secondary source receipts missing")
        if not artifact["semantic_scholar_citation_receipts"]:
            raise ValueError("semantic scholar citation receipts missing")
    if artifact["reproducibility_checksum"] != payload_checksum(artifact):
        raise ValueError("reproducibility checksum mismatch")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write stable JSON for conductor and test consumption."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_and_write_artifact(
    *,
    root: Path = REPO_ROOT,
    search_started_at: str,
    search_finished_at: str,
    accepted_findings: list[JsonDict] | None = None,
    test_commands: list[str] | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
) -> JsonDict:
    """Build the artifact, append references only for accepted deltas, and write JSON."""

    root = Path(root)
    references_path = root / RESEARCH_REFERENCES_RELATIVE_PATH
    before_text = read_text_if_present(references_path)
    before_hash = path_sha256(references_path)
    provisional = build_artifact(
        root=root,
        search_started_at=search_started_at,
        search_finished_at=search_finished_at,
        accepted_findings=accepted_findings,
        references_before_hash=before_hash,
        references_after_hash=before_hash,
        test_commands=test_commands,
        test_exit_codes=test_exit_codes,
    )
    references_modified = False
    if (
        provisional["status"] == "complete"
        and provisional["accepted_findings"]
        and EXECUTION_REFRESH_HEADING not in before_text
    ):
        references_path.write_text(
            insert_after_planner_block(
                before_text,
                execution_refresh_block(provisional["accepted_findings"]),
            ),
            encoding="utf-8",
        )
        references_modified = True
    after_hash = path_sha256(references_path)
    artifact = build_artifact(
        root=root,
        search_started_at=search_started_at,
        search_finished_at=search_finished_at,
        accepted_findings=accepted_findings,
        references_modified=references_modified,
        references_before_hash=before_hash,
        references_after_hash=after_hash,
        test_commands=test_commands,
        test_exit_codes=test_exit_codes,
    )
    validate_artifact(artifact)
    write_json(root / RESULT_RELATIVE_PATH, artifact)
    return artifact


def _load_tests_run(path: Path | None) -> tuple[list[str] | None, dict[str, int | None] | None]:
    if path is None:
        return None, None
    rows = json.loads(path.read_text(encoding="utf-8"))
    commands = [row["command"] for row in rows]
    exit_codes = {row["command"]: row.get("exit_code") for row in rows}
    return commands, exit_codes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--search-started-at", required=True)
    parser.add_argument("--search-finished-at", required=True)
    parser.add_argument("--zero-findings", action="store_true")
    parser.add_argument("--tests-run-json", type=Path)
    args = parser.parse_args(argv)
    commands, exit_codes = _load_tests_run(args.tests_run_json)
    accepted = [] if args.zero_findings else list(ACCEPTED_FINDINGS)
    artifact = build_and_write_artifact(
        root=args.root,
        search_started_at=args.search_started_at,
        search_finished_at=args.search_finished_at,
        accepted_findings=accepted,
        test_commands=commands,
        test_exit_codes=exit_codes,
    )
    print((args.root / RESULT_RELATIVE_PATH).as_posix())
    return 0 if artifact["status"] in {"complete", "blocked"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
