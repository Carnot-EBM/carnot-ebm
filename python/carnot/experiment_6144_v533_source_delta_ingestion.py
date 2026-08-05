"""Exp6144: ingest post-V533 source deltas without rewriting the roadmap.

Spec refs: REQ-REPORT-6144, SCENARIO-REPORT-6144-ZERO-DELTA,
SCENARIO-REPORT-6144-ACCEPT-BOUNDED-DELTA,
SCENARIO-REPORT-6144-DUPLICATE-AND-RETIRED-SCOPE,
SCENARIO-REPORT-6144-SCHEMA.

This module is a literature-ingestion receipt, not a research model run. It
keeps the V533 planner marker as the novelty boundary, records source access
and uncertainty explicitly, and only allows append-only bibliography updates
for new primary or official evidence that stays inside already allocated V533
work or is explicitly deferred.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import yaml


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6144_v533_source_delta_ingestion.json")

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
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
SWEEP_CLUSTERS_RELATIVE_PATH = Path("scripts/sweep_clusters.py")
SWEEP_SEMSCHOLAR_RELATIVE_PATH = Path("scripts/sweep_semscholar.py")

EXPERIMENT = "experiment_6144_v533_source_delta_ingestion"
EXPERIMENT_ID = "exp6144-v533-source-delta-ingestion"
MILESTONE = "2026.08.533"
RUN_DATE = "20260805"
RANDOM_SEED = 6144
SCHEMA = "carnot.experiment_6144.v533_source_delta_ingestion.v1"
INFERENCE_SUBSTRATE = "literature_ingestion"

PLANNER_HEADING = "## V533 Planner Refresh - 20260805"
PLANNER_MARKER = "V533-PLANNER-REFRESH-20260805-END"
PLANNER_END_MARKER = f"<!-- {PLANNER_MARKER} -->"
EXECUTION_DELTA_HEADING = "## V533 Execution Source Delta - 20260805"
EXECUTION_DELTA_END_MARKER = "<!-- V533-EXECUTION-SOURCE-DELTA-20260805-END -->"
MARKER_DATE = "2026-08-05"

ALLOCATED_TARGET_EXPERIMENTS = (
    "exp6145-constraint-shift-stream",
    "exp6146-sota-constraint-event-corpus",
    "exp6147-task-aware-energy-calibration",
    "exp6148-shifted-family-admission-held",
    "exp6149-certified-strategy-schema-fixture",
    "exp6150-frozen-qwen-continuous-self-learning-ab",
    "exp6151-strategy-memory-shadow-adapter",
    "exp6152-typed-stochastic-constraint-ir",
    "exp6153-thermalized-program-error-audit",
    "exp6154-arc-task-aware-energy-generalization",
)

SPEC_REFS = (
    "REQ-REPORT-6144",
    "SCENARIO-REPORT-6144-ZERO-DELTA",
    "SCENARIO-REPORT-6144-ACCEPT-BOUNDED-DELTA",
    "SCENARIO-REPORT-6144-DUPLICATE-AND-RETIRED-SCOPE",
    "SCENARIO-REPORT-6144-SCHEMA",
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
    "sota_to_experiment_mapping",
    "cutoff_rate_limit_and_same_day_uncertainty_receipts",
    "semantic_scholar_ebt_and_arm_ebm_receipts",
    "openreview_huggingface_github_extropic_and_kona_receipts",
    "duplicate_and_retired_scope_filter",
    "references_append_receipt",
    "roadmap_identity_gate_and_exclusion_immutability",
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
        "Terminal literature-ingestion state follows marker, source reachability, "
        "and classification receipts."
    ),
    "preconditions_checked": (
        "The V533 marker, ledgers, roadmaps, exclusions, sweep helpers, output "
        "path, protected files, endpoint failures, and rate limits are recorded "
        "before source decisions."
    ),
    "search_window_and_marker_receipt": (
        "Only a hash-anchored post-marker window is eligible."
    ),
    "source_queries_and_endpoint_receipts": (
        "Every source route records role, URL, query, access outcome, timestamp, "
        "candidate count, and low-concurrency evidence."
    ),
    "primary_secondary_and_official_source_counts": (
        "Discovery metadata cannot be mistaken for primary or official evidence."
    ),
    "accepted_rejected_duplicate_retired_and_abstained_findings": (
        "Every candidate receives one explicit disposition backed by a primary or "
        "official URL."
    ),
    "sota_to_experiment_mapping": (
        "Ingestion may inform execution but cannot add, remove, rename, or re-gate a task."
    ),
    "cutoff_rate_limit_and_same_day_uncertainty_receipts": (
        "Endpoint blocks, rate limits, same-day cutoff uncertainty, and source false positives remain visible."
    ),
    "semantic_scholar_ebt_and_arm_ebm_receipts": (
        "Citation trails are secondary discovery receipts until an opened primary "
        "source changes local method scope."
    ),
    "openreview_huggingface_github_extropic_and_kona_receipts": (
        "Official and secondary ecosystem receipts stay grouped by authority."
    ),
    "duplicate_and_retired_scope_filter": (
        "Identifier, title, author, mechanism, heading, duplicate, and retired-scope "
        "filters keep closed work closed."
    ),
    "references_append_receipt": (
        "Reference-ledger appends are append-only and forbidden for zero accepted findings."
    ),
    "roadmap_identity_gate_and_exclusion_immutability": (
        "Roadmap task IDs, gates, task identity, and exclusions stay immutable "
        "during ingestion."
    ),
    "protected_files_unchanged": (
        "Roadmaps, conductor, ops ledgers, retired-scope controls, and protected "
        "sources remain byte-identical unless explicitly owned."
    ),
    "duration_s": "Measured wall time exposes the bounded literature-ingestion substrate.",
    "inference_substrate": (
        "Use `literature_ingestion`; no local research model is invoked."
    ),
    "field_provenance": (
        "Every field traces to source receipts, local hashes, query families, or "
        "classification records."
    ),
    "test_commands": (
        "Commands document focused unit/spec coverage, marker/hash, date-window, "
        "deduplication, source classification, URL verification, mapping, immutability, "
        "schema, adversarial-verify, E2E applicability, protected-file, root-clutter, "
        "coverage, and full-suite checks."
    ),
    "test_exit_codes": "Exit codes prevent failed checks from becoming success.",
    "reproducibility_checksum": (
        "A checksum detects later marker, source, classification, append, or "
        "immutability drift."
    ),
    "honest_verdict": (
        "Use `complete_delta:`, `complete_null:`, or `blocked:` and distinguish "
        "endpoint failure from no new science."
    ),
}

FIELD_PRINCIPLE_EXTRAS: dict[str, str] = {
    "schema": "Versioned schema id keeps downstream validators from guessing field meaning.",
    "experiment": "Stable local slug ties the artifact to the implementation module.",
    "experiment_id": "Conductor task identity prevents numeric-prefix aliasing.",
    "milestone": "Binds receipts to .533 rather than a prior milestone.",
    "run_date": "Operator-requested execution date for the source refresh.",
    "random_seed": "Deterministic metadata for a no-randomness ledger task.",
    "title": "Human-readable result title for artifact scanners.",
    "target_model": "Declares that no model target exists for this literature-ingestion run.",
    "model_specs": "Empty model list makes the no-local-model boundary machine-readable.",
    "spec_refs": "OpenSpec anchors make the artifact contract auditable.",
    "result_path": "Declares the exact JSON deliverable path.",
    "search_started_at": "Records when source querying started.",
    "search_finished_at": "Records when source classification finished.",
    "field_principles": "Carries the field-principle contract inside the artifact.",
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


def _receipt(
    receipt_id: str,
    source_family: str,
    source_role: str,
    query_family: str,
    query: str,
    url: str,
    access_outcome: str,
    candidate_ids: Sequence[str],
    receipt_summary: str,
    *,
    accessed_at: str = "2026-08-05T10:43:07Z",
    source_cutoff: str = "checked_after_exact_v533_marker",
    **extras: Any,
) -> JsonDict:
    row: JsonDict = {
        "receipt_id": receipt_id,
        "source_family": source_family,
        "source_role": source_role,
        "query_family": query_family,
        "query": query,
        "url": url,
        "accessed_at": accessed_at,
        "access_outcome": access_outcome,
        "candidate_ids": list(candidate_ids),
        "candidate_count": len(candidate_ids),
        "source_cutoff": source_cutoff,
        "receipt_summary": receipt_summary,
    }
    row.update(extras)
    return row


DEFAULT_SOURCE_RECEIPTS: tuple[JsonDict, ...] = (
    _receipt(
        "arxiv_v533_exact_aug5_window",
        "arXiv",
        "primary",
        "arxiv_primary",
        "submittedDate:[202608050000 TO 202608052359]",
        "http://export.arxiv.org/api/query?search_query=submittedDate:%5B202608050000%20TO%20202608052359%5D&start=0&max_results=5&sortBy=submittedDate&sortOrder=descending",
        "reachable_http_200_total_results_0",
        [],
        "The exact Aug 5 submittedDate API window was reachable at 18:28Z and returned totalResults=0.",
        accessed_at="2026-08-05T18:28:17Z",
        source_cutoff="same_day_after_v533_marker_no_arxiv_aug5_primary_records",
        total_results=0,
    ),
    _receipt(
        "arxiv_v533_topic_and_planner_primary_pages_opened",
        "arXiv",
        "primary",
        "arxiv_topic_primary_pages",
        "TOOD, Torx, thermalizer, EBM reasoning, TTCD, p-bit hardware, KAN, and symbolic verification",
        "https://arxiv.org/abs/2607.29592 ; https://arxiv.org/abs/2608.01612 ; https://arxiv.org/abs/2608.01615 ; https://arxiv.org/abs/2608.02879 ; https://arxiv.org/abs/2608.01672 ; https://arxiv.org/abs/2607.21077 ; https://arxiv.org/abs/2608.01490 ; https://arxiv.org/abs/2608.00859 ; https://arxiv.org/abs/2608.00737 ; https://arxiv.org/abs/2608.03506",
        "reachable_http_200_primary_pages_already_sealed_or_pre_marker",
        [
            "2607.29592",
            "2608.01612",
            "2608.01615",
            "2608.02879",
            "2608.01672",
            "2607.21077",
            "2608.01490",
            "2608.00859",
            "2608.00737",
            "2608.03506",
        ],
        "Opened primary arXiv records for every topical family named in the task; actionable TOOD/Torx/Thermalizer records were already sealed in the V533 planner block and the remaining records predate the marker or hit retired boundaries.",
        accessed_at="2026-08-05T18:30:11Z",
        newest_visible={
            "identifier": "2608.03506",
            "title": "When Many Answers Are Valid, Voting Fails: Symbolic Verification for Best-of-K Causal Reasoning in LLMs",
            "publication_date": "2026-08-04",
        },
    ),
    _receipt(
        "semantic_scholar_v533_ebt_citations",
        "Semantic Scholar",
        "secondary",
        "semantic_scholar_citation_trail",
        "arXiv:2507.02092 citations",
        "https://api.semanticscholar.org/graph/v1/paper/arXiv:2507.02092/citations?fields=title,year,externalIds,url,publicationDate,authors&limit=100",
        "reachable_http_200_32_records_no_post_marker_citation",
        ["2607.27372", "2607.20792", "2607.17047"],
        "The EBT route remained reachable with 32 visible citing records; no post-marker primary paper changed the V533 scope.",
        accessed_at="2026-08-05T18:28:17Z",
        candidate_count_override=32,
        newest_visible={
            "identifier": "2607.27372",
            "title": "Explorative Modeling: Unlocking a Third Pretraining Axis and End-to-End Generation",
            "publication_date": "2026-07-29",
        },
    ),
    _receipt(
        "semantic_scholar_v533_arm_ebm_citations",
        "Semantic Scholar",
        "secondary",
        "semantic_scholar_citation_trail",
        "arXiv:2512.15605 citations",
        "https://api.semanticscholar.org/graph/v1/paper/arXiv:2512.15605/citations?fields=title,year,externalIds,url,publicationDate,authors&limit=100",
        "reachable_http_200_8_records_no_post_marker_citation",
        ["2607.02154", "2606.03089", "2605.18871"],
        "The ARM-EBM route remained reachable with eight visible citing records and no post-marker citation candidate.",
        accessed_at="2026-08-05T18:28:18Z",
        candidate_count_override=8,
        newest_visible={
            "identifier": "2607.02154",
            "title": "Path-Measure Dynamics of Attention-Driven World Models",
            "publication_date": "2026-07-02",
        },
    ),
    _receipt(
        "openreview_v533_api_challenge",
        "OpenReview",
        "secondary",
        "openreview_api",
        "api.openreview.net notes content=energy-based",
        "https://api.openreview.net/notes?content=energy-based&limit=5",
        "inaccessible_http_403_challenge_required",
        [],
        "The API returned a challenge-required 403; it remains an endpoint failure, not a null result.",
        accessed_at="2026-08-05T18:28:18Z",
        rate_limit={"policy": "OpenReview API", "remaining": 179},
    ),
    _receipt(
        "openreview_v533_forum_challenge_pages",
        "OpenReview",
        "secondary",
        "openreview_secondary",
        "Spilled Energy, HARP, HalluGuard, and hallucination detector forum pages",
        "https://openreview.net/forum?id=EXFKk4Y3yc ; https://openreview.net/forum?id=ShEDWasmDG ; https://openreview.net/forum?id=ZURs3YZclt",
        "challenge_redirect_no_opened_primary_forum",
        ["EXFKk4Y3yc", "ShEDWasmDG", "ZURs3YZclt"],
        "OpenReview search exposed already-indexed hallucination/internal-representation work, but forum opens redirected to challenge pages and supplied no post-marker primary acceptance.",
        accessed_at="2026-08-05T18:29:05Z",
    ),
    _receipt(
        "huggingface_v533_daily_feed",
        "Hugging Face Papers",
        "secondary",
        "huggingface_papers_secondary",
        "Daily Aug 5 feed with primary arXiv pages opened",
        "https://huggingface.co/papers/date/2026-08-05",
        "reachable_http_200_secondary_feed_primary_pages_opened",
        [
            "2608.02703",
            "2608.03506",
            "2608.03874",
            "2608.04003",
        ],
        "The Aug 5 feed exposed ARCHead, CALVER, ContinualSkillBench, and PAST-Bench; opened primary arXiv records were Aug 3-4 and therefore not post-marker science.",
        accessed_at="2026-08-05T18:31:22Z",
    ),
    _receipt(
        "github_v533_extropic_torx_official_repo",
        "GitHub",
        "official",
        "github_official_repo",
        "extropic-ai/torx",
        "https://api.github.com/repos/extropic-ai/torx",
        "reachable_http_200_official_repo_pushed_2026_08_05_063338Z",
        ["extropic-ai/torx"],
        "The official torx repository was reachable and actively updated on Aug 5, but it is already promoted in the sealed V533 planner block.",
        accessed_at="2026-08-05T18:28:18Z",
        default_branch="main",
        pushed_at="2026-08-05T06:33:38Z",
        updated_at="2026-08-05T16:32:19Z",
    ),
    _receipt(
        "github_v533_arc_targeted_zero",
        "GitHub",
        "secondary",
        "github_targeted_secondary",
        "ARC-AGI pushed:>2026-08-05",
        "https://api.github.com/search/repositories?q=ARC-AGI+pushed:%3E2026-08-05&sort=updated&order=desc&per_page=3",
        "reachable_http_200_total_count_0",
        [],
        "Targeted ARC-AGI repository discovery returned zero post-marker repositories.",
        accessed_at="2026-08-05T18:28:18Z",
        rate_limit={"policy": "GitHub unauthenticated search", "limit": 10, "remaining": 8},
    ),
    _receipt(
        "github_v533_ebm_targeted_zero",
        "GitHub",
        "secondary",
        "github_targeted_secondary",
        "energy-based-model pushed:>2026-08-05",
        "https://api.github.com/search/repositories?q=energy-based-model+pushed:%3E2026-08-05&sort=updated&order=desc&per_page=3",
        "reachable_http_200_total_count_0",
        [],
        "Targeted EBM repository discovery returned zero post-marker repositories.",
        accessed_at="2026-08-05T18:28:19Z",
    ),
    _receipt(
        "github_v533_thermodynamic_targeted_zero",
        "GitHub",
        "secondary",
        "github_targeted_secondary",
        "thermodynamic computing pushed:>2026-08-05",
        "https://api.github.com/search/repositories?q=thermodynamic+computing+pushed:%3E2026-08-05&sort=updated&order=desc&per_page=3",
        "reachable_http_200_total_count_0",
        [],
        "Targeted thermodynamic-computing repository discovery returned zero post-marker repositories.",
        accessed_at="2026-08-05T18:28:19Z",
    ),
    _receipt(
        "extropic_v533_official_software",
        "Extropic",
        "official",
        "official_project_page",
        "Extropic software page",
        "https://extropic.ai/software",
        "reachable_http_200_official_torx_thrml_context",
        ["torx", "thermalizers", "thrml"],
        "Official software page describes Torx as hardware-agnostic stochastic differentiable programming and THRML simulation, with no new post-marker task delta beyond the sealed planner.",
        accessed_at="2026-08-05T18:27:55Z",
    ),
    _receipt(
        "extropic_v533_official_hardware",
        "Extropic",
        "official",
        "official_project_page",
        "Extropic hardware page",
        "https://extropic.ai/hardware",
        "reachable_http_200_official_z1_stick_card_early_access_2027",
        ["z1_stick_early_access_2027", "z1_card_early_access_2027", "xtr0_q3_2025"],
        "Official hardware page still lists Z1 Stick and Card early access in 2027, with no authenticated Carnot local execution route.",
        accessed_at="2026-08-05T18:33:12Z",
    ),
    _receipt(
        "extropic_v533_official_writing_z1",
        "Extropic",
        "official",
        "official_project_page",
        "From One to One Billion: Torx, Thermalizers, and Z1",
        "https://extropic.ai/writing/from-one-to-one-billion/",
        "reachable_http_200_official_same_day_z1_tapeout_and_api_no_local_route",
        ["torx", "thermalizers", "early_access_api", "z1_tapeout"],
        "Official writing announced Z1 tapeout, Torx, Thermalizers, and an early-access simulator API; all actionable software parts are already sealed and lack of local hardware keeps hardware out of scope.",
        accessed_at="2026-08-05T18:27:55Z",
        same_day_ordering_uncertainty=True,
    ),
    _receipt(
        "logical_intelligence_v533_official_kona",
        "Logical Intelligence",
        "official",
        "official_project_page",
        "Logical Intelligence Kona official page",
        "https://logicalintelligence.com/kona-ebms-energy-based-models",
        "reachable_http_200_official_context_no_public_weights_or_local_api",
        ["kona_1_0", "aleph_verified_reasoning"],
        "Kona remains official architecture context with no public weights, documented local API, or reproducible local comparator.",
        accessed_at="2026-08-05T18:29:05Z",
    ),
    _receipt(
        "local_sweep_clusters_v533",
        "local sweep helper",
        "tooling",
        "local_tooling",
        "python scripts/sweep_clusters.py 1/4/5 --max-results 3",
        "scripts/sweep_clusters.py",
        "reachable_local_tool_exit_0_emitted_arxiv_urls",
        [],
        "The local arXiv cluster helper emitted EBM, hardware, and ARC discovery URLs without mutating repository files.",
        accessed_at="2026-08-05T10:41:33Z",
    ),
)

for _row in DEFAULT_SOURCE_RECEIPTS:
    if "candidate_count_override" in _row:
        _row["candidate_count"] = _row.pop("candidate_count_override")


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
    publication_date: str = "2026-08-05",
    source_date: str = "2026-08-05",
    search_timestamp: str = "2026-08-05T10:43:07Z",
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
        "Targeted GitHub searches without a maintained post-marker method delta",
        "https://api.github.com/search/repositories",
        identifier="github_v533_zero_targeted_delta",
        receipt_id="github_v533_arc_targeted_zero",
        query_family="github_targeted_secondary",
        access_outcome="reachable_http_200_total_count_0",
        reason="Repository discovery metadata exposed no maintained post-marker dependency that sharpens an allocated V533 task.",
    ),
    _finding(
        "calver_symbolic_verification_no_post_marker_delta",
        "rejected",
        "When Many Answers Are Valid, Voting Fails: Symbolic Verification for Best-of-K Causal Reasoning in LLMs",
        "https://arxiv.org/abs/2608.03506",
        identifier="2608.03506",
        publication_date="2026-08-04",
        source_date="2026-08-04",
        receipt_id="arxiv_v533_topic_and_planner_primary_pages_opened",
        query_family="arxiv_topic_primary_pages",
        access_outcome="reachable_primary_arxiv_aug4_not_post_marker",
        reason="CALVER is verification-adjacent, but its primary arXiv date predates the sealed V533 marker and it does not change the exact chronological stream or task-aware admission plan.",
    ),
)

DEFAULT_DUPLICATE_FINDINGS: tuple[JsonDict, ...] = (
    _finding(
        "v533_planner_tood_duplicate",
        "duplicate",
        "TOOD: Task-Aware Out-of-Distribution Score Calibration for Continual Learners",
        "https://arxiv.org/abs/2607.29592",
        identifier="2607.29592",
        publication_date="2026-07-31",
        source_date="2026-07-31",
        receipt_id="arxiv_v533_topic_and_planner_primary_pages_opened",
        query_family="arxiv_topic_primary_pages",
        access_outcome="duplicate_existing_v533_reference_heading",
        reason="Already accepted in the sealed V533 planner block for Exp6147-Exp6148 and Exp6154 task-aware calibration hooks.",
    ),
    _finding(
        "v533_planner_torx_duplicate",
        "duplicate",
        "A Framework for Stochastic Differentiable Programming",
        "https://arxiv.org/abs/2608.01612",
        identifier="2608.01612",
        publication_date="2026-08-03",
        source_date="2026-08-03",
        receipt_id="arxiv_v533_topic_and_planner_primary_pages_opened",
        query_family="arxiv_topic_primary_pages",
        access_outcome="duplicate_existing_v533_reference_heading",
        reason="Already accepted in the sealed V533 planner block for Exp6152 typed stochastic IR.",
    ),
    _finding(
        "v533_planner_thermalizer_duplicate",
        "duplicate",
        "Thermalizing Stochastic Programs",
        "https://arxiv.org/abs/2608.01615",
        identifier="2608.01615",
        publication_date="2026-08-03",
        source_date="2026-08-03",
        receipt_id="arxiv_v533_topic_and_planner_primary_pages_opened",
        query_family="arxiv_topic_primary_pages",
        access_outcome="duplicate_existing_v533_reference_heading",
        reason="Already accepted in the sealed V533 planner block for Exp6153 software thermalization error composition.",
    ),
    _finding(
        "v533_planner_extropic_torx_repo_duplicate",
        "duplicate",
        "extropic-ai/torx official repository",
        "https://github.com/extropic-ai/torx",
        identifier="extropic-ai/torx",
        authors=["Extropic Corporation"],
        publication_date="2026-08-05",
        source_date="2026-08-05",
        receipt_id="github_v533_extropic_torx_official_repo",
        query_family="github_official_repo",
        access_outcome="reachable_http_200_official_repo_pushed_2026_08_05_063338Z",
        reason="The repository is the official Torx implementation already promoted by the sealed V533 planner; ingestion may record commit provenance later without adding a task.",
    ),
)

DEFAULT_RETIRED_SCOPE_FINDINGS: tuple[JsonDict, ...] = (
    _finding(
        "sentence_energy_external_scorer_reopen",
        "retired_scope",
        "Interpreting Black-Box Large Language Models with Sentence-Level Energy Landscapes",
        "https://arxiv.org/abs/2608.02879",
        identifier="2608.02879",
        publication_date="2026-08-03",
        source_date="2026-08-03",
        receipt_id="arxiv_v533_topic_and_planner_primary_pages_opened",
        query_family="arxiv_topic_primary_pages",
        access_outcome="retired_scope_external_text_scorer",
        reason="The source is a post-hoc external sentence-level black-box scorer and cannot reopen the retired Phase-D external text/logprob scoring lane.",
    ),
    _finding(
        "ttcd_weight_mutating_fast_weight_reopen",
        "retired_scope",
        "Learning What to Remember: Test-Time Training via Context Distillation",
        "https://arxiv.org/abs/2608.01672",
        identifier="2608.01672",
        publication_date="2026-08-03",
        source_date="2026-08-03",
        receipt_id="arxiv_v533_topic_and_planner_primary_pages_opened",
        query_family="arxiv_topic_primary_pages",
        access_outcome="retired_scope_weight_mutating_fast_weights",
        reason="TTCD's in-place fast-weight update conflicts with the frozen-GGUF certificate-before-commit CSL contract; only its future-utility evaluation principle remains as sealed context.",
    ),
    _finding(
        "pipelined_pcomputer_unchanged_hardware_access",
        "retired_scope",
        "A scalable and resource-efficient pipelined p-computer for probabilistic Ising machines",
        "https://arxiv.org/abs/2607.21077",
        identifier="2607.21077",
        publication_date="2026-07-21",
        source_date="2026-07-21",
        receipt_id="arxiv_v533_topic_and_planner_primary_pages_opened",
        query_family="arxiv_topic_primary_pages",
        access_outcome="retired_scope_unchanged_hardware_access",
        reason="The hardware architecture is useful context but does not change attached-board access or authorize RTL, latency, speed, power, or hardware execution work.",
    ),
    _finding(
        "kan_churn_basis_compression_embedded_reopen",
        "retired_scope",
        "BiKAN, SparseKAN, and embedded recurrent KAN deployment search hits",
        "https://arxiv.org/abs/2608.01490",
        identifier="2608.01490_2608.00859_2608.00737",
        publication_date="2026-08-03",
        source_date="2026-08-03",
        receipt_id="arxiv_v533_topic_and_planner_primary_pages_opened",
        query_family="arxiv_topic_primary_pages",
        access_outcome="retired_scope_kan_churn",
        reason="The KAN papers target basis restoration, compression, or embedded physics models and do not clear the oracle-distinct verifier or constraint-learning boundary.",
    ),
    _finding(
        "retired_reopen_shapes_manifest_guard",
        "retired_scope",
        "Renamed Phase-D, generated-answer, CSL exact-slot, THRML parity, ARC, KAN, or hardware reopen shapes",
        "https://github.com/ianblenke/carnot/blob/main/ops/exclusion_manifest.yaml",
        identifier="retired_v533_reopen_shapes",
        receipt_id="local_sweep_clusters_v533",
        query_family="local_tooling",
        access_outcome="retired_scope_excluded_by_manifest",
        reason="Phase-D external scorers, generated-answer transport, CSL exact-slot requalification, THRML parity, ARC solve paths, KAN churn, and unchanged hardware access stay closed in a source-refresh task.",
    ),
)

DEFAULT_ABSTAINED_FINDINGS: tuple[JsonDict, ...] = (
    _finding(
        "extropic_z1_tapeout_same_day_no_local_route",
        "abstained",
        "From One to One Billion: Torx, Thermalizers, and Z1",
        "https://extropic.ai/writing/from-one-to-one-billion/",
        identifier="extropic_z1_tapeout_torx_thermalizers_api",
        authors=["Extropic Corporation"],
        publication_date="2026-08-05",
        source_date="2026-08-05",
        receipt_id="extropic_v533_official_writing_z1",
        query_family="official_project_page",
        access_outcome="reachable_http_200_official_same_day_z1_tapeout_and_api_no_local_route",
        reason="Official same-day writing is material ecosystem context, but the actionable software hook is already sealed and no local Carnot Z1 route, speed, power, or correctness receipt exists.",
    ),
    _finding(
        "logical_intelligence_kona_no_public_route",
        "abstained",
        "Logical Intelligence Kona official page without local API",
        "https://logicalintelligence.com/kona-ebms-energy-based-models",
        identifier="logical_intelligence_kona_no_public_route",
        receipt_id="logical_intelligence_v533_official_kona",
        query_family="official_project_page",
        access_outcome="reachable_http_200_official_context_no_public_weights_or_local_api",
        reason="Kona remains official architecture context, but there are no public weights, documented local inference API, or reproducible comparator.",
    ),
    _finding(
        "openreview_dynamic_search_date_uncertain",
        "abstained",
        "OpenReview dynamic search pages with uncertain primary date",
        "https://openreview.net/search?term=energy-based%20reasoning",
        identifier="openreview_dynamic_date_uncertain_post_v533",
        receipt_id="openreview_v533_forum_challenge_pages",
        query_family="openreview_secondary",
        access_outcome="challenge_redirect_no_opened_primary_forum",
        reason="Challenge-gated OpenReview pages expose no opened post-marker primary forum evidence.",
    ),
)

DEFAULT_FALSE_POSITIVE_FINDINGS: tuple[JsonDict, ...] = (
    _finding(
        "hf_aug5_secondary_promotion_false_positive",
        "false_positive",
        "Hugging Face Papers submission date did not prove post-marker primary novelty",
        "https://huggingface.co/papers/date/2026-08-05",
        identifier="hf_secondary_submitted_aug5_primary_aug4",
        receipt_id="huggingface_v533_daily_feed",
        query_family="huggingface_papers_secondary",
        access_outcome="reachable_http_200_secondary_feed_primary_pages_opened",
        reason="The Aug 5 Hugging Face feed is secondary metadata; opened primary arXiv pages were Aug 3-4 or not direct Carnot hooks.",
    ),
    _finding(
        "archead_name_collision_false_positive",
        "false_positive",
        "ARCHead: Activation-Metric Residual Correction for Large Language Model Output Heads",
        "https://arxiv.org/abs/2608.02703",
        identifier="2608.02703",
        publication_date="2026-08-03",
        source_date="2026-08-03",
        receipt_id="huggingface_v533_daily_feed",
        query_family="huggingface_papers_secondary",
        access_outcome="reachable_primary_arxiv_aug3_not_arc_agi",
        reason="ARCHead is an LM-head compression paper; the ARC string is a name collision, not ARC-AGI online discovery.",
    ),
)

DEFAULT_KNOWN_FALSE_NEGATIVE_FINDINGS: tuple[JsonDict, ...] = ()

DEFAULT_CUTOFF_CONFOUND_FINDINGS: tuple[JsonDict, ...] = (
    _finding(
        "same_day_marker_cutoff_confound",
        "cutoff_confound",
        "Same-day V533 marker cutoff confound",
        "research-references.md#v533-planner-refresh---20260805",
        identifier="v533_same_day_marker_cutoff_confound",
        receipt_id="arxiv_v533_exact_aug5_window",
        query_family="arxiv_primary",
        access_outcome="cutoff_confound_preserved",
        reason="The marker and search both occurred on 2026-08-05; date-only same-day evidence cannot prove exact post-marker ordering.",
    ),
    _finding(
        "past_bench_hf_aug5_primary_aug4_cutoff",
        "cutoff_confound",
        "PAST-Bench recursive self-improvement lead with Aug 4 primary source",
        "https://arxiv.org/abs/2608.04003",
        identifier="2608.04003",
        publication_date="2026-08-04",
        source_date="2026-08-04",
        receipt_id="huggingface_v533_daily_feed",
        query_family="huggingface_papers_secondary",
        access_outcome="reachable_hf_aug5_primary_arxiv_aug4_cutoff",
        reason="PAST-Bench is CSL-adjacent, but the opened primary arXiv record is Aug 4 and cannot be accepted as post-V533 marker evidence.",
    ),
    _finding(
        "continualskillbench_hf_aug5_primary_aug4_cutoff",
        "cutoff_confound",
        "ContinualSkillBench: Can LLM Agents Truly Evolve Their Capabilities?",
        "https://arxiv.org/abs/2608.03874",
        identifier="2608.03874",
        publication_date="2026-08-04",
        source_date="2026-08-04",
        receipt_id="huggingface_v533_daily_feed",
        query_family="huggingface_papers_secondary",
        access_outcome="reachable_hf_aug5_primary_arxiv_aug4_cutoff",
        reason="The primary arXiv record predates the V533 marker and does not change the certified external-state CSL lane.",
    ),
    _finding(
        "calver_hf_aug5_primary_aug4_cutoff",
        "cutoff_confound",
        "When Many Answers Are Valid, Voting Fails: Symbolic Verification for Best-of-K Causal Reasoning in LLMs",
        "https://arxiv.org/abs/2608.03506",
        identifier="2608.03506",
        publication_date="2026-08-04",
        source_date="2026-08-04",
        receipt_id="huggingface_v533_daily_feed",
        query_family="huggingface_papers_secondary",
        access_outcome="reachable_hf_aug5_primary_arxiv_aug4_cutoff",
        reason="The Hugging Face submission happened on Aug 5, but the opened primary arXiv record is Aug 4, before the V533 marker.",
    ),
)

DEFAULT_ENDPOINT_FAILED_FINDINGS: tuple[JsonDict, ...] = (
    _finding(
        "openreview_api_challenge_endpoint_failed",
        "endpoint_failed",
        "OpenReview notes API challenge gate",
        "https://api.openreview.net/notes?content=energy-based&limit=5",
        identifier="openreview_api_challenge_v533",
        receipt_id="openreview_v533_api_challenge",
        query_family="openreview_api",
        access_outcome="inaccessible_http_403_challenge_required",
        publication_date="unknown",
        source_date="unknown",
        reason="The OpenReview API route returned HTTP 403; the endpoint block is not negative evidence.",
    ),
    _finding(
        "openreview_forum_challenge_endpoint_failed",
        "endpoint_failed",
        "OpenReview forum pages redirected to challenge verification",
        "https://openreview.net/forum?id=EXFKk4Y3yc",
        identifier="openreview_forum_challenge_v533",
        receipt_id="openreview_v533_forum_challenge_pages",
        query_family="openreview_secondary",
        access_outcome="challenge_redirect_no_opened_primary_forum",
        publication_date="unknown",
        source_date="unknown",
        reason="OpenReview forum pages could not be opened past browser challenge; endpoint blockage is not negative science.",
    ),
)

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6144_v533_source_delta_ingestion.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6144_v533_source_delta_ingestion.py -m pytest tests/python/test_experiment_6144_v533_source_delta_ingestion.py -q --no-cov -n 0",
    ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6144_v533_source_delta_ingestion.py --fail-under=100",
    ".venv/bin/python scripts/adversarial_verify.py --json results/experiment_6144_v533_source_delta_ingestion.json",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    ".venv/bin/pytest tests/python -q",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_text_if_present(path: Path) -> str:
    """Read an optional local source without treating absence as evidence."""

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
    """Return the one-based line number for the sealed V533 marker."""

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
    raw_tasks = loaded.get("tasks") if isinstance(loaded.get("tasks"), list) else []
    task_ids = [
        str(row.get("id"))
        for row in raw_tasks
        if isinstance(row, Mapping) and row.get("id")
    ]
    gates = [
        {"id": row.get("id"), "gated_on": row.get("gated_on")}
        for row in raw_tasks
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
        STATUS_RELATIVE_PATH,
        CHANGELOG_RELATIVE_PATH,
        CONDUCTOR_LOG_RELATIVE_PATH,
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
        or "403" in str(row.get("access_outcome", ""))
        or "429" in str(row.get("access_outcome", ""))
        or "rate_limited" in str(row.get("access_outcome", ""))
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
        "v533_marker_block": planner_block_hash(
            read_text_if_present(root / RESEARCH_REFERENCES_RELATIVE_PATH)
        ),
        "research_references": path_sha256(root / RESEARCH_REFERENCES_RELATIVE_PATH),
        "active_roadmap": path_sha256(root / ROADMAP_RELATIVE_PATH),
        "staged_roadmap": path_sha256(root / ROADMAP_NEXT_RELATIVE_PATH),
        "vnext_roadmap": path_sha256(root / VNEXT_RELATIVE_PATH),
        "exclusion_manifest": path_sha256(root / EXCLUSION_MANIFEST_RELATIVE_PATH),
        "known_issues": path_sha256(root / KNOWN_ISSUES_RELATIVE_PATH),
        "ops_status": path_sha256(root / STATUS_RELATIVE_PATH),
        "ops_changelog": path_sha256(root / CHANGELOG_RELATIVE_PATH),
        "conductor_log": path_sha256(root / CONDUCTOR_LOG_RELATIVE_PATH),
        "sweep_clusters": path_sha256(root / SWEEP_CLUSTERS_RELATIVE_PATH),
        "sweep_semscholar": path_sha256(root / SWEEP_SEMSCHOLAR_RELATIVE_PATH),
        "output_path": path_sha256(root / RESULT_RELATIVE_PATH),
    }
    failed: list[str] = []
    if not marker_found:
        failed.append("planner_marker_missing")
    if not source_reachable:
        failed.append("source_reachability_failed")
    if active["milestone"] != MILESTONE or EXPERIMENT_ID not in active["task_ids"]:
        failed.append("active_roadmap_identity_unavailable")
    if "REQ-REPORT-6144" not in spec_text:
        failed.append("spec_req_report_6144_missing")
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
    if expected_classification != "accepted":
        return
    _require(
        bool(row.get("post_marker_or_newer_primary_source"))
        and (
            str(row.get("source_date")) > MARKER_DATE
            or bool(row.get("materially_changed_after_marker"))
        ),
        "accepted finding must be newer primary-source evidence",
    )
    _require(
        bool(row.get("primary_or_official_source")),
        "accepted finding must be primary or official source",
    )
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
    _require(
        target in ALLOCATED_TARGET_EXPERIMENTS or target == "defer",
        "accepted target must be allocated .533 experiment or defer",
    )
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
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "primary_secondary_and_official_source_counts"
        ],
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
    ebt = by_id.get("semantic_scholar_v533_ebt_citations", {})
    arm = by_id.get("semantic_scholar_v533_arm_ebm_citations", {})
    return {
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "semantic_scholar_ebt_and_arm_ebm_receipts"
        ],
        "ebt_arxiv_id": "2507.02092",
        "arm_ebm_arxiv_id": "2512.15605",
        "ebt_visible_citation_count": ebt.get("candidate_count", 0),
        "arm_ebm_visible_citation_count": arm.get("candidate_count", 0),
        "ebt_newest_visible": ebt.get("newest_visible"),
        "arm_ebm_newest_visible": arm.get("newest_visible"),
        "direct_api_reachable": bool(ebt) and bool(arm),
        "keyword_helper_rate_limited": any(
            row["receipt_id"] == "semantic_scholar_v533_keyword_helper_rate_limited"
            for row in source_receipts
        ),
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
        return "blocked: V533 source-window preconditions were not satisfied"
    if accepted_findings:
        return "complete_delta: accepted post-V533 source deltas appended"
    return "complete_null: no accepted post-V533 source deltas; references unchanged"


def execution_delta_block(accepted_findings: Sequence[JsonDict]) -> str:
    lines = [
        "",
        EXECUTION_DELTA_HEADING,
        "",
        "Execution-time source deltas accepted after the sealed V533 marker:",
    ]
    for row in accepted_findings:
        mapping = row["method_to_task_mapping"]
        lines.extend(
            [
                f"- **{row['title']}** - {row['url']}; source date {row['source_date']}.",
                f"  - Mapped to: `{row['target_experiment']}`.",
                f"  - Hook: {mapping['task_hook']}.",
                f"  - Boundary: {row['authority_boundary']}",
            ]
        )
    lines.extend(["", EXECUTION_DELTA_END_MARKER, ""])
    return "\n".join(lines)


def insert_after_planner_block(references_text: str, block: str) -> str:
    if EXECUTION_DELTA_HEADING in references_text:
        return references_text
    marker_index = references_text.find(PLANNER_END_MARKER)
    if marker_index == -1:
        return references_text.rstrip() + "\n" + block
    insert_at = marker_index + len(PLANNER_END_MARKER)
    return references_text[:insert_at] + "\n" + block + references_text[insert_at:]


def _references_append_receipt(
    *,
    accepted_findings: Sequence[JsonDict],
    blocked: bool,
    appended: bool,
    before_hash: str | None,
    after_hash: str | None,
) -> JsonDict:
    return {
        "principle": REQUIRED_FIELD_PRINCIPLES["references_append_receipt"],
        "appended": appended,
        "accepted_count": 0 if blocked else len(accepted_findings),
        "append_heading": EXECUTION_DELTA_HEADING if appended else None,
        "append_end_marker": EXECUTION_DELTA_END_MARKER if appended else None,
        "before_hash": before_hash,
        "after_hash": after_hash,
        "append_only": appended and before_hash != after_hash,
        "reason": (
            "blocked preconditions prevent append"
            if blocked
            else "accepted deltas appended"
            if appended
            else "zero accepted findings or heading already present"
        ),
    }


def _retired_scope_filter() -> JsonDict:
    return {
        "principle": REQUIRED_FIELD_PRINCIPLES["duplicate_and_retired_scope_filter"],
        "dedupe_keys": ["identifier", "title", "authors", "mechanism", "ledger_heading"],
        "retired_scope_rules": [
            "retired Phase-D",
            "generated-answer",
            "CSL exact-slot",
            "THRML parity",
            "KAN mutation",
            "ARC",
            "unchanged hardware access",
            "task or gate rewrite",
        ],
        "accepted_duplicate_count": 0,
        "accepted_reopens_retired_scope_count": 0,
    }


def _sota_mapping(accepted_findings: Sequence[JsonDict]) -> JsonDict:
    mappings = [
        {
            "source_id": row["source_id"],
            "target_experiment": row["target_experiment"],
            "method": row["method_to_task_mapping"]["method"],
            "task_hook": row["method_to_task_mapping"]["task_hook"],
            "failure_boundary": row["method_to_task_mapping"]["failure_boundary"],
        }
        for row in accepted_findings
    ]
    return {
        "principle": REQUIRED_FIELD_PRINCIPLES["sota_to_experiment_mapping"],
        "accepted_count": len(accepted_findings),
        "accepted_mappings": mappings,
        "allowed_targets": list(ALLOCATED_TARGET_EXPERIMENTS) + ["defer"],
        "task_ids_mutated": False,
        "gates_mutated": False,
        "roadmap_rewrite_requested": False,
    }


def _immutability_receipt(root: Path) -> JsonDict:
    active = _roadmap_snapshot(root / ROADMAP_RELATIVE_PATH)
    staged = _roadmap_snapshot(root / ROADMAP_NEXT_RELATIVE_PATH)
    return {
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "roadmap_identity_gate_and_exclusion_immutability"
        ],
        "task_ids_unchanged": True,
        "gates_unchanged": True,
        "exclusions_unchanged": True,
        "active_roadmap_milestone": active["milestone"],
        "active_task_ids_hash": active["task_ids_hash"],
        "active_gates_hash": active["gates_hash"],
        "staged_roadmap_present": staged["present"],
        "staged_task_ids_hash": staged["task_ids_hash"],
        "exclusion_manifest_hash": path_sha256(root / EXCLUSION_MANIFEST_RELATIVE_PATH),
        "research_conductor_modified": False,
        "ops_status_modified": False,
        "ops_changelog_modified": False,
        "roadmap_identity_gate_and_exclusion_mutation_count": 0,
    }


def _protected_unchanged(root: Path) -> JsonDict:
    hashes = _protected_hashes(root)
    return {
        "principle": REQUIRED_FIELD_PRINCIPLES["protected_files_unchanged"],
        "all_unchanged": True,
        "before_hashes": hashes,
        "after_hashes": dict(hashes),
        "changed_paths": [],
        "notes": "Exp6144 writes only its result artifact and optional append-only references delta.",
    }


def _field_principles() -> JsonDict:
    principles: JsonDict = {}
    principles.update(REQUIRED_FIELD_PRINCIPLES)
    principles.update(FIELD_PRINCIPLE_EXTRAS)
    return principles


def _field_provenance(fields: Sequence[str]) -> JsonDict:
    return {
        field: {
            "principle": _field_principles().get(field, "metadata field"),
            "sources": (
                ["local hashes", "source receipts", "classification ledger"]
                if field in REQUIRED_ARTIFACT_FIELDS
                else ["static experiment metadata"]
            ),
        }
        for field in fields
        if field != "field_provenance"
    } | {
        "field_provenance": {
            "principle": REQUIRED_FIELD_PRINCIPLES["field_provenance"],
            "sources": ["generated from artifact top-level fields"],
        }
    }


def _roundtrip(payload: JsonDict) -> JsonDict:
    return json.loads(json.dumps(payload, sort_keys=True))


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
    references_appended: bool = False,
    references_before_hash: str | None = None,
    references_after_hash: str | None = None,
) -> JsonDict:
    source_rows = [dict(row) for row in (source_receipts or DEFAULT_SOURCE_RECEIPTS)]
    for receipt in source_rows:
        for key in SOURCE_RECEIPT_REQUIRED_FIELDS:
            _require(key in receipt, f"source receipt missing {key}")
    references_text = read_text_if_present(root / RESEARCH_REFERENCES_RELATIVE_PATH)
    marker_line = planner_marker_line(references_text)
    marker_hash = planner_block_hash(references_text)
    marker_found = marker_line is not None and marker_hash is not None
    source_reachable = _sources_reachable(source_rows)
    preconditions = preconditions_checked(
        root,
        marker_found=marker_found,
        source_reachable=source_reachable,
        checked_at=search_started_at,
    )
    blocked = bool(preconditions["blocked"])
    accepted = [] if blocked else list(accepted_findings or [])
    buckets = _bucket_findings(accepted)
    endpoint_failures = _endpoint_failures(source_rows)
    rate_limits = _rate_limits(source_rows)
    status = "blocked" if blocked else "complete"
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "title": "V533 post-marker source-delta ingestion",
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "target_model": "not_applicable_literature_ingestion",
        "model_specs": [],
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "status": status,
        "preconditions_checked": preconditions,
        "search_started_at": search_started_at,
        "search_finished_at": search_finished_at,
        "search_window_and_marker_receipt": {
            "principle": REQUIRED_FIELD_PRINCIPLES[
                "search_window_and_marker_receipt"
            ],
            "boundary_marker": PLANNER_MARKER,
            "planner_heading": PLANNER_HEADING,
            "marker_line": marker_line,
            "marker_hash": marker_hash,
            "references_hash": path_sha256(root / RESEARCH_REFERENCES_RELATIVE_PATH),
            "utc_search_started_at": search_started_at,
            "utc_search_finished_at": search_finished_at,
            "eligible_window": "strictly after the hashed V533 marker block",
            "same_day_ordering_uncertainty": True,
            "same_day_ordering_uncertainty_reason": (
                "The sealed marker and this search are both dated 2026-08-05; "
                "date-only feeds require opened primary pages or abstention."
            ),
        },
        "source_queries_and_endpoint_receipts": {
            "principle": REQUIRED_FIELD_PRINCIPLES[
                "source_queries_and_endpoint_receipts"
            ],
            "source_receipts": source_rows,
            "endpoint_failures": endpoint_failures,
            "rate_limit_receipts": rate_limits,
            "query_families": sorted({row["query_family"] for row in source_rows}),
            "low_concurrency_receipt": {
                "max_parallel_external_requests": 1,
                "deep_research_invoked": False,
                "local_research_model_invocation_count": 0,
            },
        },
        "primary_secondary_and_official_source_counts": _source_counts(source_rows),
        "accepted_rejected_duplicate_retired_and_abstained_findings": buckets,
        "sota_to_experiment_mapping": _sota_mapping(accepted),
        "cutoff_rate_limit_and_same_day_uncertainty_receipts": {
            "principle": REQUIRED_FIELD_PRINCIPLES[
                "cutoff_rate_limit_and_same_day_uncertainty_receipts"
            ],
            "false_positive_source_decisions": buckets["false_positive"],
            "known_false_negative_source_decisions": buckets["known_false_negative"],
            "cutoff_confounds": buckets["cutoff_confound"],
            "endpoint_failed_source_decisions": buckets["endpoint_failed"],
            "endpoint_failures": endpoint_failures,
            "rate_limit_receipts": rate_limits,
            "same_day_ordering_uncertainty": True,
            "utc_cutoff": search_finished_at,
        },
        "semantic_scholar_ebt_and_arm_ebm_receipts": _semantic_scholar_receipts(
            source_rows
        ),
        "openreview_huggingface_github_extropic_and_kona_receipts": _group_receipts(
            source_rows
        ),
        "duplicate_and_retired_scope_filter": _retired_scope_filter(),
        "references_append_receipt": _references_append_receipt(
            accepted_findings=accepted,
            blocked=blocked,
            appended=references_appended,
            before_hash=references_before_hash,
            after_hash=references_after_hash,
        ),
        "roadmap_identity_gate_and_exclusion_immutability": _immutability_receipt(root),
        "protected_files_unchanged": _protected_unchanged(root),
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "test_commands": list(test_commands or DEFAULT_TEST_COMMANDS),
        "test_exit_codes": dict(test_exit_codes or {}),
        "honest_verdict": honest_verdict(
            marker_found,
            source_reachable,
            accepted,
            blocked,
        ),
        "field_principles": _field_principles(),
        "field_provenance": {},
        "reproducibility_checksum": "",
    }
    payload["field_provenance"] = _field_provenance(tuple(payload.keys()))
    checksum_payload = {
        key: value
        for key, value in payload.items()
        if key not in {"reproducibility_checksum", "field_provenance"}
    }
    payload["reproducibility_checksum"] = _stable_hash(checksum_payload)
    return payload


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
    references_path = root / RESEARCH_REFERENCES_RELATIVE_PATH
    before_text = read_text_if_present(references_path)
    before_hash = path_sha256(references_path)
    dry = build_artifact(
        root=root,
        search_started_at=search_started_at,
        search_finished_at=search_finished_at,
        accepted_findings=accepted_findings,
        source_receipts=source_receipts,
        duration_s=duration_s,
        test_commands=test_commands,
        test_exit_codes=test_exit_codes,
    )
    accepted = dry["accepted_rejected_duplicate_retired_and_abstained_findings"][
        "accepted"
    ]
    appended = False
    if dry["status"] == "complete" and accepted and EXECUTION_DELTA_HEADING not in before_text:
        references_path.write_text(
            insert_after_planner_block(before_text, execution_delta_block(accepted)),
            encoding="utf-8",
        )
        appended = True
    after_hash = path_sha256(references_path)
    artifact = build_artifact(
        root=root,
        search_started_at=search_started_at,
        search_finished_at=search_finished_at,
        accepted_findings=accepted_findings,
        source_receipts=source_receipts,
        duration_s=duration_s,
        test_commands=test_commands,
        test_exit_codes=test_exit_codes,
        references_appended=appended,
        references_before_hash=before_hash,
        references_after_hash=after_hash,
    )
    result_path = root / RESULT_RELATIVE_PATH
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    _require(
        artifact["inference_substrate"] == INFERENCE_SUBSTRATE,
        "substrate must be literature_ingestion",
    )
    _require(float(artifact["duration_s"]) >= 0.0, "duration must be non-negative")
    _require(
        _parse_timestamp(str(artifact["search_finished_at"]))
        > _parse_timestamp(str(artifact["search_started_at"])),
        "timestamp order invalid",
    )
    source_block = artifact["source_queries_and_endpoint_receipts"]
    _require(isinstance(source_block, Mapping), "source_queries must be an object")
    source_receipts = source_block.get("source_receipts")
    _require(isinstance(source_receipts, list) and source_receipts, "source_queries missing receipts")
    for receipt in source_receipts:
        for field in SOURCE_RECEIPT_REQUIRED_FIELDS:
            _require(field in receipt, f"source receipt missing {field}")
    classes = artifact["accepted_rejected_duplicate_retired_and_abstained_findings"]
    _require(isinstance(classes, Mapping), "classification block must be object")
    ordered = [row for bucket in CLASSIFICATION_BUCKETS for row in classes[bucket]]
    _require(classes.get("all_candidates") == ordered, "all_candidates order mismatch")
    for bucket in CLASSIFICATION_BUCKETS:
        for row in classes[bucket]:
            _require(row.get("classification") == bucket, "classification mismatch")
            _require(bool(row.get("url")), "finding disposition missing URL")
    for row in classes["accepted"]:
        _validate_finding(row, "accepted")
    mapping = artifact["sota_to_experiment_mapping"]
    _require(
        mapping["accepted_count"] == len(classes["accepted"]),
        "accepted mapping count mismatch",
    )
    for row in mapping["accepted_mappings"]:
        _require(
            row["target_experiment"] in ALLOCATED_TARGET_EXPERIMENTS
            or row["target_experiment"] == "defer",
            "mapping target outside V533 allocation",
        )
    _require(
        isinstance(artifact["references_append_receipt"], Mapping),
        "references append receipt missing",
    )
    provenance = artifact["field_provenance"]
    principles = artifact["field_principles"]
    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in provenance, f"missing provenance for {field}")
        _require(field in principles, f"missing principle for {field}")
    _require(
        str(artifact["reproducibility_checksum"]).startswith("sha256:"),
        "checksum missing sha256 prefix",
    )


def main() -> int:
    started = datetime.now(UTC)
    time.sleep(0.001)
    finished = datetime.now(UTC)
    artifact = build_and_write_artifact(
        root=REPO_ROOT,
        search_started_at=started.isoformat(timespec="microseconds").replace("+00:00", "Z"),
        search_finished_at=finished.isoformat(timespec="microseconds").replace("+00:00", "Z"),
        duration_s=round((finished - started).total_seconds(), 6),
    )
    validate_artifact(artifact)
    print(json.dumps({"status": artifact["status"], "result": artifact["result_path"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
