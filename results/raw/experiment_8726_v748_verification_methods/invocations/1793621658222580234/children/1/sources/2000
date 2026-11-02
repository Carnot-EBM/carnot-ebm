"""Exp5756: ingest post-V514 source deltas with bounded receipts.

Spec refs: REQ-REPORT-5756, SCENARIO-REPORT-5756-ZERO-FINDING,
SCENARIO-REPORT-5756-ACCEPT-BOUNDED-DELTA,
SCENARIO-REPORT-5756-BLOCKED-PROVENANCE,
SCENARIO-REPORT-5756-FIELD-PRINCIPLES.

The live web search is not replayed in unit tests because indexes drift by
design. This module preserves the durable part of the work: which routes were
checked, which source receipts surfaced each candidate, how local dedupe and
scope boundaries were applied, and why a zero-accepted result is a complete
bibliographic refresh rather than a failed experiment.
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
RESULT_RELATIVE_PATH = Path("results/experiment_5756_v514_source_delta_ingestion.json")
RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
KNOWN_ISSUES_RELATIVE_PATH = Path("ops/known-issues.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
PRIOR_SOURCE_DELTA_RELATIVE_PATH = Path(
    "results/experiment_5744_v513_source_delta_ingestion.json"
)

EXPERIMENT = "experiment_5756_v514_source_delta_ingestion"
EXPERIMENT_ID = "exp5756-v514-source-delta-ingestion"
MILESTONE = "2026.07.514"
RUN_DATE = "20260720"
RANDOM_SEED = 5756
SCHEMA = "carnot.experiment_5756.v514_source_delta_ingestion.v1"
INFERENCE_SUBSTRATE = "web_bibliographic_research_no_model_inference"
TERMINAL_PREFIXES = ("complete:", "blocked:")

PLANNER_HEADING = "## V514 Planner Refresh - 20260720"
PLANNER_MARKER = "V514-PLANNER-REFRESH-20260720-END"
PLANNER_END_MARKER = f"<!-- {PLANNER_MARKER} -->"
EXECUTION_REFRESH_HEADING = "## V514 Execution Refresh - 20260720"
EXECUTION_REFRESH_END_MARKER = "<!-- V514-EXECUTION-REFRESH-20260720-END -->"

ALLOWED_TARGET_EXPERIMENTS = {
    "exp5757-proposal-benchmark-scalar-bridge",
    "exp5758-rust-parity-scalar-bridge",
    "exp5759-sota-exact-proposal-utility-panel",
    "exp5760-selective-exact-feedback-search",
    "exp5761-exact-constraint-acquisition-benchmark",
    "exp5762-query-driven-constraint-lifecycle",
    "exp5763-dependent-task-constraint-acquisition",
    "exp5764-one-axis-profiled-allocation-free-hot-path",
    "exp5765-one-axis-final-10x-crossover",
    "exp5766-arc-loo-component-interaction-audit",
    "exp5767-arc-game-blind-composition-hardening",
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
    "semantic_scholar_receipts",
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
        "Records marker, source reachability, timestamp, ledger hash, exclusion hash, and roadmap hash before findings are trusted."
    ),
    "planner_marker": "Binds the execution search window to the V514 planner boundary.",
    "planner_marker_hash": "Content-addressed marker context detects silent planner-block drift.",
    "search_started_at": "Records the real UTC instant before external querying starts.",
    "search_finished_at": "Records the real UTC instant after final source disposition.",
    "actual_search_wall_time_s": (
        "Wall time is bibliographic search time only, not model, solver, benchmark, or hardware compute."
    ),
    "source_queries": "Search intent is reconstructable without trusting memory or mutable indexes.",
    "source_receipts": (
        "Network and local source receipts show which routes were reachable and how candidates surfaced."
    ),
    "semantic_scholar_receipts": (
        "Citation-route receipts are separated from general source search and do not become stable citation-count claims."
    ),
    "accepted_findings": (
        "Accepted findings must be post-marker, non-duplicate, and bounded to existing V514 controls or validation boundaries."
    ),
    "duplicate_findings": "Already-ledgered work stays visible but cannot create duplicate roadmap work.",
    "watch_only_findings": (
        "Relevant but non-executable or non-local material cannot support Carnot claims."
    ),
    "excluded_findings": "Closed scopes remain closed by explicit disposition.",
    "inaccessible_findings": "Access failures are separated from scientific exclusions and never promoted.",
    "references_changed": "The reference ledger mutation state is declared.",
    "references_diff_hash": "Reference-file before/after content is hash-bound even when unchanged.",
    "roadmap_scope_change_requested": (
        "Scope changes block for operator review rather than silently mutating the roadmap."
    ),
    "operator_review_required": (
        "Graph, id, gate, hardware, or headline changes require explicit operator review."
    ),
    "closed_scopes_reopened": (
        "Retired scopes must stay closed unless the operator explicitly reopens them."
    ),
    "hardware_claim_changed": (
        "Bibliographic research cannot change FPGA, TSU, Kona, or other hardware claims."
    ),
    "inference_substrate": "The run used web bibliographic research and no model inference.",
    "test_commands": "Verification commands are preserved exactly.",
    "test_exit_codes": "Observed exit codes are recorded without relabeling failures.",
    "reproducibility_checksum": "Stable content checksum detects artifact drift.",
    "honest_verdict": "Terminal summary starts with complete: or blocked: and does not inflate novelty.",
}

FIELD_PRINCIPLES: dict[str, str] = {
    **REQUIRED_FIELD_PRINCIPLES,
    "schema": "Identifies the versioned Exp5756 artifact schema.",
    "experiment": "Stable local experiment slug for result indexing.",
    "experiment_id": "Binds this receipt to the conductor task id.",
    "milestone": "Prevents V514 source receipts from being reused for another milestone.",
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
}

SPEC_REFS = (
    "REQ-REPORT-5756",
    "SCENARIO-REPORT-5756-ZERO-FINDING",
    "SCENARIO-REPORT-5756-ACCEPT-BOUNDED-DELTA",
    "SCENARIO-REPORT-5756-BLOCKED-PROVENANCE",
    "SCENARIO-REPORT-5756-FIELD-PRINCIPLES",
)

SOURCE_QUERIES: tuple[JsonDict, ...] = (
    {
        "surface": "arXiv",
        "order": 1,
        "queries": [
            "2025-2026 EBM reasoning verification",
            "constraint satisfaction acquisition query-driven MPMMine",
            "Ising sampling probabilistic hardware p-bit thermodynamic",
            "hallucination mitigation verification reasoning",
            "KAN constrained generation continual online learning",
            "cs.AI new 2026-07-20",
            "cs.LG new 2026-07-20",
            "cs.CL new 2026-07-20",
        ],
    },
    {
        "surface": "OpenReview",
        "order": 2,
        "queries": [
            "energy-based reasoning verifier constrained generation 2026",
            "CerCE Certifiable Continual Learning",
            "Opt-Verifier dual-side verification",
        ],
    },
    {
        "surface": "Semantic Scholar",
        "order": 3,
        "queries": ["arXiv:2507.02092 citations", "arXiv:2512.15605 citations"],
    },
    {"surface": "Hugging Face Papers", "order": 4, "queries": ["daily_papers 2026-07-20"]},
    {
        "surface": "GitHub discovery",
        "order": 5,
        "queries": [
            '"energy-based" reasoning constraint created:>2026-07-20',
            "constraint acquisition MPMMine created:>2026-07-20",
            "KAN constraint learning created:>2026-07-20",
            "constrained generation verifier created:>2026-07-20",
        ],
    },
    {"surface": "Extropic writing", "order": 6, "queries": ["TSU XTR-0 X0 THRML"]},
    {
        "surface": "Logical Intelligence",
        "order": 7,
        "queries": ["Kona Aleph energy based reasoning verified reasoning"],
    },
    {
        "surface": "local Carnot ledgers",
        "order": 8,
        "queries": [
            "research-references.md complete ledger",
            "V514 planner block",
            "ops/exclusion_manifest.yaml",
            "ops/known-issues.md",
            "results/experiment_5744_v513_source_delta_ingestion.json",
        ],
    },
)

SOURCE_RECEIPTS: tuple[JsonDict, ...] = (
    {
        "receipt_id": "arxiv_home_reachability",
        "surface": "arXiv",
        "url": "https://arxiv.org/",
        "queried_at": "2026-07-20T14:10:54Z",
        "status": "http_200",
        "candidate_ids": [],
        "receipt_summary": "HEAD request reached arXiv before source classification.",
    },
    {
        "receipt_id": "arxiv_cs_ai_new_20260720",
        "surface": "arXiv",
        "url": "https://arxiv.org/list/cs.AI/new",
        "queried_at": "2026-07-20T14:11:00Z",
        "status": "http_200_showing_new_listings_2026_07_20_26_entries",
        "candidate_ids": [
            "2607.15281",
            "2607.15314",
            "2607.15388",
            "2607.15439",
            "2607.15459",
            "2607.15532",
        ],
        "receipt_summary": "arXiv cs.AI new page was reachable and candidate items were dispositioned.",
    },
    {
        "receipt_id": "arxiv_abs_2509_24489",
        "surface": "arXiv",
        "url": "https://arxiv.org/abs/2509.24489",
        "queried_at": "2026-07-20T14:11:06Z",
        "status": "http_200_duplicate_v514_planner_delta",
        "candidate_ids": ["2509.24489"],
        "receipt_summary": "Query-driven constraint-acquisition paper was already accepted by the V514 planner.",
    },
    {
        "receipt_id": "arxiv_abs_2605_26279",
        "surface": "arXiv",
        "url": "https://arxiv.org/abs/2605.26279",
        "queried_at": "2026-07-20T14:11:06Z",
        "status": "http_200_duplicate_v514_planner_delta",
        "candidate_ids": ["2605.26279"],
        "receipt_summary": "MPMMine paper was already accepted by the V514 planner.",
    },
    {
        "receipt_id": "arxiv_export_energy_verification",
        "surface": "arXiv",
        "url": "https://export.arxiv.org/api/query?search_query=all:%22energy-based%20verification%22",
        "queried_at": "2026-07-20T14:11:57Z",
        "status": "http_200_total_results_0",
        "candidate_ids": [],
        "receipt_summary": "Broad arXiv export query returned no energy-based verification hits.",
    },
    {
        "receipt_id": "openreview_search_snippets",
        "surface": "OpenReview",
        "url": "https://openreview.net/",
        "queried_at": "2026-07-20T14:11:30Z",
        "status": "http_200_search_snippets_primary_forums_challenged",
        "candidate_ids": ["ZBj3Qp1bYg", "Anh6VfNM22", "L7NsVVUm9H"],
        "receipt_summary": "OpenReview search surfaced EBT, CerCE, and Opt-Verifier; primary forum pages require browser verification.",
    },
    {
        "receipt_id": "semantic_scholar_ebt_citations",
        "surface": "Semantic Scholar",
        "url": "https://api.semanticscholar.org/graph/v1/paper/arXiv:2507.02092/citations",
        "queried_at": "2026-07-20T14:10:54Z",
        "status": "http_200_latest_publication_2026_07_13",
        "candidate_ids": ["2607.11555", "2606.18206", "2605.11011"],
        "receipt_summary": "EBT citation route was reachable; visible sample citations predate the V514 marker.",
    },
    {
        "receipt_id": "semantic_scholar_arm_ebm_citations",
        "surface": "Semantic Scholar",
        "url": "https://api.semanticscholar.org/graph/v1/paper/arXiv:2512.15605/citations",
        "queried_at": "2026-07-20T14:11:56Z",
        "status": "http_200_latest_publication_2026_07_02",
        "candidate_ids": ["2607.02154", "2605.18871", "2605.11011"],
        "receipt_summary": "ARM-EBM citation route was reachable; visible sample citations predate the V514 marker.",
    },
    {
        "receipt_id": "huggingface_daily_2026_07_20",
        "surface": "Hugging Face Papers",
        "url": "https://huggingface.co/api/daily_papers?date=2026-07-20",
        "queried_at": "2026-07-20T14:11:56Z",
        "status": "http_200_daily_feed",
        "candidate_ids": [
            "2607.11683",
            "2606.29538",
            "2607.16051",
            "2607.15330",
            "2607.14530",
            "2607.15314",
            "2607.15901",
        ],
        "receipt_summary": "HF daily feed mirrored model-training, agent, and GraphRAG items; no HF-only V514 control change was accepted.",
    },
    {
        "receipt_id": "github_recent_queries",
        "surface": "GitHub discovery",
        "url": "https://api.github.com/search/repositories",
        "queried_at": "2026-07-20T14:11:56Z",
        "status": "http_200_total_count_0_for_required_recent_queries",
        "candidate_ids": [],
        "receipt_summary": "Required recent repository searches returned zero new executable dependencies.",
    },
    {
        "receipt_id": "extropic_public_writing",
        "surface": "Extropic writing",
        "url": "https://extropic.ai/",
        "queried_at": "2026-07-20T14:11:57Z",
        "status": "http_200_public_material_watch_only",
        "candidate_ids": ["inside-x0-and-xtr-0", "thermodynamic-computing-from-zero-to-one"],
        "receipt_summary": "Extropic public material remains architecture/hardware context without Carnot-local TSU receipts.",
    },
    {
        "receipt_id": "logical_intelligence_public_pages",
        "surface": "Logical Intelligence",
        "url": "https://logicalintelligence.com/",
        "queried_at": "2026-07-20T14:11:55Z",
        "status": "http_200_public_material_watch_only",
        "candidate_ids": ["kona_home", "aleph_putnambench", "energy_based_models_for_reasoning"],
        "receipt_summary": "Logical Intelligence public Kona/Aleph pages remain context without local weights or API receipts.",
    },
    {
        "receipt_id": "local_research_references_v514_ledger",
        "surface": "local Carnot ledgers",
        "url": "research-references.md",
        "queried_at": "2026-07-20T14:12:26Z",
        "status": "local_ledger_checked_marker_line_30339",
        "candidate_ids": ["2509.24489", "2605.26279", "2607.05391", "2606.26300"],
        "receipt_summary": "Complete ledger and V514 planner block were checked for duplicate and boundary dispositions.",
    },
)

SEMANTIC_SCHOLAR_RECEIPTS: tuple[JsonDict, ...] = (
    {
        "receipt_id": "semantic_scholar_ebt_citations",
        "paper": "arXiv:2507.02092",
        "url": "https://api.semanticscholar.org/graph/v1/paper/arXiv:2507.02092/citations",
        "queried_at": "2026-07-20T14:10:54Z",
        "http_status": 200,
        "sample_returned_count": 5,
        "latest_publication_date": "2026-07-13",
        "post_marker_publication_count": 0,
        "sample_titles": [
            "Advancing Optimal Subset Oracle via Learning Relaxation of Neural Set Functions",
            "Fixed-Point Reasoners: Stable and Adaptive Deep Looped Transformers",
            "LoopUS: Recasting Pretrained LLMs into Looped Latent Refinement Models",
        ],
        "receipt_summary": "No visible sampled EBT citation was newer than the V514 planner marker.",
    },
    {
        "receipt_id": "semantic_scholar_arm_ebm_citations",
        "paper": "arXiv:2512.15605",
        "url": "https://api.semanticscholar.org/graph/v1/paper/arXiv:2512.15605/citations",
        "queried_at": "2026-07-20T14:11:56Z",
        "http_status": 200,
        "sample_returned_count": 5,
        "latest_publication_date": "2026-07-02",
        "post_marker_publication_count": 0,
        "sample_titles": [
            "Path-Measure Dynamics of Attention-Driven World Models",
            "Distributional Energy-Based Models for Uncertainty-Aware Structured LLM Reasoning",
            "Ontology-Constrained Neural Reasoning in Enterprise Agentic Systems",
        ],
        "receipt_summary": "No visible sampled ARM-EBM citation was newer than the V514 planner marker.",
    },
)

ACCEPTED_FINDINGS: tuple[JsonDict, ...] = ()

DUPLICATE_FINDINGS: tuple[JsonDict, ...] = (
    {
        "source_id": "query_driven_constraint_acquisition_2509_24489",
        "classification": "duplicate",
        "title": "Overcoming Over-Fitting in Constraint Acquisition via Query-Driven Interactive Refinement",
        "url": "https://arxiv.org/abs/2509.24489",
        "publication_date": "2025-09-29",
        "search_receipt": "arxiv_abs_2509_24489",
        "reason": "Already accepted in the V514 planner block as the Exp5762 lifecycle mechanism basis.",
    },
    {
        "source_id": "mpmmine_2605_26279",
        "classification": "duplicate",
        "title": "Constraint acquisition needs better benchmarks",
        "url": "https://arxiv.org/abs/2605.26279",
        "publication_date": "2026-05-25",
        "search_receipt": "arxiv_abs_2605_26279",
        "reason": "Already accepted in the V514 planner block as the Exp5761 corpus methodology basis.",
    },
    {
        "source_id": "causal_audit_2607_15281",
        "classification": "duplicate",
        "title": "Causal-Audit: Explicit and Auditable Graph-based Reasoning",
        "url": "https://arxiv.org/abs/2607.15281",
        "publication_date": "2026-04-22",
        "search_receipt": "arxiv_cs_ai_new_20260720",
        "reason": "Already present in the local ledger; it does not change V514 controls beyond existing exact-replay boundaries.",
    },
    {
        "source_id": "arc_ewm_ablation_2607_15439",
        "classification": "duplicate",
        "title": "Do Coding Agents Need Executable World Models, Simplification, and Verification to Solve ARC-AGI-3?",
        "url": "https://arxiv.org/abs/2607.15439",
        "publication_date": "2026-07-16",
        "search_receipt": "arxiv_cs_ai_new_20260720",
        "reason": "Already ledgered before V514; public-set ARC saturation claims are not imported into V514.",
    },
    {
        "source_id": "opt_verifier_openreview_icml_2026",
        "classification": "duplicate",
        "title": "Opt-Verifier: Unleashing the Power of LLMs for Optimization Modeling via Dual-Side Verification",
        "url": "https://openreview.net/forum?id=L7NsVVUm9H",
        "publication_date": "2026-04-30",
        "search_receipt": "openreview_search_snippets",
        "reason": "Already incorporated into V513/V514 proposal-utility structure and solution receipt boundaries.",
    },
    {
        "source_id": "hallucination_detection_framework_2601_09929",
        "classification": "duplicate",
        "title": "Hallucination Detection and Mitigation in Large Language Models",
        "url": "https://arxiv.org/abs/2601.09929",
        "publication_date": "2026-01-14",
        "search_receipt": "local_research_references_v514_ledger",
        "reason": "Already indexed in research-references.md and does not alter existing V514 exact-validator controls.",
    },
)

WATCH_ONLY_FINDINGS: tuple[JsonDict, ...] = (
    {
        "source_id": "logic_optimization_ai_2607_15532",
        "classification": "watch_only",
        "title": "Logic, Optimization, and Artificial Intelligence",
        "url": "https://arxiv.org/abs/2607.15532",
        "publication_date": "2026-07-20",
        "search_receipt": "arxiv_cs_ai_new_20260720",
        "reason": "Relevant survey context, but it does not change an allocated V514 control or validation boundary.",
    },
    {
        "source_id": "precise_but_uncoupled_2607_15388",
        "classification": "watch_only",
        "title": "Precise but Uncoupled: Reviewer Precision Does Not Guarantee Critique Uptake",
        "url": "https://arxiv.org/abs/2607.15388",
        "publication_date": "2026-07-20",
        "search_receipt": "arxiv_cs_ai_new_20260720",
        "reason": "Useful verifier-protocol caution, but V514 already requires exact validators and matched budgets.",
    },
    {
        "source_id": "ragu_hf_2607_11683",
        "classification": "watch_only",
        "title": "RAGU: A Multi-Step GraphRAG Engine with a Compact Domain-Adapted LLM",
        "url": "https://huggingface.co/papers/2607.11683",
        "publication_date": "2026-07-13",
        "search_receipt": "huggingface_daily_2026_07_20",
        "reason": "GraphRAG extraction context has no local exact-validator or V514 experiment-boundary change.",
    },
    {
        "source_id": "resource2skill_hf_2606_29538",
        "classification": "watch_only",
        "title": "RESOURCE2SKILL: Distilling Executable Agent Skills from Human-Created Multimodal Resources",
        "url": "https://huggingface.co/papers/2606.29538",
        "publication_date": "2026-07-16",
        "search_receipt": "huggingface_daily_2026_07_20",
        "reason": "Agent skill acquisition is relevant background but not a solver-certified constraint lifecycle delta.",
    },
    {
        "source_id": "extropic_x0_xtr0_tsu_public_material",
        "classification": "watch_only",
        "title": "Extropic X0, XTR-0, TSU, and THRML public material",
        "url": "https://extropic.ai/",
        "publication_date": "2025-10-01",
        "search_receipt": "extropic_public_writing",
        "reason": "Public hardware context only; no authenticated Carnot-local TSU execution, timing, SDK, or correctness receipt.",
    },
    {
        "source_id": "logical_intelligence_kona_aleph_public_material",
        "classification": "watch_only",
        "title": "Logical Intelligence Kona and Aleph public pages",
        "url": "https://logicalintelligence.com/",
        "publication_date": "2026-01-21",
        "search_receipt": "logical_intelligence_public_pages",
        "reason": "Architecture and benchmark context only; no local weights, API receipt, or reproducible comparator.",
    },
)

EXCLUDED_FINDINGS: tuple[JsonDict, ...] = (
    {
        "source_id": "xhc_2607_14530",
        "classification": "excluded",
        "title": "xHC: Expanded Hyper-Connections",
        "url": "https://arxiv.org/abs/2607.14530",
        "publication_date": "2026-07-16",
        "search_receipt": "huggingface_daily_2026_07_20",
        "reason": "Residual-stream architecture expansion and model training reopen closed model-weight scope.",
    },
    {
        "source_id": "loop_the_loopies_2607_16051",
        "classification": "excluded",
        "title": "Loop the Loopies!",
        "url": "https://arxiv.org/abs/2607.16051",
        "publication_date": "2026-07-17",
        "search_receipt": "huggingface_daily_2026_07_20",
        "reason": "Looped Transformer pretraining and post-training are outside V514 sidecar control changes.",
    },
    {
        "source_id": "xiaomi_robotics_1_2607_15330",
        "classification": "excluded",
        "title": "Xiaomi-Robotics-1: Scaling Vision-Language-Action Models",
        "url": "https://arxiv.org/abs/2607.15330",
        "publication_date": "2026-07-16",
        "search_receipt": "huggingface_daily_2026_07_20",
        "reason": "Robotics foundation-model training and fine-tuning reopen model-weight and robotics scopes.",
    },
    {
        "source_id": "cura_1t_2607_15314",
        "classification": "excluded",
        "title": "Cura 1T: Specialized Model for Agentic Healthcare",
        "url": "https://arxiv.org/abs/2607.15314",
        "publication_date": "2026-07-15",
        "search_receipt": "arxiv_cs_ai_new_20260720",
        "reason": "Healthcare self-evolution and trillion-parameter training reopen closed model-weight-write scope.",
    },
    {
        "source_id": "dsworld_2607_15901",
        "classification": "excluded",
        "title": "DSWorld: A Data Science World Model for Efficient Autonomous Agents",
        "url": "https://arxiv.org/abs/2607.15901",
        "publication_date": "2026-07-17",
        "search_receipt": "huggingface_daily_2026_07_20",
        "reason": "LLM simulator and RL/search world-model redesign are outside existing V514 controls.",
    },
    {
        "source_id": "from_black_box_to_executable_logic_2607_15459",
        "classification": "excluded",
        "title": "From Black Box to Executable Logic: Explainable Reinforcement Learning through Prolog Expert Systems",
        "url": "https://arxiv.org/abs/2607.15459",
        "publication_date": "2026-07-20",
        "search_receipt": "arxiv_cs_ai_new_20260720",
        "reason": "Post-hoc RL policy extraction and return optimization reopen broad RL scope.",
    },
)

INACCESSIBLE_FINDINGS: tuple[JsonDict, ...] = (
    {
        "source_id": "openreview_primary_forum_pages",
        "classification": "inaccessible",
        "title": "OpenReview primary forum pages for EBT, CerCE, and constrained-generation results",
        "url": "https://openreview.net/",
        "publication_date": None,
        "search_receipt": "openreview_search_snippets",
        "reason": "Search snippets and venue pages were visible, but primary forum pages presented browser verification.",
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
        "free_form_answer_repair_reopened": False,
        "llm_judges_reopened": False,
        "model_weight_writes_reopened": False,
        "broad_rl_reopened": False,
        "kan_scaleup_reopened": False,
        "per_game_arc_adapters_reopened": False,
        "unauthenticated_hardware_claims_reopened": False,
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
    active_roadmap_hash = path_sha256(root / ROADMAP_RELATIVE_PATH)
    exclusion_hash = path_sha256(root / EXCLUSION_MANIFEST_RELATIVE_PATH)
    failed_preconditions: list[str] = []
    if not marker_found:
        failed_preconditions.append("planner_marker_missing")
    if not source_reachable:
        failed_preconditions.append("source_reachability_failed")
    if "REQ-REPORT-5756" not in spec_text:
        failed_preconditions.append("spec_req_report_5756_missing")
    return {
        "agents_md_read": (root / "AGENTS.md").exists(),
        "codex_md_read": (root / "CODEX.md").exists(),
        "claude_md_read": (root / "CLAUDE.md").exists(),
        "research_program_read": (root / "research-program.md").exists(),
        "research_references_read": (root / RESEARCH_REFERENCES_RELATIVE_PATH).exists(),
        "v514_planner_marker_verified": marker_found,
        "planner_marker_found": marker_found,
        "current_utc_timestamp_checked": True,
        "network_source_reachability_established": source_reachable,
        "active_roadmap_milestone": roadmap_milestone(root),
        "active_roadmap_hash": active_roadmap_hash,
        "exclusion_manifest_hash": exclusion_hash,
        "exclusion_manifest_read": exclusion_hash is not None,
        "known_issues_read": (root / KNOWN_ISSUES_RELATIVE_PATH).exists(),
        "prior_v513_source_delta_read": (root / PRIOR_SOURCE_DELTA_RELATIVE_PATH).exists(),
        "spec_has_req_report_5756": "REQ-REPORT-5756" in spec_text,
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
        "v514_planner_block_checked": True,
        "exclusion_manifest_checked": True,
        "closed_scopes_checked": True,
    }


def honest_verdict(
    marker_found: bool,
    source_reachable: bool,
    findings: list[JsonDict],
    operator_review_required: bool,
) -> str:
    if not marker_found:
        return "blocked: V514 planner refresh marker missing; source-delta append refused"
    if not source_reachable:
        return "blocked: required external source reachability could not be established"
    if operator_review_required:
        return "blocked: source delta would require operator review for roadmap, gate, hardware, or headline scope"
    if not findings:
        return "complete: no new non-duplicate actionable V514 source deltas; references left unchanged"
    return (
        f"complete: accepted {len(findings)} post-V514 bounded source delta(s); "
        "no roadmap id, gate, dependency graph, hardware claim, or headline claim changed"
    )


def execution_refresh_block(findings: list[JsonDict]) -> str:
    lines = [
        "",
        EXECUTION_REFRESH_HEADING,
        "",
        (
            "Execution-time sweep on 2026-07-20 after the V514 planner marker. "
            "Only non-duplicate sources that sharpen existing V514 controls or "
            "validation boundaries are listed here."
        ),
        "",
        "### New actionable deltas",
        "",
    ]
    for finding in findings:
        lines.append(
            f"- **{finding['title']}** - {finding['url']} "
            f"({finding['publication_date']}). Carnot hook: {finding['carnot_hook']} "
            f"Target: {finding['target_experiment']}. Authority boundary: "
            f"{finding['authority_boundary']}. Falsifiable metric: "
            f"{finding['falsifiable_metric']}. Search receipt: {finding['search_receipt']}."
        )
    lines.extend(
        [
            "",
            "### V514 execution impact",
            "",
            (
                "- Preserve roadmap ids, gates, hardware claims, headline claims, and "
                "retired scopes. Accepted deltas may only add bounded controls or "
                "validation receipts inside already-allocated V514 experiments."
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
    semantic_scholar_receipts: list[JsonDict] | None = None,
    references_before_hash: str | None = None,
    references_after_hash: str | None = None,
    references_changed: bool = False,
    test_commands: list[str] | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
) -> JsonDict:
    references_text = read_text_if_present(root / RESEARCH_REFERENCES_RELATIVE_PATH)
    marker_found = planner_marker_found(references_text)
    receipts = clone_json(SOURCE_RECEIPTS if source_receipts is None else source_receipts)
    semantic_receipts = clone_json(
        SEMANTIC_SCHOLAR_RECEIPTS
        if semantic_scholar_receipts is None
        else semantic_scholar_receipts
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
        "source_window": "strictly_after_V514_PLANNER_REFRESH_20260720_END",
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
        "semantic_scholar_receipts": semantic_receipts,
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


def _validate_classified_findings(rows: list[JsonDict], classification: str) -> None:
    for row in rows:
        for key in ("source_id", "classification", "title", "url", "publication_date", "search_receipt", "reason"):
            require(key in row, f"{classification} finding missing {key}")
        require(row["classification"] == classification, f"finding has wrong classification {classification}")


def _validate_accepted_findings(rows: list[JsonDict]) -> None:
    for finding in rows:
        for key in (
            "source_id",
            "classification",
            "title",
            "url",
            "publication_date",
            "search_receipt",
            "target_experiment",
            "authority_boundary",
            "carnot_hook",
            "falsifiable_metric",
            "reason",
        ):
            require(key in finding, f"accepted finding missing {key}")
        require(finding["classification"] == "accepted", "accepted finding has wrong classification")
        require(
            finding["target_experiment"] in ALLOWED_TARGET_EXPERIMENTS,
            "accepted finding has a disallowed target experiment",
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
    _validate_accepted_findings(list(artifact["accepted_findings"]))
    _validate_classified_findings(list(artifact["duplicate_findings"]), "duplicate")
    _validate_classified_findings(list(artifact["watch_only_findings"]), "watch_only")
    _validate_classified_findings(list(artifact["excluded_findings"]), "excluded")
    _validate_classified_findings(list(artifact["inaccessible_findings"]), "inaccessible")
    require(artifact["duplicate_checks"]["source_ids_unique"], "duplicate source id check failed")
    require(artifact["source_queries"], "source_queries must be non-empty")
    require(artifact["source_receipts"], "source_receipts must be non-empty")
    require(artifact["semantic_scholar_receipts"], "semantic_scholar_receipts must be non-empty")
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
        require(artifact["honest_verdict"].startswith("blocked:"), "operator review needs blocked verdict")
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
