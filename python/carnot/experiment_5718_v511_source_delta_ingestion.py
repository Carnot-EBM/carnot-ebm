"""Exp5718: ingest the V511 execution-time source delta.

Spec refs: REQ-REPORT-5718,
SCENARIO-REPORT-5718-ACCEPT-BOUNDED-DELTA,
SCENARIO-REPORT-5718-BLOCKED-MARKER,
SCENARIO-REPORT-5718-FIELD-PRINCIPLES.

The live web search is not run from the test suite because public indexes,
daily feeds, and citation APIs drift. This module records the durable part of
the work: the post-V511 source sweep, the duplicate/watch/excluded decisions,
and the single accepted source that sharpens an already-planned FR-11
regression-control boundary without changing experiment IDs or gates.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import yaml


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5718_v511_source_delta_ingestion.json")
RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
KNOWN_ISSUES_RELATIVE_PATH = Path("ops/known-issues.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")

EXPERIMENT = "experiment_5718_v511_source_delta_ingestion"
EXPERIMENT_ID = "exp5718-v511-source-delta-ingestion"
MILESTONE = "2026.07.511"
RUN_DATE = "20260719"
SEARCH_CUTOFF = "2026-07-19"
SCHEMA = "carnot.experiment_5718.v511_source_delta_ingestion.v1"
RANDOM_SEED = 5718
INFERENCE_SUBSTRATE = "web_and_bibliographic_search_only"
TERMINAL_PREFIXES = ("complete:", "blocked:")

PLANNER_MARKER = "V511 Planner Refresh - 20260715"
PLANNER_HEADING = f"## {PLANNER_MARKER}"
PLANNER_HEADING_COMPACT = PLANNER_HEADING.replace("-", "")
PLANNER_END_MARKER = "<!-- V511-PLANNER-REFRESH-20260715-END -->"
EXECUTION_REFRESH_HEADING = "## V511 Execution Refresh - 20260719"

ALLOWED_TARGET_EXPERIMENTS = {
    "exp5719-sota-answer-channel-forensics",
    "exp5720-sota-attested-exact-envelope-canary",
    "exp5721-fr11-memops-lifecycle-shadow-stream",
    "exp5722-fr11-compliance-recovery-rollback-canary",
    "exp5723-one-axis-rust-samplerbackend-integration",
    "exp5724-one-axis-rust-python-matched-crossover",
    "exp5725-arc-epistemic-ledger-live-qualification",
    "exp5726-arc-epistemic-ledger-live-ab",
    "exp5727-arc-live-self-discovery-levelup-v511",
}

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "search_timestamp_utc",
    "planner_marker",
    "sources_checked",
    "queries",
    "accepted_findings",
    "duplicate_findings",
    "watch_only_findings",
    "excluded_findings",
    "semantic_scholar_status",
    "extropic_status",
    "logical_intelligence_status",
    "huggingface_status",
    "github_status",
    "target_experiment_map",
    "roadmap_change_required",
    "references_updated",
    "inference_substrate",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "field_principles": "One-line annotations for every required headline and gate field.",
    "search_timestamp_utc": "freshness/coverage exact",
    "planner_marker": "the search window is anchored",
    "sources_checked": "coverage reconstructs",
    "queries": "coverage reconstructs",
    "accepted_findings": "accepted work has a local exact home",
    "duplicate_findings": "dispositions are explicit",
    "watch_only_findings": "unavailable systems support no claim",
    "excluded_findings": "closed scopes remain closed",
    "semantic_scholar_status": "citation-route access is honest",
    "extropic_status": "hardware access is honest",
    "logical_intelligence_status": "Kona access is honest",
    "huggingface_status": "secondary paper-feed access is honest",
    "github_status": "repository-route access is honest",
    "target_experiment_map": "accepted work has a home",
    "roadmap_change_required": "scope expansion blocks instead of mutating gates",
    "references_updated": "mutations are declared",
    "inference_substrate": "no benchmark inference occurred",
    "reproducibility_checksum": "the report is stable",
    "honest_verdict": "zero findings can be complete",
}

SPEC_REFS = (
    "REQ-REPORT-5718",
    "SCENARIO-REPORT-5718-ACCEPT-BOUNDED-DELTA",
    "SCENARIO-REPORT-5718-BLOCKED-MARKER",
    "SCENARIO-REPORT-5718-FIELD-PRINCIPLES",
)

QUERIES: tuple[JsonDict, ...] = (
    {
        "surface": "arXiv",
        "timestamp_window_utc": "2026-07-15T00:00:00Z/2026-07-19T23:59:59Z",
        "queries": [
            'all:"energy based"',
            'all:"constraint satisfaction"',
            "all:KAN",
            'all:"hallucination mitigation"',
            "all:Ising",
            'all:"constrained generation"',
            'all:"continual learning"',
            'all:"hardware sampling"',
            'all:"neural CSP"',
            'all:"energy-based verification"',
            'all:"EBM" AND all:"reasoning"',
            'all:"constraint learning"',
        ],
    },
    {
        "surface": "OpenReview",
        "queries": [
            "energy-based reasoning",
            "constrained generation",
            "continual learning verifiable",
            "ISM Self-Improving Strategy Memory",
        ],
    },
    {
        "surface": "Semantic Scholar",
        "queries": ["arXiv:2507.02092 citations", "arXiv:2512.15605 citations"],
    },
    {
        "surface": "Hugging Face Papers",
        "queries": [
            "daily_papers 2026-07-16",
            "daily_papers 2026-07-17",
            "daily_papers 2026-07-18",
            "daily_papers 2026-07-19",
        ],
    },
    {
        "surface": "GitHub discovery",
        "queries": [
            'repo search "energy-based" reasoning constraint created:>2026-07-15',
            "repo search KAN constraint learning created:>2026-07-15",
            "repo search constrained generation verifier created:>2026-07-15",
            "repo search Ising sampler created:>2026-07-15",
            "relai-ai/Continual-Learning-Terminal-Bench",
            "eth-sri/generative-compilation",
            "jinyangwu/SEED",
            "LucasBergholdt/EnergySociety",
        ],
    },
    {"surface": "Extropic writing", "queries": ["TSU", "XTR-0", "X0", "Z1"]},
    {
        "surface": "Logical Intelligence public pages",
        "queries": ["Kona", "Aleph", "Energy-Based Models", "deterministic verifiable"],
    },
    {
        "surface": "local Carnot ledgers",
        "queries": [
            "research-references.md after V511 marker",
            "research-complete.yaml",
            "research-roadmap-next.yaml",
            "openspec/change-proposals/research-roadmap-vNEXT.md",
            "ops/exclusion_manifest.yaml",
            "ops/known-issues.md",
        ],
    },
)

SOURCES_CHECKED: tuple[JsonDict, ...] = (
    {
        "surface": "arXiv",
        "status": "checked_export_api_and_primary_abs_pages",
        "decision": (
            "accepted one bounded continual-learning regression-control source; "
            "other post-marker hits were duplicate, watch-only, domain-mismatched, "
            "fine-tuning, constrained-generation, or non-local hardware scopes"
        ),
    },
    {
        "surface": "OpenReview",
        "status": "public pages challenged by browser verification; API returned 403",
        "decision": "no OpenReview-only post-marker dependency promoted",
    },
    {
        "surface": "Semantic Scholar",
        "status": "graph_api_http_200_for_EBT_and_ARM_EBM_citation_routes",
        "decision": "no citation after the V511 marker changed the task graph",
    },
    {
        "surface": "Hugging Face Papers",
        "status": "daily_papers_api_checked_for_2026_07_16_through_2026_07_19",
        "decision": "used as secondary discovery for accepted, watch-only, and excluded arXiv/GitHub hits",
    },
    {
        "surface": "GitHub discovery",
        "status": "repository_search_and_direct_repository_api_checked",
        "decision": "one accepted support repository maps to the accepted FR-11 source; others are excluded or watch-only",
    },
    {
        "surface": "Extropic writing",
        "status": "http_200_writing_index_checked",
        "decision": "public TSU writing remains watch-only with no authenticated local Carnot execution route",
    },
    {
        "surface": "Logical Intelligence public pages",
        "status": "http_200_kona_page_checked",
        "decision": "Kona remains proprietary context without local weights or reproducible receipts",
    },
    {
        "surface": "local Carnot ledgers",
        "status": "checked",
        "decision": "V511 already indexed EG-VAR, MemOps, Compliance Trap, SLEUTH, MaxSAT feedback, continual weight-write limits, e-CUSUM, ERM, TSU, and Kona boundaries",
    },
)

SOURCE_LINK_CHECKS: tuple[JsonDict, ...] = (
    {
        "source_id": "urlcheck_do_agent_optimizers_compound_2607_14004",
        "linked_source_id": "do_agent_optimizers_compound_2607_14004",
        "url": "https://arxiv.org/abs/2607.14004",
        "status": "primary_arxiv_opened_accepted_non_duplicate",
    },
    {
        "source_id": "relai_continual_learning_terminal_bench_repo",
        "url": "https://github.com/relai-ai/Continual-Learning-Terminal-Bench",
        "status": "github_api_http_200_support_repository_checked",
    },
    {
        "source_id": "urlcheck_byte_exact_kv_grafting_2607_14431",
        "linked_source_id": "byte_exact_kv_grafting_2607_14431",
        "url": "https://arxiv.org/abs/2607.14431",
        "status": "primary_arxiv_opened_watch_only_proprietary_engine",
    },
    {
        "source_id": "urlcheck_generative_compilation_2607_13921",
        "linked_source_id": "generative_compilation_2607_13921",
        "url": "https://arxiv.org/abs/2607.13921",
        "status": "primary_arxiv_opened_excluded_scope_expanding_code_generation",
    },
    {
        "source_id": "generative_compilation_repo",
        "url": "https://github.com/eth-sri/generative-compilation",
        "status": "github_api_http_200_excluded_scope_expanding_code_generation",
    },
    {
        "source_id": "semantic_scholar_ebt_route",
        "url": "https://api.semanticscholar.org/graph/v1/paper/arXiv:2507.02092/citations",
        "status": "http_200_no_post_marker_citation_delta",
    },
    {
        "source_id": "semantic_scholar_arm_ebm_route",
        "url": "https://api.semanticscholar.org/graph/v1/paper/arXiv:2512.15605/citations",
        "status": "http_200_no_post_marker_citation_delta",
    },
    {
        "source_id": "huggingface_daily_2026_07_16",
        "url": "https://huggingface.co/api/daily_papers?date=2026-07-16",
        "status": "http_200_secondary_feed_checked",
    },
    {
        "source_id": "huggingface_daily_2026_07_17",
        "url": "https://huggingface.co/api/daily_papers?date=2026-07-17",
        "status": "http_200_secondary_feed_checked",
    },
    {
        "source_id": "github_recent_energy_constraint_query",
        "url": "https://api.github.com/search/repositories?q=%22energy-based%22+reasoning+constraint+created:%3E2026-07-15",
        "status": "http_200_total_count_0",
    },
    {
        "source_id": "github_recent_constrained_generation_query",
        "url": "https://api.github.com/search/repositories?q=constrained+generation+verifier+created:%3E2026-07-15",
        "status": "http_200_one_domain_mismatched_uart_repo",
    },
    {
        "source_id": "extropic_writing",
        "url": "https://extropic.ai/writing",
        "status": "http_200_watch_only_no_local_tsu",
    },
    {
        "source_id": "logical_intelligence_kona",
        "url": "https://logicalintelligence.com/kona-ebms-energy-based-models",
        "status": "http_200_watch_only_no_local_weights_or_receipts",
    },
)

ACCEPTED_FINDINGS: tuple[JsonDict, ...] = (
    {
        "source_id": "do_agent_optimizers_compound_2607_14004",
        "title": "Do Agent Optimizers Compound? A Continual-Learning Evaluation on Terminal-Bench 2.0",
        "url": "https://arxiv.org/abs/2607.14004",
        "timestamp_utc": "2026-07-15T16:36:04Z",
        "dedupe_status": "not_present_in_local_ledgers_before_execution_refresh",
        "carnot_hook": (
            "Use a phased regression-control check for FR-11 sidecar updates: "
            "each second-phase accepted update must improve the new exact-label "
            "suffix while retaining the prior solved prefix within margin."
        ),
        "target_experiments": [
            "exp5721-fr11-memops-lifecycle-shadow-stream",
            "exp5722-fr11-compliance-recovery-rollback-canary",
        ],
        "substrate": "CPU/RAM exact-label FR-11 sidecar and KAN lifecycle replay",
        "validator_boundary": (
            "exact-label row validators, pre/post state hashes, held-out prefix "
            "retention checks, and exact rollback receipts remain authoritative"
        ),
        "falsifiable_metric": (
            "phase-2 accepted-update suffix improvement must be positive while "
            "old-prefix retention_delta stays within the preregistered margin, "
            "unsafe_update_count remains 0, and rollback restores prior state hashes"
        ),
        "roadmap_change_required_if_pursued": False,
    },
)

DUPLICATE_FINDINGS: tuple[JsonDict, ...] = (
    {
        "source_id": "eg_var_2607_12650",
        "title": "Evidence-Grounded Verified Agentic Reasoning",
        "url": "https://arxiv.org/abs/2607.12650",
        "reason": "Already indexed in the V511 planner block for Exp5720 attested exact envelopes.",
    },
    {
        "source_id": "memops_2607_12893",
        "title": "MemOps: Benchmarking Lifecycle Memory Operations in Long-Horizon Conversations",
        "url": "https://arxiv.org/abs/2607.12893",
        "reason": "Already indexed in the V511 planner block for Exp5721 lifecycle operations.",
    },
    {
        "source_id": "compliance_trap_2607_10608",
        "title": "The Compliance Trap: Diagnosing How AI Agents Consume Conflicting Memory",
        "url": "https://arxiv.org/abs/2607.10608",
        "reason": "Already indexed in the V511 planner block for Exp5722 entry/propagation/recovery.",
    },
    {
        "source_id": "sleuth_2607_12267",
        "title": "Track, Rank, Crack: Epistemic Working Memory Scales Multi-Hop Reasoning in Language Agents",
        "url": "https://arxiv.org/abs/2607.12267",
        "reason": "Already indexed in the V511 planner block for Exp5725/Exp5726 ARC epistemic ledger work.",
    },
    {
        "source_id": "maxsat_feedback_2607_12711",
        "title": "MaxSAT-Based Feedback for Guiding Vision-Language Models in Sudoku",
        "url": "https://arxiv.org/abs/2607.12711",
        "reason": "Already indexed in the V511 planner block for bounded exact-envelope conflict sets.",
    },
    {
        "source_id": "semantic_scholar_ebt_2607_11555",
        "title": "Advancing Optimal Subset Oracle via Learning Relaxation of Neural Set Functions",
        "url": "https://arxiv.org/abs/2607.11555",
        "reason": "Latest EBT citation was already disposed by the V511 planner and predates the marker.",
    },
)

WATCH_ONLY_FINDINGS: tuple[JsonDict, ...] = (
    {
        "source_id": "byte_exact_kv_grafting_2607_14431",
        "title": "Smarter and Cheaper at Once: Byte-Exact KV-Cache Grafting Turns a Frozen Small Model into a Verified-Knowledge Flywheel",
        "url": "https://arxiv.org/abs/2607.14431",
        "classification": "watch_only_proprietary_runtime",
        "reason": (
            "Fresh and relevant to frozen-weight knowledge state, but the engine is "
            "proprietary and V511 has no local KV-graft API or validator boundary."
        ),
    },
    {
        "source_id": "photonic_ising_2607_13446",
        "title": "Photonic Ising machines toward and beyond a million spins",
        "url": "https://arxiv.org/abs/2607.13446",
        "classification": "watch_only_non_local_hardware",
        "reason": "Hardware roadmap context only; no local photonic Ising execution path or speedup evidence exists.",
    },
    {
        "source_id": "kan_mlp_structured_benchmark_2607_13413",
        "title": "Is the Statistical Advantage Worth the Cost? An Empirical Comparison of KANs and MLPs for Structured Data Classification",
        "url": "https://arxiv.org/abs/2607.13413",
        "classification": "watch_only_generic_kan_benchmark",
        "reason": "Generic tabular classification cost-benefit result does not change the exact FR-11 sidecar gate.",
    },
    {
        "source_id": "energy_society_2607_14865",
        "title": "The Energy Society: A Simulation Environment for Studying Agent Cooperation under Survival Pressure",
        "url": "https://arxiv.org/abs/2607.14865",
        "classification": "watch_only_domain_mismatch",
        "reason": "Agent token-cost economy is not Carnot's constraint energy and does not map to Exp5719-Exp5727.",
    },
    {
        "source_id": "openreview_post_marker_candidates",
        "title": "OpenReview search results for ISM, projected constrained diffusion, and chance-constrained flow matching",
        "url": "https://openreview.net/",
        "classification": "watch_only_access_challenged",
        "reason": "Browser verification and API 403 prevent primary-page promotion; snippets do not justify a new V511 dependency.",
    },
)

EXCLUDED_FINDINGS: tuple[JsonDict, ...] = (
    {
        "source_id": "generative_compilation_2607_13921",
        "title": "Generative Compilation: On-the-Fly Compiler Feedback as AI Generates Code",
        "url": "https://arxiv.org/abs/2607.13921",
        "reason": "Useful code-generation verifier work, but accepting it would add a code-generation scope outside Exp5719-Exp5727.",
    },
    {
        "source_id": "seed_self_evolving_distillation_2607_14777",
        "title": "SEED: Self-Evolving On-Policy Distillation for Agentic Reinforcement Learning",
        "url": "https://arxiv.org/abs/2607.14777",
        "reason": "On-policy distillation and policy updates reopen broad RL/fine-tuning rather than the immutable-GGUF FR-11 sidecar.",
    },
    {
        "source_id": "longstraw_2607_14952",
        "title": "LongStraw: Long-Context RL Beyond 2M Tokens under a Fixed GPU Budget",
        "url": "https://arxiv.org/abs/2607.14952",
        "reason": "Long-context RL post-training is outside the no-weight-write V511 experiment graph.",
    },
    {
        "source_id": "alpha_wise_2607_15094",
        "title": "AlphaWiSE: Adaptive Weight Interpolation for Continual Multimodal Representation Learning",
        "url": "https://arxiv.org/abs/2607.15094",
        "reason": "Materializes interpolated checkpoints and therefore reopens model-weight-write scope.",
    },
    {
        "source_id": "gate_zero_growth_2607_14571",
        "title": "Gate-Zero Growth: A Geometric Framework for Function-Preserving Continual Learning",
        "url": "https://arxiv.org/abs/2607.14571",
        "reason": "Capacity growth changes model parameters and does not fit immutable GGUF FR-11.",
    },
    {
        "source_id": "groc_po_2607_13712",
        "title": "Groc-PO: Grounded Context Preference Optimization for Truthful Multimodal LLMs",
        "url": "https://arxiv.org/abs/2607.13712",
        "reason": "Preference optimization and multimodal hallucination mitigation require training and external scoring scopes.",
    },
    {
        "source_id": "conflow_2607_14424",
        "title": "ConFlow: Constraints-Guided Learning with Flow Matching for Motion Generation",
        "url": "https://arxiv.org/abs/2607.14424",
        "reason": "Training-time constrained motion generation is a diffusion scope, not a V511 exact-validator hook.",
    },
    {
        "source_id": "rollout_constrained_diffusion_2607_14398",
        "title": "Integration Matters: Rollout-Based Training for Constrained Diffusion Models",
        "url": "https://arxiv.org/abs/2607.14398",
        "reason": "Fine-tuning constrained diffusion models reopens training-time generation outside V511.",
    },
    {
        "source_id": "lyaguide_2607_14272",
        "title": "Lyapunov Guidance: A Unified Framework for Stabilizing Generative Flows",
        "url": "https://arxiv.org/abs/2607.14272",
        "reason": "Generative-flow guidance is outside the exact envelope, FR-11, sampler, and ARC tasks.",
    },
    {
        "source_id": "native_json_grammar_runtime",
        "title": "Native JSON grammar or three-model runtime restart",
        "reason": "Explicitly rejected by the operator scope and retired V511 boundaries.",
    },
    {
        "source_id": "external_generated_text_scoring",
        "title": "External generated-text scoring or LLM judge authority",
        "reason": "External text scoring cannot become the authority for V511 exact envelopes.",
    },
    {
        "source_id": "token_logit_authority",
        "title": "Token/logit authority refresh",
        "reason": "V511 preserves exact-validator authority and rejects token/logit confidence as a release gate.",
    },
    {
        "source_id": "non_local_tsu_kona_execution",
        "title": "Non-local TSU or Kona execution claims",
        "reason": "Neither Extropic TSU nor Kona exposes an authenticated local execution route.",
    },
)


def _clone_json(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def payload_checksum(payload: Mapping[str, Any]) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def path_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_text_if_present(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _planner_marker_found(references_text: str) -> bool:
    compact_text = references_text.replace("-", "")
    return PLANNER_HEADING in references_text or PLANNER_HEADING_COMPACT in compact_text


def _planner_marker_line(references_text: str) -> int | None:
    index = references_text.find(PLANNER_HEADING)
    if index < 0:
        return None
    return references_text[:index].count("\n") + 1


def _proposal_paths(root: Path) -> list[Path]:
    proposal_dir = root / "openspec/change-proposals"
    if not proposal_dir.exists():
        return []
    return sorted(proposal_dir.glob("*.md"))


def _dedupe_paths(root: Path) -> list[Path]:
    paths = [
        root / RESEARCH_REFERENCES_RELATIVE_PATH,
        root / RESEARCH_COMPLETE_RELATIVE_PATH,
        root / ROADMAP_NEXT_RELATIVE_PATH,
        root / VNEXT_RELATIVE_PATH,
        root / EXCLUSION_MANIFEST_RELATIVE_PATH,
        root / KNOWN_ISSUES_RELATIVE_PATH,
        root / CONDUCTOR_RELATIVE_PATH,
    ]
    paths.extend(_proposal_paths(root))
    return list(dict.fromkeys(paths))


def _relative_path(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _dedupe_corpus(root: Path) -> list[JsonDict]:
    checked: list[JsonDict] = []
    for path in _dedupe_paths(root):
        exists = path.exists()
        checked.append(
            {
                "path": _relative_path(root, path),
                "exists": exists,
                "sha256": path_sha256(path) if exists else None,
            }
        )
    return checked


def _roadmap_context(root: Path) -> JsonDict:
    relative = (
        ROADMAP_NEXT_RELATIVE_PATH
        if (root / ROADMAP_NEXT_RELATIVE_PATH).exists()
        else ROADMAP_RELATIVE_PATH
    )
    parsed = yaml.safe_load(_read_text_if_present(root / relative)) or {}
    tasks = parsed.get("tasks", []) if isinstance(parsed, Mapping) else []
    task_ids = [
        str(task.get("id")) for task in tasks if isinstance(task, Mapping) and task.get("id")
    ]
    milestone = str(parsed.get("milestone", "")) if isinstance(parsed, Mapping) else ""
    return {"source": relative.as_posix(), "milestone": milestone, "task_ids": task_ids}


def _normalize_timestamp(search_timestamp_utc: str | None) -> str:
    timestamp = search_timestamp_utc or datetime.now(UTC).replace(microsecond=0).isoformat()
    if timestamp.endswith("+00:00"):
        timestamp = timestamp[:-6] + "Z"
    return timestamp


def _closed_scope_review() -> JsonDict:
    return {
        "json_grammar_reopened": False,
        "external_generated_text_scoring_reopened": False,
        "token_or_logit_authority_reopened": False,
        "model_weight_writes_reopened": False,
        "ptrm_generation_reopened": False,
        "generic_exploration_signals_reopened": False,
        "transition_patching_reopened": False,
        "two_axis_exchange_reopened": False,
        "non_local_tsu_or_kona_execution_reopened": False,
        "unsupported_speedups_reopened": False,
        "operator_authorized_scope_expansion": None,
    }


def _semantic_scholar_status() -> JsonDict:
    return {
        "route": "Semantic Scholar Graph API",
        "papers": ["arXiv:2507.02092", "arXiv:2512.15605"],
        "http_status": 200,
        "access": "checked",
        "latest_ebt_citation_publication_date": "2026-07-13",
        "latest_arm_ebm_citation_publication_date": "2026-07-02",
        "post_marker_citation_count": 0,
        "honest_status": "direct citation routes were reachable and exposed no citation after the V511 marker",
        "roadmap_delta": False,
    }


def _extropic_status() -> JsonDict:
    return {
        "route": "https://extropic.ai/writing",
        "http_status": 200,
        "latest_visible_post_date": "2025-10-29",
        "latest_visible_post": "TSU 101",
        "local_execution_available": False,
        "honest_status": "public writing reachable; no authenticated Carnot TSU path",
        "roadmap_delta": False,
    }


def _logical_intelligence_status() -> JsonDict:
    return {
        "route": "https://logicalintelligence.com/kona-ebms-energy-based-models",
        "http_status": 200,
        "last_modified": "2026-06-26T23:48:05Z",
        "local_execution_available": False,
        "honest_status": "public Kona page reachable; no local weights, API receipts, or reproducible comparator",
        "roadmap_delta": False,
    }


def _huggingface_status() -> JsonDict:
    return {
        "route": "https://huggingface.co/api/daily_papers",
        "http_status": 200,
        "dates_checked": ["2026-07-16", "2026-07-17", "2026-07-18", "2026-07-19"],
        "post_marker_relevant_ids": [
            "2607.14004",
            "2607.14431",
            "2607.13921",
            "2607.14777",
            "2607.14952",
        ],
        "honest_status": "secondary feed checked; only arXiv:2607.14004 became an accepted local hook",
        "roadmap_delta": True,
    }


def _github_status() -> JsonDict:
    return {
        "route": "GitHub repository search and direct repository API",
        "http_status": 200,
        "recent_repository_searches": {
            "energy_based_reasoning_constraint_created_after_2026_07_15": 0,
            "kan_constraint_learning_created_after_2026_07_15": 0,
            "constrained_generation_verifier_created_after_2026_07_15": 1,
            "ising_sampler_created_after_2026_07_15": 0,
        },
        "direct_repositories_checked": 4,
        "accepted_support_repository": "https://github.com/relai-ai/Continual-Learning-Terminal-Bench",
        "excluded_or_watch_only_repositories": [
            "https://github.com/eth-sri/generative-compilation",
            "https://github.com/jinyangwu/SEED",
            "https://github.com/LucasBergholdt/EnergySociety",
        ],
        "honest_status": "repository route supports the accepted FR-11 source and supplies no sampler/KAN replacement",
        "roadmap_delta": True,
    }


def _accepted_findings() -> list[JsonDict]:
    return _clone_json(ACCEPTED_FINDINGS)


def _target_experiment_map(accepted_findings: list[JsonDict]) -> list[JsonDict]:
    return [
        {
            "source_id": finding["source_id"],
            "target_experiments": list(finding["target_experiments"]),
            "carnot_hook": finding["carnot_hook"],
            "substrate": finding["substrate"],
            "validator_boundary": finding["validator_boundary"],
            "falsifiable_metric": finding["falsifiable_metric"],
        }
        for finding in accepted_findings
    ]


def _honest_verdict(planner_marker_found: bool, accepted_findings: list[JsonDict]) -> str:
    if not planner_marker_found:
        return "blocked: V511 planner refresh marker missing; source-delta append refused"
    if not accepted_findings:
        return "complete: no new non-duplicate actionable V511 source deltas; references left unchanged"
    return (
        f"complete: accepted {len(accepted_findings)} non-duplicate actionable V511 "
        "source delta; no roadmap ID or gate change"
    )


def _execution_refresh_block(accepted_findings: list[JsonDict]) -> str:
    lines = [
        "",
        EXECUTION_REFRESH_HEADING,
        "",
        (
            "Execution-time sweep on 2026-07-19 after the V511 planner marker. "
            "Only sources with a non-duplicate, existing-task Carnot hook are "
            "listed here; watch-only and excluded sources are recorded in "
            "`results/experiment_5718_v511_source_delta_ingestion.json`."
        ),
        "",
        "### New actionable deltas",
        "",
    ]
    for finding in accepted_findings:
        targets = ", ".join(finding["target_experiments"])
        lines.extend(
            [
                (
                    f"- **{finding['title']}** - arXiv:2607.14004, "
                    f"{finding['url']}. Carnot hook: {finding['carnot_hook']} "
                    f"Target: {targets}. Substrate: {finding['substrate']}. "
                    f"Validator boundary: {finding['validator_boundary']}. "
                    f"Falsifiable metric: {finding['falsifiable_metric']} "
                    "This sharpens the existing FR-11 retention/rollback gate and "
                    "does not authorize model-weight writes, broad RL, or roadmap "
                    "gate changes."
                )
            ]
        )
    lines.extend(
        [
            "",
            "### V511 execution impact",
            "",
            (
                "- Preserve the existing Exp5721/Exp5722 lifecycle and rollback "
                "shape. The accepted delta only makes the regression-control "
                "metric explicit: new-suffix utility cannot count if old-prefix "
                "behavior regresses or rollback fails to restore the exact prior "
                "state."
            ),
            "",
            "<!-- V511-EXECUTION-REFRESH-20260719-END -->",
        ]
    )
    return "\n".join(lines) + "\n"


def _insert_after_planner_block(references_text: str, block: str) -> str:
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


def _append_execution_refresh_if_needed(
    root: Path, planner_marker_found: bool, accepted_findings: list[JsonDict]
) -> bool:
    if not planner_marker_found or not accepted_findings:
        return False
    references_path = root / RESEARCH_REFERENCES_RELATIVE_PATH
    references_text = _read_text_if_present(references_path)
    if EXECUTION_REFRESH_HEADING in references_text:
        return False
    updated = _insert_after_planner_block(
        references_text,
        _execution_refresh_block(accepted_findings),
    )
    references_path.write_text(updated, encoding="utf-8")
    return True


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    search_timestamp_utc: str | None = None,
    run_date: str = RUN_DATE,
    duration_s: float = 0.0,
    references_updated: bool | None = None,
    references_mutated_this_run: bool = False,
) -> JsonDict:
    references_text = _read_text_if_present(root / RESEARCH_REFERENCES_RELATIVE_PATH)
    planner_marker_found = _planner_marker_found(references_text)
    accepted_findings = _accepted_findings() if planner_marker_found else []
    if references_updated is None:
        references_updated = bool(planner_marker_found and accepted_findings)
    roadmap_change_required = any(
        bool(finding.get("roadmap_change_required_if_pursued"))
        for finding in accepted_findings
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "status": "blocked" if not planner_marker_found or roadmap_change_required else "complete",
        "milestone": MILESTONE,
        "run_date": run_date,
        "search_cutoff": SEARCH_CUTOFF,
        "search_timestamp_utc": _normalize_timestamp(search_timestamp_utc),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "spec_refs": list(SPEC_REFS),
        "planner_marker": PLANNER_MARKER,
        "planner_marker_found": planner_marker_found,
        "sources_checked": _clone_json(SOURCES_CHECKED),
        "queries": _clone_json(QUERIES),
        "source_link_checks": _clone_json(SOURCE_LINK_CHECKS),
        "dedupe_corpus_checked": _dedupe_corpus(root),
        "marker_checks": {
            "planner_marker": PLANNER_MARKER,
            "planner_heading": PLANNER_HEADING,
            "planner_marker_found": planner_marker_found,
            "planner_marker_line": _planner_marker_line(references_text),
            "search_window": "strictly_after_V511_PLANNER_REFRESH_20260715_END",
            "execution_refresh_heading": EXECUTION_REFRESH_HEADING,
            "execution_refresh_present": EXECUTION_REFRESH_HEADING in references_text,
        },
        "duplicate_checks": {
            "accepted_count": len(accepted_findings),
            "duplicate_count": len(DUPLICATE_FINDINGS),
            "watch_only_count": len(WATCH_ONLY_FINDINGS),
            "excluded_count": len(EXCLUDED_FINDINGS),
        },
        "accepted_findings": accepted_findings,
        "duplicate_findings": _clone_json(DUPLICATE_FINDINGS),
        "watch_only_findings": _clone_json(WATCH_ONLY_FINDINGS),
        "excluded_findings": _clone_json(EXCLUDED_FINDINGS),
        "semantic_scholar_status": _semantic_scholar_status(),
        "extropic_status": _extropic_status(),
        "logical_intelligence_status": _logical_intelligence_status(),
        "huggingface_status": _huggingface_status(),
        "github_status": _github_status(),
        "target_experiment_map": _target_experiment_map(accepted_findings),
        "roadmap_change_required": roadmap_change_required,
        "references_updated": references_updated,
        "references_mutated_this_run": references_mutated_this_run,
        "closed_scope_review": _closed_scope_review(),
        "roadmap_context": _roadmap_context(root),
        "field_principles": dict(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "random_seed": RANDOM_SEED,
        "honest_verdict": _honest_verdict(planner_marker_found, accepted_findings),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    _require(not missing, f"missing required fields: {missing}")
    _require(isinstance(artifact["field_principles"], Mapping), "field_principles mapping")
    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact["field_principles"], f"field_principles missing {field}")
        _require(str(artifact["field_principles"][field]).strip(), f"empty principle for {field}")
    _require(artifact["planner_marker"] == PLANNER_MARKER, "planner_marker mismatch")
    _require(isinstance(artifact["references_updated"], bool), "references_updated must be bool")
    _require(
        isinstance(artifact["roadmap_change_required"], bool),
        "roadmap_change_required must be bool",
    )
    if artifact["roadmap_change_required"]:
        _require(str(artifact["honest_verdict"]).startswith("blocked:"), "scope expansion must block")
    _require(artifact["inference_substrate"] == INFERENCE_SUBSTRATE, "inference_substrate mismatch")
    _require(str(artifact["search_timestamp_utc"]).endswith("Z"), "timestamp must end in Z")
    _require(
        str(artifact["honest_verdict"]).startswith(TERMINAL_PREFIXES),
        "honest_verdict must use terminal prefix",
    )
    for field in (
        "sources_checked",
        "queries",
        "accepted_findings",
        "duplicate_findings",
        "watch_only_findings",
        "excluded_findings",
        "target_experiment_map",
    ):
        _require(_is_sequence(artifact[field]), f"{field} must be a list")
    for field in (
        "semantic_scholar_status",
        "extropic_status",
        "logical_intelligence_status",
        "huggingface_status",
        "github_status",
    ):
        _require(isinstance(artifact[field], Mapping), f"{field} must be a mapping")
    required_finding_fields = {
        "source_id",
        "title",
        "url",
        "carnot_hook",
        "target_experiments",
        "substrate",
        "validator_boundary",
        "falsifiable_metric",
    }
    for finding in artifact["accepted_findings"]:
        missing_finding_fields = sorted(required_finding_fields - set(finding))
        _require(not missing_finding_fields, f"accepted finding missing {missing_finding_fields}")
        _require(_is_sequence(finding["target_experiments"]), "target_experiments must be a list")
        for target in finding["target_experiments"]:
            _require(str(target) in ALLOWED_TARGET_EXPERIMENTS, "accepted target experiment outside Exp5719-Exp5727")
    for row in artifact["target_experiment_map"]:
        _require(_is_sequence(row.get("target_experiments")), "target_experiments must be a list")
        for target in row["target_experiments"]:
            _require(str(target) in ALLOWED_TARGET_EXPERIMENTS, "target experiment outside Exp5719-Exp5727")
    _require(artifact["reproducibility_checksum"] == payload_checksum(artifact), "checksum mismatch")


def build_and_write_artifact(
    *,
    root: Path = REPO_ROOT,
    search_timestamp_utc: str | None = None,
    run_date: str = RUN_DATE,
    duration_s: float = 0.0,
) -> JsonDict:
    references_text = _read_text_if_present(root / RESEARCH_REFERENCES_RELATIVE_PATH)
    planner_marker_found = _planner_marker_found(references_text)
    accepted_findings = _accepted_findings() if planner_marker_found else []
    references_mutated_this_run = _append_execution_refresh_if_needed(
        root,
        planner_marker_found,
        accepted_findings,
    )
    references_updated = bool(planner_marker_found and accepted_findings)
    artifact = build_artifact(
        root=root,
        search_timestamp_utc=search_timestamp_utc,
        run_date=run_date,
        duration_s=duration_s,
        references_updated=references_updated,
        references_mutated_this_run=references_mutated_this_run,
    )
    validate_artifact(artifact)
    write_json(root / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--search-timestamp-utc", default=None)
    args = parser.parse_args(argv)
    start = time.perf_counter()
    artifact = build_and_write_artifact(
        root=args.root,
        search_timestamp_utc=args.search_timestamp_utc,
        duration_s=time.perf_counter() - start,
    )
    print(f"wrote {args.root / RESULT_RELATIVE_PATH}: {artifact['honest_verdict']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
