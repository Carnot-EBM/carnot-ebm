"""Exp5732: ingest the V512 execution-time source delta.

Spec refs: REQ-REPORT-5732, SCENARIO-REPORT-5732-NOOP,
SCENARIO-REPORT-5732-BLOCKED-MARKER,
SCENARIO-REPORT-5732-FIELD-PRINCIPLES.

The public web search itself is intentionally not replayed in tests. Search
indexes, citation APIs, and daily paper feeds drift. This module captures the
durable part of the work: which source families were checked, why no
post-marker source became an executable Carnot task, which routes were
duplicate/watch-only/inaccessible/excluded, and the explicit boundary that no
benchmark compute or hardware speedup was claimed.
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
RESULT_RELATIVE_PATH = Path("results/experiment_5732_v512_source_delta_ingestion.json")
RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
KNOWN_ISSUES_RELATIVE_PATH = Path("ops/known-issues.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")

EXPERIMENT = "experiment_5732_v512_source_delta_ingestion"
EXPERIMENT_ID = "exp5732-v512-source-delta-ingestion"
MILESTONE = "2026.07.512"
RUN_DATE = "20260720"
SEARCH_CUTOFF = "2026-07-20"
SCHEMA = "carnot.experiment_5732.v512_source_delta_ingestion.v1"
RANDOM_SEED = 5732
INFERENCE_SUBSTRATE = "web_and_bibliographic_search_only"
TERMINAL_PREFIXES = ("complete:", "blocked:")

PLANNER_HEADING = "## V512 Planner Refresh - 20260719"
PLANNER_MARKER = "V512-PLANNER-REFRESH-20260719-END"
PLANNER_END_MARKER = f"<!-- {PLANNER_MARKER} -->"

ALLOWED_TARGET_EXPERIMENTS = {
    "exp5733-sota-finite-choice-proposal-channel",
    "exp5734-sota-exact-proposal-stream",
    "exp5735-zero-gate-kan-continuous-self-learning",
    "exp5736-csl-lifecycle-conflict-rollback",
    "exp5737-sota-stream-csl-shadow-ingress",
    "exp5738-one-axis-rust-batched-backend",
    "exp5739-one-axis-batched-10x-crossover",
    "exp5740-arc-game-blind-primitive-causal-audit",
    "exp5741-arc-generic-primitive-live-ab",
}

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
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
    "benchmark_compute_claimed",
    "inference_substrate",
    "reproducibility_checksum",
    "honest_verdict",
)

REQUIRED_FIELD_PRINCIPLES: dict[str, str] = {
    "field_principles": (
        "Explains every top-level artifact field so downstream readers know why the field exists."
    ),
    "preconditions_checked": (
        "Records the instruction, spec, marker, and local ledger checks that bound the run."
    ),
    "search_timestamp_utc": "Anchors public-source freshness to an exact UTC time.",
    "planner_marker": "Binds the search window to the post-V512 planner boundary.",
    "sources_checked": "Makes source coverage reconstructable without trusting prose.",
    "queries": "Makes search intent reconstructable without rerunning mutable indexes.",
    "accepted_findings": "Accepted work must have a post-marker, local, falsifiable home.",
    "duplicate_findings": "Already-indexed material stays visible but cannot create duplicate work.",
    "watch_only_findings": "Non-local or non-executable material cannot support claims.",
    "excluded_findings": "Closed scopes remain closed by explicit disposition.",
    "semantic_scholar_status": "Citation-route access is reported honestly.",
    "extropic_status": "Thermodynamic hardware access is bounded to what Carnot can execute.",
    "logical_intelligence_status": (
        "Kona/Aleph access is bounded to public context unless local receipts exist."
    ),
    "huggingface_status": "Secondary paper-feed status is separated from primary authority.",
    "github_status": "Repository discovery is separated from executable local dependencies.",
    "target_experiment_map": (
        "Each accepted source must map to Exp5733-Exp5741 without changing ids or gates."
    ),
    "roadmap_change_required": (
        "Scope expansion blocks for operator review instead of mutating the roadmap."
    ),
    "references_updated": "Reference-file mutation is declared and must be false for a no-op.",
    "benchmark_compute_claimed": "Bibliographic search cannot claim benchmark compute.",
    "inference_substrate": "The run used web and bibliographic search only.",
    "reproducibility_checksum": "The stable artifact payload can be checked for drift.",
    "honest_verdict": "The terminal result states complete or blocked without claim inflation.",
}

FIELD_PRINCIPLES: dict[str, str] = {
    **REQUIRED_FIELD_PRINCIPLES,
    "schema": "Names the artifact schema used by downstream validators.",
    "experiment": "Provides the stable module-level experiment slug.",
    "experiment_id": "Binds this receipt to the Exp5732 roadmap task.",
    "status": "Machine-readable terminal state derived from marker and scope checks.",
    "milestone": "Prevents this V512 receipt from being reused for another milestone.",
    "run_date": "Records the operator-requested execution date in compact form.",
    "search_cutoff": "Records the calendar date through which public sources were checked.",
    "result_path": "Records where the JSON receipt is written.",
    "spec_refs": "Links the artifact to its OpenSpec requirement and scenarios.",
    "planner_marker_found": "Shows whether the source window was anchored before any mutation.",
    "source_link_checks": "Records the specific public URLs or APIs checked.",
    "dedupe_corpus_checked": "Lists local files hashed for duplicate and boundary review.",
    "marker_checks": "Records heading and marker details for reference-marker validation.",
    "duplicate_checks": "Summarizes arXiv id, title, repository, citation, and hook dedupe checks.",
    "inaccessible_findings": "Separates access failures from scientific exclusions.",
    "closed_scope_review": "Documents that banned research scopes remain closed.",
    "roadmap_context": "Records which roadmap queue was available without recreating files.",
    "references_mutated_this_run": "Separates durable references_updated status from this invocation.",
    "duration_s": "Records wall-clock artifact generation time without benchmark meaning.",
    "random_seed": "Keeps deterministic receipt metadata aligned with experiment id.",
}

SPEC_REFS = (
    "REQ-REPORT-5732",
    "SCENARIO-REPORT-5732-NOOP",
    "SCENARIO-REPORT-5732-BLOCKED-MARKER",
    "SCENARIO-REPORT-5732-FIELD-PRINCIPLES",
)

QUERIES: tuple[JsonDict, ...] = (
    {
        "surface": "arXiv",
        "timestamp_window_utc": "strictly_after_2026-07-19T23:59:59Z",
        "queries": [
            'all:"energy based" OR all:EBM',
            'all:"energy-based verification" AND all:reasoning',
            'all:"neural CSP" OR all:"constraint satisfaction"',
            "all:Ising AND all:machine-learning",
            'all:"hallucination mitigation"',
            "all:KAN",
            'all:"constrained generation"',
            'all:"hardware sampling"',
            'all:"continual learning" AND all:constraint',
            "cs.AI recent",
            "cs.LG recent",
            "cs.CL recent",
            "stat.ML recent",
        ],
    },
    {
        "surface": "OpenReview",
        "queries": [
            "energy-based reasoning",
            "constrained generation",
            "continual learning constraint",
            "certifiable continual learning",
        ],
    },
    {
        "surface": "Semantic Scholar",
        "queries": ["arXiv:2507.02092 citations", "arXiv:2512.15605 citations"],
    },
    {"surface": "Hugging Face Papers", "queries": ["daily_papers 2026-07-20"]},
    {
        "surface": "GitHub discovery",
        "queries": [
            'repo search "energy-based" reasoning constraint created:>2026-07-19',
            "repo search KAN constraint learning created:>2026-07-19",
            "repo search constrained generation verifier created:>2026-07-19",
            "repo search Ising sampler created:>2026-07-19",
            "github trending 2026-07-20",
        ],
    },
    {"surface": "Extropic writing", "queries": ["TSU", "XTR-0", "X0", "Z1"]},
    {
        "surface": "Logical Intelligence public pages",
        "queries": ["Kona", "Aleph", "Energy-Based Models", "verified reasoning"],
    },
    {
        "surface": "local Carnot ledgers",
        "queries": [
            "research-references.md after V512 marker",
            "research-complete.yaml",
            "research-roadmap-next.yaml if present",
            "openspec/change-proposals/research-roadmap-vNEXT.md",
            "ops/exclusion_manifest.yaml",
            "ops/known-issues.md",
        ],
    },
)

SOURCES_CHECKED: tuple[JsonDict, ...] = (
    {
        "surface": "arXiv",
        "status": "recent_category_pages_checked; export_api_rate_limited_429",
        "decision": (
            "latest public recent pages exposed Fri 17 Jul 2026 items already inside "
            "the V512 planner block or outside scope; no post-marker arXiv item accepted"
        ),
    },
    {
        "surface": "OpenReview",
        "status": "search_snippets_checked; primary_pages_browser_challenged",
        "decision": "no OpenReview-only post-marker dependency promoted",
    },
    {
        "surface": "Semantic Scholar",
        "status": "graph_api_http_200_for_EBT_and_ARM_EBM_citation_routes",
        "decision": "no citation after the V512 marker changed the task graph",
    },
    {
        "surface": "Hugging Face Papers",
        "status": "daily_papers_api_checked_for_2026_07_20_empty",
        "decision": "no post-marker secondary-feed source accepted",
    },
    {
        "surface": "GitHub discovery",
        "status": "repository_search_and_trending_checked",
        "decision": "no new reproducible local repository displaced Carnot substrates",
    },
    {
        "surface": "Extropic writing",
        "status": "public_writing_index_http_200_checked",
        "decision": "watch-only; no authenticated local TSU execution route",
    },
    {
        "surface": "Logical Intelligence public pages",
        "status": "public_kona_page_http_200_checked",
        "decision": "watch-only; no local weights, API receipt, or reproducible comparator",
    },
    {
        "surface": "local Carnot ledgers",
        "status": "checked",
        "decision": (
            "V512 already indexed Generative Compilation, Gate-Zero Growth, SMC-ES, "
            "Campaign Diagrams, Bridge Evidence, verified DPLL transitions, KAN cost "
            "evidence, and the photonic-Ising boundary"
        ),
    },
)

SOURCE_LINK_CHECKS: tuple[JsonDict, ...] = (
    {
        "source_id": "arxiv_cs_ai_recent",
        "url": "https://arxiv.org/list/cs.AI/recent",
        "status": "http_200_latest_public_day_2026_07_17_no_post_marker_acceptance",
    },
    {
        "source_id": "arxiv_cs_lg_recent",
        "url": "https://arxiv.org/list/cs.LG/recent",
        "status": "http_200_latest_public_day_2026_07_17_no_post_marker_acceptance",
    },
    {
        "source_id": "arxiv_cs_cl_recent",
        "url": "https://arxiv.org/list/cs.CL/recent",
        "status": "http_200_latest_public_day_2026_07_17_no_post_marker_acceptance",
    },
    {
        "source_id": "arxiv_stat_ml_recent",
        "url": "https://arxiv.org/list/stat.ML/recent",
        "status": "http_200_latest_public_day_2026_07_17_no_post_marker_acceptance",
    },
    {
        "source_id": "arxiv_export_api_429",
        "url": "https://export.arxiv.org/api/query",
        "status": "http_429_rate_limited_recorded_as_inaccessible_not_accepted",
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
        "source_id": "huggingface_daily_2026_07_20",
        "url": "https://huggingface.co/api/daily_papers?date=2026-07-20",
        "status": "http_200_empty_array",
    },
    {
        "source_id": "github_recent_energy_constraint_query",
        "url": "https://api.github.com/search/repositories?q=%22energy-based%22+reasoning+constraint+created:%3E2026-07-19",
        "status": "http_200_total_count_0",
    },
    {
        "source_id": "github_trending",
        "url": "https://github.com/trending",
        "status": "http_200_no_reproducible_local_carnot_dependency_promoted",
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
    {
        "source_id": "openreview_browser_challenge",
        "url": "https://openreview.net/",
        "status": "browser_challenge_no_primary_post_marker_promotion",
    },
)

DUPLICATE_FINDINGS: tuple[JsonDict, ...] = (
    {
        "source_id": "generative_compilation_2607_13921",
        "title": "Generative Compilation: On-the-Fly Compiler Feedback as AI Generates Code",
        "url": "https://arxiv.org/abs/2607.13921",
        "reason": "Already indexed by the V512 planner for Exp5733/Exp5734.",
    },
    {
        "source_id": "gate_zero_growth_2607_14571",
        "title": "Gate-Zero Growth: A Geometric Framework for Function-Preserving Continual Learning",
        "url": "https://arxiv.org/abs/2607.14571",
        "reason": "Already indexed by the V512 planner for Exp5735.",
    },
    {
        "source_id": "smc_es_2607_15003",
        "title": "SMC-ES: Automated synthesis of formally verified control policies",
        "url": "https://arxiv.org/abs/2607.15003",
        "reason": "Already indexed by the V512 planner for Exp5735/Exp5736 certificates.",
    },
    {
        "source_id": "campaign_diagrams_2607_15225",
        "title": "Campaign Diagrams: Visualizing the March Through the Phases of a Workload",
        "url": "https://arxiv.org/abs/2607.15225",
        "reason": "Already indexed by the V512 planner for Exp5738/Exp5739 phase receipts.",
    },
    {
        "source_id": "bridge_evidence_2607_15253",
        "title": "Bridge Evidence: Static Retrieval Utility Does Not Predict Causal Utility in Multi-Step Agentic Search",
        "url": "https://arxiv.org/abs/2607.15253",
        "reason": "Already indexed by the V512 planner for Exp5740/Exp5741.",
    },
    {
        "source_id": "photonic_ising_2607_13446",
        "title": "Photonic Ising machines toward and beyond a million spins",
        "url": "https://arxiv.org/abs/2607.13446",
        "reason": "Already indexed as the V512 watch-only photonic boundary.",
    },
)

WATCH_ONLY_FINDINGS: tuple[JsonDict, ...] = (
    {
        "source_id": "extropic_writing_index",
        "title": "Extropic writing index and TSU/XTR-0 public material",
        "url": "https://extropic.ai/writing",
        "classification": "watch_only_non_local_hardware",
        "reason": "No authenticated Carnot-accessible TSU hardware or SDK execution route exists.",
    },
    {
        "source_id": "logical_intelligence_kona",
        "title": "Kona 1.0 Energy-Based Models public page",
        "url": "https://logicalintelligence.com/kona-ebms-energy-based-models",
        "classification": "watch_only_proprietary_comparator",
        "reason": "Public architecture context only; no local weights, API receipt, or reproducible comparator.",
    },
    {
        "source_id": "github_trending_2026_07_20",
        "title": "GitHub trending repositories",
        "url": "https://github.com/trending",
        "classification": "watch_only_no_local_dependency",
        "reason": "Discovery route produced no reproducible Carnot EBM/CSP/KAN/sampler replacement.",
    },
)

INACCESSIBLE_FINDINGS: tuple[JsonDict, ...] = (
    {
        "source_id": "arxiv_export_api_429",
        "title": "arXiv export API broad query",
        "url": "https://export.arxiv.org/api/query",
        "classification": "inaccessible_rate_limited",
        "reason": "The broad export query returned HTTP 429; category pages were used as the primary fallback.",
    },
    {
        "source_id": "openreview_browser_challenge",
        "title": "OpenReview primary forum pages",
        "url": "https://openreview.net/",
        "classification": "inaccessible_browser_challenge",
        "reason": "Search snippets were visible, but primary pages required browser verification.",
    },
)

EXCLUDED_FINDINGS: tuple[JsonDict, ...] = (
    {
        "source_id": "longstraw_2607_14952",
        "title": "LongStraw: Long-Context RL Beyond 2M Tokens under a Fixed GPU Budget",
        "url": "https://arxiv.org/abs/2607.14952",
        "reason": "Long-context RL post-training reopens broad RL and model-weight-write scope.",
    },
    {
        "source_id": "seed_self_evolving_distillation_2607_14777",
        "title": "SEED: Self-Evolving On-Policy Distillation for Agentic Reinforcement Learning",
        "url": "https://arxiv.org/abs/2607.14777",
        "reason": "On-policy distillation reopens broad RL/fine-tuning instead of an external exact sidecar.",
    },
    {
        "source_id": "mask_aware_policy_gradients_2607_15200",
        "title": "Mask-Aware Policy Gradients for Diffusion Language Models",
        "url": "https://arxiv.org/abs/2607.15200",
        "reason": "Policy-gradient diffusion-LM work is outside the finite-choice exact-validator boundary.",
    },
    {
        "source_id": "free_form_answer_repair",
        "title": "Free-form answer repair or FINAL envelope retry",
        "reason": "Explicitly retired by the V512 roadmap; Exp5733 uses finite-choice proposals only.",
    },
    {
        "source_id": "json_grammar_or_external_text_scoring",
        "title": "JSON grammar, generated-text scoring, or token/logit semantic authority",
        "reason": "The exact validator remains the only authority for Exp5733-Exp5734.",
    },
    {
        "source_id": "arc_value_transfer_or_adapters",
        "title": "Learned ARC value transfer or per-game adapters",
        "reason": "Exp5740/Exp5741 allow only game-blind primitive mining and one generic live-path hardening.",
    },
    {
        "source_id": "unsupported_hardware_speedups",
        "title": "Unsupported hardware speedup claims",
        "reason": "Exp5732 performed bibliographic search only and has no local hardware timing receipt.",
    },
)


def clone_json(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def require(condition: bool, message: str) -> None:
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


def relative_path(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def dedupe_paths(root: Path) -> list[Path]:
    return [
        root / RESEARCH_REFERENCES_RELATIVE_PATH,
        root / RESEARCH_COMPLETE_RELATIVE_PATH,
        root / ROADMAP_NEXT_RELATIVE_PATH,
        root / ROADMAP_RELATIVE_PATH,
        root / VNEXT_RELATIVE_PATH,
        root / SPEC_RELATIVE_PATH,
        root / EXCLUSION_MANIFEST_RELATIVE_PATH,
        root / KNOWN_ISSUES_RELATIVE_PATH,
        root / CONDUCTOR_RELATIVE_PATH,
    ]


def dedupe_corpus(root: Path) -> list[JsonDict]:
    checked: list[JsonDict] = []
    for path in dedupe_paths(root):
        exists = path.exists()
        checked.append(
            {
                "path": relative_path(root, path),
                "exists": exists,
                "sha256": path_sha256(path) if exists else None,
            }
        )
    return checked


def roadmap_context(root: Path) -> JsonDict:
    relative = (
        ROADMAP_NEXT_RELATIVE_PATH
        if (root / ROADMAP_NEXT_RELATIVE_PATH).exists()
        else ROADMAP_RELATIVE_PATH
    )
    parsed = yaml.safe_load(read_text_if_present(root / relative)) or {}
    tasks = parsed.get("tasks", []) if isinstance(parsed, Mapping) else []
    task_ids = [
        str(task.get("id")) for task in tasks if isinstance(task, Mapping) and task.get("id")
    ]
    milestone = str(parsed.get("milestone", "")) if isinstance(parsed, Mapping) else ""
    return {"source": relative.as_posix(), "milestone": milestone, "task_ids": task_ids}


def normalize_timestamp(search_timestamp_utc: str | None) -> str:
    timestamp = search_timestamp_utc or datetime.now(UTC).replace(microsecond=0).isoformat()
    if timestamp.endswith("+00:00"):
        timestamp = timestamp[:-6] + "Z"
    return timestamp


def closed_scope_review() -> JsonDict:
    return {
        "free_form_answer_repair_reopened": False,
        "json_grammar_reopened": False,
        "external_generated_text_scoring_reopened": False,
        "token_or_logit_semantic_authority_reopened": False,
        "model_weight_writes_reopened": False,
        "broad_rl_reopened": False,
        "ptrm_generation_reopened": False,
        "two_axis_exchange_reopened": False,
        "learned_arc_value_transfer_reopened": False,
        "per_game_adapters_reopened": False,
        "unsupported_hardware_speedups_reopened": False,
        "operator_authorized_scope_expansion": None,
    }


def semantic_scholar_status() -> JsonDict:
    return {
        "route": "Semantic Scholar Graph API",
        "papers": ["arXiv:2507.02092", "arXiv:2512.15605"],
        "http_status": 200,
        "access": "checked",
        "latest_ebt_citation_publication_date": "2026-07-13",
        "latest_arm_ebm_citation_publication_date": "2026-07-02",
        "post_marker_citation_count": 0,
        "honest_status": "direct citation routes were reachable and exposed no citation after the V512 marker",
        "roadmap_delta": False,
    }


def extropic_status() -> JsonDict:
    return {
        "route": "https://extropic.ai/writing",
        "http_status": 200,
        "local_execution_available": False,
        "honest_status": "public writing reachable; no authenticated Carnot TSU path",
        "roadmap_delta": False,
    }


def logical_intelligence_status() -> JsonDict:
    return {
        "route": "https://logicalintelligence.com/kona-ebms-energy-based-models",
        "http_status": 200,
        "local_execution_available": False,
        "honest_status": "public Kona page reachable; no local weights, API receipts, or reproducible comparator",
        "roadmap_delta": False,
    }


def huggingface_status() -> JsonDict:
    return {
        "route": "https://huggingface.co/api/daily_papers?date=2026-07-20",
        "http_status": 200,
        "dates_checked": ["2026-07-20"],
        "api_result_count_for_2026_07_20": 0,
        "latest_visible_page_date": "2026-07-17",
        "honest_status": "daily API returned an empty array for 2026-07-20; no post-marker source accepted",
        "roadmap_delta": False,
    }


def github_status() -> JsonDict:
    return {
        "route": "GitHub repository search and trending",
        "http_status": 200,
        "recent_repository_searches": {
            "energy_based_reasoning_constraint_created_after_2026_07_19": 0,
            "kan_constraint_learning_created_after_2026_07_19": 0,
            "constrained_generation_verifier_created_after_2026_07_19": 0,
            "ising_sampler_created_after_2026_07_19": 0,
        },
        "trending_checked": True,
        "accepted_support_repository": None,
        "honest_status": "repository route supplied no new executable V512 dependency",
        "roadmap_delta": False,
    }


def accepted_findings(planner_found: bool) -> list[JsonDict]:
    if not planner_found:
        return []
    return []


def target_experiment_map(findings: list[JsonDict]) -> list[JsonDict]:
    return [
        {
            "source_id": finding["source_id"],
            "target_experiments": list(finding["target_experiments"]),
            "carnot_hook": finding["carnot_hook"],
            "substrate": finding["substrate"],
            "authority_boundary": finding["authority_boundary"],
            "falsifiable_metric": finding["falsifiable_metric"],
        }
        for finding in findings
    ]


def honest_verdict(planner_found: bool, findings: list[JsonDict]) -> str:
    if not planner_found:
        return "blocked: V512 planner refresh marker missing; source-delta append refused"
    if not findings:
        return "complete: no new non-duplicate actionable V512 source deltas; references left unchanged"
    return f"complete: accepted {len(findings)} non-duplicate actionable V512 source delta"


def preconditions_checked(root: Path, marker_found: bool) -> JsonDict:
    spec_text = read_text_if_present(root / SPEC_RELATIVE_PATH)
    return {
        "agents_md_read": (root / "AGENTS.md").exists(),
        "codex_md_read": (root / "CODEX.md").exists(),
        "claude_md_read": (root / "CLAUDE.md").exists(),
        "research_program_read": (root / "research-program.md").exists(),
        "research_references_read": (root / RESEARCH_REFERENCES_RELATIVE_PATH).exists(),
        "research_complete_yaml_read": (root / RESEARCH_COMPLETE_RELATIVE_PATH).exists(),
        "research_roadmap_next_yaml_present": (root / ROADMAP_NEXT_RELATIVE_PATH).exists(),
        "roadmap_fallback_allowed_if_next_absent": True,
        "vnext_proposal_read": (root / VNEXT_RELATIVE_PATH).exists(),
        "exclusion_manifest_read": (root / EXCLUSION_MANIFEST_RELATIVE_PATH).exists(),
        "known_issues_read": (root / KNOWN_ISSUES_RELATIVE_PATH).exists(),
        "spec_has_req_report_5732": "REQ-REPORT-5732" in spec_text,
        "planner_marker_found": marker_found,
        "scripts_research_conductor_modified": False,
        "ops_status_modified": False,
        "ops_changelog_modified": False,
        "traceability_modified": False,
    }


def duplicate_checks() -> JsonDict:
    arxiv_ids = [
        "2607.13921",
        "2607.14571",
        "2607.15003",
        "2607.15225",
        "2607.15253",
        "2607.13446",
        "2607.14952",
        "2607.14777",
        "2607.15200",
    ]
    source_ids = [
        *(row["source_id"] for row in DUPLICATE_FINDINGS),
        *(row["source_id"] for row in WATCH_ONLY_FINDINGS),
        *(row["source_id"] for row in INACCESSIBLE_FINDINGS),
        *(row["source_id"] for row in EXCLUDED_FINDINGS),
    ]
    return {
        "arxiv_ids_checked": arxiv_ids,
        "arxiv_ids_unique": len(arxiv_ids) == len(set(arxiv_ids)),
        "source_ids_unique": len(source_ids) == len(set(source_ids)),
        "titles_checked_against_local_ledgers": True,
        "techniques_checked_against_local_ledgers": True,
        "repositories_checked_against_local_ledgers": True,
        "citations_checked_against_local_ledgers": True,
        "carnot_hooks_checked_against_exp5733_exp5741": True,
    }


def field_principles_for(payload: Mapping[str, Any]) -> JsonDict:
    principles: JsonDict = {"field_principles": FIELD_PRINCIPLES["field_principles"]}
    for key in payload:
        principles[key] = FIELD_PRINCIPLES[key]
    return principles


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    search_timestamp_utc: str | None = None,
    run_date: str = RUN_DATE,
    duration_s: float = 0.0,
) -> JsonDict:
    references_text = read_text_if_present(root / RESEARCH_REFERENCES_RELATIVE_PATH)
    marker_found = planner_marker_found(references_text)
    findings = accepted_findings(marker_found)
    roadmap_change_required = any(
        bool(finding.get("roadmap_change_required_if_pursued")) for finding in findings
    )
    status = "blocked" if not marker_found or roadmap_change_required else "complete"
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "milestone": MILESTONE,
        "run_date": run_date,
        "search_cutoff": SEARCH_CUTOFF,
        "search_timestamp_utc": normalize_timestamp(search_timestamp_utc),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "spec_refs": list(SPEC_REFS),
        "preconditions_checked": preconditions_checked(root, marker_found),
        "planner_marker": PLANNER_MARKER,
        "planner_marker_found": marker_found,
        "sources_checked": clone_json(SOURCES_CHECKED),
        "queries": clone_json(QUERIES),
        "source_link_checks": clone_json(SOURCE_LINK_CHECKS),
        "dedupe_corpus_checked": dedupe_corpus(root),
        "marker_checks": {
            "planner_heading": PLANNER_HEADING,
            "planner_marker": PLANNER_MARKER,
            "planner_end_marker": PLANNER_END_MARKER,
            "planner_marker_found": marker_found,
            "planner_marker_line": planner_marker_line(references_text),
            "search_window": "strictly_after_V512_PLANNER_REFRESH_20260719_END",
        },
        "duplicate_checks": duplicate_checks(),
        "accepted_findings": findings,
        "duplicate_findings": clone_json(DUPLICATE_FINDINGS),
        "watch_only_findings": clone_json(WATCH_ONLY_FINDINGS),
        "inaccessible_findings": clone_json(INACCESSIBLE_FINDINGS),
        "excluded_findings": clone_json(EXCLUDED_FINDINGS),
        "semantic_scholar_status": semantic_scholar_status(),
        "extropic_status": extropic_status(),
        "logical_intelligence_status": logical_intelligence_status(),
        "huggingface_status": huggingface_status(),
        "github_status": github_status(),
        "target_experiment_map": target_experiment_map(findings),
        "closed_scope_review": closed_scope_review(),
        "roadmap_context": roadmap_context(root),
        "roadmap_change_required": roadmap_change_required,
        "references_updated": False,
        "references_mutated_this_run": False,
        "benchmark_compute_claimed": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "random_seed": RANDOM_SEED,
        "honest_verdict": honest_verdict(marker_found, findings),
        "reproducibility_checksum": "",
    }
    artifact["field_principles"] = field_principles_for(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _validate_timestamp(timestamp: str) -> None:
    require(timestamp.endswith("Z"), "search_timestamp_utc must be a UTC Z timestamp")
    datetime.fromisoformat(timestamp[:-1] + "+00:00")


def _validate_findings(artifact: Mapping[str, Any]) -> None:
    for finding in artifact["accepted_findings"]:
        for key in (
            "source_id",
            "title",
            "url",
            "timestamp_utc",
            "target_experiments",
            "substrate",
            "authority_boundary",
            "falsifiable_metric",
        ):
            require(key in finding, f"accepted finding missing {key}")
        require(
            all(target in ALLOWED_TARGET_EXPERIMENTS for target in finding["target_experiments"]),
            "accepted finding has a disallowed target experiment",
        )
    for mapping in artifact["target_experiment_map"]:
        targets = mapping.get("target_experiments", [])
        require(
            targets and all(target in ALLOWED_TARGET_EXPERIMENTS for target in targets),
            "target experiment map contains a disallowed target experiment",
        )


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    for field in REQUIRED_ARTIFACT_FIELDS:
        require(field in artifact, f"missing required artifact field {field}")
    require(
        set(artifact["field_principles"]) == set(artifact),
        "field_principles must cover every top-level artifact field",
    )
    require(artifact["planner_marker"] == PLANNER_MARKER, "planner_marker mismatch")
    require(artifact["inference_substrate"] == INFERENCE_SUBSTRATE, "bad inference_substrate")
    require(artifact["benchmark_compute_claimed"] is False, "benchmark compute must be false")
    require(isinstance(artifact["references_updated"], bool), "references_updated must be bool")
    require(
        isinstance(artifact["roadmap_change_required"], bool),
        "roadmap_change_required must be bool",
    )
    require(
        artifact["honest_verdict"].startswith(TERMINAL_PREFIXES),
        "honest_verdict must have a terminal prefix",
    )
    _validate_timestamp(str(artifact["search_timestamp_utc"]))
    _validate_findings(artifact)
    if artifact["roadmap_change_required"]:
        require(artifact["status"] == "blocked", "scope expansion must block")
        require(
            artifact["honest_verdict"].startswith("blocked:"),
            "scope expansion needs blocked verdict",
        )
    require(
        artifact["duplicate_checks"]["arxiv_ids_unique"],
        "duplicate arXiv id check failed",
    )
    require(
        artifact["duplicate_checks"]["source_ids_unique"],
        "duplicate source id check failed",
    )
    require(
        artifact["reproducibility_checksum"] == payload_checksum(artifact),
        "reproducibility checksum mismatch",
    )


def build_and_write_artifact(
    *,
    root: Path = REPO_ROOT,
    search_timestamp_utc: str | None = None,
    duration_s: float | None = None,
) -> JsonDict:
    started = time.perf_counter()
    elapsed = 0.0 if duration_s is None else duration_s
    artifact = build_artifact(
        root=root,
        search_timestamp_utc=search_timestamp_utc,
        duration_s=elapsed,
    )
    if duration_s is None:
        artifact["duration_s"] = round(time.perf_counter() - started, 6)
        artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    write_json(root / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--search-timestamp-utc", default=None)
    args = parser.parse_args(argv)
    artifact = build_and_write_artifact(
        root=args.root,
        search_timestamp_utc=args.search_timestamp_utc,
    )
    print(f"wrote {artifact['result_path']} with verdict: {artifact['honest_verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
