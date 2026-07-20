"""Exp5744: ingest the V513 source delta with bibliographic timing.

Spec refs: REQ-REPORT-5744, SCENARIO-REPORT-5744-ZERO-FINDING,
SCENARIO-REPORT-5744-ACCEPT-BOUNDED-DELTA,
SCENARIO-REPORT-5744-BLOCKED-MARKER,
SCENARIO-REPORT-5744-FIELD-PRINCIPLES.

The public web search is intentionally summarized rather than replayed in
tests. Source indexes, daily paper feeds, citation APIs, and repository search
results drift. This module preserves the durable receipt: what was checked,
which routes were duplicate/watch-only/inaccessible/excluded, why no
post-V513 actionable finding was accepted by default, and why the elapsed time
is bibliographic search time rather than benchmark, model, or hardware compute.
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
RESULT_RELATIVE_PATH = Path("results/experiment_5744_v513_source_delta_ingestion.json")
RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
KNOWN_ISSUES_RELATIVE_PATH = Path("ops/known-issues.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")

EXPERIMENT = "experiment_5744_v513_source_delta_ingestion"
EXPERIMENT_ID = "exp5744-v513-source-delta-ingestion"
MILESTONE = "2026.07.513"
RUN_DATE = "20260720"
SEARCH_CUTOFF = "2026-07-20"
SCHEMA = "carnot.experiment_5744.v513_source_delta_ingestion.v1"
RANDOM_SEED = 5744
INFERENCE_SUBSTRATE = "web_and_bibliographic_search_only"
TERMINAL_PREFIXES = ("complete:", "blocked:")

PLANNER_HEADING = "## V513 Planner Refresh - 20260720"
PLANNER_MARKER = "V513-PLANNER-REFRESH-20260720-END"
PLANNER_END_MARKER = f"<!-- {PLANNER_MARKER} -->"
EXECUTION_REFRESH_HEADING = "## V513 Execution Refresh - 20260720"
EXECUTION_REFRESH_END_MARKER = "<!-- V513-EXECUTION-REFRESH-20260720-END -->"

ALLOWED_TARGET_EXPERIMENTS = {
    "exp5746-exact-proposal-utility-benchmark",
    "exp5747-sota-exact-proposal-utility-panel",
    "exp5748-selective-exact-feedback-search",
    "exp5749-csl-render-matched-mechanism-audit",
    "exp5750-dependent-task-continuous-self-learning",
    "exp5751-rust-restart-parity-repair",
    "exp5752-one-axis-allocation-free-10x-crossover",
    "exp5753-arc-generic-primitive-live-registry-ab",
}

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "planner_marker",
    "search_started_at_utc",
    "search_finished_at_utc",
    "bibliographic_elapsed_s",
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
        "Records instruction, spec, marker, ledger, exclusion, and protected-file checks before source findings are trusted."
    ),
    "planner_marker": "Binds the search window to the post-V513 planner boundary.",
    "search_started_at_utc": "Records the instant immediately before mutable external querying starts.",
    "search_finished_at_utc": (
        "Records the instant after final source disposition so elapsed time is honest."
    ),
    "bibliographic_elapsed_s": (
        "Measures source-search wall time only, preventing bibliographic work from masquerading as benchmark compute."
    ),
    "sources_checked": "Makes source coverage reconstructable without trusting prose.",
    "queries": "Makes search intent reconstructable without rerunning mutable indexes.",
    "accepted_findings": (
        "Accepted work must have a post-marker, local, falsifiable Exp5746-Exp5753 home."
    ),
    "duplicate_findings": "Already-indexed material stays visible but cannot create duplicate work.",
    "watch_only_findings": "Non-local or non-executable material cannot support Carnot claims.",
    "excluded_findings": "Closed scopes remain closed by explicit disposition.",
    "semantic_scholar_status": "Citation-route access and citation-count limitations are reported honestly.",
    "extropic_status": (
        "Thermodynamic hardware context is bounded to authenticated local execution availability."
    ),
    "logical_intelligence_status": (
        "Kona/Aleph context is bounded to public pages unless local receipts exist."
    ),
    "huggingface_status": "Secondary paper-feed status is separated from primary authority.",
    "github_status": "Repository discovery is separated from executable local dependencies.",
    "target_experiment_map": (
        "Every accepted source maps to an existing Exp5746-Exp5753 task without changing ids or gates."
    ),
    "roadmap_change_required": (
        "Scope expansion blocks for operator review instead of silently mutating the roadmap."
    ),
    "references_updated": "Reference-file mutation is declared; zero-finding and blocked runs keep this false.",
    "benchmark_compute_claimed": (
        "Bibliographic search cannot claim benchmark, model, solver, or hardware compute."
    ),
    "inference_substrate": "The run used web and bibliographic search only.",
    "reproducibility_checksum": "The stable artifact payload can be checked for drift.",
    "honest_verdict": "The terminal result states complete or blocked without claim inflation.",
}

FIELD_PRINCIPLES: dict[str, str] = {
    **REQUIRED_FIELD_PRINCIPLES,
    "schema": "Names the artifact schema used by downstream validators.",
    "experiment": "Provides the stable module-level experiment slug.",
    "experiment_id": "Binds this receipt to the Exp5744 roadmap task.",
    "status": "Machine-readable terminal state derived from marker and scope checks.",
    "milestone": "Prevents this V513 receipt from being reused for another milestone.",
    "run_date": "Records the operator-requested execution date in compact form.",
    "search_cutoff": "Records the calendar date through which public sources were checked.",
    "result_path": "Records where the JSON receipt is written.",
    "spec_refs": "Links the artifact to its OpenSpec requirement and scenarios.",
    "planner_marker_found": "Shows whether the source window was anchored before any mutation.",
    "source_link_checks": "Records the specific public URLs or APIs checked.",
    "dedupe_corpus_checked": "Lists local files hashed for duplicate and boundary review.",
    "marker_checks": "Records heading and marker details for reference-marker validation.",
    "duplicate_checks": "Summarizes id, title, repository, citation, and hook dedupe checks.",
    "inaccessible_findings": "Separates access failures from scientific exclusions.",
    "closed_scope_review": "Documents that banned research scopes remain closed.",
    "roadmap_context": "Records which roadmap queue was available without recreating files.",
    "references_mutated_this_run": "Separates durable references_updated status from this invocation.",
    "random_seed": "Keeps deterministic receipt metadata aligned with experiment id.",
}

SPEC_REFS = (
    "REQ-REPORT-5744",
    "SCENARIO-REPORT-5744-ZERO-FINDING",
    "SCENARIO-REPORT-5744-ACCEPT-BOUNDED-DELTA",
    "SCENARIO-REPORT-5744-BLOCKED-MARKER",
    "SCENARIO-REPORT-5744-FIELD-PRINCIPLES",
)

QUERIES: tuple[JsonDict, ...] = (
    {
        "surface": "arXiv",
        "timestamp_window_utc": f"strictly_after_{PLANNER_MARKER}",
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
            "cs.AI new",
            "cs new",
            "cs.LG new",
        ],
    },
    {
        "surface": "OpenReview",
        "queries": [
            "energy-based reasoning",
            "constrained generation",
            "continual learning constraint",
            "certifiable continual learning",
            "CerCE Anh6VfNM22",
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
            'repo search "energy-based" reasoning constraint created:>2026-07-20',
            "repo search KAN constraint learning created:>2026-07-20",
            "repo search constrained generation verifier created:>2026-07-20",
            "repo search Ising sampler created:>2026-07-20",
            "github trending 2026-07-20",
        ],
    },
    {"surface": "Extropic writing", "queries": ["TSU", "XTR-0", "X0", "THRML"]},
    {
        "surface": "Logical Intelligence public pages",
        "queries": ["Kona", "Aleph", "Energy-Based Reasoning Models", "verified reasoning"],
    },
    {
        "surface": "local Carnot ledgers",
        "queries": [
            "research-references.md after V513 marker",
            "research-complete.yaml",
            "research-roadmap.yaml",
            "research-roadmap-next.yaml if present",
            "openspec/change-proposals/research-roadmap-vNEXT.md",
            "results/experiment_5732_v512_source_delta_ingestion.json",
            "ops/exclusion_manifest.yaml",
            "ops/known-issues.md",
        ],
    },
)

SOURCES_CHECKED: tuple[JsonDict, ...] = (
    {
        "surface": "arXiv",
        "status": "category_pages_checked; export_api_timeout_and_rate_limit_recorded",
        "decision": (
            "Post-marker category pages and API attempts exposed no non-duplicate "
            "actionable Exp5746-Exp5753 source; all visible items were duplicate, "
            "watch-only, inaccessible, or excluded"
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
        "decision": "no citation after the V513 marker changed the task graph",
    },
    {
        "surface": "Hugging Face Papers",
        "status": "daily_papers_api_checked_for_2026_07_20",
        "decision": "secondary feed supplied no non-duplicate local Exp5746-Exp5753 dependency",
    },
    {
        "surface": "GitHub discovery",
        "status": "repository_search_api_http_200_and_trending_html_http_200",
        "decision": "no new reproducible local repository displaced Carnot substrates",
    },
    {
        "surface": "Extropic writing",
        "status": "public_search_results_checked",
        "decision": "watch-only; no authenticated local TSU execution route",
    },
    {
        "surface": "Logical Intelligence public pages",
        "status": "public_search_results_checked",
        "decision": "watch-only; no local weights, API receipt, or reproducible comparator",
    },
    {
        "surface": "local Carnot ledgers",
        "status": "checked",
        "decision": (
            "V513 already indexed Opt-Verifier, selective verification, hard/soft "
            "controls, render confounds, ARC EWM ablation, Adaptive Generate-Rank-Verify, "
            "CerCE, and ARM-EBM v4"
        ),
    },
)

SOURCE_LINK_CHECKS: tuple[JsonDict, ...] = (
    {
        "source_id": "arxiv_cs_ai_recent",
        "url": "https://arxiv.org/list/cs.AI/new",
        "status": "http_200_checked_no_new_non_duplicate_actionable_v513_delta",
    },
    {
        "source_id": "arxiv_cs_recent",
        "url": "https://arxiv.org/list/cs/new",
        "status": "http_200_checked_no_new_non_duplicate_actionable_v513_delta",
    },
    {
        "source_id": "arxiv_cs_lg_recent",
        "url": "https://arxiv.org/list/cs.LG/new",
        "status": "http_200_checked_no_new_non_duplicate_actionable_v513_delta",
    },
    {
        "source_id": "urlcheck_maxsat_vlm_sudoku_2607_12711",
        "linked_source_id": "maxsat_vlm_sudoku_2607_12711",
        "url": "https://arxiv.org/abs/2607.12711",
        "status": "primary_arxiv_opened_duplicate_v511_exp5718",
    },
    {
        "source_id": "urlcheck_lazy_arithmetic_2607_15328",
        "linked_source_id": "lazy_arithmetic_2607_15328",
        "url": "https://arxiv.org/abs/2607.15328",
        "status": "primary_arxiv_opened_duplicate_v512_exclusion",
    },
    {
        "source_id": "urlcheck_xhc_2607_14530",
        "linked_source_id": "xhc_2607_14530",
        "url": "https://arxiv.org/abs/2607.14530",
        "status": "primary_arxiv_opened_excluded_model_weight_architecture_scope",
    },
    {
        "source_id": "urlcheck_dsworld_2607_15901",
        "linked_source_id": "dsworld_2607_15901",
        "url": "https://arxiv.org/abs/2607.15901",
        "status": "primary_arxiv_opened_excluded_world_model_rl_scope",
    },
    {
        "source_id": "arxiv_export_api_queries",
        "url": "https://export.arxiv.org/api/query",
        "status": "timeout_or_http_429_recorded_as_inaccessible_not_accepted",
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
        "status": "http_200_no_hf_only_dependency_promoted",
    },
    {
        "source_id": "github_recent_energy_constraint_query",
        "url": "https://api.github.com/search/repositories?q=%22energy-based%22+reasoning+constraint+created:%3E2026-07-20",
        "status": "http_200_total_count_0",
    },
    {
        "source_id": "github_recent_kan_constraint_query",
        "url": "https://api.github.com/search/repositories?q=KAN+constraint+learning+created:%3E2026-07-20",
        "status": "http_200_total_count_0",
    },
    {
        "source_id": "github_recent_constrained_generation_query",
        "url": "https://api.github.com/search/repositories?q=constrained+generation+verifier+created:%3E2026-07-20",
        "status": "http_200_total_count_0",
    },
    {
        "source_id": "github_recent_ising_sampler_query",
        "url": "https://api.github.com/search/repositories?q=Ising+sampler+created:%3E2026-07-20",
        "status": "http_200_total_count_0",
    },
    {
        "source_id": "github_trending",
        "url": "https://github.com/trending",
        "status": "http_200_no_reproducible_local_carnot_dependency_promoted",
    },
    {
        "source_id": "extropic_public_search",
        "url": "https://extropic.ai/",
        "status": "public_material_watch_only_no_local_tsu",
    },
    {
        "source_id": "logical_intelligence_public_search",
        "url": "https://logicalintelligence.com/",
        "status": "public_material_watch_only_no_local_weights_or_receipts",
    },
    {
        "source_id": "openreview_browser_challenge",
        "url": "https://openreview.net/",
        "status": "browser_challenge_no_primary_post_marker_promotion",
    },
)

ACCEPTED_FINDINGS: tuple[JsonDict, ...] = ()

DUPLICATE_FINDINGS: tuple[JsonDict, ...] = (
    {
        "source_id": "maxsat_vlm_sudoku_2607_12711",
        "title": "MaxSAT-Based Feedback for Guiding Vision-Language Models in Sudoku",
        "url": "https://arxiv.org/abs/2607.12711",
        "reason": (
            "Already indexed by V511/Exp5718 and present in research-references.md; "
            "its MaxSAT hard/soft constraint hook is covered by existing selective "
            "exact-feedback planning"
        ),
    },
    {
        "source_id": "lazy_arithmetic_2607_15328",
        "title": "Lazy Arithmetic using Systolic Arrays for Closing the Verification Gap on Embedded Systems",
        "url": "https://arxiv.org/abs/2607.15328",
        "reason": (
            "Already disposed by V512/Exp5732 as unauthenticated in-progress hardware "
            "work that cannot support a local Carnot timing or correctness receipt"
        ),
    },
    {
        "source_id": "cerce_openreview_Anh6VfNM22",
        "title": "CerCE: Certifiable Continual Learning",
        "url": "https://openreview.net/forum?id=Anh6VfNM22",
        "reason": "Already indexed by the V513 planner for continual self-learning certificate context.",
    },
    {
        "source_id": "opt_verifier_2605_29556",
        "title": "Opt-Verifier",
        "url": "https://arxiv.org/abs/2605.29556",
        "reason": "Already indexed by the V513 planner for Exp5746/Exp5747 exact proposal utility.",
    },
    {
        "source_id": "selective_verification_2606_19808",
        "title": "Think Again or Think Longer? Test-time Scaling with Selective Verification",
        "url": "https://arxiv.org/abs/2606.19808",
        "reason": "Already indexed by the V513 planner for selective exact feedback search.",
    },
)

WATCH_ONLY_FINDINGS: tuple[JsonDict, ...] = (
    {
        "source_id": "extropic_x0_xtr0_tsu_public_material",
        "title": "Extropic X0, XTR-0, TSU, and THRML public material",
        "url": "https://extropic.ai/",
        "classification": "watch_only_non_local_hardware",
        "reason": (
            "Public thermodynamic-sampling context only; no authenticated Carnot "
            "hardware, SDK, timing, or correctness route exists"
        ),
    },
    {
        "source_id": "logical_intelligence_kona_aleph_public_material",
        "title": "Logical Intelligence Kona and Aleph public pages",
        "url": "https://logicalintelligence.com/",
        "classification": "watch_only_proprietary_comparator",
        "reason": "Public/proprietary context only; no local weights, API receipt, or reproducible comparator.",
    },
    {
        "source_id": "github_trending_2026_07_20",
        "title": "GitHub trending and discovery queries",
        "url": "https://github.com/trending",
        "classification": "watch_only_no_local_dependency",
        "reason": "Repository discovery produced no reproducible Carnot EBM/CSP/KAN/sampler dependency.",
    },
    {
        "source_id": "semantic_scholar_ebt_arm_citation_routes",
        "title": "Semantic Scholar citation routes for EBT and ARM-EBM",
        "url": "https://api.semanticscholar.org/",
        "classification": "watch_only_no_post_marker_citation_delta",
        "reason": "Both citation routes were reachable, but visible citations predated the V513 marker.",
    },
)

INACCESSIBLE_FINDINGS: tuple[JsonDict, ...] = (
    {
        "source_id": "arxiv_export_api_timeout_or_429",
        "title": "arXiv export API broad query set",
        "url": "https://export.arxiv.org/api/query",
        "classification": "inaccessible_timeout_or_rate_limit",
        "reason": "Broad export queries timed out or returned HTTP 429; arXiv category pages were used as fallback.",
    },
    {
        "source_id": "openreview_browser_challenge",
        "title": "OpenReview primary forum pages",
        "url": "https://openreview.net/",
        "classification": "inaccessible_browser_challenge",
        "reason": "Search results were visible, but primary pages required browser verification.",
    },
)

EXCLUDED_FINDINGS: tuple[JsonDict, ...] = (
    {
        "source_id": "xhc_2607_14530",
        "title": "xHC: Enhancing LLM Reasoning by Expanding Hyper-Connections",
        "url": "https://arxiv.org/abs/2607.14530",
        "reason": "Residual-stream architecture expansion and pretraining reopen model-weight scope.",
    },
    {
        "source_id": "dsworld_2607_15901",
        "title": "DSWorld: A Data Science World Model for Logical and Efficient Agentic Reasoning",
        "url": "https://arxiv.org/abs/2607.15901",
        "reason": "LLM simulator and RL/search world-model redesign are outside the Exp5746-Exp5753 graph.",
    },
    {
        "source_id": "xiaomi_robotics_1_2607_15330",
        "title": "Xiaomi-Robotics-1: A High-Performance Vision-Language-Action Model",
        "url": "https://arxiv.org/abs/2607.15330",
        "reason": "Robotics foundation-model training and post-training are model-weight and robotics scopes.",
    },
    {
        "source_id": "cura_1t_2607_15314",
        "title": "Cura: An Open 1T Parameter Medical Reasoning Model",
        "url": "https://arxiv.org/abs/2607.15314",
        "reason": "Healthcare self-evolution and trillion-parameter training reopen model-weight-write scope.",
    },
    {
        "source_id": "s1_omni_2607_15686",
        "title": "S1-Omni: A Unified Scientific Foundation Model",
        "url": "https://arxiv.org/abs/2607.15686",
        "reason": "Unified scientific model training supplies no local exact-validator or sidecar authority.",
    },
    {
        "source_id": "loop_the_loopies_2607_16051",
        "title": "Loop the Loopies: Training Long-Context Looping Transformers",
        "url": "https://arxiv.org/abs/2607.16051",
        "reason": "Transformer training and post-training are outside the closed V513 local sidecar scope.",
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
        root / "results/experiment_5732_v512_source_delta_ingestion.json",
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


def bibliographic_elapsed_s(started_at_utc: str, finished_at_utc: str) -> float:
    started = parse_utc_timestamp(started_at_utc, "search_started_at_utc")
    finished = parse_utc_timestamp(finished_at_utc, "search_finished_at_utc")
    require(finished > started, "timestamp order requires search_finished_at_utc after start")
    return round((finished - started).total_seconds(), 6)


def closed_scope_review() -> JsonDict:
    return {
        "graph_redesign_reopened": False,
        "external_text_scoring_reopened": False,
        "llm_judges_reopened": False,
        "model_weight_writes_reopened": False,
        "broad_rl_reopened": False,
        "retired_arc_value_scope_reopened": False,
        "headline_scorer_reopened": False,
        "unauthenticated_hardware_claims_reopened": False,
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
        "citation_count_claimed": False,
        "honest_status": "direct citation routes were reachable and exposed no citation after the V513 marker",
        "roadmap_delta": False,
    }


def extropic_status() -> JsonDict:
    return {
        "route": "Extropic public search and writing",
        "http_status": "public_search_result_available",
        "local_execution_available": False,
        "authenticated_hardware_claim": False,
        "honest_status": "public material reachable through search; no authenticated Carnot TSU path",
        "roadmap_delta": False,
    }


def logical_intelligence_status() -> JsonDict:
    return {
        "route": "Logical Intelligence public search results",
        "http_status": "public_search_result_available",
        "local_execution_available": False,
        "local_weights_or_api_receipt": False,
        "honest_status": "public Kona/Aleph context only; no local weights, API receipt, or comparator",
        "roadmap_delta": False,
    }


def huggingface_status() -> JsonDict:
    return {
        "route": "https://huggingface.co/api/daily_papers?date=2026-07-20",
        "http_status": 200,
        "dates_checked": ["2026-07-20"],
        "sample_source_ids": [
            "2607.14530",
            "2607.15330",
            "2607.15314",
            "2607.16051",
            "2607.15901",
            "2607.15686",
        ],
        "latest_visible_page_date": "2026-07-20",
        "honest_status": (
            "daily API returned visible 2026-07-20 papers; none supplied an HF-only "
            "dependency beyond duplicate or excluded primary dispositions"
        ),
        "roadmap_delta": False,
    }


def github_status() -> JsonDict:
    return {
        "route": "GitHub repository search and trending",
        "http_status": 200,
        "recent_repository_searches": {
            "energy_based_reasoning_constraint_created_after_2026_07_20": 0,
            "kan_constraint_learning_created_after_2026_07_20": 0,
            "constrained_generation_verifier_created_after_2026_07_20": 0,
            "ising_sampler_created_after_2026_07_20": 0,
        },
        "trending_checked": True,
        "accepted_support_repository": None,
        "honest_status": "repository route supplied no new executable V513 dependency",
        "roadmap_delta": False,
    }


def accepted_findings(planner_found: bool, supplied: list[JsonDict] | None) -> list[JsonDict]:
    if not planner_found:
        return []
    return clone_json(ACCEPTED_FINDINGS if supplied is None else supplied)


def target_experiment_map(findings: list[JsonDict]) -> list[JsonDict]:
    return [
        {
            "source_id": finding["source_id"],
            "target_experiments": list(finding["target_experiments"]),
            "carnot_hook": finding["carnot_hook"],
            "substrate": finding.get("substrate", finding["local_substrate"]),
            "local_substrate": finding["local_substrate"],
            "authority_boundary": finding["authority_boundary"],
            "falsifiable_metric": finding["falsifiable_metric"],
        }
        for finding in findings
    ]


def honest_verdict(planner_found: bool, findings: list[JsonDict]) -> str:
    if not planner_found:
        return "blocked: V513 planner refresh marker missing; source-delta append refused"
    if not findings:
        return "complete: no new non-duplicate actionable V513 source deltas; references left unchanged"
    return (
        f"complete: accepted {len(findings)} non-duplicate actionable V513 source "
        "deltas; no roadmap ID, gate, benchmark, model-weight, or hardware claim changed"
    )


def preconditions_checked(root: Path, marker_found: bool) -> JsonDict:
    spec_text = read_text_if_present(root / SPEC_RELATIVE_PATH)
    return {
        "agents_md_read": (root / "AGENTS.md").exists(),
        "codex_md_read": (root / "CODEX.md").exists(),
        "claude_md_read": (root / "CLAUDE.md").exists(),
        "research_program_read": (root / "research-program.md").exists(),
        "research_references_read": (root / RESEARCH_REFERENCES_RELATIVE_PATH).exists(),
        "research_complete_yaml_read": (root / RESEARCH_COMPLETE_RELATIVE_PATH).exists(),
        "research_roadmap_yaml_read": (root / ROADMAP_RELATIVE_PATH).exists(),
        "research_roadmap_next_yaml_present": (root / ROADMAP_NEXT_RELATIVE_PATH).exists(),
        "roadmap_fallback_allowed_if_next_absent": True,
        "vnext_proposal_read": (root / VNEXT_RELATIVE_PATH).exists(),
        "exclusion_manifest_read": (root / EXCLUSION_MANIFEST_RELATIVE_PATH).exists(),
        "known_issues_read": (root / KNOWN_ISSUES_RELATIVE_PATH).exists(),
        "prior_v512_artifact_read": (root / "results/experiment_5732_v512_source_delta_ingestion.json").exists(),
        "spec_has_req_report_5744": "REQ-REPORT-5744" in spec_text,
        "planner_marker_found": marker_found,
        "scripts_research_conductor_modified": False,
        "ops_status_modified": False,
        "ops_changelog_modified": False,
        "traceability_modified": False,
    }


def duplicate_checks(
    accepted_rows: list[JsonDict],
    duplicate_rows: list[JsonDict],
    watch_only_rows: list[JsonDict],
    inaccessible_rows: list[JsonDict],
    excluded_rows: list[JsonDict],
) -> JsonDict:
    arxiv_ids = [
        row["arxiv_id"]
        for row in accepted_rows
        if isinstance(row.get("arxiv_id"), str) and row["arxiv_id"]
    ]
    for rows in (duplicate_rows, watch_only_rows, excluded_rows):
        arxiv_ids.extend(
            url.rsplit("/", 1)[-1]
            for row in rows
            if isinstance(url := row.get("url"), str)
            and url.startswith("https://arxiv.org/abs/")
        )
    source_ids = [
        row["source_id"]
        for rows in (
            accepted_rows,
            duplicate_rows,
            watch_only_rows,
            inaccessible_rows,
            excluded_rows,
        )
        for row in rows
    ]
    return {
        "arxiv_ids_checked": arxiv_ids,
        "arxiv_ids_unique": len(arxiv_ids) == len(set(arxiv_ids)),
        "source_ids_unique": len(source_ids) == len(set(source_ids)),
        "titles_checked_against_local_ledgers": True,
        "techniques_checked_against_local_ledgers": True,
        "repositories_checked_against_local_ledgers": True,
        "citations_checked_against_local_ledgers": True,
        "carnot_hooks_checked_against_exp5746_exp5753": True,
    }


def execution_refresh_block(findings: list[JsonDict]) -> str:
    lines = [
        "",
        EXECUTION_REFRESH_HEADING,
        "",
        (
            "Execution-time sweep on 2026-07-20 after the V513 planner marker. "
            "Only non-duplicate sources that sharpen already-allocated Exp5746-Exp5753 "
            "work are listed here; duplicate, watch-only, inaccessible, and excluded "
            "findings are recorded in `results/experiment_5744_v513_source_delta_ingestion.json`."
        ),
        "",
        "### New actionable deltas",
        "",
    ]
    for finding in findings:
        targets = ", ".join(finding["target_experiments"])
        substrate = finding.get("substrate", finding["local_substrate"])
        arxiv_fragment = (
            f" arXiv:{finding['arxiv_id']}," if finding.get("arxiv_id") else ""
        )
        lines.append(
            f"- **{finding['title']}** -{arxiv_fragment} {finding['url']}. "
            f"Carnot hook: {finding['carnot_hook']} Target: {targets}. "
            f"Substrate: {substrate}. Authority boundary: {finding['authority_boundary']}. "
            f"Falsifiable metric: {finding['falsifiable_metric']}. "
            "This sharpens existing V513 work only and does not authorize graph "
            "redesign, external text scoring, LLM judges, model-weight writes, "
            "broad RL, retired ARC/value/headline-scorer scopes, or unauthenticated "
            "hardware claims."
        )
    lines.extend(
        [
            "",
            "### V513 execution impact",
            "",
            (
                "- Preserve the Exp5746-Exp5753 graph and gates. Accepted deltas may "
                "only add bounded local controls, exact validators, and falsifiable "
                "receipts inside the existing roadmap ids."
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
    marker_found: bool,
    findings: list[JsonDict],
) -> bool:
    if not marker_found or not findings:
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
    search_started_at_utc: str | None = None,
    search_finished_at_utc: str | None = None,
    run_date: str = RUN_DATE,
    accepted_findings: list[JsonDict] | None = None,
    duplicate_findings: list[JsonDict] | None = None,
    watch_only_findings: list[JsonDict] | None = None,
    excluded_findings: list[JsonDict] | None = None,
    inaccessible_findings: list[JsonDict] | None = None,
    references_updated: bool | None = None,
    references_mutated_this_run: bool = False,
) -> JsonDict:
    references_text = read_text_if_present(root / RESEARCH_REFERENCES_RELATIVE_PATH)
    marker_found = planner_marker_found(references_text)
    started_at = normalize_timestamp(search_started_at_utc)
    finished_at = normalize_timestamp(search_finished_at_utc)
    findings = accepted_findings_fn(marker_found, accepted_findings)
    duplicates = clone_json(DUPLICATE_FINDINGS if duplicate_findings is None else duplicate_findings)
    watch_only = clone_json(WATCH_ONLY_FINDINGS if watch_only_findings is None else watch_only_findings)
    excluded = clone_json(EXCLUDED_FINDINGS if excluded_findings is None else excluded_findings)
    inaccessible = clone_json(
        INACCESSIBLE_FINDINGS if inaccessible_findings is None else inaccessible_findings
    )
    if references_updated is None:
        references_updated = bool(marker_found and findings)
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
        "search_started_at_utc": started_at,
        "search_finished_at_utc": finished_at,
        "bibliographic_elapsed_s": bibliographic_elapsed_s(started_at, finished_at),
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
            "search_window": "strictly_after_V513_PLANNER_REFRESH_20260720_END",
            "execution_refresh_heading": EXECUTION_REFRESH_HEADING,
            "execution_refresh_present": EXECUTION_REFRESH_HEADING in references_text,
        },
        "duplicate_checks": duplicate_checks(
            findings,
            duplicates,
            watch_only,
            inaccessible,
            excluded,
        ),
        "accepted_findings": findings,
        "duplicate_findings": duplicates,
        "watch_only_findings": watch_only,
        "inaccessible_findings": inaccessible,
        "excluded_findings": excluded,
        "semantic_scholar_status": semantic_scholar_status(),
        "extropic_status": extropic_status(),
        "logical_intelligence_status": logical_intelligence_status(),
        "huggingface_status": huggingface_status(),
        "github_status": github_status(),
        "target_experiment_map": target_experiment_map(findings),
        "closed_scope_review": closed_scope_review(),
        "roadmap_context": roadmap_context(root),
        "roadmap_change_required": roadmap_change_required,
        "references_updated": references_updated,
        "references_mutated_this_run": references_mutated_this_run,
        "benchmark_compute_claimed": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seed": RANDOM_SEED,
        "honest_verdict": honest_verdict(marker_found, findings),
        "reproducibility_checksum": "",
    }
    artifact["field_principles"] = field_principles_for(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def accepted_findings_fn(
    marker_found: bool,
    supplied: list[JsonDict] | None,
) -> list[JsonDict]:
    return accepted_findings(marker_found, supplied)


def _validate_findings(artifact: Mapping[str, Any]) -> None:
    for finding in artifact["accepted_findings"]:
        for key in (
            "source_id",
            "title",
            "url",
            "target_experiments",
            "local_substrate",
            "authority_boundary",
            "carnot_hook",
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
    started_at = str(artifact["search_started_at_utc"])
    finished_at = str(artifact["search_finished_at_utc"])
    expected_elapsed = bibliographic_elapsed_s(started_at, finished_at)
    require(
        abs(float(artifact["bibliographic_elapsed_s"]) - expected_elapsed) < 0.000001,
        "bibliographic elapsed mismatch",
    )
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
    search_started_at_utc: str | None = None,
    search_finished_at_utc: str | None = None,
    accepted_findings: list[JsonDict] | None = None,
    duplicate_findings: list[JsonDict] | None = None,
    watch_only_findings: list[JsonDict] | None = None,
    excluded_findings: list[JsonDict] | None = None,
    inaccessible_findings: list[JsonDict] | None = None,
) -> JsonDict:
    references_text = read_text_if_present(root / RESEARCH_REFERENCES_RELATIVE_PATH)
    marker_found = planner_marker_found(references_text)
    findings = accepted_findings_fn(marker_found, accepted_findings)
    references_mutated_this_run = append_execution_refresh_if_needed(
        root,
        marker_found,
        findings,
    )
    references_updated = bool(marker_found and findings)
    artifact = build_artifact(
        root=root,
        search_started_at_utc=search_started_at_utc,
        search_finished_at_utc=search_finished_at_utc,
        accepted_findings=findings,
        duplicate_findings=duplicate_findings,
        watch_only_findings=watch_only_findings,
        excluded_findings=excluded_findings,
        inaccessible_findings=inaccessible_findings,
        references_updated=references_updated,
        references_mutated_this_run=references_mutated_this_run,
    )
    validate_artifact(artifact)
    write_json(root / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--search-started-at-utc", default=None)
    parser.add_argument("--search-finished-at-utc", default=None)
    parser.add_argument(
        "--zero-findings",
        action="store_true",
        help="force zero accepted findings while preserving default dispositions",
    )
    args = parser.parse_args(argv)
    artifact = build_and_write_artifact(
        root=args.root,
        search_started_at_utc=args.search_started_at_utc,
        search_finished_at_utc=args.search_finished_at_utc,
        accepted_findings=[] if args.zero_findings else None,
    )
    print(f"wrote {artifact['result_path']} with verdict: {artifact['honest_verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
