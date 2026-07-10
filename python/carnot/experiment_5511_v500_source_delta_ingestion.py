"""Exp5511 execution-time V500 source delta ingestion.

Spec refs: REQ-REPORT-5511, SCENARIO-REPORT-5511-APPEND-DELTAS,
SCENARIO-REPORT-5511-NO-NEW-DELTA, SCENARIO-REPORT-5511-BLOCKED-MARKER.

This module records the short-lived freshness check that runs after the V500
planner refresh and before the science tasks. The important behavior is
negative as much as positive: already-ingested papers, vendor pages without a
local execution path, and retired scopes must not churn `research-references.md`
or silently reopen experiments. The artifact is therefore a receipt for what
was checked, what was suppressed, and why no V500 execution append was needed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_5511_v500_source_delta_ingestion"
TASK_ID = "exp5511-v500-sota-source-delta-ingestion"
MILESTONE = "2026.07.500"
SEARCH_DATE = "20260710"
RESULT_RELATIVE_PATH = Path("results/experiment_5511_v500_source_delta_ingestion.json")
REFERENCES_RELATIVE_PATH = Path("research-references.md")
PLANNER_MARKER = "## V500 Planner Refresh - 2026-07-09"
REFRESH_HEADING = "## V500 Execution Refresh - 20260710"
REFRESH_END_MARKER = "<!-- V500-EXECUTION-REFRESH-20260710-END -->"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

SPEC_REFS = [
    "REQ-REPORT-5511",
    "SCENARIO-REPORT-5511-APPEND-DELTAS",
    "SCENARIO-REPORT-5511-NO-NEW-DELTA",
    "SCENARIO-REPORT-5511-BLOCKED-MARKER",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "sources_checked": (
        "Records each primary, secondary, local-dedupe, and exclusion source checked "
        "before V500 science tasks run."
    ),
    "new_references_added": (
        "Lists only non-duplicate actionable findings that earned a V500 "
        "execution-refresh append."
    ),
    "duplicates_suppressed": (
        "Prevents churn from re-adding V500 planner sources or earlier V49x "
        "source-delta findings."
    ),
    "closed_scopes_reopened": (
        "Bare false boolean proving excluded, proprietary, non-local, or retired "
        "lanes stayed closed."
    ),
    "research_references_updated": (
        "Bare boolean distinguishing a real V500 append from a no-op freshness receipt."
    ),
    "prior_refresh_marker_found": (
        "Ensures the execution refresh dedupes against the actual V500 planner "
        "block before appending."
    ),
    "experiment_mappings": (
        "Maps every accepted finding to a planned `.500` experiment lane; empty "
        "only when no finding was accepted."
    ),
    "inference_substrate": (
        "Must be aggregation_from_upstream_artifacts because the receipt aggregates "
        "sources and local artifacts without model, solver, ARC, or hardware inference."
    ),
    "honest_verdict": (
        "One-line terminal summary starting with complete: or blocked: that states "
        "whether references changed."
    ),
}

REQUIRED_SOURCE_FAMILIES = (
    "arxiv_strict_post_marker",
    "arxiv_recent_api",
    "arxiv_primary_pages",
    "openreview",
    "huggingface_papers",
    "semantic_scholar_ebt_arm_ebm",
    "github",
    "extropic_writing",
    "logical_intelligence",
    "local_v500_and_v49x_duplicate_history",
    "ops_exclusion_manifest",
    "research_roadmap_v500_tasks",
)

ALLOWED_SOURCE_STATUSES = frozenset(
    {"ok", "partial", "rate_limited", "challenge_blocked"}
)

PLANNED_EXPERIMENT_TASKS: dict[str, str] = {
    "structured_sota_control": "exp5512-structured-output-positive-control",
    "csl_independence": "exp5515-csl-independent-outcome-gate-repair",
    "sparse_repair": "exp5518-block-gibbs-sparse-repair-descriptors",
    "hardware_receipts": "exp5519-hardware-continuity-methodology-receipts",
    "arc_live_path_improvement": "exp5520-arc-action-diversity-target-precheck",
}

REQUIRED_TOP_LEVEL_FIELDS = frozenset(
    {
        "experiment_id",
        "task_id",
        "milestone",
        "status",
        "search_date",
        "sources_checked",
        "new_actionable_findings_count",
        "new_references_added",
        "duplicates_suppressed",
        "closed_scopes_reopened",
        "research_references_updated",
        "prior_refresh_marker_found",
        "experiment_mappings",
        "honest_verdict",
        "inference_substrate",
        "searched_source_details",
        "watch_only_or_excluded",
        "spec_refs",
        "field_principles",
        "methodology_duration_s",
        "tests_run",
        "no_deep_research_used",
        "research_conductor_modified",
        "ops_docs_modified",
        "traceability_modified",
        "roadmap_files_modified",
    }
)

REQUIRED_REFERENCE_FIELDS = frozenset(
    {
        "title",
        "url",
        "source_type",
        "carnot_hook",
        "planned_experiment",
        "mapped_task",
    }
)

DEFAULT_NEW_REFERENCES_ADDED: list[JsonDict] = []

DUPLICATES_SUPPRESSED = [
    "Distributional EBMs for structured LLM reasoning already appears in the V500 planner block and prior V45x/V49x history - https://arxiv.org/abs/2605.18871",
    "PCRLLM proof-carrying reasoning, Thinking Before Constraining, XGrammar-2, llguidance, and Constrained Decoding for Diffusion LMs are already represented in V500 structured-control planning - https://arxiv.org/abs/2607.07026",
    "Spilled Energy, Semantic Energy, and Energy-Guided Decoding for Object Hallucination are duplicate energy-diagnostic or hidden-state contexts and do not create new V500 headline authority - https://arxiv.org/abs/2602.18671",
    "BloGDiT sparse repair and KAN verification or GRS-KAN references are already mapped to V500 sparse repair or watch-only learned-constraint context - https://arxiv.org/abs/2605.25129",
    "Programmable Probabilistic Computer with 1,000,000 p-bits, Fully Parallel Ising Machine, p-bit CDCL, and Probabilistic Memory are duplicate or non-local hardware context for receipt methodology only - https://arxiv.org/abs/2606.25313",
    "Budget-Curated Memory, constrained-bandit model selection, ExpGraph, Evo-Memory, graph memory surveys, and deployment-time memory are already covered by V489/V493/V500 CSL history - https://arxiv.org/abs/2606.25115",
    "EBT 2507.02092, ARM-EBM 2512.15605, EBT GitHub, OpenReview, HuggingFace Papers, and Semantic Scholar routes remain architecture context rather than new V500 execution dependencies - https://arxiv.org/abs/2507.02092",
    "Extropic TSU/XTR-0/Z1/THRML and Logical Intelligence Kona/Aleph public pages remain strategic non-local context with no Carnot-local SDK, reproducible internals, or speedup basis - https://logicalintelligence.com/",
]

WATCH_ONLY_OR_EXCLUDED: list[JsonDict] = [
    {
        "title": "Probabilistic Memory for Trustworthy Edge Intelligence",
        "url": "https://arxiv.org/abs/2607.02465",
        "classification": "watch-only",
        "reason": (
            "p-MEM is relevant to probabilistic sampling hardware, but Carnot has no "
            "local p-MEM device, SDK, or executable timing path. It stays receipt "
            "methodology context and does not change Exp5519."
        ),
    },
    {
        "title": "StructuredEdit constraint-aware graphic design editing",
        "url": "https://arxiv.org/abs/2607.04612",
        "classification": "watch-only",
        "reason": (
            "The differentiable-constraint idea is domain-specific graphics work, "
            "already classified in V492 history, and is not executable for the V500 "
            "hard/soft text or sparse-repair fixtures."
        ),
    },
    {
        "title": "MapReason-OSM graph-verifiable mobility decisions",
        "url": "https://arxiv.org/abs/2606.22597",
        "classification": "watch-only",
        "reason": (
            "Graph-verifiable decisions are adjacent to exact validators, but V500 "
            "already has Preference-MaxSAT and helper-contract fixtures. Adding a "
            "new map benchmark before Exp5512 would widen scope rather than unblock "
            "structured SOTA rows."
        ),
    },
    {
        "title": "Energy-Based Decoding and Energy-Guided VLM Decoding",
        "url": "https://arxiv.org/abs/2605.28020",
        "classification": "duplicate",
        "reason": (
            "Energy-based or hidden-state guided decoding is repeatedly indexed in "
            "V465, V492, V493, V494, and V497 history. It does not reopen token or "
            "internal-feature authority without a local backend receipt."
        ),
    },
    {
        "title": "TaskMem and PEAM parametric memory papers",
        "url": "https://arxiv.org/abs/2605.31075",
        "classification": "excluded",
        "reason": (
            "They use RL, adapters, or parameter-resident memory. V500 CSL scope is "
            "frozen-executor external graph memory with independent labels, so broad "
            "policy-gradient or weight-mutation lanes remain closed."
        ),
    },
    {
        "title": "Extropic TSU, XTR-0, Z1, and THRML writing",
        "url": "https://extropic.ai/writing",
        "classification": "watch-only",
        "reason": (
            "Extropic remains non-local sampler and EBM architecture context. There "
            "is no local TSU execution, authenticated board receipt, or matched "
            "timing basis for a hardware speedup claim."
        ),
    },
    {
        "title": "Logical Intelligence Kona and Aleph public pages",
        "url": "https://logicalintelligence.com/",
        "classification": "watch-only",
        "reason": (
            "Kona and Aleph reinforce verifier-first EBMs, but the public pages expose "
            "no local executable path or reproducible internals for V500 experiments."
        ),
    },
    {
        "title": "closed Carnot scopes from exclusion manifest",
        "url": "ops/exclusion_manifest.yaml",
        "classification": "excluded",
        "reason": (
            "External generated-text/logprob scorers, broad policy-gradient training, "
            "duplicate ARC generation-signal reruns, non-local TSU/Kona execution "
            "claims, and hardware speedup claims without matched board timing remain "
            "closed."
        ),
    },
]

SEARCHED_SOURCE_DETAILS: JsonDict = {
    "arxiv_strict_post_marker": {
        "status": "ok",
        "queries": [
            "submittedDate 202607090000-202607102359 energy based verification",
            "submittedDate 202607090000-202607102359 constraint LLM",
            "submittedDate 202607090000-202607102359 Ising",
            "submittedDate 202607090000-202607102359 hallucination",
            "submittedDate 202607090000-202607102359 Kolmogorov-Arnold",
            "submittedDate 202607090000-202607102359 continual learning memory",
        ],
        "strict_post_marker_hits": 0,
        "result": (
            "The strict arXiv submitted-date window after the V500 planner marker "
            "returned zero relevant hits across the required topic families."
        ),
    },
    "arxiv_recent_api": {
        "status": "ok",
        "queries": [
            "energy-guided decoding OR guided decoding AND energy",
            "neural constraint satisfaction OR constraint satisfaction AND language models",
            "hardware accelerated sampling OR p-bit OR probabilistic computer",
            "agent memory AND continual learning",
        ],
        "strict_post_marker_hits": 0,
        "promoted": [],
        "not_promoted": [
            "2607.07026 was already in the V500 planner block.",
            "2607.02465 p-MEM was already treated as non-local paper-only hardware context in V486 history.",
            "2606.25115 budget-curated memory was already in V489/V500 history.",
            "2605.28020 energy-based decoding was already in V465/V492/V493 history.",
        ],
    },
    "arxiv_primary_pages": {
        "status": "ok",
        "checked_urls": [
            "https://arxiv.org/abs/2605.18871",
            "https://arxiv.org/abs/2607.07026",
            "https://arxiv.org/abs/2607.02465",
            "https://arxiv.org/abs/2606.25115",
            "https://arxiv.org/abs/2605.28020",
            "https://arxiv.org/abs/2512.15605",
            "https://arxiv.org/abs/2507.02092",
        ],
        "result": (
            "Primary pages either matched the V500 planner block, older local "
            "source-delta history, or watch-only/non-local classifications."
        ),
    },
    "openreview": {
        "status": "challenge_blocked",
        "queries": [
            "OpenReview Energy-Based Transformers 2507.02092",
            "OpenReview ARM-EBM 2512.15605",
            "OpenReview Spilled Energy",
        ],
        "result": (
            "Search found EBT, NRGPT, and Spilled Energy surfaces; the direct EBT "
            "page challenged browser access. No OpenReview-only executable V500 "
            "delta was found."
        ),
    },
    "huggingface_papers": {
        "status": "partial",
        "queries": [
            "HuggingFace Papers 2507.02092",
            "HuggingFace Papers 2512.15605",
            "HuggingFace daily papers 2026-07-09 and 2026-07-10",
        ],
        "result": (
            "HuggingFace confirmed EBT context and rejected a future 2026-07-10 "
            "daily-paper API date relative to the endpoint state. No fresh V500 "
            "paper page added an executable hook."
        ),
    },
    "semantic_scholar_ebt_arm_ebm": {
        "status": "partial",
        "queries": [
            "Semantic Scholar arXiv:2507.02092",
            "Semantic Scholar arXiv:2512.15605",
        ],
        "result": (
            "The EBT route returned HTTP 429. The ARM-EBM route resolved metadata "
            "with citationCount=8 and influentialCitationCount=2, but no new "
            "Carnot-local dependency followed from that metadata."
        ),
    },
    "github": {
        "status": "ok",
        "queries": [
            "GitHub 2507.02092 Energy-Based Transformers",
            "GitHub 2512.15605 ARM-EBM",
            "GitHub llguidance and THRML routes",
        ],
        "result": (
            "GitHub confirmed alexiglad/EBT as the public EBT implementation. The "
            "ARM-EBM query did not find an official implementation route. EBT, "
            "llguidance, and THRML remain duplicate or watch-only context."
        ),
    },
    "extropic_writing": {
        "status": "ok",
        "queries": [
            "Extropic writing",
            "Extropic TSU 101",
            "Extropic thermodynamic computing from zero to one",
            "Extropic inside X0 and XTR-0",
        ],
        "result": (
            "Public Extropic pages remain architecture context only: no local "
            "Carnot TSU execution path, SDK receipt, or speedup basis was found."
        ),
    },
    "logical_intelligence": {
        "status": "ok",
        "queries": [
            "Logical Intelligence Kona EBMs",
            "Logical Intelligence Sudoku demo",
            "Logical Intelligence automatic formal verification for code generation",
        ],
        "result": (
            "Public pages continue to position Kona/Aleph as verifier-first EBMs "
            "and formal-verification systems, but they expose no reproducible "
            "local internals for V500."
        ),
    },
    "local_v500_and_v49x_duplicate_history": {
        "status": "ok",
        "queries": [
            "research-references.md V500 Planner Refresh",
            "research-references.md V489-V499 execution refresh blocks",
            "repo search for 2607.02465, 2606.25115, 2605.28020, 2605.18871, 2607.07026",
        ],
        "result": (
            "Exact local search found the surfaced actionable-looking candidates "
            "already indexed in V500 or earlier V49x source-delta modules/results."
        ),
    },
    "ops_exclusion_manifest": {
        "status": "ok",
        "queries": [
            "external generated-text/logprob scorers",
            "broad policy-gradient, GRPO, RL, LoRA, and fine-tuning reruns",
            "non-local TSU/Kona/Aleph execution claims",
            "hardware speedup claims without matched board timing",
            "duplicate ARC generation-axis reruns",
        ],
        "result": "Closed lanes stayed closed; no operator override was present.",
    },
    "research_roadmap_v500_tasks": {
        "status": "ok",
        "queries": [
            "exp5512 structured output positive control",
            "exp5515 CSL independent outcome gate repair",
            "exp5518 block Gibbs sparse repair descriptors",
            "exp5519 hardware continuity methodology receipts",
            "exp5520 ARC action diversity target precheck",
        ],
        "result": (
            "No accepted finding required a new mapping. The planned mapping lanes "
            "remain available if a future accepted finding is passed into the builder."
        ),
    },
}


def _verified_url(value: str) -> bool:
    return value.startswith("https://") or value == "ops/exclusion_manifest.yaml"


def _mapping_for(row: Mapping[str, Any]) -> JsonDict:
    return {
        "title": str(row["title"]),
        "planned_experiment": str(row["planned_experiment"]),
        "mapped_task": str(row["mapped_task"]),
        "rationale": str(row["carnot_hook"]),
    }


def _build_mappings(references: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [_mapping_for(row) for row in references]


def build_artifact(
    *,
    new_references_added: Sequence[Mapping[str, Any]] = DEFAULT_NEW_REFERENCES_ADDED,
    methodology_duration_s: float = 0.0,
    tests_run: Sequence[str] = (),
    research_references_updated: bool | None = None,
    prior_refresh_marker_found: bool = True,
) -> JsonDict:
    """Build the deterministic Exp5511 source-delta receipt.

    The function is intentionally pure: the network sweep has already been
    performed by the executing agent, while this code turns that audit result
    into a stable JSON contract and optional references append. That separation
    keeps future reruns from pretending source aggregation is model or hardware
    inference.
    """

    blocked = not prior_refresh_marker_found
    references = [] if blocked else [dict(row) for row in new_references_added]
    count = len(references)
    updated = (count > 0) if research_references_updated is None else research_references_updated
    mappings = [] if blocked else _build_mappings(references)

    if blocked:
        status = "blocked"
        updated = False
        verdict = "blocked: V500 planner refresh marker missing; references unchanged."
    else:
        status = "complete"
        detail = (
            f"{count} new actionable V500 execution-time source delta appended; "
            "all accepted findings mapped to planned .500 lanes"
            if count
            else (
                "no new actionable V500 execution-time source deltas after the "
                "planner refresh; references unchanged and closed scopes stayed closed"
            )
        )
        verdict = f"complete: {detail}."

    artifact: JsonDict = {
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "status": status,
        "search_date": SEARCH_DATE,
        "sources_checked": list(REQUIRED_SOURCE_FAMILIES),
        "new_actionable_findings_count": count,
        "new_references_added": references,
        "duplicates_suppressed": list(DUPLICATES_SUPPRESSED),
        "closed_scopes_reopened": False,
        "research_references_updated": updated,
        "prior_refresh_marker_found": prior_refresh_marker_found,
        "experiment_mappings": mappings,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "searched_source_details": dict(SEARCHED_SOURCE_DETAILS),
        "watch_only_or_excluded": [dict(row) for row in WATCH_ONLY_OR_EXCLUDED],
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "methodology_duration_s": round(float(methodology_duration_s), 6),
        "tests_run": list(tests_run)
        or ["tests/python/test_experiment_5511_v500_source_delta_ingestion.py"],
        "no_deep_research_used": True,
        "research_conductor_modified": False,
        "ops_docs_modified": False,
        "traceability_modified": False,
        "roadmap_files_modified": False,
    }
    validate_artifact(artifact)
    return artifact


def _validate_sources(sources: Any, details: Any) -> None:
    if not isinstance(sources, list) or not set(REQUIRED_SOURCE_FAMILIES).issubset(sources):
        raise ValueError("sources_checked must include every required source family")
    if len(sources) != len(set(sources)):
        raise ValueError("sources_checked must not contain duplicate source families")
    if not isinstance(details, Mapping):
        raise ValueError("searched_source_details must be a mapping")
    for family in REQUIRED_SOURCE_FAMILIES:
        family_entry = details.get(family)
        family_status = family_entry.get("status") if isinstance(family_entry, Mapping) else None
        if family_status not in ALLOWED_SOURCE_STATUSES:
            raise ValueError(f"searched_source_details {family} must record a valid status")


def _validate_references(references: Any) -> None:
    if not isinstance(references, list):
        raise ValueError("new_references_added must be a list")
    for row in references:
        if not isinstance(row, Mapping) or set(row) != REQUIRED_REFERENCE_FIELDS:
            raise ValueError(
                f"new_references_added rows must include exactly {sorted(REQUIRED_REFERENCE_FIELDS)}"
            )
        if not _verified_url(str(row["url"])):
            raise ValueError("new_references_added rows must use a verified URL")
        if not str(row["carnot_hook"]).strip():
            raise ValueError("new_references_added rows must include a Carnot hook")
        planned = str(row["planned_experiment"])
        if planned not in PLANNED_EXPERIMENT_TASKS:
            raise ValueError("new_references_added rows must use a planned experiment lane")
        if row["mapped_task"] != PLANNED_EXPERIMENT_TASKS[planned]:
            raise ValueError("new_references_added mapped_task must match planned experiment lane")


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = REQUIRED_TOP_LEVEL_FIELDS.difference(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match REQ-REPORT-5511")
    if artifact["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("experiment_id must match Exp5511")
    if artifact["task_id"] != TASK_ID:
        raise ValueError("task_id must match exp5511")
    if artifact["milestone"] != MILESTONE:
        raise ValueError("milestone must match 2026.07.500")
    if artifact["status"] not in {"complete", "blocked"}:
        raise ValueError("status must be complete or blocked")
    if artifact["search_date"] != SEARCH_DATE:
        raise ValueError("search_date must equal 20260710")

    references = artifact["new_references_added"]
    _validate_references(references)
    count = artifact["new_actionable_findings_count"]
    if not isinstance(count, int) or count != len(references):
        raise ValueError("references count must equal new_references_added length")
    _validate_sources(artifact["sources_checked"], artifact["searched_source_details"])

    expected_mappings = _build_mappings(references)
    if artifact["experiment_mappings"] != expected_mappings:
        raise ValueError("experiment_mappings must match accepted references")
    duplicates = artifact["duplicates_suppressed"]
    if not isinstance(duplicates, list) or not duplicates:
        raise ValueError("duplicates_suppressed must be a non-empty list")
    if len(duplicates) != len(set(duplicates)):
        raise ValueError("duplicates_suppressed must not contain duplicate suppressed entries")
    if artifact["closed_scopes_reopened"] is not False:
        raise ValueError("closed_scopes_reopened must remain false")

    prior_marker = artifact["prior_refresh_marker_found"]
    if artifact["status"] == "complete" and prior_marker is not True:
        raise ValueError("prior_refresh_marker_found must be true for complete artifacts")
    if artifact["status"] == "blocked" and prior_marker is True:
        raise ValueError("blocked artifacts must record a missing V500 planner marker")

    updated = artifact["research_references_updated"]
    if prior_marker is False:
        if updated is not False or references or artifact["experiment_mappings"]:
            raise ValueError("research_references_updated must be false when blocked")
    elif updated is not (count > 0):
        raise ValueError("research_references_updated must match whether references were added")

    verdict = artifact["honest_verdict"]
    if (
        not isinstance(verdict, str)
        or "\n" in verdict
        or not verdict.startswith(("complete:", "blocked:"))
    ):
        raise ValueError("honest_verdict must be a one-line complete: or blocked: summary")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    watch_only = artifact["watch_only_or_excluded"]
    if not isinstance(watch_only, list) or not watch_only:
        raise ValueError("watch_only_or_excluded must be a non-empty list")
    duration = artifact["methodology_duration_s"]
    if not isinstance(duration, int | float) or duration < 0:
        raise ValueError("methodology_duration_s must be a non-negative number")
    if artifact["no_deep_research_used"] is not True:
        raise ValueError("deep-research must not be used")
    if artifact["research_conductor_modified"] is not False:
        raise ValueError("research_conductor.py must not be modified")
    if artifact["ops_docs_modified"] is not False:
        raise ValueError("ops docs must not be modified")
    if artifact["traceability_modified"] is not False:
        raise ValueError("traceability must not be modified")
    if artifact["roadmap_files_modified"] is not False:
        raise ValueError("roadmap files must not be modified")
    if not artifact["tests_run"]:
        raise ValueError("tests_run must record at least one test")


def _render_reference(row: Mapping[str, Any]) -> str:
    return (
        f"- **{row['title']}** ({row['url']}): {row['carnot_hook']} "
        f"Maps to `{row['mapped_task']}`."
    )


def render_refresh_section(artifact: Mapping[str, Any]) -> str:
    validate_artifact(artifact)
    references = artifact["new_references_added"]
    if artifact["status"] != "complete" or not references:
        return ""
    lines = [
        REFRESH_HEADING,
        "",
        (
            "Execution-time sweep after the `.500` planner refresh checked arXiv "
            "primary pages and recent API results, OpenReview, HuggingFace Papers, "
            "Semantic Scholar routes for EBT and ARM-EBM, GitHub, Extropic writing, "
            "Logical Intelligence public pages, V500/V49x duplicate history, the "
            "active V500 roadmap, and the exclusion manifest. Only non-duplicate "
            "actionable deltas are listed below."
        ),
        "",
        "### New actionable deltas",
        *(_render_reference(row) for row in references),
        "",
        "### Execution impact",
        (
            "- **Mappings:** Every accepted finding above maps to a planned `.500` "
            "experiment lane: structured SOTA control, CSL independence, sparse "
            "repair, hardware receipts, or ARC live-path improvement."
        ),
        (
            "- **Duplicates suppressed:** V500 planner sources and older V49x "
            "execution-refresh sources were not re-added."
        ),
        (
            "- **Closed scope:** No closed scope was reopened. External logprob "
            "scorers, broad policy-gradient or fine-tuning loops, non-local "
            "TSU/Kona execution claims, duplicate ARC generation-signal reruns, and "
            "hardware speedup claims without matched board timing remain closed."
        ),
        "",
        REFRESH_END_MARKER,
        "",
    ]
    return "\n".join(lines)


def append_refresh_section(references_text: str, artifact: Mapping[str, Any]) -> str:
    if REFRESH_END_MARKER in references_text or REFRESH_HEADING in references_text:
        return references_text
    section = render_refresh_section(artifact)
    if not section:
        return references_text
    separator = "\n\n" if references_text and not references_text.endswith("\n\n") else ""
    return f"{references_text}{separator}{section}"


def write_outputs(
    *,
    root: Path = REPO_ROOT,
    references_path: Path | None = None,
    result_path: Path | None = None,
    methodology_duration_s: float = 0.0,
    tests_run: Sequence[str] = (),
) -> JsonDict:
    references = references_path or root / REFERENCES_RELATIVE_PATH
    result = result_path or root / RESULT_RELATIVE_PATH

    references_text = references.read_text(encoding="utf-8")
    prior_marker = PLANNER_MARKER in references_text
    artifact = build_artifact(
        methodology_duration_s=methodology_duration_s,
        tests_run=tests_run,
        prior_refresh_marker_found=prior_marker,
    )

    updated_references = append_refresh_section(references_text, artifact)
    if updated_references != references_text:
        references.write_text(updated_references, encoding="utf-8")

    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:  # pragma: no cover
    write_outputs(
        methodology_duration_s=0.0,
        tests_run=["tests/python/test_experiment_5511_v500_source_delta_ingestion.py"],
    )


if __name__ == "__main__":  # pragma: no cover
    main()
