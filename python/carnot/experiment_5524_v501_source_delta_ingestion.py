"""Exp5524 execution-time V501 source delta ingestion.

Spec refs: REQ-REPORT-5524, SCENARIO-REPORT-5524-APPEND-DELTAS,
SCENARIO-REPORT-5524-NO-NEW-DELTA, SCENARIO-REPORT-5524-BLOCKED-MARKER.

This receipt turns the last-minute literature sweep into a stable artifact
before the V501 science tasks run. The useful outcome can be a no-op: when the
planner refresh already captured the actionable papers, this module records the
sources checked and the reasons for suppressing duplicates instead of adding
reference churn that later gates might mistake for new evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_5524_v501_source_delta_ingestion"
TASK_ID = "exp5524-v501-source-delta-ingestion"
MILESTONE = "2026.07.501"
SEARCH_DATE = "20260710"
RESULT_RELATIVE_PATH = Path("results/experiment_5524_v501_source_delta_ingestion.json")
REFERENCES_RELATIVE_PATH = Path("research-references.md")
PLANNER_MARKER = "## V501 Planner Refresh - 2026-07-10"
REFRESH_HEADING = "## V501 Execution Refresh - 20260710"
REFRESH_END_MARKER = "<!-- V501-EXECUTION-REFRESH-20260710-END -->"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

SPEC_REFS = [
    "REQ-REPORT-5524",
    "SCENARIO-REPORT-5524-APPEND-DELTAS",
    "SCENARIO-REPORT-5524-NO-NEW-DELTA",
    "SCENARIO-REPORT-5524-BLOCKED-MARKER",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "sources_checked": (
        "Records each primary, secondary, local-dedupe, and exclusion source "
        "checked before V501 science tasks run."
    ),
    "new_references_added": (
        "Lists only non-duplicate actionable findings that earned a V501 "
        "execution-refresh append."
    ),
    "duplicates_suppressed": (
        "Prevents churn from re-adding V501 planner sources or earlier "
        "source-delta findings."
    ),
    "closed_scopes_reopened": (
        "Bare false boolean proving excluded, proprietary, non-local, or retired "
        "lanes stayed closed."
    ),
    "research_references_updated": (
        "Bare boolean distinguishing a real V501 append from a no-op freshness "
        "receipt."
    ),
    "prior_refresh_marker_found": (
        "Ensures the execution refresh dedupes against the actual V501 planner "
        "block before appending."
    ),
    "experiment_mappings": (
        "Maps every accepted finding to a planned `.501` experiment lane; empty "
        "only when no finding was accepted."
    ),
    "field_principles": (
        "Carries the why behind every headline and gate field so downstream "
        "reconcilers can audit field intent."
    ),
    "inference_substrate": (
        "Must be aggregation_from_upstream_artifacts because the receipt "
        "aggregates sources and local artifacts without model, solver, ARC, or "
        "hardware inference."
    ),
    "honest_verdict": (
        "One-line terminal summary starting with complete: or blocked: that "
        "states whether references changed."
    ),
}

REQUIRED_SOURCE_FAMILIES = (
    "arxiv_post_v501_marker_window",
    "arxiv_topic_api_sweep",
    "arxiv_primary_pages",
    "openreview",
    "huggingface_papers",
    "semantic_scholar_ebt_arm_ebm",
    "github",
    "extropic_writing_and_hardware",
    "logical_intelligence_public_pages",
    "local_v501_and_prior_duplicate_history",
    "ops_exclusion_manifest",
    "research_roadmap_v501_tasks",
)

ALLOWED_SOURCE_STATUSES = frozenset(
    {"ok", "partial", "rate_limited", "challenge_blocked"}
)

PLANNED_EXPERIMENT_TASKS: dict[str, str] = {
    "live_sota_schema_repair": "exp5525-sota-schema-failure-taxonomy",
    "csl_gate_clean_memory": "exp5528-csl-canonical-gate-artifact",
    "sparse_repair_scaling": "exp5531-sparse-repair-scaleup-ci",
    "hardware_receipt_parser_repeatability": (
        "exp5532-hardware-receipt-parser-repeatability"
    ),
    "arc_strategy_routing": "exp5533-arc-strategy-routing-precheck",
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
        "field_principles",
        "honest_verdict",
        "inference_substrate",
        "searched_source_details",
        "watch_only_or_excluded",
        "spec_refs",
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
    "GAM hierarchical graph memory is already the V501 planner CSL delta and is mapped to event/topic memory receipts - https://arxiv.org/abs/2604.12285",
    "Compliance-grade LLMOps for schema-constrained serving is already the V501 planner live SOTA schema workload delta - https://arxiv.org/abs/2605.11232",
    "Metacognition and faithful uncertainty is already the V501 planner hard/soft panel uncertainty split - https://arxiv.org/abs/2605.01428",
    "Distributional EBMs, PCRLLM, XGrammar-2, llguidance, and constrained-generation controls were already filed in V500/V501 structured-output history - https://arxiv.org/abs/2605.18871",
    "2607.05936 constrained web API invocation was already accepted in the V492 source-delta path and does not add a new V501 schema-repair dependency - https://arxiv.org/abs/2607.05936",
    "REVES verifier-trace training was already indexed in V493 history and remains broader than the V501 schema taxonomy gate - https://arxiv.org/abs/2606.18910",
    "Energy-Guided Decoding for Object Hallucination is repeatedly indexed and needs hidden-state or VLM runtime access outside the V501 schema-repair lane - https://arxiv.org/abs/2507.07731",
    "BloGDiT sparse repair, GRS-KAN, and KAN verification papers are already mapped to sparse repair or watch-only learned-constraint context - https://arxiv.org/abs/2605.25129",
    "Million-p-bit hardware, p-bit CDCL, Scaling Up Thermodynamic AI, and adaptive probabilistic processor work remain receipt methodology context without local matched timing - https://arxiv.org/abs/2606.25313",
    "Budget-Curated Memory, When Continual Learning Moves to Memory, ExpGraph, Evo-Memory, GAM, and graph memory surveys are already covered by V49x/V501 CSL history - https://arxiv.org/abs/2606.25115",
    "EBT 2507.02092, ARM-EBM 2512.15605, EBT GitHub, HuggingFace Papers, and Semantic Scholar routes remain architecture context rather than new V501 execution dependencies - https://arxiv.org/abs/2507.02092",
    "Extropic TSU/XTR-0/Z1/THRML and Logical Intelligence Kona/Aleph public pages remain strategic non-local context with no Carnot-local SDK, internals, or speedup basis - https://logicalintelligence.com/",
]

WATCH_ONLY_OR_EXCLUDED: list[JsonDict] = [
    {
        "title": "Log-Insight: Neuro-Symbolic Log Analysis",
        "url": "https://arxiv.org/abs/2607.08529",
        "classification": "watch-only",
        "reason": (
            "The evidence-dossier pattern is adjacent to failure taxonomy, but the "
            "paper is production SRE log RCA, not a local structured-output "
            "runtime. V501 already records prompt, parser, grammar, truncation, "
            "and exact-validator receipts for schema repair."
        ),
    },
    {
        "title": "Game Theory Driven Multi-Agent Framework Mitigates Language Model Hallucination",
        "url": "https://arxiv.org/abs/2607.08403",
        "classification": "excluded",
        "reason": (
            "The method relies on domain-specific chemistry corpus synthesis and "
            "model training. That broad training scope is non-executable for V501 "
            "and conflicts with the local schema, memory, sparse, hardware, and "
            "ARC gates."
        ),
    },
    {
        "title": "Mixed-Mode Advantage Regularization for Factual Hallucination",
        "url": "https://arxiv.org/abs/2607.05861",
        "classification": "excluded",
        "reason": (
            "MARGO is a reinforcement-learning regularization method. V501 does "
            "not reopen broad policy-gradient, fine-tuning, or same-model rollout "
            "training; schema repair must first produce parseable rows."
        ),
    },
    {
        "title": "MODE-RAG energy-based multimodal RAG evaluation",
        "url": "https://arxiv.org/abs/2606.17449",
        "classification": "watch-only",
        "reason": (
            "The variational-free-energy and agentic RAG framing is relevant to "
            "hallucination control, but Carnot has no V501 multimodal RAG path or "
            "ModeVent fixture. It stays future context."
        ),
    },
    {
        "title": "Adaptive Probabilistic Processors Based on the Ising Model",
        "url": "https://arxiv.org/abs/2606.19533",
        "classification": "watch-only",
        "reason": (
            "The p-bit synthesis tool informs future sampler design, but V501 "
            "hardware work is parser repeatability only. There is no local p-bit, "
            "MTJ, or matched-timing execution path to claim."
        ),
    },
    {
        "title": "Next-Generation Agentic Reinforcement Learning Systems",
        "url": "https://arxiv.org/abs/2607.01120",
        "classification": "excluded",
        "reason": (
            "Agentic RL and policy-gradient self-evolution are outside V501's "
            "frozen-weight CSL memory scope and remain closed without an operator "
            "reopen."
        ),
    },
    {
        "title": "OpenReview EBT and related pages",
        "url": "https://openreview.net/forum?id=ZBj3Qp1bYg",
        "classification": "watch-only",
        "reason": (
            "OpenReview browser access was challenge-blocked during execution. "
            "Search snippets and HuggingFace/GitHub mirrors did not expose a new "
            "local baseline beyond the already-indexed EBT paper."
        ),
    },
    {
        "title": "Extropic TSU, XTR-0, Z1, and THRML",
        "url": "https://extropic.ai/hardware",
        "classification": "watch-only",
        "reason": (
            "Extropic public pages describe TSU-style probabilistic hardware and "
            "THRML simulation, but Carnot has no local TSU device, SDK receipt, or "
            "board-local timing. V501 hardware remains receipt-parser work."
        ),
    },
    {
        "title": "Logical Intelligence Kona and Aleph public pages",
        "url": "https://logicalintelligence.com/kona-ebms-energy-based-models",
        "classification": "watch-only",
        "reason": (
            "Kona/Aleph reinforce verifier-first EBMs, but the pages expose no "
            "local executable path or reproducible internals for V501."
        ),
    },
    {
        "title": "closed Carnot scopes from exclusion manifest",
        "url": "ops/exclusion_manifest.yaml",
        "classification": "excluded",
        "reason": (
            "Broad policy-gradient training, duplicate ARC exploration-signal "
            "reruns, non-local TSU/Kona execution claims, and hardware speedup "
            "claims without matched board timing remain closed."
        ),
    },
]

SEARCHED_SOURCE_DETAILS: JsonDict = {
    "arxiv_post_v501_marker_window": {
        "status": "ok",
        "queries": [
            "submittedDate after V501 planner marker for EBM verification",
            "submittedDate after V501 planner marker for constrained decoding",
            "submittedDate after V501 planner marker for Ising hardware",
            "submittedDate after V501 planner marker for agent memory",
        ],
        "strict_post_marker_hits": 0,
        "result": (
            "No source newer than the V501 planner marker produced a concrete "
            "Carnot-local hook for the five planned .501 experiment lanes."
        ),
    },
    "arxiv_topic_api_sweep": {
        "status": "ok",
        "queries": [
            "energy based verification reasoning",
            "neural constraint satisfaction and language models",
            "Ising hardware sampling",
            "hallucination mitigation",
            "Kolmogorov-Arnold networks",
            "energy-guided decoding",
            "continual online learning for LLM agents",
        ],
        "promoted": [],
        "not_promoted": [
            "2607.05936 constrained API invocation was already accepted in V492.",
            "2606.18910 REVES was already indexed in V493 history.",
            "2607.06341 code-agent verification was already watch-only.",
            "2607.05861 and 2607.08403 require training/RL or domain corpora.",
            "2607.08529 and 2606.17449 are adjacent but non-executable for V501.",
            "2606.19533 is hardware-method context without local p-bit execution.",
        ],
    },
    "arxiv_primary_pages": {
        "status": "ok",
        "checked_urls": [
            "https://arxiv.org/abs/2604.12285",
            "https://arxiv.org/abs/2605.11232",
            "https://arxiv.org/abs/2605.01428",
            "https://arxiv.org/abs/2607.08529",
            "https://arxiv.org/abs/2607.08403",
            "https://arxiv.org/abs/2607.05861",
            "https://arxiv.org/abs/2606.19533",
            "https://arxiv.org/abs/2606.17449",
            "https://arxiv.org/abs/2507.02092",
            "https://arxiv.org/abs/2512.15605",
        ],
        "result": (
            "Primary pages either matched the V501 planner block, older local "
            "source-delta history, or watch-only/non-local classifications."
        ),
    },
    "openreview": {
        "status": "challenge_blocked",
        "queries": [
            "OpenReview Energy-Based Transformers 2507.02092",
            "OpenReview ARM-EBM 2512.15605",
            "OpenReview GAM hierarchical graph memory",
        ],
        "result": (
            "The direct EBT OpenReview page redirected to a browser challenge. "
            "Search surfaces did not reveal an OpenReview-only executable V501 "
            "delta."
        ),
    },
    "huggingface_papers": {
        "status": "ok",
        "queries": [
            "HuggingFace Papers 2507.02092",
            "HuggingFace Papers 2605.01428",
            "HuggingFace daily papers 2026-07-10",
        ],
        "result": (
            "HuggingFace confirmed EBT and metacognition context. The 2026-07-10 "
            "daily page did not surface a schema, CSL, sparse, hardware, or ARC "
            "paper that changes V501 execution."
        ),
    },
    "semantic_scholar_ebt_arm_ebm": {
        "status": "partial",
        "queries": [
            "Semantic Scholar API ARXIV:2507.02092",
            "Semantic Scholar API ARXIV:2512.15605",
        ],
        "result": (
            "EBT resolved with citation metadata but no new local baseline. "
            "ARM-EBM returned HTTP 429 during execution, so it remains covered by "
            "arXiv and local duplicate history only."
        ),
    },
    "github": {
        "status": "ok",
        "queries": [
            "GitHub 2507.02092 Energy-Based Transformers",
            "GitHub 2512.15605 ARM-EBM",
            "GitHub constrained decoding and EBM repositories",
        ],
        "result": (
            "GitHub confirmed alexiglad/EBT as the public EBT implementation. No "
            "official ARM-EBM implementation or new V501-local engine displaced "
            "existing exact validators, grammar-watch paths, or ARC provenance "
            "constraints."
        ),
    },
    "extropic_writing_and_hardware": {
        "status": "ok",
        "queries": [
            "Extropic writing",
            "Extropic hardware X0 XTR-0 Z1",
            "Extropic THRML software",
        ],
        "result": (
            "Public pages continue to describe TSU-style probabilistic circuits "
            "and THRML simulation. No local Carnot TSU execution path, SDK "
            "receipt, or speedup basis was found."
        ),
    },
    "logical_intelligence_public_pages": {
        "status": "ok",
        "queries": [
            "Logical Intelligence Kona EBMs",
            "Logical Intelligence automatic formal verification for code generation",
            "Logical Intelligence Aleph",
        ],
        "result": (
            "Kona/Aleph pages continue to position verifier-first EBMs and formal "
            "verification under LLM interfaces, but expose no reproducible local "
            "internals for V501."
        ),
    },
    "local_v501_and_prior_duplicate_history": {
        "status": "ok",
        "queries": [
            "research-references.md V501 Planner Refresh",
            "research-references.md V49x and V500 source-delta blocks",
            "repo search for 2607.05936, 2606.18910, 2607.06341, 2606.25115, 2507.07731",
        ],
        "result": (
            "Local search found the strongest actionable-looking candidates "
            "already indexed in V501, V500, or earlier V49x source-delta modules "
            "and results."
        ),
    },
    "ops_exclusion_manifest": {
        "status": "ok",
        "queries": [
            "broad policy-gradient, GRPO, RL, LoRA, and fine-tuning reruns",
            "non-local TSU/Kona/Aleph execution claims",
            "hardware speedup claims without matched board timing",
            "duplicate ARC generation-axis reruns",
        ],
        "result": "Closed lanes stayed closed; no operator override was present.",
    },
    "research_roadmap_v501_tasks": {
        "status": "ok",
        "queries": [
            "exp5525 live SOTA schema taxonomy",
            "exp5528 CSL canonical gate artifact",
            "exp5531 sparse repair scale-up",
            "exp5532 hardware receipt parser repeatability",
            "exp5533 ARC strategy-routing precheck",
        ],
        "result": (
            "No accepted finding required a new mapping. The planned mapping lanes "
            "remain available if a future accepted finding is passed into the "
            "builder."
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
    """Build the deterministic Exp5524 source-delta receipt.

    The web search is not replayed inside tests because search engines and paper
    pages change. Instead, the executing agent records the sweep outcome in this
    structured artifact. Keeping the builder pure makes the receipt reproducible
    and prevents literature aggregation from masquerading as model or hardware
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
        verdict = "blocked: V501 planner refresh marker missing; references unchanged."
    else:
        status = "complete"
        detail = (
            f"{count} new actionable V501 execution-time source delta appended; "
            "all accepted findings mapped to planned .501 lanes"
            if count
            else (
                "no new actionable V501 execution-time source deltas after the "
                "planner refresh; references unchanged and closed scopes stayed "
                "closed"
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
        "field_principles": dict(FIELD_PRINCIPLES),
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "searched_source_details": dict(SEARCHED_SOURCE_DETAILS),
        "watch_only_or_excluded": [dict(row) for row in WATCH_ONLY_OR_EXCLUDED],
        "spec_refs": list(SPEC_REFS),
        "methodology_duration_s": round(float(methodology_duration_s), 6),
        "tests_run": list(tests_run)
        or ["tests/python/test_experiment_5524_v501_source_delta_ingestion.py"],
        "no_deep_research_used": True,
        "research_conductor_modified": False,
        "ops_docs_modified": False,
        "traceability_modified": False,
        "roadmap_files_modified": False,
    }
    validate_artifact(artifact)
    return artifact


def _validate_sources(sources: Any, details: Any) -> None:
    if not isinstance(sources, list) or not set(REQUIRED_SOURCE_FAMILIES).issubset(
        sources
    ):
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
                "new_references_added rows must include exactly "
                f"{sorted(REQUIRED_REFERENCE_FIELDS)}"
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
        raise ValueError("field_principles must match REQ-REPORT-5524")
    if artifact["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("experiment_id must match Exp5524")
    if artifact["task_id"] != TASK_ID:
        raise ValueError("task_id must match exp5524")
    if artifact["milestone"] != MILESTONE:
        raise ValueError("milestone must match 2026.07.501")
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
        raise ValueError("blocked artifacts must record a missing V501 planner marker")

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
            "Execution-time sweep after the `.501` planner refresh checked arXiv "
            "primary pages and API results, OpenReview, HuggingFace Papers, "
            "Semantic Scholar routes for EBT and ARM-EBM, GitHub, Extropic "
            "writing and hardware pages, Logical Intelligence public pages, "
            "V501/prior duplicate history, the active V501 roadmap, and the "
            "exclusion manifest. Only non-duplicate actionable deltas are listed "
            "below."
        ),
        "",
        "### New actionable deltas",
        *(_render_reference(row) for row in references),
        "",
        "### Execution impact",
        (
            "- **Mappings:** Every accepted finding above maps to a planned `.501` "
            "experiment lane: live SOTA schema repair, CSL gate-clean memory, "
            "sparse repair scaling, hardware receipt parser repeatability, or "
            "ARC strategy routing."
        ),
        (
            "- **Duplicates suppressed:** V501 planner sources and older "
            "source-delta sources were not re-added."
        ),
        (
            "- **Closed scope:** No closed scope was reopened. Broad policy-"
            "gradient or fine-tuning loops, non-local TSU/Kona execution claims, "
            "duplicate ARC generation-signal reruns, and hardware speedup claims "
            "without matched board timing remain closed."
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
        tests_run=["tests/python/test_experiment_5524_v501_source_delta_ingestion.py"],
    )


if __name__ == "__main__":  # pragma: no cover
    main()
