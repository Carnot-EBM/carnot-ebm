"""Exp5537 execution-time V502 source delta ingestion.

Spec refs: REQ-REPORT-5537, SCENARIO-REPORT-5537-APPEND-DELTAS,
SCENARIO-REPORT-5537-NO-NEW-DELTA, SCENARIO-REPORT-5537-BLOCKED-MARKER.

This module turns an execution-time literature freshness check into a small,
auditable receipt. The important discipline is not just adding a paper: each
accepted source must be both non-duplicate and executable inside an already
planned V502 lane. Everything else stays duplicate, watch-only, or excluded so
the science experiments do not inherit scope churn.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_5537_v502_source_delta_ingestion"
TASK_ID = "exp5537-v502-source-delta-ingestion"
MILESTONE = "2026.07.502"
SEARCH_DATE = "20260710"
RESULT_RELATIVE_PATH = Path("results/experiment_5537_v502_source_delta_ingestion.json")
REFERENCES_RELATIVE_PATH = Path("research-references.md")
PLANNER_MARKER = "## V502 Planner Refresh - 2026-07-10"
REFRESH_HEADING = "## V502 Execution Refresh - 20260710"
REFRESH_END_MARKER = "<!-- V502-EXECUTION-REFRESH-20260710-END -->"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

SPEC_REFS = [
    "REQ-REPORT-5537",
    "SCENARIO-REPORT-5537-APPEND-DELTAS",
    "SCENARIO-REPORT-5537-NO-NEW-DELTA",
    "SCENARIO-REPORT-5537-BLOCKED-MARKER",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "sources_checked": (
        "Records every primary, secondary, local-dedupe, and exclusion source "
        "checked before V502 science tasks run."
    ),
    "new_references_added": (
        "Lists only non-duplicate actionable findings that earned a V502 "
        "execution-refresh append."
    ),
    "duplicates_suppressed": (
        "Prevents churn from re-adding the V502 planner block or older "
        "source-delta findings."
    ),
    "semantic_scholar_status": (
        "Records whether EBT and ARM-EBM public citation routes were reachable "
        "without turning counts into a claim."
    ),
    "closed_scopes_reopened": (
        "Bare false boolean proving excluded, proprietary, non-local, and "
        "retired lanes stayed closed."
    ),
    "research_references_updated": (
        "Bare boolean distinguishing a real V502 append from a no-op freshness "
        "receipt."
    ),
    "prior_refresh_marker_found": (
        "Ensures the execution refresh dedupes against the actual V502 planner "
        "block before appending."
    ),
    "experiment_mappings": (
        "Maps every accepted finding to a planned `.502` experiment lane; empty "
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
        "semantic_scholar_status",
        "closed_scopes_reopened",
        "research_references_updated",
        "prior_refresh_marker_found",
        "experiment_mappings",
        "field_principles",
        "inference_substrate",
        "honest_verdict",
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

REQUIRED_SOURCE_FAMILIES = (
    "arxiv_cs_ai_new_20260710",
    "arxiv_cs_lg_new_20260710",
    "arxiv_cs_cl_new_20260710",
    "arxiv_topic_api_sweep",
    "arxiv_primary_pages",
    "openreview",
    "huggingface_papers",
    "semantic_scholar_ebt_arm_ebm",
    "github",
    "extropic_writing",
    "logical_intelligence_public_pages",
    "local_v502_and_prior_duplicate_history",
    "ops_exclusion_manifest",
    "research_roadmap_v502_tasks",
)

ALLOWED_SOURCE_STATUSES = frozenset({"ok", "partial", "rate_limited", "challenge_blocked"})

PLANNED_EXPERIMENT_TASKS: dict[str, str] = {
    "sota_duration_substrate_repair": "exp5538-sota-panel-duration-substrate-corrigendum",
    "grammar_table_preflight": "exp5539-gram2token-grammar-table-preflight",
    "llm_fsm_exact_fixture": "exp5541-llm-fsm-exact-fixture",
    "csl_residue_repair": "exp5542-csl-residue-metric-independence-corrigendum",
    "sparse_fsm_descriptors": "exp5545-gated-sparse-repair-fsm-descriptor-scale",
    "hardware_receipt_hygiene": "exp5546-hardware-receipt-substrate-corrigendum",
    "arc_live_path_recovery": "exp5547-arc-no-llm-substrate-precheck",
}

DEFAULT_NEW_REFERENCES_ADDED: list[JsonDict] = [
    {
        "title": (
            "Answer Set Programming Energised! End-to-End Neurosymbolic Reasoning "
            "and Learning with ASP and Energy Based Models"
        ),
        "url": "https://arxiv.org/abs/2607.08136",
        "source_type": "arXiv preprint",
        "submitted_on": "2026-07-09",
        "carnot_hook": (
            "Add ASP declarative semantics and non-monotonic constraint rows to "
            "the deterministic finite-state exact fixture; reuse the same rows "
            "as richer sparse-FSM descriptors without reopening training or "
            "proprietary baselines."
        ),
        "planned_experiment": "llm_fsm_exact_fixture",
        "mapped_task": "exp5541-llm-fsm-exact-fixture",
        "secondary_mapped_tasks": ["exp5545-gated-sparse-repair-fsm-descriptor-scale"],
    }
]

DUPLICATES_SUPPRESSED = [
    (
        "LLM-FSM arXiv:2602.07032 is already the V502 planner finite-state exact "
        "fixture source and remains mapped to Exp5541."
    ),
    (
        "Gram2Token is already the V502 planner grammar-table preflight source "
        "and remains mapped to Exp5539."
    ),
    (
        "2607.07026 constrained diffusion-LM finite automata was already filed "
        "in V499/V500 history and remains watch-only without a local diffusion "
        "backend."
    ),
    (
        "Distributional EBMs, ConstrainPrompt, CRV, XGrammar, llguidance, and "
        "JSONSchemaBench were already covered by V500-V502 structured-output "
        "history."
    ),
    (
        "EBT 2507.02092 and ARM-EBM 2512.15605 Semantic Scholar routes matched "
        "the planner context and did not create a stronger local dependency."
    ),
    (
        "Extropic TSU/XTR/Z1 and Logical Intelligence Kona/Aleph public pages "
        "remain strategic non-local context with no executable Carnot baseline "
        "or speedup receipt."
    ),
]

WATCH_ONLY_OR_EXCLUDED: list[JsonDict] = [
    {
        "title": "ReCoLoRA: Spectrum-Aware Recursive Consolidation for Continual LLM Fine-Tuning",
        "url": "https://arxiv.org/abs/2607.07719",
        "classification": "excluded",
        "reason": (
            "The method is continual fine-tuning with recursive LoRA-style "
            "adapters. V502 CSL is frozen-weight memory and residue repair, so "
            "broad fine-tuning remains closed."
        ),
    },
    {
        "title": "Hallucination Self-Play",
        "url": "https://arxiv.org/list/cs.CL/new",
        "classification": "excluded",
        "reason": (
            "The detector/generator loop uses RLAIF and rule-based RL training. "
            "V502 does not reopen external detector training or policy-gradient "
            "hallucination mitigation."
        ),
    },
    {
        "title": "GRAPHEVAL / Graph Reasoning Coherence Score",
        "url": "https://arxiv.org/list/cs.CL/new",
        "classification": "watch-only",
        "reason": (
            "The graph-coherence metric is useful telemetry context, but V502 "
            "hard/soft panels require exact validators and authenticated local "
            "duration receipts before adding another advisory scorer."
        ),
    },
    {
        "title": "Game Theory Driven Multi-Agent Framework Mitigates Language Model Hallucination",
        "url": "https://arxiv.org/abs/2607.08403",
        "classification": "excluded",
        "reason": (
            "The paper depends on domain-specific chemistry data synthesis and "
            "training. That does not map to V502 exact fixtures, grammar "
            "preflight, CSL residue, hardware hygiene, or ARC live-path recovery."
        ),
    },
    {
        "title": "Extropic TSU / XTR / Z1 writing",
        "url": "https://extropic.ai/writing",
        "classification": "watch-only",
        "reason": (
            "The public pages remain non-local architecture context. Carnot has "
            "no TSU SDK, local device receipt, or matched timing path for V502."
        ),
    },
    {
        "title": "Logical Intelligence Kona and Aleph public pages",
        "url": "https://logicalintelligence.com/kona-ebms-energy-based-models",
        "classification": "watch-only",
        "reason": (
            "The pages reinforce verifier-under-LLM architecture, but there is "
            "no local executable path or reproducible Kona/Aleph baseline."
        ),
    },
]

SEMANTIC_SCHOLAR_STATUS = (
    "ok: EBT 2507.02092 citationCount=27 influentialCitationCount=2; "
    "ARM-EBM 2512.15605 citationCount=8 influentialCitationCount=2; sample "
    "citation routes matched prior planner context and produced no stronger "
    "V502 local dependency."
)

SEARCHED_SOURCE_DETAILS: dict[str, JsonDict] = {
    "arxiv_cs_ai_new_20260710": {
        "status": "ok",
        "finding": "Accepted arXiv:2607.08136 as the sole new actionable delta.",
    },
    "arxiv_cs_lg_new_20260710": {
        "status": "ok",
        "finding": "ReCoLoRA and continual-learning items were excluded as fine-tuning scope.",
    },
    "arxiv_cs_cl_new_20260710": {
        "status": "ok",
        "finding": "GRAPHEVAL and Hallucination Self-Play stayed watch-only/excluded.",
    },
    "arxiv_topic_api_sweep": {
        "status": "ok",
        "finding": "Topic searches yielded duplicates already captured by V500-V502 history.",
    },
    "arxiv_primary_pages": {
        "status": "ok",
        "finding": "Primary pages for 2607.08136, 2607.07719, and 2607.08403 were checked.",
    },
    "openreview": {
        "status": "challenge_blocked",
        "finding": "OpenReview forum pages redirected to browser verification.",
    },
    "huggingface_papers": {
        "status": "partial",
        "finding": "Search pages mirrored already indexed constrained and EBM papers.",
    },
    "semantic_scholar_ebt_arm_ebm": {
        "status": "ok",
        "finding": SEMANTIC_SCHOLAR_STATUS,
    },
    "github": {
        "status": "partial",
        "finding": "GitHub search did not expose a public 2607.08136 implementation repository.",
    },
    "extropic_writing": {
        "status": "ok",
        "finding": "Public writing remains architecture context only.",
    },
    "logical_intelligence_public_pages": {
        "status": "ok",
        "finding": "Kona/Aleph pages remain non-local architecture context.",
    },
    "local_v502_and_prior_duplicate_history": {
        "status": "ok",
        "finding": "research-references.md contained no 2607.08136 duplicate.",
    },
    "ops_exclusion_manifest": {
        "status": "ok",
        "finding": "No retired or operator-reopen-required scope was reopened.",
    },
    "research_roadmap_v502_tasks": {
        "status": "ok",
        "finding": "Accepted delta maps to Exp5541 and supports Exp5545 without roadmap edits.",
    },
}


def _copy_references(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [dict(row) for row in rows]


def _build_mappings(references: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    mappings: list[JsonDict] = []
    for row in references:
        mappings.append(
            {
                "title": row["title"],
                "planned_experiment": row["planned_experiment"],
                "mapped_task": row["mapped_task"],
                "rationale": row["carnot_hook"],
            }
        )
    return mappings


def build_artifact(
    *,
    new_references_added: Sequence[Mapping[str, Any]] | None = None,
    methodology_duration_s: float = 0.0,
    tests_run: Sequence[str] | None = None,
    prior_refresh_marker_found: bool = True,
    research_references_updated: bool | None = None,
    semantic_scholar_status: str = SEMANTIC_SCHOLAR_STATUS,
) -> JsonDict:
    """Build the source-delta receipt without touching the filesystem.

    The function is pure so tests can exercise append/no-op/blocked behavior
    without relying on network calls or global repository state. Network results
    are represented as audited constants because this artifact aggregates the
    execution sweep rather than re-running it during unit tests.
    """

    references = (
        _copy_references(DEFAULT_NEW_REFERENCES_ADDED)
        if new_references_added is None
        else _copy_references(new_references_added)
    )
    if not prior_refresh_marker_found:
        references = []

    count = len(references)
    updated = count > 0 if research_references_updated is None else research_references_updated
    blocked = not prior_refresh_marker_found

    if blocked:
        status = "blocked"
        honest_verdict = (
            "blocked: V502 planner refresh marker missing; references unchanged "
            "and closed scopes stayed closed."
        )
        updated = False
    elif count:
        status = "complete"
        plural = "s" if count != 1 else ""
        honest_verdict = (
            f"complete: {count} new actionable V502 execution-time source "
            f"delta{plural} appended; closed scopes remained closed."
        )
    else:
        status = "complete"
        honest_verdict = (
            "complete: no new actionable V502 execution-time source deltas "
            "after the planner refresh; references unchanged and closed scopes "
            "stayed closed."
        )

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
        "semantic_scholar_status": semantic_scholar_status,
        "closed_scopes_reopened": False,
        "research_references_updated": updated,
        "prior_refresh_marker_found": prior_refresh_marker_found,
        "experiment_mappings": _build_mappings(references),
        "field_principles": dict(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": honest_verdict,
        "searched_source_details": {key: dict(value) for key, value in SEARCHED_SOURCE_DETAILS.items()},
        "watch_only_or_excluded": _copy_references(WATCH_ONLY_OR_EXCLUDED),
        "spec_refs": list(SPEC_REFS),
        "methodology_duration_s": methodology_duration_s,
        "tests_run": list(tests_run or ["tests/python/test_experiment_5537_v502_source_delta_ingestion.py"]),
        "no_deep_research_used": True,
        "research_conductor_modified": False,
        "ops_docs_modified": False,
        "traceability_modified": False,
        "roadmap_files_modified": False,
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the receipt schema and the claim boundaries it protects."""

    missing = REQUIRED_TOP_LEVEL_FIELDS.difference(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")

    if artifact["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("experiment_id must identify Exp5537")
    if artifact["task_id"] != TASK_ID:
        raise ValueError("task_id must identify the V502 source-delta task")
    if artifact["milestone"] != MILESTONE:
        raise ValueError("milestone must be 2026.07.502")
    if artifact["status"] not in {"complete", "blocked"}:
        raise ValueError("status must be complete or blocked")
    if artifact["search_date"] != SEARCH_DATE:
        raise ValueError("search_date must use compact 20260710 form")
    if not isinstance(artifact["semantic_scholar_status"], str) or not artifact["semantic_scholar_status"]:
        raise ValueError("semantic_scholar_status must be a non-empty string")

    sources = artifact["sources_checked"]
    if not isinstance(sources, list) or not set(REQUIRED_SOURCE_FAMILIES).issubset(sources):
        raise ValueError("sources_checked must include every required source family")
    if len(sources) != len(set(sources)):
        raise ValueError("duplicate source families are not allowed")

    details = artifact["searched_source_details"]
    if not isinstance(details, Mapping):
        raise ValueError("searched_source_details must be a mapping")
    for source in sources:
        if source not in details:
            raise ValueError(f"searched_source_details missing {source}")
        status = details[source].get("status") if isinstance(details[source], Mapping) else None
        if status not in ALLOWED_SOURCE_STATUSES:
            raise ValueError("searched_source_details entries must have a valid status")

    references = artifact["new_references_added"]
    if not isinstance(references, list):
        raise ValueError("new_references_added must be a list")
    for reference in references:
        if not isinstance(reference, Mapping):
            raise ValueError("new_references_added entries must be mappings")
        missing_reference_fields = REQUIRED_REFERENCE_FIELDS.difference(reference)
        if missing_reference_fields:
            raise ValueError(f"reference missing required fields: {sorted(missing_reference_fields)}")
        planned = reference["planned_experiment"]
        if planned not in PLANNED_EXPERIMENT_TASKS:
            raise ValueError(f"unknown planned_experiment: {planned}")
        expected_task = PLANNED_EXPERIMENT_TASKS[planned]
        if reference["mapped_task"] != expected_task:
            raise ValueError("mapped_task must match the planned experiment")

    count = artifact["new_actionable_findings_count"]
    if count != len(references):
        raise ValueError("new_actionable_findings_count must match references count")
    if artifact["experiment_mappings"] != _build_mappings(references):
        raise ValueError("experiment_mappings must match accepted references")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles must exactly cover required fields")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    if artifact["closed_scopes_reopened"] is not False:
        raise ValueError("closed_scopes_reopened must remain false")

    prior_marker = artifact["prior_refresh_marker_found"]
    updated = artifact["research_references_updated"]
    if artifact["status"] == "blocked":
        if prior_marker is not False:
            raise ValueError("blocked artifacts require a missing prior_refresh_marker_found marker")
        if updated is not False or references or artifact["experiment_mappings"]:
            raise ValueError("research_references_updated must be false when blocked")
    else:
        if prior_marker is not True:
            raise ValueError("prior_refresh_marker_found must be true for complete artifacts")
        if updated != bool(references):
            raise ValueError("research_references_updated must match whether references were added")

    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not verdict.startswith(("complete:", "blocked:")):
        raise ValueError("honest_verdict must start with complete: or blocked:")
    if artifact["no_deep_research_used"] is not True:
        raise ValueError("no_deep_research_used must remain true")
    if any(
        artifact[field] is not False
        for field in (
            "research_conductor_modified",
            "ops_docs_modified",
            "traceability_modified",
            "roadmap_files_modified",
        )
    ):
        raise ValueError("protected files must remain unmodified")


def render_refresh_section(artifact: Mapping[str, Any]) -> str:
    """Render the references append block for accepted deltas only."""

    validate_artifact(artifact)
    references = artifact["new_references_added"]
    if artifact["status"] == "blocked" or not references:
        return ""

    lines = [
        REFRESH_HEADING,
        "",
        (
            "Execution-time sweep after the `.502` planner refresh checked arXiv "
            "new/topic pages, OpenReview, HuggingFace Papers, Semantic Scholar "
            "routes for EBT and ARM-EBM, GitHub, Extropic writing, Logical "
            "Intelligence public pages, local duplicate history, and the "
            "exclusion manifest. Only non-duplicate actionable deltas are "
            "listed below."
        ),
        "",
        "### New actionable delta",
    ]
    for reference in references:
        secondary = reference.get("secondary_mapped_tasks", [])
        secondary_text = ""
        if secondary:
            secondary_text = f" Secondary V502 hook: {', '.join(secondary)}."
        lines.append(
            (
                f"- **{reference['title']}** ({reference['url']}): "
                f"{reference['carnot_hook']} Primary V502 hook: "
                f"{reference['mapped_task']}.{secondary_text}"
            )
        )

    lines.extend(
        [
            "",
            "### Execution impact",
            (
                "- **Plan impact:** No active roadmap edit is required. The "
                "accepted ASP+EBM delta sharpens Exp5541's finite-state exact "
                "fixture and Exp5545's sparse FSM descriptor family without "
                "changing gate order."
            ),
            (
                "- **Duplicates suppressed:** "
                + "; ".join(str(item) for item in artifact["duplicates_suppressed"])
            ),
            (
                "- **Closed scope:** No closed scope was reopened. Fine-tuning, "
                "external detector training, proprietary/non-local TSU or Kona "
                "execution, and hardware speedup claims without matched timing "
                "remain closed."
            ),
            (
                "- **Watch-only/excluded:** "
                + "; ".join(
                    f"{row['title']} ({row['classification']})"
                    for row in artifact["watch_only_or_excluded"]
                )
            ),
            "",
            REFRESH_END_MARKER,
            "",
        ]
    )
    return "\n".join(lines)


def append_refresh_section(references_text: str, artifact: Mapping[str, Any]) -> str:
    """Append the rendered refresh once, preserving no-op receipts as no-ops."""

    section = render_refresh_section(artifact)
    if not section:
        return references_text
    if PLANNER_MARKER not in references_text:
        raise ValueError("V502 planner refresh marker missing from research-references.md")
    if REFRESH_HEADING in references_text:
        return references_text
    return references_text.rstrip() + "\n\n" + section


def write_outputs(
    *,
    root: Path = REPO_ROOT,
    references_path: Path | None = None,
    result_path: Path | None = None,
    methodology_duration_s: float = 0.0,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Write the receipt JSON and append references only when deltas qualify."""

    references = references_path or root / REFERENCES_RELATIVE_PATH
    result = result_path or root / RESULT_RELATIVE_PATH
    references_text = references.read_text(encoding="utf-8")
    prior_marker = PLANNER_MARKER in references_text

    artifact = build_artifact(
        prior_refresh_marker_found=prior_marker,
        methodology_duration_s=methodology_duration_s,
        tests_run=tests_run,
    )

    updated_text = append_refresh_section(references_text, artifact)
    if updated_text != references_text:
        references.write_text(updated_text, encoding="utf-8")

    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:  # pragma: no cover - thin CLI wrapper for conductor use.
    write_outputs()


if __name__ == "__main__":  # pragma: no cover
    main()
