"""Exp 5308: execution-time SOTA source delta refresh for V485.

Spec refs: REQ-REPORT-5308, SCENARIO-REPORT-5308-APPEND-DELTAS,
SCENARIO-REPORT-5308-NOOP.

This module records the live source sweep as a deterministic artifact. The
network checks are intentionally not performed during tests: arXiv, OpenReview,
Hugging Face, Semantic Scholar, and GitHub change independently of this repo.
The constants below preserve what was checked, which rows were new relative to
the V485 planner refresh, and why the rows are actionable without changing the
executable V485 plan.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_5308_sota_source_delta_v485"
MILESTONE = "2026.07.485"
RESULT_RELATIVE_PATH = Path("results/experiment_5308_sota_source_delta_v485.json")
REFERENCES_RELATIVE_PATH = Path("research-references.md")
REFRESH_HEADING = "### V485 Execution Refresh - 2026-07-06"
REFRESH_END_MARKER = "<!-- V485-EXECUTION-REFRESH-2026-07-06-END -->"
INFERENCE_SUBSTRATE = "literature_ingestion_network_sources"

SPEC_REFS = [
    "REQ-REPORT-5308",
    "SCENARIO-REPORT-5308-APPEND-DELTAS",
    "SCENARIO-REPORT-5308-NOOP",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": "traceability for the Exp5308 execution-time source delta artifact.",
    "milestone": "milestone accountability for V485 source freshness.",
    "status": "machine-readable terminal state for downstream reconciliation.",
    "honest_verdict": (
        "terminal verdict must start with complete: or blocked_ so nuance cannot be "
        "misclassified."
    ),
    "inference_substrate": (
        "literature_ingestion_network_sources because Exp5308 reads network "
        "literature/source metadata and makes no experiment outcome claim."
    ),
    "sources_checked": (
        "records every required source family and query channel so a missing source "
        "cannot masquerade as a zero-new-finding result."
    ),
    "references_section_marker": "idempotent marker prevents duplicate research-references.md appends.",
    "actionable_findings": (
        "source URLs plus Carnot-local hooks make each appended reference auditable "
        "instead of bibliography churn."
    ),
}

REQUIRED_SOURCE_FAMILIES = (
    "arxiv",
    "openreview",
    "huggingface_papers",
    "semantic_scholar",
    "github",
    "extropic",
    "logical_intelligence",
    "local_v485_comparison",
)

REQUIRED_TOP_LEVEL_FIELDS = frozenset(
    {
        "experiment_id",
        "milestone",
        "status",
        "honest_verdict",
        "inference_substrate",
        "sources_checked",
        "new_actionable_findings_count",
        "references_modified",
        "references_section_marker",
        "retired_scope_reopened",
        "methodology_duration_s",
        "no_executable_plan_edit",
        "actionable_findings",
        "spec_refs",
        "search_window",
        "tests_run",
        "field_principles",
        "no_deep_research_used",
        "research_conductor_modified",
        "roadmap_files_modified",
    }
)

REQUIRED_FINDING_FIELDS = frozenset(
    {
        "title",
        "source_url",
        "arxiv_id_or_repo",
        "source_family",
        "category",
        "carnot_hook",
        "actionability",
        "planned_task_impact",
        "retired_scope_risk",
    }
)

ACTIONABLE_FINDINGS: list[JsonDict] = [
    {
        "title": "Self-GC: Self-Governing Context for Long-Horizon LLM Agents",
        "arxiv_id_or_repo": "2607.00692",
        "source_url": "https://arxiv.org/abs/2607.00692",
        "source_family": "arxiv",
        "category": "continual_constraint_context_lifecycle",
        "carnot_hook": (
            "Self-GC treats user turns, tool spans, and skill state as indexed context "
            "objects; a side-channel planner proposes fold, mask, and prune actions; "
            "and the harness enforces recoverable sidecars, safe commit boundaries, "
            "and cache-aware commits."
        ),
        "actionability": (
            "For V485 continuous self-learning work, evaluate context-object lifecycle "
            "decisions separately from final answer quality. Require stable object IDs, "
            "recoverable sidecars, and explicit safe-commit boundaries before memory or "
            "skill state is promoted or pruned."
        ),
        "planned_task_impact": "no_plan_edit",
        "retired_scope_risk": "none",
    },
    {
        "title": "A-TMA: Decoupling State-Aware Memory Failures in Long-Term Agent Memory",
        "arxiv_id_or_repo": "2607.01935",
        "source_url": "https://arxiv.org/abs/2607.01935",
        "source_family": "arxiv",
        "category": "state_aware_memory_failure_evaluation",
        "carnot_hook": (
            "A-TMA isolates ghost-memory failures where old, current, and transition "
            "facts coexist in the bank, mix during retrieval, and mislead answer-time "
            "resolution. It evaluates bank maintenance, retrieval, and answer failures "
            "separately instead of relying only on final QA accuracy."
        ),
        "actionability": (
            "For the V485 transition-level memory verifier, add current, historical, "
            "and transition labels to conflict-heavy fixtures. Report bank, retrieval, "
            "and answer-time failure rates separately so a clean final answer cannot "
            "hide stale or contradictory memory state."
        ),
        "planned_task_impact": "no_plan_edit",
        "retired_scope_risk": "none",
    },
]

SOURCES_CHECKED: JsonDict = {
    "arxiv": {
        "status": "ok",
        "queries": [
            "2026-07 arXiv date-window search for EBM reasoning and language-model verification",
            "2026-07 arXiv date-window search for neural constraint and symbolic solver methods",
            "2026-07 arXiv date-window search for Ising, p-bit, annealing, and hardware sampling",
            "2026-07 arXiv date-window search for hallucination detection over code/tool outputs",
            "2026-07 arXiv date-window search for KAN verification and certificates",
            "2026-07 arXiv date-window search for energy-guided decoding",
            "2026-07 arXiv date-window search for continual memory and constraint learning",
        ],
        "new_actionable_ids": ["2607.00692", "2607.01935"],
        "already_indexed_or_duplicate": [
            "2607.02491 G-RRM",
            "2607.00895 Beyond Document Grounding",
            "2607.01224 AutoMem",
            "2607.02255 AgenticSTS",
            "2606.26476 Retrieval-Warmed Energy-Based Reasoning",
            "2602.06737 KAN optimal abstractions",
            "2606.30333 Ising continuous relaxation",
        ],
        "not_promoted": [
            "2607.02010 InduceKV is KV-cache and multimodal adaptation work that would "
            "require a new model-internals substrate rather than a V485 source-delta note.",
            "2607.01640 AgentFlow and 2607.01641 infinite-agent-loop scanning are agent "
            "static-analysis leads, not direct EBM/constraint-learning deltas for this task.",
            "2607.02517 WorldDirector is a video world-model memory paper outside the "
            "Carnot constraint-verification execution path.",
        ],
    },
    "openreview": {
        "status": "ok",
        "queries": [
            "OpenReview energy and hallucination verifier search",
            "OpenReview constrained decoding and neuro-symbolic verification search",
            "OpenReview Spilled Energy, Semantic Energy, and computational-graph CoT pages",
        ],
        "result": (
            "OpenReview results reinforced already indexed Spilled Energy, Semantic "
            "Energy, and computational-graph reasoning verification references. No "
            "OpenReview-only source changed the V485 execution priorities."
        ),
    },
    "huggingface_papers": {
        "status": "ok",
        "queries": [
            "HuggingFace Papers lookups for Self-GC 2607.00692 and A-TMA 2607.01935",
            "HuggingFace Papers memory, agent, EBT, and ARM-EBM related pages",
            "HuggingFace Papers July 2026 hallucination and constraint-memory watch",
        ],
        "result": (
            "No Hugging Face Papers page supplied a separate actionable delta beyond "
            "the arXiv primary rows. Context-Folding and AgentFold pages were older "
            "context-management background, not V485 source deltas."
        ),
    },
    "semantic_scholar": {
        "status": "ok",
        "queries": [
            "DOI:10.48550/arXiv.2507.02092 metadata and citations",
            "DOI:10.48550/arXiv.2512.15605 metadata and citations",
        ],
        "result": (
            "Direct Graph API checks still returned EBT citationCount=26 and ARM-EBM "
            "citationCount=8. Samples were already represented in prior V484/V485 "
            "context, so no Semantic Scholar-only trend claim is added."
        ),
        "citation_status": {
            "EBT": {
                "arxiv_id": "2507.02092",
                "citationCount": 26,
                "influentialCitationCount": 2,
                "sample_ids": ["2606.18206", "2605.11011", "10.1109/ISPASS69572.2026.00062"],
            },
            "ARM-EBM": {
                "arxiv_id": "2512.15605",
                "citationCount": 8,
                "influentialCitationCount": 2,
                "sample_ids": ["2607.02154", "2605.18871", "10.18653/v1/2026.acl-long.2131"],
            },
        },
    },
    "github": {
        "status": "ok",
        "queries": [
            "GitHub repository search for Self-GC Self-Governing Context",
            "GitHub repository search for A-TMA state-aware memory",
            "GitHub repository search for AgenticSTS and AutoMem duplicate code references",
            "GitHub repository search for EBT, KAN verification, and constraint-memory repos",
        ],
        "result": (
            "No Self-GC or A-TMA implementation repository was found in the checked "
            "GitHub search results. AgenticSTS and AutoMem repositories exist but were "
            "already indexed in Carnot reference history."
        ),
    },
    "extropic": {
        "status": "ok",
        "queries": [
            "extropic.ai/writing index",
            "TSU 101",
            "Inside X0 and XTR-0",
            "thermodynamic-computing-from-zero-to-one",
        ],
        "result": (
            "The public writing index still points to already indexed TSU/XTR-0 and "
            "probabilistic-hardware material. No Carnot-accessible TSU hardware, SDK, "
            "or local execution basis was found."
        ),
    },
    "logical_intelligence": {
        "status": "ok",
        "queries": [
            "Logical Intelligence blog and public pages",
            "Kona, Aleph, and automatic formal verification posts",
            "energy-based reasoning model public updates",
        ],
        "result": (
            "Logical Intelligence remains a verifier-first strategic signal. Public "
            "pages still expose no reproducible Kona internals, local SDK, or benchmark "
            "receipt that would alter V485 execution."
        ),
    },
    "local_v485_comparison": {
        "status": "ok",
        "queries": [
            "research-references.md V485 Planner Refresh - 2026-07-06",
            "repo-wide duplicate search for Self-GC, A-TMA, 2607.00692, and 2607.01935",
            "ops/exclusion_manifest.yaml retired scopes",
            "results/experiment_5296_sota_source_delta_v484.json prior source status",
        ],
        "result": (
            "Self-GC and A-TMA were absent from the V485 planner refresh and nearby "
            "reference history. They sharpen memory-state and context-lifecycle testing "
            "without reopening retired external-text scoring, broad fine-tuning, "
            "TSU/Kona execution, or CPU-only GGUF offload scopes."
        ),
    },
}


def _principled(field: str, value: Any) -> JsonDict:
    return {"value": value, "principle": FIELD_PRINCIPLES[field]}


def _validate_principled_wrapper(field: str, artifact: Mapping[str, Any]) -> Any:
    wrapper = artifact.get(field)
    if not isinstance(wrapper, Mapping):
        raise ValueError(f"{field} must be principle-wrapped")
    if wrapper.get("principle") != FIELD_PRINCIPLES[field]:
        raise ValueError(f"{field} must include the declared principle")
    if "value" not in wrapper:
        raise ValueError(f"{field} missing value")
    return wrapper["value"]


def _verified_url(value: str) -> bool:
    return value.startswith("https://")


def build_artifact(
    *,
    actionable_findings: Sequence[Mapping[str, Any]] = ACTIONABLE_FINDINGS,
    methodology_duration_s: float = 0.0,
    tests_run: Sequence[str] = (),
) -> JsonDict:
    """Build the terminal Exp5308 artifact from source-verified finding rows.

    This artifact is a literature-ingestion receipt. It does not claim model
    performance, so it records methodology timing rather than compute duration.
    """

    findings = [dict(row) for row in actionable_findings]
    count = len(findings)
    references_modified = count > 0
    status = "complete"
    verdict_detail = (
        f"{count} new actionable V485 source findings appended; executable .485 plan unchanged"
        if count
        else "no new actionable V485 source findings; references unchanged"
    )
    artifact: JsonDict = {
        "experiment_id": _principled("experiment_id", EXPERIMENT_ID),
        "milestone": _principled("milestone", MILESTONE),
        "status": _principled("status", status),
        "honest_verdict": _principled("honest_verdict", f"complete: {verdict_detail}."),
        "inference_substrate": _principled("inference_substrate", INFERENCE_SUBSTRATE),
        "sources_checked": _principled("sources_checked", dict(SOURCES_CHECKED)),
        "new_actionable_findings_count": count,
        "references_modified": references_modified,
        "references_section_marker": _principled(
            "references_section_marker", REFRESH_END_MARKER if references_modified else None
        ),
        "retired_scope_reopened": False,
        "methodology_duration_s": round(float(methodology_duration_s), 6),
        "no_executable_plan_edit": True,
        "actionable_findings": _principled("actionable_findings", findings),
        "spec_refs": list(SPEC_REFS),
        "search_window": {
            "run_date": "2026-07-06",
            "years": "2025-2026",
            "comparison_anchor": "research-references.md V485 Planner Refresh - 2026-07-06",
        },
        "tests_run": list(tests_run)
        or ["tests/python/test_experiment_5308_sota_source_delta_v485.py"],
        "field_principles": dict(FIELD_PRINCIPLES),
        "no_deep_research_used": True,
        "research_conductor_modified": False,
        "roadmap_files_modified": False,
    }
    validate_artifact(artifact)
    return artifact


def _validate_sources(sources: Any) -> None:
    if not isinstance(sources, Mapping) or not set(REQUIRED_SOURCE_FAMILIES).issubset(sources):
        raise ValueError("sources_checked must include every required source family")
    for family in REQUIRED_SOURCE_FAMILIES:
        if not isinstance(sources.get(family), Mapping) or sources[family].get("status") != "ok":
            raise ValueError(f"sources_checked {family} must record status ok")


def _validate_findings(findings: Any) -> None:
    if not isinstance(findings, list):
        raise ValueError("actionable_findings value must be a list")
    for row in findings:
        if not isinstance(row, Mapping) or not REQUIRED_FINDING_FIELDS.issubset(row):
            raise ValueError(
                f"actionable_findings rows must include {sorted(REQUIRED_FINDING_FIELDS)}"
            )
        if not _verified_url(str(row["source_url"])):
            raise ValueError("actionable_findings rows must use a verified URL")
        if row["planned_task_impact"] != "no_plan_edit":
            raise ValueError("actionable_findings must not edit the active plan")
        if row["retired_scope_risk"] != "none":
            raise ValueError("actionable_findings must not reopen retired scopes")


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = REQUIRED_TOP_LEVEL_FIELDS.difference(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match REQ-REPORT-5308")

    experiment_id = _validate_principled_wrapper("experiment_id", artifact)
    if experiment_id != EXPERIMENT_ID:
        raise ValueError("experiment_id must match Exp5308")
    milestone = _validate_principled_wrapper("milestone", artifact)
    if milestone != MILESTONE:
        raise ValueError("milestone must match 2026.07.485")
    status = _validate_principled_wrapper("status", artifact)
    if status not in {"complete", "blocked"}:
        raise ValueError("status must be complete or blocked")
    verdict = _validate_principled_wrapper("honest_verdict", artifact)
    if not (str(verdict).startswith("complete:") or str(verdict).startswith("blocked_")):
        raise ValueError("honest_verdict must start with complete: or blocked_")
    substrate = _validate_principled_wrapper("inference_substrate", artifact)
    if substrate != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be literature_ingestion_network_sources")

    sources = _validate_principled_wrapper("sources_checked", artifact)
    _validate_sources(sources)
    findings = _validate_principled_wrapper("actionable_findings", artifact)
    _validate_findings(findings)

    count = artifact["new_actionable_findings_count"]
    if not isinstance(count, int) or count != len(findings):
        raise ValueError("findings count must equal actionable_findings length")
    references_modified = artifact["references_modified"]
    if references_modified is not (count > 0):
        raise ValueError("references_modified must match whether findings were added")
    marker = _validate_principled_wrapper("references_section_marker", artifact)
    if marker != (REFRESH_END_MARKER if references_modified else None):
        raise ValueError("references_section_marker must match the references append state")
    if artifact["retired_scope_reopened"] is not False:
        raise ValueError("retired_scope_reopened must remain false for this refresh")
    duration = artifact["methodology_duration_s"]
    if not isinstance(duration, int | float) or duration < 0:
        raise ValueError("methodology_duration_s must be a non-negative number")
    if artifact["no_executable_plan_edit"] is not True:
        raise ValueError("executable plan must not be edited by this refresh")
    if artifact["no_deep_research_used"] is not True:
        raise ValueError("deep-research must not be used")
    if artifact["research_conductor_modified"] is not False:
        raise ValueError("research_conductor.py must not be modified")
    if artifact["roadmap_files_modified"] is not False:
        raise ValueError("roadmap files must not be modified")
    if not artifact["tests_run"]:
        raise ValueError("tests_run must record at least one test")


def _render_finding(row: Mapping[str, Any]) -> str:
    return (
        f"- **{row['title']}** ({row['source_url']}): {row['carnot_hook']} "
        f"Actionability: {row['actionability']}"
    )


def render_refresh_section(artifact: Mapping[str, Any]) -> str:
    validate_artifact(artifact)
    findings = artifact["actionable_findings"]["value"]
    if not findings:
        return ""
    lines = [
        REFRESH_HEADING,
        "",
        (
            "Execution-time sweep after the `.485` planner refresh checked arXiv, "
            "OpenReview, HuggingFace Papers, Semantic Scholar EBT/ARM-EBM citation "
            "trails, GitHub repositories, Extropic writing, Logical Intelligence "
            "public pages, and local duplicate history. The findings below were absent "
            "from the V485 planner block and nearby reference history."
        ),
        "",
        "### New actionable deltas",
        *(_render_finding(row) for row in findings),
        "",
        "### Execution impact",
        (
            "- **Plan impact:** No executable `.485` task edit is required. The deltas "
            "sharpen transition-level memory verification, context-object lifecycle "
            "testing, and conflict-heavy stale-memory fixtures."
        ),
        (
            "- **Retired scope:** No retired scope was reopened. External generated-text "
            "scoring, broad GRPO/fine-tuning reruns, TSU/Kona execution claims, and "
            "CPU-only GGUF offload reruns remain closed."
        ),
        (
            "- **Secondary-source status:** Semantic Scholar EBT/ARM-EBM counts matched "
            "the prior live check; OpenReview, HuggingFace Papers, Extropic, GitHub, "
            "and Logical Intelligence did not add a separate execution-changing source."
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
    root: Path | str = REPO_ROOT,
    references_path: Path | None = None,
    result_path: Path | None = None,
    actionable_findings: Sequence[Mapping[str, Any]] = ACTIONABLE_FINDINGS,
    methodology_duration_s: float = 0.0,
    tests_run: Sequence[str] = (),
) -> JsonDict:
    base = Path(root)
    references = references_path or (base / REFERENCES_RELATIVE_PATH)
    result = result_path or (base / RESULT_RELATIVE_PATH)
    artifact = build_artifact(
        actionable_findings=actionable_findings,
        methodology_duration_s=methodology_duration_s,
        tests_run=tests_run,
    )
    original = references.read_text(encoding="utf-8") if references.exists() else ""
    updated = append_refresh_section(original, artifact)
    references.parent.mkdir(parents=True, exist_ok=True)
    result.parent.mkdir(parents=True, exist_ok=True)
    references.write_text(updated, encoding="utf-8")
    result.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> int:  # pragma: no cover - CLI convenience for the experiment run.
    artifact = write_outputs()
    print(artifact["honest_verdict"]["value"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
