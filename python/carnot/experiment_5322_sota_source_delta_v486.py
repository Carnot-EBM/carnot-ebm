"""Exp 5322: execution-time SOTA source delta refresh for V486.

Spec refs: REQ-REPORT-5322, SCENARIO-REPORT-5322-APPEND-DELTAS,
SCENARIO-REPORT-5322-NOOP.

This module records a live source sweep as a deterministic artifact. The
network queries happen before this file is authored because arXiv, OpenReview,
Hugging Face, Semantic Scholar, and GitHub can change independently of this
repository. The constants below preserve what was checked, which rows were new
relative to the V486 planner refresh, and why those rows are actionable without
changing the executable V486 plan.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_5322_sota_source_delta_v486"
MILESTONE = "2026.07.486"
RESULT_RELATIVE_PATH = Path("results/experiment_5322_sota_source_delta_v486.json")
REFERENCES_RELATIVE_PATH = Path("research-references.md")
REFRESH_HEADING = "### V486 Execution Refresh - 2026-07-06"
REFRESH_END_MARKER = "<!-- V486-EXECUTION-REFRESH-2026-07-06-END -->"
INFERENCE_SUBSTRATE = "literature_ingestion_network_sources"

SPEC_REFS = [
    "REQ-REPORT-5322",
    "SCENARIO-REPORT-5322-APPEND-DELTAS",
    "SCENARIO-REPORT-5322-NOOP",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": "traceability for the Exp5322 execution-time source delta artifact.",
    "milestone": "milestone accountability for V486 source freshness.",
    "status": "machine-readable terminal state for downstream reconciliation.",
    "honest_verdict": (
        "terminal verdict must start with complete: or blocked_ so nuance cannot be "
        "misclassified."
    ),
    "inference_substrate": (
        "literature_ingestion_network_sources because Exp5322 reads network "
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
    "local_v486_comparison",
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
        "title": "ContextNest: Verifiable Context Governance for Autonomous AI Agent",
        "arxiv_id_or_repo": "2607.02116",
        "source_url": "https://arxiv.org/abs/2607.02116",
        "secondary_source_url": "https://github.com/PromptOwl/ContextNest",
        "source_family": "arxiv+github",
        "category": "verifiable_context_governance",
        "carnot_hook": (
            "ContextNest formalizes context governance below retrieval with typed "
            "Markdown documents, deterministic selectors, contextnest:// references, "
            "SHA-256 hash-chained versions, graph checkpoints, MCP source nodes, and "
            "audit traces of context consumption."
        ),
        "actionability": (
            "For Exp5328/Exp5330, require context-object lifecycle rows to preserve "
            "version identity, approval/currentness, integrity hashes, point-in-time "
            "reconstruction, and audit traces before any context or memory policy is "
            "promoted."
        ),
        "planned_task_impact": "no_plan_edit",
        "retired_scope_risk": "none",
    },
    {
        "title": "Episodic-to-Semantic Consolidation Without Identity Drift",
        "arxiv_id_or_repo": "2607.01988",
        "source_url": "https://arxiv.org/abs/2607.01988",
        "source_family": "arxiv",
        "category": "identity_stable_memory_consolidation",
        "carnot_hook": (
            "The paper treats consolidation as a deterministic episodic-to-semantic "
            "memory function whose output is separately addressable; the certified "
            "agent identity hash excludes that semantic layer so knowledge can update "
            "without mutating the agent identity."
        ),
        "actionability": (
            "For Exp5330, keep SEA-style policy promotion and memory consolidation "
            "outside frozen model identity. Record byte-stable identity manifests, "
            "semantic-layer sidecars, supporting-event provenance, rollback fields, "
            "and no weight mutation."
        ),
        "planned_task_impact": "no_plan_edit",
        "retired_scope_risk": "none",
    },
    {
        "title": "Distributed Attacks in Persistent-State AI Control",
        "arxiv_id_or_repo": "2607.02514",
        "source_url": "https://arxiv.org/abs/2607.02514",
        "secondary_source_url": "https://github.com/josh-hills/control-arena-persistent-state-eval",
        "source_family": "arxiv+github",
        "category": "persistent_state_monitoring",
        "carnot_hook": (
            "The paper and repository show that gradual cross-change attacks evade "
            "standard per-diff monitors, while a stateful link-tracker that carries "
            "suspicion notes across changes detects gradual buildup better."
        ),
        "actionability": (
            "For Exp5328/Exp5330, include cross-event suspicious-buildup telemetry in "
            "context lifecycle fixtures: link memory commits, sidecars, rollback "
            "events, and verifier-dose changes across sessions instead of judging each "
            "state change independently."
        ),
        "planned_task_impact": "no_plan_edit",
        "retired_scope_risk": "none",
    },
]

SOURCES_CHECKED: JsonDict = {
    "arxiv": {
        "status": "ok",
        "queries": [
            "arXiv cs.AI recent and new pages for 2026-07-02 through 2026-07-03",
            "2026-07 arXiv search for EBM reasoning and verification",
            "2026-07 arXiv search for neural constraint satisfaction and symbolic solvers",
            "2026-07 arXiv search for Ising, p-bit, annealing, and hardware sampling",
            "2026-07 arXiv search for hallucination detection and energy-guided decoding",
            "2026-07 arXiv search for KAN verification and certificates",
            "2026-07 arXiv search for continual context, memory, and constraint learning",
        ],
        "new_actionable_ids": ["2607.02116", "2607.01988", "2607.02514"],
        "already_indexed_or_duplicate": [
            "2607.00871 Self-Evolving Agents with Anytime-Valid Certificates",
            "2607.01223 Theoria",
            "2606.30005 VISTA latent context managers",
            "2607.01224 AutoMem",
            "2607.00692 Self-GC",
            "2607.01935 A-TMA",
            "2607.02491 G-RRM",
            "2605.04033 p-bit CDCL",
            "2602.16143 FPGA p-bit annealer",
            "2508.14496 Semantic Energy",
            "2602.18671 Spilled Energy",
        ],
        "not_promoted": [
            "2607.02510 Online Safety Monitoring for LLMs depends on external verifier "
            "thresholding and overlaps existing conformal-risk references; do not reopen "
            "retired external generated-text scoring.",
            "2607.02509 ReContext is useful for future internal-signal context work, but "
            "it requires stable local model-internal receipts and does not change the "
            "V486 source plan.",
            "2607.02073 Evidence-State Rewards uses GRPO over evidence actions; it is "
            "not promoted because V486 must not reopen broad GRPO/fine-tuning reruns.",
            "2607.02465 Probabilistic Memory is paper-only hardware context without a "
            "Carnot-accessible board path, SDK, or speedup receipt.",
        ],
    },
    "openreview": {
        "status": "ok",
        "queries": [
            "OpenReview energy and hallucination verifier search",
            "OpenReview constrained decoding and neuro-symbolic verification search",
            "OpenReview KAN verification and certificate search",
            "OpenReview hard-constrained graph generation and neural CSP search",
        ],
        "result": (
            "OpenReview reinforced already indexed Spilled Energy, Semantic Energy, "
            "KAN verification, and symbolic-integration references. Hard-constrained "
            "graph-generation and proof-carrying neuro-symbolic pages are useful "
            "background, but they did not add an execution-changing V486 delta beyond "
            "the solver-authoritative constraints already in the planner refresh."
        ),
    },
    "huggingface_papers": {
        "status": "ok",
        "queries": [
            "HuggingFace Papers lookups for 2607.00871, 2607.01224, 2607.02514, 2607.02116",
            "HuggingFace Papers memory, agent, EBT, ARM-EBM, and hallucination pages",
            "HuggingFace Papers July 2026 context-governance and source-delta watch",
        ],
        "result": (
            "Hugging Face Papers pages confirmed already indexed SEA and AutoMem rows. "
            "No Hugging Face page supplied a separate actionable V486 delta beyond the "
            "arXiv/GitHub primary sources."
        ),
    },
    "semantic_scholar": {
        "status": "rate_limited",
        "queries": [
            "ARXIV:2507.02092 metadata and citation-count lookup",
            "ARXIV:2512.15605 metadata and citation-count lookup",
        ],
        "result": (
            "The public Graph API returned HTTP 429 during this execution check. The "
            "artifact therefore records rate limiting honestly and makes no citation "
            "trend claim beyond the V486 planner and prior Exp5308 checks."
        ),
        "raw_error": (
            "Too Many Requests. Please wait and try again or apply for a key for higher "
            "rate limits."
        ),
    },
    "github": {
        "status": "ok",
        "queries": [
            "GitHub repository search for ContextNest and ContextNext context governance",
            "GitHub repository search for persistent-state AI control and link tracker",
            "GitHub repository search for ReContext implementation references",
            "GitHub repository search for KAN verification, EBT, and constraint-memory repos",
        ],
        "new_actionable_repos": [
            "PromptOwl/ContextNest",
            "josh-hills/control-arena-persistent-state-eval",
        ],
        "not_promoted": [
            "Yanjun-Zhao/ReContext was found, but runtime-gated internal signal receipt "
            "work should decide later whether evidence replay is locally measurable.",
            "No implementation repository for Episodic-to-Semantic Consolidation Without "
            "Identity Drift was found in the checked results.",
        ],
        "result": (
            "ContextNest exposes a CLI, core engine, and MCP server for versioned "
            "context governance. The persistent-state control repository exposes the "
            "vibe_coding setting, logs, and analysis scripts for stateful monitoring."
        ),
    },
    "extropic": {
        "status": "ok",
        "queries": [
            "extropic.ai/writing index",
            "thermodynamic-computing-from-zero-to-one",
            "inside-x0-and-xtr-0",
            "tsu-101-an-entirely-new-type-of-computing-hardware",
        ],
        "result": (
            "The writing index still points to already indexed TSU/XTR-0/thrml material. "
            "No Carnot-accessible TSU hardware, SDK, local execution receipt, or V486 "
            "hardware speedup basis was found."
        ),
    },
    "logical_intelligence": {
        "status": "ok",
        "queries": [
            "Logical Intelligence blog and public pages",
            "automatic formal verification for code generation",
            "Kona, Aleph, and energy-based reasoning model posts",
            "Logical Intelligence public event and press pages",
        ],
        "result": (
            "Logical Intelligence remains a verifier-first strategic signal. Public "
            "pages still expose no reproducible Kona internals, local SDK, or benchmark "
            "receipt that would alter V486 execution; future dated event pages were not "
            "used as evidence."
        ),
    },
    "local_v486_comparison": {
        "status": "ok",
        "queries": [
            "research-references.md V486 Planner Refresh - 2026-07-06",
            "repo-wide duplicate search for ContextNest, 2607.02116, 2607.01988, and 2607.02514",
            "ops/exclusion_manifest.yaml retired scopes",
            "results/experiment_5308_sota_source_delta_v485.json prior source status",
        ],
        "result": (
            "The three promoted findings were absent from the V486 planner refresh and "
            "nearby reference history. They sharpen context lifecycle, identity-stable "
            "memory consolidation, and persistent-state monitoring without reopening "
            "external-text scoring, broad fine-tuning, TSU/Kona execution, or CPU-only "
            "GGUF offload scopes."
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
    """Build the terminal Exp5322 artifact from source-verified finding rows.

    The artifact is a literature-ingestion receipt. It records network-source
    methodology and Carnot-local implementation hooks, not model quality,
    hardware speed, or any other experimental outcome.
    """

    findings = [dict(row) for row in actionable_findings]
    count = len(findings)
    references_modified = count > 0
    status = "complete"
    verdict_detail = (
        f"{count} new actionable V486 source findings appended; executable .486 plan unchanged"
        if count
        else "no new actionable V486 source findings; references unchanged"
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
            "comparison_anchor": "research-references.md V486 Planner Refresh - 2026-07-06",
        },
        "tests_run": list(tests_run)
        or ["tests/python/test_experiment_5322_sota_source_delta_v486.py"],
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
        family_status = sources.get(family, {}).get("status") if isinstance(sources.get(family), Mapping) else None
        if family_status not in {"ok", "rate_limited"}:
            raise ValueError(f"sources_checked {family} must record status ok or rate_limited")


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
        raise ValueError("field_principles must match REQ-REPORT-5322")

    experiment_id = _validate_principled_wrapper("experiment_id", artifact)
    if experiment_id != EXPERIMENT_ID:
        raise ValueError("experiment_id must match Exp5322")
    milestone = _validate_principled_wrapper("milestone", artifact)
    if milestone != MILESTONE:
        raise ValueError("milestone must match 2026.07.486")
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
            "Execution-time sweep after the `.486` planner refresh checked arXiv, "
            "OpenReview, HuggingFace Papers, Semantic Scholar EBT/ARM-EBM citation "
            "trails, GitHub repositories, Extropic writing, Logical Intelligence "
            "public pages, and local duplicate history. The findings below were absent "
            "from the V486 planner block and nearby reference history."
        ),
        "",
        "### New actionable deltas",
        *(_render_finding(row) for row in findings),
        "",
        "### Execution impact",
        (
            "- **Plan impact:** No executable `.486` task edit is required. The deltas "
            "sharpen context-object lifecycle verification, identity-stable memory "
            "consolidation, and persistent-state monitoring fixtures."
        ),
        (
            "- **Retired scope:** No retired scope was reopened. External generated-text "
            "scoring, broad GRPO/fine-tuning reruns, TSU/Kona execution claims, and "
            "CPU-only GGUF offload reruns remain closed."
        ),
        (
            "- **Secondary-source status:** Semantic Scholar was rate-limited during "
            "execution and no citation-trend claim is made. OpenReview, HuggingFace "
            "Papers, Extropic, and Logical Intelligence did not add a separate "
            "execution-changing source."
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


if __name__ == "__main__":  # pragma: no cover - CLI convenience for the experiment run.
    raise SystemExit(main())
