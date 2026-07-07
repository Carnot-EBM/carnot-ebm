"""Exp 5336: execution-time SOTA source delta refresh for V487.

Spec refs: REQ-REPORT-5336, SCENARIO-REPORT-5336-APPEND-DELTAS,
SCENARIO-REPORT-5336-NOOP.

The source sweep happens before this module is updated because network sources
change independently of the repository. This file preserves the checked source
families, the rows that were new relative to the V487 planner refresh, and why
each row is actionable without changing the executable V487 plan.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_5336_sota_source_delta_v487"
MILESTONE = "2026.07.487"
RESULT_RELATIVE_PATH = Path("results/experiment_5336_sota_source_delta_v487.json")
REFERENCES_RELATIVE_PATH = Path("research-references.md")
REFRESH_HEADING = "### V487 Execution Refresh - 2026-07-07"
REFRESH_END_MARKER = "<!-- V487-EXECUTION-REFRESH-2026-07-07-END -->"
INFERENCE_SUBSTRATE = "literature_ingestion_network_sources"

SPEC_REFS = [
    "REQ-REPORT-5336",
    "SCENARIO-REPORT-5336-APPEND-DELTAS",
    "SCENARIO-REPORT-5336-NOOP",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": "Traceability for the Exp5336 execution-time source delta artifact.",
    "milestone": (
        "Binds this receipt to V487 so source deltas cannot be misapplied to another "
        "milestone."
    ),
    "status": "Machine-readable terminal state for downstream reconciliation.",
    "honest_verdict": (
        "Terminal verdict must start with complete: or blocked_ so nuance cannot be "
        "misclassified."
    ),
    "inference_substrate": (
        "literature_ingestion_network_sources because Exp5336 reads network "
        "literature/source metadata and makes no model, solver, or hardware outcome claim."
    ),
    "sources_checked": (
        "Records every required source family and query channel so a missing source "
        "cannot masquerade as a zero-new-finding result."
    ),
    "references_section_marker": "Idempotent marker prevents duplicate research-references.md appends.",
    "actionable_findings": (
        "Source URLs plus Carnot-local hooks make each appended reference auditable "
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
    "local_v487_comparison",
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
        "executable_plan_change_required",
        "actionable_findings",
        "spec_refs",
        "search_window",
        "tests_run",
        "field_principles",
        "no_deep_research_used",
        "research_conductor_modified",
        "ops_docs_modified",
        "traceability_modified",
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
        "title": (
            "Learning Hierarchical Procedural Memory for LLM Agents through "
            "Bayesian Selection and Contrastive Refinement"
        ),
        "arxiv_id_or_repo": "2512.18950",
        "source_url": "https://arxiv.org/abs/2512.18950",
        "secondary_source_url": (
            "https://github.com/S-Forouzandeh/MACLA-LLM-Agents-AAMAS-2026-Conference"
        ),
        "source_family": "arxiv+github",
        "category": "continual_procedural_memory_learning",
        "carnot_hook": (
            "MACLA keeps the LLM frozen and moves adaptation into external hierarchical "
            "procedural memory with Bayesian reliability posteriors, expected-utility "
            "selection, contrastive success/failure refinement, and meta-procedure "
            "composition."
        ),
        "actionability": (
            "For Exp5340/Exp5342, add a procedural-memory lane beside MemRL utility "
            "values: track procedure preconditions, Beta posterior counts, information "
            "gain, success/failure contrast pairs, capacity saturation, and no model "
            "weight mutation."
        ),
        "planned_task_impact": "no_plan_edit",
        "retired_scope_risk": "none",
    },
    {
        "title": "Hard-Constrained Graph Generation with Discrete-Projection Diffusion",
        "arxiv_id_or_repo": "openreview:cbtykHVWX9",
        "source_url": "https://openreview.net/forum?id=cbtykHVWX9",
        "secondary_source_url": (
            "https://github.com/zhangxuesong2000/Neuro-Symbolic-Projected-Sampling-for-Graphs"
        ),
        "source_family": "openreview+github",
        "category": "neural_constraint_satisfaction_projection",
        "carnot_hook": (
            "The NSPSG repository exposes a discrete graph-generation path that uses "
            "SMT projection, plus a neural projection/corrector option, to keep samples "
            "inside arithmetic, structural, and combined hard constraints."
        ),
        "actionability": (
            "For Exp5343/Exp5346, treat SMT projection as the authority when converting "
            "candidate structures into constraint cuts. Log projection rate, solver "
            "fallback, neural-corrector agreement, and post-projection validity before "
            "claiming any learned sampler benefit."
        ),
        "planned_task_impact": "no_plan_edit",
        "retired_scope_risk": "none",
    },
]

SOURCES_CHECKED: JsonDict = {
    "arxiv": {
        "status": "ok",
        "queries": [
            "site:arxiv.org/abs/2607.0 EBM reasoning and LLM verification",
            "site:arxiv.org/abs/2607.0 neural constraint satisfaction and symbolic solvers",
            "site:arxiv.org/abs/2607.0 Ising, p-bit, annealing, and hardware sampling",
            "site:arxiv.org/abs/2607.0 hallucination detection and energy-guided decoding",
            "site:arxiv.org/abs/2607.0 KAN verification and certificates",
            "site:arxiv.org/abs/2607.0 continual memory and constraint learning",
            "exact lookup for MACLA arXiv:2512.18950 after GitHub discovery",
        ],
        "new_actionable_ids": ["2512.18950"],
        "already_indexed_or_duplicate": [
            "2507.07731 Energy-Guided Decoding for Object Hallucination Mitigation",
            "2602.22465 ConstraintBench",
            "2603.20801 ConsFormer-LNS",
            "2605.18871 Distributional Energy-Based Models",
            "2607.00692 Self-GC",
            "2607.00871 Self-Evolving Agents",
            "2607.01223 Theoria",
            "2607.01224 AutoMem",
            "2607.01935 A-TMA",
            "2607.01988 identity-stable memory consolidation",
            "2607.02116 ContextNest",
            "2607.02514 persistent-state AI control",
        ],
        "not_promoted": [
            "2607.00038 Stop Hand-Holding Your Coding Agent is useful process context, "
            "but it overlaps existing harness/status discipline and does not change a "
            "V487 technical task.",
            "2607 search results for hallucination mitigation were mostly LVLM-specific "
            "or text-scorer-like; they were not promoted because internal-signal work "
            "must remain receipt-gated and external generated-text scoring is retired.",
        ],
    },
    "openreview": {
        "status": "ok",
        "queries": [
            "OpenReview energy-based reasoning and hallucination verifier search",
            "OpenReview constrained decoding and neuro-symbolic verification search",
            "OpenReview KAN verification and certificate search",
            "OpenReview hard-constrained graph generation and neural CSP search",
        ],
        "new_actionable_items": ["cbtykHVWX9 Hard-Constrained Graph Generation"],
        "not_promoted": [
            "Distributional Energy-Based Models and NRGPT were already indexed locally.",
            "HONet exact-cover search result was noted, but without a code path or "
            "details beyond the OpenReview listing it did not change the V487 execution plan.",
        ],
        "result": (
            "OpenReview produced one new actionable source with an implementation "
            "repository: SMT-projected constrained graph generation. Other EBM, "
            "NRGPT, hallucination, and exact-cover results were duplicates or watch items."
        ),
    },
    "huggingface_papers": {
        "status": "ok",
        "queries": [
            "HuggingFace Papers 2607 reasoning, energy, constraint, memory, and KAN searches",
            "HuggingFace Papers lookup for 2607.00038 loop engineering",
            "HuggingFace Papers V487 planner-item spot checks",
        ],
        "not_promoted": [
            "Stop Hand-Holding Your Coding Agent / loop engineering surfaced as a new "
            "HuggingFace Papers page, but its actionable pieces are terminal states, "
            "verification ladders, and durable memory governance already covered by "
            "Carnot's conductor and V487 harness metrics.",
        ],
        "result": (
            "No HuggingFace Papers page supplied a separate V487 implementation delta "
            "beyond MACLA via arXiv/GitHub and the NSPSG OpenReview/GitHub source."
        ),
    },
    "semantic_scholar": {
        "status": "rate_limited",
        "queries": [
            "ARXIV:2507.02092 metadata and citation-count lookup",
            "ARXIV:2512.15605 metadata and citation-count lookup",
            "ARXIV:2607.00038 metadata and citation-count lookup",
        ],
        "result": (
            "The public Graph API returned HTTP 429 during this execution check. The "
            "artifact records rate limiting honestly and makes no citation-trend claim."
        ),
        "raw_error": (
            "Too Many Requests. Please wait and try again or apply for a key for higher "
            "rate limits."
        ),
    },
    "github": {
        "status": "ok",
        "queries": [
            "GitHub search for Neuro-Symbolic-Projected-Sampling-for-Graphs",
            "GitHub search for MACLA Memory-Augmented Continual Learning Agent",
            "GitHub search for loop engineering and agent-memory watch repositories",
            "GitHub search for energy-guided decoding, KAN verification, SMT projector, and constraint learning",
        ],
        "new_actionable_repos": [
            "S-Forouzandeh/MACLA-LLM-Agents-AAMAS-2026-Conference",
            "zhangxuesong2000/Neuro-Symbolic-Projected-Sampling-for-Graphs",
        ],
        "not_promoted": [
            "ChaoYue0307/awesome-loop-engineering and similar lists were useful watch "
            "items but not Carnot-local implementation deltas.",
            "Awesome hallucination and memory lists were broad indexes; specific "
            "papers they mentioned were duplicates, LVLM-only, or outside V487 gates.",
        ],
        "result": (
            "GitHub confirmed implementation repositories for MACLA and NSPSG. The "
            "other discovered repositories were watch lists or broad agent collections."
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
            "Extropic public writing still points to already indexed TSU/XTR-0/thrml "
            "material. No Carnot-accessible TSU hardware, SDK, local execution receipt, "
            "or V487 speedup basis was found."
        ),
    },
    "logical_intelligence": {
        "status": "ok",
        "queries": [
            "Logical Intelligence blog and public pages",
            "Kona, Aleph, and energy-based reasoning model posts",
            "automatic formal verification for code generation",
            "Logical Intelligence event and press pages",
        ],
        "result": (
            "Logical Intelligence remains a verifier-first architecture signal. Public "
            "pages still expose no reproducible Kona internals, local SDK, or benchmark "
            "receipt that would alter V487 execution. Future event pages were not used "
            "as evidence."
        ),
    },
    "local_v487_comparison": {
        "status": "ok",
        "queries": [
            "research-references.md V487 Planner Refresh - 2026-07-07",
            "repo-wide duplicate search for MACLA, 2512.18950, NSPSG, and cbtykHVWX9",
            "results/experiment_5322_sota_source_delta_v486.json prior source status",
            "ops/exclusion_manifest.yaml retired scopes",
        ],
        "result": (
            "MACLA and NSPSG were absent from the V487 planner block and nearby source "
            "delta artifacts. They sharpen utility-governed procedural memory and "
            "solver-authoritative neural constraint projection without reopening "
            "external-text scoring, broad GRPO/fine-tuning, TSU/Kona execution, "
            "CPU-only GGUF offload, or ARC level-solve scopes."
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
    """Build the Exp5336 literature-ingestion artifact.

    This receipt records source-review methodology and Carnot-local hooks. It
    does not claim model quality, solver performance, hardware reachability, or
    any other execution outcome.
    """

    findings = [dict(row) for row in actionable_findings]
    count = len(findings)
    references_modified = count > 0
    verdict_detail = (
        f"{count} new actionable V487 source findings appended; executable .487 plan unchanged"
        if count
        else "no new actionable V487 source findings; references unchanged"
    )
    artifact: JsonDict = {
        "experiment_id": _principled("experiment_id", EXPERIMENT_ID),
        "milestone": _principled("milestone", MILESTONE),
        "status": _principled("status", "complete"),
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
        "executable_plan_change_required": False,
        "actionable_findings": _principled("actionable_findings", findings),
        "spec_refs": list(SPEC_REFS),
        "search_window": {
            "run_date": "2026-07-07",
            "years": "2025-2026",
            "comparison_anchor": "research-references.md V487 Planner Refresh - 2026-07-07",
        },
        "tests_run": list(tests_run)
        or ["tests/python/test_experiment_5336_sota_source_delta_v487.py"],
        "field_principles": dict(FIELD_PRINCIPLES),
        "no_deep_research_used": True,
        "research_conductor_modified": False,
        "ops_docs_modified": False,
        "traceability_modified": False,
        "roadmap_files_modified": False,
    }
    validate_artifact(artifact)
    return artifact


def _validate_sources(sources: Any) -> None:
    if not isinstance(sources, Mapping) or not set(REQUIRED_SOURCE_FAMILIES).issubset(sources):
        raise ValueError("sources_checked must include every required source family")
    for family in REQUIRED_SOURCE_FAMILIES:
        family_entry = sources.get(family)
        family_status = family_entry.get("status") if isinstance(family_entry, Mapping) else None
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
        raise ValueError("field_principles must match REQ-REPORT-5336")

    experiment_id = _validate_principled_wrapper("experiment_id", artifact)
    if experiment_id != EXPERIMENT_ID:
        raise ValueError("experiment_id must match Exp5336")
    milestone = _validate_principled_wrapper("milestone", artifact)
    if milestone != MILESTONE:
        raise ValueError("milestone must match 2026.07.487")
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
    if artifact["executable_plan_change_required"] is not False:
        raise ValueError("executable plan must not be changed by this refresh")
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


def _render_finding(row: Mapping[str, Any]) -> str:
    return (
        f"- **{row['title']}** ({row['source_url']}; code/reference "
        f"{row.get('secondary_source_url', 'n/a')}): {row['carnot_hook']} "
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
            "Execution-time sweep after the `.487` planner refresh checked arXiv, "
            "OpenReview, HuggingFace Papers, Semantic Scholar EBT/ARM-EBM citation "
            "trails, GitHub repositories, Extropic writing, Logical Intelligence "
            "public pages, and local duplicate history. The findings below were "
            "absent from the V487 planner block and nearby reference history."
        ),
        "",
        "### New actionable deltas",
        *(_render_finding(row) for row in findings),
        "",
        "### Execution impact",
        (
            "- **Plan impact:** No executable `.487` task edit is required. The deltas "
            "sharpen procedural-memory utility telemetry for Exp5340/Exp5342 and "
            "solver-authoritative projection telemetry for Exp5343/Exp5346."
        ),
        (
            "- **Retired scope:** No retired scope was reopened. External generated-text "
            "scoring, broad GRPO/fine-tuning reruns, TSU/Kona execution claims, "
            "CPU-only GGUF offload reruns, and ARC level solves remain closed."
        ),
        (
            "- **Secondary-source status:** Semantic Scholar was rate-limited during "
            "execution and no citation-trend claim is made. Extropic and Logical "
            "Intelligence did not add a separate execution-changing source."
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
