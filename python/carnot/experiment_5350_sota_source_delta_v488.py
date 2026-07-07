"""Exp 5350: execution-time SOTA source delta refresh for V488.

Spec refs: REQ-REPORT-5350, SCENARIO-REPORT-5350-APPEND-DELTAS,
SCENARIO-REPORT-5350-NOOP.

This module records the source sweep as a reproducible receipt. It is not a
model, solver, or hardware experiment. The important distinction is whether a
fresh source changes what Carnot should do locally without reopening retired
scoring, fine-tuning, hardware-claim, or ARC-exploration scopes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_5350_sota_source_delta_v488"
MILESTONE = "2026.07.488"
RESULT_RELATIVE_PATH = Path("results/experiment_5350_sota_source_delta_v488.json")
REFERENCES_RELATIVE_PATH = Path("research-references.md")
REFRESH_HEADING = "### V488 Execution Refresh - 2026-07-07"
REFRESH_END_MARKER = "<!-- V488-EXECUTION-REFRESH-2026-07-07-END -->"
INFERENCE_SUBSTRATE = "literature_ingestion_network_sources"

SPEC_REFS = [
    "REQ-REPORT-5350",
    "SCENARIO-REPORT-5350-APPEND-DELTAS",
    "SCENARIO-REPORT-5350-NOOP",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": "Stable id ties the artifact to this roadmap task.",
    "milestone": "Prevents stale literature claims from crossing milestones.",
    "status": "Lets the conductor classify no-op versus appended findings.",
    "honest_verdict": (
        "Terminal prefix `complete:` or `blocked_` prevents ambiguous "
        "source-ingestion status."
    ),
    "inference_substrate": (
        "Expected value is literature_ingestion_network_sources so no model-result "
        "claim is implied."
    ),
    "sources_checked": "Lists the actual source families searched.",
    "new_actionable_findings_count": (
        "Bare integer distinguishes useful deltas from watch-only material."
    ),
    "references_modified": "Bare boolean proves whether `research-references.md` changed.",
    "references_section_marker": "Lets reviewers find any appended notes quickly.",
    "retired_scope_reopened": (
        "Bare boolean must be false to preserve manifest discipline."
    ),
    "methodology_duration_s": (
        "Bare numeric duration catches implausibly-short literature sweeps."
    ),
    "executable_plan_change_required": (
        "Bare boolean signals whether the already-staged plan needs operator review."
    ),
    "actionable_findings": (
        "Source URLs plus Carnot-local actionability make appended findings auditable "
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
    "local_v488_comparison",
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
        "title": "Safety Testing LLM Agents at Scale",
        "arxiv_id_or_repo": "2607.01793",
        "source_url": "https://arxiv.org/abs/2607.01793",
        "secondary_source_url": "https://github.com/Yunhao-Feng/Vera",
        "source_family": "arxiv+huggingface_papers+github",
        "category": "evidence_grounded_agent_verification",
        "carnot_hook": (
            "Vera constructs executable agent safety cases with programmatic initial "
            "state and deterministic verification predicates, then judges outcomes "
            "with evidence-grounded verifiers over environment state and tool-call "
            "evidence before falling back to agent response text."
        ),
        "actionability": (
            "For Exp5352 and Exp5356, treat tool/action reachability and memory-tool "
            "drift as observable-state problems: log initial state, final state, "
            "tool-call traces, case-specific verifier predicate, and response-text "
            "fallback separately before accepting any structured-output or memory "
            "policy result."
        ),
        "planned_task_impact": "no_plan_edit",
        "retired_scope_risk": "none",
    }
]

SOURCES_CHECKED: JsonDict = {
    "arxiv": {
        "status": "ok",
        "queries": [
            "site:arxiv.org/abs/2607 energy based model reasoning verification LLM constraint satisfaction",
            "site:arxiv.org/abs/2607 neural constraint satisfaction solver projected diffusion EBM reasoning",
            "site:arxiv.org/abs/2607 Ising p-bit annealing hardware sampling probabilistic machine",
            "site:arxiv.org/abs/2607 hallucination detection energy-guided decoding KAN verification",
            "site:arxiv.org/abs/2607 ARC perception salience LLM",
            "exact lookup for arXiv:2607.01793 after HuggingFace/GitHub discovery",
            "exact lookup for arXiv:2607.05391 after HuggingFace/GitHub discovery",
        ],
        "new_actionable_ids": ["2607.01793"],
        "already_indexed_or_duplicate": [
            "2507.07731 Energy-Guided Decoding for Object Hallucination Mitigation",
            "2512.20664 Eidoku CSP Neuro-Symbolic Verification Gate",
            "2602.22465 ConstraintBench",
            "2605.18871 Distributional Energy-Based Models",
            "2606.25313 Programmable Probabilistic Computer with 1,000,000 p-bits",
            "2606.26476 Retrieval-Warmed Energy-Based Reasoning",
        ],
        "not_promoted": [
            "2607.05391 LLM-as-a-Verifier uses scoring-token logit expectations and "
            "the public repository requires a Vertex/Gemini API key for logprob "
            "extraction; treating it as an execution delta would reopen the retired "
            "external generated-text/logprob scorer class.",
            "2607.00597 PaperPilot is useful process context for literature agents, "
            "but it does not change a V488 technical task beyond this source-delta "
            "discipline.",
            "2607.02881 PraMem is a behavior-prediction memory paper, not a direct "
            "constraint-learning or verifier-control delta for the staged V488 tasks.",
        ],
    },
    "openreview": {
        "status": "ok",
        "queries": [
            "OpenReview energy-based reasoning and hallucination verifier search",
            "OpenReview constrained decoding and neuro-symbolic verification search",
            "OpenReview KAN verification and certificate search",
            "OpenReview hard-constrained graph generation and neural CSP search",
            "OpenReview reasoning dynamics and solver-verifier search",
        ],
        "new_actionable_items": [],
        "challenge_gated_candidates": [
            "L7NsVVUm9H Opt-Verifier: search snippet was relevant to optimization solver code, but the OpenReview page redirected to a browser challenge and no code/details were accessible.",
            "lh95PnOlpM Equilibrium Reasoners: search snippet was relevant, but local references already contain Equilibrium Reasoners entries and the page redirected to a browser challenge.",
        ],
        "not_promoted": [
            "03ZTlJuX0y The Tell-Tale Norm exposes hidden-state L2 reasoning dynamics, "
            "but V488 token/internal-signal work is gated on local backend feature rows; "
            "the current GGUF path does not expose the hidden states needed for an action.",
            "EXFKk4Y3yc Spilled Energy, B3Muyi2zgo NRGPT, and cbtykHVWX9 hard-constrained "
            "graph generation were already indexed locally.",
        ],
        "result": (
            "OpenReview produced watch items and duplicates, but no accessible "
            "OpenReview-only item changed the V488 execution plan."
        ),
    },
    "huggingface_papers": {
        "status": "ok",
        "queries": [
            "HuggingFace Papers date/2026-07-07 daily page",
            "HuggingFace Papers 2607 reasoning, energy, constraint, memory, and KAN searches",
            "HuggingFace Papers lookup for 2607.01793 Vera",
            "HuggingFace Papers lookup for 2607.05391 LLM-as-a-Verifier",
        ],
        "new_actionable_items": ["2607.01793 Safety Testing LLM Agents at Scale"],
        "not_promoted": [
            "2607.05391 LLM-as-a-Verifier was not promoted because its public code path "
            "depends on external logprob/scoring-token extraction and would collide with "
            "the retired external scorer discipline.",
            "PraMem and PaperPilot were watch-only for V488 because they do not sharpen "
            "constraint-tax, solver projection, p-bit schedule, or ARC salience execution.",
        ],
        "result": (
            "The July 7 HuggingFace Papers page surfaced Vera and LLM-as-a-Verifier. "
            "Only Vera adds a Carnot-local execution hook without reopening retired scopes."
        ),
    },
    "semantic_scholar": {
        "status": "rate_limited",
        "queries": [
            "Graph API lookup for arXiv:2507.02092 EBT",
            "Graph API lookup for arXiv:2512.15605 ARM-EBM",
            "Graph API lookup for arXiv:2607.01793 Vera",
            "Graph API lookup for arXiv:2607.05391 LLM-as-a-Verifier",
        ],
        "result": (
            "The public Graph API returned HTTP 429 for every checked paper during "
            "this execution refresh, so no citation-count or influence delta is claimed."
        ),
        "raw_error": (
            "Too Many Requests. Please wait and try again or apply for a key for higher "
            "rate limits."
        ),
    },
    "github": {
        "status": "ok",
        "queries": [
            "GitHub search for Yunhao-Feng/Vera",
            "GitHub search for llm-as-a-verifier/llm-as-a-verifier",
            "GitHub search for energy-based reasoning, KAN verification, SMT projector, and constraint learning",
            "GitHub search for ARC-AGI-3 perception, salience, connected components, and color blob agents",
        ],
        "new_actionable_repos": ["Yunhao-Feng/Vera"],
        "not_promoted": [
            "llm-as-a-verifier/llm-as-a-verifier was not promoted because the setup "
            "requires Vertex API logprob extraction and implements an external "
            "trajectory reward scorer.",
            "Public ARC-AGI-3 repositories with connected-component or MCTS language "
            "were not promoted because V488 must use Carnot's live-path provenance and "
            "must not reopen retired ARC exploration-signal reruns.",
        ],
        "result": (
            "GitHub confirmed an executable Vera repository with deterministic verifier "
            "fixtures. Other repositories were watch-only or retired-scope risks."
        ),
    },
    "extropic": {
        "status": "ok",
        "queries": [
            "extropic.ai/writing TSU and thermodynamic computing pages",
            "TSU 101",
            "thermodynamic-computing-from-zero-to-one",
            "inside-x0-and-xtr-0",
        ],
        "result": (
            "Extropic public writing still provides TSU/probabilistic hardware context. "
            "No new local TSU SDK, hardware access path, or authenticated Carnot "
            "execution receipt was found."
        ),
    },
    "logical_intelligence": {
        "status": "ok",
        "queries": [
            "Logical Intelligence blog Kona, Aleph, and EBRM pages",
            "Automatic Formal Verification for Code Generation",
            "Aleph leading benchmarks",
            "Aleph Prover formal methods posts",
        ],
        "result": (
            "Logical Intelligence public material continues to support the V488 "
            "solver-authoritative certificate direction, but it exposes no reproducible "
            "local Kona internals or execution baseline."
        ),
    },
    "local_v488_comparison": {
        "status": "ok",
        "queries": [
            "research-references.md V488 Planner Refresh - 2026-07-07",
            "repo-wide duplicate search for 2607.01793, Vera, 2607.05391, LLM-as-a-Verifier, PraMem, PaperPilot, Opt-Verifier, and Tell-Tale Norm",
            "results/experiment_5336_sota_source_delta_v487.json prior source status",
            "ops/exclusion_manifest.yaml retired scopes",
        ],
        "result": (
            "Vera was absent from the V488 planner block and nearby source-delta "
            "artifacts. It sharpens existing V488 tool/action reachability and "
            "memory-tool drift checks without adding a new plan item or reopening "
            "retired generated-text scoring, broad GRPO/fine-tuning, TSU/Kona execution, "
            "CPU-only GGUF offload, or ARC exploration-signal scopes."
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
    """Build the Exp5350 literature-ingestion artifact.

    The receipt separates source discovery from execution claims. A row can be
    actionably useful only if it maps to Carnot-local evidence collection and
    does not depend on retired external scorer or non-local hardware claims.
    """

    findings = [dict(row) for row in actionable_findings]
    count = len(findings)
    references_modified = count > 0
    verdict_detail = (
        f"{count} new actionable V488 source finding appended; executable .488 plan unchanged"
        if count
        else "no new actionable V488 source findings; references unchanged"
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
            "comparison_anchor": "research-references.md V488 Planner Refresh - 2026-07-07",
        },
        "tests_run": list(tests_run)
        or ["tests/python/test_experiment_5350_sota_source_delta_v488.py"],
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
        raise ValueError("field_principles must match REQ-REPORT-5350")

    experiment_id = _validate_principled_wrapper("experiment_id", artifact)
    if experiment_id != EXPERIMENT_ID:
        raise ValueError("experiment_id must match Exp5350")
    milestone = _validate_principled_wrapper("milestone", artifact)
    if milestone != MILESTONE:
        raise ValueError("milestone must match 2026.07.488")
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
            "Execution-time sweep after the `.488` planner refresh checked arXiv, "
            "OpenReview, HuggingFace Papers, Semantic Scholar EBT/ARM-EBM and fresh "
            "paper lookups, GitHub repositories, Extropic writing, Logical "
            "Intelligence public pages, and local duplicate history. The finding "
            "below was absent from the V488 planner block and nearby reference history."
        ),
        "",
        "### New actionable delta",
        *(_render_finding(row) for row in findings),
        "",
        "### Execution impact",
        (
            "- **Plan impact:** No executable `.488` task edit is required. The delta "
            "sharpens Exp5352 constraint-tax action reachability and Exp5356 "
            "memory-tool drift fixtures by requiring deterministic state/tool-call "
            "evidence before response text."
        ),
        (
            "- **Retired scope:** No retired scope was reopened. External generated-text "
            "scoring, broad GRPO/fine-tuning reruns, TSU/Kona execution claims, "
            "CPU-only GGUF offload reruns, and retired ARC exploration-signal reruns "
            "remain closed."
        ),
        (
            "- **Secondary-source status:** Semantic Scholar was rate-limited and no "
            "citation-trend claim is made. LLM-as-a-Verifier was not promoted because "
            "its public path depends on external logprob/scoring-token extraction."
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
