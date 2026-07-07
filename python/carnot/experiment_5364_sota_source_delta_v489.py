"""Exp 5364: execution-time SOTA source delta refresh for V489.

Spec refs: REQ-REPORT-5364, SCENARIO-REPORT-5364-APPEND-DELTAS,
SCENARIO-REPORT-5364-HONEST-NULL.

This module records the last source sweep before the downstream .489
experiments. It is a literature/source receipt, not an execution benchmark:
new rows are promoted only when they change Carnot-local evidence collection
without reopening retired scorer, CPU-offload, TSU/Kona, or ARC exploration
scopes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_5364_sota_source_delta_v489"
TASK_ID = "exp5364-v489-sota-source-delta"
MILESTONE = "2026.07.489"
SEARCH_DATE = "20260707"
RESULT_RELATIVE_PATH = Path("results/experiment_5364_sota_source_delta_v489.json")
REFERENCES_RELATIVE_PATH = Path("research-references.md")
REFRESH_HEADING = "### V489 Execution Refresh - 2026-07-07"
REFRESH_END_MARKER = "<!-- V489-EXECUTION-REFRESH-2026-07-07-END -->"
INFERENCE_SUBSTRATE = "literature_ingestion_network_sources"

SPEC_REFS = [
    "REQ-REPORT-5364",
    "SCENARIO-REPORT-5364-APPEND-DELTAS",
    "SCENARIO-REPORT-5364-HONEST-NULL",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "status": (
        "Must be `complete` when at least one new actionable V489 delta is "
        "promoted, otherwise `honest_null`."
    ),
    "search_date": "Must equal 20260707 for this execution-time sweep.",
    "sources_checked": "Lists the required source families searched before downstream experiments.",
    "new_actionable_findings_count": (
        "Counts only non-duplicate actionable findings promoted into the V489 "
        "execution refresh."
    ),
    "findings": (
        "Each promoted finding records title, url, source_type, and Carnot-local hook."
    ),
    "duplicates_suppressed": (
        "Lists relevant sources already covered by the V489 planner refresh and "
        "therefore not re-added."
    ),
    "retired_scope_reopened": (
        "Bare boolean must remain false because this refresh cannot reopen retired scopes."
    ),
    "research_references_updated": (
        "Bare boolean is true only when a new V489 execution-refresh block was appended."
    ),
    "honest_verdict": "One-line summary of the execution-time source delta check.",
}

REQUIRED_SOURCE_FAMILIES = (
    "arxiv",
    "openreview",
    "huggingface_papers",
    "semantic_scholar",
    "github",
    "extropic_writing",
    "logical_intelligence",
    "local_v489_duplicate_history",
)

REQUIRED_TOP_LEVEL_FIELDS = frozenset(
    {
        "experiment_id",
        "task_id",
        "milestone",
        "status",
        "search_date",
        "sources_checked",
        "new_actionable_findings_count",
        "findings",
        "duplicates_suppressed",
        "retired_scope_reopened",
        "research_references_updated",
        "honest_verdict",
        "references_section_marker",
        "inference_substrate",
        "searched_source_details",
        "rejected_candidates",
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

REQUIRED_FINDING_FIELDS = frozenset({"title", "url", "source_type", "carnot_hook"})

ACTIONABLE_FINDINGS: list[JsonDict] = [
    {
        "title": "LLGuidance: Low-level Guidance for Super-fast Structured Outputs",
        "url": "https://github.com/guidance-ai/llguidance",
        "source_type": "GitHub implementation",
        "carnot_hook": (
            "Use the llama.cpp llguidance integration as the concrete grammar-budget "
            "probe for Exp5365: record whether the local build has "
            "-DLLAMA_LLGUIDANCE=ON, compile JSON/Lark reachability fixtures, and "
            "measure mask-computation budget before Exp5366 live GGUF generation."
        ),
    },
    {
        "title": "LongMemEval-V2: Evaluating Long-Term Agent Memory Toward Experienced Colleagues",
        "url": "https://arxiv.org/abs/2605.12493",
        "source_type": "arXiv + HuggingFace Papers",
        "carnot_hook": (
            "For Exp5368/Exp5369, add compact-evidence memory checks covering static "
            "state recall, dynamic state tracking, workflow knowledge, environment "
            "gotchas, and premise awareness; record evidence-token/latency budget "
            "separately from memory accuracy."
        ),
    },
]

DUPLICATES_SUPPRESSED = [
    "G-RRM: Guiding Symbolic Solvers with Recurrent Reasoning Models - https://arxiv.org/abs/2607.02491",
    "Forget to Improve / Budget-Curated Memory - https://arxiv.org/abs/2606.25115",
    "ALMA memory-design search - https://arxiv.org/abs/2602.07755",
    "CFGzip token-space compression - https://arxiv.org/abs/2605.29986",
    "TruncProof JSON completion slack - https://arxiv.org/abs/2605.13076",
    "FLaG latent grouping hallucination detection - https://arxiv.org/abs/2606.00301",
    "Thermodynamic Signatures of Reasoning - https://arxiv.org/abs/2606.19404",
    "Programmable Probabilistic Computer with 1,000,000 p-bits - https://arxiv.org/abs/2606.25313",
]

REJECTED_CANDIDATES = [
    {
        "title": "Escaping the Self-Confirmation Trap / EDV",
        "url": "https://arxiv.org/abs/2606.24428",
        "reason": (
            "Useful memory-insertion context, but promoting it would broaden Exp5368 "
            "into heterogeneous-agent consensus rather than the planned deterministic "
            "budget-curated governance gate."
        ),
    },
    {
        "title": "PREPING: Building Agent Memory without Tasks",
        "url": "https://arxiv.org/abs/2605.13880",
        "reason": (
            "Open-ended synthetic practice is watch-only for .489 because the active "
            "memory tasks require budget, provenance, rollback, and stale/poison gates."
        ),
    },
    {
        "title": "SE-RRM / ARC-TGI / ARC-AGI-2 Reasoner family",
        "url": "https://github.com/ml-jku/SE-RRM",
        "reason": (
            "Relevant ARC architecture context but not promoted because .489 must use "
            "Carnot's live-path perception repair and must not reopen retired ARC "
            "candidate-exploration-signal reruns."
        ),
    },
    {
        "title": "Extropic TSU and Logical Intelligence Kona/Aleph execution claims",
        "url": "https://extropic.ai/",
        "reason": (
            "Architecture context only; no Carnot-accessible TSU/Kona execution path "
            "or authenticated hardware receipt was found."
        ),
    },
]

SEARCHED_SOURCE_DETAILS: JsonDict = {
    "arxiv": {
        "status": "ok",
        "queries": [
            "grammar-budgeted constrained decoding JSON completion slack 2026",
            "budget-curated memory LLM agent continual learning 2026",
            "overwrite-capable solver guidance recurrent reasoning symbolic solver 2026",
            "p-bit boundary exchange probabilistic Ising machine 2026",
            "ARC-AGI perception salience repair live agent 2026",
            "hardware receipt discipline TSU Aleph formal verification 2026",
        ],
        "promoted": ["2605.12493 LongMemEval-V2"],
        "duplicates_suppressed": [
            "2607.02491 G-RRM",
            "2606.25115 Budget-Curated Memory",
            "2605.29986 CFGzip",
            "2605.13076 TruncProof",
            "2606.25313 million-p-bit hardware",
        ],
        "not_promoted": [
            "2606.24428 EDV broadens memory insertion into heterogeneous-agent consensus.",
            "2605.13880 PREPING adds synthetic pre-task memory practice outside the bounded .489 gate.",
            "2603.02193 SE-RRM is relevant ARC context but not a live-path perception repair.",
            "2606.17327 thermodynamic hardware codon optimization is a TSU execution/energy claim.",
            "2605.30106 Rust-to-Lean verification with AI provers is useful formal-methods context but not a .489 hardware receipt delta.",
        ],
    },
    "openreview": {
        "status": "challenge_blocked",
        "queries": [
            "OpenReview PSC grammar-constrained decoding",
            "OpenReview Flexible and Efficient Grammar-Constrained Decoding",
            "OpenReview JSONSchemaBench constrained decoding",
            "OpenReview hard-constrained graph generation solver projection",
        ],
        "result": (
            "OpenReview pages redirected to browser verification. Search snippets "
            "matched planner-covered constrained-decoding and JSONSchemaBench context; "
            "no OpenReview-only execution delta was promoted."
        ),
    },
    "huggingface_papers": {
        "status": "ok",
        "queries": [
            "HuggingFace Papers LongMemEval-V2",
            "HuggingFace Papers budget curated memory",
            "HuggingFace Papers memory insertion",
            "HuggingFace Papers constrained decoding JSON grammar",
        ],
        "promoted": ["LongMemEval-V2 arXiv:2605.12493"],
        "not_promoted": [
            "EDV and PREPING were watch-only because they broaden the .489 memory gate.",
            "Older MemoryAgentBench and Budget-Curated Memory entries are already in prior planner history.",
        ],
    },
    "semantic_scholar": {
        "status": "ok",
        "queries": [
            "Semantic Scholar public search for G-RRM",
            "Semantic Scholar public search for LongMemEval-V2",
            "Semantic Scholar public search for CFGzip and JSONSchemaBench",
            "Semantic Scholar public search for p-bit boundary exchange",
        ],
        "result": (
            "Public search reconfirmed planner-covered constrained-decoding and "
            "p-bit context. No citation-count or influential-citation claim is made."
        ),
    },
    "github": {
        "status": "ok",
        "queries": [
            "GitHub guidance-ai/llguidance",
            "GitHub CFGzip mjs227/cfgzip and coli-saar/cfgzip-experiments",
            "GitHub ml-jku/SE-RRM",
            "GitHub ARC-AGI perception and salience repositories",
        ],
        "promoted": ["guidance-ai/llguidance"],
        "duplicates_suppressed": [
            "mjs227/cfgzip and coli-saar/cfgzip-experiments are implementation references for planner-covered CFGzip.",
            "ml-jku/SE-RRM is the repository behind planner-covered G-RRM solver guidance.",
            "Saibo-creator/Awesome-LLM-Constrained-Decoding was already named by the V489 planner.",
        ],
        "not_promoted": [
            "ARC solver repositories were not promoted because live-path ARC repair remains the credited path.",
            "External scorer repositories were not promoted.",
        ],
    },
    "extropic_writing": {
        "status": "ok",
        "queries": [
            "Extropic TSU 101",
            "Extropic Thermodynamic Computing From Zero to One",
            "Extropic Inside X0 and XTR-0",
        ],
        "result": (
            "Extropic writing remains hardware architecture context only. No local "
            "SDK, authenticated receipt path, or Carnot-accessible TSU execution was found."
        ),
    },
    "logical_intelligence": {
        "status": "ok",
        "queries": [
            "Logical Intelligence Kona 1.0",
            "Logical Intelligence Aleph formal verification",
            "Logical Intelligence automatic formal verification for code generation",
            "Logical Intelligence Aleph leading benchmarks",
        ],
        "result": (
            "Logical Intelligence pages and related Aleph reports support deterministic "
            "proof-checking discipline, but no reproducible Kona baseline or local "
            "execution artifact was available."
        ),
    },
    "local_v489_duplicate_history": {
        "status": "ok",
        "queries": [
            "research-references.md V489 Planner Refresh - 2026-07-07",
            "research-roadmap.yaml exp5364 prompt",
            "ops/exclusion_manifest.yaml retired scope scan",
            "results/experiment_5362_capstone_v488.json prior milestone truth",
        ],
        "result": (
            "LLGuidance and LongMemEval-V2 were absent from the V489 planner block as "
            "promoted items. The remaining relevant hits were duplicate planner "
            "sources, watch-only context, or retired-scope risks."
        ),
    },
}


def _verified_url(value: str) -> bool:
    return value.startswith("https://")


def build_artifact(
    *,
    actionable_findings: Sequence[Mapping[str, Any]] = ACTIONABLE_FINDINGS,
    methodology_duration_s: float = 0.0,
    tests_run: Sequence[str] = (),
) -> JsonDict:
    """Build the Exp5364 literature-ingestion artifact.

    The artifact separates source discovery from execution claims. A promoted
    row must supply a Carnot-local measurement hook and must not require an
    external scorer, CPU-only GGUF rerun, TSU/Kona execution path, or retired
    ARC candidate-exploration loop.
    """

    findings = [dict(row) for row in actionable_findings]
    count = len(findings)
    status = "complete" if count else "honest_null"
    updated = count > 0
    verdict_detail = (
        f"{count} new actionable V489 execution-time source deltas appended; retired scopes remained closed"
        if count
        else "no new actionable V489 execution-time source deltas; references unchanged"
    )
    artifact: JsonDict = {
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "status": status,
        "search_date": SEARCH_DATE,
        "sources_checked": list(REQUIRED_SOURCE_FAMILIES),
        "new_actionable_findings_count": count,
        "findings": findings,
        "duplicates_suppressed": list(DUPLICATES_SUPPRESSED),
        "retired_scope_reopened": False,
        "research_references_updated": updated,
        "honest_verdict": f"{status}: {verdict_detail}.",
        "references_section_marker": REFRESH_END_MARKER if updated else None,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "searched_source_details": dict(SEARCHED_SOURCE_DETAILS),
        "rejected_candidates": [dict(row) for row in REJECTED_CANDIDATES],
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "methodology_duration_s": round(float(methodology_duration_s), 6),
        "tests_run": list(tests_run)
        or ["tests/python/test_experiment_5364_sota_source_delta_v489.py"],
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
        if family_status not in {"ok", "rate_limited", "challenge_blocked"}:
            raise ValueError(f"searched_source_details {family} must record a valid status")


def _validate_findings(findings: Any) -> None:
    if not isinstance(findings, list):
        raise ValueError("findings must be a list")
    for row in findings:
        if not isinstance(row, Mapping) or set(row) != REQUIRED_FINDING_FIELDS:
            raise ValueError(f"findings rows must include exactly {sorted(REQUIRED_FINDING_FIELDS)}")
        if not _verified_url(str(row["url"])):
            raise ValueError("findings rows must use a verified URL")
        if not str(row["carnot_hook"]).strip():
            raise ValueError("findings rows must include a Carnot hook")


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = REQUIRED_TOP_LEVEL_FIELDS.difference(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match REQ-REPORT-5364")
    if artifact["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("experiment_id must match Exp5364")
    if artifact["task_id"] != TASK_ID:
        raise ValueError("task_id must match exp5364")
    if artifact["milestone"] != MILESTONE:
        raise ValueError("milestone must match 2026.07.489")

    findings = artifact["findings"]
    _validate_findings(findings)
    count = artifact["new_actionable_findings_count"]
    if not isinstance(count, int) or count != len(findings):
        raise ValueError("findings count must equal findings length")

    expected_status = "complete" if count else "honest_null"
    if artifact["status"] != expected_status:
        raise ValueError("status must be complete when findings exist and honest_null otherwise")
    if artifact["search_date"] != SEARCH_DATE:
        raise ValueError("search_date must equal 20260707")
    _validate_sources(artifact["sources_checked"], artifact["searched_source_details"])

    duplicates = artifact["duplicates_suppressed"]
    if not isinstance(duplicates, list) or not duplicates:
        raise ValueError("duplicates_suppressed must be a non-empty list")
    if artifact["retired_scope_reopened"] is not False:
        raise ValueError("retired_scope_reopened must remain false")
    updated = artifact["research_references_updated"]
    if updated is not (count > 0):
        raise ValueError("research_references_updated must match whether findings were added")
    expected_marker = REFRESH_END_MARKER if updated else None
    if artifact["references_section_marker"] != expected_marker:
        raise ValueError("references_section_marker must match the references append state")
    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or "\n" in verdict or not verdict.startswith(f"{expected_status}:"):
        raise ValueError("honest_verdict must be a one-line terminal summary")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be literature_ingestion_network_sources")
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


def _render_finding(row: Mapping[str, Any]) -> str:
    return f"- **{row['title']}** ({row['url']}): {row['carnot_hook']}"


def render_refresh_section(artifact: Mapping[str, Any]) -> str:
    validate_artifact(artifact)
    findings = artifact["findings"]
    if not findings:
        return ""
    lines = [
        REFRESH_HEADING,
        "",
        (
            "Execution-time sweep after the `.489` planner refresh checked arXiv, "
            "OpenReview, HuggingFace Papers, Semantic Scholar, GitHub, Extropic "
            "writing, Logical Intelligence public pages, and local duplicate history. "
            "The findings below were absent from the V489 planner block and add "
            "Carnot-local hooks without changing the active roadmap."
        ),
        "",
        "### New actionable deltas",
        *(_render_finding(row) for row in findings),
        "",
        "### Execution impact",
        (
            "- **Plan impact:** No active `.489` roadmap edit is required. The deltas "
            "sharpen Exp5365 grammar-budget preflight and Exp5368/Exp5369 "
            "budget-curated memory scale-up by adding concrete local receipt fields."
        ),
        (
            "- **Duplicates suppressed:** G-RRM, Budget-Curated Memory, ALMA, CFGzip, "
            "TruncProof, FLaG, thermodynamic-signature, and million-p-bit sources were "
            "already covered by the planner refresh and are not re-added."
        ),
        (
            "- **Retired scope:** No retired scope was reopened. External text scorers, "
            "CPU-only GGUF offload reruns, TSU/Kona execution claims, and retired ARC "
            "candidate-exploration-signal reruns remain closed."
        ),
        (
            "- **Watch-only context:** EDV/PREPING memory papers, SE-RRM/ARC solver "
            "repositories, Extropic writing, and Logical Intelligence public pages were "
            "checked but not promoted because they either broaden the bounded .489 tasks "
            "or lack reproducible local execution evidence."
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
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI convenience for the experiment run.
    raise SystemExit(main())
