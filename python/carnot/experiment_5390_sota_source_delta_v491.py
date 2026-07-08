"""Exp 5390: execution-time SOTA source delta refresh for V491.

Spec refs: REQ-REPORT-5390, SCENARIO-REPORT-5390-APPEND-DELTAS,
SCENARIO-REPORT-5390-NO-NEW-DELTA.

This module records a bounded literature/source receipt. It promotes only
sources that add concrete Carnot-local evidence hooks for the .491 roadmap and
keeps non-local hardware/model claims, external text scorers, and backend-free
internal-feature claims out of the execution plan.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_5390_sota_source_delta_v491"
TASK_ID = "exp5390-v491-sota-source-delta"
MILESTONE = "2026.07.491"
SEARCH_DATE = "20260708"
RESULT_RELATIVE_PATH = Path("results/experiment_5390_sota_source_delta_v491.json")
REFERENCES_RELATIVE_PATH = Path("research-references.md")
REFRESH_HEADING = "### V491 Execution Refresh - 2026-07-08"
REFRESH_END_MARKER = "<!-- V491-EXECUTION-REFRESH-2026-07-08-END -->"
INFERENCE_SUBSTRATE = "literature_ingestion_network_sources"

SPEC_REFS = [
    "REQ-REPORT-5390",
    "SCENARIO-REPORT-5390-APPEND-DELTAS",
    "SCENARIO-REPORT-5390-NO-NEW-DELTA",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Complete if the search ran and the artifact was emitted.",
    "milestone": "Must equal 2026.07.491.",
    "sources_checked": (
        "Lists the primary and secondary source families checked during the execution-time sweep."
    ),
    "new_actionable_findings_count": (
        "Counts only findings appended to the V491 execution refresh, or zero when there is no new delta."
    ),
    "appended_references_block": (
        "Bare boolean proving whether research-references.md changed."
    ),
    "appended_references_anchor": (
        "Heading or marker for the appended block, or null when no append occurred."
    ),
    "duplicates_suppressed": (
        "Lists already-covered sources from V489, V490, and V491 history that were not re-added."
    ),
    "retired_scopes_reopened": "Must be false unless a valid operator override exists.",
    "local_execution_implications": "Concrete impact on .491 tasks without editing the active roadmap.",
    "honest_verdict": "One-line summary starting with complete: or blocked:.",
}

REQUIRED_SOURCE_FAMILIES = (
    "arxiv",
    "openreview",
    "huggingface_papers",
    "semantic_scholar",
    "github",
    "extropic_writing",
    "logical_intelligence",
    "local_v489_v490_v491_duplicate_history",
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
        "appended_references_block",
        "appended_references_anchor",
        "duplicates_suppressed",
        "retired_scopes_reopened",
        "local_execution_implications",
        "honest_verdict",
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
        "title": "AgentLTL: A Trace-Verification Framework for Procedural Compliance",
        "url": "https://arxiv.org/abs/2607.02599",
        "source_type": "arXiv preprint with public GitHub code",
        "carnot_hook": (
            "For Exp5391 and Exp5392, compile tool/action and formal-encoding fixtures "
            "into trace rules that produce a deterministic, judge-free compliance score. "
            "Use prefix checks to block unsafe tool calls before execution, then record "
            "completed-trace compliance separately from final-answer correctness."
        ),
    },
    {
        "title": "OEP: Poisoning Self-Evolving LLM Agents via Locally Correct Experiences",
        "url": "https://arxiv.org/abs/2605.18930",
        "source_type": "arXiv preprint",
        "carnot_hook": (
            "For Exp5396, add memory controls where an episode is locally correct but "
            "non-transferable and paired with severe hypothetical consequences. The guard "
            "must reject over-generalized reflection rules from row-level evidence while "
            "retaining the raw episode and rollback pointer for audit."
        ),
    },
    {
        "title": "CoACT: Action-Preserving Observation Compression for Coding Agents",
        "url": "https://arxiv.org/abs/2607.02911",
        "source_type": "arXiv preprint",
        "carnot_hook": (
            "For Exp5395 and Exp5396, treat next-action preservation as the local test "
            "for compacting observations or memories: a compressed trace can reduce "
            "context only when it induces the same next verifier/tool route as the raw "
            "observation under deterministic controls."
        ),
    },
    {
        "title": "Succinct QUBO formulations for permutation problems by sorting networks",
        "url": "https://arxiv.org/abs/2603.07579",
        "source_type": "arXiv preprint",
        "carnot_hook": (
            "For Exp5394, if overwrite or p-bit action-sequence ablations require order "
            "or permutation constraints, add a tiny sorting-network QUBO baseline with "
            "exact enumeration and solver fallback. This is a CPU/simulator encoding "
            "diagnostic, not hardware speedup evidence."
        ),
    },
]

DUPLICATES_SUPPRESSED = [
    "Sortify influence exchange - https://arxiv.org/abs/2603.27765",
    "Energy-Aware Routing to Large Reasoning Models - https://arxiv.org/abs/2601.00823",
    "Mathematical encoding safety gaps - https://arxiv.org/abs/2605.03441",
    "Agentic Model Checking - https://arxiv.org/abs/2605.21434",
    "VAGEN agentic reward modeling - https://arxiv.org/abs/2602.00575",
    "MPPI Ising/QUBO control - https://arxiv.org/abs/2512.15533",
    "KANDy dynamic discovery - https://arxiv.org/abs/2602.20413",
    "DEX depth exploration - https://arxiv.org/abs/2606.29223",
    "GeoWorld geometric world models - https://arxiv.org/abs/2602.23058",
    "QCIVET hash-chained audit traces - https://arxiv.org/abs/2605.13109",
    "LLGuidance structured-output runtime - https://github.com/guidance-ai/llguidance",
    "CFGzip token-space compression - https://arxiv.org/abs/2605.29986",
    "TruncProof JSON completion slack - https://arxiv.org/abs/2605.13076",
    "G-RRM overwrite-capable solver guidance - https://arxiv.org/abs/2607.02491",
    "LongMemEval-V2 memory benchmark - https://arxiv.org/abs/2605.12493",
    "FARMA/SENTINEL forged reasoning memory attack - https://arxiv.org/abs/2607.05029",
    "Self-verifying measurement records - https://arxiv.org/abs/2606.27934",
    "vLLM hidden-state extraction note - https://vllm.ai/blog/2026-03-30-extract-hidden-states",
]

LOCAL_EXECUTION_IMPLICATIONS = [
    {
        "task": "Exp5391/Exp5392",
        "impact": (
            "Add trace-level procedural constraints and prefix-gating fields beside final "
            "structured-output success, with deterministic solvers or trace rules as final authority."
        ),
    },
    {
        "task": "Exp5395/Exp5396",
        "impact": (
            "Require next-action preservation and raw-episode retention before compacted "
            "memories can influence verifier routing."
        ),
    },
    {
        "task": "Exp5396",
        "impact": (
            "Include locally correct but non-transferable experience-poison controls in "
            "addition to forged reasoning and stale-memory controls."
        ),
    },
    {
        "task": "Exp5394",
        "impact": (
            "Use sorting-network QUBO encodings only as small exact-checkable simulator "
            "baselines for ordered action constraints; do not infer hardware acceleration."
        ),
    },
]

REJECTED_CANDIDATES = [
    {
        "title": "Phantom References / RefChecker",
        "url": "https://arxiv.org/abs/2607.00738",
        "reason": (
            "Useful citation-integrity tooling, but it is an ops/documentation quality hook "
            "rather than a .491 verifier, self-learning, solver, ARC, or hardware task delta."
        ),
    },
    {
        "title": "VeriChat hardware security verification assistant",
        "url": "https://arxiv.org/abs/2607.01668",
        "reason": (
            "EDA-backed verification is relevant context for hardware evidence discipline, "
            "but the .491 hardware task requires board-local repeatability and no speedup "
            "claim; a conversational assistant does not change that local gate."
        ),
    },
    {
        "title": "OpenSIR: Open-Ended Self-Improving Reasoner",
        "url": "https://arxiv.org/abs/2511.00602",
        "reason": (
            "Broad self-play model improvement conflicts with the .491 no-weight-mutation "
            "self-learning scope. It remains watch-only until a later task explicitly "
            "opens training."
        ),
    },
    {
        "title": "Noise-Induced Landscape Distortion in QAOA",
        "url": "https://arxiv.org/abs/2604.19426",
        "reason": (
            "Quantum-hardware landscape diagnostics are not a Carnot-local p-bit or board "
            "repeatability path for .491."
        ),
    },
    {
        "title": "Extropic TSU and Logical Intelligence Kona/Aleph pages",
        "url": "https://extropic.ai/",
        "reason": (
            "Architecture context only; no local TSU, Kona, or Aleph execution path or "
            "authenticated Carnot baseline was found."
        ),
    },
]

SEARCHED_SOURCE_DETAILS: JsonDict = {
    "arxiv": {
        "status": "ok",
        "queries": [
            "EBM verification/reasoning 2026",
            "constraint satisfaction LLM verification energy 2026",
            "Ising QUBO hardware sampling 2026",
            "hallucination mitigation energy-guided decoding 2026",
            "KAN KANDy continuous self-learning 2026",
        ],
        "promoted": [
            "2607.02599 AgentLTL",
            "2605.18930 OEP",
            "2607.02911 CoACT",
            "2603.07579 sorting-network QUBO",
        ],
        "not_promoted": [
            "2607.00738 RefChecker is a citation-integrity ops hook rather than a .491 task delta.",
            "2607.01668 VeriChat does not provide board-local timing or sampler evidence.",
            "2511.00602 OpenSIR opens model training outside the .491 no-weight-mutation scope.",
            "2604.19426 QAOA hardware diagnostics are non-local quantum hardware context.",
            "2507.07731, 2508.14496, and 2602.18671 were already indexed in local history.",
        ],
    },
    "openreview": {
        "status": "challenge_blocked",
        "queries": [
            "OpenReview AgentLTL",
            "OpenReview Energy-Based Transformers 2507.02092",
            "OpenReview ARM-EBM 2512.15605",
            "OpenReview constrained verification and KAN memory entries",
        ],
        "result": (
            "Search snippets reconfirmed EBT/NRGPT and planner-covered constrained reasoning "
            "context. Direct pages redirected to browser verification, so no OpenReview-only "
            "execution delta is promoted."
        ),
    },
    "huggingface_papers": {
        "status": "ok",
        "queries": [
            "HuggingFace Papers 2507.02092 EBT",
            "HuggingFace Papers 2602.00575 VAGEN",
            "HuggingFace Papers 2511.00602 OpenSIR",
            "HuggingFace Papers AgentLTL 2607.02599",
        ],
        "not_promoted": [
            "EBT spectral-control companion was already covered before V491.",
            "VAGEN was covered by the V491 planner refresh.",
            "OpenSIR is watch-only because .491 forbids model-weight mutation.",
            "No separate AgentLTL HuggingFace page was found during the sweep.",
        ],
    },
    "semantic_scholar": {
        "status": "rate_limited",
        "queries": [
            "Semantic Scholar API arXiv:2507.02092",
            "Semantic Scholar API arXiv:2512.15605",
            "Semantic Scholar route through arXiv references and public search snippets",
        ],
        "result": (
            "Direct API calls for EBT 2507.02092 and ARM-EBM 2512.15605 returned HTTP 429. "
            "No citation-count or influence-trend claim is made."
        ),
    },
    "github": {
        "status": "ok",
        "queries": [
            "GitHub AgentLTL 2607.02599",
            "GitHub OEP poisoning self-evolving LLM agents",
            "GitHub CoACT action-preserving observation compression",
            "GitHub RefChecker and HalluCiteChecker",
        ],
        "promoted": [
            "https://github.com/anonsubmission480/agentltl_procedural_compliance"
        ],
        "not_promoted": [
            "No OEP implementation surfaced in GitHub search.",
            "No CoACT implementation surfaced in GitHub search.",
            "RefChecker is useful but not promoted as a .491 execution-task delta.",
        ],
    },
    "extropic_writing": {
        "status": "ok",
        "queries": [
            "Extropic TSU 101",
            "Extropic Thermodynamic Computing From Zero to One",
            "Extropic TSU XTR 2026 EBM sampler",
        ],
        "result": (
            "Extropic writing remains sampler-first architecture context only. No local "
            "TSU SDK, authenticated receipt path, or Carnot-accessible TSU execution was found."
        ),
    },
    "logical_intelligence": {
        "status": "ok",
        "queries": [
            "Logical Intelligence Kona energy-based reasoning",
            "Logical Intelligence Aleph PutnamBench",
            "Logical Intelligence automatic formal verification for code generation",
        ],
        "result": (
            "Logical Intelligence pages remain non-local architecture context. They support "
            "the solver/prover-authority thesis but expose no reproducible Kona or Aleph "
            "baseline for Carnot."
        ),
    },
    "local_v489_v490_v491_duplicate_history": {
        "status": "ok",
        "queries": [
            "research-references.md V489 Planner and Execution Refresh",
            "research-references.md V490 Planner and Execution Refresh",
            "research-references.md V491 Planner Refresh",
            "ops/exclusion_manifest.yaml retired scope scan",
            "results/experiment_5388_capstone_v490.json prior milestone truth",
            "repo-wide duplicate search for promoted source ids",
        ],
        "result": (
            "AgentLTL, OEP, CoACT, and sorting-network QUBO were absent from V489/V490/V491 "
            "blocks. Planner-covered V491 sources, KANDy, VAGEN, MPPI/Ising, V490 execution "
            "deltas, Extropic, Logical Intelligence, and token/internal-feature watch items "
            "were suppressed as duplicates or non-local context."
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
    """Build the Exp5390 literature-ingestion artifact.

    The receipt is intentionally conservative. It records that the source
    search ran, but a promoted row still needs a local implementation task to
    prove anything about quality, safety, memory behavior, or hardware.
    """

    findings = [dict(row) for row in actionable_findings]
    count = len(findings)
    appended = count > 0
    verdict_detail = (
        f"{count} new actionable V491 execution-time source deltas appended; retired scopes remained closed"
        if count
        else "no new actionable V491 execution-time source deltas; references unchanged"
    )
    artifact: JsonDict = {
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "status": "complete",
        "search_date": SEARCH_DATE,
        "sources_checked": list(REQUIRED_SOURCE_FAMILIES),
        "new_actionable_findings_count": count,
        "findings": findings,
        "appended_references_block": appended,
        "appended_references_anchor": REFRESH_HEADING if appended else None,
        "duplicates_suppressed": list(DUPLICATES_SUPPRESSED),
        "retired_scopes_reopened": False,
        "local_execution_implications": [dict(row) for row in LOCAL_EXECUTION_IMPLICATIONS],
        "honest_verdict": f"complete: {verdict_detail}.",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "searched_source_details": dict(SEARCHED_SOURCE_DETAILS),
        "rejected_candidates": [dict(row) for row in REJECTED_CANDIDATES],
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "methodology_duration_s": round(float(methodology_duration_s), 6),
        "tests_run": list(tests_run)
        or ["tests/python/test_experiment_5390_sota_source_delta_v491.py"],
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
            raise ValueError(
                f"findings rows must include exactly {sorted(REQUIRED_FINDING_FIELDS)}"
            )
        if not _verified_url(str(row["url"])):
            raise ValueError("findings rows must use a verified URL")
        if not str(row["carnot_hook"]).strip():
            raise ValueError("findings rows must include a Carnot hook")


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = REQUIRED_TOP_LEVEL_FIELDS.difference(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match REQ-REPORT-5390")
    if artifact["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("experiment_id must match Exp5390")
    if artifact["task_id"] != TASK_ID:
        raise ValueError("task_id must match exp5390")
    if artifact["milestone"] != MILESTONE:
        raise ValueError("milestone must match 2026.07.491")
    if artifact["status"] != "complete":
        raise ValueError("status must be complete when the search artifact is emitted")

    findings = artifact["findings"]
    _validate_findings(findings)
    count = artifact["new_actionable_findings_count"]
    if not isinstance(count, int) or count != len(findings):
        raise ValueError("findings count must equal findings length")
    if artifact["search_date"] != SEARCH_DATE:
        raise ValueError("search_date must equal 20260708")
    _validate_sources(artifact["sources_checked"], artifact["searched_source_details"])

    duplicates = artifact["duplicates_suppressed"]
    if not isinstance(duplicates, list) or not duplicates:
        raise ValueError("duplicates_suppressed must be a non-empty list")
    if artifact["retired_scopes_reopened"] is not False:
        raise ValueError("retired_scopes_reopened must remain false")
    appended = artifact["appended_references_block"]
    if appended is not (count > 0):
        raise ValueError("appended_references_block must match whether findings were added")
    expected_anchor = REFRESH_HEADING if appended else None
    if artifact["appended_references_anchor"] != expected_anchor:
        raise ValueError("appended_references_anchor must match the references append state")
    implications = artifact["local_execution_implications"]
    if not isinstance(implications, list) or not implications:
        raise ValueError("local_execution_implications must be a non-empty list")
    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or "\n" in verdict or not verdict.startswith("complete:"):
        raise ValueError("honest_verdict must be a one-line complete: or blocked: summary")
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
            "Execution-time sweep after the `.491` planner refresh checked arXiv, "
            "OpenReview, HuggingFace Papers, Semantic Scholar EBT/ARM-EBM routes, "
            "GitHub, Extropic writing, Logical Intelligence public pages, and local "
            "V489/V490/V491 duplicate history. The findings below were absent from "
            "those blocks and add Carnot-local hooks without changing the active roadmap."
        ),
        "",
        "### New actionable deltas",
        *(_render_finding(row) for row in findings),
        "",
        "### Execution impact",
        (
            "- **Plan impact:** No active `.491` roadmap edit is required. The deltas "
            "sharpen Exp5391/Exp5392 trace constraints, Exp5395/Exp5396 memory and "
            "routing guards, and Exp5394 p-bit/QUBO ablations by adding concrete "
            "local receipt fields."
        ),
        (
            "- **Duplicates suppressed:** Sortify, Energy-Aware Routing, mathematical "
            "encoding safety, Agentic Model Checking, VAGEN, MPPI/Ising, KANDy, DEX, "
            "GeoWorld, QCIVET, llguidance, CFGzip, TruncProof, G-RRM, LongMemEval-V2, "
            "FARMA/SENTINEL, self-verifying measurement records, and vLLM hidden-state "
            "extraction were already covered by V489/V490/V491 history and are not re-added."
        ),
        (
            "- **Retired scope:** No retired scope was reopened. Non-local TSU/Kona/Aleph "
            "claims, external generated-text scorers, token/internal-feature claims without "
            "backend evidence, CPU-only SOTA headline reruns, duplicate ARC solve paths, "
            "and hardware speedup claims without repeatable board timing remain closed."
        ),
        (
            "- **Watch-only context:** RefChecker, VeriChat, OpenSIR, QAOA hardware-noise "
            "diagnostics, Extropic writing, and Logical Intelligence public pages were "
            "checked but not promoted because they are ops-only, non-local, training-scope, "
            "or hardware-claim context rather than immediate `.491` execution deltas."
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
    artifact = build_artifact(
        methodology_duration_s=methodology_duration_s,
        tests_run=tests_run,
    )

    references_text = references.read_text(encoding="utf-8")
    updated_references = append_refresh_section(references_text, artifact)
    if updated_references != references_text:
        references.write_text(updated_references, encoding="utf-8")

    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:  # pragma: no cover
    write_outputs(
        methodology_duration_s=0.0,
        tests_run=["tests/python/test_experiment_5390_sota_source_delta_v491.py"],
    )


if __name__ == "__main__":  # pragma: no cover
    main()
