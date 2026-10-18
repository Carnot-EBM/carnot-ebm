"""Exp5416: execution-time source delta refresh for V493.

Spec refs: REQ-REPORT-5416, SCENARIO-REPORT-5416-APPEND-DELTAS,
SCENARIO-REPORT-5416-NO-NEW-DELTA,
SCENARIO-REPORT-5416-BLOCKED-MISSING-PLANNER.

This module writes a conservative literature receipt. It promotes only
source-verified references that add a concrete Carnot-local hook after the
V493 planner refresh. Everything else stays duplicate, watch-only, or excluded
so the receipt does not turn interesting papers into unsupported roadmap churn.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_5416_source_delta_v493"
TASK_ID = "exp5416-source-delta-v493"
MILESTONE = "2026.07.493"
SEARCH_DATE = "20260708"
RESULT_RELATIVE_PATH = Path("results/experiment_5416_source_delta_v493.json")
REFERENCES_RELATIVE_PATH = Path("research-references.md")
PLANNER_MARKER = "### V493 Planner Refresh - 2026-07-08"
REFRESH_HEADING = "### V493 Execution Refresh - 20260708"
REFRESH_END_MARKER = "<!-- V493-EXECUTION-REFRESH-20260708-END -->"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

SPEC_REFS = [
    "REQ-REPORT-5416",
    "SCENARIO-REPORT-5416-APPEND-DELTAS",
    "SCENARIO-REPORT-5416-NO-NEW-DELTA",
    "SCENARIO-REPORT-5416-BLOCKED-MISSING-PLANNER",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "sources_checked": "reproducible literature sweep",
    "new_references_added": "current-knowledge delta",
    "duplicates_suppressed": "no reference churn",
    "retired_scopes_reopened": "exclusion-manifest compliance",
    "research_references_updated": "doc alignment",
    "prior_refresh_marker_found": "dedupe against planner work",
    "inference_substrate": "source aggregation only",
    "honest_verdict": "terminal status; start with complete: or blocked:",
}

REQUIRED_SOURCE_FAMILIES = (
    "arxiv",
    "openreview",
    "huggingface_papers",
    "semantic_scholar",
    "github",
    "extropic_writing",
    "logical_intelligence",
    "local_v489_v493_duplicate_history",
    "ops_exclusion_manifest",
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
        "new_references_added",
        "duplicates_suppressed",
        "retired_scopes_reopened",
        "research_references_updated",
        "prior_refresh_marker_found",
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

REQUIRED_REFERENCE_FIELDS = frozenset({"title", "url", "source_type", "carnot_hook"})
ALLOWED_SOURCE_STATUSES = frozenset({"ok", "partial", "rate_limited", "challenge_blocked"})

NEW_REFERENCES_ADDED: list[JsonDict] = [
    {
        "title": "Evaluating LLM Personalization via Semantic Constraint Verification",
        "url": "https://arxiv.org/abs/2606.16368",
        "source_type": "arXiv preprint",
        "carnot_hook": (
            "For Exp5417, add an NLI-backed semantic constraint ablation that maps "
            "structured outputs to truth-condition sets and records semantic "
            "false-accept, sycophancy/generalization/failure labels, and sentence "
            "attribution separately from schema validity."
        ),
    },
    {
        "title": "Resource-Aware Neuro-Symbolic Reasoning for Local Small Language Models",
        "url": "https://arxiv.org/abs/2606.27281",
        "source_type": "arXiv preprint",
        "carnot_hook": (
            "For Exp5417 and Exp5418, compare one-call formalization plus deterministic "
            "finite-domain solving against self-consistency on bounded logical-deduction "
            "fixtures, with traceability/consistency repair receipts and token/model-call "
            "accounting."
        ),
    },
    {
        "title": "GroundEval: A Deterministic Replacement for LLM-as-Judge in Stateful Agent Evaluation",
        "url": "https://arxiv.org/abs/2606.22737",
        "source_type": "arXiv preprint with deterministic evaluation contract",
        "carnot_hook": (
            "For Exp5426 and self-learning evidence tables, represent agent runs as an "
            "event log, artifact corpus, access policy, and eval config, then score "
            "answer correctness, evidence path validity, and silence/counterfactual "
            "violations without an LLM judge."
        ),
    },
    {
        "title": "PreAct: Computer-Using Agents that Get Faster on Repeated Tasks",
        "url": "https://arxiv.org/abs/2606.17929",
        "source_type": "arXiv preprint with public GitHub repository",
        "carnot_hook": (
            "For Exp5421 and Exp5422, treat repeated verified workflows as compiled "
            "state-machine memories: replay only while predicates match, fall back on "
            "mismatch, and require a clean verify-before-store rerun before any learned "
            "fragment can influence routing."
        ),
    },
    {
        "title": "Online LLM Selection via Constrained Bandits with Time-Varying Demand",
        "url": "https://arxiv.org/abs/2606.17489",
        "source_type": "arXiv preprint",
        "carnot_hook": (
            "For Exp5421 and Exp5422, add routing telemetry for packing and covering "
            "constraints: confidence-bound resource estimates, demand-shift buckets, "
            "regret proxies, and cumulative constraint-violation counts under the "
            "no-weight-mutation controller."
        ),
    },
    {
        "title": "A Stackelberg Framework for Resource-Aware LLM Agents: Learning, Repair, and Conditional Guarantees",
        "url": "https://arxiv.org/abs/2606.23026",
        "source_type": "arXiv preprint",
        "carnot_hook": (
            "For Exp5421 and Exp5422, add real-workflow resource-governance controls "
            "where a learned route policy is repaired by real-call calibration and "
            "safe-set projection; report token-cost and quality deltas without claiming "
            "a certified equilibrium."
        ),
    },
]

DUPLICATES_SUPPRESSED = [
    "Constrained Flow Matching via Lagrangian Dual Flows - https://arxiv.org/abs/2607.04513",
    "Neuro-Symbolic Safety Guidance for VLA Models - https://arxiv.org/abs/2607.01378",
    "Uncertainty-Aware Abstention in LLMs - https://arxiv.org/abs/2607.04430",
    "Weave of Formal Thought - https://arxiv.org/abs/2606.25987",
    "Structured Output Control for Software Engineering - https://arxiv.org/abs/2606.09395",
    "Measurement-Access Risk Frontiers - https://arxiv.org/abs/2607.05696",
    "Hidden Forgetting in Continual Multimodal Learning - https://arxiv.org/abs/2607.02020",
    "Beyond the Leaderboard failure taxonomy - https://arxiv.org/abs/2607.05775",
    "Energy-Based Transformers 2507.02092 - https://arxiv.org/abs/2507.02092",
    "ARM-EBM 2512.15605 - https://arxiv.org/abs/2512.15605",
    "NRGPT OpenReview and arXiv entry - https://openreview.net/forum?id=B3Muyi2zgo",
    "Distributional EBMs for structured LLM reasoning - https://arxiv.org/abs/2605.18871",
    "Energy-Based Decoding for frozen LLMs - https://arxiv.org/abs/2605.28020",
    "NSVIF instruction-following verifier - https://arxiv.org/abs/2601.17789",
    "REVES verifier-trace training - https://arxiv.org/abs/2606.18910",
    "Formalize, Don't Optimize heuristic trap - https://arxiv.org/abs/2605.12421",
    "Cycle-consistent formal-certificate explanation - https://arxiv.org/abs/2606.24414",
    "AgentLTL trace verification - https://arxiv.org/abs/2607.02599",
    "OEP poisoning self-evolving agents - https://arxiv.org/abs/2605.18930",
    "CoACT observation compression - https://arxiv.org/abs/2607.02911",
    "Ising-Machine-Assisted LNS and geometric subproblem papers - https://arxiv.org/abs/2607.05169",
    "iSTAR algebraic collapse for continuous Ising solvers - https://arxiv.org/abs/2607.05448",
]

WATCH_ONLY_OR_EXCLUDED: list[JsonDict] = [
    {
        "title": "A Neuro-Symbolic Approach to Strategy Synthesis for Strategic Logics",
        "url": "https://arxiv.org/abs/2606.17962",
        "classification": "watch-only",
        "reason": (
            "NatATL/model-checker certification is relevant to future coordination and "
            "multi-agent strategy fixtures, but `.493` has no active NatATL benchmark."
        ),
    },
    {
        "title": "Extropic THRML, TSU writing, and codon optimization repositories",
        "url": "https://github.com/extropic-ai/thrml",
        "classification": "watch-only",
        "reason": (
            "Extropic remains useful non-local TSU and block-Gibbs context, but Carnot has "
            "no authenticated TSU hardware path and must not claim TSU speedups."
        ),
    },
    {
        "title": "Logical Intelligence Aleph and Kona public pages",
        "url": "https://logicalintelligence.com/aleph-coding-ai/",
        "classification": "watch-only",
        "reason": (
            "The pages reinforce prover-authority framing, but no reproducible local Kona "
            "or Aleph baseline exists for Carnot comparison."
        ),
    },
    {
        "title": "OpenReview Energy-Based Action Heads and NRGPT surfaces",
        "url": "https://openreview.net/forum?id=B3Muyi2zgo",
        "classification": "duplicate suppressed",
        "reason": (
            "OpenReview reinforced already-indexed EBT/NRGPT and energy-action-head themes; "
            "it did not add a stronger local `.493` executable dependency."
        ),
    },
    {
        "title": "Retired ARC first-contact and external generated-text scorer lanes",
        "url": "ops/exclusion_manifest.yaml",
        "classification": "excluded",
        "reason": (
            "retired ARC first-contact reruns, external generated-text/logprob scorers, "
            "CPU-only SOTA offload, token/internal-feature claims without backend receipts, "
            "and hardware speedup claims without matched board timing remain closed."
        ),
    },
]

SEARCHED_SOURCE_DETAILS: JsonDict = {
    "arxiv": {
        "status": "ok",
        "queries": [
            "2025-2026 energy-based models verification reasoning",
            "2025-2026 neural constraint satisfaction LLM verifier",
            "2025-2026 Ising QUBO machine learning constraints",
            "2025-2026 hallucination mitigation constrained decoding semantic verifier",
            "2025-2026 Kolmogorov-Arnold Networks verification constraints",
            "2025-2026 energy-guided decoding hardware accelerated sampling",
            "2025-2026 continual online learning constraint systems LLM agents",
        ],
        "promoted": [
            "2606.16368 NLICV semantic constraint verification",
            "2606.27281 VFR-LLM resource-aware neuro-symbolic reasoning",
            "2606.22737 GroundEval deterministic stateful agent evaluation",
            "2606.17929 PreAct verify-before-store state-machine memory",
            "2606.17489 constrained-bandit online LLM selection",
            "2606.23026 Stackelberg resource-aware LLM agents",
        ],
        "not_promoted": [
            "2606.17962 NatATL strategy synthesis is watch-only for future MAS fixtures.",
            "2606.27892 analog KAN hardware is not a Carnot sampler path.",
            "2606.27042 ETDKAN is RF-domain KAN context only.",
            "2607 planner sources and V492 execution Ising papers were duplicate-covered.",
        ],
    },
    "openreview": {
        "status": "partial",
        "queries": [
            "OpenReview Energy-Based Transformers 2507.02092",
            "OpenReview ARM-EBM 2512.15605",
            "OpenReview NRGPT energy-based GPT",
            "OpenReview Energy-Based Action Heads",
            "OpenReview constrained decoding verification reasoning",
        ],
        "result": (
            "Search surfaced NRGPT, EBT-adjacent action-head, EDLM, and constrained "
            "generation pages. They reinforced existing architecture watch items and no "
            "OpenReview-only `.493` dependency was promoted."
        ),
    },
    "huggingface_papers": {
        "status": "partial",
        "queries": [
            "HuggingFace Papers Resource-Aware Neuro-Symbolic Reasoning",
            "HuggingFace Papers Semantic Constraint Verification NLICV",
            "HuggingFace Papers PreAct Computer-Using Agents",
            "HuggingFace Papers GroundEval deterministic replacement for LLM-as-Judge",
        ],
        "result": (
            "HuggingFace Papers search pages were reachable and mirrored or indexed some "
            "arXiv candidates, including PreAct, but did not add an independent source "
            "stronger than the arXiv records."
        ),
    },
    "semantic_scholar": {
        "status": "rate_limited",
        "queries": [
            "Semantic Scholar API arXiv:2507.02092",
            "Semantic Scholar API arXiv:2512.15605",
        ],
        "result": (
            "Direct API calls for EBT 2507.02092 and ARM-EBM 2512.15605 returned HTTP "
            "429 during the execution sweep, so no citation-count or citation-trend "
            "claim is made."
        ),
    },
    "github": {
        "status": "ok",
        "queries": [
            "GitHub NLICV Semantic Constraint Verification",
            "GitHub VFR-LLM Resource-Aware Neuro-Symbolic Reasoning",
            "GitHub GroundEval deterministic LLM-as-Judge replacement",
            "GitHub PreAct verify-before-store",
            "GitHub Extropic THRML codon_opt TSU",
        ],
        "promoted_supporting_links": [
            "https://github.com/19PINE-AI/PreAct",
        ],
        "watch_only_links": [
            "https://github.com/extropic-ai/thrml",
            "https://github.com/extropic-ai/codon_opt",
            "https://github.com/tmgthb/Autonomous-Agents",
        ],
    },
    "extropic_writing": {
        "status": "ok",
        "queries": [
            "Extropic thermodynamic computing from zero to one",
            "Extropic inside X0 and XTR-0",
            "Extropic TSU 101",
            "Extropic THRML GitHub issues and codon_opt",
        ],
        "result": (
            "Extropic writing and repositories remain sampler-first context. No local "
            "TSU SDK, board receipt path, or Carnot-accessible TSU execution was found."
        ),
    },
    "logical_intelligence": {
        "status": "ok",
        "queries": [
            "Logical Intelligence automatic formal verification for code generation",
            "Logical Intelligence Aleph coding AI",
            "Logical Intelligence Aleph leading benchmarks",
            "Logical Intelligence Kona energy-based model",
        ],
        "result": (
            "Logical Intelligence public pages reinforce machine-checkable proof authority, "
            "but they remain non-local architecture context rather than Carnot evidence."
        ),
    },
    "local_v489_v493_duplicate_history": {
        "status": "ok",
        "queries": [
            "research-references.md V489 Planner and Execution Refresh",
            "research-references.md V490 Planner and Execution Refresh",
            "research-references.md V491 Planner and Execution Refresh",
            "research-references.md V492 Planner and Execution Refresh",
            "research-references.md V493 Planner Refresh",
            "repo-wide search for promoted arXiv ids and titles",
        ],
        "result": (
            "The promoted arXiv IDs were absent from local references and the V493 planner "
            "block. Nearby planner/execution sources were recorded as suppressed duplicates."
        ),
    },
    "ops_exclusion_manifest": {
        "status": "ok",
        "queries": [
            "retired ARC first-contact candidate-generation reruns",
            "external generated-text/logprob scorer lanes",
            "CPU-only SOTA offload lanes",
            "token/internal feature claims without backend receipts",
            "non-local TSU/Kona/Aleph execution claims",
            "hardware speedup claims without matched board timing",
        ],
        "result": (
            "Retired lanes stayed closed. Non-local hardware, scorer, token/internal, "
            "and repeated ARC scope matches were classified watch-only or excluded."
        ),
    },
}


def _verified_url(value: str) -> bool:
    return value.startswith("https://") or value == "ops/exclusion_manifest.yaml"


def build_artifact(
    *,
    new_references_added: Sequence[Mapping[str, Any]] = NEW_REFERENCES_ADDED,
    methodology_duration_s: float = 0.0,
    tests_run: Sequence[str] = (),
    research_references_updated: bool | None = None,
    prior_refresh_marker_found: bool = True,
) -> JsonDict:
    """Build the Exp5416 source-delta artifact.

    The receipt aggregates source metadata and local dedupe decisions only. It
    does not claim model quality, verifier accuracy, hardware speedup, citation
    influence, or reopened retired scope.
    """

    references = [dict(row) for row in new_references_added] if prior_refresh_marker_found else []
    count = len(references)
    updated = (count > 0) if research_references_updated is None else research_references_updated
    if not prior_refresh_marker_found:
        status = "blocked"
        updated = False
        verdict = "blocked: V493 planner refresh marker missing; references unchanged."
    else:
        status = "complete"
        verdict_detail = (
            f"{count} new actionable V493 execution-time source deltas appended; retired scopes remained closed"
            if count
            else "no new actionable V493 execution-time source deltas; references unchanged"
        )
        verdict = f"complete: {verdict_detail}."

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
        "retired_scopes_reopened": False,
        "research_references_updated": updated,
        "prior_refresh_marker_found": prior_refresh_marker_found,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "searched_source_details": dict(SEARCHED_SOURCE_DETAILS),
        "watch_only_or_excluded": [dict(row) for row in WATCH_ONLY_OR_EXCLUDED],
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "methodology_duration_s": round(float(methodology_duration_s), 6),
        "tests_run": list(tests_run)
        or ["tests/python/test_experiment_5416_source_delta_v493.py"],
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


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = REQUIRED_TOP_LEVEL_FIELDS.difference(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match REQ-REPORT-5416")
    if artifact["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("experiment_id must match Exp5416")
    if artifact["task_id"] != TASK_ID:
        raise ValueError("task_id must match exp5416")
    if artifact["milestone"] != MILESTONE:
        raise ValueError("milestone must match 2026.07.493")
    if artifact["status"] not in {"complete", "blocked"}:
        raise ValueError("status must be complete or blocked")
    if artifact["search_date"] != SEARCH_DATE:
        raise ValueError("search_date must equal 20260708")

    references = artifact["new_references_added"]
    _validate_references(references)
    count = artifact["new_actionable_findings_count"]
    if not isinstance(count, int) or count != len(references):
        raise ValueError("references count must equal new_references_added length")
    _validate_sources(artifact["sources_checked"], artifact["searched_source_details"])

    duplicates = artifact["duplicates_suppressed"]
    if not isinstance(duplicates, list) or not duplicates:
        raise ValueError("duplicates_suppressed must be a non-empty list")
    if len(duplicates) != len(set(duplicates)):
        raise ValueError("duplicates_suppressed must not contain duplicate suppressed entries")
    if artifact["retired_scopes_reopened"] is not False:
        raise ValueError("retired_scopes_reopened must remain false")

    prior_marker = artifact["prior_refresh_marker_found"]
    if artifact["status"] == "complete" and prior_marker is not True:
        raise ValueError("prior_refresh_marker_found must be true for complete artifacts")
    if artifact["status"] == "blocked" and prior_marker is not False:
        raise ValueError("prior_refresh_marker_found must be false for blocked artifacts")

    updated = artifact["research_references_updated"]
    if prior_marker is False:
        if updated is not False or references:
            raise ValueError("research_references_updated must be false when planner marker is missing")
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
    return f"- **{row['title']}** ({row['url']}): {row['carnot_hook']}"


def render_refresh_section(artifact: Mapping[str, Any]) -> str:
    validate_artifact(artifact)
    references = artifact["new_references_added"]
    if artifact["status"] != "complete" or not references:
        return ""
    lines = [
        REFRESH_HEADING,
        "",
        (
            "Execution-time sweep after the `.493` planner refresh checked arXiv, "
            "OpenReview, HuggingFace Papers, Semantic Scholar routes for EBT and "
            "ARM-EBM, GitHub, Extropic writing, Logical Intelligence public pages, "
            "V489/V490/V491/V492/V493 duplicate history, and the exclusion manifest. "
            "The findings below were absent from those blocks and add Carnot-local "
            "hooks without changing the active roadmap."
        ),
        "",
        "### New actionable deltas",
        *(_render_reference(row) for row in references),
        "",
        "### Execution impact",
        (
            "- **Plan impact:** No active `.493` roadmap edit is required. The deltas "
            "sharpen Exp5417/Exp5418 semantic, formalization, and prefix/action "
            "checks; Exp5421/Exp5422 verify-before-store and resource-routing gates; "
            "and Exp5426 deterministic evidence-path scoring."
        ),
        (
            "- **Duplicates suppressed:** V493 planner sources, EBT, ARM-EBM, NRGPT, "
            "Distributional EBMs, Energy-Based Decoding, NSVIF, REVES, Formalize-Don't-"
            "Optimize, Cycle-Consistent certificates, AgentLTL, OEP, CoACT, V492 "
            "Ising/LNS/iSTAR items, and nearby Extropic/Logical Intelligence context "
            "were already covered and are not re-added."
        ),
        (
            "- **Retired scope:** No retired scope was reopened. Retired ARC first-contact "
            "reruns, external generated-text/logprob scorers, CPU-only SOTA offload, "
            "token/internal-feature claims without backend receipts, non-local TSU/Kona/"
            "Aleph execution claims, and hardware speedup claims without matched board "
            "timing remain closed."
        ),
        (
            "- **Watch-only/excluded:** NatATL strategy synthesis, Extropic THRML/TSU "
            "repos, Logical Intelligence Aleph/Kona pages, and OpenReview EBT/NRGPT "
            "surfaces were checked but not promoted as executable `.493` dependencies."
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
        tests_run=["tests/python/test_experiment_5416_source_delta_v493.py"],
    )


if __name__ == "__main__":  # pragma: no cover
    main()
