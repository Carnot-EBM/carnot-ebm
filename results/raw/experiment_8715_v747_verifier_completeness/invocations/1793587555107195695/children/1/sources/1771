"""Exp5455: execution-time source delta refresh for V496.

Spec refs: REQ-REPORT-5455, SCENARIO-REPORT-5455-APPEND-DELTAS,
SCENARIO-REPORT-5455-NO-NEW-DELTA,
SCENARIO-REPORT-5455-BLOCKED-MISSING-PLANNER.

This module turns a literature sweep into an auditable receipt. It promotes a
source only when it adds a concrete Carnot-local experiment hook that is absent
from the V496 planner block and nearby V490-V496 source-delta history. Sources
that merely reinforce old ideas stay in duplicates, watch-only, or excluded
lists, so the execution refresh does not reopen retired lanes or create
reference churn.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_5455_source_delta_v496"
TASK_ID = "exp5455-source-delta-v496"
MILESTONE = "2026.07.496"
SEARCH_DATE = "20260709"
RESULT_RELATIVE_PATH = Path("results/experiment_5455_source_delta_v496.json")
REFERENCES_RELATIVE_PATH = Path("research-references.md")
PLANNER_MARKER = "### V496 Planner Refresh - 20260709"
REFRESH_HEADING = "### V496 Execution Refresh - 20260709"
REFRESH_END_MARKER = "<!-- V496-EXECUTION-REFRESH-20260709-END -->"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

SPEC_REFS = [
    "REQ-REPORT-5455",
    "SCENARIO-REPORT-5455-APPEND-DELTAS",
    "SCENARIO-REPORT-5455-NO-NEW-DELTA",
    "SCENARIO-REPORT-5455-BLOCKED-MISSING-PLANNER",
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
    "local_v490_v496_duplicate_history",
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
        "title": "Projectional Decoding: Towards Semantic-Aware LLM Generation",
        "url": "https://arxiv.org/abs/2605.30054",
        "source_type": "arXiv preprint / FSE 2026 IVR track paper",
        "carnot_hook": (
            "For Exp5459 and guided-decoding distortion guards, keep a partial "
            "semantic artifact graph beside token text, attach uncertainty and "
            "error nodes, and run incremental semantic validators before "
            "accepting locally valid but semantically wrong continuations. Exact "
            "final verifiers and solvers remain authority."
        ),
    },
    {
        "title": "Guiding Human Validation of LLM-Generated Code via Verifiable Literate Programming",
        "url": "https://arxiv.org/abs/2607.02333",
        "source_type": "arXiv preprint",
        "carnot_hook": (
            "For Exp5458 and AST/KB witness rows, add a doc-to-code trace-link "
            "fixture where unambiguous semantic documentation yields API-usage "
            "checks, formal-property checks, and suspicious documentation line "
            "IDs. Treat human validation as optional annotation, not a headline "
            "automation claim."
        ),
    },
    {
        "title": "Safe and Adaptive Cloud Healing: Verifying LLM-Generated Recovery Plans with a Neural-Symbolic World Model",
        "url": "https://arxiv.org/abs/2607.01595",
        "source_type": "arXiv preprint",
        "carnot_hook": (
            "For Exp5460 action/memory policy and tool-state validity, model "
            "recovery or tool plans as semantic primitive sequences, simulate "
            "feasibility in a deterministic world model before execution, and "
            "log failed primitive or precondition IDs. Do not import the DRL "
            "meta-prompt optimizer into V496."
        ),
    },
    {
        "title": "Beyond Perplexity: A Behavioral Evaluation Framework for Deployment-Memory Claims in LLM Test-Time Training",
        "url": "https://arxiv.org/abs/2607.00368",
        "source_type": "arXiv preprint",
        "carnot_hook": (
            "For Exp5460 and Exp5461 continuous self-learning, require a "
            "behavioral memory evidence ladder: recall after support removal, "
            "paraphrase robustness, locality, conflict handling, downstream "
            "action use, and matched explicit-memory/no-memory baselines before "
            "crediting online learning. Do not reopen LoRA or broad TTT fine-tuning."
        ),
    },
]

DUPLICATES_SUPPRESSED = [
    "V496 planner sources: Chance-Constrained Inference, CoCoA, DAVinCI, OLIVIA, CL-Bench, LCAD, strict-constraint distortion, tractable locally constrained decoding, KAN PWA/MILP, STATIC trie decoding, million-p-bit sampling, governed evolving memory, and minimal-core repair were already added - https://arxiv.org/abs/2602.01637",
    "AgentLTL trace verification was promoted in V491/V492 history and remains the deterministic procedural-compliance reference - https://arxiv.org/abs/2607.02599",
    "LLM-as-a-Verifier was already suppressed because its public path depends on external scoring-token/logprob extraction and GRPO/RL feedback, not Carnot-local exact verifiers - https://arxiv.org/abs/2607.05391",
    "Harnessing Code Agents for Automatic Software Verification remains watch-only until deterministic proof artifacts exist; it was already classified in the V492 source-delta artifact - https://arxiv.org/abs/2607.06341",
    "Formal Disco is not promoted because open-ended synthetic verified-program generation and iterative fine-tuning are outside V496 execution scope - https://arxiv.org/abs/2607.04631",
    "HCRC predicate-gated execution reinforces verifier-before-state-transition design but lacks a stronger local hook than existing verifier-potential, AgentLTL, and row-independent metric work - https://arxiv.org/abs/2607.04562",
    "Benchmarking Continual Agent Memory for Online Learning, Transfer, and Forgetting already appears in local memory sweeps and is not re-added - https://openreview.net/forum?id=MSXbrNExax",
    "DCCD, structural-equivalence grammar-constrained decoding, STATIC, llguidance, and constrained-decoding lists already cover syntax/prefix control boundaries - https://arxiv.org/abs/2603.03305",
    "PIM inertia, million-p-bit, p-dit, p-bit guided CDCL, p-dit QAP, and Potts mean-field sources already cover hardware sampling context; no hardware speedup claim is reopened - https://arxiv.org/abs/2606.25313",
    "EBT, ARM-EBM, NRGPT, EBT-Policy, and HuggingFace/OpenReview EBT surfaces remain architecture context already covered by V490-V496 history - https://arxiv.org/abs/2507.02092",
    "Automating Quality Assessment with NLP of LLM-Generated Defeaters is watch-only because it relies on learned meta-classifiers over subjective expert labels rather than exact verifier authority - https://arxiv.org/abs/2607.06039",
    "Extropic TSU/XTR-0/THRML writing remains non-local hardware context without authenticated Carnot TSU execution - https://extropic.ai/writing",
    "Logical Intelligence Aleph/Kona public pages remain non-local architecture context without reproducible local Aleph or Kona baselines - https://logicalintelligence.com/",
]

WATCH_ONLY_OR_EXCLUDED: list[JsonDict] = [
    {
        "title": "LLM-as-a-Verifier logit-expectation scorer and RL feedback",
        "url": "https://arxiv.org/abs/2607.05391",
        "classification": "excluded",
        "reason": (
            "The paper is relevant verification context, but its public scoring "
            "path depends on scoring-token logit expectations, repeated LLM "
            "evaluation, and SAC/GRPO feedback. External generated-text/logprob "
            "scorers and broad GRPO/RL lanes stay closed for V496."
        ),
    },
    {
        "title": "PASE DRL meta-prompt optimizer",
        "url": "https://arxiv.org/abs/2607.01595",
        "classification": "excluded",
        "reason": (
            "The semantic primitive and deterministic world-model verification "
            "pattern is promoted, but DRL prompt optimization is excluded from "
            "V496. Exact simulation, tool-state checks, and final verifiers keep "
            "exact final authority."
        ),
    },
    {
        "title": "Beyond Perplexity LoRA and broad test-time training path",
        "url": "https://arxiv.org/abs/2607.00368",
        "classification": "excluded",
        "reason": (
            "The behavioral evidence ladder is promoted for CSL evaluation, but "
            "LoRA and broad TTT/fine-tuning remain outside this source-delta task."
        ),
    },
    {
        "title": "Extropic TSU, XTR-0, and THRML writing",
        "url": "https://extropic.ai/writing",
        "classification": "watch-only",
        "reason": (
            "Extropic remains EBM sampler architecture context, but Carnot has no "
            "local TSU SDK, board receipt path, or authenticated TSU execution. "
            "Keep it as non-local TSU context only."
        ),
    },
    {
        "title": "Logical Intelligence Aleph and Kona public pages",
        "url": "https://logicalintelligence.com/",
        "classification": "watch-only",
        "reason": (
            "Aleph and Kona reinforce formal-verifier and EBM reasoning direction, "
            "but no reproducible local Kona or Aleph baseline exists for Carnot "
            "comparison."
        ),
    },
    {
        "title": "OpenReview EBT, ARM-EBM, and continual-memory surfaces",
        "url": "https://openreview.net/",
        "classification": "watch-only",
        "reason": (
            "OpenReview searches surfaced EBT, Benchmarking Continual Agent "
            "Memory, LCAD, and constrained-decoding material, but browser "
            "challenges blocked full forum reads and no surfaced item replaces "
            "Carnot's local exact solver or verifier authority."
        ),
    },
    {
        "title": "external generated-text/logprob scorers, token/internal features, duplicate ARC lanes, and unsupported hardware speedups",
        "url": "ops/exclusion_manifest.yaml",
        "classification": "excluded",
        "reason": (
            "external generated-text scorers, token/internal-feature claims without "
            "backend receipts, duplicate ARC lanes, broad GRPO/fine-tuning or "
            "LoRA reruns, non-local TSU/Kona/Aleph execution claims, and hardware "
            "speedup claims without matched board timing remain closed."
        ),
    },
]

SEARCHED_SOURCE_DETAILS: JsonDict = {
    "arxiv": {
        "status": "ok",
        "queries": [
            "2025-2026 energy-based models verification reasoning LLM constraints",
            "2025-2026 neural constraint satisfaction LLM verifier constraint systems",
            "2025-2026 Ising applications in ML and p-bit or p-dit sampling",
            "2025-2026 hallucination mitigation and energy-guided decoding",
            "2025-2026 Kolmogorov-Arnold Networks verification constraints",
            "2025-2026 hardware-accelerated sampling p-bit Ising",
            "2025-2026 continual online learning for constraint systems",
            "site:arxiv.org/abs/2607 constraint LLM verification model checking",
        ],
        "promoted": [
            "2605.30054 Projectional Decoding",
            "2607.02333 Verifiable Literate Programming",
            "2607.01595 PASE neural-symbolic world-model verification",
            "2607.00368 behavioral memory-claim evaluation",
        ],
        "not_promoted": [
            "2607.05391 LLM-as-a-Verifier uses scoring-token logits and RL feedback.",
            "2607.02599 AgentLTL is duplicate-covered in V491/V492.",
            "2607.06341 code-agent verification was already watch-only in Exp5403.",
            "2607.04631 Formal Disco depends on synthetic-data generation and fine-tuning scope.",
            "2607.04562 HCRC duplicates predicate-gated verification architecture context.",
            "2607.06039 defeater assessment uses learned meta-classifiers and subjective labels.",
            "2606.25313 million-p-bit and 2506.00269 p-dit hardware references are duplicate-covered.",
            "2602.06737 KAN PWA/MILP verification was already indexed repeatedly.",
        ],
    },
    "openreview": {
        "status": "challenge_blocked",
        "queries": [
            "OpenReview Energy-Based Transformers 2507.02092",
            "OpenReview ARM-EBM 2512.15605",
            "OpenReview neural constraint satisfaction certified correctness",
            "OpenReview Benchmarking Continual Agent Memory MSXbrNExax",
            "OpenReview LCAD rbl8fHjLuF",
        ],
        "result": (
            "OpenReview forum pages redirected to browser verification challenges. "
            "Search metadata surfaced EBT, Benchmarking Continual Agent Memory, "
            "LCAD, constrained decoding, and neural-CSP material, all already "
            "duplicate-covered or watch-only for V496."
        ),
    },
    "huggingface_papers": {
        "status": "ok",
        "queries": [
            "HuggingFace Papers EBT 2507.02092",
            "HuggingFace Papers ARM-EBM 2512.15605",
            "HuggingFace Papers Projectional Decoding 2605.30054",
            "HuggingFace Papers Verifiable Literate Programming 2607.02333",
            "HuggingFace Papers LLM-as-a-Verifier 2607.05391",
        ],
        "result": (
            "HuggingFace Papers confirmed EBT metadata and community discussion as "
            "duplicate context. Searches for the newly promoted V496 execution "
            "items primarily routed to arXiv or generic paper search surfaces; no "
            "HuggingFace page displaced the primary arXiv hooks or local runtime "
            "requirements."
        ),
    },
    "semantic_scholar": {
        "status": "partial",
        "queries": [
            "Semantic Scholar public route for EBT 2507.02092",
            "Semantic Scholar public route for ARM-EBM 2512.15605",
            "Semantic Scholar links on arXiv pages for 2605.30054, 2607.02333, 2607.01595, and 2607.00368",
        ],
        "result": (
            "Public routes remained citation/metadata context and did not surface "
            "a stronger Carnot-local V496 dependency than the source papers. EBT "
            "and ARM-EBM stay duplicate-covered architecture context."
        ),
    },
    "github": {
        "status": "partial",
        "queries": [
            "GitHub Projectional Decoding 2605.30054",
            "GitHub Verifiable Literate Programming 2607.02333",
            "GitHub PASE neural-symbolic world model 2607.01595",
            "GitHub Beyond Perplexity deployment memory claims 2607.00368",
            "GitHub EBT 2507.02092 and ARM-EBM 2512.15605",
            "GitHub Extropic THRML and constrained-decoding watch lists",
        ],
        "watch_only_links": [
            "https://github.com/alexiglad/EBT",
            "https://github.com/extropic-ai/thrml",
            "https://github.com/FrontisAI/Awesome-Self-Improving-Agents",
        ],
        "result": (
            "GitHub search did not reveal official implementations for the four "
            "promoted source deltas. Existing EBT, THRML, constrained-decoding, "
            "and self-improving-agent lists remain watch-only implementation "
            "context and do not replace Carnot exact verifiers or receipts."
        ),
    },
    "extropic_writing": {
        "status": "ok",
        "queries": [
            "Extropic writing page",
            "Extropic TSU 101",
            "Extropic inside X0 and XTR-0",
            "Extropic THRML repository",
        ],
        "result": (
            "Extropic still provides TSU and thermodynamic-computing context, but "
            "no local Carnot TSU SDK, execution receipt, or speedup basis was found."
        ),
    },
    "logical_intelligence": {
        "status": "ok",
        "queries": [
            "Logical Intelligence homepage",
            "Logical Intelligence automatic formal verification for code generation",
            "Logical Intelligence Aleph leading benchmarks",
            "Logical Intelligence Kona energy-based model",
        ],
        "result": (
            "Logical Intelligence pages reinforce formal verification plus EBM "
            "reasoning, but remain non-local architecture context with no "
            "reproducible Aleph/Kona baseline for Carnot."
        ),
    },
    "local_v490_v496_duplicate_history": {
        "status": "ok",
        "queries": [
            "research-references.md V490 Planner and Execution Refresh",
            "research-references.md V491 Planner and Execution Refresh",
            "research-references.md V492 Planner and Execution Refresh",
            "research-references.md V493 Planner and Execution Refresh",
            "research-references.md V494 Planner and Execution Refresh",
            "research-references.md V495 Planner and Execution Refresh",
            "research-references.md V496 Planner Refresh",
            "repo-wide search for promoted arXiv ids and titles",
        ],
        "result": (
            "Exact searches found no local entries for Projectional Decoding "
            "2605.30054, Verifiable Literate Programming 2607.02333, PASE "
            "2607.01595, or Beyond Perplexity 2607.00368. AgentLTL, LLM-as-a-"
            "Verifier, Benchmarking Continual Agent Memory, hardware p-bit/p-dit "
            "sources, EBT, ARM-EBM, DCCD, STATIC, and KAN verification material "
            "were suppressed as duplicate or watch-only."
        ),
    },
    "ops_exclusion_manifest": {
        "status": "ok",
        "queries": [
            "external generated-text/logprob scorer lanes",
            "broad fine-tuning, GRPO, RL, TTT, and LoRA reruns",
            "CPU-only SOTA headline/offload reruns",
            "token/internal feature claims without backend receipts",
            "non-local TSU/Kona/Aleph execution claims",
            "duplicate ARC solve or first-contact exploration lanes",
            "hardware speedup claims without matched board timing",
        ],
        "result": (
            "Retired and non-local lanes stayed closed. New deltas are constrained "
            "to semantic graph fixtures, doc/code trace-link witnesses, deterministic "
            "world-model feasibility checks, and behavioral CSL evidence accounting."
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
    """Build the Exp5455 source-delta artifact.

    The receipt is intentionally about source aggregation. It records what was
    checked and how dedupe decisions were made, but it does not claim model
    quality, verifier accuracy, citation influence, hardware speedup, or a
    reopened retired scope.
    """

    references = [dict(row) for row in new_references_added] if prior_refresh_marker_found else []
    count = len(references)
    updated = (count > 0) if research_references_updated is None else research_references_updated
    if not prior_refresh_marker_found:
        status = "blocked"
        updated = False
        verdict = "blocked: V496 planner refresh marker missing; references unchanged."
    else:
        status = "complete"
        verdict_detail = (
            f"{count} new actionable V496 execution-time source deltas appended; retired scopes remained closed"
            if count
            else "no new actionable V496 execution-time source deltas; references unchanged"
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
        or ["tests/python/test_experiment_5455_source_delta_v496.py"],
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
        raise ValueError("field_principles must match REQ-REPORT-5455")
    if artifact["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("experiment_id must match Exp5455")
    if artifact["task_id"] != TASK_ID:
        raise ValueError("task_id must match exp5455")
    if artifact["milestone"] != MILESTONE:
        raise ValueError("milestone must match 2026.07.496")
    if artifact["status"] not in {"complete", "blocked"}:
        raise ValueError("status must be complete or blocked")
    if artifact["search_date"] != SEARCH_DATE:
        raise ValueError("search_date must equal 20260709")

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
            "Execution-time sweep after the `.496` planner refresh checked arXiv, "
            "OpenReview, HuggingFace Papers, Semantic Scholar routes for EBT and "
            "ARM-EBM, GitHub, Extropic writing, Logical Intelligence public pages, "
            "V490/V491/V492/V493/V494/V495/V496 duplicate history, and the exclusion "
            "manifest. The findings below were absent from those blocks and add "
            "Carnot-local hooks without changing the active roadmap."
        ),
        "",
        "### New actionable deltas",
        *(_render_reference(row) for row in references),
        "",
        "### Execution impact",
        (
            "- **Plan impact:** No active `.496` roadmap edit is required. The deltas "
            "sharpen Exp5458/Exp5459 semantic graph, doc/code witness, and "
            "distortion-guard receipts; Exp5460/Exp5461 tool-state and CSL "
            "behavioral-memory evidence; and later guided-decoding acceptance "
            "checks without expanding scope."
        ),
        (
            "- **Duplicates suppressed:** V496 planner sources, AgentLTL, "
            "LLM-as-a-Verifier, Harnessing Code Agents, Formal Disco, HCRC, "
            "Benchmarking Continual Agent Memory, DCCD, STATIC, llguidance, "
            "million-p-bit, p-dit, p-bit guided CDCL, KAN PWA/MILP, EBT, ARM-EBM, "
            "NRGPT, LCAD, and prior Extropic/Logical Intelligence context were "
            "already covered or stayed watch-only and are not re-added."
        ),
        (
            "- **Retired scope:** No retired scope was reopened. LLM-as-a-Verifier "
            "logit scorers, PASE DRL optimization, LoRA/TTT fine-tuning, broad "
            "GRPO/RL reruns, external generated-text/logprob scorers, token/internal "
            "feature claims without backend receipts, non-local TSU/Kona/Aleph "
            "execution claims, duplicate ARC lanes, and hardware speedup claims "
            "without matched board timing remain closed."
        ),
        (
            "- **Watch-only/excluded:** Extropic TSU/XTR-0/THRML writing, Logical "
            "Intelligence Aleph/Kona pages, OpenReview EBT/LCAD/continual-memory "
            "surfaces, Formal Disco fine-tuning, HCRC, defeater-quality "
            "meta-classifiers, and Semantic Scholar citation-route material were "
            "checked but not promoted as executable `.496` dependencies."
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
        tests_run=["tests/python/test_experiment_5455_source_delta_v496.py"],
    )


if __name__ == "__main__":  # pragma: no cover
    main()
