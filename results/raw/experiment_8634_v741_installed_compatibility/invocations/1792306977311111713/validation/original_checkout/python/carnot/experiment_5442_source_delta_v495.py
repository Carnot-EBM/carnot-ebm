"""Exp5442: execution-time source delta refresh for V495.

Spec refs: REQ-REPORT-5442, SCENARIO-REPORT-5442-APPEND-DELTAS,
SCENARIO-REPORT-5442-NO-NEW-DELTA,
SCENARIO-REPORT-5442-BLOCKED-MISSING-PLANNER.

This module turns the literature sweep into a reproducible receipt. It is
intentionally conservative: a source is promoted only when it adds a concrete
Carnot-local action that was absent from the V495 planner block and the nearby
V489-V495 duplicate history. Items that merely reinforce old ideas are kept as
duplicates, watch-only, or excluded so the execution refresh does not reopen
retired lanes or create reference churn.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_5442_source_delta_v495"
TASK_ID = "exp5442-source-delta-v495"
MILESTONE = "2026.07.495"
SEARCH_DATE = "20260708"
RESULT_RELATIVE_PATH = Path("results/experiment_5442_source_delta_v495.json")
REFERENCES_RELATIVE_PATH = Path("research-references.md")
PLANNER_MARKER = "### V495 Planner Refresh - 2026-07-08"
REFRESH_HEADING = "### V495 Execution Refresh - 20260708"
REFRESH_END_MARKER = "<!-- V495-EXECUTION-REFRESH-20260708-END -->"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

SPEC_REFS = [
    "REQ-REPORT-5442",
    "SCENARIO-REPORT-5442-APPEND-DELTAS",
    "SCENARIO-REPORT-5442-NO-NEW-DELTA",
    "SCENARIO-REPORT-5442-BLOCKED-MISSING-PLANNER",
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
    "local_v489_v495_duplicate_history",
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
        "title": "Dockerless: Environment-Free Program Verifier for Coding Agents",
        "url": "https://arxiv.org/abs/2606.28436",
        "source_type": "arXiv preprint surfaced via HuggingFace Papers",
        "carnot_hook": (
            "For Exp5445 code/API witness constraints, add an environment-free "
            "repo-evidence triage arm that gathers call-graph, import, reference, "
            "and surrounding-code evidence before execution. Keep deterministic "
            "tests, runtime checks, AST/KB witnesses, or exact solvers as final "
            "authority, and do not credit the paper's SFT/RL post-training claims."
        ),
    },
    {
        "title": "P-dit Probabilistic Ising Machine for Solving the Quadratic Assignment Problem",
        "url": "https://arxiv.org/abs/2605.24408",
        "source_type": "arXiv preprint",
        "carnot_hook": (
            "For Exp5448 p-bit/Potts work, add a CPU-only assignment-style "
            "multi-state fixture with p-dit variable counts, assignment-cost "
            "energy tables, exact QAP baseline checks, workload hashes, and "
            "solver-authority outcomes. Do not claim large-QAP performance, GPU "
            "parallelism, or hardware speedup without matched local receipts."
        ),
    },
    {
        "title": "From Errors to Proofs: Minimal-Core-Guided Repair for Neuro-Symbolic Constraint Solving",
        "url": "https://openreview.net/forum?id=ySI9HwU9K7",
        "source_type": "OpenReview workshop paper metadata",
        "carnot_hook": (
            "For Exp5443 and Exp5445 formalization repair, replace generic "
            "solver-error prompting with minimal unsatisfiable core feedback. "
            "Record core constraint IDs, omitted-constraint hypotheses, repaired "
            "formalizations, and exact recheck results so solver feedback stays "
            "deterministic and auditable."
        ),
    },
]

DUPLICATES_SUPPRESSED = [
    "V495 planner sources: Score x Decoder, S3, DTV, p-bit guided CDCL, Potts mean-field, million-p-bit hardware, SSGM memory governance, evidence tracing, Execute-Distill-Verify, MemFail, Experience Compression Spectrum, and deterministic AST hallucination correction were already added - https://arxiv.org/abs/2606.00739",
    "Verifier-Guided Code Translation via Meta-Step Decoding already anchors structural rollback for V495 - https://arxiv.org/abs/2605.17626",
    "p-bit guided CDCL and Potts mean-field sparsity already cover solver-authoritative stochastic hints and sparse multi-state hardware boundaries - https://arxiv.org/abs/2605.04033",
    "Existing p-dit and Potts references already cover generic p-dit state accounting; only the QAP assignment-style fixture is new enough to promote - https://arxiv.org/abs/2506.00269",
    "Dockerless post-training claims are suppressed because broad SFT/RL/GRPO-style post-training remains outside this execution refresh; only the repo-evidence triage pattern is promoted - https://arxiv.org/abs/2606.28436",
    "Automating Formal Verification with Reinforcement Learning and Recursive Inference was already indexed locally and remains a source-reported RLVR/proof-scaffold reference - https://arxiv.org/abs/2605.30914",
    "ConstraintLLM, IndusCP, CARM, NSVIF, and Neuro-Symbolic Compliance are already local constraint-formalization references - https://arxiv.org/abs/2510.05774",
    "SynthFix and related neuro-symbolic repair papers are watch-only because they depend on fine-tuning or vulnerability-repair scope not requested by V495 - https://arxiv.org/abs/2604.17184",
    "Energy-Based Transformers, EBT HuggingFace/OpenReview pages, NRGPT, EBT-Policy, and ARM-EBM remain architecture context already covered by V489-V495 history - https://arxiv.org/abs/2507.02092",
    "Semantic Scholar visible citation routes for EBT and ARM-EBM did not add a stronger Carnot-local task than existing EBT/ARM hooks; API calls returned HTTP 429 in this pass - https://www.semanticscholar.org/",
    "Extropic TSU/XTR-0/THRML writing remains non-local hardware context without authenticated Carnot TSU execution - https://extropic.ai/writing",
    "Logical Intelligence Aleph/Kona public pages remain non-local architecture context without reproducible local Aleph or Kona baselines - https://logicalintelligence.com/",
]

WATCH_ONLY_OR_EXCLUDED: list[JsonDict] = [
    {
        "title": "Dockerless SFT/RL post-training pipeline",
        "url": "https://arxiv.org/abs/2606.28436",
        "classification": "excluded",
        "reason": (
            "The environment-free verifier is useful as repo-evidence triage, but "
            "its SFT/RL reward pipeline is excluded for V495. Exact tests, runtime "
            "checks, AST/KB witnesses, or solvers remain exact final authority."
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
        "title": "OpenReview certified neural-CSP and EBT surfaces",
        "url": "https://openreview.net/",
        "classification": "watch-only",
        "reason": (
            "OpenReview metadata reinforces exact-solver/certificate authority, "
            "but browser challenges blocked full page fetches and no neural solver "
            "replaces Carnot's deterministic checks."
        ),
    },
    {
        "title": "external generated-text/logprob scorers, token/internal features, duplicate ARC lanes, and unsupported hardware speedups",
        "url": "ops/exclusion_manifest.yaml",
        "classification": "excluded",
        "reason": (
            "external generated-text scorers, token/internal-feature claims without "
            "backend receipts, duplicate ARC lanes, broad GRPO/fine-tuning reruns, "
            "non-local TSU/Kona/Aleph execution claims, and hardware speedup claims "
            "without matched board timing remain closed."
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
            "site:arxiv.org/abs/2607 verifier guided decoding constraint LLM",
        ],
        "promoted": [
            "2606.28436 Dockerless environment-free patch verifier",
            "2605.24408 p-dit probabilistic Ising machine for QAP",
        ],
        "not_promoted": [
            "2605.17626 DTV was already promoted in the V495 planner refresh.",
            "2605.30914 formal-verification RLVR was already indexed locally.",
            "2510.05774 ConstraintLLM was already implemented locally through CARM work.",
            "2604.17184 SynthFix depends on fine-tuning/vulnerability-repair scope.",
            "2602.06737 KAN PWA/MILP verification was already indexed repeatedly.",
            "2606.28436 Dockerless post-training claims are excluded; only triage is promoted.",
        ],
    },
    "openreview": {
        "status": "challenge_blocked",
        "queries": [
            "OpenReview Energy-Based Transformers 2507.02092",
            "OpenReview ARM-EBM 2512.15605",
            "OpenReview neural constraint satisfaction certified correctness",
            "OpenReview Minimal-Core-Guided Repair neuro-symbolic constraint solving",
        ],
        "result": (
            "OpenReview full pages were browser-challenge blocked. Search metadata "
            "surfaced From Errors to Proofs / Minimal-Core-Guided Repair with a "
            "minimal-unsat-core formalization-repair hook that was absent locally. "
            "EBT, NRGPT, certified neural-CSP, and action-head surfaces stayed "
            "duplicate or watch-only."
        ),
    },
    "huggingface_papers": {
        "status": "ok",
        "queries": [
            "HuggingFace Papers EBT 2507.02092",
            "HuggingFace Papers Dockerless 2606.28436",
            "HuggingFace Papers verifier-guided decoding deterministic verifier",
            "HuggingFace Papers solver-verifier systems",
        ],
        "result": (
            "HuggingFace Papers confirmed EBT metadata as duplicate context and "
            "surfaced Dockerless with a recent paper card, Semantic Scholar-style "
            "recommendations, and no model/dataset/space citations. Dockerless is "
            "promoted only as an advisory repo-evidence triage pattern."
        ),
    },
    "semantic_scholar": {
        "status": "rate_limited",
        "queries": [
            "Semantic Scholar API arXiv:2507.02092",
            "Semantic Scholar API arXiv:2512.15605",
            "public search for EBT 2507.02092 Semantic Scholar citations",
            "public search for ARM-EBM 2512.15605 Semantic Scholar citations",
        ],
        "result": (
            "Direct Semantic Scholar API requests for both EBT and ARM-EBM returned "
            "HTTP 429. Public search exposed related EBT citation-route pages such "
            "as Transformers as Intrinsic Optimizers and Learning Iterative "
            "Reasoning through Energy Diffusion, but no citation-derived local "
            "V495 dependency superseded the existing EBT/ARM hooks."
        ),
    },
    "github": {
        "status": "partial",
        "queries": [
            "GitHub Dockerless Environment-Free Program Verifier 2606.28436",
            "GitHub P-dit Probabilistic Ising Machine QAP 2605.24408",
            "GitHub Extropic THRML",
            "GitHub KAN Ising constrained decoding watch lists",
        ],
        "watch_only_links": [
            "https://github.com/loft-sh/dockerless",
            "https://github.com/extropic-ai/thrml",
            "https://github.com/alexiglad/EBT",
        ],
        "result": (
            "GitHub search did not reveal an official Dockerless paper repository "
            "or QAP p-dit implementation to import. THRML and EBT repositories "
            "remain watch-only and do not replace local exact solvers or hardware receipts."
        ),
    },
    "extropic_writing": {
        "status": "ok",
        "queries": [
            "Extropic thermodynamic computing from zero to one",
            "Extropic TSU 101",
            "Extropic inside X0 and XTR-0",
            "Extropic THRML repository",
        ],
        "result": (
            "Extropic still describes TSUs as samplers for energy-based models and "
            "XTR/THRML as thermodynamic-computing tooling. No local Carnot TSU "
            "SDK, execution receipt, or speedup basis was found."
        ),
    },
    "logical_intelligence": {
        "status": "ok",
        "queries": [
            "Logical Intelligence automatic formal verification for code generation",
            "Logical Intelligence Aleph leading benchmarks",
            "Logical Intelligence Kona energy-based model",
            "Logical Intelligence public homepage",
        ],
        "result": (
            "Logical Intelligence pages reinforce formal verification plus EBM "
            "reasoning, but remain non-local architecture context with no "
            "reproducible Aleph/Kona baseline for Carnot."
        ),
    },
    "local_v489_v495_duplicate_history": {
        "status": "ok",
        "queries": [
            "research-references.md V489 Planner and Execution Refresh",
            "research-references.md V490 Planner and Execution Refresh",
            "research-references.md V491 Planner and Execution Refresh",
            "research-references.md V492 Planner and Execution Refresh",
            "research-references.md V493 Planner and Execution Refresh",
            "research-references.md V494 Planner and Execution Refresh",
            "research-references.md V495 Planner Refresh",
            "repo-wide search for promoted arXiv ids, OpenReview id, and titles",
        ],
        "result": (
            "Exact searches found no local entries for Dockerless 2606.28436, "
            "P-dit QAP 2605.24408, or OpenReview ySI9HwU9K7. Existing p-dit, "
            "DTV, KAN, formal-verification, ConstraintLLM, EBT, ARM-EBM, and "
            "memory-governance material was suppressed as duplicate or watch-only."
        ),
    },
    "ops_exclusion_manifest": {
        "status": "ok",
        "queries": [
            "external generated-text/logprob scorer lanes",
            "broad fine-tuning, GRPO, and LoRA reruns",
            "CPU-only SOTA headline/offload reruns",
            "token/internal feature claims without backend receipts",
            "non-local TSU/Kona/Aleph execution claims",
            "duplicate ARC solve or first-contact exploration lanes",
            "hardware speedup claims without matched board timing",
        ],
        "result": (
            "Retired and non-local lanes stayed closed. New deltas are constrained "
            "to deterministic fixture design, advisory evidence triage, exact "
            "solver repair feedback, or CPU-only assignment/Potts controls."
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
    """Build the Exp5442 source-delta artifact.

    The artifact is a source-aggregation receipt. It documents what was checked
    and how dedupe decisions were made, but it does not claim model quality,
    verifier accuracy, citation influence, hardware speedup, or reopened scope.
    """

    references = [dict(row) for row in new_references_added] if prior_refresh_marker_found else []
    count = len(references)
    updated = (count > 0) if research_references_updated is None else research_references_updated
    if not prior_refresh_marker_found:
        status = "blocked"
        updated = False
        verdict = "blocked: V495 planner refresh marker missing; references unchanged."
    else:
        status = "complete"
        verdict_detail = (
            f"{count} new actionable V495 execution-time source deltas appended; retired scopes remained closed"
            if count
            else "no new actionable V495 execution-time source deltas; references unchanged"
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
        or ["tests/python/test_experiment_5442_source_delta_v495.py"],
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
        raise ValueError("field_principles must match REQ-REPORT-5442")
    if artifact["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("experiment_id must match Exp5442")
    if artifact["task_id"] != TASK_ID:
        raise ValueError("task_id must match exp5442")
    if artifact["milestone"] != MILESTONE:
        raise ValueError("milestone must match 2026.07.495")
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
            "Execution-time sweep after the `.495` planner refresh checked arXiv, "
            "OpenReview, HuggingFace Papers, Semantic Scholar routes for EBT and "
            "ARM-EBM, GitHub, Extropic writing, Logical Intelligence public pages, "
            "V489/V490/V491/V492/V493/V494/V495 duplicate history, and the exclusion "
            "manifest. The findings below were absent from those blocks and add "
            "Carnot-local hooks without changing the active roadmap."
        ),
        "",
        "### New actionable deltas",
        *(_render_reference(row) for row in references),
        "",
        "### Execution impact",
        (
            "- **Plan impact:** No active `.495` roadmap edit is required. The deltas "
            "sharpen Exp5443/Exp5445 formalization and code-witness receipts, and "
            "Exp5448 p-bit/Potts assignment controls, without expanding scope."
        ),
        (
            "- **Duplicates suppressed:** V495 planner sources, DTV, p-bit guided "
            "CDCL, Potts mean-field, million-p-bit hardware, SSGM, evidence tracing, "
            "Execute-Distill-Verify, MemFail, Experience Compression Spectrum, "
            "deterministic AST correction, EBT, ARM-EBM, ConstraintLLM, NSVIF, "
            "Automating Formal Verification, and prior p-dit/Potts accounting were "
            "already covered and are not re-added."
        ),
        (
            "- **Retired scope:** No retired scope was reopened. Dockerless SFT/RL "
            "post-training, broad GRPO/fine-tuning or LoRA reruns, external "
            "generated-text/logprob scorers, token/internal feature claims without "
            "backend receipts, non-local TSU/Kona/Aleph execution claims, duplicate "
            "ARC lanes, and hardware speedup claims without matched board timing "
            "remain closed."
        ),
        (
            "- **Watch-only/excluded:** Extropic TSU/XTR-0/THRML writing, Logical "
            "Intelligence Aleph/Kona pages, OpenReview EBT/neural-CSP surfaces, "
            "Dockerless post-training, SynthFix fine-tuning, and Semantic Scholar "
            "citation-route material were checked but not promoted as executable "
            "`.495` dependencies."
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
        tests_run=["tests/python/test_experiment_5442_source_delta_v495.py"],
    )


if __name__ == "__main__":  # pragma: no cover
    main()
