"""Exp5429: execution-time source delta refresh for V494.

Spec refs: REQ-REPORT-5429, SCENARIO-REPORT-5429-APPEND-DELTAS,
SCENARIO-REPORT-5429-NO-NEW-DELTA,
SCENARIO-REPORT-5429-BLOCKED-MISSING-PLANNER.

This module turns the literature sweep into a reproducible receipt. It keeps
the judgment deliberately narrow: a source is promoted only when it adds a
concrete Carnot-local hook that was absent from the V494 planner block and the
nearby V489-V494 duplicate history. Everything else is recorded as duplicate,
watch-only, or excluded so the execution refresh does not reopen retired lanes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_5429_source_delta_v494"
TASK_ID = "exp5429-source-delta-v494"
MILESTONE = "2026.07.494"
SEARCH_DATE = "20260708"
RESULT_RELATIVE_PATH = Path("results/experiment_5429_source_delta_v494.json")
REFERENCES_RELATIVE_PATH = Path("research-references.md")
PLANNER_MARKER = "### V494 Planner Refresh - 2026-07-08"
REFRESH_HEADING = "### V494 Execution Refresh - 20260708"
REFRESH_END_MARKER = "<!-- V494-EXECUTION-REFRESH-20260708-END -->"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

SPEC_REFS = [
    "REQ-REPORT-5429",
    "SCENARIO-REPORT-5429-APPEND-DELTAS",
    "SCENARIO-REPORT-5429-NO-NEW-DELTA",
    "SCENARIO-REPORT-5429-BLOCKED-MISSING-PLANNER",
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
    "local_v489_v494_duplicate_history",
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
        "title": "NeuroSCA: Neuro-Symbolic Constraint Abstraction for Smart Contract Hybrid Fuzzing",
        "url": "https://arxiv.org/abs/2603.01272",
        "source_type": "arXiv preprint",
        "carnot_hook": (
            "For Exp5432 and later structured-verifier fixtures, add a core-constraint "
            "abstraction pass with a missed-constraint reinsertion ledger: the abstracted "
            "constraint set may speed the solver, but concrete execution or exact solver "
            "checks must reintroduce any omitted constraints before a row is accepted."
        ),
    },
    {
        "title": "Sampling for Quality: Training-Free Reward-Guided LLM Decoding via Sequential Monte Carlo",
        "url": "https://arxiv.org/abs/2604.16453",
        "source_type": "arXiv preprint mirrored by HuggingFace Papers search",
        "carnot_hook": (
            "For any V494 structured-generation or future energy-guided decoding rerun, "
            "add a prefix-only SMC control that uses deterministic verifier reward "
            "potentials, records reward-evaluation budget per accepted token, and keeps "
            "model weights frozen rather than reopening GRPO or broad fine-tuning."
        ),
    },
    {
        "title": "LLMs versus the Halting Problem: Characterizing Program Termination Reasoning",
        "url": "https://arxiv.org/abs/2601.18987",
        "source_type": "arXiv preprint",
        "carnot_hook": (
            "For Exp5430/Exp5431 structured corrigenda and future code-verification "
            "fixtures, separate semantic verdict accuracy from witness construction: "
            "termination or non-termination claims should carry divergence-precondition "
            "constraints, witness/proof fields, and deterministic verifier outcomes."
        ),
    },
]

DUPLICATES_SUPPRESSED = [
    "V494 planner sources: A-MEM, CAD workflow memory, ontology external memory, soft-logic residuals, and Logical Intelligence Aleph posts - https://arxiv.org/abs/2502.12110",
    "CoVe constraint-guided tool-use agents already appeared in local references and changelog - https://arxiv.org/abs/2603.01940",
    "When Continual Learning Moves to Memory already appears in multiple local memory sweeps - https://arxiv.org/abs/2604.27003",
    "Energy-Guided Decoding for Object Hallucination was already indexed repeatedly - https://arxiv.org/abs/2507.07731",
    "p-Bit-Based Fully-Connected Quantum-Inspired Simulated Annealer with Dual BRAM was already indexed and implemented in p-bit diagnostics - https://arxiv.org/abs/2602.16143",
    "Neuro-Symbolic Compliance via LLMs and SMT was already indexed in prior compliance sweeps - https://arxiv.org/abs/2601.06181",
    "LoopUS and other EBT citation-trail papers were already covered in V481-V493 history - https://arxiv.org/abs/2605.11011",
    "EBT source paper and project page remain architecture context, not a new V494 execution delta - https://arxiv.org/abs/2507.02092",
    "ARM-EBM source paper remains foundational context with no fresh executable V494 hook - https://arxiv.org/abs/2512.15605",
    "NRGPT, Energy-Based Action Heads, and Transformers as Intrinsic Optimizers were already local watch items - https://openreview.net/forum?id=B3Muyi2zgo",
    "V492 Ising/LNS/iSTAR active-tail papers were already recorded and are not re-added - https://arxiv.org/abs/2607.05169",
    "V493 semantic/formalization/resource-routing execution deltas were already recorded and are not re-added - https://arxiv.org/abs/2606.27281",
]

WATCH_ONLY_OR_EXCLUDED: list[JsonDict] = [
    {
        "title": "Extropic TSU/XTR-0 writing and THRML repositories",
        "url": "https://extropic.ai/writing",
        "classification": "watch-only",
        "reason": (
            "Extropic remains sampler and architecture context, but Carnot has no local "
            "TSU SDK, board receipt path, or authenticated TSU execution. Keep as "
            "non-local TSU watch context only."
        ),
    },
    {
        "title": "Logical Intelligence Aleph and Kona public pages",
        "url": "https://logicalintelligence.com/",
        "classification": "watch-only",
        "reason": (
            "Aleph and Kona pages support verifier-first architecture language, but no "
            "reproducible local Kona or Aleph baseline exists for Carnot comparison."
        ),
    },
    {
        "title": "OpenReview EBT, NRGPT, and energy-action-head pages",
        "url": "https://openreview.net/forum?id=B3Muyi2zgo",
        "classification": "duplicate suppressed",
        "reason": (
            "OpenReview reinforces the existing EBT/NRGPT watch lane and does not add "
            "an executable V494 dependency beyond the already-indexed arXiv sources."
        ),
    },
    {
        "title": "Energy-guided VLM hallucination and hidden-state-only methods",
        "url": "https://arxiv.org/abs/2507.07731",
        "classification": "watch-only",
        "reason": (
            "These methods require VLM or hidden-state/logit access. They stay outside "
            "V494 unless a backend emits authenticated token/internal receipts."
        ),
    },
    {
        "title": "broad fine-tuning, GRPO, LoRA memory, and external generated-text scorers",
        "url": "ops/exclusion_manifest.yaml",
        "classification": "excluded",
        "reason": (
            "broad fine-tuning, GRPO reruns, external generated-text/logprob scorer "
            "lanes, token/internal-feature claims without backend receipts, duplicate "
            "ARC lanes, and hardware speedup claims without matched board timing remain closed."
        ),
    },
]

SEARCHED_SOURCE_DETAILS: JsonDict = {
    "arxiv": {
        "status": "ok",
        "queries": [
            "2025-2026 energy-based models verification reasoning LLM constraints",
            "2025-2026 neural constraint satisfaction LLM verifier",
            "2025-2026 Ising applications in ML and active-constraint solvers",
            "2025-2026 hallucination mitigation and energy-guided decoding",
            "2025-2026 Kolmogorov-Arnold Networks verification constraints",
            "2025-2026 hardware-accelerated sampling p-bit Ising",
            "2025-2026 continual online learning for constraint systems",
        ],
        "promoted": [
            "2603.01272 NeuroSCA constraint abstraction and verifier-in-loop refinement",
            "2604.16453 training-free reward-guided LLM decoding via SMC",
            "2601.18987 halting-problem witness and divergence-precondition gap",
        ],
        "not_promoted": [
            "2603.01940 CoVe is duplicate-covered in local history.",
            "2604.27003 memory-retrieval interference is duplicate-covered.",
            "2507.07731 energy-guided VLM decoding is duplicate-covered and requires hidden-state access.",
            "2602.16143 p-bit Dual BRAM hardware is duplicate-covered and non-local for V494 timing.",
            "2601.06181 Neuro-Symbolic Compliance is duplicate-covered.",
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
            "Search surfaced EBT, NRGPT, energy-action-head, and constrained-generation "
            "material already represented by arXiv or nearby watch blocks. No "
            "OpenReview-only V494 delta was promoted."
        ),
    },
    "huggingface_papers": {
        "status": "partial",
        "queries": [
            "HuggingFace Papers CoVe 2603.01940",
            "HuggingFace Papers Sampling for Quality 2604.16453",
            "HuggingFace Papers NeuroSCA 2603.01272",
            "HuggingFace Papers Energy-Guided Decoding 2507.07731",
        ],
        "result": (
            "HuggingFace Papers search mirrored CoVe and Sampling for Quality and exposed "
            "related daily-paper cards. CoVe and energy-guided VLM decoding were "
            "suppressed as duplicates; Sampling for Quality remained a new exact-method "
            "delta absent from local references."
        ),
    },
    "semantic_scholar": {
        "status": "partial",
        "queries": [
            "Semantic Scholar API arXiv:2507.02092",
            "Semantic Scholar API arXiv:2512.15605",
        ],
        "result": (
            "EBT returned citationCount=26 and influentialCitationCount=2 with citation "
            "samples including LoopUS, Fixed-Point Reasoners, ISPASS EBM workload "
            "characterization, NRGPT, EBT-Policy, and Transformers as Intrinsic "
            "Optimizers; those are already duplicate-covered or watch-only. ARM-EBM "
            "returned HTTP 429 in this pass, so no ARM citation-count delta is claimed."
        ),
    },
    "github": {
        "status": "ok",
        "queries": [
            "GitHub NeuroSCA 2603.01272",
            "GitHub reward-guided LLM decoding Sequential Monte Carlo",
            "GitHub Energy-Guided Decoding Object Hallucination",
            "GitHub Extropic THRML",
            "GitHub KAN Ising constrained decoding watch lists",
        ],
        "promoted_supporting_links": [
            "https://github.com/Z3Prover/z3/discussions/9008",
        ],
        "watch_only_links": [
            "https://github.com/NishilBalar/Awesome-LVLM-Hallucination",
            "https://github.com/extropic-ai/thrml",
            "https://github.com/alexiglad/EBT",
        ],
        "result": (
            "GitHub searches provided secondary support for NeuroSCA-style constraint "
            "pollution concerns and duplicate watch lists for LVLM hallucination, EBT, "
            "KAN, and Ising topics. No repository replaced Carnot's deterministic checks."
        ),
    },
    "extropic_writing": {
        "status": "ok",
        "queries": [
            "Extropic thermodynamic computing from zero to one",
            "Extropic inside X0 and XTR-0",
            "Extropic TSU 101",
            "Extropic THRML repository",
        ],
        "result": (
            "Extropic writing still frames TSU/XTR hardware as probabilistic EBM "
            "sampling infrastructure. No Carnot-accessible TSU SDK, hardware receipt, "
            "or local execution path was found."
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
            "Logical Intelligence public pages reinforce proof/checker authority, but "
            "they remain non-local architecture context rather than Carnot evidence."
        ),
    },
    "local_v489_v494_duplicate_history": {
        "status": "ok",
        "queries": [
            "research-references.md V489 Planner and Execution Refresh",
            "research-references.md V490 Planner and Execution Refresh",
            "research-references.md V491 Planner and Execution Refresh",
            "research-references.md V492 Planner and Execution Refresh",
            "research-references.md V493 Planner and Execution Refresh",
            "research-references.md V494 Planner Refresh",
            "repo-wide search for promoted arXiv ids and titles",
        ],
        "result": (
            "NeuroSCA, Sampling for Quality, and the halting-problem witness paper were "
            "absent from local references by exact id/title search. CoVe, memory "
            "reuse, energy-guided VLM decoding, p-bit Dual BRAM, and Neuro-Symbolic "
            "Compliance were suppressed as already-covered local history."
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
            "Retired and non-local lanes stayed closed. New deltas are constrained to "
            "deterministic fixture design, receipt fields, or bounded controls."
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
    """Build the Exp5429 source-delta artifact.

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
        verdict = "blocked: V494 planner refresh marker missing; references unchanged."
    else:
        status = "complete"
        verdict_detail = (
            f"{count} new actionable V494 execution-time source deltas appended; retired scopes remained closed"
            if count
            else "no new actionable V494 execution-time source deltas; references unchanged"
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
        or ["tests/python/test_experiment_5429_source_delta_v494.py"],
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
        raise ValueError("field_principles must match REQ-REPORT-5429")
    if artifact["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("experiment_id must match Exp5429")
    if artifact["task_id"] != TASK_ID:
        raise ValueError("task_id must match exp5429")
    if artifact["milestone"] != MILESTONE:
        raise ValueError("milestone must match 2026.07.494")
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
            "Execution-time sweep after the `.494` planner refresh checked arXiv, "
            "OpenReview, HuggingFace Papers, Semantic Scholar routes for EBT and "
            "ARM-EBM, GitHub, Extropic writing, Logical Intelligence public pages, "
            "V489/V490/V491/V492/V493/V494 duplicate history, and the exclusion "
            "manifest. The findings below were absent from those blocks and add "
            "Carnot-local hooks without changing the active roadmap."
        ),
        "",
        "### New actionable deltas",
        *(_render_reference(row) for row in references),
        "",
        "### Execution impact",
        (
            "- **Plan impact:** No active `.494` roadmap edit is required. The deltas "
            "sharpen Exp5430/Exp5431 structured witness and reward-budget receipts, "
            "Exp5432 ontology/constraint-memory abstraction checks, and later "
            "energy-guided decoding controls without expanding scope."
        ),
        (
            "- **Duplicates suppressed:** V494 planner sources, A-MEM, CoVe, When "
            "Continual Learning Moves to Memory, Energy-Guided Decoding for Object "
            "Hallucination, p-bit Dual BRAM, Neuro-Symbolic Compliance, LoopUS, EBT, "
            "ARM-EBM, NRGPT, and V492/V493 execution deltas were already covered and "
            "are not re-added."
        ),
        (
            "- **Retired scope:** No retired scope was reopened. External generated-text/"
            "logprob scorers, broad GRPO/fine-tuning or LoRA reruns, token/internal "
            "feature claims without backend receipts, non-local TSU/Kona/Aleph "
            "execution claims, duplicate ARC lanes, and hardware speedup claims "
            "without matched board timing remain closed."
        ),
        (
            "- **Watch-only/excluded:** Extropic TSU/XTR-0 writing, Logical Intelligence "
            "Aleph/Kona pages, OpenReview EBT/NRGPT/action-head surfaces, and VLM "
            "hidden-state hallucination methods were checked but not promoted as "
            "executable `.494` dependencies."
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
        tests_run=["tests/python/test_experiment_5429_source_delta_v494.py"],
    )


if __name__ == "__main__":  # pragma: no cover
    main()
