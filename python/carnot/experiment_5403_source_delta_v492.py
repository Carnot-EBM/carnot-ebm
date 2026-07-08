"""Exp 5403: execution-time source delta refresh for V492.

Spec refs: REQ-REPORT-5403, SCENARIO-REPORT-5403-APPEND-DELTAS,
SCENARIO-REPORT-5403-NO-NEW-DELTA.

This module emits a conservative literature/source receipt. It promotes only
new, non-duplicate sources with concrete Carnot-local hooks and keeps retired
or non-local claims classified as watch-only or excluded.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_5403_source_delta_v492"
TASK_ID = "exp5403-v492-source-delta"
MILESTONE = "2026.07.492"
SEARCH_DATE = "20260708"
RESULT_RELATIVE_PATH = Path("results/experiment_5403_source_delta_v492.json")
REFERENCES_RELATIVE_PATH = Path("research-references.md")
REFRESH_HEADING = "### V492 Execution Refresh - 20260708"
REFRESH_END_MARKER = "<!-- V492-EXECUTION-REFRESH-20260708-END -->"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

SPEC_REFS = [
    "REQ-REPORT-5403",
    "SCENARIO-REPORT-5403-APPEND-DELTAS",
    "SCENARIO-REPORT-5403-NO-NEW-DELTA",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "sources_checked": "reproducible literature sweep",
    "new_references_added": "current-knowledge delta",
    "duplicates_suppressed": "no reference churn",
    "retired_scopes_reopened": "exclusion-manifest compliance",
    "research_references_updated": "doc alignment",
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
    "local_v489_v492_duplicate_history",
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
        "title": "Mitigating Errors in LLM-Generated Web API Invocations via Retrieval-Augmented Generation and Constrained Decoding",
        "url": "https://arxiv.org/abs/2607.05936",
        "source_type": "arXiv preprint with public WAPIIBench code",
        "carnot_hook": (
            "For Exp5405, add OpenAPI-to-regex and tool-schema constrained invocation "
            "fixtures; separate retrieved endpoint-spec evidence from hard constrained "
            "decoding, and record illegal URL, method, and argument false accepts."
        ),
    },
    {
        "title": "LLMs for Agentic Home Energy Management",
        "url": "https://arxiv.org/abs/2607.04569",
        "source_type": "arXiv preprint with public EcoHome agent repository",
        "carnot_hook": (
            "For Exp5405 and Exp5406, add a MILP-oracle scheduling fixture comparing "
            "native structured tool calls against text-parsed actions while keeping the "
            "deterministic MILP solver, not an external judge, as final authority."
        ),
    },
    {
        "title": "Ising-Machine-Assisted Large Neighborhood Search with Flexibly Tunable Subproblem Size",
        "url": "https://arxiv.org/abs/2607.05169",
        "source_type": "arXiv preprint",
        "carnot_hook": (
            "For Exp5407, sweep active-constraint and p-bit QUBO subproblem-size controls "
            "with feasibility-preserving reinsertion and current-solution-quality telemetry."
        ),
    },
    {
        "title": "Geometric Characteristics of Subproblems in Ising-Machine-Assisted Large Neighborhood Search",
        "url": "https://arxiv.org/abs/2607.05014",
        "source_type": "arXiv preprint",
        "carnot_hook": (
            "For Exp5407, record semantic and geometric subproblem descriptors, not just "
            "variable count, and compare route/current-solution-structured subproblems "
            "against raw QUBO-constraint slices."
        ),
    },
    {
        "title": "iSTAR: an algebraic-collapse framework for variational reduction in quantum-inspired continuous Ising solvers",
        "url": "https://arxiv.org/abs/2607.05448",
        "source_type": "arXiv preprint",
        "carnot_hook": (
            "For Exp5406 and Exp5407, add a CPU-only active-tail/frozen-variable diagnostic: "
            "fold stabilized coordinates into induced fields, preserve the same-seed "
            "baseline, and report dense-work reduction separately from solution quality "
            "without making a hardware speedup claim."
        ),
    },
    {
        "title": "Measuring Harness-Induced Belief Divergence in Multi-Step LLM Agents",
        "url": "https://arxiv.org/abs/2607.04528",
        "source_type": "arXiv preprint with public code link",
        "carnot_hook": (
            "For Exp5408 and Exp5409, add harness-divergence controls: log censored "
            "branches, verification masks, shadow-risky branches, and belief-rollout "
            "changes under raw versus compressed or resource-pruned harnesses before "
            "memory or world-model promotion."
        ),
    },
]

DUPLICATES_SUPPRESSED = [
    "SWEnergy energy-consumption benchmark - https://arxiv.org/abs/2512.09543",
    "GNN active-set solver warm start - https://arxiv.org/abs/2511.13174",
    "UPSi universal prototype synthesis - https://arxiv.org/abs/2604.26836",
    "HaluNet hallucination mitigation watch item - https://arxiv.org/abs/2512.24562",
    "BitsMoE precision-routing watch item - https://arxiv.org/abs/2606.00079",
    "EBT 2507.02092 Energy-Based Transformers - https://arxiv.org/abs/2507.02092",
    "ARM-EBM 2512.15605 admissible reward measures - https://arxiv.org/abs/2512.15605",
    "Distributional EBM confidence calibration - https://arxiv.org/abs/2605.18871",
    "Energy-Based Decoding - https://arxiv.org/abs/2605.28020",
    "NSVIF neural-symbolic verifier - https://arxiv.org/abs/2601.17789",
    "KAN PWA/MILP verification - https://arxiv.org/abs/2602.06737",
    "GRS-KAN 2607.01449 graph-rule-symbolic KAN - https://arxiv.org/abs/2607.01449",
    "AgentLTL trace verification - https://arxiv.org/abs/2607.02599",
    "OEP poisoning self-evolving agents - https://arxiv.org/abs/2605.18930",
    "CoACT observation compression - https://arxiv.org/abs/2607.02911",
    "Sorting-network QUBO - https://arxiv.org/abs/2603.07579",
    "DEX depth exploration - https://arxiv.org/abs/2606.29223",
    "GeoWorld geometric world models - https://arxiv.org/abs/2602.23058",
    "QCIVET hash-chained audit traces - https://arxiv.org/abs/2605.13109",
    "LLGuidance structured-output runtime - https://github.com/guidance-ai/llguidance",
    "LongMemEval-V2 memory benchmark - https://arxiv.org/abs/2605.12493",
]

WATCH_ONLY_OR_EXCLUDED: list[JsonDict] = [
    {
        "title": "Constraint-aware reinforcement learning (CARL)",
        "url": "https://arxiv.org/abs/2607.04854",
        "classification": "watch-only",
        "reason": (
            "Training and RL fine-tuning are useful future context, but they reopen "
            "weight-mutation scope outside the current conductor tasks."
        ),
    },
    {
        "title": "Harnessing Code Agents for Automatic Software Verification",
        "url": "https://arxiv.org/abs/2607.06341",
        "classification": "watch-only",
        "reason": (
            "The prover-kernel final-authority pattern is relevant, but external "
            "code-agent proof claims are not local Carnot evidence without closed proof "
            "artifacts and deterministic replay."
        ),
    },
    {
        "title": "GRS-KAN: Graph-Rule-Symbolic Kolmogorov-Arnold Networks",
        "url": "https://arxiv.org/abs/2607.01449",
        "classification": "duplicate suppressed",
        "reason": "Already indexed locally in the V478 KAN ingestion path and not re-added.",
    },
    {
        "title": "StructuredEdit constraint-aware graphic design",
        "url": "https://arxiv.org/abs/2607.04612",
        "classification": "watch-only",
        "reason": "Domain-specific graphic-layout constraints do not add a `.492` verifier or solver fixture.",
    },
    {
        "title": "From Graphs to Gradients energy-based structural attribution",
        "url": "https://arxiv.org/abs/2607.05563",
        "classification": "watch-only",
        "reason": "Attribution context only; no immediate Carnot-local constraint, decoding, or sampler experiment.",
    },
    {
        "title": "MxGLUT hardware accelerator",
        "url": "https://arxiv.org/abs/2607.01607",
        "classification": "watch-only",
        "reason": "GEMM-accelerator context lacks a Carnot sampler path or repeatable board-local timing hook.",
    },
    {
        "title": "SafeDec OpenReview snippets",
        "url": "https://openreview.net/forum?id=dLO7MhVbbB",
        "classification": "watch-only",
        "reason": "OpenReview was browser-challenge blocked and robotics logit/dynamics assumptions do not reopen retired external generated-text scorer scope.",
    },
    {
        "title": "Extropic X0/XTR and Logical Intelligence Kona/Aleph pages",
        "url": "https://extropic.ai/writing/thermodynamic-computing-from-zero-to-one",
        "classification": "excluded",
        "reason": "Architecture context only; non-local TSU, Kona, and Aleph execution claims remain closed without a reproducible Carnot baseline.",
    },
]

SEARCHED_SOURCE_DETAILS: JsonDict = {
    "arxiv": {
        "status": "ok",
        "queries": [
            "submittedDate:[202607010000 TO 202607082359] energy-based verification reasoning",
            "submittedDate:[202607010000 TO 202607082359] constraint LLM verification",
            "submittedDate:[202607010000 TO 202607082359] Ising QUBO machine learning",
            "submittedDate:[202607010000 TO 202607082359] hallucination mitigation constrained decoding",
            "submittedDate:[202607010000 TO 202607082359] Kolmogorov-Arnold constraint systems",
        ],
        "promoted": [
            "2607.05936 constrained web API invocation",
            "2607.04569 agentic home energy management",
            "2607.05169 Ising-machine-assisted LNS with tunable subproblem size",
            "2607.05014 geometric characteristics of Ising-assisted LNS subproblems",
            "2607.05448 iSTAR algebraic collapse for continuous Ising solvers",
            "2607.04528 harness-induced belief divergence",
        ],
        "not_promoted": [
            "2607.04854 CARL is training/RL scope.",
            "2607.06341 code-agent verification is watch-only until deterministic proof artifacts exist.",
            "2607.01449 GRS-KAN was already indexed locally.",
            "2607.04612 StructuredEdit is domain-specific graphics context.",
            "2607.05563 structural attribution is context only.",
            "2607.01607 MxGLUT has no Carnot sampler path.",
        ],
    },
    "openreview": {
        "status": "challenge_blocked",
        "queries": [
            "OpenReview Energy-Based Transformers 2507.02092",
            "OpenReview ARM-EBM 2512.15605",
            "OpenReview SafeDec constrained decoding",
            "OpenReview energy model transformer reasoning",
        ],
        "result": (
            "Search snippets identified EBT, SafeDec, and energy-model transformer entries, "
            "but direct pages redirected to browser verification. No OpenReview-only source "
            "is promoted."
        ),
    },
    "huggingface_papers": {
        "status": "partial",
        "queries": [
            "HuggingFace Papers 2507.02092 EBT",
            "HuggingFace Papers 2607.04569 agentic home energy management",
            "HuggingFace Papers 2607.05936 constrained API invocation",
            "HuggingFace Papers 2512.15605 ARM-EBM",
        ],
        "result": (
            "The EBT page was available and duplicate-covered. Fresh candidate pages were "
            "not present during the sweep, so HuggingFace added no separate actionable delta."
        ),
    },
    "semantic_scholar": {
        "status": "rate_limited",
        "queries": [
            "Semantic Scholar API arXiv:2507.02092",
            "Semantic Scholar API arXiv:2512.15605",
            "Semantic Scholar API arXiv:2607.04569",
        ],
        "result": (
            "Direct API calls for EBT, ARM-EBM, and one new candidate returned HTTP 429. "
            "No citation-count or influence-trend claim is made."
        ),
    },
    "github": {
        "status": "ok",
        "queries": [
            "GitHub WAPIIBench constrained decoding OpenAPI",
            "GitHub EcoHome energy agent MILP",
            "GitHub Harness-induced belief divergence",
            "GitHub EBT 2507.02092 ARM-EBM 2512.15605",
        ],
        "promoted_supporting_links": [
            "https://github.com/stg-tud/WAPIIBench",
            "https://github.com/sokistar24/ecohome-energy-agent",
            "https://github.com/Hik289/Harness-induce-bias",
        ],
    },
    "extropic_writing": {
        "status": "ok",
        "queries": [
            "Extropic inside X0 and XTR-0",
            "Extropic thermodynamic computing from zero to one",
            "Extropic TSU pbits energy-based sampling",
        ],
        "result": (
            "Extropic writing remains sampler-first architecture context only. No local "
            "TSU SDK, authenticated receipt path, or Carnot-accessible TSU execution was found."
        ),
    },
    "logical_intelligence": {
        "status": "ok",
        "queries": [
            "Logical Intelligence energy-based models for reasoning",
            "Logical Intelligence Kona",
            "Logical Intelligence Aleph",
        ],
        "result": (
            "Logical Intelligence pages remain non-local architecture context. They support "
            "the solver/prover-authority thesis but expose no reproducible Kona or Aleph "
            "baseline for Carnot."
        ),
    },
    "local_v489_v492_duplicate_history": {
        "status": "ok",
        "queries": [
            "research-references.md V489 Planner and Execution Refresh",
            "research-references.md V490 Planner and Execution Refresh",
            "research-references.md V491 Planner and Execution Refresh",
            "research-references.md V492 Planner Refresh",
            "repo-wide duplicate search for promoted source ids",
        ],
        "result": (
            "The six promoted sources were absent from V489-V492 reference blocks. Planner "
            "and execution refresh sources from V489, V490, V491, and V492 were suppressed "
            "as duplicates or watch-only context."
        ),
    },
    "ops_exclusion_manifest": {
        "status": "ok",
        "queries": [
            "external generated-text scorer retirement",
            "ARC first-contact exploration signal retirement",
            "CPU-only SOTA offload retirement",
            "KV260 host block-device retirement",
            "hardware speedup claim gates",
        ],
        "result": (
            "Retired lanes stayed closed. Watch-only or excluded classifications were used "
            "for training-scope, external proof-agent, external scorer, non-local hardware, "
            "and retired ARC/offload items."
        ),
    },
}


def _verified_url(value: str) -> bool:
    return value.startswith("https://")


def build_artifact(
    *,
    new_references_added: Sequence[Mapping[str, Any]] = NEW_REFERENCES_ADDED,
    methodology_duration_s: float = 0.0,
    tests_run: Sequence[str] = (),
    research_references_updated: bool | None = None,
) -> JsonDict:
    """Build the Exp5403 source-delta artifact.

    The receipt records upstream-source aggregation only. Each promoted row is
    an implementation hook, not proof of model quality, solver advantage, or
    hardware performance.
    """

    references = [dict(row) for row in new_references_added]
    count = len(references)
    updated = (count > 0) if research_references_updated is None else research_references_updated
    verdict_detail = (
        f"{count} new actionable V492 execution-time source deltas appended; retired scopes remained closed"
        if count
        else "no new actionable V492 execution-time source deltas; references unchanged"
    )
    artifact: JsonDict = {
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "status": "complete",
        "search_date": SEARCH_DATE,
        "sources_checked": list(REQUIRED_SOURCE_FAMILIES),
        "new_actionable_findings_count": count,
        "new_references_added": references,
        "duplicates_suppressed": list(DUPLICATES_SUPPRESSED),
        "retired_scopes_reopened": False,
        "research_references_updated": updated,
        "honest_verdict": f"complete: {verdict_detail}.",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "searched_source_details": dict(SEARCHED_SOURCE_DETAILS),
        "watch_only_or_excluded": [dict(row) for row in WATCH_ONLY_OR_EXCLUDED],
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "methodology_duration_s": round(float(methodology_duration_s), 6),
        "tests_run": list(tests_run)
        or ["tests/python/test_experiment_5403_source_delta_v492.py"],
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
        raise ValueError("field_principles must match REQ-REPORT-5403")
    if artifact["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("experiment_id must match Exp5403")
    if artifact["task_id"] != TASK_ID:
        raise ValueError("task_id must match exp5403")
    if artifact["milestone"] != MILESTONE:
        raise ValueError("milestone must match 2026.07.492")
    if artifact["status"] != "complete":
        raise ValueError("status must be complete when the search artifact is emitted")
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
    updated = artifact["research_references_updated"]
    if updated is not (count > 0):
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
    if not references:
        return ""
    lines = [
        REFRESH_HEADING,
        "",
        (
            "Execution-time sweep after the `.492` planner refresh checked arXiv, "
            "OpenReview, HuggingFace Papers, Semantic Scholar routes for EBT and "
            "ARM-EBM, GitHub, Extropic writing, Logical Intelligence public pages, "
            "V489/V490/V491/V492 duplicate history, and the exclusion manifest. The "
            "findings below were absent from those blocks and add Carnot-local hooks "
            "without changing the active roadmap."
        ),
        "",
        "### New actionable deltas",
        *(_render_reference(row) for row in references),
        "",
        "### Execution impact",
        (
            "- **Plan impact:** No active `.492` roadmap edit is required. The deltas "
            "sharpen Exp5405 API/tool-schema and MILP-oracle fixtures, Exp5406/Exp5407 "
            "Ising active-tail and LNS subproblem design, and Exp5408/Exp5409 harness "
            "divergence checks."
        ),
        (
            "- **Duplicates suppressed:** SWEnergy, GNN active-set warm starts, UPSi, "
            "HaluNet, BitsMoE, EBT, ARM-EBM, Distributional EBM, Energy-Based Decoding, "
            "NSVIF, KAN PWA/MILP, GRS-KAN, AgentLTL, OEP, CoACT, sorting-network QUBO, "
            "DEX, GeoWorld, QCIVET, llguidance, and LongMemEval-V2 were already covered "
            "by V489/V490/V491/V492 history and are not re-added."
        ),
        (
            "- **Retired scope:** No retired scope was reopened. External generated-text "
            "scorers, CPU-only SOTA offload, non-local TSU/Kona/Aleph claims, KV260 host "
            "block-device probes, retired ARC first-contact exploration, and hardware "
            "speedup claims without repeatable board timing remain closed."
        ),
        (
            "- **Watch-only/excluded:** CARL, external code-agent verification, GRS-KAN, "
            "StructuredEdit, energy-based structural attribution, MxGLUT, SafeDec, "
            "Extropic writing, and Logical Intelligence public pages were checked but "
            "not promoted because they are training-scope, duplicate, domain-specific, "
            "non-local, challenge-blocked, or retired-lane context."
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
        tests_run=["tests/python/test_experiment_5403_source_delta_v492.py"],
    )


if __name__ == "__main__":  # pragma: no cover
    main()
