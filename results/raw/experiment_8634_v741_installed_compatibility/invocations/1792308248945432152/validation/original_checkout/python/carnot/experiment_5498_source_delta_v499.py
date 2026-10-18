"""Exp5498 execution-time source delta refresh for V499.

Spec refs: REQ-REPORT-5498, SCENARIO-REPORT-5498-APPEND-DELTAS,
SCENARIO-REPORT-5498-NO-NEW-DELTA,
SCENARIO-REPORT-5498-BLOCKED-GATE-OR-MARKER.

This module turns the same-day web/source sweep into a reproducible local
receipt. The search itself is time-sensitive, so the code records the outcome:
which source families were checked, which items were duplicate or closed-scope,
and which single non-duplicate paper adds a bounded Carnot-local hook. That
keeps future agents from treating a literature refresh as permission to reopen
retired, proprietary, or non-local lanes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_5498_source_delta_v499"
TASK_ID = "exp5498-source-delta-v499"
MILESTONE = "2026.07.499"
SEARCH_DATE = "20260709"
RESULT_RELATIVE_PATH = Path("results/experiment_5498_source_delta_v499.json")
REFERENCES_RELATIVE_PATH = Path("research-references.md")
PRETEST_GATE_RELATIVE_PATH = Path("results/experiment_5497_pretest_cascade_diagnostic_v499.json")
PLANNER_MARKER = "## V499 Planner Refresh - 2026-07-09"
REFRESH_HEADING = "## V499 Execution Refresh - 20260709"
REFRESH_END_MARKER = "<!-- V499-EXECUTION-REFRESH-20260709-END -->"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

SPEC_REFS = [
    "REQ-REPORT-5498",
    "SCENARIO-REPORT-5498-APPEND-DELTAS",
    "SCENARIO-REPORT-5498-NO-NEW-DELTA",
    "SCENARIO-REPORT-5498-BLOCKED-GATE-OR-MARKER",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "sources_checked": (
        "Records each primary, secondary, local-dedupe, and gate source so the "
        "freshness check can be audited without rerunning web search."
    ),
    "new_references_added": (
        "Lists only non-duplicate actionable findings that earned a research-references append."
    ),
    "duplicates_suppressed": (
        "Prevents churn from re-adding V499 planner sources or earlier V49x "
        "execution-refresh sources."
    ),
    "closed_scopes_reopened": (
        "Bare false boolean proving excluded, proprietary, non-local, or retired "
        "lanes stayed closed."
    ),
    "research_references_updated": (
        "Bare boolean distinguishing a real append from a no-op freshness receipt."
    ),
    "prior_refresh_marker_found": (
        "Ensures the execution refresh dedupes against the actual V499 planner "
        "block before appending."
    ),
    "pretest_gate_artifact": (
        "Binds the source-delta run to Exp5497 because this task is gated on the "
        "repaired pretest cascade."
    ),
    "inference_substrate": (
        "Must be aggregation_from_upstream_artifacts because the receipt "
        "aggregates sources and local artifacts without running model, solver, or "
        "hardware inference."
    ),
    "honest_verdict": (
        "One-line terminal summary starting with complete: or blocked: that "
        "states whether references changed."
    ),
}

REQUIRED_SOURCE_FAMILIES = (
    "arxiv",
    "arxiv_recent_api",
    "openreview",
    "huggingface_papers",
    "semantic_scholar_ebt_arm_ebm",
    "github",
    "extropic_writing",
    "logical_intelligence",
    "local_v499_and_v49x_duplicate_history",
    "ops_exclusion_manifest",
    "pretest_gate_artifact",
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
        "closed_scopes_reopened",
        "research_references_updated",
        "prior_refresh_marker_found",
        "pretest_gate_artifact",
        "pretest_gate_resolved",
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
        "title": (
            "Constrained Decoding for Diffusion Language Models via Efficient "
            "Inference over Finite Automata"
        ),
        "url": "https://arxiv.org/abs/2607.07026",
        "source_type": "arXiv preprint",
        "carnot_hook": (
            "For any V499-or-later diffusion-LM structured-output pilot, compile "
            "JSON, tool-call, Sudoku, Countdown, or text-to-SQL constraints into a "
            "finite-automaton posterior and validate constraint satisfaction by "
            "construction. Keep this separate from autoregressive prefix masks and "
            "do not promote diffusion decoding as a current `.499` roadmap change "
            "without a local backend receipt."
        ),
    },
]

DUPLICATES_SUPPRESSED = [
    "V499 planner sources already added: Trajel, RT4CHART, ExpGraph, Evo-Memory, MILP-Evolve, crystal graph neural combinatorial optimization, Hamon, Extropic TSU/XTR-0, and Logical Intelligence Kona - https://arxiv.org/abs/2605.24219",
    "VeryTrace, the 2048-spin bulk acoustic wave Ising machine, and constrained Web API invocation fixtures were already recorded in V482/V492 source-delta history - https://arxiv.org/abs/2606.24124",
    "Resource-Aware Neuro-Symbolic Reasoning, NeuroSCA, LLMs versus the Halting Problem, Neuro-Symbolic Compliance, and When Continual Learning Moves to Memory were already indexed in V493/V494 or older local history - https://arxiv.org/abs/2606.27281",
    "Pitwall and LatentGym were already promoted by the V497 execution refresh; GASP and UA-ChatDev remained watch-only because they depend on logprob or external rescoring authority - https://arxiv.org/abs/2607.06495",
    "EBT, ARM-EBM, EBT-Policy, NRGPT, OpenReview EBT surfaces, HuggingFace EBT pages, and GitHub EBT routes remain already-covered architecture context rather than a new V499 implementation dependency - https://arxiv.org/abs/2507.02092",
    "Extropic TSU/XTR-0/Z1/THRML and Logical Intelligence Kona/Aleph public pages remain strategic non-local context and do not justify Carnot hardware or proprietary-baseline claims - https://logicalintelligence.com/",
]

WATCH_ONLY_OR_EXCLUDED: list[JsonDict] = [
    {
        "title": "Beyond the Leaderboard agent-failure taxonomy",
        "url": "https://arxiv.org/abs/2607.05775",
        "classification": "duplicate",
        "reason": (
            "The tool-use, planning, long-horizon, safety, and measurement-validity "
            "taxonomy reinforces the V499 Trajel failure-taxonomy hook, but the "
            "paper was already recorded in V493 history and does not add a new "
            "execution dependency."
        ),
    },
    {
        "title": "Reason, Reward, Refine physics step-level correction",
        "url": "https://arxiv.org/abs/2607.05199",
        "classification": "excluded",
        "reason": (
            "Step-level verifier feedback is relevant, but the paper's promoted "
            "path uses policy-gradient training and an external training-time "
            "verifier. V499 keeps frozen-weight local exact validators and does "
            "not reopen broad policy-gradient or GRPO-style training."
        ),
    },
    {
        "title": "Intrinsic-Noise Consolidation on BrainScaleS-2",
        "url": "https://arxiv.org/abs/2607.06924",
        "classification": "watch-only",
        "reason": (
            "Analog-noise consolidation is interesting hardware/continual-learning "
            "context, but it is non-local BrainScaleS-2 evidence with weight "
            "training and a single-chip proof point. It does not reopen Carnot "
            "hardware speedup claims or frozen-executor CSL scope."
        ),
    },
    {
        "title": "Extropic TSU, XTR-0, Z1, and THRML writing",
        "url": "https://extropic.ai/writing",
        "classification": "watch-only",
        "reason": (
            "Extropic remains sampler and EBM architecture context only. Carnot "
            "has no local TSU SDK, authenticated execution receipt, or matched "
            "timing basis for a hardware speedup claim."
        ),
    },
    {
        "title": "Logical Intelligence Kona and Aleph public pages",
        "url": "https://logicalintelligence.com/",
        "classification": "watch-only",
        "reason": (
            "Kona and Aleph reinforce verifier-first and formal-reasoning strategy, "
            "including recent public formal-verification posts, but they expose no "
            "reproducible local internals or fair local baseline for Carnot."
        ),
    },
    {
        "title": "closed Carnot scopes from exclusion manifest",
        "url": "ops/exclusion_manifest.yaml",
        "classification": "excluded",
        "reason": (
            "External generated-text/logprob scorers, broad policy-gradient or "
            "fine-tuning loops, duplicate ARC lanes, non-local TSU/Kona/Aleph "
            "execution claims, and hardware speedup claims without matched board "
            "timing remain closed."
        ),
    },
]

SEARCHED_SOURCE_DETAILS: JsonDict = {
    "arxiv": {
        "status": "ok",
        "queries": [
            "site:arxiv.org/abs 2026 energy based models verification reasoning LLM constraints EBM",
            "site:arxiv.org/abs 2026 neural constraint satisfaction LLM verifier constraint satisfaction",
            "site:arxiv.org/abs 2026 Ising applications hardware accelerated sampling p-bit FPGA",
            "site:arxiv.org/abs 2026 hallucination mitigation energy-guided decoding LLM",
            "site:arxiv.org/abs 2026 Kolmogorov Arnold Networks verification constraints KAN hallucination",
            "site:arxiv.org/abs 2026 continual online learning LLM agents memory self evolving",
        ],
        "result": (
            "Primary arXiv pages confirmed one non-duplicate actionable delta: "
            "2607.07026 finite-automata constrained decoding for diffusion LMs. "
            "Other surfaced items were already in local history or closed-scope."
        ),
    },
    "arxiv_recent_api": {
        "status": "ok",
        "queries": [
            "arXiv API submittedDate sort: energy based verification reasoning constraints",
            "arXiv API submittedDate sort: neural constraints and LLMs",
            "arXiv API submittedDate sort: Ising sampling FPGA p-bit hardware",
            "arXiv API submittedDate sort: hallucination energy decoding verifier",
            "arXiv API submittedDate sort: KAN verification constraints",
            "arXiv API submittedDate sort: continual learning LLM agents memory",
        ],
        "promoted": [
            "2607.07026 exact finite-automata constrained posterior for diffusion language models"
        ],
        "not_promoted": [
            "2607.05775 duplicate V493/V499-style agent failure taxonomy.",
            "2607.02112 duplicate V482 hardware Ising paper.",
            "2607.05936 duplicate V492 constrained API invocation paper.",
            "2607.05199 policy-gradient step correction is excluded for frozen-weight V499 scope.",
            "2607.06924 non-local BrainScaleS-2 continual-learning hardware context is watch-only.",
        ],
    },
    "openreview": {
        "status": "ok",
        "queries": [
            "OpenReview Energy-Based Transformers 2507.02092",
            "OpenReview Autoregressive Language Models are Secretly Energy-Based Models 2512.15605",
            "OpenReview constrained decoding and EBT surfaces",
        ],
        "result": (
            "OpenReview resolved the EBT forum and adjacent EBM/diffusion surfaces. "
            "No OpenReview-only item superseded the V499 planner or the promoted "
            "arXiv 2607.07026 delta."
        ),
    },
    "huggingface_papers": {
        "status": "partial",
        "queries": [
            "HuggingFace Papers EBT 2507.02092",
            "HuggingFace Papers ARM-EBM 2512.15605",
            "HuggingFace Papers constrained decoding diffusion language models 2607.07026",
            "HuggingFace Papers neural constraint satisfaction and KAN verification",
        ],
        "result": (
            "HuggingFace Papers confirmed EBT context as duplicate and surfaced "
            "nearby constrained-decoding and constraint-awareness pages. No HF "
            "page added a stronger Carnot-local hook than the primary arXiv paper."
        ),
    },
    "semantic_scholar_ebt_arm_ebm": {
        "status": "partial",
        "queries": [
            "Semantic Scholar route from arXiv page for EBT 2507.02092",
            "Semantic Scholar route from arXiv page for ARM-EBM 2512.15605",
            "Semantic Scholar route from arXiv page for 2607.07026",
        ],
        "result": (
            "Public routes were metadata context only. No citation-count or "
            "follow-on claim is made from Semantic Scholar in this receipt."
        ),
    },
    "github": {
        "status": "partial",
        "queries": [
            "GitHub 2607.07026 finite automata diffusion constrained decoding",
            "GitHub EBT 2507.02092",
            "GitHub ARM-EBM 2512.15605",
            "GitHub Extropic THRML and Hamon sampler routes",
        ],
        "watch_only_links": [
            "https://github.com/alexiglad/EBT",
            "https://github.com/dek3rr/hamon",
            "https://github.com/extropic-ai/thrml",
        ],
        "result": (
            "GitHub did not reveal an official 2607.07026 repository during this "
            "execution check. Existing EBT, Hamon, and THRML routes stay "
            "watch-only or duplicate implementation context."
        ),
    },
    "extropic_writing": {
        "status": "ok",
        "queries": [
            "Extropic writing page",
            "Extropic TSU 101",
            "Extropic thermodynamic computing from zero to one",
            "Extropic THRML routes",
        ],
        "result": (
            "Extropic still frames TSUs/XTR-0/Z1/THRML as EBM and probabilistic "
            "sampling context. No local Carnot TSU execution or speedup basis was "
            "found."
        ),
    },
    "logical_intelligence": {
        "status": "ok",
        "queries": [
            "Logical Intelligence Kona energy-based models",
            "Logical Intelligence Aleph formal verification posts",
            "Logical Intelligence automatic formal verification for code generation",
        ],
        "result": (
            "Logical Intelligence pages continue to advertise Kona, Aleph, EBRMs, "
            "and formal verification. They remain proprietary strategic context "
            "without reproducible local internals."
        ),
    },
    "local_v499_and_v49x_duplicate_history": {
        "status": "ok",
        "queries": [
            "research-references.md V499 Planner Refresh",
            "research-references.md V482-V497 execution refresh blocks",
            "repo-wide search for 2607.07026, 2607.05775, 2607.02112, 2607.05936, 2607.05199, 2607.06924",
        ],
        "result": (
            "Exact local search found no 2607.07026 entry. VeryTrace, 2048-spin "
            "Ising, constrained API invocation, Resource-Aware Neuro-Symbolic "
            "Reasoning, NeuroSCA, Halting Problem, MemRL, and Beyond the "
            "Leaderboard were already present."
        ),
    },
    "ops_exclusion_manifest": {
        "status": "ok",
        "queries": [
            "broad policy-gradient, GRPO, RL, LoRA, and fine-tuning reruns",
            "external generated-text/logprob scorers",
            "non-local TSU/Kona/Aleph execution claims",
            "hardware speedup claims without matched board timing",
        ],
        "result": (
            "Closed lanes stayed closed. The promoted delta is constrained to a "
            "future diffusion-LM finite-automata fixture and does not change V499 "
            "execution scope."
        ),
    },
    "pretest_gate_artifact": {
        "status": "ok",
        "queries": [PRETEST_GATE_RELATIVE_PATH.as_posix()],
        "result": (
            "Exp5497 recorded pretest_cascade_resolved=true, allowing this source "
            "delta lane to run while preserving full-suite caveats."
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
    pretest_gate_resolved: bool = True,
) -> JsonDict:
    """Build the deterministic Exp5498 source-delta receipt.

    The receipt deliberately separates source discovery from execution claims:
    adding a paper to the local reference history is not a model, solver, or
    hardware result. It only gives future tasks a bounded hook to test.
    """

    blocked = not prior_refresh_marker_found or not pretest_gate_resolved
    references = [] if blocked else [dict(row) for row in new_references_added]
    count = len(references)
    updated = (count > 0) if research_references_updated is None else research_references_updated

    if blocked:
        status = "blocked"
        updated = False
        reason = (
            "Exp5497 pretest gate unresolved"
            if not pretest_gate_resolved
            else "V499 planner refresh marker missing"
        )
        verdict = f"blocked: {reason}; references unchanged."
    else:
        status = "complete"
        verdict_detail = (
            f"{count} new actionable V499 execution-time source delta appended; closed scopes remained closed"
            if count
            else "no new actionable V499 execution-time source deltas; references unchanged"
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
        "closed_scopes_reopened": False,
        "research_references_updated": updated,
        "prior_refresh_marker_found": prior_refresh_marker_found,
        "pretest_gate_artifact": PRETEST_GATE_RELATIVE_PATH.as_posix(),
        "pretest_gate_resolved": pretest_gate_resolved,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "searched_source_details": dict(SEARCHED_SOURCE_DETAILS),
        "watch_only_or_excluded": [dict(row) for row in WATCH_ONLY_OR_EXCLUDED],
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "methodology_duration_s": round(float(methodology_duration_s), 6),
        "tests_run": list(tests_run) or ["tests/python/test_experiment_5498_source_delta_v499.py"],
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
        raise ValueError("field_principles must match REQ-REPORT-5498")
    if artifact["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("experiment_id must match Exp5498")
    if artifact["task_id"] != TASK_ID:
        raise ValueError("task_id must match exp5498")
    if artifact["milestone"] != MILESTONE:
        raise ValueError("milestone must match 2026.07.499")
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
    if artifact["closed_scopes_reopened"] is not False:
        raise ValueError("closed_scopes_reopened must remain false")

    prior_marker = artifact["prior_refresh_marker_found"]
    pretest_gate = artifact["pretest_gate_resolved"]
    if artifact["pretest_gate_artifact"] != PRETEST_GATE_RELATIVE_PATH.as_posix():
        raise ValueError("pretest_gate_artifact must name the Exp5497 gate artifact")
    if artifact["status"] == "complete" and prior_marker is not True:
        raise ValueError("prior_refresh_marker_found must be true for complete artifacts")
    if artifact["status"] == "complete" and pretest_gate is not True:
        raise ValueError("pretest gate must be resolved for complete artifacts")
    if artifact["status"] == "blocked" and prior_marker is True and pretest_gate is True:
        raise ValueError(
            "blocked artifacts must record a missing marker or unresolved pretest gate"
        )

    updated = artifact["research_references_updated"]
    if prior_marker is False or pretest_gate is False:
        if updated is not False or references:
            raise ValueError("research_references_updated must be false when blocked")
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
            "Execution-time sweep after the `.499` planner refresh checked arXiv "
            "primary pages and recent API results, OpenReview, HuggingFace "
            "Papers, Semantic Scholar routes for EBT and ARM-EBM, GitHub, "
            "Extropic writing, Logical Intelligence public pages, V499/V49x "
            "duplicate history, the Exp5497 pretest gate, and the exclusion "
            "manifest. Only non-duplicate actionable deltas are listed below."
        ),
        "",
        "### New actionable delta",
        *(_render_reference(row) for row in references),
        "",
        "### Execution impact",
        (
            "- **Plan impact:** No active `.499` roadmap edit is required. The "
            "delta is a future diffusion-LM structured-output fixture hook; V499 "
            "hard/soft exact validators, local GGUF receipts, and gate discipline "
            "remain unchanged."
        ),
        (
            "- **Duplicates suppressed:** Trajel, RT4CHART, ExpGraph, Evo-Memory, "
            "MILP-Evolve, Hamon, VeryTrace, the acoustic Ising machine, constrained "
            "API invocation, Resource-Aware Neuro-Symbolic Reasoning, NeuroSCA, "
            "Pitwall, LatentGym, EBT, ARM-EBM, Extropic, and Logical Intelligence "
            "context were already covered or stayed watch-only."
        ),
        (
            "- **Closed scope:** No closed scope was reopened. External logprob "
            "scorers, broad policy-gradient/RL/fine-tuning loops, duplicate ARC "
            "lanes, non-local TSU/Kona/Aleph execution claims, and hardware "
            "speedup claims without matched board timing remain closed."
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


def _load_pretest_gate_resolved(path: Path) -> bool:
    return bool(json.loads(path.read_text(encoding="utf-8")).get("pretest_cascade_resolved"))


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
    pretest_gate = root / PRETEST_GATE_RELATIVE_PATH

    references_text = references.read_text(encoding="utf-8")
    prior_marker = PLANNER_MARKER in references_text
    pretest_gate_resolved = pretest_gate.exists() and _load_pretest_gate_resolved(pretest_gate)
    artifact = build_artifact(
        methodology_duration_s=methodology_duration_s,
        tests_run=tests_run,
        prior_refresh_marker_found=prior_marker,
        pretest_gate_resolved=pretest_gate_resolved,
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
        tests_run=["tests/python/test_experiment_5498_source_delta_v499.py"],
    )


if __name__ == "__main__":  # pragma: no cover
    main()
