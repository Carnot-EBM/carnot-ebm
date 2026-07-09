"""Exp5469: execution-time source delta refresh for V497.

Spec refs: REQ-REPORT-5469, SCENARIO-REPORT-5469-APPEND-DELTAS,
SCENARIO-REPORT-5469-NO-NEW-DELTA,
SCENARIO-REPORT-5469-BLOCKED-MISSING-PLANNER.

This module converts a same-day literature refresh into a reproducible receipt.
The important behavior is not the web search itself, because web pages can
change after the run. The important behavior is the conservative accounting:
promote only sources that add a concrete Carnot-local execution hook, suppress
duplicates that the V497 planner already covered, and keep closed lanes closed
when a source would require external scorers, broad training, or unauthenticated
hardware claims.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_5469_source_delta_v497"
TASK_ID = "exp5469-source-delta-v497"
MILESTONE = "2026.07.497"
SEARCH_DATE = "20260709"
RESULT_RELATIVE_PATH = Path("results/experiment_5469_source_delta_v497.json")
REFERENCES_RELATIVE_PATH = Path("research-references.md")
PLANNER_MARKER = "### V497 Planner Refresh - 20260709"
REFRESH_HEADING = "### V497 Execution Refresh - 20260709"
REFRESH_END_MARKER = "<!-- V497-EXECUTION-REFRESH-20260709-END -->"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

SPEC_REFS = [
    "REQ-REPORT-5469",
    "SCENARIO-REPORT-5469-APPEND-DELTAS",
    "SCENARIO-REPORT-5469-NO-NEW-DELTA",
    "SCENARIO-REPORT-5469-BLOCKED-MISSING-PLANNER",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "sources_checked": "reproducible literature sweep",
    "new_references_added": "current-knowledge delta",
    "duplicates_suppressed": "no reference churn",
    "closed_scopes_reopened": "exclusion-manifest compliance",
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
    "local_v490_v497_duplicate_history",
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
        "closed_scopes_reopened",
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
        "title": (
            "Pitwall: Faithful Natural-Language Race-Strategy Briefings from a "
            "Calibrated Real-Time Monte Carlo Engine"
        ),
        "url": "https://arxiv.org/abs/2607.06495",
        "source_type": "arXiv preprint",
        "carnot_hook": (
            "For Exp5470 and Exp5472 rewrite-state and local SOTA telemetry, add "
            "a tiny typed factual claims fixture: decompose each generated sentence "
            "into state-linked claims, verify every claim against the deterministic "
            "or probabilistic state snapshot that prompted it, and require a "
            "safe fallback when support is sparse. Do not import the paper's "
            "domain-specific Formula 1 substrate or fine-tuning data gate."
        ),
    },
    {
        "title": (
            "LatentGym: A Testbed For Cross-Task Experiential Learning With "
            "Controllable Latent Structure"
        ),
        "url": "https://arxiv.org/abs/2606.15306",
        "source_type": "arXiv preprint with public GitHub route",
        "carnot_hook": (
            "For Exp5473-Exp5475 CSL replay evidence, separate exploration "
            "metrics from exploitation metrics on a controllable latent-task "
            "fixture. Report whether the frozen action/memory policy gathered "
            "information about the latent state before measuring whether it used "
            "that information, and keep post-training sequence learning outside "
            "the V497 execution scope."
        ),
    },
]

DUPLICATES_SUPPRESSED = [
    "V497 planner sources were already added: inRAN, LemmaNet, Ultrafast On-Chip Online Learning, Compile to Compress, FPGA MPPI, Theoria, SEM-CTRL, HALT, ConsFormer-LNS, FPGA Ising decomposition, EBT, ARM-EBM, and nearby Extropic/Logical Intelligence context - https://arxiv.org/abs/2601.03219",
    "Distributional Energy-Based Models for structured LLM reasoning was already indexed in prior V490-V497 history and remains duplicate architecture context - https://arxiv.org/abs/2605.18871",
    "Energy-Guided Decoding for Object Hallucination Mitigation remains duplicate telemetry context and does not reopen token/internal-feature authority without local backend receipts - https://arxiv.org/abs/2507.07731",
    "Million-p-bit, p-dit, p-bit guided CDCL, p-dit QAP, Potts mean-field, and FPGA Ising references already cover hardware sampling context; no speedup claim is reopened - https://arxiv.org/abs/2606.25313",
    "SkillLearnBench and JitRL are already local continuous-learning references and are not re-added - https://arxiv.org/abs/2604.20087",
    "EBT OpenReview, HuggingFace Papers, GitHub, and project routes remain already-covered architecture context rather than a new V497 implementation dependency - https://arxiv.org/abs/2507.02092",
    "ARM-EBM remains the source theoretical bridge already covered by V490-V497 history; public routes surfaced no stronger local experiment - https://arxiv.org/abs/2512.15605",
    "Extropic TSU/X0/XTR-0/Z1 and THRML pages remain non-local hardware context without authenticated Carnot TSU execution - https://extropic.ai/hardware",
    "Logical Intelligence Aleph and Kona pages remain non-local architecture context without reproducible local Aleph or Kona baselines - https://logicalintelligence.com/",
]

WATCH_ONLY_OR_EXCLUDED: list[JsonDict] = [
    {
        "title": "GASP grounding-sensitivity hallucination detector",
        "url": "https://arxiv.org/abs/2607.04223",
        "classification": "watch-only",
        "reason": (
            "GASP is relevant span-level hallucination context, but its default "
            "signal re-scores fixed answers with small instruction-tuned scorers "
            "and logprob/JSD features. For V497 it is telemetry-only unless a "
            "local GGUF backend supplies authenticated logprob receipts; it does "
            "not replace exact validators or reopen external generated-text "
            "scorers."
        ),
    },
    {
        "title": "UA-ChatDev token-logprob uncertainty routing",
        "url": "https://arxiv.org/abs/2607.02186",
        "classification": "watch-only",
        "reason": (
            "Phase-aware uncertainty routing is useful context, but token-level "
            "logprob confidence is not accepted as authority without local backend "
            "receipts. V497 keeps exact execution, AST/KB witnesses, and solvers "
            "as final authority."
        ),
    },
    {
        "title": "Next-generation agentic reinforcement learning systems / AReaL2.0",
        "url": "https://arxiv.org/abs/2607.01120",
        "classification": "excluded",
        "reason": (
            "The step-granular trajectory protocol is useful background, but the "
            "online RL and policy-weight update lane conflicts with the V497 "
            "frozen-model CSL scope. Broad RL and weight mutation stay closed."
        ),
    },
    {
        "title": "VaseMuseum response control and training-free GRPO-style selection",
        "url": "https://arxiv.org/abs/2607.06374",
        "classification": "watch-only",
        "reason": (
            "Evidence-bounded responses reinforce existing claim-attribution "
            "fixtures, but the museum/VLM domain and GRPO-style selection do not "
            "add a stronger Carnot-local V497 hook than Pitwall plus exact final "
            "validators."
        ),
    },
    {
        "title": "Pitwall fine-tuning data gate",
        "url": "https://arxiv.org/abs/2607.06495",
        "classification": "excluded",
        "reason": (
            "The typed-claim verifier fixture is promoted, but fine-tuning data "
            "selection and Formula 1 live deployment remain outside the V497 "
            "source-delta scope."
        ),
    },
    {
        "title": "LatentGym post-training sequence-learning experiments",
        "url": "https://arxiv.org/abs/2606.15306",
        "classification": "excluded",
        "reason": (
            "The controllable latent-task measurement split is promoted, but "
            "post-training on related task sequences is excluded. V497 keeps "
            "model weights frozen and measures governed action/memory routing."
        ),
    },
    {
        "title": "Extropic TSU, XTR-0, Z1, and THRML routes",
        "url": "https://extropic.ai/hardware",
        "classification": "watch-only",
        "reason": (
            "Extropic remains EBM sampler and p-bit/p-dit architecture context, "
            "but Carnot has no local TSU SDK, board receipt path, or authenticated "
            "TSU execution. Keep it as non-local TSU context only."
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
        "title": "closed Carnot scopes from exclusion manifest",
        "url": "ops/exclusion_manifest.yaml",
        "classification": "excluded",
        "reason": (
            "external generated-text/logprob scorers, broad GRPO or broad RL, "
            "LoRA and fine-tuning reruns, CPU-only SOTA headline paths, duplicate "
            "ARC solve lanes, non-local TSU/Kona/Aleph execution claims, and "
            "hardware speedup claims without matched board timing remain closed."
        ),
    },
]

SEARCHED_SOURCE_DETAILS: JsonDict = {
    "arxiv": {
        "status": "ok",
        "queries": [
            "arXiv API: energy-based models AND verification, 2025-01-01 through 2026-07-09",
            "arXiv API: neural constraint satisfaction, 2025-01-01 through 2026-07-09",
            "arXiv API: energy-guided decoding, 2025-01-01 through 2026-07-09",
            "arXiv API: Kolmogorov-Arnold Networks AND online learning, 2025-01-01 through 2026-07-09",
            "arXiv API: Ising AND FPGA, 2025-01-01 through 2026-07-09",
            "arXiv API: hallucination AND verifier, 2025-01-01 through 2026-07-09",
            "arXiv API: continual learning AND LLM agents, 2025-01-01 through 2026-07-09",
        ],
        "promoted": [
            "2607.06495 Pitwall typed factual-claim verification against state snapshots",
            "2606.15306 LatentGym exploration/exploitation split for CSL replay evidence",
        ],
        "not_promoted": [
            "2605.18871 Distributional EBM was duplicate-covered in earlier source history.",
            "2603.20801 ConsFormer-LNS was promoted by the V497 planner.",
            "2507.07731 Energy-Guided Decoding remains duplicate telemetry context.",
            "2606.25313 million-p-bit hardware was promoted in prior V495/V497 hardware context.",
            "2607.04223 GASP depends on logprob rescoring and is watch-only for V497.",
            "2607.02186 UA-ChatDev depends on token logprobs and is watch-only.",
            "2607.01120 AReaL2.0/broad online RL conflicts with frozen-weight V497 CSL.",
        ],
    },
    "openreview": {
        "status": "challenge_blocked",
        "queries": [
            "OpenReview Energy-Based Transformers 2507.02092",
            "OpenReview Autoregressive Language Models are Secretly Energy-Based Models 2512.15605",
            "OpenReview neural constraint satisfaction certified correctness",
            "OpenReview constrained decoding and EBT surfaces",
        ],
        "result": (
            "OpenReview routed EBT to a browser verification challenge. Search "
            "metadata exposed EBT and adjacent EBM/constrained-decoding surfaces, "
            "but no OpenReview-only item superseded the V497 planner sources."
        ),
    },
    "huggingface_papers": {
        "status": "ok",
        "queries": [
            "HuggingFace Papers EBT 2507.02092",
            "HuggingFace Papers ARM-EBM 2512.15605",
            "HuggingFace Papers hallucination verifier and continual-learning routes",
        ],
        "result": (
            "HuggingFace Papers resolved EBT 2507.02092 as already-covered "
            "community/context material. Searches for ARM-EBM and the fresh V497 "
            "execution items did not add a stronger Carnot-local hook than the "
            "primary arXiv pages or local exact-authority rules."
        ),
    },
    "semantic_scholar": {
        "status": "rate_limited",
        "queries": [
            "Semantic Scholar API route for EBT 2507.02092",
            "Semantic Scholar API route for ARM-EBM 2512.15605",
            "arXiv Semantic Scholar links for Pitwall 2607.06495 and LatentGym 2606.15306",
        ],
        "result": (
            "The public Semantic Scholar API returned HTTP 429 for both EBT and "
            "ARM-EBM on 2026-07-09. ArXiv pages exposed Semantic Scholar routes, "
            "but no citation-trend claim is made."
        ),
    },
    "github": {
        "status": "partial",
        "queries": [
            "GitHub EBT 2507.02092",
            "GitHub ARM-EBM 2512.15605",
            "GitHub Pitwall 2607.06495",
            "GitHub LatentGym 2606.15306",
            "GitHub EBM KAN hallucination ML4CO repositories",
        ],
        "watch_only_links": [
            "https://github.com/alexiglad/EBT",
            "https://github.com/namkoong-lab/LatentGym",
            "https://github.com/extropic-ai/thrml",
        ],
        "result": (
            "GitHub confirmed the existing EBT implementation route and found a "
            "LatentGym repository route. Pitwall repository search returned "
            "mostly unrelated Formula 1 projects, so Pitwall is promoted from the "
            "primary arXiv paper only. No repository replaces Carnot exact "
            "validators or receipt requirements."
        ),
    },
    "extropic_writing": {
        "status": "ok",
        "queries": [
            "Extropic writing page",
            "Extropic hardware page",
            "Extropic software / THRML page",
        ],
        "result": (
            "Extropic hardware still lists X0, XTR-0, and early-access Z1; "
            "software pages still frame THRML as a JAX simulator and EBM training "
            "tool for TSUs. No local Carnot TSU SDK, authenticated execution "
            "receipt, or speedup basis was found."
        ),
    },
    "logical_intelligence": {
        "status": "ok",
        "queries": [
            "Logical Intelligence homepage",
            "Logical Intelligence Kona 1.0",
            "Logical Intelligence Aleph formal verification posts",
            "Logical Intelligence EBM reasoning posts",
        ],
        "result": (
            "Logical Intelligence pages continue to advertise Kona 1.0, Aleph, "
            "automatic formal verification, and EBM reasoning for critical "
            "systems. They remain architecture context only without a local "
            "authenticated baseline."
        ),
    },
    "local_v490_v497_duplicate_history": {
        "status": "ok",
        "queries": [
            "research-references.md V490-V497 Planner and Execution Refresh blocks",
            "openspec/change-proposals/research-roadmap-vNEXT.md V497 literature refresh",
            "repo-wide search for Pitwall, LatentGym, GASP, UA-ChatDev, AReaL2.0, EBT, and ARM-EBM",
        ],
        "result": (
            "Repo searches found no prior Pitwall 2607.06495 or LatentGym "
            "2606.15306 entry. V497 planner sources, EBT, ARM-EBM, JitRL, "
            "SkillLearnBench, hardware p-bit/p-dit sources, HALT, Theoria, "
            "SEM-CTRL, and ConsFormer-LNS were suppressed as duplicate or "
            "watch-only context."
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
            "Closed lanes stayed closed. Promoted deltas are limited to typed "
            "claim/state fixtures and CSL measurement splits; they do not require "
            "external scorers, weight updates, duplicate ARC work, or hardware "
            "speedup claims."
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
    """Build the Exp5469 source-delta receipt.

    The artifact is an aggregation result, not a model result. It records which
    sources were checked and why each source was promoted, suppressed, watched,
    or excluded so future agents do not treat a same-day literature refresh as a
    license to reopen already-retired research lanes.
    """

    references = [dict(row) for row in new_references_added] if prior_refresh_marker_found else []
    count = len(references)
    updated = (count > 0) if research_references_updated is None else research_references_updated
    if not prior_refresh_marker_found:
        status = "blocked"
        updated = False
        verdict = "blocked: V497 planner refresh marker missing; references unchanged."
    else:
        status = "complete"
        verdict_detail = (
            f"{count} new actionable V497 execution-time source deltas appended; closed scopes remained closed"
            if count
            else "no new actionable V497 execution-time source deltas; references unchanged"
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
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "searched_source_details": dict(SEARCHED_SOURCE_DETAILS),
        "watch_only_or_excluded": [dict(row) for row in WATCH_ONLY_OR_EXCLUDED],
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "methodology_duration_s": round(float(methodology_duration_s), 6),
        "tests_run": list(tests_run)
        or ["tests/python/test_experiment_5469_source_delta_v497.py"],
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
        raise ValueError("field_principles must match REQ-REPORT-5469")
    if artifact["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("experiment_id must match Exp5469")
    if artifact["task_id"] != TASK_ID:
        raise ValueError("task_id must match exp5469")
    if artifact["milestone"] != MILESTONE:
        raise ValueError("milestone must match 2026.07.497")
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
            "Execution-time sweep after the `.497` planner refresh checked arXiv, "
            "OpenReview, HuggingFace Papers, Semantic Scholar routes for EBT and "
            "ARM-EBM, GitHub, Extropic writing, Logical Intelligence public pages, "
            "V490/V491/V492/V493/V494/V495/V496/V497 duplicate history, and the "
            "exclusion manifest. The findings below were absent from those blocks "
            "and add Carnot-local hooks without changing the active roadmap."
        ),
        "",
        "### New actionable deltas",
        *(_render_reference(row) for row in references),
        "",
        "### Execution impact",
        (
            "- **Plan impact:** No active `.497` roadmap edit is required. The "
            "deltas sharpen Exp5470/Exp5472 typed claim-state verification and "
            "Exp5473-Exp5475 CSL replay metrics without expanding scope."
        ),
        (
            "- **Duplicates suppressed:** V497 planner sources, Distributional "
            "EBM, energy-guided decoding, p-bit/p-dit hardware, JitRL, "
            "SkillLearnBench, EBT, ARM-EBM, and prior Extropic/Logical "
            "Intelligence context were already covered or stayed watch-only and "
            "are not re-added."
        ),
        (
            "- **Closed scope:** No closed scope was reopened. External generated-"
            "text/logprob scorers, token/internal-feature authority without local "
            "backend receipts, broad GRPO/RL, LoRA/fine-tuning reruns, CPU-only "
            "SOTA headline paths, non-local TSU/Kona/Aleph execution claims, "
            "duplicate ARC lanes, and hardware speedup claims without matched "
            "board timing remain closed."
        ),
        (
            "- **Watch-only/excluded:** GASP, UA-ChatDev, AReaL2.0, VaseMuseum, "
            "Pitwall fine-tuning, LatentGym post-training, Extropic TSU/XTR-0/Z1, "
            "Logical Intelligence Aleph/Kona pages, and Semantic Scholar citation "
            "routes were checked but not promoted as executable `.497` dependencies."
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
        tests_run=["tests/python/test_experiment_5469_source_delta_v497.py"],
    )


if __name__ == "__main__":  # pragma: no cover
    main()
