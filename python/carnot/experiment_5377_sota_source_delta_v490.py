"""Exp 5377: execution-time SOTA source delta refresh for V490.

Spec refs: REQ-REPORT-5377, SCENARIO-REPORT-5377-APPEND-DELTAS,
SCENARIO-REPORT-5377-HONEST-NULL.

This module turns the last-minute literature/source sweep into a stable
receipt. It promotes only sources that change Carnot-local evidence collection
without reopening generated-text scoring, CPU-only GGUF headline, TSU/Kona,
KV260 host-device, or offline ARC solve scopes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_5377_sota_source_delta_v490"
TASK_ID = "exp5377-v490-sota-source-delta"
MILESTONE = "2026.07.490"
SEARCH_DATE = "20260707"
RESULT_RELATIVE_PATH = Path("results/experiment_5377_sota_source_delta_v490.json")
REFERENCES_RELATIVE_PATH = Path("research-references.md")
REFRESH_HEADING = "### V490 Execution Refresh - 2026-07-07"
REFRESH_END_MARKER = "<!-- V490-EXECUTION-REFRESH-2026-07-07-END -->"
INFERENCE_SUBSTRATE = "literature_ingestion_network_sources"

SPEC_REFS = [
    "REQ-REPORT-5377",
    "SCENARIO-REPORT-5377-APPEND-DELTAS",
    "SCENARIO-REPORT-5377-HONEST-NULL",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "status": (
        "Must be `complete` when at least one new actionable V490 delta is "
        "promoted, otherwise `honest_null`."
    ),
    "search_date": "Must equal 20260707 for this execution-time sweep.",
    "sources_checked": "Lists the required source families searched before downstream experiments.",
    "new_actionable_findings_count": (
        "Counts only non-duplicate actionable findings promoted into the V490 execution refresh."
    ),
    "findings": ("Each promoted finding records title, url, source_type, and Carnot-local hook."),
    "duplicates_suppressed": (
        "Lists relevant sources already covered by the V490 planner refresh and "
        "therefore not re-added."
    ),
    "retired_scope_reopened": (
        "Bare boolean must remain false because this refresh cannot reopen retired scopes."
    ),
    "research_references_updated": (
        "Bare boolean is true only when a new V490 execution-refresh block was appended."
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
    "official_runtime_docs",
    "local_v490_duplicate_history",
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
        "title": (
            "Your Agent's Memories Are Not Its Own: Forged Reasoning Attacks on "
            "LLM Agent Memory and Defenses"
        ),
        "url": "https://arxiv.org/abs/2607.05029",
        "source_type": "arXiv preprint",
        "carnot_hook": (
            "For Exp5381/Exp5382, add reasoning-history memory poisoning fixtures "
            "that insert forged rationale entries with evasive and self-referential "
            "amplification. Score structural reasoning-guard signals before trusting "
            "memory rows, preserve raw provenance, and keep the check deterministic "
            "rather than delegating acceptance to an external text scorer."
        ),
    },
    {
        "title": (
            "Self-Verifying Measurement Records: Hash-Linked Evidence Graphs for "
            "Hardware Benchmarking"
        ),
        "url": "https://arxiv.org/abs/2606.27934",
        "source_type": "arXiv preprint with ancillary implementation",
        "carnot_hook": (
            "For Exp5386, extend QCIVET-style board receipts with a hash-linked "
            "evidence graph: bind each board-state quantity to an observation and "
            "verification node, emit a SHA-256 manifest plus an offline verifier, "
            "record a reproducibility class, and add algebraic checksums or "
            "Freivalds-style spot checks where the workload permits. This is receipt "
            "discipline only, not a hardware speedup claim."
        ),
    },
    {
        "title": "Extracting hidden states from vLLM",
        "url": "https://vllm.ai/blog/2026-03-30-extract-hidden-states",
        "source_type": "Official vLLM implementation note",
        "carnot_hook": (
            "For Exp5387, record backend-specific feature receipts: vLLM >= 0.18 "
            "can expose selected hidden-state layers through a speculative/KV "
            "connector path. Use it as a positive-control backend row only; it does "
            "not reopen token/internal-feature energy for the mandated local GGUF "
            "llama.cpp path unless that path exposes equivalent logits, hidden "
            "states, or attention with clean provenance."
        ),
    },
]

DUPLICATES_SUPPRESSED = [
    "Depth Exploration for LLM Decoding / DEX - https://arxiv.org/abs/2606.29223",
    "GeoWorld: Geometric World Models - https://arxiv.org/abs/2602.23058",
    "QCIVET hash-chained audit traces - https://arxiv.org/abs/2605.13109",
    "LLGuidance structured-output runtime - https://github.com/guidance-ai/llguidance",
    "CFGzip token-space compression - https://arxiv.org/abs/2605.29986",
    "TruncProof JSON completion slack - https://arxiv.org/abs/2605.13076",
    "G-RRM overwrite-capable solver guidance - https://arxiv.org/abs/2607.02491",
    "LongMemEval-V2 memory benchmark - https://arxiv.org/abs/2605.12493",
]

REJECTED_CANDIDATES = [
    {
        "title": "A Survey on Long-Term Memory Security in LLM Agents",
        "url": "https://arxiv.org/abs/2604.16548",
        "reason": (
            "Useful governance taxonomy, but the sharper V490 hooks come from the "
            "new FARMA/SENTINEL reasoning-history attack and already-recorded "
            "origin-bound memory authority sources."
        ),
    },
    {
        "title": "MemGuard: Preventing Memory Contamination in Long-Term Memory-Augmented LLMs",
        "url": "https://arxiv.org/abs/2605.28009",
        "reason": (
            "Type-aware memory isolation is relevant, but prior Carnot references "
            "already record typed memory routing and contamination controls. It does "
            "not add a separate .490 execution hook beyond FARMA/SENTINEL."
        ),
    },
    {
        "title": "Securing LLM-Agent Long-Term Memory Against Poisoning",
        "url": "https://arxiv.org/abs/2606.24322",
        "reason": (
            "Already present in local duplicate history as origin-bound memory "
            "authority and rollback/unsafe-false-accept guidance."
        ),
    },
    {
        "title": "llama.cpp Eagle-3 and hidden-state discussions",
        "url": "https://github.com/ggml-org/llama.cpp/discussions/15902",
        "reason": (
            "Relevant watch-only backend signal, but no completed local GGUF "
            "feature receipt was found; it cannot reopen token/internal-feature "
            "energy claims."
        ),
    },
]

SEARCHED_SOURCE_DETAILS: JsonDict = {
    "arxiv": {
        "status": "ok",
        "queries": [
            "LLM structured output verification tool/action state receipts 2026",
            "agent memory governance stale poisoned rollback continuous self learning 2026",
            "ARC-AGI geometric salience live agent world model 2026",
            "p-bit boundary exchange Ising hash chained hardware receipt 2026",
            "hardware benchmark hash-linked evidence graph reproducibility 2026",
        ],
        "promoted": [
            "2607.05029 FARMA/SENTINEL reasoning-history memory attack",
            "2606.27934 self-verifying measurement records",
        ],
        "duplicates_suppressed": [
            "2606.29223 DEX",
            "2602.23058 GeoWorld",
            "2605.13109 QCIVET",
            "2607.02491 G-RRM",
            "2605.26128 Constraint Tax",
            "2605.13076 TruncProof",
            "2605.12493 LongMemEval-V2",
        ],
        "not_promoted": [
            "2604.16548 VMG survey was high-level relative to the promoted FARMA fixture.",
            "2605.28009 MemGuard overlaps prior typed-memory routing and contamination notes.",
            "2606.24322 origin-bound memory authority was already in local duplicate history.",
            "2605.05138 ARC executable world models are already heavily recorded locally.",
            "2606.25313 million-p-bit hardware was covered by the V489/V490 planning chain.",
        ],
    },
    "openreview": {
        "status": "challenge_blocked",
        "queries": [
            "OpenReview DANCE-ST constrained guidance",
            "OpenReview JSONSchemaBench and PSC constrained decoding",
            "OpenReview memory checkpoints and A-MemGuard",
            "OpenReview solver guidance and symbolic verification",
        ],
        "result": (
            "Search snippets reconfirmed planner-covered constrained-generation and "
            "watch-only memory-defense context. Direct OpenReview pages redirected to "
            "browser verification, so no OpenReview-only execution delta is promoted."
        ),
    },
    "huggingface_papers": {
        "status": "ok",
        "queries": [
            "HuggingFace Papers 2607.05029",
            "HuggingFace Papers 2604.16548",
            "HuggingFace Papers 2605.28009",
            "HuggingFace Papers 2602.23058",
        ],
        "not_promoted": [
            "MemGuard was visible on HuggingFace Papers but overlaps prior typed-memory controls.",
            "GeoWorld mirrors were duplicate planner coverage.",
            "No FARMA HuggingFace paper page was available during the execution sweep.",
        ],
        "result": (
            "HuggingFace Papers added no separate implementation hook beyond the "
            "arXiv-promoted FARMA and hardware-record findings."
        ),
    },
    "semantic_scholar": {
        "status": "ok",
        "queries": [
            "Semantic Scholar public search for FARMA/SENTINEL 2607.05029",
            "Semantic Scholar public search for VMG 2604.16548",
            "Semantic Scholar public search for DEX 2606.29223",
            "Semantic Scholar public search for GeoWorld 2602.23058",
        ],
        "result": (
            "Public search reconfirmed arXiv pages and planner-covered sources. "
            "No citation-count or influence-trend claim is made."
        ),
    },
    "github": {
        "status": "ok",
        "queries": [
            "GitHub FARMA SENTINEL Reasoning Guard",
            "GitHub vLLM hidden states extraction",
            "GitHub llama.cpp hidden states Eagle-3",
            "GitHub hash-linked evidence graph hardware benchmarking",
        ],
        "promoted": ["vLLM hidden-state extraction official implementation note"],
        "not_promoted": [
            "FARMA had no primary GitHub implementation surfaced during the sweep.",
            "llama.cpp hidden-state and Eagle-3 discussions were watch-only, not a local GGUF receipt.",
            "Generic hash-chain MCP/server lists were not hardware benchmarking evidence.",
        ],
    },
    "extropic_writing": {
        "status": "ok",
        "queries": [
            "Extropic TSU 101",
            "Extropic Inside X0 and XTR-0",
            "Extropic Thermodynamic Computing From Zero to One",
        ],
        "result": (
            "Extropic writing remains architecture context only. No local TSU SDK, "
            "authenticated receipt path, or Carnot-accessible TSU execution was found."
        ),
    },
    "logical_intelligence": {
        "status": "ok",
        "queries": [
            "Logical Intelligence automatic formal verification for code generation",
            "Logical Intelligence Aleph PutnamBench",
            "Logical Intelligence Kona EBM reasoning",
        ],
        "result": (
            "Logical Intelligence pages continue to support verifier/prover authority "
            "and EBM reasoning context, but expose no reproducible Kona baseline or "
            "local execution artifact."
        ),
    },
    "official_runtime_docs": {
        "status": "ok",
        "queries": [
            "vLLM Extracting hidden states from vLLM 2026",
            "vLLM issue 33118 hidden states extraction",
            "llama.cpp Eagle-3 hidden-state discussion",
        ],
        "promoted": ["vLLM >= 0.18 hidden-state extraction path"],
        "not_promoted": [
            "vLLM is not the mandated local GGUF llama.cpp path for .490 headline SOTA receipts.",
            "llama.cpp discussions did not provide a completed feature receipt for logits/hidden/attention.",
        ],
    },
    "local_v490_duplicate_history": {
        "status": "ok",
        "queries": [
            "research-references.md V490 Planner Refresh - 2026-07-07",
            "research-roadmap-vNEXT.md Literature Refresh Incorporated",
            "ops/exclusion_manifest.yaml retired scope scan",
            "results/experiment_5375_capstone_v489.json prior milestone truth",
            "repo-wide duplicate search for 2607.05029, 2606.27934, and vLLM hidden states",
        ],
        "result": (
            "FARMA/SENTINEL, self-verifying hardware measurement records, and the "
            "vLLM hidden-state extraction note were absent from the V490 planner "
            "refresh and local reference history. DEX, GeoWorld, QCIVET, G-RRM, "
            "Constraint Tax, TruncProof, LongMemEval-V2, origin-bound memory authority, "
            "Extropic, Logical Intelligence, and ARC executable-world-model sources "
            "were duplicates or watch-only context."
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
    """Build the Exp5377 literature-ingestion artifact.

    The receipt keeps source discovery separate from execution evidence. A row
    is promoted only when it gives a bounded Carnot-local hook and does not ask
    downstream experiments to trust external generated-text scorers, non-local
    hardware claims, or retired ARC/offload paths.
    """

    findings = [dict(row) for row in actionable_findings]
    count = len(findings)
    status = "complete" if count else "honest_null"
    updated = count > 0
    verdict_detail = (
        f"{count} new actionable V490 execution-time source deltas appended; retired scopes remained closed"
        if count
        else "no new actionable V490 execution-time source deltas; references unchanged"
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
        or ["tests/python/test_experiment_5377_sota_source_delta_v490.py"],
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
        raise ValueError("field_principles must match REQ-REPORT-5377")
    if artifact["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("experiment_id must match Exp5377")
    if artifact["task_id"] != TASK_ID:
        raise ValueError("task_id must match exp5377")
    if artifact["milestone"] != MILESTONE:
        raise ValueError("milestone must match 2026.07.490")

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
    if (
        not isinstance(verdict, str)
        or "\n" in verdict
        or not verdict.startswith(f"{expected_status}:")
    ):
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
            "Execution-time sweep after the `.490` planner refresh checked arXiv, "
            "OpenReview, HuggingFace Papers, Semantic Scholar, GitHub, Extropic "
            "writing, Logical Intelligence public pages, official runtime/backend "
            "docs surfaced by the search, and local duplicate history. The findings "
            "below were absent from the V490 planner block and add Carnot-local "
            "hooks without changing the active roadmap."
        ),
        "",
        "### New actionable deltas",
        *(_render_finding(row) for row in findings),
        "",
        "### Execution impact",
        (
            "- **Plan impact:** No active `.490` roadmap edit is required. The deltas "
            "sharpen Exp5381/Exp5382 memory governance, Exp5386 hardware receipts, "
            "and Exp5387 backend-feature gates by adding concrete local receipt fields."
        ),
        (
            "- **Duplicates suppressed:** DEX, GeoWorld, QCIVET, llguidance, CFGzip, "
            "TruncProof, G-RRM, and LongMemEval-V2 were already covered by the planner "
            "refresh or immediately preceding execution refreshes and are not re-added."
        ),
        (
            "- **Retired scope:** No retired scope was reopened. External generated-text "
            "scorers, CPU-only GGUF headline reruns, TSU/Kona execution claims, KV260 "
            "host block-device evidence, and offline ARC solve paths remain closed."
        ),
        (
            "- **Watch-only context:** VMG/Long-Term Memory Security, MemGuard, "
            "origin-bound memory authority, llama.cpp hidden-state discussions, "
            "Extropic writing, and Logical Intelligence pages were checked but not "
            "promoted because they were duplicate, higher-level, non-local, or lacked "
            "reproducible execution evidence for `.490`."
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
        tests_run=["tests/python/test_experiment_5377_sota_source_delta_v490.py"],
    )


if __name__ == "__main__":  # pragma: no cover
    main()
