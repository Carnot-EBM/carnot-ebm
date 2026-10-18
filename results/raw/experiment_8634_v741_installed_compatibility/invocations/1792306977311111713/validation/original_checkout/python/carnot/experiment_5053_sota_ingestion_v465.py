"""Exp 5053 SOTA ingestion for the .465 research frontier.

Spec refs: REQ-REPORT-5053, SCENARIO-REPORT-5053,
SCENARIO-REPORT-5053-DUPLICATE-FILTER.

This module records a literature ingestion pass. It is deliberately
deterministic: web research is summarized into source rows, duplicate filtering
is testable, and rerunning the module rewrites the JSON artifact and managed
reference section without changing unrelated docs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import json
import os
from pathlib import Path
from typing import Any


RESULT_RELATIVE_PATH = "results/experiment_5053_sota_ingestion_v465.json"
REFERENCES_RELATIVE_PATH = "research-references.md"
HONEST_VERDICT = "success_sota_ingestion_v465_actionable_references_added"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
REFERENCES_SECTION_START = "<!-- EXP5053-SOTA-INGESTION-V465-REFERENCES-START -->"
REFERENCES_SECTION_END = "<!-- EXP5053-SOTA-INGESTION-V465-REFERENCES-END -->"
TERMINAL_PREFIXES = ("blocked_", "complete:", "complete_", "success:", "success_")

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "references_added",
    "n_sources_checked",
    "selected_sources",
    "next_milestone_candidates",
    "research_references_updated",
    "inference_substrate",
    "preconditions_checked",
    "duplicate_filter",
    "field_principles",
    "reproducibility_checksum",
)

REQUIRED_USER_FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; complete verdict is "
            "success_sota_ingestion_v465_actionable_references_added."
        )
    },
    "references_added": {
        "principle": (
            "only genuinely actionable nonduplicate sources added to "
            "research-references.md, each with URL and concrete Carnot hook."
        )
    },
    "n_sources_checked": {
        "principle": (
            "count of source records checked across arXiv, OpenReview, Hugging Face "
            "Papers, GitHub, Extropic, Logical Intelligence, and EBT/ARM-EBM trails."
        )
    },
    "selected_sources": {
        "principle": (
            "deduplicated source records selected for .465 planning; no repeats from "
            "Exp 5038 or the .464 planning sweep."
        )
    },
    "next_milestone_candidates": {
        "principle": (
            "candidate .465 follow-up experiments with the source IDs that justify them."
        )
    },
    "research_references_updated": {
        "principle": (
            "true only when the managed Exp 5053 section has been written and validated."
        )
    },
}

FIELD_PRINCIPLES = {
    **REQUIRED_USER_FIELD_PRINCIPLES,
    "inference_substrate": {
        "principle": "aggregation_from_upstream_artifacts; no model training or live inference."
    },
    "preconditions_checked": {
        "principle": (
            "records required file reads, search channels, and forbidden side effects."
        )
    },
    "duplicate_filter": {
        "principle": (
            "explicit record of selected IDs and already-ingested IDs rejected before "
            "reference insertion."
        )
    },
    "field_principles": {"principle": "principle annotations are part of the artifact contract."},
    "reproducibility_checksum": {
        "principle": "sha256 over the stable selected-source and duplicate-filter payload."
    },
}

EXP5038_SOURCE_IDS = frozenset(
    {
        "2505.14999",
        "2605.10325",
        "2606.11209",
        "2503.22480",
        "2602.06291",
    }
)

V464_DUPLICATE_SOURCE_IDS = frozenset(
    {
        "2507.02092",
        "2507.07731",
        "2510.27545",
        "2512.05439",
        "2512.12850",
        "2512.15605",
        "2601.17789",
        "2601.21484",
        "2602.03034",
        "2603.02119",
        "2603.21558",
        "2606.18910",
        "2606.21724",
        "2606.25313",
        "2606.29366",
        "2605.12421",
    }
)

REQUIRED_DUPLICATE_IDS_IN_SPEC = frozenset(V464_DUPLICATE_SOURCE_IDS)
ALREADY_INGESTED_SOURCE_IDS = EXP5038_SOURCE_IDS | V464_DUPLICATE_SOURCE_IDS

SELECTED_SOURCE_IDS = [
    "2504.04718",
    "2510.08992",
    "2602.07223",
    "2604.01993",
    "2604.12046",
    "2605.26942",
    "2605.28020",
    "2606.26108",
]

SELECTED_SOURCES = [
    {
        "source_id": "2504.04718",
        "title": "T1: Tool-integrated Verification for Test-time Compute Scaling in Small Language Models",
        "url": "https://arxiv.org/abs/2504.04718",
        "channels": ["arXiv", "OpenReview", "Hugging Face Papers"],
        "tracks": ["verifier moat", "constraint satisfaction"],
        "source_signal": (
            "External tools handle memorization-heavy checks before a small verifier "
            "ranks candidates, giving a direct cheap-verifier path for test-time scaling."
        ),
        "carnot_hook": (
            ".465: add a tool-first cheap verifier gate before D6 judge fallback and "
            "charge tool calls separately from final verifier selection."
        ),
        "actionable": True,
    },
    {
        "source_id": "2510.08992",
        "title": "Constraints-of-Thought: A Framework for Constrained Reasoning in Language-Model-Guided Search",
        "url": "https://arxiv.org/abs/2510.08992",
        "channels": ["arXiv", "citation trail"],
        "tracks": ["constraint satisfaction", "verifier moat"],
        "source_signal": (
            "Represents each search step as intent plus constraint and uses that "
            "representation to prune infeasible MCTS branches."
        ),
        "carnot_hook": (
            ".465: prototype intent-constraint trace features for verifier training "
            "and measure whether pruning beats tuned self-consistency on frozen tasks."
        ),
        "actionable": True,
    },
    {
        "source_id": "2602.07223",
        "title": "Vegas: Self-Speculative Decoding with Verification-Guided Sparse Attention",
        "url": "https://arxiv.org/abs/2602.07223",
        "channels": ["arXiv", "GitHub/citation trail"],
        "tracks": ["hardware-accelerated decoding", "verifier moat"],
        "source_signal": (
            "Uses verification-guided sparse attention to make self-speculative "
            "decoding more efficient without a second draft model."
        ),
        "carnot_hook": (
            ".465: add a decoding-cost baseline where verifier evidence gates sparse "
            "attention before any larger judge call."
        ),
        "actionable": True,
    },
    {
        "source_id": "2604.01993",
        "title": "SAFE: An LLM-as-Verifier Framework for Evidence-Grounded Multi-Hop Reasoning",
        "url": "https://arxiv.org/abs/2604.01993",
        "channels": ["arXiv", "Hugging Face Papers"],
        "tracks": ["hallucination mitigation", "verifier moat"],
        "source_signal": (
            "Splits multi-hop claims into evidence-grounded checks so verifier output "
            "is tied to cited support instead of a free-form judge vote."
        ),
        "carnot_hook": (
            ".465: extend retrieval-NLI grounding with SAFE-style per-hop evidence "
            "checks and compare against the current semantic consistency verifier."
        ),
        "actionable": True,
    },
    {
        "source_id": "2604.12046",
        "title": "Think Through Uncertainty: Improving Long-Form Generation Factuality via Reasoning Calibration",
        "url": "https://arxiv.org/abs/2604.12046",
        "channels": ["arXiv", "Hugging Face Papers"],
        "tracks": ["hallucination mitigation", "continual verifier learning"],
        "source_signal": (
            "Calibrates generation around uncertainty-aware reasoning traces, making "
            "the uncertainty signal useful to downstream factuality checks."
        ),
        "carnot_hook": (
            ".465: use uncertainty trace features as replay examples for continual "
            "verifier calibration instead of only scalar confidence thresholds."
        ),
        "actionable": True,
    },
    {
        "source_id": "2605.26942",
        "title": "Neuro-Symbolic Verification of LLM Outputs for Data-Sensitive Domains",
        "url": "https://arxiv.org/abs/2605.26942",
        "channels": ["arXiv", "citation trail"],
        "tracks": ["constraint satisfaction", "hallucination mitigation"],
        "source_signal": (
            "Combines symbolic policy checks with neural output interpretation for "
            "domains where violations are concrete and auditable."
        ),
        "carnot_hook": (
            ".465: add a neuro-symbolic compliance fixture for the verifier moat so "
            "policy violations are evaluated as structured constraints."
        ),
        "actionable": True,
    },
    {
        "source_id": "2605.28020",
        "title": (
            "The Missing Piece in Pre-trained Model Evaluation: Reward-Guided Decoding "
            "Unlocks Task-Oriented Behavior Without Parameter Updates"
        ),
        "url": "https://arxiv.org/abs/2605.28020",
        "channels": ["arXiv", "GitHub/citation trail"],
        "tracks": ["energy-guided decoding", "hardware-accelerated decoding"],
        "source_signal": (
            "Uses reward-guided decoding as a parameter-free way to steer task "
            "behavior, making decoding-time guidance the intervention to measure."
        ),
        "carnot_hook": (
            ".465: adapt the semantic-energy selector into a reward-guided decoding "
            "arm and report accuracy per additional sampled token."
        ),
        "actionable": True,
    },
    {
        "source_id": "2606.26108",
        "title": "Where Larger Models Excel: The Primacy of Constraint-Guided Reasoning",
        "url": "https://arxiv.org/abs/2606.26108",
        "channels": ["arXiv", "Logical Intelligence/citation trail"],
        "tracks": ["constraint satisfaction", "continual verifier learning"],
        "source_signal": (
            "Frames scale gains as improved constraint-guided reasoning, which gives "
            "the verifier program a target for smaller-model distillation."
        ),
        "carnot_hook": (
            ".465: mine constraint-guided failure modes into a continual verifier "
            "learning replay set and separate scale benefit from verifier benefit."
        ),
        "actionable": True,
    },
]

DUPLICATE_SOURCES = [
    {
        "source_id": "2512.05439",
        "title": "BEAVER",
        "url": "https://arxiv.org/abs/2512.05439",
        "channels": ["arXiv", "Hugging Face Papers"],
        "tracks": ["verifier moat"],
        "source_signal": "Already recorded by the .464 planning sweep.",
        "carnot_hook": ".464 duplicate; do not re-add for .465.",
        "actionable": True,
    },
    {
        "source_id": "2512.12850",
        "title": "KANELE",
        "url": "https://arxiv.org/abs/2512.12850",
        "channels": ["arXiv"],
        "tracks": ["KAN/KANFIS"],
        "source_signal": "Already recorded by the .464 planning sweep.",
        "carnot_hook": ".464 duplicate; do not re-add for .465.",
        "actionable": True,
    },
    {
        "source_id": "2602.03034",
        "title": "KANFIS",
        "url": "https://arxiv.org/abs/2602.03034",
        "channels": ["arXiv", "GitHub/citation trail"],
        "tracks": ["KAN/KANFIS"],
        "source_signal": "Already recorded by the .464 planning sweep.",
        "carnot_hook": ".464 duplicate; do not re-add for .465.",
        "actionable": True,
    },
    {
        "source_id": "2507.02092",
        "title": "Energy-Based Transformers are Scalable Learners and Thinkers",
        "url": "https://arxiv.org/abs/2507.02092",
        "channels": ["arXiv", "Extropic/citation trail"],
        "tracks": ["energy-guided decoding"],
        "source_signal": "Already recorded by the .464 planning sweep.",
        "carnot_hook": ".464 duplicate; do not re-add for .465.",
        "actionable": True,
    },
    {
        "source_id": "2512.15605",
        "title": "ARM-EBM",
        "url": "https://arxiv.org/abs/2512.15605",
        "channels": ["arXiv", "citation trail"],
        "tracks": ["energy-guided decoding"],
        "source_signal": "Already recorded by the .464 planning sweep.",
        "carnot_hook": ".464 duplicate; do not re-add for .465.",
        "actionable": True,
    },
]

CANDIDATE_SOURCES = SELECTED_SOURCES + DUPLICATE_SOURCES
CANDIDATE_SOURCES_BY_ID = {source["source_id"]: source for source in CANDIDATE_SOURCES}
CANDIDATE_SOURCE_IDS = [source["source_id"] for source in CANDIDATE_SOURCES]

SOURCES_CHECKED = [
    {
        "channel": "arXiv",
        "query": "verifier moat, constraint satisfaction, energy-guided decoding 2025 2026",
        "source_ids": SELECTED_SOURCE_IDS
        + ["2512.05439", "2512.12850", "2602.03034", "2507.02092", "2512.15605"],
        "result": "selected nonduplicates; rejected prior .464 and Exp 5038 IDs",
    },
    {
        "channel": "OpenReview",
        "query": "ICLR 2026 verifier test-time compute small verifier T1",
        "source_ids": ["2504.04718"],
        "result": "T1 kept because it supplies a concrete tool-first verifier gate",
    },
    {
        "channel": "Hugging Face Papers",
        "query": "2026 verifier factuality constraint decoding",
        "source_ids": ["2504.04718", "2604.01993", "2604.12046"],
        "result": "papers matched arXiv records and nonduplicate Carnot hooks",
    },
    {
        "channel": "GitHub",
        "query": "verification guided sparse attention reward guided decoding code 2026",
        "source_ids": ["2602.07223", "2605.28020"],
        "result": "hardware/sampling entries kept as decoding-cost experiment hooks",
    },
    {
        "channel": "Extropic",
        "query": "thermodynamic sampling energy based transformer p-bit EBT ARM-EBM",
        "source_ids": ["2507.02092", "2512.15605", "2606.25313"],
        "result": "no new nonduplicate Carnot hook beyond .464 energy hardware entries",
    },
    {
        "channel": "Logical Intelligence",
        "query": "Kona constraint guided reasoning verification 2026",
        "source_ids": ["2606.26108"],
        "result": "kept constraint-guided reasoning taxonomy as verifier replay hook",
    },
    {
        "channel": "citation trails",
        "query": "EBT ARM-EBM KAN KANFIS continual verifier hallucination mitigation",
        "source_ids": [
            "2512.12850",
            "2602.03034",
            "2507.02092",
            "2512.15605",
            "2604.01993",
            "2604.12046",
            "2605.26942",
        ],
        "result": "KAN/KANFIS and EBT/ARM-EBM were duplicate-only; verification sources kept",
    },
    {
        "channel": "Exp 5038 artifact",
        "query": "prior verifier-moat selected source IDs",
        "source_ids": sorted(EXP5038_SOURCE_IDS),
        "result": "excluded all Exp 5038 sources from .465 additions",
    },
]

NEXT_MILESTONE_CANDIDATES = [
    {
        "candidate": "Tool-first verifier gate before judge fallback",
        "candidate_flag": "flagged_for_v465 (.465): tool_first_small_verifier_gate",
        "source_ids": ["2504.04718", "2604.01993"],
        "why": (
            "T1 and SAFE give a low-cost path where tools and evidence checks reduce "
            "the burden on a small verifier before expensive judge escalation."
        ),
    },
    {
        "candidate": "Constraint-guided verifier replay set",
        "candidate_flag": "flagged_for_v465 (.465): constraint_replay_continual_verifier",
        "source_ids": ["2510.08992", "2605.26942", "2606.26108"],
        "why": (
            "Constraint traces, neuro-symbolic checks, and scale taxonomy can be "
            "converted into verifier replay fixtures that distinguish constraint "
            "following from generic model scale."
        ),
    },
    {
        "candidate": "Guided decoding cost frontier",
        "candidate_flag": "flagged_for_v465 (.465): guided_decoding_cost_frontier",
        "source_ids": ["2602.07223", "2605.28020"],
        "why": (
            "Verification-guided sparse attention and reward-guided decoding let .465 "
            "measure whether sampling-time guidance buys accuracy per token instead "
            "of only per model call."
        ),
    },
    {
        "candidate": "Uncertainty-calibrated factuality loop",
        "candidate_flag": "flagged_for_v465 (.465): uncertainty_calibrated_factuality_loop",
        "source_ids": ["2604.01993", "2604.12046"],
        "why": (
            "Evidence-grounded verification plus uncertainty trace calibration gives "
            "a concrete hallucination-mitigation loop for continual verifier learning."
        ),
    },
]


def _require(condition: bool, message: str) -> None:
    if not condition:  # pragma: no cover
        raise ValueError(message)


def _stable_checksum(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _source_id_or_url_seen(source: Mapping[str, Any], existing_reference_text: str) -> bool:
    source_id = str(source["source_id"])
    return source_id in existing_reference_text or str(source["url"]) in existing_reference_text


def filter_actionable_sources(
    candidates: Iterable[Mapping[str, Any]], *, existing_reference_text: str
) -> dict[str, list[dict[str, Any]]]:
    """Select actionable sources while rejecting known and already-written duplicates."""

    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for candidate in candidates:
        source = dict(candidate)
        source_id = str(source["source_id"])
        if source_id in ALREADY_INGESTED_SOURCE_IDS:
            rejected.append({**source, "reason": "already_ingested"})
        elif _source_id_or_url_seen(source, existing_reference_text):
            rejected.append({**source, "reason": "already_in_research_references"})
        elif source.get("actionable") is not True:
            rejected.append({**source, "reason": "not_actionable"})
        else:
            selected.append(source)
    return {"selected": selected, "rejected": rejected}


def _build_duplicate_filter() -> dict[str, Any]:
    filtered = filter_actionable_sources(CANDIDATE_SOURCES, existing_reference_text="")
    selected_ids = [source["source_id"] for source in filtered["selected"]]
    rejected_duplicates = [
        {"source_id": source["source_id"], "title": source["title"], "reason": source["reason"]}
        for source in filtered["rejected"]
        if source["reason"] == "already_ingested"
    ]
    return {
        "selected_source_ids": selected_ids,
        "rejected_duplicate_source_ids": [row["source_id"] for row in rejected_duplicates],
        "rejected_duplicates": rejected_duplicates,
        "exp5038_source_ids_checked": sorted(EXP5038_SOURCE_IDS),
        "v464_duplicate_source_ids_checked": sorted(V464_DUPLICATE_SOURCE_IDS),
        "kan_kanfis_status": (
            "no_new_nonduplicate_actionable_source_found; KANELE and KANFIS were "
            "already recorded in the .464 sweep"
        ),
        "ebt_arm_ebm_status": (
            "no_new_nonduplicate_actionable_source_found; EBT, ARM-EBM, and p-bit "
            "hardware hits were already recorded in the .464 sweep"
        ),
    }


def _build_references_added(selected_sources: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "source_id": source["source_id"],
            "title": source["title"],
            "url": source["url"],
            "carnot_hook": source["carnot_hook"],
        }
        for source in selected_sources
    ]


def _build_preconditions_checked() -> dict[str, Any]:
    return {
        "agents_md_read": True,
        "codex_md_read": True,
        "research_references_read": True,
        "research_program_read": True,
        "exp5038_artifact_read": True,
        "roadmap_vnext_read": True,
        "arxiv_checked": True,
        "openreview_checked": True,
        "huggingface_papers_checked": True,
        "github_checked": True,
        "extropic_checked": True,
        "logical_intelligence_checked": True,
        "ebt_arm_ebm_citation_trails_checked": True,
        "deep_research_invoked": False,
        "research_conductor_modified": False,
        "ops_docs_modified": False,
        "kan_kanfis_duplicate_only": True,
        "ebt_arm_ebm_duplicate_only": True,
    }


def build_artifact(*, research_references_updated: bool) -> dict[str, Any]:
    """Build and validate the Exp 5053 JSON artifact."""

    duplicate_filter = _build_duplicate_filter()
    selected_sources = [dict(source) for source in SELECTED_SOURCES]
    checksum_payload = {
        "selected_source_ids": SELECTED_SOURCE_IDS,
        "duplicate_filter": duplicate_filter,
        "next_milestone_candidates": NEXT_MILESTONE_CANDIDATES,
    }
    artifact: dict[str, Any] = {
        "honest_verdict": HONEST_VERDICT,
        "references_added": _build_references_added(selected_sources),
        "n_sources_checked": len(SOURCES_CHECKED),
        "selected_sources": selected_sources,
        "next_milestone_candidates": [dict(candidate) for candidate in NEXT_MILESTONE_CANDIDATES],
        "research_references_updated": research_references_updated,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": _build_preconditions_checked(),
        "duplicate_filter": duplicate_filter,
        "field_principles": dict(FIELD_PRINCIPLES),
        "reproducibility_checksum": _stable_checksum(checksum_payload),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed if the artifact drifts from REQ-REPORT-5053."""

    _require(set(artifact) == set(REQUIRED_ARTIFACT_FIELDS), "artifact fields mismatch")
    verdict = artifact["honest_verdict"]
    _require(
        isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES),
        "honest_verdict must use a terminal prefix",
    )
    _require(verdict == HONEST_VERDICT, "unexpected complete verdict")
    _require(
        artifact["inference_substrate"] == INFERENCE_SUBSTRATE,
        "inference substrate mismatch",
    )
    _require(artifact["field_principles"] == FIELD_PRINCIPLES, "field principles mismatch")
    _require(
        artifact["research_references_updated"] in {True, False},
        "research_references_updated must be a bool",
    )
    _require(
        artifact["n_sources_checked"] == len(SOURCES_CHECKED),
        "source check count mismatch",
    )

    preconditions = artifact["preconditions_checked"]
    _require(isinstance(preconditions, dict), "preconditions must be a dict")
    _require(preconditions.get("deep_research_invoked") is False, "deep-research banned")
    _require(
        preconditions.get("research_conductor_modified") is False,
        "research conductor must not be modified",
    )
    _require(preconditions.get("ops_docs_modified") is False, "ops docs must not be modified")

    selected_sources = artifact["selected_sources"]
    _require(isinstance(selected_sources, list), "selected_sources must be a list")
    selected_ids = [source["source_id"] for source in selected_sources]
    _require(selected_ids == SELECTED_SOURCE_IDS, "selected source IDs mismatch")
    _require(
        not set(selected_ids).intersection(ALREADY_INGESTED_SOURCE_IDS),
        "selected source repeats prior ingestion",
    )
    for source in selected_sources:
        _require(source["actionable"] is True, "selected source must be actionable")
        _require(str(source["url"]).startswith("https://arxiv.org/abs/"), "bad source URL")
        _require(".465" in str(source["carnot_hook"]), "missing .465 hook")
        _require(source["tracks"], "selected source must declare tracks")

    references_added = artifact["references_added"]
    _require(
        [entry["source_id"] for entry in references_added] == SELECTED_SOURCE_IDS,
        "references_added source IDs mismatch",
    )
    for entry in references_added:
        _require(str(entry["url"]).startswith("https://arxiv.org/abs/"), "bad ref URL")
        _require(".465" in str(entry["carnot_hook"]), "reference hook missing .465")

    duplicate_filter = artifact["duplicate_filter"]
    _require(
        duplicate_filter["selected_source_ids"] == SELECTED_SOURCE_IDS,
        "duplicate-filter selected IDs mismatch",
    )
    _require(
        {"2512.05439", "2602.03034"}.issubset(
            set(duplicate_filter["rejected_duplicate_source_ids"])
        ),
        "required duplicate rejection missing",
    )
    _require(
        duplicate_filter["kan_kanfis_status"].startswith("no_new_nonduplicate"),
        "KAN/KANFIS duplicate status mismatch",
    )
    _require(
        duplicate_filter["ebt_arm_ebm_status"].startswith("no_new_nonduplicate"),
        "EBT/ARM-EBM duplicate status mismatch",
    )

    candidates = artifact["next_milestone_candidates"]
    _require(len(candidates) >= 3, "need at least three milestone candidates")
    for candidate in candidates:
        _require(".465" in candidate["candidate_flag"], "candidate must target .465")
        _require(candidate["source_ids"], "candidate needs source IDs")

    expected_checksum = _stable_checksum(
        {
            "selected_source_ids": SELECTED_SOURCE_IDS,
            "duplicate_filter": duplicate_filter,
            "next_milestone_candidates": NEXT_MILESTONE_CANDIDATES,
        }
    )
    _require(
        artifact["reproducibility_checksum"] == expected_checksum,
        "reproducibility checksum mismatch",
    )


def _build_references_section(artifact: Mapping[str, Any]) -> str:
    validate_artifact(artifact)
    lines = [
        REFERENCES_SECTION_START,
        "## Exp 5053 .465 SOTA ingestion source set",
        "",
        f"- **Artifact:** `{RESULT_RELATIVE_PATH}`",
        f"- **Honest verdict:** `{artifact['honest_verdict']}`",
        "- **Scope:** verifier moat, guided decoding, constraints, hallucination mitigation, continual verifier learning, and sampling hardware.",
        "- **Duplicate handling:** prior Exp 5038 and .464 planning-sweep entries were checked and not re-added.",
        "",
    ]
    for source in artifact["selected_sources"]:
        lines.extend(
            [
                f"### {source['title']}",
                f"- **Source:** arXiv:{source['source_id']} - {source['url']}",
                f"- **Tracks:** {', '.join(source['tracks'])}",
                f"- **Carnot hook:** {source['carnot_hook']}",
                f"- **Actionability:** {source['source_signal']}",
                "",
            ]
        )
    lines.extend([REFERENCES_SECTION_END, ""])
    return "\n".join(lines)


def _replace_marked_section(text: str, section: str) -> str:
    if REFERENCES_SECTION_START not in text:
        return f"{text.rstrip()}\n\n{section}".lstrip()
    before, rest = text.split(REFERENCES_SECTION_START, 1)
    _require(
        REFERENCES_SECTION_END in rest,
        "existing Exp 5053 reference section is missing end marker",
    )
    _, after = rest.split(REFERENCES_SECTION_END, 1)
    return f"{before.rstrip()}\n\n{section}{after.lstrip()}"


def update_research_references_text(text: str, artifact: Mapping[str, Any]) -> str:
    """Insert or replace the managed Exp 5053 section in research references."""

    section = _build_references_section(artifact)
    updated = _replace_marked_section(text, section)
    validate_research_references_text(updated, artifact)
    return updated


def validate_research_references_text(text: str, artifact: Mapping[str, Any]) -> None:
    """Validate the managed reference section without parsing the whole bibliography."""

    _require(REFERENCES_SECTION_START in text, "reference section start missing")
    _require(REFERENCES_SECTION_END in text, "reference section end missing")
    section = text.split(REFERENCES_SECTION_START, 1)[1].split(REFERENCES_SECTION_END, 1)[0]
    _require(HONEST_VERDICT in section, "honest verdict missing from reference section")
    for source_id in SELECTED_SOURCE_IDS:
        _require(f"arXiv:{source_id}" in section, f"missing selected source {source_id}")
        _require(f"https://arxiv.org/abs/{source_id}" in section, "missing source URL")
    _require("Carnot hook" in section, "Carnot hooks missing from reference section")
    _require("arXiv:2512.05439" not in section, "duplicate BEAVER source was re-added")
    _require(
        artifact["research_references_updated"] is True,
        "references can only validate against a write-complete artifact",
    )


def write_outputs(*, artifact_path: Path, references_path: Path) -> dict[str, Any]:
    """Write the stable JSON artifact and managed reference section."""

    artifact = build_artifact(research_references_updated=True)
    references_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    existing_references = (
        references_path.read_text(encoding="utf-8")
        if references_path.exists()
        else "# Research References\n\n"
    )
    updated_references = update_research_references_text(existing_references, artifact)
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    references_path.write_text(updated_references, encoding="utf-8")
    return artifact


def main() -> int:
    root = Path(os.environ.get("CARNOT_EXP5053_ROOT", Path(__file__).resolve().parents[2]))
    write_outputs(
        artifact_path=root / RESULT_RELATIVE_PATH,
        references_path=root / REFERENCES_RELATIVE_PATH,
    )
    print(HONEST_VERDICT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
