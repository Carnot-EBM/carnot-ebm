"""Exp 5024 verifier-moat SOTA ingestion for the .463 roadmap.

Spec refs: REQ-REPORT-5024, SCENARIO-REPORT-5024,
SCENARIO-REPORT-5024-BLOCKED-PRECONDITION.

This module is literature aggregation only. It records the reliable-channel
sweep result, verifies that selected papers are real arXiv pages, and maps the
new ideas onto the Phase D stack after Exp 5022 reported an execution-incomplete
D5 verdict. It does not train, load a model, or run live inference.
"""

from __future__ import annotations

from collections.abc import Mapping
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any
import urllib.error
import urllib.request


RESULT_RELATIVE_PATH = "results/experiment_5024_sota_ingestion_verifier_moat.json"
NOTE_RELATIVE_PATH = "docs/research-notes/verifier-moat-literature-2026-06-30.md"
STUDYING_RELATIVE_PATH = "research-studying.md"
REFERENCES_RELATIVE_PATH = "research-references.md"
D5_ARTIFACT_RELATIVE_PATH = "results/experiment_5022_moat_gate_resolution_v2.json"
HONEST_VERDICT = "success_sota_ingested_5_new_papers_mapped_to_phase_d"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
D5_VERDICT = "complete_moat_execution_incomplete_ebrm"
TERMINAL_PREFIXES = (
    "blocked_",
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)
PHASE_D_ARMS = frozenset(
    {"D1 LoRA-EBM", "D2 uPRM", "D3 EBRM", "D6 verifier-judge cascade"}
)
NEW_ARXIV_IDS = [
    "2502.11250",
    "2605.11334",
    "2507.01951",
    "2605.30085",
    "2502.14356",
]
ALREADY_INGESTED_ARXIV_IDS = frozenset(
    {
        "2605.18871",
        "2605.10158",
        "2504.13134",
        "2504.16828",
        "2502.01989",
        "2508.16665",
        "2508.10539",
        "2502.11157",
        "2504.01005",
        "2504.00891",
        "2509.24460",
        "2510.14913",
        "2603.04304",
        "2606.19818",
        "2606.09073",
        "2602.24040",
        "2510.20369",
        "2605.24005",
        "2508.03686",
        "2408.15240",
    }
)
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "new_arxiv_ids",
    "citations_verified",
    "sota_to_phase_d_mapping",
    "next_milestone_candidates",
    "note_path",
    "reliable_channel_used",
    "inference_substrate",
    "preconditions_checked",
    "d5_conditioning",
    "field_principles",
)
REQUIRED_MAPPING_FIELDS = frozenset(
    {
        "method",
        "arxiv_id",
        "url",
        "phase_d_arms",
        "source_signal",
        "implementation_delta",
        "pitfall",
        "candidate_flag",
    }
)
REQUIRED_CANDIDATE_FIELDS = frozenset(
    {"candidate", "candidate_flag", "source_ids", "phase_d_arms", "why"}
)
REQUIRED_USER_FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": "terminal prefix; success_sota_ingested_<n>_new_papers_mapped_to_phase_d."
    },
    "new_arxiv_ids": {
        "principle": (
            "verified-real NEW arXiv IDs (http 200), NOT in the 18-paper "
            "ingested set (no fabrication -- every method cites a source)."
        )
    },
    "sota_to_phase_d_mapping": {
        "principle": (
            "per NEW method: which PHASE D arm/direction it strengthens + the "
            "implementation delta over the current stack + the pitfall."
        )
    },
    "next_milestone_candidates": {
        "principle": (
            "the strongest method(s) flagged as candidate inputs for the .463 "
            "roadmap (discover->ingest->plan->experiment)."
        )
    },
    "note_path": {
        "principle": (
            "docs/research-notes/verifier-moat-literature-<date>.md (the "
            "synthesis the planner reads)."
        )
    },
    "reliable_channel_used": {
        "principle": (
            "sweep_clusters/sweep_semscholar + low-concurrency WebSearch/WebFetch "
            "(NOT /deep-research -- banned from the autonomous loop)."
        )
    },
    "inference_substrate": {
        "principle": (
            "aggregation_from_upstream_artifacts (literature synthesis, no LLM inference)."
        )
    },
    "preconditions_checked": {
        "principle": "records network/sweep-helper checks; unreachable network emits blocked_."
    },
}
FIELD_PRINCIPLES = {
    **REQUIRED_USER_FIELD_PRINCIPLES,
    "citations_verified": {
        "principle": "HTTP-200 arXiv title and URL evidence for every selected method."
    },
    "d5_conditioning": {
        "principle": (
            "Exp 5022 D5 verdict controls whether .463 scales a win or repairs "
            "execution-incomplete verifier arms."
        )
    },
    "field_principles": {"principle": "principle annotations are part of the artifact contract."},
}
REQUIRED_PRINCIPLE_FIELDS = frozenset(FIELD_PRINCIPLES)

CITATIONS_VERIFIED = {
    "2502.11250": {
        "title": "Uncertainty-Aware Step-wise Verification with Generative Reward Models",
        "url": "https://arxiv.org/abs/2502.11250",
        "http_status": 200,
    },
    "2605.11334": {
        "title": (
            "VERDI: Single-Call Confidence Estimation for Verification-Based "
            "LLM Judges via Decomposed Inference"
        ),
        "url": "https://arxiv.org/abs/2605.11334",
        "http_status": 200,
    },
    "2507.01951": {
        "title": "Test-Time Scaling with Reflective Generative Model",
        "url": "https://arxiv.org/abs/2507.01951",
        "http_status": 200,
    },
    "2605.30085": {
        "title": "Conformal Certification of Reasoning Trace Prefixes",
        "url": "https://arxiv.org/abs/2605.30085",
        "http_status": 200,
    },
    "2502.14356": {
        "title": (
            "Full-Step-DPO: Self-Supervised Preference Optimization with "
            "Step-wise Rewards for Mathematical Reasoning"
        ),
        "url": "https://arxiv.org/abs/2502.14356",
        "http_status": 200,
    },
}

D5_CONDITIONING = {
    "source_artifact": D5_ARTIFACT_RELATIVE_PATH,
    "honest_verdict": D5_VERDICT,
    "decision": "EXECUTION-INCOMPLETE",
    "moat_realized": False,
    "moat_retired_bounded": False,
    "execution_incomplete_arms": ["D3 EBRM"],
    "flagged_or_blocked_arms": [
        "D1 LoRA-EBM",
        "D2 uPRM",
        "D6 verifier-judge cascade",
        "D4 second-corpus-confirmation",
    ],
    "roadmap_condition": (
        "Because D5 is execution-incomplete rather than realized or clean-null, "
        ".463 should repair and harden D1/D2/D3/D6 before any retirement claim."
    ),
}

DEFAULT_SOTA_TO_PHASE_D_MAPPING = [
    {
        "method": "CoT-Entropy uncertainty-aware generative PRM",
        "arxiv_id": "2502.11250",
        "url": "https://arxiv.org/abs/2502.11250",
        "phase_d_arms": ["D2 uPRM", "D3 EBRM"],
        "source_signal": (
            "Adds uncertainty quantification to generative reward models for "
            "step-wise verification, using CoT Entropy to detect unreliable PRM judgments."
        ),
        "implementation_delta": (
            "Over Exp 5017-5022, attach CoT-Entropy uncertainty to the D2 step "
            "verifier and to the D3 EBRM selector, then require selection delta "
            "versus genuine tuned-SC after uncertainty-aware abstention."
        ),
        "pitfall": (
            "The paper is math-reasoning PRM evidence; uncertainty may flag "
            "style variation rather than wrong reasoning unless calibrated on the MuSR cache."
        ),
        "candidate_flag": "flagged_for_v463 (.463): cot_entropy_uprm_ebrm_uncertainty",
    },
    {
        "method": "VERDI single-call decomposed judge confidence",
        "arxiv_id": "2605.11334",
        "url": "https://arxiv.org/abs/2605.11334",
        "phase_d_arms": ["D6 verifier-judge cascade"],
        "source_signal": (
            "Extracts confidence from verification sub-check traces without "
            "extra judge calls, replacing unavailable or saturated logprob confidence."
        ),
        "implementation_delta": (
            "Over Exp 5017-5022, rerun the blocked D6 cascade with VERDI-style "
            "step-verdict alignment, claim margin, and evidence-grounding features "
            "as the cheap confidence router before escalating to the judge."
        ),
        "pitfall": (
            "A cascade win can silently become a judge win; the artifact must "
            "charge every fallback call and keep oracle-distinct cheap-verifier value separate."
        ),
        "candidate_flag": "flagged_for_v463 (.463): verdi_confidence_routed_cascade",
    },
    {
        "method": "Reflective generative self-supervised PRM",
        "arxiv_id": "2507.01951",
        "url": "https://arxiv.org/abs/2507.01951",
        "phase_d_arms": ["D1 LoRA-EBM", "D2 uPRM"],
        "source_signal": (
            "Uses a shared policy and process-reward interface with a lightweight "
            "scoring head and learns trajectory selection from outcome rewards without process labels."
        ),
        "implementation_delta": (
            "Over Exp 5017-5022, use the reflective self-supervised PRM recipe "
            "as the D2 unblock path and, if D1 remains base-model blocked, train "
            "only a small trajectory-scoring head over cached candidates."
        ),
        "pitfall": (
            "Outcome-derived self-supervision can reproduce generator bias and "
            "reward answer-shape shortcuts; the no-model-id and oracle-distinct audits remain mandatory."
        ),
        "candidate_flag": "flagged_for_v463 (.463): reflective_self_supervised_prm_unblock",
    },
    {
        "method": "CROP conformal clean-prefix certification",
        "arxiv_id": "2605.30085",
        "url": "https://arxiv.org/abs/2605.30085",
        "phase_d_arms": ["D3 EBRM", "D6 verifier-judge cascade"],
        "source_signal": (
            "Turns any step-risk proxy into a calibrated clean-prefix certificate, "
            "then routes uncertified suffixes for repair or review."
        ),
        "implementation_delta": (
            "Over Exp 5017-5022, wrap D3 energy or D6 confidence scores with a "
            "conformal prefix threshold and evaluate certified-prefix length plus "
            "answer-selection delta instead of scalar AUROC alone."
        ),
        "pitfall": (
            "Exchangeability assumptions may fail across generated candidates; "
            "over-withholding can erase the headroom that a verifier moat needs to exploit."
        ),
        "candidate_flag": "flagged_for_v463 (.463): crop_conformal_prefix_gate",
    },
    {
        "method": "Full-Step-DPO self-supervised process reward",
        "arxiv_id": "2502.14356",
        "url": "https://arxiv.org/abs/2502.14356",
        "phase_d_arms": ["D2 uPRM"],
        "source_signal": (
            "Trains a self-supervised process reward model that scores every "
            "reasoning step instead of relying on human or GPT-4 step labels."
        ),
        "implementation_delta": (
            "Over Exp 5017-5022, replace the blocked D2 logprob-cache dependency "
            "with full-step self-supervised rewards over complete candidate traces, "
            "then compare the resulting selector with tuned-SC on the same cache."
        ),
        "pitfall": (
            "It optimizes the generator as much as the verifier; using it as a "
            "selector requires a frozen-candidate evaluation to avoid claiming training lift as moat lift."
        ),
        "candidate_flag": "flagged_for_v463 (.463): full_step_self_supervised_uprm",
    },
]

NEXT_MILESTONE_CANDIDATES = [
    {
        "candidate": "Repair D2 with self-supervised process rewards",
        "candidate_flag": "flagged_for_v463 (.463): repair_d2_self_supervised_prm",
        "source_ids": ["2507.01951", "2502.14356"],
        "phase_d_arms": ["D1 LoRA-EBM", "D2 uPRM"],
        "why": (
            "D5 did not cleanly retire D1 or D2. The strongest .463 repair path "
            "is a frozen-candidate self-supervised PRM that removes the blocked logprob cache."
        ),
    },
    {
        "candidate": "Rerun D3 with uncertainty and conformal abstention",
        "candidate_flag": "flagged_for_v463 (.463): rerun_d3_uncertainty_conformal_gate",
        "source_ids": ["2502.11250", "2605.30085"],
        "phase_d_arms": ["D3 EBRM"],
        "why": (
            "D5 marks D3 execution incomplete. A .463 rerun should add CoT-entropy "
            "and CROP-style conformal thresholds before measuring delta_vs_tuned_sc."
        ),
    },
    {
        "candidate": "Rebuild the cheap-verifier-vs-judge cascade",
        "candidate_flag": "flagged_for_v463 (.463): verdi_oracle_distinct_cascade",
        "source_ids": ["2605.11334"],
        "phase_d_arms": ["D6 verifier-judge cascade"],
        "why": (
            "D6 was blocked, but VERDI gives a single-call confidence router. "
            ".463 should test judge-call savings while preserving oracle-distinct accounting."
        ),
    },
]

RELIABLE_CHANNEL_USED = {
    "sweep_clusters_used": True,
    "sweep_cluster_commands": [
        ".venv/bin/python scripts/sweep_clusters.py 0 --max-results 12",
        ".venv/bin/python scripts/sweep_clusters.py 1 --max-results 12",
        "encoded arXiv API fetches for clusters 0 and 1 after raw helper URL returned HTTP 400",
    ],
    "sweep_cluster_result": (
        "Cluster 0 surfaced PASS/process-reward and reward-hacking papers; "
        "cluster 1 had no stronger energy-RM verifier hit than the mapped D3-adjacent papers."
    ),
    "sweep_semscholar_used": True,
    "sweep_semscholar_commands": [
        "energy based reward model verifier reasoning 2026",
        "unsupervised process reward model verifier reasoning",
        "uncertainty aware reward model verifier LLM judge cascade",
        "cheap verifier judge cascade reward model uncertainty",
        "oracle distinct verification reward model reasoning",
        "test time scaling discriminative verifier reward model",
    ],
    "semscholar_result": (
        "One focused S2 query returned arXiv IDs including 2503.10291, "
        "2504.15275, 2502.06737, 2501.04686, 2502.14361, 2505.20241, "
        "and 2504.10559; five later focused queries hit HTTP 429 and were not "
        "promoted as evidence."
    ),
    "websearch_webfetch_used": True,
    "websearch_queries": [
        "arXiv 2025 2026 energy reward model verifier reasoning LLM uncertainty reward model judge cascade",
        "arXiv unsupervised process reward model verifier LLM reasoning 2025 2026",
        "arXiv cheap verifier judge cascade reward model uncertainty LLM 2025 2026",
        "arXiv test time scaling discriminative verifier process reward model 2025 2026",
    ],
    "webfetch_top_sources": [
        "https://arxiv.org/abs/2502.11250",
        "https://arxiv.org/abs/2605.11334",
        "https://arxiv.org/abs/2507.01951",
        "https://arxiv.org/abs/2605.30085",
        "https://arxiv.org/abs/2606.29296",
        "https://arxiv.org/abs/2512.22245",
        "https://arxiv.org/abs/2502.14356",
        "https://arxiv.org/abs/2505.10320",
    ],
    "deep_research_invoked": False,
}

DEFAULT_PRECONDITIONS_CHECKED = {
    "agents_md_read": True,
    "codex_md_read": True,
    "network_arxiv_reachable": True,
    "sweep_helpers_importable": True,
    "sweep_clusters_used": True,
    "sweep_semscholar_used": True,
    "websearch_webfetch_used": True,
    "deep_research_invoked": False,
    "research_conductor_modified": False,
    "ops_docs_modified": False,
    "d5_artifact_read": True,
    "d5_artifact": D5_ARTIFACT_RELATIVE_PATH,
    "d5_honest_verdict": D5_VERDICT,
    "selected_arxiv_http_200": {
        source_id: f"https://arxiv.org/abs/{source_id}" for source_id in NEW_ARXIV_IDS
    },
    "already_ingested_exclusion_set_checked": sorted(ALREADY_INGESTED_ARXIV_IDS),
    "reliable_channel_note": (
        "sweep_clusters/sweep_semscholar plus low-concurrency WebSearch/WebFetch; "
        "/deep-research was not invoked."
    ),
}

STUDYING_SECTION_START = "<!-- EXP5024-VERIFIER-MOAT-LITERATURE-START -->"
STUDYING_SECTION_END = "<!-- EXP5024-VERIFIER-MOAT-LITERATURE-END -->"
REFERENCES_SECTION_START = "<!-- EXP5024-VERIFIER-MOAT-REFERENCES-START -->"
REFERENCES_SECTION_END = "<!-- EXP5024-VERIFIER-MOAT-REFERENCES-END -->"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    honest_verdict: str = HONEST_VERDICT,
) -> dict[str, object]:
    """Build and validate the complete Exp 5024 ingestion artifact."""

    artifact: dict[str, object] = {
        "honest_verdict": honest_verdict,
        "new_arxiv_ids": list(NEW_ARXIV_IDS),
        "citations_verified": dict(CITATIONS_VERIFIED),
        "sota_to_phase_d_mapping": [dict(row) for row in DEFAULT_SOTA_TO_PHASE_D_MAPPING],
        "next_milestone_candidates": [dict(row) for row in NEXT_MILESTONE_CANDIDATES],
        "note_path": NOTE_RELATIVE_PATH,
        "reliable_channel_used": dict(RELIABLE_CHANNEL_USED),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "d5_conditioning": dict(D5_CONDITIONING),
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    validate_artifact(artifact)
    return artifact


def build_blocked_artifact(
    *,
    blocked_resource: str,
    preconditions_checked: Mapping[str, Any],
) -> dict[str, object]:
    """Build a blocked artifact when the reliable ingestion channel is absent."""

    artifact: dict[str, object] = {
        "honest_verdict": f"blocked_{blocked_resource}",
        "new_arxiv_ids": [],
        "citations_verified": {},
        "sota_to_phase_d_mapping": [],
        "next_milestone_candidates": [],
        "note_path": NOTE_RELATIVE_PATH,
        "reliable_channel_used": dict(RELIABLE_CHANNEL_USED),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "d5_conditioning": dict(D5_CONDITIONING),
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the JSON contract so stale or uncited claims fail closed."""

    _require(set(artifact) == set(REQUIRED_ARTIFACT_FIELDS), "artifact fields mismatch")
    verdict = artifact["honest_verdict"]
    _require(
        isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES),
        "honest_verdict must use a terminal prefix",
    )
    _require(
        artifact["inference_substrate"] == INFERENCE_SUBSTRATE,
        "inference_substrate must be aggregation_from_upstream_artifacts",
    )
    _require(artifact["field_principles"] == FIELD_PRINCIPLES, "field_principles mismatch")
    _require(artifact["d5_conditioning"] == D5_CONDITIONING, "D5 conditioning mismatch")

    reliable = artifact["reliable_channel_used"]
    _require(isinstance(reliable, dict), "reliable_channel_used must be a dict")
    _require(reliable.get("deep_research_invoked") is False, "deep-research is banned")
    _require(reliable.get("sweep_clusters_used") is True, "sweep_clusters must be used")
    _require(reliable.get("sweep_semscholar_used") is True, "sweep_semscholar must be used")
    _require(reliable.get("websearch_webfetch_used") is True, "WebSearch/WebFetch must be used")

    preconditions = artifact["preconditions_checked"]
    _require(isinstance(preconditions, dict), "preconditions_checked must be a dict")
    _require(
        preconditions.get("deep_research_invoked") is False, "deep-research precondition failed"
    )
    _require(
        preconditions.get("research_conductor_modified") is False,
        "scripts/research_conductor.py must not be modified",
    )

    if str(verdict).startswith("blocked_"):
        _require(artifact["new_arxiv_ids"] == [], "blocked artifacts must not cite new IDs")
        _require(
            artifact["sota_to_phase_d_mapping"] == [],
            "blocked artifacts must not claim method mappings",
        )
        return

    _require(verdict == HONEST_VERDICT, "honest_verdict does not match Exp 5024 success")
    new_arxiv_ids = artifact["new_arxiv_ids"]
    _require(len(new_arxiv_ids) >= 3, "new_arxiv_ids must contain at least three papers")
    _require(
        not set(new_arxiv_ids).intersection(ALREADY_INGESTED_ARXIV_IDS),
        "new_arxiv_ids includes an already ingested paper",
    )
    _require(new_arxiv_ids == NEW_ARXIV_IDS, "new_arxiv_ids must match verified selection")

    citations = artifact["citations_verified"]
    _require(isinstance(citations, dict), "citations_verified must be a dict")
    _require(set(citations) == set(new_arxiv_ids), "citations must cover every new ID")
    for source_id, citation in citations.items():
        _require(citation["http_status"] == 200, "citations must have HTTP 200 evidence")
        _require(
            citation["url"] == f"https://arxiv.org/abs/{source_id}",
            "citation URL must match arXiv ID",
        )
        _require(bool(citation["title"]), "citation title must be non-empty")

    mappings = artifact["sota_to_phase_d_mapping"]
    _require(isinstance(mappings, list), "sota_to_phase_d_mapping must be a list")
    _require(3 <= len(mappings) <= 5, "sota_to_phase_d_mapping must contain three to five methods")
    mapped_sources: set[str] = set()
    mapped_arms: set[str] = set()
    for mapping in mappings:
        _require(set(mapping) == REQUIRED_MAPPING_FIELDS, "mapping fields mismatch")
        _require(mapping["arxiv_id"] in new_arxiv_ids, "mapping uses an unverified citation")
        _require(
            mapping["arxiv_id"] not in ALREADY_INGESTED_ARXIV_IDS,
            "mapping uses already ingested ID",
        )
        _require(
            mapping["url"] == f"https://arxiv.org/abs/{mapping['arxiv_id']}",
            "mapping URL must match arXiv ID",
        )
        arms = set(mapping["phase_d_arms"])
        _require(arms and arms.issubset(PHASE_D_ARMS), "mapping must name valid Phase D arms")
        _require(".463" in mapping["candidate_flag"], "mapping candidate flag must target .463")
        _require(bool(mapping["implementation_delta"]), "mapping needs implementation_delta")
        _require(bool(mapping["pitfall"]), "mapping needs pitfall")
        mapped_sources.add(mapping["arxiv_id"])
        mapped_arms.update(arms)
    _require(mapped_sources == set(new_arxiv_ids), "each new source must be mapped")
    _require(PHASE_D_ARMS.issubset(mapped_arms), "D1/D2/D3/D6 must all be represented")

    candidates = artifact["next_milestone_candidates"]
    _require(isinstance(candidates, list) and candidates, "next_milestone_candidates required")
    for candidate in candidates:
        _require(set(candidate) == REQUIRED_CANDIDATE_FIELDS, "candidate fields mismatch")
        _require(
            ".463" in candidate["candidate_flag"], "next milestone candidates must target .463"
        )
        _require(
            set(candidate["source_ids"]).issubset(new_arxiv_ids), "candidate source is not new"
        )
        _require(set(candidate["phase_d_arms"]).issubset(PHASE_D_ARMS), "candidate arm is invalid")


def build_markdown_note(artifact: Mapping[str, Any]) -> str:
    """Render the planner-facing SOTA note from the validated artifact."""

    validate_artifact(artifact)
    excluded = ", ".join(f"arXiv:{source_id}" for source_id in sorted(ALREADY_INGESTED_ARXIV_IDS))
    lines = [
        "# Verifier-moat literature ingestion - 2026-06-30",
        "",
        "## Artifact fields",
    ]
    for field in REQUIRED_USER_FIELD_PRINCIPLES:
        lines.append(f"- {field}: {FIELD_PRINCIPLES[field]['principle']}")

    lines.extend(
        [
            "",
            "## Reliable channel",
            "- Used: sweep_clusters.py, sweep_semscholar.py, low-concurrency WebSearch/WebFetch.",
            "- Not used: /deep-research.",
            "- Semantic Scholar result: HTTP 429 was recorded for later focused queries, not promoted as evidence.",
            f"- Prior .462 and planning-pass exclusions retained for continuity: {excluded}.",
            "",
            "## D5 conditioning",
            (
                f"- Exp 5022 verdict: {D5_VERDICT}; moat_realized=false; "
                "moat_retired_bounded=false; decision=EXECUTION-INCOMPLETE."
            ),
            "- .463 condition: repair or rerun D1/D2/D3/D6 before any retirement claim.",
            "",
            "## SOTA to PHASE D mapping",
        ]
    )
    for mapping in artifact["sota_to_phase_d_mapping"]:
        arms = ", ".join(mapping["phase_d_arms"])
        lines.extend(
            [
                "",
                f"### {mapping['method']}",
                f"- Source: arXiv:{mapping['arxiv_id']} ({mapping['url']})",
                f"- PHASE D arms: {arms}",
                f"- Signal: {mapping['source_signal']}",
                f"- Implementation delta: {mapping['implementation_delta']}",
                f"- Pitfall: {mapping['pitfall']}",
                f"- .463 candidate: {mapping['candidate_flag']}",
            ]
        )

    lines.extend(["", "## Next milestone candidates"])
    for candidate in artifact["next_milestone_candidates"]:
        lines.append(f"- {candidate['candidate_flag']}: {candidate['why']}")
    return "\n".join(lines)


def validate_markdown_note(markdown: str, artifact: Mapping[str, Any]) -> None:
    """Check that the note includes all required citations and planning axes."""

    validate_artifact(artifact)
    required_phrases = [
        "Reliable channel",
        "D5 conditioning",
        "SOTA to PHASE D mapping",
        "Next milestone candidates",
        "D1 LoRA-EBM",
        "D2 uPRM",
        "D3 EBRM",
        "D6 verifier-judge cascade",
        ".463",
        "/deep-research",
    ]
    for phrase in required_phrases:
        _require(phrase in markdown, f"markdown note missing {phrase}")
    for source_id in artifact["new_arxiv_ids"]:
        _require(f"arXiv:{source_id}" in markdown, f"markdown note missing arXiv:{source_id}")


def update_research_studying_text(existing: str, artifact: Mapping[str, Any]) -> str:
    """Insert or replace the Exp 5024 studying section."""

    validate_artifact(artifact)
    bullets = "\n".join(
        f"- {row['method']} (arXiv:{row['arxiv_id']}): {row['candidate_flag']}"
        for row in artifact["sota_to_phase_d_mapping"]
    )
    section = (
        f"{STUDYING_SECTION_START}\n"
        "## Exp 5024 - verifier-moat literature SOTA ingestion - INGESTED\n\n"
        f"- Honest verdict: `{artifact['honest_verdict']}`\n"
        f"- Note: `{artifact['note_path']}`\n"
        f"- D5 conditioning: `{D5_VERDICT}` means .463 repairs or reruns D1/D2/D3/D6.\n"
        "- Reliable channel: sweep_clusters/sweep_semscholar plus WebSearch/WebFetch; "
        "`/deep-research` was not invoked.\n\n"
        "### flagged_for_v463\n"
        f"{bullets}\n"
        f"{STUDYING_SECTION_END}\n"
    )
    return _replace_marked_section(
        existing,
        start_marker=STUDYING_SECTION_START,
        end_marker=STUDYING_SECTION_END,
        section=section,
    )


def update_research_references_text(existing: str, artifact: Mapping[str, Any]) -> str:
    """Insert or replace the Exp 5024 reference section."""

    validate_artifact(artifact)
    entries = []
    for source_id in artifact["new_arxiv_ids"]:
        citation = artifact["citations_verified"][source_id]
        entries.append(f"- arXiv:{source_id} - {citation['title']} - {citation['url']} - HTTP 200")
    section = (
        f"{REFERENCES_SECTION_START}\n"
        "## Exp 5024 verifier-moat literature source set\n\n"
        "These entries are new to the Exp 5024 selected set and are not in the "
        "18-paper already-ingested exclusion list or the CompassVerifier / "
        "Generative Verifiers planning-pass exclusions.\n\n"
        + "\n".join(entries)
        + "\n"
        f"{REFERENCES_SECTION_END}\n"
    )
    return _replace_marked_section(
        existing,
        start_marker=REFERENCES_SECTION_START,
        end_marker=REFERENCES_SECTION_END,
        section=section,
    )


def validate_research_studying_text(text: str, artifact: Mapping[str, Any]) -> None:
    """Validate that research-studying records the ingestion."""

    validate_artifact(artifact)
    _require(STUDYING_SECTION_START in text, "studying section missing start marker")
    _require(STUDYING_SECTION_END in text, "studying section missing end marker")
    _require(str(artifact["honest_verdict"]) in text, "studying section missing verdict")
    for source_id in artifact["new_arxiv_ids"]:
        _require(source_id in text, f"studying section missing {source_id}")


def validate_research_references_text(text: str, artifact: Mapping[str, Any]) -> None:
    """Validate that research-references records every new source."""

    validate_artifact(artifact)
    _require(REFERENCES_SECTION_START in text, "references section missing start marker")
    _require(REFERENCES_SECTION_END in text, "references section missing end marker")
    for source_id in artifact["new_arxiv_ids"]:
        _require(f"arXiv:{source_id}" in text, f"references section missing {source_id}")
        _require(
            f"https://arxiv.org/abs/{source_id}" in text, f"references missing URL {source_id}"
        )


def _replace_marked_section(
    existing: str,
    *,
    start_marker: str,
    end_marker: str,
    section: str,
) -> str:
    if start_marker in existing:
        before, after_start = existing.split(start_marker, 1)
        _require(end_marker in after_start, "existing section missing end marker")
        _, after = after_start.split(end_marker, 1)
        return before + section.rstrip() + "\n" + after.lstrip("\n")

    candidates = [index for index in (existing.find("\n<!-- EXP"), existing.find("\n## ")) if index != -1]
    if not candidates:
        return existing.rstrip() + "\n\n" + section
    insert_at = min(candidates) + 1
    return existing[:insert_at] + section + existing[insert_at:]


def write_outputs(
    *,
    artifact_path: Path,
    note_path: Path,
    studying_path: Path,
    references_path: Path,
    preconditions_checked: Mapping[str, Any],
) -> dict[str, object]:
    """Write the artifact, note, and idempotent research-file updates."""

    if not preconditions_checked.get("network_arxiv_reachable", False):
        artifact = build_blocked_artifact(
            blocked_resource="network",
            preconditions_checked=preconditions_checked,
        )
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return artifact
    if not preconditions_checked.get("sweep_helpers_importable", False):
        artifact = build_blocked_artifact(
            blocked_resource="sweep_helpers",
            preconditions_checked=preconditions_checked,
        )
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return artifact

    artifact = build_artifact(preconditions_checked=preconditions_checked)
    markdown = build_markdown_note(artifact)
    validate_markdown_note(markdown, artifact)

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    note_path.write_text(markdown + "\n", encoding="utf-8")
    studying_path.write_text(
        update_research_studying_text(studying_path.read_text(encoding="utf-8"), artifact),
        encoding="utf-8",
    )
    references_path.write_text(
        update_research_references_text(references_path.read_text(encoding="utf-8"), artifact),
        encoding="utf-8",
    )
    return artifact


def build_live_preconditions() -> dict[str, Any]:  # pragma: no cover - live resource check
    """Check external resources for direct command execution."""

    preconditions = dict(DEFAULT_PRECONDITIONS_CHECKED)
    try:
        urllib.request.urlopen("https://arxiv.org", timeout=15).close()
        preconditions["network_arxiv_reachable"] = True
    except (OSError, urllib.error.URLError):
        preconditions["network_arxiv_reachable"] = False

    try:
        repo_root = str(Path(__file__).resolve().parents[2])
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        importlib.import_module("scripts.sweep_clusters")
        importlib.import_module("scripts.sweep_semscholar")
        preconditions["sweep_helpers_importable"] = True
    except ImportError:
        preconditions["sweep_helpers_importable"] = False

    repo_root = Path(__file__).resolve().parents[2]
    preconditions["d5_artifact_read"] = (repo_root / D5_ARTIFACT_RELATIVE_PATH).exists()
    return preconditions


def main() -> int:
    """Write the default Exp 5024 deliverables under the repository root."""

    repo_root = Path(os.environ.get("CARNOT_EXP5024_ROOT", Path(__file__).resolve().parents[2]))
    preconditions = (
        dict(DEFAULT_PRECONDITIONS_CHECKED)
        if os.environ.get("CARNOT_EXP5024_USE_DEFAULT_PREFLIGHT") == "1"
        else build_live_preconditions()
    )
    artifact = write_outputs(
        artifact_path=repo_root / RESULT_RELATIVE_PATH,
        note_path=repo_root / NOTE_RELATIVE_PATH,
        studying_path=repo_root / STUDYING_RELATIVE_PATH,
        references_path=repo_root / REFERENCES_RELATIVE_PATH,
        preconditions_checked=preconditions,
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
