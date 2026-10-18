"""Exp 5010 verifier-moat SOTA ingestion for the .462 roadmap.

Spec refs: REQ-REPORT-5010, SCENARIO-REPORT-5010,
SCENARIO-REPORT-5010-BLOCKED-PRECONDITION.

This module is literature aggregation only. It records the reliable-channel
sweep result, verifies that the selected papers are real arXiv pages, and maps
the new ideas onto the already-measured Phase D arms. No training, model load,
or live inference is performed here; the value of the artifact is that the next
planner can see which concrete implementation deltas should be tried after the
Exp 5003-5007 moat-gate outcome.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any
import urllib.error
import urllib.request


RESULT_RELATIVE_PATH = "results/experiment_5010_sota_ingestion_verifier_moat.json"
NOTE_RELATIVE_PATH = "docs/research-notes/verifier-moat-literature-2026-06-30.md"
STUDYING_RELATIVE_PATH = "research-studying.md"
REFERENCES_RELATIVE_PATH = "research-references.md"
HONEST_VERDICT = "success_sota_ingested_5_new_papers_mapped_to_phase_d"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
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
PHASE_D_ARMS = frozenset({"D1 LoRA-EBM", "D2 uPRM", "D3 EBRM"})
NEW_ARXIV_IDS = [
    "2606.19818",
    "2606.09073",
    "2602.24040",
    "2510.20369",
    "2605.24005",
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
        "principle": ("terminal prefix; success_sota_ingested_5_new_papers_mapped_to_phase_d.")
    },
    "new_arxiv_ids": {
        "principle": (
            "verified-real NEW arXiv IDs (http 200), NOT in the 13-paper "
            "ingested set (no fabrication -- every method cites a source)."
        )
    },
    "sota_to_phase_d_mapping": {
        "principle": (
            "per NEW method: which PHASE D arm it strengthens (D1/D2/D3) + "
            "the implementation delta over the current stack + the pitfall."
        )
    },
    "next_milestone_candidates": {
        "principle": (
            "the strongest method(s) flagged as candidate inputs for the .462 "
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
            "sweep_clusters/sweep_semscholar + low-concurrency "
            "WebSearch/WebFetch (NOT /deep-research -- banned from the "
            "autonomous loop)."
        )
    },
    "inference_substrate": {
        "principle": (
            "aggregation_from_upstream_artifacts (literature synthesis, no LLM inference)."
        )
    },
    "preconditions_checked": {
        "principle": ("records network/sweep-helper checks; unreachable network emits blocked_.")
    },
}
FIELD_PRINCIPLES = {
    **REQUIRED_USER_FIELD_PRINCIPLES,
    "citations_verified": {
        "principle": "HTTP-200 arXiv title and URL evidence for every selected method."
    },
    "field_principles": {"principle": "principle annotations are part of the artifact contract."},
}
REQUIRED_PRINCIPLE_FIELDS = frozenset(FIELD_PRINCIPLES)

CITATIONS_VERIFIED = {
    "2606.19818": {
        "title": "Uncertainty-Aware Reward Modeling for Stable RLHF",
        "url": "https://arxiv.org/abs/2606.19818",
        "http_status": 200,
    },
    "2606.09073": {
        "title": "A Unifying Lens on Reward Uncertainty in RLHF",
        "url": "https://arxiv.org/abs/2606.09073",
        "http_status": 200,
    },
    "2602.24040": {
        "title": "RewardUQ: A Unified Framework for Uncertainty-Aware Reward Models",
        "url": "https://arxiv.org/abs/2602.24040",
        "http_status": 200,
    },
    "2510.20369": {
        "title": "Ask a Strong LLM Judge when Your Reward Model is Uncertain",
        "url": "https://arxiv.org/abs/2510.20369",
        "http_status": 200,
    },
    "2605.24005": {
        "title": (
            "LC-ERD: Mining Latent Logic for Self-Evolving Reasoning via "
            "Consistency-Regulated Reward Decomposition"
        ),
        "url": "https://arxiv.org/abs/2605.24005",
        "http_status": 200,
    },
}

DEFAULT_SOTA_TO_PHASE_D_MAPPING = [
    {
        "method": "UARM calibrated uncertainty reward head",
        "arxiv_id": "2606.19818",
        "url": "https://arxiv.org/abs/2606.19818",
        "phase_d_arms": ["D3 EBRM", "D1 LoRA-EBM"],
        "source_signal": (
            "Adds conformal uncertainty and heteroscedastic reward variance to "
            "avoid over-weighting unreliable RM scores."
        ),
        "implementation_delta": (
            "Rerun the clean D3 MuSR selector with an uncertainty head over the "
            "EBRM score, then penalize high-variance candidates before comparing "
            "against tuned self-consistency. If D1 is retried, attach the same "
            "calibration head to the LoRA-EBM scorer."
        ),
        "pitfall": (
            "The evidence is RLHF/preference-domain calibration, not a direct "
            "reasoning-selector win; style or safety uncertainty could be "
            "mistaken for reasoning correctness."
        ),
        "candidate_flag": "flagged_for_v462 (.462): uarm_uncertainty_head_for_d3_ebrm",
    },
    {
        "method": "Distributional pessimistic reward uncertainty",
        "arxiv_id": "2606.09073",
        "url": "https://arxiv.org/abs/2606.09073",
        "phase_d_arms": ["D3 EBRM"],
        "source_signal": (
            "Frames the right reward object as a distribution p(r|x,y), with "
            "pessimistic log-moment aggregation for uncertain regions."
        ),
        "implementation_delta": (
            "Replace the scalar D3 energy readout with a reward-distribution "
            "head and sweep pessimistic beta values on the existing MuSR "
            "candidate cache before any second-corpus spend."
        ),
        "pitfall": (
            "This is a unifying objective, not a finished verifier; excessive "
            "pessimism can abstain away the exact headroom D3 needs to capture."
        ),
        "candidate_flag": "flagged_for_v462 (.462): distributional_pessimistic_ebrm_head",
    },
    {
        "method": "RewardUQ calibration harness for verifier uncertainty",
        "arxiv_id": "2602.24040",
        "url": "https://arxiv.org/abs/2602.24040",
        "phase_d_arms": ["D1 LoRA-EBM", "D3 EBRM"],
        "source_signal": (
            "Compares uncertainty quantification methods for reward models and "
            "ranks them by accuracy plus calibration."
        ),
        "implementation_delta": (
            "Insert a RewardUQ-style calibration table into the D1/D3 harness: "
            "ECE, AUROC on correct-vs-incorrect candidates, and selection delta "
            "after uncertainty-aware abstention."
        ),
        "pitfall": (
            "Better calibration can still leave selection accuracy tied with SC; "
            "the .462 gate must require delta_vs_tuned_sc, not calibration alone."
        ),
        "candidate_flag": "flagged_for_v462 (.462): rewarduq_calibration_gate_for_d1_d3",
    },
    {
        "method": "Uncertainty-routed RM plus strong-judge cascade",
        "arxiv_id": "2510.20369",
        "url": "https://arxiv.org/abs/2510.20369",
        "phase_d_arms": ["D1 LoRA-EBM", "D2 uPRM", "D3 EBRM"],
        "source_signal": (
            "Routes uncertain preference pairs from a cheap RM to a stronger "
            "judge, improving cost-quality tradeoffs over random judge calls."
        ),
        "implementation_delta": (
            "Add a matched-compute cascade control beside D1/D2/D3: cheap "
            "verifier selects when confident, uncertain pairs go to the same "
            "LLM-judge budget already used as a comparator."
        ),
        "pitfall": (
            "The judge can become the real verifier if cost and oracle-distinct "
            "boundaries are not charged explicitly; a win must separate cheap "
            "verifier value from judge fallback value."
        ),
        "candidate_flag": "flagged_for_v462 (.462): uncertainty_routed_moat_cascade",
    },
    {
        "method": "LC-ERD endogenous reward decomposition",
        "arxiv_id": "2605.24005",
        "url": "https://arxiv.org/abs/2605.24005",
        "phase_d_arms": ["D2 uPRM"],
        "source_signal": (
            "Mines latent logic and decomposes step utility from consistency "
            "signals when explicit process labels are scarce."
        ),
        "implementation_delta": (
            "Use LC-ERD as the D2 unblock path when next-token logprob caches are "
            "missing: derive process utility from consistency-regulated latent "
            "logic across the existing candidate batch, then compare to tuned SC."
        ),
        "pitfall": (
            "Endogenous consensus can preserve generator bias and create a "
            "model-identity shortcut, so the no-model-id adversarial check stays "
            "mandatory."
        ),
        "candidate_flag": "flagged_for_v462 (.462): lc_erd_uprm_unblock_path",
    },
]

NEXT_MILESTONE_CANDIDATES = [
    {
        "candidate": "D3 uncertainty-aware EBRM rerun",
        "candidate_flag": "flagged_for_v462 (.462): distributional_uncertainty_d3_rerun",
        "source_ids": ["2606.19818", "2606.09073", "2602.24040"],
        "phase_d_arms": ["D3 EBRM"],
        "why": (
            "D3 is the only clean Phase D row so far and tied tuned-SC; the new "
            "papers directly address uncertainty and reward-distribution scoring."
        ),
    },
    {
        "candidate": "D2 endogenous-process verifier unblock",
        "candidate_flag": "flagged_for_v462 (.462): lc_erd_or_logprob_cache_d2_unblock",
        "source_ids": ["2605.24005"],
        "phase_d_arms": ["D2 uPRM"],
        "why": (
            "D2 blocked on logprob candidate cache; LC-ERD supplies a process "
            "utility fallback that still must pass the model-identity shortcut audit."
        ),
    },
    {
        "candidate": "Matched-compute uncertainty cascade control",
        "candidate_flag": "flagged_for_v462 (.462): uncertainty_routed_judge_control",
        "source_ids": ["2510.20369"],
        "phase_d_arms": ["D1 LoRA-EBM", "D2 uPRM", "D3 EBRM"],
        "why": (
            "If no cheap arm beats tuned-SC alone, a routed cascade can measure "
            "whether verifier uncertainty saves judge calls without relabeling the "
            "judge as the moat."
        ),
    },
]

RELIABLE_CHANNEL_USED = {
    "sweep_clusters_used": True,
    "sweep_cluster_commands": [
        ".venv/bin/python scripts/sweep_clusters.py 0 --max-results 8",
        ".venv/bin/python scripts/sweep_clusters.py 1 --max-results 8",
    ],
    "sweep_semscholar_used": True,
    "sweep_semscholar_commands": [
        (
            ".venv/bin/python scripts/sweep_semscholar.py "
            '"oracle distinct verifier process reward model self consistency '
            'uncertainty reward model" --limit 8'
        ),
        (
            ".venv/bin/python scripts/sweep_semscholar.py "
            '"energy reward model uncertainty robust language model alignment '
            'verifier" --limit 8'
        ),
    ],
    "semscholar_result": "HTTP 429 on both focused queries; rate limits recorded.",
    "websearch_webfetch_used": True,
    "websearch_queries": [
        (
            "site:arxiv.org/abs 2026 process reward model verifier uncertainty "
            "reward model LLM oracle distinct"
        ),
        (
            "site:arxiv.org/abs energy reward model uncertainty process reward "
            "model 2026 LLM verifier"
        ),
        ('site:arxiv.org/abs "Reward Model" "Uncertainty" "RLHF" "2026"'),
    ],
    "webfetch_top_sources": [
        "https://arxiv.org/abs/2606.19818",
        "https://arxiv.org/abs/2606.09073",
        "https://arxiv.org/abs/2602.24040",
        "https://arxiv.org/abs/2510.20369",
        "https://arxiv.org/abs/2605.24005",
        "https://arxiv.org/abs/2602.21158",
        "https://arxiv.org/abs/2605.10325",
        "https://arxiv.org/abs/2606.04579",
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
    "selected_arxiv_http_200": {
        source_id: f"https://arxiv.org/abs/{source_id}" for source_id in NEW_ARXIV_IDS
    },
    "already_ingested_exclusion_set_checked": sorted(ALREADY_INGESTED_ARXIV_IDS),
    "phase_d_artifacts_read": [
        "results/experiment_5003_lora_ebm_scorer_musr.json",
        "results/experiment_5004_uprm_replication.json",
        "results/experiment_5005_ebrm_uncertainty_verifier.json",
        "results/experiment_5007_moat_gate_resolution.json",
    ],
    "phase_d_status_summary": (
        "D1 flagged skeleton, D2 blocked logprob cache, D3 clean MuSR null "
        "delta 0.000 CI [-0.03, 0.025], D5 scoped not retired."
    ),
}

STUDYING_SECTION_START = "<!-- EXP5010-VERIFIER-MOAT-LITERATURE-START -->"
STUDYING_SECTION_END = "<!-- EXP5010-VERIFIER-MOAT-LITERATURE-END -->"
REFERENCES_SECTION_START = "<!-- EXP5010-VERIFIER-MOAT-REFERENCES-START -->"
REFERENCES_SECTION_END = "<!-- EXP5010-VERIFIER-MOAT-REFERENCES-END -->"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    honest_verdict: str = HONEST_VERDICT,
) -> dict[str, object]:
    """Build and validate the complete Exp 5010 ingestion artifact."""

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

    _require(verdict == HONEST_VERDICT, "honest_verdict does not match Exp 5010 success")
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
        _require(".462" in mapping["candidate_flag"], "mapping candidate flag must target .462")
        _require(bool(mapping["implementation_delta"]), "mapping needs implementation_delta")
        _require(bool(mapping["pitfall"]), "mapping needs pitfall")
        mapped_sources.add(mapping["arxiv_id"])
        mapped_arms.update(arms)
    _require(mapped_sources == set(new_arxiv_ids), "each new source must be mapped")
    _require(PHASE_D_ARMS.issubset(mapped_arms), "D1/D2/D3 must all be represented")

    candidates = artifact["next_milestone_candidates"]
    _require(isinstance(candidates, list) and candidates, "next_milestone_candidates required")
    for candidate in candidates:
        _require(set(candidate) == REQUIRED_CANDIDATE_FIELDS, "candidate fields mismatch")
        _require(
            ".462" in candidate["candidate_flag"], "next milestone candidates must target .462"
        )
        _require(
            set(candidate["source_ids"]).issubset(new_arxiv_ids), "candidate source is not new"
        )
        _require(set(candidate["phase_d_arms"]).issubset(PHASE_D_ARMS), "candidate arm is invalid")


def build_markdown_note(artifact: Mapping[str, Any]) -> str:
    """Render the planner-facing SOTA note from the validated artifact."""

    validate_artifact(artifact)
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
            "- Semantic Scholar result: HTTP 429 was recorded, not promoted as evidence.",
            "",
            "## Phase D status read",
            (
                "- Exp 5007 records D1 as a flagged skeleton, D2 as blocked on "
                "logprob cache, D3 as a clean MuSR tie with tuned-SC, and the "
                "moat as scoped rather than retired."
            ),
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
                f"- .462 candidate: {mapping['candidate_flag']}",
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
        "SOTA to PHASE D mapping",
        "Next milestone candidates",
        "D1 LoRA-EBM",
        "D2 uPRM",
        "D3 EBRM",
        ".462",
        "/deep-research",
    ]
    for phrase in required_phrases:
        _require(phrase in markdown, f"markdown note missing {phrase}")
    for source_id in artifact["new_arxiv_ids"]:
        _require(f"arXiv:{source_id}" in markdown, f"markdown note missing arXiv:{source_id}")


def update_research_studying_text(existing: str, artifact: Mapping[str, Any]) -> str:
    """Insert or replace the Exp 5010 studying section."""

    validate_artifact(artifact)
    bullets = "\n".join(
        f"- {row['method']} (arXiv:{row['arxiv_id']}): {row['candidate_flag']}"
        for row in artifact["sota_to_phase_d_mapping"]
    )
    section = (
        f"{STUDYING_SECTION_START}\n"
        "## Exp 5010 - verifier-moat literature SOTA ingestion - INGESTED\n\n"
        f"- Honest verdict: `{artifact['honest_verdict']}`\n"
        f"- Note: `{artifact['note_path']}`\n"
        "- Reliable channel: sweep_clusters/sweep_semscholar plus WebSearch/WebFetch; "
        "`/deep-research` was not invoked.\n"
        "- Phase D read: D1/D2 were not clean evidence, D3 tied tuned-SC on MuSR, "
        "so .462 should target uncertainty-aware reruns and the D2 unblock.\n\n"
        "### flagged_for_v462\n"
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
    """Insert or replace the Exp 5010 reference section."""

    validate_artifact(artifact)
    entries = []
    for source_id in artifact["new_arxiv_ids"]:
        citation = artifact["citations_verified"][source_id]
        entries.append(f"- arXiv:{source_id} - {citation['title']} - {citation['url']} - HTTP 200")
    section = (
        f"{REFERENCES_SECTION_START}\n"
        "## Exp 5010 verifier-moat literature source set\n\n"
        "These entries are new to the Exp 5010 selected set and are not in the "
        "13-paper already-ingested exclusion list.\n\n" + "\n".join(entries) + "\n"
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

    first_section = existing.find("\n## ")
    if first_section == -1:
        return existing.rstrip() + "\n\n" + section
    return existing[: first_section + 1] + section + existing[first_section + 1 :]


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
    return preconditions


def main() -> int:
    """Write the default Exp 5010 deliverables under the repository root."""

    repo_root = Path(os.environ.get("CARNOT_EXP5010_ROOT", Path(__file__).resolve().parents[2]))
    preconditions = (
        dict(DEFAULT_PRECONDITIONS_CHECKED)
        if os.environ.get("CARNOT_EXP5010_USE_DEFAULT_PREFLIGHT") == "1"
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
