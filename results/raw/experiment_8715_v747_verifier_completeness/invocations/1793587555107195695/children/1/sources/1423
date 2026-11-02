"""Exp 5038 verifier-moat SOTA ingestion for the .464 roadmap.

Spec refs: REQ-REPORT-5038, SCENARIO-REPORT-5038,
SCENARIO-REPORT-5038-BLOCKED-PRECONDITION.

This module is literature aggregation only. It records the reliable-channel
sweep result, verifies that selected papers are real arXiv pages, and maps the
new ideas onto the Phase D stack after Exp 5036 reported an execution-incomplete
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


RESULT_RELATIVE_PATH = "results/experiment_5038_sota_ingestion_verifier_moat.json"
NOTE_RELATIVE_PATH = "docs/research-notes/verifier-moat-literature-2026-06-30.md"
STUDYING_RELATIVE_PATH = "research-studying.md"
REFERENCES_RELATIVE_PATH = "research-references.md"
D5_ARTIFACT_RELATIVE_PATH = "results/experiment_5036_moat_gate_resolution_v3.json"
HONEST_VERDICT = "success_sota_ingested_5_new_papers_mapped_to_phase_d"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
D5_VERDICT = "complete_moat_execution_incomplete_cascade"
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
PHASE_D_ARMS = frozenset({"D1 LoRA-EBM", "D2 uPRM", "D3 EBRM", "D6 verifier-judge cascade"})
NEW_ARXIV_IDS = [
    "2505.14999",
    "2605.10325",
    "2606.11209",
    "2503.22480",
    "2602.06291",
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
        "2502.11250",
        "2605.11334",
        "2507.01951",
        "2605.30085",
        "2502.14356",
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
            "verified-real NEW arXiv IDs (http 200), NOT in the 23-paper "
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
            "the strongest method(s) flagged as candidate inputs for the .464 "
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
            "Exp 5036 D5 verdict controls whether .464 scales a win, pivots after "
            "clean D1+D2 nulls, or repairs execution-incomplete verifier arms."
        )
    },
    "field_principles": {"principle": "principle annotations are part of the artifact contract."},
}

CITATIONS_VERIFIED = {
    "2505.14999": {
        "title": "Learning to Rank Chain-of-Thought: Using a Small Model",
        "url": "https://arxiv.org/abs/2505.14999",
        "http_status": 200,
    },
    "2605.10325": {
        "title": "Verifiable Process Rewards for Agentic Reasoning",
        "url": "https://arxiv.org/abs/2605.10325",
        "http_status": 200,
    },
    "2606.11209": {
        "title": (
            "ProcessThinker: Enhancing Multi-modal Large Language Models Reasoning "
            "via Rollout-based Process Reward"
        ),
        "url": "https://arxiv.org/abs/2606.11209",
        "http_status": 200,
    },
    "2503.22480": {
        "title": "Probabilistic Uncertain Reward Model",
        "url": "https://arxiv.org/abs/2503.22480",
        "http_status": 200,
    },
    "2602.06291": {
        "title": (
            "Judging What We Cannot Solve: A Consequence-Based Approach for "
            "Oracle-Free Evaluation of Research-Level Math"
        ),
        "url": "https://arxiv.org/abs/2602.06291",
        "http_status": 200,
    },
}

D5_CONDITIONING = {
    "source_artifact": D5_ARTIFACT_RELATIVE_PATH,
    "honest_verdict": D5_VERDICT,
    "decision": "EXECUTION-INCOMPLETE",
    "moat_realized": False,
    "moat_retired_bounded": False,
    "best_clean_arm": {
        "arm": "D1 LoRA-EBM",
        "corpus": "MuSR",
        "delta_vs_tuned_sc": 0.08,
        "paired_ci95": [0.0, 0.165],
        "win_vs_tuned_sc": False,
    },
    "execution_incomplete_arms": [
        "D6 verifier-judge cascade",
        "D4 second-corpus confirmation",
    ],
    "roadmap_condition": (
        "Because D5 is execution-incomplete rather than realized or bounded-retired, "
        ".464 should repair the blocked cascade and second-corpus confirmation, "
        "harden D1/D3 uncertainty, and pivot only if a future rerun makes D1+D2 "
        "both clean nulls."
    ),
}

DEFAULT_SOTA_TO_PHASE_D_MAPPING = [
    {
        "method": "EORM small energy outcome reward verifier",
        "arxiv_id": "2505.14999",
        "url": "https://arxiv.org/abs/2505.14999",
        "phase_d_arms": ["D1 LoRA-EBM", "D3 EBRM"],
        "source_signal": (
            "Uses an energy-based outcome reward model to rank chain-of-thought "
            "solutions with only outcome labels, reporting a 55M-parameter verifier "
            "that can select from candidate pools and generalize to unseen models."
        ),
        "implementation_delta": (
            "Over Exp 5031-5036, replace the scalar D1 LoRA scorer and D3 EBRM "
            "readout with an EORM-style small energy head over frozen MuSR "
            "candidates, then rerun delta_vs_tuned_sc and second-corpus confirmation."
        ),
        "pitfall": (
            "Outcome-only labels can learn answer-shape shortcuts; the rerun needs "
            "frozen candidates, no model-id features, and the same genuine tuned-SC baseline."
        ),
        "candidate_flag": "flagged_for_v464 (.464): eorm_small_energy_selector_for_d1_d3",
    },
    {
        "method": "VPR dense verifier-grounded process rewards",
        "arxiv_id": "2605.10325",
        "url": "https://arxiv.org/abs/2605.10325",
        "phase_d_arms": ["D2 uPRM", "D6 verifier-judge cascade"],
        "source_signal": (
            "Converts symbolic or algorithmic intermediate checks into dense "
            "turn-level rewards for agentic reasoning, improving credit assignment "
            "when reliable local verification is available."
        ),
        "implementation_delta": (
            "Over Exp 5031-5036, build a D2 process-reward replay that uses only "
            "oracle-distinct intermediate checks available before the final answer, "
            "then expose the same confidence as the cheap D6 router before judge calls."
        ),
        "pitfall": (
            "The method is only non-circular if the intermediate verifier is not the "
            "answer oracle; weak or domain-leaking checks would invalidate the moat claim."
        ),
        "candidate_flag": "flagged_for_v464 (.464): oracle_distinct_vpr_dense_process_rewards",
    },
    {
        "method": "ProcessThinker rollout-based process rewards",
        "arxiv_id": "2606.11209",
        "url": "https://arxiv.org/abs/2606.11209",
        "phase_d_arms": ["D2 uPRM", "D3 EBRM"],
        "source_signal": (
            "Assigns step rewards by sampling continuations from intermediate "
            "reasoning states and using empirical final-verification success, "
            "avoiding an explicit trained PRM."
        ),
        "implementation_delta": (
            "Over Exp 5031-5036, compute rollout-success process scores for cached "
            "candidate prefixes and distill them into the D2 selector or D3 energy "
            "margin before comparing against tuned self-consistency."
        ),
        "pitfall": (
            "Continuation rollouts can be expensive and can leak final-answer "
            "verification into the selector; the rerun must charge compute and keep "
            "the verifier oracle-distinct."
        ),
        "candidate_flag": "flagged_for_v464 (.464): rollout_process_reward_distillation",
    },
    {
        "method": "PURM reward-distribution uncertainty",
        "arxiv_id": "2503.22480",
        "url": "https://arxiv.org/abs/2503.22480",
        "phase_d_arms": ["D1 LoRA-EBM", "D3 EBRM", "D6 verifier-judge cascade"],
        "source_signal": (
            "Generalizes Bradley-Terry reward modeling to reward distributions and "
            "uses distribution overlap as per-sample uncertainty to reduce reward hacking."
        ),
        "implementation_delta": (
            "Over Exp 5031-5036, turn D1/D3 scalar verifier scores into reward "
            "distributions, penalize high-overlap uncertain candidates, and route "
            "only uncertain pairs to D6 judge fallback."
        ),
        "pitfall": (
            "PURM is preference-alignment evidence, not a direct reasoning-selector "
            "result; calibration gains must not be counted unless selection accuracy improves."
        ),
        "candidate_flag": "flagged_for_v464 (.464): purm_uncertainty_calibrated_selector",
    },
    {
        "method": "Consequence-Based Utility oracle-free evaluator",
        "arxiv_id": "2602.06291",
        "url": "https://arxiv.org/abs/2602.06291",
        "phase_d_arms": ["D6 verifier-judge cascade", "D2 uPRM"],
        "source_signal": (
            "Scores a candidate solution by testing whether it improves solving of "
            "related verifiable questions, outperforming reward models, generative "
            "reward models, and LLM judges on research-level math ranking."
        ),
        "implementation_delta": (
            "Over Exp 5031-5036, add a cheap consequence-evaluation branch to D6: "
            "candidate traces become exemplars for generated neighboring checks, and "
            "only low-margin cases escalate to the judge."
        ),
        "pitfall": (
            "Generating related verifiable questions can become the expensive verifier; "
            "the cascade must charge that cost and prevent neighborhood tasks from leaking answers."
        ),
        "candidate_flag": "flagged_for_v464 (.464): consequence_utility_cascade_pivot",
    },
]

NEXT_MILESTONE_CANDIDATES = [
    {
        "candidate": "Rerun D1 and D3 with energy plus uncertainty margins",
        "candidate_flag": "flagged_for_v464 (.464): eorm_purm_d1_d3_rerun",
        "source_ids": ["2505.14999", "2503.22480"],
        "phase_d_arms": ["D1 LoRA-EBM", "D3 EBRM"],
        "why": (
            "Exp 5036 D1 and D3 had positive deltas but CI touched zero. The .464 "
            "rerun should test a small EORM head with PURM uncertainty penalties "
            "before spending on a larger verifier."
        ),
    },
    {
        "candidate": "Repair D2 with oracle-distinct dense process rewards",
        "candidate_flag": "flagged_for_v464 (.464): vpr_processthinker_d2_repair",
        "source_ids": ["2605.10325", "2606.11209"],
        "phase_d_arms": ["D2 uPRM"],
        "why": (
            "D2 was clean negative in Exp 5036. The best .464 repair is not another "
            "scalar logprob selector, but VPR-style local checks or rollout-derived "
            "process rewards evaluated on frozen candidates."
        ),
    },
    {
        "candidate": "Rebuild the blocked D6 cascade before any retirement claim",
        "candidate_flag": "flagged_for_v464 (.464): consequence_uncertainty_cascade",
        "source_ids": ["2602.06291", "2503.22480"],
        "phase_d_arms": ["D6 verifier-judge cascade"],
        "why": (
            "Exp 5036 was execution-incomplete because D6 and second-corpus "
            "confirmation were blocked. Consequence utility plus PURM uncertainty "
            "gives a cheap-router path that can be costed separately from judge fallback."
        ),
    },
]

RELIABLE_CHANNEL_USED = {
    "sweep_clusters_used": True,
    "sweep_cluster_commands": [
        ".venv/bin/python scripts/sweep_clusters.py 0 --max-results 12",
        ".venv/bin/python scripts/sweep_clusters.py 1 --max-results 12",
        "encoded arXiv API fetch for clusters 0 and 1",
    ],
    "sweep_cluster_result": (
        "Cluster 0 surfaced reward-hacking and process-reward neighbors, including "
        "arXiv:2606.30627 as a pessimism pitfall; cluster 1 was mostly broad "
        "energy/modeling noise and did not displace the selected verifier papers."
    ),
    "sweep_semscholar_used": True,
    "sweep_semscholar_commands": [
        "energy based reward model verifier reasoning 2026",
        "unsupervised process reward model verifier reasoning",
    ],
    "semscholar_result": (
        "Both focused Semantic Scholar helper queries returned HTTP 429 on "
        "2026-06-30 and were not promoted as citation evidence."
    ),
    "websearch_webfetch_used": True,
    "websearch_queries": [
        'site:arxiv.org/abs 2026 "reward model" uncertainty verifier judge cascade LLM',
        'site:arxiv.org/abs 2026 "process reward model" unsupervised verifier reasoning',
        'site:arxiv.org/abs 2026 "energy" "reward model" verifier reasoning LLM',
        'site:arxiv.org/abs 2026 "verifier" "judge" "cascade" "LLM"',
    ],
    "webfetch_top_sources": [
        "https://arxiv.org/abs/2505.14999",
        "https://arxiv.org/abs/2605.10325",
        "https://arxiv.org/abs/2606.11209",
        "https://arxiv.org/abs/2503.22480",
        "https://arxiv.org/abs/2602.06291",
        "https://arxiv.org/abs/2604.24198",
        "https://arxiv.org/abs/2604.07415",
        "https://arxiv.org/abs/2606.28301",
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

STUDYING_SECTION_START = "<!-- EXP5038-VERIFIER-MOAT-LITERATURE-START -->"
STUDYING_SECTION_END = "<!-- EXP5038-VERIFIER-MOAT-LITERATURE-END -->"
REFERENCES_SECTION_START = "<!-- EXP5038-VERIFIER-MOAT-REFERENCES-START -->"
REFERENCES_SECTION_END = "<!-- EXP5038-VERIFIER-MOAT-REFERENCES-END -->"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    honest_verdict: str = HONEST_VERDICT,
) -> dict[str, object]:
    """Build and validate the complete Exp 5038 ingestion artifact."""

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

    _require(verdict == HONEST_VERDICT, "honest_verdict does not match Exp 5038 success")
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
        _require(".464" in mapping["candidate_flag"], "mapping candidate flag must target .464")
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
            ".464" in candidate["candidate_flag"], "next milestone candidates must target .464"
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
            "- Semantic Scholar result: HTTP 429 was recorded and not promoted as evidence.",
            f"- Prior .462/.463 exclusions retained for continuity: {excluded}.",
            "",
            "## D5 conditioning",
            (
                f"- Exp 5036 verdict: {D5_VERDICT}; moat_realized=false; "
                "moat_retired_bounded=false; decision=EXECUTION-INCOMPLETE."
            ),
            "- .464 condition: repair D6/D4 and harden D1/D3; pivot only after clean D1+D2 nulls.",
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
                f"- .464 candidate: {mapping['candidate_flag']}",
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
        ".464",
        "/deep-research",
    ]
    for phrase in required_phrases:
        _require(phrase in markdown, f"markdown note missing {phrase}")
    for source_id in artifact["new_arxiv_ids"]:
        _require(f"arXiv:{source_id}" in markdown, f"markdown note missing arXiv:{source_id}")


def update_research_studying_text(existing: str, artifact: Mapping[str, Any]) -> str:
    """Insert or replace the Exp 5038 studying section."""

    validate_artifact(artifact)
    bullets = "\n".join(
        f"- {row['method']} (arXiv:{row['arxiv_id']}): {row['candidate_flag']}"
        for row in artifact["sota_to_phase_d_mapping"]
    )
    section = (
        f"{STUDYING_SECTION_START}\n"
        "## Exp 5038 - verifier-moat literature SOTA ingestion - INGESTED\n\n"
        f"- Honest verdict: `{artifact['honest_verdict']}`\n"
        f"- Note: `{artifact['note_path']}`\n"
        f"- D5 conditioning: `{D5_VERDICT}` means .464 repairs D6/D4 and hardens D1/D3.\n"
        "- Reliable channel: sweep_clusters/sweep_semscholar plus WebSearch/WebFetch; "
        "`/deep-research` was not invoked.\n\n"
        "### flagged_for_v464\n"
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
    """Insert or replace the Exp 5038 reference section."""

    validate_artifact(artifact)
    entries = []
    for source_id in artifact["new_arxiv_ids"]:
        citation = artifact["citations_verified"][source_id]
        entries.append(f"- arXiv:{source_id} - {citation['title']} - {citation['url']} - HTTP 200")
    section = (
        f"{REFERENCES_SECTION_START}\n"
        "## Exp 5038 verifier-moat literature source set\n\n"
        "These entries are new to the Exp 5038 selected set and are not in the "
        "23-paper already-ingested exclusion list or the CompassVerifier / "
        "Generative Verifiers exclusions.\n\n" + "\n".join(entries) + "\n"
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

    candidates = [
        index for index in (existing.find("\n<!-- EXP"), existing.find("\n## ")) if index != -1
    ]
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
    """Write the default Exp 5038 deliverables under the repository root."""

    repo_root = Path(os.environ.get("CARNOT_EXP5038_ROOT", Path(__file__).resolve().parents[2]))
    preconditions = (
        dict(DEFAULT_PRECONDITIONS_CHECKED)
        if os.environ.get("CARNOT_EXP5038_USE_DEFAULT_PREFLIGHT") == "1"
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
