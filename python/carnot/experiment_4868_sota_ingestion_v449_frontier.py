"""Exp 4868 SOTA ingestion for the .449 generation-wall frontier.

Spec refs: REQ-ARC-WMTE-4868,
SCENARIO-ARC-WMTE-4868-V449-FRONTIER-MAPPED,
SCENARIO-ARC-WMTE-4868-NO-FABRICATION.

This module writes a deterministic literature-ingestion artifact for the .449
roadmap. Exp 4861's checked-in A1 artifact is blocked and has a null
``fork_verdict`` rather than a measured fork. The operator context reserves the
likely ``INDUCER_CEILING`` branch, so this workflow maps SOTA methods to
world-model inducer quality while preserving the caveat that A1 did not measure
the branch in the committed artifact.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
from typing import Any


JsonMap = Mapping[str, Any]

RESULT_RELATIVE_PATH = "results/experiment_4868_sota_ingestion_v449_frontier.json"
STUDYING_RELATIVE_PATH = "research-studying.md"
REFERENCES_RELATIVE_PATH = "research-references.md"
UPSTREAM_FORK_ARTIFACT = "results/experiment_4861_generation_wall_fork_probe.json"
UPSTREAM_EXPRESSIBILITY_ARTIFACT = "results/experiment_4858_sota_ingestion_generation_expressibility.json"
NOTE_PATH = "research-studying.md#exp-4868-sota-ingestion-v449-frontier"
REFERENCES_PATH = "research-references.md#exp-4868-v449-frontier-source-set"
RANDOM_SEED = 4868
DURATION_S = 0.0001
HONEST_VERDICT = "success_sota_ingestion_v449_frontier_mapped"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
AIMED_AT_FORK_VERDICT = "INDUCER_CEILING"
A1_FORK_VERDICT_READ: str | None = None
REQUESTED_FALLBACK_REASON = (
    "Exp 4861 is blocked_generator_unavailable with fork_verdict=null; "
    "operator context reserved the likely INDUCER_CEILING branch for .449."
)
STUDYING_SECTION_START = "<!-- EXP4868-SOTA-INGESTION-V449-FRONTIER-START -->"
STUDYING_SECTION_END = "<!-- EXP4868-SOTA-INGESTION-V449-FRONTIER-END -->"
REFERENCES_SECTION_START = "<!-- EXP4868-V449-FRONTIER-REFERENCES-START -->"
REFERENCES_SECTION_END = "<!-- EXP4868-V449-FRONTIER-REFERENCES-END -->"
TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)

REQUIRED_USER_FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; mapping emitted is "
            "success_sota_ingestion_v449_frontier_mapped."
        )
    },
    "methods_mapped": {
        "principle": (
            "the strongest 3-5 methods aimed at A1's fork verdict "
            "(inducer-quality OR guided-planning), each with a real arXiv ID."
        )
    },
    "arxiv_ids_cited": {
        "principle": (
            "every method claim must cite a verifiable arXiv ID "
            "(no fabrication -- adversarial_verify bar)."
        )
    },
    "aimed_at_fork_verdict": {
        "principle": (
            "the A1 fork_verdict the ingestion targets (INDUCER_CEILING -> "
            "inducer quality; GUIDANCE/PLANNER -> planning/search)."
        )
    },
    "flagged_for_v449": {
        "principle": "the strongest method(s) flagged so the .449 planner reads the mapping."
    },
    "inference_substrate": {
        "principle": "aggregation_from_upstream_artifacts (0.0001s floor)."
    },
}
FIELD_PRINCIPLES = {
    **REQUIRED_USER_FIELD_PRINCIPLES,
    "preconditions_checked": {
        "principle": "records reliable-channel checks and explicitly excludes banned channels."
    },
    "citations": {
        "principle": "HTTP-200 arXiv source metadata backing every method claim."
    },
    "fresh_sweep": {
        "principle": "records focused sweep_clusters, sweep_semscholar, and WebFetch provenance."
    },
    "upstream_artifacts": {
        "principle": "binds the mapping to Exp 4861 and the Exp 4858 handoff."
    },
    "sota_to_experiment_mapping_note": {
        "principle": "states how each SOTA method becomes a .449 inducer-quality experiment."
    },
    "note_path": {
        "principle": "points to the idempotent research-studying.md ingestion note."
    },
    "references_path": {
        "principle": "points to the idempotent research-references.md source section."
    },
    "random_seed": {
        "principle": "deterministic experiment identifier for reproducible artifact generation."
    },
    "duration_s": {
        "principle": "0.0001s floor for aggregation-only inference substrate."
    },
    "reproducibility_checksum": {
        "principle": "content hash of citations, method map, flags, upstream context, and mapping note."
    },
}
REQUIRED_PRINCIPLE_FIELDS = frozenset(FIELD_PRINCIPLES)
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "methods_mapped",
    "arxiv_ids_cited",
    "aimed_at_fork_verdict",
    "flagged_for_v449",
    "inference_substrate",
    "preconditions_checked",
    "citations",
    "fresh_sweep",
    "upstream_artifacts",
    "sota_to_experiment_mapping_note",
    "note_path",
    "references_path",
    "random_seed",
    "duration_s",
    "reproducibility_checksum",
    "field_principles",
)
REQUIRED_METHOD_FIELDS = frozenset(
    {
        "method",
        "track",
        "source_ids",
        "maps_to_frontier",
        "targets_fork_verdict",
        "fork_mapping",
        "evidence",
        "experiment_graft",
        "validation_gate",
        "sovereignty_note",
        "fails_when",
        "roadmap_candidate",
    }
)
REQUIRED_CITATION_FIELDS = frozenset({"title", "url", "http_status"})
REQUIRED_PRECONDITION_FIELDS = frozenset(
    {
        "agents_md_read",
        "codex_md_read",
        "research_studying_present",
        "research_references_present",
        "upstream_fork_artifact_present",
        "upstream_expressibility_artifact_present",
        "a1_fork_verdict_read",
        "a1_fork_verdict",
        "aimed_at_fork_verdict",
        "requested_fallback_reason",
        "sweep_clusters_used",
        "sweep_cluster_ids",
        "sweep_cluster_urls",
        "sweep_semscholar_used",
        "sweep_semscholar_queries",
        "sweep_semscholar_result",
        "semantic_scholar_unique_arxiv_ids",
        "websearch_webfetch_used",
        "websearch_webfetch_top_sources",
        "top_source_count",
        "arxiv_http_200_verified_ids",
        "deep_research_invoked",
        "retired_coverage_classes_reingested",
        "exploration_strategy_reingested",
        "energy_classes_reingested",
        "model_load",
        "training_launched",
        "leaderboard_submission",
        "solve_claim_made",
        "research_conductor_modified",
        "ops_docs_modified",
    }
)
REQUIRED_FRESH_SWEEP_FIELDS = frozenset(
    {
        "filtered_track",
        "cluster_ids",
        "semantic_scholar_queries",
        "semantic_scholar_result",
        "semantic_scholar_unique_arxiv_ids",
        "webfetch_top_sources",
    }
)
REQUIRED_UPSTREAM_FIELDS = frozenset(
    {
        "fork_probe_artifact",
        "expressibility_handoff",
        "a1_honest_verdict",
        "a1_fork_verdict",
        "aimed_at_fork_verdict",
        "blocked_a1_recorded_honestly",
        "requested_fallback_reason",
        "carried_forward_from_v447",
        "retired_classes_kept_closed",
    }
)
REQUIRED_MAPPING_NOTE_FIELDS = frozenset(
    {
        "summary",
        "terminal_success",
        "source_ids",
        "root_cause",
        "planner_instruction",
        "a1_caveat",
    }
)
REQUIRED_SOURCE_IDS = frozenset(
    {
        "2203.13474",
        "2502.07786",
        "2506.02918",
        "2507.03160",
        "2507.15877",
        "2509.03956",
        "2605.05138",
        "2606.11521",
    }
)
REQUIRED_TRACKS = frozenset(
    {
        "family_b_executable_world_model_inducer",
        "test_time_world_model_adaptation",
        "counterexample_guided_inducer_refinement",
        "local_open_code_inducer",
    }
)
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(f"arXiv:{source_id}" for source_id in REQUIRED_SOURCE_IDS)

CLUSTER_5_URL = (
    'http://export.arxiv.org/api/query?search_query=(abs:"affordance"+OR+'
    'abs:"action+effect"+OR+abs:"clickability"+OR+abs:"frame+prediction"+OR+'
    'abs:"intrinsic+motivation"+OR+abs:"directed+exploration"+OR+'
    'abs:"novelty+search")+AND+(abs:"reinforcement+learning"+OR+abs:"agent"+OR+'
    'abs:"exploration"+OR+abs:"interactive+environment"+OR+abs:"ARC")&start=0&'
    "max_results=8&sortBy=submittedDate&sortOrder=descending"
)
CLUSTER_6_URL = (
    'http://export.arxiv.org/api/query?search_query=(abs:"neural+guided+search"+OR+'
    'abs:"learned+heuristic"+OR+abs:"value+guided+search"+OR+'
    'abs:"program+induction"+OR+abs:"world+model"+OR+abs:"goal+induction")+'
    'AND+(abs:"planning"+OR+abs:"agent"+OR+abs:"reasoning"+OR+'
    'abs:"reinforcement+learning")&start=0&max_results=8&sortBy=submittedDate&'
    "sortOrder=descending"
)
SEMANTIC_SCHOLAR_QUERIES = [
    "executable world models ARC AGI coding agents",
    "test time training world model dynamics planning",
    "counterexample guided program synthesis repair coding agents",
    "small coding model program synthesis local open source",
]
SEMANTIC_SCHOLAR_UNIQUE_ARXIV_IDS = [
    "2011.11751",
    "2409.19142",
    "2506.02918",
    "2506.16565",
]
SEMANTIC_SCHOLAR_RESULT = (
    "Four focused queries: first, third, and fourth returned HTTP 429; "
    "the test-time dynamics query returned 4 unique arxiv IDs."
)
WEBSEARCH_WEBFETCH_TOP_SOURCES = [
    "https://arxiv.org/abs/2203.13474",
    "https://arxiv.org/abs/2502.07786",
    "https://arxiv.org/abs/2506.02918",
    "https://arxiv.org/abs/2507.03160",
    "https://arxiv.org/abs/2507.15877",
    "https://arxiv.org/abs/2509.03956",
    "https://arxiv.org/abs/2605.05138",
    "https://arxiv.org/abs/2606.11521",
]

CITATIONS = {
    "2203.13474": {
        "title": "CodeGen: An Open Large Language Model for Code with Multi-Turn Program Synthesis",
        "url": "https://arxiv.org/abs/2203.13474",
        "http_status": 200,
    },
    "2502.07786": {
        "title": "Counterexample Guided Program Repair Using Zero-Shot Learning and MaxSAT-based Fault Localization",
        "url": "https://arxiv.org/abs/2502.07786",
        "http_status": 200,
    },
    "2506.02918": {
        "title": "World Modelling Improves Language Model Agents",
        "url": "https://arxiv.org/abs/2506.02918",
        "http_status": 200,
    },
    "2507.03160": {
        "title": "Assessing Small Language Models for Code Generation: An Empirical Study with Benchmarks",
        "url": "https://arxiv.org/abs/2507.03160",
        "http_status": 200,
    },
    "2507.15877": {
        "title": (
            "Out-of-Distribution Generalization in the ARC-AGI Domain: Comparing "
            "Execution-Guided Neural Program Synthesis and Test-Time Fine-Tuning"
        ),
        "url": "https://arxiv.org/abs/2507.15877",
        "http_status": 200,
    },
    "2509.03956": {
        "title": "World Model Implanting for Test-time Adaptation of Embodied Agents",
        "url": "https://arxiv.org/abs/2509.03956",
        "http_status": 200,
    },
    "2605.05138": {
        "title": "Executable World Models for ARC-AGI-3 in the Era of Coding Agents",
        "url": "https://arxiv.org/abs/2605.05138",
        "http_status": 200,
    },
    "2606.11521": {
        "title": "Counterexample Guided Learning in the Large using Reasoning Agents",
        "url": "https://arxiv.org/abs/2606.11521",
        "http_status": 200,
    },
}

FLAGGED_FOR_V449 = [
    {
        "candidate": "family_b_executable_world_model_inducer_ladder",
        "flag": (
            "flagged_for_v449: family_b_executable_world_model_inducer_ladder "
            "(arXiv:2605.05138 + arXiv:2507.03160 + arXiv:2203.13474)"
        ),
        "source_ids": ["2605.05138", "2507.03160", "2203.13474"],
        "maps_to_frontier": ".449",
    },
    {
        "candidate": "test_time_world_model_adaptation_loop",
        "flag": (
            "flagged_for_v449: test_time_world_model_adaptation_loop "
            "(arXiv:2506.02918 + arXiv:2509.03956 + arXiv:2507.15877)"
        ),
        "source_ids": ["2506.02918", "2509.03956", "2507.15877"],
        "maps_to_frontier": ".449",
    },
    {
        "candidate": "cegis_world_model_refinement_loop",
        "flag": (
            "flagged_for_v449: cegis_world_model_refinement_loop "
            "(arXiv:2606.11521 + arXiv:2502.07786 + arXiv:2507.15877)"
        ),
        "source_ids": ["2606.11521", "2502.07786", "2507.15877"],
        "maps_to_frontier": ".449",
    },
]

DEFAULT_METHODS_MAPPED = [
    {
        "method": "Family-B executable world-model inducer quality ladder",
        "track": "family_b_executable_world_model_inducer",
        "source_ids": ["2605.05138", "2507.03160", "2203.13474"],
        "maps_to_frontier": ".449",
        "targets_fork_verdict": AIMED_AT_FORK_VERDICT,
        "fork_mapping": (
            "INDUCER_CEILING means the executable model is inaccurate before planning; "
            "compare the strong Family-B coding-agent inducer against local small/open "
            "code inducers under the same held-out transition gate."
        ),
        "evidence": (
            "arXiv:2605.05138 reports verifier-driven executable Python world models "
            "with strong coding agents; arXiv:2507.03160 and arXiv:2203.13474 bound "
            "what local/open code models can plausibly supply."
        ),
        "experiment_graft": (
            "Build a two-lane inducer harness: cloud-strength Family-B reference lane "
            "for ceiling measurement, local open-code lane for sovereign deployment, "
            "both emitting the same executable engine interface."
        ),
        "validation_gate": (
            "Pass only if held-out off-path transition accuracy improves before any "
            "planner reranking; otherwise .449 retires the inducer upgrade."
        ),
        "sovereignty_note": (
            "The cloud lane is a measurement oracle for capability, not the desired "
            "deployment path; the local lane preserves air-gapped operation."
        ),
        "fails_when": (
            "The strong inducer still overfits observed prefixes, or the local open "
            "inducer cannot approach the cloud reference without forbidden network access."
        ),
        "roadmap_candidate": FLAGGED_FOR_V449[0]["flag"],
    },
    {
        "method": "Test-time world-model and dynamics adaptation loop",
        "track": "test_time_world_model_adaptation",
        "source_ids": ["2506.02918", "2509.03956", "2507.15877"],
        "maps_to_frontier": ".449",
        "targets_fork_verdict": AIMED_AT_FORK_VERDICT,
        "fork_mapping": (
            "INDUCER_CEILING can be attacked by adapting the dynamics model at test "
            "time from observed transitions before planning through it."
        ),
        "evidence": (
            "arXiv:2506.02918 adds internal state prediction to language agents, "
            "arXiv:2509.03956 composes world models at test time, and "
            "arXiv:2507.15877 frames ARC test-time fine-tuning versus "
            "execution-guided synthesis."
        ),
        "experiment_graft": (
            "After cold-start transition collection, fit or select a small dynamics "
            "adapter, then rerun the held-out transition score before plan_in_model."
        ),
        "validation_gate": (
            "Only count improvements that raise held-out transition accuracy on games "
            "not used for the adapter's observed-prefix fit."
        ),
        "sovereignty_note": (
            "The adapter can be trained or selected locally from game observations, "
            "which keeps the improvement air-gapped."
        ),
        "fails_when": (
            "The adapter memorizes prefix frames, loses hidden state, or improves "
            "in-distribution replay without raising held-out dynamics accuracy."
        ),
        "roadmap_candidate": FLAGGED_FOR_V449[1]["flag"],
    },
    {
        "method": "Counterexample-guided executable world-model refinement",
        "track": "counterexample_guided_inducer_refinement",
        "source_ids": ["2606.11521", "2502.07786", "2507.15877"],
        "maps_to_frontier": ".449",
        "targets_fork_verdict": AIMED_AT_FORK_VERDICT,
        "fork_mapping": (
            "INDUCER_CEILING becomes a refinement loop: failed held-out transitions "
            "become counterexamples that revise the executable engine instead of "
            "merely rejecting it."
        ),
        "evidence": (
            "arXiv:2606.11521 shows counterexamples can improve LLM symbolic "
            "induction; arXiv:2502.07786 uses CEGIS-style LLM repair; "
            "arXiv:2507.15877 supports execution-guided ARC synthesis."
        ),
        "experiment_graft": (
            "Wrap the engine verifier in a CEGIS loop that converts off-path mismatch "
            "rows into minimal failing transition tests and asks the inducer to repair "
            "the engine."
        ),
        "validation_gate": (
            "Accept a refined engine only when the repair fixes held-out counterexamples "
            "without regressing observed-prefix replay."
        ),
        "sovereignty_note": (
            "Counterexamples are produced by the local executable verifier, so even a "
            "small local inducer receives precise feedback without cloud traces."
        ),
        "fails_when": (
            "Counterexamples are too sparse, repairs overfit the latest failing row, "
            "or the executable representation cannot express the hidden mechanic."
        ),
        "roadmap_candidate": FLAGGED_FOR_V449[2]["flag"],
    },
    {
        "method": "Local open-code inducer distillation and self-correction",
        "track": "local_open_code_inducer",
        "source_ids": ["2507.03160", "2203.13474", "2502.07786"],
        "maps_to_frontier": ".449",
        "targets_fork_verdict": AIMED_AT_FORK_VERDICT,
        "fork_mapping": (
            "INDUCER_CEILING requires a stronger air-gapped inducer, so the local lane "
            "should use open code-model selection plus verifier feedback rather than "
            "another generic prompt."
        ),
        "evidence": (
            "arXiv:2507.03160 evaluates compact open code models, arXiv:2203.13474 "
            "establishes open multi-turn program synthesis, and arXiv:2502.07786 "
            "shows verifier feedback can improve LLM repair."
        ),
        "experiment_graft": (
            "Benchmark candidate local code models on executable-engine synthesis, then "
            "distill the successful prompting and repair traces into the chosen local "
            "inducer lane."
        ),
        "validation_gate": (
            "Promote a local inducer only if it beats the current Qwen3.5-9B-MTP "
            "engine accuracy under the same A1 held-out-game set."
        ),
        "sovereignty_note": (
            "This is the deployment candidate: all inference and refinement stays on "
            "local hardware after the cloud reference has measured the ceiling."
        ),
        "fails_when": (
            "The best local model cannot synthesize executable state updates, or "
            "self-correction loops repeatedly repair syntax while dynamics remain wrong."
        ),
        "roadmap_candidate": FLAGGED_FOR_V449[0]["flag"],
    },
]

DEFAULT_FRESH_SWEEP = {
    "filtered_track": (
        "V449 frontier for INDUCER_CEILING: executable world-model induction, "
        "test-time dynamics adaptation, counterexample-guided repair, and local "
        "open code-model inducers"
    ),
    "cluster_ids": [5, 6],
    "semantic_scholar_queries": SEMANTIC_SCHOLAR_QUERIES,
    "semantic_scholar_result": SEMANTIC_SCHOLAR_RESULT,
    "semantic_scholar_unique_arxiv_ids": SEMANTIC_SCHOLAR_UNIQUE_ARXIV_IDS,
    "webfetch_top_sources": WEBSEARCH_WEBFETCH_TOP_SOURCES,
}
DEFAULT_PRECONDITIONS_CHECKED = {
    "agents_md_read": True,
    "codex_md_read": True,
    "research_studying_present": True,
    "research_references_present": True,
    "upstream_fork_artifact_present": True,
    "upstream_expressibility_artifact_present": True,
    "a1_fork_verdict_read": True,
    "a1_fork_verdict": A1_FORK_VERDICT_READ,
    "aimed_at_fork_verdict": AIMED_AT_FORK_VERDICT,
    "requested_fallback_reason": REQUESTED_FALLBACK_REASON,
    "sweep_clusters_used": True,
    "sweep_cluster_ids": [5, 6],
    "sweep_cluster_urls": [CLUSTER_5_URL, CLUSTER_6_URL],
    "sweep_semscholar_used": True,
    "sweep_semscholar_queries": SEMANTIC_SCHOLAR_QUERIES,
    "sweep_semscholar_result": SEMANTIC_SCHOLAR_RESULT,
    "semantic_scholar_unique_arxiv_ids": SEMANTIC_SCHOLAR_UNIQUE_ARXIV_IDS,
    "websearch_webfetch_used": True,
    "websearch_webfetch_top_sources": WEBSEARCH_WEBFETCH_TOP_SOURCES,
    "top_source_count": len(WEBSEARCH_WEBFETCH_TOP_SOURCES),
    "arxiv_http_200_verified_ids": [
        f"https://arxiv.org/abs/{source_id}" for source_id in sorted(REQUIRED_SOURCE_IDS)
    ],
    "deep_research_invoked": False,
    "retired_coverage_classes_reingested": False,
    "exploration_strategy_reingested": False,
    "energy_classes_reingested": False,
    "model_load": False,
    "training_launched": False,
    "leaderboard_submission": False,
    "solve_claim_made": False,
    "research_conductor_modified": False,
    "ops_docs_modified": False,
}
DEFAULT_UPSTREAM_ARTIFACTS = {
    "fork_probe_artifact": UPSTREAM_FORK_ARTIFACT,
    "expressibility_handoff": UPSTREAM_EXPRESSIBILITY_ARTIFACT,
    "a1_honest_verdict": "blocked_generator_unavailable",
    "a1_fork_verdict": A1_FORK_VERDICT_READ,
    "aimed_at_fork_verdict": AIMED_AT_FORK_VERDICT,
    "blocked_a1_recorded_honestly": True,
    "requested_fallback_reason": REQUESTED_FALLBACK_REASON,
    "carried_forward_from_v447": [
        "executable_world_model_action_programmer",
        "execution_guided_neural_program_synthesis",
        "family_b_executable_world_model_reference",
    ],
    "retired_classes_kept_closed": [
        "macro-vocab",
        "click-heatmap",
        "exploration-strategy",
        "energy classes",
    ],
}
DEFAULT_MAPPING_NOTE = {
    "summary": (
        ".449 should target INDUCER_CEILING: improve the executable world-model "
        "inducer before adding planner/search complexity."
    ),
    "terminal_success": HONEST_VERDICT,
    "source_ids": sorted(REQUIRED_SOURCE_IDS),
    "root_cause": "world-model inducer quality",
    "planner_instruction": (
        "Stage a Family-B reference inducer, test-time dynamics adaptation, and "
        "counterexample-guided refinement; promote the local open inducer only when "
        "held-out transition accuracy improves."
    ),
    "a1_caveat": (
        "The committed Exp 4861 A1 artifact is blocked/null, so this maps the "
        "operator-reserved likely INDUCER_CEILING branch without claiming A1 measured it."
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def source_set_checksum(
    citations: JsonMap,
    methods: Sequence[JsonMap],
    flags: Sequence[JsonMap],
    upstream_artifacts: JsonMap,
    mapping_note: JsonMap,
) -> str:
    payload = json.dumps(
        {
            "citations": citations,
            "flags": list(flags),
            "mapping_note": mapping_note,
            "methods": list(methods),
            "upstream_artifacts": upstream_artifacts,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


REPRODUCIBILITY_CHECKSUM = source_set_checksum(
    CITATIONS,
    DEFAULT_METHODS_MAPPED,
    FLAGGED_FOR_V449,
    DEFAULT_UPSTREAM_ARTIFACTS,
    DEFAULT_MAPPING_NOTE,
)


def build_artifact(
    *,
    methods_mapped: Sequence[JsonMap] = DEFAULT_METHODS_MAPPED,
    citations: JsonMap = CITATIONS,
    preconditions_checked: JsonMap = DEFAULT_PRECONDITIONS_CHECKED,
    fresh_sweep: JsonMap = DEFAULT_FRESH_SWEEP,
    flagged_for_v449: Sequence[JsonMap] = FLAGGED_FOR_V449,
    upstream_artifacts: JsonMap = DEFAULT_UPSTREAM_ARTIFACTS,
    sota_to_experiment_mapping_note: JsonMap = DEFAULT_MAPPING_NOTE,
    honest_verdict: str = HONEST_VERDICT,
) -> dict[str, object]:
    artifact: dict[str, object] = {
        "honest_verdict": honest_verdict,
        "methods_mapped": [dict(method) for method in methods_mapped],
        "arxiv_ids_cited": sorted(citations),
        "aimed_at_fork_verdict": AIMED_AT_FORK_VERDICT,
        "flagged_for_v449": [dict(flag) for flag in flagged_for_v449],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "citations": {source_id: dict(citation) for source_id, citation in citations.items()},
        "fresh_sweep": dict(fresh_sweep),
        "upstream_artifacts": dict(upstream_artifacts),
        "sota_to_experiment_mapping_note": dict(sota_to_experiment_mapping_note),
        "note_path": NOTE_PATH,
        "references_path": REFERENCES_PATH,
        "random_seed": RANDOM_SEED,
        "duration_s": DURATION_S,
        "reproducibility_checksum": source_set_checksum(
            citations,
            methods_mapped,
            flagged_for_v449,
            upstream_artifacts,
            sota_to_experiment_mapping_note,
        ),
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: JsonMap) -> None:
    missing = set(REQUIRED_ARTIFACT_FIELDS).difference(artifact)
    extra = set(artifact).difference(REQUIRED_ARTIFACT_FIELDS)
    _require(not missing, f"artifact missing required fields: {sorted(missing)}")
    _require(not extra, f"artifact has unexpected fields: {sorted(extra)}")
    _require(isinstance(artifact["honest_verdict"], str), "honest_verdict must be a string")
    _require(
        artifact["honest_verdict"].startswith(TERMINAL_PREFIXES),
        "honest_verdict must use a terminal prefix",
    )
    _require(artifact["honest_verdict"] == HONEST_VERDICT, f"honest_verdict must equal {HONEST_VERDICT!r}")
    _require(
        artifact["aimed_at_fork_verdict"] == AIMED_AT_FORK_VERDICT,
        "aimed_at_fork_verdict must be INDUCER_CEILING",
    )
    _require(artifact["inference_substrate"] == INFERENCE_SUBSTRATE, "inference_substrate must be aggregation-only")
    _require(artifact["field_principles"] == FIELD_PRINCIPLES, "field_principles must match annotations")
    _require(artifact["note_path"] == NOTE_PATH, "note_path must point at the Exp 4868 note")
    _require(artifact["references_path"] == REFERENCES_PATH, "references_path must point at Exp 4868 references")
    _require(artifact["random_seed"] == RANDOM_SEED, "random_seed must be the experiment id")
    _require(artifact["duration_s"] == DURATION_S, "duration_s must preserve the 0.0001s floor")
    _validate_citations(artifact["citations"], artifact["arxiv_ids_cited"])
    _validate_methods(artifact["methods_mapped"], artifact["arxiv_ids_cited"])
    _validate_flags(artifact["flagged_for_v449"], artifact["arxiv_ids_cited"])
    _validate_preconditions(artifact["preconditions_checked"])
    _validate_fresh_sweep(artifact["fresh_sweep"])
    _validate_upstream_artifacts(artifact["upstream_artifacts"])
    _validate_mapping_note(artifact["sota_to_experiment_mapping_note"], artifact["arxiv_ids_cited"])
    _require(
        artifact["reproducibility_checksum"]
        == source_set_checksum(
            artifact["citations"],
            artifact["methods_mapped"],
            artifact["flagged_for_v449"],
            artifact["upstream_artifacts"],
            artifact["sota_to_experiment_mapping_note"],
        ),
        "reproducibility checksum must hash citations, methods, flags, upstream context, and mapping note",
    )


def _validate_citations(citations: object, arxiv_ids_cited: object) -> None:
    _require(isinstance(citations, Mapping), "citations must be a mapping")
    _require(set(citations) == REQUIRED_SOURCE_IDS, "citations must exactly cover verified source IDs")
    _require(arxiv_ids_cited == sorted(REQUIRED_SOURCE_IDS), "arxiv_ids_cited must list every verified arXiv ID")
    for source_id, citation in citations.items():
        _require(isinstance(citation, Mapping), "each citation must be a mapping")
        _require(set(citation) == REQUIRED_CITATION_FIELDS, "each citation must contain title, url, http_status")
        _require(citation["url"] == f"https://arxiv.org/abs/{source_id}", "citation url must match arXiv ID")
        _require(citation["http_status"] == 200, "each citation http_status must be 200")
        _require(bool(citation["title"]), "each citation title must be non-empty")


def _validate_methods(methods: object, arxiv_ids_cited: object) -> None:
    _require(isinstance(methods, Sequence) and not isinstance(methods, str | bytes), "methods_mapped must be a list")
    _require(3 <= len(methods) <= 5, "methods_mapped must contain three to five methods")
    cited = set(arxiv_ids_cited)
    tracks: set[str] = set()
    for method in methods:
        _require(isinstance(method, Mapping), "each method must be a mapping")
        _require(set(method) == REQUIRED_METHOD_FIELDS, "each method must match the required method schema")
        source_ids = method["source_ids"]
        _require(
            isinstance(source_ids, Sequence) and not isinstance(source_ids, str | bytes) and bool(source_ids),
            "each method must cite source_ids",
        )
        _require(set(source_ids).issubset(cited), "method source_ids must be verified citations")
        _require(method["maps_to_frontier"] == ".449", "method must map to the .449 frontier")
        _require(method["targets_fork_verdict"] == AIMED_AT_FORK_VERDICT, "method must target fork verdict")
        _require("inducer" in str(method["fork_mapping"]).lower(), "method must map to inducer quality")
        _require("arXiv:" in str(method["evidence"]), "method evidence must cite arXiv IDs")
        _require(bool(method["experiment_graft"]), "each method needs an experiment graft")
        _require(bool(method["validation_gate"]), "each method needs a validation gate")
        _require(bool(method["sovereignty_note"]), "each method needs a sovereignty note")
        _require(bool(method["fails_when"]), "each method needs fails_when")
        _require("flagged_for_v449" in str(method["roadmap_candidate"]), "each method needs a .449 roadmap candidate")
        tracks.add(str(method["track"]))
    _require(REQUIRED_TRACKS == tracks, "methods_mapped missing required .449 inducer tracks")


def _validate_flags(flags: object, arxiv_ids_cited: object) -> None:
    _require(
        isinstance(flags, Sequence) and not isinstance(flags, str | bytes) and bool(flags),
        "flagged_for_v449 required",
    )
    cited = set(arxiv_ids_cited)
    for flag in flags:
        _require(isinstance(flag, Mapping), "each flagged_for_v449 entry must be a mapping")
        _require("candidate" in flag and "flag" in flag, "each flagged_for_v449 entry needs candidate and flag")
        _require("flagged_for_v448" not in json.dumps(flag, sort_keys=True), "stale .448 flag found in flagged_for_v449")
        _require("flagged_for_v449" in str(flag["flag"]), "flagged_for_v449 entries must carry the .449 flag")
        _require(set(flag.get("source_ids", [])).issubset(cited), "flagged_for_v449 source_ids must be verified")


def _validate_preconditions(preconditions: object) -> None:
    _require(isinstance(preconditions, Mapping), "preconditions_checked must be a mapping")
    _require(set(preconditions) == REQUIRED_PRECONDITION_FIELDS, "preconditions_checked must match schema")
    _require(preconditions["research_studying_present"] is True, "research-studying precondition must pass")
    _require(preconditions["research_references_present"] is True, "research-references precondition must pass")
    _require(preconditions["upstream_fork_artifact_present"] is True, "upstream Exp 4861 artifact must be present")
    _require(
        preconditions["upstream_expressibility_artifact_present"] is True,
        "upstream Exp 4858 artifact must be present",
    )
    _require(preconditions["a1_fork_verdict_read"] is True, "A1 fork verdict must be read")
    _require(preconditions["a1_fork_verdict"] is None, "blocked/null A1 fork verdict must be recorded honestly")
    _require(preconditions["aimed_at_fork_verdict"] == AIMED_AT_FORK_VERDICT, "precondition fork target mismatch")
    _require(preconditions["requested_fallback_reason"] == REQUESTED_FALLBACK_REASON, "fallback reason mismatch")
    _require(preconditions["sweep_clusters_used"] is True, "sweep_clusters must be used")
    _require(preconditions["sweep_cluster_ids"] == [5, 6], "sweep cluster IDs must be [5, 6]")
    _require(preconditions["sweep_semscholar_used"] is True, "sweep_semscholar must be used")
    _require(
        preconditions["semantic_scholar_unique_arxiv_ids"] == SEMANTIC_SCHOLAR_UNIQUE_ARXIV_IDS,
        "Semantic Scholar IDs must match reliable-channel output",
    )
    _require(preconditions["websearch_webfetch_used"] is True, "WebSearch/WebFetch must be used")
    _require(5 <= int(preconditions["top_source_count"]) <= 8, "top_source_count must record top five to eight sources")
    _require(preconditions["deep_research_invoked"] is False, "deep-research must not be invoked")
    _require(preconditions["retired_coverage_classes_reingested"] is False, "retired coverage must not be reingested")
    _require(preconditions["exploration_strategy_reingested"] is False, "exploration strategy must not be reingested")
    _require(preconditions["energy_classes_reingested"] is False, "energy classes must not be reingested")
    _require(preconditions["model_load"] is False, "model load must not occur")
    _require(preconditions["training_launched"] is False, "training must not be launched")
    _require(preconditions["leaderboard_submission"] is False, "leaderboard submission must not occur")
    _require(preconditions["solve_claim_made"] is False, "solve claim must remain false")
    _require(preconditions["research_conductor_modified"] is False, "research_conductor must not be modified")
    _require(preconditions["ops_docs_modified"] is False, "ops docs must not be modified by this workflow")


def _validate_fresh_sweep(fresh_sweep: object) -> None:
    _require(isinstance(fresh_sweep, Mapping), "fresh_sweep must be a mapping")
    _require(set(fresh_sweep) == REQUIRED_FRESH_SWEEP_FIELDS, "fresh_sweep must match schema")
    _require(fresh_sweep["cluster_ids"] == [5, 6], "fresh_sweep must record clusters 5 and 6")
    _require(
        fresh_sweep["semantic_scholar_unique_arxiv_ids"] == SEMANTIC_SCHOLAR_UNIQUE_ARXIV_IDS,
        "fresh_sweep Semantic Scholar IDs must match reliable-channel output",
    )
    sources = fresh_sweep["webfetch_top_sources"]
    _require(
        isinstance(sources, Sequence) and not isinstance(sources, str | bytes) and 5 <= len(sources) <= 8,
        "fresh_sweep must record top five to eight WebFetch sources",
    )
    _require(list(sources) == WEBSEARCH_WEBFETCH_TOP_SOURCES, "fresh_sweep sources must match verified source set")


def _validate_upstream_artifacts(upstream_artifacts: object) -> None:
    _require(isinstance(upstream_artifacts, Mapping), "upstream_artifacts must be a mapping")
    _require(set(upstream_artifacts) == REQUIRED_UPSTREAM_FIELDS, "upstream_artifacts must match schema")
    _require(upstream_artifacts["fork_probe_artifact"] == UPSTREAM_FORK_ARTIFACT, "upstream must cite Exp 4861")
    _require(
        upstream_artifacts["expressibility_handoff"] == UPSTREAM_EXPRESSIBILITY_ARTIFACT,
        "upstream must cite Exp 4858 handoff",
    )
    _require(upstream_artifacts["a1_fork_verdict"] is None, "upstream blocked/null A1 fork must be recorded")
    _require(
        upstream_artifacts["aimed_at_fork_verdict"] == AIMED_AT_FORK_VERDICT,
        "upstream fork target must match artifact",
    )
    _require(upstream_artifacts["blocked_a1_recorded_honestly"] is True, "blocked A1 must be explicit")
    closed = upstream_artifacts["retired_classes_kept_closed"]
    _require("macro-vocab" in closed, "retired macro-vocab coverage must stay closed")
    _require("energy classes" in closed, "retired energy classes must stay closed")


def _validate_mapping_note(mapping_note: object, arxiv_ids_cited: object) -> None:
    _require(isinstance(mapping_note, Mapping), "SOTA mapping note must be a mapping")
    _require(set(mapping_note) == REQUIRED_MAPPING_NOTE_FIELDS, "mapping note must match schema")
    _require(mapping_note["terminal_success"] == HONEST_VERDICT, "mapping note terminal success must match verdict")
    _require(mapping_note["root_cause"] == "world-model inducer quality", "mapping note root cause must match")
    _require("INDUCER_CEILING" in str(mapping_note["summary"]), "mapping note must mention INDUCER_CEILING")
    _require("blocked/null" in str(mapping_note["a1_caveat"]), "mapping note must preserve blocked/null A1 caveat")
    source_ids = mapping_note["source_ids"]
    _require(
        isinstance(source_ids, Sequence) and not isinstance(source_ids, str | bytes) and bool(source_ids),
        "mapping note must cite source_ids",
    )
    _require(set(source_ids).issubset(set(arxiv_ids_cited)), "mapping note source_ids must be verified")


def build_research_studying_section(artifact: JsonMap | None = None) -> str:
    result = dict(artifact or build_artifact())
    validate_artifact(result)
    citations = result["citations"]
    citation_lines = "\n".join(
        f"- arXiv:{source_id} -- {citations[source_id]['title']}" for source_id in sorted(citations)
    )
    method_lines = "\n".join(
        (
            f"- **{method['method']}** ({', '.join('arXiv:' + source for source in method['source_ids'])}): "
            f"maps to {method['maps_to_frontier']} / {method['targets_fork_verdict']}. "
            f"Fork mapping: {method['fork_mapping']} Evidence: {method['evidence']} "
            f"Experiment graft: {method['experiment_graft']} Validation gate: {method['validation_gate']} "
            f"Sovereignty: {method['sovereignty_note']} Fails when: {method['fails_when']}"
        )
        for method in result["methods_mapped"]
    )
    flag_lines = "\n".join(flag["flag"] for flag in result["flagged_for_v449"])
    return f"""{STUDYING_SECTION_START}
## 2026-06-27 Exp 4868 - .449 V449 frontier SOTA ingestion - INGESTED

**Status:** INGESTED into `results/experiment_4868_sota_ingestion_v449_frontier.json`.

**Preconditions:** `research-studying.md`, `research-references.md`,
`results/experiment_4861_generation_wall_fork_probe.json`, and
`results/experiment_4858_sota_ingestion_generation_expressibility.json` were
present. The checked-in Exp 4861 A1 `fork_verdict` is blocked/null
(`honest_verdict=blocked_generator_unavailable`), so this note follows the
operator-reserved likely `INDUCER_CEILING` branch without claiming A1 measured
it. `scripts/sweep_clusters.py` emitted ARC action-effect and
neural-guided-search/world-model cluster URLs. `scripts/sweep_semscholar.py`
was run on four focused queries; HTTP 429 limited three queries, and the
test-time dynamics query returned arXiv IDs recorded in the artifact. Low-
concurrency WebSearch/WebFetch plus direct arXiv HTTP checks verified the top
eight papers listed below. `/deep-research` was not invoked. The retired
macro-vocab/click-heatmap coverage, exploration-strategy, and energy classes
were not re-ingested. No model load, training, leaderboard submission, or solve
claim was made; this is a no solve claim ingestion note.

**A1 fork targeted:** `INDUCER_CEILING`, with the caveat that the committed A1
artifact is blocked/null. The .449 handoff is to improve world-model inducer
accuracy before investing in more planner/search machinery.

**Verified source set:**
{citation_lines}

**SOTA -> .449 frontier mapping:**
{method_lines}

{flag_lines}

**Bottom line for .449:** stage the Family-B executable-world-model inducer as
the capability reference, add test-time dynamics adaptation, and wrap the
induced engine in counterexample-guided refinement. Promote the local open
inducer only when held-out transition accuracy improves under the same A1 gate.
{STUDYING_SECTION_END}"""


def build_research_references_section(artifact: JsonMap | None = None) -> str:
    result = dict(artifact or build_artifact())
    validate_artifact(result)
    citations = result["citations"]
    source_lines = "\n".join(
        (
            f"- **arXiv:{source_id} -- {citations[source_id]['title']}.** "
            "Exp 4868 use: V449 INDUCER_CEILING source for improving executable "
            "world-model induction or preserving a local open inducer path."
        )
        for source_id in sorted(citations)
    )
    return f"""{REFERENCES_SECTION_START}
## 2026-06-27 Exp 4868 V449 frontier source set

Reliable-channel ingestion for `.449`, aimed at the reserved
`INDUCER_CEILING` branch while recording that Exp 4861 is blocked/null. These
papers are marked INGESTED for the V449 frontier roadmap handoff:

{source_lines}
{REFERENCES_SECTION_END}"""


def update_research_studying_text(text: str, artifact: JsonMap | None = None) -> str:
    section = build_research_studying_section(artifact)
    return _replace_or_insert_marked_section(text, STUDYING_SECTION_START, STUDYING_SECTION_END, section)


def update_research_references_text(text: str, artifact: JsonMap | None = None) -> str:
    section = build_research_references_section(artifact)
    return _replace_or_insert_marked_section(text, REFERENCES_SECTION_START, REFERENCES_SECTION_END, section)


def _replace_or_insert_marked_section(text: str, start_marker: str, end_marker: str, section: str) -> str:
    start = text.find(start_marker)
    if start >= 0:
        end = text.find(end_marker, start)
        _require(end >= 0, "existing marked section missing end marker")
        end += len(end_marker)
        before = text[:start].rstrip()
        tail = text[end:].lstrip()
        removed = before + ("\n\n" + tail if tail else "\n")
        insert_at = _markdown_insert_index(removed)
        if insert_at >= 0:
            return removed[:insert_at].rstrip() + "\n\n" + section + "\n\n" + removed[insert_at:].lstrip()
        return before + "\n\n" + section + ("\n\n" + tail if tail else "\n")
    insert_at = _markdown_insert_index(text)
    return (
        text[:insert_at].rstrip() + "\n\n" + section + "\n\n" + text[insert_at:].lstrip()
        if insert_at >= 0
        else text.rstrip() + "\n\n" + section + "\n"
    )


def _markdown_insert_index(text: str) -> int:
    first_marker = text.find("\n<!-- EXP")
    if first_marker >= 0:
        return first_marker + 1
    first_section = text.find("\n## ")
    return first_section + 1 if first_section >= 0 else -1


def validate_research_studying_text(text: str, artifact: JsonMap | None = None) -> None:
    result = dict(artifact or build_artifact())
    validate_artifact(result)
    start = text.find(STUDYING_SECTION_START)
    end = text.find(STUDYING_SECTION_END, start)
    _require(start >= 0 and end > start, "research-studying Exp 4868 section missing")
    section = text[start:end]
    _require("INGESTED" in section, "research-studying section must mark Exp 4868 ingested")
    _require("SOTA -> .449 frontier mapping" in section, "research-studying section missing mapping note")
    _require("flagged_for_v449" in section, "research-studying section missing .449 flags")
    _require("INDUCER_CEILING" in section, "research-studying section must name fork target")
    _require("blocked/null" in section, "research-studying section must preserve blocked/null caveat")
    _require("no solve claim" in section, "research-studying section must preserve no solve claim")
    for required in NOTE_REQUIRED_SOURCE_CITATIONS:
        _require(required in section, f"research-studying section missing {required}")


def validate_research_references_text(text: str, artifact: JsonMap | None = None) -> None:
    result = dict(artifact or build_artifact())
    validate_artifact(result)
    start = text.find(REFERENCES_SECTION_START)
    end = text.find(REFERENCES_SECTION_END, start)
    _require(start >= 0 and end > start, "research-references Exp 4868 section missing")
    section = text[start:end]
    _require("Exp 4868 V449 frontier source set" in section, "references section missing title")
    _require("INGESTED" in section, "references section must mark sources ingested")
    for source_id in REQUIRED_SOURCE_IDS:
        _require(f"arXiv:{source_id}" in section, f"references section missing arXiv:{source_id}")


def write_outputs(
    *,
    artifact_path: Path,
    studying_path: Path,
    references_path: Path,
    artifact: JsonMap | None = None,
) -> dict[str, object]:
    result = dict(artifact or build_artifact())
    validate_artifact(result)

    studying_text = studying_path.read_text(encoding="utf-8")
    updated_studying = update_research_studying_text(studying_text, result)
    validate_research_studying_text(updated_studying, result)

    references_text = references_path.read_text(encoding="utf-8")
    updated_references = update_research_references_text(references_text, result)
    validate_research_references_text(updated_references, result)

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    studying_path.write_text(updated_studying, encoding="utf-8")
    references_path.write_text(updated_references, encoding="utf-8")
    return result


def main() -> int:
    root = Path(os.environ.get("CARNOT_EXP4868_ROOT", Path.cwd()))
    artifact = write_outputs(
        artifact_path=root / RESULT_RELATIVE_PATH,
        studying_path=root / STUDYING_RELATIVE_PATH,
        references_path=root / REFERENCES_RELATIVE_PATH,
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
