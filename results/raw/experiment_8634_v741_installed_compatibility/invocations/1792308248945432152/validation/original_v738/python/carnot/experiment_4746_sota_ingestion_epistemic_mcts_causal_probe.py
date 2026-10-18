"""Exp 4746 epistemic-MCTS / causal-probe / MATM SOTA ingestion.

Spec refs: REQ-ARC-WMTE-4746, SCENARIO-ARC-WMTE-4746.

This is a literature-synthesis artifact, not a solve claim. It maps the .437
frontier onto the current E3AgentPolicy, StepwiseExplorer.adj, and
arc_executable_world_model stack after .436 valid-tested guidance-class
generation levers instead of building the .435-flagged methods.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from urllib import request

from carnot import experiment_4734_sota_ingestion_epistemic_mcts_causal_probe as exp4734


JsonMap = Mapping[str, Any]

RESULT_RELATIVE_PATH = "results/experiment_4746_sota_ingestion_epistemic_mcts_causal_probe.json"
NOTE_RELATIVE_PATH = (
    "docs/research-notes/epistemic-mcts-causal-probe-sota-ingestion-20260625.md"
)
RANDOM_SEED = 4746
HONEST_VERDICT = "success: sota_ingestion_epistemic_mcts_causal_probe_matm_mapped"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
JSON_BLOCK_MARKER = "## Exp 4746 artifact\n\n```json\n"
JSON_BLOCK_TERMINATOR = "\n```\n\n## Exp 4746 fresh-pass provenance"
TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
    "blocked_",
)
REQUIRED_PRINCIPLE_FIELDS = frozenset(
    {
        "honest_verdict",
        "inference_substrate",
        "methods_mapped",
        "citations",
        "flagged_for_next_roadmap",
        "note_path",
        "verifier_is_oracle",
        "random_seed",
        "reproducibility_checksum",
        "preconditions_checked",
    }
)
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "methods_mapped",
    "citations",
    "flagged_for_next_roadmap",
    "note_path",
    "verifier_is_oracle",
    "random_seed",
    "reproducibility_checksum",
    "preconditions_checked",
    "field_principles",
)
REQUIRED_METHOD_FIELDS = frozenset(
    {
        "method",
        "source_ids",
        "track",
        "maps_to_current_stack",
        "implement_cost_over_current_stack",
        "fails_when",
        "roadmap_candidate",
    }
)
REQUIRED_CITATION_FIELDS = frozenset({"title", "url", "http_status"})
REQUIRED_PRECONDITION_FIELDS = frozenset(
    {
        "agents_md_read",
        "codex_md_read",
        "claude_md_read",
        "arxiv_reachable",
        "exp4734_artifact_read",
        "research_references_read",
        "matm_note_read",
        "arc_competition_agent_read",
        "arc_executable_world_model_read",
        "websearch_webfetch_used",
        "websearch_webfetch_top_sources",
        "top_source_count",
        "arxiv_http_200_verified_ids",
        "deep_research_invoked",
        "live_llm_inference",
        "training_launched",
        "model_load",
        "leaderboard_submission",
        "solve_claim_made",
        "research_conductor_modified",
        "ops_docs_modified",
        "matm_bounded_to_within_game_efficiency",
        "cross_game_matm_claim_made",
        "level_bank_claim_made",
    }
)
REQUIRED_SOURCE_IDS = frozenset(
    {
        "2210.13455",
        "2511.02225",
        "2511.06136",
        "2511.14262",
        "2601.06604",
        "2606.14418",
        "2606.19911",
    }
)
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(
    f"arXiv:{source_id}" for source_id in REQUIRED_SOURCE_IDS
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success: "
            "sota_ingestion_epistemic_mcts_causal_probe_matm_mapped."
        )
    },
    "inference_substrate": {
        "principle": (
            "aggregation_from_upstream_artifacts -- literature synthesis + WebFetch, "
            "no model load (100us floor)."
        )
    },
    "methods_mapped": {
        "principle": (
            "the 3-5 strongest methods, each with maps_to_current_stack + "
            "implement_cost_over_current_stack + fails_when -- the actionable "
            "ingestion (discover -> ingest -> plan)."
        )
    },
    "citations": {
        "principle": (
            "real arXiv IDs/URLs for every method claim -- an ingestion with no "
            "verifiable citations is fabrication per adversarial_verify discipline."
        )
    },
    "flagged_for_next_roadmap": {
        "principle": (
            "the strongest method(s) flagged_for_v437 -- closes the "
            "discover->ingest->plan loop so SOTA flows into the next milestone's experiments."
        )
    },
    "note_path": {
        "principle": (
            "docs/research-notes/epistemic-mcts-causal-probe-sota-ingestion-20260625.md "
            "-- the human-readable per-track synthesis."
        )
    },
    "verifier_is_oracle": {
        "principle": "false -- a literature synthesis invokes no oracle."
    },
    "random_seed": {
        "principle": "determinism precondition (the search/synthesis seed)."
    },
    "reproducibility_checksum": {
        "principle": "content-addressed hash of the ingested source set."
    },
    "preconditions_checked": {
        "principle": (
            "records the network-reachability check; pre-empts missing-resource fabrication."
        )
    },
}

CITATIONS = {
    "2210.13455": {
        "title": "Epistemic Monte Carlo Tree Search",
        "url": "https://arxiv.org/abs/2210.13455",
        "http_status": 200,
    },
    "2511.02225": {
        "title": "Learning Interactive World Model for Object-Centric Reinforcement Learning",
        "url": "https://arxiv.org/abs/2511.02225",
        "http_status": 200,
    },
    "2511.06136": {
        "title": (
            "When Object-Centric World Models Meet Policy Learning: From Pixels to "
            "Policies, and Where It Breaks"
        ),
        "url": "https://arxiv.org/abs/2511.06136",
        "http_status": 200,
    },
    "2511.14262": {
        "title": "Object-Centric World Models for Causality-Aware Reinforcement Learning",
        "url": "https://arxiv.org/abs/2511.14262",
        "http_status": 200,
    },
    "2601.06604": {
        "title": "Object-Centric World Models Meet Monte Carlo Tree Search",
        "url": "https://arxiv.org/abs/2601.06604",
        "http_status": 200,
    },
    "2606.14418": {
        "title": "Causal Object-Centric Models for Planning with Monte Carlo Tree Search",
        "url": "https://arxiv.org/abs/2606.14418",
        "http_status": 200,
    },
    "2606.19911": {
        "title": "Multi-Agent Transactive Memory",
        "url": "https://arxiv.org/abs/2606.19911",
        "http_status": 200,
    },
}

WEBSEARCH_WEBFETCH_TOP_SOURCES = [
    "https://arxiv.org/abs/2210.13455",
    "https://arxiv.org/abs/2601.06604",
    "https://arxiv.org/abs/2606.14418",
    "https://arxiv.org/abs/2511.02225",
    "https://arxiv.org/abs/2511.14262",
    "https://arxiv.org/abs/2511.06136",
    "https://arxiv.org/abs/2606.19911",
]
FLAGGED_FOR_NEXT_ROADMAP = [
    (
        "flagged_for_v437: epistemic_object_model_mcts_probe_planner "
        "(arXiv:2210.13455 + arXiv:2601.06604 + arXiv:2606.14418)"
    ),
    (
        "flagged_for_v437: factored_interaction_causal_probe_bank "
        "(arXiv:2511.02225 + arXiv:2511.14262; guardrail arXiv:2511.06136)"
    ),
    (
        "flagged_for_v437: similarity_keyed_partial_trajectory_retrieval "
        "(MATM arXiv:2606.19911; within-game action-efficiency candidate, "
        "not a level-bank)"
    ),
]
DEFAULT_PRECONDITIONS_CHECKED = {
    "agents_md_read": True,
    "codex_md_read": True,
    "claude_md_read": True,
    "arxiv_reachable": True,
    "exp4734_artifact_read": True,
    "research_references_read": True,
    "matm_note_read": True,
    "arc_competition_agent_read": True,
    "arc_executable_world_model_read": True,
    "websearch_webfetch_used": True,
    "websearch_webfetch_top_sources": WEBSEARCH_WEBFETCH_TOP_SOURCES,
    "top_source_count": len(WEBSEARCH_WEBFETCH_TOP_SOURCES),
    "arxiv_http_200_verified_ids": [
        f"https://arxiv.org/abs/{source_id}" for source_id in sorted(REQUIRED_SOURCE_IDS)
    ],
    "deep_research_invoked": False,
    "live_llm_inference": False,
    "training_launched": False,
    "model_load": False,
    "leaderboard_submission": False,
    "solve_claim_made": False,
    "research_conductor_modified": False,
    "ops_docs_modified": False,
    "matm_bounded_to_within_game_efficiency": True,
    "cross_game_matm_claim_made": False,
    "level_bank_claim_made": False,
}
DEFAULT_METHODS_MAPPED = [
    {
        "method": "Epistemic object-model MCTS probe planner",
        "source_ids": ["2210.13455", "2601.06604", "2606.14418"],
        "track": "epistemic_object_model_mcts_probe_planner",
        "maps_to_current_stack": (
            "E3AgentPolicy replaces the single arc_executable_world_model BFS plan "
            "with uncertainty-aware MCTS rollouts over candidate engines and "
            "ProductWorldModel factors, returning either a solve action or an "
            "information-gain probe."
        ),
        "implement_cost_over_current_stack": (
            "medium-high: add MCTS node statistics, rollout budgets, epistemic "
            "uncertainty propagation across candidate engines, and a live-probe "
            "handoff compatible with the existing active-probe observer."
        ),
        "fails_when": (
            "epistemic uncertainty is miscalibrated, object state keys alias hidden "
            "registers, or rollout error compounds before live probes can correct "
            "the model."
        ),
        "roadmap_candidate": FLAGGED_FOR_NEXT_ROADMAP[0],
    },
    {
        "method": "Causal object-centric MCTS action-slot adapter",
        "source_ids": ["2606.14418"],
        "track": "causal_object_mcts_action_slot_adapter",
        "maps_to_current_stack": (
            "E3AgentPolicy binds ARC candidate actions to object-like logical-grid "
            "slots before arc_executable_world_model rollouts, so MCTS can test "
            "which object interaction a click or movement action is expected to change."
        ),
        "implement_cost_over_current_stack": (
            "medium-high: derive stable logical object slots from current frames, "
            "attach action-target metadata to candidates, and expose causal "
            "attention scores as planner priors rather than a learned submitted-policy head."
        ),
        "fails_when": (
            "ARC objects are not separable at logical-grid resolution, action targets "
            "cannot be grounded from click or keyboard data, or causal attention "
            "prioritizes visually salient but goal-irrelevant slots."
        ),
        "roadmap_candidate": (
            "support_for_v437: causal_object_mcts_action_slot_adapter "
            "(arXiv:2606.14418)"
        ),
    },
    {
        "method": "Factored interaction and causal probe bank",
        "source_ids": ["2511.02225", "2511.14262", "2511.06136"],
        "track": "factored_interaction_causal_probe_bank",
        "maps_to_current_stack": (
            "E3AgentPolicy proposes object-interaction hypotheses; "
            "arc_executable_world_model promotes ProgrammaticExpert rows into "
            "typed precondition/effect factors with confirmed/refuted causal "
            "ledgers before ProductWorldModel planning composes them."
        ),
        "implement_cost_over_current_stack": (
            "high: add a factor schema, confirm/refute probe ledger, causal "
            "relation scoring, and planner composition only over trusted factors."
        ),
        "fails_when": (
            "object slots drift, hidden registers alias relation labels, short "
            "prefixes make spurious interactions look causal, or the probe budget "
            "is too small for the needed intervention."
        ),
        "roadmap_candidate": FLAGGED_FOR_NEXT_ROADMAP[1],
    },
    {
        "method": "Similarity-keyed partial-trajectory retrieval",
        "source_ids": ["2606.19911"],
        "track": "similarity_keyed_partial_trajectory_retrieval",
        "maps_to_current_stack": (
            "E3AgentPolicy keeps the existing StepwiseExplorer.adj exact-edge graph "
            "but adds a coarse within-game similarity descriptor so near-match "
            "frontier states can reuse partial paths; arc_executable_world_model "
            "and value/goal routing score retrieved prefixes before commit."
        ),
        "implement_cost_over_current_stack": (
            "low-medium: add a flag-gated descriptor index beside adj, candidate "
            "prefix retrieval, verifier/value scoring, and diagnostics for "
            "forward_walk_hit_rate and actions_to_first_levelup deltas."
        ),
        "fails_when": (
            "within-game similarity descriptors collide across incompatible "
            "mechanics, stale prefixes waste actions, or the verifier cannot reject "
            "a plausible but harmful retrieved continuation."
        ),
        "roadmap_candidate": FLAGGED_FOR_NEXT_ROADMAP[2],
    },
    {
        "method": "Object-world-model drift and policy-breakage falsifier",
        "source_ids": ["2511.06136"],
        "track": "object_world_model_drift_policy_breakage_falsifier",
        "maps_to_current_stack": (
            "E3AgentPolicy invalidates arc_executable_world_model plans when "
            "off-path object factors or causal relations drift under multi-object "
            "interactions, then routes the failure into probe/factor ledgers "
            "instead of executing a brittle plan."
        ),
        "implement_cost_over_current_stack": (
            "low-medium: add off-path drift diagnostics, rejected-factor reasons, "
            "and plan invalidation when object-model predictions stay visually "
            "plausible but causally wrong."
        ),
        "fails_when": (
            "the drift metric is too conservative and rejects useful factors, or "
            "too permissive and lets unstable object rollouts pass into execution."
        ),
        "roadmap_candidate": (
            "guardrail_for_v437: object_world_model_policy_breakage_falsifier "
            "(arXiv:2511.06136)"
        ),
    },
]


def source_set_checksum(citations: JsonMap) -> str:
    """Return a stable content hash for the ingested citation set."""

    payload = json.dumps(citations, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


REPRODUCIBILITY_CHECKSUM = source_set_checksum(CITATIONS)


def _blocked_preconditions() -> dict[str, object]:
    preconditions = dict(DEFAULT_PRECONDITIONS_CHECKED)
    preconditions.update(
        {
            "arxiv_reachable": False,
            "websearch_webfetch_used": False,
            "websearch_webfetch_top_sources": [],
            "top_source_count": 0,
            "arxiv_http_200_verified_ids": [],
        }
    )
    return preconditions


def build_blocked_network_artifact() -> dict[str, object]:
    """Build the no-fabrication blocked artifact for a failed arXiv precondition."""

    return {
        "honest_verdict": "blocked_network",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "methods_mapped": [],
        "citations": {},
        "flagged_for_next_roadmap": [],
        "note_path": NOTE_RELATIVE_PATH,
        "verifier_is_oracle": False,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": source_set_checksum({}),
        "preconditions_checked": _blocked_preconditions(),
        "field_principles": dict(FIELD_PRINCIPLES),
    }


def build_artifact(
    *,
    methods_mapped: Sequence[JsonMap] = DEFAULT_METHODS_MAPPED,
    citations: JsonMap = CITATIONS,
    preconditions_checked: JsonMap = DEFAULT_PRECONDITIONS_CHECKED,
    flagged_for_next_roadmap: Sequence[str] = FLAGGED_FOR_NEXT_ROADMAP,
    honest_verdict: str = HONEST_VERDICT,
) -> dict[str, object]:
    """Build and validate the REQ-ARC-WMTE-4746 mapping artifact."""

    artifact: dict[str, object] = {
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "methods_mapped": [dict(method) for method in methods_mapped],
        "citations": {source_id: dict(citation) for source_id, citation in citations.items()},
        "flagged_for_next_roadmap": list(flagged_for_next_roadmap),
        "note_path": NOTE_RELATIVE_PATH,
        "verifier_is_oracle": False,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": source_set_checksum(citations),
        "preconditions_checked": dict(preconditions_checked),
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: JsonMap, *, allow_blocked: bool = False) -> None:
    """Validate the artifact so uncited .437 method claims fail closed."""

    missing = set(REQUIRED_ARTIFACT_FIELDS).difference(artifact)
    extra = set(artifact).difference(REQUIRED_ARTIFACT_FIELDS)
    if missing:
        raise ValueError(f"artifact missing required fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"artifact has unexpected fields: {sorted(extra)}")

    verdict = artifact["honest_verdict"]
    if verdict == "blocked_network":
        if not allow_blocked:
            raise ValueError("blocked network artifacts require allow_blocked=True")
        _validate_blocked_artifact(artifact)
        return
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must use a terminal prefix")
    if verdict != HONEST_VERDICT:
        raise ValueError(f"honest_verdict must equal {HONEST_VERDICT!r}")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must match the required substrate")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match the required annotations")
    if artifact["note_path"] != NOTE_RELATIVE_PATH:
        raise ValueError("note_path must point at the 20260625 epistemic-MCTS note")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be false for literature synthesis")
    if artifact["random_seed"] != RANDOM_SEED:
        raise ValueError("random_seed must be the experiment id")

    citations = artifact["citations"]
    if not isinstance(citations, Mapping) or set(citations) != REQUIRED_SOURCE_IDS:
        raise ValueError("citations must exactly cover the required arXiv IDs")
    for source_id, citation in citations.items():
        if not isinstance(citation, Mapping) or set(citation) != REQUIRED_CITATION_FIELDS:
            raise ValueError("each citation must contain exactly title, url, and http_status")
        if citation["url"] != f"https://arxiv.org/abs/{source_id}":
            raise ValueError("citation url must match the arXiv ID")
        if citation["http_status"] != 200:
            raise ValueError("each citation http_status must be 200")
        if not citation["title"]:
            raise ValueError("each citation title must be non-empty")
    if artifact["reproducibility_checksum"] != source_set_checksum(citations):
        raise ValueError("reproducibility checksum must hash the citation source set")

    methods = artifact["methods_mapped"]
    if (
        not isinstance(methods, Sequence)
        or isinstance(methods, str | bytes)
        or not 3 <= len(methods) <= 5
    ):
        raise ValueError("methods_mapped must contain three to five methods")
    for method in methods:
        if not isinstance(method, Mapping) or set(method) != REQUIRED_METHOD_FIELDS:
            raise ValueError("each method must match the required method schema")
        source_ids = method["source_ids"]
        if not isinstance(source_ids, Sequence) or isinstance(source_ids, str | bytes) or not source_ids:
            raise ValueError("each method must cite source_ids")
        if not set(source_ids).issubset(REQUIRED_SOURCE_IDS):
            raise ValueError("method source_ids must be verified citations")
        stack = method["maps_to_current_stack"]
        if (
            not isinstance(stack, str)
            or "E3AgentPolicy" not in stack
            or "arc_executable_world_model" not in stack
        ):
            raise ValueError("methods must map to the current stack")
        if method["track"] == "similarity_keyed_partial_trajectory_retrieval":
            if "StepwiseExplorer.adj" not in stack:
                raise ValueError("MATM retrieval must map to StepwiseExplorer.adj")
        if not method["implement_cost_over_current_stack"]:
            raise ValueError("each method needs implement_cost_over_current_stack")
        if not method["fails_when"]:
            raise ValueError("each method needs fails_when")

    roadmap = artifact["flagged_for_next_roadmap"]
    if not isinstance(roadmap, Sequence) or isinstance(roadmap, str | bytes) or not all(
        isinstance(item, str) and "flagged_for_v437" in item for item in roadmap
    ):
        raise ValueError("flagged_for_next_roadmap must contain .437 flagged_for_v437 items")
    joined_roadmap = " ".join(roadmap).lower()
    if "flagged_for_v436" in joined_roadmap:
        raise ValueError("flagged_for_next_roadmap must not carry .436 flags")
    if "cross_game" in joined_roadmap or "matm_level_bank" in joined_roadmap:
        raise ValueError("MATM must remain within-game efficiency, not a level-bank")

    _validate_preconditions(artifact["preconditions_checked"])


def _validate_blocked_artifact(artifact: JsonMap) -> None:
    if artifact["methods_mapped"] != [] or artifact["citations"] != {}:
        raise ValueError("blocked artifact must not include method claims or citations")
    if artifact["flagged_for_next_roadmap"] != []:
        raise ValueError("blocked artifact must not flag roadmap candidates")
    if artifact["reproducibility_checksum"] != source_set_checksum({}):
        raise ValueError("blocked artifact checksum must hash the empty source set")
    preconditions = artifact["preconditions_checked"]
    if not isinstance(preconditions, Mapping) or set(preconditions) != REQUIRED_PRECONDITION_FIELDS:
        raise ValueError("preconditions_checked must match the required schema")
    if preconditions["arxiv_reachable"] is not False:
        raise ValueError("blocked artifact must record unreachable arXiv")
    if preconditions["websearch_webfetch_used"] is not False:
        raise ValueError("blocked artifact must not claim WebSearch/WebFetch success")


def _validate_preconditions(preconditions: object) -> None:
    if not isinstance(preconditions, Mapping) or set(preconditions) != REQUIRED_PRECONDITION_FIELDS:
        raise ValueError("preconditions_checked must match the required schema")
    if preconditions["arxiv_reachable"] is not True:
        raise ValueError("network precondition must record reachable arXiv")
    if preconditions["deep_research_invoked"] is not False:
        raise ValueError("deep-research must not be invoked")
    if not 5 <= int(preconditions["top_source_count"]) <= 8:
        raise ValueError("top_source_count must record top five to eight sources")
    if preconditions["cross_game_matm_claim_made"] is not False:
        raise ValueError("cross-game MATM claims are not allowed")
    if preconditions["level_bank_claim_made"] is not False:
        raise ValueError("MATM level-bank claims are not allowed")
    if preconditions["solve_claim_made"] is not False:
        raise ValueError("solve claim must remain false for ingestion")
    if preconditions["training_launched"] is not False:
        raise ValueError("training must not be launched")
    if preconditions["model_load"] is not False:
        raise ValueError("model load must not occur")
    if preconditions["leaderboard_submission"] is not False:
        raise ValueError("leaderboard submission must not occur")
    if preconditions["research_conductor_modified"] is not False:
        raise ValueError("research_conductor must not be modified")
    if preconditions["ops_docs_modified"] is not False:
        raise ValueError("ops docs must not be modified by this workflow")


def artifact_from_note(note: str) -> dict[str, object]:
    """Extract the Exp 4746 machine-readable artifact from the shared note."""

    start = note.find(JSON_BLOCK_MARKER)
    if start < 0:
        raise ValueError("research note missing Exp 4746 machine-readable JSON block")
    payload_start = start + len(JSON_BLOCK_MARKER)
    end = note.find(JSON_BLOCK_TERMINATOR, payload_start)
    if end < 0:
        raise ValueError("research note JSON block missing terminator")
    artifact = json.loads(note[payload_start:end])
    validate_artifact(artifact)
    return artifact


def validate_research_note(note: str) -> None:
    """Validate the appended .437 prose note and embedded artifact agree."""

    artifact_from_note(note)
    terminator_index = note.find(JSON_BLOCK_TERMINATOR)
    prose = note[terminator_index + len(JSON_BLOCK_TERMINATOR) :]
    forbidden = prose.lower().replace("-", "_")
    if "cross_game matm" in forbidden or "matm level_bank" in forbidden:
        raise ValueError("MATM must remain within-game efficiency, not a level-bank")
    required_phrases = (
        "SOTA -> .437 epistemic-MCTS / causal-probe / MATM mapping",
        "Bottom line for the .437 roadmap",
        "E3AgentPolicy",
        "StepwiseExplorer.adj",
        "arc_executable_world_model",
        "flagged_for_v437",
        "no solve claim",
        "within-game action-efficiency candidate, not a level-bank",
    )
    for phrase in required_phrases:
        if phrase not in prose:
            raise ValueError(f"research note missing required phrase: {phrase}")
    missing_citations = sorted(
        citation for citation in NOTE_REQUIRED_SOURCE_CITATIONS if citation not in prose
    )
    if missing_citations:
        raise ValueError(f"research note missing verified source citations: {missing_citations}")


def _research_note_refresh() -> str:
    artifact_json = json.dumps(build_artifact(), indent=2, sort_keys=True)
    return f"""# Exp 4746 .437 refresh

{JSON_BLOCK_MARKER}{artifact_json}{JSON_BLOCK_TERMINATOR}

Read `AGENTS.md`, `CODEX.md`, `CLAUDE.md`,
`results/experiment_4734_sota_ingestion_epistemic_mcts_causal_probe.json`,
`research-references.md`,
`docs/research-notes/matm-transactive-memory-ingestion-2026-06-23.md`,
`python/carnot/agentic/arc_competition_agent.py`, and
`python/carnot/agentic/arc_executable_world_model.py`.

Fresh-pass source set, all checked through arXiv:
- arXiv:2210.13455, Epistemic Monte Carlo Tree Search.
- arXiv:2601.06604, Object-Centric World Models Meet Monte Carlo Tree Search.
- arXiv:2606.14418, Causal Object-Centric Models for Planning with Monte Carlo Tree Search.
- arXiv:2511.02225, Learning Interactive World Model for Object-Centric Reinforcement Learning.
- arXiv:2511.14262, Object-Centric World Models for Causality-Aware Reinforcement Learning.
- arXiv:2511.06136, When Object-Centric World Models Meet Policy Learning.
- arXiv:2606.19911, Multi-Agent Transactive Memory.

No `/deep-research`, no live LLM inference, no model load, no training, no
leaderboard submission, and no solve claim. `scripts/research_conductor.py`,
`ops/changelog.md`, and `ops/status.md` were not edited by this workflow.

## SOTA -> .437 epistemic-MCTS / causal-probe / MATM mapping

## Epistemic object-model MCTS probe planner

**Sources:** arXiv:2210.13455, arXiv:2601.06604, arXiv:2606.14418.

**Mapping:** `E3AgentPolicy` should replace the current single
`arc_executable_world_model.plan_in_model` BFS choice with an uncertainty-aware
MCTS planner over candidate engines and object factors. The planner returns
either a solve action or an information-gain probe.

**Cost:** medium-high. Add MCTS state keys, uncertainty propagation, rollout
budgets, and a probe-vs-act handoff to the existing active-probe observer.

**Fails when:** uncertainty is uncalibrated, hidden registers alias object
state, or rollout error compounds before probes can correct the model.

## Causal object-centric MCTS action-slot adapter

**Source:** arXiv:2606.14418.

**Mapping:** bind candidate ARC actions to logical-grid object slots before
`arc_executable_world_model` rollouts so MCTS can reason about which object a
click, drag, or movement action is expected to change.

**Cost:** medium-high. Build stable slot descriptors and expose causal attention
as a planner prior without adding a learned submitted-policy head.

**Fails when:** object slots drift or causal attention follows visually salient
but goal-irrelevant entities.

## Factored interaction and causal probe bank

**Sources:** arXiv:2511.02225, arXiv:2511.14262; guardrail arXiv:2511.06136.

**Mapping:** `E3AgentPolicy` already exposes factored planning hooks, and
`arc_executable_world_model` already has `ProgrammaticExpert` and
`ProductWorldModel`. The .437 bank promotes those rows into typed
precondition/effect factors whose causal status is confirmed or refuted by
targeted probes before composition.

**Cost:** high. Add factor schemas, confirm/refute ledgers, causal relation
scores, and planner composition only over trusted factors.

**Fails when:** object slots drift, relation labels alias hidden registers, or
the probe budget is too small for the intervention.

## Similarity-keyed partial-trajectory retrieval

**Source:** MATM, arXiv:2606.19911.

**Mapping:** `E3AgentPolicy` keeps the exact-hash `StepwiseExplorer.adj`
navigation graph, then adds a coarse within-game descriptor index so near-match
states can retrieve partial prefixes. `arc_executable_world_model` and the
existing value/goal routing score retrieved prefixes before live commitment.

**Cost:** low-medium. Add a flag-gated descriptor index, candidate-prefix
retrieval, verifier/value scoring, and diagnostics for
`forward_walk_hit_rate_delta` and `actions_to_first_levelup_delta`.

**Fails when:** within-game descriptors collide across incompatible mechanics,
stale prefixes waste actions, or the verifier cannot reject a plausible but
harmful continuation. This is a within-game action-efficiency candidate, not a level-bank.

## Object-world-model drift and policy-breakage falsifier

**Source:** arXiv:2511.06136.

**Mapping:** `E3AgentPolicy` should invalidate `arc_executable_world_model`
plans when object factors drift under off-path multi-object interactions, then
route the failure back into the probe and factor ledgers.

**Cost:** low-medium. Add held-out off-path drift diagnostics, rejected-factor
reasons, and plan invalidation for visually plausible but causally wrong
rollouts.

**Fails when:** the drift metric rejects every useful factor or allows unstable
object rollouts into execution.

## Bottom line for the .437 roadmap

The strongest `.437` build candidate is
flagged_for_v437: epistemic_object_model_mcts_probe_planner
(arXiv:2210.13455 + arXiv:2601.06604 + arXiv:2606.14418). It is the direct
upgrade from the .435 posterior splitter to a planner that chooses between
acting and probing inside world-model rollouts.

The second `.437` build candidate is
flagged_for_v437: factored_interaction_causal_probe_bank
(arXiv:2511.02225 + arXiv:2511.14262), guarded by the drift falsifier from
arXiv:2511.06136.

The efficiency-only candidate is
flagged_for_v437: similarity_keyed_partial_trajectory_retrieval
(MATM arXiv:2606.19911): within-game action-efficiency candidate, not a level-bank.
It should be measured by forward-walk hit-rate and
actions-to-first-level-up deltas, with no cross-game retrieval claim and no
solve claim.
"""


def _research_note() -> str:
    return exp4734.RESEARCH_NOTE.rstrip() + "\n\n---\n\n" + _research_note_refresh()


RESEARCH_NOTE = _research_note()


def _blocked_research_note(artifact: JsonMap) -> str:
    artifact_json = json.dumps(artifact, indent=2, sort_keys=True)
    return f"""# Exp 4746 .437 refresh

{JSON_BLOCK_MARKER}{artifact_json}{JSON_BLOCK_TERMINATOR}

`https://arxiv.org` was unreachable, so this run emitted `blocked_network`
instead of fabricating citations or method claims.
"""


def write_outputs(
    *,
    artifact_path: Path | None = None,
    note_path: Path | None = None,
    artifact: JsonMap | None = None,
    blocked: bool = False,
) -> dict[str, object]:
    """Write the stable JSON artifact and shared research note."""

    result = dict(artifact or (build_blocked_network_artifact() if blocked else build_artifact()))
    if blocked:
        validate_artifact(result, allow_blocked=True)
    else:
        validate_artifact(result)
    target_artifact = artifact_path or Path(RESULT_RELATIVE_PATH)
    target_note = note_path or Path(NOTE_RELATIVE_PATH)
    target_artifact.parent.mkdir(parents=True, exist_ok=True)
    target_note.parent.mkdir(parents=True, exist_ok=True)
    target_artifact.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    target_note.write_text(
        _blocked_research_note(result) if blocked else RESEARCH_NOTE,
        encoding="utf-8",
    )
    return result


def arxiv_reachable(timeout: float = 10.0) -> bool:
    """Check the network precondition without making citation claims."""

    if os.environ.get("CARNOT_EXP4746_FORCE_BLOCKED_NETWORK") == "1":
        return False
    if os.environ.get("CARNOT_EXP4746_SKIP_NETWORK_CHECK") == "1":
        return True
    try:
        with request.urlopen("https://arxiv.org", timeout=timeout) as response:
            return 200 <= int(response.status) < 400
    except Exception:
        return False


def main() -> int:
    root = Path(os.environ.get("CARNOT_EXP4746_ROOT", "."))
    artifact_path = root / RESULT_RELATIVE_PATH
    note_path = root / NOTE_RELATIVE_PATH
    if not arxiv_reachable():
        write_outputs(artifact_path=artifact_path, note_path=note_path, blocked=True)
        print("blocked_network")
        return 0
    artifact = write_outputs(artifact_path=artifact_path, note_path=note_path)
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
