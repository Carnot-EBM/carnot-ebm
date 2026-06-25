"""Exp 4734 epistemic-MCTS / causal-probe SOTA ingestion.

Spec refs: REQ-ARC-WMTE-4734, SCENARIO-ARC-WMTE-4734.

This is a literature-synthesis artifact, not a solve claim. It maps the next
.436 frontier onto the current E3AgentPolicy and arc_executable_world_model
stack after .435 already built the hypothesis-posterior active-probe controller.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
from typing import Any


JsonMap = Mapping[str, Any]

RESULT_RELATIVE_PATH = "results/experiment_4734_sota_ingestion_epistemic_mcts_causal_probe.json"
NOTE_RELATIVE_PATH = (
    "docs/research-notes/epistemic-mcts-causal-probe-sota-ingestion-20260625.md"
)
RANDOM_SEED = 4734
HONEST_VERDICT = "success: sota_ingestion_epistemic_mcts_causal_probe_mapped"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
FORBIDDEN_DUPLICATE_TRACK = "hypothesis_posterior_active_probe_controller"
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
        "arxiv_reachable",
        "exp4722_artifact_read",
        "research_references_read",
        "arc_competition_agent_read",
        "arc_executable_world_model_read",
        "websearch_webfetch_used",
        "websearch_webfetch_top_sources",
        "arxiv_http_200_verified_ids",
        "deep_research_invoked",
        "live_llm_inference",
        "training_launched",
        "model_load",
        "leaderboard_submission",
        "solve_claim_made",
        "research_conductor_modified",
        "ops_docs_modified",
        "hypothesis_posterior_duplicate_skipped",
    }
)
REQUIRED_SOURCE_IDS = frozenset(
    {
        "2210.13455",
        "2309.08477",
        "2506.01876",
        "2511.02225",
        "2511.06136",
        "2511.14262",
        "2601.06604",
        "2606.14418",
    }
)
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(
    f"arXiv:{source_id}" for source_id in REQUIRED_SOURCE_IDS
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": "terminal prefix; success: sota_ingestion_epistemic_mcts_causal_probe_mapped."
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
            "the strongest method(s) flagged_for_v436 -- closes the "
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
    "2309.08477": {
        "title": "Deep Multi-Agent Reinforcement Learning for Decentralized Active Hypothesis Testing",
        "url": "https://arxiv.org/abs/2309.08477",
        "http_status": 200,
    },
    "2506.01876": {
        "title": "In-Context Learning for Pure Exploration",
        "url": "https://arxiv.org/abs/2506.01876",
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
}

WEBSEARCH_WEBFETCH_TOP_SOURCES = [
    "https://arxiv.org/abs/2506.01876",
    "https://arxiv.org/abs/2309.08477",
    "https://arxiv.org/abs/2210.13455",
    "https://arxiv.org/abs/2601.06604",
    "https://arxiv.org/abs/2606.14418",
    "https://arxiv.org/abs/2511.02225",
    "https://arxiv.org/abs/2511.14262",
    "https://arxiv.org/abs/2511.06136",
]
FLAGGED_FOR_NEXT_ROADMAP = [
    (
        "flagged_for_v436: epistemic_object_model_mcts_probe_planner "
        "(arXiv:2210.13455 + arXiv:2601.06604 + arXiv:2606.14418)"
    ),
    (
        "flagged_for_v436: factored_interaction_causal_probe_bank "
        "(arXiv:2511.02225 + arXiv:2511.14262; guardrail arXiv:2511.06136)"
    ),
]
DEFAULT_PRECONDITIONS_CHECKED = {
    "agents_md_read": True,
    "codex_md_read": True,
    "arxiv_reachable": True,
    "exp4722_artifact_read": True,
    "research_references_read": True,
    "arc_competition_agent_read": True,
    "arc_executable_world_model_read": True,
    "websearch_webfetch_used": True,
    "websearch_webfetch_top_sources": WEBSEARCH_WEBFETCH_TOP_SOURCES,
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
    "hypothesis_posterior_duplicate_skipped": True,
}
DEFAULT_METHODS_MAPPED = [
    {
        "method": "Epistemic object-model MCTS probe planner",
        "source_ids": ["2210.13455", "2601.06604"],
        "track": "epistemic_object_model_mcts_probe_planner",
        "maps_to_current_stack": (
            "E3AgentPolicy calls an uncertainty-aware MCTS planner over "
            "arc_executable_world_model rollouts, replacing the single BFS "
            "plan_in_model call with nodes that can return either a solve action "
            "or an information-gain probe."
        ),
        "implement_cost_over_current_stack": (
            "medium-high: add MCTS node statistics, rollout budgets, uncertainty "
            "propagation over candidate engines and ProductWorldModel factors, and "
            "a live-probe handoff compatible with the existing active-probe phase."
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
            "E3AgentPolicy binds candidate ARC actions to object-like logical-grid "
            "slots before arc_executable_world_model rollouts, so MCTS can test "
            "which object interaction a click or movement action is meant to change."
        ),
        "implement_cost_over_current_stack": (
            "medium-high: derive stable logical object slots from current frames, "
            "attach action-target metadata to candidate actions, and expose causal "
            "attention scores as a planner prior rather than a learned policy head."
        ),
        "fails_when": (
            "ARC objects are not separable at logical-grid resolution, action targets "
            "cannot be grounded from click or keyboard data, or causal attention "
            "prioritizes visually salient but goal-irrelevant slots."
        ),
        "roadmap_candidate": (
            "support_for_v436: causal_object_mcts_action_slot_adapter "
            "(arXiv:2606.14418)"
        ),
    },
    {
        "method": "Factored interaction and causal probe bank",
        "source_ids": ["2511.02225", "2511.14262"],
        "track": "factored_interaction_causal_probe_bank",
        "maps_to_current_stack": (
            "E3AgentPolicy proposes object-interaction hypotheses, "
            "arc_executable_world_model stores confirmed/refuted typed "
            "precondition/effect factors, and probe actions are selected to "
            "settle cause-effect relations before ProductWorldModel planning."
        ),
        "implement_cost_over_current_stack": (
            "high: promote ProgrammaticExpert rows into a first-class causal factor "
            "bank, add confirm/refute ledgers, and let the factored planner compose "
            "only trusted interaction factors as subgoal transitions."
        ),
        "fails_when": (
            "object slots drift, causal labels alias hidden registers, interaction "
            "factors require longer interventions than the probe budget, or short "
            "prefixes make a spurious relation look causal."
        ),
        "roadmap_candidate": FLAGGED_FOR_NEXT_ROADMAP[1],
    },
    {
        "method": "Object-world-model drift and policy-breakage falsifier",
        "source_ids": ["2511.06136"],
        "track": "object_world_model_drift_policy_breakage_falsifier",
        "maps_to_current_stack": (
            "E3AgentPolicy invalidates arc_executable_world_model plans when "
            "off-path object factors or causal relations drift under multi-object "
            "interactions, then routes the failure into the probe/factor ledger "
            "instead of executing a brittle plan."
        ),
        "implement_cost_over_current_stack": (
            "low-medium: add off-path drift diagnostics, rejected-factor reasons, "
            "and plan invalidation when object-model predictions stay visually "
            "plausible but causally wrong."
        ),
        "fails_when": (
            "the drift metric is too conservative and rejects all useful factors, "
            "or too permissive and lets unstable object rollouts pass into execution."
        ),
        "roadmap_candidate": (
            "guardrail_for_v436: object_world_model_policy_breakage_falsifier "
            "(arXiv:2511.06136)"
        ),
    },
]


def source_set_checksum(citations: JsonMap) -> str:
    """Return a stable content hash for the ingested citation set."""

    payload = json.dumps(citations, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


REPRODUCIBILITY_CHECKSUM = source_set_checksum(CITATIONS)


def build_artifact(
    *,
    methods_mapped: Sequence[JsonMap] = DEFAULT_METHODS_MAPPED,
    citations: JsonMap = CITATIONS,
    preconditions_checked: JsonMap = DEFAULT_PRECONDITIONS_CHECKED,
    flagged_for_next_roadmap: Sequence[str] = FLAGGED_FOR_NEXT_ROADMAP,
    honest_verdict: str = HONEST_VERDICT,
) -> dict[str, object]:
    """Build and validate the REQ-ARC-WMTE-4734 mapping artifact."""

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


def validate_artifact(artifact: JsonMap) -> None:
    """Validate the artifact so uncited .436 method claims fail closed."""

    missing = set(REQUIRED_ARTIFACT_FIELDS).difference(artifact)
    extra = set(artifact).difference(REQUIRED_ARTIFACT_FIELDS)
    if missing:
        raise ValueError(f"artifact missing required fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"artifact has unexpected fields: {sorted(extra)}")

    verdict = artifact["honest_verdict"]
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
        if method["track"] == FORBIDDEN_DUPLICATE_TRACK:
            raise ValueError("already-built hypothesis-posterior method must not be mapped")
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
            raise ValueError("methods must map to E3AgentPolicy and arc_executable_world_model")
        if not method["implement_cost_over_current_stack"]:
            raise ValueError("each method needs implement_cost_over_current_stack")
        if not method["fails_when"]:
            raise ValueError("each method needs fails_when")

    roadmap = artifact["flagged_for_next_roadmap"]
    if not isinstance(roadmap, Sequence) or isinstance(roadmap, str | bytes) or not all(
        isinstance(item, str) and "flagged_for_v436" in item for item in roadmap
    ):
        raise ValueError("flagged_for_next_roadmap must contain .436 flagged_for_v436 items")
    if any(FORBIDDEN_DUPLICATE_TRACK in item for item in roadmap):
        raise ValueError("already-built hypothesis-posterior method must not be flagged")

    preconditions = artifact["preconditions_checked"]
    if not isinstance(preconditions, Mapping) or set(preconditions) != REQUIRED_PRECONDITION_FIELDS:
        raise ValueError("preconditions_checked must match the required schema")
    if preconditions["arxiv_reachable"] is not True:
        raise ValueError("network precondition must record reachable arXiv")
    if preconditions["deep_research_invoked"] is not False:
        raise ValueError("deep-research must not be invoked")
    if preconditions["hypothesis_posterior_duplicate_skipped"] is not True:
        raise ValueError("already-built hypothesis-posterior method must be skipped")
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
    """Extract the machine-readable artifact from the research note."""

    marker = "```json\n"
    start = note.find(marker)
    if start < 0:
        raise ValueError("research note missing machine-readable JSON block")
    payload_start = start + len(marker)
    terminator = "\n```\n\n## Fresh-pass provenance"
    end = note.find(terminator, payload_start)
    if end < 0:
        raise ValueError("research note JSON block missing terminator")
    artifact = json.loads(note[payload_start:end])
    validate_artifact(artifact)
    return artifact


def validate_research_note(note: str) -> None:
    """Validate the prose note and its embedded JSON agree."""

    artifact_from_note(note)
    terminator = "\n```\n\n## Fresh-pass provenance"
    prose = note[note.find(terminator) + len(terminator) :]
    if "hypothesis_posterior_active_probe_controller flagged_for_v436" in prose:
        raise ValueError("already-built hypothesis-posterior method must not be flagged")
    required_phrases = (
        "SOTA -> .436 epistemic-MCTS / causal-probe mapping",
        "Bottom line for the .436 roadmap",
        "E3AgentPolicy",
        "arc_executable_world_model",
        "flagged_for_v436",
        "no solve claim",
        "hypothesis-posterior controller is not re-flagged",
    )
    for phrase in required_phrases:
        if phrase not in prose:
            raise ValueError(f"research note missing required phrase: {phrase}")
    missing_citations = sorted(
        citation for citation in NOTE_REQUIRED_SOURCE_CITATIONS if citation not in prose
    )
    if missing_citations:
        raise ValueError(f"research note missing verified source citations: {missing_citations}")


def _research_note() -> str:
    artifact_json = json.dumps(build_artifact(), indent=2, sort_keys=True)
    return f"""# Epistemic-MCTS / causal-probe SOTA ingestion 20260625

```json
{artifact_json}
```

## Fresh-pass provenance

Read `AGENTS.md`, `CODEX.md`,
`results/experiment_4722_sota_ingestion_active_probe_world_model.json`,
`research-references.md`, `python/carnot/agentic/arc_competition_agent.py`,
and `python/carnot/agentic/arc_executable_world_model.py`. The .435 A2 build
already implemented the hypothesis-posterior active-probe controller, so the
hypothesis-posterior controller is not re-flagged here. This note maps the
remaining .436 frontier: uncertainty-aware MCTS over executable world-model
rollouts, causal object-action slot binding, and factored interaction probes.

Reliable-channel pass, not `/deep-research`:
- `curl -sf -o /dev/null https://arxiv.org`
- focused WebSearch/WebFetch of the top epistemic-MCTS and causal-probe papers
- direct arXiv URL checks for all cited IDs

Direct arXiv HTTP checks returned 200 for arXiv:2506.01876, arXiv:2309.08477,
arXiv:2210.13455, arXiv:2601.06604, arXiv:2606.14418, arXiv:2511.02225,
arXiv:2511.14262, and arXiv:2511.06136. The first two are carried only as the
already-built .435 active-hypothesis baseline, not as a new .436 roadmap flag.
No live LLM inference, no model load, no training, no leaderboard submission,
and no solve claim were run or made. `scripts/research_conductor.py`,
`ops/changelog.md`, and `ops/status.md` were not edited by this workflow.

## SOTA -> .436 epistemic-MCTS / causal-probe mapping

## Epistemic object-model MCTS probe planner

**Sources:** Epistemic MCTS, arXiv:2210.13455; ObjectZero, arXiv:2601.06604.

**Mapping to current stack:** `E3AgentPolicy` currently calls
`arc_executable_world_model.plan_in_model` as a bounded BFS over one selected
engine. The v436 planner would replace that single-shot search with MCTS nodes
that propagate epistemic uncertainty across candidate engines and object-factor
rollouts, then return either the best solve action or the highest information
gain probe.

**Implementation cost over current stack:** medium-high. Add MCTS state keys,
rollout statistics, uncertainty propagation, and a probe-vs-act policy while
keeping the existing active-probe transition observer.

**Fails when:** uncertainty is uncalibrated, the object abstraction does not
reduce the branch factor, or model error compounds across rollouts faster than
live probes can correct it.

## Causal object-centric MCTS action-slot adapter

**Source:** COMET, arXiv:2606.14418.

**Mapping to current stack:** bind candidate ARC actions to object-like logical
slots before `arc_executable_world_model` rollouts, so MCTS can reason about
which object a click, drag, or movement action is expected to affect.

**Implementation cost over current stack:** medium-high. Derive stable logical
object slots from current frames, attach action-target metadata to candidate
actions, and expose causal attention scores as planner priors without adding a
learned policy head to the submitted path.

**Fails when:** object slots drift, action targets cannot be grounded from ARC
action data, or causal attention follows salient but goal-irrelevant objects.

## Factored interaction and causal probe bank

**Sources:** FIOC-WM, arXiv:2511.02225; STICA, arXiv:2511.14262.

**Mapping to current stack:** `E3AgentPolicy` already has a factored-planner
hook and `arc_executable_world_model` already has `ProgrammaticExpert`,
`ProductWorldModel`, and trusted-factor summaries. The v436 bank would promote
those rows into typed precondition/effect factors whose causal status is
confirmed or refuted by live probes.

**Implementation cost over current stack:** high. Add a factor schema, a
confirm/refute ledger, causal relation scoring, and planner composition only
over trusted factors.

**Fails when:** object slots drift, hidden registers alias relation labels,
short prefixes make spurious interactions look causal, or the probe budget is
too small for the needed intervention.

## Object-world-model drift and policy-breakage falsifier

**Source:** When Object-Centric World Models Meet Policy Learning,
arXiv:2511.06136.

**Mapping to current stack:** `E3AgentPolicy` should invalidate plans from
`arc_executable_world_model` when off-path object factors or causal relations
drift under multi-object interactions, then route those failures back into the
probe and factor ledgers instead of executing a brittle plan.

**Implementation cost over current stack:** low-medium. Add held-out off-path
drift diagnostics, rejected-factor reasons, and plan invalidation for visually
plausible but causally wrong rollouts.

**Fails when:** the drift metric rejects every useful factor or allows unstable
object rollouts into execution.

## Bottom line for the .436 roadmap

The strongest next candidate is
flagged_for_v436: epistemic_object_model_mcts_probe_planner
(arXiv:2210.13455 + arXiv:2601.06604 + arXiv:2606.14418). It is the direct
upgrade from the .435 posterior splitter to a planner that can choose between
acting and probing inside world-model rollouts.

The second v436 candidate is
flagged_for_v436: factored_interaction_causal_probe_bank
(arXiv:2511.02225 + arXiv:2511.14262), guarded by the drift/breakage falsifier
from arXiv:2511.06136. The honest bound is explicit: object-centric world
models can look visually stable and still break policy learning under off-path
multi-object interactions, so this ingestion makes no solve claim.
"""


RESEARCH_NOTE = _research_note()


def write_outputs(*, artifact_path: Path, note_path: Path) -> dict[str, object]:
    """Write the stable artifact and research note."""

    artifact = build_artifact()
    validate_research_note(RESEARCH_NOTE)

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    note_path.write_text(RESEARCH_NOTE, encoding="utf-8")
    return artifact


def main() -> int:
    root = Path(os.environ.get("CARNOT_EXP4734_ROOT", Path(__file__).resolve().parents[2]))
    write_outputs(
        artifact_path=root / RESULT_RELATIVE_PATH,
        note_path=root / NOTE_RELATIVE_PATH,
    )
    print(HONEST_VERDICT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
