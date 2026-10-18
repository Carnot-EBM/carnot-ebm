"""Exp 4709 structured-world-model SOTA ingestion.

Spec refs: REQ-ARC-WMTE-4709, SCENARIO-ARC-WMTE-4709.

This artifact maps the fallback after the .433 perception and amortized
exploration arms do not bank a live new level. It is a literature-ingestion
deliverable, not a solve claim: every method claim is tied to a verified arXiv
record and to the current live E3 / executable-world-model stack.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
from typing import Any


JsonMap = Mapping[str, Any]

RESULT_RELATIVE_PATH = "results/experiment_4709_sota_ingestion_structured_world_model.json"
NOTE_RELATIVE_PATH = (
    "docs/research-notes/structured-world-model-active-probing-sota-ingestion-2026-06-25.md"
)
RANDOM_SEED = 4709
HONEST_VERDICT = "success: sota_ingestion_structured_world_model_mapped"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
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
        "deep_research_not_used",
        "methods_mapped",
        "citations_verified",
        "flagged_for_next_roadmap",
        "note_path",
        "preconditions_checked",
    }
)
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "deep_research_not_used",
    "methods_mapped",
    "citations_verified",
    "flagged_for_next_roadmap",
    "note_path",
    "preconditions_checked",
    "random_seed",
    "field_principles",
)
REQUIRED_METHOD_FIELDS = frozenset(
    {
        "method",
        "source_ids",
        "track",
        "residual_scope",
        "implement_cost_over_current_stack",
        "maps_to_current_stack",
        "fails_when",
        "roadmap_candidate",
    }
)
REQUIRED_CITATION_FIELDS = frozenset({"title", "url", "http_status"})
REQUIRED_PRECONDITION_FIELDS = frozenset(
    {
        "agents_md_read",
        "codex_md_read",
        "network_hf_models_reachable",
        "sweep_clusters_help_ok",
        "exp4697_artifact_read",
        "exp4697_note_read",
        "exp4700_artifact_read",
        "exp4701_artifact_read",
        "arc_executable_world_model_read",
        "research_studying_read",
        "research_references_read",
        "sweep_clusters_used",
        "sweep_clusters_urls",
        "sweep_semscholar_used",
        "sweep_semscholar_queries",
        "sweep_semscholar_arxiv_ids",
        "arxiv_http_200_verified_ids",
        "websearch_webfetch_top_sources",
        "deep_research_invoked",
        "live_llm_inference",
        "training_launched",
        "model_load",
        "leaderboard_submission",
        "ops_docs_modified",
        "research_conductor_modified",
    }
)
REQUIRED_SOURCE_IDS = frozenset(
    {
        "2210.13455",
        "2307.02427",
        "2309.08477",
        "2410.08822",
        "2506.01876",
        "2511.02225",
        "2511.06136",
        "2601.06604",
    }
)
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(
    f"arXiv:{source_id}" for source_id in REQUIRED_SOURCE_IDS
)

A1_RESIDUAL = "object_centric_perception_no_new_level_residual_offpath_calibration_insufficient"
A2_RESIDUAL = "amortized_prior_go_explore_no_coverage_gain_residual_logged"
NEXT_WALL = "structured-world-model / active-probing next wall"
FLAGGED_FOR_NEXT_ROADMAP = [
    (
        "flagged_for_v434: factored_object_relational_executable_world_model "
        "(arXiv:2511.02225 + arXiv:2410.08822 + arXiv:2307.02427)"
    ),
    (
        "flagged_for_v434: object_model_mcts_with_epistemic_probe_planning "
        "(arXiv:2601.06604 + arXiv:2210.13455)"
    ),
    (
        "flagged_for_v434: hypothesis_driven_active_probe_loop "
        "(arXiv:2506.01876 + arXiv:2309.08477)"
    ),
]

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": "terminal prefix; success: sota_ingestion_structured_world_model_mapped."
    },
    "inference_substrate": {
        "principle": (
            "aggregation_from_upstream_artifacts -- literature read + synthesis, "
            "no model load (100us floor)."
        )
    },
    "deep_research_not_used": {
        "principle": (
            "MUST be true -- /deep-research is BANNED in the autonomous loop; used "
            "sweep helpers + low-concurrency WebSearch/WebFetch."
        )
    },
    "methods_mapped": {
        "principle": (
            "the strongest 3-5 SOTA methods with REAL arXiv IDs + per-method "
            "implement-cost-over-current-stack + fails_when (no citation = fabrication)."
        )
    },
    "citations_verified": {
        "principle": (
            "each cited arXiv ID with an HTTP-200 verification -- pre-empts fabricated "
            "citations."
        )
    },
    "flagged_for_next_roadmap": {
        "principle": (
            "the strongest method(s) flagged as candidate .434 inputs "
            "(flagged_for_v434) -- closes discover->ingest->plan->experiment."
        )
    },
    "note_path": {
        "principle": (
            "the per-track research-note path (the SOTA-Ingestion Cycle deliverable)."
        )
    },
    "preconditions_checked": {
        "principle": "records network reachability verified; pre-empts fabricated citations."
    },
}

CITATIONS_VERIFIED = {
    "2210.13455": {
        "title": "Epistemic Monte Carlo Tree Search",
        "url": "https://arxiv.org/abs/2210.13455",
        "http_status": 200,
    },
    "2307.02427": {
        "title": "FOCUS: Object-Centric World Models for Robotics Manipulation",
        "url": "https://arxiv.org/abs/2307.02427",
        "http_status": 200,
    },
    "2309.08477": {
        "title": "Deep Multi-Agent Reinforcement Learning for Decentralized Active Hypothesis Testing",
        "url": "https://arxiv.org/abs/2309.08477",
        "http_status": 200,
    },
    "2410.08822": {
        "title": (
            "SOLD: Slot Object-Centric Latent Dynamics Models for Relational "
            "Manipulation Learning from Pixels"
        ),
        "url": "https://arxiv.org/abs/2410.08822",
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
    "2601.06604": {
        "title": "Object-Centric World Models Meet Monte Carlo Tree Search",
        "url": "https://arxiv.org/abs/2601.06604",
        "http_status": 200,
    },
}

SWEEP_CLUSTER_URLS = [
    (
        "http://export.arxiv.org/api/query?search_query="
        '(abs:"neural+guided+search"+OR+abs:"learned+heuristic"+OR+'
        'abs:"value+guided+search"+OR+abs:"program+induction"+OR+'
        'abs:"world+model"+OR+abs:"goal+induction")+AND+'
        '(abs:"planning"+OR+abs:"agent"+OR+abs:"reasoning"+OR+'
        'abs:"reinforcement+learning")&start=0&max_results=8'
        "&sortBy=submittedDate&sortOrder=descending"
    ),
    (
        "http://export.arxiv.org/api/query?search_query="
        '(abs:"affordance"+OR+abs:"action+effect"+OR+abs:"clickability"+OR+'
        'abs:"frame+prediction"+OR+abs:"intrinsic+motivation"+OR+'
        'abs:"directed+exploration"+OR+abs:"novelty+search")+AND+'
        '(abs:"reinforcement+learning"+OR+abs:"agent"+OR+abs:"exploration"+OR+'
        'abs:"interactive+environment"+OR+abs:"ARC")&start=0&max_results=8'
        "&sortBy=submittedDate&sortOrder=descending"
    ),
    (
        "http://export.arxiv.org/api/query?search_query="
        '(abs:"active+inference"+OR+abs:"free+energy"+OR+'
        'abs:"free+energy+principle"+OR+abs:"predictive+coding"+OR+'
        'abs:"world+model")+AND+'
        '(abs:"LLM"+OR+abs:"language+model"+OR+abs:"reasoning")'
        "&start=0&max_results=8&sortBy=submittedDate&sortOrder=descending"
    ),
]
S2_QUERIES = [
    "object-centric world model interactive agent transition model planning",
    "hypothesis driven active probing active learning reinforcement learning agents",
]
S2_ARXIV_IDS = [
    "2503.06170",
    "2401.08577",
    "2601.06604",
    "2606.08775",
    "2408.11816",
    "2511.02225",
    "2502.07600",
    "2508.19828",
    "2309.08477",
]
WEBSEARCH_WEBFETCH_TOP_SOURCES = [
    "https://arxiv.org/abs/2410.08822",
    "https://arxiv.org/abs/2511.02225",
    "https://arxiv.org/abs/2601.06604",
    "https://arxiv.org/abs/2511.06136",
    "https://arxiv.org/abs/2307.02427",
    "https://arxiv.org/abs/2210.13455",
    "https://arxiv.org/abs/2506.01876",
    "https://arxiv.org/abs/2309.08477",
]
DEFAULT_PRECONDITIONS_CHECKED = {
    "agents_md_read": True,
    "codex_md_read": True,
    "network_hf_models_reachable": True,
    "sweep_clusters_help_ok": True,
    "exp4697_artifact_read": True,
    "exp4697_note_read": True,
    "exp4700_artifact_read": True,
    "exp4701_artifact_read": True,
    "arc_executable_world_model_read": True,
    "research_studying_read": True,
    "research_references_read": True,
    "sweep_clusters_used": True,
    "sweep_clusters_urls": SWEEP_CLUSTER_URLS,
    "sweep_semscholar_used": True,
    "sweep_semscholar_queries": S2_QUERIES,
    "sweep_semscholar_arxiv_ids": S2_ARXIV_IDS,
    "arxiv_http_200_verified_ids": [
        f"https://arxiv.org/abs/{source_id}" for source_id in sorted(REQUIRED_SOURCE_IDS)
    ],
    "websearch_webfetch_top_sources": WEBSEARCH_WEBFETCH_TOP_SOURCES,
    "deep_research_invoked": False,
    "live_llm_inference": False,
    "training_launched": False,
    "model_load": False,
    "leaderboard_submission": False,
    "ops_docs_modified": False,
    "research_conductor_modified": False,
}

RESIDUAL_SCOPE = (
    "A1 residual: object_centric_perception_no_new_level_residual_"
    "offpath_calibration_insufficient, with deployable object-centric coverage still "
    "not banking a level; A2 residual: amortized_prior_go_explore_no_coverage_gain_"
    "residual_logged, with candidate_generation_coverage_with_prior equal to the "
    "no-prior baseline. The scoped fallback is the structured-world-model / "
    "active-probing next wall: induce an executable object-relational transition "
    "model at runtime, plan inside it, and run targeted probes that confirm or "
    "refute explicit mechanic hypotheses before spending more live actions."
)
DEFAULT_METHODS_MAPPED = [
    {
        "method": "Factored object-relational executable transition model",
        "source_ids": ["2511.02225", "2410.08822", "2307.02427"],
        "track": "factored_object_relational_executable_world_model",
        "residual_scope": RESIDUAL_SCOPE,
        "implement_cost_over_current_stack": (
            "high: lift A1 object slots and relations into typed transition factors, "
            "extend arc_executable_world_model beyond full-grid exact matching into "
            "held-out object/interaction trust, and let A2 traces seed factor induction "
            "rather than only ranking first-contact actions."
        ),
        "maps_to_current_stack": (
            "live E3 explorer uses the induced factors as its planning substrate; "
            "arc_executable_world_model becomes a product of object and interaction "
            "rules instead of one monolithic grid engine; A1 object-centric perception "
            "supplies slots and relations; A2 amortized prior plus Go-Explore supplies "
            "replayable prefixes and action-effect evidence."
        ),
        "fails_when": (
            "object slots drift under off-path interactions, interaction factors alias "
            "hidden registers, or the trusted-factor ledger overfits short public prefixes "
            "and produces plans that fail on live transitions."
        ),
        "roadmap_candidate": FLAGGED_FOR_NEXT_ROADMAP[0],
    },
    {
        "method": "Object-model MCTS with epistemic probe planning",
        "source_ids": ["2601.06604", "2210.13455"],
        "track": "object_model_mcts_with_epistemic_probe_planning",
        "residual_scope": RESIDUAL_SCOPE,
        "implement_cost_over_current_stack": (
            "medium-high: replace the current bounded BFS plan_in_model fallback with "
            "an MCTS planner over the object-relational model, propagate model uncertainty "
            "through rollouts, and allocate live probes to high-value uncertain branches "
            "before executing a candidate solution prefix."
        ),
        "maps_to_current_stack": (
            "live E3 explorer asks MCTS for both solve actions and probe actions; "
            "arc_executable_world_model supplies the rollout engine and trust weights; "
            "A1 object-centric perception defines object graph states; A2 amortized prior "
            "plus Go-Explore returns to archived cells before testing uncertain branches."
        ),
        "fails_when": (
            "uncertainty is uncalibrated, model errors compound over long rollouts, the "
            "branching factor remains grid-scale rather than object-scale, or live action "
            "budgets cannot afford enough confirmation probes."
        ),
        "roadmap_candidate": FLAGGED_FOR_NEXT_ROADMAP[1],
    },
    {
        "method": "Hypothesis-driven active probe loop",
        "source_ids": ["2506.01876", "2309.08477"],
        "track": "hypothesis_driven_active_probe_loop",
        "residual_scope": RESIDUAL_SCOPE,
        "implement_cost_over_current_stack": (
            "medium: add an explicit mechanic-hypothesis table, synthesize discriminating "
            "probe actions from the current object model, update posterior support after "
            "each observed transition, and expose stop/continue decisions to the live E3 "
            "explorer before it commits to a solve plan."
        ),
        "maps_to_current_stack": (
            "live E3 explorer alternates perceive -> hypothesize -> test -> refine; "
            "arc_executable_world_model predicts each hypothesis' transition outcome; "
            "A1 object-centric perception grounds the hypothesis predicates; A2 amortized "
            "prior plus Go-Explore supplies candidate probes and replayable reset points."
        ),
        "fails_when": (
            "the hypothesis class omits the true mechanic, probe outcomes are not "
            "distinguishable at logical-grid resolution, or the agent spends its action "
            "budget identifying a rule that is not sufficient for level completion."
        ),
        "roadmap_candidate": FLAGGED_FOR_NEXT_ROADMAP[2],
    },
    {
        "method": "Latent-drift and policy-breakage guardrails for object world models",
        "source_ids": ["2511.06136"],
        "track": "object_world_model_policy_breakage_guardrails",
        "residual_scope": RESIDUAL_SCOPE,
        "implement_cost_over_current_stack": (
            "low-medium: add held-out off-path drift diagnostics, per-factor rejection "
            "reasons, and plan invalidation when object latents or relations shift under "
            "multi-object interactions."
        ),
        "maps_to_current_stack": (
            "live E3 explorer refuses brittle plans when drift rises; "
            "arc_executable_world_model records why factors were rejected; A1 "
            "object-centric perception supplies latent stability checks; A2 amortized "
            "prior plus Go-Explore collects the off-path transitions that expose breakage."
        ),
        "fails_when": (
            "the drift metric is too conservative and rejects every useful induced model, "
            "or too permissive and lets visually plausible but causally wrong object "
            "rollouts pass into execution."
        ),
        "roadmap_candidate": "guardrail_for_v434: prevent object-model planning false positives",
    },
]

STUDYING_SECTION_START = "<!-- EXP4709-STRUCTURED-WORLD-MODEL-SOTA-START -->"
STUDYING_SECTION_END = "<!-- EXP4709-STRUCTURED-WORLD-MODEL-SOTA-END -->"
STUDYING_SECTION = f"""{STUDYING_SECTION_START}
## 2026-06-25 Exp 4709 - .434 structured-world-model SOTA ingestion - INGESTED

**Status:** INGESTED into `{NOTE_RELATIVE_PATH}`.

**Filtered track:** fallback beyond `.433` A1/A2. A1 closed with
`{A1_RESIDUAL}` and A2 closed with `{A2_RESIDUAL}`. The next wall is the
`{NEXT_WALL}`: the explorer needs an induced object-relational transition model
that it can plan in, plus targeted probes that confirm or refute mechanic
hypotheses.

**Preconditions:** Hugging Face model API reachability returned `net_ok`.
`scripts/sweep_clusters.py --help` exited cleanly. `scripts/sweep_clusters.py`
emitted the ARC neural-guided-search, action-effect, and world-model cluster
URLs. `scripts/sweep_semscholar.py` returned object-centric and
active-hypothesis-testing arXiv IDs. Low-concurrency WebSearch/WebFetch plus
direct arXiv HTTP checks verified arXiv:2410.08822, arXiv:2511.02225,
arXiv:2601.06604, arXiv:2511.06136, arXiv:2307.02427, arXiv:2210.13455,
arXiv:2506.01876, and arXiv:2309.08477. `/deep-research` was not invoked.

**Methods marked ingested:** factored object-relational executable transition
model, object-model MCTS with epistemic probe planning, hypothesis-driven
active probe loop, and object-world-model drift guardrails.

flagged_for_v434: factored_object_relational_executable_world_model
(arXiv:2511.02225 + arXiv:2410.08822 + arXiv:2307.02427)

flagged_for_v434: object_model_mcts_with_epistemic_probe_planning
(arXiv:2601.06604 + arXiv:2210.13455)

flagged_for_v434: hypothesis_driven_active_probe_loop
(arXiv:2506.01876 + arXiv:2309.08477)

**Bottom line for .434:** build the factored object-relational executable
world model first, then use object-model MCTS and active probes to decide which
uncertain mechanics deserve live actions. Keep the drift guardrail as the
failure detector so object-centric perception does not create a false sense of
control.
{STUDYING_SECTION_END}
"""


def build_artifact(
    *,
    methods_mapped: Sequence[JsonMap] = DEFAULT_METHODS_MAPPED,
    citations_verified: JsonMap = CITATIONS_VERIFIED,
    preconditions_checked: JsonMap = DEFAULT_PRECONDITIONS_CHECKED,
    flagged_for_next_roadmap: Sequence[str] = FLAGGED_FOR_NEXT_ROADMAP,
    honest_verdict: str = HONEST_VERDICT,
) -> dict[str, object]:
    """Build and validate the REQ-ARC-WMTE-4709 mapping artifact."""

    artifact: dict[str, object] = {
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "deep_research_not_used": True,
        "methods_mapped": [dict(method) for method in methods_mapped],
        "citations_verified": {
            source_id: dict(citation) for source_id, citation in citations_verified.items()
        },
        "flagged_for_next_roadmap": list(flagged_for_next_roadmap),
        "note_path": NOTE_RELATIVE_PATH,
        "preconditions_checked": dict(preconditions_checked),
        "random_seed": RANDOM_SEED,
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: JsonMap) -> None:
    """Validate the artifact so uncited .434 method claims fail closed."""

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
        raise ValueError("note_path must point at the 2026-06-25 structured-world-model note")
    if artifact["deep_research_not_used"] is not True:
        raise ValueError("deep_research_not_used must be true")
    if artifact["random_seed"] != RANDOM_SEED:
        raise ValueError("random_seed must be the experiment id")

    citations = artifact["citations_verified"]
    if not isinstance(citations, Mapping) or set(citations) != REQUIRED_SOURCE_IDS:
        raise ValueError("citations_verified must exactly cover the required arXiv IDs")
    for source_id, citation in citations.items():
        if not isinstance(citation, Mapping) or set(citation) != REQUIRED_CITATION_FIELDS:
            raise ValueError("each citation must contain exactly title, url, and http_status")
        if citation["url"] != f"https://arxiv.org/abs/{source_id}":
            raise ValueError("citation url must match the arXiv ID")
        if citation["http_status"] != 200:
            raise ValueError("each citation http_status must be 200")
        if not citation["title"]:
            raise ValueError("each citation title must be non-empty")

    methods = artifact["methods_mapped"]
    if not isinstance(methods, Sequence) or isinstance(methods, str | bytes) or not 3 <= len(methods) <= 5:
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
        if not isinstance(stack, str) or not all(
            phrase in stack
            for phrase in (
                "live E3 explorer",
                "arc_executable_world_model",
                "A1 object-centric perception",
                "A2 amortized prior plus Go-Explore",
            )
        ):
            raise ValueError(
                "methods must map to live E3 explorer, arc_executable_world_model, "
                "A1 object-centric perception, and A2 amortized prior plus Go-Explore"
            )
        residual_scope = method["residual_scope"]
        if not isinstance(residual_scope, str) or not all(
            phrase in residual_scope for phrase in (A1_RESIDUAL, A2_RESIDUAL, NEXT_WALL)
        ):
            raise ValueError("methods must state the .433 residuals and next wall")
        if not method["implement_cost_over_current_stack"]:
            raise ValueError("each method needs implement_cost_over_current_stack")
        if not method["fails_when"]:
            raise ValueError("each method needs fails_when")

    roadmap = artifact["flagged_for_next_roadmap"]
    if not isinstance(roadmap, Sequence) or isinstance(roadmap, str | bytes) or not all(
        isinstance(item, str) and "flagged_for_v434" in item for item in roadmap
    ):
        raise ValueError("flagged_for_next_roadmap must contain .434 flagged_for_v434 items")

    preconditions = artifact["preconditions_checked"]
    if not isinstance(preconditions, Mapping) or set(preconditions) != REQUIRED_PRECONDITION_FIELDS:
        raise ValueError("preconditions_checked must match the required schema")
    if preconditions["network_hf_models_reachable"] is not True:
        raise ValueError("network precondition must be true")
    if preconditions["sweep_clusters_help_ok"] is not True:
        raise ValueError("sweep_clusters precondition must be true")
    if preconditions["deep_research_invoked"] is not False:
        raise ValueError("deep-research must not be invoked")
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
    required_phrases = (
        "SOTA -> .434 structured-world-model mapping",
        "Bottom line for the .434 roadmap",
        "live E3 explorer",
        "arc_executable_world_model",
        "A1 object-centric perception",
        "A2 amortized prior plus Go-Explore",
        A1_RESIDUAL,
        A2_RESIDUAL,
        NEXT_WALL,
        "flagged_for_v434",
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
    return f"""# Structured-world-model and active-probing SOTA ingestion 2026-06-25

```json
{artifact_json}
```

## Fresh-pass provenance

Read `AGENTS.md`, `CODEX.md`,
`results/experiment_4697_sota_ingestion_amortized_exploration.json`,
`docs/research-notes/amortized-exploration-sota-ingestion-2026-06-24.md`,
`results/experiment_4700_object_centric_perception_proposal_live.json`,
`results/experiment_4701_amortized_exploration_prior_go_explore_live.json`,
`python/carnot/agentic/arc_executable_world_model.py`, `research-studying.md`,
and `research-references.md`. A1 closed with `{A1_RESIDUAL}`: deployable
object-centric proposal coverage improved, but no live new level was banked.
A2 closed with `{A2_RESIDUAL}`: the amortized prior plus Go-Explore archive did
not raise candidate-generation coverage over the no-prior baseline. The .434
scope is therefore the {NEXT_WALL}: if perception and amortized exploration do
not surface the winning prefix, induce a structured executable transition model
and make the explorer plan and probe inside it.

Reliable-channel pass, not `/deep-research`:
- `curl -sf -o /dev/null https://huggingface.co/api/models`
- `.venv/bin/python scripts/sweep_clusters.py --help`
- `.venv/bin/python scripts/sweep_clusters.py 6 --max-results 8`
- `.venv/bin/python scripts/sweep_clusters.py 5 --max-results 8`
- `.venv/bin/python scripts/sweep_clusters.py 3 --max-results 8`
- `.venv/bin/python scripts/sweep_semscholar.py "object-centric world model interactive agent transition model planning" --limit 8`
- `.venv/bin/python scripts/sweep_semscholar.py "hypothesis driven active probing active learning reinforcement learning agents" --limit 8`
- low-concurrency WebSearch/WebFetch of the top structured-world-model and active-probing papers
- direct arXiv HTTP checks for all cited IDs

Direct arXiv HTTP checks returned 200 for arXiv:2410.08822, arXiv:2511.02225,
arXiv:2601.06604, arXiv:2511.06136, arXiv:2307.02427, arXiv:2210.13455,
arXiv:2506.01876, and arXiv:2309.08477. No live LLM inference, no training,
no leaderboard submission, no model load, and no live solve claim were run or
made. `scripts/research_conductor.py`, `ops/changelog.md`, and `ops/status.md`
were not edited by this workflow.

## SOTA -> .434 structured-world-model mapping

## Factored object-relational executable transition model

**Sources:** FIOC-WM, arXiv:2511.02225; SOLD, arXiv:2410.08822; FOCUS,
arXiv:2307.02427.

**Mapping to current stack:** convert A1's connected components, relation
keypoints, and object slots into typed transition factors. Extend
`arc_executable_world_model` from monolithic grid engines and exact-match
verification into a held-out trust ledger over object and interaction effects.
A1 object-centric perception supplies the representation substrate.
A2 amortized prior plus Go-Explore supplies replayable prefixes and action-effect
observations for factor induction.

**Implementation cost over current stack:** high. It requires a new factor
schema, held-out interaction scoring, and a planner that composes trusted
factors without assuming full-grid prediction is perfect.

**Fails when:** object slots drift under off-path interactions, interaction
factors alias hidden registers, or short prefixes overfit public-game mechanics.

## Object-model MCTS with epistemic probe planning

**Sources:** ObjectZero, arXiv:2601.06604; Epistemic MCTS, arXiv:2210.13455.

**Mapping to current stack:** replace the current bounded BFS-only
`plan_in_model` fallback with MCTS over the induced object model. The live E3
explorer asks the planner for both solution actions and probe actions, while
Go-Explore returns to archived states before testing uncertain branches.

**Implementation cost over current stack:** medium-high. The product world
model already exposes an executable engine, but MCTS needs state keys, rollout
budgets, uncertainty propagation, and a live-action policy for when to probe
versus execute.

**Fails when:** uncertainty is uncalibrated, model errors compound over rollout
depth, or the object abstraction fails to reduce the branching factor enough.

## Hypothesis-driven active probe loop

**Sources:** In-Context Pure Explorer, arXiv:2506.01876; MARLA for active
hypothesis testing, arXiv:2309.08477.

**Mapping to current stack:** make the agent maintain explicit hypotheses such
as "clicking a same-color object rewrites a target relation" or "the HUD count
gates level completion." `arc_executable_world_model` predicts outcomes under
each hypothesis, A1 grounds predicates in object slots, and A2/Go-Explore
provide candidate probes and reset points.

**Implementation cost over current stack:** medium. The current live E3 loop
already observes transitions; the missing piece is the explicit
perceive -> hypothesize -> test -> refine table plus targeted probe selection.

**Fails when:** the true mechanic is outside the hypothesis class, probe
outcomes are visually indistinguishable, or the action budget is spent
identifying a rule that is not sufficient to complete the level.

## Latent-drift and policy-breakage guardrails

**Source:** When Object-Centric World Models Meet Policy Learning, arXiv:2511.06136.

**Mapping to current stack:** add held-out off-path drift diagnostics and plan
invalidation so object-centric perception cannot create a false confidence
signal. The live E3 explorer refuses brittle object-model plans when A1 slots
or relations shift under multi-object interactions; `arc_executable_world_model`
records rejected factors and A2/Go-Explore collects the transitions that expose
breakage.

**Implementation cost over current stack:** low-medium. The ledger can be added
beside the existing rejected-factor diagnostics and verifier mismatch artifacts.

**Fails when:** the drift metric rejects every useful induced model or permits
visually plausible but causally wrong rollouts.

## Bottom line for the .434 roadmap

The strongest .434 input is
flagged_for_v434: factored_object_relational_executable_world_model
(arXiv:2511.02225 + arXiv:2410.08822 + arXiv:2307.02427). It attacks the next
wall directly by giving the explorer a structured executable transition model
instead of another proposal prior.

The planning companion is
flagged_for_v434: object_model_mcts_with_epistemic_probe_planning
(arXiv:2601.06604 + arXiv:2210.13455), and the active-learning companion is
flagged_for_v434: hypothesis_driven_active_probe_loop
(arXiv:2506.01876 + arXiv:2309.08477). Together they make the explorer choose
live actions that either solve in the induced model or maximally reduce
uncertainty about the game's mechanic.
"""


RESEARCH_NOTE = _research_note()


def _upsert_section(content: str, section: str, start_marker: str, end_marker: str) -> str:
    section = section.rstrip() + "\n"
    start = content.find(start_marker)
    end = content.find(end_marker)
    if start >= 0 and end >= start:
        return content[:start] + section + content[end + len(end_marker) :].lstrip("\n")
    separator = "" if content.endswith("\n") else "\n"
    return content + separator + "\n" + section


def write_outputs(
    *,
    artifact_path: Path,
    note_path: Path,
    studying_path: Path,
) -> dict[str, object]:
    """Write the stable artifact, research note, and research-studying marker."""

    artifact = build_artifact()
    validate_research_note(RESEARCH_NOTE)

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    note_path.write_text(RESEARCH_NOTE, encoding="utf-8")

    current = studying_path.read_text(encoding="utf-8") if studying_path.exists() else ""
    studying_path.write_text(
        _upsert_section(current, STUDYING_SECTION, STUDYING_SECTION_START, STUDYING_SECTION_END),
        encoding="utf-8",
    )
    return artifact


def main() -> int:
    root = Path(os.environ.get("CARNOT_EXP4709_ROOT", Path(__file__).resolve().parents[2]))
    write_outputs(
        artifact_path=root / RESULT_RELATIVE_PATH,
        note_path=root / NOTE_RELATIVE_PATH,
        studying_path=root / "research-studying.md",
    )
    print(HONEST_VERDICT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
