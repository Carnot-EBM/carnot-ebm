"""Exp 4685 directed-exploration SOTA ingestion.

Spec refs: REQ-ARC-WMTE-4685, SCENARIO-ARC-WMTE-4685.

This is a literature-to-experiment mapping artifact. It targets the .432
fallback where the live agent cannot reliably make a winning L1 trajectory
appear in the candidate distribution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
from typing import Any


JsonMap = Mapping[str, Any]

RESULT_RELATIVE_PATH = "results/experiment_4685_sota_ingestion_directed_exploration.json"
NOTE_RELATIVE_PATH = (
    "docs/research-notes/directed-exploration-sota-ingestion-2026-06-24.md"
)
RANDOM_SEED = 4685
HONEST_VERDICT = "success: sota_ingestion_directed_exploration_mapped"
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
        "exp4673_artifact_read",
        "exp4673_note_read",
        "exp4676_artifact_read",
        "exp4677_artifact_read",
        "research_studying_read",
        "research_references_read",
        "sweep_clusters_used",
        "sweep_clusters_urls",
        "sweep_semscholar_used",
        "sweep_semscholar_queries",
        "sweep_semscholar_arxiv_ids",
        "sweep_semscholar_rate_limited_queries",
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
        "1712.06560",
        "1810.12894",
        "2002.06038",
        "2005.05960",
        "2102.11137",
        "2502.10077",
        "2505.10819",
        "2603.02045",
    }
)
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(
    f"arXiv:{source_id}" for source_id in REQUIRED_SOURCE_IDS
)

L1_FIRST_CONTACT = "l1_first_contact"
A1_RESIDUAL = "value_head_still_not_separating"
A2_RESIDUAL = "experts_overfit_prefix"
FLAGGED_FOR_NEXT_ROADMAP = [
    (
        "flagged_for_v432: controllable_novelty_e3_proposal_policy "
        "(arXiv:2002.06038 + arXiv:1810.12894 + arXiv:2603.02045)"
    ),
    (
        "flagged_for_v432: program_synthesis_action_effect_proposal_filter "
        "(arXiv:2505.10819 + arXiv:2102.11137)"
    ),
]

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": "terminal prefix; success: sota_ingestion_directed_exploration_mapped."
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
            "the strongest method(s) flagged as candidate .432 inputs "
            "(flagged_for_v432) -- closes discover->ingest->plan->experiment."
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
    "1712.06560": {
        "title": (
            "Improving Exploration in Evolution Strategies for Deep Reinforcement "
            "Learning via a Population of Novelty-Seeking Agents"
        ),
        "url": "https://arxiv.org/abs/1712.06560",
        "http_status": 200,
    },
    "1810.12894": {
        "title": "Exploration by Random Network Distillation",
        "url": "https://arxiv.org/abs/1810.12894",
        "http_status": 200,
    },
    "2002.06038": {
        "title": "Never Give Up: Learning Directed Exploration Strategies",
        "url": "https://arxiv.org/abs/2002.06038",
        "http_status": 200,
    },
    "2005.05960": {
        "title": "Planning to Explore via Self-Supervised World Models",
        "url": "https://arxiv.org/abs/2005.05960",
        "http_status": 200,
    },
    "2102.11137": {
        "title": "Program Synthesis Guided Reinforcement Learning for Partially Observed Environments",
        "url": "https://arxiv.org/abs/2102.11137",
        "http_status": 200,
    },
    "2502.10077": {
        "title": "Towards Empowerment Gain through Causal Structure Learning in Model-Based RL",
        "url": "https://arxiv.org/abs/2502.10077",
        "http_status": 200,
    },
    "2505.10819": {
        "title": (
            "PoE-World: Compositional World Modeling with Products of "
            "Programmatic Experts"
        ),
        "url": "https://arxiv.org/abs/2505.10819",
        "http_status": 200,
    },
    "2603.02045": {
        "title": "Expanding LLM Agent Boundaries with Strategy-Guided Exploration",
        "url": "https://arxiv.org/abs/2603.02045",
        "http_status": 200,
    },
}

SWEEP_CLUSTER_URLS = [
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
        '(abs:"neural+guided+search"+OR+abs:"learned+heuristic"+OR+'
        'abs:"value+guided+search"+OR+abs:"program+induction"+OR+'
        'abs:"world+model"+OR+abs:"goal+induction")+AND+'
        '(abs:"planning"+OR+abs:"agent"+OR+abs:"reasoning"+OR+'
        'abs:"reinforcement+learning")&start=0&max_results=8'
        "&sortBy=submittedDate&sortOrder=descending"
    ),
]
S2_QUERIES = [
    "directed exploration intrinsic motivation novelty search empowerment interactive agents",
    "curiosity-driven exploration random network distillation episodic novelty reinforcement learning",
    "program synthesis action model induction interactive agents world models actions effects",
    "action effect prediction affordance induction interactive reinforcement learning program synthesis",
]
WEBSEARCH_WEBFETCH_TOP_SOURCES = [
    "https://arxiv.org/abs/2002.06038",
    "https://arxiv.org/abs/1810.12894",
    "https://arxiv.org/abs/2005.05960",
    "https://arxiv.org/abs/1712.06560",
    "https://arxiv.org/abs/2502.10077",
    "https://arxiv.org/abs/2603.02045",
    "https://arxiv.org/abs/2102.11137",
    "https://arxiv.org/abs/2505.10819",
]
DEFAULT_PRECONDITIONS_CHECKED = {
    "agents_md_read": True,
    "codex_md_read": True,
    "network_hf_models_reachable": True,
    "sweep_clusters_help_ok": True,
    "exp4673_artifact_read": True,
    "exp4673_note_read": True,
    "exp4676_artifact_read": True,
    "exp4677_artifact_read": True,
    "research_studying_read": True,
    "research_references_read": True,
    "sweep_clusters_used": True,
    "sweep_clusters_urls": SWEEP_CLUSTER_URLS,
    "sweep_semscholar_used": True,
    "sweep_semscholar_queries": S2_QUERIES,
    "sweep_semscholar_arxiv_ids": [],
    "sweep_semscholar_rate_limited_queries": S2_QUERIES,
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
    "L1-first-contact wall: A1 reports wall_diagnosis=l1_first_contact and "
    "residual_cause_hypothesis=value_head_still_not_separating with generic "
    "first-win rate 0.04; A2 reports candidate_generation_coverage_factored=0.0 "
    "and residual_bridge_gap=experts_overfit_prefix, so .432 must change the "
    "action-proposal distribution before A1/A2 selection or planning."
)
DEFAULT_METHODS_MAPPED = [
    {
        "method": "Episodic controllable-novelty policy family for L1 first contact",
        "source_ids": ["2002.06038", "1810.12894", "2603.02045"],
        "track": "ngu_rnd_controllable_novelty_e3_proposer",
        "residual_scope": RESIDUAL_SCOPE,
        "implement_cost_over_current_stack": (
            "medium: add a state/action embedding over visible frame deltas and "
            "action-effect features, keep an episodic kNN novelty table plus an "
            "RND-style long-horizon novelty score, and run several exploration "
            "temperatures inside the same live E3 budget."
        ),
        "maps_to_current_stack": (
            "live E3 explorer receives a controllable intrinsic proposal bonus "
            "before blind or value-ranked actions, A1 hierarchical subgoal search "
            "only consumes the discovered first-contact trajectory, and A2 factored "
            "planner audits whether the novelty-selected actions have stable effects."
        ),
        "fails_when": (
            "the embedding treats cosmetic frame changes as controllable novelty, "
            "the kNN table aliases distinct mechanics, novelty repeatedly revisits "
            "non-winning states, or strategy diversity generates language plans that "
            "do not ground to valid ARC actions."
        ),
        "roadmap_candidate": FLAGGED_FOR_NEXT_ROADMAP[0],
    },
    {
        "method": "Plan2Explore-style disagreement frontier sampler with empowerment guard",
        "source_ids": ["2005.05960", "2502.10077"],
        "track": "model_disagreement_empowerment_frontier_sampler",
        "residual_scope": RESIDUAL_SCOPE,
        "implement_cost_over_current_stack": (
            "high: maintain a tiny ensemble over transition/effect predictions from "
            "live E3 traces, score short action sequences by predicted disagreement "
            "and causal controllability, then replay only the top frontier-expanding "
            "sequences through the existing harness."
        ),
        "maps_to_current_stack": (
            "live E3 explorer samples short sequences that are expected to expose "
            "new controllable effects, A1 hierarchical subgoal search is delayed "
            "until those effects create a reachable L1 contact, and A2 factored "
            "planner receives better transition evidence instead of overfit prefixes."
        ),
        "fails_when": (
            "the ensemble is undertrained on only a few public-game transitions, "
            "disagreement is high because of visual noise rather than mechanics, "
            "or empowerment rewards controllability that is unrelated to the L1 win."
        ),
        "roadmap_candidate": FLAGGED_FOR_NEXT_ROADMAP[0],
    },
    {
        "method": "Novelty/QD population over replayable action prefixes",
        "source_ids": ["1712.06560", "1810.12894"],
        "track": "novelty_qd_action_prefix_archive",
        "residual_scope": RESIDUAL_SCOPE,
        "implement_cost_over_current_stack": (
            "medium: keep a MAP-Elites-style archive of replayable action prefixes "
            "using descriptors such as changed-cell topology, object motion class, "
            "HUD/register deltas, and novelty score; mutate prefixes only through "
            "actions that remain valid under live E3 replay."
        ),
        "maps_to_current_stack": (
            "live E3 explorer gets a diversified prefix generator instead of one "
            "depth-first action stream, A1 hierarchical subgoal search uses archive "
            "elites as first-contact candidates, and A2 factored planner checks "
            "whether elite descriptors correspond to reusable action effects."
        ),
        "fails_when": (
            "the behavior descriptors ignore the hidden winning mechanic, archive "
            "mutation destroys replayability, or the method rediscovers diverse "
            "near-misses without inserting the rare winning L1 prefix."
        ),
        "roadmap_candidate": FLAGGED_FOR_NEXT_ROADMAP[0],
    },
    {
        "method": "Strategy-guided exploration for language-action proposal diversity",
        "source_ids": ["2603.02045", "2002.06038"],
        "track": "strategy_guided_language_action_exploration",
        "residual_scope": RESIDUAL_SCOPE,
        "implement_cost_over_current_stack": (
            "medium: generate a small batch of natural-language strategies at mixed "
            "temperatures, condition the action proposer on each strategy, and "
            "reflect only on replayed outcomes so the strategy pool is grounded in "
            "observed live E3 transitions."
        ),
        "maps_to_current_stack": (
            "live E3 explorer explores strategy-conditioned action streams, A1 "
            "hierarchical subgoal search is reused only after a strategy finds L1 "
            "contact, and A2 factored planner labels which strategies produced "
            "trustworthy action effects."
        ),
        "fails_when": (
            "strategy text becomes another ungrounded subgoal layer, outcome "
            "reflection rewards plausible explanations rather than replayed state "
            "change, or mixed-temperature sampling spends the budget on duplicate "
            "mechanics."
        ),
        "roadmap_candidate": FLAGGED_FOR_NEXT_ROADMAP[0],
    },
    {
        "method": "Program-synthesis action-effect induction for proposal pruning",
        "source_ids": ["2505.10819", "2102.11137"],
        "track": "program_synthesis_action_effect_proposal_filter",
        "residual_scope": RESIDUAL_SCOPE,
        "implement_cost_over_current_stack": (
            "medium-high: synthesize small per-game action->effect programs from "
            "observed prefixes, reject programs that fail held-out transitions, "
            "and use surviving programs to propose mechanically relevant clicks or "
            "key actions rather than blind spatial sweeps."
        ),
        "maps_to_current_stack": (
            "live E3 explorer filters primitive proposals through induced action "
            "effects, A1 hierarchical subgoal search receives mechanically reachable "
            "first-contact prefixes, and A2 factored planner is narrowed to trusted "
            "program factors instead of composing prefix-overfit experts."
        ),
        "fails_when": (
            "the program overfits the first few prefixes, held-out transition trust "
            "is too sparse to reject brittle rules, hidden game state determines the "
            "effect, or the induced program explains effects but still cannot target "
            "the winning action."
        ),
        "roadmap_candidate": FLAGGED_FOR_NEXT_ROADMAP[1],
    },
]

STUDYING_SECTION_START = "<!-- EXP4685-DIRECTED-EXPLORATION-SOTA-START -->"
STUDYING_SECTION_END = "<!-- EXP4685-DIRECTED-EXPLORATION-SOTA-END -->"
STUDYING_SECTION = f"""{STUDYING_SECTION_START}
## 2026-06-24 Exp 4685 - .432 directed-exploration SOTA ingestion - INGESTED

**Status:** INGESTED into `{NOTE_RELATIVE_PATH}`.

**Filtered track:** fallback beyond the `.431` A1 hierarchical subgoal search
and A2 PoE-World planner. A1 closed at `wall_diagnosis=l1_first_contact` with
`value_head_still_not_separating`; A2 closed with
`candidate_generation_coverage_factored=0.0` and `experts_overfit_prefix`.
The live gap is now action-proposal coverage: make a winning L1 trajectory
appear before A1/A2 can select, decompose, or plan over it.

**Preconditions:** Hugging Face model API reachability returned `net_ok`.
`scripts/sweep_clusters.py --help` exited cleanly. `scripts/sweep_clusters.py`
emitted the ARC exploration and neural-guided-search cluster URLs.
`scripts/sweep_semscholar.py` returned HTTP 429 for the four focused queries
and no S2-only source was promoted. Low-concurrency WebSearch/WebFetch plus
direct arXiv HTTP checks verified arXiv:2002.06038, arXiv:1810.12894,
arXiv:2005.05960, arXiv:1712.06560, arXiv:2502.10077, arXiv:2603.02045,
arXiv:2102.11137, and arXiv:2505.10819. `/deep-research` was not invoked.

**Methods marked ingested:** episodic controllable-novelty policy family,
Plan2Explore-style disagreement plus empowerment, novelty/QD replayable action
prefix archives, strategy-guided language-action exploration, and
program-synthesis action-effect proposal filtering.

flagged_for_v432: controllable_novelty_e3_proposal_policy
(arXiv:2002.06038 + arXiv:1810.12894 + arXiv:2603.02045)

flagged_for_v432: program_synthesis_action_effect_proposal_filter
(arXiv:2505.10819 + arXiv:2102.11137)

**Bottom line for .432:** build the controllable-novelty proposal policy first
because it directly attacks the L1-first-contact distribution gap. Add the
program-synthesis action-effect filter as the second arm when enough trusted
prefix transitions exist to avoid the A2 `experts_overfit_prefix` failure.
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
    """Build and validate the REQ-ARC-WMTE-4685 mapping artifact."""

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
    """Validate the artifact so uncited .432 directed-exploration claims fail closed."""

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
        raise ValueError("note_path must point at the 2026-06-24 directed-exploration note")
    if artifact["deep_research_not_used"] is not True:
        raise ValueError("deep_research_not_used must be true")
    if artifact["random_seed"] != RANDOM_SEED:
        raise ValueError("random_seed must be the bare integer 4685")

    citations = artifact["citations_verified"]
    if not isinstance(citations, dict) or set(citations) != REQUIRED_SOURCE_IDS:
        raise ValueError("citations_verified must include exactly the required source IDs")
    for source_id, citation in citations.items():
        if not isinstance(citation, dict) or set(citation) != REQUIRED_CITATION_FIELDS:
            raise ValueError("each citation must contain exactly title, url, and http_status")
        if citation["url"] != f"https://arxiv.org/abs/{source_id}":
            raise ValueError(f"citation url must match arXiv source ID {source_id}")
        if citation["http_status"] != 200:
            raise ValueError("citation http_status must be 200")
        if not isinstance(citation["title"], str) or not citation["title"].strip():
            raise ValueError("citation title must be a non-empty string")

    methods = artifact["methods_mapped"]
    if not isinstance(methods, list) or not 3 <= len(methods) <= 5:
        raise ValueError("methods_mapped must contain three to five methods")
    for method in methods:
        if not isinstance(method, dict) or set(method) != REQUIRED_METHOD_FIELDS:
            raise ValueError("each method must contain the exact required method fields")
        source_ids = method["source_ids"]
        if not isinstance(source_ids, list) or not source_ids:
            raise ValueError("method source_ids must be a non-empty list")
        if any(source_id not in citations for source_id in source_ids):
            raise ValueError("method source_ids must all have verified citations")
        mapping = method["maps_to_current_stack"]
        if (
            not isinstance(mapping, str)
            or "live E3 explorer" not in mapping
            or "A1 hierarchical subgoal search" not in mapping
            or "A2 factored planner" not in mapping
        ):
            raise ValueError(
                "method mapping must name live E3 explorer, "
                "A1 hierarchical subgoal search, and A2 factored planner"
            )
        residual_scope = method["residual_scope"]
        if (
            not isinstance(residual_scope, str)
            or L1_FIRST_CONTACT not in residual_scope
            or A1_RESIDUAL not in residual_scope
            or A2_RESIDUAL not in residual_scope
        ):
            raise ValueError("method residual_scope must name the L1-first-contact residuals")
        for field in REQUIRED_METHOD_FIELDS - {"source_ids"}:
            value = method[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"method {field} must be a non-empty string")

    flagged = artifact["flagged_for_next_roadmap"]
    if flagged != FLAGGED_FOR_NEXT_ROADMAP or not all(
        "flagged_for_v432" in item for item in flagged
    ):
        raise ValueError("flagged_for_next_roadmap must name the .432 candidates")

    preconditions = artifact["preconditions_checked"]
    if not isinstance(preconditions, dict) or set(preconditions) != REQUIRED_PRECONDITION_FIELDS:
        raise ValueError("preconditions_checked must contain the exact required fields")
    if preconditions["network_hf_models_reachable"] is not True:
        raise ValueError("network reachability precondition must be true")
    if preconditions["sweep_clusters_help_ok"] is not True:
        raise ValueError("sweep_clusters helper precondition must be true")
    if preconditions["deep_research_invoked"] is not False:
        raise ValueError("deep-research must not be invoked")
    if preconditions["research_conductor_modified"] is not False:
        raise ValueError("research_conductor.py must not be modified")
    if preconditions["ops_docs_modified"] is not False:
        raise ValueError("ops docs must not be modified by Exp 4685")


def artifact_from_note(markdown: str) -> dict[str, object]:
    """Extract and validate the machine-readable JSON block from a note."""

    marker = "```json"
    start = markdown.find(marker)
    if start == -1:
        raise ValueError("research note missing machine-readable JSON block")
    json_start = markdown.find("\n", start) + 1
    json_end = markdown.find("```", json_start)
    if json_end == -1:
        raise ValueError("research note missing machine-readable JSON block terminator")
    artifact = json.loads(markdown[json_start:json_end])
    validate_artifact(artifact)
    return artifact


def validate_research_note(markdown: str) -> None:
    """Check that the paired note maps verified sources to .432 work."""

    artifact_from_note(markdown)
    required_phrases = (
        "Fresh-pass provenance",
        "SOTA -> .432 directed-exploration mapping",
        "live E3 explorer",
        "A1 hierarchical subgoal search",
        "A2 factored planner",
        "l1_first_contact",
        "value_head_still_not_separating",
        "experts_overfit_prefix",
        "Bottom line for the .432 roadmap",
        "flagged_for_v432",
        "Never Give Up",
        "Plan2Explore",
        "PoE-World",
        "No live LLM inference",
        "No training",
        "No leaderboard submission",
    )
    missing_phrases = [phrase for phrase in required_phrases if phrase not in markdown]
    if missing_phrases:
        raise ValueError(f"research note missing required phrase(s): {missing_phrases}")
    json_block_end = markdown.find("```\n\n## Fresh-pass provenance")
    prose = markdown[json_block_end:] if json_block_end != -1 else markdown
    missing_sources = [source for source in NOTE_REQUIRED_SOURCE_CITATIONS if source not in prose]
    if missing_sources:
        raise ValueError(f"research note missing verified source citations: {missing_sources}")


def _make_research_note(artifact: JsonMap) -> str:
    artifact_json = json.dumps(artifact, indent=2, sort_keys=True)
    return f"""# Directed-exploration SOTA ingestion 2026-06-24

```json
{artifact_json}
```

## Fresh-pass provenance

Read `AGENTS.md`, `CODEX.md`,
`results/experiment_4673_sota_ingestion_structural_deepening.json`,
`docs/research-notes/structural-deepening-sota-ingestion-2026-06-24.md`,
`results/experiment_4676_hierarchical_subgoal_search_live.json`,
`results/experiment_4677_poe_world_factored_subgoal_planner.json`,
`research-studying.md`, and `research-references.md`. The current stack is the
live E3 explorer with A1 hierarchical subgoal search and A2 factored planner
available only after a candidate trajectory exists. A1 closed with
`wall_diagnosis=l1_first_contact`, `value_head_still_not_separating`, and
generic first-win rate 0.04. A2 closed with
`candidate_generation_coverage_factored=0.0` and `experts_overfit_prefix`.
The `.432` scope is therefore not another selector; it is directed proposal
coverage so a winning L1 trajectory enters the pool.

Reliable-channel pass, not `/deep-research`:
- `curl -sf -o /dev/null https://huggingface.co/api/models`
- `.venv/bin/python scripts/sweep_clusters.py --help`
- `.venv/bin/python scripts/sweep_clusters.py 5 --max-results 8`
- `.venv/bin/python scripts/sweep_clusters.py 6 --max-results 8`
- `.venv/bin/python scripts/sweep_semscholar.py - --limit 8` with four focused queries
- low-concurrency WebSearch/WebFetch of the top directed-exploration and program-synthesis action-model papers
- direct arXiv HTTP checks for all cited IDs

Semantic Scholar returned HTTP 429 for the focused queries and no S2-only
source was promoted. Direct arXiv HTTP checks returned 200 for
arXiv:2002.06038, arXiv:1810.12894, arXiv:2005.05960, arXiv:1712.06560,
arXiv:2502.10077, arXiv:2603.02045, arXiv:2102.11137, and arXiv:2505.10819.
No live LLM inference, No training, No leaderboard submission, no model load,
and no live solve claim were run or made. `scripts/research_conductor.py`,
`ops/changelog.md`, and `ops/status.md` were not edited by this workflow.

## SOTA -> .432 directed-exploration mapping

## Episodic controllable-novelty policy family for L1 first contact

**Sources:** Never Give Up, arXiv:2002.06038; Exploration by Random Network
Distillation, arXiv:1810.12894; Strategy-Guided Exploration, arXiv:2603.02045.

**Mapping to current stack:** the live E3 explorer should score proposed
actions by controllable novelty before A1 or A2 ever sees a trajectory. A1
hierarchical subgoal search consumes the discovered first-contact trace only
after the proposal policy finds it. A2 factored planner audits whether the
novelty-selected actions express stable effects.

**Implementation cost over current stack:** medium. Add an embedding over
visible deltas/action effects, an episodic kNN novelty table, an RND-style
lifelong novelty score, and a small family of exploration temperatures under
the same replay gates.

**Fails when:** the embedding rewards cosmetic changes, KNN aliases mechanics,
or language strategy diversity produces plans that do not ground to valid ARC
actions.

## Plan2Explore-style disagreement frontier sampler with empowerment guard

**Sources:** Plan2Explore, arXiv:2005.05960; empowerment through causal
learning, arXiv:2502.10077.

**Mapping to current stack:** the live E3 explorer samples short sequences with
high predicted future novelty and controllability. A1 hierarchical subgoal
search is delayed until those sequences reveal an L1-contact candidate. A2
factored planner receives better transition evidence instead of composing
prefix-overfit experts.

**Implementation cost over current stack:** high. Maintain a small transition
ensemble, score short action sequences by predicted disagreement and causal
control, and replay only the top frontier-expanding sequences.

**Fails when:** transition data is too sparse, ensemble disagreement tracks
visual noise, or empowerment finds controllable states unrelated to the win.

## Novelty/QD population over replayable action prefixes

**Sources:** novelty-seeking ES/QD, arXiv:1712.06560; RND, arXiv:1810.12894.

**Mapping to current stack:** the live E3 explorer gets a replayable prefix
archive instead of a single depth-first stream. A1 hierarchical subgoal search
uses archive elites as first-contact candidates. A2 factored planner checks
whether elite descriptors correspond to reusable action effects.

**Implementation cost over current stack:** medium. Keep behavior descriptors
for changed-cell topology, object motion, HUD/register deltas, and novelty;
mutate only prefixes that survive the replay gate.

**Fails when:** descriptors miss the hidden mechanic, mutation breaks
replayability, or the archive diversifies near-misses without inserting the
rare winning L1 prefix.

## Strategy-guided exploration for language-action proposal diversity

**Sources:** Strategy-Guided Exploration, arXiv:2603.02045; Never Give Up,
arXiv:2002.06038.

**Mapping to current stack:** the live E3 explorer runs a small batch of
strategy-conditioned action streams. A1 hierarchical subgoal search starts only
after one strategy discovers L1 contact. A2 factored planner labels which
strategies produced trustworthy effects.

**Implementation cost over current stack:** medium. Generate concise strategy
sketches at mixed temperatures, condition action proposal on each strategy, and
reflect only on replayed outcomes.

**Fails when:** strategy text becomes another ungrounded subgoal layer, outcome
reflection rewards plausible explanation instead of state change, or the batch
duplicates one mechanic.

## Program-synthesis action-effect induction for proposal pruning

**Sources:** PoE-World, arXiv:2505.10819; model predictive program synthesis,
arXiv:2102.11137.

**Mapping to current stack:** the live E3 explorer filters primitive proposals
through per-game action-effect programs. A1 hierarchical subgoal search receives
mechanically reachable first-contact prefixes. A2 factored planner narrows to
trusted program factors instead of repeating the `experts_overfit_prefix`
failure.

**Implementation cost over current stack:** medium-high. Synthesize small
action->effect programs, reject programs that fail held-out transitions, and
use surviving programs to propose relevant clicks or key actions rather than
blind sweeps.

**Fails when:** the program overfits early prefixes, held-out transition trust
is too sparse, hidden state determines the effect, or the induced program
explains effects without targeting the winning action.

## Bottom line for the .432 roadmap

1. Build `flagged_for_v432: controllable_novelty_e3_proposal_policy` first.
   It directly attacks the `l1_first_contact` distribution gap: the current
   explorer reaches L1 on only 1/25 games, so the proposal distribution must
   be widened toward controllable novelty before selection helps.
2. Keep `flagged_for_v432: program_synthesis_action_effect_proposal_filter` as
   the second arm. It is the program-synthesis answer to blind clicks, but it
   must include held-out transition rejection because A2 already exposed the
   `experts_overfit_prefix` failure mode.
3. Treat Plan2Explore/empowerment and novelty/QD archives as support arms when
   the lightweight transition evidence is sufficient; both are valuable, but
   they can chase controllable non-wins if promoted without replay gates.
"""


DEFAULT_ARTIFACT = build_artifact()
RESEARCH_NOTE = _make_research_note(DEFAULT_ARTIFACT)
validate_research_note(RESEARCH_NOTE)


def _with_studying_section(existing: str) -> str:
    if STUDYING_SECTION_START in existing and STUDYING_SECTION_END in existing:
        before, rest = existing.split(STUDYING_SECTION_START, 1)
        _, after = rest.split(STUDYING_SECTION_END, 1)
        return before.rstrip() + "\n\n" + STUDYING_SECTION.rstrip() + after
    return existing.rstrip() + "\n\n" + STUDYING_SECTION.rstrip() + "\n"


def write_outputs(
    *,
    artifact_path: Path,
    note_path: Path,
    studying_path: Path,
) -> dict[str, object]:
    """Write the JSON artifact, markdown note, and studying queue update."""

    artifact = build_artifact()
    note = _make_research_note(artifact)
    validate_research_note(note)

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    note_path.write_text(note + "\n", encoding="utf-8")
    studying_path.write_text(
        _with_studying_section(studying_path.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    return artifact


def main() -> int:
    """Write the default Exp 4685 deliverables under the repository root."""

    repo_root = Path(os.environ.get("CARNOT_EXP4685_ROOT", Path(__file__).resolve().parents[2]))
    artifact = write_outputs(
        artifact_path=repo_root / RESULT_RELATIVE_PATH,
        note_path=repo_root / NOTE_RELATIVE_PATH,
        studying_path=repo_root / "research-studying.md",
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
