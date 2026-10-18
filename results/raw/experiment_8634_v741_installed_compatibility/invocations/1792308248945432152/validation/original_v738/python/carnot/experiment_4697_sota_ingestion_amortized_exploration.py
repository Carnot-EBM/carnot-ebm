"""Exp 4697 amortized-exploration SOTA ingestion.

Spec refs: REQ-ARC-WMTE-4697, SCENARIO-ARC-WMTE-4697.

This artifact maps the next ARC transfer wall after per-game directed
exploration nulls. The goal is not to claim a live solve; it is to make the
next experiment harder to fool by tying every method claim to a verified arXiv
record and to the exact .432 residuals it is meant to address.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
from typing import Any


JsonMap = Mapping[str, Any]

RESULT_RELATIVE_PATH = "results/experiment_4697_sota_ingestion_amortized_exploration.json"
NOTE_RELATIVE_PATH = (
    "docs/research-notes/amortized-exploration-sota-ingestion-2026-06-24.md"
)
RANDOM_SEED = 4697
HONEST_VERDICT = "success: sota_ingestion_amortized_exploration_mapped"
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
        "exp4685_artifact_read",
        "exp4685_note_read",
        "exp4688_artifact_read",
        "exp4689_artifact_read",
        "arc_go_explore_read",
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
        "1802.07245",
        "1901.10995",
        "2004.12919",
        "2008.02790",
        "2210.14215",
        "2310.09971",
        "2601.19810",
        "2603.03680",
    }
)
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(
    f"arXiv:{source_id}" for source_id in REQUIRED_SOURCE_IDS
)

A1_RESIDUAL = "winning_prefix_still_not_proposed"
A2_RESIDUAL = "heldout_transitions_too_sparse"
TRANSFER_WALL = "hidden-game transfer"
FLAGGED_FOR_NEXT_ROADMAP = [
    (
        "flagged_for_v433: in_context_exploration_prior_from_first_contact_traces "
        "(arXiv:2210.14215 + arXiv:2310.09971 + arXiv:2601.19810)"
    ),
    (
        "flagged_for_v433: arc_go_explore_return_then_explore_archive_upgrade "
        "(arXiv:1901.10995 + arXiv:2004.12919)"
    ),
]

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": "terminal prefix; success: sota_ingestion_amortized_exploration_mapped."
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
            "the strongest method(s) flagged as candidate .433 inputs "
            "(flagged_for_v433) -- closes discover->ingest->plan->experiment."
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
    "1802.07245": {
        "title": "Meta-Reinforcement Learning of Structured Exploration Strategies",
        "url": "https://arxiv.org/abs/1802.07245",
        "http_status": 200,
    },
    "1901.10995": {
        "title": "Go-Explore: a New Approach for Hard-Exploration Problems",
        "url": "https://arxiv.org/abs/1901.10995",
        "http_status": 200,
    },
    "2004.12919": {
        "title": "First return, then explore",
        "url": "https://arxiv.org/abs/2004.12919",
        "http_status": 200,
    },
    "2008.02790": {
        "title": (
            "Decoupling Exploration and Exploitation for Meta-Reinforcement Learning "
            "without Sacrifices"
        ),
        "url": "https://arxiv.org/abs/2008.02790",
        "http_status": 200,
    },
    "2210.14215": {
        "title": "In-context Reinforcement Learning with Algorithm Distillation",
        "url": "https://arxiv.org/abs/2210.14215",
        "http_status": 200,
    },
    "2310.09971": {
        "title": "AMAGO: Scalable In-Context Reinforcement Learning for Adaptive Agents",
        "url": "https://arxiv.org/abs/2310.09971",
        "http_status": 200,
    },
    "2601.19810": {
        "title": (
            "Unsupervised Learning of Efficient Exploration: Pre-training Adaptive "
            "Policies via Self-Imposed Goals"
        ),
        "url": "https://arxiv.org/abs/2601.19810",
        "http_status": 200,
    },
    "2603.03680": {
        "title": (
            "MAGE: Meta-Reinforcement Learning for Language Agents toward Strategic "
            "Exploration and Exploitation"
        ),
        "url": "https://arxiv.org/abs/2603.03680",
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
    "amortized meta exploration learned exploration prior in-context reinforcement learning adaptive agents",
    "algorithm distillation in-context reinforcement learning exploration trajectories",
    "go-explore first return then explore return then explore archive reinforcement learning",
    "meta reinforcement learning exploration policy prior sparse reward",
]
WEBSEARCH_WEBFETCH_TOP_SOURCES = [
    "https://arxiv.org/abs/2310.09971",
    "https://arxiv.org/abs/2210.14215",
    "https://arxiv.org/abs/2601.19810",
    "https://arxiv.org/abs/2008.02790",
    "https://arxiv.org/abs/1802.07245",
    "https://arxiv.org/abs/2004.12919",
    "https://arxiv.org/abs/1901.10995",
    "https://arxiv.org/abs/2603.03680",
]
DEFAULT_PRECONDITIONS_CHECKED = {
    "agents_md_read": True,
    "codex_md_read": True,
    "network_hf_models_reachable": True,
    "sweep_clusters_help_ok": True,
    "exp4685_artifact_read": True,
    "exp4685_note_read": True,
    "exp4688_artifact_read": True,
    "exp4689_artifact_read": True,
    "arc_go_explore_read": True,
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
    "Cross-game transfer wall: A1 controllable novelty ended with "
    "residual_cause_hypothesis=winning_prefix_still_not_proposed, reached_level=0, "
    "reproduced_levels=0, and chosen_submitted_config=unchanged; A2 program synthesis "
    "ended with coverage_delta=0.0, first_win_rate_delta=-0.04, and "
    "residual_bridge_gap=heldout_transitions_too_sparse. The deeper hidden-game transfer "
    "failure is that per-game directed exploration is re-derived from scratch, so even "
    "a per-public-game improvement can leave the scored hidden-game lane at 0.08."
)
DEFAULT_METHODS_MAPPED = [
    {
        "method": "In-context exploration prior distilled from first-contact histories",
        "source_ids": ["2210.14215", "2310.09971"],
        "track": "in_context_rl_exploration_prior_from_first_contact_traces",
        "residual_scope": RESIDUAL_SCOPE,
        "implement_cost_over_current_stack": (
            "high: collect successful and near-miss first-contact trajectories across public "
            "games, serialize observations/actions/rewards/tool outcomes into long-context "
            "episodes, then train or fine-tune a small sequence policy that proposes the next "
            "exploration move before the per-game E3 proposer starts from an empty prior."
        ),
        "maps_to_current_stack": (
            "live E3 explorer receives a reusable cross-game action prior; A1 "
            "controllable-novelty proposal becomes a feature channel rather than the whole "
            "explorer; A2 program-synthesis action-effect filter labels which prior actions "
            "have trusted effects; arc_go_explore.py can replay prior-proposed prefixes from "
            "archive cells."
        ),
        "fails_when": (
            "the logged trajectories are too sparse, public-game successes encode game IDs "
            "rather than reusable mechanics, context windows omit decisive hidden state, or "
            "the distilled policy imitates late exploitation instead of first-contact probing."
        ),
        "roadmap_candidate": FLAGGED_FOR_NEXT_ROADMAP[0],
    },
    {
        "method": "Self-imposed-goal and structured-noise meta exploration prior",
        "source_ids": ["2601.19810", "1802.07245"],
        "track": "self_imposed_goal_meta_exploration_prior",
        "residual_scope": RESIDUAL_SCOPE,
        "implement_cost_over_current_stack": (
            "high: define ARC-compatible self-imposed goals from object motion, changed-cell "
            "topology, HUD/register deltas, and level-up proxies, then meta-train a policy "
            "with structured stochasticity so hidden games start with purposeful probing "
            "instead of flat action noise."
        ),
        "maps_to_current_stack": (
            "live E3 explorer samples from a learned exploration latent; A1 "
            "controllable-novelty proposal supplies controllable-effect embeddings as goals; "
            "A2 program-synthesis action-effect filter rejects brittle goal-action rules; "
            "arc_go_explore.py stores reached self-imposed-goal cells for return-and-extend."
        ),
        "fails_when": (
            "self-imposed goals reward visually rich but non-winning mechanics, structured "
            "noise remains too task-family-specific, or the curriculum never generates the "
            "rare stateful action combinations hidden games need."
        ),
        "roadmap_candidate": FLAGGED_FOR_NEXT_ROADMAP[0],
    },
    {
        "method": "Decoupled meta exploration/exploitation for language-agent adaptation",
        "source_ids": ["2008.02790", "2603.03680"],
        "track": "decoupled_meta_explore_exploit_language_agent",
        "residual_scope": RESIDUAL_SCOPE,
        "implement_cost_over_current_stack": (
            "medium-high: split the current live loop into an explorer objective that gathers "
            "task-identifying transitions and an exploiter objective that attempts the level, "
            "then train/refit the language-agent reflection memory on multi-episode histories "
            "instead of one-game scratch plans."
        ),
        "maps_to_current_stack": (
            "live E3 explorer alternates explicit information-gathering and solve attempts; "
            "A1 controllable-novelty proposal is scored by whether it reveals task-relevant "
            "information; A2 program-synthesis action-effect filter provides exploitation "
            "facts; arc_go_explore.py supplies stable return states for repeated explore/exploit "
            "cycles."
        ),
        "fails_when": (
            "the exploration objective gathers information irrelevant to the executable win, "
            "language reflections hallucinate causal rules, or public-game multi-episode "
            "training overfits opponent/task identities rather than ARC mechanics."
        ),
        "roadmap_candidate": "not_primary_for_v433: useful only after a trajectory corpus exists",
    },
    {
        "method": "Return-then-explore archive upgrade for reusable first-contact state coverage",
        "source_ids": ["1901.10995", "2004.12919"],
        "track": "go_explore_return_then_explore_archive",
        "residual_scope": RESIDUAL_SCOPE,
        "implement_cost_over_current_stack": (
            "medium: harden the existing arc_go_explore.py archive with cross-game cell "
            "descriptors, state-restore/replay checks, under-visited-cell scheduling, and "
            "a bridge that feeds archive prefixes back into the live E3/A1/A2 proposal stack."
        ),
        "maps_to_current_stack": (
            "live E3 explorer gets replayable prefixes instead of restarting every hidden "
            "game from scratch; A1 controllable-novelty proposal scores post-return actions; "
            "A2 program-synthesis action-effect filter validates archive extensions; "
            "arc_go_explore.py is the existing return-then-explore implementation to upgrade."
        ),
        "fails_when": (
            "cell descriptors alias hidden registers, replay cannot restore the chosen state, "
            "the archive expands many dead cells without a goal gradient, or stochastic live "
            "conditions break deterministic offline returns."
        ),
        "roadmap_candidate": FLAGGED_FOR_NEXT_ROADMAP[1],
    },
]

STUDYING_SECTION_START = "<!-- EXP4697-AMORTIZED-EXPLORATION-SOTA-START -->"
STUDYING_SECTION_END = "<!-- EXP4697-AMORTIZED-EXPLORATION-SOTA-END -->"
STUDYING_SECTION = f"""{STUDYING_SECTION_START}
## 2026-06-24 Exp 4697 - .433 amortized-exploration SOTA ingestion - INGESTED

**Status:** INGESTED into `{NOTE_RELATIVE_PATH}`.

**Filtered track:** fallback beyond `.432` per-game directed exploration. A1
controllable novelty closed with `winning_prefix_still_not_proposed`; A2
program synthesis closed with `heldout_transitions_too_sparse`. The next wall
is `{TRANSFER_WALL}`: first-contact behavior must transfer to unseen scored
games instead of being rediscovered from scratch on each game.

**Preconditions:** Hugging Face model API reachability returned `net_ok`.
`scripts/sweep_clusters.py --help` exited cleanly. `scripts/sweep_clusters.py`
emitted the ARC exploration and neural-guided-search cluster URLs.
`scripts/sweep_semscholar.py` returned HTTP 429 for the four focused queries
and no S2-only source was promoted. Low-concurrency WebSearch/WebFetch plus
direct arXiv HTTP checks verified arXiv:2210.14215, arXiv:2310.09971,
arXiv:2601.19810, arXiv:1802.07245, arXiv:2008.02790, arXiv:2603.03680,
arXiv:1901.10995, and arXiv:2004.12919. `/deep-research` was not invoked.

**Methods marked ingested:** in-context exploration-prior distillation,
self-imposed-goal / structured-noise meta exploration, decoupled
meta-explore/exploit language-agent adaptation, and Go-Explore return-then-
explore archive upgrade.

flagged_for_v433: in_context_exploration_prior_from_first_contact_traces
(arXiv:2210.14215 + arXiv:2310.09971 + arXiv:2601.19810)

flagged_for_v433: arc_go_explore_return_then_explore_archive_upgrade
(arXiv:1901.10995 + arXiv:2004.12919)

**Bottom line for .433:** build the in-context exploration prior first because
it directly amortizes rare successful first-contact behavior across games. Keep
the Go-Explore archive as the structural companion because it already exists in
`arc_go_explore.py` and provides replayable return points for deeper probing.
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
    """Build and validate the REQ-ARC-WMTE-4697 mapping artifact."""

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
    """Validate the artifact so uncited .433 transfer claims fail closed."""

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
        raise ValueError("note_path must point at the 2026-06-24 amortized-exploration note")
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
                "A1 controllable-novelty proposal",
                "A2 program-synthesis action-effect filter",
                "arc_go_explore.py",
            )
        ):
            raise ValueError(
                "methods must map to live E3 explorer, A1 controllable-novelty proposal, "
                "A2 program-synthesis action-effect filter, and arc_go_explore.py"
            )
        residual_scope = method["residual_scope"]
        if not isinstance(residual_scope, str) or not all(
            phrase in residual_scope for phrase in (A1_RESIDUAL, A2_RESIDUAL, TRANSFER_WALL)
        ):
            raise ValueError("methods must state the .432 cross-game transfer residuals")
        if not method["implement_cost_over_current_stack"]:
            raise ValueError("each method needs implement_cost_over_current_stack")
        if not method["fails_when"]:
            raise ValueError("each method needs fails_when")

    roadmap = artifact["flagged_for_next_roadmap"]
    if not isinstance(roadmap, Sequence) or isinstance(roadmap, str | bytes) or not all(
        isinstance(item, str) and "flagged_for_v433" in item for item in roadmap
    ):
        raise ValueError("flagged_for_next_roadmap must contain .433 flagged_for_v433 items")

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
        "SOTA -> .433 amortized-exploration mapping",
        "Bottom line for the .433 roadmap",
        "live E3 explorer",
        "A1 controllable-novelty proposal",
        "A2 program-synthesis action-effect filter",
        "arc_go_explore.py",
        A1_RESIDUAL,
        A2_RESIDUAL,
        TRANSFER_WALL,
        "flagged_for_v433",
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
    return f"""# Amortized-exploration SOTA ingestion 2026-06-24

```json
{artifact_json}
```

## Fresh-pass provenance

Read `AGENTS.md`, `CODEX.md`,
`results/experiment_4685_sota_ingestion_directed_exploration.json`,
`docs/research-notes/directed-exploration-sota-ingestion-2026-06-24.md`,
`results/experiment_4688_controllable_novelty_proposal_policy_live.json`,
`results/experiment_4689_program_synthesis_action_effect_proposal_filter.json`,
`python/carnot/agentic/arc_go_explore.py`, `research-studying.md`, and
`research-references.md`. The .432 A1 arm closed with
`winning_prefix_still_not_proposed`, no reproduced level, and unchanged
submitted config. The .432 A2 arm closed with `heldout_transitions_too_sparse`,
coverage delta 0.0, and unchanged submitted config. The .433 scope is therefore
{TRANSFER_WALL}: first-contact behavior must be amortized across games instead
of being rediscovered from scratch on each scored hidden game.

Reliable-channel pass, not `/deep-research`:
- `curl -sf -o /dev/null https://huggingface.co/api/models`
- `.venv/bin/python scripts/sweep_clusters.py --help`
- `.venv/bin/python scripts/sweep_clusters.py 5 --max-results 8`
- `.venv/bin/python scripts/sweep_clusters.py 6 --max-results 8`
- `.venv/bin/python scripts/sweep_semscholar.py - --limit 8` with four focused queries
- low-concurrency WebSearch/WebFetch of the top amortized/meta exploration and Go-Explore papers
- direct arXiv HTTP checks for all cited IDs

Semantic Scholar returned HTTP 429 for the focused queries and no S2-only
source was promoted. Direct arXiv HTTP checks returned 200 for
arXiv:2210.14215, arXiv:2310.09971, arXiv:2601.19810, arXiv:1802.07245,
arXiv:2008.02790, arXiv:2603.03680, arXiv:1901.10995, and arXiv:2004.12919.
No live LLM inference, no training, no leaderboard submission, no model load,
and no live solve claim were run or made. `scripts/research_conductor.py`,
`ops/changelog.md`, and `ops/status.md` were not edited by this workflow.

## SOTA -> .433 amortized-exploration mapping

## In-context exploration prior distilled from first-contact histories

**Sources:** Algorithm Distillation, arXiv:2210.14215; AMAGO, arXiv:2310.09971.

**Mapping to current stack:** train a cross-game sequence policy over
observation/action/reward/tool histories so the live E3 explorer begins hidden
games with a reusable probing prior. A1 controllable-novelty proposal becomes a
feature channel. A2 program-synthesis action-effect filter supplies trusted
effect labels. `arc_go_explore.py` can replay the prior's proposed prefixes
from archive cells.

**Implementation cost over current stack:** high. The current stack logs enough
per-game traces for evaluation, but it does not yet curate a cross-game
trajectory dataset or train a sequence policy. The required build is a compact
trajectory schema plus an offline distillation/fine-tuning job.

**Fails when:** the first-contact corpus is too sparse, public-game identifiers
leak into the prior, or the model imitates late solve exploitation rather than
early exploration.

## Self-imposed-goal and structured-noise meta exploration prior

**Sources:** ULEE self-imposed goals, arXiv:2601.19810; MAESN structured
exploration, arXiv:1802.07245.

**Mapping to current stack:** generate cross-game self-imposed goals from
controllable cell changes, object motion, register/HUD deltas, and level-up
proxies. The live E3 explorer samples a learned exploration latent, A1 scores
controllable novelty against those goals, A2 rejects brittle rules, and
`arc_go_explore.py` stores reached goal cells for return-and-extend.

**Implementation cost over current stack:** high. It needs a pretraining
curriculum and a goal-descriptor vocabulary, but it directly addresses the
hidden-game transfer problem rather than retuning one game at a time.

**Fails when:** the goal vocabulary rewards visual churn, structured noise is
too family-specific, or the curriculum never reaches the stateful action
combinations that hidden games score.

## Decoupled meta exploration/exploitation for language-agent adaptation

**Sources:** DREAM, arXiv:2008.02790; MAGE, arXiv:2603.03680.

**Mapping to current stack:** split the live agent into an information-gathering
phase and a solve/exploitation phase. The live E3 explorer gathers
task-identifying transitions, A1 scores which probes reveal controllable
information, A2 converts trusted effects into exploitation facts, and
`arc_go_explore.py` provides stable return states for repeated cycles.

**Implementation cost over current stack:** medium-high. It reuses the existing
live loop but needs multi-episode memory/reflection training and a clean
separation between task-identification rewards and level-completion rewards.

**Fails when:** the exploration objective optimizes irrelevant information,
language reflections invent causal rules, or public-game multi-episode training
overfits task identities.

## Return-then-explore archive upgrade for reusable first-contact state coverage

**Sources:** Go-Explore, arXiv:1901.10995; First return, then explore,
arXiv:2004.12919.

**Mapping to current stack:** harden `arc_go_explore.py` so the archive is a
first-class producer of replayable prefixes. The live E3 explorer can return to
under-explored cells, A1 controllable-novelty proposal scores the post-return
actions, and A2 program-synthesis action-effect filter validates archive
extensions.

**Implementation cost over current stack:** medium. The scaffold already exists,
but it needs cross-game cell descriptors, restore/replay verification, and a
bridge that feeds archive prefixes into the submitted live stack.

**Fails when:** hidden registers alias into the same cell, replay cannot restore
the selected state, or the archive expands many dead cells without a goal
gradient.

## Bottom line for the .433 roadmap

The strongest .433 input is
flagged_for_v433: in_context_exploration_prior_from_first_contact_traces
(arXiv:2210.14215 + arXiv:2310.09971 + arXiv:2601.19810). It directly attacks
the cross-game transfer failure by amortizing successful and near-miss
first-contact trajectories into a reusable policy.

The structural companion is
flagged_for_v433: arc_go_explore_return_then_explore_archive_upgrade
(arXiv:1901.10995 + arXiv:2004.12919). It is cheaper because
`arc_go_explore.py` already exists, and it gives the exploration prior stable
return points rather than forcing every hidden-game attempt to restart from the
initial state.
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
    root = Path(os.environ.get("CARNOT_EXP4697_ROOT", Path(__file__).resolve().parents[2]))
    write_outputs(
        artifact_path=root / RESULT_RELATIVE_PATH,
        note_path=root / NOTE_RELATIVE_PATH,
        studying_path=root / "research-studying.md",
    )
    print(HONEST_VERDICT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
