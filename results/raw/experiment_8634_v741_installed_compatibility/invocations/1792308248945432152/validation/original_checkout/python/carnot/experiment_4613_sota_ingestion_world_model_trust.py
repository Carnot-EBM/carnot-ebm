"""Exp 4613 world-model trust SOTA ingestion.

Spec refs: REQ-ARC-WMTE-4613, SCENARIO-ARC-WMTE-4613.

This is a literature-to-experiment mapping artifact, not a benchmark run. It
records the 2026-06-23 focused pass over executable world models, verifier
scaling, closed-loop world-model evaluation, learned search heuristics, and
goal-conditioned values, then maps them onto the current A1 trust-energy and
A2 scored-agent integration stack.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
from typing import Any


JsonMap = Mapping[str, Any]

RESULT_RELATIVE_PATH = "results/experiment_4613_sota_ingestion_world_model_trust.json"
NOTE_RELATIVE_PATH = "docs/research-notes/world-model-trust-literature-2026-06-23.md"
RANDOM_SEED = 4613
HONEST_VERDICT = "success: sota_ingestion_world_model_trust_mapped"
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
        "methods_mapped",
        "flagged_for_next_roadmap",
        "note_path",
        "deep_research_not_used",
        "preconditions_checked",
    }
)
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "methods_mapped",
    "flagged_for_next_roadmap",
    "note_path",
    "deep_research_not_used",
    "preconditions_checked",
    "citations_verified",
    "random_seed",
    "field_principles",
)
REQUIRED_METHOD_FIELDS = frozenset(
    {
        "method",
        "source_ids",
        "track",
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
        "exp4601_artifact_read",
        "research_studying_read",
        "research_references_read",
        "search_layer_template_read",
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
        "2605.05138",
        "2502.01989",
        "2510.18135",
        "2511.09515",
        "2102.04518",
        "2406.04935",
        "2206.03023",
        "2502.20379",
    }
)
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(
    f"arXiv:{source_id}" for source_id in REQUIRED_SOURCE_IDS
)
FLAGGED_FOR_NEXT_ROADMAP = [
    (
        "flagged_for_v426: executable_world_model_plus_multi_verifier_trust_energy "
        "(arXiv:2605.05138 + arXiv:2502.20379)"
    ),
    (
        "flagged_for_v426: goal_conditioned_spatial_value_tiebreaker "
        "(arXiv:2102.04518 + arXiv:2406.04935 + arXiv:2206.03023)"
    ),
]

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": "terminal prefix; success: sota_ingestion_world_model_trust_mapped."
    },
    "inference_substrate": {
        "principle": (
            "aggregation_from_upstream_artifacts -- literature read + synthesis, "
            "no model load (100us floor)."
        )
    },
    "methods_mapped": {
        "principle": (
            "the strongest 3-5 SOTA methods with REAL arXiv IDs + per-method "
            "implement-cost-over-current-stack + fails_when -- the actionable "
            "ingestion (no citation = fabrication)."
        )
    },
    "flagged_for_next_roadmap": {
        "principle": (
            "the strongest method(s) flagged as candidate .426 inputs -- closes "
            "discover->ingest->plan->experiment."
        )
    },
    "note_path": {
        "principle": (
            "docs/research-notes/world-model-trust-literature-2026-06-23.md -- "
            "the per-track note (the SOTA-Ingestion Cycle deliverable)."
        )
    },
    "deep_research_not_used": {
        "principle": (
            "MUST be true -- /deep-research is banned in the autonomous loop; used "
            "sweep helpers + low-concurrency WebSearch/WebFetch."
        )
    },
    "preconditions_checked": {
        "principle": ("records network reachability verified; pre-empts fabricated citations.")
    },
}

CITATIONS_VERIFIED = {
    "2605.05138": {
        "title": "Executable World Models for ARC-AGI-3 in the Era of Coding Agents",
        "url": "https://arxiv.org/abs/2605.05138",
        "http_status": 200,
    },
    "2502.01989": {
        "title": "VFScale: Intrinsic Reasoning through Verifier-Free Test-time Scalable Diffusion Model",
        "url": "https://arxiv.org/abs/2502.01989",
        "http_status": 200,
    },
    "2510.18135": {
        "title": "World-in-World: World Models in a Closed-Loop World",
        "url": "https://arxiv.org/abs/2510.18135",
        "http_status": 200,
    },
    "2511.09515": {
        "title": "WMPO: World Model-based Policy Optimization for Vision-Language-Action Models",
        "url": "https://arxiv.org/abs/2511.09515",
        "http_status": 200,
    },
    "2102.04518": {
        "title": "A* Search Without Expansions: Learning Heuristic Functions with Deep Q-Networks",
        "url": "https://arxiv.org/abs/2102.04518",
        "http_status": 200,
    },
    "2406.04935": {
        "title": "SLOPE: Search with Learned Optimal Pruning-based Expansion",
        "url": "https://arxiv.org/abs/2406.04935",
        "http_status": 200,
    },
    "2206.03023": {
        "title": "How Far I'll Go: Offline Goal-Conditioned Reinforcement Learning via f-Advantage Regression",
        "url": "https://arxiv.org/abs/2206.03023",
        "http_status": 200,
    },
    "2502.20379": {
        "title": "Multi-Agent Verification: Scaling Test-Time Compute with Multiple Verifiers",
        "url": "https://arxiv.org/abs/2502.20379",
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
        '(abs:"active+inference"+OR+abs:"free+energy"+OR+'
        'abs:"free+energy+principle"+OR+abs:"predictive+coding"+OR+'
        'abs:"world+model")+AND+'
        '(abs:"LLM"+OR+abs:"language+model"+OR+abs:"reasoning")'
        "&start=0&max_results=8&sortBy=submittedDate&sortOrder=descending"
    ),
]
S2_QUERIES = [
    "Executable World Models ARC-AGI-3 verifier world model trust",
    "VFScale verifier scaling agent generalization",
    "World-in-World 2510.18135 world model agent",
    "WMPO 2511.09515 world model policy optimization",
    (
        "DeepCubeA learned heuristic A* SLOPE learned optimal pruning expansion "
        "goal conditioned value HER 2206.03023"
    ),
]
WEBSEARCH_WEBFETCH_TOP_SOURCES = [
    f"https://arxiv.org/abs/{source_id}"
    for source_id in (
        "2605.05138",
        "2502.01989",
        "2510.18135",
        "2511.09515",
        "2102.04518",
        "2406.04935",
        "2206.03023",
        "2502.20379",
    )
]

DEFAULT_PRECONDITIONS_CHECKED = {
    "agents_md_read": True,
    "codex_md_read": True,
    "network_hf_models_reachable": True,
    "exp4601_artifact_read": True,
    "research_studying_read": True,
    "research_references_read": True,
    "search_layer_template_read": True,
    "sweep_clusters_used": True,
    "sweep_clusters_urls": SWEEP_CLUSTER_URLS,
    "sweep_semscholar_used": True,
    "sweep_semscholar_queries": S2_QUERIES,
    "sweep_semscholar_arxiv_ids": [],
    "sweep_semscholar_rate_limited_queries": S2_QUERIES,
    "arxiv_http_200_verified_ids": list(WEBSEARCH_WEBFETCH_TOP_SOURCES),
    "websearch_webfetch_top_sources": WEBSEARCH_WEBFETCH_TOP_SOURCES,
    "deep_research_invoked": False,
    "live_llm_inference": False,
    "training_launched": False,
    "model_load": False,
    "leaderboard_submission": False,
    "ops_docs_modified": False,
    "research_conductor_modified": False,
}

DEFAULT_METHODS_MAPPED = [
    {
        "method": "Executable world-model induction plus multi-verifier trust energy",
        "source_ids": ["2605.05138", "2502.20379"],
        "track": "executable_world_model_trust",
        "implement_cost_over_current_stack": (
            "medium: keep the current A1 trust-energy selector, replace the weak "
            "binary gate with an executable-model candidate pool, and add "
            "A2-compatible aspect scores for transition fidelity, changed-cell "
            "coverage, goal predicate consistency, and plan executability."
        ),
        "maps_to_current_stack": (
            "A1 already ranks candidates by oracle-distinct trust energy, while "
            "A2 needs that selected model to reach the scored E3AgentPolicy. "
            "Executable World Models supplies the induce->verify->plan loop; "
            "Multi-Agent Verification supplies the multi-aspect verifier scaling "
            "pattern without making the game oracle the verifier."
        ),
        "fails_when": (
            "the model candidate pool is still empty, the verifier rewards an "
            "identity or near-identity transition, aspect verifiers share the same "
            "blind spot, or A2 never routes the trusted model into the scored agent."
        ),
        "roadmap_candidate": FLAGGED_FOR_NEXT_ROADMAP[0],
    },
    {
        "method": "Intrinsic energy search as a verifier-free cautionary control",
        "source_ids": ["2502.01989"],
        "track": "energy_search_control",
        "implement_cost_over_current_stack": (
            "low-to-medium for a control, high for the full method: expose A1 "
            "trust energy as a sample/search controller and compare it against "
            "execution-grounded verification before considering any learned "
            "diffusion-energy analogue."
        ),
        "maps_to_current_stack": (
            "VFScale is relevant because it uses an intrinsic energy function as "
            "the verifier for test-time search. For Carnot A1, that is a negative "
            "control unless the energy is grounded by transition execution; for "
            "A2, the same control must improve live-score behavior before adoption."
        ),
        "fails_when": (
            "the energy becomes a self-referential learned score, hMCTS improves "
            "internal consistency but not executable transition generalization, or "
            "the control is treated as evidence that the ARC oracle was avoided."
        ),
        "roadmap_candidate": "flagged_for_v426: trust_energy_vs_intrinsic_energy_control",
    },
    {
        "method": "Closed-loop world-model utility gate and imagined policy repair",
        "source_ids": ["2510.18135", "2511.09515"],
        "track": "closed_loop_world_model_policy",
        "implement_cost_over_current_stack": (
            "medium for the World-in-World style gate, high for WMPO: add a "
            "closed-loop success/control metric for each trusted model now; defer "
            "policy optimization in imagined trajectories until the symbolic model "
            "passes held-out transition and A2 action-efficiency checks."
        ),
        "maps_to_current_stack": (
            "World-in-World says A1 world models should be judged by closed-loop "
            "task success, not visual or rollout plausibility. WMPO adds an A2 "
            "repair path: optimize policy behavior inside the trusted model before "
            "touching the real environment."
        ),
        "fails_when": (
            "the model is visually or locally plausible but uncontrollable, imagined "
            "rollouts drift from real ARC transitions, or optimization overfits the "
            "public-game simulator and regresses scored-agent efficiency."
        ),
        "roadmap_candidate": "flagged_for_v426: closed_loop_trust_utility_gate_before_policy_repair",
    },
    {
        "method": "Learned value and optimal-pruning search over trusted models",
        "source_ids": ["2102.04518", "2406.04935"],
        "track": "learned_heuristic_search",
        "implement_cost_over_current_stack": (
            "low-to-medium: wire the existing SpatialValueNet-style value as a "
            "same-depth tie-breaker first, then add SLOPE-like pruning only after "
            "A2 parity and no-regression tests show it does not hide valid branches."
        ),
        "maps_to_current_stack": (
            "DeepCubeA/Q* and SLOPE support the .425 finding that a learned value "
            "can cut expansions while classical search keeps legality. The A1/A2 "
            "version is search over a trusted executable model, with depth and "
            "reproduction gates still primary."
        ),
        "fails_when": (
            "the learned value is trained on shallow public levels only, pruning "
            "drops the only branch that reveals a hidden register, or A2 uses the "
            "value as a heavy priority instead of a bounded tie-breaker."
        ),
        "roadmap_candidate": FLAGGED_FOR_NEXT_ROADMAP[1],
    },
    {
        "method": "Goal-conditioned value for level-to-level generalization",
        "source_ids": ["2206.03023"],
        "track": "goal_conditioned_value",
        "implement_cost_over_current_stack": (
            "medium: condition the value head on the currently induced level goal "
            "or register-aware GOAL predicate, train from offline traces and "
            "self-play failures, and expose it only as an A2 tie-breaker until "
            "per-level no-regression checks pass."
        ),
        "maps_to_current_stack": (
            "The requested UVFA/HER-adjacent citation resolves to GoFAR, not a "
            "UVFA/HER primary paper. The usable point is still goal-conditioned "
            "offline value learning: A1 supplies the trusted transition model and "
            "goal predicate, while A2 needs a dense value that changes when the "
            "level goal changes."
        ),
        "fails_when": (
            "the goal predicate is wrong, hindsight relabeling or offline data "
            "smears incompatible level goals together, or the dense value overrides "
            "the scored-agent preservation gate."
        ),
        "roadmap_candidate": FLAGGED_FOR_NEXT_ROADMAP[1],
    },
]

STUDYING_SECTION_START = "<!-- EXP4613-WORLD-MODEL-TRUST-SOTA-START -->"
STUDYING_SECTION_END = "<!-- EXP4613-WORLD-MODEL-TRUST-SOTA-END -->"
STUDYING_SECTION = f"""{STUDYING_SECTION_START}
## 2026-06-23 Exp 4613 - .425 world-model trust SOTA ingestion - INGESTED

**Status:** INGESTED into `{NOTE_RELATIVE_PATH}`.

**Filtered track:** world-model trust energy, scored-agent verifier
integration, closed-loop model utility, learned heuristic search, and
goal-conditioned value for level-to-level generalization.

**Preconditions:** Hugging Face model API reachability returned `net_ok`.
`scripts/sweep_clusters.py` emitted focused world-model/search URLs. Semantic
Scholar returned HTTP 429 for the five focused queries, so no S2-only source was
promoted. Low-concurrency WebSearch/WebFetch plus direct arXiv HTTP checks
verified arXiv:2605.05138, arXiv:2502.01989, arXiv:2510.18135,
arXiv:2511.09515, arXiv:2102.04518, arXiv:2406.04935, arXiv:2206.03023, and
arXiv:2502.20379. `/deep-research` was not invoked.

**Methods marked ingested:** executable world-model induction plus
multi-verifier trust energy; VFScale as an intrinsic-energy control;
closed-loop world-model utility plus imagined policy repair; learned
value/pruning search; and goal-conditioned value learning. Note: arXiv:2206.03023
is GoFAR, not a UVFA/HER primary paper, so it is used as the goal-conditioned
offline value reference rather than mislabeled.

flagged_for_v426: executable_world_model_plus_multi_verifier_trust_energy
(arXiv:2605.05138 + arXiv:2502.20379)

flagged_for_v426: goal_conditioned_spatial_value_tiebreaker
(arXiv:2102.04518 + arXiv:2406.04935 + arXiv:2206.03023)

**Bottom line for .426:** make executable world-model induction the A1 source
of candidate models, score it with multi-aspect trust energy, route only
trusted models into A2, and use learned/goal-conditioned value strictly as a
bounded search tie-breaker until no-regression gates pass.
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
    """Build and validate the REQ-ARC-WMTE-4613 mapping artifact."""

    artifact: dict[str, object] = {
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "methods_mapped": [dict(method) for method in methods_mapped],
        "flagged_for_next_roadmap": list(flagged_for_next_roadmap),
        "note_path": NOTE_RELATIVE_PATH,
        "deep_research_not_used": True,
        "preconditions_checked": dict(preconditions_checked),
        "citations_verified": {
            source_id: dict(citation) for source_id, citation in citations_verified.items()
        },
        "random_seed": RANDOM_SEED,
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: JsonMap) -> None:
    """Validate the artifact so uncited method claims fail closed."""

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
        raise ValueError("note_path must point at the 2026-06-23 world-model trust note")
    if artifact["deep_research_not_used"] is not True:
        raise ValueError("deep_research_not_used must be true")
    if artifact["random_seed"] != RANDOM_SEED:
        raise ValueError("random_seed must be the bare integer 4613")

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
        if (
            "A1" not in method["maps_to_current_stack"]
            or "A2" not in method["maps_to_current_stack"]
        ):
            raise ValueError("method mapping must name A1 and A2")
        for field in REQUIRED_METHOD_FIELDS - {"source_ids"}:
            value = method[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"method {field} must be a non-empty string")

    flagged = artifact["flagged_for_next_roadmap"]
    if flagged != FLAGGED_FOR_NEXT_ROADMAP or not all(
        "flagged_for_v426" in item for item in flagged
    ):
        raise ValueError("flagged_for_next_roadmap must name the .426 candidates")

    preconditions = artifact["preconditions_checked"]
    if not isinstance(preconditions, dict) or set(preconditions) != REQUIRED_PRECONDITION_FIELDS:
        raise ValueError("preconditions_checked must contain the exact required fields")
    if preconditions["network_hf_models_reachable"] is not True:
        raise ValueError("network reachability precondition must be true")
    if preconditions["deep_research_invoked"] is not False:
        raise ValueError("deep-research must not be invoked")
    if preconditions["research_conductor_modified"] is not False:
        raise ValueError("research_conductor.py must not be modified")
    if preconditions["ops_docs_modified"] is not False:
        raise ValueError("ops docs must not be modified by Exp 4613")


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
    """Check that the paired note maps verified sources to .426 work."""

    artifact_from_note(markdown)
    required_phrases = (
        "Fresh-pass provenance",
        "SOTA -> experiment mapping",
        "A1 trust-energy",
        "A2 scored-agent",
        "Bottom line for the .426 roadmap",
        "flagged_for_v426",
        "not a UVFA/HER primary paper",
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
    return f"""# World-model trust literature ingestion 2026-06-23

```json
{artifact_json}
```

## Fresh-pass provenance

Read `AGENTS.md`, `CODEX.md`, `results/experiment_4601_sota_ingestion_generation.json`,
`research-studying.md`, `research-references.md`, and
`docs/research-notes/search-layer-literature-2026-06-11.md`. The filtered track
was the .425 headline open problem: A1 trust-energy for executable world models
plus A2 scored-agent verifier integration, feeding candidate methods forward to
the .426 roadmap.

Reliable-channel pass, not `/deep-research`:
- `.venv/bin/python -c "import urllib.request; urllib.request.urlopen('https://huggingface.co/api/models', timeout=10); print('net_ok')"`
- `.venv/bin/python scripts/sweep_clusters.py 6 --max-results 8`
- `.venv/bin/python scripts/sweep_clusters.py 3 --max-results 8`
- `.venv/bin/python scripts/sweep_semscholar.py - --limit 8` with five focused queries
- low-concurrency WebSearch/WebFetch of the top arXiv papers
- direct arXiv HTTP checks for all cited IDs

Semantic Scholar returned HTTP 429 for the five focused queries, so no
Semantic-Scholar-only source was promoted. Direct arXiv HTTP checks returned
200 for arXiv:2605.05138, arXiv:2502.01989, arXiv:2510.18135,
arXiv:2511.09515, arXiv:2102.04518, arXiv:2406.04935, arXiv:2206.03023, and
arXiv:2502.20379. No live LLM inference, No training, No leaderboard submission,
no model load, and no live solve claim were run or made.
`scripts/research_conductor.py`, `ops/changelog.md`, and `ops/status.md` were
not edited by this workflow.

## SOTA -> experiment mapping

## Executable world-model trust plus multi-verifier scoring

**Sources:** Executable World Models, arXiv:2605.05138; Multi-Agent
Verification, arXiv:2502.20379.

**Mapping to A1 trust-energy / A2 scored-agent integration:** A1 should make
the executable model the candidate object, not just a final-plan reranker. Score
candidate models with multiple execution-grounded aspects: transition fidelity,
changed-cell coverage, goal predicate consistency, and plan executability. A2
then imports only trusted models into the scored policy.

**Implementation cost over current stack:** medium. The selector exists, but
the candidate pool and multi-aspect scoring need to be wired into the live
E3AgentPolicy path.

**Fails when:** candidate generation is empty, the verifier accepts identity
dynamics, all aspect verifiers share one blind spot, or A2 never consumes the
trusted model.

## VFScale intrinsic energy as a control

**Source:** VFScale, arXiv:2502.01989.

**Mapping to A1 trust-energy / A2 scored-agent integration:** VFScale is useful
as a contrast, not as a direct drop-in. It makes intrinsic learned energy act as
the verifier during test-time search. Carnot should test that pattern only as a
control against execution-grounded trust energy, because A1 must stay
oracle-distinct and A2 must improve real scored behavior.

**Implementation cost over current stack:** low-to-medium for a control, high
for the full diffusion-style method.

**Fails when:** internal energy becomes self-referential, hMCTS improves only
sample consistency, or the result is mistaken for transition verification.

## Closed-loop model utility and imagined policy repair

**Sources:** World-in-World, arXiv:2510.18135; WMPO, arXiv:2511.09515.

**Mapping to A1 trust-energy / A2 scored-agent integration:** World-in-World
sets the right gate: judge a world model by closed-loop task utility, not
rollout plausibility. WMPO suggests a later repair loop where policy behavior is
optimized inside a trusted model before using the real environment. For .426,
the cheap step is the closed-loop utility gate; policy optimization should wait
until trusted symbolic models pass held-out checks.

**Implementation cost over current stack:** medium for the gate, high for
imagined policy optimization.

**Fails when:** the model is plausible but uncontrollable, imagined rollouts
drift from ARC transitions, or optimization overfits the public games.

## Learned heuristic and pruning search

**Sources:** Q*/DeepCubeA search, arXiv:2102.04518; SLOPE, arXiv:2406.04935.

**Mapping to A1 trust-energy / A2 scored-agent integration:** These papers
support the .425 value-positive: use learned state-action value or learned
near-optimal-path distance to cut expansions while classical search and the
trusted executable model preserve legality. In A2, value should start as a
same-depth tie-breaker, not a heavy priority.

**Implementation cost over current stack:** low-to-medium. SpatialValueNet
already exists as a dev-side positive; the work is tying it to trusted-model
search and adding no-regression gates.

**Fails when:** the value is trained only on shallow public states, pruning
hides the one branch that exposes a hidden register, or A2 lets value override
the reproduction gate.

## Goal-conditioned value for the level boundary

**Source:** GoFAR, arXiv:2206.03023. Note: this is not a UVFA/HER primary paper;
it is the requested goal-conditioned offline value reference resolved to a real
arXiv ID.

**Mapping to A1 trust-energy / A2 scored-agent integration:** A1 supplies the
registered state and goal predicate; A2 needs a dense value that changes when
the level goal changes. The .426 version should condition the value on the
current induced GOAL predicate and use it only as a bounded tie-breaker until
per-level preservation passes.

**Implementation cost over current stack:** medium. It needs goal-conditioned
trace labels and failure relabeling, but it can reuse the existing value net and
offline self-play traces.

**Fails when:** the goal predicate is wrong, relabeling smears incompatible
level goals together, or dense value overrides the scored-agent preservation
gate.

## Bottom line for the .426 roadmap

1. Build `flagged_for_v426: executable_world_model_plus_multi_verifier_trust_energy`
   first: executable model candidates from arXiv:2605.05138, multi-aspect
   verifier scaling from arXiv:2502.20379, and a closed-loop utility gate from
   arXiv:2510.18135.
2. Add `flagged_for_v426: goal_conditioned_spatial_value_tiebreaker` as the
   value/search support lever: Q*/DeepCubeA arXiv:2102.04518, SLOPE
   arXiv:2406.04935, and GoFAR arXiv:2206.03023.
3. Keep VFScale arXiv:2502.01989 as a control that prevents A1 trust energy
   from drifting into an ungrounded intrinsic score.
4. Defer WMPO arXiv:2511.09515-style imagined policy optimization until the
   trusted symbolic model and A2 no-regression gates are green.
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
    """Write the default Exp 4613 deliverables under the repository root."""

    repo_root = Path(os.environ.get("CARNOT_EXP4613_ROOT", Path(__file__).resolve().parents[2]))
    artifact = write_outputs(
        artifact_path=repo_root / RESULT_RELATIVE_PATH,
        note_path=repo_root / NOTE_RELATIVE_PATH,
        studying_path=repo_root / "research-studying.md",
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
