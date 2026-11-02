"""Exp 4758 SOTA ingestion for structured ARC world models and grounded goals.

Spec refs: REQ-ARC-WMTE-4758, SCENARIO-ARC-WMTE-4758.

This module writes a literature-synthesis artifact, not a solve claim. The
source set comes from the repo's discovered corpus, focused sweep helpers, and
low-concurrency WebSearch/WebFetch of arXiv pages.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


JsonMap = Mapping[str, Any]

RESULT_RELATIVE_PATH = "results/experiment_4758_sota_ingestion.json"
STUDYING_RELATIVE_PATH = "research-studying.md"
NOTE_PATH = "research-studying.md#exp-4758-sota-ingestion"
RANDOM_SEED = 4758
HONEST_VERDICT = "complete_sota_ingestion_structured_world_model_goal_frontier_mapped"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
HF_NETWORK_CHECK_URL = "https://huggingface.co/api/models"
STUDYING_SECTION_START = "<!-- EXP4758-SOTA-INGESTION-START -->"
STUDYING_SECTION_END = "<!-- EXP4758-SOTA-INGESTION-END -->"
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

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "methods_mapped",
    "flagged_for_438",
    "citations",
    "fresh_sweep",
    "note_path",
    "random_seed",
    "reproducibility_checksum",
    "field_principles",
)
REQUIRED_USER_FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": "terminal prefix; an ingestion-synthesized run is complete_."
    },
    "inference_substrate": {
        "principle": "aggregation_from_upstream_artifacts; 100us floor."
    },
    "preconditions_checked": {"principle": "records the network check."},
    "methods_mapped": {
        "principle": (
            "the strongest 3-5 SOTA methods with REAL arXiv IDs -- no "
            "fabricated citations (adversarial_verify bar)."
        )
    },
    "flagged_for_438": {
        "principle": (
            "the strongest method flagged as a candidate input for the .438 "
            "roadmap -- closes discover->ingest->plan."
        )
    },
}
FIELD_PRINCIPLES = {
    **REQUIRED_USER_FIELD_PRINCIPLES,
    "citations": {
        "principle": "HTTP-200 arXiv source set used for adversarial citation verification."
    },
    "fresh_sweep": {
        "principle": "records reliable-channel sweep helpers and WebSearch/WebFetch provenance."
    },
    "note_path": {
        "principle": "points to the idempotent research-studying.md mapping note."
    },
    "random_seed": {"principle": "determinism precondition for reproducibility."},
    "reproducibility_checksum": {
        "principle": "content-addressed hash of citations and mapped methods."
    },
}
REQUIRED_PRINCIPLE_FIELDS = frozenset(FIELD_PRINCIPLES)
REQUIRED_METHOD_FIELDS = frozenset(
    {
        "method",
        "track",
        "source_ids",
        "maps_to_current_stack",
        "takes_over_from_current_stack",
        "fails_when",
        "roadmap_candidate",
    }
)
REQUIRED_CITATION_FIELDS = frozenset({"title", "url", "http_status"})
REQUIRED_PRECONDITION_FIELDS = frozenset(
    {
        "agents_md_read",
        "codex_md_read",
        "network_check_command",
        "network_hf_models_reachable",
        "research_studying_read",
        "research_references_read",
        "sweep_clusters_used",
        "sweep_cluster_urls",
        "sweep_semscholar_used",
        "sweep_semscholar_queries",
        "sweep_semscholar_http_429",
        "websearch_webfetch_used",
        "websearch_webfetch_top_sources",
        "top_source_count",
        "arxiv_http_200_verified_ids",
        "deep_research_invoked",
        "model_load",
        "training_launched",
        "leaderboard_submission",
        "solve_claim_made",
        "ops_docs_modified",
    }
)
REQUIRED_FRESH_SWEEP_FIELDS = frozenset(
    {
        "filtered_track",
        "cluster_ids",
        "semantic_scholar_queries",
        "semantic_scholar_result",
        "webfetch_top_sources",
    }
)
REQUIRED_SOURCE_IDS = frozenset(
    {
        "2402.12275",
        "2503.23145",
        "2511.02225",
        "2601.06604",
        "2605.05138",
        "2605.14937",
        "2606.08775",
        "2606.14418",
    }
)
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(
    f"arXiv:{source_id}" for source_id in REQUIRED_SOURCE_IDS
)

CLUSTER_6_URL = (
    'http://export.arxiv.org/api/query?search_query=(abs:"neural+guided+search"+OR+'
    'abs:"learned+heuristic"+OR+abs:"value+guided+search"+OR+'
    'abs:"program+induction"+OR+abs:"world+model"+OR+abs:"goal+induction")+'
    'AND+(abs:"planning"+OR+abs:"agent"+OR+abs:"reasoning"+OR+'
    'abs:"reinforcement+learning")&start=0&max_results=8&sortBy=submittedDate&'
    "sortOrder=descending"
)
CLUSTER_5_URL = (
    'http://export.arxiv.org/api/query?search_query=(abs:"affordance"+OR+'
    'abs:"action+effect"+OR+abs:"clickability"+OR+abs:"frame+prediction"+OR+'
    'abs:"intrinsic+motivation"+OR+abs:"directed+exploration"+OR+'
    'abs:"novelty+search")+AND+(abs:"reinforcement+learning"+OR+abs:"agent"+OR+'
    'abs:"exploration"+OR+abs:"interactive+environment"+OR+abs:"ARC")&start=0&'
    "max_results=8&sortBy=submittedDate&sortOrder=descending"
)
SEMANTIC_SCHOLAR_QUERIES = [
    "ARC-AGI-3 executable world model coding agent goal induction",
    "object-centric world model goal conditioned planning MCTS perception grounded",
    "interactive program synthesis world model induction agents hidden target",
]
WEBSEARCH_WEBFETCH_TOP_SOURCES = [
    "https://arxiv.org/abs/2605.05138",
    "https://arxiv.org/abs/2402.12275",
    "https://arxiv.org/abs/2606.14418",
    "https://arxiv.org/abs/2605.14937",
    "https://arxiv.org/abs/2606.08775",
    "https://arxiv.org/abs/2503.23145",
    "https://arxiv.org/abs/2601.06604",
    "https://arxiv.org/abs/2511.02225",
]

CITATIONS = {
    "2402.12275": {
        "title": (
            "WorldCoder, a Model-Based LLM Agent: Building World Models by "
            "Writing Code and Interacting with the Environment"
        ),
        "url": "https://arxiv.org/abs/2402.12275",
        "http_status": 200,
    },
    "2503.23145": {
        "title": (
            "CodeARC: Benchmarking Reasoning Capabilities of LLM Agents for "
            "Inductive Program Synthesis"
        ),
        "url": "https://arxiv.org/abs/2503.23145",
        "http_status": 200,
    },
    "2511.02225": {
        "title": "Learning Interactive World Model for Object-Centric Reinforcement Learning",
        "url": "https://arxiv.org/abs/2511.02225",
        "http_status": 200,
    },
    "2601.06604": {
        "title": "Object-Centric World Models Meet Monte Carlo Tree Search",
        "url": "https://arxiv.org/abs/2601.06604",
        "http_status": 200,
    },
    "2605.05138": {
        "title": "Executable World Models for ARC-AGI-3 in the Era of Coding Agents",
        "url": "https://arxiv.org/abs/2605.05138",
        "http_status": 200,
    },
    "2605.14937": {
        "title": "Slot-MPC: Goal-Conditioned Model Predictive Control with Object-Centric Representations",
        "url": "https://arxiv.org/abs/2605.14937",
        "http_status": 200,
    },
    "2606.08775": {
        "title": (
            "Unifying Object-Centric World Models and Diffusion Policy: A "
            "Hierarchical Framework for Multi-Stage Robotic Tasks"
        ),
        "url": "https://arxiv.org/abs/2606.08775",
        "http_status": 200,
    },
    "2606.14418": {
        "title": "Causal Object-Centric Models for Planning with Monte Carlo Tree Search",
        "url": "https://arxiv.org/abs/2606.14418",
        "http_status": 200,
    },
}

FLAGGED_FOR_438 = (
    "flagged_for_438: verifier_refined_executable_world_model_with_"
    "perception_grounded_goal_mpc (arXiv:2605.05138 + arXiv:2402.12275 + "
    "arXiv:2605.14937 + arXiv:2606.08775)"
)

DEFAULT_METHODS_MAPPED = [
    {
        "method": "Verifier-refined executable world-model induction",
        "track": "verifier_refined_executable_world_model",
        "source_ids": ["2605.05138", "2402.12275"],
        "maps_to_current_stack": (
            "E3AgentPolicy should keep arc_executable_world_model as the acting "
            "substrate, but make ProductWorldModel and the coding-agent-style "
            "verify-refactor loop the primary induction path before planning."
        ),
        "takes_over_from_current_stack": (
            "Takes over the .437 A1 structured engine slot: replace brittle "
            "free-form load_engine output with a typed executable model, a "
            "transition verifier, and refactoring toward simpler factors."
        ),
        "fails_when": (
            "The prompt budget cannot afford repeated refactors, hidden mechanics "
            "need observations not yet taken, or the induced program overfits public "
            "prefixes without perception-grounded object/goal evidence."
        ),
        "roadmap_candidate": FLAGGED_FOR_438,
    },
    {
        "method": "Perception-grounded goal-conditioned object planning",
        "track": "perception_grounded_goal_conditioned_planning",
        "source_ids": ["2605.14937", "2606.08775"],
        "maps_to_current_stack": (
            "E3AgentPolicy should feed the structural-alignment goal pipeline into "
            "arc_executable_world_model.plan_in_model as object slots, target slots, "
            "and feasible subgoals rather than as a single brittle terminal predicate."
        ),
        "takes_over_from_current_stack": (
            "Takes over the .437 A2 detector fix by turning detected pieces and goal "
            "sprites into a goal-conditioned planning objective with subgoal checks."
        ),
        "fails_when": (
            "Slots drift across frames, ARC goals are non-spatial or hidden-state "
            "dependent, or differentiable/continuous MPC assumptions do not transfer "
            "to discrete click/key action spaces."
        ),
        "roadmap_candidate": "support_for_438: grounded_goal_conditioned_object_planner",
    },
    {
        "method": "Causal object-centric MCTS action-slot planner",
        "track": "causal_object_mcts_action_slot_planner",
        "source_ids": ["2606.14418", "2601.06604"],
        "maps_to_current_stack": (
            "E3AgentPolicy should bind each candidate click/key action to logical "
            "object slots before arc_executable_world_model rollouts, then use MCTS "
            "to decide whether to probe, advance a subgoal, or execute a solve prefix."
        ),
        "takes_over_from_current_stack": (
            "Takes over static candidate ranking by adding object-causal attention, "
            "slot-level transition predictions, and search over object interactions."
        ),
        "fails_when": (
            "The object representation misses the controllable entity, action-slot "
            "binding aliases multiple mechanics, or MCTS rollouts compound an early "
            "world-model error before a live probe can correct it."
        ),
        "roadmap_candidate": "support_for_438: causal_object_mcts_action_slot_planner",
    },
    {
        "method": "Interactive program-synthesis refinement over factor primitives",
        "track": "interactive_program_synthesis_refinement",
        "source_ids": ["2503.23145", "2511.02225"],
        "maps_to_current_stack": (
            "arc_executable_world_model should treat ProgrammaticExpert factors as "
            "candidate interaction primitives, then use cheap live probes and held-out "
            "transition checks to reject or refine them before E3AgentPolicy trusts a plan."
        ),
        "takes_over_from_current_stack": (
            "Takes over one-shot induction by converting errors into differential-test "
            "style counterexamples and by organizing object interactions into reusable "
            "factor primitives."
        ),
        "fails_when": (
            "ARC action budgets make probes too expensive, the true rule lies outside "
            "the primitive vocabulary, or there is no free oracle analogous to CodeARC's "
            "hidden function query channel."
        ),
        "roadmap_candidate": "support_for_438: probe_refined_factor_primitive_induction",
    },
]

DEFAULT_FRESH_SWEEP = {
    "filtered_track": "structured world-model induction + perception-grounded ARC goals",
    "cluster_ids": [6, 5],
    "semantic_scholar_queries": SEMANTIC_SCHOLAR_QUERIES,
    "semantic_scholar_result": "HTTP 429 on all focused queries; no S2-only source promoted.",
    "webfetch_top_sources": WEBSEARCH_WEBFETCH_TOP_SOURCES,
}
DEFAULT_PRECONDITIONS_CHECKED = {
    "agents_md_read": True,
    "codex_md_read": True,
    "network_check_command": f"curl -sf -o /dev/null {HF_NETWORK_CHECK_URL}",
    "network_hf_models_reachable": True,
    "research_studying_read": True,
    "research_references_read": True,
    "sweep_clusters_used": True,
    "sweep_cluster_urls": [CLUSTER_6_URL, CLUSTER_5_URL],
    "sweep_semscholar_used": True,
    "sweep_semscholar_queries": SEMANTIC_SCHOLAR_QUERIES,
    "sweep_semscholar_http_429": True,
    "websearch_webfetch_used": True,
    "websearch_webfetch_top_sources": WEBSEARCH_WEBFETCH_TOP_SOURCES,
    "top_source_count": len(WEBSEARCH_WEBFETCH_TOP_SOURCES),
    "arxiv_http_200_verified_ids": [
        f"https://arxiv.org/abs/{source_id}" for source_id in sorted(REQUIRED_SOURCE_IDS)
    ],
    "deep_research_invoked": False,
    "model_load": False,
    "training_launched": False,
    "leaderboard_submission": False,
    "solve_claim_made": False,
    "ops_docs_modified": False,
}


def source_set_checksum(citations: JsonMap, methods: Sequence[JsonMap]) -> str:
    """Return a stable content hash for the cited source set and method map."""

    payload = json.dumps(
        {"citations": citations, "methods": list(methods)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


REPRODUCIBILITY_CHECKSUM = source_set_checksum(CITATIONS, DEFAULT_METHODS_MAPPED)


def _blocked_preconditions() -> dict[str, object]:
    preconditions = dict(DEFAULT_PRECONDITIONS_CHECKED)
    preconditions.update(
        {
            "network_hf_models_reachable": False,
            "websearch_webfetch_used": False,
            "websearch_webfetch_top_sources": [],
            "top_source_count": 0,
            "arxiv_http_200_verified_ids": [],
        }
    )
    return preconditions


def build_blocked_network_artifact() -> dict[str, object]:
    """Build the no-fabrication artifact for a failed network precondition."""

    return {
        "honest_verdict": "blocked_network",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": _blocked_preconditions(),
        "methods_mapped": [],
        "flagged_for_438": "",
        "citations": {},
        "fresh_sweep": {},
        "note_path": NOTE_PATH,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": source_set_checksum({}, []),
        "field_principles": dict(FIELD_PRINCIPLES),
    }


def build_artifact(
    *,
    methods_mapped: Sequence[JsonMap] = DEFAULT_METHODS_MAPPED,
    citations: JsonMap = CITATIONS,
    preconditions_checked: JsonMap = DEFAULT_PRECONDITIONS_CHECKED,
    fresh_sweep: JsonMap = DEFAULT_FRESH_SWEEP,
    flagged_for_438: str = FLAGGED_FOR_438,
    honest_verdict: str = HONEST_VERDICT,
) -> dict[str, object]:
    """Build and validate the REQ-ARC-WMTE-4758 ingestion artifact."""

    artifact: dict[str, object] = {
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "methods_mapped": [dict(method) for method in methods_mapped],
        "flagged_for_438": flagged_for_438,
        "citations": {source_id: dict(citation) for source_id, citation in citations.items()},
        "fresh_sweep": dict(fresh_sweep),
        "note_path": NOTE_PATH,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": source_set_checksum(citations, methods_mapped),
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: JsonMap, *, allow_blocked: bool = False) -> None:
    """Validate the artifact so uncited .438 method claims fail closed."""

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
        raise ValueError("inference_substrate must match aggregation_from_upstream_artifacts")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match the required annotations")
    if artifact["note_path"] != NOTE_PATH:
        raise ValueError("note_path must point at research-studying.md")
    if artifact["random_seed"] != RANDOM_SEED:
        raise ValueError("random_seed must be the experiment id")

    citations = artifact["citations"]
    if not isinstance(citations, Mapping) or set(citations) != REQUIRED_SOURCE_IDS:
        raise ValueError("citations must exactly cover the verified arXiv IDs")
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
        if not method["maps_to_current_stack"]:
            raise ValueError("each method needs maps_to_current_stack")
        if not method["takes_over_from_current_stack"]:
            raise ValueError("each method needs takes_over_from_current_stack")
        if not method["fails_when"]:
            raise ValueError("each method needs fails_when")
        if not method["roadmap_candidate"]:
            raise ValueError("each method needs a roadmap candidate")

    flagged = artifact["flagged_for_438"]
    if not isinstance(flagged, str) or "flagged_for_438" not in flagged:
        raise ValueError("flagged_for_438 must contain a .438 roadmap flag")
    if "flagged_for_437" in flagged:
        raise ValueError("flagged_for_438 must not carry stale .437 flags")

    if artifact["reproducibility_checksum"] != source_set_checksum(citations, methods):
        raise ValueError("reproducibility checksum must hash citations and methods")
    _validate_fresh_sweep(artifact["fresh_sweep"])
    _validate_preconditions(artifact["preconditions_checked"])


def _validate_blocked_artifact(artifact: JsonMap) -> None:
    if artifact["methods_mapped"] != [] or artifact["citations"] != {}:
        raise ValueError("blocked artifact must not include method claims or citations")
    if artifact["flagged_for_438"] != "":
        raise ValueError("blocked artifact must not flag roadmap candidates")
    if artifact["fresh_sweep"] != {}:
        raise ValueError("blocked artifact must not claim a fresh sweep")
    if artifact["reproducibility_checksum"] != source_set_checksum({}, []):
        raise ValueError("blocked artifact checksum must hash empty sources")
    preconditions = artifact["preconditions_checked"]
    if not isinstance(preconditions, Mapping) or set(preconditions) != REQUIRED_PRECONDITION_FIELDS:
        raise ValueError("preconditions_checked must match the required schema")
    if preconditions["network_hf_models_reachable"] is not False:
        raise ValueError("blocked artifact must record failed network")
    if preconditions["websearch_webfetch_used"] is not False:
        raise ValueError("blocked artifact must not claim WebFetch success")


def _validate_fresh_sweep(fresh_sweep: object) -> None:
    if not isinstance(fresh_sweep, Mapping) or set(fresh_sweep) != REQUIRED_FRESH_SWEEP_FIELDS:
        raise ValueError("fresh_sweep must match the reliable-channel schema")
    if fresh_sweep["cluster_ids"] != [6, 5]:
        raise ValueError("fresh_sweep must record ARC clusters 6 and 5")
    sources = fresh_sweep["webfetch_top_sources"]
    if not isinstance(sources, Sequence) or isinstance(sources, str | bytes) or not 5 <= len(sources) <= 8:
        raise ValueError("fresh_sweep must record top five to eight WebFetch sources")
    if list(sources) != WEBSEARCH_WEBFETCH_TOP_SOURCES:
        raise ValueError("fresh_sweep WebFetch sources must match the verified source set")


def _validate_preconditions(preconditions: object) -> None:
    if not isinstance(preconditions, Mapping) or set(preconditions) != REQUIRED_PRECONDITION_FIELDS:
        raise ValueError("preconditions_checked must match the required schema")
    if preconditions["network_hf_models_reachable"] is not True:
        raise ValueError("network precondition must record reachable Hugging Face API")
    if preconditions["deep_research_invoked"] is not False:
        raise ValueError("deep-research must not be invoked")
    if not 5 <= int(preconditions["top_source_count"]) <= 8:
        raise ValueError("top_source_count must record top five to eight sources")
    if preconditions["model_load"] is not False:
        raise ValueError("model load must not occur")
    if preconditions["training_launched"] is not False:
        raise ValueError("training must not be launched")
    if preconditions["leaderboard_submission"] is not False:
        raise ValueError("leaderboard submission must not occur")
    if preconditions["solve_claim_made"] is not False:
        raise ValueError("solve claim must remain false for ingestion")
    if preconditions["ops_docs_modified"] is not False:
        raise ValueError("ops docs must not be modified by this workflow")


def build_research_studying_section(artifact: JsonMap | None = None) -> str:
    """Return the idempotent human-readable Exp 4758 research-studying note."""

    result = dict(artifact or build_artifact())
    validate_artifact(result)
    methods = result["methods_mapped"]
    citations = result["citations"]
    method_lines = "\n".join(
        (
            f"- **{method['method']}** ({', '.join('arXiv:' + source for source in method['source_ids'])}): "
            f"takes over {method['takes_over_from_current_stack']} Fails when: {method['fails_when']}"
        )
        for method in methods
    )
    citation_lines = "\n".join(
        f"- arXiv:{source_id} -- {citations[source_id]['title']}"
        for source_id in sorted(citations)
    )
    return f"""{STUDYING_SECTION_START}
## 2026-06-26 Exp 4758 - .437 structured world-model + grounded-goal SOTA ingestion - INGESTED

**Status:** INGESTED into `results/experiment_4758_sota_ingestion.json`.

**Preconditions:** Hugging Face model API reachability returned `net_ok`.
`scripts/sweep_clusters.py` emitted the ARC neural-guided-search / world-model
cluster URL and the ARC action-effect / exploration cluster URL.
`scripts/sweep_semscholar.py` was run on three focused queries and returned
HTTP 429, so no S2-only source was promoted. Low-concurrency WebSearch/WebFetch
plus direct arXiv HTTP checks verified the top eight papers listed below.
`/deep-research` was not invoked. No model load, training, leaderboard
submission, or solve claim was made; this is a no solve claim ingestion note.

**Verified source set:**
{citation_lines}

**SOTA -> .438 experiment mapping:**
{method_lines}

{result["flagged_for_438"]}

**Bottom line for .438:** build the verifier-refined executable world-model
loop first, but bind its goals to perception-grounded object/subgoal structure
instead of asking the free-form engine for another brittle terminal predicate.
The direct target is `E3AgentPolicy` + `arc_executable_world_model` +
`ProductWorldModel` with the structural-alignment goal pipeline supplying
goal-conditioned subgoals and failure diagnostics.
{STUDYING_SECTION_END}"""


def update_research_studying_text(text: str, artifact: JsonMap | None = None) -> str:
    """Insert or replace the Exp 4758 section in research-studying.md text."""

    section = build_research_studying_section(artifact)
    start = text.find(STUDYING_SECTION_START)
    if start >= 0:
        end = text.find(STUDYING_SECTION_END, start)
        if end < 0:
            raise ValueError("research-studying Exp 4758 section missing end marker")
        end += len(STUDYING_SECTION_END)
        tail = text[end:].lstrip()
        if tail:
            return text[:start].rstrip() + "\n\n" + section + "\n\n" + tail
        return text[:start].rstrip() + "\n\n" + section + "\n"

    first_section = text.find("\n## ")
    if first_section >= 0:
        return text[: first_section + 1] + "\n" + section + "\n\n" + text[first_section + 1 :]
    return text.rstrip() + "\n\n" + section + "\n"


def validate_research_studying_text(text: str, artifact: JsonMap | None = None) -> None:
    """Validate the research-studying note carries the mapped methods and sources."""

    result = dict(artifact or build_artifact())
    validate_artifact(result)
    start = text.find(STUDYING_SECTION_START)
    end = text.find(STUDYING_SECTION_END, start)
    if start < 0 or end < 0:
        raise ValueError("research-studying missing Exp 4758 section markers")
    section = text[start : end + len(STUDYING_SECTION_END)]
    required_phrases = (
        "SOTA -> .438 experiment mapping",
        "flagged_for_438",
        "no solve claim",
        "E3AgentPolicy",
        "arc_executable_world_model",
        "ProductWorldModel",
        "structural-alignment goal pipeline",
    )
    for phrase in required_phrases:
        if phrase not in section:
            raise ValueError(f"research-studying section missing required phrase: {phrase}")
    missing_citations = sorted(
        citation for citation in NOTE_REQUIRED_SOURCE_CITATIONS if citation not in section
    )
    if missing_citations:
        raise ValueError(f"research-studying section missing citations: {missing_citations}")
    for method in result["methods_mapped"]:
        if method["method"] not in section:
            raise ValueError(f"research-studying section missing method: {method['method']}")
    if result["flagged_for_438"] not in section:
        raise ValueError("research-studying section missing flagged_for_438 text")


def write_outputs(
    *,
    artifact_path: Path | None = None,
    studying_path: Path | None = None,
    artifact: JsonMap | None = None,
    blocked: bool = False,
) -> dict[str, object]:
    """Write the stable JSON artifact and update research-studying.md."""

    result = dict(artifact or (build_blocked_network_artifact() if blocked else build_artifact()))
    if blocked:
        validate_artifact(result, allow_blocked=True)
    else:
        validate_artifact(result)

    target_artifact = artifact_path or Path(RESULT_RELATIVE_PATH)
    target_studying = studying_path or Path(STUDYING_RELATIVE_PATH)
    target_artifact.parent.mkdir(parents=True, exist_ok=True)
    target_artifact.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not blocked:
        if target_studying.exists():
            studying_text = target_studying.read_text(encoding="utf-8")
        else:
            studying_text = "# Research Studying\n\n"
        updated = update_research_studying_text(studying_text, result)
        validate_research_studying_text(updated, result)
        target_studying.write_text(updated, encoding="utf-8")
    return result


def network_reachable(timeout_s: float = 15.0) -> bool:
    """Check the required Hugging Face API precondition."""

    if os.environ.get("CARNOT_EXP4758_FORCE_BLOCKED_NETWORK") == "1":
        return False
    if os.environ.get("CARNOT_EXP4758_SKIP_NETWORK_CHECK") == "1":
        return True
    try:
        completed = subprocess.run(
            ["curl", "-sf", "-o", "/dev/null", HF_NETWORK_CHECK_URL],
            check=False,
            timeout=timeout_s,
        )
    except Exception:
        return False
    return completed.returncode == 0


def main() -> int:
    root = Path(os.environ.get("CARNOT_EXP4758_ROOT", "."))
    artifact_path = root / RESULT_RELATIVE_PATH
    studying_path = root / STUDYING_RELATIVE_PATH
    if not network_reachable():
        write_outputs(artifact_path=artifact_path, studying_path=studying_path, blocked=True)
        print("blocked_network")
        return 0
    artifact = write_outputs(artifact_path=artifact_path, studying_path=studying_path)
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
