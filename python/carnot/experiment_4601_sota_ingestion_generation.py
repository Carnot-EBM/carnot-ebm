"""Exp 4601 generation and world-model SOTA ingestion.

Spec refs: REQ-REPORT-4601, SCENARIO-REPORT-4601.

This module records a literature-synthesis artifact, not a benchmark result.
The current ARC wall is candidate generation: Exp 4592 made one extra winner
appear after wiring, while Exp 4594 showed the current goal-energy prior still
adds no value. The artifact below maps current world-model and generation SOTA
onto that gap without launching models, training, or leaderboard submissions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
from typing import Any


REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "honest_verdict",
        "inference_substrate",
        "methods_mapped",
        "citations_verified",
        "flagged_for_next_roadmap",
        "preconditions_checked",
        "research_note_path",
        "random_seed",
        "field_principles",
    }
)
REQUIRED_METHOD_FIELDS = frozenset(
    {
        "method",
        "source_ids",
        "generation_track",
        "takes_over_current_a1_a3_mechanisms",
        "fails_when",
        "v425_candidate",
    }
)
REQUIRED_CITATION_FIELDS = frozenset({"title", "url", "http_status"})
REQUIRED_PRECONDITION_FIELDS = frozenset(
    {
        "agents_md_read",
        "codex_md_read",
        "sweep_clusters_help_exit_0",
        "arxiv_api_reachable",
        "research_references_424_filtered",
        "planner_confirmation_addendum_filtered",
        "research_studying_filtered",
        "research_studying_updated",
        "exp4592_artifact_read",
        "exp4594_artifact_read",
        "sweep_clusters_used",
        "sweep_clusters_urls",
        "sweep_semscholar_used",
        "sweep_semscholar_queries",
        "sweep_semscholar_arxiv_ids",
        "sweep_semscholar_rate_limited_queries",
        "sweep_semscholar_failed_queries",
        "arxiv_http_200_verified_ids",
        "websearch_webfetch_top_sources",
        "deep_research_invoked",
        "live_llm_inference",
        "training_launched",
        "leaderboard_submission",
        "live_solve_claim",
        "ops_docs_modified",
        "research_conductor_modified",
    }
)
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
VALID_GENERATION_TRACKS = frozenset(
    {
        "executable_world_model_induction",
        "curriculum_perception_grounding",
        "skill_controller_synthesis",
        "exploration_oracle_curiosity",
        "adaptive_symbolic_world_model",
    }
)
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
DEFAULT_HONEST_VERDICT = "complete: sota_ingestion_generation_mapped"
DEFAULT_RANDOM_SEED = 4601
RESEARCH_NOTE_RELATIVE_PATH = (
    "docs/research-notes/sota-ingestion-generation-world-model-424-2026-06-22.md"
)
STUDYING_SECTION_START = "<!-- EXP4601-GENERATION-SOTA-START -->"
STUDYING_SECTION_END = "<!-- EXP4601-GENERATION-SOTA-END -->"
FLAGGED_FOR_NEXT_ROADMAP = (
    "flagged_for_v425: executable_world_model_energy_config_space_generation_prior "
    "(arXiv:2605.05138 + arXiv:2510.04542)"
)

FIELD_PRINCIPLES = {
    "honest_verdict": "terminal prefix; complete: sota_ingestion_generation_mapped.",
    "inference_substrate": (
        "aggregation_from_upstream_artifacts -- literature synthesis, no model load "
        "(100us floor)."
    ),
    "methods_mapped": (
        "the 3-5 strongest GENERATION/world-model-induction methods with REAL "
        "arXiv IDs -- the shoulders-of-giants anti-rederivation check."
    ),
    "citations_verified": (
        "every method claim cites a verifiable arXiv ID/URL -- the no-fabrication "
        "bar (same as any results artifact)."
    ),
    "flagged_for_next_roadmap": (
        "the strongest method flagged as a .425 candidate -- closes the "
        "discover->ingest->plan loop."
    ),
    "preconditions_checked": (
        "records resources verified; pre-empts missing-resource fabrication."
    ),
    "research_note_path": "repo-relative markdown path for deterministic parsing.",
    "random_seed": "bare integer seed for reproducible artifact generation.",
    "field_principles": "principle annotations for every top-level artifact field.",
}

REQUIRED_VERIFIED_SOURCE_IDS = frozenset(
    {
        "2510.04542",
        "2507.12821",
        "2510.12088",
        "2502.13200",
        "2505.19095",
        "2603.17683",
        "2502.00225",
        "2605.05138",
        "2605.10999",
        "2603.24621",
    }
)

CITATIONS_VERIFIED = {
    "2510.04542": {
        "title": "Code World Models for General Game Playing",
        "url": "https://arxiv.org/abs/2510.04542",
        "http_status": 200,
    },
    "2507.12821": {
        "title": "Assessing Adaptive World Models in Machines with Novel Games",
        "url": "https://arxiv.org/abs/2507.12821",
        "http_status": 200,
    },
    "2510.12088": {
        "title": (
            "One Life to Learn: Inferring Symbolic World Models for Stochastic "
            "Environments from Unguided Exploration"
        ),
        "url": "https://arxiv.org/abs/2510.12088",
        "http_status": 200,
    },
    "2502.13200": {
        "title": "Learning To Explore With Predictive World Model Via Self-Supervised Learning",
        "url": "https://arxiv.org/abs/2502.13200",
        "http_status": 200,
    },
    "2505.19095": {
        "title": "ScreenExplorer: Training a Vision-Language Model for Diverse Exploration in Open GUI World",
        "url": "https://arxiv.org/abs/2505.19095",
        "http_status": 200,
    },
    "2603.17683": {
        "title": (
            "Sensi: Learn One Thing at a Time -- Curriculum-Based Test-Time "
            "Learning for LLM Game Agents"
        ),
        "url": "https://arxiv.org/abs/2603.17683",
        "http_status": 200,
    },
    "2502.00225": {
        "title": "Should You Use Your Large Language Model to Explore or Exploit?",
        "url": "https://arxiv.org/abs/2502.00225",
        "http_status": 200,
    },
    "2605.05138": {
        "title": "Executable World Models for ARC-AGI-3 in the Era of Coding Agents",
        "url": "https://arxiv.org/abs/2605.05138",
        "http_status": 200,
    },
    "2605.10999": {
        "title": "SkillGen: Verified Inference-Time Agent Skill Synthesis",
        "url": "https://arxiv.org/abs/2605.10999",
        "http_status": 200,
    },
    "2603.24621": {
        "title": "ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence",
        "url": "https://arxiv.org/abs/2603.24621",
        "http_status": 200,
    },
    "2605.16986": {
        "title": "Skills on the Fly: Test-Time Adaptive Skill Synthesis for LLM Agents",
        "url": "https://arxiv.org/abs/2605.16986",
        "http_status": 200,
    },
    "2605.08083": {
        "title": "LLMs Improving LLMs: Agentic Discovery for Test-Time Scaling",
        "url": "https://arxiv.org/abs/2605.08083",
        "http_status": 200,
    },
}
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(
    f"arXiv:{source_id}" for source_id in CITATIONS_VERIFIED
)

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
    "ARC-AGI-3 executable world models Code World Models Sensi SkillGen candidate generation",
    "world model induction novel games exploration oracle predictive world model ScreenExplorer",
]

WEBSEARCH_WEBFETCH_TOP_SOURCES = [
    "https://arxiv.org/abs/2605.05138",
    "https://arxiv.org/abs/2510.04542",
    "https://arxiv.org/abs/2603.17683",
    "https://arxiv.org/abs/2605.10999",
    "https://arxiv.org/abs/2502.00225",
    "https://arxiv.org/abs/2507.12821",
    "https://arxiv.org/abs/2510.12088",
    "https://arxiv.org/abs/2603.24621",
]

DEFAULT_PRECONDITIONS_CHECKED = {
    "agents_md_read": True,
    "codex_md_read": True,
    "sweep_clusters_help_exit_0": True,
    "arxiv_api_reachable": True,
    "research_references_424_filtered": True,
    "planner_confirmation_addendum_filtered": True,
    "research_studying_filtered": True,
    "research_studying_updated": True,
    "exp4592_artifact_read": True,
    "exp4594_artifact_read": True,
    "sweep_clusters_used": True,
    "sweep_clusters_urls": SWEEP_CLUSTER_URLS,
    "sweep_semscholar_used": True,
    "sweep_semscholar_queries": S2_QUERIES,
    "sweep_semscholar_arxiv_ids": [],
    "sweep_semscholar_rate_limited_queries": [S2_QUERIES[0]],
    "sweep_semscholar_failed_queries": [S2_QUERIES[1]],
    "arxiv_http_200_verified_ids": list(CITATIONS_VERIFIED),
    "websearch_webfetch_top_sources": WEBSEARCH_WEBFETCH_TOP_SOURCES,
    "deep_research_invoked": False,
    "live_llm_inference": False,
    "training_launched": False,
    "leaderboard_submission": False,
    "live_solve_claim": False,
    "ops_docs_modified": False,
    "research_conductor_modified": False,
}

DEFAULT_METHODS_MAPPED = [
    {
        "method": "Executable code world models plus verified planning",
        "source_ids": ["2605.05138", "2510.04542", "2603.24621"],
        "generation_track": "executable_world_model_induction",
        "takes_over_current_a1_a3_mechanisms": (
            "Exp 4592 showed wiring can make one extra winner appear, but the "
            "toolkit still mostly emits no winning candidate. Executable World "
            "Models and Code World Models take over Exp 4592 by generating an "
            "explicit Python transition model, verifying it against observed "
            "transitions, and planning through it. They take over Exp 4594 by "
            "using objective energy as the trust/goal prior that selects and "
            "repairs model candidates before action generation."
        ),
        "fails_when": (
            "the visible-state parser is wrong, the transition verifier accepts a "
            "near-identity or overfit model, the private-set harness exposes a "
            "leakage assumption, or the plan is scored only by the win oracle."
        ),
        "v425_candidate": FLAGGED_FOR_NEXT_ROADMAP,
    },
    {
        "method": "Sensi curriculum test-time learning with perception-gated generation",
        "source_ids": ["2603.17683", "2603.24621"],
        "generation_track": "curriculum_perception_grounding",
        "takes_over_current_a1_a3_mechanisms": (
            "Sensi maps onto Exp 4592 as the warning and control for the LLM tail "
            "generator: split perception from action, advance through a small "
            "curriculum, and measure whether the agent can read the grid before "
            "asking it to generate a plan. For Exp 4594, the objective energy "
            "should gate curriculum advancement and reject perception-incoherent "
            "states rather than only rank final plans."
        ),
        "fails_when": (
            "the LLM reads raw grid text incorrectly. Sensi v2 reports zero solved "
            "levels despite high sample efficiency because the bottleneck moved to "
            "perceptual grounding, which exactly matches Carnot's generation-not-"
            "ranking diagnosis."
        ),
        "v425_candidate": "flagged_for_v425: sensi_perception_gate_for_llm_tail_generator",
    },
    {
        "method": "Verified inference-time skill and controller synthesis",
        "source_ids": ["2605.10999", "2605.16986", "2605.08083"],
        "generation_track": "skill_controller_synthesis",
        "takes_over_current_a1_a3_mechanisms": (
            "Exp 4592 currently wires known toolkit skills, but unseen mechanics "
            "need new candidate procedures. SkillGen, Skills on the Fly, and "
            "AutoTTS take over by synthesizing a temporary skill or controller "
            "from successful and failed trajectories, then verifying its net "
            "effect. Exp 4594's energy prior becomes the fitness signal for "
            "repairs, regressions, and controller pruning."
        ),
        "fails_when": (
            "the synthesized skill is only prose, the skill is not executed against "
            "matched with/without controls, failed trajectories are omitted, or "
            "the controller is tuned on seen public games without hidden-style "
            "variant checks."
        ),
        "v425_candidate": "flagged_for_v425: verified_skill_synthesis_over_arc_solver_kit_failures",
    },
    {
        "method": "LLM exploration oracle plus predictive-world-model curiosity",
        "source_ids": ["2502.00225", "2502.13200", "2505.19095"],
        "generation_track": "exploration_oracle_curiosity",
        "takes_over_current_a1_a3_mechanisms": (
            "Exp 4592 needs candidate action sets that are larger than a fixed "
            "router pool but smaller than blind search. The exploration-oracle "
            "pattern asks a model or heuristic to propose semantically plausible "
            "actions, then lets cheap search and environment feedback dispose of "
            "them. Exp 4594's energy prior becomes curiosity/novelty and predicted "
            "progress energy, not a terminal reranker."
        ),
        "fails_when": (
            "the action semantics are not grounded in visible objects, the oracle "
            "is treated as the exploiter instead of a candidate-set generator, or "
            "curiosity rewards no-op diversity rather than goal-relevant state "
            "changes."
        ),
        "v425_candidate": "flagged_for_v425: semantic_action_set_generator_plus_energy_search",
    },
    {
        "method": "Adaptive symbolic world-model induction from novel-game exploration",
        "source_ids": ["2507.12821", "2510.12088"],
        "generation_track": "adaptive_symbolic_world_model",
        "takes_over_current_a1_a3_mechanisms": (
            "The novel-games and One Life lines take over Exp 4592 by treating "
            "first contact as rapid symbolic world-model induction from unguided "
            "exploration rather than selection from an existing pool. They take "
            "over Exp 4594 by making objective energy an epistemic prior: prefer "
            "candidate models and action probes that explain more transitions and "
            "reduce uncertainty about goal-relevant dynamics."
        ),
        "fails_when": (
            "the exploration trace is too short to identify hidden registers, the "
            "symbolic vocabulary cannot represent the mechanic, or the energy "
            "prior penalizes uncertainty so strongly that it avoids informative "
            "experiments."
        ),
        "v425_candidate": "flagged_for_v425: symbolic_world_model_induction_with_epistemic_energy_prior",
    },
]

STUDYING_SECTION = f"""{STUDYING_SECTION_START}
## 2026-06-22 Exp 4601 - .424 generation SOTA ingestion - INGESTED

**Status:** INGESTED into `{RESEARCH_NOTE_RELATIVE_PATH}`.

**Filtered track:** candidate generation on first contact, executable/symbolic
world-model induction, perceptual grounding, verified skill/controller
synthesis, exploration oracles, and objective energy as a generation prior.

**Preconditions:** `.venv/bin/python scripts/sweep_clusters.py --help`
succeeded and the arXiv API reachability check succeeded. Cluster helpers 5 and
6 emitted focused exploration/world-model URLs. Semantic Scholar returned HTTP
429 for the focused ARC/CWM/Sensi/SkillGen query and HTTP 500 for the broader
world-model exploration query, so no S2-only source was promoted. Direct arXiv
HTTP checks verified all cited IDs. `/deep-research` was not invoked. No live
LLM inference, training run, leaderboard submission, ops/status/traceability
edit, or `scripts/research_conductor.py` edit occurred.

**Methods marked ingested:** Executable World Models plus Code World Models
(arXiv:2605.05138, arXiv:2510.04542, arXiv:2603.24621), Sensi perceptual
grounding and curriculum test-time learning (arXiv:2603.17683), verified
skill/controller synthesis (arXiv:2605.10999, arXiv:2605.16986,
arXiv:2605.08083), exploration-oracle / predictive-world-model curiosity
(arXiv:2502.00225, arXiv:2502.13200, arXiv:2505.19095), and adaptive symbolic
world-model induction for novel games (arXiv:2507.12821, arXiv:2510.12088).

Exp 4592 status mapped honestly: `winner_generated=2/25`, improving over the
1/25 baseline but leaving the generation wall mostly open. Exp 4594 status
mapped honestly: `complete: goal_energy_prior_no_value_honest_null_gap_sharpened`.

flagged_for_v425: executable_world_model_energy_config_space_generation_prior
(arXiv:2605.05138 + arXiv:2510.04542).

**Bottom line for .425:** make executable world-model induction the candidate
generator, and use objective energy as a trust/goal/repair prior inside
generation rather than another final reranker.
{STUDYING_SECTION_END}
"""


def build_artifact(
    *,
    methods_mapped: Sequence[Mapping[str, Any]] = DEFAULT_METHODS_MAPPED,
    citations_verified: Mapping[str, Mapping[str, Any]] = CITATIONS_VERIFIED,
    preconditions_checked: Mapping[str, Any] = DEFAULT_PRECONDITIONS_CHECKED,
    flagged_for_next_roadmap: str = FLAGGED_FOR_NEXT_ROADMAP,
    honest_verdict: str = DEFAULT_HONEST_VERDICT,
) -> dict[str, object]:
    """Build and validate the REQ-REPORT-4601 mapping artifact."""

    artifact: dict[str, object] = {
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "methods_mapped": [dict(method) for method in methods_mapped],
        "citations_verified": {
            source_id: dict(citation)
            for source_id, citation in citations_verified.items()
        },
        "flagged_for_next_roadmap": flagged_for_next_roadmap,
        "preconditions_checked": dict(preconditions_checked),
        "research_note_path": RESEARCH_NOTE_RELATIVE_PATH,
        "random_seed": DEFAULT_RANDOM_SEED,
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the JSON contract so uncited generation claims fail closed."""

    missing = REQUIRED_ARTIFACT_FIELDS.difference(artifact)
    extra = set(artifact).difference(REQUIRED_ARTIFACT_FIELDS)
    if missing:
        raise ValueError(f"artifact missing required fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"artifact has unexpected fields: {sorted(extra)}")

    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must use a terminal prefix")
    if verdict != DEFAULT_HONEST_VERDICT:
        raise ValueError(f"honest_verdict must equal {DEFAULT_HONEST_VERDICT!r}")

    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match the required annotations")
    if artifact["random_seed"] != DEFAULT_RANDOM_SEED:
        raise ValueError("random_seed must be the bare integer 4601")
    if artifact["research_note_path"] != RESEARCH_NOTE_RELATIVE_PATH:
        raise ValueError("research_note_path must point at the Exp 4601 note")

    citations = artifact["citations_verified"]
    if not isinstance(citations, dict) or not REQUIRED_VERIFIED_SOURCE_IDS.issubset(citations):
        raise ValueError("citations_verified must include all required source IDs")
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
    if not isinstance(methods, list) or len(methods) < 3 or len(methods) > 5:
        raise ValueError("methods_mapped must contain three to five methods")
    for method in methods:
        if not isinstance(method, dict) or set(method) != REQUIRED_METHOD_FIELDS:
            raise ValueError(
                "each method must contain exactly method, source_ids, "
                "generation_track, takes_over_current_a1_a3_mechanisms, "
                "fails_when, and v425_candidate"
            )
        source_ids = method["source_ids"]
        if not isinstance(source_ids, list) or not source_ids:
            raise ValueError("method source_ids must be a non-empty list")
        if any(source_id not in citations for source_id in source_ids):
            raise ValueError("method source_ids must all have verified citations")
        if method["generation_track"] not in VALID_GENERATION_TRACKS:
            raise ValueError("method generation_track is not recognized")
        mapping = method["takes_over_current_a1_a3_mechanisms"]
        if "Exp 4592" not in mapping and "Exp 4594" not in mapping:
            raise ValueError("method mapping must name Exp 4592 or Exp 4594")
        for field in REQUIRED_METHOD_FIELDS - {"source_ids"}:
            value = method[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"method {field} must be a non-empty string")

    flagged = artifact["flagged_for_next_roadmap"]
    if flagged != FLAGGED_FOR_NEXT_ROADMAP or "flagged_for_v425" not in flagged:
        raise ValueError("flagged_for_next_roadmap must name the .425 executable-world-model candidate")

    preconditions = artifact["preconditions_checked"]
    if not isinstance(preconditions, dict) or set(preconditions) != REQUIRED_PRECONDITION_FIELDS:
        raise ValueError("preconditions_checked must contain the exact required fields")
    if preconditions["sweep_clusters_help_exit_0"] is not True:
        raise ValueError("sweep_clusters.py --help precondition must be true")
    if preconditions["arxiv_api_reachable"] is not True:
        raise ValueError("arXiv API precondition must be true")
    if preconditions["deep_research_invoked"] is not False:
        raise ValueError("deep-research must not be invoked")
    if preconditions["live_llm_inference"] is not False:
        raise ValueError("live LLM inference must not run")
    if preconditions["training_launched"] is not False:
        raise ValueError("training must not launch")
    if preconditions["leaderboard_submission"] is not False:
        raise ValueError("leaderboard submission must not happen")
    if preconditions["ops_docs_modified"] is not False:
        raise ValueError("ops docs must not be modified by Exp 4601")
    if preconditions["research_conductor_modified"] is not False:
        raise ValueError("research_conductor.py must not be modified")


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
    """Check that the paired note maps verified sources to .425 work."""

    artifact_from_note(markdown)
    required_phrases = (
        "Fresh-pass provenance",
        "Exp 4592 A1",
        "Exp 4594 A3",
        "SOTA -> experiment mapping",
        "Executable world models",
        "Code World Models",
        "Sensi",
        "objective energy",
        "No training",
        "No live LLM inference",
        "No leaderboard submission",
        "flagged_for_v425",
    )
    missing_phrases = [phrase for phrase in required_phrases if phrase not in markdown]
    if missing_phrases:
        raise ValueError(f"research note missing required phrase(s): {missing_phrases}")
    json_block_end = markdown.find("```\n\n## Fresh-pass provenance")
    prose = markdown[json_block_end:] if json_block_end != -1 else markdown
    missing_sources = [source for source in NOTE_REQUIRED_SOURCE_CITATIONS if source not in prose]
    if missing_sources:
        raise ValueError(f"research note missing verified source citations: {missing_sources}")
    if "perceptual-grounding wall" not in markdown:
        raise ValueError("research note must preserve the Sensi perceptual-grounding wall")


def _make_research_note(artifact: Mapping[str, object]) -> str:
    artifact_json = json.dumps(artifact, indent=2, sort_keys=True)
    return f"""# SOTA ingestion 2026-06-22: candidate generation and world-model induction for .425

```json
{artifact_json}
```

## Fresh-pass provenance

Read `AGENTS.md`, `CODEX.md`, the `.424` sweep and planner-confirmation
addendum in `research-references.md`, `research-studying.md`, Exp 4592
(`results/experiment_4592_generation_completeness_wiring.json`), and Exp 4594
(`results/experiment_4594_goal_energy_generation_prior.json`). The filtered
track was candidate generation, first-contact world-model induction,
perceptual grounding, verified skill/controller synthesis, exploration oracles,
and objective energy as a generation prior.

Reliable-channel pass, not `/deep-research`:
- `.venv/bin/python scripts/sweep_clusters.py --help`
- `curl -sf -o /dev/null "https://export.arxiv.org/api/query?search_query=all:test"`
- `.venv/bin/python scripts/sweep_clusters.py 5 --max-results 8`
- `.venv/bin/python scripts/sweep_clusters.py 6 --max-results 8`
- `.venv/bin/python scripts/sweep_semscholar.py "ARC-AGI-3 executable world models Code World Models Sensi SkillGen candidate generation" --limit 8`
- `.venv/bin/python scripts/sweep_semscholar.py "world model induction novel games exploration oracle predictive world model ScreenExplorer" --limit 8`

Cluster helper URLs were emitted for exploration/affordance and
world-model/goal-induction tracks. Semantic Scholar returned HTTP 429 on the
focused ARC/CWM/Sensi/SkillGen query and HTTP 500 on the broader world-model
exploration query, so no S2-only paper is promoted. Low-concurrency
WebSearch/WebFetch plus direct arXiv HTTP-200 checks verified arXiv:2510.04542,
arXiv:2507.12821, arXiv:2510.12088, arXiv:2502.13200, arXiv:2505.19095,
arXiv:2603.17683, arXiv:2502.00225, arXiv:2605.05138, arXiv:2605.10999,
arXiv:2603.24621, arXiv:2605.16986, and arXiv:2605.08083.

No training, No live LLM inference, No leaderboard submission, and no live solve
claim were run or made. `ops/changelog.md`, `ops/status.md`,
`_bmad/traceability.md`, and `scripts/research_conductor.py` were not edited by
this workflow.

## Exp 4592 A1 and Exp 4594 A3 status

Exp 4592 is the current A1 wiring reference: `winner_generated=2/25`, up from
the 1/25 baseline. That is a real but small crack in the generation wall; most
held-out variants still do not get a winning candidate.

Exp 4594 is the current A3 objective-energy reference:
`complete: goal_energy_prior_no_value_honest_null_gap_sharpened`. The current
goal-energy prior did not lift winner generation, so the next use of objective
energy should be inside a stronger generator: trust a world model, select a
candidate skill/controller, or drive exploration toward informative states.

## SOTA -> experiment mapping

## Executable world models

**Sources:** Executable World Models, arXiv:2605.05138; Code World Models,
arXiv:2510.04542; ARC-AGI-3 report, arXiv:2603.24621.

**Mapping to Exp 4592/4594:** use the A1 wiring harness to run a generated
Python world model and planner, not just a pre-existing skill route. Use A3
objective energy as model-trust, goal-progress, and repair energy while the
candidate is being generated. This is the strongest .425 candidate because it
directly attacks "winner absent from the pool."

**Failure mode:** the visible-state parser or transition verifier can accept a
near-identity or overfit model; Sensi shows that bad perception can make a
sample-efficient system generate nothing useful.

## Sensi curriculum and perception gate

**Sources:** Sensi, arXiv:2603.17683; ARC-AGI-3 report, arXiv:2603.24621.

**Mapping to Exp 4592/4594:** Sensi is decision-grade negative evidence for
raw LLM-on-grid generation: its v2 curriculum reached high sample efficiency
but solved zero levels because the bottleneck moved to perceptual grounding.
Use it as a .425 diagnostic gate: before the LLM tail generator can plan, it
must pass an object-centric grid-reading check, and A3 energy should reject
perception-incoherent states. This is the perceptual-grounding wall in one line.

**Failure mode:** an LLM can become self-consistent about a wrong grid reading,
making the generated candidate set precise, cheap, and still wrong.

## Verified skill and controller synthesis

**Sources:** SkillGen, arXiv:2605.10999; Skills on the Fly, arXiv:2605.16986;
AutoTTS, arXiv:2605.08083.

**Mapping to Exp 4592/4594:** A1 can synthesize a temporary skill/controller
from successful and failed trajectories when the static toolkit has no winning
route. A3 energy should be the measured with/without fitness signal: repairs,
regressions, and controller-pruning decisions must be execution checked.

**Failure mode:** a prose skill that is not executed and ablated is just another
prompt. It can overfit public games or hide regressions on mechanics the static
toolkit already solved.

## Exploration oracle and predictive curiosity

**Sources:** Should You Use Your LLM to Explore or Exploit, arXiv:2502.00225;
Learning To Explore With Predictive World Model, arXiv:2502.13200;
ScreenExplorer, arXiv:2505.19095.

**Mapping to Exp 4592/4594:** let an LLM or structured heuristic generate a
small semantic action set, then let cheap search and environment feedback test
it. Objective energy becomes curiosity, novelty, and predicted progress during
candidate generation, not a final score on a fixed pool.

**Failure mode:** if the proposed action set is not grounded in visible objects,
the oracle narrows the search in the wrong direction. If curiosity rewards
frame churn or no-op diversity, it can worsen action efficiency.

## Adaptive symbolic world-model induction

**Sources:** Assessing Adaptive World Models in Machines with Novel Games,
arXiv:2507.12821; One Life to Learn, arXiv:2510.12088.

**Mapping to Exp 4592/4594:** treat a new ARC game as a rapid symbolic
world-model-induction task from unguided exploration. A3 energy becomes an
epistemic prior: choose probes and candidate models that explain observed
transitions and reduce uncertainty about goal-relevant dynamics.

**Failure mode:** the trace may be too short, hidden registers may be outside
the symbolic vocabulary, or the energy prior may punish uncertainty so hard that
the agent avoids informative probes.

## Flagged for .425

flagged_for_v425: executable_world_model_energy_config_space_generation_prior
(arXiv:2605.05138 + arXiv:2510.04542).

Bottom line: run executable world-model induction as the .425 candidate
generator. Keep Sensi as the perceptual-grounding guard, use SkillGen-style
synthesis for residual toolkit gaps, and use objective energy as the
trust/goal/repair prior inside generation.
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
    """Write the note, JSON artifact, and idempotent studying update."""

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
    """Write the default Exp 4601 deliverables under the repository root."""

    repo_root = Path(os.environ.get("CARNOT_EXP4601_ROOT", Path(__file__).resolve().parents[2]))
    artifact = write_outputs(
        artifact_path=repo_root / "results/experiment_4601_sota_ingestion_generation.json",
        note_path=repo_root / RESEARCH_NOTE_RELATIVE_PATH,
        studying_path=repo_root / "research-studying.md",
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
