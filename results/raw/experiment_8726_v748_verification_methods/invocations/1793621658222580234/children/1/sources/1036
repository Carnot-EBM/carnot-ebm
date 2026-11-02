"""Exp 4637 intrinsic-motivation / action-effect SOTA ingestion.

Spec refs: REQ-ARC-WMTE-4637, SCENARIO-ARC-WMTE-4637.

This is a literature-to-experiment mapping artifact, not a benchmark run. It
records the 2026-06-23 focused pass over dense online curiosity,
learning-progress rewards, noisy-TV guards, and ARC action-effect exploration,
then maps them onto the current A1 dense-curiosity and A2 action-effect stack
for .428 planning.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
from typing import Any


JsonMap = Mapping[str, Any]

RESULT_RELATIVE_PATH = "results/experiment_4637_sota_ingestion_intrinsic_motivation.json"
NOTE_RELATIVE_PATH = (
    "docs/research-notes/"
    "intrinsic-motivation-action-effect-literature-2026-06-23.md"
)
RANDOM_SEED = 4637
HONEST_VERDICT = "success: sota_ingestion_intrinsic_motivation_action_effect_mapped"
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
        "exp4625_artifact_read",
        "offline_live_bridge_note_read",
        "bridge_diagnosis_note_read",
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
        "2604.18701",
        "2509.25438",
        "2102.04399",
        "1705.05363",
        "1810.12894",
        "2601.10904",
        "2603.24621",
        "2512.24156",
        "2605.05138",
    }
)
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(
    f"arXiv:{source_id}" for source_id in REQUIRED_SOURCE_IDS
)
FLAGGED_FOR_NEXT_ROADMAP = [
    (
        "flagged_for_v428: curiosity_critic_learning_progress_dense_reward "
        "(arXiv:2604.18701 + arXiv:2509.25438)"
    ),
    (
        "flagged_for_v428: noisy_tv_aware_action_effect_uncertainty_gate "
        "(arXiv:2102.04399 + arXiv:2509.25438)"
    ),
    (
        "flagged_for_v428: clickability_action_effect_expansion_prior "
        "(arXiv:2601.10904 + arXiv:2603.24621)"
    ),
    (
        "flagged_for_v428: graph_executable_world_model_action_effect_planner "
        "(arXiv:2512.24156 + arXiv:2605.05138)"
    ),
]

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; "
            "success: sota_ingestion_intrinsic_motivation_action_effect_mapped."
        )
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
            "the strongest method(s) flagged as candidate .428 inputs -- closes "
            "discover->ingest->plan->experiment."
        )
    },
    "note_path": {
        "principle": (
            "docs/research-notes/"
            "intrinsic-motivation-action-effect-literature-2026-06-23.md -- "
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
        "principle": "records network reachability verified; pre-empts fabricated citations."
    },
}

CITATIONS_VERIFIED = {
    "2604.18701": {
        "title": (
            "Curiosity-Critic: Cumulative Prediction Error Improvement as a "
            "Tractable Intrinsic Reward for World Model Training"
        ),
        "url": "https://arxiv.org/abs/2604.18701",
        "http_status": 200,
    },
    "2509.25438": {
        "title": "Beyond Noisy-TVs: Noise-Robust Exploration Via Learning Progress Monitoring",
        "url": "https://arxiv.org/abs/2509.25438",
        "http_status": 200,
    },
    "2102.04399": {
        "title": (
            "How to Stay Curious while Avoiding Noisy TVs using Aleatoric "
            "Uncertainty Estimation"
        ),
        "url": "https://arxiv.org/abs/2102.04399",
        "http_status": 200,
    },
    "1705.05363": {
        "title": "Curiosity-driven Exploration by Self-supervised Prediction",
        "url": "https://arxiv.org/abs/1705.05363",
        "http_status": 200,
    },
    "1810.12894": {
        "title": "Exploration by Random Network Distillation",
        "url": "https://arxiv.org/abs/1810.12894",
        "http_status": 200,
    },
    "2601.10904": {
        "title": "ARC Prize 2025: Technical Report",
        "url": "https://arxiv.org/abs/2601.10904",
        "http_status": 200,
    },
    "2603.24621": {
        "title": "ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence",
        "url": "https://arxiv.org/abs/2603.24621",
        "http_status": 200,
    },
    "2512.24156": {
        "title": "Graph-Based Exploration for ARC-AGI-3 Interactive Reasoning Tasks",
        "url": "https://arxiv.org/abs/2512.24156",
        "http_status": 200,
    },
    "2605.05138": {
        "title": "Executable World Models for ARC-AGI-3 in the Era of Coding Agents",
        "url": "https://arxiv.org/abs/2605.05138",
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
    "Curiosity-Critic cumulative prediction error improvement intrinsic reward 2604.18701",
    "intrinsic curiosity module random network distillation prediction error curiosity exploration",
    "learning progress epistemic aleatoric uncertainty exploration reinforcement learning",
    "ARC-AGI-3 clickability action effect CNN ARC Prize 2025 2601.10904",
    "Graph-Based Exploration ARC-AGI-3 2512.24156 Executable World Models 2605.05138",
]
WEBSEARCH_WEBFETCH_TOP_SOURCES = [
    f"https://arxiv.org/abs/{source_id}"
    for source_id in (
        "2604.18701",
        "2509.25438",
        "2102.04399",
        "1705.05363",
        "1810.12894",
        "2601.10904",
        "2603.24621",
        "2512.24156",
        "2605.05138",
    )
]

DEFAULT_PRECONDITIONS_CHECKED = {
    "agents_md_read": True,
    "codex_md_read": True,
    "network_hf_models_reachable": True,
    "exp4625_artifact_read": True,
    "offline_live_bridge_note_read": True,
    "bridge_diagnosis_note_read": True,
    "research_studying_read": True,
    "research_references_read": True,
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
        "method": "Curiosity-Critic / LPM dense learning-progress reward",
        "source_ids": ["2604.18701", "2509.25438"],
        "track": "dense_online_intrinsic_reward",
        "implement_cost_over_current_stack": (
            "medium-high: log the A2 action-effect model's per-transition "
            "prediction error, train a small A1 dense-curiosity critic to estimate "
            "the asymptotic/noise-floor error for each transition class, and feed "
            "only the positive learning-progress residual into expansion priority."
        ),
        "maps_to_current_stack": (
            "A1 dense-curiosity becomes a learnability estimator rather than a raw "
            "surprise score; A2 action-effect keeps the existing frame-change "
            "predictor but receives a dense reward for transitions whose error is "
            "still reducible."
        ),
        "fails_when": (
            "the baseline critic mistakes hidden deterministic state for "
            "irreducible noise, logged transition classes are too sparse to learn "
            "the floor, or the intrinsic residual overwhelms first-win/action "
            "efficiency gates."
        ),
        "roadmap_candidate": FLAGGED_FOR_NEXT_ROADMAP[0],
    },
    {
        "method": "Aleatoric-noise guard for curiosity and action-effect rewards",
        "source_ids": ["2102.04399", "2509.25438"],
        "track": "epistemic_vs_aleatoric_filtering",
        "implement_cost_over_current_stack": (
            "medium: add mean/variance or previous-error heads beside the A2 "
            "action-effect predictor, down-weight high-variance transitions, and "
            "use the guarded score as an A1 dense-curiosity eligibility mask."
        ),
        "maps_to_current_stack": (
            "A1 dense-curiosity stops rewarding noisy-TV-like screen changes; A2 "
            "action-effect can still predict clickability, but only transitions "
            "classified as learnable receive exploration priority."
        ),
        "fails_when": (
            "the variance head is undertrained, rare but decisive transitions look "
            "aleatoric early, or the guard suppresses the only probe that reveals a "
            "hidden register."
        ),
        "roadmap_candidate": FLAGGED_FOR_NEXT_ROADMAP[1],
    },
    {
        "method": "ICM/RND prediction-error curiosity as a cheap control floor",
        "source_ids": ["1705.05363", "1810.12894"],
        "track": "prediction_error_baseline_floor",
        "implement_cost_over_current_stack": (
            "low-to-medium: reuse the current frame-delta/action-effect tensors to "
            "train an inverse-dynamics feature space for ICM and a fixed-target "
            "embedding for RND, then compare both against the existing A1 dense "
            "curiosity score under matched action budgets."
        ),
        "maps_to_current_stack": (
            "A1 dense-curiosity gets a cheap baseline that validates whether any "
            "learned intrinsic reward beats raw prediction error; A2 action-effect "
            "uses the same transitions so no new environment substrate is needed."
        ),
        "fails_when": (
            "raw prediction error chases stochastic animation, RND novelty decays "
            "before the useful mechanic is discovered, or the baseline improves "
            "coverage without reducing actions-to-first-win."
        ),
        "roadmap_candidate": "candidate_for_v428_control: icm_rnd_prediction_error_floor",
    },
    {
        "method": "Clickability / action-effect expansion prior under ARC efficiency scoring",
        "source_ids": ["2601.10904", "2603.24621"],
        "track": "clickability_action_effect_expansion",
        "implement_cost_over_current_stack": (
            "low-to-medium: keep the current A2 action-effect CNN as a candidate "
            "expansion prior, train it on cached human/self-play transition rows, "
            "and gate it by first-win action efficiency instead of treating it as a "
            "post-hoc reranker."
        ),
        "maps_to_current_stack": (
            "A1 dense-curiosity supplies the dense learnability signal that tells "
            "the explorer when to keep probing; A2 action-effect turns the signal "
            "into fewer no-op or non-changing clicks under ARC-AGI-3's efficiency "
            "metric."
        ),
        "fails_when": (
            "the predictor only ranks a fixed candidate pool, useful actions are "
            "not generated in the first place, or clickability improves frame "
            "change while failing to improve level completion and action count."
        ),
        "roadmap_candidate": FLAGGED_FOR_NEXT_ROADMAP[2],
    },
    {
        "method": "Graph/executable-world-model action-effect planner",
        "source_ids": ["2512.24156", "2605.05138"],
        "track": "state_graph_world_model_action_effect",
        "implement_cost_over_current_stack": (
            "medium-high: persist a graph of tested state-action pairs, route "
            "untested but learnable edges through A1 dense-curiosity, and promote "
            "only verified A2 action-effect transitions into executable planning "
            "or shortest-path reuse."
        ),
        "maps_to_current_stack": (
            "A1 dense-curiosity chooses which frontier edges are worth testing; A2 "
            "action-effect supplies the transition predictions and the graph/world "
            "model prevents repeated actions that cannot change the state."
        ),
        "fails_when": (
            "state hashing aliases hidden registers, executable models pass "
            "prefix observations but fail held-out transitions, or graph "
            "exploration broadens coverage while spending more actions than the "
            "current live explorer."
        ),
        "roadmap_candidate": FLAGGED_FOR_NEXT_ROADMAP[3],
    },
]

STUDYING_SECTION_START = "<!-- EXP4637-INTRINSIC-ACTION-EFFECT-SOTA-START -->"
STUDYING_SECTION_END = "<!-- EXP4637-INTRINSIC-ACTION-EFFECT-SOTA-END -->"
STUDYING_SECTION = f"""{STUDYING_SECTION_START}
## 2026-06-23 Exp 4637 - .427 intrinsic-motivation/action-effect SOTA ingestion - INGESTED

**Status:** INGESTED into `{NOTE_RELATIVE_PATH}`.

**Filtered track:** dense online intrinsic motivation and action-effect
prediction for the .427 live-exploration problem: replace raw surprise with
learning progress, suppress noisy-TV transitions, and turn clickability /
action-effect predictions into fewer wasted actions.

**Preconditions:** Hugging Face model API reachability returned `net_ok`.
`scripts/sweep_clusters.py` emitted focused ARC exploration and neural-guided
search URLs. `scripts/sweep_semscholar.py` returned HTTP 429 for all five
focused queries and no S2-only source was promoted. Low-concurrency
WebSearch/WebFetch plus direct arXiv HTTP checks verified arXiv:2604.18701,
arXiv:2509.25438, arXiv:2102.04399, arXiv:1705.05363, arXiv:1810.12894,
arXiv:2601.10904, arXiv:2603.24621, arXiv:2512.24156, and arXiv:2605.05138.
`/deep-research` was not invoked.

**Methods marked ingested:** Curiosity-Critic cumulative prediction-error
improvement, Learning Progress Monitoring, aleatoric-noise curiosity guards,
ICM/RND prediction-error controls, ARC clickability/action-effect expansion,
graph-based exploration, and executable-world-model action-effect planning.

flagged_for_v428: curiosity_critic_learning_progress_dense_reward
(arXiv:2604.18701 + arXiv:2509.25438)

flagged_for_v428: noisy_tv_aware_action_effect_uncertainty_gate
(arXiv:2102.04399 + arXiv:2509.25438)

flagged_for_v428: clickability_action_effect_expansion_prior
(arXiv:2601.10904 + arXiv:2603.24621)

flagged_for_v428: graph_executable_world_model_action_effect_planner
(arXiv:2512.24156 + arXiv:2605.05138)

**Bottom line for .428:** build Curiosity-Critic/LPM-style learning-progress
rewards over the existing action-effect predictor first, add the aleatoric guard
before any scored-agent use, and evaluate graph/executable action-effect
planning only behind matched action-efficiency no-regression gates.
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
    """Build and validate the REQ-ARC-WMTE-4637 mapping artifact."""

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
    """Validate the artifact so uncited intrinsic-reward claims fail closed."""

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
        raise ValueError("note_path must point at the 2026-06-23 intrinsic note")
    if artifact["deep_research_not_used"] is not True:
        raise ValueError("deep_research_not_used must be true")
    if artifact["random_seed"] != RANDOM_SEED:
        raise ValueError("random_seed must be the bare integer 4637")

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
        if not isinstance(mapping, str) or "A1" not in mapping or "A2" not in mapping:
            raise ValueError("method mapping must name A1 and A2")
        for field in REQUIRED_METHOD_FIELDS - {"source_ids"}:
            value = method[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"method {field} must be a non-empty string")

    flagged = artifact["flagged_for_next_roadmap"]
    if flagged != FLAGGED_FOR_NEXT_ROADMAP or not all(
        "flagged_for_v428" in item for item in flagged
    ):
        raise ValueError("flagged_for_next_roadmap must name the .428 candidates")

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
        raise ValueError("ops docs must not be modified by Exp 4637")


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
    """Check that the paired note maps verified sources to .428 work."""

    artifact_from_note(markdown)
    required_phrases = (
        "Fresh-pass provenance",
        "SOTA -> experiment mapping",
        "A1 dense-curiosity",
        "A2 action-effect",
        "Bottom line for the .428 roadmap",
        "flagged_for_v428",
        "Curiosity-Critic",
        "Learning Progress Monitoring",
        "ICM",
        "RND",
        "Graph-Based Exploration",
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
    return f"""# Intrinsic motivation / action-effect literature ingestion 2026-06-23

```json
{artifact_json}
```

## Fresh-pass provenance

Read `AGENTS.md`, `CODEX.md`,
`results/experiment_4625_sota_ingestion_offline_live_bridge.json`,
`docs/research-notes/offline-live-bridge-literature-2026-06-23.md`,
`docs/research-notes/arc-representation-not-the-bottleneck-2026-06-23.md`,
`research-studying.md`, and `research-references.md`. The filtered track was
the .427 headline open problem: GENERATE better live exploration through dense
online intrinsic-motivation / learning-progress signals plus action-effect
prediction for action efficiency, feeding candidate methods forward to .428.

Reliable-channel pass, not `/deep-research`:
- `.venv/bin/python -c "import urllib.request; urllib.request.urlopen('https://huggingface.co/api/models', timeout=10); print('net_ok')"`
- `.venv/bin/python scripts/sweep_clusters.py 5 --max-results 8`
- `.venv/bin/python scripts/sweep_clusters.py 6 --max-results 8`
- `.venv/bin/python scripts/sweep_semscholar.py - --limit 8` with five focused queries
- low-concurrency WebSearch/WebFetch of the top arXiv papers
- direct arXiv HTTP checks for all cited IDs

Semantic Scholar returned HTTP 429 for the five focused queries, so no
Semantic-Scholar-only source was promoted. Direct arXiv HTTP checks returned
200 for arXiv:2604.18701, arXiv:2509.25438, arXiv:2102.04399,
arXiv:1705.05363, arXiv:1810.12894, arXiv:2601.10904, arXiv:2603.24621,
arXiv:2512.24156, and arXiv:2605.05138. No live LLM inference, No training,
No leaderboard submission, no model load, and no live solve claim were run or
made. `scripts/research_conductor.py`, `ops/changelog.md`, and `ops/status.md`
were not edited by this workflow.

## SOTA -> experiment mapping

## Curiosity-Critic plus Learning Progress Monitoring

**Sources:** Curiosity-Critic, arXiv:2604.18701; Learning Progress Monitoring,
arXiv:2509.25438.

**Mapping to A1 dense-curiosity / A2 action-effect:** the current stack already
has an action-effect predictor, but raw prediction error is a bad dense reward
because it keeps paying for stochastic or already-learned transitions.
Curiosity-Critic turns the reward into improvement over an estimated
asymptotic error baseline, while Learning Progress Monitoring rewards model
improvement rather than novelty. The .428 implementation should attach this
critic to the A2 transition-error stream and expose the residual as the A1
dense-curiosity score.

**Implementation cost over current stack:** medium-high. Needs transition-error
logging, a small scalar baseline/error critic, held-out transition-class checks,
and action-efficiency no-regression gates.

**Fails when:** hidden deterministic state is misclassified as irreducible
noise, the baseline critic is too data-starved, or the dense reward becomes a
goal in itself rather than a way to reduce actions-to-first-win.

## Aleatoric-noise guard for curiosity rewards

**Sources:** Aleatoric Mapping Agents, arXiv:2102.04399; Learning Progress
Monitoring, arXiv:2509.25438.

**Mapping to A1 dense-curiosity / A2 action-effect:** the noisy-TV failure mode
is directly relevant to ARC sprites and UI effects: frame changes can be real
but not controllably useful. Add a variance/previous-error head beside the A2
action-effect predictor, then let A1 dense-curiosity prioritize only transitions
whose uncertainty appears reducible.

**Implementation cost over current stack:** medium. Adds one or two heads to
the predictor and a calibration split for stochastic vs learnable transitions.

**Fails when:** the guard suppresses rare discovery actions, the variance head
learns visual noise rather than action-conditioned unpredictability, or the
gate improves coverage but not action efficiency.

## ICM and RND as prediction-error control floors

**Sources:** ICM / self-supervised prediction curiosity, arXiv:1705.05363; RND,
arXiv:1810.12894.

**Mapping to A1 dense-curiosity / A2 action-effect:** ICM and RND are not the
recommended endpoint, but they are the control floor .428 must beat. Both can
be implemented over the same A2 frame-delta/action-effect tensors, giving A1 a
cheap check that learning-progress rewards really outperform raw prediction
error or novelty under matched action budgets.

**Implementation cost over current stack:** low-to-medium. Reuse transition
features, train the small curiosity heads, and compare first-win/action counts
against the current dense-curiosity loop.

**Fails when:** prediction error locks onto stochastic animations, RND novelty
vanishes too early, or a coverage gain does not translate into fewer actions.

## Clickability / action-effect expansion prior

**Sources:** ARC Prize 2025 technical report, arXiv:2601.10904; ARC-AGI-3
benchmark report, arXiv:2603.24621. Supplemental operational context from the
existing corpus: StochasticGoose-style clickability/action-effect code is not
promoted as an arXiv source, so the claim carried forward here is the
experiment design: use action-effect prediction under ARC's efficiency metric,
not a leaderboard-reproduction claim.

**Mapping to A1 dense-curiosity / A2 action-effect:** the .422/.427 lesson is
that ranking an already-bad candidate pool is not enough. A1 dense-curiosity
should decide where the explorer still has learnable action effects, and A2
should use the action-effect predictor during expansion so no-op actions are
not generated as often.

**Implementation cost over current stack:** low-to-medium. The predictor exists;
the work is moving it from post-hoc ranker to candidate-expansion prior and
measuring first-win/action efficiency.

**Fails when:** the useful action is absent from the candidate generator,
clickability predicts frame change without goal relevance, or the predictor is
trained on seen games and fails on hidden mechanics.

## Graph/executable-world-model action-effect planner

**Sources:** Graph-Based Exploration for ARC-AGI-3, arXiv:2512.24156;
Executable World Models for ARC-AGI-3, arXiv:2605.05138.

**Mapping to A1 dense-curiosity / A2 action-effect:** Graph-Based Exploration
shows the action-efficiency value of recording tested state-action pairs and
prioritizing untested edges; Executable World Models shows the higher-cost
variant where verified transitions become a planning substrate. For .428, A1
dense-curiosity picks which untested edges are learnable, and A2 action-effect
decides which predicted transitions are worth adding to the graph/model.

**Implementation cost over current stack:** medium-high. Needs stable state
hashing, tested-action ledgers, held-out transition verification, and a strict
rule that graph/world-model planning cannot increase action count at equal
first-win rate.

**Fails when:** hidden registers alias in the graph, executable models overfit
prefix observations, or systematic exploration spends more actions than the
current live explorer.

## Bottom line for the .428 roadmap

1. Build `flagged_for_v428: curiosity_critic_learning_progress_dense_reward`
   first: Curiosity-Critic arXiv:2604.18701 plus LPM arXiv:2509.25438 gives the
   dense reward that should replace raw surprise in A1.
2. Pair it immediately with
   `flagged_for_v428: noisy_tv_aware_action_effect_uncertainty_gate`:
   arXiv:2102.04399 and arXiv:2509.25438 are the guard against rewarding
   irreducible or useless frame changes.
3. Keep ICM arXiv:1705.05363 and RND arXiv:1810.12894 as matched-budget
   baselines, not as the final method, because .428 needs evidence that
   learning progress beats cheap raw prediction error.
4. Promote `flagged_for_v428: clickability_action_effect_expansion_prior` only
   if it changes candidate generation, not just ranking; the ARC reports
   arXiv:2601.10904 and arXiv:2603.24621 justify the action-efficiency target.
5. Use graph/executable planning, Graph-Based Exploration arXiv:2512.24156 plus
   Executable World Models arXiv:2605.05138, as the second-stage planner after
   dense curiosity and action-effect prediction pass no-regression gates.
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
    """Write the default Exp 4637 deliverables under the repository root."""

    repo_root = Path(os.environ.get("CARNOT_EXP4637_ROOT", Path(__file__).resolve().parents[2]))
    artifact = write_outputs(
        artifact_path=repo_root / RESULT_RELATIVE_PATH,
        note_path=repo_root / NOTE_RELATIVE_PATH,
        studying_path=repo_root / "research-studying.md",
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
