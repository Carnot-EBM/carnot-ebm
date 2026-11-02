"""Exp 4589 skill-routing and env-adaptive replay SOTA ingestion.

Spec refs: REQ-REPORT-4589, SCENARIO-REPORT-4589.

This module records a literature-synthesis artifact. It does not run the ARC
agent, train a model, or submit to the leaderboard. The deterministic writer
makes the markdown note, result JSON, and studying-queue update safe to rerun.
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
        "target_track",
        "takes_over_current_a1_a3_mechanisms",
        "fails_when",
        "v424_candidate",
    }
)
REQUIRED_CITATION_FIELDS = frozenset({"title", "url", "http_status"})
REQUIRED_PRECONDITION_FIELDS = frozenset(
    {
        "agents_md_read",
        "codex_md_read",
        "sweep_clusters_help_exit_0",
        "arxiv_api_reachable",
        "research_studying_filtered",
        "research_references_filtered",
        "exp4580_spec_read",
        "exp4582_spec_read",
        "exp4580_artifact_read",
        "exp4582_artifact_read",
        "research_studying_updated",
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
VALID_TARGET_TRACKS = frozenset({"feature_skill_routing", "env_adaptive_replay"})
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
DEFAULT_HONEST_VERDICT = "complete: sota_ingestion_feature_router_mapped"
DEFAULT_RANDOM_SEED = 4589
RESEARCH_NOTE_RELATIVE_PATH = (
    "docs/research-notes/sota-ingestion-feature-router-423-2026-06-22.md"
)
STUDYING_SECTION_START = "<!-- EXP4589-FEATURE-ROUTER-SOTA-START -->"
STUDYING_SECTION_END = "<!-- EXP4589-FEATURE-ROUTER-SOTA-END -->"

FIELD_PRINCIPLES = {
    "honest_verdict": "terminal prefix; complete: sota_ingestion_feature_router_mapped.",
    "inference_substrate": (
        "aggregation_from_upstream_artifacts -- literature synthesis, no model load "
        "(100us floor)."
    ),
    "methods_mapped": (
        "the 3-5 strongest methods with REAL arXiv IDs -- the shoulders-of-giants "
        "anti-rederivation check."
    ),
    "citations_verified": (
        "every method claim cites a verifiable arXiv ID/URL -- the no-fabrication "
        "bar (same as any results artifact)."
    ),
    "flagged_for_next_roadmap": (
        "the strongest method flagged as a .424 candidate -- closes the "
        "discover->ingest->plan loop."
    ),
    "preconditions_checked": (
        "records resources verified; pre-empts missing-resource fabrication."
    ),
    "research_note_path": "repo-relative markdown path for deterministic parsing.",
    "random_seed": "bare integer seed for reproducible artifact generation.",
    "field_principles": "principle annotations for every top-level artifact field.",
}

CITATIONS_VERIFIED = {
    "2603.22455": {
        "title": "SkillRouter: Skill Routing for LLM Agents at Scale",
        "url": "https://arxiv.org/abs/2603.22455",
        "http_status": 200,
    },
    "2605.12039": {
        "title": (
            "SkillGraph: Skill-Augmented Reinforcement Learning for Agents via "
            "Evolving Skill Graphs"
        ),
        "url": "https://arxiv.org/abs/2605.12039",
        "http_status": 200,
    },
    "2606.06079": {
        "title": "SkillComposer: Learning to Evolve Agent Skills for Specification and Generalization",
        "url": "https://arxiv.org/abs/2606.06079",
        "http_status": 200,
    },
    "2602.01869": {
        "title": "Skill-Pro: Learning Reusable Skills from Experience via Non-Parametric PPO for LLM Agents",
        "url": "https://arxiv.org/abs/2602.01869",
        "http_status": 200,
    },
    "2602.08234": {
        "title": "SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning",
        "url": "https://arxiv.org/abs/2602.08234",
        "http_status": 200,
    },
    "2512.24156": {
        "title": "Graph-Based Exploration for ARC-AGI-3 Interactive Reasoning Tasks",
        "url": "https://arxiv.org/abs/2512.24156",
        "http_status": 200,
    },
    "2603.24621": {
        "title": "ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence",
        "url": "https://arxiv.org/abs/2603.24621",
        "http_status": 200,
    },
}
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(f"arXiv:{source}" for source in CITATIONS_VERIFIED)

SWEEP_CLUSTER_URLS = [
    (
        "http://export.arxiv.org/api/query?search_query="
        '(abs:"affordance"+OR+abs:"action+effect"+OR+abs:"clickability"+OR+'
        'abs:"frame+prediction"+OR+abs:"intrinsic+motivation"+OR+'
        'abs:"directed+exploration"+OR+abs:"novelty+search")+AND+'
        '(abs:"reinforcement+learning"+OR+abs:"agent"+OR+'
        'abs:"exploration"+OR+abs:"interactive+environment"+OR+abs:"ARC")'
        "&start=0&max_results=8&sortBy=submittedDate&sortOrder=descending"
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
    "SkillRouter Skill Routing LLM Agents at Scale",
    "SkillGraph skill augmented RL evolving skill graphs",
    "SkillComposer learning to evolve agent skills specification generalization Skill-Pro reusable skills non-parametric PPO",
    "ARC-AGI-3 graph-based exploration environment drift replay skill routing",
]

WEBSEARCH_WEBFETCH_TOP_SOURCES = [
    CITATIONS_VERIFIED["2603.22455"]["url"],
    CITATIONS_VERIFIED["2605.12039"]["url"],
    CITATIONS_VERIFIED["2606.06079"]["url"],
    CITATIONS_VERIFIED["2602.01869"]["url"],
    CITATIONS_VERIFIED["2602.08234"]["url"],
    CITATIONS_VERIFIED["2512.24156"]["url"],
    CITATIONS_VERIFIED["2603.24621"]["url"],
]

DEFAULT_PRECONDITIONS_CHECKED = {
    "agents_md_read": True,
    "codex_md_read": True,
    "sweep_clusters_help_exit_0": True,
    "arxiv_api_reachable": True,
    "research_studying_filtered": True,
    "research_references_filtered": True,
    "exp4580_spec_read": True,
    "exp4582_spec_read": True,
    "exp4580_artifact_read": True,
    "exp4582_artifact_read": True,
    "research_studying_updated": True,
    "sweep_clusters_used": True,
    "sweep_clusters_urls": SWEEP_CLUSTER_URLS,
    "sweep_semscholar_used": True,
    "sweep_semscholar_queries": S2_QUERIES,
    "sweep_semscholar_arxiv_ids": [],
    "sweep_semscholar_rate_limited_queries": S2_QUERIES[:3],
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
        "method": "SkillRouter full-text retrieve-and-rerank over the solver toolkit",
        "source_ids": ["2603.22455"],
        "target_track": "feature_skill_routing",
        "takes_over_current_a1_a3_mechanisms": (
            "Exp 4582 currently classifies first-K early-play effects into a small "
            "mechanic class and routes to a fixed approach, but the artifact ended "
            "as a null because winner generation stayed mostly absent. SkillRouter "
            "takes over Exp 4582 by retrieving and reranking full skill bodies from "
            "the arc_solver_kit toolkit before planning, rather than exposing only "
            "short names or hand-coded class labels."
        ),
        "fails_when": (
            "skill bodies are hidden at routing time, the library is not execution "
            "validated, retrieval is evaluated only on seen public games, or routing "
            "is left as a final candidate reranker after the winning action is absent."
        ),
        "v424_candidate": (
            "flagged_for_v424: SkillRouter-style full-body routing over arc_solver_kit "
            "skills for the Exp 4582 seen-to-hidden feature router"
        ),
    },
    {
        "method": "SkillGraph plus SkillRL evolving skill-library structure",
        "source_ids": ["2605.12039", "2602.08234"],
        "target_track": "feature_skill_routing",
        "takes_over_current_a1_a3_mechanisms": (
            "Exp 4582 stores a flat mechanic-class to approach preference learned from "
            "positive and negative traces. SkillGraph and SkillRL take over that "
            "Exp 4582 policy by turning traces into a structured skill graph or "
            "SkillBank with dependency edges, failure lessons, and adaptive retrieval "
            "for general and task-specific heuristics."
        ),
        "fails_when": (
            "the graph grows from raw trajectories without deduplication, failed traces "
            "are not distilled into negative routing constraints, or dependency edges "
            "are trusted without replay through the current ARC environment."
        ),
        "v424_candidate": (
            "flagged_for_v424: SkillGraph/SkillRL library maintenance behind the Exp "
            "4582 router, with replay-gated skill insertion"
        ),
    },
    {
        "method": "SkillComposer and Skill-Pro skill merge into executable reusable procedures",
        "source_ids": ["2606.06079", "2602.01869"],
        "target_track": "feature_skill_routing",
        "takes_over_current_a1_a3_mechanisms": (
            "Exp 4582 exposes residual gaps by mechanic class, and Exp 4580 banks "
            "trajectory evidence when a procedure is replayable. SkillComposer and "
            "Skill-Pro take over this handoff by creating, improving, merging, and "
            "verifying reusable skills with activation/execution/termination conditions "
            "instead of repeatedly deriving per-game recipes."
        ),
        "fails_when": (
            "merged skills become too abstract to trigger reliably, task-specific "
            "skills are allowed to leak public-game identity, or reusable procedures "
            "lack an offline replay gate before entering the toolkit."
        ),
        "v424_candidate": (
            "flagged_for_v424: SkillComposer/Skill-Pro merge pass over Exp 4582 "
            "mechanic gaps before persisting new solver primitives"
        ),
    },
    {
        "method": "Graph-Based Exploration as env-derived robust replay generator",
        "source_ids": ["2512.24156"],
        "target_track": "env_adaptive_replay",
        "takes_over_current_a1_a3_mechanisms": (
            "Exp 4580 succeeded by closing the package gap and recovering sc25 with "
            "an env-adaptive replay path instead of trusting stale flat coordinates. "
            "Graph-Based Exploration takes over Exp 4580's replay fragility by "
            "re-deriving state-action paths from current frames, visited-state graphs, "
            "and untested action priorities when frozen replay no longer matches."
        ),
        "fails_when": (
            "the game requires hidden carry-state induction, graph keys ignore layout "
            "version drift, exploration budgets do not preserve action efficiency, or "
            "newly recovered paths are not rechecked through offline reproduction."
        ),
        "v424_candidate": (
            "flagged_for_v424: graph-explore replay regeneration for Exp 4580 "
            "version-drift rows before falling back to stale coordinate banks"
        ),
    },
    {
        "method": "ARC-AGI-3 efficiency-and-drift evaluation contract",
        "source_ids": ["2603.24621"],
        "target_track": "env_adaptive_replay",
        "takes_over_current_a1_a3_mechanisms": (
            "Exp 4580's headline is not a new solve; it converts 53 reproduced public "
            "levels into 53 live-submittable levels by requiring environment-matched "
            "trajectories or an env-adaptive resolver. The ARC-AGI-3 report takes over "
            "the Exp 4580 acceptance contract: novel interactive environments require "
            "goal inference, dynamics modeling, planning, and action-efficient replay "
            "under changing layouts."
        ),
        "fails_when": (
            "the next roadmap optimizes only the public-game package gap, ignores "
            "actions-to-first-levelup, treats official and community harness evidence "
            "as identical, or promotes env-adaptive replay without held-out layout drift."
        ),
        "v424_candidate": (
            "flagged_for_v424: ARC-AGI-3 drift-aware live-submittable gate for every "
            "Exp 4580 replay primitive"
        ),
    },
]

FLAGGED_FOR_NEXT_ROADMAP = (
    "flagged_for_v424: implement SkillRouter-style full-body routing over "
    "arc_solver_kit skills, backed by SkillGraph/SkillRL trace distillation and "
    "graph-explore env-adaptive replay regeneration for drifted rows "
    "(arXiv:2603.22455 + arXiv:2605.12039 + arXiv:2512.24156)"
)


def _fail(message: str) -> None:
    raise ValueError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _list_value(value: object) -> bool:
    return isinstance(value, list)


def _nonempty_list(value: object) -> bool:
    return isinstance(value, list) and bool(value)


def build_artifact(
    *,
    methods_mapped: Sequence[Mapping[str, object]] | None = None,
    preconditions_checked: Mapping[str, object] | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    honest_verdict: str = DEFAULT_HONEST_VERDICT,
) -> dict[str, object]:
    """Build the deterministic artifact embedded in the markdown note."""

    chosen_methods = DEFAULT_METHODS_MAPPED if methods_mapped is None else methods_mapped
    chosen_preconditions = (
        DEFAULT_PRECONDITIONS_CHECKED if preconditions_checked is None else preconditions_checked
    )
    return {
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "methods_mapped": [dict(method) for method in chosen_methods],
        "citations_verified": {key: dict(value) for key, value in CITATIONS_VERIFIED.items()},
        "flagged_for_next_roadmap": FLAGGED_FOR_NEXT_ROADMAP,
        "preconditions_checked": dict(chosen_preconditions),
        "research_note_path": RESEARCH_NOTE_RELATIVE_PATH,
        "random_seed": random_seed,
        "field_principles": dict(FIELD_PRINCIPLES),
    }


def _validate_preconditions(row: object) -> None:
    _require(
        isinstance(row, Mapping) and set(row) == REQUIRED_PRECONDITION_FIELDS,
        "preconditions_checked must have exactly the required fields",
    )
    expected_true = {
        "agents_md_read": "AGENTS.md",
        "codex_md_read": "CODEX.md",
        "sweep_clusters_help_exit_0": "sweep_clusters.py --help",
        "arxiv_api_reachable": "arXiv API",
        "research_studying_filtered": "research-studying.md filtered pass",
        "research_references_filtered": "research-references.md filtered pass",
        "exp4580_spec_read": "REQ-CAPSTONE-4580 spec",
        "exp4582_spec_read": "REQ-CAPSTONE-4582 spec",
        "exp4580_artifact_read": "Exp 4580 env-adaptive replay artifact",
        "exp4582_artifact_read": "Exp 4582 feature-router artifact",
        "research_studying_updated": "research-studying.md update",
        "sweep_clusters_used": "sweep_clusters.py",
        "sweep_semscholar_used": "sweep_semscholar.py",
    }
    for key, label in expected_true.items():
        _require(row.get(key) is True, f"preconditions_checked must record {label}")

    expected_false = {
        "deep_research_invoked": "deep-research",
        "live_llm_inference": "live inference",
        "training_launched": "training",
        "leaderboard_submission": "leaderboard",
        "live_solve_claim": "live solve",
        "ops_docs_modified": "ops docs",
        "research_conductor_modified": "scripts/research_conductor.py",
    }
    for key, label in expected_false.items():
        _require(row.get(key) is False, f"preconditions_checked must record no {label}")

    _require(
        row.get("sweep_clusters_urls") == SWEEP_CLUSTER_URLS,
        "preconditions_checked must record the focused cluster 5/6 URLs",
    )
    _require(
        _nonempty_list(row.get("sweep_semscholar_queries")),
        "preconditions_checked must record Semantic Scholar queries",
    )
    _require(
        _list_value(row.get("sweep_semscholar_arxiv_ids")),
        "preconditions_checked must record Semantic Scholar arXiv ids",
    )
    _require(
        _nonempty_list(row.get("sweep_semscholar_rate_limited_queries")),
        "preconditions_checked must record Semantic Scholar HTTP 429 queries",
    )
    _require(
        _nonempty_list(row.get("arxiv_http_200_verified_ids"))
        and set(CITATIONS_VERIFIED).issubset(set(row["arxiv_http_200_verified_ids"])),
        "preconditions_checked must include all verified arXiv ids",
    )
    _require(
        _nonempty_list(row.get("websearch_webfetch_top_sources"))
        and set(WEBSEARCH_WEBFETCH_TOP_SOURCES).issubset(
            set(row["websearch_webfetch_top_sources"])
        ),
        "preconditions_checked must include WebSearch/WebFetch source URLs",
    )


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the artifact before writing or embedding it."""

    fields = set(artifact)
    missing = REQUIRED_ARTIFACT_FIELDS - fields
    extra = fields - REQUIRED_ARTIFACT_FIELDS
    _require(not missing, f"artifact missing required fields: {sorted(missing)}")
    _require(not extra, f"artifact has unexpected fields: {sorted(extra)}")

    verdict = artifact["honest_verdict"]
    _require(
        isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES),
        "honest_verdict must use a terminal prefix",
    )
    _require(
        verdict == DEFAULT_HONEST_VERDICT,
        "honest_verdict must match REQ-REPORT-4589 complete path",
    )
    _require(
        artifact["inference_substrate"] == INFERENCE_SUBSTRATE,
        "inference_substrate must declare aggregation from upstream artifacts",
    )
    _require(
        artifact["field_principles"] == FIELD_PRINCIPLES
        and set(artifact["field_principles"]) == REQUIRED_ARTIFACT_FIELDS,
        "field_principles must match REQ-REPORT-4589",
    )
    _require(
        isinstance(artifact["random_seed"], int) and not isinstance(artifact["random_seed"], bool),
        "random_seed must be an integer",
    )
    _require(
        artifact["research_note_path"] == RESEARCH_NOTE_RELATIVE_PATH,
        "research_note_path must be the repo-relative note path",
    )

    citations = artifact["citations_verified"]
    _require(isinstance(citations, Mapping), "citations_verified must be a mapping")
    _require(
        citations == CITATIONS_VERIFIED,
        "citations_verified must match verified arXiv metadata",
    )
    for citation in citations.values():
        _require(
            isinstance(citation, Mapping) and set(citation) == REQUIRED_CITATION_FIELDS,
            "each citation must include title, url, and http_status",
        )

    methods = artifact["methods_mapped"]
    _require(isinstance(methods, list), "methods_mapped must be a list")
    _require(3 <= len(methods) <= 5, "methods_mapped must contain three to five methods")
    used_method_sources: set[str] = set()
    seen_tracks: set[str] = set()
    for method in methods:
        _require(
            isinstance(method, Mapping) and set(method) == REQUIRED_METHOD_FIELDS,
            "each methods_mapped entry must have exactly the required fields",
        )
        _require(
            _nonempty_list(method.get("source_ids"))
            and set(method["source_ids"]).issubset(set(CITATIONS_VERIFIED)),
            "methods_mapped source_ids must use verified citations",
        )
        used_method_sources.update(str(source) for source in method["source_ids"])
        track = method["target_track"]
        _require(track in VALID_TARGET_TRACKS, "methods_mapped target_track must be valid")
        seen_tracks.add(str(track))
        for key in (
            "method",
            "takes_over_current_a1_a3_mechanisms",
            "fails_when",
            "v424_candidate",
        ):
            _require(
                isinstance(method[key], str) and bool(method[key].strip()),
                f"methods_mapped field {key!r} must be a non-empty string",
            )
        mapping = method["takes_over_current_a1_a3_mechanisms"]
        _require(
            "Exp 4580" in mapping or "Exp 4582" in mapping,
            "methods_mapped must map onto Exp 4580 or Exp 4582",
        )
        _require(
            method["v424_candidate"].startswith("flagged_for_v424:"),
            "methods_mapped v424_candidate must flag a .424 input",
        )
    _require(
        used_method_sources == set(CITATIONS_VERIFIED),
        "methods_mapped must use every verified citation",
    )
    _require(
        seen_tracks == set(VALID_TARGET_TRACKS),
        "methods_mapped must cover feature routing and env-adaptive replay",
    )

    _require(
        artifact["flagged_for_next_roadmap"] == FLAGGED_FOR_NEXT_ROADMAP
        and str(artifact["flagged_for_next_roadmap"]).startswith("flagged_for_v424:"),
        "flagged_for_next_roadmap must match the verified .424 candidate",
    )
    _validate_preconditions(artifact["preconditions_checked"])


def _artifact_json(artifact: Mapping[str, Any]) -> str:
    return json.dumps(artifact, indent=2, sort_keys=True)


def render_research_note(artifact: Mapping[str, Any]) -> str:
    """Render markdown with the artifact block first for automated parsing."""

    validate_artifact(artifact)
    source_line = ", ".join(f"arXiv:{source}" for source in CITATIONS_VERIFIED)
    return f"""# Feature-router and env-adaptive replay SOTA ingestion .423 - 2026-06-22

```json
{_artifact_json(artifact)}
```

Reliable channel only: `scripts/sweep_clusters.py`, `scripts/sweep_semscholar.py`,
arXiv abs-page HTTP-200 checks, and low-concurrency WebSearch/WebFetch of seven
top skill-routing / skill-library / env-adaptive replay sources. Preconditions
passed before any claim was promoted: `.venv/bin/python scripts/sweep_clusters.py
--help` exited zero and `curl -sf -o /dev/null
https://export.arxiv.org/api/query?search_query=all:test` confirmed arXiv API
reachability. `scripts/sweep_clusters.py 5 --max-results 8` and
`scripts/sweep_clusters.py 6 --max-results 8` emitted the focused cluster URLs.
`scripts/sweep_semscholar.py` ran four focused queries; three returned HTTP 429
and no S2-only arXiv ID was promoted. No `/deep-research` call was made. No training,
live LLM inference, leaderboard submission, or live solve was launched.
No ops/status/traceability files or `scripts/research_conductor.py` were modified.

Already-discovered corpus read through a skill-routing and env-adaptive replay
filter: `research-studying.md`, `research-references.md`,
`openspec/capabilities/capstone/spec.md` at `REQ-CAPSTONE-4580` and
`REQ-CAPSTONE-4582`, `results/experiment_4580_live_submission_gap_close.json`,
and `results/experiment_4582_feature_router_transfer.json`. Exp 4580 succeeded
as a packaging/replay result: live-submittable levels rose from 33 to 53 and
`sc25` was recovered by env-adaptive replay. Exp 4582 was an honest no-value
feature-router null: router and baseline both measured 0.04 generic transfer,
random-route control did not pass, false-negative risk stayed open, and residual
generation gaps remain by mechanic class.

Sources checked: {source_line}.

## Per-Method Mapping

- **SkillRouter full-text retrieve-and-rerank** (arXiv:2603.22455): the strongest
  A3 lesson is that Exp 4582 should route over full solver-skill bodies, not only
  a small mechanic-class label. It fails if hidden skill bodies or stale metadata
  become the routing substrate, or if routing happens after the winning action is
  already absent from the candidate pool.
- **SkillGraph plus SkillRL evolving skill-library structure** (arXiv:2605.12039,
  arXiv:2602.08234): turn Exp 4582 positive and negative traces into a graph or
  SkillBank with dependency edges, failure lessons, and adaptive retrieval. It
  fails if graph edges are trusted without current-environment replay validation.
- **SkillComposer and Skill-Pro executable skill reuse** (arXiv:2606.06079,
  arXiv:2602.01869): create, improve, merge, and verify skills so recurring ARC
  mechanics become reusable procedures with activation/execution/termination
  conditions. It fails when merged skills are too abstract or public-game identity
  leaks into the reusable primitive.
- **Graph-Based Exploration as robust replay regeneration** (arXiv:2512.24156):
  take Exp 4580's env-adaptive replay success and re-derive action paths from the
  current frame/state graph when flat coordinates drift. It fails on hidden-state
  or mechanic-limited games unless the graph key and action probes expose the
  latent state.
- **ARC-AGI-3 efficiency-and-drift evaluation contract** (arXiv:2603.24621): keep
  Exp 4580 honest by treating live-submittable replay, environment match, and
  actions-to-first-levelup as the score-facing contract. It fails if .424 optimizes
  only the public-game package gap and ignores hidden-layout or action-efficiency
  generalization.

## .424 Candidate

{FLAGGED_FOR_NEXT_ROADMAP}

The practical next experiment should keep Exp 4580's offline reproduction and
env-match gates, keep Exp 4582's random-route and false-negative controls, and
replace the flat route table with full-body skill retrieval plus graph-validated
replay regeneration. SkillRouter supplies the selection mechanism, SkillGraph and
SkillRL supply the evolving library structure, SkillComposer and Skill-Pro supply
the skill create/merge/reuse operations, Graph-Based Exploration supplies the
layout-drift replay fallback, and ARC-AGI-3 supplies the action-efficient
evaluation contract.
"""


def artifact_from_note(note: str) -> dict[str, object]:
    """Extract the machine-readable JSON block from the markdown note."""

    start_marker = "```json\n"
    end_marker = "\n```"
    start = note.find(start_marker)
    _require(start != -1, "research note missing machine-readable JSON block")
    start += len(start_marker)
    end = note.find(end_marker, start)
    _require(end != -1, "research note missing machine-readable JSON block terminator")
    artifact = json.loads(note[start:end])
    validate_artifact(artifact)
    return artifact


def validate_research_note(note: str) -> None:
    """Check citations, required language, and the embedded artifact."""

    missing_sources = sorted(
        source for source in NOTE_REQUIRED_SOURCE_CITATIONS if source not in note
    )
    _require(
        not missing_sources,
        f"research note missing verified source citations: {missing_sources}",
    )
    required_phrases = [
        "Reliable channel",
        "sweep_clusters.py",
        "sweep_semscholar.py",
        "/deep-research",
        "No training",
        "skill-routing",
        "env-adaptive replay",
        "Exp 4580",
        "Exp 4582",
        "SkillRouter",
        "SkillGraph",
        "SkillComposer",
        "Skill-Pro",
        "SkillRL",
        "Graph-Based Exploration",
        "ARC-AGI-3",
        "flagged_for_v424",
        "aggregation_from_upstream_artifacts",
        "scripts/research_conductor.py",
    ]
    for phrase in required_phrases:
        _require(phrase in note, f"research note missing required phrase: {phrase}")
    artifact_from_note(note)


RESEARCH_NOTE = render_research_note(build_artifact())


def render_research_studying_entry() -> str:
    """Render the idempotent research-studying queue update."""

    return f"""{STUDYING_SECTION_START}
## 2026-06-22 Exp 4589 - .423 feature-router SOTA ingestion - INGESTED

**Status:** INGESTED into `docs/research-notes/sota-ingestion-feature-router-423-2026-06-22.md`
and `results/experiment_4589_sota_ingestion_feature_router.json`.

**Preconditions:** `scripts/sweep_clusters.py --help` succeeded; the arXiv API
reachability check succeeded; `scripts/sweep_clusters.py` clusters 5 and 6
emitted focused URLs; `scripts/sweep_semscholar.py` returned HTTP 429 for three
focused queries and no S2-only arXiv ID was promoted. Top sources were verified
by arXiv abs-page HTTP 200 and low-concurrency WebSearch/WebFetch of the seven
arXiv sources. `/deep-research` was not invoked. No live solve, training run,
live LLM inference, leaderboard submission, ops/status/traceability edit, or
`scripts/research_conductor.py` edit occurred.

**Methods marked ingested:** SkillRouter full-text skill routing
(arXiv:2603.22455), SkillGraph evolving skill graphs (arXiv:2605.12039),
SkillComposer skill create/merge/improve (arXiv:2606.06079), Skill-Pro reusable
procedural skills (arXiv:2602.01869), SkillRL/SkillBank recursive skill
distillation (arXiv:2602.08234), Graph-Based Exploration for ARC-AGI-3
(arXiv:2512.24156), and the ARC-AGI-3 efficiency/drift contract
(arXiv:2603.24621).

{FLAGGED_FOR_NEXT_ROADMAP}
{STUDYING_SECTION_END}"""


def update_research_studying(path: Path) -> None:
    """Insert or replace the Exp 4589 studying-queue section."""

    entry = render_research_studying_entry()
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    start = existing.find(STUDYING_SECTION_START)
    end = existing.find(STUDYING_SECTION_END)
    if start != -1 and end != -1 and end >= start:
        end += len(STUDYING_SECTION_END)
        updated = existing[:start].rstrip() + "\n\n" + entry + "\n\n" + existing[end:].lstrip()
    else:
        updated = existing.rstrip() + "\n\n" + entry + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(updated, encoding="utf-8")


def write_outputs(
    *,
    artifact_path: Path,
    note_path: Path,
    studying_path: Path,
) -> dict[str, object]:
    """Write the result JSON, markdown note, and research-studying queue entry."""

    artifact = build_artifact()
    validate_artifact(artifact)
    validate_research_note(RESEARCH_NOTE)

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(_artifact_json(artifact) + "\n", encoding="utf-8")

    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(RESEARCH_NOTE.rstrip() + "\n", encoding="utf-8")

    update_research_studying(studying_path)
    return artifact


def main() -> int:
    root = Path(os.environ.get("CARNOT_EXP4589_ROOT", "."))
    artifact = write_outputs(
        artifact_path=root / "results/experiment_4589_sota_ingestion_feature_router.json",
        note_path=root / RESEARCH_NOTE_RELATIVE_PATH,
        studying_path=root / "research-studying.md",
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
