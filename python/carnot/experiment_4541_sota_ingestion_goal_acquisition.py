"""Exp 4541 goal-acquisition SOTA ingestion for the `.420` hand-off.

Spec refs: REQ-REPORT-4541, SCENARIO-REPORT-4541.

This module records a literature-synthesis artifact. It does not run the ARC
agent, train a model, or submit to the leaderboard. The deterministic output
keeps the markdown note, result JSON, and studying-queue update testable.
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
        "takes_over_current_reinduction",
        "fails_when",
        "v420_candidate",
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
        "prior_reinduction_spec_read",
        "prior_reinduction_artifact_read",
        "research_studying_updated",
        "sweep_clusters_used",
        "sweep_clusters_urls",
        "sweep_semscholar_used",
        "sweep_semscholar_queries",
        "sweep_semscholar_arxiv_ids",
        "sweep_semscholar_rate_limited_queries",
        "arxiv_http_200_verified_ids",
        "websearch_webfetch_top_sources",
        "superseded_navigation_reingested",
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
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
DEFAULT_HONEST_VERDICT = "complete: sota_ingestion_goal_acquisition_mapped"
DEFAULT_RANDOM_SEED = 4541
RESEARCH_NOTE_RELATIVE_PATH = "docs/research-notes/arc-goal-acquisition-sota-419.md"
STUDYING_SECTION_START = "<!-- EXP4541-GOAL-ACQUISITION-SOTA-START -->"
STUDYING_SECTION_END = "<!-- EXP4541-GOAL-ACQUISITION-SOTA-END -->"

FIELD_PRINCIPLES = {
    "honest_verdict": "terminal prefix; complete: sota_ingestion_goal_acquisition_mapped.",
    "inference_substrate": (
        "aggregation_from_upstream_artifacts -- literature synthesis, no model load."
    ),
    "methods_mapped": (
        "the 3-5 strongest methods with REAL arXiv IDs -- the "
        "shoulders-of-giants anti-rederivation check."
    ),
    "citations_verified": (
        "every method claim cites a verifiable arXiv ID/URL -- the "
        "no-fabrication bar (same as any results artifact)."
    ),
    "flagged_for_next_roadmap": (
        "the strongest method flagged as a .420 candidate -- closes the "
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
    "2603.24621": {
        "title": "ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence",
        "url": "https://arxiv.org/abs/2603.24621",
        "http_status": 200,
    },
    "2605.05138": {
        "title": "Executable World Models for ARC-AGI-3 in the Era of Coding Agents",
        "url": "https://arxiv.org/abs/2605.05138",
        "http_status": 200,
    },
    "2601.10904": {
        "title": "ARC Prize 2025: Technical Report",
        "url": "https://arxiv.org/abs/2601.10904",
        "http_status": 200,
    },
    "2507.14172": {
        "title": "Self-Improving Language Models for Evolutionary Program Synthesis: A Case Study on ARC-AGI",
        "url": "https://arxiv.org/abs/2507.14172",
        "http_status": 200,
    },
    "2512.22336": {
        "title": "Agent2World: Learning to Generate Symbolic World Models via Adaptive Multi-Agent Feedback",
        "url": "https://arxiv.org/abs/2512.22336",
        "http_status": 200,
    },
    "2604.08792": {
        "title": "Choose, Don't Label: Multiple-Choice Query Synthesis for Program Disambiguation",
        "url": "https://arxiv.org/abs/2604.08792",
        "http_status": 200,
    },
    "2411.17708": {
        "title": "Towards Efficient Neurally-Guided Program Induction for ARC-AGI",
        "url": "https://arxiv.org/abs/2411.17708",
        "http_status": 200,
    },
    "2310.19791": {
        "title": "LILO: Learning Interpretable Libraries by Compressing and Documenting Code",
        "url": "https://arxiv.org/abs/2310.19791",
        "http_status": 200,
    },
}
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(f"arXiv:{source}" for source in CITATIONS_VERIFIED)

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
    "ARC-AGI-3 goal acquisition executable world models program induction",
    "ARC AGI goal induction program synthesis refinement loop",
    "interactive agents goal-shift detection world model induction",
    "ARC Prize 2025 program synthesis refinement loop ARC-AGI",
    "executable world model goal acquisition ARC-AGI-3 Family-B",
]

DEFAULT_PRECONDITIONS_CHECKED = {
    "agents_md_read": True,
    "codex_md_read": True,
    "sweep_clusters_help_exit_0": True,
    "arxiv_api_reachable": True,
    "research_studying_filtered": True,
    "research_references_filtered": True,
    "prior_reinduction_spec_read": True,
    "prior_reinduction_artifact_read": True,
    "research_studying_updated": True,
    "sweep_clusters_used": True,
    "sweep_clusters_urls": SWEEP_CLUSTER_URLS,
    "sweep_semscholar_used": True,
    "sweep_semscholar_queries": S2_QUERIES,
    "sweep_semscholar_arxiv_ids": ["2507.14172", "2603.20334", "2603.13372", "2601.10904"],
    "sweep_semscholar_rate_limited_queries": [
        "ARC-AGI-3 goal acquisition executable world models program induction",
        "interactive agents goal-shift detection world model induction",
        "ARC Prize 2025 program synthesis refinement loop ARC-AGI",
    ],
    "arxiv_http_200_verified_ids": list(CITATIONS_VERIFIED),
    "websearch_webfetch_top_sources": [citation["url"] for citation in CITATIONS_VERIFIED.values()],
    "superseded_navigation_reingested": False,
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
        "method": "Family-B executable world-model re-induction",
        "source_ids": ["2605.05138", "2603.24621"],
        "takes_over_current_reinduction": (
            "Exp 4533 currently clears stale induction state after a level-up "
            "and asks the offline DSL path for a new level-conditioned GOAL "
            "predicate. This method takes over that post-level-up induction "
            "slot with a verifier-driven executable Python world model: induce "
            "dynamics and GOAL separately, verify predicted transitions against "
            "post-transition frames, refactor toward simpler state variables, "
            "then route search with the new predicate."
        ),
        "fails_when": (
            "the executable model is treated as proof without held-out "
            "transition checks, or when the induced simulator explains L1 "
            "dynamics but keeps the stale L1 completion predicate after the "
            "episode shifts to L2."
        ),
        "v420_candidate": (
            "flagged_for_v420: Family-B executable re-induction loop for each "
            "level-up, with separate GOAL and transition checks"
        ),
    },
    {
        "method": "refinement-loop program synthesis over candidate GOAL predicates",
        "source_ids": ["2601.10904", "2507.14172"],
        "takes_over_current_reinduction": (
            "Exp 4533 uses a single deterministic re-induction pass per "
            "level-up. This method turns that pass into a bounded refinement "
            "loop: generate several candidate GOAL/dynamics programs, execute "
            "them on post-transition observations, keep counterexample failures "
            "as feedback, and re-synthesize before the next frontier batch."
        ),
        "fails_when": (
            "the loop optimizes static ARC-AGI grid transforms rather than "
            "interactive ARC-AGI-3 state/action traces, or when hindsight "
            "fine-tuning is assumed available inside the autonomous sprint."
        ),
        "v420_candidate": (
            "flagged_for_v420: bounded ARC Prize/SOAR-style refinement loop "
            "around exp4533 candidate GOAL programs"
        ),
    },
    {
        "method": "adaptive behavior-test goal-shift detector",
        "source_ids": ["2512.22336", "2604.08792"],
        "takes_over_current_reinduction": (
            "Exp 4533 detects only the level counter increment. This method "
            "adds intra-episode goal-shift detection by synthesizing behavior "
            "tests that distinguish stale and new GOAL candidates; when tests "
            "disagree, the route abstains or re-induces instead of continuing "
            "with the old predicate."
        ),
        "fails_when": (
            "the test generator depends on web-search agents, human intent "
            "answers, or labels unavailable to the offline ARC agent; the "
            "replacement answerer must be executable evidence from frames."
        ),
        "v420_candidate": (
            "flagged_for_v420: adaptive behavior-test harness for detecting "
            "within-episode GOAL shifts after level-up"
        ),
    },
    {
        "method": "neural-guided DSL/library induction for reusable level predicates",
        "source_ids": ["2411.17708", "2310.19791"],
        "takes_over_current_reinduction": (
            "Exp 4533 re-induces from the current episode in isolation. This "
            "method keeps the re-induction trigger but changes the search "
            "space: retrieve documented predicate/world-model primitives from "
            "the solved ARC corpus, then neurally order compact DSL candidates "
            "for the new level before falling back to blind enumeration."
        ),
        "fails_when": (
            "library compression memorizes game-specific coordinates or L1 "
            "surface predicates; every reused primitive still needs "
            "representation-correct post-level-up validation."
        ),
        "v420_candidate": (
            "flagged_for_v420: LILO/neural-guided predicate library routed by "
            "the exp4533 level-up trigger"
        ),
    },
]

FLAGGED_FOR_NEXT_ROADMAP = (
    "flagged_for_v420: Family-B executable re-induction loop for each level-up, "
    "with separate GOAL-vs-dynamics candidates, adaptive behavior tests for "
    "goal-shift detection, and a bounded refinement loop around exp4533"
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
        "prior_reinduction_spec_read": "REQ-ARC-WMTE-4533 spec",
        "prior_reinduction_artifact_read": "Exp 4533 re-induction artifact",
        "research_studying_updated": "research-studying.md update",
        "sweep_clusters_used": "sweep_clusters.py",
        "sweep_semscholar_used": "sweep_semscholar.py",
    }
    for key, label in expected_true.items():
        _require(row.get(key) is True, f"preconditions_checked must record {label}")

    expected_false = {
        "superseded_navigation_reingested": "superseded navigation",
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
        _nonempty_list(row.get("sweep_clusters_urls")),
        "preconditions_checked must record cluster URLs",
    )
    _require(
        row.get("sweep_clusters_urls") == SWEEP_CLUSTER_URLS,
        "preconditions_checked must record the focused cluster 6/3 URLs",
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
        and {citation["url"] for citation in CITATIONS_VERIFIED.values()}.issubset(
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
        artifact["inference_substrate"] == INFERENCE_SUBSTRATE,
        "inference_substrate must declare aggregation from upstream artifacts",
    )
    _require(
        artifact["field_principles"] == FIELD_PRINCIPLES
        and set(artifact["field_principles"]) == REQUIRED_ARTIFACT_FIELDS,
        "field_principles must match REQ-REPORT-4541",
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
        for key in (
            "method",
            "takes_over_current_reinduction",
            "fails_when",
            "v420_candidate",
        ):
            _require(
                isinstance(method[key], str) and bool(method[key].strip()),
                f"methods_mapped field {key!r} must be a non-empty string",
            )
        mapping = method["takes_over_current_reinduction"]
        _require(
            "Exp 4533" in mapping or "exp4533" in mapping,
            "methods_mapped must map onto the current Exp 4533 re-induction mechanism",
        )
        _require(
            method["v420_candidate"].startswith("flagged_for_v420:"),
            "methods_mapped v420_candidate must flag a .420 input",
        )
    _require(
        used_method_sources == set(CITATIONS_VERIFIED),
        "methods_mapped must use every verified citation",
    )

    _require(
        artifact["flagged_for_next_roadmap"] == FLAGGED_FOR_NEXT_ROADMAP
        and str(artifact["flagged_for_next_roadmap"]).startswith("flagged_for_v420:"),
        "flagged_for_next_roadmap must match the verified .420 candidate",
    )
    _validate_preconditions(artifact["preconditions_checked"])


def _artifact_json(artifact: Mapping[str, Any]) -> str:
    return json.dumps(artifact, indent=2, sort_keys=True)


def render_research_note(artifact: Mapping[str, Any]) -> str:
    """Render markdown with the artifact block first for automated parsing."""

    validate_artifact(artifact)
    source_line = ", ".join(f"arXiv:{source}" for source in CITATIONS_VERIFIED)
    return f"""# ARC goal-acquisition SOTA ingestion .419 - 2026-06-21

```json
{_artifact_json(artifact)}
```

Reliable channel only: `scripts/sweep_clusters.py`, `scripts/sweep_semscholar.py`,
arXiv abs-page HTTP-200 checks, and low-concurrency WebSearch/WebFetch of the
top eight goal-acquisition and world-model sources. Preconditions passed before
any claim was promoted: `.venv/bin/python scripts/sweep_clusters.py --help`
exited zero and
`curl -sf -o /dev/null https://export.arxiv.org/api/query?search_query=all:test`
confirmed arXiv API reachability. `scripts/sweep_clusters.py 6 --max-results 8`
and `scripts/sweep_clusters.py 3 --max-results 8` emitted the goal/world-model
cluster URLs. `scripts/sweep_semscholar.py` returned arXiv:2507.14172,
arXiv:2603.20334, arXiv:2603.13372, and arXiv:2601.10904, with HTTP 429 on
three focused queries, so no S2-only claim was promoted. No `/deep-research`
call was made. No training, live LLM inference, leaderboard submission, or live
solve was launched. No ops/status/traceability files or
`scripts/research_conductor.py` were modified, and the navigation thread is superseded
rather than re-ingested.

Already-discovered corpus read through an ARC goal-acquisition / world-model
induction filter: `research-studying.md`, `research-references.md`,
`openspec/capabilities/arc-world-model-trust-energy/spec.md` at
`REQ-ARC-WMTE-4533`, and
`results/experiment_4533_per_level_goal_reinduction.json`. The current
mechanism this maps onto is exp4533: after a level-up it clears stale induction
state, re-runs post-transition induction, and biases depth-primary frontier
search with a new level-conditioned GOAL predicate. The .419 headline is
reaching deeper levels through per-level / intra-episode GOAL induction,
goal-shift detection, Family-B executable world-model induction, and
refinement-loop program synthesis.

Sources checked: {source_line}.

## Per-Method Mapping

- **Family-B executable world-model re-induction** (arXiv:2605.05138,
  arXiv:2603.24621): replace the single exp4533 offline-DSL predicate pass
  with a verifier-driven loop that induces GOAL and transition candidates
  separately, checks post-transition held-out transitions, refactors state, and
  plans only through accepted executable models. This is the strongest .420
  candidate because it is the closest direct fit to ARC-AGI-3 goal acquisition.
- **Refinement-loop program synthesis over candidate GOAL predicates**
  (arXiv:2601.10904, arXiv:2507.14172): make the post-level-up induction pass
  iterative. Failed candidates become execution counterexamples for a bounded
  re-synthesis loop before the next frontier batch. It fails if imported as a
  static ARC-AGI grid-transform solver without interactive action traces.
- **Adaptive behavior-test goal-shift detector** (arXiv:2512.22336,
  arXiv:2604.08792): synthesize behavior tests that distinguish stale and new
  GOAL candidates after a level-up. It fails if the test answerer is a human or
  web-search agent rather than executable frame evidence.
- **Neural-guided DSL/library induction for reusable level predicates**
  (arXiv:2411.17708, arXiv:2310.19791): retrieve documented primitives and
  neurally order compact DSL candidates after the exp4533 trigger. It fails
  when compressed libraries memorize coordinates or old L1 predicates instead
  of representation-correct post-level-up rules.

## bottom line for the .420 roadmap

{FLAGGED_FOR_NEXT_ROADMAP}

The practical next experiment should keep exp4533's level-up trigger and
depth-primary route, but replace the one-shot predicate induction body with a
small Family-B executable world-model loop. GOAL-vs-dynamics separation is the
first check. Adaptive behavior tests should detect a within-episode goal-shift
before search spends another batch under a stale predicate. Refinement-loop
program synthesis should be bounded and execution-grounded; no live LLM load,
training run, leaderboard submission, or new solve claim is implied by this
ingestion artifact.
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
        "per-level / intra-episode GOAL induction",
        "goal-shift detection",
        "Family-B executable world-model",
        "refinement-loop program synthesis",
        "exp4533",
        "GOAL-vs-dynamics",
        "navigation thread is superseded",
        "flagged_for_v420",
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
## 2026-06-21 Exp 4541 - .419 goal-acquisition SOTA ingestion - INGESTED

**Status:** INGESTED into `docs/research-notes/arc-goal-acquisition-sota-419.md`
and `results/experiment_4541_sota_ingestion_goal_acquisition.json`.

**Preconditions:** `scripts/sweep_clusters.py --help` succeeded; the arXiv API
reachability check succeeded; `scripts/sweep_clusters.py` clusters 6 and 3
emitted focused URLs; `scripts/sweep_semscholar.py` returned arXiv:2507.14172,
arXiv:2603.20334, arXiv:2603.13372, and arXiv:2601.10904 with HTTP 429 on
three focused queries; top sources were verified by arXiv abs-page HTTP 200 and
low-concurrency WebSearch/WebFetch. `/deep-research` was not invoked. The .418
navigation thread is superseded and was not re-ingested. No live solve,
training run, leaderboard submission, ops/status/traceability edit, or
`scripts/research_conductor.py` edit occurred.

**Methods marked ingested:** Family-B executable world-model re-induction
(arXiv:2605.05138, arXiv:2603.24621), refinement-loop program synthesis
(arXiv:2601.10904, arXiv:2507.14172), adaptive behavior-test goal-shift
detection (arXiv:2512.22336, arXiv:2604.08792), and neural-guided DSL/library
induction for reusable level predicates (arXiv:2411.17708, arXiv:2310.19791).

{FLAGGED_FOR_NEXT_ROADMAP}
{STUDYING_SECTION_END}"""


def update_research_studying(path: Path) -> None:
    """Insert or replace the Exp 4541 studying-queue section."""

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
    root = Path(os.environ.get("CARNOT_EXP4541_ROOT", "."))
    artifact = write_outputs(
        artifact_path=root / "results/experiment_4541_sota_ingestion_goal_acquisition.json",
        note_path=root / RESEARCH_NOTE_RELATIVE_PATH,
        studying_path=root / "research-studying.md",
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
