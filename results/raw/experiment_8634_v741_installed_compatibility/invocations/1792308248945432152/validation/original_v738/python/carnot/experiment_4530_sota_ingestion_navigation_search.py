"""Exp 4530 navigation-cost search SOTA ingestion for the `.419` hand-off.

Spec refs: REQ-REPORT-4530, SCENARIO-REPORT-4530.

This module records a literature-synthesis artifact. It does not run the ARC
agent, train a model, or submit to the leaderboard. The output is deterministic
so the markdown note, result JSON, and studying-queue update can be tested.
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
        "takes_over_current_explorer",
        "fails_when",
        "v419_candidate",
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
        "prior_action_efficiency_note_read",
        "forward_walk_artifact_read",
        "nav_metric_artifact_read",
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
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
DEFAULT_HONEST_VERDICT = "complete: sota_ingestion_navigation_search_mapped"
DEFAULT_RANDOM_SEED = 4530
RESEARCH_NOTE_RELATIVE_PATH = "docs/research-notes/arc-navigation-search-sota-418.md"
STUDYING_SECTION_START = "<!-- EXP4530-NAVIGATION-SEARCH-SOTA-START -->"
STUDYING_SECTION_END = "<!-- EXP4530-NAVIGATION-SEARCH-SOTA-END -->"

FIELD_PRINCIPLES = {
    "honest_verdict": "terminal prefix; complete: sota_ingestion_navigation_search_mapped.",
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
        "the strongest method flagged as a .419 candidate -- closes the "
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
    "1906.05253": {
        "title": "Search on the Replay Buffer: Bridging Planning and Reinforcement Learning",
        "url": "https://arxiv.org/abs/1906.05253",
        "http_status": 200,
    },
    "2602.00460": {
        "title": "Search Inspired Exploration in Reinforcement Learning",
        "url": "https://arxiv.org/abs/2602.00460",
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
    "2605.25931": {
        "title": "Explore Before You Solve: The Speed--Depth Trade-off in Epistemic Agents for ARC-AGI-3",
        "url": "https://arxiv.org/abs/2605.25931",
        "http_status": 200,
    },
    "2304.05506": {
        "title": "Frontier Semantic Exploration for Visual Target Navigation",
        "url": "https://arxiv.org/abs/2304.05506",
        "http_status": 200,
    },
    "2603.05377": {
        "title": "OpenFrontier: General Navigation with Visual-Language Grounded Frontiers",
        "url": "https://arxiv.org/abs/2603.05377",
        "http_status": 200,
    },
    "1810.02274": {
        "title": "Episodic Curiosity through Reachability",
        "url": "https://arxiv.org/abs/1810.02274",
        "http_status": 200,
    },
}
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(f"arXiv:{source}" for source in CITATIONS_VERIFIED)

INPUT_NOTES_READ = [
    "research-studying.md",
    "research-references.md",
    "docs/research-notes/arc-action-efficiency-sota-417.md",
    "results/experiment_4523_forward_walk_navigation.json",
    "results/experiment_4527_nav_metric_harness.json",
]

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
    "RESET-free tree search interactive agents navigation cost frontier",
    "backtracking efficient exploration reinforcement learning frontier navigation cost",
    "replay buffer graph search shortest path reinforcement learning interactive environments",
    "go-explore archive return explore hard exploration no reset",
    "amortized search agents cannot teleport physically navigate",
]

DEFAULT_PRECONDITIONS_CHECKED = {
    "agents_md_read": True,
    "codex_md_read": True,
    "sweep_clusters_help_exit_0": True,
    "arxiv_api_reachable": True,
    "research_studying_filtered": True,
    "research_references_filtered": True,
    "prior_action_efficiency_note_read": True,
    "forward_walk_artifact_read": True,
    "nav_metric_artifact_read": True,
    "research_studying_updated": True,
    "sweep_clusters_used": True,
    "sweep_clusters_urls": SWEEP_CLUSTER_URLS,
    "sweep_semscholar_used": True,
    "sweep_semscholar_queries": S2_QUERIES,
    "sweep_semscholar_arxiv_ids": [],
    "sweep_semscholar_rate_limited_queries": S2_QUERIES[1:],
    "arxiv_http_200_verified_ids": list(CITATIONS_VERIFIED),
    "websearch_webfetch_top_sources": [citation["url"] for citation in CITATIONS_VERIFIED.values()],
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
        "method": "replay-buffer graph search for physical return paths",
        "source_ids": ["1906.05253"],
        "takes_over_current_explorer": (
            ".418 StepwiseExplorer forward-walk navigation fix: promote the "
            "visited frame/action-prefix log into a graph, use exact "
            "_shortest_path reachability as the edge cost, and choose frontier "
            "nodes by navigation cost already payable from the current node"
        ),
        "fails_when": (
            "the distance metric is learned or stale rather than exact; any "
            "unverified edge can turn a RESET-free improvement into a hidden "
            "teleport assumption"
        ),
        "v419_candidate": (
            "flagged_for_v419: SoRB-style replay-buffer graph over "
            "StepwiseExplorer nodes with exact _shortest_path costs"
        ),
    },
    {
        "method": "search-inspired reachable-frontier subgoal control",
        "source_ids": ["2602.00460", "1810.02274"],
        "takes_over_current_explorer": (
            ".418 StepwiseExplorer forward-walk navigation fix: use "
            "cost-to-come, cost-to-go, and reachability novelty as equal-depth "
            "frontier tie-breaks after depth remains primary"
        ),
        "fails_when": (
            "episode-start subgoal selection is copied literally; ARC cannot "
            "spend extra actions returning to a frontier unless the path is "
            "already forward-walkable or replay-accounted"
        ),
        "v419_candidate": (
            "flagged_for_v419: SIERL-style frontier score with a reachability "
            "novelty guard, never overriding depth"
        ),
    },
    {
        "method": "Go-Explore archive discipline without state restore",
        "source_ids": ["1901.10995", "2004.12919"],
        "takes_over_current_explorer": (
            ".418 StepwiseExplorer forward-walk navigation fix: keep the "
            "archive of promising states but replace simulator state restore "
            "with policy-based or exact replay returns that charge every action"
        ),
        "fails_when": (
            "the method relies on emulator state restore, uncharged RESET, or "
            "post-hoc robustification training; the live ARC agent must "
            "physically navigate to the frontier it probes"
        ),
        "v419_candidate": (
            "flagged_for_v419: Go-Explore archive rows with charged return "
            "prefixes and RESET fallback diagnostics"
        ),
    },
    {
        "method": "embodied frontier navigation scoring",
        "source_ids": ["2304.05506", "2603.05377"],
        "takes_over_current_explorer": (
            ".418 StepwiseExplorer forward-walk navigation fix: score "
            "frontier nodes as navigation targets with information gain and "
            "reachable-path cost, not just local action priority"
        ),
        "fails_when": (
            "visual-language semantics or dense maps are treated as available "
            "inside ARC; only the frontier-cost abstraction transfers cleanly"
        ),
        "v419_candidate": (
            "flagged_for_v419: embodied-navigation frontier score as a "
            "diagnostic secondary term behind exact ARC reachability"
        ),
    },
    {
        "method": "ARC speed-depth budget controller",
        "source_ids": ["2605.25931"],
        "takes_over_current_explorer": (
            ".418 StepwiseExplorer forward-walk navigation fix: measure "
            "whether batching and navigation-cost tie-breaks stay on the "
            "action-efficiency frontier instead of only increasing search depth"
        ),
        "fails_when": (
            "public-game shortcuts or null-coordinate exploits drive the "
            "budget policy; hidden-game action efficiency must be measured by "
            "the canonical gate, not inferred from public quirks"
        ),
        "v419_candidate": (
            "flagged_for_v419: AERA-style speed-depth ledger for every "
            "frontier batch and navigation-cost treatment"
        ),
    },
]

FLAGGED_FOR_NEXT_ROADMAP = (
    "flagged_for_v419: SoRB-style replay-buffer graph over StepwiseExplorer "
    "frontier nodes, with exact _shortest_path navigation costs, charged "
    "return prefixes, RESET fallback diagnostics, and the existing CORE "
    "median-action gate as the acceptance metric"
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
        "prior_action_efficiency_note_read": "prior action-efficiency note",
        "forward_walk_artifact_read": "forward-walk navigation artifact",
        "nav_metric_artifact_read": "nav metric artifact",
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
        _nonempty_list(row.get("sweep_clusters_urls")),
        "preconditions_checked must record cluster URLs",
    )
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
        "field_principles must match REQ-REPORT-4530",
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
            "takes_over_current_explorer",
            "fails_when",
            "v419_candidate",
        ):
            _require(
                isinstance(method[key], str) and bool(method[key].strip()),
                f"methods_mapped field {key!r} must be a non-empty string",
            )
        mapping = method["takes_over_current_explorer"]
        _require(
            ".418 StepwiseExplorer forward-walk navigation fix" in mapping,
            "methods_mapped must map onto the current .418 explorer",
        )
        _require(
            method["v419_candidate"].startswith("flagged_for_v419:"),
            "methods_mapped v419_candidate must flag a .419 input",
        )
    _require(
        used_method_sources == set(CITATIONS_VERIFIED),
        "methods_mapped must use every verified citation",
    )

    _require(
        artifact["flagged_for_next_roadmap"] == FLAGGED_FOR_NEXT_ROADMAP
        and str(artifact["flagged_for_next_roadmap"]).startswith("flagged_for_v419:"),
        "flagged_for_next_roadmap must match the verified .419 candidate",
    )
    _validate_preconditions(artifact["preconditions_checked"])


def _artifact_json(artifact: Mapping[str, Any]) -> str:
    return json.dumps(artifact, indent=2, sort_keys=True)


def render_research_note(artifact: Mapping[str, Any]) -> str:
    """Render markdown with the artifact block first for automated parsing."""

    validate_artifact(artifact)
    source_line = ", ".join(f"arXiv:{source}" for source in CITATIONS_VERIFIED)
    return f"""# ARC navigation-search SOTA ingestion .418 - 2026-06-21

```json
{_artifact_json(artifact)}
```

Reliable channel only: `scripts/sweep_clusters.py`, `scripts/sweep_semscholar.py`,
arXiv abs-page HTTP-200 checks, and low-concurrency WebSearch/WebFetch of the
top eight navigation/search sources. Preconditions passed before any claim was
promoted: `.venv/bin/python scripts/sweep_clusters.py --help` exited zero and
`curl -sf -o /dev/null https://export.arxiv.org/api/query?search_query=all:test`
confirmed arXiv API reachability. `scripts/sweep_clusters.py 5 --max-results 8`
and `scripts/sweep_clusters.py 6 --max-results 8` emitted the ARC
action-efficiency and neural-guided-search cluster URLs. Semantic Scholar returned zero unique arXiv IDs
on the focused navigation pass and HTTP 429 on four of five queries, so no
S2-only claim was promoted. No `/deep-research`
call was made. No training, live LLM inference, leaderboard submission, or live
solve was launched. No ops/status/traceability files or
`scripts/research_conductor.py` were modified.

Already-discovered corpus read through an ARC navigation/search filter:
`research-studying.md`, `research-references.md`,
`docs/research-notes/arc-action-efficiency-sota-417.md`,
`results/experiment_4523_forward_walk_navigation.json`, and
`results/experiment_4527_nav_metric_harness.json`. The .418 state this maps
onto is the `StepwiseExplorer` forward-walk navigation fix: depth stays primary,
`_shortest_path` exact reachability can break equal-depth ties, frontier batches
amortize navigation already paid to a node, and RESET replay remains a fallback
diagnostic rather than a free teleport.

Sources checked: {source_line}.

## Per-Method Mapping

- **Replay-buffer graph search for physical return paths** (arXiv:1906.05253):
  take over the replay/navigation substrate by turning visited frame hashes and
  action prefixes into a graph. The .419 version should use exact `_shortest_path`
  costs instead of a learned distance metric, because the ARC agent cannot
  teleport to a frontier node. This is the strongest backtrack-efficient,
  RESET-free tree-search graft.
- **Search-inspired reachable-frontier subgoal control** (arXiv:2602.00460,
  arXiv:1810.02274): take over equal-depth frontier ordering with cost-to-come,
  cost-to-go, and reachability novelty. It fails if copied as an episode-start
  subgoal policy that spends uncharged return actions.
- **Go-Explore archive discipline without state restore** (arXiv:1901.10995,
  arXiv:2004.12919): preserve the archive-return-explore loop but charge every
  return path through policy-based or exact replay returns. It fails when state
  restore, RESET, or robustification training hides the physical navigation cost.
- **Embodied frontier navigation scoring** (arXiv:2304.05506,
  arXiv:2603.05377): borrow the frontier navigation cost framing from visual
  navigation, but keep only the reachable-frontier abstraction. Language priors
  and dense maps are not available inside ARC.
- **ARC speed-depth budget controller** (arXiv:2605.25931): keep the .418
  frontier-batch and nav-cost sweep honest by asking whether extra search depth
  stays on the action-efficiency frontier. It fails if public-game shortcuts are
  mistaken for hidden-game efficiency.

## bottom line for the .419 roadmap

{FLAGGED_FOR_NEXT_ROADMAP}

The method should take over only the navigation layer of the current explorer:
build the replay-buffer graph from real `StepwiseExplorer` nodes, score frontier
targets by exact forward-walk distance first, charge any replay suffix, and keep
the CORE median-action gate from `experiment_4523_forward_walk_navigation.json`.
SIERL and Go-Explore remain supporting controls for frontier priority and archive
discipline. Embodied frontier navigation and AERA are diagnostics for path cost
and speed-depth accounting, not permission to add a new planner that bypasses the
existing submission gate.
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
        "Semantic Scholar returned zero unique arXiv IDs",
        "backtrack-efficient",
        "RESET-free",
        "cannot teleport",
        "StepwiseExplorer",
        "_shortest_path",
        "frontier navigation cost",
        "flagged_for_v419",
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
## 2026-06-21 Exp 4530 - .418 navigation-search SOTA ingestion - INGESTED

**Status:** INGESTED into `docs/research-notes/arc-navigation-search-sota-418.md`
and `results/experiment_4530_sota_ingestion_navigation_search.json`.

**Preconditions:** `scripts/sweep_clusters.py --help` succeeded; the arXiv API
reachability check succeeded; `scripts/sweep_clusters.py` clusters 5 and 6
emitted focused URLs; `scripts/sweep_semscholar.py` returned zero unique arXiv
IDs and HTTP 429 on four focused navigation/search queries; top sources were
verified by arXiv abs-page HTTP 200 and low-concurrency WebSearch/WebFetch.
`/deep-research` was not invoked. No live solve, training run, leaderboard
submission, ops/status/traceability edit, or `scripts/research_conductor.py`
edit occurred.

**Methods marked ingested:** SoRB replay-buffer graph search (arXiv:1906.05253),
SIERL reachable-frontier subgoal control with reachability novelty
(arXiv:2602.00460, arXiv:1810.02274), Go-Explore / First-return-then-explore
archive discipline (arXiv:1901.10995, arXiv:2004.12919), embodied frontier
navigation scoring (arXiv:2304.05506, arXiv:2603.05377), and AERA speed-depth
budget control for ARC-AGI-3 (arXiv:2605.25931).

{FLAGGED_FOR_NEXT_ROADMAP}
{STUDYING_SECTION_END}"""


def update_research_studying(path: Path) -> None:
    """Insert or replace the Exp 4530 studying-queue section."""

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
    root = Path(os.environ.get("CARNOT_EXP4530_ROOT", "."))
    artifact = write_outputs(
        artifact_path=root / "results/experiment_4530_sota_ingestion_navigation_search.json",
        note_path=root / RESEARCH_NOTE_RELATIVE_PATH,
        studying_path=root / "research-studying.md",
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
