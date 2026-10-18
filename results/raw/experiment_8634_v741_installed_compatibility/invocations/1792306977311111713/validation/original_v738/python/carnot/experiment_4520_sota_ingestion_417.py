"""Exp 4520 action-efficiency SOTA ingestion for the `.418` hand-off.

Spec refs: REQ-REPORT-4520, SCENARIO-REPORT-4520.

This module records a literature-synthesis artifact. It does not train an ARC
model, run a live solve, or submit to the leaderboard. The output is
deterministic so the markdown note, result JSON, and studying-queue update can
be validated by tests and by the conductor reconciler.
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
        "preconditions_checked",
        "source_ids",
        "methods_mapped",
        "citations",
        "v418_flagged_candidates",
        "research_note_path",
        "random_seed",
        "field_principles",
    }
)
REQUIRED_METHOD_FIELDS = frozenset(
    {
        "method",
        "source_ids",
        "takes_over_current_stack",
        "fails_when",
        "v418_candidate",
    }
)
REQUIRED_CITATION_FIELDS = frozenset({"title", "url", "http_status"})
REQUIRED_PRECONDITION_FIELDS = frozenset(
    {
        "agents_md_read",
        "codex_md_read",
        "research_studying_filtered",
        "research_references_filtered",
        "research_studying_updated",
        "input_notes_read",
        "network_precondition_hf_models_exit_0",
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
DEFAULT_HONEST_VERDICT = "complete: action_efficiency_sota_417_mapped_for_v418"
DEFAULT_RANDOM_SEED = 4520
RESEARCH_NOTE_RELATIVE_PATH = "docs/research-notes/arc-action-efficiency-sota-417.md"
STUDYING_SECTION_START = "<!-- EXP4520-ACTION-EFFICIENCY-SOTA-START -->"
STUDYING_SECTION_END = "<!-- EXP4520-ACTION-EFFICIENCY-SOTA-END -->"

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "terminal prefix; e.g. complete: "
        "action_efficiency_sota_417_mapped_for_v418."
    ),
    "inference_substrate": (
        "aggregation_from_upstream_artifacts -- literature synthesis, no compute "
        "(100us floor)."
    ),
    "preconditions_checked": (
        "records network was verified; pre-empts fabricated-citation failure."
    ),
    "source_ids": "five to eight real arXiv IDs promoted by the reliable channel.",
    "methods_mapped": (
        "the strongest 3-5 methods with real arXiv IDs -- a claim without a "
        "verifiable citation is fabrication."
    ),
    "citations": (
        "real arXiv IDs / URLs for every method claim (the two-source / "
        "pre-claim checklist)."
    ),
    "v418_flagged_candidates": (
        "closes the discover->ingest->plan loop so SOTA flows into .418 "
        "experiments."
    ),
    "research_note_path": "repo-relative markdown path for deterministic parsing.",
    "random_seed": "bare integer seed for reproducible artifact generation.",
    "field_principles": "principle annotations for every top-level artifact field.",
}

CITATIONS = {
    "2008.09241": {
        "title": "Learning Affordance Landscapes for Interaction Exploration in 3D Environments",
        "url": "https://arxiv.org/abs/2008.09241",
        "http_status": 200,
    },
    "2501.06047": {
        "title": "Learning Affordances from Interactive Exploration using an Object-level Map",
        "url": "https://arxiv.org/abs/2501.06047",
        "http_status": 200,
    },
    "2602.00460": {
        "title": "Search Inspired Exploration in Reinforcement Learning",
        "url": "https://arxiv.org/abs/2602.00460",
        "http_status": 200,
    },
    "2602.03201": {
        "title": "SLOPE: Optimistic Potential Landscape Shaping for Model-based Reinforcement Learning",
        "url": "https://arxiv.org/abs/2602.03201",
        "http_status": 200,
    },
    "1511.05952": {
        "title": "Prioritized Experience Replay",
        "url": "https://arxiv.org/abs/1511.05952",
        "http_status": 200,
    },
    "1704.03732": {
        "title": "Deep Q-learning from Demonstrations",
        "url": "https://arxiv.org/abs/1704.03732",
        "http_status": 200,
    },
    "1901.10995": {
        "title": "Go-Explore: a New Approach for Hard-Exploration Problems",
        "url": "https://arxiv.org/abs/1901.10995",
        "http_status": 200,
    },
    "2602.05832": {
        "title": "UI-Mem: Self-Evolving Experience Memory for Online Reinforcement Learning in Mobile GUI Agents",
        "url": "https://arxiv.org/abs/2602.05832",
        "http_status": 200,
    },
}
DEFAULT_SOURCE_IDS = list(CITATIONS)
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(f"arXiv:{source}" for source in CITATIONS)

INPUT_NOTES_READ = [
    "docs/research-notes/arc-leaderboard-competitive-intel-2026-06-20.md",
    "docs/research-notes/arc-417-shaping-action-efficiency.md",
    "research-studying.md",
    "research-references.md",
    "docs/research-notes/arc-imitation-sota-415.md",
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
    "action effect model affordance exploration reinforcement learning persistent memory",
    "experience replay for search action efficiency exploration reinforcement learning",
    "persistent action memory action effect reinforcement learning exploration",
    "action effect prediction reinforcement learning exploration affordance",
    "clickability visual affordance reinforcement learning interactive exploration",
]

DEFAULT_PRECONDITIONS_CHECKED = {
    "agents_md_read": True,
    "codex_md_read": True,
    "research_studying_filtered": True,
    "research_references_filtered": True,
    "research_studying_updated": True,
    "input_notes_read": INPUT_NOTES_READ,
    "network_precondition_hf_models_exit_0": True,
    "sweep_clusters_used": True,
    "sweep_clusters_urls": SWEEP_CLUSTER_URLS,
    "sweep_semscholar_used": True,
    "sweep_semscholar_queries": S2_QUERIES,
    "sweep_semscholar_arxiv_ids": [
        "2509.10511",
        "2210.07553",
        "2601.08665",
        "2507.07451",
        "2511.03405",
        "2402.18487",
        "2207.07791",
    ],
    "sweep_semscholar_rate_limited_queries": [
        "experience replay for search action efficiency exploration reinforcement learning",
        "persistent action memory action effect reinforcement learning exploration",
        "action effect prediction reinforcement learning exploration affordance",
        "clickability visual affordance reinforcement learning interactive exploration",
    ],
    "arxiv_http_200_verified_ids": DEFAULT_SOURCE_IDS,
    "websearch_webfetch_top_sources": [citation["url"] for citation in CITATIONS.values()],
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
        "method": "affordance-landscape clickability pruning",
        "source_ids": ["2008.09241", "2501.06047"],
        "takes_over_current_stack": (
            "offline-search + lazy value head + frame-change predictor: replace "
            "blind candidate expansion with a frame-only click/action affordance "
            "mask before BFS spends actions"
        ),
        "fails_when": (
            "availability is mistaken for progress; a clickable cell can still "
            "be a loop unless the lazy value/energy term verifies movement "
            "toward level progress"
        ),
        "v418_candidate": (
            "flagged_for_v418: train the frame-change predictor as an "
            "affordance heatmap and prune predicted no-op action/click cells"
        ),
    },
    {
        "method": "search-inspired frontier control",
        "source_ids": ["2602.00460", "1901.10995"],
        "takes_over_current_stack": (
            "offline-search + lazy value head + frame-change predictor: choose "
            "frontier state-action pairs that are reachable, not exhausted, and "
            "promising under lazy value instead of repeatedly expanding flat BFS"
        ),
        "fails_when": (
            "frontier bookkeeping becomes a second solver with hand-tuned cell "
            "abstractions; ARC acceptance must stay actions-to-first-levelup at "
            "equal solve-rate"
        ),
        "v418_candidate": (
            "flagged_for_v418: add a SIERL/Go-Explore frontier queue over cached "
            "frame hashes and replayable action prefixes"
        ),
    },
    {
        "method": "prioritized replay with demonstration seeding",
        "source_ids": ["1511.05952", "1704.03732"],
        "takes_over_current_stack": (
            "offline-search + lazy value head + frame-change predictor: sample "
            "rare progress, human replay, and high-TD-error transitions before "
            "uniform self-play when training the predictor and lazy value head"
        ),
        "fails_when": (
            "public-game demonstrations dominate hidden-game behavior; priorities "
            "must decay unless held-out variants show equal or better progress"
        ),
        "v418_candidate": (
            "flagged_for_v418: seed predictor/value batches with PER/DQfD-style "
            "expert transitions, then anneal after self-play catches up"
        ),
    },
    {
        "method": "persistent hierarchical action memory",
        "source_ids": ["2602.05832"],
        "takes_over_current_stack": (
            "offline-search + lazy value head + frame-change predictor: persist "
            "action-effect templates, failure cautions, and successful prefixes "
            "across games as retrieval hints rather than relearning from scratch"
        ),
        "fails_when": (
            "retrieval is not gated by semantic/frame similarity; irrelevant "
            "memory can waste actions faster than blind search"
        ),
        "v418_candidate": (
            "flagged_for_v418: create a PersistentAEM-style store of "
            "frame-diff/action/reward templates with caution suppression"
        ),
    },
    {
        "method": "optimistic potential shaping for sparse progress",
        "source_ids": ["2602.03201"],
        "takes_over_current_stack": (
            "offline-search + lazy value head + frame-change predictor: use an "
            "optimistic potential term beside lazy value so rare level-progress "
            "signals rank survivors after no-op pruning"
        ),
        "fails_when": (
            "the potential is learned from frame marginals or dense proxy rewards "
            "without structural checks; it can over-rank visually novel dead ends"
        ),
        "v418_candidate": (
            "flagged_for_v418: add SLOPE-style upper-bound progress potential as "
            "a ranking-only feature after frame-change pruning"
        ),
    },
]

V418_FLAGGED_CANDIDATES = [
    (
        "flagged_for_v418: affordance-pruned frame-change/clickability model "
        "anchored by arXiv:2008.09241 and arXiv:2501.06047"
    ),
    (
        "flagged_for_v418: SIERL/Go-Explore frontier queue over replayable "
        "offline-search states anchored by arXiv:2602.00460 and arXiv:1901.10995"
    ),
    (
        "flagged_for_v418: PER/DQfD transition sampler for the frame-change "
        "predictor and lazy value head anchored by arXiv:1511.05952 and "
        "arXiv:1704.03732"
    ),
    (
        "flagged_for_v418: UI-Mem-style persistent cross-game action memory "
        "with caution suppression anchored by arXiv:2602.05832"
    ),
    (
        "flagged_for_v418: SLOPE-style optimistic potential ranking after "
        "no-op pruning anchored by arXiv:2602.03201"
    ),
]


def _fail(message: str) -> None:
    raise ValueError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _nonempty_list(value: object) -> bool:
    return isinstance(value, list) and bool(value)


def build_artifact(
    *,
    source_ids: Sequence[str] | None = None,
    methods_mapped: Sequence[Mapping[str, object]] | None = None,
    preconditions_checked: Mapping[str, object] | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    honest_verdict: str = DEFAULT_HONEST_VERDICT,
) -> dict[str, object]:
    """Build the deterministic artifact embedded in the markdown note."""

    chosen_source_ids = DEFAULT_SOURCE_IDS if source_ids is None else list(source_ids)
    chosen_methods = DEFAULT_METHODS_MAPPED if methods_mapped is None else methods_mapped
    chosen_preconditions = (
        DEFAULT_PRECONDITIONS_CHECKED
        if preconditions_checked is None
        else preconditions_checked
    )
    return {
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(chosen_preconditions),
        "source_ids": list(chosen_source_ids),
        "methods_mapped": [dict(method) for method in chosen_methods],
        "citations": {key: dict(value) for key, value in CITATIONS.items()},
        "v418_flagged_candidates": list(V418_FLAGGED_CANDIDATES),
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
        "research_studying_filtered": "research-studying.md filtered pass",
        "research_references_filtered": "research-references.md filtered pass",
        "research_studying_updated": "research-studying.md update",
        "network_precondition_hf_models_exit_0": "network precondition",
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
        "ops_docs_modified": "ops docs modification",
        "research_conductor_modified": "scripts/research_conductor.py",
    }
    for key, label in expected_false.items():
        _require(row.get(key) is False, f"preconditions_checked must record no {label}")

    _require(
        row.get("input_notes_read") == INPUT_NOTES_READ,
        "preconditions_checked must record required input notes",
    )
    _require(
        _nonempty_list(row.get("sweep_clusters_urls")),
        "preconditions_checked must record cluster URLs",
    )
    _require(
        _nonempty_list(row.get("sweep_semscholar_queries")),
        "preconditions_checked must record Semantic Scholar queries",
    )
    _require(
        _nonempty_list(row.get("sweep_semscholar_arxiv_ids")),
        "preconditions_checked must record Semantic Scholar arXiv ids",
    )
    _require(
        _nonempty_list(row.get("sweep_semscholar_rate_limited_queries")),
        "preconditions_checked must record Semantic Scholar HTTP 429 queries",
    )
    _require(
        _nonempty_list(row.get("arxiv_http_200_verified_ids"))
        and set(DEFAULT_SOURCE_IDS).issubset(set(row["arxiv_http_200_verified_ids"])),
        "preconditions_checked must include all verified arXiv ids",
    )
    _require(
        _nonempty_list(row.get("websearch_webfetch_top_sources"))
        and {citation["url"] for citation in CITATIONS.values()}.issubset(
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
        "field_principles must match REQ-REPORT-4520",
    )
    _require(
        isinstance(artifact["random_seed"], int)
        and not isinstance(artifact["random_seed"], bool),
        "random_seed must be an integer",
    )
    _require(
        artifact["research_note_path"] == RESEARCH_NOTE_RELATIVE_PATH,
        "research_note_path must be the repo-relative note path",
    )

    source_ids = artifact["source_ids"]
    _require(
        isinstance(source_ids, list) and 5 <= len(source_ids) <= 8,
        "source_ids must contain five to eight verified arXiv ids",
    )
    seen_sources: set[str] = set()
    for source in source_ids:
        _require(isinstance(source, str), "source_ids entries must be strings")
        _require(source in CITATIONS, f"source {source!r} is not a verified arXiv id")
        _require(source not in seen_sources, f"duplicate source in source_ids: {source}")
        seen_sources.add(source)

    citations = artifact["citations"]
    _require(isinstance(citations, Mapping), "citations must be a mapping")
    _require(set(citations) == set(source_ids), "citations must cover every source_id")
    for source, citation in citations.items():
        _require(source in CITATIONS, "citations must use verified source ids")
        _require(
            isinstance(citation, Mapping) and set(citation) == REQUIRED_CITATION_FIELDS,
            "each citation must include title, url, and http_status",
        )
        _require(citation == CITATIONS[source], "citations must match verified arXiv metadata")

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
            and set(method["source_ids"]).issubset(set(source_ids)),
            "methods_mapped source_ids must use verified citations",
        )
        used_method_sources.update(str(source) for source in method["source_ids"])
        for key in ("method", "takes_over_current_stack", "fails_when", "v418_candidate"):
            _require(
                isinstance(method[key], str) and bool(method[key].strip()),
                f"methods_mapped field {key!r} must be a non-empty string",
            )
        _require(
            "offline-search + lazy value head + frame-change predictor"
            in method["takes_over_current_stack"],
            "methods_mapped must map onto the current stack",
        )
        _require(
            method["v418_candidate"].startswith("flagged_for_v418:"),
            "methods_mapped v418_candidate must flag a .418 input",
        )
    _require(used_method_sources == set(source_ids), "methods_mapped must use every citation")

    candidates = artifact["v418_flagged_candidates"]
    _require(
        isinstance(candidates, list) and len(candidates) == len(V418_FLAGGED_CANDIDATES),
        "v418_flagged_candidates must list the planned .418 inputs",
    )
    _require(
        candidates == V418_FLAGGED_CANDIDATES
        and all(str(candidate).startswith("flagged_for_v418:") for candidate in candidates),
        "v418_flagged_candidates must match the verified roadmap candidates",
    )
    _validate_preconditions(artifact["preconditions_checked"])


def _artifact_json(artifact: Mapping[str, Any]) -> str:
    return json.dumps(artifact, indent=2, sort_keys=True)


def render_research_note(artifact: Mapping[str, Any]) -> str:
    """Render markdown with the artifact block first for automated parsing."""

    validate_artifact(artifact)
    source_line = ", ".join(f"arXiv:{source}" for source in DEFAULT_SOURCE_IDS)
    return f"""# ARC action-efficiency SOTA ingestion .417 - 2026-06-20

```json
{_artifact_json(artifact)}
```

Reliable channel only: `scripts/sweep_clusters.py`, `scripts/sweep_semscholar.py`,
arXiv abs-page HTTP-200 checks, and low-concurrency WebSearch/WebFetch of the
top eight action-efficient exploration sources. The network precondition
`curl -sf -o /dev/null https://huggingface.co/api/models` succeeded before any
claim was promoted. `scripts/sweep_clusters.py 5 --max-results 8` and
`scripts/sweep_clusters.py 6 --max-results 8` emitted the action-efficiency and
neural-guided-search cluster URLs. Semantic Scholar returned seven candidate
arXiv IDs across focused queries and reported HTTP 429 on the replay/memory
queries, so no S2-only claim was promoted. No `/deep-research` call was made.
No training, live LLM inference, leaderboard submission, or live solve was
launched. No ops/status/traceability files or `scripts/research_conductor.py`
were modified.

Sources checked: {source_line}.

## Per-Method Mapping

- **Affordance-landscape clickability pruning** (arXiv:2008.09241,
  arXiv:2501.06047): take over the front of the explorer by predicting which
  action/click cells are feasible and likely to change the frame, then prune
  predicted no-ops before BFS. This is the cleanest action-effect/clickability
  graft onto the offline-search + lazy value head + frame-change predictor
  stack. It fails when availability is treated as progress; every survivor
  still needs value/energy or level-progress checks.
- **Search-inspired frontier control** (arXiv:2602.00460, arXiv:1901.10995):
  replace flat breadth expansion with a frontier queue over reachable cached
  frame hashes and replayable prefixes. It fails when the frontier abstraction
  becomes benchmark-specific or loses the action-count acceptance gate.
- **Prioritized replay with demonstration seeding** (arXiv:1511.05952,
  arXiv:1704.03732): train the frame-change predictor and lazy value head from
  high-progress, high-error, and human/demo transitions before uniform self-play.
  It fails when public-game demonstrations remain over-prioritized after
  held-out variants stop improving.
- **Persistent hierarchical action memory** (arXiv:2602.05832): persist
  successful prefixes, frame-diff/action/reward templates, and failure cautions
  across games. It fails when retrieval is not gated by similarity and stale
  memory wastes actions.
- **Optimistic potential shaping** (arXiv:2602.03201): add a ranking-only
  potential term after frame-change pruning so rare sparse-progress signals are
  not flattened. It fails when a proxy potential over-ranks visually novel dead
  ends without structural checks.

## bottom line for the .418 roadmap

The strongest `.418` input is the combined **affordance-pruned frame-change
predictor plus frontier control**: prune predicted no-ops first, then let a
SIERL/Go-Explore-style frontier queue choose among reachable survivors under
the lazy value head. PER/DQfD replay seeding is the training substrate for that
predictor/value pair. UI-Mem-style persistent action memory is second-line:
use it only with similarity-gated retrieval and caution suppression. SLOPE-style
potential shaping is ranking-only until it proves actions-to-first-levelup
improvement at equal solve-rate.

{chr(10).join(V418_FLAGGED_CANDIDATES)}
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
        "HTTP 429",
        "action-effect/clickability",
        "offline-search + lazy value head + frame-change predictor",
        "bottom line for the .418 roadmap",
        "flagged_for_v418",
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
## 2026-06-20 Exp 4520 - .417 action-efficiency SOTA ingestion - INGESTED

**Status:** INGESTED into `docs/research-notes/arc-action-efficiency-sota-417.md`
and `results/experiment_4520_sota_ingestion_417.json`.

**Preconditions:** Hugging Face API reachability succeeded; `scripts/sweep_clusters.py`
clusters 5 and 6 emitted focused URLs; `scripts/sweep_semscholar.py` produced
seven arXiv candidates and HTTP 429 on replay/memory queries; top sources were
verified by arXiv abs-page HTTP 200 and low-concurrency WebSearch/WebFetch.
`/deep-research` was not invoked. No live solve, training run, leaderboard
submission, ops/status/traceability edit, or `scripts/research_conductor.py`
edit occurred.

**Methods marked ingested:** affordance-landscape clickability pruning
(arXiv:2008.09241, arXiv:2501.06047), SIERL/Go-Explore frontier control
(arXiv:2602.00460, arXiv:1901.10995), PER/DQfD replay seeding
(arXiv:1511.05952, arXiv:1704.03732), UI-Mem-style persistent action memory
(arXiv:2602.05832), and SLOPE-style optimistic potential shaping
(arXiv:2602.03201).

flagged_for_v418: affordance-pruned frame-change/clickability plus
SIERL/Go-Explore frontier control over replayable offline-search states; use
PER/DQfD replay seeding for predictor/value training, add UI-Mem-style
persistent action memory behind similarity-gated retrieval, and keep SLOPE as
ranking-only until it reduces actions-to-first-levelup at equal solve-rate.
{STUDYING_SECTION_END}"""


def update_research_studying(path: Path) -> None:
    """Insert or replace the Exp 4520 studying-queue section."""

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
    root = Path(os.environ.get("CARNOT_EXP4520_ROOT", "."))
    artifact = write_outputs(
        artifact_path=root / "results/experiment_4520_sota_ingestion_417.json",
        note_path=root / RESEARCH_NOTE_RELATIVE_PATH,
        studying_path=root / "research-studying.md",
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
