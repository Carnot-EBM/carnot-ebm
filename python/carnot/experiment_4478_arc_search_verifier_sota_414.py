"""Exp 4478 ARC search/verifier SOTA ingestion for the `.415` hand-off.

Spec refs: REQ-REPORT-4478, SCENARIO-REPORT-4478.

This module writes a planning artifact, not a solve artifact. The important
distinction is provenance: the note aggregates already-read repo research files,
the focused sweep helper, and arXiv/WebFetch checks, so it must declare
``aggregation_from_upstream_artifacts`` and must not imply that Carnot solved a
new ARC level.
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
        "offline_reproduced",
        "reproduced_levels",
        "preconditions_checked",
        "source_ids",
        "methods",
        "gap_mapping",
        "strongest_for_v415",
        "research_note_path",
        "random_seed",
        "field_principles",
    }
)
REQUIRED_METHOD_FIELDS = frozenset(
    {"name", "arxiv_id", "mapped_gap", "stack_mapping", "pitfall"}
)
REQUIRED_PRECONDITION_FIELDS = frozenset(
    {
        "agents_md_read",
        "codex_md_read",
        "research_studying_filtered",
        "research_references_filtered",
        "sweep_clusters_help_succeeded",
        "sweep_clusters_urls",
        "arxiv_http_200_verified_ids",
        "websearch_webfetch_top_sources",
        "deep_research_invoked",
        "live_llm_inference",
        "training_launched",
        "leaderboard_submission",
        "live_solve_claim",
        "ops_docs_modified",
    }
)
GAP_KEYS = frozenset(
    {
        "GAP-ARCH-FEATURES",
        "GAP-ARCH-GOAL",
        "GAP-ARCH-NO-HIERARCHICAL-SEARCH",
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
DEFAULT_HONEST_VERDICT = "complete: arc_search_verifier_sota_414_mapped_for_v415"
DEFAULT_RANDOM_SEED = 4478
RESEARCH_NOTE_RELATIVE_PATH = "docs/research-notes/arc-search-verifier-sota-414.md"

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "MUST start with a terminal prefix "
        "complete:/complete_/success:/success_/passed:/passed_/shipped:/shipped_ "
        "so the reconciler classifies it as terminal "
        "(Verdict Terminal-Prefix Discipline)."
    ),
    "inference_substrate": (
        "explicit declaration "
        "(live_llm_inference | verifier_ensemble_against_cached_candidates | "
        "aggregation_from_upstream_artifacts) so adversarial_verify applies the "
        "right floor."
    ),
    "offline_reproduced": (
        "a solve not reproducible offline is wasted effort -- only reproduced "
        "levels count (ARC Solve Reproducibility)."
    ),
    "reproduced_levels": (
        "headline metric reproducible_total_levels grows monotonically; report "
        "the count banked, real-env-confirmed."
    ),
    "preconditions_checked": (
        "records WHICH resources were verified before launching; pre-empts the "
        "silent-missing-resource fabrication mode."
    ),
}

VERIFIED_SOURCE_URLS = {
    "2512.24156": "https://arxiv.org/abs/2512.24156",
    "2603.24621": "https://arxiv.org/abs/2603.24621",
    "2605.05138": "https://arxiv.org/abs/2605.05138",
    "2606.12316": "https://arxiv.org/abs/2606.12316",
    "2512.22336": "https://arxiv.org/abs/2512.22336",
    "2605.25931": "https://arxiv.org/abs/2605.25931",
    "2604.08792": "https://arxiv.org/abs/2604.08792",
    "2402.08147": "https://arxiv.org/abs/2402.08147",
}
DEFAULT_SOURCE_IDS = list(VERIFIED_SOURCE_URLS)
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(f"arXiv:{source}" for source in VERIFIED_SOURCE_URLS)

DEFAULT_STRONGEST_FOR_V415 = (
    "flagged_for_v415: graph-state/delta-feature verifier plus hierarchical "
    "verified search, anchored by arXiv:2512.24156, arXiv:2606.12316, and "
    "arXiv:2402.08147"
)

DEFAULT_PRECONDITIONS_CHECKED = {
    "agents_md_read": True,
    "codex_md_read": True,
    "research_studying_filtered": True,
    "research_references_filtered": True,
    "sweep_clusters_help_succeeded": True,
    "sweep_clusters_urls": [
        (
            "http://export.arxiv.org/api/query?search_query="
            '(abs:"verifier+ensemble"+OR+abs:"verifier+ensembles"+OR+'
            'abs:"null+space"+OR+abs:"specification+gaming"+OR+'
            'abs:"process+reward+model"+OR+abs:"deliberative+alignment"+OR+'
            'abs:"reward+hacking")&start=0&max_results=8'
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
    ],
    "arxiv_http_200_verified_ids": DEFAULT_SOURCE_IDS,
    "websearch_webfetch_top_sources": list(VERIFIED_SOURCE_URLS.values()),
    "deep_research_invoked": False,
    "live_llm_inference": False,
    "training_launched": False,
    "leaderboard_submission": False,
    "live_solve_claim": False,
    "ops_docs_modified": False,
}

GAP_MAPPING = {
    "GAP-ARCH-FEATURES": {
        "principle": (
            "Relational/delta verifier features should describe object slots, "
            "frame-change facts, transition deltas, and state-graph context "
            "before a candidate is scored."
        ),
        "source_ids": ["2606.12316", "2512.24156", "2605.25931"],
        "next_experiment": (
            "Replace the weak flat hand-feature vector with object-slot, "
            "transition-delta, frame-change, and frontier-state features."
        ),
    },
    "GAP-ARCH-GOAL": {
        "principle": (
            "Goal-vs-dynamics induction must separate what counts as winning "
            "from the transition model that predicts how actions change state."
        ),
        "source_ids": ["2603.24621", "2605.05138", "2512.22336", "2604.08792"],
        "next_experiment": (
            "Induce candidate goal predicates separately from executable "
            "dynamics, then use verifier-rejected trajectories and "
            "disambiguating queries to decide or abstain."
        ),
    },
    "GAP-ARCH-NO-HIERARCHICAL-SEARCH": {
        "principle": (
            "Hierarchical/MCTS search should expand a verified state-action "
            "graph, backpropagate verifier feedback, and spend actions on "
            "frontier states rather than flat repeated BFS."
        ),
        "source_ids": ["2512.24156", "2605.05138", "2605.25931", "2402.08147"],
        "next_experiment": (
            "Add a frontier graph plus MCTS-style verifier-guided expansion "
            "over partial plans and candidate world-model edits."
        ),
    },
}

DEFAULT_METHODS = [
    {
        "name": "Graph-Based Exploration for ARC-AGI-3 Interactive Reasoning Tasks",
        "arxiv_id": "2512.24156",
        "mapped_gap": "GAP-ARCH-NO-HIERARCHICAL-SEARCH",
        "stack_mapping": (
            "Use explicit state-action graphs, shortest paths to untested "
            "state-action pairs, and salience-prioritized actions as the search "
            "baseline that a verifier-routed planner must beat."
        ),
        "pitfall": (
            "It is training-free exploration, not a learned verifier; Carnot "
            "still needs reproducible env traces before any solve is banked."
        ),
    },
    {
        "name": "ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence",
        "arxiv_id": "2603.24621",
        "mapped_gap": "GAP-ARCH-GOAL",
        "stack_mapping": (
            "Treat goal discovery, dynamics modeling, and efficient planning as "
            "separate measured axes rather than one opaque route score."
        ),
        "pitfall": (
            "The benchmark definition is not a method; it anchors metrics and "
            "prevents goal/dynamics claims from being conflated."
        ),
    },
    {
        "name": "Executable World Models for ARC-AGI-3",
        "arxiv_id": "2605.05138",
        "mapped_gap": "GAP-ARCH-GOAL",
        "stack_mapping": (
            "Maintain an executable model, verify it against observations, "
            "refactor toward simpler dynamics, and plan through it only after "
            "the verifier accepts the predicted transition behavior."
        ),
        "pitfall": (
            "Published playthroughs are not Carnot evidence; fresh local "
            "reproduction and leakage controls remain the only bankable signal."
        ),
    },
    {
        "name": "Slots, Transitions, Loops: Learning Composable World Models for ARC",
        "arxiv_id": "2606.12316",
        "mapped_gap": "GAP-ARCH-FEATURES",
        "stack_mapping": (
            "Port object slots, demonstration-conditioned summaries, looped "
            "transitions, and correction signals into the verifier feature bank."
        ),
        "pitfall": (
            "ARC-1/2 grid transitions omit interactive action costs and hidden "
            "state, so the features must be checked against live-game deltas."
        ),
    },
    {
        "name": "Agent2World adaptive symbolic world-model feedback",
        "arxiv_id": "2512.22336",
        "mapped_gap": "GAP-ARCH-GOAL",
        "stack_mapping": (
            "Use adaptive unit tests and simulation-based validation to expose "
            "behavior-level world-model errors before the planner trusts a rule."
        ),
        "pitfall": (
            "The paper includes a web-searching agent stage; this artifact only "
            "takes the behavior-aware testing pattern, not the live research loop."
        ),
    },
    {
        "name": "AERA speed-depth explore/verify/plan framework",
        "arxiv_id": "2605.25931",
        "mapped_gap": "GAP-ARCH-NO-HIERARCHICAL-SEARCH",
        "stack_mapping": (
            "Budget exploration for information gain, then verify and plan; use "
            "its benchmark critique as a guard against public-set shortcuts."
        ),
        "pitfall": (
            "The public-set vulnerability means public scores cannot be promoted "
            "without private-style or hidden-state robustness checks."
        ),
    },
    {
        "name": "Choose, Don't Label program-disambiguation queries",
        "arxiv_id": "2604.08792",
        "mapped_gap": "GAP-ARCH-GOAL",
        "stack_mapping": (
            "When candidate dynamics agree on demos but imply different goals, "
            "synthesize a discriminating behavior query and require replayable "
            "evidence to choose or abstain."
        ),
        "pitfall": (
            "The original uses humans for intent answers; Carnot must replace "
            "that answerer with executable evidence, or mark underdetermined."
        ),
    },
    {
        "name": "VerMCTS verifier-guided tree search",
        "arxiv_id": "2402.08147",
        "mapped_gap": "GAP-ARCH-NO-HIERARCHICAL-SEARCH",
        "stack_mapping": (
            "Adapt verifier-scored partial-program MCTS to partial world-model "
            "edits and plan prefixes, using verifier failures to avoid doomed "
            "branches early."
        ),
        "pitfall": (
            "Dafny/Coq verified code is not ARC control; the transferable part "
            "is verifier-in-the-loop tree search, not the proof benchmark."
        ),
    },
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
    methods: Sequence[Mapping[str, str]] | None = None,
    preconditions_checked: Mapping[str, object] | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    honest_verdict: str = DEFAULT_HONEST_VERDICT,
) -> dict[str, object]:
    """Build the deterministic artifact that the note embeds.

    The function is deliberately pure so tests can compare the note JSON and the
    standalone result JSON byte-for-byte after parsing.
    """

    chosen_source_ids = DEFAULT_SOURCE_IDS if source_ids is None else list(source_ids)
    chosen_methods = DEFAULT_METHODS if methods is None else methods
    chosen_preconditions = (
        DEFAULT_PRECONDITIONS_CHECKED if preconditions_checked is None else preconditions_checked
    )
    return {
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "preconditions_checked": dict(chosen_preconditions),
        "source_ids": list(chosen_source_ids),
        "methods": [dict(method) for method in chosen_methods],
        "gap_mapping": {
            gap_id: dict(details) for gap_id, details in GAP_MAPPING.items()
        },
        "strongest_for_v415": DEFAULT_STRONGEST_FOR_V415,
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
        "sweep_clusters_help_succeeded": "sweep_clusters help",
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
    }
    for key, label in expected_false.items():
        _require(row.get(key) is False, f"preconditions_checked must record no {label}")

    _require(_nonempty_list(row.get("sweep_clusters_urls")), "preconditions_checked must record cluster URLs")
    _require(
        _nonempty_list(row.get("arxiv_http_200_verified_ids"))
        and set(DEFAULT_SOURCE_IDS).issubset(set(row["arxiv_http_200_verified_ids"])),
        "preconditions_checked must include all verified arXiv ids",
    )
    _require(
        _nonempty_list(row.get("websearch_webfetch_top_sources"))
        and set(VERIFIED_SOURCE_URLS.values()).issubset(
            set(row["websearch_webfetch_top_sources"])
        ),
        "preconditions_checked must include WebSearch/WebFetch source URLs",
    )


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the artifact before writing or embedding it.

    Research notes are easy to copy-forward incorrectly. This validator keeps
    the machine-readable block exact so the reconciler can trust the fields.
    """

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
        artifact["offline_reproduced"] is False,
        "offline_reproduced must be bare bool false for this ingestion artifact",
    )
    _require(
        artifact["reproduced_levels"] == 0
        and isinstance(artifact["reproduced_levels"], int)
        and not isinstance(artifact["reproduced_levels"], bool),
        "reproduced_levels must be bare int 0 for this ingestion artifact",
    )
    _require(
        artifact["field_principles"] == FIELD_PRINCIPLES,
        "field_principles must match REQ-REPORT-4478",
    )
    _require(
        isinstance(artifact["random_seed"], int) and not isinstance(artifact["random_seed"], bool),
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
        _require(source in VERIFIED_SOURCE_URLS, f"source {source!r} is not a verified arXiv id")
        _require(source not in seen_sources, f"duplicate source in source_ids: {source}")
        seen_sources.add(source)

    methods = artifact["methods"]
    _require(isinstance(methods, list), "methods must be a list")
    method_sources: list[str] = []
    for method in methods:
        _require(
            isinstance(method, Mapping) and set(method) == REQUIRED_METHOD_FIELDS,
            "each method must be a dict with exactly the required fields",
        )
        for key, value in method.items():
            _require(
                isinstance(value, str) and bool(value.strip()),
                f"method field {key!r} must be a non-empty string",
            )
        source = method["arxiv_id"]
        _require(source in source_ids, "methods arxiv ids must match source_ids")
        _require(method["mapped_gap"] in GAP_KEYS, "method mapped_gap must name a required gap")
        method_sources.append(source)
    _require(method_sources == source_ids, "methods arxiv ids must match source_ids")

    gap_mapping = artifact["gap_mapping"]
    _require(isinstance(gap_mapping, Mapping) and set(gap_mapping) == GAP_KEYS, "gap_mapping must cover required gaps")
    for gap_id, row in gap_mapping.items():
        _require(isinstance(row, Mapping), f"gap_mapping {gap_id} must be a mapping")
        _require("principle" in row and "source_ids" in row and "next_experiment" in row, f"gap_mapping {gap_id} missing fields")
        _require(isinstance(row["principle"], str) and row["principle"].strip(), f"gap_mapping {gap_id} needs a principle")
        _require(
            isinstance(row["next_experiment"], str) and row["next_experiment"].strip(),
            f"gap_mapping {gap_id} needs a next_experiment",
        )
        _require(
            isinstance(row["source_ids"], list)
            and row["source_ids"]
            and set(row["source_ids"]).issubset(set(source_ids)),
            f"gap_mapping {gap_id} source_ids must be verified",
        )

    _require(
        artifact["strongest_for_v415"] == DEFAULT_STRONGEST_FOR_V415,
        "strongest_for_v415 must name the verified .415 package",
    )
    _validate_preconditions(artifact["preconditions_checked"])


def _artifact_json(artifact: Mapping[str, Any]) -> str:
    return json.dumps(artifact, indent=2, sort_keys=True)


def render_research_note(artifact: Mapping[str, Any]) -> str:
    """Render markdown with the artifact block first for automated parsing."""

    validate_artifact(artifact)
    source_line = ", ".join(f"arXiv:{source}" for source in DEFAULT_SOURCE_IDS)
    return f"""# ARC search/verifier SOTA ingestion .414 - 2026-06-20

```json
{_artifact_json(artifact)}
```

Reliable channel only: `research-studying.md`, `research-references.md`,
`scripts/sweep_clusters.py`, arXiv abs-page HTTP-200 checks, and
low-concurrency WebSearch/WebFetch of the top eight ARC search/verifier
sources. `.venv/bin/python scripts/sweep_clusters.py --help` succeeded.
`scripts/sweep_clusters.py 0 --max-results 8` and
`scripts/sweep_clusters.py 3 --max-results 8` emitted verifier and world-model
cluster URLs. No `/deep-research` call was made. No live solve, live LLM
inference, training run, or leaderboard submission was launched.

Sources checked: {source_line}.

## Gap Mapping

- GAP-ARCH-FEATURES: relational/delta verifier features should combine
  object slots, frame-change facts, transition deltas, and local state-graph
  context before scoring candidate plans.
- GAP-ARCH-GOAL: goal-vs-dynamics induction should infer the win predicate
  separately from the transition model, then use replayable counterexamples or
  disambiguating behavior queries when candidates disagree.
- GAP-ARCH-NO-HIERARCHICAL-SEARCH: hierarchical/MCTS verifier-guided search
  should replace flat repeated BFS with state-graph frontier expansion and
  verifier feedback over partial plans or world-model edits.

## Focused Sweep Result

- Graph-Based Exploration for ARC-AGI-3, arXiv:2512.24156, is the strongest
  direct search baseline: explicit state-action graph, salience-prioritized
  actions, and shortest paths to untested state-action pairs.
- ARC-AGI-3 benchmark, arXiv:2603.24621, is the metric anchor: agents must
  explore, infer goals, build dynamics, and plan efficiently without language
  instructions.
- Executable World Models, arXiv:2605.05138, is the verifier-grounded dynamics
  substrate: maintain executable model, verify against observations, simplify,
  and plan through it.
- Loop-OWM, arXiv:2606.12316, is the best fit for relational/delta state
  features: slots, transitions, loops, dense propagation, and correction.
- Agent2World, arXiv:2512.22336, supplies adaptive behavior-level tests for
  executable symbolic world models.
- AERA, arXiv:2605.25931, sharpens the explore/verify/plan action budget and
  warns that public ARC-AGI-3 scores can be shortcut by trivial strategies.
- Choose, Don't Label, arXiv:2604.08792, supplies the disambiguating query
  pattern for underdetermined candidate goals or dynamics.
- VerMCTS, arXiv:2402.08147, supplies verifier-in-the-loop tree search over
  partial programs; the transferable part is MCTS-style expansion with cheap
  verifier rejection.

## SOTA->Experiment Mapping

For `.415`, build the graph-state/delta-feature verifier plus hierarchical
verified search package: a state-action graph feeds relational/delta features
into the verifier, candidate goals are induced separately from executable
dynamics, and MCTS-style expansion uses verifier feedback to prune partial
plans or world-model edits. Count this artifact as a planning hand-off only:
`offline_reproduced=false`, `reproduced_levels=0`, and
`inference_substrate=aggregation_from_upstream_artifacts`.

{DEFAULT_STRONGEST_FOR_V415}
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
    """Check citations, required gap language, and the embedded artifact."""

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
        "/deep-research",
        "No live solve",
        "relational/delta",
        "goal-vs-dynamics",
        "hierarchical/MCTS",
        "GAP-ARCH-FEATURES",
        "GAP-ARCH-GOAL",
        "GAP-ARCH-NO-HIERARCHICAL-SEARCH",
        "flagged_for_v415",
        "aggregation_from_upstream_artifacts",
    ]
    for phrase in required_phrases:
        _require(phrase in note, f"research note missing required phrase: {phrase}")
    artifact_from_note(note)


RESEARCH_NOTE = render_research_note(build_artifact())


def write_outputs(*, artifact_path: Path, note_path: Path) -> dict[str, object]:
    """Write the result JSON and markdown note with matching artifact content."""

    artifact = build_artifact()
    validate_artifact(artifact)
    validate_research_note(RESEARCH_NOTE)

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(_artifact_json(artifact) + "\n", encoding="utf-8")

    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(RESEARCH_NOTE.rstrip() + "\n", encoding="utf-8")

    return artifact


def main() -> int:
    root_override = os.environ.get("CARNOT_EXP4478_ROOT")
    repo_root = Path(root_override) if root_override else Path(__file__).resolve().parents[2]
    artifact = write_outputs(
        artifact_path=repo_root / "results/experiment_4478_arc_search_verifier_sota_414.json",
        note_path=repo_root / RESEARCH_NOTE_RELATIVE_PATH,
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
