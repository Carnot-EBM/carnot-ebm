"""Exp 4553 LLM-inducer SOTA ingestion for the `.421` hand-off.

Spec refs: REQ-REPORT-4553, SCENARIO-REPORT-4553.

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
        "takes_over_current_a1_llm_proposer",
        "fails_when",
        "v421_candidate",
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
        "a1_llm_proposer_spec_read",
        "a1_llm_proposer_artifact_read",
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
DEFAULT_HONEST_VERDICT = "complete: sota_ingestion_llm_inducer_mapped"
DEFAULT_RANDOM_SEED = 4553
RESEARCH_NOTE_RELATIVE_PATH = "docs/research-notes/arc-llm-inducer-sota-420.md"
STUDYING_SECTION_START = "<!-- EXP4553-LLM-INDUCER-SOTA-START -->"
STUDYING_SECTION_END = "<!-- EXP4553-LLM-INDUCER-SOTA-END -->"

FIELD_PRINCIPLES = {
    "honest_verdict": "terminal prefix; complete: sota_ingestion_llm_inducer_mapped.",
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
        "the strongest method flagged as a .421 candidate -- closes the "
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
    "2605.05138": {
        "title": "Executable World Models for ARC-AGI-3 in the Era of Coding Agents",
        "url": "https://arxiv.org/abs/2605.05138",
        "http_status": 200,
    },
    "2510.14331": {
        "title": "LLM Priors for ERM over Programs",
        "url": "https://arxiv.org/abs/2510.14331",
        "http_status": 200,
    },
    "2305.14591": {
        "title": "ALGO: Synthesizing Algorithmic Programs with LLM-Generated Oracle Verifiers",
        "url": "https://arxiv.org/abs/2305.14591",
        "http_status": 200,
    },
    "2603.20334": {
        "title": "Procedural Refinement by LLM-driven Algorithmic Debugging for ARC-AGI-2",
        "url": "https://arxiv.org/abs/2603.20334",
        "http_status": 200,
    },
    "2606.11521": {
        "title": "Counterexample Guided Learning in the Large using Reasoning Agents",
        "url": "https://arxiv.org/abs/2606.11521",
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
    "LLM world model induction ARC-AGI-3 executable world models",
    "LLM priors ERM over programs ALGO program synthesis",
    "procedural refinement counterexample guided learning verifier guided refinement",
    "goal acquisition ARC-AGI-3 goal shift detection",
]

DEFAULT_PRECONDITIONS_CHECKED = {
    "agents_md_read": True,
    "codex_md_read": True,
    "sweep_clusters_help_exit_0": True,
    "arxiv_api_reachable": True,
    "research_studying_filtered": True,
    "research_references_filtered": True,
    "a1_llm_proposer_spec_read": True,
    "a1_llm_proposer_artifact_read": True,
    "research_studying_updated": True,
    "sweep_clusters_used": True,
    "sweep_clusters_urls": SWEEP_CLUSTER_URLS,
    "sweep_semscholar_used": True,
    "sweep_semscholar_queries": S2_QUERIES,
    "sweep_semscholar_arxiv_ids": [],
    "sweep_semscholar_rate_limited_queries": S2_QUERIES,
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
        "method": "Family-B executable world-model induction for per-level re-induction",
        "source_ids": ["2605.05138", "2603.24621"],
        "takes_over_current_a1_llm_proposer": (
            "Exp 4544 already asks the LLM for separate GOAL and DYNAMICS "
            "candidates plus a plan. This method makes that slot stricter: "
            "the proposer emits an executable Python world model, verifies "
            "prior observations and held-out post-level-up transitions, "
            "refactors toward simpler state variables, then plans only through "
            "accepted GOAL+DYNAMICS candidates."
        ),
        "fails_when": (
            "the induced executable model is trusted as proof without held-out "
            "transition checks, or when an L1 completion predicate survives a "
            "new ARC-AGI-3 goal-acquisition phase after the level shifts."
        ),
        "v421_candidate": (
            "flagged_for_v421: Family-B executable GOAL+DYNAMICS proposer "
            "inside the Exp 4544 re-induction hook"
        ),
    },
    {
        "method": "LLM-PV proposal distribution with held-out execution selection",
        "source_ids": ["2510.14331"],
        "takes_over_current_a1_llm_proposer": (
            "Exp 4544 uses a single local GGUF call per refinement round. "
            "LLM-PV reframes the call as a proposal distribution over compact "
            "candidate programs: sample bounded GOAL, DYNAMICS, and plan "
            "candidates, execute each on held-out transitions, and select the "
            "lowest verifier-energy candidate without gradient updates."
        ),
        "fails_when": (
            "the validation set is not separated from the evidence used to "
            "prompt the proposer, or when the proposal count grows into an "
            "unbounded best-of-N search that loses the action-efficiency target."
        ),
        "v421_candidate": (
            "flagged_for_v421: bounded LLM-PV sampling around each Exp 4544 "
            "GOAL+DYNAMICS refinement round"
        ),
    },
    {
        "method": "ALGO-style LLM-generated oracle verifier for candidate plans",
        "source_ids": ["2305.14591"],
        "takes_over_current_a1_llm_proposer": (
            "Exp 4544 feeds verifier failures back to the proposer, but its "
            "candidate verifier is fixed. ALGO suggests asking the proposer "
            "for a small executable reference oracle or exhaustive checker "
            "alongside the plan, then using that checker to guide search and "
            "reject candidate algorithms before acting."
        ),
        "fails_when": (
            "the LLM-generated oracle is itself unchecked; in ARC-AGI-3 it "
            "must be validated against observed transitions before it can "
            "score plans or the loop becomes verifier self-confirmation."
        ),
        "v421_candidate": (
            "flagged_for_v421: proposer-emitted oracle checkers validated by "
            "observed transitions before plan selection"
        ),
    },
    {
        "method": "ABPR procedural refinement with executable trace feedback",
        "source_ids": ["2603.20334"],
        "takes_over_current_a1_llm_proposer": (
            "Exp 4544 has a bounded verifier-guided refinement loop, but the "
            "feedback can be only a first counterexample or reachability "
            "failure. ABPR upgrades that feedback into structured executable "
            "trace evidence about which subgoal, relation, or transformation "
            "failed, then asks the proposer to repair that specific part."
        ),
        "fails_when": (
            "Prolog-style proof traces are copied without an ARC-AGI-3 state "
            "representation, or when refinement optimizes final-frame agreement "
            "instead of the GOAL and DYNAMICS clauses used by the live planner."
        ),
        "v421_candidate": (
            "flagged_for_v421: structured trace feedback for the existing "
            "three-round Exp 4544 refinement budget"
        ),
    },
    {
        "method": "counterexample-guided learning for verifier-returned failures",
        "source_ids": ["2606.11521"],
        "takes_over_current_a1_llm_proposer": (
            "Exp 4544 already returns a first verifier counterexample or "
            "plan-reachability failure. Counterexample Guided Learning turns "
            "that into the central teaching signal: minimize or cluster the "
            "failing trace, feed it back to the local proposer, and stop after "
            "the bounded refinement budget or a reachable accepted plan."
        ),
        "fails_when": (
            "counterexamples are appended as extra prompt text without "
            "normalization, or when the teacher returns examples outside the "
            "observed ARC state/action language the proposer can execute."
        ),
        "v421_candidate": (
            "flagged_for_v421: minimized verifier counterexamples as the "
            "primary repair signal for Exp 4544"
        ),
    },
]

FLAGGED_FOR_NEXT_ROADMAP = (
    "flagged_for_v421: combine Family-B executable world-model induction "
    "(arXiv:2605.05138) with bounded counterexample-guided refinement "
    "(arXiv:2606.11521) inside the Exp 4544 GOAL+DYNAMICS+plan proposer"
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
        "a1_llm_proposer_spec_read": "REQ-ARC-WMTE-4544 spec",
        "a1_llm_proposer_artifact_read": "Exp 4544 LLM-proposer artifact",
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
        verdict == DEFAULT_HONEST_VERDICT,
        "honest_verdict must match REQ-REPORT-4553 complete path",
    )
    _require(
        artifact["inference_substrate"] == INFERENCE_SUBSTRATE,
        "inference_substrate must declare aggregation from upstream artifacts",
    )
    _require(
        artifact["field_principles"] == FIELD_PRINCIPLES
        and set(artifact["field_principles"]) == REQUIRED_ARTIFACT_FIELDS,
        "field_principles must match REQ-REPORT-4553",
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
            "takes_over_current_a1_llm_proposer",
            "fails_when",
            "v421_candidate",
        ):
            _require(
                isinstance(method[key], str) and bool(method[key].strip()),
                f"methods_mapped field {key!r} must be a non-empty string",
            )
        mapping = method["takes_over_current_a1_llm_proposer"]
        _require(
            "Exp 4544" in mapping or "exp4544" in mapping,
            "methods_mapped must map onto the current Exp 4544 A1 LLM-proposer mechanism",
        )
        _require(
            method["v421_candidate"].startswith("flagged_for_v421:"),
            "methods_mapped v421_candidate must flag a .421 input",
        )
    _require(
        used_method_sources == set(CITATIONS_VERIFIED),
        "methods_mapped must use every verified citation",
    )

    _require(
        artifact["flagged_for_next_roadmap"] == FLAGGED_FOR_NEXT_ROADMAP
        and str(artifact["flagged_for_next_roadmap"]).startswith("flagged_for_v421:"),
        "flagged_for_next_roadmap must match the verified .421 candidate",
    )
    _validate_preconditions(artifact["preconditions_checked"])


def _artifact_json(artifact: Mapping[str, Any]) -> str:
    return json.dumps(artifact, indent=2, sort_keys=True)


def render_research_note(artifact: Mapping[str, Any]) -> str:
    """Render markdown with the artifact block first for automated parsing."""

    validate_artifact(artifact)
    source_line = ", ".join(f"arXiv:{source}" for source in CITATIONS_VERIFIED)
    return f"""# ARC LLM-inducer SOTA ingestion .420 - 2026-06-21

```json
{_artifact_json(artifact)}
```

Reliable channel only: `scripts/sweep_clusters.py`, `scripts/sweep_semscholar.py`,
arXiv abs-page HTTP-200 checks, and low-concurrency WebSearch/WebFetch of six
top LLM-inducer / verifier-refinement / goal-acquisition sources. Preconditions
passed before any claim was promoted: `.venv/bin/python scripts/sweep_clusters.py --help`
exited zero and
`curl -sf -o /dev/null https://export.arxiv.org/api/query?search_query=all:test`
confirmed arXiv API reachability. `scripts/sweep_clusters.py 6 --max-results 8`
and `scripts/sweep_clusters.py 3 --max-results 8` emitted the focused world-model
cluster URLs. `scripts/sweep_semscholar.py` ran four focused queries and returned
HTTP 429 on all four, so no S2-only claim was promoted. No `/deep-research`
call was made. No training, live LLM inference, leaderboard submission, or live
solve was launched. No ops/status/traceability files or
`scripts/research_conductor.py` were modified.

Already-discovered corpus read through an LLM-world-model-induction /
verifier-guided-refinement / goal-shift filter: `research-studying.md`,
`research-references.md`, `openspec/capabilities/arc-world-model-trust-energy/spec.md`
at `REQ-ARC-WMTE-4544`, and
`results/experiment_4544_llm_proposer_reinduction.json`. The current mechanism
this maps onto is Exp 4544: A1 asks the local generator for separate
GOAL+DYNAMICS+plan candidates and runs a bounded verifier-guided refinement loop
where verifier counterexamples or plan-reachability failures are fed back to the
proposer for at most three rounds.

Sources checked: {source_line}.

## Per-Method Mapping

- **Family-B executable world-model induction** (arXiv:2605.05138,
  arXiv:2603.24621): take over the Exp 4544 post-level-up proposer with an
  executable Python world model that separates GOAL and DYNAMICS, verifies
  transitions, refactors toward simpler state, and plans only through accepted
  models. This is the strongest .421 candidate because it matches ARC-AGI-3's
  goal acquisition and world-model demands directly.
- **LLM-PV proposal distribution with held-out execution selection**
  (arXiv:2510.14331): treat the local LLM as a bounded proposal distribution
  over candidate programs, execute and score each candidate on held-out
  transitions, and select without gradient updates. It fails if the validation
  transitions leak into the prompt or if best-of-N becomes unbounded.
- **ALGO-style LLM-generated oracle verifier** (arXiv:2305.14591): have the
  proposer emit a small executable checker or reference oracle with the plan,
  then validate that checker against observations before using it to guide
  search. It fails if the LLM-generated oracle is accepted without independent
  transition checks.
- **ABPR procedural refinement with executable trace feedback**
  (arXiv:2603.20334): replace generic "plan failed" feedback with structured
  trace evidence about which subgoal, relation, or transformation failed. It
  fails if ARC-AGI-2 static-grid proof traces are copied without an ARC-AGI-3
  state/action representation.
- **Counterexample Guided Learning** (arXiv:2606.11521): make minimized verifier
  counterexamples the primary teaching signal inside the three-round Exp 4544
  bounded verifier-guided refinement loop. It fails when counterexamples are
  unnormalized prompt text rather than executable state/action evidence.

## .421 Candidate

{FLAGGED_FOR_NEXT_ROADMAP}

The practical next experiment should keep Exp 4544's trigger, local proposer,
oracle-distinct verifier energy, and three-round cap, but make the proposal unit
an executable GOAL+DYNAMICS world model plus a candidate plan. LLM-PV supplies
the bounded proposal-selection semantics, ALGO supplies an optional generated
checker only after independent validation, ABPR supplies structured trace
feedback, and Counterexample Guided Learning supplies the repair signal. The
goal-shift risk from ARC-AGI-3 means a stale L1 predicate must be treated as a
counterexample after any level-up.
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
        "GOAL+DYNAMICS+plan proposer",
        "bounded verifier-guided refinement loop",
        "Exp 4544",
        "LLM-PV",
        "ALGO",
        "ABPR",
        "Counterexample Guided Learning",
        "goal-shift",
        "flagged_for_v421",
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
## 2026-06-21 Exp 4553 - .420 LLM-inducer SOTA ingestion - INGESTED

**Status:** INGESTED into `docs/research-notes/arc-llm-inducer-sota-420.md`
and `results/experiment_4553_sota_ingestion_llm_inducer.json`.

**Preconditions:** `scripts/sweep_clusters.py --help` succeeded; the arXiv API
reachability check succeeded; `scripts/sweep_clusters.py` clusters 6 and 3
emitted focused URLs; `scripts/sweep_semscholar.py` ran four focused queries
and returned HTTP 429 on all four, so no S2-only source was promoted. Top
sources were verified by arXiv abs-page HTTP 200 and low-concurrency
WebSearch/WebFetch. `/deep-research` was not invoked. No live solve, training
run, live LLM inference, leaderboard submission, ops/status/traceability edit,
or `scripts/research_conductor.py` edit occurred.

**Methods marked ingested:** Family-B executable world-model induction
(arXiv:2605.05138, arXiv:2603.24621), LLM-PV proposal distribution with held-out
execution selection (arXiv:2510.14331), ALGO LLM-generated oracle verification
(arXiv:2305.14591), ABPR procedural refinement (arXiv:2603.20334), and
Counterexample Guided Learning (arXiv:2606.11521).

{FLAGGED_FOR_NEXT_ROADMAP}
{STUDYING_SECTION_END}"""


def update_research_studying(path: Path) -> None:
    """Insert or replace the Exp 4553 studying-queue section."""

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
    root = Path(os.environ.get("CARNOT_EXP4553_ROOT", "."))
    artifact = write_outputs(
        artifact_path=root / "results/experiment_4553_sota_ingestion_llm_inducer.json",
        note_path=root / RESEARCH_NOTE_RELATIVE_PATH,
        studying_path=root / "research-studying.md",
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
