"""Exp 4565 verifier-router SOTA ingestion for the `.422` hand-off.

Spec refs: REQ-REPORT-4565, SCENARIO-REPORT-4565.

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
        "takes_over_current_a1_verifier_router",
        "fails_when",
        "v422_candidate",
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
        "exp4556_spec_read",
        "exp4556_artifact_read",
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
DEFAULT_HONEST_VERDICT = "complete: sota_ingestion_verifier_router_mapped"
DEFAULT_RANDOM_SEED = 4565
RESEARCH_NOTE_RELATIVE_PATH = (
    "docs/research-notes/sota-ingestion-verifier-router-421-2026-06-21.md"
)
STUDYING_SECTION_START = "<!-- EXP4565-VERIFIER-ROUTER-SOTA-START -->"
STUDYING_SECTION_END = "<!-- EXP4565-VERIFIER-ROUTER-SOTA-END -->"

FIELD_PRINCIPLES = {
    "honest_verdict": "terminal prefix; complete: sota_ingestion_verifier_router_mapped.",
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
        "the strongest method flagged as a .422 candidate -- closes the "
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
    "2601.22607": {
        "title": (
            "From Self-Evolving Synthetic Data to Verifiable-Reward RL: "
            "Post-Training Multi-turn Interactive Tool-Using Agents"
        ),
        "url": "https://arxiv.org/abs/2601.22607",
        "http_status": 200,
    },
    "2505.24760": {
        "title": (
            "REASONING GYM: Reasoning Environments for Reinforcement Learning "
            "with Verifiable Rewards"
        ),
        "url": "https://arxiv.org/abs/2505.24760",
        "http_status": 200,
    },
    "2602.01070": {
        "title": "What If We Allocate Test-Time Compute Adaptively?",
        "url": "https://arxiv.org/abs/2602.01070",
        "http_status": 200,
    },
    "2510.14913": {
        "title": "Budget-aware Test-time Scaling via Discriminative Verification",
        "url": "https://arxiv.org/abs/2510.14913",
        "http_status": 200,
    },
    "2601.09692": {
        "title": "Routing with Generated Data: Annotation-Free LLM Skill Estimation and Expert Selection",
        "url": "https://arxiv.org/abs/2601.09692",
        "http_status": 200,
    },
    "2606.06098": {
        "title": "IR3DE: A Linear Router for Large Language Models",
        "url": "https://arxiv.org/abs/2606.06098",
        "http_status": 200,
    },
    "2605.05138": {
        "title": "Executable World Models for ARC-AGI-3 in the Era of Coding Agents",
        "url": "https://arxiv.org/abs/2605.05138",
        "http_status": 200,
    },
    "2606.11521": {
        "title": "Counterexample Guided Learning in the Large using Reasoning Agents",
        "url": "https://arxiv.org/abs/2606.11521",
        "http_status": 200,
    },
}
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(f"arXiv:{source}" for source in CITATIONS_VERIFIED)

SWEEP_CLUSTER_URLS = [
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
        '(abs:"neural+guided+search"+OR+abs:"learned+heuristic"+OR+'
        'abs:"value+guided+search"+OR+abs:"program+induction"+OR+'
        'abs:"world+model"+OR+abs:"goal+induction")+AND+'
        '(abs:"planning"+OR+abs:"agent"+OR+abs:"reasoning"+OR+'
        'abs:"reinforcement+learning")&start=0&max_results=8'
        "&sortBy=submittedDate&sortOrder=descending"
    ),
]

S2_QUERIES = [
    "learned verifier process reward model cross task transfer tool use agent",
    "verifier guided candidate routing learned ranker best first search reasoning",
    "verifiable reward reinforcement learning cross domain generalization",
    "executable world model ARC-AGI-3 counterexample guided learning",
    "budget aware discriminative cascade router learned verifier",
]

DEFAULT_PRECONDITIONS_CHECKED = {
    "agents_md_read": True,
    "codex_md_read": True,
    "sweep_clusters_help_exit_0": True,
    "arxiv_api_reachable": True,
    "research_studying_filtered": True,
    "research_references_filtered": True,
    "exp4556_spec_read": True,
    "exp4556_artifact_read": True,
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
        "method": "self-evolving verifiable-reward RL for cross-game verifier transfer",
        "source_ids": ["2601.22607", "2505.24760"],
        "takes_over_current_a1_verifier_router": (
            "Exp 4556 has a learned DiscriminativeVerifier ranking cached "
            "rich_action_candidates, but its training signal is fixed. This "
            "method turns public-game variants and generated ARC-style tasks "
            "into verifier-scored RLVR data, using executable checks and "
            "held-out games to refresh the router without treating the win "
            "checker as the ranking signal."
        ),
        "fails_when": (
            "generated tasks leak game identity or target actions, the user or "
            "environment simulator is not checked by execution, or transfer is "
            "claimed on the same games used to evolve the verifier data."
        ),
        "v422_candidate": (
            "flagged_for_v422: self-evolving verifiable-reward data for the "
            "Exp 4556 DiscriminativeVerifier router"
        ),
    },
    {
        "method": "adaptive PRM-guided candidate expansion for best-first search",
        "source_ids": ["2602.01070"],
        "takes_over_current_a1_verifier_router": (
            "Exp 4556 uses the DiscriminativeVerifier as a final candidate "
            "ranker. Adaptive PRM-guided allocation promotes it into the live "
            "best-first controller: prune low-score branches, expand high-score "
            "branches, and spend action budget only where the learned process "
            "reward predicts useful progress."
        ),
        "fails_when": (
            "the verifier only reranks after generation, branch expansion is "
            "not tied to a fixed action budget, or process scores are allowed "
            "to override the exact offline reproduction gate."
        ),
        "v422_candidate": (
            "flagged_for_v422: PRM-guided adaptive expansion over Exp 4556 "
            "DiscriminativeVerifier scores"
        ),
    },
    {
        "method": "budget-aware discriminative verification hybrid",
        "source_ids": ["2510.14913"],
        "takes_over_current_a1_verifier_router": (
            "Exp 4556 showed that a cheap DiscriminativeVerifier router can "
            "be oracle-distinct but still null on live transfer. The "
            "budget-aware hybrid keeps the cheap score, combines it with "
            "agreement among candidate families, and reserves costly expansion "
            "for uncertain or high-upside frames."
        ),
        "fails_when": (
            "self-consistency counts near-duplicate replay actions, the router "
            "is compared against a weaker budget than the baseline, or the "
            "hybrid is used after the first-contact action budget has already "
            "been spent."
        ),
        "v422_candidate": (
            "flagged_for_v422: budget-aware verifier plus agreement gate for "
            "Exp 4556 candidate routing"
        ),
    },
    {
        "method": "generated-data and linear domain-expert routing",
        "source_ids": ["2601.09692", "2606.06098"],
        "takes_over_current_a1_verifier_router": (
            "Exp 4556 has one cross-game DiscriminativeVerifier head. RGD, "
            "CASCAL, and IR3DE suggest splitting by mechanic fingerprints: use "
            "generated variants to discover which router head or feature slice "
            "is reliable, then dispatch with a cheap linear or query-only "
            "router before ranking actions."
        ),
        "fails_when": (
            "generated variants do not create performance differentiation, the "
            "route feature is a disguised game ID, or the router requires "
            "retraining from scratch when a new public-game family is added."
        ),
        "v422_candidate": (
            "flagged_for_v422: mechanic-fingerprint router heads around the "
            "Exp 4556 DiscriminativeVerifier"
        ),
    },
    {
        "method": "executable world-model induction with counterexample-guided repair",
        "source_ids": ["2605.05138", "2606.11521"],
        "takes_over_current_a1_verifier_router": (
            "Exp 4556 ranks primitive action candidates, but it lacks a richer "
            "candidate source when the fixed action set is uninformative. The "
            "world-model branch induces executable GOAL and DYNAMICS code, "
            "feeds its plan branches to the DiscriminativeVerifier, and uses "
            "verifier-rejected states as counterexamples for the next bounded "
            "repair round."
        ),
        "fails_when": (
            "the induced model is trusted as an oracle, counterexamples are "
            "pasted as unstructured prompt text, or repair loops continue "
            "after the fixed first-contact action budget is exhausted."
        ),
        "v422_candidate": (
            "flagged_for_v422: executable branch candidates plus "
            "counterexample repair for Exp 4556 routing"
        ),
    },
]

FLAGGED_FOR_NEXT_ROADMAP = (
    "flagged_for_v422: adaptive PRM-guided candidate expansion over the Exp 4556 "
    "DiscriminativeVerifier, trained and refreshed with self-evolving "
    "verifiable-reward data (arXiv:2602.01070 + arXiv:2601.22607)"
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
        "exp4556_spec_read": "REQ-CAPSTONE-4556 spec",
        "exp4556_artifact_read": "Exp 4556 verifier-router artifact",
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
        "preconditions_checked must record the focused cluster 0/6 URLs",
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
        "honest_verdict must match REQ-REPORT-4565 complete path",
    )
    _require(
        artifact["inference_substrate"] == INFERENCE_SUBSTRATE,
        "inference_substrate must declare aggregation from upstream artifacts",
    )
    _require(
        artifact["field_principles"] == FIELD_PRINCIPLES
        and set(artifact["field_principles"]) == REQUIRED_ARTIFACT_FIELDS,
        "field_principles must match REQ-REPORT-4565",
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
            "takes_over_current_a1_verifier_router",
            "fails_when",
            "v422_candidate",
        ):
            _require(
                isinstance(method[key], str) and bool(method[key].strip()),
                f"methods_mapped field {key!r} must be a non-empty string",
            )
        mapping = method["takes_over_current_a1_verifier_router"]
        _require(
            "Exp 4556" in mapping and "DiscriminativeVerifier" in mapping,
            "methods_mapped must map onto the current Exp 4556 DiscriminativeVerifier router",
        )
        _require(
            method["v422_candidate"].startswith("flagged_for_v422:"),
            "methods_mapped v422_candidate must flag a .422 input",
        )
    _require(
        used_method_sources == set(CITATIONS_VERIFIED),
        "methods_mapped must use every verified citation",
    )

    _require(
        artifact["flagged_for_next_roadmap"] == FLAGGED_FOR_NEXT_ROADMAP
        and str(artifact["flagged_for_next_roadmap"]).startswith("flagged_for_v422:"),
        "flagged_for_next_roadmap must match the verified .422 candidate",
    )
    _validate_preconditions(artifact["preconditions_checked"])


def _artifact_json(artifact: Mapping[str, Any]) -> str:
    return json.dumps(artifact, indent=2, sort_keys=True)


def render_research_note(artifact: Mapping[str, Any]) -> str:
    """Render markdown with the artifact block first for automated parsing."""

    validate_artifact(artifact)
    source_line = ", ".join(f"arXiv:{source}" for source in CITATIONS_VERIFIED)
    return f"""# Verifier-router SOTA ingestion .421 - 2026-06-21

```json
{_artifact_json(artifact)}
```

Reliable channel only: `scripts/sweep_clusters.py`, `scripts/sweep_semscholar.py`,
arXiv abs-page HTTP-200 checks, and low-concurrency WebSearch/WebFetch of eight
top learned-verifier / PRM-routing / generated-router / world-model sources.
Preconditions passed before any claim was promoted:
`.venv/bin/python scripts/sweep_clusters.py --help` exited zero and
`curl -sf -o /dev/null https://export.arxiv.org/api/query?search_query=all:test`
confirmed arXiv API reachability. `scripts/sweep_clusters.py 0 --max-results 8`
and `scripts/sweep_clusters.py 6 --max-results 8` emitted the focused verifier
and learned-search cluster URLs. `scripts/sweep_semscholar.py` ran five focused
queries and returned HTTP 429 on all five, so no S2-only claim was promoted. No
`/deep-research` call was made. No training, live LLM inference, leaderboard
submission, or live solve was launched. No ops/status/traceability files or
`scripts/research_conductor.py` were modified.

Already-discovered corpus read through a learned-verifier cross-task transfer /
verifier-guided routing filter: `research-studying.md`, `research-references.md`,
`openspec/capabilities/capstone/spec.md` at `REQ-CAPSTONE-4556`, and
`results/experiment_4556_verifier_router_generic_transfer.json`. The current
mechanism this maps onto is Exp 4556: a trained cross-game `DiscriminativeVerifier`
ranks `rich_action_candidates` by learned `cross_game_features_v3` scores without
using executable win-checks as the ranking signal. The 2026-06-21 artifact was
an honest live-transfer null, so .422 should turn the same learned signal into a
budget-aware live controller rather than merely reranking a fixed candidate list.

Sources checked: {source_line}.

## Per-Method Mapping

- **Self-evolving verifiable-reward RL** (arXiv:2601.22607, arXiv:2505.24760):
  take over the Exp 4556 verifier-training substrate by generating verifier-
  checked ARC-style variants and refreshing the learned router on held-out
  games. This carries the cross-task transfer lesson: exact checkers create the
  reward substrate, but held-out games must remain untouched.
- **Adaptive PRM-guided candidate expansion** (arXiv:2602.01070): take over the
  live Exp 4556 candidate router by using the DiscriminativeVerifier as a
  process-reward controller for best-first prune/expand decisions, not just as a
  final sorter.
- **Budget-aware discriminative verification** (arXiv:2510.14913): keep the
  cheap learned verifier score, add agreement among candidate families, and
  spend expensive expansion only on uncertain or high-upside frames.
- **CASCAL / IR3DE generated-data routing** (arXiv:2601.09692, arXiv:2606.06098):
  split the current single Exp 4556 verifier head into cheap mechanic-fingerprint
  dispatch over router heads, trained from generated variants that actually
  differentiate candidate performance.
- **Executable World Models plus Counterexample Guided Learning**
  (arXiv:2605.05138, arXiv:2606.11521): add richer executable branch candidates
  when primitive action candidates are weak, then feed verifier-rejected states
  back as bounded counterexamples.

## .422 Candidate

{FLAGGED_FOR_NEXT_ROADMAP}

The practical next experiment should keep Exp 4556's oracle-distinct learned
score, same manufactured-variant measurement, random-router control, and offline
reproduction gate. The change is where the score acts: it should drive adaptive
best-first expansion and budget allocation before actions are spent. Self-
evolving verifiable-reward data supplies the retraining stream, budget-aware
discriminative verification supplies the cheap hybrid scoring rule, CASCAL/IR3DE
supplies mechanic dispatch, and executable world models plus Counterexample
Guided Learning supply fallback branches and repair when the primitive candidate
set is too weak.
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
        "Exp 4556",
        "DiscriminativeVerifier",
        "adaptive PRM-guided candidate expansion",
        "Budget-aware discriminative verification",
        "CASCAL",
        "IR3DE",
        "Executable World Models",
        "Counterexample Guided Learning",
        "flagged_for_v422",
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
## 2026-06-21 Exp 4565 - .421 verifier-router SOTA ingestion - INGESTED

**Status:** INGESTED into `docs/research-notes/sota-ingestion-verifier-router-421-2026-06-21.md`
and `results/experiment_4565_sota_ingestion_verifier_router.json`.

**Preconditions:** `scripts/sweep_clusters.py --help` succeeded; the arXiv API
reachability check succeeded; `scripts/sweep_clusters.py` clusters 0 and 6
emitted focused URLs; `scripts/sweep_semscholar.py` ran five focused queries
and returned HTTP 429 on all five, so no S2-only source was promoted. Top
sources were verified by arXiv abs-page HTTP 200 and low-concurrency
WebSearch/WebFetch. `/deep-research` was not invoked. No live solve, training
run, live LLM inference, leaderboard submission, ops/status/traceability edit,
or `scripts/research_conductor.py` edit occurred.

**Methods marked ingested:** self-evolving verifiable-reward RL for cross-game
verifier transfer (arXiv:2601.22607, arXiv:2505.24760), adaptive PRM-guided
candidate expansion (arXiv:2602.01070), budget-aware discriminative
verification (arXiv:2510.14913), CASCAL / IR3DE generated-data routing
(arXiv:2601.09692, arXiv:2606.06098), and executable world-model plus
counterexample-guided repair (arXiv:2605.05138, arXiv:2606.11521).

{FLAGGED_FOR_NEXT_ROADMAP}
{STUDYING_SECTION_END}"""


def update_research_studying(path: Path) -> None:
    """Insert or replace the Exp 4565 studying-queue section."""

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
    root = Path(os.environ.get("CARNOT_EXP4565_ROOT", "."))
    artifact = write_outputs(
        artifact_path=root / "results/experiment_4565_sota_ingestion_verifier_router.json",
        note_path=root / RESEARCH_NOTE_RELATIVE_PATH,
        studying_path=root / "research-studying.md",
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
