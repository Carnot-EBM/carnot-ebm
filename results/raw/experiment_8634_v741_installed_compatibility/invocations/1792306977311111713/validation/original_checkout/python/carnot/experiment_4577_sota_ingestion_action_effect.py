"""Exp 4577 action-effect SOTA ingestion for the `.423` hand-off.

Spec refs: REQ-REPORT-4577, SCENARIO-REPORT-4577.

This module records a literature-synthesis artifact. It intentionally does not
run the ARC agent, train a model, or submit to the leaderboard. The deterministic
writer makes the markdown note, result JSON, and studying-queue update easy to
test and safe to rerun.
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
        "takes_over_current_a1_a2_mechanisms",
        "fails_when",
        "v423_candidate",
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
        "exp4568_spec_read",
        "exp4569_spec_read",
        "exp4568_artifact_read",
        "exp4569_artifact_read",
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
VALID_TARGET_TRACKS = frozenset(
    {
        "a1_action_effect_predictor",
        "a2_verifier_guided_expansion",
    }
)
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
DEFAULT_HONEST_VERDICT = "complete: sota_ingestion_action_effect_mapped"
DEFAULT_RANDOM_SEED = 4577
RESEARCH_NOTE_RELATIVE_PATH = (
    "docs/research-notes/sota-ingestion-action-effect-422-2026-06-22.md"
)
STUDYING_SECTION_START = "<!-- EXP4577-ACTION-EFFECT-SOTA-START -->"
STUDYING_SECTION_END = "<!-- EXP4577-ACTION-EFFECT-SOTA-END -->"

FIELD_PRINCIPLES = {
    "honest_verdict": "terminal prefix; complete: sota_ingestion_action_effect_mapped.",
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
        "the strongest method flagged as a .423 candidate -- closes the "
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
    "2502.18407": {
        "title": "AgentRM: Enhancing Agent Generalization with Reward Modeling",
        "url": "https://arxiv.org/abs/2502.18407",
        "http_status": 200,
    },
    "2504.16828": {
        "title": "Process Reward Models That Think",
        "url": "https://arxiv.org/abs/2504.16828",
        "http_status": 200,
    },
    "2502.00271": {
        "title": "Scaling Flaws of Verifier-Guided Search in Mathematical Reasoning",
        "url": "https://arxiv.org/abs/2502.00271",
        "http_status": 200,
    },
    "2602.01070": {
        "title": "What If We Allocate Test-Time Compute Adaptively?",
        "url": "https://arxiv.org/abs/2602.01070",
        "http_status": 200,
    },
    "2601.22607": {
        "title": (
            "From Self-Evolving Synthetic Data to Verifiable-Reward RL: "
            "Post-Training Multi-turn Interactive Tool-Using Agents"
        ),
        "url": "https://arxiv.org/abs/2601.22607",
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
    "ARC-AGI-3 clickability action effect frame change CNN interactive agent exploration",
    "verifier guided candidate expansion process reward model AgentRM ThinkPRM adaptive test time compute",
]

WEBSEARCH_WEBFETCH_TOP_SOURCES = [
    CITATIONS_VERIFIED["2603.24621"]["url"],
    CITATIONS_VERIFIED["2502.18407"]["url"],
    CITATIONS_VERIFIED["2504.16828"]["url"],
    CITATIONS_VERIFIED["2502.00271"]["url"],
    CITATIONS_VERIFIED["2602.01070"]["url"],
    CITATIONS_VERIFIED["2601.22607"]["url"],
    "https://github.com/DriesSmit/ARC3-solution",
]

DEFAULT_PRECONDITIONS_CHECKED = {
    "agents_md_read": True,
    "codex_md_read": True,
    "sweep_clusters_help_exit_0": True,
    "arxiv_api_reachable": True,
    "research_studying_filtered": True,
    "research_references_filtered": True,
    "exp4568_spec_read": True,
    "exp4569_spec_read": True,
    "exp4568_artifact_read": True,
    "exp4569_artifact_read": True,
    "research_studying_updated": True,
    "sweep_clusters_used": True,
    "sweep_clusters_urls": SWEEP_CLUSTER_URLS,
    "sweep_semscholar_used": True,
    "sweep_semscholar_queries": S2_QUERIES,
    "sweep_semscholar_arxiv_ids": [],
    "sweep_semscholar_rate_limited_queries": S2_QUERIES,
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
        "method": "StochasticGoose-style learned frame-change clickability predictor",
        "source_ids": ["2603.24621"],
        "target_track": "a1_action_effect_predictor",
        "takes_over_current_a1_a2_mechanisms": (
            "Exp 4568 trained and wired a pooled clickability/action-effect "
            "predictor but ended as an honest no-gain null because it only "
            "reranked existing candidates. This method keeps the ARC-AGI-3 "
            "action-efficiency target and changes the role: train a coarse CNN "
            "or equivalent frame-change model to decide which action families "
            "and click locations enter the candidate set before the explorer "
            "spends actions."
        ),
        "fails_when": (
            "the predictor is evaluated only on frame-change classification, "
            "allowed to suppress rare necessary actions without a positive "
            "control, or left as a final sorter over a pool where the winning "
            "action is absent."
        ),
        "v423_candidate": (
            "flagged_for_v423: StochasticGoose-style action-effect model used "
            "as candidate expansion prior, not only as Exp 4568 reranker"
        ),
    },
    {
        "method": "AgentRM generalizable reward model for agent search",
        "source_ids": ["2502.18407", "2502.00271"],
        "target_track": "a2_verifier_guided_expansion",
        "takes_over_current_a1_a2_mechanisms": (
            "Exp 4569 promoted a learned DiscriminativeVerifier into frontier "
            "expansion but still nulled on generic transfer. AgentRM takes over "
            "that Exp 4569 control point by scoring partial agent trajectories "
            "for test-time search and beam/frontier selection, while the "
            "scaling-flaws paper supplies the guardrail against verifier-only "
            "pruning on hard or out-of-distribution branches."
        ),
        "fails_when": (
            "reward scores replace the exact reproduction gate, the reward "
            "model is trained on the held-out games used for transfer claims, "
            "or branch pruning is irreversible when the verifier is uncertain."
        ),
        "v423_candidate": (
            "flagged_for_v423: AgentRM-style trajectory reward for bounded "
            "Exp 4569 candidate expansion with anti-pruning guardrails"
        ),
    },
    {
        "method": "ThinkPRM generative process verifier for expansion quality",
        "source_ids": ["2504.16828", "2502.00271"],
        "target_track": "a2_verifier_guided_expansion",
        "takes_over_current_a1_a2_mechanisms": (
            "Exp 4569 used a cheap discriminative score but lacked a stronger "
            "explanation-based process verifier for ambiguous branches. "
            "ThinkPRM takes over the expensive-check tier: ask for generative "
            "step verification only on high-upside candidate branches, then "
            "feed the result into best-first expansion while keeping the "
            "scaling-flaws caution as a random-priority and repeated-sampling "
            "control requirement."
        ),
        "fails_when": (
            "long verification chains consume the first-contact action budget, "
            "local step plausibility is mistaken for final progress, or a "
            "single ThinkPRM score is allowed to eliminate every alternative "
            "branch."
        ),
        "v423_candidate": (
            "flagged_for_v423: ThinkPRM only as sparse expensive check inside "
            "Exp 4569 expansion, never as sole branch killer"
        ),
    },
    {
        "method": "adaptive PRM-guided best-first candidate expansion",
        "source_ids": ["2602.01070", "2502.00271"],
        "target_track": "a2_verifier_guided_expansion",
        "takes_over_current_a1_a2_mechanisms": (
            "Exp 4569 already tries verifier-guided expansion, but its null "
            "shows the score must allocate expansion budget more carefully. "
            "Adaptive test-time compute allocation takes over the live "
            "best-first scheduler: aggregate process rewards to choose which "
            "frontier nodes to expand, when to widen, and when to fall back to "
            "less verifier-dependent search."
        ),
        "fails_when": (
            "the controller spends extra compute without lowering actions to "
            "first level-up, compares against an easier baseline budget, or "
            "inherits verifier-guided search scaling flaws by pruning valid "
            "paths on weak PRM evidence."
        ),
        "v423_candidate": (
            "flagged_for_v423: adaptive PRM-guided expansion scheduler over "
            "Exp 4569 with repeated-sampling/random-priority controls"
        ),
    },
    {
        "method": "self-evolving verifiable-reward data for predictor and verifier refresh",
        "source_ids": ["2601.22607", "2502.00271"],
        "target_track": "a2_verifier_guided_expansion",
        "takes_over_current_a1_a2_mechanisms": (
            "Exp 4568 and Exp 4569 both depend on fixed local corpora. "
            "Self-evolving verifiable-reward data takes over the refresh loop: "
            "generate new interaction traces with executable per-instance "
            "checks, add hard negatives where verifier-guided search pruned "
            "valid paths, and retrain the action-effect prior and expansion "
            "verifier only after held-out checks stay separate."
        ),
        "fails_when": (
            "synthetic traces leak game identity, generated checkers are not "
            "execution-tested, or the self-evolving loop optimizes the verifier "
            "instead of held-out action efficiency."
        ),
        "v423_candidate": (
            "flagged_for_v423: self-evolving checked traces to refresh the "
            "Exp 4568 predictor and Exp 4569 expansion verifier"
        ),
    },
]

FLAGGED_FOR_NEXT_ROADMAP = (
    "flagged_for_v423: use a StochasticGoose-style learned action-effect model "
    "as the candidate-expansion prior, then allocate Exp 4569 best-first "
    "frontier budget with adaptive PRM guidance and scaling-flaw controls "
    "(arXiv:2603.24621 + arXiv:2602.01070 + arXiv:2502.00271)"
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
    """Build the deterministic artifact embedded in the markdown note.

    The artifact is a receipt for literature aggregation, not a measurement run.
    Keeping it deterministic lets the tests prove that every promoted method has
    an explicit citation and a mapped hand-off into the current A1/A2 mechanisms.
    """

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
        "exp4568_spec_read": "REQ-ARC-FCP-4568 spec",
        "exp4569_spec_read": "REQ-CAPSTONE-4569 spec",
        "exp4568_artifact_read": "Exp 4568 clickability artifact",
        "exp4569_artifact_read": "Exp 4569 verifier expansion artifact",
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
        "honest_verdict must match REQ-REPORT-4577 complete path",
    )
    _require(
        artifact["inference_substrate"] == INFERENCE_SUBSTRATE,
        "inference_substrate must declare aggregation from upstream artifacts",
    )
    _require(
        artifact["field_principles"] == FIELD_PRINCIPLES
        and set(artifact["field_principles"]) == REQUIRED_ARTIFACT_FIELDS,
        "field_principles must match REQ-REPORT-4577",
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
            "takes_over_current_a1_a2_mechanisms",
            "fails_when",
            "v423_candidate",
        ):
            _require(
                isinstance(method[key], str) and bool(method[key].strip()),
                f"methods_mapped field {key!r} must be a non-empty string",
            )
        mapping = method["takes_over_current_a1_a2_mechanisms"]
        _require(
            "Exp 4568" in mapping or "Exp 4569" in mapping,
            "methods_mapped must map onto Exp 4568 or Exp 4569",
        )
        _require(
            method["v423_candidate"].startswith("flagged_for_v423:"),
            "methods_mapped v423_candidate must flag a .423 input",
        )
    _require(
        used_method_sources == set(CITATIONS_VERIFIED),
        "methods_mapped must use every verified citation",
    )
    _require(
        seen_tracks == set(VALID_TARGET_TRACKS),
        "methods_mapped must cover A1 action-effect and A2 verifier expansion",
    )

    _require(
        artifact["flagged_for_next_roadmap"] == FLAGGED_FOR_NEXT_ROADMAP
        and str(artifact["flagged_for_next_roadmap"]).startswith("flagged_for_v423:"),
        "flagged_for_next_roadmap must match the verified .423 candidate",
    )
    _validate_preconditions(artifact["preconditions_checked"])


def _artifact_json(artifact: Mapping[str, Any]) -> str:
    return json.dumps(artifact, indent=2, sort_keys=True)


def render_research_note(artifact: Mapping[str, Any]) -> str:
    """Render markdown with the artifact block first for automated parsing."""

    validate_artifact(artifact)
    source_line = ", ".join(f"arXiv:{source}" for source in CITATIONS_VERIFIED)
    return f"""# Action-effect SOTA ingestion .422 - 2026-06-22

```json
{_artifact_json(artifact)}
```

Reliable channel only: `scripts/sweep_clusters.py`, `scripts/sweep_semscholar.py`,
arXiv abs-page HTTP-200 checks, and low-concurrency WebSearch/WebFetch of seven
top action-effect / clickability / verifier-guided candidate expansion sources.
Preconditions passed before any claim was promoted:
`.venv/bin/python scripts/sweep_clusters.py --help` exited zero and
`curl -sf -o /dev/null https://export.arxiv.org/api/query?search_query=all:test`
confirmed arXiv API reachability. `scripts/sweep_clusters.py 5 --max-results 8`
and `scripts/sweep_clusters.py 6 --max-results 8` emitted the focused
action-effect and learned-search cluster URLs. `scripts/sweep_semscholar.py` ran
two focused queries and returned HTTP 429 on both, so no S2-only claim was
promoted. No `/deep-research` call was made. No training, live LLM inference,
leaderboard submission, or live solve was launched. No ops/status/traceability
files or `scripts/research_conductor.py` were modified.

Already-discovered corpus read through a learned action-effect / clickability /
exploration-efficiency and verifier-guided candidate expansion filter:
`research-studying.md`, `research-references.md`,
`openspec/capabilities/arc-human-replay-frame-change/spec.md` at
`REQ-ARC-FCP-4568`, `openspec/capabilities/capstone/spec.md` at
`REQ-CAPSTONE-4569`, `results/experiment_4568_clickability_action_effect_predictor.json`,
and `results/experiment_4569_verifier_guided_expansion.json`. Exp 4568 was an
honest clickability null: the local predictor learned a positive control but did
not reduce held-out actions because it reranked a fixed candidate pool. Exp 4569
was also a no-value/null or negative transfer for verifier-guided candidate
expansion: candidate generation stayed the residual bottleneck, the
random-priority control did not pass, and the winner was still absent from most
frontiers.

Sources checked: {source_line}. The WebSearch/WebFetch pass also checked the
StochasticGoose/Tufa implementation source at https://github.com/DriesSmit/ARC3-solution.

## Per-Method Mapping

- **StochasticGoose-style learned frame-change clickability predictor**
  (arXiv:2603.24621): the strongest A1 lesson is not to keep Exp 4568 as a
  final sorter. The action-effect model must decide which action families and
  click locations enter the search frontier before first-contact actions are
  spent. It fails when measured only as frame-change accuracy or when it prunes
  rare necessary actions without a positive-control and recall guard.
- **AgentRM generalizable reward model for agent search** (arXiv:2502.18407,
  arXiv:2502.00271): take over Exp 4569's expansion priority by scoring partial
  trajectories for test-time beam/frontier control. The Scaling Flaws caution
  requires reversible or fallback pruning, repeated-sampling controls, and
  held-out games for transfer claims.
- **ThinkPRM generative process verifier** (arXiv:2504.16828,
  arXiv:2502.00271): use a long-CoT process verifier only as the sparse,
  expensive check on ambiguous high-upside branches inside Exp 4569. It fails
  if verification cost eats the first-contact action budget or local step
  plausibility is mistaken for final progress.
- **adaptive PRM-guided candidate expansion / best-first scheduling**
  (arXiv:2602.01070,
  arXiv:2502.00271): promote verifier scores from reranking into online budget
  allocation: widen, deepen, prune, or fall back based on process-reward
  aggregates. It fails if it spends more compute without lowering actions to
  first level-up or silently inherits verifier-guided-search scaling flaws.
- **Self-evolving verifiable-reward data refresh** (arXiv:2601.22607,
  arXiv:2502.00271): refresh both the Exp 4568 action-effect prior and Exp 4569
  verifier with executable checked traces and hard negatives from verifier
  pruning failures. It fails if generated traces leak game identity, checkers
  are not execution-tested, or the loop optimizes the verifier rather than
  held-out action efficiency.

## .423 Candidate

{FLAGGED_FOR_NEXT_ROADMAP}

The practical next experiment should keep the exp4568 positive-control guard,
the exp4569 random-priority/repeated-sampling controls, and the offline
reproduction gate. The implementation change is to move learned clickability
upstream into branch generation, then let adaptive PRM-guided candidate expansion
allocate frontier budget over those branches. AgentRM supplies the
agent-search reward-model precedent, ThinkPRM supplies the expensive
explanation-quality checker for ambiguous branches, self-evolving
verifiable-reward data supplies a refresh stream, and Scaling Flaws supplies the
guardrail: no learned verifier may be the sole irreversible branch killer.
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
        "action-effect / clickability",
        "verifier-guided candidate expansion",
        "Exp 4568",
        "Exp 4569",
        "StochasticGoose",
        "AgentRM",
        "ThinkPRM",
        "Scaling Flaws",
        "adaptive PRM-guided candidate expansion",
        "self-evolving verifiable-reward",
        "flagged_for_v423",
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
## 2026-06-22 Exp 4577 - .422 action-effect SOTA ingestion - INGESTED

**Status:** INGESTED into `docs/research-notes/sota-ingestion-action-effect-422-2026-06-22.md`
and `results/experiment_4577_sota_ingestion_action_effect.json`.

**Preconditions:** `scripts/sweep_clusters.py --help` succeeded; the arXiv API
reachability check succeeded; `scripts/sweep_clusters.py` clusters 5 and 6
emitted focused URLs; `scripts/sweep_semscholar.py` returned HTTP 429 for both
focused queries, so no S2-only source was promoted. Top sources were verified
by arXiv abs-page HTTP 200 and low-concurrency WebSearch/WebFetch of the six
arXiv sources plus the StochasticGoose implementation URL. `/deep-research` was
not invoked. No live solve, training run, live LLM inference, leaderboard
submission, ops/status/traceability edit, or `scripts/research_conductor.py`
edit occurred.

**Methods marked ingested:** StochasticGoose-style learned frame-change
clickability predictor (arXiv:2603.24621), AgentRM generalizable reward-model
search (arXiv:2502.18407), ThinkPRM generative process verifier
(arXiv:2504.16828), Scaling Flaws verifier-guided-search caution
(arXiv:2502.00271), adaptive PRM-guided best-first candidate expansion
(arXiv:2602.01070), and self-evolving verifiable-reward data refresh
(arXiv:2601.22607).

{FLAGGED_FOR_NEXT_ROADMAP}
{STUDYING_SECTION_END}"""


def update_research_studying(path: Path) -> None:
    """Insert or replace the Exp 4577 studying-queue section."""

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
    root = Path(os.environ.get("CARNOT_EXP4577_ROOT", "."))
    artifact = write_outputs(
        artifact_path=root / "results/experiment_4577_sota_ingestion_action_effect.json",
        note_path=root / RESEARCH_NOTE_RELATIVE_PATH,
        studying_path=root / "research-studying.md",
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
