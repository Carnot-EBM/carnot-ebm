"""Exp 4477 SOTA ingestion for the `.413` precision hand-off.

Spec refs: REQ-REPORT-4477, SCENARIO-REPORT-4477.

This module records a planning artifact, not a live solve. It preserves the
reliable-channel literature pass and hands one method to `.414` without
implying that Carnot ran a leaderboard submission or trained a model.
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
        "methods",
        "flagged_for_v414",
        "sota_to_experiment_mapping_note",
        "preconditions_checked",
        "random_seed",
        "research_note_path",
        "field_principles",
    }
)
REQUIRED_METHOD_FIELDS = frozenset({"name", "arxiv_id", "what_it_takes_over_our_stack", "pitfalls"})
REQUIRED_PRECONDITION_FIELDS = frozenset(
    {
        "sweep_clusters_help_succeeded",
        "arxiv_reachable",
        "sweep_clusters_ran",
        "sweep_clusters_urls",
        "sweep_semscholar_ran",
        "sweep_semscholar_status",
        "top_abstracts_webfetched",
        "arxiv_http_200_verified_ids",
        "arxiv_http_200_verified_urls",
        "websearch_webfetch_reachable",
        "deep_research_invoked",
        "leaderboard_submission",
        "training_launched",
        "live_solve_claim",
        "research_conductor_modified",
        "cpu_only",
    }
)
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")
DEFAULT_HONEST_VERDICT = "complete: sota_ingestion_413_mapped_for_v414"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
DEFAULT_RANDOM_SEED = 4477
RESEARCH_NOTE_RELATIVE_PATH = "docs/research-notes/sota-ingestion-413-2026-06-20.md"
DEFAULT_FLAGGED_FOR_V414 = (
    "Socrates-style multiple-choice query synthesis for GAP-5 "
    "demo-underdetermination (arXiv:2604.08792)"
)

FIELD_PRINCIPLES = {
    "honest_verdict": {"principle": "terminal-prefixed"},
    "methods": {
        "principle": (
            "list of {name, arxiv_id, what_it_takes_over_our_stack, pitfalls} -- "
            "each with a VERIFIED citation (no citation = fabrication)"
        )
    },
    "flagged_for_v414": {
        "principle": (
            "the single strongest method fed forward so SOTA flows into the next "
            "milestone (discover->ingest->plan->experiment)"
        )
    },
    "inference_substrate": {
        "principle": (
            "aggregation_from_upstream_artifacts -- CPU-only reliable-channel "
            "ingestion; no live solve claim"
        )
    },
    "sota_to_experiment_mapping_note": {
        "principle": "Concrete SOTA->experiment mapping for the `.414` planner."
    },
    "preconditions_checked": {
        "principle": "Reliable-channel, no-deep-research, no-leaderboard provenance."
    },
    "random_seed": {"principle": "Deterministic focused sweep seed."},
    "research_note_path": {"principle": "Repo-relative SOTA mapping note emitted with artifact."},
}

VERIFIED_SOURCE_URLS = {
    "2606.11521": "https://arxiv.org/abs/2606.11521",
    "2605.27051": "https://arxiv.org/abs/2605.27051",
    "2604.08792": "https://arxiv.org/abs/2604.08792",
    "2307.03966": "https://arxiv.org/abs/2307.03966",
    "2604.02434": "https://arxiv.org/abs/2604.02434",
    "2605.05138": "https://arxiv.org/abs/2605.05138",
    "2606.12316": "https://arxiv.org/abs/2606.12316",
    "2512.24156": "https://arxiv.org/abs/2512.24156",
}
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(f"arXiv:{source}" for source in VERIFIED_SOURCE_URLS)

DEFAULT_PRECONDITIONS_CHECKED = {
    "sweep_clusters_help_succeeded": True,
    "arxiv_reachable": True,
    "sweep_clusters_ran": True,
    "sweep_clusters_urls": [
        "scripts/sweep_clusters.py 0 --max-results 8",
        "scripts/sweep_clusters.py 3 --max-results 8",
    ],
    "sweep_semscholar_ran": True,
    "sweep_semscholar_status": (
        "five focused queries ran; Semantic Scholar returned six unique arXiv IDs "
        "and HTTP 429 on two queries, so no S2-only non-arXiv source was promoted"
    ),
    "top_abstracts_webfetched": sorted(VERIFIED_SOURCE_URLS.values()),
    "arxiv_http_200_verified_ids": sorted(VERIFIED_SOURCE_URLS),
    "arxiv_http_200_verified_urls": sorted(VERIFIED_SOURCE_URLS.values()),
    "websearch_webfetch_reachable": True,
    "deep_research_invoked": False,
    "leaderboard_submission": False,
    "training_launched": False,
    "live_solve_claim": False,
    "research_conductor_modified": False,
    "cpu_only": True,
}

SOTA_TO_EXPERIMENT_MAPPING_NOTE = (
    "SOTA->experiment map for .414: promote Socrates-style multiple-choice "
    "query synthesis for program disambiguation (arXiv:2604.08792) as the "
    "strongest handoff, because .413 already executed the CEGIS lever and "
    "GAP-5 demo-underdetermination is now the load-bearing precision frontier. "
    "Turn sibling inputs, verifier-rejected states, and candidate program "
    "clusters into discriminating queries; accept a program only when the "
    "execution verifier, cross-example consistency, and the disambiguation "
    "query agree, otherwise abstain. Keep CGL (arXiv:2606.11521) and ConVer "
    "(arXiv:2605.27051) as the refute->repair loop templates, use multi-intent "
    "detection (arXiv:2307.03966) for the ambiguity tripwire, use "
    "compositional neuro-symbolic consistency filtering (arXiv:2604.02434) for "
    "agreement-gated precision, and keep Executable World Models "
    "(arXiv:2605.05138), Loop-OWM (arXiv:2606.12316), and graph exploration "
    "(arXiv:2512.24156) as the ARC-AGI-3 world-model and meta-solver baselines."
)

DEFAULT_METHODS = [
    {
        "name": "Counterexample Guided Learning in the Large using Reasoning Agents",
        "arxiv_id": "2606.11521",
        "what_it_takes_over_our_stack": (
            "Keep the .413 CEGIS loop as a reusable primitive: every rejected "
            "rule must emit a compact execution counterexample, a failure "
            "cluster, and a re-induction prompt for the next operator attempt."
        ),
        "pitfalls": (
            "The reported domain is regex induction, not ARC-AGI-3 games; use it "
            "as a feedback protocol only, and keep all solve claims behind the "
            "offline reproduction gate."
        ),
    },
    {
        "name": "ConVer contract and loop-invariant CEGAR-CEGIS verification",
        "arxiv_id": "2605.27051",
        "what_it_takes_over_our_stack": (
            "Lift the verifier from single predicate checks to compositional "
            "contracts: generate a candidate rule contract, check it against the "
            "execution trace, and refine only the failed contract fragment."
        ),
        "pitfalls": (
            "C programs with assertions differ from grid games; over-specific "
            "contracts can launder per-game constants unless held-out variants "
            "and literal scans stay in the gate."
        ),
    },
    {
        "name": "Choose, Don't Label multiple-choice query synthesis",
        "arxiv_id": "2604.08792",
        "what_it_takes_over_our_stack": (
            "Convert GAP-5 into an active disambiguation step: synthesize a "
            "small set of candidate behaviors over sibling inputs or replayable "
            "states, then require the execution verifier to choose or abstain."
        ),
        "pitfalls": (
            "The original loop assumes a human can answer high-level queries; "
            "Carnot must replace that answerer with replayable env evidence, or "
            "mark the rule underdetermined instead of guessing."
        ),
    },
    {
        "name": "Multi-intent detection for programming-by-example ambiguity",
        "arxiv_id": "2307.03966",
        "what_it_takes_over_our_stack": (
            "Add a pre-acceptance ambiguity tripwire that detects whether the "
            "demo examples admit multiple structurally distinct programs before "
            "the agreement gate promotes a candidate."
        ),
        "pitfalls": (
            "The paper targets string/data-mapping PBE; ARC demos need object, "
            "palette, and action-state features rather than only input-output "
            "string properties."
        ),
    },
    {
        "name": "Compositional neuro-symbolic cross-example consistency filtering",
        "arxiv_id": "2604.02434",
        "what_it_takes_over_our_stack": (
            "Use object-level representations plus symbolic cross-example "
            "consistency as the precision filter for independent induced rules "
            "before any candidate reaches the reproduction gate."
        ),
        "pitfalls": (
            "ARC-AGI-2 static grids are not interactive games; the filter must "
            "also inspect action transitions and sibling-input disagreement for "
            "GAP-5."
        ),
    },
    {
        "name": "Executable World Models for ARC-AGI-3",
        "arxiv_id": "2605.05138",
        "what_it_takes_over_our_stack": (
            "Keep the executable Python world model as the phase-FSM substrate "
            "for sc25-style operators and for replayable disambiguation queries."
        ),
        "pitfalls": (
            "Published agent results are not Carnot evidence; fresh workspaces, "
            "leakage audits, and no-leaderboard discipline remain mandatory."
        ),
    },
    {
        "name": "Loop-OWM object-centric composable world models",
        "arxiv_id": "2606.12316",
        "what_it_takes_over_our_stack": (
            "Represent cast-grid, color-match, and sprite-resize mechanics as "
            "slots plus looped transitions, so .414 can test rule transfer by "
            "state structure rather than prompt similarity."
        ),
        "pitfalls": (
            "ARC-1/2 visual-symbolic transitions do not include action cost or "
            "goal discovery; integrate with executable replay before treating it "
            "as an ARC-AGI-3 method."
        ),
    },
    {
        "name": "Graph-Based Exploration for ARC-AGI-3 Interactive Reasoning",
        "arxiv_id": "2512.24156",
        "what_it_takes_over_our_stack": (
            "Use explicit state-action graphs as the baseline meta-solver and as "
            "the source of discriminating untested transitions for GAP-5 queries."
        ),
        "pitfalls": (
            "The paper reports a training-free exploration baseline, not a rule "
            "induction verifier; it can guide query selection but cannot certify "
            "a learned rule by itself."
        ),
    },
]

RESEARCH_NOTE = """# SOTA ingestion 2026-06-20: .413 precision and underdetermination map for .414

Reliable channel only: `scripts/sweep_clusters.py`, `scripts/sweep_semscholar.py`,
arXiv abs-page HTTP-200 checks, and low-concurrency WebSearch/WebFetch of the
top abstracts. `.venv/bin/python scripts/sweep_clusters.py --help` succeeded
and arXiv was reachable. `scripts/sweep_clusters.py` emitted focused verifier
and world-model cluster URLs. `scripts/sweep_semscholar.py` ran five focused
queries; Semantic Scholar returned six unique arXiv IDs and HTTP 429 on two
queries, so no S2-only non-arXiv source was promoted. `/deep-research` was not
invoked. No leaderboard submission was made. No live solve or training run was
launched.

## .413 outcome conditioning

Exp 4467 banked dc22 through counterexample-guided config-rule grounding,
Exp 4468 moved sc25 L2-L5 from provisional to reproduced, Exp 4469 banked a
generic sc25 cast-grid phase-FSM L1 operator, and Exp 4470 banked sb26. Exp
4474 kept the GAP-4 execution-verifier regression guard green. The remaining
frontier is not "try CEGIS"; it is program-induction precision, agreement
acceptance, and GAP-5 demo-underdetermination.

## Focused sweep result

- Counterexample Guided Learning in the Large using Reasoning Agents,
  arXiv:2606.11521, remains the clean feedback-loop template for rejected
  executable rules.
- ConVer contract and loop-invariant CEGAR-CEGIS verification, arXiv:2605.27051,
  supplies a scalable generate-check-refine contract pattern.
- Choose, Don't Label, arXiv:2604.08792, is the strongest `.414` method: turn
  ambiguous programs into multiple-choice discriminating behaviors instead of
  trusting a demo-perfect but underdetermined rule.
- Multi-Intent Detection in PBE, arXiv:2307.03966, gives a direct ambiguity
  detector precedent for examples that admit multiple intents.
- Compositional Neuro-Symbolic Reasoning, arXiv:2604.02434, maps to
  cross-example consistency filtering before a candidate reaches the gate.
- Executable World Models for ARC-AGI-3, arXiv:2605.05138, remains the
  replayable phase-FSM world-model substrate.
- Loop-OWM, arXiv:2606.12316, supplies object-centric slot/transition structure.
- Graph-Based Exploration for ARC-AGI-3, arXiv:2512.24156, is the explicit
  state-action graph baseline and source of untested transition queries.

## SOTA->experiment mapping

The `.414` planner should build a GAP-5-aware tiered acceptance harness:
induce multiple candidate programs, execute them on all demos and sibling
inputs, synthesize a Socrates-style discriminating behavior when programs agree
on the target but diverge elsewhere, and abstain when the executable evidence
cannot resolve the ambiguity. Then apply cross-example consistency and exact
replay before promoting the candidate. This feeds the open re86
pattern-match/sprite-resize gap and future manufactured variants without
claiming a live solve.

flagged_for_v414: Socrates-style multiple-choice query synthesis for GAP-5 demo-underdetermination (arXiv:2604.08792)
"""

STUDYING_SECTION = """## 2026-06-20 Exp 4477 - .413 SOTA ingestion ingested

**Status:** INGESTED into `results/experiment_4477_sota_ingestion_413.json` and
`docs/research-notes/sota-ingestion-413-2026-06-20.md`.

**Preconditions:** reliable channel reachable on CPU. The command
`.venv/bin/python scripts/sweep_clusters.py --help` succeeded; the arXiv
reachability check succeeded. `scripts/sweep_clusters.py` emitted focused
verifier and world-model cluster URLs. `scripts/sweep_semscholar.py` ran five
focused queries; Semantic Scholar returned six unique arXiv IDs and HTTP 429 on
two queries, so no S2-only non-arXiv source was promoted. Low-concurrency
WebSearch/WebFetch plus arXiv abs-page HTTP 200 checks verified
arXiv:2606.11521, arXiv:2605.27051, arXiv:2604.08792, arXiv:2307.03966,
arXiv:2604.02434, arXiv:2605.05138, arXiv:2606.12316, and arXiv:2512.24156.
The banned `/deep-research` channel was not invoked. No leaderboard submission,
live solve, or training run was launched.

**.413 outcome conditioning:** Exp 4467 banked dc22, Exp 4468 banked sc25 L2-L5,
Exp 4469 banked generic sc25 cast-grid L1, Exp 4470 banked sb26, and Exp 4474
kept the GAP-4 regression guard green. GAP-5 demo-underdetermination remains
the program-induction precision frontier for `.414`.

**Fresh-pass candidates marked ingested:** Counterexample Guided Learning
(arXiv:2606.11521), ConVer CEGAR-CEGIS verification (arXiv:2605.27051),
Choose, Don't Label program disambiguation (arXiv:2604.08792), PBE multi-intent
detection (arXiv:2307.03966), compositional neuro-symbolic consistency filtering
(arXiv:2604.02434), Executable World Models (arXiv:2605.05138), Loop-OWM
(arXiv:2606.12316), and graph-based ARC-AGI-3 exploration (arXiv:2512.24156).

flagged_for_v414: Socrates-style multiple-choice query synthesis for GAP-5 demo-underdetermination (arXiv:2604.08792)

random_seed=4477

**SOTA->experiment mapping note:** Build a GAP-5-aware tiered acceptance
harness: independent program induction plus cross-example consistency; when
programs agree on one target but diverge on sibling inputs, synthesize a
discriminating query and accept only if replayable executable evidence resolves
the ambiguity. Otherwise abstain.
"""


def _fail(message: str) -> None:
    raise ValueError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _nonempty_list(value: object) -> bool:
    return isinstance(value, list) and bool(value)


def build_artifact(
    *,
    methods: Sequence[Mapping[str, str]] | None = None,
    preconditions_checked: Mapping[str, object] | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    honest_verdict: str = DEFAULT_HONEST_VERDICT,
    flagged_for_v414: str = DEFAULT_FLAGGED_FOR_V414,
) -> dict[str, object]:
    """Build the deterministic Exp 4477 planning artifact."""

    source_methods = DEFAULT_METHODS if methods is None else methods
    source_preconditions = (
        DEFAULT_PRECONDITIONS_CHECKED if preconditions_checked is None else preconditions_checked
    )
    return {
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "methods": [dict(method) for method in source_methods],
        "flagged_for_v414": flagged_for_v414,
        "sota_to_experiment_mapping_note": SOTA_TO_EXPERIMENT_MAPPING_NOTE,
        "preconditions_checked": dict(source_preconditions),
        "random_seed": random_seed,
        "research_note_path": RESEARCH_NOTE_RELATIVE_PATH,
        "field_principles": dict(FIELD_PRINCIPLES),
    }


def _validate_preconditions(row: object) -> None:
    _require(
        isinstance(row, Mapping) and set(row) == REQUIRED_PRECONDITION_FIELDS,
        "preconditions_checked must have exactly the required fields",
    )
    expected_true = {
        "sweep_clusters_help_succeeded": "sweep_clusters help",
        "arxiv_reachable": "arXiv reachability",
        "sweep_clusters_ran": "sweep_clusters",
        "sweep_semscholar_ran": "sweep_semscholar",
        "websearch_webfetch_reachable": "WebSearch/WebFetch",
        "cpu_only": "CPU",
    }
    for key, label in expected_true.items():
        _require(row.get(key) is True, f"preconditions_checked must record {label} success")
    expected_false = {
        "deep_research_invoked": "deep-research",
        "leaderboard_submission": "leaderboard",
        "training_launched": "training",
        "live_solve_claim": "live solve",
        "research_conductor_modified": "research_conductor",
    }
    for key, label in expected_false.items():
        _require(row.get(key) is False, f"preconditions_checked must record no {label}")
    _require(
        _nonempty_list(row.get("sweep_clusters_urls")),
        "preconditions_checked must record cluster URLs",
    )
    _require(
        isinstance(row.get("sweep_semscholar_status"), str)
        and bool(str(row["sweep_semscholar_status"]).strip()),
        "preconditions_checked must record Semantic Scholar status",
    )
    _require(
        _nonempty_list(row.get("top_abstracts_webfetched")),
        "preconditions_checked must record top abstracts fetched",
    )
    _require(
        _nonempty_list(row.get("arxiv_http_200_verified_ids"))
        and not (set(VERIFIED_SOURCE_URLS) - set(row["arxiv_http_200_verified_ids"])),
        "preconditions_checked must include all verified arXiv ids",
    )
    _require(
        _nonempty_list(row.get("arxiv_http_200_verified_urls"))
        and not (set(VERIFIED_SOURCE_URLS.values()) - set(row["arxiv_http_200_verified_urls"])),
        "preconditions_checked must include all HTTP 200 source URLs",
    )


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 4477 artifact before writing it to disk."""

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
        artifact["field_principles"] == FIELD_PRINCIPLES,
        "field_principles must match REQ-REPORT-4477",
    )
    _require(
        isinstance(artifact["random_seed"], int) and not isinstance(artifact["random_seed"], bool),
        "random_seed must be an integer",
    )
    _require(
        artifact["research_note_path"] == RESEARCH_NOTE_RELATIVE_PATH,
        "research_note_path must be the repo-relative note path",
    )
    note = artifact["sota_to_experiment_mapping_note"]
    _require(
        isinstance(note, str) and "SOTA->experiment" in note,
        "sota_to_experiment_mapping_note must contain the mapping note",
    )

    _validate_preconditions(artifact["preconditions_checked"])

    methods = artifact["methods"]
    _require(
        isinstance(methods, list) and 5 <= len(methods) <= 8,
        "methods must contain five to eight verified methods",
    )

    seen_sources: set[str] = set()
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
        _require(
            source in VERIFIED_SOURCE_URLS, f"method source {source!r} is not a verified arXiv id"
        )
        _require(source not in seen_sources, f"duplicate source in methods: {source}")
        seen_sources.add(source)

    flagged = artifact["flagged_for_v414"]
    _require(
        isinstance(flagged, str) and flagged == DEFAULT_FLAGGED_FOR_V414,
        "flagged_for_v414 must name the single strongest verified method",
    )


def validate_research_note(section: str) -> None:
    """Check that the research note preserves citations and the `.414` hand-off."""

    missing_sources = sorted(
        source for source in NOTE_REQUIRED_SOURCE_CITATIONS if source not in section
    )
    _require(
        not missing_sources, f"research note missing verified source citations: {missing_sources}"
    )
    required_phrases = [
        "SOTA->experiment",
        "Reliable channel",
        "HTTP 429",
        "/deep-research",
        "No leaderboard submission",
        "flagged_for_v414",
        DEFAULT_FLAGGED_FOR_V414,
    ]
    for phrase in required_phrases:
        _require(phrase in section, f"research note missing required phrase: {phrase}")


def validate_studying_section(section: str) -> None:
    """Check that the studying entry marks Exp 4477 ingested with citations."""

    required_phrases = [
        "2026-06-20 Exp 4477",
        "INGESTED",
        "sweep_clusters.py",
        "sweep_semscholar.py",
        "HTTP 429",
        "/deep-research",
        "No leaderboard submission",
        "flagged_for_v414",
        f"random_seed={DEFAULT_RANDOM_SEED}",
        DEFAULT_FLAGGED_FOR_V414,
        "SOTA->experiment",
    ]
    for phrase in required_phrases:
        _require(phrase in section, f"studying section missing required phrase: {phrase}")
    missing_sources = sorted(
        source for source in NOTE_REQUIRED_SOURCE_CITATIONS if source not in section
    )
    _require(
        not missing_sources,
        f"studying section missing verified source citations: {missing_sources}",
    )


def _with_studying_section(existing: str) -> str:
    marker = "## 2026-06-20 Exp 4477"
    next_marker = "\n## "
    section = STUDYING_SECTION.rstrip() + "\n"

    if marker in existing:
        start = existing.index(marker)
        next_start = existing.find(next_marker, start + 1)
        if next_start == -1:
            return existing[:start] + section
        return existing[:start] + section + existing[next_start:]

    if existing.startswith("## "):
        return section + "\n" + existing

    first_section = existing.find(next_marker)
    if first_section == -1:
        return existing.rstrip() + "\n\n" + section
    return existing[: first_section + 1] + section + "\n" + existing[first_section + 1 :]


def write_outputs(
    *,
    artifact_path: Path,
    note_path: Path,
    studying_path: Path,
) -> dict[str, object]:
    """Write the JSON artifact, research note, and idempotent studying entry."""

    artifact = build_artifact()
    validate_artifact(artifact)
    validate_research_note(RESEARCH_NOTE)
    validate_studying_section(STUDYING_SECTION)

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(RESEARCH_NOTE.rstrip() + "\n", encoding="utf-8")

    existing_studying = studying_path.read_text(encoding="utf-8") if studying_path.exists() else ""
    studying_path.write_text(_with_studying_section(existing_studying), encoding="utf-8")

    return artifact


def main() -> int:
    root_override = os.environ.get("CARNOT_EXP4477_ROOT")
    repo_root = Path(root_override) if root_override else Path(__file__).resolve().parents[2]
    artifact = write_outputs(
        artifact_path=repo_root / "results/experiment_4477_sota_ingestion_413.json",
        note_path=repo_root / RESEARCH_NOTE_RELATIVE_PATH,
        studying_path=repo_root / "research-studying.md",
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
