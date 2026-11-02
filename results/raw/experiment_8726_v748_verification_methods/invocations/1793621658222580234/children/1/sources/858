"""Exp 4452 SOTA ingestion for the `.411` ARC library-learning hand-off.

Spec refs: REQ-REPORT-4452, SCENARIO-REPORT-4452.

This module records a planning artifact, not a live solve. The point is to
preserve a reliable-channel literature pass and hand one method to `.412`
without implying that Carnot ran a leaderboard submission or trained a model.
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
        "flagged_for_v412",
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
DEFAULT_HONEST_VERDICT = "complete: sota_ingestion_411_mapped_for_v412"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
DEFAULT_RANDOM_SEED = 4452
RESEARCH_NOTE_RELATIVE_PATH = "docs/research-notes/sota-ingestion-411-2026-06-19.md"
DEFAULT_FLAGGED_FOR_V412 = (
    "LILO-style documented library induction over the ARC solver corpus (arXiv:2310.19791)"
)

FIELD_PRINCIPLES = {
    "honest_verdict": {"principle": "terminal-prefixed"},
    "methods": {
        "principle": (
            "list of {name, arxiv_id, what_it_takes_over_our_stack, pitfalls} -- "
            "each with a VERIFIED citation (no citation = fabrication)"
        )
    },
    "flagged_for_v412": {
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
        "principle": "Concrete SOTA->experiment mapping for the `.412` planner."
    },
    "preconditions_checked": {
        "principle": "Reliable-channel, no-deep-research, no-leaderboard provenance."
    },
    "random_seed": {"principle": "Deterministic focused sweep seed."},
    "research_note_path": {"principle": "Repo-relative SOTA mapping note emitted with artifact."},
}

VERIFIED_SOURCE_URLS = {
    "2310.19791": "https://arxiv.org/abs/2310.19791",
    "2006.08381": "https://arxiv.org/abs/2006.08381",
    "2211.16605": "https://arxiv.org/abs/2211.16605",
    "2405.15880": "https://arxiv.org/abs/2405.15880",
    "2503.23145": "https://arxiv.org/abs/2503.23145",
    "2605.05138": "https://arxiv.org/abs/2605.05138",
    "2606.12316": "https://arxiv.org/abs/2606.12316",
    "2603.05099": "https://arxiv.org/abs/2603.05099",
}
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(f"arXiv:{source}" for source in VERIFIED_SOURCE_URLS)

DEFAULT_PRECONDITIONS_CHECKED = {
    "sweep_clusters_help_succeeded": True,
    "arxiv_reachable": True,
    "sweep_clusters_ran": True,
    "sweep_clusters_urls": [
        "scripts/sweep_clusters.py 3 --max-results 8",
        "scripts/sweep_clusters.py 0 --max-results 8",
    ],
    "sweep_semscholar_ran": True,
    "sweep_semscholar_status": (
        "five focused queries ran; Semantic Scholar returned HTTP 429 on four "
        "queries and surfaced arXiv:2503.23145 plus adjacent non-promoted IDs"
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
    "SOTA->experiment map for .412: keep LILO-style documented library "
    "induction (arXiv:2310.19791) as the strongest hand-off because .411 needs "
    "generic reusable primitives, not another ad hoc first-contact solve. "
    "Compress solved predicates and world-model snippets with DreamCoder "
    "(arXiv:2006.08381) and Stitch (arXiv:2211.16605), document them LILO-style, "
    "stress the abstractions with HYSYNTH (arXiv:2405.15880) and CodeARC "
    "(arXiv:2503.23145) counterexamples, and validate transfer inside the "
    "Executable World Models harness (arXiv:2605.05138) with Loop-OWM "
    "(arXiv:2606.12316) object-state tests plus ARC-TGI (arXiv:2603.05099) "
    "held-out task-family variation."
)

DEFAULT_METHODS = [
    {
        "name": "LILO documented library induction",
        "arxiv_id": "2310.19791",
        "what_it_takes_over_our_stack": (
            "Compress the solved ARC predicates, world-model fragments, and "
            "primitive ledger into named, documented operators retrievable by "
            "the .412 planner before candidate generation."
        ),
        "pitfalls": (
            "Auto-discovered libraries can hide game constants; require literal "
            "scans, held-out games, and reproduction gates before credit."
        ),
    },
    {
        "name": "DreamCoder wake-sleep library learning",
        "arxiv_id": "2006.08381",
        "what_it_takes_over_our_stack": (
            "Use wake-sleep abstraction discovery as the search-prior backbone "
            "over reusable solver programs and replayed ARC tasks."
        ),
        "pitfalls": (
            "It only learns useful concepts if the DSL covers the real mechanic "
            "space; missing ARC primitives become compressed blind spots."
        ),
    },
    {
        "name": "Stitch top-down synthesis for library learning",
        "arxiv_id": "2211.16605",
        "what_it_takes_over_our_stack": (
            "Run corpus-guided compression over existing solver code to extract "
            "fast candidate abstractions before LILO documentation and retrieval."
        ),
        "pitfalls": (
            "Pure compression can produce opaque helpers; each abstraction needs "
            "docstrings, examples, and execution tests."
        ),
    },
    {
        "name": "HYSYNTH context-free LLM approximation",
        "arxiv_id": "2405.15880",
        "what_it_takes_over_our_stack": (
            "Fit task-local synthesis surrogates from LLM completions so the "
            "planner searches rule space instead of accepting one completion."
        ),
        "pitfalls": (
            "The surrogate can overfit prompt artifacts; keep verifier-returned "
            "counterexamples and cold controls in the loop."
        ),
    },
    {
        "name": "CodeARC differential-query program induction",
        "arxiv_id": "2503.23145",
        "what_it_takes_over_our_stack": (
            "Turn residual solver failures into targeted input/state queries and "
            "iteratively refine candidate predicates against differential tests."
        ),
        "pitfalls": (
            "A hidden target oracle is leakage if used at solve time; use it only "
            "for offline induction and separate final reproduction gates."
        ),
    },
    {
        "name": "Executable ARC-AGI-3 world models",
        "arxiv_id": "2605.05138",
        "what_it_takes_over_our_stack": (
            "Use the induce-verify-refactor-plan harness as the live transfer "
            "surface that consumes the documented primitive library."
        ),
        "pitfalls": (
            "Fresh agent and clean workspace isolation are load-bearing; cross-game "
            "file leakage invalidates generic transfer."
        ),
    },
    {
        "name": "Loop-OWM composable object-centric world-model transfer",
        "arxiv_id": "2606.12316",
        "what_it_takes_over_our_stack": (
            "Represent ARC rules as demonstration-conditioned object-state "
            "transitions to test whether primitives transfer by structure."
        ),
        "pitfalls": (
            "ARC-1/2 transition prediction does not prove interactive game control; "
            "add action-cost, goal-inference, and reproduction checks."
        ),
    },
    {
        "name": "ARC-TGI generator-backed held-out task families",
        "arxiv_id": "2603.05099",
        "what_it_takes_over_our_stack": (
            "Generate verified task-family variants so induced primitives face "
            "variation instead of memorizing one public trace."
        ),
        "pitfalls": (
            "Synthetic families can teach benchmark shortcuts; retain human-like "
            "constraints and public/private split discipline."
        ),
    },
]

RESEARCH_NOTE = """# SOTA ingestion 2026-06-19: .411 library-learning map for .412

Reliable channel only: `scripts/sweep_clusters.py`, `scripts/sweep_semscholar.py`,
arXiv abs-page HTTP-200 checks, and low-concurrency WebSearch/WebFetch of the
top abstracts. `scripts/sweep_clusters.py --help` succeeded and arXiv was
reachable. Semantic Scholar returned HTTP 429 on four of five focused queries;
the one successful counterexample-guided query surfaced CodeARC plus adjacent
non-promoted IDs. `/deep-research` was not invoked. No leaderboard submission
was made. No live solve or training run was launched.

## Focused sweep result

- LILO documented library induction, arXiv:2310.19791, is still the strongest
  fit for `.412`: synthesize, compress, and document reusable abstractions over
  the ARC solver corpus.
- DreamCoder wake-sleep abstraction discovery, arXiv:2006.08381, supplies the
  older generalizable library-learning backbone.
- Stitch top-down synthesis, arXiv:2211.16605, supplies the scalable compressor
  that LILO builds on.
- HYSYNTH context-free LLM approximation, arXiv:2405.15880, maps to task-local
  symbolic search from LLM completions.
- CodeARC differential-query induction, arXiv:2503.23145, maps to
  counterexample-led refinement of open verifier and predicate gaps.
- Executable World Models for ARC-AGI-3, arXiv:2605.05138, remains the harness
  for generic interactive-game transfer without game-specific prompts.
- Loop-OWM object-centric world-model transfer, arXiv:2606.12316, is the
  freshest direct ARC transfer paper and supplies object-state transition tests.
- ARC-TGI task-family generators, arXiv:2603.05099, supplies held-out
  variation so libraries are tested on rule families rather than one trace.

## SOTA->experiment mapping

The `.412` planner should run LILO-style documented primitive induction over the
solved ARC corpus: compress existing predicates/world-model snippets, generate
human-readable names and docstrings, retrieve those primitives before first
contact, and score only held-out reproduction-gated improvements. Loop-OWM and
Executable World Models provide the transfer evaluation substrate; CodeARC and
HYSYNTH provide counterexample-guided repair when a documented primitive fails.

flagged_for_v412: LILO-style documented library induction over the ARC solver corpus (arXiv:2310.19791)
"""

STUDYING_SECTION = """## 2026-06-19 Exp 4452 - .411 SOTA ingestion ingested

**Status:** INGESTED into `results/experiment_4452_sota_ingestion_411.json` and
`docs/research-notes/sota-ingestion-411-2026-06-19.md`.

**Preconditions:** reliable channel reachable on CPU. The command
`scripts/sweep_clusters.py --help` succeeded; the arXiv reachability check
succeeded. `scripts/sweep_clusters.py` emitted focused cluster URLs.
`scripts/sweep_semscholar.py` ran five focused queries; Semantic Scholar
returned HTTP 429 on four queries and surfaced CodeARC from the
counterexample-guided query. Low-concurrency WebSearch/WebFetch plus arXiv
abs-page HTTP 200 checks verified arXiv:2310.19791, arXiv:2006.08381,
arXiv:2211.16605, arXiv:2405.15880, arXiv:2503.23145, arXiv:2605.05138,
arXiv:2606.12316, and arXiv:2603.05099. The banned `/deep-research` channel was
not invoked. No leaderboard submission, live solve, or training run was
launched.

**Fresh-pass candidates marked ingested:** LILO (arXiv:2310.19791), DreamCoder
(arXiv:2006.08381), Stitch (arXiv:2211.16605), HYSYNTH (arXiv:2405.15880),
CodeARC (arXiv:2503.23145), Executable World Models (arXiv:2605.05138),
Loop-OWM (arXiv:2606.12316), and ARC-TGI (arXiv:2603.05099).

flagged_for_v412: LILO-style documented library induction over the ARC solver corpus (arXiv:2310.19791)

random_seed=4452

**SOTA->experiment mapping note:** Build a documented primitive-library
induction pass over solved predicates, executable world models, and primitive
ledger rows; retrieve those primitives during first-contact solving; and count
only held-out, reproduction-gated improvements.
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
    flagged_for_v412: str = DEFAULT_FLAGGED_FOR_V412,
) -> dict[str, object]:
    """Build the deterministic Exp 4452 planning artifact."""

    source_methods = DEFAULT_METHODS if methods is None else methods
    source_preconditions = (
        DEFAULT_PRECONDITIONS_CHECKED if preconditions_checked is None else preconditions_checked
    )
    return {
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "methods": [dict(method) for method in source_methods],
        "flagged_for_v412": flagged_for_v412,
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
    """Validate the Exp 4452 artifact before writing it to disk."""

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
        "field_principles must match REQ-REPORT-4452",
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

    flagged = artifact["flagged_for_v412"]
    _require(
        isinstance(flagged, str) and flagged == DEFAULT_FLAGGED_FOR_V412,
        "flagged_for_v412 must name the single strongest verified method",
    )


def validate_research_note(section: str) -> None:
    """Check that the research note preserves citations and the `.412` hand-off."""

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
        "flagged_for_v412",
        DEFAULT_FLAGGED_FOR_V412,
    ]
    for phrase in required_phrases:
        _require(phrase in section, f"research note missing required phrase: {phrase}")


def validate_studying_section(section: str) -> None:
    """Check that the studying entry marks Exp 4452 ingested with citations."""

    required_phrases = [
        "2026-06-19 Exp 4452",
        "INGESTED",
        "sweep_clusters.py",
        "sweep_semscholar.py",
        "HTTP 429",
        "/deep-research",
        "No leaderboard submission",
        "flagged_for_v412",
        f"random_seed={DEFAULT_RANDOM_SEED}",
        DEFAULT_FLAGGED_FOR_V412,
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
    marker = "## 2026-06-19 Exp 4452"
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
    root_override = os.environ.get("CARNOT_EXP4452_ROOT")
    repo_root = Path(root_override) if root_override else Path(__file__).resolve().parents[2]
    artifact = write_outputs(
        artifact_path=repo_root / "results/experiment_4452_sota_ingestion_411.json",
        note_path=repo_root / RESEARCH_NOTE_RELATIVE_PATH,
        studying_path=repo_root / "research-studying.md",
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
