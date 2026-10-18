"""Exp 4464 SOTA ingestion for the `.412` counterexample-guided hand-off.

Spec refs: REQ-REPORT-4464, SCENARIO-REPORT-4464.

This module records a planning artifact, not a live solve. It preserves the
reliable-channel literature pass and hands one method to `.413` without
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
        "flagged_for_v413",
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
DEFAULT_HONEST_VERDICT = "complete: sota_ingestion_412_mapped_for_v413"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
DEFAULT_RANDOM_SEED = 4464
RESEARCH_NOTE_RELATIVE_PATH = "docs/research-notes/sota-ingestion-412-2026-06-19.md"
DEFAULT_FLAGGED_FOR_V413 = (
    "Counterexample-guided re-induction from rejecting execution states "
    "(arXiv:2606.11521; SMT-checked CEGIS predecessor arXiv:2309.16436)"
)

FIELD_PRINCIPLES = {
    "honest_verdict": {"principle": "terminal-prefixed"},
    "methods": {
        "principle": (
            "list of {name, arxiv_id, what_it_takes_over_our_stack, pitfalls} -- "
            "each with a VERIFIED citation (no citation = fabrication)"
        )
    },
    "flagged_for_v413": {
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
        "principle": "Concrete SOTA->experiment mapping for the `.413` planner."
    },
    "preconditions_checked": {
        "principle": "Reliable-channel, no-deep-research, no-leaderboard provenance."
    },
    "random_seed": {"principle": "Deterministic focused sweep seed."},
    "research_note_path": {"principle": "Repo-relative SOTA mapping note emitted with artifact."},
}

VERIFIED_SOURCE_URLS = {
    "2309.16436": "https://arxiv.org/abs/2309.16436",
    "2606.11521": "https://arxiv.org/abs/2606.11521",
    "2507.14172": "https://arxiv.org/abs/2507.14172",
    "2411.17708": "https://arxiv.org/abs/2411.17708",
    "2411.02272": "https://arxiv.org/abs/2411.02272",
    "2605.05138": "https://arxiv.org/abs/2605.05138",
    "2606.12316": "https://arxiv.org/abs/2606.12316",
    "2603.13372": "https://arxiv.org/abs/2603.13372",
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
        "five focused queries ran; Semantic Scholar returned HTTP 429 on all "
        "five queries, so no S2-only source was promoted"
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
    "SOTA->experiment map for .413: promote counterexample-guided re-induction "
    "from rejecting execution states (arXiv:2606.11521) as the strongest "
    "handoff, with SMT-checked CEGIS (arXiv:2309.16436) as the formal loop "
    "template. Feed verifier-rejected dc22, tr87, and sc25 execution states "
    "back into the generic operator prompt, cluster the failures, re-induce "
    "the rule, and accept only reproduction-gated fixes. Use SOAR "
    "(arXiv:2507.14172) and neurally-guided induction (arXiv:2411.17708) for "
    "search pressure, induction+transduction (arXiv:2411.02272) for routing, "
    "Executable World Models (arXiv:2605.05138) plus Loop-OWM "
    "(arXiv:2606.12316) for phase-FSM and object-state tests, and the ARC "
    "survey (arXiv:2603.13372) as the meta-solver context."
)

DEFAULT_METHODS = [
    {
        "name": "Counterexample-Guided Learning in the Large",
        "arxiv_id": "2606.11521",
        "what_it_takes_over_our_stack": (
            "Turn every verifier-rejected generic operator run into structured "
            "feedback: rejected state, expected predicate, observed predicate, "
            "and counterexample cluster. Re-prompt the inducer with that state "
            "until dc22, tr87, or sc25 reproduces under the offline env."
        ),
        "pitfalls": (
            "The paper studies regex induction, not ARC games; Carnot must "
            "translate execution failures into compact, replayable grid/game "
            "counterexamples and avoid treating feedback as a live solve claim."
        ),
    },
    {
        "name": "LLM plus SMT counterexample-guided inductive synthesis",
        "arxiv_id": "2309.16436",
        "what_it_takes_over_our_stack": (
            "Use exact execution checks or small SMT encodings as the verifier "
            "that returns counterexamples to the LLM learner before any generic "
            "rule is accepted."
        ),
        "pitfalls": (
            "Blocks-world planning is simpler than ARC-AGI-3; incomplete SMT "
            "encodings or hand-written constraints can launder game-specific "
            "knowledge into the induction loop."
        ),
    },
    {
        "name": "SOAR self-improving evolutionary ARC synthesis",
        "arxiv_id": "2507.14172",
        "what_it_takes_over_our_stack": (
            "Wrap the re-induction loop in evolutionary search and bank verified "
            "repair attempts as hindsight examples for future generic operators."
        ),
        "pitfalls": (
            "Fine-tuning and self-improvement are outside this CPU-only "
            "ingestion; any future training must separate held-out games and "
            "avoid replay leakage."
        ),
    },
    {
        "name": "Neurally-guided ARC program induction",
        "arxiv_id": "2411.17708",
        "what_it_takes_over_our_stack": (
            "Constrain tr87-style glyph rewrite synthesis to a small grid DSL "
            "and use a learned prior only to order executable candidates."
        ),
        "pitfalls": (
            "A narrow DSL can miss mechanics, while an unconstrained DSL can "
            "explode search; the verifier must remain the authority."
        ),
    },
    {
        "name": "Induction plus transduction routing for ARC",
        "arxiv_id": "2411.02272",
        "what_it_takes_over_our_stack": (
            "Route each open gap between precise induced programs and direct "
            "state/output prediction, then require the chosen path to reproduce "
            "through the same offline gate."
        ),
        "pitfalls": (
            "Static ARC input-output success does not prove interactive control; "
            "transductive shortcuts need separate action and state validation."
        ),
    },
    {
        "name": "Executable ARC-AGI-3 world models",
        "arxiv_id": "2605.05138",
        "what_it_takes_over_our_stack": (
            "Use the induce-verify-refactor-plan harness as the phase-FSM test "
            "bed for sc25 and other interactive games after a rule re-induces."
        ),
        "pitfalls": (
            "Fresh workspaces and leakage audits are load-bearing; closed-agent "
            "headline numbers are not evidence that Carnot solved anything."
        ),
    },
    {
        "name": "Loop-OWM object-centric transition loops",
        "arxiv_id": "2606.12316",
        "what_it_takes_over_our_stack": (
            "Represent cast-grid, toggle, and glyph mechanics as slots plus "
            "looped object-state transitions before translating them into "
            "generic world-model operators."
        ),
        "pitfalls": (
            "ARC-1/2 visual-symbolic transition prediction is not the same as "
            "interactive game planning; add goal, action-cost, and reproduction "
            "checks."
        ),
    },
    {
        "name": "ARC living survey meta-solver context",
        "arxiv_id": "2603.13372",
        "what_it_takes_over_our_stack": (
            "Use the survey's cross-paradigm map to keep .413 focused on "
            "refinement loops, program synthesis, and interactive transfer "
            "instead of another single-model prompt sweep."
        ),
        "pitfalls": (
            "A survey is context, not a runnable method; it must not be counted "
            "as experimental evidence or a leaderboard result."
        ),
    },
]

RESEARCH_NOTE = """# SOTA ingestion 2026-06-19: .412 counterexample-guided map for .413

Reliable channel only: `scripts/sweep_clusters.py`, `scripts/sweep_semscholar.py`,
arXiv abs-page HTTP-200 checks, and low-concurrency WebSearch/WebFetch of the
top abstracts. `scripts/sweep_clusters.py --help` succeeded and arXiv was
reachable. `scripts/sweep_clusters.py` emitted focused verifier and world-model
cluster URLs. `scripts/sweep_semscholar.py` ran five focused queries; Semantic
Scholar returned HTTP 429 on all five, so no S2-only source was promoted.
`/deep-research` was not invoked. No leaderboard submission was made. No live
solve or training run was launched.

## Focused sweep result

- Counterexample-Guided Learning in the Large, arXiv:2606.11521, is the
  freshest and strongest fit for `.413`: use verifier feedback as a structured
  counterexample, then re-induce from the rejecting execution state.
- Neuro-Symbolic Reasoning for Planning, arXiv:2309.16436, supplies the formal
  CEGIS loop: LLM learner, exact verifier, counterexample, revised candidate.
- SOAR self-improving evolutionary synthesis, arXiv:2507.14172, maps to
  evolutionary repair and hindsight banking after a generic operator fails.
- Towards Efficient Neurally-Guided Program Induction for ARC-AGI,
  arXiv:2411.17708, maps to ordered search over a compact glyph/grid DSL.
- Combining Induction and Transduction for Abstract Reasoning,
  arXiv:2411.02272, maps to routing between exact induced programs and direct
  state predictions.
- Executable World Models for ARC-AGI-3, arXiv:2605.05138, remains the
  interactive phase-FSM verification harness.
- Loop-OWM, arXiv:2606.12316, supplies object-centric slot and transition-loop
  structure for cast-grid and toggle mechanics.
- The ARC of Progress living survey, arXiv:2603.13372, anchors the meta-solver
  context and warns that refinement loops remain load-bearing across ARC
  versions.

## SOTA->experiment mapping

The `.413` planner should implement counterexample-guided re-induction as the
front door for the remaining generic-solver failures. When dc22, tr87, or sc25
rejects a proposed generic rule, record the exact execution state and failed
predicate, cluster related failures, re-prompt the inducer with that
counterexample, and accept only reproduction-gated fixes. SOAR and
neurally-guided induction provide search pressure; induction+transduction
provides routing; Executable World Models and Loop-OWM provide phase-FSM and
object-state verification targets.

flagged_for_v413: Counterexample-guided re-induction from rejecting execution states (arXiv:2606.11521; SMT-checked CEGIS predecessor arXiv:2309.16436)
"""

STUDYING_SECTION = """## 2026-06-19 Exp 4464 - .412 SOTA ingestion ingested

**Status:** INGESTED into `results/experiment_4464_sota_ingestion_412.json` and
`docs/research-notes/sota-ingestion-412-2026-06-19.md`.

**Preconditions:** reliable channel reachable on CPU. The command
`scripts/sweep_clusters.py --help` succeeded; the arXiv reachability check
succeeded. `scripts/sweep_clusters.py` emitted focused verifier and world-model
cluster URLs. `scripts/sweep_semscholar.py` ran five focused queries; Semantic
Scholar returned HTTP 429 on all five queries, so no S2-only source was
promoted. Low-concurrency WebSearch/WebFetch plus arXiv abs-page HTTP 200 checks
verified arXiv:2309.16436, arXiv:2606.11521, arXiv:2507.14172,
arXiv:2411.17708, arXiv:2411.02272, arXiv:2605.05138, arXiv:2606.12316, and
arXiv:2603.13372. The banned `/deep-research` channel was not invoked.
No leaderboard submission, live solve, or training run was launched.

**Fresh-pass candidates marked ingested:** Counterexample-Guided Learning
(arXiv:2606.11521), SMT-checked CEGIS (arXiv:2309.16436), SOAR
(arXiv:2507.14172), neurally-guided program induction (arXiv:2411.17708),
induction+transduction routing (arXiv:2411.02272), Executable World Models
(arXiv:2605.05138), Loop-OWM (arXiv:2606.12316), and ARC living survey context
(arXiv:2603.13372).

flagged_for_v413: Counterexample-guided re-induction from rejecting execution states (arXiv:2606.11521; SMT-checked CEGIS predecessor arXiv:2309.16436)

random_seed=4464

**SOTA->experiment mapping note:** Add a counterexample-guided re-induction
loop to the generic solver: feed verifier-rejected execution states back into
dc22 config/toggle induction, tr87 glyph-rewrite induction, and sc25 phase-FSM
world-model induction, then count only reproduction-gated fixes.
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
    flagged_for_v413: str = DEFAULT_FLAGGED_FOR_V413,
) -> dict[str, object]:
    """Build the deterministic Exp 4464 planning artifact."""

    source_methods = DEFAULT_METHODS if methods is None else methods
    source_preconditions = (
        DEFAULT_PRECONDITIONS_CHECKED if preconditions_checked is None else preconditions_checked
    )
    return {
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "methods": [dict(method) for method in source_methods],
        "flagged_for_v413": flagged_for_v413,
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
    """Validate the Exp 4464 artifact before writing it to disk."""

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
        "field_principles must match REQ-REPORT-4464",
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

    flagged = artifact["flagged_for_v413"]
    _require(
        isinstance(flagged, str) and flagged == DEFAULT_FLAGGED_FOR_V413,
        "flagged_for_v413 must name the single strongest verified method",
    )


def validate_research_note(section: str) -> None:
    """Check that the research note preserves citations and the `.413` hand-off."""

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
        "flagged_for_v413",
        DEFAULT_FLAGGED_FOR_V413,
    ]
    for phrase in required_phrases:
        _require(phrase in section, f"research note missing required phrase: {phrase}")


def validate_studying_section(section: str) -> None:
    """Check that the studying entry marks Exp 4464 ingested with citations."""

    required_phrases = [
        "2026-06-19 Exp 4464",
        "INGESTED",
        "sweep_clusters.py",
        "sweep_semscholar.py",
        "HTTP 429",
        "/deep-research",
        "No leaderboard submission",
        "flagged_for_v413",
        f"random_seed={DEFAULT_RANDOM_SEED}",
        DEFAULT_FLAGGED_FOR_V413,
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
    marker = "## 2026-06-19 Exp 4464"
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
    root_override = os.environ.get("CARNOT_EXP4464_ROOT")
    repo_root = Path(root_override) if root_override else Path(__file__).resolve().parents[2]
    artifact = write_outputs(
        artifact_path=repo_root / "results/experiment_4464_sota_ingestion_412.json",
        note_path=repo_root / RESEARCH_NOTE_RELATIVE_PATH,
        studying_path=repo_root / "research-studying.md",
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
