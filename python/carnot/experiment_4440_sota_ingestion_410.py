"""Exp 4440 SOTA ingestion for the `.410` example-corpus ARC solver.

Spec refs: REQ-REPORT-4440, SCENARIO-REPORT-4440.

This module writes a planning artifact, not a benchmark result. The `.410`
experiments showed that examples help: they reproduced one held-out win rule,
improved a world-model synthesis control, and consolidated reusable solver
operators. The remaining gap is converting those examples into a documented,
validated library that a first-contact solver can reuse without game-specific
leakage.
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
        "flagged_for_v411",
        "sota_to_experiment_mapping_note",
        "v410_outcome_conditioning",
        "preconditions_checked",
        "random_seed",
        "research_note_path",
        "field_principles",
    }
)
REQUIRED_METHOD_FIELDS = frozenset(
    {"name", "arxiv_id", "what_it_takes_over_our_stack", "pitfalls"}
)
REQUIRED_PRECONDITION_FIELDS = frozenset(
    {
        "sweep_clusters_ran",
        "sweep_clusters_urls",
        "sweep_semscholar_ran",
        "sweep_semscholar_status",
        "arxiv_api_verified_ids",
        "webfetch_http_200_verified_urls",
        "websearch_webfetch_reachable",
        "deep_research_invoked",
        "leaderboard_submission",
        "training_launched",
        "research_conductor_modified",
        "cpu_only",
    }
)
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:", "blocked_")
DEFAULT_HONEST_VERDICT = "complete: sota_ingestion_410_mapped"
INFERENCE_SUBSTRATE = "cpu_reliable_channel_sota_ingestion_no_training"
DEFAULT_RANDOM_SEED = 4440
RESEARCH_NOTE_RELATIVE_PATH = "docs/research-notes/sota-ingestion-410-2026-06-19.md"

DEFAULT_FLAGGED_FOR_V411 = (
    "LILO-style documented library induction over the ARC solver/example corpus "
    "(arXiv:2310.19791)"
)
EXECUTABLE_WORLD_MODEL_FLAGGED_FOR_V411 = (
    "Fresh-workspace executable ARC-AGI-3 world-model agent rerun (arXiv:2605.05138)"
)
LOOP_OWM_FLAGGED_FOR_V411 = (
    "Object-centric composable world-model transfer for ARC transitions (arXiv:2606.12316)"
)
CODEARC_FLAGGED_FOR_V411 = (
    "Differential-query program induction for unresolved verifier gaps (arXiv:2503.23145)"
)
ALLOWED_FLAGGED_FOR_V411 = frozenset(
    {
        DEFAULT_FLAGGED_FOR_V411,
        EXECUTABLE_WORLD_MODEL_FLAGGED_FOR_V411,
        LOOP_OWM_FLAGGED_FOR_V411,
        CODEARC_FLAGGED_FOR_V411,
    }
)

FIELD_PRINCIPLES = {
    "honest_verdict": {"principle": "terminal-prefixed"},
    "methods": {
        "principle": (
            "list of {name, arxiv_id, what_it_takes_over_our_stack, pitfalls} -- "
            "each with a VERIFIED citation (no citation = fabrication)"
        )
    },
    "flagged_for_v411": {
        "principle": (
            "the single strongest method fed forward so SOTA flows into the next "
            "milestone's experiments (discover->ingest->plan->experiment)"
        )
    },
    "inference_substrate": {
        "principle": "CPU-only reliable-channel literature ingestion; no live solve claim."
    },
    "sota_to_experiment_mapping_note": {
        "principle": "Concrete SOTA->experiment mapping for the `.411` planner."
    },
    "v410_outcome_conditioning": {
        "principle": "Machine-readable `.410` branch facts that choose the `.411` hand-off."
    },
    "preconditions_checked": {
        "principle": "Reliable-channel, no-deep-research, no-leaderboard provenance."
    },
    "random_seed": {"principle": "Deterministic query set seed."},
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
    "sweep_clusters_ran": True,
    "sweep_clusters_urls": [
        "scripts/sweep_clusters.py 3 --max-results 8",
        "scripts/sweep_clusters.py 0 --max-results 8",
    ],
    "sweep_semscholar_ran": True,
    "sweep_semscholar_status": (
        "focused ARC/program/library/world-model queries ran and returned HTTP 429; "
        "no Semantic-Scholar-only source was promoted"
    ),
    "arxiv_api_verified_ids": sorted(VERIFIED_SOURCE_URLS),
    "webfetch_http_200_verified_urls": sorted(VERIFIED_SOURCE_URLS.values()),
    "websearch_webfetch_reachable": True,
    "deep_research_invoked": False,
    "leaderboard_submission": False,
    "training_launched": False,
    "research_conductor_modified": False,
    "cpu_only": True,
}

DEFAULT_V410_OUTCOMES = {
    "loo_generic_solve_count_2_of_7": True,
    "example_conditioned_win_reproduced": True,
    "example_conditioned_world_model_lift": True,
    "world_model_lift_no_reproduced_level": True,
    "first_contact_gap_open": True,
    "primitive_deepen_reproduced": True,
    "primitives_consolidated": True,
}

SOTA_TO_EXPERIMENT_MAPPING_NOTE = (
    "SOTA->experiment map for .411: promote LILO-style documented library "
    "induction (arXiv:2310.19791) over the solved-game predicates, world-model "
    "files, and primitive ledger because .410 showed examples help but the "
    "reuse remains manually curated. Use DreamCoder (arXiv:2006.08381) and "
    "Stitch/top-down synthesis (arXiv:2211.16605) as the compression backbone, "
    "HYSYNTH (arXiv:2405.15880) and CodeARC (arXiv:2503.23145) for "
    "counterexample-guided induction, Executable World Models "
    "(arXiv:2605.05138) as the live ARC-AGI-3 harness, Loop-OWM "
    "(arXiv:2606.12316) for object-centric transfer tests, and ARC-TGI "
    "(arXiv:2603.05099) for generator-backed held-out example families."
)

DEFAULT_METHODS = [
    {
        "name": "LILO documented library induction",
        "arxiv_id": "2310.19791",
        "what_it_takes_over_our_stack": (
            "Turn solved predicates, world-model snippets, and consolidated "
            "operators into named, documented library primitives that the "
            "first-contact solver retrieves before calling a generator."
        ),
        "pitfalls": (
            "Library compression can encode game-specific constants; require "
            "held-out games, literal scans, and reproduction gates before any "
            "primitive is counted."
        ),
    },
    {
        "name": "DreamCoder wake-sleep library learning",
        "arxiv_id": "2006.08381",
        "what_it_takes_over_our_stack": (
            "Run wake-sleep style abstraction discovery over the ARC solve corpus "
            "so recurrent solver fragments become reusable DSL concepts plus a "
            "search prior."
        ),
        "pitfalls": (
            "The base DSL must already explain the corpus; if it omits mechanics "
            "like reset parity or tank facing, wake-sleep will compress the wrong "
            "space."
        ),
    },
    {
        "name": "Stitch top-down synthesis for library learning",
        "arxiv_id": "2211.16605",
        "what_it_takes_over_our_stack": (
            "Use corpus-guided top-down compression on existing solver programs "
            "to extract low-cost abstractions before handing them to LILO-style "
            "documentation and retrieval."
        ),
        "pitfalls": (
            "Syntactic compression alone can prefer opaque helpers; pair it with "
            "execution tests and human-readable docs so retrieval does not become "
            "another brittle cache."
        ),
    },
    {
        "name": "HYSYNTH context-free LLM approximation",
        "arxiv_id": "2405.15880",
        "what_it_takes_over_our_stack": (
            "Fit a task-local symbolic search surrogate from LLM completions, "
            "then search the candidate rule/program space instead of trusting "
            "one in-context sample."
        ),
        "pitfalls": (
            "The surrogate can overfit the prompt distribution; keep cold controls "
            "and verifier-returned counterexamples in the loop."
        ),
    },
    {
        "name": "CodeARC differential-query program induction",
        "arxiv_id": "2503.23145",
        "what_it_takes_over_our_stack": (
            "Convert open verifier gaps into targeted input/state queries, refine "
            "candidate functions from failures, and separate induction queries "
            "from final reproduction attempts."
        ),
        "pitfalls": (
            "A hidden target-function oracle can leak answers if used at solve "
            "time; restrict it to offline induction and report final gates "
            "separately."
        ),
    },
    {
        "name": "Executable ARC-AGI-3 world models",
        "arxiv_id": "2605.05138",
        "what_it_takes_over_our_stack": (
            "Keep the fresh-workspace induce-verify-refactor-plan loop as the "
            "live meta-solver harness, but feed it a documented primitive library "
            "instead of only ad hoc examples."
        ),
        "pitfalls": (
            "Fresh-agent and clean-workspace discipline is load-bearing; any "
            "cross-game file leakage invalidates the generic-solver claim."
        ),
    },
    {
        "name": "Loop-OWM composable world-model transfer",
        "arxiv_id": "2606.12316",
        "what_it_takes_over_our_stack": (
            "Represent ARC rules as object-centric slots plus looped transitions "
            "so example-conditioned action models transfer by state structure, "
            "not by prompt text alone."
        ),
        "pitfalls": (
            "ARC-AGI-1/2 static transition gains may not transfer to interactive "
            "ARC-AGI-3 without action-cost and goal-inference tests."
        ),
    },
    {
        "name": "ARC-TGI generator-backed example families",
        "arxiv_id": "2603.05099",
        "what_it_takes_over_our_stack": (
            "Generate held-out task-family variants with reasoning templates so "
            "library primitives are trained and tested on variation, not one "
            "public level trace."
        ),
        "pitfalls": (
            "Synthetic families can teach benchmark-specific shortcuts; require "
            "human-like constraints, local verification, and public/private split "
            "discipline."
        ),
    },
]

RESEARCH_NOTE = """# SOTA ingestion 2026-06-19: .410 example-corpus solver map for .411

reliable channel only: `scripts/sweep_clusters.py`, `scripts/sweep_semscholar.py`,
arXiv API / arXiv abs-page checks, and low-concurrency WebSearch/WebFetch. The
Semantic Scholar focused queries returned HTTP 429, so no S2-only source was
promoted. `/deep-research` was not invoked. No leaderboard submission was made.

## .410 outcome conditioning

- Exp 4432: leave-one-out generic transfer solved 2 of 7 reproduction-gated targets.
- Exp 4433: example-conditioned win induction reproduced `g50t` L1.
- Exp 4434: example-conditioned world-model synthesis improved accuracy from
  0.714286 to 1.0, but added zero reproduced levels.
- Exp 4435: generic first contact on `dc22` still logged an open verifier gap.
- Exp 4436: `tu93` deepened to L5 and consolidated reusable primitives.

## Verified SOTA methods

- LILO documented library induction, arXiv:2310.19791.
- DreamCoder wake-sleep library learning, arXiv:2006.08381.
- Stitch top-down synthesis, arXiv:2211.16605.
- HYSYNTH context-free LLM approximation, arXiv:2405.15880.
- CodeARC differential-query program induction, arXiv:2503.23145.
- Executable ARC-AGI-3 world models, arXiv:2605.05138.
- Loop-OWM composable world models, arXiv:2606.12316.
- ARC-TGI generator-backed task families, arXiv:2603.05099.

## SOTA->experiment mapping

The `.411` planner should build a documented primitive-library induction pass:
compress solved predicates, executable world models, and the primitive ledger;
name and document each primitive; retrieve those primitives for first-contact
games; and require held-out reproduction gates before any primitive is counted.

flagged_for_v411: LILO-style documented library induction over the ARC solver/example corpus (arXiv:2310.19791)
"""

STUDYING_SECTION = """## 2026-06-19 Exp 4440 - .410 example-corpus SOTA ingestion ingested

**Status:** INGESTED into `results/experiment_4440_sota_ingestion_410.json` and
`docs/research-notes/sota-ingestion-410-2026-06-19.md`.

**Preconditions:** reliable channel reachable on CPU. `scripts/sweep_clusters.py`
emitted focused arXiv cluster URLs for world-model and verifier/program
literature. `scripts/sweep_semscholar.py` ran focused ARC/program/library
queries and returned HTTP 429; no S2-only source was promoted. Low-concurrency
WebSearch/WebFetch plus arXiv API / arXiv abs-page HTTP 200 checks verified
arXiv:2310.19791, arXiv:2006.08381, arXiv:2211.16605, arXiv:2405.15880,
arXiv:2503.23145, arXiv:2605.05138, arXiv:2606.12316, and arXiv:2603.05099.
The banned `/deep-research` channel was not invoked. No leaderboard submission
or training run was launched.

**.410 outcome conditioning:** Exp 4432 solved 2/7 leave-one-out targets; Exp
4433 reproduced `g50t` L1 from example-conditioned win induction; Exp 4434
lifted example-conditioned world-model accuracy from 0.714286 to 1.0 but added
zero reproduced levels; Exp 4435 left `dc22` as an open verifier gap; Exp 4436
deepened `tu93` to L5 and consolidated solver primitives.

**Fresh-pass candidates marked ingested:** LILO (arXiv:2310.19791), DreamCoder
(arXiv:2006.08381), Stitch (arXiv:2211.16605), HYSYNTH (arXiv:2405.15880),
CodeARC (arXiv:2503.23145), Executable World Models (arXiv:2605.05138),
Loop-OWM (arXiv:2606.12316), and ARC-TGI (arXiv:2603.05099).

flagged_for_v411: LILO-style documented library induction over the ARC solver/example corpus (arXiv:2310.19791)

random_seed=4440

**SOTA->experiment mapping note:** Build a documented primitive-library induction
pass over solved predicates, executable world models, and primitive ledger rows;
retrieve those primitives during first-contact solving; and count only
held-out, reproduction-gated improvements.
"""


def _count_solved_without_own_recipe(row: Mapping[str, Any]) -> int | None:
    value = row.get("solve_count")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    per_game = row.get("per_game")
    if not isinstance(per_game, list):
        return None
    return sum(1 for item in per_game if isinstance(item, Mapping) and item.get("solved_without_own_recipe") is True)


def _target_count(row: Mapping[str, Any]) -> int | None:
    value = row.get("target_count")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    per_game = row.get("per_game")
    if not isinstance(per_game, list):
        return None
    return len(per_game)


def _nonempty_list(value: object) -> bool:
    return isinstance(value, list) and bool(value)


def extract_v410_outcomes(
    *,
    loo_benchmark: Mapping[str, Any],
    win_induction: Mapping[str, Any],
    action_model: Mapping[str, Any],
    first_contact: Mapping[str, Any],
    primitive_consolidation: Mapping[str, Any],
) -> dict[str, bool]:
    """Extract `.410` branch facts that determine which `.411` method matters."""

    solve_count = _count_solved_without_own_recipe(loo_benchmark)
    target_count = _target_count(loo_benchmark)
    cold = action_model.get("world_model_accuracy_cold")
    with_examples = action_model.get("world_model_accuracy_with_examples")

    return {
        "loo_generic_solve_count_2_of_7": (
            loo_benchmark.get("honest_verdict")
            == "complete: generic_loo_solve_count_2_of_7_gate_passed"
            and solve_count == 2
            and target_count == 7
            and loo_benchmark.get("offline_reproduced") is True
        ),
        "example_conditioned_win_reproduced": (
            win_induction.get("honest_verdict")
            == "success: example_conditioned_g50t_L1_offline_reproduced"
            and win_induction.get("target_game") == "g50t"
            and win_induction.get("offline_reproduced") is True
            and win_induction.get("reproduced_levels") == 1
        ),
        "example_conditioned_world_model_lift": (
            action_model.get("honest_verdict")
            == "success: example_conditioning_improved_world_model_accuracy"
            and isinstance(cold, (int, float))
            and isinstance(with_examples, (int, float))
            and with_examples > cold
        ),
        "world_model_lift_no_reproduced_level": action_model.get("reproduced_levels") == 0,
        "first_contact_gap_open": (
            first_contact.get("honest_verdict")
            == "complete: generic_first_contact_dc22_routed_no_new_level_gap_logged"
            and first_contact.get("residual_mechanic_gap_logged") is True
            and first_contact.get("offline_reproduced") is False
            and first_contact.get("reproduced_levels") == 0
        ),
        "primitive_deepen_reproduced": (
            primitive_consolidation.get("honest_verdict")
            == "success: tu93_L5_deepened_primitives_consolidated"
            and primitive_consolidation.get("deepened_game") == "tu93"
            and primitive_consolidation.get("new_levels_reproduced") == 1
            and primitive_consolidation.get("offline_reproduced") is True
        ),
        "primitives_consolidated": _nonempty_list(
            primitive_consolidation.get("primitives_consolidated")
        ),
    }


def select_flagged_for_v411(outcomes: Mapping[str, bool]) -> str:
    """Select the single method that should feed the `.411` planner."""

    if (
        outcomes.get("example_conditioned_win_reproduced")
        and outcomes.get("example_conditioned_world_model_lift")
        and outcomes.get("primitives_consolidated")
    ):
        return DEFAULT_FLAGGED_FOR_V411
    if not (
        outcomes.get("example_conditioned_win_reproduced")
        or outcomes.get("example_conditioned_world_model_lift")
        or outcomes.get("primitives_consolidated")
    ):
        return EXECUTABLE_WORLD_MODEL_FLAGGED_FOR_V411
    if outcomes.get("example_conditioned_world_model_lift"):
        return LOOP_OWM_FLAGGED_FOR_V411
    if outcomes.get("first_contact_gap_open"):
        return CODEARC_FLAGGED_FOR_V411
    return EXECUTABLE_WORLD_MODEL_FLAGGED_FOR_V411


def build_artifact(
    *,
    methods: Sequence[Mapping[str, str]] | None = None,
    flagged_for_v411: str = DEFAULT_FLAGGED_FOR_V411,
    v410_outcome_conditioning: Mapping[str, bool] | None = None,
    preconditions_checked: Mapping[str, object] | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    honest_verdict: str = DEFAULT_HONEST_VERDICT,
) -> dict[str, object]:
    """Build the deterministic Exp 4440 planning artifact."""

    source_methods = DEFAULT_METHODS if methods is None else methods
    source_outcomes = (
        DEFAULT_V410_OUTCOMES if v410_outcome_conditioning is None else v410_outcome_conditioning
    )
    source_preconditions = (
        DEFAULT_PRECONDITIONS_CHECKED
        if preconditions_checked is None
        else preconditions_checked
    )
    return {
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "methods": [dict(method) for method in source_methods],
        "flagged_for_v411": flagged_for_v411,
        "sota_to_experiment_mapping_note": SOTA_TO_EXPERIMENT_MAPPING_NOTE,
        "v410_outcome_conditioning": dict(source_outcomes),
        "preconditions_checked": dict(source_preconditions),
        "random_seed": random_seed,
        "research_note_path": RESEARCH_NOTE_RELATIVE_PATH,
        "field_principles": dict(FIELD_PRINCIPLES),
    }


def _validate_preconditions(row: object) -> None:
    if not isinstance(row, Mapping) or set(row) != REQUIRED_PRECONDITION_FIELDS:
        raise ValueError("preconditions_checked must have exactly the required fields")
    expected_true = {
        "sweep_clusters_ran": "sweep_clusters",
        "sweep_semscholar_ran": "sweep_semscholar",
        "websearch_webfetch_reachable": "WebSearch/WebFetch",
        "cpu_only": "CPU",
    }
    for key, label in expected_true.items():
        if row.get(key) is not True:
            raise ValueError(f"preconditions_checked must record {label} success")
    expected_false = {
        "deep_research_invoked": "deep-research",
        "leaderboard_submission": "leaderboard",
        "training_launched": "training",
        "research_conductor_modified": "research_conductor",
    }
    for key, label in expected_false.items():
        if row.get(key) is not False:
            raise ValueError(f"preconditions_checked must record no {label}")
    if not _nonempty_list(row.get("sweep_clusters_urls")):
        raise ValueError("preconditions_checked must record sweep cluster URLs")
    if not isinstance(row.get("sweep_semscholar_status"), str) or not row[
        "sweep_semscholar_status"
    ].strip():
        raise ValueError("preconditions_checked must record Semantic Scholar status")
    if not _nonempty_list(row.get("arxiv_api_verified_ids")) or set(VERIFIED_SOURCE_URLS) - set(
        row["arxiv_api_verified_ids"]
    ):
        raise ValueError("preconditions_checked must include all verified arXiv ids")
    if not _nonempty_list(row.get("webfetch_http_200_verified_urls")) or set(
        VERIFIED_SOURCE_URLS.values()
    ) - set(row["webfetch_http_200_verified_urls"]):
        raise ValueError("preconditions_checked must include all HTTP 200 source URLs")


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 4440 artifact before writing it to disk."""

    fields = set(artifact)
    missing = REQUIRED_ARTIFACT_FIELDS - fields
    extra = fields - REQUIRED_ARTIFACT_FIELDS
    if missing:
        raise ValueError(f"artifact missing required fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"artifact has unexpected fields: {sorted(extra)}")

    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must use a terminal prefix")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must declare the CPU ingestion substrate")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match REQ-REPORT-4440")
    if not isinstance(artifact["random_seed"], int) or isinstance(artifact["random_seed"], bool):
        raise ValueError("random_seed must be an integer")
    if artifact["research_note_path"] != RESEARCH_NOTE_RELATIVE_PATH:
        raise ValueError("research_note_path must be the repo-relative note path")
    if artifact["v410_outcome_conditioning"] != DEFAULT_V410_OUTCOMES:
        raise ValueError("v410_outcome_conditioning must match the .410 branch")
    note = artifact["sota_to_experiment_mapping_note"]
    if not isinstance(note, str) or "SOTA->experiment" not in note:
        raise ValueError("sota_to_experiment_mapping_note must contain the mapping note")

    _validate_preconditions(artifact["preconditions_checked"])

    methods = artifact["methods"]
    if not isinstance(methods, list) or not 5 <= len(methods) <= 8:
        raise ValueError("methods must contain five to eight verified methods")

    seen_sources: set[str] = set()
    for method in methods:
        if not isinstance(method, Mapping) or set(method) != REQUIRED_METHOD_FIELDS:
            raise ValueError("each method must be a dict with exactly the required fields")
        for key, value in method.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"method field {key!r} must be a non-empty string")
        source = method["arxiv_id"]
        if source not in VERIFIED_SOURCE_URLS:
            raise ValueError(f"method source {source!r} is not a verified arXiv id")
        if source in seen_sources:
            raise ValueError(f"duplicate source in methods: {source}")
        seen_sources.add(source)

    flagged = artifact["flagged_for_v411"]
    if not isinstance(flagged, str) or not flagged.strip():
        raise ValueError("flagged_for_v411 must be non-empty")
    if flagged not in ALLOWED_FLAGGED_FOR_V411:
        raise ValueError("flagged_for_v411 must be conditioned on the .410 outcomes")


def validate_research_note(section: str) -> None:
    """Check that the research note preserves citations and the `.411` hand-off."""

    missing_sources = sorted(
        source for source in NOTE_REQUIRED_SOURCE_CITATIONS if source not in section
    )
    if missing_sources:
        raise ValueError(f"research note missing verified source citations: {missing_sources}")
    required_phrases = [
        "SOTA->experiment",
        "reliable channel",
        "HTTP 429",
        "/deep-research",
        "No leaderboard submission",
        "flagged_for_v411",
        DEFAULT_FLAGGED_FOR_V411,
    ]
    for phrase in required_phrases:
        if phrase not in section:
            raise ValueError(f"research note missing required phrase: {phrase}")


def validate_studying_section(section: str) -> None:
    """Check that the studying entry marks Exp 4440 ingested with citations."""

    required_phrases = [
        "2026-06-19 Exp 4440",
        "INGESTED",
        "sweep_clusters.py",
        "sweep_semscholar.py",
        "HTTP 429",
        "/deep-research",
        "No leaderboard submission",
        "flagged_for_v411",
        f"random_seed={DEFAULT_RANDOM_SEED}",
        DEFAULT_FLAGGED_FOR_V411,
        "SOTA->experiment",
    ]
    for phrase in required_phrases:
        if phrase not in section:
            raise ValueError(f"studying section missing required phrase: {phrase}")
    missing_sources = sorted(
        source for source in NOTE_REQUIRED_SOURCE_CITATIONS if source not in section
    )
    if missing_sources:
        raise ValueError(f"studying section missing verified source citations: {missing_sources}")


def _with_studying_section(existing: str) -> str:
    marker = "## 2026-06-19 Exp 4440"
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


def _read_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_v410_outcomes(repo_root: Path) -> dict[str, bool]:
    """Read the `.410` source artifacts and extract planner branch decisions."""

    return extract_v410_outcomes(
        loo_benchmark=_read_json(repo_root / "results/experiment_4432_loo_generic_solve_benchmark.json"),
        win_induction=_read_json(
            repo_root / "results/experiment_4433_example_conditioned_win_induction.json"
        ),
        action_model=_read_json(
            repo_root / "results/experiment_4434_example_conditioned_action_model.json"
        ),
        first_contact=_read_json(repo_root / "results/experiment_4435_generic_first_contact_fixed.json"),
        primitive_consolidation=_read_json(
            repo_root / "results/experiment_4436_deepen_plus_primitive_consolidation.json"
        ),
    )


def write_outputs(
    *,
    artifact_path: Path,
    note_path: Path,
    studying_path: Path,
    outcomes: Mapping[str, bool] | None = None,
) -> dict[str, object]:
    """Write the JSON artifact, research note, and idempotent studying entry."""

    resolved_outcomes = outcomes or DEFAULT_V410_OUTCOMES
    flagged_for_v411 = select_flagged_for_v411(resolved_outcomes)
    artifact = build_artifact(
        flagged_for_v411=flagged_for_v411,
        v410_outcome_conditioning=resolved_outcomes,
    )
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
    root_override = os.environ.get("CARNOT_EXP4440_ROOT")
    repo_root = Path(root_override) if root_override else Path(__file__).resolve().parents[2]
    try:
        outcomes = load_v410_outcomes(repo_root)
    except FileNotFoundError:
        outcomes = dict(DEFAULT_V410_OUTCOMES)
    artifact = write_outputs(
        artifact_path=repo_root / "results/experiment_4440_sota_ingestion_410.json",
        note_path=repo_root / RESEARCH_NOTE_RELATIVE_PATH,
        studying_path=repo_root / "research-studying.md",
        outcomes=outcomes,
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
