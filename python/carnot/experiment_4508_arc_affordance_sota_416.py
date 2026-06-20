"""Exp 4508 ARC affordance/action-effect SOTA ingestion for the `.417` hand-off.

Spec refs: REQ-REPORT-4508, SCENARIO-REPORT-4508.

This module records a research-planning artifact. It does not train the
frame-change predictor, run a live ARC solve, or submit to the leaderboard. The
artifact is deterministic so the markdown note and result JSON can be compared
directly by tests and by the conductor reconciler.
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
        "methods",
        "frame_change_mapping",
        "energy_augmented_mapping",
        "strongest_for_v417",
        "research_note_path",
        "random_seed",
        "field_principles",
    }
)
REQUIRED_METHOD_FIELDS = frozenset(
    {"name", "arxiv_id", "mapped_application", "stack_mapping", "pitfall"}
)
REQUIRED_PRECONDITION_FIELDS = frozenset(
    {
        "agents_md_read",
        "codex_md_read",
        "research_studying_filtered",
        "research_references_filtered",
        "frame_change_notes_read",
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
APPLICATION_KEYS = frozenset(
    {
        "GAP-ARCH-AFFORDANCE-PRUNING",
        "GAP-ARCH-FRAME-CHANGE-PREDICTOR",
        "GAP-ARCH-ACTION-EFFECT-REPRESENTATION",
        "GAP-ARCH-FRONTIER-EXPLORATION",
        "GAP-ARCH-ENERGY-PROGRESS-SHAPING",
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
DEFAULT_HONEST_VERDICT = "complete: arc_affordance_sota_416_mapped_for_v417"
DEFAULT_RANDOM_SEED = 4508
RESEARCH_NOTE_RELATIVE_PATH = "docs/research-notes/arc-affordance-sota-416.md"

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "MUST start with terminal prefix "
        "complete:/complete_/success:/success_/passed:/passed_/shipped:/shipped_ "
        "(Verdict Terminal-Prefix Discipline)."
    ),
    "inference_substrate": (
        "explicit substrate so adversarial_verify applies the right duration floor."
    ),
    "preconditions_checked": (
        "records WHICH resources were verified; pre-empts silent-missing-resource "
        "fabrication."
    ),
    "source_ids": "only arXiv IDs whose abs pages were HTTP-200 checked may anchor the mapping.",
    "methods": "each source maps to a concrete ARC action-efficiency decision and caveat.",
    "frame_change_mapping": (
        "maps affordance/action-effect evidence onto frame-change predictor pruning."
    ),
    "energy_augmented_mapping": (
        "maps progress/potential evidence onto P(frame_change) * (-delta_E) ranking."
    ),
    "strongest_for_v417": "names the single strongest next hand-off for .417.",
    "research_note_path": "repo-relative markdown path for deterministic parsing.",
    "random_seed": "bare integer seed for reproducible artifact generation.",
    "field_principles": "principle annotations for every top-level artifact field.",
}

VERIFIED_SOURCE_URLS = {
    "2006.15085": "https://arxiv.org/abs/2006.15085",
    "2008.09241": "https://arxiv.org/abs/2008.09241",
    "2404.15648": "https://arxiv.org/abs/2404.15648",
    "2407.10341": "https://arxiv.org/abs/2407.10341",
    "2501.06047": "https://arxiv.org/abs/2501.06047",
    "2601.07060": "https://arxiv.org/abs/2601.07060",
    "2602.00460": "https://arxiv.org/abs/2602.00460",
    "2602.03201": "https://arxiv.org/abs/2602.03201",
}
DEFAULT_SOURCE_IDS = list(VERIFIED_SOURCE_URLS)
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(
    f"arXiv:{source}" for source in VERIFIED_SOURCE_URLS
)

FRAME_CHANGE_NOTES = [
    "docs/research-notes/arc-frame-change-predictor-spec.md",
    "docs/research-notes/arc-energy-augmented-strategy.md",
    "docs/research-notes/arc-417-shaping-action-efficiency.md",
    "docs/research-notes/arc-imitation-sota-415.md",
]

DEFAULT_STRONGEST_FOR_V417 = (
    "flagged_for_v417: affordance-pruned frame-change predictor with "
    "SLOPE-style optimistic energy progress shaping, anchored by "
    "arXiv:2008.09241, arXiv:2006.15085, and arXiv:2602.03201"
)

DEFAULT_PRECONDITIONS_CHECKED = {
    "agents_md_read": True,
    "codex_md_read": True,
    "research_studying_filtered": True,
    "research_references_filtered": True,
    "frame_change_notes_read": FRAME_CHANGE_NOTES,
    "sweep_clusters_help_succeeded": True,
    "sweep_clusters_urls": [
        (
            "http://export.arxiv.org/api/query?search_query="
            '(abs:"active+inference"+OR+abs:"free+energy"+OR+'
            'abs:"free+energy+principle"+OR+abs:"predictive+coding"+OR+'
            'abs:"world+model")+AND+'
            '(abs:"LLM"+OR+abs:"language+model"+OR+abs:"reasoning")'
            "&start=0&max_results=8&sortBy=submittedDate&sortOrder=descending"
        ),
        (
            "http://export.arxiv.org/api/query?search_query="
            '(abs:"verifier+ensemble"+OR+abs:"verifier+ensembles"+OR+'
            'abs:"null+space"+OR+abs:"specification+gaming"+OR+'
            'abs:"process+reward+model"+OR+abs:"deliberative+alignment"+OR+'
            'abs:"reward+hacking")&start=0&max_results=8'
            "&sortBy=submittedDate&sortOrder=descending"
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

FRAME_CHANGE_MAPPING = {
    "target": "affordance-pruned frame-change predictor",
    "source_ids": ["2006.15085", "2008.09241", "2501.06047", "2404.15648"],
    "candidate_policy": (
        "Predict which action/click cells are feasible and likely to change the "
        "frame, then prune low-affordance no-op candidates before BFS expansion."
    ),
    "training_signal": (
        "Use human replay and cached explorer transitions as action-effect labels: "
        "frame_delta, click cell, action_id, and object/contact region."
    ),
    "acceptance_gate": (
        "reduce median actions-to-first-levelup without reducing held-out solve-rate"
    ),
}

ENERGY_AUGMENTED_MAPPING = {
    "target": "energy-augmented ranking for action efficiency",
    "source_ids": ["2407.10341", "2601.07060", "2602.00460", "2602.03201"],
    "ranking_formula": "P(frame_change) * (-delta_E)",
    "energy_policy": (
        "Use structural objective energy as a progress potential so the explorer "
        "prefers changes that look goal-consistent, not merely non-zero."
    ),
    "caveat": (
        "energy only helps if computed over structural features; frame-marginal "
        "energy must stay a null until transfer is proven."
    ),
}

DEFAULT_METHODS = [
    {
        "name": "What can I do here? A Theory of Affordances in Reinforcement Learning",
        "arxiv_id": "2006.15085",
        "mapped_application": "GAP-ARCH-AFFORDANCE-PRUNING",
        "stack_mapping": (
            "Treat affordances as a learned feasible-action mask that reduces the "
            "branching factor before transition/value scoring."
        ),
        "pitfall": (
            "availability is not progress; the mask must be paired with energy or "
            "level-progress checks before solve claims."
        ),
    },
    {
        "name": "Learning Affordance Landscapes for Interaction Exploration in 3D Environments",
        "arxiv_id": "2008.09241",
        "mapped_application": "GAP-ARCH-FRAME-CHANGE-PREDICTOR",
        "stack_mapping": (
            "Port the image-region-to-action-success idea into an ARC click "
            "heatmap plus ACTION1-5 frame-change head."
        ),
        "pitfall": (
            "the source setting is RGB-D 3D exploration; ARC use must stay "
            "frame-only and cannot read hidden environment state."
        ),
    },
    {
        "name": "Cross-Embodied Affordance Transfer through Learning Affordance Equivalences",
        "arxiv_id": "2404.15648",
        "mapped_application": "GAP-ARCH-ACTION-EFFECT-REPRESENTATION",
        "stack_mapping": (
            "Use object/action/effect triples as the representation target for "
            "cached ARC action-effect examples."
        ),
        "pitfall": (
            "robot trajectory transfer does not directly imply discrete ARC action "
            "transfer; it only motivates the representation."
        ),
    },
    {
        "name": "Affordance-Guided Reinforcement Learning via Visual Prompting",
        "arxiv_id": "2407.10341",
        "mapped_application": "GAP-ARCH-ENERGY-PROGRESS-SHAPING",
        "stack_mapping": (
            "Take the dense affordance-shaped reward pattern, but replace VLM "
            "keypoint rewards with local structural energy progress."
        ),
        "pitfall": (
            "a live VLM reward is not an ARC competition substrate; this artifact "
            "uses only the shaping pattern."
        ),
    },
    {
        "name": "Learning Affordances from Interactive Exploration using an Object-level Map",
        "arxiv_id": "2501.06047",
        "mapped_application": "GAP-ARCH-FRAME-CHANGE-PREDICTOR",
        "stack_mapping": (
            "Track object instances across views so repeated ARC transitions "
            "produce denser action-effect labels instead of isolated pixels."
        ),
        "pitfall": (
            "object mapping must be deterministic from frames; no game internals "
            "or private state can enter the labels."
        ),
    },
    {
        "name": "PALM: Progress-Aware Policy Learning via Affordance Reasoning",
        "arxiv_id": "2601.07060",
        "mapped_application": "GAP-ARCH-ENERGY-PROGRESS-SHAPING",
        "stack_mapping": (
            "Use progress-aware affordance reasoning to separate repeated "
            "frame-changing actions from actions that advance a subtask."
        ),
        "pitfall": (
            "PALM is a VLA manipulation stack; Carnot should borrow progress cues, "
            "not the model substrate."
        ),
    },
    {
        "name": "Search Inspired Exploration in Reinforcement Learning",
        "arxiv_id": "2602.00460",
        "mapped_application": "GAP-ARCH-FRONTIER-EXPLORATION",
        "stack_mapping": (
            "Select frontier state-action pairs by cost-to-come/cost-to-go and "
            "learning progress instead of flat repeated BFS expansion."
        ),
        "pitfall": (
            "online RL frontier growth is not a banked ARC solve; use the idea as "
            "a pruning and ordering policy over cached candidates."
        ),
    },
    {
        "name": "SLOPE: Optimistic Potential Landscape Shaping for Model-based Reinforcement Learning",
        "arxiv_id": "2602.03201",
        "mapped_application": "GAP-ARCH-ENERGY-PROGRESS-SHAPING",
        "stack_mapping": (
            "Map optimistic potential landscapes onto Carnot's objective energy "
            "term so sparse level-progress can guide search earlier."
        ),
        "pitfall": (
            "optimistic learned rewards can overstate progress; structural energy "
            "and solve-rate guards must stay authoritative."
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
    """Build the deterministic artifact embedded in the markdown note."""

    chosen_source_ids = DEFAULT_SOURCE_IDS if source_ids is None else list(source_ids)
    chosen_methods = DEFAULT_METHODS if methods is None else methods
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
        "methods": [dict(method) for method in chosen_methods],
        "frame_change_mapping": dict(FRAME_CHANGE_MAPPING),
        "energy_augmented_mapping": dict(ENERGY_AUGMENTED_MAPPING),
        "strongest_for_v417": DEFAULT_STRONGEST_FOR_V417,
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

    _require(
        row.get("frame_change_notes_read") == FRAME_CHANGE_NOTES,
        "preconditions_checked must record frame-change and energy note resources",
    )
    _require(
        _nonempty_list(row.get("sweep_clusters_urls")),
        "preconditions_checked must record cluster URLs",
    )
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
        "field_principles must match REQ-REPORT-4508",
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
        _require(
            method["mapped_application"] in APPLICATION_KEYS,
            "method mapped_application must name a required ARC application",
        )
        method_sources.append(source)
    _require(method_sources == source_ids, "methods arxiv ids must match source_ids")

    frame_mapping = artifact["frame_change_mapping"]
    _require(
        isinstance(frame_mapping, Mapping),
        "frame_change_mapping must be a mapping",
    )
    _require(
        set(frame_mapping.get("source_ids", [])) == set(FRAME_CHANGE_MAPPING["source_ids"]),
        "frame_change_mapping must preserve verified frame-change sources",
    )
    _require(
        "prune" in str(frame_mapping.get("candidate_policy", "")).lower(),
        "frame_change_mapping must describe candidate pruning",
    )

    energy_mapping = artifact["energy_augmented_mapping"]
    _require(
        isinstance(energy_mapping, Mapping),
        "energy_augmented_mapping must be a mapping",
    )
    _require(
        energy_mapping.get("ranking_formula") == "P(frame_change) * (-delta_E)",
        "energy_augmented_mapping ranking_formula must be P(frame_change) * (-delta_E)",
    )
    _require(
        set(energy_mapping.get("source_ids", []))
        == set(ENERGY_AUGMENTED_MAPPING["source_ids"]),
        "energy_augmented_mapping must preserve verified energy/progress sources",
    )

    _require(
        artifact["strongest_for_v417"] == DEFAULT_STRONGEST_FOR_V417,
        "strongest_for_v417 must name the verified .417 package",
    )
    _validate_preconditions(artifact["preconditions_checked"])


def _artifact_json(artifact: Mapping[str, Any]) -> str:
    return json.dumps(artifact, indent=2, sort_keys=True)


def render_research_note(artifact: Mapping[str, Any]) -> str:
    """Render markdown with the artifact block first for automated parsing."""

    validate_artifact(artifact)
    source_line = ", ".join(f"arXiv:{source}" for source in DEFAULT_SOURCE_IDS)
    return f"""# ARC affordance/action-effect SOTA ingestion .416 - 2026-06-20

```json
{_artifact_json(artifact)}
```

Reliable channel only: `research-studying.md`, `research-references.md`,
`scripts/sweep_clusters.py`, ARC frame-change/energy strategy notes, arXiv
abs-page HTTP-200 checks, and low-concurrency WebSearch/WebFetch of the top
eight affordance learning, action-effect model, and sparse-reward exploration
sources. `.venv/bin/python scripts/sweep_clusters.py --help` succeeded.
`scripts/sweep_clusters.py 3 --max-results 8` and
`scripts/sweep_clusters.py 0 --max-results 8` emitted the focused fresh-pass
cluster URLs. No `/deep-research` call was made. No training, live LLM
inference, leaderboard submission, or live solve was launched. No
ops/status/traceability files were modified.

Sources checked: {source_line}.

## Focused Finding

The direct fit for `.417` is an affordance-pruned frame-change predictor:
learn which action/click cells are feasible and likely to change the frame,
then prune low-affordance candidates before the explorer spends actions on
them. Affordance theory (arXiv:2006.15085) supports the branch-factor reduction;
interaction-exploration affordance landscapes (arXiv:2008.09241) give the
closest CNN-style image-region-to-action-success template; object-map
interactive exploration (arXiv:2501.06047) and action-effect-object
representations (arXiv:2404.15648) sharpen the label representation.

## Energy-Augmented Ranking

The Carnot-specific graft is energy-augmented ranking, not a pure copy of an
affordance classifier. Use affordance probability to remove likely no-ops, then
rank survivors by `P(frame_change) * (-delta_E)`. KAGI (arXiv:2407.10341) and
PALM (arXiv:2601.07060) support progress-aware affordance shaping; SIERL
(arXiv:2602.00460) supports frontier selection by learning progress and
cost-to-go; SLOPE (arXiv:2602.03201) is the strongest energy analogue because
it replaces flat sparse rewards with an optimistic potential landscape.

## SOTA->Experiment Mapping

- Frame-change predictor: train an action-effect model over frame/action/click
  labels and use it to prune no-op candidates before BFS expansion.
- Energy-augmented ranking: score remaining candidates by
  `P(frame_change) * (-delta_E)` where `delta_E` comes only from structural
  objective-energy features.
- Frontier policy: choose state-action frontiers that are reachable, not fully
  exhausted, and promising under the energy-progress term.

For `.417`, the strongest hand-off is the affordance-pruned frame-change
predictor with SLOPE-style optimistic energy progress shaping. Treat this as a
planning artifact only: `inference_substrate=aggregation_from_upstream_artifacts`.

{DEFAULT_STRONGEST_FOR_V417}
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
        "/deep-research",
        "No training",
        "affordance-pruned frame-change predictor",
        "action-effect",
        "energy-augmented ranking",
        "P(frame_change) * (-delta_E)",
        "SLOPE",
        "flagged_for_v417",
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
    root = Path(os.environ.get("CARNOT_EXP4508_ROOT", "."))
    artifact = write_outputs(
        artifact_path=root / "results/experiment_4508_arc_affordance_sota_416.json",
        note_path=root / RESEARCH_NOTE_RELATIVE_PATH,
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
