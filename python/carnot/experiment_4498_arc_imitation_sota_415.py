"""Exp 4498 ARC imitation/replay SOTA ingestion for the `.416` hand-off.

Spec refs: REQ-REPORT-4498, SCENARIO-REPORT-4498.

This module records a research-planning artifact, not a training run or solve.
The provenance is an aggregation of existing repo notes, the focused sweep
helper, arXiv HTTP checks, and low-concurrency WebSearch/WebFetch sources.
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
        "human_corpus",
        "leaderboard_dqn_mapping",
        "arc_mapping",
        "strongest_for_v416",
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
        "arc_human_replay_notes_read",
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
        "GAP-ARCH-FRAME-CHANGE-PREDICTOR",
        "GAP-ARCH-VALUE-ENERGY-HEADS",
        "GAP-ARCH-EXPERT-INJECTION-REPLAY",
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
DEFAULT_HONEST_VERDICT = "complete: arc_imitation_sota_415_mapped_for_v416"
DEFAULT_RANDOM_SEED = 4498
RESEARCH_NOTE_RELATIVE_PATH = "docs/research-notes/arc-imitation-sota-415.md"

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "MUST start with terminal prefix "
        "complete:/complete_/success:/success_/passed:/passed_/shipped:/shipped_ "
        "(Verdict Terminal-Prefix Discipline)."
    ),
    "inference_substrate": (
        "explicit (live_llm_inference | verifier_ensemble_against_cached_candidates | "
        "aggregation_from_upstream_artifacts) so adversarial_verify applies the right "
        "duration floor."
    ),
    "preconditions_checked": (
        "records WHICH resources were verified; pre-empts silent-missing-resource "
        "fabrication."
    ),
    "source_ids": (
        "only arXiv IDs whose abs pages were HTTP-200 checked may anchor the "
        "SOTA mapping."
    ),
    "methods": (
        "each source must map to one concrete ARC training decision and one caveat."
    ),
    "human_corpus": (
        "public replay counts must stay bare facts, not inferred hidden-eval solve "
        "claims."
    ),
    "leaderboard_dqn_mapping": (
        "records the expert-injection mechanism separately from Carnot's own "
        "training status."
    ),
    "arc_mapping": (
        "maps literature to the actual queued ARC gaps, so follow-on work is "
        "actionable."
    ),
    "strongest_for_v416": (
        "names the single strongest next hand-off without implying it has already "
        "been trained."
    ),
    "research_note_path": "repo-relative markdown path for deterministic parsing.",
    "random_seed": "bare integer seed for reproducible artifact generation.",
    "field_principles": "principle annotations for every top-level artifact field.",
}

VERIFIED_SOURCE_URLS = {
    "1704.03732": "https://arxiv.org/abs/1704.03732",
    "1511.05952": "https://arxiv.org/abs/1511.05952",
    "2206.11795": "https://arxiv.org/abs/2206.11795",
    "2110.06169": "https://arxiv.org/abs/2110.06169",
    "1905.11108": "https://arxiv.org/abs/1905.11108",
    "2302.02948": "https://arxiv.org/abs/2302.02948",
    "2407.15007": "https://arxiv.org/abs/2407.15007",
    "2405.17476": "https://arxiv.org/abs/2405.17476",
}
DEFAULT_SOURCE_IDS = list(VERIFIED_SOURCE_URLS)
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(
    f"arXiv:{source}" for source in VERIFIED_SOURCE_URLS
)

ARC_HUMAN_REPLAY_NOTES = [
    "docs/research-notes/arc-human-baseline-and-replay-signal.md",
    "docs/research-notes/arc-human-replay-application-spec.md",
    "docs/research-notes/arc-frame-change-predictor-spec.md",
    "docs/research-notes/arc-world-model-trust-energy-spec.md",
    "docs/research-notes/arc-leaderboard-competitive-intel-2026-06-20.md",
]

DEFAULT_STRONGEST_FOR_V416 = (
    "flagged_for_v416: DQfD/PER-style human-replay expert-injection for the "
    "ARC frame-change predictor and value/energy heads, anchored by "
    "arXiv:1704.03732, arXiv:1511.05952, and arXiv:2206.11795"
)

HUMAN_CORPUS = {
    "source": "ARC-AGI-3 public-demo human replay corpus",
    "public_games": 25,
    "replay_count": 342,
    "example_count": 14672,
    "frame_changing_actions": 14243,
    "frame_change_rate": 14243 / 14672,
    "level_progress_positive_count": 132,
    "usage": (
        "bootstrap frame-change/clickability, behavior-prior, and value/energy "
        "heads from frame-derived features"
    ),
    "caveat": (
        "public games only; value transfers only through held-out variants or "
        "hidden-game generalization, never public replay memorization"
    ),
}

LEADERBOARD_DQN_MAPPING = {
    "source_note": "docs/research-notes/arc-leaderboard-competitive-intel-2026-06-20.md",
    "pattern": "prioritized replay plus expert-injection",
    "expert_priority_multiplier": 5,
    "dqn_stack_components": [
        "prioritized_experience_replay",
        "expert_imitation_demo_seed",
        "persistent_action_effect_memory",
        "attention_cnn_value_net",
    ],
    "carnot_mapping": (
        "seed public human replay transitions into replay/value batches with "
        "extra priority, then decay only after cached self-play transitions "
        "prove equal or higher progress"
    ),
}

DEFAULT_PRECONDITIONS_CHECKED = {
    "agents_md_read": True,
    "codex_md_read": True,
    "research_studying_filtered": True,
    "research_references_filtered": True,
    "arc_human_replay_notes_read": ARC_HUMAN_REPLAY_NOTES,
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

ARC_MAPPING = {
    "GAP-ARCH-FRAME-CHANGE-PREDICTOR": {
        "principle": (
            "Turn human frame/action/click demonstrations into a frame-only "
            "predictor for which candidate actions change the screen."
        ),
        "source_ids": ["2206.11795", "2407.15007", "2405.17476"],
        "next_experiment": (
            "Pretrain click heatmap and ACTION1-5 heads on the 14,672-example "
            "human corpus before mixing self-generated transitions."
        ),
    },
    "GAP-ARCH-VALUE-ENERGY-HEADS": {
        "principle": (
            "Use human progress trajectories as offline value/energy labels while "
            "avoiding out-of-dataset action optimism."
        ),
        "source_ids": ["2110.06169", "1905.11108", "2405.17476"],
        "next_experiment": (
            "Train value and contrastive energy heads from level_progress, "
            "steps-to-go, and human-vs-corrupted state/action pairs."
        ),
    },
    "GAP-ARCH-EXPERT-INJECTION-REPLAY": {
        "principle": (
            "Keep scarce expert demonstrations active in replay so sparse-reward "
            "training starts from useful behavior instead of no-op exploration."
        ),
        "source_ids": ["1704.03732", "1511.05952", "2302.02948"],
        "next_experiment": (
            "Seed human replay transitions at 5x priority in the DQN/value replay "
            "queue, then anneal priority only after self-play produces equal "
            "progress evidence."
        ),
    },
}

DEFAULT_METHODS = [
    {
        "name": "Deep Q-learning from Demonstrations",
        "arxiv_id": "1704.03732",
        "mapped_application": "GAP-ARCH-EXPERT-INJECTION-REPLAY",
        "stack_mapping": (
            "Use DQfD's mixture of TD learning, supervised expert-action loss, "
            "and prioritized replay as the direct template for seeding ARC human "
            "replays into the value/replay stack."
        ),
        "pitfall": (
            "DQfD is not a solve recipe; it improves sparse-reward exploration "
            "only if the demonstration distribution generalizes beyond public games."
        ),
    },
    {
        "name": "Prioritized Experience Replay",
        "arxiv_id": "1511.05952",
        "mapped_application": "GAP-ARCH-EXPERT-INJECTION-REPLAY",
        "stack_mapping": (
            "Prioritize rare high-progress and expert transitions rather than "
            "sampling human and self-play transitions uniformly."
        ),
        "pitfall": (
            "Priority can overfit public-game demos unless held-out variant "
            "transfer is the acceptance gate."
        ),
    },
    {
        "name": "Video PreTraining",
        "arxiv_id": "2206.11795",
        "mapped_application": "GAP-ARCH-FRAME-CHANGE-PREDICTOR",
        "stack_mapping": (
            "Treat frame/action replay as behavior pretraining: learn a visual "
            "action prior and clickability model before RL-style fine-tuning."
        ),
        "pitfall": (
            "ARC replays already have actions, but only 25 public games; the "
            "model must remain frame-only and transfer-tested."
        ),
    },
    {
        "name": "Offline Reinforcement Learning with Implicit Q-Learning",
        "arxiv_id": "2110.06169",
        "mapped_application": "GAP-ARCH-VALUE-ENERGY-HEADS",
        "stack_mapping": (
            "Fit value/energy heads from logged actions without querying unseen "
            "actions, then extract an advantage-weighted behavior policy."
        ),
        "pitfall": (
            "IQL is only appropriate after reward/progress labels are clean; "
            "frame_delta alone is not the same as task progress."
        ),
    },
    {
        "name": "SQIL imitation via sparse rewards",
        "arxiv_id": "1905.11108",
        "mapped_application": "GAP-ARCH-VALUE-ENERGY-HEADS",
        "stack_mapping": (
            "Use simple demonstration-match rewards as a first imitation-energy "
            "baseline before adding more brittle inverse-RL machinery."
        ),
        "pitfall": (
            "Matching public expert states can still be the wrong objective on "
            "novel games, so progress labels and variants remain required."
        ),
    },
    {
        "name": "Efficient Online RL with Offline Data",
        "arxiv_id": "2302.02948",
        "mapped_application": "GAP-ARCH-EXPERT-INJECTION-REPLAY",
        "stack_mapping": (
            "Keep human replay data in the replay buffer during online/self-play "
            "updates instead of treating it as one-off pretraining."
        ),
        "pitfall": (
            "Carnot's current task is offline/cached; any online step must remain "
            "competition-legal and separately gated."
        ),
    },
    {
        "name": "Is Behavior Cloning All You Need?",
        "arxiv_id": "2407.15007",
        "mapped_application": "GAP-ARCH-FRAME-CHANGE-PREDICTOR",
        "stack_mapping": (
            "Justifies a supervised behavior-cloning first pass for the click "
            "prior when payoffs are bounded and labels are clean."
        ),
        "pitfall": (
            "The theorem does not remove distribution-shift risk; hidden-game "
            "evaluation still requires variant transfer."
        ),
    },
    {
        "name": "How to Leverage Diverse Demonstrations in Offline Imitation Learning",
        "arxiv_id": "2405.17476",
        "mapped_application": "GAP-ARCH-VALUE-ENERGY-HEADS",
        "stack_mapping": (
            "Select non-expert or self-play actions by resultant-state progress "
            "toward human-like states, then weight behavior cloning accordingly."
        ),
        "pitfall": (
            "Resultant-state overlap can be misleading when public-game states "
            "do not cover hidden mechanics."
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
        "human_corpus": dict(HUMAN_CORPUS),
        "leaderboard_dqn_mapping": {
            key: list(value) if isinstance(value, list) else value
            for key, value in LEADERBOARD_DQN_MAPPING.items()
        },
        "arc_mapping": {
            gap_id: dict(details) for gap_id, details in ARC_MAPPING.items()
        },
        "strongest_for_v416": DEFAULT_STRONGEST_FOR_V416,
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
        row.get("arc_human_replay_notes_read") == ARC_HUMAN_REPLAY_NOTES,
        "preconditions_checked must record ARC human-replay note resources",
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
        "field_principles must match REQ-REPORT-4498",
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

    _require(
        artifact["human_corpus"] == HUMAN_CORPUS,
        "human_corpus must preserve the verified ARC public replay counts",
    )
    _require(
        artifact["leaderboard_dqn_mapping"] == LEADERBOARD_DQN_MAPPING,
        "leaderboard_dqn_mapping must preserve expert injection facts",
    )

    arc_mapping = artifact["arc_mapping"]
    _require(
        isinstance(arc_mapping, Mapping) and set(arc_mapping) == APPLICATION_KEYS,
        "arc_mapping must cover required ARC applications",
    )
    for gap_id, row in arc_mapping.items():
        _require(isinstance(row, Mapping), f"arc_mapping {gap_id} must be a mapping")
        _require(
            "principle" in row and "source_ids" in row and "next_experiment" in row,
            f"arc_mapping {gap_id} missing fields",
        )
        _require(
            isinstance(row["principle"], str) and row["principle"].strip(),
            f"arc_mapping {gap_id} needs a principle",
        )
        _require(
            isinstance(row["next_experiment"], str) and row["next_experiment"].strip(),
            f"arc_mapping {gap_id} needs a next_experiment",
        )
        _require(
            isinstance(row["source_ids"], list)
            and row["source_ids"]
            and set(row["source_ids"]).issubset(set(source_ids)),
            f"arc_mapping {gap_id} source_ids must be verified",
        )

    _require(
        artifact["strongest_for_v416"] == DEFAULT_STRONGEST_FOR_V416,
        "strongest_for_v416 must name the verified .416 package",
    )
    _validate_preconditions(artifact["preconditions_checked"])


def _artifact_json(artifact: Mapping[str, Any]) -> str:
    return json.dumps(artifact, indent=2, sort_keys=True)


def render_research_note(artifact: Mapping[str, Any]) -> str:
    """Render markdown with the artifact block first for automated parsing."""

    validate_artifact(artifact)
    source_line = ", ".join(f"arXiv:{source}" for source in DEFAULT_SOURCE_IDS)
    return f"""# ARC imitation/replay SOTA ingestion .415 - 2026-06-20

```json
{_artifact_json(artifact)}
```

Reliable channel only: `research-studying.md`, `research-references.md`,
`scripts/sweep_clusters.py`, ARC human-replay notes, arXiv abs-page HTTP-200
checks, and low-concurrency WebSearch/WebFetch of the top eight
imitation/behavior-cloning/offline-RL sources. `.venv/bin/python
scripts/sweep_clusters.py --help` succeeded. `scripts/sweep_clusters.py 0
--max-results 8` and `scripts/sweep_clusters.py 3 --max-results 8` emitted
focused cluster URLs used as the repo-standard fresh-pass substrate. No
`/deep-research` call was made. No training, live LLM inference, leaderboard
submission, or live solve was launched. No ops/status/traceability files were
modified.

Sources checked: {source_line}.

## Local Replay Signal

The active human-replay direction starts from the 14,672-example ARC public
human corpus across 25 public games. The useful labels are frame_delta,
level_progress, action_id, and click location. Humans changed the frame on
14,243 actions and produced 132 level-progress positives, which makes the data
well aligned with frame-change/clickability and sparse progress heads. The
honest caveat is that the corpus is public-game-only; it is training signal for
generalization, not a hidden-eval solve.

## Literature Mapping

- DQfD, arXiv:1704.03732, is the strongest expert-injection template: combine
  TD updates, supervised demonstrator-action loss, and prioritized replay.
- Prioritized Experience Replay, arXiv:1511.05952, supplies the queue policy:
  replay important transitions more often instead of uniform sampling.
- VPT, arXiv:2206.11795, supplies the visual behavior-pretraining analogy for
  frame/action demonstrations and later fine-tuning.
- IQL, arXiv:2110.06169, supplies the offline RL guardrail for value heads:
  avoid evaluating out-of-dataset actions while improving over behavior data.
- SQIL, arXiv:1905.11108, supplies a simple sparse-reward imitation baseline
  before more complex inverse-RL machinery.
- RLPD, arXiv:2302.02948, supports keeping offline human data in the replay
  buffer during later off-policy updates.
- Behavior Cloning Horizon, arXiv:2407.15007, makes the supervised behavior
  cloning first pass defensible when payoffs are bounded and labels are clean.
- Diverse Demonstrations IL, arXiv:2405.17476, maps self-play transitions by
  resultant-state progress toward expert-state manifolds.

## SOTA->Experiment Mapping

- GAP-ARCH-FRAME-CHANGE-PREDICTOR: pretrain the click heatmap and ACTION1-5
  heads from human frame/action/click labels before mixing cached self-play.
- GAP-ARCH-VALUE-ENERGY-HEADS: use `level_progress`, steps-to-go, and
  human-vs-corrupted state/action pairs to bootstrap value and contrastive
  energy heads.
- GAP-ARCH-EXPERT-INJECTION-REPLAY: seed the leaderboard-DQN-style replay queue
  with human demonstrations at 5x priority, then anneal only after self-play
  produces equal progress evidence.

For `.416`, the strongest hand-off is the expert-injection replay package:
DQfD/PER-style human replay injection for the frame-change predictor and
value/energy heads, with VPT-style frame-action behavior pretraining as the
supervised front end. Count this as a planning artifact only:
`inference_substrate=aggregation_from_upstream_artifacts`.

{DEFAULT_STRONGEST_FOR_V416}
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
    """Check citations, required mapping language, and the embedded artifact."""

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
        "No ops/status/traceability",
        "behavior cloning",
        "offline RL",
        "prioritized replay",
        "expert-injection",
        "14,672-example",
        "GAP-ARCH-FRAME-CHANGE-PREDICTOR",
        "GAP-ARCH-VALUE-ENERGY-HEADS",
        "GAP-ARCH-EXPERT-INJECTION-REPLAY",
        "flagged_for_v416",
        "aggregation_from_upstream_artifacts",
    ]
    for phrase in required_phrases:
        _require(phrase in note, f"research note missing required phrase: {phrase}")
    artifact_from_note(note)


RESEARCH_NOTE = render_research_note(build_artifact())


def write_outputs(
    *,
    artifact_path: Path | str | None = None,
    note_path: Path | str | None = None,
) -> dict[str, object]:
    """Write the stable JSON artifact and research note."""

    artifact = build_artifact()
    note = render_research_note(artifact)
    validate_research_note(note)

    artifact_target = Path(artifact_path or "results/experiment_4498_arc_imitation_sota_415.json")
    note_target = Path(note_path or RESEARCH_NOTE_RELATIVE_PATH)
    artifact_target.parent.mkdir(parents=True, exist_ok=True)
    note_target.parent.mkdir(parents=True, exist_ok=True)
    artifact_target.write_text(_artifact_json(artifact) + "\n", encoding="utf-8")
    note_target.write_text(note, encoding="utf-8")
    return artifact


def main() -> int:
    root = Path(os.environ.get("CARNOT_EXP4498_ROOT", "."))
    artifact = write_outputs(
        artifact_path=root / "results/experiment_4498_arc_imitation_sota_415.json",
        note_path=root / RESEARCH_NOTE_RELATIVE_PATH,
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
