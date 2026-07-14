"""Experiment 5619: ARC forward/inverse transition-cycle verifier artifact.

Spec refs: REQ-ARC-WMTE-5619,
SCENARIO-ARC-WMTE-5619-CYCLE-ADMISSION,
SCENARIO-ARC-WMTE-5619-CORRUPTION-REJECTION.

The experiment uses already captured public-game transition traces. It does not read game source,
does not call a per-game adapter, does not run exhaustive BFS, and does not claim a new level.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from carnot.agentic.arc_transition_cycle_verifier import (
    ObservedTransition,
    TransitionCycleDecision,
    TransitionCycleVerifier,
    make_corrupted_transition,
)


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_5619_arc_forward_inverse_transition_cycle"
EXPERIMENT_ID = 5619
RESULT_RELATIVE_PATH = f"results/{EXPERIMENT}.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
INFERENCE_SUBSTRATE = "offline_arcade_live_agent_runtime_filters_no_new_llm"
SOLVE_PROVENANCE = "development_proxy"
RANDOM_SEEDS = [5619]
DEFAULT_ROSTER = ("dc22", "bp35", "s5i5")
CORRUPTION_CONDITIONS = (
    "permuted_action",
    "mismatched_successor",
    "noop_substitution",
    "wrong_object_change",
)

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "registry_precheck",
    "trace_roster",
    "source_files_read",
    "per_game_adapter_used",
    "transition_feature_contract",
    "heldout_transitions_by_condition",
    "coverage",
    "abstention_rate",
    "inverse_action_accuracy",
    "forward_replay_error",
    "valid_transition_accept_rate",
    "valid_transition_accept_count",
    "corruption_reject_rate",
    "unsafe_transition_accept_count",
    "cycle_verifier_positive_control_rate",
    "per_game_heterogeneity",
    "runtime_overhead",
    "immutable_update_receipts",
    "solve_provenance",
    "inference_substrate",
    "random_seeds",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "field_principles": {
        "principle": "principle annotations are carried in the artifact so every required 5619 field is auditable.",
    },
    "registry_precheck": {
        "principle": "known public reproduced levels only; no duplicate solve or new-level credit is implied.",
    },
    "trace_roster": {
        "principle": "every measured transition has auditable game/session/episode provenance from agent-owned observations.",
    },
    "source_files_read": {
        "principle": "false excludes outer-loop source reverse engineering from the measurement.",
    },
    "per_game_adapter_used": {
        "principle": "false proves the verifier is generic and not a hand GameAdapter.",
    },
    "transition_feature_contract": {
        "principle": "features are derived from before/action/after frames, not source code, adapters, or hidden oracle flags.",
    },
    "heldout_transitions_by_condition": {
        "principle": "valid and adversarial sample sizes are explicit before interpreting rates.",
    },
    "coverage": {
        "principle": "the verifier must expose how much of heldout data received a non-abstaining decision.",
    },
    "abstention_rate": {
        "principle": "an over-abstaining verifier is safe but not useful.",
    },
    "inverse_action_accuracy": {
        "principle": "the inverse reachability factor is measured separately from successor plausibility.",
    },
    "forward_replay_error": {
        "principle": "effect consistency is measured directly instead of inferred from acceptance.",
    },
    "valid_transition_accept_rate": {
        "principle": "coverage is visible; an all-abstain verifier is not useful.",
    },
    "valid_transition_accept_count": {
        "principle": "the numerator behind valid-transition coverage is auditable.",
    },
    "corruption_reject_rate": {
        "principle": "negative controls prove bad updates are rejected.",
    },
    "unsafe_transition_accept_count": {
        "principle": "fail-closed admission prevents corrupted updates from mutating the world model.",
    },
    "cycle_verifier_positive_control_rate": {
        "principle": "the downstream gate exposes a numeric positive-control pass rate.",
    },
    "per_game_heterogeneity": {
        "principle": "within-game fitting is reported per game and no cross-game transfer is claimed.",
    },
    "runtime_overhead": {
        "principle": "live-path cost is bounded and not hidden inside aggregate wall time.",
    },
    "immutable_update_receipts": {
        "principle": "only admitted valid transitions create durable update receipts.",
    },
    "solve_provenance": {
        "principle": "development_proxy -- this task receives no new ARC level credit.",
    },
    "inference_substrate": {
        "principle": "offline_arcade_live_agent_runtime_filters_no_new_llm -- no new LLM was invoked.",
    },
    "random_seeds": {
        "principle": "determinism is the precondition for replaying the evaluation.",
    },
    "reproducibility_checksum": {
        "principle": "content-addressed replay inputs catch silent corpus or threshold drift.",
    },
    "honest_verdict": {
        "principle": "terminal prefix records whether the verifier was useful, over-abstaining, or unsafe.",
    },
}


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=True)


def _sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def read_yaml(path: Path) -> JsonDict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _registry_rows(registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = registry.get("games", {})
    if isinstance(rows, Mapping):
        return {str(game): row for game, row in rows.items() if isinstance(row, Mapping)}
    return {
        str(row.get("game")): row for row in rows if isinstance(row, Mapping) and row.get("game")
    }


def _is_reproduced(row: Mapping[str, Any] | None) -> bool:
    if not row:
        return False
    if str(row.get("reproducibility", "")).lower() == "reproduced":
        return True
    return bool(row.get("offline_reproduced")) or int(row.get("levels_reproduced") or 0) > 0


def registry_precheck(
    roster: Sequence[str],
    registry: Mapping[str, Any],
    transitions_by_game: Mapping[str, Sequence[ObservedTransition]],
) -> JsonDict:
    rows = _registry_rows(registry)
    roster_rows = []
    selected = []
    for game in roster:
        reg = rows.get(str(game))
        transition_count = len(transitions_by_game.get(str(game), ()))
        reproduced = _is_reproduced(reg)
        usable = reproduced and transition_count > 0
        if usable:
            selected.append(str(game))
        roster_rows.append(
            {
                "game": str(game),
                "registry_reproduced": bool(reproduced),
                "levels_reproduced": int((reg or {}).get("levels_reproduced") or 0),
                "agent_owned_transition_count": int(transition_count),
                "selected": bool(usable),
            }
        )
    return {
        "ok": len(selected) >= 3,
        "only_known_levels_used": all(
            row["registry_reproduced"] for row in roster_rows if row["selected"]
        ),
        "source_files_read": False,
        "per_game_adapter_used": False,
        "selected_games": selected,
        "roster_rows": roster_rows,
        "registry_total_levels": int(registry.get("reproducible_total_levels") or 0),
    }


def load_agent_owned_transitions(
    roster: Sequence[str] = DEFAULT_ROSTER,
    *,
    root: Path | None = None,
    max_per_game: int = 144,
) -> dict[str, list[ObservedTransition]]:
    from carnot.agentic.arc_transition_capture import TransitionCorpus

    corpus = TransitionCorpus(root=root)
    out: dict[str, list[ObservedTransition]] = {}
    for game in roster:
        rows: list[ObservedTransition] = []
        for index, transition in enumerate(corpus.load(str(game))):
            state = np.asarray(transition.grid, dtype=np.int16)
            successor = np.asarray(transition.next_grid, dtype=np.int16)
            if state.shape != successor.shape or not np.any(state != successor):
                continue
            rows.append(
                ObservedTransition(
                    game=str(game),
                    episode=f"{game}-episode-{index // 16}",
                    step=index,
                    state=state,
                    action=int(transition.action),
                    data=transition.data,
                    successor=successor,
                )
            )
            if len(rows) >= max_per_game:
                break
        out[str(game)] = rows
    return out


def _split_calibration_heldout(
    rows: Sequence[ObservedTransition],
    *,
    random_seed: int,
    heldout_count: int,
) -> tuple[list[ObservedTransition], list[ObservedTransition]]:
    episodes: dict[str, list[ObservedTransition]] = {}
    for row in rows:
        episodes.setdefault(row.episode, []).append(row)
    episode_ids = sorted(episodes)
    rng = random.Random(random_seed)
    rng.shuffle(episode_ids)
    heldout: list[ObservedTransition] = []
    calibration: list[ObservedTransition] = []
    for episode in episode_ids:
        target = heldout if len(heldout) < heldout_count else calibration
        target.extend(episodes[episode])
    if not calibration and heldout:
        split = max(1, len(heldout) // 2)
        calibration, heldout = heldout[:split], heldout[split:]
    return calibration, heldout[:heldout_count]


def _condition_rows(
    valid_rows: Sequence[ObservedTransition],
) -> dict[str, list[ObservedTransition]]:
    rows = {"valid": list(valid_rows)}
    for condition in CORRUPTION_CONDITIONS:
        corrupted: list[ObservedTransition] = []
        for index, row in enumerate(valid_rows):
            other = valid_rows[(index + 1) % len(valid_rows)]
            corrupted.append(
                make_corrupted_transition(
                    row,
                    condition,
                    replacement_action=(1 if int(row.action) != 1 else 6),
                    replacement_successor=other.successor,
                )
            )
        rows[condition] = corrupted
    return rows


def _rate(numerator: int, denominator: int) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def _mean(values: Sequence[float]) -> float:
    return round(float(sum(values)) / float(len(values)), 6) if values else 0.0


def _summarize_decisions(
    decisions_by_condition: Mapping[str, Sequence[TransitionCycleDecision]],
) -> JsonDict:
    valid = list(decisions_by_condition.get("valid", ()))
    corrupt = [
        decision
        for condition, decisions in decisions_by_condition.items()
        if condition != "valid"
        for decision in decisions
    ]
    valid_accepts = [decision for decision in valid if decision.admitted]
    valid_non_abstain = [decision for decision in valid if not decision.abstained]
    corrupt_rejected = [decision for decision in corrupt if not decision.admitted]
    inverse_correct = [decision for decision in valid if decision.inverse_action_matches]
    forward_errors = [
        float(decision.forward_replay_error)
        for decision in valid_accepts
        if decision.forward_replay_error is not None
    ]
    return {
        "coverage": _rate(len(valid_non_abstain), len(valid)),
        "abstention_rate": _rate(
            sum(1 for decision in valid if decision.abstained),
            len(valid),
        ),
        "inverse_action_accuracy": _rate(len(inverse_correct), len(valid)),
        "forward_replay_error": _mean(forward_errors),
        "valid_transition_accept_rate": _rate(len(valid_accepts), len(valid)),
        "valid_transition_accept_count": int(len(valid_accepts)),
        "corruption_reject_rate": _rate(len(corrupt_rejected), len(corrupt)),
        "unsafe_transition_accept_count": int(sum(1 for decision in corrupt if decision.admitted)),
        "cycle_verifier_positive_control_rate": _rate(len(valid_accepts), len(valid)),
        "immutable_update_receipts": [
            dict(decision.update_receipt) for decision in valid_accepts if decision.update_receipt
        ],
    }


def build_artifact(
    *,
    transitions_by_game: Mapping[str, Sequence[ObservedTransition]] | None = None,
    registry: Mapping[str, Any] | None = None,
    roster: Sequence[str] = DEFAULT_ROSTER,
    random_seed: int = 5619,
    heldout_per_game: int = 36,
    root: Path = REPO_ROOT,
) -> JsonDict:
    start = time.perf_counter()
    registry_data = dict(registry or read_yaml(root / REGISTRY_RELATIVE_PATH))
    traces = (
        {game: list(rows) for game, rows in transitions_by_game.items()}
        if transitions_by_game is not None
        else load_agent_owned_transitions(roster)
    )
    roster = tuple(str(game) for game in roster if str(game) in traces)
    if not roster and transitions_by_game is not None:
        roster = tuple(sorted(str(game) for game in traces))
    precheck = registry_precheck(roster, registry_data, traces)
    selected_games = list(precheck["selected_games"])
    per_game: dict[str, JsonDict] = {}
    decisions_by_condition: dict[str, list[TransitionCycleDecision]] = {
        "valid": [],
        **{condition: [] for condition in CORRUPTION_CONDITIONS},
    }

    for game_index, game in enumerate(selected_games):
        calibration, heldout = _split_calibration_heldout(
            traces[game],
            random_seed=random_seed + game_index,
            heldout_count=heldout_per_game,
        )
        verifier = TransitionCycleVerifier(min_support=2).fit(calibration)
        condition_rows = _condition_rows(heldout)
        game_decisions: dict[str, list[TransitionCycleDecision]] = {}
        for condition, rows in condition_rows.items():
            game_decisions[condition] = [verifier.evaluate(row) for row in rows]
            decisions_by_condition[condition].extend(game_decisions[condition])
        summary = _summarize_decisions(game_decisions)
        per_game[game] = {
            "calibration_transitions": int(len(calibration)),
            "heldout_transitions": int(len(heldout)),
            "valid_transition_accept_rate": summary["valid_transition_accept_rate"],
            "corruption_reject_rate": summary["corruption_reject_rate"],
            "abstention_rate": summary["abstention_rate"],
        }

    summary = _summarize_decisions(decisions_by_condition)
    heldout_counts = {
        condition: int(len(decisions)) for condition, decisions in decisions_by_condition.items()
    }
    duration_s = round(time.perf_counter() - start, 6)
    total_evaluated = sum(heldout_counts.values())
    overhead = {
        "duration_s": duration_s,
        "evaluated_transitions": int(total_evaluated),
        "seconds_per_transition": round(duration_s / total_evaluated, 8)
        if total_evaluated
        else None,
    }
    unsafe = int(summary["unsafe_transition_accept_count"])
    over_abstain = (
        float(summary["valid_transition_accept_rate"]) < 0.1
        or float(summary["abstention_rate"]) > 0.5
    )
    verdict = (
        "complete: transition_cycle_verifier_rejected_corruptions_and_admitted_valid_updates"
        if unsafe == 0 and not over_abstain
        else "complete: transition_cycle_verifier_safe_over_abstaining_not_useful_terminal"
    )
    checksum_payload = {
        "experiment": EXPERIMENT,
        "roster": selected_games,
        "heldout_transitions_by_condition": heldout_counts,
        "per_game_heterogeneity": per_game,
        "random_seed": random_seed,
        "summary": {
            key: value for key, value in summary.items() if key != "immutable_update_receipts"
        },
    }
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": "carnot.exp5619.arc_forward_inverse_transition_cycle.v1",
        "date": "20260714",
        "field_principles": FIELD_PRINCIPLES,
        "registry_precheck": precheck,
        "trace_roster": [
            {
                "game": game,
                "provenance": "data/arc_transition_corpus agent-owned runtime observations",
                "calibration_transitions": per_game[game]["calibration_transitions"],
                "heldout_transitions": per_game[game]["heldout_transitions"],
            }
            for game in selected_games
        ],
        "source_files_read": False,
        "per_game_adapter_used": False,
        "transition_feature_contract": {
            "inputs": ["state", "action", "successor"],
            "features": [
                "changed_cell_count",
                "delta_color_pairs",
                "changed_bbox_shape",
                "click_relative_offsets",
            ],
            "fit_scope": "within_game_live_session_only",
            "cross_game_transfer_claim": False,
        },
        "heldout_transitions_by_condition": heldout_counts,
        "per_game_heterogeneity": per_game,
        "runtime_overhead": overhead,
        "solve_provenance": SOLVE_PROVENANCE,
        "source_files_read_note": "No environment_files game source was read by this builder.",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_note": "Offline public-transition verifier scoring only; no GGUF/LLM invocation.",
        "random_seeds": [int(random_seed)],
        "reproducibility_checksum": _sha256(checksum_payload),
        "honest_verdict": verdict,
        **summary,
    }
    return artifact


def main() -> None:
    artifact = build_artifact()
    write_json(REPO_ROOT / RESULT_RELATIVE_PATH, artifact)


if __name__ == "__main__":
    main()
